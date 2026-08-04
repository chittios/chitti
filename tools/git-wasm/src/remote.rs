//! Smart-HTTP remotes: `clone` (upload-pack) and `push` (receive-pack).
//!
//! The git smart-HTTP protocol over the kernel's `chitti::host_http` import:
//! `GET $URL/info/refs?service=…` for the advertised refs, then a `POST` to
//! `…/git-upload-pack` / `…/git-receive-pack`. Clone parses the returned
//! **packfile** (commits/trees/blobs + OFS/REF deltas) into loose objects and
//! checks out HEAD; push builds a packfile from our local objects and sends it.

use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use crate::{
    fs_write, from_hex, http, to_hex,
};
use crate::git::{current_branch, git_dir, head_commit, read_loose, write_loose, write_ref, parse_tree, obj_raw};
use crate::{zlib_decompress, zlib_deflate, fs_read};
use crate::git_root;

/// Capabilities we request from upload-pack (must be a subset of what the
/// server advertises — requesting `multi_ack_detailed`/`report-status` from a
/// server that only offers `multi_ack` makes it reject the whole request).
const CAPS: &str = "multi_ack thin-pack ofs-delta no-progress";

/// pkt-line: `4-hex len + payload`; `0000` = flush.
fn parse_pkt_lines(data: &[u8]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let mut p = 0usize;
    while p + 4 <= data.len() {
        let Ok(len) = u16::from_str_radix(core::str::from_utf8(&data[p..p + 4]).unwrap_or("0000"), 16) else {
            break;
        };
        let len = len as usize;
        if len == 0 {
            p += 4; // flush
            continue;
        }
        let end = (p + len).min(data.len());
        out.push(data[p + 4..end].to_vec());
        p = end;
    }
    out
}

/// Pack object header varint: first byte = type (bits 4-6), size's low nibble
/// (bits 0-3), bit 7 = "more size follows"; continuation bytes add 7 size bits.
fn read_varint(data: &[u8], p: &mut usize) -> Option<(u8, u64)> {
    let b0 = *data.get(*p)?;
    *p += 1;
    let typ = (b0 >> 4) & 0x7;
    let mut size = (b0 & 0x0f) as u64;
    let mut shift = 4;
    let mut b = b0;
    while b & 0x80 != 0 {
        b = *data.get(*p)?;
        *p += 1;
        size |= ((b & 0x7f) as u64) << shift;
        shift += 7;
    }
    Some((typ, size))
}

/// OFS_DELTA's negative offset (from the end of the header back).
fn read_ofs_offset(data: &[u8], p: &mut usize) -> Option<u64> {
    let mut c = *data.get(*p)?;
    *p += 1;
    let mut off = (c & 0x7f) as u64;
    while c & 0x80 != 0 {
        c = *data.get(*p)?;
        *p += 1;
        off = ((off + 1) << 7) | (c & 0x7f) as u64;
    }
    Some(off)
}

fn apply_delta(base: &[u8], delta: &[u8]) -> Option<Vec<u8>> {
    let mut p = 0usize;
    let _ = read_varint(delta, &mut p)?;
    let _ = read_varint(delta, &mut p)?;
    let mut out = Vec::new();
    while p < delta.len() {
        let op = delta[p];
        p += 1;
        if op & 0x80 != 0 {
            let mut offset = 0u64;
            let mut size = 0u64;
            for i in 0..4 {
                if op & (1 << i) != 0 {
                    offset |= (*delta.get(p)? as u64) << (8 * i);
                    p += 1;
                }
            }
            for i in 0..3 {
                if op & (1 << (4 + i)) != 0 {
                    size |= (*delta.get(p)? as u64) << (8 * i);
                    p += 1;
                }
            }
            if size == 0 {
                size = 0x10000;
            }
            let (off, sz) = (offset as usize, size as usize);
            if off + sz > base.len() {
                return None;
            }
            out.extend_from_slice(&base[off..off + sz]);
        } else if op != 0 {
            let n = op as usize;
            let end = p + n;
            if end > delta.len() {
                return None;
            }
            out.extend_from_slice(&delta[p..end]);
            p = end;
        } else {
            return None;
        }
    }
    Some(out)
}

fn kind_name(t: u8) -> &'static str {
    match t {
        1 => "commit",
        2 => "tree",
        3 => "blob",
        4 => "tag",
        _ => "unknown",
    }
}

/// Inflate the pack object whose zlib body starts at `pack[start..]`, expected
/// to decompress to `want_size` bytes. The kernel's decoder rejects trailing
/// bytes (adler check), so the exact window is the first whose decompressed
/// length matches the declared size — no false boundaries.
fn inflate_object(pack: &[u8], start: usize, want_size: u64) -> Option<(Vec<u8>, usize)> {
    let mut end = (start + 6).min(pack.len());
    while end <= pack.len() {
        if let Some((dec, consumed)) = zlib_decompress(&pack[start..end]) {
            if consumed == end - start && dec.len() as u64 == want_size {
                return Some((dec, end - start));
            }
        }
        end += 1;
    }
    None
}

/// Write a checkout-tree recursively (overwriting the working tree).
fn checkout_tree(tree_sha: &str) {
    let Some((_, t)) = read_loose(tree_sha) else { return };
    let Some(ents) = parse_tree(&t) else { return };
    let root = git_root();
    for (mode, name, sha) in ents {
        if mode == "40000" {
            checkout_tree(&sha);
        } else if let Some((_, blob)) = read_loose(&sha) {
            fs_write(&alloc::format!("{root}/{name}"), &blob);
        }
    }
}

/// Flatten a tree into `(sha, path)` blob leaves (full relative paths).
fn collect_blobs(tree_sha: &str) -> Vec<(String, String)> {
    fn rec(tree_sha: &str, prefix: &str, out: &mut Vec<(String, String)>) {
        let Some((_, t)) = read_loose(tree_sha) else { return };
        let Some(ents) = crate::git::parse_tree(&t) else { return };
        for (mode, name, sha) in ents {
            let full = if prefix.is_empty() { name.clone() } else { alloc::format!("{prefix}/{name}") };
            if mode == "40000" {
                rec(&sha, &full, out);
            } else {
                out.push((sha, full));
            }
        }
    }
    let mut out = Vec::new();
    rec(tree_sha, "", &mut out);
    out
}

// --- clone ----------------------------------------------------------------------

pub fn clone(args: &str) -> String {
    let toks: Vec<&str> = args.split_whitespace().collect();
    let Some(url) = toks.first().map(|s| s.to_string()) else {
        return "usage: /git clone <url> [dir]".to_string();
    };
    if url.is_empty() {
        return "usage: /git clone <url> [dir]".to_string();
    }
    let base = url.trim_end_matches('/').to_string();
    // Clone target: the named dir (any folder), else a folder named after the
    // repo basename **in the shell's current directory** — the git-CLI shape.
    let target = toks
        .get(1)
        .map(|d| crate::normalize_path(d))
        .filter(|d| !d.is_empty() && d != "/")
        .unwrap_or_else(|| {
            let name = base.rsplit('/').next().unwrap_or("repo").trim_end_matches(".git");
            alloc::format!("{}/{name}", crate::base_dir())
        });
    // Point the working directory at the target before any local writes, so
    // the object store/refs/checkout all land inside the cloned folder.
    crate::set_cwd(&target);

    // 1. Advertised refs.
    let adv = match http(
        "GET",
        &alloc::format!("{base}/info/refs?service=git-upload-pack"),
        &[("Accept", "application/x-git-upload-pack-advertisement")],
        b"",
    ) {
        Ok((200, b)) => b,
        Ok((s, _)) => return alloc::format!("error: info/refs status {s}"),
        Err(_) => return "error: info/refs request failed".to_string(),
    };
    let mut refs: Vec<(String, String)> = Vec::new();
    for line in parse_pkt_lines(&adv) {
        let s = String::from_utf8_lossy(&line).to_string();
        if s.is_empty() || s.starts_with('#') || s.len() < 41 {
            continue;
        }
        let sha = s[..40].to_string();
        let name = s[40..].split('\0').next().unwrap_or("").trim().to_string();
        if sha.bytes().all(|b| b.is_ascii_hexdigit()) && !name.is_empty() {
            refs.push((sha, name));
        }
    }
    if refs.is_empty() {
        return "error: no refs advertised (empty or private repo?)".to_string();
    }
    let (head_sha, head_name) = refs
        .iter()
        .find(|(_, n)| n == "HEAD")
        .or_else(|| refs.iter().find(|(_, n)| n == "refs/heads/master" || n == "refs/heads/main"))
        .or_else(|| refs.iter().find(|(_, n)| n.starts_with("refs/heads/")))
        .cloned()
        .unwrap_or_else(|| refs[0].clone());

    // 2. Fetch the objects for HEAD. The pkt-line length counts the 4 hex chars
    // + the payload including its trailing LF.
    let want = alloc::format!(
        "{:04x}want {head_sha}\0{CAPS}\n",
        4 + 5 + 40 + 1 + CAPS.len() + 1
    );
    let req = alloc::format!("{want}00000009done\n");
    let pack = match http(
        "POST",
        &alloc::format!("{base}/git-upload-pack"),
        &[
            ("Content-Type", "application/x-git-upload-pack-request"),
            ("Accept", "application/x-git-upload-pack-result"),
        ],
        req.as_bytes(),
    ) {
        Ok((200, b)) => b,
        Ok((s, _)) => return alloc::format!("error: upload-pack status {s}"),
        Err(_) => return "error: upload-pack request failed".to_string(),
    };
    // The packfile starts at the `PACK` magic (pkt-line ACK/NAK precedes it).
    let Some(pi) = pack.windows(4).position(|w| w == b"PACK") else {
        return "error: no packfile in the upload-pack response".to_string();
    };

    // 3. Parse the packfile. `parsed` holds (stream pos, kind, content); deltas
    // resolve against earlier objects (OFS) or by sha (REF / loose).
    let mut parsed: Vec<(usize, String, Vec<u8>)> = Vec::new();
    let mut p = pi;
    if &pack[p..p + 4] != b"PACK" {
        return "error: bad PACK header".to_string();
    }
    p += 4;
    let _version = u32::from_be_bytes(pack[p..p + 4].try_into().unwrap_or([0; 4]));
    p += 4;
    let count = u32::from_be_bytes(pack[p..p + 4].try_into().unwrap_or([0; 4])) as usize;
    p += 4;
    for _ in 0..count {
        let obj_start = p;
        let (typ, want_size) = match read_varint(&pack, &mut p) {
            Some(v) => v,
            None => return "error: bad object header".to_string(),
        };
        let (dec, stream_len) = match inflate_object(&pack, p, want_size) {
            Some(v) => v,
            None => return "error: object inflate failed".to_string(),
        };
        p += stream_len;
        match typ {
            6 | 7 => {
                // The payload starts with the base reference, then the delta.
                let (base, delta_start): (Option<(String, Vec<u8>)>, usize) = if typ == 7 {
                    // REF_DELTA: 20-byte base sha.
                    let base_sha = to_hex(&dec[..20.min(dec.len())]);
                    let base = parsed
                        .iter()
                        .rev()
                        .find(|(_, k, c)| {
                            crate::sha1(&obj_raw(k, c)) == base_sha
                        })
                        .map(|(_, k, c)| (k.clone(), c.clone()))
                        .or_else(|| read_loose(&base_sha));
                    (base, 20)
                } else {
                    // OFS_DELTA: negative offset back to the base object's stream position.
                    let mut op = 0usize;
                    let off = match read_ofs_offset(&dec, &mut op) {
                        Some(o) => o,
                        None => return "error: bad ofs-delta".to_string(),
                    };
                    let base_pos = match obj_start.checked_sub(off as usize) {
                        Some(b) => b,
                        None => return "error: bad ofs-delta offset".to_string(),
                    };
                    let base = parsed
                        .iter()
                        .rev()
                        .find(|(pos, _, _)| *pos == base_pos)
                        .map(|(_, k, c)| (k.clone(), c.clone()));
                    (base, op)
                };
                let (bk, bc) = match base {
                    Some(b) => b,
                    None => return "error: delta base not found".to_string(),
                };
                let full = match apply_delta(&bc, &dec[delta_start.min(dec.len())..]) {
                    Some(f) => f,
                    None => return "error: delta apply failed".to_string(),
                };
                parsed.push((obj_start, bk, full));
            }
            _ => {
                parsed.push((obj_start, kind_name(typ).to_string(), dec));
            }
        }
    }

    // 4. Write loose objects, refs, checkout HEAD.
    let mut n_obj = 0usize;
    for (_, kind, content) in &parsed {
        if kind != "unknown" {
            write_loose(kind, content);
            n_obj += 1;
        }
    }
    for (sha, name) in &refs {
        if let Some(branch) = name.strip_prefix("refs/heads/") {
            write_ref(branch, sha);
        }
    }
    let Some((_, commit)) = read_loose(&head_sha) else {
        return "error: HEAD commit missing after clone".to_string();
    };
    let tree_sha = String::from_utf8_lossy(&commit)
        .lines()
        .find_map(|l| l.strip_prefix("tree ").map(|t| t.to_string()));
    if let Some(t) = tree_sha.clone() {
        checkout_tree(&t);
        // Stage the checked-out HEAD tree so `git status` is clean after clone.
        let mut index: Vec<(String, String, String)> = Vec::new();
        for (sha, path) in collect_blobs(&t) {
            index.push((String::from("100644"), sha, path));
        }
        crate::git::write_index(&index);
    }
    // HEAD → the branch the server's HEAD points at (resolve by sha, so a
    // clone always lands on a branch whose objects it actually fetched).
    let head_ref = if head_name.starts_with("refs/heads/") {
        head_name.clone()
    } else {
        refs.iter()
            .find(|(sha, name)| name.starts_with("refs/heads/") && sha == &head_sha)
            .map(|(_, n)| n.clone())
            .unwrap_or_else(|| "refs/heads/master".to_string())
    };
    fs_write(&alloc::format!("{}/HEAD", git_dir()), alloc::format!("ref: {head_ref}\n").as_bytes());
    // Record `origin` so `/git push` works without a URL.
    fs_write(
        &alloc::format!("{}/config", git_dir()),
        alloc::format!("[remote \"origin\"]\n\turl = {url}\n\tfetch = +refs/heads/*:refs/remotes/origin/*\n").as_bytes(),
    );
    alloc::format!(
        "ok: cloned {head_name} ({}) into {target} — {} objects, {} ref(s)",
        &head_sha[..8.min(head_sha.len())],
        n_obj,
        refs.len()
    )
}

// --- push ------------------------------------------------------------------------

/// The `origin` remote's URL from `.git/config` (so `/git push` works without
/// a URL).
pub(crate) fn origin_url() -> Option<String> {
    let config = fs_read(&alloc::format!("{}/config", git_dir()))?;
    for line in String::from_utf8_lossy(&config).lines() {
        let t = line.trim();
        if let Some(v) = t.strip_prefix("url = ") {
            return Some(v.to_string());
        }
    }
    None
}

pub fn push(args: &str) -> String {
    // `git push [url]` — a bare `/git push` uses the `origin` remote.
    let url = args
        .split_whitespace()
        .next()
        .map(|s| s.to_string())
        .filter(|s| !s.is_empty())
        .or_else(origin_url);
    let Some(url) = url else {
        return "error: no remote (clone/init with a URL first, or `git push <url>`)".to_string();
    };
    let base = url.trim_end_matches('/').to_string();
    let Some(head) = head_commit() else {
        return "error: nothing to push (no commits)".to_string();
    };
    let branch = current_branch();

    // 1. Advertised receive-pack refs.
    let adv = match http(
        "GET",
        &alloc::format!("{base}/info/refs?service=git-receive-pack"),
        &[("Accept", "application/x-git-receive-pack-advertisement")],
        b"",
    ) {
        Ok((200, b)) => b,
        Ok((s, _)) => return alloc::format!("error: info/refs status {s}"),
        Err(_) => return "error: info/refs request failed".to_string(),
    };
    let mut remote_sha: Option<String> = None;
    for line in parse_pkt_lines(&adv) {
        let s = String::from_utf8_lossy(&line).to_string();
        if s.len() >= 40 {
            let name = s[40..].split('\0').next().unwrap_or("").trim().to_string();
            if name == alloc::format!("refs/heads/{branch}") {
                remote_sha = Some(s[..40].to_string());
            }
        }
    }
    let old = remote_sha.unwrap_or_else(|| "0".repeat(40));

    // 2. Collect objects reachable from HEAD but not `old`.
    let mut need: Vec<String> = Vec::new();
    let mut seen = alloc::collections::BTreeSet::new();
    let mut stack: Vec<String> = vec![head.clone()];
    while let Some(sha) = stack.pop() {
        if seen.contains(&sha) || sha == old {
            continue;
        }
        seen.insert(sha.clone());
        need.push(sha.clone());
        let Some((kind, content)) = read_loose(&sha) else { continue };
        if kind == "commit" {
            let text = String::from_utf8_lossy(&content).to_string();
            if let Some(t) = text.lines().find_map(|l| l.strip_prefix("tree ")) {
                stack.push(t.to_string());
            }
            for l in text.lines().filter_map(|l| l.strip_prefix("parent ")) {
                stack.push(l.to_string());
            }
        } else if kind == "tree" {
            if let Some(ents) = parse_tree(&content) {
                for (_, _, s) in ents {
                    stack.push(s);
                }
            }
        }
    }

    // 3. Build a packfile (no deltas; stored-block deflate).
    let mut pack: Vec<u8> = Vec::new();
    pack.extend_from_slice(b"PACK");
    pack.extend_from_slice(&2u32.to_be_bytes());
    pack.extend_from_slice(&(need.len() as u32).to_be_bytes());
    for sha in need.iter() {
        let Some((kind, content)) = read_loose(sha) else { continue };
        // Pack objects carry the type + size in the varint header — the packed
        // bytes are the **header-free content** (embedding the loose-object
        // `type size\0` header makes git read a double header and mis-hash).
        let typ = match kind.as_str() {
            "commit" => 1,
            "tree" => 2,
            "blob" => 3,
            _ => 3,
        };
        let mut size = content.len() as u64;
        let mut first = ((typ as u64) << 4) | (size & 0x0f);
        size >>= 4;
        if size > 0 {
            first |= 0x80;
        }
        let mut hdr = vec![first as u8];
        while size > 0 {
            let mut b = (size & 0x7f) as u8;
            size >>= 7;
            if size > 0 {
                b |= 0x80;
            }
            hdr.push(b);
        }
        pack.extend_from_slice(&hdr);
        let z = zlib_deflate(&content).unwrap_or_default();
        pack.extend_from_slice(&z);
    }
    // Trailing sha1 of everything so far.
    let trailer = crate::sha1(&pack);
    let trailer = from_hex(&trailer).unwrap_or_default();
    pack.extend_from_slice(&trailer);

    // 4. Send: `old new ref\0caps\n` (pkt-line) + flush + packfile.
    let cmd = alloc::format!("{old} {head} refs/heads/{branch}\0report-status\n");
    let mut body = alloc::format!("{:04x}{cmd}0000", 4 + cmd.len()).into_bytes();
    body.extend_from_slice(&pack);
    match http(
        "POST",
        &alloc::format!("{base}/git-receive-pack"),
        &[("Content-Type", "application/x-git-receive-pack-request")],
        &body,
    ) {
        Ok((200, resp)) => {
            // git-http-backend reports failures in the body (`unpack ok` /
            // `ng <ref> <reason>` pkt-lines), never as a non-200 — read them.
            let text = String::from_utf8_lossy(&resp);
            if text.contains("unpack ok") && !text.contains("ng ") {
                alloc::format!("ok: pushed {branch} ({})", &head[..8.min(head.len())])
            } else {
                alloc::format!("error: push rejected by server ({} obj, {}B pack): {}", need.len(), pack.len(), text.trim().chars().take(120).collect::<String>())
            }
        }
        Ok((s, _)) => alloc::format!("error: receive-pack status {s}"),
        Err(_) => "error: receive-pack request failed".to_string(),
    }
}
