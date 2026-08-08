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
    fs_write, from_hex, host_ssh, http, to_hex,
};
use crate::git::{
    checkout_tree, collect_blobs, current_branch, git_dir, head_commit, obj_raw, parse_tree,
    read_loose, write_ref,
};
use crate::{fs_read, zlib_deflate};
use alloc::collections::BTreeMap;

/// Capabilities we request from upload-pack (must be a subset of what the
/// server advertises — requesting `multi_ack_detailed`/`report-status` from a
/// server that only offers `multi_ack` makes it reject the whole request).
///
/// Deliberately **not** `thin-pack`: a thin pack may carry deltas whose base was
/// never sent, on the promise that the client already has it. A clone has nothing,
/// so the only thing that capability can buy here is `delta base not found`.
const CAPS: &str = "multi_ack ofs-delta no-progress";

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

/// How a delta object names the object it is a delta against. Both forms sit
/// **raw** in the packfile, ahead of the compressed delta — see the walk.
enum BaseRef {
    /// Not a delta.
    None,
    /// `OBJ_OFS_DELTA`: bytes to count back from this object's own header.
    Ofs(u64),
    /// `OBJ_REF_DELTA`: the base's sha, which may be an object on disk.
    Sha(String),
}

/// One object recovered from the packfile, with the sha it hashes to (computed
/// once — it names the loose file *and* indexes the object as a delta base).
struct PackObj {
    kind: String,
    sha: String,
    content: Vec<u8>,
}

/// Write a loose object whose sha is already known, skipping the re-hash
/// [`write_loose`] would do.
fn write_object(kind: &str, sha: &str, content: &[u8]) -> bool {
    let Some(z) = zlib_deflate(&obj_raw(kind, content)) else { return false };
    fs_write(&crate::git::obj_path(sha), &z)
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

/// Inflate the pack object whose zlib body starts at `pack[start..]`, expected to
/// decompress to `want_size` bytes; returns it with the number of input bytes the
/// stream occupied.
///
/// One call. `host_inflate` tolerates trailing bytes and reports how much of the
/// input the stream consumed, which is exactly the question "where does the next
/// object start" — so handing it the whole remainder is correct. This used to
/// widen a window one byte at a time looking for a length that matched, i.e. a
/// full inflate per byte of every object: for this pack's 172 objects that is on
/// the order of a hundred thousand decompressions of up to 180 KiB each, which
/// exhausts the fuel budget long before it finishes.
fn inflate_object(pack: &[u8], start: usize, want_size: u64) -> Option<(Vec<u8>, usize)> {
    let (dec, consumed) = crate::zlib_decompress_hint(pack.get(start..)?, want_size as usize)?;
    // The declared size is the object's own claim about itself; a stream that
    // decompresses to something else means the pack is not what its header says.
    if dec.len() as u64 != want_size {
        return None;
    }
    Some((dec, consumed))
}

// --- clone ----------------------------------------------------------------------


/// A remote's transport: smart HTTP, or `git-upload-pack` over SSH.
///
/// The two differ in more than the bytes they move. Over HTTP each phase is its
/// own request, so the body's end delimits the advertisement. Over SSH there is
/// **one bidirectional stream**: the server sends its advertisement and then
/// waits, so the client must stop reading at the flush packet rather than at end
/// of stream — a reader that waits for EOF deadlocks against a server that is
/// waiting for the wants.
enum Transport {
    Http { base: String },
    Ssh { session: u32 },
}

impl Transport {
    fn open(url: &str) -> Result<Self, String> {
        match crate::sshurl::parse(url) {
            None => Ok(Transport::Http {
                base: url.trim_end_matches('/').to_string(),
            }),
            Some(r) => {
                let req = alloc::format!(
                    "{{\"op\":\"open\",\"u\":\"{}\",\"h\":\"{}\",\"p\":{},\"c\":\"{}\"}}",
                    json_escape(&r.user),
                    json_escape(&r.host),
                    r.port,
                    json_escape(&r.upload_pack())
                );
                match ssh_call(&req) {
                    Ok((id, _)) if id > 0 => Ok(Transport::Ssh { session: id as u32 }),
                    Ok((_, msg)) | Err(msg) => Err(alloc::format!("error: ssh: {msg}")),
                }
            }
        }
    }

    /// The ref advertisement.
    fn advertise(&self) -> Result<Vec<u8>, String> {
        match self {
            Transport::Http { base } => match http(
                "GET",
                &alloc::format!("{base}/info/refs?service=git-upload-pack"),
                &[("Accept", "application/x-git-upload-pack-advertisement")],
                b"",
            ) {
                Ok((200, b)) => Ok(b),
                Ok((s, _)) => Err(alloc::format!("error: info/refs status {s}")),
                Err(_) => Err("error: info/refs request failed".to_string()),
            },
            // **Stop at the flush packet**, not at EOF.
            Transport::Ssh { session } => ssh_read_until_flush(*session),
        }
    }

    /// Send the wants and read the packfile.
    fn fetch(&self, request: &str) -> Result<Vec<u8>, String> {
        match self {
            Transport::Http { base } => match http(
                "POST",
                &alloc::format!("{base}/git-upload-pack"),
                &[
                    ("Content-Type", "application/x-git-upload-pack-request"),
                    ("Accept", "application/x-git-upload-pack-result"),
                ],
                request.as_bytes(),
            ) {
                Ok((200, b)) => Ok(b),
                Ok((s, _)) => Err(alloc::format!("error: upload-pack status {s}")),
                Err(_) => Err("error: upload-pack request failed".to_string()),
            },
            Transport::Ssh { session } => {
                let b64 = b64_encode(request.as_bytes());
                let req = alloc::format!("{{\"op\":\"write\",\"s\":{session},\"b\":\"{b64}\"}}");
                if let Ok((n, msg)) = ssh_call(&req) {
                    if n < 0 {
                        return Err(alloc::format!("error: ssh write: {msg}"));
                    }
                }
                // Read to the end of the stream: the pack is the last thing the
                // server sends, so here EOF *is* the delimiter.
                ssh_read_all(*session)
            }
        }
    }

    fn close(&self) {
        if let Transport::Ssh { session } = self {
            let _ = ssh_call(&alloc::format!("{{\"op\":\"close\",\"s\":{session}}}"));
        }
    }
}

/// One `host_ssh` call. Returns `(code, message)` — the message carries the
/// host's error text on a negative code, which is the difference between
/// "permission denied" and "no route" reaching the user.
fn ssh_call(req: &str) -> Result<(i64, String), String> {
    let mut buf = alloc::vec![0u8; 64 * 1024];
    let r = unsafe {
        host_ssh(
            req.as_ptr(),
            req.len() as i32,
            buf.as_mut_ptr(),
            buf.len() as i32,
        )
    };
    if r < 0 {
        let msg = String::from_utf8_lossy(&buf[..buf.len().min(512)])
            .trim_end_matches('\0')
            .trim()
            .to_string();
        return Ok((r, if msg.is_empty() { alloc::format!("error {r}") } else { msg }));
    }
    Ok((r, String::new()))
}

/// Read one chunk from an SSH session, growing the buffer when the host says the
/// data was longer than it fit — the same contract every other filling import
/// follows.
fn ssh_read_chunk(session: u32) -> Result<Vec<u8>, String> {
    let req = alloc::format!("{{\"op\":\"read\",\"s\":{session}}}");
    let mut cap = 64 * 1024usize;
    for _ in 0..2 {
        let mut buf = alloc::vec![0u8; cap];
        let r = unsafe {
            host_ssh(
                req.as_ptr(),
                req.len() as i32,
                buf.as_mut_ptr(),
                buf.len() as i32,
            )
        };
        if r < 0 {
            let msg = String::from_utf8_lossy(&buf[..buf.len().min(512)]).trim().to_string();
            return Err(alloc::format!("error: ssh read: {msg}"));
        }
        let n = r as usize;
        if n <= cap {
            buf.truncate(n);
            return Ok(buf);
        }
        cap = n;
    }
    Err("error: ssh read did not fit twice".to_string())
}

/// Read pkt-lines until the flush packet that ends the advertisement.
fn ssh_read_until_flush(session: u32) -> Result<Vec<u8>, String> {
    let mut out: Vec<u8> = Vec::new();
    loop {
        // Is a top-level flush already present?
        if pkt_ends_with_flush(&out) {
            return Ok(out);
        }
        let chunk = ssh_read_chunk(session)?;
        if chunk.is_empty() {
            // EOF before a flush: the server said something and quit, which is
            // how a missing repository or a refused key arrives.
            if out.is_empty() {
                return Err("error: the server closed the connection with no refs".to_string());
            }
            return Ok(out);
        }
        out.extend_from_slice(&chunk);
    }
}

/// Walk the pkt-line framing and report whether a flush (`0000`) terminates it.
fn pkt_ends_with_flush(buf: &[u8]) -> bool {
    let mut i = 0usize;
    while i + 4 <= buf.len() {
        let Ok(hex) = core::str::from_utf8(&buf[i..i + 4]) else {
            return false;
        };
        let Ok(len) = usize::from_str_radix(hex, 16) else {
            return false;
        };
        if len == 0 {
            return true; // flush
        }
        if len < 4 || i + len > buf.len() {
            return false; // incomplete
        }
        i += len;
    }
    false
}

fn ssh_read_all(session: u32) -> Result<Vec<u8>, String> {
    let mut out: Vec<u8> = Vec::new();
    loop {
        let chunk = ssh_read_chunk(session)?;
        if chunk.is_empty() {
            return Ok(out);
        }
        out.extend_from_slice(&chunk);
    }
}

fn json_escape(s: &str) -> String {
    let mut o = String::new();
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c => o.push(c),
        }
    }
    o
}

fn b64_encode(data: &[u8]) -> String {
    const T: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut o = String::new();
    for c in data.chunks(3) {
        let b = [c[0], *c.get(1).unwrap_or(&0), *c.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        o.push(T[(n >> 18) as usize & 63] as char);
        o.push(T[(n >> 12) as usize & 63] as char);
        o.push(if c.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        o.push(if c.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    o
}

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

    // 1. Advertised refs, over whichever transport the URL names.
    let transport = match Transport::open(&base) {
        Ok(t) => t,
        Err(e) => return e,
    };
    let adv = match transport.advertise() {
        Ok(b) => b,
        Err(e) => {
            transport.close();
            return e;
        }
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
    let pack = match transport.fetch(&req) {
        Ok(b) => b,
        Err(e) => {
            transport.close();
            return e;
        }
    };
    transport.close();
    // The packfile starts at the `PACK` magic (pkt-line ACK/NAK precedes it).
    let Some(pi) = pack.windows(4).position(|w| w == b"PACK") else {
        return "error: no packfile in the upload-pack response".to_string();
    };

    // 3. Parse the packfile. Deltas resolve against an earlier object by stream
    // position (OFS) or by sha (REF / loose), so both indexes are built as we go —
    // the sha one especially, because the alternative is re-hashing every object
    // already parsed on every REF_DELTA.
    let mut parsed: Vec<PackObj> = Vec::new();
    let mut by_pos: BTreeMap<usize, usize> = BTreeMap::new();
    let mut by_sha: BTreeMap<String, usize> = BTreeMap::new();
    let mut p = pi;
    // Header: magic, version, object count. Bounds-checked because a short read
    // here is an index panic, and a guest panic is `loop {}` until the fuel runs
    // out — an error message beats a hang.
    if pack.len() < p + 12 {
        return "error: truncated PACK header".to_string();
    }
    p += 4;
    let _version = u32::from_be_bytes(pack[p..p + 4].try_into().unwrap_or([0; 4]));
    p += 4;
    let count = u32::from_be_bytes(pack[p..p + 4].try_into().unwrap_or([0; 4])) as usize;
    p += 4;
    for i in 0..count {
        let obj_start = p;
        let (typ, want_size) = match read_varint(&pack, &mut p) {
            Some(v) => v,
            None => return alloc::format!("error: bad object header (object {i} of {count})"),
        };
        // **A delta object's base reference is RAW, not compressed.** It sits in the
        // packfile between the type/size header and the zlib stream — an offset
        // varint for OFS_DELTA, twenty bytes of sha for REF_DELTA — and the
        // compressed payload that follows is the delta *in full*.
        //
        // Reading it out of the decompressed bytes instead, as this did, means
        // inflating from twenty bytes too early: the decoder is handed the sha and
        // answers `not deflate`, which is reported as a corrupt object. That is what
        // `object inflate failed (object 5 of 172)` was — the first delta in the
        // pack, and so every clone of any repository whose history the server chose
        // to delta-compress, which is all of them past a few commits.
        let base = match typ {
            7 => {
                let Some(raw) = pack.get(p..p + 20) else {
                    return "error: truncated ref-delta base".to_string();
                };
                p += 20;
                BaseRef::Sha(to_hex(raw))
            }
            6 => match read_ofs_offset(&pack, &mut p) {
                Some(off) => BaseRef::Ofs(off),
                None => return "error: bad ofs-delta".to_string(),
            },
            _ => BaseRef::None,
        };
        let (dec, stream_len) = match inflate_object(&pack, p, want_size) {
            Some(v) => v,
            None => {
                return alloc::format!(
                    "error: object inflate failed (object {i} of {count}, type {typ}, {} pack bytes)",
                    pack.len() - pi
                )
            }
        };
        p += stream_len;
        let (kind, content) = match base {
            BaseRef::None => (kind_name(typ).to_string(), dec),
            b => {
                let found = match &b {
                    // A REF_DELTA may name a base that arrived earlier in this pack
                    // or one already on disk.
                    BaseRef::Sha(sha) => by_sha
                        .get(sha)
                        .map(|&i| (parsed[i].kind.clone(), parsed[i].content.clone()))
                        .or_else(|| read_loose(sha)),
                    // An OFS_DELTA counts backwards from its own header to the base
                    // object's header.
                    BaseRef::Ofs(off) => match obj_start.checked_sub(*off as usize) {
                        Some(pos) => by_pos
                            .get(&pos)
                            .map(|&i| (parsed[i].kind.clone(), parsed[i].content.clone())),
                        None => return "error: bad ofs-delta offset".to_string(),
                    },
                    BaseRef::None => None,
                };
                let Some((bk, bc)) = found else {
                    return "error: delta base not found".to_string();
                };
                // A delta reconstructs an object of the base's *kind* — the type
                // field said "delta", so this is the only place the kind comes from.
                match apply_delta(&bc, &dec) {
                    Some(f) => (bk, f),
                    None => return "error: delta apply failed".to_string(),
                }
            }
        };
        // Hashed once, here: the sha is needed to index this object as a delta base
        // and again to name its loose file, and it is the same sha both times.
        let sha = crate::git::hash_object(&kind, &content);
        by_pos.insert(obj_start, parsed.len());
        by_sha.insert(sha.clone(), parsed.len());
        parsed.push(PackObj { kind, sha, content });
    }

    // 4. Write loose objects, refs, checkout HEAD.
    let mut n_obj = 0usize;
    for o in &parsed {
        if o.kind != "unknown" && write_object(&o.kind, &o.sha, &o.content) {
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
        checkout_tree(&t, "");
        // Stage the checked-out HEAD tree so `git status` is clean after clone —
        // same walk, so the index and the working tree cannot disagree, and each
        // entry keeps the tree's own mode rather than a hardcoded 100644 (which
        // made every executable file look modified on the next commit).
        crate::git::write_index(&collect_blobs(&t));
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
