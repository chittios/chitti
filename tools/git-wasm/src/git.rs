//! Local git operations: the loose-object database, repo state (HEAD/refs/
//! index), and the working-tree commands (init/status/add/commit/log/branch/
//! checkout). All I/O goes through the host imports in [`crate`].

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use crate::{from_hex, fs_exists, fs_list, fs_read, fs_write, git_root, sha1, to_hex, zlib_decompress, zlib_deflate};

pub(crate) fn git_dir() -> String {
    alloc::format!("{}/.git", git_root())
}

pub(crate) fn obj_path(sha: &str) -> String {
    alloc::format!("{}/objects/{}/{sha}", git_dir(), &sha[..2])
}

pub(crate) fn obj_raw(kind: &str, content: &[u8]) -> Vec<u8> {
    let mut raw = alloc::format!("{kind} {}\0", content.len()).into_bytes();
    raw.extend_from_slice(content);
    raw
}

pub(crate) fn hash_object(kind: &str, content: &[u8]) -> String {
    sha1(&obj_raw(kind, content))
}

pub(crate) fn write_loose(kind: &str, content: &[u8]) -> Option<String> {
    let sha = hash_object(kind, content);
    let z = zlib_deflate(&obj_raw(kind, content))?;
    fs_write(&obj_path(&sha), &z).then_some(sha)
}

pub(crate) fn read_loose(sha: &str) -> Option<(String, Vec<u8>)> {
    if sha.len() != 40 {
        return None;
    }
    let z = fs_read(&obj_path(sha))?;
    let (raw, _) = zlib_decompress(&z)?;
    let nul = raw.iter().position(|&b| b == 0)?;
    let head = core::str::from_utf8(&raw[..nul]).ok()?;
    let (kind, _sz) = head.split_once(' ')?;
    Some((kind.to_string(), raw[nul + 1..].to_vec()))
}

pub(crate) fn parse_tree(content: &[u8]) -> Option<Vec<(String, String, String)>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < content.len() {
        let sp = content[i..].iter().position(|&b| b == b' ')? + i;
        let nul = content[sp + 1..].iter().position(|&b| b == 0)? + sp + 1;
        if nul + 20 > content.len() {
            return None;
        }
        out.push((
            core::str::from_utf8(&content[i..sp]).ok()?.to_string(),
            core::str::from_utf8(&content[sp + 1..nul]).ok()?.to_string(),
            to_hex(&content[nul + 1..nul + 21]),
        ));
        i = nul + 21;
    }
    Some(out)
}

pub(crate) fn tree_content(entries: &[(String, String, String)]) -> Option<Vec<u8>> {
    let mut e = entries.to_vec();
    e.sort_by(|a, b| a.1.cmp(&b.1));
    let mut out = Vec::new();
    for (mode, name, sha) in e {
        let bin = from_hex(&sha)?;
        out.extend_from_slice(mode.as_bytes());
        out.push(b' ');
        out.extend_from_slice(name.as_bytes());
        out.push(0);
        out.extend_from_slice(&bin);
    }
    Some(out)
}

pub(crate) fn current_branch() -> String {
    match fs_read(&alloc::format!("{}/HEAD", git_dir())) {
        Some(h) => {
            let s = String::from_utf8_lossy(&h).trim().to_string();
            s.strip_prefix("ref: refs/heads/").map(|b| b.to_string()).unwrap_or(s)
        }
        None => String::from("master"),
    }
}

pub(crate) fn read_ref(name: &str) -> Option<String> {
    let b = fs_read(&alloc::format!("{}/refs/heads/{name}", git_dir()))?;
    let s = String::from_utf8_lossy(&b).trim().to_string();
    if s.is_empty() {
        return None;
    }
    Some(s)
}

pub(crate) fn write_ref(name: &str, sha: &str) {
    fs_write(&alloc::format!("{}/refs/heads/{name}", git_dir()), alloc::format!("{sha}\n").as_bytes());
}

pub(crate) fn head_commit() -> Option<String> {
    let br = current_branch();
    read_ref(&br)
}

pub(crate) fn read_index() -> Vec<(String, String, String)> {
    let mut out = Vec::new();
    if let Some(b) = fs_read(&alloc::format!("{}/index", git_dir())) {
        for line in String::from_utf8_lossy(&b).lines() {
            let mut it = line.splitn(3, ' ');
            if let (Some(mode), Some(sha), Some(path)) = (it.next(), it.next(), it.next()) {
                out.push((mode.to_string(), sha.to_string(), path.to_string()));
            }
        }
    }
    out
}

pub(crate) fn write_index(entries: &[(String, String, String)]) {
    let mut s = String::new();
    for (mode, sha, path) in entries {
        s.push_str(&alloc::format!("{mode} {sha} {path}\n"));
    }
    fs_write(&alloc::format!("{}/index", git_dir()), s.as_bytes());
}

pub(crate) fn repo_rel(full: &str) -> String {
    let root = git_root();
    full.strip_prefix(&root).unwrap_or(full).trim_start_matches('/').to_string()
}

/// Does `.gitignore` in `repo` (and any `.gitignore` in a directory on `rel`'s
/// path) mark `rel` as ignored? Only consulted for **untracked** paths, which
/// is where gitignore applies.
fn is_ignored(repo: &str, rel: &str) -> bool {
    if rel.is_empty() {
        return false;
    }
    let dirs: Vec<&str> = rel.split('/').filter(|c| !c.is_empty()).collect();
    let mut ignored = false;
    let mut prefix = String::new();
    let mut cur = repo.to_string();
    for depth in 0..=dirs.len() {
        if let Some(content) = fs_read(&alloc::format!("{cur}/.gitignore")) {
            for line in String::from_utf8_lossy(&content).lines() {
                let pat = line.trim_end();
                if pat.is_empty() || pat.starts_with('#') {
                    continue;
                }
                let (neg, p) = if let Some(rest) = pat.strip_prefix('!') {
                    (true, rest.trim_end())
                } else {
                    (false, pat)
                };
                if match_pattern(p, rel) {
                    ignored = !neg;
                }
            }
        }
        if depth == dirs.len() {
            break;
        }
        if !prefix.is_empty() {
            prefix.push('/');
        }
        prefix.push_str(dirs[depth]);
        cur = alloc::format!("{repo}/{prefix}");
    }
    ignored
}

/// fnmatch: `*` (not `/`), `**` (any, incl `/`), `?` (single, not `/`).
fn fnmatch(pat: &str, text: &str) -> bool {
    fn go(p: &[u8], t: &[u8]) -> bool {
        if p.is_empty() {
            return t.is_empty();
        }
        match p[0] {
            b'*' => {
                let double = p.len() > 1 && p[1] == b'*';
                let rest = if double { &p[2..] } else { &p[1..] };
                for i in 0..=t.len() {
                    if !double && t[..i].contains(&b'/') {
                        break;
                    }
                    if go(rest, &t[i..]) {
                        return true;
                    }
                }
                false
            }
            b'?' => !t.is_empty() && t[0] != b'/' && go(&p[1..], &t[1..]),
            c => !t.is_empty() && t[0] == c && go(&p[1..], &t[1..]),
        }
    }
    go(pat.as_bytes(), text.as_bytes())
}

/// Does one gitignore pattern match the (root-relative) `rel` path?
fn match_pattern(pattern: &str, rel: &str) -> bool {
    let mut pat = pattern.trim_end_matches('/');
    if pat.is_empty() {
        return false;
    }
    let anchored = pat.starts_with('/');
    let pat = pat.trim_start_matches('/');
    let comps: Vec<&str> = rel.split('/').filter(|c| !c.is_empty()).collect();
    if comps.is_empty() {
        return false;
    }
    if anchored {
        // Root-anchored: relative to the repo root.
        if fnmatch(pat, rel) {
            return true;
        }
        let pcomps: Vec<&str> = pat.split('/').collect();
        return pcomps.len() < comps.len()
            && fnmatch(&pcomps.join("/"), &comps[..pcomps.len()].join("/"));
    }
    if pat.contains('/') {
        // Relative to the `.gitignore`'s directory (we only collect root-level
        // ones, so root-relative applies).
        if fnmatch(pat, rel) {
            return true;
        }
        let pcomps: Vec<&str> = pat.split('/').collect();
        return pcomps.len() < comps.len()
            && fnmatch(&pcomps.join("/"), &comps[..pcomps.len()].join("/"));
    }
    // No slash: matches the basename at any depth; matching an intermediate
    // directory ignores everything under it.
    for (i, c) in comps.iter().enumerate() {
        if fnmatch(pat, c) {
            return true; // a component matches -> this path is ignored
        }
        let _ = i;
    }
    false
}

pub(crate) fn working_files(dir: &str) -> Vec<String> {
    let repo = git_root();
    let mut out = Vec::new();
    for (name, is_dir, _sz) in fs_list(dir) {
        if name == ".git" {
            continue;
        }
        let full = alloc::format!("{dir}/{name}");
        let rel = full.strip_prefix(&repo).unwrap_or(&full).trim_start_matches('/').to_string();
        if is_ignored(&repo, &rel) {
            continue;
        }
        if is_dir {
            out.extend(working_files(&full));
        } else {
            out.push(full);
        }
    }
    out
}

pub(crate) fn build_tree(dir: &str) -> Option<(String, usize)> {
    let mut entries = Vec::new();
    let mut nfiles = 0;
    for (name, is_dir, _sz) in fs_list(dir) {
        if name == ".git" {
            continue;
        }
        let full = alloc::format!("{dir}/{name}");
        if is_dir {
            if let Some((sha, n)) = build_tree(&full) {
                entries.push((String::from("40000"), name, sha));
                nfiles += n;
            }
        } else if let Some(bytes) = fs_read(&full) {
            let sha = write_loose("blob", &bytes)?;
            entries.push((String::from("100644"), name, sha));
            nfiles += 1;
        }
    }
    Some((write_loose("tree", &tree_content(&entries)?)?, nfiles))
}

pub(crate) fn build_tree_from_index(index: &[(String, String, String)]) -> Option<(String, usize)> {
    fn rec(prefix: &str, index: &[(String, String, String)]) -> Option<(String, usize)> {
        let mut names: Vec<String> = Vec::new();
        for (_, _, path) in index {
            if path.starts_with(prefix) {
                if let Some(first) = path[prefix.len()..].split('/').next() {
                    if !first.is_empty() && !names.iter().any(|n| n == first) {
                        names.push(first.to_string());
                    }
                }
            }
        }
        names.sort_unstable();
        let mut entries = Vec::new();
        let mut nfiles = 0;
        for name in names {
            let child_prefix = if prefix.is_empty() {
                alloc::format!("{name}/")
            } else {
                alloc::format!("{prefix}{name}/")
            };
            if let Some((mode, sha, _)) = index.iter().find(|(_, _, p)| *p == child_prefix.trim_end_matches('/')) {
                entries.push((mode.clone(), name, sha.clone()));
                nfiles += 1;
            } else if let Some((sha, n)) = rec(&child_prefix, index) {
                entries.push((String::from("40000"), name, sha));
                nfiles += n;
            }
        }
        Some((write_loose("tree", &tree_content(&entries)?)?, nfiles))
    }
    rec("", index)
}

// --- commands ------------------------------------------------------------------

pub(crate) fn git_init(args: &str) -> String {
    // `git init [dir]` — target defaults to the shell's current directory.
    let target = args
        .split_whitespace()
        .next()
        .filter(|t| !t.starts_with('-'))
        .map(|t| crate::normalize_path(t))
        .unwrap_or_else(crate::base_dir);
    fs_write(&alloc::format!("{target}/.git/HEAD"), b"ref: refs/heads/master\n");
    crate::set_cwd(&target);
    alloc::format!("ok: initialized empty git repo at {target}")
}

pub(crate) fn git_status(_args: &str) -> String {
    let br = current_branch();
    let head = head_commit();
    let mut out = alloc::format!("on branch {br}");
    if let Some(h) = &head {
        out.push_str(&alloc::format!(" at {}", &h[..8.min(h.len())]));
    }
    let index = read_index();
    let working = working_files(&git_root());
    // Index vs HEAD tree: an index entry is "staged" unless the HEAD tree holds
    // the same path with the same blob.
    let head_blobs: Vec<(String, String)> = head
        .as_ref()
        .and_then(|h| {
            // The commit names its tree; walk that tree's blobs.
            let (_, c) = read_loose(h)?;
            let text = String::from_utf8_lossy(&c);
            let tree_sha = text.lines().find_map(|l| l.strip_prefix("tree "))?;
            let (_, tc) = read_loose(&tree_sha)?;
            let t = parse_tree(&tc)?;
            collect_blobs(&t)
        })
        .unwrap_or_default();
    let staged: Vec<String> = index
        .iter()
        .filter(|(_, sha, path)| !head_blobs.iter().any(|(s, p)| s == sha && p == path))
        .map(|(_, _, p)| p.clone())
        .collect();
    let indexed: Vec<&String> = index.iter().map(|(_, _, p)| p).collect();
    let unstaged: Vec<String> = working
        .iter()
        .filter(|f| !indexed.iter().any(|p| **p == repo_rel(f)))
        .cloned()
        .collect();
    if !staged.is_empty() {
        out.push_str("\nstaged:");
        for p in staged {
            out.push_str(&alloc::format!("\n  + {p}"));
        }
    }
    if !unstaged.is_empty() {
        out.push_str("\nuntracked / modified:");
        for p in unstaged {
            out.push_str(&alloc::format!("\n  ? {p}"));
        }
    }
    if !out.contains("staged:") && !out.contains("untracked") {
        out.push_str("\nnothing to commit, working tree clean");
    }
    out
}

pub(crate) fn collect_blobs(tree: &[(String, String, String)]) -> Option<Vec<(String, String)>> {
    let mut out = Vec::new();
    for (mode, name, sha) in tree {
        if mode == "40000" {
            let (_, c) = read_loose(sha)?;
            out.extend(collect_blobs(&parse_tree(&c)?)?);
        } else {
            out.push((sha.clone(), name.clone()));
        }
    }
    Some(out)
}

pub(crate) fn git_add(args: &str) -> String {
    let mut index = read_index();
    let mut paths: Vec<String> = Vec::new();
    let mut add_all = false;
    for t in args.split_whitespace() {
        if t == "." || t == "-A" || t == "-a" {
            add_all = true;
        } else if t.starts_with('/') {
            let rel = t.trim_start_matches('/').to_string();
            paths.push(rel);
        }
    }
    let root = git_root();
    let targets: Vec<String> = if add_all || paths.is_empty() {
        working_files(&root)
    } else {
        paths.iter().map(|p| alloc::format!("{root}/{p}")).collect()
    };
    let mut added = 0usize;
    for full in targets {
        let Some(bytes) = fs_read(&full) else { continue };
        if bytes.is_empty() && !fs_exists(&full) {
            continue;
        }
        let Some(sha) = write_loose("blob", &bytes) else { continue };
        let rel = repo_rel(&full);
        if !index.iter().any(|(_, s, p)| p == &rel && s == &sha) {
            index.push((String::from("100644"), sha, rel));
            added += 1;
        }
    }
    write_index(&index);
    alloc::format!("ok: staged {added} file(s)")
}

pub(crate) fn git_commit(args: &str) -> String {
    let toks: Vec<&str> = args.split_whitespace().collect();
    let mut msg = String::new();
    let mut i = 0;
    while i < toks.len() {
        if toks[i] == "-m" && i + 1 < toks.len() {
            msg = toks[i + 1..].join(" ");
            break;
        }
        i += 1;
    }
    if msg.trim().is_empty() {
        return "error: commit message required (-m \"...\")".to_string();
    }
    let index = read_index();
    let (tree, n) = if index.is_empty() {
        build_tree(&git_root()).unwrap_or((String::new(), 0))
    } else {
        build_tree_from_index(&index).unwrap_or((String::new(), 0))
    };
    if tree.is_empty() || n == 0 {
        return "error: nothing to commit (no files)".to_string();
    }
    let parents: Vec<String> = head_commit().into_iter().collect();
    let mut content = alloc::format!("tree {tree}\n");
    for p in &parents {
        content.push_str(&alloc::format!("parent {p}\n"));
    }
    let who = alloc::format!("Chitti <chitti@localhost> {} +0000", now_unix());
    content.push_str(&alloc::format!("author {who}\ncommitter {who}\n\n{msg}\n"));
    let Some(sha) = write_loose("commit", content.as_bytes()) else {
        return "error: could not write commit".to_string();
    };
    write_ref(&current_branch(), &sha);
    alloc::format!("ok: [{}{}] {msg}", &sha[..8.min(sha.len())], if parents.is_empty() { " (root)" } else { "" })
}

pub(crate) fn git_log(args: &str) -> String {
    let mut n = args.split_whitespace().find_map(|t| t.parse::<usize>().ok()).unwrap_or(20);
    let mut sha = head_commit();
    let mut out = String::new();
    while let Some(h) = sha {
        if n == 0 {
            break;
        }
        n -= 1;
        let Some((_, c)) = read_loose(&h) else { break };
        let text = String::from_utf8_lossy(&c).to_string();
        let msg = text.split("\n\n").nth(1).map(|m| m.trim()).unwrap_or("");
        out.push_str(&alloc::format!("commit {}\n  {}\n\n", &h[..12.min(h.len())], msg));
        sha = text.lines().find_map(|l| l.strip_prefix("parent ").map(|p| p.to_string()));
    }
    if out.is_empty() {
        return "error: no commits yet".to_string();
    }
    out
}

pub(crate) fn git_branch(_args: &str) -> String {
    let cur = current_branch();
    let mut out = String::new();
    for (name, is_dir, _sz) in fs_list(&alloc::format!("{}/refs/heads", git_dir())) {
        if is_dir || name.is_empty() {
            continue;
        }
        let mark = if name == cur { "*" } else { " " };
        out.push_str(&alloc::format!("{mark} {name}\n"));
    }
    if out.is_empty() {
        out.push_str(&alloc::format!("* {cur}\n"));
    }
    out
}

pub(crate) fn git_checkout(args: &str) -> String {
    let branch = args
        .split_whitespace()
        .find(|t| !t.starts_with('/') && !t.starts_with('-'))
        .unwrap_or("master")
        .to_string();
    let sha = read_ref(&branch).unwrap_or_else(|| branch.clone());
    let Some((_, commit)) = read_loose(&sha) else {
        return alloc::format!("error: no such branch or commit '{branch}'");
    };
    let tree_sha = String::from_utf8_lossy(&commit)
        .lines()
        .find_map(|l| l.strip_prefix("tree ").map(|t| t.to_string()));
    let Some(tree_sha) = tree_sha else {
        return "error: commit has no tree".to_string();
    };
    checkout_tree(&tree_sha);
    fs_write(&alloc::format!("{}/HEAD", git_dir()), alloc::format!("ref: refs/heads/{branch}\n").as_bytes());
    alloc::format!("ok: switched to '{branch}' ({})", &sha[..8.min(sha.len())])
}

pub(crate) fn checkout_tree(tree_sha: &str) {
    let Some((_, t)) = read_loose(tree_sha) else { return };
    let Some(ents) = parse_tree(&t) else { return };
    for (mode, name, sha) in ents {
        let root = git_root();
        let full = alloc::format!("{root}/{name}");
        if mode == "40000" {
            checkout_tree(&sha);
        } else if let Some((_, blob)) = read_loose(&sha) {
            fs_write(&full, &blob);
        }
    }
}

pub(crate) fn now_unix() -> i64 {
    unsafe { crate::host_now_unix() }
}

/// Parse a full `/git <args>` line and dispatch.
pub fn command(args: &str) -> String {
    let args = args.trim();
    let (sub, rest) = match args.split_once(char::is_whitespace) {
        Some((s, r)) => (s, r.trim()),
        None => (args, ""),
    };
    match sub {
        "" | "help" => alloc::format!(
            "git> usage: /git init|status|add <path>|commit -m <msg>|log|branch|checkout <branch>|clone <url> [dir]|push <url>"
        ),
        "init" => git_init(rest),
        "status" => git_status(rest),
        "add" => git_add(rest),
        "commit" => git_commit(rest),
        "log" => git_log(rest),
        "branch" => git_branch(rest),
        "checkout" => git_checkout(rest),
        "clone" => crate::remote::clone(rest),
        "push" => crate::remote::push(rest),
        other => alloc::format!("git> unknown subcommand '{other}' (try /git help)"),
    }
}
