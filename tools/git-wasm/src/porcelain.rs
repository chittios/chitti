//! The rest of the porcelain: `remote`, `worktree`, `config`, `diff`, `rm`,
//! `mv`, `reset`, `restore`, `show`, `tag`, `rev-parse`, `ls-files`,
//! `cat-file`, `clean`, `switch`.
//!
//! All of it sits on the object/ref/index plumbing in [`crate::git`]; nothing
//! here invents a storage format. Where a command cannot be done honestly —
//! a three-way merge, a rebase — it says so and stops, rather than doing
//! something adjacent and calling it the same name.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::config;
use crate::git::{
    build_tree_from_index, checkout_tree, collect_blobs, current_branch, git_dir, head_commit,
    hash_object, read_index, read_loose, read_ref, repo_rel, working_files, write_index,
    write_loose, write_ref,
};
use crate::{base_dir, fs_exists, fs_read, fs_remove, fs_write};

/// Entry names in a store directory. `fs_list` yields `(name, is_dir, size)`;
/// everything here wants just the names.
/// Resolve a user-typed path against the repository root.
///
/// `crate::normalize_path` makes a bare name **store-absolute** (`a.txt` ->
/// `/a.txt`), which is right for a store key and wrong for a command argument:
/// `git mv a.txt b.txt` means the two files in this repository, not two at the
/// root of the filesystem.
/// Read a ref by its **full** name (`refs/tags/v1`, `refs/remotes/origin/main`).
///
/// [`crate::git::read_ref`] takes a *bare branch* name and prepends
/// `refs/heads/` itself, so passing it a full name looks for
/// `refs/heads/refs/tags/v1` — a lookup that always misses and reads as "no
/// such tag" rather than as a bug.
pub(crate) fn read_ref_raw(full: &str) -> Option<String> {
    let b = fs_read(&alloc::format!("{}/{full}", git_dir()))?;
    let s = String::from_utf8_lossy(&b).trim().to_string();
    (!s.is_empty()).then_some(s)
}

pub(crate) fn write_ref_raw(full: &str, sha: &str) {
    fs_write(
        &alloc::format!("{}/{full}", git_dir()),
        alloc::format!("{sha}\n").as_bytes(),
    );
}

fn repo_path(p: &str) -> String {
    if p.starts_with('/') {
        crate::normalize_path(p)
    } else {
        crate::normalize_path(&alloc::format!("{}/{p}", base_dir()))
    }
}

fn names_in(dir: &str) -> Vec<String> {
    crate::fs_list(dir).into_iter().map(|(n, _, _)| n).collect()
}

/// Which configuration file a `config` operation is about.
///
/// Git has three; we have the two that mean something here. There is no
/// `--system`: it would be a file no installer writes and no user could find,
/// so asking for one is refused by name rather than silently treated as global.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Scope {
    /// `.git/config` — this repository only.
    Local,
    /// `~/.gitconfig` — every repository this user opens.
    Global,
}

pub(crate) fn config_path(scope: Scope) -> String {
    match scope {
        Scope::Local => alloc::format!("{}/config", git_dir()),
        Scope::Global => alloc::format!("{}/.gitconfig", crate::user_home()),
    }
}

pub(crate) fn load_scope(scope: Scope) -> Vec<config::Entry> {
    match fs_read(&config_path(scope)) {
        Some(b) => config::parse(&String::from_utf8_lossy(&b)),
        None => Vec::new(),
    }
}

/// The local file, which is what `remote` and friends operate on.
pub(crate) fn load_config() -> Vec<config::Entry> {
    load_scope(Scope::Local)
}

fn save_scope(scope: Scope, entries: &[config::Entry]) -> bool {
    fs_write(&config_path(scope), config::render(entries).as_bytes())
}

fn save_config(entries: &[config::Entry]) -> bool {
    save_scope(Scope::Local, entries)
}

/// Look a key up with git's precedence: **local wins over global**.
///
/// That order is the whole point of having two files — a repository-specific
/// `user.email` has to beat the one you set once for everything, or the
/// per-repository setting is decorative.
pub(crate) fn config_lookup(name: &str) -> Option<String> {
    let local = load_scope(Scope::Local);
    if let Some(v) = config::get(&local, name) {
        return Some(v.to_string());
    }
    config::get(&load_scope(Scope::Global), name).map(|v| v.to_string())
}

fn in_repo() -> bool {
    fs_exists(&alloc::format!("{}/HEAD", git_dir()))
}

fn no_repo() -> String {
    "error: not a git repository (run /git init)".to_string()
}

// --- remote ----------------------------------------------------------------

/// `git remote [-v] | add <name> <url> | remove|rm <name> | rename <old> <new>
/// | set-url <name> <url> | get-url <name> | show <name>`
pub fn remote(args: &str) -> String {
    if !in_repo() {
        return no_repo();
    }
    let mut entries = load_config();
    let toks: Vec<&str> = args.split_whitespace().collect();
    match toks.first().copied() {
        None => {
            let names = config::subsections(&entries, "remote");
            if names.is_empty() {
                return "ok: no remotes".to_string();
            }
            names.join("\n")
        }
        Some("-v") | Some("--verbose") => {
            let names = config::subsections(&entries, "remote");
            if names.is_empty() {
                return "ok: no remotes".to_string();
            }
            let mut out = Vec::new();
            for n in names {
                let url = config::get(&entries, &alloc::format!("remote.{n}.url")).unwrap_or("");
                // git prints one line per direction; we have no separate push URL.
                out.push(alloc::format!("{n}\t{url} (fetch)"));
                out.push(alloc::format!("{n}\t{url} (push)"));
            }
            out.join("\n")
        }
        Some("add") => {
            let (Some(name), Some(url)) = (toks.get(1), toks.get(2)) else {
                return "usage: /git remote add <name> <url>".to_string();
            };
            if config::get(&entries, &alloc::format!("remote.{name}.url")).is_some() {
                return alloc::format!("error: remote '{name}' already exists");
            }
            config::set(&mut entries, &alloc::format!("remote.{name}.url"), url);
            config::set(
                &mut entries,
                &alloc::format!("remote.{name}.fetch"),
                &alloc::format!("+refs/heads/*:refs/remotes/{name}/*"),
            );
            save_config(&entries);
            alloc::format!("ok: added remote {name} -> {url}")
        }
        Some("remove") | Some("rm") => {
            let Some(name) = toks.get(1) else {
                return "usage: /git remote remove <name>".to_string();
            };
            if config::remove_section(&mut entries, "remote", name) == 0 {
                return alloc::format!("error: no such remote '{name}'");
            }
            save_config(&entries);
            alloc::format!("ok: removed remote {name}")
        }
        Some("rename") => {
            let (Some(old), Some(new)) = (toks.get(1), toks.get(2)) else {
                return "usage: /git remote rename <old> <new>".to_string();
            };
            let Some(url) = config::get(&entries, &alloc::format!("remote.{old}.url")).map(|s| s.to_string())
            else {
                return alloc::format!("error: no such remote '{old}'");
            };
            config::remove_section(&mut entries, "remote", old);
            config::set(&mut entries, &alloc::format!("remote.{new}.url"), &url);
            config::set(
                &mut entries,
                &alloc::format!("remote.{new}.fetch"),
                &alloc::format!("+refs/heads/*:refs/remotes/{new}/*"),
            );
            save_config(&entries);
            alloc::format!("ok: renamed remote {old} -> {new}")
        }
        Some("set-url") => {
            let (Some(name), Some(url)) = (toks.get(1), toks.get(2)) else {
                return "usage: /git remote set-url <name> <url>".to_string();
            };
            if config::get(&entries, &alloc::format!("remote.{name}.url")).is_none() {
                return alloc::format!("error: no such remote '{name}'");
            }
            config::set(&mut entries, &alloc::format!("remote.{name}.url"), url);
            save_config(&entries);
            alloc::format!("ok: {name} -> {url}")
        }
        Some("get-url") => {
            let Some(name) = toks.get(1) else {
                return "usage: /git remote get-url <name>".to_string();
            };
            match config::get(&entries, &alloc::format!("remote.{name}.url")) {
                Some(u) => u.to_string(),
                None => alloc::format!("error: no such remote '{name}'"),
            }
        }
        Some("show") => {
            let Some(name) = toks.get(1) else {
                return "usage: /git remote show <name>".to_string();
            };
            let Some(url) = config::get(&entries, &alloc::format!("remote.{name}.url")) else {
                return alloc::format!("error: no such remote '{name}'");
            };
            let mut out = alloc::format!("* remote {name}\n  URL: {url}");
            for (sha, r) in all_refs() {
                if let Some(b) = r.strip_prefix(&alloc::format!("refs/remotes/{name}/")) {
                    out.push_str(&alloc::format!("\n  {b} {}", &sha[..8.min(sha.len())]));
                }
            }
            out
        }
        Some(other) => alloc::format!("error: unknown remote subcommand '{other}'"),
    }
}

/// Every ref in the repository as `(sha, name)`.
pub(crate) fn all_refs() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for dir in ["refs/heads", "refs/tags", "refs/remotes"] {
        collect_refs(&alloc::format!("{}/{dir}", git_dir()), dir, &mut out);
    }
    out
}

fn collect_refs(dir: &str, prefix: &str, out: &mut Vec<(String, String)>) {
    for name in names_in(dir) {
        let full = alloc::format!("{dir}/{name}");
        let refname = alloc::format!("{prefix}/{name}");
        if let Some(b) = fs_read(&full) {
            let sha = String::from_utf8_lossy(&b).trim().to_string();
            if sha.len() == 40 {
                out.push((sha, refname));
                continue;
            }
        }
        collect_refs(&full, &refname, out);
    }
}

// --- worktree --------------------------------------------------------------

/// `git worktree add <path> [branch] | list | remove <path>`
///
/// A linked worktree is a directory whose `.git` is a **file** pointing at
/// `<repo>/.git/worktrees/<id>`, which in turn records where the checkout lives
/// and holds that tree's own `HEAD`. The object store and `refs/` stay shared —
/// that sharing is the whole point, and copying them instead would produce two
/// repositories that silently diverge.
pub fn worktree(args: &str) -> String {
    if !in_repo() {
        return no_repo();
    }
    let toks: Vec<&str> = args.split_whitespace().collect();
    let main = git_dir();
    match toks.first().copied() {
        None | Some("list") => {
            let mut out = alloc::format!("{}  {}  [{}]", base_dir(), short(&head_commit()), current_branch());
            for id in names_in(&alloc::format!("{main}/worktrees")) {
                let base = alloc::format!("{main}/worktrees/{id}");
                let path = read_line_file(&alloc::format!("{base}/gitdir"))
                    .map(|p| p.trim_end_matches("/.git").to_string())
                    .unwrap_or_else(|| id.clone());
                let head = read_line_file(&alloc::format!("{base}/HEAD")).unwrap_or_default();
                let branch = head.strip_prefix("ref: ").unwrap_or("detached").to_string();
                let sha = read_ref(branch.trim().trim_start_matches("refs/heads/")).unwrap_or_default();
                out.push_str(&alloc::format!(
                    "\n{path}  {}  [{}]",
                    short(&Some(sha)),
                    branch.trim().trim_start_matches("refs/heads/")
                ));
            }
            out
        }
        Some("add") => {
            let Some(path) = toks.get(1) else {
                return "usage: /git worktree add <path> [branch]".to_string();
            };
            let path = repo_path(path);
            if fs_exists(&alloc::format!("{path}/.git")) {
                return alloc::format!("error: {path} is already a worktree");
            }
            // The branch defaults to one named after the directory, as git does.
            let branch = toks
                .get(2)
                .map(|s| s.to_string())
                .unwrap_or_else(|| path.rsplit('/').next().unwrap_or("work").to_string());
            let refname = alloc::format!("refs/heads/{branch}");
            let Some(base_sha) = read_ref(&branch).or_else(head_commit) else {
                return "error: nothing committed yet — a worktree needs a commit".to_string();
            };
            // **A branch may be checked out in only one worktree.** Two trees on
            // one branch would each commit over the other's work, which git
            // refuses for exactly that reason.
            if branch == current_branch() {
                return alloc::format!(
                    "error: '{branch}' is already checked out in the main worktree"
                );
            }
            for id in names_in(&alloc::format!("{main}/worktrees")) {
                let h = read_line_file(&alloc::format!("{main}/worktrees/{id}/HEAD")).unwrap_or_default();
                if h.trim() == alloc::format!("ref: {refname}") {
                    return alloc::format!("error: '{branch}' is already checked out in another worktree");
                }
            }
            if read_ref(&branch).is_none() {
                write_ref(&branch, &base_sha);
            }

            let id = path.rsplit('/').next().unwrap_or("wt").to_string();
            let admin = alloc::format!("{main}/worktrees/{id}");
            fs_write(&alloc::format!("{admin}/gitdir"), alloc::format!("{path}/.git\n").as_bytes());
            fs_write(&alloc::format!("{admin}/HEAD"), alloc::format!("ref: {refname}\n").as_bytes());
            fs_write(&alloc::format!("{admin}/commondir"), b"../..\n");
            // The `.git` **file** is what makes the directory a worktree.
            fs_write(&alloc::format!("{path}/.git"), alloc::format!("gitdir: {admin}\n").as_bytes());

            // Check the branch's tree out into the new directory.
            let Some((_, commit)) = read_loose(&base_sha) else {
                return "error: could not read the branch commit".to_string();
            };
            let Some(tree) = tree_of_commit(&commit) else {
                return "error: the commit names no tree".to_string();
            };
            // **Not** `set_cwd` + `checkout_tree`: that would repoint `git_dir()`
            // at the new tree's `.git` *file*, and every object read would fail
            // against it. The blobs are written directly instead, which is also
            // what keeps the object store shared rather than copied.
            for (_, blob, rel) in collect_blobs(&tree) {
                if let Some((_, data)) = read_loose(&blob) {
                    fs_write(&alloc::format!("{path}/{rel}"), &data);
                }
            }
            alloc::format!("ok: worktree {path} on {branch} ({})", short(&Some(base_sha)))
        }
        Some("remove") => {
            let Some(path) = toks.get(1) else {
                return "usage: /git worktree remove <path>".to_string();
            };
            let path = repo_path(path);
            let id = path.rsplit('/').next().unwrap_or("").to_string();
            let admin = alloc::format!("{main}/worktrees/{id}");
            if !fs_exists(&alloc::format!("{admin}/gitdir")) {
                return alloc::format!("error: {path} is not a registered worktree");
            }
            for f in ["gitdir", "HEAD", "commondir"] {
                fs_remove(&alloc::format!("{admin}/{f}"));
            }
            fs_remove(&alloc::format!("{path}/.git"));
            alloc::format!("ok: removed worktree {path} (its files were left in place)")
        }
        Some(other) => alloc::format!("error: unknown worktree subcommand '{other}'"),
    }
}

fn read_line_file(path: &str) -> Option<String> {
    fs_read(path).map(|b| String::from_utf8_lossy(&b).trim().to_string())
}

fn short(sha: &Option<String>) -> String {
    match sha {
        Some(s) if s.len() >= 8 => s[..8].to_string(),
        Some(s) => s.clone(),
        None => "(none)".to_string(),
    }
}

pub(crate) fn tree_of_commit(commit: &[u8]) -> Option<String> {
    String::from_utf8_lossy(commit)
        .lines()
        .find_map(|l| l.strip_prefix("tree ").map(|s| s.trim().to_string()))
}

// --- config ----------------------------------------------------------------

/// `git config [--local|--global] <name> [value] | --get | --unset | --list`
///
/// Reads merge the two files with **local winning**; writes go to whichever
/// file was named, defaulting to local as git does. A `--global` read outside a
/// repository still works, which is what makes `/git config --global user.name`
/// usable before you have cloned anything.
pub fn config_cmd(args: &str) -> String {
    let mut scope: Option<Scope> = None;
    let mut rest: Vec<&str> = Vec::new();
    for t in args.split_whitespace() {
        match t {
            "--global" => scope = Some(Scope::Global),
            "--local" => scope = Some(Scope::Local),
            "--system" => {
                return "error: --system configuration does not exist on this OS \
                        (use --global)"
                    .to_string()
            }
            other => rest.push(other),
        }
    }
    // Only a *write* or an explicitly-local operation needs a repository; a
    // global read or write does not.
    let needs_repo = scope != Some(Scope::Global);
    if needs_repo && !in_repo() {
        return no_repo();
    }
    let write_scope = scope.unwrap_or(Scope::Local);

    match rest.first().copied() {
        None | Some("--list") | Some("-l") => {
            let entries = match scope {
                Some(s) => load_scope(s),
                None => {
                    // Merged view, local last so it is the one a reader sees.
                    let mut e = load_scope(Scope::Global);
                    for l in load_scope(Scope::Local) {
                        config::set(&mut e, &l.name(), &l.value);
                    }
                    e
                }
            };
            if entries.is_empty() {
                return "ok: no configuration".to_string();
            }
            entries
                .iter()
                .map(|e| alloc::format!("{}={}", e.name(), e.value))
                .collect::<Vec<_>>()
                .join("\n")
        }
        Some("--get") => {
            let Some(name) = rest.get(1) else {
                return "usage: /git config [--global] --get <name>".to_string();
            };
            let found = match scope {
                Some(s) => config::get(&load_scope(s), name).map(|v| v.to_string()),
                None => config_lookup(name),
            };
            found.unwrap_or_else(|| "error: no such key".to_string())
        }
        Some("--unset") => {
            let Some(name) = rest.get(1) else {
                return "usage: /git config [--global] --unset <name>".to_string();
            };
            let mut entries = load_scope(write_scope);
            if config::unset(&mut entries, name) == 0 {
                return "error: no such key".to_string();
            }
            save_scope(write_scope, &entries);
            alloc::format!("ok: unset {name}")
        }
        Some(name) => match rest.get(1) {
            None => {
                let found = match scope {
                    Some(s) => config::get(&load_scope(s), name).map(|v| v.to_string()),
                    None => config_lookup(name),
                };
                found.unwrap_or_else(|| "error: no such key".to_string())
            }
            Some(_) => {
                // The value is everything after the name, so it may contain
                // spaces — `user.name Ada Lovelace` is one value, not two.
                let value = rest[1..].join(" ");
                let mut entries = load_scope(write_scope);
                if !config::set(&mut entries, name, &value) {
                    return "error: a name must be section.key or section.sub.key".to_string();
                }
                save_scope(write_scope, &entries);
                alloc::format!(
                    "ok: {name}={value}{}",
                    if write_scope == Scope::Global { " (global)" } else { "" }
                )
            }
        },
    }
}

// --- index / working tree --------------------------------------------------

/// `git rm [--cached] <path…>`
pub fn rm(args: &str) -> String {
    if !in_repo() {
        return no_repo();
    }
    let mut cached = false;
    let mut paths: Vec<String> = Vec::new();
    for t in args.split_whitespace() {
        match t {
            "--cached" => cached = true,
            "-r" | "-f" => {} // accepted and implied
            p => paths.push(repo_path(p)),
        }
    }
    if paths.is_empty() {
        return "usage: /git rm [--cached] <path…>".to_string();
    }
    let mut index = read_index();
    let mut n = 0usize;
    for p in &paths {
        let rel = repo_rel(p);
        let before = index.len();
        index.retain(|(_, _, name)| name != &rel && !name.starts_with(&alloc::format!("{rel}/")));
        n += before - index.len();
        if !cached {
            fs_remove(p);
        }
    }
    write_index(&index);
    alloc::format!("ok: removed {n} path(s) from the index")
}

/// `git mv <src> <dst>`
pub fn mv(args: &str) -> String {
    if !in_repo() {
        return no_repo();
    }
    let toks: Vec<&str> = args.split_whitespace().collect();
    let (Some(src), Some(dst)) = (toks.first(), toks.get(1)) else {
        return "usage: /git mv <src> <dst>".to_string();
    };
    let (src, dst) = (repo_path(src), repo_path(dst));
    let Some(data) = fs_read(&src) else {
        return alloc::format!("error: no such file {src}");
    };
    if !fs_write(&dst, &data) {
        return alloc::format!("error: could not write {dst}");
    }
    fs_remove(&src);
    let (from, to) = (repo_rel(&src), repo_rel(&dst));
    let mut index = read_index();
    for e in index.iter_mut() {
        if e.2 == from {
            e.2 = to.clone();
        }
    }
    write_index(&index);
    alloc::format!("ok: {from} -> {to}")
}

/// `git reset [--soft|--mixed|--hard] [<commit>]`
pub fn reset(args: &str) -> String {
    if !in_repo() {
        return no_repo();
    }
    let mut mode = "--mixed";
    let mut target: Option<String> = None;
    for t in args.split_whitespace() {
        match t {
            "--soft" | "--mixed" | "--hard" => mode = t,
            other => target = Some(other.to_string()),
        }
    }
    let Some(sha) = target.and_then(|t| resolve_rev(&t)).or_else(head_commit) else {
        return "error: nothing to reset to".to_string();
    };
    // Move the branch first: every mode does that much.
    let branch = current_branch();
    write_ref(&branch, &sha);
    if mode == "--soft" {
        return alloc::format!("ok: {branch} -> {} (index and files kept)", short(&Some(sha)));
    }
    let Some((_, commit)) = read_loose(&sha) else {
        return "error: could not read that commit".to_string();
    };
    let Some(tree) = tree_of_commit(&commit) else {
        return "error: the commit names no tree".to_string();
    };
    let blobs = collect_blobs(&tree);
    write_index(&blobs);
    if mode == "--hard" {
        checkout_tree(&tree, "");
        return alloc::format!("ok: {branch} -> {} (files replaced)", short(&Some(sha)));
    }
    alloc::format!("ok: {branch} -> {} (files kept)", short(&Some(sha)))
}

/// `git restore [--staged] <path…>` — put a file back from the index or HEAD.
pub fn restore(args: &str) -> String {
    if !in_repo() {
        return no_repo();
    }
    let mut staged = false;
    let mut paths: Vec<String> = Vec::new();
    for t in args.split_whitespace() {
        match t {
            "--staged" | "--cached" => staged = true,
            "--" => {}
            p => paths.push(repo_path(p)),
        }
    }
    if paths.is_empty() {
        return "usage: /git restore [--staged] <path…>".to_string();
    }
    let head_blobs = head_commit()
        .and_then(|s| read_loose(&s))
        .and_then(|(_, c)| tree_of_commit(&c))
        .map(|t| collect_blobs(&t))
        .unwrap_or_default();
    let index = read_index();
    let mut n = 0usize;
    for p in &paths {
        let rel = repo_rel(p);
        let source = if staged { &head_blobs } else { &index };
        let Some((_, sha, _)) = source.iter().find(|(_, _, name)| name == &rel) else {
            continue;
        };
        if staged {
            // Only the index moves; the file on disk is left alone.
            let mut idx = read_index();
            if let Some(e) = idx.iter_mut().find(|(_, _, n)| n == &rel) {
                e.1 = sha.clone();
            }
            write_index(&idx);
        } else if let Some((_, data)) = read_loose(sha) {
            fs_write(p, &data);
        }
        n += 1;
    }
    alloc::format!("ok: restored {n} path(s)")
}

/// `git clean [-n]` — remove untracked files (dry-run without `-f`, as git does).
pub fn clean(args: &str) -> String {
    if !in_repo() {
        return no_repo();
    }
    let force = args.split_whitespace().any(|t| t == "-f" || t == "-fd");
    let index = read_index();
    let mut out = Vec::new();
    for f in working_files(&base_dir()) {
        let rel = repo_rel(&f);
        if index.iter().any(|(_, _, n)| n == &rel) {
            continue;
        }
        if force {
            fs_remove(&f);
            out.push(alloc::format!("removed {rel}"));
        } else {
            out.push(alloc::format!("would remove {rel}"));
        }
    }
    if out.is_empty() {
        return "ok: nothing to clean".to_string();
    }
    if !force {
        out.push("(use /git clean -f to actually remove them)".to_string());
    }
    out.join("\n")
}

// --- inspection ------------------------------------------------------------

/// Resolve a revision: a full or abbreviated sha, `HEAD`, a branch, or a tag.
pub(crate) fn resolve_rev(rev: &str) -> Option<String> {
    let rev = rev.trim();
    if rev.is_empty() {
        return None;
    }
    if rev == "HEAD" {
        return head_commit();
    }
    if let Some(s) = read_ref(rev) {
        return Some(s); // a bare branch name
    }
    for prefix in ["refs/tags/", "refs/remotes/"] {
        if let Some(s) = read_ref_raw(&alloc::format!("{prefix}{rev}")) {
            return Some(s);
        }
    }
    if rev.starts_with("refs/") {
        if let Some(s) = read_ref_raw(rev) {
            return Some(s);
        }
    }
    // A full sha, or a unique abbreviation of one.
    if rev.len() >= 4 && rev.bytes().all(|b| b.is_ascii_hexdigit()) {
        if rev.len() == 40 && read_loose(rev).is_some() {
            return Some(rev.to_string());
        }
        let matches: Vec<String> = all_object_shas()
            .into_iter()
            .filter(|s| s.starts_with(rev))
            .collect();
        // An ambiguous abbreviation is an error, not a coin flip.
        if matches.len() == 1 {
            return Some(matches[0].clone());
        }
    }
    None
}

fn all_object_shas() -> Vec<String> {
    let mut out = Vec::new();
    let root = alloc::format!("{}/objects", git_dir());
    for d in names_in(&root) {
        if d.len() != 2 {
            continue;
        }
        // **The file name is the whole sha here**, not the last 38 characters
        // real git uses — `obj_path` writes `objects/<ab>/<full-sha>`. Joining
        // the directory to the name would build a 42-character string that
        // matches no abbreviation and no full sha.
        for f in names_in(&alloc::format!("{root}/{d}")) {
            if f.len() == 40 {
                out.push(f);
            }
        }
    }
    out
}

/// `git rev-parse <rev…>`
pub fn rev_parse(args: &str) -> String {
    if !in_repo() {
        return no_repo();
    }
    let mut out = Vec::new();
    for t in args.split_whitespace() {
        match t {
            "--abbrev-ref" => continue,
            "--git-dir" => out.push(git_dir()),
            "--show-toplevel" => out.push(base_dir()),
            rev => match resolve_rev(rev) {
                Some(s) => out.push(s),
                None => out.push(alloc::format!("error: unknown revision '{rev}'")),
            },
        }
    }
    if out.is_empty() {
        out.push(head_commit().unwrap_or_else(|| "error: no commits".to_string()));
    }
    out.join("\n")
}

/// `git ls-files` — what is in the index.
pub fn ls_files(_args: &str) -> String {
    if !in_repo() {
        return no_repo();
    }
    let index = read_index();
    if index.is_empty() {
        return "ok: the index is empty".to_string();
    }
    index
        .iter()
        .map(|(_, _, name)| name.clone())
        .collect::<Vec<_>>()
        .join("\n")
}

/// `git cat-file [-t|-s|-p] <object>`
pub fn cat_file(args: &str) -> String {
    if !in_repo() {
        return no_repo();
    }
    let toks: Vec<&str> = args.split_whitespace().collect();
    let (flag, rev) = match toks.first().copied() {
        Some(f) if f.starts_with('-') => (f, toks.get(1).copied().unwrap_or("")),
        _ => ("-p", toks.first().copied().unwrap_or("")),
    };
    let Some(sha) = resolve_rev(rev) else {
        return alloc::format!("error: unknown object '{rev}'");
    };
    let Some((kind, content)) = read_loose(&sha) else {
        return alloc::format!("error: cannot read object {sha}");
    };
    match flag {
        "-t" => kind,
        "-s" => alloc::format!("{}", content.len()),
        _ => {
            if kind == "tree" {
                crate::git::parse_tree(&content)
                    .unwrap_or_default()
                    .iter()
                    .map(|(mode, sha, name)| alloc::format!("{mode} {sha}\t{name}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            } else {
                String::from_utf8_lossy(&content).into_owned()
            }
        }
    }
}

/// `git show [<rev>]` — a commit's metadata and the paths it changed.
pub fn show(args: &str) -> String {
    if !in_repo() {
        return no_repo();
    }
    let rev = args.split_whitespace().next().unwrap_or("HEAD");
    let Some(sha) = resolve_rev(rev) else {
        return alloc::format!("error: unknown revision '{rev}'");
    };
    let Some((kind, content)) = read_loose(&sha) else {
        return alloc::format!("error: cannot read {sha}");
    };
    if kind != "commit" {
        return cat_file(&alloc::format!("-p {rev}"));
    }
    let text = String::from_utf8_lossy(&content).into_owned();
    let tree = tree_of_commit(&content).unwrap_or_default();
    let parent = text
        .lines()
        .find_map(|l| l.strip_prefix("parent ").map(|s| s.trim().to_string()));
    let mut out = alloc::format!("commit {sha}\n{text}");
    let now = collect_blobs(&tree);
    let before = parent
        .and_then(|p| read_loose(&p))
        .and_then(|(_, c)| tree_of_commit(&c))
        .map(|t| collect_blobs(&t))
        .unwrap_or_default();
    for line in diff_trees(&before, &now) {
        out.push('\n');
        out.push_str(&line);
    }
    out
}

/// Name-status difference between two blob lists.
fn diff_trees(
    before: &[(String, String, String)],
    after: &[(String, String, String)],
) -> Vec<String> {
    let mut out = Vec::new();
    for (_, sha, name) in after {
        match before.iter().find(|(_, _, n)| n == name) {
            None => out.push(alloc::format!("A\t{name}")),
            Some((_, old, _)) if old != sha => out.push(alloc::format!("M\t{name}")),
            _ => {}
        }
    }
    for (_, _, name) in before {
        if !after.iter().any(|(_, _, n)| n == name) {
            out.push(alloc::format!("D\t{name}"));
        }
    }
    out
}

/// `git diff [--cached] [<rev>]` — a **line** diff of the working tree.
///
/// A real unified diff with hunk headers, which is what makes the output useful
/// rather than decorative; the algorithm is a common-prefix/suffix trim plus a
/// straight replacement in the middle. That is not Myers, so a small edit in a
/// large file produces one big hunk rather than several tight ones — correct,
/// and honest about being coarse, which is better than a wrong minimal diff.
pub fn diff(args: &str) -> String {
    if !in_repo() {
        return no_repo();
    }
    let cached = args.split_whitespace().any(|t| t == "--cached" || t == "--staged");
    let rev = args
        .split_whitespace()
        .find(|t| !t.starts_with("--"))
        .unwrap_or("HEAD");
    let head_blobs = resolve_rev(rev)
        .and_then(|s| read_loose(&s))
        .and_then(|(_, c)| tree_of_commit(&c))
        .map(|t| collect_blobs(&t))
        .unwrap_or_default();

    let mut out: Vec<String> = Vec::new();
    if cached {
        // Index against the commit: name-status only, since the index holds
        // shas rather than content the user edited.
        let index = read_index();
        for line in diff_trees(&head_blobs, &index) {
            out.push(line);
        }
        if out.is_empty() {
            return "ok: no staged changes".to_string();
        }
        return out.join("\n");
    }

    for f in working_files(&base_dir()) {
        let rel = repo_rel(&f);
        let old = head_blobs
            .iter()
            .find(|(_, _, n)| n == &rel)
            .and_then(|(_, sha, _)| read_loose(sha))
            .map(|(_, c)| c)
            .unwrap_or_default();
        let new = fs_read(&f).unwrap_or_default();
        if old == new {
            continue;
        }
        out.push(alloc::format!("diff --git a/{rel} b/{rel}"));
        if old.is_empty() {
            out.push("new file".to_string());
        }
        out.push(alloc::format!("--- a/{rel}"));
        out.push(alloc::format!("+++ b/{rel}"));
        out.extend(unified(&String::from_utf8_lossy(&old), &String::from_utf8_lossy(&new)));
    }
    for (_, _, rel) in &head_blobs {
        if !fs_exists(&alloc::format!("{}/{rel}", base_dir())) {
            out.push(alloc::format!("diff --git a/{rel} b/{rel}"));
            out.push("deleted file".to_string());
        }
    }
    if out.is_empty() {
        return "ok: no changes".to_string();
    }
    out.join("\n")
}

/// One hunk covering everything between the common prefix and suffix.
fn unified(old: &str, new: &str) -> Vec<String> {
    let a: Vec<&str> = old.lines().collect();
    let b: Vec<&str> = new.lines().collect();
    let mut start = 0usize;
    while start < a.len() && start < b.len() && a[start] == b[start] {
        start += 1;
    }
    let mut back = 0usize;
    while back < a.len() - start && back < b.len() - start && a[a.len() - 1 - back] == b[b.len() - 1 - back] {
        back += 1;
    }
    let a_mid = &a[start..a.len() - back];
    let b_mid = &b[start..b.len() - back];
    let mut out = alloc::vec![alloc::format!(
        "@@ -{},{} +{},{} @@",
        start + 1,
        a_mid.len(),
        start + 1,
        b_mid.len()
    )];
    for l in a_mid {
        out.push(alloc::format!("-{l}"));
    }
    for l in b_mid {
        out.push(alloc::format!("+{l}"));
    }
    out
}

/// `git tag [<name> [<rev>]] | -d <name>` — lightweight tags only.
pub fn tag(args: &str) -> String {
    if !in_repo() {
        return no_repo();
    }
    let toks: Vec<&str> = args.split_whitespace().collect();
    match toks.first().copied() {
        None | Some("-l") | Some("--list") => {
            let names: Vec<String> = all_refs()
                .into_iter()
                .filter_map(|(_, r)| r.strip_prefix("refs/tags/").map(|s| s.to_string()))
                .collect();
            if names.is_empty() {
                return "ok: no tags".to_string();
            }
            names.join("\n")
        }
        Some("-d") | Some("--delete") => {
            let Some(name) = toks.get(1) else {
                return "usage: /git tag -d <name>".to_string();
            };
            let path = alloc::format!("{}/refs/tags/{name}", git_dir());
            if !fs_exists(&path) {
                return alloc::format!("error: no such tag '{name}'");
            }
            fs_remove(&path);
            alloc::format!("ok: deleted tag {name}")
        }
        // An annotated tag needs a tag *object*; we make lightweight ones and
        // say so rather than silently producing a different kind of tag.
        Some("-a") | Some("-m") => {
            "error: annotated tags are not implemented (lightweight tags only)".to_string()
        }
        Some(name) => {
            let rev = toks.get(1).copied().unwrap_or("HEAD");
            let Some(sha) = resolve_rev(rev) else {
                return alloc::format!("error: unknown revision '{rev}'");
            };
            if fs_exists(&alloc::format!("{}/refs/tags/{name}", git_dir())) {
                return alloc::format!("error: tag '{name}' already exists");
            }
            write_ref_raw(&alloc::format!("refs/tags/{name}"), &sha);
            alloc::format!("ok: tag {name} -> {}", short(&Some(sha)))
        }
    }
}

/// Stage every changed path (`git add -A` in effect), returning how many.
pub(crate) fn stage_all() -> usize {
    let mut index = read_index();
    let mut n = 0;
    for f in working_files(&base_dir()) {
        let rel = repo_rel(&f);
        let Some(data) = fs_read(&f) else { continue };
        let sha = hash_object("blob", &data);
        if write_loose("blob", &data).is_none() {
            continue;
        }
        match index.iter_mut().find(|(_, _, n)| n == &rel) {
            Some(e) => {
                if e.1 != sha {
                    e.1 = sha;
                    n += 1;
                }
            }
            None => {
                index.push(("100644".to_string(), sha, rel));
                n += 1;
            }
        }
    }
    write_index(&index);
    let _ = build_tree_from_index(&read_index());
    n
}
