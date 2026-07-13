//! Pure virtual-path helpers over a flat path→bytes store.
//!
//! The store holds path-like keys (`/agent/1/SOUL.md`, `/configs/core/ui.json`)
//! without real directory inodes. These functions synthesise a Linux-like
//! directory view: immediate children only, directories inferred from path
//! prefixes (and optional `.keep` markers). No I/O — callers supply the path
//! list so unit tests exercise the logic off-hardware.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// One entry in a virtual directory listing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirEntry {
    /// Basename only (no `/`).
    pub name: String,
    pub is_dir: bool,
    /// File size in bytes; `0` for directories.
    pub size: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntryClass {
    File,
    Dir,
}

/// Normalise a path: collapse `/./`, `//`, resolve `..`, strip trailing `/`
/// (except for the root `/`). Bare names (no leading `/`) stay relative.
pub fn normalize(path: &str) -> String {
    let p = path.trim();
    if p.is_empty() || p == "." {
        return String::from("/");
    }
    let absolute = p.starts_with('/');
    let mut parts: Vec<&str> = Vec::new();
    for seg in p.split('/') {
        if seg.is_empty() || seg == "." {
            continue;
        }
        if seg == ".." {
            parts.pop();
            continue;
        }
        parts.push(seg);
    }
    if absolute {
        if parts.is_empty() {
            String::from("/")
        } else {
            alloc::format!("/{}", parts.join("/"))
        }
    } else if parts.is_empty() {
        String::from(".")
    } else {
        parts.join("/")
    }
}

/// Parent directory of `path`, or `None` for `/` / bare single-component.
pub fn parent(path: &str) -> Option<String> {
    let n = normalize(path);
    if n == "/" || n == "." {
        return None;
    }
    match n.rfind('/') {
        Some(0) => Some(String::from("/")),
        Some(i) => Some(n[..i].to_string()),
        None => None,
    }
}

/// Final path component.
pub fn basename(path: &str) -> String {
    let n = normalize(path);
    if n == "/" {
        return String::from("/");
    }
    n.rsplit('/').next().unwrap_or(&n).to_string()
}

/// Join `dir` and `name` (name should not contain `/`).
pub fn join(dir: &str, name: &str) -> String {
    let d = normalize(dir);
    let n = name.trim().trim_matches('/');
    if n.is_empty() {
        return d;
    }
    if d == "/" {
        alloc::format!("/{n}")
    } else if d == "." {
        n.to_string()
    } else {
        alloc::format!("{d}/{n}")
    }
}

/// Relative rest of `path` under `dir`, or `None` if not a descendant.
fn relative_rest<'a>(dir: &str, path: &'a str) -> Option<&'a str> {
    if dir == "/" {
        if path == "/" {
            return None;
        }
        if path.starts_with('/') {
            return Some(path.trim_start_matches('/'));
        }
        // Bare store keys (`skills/1/...`) count as root children.
        return Some(path);
    }
    if path == dir {
        return None;
    }
    let prefix = alloc::format!("{dir}/");
    path.strip_prefix(prefix.as_str())
}

/// Whether `path` has any store key beneath it (directory).
pub fn has_children(path: &str, keys: &[String]) -> bool {
    let n = normalize(path);
    keys.iter().any(|k| relative_rest(&n, &normalize(k)).is_some())
}

/// Classify a path: file (exact key, no children), dir (has children), or missing.
pub fn classify(path: &str, keys: &[String]) -> Option<EntryClass> {
    let n = normalize(path);
    if n == "/" {
        return Some(EntryClass::Dir);
    }
    let exact = keys.iter().any(|k| normalize(k) == n);
    if has_children(&n, keys) {
        return Some(EntryClass::Dir);
    }
    if exact {
        return Some(EntryClass::File);
    }
    None
}

/// Immediate children of `dir` from `(path, size)` store keys.
/// Directories first when `dirs_first`; `.keep` markers are not listed as
/// files (they only create empty-dir parents).
pub fn list_dir(dir: &str, entries: &[(String, usize)], dirs_first: bool) -> Vec<DirEntry> {
    let dir = normalize(dir);
    // name -> (is_dir, size)
    let mut map: BTreeMap<String, (bool, usize)> = BTreeMap::new();

    for (path, size) in entries {
        let kn = normalize(path);
        let Some(rest) = relative_rest(&dir, &kn) else {
            continue;
        };
        if rest.is_empty() {
            continue;
        }
        let mut segs = rest.split('/');
        let Some(name) = segs.next() else {
            continue;
        };
        if name.is_empty() {
            continue;
        }
        let deeper = segs.next().is_some();
        // `dir/.keep` would show name=".keep" — skip as listing noise.
        if !deeper && name == ".keep" {
            continue;
        }
        let e = map.entry(name.to_string()).or_insert((false, 0));
        if deeper {
            e.0 = true;
            e.1 = 0;
        } else if !e.0 {
            e.1 = *size;
        }
    }

    let mut out: Vec<DirEntry> = map
        .into_iter()
        .map(|(name, (is_dir, size))| DirEntry {
            name,
            is_dir,
            size: if is_dir { 0 } else { size },
        })
        .collect();

    if dirs_first {
        out.sort_by(|a, b| match (a.is_dir, b.is_dir) {
            (true, false) => core::cmp::Ordering::Less,
            (false, true) => core::cmp::Ordering::Greater,
            _ => a.name.cmp(&b.name),
        });
    } else {
        out.sort_by(|a, b| a.name.cmp(&b.name));
    }
    out
}

/// All keys under `dir` (descendants). When `include_self`, also the exact
/// file key `dir` if present.
pub fn keys_under(dir: &str, keys: &[String], include_self: bool) -> Vec<String> {
    let dir = normalize(dir);
    let mut out = Vec::new();
    for k in keys {
        let kn = normalize(k);
        if include_self && kn == dir {
            out.push(kn);
            continue;
        }
        if relative_rest(&dir, &kn).is_some() {
            out.push(kn);
        }
    }
    out
}

/// Destination when copying/moving `src` → `dst`. If `dst` is a directory,
/// land at `dst/basename(src)`.
pub fn resolve_dest(src: &str, dst: &str, dst_is_dir: bool) -> String {
    let src = normalize(src);
    let dst = normalize(dst);
    if dst_is_dir {
        join(&dst, &basename(&src))
    } else {
        dst
    }
}

/// Map `key` from a tree rooted at `src_root` into the same relative place
/// under `dst_root`. `None` if `key` is not under `src_root`.
pub fn remap_under(src_root: &str, dst_root: &str, key: &str) -> Option<String> {
    let src_root = normalize(src_root);
    let dst_root = normalize(dst_root);
    let key = normalize(key);
    if key == src_root {
        return Some(dst_root);
    }
    if src_root == "/" {
        let rest = key.trim_start_matches('/');
        if rest.is_empty() {
            return Some(dst_root);
        }
        return Some(if dst_root == "/" {
            alloc::format!("/{rest}")
        } else {
            alloc::format!("{dst_root}/{rest}")
        });
    }
    let prefix = alloc::format!("{src_root}/");
    let rest = key.strip_prefix(prefix.as_str())?;
    Some(if dst_root == "/" {
        alloc::format!("/{rest}")
    } else {
        alloc::format!("{dst_root}/{rest}")
    })
}

/// Long listing line: type flag, size, name.
pub fn format_long(e: &DirEntry) -> String {
    if e.is_dir {
        alloc::format!("d {:>10}  {}/", "", e.name)
    } else {
        alloc::format!("- {:>10}  {}", e.size, e.name)
    }
}

/// Short listing name (dirs get a trailing `/`).
pub fn format_short(e: &DirEntry) -> String {
    if e.is_dir {
        alloc::format!("{}/", e.name)
    } else {
        e.name.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::String;
    use alloc::vec;

    fn entries(paths: &[&str]) -> Vec<(String, usize)> {
        paths.iter().map(|p| (String::from(*p), 10)).collect()
    }

    #[test_case]
    fn normalize_collapses_and_strips() {
        assert_eq!(normalize("/"), "/");
        assert_eq!(normalize("/agent//1/./SOUL.md"), "/agent/1/SOUL.md");
        assert_eq!(normalize("/agent/1/../2"), "/agent/2");
        assert_eq!(normalize("foo/bar"), "foo/bar");
        assert_eq!(normalize(""), "/");
    }

    #[test_case]
    fn list_dir_immediate_children_only() {
        let ents = entries(&[
            "/agent/1/SOUL.md",
            "/agent/1/MEMORY.md",
            "/agent/1/skills/.keep",
            "/agent/1/memory/.keep",
            "/agent/9001/SOUL.md",
            "/configs/core/ui.json",
            "/downloads/pic.png",
        ]);
        let root = list_dir("/", &ents, true);
        let names: Vec<&str> = root.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, ["agent", "configs", "downloads"]);
        assert!(root.iter().all(|e| e.is_dir));

        let agent = list_dir("/agent", &ents, true);
        assert_eq!(agent.len(), 2);
        assert!(agent.iter().all(|e| e.is_dir));

        let home = list_dir("/agent/1", &ents, true);
        let home_names: Vec<String> = home.iter().map(|e| format_short(e)).collect();
        assert!(home_names.iter().any(|n| n == "SOUL.md"));
        assert!(home_names.iter().any(|n| n == "MEMORY.md"));
        assert!(home_names.iter().any(|n| n == "skills/"));
        assert!(home_names.iter().any(|n| n == "memory/"));
        assert!(!home.iter().any(|e| e.name.contains('/')));
    }

    #[test_case]
    fn classify_and_keys_under() {
        let ks: Vec<String> = vec![
            String::from("/agent/1/SOUL.md"),
            String::from("/agent/1/skills/.keep"),
            String::from("/agent/1/note.txt"),
        ];
        assert_eq!(classify("/agent/1", &ks), Some(EntryClass::Dir));
        assert_eq!(classify("/agent/1/SOUL.md", &ks), Some(EntryClass::File));
        assert_eq!(classify("/missing", &ks), None);

        let under = keys_under("/agent/1", &ks, false);
        assert_eq!(under.len(), 3);
        assert!(under.iter().all(|k| k.starts_with("/agent/1/")));
    }

    #[test_case]
    fn remap_copy_tree() {
        assert_eq!(
            remap_under("/a", "/b", "/a/x.txt").as_deref(),
            Some("/b/x.txt")
        );
        assert_eq!(
            remap_under("/a", "/b/c", "/a/d/e").as_deref(),
            Some("/b/c/d/e")
        );
        assert_eq!(remap_under("/a", "/b", "/z"), None);
        assert_eq!(
            resolve_dest("/a/f.txt", "/dest", true),
            String::from("/dest/f.txt")
        );
        assert_eq!(
            resolve_dest("/a/f.txt", "/dest/new.txt", false),
            String::from("/dest/new.txt")
        );
    }

    #[test_case]
    fn hides_keep_marker_noise() {
        let ents = entries(&["/agent/1/skills/.keep"]);
        let skills = list_dir("/agent/1/skills", &ents, true);
        assert!(skills.is_empty(), "empty dir shows empty: {skills:?}");
        let home = list_dir("/agent/1", &ents, true);
        assert_eq!(home.len(), 1);
        assert!(home[0].is_dir && home[0].name == "skills");
    }

    #[test_case]
    fn bare_keys_appear_at_root() {
        let ents = entries(&["skills/1/body.md", "/agent/1/x"]);
        let root = list_dir("/", &ents, true);
        assert!(root.iter().any(|e| e.name == "skills" && e.is_dir));
        assert!(root.iter().any(|e| e.name == "agent" && e.is_dir));
    }
}
