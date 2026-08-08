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

/// Normalise without allocating when the path is already canonical.
///
/// The scope gate is ~68% of the authorization decision (`synapse::bench`), and
/// most of that was this function: `glob_covers` normalises **both** sides on
/// every ledger entry it examines, so a constant grant like `/agent/7/**` was
/// re-normalised on every call forever and the target -- already normalised by
/// the executor -- was normalised a second time. Nearly every real path is
/// already canonical, so the fix is to notice that rather than to skip the
/// normalisation, which is load-bearing: it is what stops `/agent/7/../../etc`
/// from being covered by a grant of `/agent/7/**`.
pub fn normalize_cow(path: &str) -> alloc::borrow::Cow<'_, str> {
    let p = path.trim();
    // Canonical: absolute, no empty/dot/dotdot segment, no trailing slash.
    let canonical = p.len() > 1
        && p.starts_with('/')
        && !p.ends_with('/')
        && !p.contains("//")
        && !p.contains("/./")
        && !p.contains("/../")
        && !p.ends_with("/.")
        && !p.ends_with("/..");
    if canonical {
        return alloc::borrow::Cow::Borrowed(p);
    }
    alloc::borrow::Cow::Owned(normalize(p))
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

/// Long listing line: type flag, mode octal, uid, mtime, size, name.
///
/// `mode` / `uid` / `mtime` are optional soft metadata (zeros if unknown).
pub fn format_long(e: &DirEntry) -> String {
    format_long_meta(e, 0, 0, 0)
}

/// A byte count in `ls -h` form: `86`, `4.6K`, `1.2M`, `3.0G`.
///
/// A fraction only below 10 of a unit, since `11.5K` is noise where `12K` is not.
/// Bytes never take a suffix and never take a fraction. Integer arithmetic
/// throughout — the kernel's listing path has no business pulling in float
/// formatting, and `(n * 10 + half) / unit` is the same rounding as
/// `round(n / unit * 10)`.
///
/// **Rounds half-up, where GNU `ls -h` rounds up.** So 4760 B reads `4.6K` here
/// and `4.7K` there. Stated because someone will eventually diff the two: this
/// is a deliberate choice (half-up is what a size *is*, nearest), not a
/// rounding bug.
pub fn human_size(n: u64) -> String {
    const UNITS: [(u64, char); 4] =
        [(1 << 30, 'G'), (1 << 20, 'M'), (1 << 10, 'K'), (1, ' ')];
    for (unit, suffix) in UNITS {
        if n < unit || unit == 1 {
            continue;
        }
        // Tenths, rounded half-up.
        let tenths = (n * 10 + unit / 2) / unit;
        // Rounding can carry into the next unit (1048000 B -> "1024.0K"); print
        // it as the unit it rounded into rather than as an out-of-range number.
        return if tenths >= 10_240 {
            alloc::format!("{}{}", tenths / 10240, next_suffix(suffix))
        } else if tenths < 100 {
            alloc::format!("{}.{}{}", tenths / 10, tenths % 10, suffix)
        } else {
            alloc::format!("{}{}", (tenths + 5) / 10, suffix)
        };
    }
    alloc::format!("{n}")
}

/// The unit above `s`, for the rounding carry in [`human_size`].
fn next_suffix(s: char) -> char {
    match s {
        'K' => 'M',
        'M' => 'G',
        _ => 'T',
    }
}

/// Like [`format_long`], with explicit soft metadata fields.
pub fn format_long_meta(e: &DirEntry, mode: u16, uid: u32, mtime: u32) -> String {
    format_long_meta_sized(e, mode, uid, mtime, false)
}

/// [`format_long_meta`], with `ls -h` sizes when `human`.
pub fn format_long_meta_sized(
    e: &DirEntry,
    mode: u16,
    uid: u32,
    mtime: u32,
    human: bool,
) -> String {
    let size = if human {
        human_size(e.size as u64)
    } else {
        alloc::format!("{}", e.size)
    };
    let t = if e.is_dir { 'd' } else { '-' };
    let name = if e.is_dir {
        alloc::format!("{}/", e.name)
    } else {
        e.name.clone()
    };
    // mtime: raw unix when non-zero (shell can pretty-print later).
    if mtime == 0 && mode == 0 && uid == 0 {
        return alloc::format!("{t} {size:>10}  {name}");
    }
    alloc::format!(
        "{t} {:04o} uid={:<4} mtime={:<10} {size:>10}  {name}",
        mode & 0o7777,
        uid,
        mtime,
    )
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
    /// `ls -h` sizes: bytes bare, one decimal below ten of a unit, none above.
    #[test_case]
    fn human_size_matches_the_ls_h_shape() {
        use super::human_size;
        assert_eq!(human_size(0), "0");
        assert_eq!(human_size(86), "86");
        assert_eq!(human_size(1023), "1023");
        assert_eq!(human_size(1024), "1.0K");
        // Half-up, not GNU's ceiling: 4760 B is 4.648 K, which GNU shows as 4.7K.
        assert_eq!(human_size(4760), "4.6K");
        assert_eq!(human_size(9617), "9.4K");
        // At ten of a unit the fraction is dropped: `11.5K` is noise, `12K` is not.
        assert_eq!(human_size(11729), "12K");
        assert_eq!(human_size(1 << 20), "1.0M");
        assert_eq!(human_size(1 << 30), "1.0G");
    }

    /// Rounding must not print a number that is out of its unit's range.
    ///
    /// 1 047 552 B is `1023.0K` before rounding and `1024.0K` after — which reads
    /// as a bug even though the arithmetic is right. It carries into the next
    /// unit instead.
    #[test_case]
    fn human_size_carries_instead_of_printing_1024_of_a_unit() {
        use super::human_size;
        assert_eq!(human_size(1_048_575), "1M");
        assert_eq!(human_size((1 << 30) - 1), "1G");
        assert_eq!(human_size((1 << 20) - 1), "1M");
    }

    /// The long listing uses those sizes only when asked, so the default output
    /// is byte-identical to before.
    #[test_case]
    fn long_listing_is_unchanged_unless_human_is_requested() {
        use super::{format_long_meta, format_long_meta_sized, DirEntry};
        let e = DirEntry { name: alloc::string::String::from("gobreaker.go"), is_dir: false, size: 9617 };
        assert_eq!(format_long_meta(&e, 0o644, 0, 1), format_long_meta_sized(&e, 0o644, 0, 1, false));
        assert!(format_long_meta_sized(&e, 0o644, 0, 1, false).contains("9617"));
        assert!(format_long_meta_sized(&e, 0o644, 0, 1, true).contains("9.4K"));
    }

    /// The fast path must be indistinguishable from the slow one.
    ///
    /// `normalize_cow` decides by inspection whether a path is already
    /// canonical and borrows if so. If that predicate is ever wrong in the
    /// permissive direction, `glob_covers` compares an unnormalised target --
    /// and a grant of `/agent/7/**` starts covering `/agent/7/../../etc/passwd`.
    /// So the two must agree on every input, especially the escaping ones.
    #[test_case]
    fn normalize_cow_agrees_with_normalize() {
        let cases = [
            "/a/b", "/", "", ".", "//a//b", "/a/./b", "/a/../b", "/a/b/", "/a/b/..",
            "/a/b/.", "/agent/7/../../etc/passwd", "a/b", "relative", "/a//", "/..",
            "/a/b//c/./d/../e", "  /a/b  ", "/a/..b", "/a/b..c",
        ];
        for c in cases {
            assert_eq!(
                normalize_cow(c).as_ref(),
                normalize(c).as_str(),
                "normalize_cow disagreed with normalize on {c:?}"
            );
        }
        // And the fast path must actually be taken for the common shape, or the
        // optimisation is dead code.
        assert!(matches!(normalize_cow("/agent/7/notes.md"), alloc::borrow::Cow::Borrowed(_)));
        assert!(matches!(normalize_cow("/a/../b"), alloc::borrow::Cow::Owned(_)));
    }

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
