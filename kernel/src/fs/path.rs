//! Pure path helpers for the VFS mount table.
//!
//! No I/O: callers supply the mount-point list so unit tests exercise the
//! longest-prefix match without a disk.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Normalise a user path: absolute, collapse `/./` and `//`, resolve `..`,
/// strip a trailing slash (except for `/`).
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

/// One mount point: only the path prefix is needed for resolution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MountPrefix {
    pub path: String,
}

/// Resolve `full` against `mounts`: longest mount-point prefix wins.
///
/// Returns `(mount_index, relative_path)` where `relative_path` has no leading
/// slash (empty string means the mount root itself).
///
/// Rules:
/// - A mount at `/` matches everything (relative is the path without the
///   leading slash).
/// - A mount at `/mnt` matches `/mnt` and `/mnt/foo`, not `/mnt2`.
/// - Longer prefixes beat shorter ones (`/mnt/a` over `/mnt`).
pub fn resolve(full: &str, mounts: &[MountPrefix]) -> Option<(usize, String)> {
    let full = normalize(full);
    if mounts.is_empty() {
        return None;
    }
    let mut best: Option<(usize, usize)> = None; // (index, prefix_len)
    for (i, m) in mounts.iter().enumerate() {
        let mp = normalize(&m.path);
        let matched = if mp == "/" {
            true
        } else {
            full == mp || full.starts_with(&alloc::format!("{mp}/"))
        };
        if !matched {
            continue;
        }
        // Score by prefix length so `/mnt/data` beats `/mnt` beats `/`.
        let len = if mp == "/" { 1 } else { mp.len() };
        if best.map(|(_, l)| len > l).unwrap_or(true) {
            best = Some((i, len));
        }
    }
    let (idx, _) = best?;
    let mp = normalize(&mounts[idx].path);
    let rel = if mp == "/" {
        full.trim_start_matches('/').to_string()
    } else if full == mp {
        String::new()
    } else {
        full[mp.len()..].trim_start_matches('/').to_string()
    };
    Some((idx, rel))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;

    fn mps(paths: &[&str]) -> Vec<MountPrefix> {
        paths
            .iter()
            .map(|p| MountPrefix {
                path: (*p).to_string(),
            })
            .collect()
    }

    #[test_case]
    fn normalize_collapses_dots_and_slashes() {
        assert_eq!(normalize("/a//b/./c/"), "/a/b/c");
        assert_eq!(normalize("/a/b/../c"), "/a/c");
        assert_eq!(normalize("/"), "/");
        assert_eq!(normalize(""), "/");
    }

    #[test_case]
    fn longest_prefix_wins() {
        let m = mps(&["/", "/mnt", "/mnt/data"]);
        let (i, rel) = resolve("/mnt/data/x.bin", &m).unwrap();
        assert_eq!(m[i].path, "/mnt/data");
        assert_eq!(rel, "x.bin");
    }

    #[test_case]
    fn mnt_does_not_match_mnt2() {
        let m = mps(&["/mnt"]);
        assert!(resolve("/mnt2/foo", &m).is_none());
        let (i, rel) = resolve("/mnt/foo", &m).unwrap();
        assert_eq!(m[i].path, "/mnt");
        assert_eq!(rel, "foo");
    }

    #[test_case]
    fn root_mount_catches_everything_else() {
        let m = mps(&["/", "/mnt"]);
        let (i, rel) = resolve("/agent/1/SOUL.md", &m).unwrap();
        assert_eq!(m[i].path, "/");
        assert_eq!(rel, "agent/1/SOUL.md");
        let (j, rel2) = resolve("/mnt/x", &m).unwrap();
        assert_eq!(m[j].path, "/mnt");
        assert_eq!(rel2, "x");
    }

    #[test_case]
    fn mount_root_itself_has_empty_relative() {
        let m = mps(&["/mnt"]);
        let (_, rel) = resolve("/mnt", &m).unwrap();
        assert_eq!(rel, "");
        let (_, rel2) = resolve("/mnt/", &m).unwrap();
        assert_eq!(rel2, "");
    }

    #[test_case]
    fn no_mounts_resolves_nothing() {
        assert!(resolve("/anything", &[]).is_none());
    }
}
