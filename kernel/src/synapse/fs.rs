//! The Synapse in-memory file store (`CHITTI_OS_HANDOFF.md` Phase 4): the
//! "disk" the `mem_fs_read` / `mem_fs_write` / `list` primitives act on. It
//! is a flat path -> bytes map guarded by the kernel `Locked` lock, so a
//! primitive's mutation is observable to a later read -- the observable side
//! effect the phase's acceptance test checks for.
//!
//! Paths are hierarchical *by convention* (`/agent/1/SOUL.md`); the backend
//! is still flat. [`list_dir`], [`mkdir`], [`copy`], [`rename`], and
//! [`remove`] synthesise a Linux-like directory tree over those keys (see
//! [`super::vpath`]).
//!
//! This is deliberately below the determinism boundary: it holds no
//! ambient-authority API of its own. Nothing here checks capabilities;
//! reaching these functions at all is gated upstream by the executor's
//! capability check, and they are only ever called from a validated call.
//! (Phase 5 backs this with the persistent two-tier memory store.)

use super::vpath::{self, DirEntry, EntryClass};
use crate::mm::Locked;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

/// The store backend. The default is a pure in-memory map — deterministic, used
/// by the test suite and the live (non-installed) ISO where nothing persists.
/// On an installed system, [`mount_ext4`] swaps in an `Ext4Store` so every agent
/// write is durably persisted to the ext4 data partition and survives a reboot
/// (`CHITTI_OS_HANDOFF.md` Phase 5 two-tier memory, made durable).
enum Backend {
    Memory(BTreeMap<String, Vec<u8>>),
    Ext4(crate::block::ext4_store::Ext4Store),
}

static STORE: Locked<Backend> = Locked::new(Backend::Memory(BTreeMap::new()));

/// Adopt an ext4-backed store as the persistent backend. Any files already
/// written to the in-memory backend are migrated into it (and thus persisted),
/// so a boot sequence that wrote before the disk was mounted keeps its state.
pub fn mount_ext4(mut store: crate::block::ext4_store::Ext4Store) {
    STORE.with(|b| {
        if let Backend::Memory(m) = b {
            for (k, v) in m.iter() {
                store.write(k, v);
            }
        }
        *b = Backend::Ext4(store);
    });
}

/// Create or replace the file at `path` with `contents`.
pub fn write(path: &str, contents: &[u8]) {
    STORE.with(|b| match b {
        Backend::Memory(s) => {
            s.insert(String::from(path), contents.to_vec());
        }
        Backend::Ext4(s) => s.write(path, contents),
    });
}

/// Read the file at `path`, or `None` if it does not exist.
pub fn read(path: &str) -> Option<Vec<u8>> {
    STORE.with(|b| match b {
        Backend::Memory(s) => s.get(path).cloned(),
        Backend::Ext4(s) => s.read(path),
    })
}

/// Whether a file exists at `path`.
pub fn exists(path: &str) -> bool {
    STORE.with(|b| match b {
        Backend::Memory(s) => s.contains_key(path),
        Backend::Ext4(s) => s.exists(path),
    })
}

/// All file paths, sorted, so `list` output is deterministic run-to-run.
pub fn list() -> Vec<String> {
    STORE.with(|b| match b {
        Backend::Memory(s) => s.keys().cloned().collect(),
        Backend::Ext4(s) => s.list(),
    })
}

/// Delete the file at `path`. Returns whether a file was actually removed.
/// **Destructive / irreversible** -- the reason `mem_fs_delete` is gated on
/// provenance by the Synapse taint gate (Phase 6).
pub fn delete(path: &str) -> bool {
    STORE.with(|b| match b {
        Backend::Memory(s) => s.remove(path).is_some(),
        Backend::Ext4(s) => s.delete(path),
    })
}

/// Byte size of a file key, or `None` if missing.
pub fn size_of(path: &str) -> Option<usize> {
    STORE.with(|b| match b {
        Backend::Memory(s) => s.get(path).map(|v| v.len()),
        Backend::Ext4(s) => s.read(path).map(|v| v.len()),
    })
}

/// All store keys with their sizes (for hierarchical listing).
fn list_with_sizes() -> Vec<(String, usize)> {
    STORE.with(|b| match b {
        Backend::Memory(s) => s.iter().map(|(k, v)| (k.clone(), v.len())).collect(),
        Backend::Ext4(s) => s
            .list()
            .into_iter()
            .map(|k| {
                let sz = s.read(&k).map(|v| v.len()).unwrap_or(0);
                (k, sz)
            })
            .collect(),
    })
}

/// Linux-like immediate children of `dir` over the flat store.
pub fn list_dir(dir: &str) -> Vec<DirEntry> {
    let entries = list_with_sizes();
    vpath::list_dir(dir, &entries, true)
}

/// Classify `path` as file, directory, or missing.
pub fn classify(path: &str) -> Option<EntryClass> {
    let keys = list();
    vpath::classify(path, &keys)
}

/// Whether `path` is a virtual directory (has children or is `/`).
pub fn is_dir(path: &str) -> bool {
    matches!(classify(path), Some(EntryClass::Dir))
}

/// Whether `path` is an exact file key.
pub fn is_file(path: &str) -> bool {
    matches!(classify(path), Some(EntryClass::File))
}

/// Create a directory by writing an empty `.keep` marker (idempotent).
/// With `parents`, create every missing ancestor the same way.
pub fn mkdir(path: &str, parents: bool) -> Result<(), &'static str> {
    let path = vpath::normalize(path);
    if path == "/" {
        return Ok(());
    }
    if is_file(&path) {
        return Err("file exists");
    }
    if !parents {
        if let Some(p) = vpath::parent(&path) {
            if p != "/" && classify(&p).is_none() {
                return Err("no such directory");
            }
        }
    } else {
        // Ensure each ancestor exists as a dir (marker).
        let mut cur = String::from("/");
        let rest = path.trim_start_matches('/');
        for seg in rest.split('/') {
            if seg.is_empty() {
                continue;
            }
            cur = if cur == "/" {
                alloc::format!("/{seg}")
            } else {
                alloc::format!("{cur}/{seg}")
            };
            if is_file(&cur) {
                return Err("file exists in path");
            }
            let keep = alloc::format!("{cur}/.keep");
            if !exists(&keep) && !is_dir(&cur) {
                write(&keep, b"");
            }
        }
        return Ok(());
    }
    let keep = alloc::format!("{path}/.keep");
    if !exists(&keep) {
        write(&keep, b"");
    }
    Ok(())
}

/// Create an empty file (or update mtime-equivalent by rewriting).
pub fn touch(path: &str) -> Result<(), &'static str> {
    let path = vpath::normalize(path);
    if path == "/" {
        return Err("is a directory");
    }
    if is_dir(&path) {
        return Err("is a directory");
    }
    if let Some(p) = vpath::parent(&path) {
        if p != "/" && classify(&p).is_none() {
            // Auto-create parent chain for convenience (touch often implies).
            mkdir(&p, true)?;
        }
    }
    if !exists(&path) {
        write(&path, b"");
    } else {
        // Re-write existing bytes to bump durable store.
        if let Some(bytes) = read(&path) {
            write(&path, &bytes);
        }
    }
    Ok(())
}

/// Copy a file, or a directory tree when `recursive`.
pub fn copy(src: &str, dst: &str, recursive: bool) -> Result<usize, &'static str> {
    let src = vpath::normalize(src);
    let dst_in = vpath::normalize(dst);
    let keys = list();
    let src_class = vpath::classify(&src, &keys).ok_or("no such file or directory")?;
    let dst_is_dir = matches!(vpath::classify(&dst_in, &keys), Some(EntryClass::Dir));
    let dst = vpath::resolve_dest(&src, &dst_in, dst_is_dir);

    match src_class {
        EntryClass::File => {
            if is_dir(&dst) {
                return Err("cannot overwrite directory");
            }
            let data = read(&src).ok_or("no such file or directory")?;
            if let Some(p) = vpath::parent(&dst) {
                if p != "/" && classify(&p).is_none() {
                    mkdir(&p, true)?;
                }
            }
            write(&dst, &data);
            Ok(1)
        }
        EntryClass::Dir => {
            if !recursive {
                return Err("omitting directory (use -r)");
            }
            if dst == src || dst.starts_with(&alloc::format!("{src}/")) {
                return Err("cannot copy directory into itself");
            }
            let under = vpath::keys_under(&src, &keys, true);
            if under.is_empty() {
                // Empty dir: just create destination marker.
                mkdir(&dst, true)?;
                return Ok(0);
            }
            let mut n = 0usize;
            for k in under {
                let Some(mapped) = vpath::remap_under(&src, &dst, &k) else {
                    continue;
                };
                if let Some(data) = read(&k) {
                    if let Some(p) = vpath::parent(&mapped) {
                        if p != "/" && classify(&p).is_none() {
                            let _ = mkdir(&p, true);
                        }
                    }
                    write(&mapped, &data);
                    n += 1;
                }
            }
            Ok(n)
        }
    }
}

/// Rename/move a file or directory tree.
pub fn rename(src: &str, dst: &str) -> Result<usize, &'static str> {
    let src = vpath::normalize(src);
    let dst_in = vpath::normalize(dst);
    let keys = list();
    let src_class = vpath::classify(&src, &keys).ok_or("no such file or directory")?;
    let dst_is_dir = matches!(vpath::classify(&dst_in, &keys), Some(EntryClass::Dir));
    let dst = vpath::resolve_dest(&src, &dst_in, dst_is_dir);

    if dst == src {
        return Ok(0);
    }
    if dst.starts_with(&alloc::format!("{src}/")) {
        return Err("cannot move directory into itself");
    }

    match src_class {
        EntryClass::File => {
            if is_dir(&dst) {
                return Err("cannot overwrite directory");
            }
            let data = read(&src).ok_or("no such file or directory")?;
            if let Some(p) = vpath::parent(&dst) {
                if p != "/" && classify(&p).is_none() {
                    mkdir(&p, true)?;
                }
            }
            write(&dst, &data);
            delete(&src);
            Ok(1)
        }
        EntryClass::Dir => {
            let under = vpath::keys_under(&src, &keys, true);
            let mut n = 0usize;
            // Write all destinations first, then delete sources (safer if
            // rewrite-on-sync backend fails mid-way — still best-effort).
            let mut pairs: Vec<(String, Vec<u8>, String)> = Vec::new();
            for k in &under {
                let Some(mapped) = vpath::remap_under(&src, &dst, k) else {
                    continue;
                };
                if let Some(data) = read(k) {
                    pairs.push((k.clone(), data, mapped));
                }
            }
            if pairs.is_empty() {
                mkdir(&dst, true)?;
            }
            for (_, data, mapped) in &pairs {
                if let Some(p) = vpath::parent(mapped) {
                    if p != "/" && classify(&p).is_none() {
                        let _ = mkdir(&p, true);
                    }
                }
                write(mapped, data);
                n += 1;
            }
            for (k, _, _) in &pairs {
                delete(k);
            }
            // Drop empty-dir self key if any.
            delete(&src);
            Ok(n)
        }
    }
}

/// Remove a file, or a directory tree when `recursive`.
pub fn remove(path: &str, recursive: bool) -> Result<usize, &'static str> {
    let path = vpath::normalize(path);
    if path == "/" {
        return Err("refusing to remove /");
    }
    let keys = list();
    match vpath::classify(&path, &keys) {
        None => Err("no such file or directory"),
        Some(EntryClass::File) => {
            if delete(&path) {
                Ok(1)
            } else {
                Err("no such file or directory")
            }
        }
        Some(EntryClass::Dir) => {
            if !recursive {
                return Err("is a directory (use -r)");
            }
            let under = vpath::keys_under(&path, &keys, true);
            let mut n = 0usize;
            for k in under {
                if delete(&k) {
                    n += 1;
                }
            }
            Ok(n)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn write_then_read_roundtrips() {
        write("fs_test_a", b"hello");
        assert_eq!(read("fs_test_a").as_deref(), Some(&b"hello"[..]));
        // Overwrite replaces contents.
        write("fs_test_a", b"world");
        assert_eq!(read("fs_test_a").as_deref(), Some(&b"world"[..]));
        assert!(exists("fs_test_a"));
        assert!(!exists("fs_test_missing"));
    }

    #[test_case]
    fn list_is_sorted_and_contains_written_paths() {
        write("fs_test_zeta", b"1");
        write("fs_test_alpha", b"2");
        let all = list();
        // Our two keys appear, and the whole listing is sorted.
        assert!(all.iter().any(|p| p == "fs_test_zeta"));
        assert!(all.iter().any(|p| p == "fs_test_alpha"));
        assert!(all.windows(2).all(|w| w[0] <= w[1]), "listing not sorted: {all:?}");
    }

    #[test_case]
    fn hierarchical_ls_mkdir_cp_mv_rm() {
        // Isolate under a unique prefix so other tests' keys don't pollute.
        let base = "/fs_hier_test";
        let _ = remove(base, true);

        assert!(mkdir(&alloc::format!("{base}/a/b"), true).is_ok());
        write(&alloc::format!("{base}/a/b/note.txt"), b"hello");
        write(&alloc::format!("{base}/a/readme"), b"r");

        let top = list_dir(base);
        assert!(top.iter().any(|e| e.name == "a" && e.is_dir));
        let a = list_dir(&alloc::format!("{base}/a"));
        assert!(a.iter().any(|e| e.name == "b" && e.is_dir));
        assert!(a.iter().any(|e| e.name == "readme" && !e.is_dir));
        // Immediate children only — not nested note.txt.
        assert!(!a.iter().any(|e| e.name == "note.txt"));

        assert_eq!(copy(&alloc::format!("{base}/a/b/note.txt"), &alloc::format!("{base}/a/copy.txt"), false), Ok(1));
        assert_eq!(read(&alloc::format!("{base}/a/copy.txt")).as_deref(), Some(&b"hello"[..]));

        assert!(copy(&alloc::format!("{base}/a/b"), &alloc::format!("{base}/a/b2"), true).is_ok());
        assert_eq!(
            read(&alloc::format!("{base}/a/b2/note.txt")).as_deref(),
            Some(&b"hello"[..])
        );

        assert!(rename(&alloc::format!("{base}/a/copy.txt"), &alloc::format!("{base}/a/moved.txt")).is_ok());
        assert!(!exists(&alloc::format!("{base}/a/copy.txt")));
        assert!(exists(&alloc::format!("{base}/a/moved.txt")));

        assert!(remove(&alloc::format!("{base}/a/b2"), true).is_ok());
        assert!(!exists(&alloc::format!("{base}/a/b2/note.txt")));

        let _ = remove(base, true);
    }
}

