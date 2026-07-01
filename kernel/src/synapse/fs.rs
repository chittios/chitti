//! The Synapse in-memory file store (`CHITTI_OS_HANDOFF.md` Phase 4): the
//! "disk" the `mem_fs_read` / `mem_fs_write` / `list` primitives act on. It
//! is a flat path -> bytes map guarded by the kernel `Locked` lock, so a
//! primitive's mutation is observable to a later read -- the observable side
//! effect the phase's acceptance test checks for.
//!
//! This is deliberately below the determinism boundary: it holds no
//! ambient-authority API of its own. Nothing here checks capabilities;
//! reaching these functions at all is gated upstream by the executor's
//! capability check, and they are only ever called from a validated call.
//! (Phase 5 backs this with the persistent two-tier memory store.)

use crate::mm::Locked;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

static STORE: Locked<BTreeMap<String, Vec<u8>>> = Locked::new(BTreeMap::new());

/// Create or replace the file at `path` with `contents`.
pub fn write(path: &str, contents: &[u8]) {
    STORE.with(|s| {
        s.insert(String::from(path), contents.to_vec());
    });
}

/// Read the file at `path`, or `None` if it does not exist.
pub fn read(path: &str) -> Option<Vec<u8>> {
    STORE.with(|s| s.get(path).cloned())
}

/// Whether a file exists at `path`.
pub fn exists(path: &str) -> bool {
    STORE.with(|s| s.contains_key(path))
}

/// All file paths, sorted (the `BTreeMap` already keeps them ordered), so
/// `list` output is deterministic run-to-run.
pub fn list() -> Vec<String> {
    STORE.with(|s| s.keys().cloned().collect())
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
}
