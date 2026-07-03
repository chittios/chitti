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

/// The store backend. The default is a pure in-memory map — deterministic, used
/// by the test suite and the live (non-installed) ISO where nothing persists.
/// On an installed system, [`mount_ext4`] swaps in an `Ext4Store` so every agent
/// write is durably persisted to the ext4 data partition and survives a reboot
/// (`CHITTI_OS_HANDOFF.md` Phase 5 two-tier memory, made durable).
enum Backend {
    Memory(BTreeMap<String, Vec<u8>>),
    #[cfg(target_arch = "x86_64")]
    Ext4(crate::block::ext4_store::Ext4Store),
}

static STORE: Locked<Backend> = Locked::new(Backend::Memory(BTreeMap::new()));

/// Adopt an ext4-backed store as the persistent backend. Any files already
/// written to the in-memory backend are migrated into it (and thus persisted),
/// so a boot sequence that wrote before the disk was mounted keeps its state.
#[cfg(target_arch = "x86_64")]
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
        #[cfg(target_arch = "x86_64")]
        Backend::Ext4(s) => s.write(path, contents),
    });
}

/// Read the file at `path`, or `None` if it does not exist.
pub fn read(path: &str) -> Option<Vec<u8>> {
    STORE.with(|b| match b {
        Backend::Memory(s) => s.get(path).cloned(),
        #[cfg(target_arch = "x86_64")]
        Backend::Ext4(s) => s.read(path),
    })
}

/// Whether a file exists at `path`.
pub fn exists(path: &str) -> bool {
    STORE.with(|b| match b {
        Backend::Memory(s) => s.contains_key(path),
        #[cfg(target_arch = "x86_64")]
        Backend::Ext4(s) => s.exists(path),
    })
}

/// All file paths, sorted, so `list` output is deterministic run-to-run.
pub fn list() -> Vec<String> {
    STORE.with(|b| match b {
        Backend::Memory(s) => s.keys().cloned().collect(),
        #[cfg(target_arch = "x86_64")]
        Backend::Ext4(s) => s.list(),
    })
}

/// Delete the file at `path`. Returns whether a file was actually removed.
/// **Destructive / irreversible** -- the reason `mem_fs_delete` is gated on
/// provenance by the Synapse taint gate (Phase 6).
pub fn delete(path: &str) -> bool {
    STORE.with(|b| match b {
        Backend::Memory(s) => s.remove(path).is_some(),
        #[cfg(target_arch = "x86_64")]
        Backend::Ext4(s) => s.delete(path),
    })
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
