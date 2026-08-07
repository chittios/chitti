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

/// Soft metadata for store keys (mtime / owner). Not a security boundary —
/// Synapse path scope is. Absent means "never written under tracked meta"
/// (defaults: uid 0, mode 0o644, mtime 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FileMeta {
    pub uid: u32,
    pub mode: u16,
    pub mtime: u32,
    pub ctime: u32,
}

impl Default for FileMeta {
    fn default() -> Self {
        FileMeta {
            uid: 0,
            mode: 0o644,
            mtime: 0,
            ctime: 0,
        }
    }
}

static META: Locked<BTreeMap<String, FileMeta>> = Locked::new(BTreeMap::new());

fn wall_secs() -> u32 {
    let t = crate::clock::now_unix();
    if t < 0 {
        0
    } else {
        t as u32
    }
}

fn touch_meta(path: &str) {
    let key = vpath::normalize(path);
    META.with(|m| {
        let mut meta = m.get(&key).copied().unwrap_or_default();
        let t = wall_secs();
        meta.mtime = t;
        meta.ctime = t;
        if meta.mode == 0 {
            meta.mode = 0o644;
        }
        m.insert(key, meta);
    });
}

fn drop_meta(path: &str) {
    let key = vpath::normalize(path);
    META.with(|m| {
        m.remove(&key);
    });
}

/// Metadata for a store key, if recorded.
pub fn meta(path: &str) -> Option<FileMeta> {
    if credential_refused(path) {
        return None;
    }
    let key = vpath::normalize(path);
    META.with(|m| m.get(&key).copied())
}

/// Set owner (agent id). Cosmetic — does not grant authority.
pub fn chown(path: &str, uid: u32) -> Result<(), &'static str> {
    if credential_refused(path) {
        return Err(CREDENTIAL_REFUSAL);
    }
    let key = vpath::normalize(path);
    if !exists(&key) {
        return Err("no such file");
    }
    META.with(|m| {
        let mut meta = m.get(&key).copied().unwrap_or_default();
        meta.uid = uid;
        meta.ctime = wall_secs();
        m.insert(key, meta);
    });
    Ok(())
}

/// Set permission bits (low 12). Cosmetic for listings.
pub fn chmod(path: &str, mode_bits: u16) -> Result<(), &'static str> {
    if credential_refused(path) {
        return Err(CREDENTIAL_REFUSAL);
    }
    let key = vpath::normalize(path);
    if !exists(&key) {
        return Err("no such file");
    }
    META.with(|m| {
        let mut meta = m.get(&key).copied().unwrap_or_default();
        meta.mode = mode_bits & 0x0fff;
        meta.ctime = wall_secs();
        m.insert(key, meta);
    });
    Ok(())
}

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

/// Integrity tag per stored object: the provenance of whatever last wrote it.
///
/// A parallel map rather than a field on the value, deliberately: `Backend` also
/// has ext4 and disk variants, and threading a tag through them would mean a
/// format change for a property that only matters while the object is being
/// reasoned about. Absent means "not written under a tracked justification",
/// which is read as trusted -- the kernel and the boot-time installers write
/// most of the store.
static TAINT: Locked<BTreeMap<String, crate::security::Provenance>> = Locked::new(BTreeMap::new());

/// Write, recording the provenance of the justification that authorised it.
///
/// This is what lets a later destructive call ask "is the thing I am about to
/// delete something an injection put here?" instead of only "was anything
/// untrusted in the context?".
///
/// The tag **joins** with whatever was there rather than replacing it: a file
/// written under a trusted justification and later edited under an injection
/// holds attacker-chosen bytes, so it is tainted from that point on. Replacing
/// would let a single trusted overwrite launder the object -- and an overwrite
/// is exactly what an agent does after reading a poisoned document.
///
/// The path is normalised first, because [`write`] normalises its key and a tag
/// filed under the raw string would never be found again.
pub fn write_tagged(path: &str, contents: &[u8], prov: crate::security::Provenance) {
    // `write` would refuse the credential record anyway; returning here as well
    // stops a taint tag being filed for a file that was never created.
    if credential_refused(path) {
        return;
    }
    write(path, contents);
    let key = vpath::normalize(path);
    TAINT.with(|m| {
        let joined = match m.get(&key) {
            Some(prev) => prev.join(prov),
            None => prov,
        };
        m.insert(key, joined);
    });
}

/// The integrity tag of a stored object, if one was recorded.
///
/// Normalises, for the same reason [`write_tagged`] does: the caller here is a
/// gate holding a model-supplied argument, and `/a/../b` must find `/b`'s tag.
pub fn provenance(path: &str) -> Option<crate::security::Provenance> {
    let key = vpath::normalize(path);
    TAINT.with(|m| m.get(&key).copied())
}

/// Whether this object was last written under untrusted justification.
pub fn is_tainted(path: &str) -> bool {
    provenance(path) == Some(crate::security::Provenance::UntrustedIngested)
}

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

/// Whether agent writes go to the ext4 data partition (survives reboot).
///
/// `false` only on the live ISO/image (or when the data volume failed to open)
/// where the backend is pure memfs. On an installed permanent disk this must
/// be `true` after [`mount_ext4`].
pub fn is_durable() -> bool {
    STORE.with(|b| matches!(b, Backend::Ext4(_)))
}

/// Human label for the current store backend (`"ext4"` or `"memfs"`).
pub fn backend_name() -> &'static str {
    if is_durable() {
        "ext4"
    } else {
        "memfs"
    }
}

/// Batch a group of writes into a single on-disk flush.
///
/// The ext4 backend applies mutations through live RW (or, on fallback, one full
/// format). Installing the boot agent roster (~120 files) without a batch still
/// works but costs one disk round-trip per file; wrapping the group is one flush.
/// Anything writing many files in one go should still batch.
///
/// Reads inside a batch are correct — the backend serves them from its
/// authoritative in-memory cache; only the on-disk copy lags until [`end_batch`].
/// A no-op on the memory backend, which has nothing to flush.
pub fn begin_batch() {
    STORE.with(|b| match b {
        Backend::Memory(_) => {}
        Backend::Ext4(s) => s.begin_batch(),
    });
}

/// End a [`begin_batch`] group, flushing once if anything changed.
pub fn end_batch() {
    STORE.with(|b| match b {
        Backend::Memory(_) => {}
        Backend::Ext4(s) => s.end_batch(),
    });
}

// ── The login credential record ──────────────────────────────────────────
//
// The Synapse executor denies the credential path at Gate 4, which is the
// architecturally correct statement of the policy — but it is **not** what
// enforces it. `/cat`, `/rm`, `/cp`, `/mv`, `/touch`, `/grep` and `/glob` are
// `ToolBinding::Shell` tools that reach this module *directly* through
// `shell::fs`, never entering the executor at all, and so do the editor and the
// download path. So the refusal has to live here, at the store facade every one
// of them shares.
//
// Everything is refused — read included. The record is a salt plus a PBKDF2
// digest, i.e. an offline cracking target, and deleting it would remove the gate
// entirely. `exists` is deliberately left open: `/passwd status` needs it and it
// discloses nothing that `list`'s absence would not.

/// What a refused **mutation** reports.
///
/// Only the `Result`-returning entry points can say this; the `Option`-returning
/// readers (`read`, `meta`, `size_of`) have no channel for a reason, so their
/// callers render "not found". That asymmetry is fine and not worth plumbing
/// away: the path is a compile-time constant, so an attacker who can call `/cat`
/// on it already knows it exists. The properties that matter are that the bytes
/// never come back and the record cannot be changed — which is what
/// `no_shell_file_command_can_read_or_remove_the_credential_record` asserts.
pub const CREDENTIAL_REFUSAL: &str = "refused: the login credential record is not reachable from the filesystem";

/// Set while `auth::store` is legitimately touching the record. The only way
/// through [`credential_refused`].
static CREDENTIAL_ACCESS: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// RAII permission to touch the credential record, held only by
/// [`crate::auth::store`]. Restores the previous value on drop rather than
/// clearing unconditionally, so a nested access (a save that reads first) cannot
/// drop the outer guard's permission.
pub(crate) struct CredentialAccess(bool);

impl CredentialAccess {
    pub(crate) fn new() -> CredentialAccess {
        CredentialAccess(CREDENTIAL_ACCESS.swap(true, core::sync::atomic::Ordering::Relaxed))
    }
}

impl Drop for CredentialAccess {
    fn drop(&mut self) {
        CREDENTIAL_ACCESS.store(self.0, core::sync::atomic::Ordering::Relaxed);
    }
}

/// Whether this store operation must be refused because it names the login
/// credential record and no [`CredentialAccess`] is held.
fn credential_refused(path: &str) -> bool {
    !CREDENTIAL_ACCESS.load(core::sync::atomic::Ordering::Relaxed) && crate::auth::is_credential_path(path)
}

/// Create or replace the file at `path` with `contents`.
/// Paths are normalised (`..` / `//` collapsed) so keys stay canonical.
pub fn write(path: &str, contents: &[u8]) {
    if credential_refused(path) {
        return;
    }
    let path = vpath::normalize(path);
    STORE.with(|b| match b {
        Backend::Memory(s) => {
            s.insert(path.clone(), contents.to_vec());
        }
        Backend::Ext4(s) => s.write(&path, contents),
    });
    touch_meta(&path);
}

/// Read the file at `path`, or `None` if it does not exist.
pub fn read(path: &str) -> Option<Vec<u8>> {
    if credential_refused(path) {
        return None;
    }
    let path = vpath::normalize(path);
    STORE.with(|b| match b {
        Backend::Memory(s) => s.get(&path).cloned(),
        Backend::Ext4(s) => s.read(&path),
    })
}

/// Whether a file exists at `path`.
pub fn exists(path: &str) -> bool {
    let path = vpath::normalize(path);
    STORE.with(|b| match b {
        Backend::Memory(s) => s.contains_key(&path),
        Backend::Ext4(s) => s.exists(&path),
    })
}

/// All file paths, sorted, so `list` output is deterministic run-to-run.
///
/// The credential record is filtered out: `list` feeds `glob`, `grep` and the
/// tool router's `readable_paths`, so leaving it in would let an agent find it
/// (and, with `grep`, oracle its contents a byte at a time) without ever calling
/// `read`.
pub fn list() -> Vec<String> {
    let all: Vec<String> = STORE.with(|b| match b {
        Backend::Memory(s) => s.keys().cloned().collect(),
        Backend::Ext4(s) => s.list(),
    });
    if CREDENTIAL_ACCESS.load(core::sync::atomic::Ordering::Relaxed) {
        return all;
    }
    all.into_iter().filter(|p| !crate::auth::is_credential_path(p)).collect()
}

/// Delete the file at `path`. Returns whether a file was actually removed.
/// **Destructive / irreversible** -- the reason `mem_fs_delete` is gated on
/// provenance by the Synapse taint gate (Phase 6).
pub fn delete(path: &str) -> bool {
    if credential_refused(path) {
        return false;
    }
    let path = vpath::normalize(path);
    // Drop the integrity tag with the object. Keeping it would make a *new*
    // file created at the same path inherit the deleted one's taint, which
    // reads as a laundering defence and is really just a stale key.
    TAINT.with(|m| m.remove(&path));
    drop_meta(&path);
    STORE.with(|b| match b {
        Backend::Memory(s) => s.remove(&path).is_some(),
        Backend::Ext4(s) => s.delete(&path),
    })
}

/// Byte size of a file key, or `None` if missing.
pub fn size_of(path: &str) -> Option<usize> {
    if credential_refused(path) {
        return None;
    }
    let path = vpath::normalize(path);
    STORE.with(|b| match b {
        Backend::Memory(s) => s.get(&path).map(|v| v.len()),
        Backend::Ext4(s) => s.read(&path).map(|v| v.len()),
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
    if credential_refused(path) {
        return Err(CREDENTIAL_REFUSAL);
    }
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
    if credential_refused(path) {
        return Err(CREDENTIAL_REFUSAL);
    }
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
    // Both ends: copying *from* the record would duplicate the verifier to a
    // readable path, and copying *onto* it would replace the password.
    if credential_refused(src) || credential_refused(dst) {
        return Err(CREDENTIAL_REFUSAL);
    }
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
    // Both ends: renaming the record *away* removes the gate as surely as
    // deleting it, and renaming something *onto* it replaces the password.
    if credential_refused(src) || credential_refused(dst) {
        return Err(CREDENTIAL_REFUSAL);
    }
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
            // Preserve soft meta across the rename when possible.
            let old_meta = meta(&src);
            write(&dst, &data);
            if let Some(m) = old_meta {
                META.with(|map| {
                    map.insert(dst.clone(), m);
                });
            }
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
            for (k, data, mapped) in &pairs {
                if let Some(p) = vpath::parent(mapped) {
                    if p != "/" && classify(&p).is_none() {
                        let _ = mkdir(&p, true);
                    }
                }
                let old_meta = meta(k);
                write(mapped, data);
                if let Some(m) = old_meta {
                    META.with(|map| {
                        map.insert(mapped.clone(), m);
                    });
                }
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
///
/// A recursive remove of an *ancestor* (`rm -r /configs/core`) is covered too,
/// and by construction rather than by a second check: the tree comes from
/// [`list`], which filters the credential record out, and [`delete`] guards it
/// again anyway.
pub fn remove(path: &str, recursive: bool) -> Result<usize, &'static str> {
    if credential_refused(path) {
        return Err(CREDENTIAL_REFUSAL);
    }
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

    /// The object-provenance half of the value-granular experiment: a write
    /// under an untrusted justification marks the object, a trusted one does
    /// not, and the mark survives a later trusted overwrite.
    ///
    /// This test exists because the mechanism shipped with **no producer** --
    /// `write_tagged` had zero callers, so `is_tainted` was constant `false`
    /// and the E3b measurement priced only its other half. Nothing failed,
    /// because an uninvoked gate refuses nothing. Same shape as the laundering
    /// census: the way to catch a missing producer is to assert the tag
    /// arrives, not to assert a call was refused.
    #[test_case]
    fn a_write_records_the_justification_that_authorised_it() {
        use crate::security::Provenance;

        write_tagged("/fs_test_prov/clean", b"a", Provenance::UserTyped);
        assert_eq!(provenance("/fs_test_prov/clean"), Some(Provenance::UserTyped));
        assert!(!is_tainted("/fs_test_prov/clean"));

        write_tagged("/fs_test_prov/dirty", b"a", Provenance::UntrustedIngested);
        assert!(is_tainted("/fs_test_prov/dirty"));

        // Join, not replace: a trusted overwrite does not launder the object.
        write_tagged("/fs_test_prov/dirty", b"b", Provenance::UserTyped);
        assert!(is_tainted("/fs_test_prov/dirty"), "a trusted overwrite laundered a tainted object");

        // Normalisation agrees between writer and reader.
        assert!(is_tainted("/fs_test_prov/sub/../dirty"));

        // Deleting drops the tag, so a fresh file at the path starts clean.
        delete("/fs_test_prov/dirty");
        assert_eq!(provenance("/fs_test_prov/dirty"), None);
        write_tagged("/fs_test_prov/dirty", b"c", Provenance::UserTyped);
        assert!(!is_tainted("/fs_test_prov/dirty"));

        // An untracked write leaves no tag -- absent reads as trusted, which is
        // what the boot-time installers rely on.
        write("/fs_test_prov/untracked", b"a");
        assert_eq!(provenance("/fs_test_prov/untracked"), None);

        let _ = delete("/fs_test_prov/clean");
        let _ = delete("/fs_test_prov/dirty");
        let _ = delete("/fs_test_prov/untracked");
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

