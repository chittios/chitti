//! A persistent, ext4-backed key/value store for `synapse::fs` — so agent
//! runtime writes survive reboots on the installed system.
//!
//! The small set of agent files is kept in an in-memory cache. Persistence
//! prefers the live [`super::ext4_rw::Ext4Rw`] volume (allocate/free, real
//! directories) so a single mutation is O(file). Fallbacks, in order:
//!
//! 1. **Live RW** ([`Ext4Rw`]) — hierarchical paths, create/delete/grow/shrink
//! 2. **Same-size data rewrite** — when this session still holds an
//!    [`Ext4Layout`] from a full format (legacy path)
//! 3. **Full format** ([`Ext4Writer`]) — last resort / empty volume bootstrap
//!
//! Legacy volumes stored synapse keys as percent-encoded single-component
//! root names (`%2F` for `/`). Mount migrates those into real directories once.

use crate::block::ext4::{Ext4Writer, FileSpec};
use crate::block::ext4_read::Ext4Reader;
use crate::block::ext4_rw::Ext4Rw;
use crate::block::volcrypto::CryptoPart;
use crate::block::DiskDevice;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

/// Active volume encryption (AES-XTS over the payload after the C4VE header).
#[derive(Clone, Copy)]
struct CryptoState {
    key: [u8; 32],
    hdr_sectors: u64,
}

/// synapse::fs keys are flat strings that routinely contain `/`
/// (e.g. `/sessions/5/cmp/26`, `skills/1/body.md`), but `/` is the path separator
/// and is illegal inside an ext4 directory-entry name. Percent-encode `/` (and
/// `%` itself, so the mapping is reversible) into a legal single-component
/// filename — used by the full-format fallback, legacy migration, and
/// `/install`, which seeds a freshly formatted data partition with the live
/// store's contents and therefore has to write exactly the names a later
/// [`Ext4Store::mount`] will decode.
pub fn key_encode(key: &str) -> String {
    let mut out = String::with_capacity(key.len());
    for c in key.chars() {
        match c {
            '%' => out.push_str("%25"),
            '/' => out.push_str("%2F"),
            _ => out.push(c),
        }
    }
    out
}

/// Decode a store-encoded ext4 dir-entry name back to its synapse key
/// (`%2F` -> `/`, `%25` -> `%`). Public so `/ls` can display store keys.
pub fn key_decode(name: &str) -> String {
    let b = name.as_bytes();
    let mut out = String::with_capacity(name.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() + 1 && i + 3 <= b.len() {
            match &name[i..i + 3] {
                "%2F" | "%2f" => {
                    out.push('/');
                    i += 3;
                    continue;
                }
                "%25" => {
                    out.push('%');
                    i += 3;
                    continue;
                }
                _ => {}
            }
        }
        out.push(b[i] as char);
        i += 1;
    }
    out
}

fn canon_key(path: &str) -> String {
    if path.starts_with('/') {
        String::from(path)
    } else {
        format!("/{path}")
    }
}

pub struct Ext4Store {
    /// Global disk index ([`crate::block::probe_disk_nth`]), re-opened on each
    /// sync. Holding a live [`DiskDevice`] monopolised AHCI ports and left
    /// `/disks` empty after the boot mount (re-bringup of the same port failed
    /// or corrupted the store's DMA). NVMe is already a shared controller; the
    /// index model is correct for every transport.
    disk: usize,
    start: u64,
    count: u64,
    crypto: Option<CryptoState>,
    cache: BTreeMap<String, Vec<u8>>,
    /// While set, mutations only touch `cache`; see [`Ext4Store::begin_batch`].
    defer: bool,
    /// A mutation happened while deferring, so the flush has work to do.
    dirty: bool,
    /// Block placement from the last successful full format, for the same-size
    /// incremental fallback.
    layout: Option<crate::block::ext4::Ext4Layout>,
    /// Keys written/updated since the last successful sync.
    changed: BTreeSet<String>,
    /// Keys deleted since the last successful sync (not present in `cache`).
    deleted: BTreeSet<String>,
}

impl Ext4Store {
    /// Mount a plain (unencrypted) ext4 data partition on global disk `disk`.
    pub fn mount(disk: usize, start: u64, count: u64) -> Option<Ext4Store> {
        Self::mount_inner(disk, start, count, None)
    }

    /// Mount a C4VE-encrypted ext4 payload (after successful [`crate::block::volcrypto::unlock`]).
    pub fn mount_encrypted(
        disk: usize,
        start: u64,
        count: u64,
        key: [u8; 32],
        hdr_sectors: u64,
    ) -> Option<Ext4Store> {
        Self::mount_inner(disk, start, count, Some(CryptoState { key, hdr_sectors }))
    }

    fn mount_inner(
        disk: usize,
        start: u64,
        count: u64,
        crypto: Option<CryptoState>,
    ) -> Option<Ext4Store> {
        let mut dev = crate::block::probe_disk_nth(disk)?;
        let mut cache = BTreeMap::new();
        let mut opened = false;
        {
            let mut part = Self::open_part(&mut dev, start, count, crypto);
            // Live volume: migrate legacy `%2F` root names, then walk the tree.
            // An empty freshly-formatted data partition is a successful open —
            // do NOT require Ext4Reader after this (it used to fail-close empty
            // volumes into memfs with `?` on the RO reader).
            if let Ok(mut vol) = Ext4Rw::open(&mut part) {
                opened = true;
                if let Ok(root) = vol.readdir("/") {
                    for (name, is_dir) in root {
                        if is_dir {
                            continue;
                        }
                        let legacy = name.contains("%2F")
                            || name.contains("%2f")
                            || name.contains("%25");
                        if !legacy {
                            continue;
                        }
                        if let Ok(data) = vol.read(&name) {
                            let key = key_decode(&name);
                            if vol.write(&key, &data).is_ok() {
                                let _ = vol.unlink(&name);
                                crate::ktrace::log_fmt(format_args!(
                                    "ext4_store: migrated legacy key {} -> {}",
                                    name, key
                                ));
                            }
                        }
                    }
                }
                if let Ok(files) = vol.list_files_recursive("/") {
                    for path in files {
                        if let Ok(data) = vol.read(&path) {
                            cache.insert(canon_key(&path), data);
                        }
                    }
                }
            }
            // Fallback when Ext4Rw cannot open (foreign geometry): flat RO walk.
            if !opened {
                let mut part = Self::open_part(&mut dev, start, count, crypto);
                let mut r = Ext4Reader::open(&mut part)?;
                opened = true;
                for (name, _ino, is_dir) in r.list_root() {
                    if is_dir {
                        continue;
                    }
                    if let Some(sz) = r.file_size(&name) {
                        let mut buf = vec![0u8; sz as usize];
                        let n = r.read_root_file(&name, &mut buf).unwrap_or(0);
                        buf.truncate(n);
                        cache.insert(key_decode(&name), buf);
                    }
                }
            }
        }
        if !opened {
            return None;
        }
        // Drop `dev` before returning so later `/disks` / VFS probes can open
        // the same controller without fighting a held AHCI port.
        drop(dev);
        crate::ktrace::log_fmt(format_args!(
            "ext4_store: mounted disk {} lba {} ({} sectors), {} file(s) recovered{}",
            disk,
            start,
            count,
            cache.len(),
            if crypto.is_some() { " (encrypted)" } else { "" }
        ));
        Some(Ext4Store {
            disk,
            start,
            count,
            crypto,
            cache,
            defer: false,
            dirty: false,
            layout: None,
            changed: BTreeSet::new(),
            deleted: BTreeSet::new(),
        })
    }

    fn open_part(
        dev: &mut DiskDevice,
        start: u64,
        count: u64,
        crypto: Option<CryptoState>,
    ) -> CryptoPart<'_, DiskDevice> {
        match crypto {
            Some(c) => CryptoPart::encrypted(dev, start, count, c.hdr_sectors, c.key),
            None => CryptoPart::plain(dev, start, count),
        }
    }

    /// Open the data partition on a freshly probed disk for one sync pass.
    fn with_part<R>(&mut self, f: impl FnOnce(&mut CryptoPart<'_, DiskDevice>) -> R) -> Option<R> {
        let mut dev = crate::block::probe_disk_nth(self.disk)?;
        let mut part = Self::open_part(&mut dev, self.start, self.count, self.crypto);
        Some(f(&mut part))
    }

    /// Persist pending mutations. Prefers live RW; falls back to same-size
    /// rewrite then full format.
    fn sync(&mut self) {
        if self.sync_live() {
            return;
        }
        if self.sync_incremental() {
            return;
        }
        self.sync_full();
    }

    /// Apply creates/updates/deletes through [`Ext4Rw`]. Returns false when the
    /// volume cannot be opened or a mutation fails (caller falls back).
    fn sync_live(&mut self) -> bool {
        let changed: Vec<String> = self.changed.iter().cloned().collect();
        let deleted: Vec<String> = self.deleted.iter().cloned().collect();
        if changed.is_empty() && deleted.is_empty() {
            return true;
        }
        // Snapshot write payloads before borrowing the device for IO.
        let writes: Vec<(String, Vec<u8>)> = changed
            .iter()
            .filter_map(|name| self.cache.get(name).map(|d| (name.clone(), d.clone())))
            .collect();
        let deleted_c = deleted.clone();
        let ok = self.with_part(|part| {
            let mut vol = match Ext4Rw::open(part) {
                Ok(v) => v,
                Err(e) => {
                    crate::ktrace::log_fmt(format_args!(
                        "ext4_store: live RW open failed ({e:?}); trying fallbacks"
                    ));
                    return false;
                }
            };
            for name in &deleted_c {
                if let Err(e) = vol.unlink(name) {
                    if e != crate::block::ext4_rw::Ext4RwError::NotFound {
                        crate::ktrace::log_fmt(format_args!(
                            "ext4_store: live unlink {name}: {e:?}"
                        ));
                        return false;
                    }
                }
                let enc = key_encode(name);
                if enc != *name {
                    let _ = vol.unlink(&enc);
                }
            }
            for (name, data) in &writes {
                if let Err(e) = vol.write(name, data) {
                    crate::ktrace::log_fmt(format_args!(
                        "ext4_store: live write {name}: {e:?} -- falling back"
                    ));
                    return false;
                }
            }
            true
        });
        if ok != Some(true) {
            if ok.is_none() {
                crate::ktrace::log_fmt(format_args!(
                    "ext4_store: disk {} gone during live sync",
                    self.disk
                ));
            }
            return false;
        }
        self.changed.clear();
        self.deleted.clear();
        // A live write invalidates any remembered full-format layout (block
        // placement is no longer the sequential format order).
        self.layout = None;
        crate::ktrace::log_fmt(format_args!(
            "ext4_store: live sync ({} written, {} deleted)",
            changed.len(),
            deleted.len()
        ));
        true
    }

    /// Same-name/same-size content rewrite using a remembered format layout.
    fn sync_incremental(&mut self) -> bool {
        if !self.deleted.is_empty() {
            return false;
        }
        let Some(layout) = self.layout.clone() else {
            return false;
        };
        let want: Vec<(String, usize)> =
            self.cache.iter().map(|(k, v)| (key_encode(k), v.len())).collect();
        if want != layout.signature() {
            return false;
        }
        let changed: Vec<String> = self.changed.iter().cloned().collect();
        for name in &changed {
            let enc = key_encode(name);
            let Some(blocks) = layout.blocks_of(&enc) else {
                return false;
            };
            let Some(data) = self.cache.get(name) else {
                return false;
            };
            let data = data.clone();
            let blocks = blocks.to_vec();
            let wrote = self.with_part(|part| {
                crate::block::ext4::write_file_blocks(part, &blocks, &data)
            });
            match wrote {
                Some(Ok(())) => {}
                Some(Err(e)) => {
                    crate::ktrace::log_fmt(format_args!(
                        "ext4_store: incremental write failed: {:?} -- falling back to a full format",
                        e
                    ));
                    return false;
                }
                None => {
                    crate::ktrace::log_fmt(format_args!(
                        "ext4_store: disk {} gone during incremental sync",
                        self.disk
                    ));
                    return false;
                }
            }
        }
        self.changed.clear();
        crate::ktrace::log_fmt(format_args!(
            "ext4_store: incremental sync ({} file(s) rewritten, {} untouched)",
            changed.len(),
            layout.files.len().saturating_sub(changed.len())
        ));
        true
    }

    fn sync_full(&mut self) {
        // Own all bytes before opening the partition (borrow checker).
        let owned: Vec<(String, Vec<u8>)> = self
            .cache
            .iter()
            .map(|(k, v)| (key_encode(k), v.clone()))
            .collect();
        let files: Vec<FileSpec> = owned
            .iter()
            .map(|(n, d)| FileSpec {
                name: n.as_str(),
                data: d.as_slice(),
            })
            .collect();
        let result = self.with_part(|part| Ext4Writer::format_with_layout(part, &files));
        match result {
            Some(Ok(layout)) => {
                self.layout = Some(layout);
                self.changed.clear();
                self.deleted.clear();
            }
            Some(Err(e)) => {
                self.layout = None;
                crate::ktrace::log_fmt(format_args!("ext4_store: sync failed: {:?}", e));
            }
            None => {
                self.layout = None;
                crate::ktrace::log_fmt(format_args!(
                    "ext4_store: disk {} gone during full sync",
                    self.disk
                ));
            }
        }
    }

    /// Stop syncing after every mutation; the caller must call [`Self::end_batch`].
    ///
    /// Boot agent install writes ~120 files; batching turns that into one flush
    /// (live RW applies each file once, or a single full format on fallback).
    pub fn begin_batch(&mut self) {
        self.defer = true;
    }

    /// Resume immediate syncing, flushing once if anything changed.
    pub fn end_batch(&mut self) {
        self.defer = false;
        if self.dirty {
            self.sync();
            self.dirty = false;
        }
    }

    pub fn write(&mut self, name: &str, data: &[u8]) {
        if self.cache.get(name).is_some_and(|old| old.as_slice() == data) {
            return;
        }
        self.cache.insert(String::from(name), data.to_vec());
        self.changed.insert(String::from(name));
        self.deleted.remove(name);
        self.touched();
    }

    fn touched(&mut self) {
        if self.defer {
            self.dirty = true;
        } else {
            self.sync();
        }
    }

    pub fn read(&self, name: &str) -> Option<Vec<u8>> {
        self.cache.get(name).cloned()
    }
    pub fn exists(&self, name: &str) -> bool {
        self.cache.contains_key(name)
    }
    pub fn list(&self) -> Vec<String> {
        self.cache.keys().cloned().collect()
    }
    pub fn delete(&mut self, name: &str) -> bool {
        let removed = self.cache.remove(name).is_some();
        if removed {
            self.changed.remove(name);
            self.deleted.insert(String::from(name));
            self.touched();
        }
        removed
    }
}
