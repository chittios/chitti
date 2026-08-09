//! Global mount table: path prefix → (disk, LBA range, filesystem type).
//!
//! Callers bind a detected volume with [`mount`] / [`umount`]; the shell's
//! `/mount` `/umount` `/mounts` are thin wrappers. Path resolution lives in
//! [`super::path`]; I/O in [`super::vfs`].

use super::detect::{self, FsType, Volume};
use super::path::{self, MountPrefix};
use crate::mm::Locked;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// One bound volume.
#[derive(Clone, Debug)]
pub struct MountEntry {
    /// Absolute mount point (`/`, `/mnt`, …).
    pub path: String,
    /// Index into [`crate::block::probe_disk_nth`].
    pub disk: usize,
    pub start_lba: u64,
    pub sectors: u64,
    pub fs: FsType,
    pub label: Option<String>,
    /// Foreign volumes stay read-only unless explicitly granted.
    pub writable: bool,
}

static TABLE: Locked<Vec<MountEntry>> = Locked::new(Vec::new());

/// Snapshot of the table (for `/mounts` and pure resolve).
pub fn list() -> Vec<MountEntry> {
    TABLE.with(|t| t.clone())
}

/// Exact mount-point lookup.
pub fn by_path(path: &str) -> Option<MountEntry> {
    let path = path::normalize(path);
    TABLE.with(|t| t.iter().find(|m| m.path == path).cloned())
}

/// Longest-prefix resolve: `(entry, relative path inside the volume)`.
pub fn resolve(full: &str) -> Option<(MountEntry, String)> {
    let full = path::normalize(full);
    TABLE.with(|t| {
        let prefixes: Vec<MountPrefix> = t
            .iter()
            .map(|m| MountPrefix {
                path: m.path.clone(),
            })
            .collect();
        let (idx, rel) = path::resolve(&full, &prefixes)?;
        Some((t[idx].clone(), rel))
    })
}

/// Whether `path` is already an exact mount point.
pub fn is_busy(path: &str) -> bool {
    by_path(path).is_some()
}

/// First free default mount point: `/mnt`, then `/mnt2`, `/mnt3`, …
pub fn next_default_path() -> String {
    TABLE.with(|t| {
        if !t.iter().any(|x| x.path == "/mnt") {
            return "/mnt".to_string();
        }
        (2..)
            .map(|i| alloc::format!("/mnt{i}"))
            .find(|p| !t.iter().any(|x| x.path == *p))
            .unwrap()
    })
}

/// Errors from bind/unbind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MountError {
    Busy,
    NotFound,
    NoDisk,
    NoVolume,
    /// Refused to bind (e.g. unknown FS with no reader).
    Unsupported,
}

/// Bind volume `vol` of `disk` at `path` (or the next free `/mnt*`).
///
/// `writable` requests RW. Supported RW: **FAT16/32** and **ext2/3/4**.
/// **NTFS** mounts read-only (write path not implemented). Unknown FS refuses.
pub fn mount(
    disk: usize,
    vol: usize,
    path: Option<&str>,
    writable: bool,
) -> Result<MountEntry, MountError> {
    let Some(mut dev) = crate::block::probe_disk_nth(disk) else {
        return Err(MountError::NoDisk);
    };
    let vols = detect::probe(&mut dev);
    let Some(v) = vols.get(vol).cloned() else {
        return Err(MountError::NoVolume);
    };
    let path = match path {
        None | Some("") => next_default_path(),
        Some(p) => {
            let p = path::normalize(p);
            // `normalize("")` is `/` — do not treat a missing path as root.
            if p.is_empty() || p == "." {
                next_default_path()
            } else {
                p
            }
        }
    };
    if is_busy(&path) {
        return Err(MountError::Busy);
    }
    // FS types we can open at all.
    let can_rw = matches!(
        v.fs,
        FsType::Fat16 | FsType::Fat32 | FsType::Ext2 | FsType::Ext3 | FsType::Ext4 | FsType::ExFat
    );
    let can_ro = can_rw || matches!(v.fs, FsType::Ntfs);
    if !can_ro {
        return Err(MountError::Unsupported);
    }
    // NTFS: mount allowed but never writable until a writer exists.
    let wanted_rw = writable;
    let writable = writable && can_rw && v.fs != FsType::Ntfs;
    let entry = MountEntry {
        path: path.clone(),
        disk,
        start_lba: v.start_lba,
        sectors: v.sectors,
        fs: v.fs,
        label: v.label.clone(),
        writable,
    };
    TABLE.with(|t| t.push(entry.clone()));
    crate::ktrace::log_fmt(format_args!(
        "fs.mount: {} -> disk {} vol {} ({}, {} MiB, rw={})",
        entry.path,
        disk,
        vol,
        entry.fs.name(),
        entry.sectors * 512 / 1024 / 1024,
        entry.writable
    ));
    if wanted_rw && !writable && matches!(v.fs, FsType::Ntfs) {
        crate::ktrace::log_fmt(format_args!(
            "fs.mount: {} is {}; writes not implemented — mounted read-only",
            entry.fs.name(),
            entry.path
        ));
    }
    Ok(entry)
}

/// Bind a pre-detected volume (used by auto-mount at boot).
pub fn mount_entry(entry: MountEntry) -> Result<(), MountError> {
    if is_busy(&entry.path) {
        return Err(MountError::Busy);
    }
    let path = entry.path.clone();
    TABLE.with(|t| t.push(entry));
    crate::ktrace::log_fmt(format_args!("fs.mount: {path} (entry)"));
    Ok(())
}

/// Bind a **host shared folder** (9P) at `path`.
///
/// Unlike every other mount this has no block device: `disk`, `start_lba` and
/// `sectors` are meaningless and are never read, because [`crate::fs::host`]
/// intercepts the operation before the VFS reaches `probe_disk_nth`. `disk` is
/// `usize::MAX` rather than 0 deliberately — a path that *did* reach the block
/// layer then fails loudly instead of quietly operating on the boot disk.
pub fn mount_host(path: &str, tag: String) -> Result<(), MountError> {
    let path = path::normalize(path);
    if is_busy(&path) {
        return Err(MountError::Busy);
    }
    TABLE.with(|t| {
        t.push(MountEntry {
            path: path.clone(),
            disk: usize::MAX,
            start_lba: 0,
            sectors: 0,
            fs: FsType::Host,
            label: Some(tag),
            writable: true,
        })
    });
    crate::ktrace::log_fmt(format_args!("fs.mount: {path} (9P host folder)"));
    Ok(())
}

/// Remove the mount at `path`.
pub fn umount(path: &str) -> Result<(), MountError> {
    let path = path::normalize(path);
    let removed = TABLE.with(|t| {
        let before = t.len();
        t.retain(|x| x.path != path);
        before - t.len()
    });
    if removed == 0 {
        return Err(MountError::NotFound);
    }
    crate::ktrace::log_fmt(format_args!("fs.umount: {path}"));
    Ok(())
}

/// Drop mounts whose `disk` index no longer probes (USB stick yanked, etc.).
///
/// USB MSC is last in the probe order, so removing it does not renumber
/// internal disks — only mounts that named a now-missing index are removed.
/// Returns how many mounts were pruned.
pub fn prune_missing_disks() -> usize {
    let removed = TABLE.with(|t| {
        let before = t.len();
        t.retain(|m| crate::block::probe_disk_nth(m.disk).is_some());
        before - t.len()
    });
    if removed > 0 {
        crate::ktrace::log_fmt(format_args!(
            "fs.mount: pruned {removed} mount(s) whose disk is gone"
        ));
    }
    removed
}

/// A data volume suitable for durable agent state (and auto-mount at `/`).
#[derive(Clone, Debug)]
pub struct DataVolume {
    pub disk: usize,
    pub start_lba: u64,
    pub sectors: u64,
    pub fs: FsType,
    pub label: Option<String>,
    /// C4VE header present on the partition.
    pub encrypted: bool,
    /// Selected via GPT name `Chitti Data` (preferred) vs content heuristic.
    pub named: bool,
}

/// Locate the Chitti **data** partition across every disk.
///
/// Preference order:
/// 1. GPT partition named `Chitti Data` (what `/install` creates)
/// 2. First ext2/3/4 volume that is not the OS/model partition (no `*.gguf`,
///    no `chitti-kernel`, no `limine.conf` at root)
/// 3. C4VE-encrypted volumes (only as a last heuristic hit)
///
/// Used by both the synapse durable store and VFS auto-mount so they never
/// disagree about which volume is "the" data disk.
pub fn find_data_volume() -> Option<DataVolume> {
    use crate::block::ext4_read::Ext4Reader;
    use crate::block::gpt;
    use crate::block::volcrypto;
    use crate::block::Partition;

    let mut named: Option<DataVolume> = None;
    let mut heuristic: Option<DataVolume> = None;

    for disk in 0..16usize {
        let Some(mut dev) = crate::block::probe_disk_nth(disk) else {
            break;
        };

        if named.is_none() {
            if let Some((_chitti, parts)) = gpt::read(&mut dev) {
                for p in &parts {
                    if p.name != "Chitti Data" {
                        continue;
                    }
                    let start = p.first_lba;
                    let sectors = p.last_lba.saturating_sub(p.first_lba).saturating_add(1);
                    let encrypted = volcrypto::probe_encrypted(&mut dev, start).is_some();
                    // Classify FS when possible (plain ext4 after install).
                    let mut fs = FsType::Ext4;
                    let mut label = Some(String::from("Chitti Data"));
                    for v in detect::probe(&mut dev) {
                        if v.start_lba == start {
                            fs = v.fs;
                            if v.label.is_some() {
                                label = v.label.clone();
                            }
                            break;
                        }
                    }
                    named = Some(DataVolume {
                        disk,
                        start_lba: start,
                        sectors,
                        fs,
                        label,
                        encrypted,
                        named: true,
                    });
                    break;
                }
            }
        }
        if named.is_some() {
            continue;
        }

        for v in detect::probe(&mut dev) {
            if volcrypto::probe_encrypted(&mut dev, v.start_lba).is_some() {
                if heuristic.is_none() {
                    heuristic = Some(DataVolume {
                        disk,
                        start_lba: v.start_lba,
                        sectors: v.sectors,
                        fs: v.fs,
                        label: v.label.clone(),
                        encrypted: true,
                        named: false,
                    });
                }
                continue;
            }
            if !matches!(v.fs, FsType::Ext2 | FsType::Ext3 | FsType::Ext4) {
                continue;
            }
            let mut part = Partition::new(&mut dev, v.start_lba, v.sectors);
            let is_os_or_model = Ext4Reader::open(&mut part)
                .map(|mut r| {
                    r.list_root().iter().any(|(n, _, _)| {
                        n.contains(".gguf") || n == "chitti-kernel" || n == "limine.conf"
                    })
                })
                .unwrap_or(true);
            if !is_os_or_model && heuristic.is_none() {
                heuristic = Some(DataVolume {
                    disk,
                    start_lba: v.start_lba,
                    sectors: v.sectors,
                    fs: v.fs,
                    label: v.label.clone(),
                    encrypted: false,
                    named: false,
                });
            }
        }
    }
    named.or(heuristic)
}

/// Auto-mount the ext4 **data** partition at `/` (RW).
///
/// Same volume selection as the durable synapse store ([`find_data_volume`]).
/// Defaults to **writable** so `/mkdir` / `/touch` / VFS writes work on the
/// installed system without a manual `/mount … rw`. No-op if `/` is busy or no
/// data volume exists (live ISO / image).
pub fn auto_mount_data_root() -> Option<MountEntry> {
    if is_busy("/") {
        return by_path("/");
    }
    let v = find_data_volume()?;
    // Encrypted volumes need an unlock first; do not auto-bind them.
    if v.encrypted {
        crate::ktrace::log(
            "fs.mount",
            "data volume is C4VE-encrypted; use /unlock before auto-mount",
        );
        return None;
    }
    if !matches!(
        v.fs,
        FsType::Ext2 | FsType::Ext3 | FsType::Ext4 | FsType::Fat16 | FsType::Fat32
    ) {
        return None;
    }
    let entry = MountEntry {
        path: String::from("/"),
        disk: v.disk,
        start_lba: v.start_lba,
        sectors: v.sectors,
        fs: v.fs,
        label: v.label.clone(),
        writable: true, // installed data volume is RW by default
    };
    let _ = mount_entry(entry.clone());
    crate::ktrace::log_fmt(format_args!(
        "fs.mount: / -> disk {} lba {} ({}, {} MiB, rw) [auto{}]",
        v.disk,
        v.start_lba,
        v.fs.name(),
        v.sectors * 512 / 1024 / 1024,
        if v.named { ", Chitti Data" } else { "" }
    ));
    Some(entry)
}

/// Probe volumes on a disk (shell `/disks` helper).
pub fn probe_disk_volumes(disk: usize) -> Option<Vec<Volume>> {
    let mut dev = crate::block::probe_disk_nth(disk)?;
    Some(detect::probe(&mut dev))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fs::path::MountPrefix;
    use alloc::vec;

    #[test_case]
    fn resolve_uses_table_longest_prefix() {
        // Pure path logic is covered in path::tests; here we only check the
        // prefix vector shape the table would build.
        let prefixes = [
            MountPrefix {
                path: String::from("/"),
            },
            MountPrefix {
                path: String::from("/mnt"),
            },
        ];
        let (i, rel) = path::resolve("/mnt/a", &prefixes).unwrap();
        assert_eq!(i, 1);
        assert_eq!(rel, "a");
    }
}
