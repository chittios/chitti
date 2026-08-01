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
        FsType::Fat16 | FsType::Fat32 | FsType::Ext2 | FsType::Ext3 | FsType::Ext4
    );
    let can_ro = can_rw || matches!(v.fs, FsType::Ntfs | FsType::ExFat);
    if !can_ro {
        return Err(MountError::Unsupported);
    }
    // NTFS/exFAT: mount allowed but never writable until a writer exists.
    let wanted_rw = writable;
    let writable = writable && can_rw && !matches!(v.fs, FsType::Ntfs | FsType::ExFat);
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
    if wanted_rw && !writable && matches!(v.fs, FsType::Ntfs | FsType::ExFat) {
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

/// Auto-mount the ext4 **data** partition at `/` (same heuristic as before):
/// first ext4 that holds neither a model (`*.gguf`) nor the OS kernel/limine.
/// No-op if none is present or `/` is already mounted.
pub fn auto_mount_data_root() -> Option<MountEntry> {
    if is_busy("/") {
        return by_path("/");
    }
    use crate::block::{ext4_read::Ext4Reader, Partition};
    for disk in 0..4usize {
        let Some(mut dev) = crate::block::probe_disk_nth(disk) else {
            continue;
        };
        let vols = detect::probe(&mut dev);
        for (vi, v) in vols.iter().enumerate() {
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
            if is_os_or_model {
                continue;
            }
            let entry = MountEntry {
                path: String::from("/"),
                disk,
                start_lba: v.start_lba,
                sectors: v.sectors,
                fs: v.fs,
                label: v.label.clone(),
                // Data partition backs the store; shell writes go through
                // synapse::fs, not raw VFS writes, so leave RO at this layer.
                writable: false,
            };
            let _ = mount_entry(entry.clone());
            crate::ktrace::log_fmt(format_args!(
                "fs.mount: / -> disk {} vol {} ({}, {} MiB) [auto]",
                disk,
                vi,
                v.fs.name(),
                v.sectors * 512 / 1024 / 1024
            ));
            return Some(entry);
        }
    }
    None
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
