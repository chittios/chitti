//! Virtual filesystem facade: one path namespace over the mount table + the
//! Synapse store.
//!
//! ## Layout
//! - **Synapse store** (`crate::synapse::fs`) — agent homes, configs, sessions.
//! - **Mounted volumes** — FAT16/32 and ext2/3/4 with optional **RW**; NTFS is
//!   detected and mountable **read-only** (list + read files).
//!
//! Callers use [`read`] / [`write`] / [`readdir`] / [`mkdir`] / [`unlink`] rather
//! than opening `ext4_*` / `fat_*` directly.

use super::detect::FsType;
use super::mount::{self, MountEntry};
use super::path;
use alloc::string::String;
use alloc::vec::Vec;

/// VFS errors surfaced to shell / tools.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VfsError {
    NotFound,
    NotAFile,
    NotADir,
    NotMounted,
    Unsupported,
    Io,
    ReadOnly,
}

/// One directory entry from [`readdir`].
#[derive(Clone, Debug)]
pub struct DirEntry {
    pub name: String,
    pub is_dir: bool,
    pub size: u64,
}

/// File metadata (stat), agent-native `uid` = creating agent or 0.
#[derive(Clone, Debug)]
pub struct FileStat {
    pub mode: u16,
    pub uid: u32,
    pub size: u64,
    pub mtime: u32,
    pub ctime: u32,
    pub is_dir: bool,
}

/// Stat a path: store soft-meta first, else writable ext4 volume inode.
pub fn stat(path: &str) -> Result<FileStat, VfsError> {
    let path = path::normalize(path);
    if crate::synapse::fs::exists(&path) || crate::synapse::fs::is_dir(&path) {
        let size = crate::synapse::fs::size_of(&path).unwrap_or(0) as u64;
        let is_dir = crate::synapse::fs::is_dir(&path);
        let m = crate::synapse::fs::meta(&path).unwrap_or_default();
        return Ok(FileStat {
            // File type bits match ext (S_IFDIR=0x4000, S_IFREG=0x8000).
            mode: if is_dir {
                0x4000 | (m.mode & 0x0fff)
            } else {
                0x8000 | (m.mode & 0x0fff)
            },
            uid: m.uid,
            size,
            mtime: m.mtime,
            ctime: m.ctime,
            is_dir,
        });
    }
    let (mt, rel) = mount::resolve(&path).ok_or(VfsError::NotMounted)?;
    // Before the ext-only gate below, which would otherwise refuse every host
    // path as Unsupported.
    if super::host::is_host(&mt) {
        return super::host::stat(if rel.is_empty() { "/" } else { &rel });
    }
    if !mt.writable || !matches!(mt.fs, FsType::Ext2 | FsType::Ext3 | FsType::Ext4) {
        return Err(VfsError::Unsupported);
    }
    let mut dev = crate::block::probe_disk_nth(mt.disk).ok_or(VfsError::Io)?;
    let mut part = crate::block::Partition::new(&mut dev, mt.start_lba, mt.sectors);
    let mut vol = crate::block::ext4_rw::Ext4Rw::open(&mut part).map_err(|_| VfsError::Io)?;
    let rel = if rel.is_empty() { "/" } else { rel.as_str() };
    let s = vol.stat(rel).map_err(|_| VfsError::NotFound)?;
    Ok(FileStat {
        mode: s.mode,
        uid: s.uid,
        size: s.size,
        mtime: s.mtime,
        ctime: s.ctime,
        is_dir: s.is_dir(),
    })
}

/// Rename within the store (synapse tree). Volume renames require a writable
/// ext4 mount and go through Ext4Rw.
pub fn rename(from: &str, to: &str) -> Result<(), VfsError> {
    let from = path::normalize(from);
    let to = path::normalize(to);
    // Prefer store when source is a store key / virtual dir.
    if crate::synapse::fs::exists(&from) || crate::synapse::fs::is_dir(&from) {
        crate::synapse::fs::rename(&from, &to).map_err(|_| VfsError::Io)?;
        return Ok(());
    }
    let (mt, rel_from) = mount::resolve(&from).ok_or(VfsError::NotMounted)?;
    if !mt.writable {
        return Err(VfsError::ReadOnly);
    }
    let (_mt2, rel_to) = mount::resolve(&to).ok_or(VfsError::NotMounted)?;
    // Same mount only (longest prefix must match).
    if mount::resolve(&from).map(|(m, _)| m.path) != mount::resolve(&to).map(|(m, _)| m.path) {
        return Err(VfsError::Unsupported);
    }
    if !matches!(mt.fs, FsType::Ext2 | FsType::Ext3 | FsType::Ext4) {
        return Err(VfsError::Unsupported);
    }
    let mut dev = crate::block::probe_disk_nth(mt.disk).ok_or(VfsError::Io)?;
    let mut part = crate::block::Partition::new(&mut dev, mt.start_lba, mt.sectors);
    let mut vol = crate::block::ext4_rw::Ext4Rw::open(&mut part).map_err(|_| VfsError::Io)?;
    vol.rename(&rel_from, &rel_to).map_err(|e| match e {
        crate::block::ext4_rw::Ext4RwError::NotFound => VfsError::NotFound,
        crate::block::ext4_rw::Ext4RwError::Exists => VfsError::NotAFile,
        crate::block::ext4_rw::Ext4RwError::NotEmpty => VfsError::NotADir,
        _ => VfsError::Io,
    })
}

/// Read a file by absolute path.
///
/// Order: Synapse store (if present) → longest-prefix mount → `NotFound`.
pub fn read(path: &str) -> Result<Vec<u8>, VfsError> {
    let path = path::normalize(path);
    if let Some(bytes) = crate::synapse::fs::read(&path) {
        return Ok(bytes);
    }
    read_mount(&path)
}

/// Read only from a mounted volume (skip the store). Used when the caller
/// already knows the path is under `/mnt…`.
pub fn read_mount(path: &str) -> Result<Vec<u8>, VfsError> {
    let path = path::normalize(path);
    let (mt, rel) = mount::resolve(&path).ok_or(VfsError::NotMounted)?;
    if rel.is_empty() {
        return Err(VfsError::NotAFile);
    }
    read_on_volume(&mt, &rel)
}

fn read_on_volume(mt: &MountEntry, rel: &str) -> Result<Vec<u8>, VfsError> {
    // A host shared folder has no disk under it, so it must be dispatched
    // before `probe_disk_nth` — which would fail on its sentinel index.
    if super::host::is_host(mt) {
        return super::host::read(rel);
    }
    let mut dev = crate::block::probe_disk_nth(mt.disk).ok_or(VfsError::Io)?;
    let mut part = crate::block::Partition::new(&mut dev, mt.start_lba, mt.sectors);
    match mt.fs {
        FsType::Fat16 | FsType::Fat32 => {
            if let Ok(mut vol) = crate::block::fat_rw::FatRw::open(&mut part) {
                return vol.read(rel).map_err(|e| match e {
                    crate::block::fat_rw::FatRwError::NotFound => VfsError::NotFound,
                    crate::block::fat_rw::FatRwError::NotAFile => VfsError::NotAFile,
                    _ => VfsError::Io,
                });
            }
            crate::block::fat_read::FatReader::open(&mut part)
                .and_then(|mut r| r.read_file(rel))
                .ok_or(VfsError::NotFound)
        }
        FsType::Ext2 | FsType::Ext3 | FsType::Ext4 => {
            // Hierarchical Ext4Rw first (works for nested paths; journal is safe).
            if let Ok(mut vol) = crate::block::ext4_rw::Ext4Rw::open(&mut part) {
                return vol.read(rel).map_err(|_| VfsError::NotFound);
            }
            // Legacy flat root-only reader.
            if rel.contains('/') {
                return Err(VfsError::NotFound);
            }
            let mut r = crate::block::ext4_read::Ext4Reader::open(&mut part).ok_or(VfsError::Io)?;
            let sz = r.file_size(rel).ok_or(VfsError::NotFound)? as usize;
            let mut buf = alloc::vec![0u8; sz];
            let n = r.read_root_file(rel, &mut buf).ok_or(VfsError::NotFound)?;
            buf.truncate(n);
            Ok(buf)
        }
        FsType::Ntfs => {
            let mut vol = crate::block::ntfs_read::NtfsReader::open(&mut part)
                .ok_or(VfsError::Unsupported)?;
            vol.read_file(rel).ok_or(VfsError::NotFound)
        }
        FsType::ExFat => Err(VfsError::Unsupported),
        _ => Err(VfsError::Unsupported),
    }
}

/// Write a file on a **writable** mount (FAT or ext). Store paths use
/// [`crate::synapse::fs::write`].
pub fn write(path: &str, data: &[u8]) -> Result<(), VfsError> {
    let path = path::normalize(path);
    // Prefer store if the key already exists or is under agent homes, or if
    // the path lives on the auto-mounted data root (same volume as Ext4Store).
    if crate::synapse::fs::exists(&path)
        || path.starts_with("/agent/")
        || path.starts_with("/configs/")
        || path.starts_with("/sessions/")
        || path.starts_with("/skills/")
        || path.starts_with("/downloads/")
    {
        crate::synapse::fs::write(&path, data);
        return Ok(());
    }
    let (mt, rel) = mount::resolve(&path).ok_or(VfsError::NotMounted)?;
    if mt.path == "/" {
        crate::synapse::fs::write(&path, data);
        return Ok(());
    }
    if rel.is_empty() {
        return Err(VfsError::NotAFile);
    }
    if !mt.writable {
        return Err(VfsError::ReadOnly);
    }
    if super::host::is_host(&mt) {
        return super::host::write(&rel, data);
    }
    let mut dev = crate::block::probe_disk_nth(mt.disk).ok_or(VfsError::Io)?;
    let mut part = crate::block::Partition::new(&mut dev, mt.start_lba, mt.sectors);
    match mt.fs {
        FsType::Fat16 | FsType::Fat32 => {
            let mut vol = crate::block::fat_rw::FatRw::open(&mut part).map_err(|_| VfsError::Io)?;
            vol.write(&rel, data).map_err(|e| match e {
                crate::block::fat_rw::FatRwError::BadName => VfsError::Unsupported,
                crate::block::fat_rw::FatRwError::Full => VfsError::Io,
                crate::block::fat_rw::FatRwError::NotAFile => VfsError::NotAFile,
                _ => VfsError::Io,
            })
        }
        FsType::Ext2 | FsType::Ext3 | FsType::Ext4 => {
            let mut vol = crate::block::ext4_rw::Ext4Rw::open(&mut part).map_err(|_| VfsError::Io)?;
            vol.write(&rel, data).map_err(|_| VfsError::Io)
        }
        FsType::Ntfs | FsType::ExFat => Err(VfsError::Unsupported), // NTFS write not implemented
        _ => Err(VfsError::Unsupported),
    }
}

/// Create a directory on a writable mount.
///
/// Paths under the auto-mounted data root (`/`) go through the durable
/// synapse store (not raw Ext4Rw), so they appear in `/ls` and stay in the
/// Ext4Store cache. Foreign mounts (`/mnt…`) use the volume writer.
pub fn mkdir(path: &str) -> Result<(), VfsError> {
    let path = path::normalize(path);
    if path.starts_with("/agent/")
        || path.starts_with("/configs/")
        || path.starts_with("/sessions/")
        || path.starts_with("/skills/")
    {
        return crate::synapse::fs::mkdir(&path, true).map_err(|_| VfsError::Io);
    }
    let (mt, rel) = mount::resolve(&path).ok_or(VfsError::NotMounted)?;
    // Data volume at `/` == synapse store. Always create via store markers.
    if mt.path == "/" {
        return crate::synapse::fs::mkdir(&path, true).map_err(|_| VfsError::Io);
    }
    if rel.is_empty() {
        return Ok(());
    }
    if !mt.writable {
        return Err(VfsError::ReadOnly);
    }
    if super::host::is_host(&mt) {
        return super::host::mkdir(&rel);
    }
    let mut dev = crate::block::probe_disk_nth(mt.disk).ok_or(VfsError::Io)?;
    let mut part = crate::block::Partition::new(&mut dev, mt.start_lba, mt.sectors);
    match mt.fs {
        FsType::Fat16 | FsType::Fat32 => {
            let mut vol = crate::block::fat_rw::FatRw::open(&mut part).map_err(|_| VfsError::Io)?;
            vol.mkdir(&rel).map_err(|e| match e {
                crate::block::fat_rw::FatRwError::Exists => VfsError::NotAFile, // already there
                crate::block::fat_rw::FatRwError::BadName => VfsError::Unsupported,
                crate::block::fat_rw::FatRwError::NotADir => VfsError::NotADir,
                _ => VfsError::Io,
            })
        }
        FsType::Ext2 | FsType::Ext3 | FsType::Ext4 => {
            let mut vol = crate::block::ext4_rw::Ext4Rw::open(&mut part).map_err(|_| VfsError::Io)?;
            vol.mkdir(&rel).map_err(|_| VfsError::Io)
        }
        _ => Err(VfsError::Unsupported),
    }
}

/// Unlink a file on a writable mount.
pub fn unlink(path: &str) -> Result<(), VfsError> {
    let path = path::normalize(path);
    if crate::synapse::fs::exists(&path) {
        if crate::synapse::fs::delete(&path) {
            return Ok(());
        }
        return Err(VfsError::NotFound);
    }
    let (mt, rel) = mount::resolve(&path).ok_or(VfsError::NotMounted)?;
    if rel.is_empty() {
        return Err(VfsError::NotAFile);
    }
    if !mt.writable {
        return Err(VfsError::ReadOnly);
    }
    if super::host::is_host(&mt) {
        return super::host::unlink(&rel);
    }
    let mut dev = crate::block::probe_disk_nth(mt.disk).ok_or(VfsError::Io)?;
    let mut part = crate::block::Partition::new(&mut dev, mt.start_lba, mt.sectors);
    match mt.fs {
        FsType::Fat16 | FsType::Fat32 => {
            let mut vol = crate::block::fat_rw::FatRw::open(&mut part).map_err(|_| VfsError::Io)?;
            vol.unlink(&rel).map_err(|e| match e {
                crate::block::fat_rw::FatRwError::NotFound => VfsError::NotFound,
                crate::block::fat_rw::FatRwError::NotAFile => VfsError::NotAFile,
                _ => VfsError::Io,
            })
        }
        FsType::Ext2 | FsType::Ext3 | FsType::Ext4 => {
            let mut vol = crate::block::ext4_rw::Ext4Rw::open(&mut part).map_err(|_| VfsError::Io)?;
            vol.unlink(&rel).map_err(|_| VfsError::NotFound)
        }
        _ => Err(VfsError::Unsupported),
    }
}

/// List a directory.
///
/// - Paths that classify as a **store** directory use the hierarchical store
///   listing (never dump percent-encoded keys).
/// - Mount roots and volume-relative dirs use the volume reader.
pub fn readdir(path: &str) -> Result<Vec<DirEntry>, VfsError> {
    let path = path::normalize(path);

    // Store directory wins when it classifies as a dir (including `/` with
    // agent trees). Empty store still classifies `/` as a dir.
    if crate::synapse::fs::is_dir(&path) {
        let entries = crate::synapse::fs::list_dir(&path);
        return Ok(entries
            .into_iter()
            .map(|e| DirEntry {
                name: e.name,
                is_dir: e.is_dir,
                size: e.size as u64,
            })
            .collect());
    }

    // Exact mount point (e.g. /mnt) → volume root listing.
    if let Some(mt) = mount::by_path(&path) {
        return readdir_volume_root(&mt);
    }

    // Path under a mount (including subdirectories).
    if let Some((mt, rel)) = mount::resolve(&path) {
        return readdir_on_volume(&mt, if rel.is_empty() { "/" } else { &rel });
    }

    Err(VfsError::NotFound)
}

fn readdir_volume_root(mt: &MountEntry) -> Result<Vec<DirEntry>, VfsError> {
    readdir_on_volume(mt, "/")
}

fn readdir_on_volume(mt: &MountEntry, rel: &str) -> Result<Vec<DirEntry>, VfsError> {
    if super::host::is_host(mt) {
        return super::host::readdir(rel);
    }
    let mut dev = crate::block::probe_disk_nth(mt.disk).ok_or(VfsError::Io)?;
    let mut part = crate::block::Partition::new(&mut dev, mt.start_lba, mt.sectors);
    match mt.fs {
        FsType::Fat16 | FsType::Fat32 => {
            let mut vol = crate::block::fat_rw::FatRw::open(&mut part).map_err(|_| VfsError::Io)?;
            let ents = vol.readdir(rel).map_err(|e| match e {
                crate::block::fat_rw::FatRwError::NotADir => VfsError::NotADir,
                crate::block::fat_rw::FatRwError::NotFound => VfsError::NotFound,
                _ => VfsError::Io,
            })?;
            Ok(ents
                .into_iter()
                .map(|(name, size, is_dir)| DirEntry {
                    name,
                    is_dir,
                    size: size as u64,
                })
                .collect())
        }
        FsType::Ext2 | FsType::Ext3 | FsType::Ext4 => {
            if let Ok(mut vol) = crate::block::ext4_rw::Ext4Rw::open(&mut part) {
                let ents = vol.readdir(rel).map_err(|_| VfsError::NotFound)?;
                return Ok(ents
                    .into_iter()
                    .map(|(name, is_dir)| DirEntry {
                        name,
                        is_dir,
                        size: 0,
                    })
                    .collect());
            }
            if rel != "/" && !rel.is_empty() {
                return Err(VfsError::Unsupported);
            }
            let mut r = crate::block::ext4_read::Ext4Reader::open(&mut part).ok_or(VfsError::Io)?;
            Ok(r.list_root()
                .into_iter()
                .map(|(name, _ino, is_dir)| {
                    let shown = crate::block::ext4_store::key_decode(&name);
                    let base = crate::synapse::vpath::basename(&shown);
                    DirEntry {
                        name: base,
                        is_dir,
                        size: 0,
                    }
                })
                .collect())
        }
        FsType::Ntfs => {
            let mut vol = crate::block::ntfs_read::NtfsReader::open(&mut part)
                .ok_or(VfsError::Unsupported)?;
            let rel = if rel == "/" { "" } else { rel };
            let ents = vol.readdir(rel).ok_or(VfsError::NotFound)?;
            Ok(ents
                .into_iter()
                .map(|(name, size, is_dir)| DirEntry {
                    name,
                    is_dir,
                    size,
                })
                .collect())
        }
        _ => Err(VfsError::Unsupported),
    }
}

/// Whether a path exists as a store file/dir or a mount file.
pub fn exists(path: &str) -> bool {
    let path = path::normalize(path);
    if crate::synapse::fs::exists(&path) || crate::synapse::fs::is_dir(&path) {
        return true;
    }
    // A host folder is asked directly, because the fallback below proves
    // existence by *reading* — which reports every directory as absent, and
    // `/cd /host/sub` depends on the difference.
    if let Some((mt, rel)) = mount::resolve(&path) {
        if super::host::is_host(&mt) {
            return super::host::exists(if rel.is_empty() { "/" } else { &rel });
        }
    }
    read_mount(&path).is_ok()
}

/// Scan every disk + volume for the first readable file named one of `names`
/// (independent of the mount table — ESP / bundled assets).
pub fn find_on_disks(names: &[&str]) -> Option<Vec<u8>> {
    for disk in 0..4usize {
        let Some(mut dev) = crate::block::probe_disk_nth(disk) else {
            continue;
        };
        for v in crate::fs::detect::probe(&mut dev) {
            let mut part = crate::block::Partition::new(&mut dev, v.start_lba, v.sectors);
            for name in names {
                let data = match v.fs {
                    FsType::Fat16 | FsType::Fat32 => {
                        crate::block::fat_read::FatReader::open(&mut part)
                            .and_then(|mut r| r.read_file(name))
                    }
                    FsType::Ext2 | FsType::Ext3 | FsType::Ext4 => {
                        crate::block::ext4_read::Ext4Reader::open(&mut part).and_then(|mut r| {
                            let sz = r.file_size(name)? as usize;
                            let mut buf = alloc::vec![0u8; sz];
                            let n = r.read_root_file(name, &mut buf)?;
                            buf.truncate(n);
                            Some(buf)
                        })
                    }
                    _ => None,
                };
                if data.is_some() {
                    return data;
                }
            }
        }
    }
    None
}
