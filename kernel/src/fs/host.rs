//! **Host shared folder** — the VFS adapter over [`crate::fs::ninep`].
//!
//! A 9P mount is the first filesystem here with **no block device under it**.
//! The mount table is otherwise block-shaped (`disk` index + `start_lba` +
//! `sectors`), and every VFS operation resolves a mount and then calls
//! `probe_disk_nth`, so a host mount is distinguished by its
//! [`FsType::Host`](super::detect::FsType::Host) and dispatched here *before*
//! that happens. Its `disk`/`start_lba`/`sectors` are meaningless and never
//! read.
//!
//! Errors are mapped rather than passed through, because a Linux errno from the
//! host is not something the shell should print. The mapping is deliberately
//! narrow: anything unrecognised becomes [`VfsError::Io`] rather than being
//! guessed at.

use super::detect::FsType;
use super::mount::{self, MountEntry};
use super::vfs::{DirEntry, FileStat, VfsError};
use crate::fs::ninep::wire::P9Error;
use alloc::string::String;
use alloc::vec::Vec;

/// Where a host folder is attached by default. Chosen over `/mnt` so a shared
/// folder is never confused with a mounted disk, and so scripts can rely on it.
pub const HOST_MOUNT: &str = "/host";

/// Whether this mount is a host shared folder rather than a block volume.
pub fn is_host(mt: &MountEntry) -> bool {
    mt.fs == FsType::Host
}

/// The host mount, if one is attached.
pub fn mount_entry() -> Option<MountEntry> {
    mount::list().into_iter().find(|m| is_host(m))
}

/// Whether a host shared folder is attached and usable.
pub fn present() -> bool {
    crate::drivers::virtio_9p::present()
}

/// Map a 9P failure onto the VFS's error set.
fn map(e: P9Error) -> VfsError {
    match e {
        P9Error::Server(2) => VfsError::NotFound,   // ENOENT
        P9Error::Server(13) => VfsError::ReadOnly,  // EACCES
        P9Error::Server(20) => VfsError::NotADir,   // ENOTDIR
        P9Error::Server(21) => VfsError::NotAFile,  // EISDIR
        P9Error::Server(30) => VfsError::ReadOnly,  // EROFS
        _ => VfsError::Io,
    }
}

/// The device is attached but currently in use by another operation.
///
/// This is a real state rather than an impossible one: the session is taken out
/// of its lock for the duration of a call, so a re-entrant VFS call — a tool
/// invoked from inside a directory listing, say — finds it absent.
fn busy<T>() -> Result<T, VfsError> {
    Err(VfsError::Io)
}

/// A relative path inside the export, as 9P wants it.
fn rel_path(rel: &str) -> String {
    let mut s = String::from("/");
    s.push_str(rel.trim_start_matches('/'));
    s
}

pub fn read(rel: &str) -> Result<Vec<u8>, VfsError> {
    let p = rel_path(rel);
    match crate::drivers::virtio_9p::with_session(|s| s.read_file(&p)) {
        Some(r) => r.map_err(map),
        None => busy(),
    }
}

pub fn write(rel: &str, data: &[u8]) -> Result<(), VfsError> {
    let p = rel_path(rel);
    match crate::drivers::virtio_9p::with_session(|s| s.write_file(&p, data)) {
        Some(r) => r.map_err(map),
        None => busy(),
    }
}

pub fn mkdir(rel: &str) -> Result<(), VfsError> {
    let p = rel_path(rel);
    match crate::drivers::virtio_9p::with_session(|s| s.mkdir(&p)) {
        Some(r) => r.map_err(map),
        None => busy(),
    }
}

/// Remove a file or directory. Which one it is has to be known before asking,
/// because `Tunlinkat` takes `AT_REMOVEDIR` as a flag — so this stats first
/// rather than trying one and retrying with the other, which would turn every
/// directory removal into two round trips *and* an error in the log.
pub fn unlink(rel: &str) -> Result<(), VfsError> {
    let p = rel_path(rel);
    match crate::drivers::virtio_9p::with_session(|s| {
        let is_dir = s.getattr(&p).map(|a| a.qid.is_dir()).unwrap_or(false);
        s.unlink(&p, is_dir)
    }) {
        Some(r) => r.map_err(map),
        None => busy(),
    }
}

pub fn stat(rel: &str) -> Result<FileStat, VfsError> {
    let p = rel_path(rel);
    let a = match crate::drivers::virtio_9p::with_session(|s| s.getattr(&p)) {
        Some(r) => r.map_err(map)?,
        None => return busy(),
    };
    Ok(FileStat {
        // The host's mode already carries the ext-style type bits, which is
        // what `FileStat::mode` means here.
        mode: a.mode as u16,
        uid: a.uid,
        size: a.size,
        mtime: a.mtime as u32,
        ctime: a.ctime as u32,
        is_dir: a.qid.is_dir(),
    })
}

pub fn readdir(rel: &str) -> Result<Vec<DirEntry>, VfsError> {
    let p = rel_path(rel);
    let ents = match crate::drivers::virtio_9p::with_session(|s| s.readdir(&p)) {
        Some(r) => r.map_err(map)?,
        None => return busy(),
    };
    Ok(ents
        .into_iter()
        // `.` and `..` are the host's, not ours: every other filesystem here
        // reports a directory's contents without them, and leaking them makes
        // a recursive walk loop.
        .filter(|e| e.name != "." && e.name != "..")
        .map(|e| DirEntry {
            is_dir: e.qid.is_dir(),
            // 9P readdir carries no size; a `stat` per entry would turn one
            // listing into N round trips. `/ls` shows a size of 0 for host
            // files, and `stat` on the file itself is exact.
            size: 0,
            name: e.name,
        })
        .collect())
}

pub fn exists(rel: &str) -> bool {
    let p = rel_path(rel);
    crate::drivers::virtio_9p::with_session(|s| s.exists(&p).is_some()).unwrap_or(false)
}

/// Attach the host folder at [`HOST_MOUNT`], if the device is present.
///
/// Called once at boot. Absent hardware is the common case and is silent.
pub fn attach_at_boot() {
    let Some(tag) = crate::drivers::virtio_9p::init() else {
        return;
    };
    match mount::mount_host(HOST_MOUNT, tag.clone()) {
        Ok(()) => crate::ktrace::log_fmt(format_args!(
            "host folder: '{tag}' mounted at {HOST_MOUNT}"
        )),
        Err(e) => crate::ktrace::log_fmt(format_args!(
            "host folder: '{tag}' attached but could not mount at {HOST_MOUNT}: {e:?}"
        )),
    }
}
