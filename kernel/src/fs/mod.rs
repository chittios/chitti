//! Filesystem support for ChittiOS.
//!
//! - [`detect`] — superblock/BPB type sniffer (partition walk, labels)
//! - [`path`] — pure path normalize + longest-prefix mount resolve
//! - [`mount`] — global mount table
//! - [`vfs`] — unified read/write/readdir over the Synapse store + mounts
//! - [`ninep`] — 9P2000.L, the protocol behind a **host shared folder**; the
//!   one filesystem here with no block device under it
//! - [`host`] — that folder as a mount, dispatched before the block path
//!
//! On-disk formats on **internal and external** disks (USB MSC included):
//! - **FAT16/32** — full RW (create/write/unlink/mkdir; 8.3 create names)
//! - **ext2/3/4** — full RW via [`crate::block::ext4_rw`] (journal when safe)
//! - **exFAT** — full RW via [`crate::block::exfat_rw`] (ASCII names on write,
//!   full UTF-16 names on read)
//! - **NTFS** — detect + mount **read-only** (writer not implemented)
//!
//! Agent durable state still goes through `synapse::fs` (ext4-backed store).

pub mod detect;
pub mod host;
pub mod mount;
pub mod ninep;
pub mod path;
pub mod vfs;

pub use detect::{FsType, Volume};
pub use mount::MountEntry;
pub use vfs::{DirEntry, FileStat, VfsError};
