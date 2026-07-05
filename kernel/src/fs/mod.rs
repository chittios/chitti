//! Filesystem support. The on-disk filesystems Chitti actually uses are
//! **ext4** (the default — durable agent state, OS partition) and **FAT**
//! (UEFI ESP), read/written through `crate::block::{ext4_read, ext4_store,
//! fat_read, fat_write}`. This module hosts the shared filesystem-type
//! [`detect`]or that sniffs a volume's superblock/BPB to route it to the
//! right reader. (The old SimpleFS demo filesystem was removed — ext4 is the
//! default filesystem.)

pub mod detect;
