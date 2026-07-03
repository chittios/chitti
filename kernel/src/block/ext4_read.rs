//! A read-only **ext4/ext2 reader** — the counterpart to `block::ext4` — so the
//! installed system can pull files (the model) off its ext4 OS partition at
//! runtime. Parses the superblock, block-group descriptors, inodes, the root
//! directory, and file data via either **block maps** (12 direct + single/
//! double indirect — what our writer produces) or **extents** (so it also reads
//! a real `mke2fs` ext4). File reads stream a block at a time into a caller
//! buffer, so the model never needs a second full-size copy in the heap.

use crate::block::{BlockDevice, BlockError, BLOCK_SIZE};
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

const ROOT_INO: u64 = 2;
const EXT4_EXTENTS_FL: u32 = 0x0008_0000;
const EXTENT_MAGIC: u16 = 0xf30a;

fn le16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn le32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

/// A parsed inode (the fields we need).
pub struct Inode {
    pub mode: u16,
    pub size: u64,
    pub flags: u32,
    pub iblock: [u8; 60], // raw i_block area (block map or extent tree root)
}

pub struct Ext4Reader<'d, D: BlockDevice> {
    dev: &'d mut D,
    pub block_size: usize,
    inode_size: usize,
    inodes_per_group: u32,
    gdt_lba_block: u64, // fs-block index of the GDT
    desc_size: usize,
}

impl<'d, D: BlockDevice> Ext4Reader<'d, D> {
    /// Open + validate an ext filesystem on `dev`.
    pub fn open(dev: &'d mut D) -> Option<Ext4Reader<'d, D>> {
        // Superblock is at byte offset 1024. Read the two 512-B sectors holding it.
        let mut sb = [0u8; 1024];
        let mut s0 = [0u8; BLOCK_SIZE];
        dev.read_block(2, &mut s0).ok()?; // for 4K blocks the SB is in block 0 at 1024 => sector 2
        sb[..512].copy_from_slice(&s0);
        dev.read_block(3, &mut s0).ok()?;
        sb[512..].copy_from_slice(&s0);
        if le16(&sb, 0x38) != 0xef53 {
            return None;
        }
        let block_size = 1024usize << le32(&sb, 0x18);
        let inodes_per_group = le32(&sb, 0x28);
        let first_data_block = le32(&sb, 0x14);
        let inode_size = if le32(&sb, 0x4c) >= 1 { le16(&sb, 0x58) as usize } else { 128 };
        // 64bit feature grows the group descriptor to s_desc_size.
        let incompat = le32(&sb, 0x60);
        let desc_size = if incompat & 0x80 != 0 { le16(&sb, 0xfe) as usize } else { 32 };
        Some(Ext4Reader {
            dev,
            block_size,
            inode_size,
            inodes_per_group,
            gdt_lba_block: first_data_block as u64 + 1,
            desc_size,
        })
    }

    /// Read one filesystem block into `buf` (must be `block_size`).
    fn read_fs_block(&mut self, fsblk: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        let spb = (self.block_size / BLOCK_SIZE) as u64;
        for s in 0..spb {
            self.dev.read_block(fsblk * spb + s, &mut buf[s as usize * BLOCK_SIZE..(s as usize + 1) * BLOCK_SIZE])?;
        }
        Ok(())
    }

    fn read_inode(&mut self, ino: u64) -> Option<Inode> {
        let group = ((ino - 1) / self.inodes_per_group as u64) as u64;
        let index = ((ino - 1) % self.inodes_per_group as u64) as usize;
        // Group descriptor -> inode table block.
        let desc_byte = self.gdt_lba_block as u64 * self.block_size as u64 + group * self.desc_size as u64;
        let mut blk = vec![0u8; self.block_size];
        self.read_fs_block(desc_byte / self.block_size as u64, &mut blk).ok()?;
        let doff = (desc_byte % self.block_size as u64) as usize;
        let itable_lo = le32(&blk, doff + 8) as u64;
        let itable_hi = if self.desc_size >= 64 { le32(&blk, doff + 0x28) as u64 } else { 0 };
        let itable = itable_lo | (itable_hi << 32);
        // Read the inode.
        let byte = itable * self.block_size as u64 + (index * self.inode_size) as u64;
        self.read_fs_block(byte / self.block_size as u64, &mut blk).ok()?;
        let o = (byte % self.block_size as u64) as usize;
        let mode = le16(&blk, o);
        let size = le32(&blk, o + 4) as u64 | ((le32(&blk, o + 108) as u64) << 32);
        let flags = le32(&blk, o + 0x20);
        let mut iblock = [0u8; 60];
        iblock.copy_from_slice(&blk[o + 40..o + 100]);
        Some(Inode { mode, size, flags, iblock })
    }

    /// Map a file logical block to a physical fs block (block-map or extents).
    fn map_block(&mut self, inode: &Inode, lblk: u64) -> Option<u64> {
        if inode.flags & EXT4_EXTENTS_FL != 0 {
            self.map_extent(&inode.iblock.to_vec(), lblk, 0)
        } else {
            self.map_block_ptr(inode, lblk)
        }
    }

    fn map_block_ptr(&mut self, inode: &Inode, lblk: u64) -> Option<u64> {
        let ib = |i: usize| le32(&inode.iblock, i * 4) as u64;
        let ppb = (self.block_size / 4) as u64;
        if lblk < 12 {
            return Some(ib(lblk as usize));
        }
        let mut buf = vec![0u8; self.block_size];
        let l = lblk - 12;
        if l < ppb {
            self.read_fs_block(ib(12), &mut buf).ok()?;
            return Some(le32(&buf, (l * 4) as usize) as u64);
        }
        let l2 = l - ppb;
        if l2 < ppb * ppb {
            self.read_fs_block(ib(13), &mut buf).ok()?;
            let si = le32(&buf, ((l2 / ppb) * 4) as usize) as u64;
            self.read_fs_block(si, &mut buf).ok()?;
            return Some(le32(&buf, ((l2 % ppb) * 4) as usize) as u64);
        }
        None // triple-indirect not needed at our sizes
    }

    /// Resolve `lblk` through an extent tree node held in `node` (60-byte inode
    /// root, or a `block_size` interior/leaf block).
    fn map_extent(&mut self, node: &[u8], lblk: u64, _depth_guard: u32) -> Option<u64> {
        if le16(node, 0) != EXTENT_MAGIC {
            return None;
        }
        let entries = le16(node, 2) as usize;
        let depth = le16(node, 6);
        if depth == 0 {
            // Leaf: ext4_extent entries (12 bytes) after the 12-byte header.
            for i in 0..entries {
                let e = 12 + i * 12;
                let ee_block = le32(node, e) as u64;
                let ee_len = (le16(node, e + 4) & 0x7fff) as u64;
                let start = (le16(node, e + 6) as u64) << 32 | le32(node, e + 8) as u64;
                if lblk >= ee_block && lblk < ee_block + ee_len {
                    return Some(start + (lblk - ee_block));
                }
            }
            None
        } else {
            // Interior: ext4_extent_idx entries; find the child covering lblk.
            let mut child = None;
            for i in 0..entries {
                let e = 12 + i * 12;
                let ei_block = le32(node, e) as u64;
                let leaf = (le16(node, e + 8) as u64) << 32 | le32(node, e + 4) as u64;
                if lblk >= ei_block {
                    child = Some(leaf);
                }
            }
            let child = child?;
            let mut buf = vec![0u8; self.block_size];
            self.read_fs_block(child, &mut buf).ok()?;
            self.map_extent(&buf, lblk, _depth_guard + 1)
        }
    }

    /// List the root directory: `(name, inode, is_dir)`.
    pub fn list_root(&mut self) -> Vec<(String, u64, bool)> {
        let mut out = Vec::new();
        let Some(root) = self.read_inode(ROOT_INO) else { return out };
        let nblocks = (root.size as usize).div_ceil(self.block_size);
        let mut buf = vec![0u8; self.block_size];
        for lb in 0..nblocks as u64 {
            let Some(pb) = self.map_block(&root, lb) else { break };
            if pb == 0 || self.read_fs_block(pb, &mut buf).is_err() {
                break;
            }
            let mut off = 0usize;
            while off + 8 <= self.block_size {
                let ino = le32(&buf, off) as u64;
                let rec_len = le16(&buf, off + 4) as usize;
                let name_len = buf[off + 6] as usize;
                let ftype = buf[off + 7];
                if rec_len < 8 {
                    break;
                }
                if ino != 0 && name_len > 0 && off + 8 + name_len <= self.block_size {
                    let name = String::from_utf8_lossy(&buf[off + 8..off + 8 + name_len]).into_owned();
                    if name != "." && name != ".." {
                        out.push((name, ino, ftype == 2));
                    }
                }
                off += rec_len;
            }
        }
        out
    }

    /// Look up `name` in the root directory, returning its inode number.
    pub fn lookup_root(&mut self, name: &str) -> Option<u64> {
        self.list_root().into_iter().find(|(n, _, _)| n == name).map(|(_, ino, _)| ino)
    }

    /// The size of root file `name`.
    pub fn file_size(&mut self, name: &str) -> Option<u64> {
        let ino = self.lookup_root(name)?;
        Some(self.read_inode(ino)?.size)
    }

    /// Read root file `name` into `dst` (up to `dst.len()`), returning the bytes
    /// read. Streams block by block, so `dst` can be a large frame buffer.
    pub fn read_root_file(&mut self, name: &str, dst: &mut [u8]) -> Option<usize> {
        let ino = self.lookup_root(name)?;
        let inode = self.read_inode(ino)?;
        let size = (inode.size as usize).min(dst.len());
        let mut buf = vec![0u8; self.block_size];
        let mut done = 0usize;
        let mut lb = 0u64;
        while done < size {
            let pb = self.map_block(&inode, lb)?;
            if pb == 0 {
                break;
            }
            self.read_fs_block(pb, &mut buf).ok()?;
            let take = (size - done).min(self.block_size);
            dst[done..done + take].copy_from_slice(&buf[..take]);
            done += take;
            lb += 1;
        }
        Some(done)
    }
}
