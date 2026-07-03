//! SimpleFS: a small but real on-disk filesystem (`CHITTI_OS_HANDOFF.md`
//! Phase 7 stretch). It sits on any [`BlockDevice`] and stores files durably
//! in that device's blocks -- unlike the in-memory Synapse store, a file
//! written here survives an unmount/remount (and, on a virtio disk, a reboot).
//!
//! Deliberately simple, flat (no directories), and correct over clever:
//!
//! * **Block 0** is the superblock (magic, geometry).
//! * **Blocks `[1 .. data_start)`** are the inode table: a fixed number of
//!   64-byte inodes (8 per block).
//! * **Blocks `[data_start .. )`** are the data region.
//!
//! Each inode has up to `NDIRECT` direct block pointers, so a file is capped
//! at `NDIRECT * BLOCK_SIZE` bytes. Free data blocks are found by scanning the
//! live inodes' pointers -- no separate bitmap to keep consistent, which for a
//! small filesystem is simpler and impossible to desync. Every operation is
//! write-through (no cache), so on-disk state always reflects the last call.

use crate::block::{BlockDevice, BlockError, BLOCK_SIZE};
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

pub mod detect;

const MAGIC: u32 = 0x0C_11_77_F5;
/// SimpleFS superblock magic, exposed for `fs::detect` to identify our volumes.
pub const SIMPLEFS_MAGIC: u32 = MAGIC;
const VERSION: u32 = 1;
const INODE_SIZE: usize = 64;
const INODES_PER_BLOCK: usize = BLOCK_SIZE / INODE_SIZE; // 8
const NDIRECT: usize = 8;
const NAME_LEN: usize = 24;
/// Largest file SimpleFS can store (8 direct blocks * 512 bytes).
pub const MAX_FILE_SIZE: usize = NDIRECT * BLOCK_SIZE;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FsError {
    Block(BlockError),
    BadSuperblock,
    NoFreeInode,
    NoFreeBlock,
    FileTooLarge,
    NotFound,
    NameTooLong,
}

impl From<BlockError> for FsError {
    fn from(e: BlockError) -> Self {
        FsError::Block(e)
    }
}

// --- little-endian field helpers -----------------------------------------

fn rd_u32(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}
fn wr_u32(buf: &mut [u8], off: usize, v: u32) {
    buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

/// One on-disk inode, decoded. `used == 0` marks a free slot.
#[derive(Clone)]
struct Inode {
    used: u32,
    size: u32,
    name: [u8; NAME_LEN],
    blocks: [u32; NDIRECT],
}

impl Inode {
    fn empty() -> Self {
        Self { used: 0, size: 0, name: [0; NAME_LEN], blocks: [0; NDIRECT] }
    }

    fn decode(raw: &[u8]) -> Self {
        let mut name = [0u8; NAME_LEN];
        name.copy_from_slice(&raw[8..8 + NAME_LEN]);
        let mut blocks = [0u32; NDIRECT];
        for (i, b) in blocks.iter_mut().enumerate() {
            *b = rd_u32(raw, 32 + i * 4);
        }
        Self { used: rd_u32(raw, 0), size: rd_u32(raw, 4), name, blocks }
    }

    fn encode(&self, raw: &mut [u8]) {
        wr_u32(raw, 0, self.used);
        wr_u32(raw, 4, self.size);
        raw[8..8 + NAME_LEN].copy_from_slice(&self.name);
        for (i, b) in self.blocks.iter().enumerate() {
            wr_u32(raw, 32 + i * 4, *b);
        }
    }

    fn name_str(&self) -> &str {
        let end = self.name.iter().position(|&c| c == 0).unwrap_or(NAME_LEN);
        core::str::from_utf8(&self.name[..end]).unwrap_or("")
    }

    fn matches(&self, name: &str) -> bool {
        self.used != 0 && self.name_str() == name
    }
}

/// A mounted SimpleFS instance owning its backing device.
pub struct SimpleFs<D: BlockDevice> {
    dev: D,
    total_blocks: u32,
    inode_count: u32,
    inode_start: u32,
    data_start: u32,
}

impl<D: BlockDevice> SimpleFs<D> {
    /// Format `dev` with `inode_count` inodes and return the mounted fs. Wipes
    /// the superblock and inode table (data blocks are left as-is; they are
    /// only ever read after being written through an inode).
    pub fn format(mut dev: D, inode_count: u32) -> Result<Self, FsError> {
        let total_blocks = dev.block_count() as u32;
        let inode_blocks = inode_count.div_ceil(INODES_PER_BLOCK as u32);
        let inode_start = 1u32;
        let data_start = inode_start + inode_blocks;
        if data_start >= total_blocks {
            return Err(FsError::BadSuperblock);
        }

        // Superblock.
        let mut sb = [0u8; BLOCK_SIZE];
        wr_u32(&mut sb, 0, MAGIC);
        wr_u32(&mut sb, 4, VERSION);
        wr_u32(&mut sb, 8, total_blocks);
        wr_u32(&mut sb, 12, inode_count);
        wr_u32(&mut sb, 16, inode_start);
        wr_u32(&mut sb, 20, data_start);
        dev.write_block(0, &sb)?;

        // Zero the inode table (all slots free).
        let zero = [0u8; BLOCK_SIZE];
        for b in inode_start..data_start {
            dev.write_block(b as u64, &zero)?;
        }

        Ok(Self { dev, total_blocks, inode_count, inode_start, data_start })
    }

    /// Mount `dev` if it already holds a SimpleFS, otherwise format it. The
    /// boot path uses this so a fresh disk is formatted once and thereafter
    /// mounted (preserving its contents across reboots).
    pub fn mount_or_format(mut dev: D, inode_count: u32) -> Result<Self, FsError> {
        let mut sb = [0u8; BLOCK_SIZE];
        dev.read_block(0, &mut sb)?;
        if rd_u32(&sb, 0) == MAGIC && rd_u32(&sb, 4) == VERSION {
            crate::ktrace::log("fs", "mounted existing SimpleFS from disk");
            Self::mount(dev)
        } else {
            crate::ktrace::log("fs", "no SimpleFS found; formatting disk");
            Self::format(dev, inode_count)
        }
    }

    /// Mount an already-formatted device, verifying the superblock.
    pub fn mount(mut dev: D) -> Result<Self, FsError> {
        let mut sb = [0u8; BLOCK_SIZE];
        dev.read_block(0, &mut sb)?;
        if rd_u32(&sb, 0) != MAGIC || rd_u32(&sb, 4) != VERSION {
            return Err(FsError::BadSuperblock);
        }
        Ok(Self {
            dev,
            total_blocks: rd_u32(&sb, 8),
            inode_count: rd_u32(&sb, 12),
            inode_start: rd_u32(&sb, 16),
            data_start: rd_u32(&sb, 20),
        })
    }

    /// Give the backing device back (e.g. to remount it).
    pub fn unmount(self) -> D {
        self.dev
    }

    // --- inode table access ---

    fn read_inode(&mut self, i: u32) -> Result<Inode, FsError> {
        let block = self.inode_start + i / INODES_PER_BLOCK as u32;
        let off = (i as usize % INODES_PER_BLOCK) * INODE_SIZE;
        let mut buf = [0u8; BLOCK_SIZE];
        self.dev.read_block(block as u64, &mut buf)?;
        Ok(Inode::decode(&buf[off..off + INODE_SIZE]))
    }

    fn write_inode(&mut self, i: u32, inode: &Inode) -> Result<(), FsError> {
        let block = self.inode_start + i / INODES_PER_BLOCK as u32;
        let off = (i as usize % INODES_PER_BLOCK) * INODE_SIZE;
        let mut buf = [0u8; BLOCK_SIZE];
        self.dev.read_block(block as u64, &mut buf)?;
        inode.encode(&mut buf[off..off + INODE_SIZE]);
        self.dev.write_block(block as u64, &buf)?;
        Ok(())
    }

    fn find_inode(&mut self, name: &str) -> Result<Option<u32>, FsError> {
        for i in 0..self.inode_count {
            if self.read_inode(i)?.matches(name) {
                return Ok(Some(i));
            }
        }
        Ok(None)
    }

    /// Bitmap of data blocks currently referenced by some live inode, so a new
    /// allocation avoids them. `skip` (if any) excludes one inode's blocks --
    /// used when rewriting a file so its own old blocks can be reused.
    fn in_use_blocks(&mut self, skip: Option<u32>) -> Result<Vec<bool>, FsError> {
        let mut used = vec![false; self.total_blocks as usize];
        for i in 0..self.inode_count {
            if Some(i) == skip {
                continue;
            }
            let inode = self.read_inode(i)?;
            if inode.used == 0 {
                continue;
            }
            let nblocks = (inode.size as usize).div_ceil(BLOCK_SIZE);
            for b in inode.blocks.iter().take(nblocks) {
                if (*b as usize) < used.len() {
                    used[*b as usize] = true;
                }
            }
        }
        Ok(used)
    }

    // --- public file operations ---

    /// Create or overwrite `name` with `data`.
    pub fn write(&mut self, name: &str, data: &[u8]) -> Result<(), FsError> {
        if name.len() > NAME_LEN {
            return Err(FsError::NameTooLong);
        }
        if data.len() > MAX_FILE_SIZE {
            return Err(FsError::FileTooLarge);
        }

        // Reuse the inode if the file exists, else allocate a free one.
        let idx = match self.find_inode(name)? {
            Some(i) => i,
            None => (0..self.inode_count)
                .find_map(|i| match self.read_inode(i) {
                    Ok(n) if n.used == 0 => Some(Ok(i)),
                    Ok(_) => None,
                    Err(e) => Some(Err(e)),
                })
                .transpose()?
                .ok_or(FsError::NoFreeInode)?,
        };

        // Allocate fresh data blocks, avoiding everyone else's (but reusing
        // this inode's own old blocks, which we're about to overwrite).
        let nblocks = data.len().div_ceil(BLOCK_SIZE);
        let mut used = self.in_use_blocks(Some(idx))?;
        let mut inode = Inode::empty();
        inode.used = 1;
        inode.size = data.len() as u32;
        let name_bytes = name.as_bytes();
        inode.name[..name_bytes.len()].copy_from_slice(name_bytes);

        let mut next = self.data_start as usize;
        for (k, chunk) in data.chunks(BLOCK_SIZE).enumerate() {
            // Find the next free data block.
            while next < used.len() && used[next] {
                next += 1;
            }
            if next >= used.len() {
                return Err(FsError::NoFreeBlock);
            }
            used[next] = true;
            let blk = next as u32;
            inode.blocks[k] = blk;

            let mut buf = [0u8; BLOCK_SIZE];
            buf[..chunk.len()].copy_from_slice(chunk);
            self.dev.write_block(blk as u64, &buf)?;
        }
        let _ = nblocks;

        self.write_inode(idx, &inode)?;
        Ok(())
    }

    /// Read `name`'s contents, or `NotFound`.
    pub fn read(&mut self, name: &str) -> Result<Vec<u8>, FsError> {
        let idx = self.find_inode(name)?.ok_or(FsError::NotFound)?;
        let inode = self.read_inode(idx)?;
        let size = inode.size as usize;
        let mut out = Vec::with_capacity(size);
        let nblocks = size.div_ceil(BLOCK_SIZE);
        for b in inode.blocks.iter().take(nblocks) {
            let mut buf = [0u8; BLOCK_SIZE];
            self.dev.read_block(*b as u64, &mut buf)?;
            let take = (size - out.len()).min(BLOCK_SIZE);
            out.extend_from_slice(&buf[..take]);
        }
        Ok(out)
    }

    /// Whether `name` exists.
    pub fn exists(&mut self, name: &str) -> Result<bool, FsError> {
        Ok(self.find_inode(name)?.is_some())
    }

    /// Delete `name` (freeing its inode and, implicitly, its data blocks).
    pub fn delete(&mut self, name: &str) -> Result<(), FsError> {
        let idx = self.find_inode(name)?.ok_or(FsError::NotFound)?;
        self.write_inode(idx, &Inode::empty())?;
        Ok(())
    }

    /// All file names, in inode order.
    pub fn list(&mut self) -> Result<Vec<String>, FsError> {
        let mut names = Vec::new();
        for i in 0..self.inode_count {
            let inode = self.read_inode(i)?;
            if inode.used != 0 {
                names.push(String::from(inode.name_str()));
            }
        }
        Ok(names)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::ramdisk::RamDisk;

    fn fresh() -> SimpleFs<RamDisk> {
        // 256 blocks = 128 KiB; 64 inodes.
        SimpleFs::format(RamDisk::new(256), 64).unwrap()
    }

    #[test_case]
    fn format_write_read_roundtrips() {
        let mut fs = fresh();
        fs.write("greeting", b"hello world").unwrap();
        assert_eq!(fs.read("greeting").unwrap(), b"hello world");
        assert!(fs.exists("greeting").unwrap());
        assert!(!fs.exists("absent").unwrap());
    }

    #[test_case]
    fn survives_unmount_and_remount() {
        // The load-bearing filesystem property: data lives in the device's
        // blocks, so a remount from the *same* device sees it.
        let mut fs = fresh();
        fs.write("persist", b"durable bytes").unwrap();
        fs.write("second", b"another file").unwrap();
        let dev = fs.unmount();

        let mut fs2 = SimpleFs::mount(dev).unwrap();
        assert_eq!(fs2.read("persist").unwrap(), b"durable bytes");
        assert_eq!(fs2.read("second").unwrap(), b"another file");
    }

    #[test_case]
    fn overwrite_and_delete() {
        let mut fs = fresh();
        fs.write("f", b"v1").unwrap();
        fs.write("f", b"a much longer version two").unwrap();
        assert_eq!(fs.read("f").unwrap(), b"a much longer version two");
        fs.delete("f").unwrap();
        assert!(!fs.exists("f").unwrap());
        assert_eq!(fs.read("f"), Err(FsError::NotFound));
    }

    #[test_case]
    fn multi_block_file_roundtrips() {
        let mut fs = fresh();
        // 1500 bytes spans three 512-byte data blocks.
        let data: Vec<u8> = (0..1500u32).map(|i| (i % 251) as u8).collect();
        fs.write("big", &data).unwrap();
        assert_eq!(fs.read("big").unwrap(), data);
        // Too large is rejected, not silently truncated.
        let huge = vec![0u8; MAX_FILE_SIZE + 1];
        assert_eq!(fs.write("huge", &huge), Err(FsError::FileTooLarge));
    }

    #[test_case]
    fn listing_reflects_live_files() {
        let mut fs = fresh();
        fs.write("alpha", b"1").unwrap();
        fs.write("beta", b"2").unwrap();
        fs.write("gamma", b"3").unwrap();
        fs.delete("beta").unwrap();
        let mut names = fs.list().unwrap();
        names.sort();
        assert_eq!(names, alloc::vec![String::from("alpha"), String::from("gamma")]);
    }
}
