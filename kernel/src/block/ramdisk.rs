//! A RAM-backed [`BlockDevice`]: a `Vec<u8>` treated as an array of sectors.
//! It is always available (no hardware), fully deterministic, and is what the
//! test suite mounts the filesystem on. It does not survive a reboot -- that
//! is the virtio disk's job -- but it exercises the exact same filesystem code
//! path, including a mount/unmount/remount round-trip that proves data really
//! lives in the device's blocks rather than in filesystem memory.

use super::{BlockDevice, BlockError, BLOCK_SIZE};
use alloc::vec;
use alloc::vec::Vec;

pub struct RamDisk {
    blocks: Vec<u8>,
    count: u64,
}

impl RamDisk {
    /// A zeroed RAM disk of `block_count` 512-byte blocks.
    pub fn new(block_count: u64) -> Self {
        Self { blocks: vec![0u8; block_count as usize * BLOCK_SIZE], count: block_count }
    }

    fn range(&self, index: u64) -> Result<core::ops::Range<usize>, BlockError> {
        if index >= self.count {
            return Err(BlockError::OutOfRange);
        }
        let start = index as usize * BLOCK_SIZE;
        Ok(start..start + BLOCK_SIZE)
    }
}

impl BlockDevice for RamDisk {
    fn block_count(&self) -> u64 {
        self.count
    }

    fn read_block(&mut self, index: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        if buf.len() != BLOCK_SIZE {
            return Err(BlockError::BadBufferLen);
        }
        let r = self.range(index)?;
        buf.copy_from_slice(&self.blocks[r]);
        Ok(())
    }

    fn write_block(&mut self, index: u64, buf: &[u8]) -> Result<(), BlockError> {
        if buf.len() != BLOCK_SIZE {
            return Err(BlockError::BadBufferLen);
        }
        let r = self.range(index)?;
        self.blocks[r].copy_from_slice(buf);
        Ok(())
    }
}
