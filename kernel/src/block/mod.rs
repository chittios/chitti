//! Block devices (`CHITTI_OS_HANDOFF.md` Phase 7 stretch: "block-device +
//! real FS"). A [`BlockDevice`] is the narrow, sector-oriented interface a
//! filesystem sits on: fixed-size blocks, read one, write one. It abstracts
//! over the backing store so the same filesystem (`crate::fs`) runs unchanged
//! on a RAM disk (deterministic, used by the test suite) or a real virtio-blk
//! disk (persistent across reboots, used at boot).

pub mod ext4;
pub mod ext4_read;
#[cfg(target_arch = "x86_64")]
pub mod ext4_store;
pub mod fat;
pub mod gpt;
pub mod ramdisk;
#[cfg(target_arch = "x86_64")]
pub mod virtio;

/// Every block device here uses the classic 512-byte sector.
pub const BLOCK_SIZE: usize = 512;

/// A view of a sub-range of another block device — a single partition. Block
/// `i` of the partition maps to block `start + i` of the underlying device, so
/// a filesystem (e.g. SimpleFS) can be created/mounted within a partition.
pub struct Partition<'a, D: BlockDevice> {
    dev: &'a mut D,
    start: u64,
    count: u64,
}

impl<'a, D: BlockDevice> Partition<'a, D> {
    pub fn new(dev: &'a mut D, start: u64, count: u64) -> Self {
        Self { dev, start, count }
    }
}

impl<D: BlockDevice> BlockDevice for Partition<'_, D> {
    fn block_count(&self) -> u64 {
        self.count
    }
    fn read_block(&mut self, index: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        if index >= self.count {
            return Err(BlockError::OutOfRange);
        }
        self.dev.read_block(self.start + index, buf)
    }
    fn write_block(&mut self, index: u64, buf: &[u8]) -> Result<(), BlockError> {
        if index >= self.count {
            return Err(BlockError::OutOfRange);
        }
        self.dev.write_block(self.start + index, buf)
    }
}

/// A fixed-block-size storage device. Reads and writes operate on whole
/// blocks; `buf` must be exactly [`BLOCK_SIZE`] bytes. Both take `&mut self`
/// because a real device (virtio) mutates queue state even to read.
pub trait BlockDevice {
    /// Number of addressable blocks.
    fn block_count(&self) -> u64;

    /// Read block `index` into `buf` (which must be `BLOCK_SIZE` bytes).
    fn read_block(&mut self, index: u64, buf: &mut [u8]) -> Result<(), BlockError>;

    /// Write `buf` (which must be `BLOCK_SIZE` bytes) to block `index`.
    fn write_block(&mut self, index: u64, buf: &[u8]) -> Result<(), BlockError>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockError {
    /// Block index past the end of the device.
    OutOfRange,
    /// A caller passed a buffer that was not exactly `BLOCK_SIZE`.
    BadBufferLen,
    /// The underlying device failed the request.
    DeviceError,
}
