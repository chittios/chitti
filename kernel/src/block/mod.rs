//! Block devices (`CHITTI_OS_HANDOFF.md` Phase 7 stretch: "block-device +
//! real FS"). A [`BlockDevice`] is the narrow, sector-oriented interface a
//! filesystem sits on: fixed-size blocks, read one, write one. It abstracts
//! over the backing store so the same filesystem (`crate::fs`) runs unchanged
//! on a RAM disk (deterministic, used by the test suite) or a real virtio-blk
//! disk (persistent across reboots, used at boot).

pub mod ramdisk;
#[cfg(target_arch = "x86_64")]
pub mod virtio;

/// Every block device here uses the classic 512-byte sector.
pub const BLOCK_SIZE: usize = 512;

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
