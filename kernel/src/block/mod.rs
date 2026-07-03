//! Block devices (`CHITTI_OS_HANDOFF.md` Phase 7 stretch: "block-device +
//! real FS"). A [`BlockDevice`] is the narrow, sector-oriented interface a
//! filesystem sits on: fixed-size blocks, read one, write one. It abstracts
//! over the backing store so the same filesystem (`crate::fs`) runs unchanged
//! on a RAM disk (deterministic, used by the test suite) or a real virtio-blk
//! disk (persistent across reboots, used at boot).

pub mod ahci;
pub mod ext4;
pub mod ext4_read;
pub mod ext4_store;
pub mod fat;
pub mod fat_read;
pub mod gpt;
pub mod nvme;
pub mod ramdisk;
#[cfg(target_arch = "x86_64")]
pub mod virtio;

/// A DMA region: the physical address a device is programmed with, and the
/// virtual address the CPU touches the same bytes through. On aarch64 (identity
/// map / stub HHDM) `phys` derives from `virt` via `dma_to_phys`; on x86 the two
/// differ (the frame allocator gives a physical frame, reached via the HHDM).
/// This is the platform seam that lets the shared NVMe/AHCI cores allocate and
/// address DMA memory without knowing the arch — the same idea as the `xhci`
/// core's `(phys, virt)` allocator.
#[derive(Clone, Copy)]
pub struct Dma {
    pub phys: u64,
    pub virt: u64,
}

/// Allocate a physically-contiguous, zeroed, 4 KiB-aligned DMA region of at
/// least `bytes`, or `None` on failure. Each arch supplies its own (aarch64:
/// heap `alloc_zeroed` + `dma_to_phys`; x86: `mm::alloc_dma`).
pub type DmaAlloc = fn(usize) -> Option<Dma>;

/// The concrete disk device for this arch, behind the shared [`BlockDevice`]
/// API — a transport-selecting `Disk` enum on both arches (virtio, plus the
/// real-hardware controllers NVMe and AHCI discovered over PCIe). Consumers
/// (fs, ext4, install, persistence) name `DiskDevice` and call [`probe_disk`],
/// so the whole storage stack is arch-independent; only device discovery is
/// arch-specific (per the dual-arch standing rule).
#[cfg(target_arch = "x86_64")]
pub type DiskDevice = crate::arch::x86_64::disk::Disk;
#[cfg(target_arch = "aarch64")]
pub type DiskDevice = crate::arch::aarch64::disk::Disk;

/// Probe for the boot disk on this arch, if present (the first block device).
pub fn probe_disk() -> Option<DiskDevice> {
    probe_disk_nth(0)
}

/// Probe the `n`-th block device (0-based) on this arch, if present. More than
/// one appears when a boot ESP disk and a data/target disk are both attached
/// (the aarch64 UEFI flow); `/install` uses this to find the payload source vs
/// the install target on both arches through the same API.
pub fn probe_disk_nth(n: usize) -> Option<DiskDevice> {
    #[cfg(target_arch = "x86_64")]
    {
        crate::arch::x86_64::disk::Disk::probe_nth(n)
    }
    #[cfg(target_arch = "aarch64")]
    {
        crate::arch::aarch64::disk::Disk::probe_nth(n)
    }
}

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
    // Forward batched IO to the underlying device (else the default loop would
    // fall back to per-sector requests and lose the batching).
    fn read_blocks(&mut self, index: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        let n = (buf.len() / BLOCK_SIZE) as u64;
        if index + n > self.count {
            return Err(BlockError::OutOfRange);
        }
        self.dev.read_blocks(self.start + index, buf)
    }
    fn write_blocks(&mut self, index: u64, buf: &[u8]) -> Result<(), BlockError> {
        let n = (buf.len() / BLOCK_SIZE) as u64;
        if index + n > self.count {
            return Err(BlockError::OutOfRange);
        }
        self.dev.write_blocks(self.start + index, buf)
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

    /// Read `buf.len()/BLOCK_SIZE` consecutive blocks starting at `index`.
    /// Default: a per-block loop. Real devices (virtio) override this with
    /// multi-sector requests — one polled round trip per tens of KiB instead of
    /// per 512 bytes, the difference between minutes and seconds for a large
    /// file (e.g. the model in `/install`). `buf` must be a BLOCK_SIZE multiple.
    fn read_blocks(&mut self, index: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        for (i, chunk) in buf.chunks_mut(BLOCK_SIZE).enumerate() {
            self.read_block(index + i as u64, chunk)?;
        }
        Ok(())
    }

    /// Write `buf.len()/BLOCK_SIZE` consecutive blocks starting at `index`.
    /// See [`BlockDevice::read_blocks`] for the batching contract.
    fn write_blocks(&mut self, index: u64, buf: &[u8]) -> Result<(), BlockError> {
        for (i, chunk) in buf.chunks(BLOCK_SIZE).enumerate() {
            self.write_block(index + i as u64, chunk)?;
        }
        Ok(())
    }
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
