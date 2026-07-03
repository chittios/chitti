//! **virtio-blk over the virtio-mmio transport** — the aarch64 counterpart to
//! the x86 `block::virtio` driver (which uses legacy PCI port I/O). QEMU's
//! `virt` machine has no PCI port space; a `-drive ... -device
//! virtio-blk-device` lands on a virtio-mmio slot, the same window the
//! virtio-input keyboard uses. This driver scans those slots for a block device
//! (id 2), brings it up (both legacy v1 and modern v2 transports), and services
//! one polled sector request at a time via a 3-descriptor chain — exactly what
//! the filesystem/ext4/install stack needs, behind the shared [`BlockDevice`]
//! API so that stack runs unchanged on both arches.
//!
//! DMA: the virtqueue + request buffers are page-aligned identity memory
//! (VA == PA on the aarch64 identity map — the same assumption ramfb and
//! virtio-input rely on), handed to the device by physical address; `dsb`
//! fences order the CPU's writes before each device notification.

use crate::block::{BlockDevice, BlockError, BLOCK_SIZE};
use alloc::alloc::{alloc_zeroed, Layout};
use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{fence, Ordering};

// virtio-mmio register offsets (shared with virtio_input).
const MAGIC: usize = 0x000;
const VERSION: usize = 0x004;
const DEVICE_ID: usize = 0x008;
const DEVICE_FEATURES: usize = 0x010;
const DEVICE_FEATURES_SEL: usize = 0x014;
const DRIVER_FEATURES: usize = 0x020;
const DRIVER_FEATURES_SEL: usize = 0x024;
const GUEST_PAGE_SIZE: usize = 0x028; // legacy (v1) only
const QUEUE_SEL: usize = 0x030;
const QUEUE_NUM_MAX: usize = 0x034;
const QUEUE_NUM: usize = 0x038;
const QUEUE_ALIGN: usize = 0x03c; // legacy (v1) only
const QUEUE_PFN: usize = 0x040; // legacy (v1) only
const QUEUE_READY: usize = 0x044; // modern (v2) only
const QUEUE_NOTIFY: usize = 0x050;
const STATUS: usize = 0x070;
const QUEUE_DESC_LOW: usize = 0x080;
const QUEUE_DESC_HIGH: usize = 0x084;
const QUEUE_DRIVER_LOW: usize = 0x090;
const QUEUE_DRIVER_HIGH: usize = 0x094;
const QUEUE_DEVICE_LOW: usize = 0x0a0;
const QUEUE_DEVICE_HIGH: usize = 0x0a4;
const CONFIG: usize = 0x100; // device-specific config (blk capacity @ +0)

const S_ACK: u32 = 1;
const S_DRIVER: u32 = 2;
const S_DRIVER_OK: u32 = 4;
const S_FEATURES_OK: u32 = 8;

const VIRTIO_ID_BLOCK: u32 = 2;
const VIRTQ_DESC_F_NEXT: u16 = 1;
const VIRTQ_DESC_F_WRITE: u16 = 2;
const VIRTIO_BLK_T_IN: u32 = 0;
const VIRTIO_BLK_T_OUT: u32 = 1;

const MMIO_BASE: usize = 0x0a00_0000;
const MMIO_STRIDE: usize = 0x200;
const MMIO_SLOTS: usize = 32;

const QSIZE: usize = 8;

#[inline]
fn dsb() {
    unsafe { core::arch::asm!("dsb sy", options(nomem, nostack, preserves_flags)) };
}
unsafe fn reg_read(base: usize, off: usize) -> u32 {
    unsafe { read_volatile((base + off) as *const u32) }
}
unsafe fn reg_write(base: usize, off: usize, val: u32) {
    unsafe { write_volatile((base + off) as *mut u32, val) };
}
fn align_up(x: usize, a: usize) -> usize {
    (x + a - 1) & !(a - 1)
}

/// Page-aligned, zeroed, leaked identity memory (VA == PA on the aarch64 map).
fn alloc_ident(bytes: usize) -> u64 {
    let layout = Layout::from_size_align(bytes.max(1), 4096).unwrap();
    // SAFETY: nonzero layout; leaked and used only as device-shared DMA.
    let p = unsafe { alloc_zeroed(layout) };
    assert!(!p.is_null(), "virtio_blk: DMA alloc failed");
    p as u64
}

unsafe fn wr8(a: u64, v: u8) {
    unsafe { write_volatile(a as *mut u8, v) };
}
unsafe fn wr16(a: u64, v: u16) {
    unsafe { write_volatile(a as *mut u16, v) };
}
unsafe fn wr32(a: u64, v: u32) {
    unsafe { write_volatile(a as *mut u32, v) };
}
unsafe fn wr64(a: u64, v: u64) {
    unsafe { write_volatile(a as *mut u64, v) };
}
unsafe fn rd8(a: u64) -> u8 {
    unsafe { read_volatile(a as *const u8) }
}
unsafe fn rd16(a: u64) -> u16 {
    unsafe { read_volatile(a as *const u16) }
}

/// A polled virtio-mmio block device.
pub struct VirtioBlkMmio {
    base: usize,
    capacity: u64, // 512-byte sectors
    qsize: u16,
    q_desc: u64,
    q_avail: u64,
    q_used: u64,
    req: u64, // request scratch page (VA==PA)
    avail_idx: u16,
}

impl VirtioBlkMmio {
    /// Scan the virtio-mmio window for a block device and bring it up.
    pub fn probe() -> Option<VirtioBlkMmio> {
        let mut base = 0usize;
        let mut version = 0u32;
        for slot in 0..MMIO_SLOTS {
            let b = MMIO_BASE + slot * MMIO_STRIDE;
            // SAFETY: scanning the fixed virtio-mmio window; 32-bit registers.
            unsafe {
                let v = reg_read(b, VERSION);
                if reg_read(b, MAGIC) == 0x7472_6976 && (v == 1 || v == 2) && reg_read(b, DEVICE_ID) == VIRTIO_ID_BLOCK {
                    base = b;
                    version = v;
                    break;
                }
            }
        }
        if base == 0 {
            return None;
        }
        Self::init(base, version)
    }

    fn init(base: usize, version: u32) -> Option<VirtioBlkMmio> {
        let req = alloc_ident(4096);
        // SAFETY: `base` is a confirmed virtio-blk MMIO block; single-core boot.
        unsafe {
            reg_write(base, STATUS, 0);
            reg_write(base, STATUS, S_ACK);
            reg_write(base, STATUS, S_ACK | S_DRIVER);

            reg_write(base, QUEUE_SEL, 0);
            let qmax = reg_read(base, QUEUE_NUM_MAX);
            if qmax == 0 {
                return None;
            }
            let qsize = (QSIZE as u32).min(qmax) as u16;
            reg_write(base, QUEUE_NUM, qsize as u32);
            let qs = qsize as usize;

            let (desc, avail, used) = if version == 2 {
                reg_write(base, DEVICE_FEATURES_SEL, 1);
                let _ = reg_read(base, DEVICE_FEATURES);
                reg_write(base, DRIVER_FEATURES_SEL, 1);
                reg_write(base, DRIVER_FEATURES, 1); // ack VIRTIO_F_VERSION_1 (bit 32)
                reg_write(base, DRIVER_FEATURES_SEL, 0);
                reg_write(base, DRIVER_FEATURES, 0);
                reg_write(base, STATUS, S_ACK | S_DRIVER | S_FEATURES_OK);
                if reg_read(base, STATUS) & S_FEATURES_OK == 0 {
                    return None;
                }
                let desc = alloc_ident(qs * 16);
                let avail = alloc_ident(6 + qs * 2);
                let used = alloc_ident(6 + qs * 8);
                reg_write(base, QUEUE_DESC_LOW, desc as u32);
                reg_write(base, QUEUE_DESC_HIGH, (desc >> 32) as u32);
                reg_write(base, QUEUE_DRIVER_LOW, avail as u32);
                reg_write(base, QUEUE_DRIVER_HIGH, (avail >> 32) as u32);
                reg_write(base, QUEUE_DEVICE_LOW, used as u32);
                reg_write(base, QUEUE_DEVICE_HIGH, (used >> 32) as u32);
                reg_write(base, QUEUE_READY, 1);
                reg_write(base, STATUS, S_ACK | S_DRIVER | S_FEATURES_OK | S_DRIVER_OK);
                (desc, avail, used)
            } else {
                // legacy (v1): one contiguous region addressed by page frame.
                reg_write(base, DRIVER_FEATURES_SEL, 0);
                reg_write(base, DRIVER_FEATURES, 0);
                reg_write(base, GUEST_PAGE_SIZE, 4096);
                reg_write(base, QUEUE_ALIGN, 4096);
                let used_off = align_up(qs * 16 + (6 + qs * 2), 4096);
                let region = alloc_ident(used_off + 6 + qs * 8);
                reg_write(base, QUEUE_PFN, (region >> 12) as u32);
                reg_write(base, STATUS, S_ACK | S_DRIVER | S_DRIVER_OK);
                (region, region + (qs * 16) as u64, region + used_off as u64)
            };

            let capacity = (reg_read(base, CONFIG) as u64) | ((reg_read(base, CONFIG + 4) as u64) << 32);
            crate::ktrace::log_fmt(format_args!(
                "virtio-blk-mmio: up at {base:#x} (v{version}), {capacity} sectors ({} MiB), q{qsize}",
                capacity * 512 / (1024 * 1024)
            ));
            Some(VirtioBlkMmio { base, capacity, qsize, q_desc: desc, q_avail: avail, q_used: used, req, avail_idx: 0 })
        }
    }

    /// Issue one polled sector request. `buf` is 512 bytes.
    fn request(&mut self, write: bool, sector: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        const HDR: u64 = 0;
        const STAT: u64 = 16;
        const DATA: u64 = 512;
        // SAFETY: all addresses lie within the queue/request DMA regions from
        // `init`; the 3-descriptor chain follows the virtio-blk spec.
        unsafe {
            wr32(self.req + HDR, if write { VIRTIO_BLK_T_OUT } else { VIRTIO_BLK_T_IN });
            wr32(self.req + HDR + 4, 0);
            wr64(self.req + HDR + 8, sector);
            if write {
                core::ptr::copy_nonoverlapping(buf.as_ptr(), (self.req + DATA) as *mut u8, BLOCK_SIZE);
            }
            wr8(self.req + STAT, 0xff);

            let d = |i: u64| self.q_desc + i * 16;
            wr64(d(0), self.req + HDR);
            wr32(d(0) + 8, 16);
            wr16(d(0) + 12, VIRTQ_DESC_F_NEXT);
            wr16(d(0) + 14, 1);
            wr64(d(1), self.req + DATA);
            wr32(d(1) + 8, BLOCK_SIZE as u32);
            wr16(d(1) + 12, VIRTQ_DESC_F_NEXT | if write { 0 } else { VIRTQ_DESC_F_WRITE });
            wr16(d(1) + 14, 2);
            wr64(d(2), self.req + STAT);
            wr32(d(2) + 8, 1);
            wr16(d(2) + 12, VIRTQ_DESC_F_WRITE);
            wr16(d(2) + 14, 0);

            let slot = self.avail_idx % self.qsize;
            wr16(self.q_avail + 4 + slot as u64 * 2, 0);
            fence(Ordering::SeqCst);
            self.avail_idx = self.avail_idx.wrapping_add(1);
            wr16(self.q_avail + 2, self.avail_idx);
            dsb();
            reg_write(self.base, QUEUE_NOTIFY, 0);

            let mut spins: u64 = 0;
            while rd16(self.q_used + 2) != self.avail_idx {
                core::hint::spin_loop();
                spins += 1;
                if spins > 2_000_000_000 {
                    return Err(BlockError::DeviceError);
                }
            }
            fence(Ordering::SeqCst);
            if rd8(self.req + STAT) != 0 {
                return Err(BlockError::DeviceError);
            }
            if !write {
                core::ptr::copy_nonoverlapping((self.req + DATA) as *const u8, buf.as_mut_ptr(), BLOCK_SIZE);
            }
        }
        Ok(())
    }
}

impl BlockDevice for VirtioBlkMmio {
    fn block_count(&self) -> u64 {
        self.capacity
    }
    fn read_block(&mut self, index: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        if buf.len() != BLOCK_SIZE {
            return Err(BlockError::BadBufferLen);
        }
        if index >= self.capacity {
            return Err(BlockError::OutOfRange);
        }
        self.request(false, index, buf)
    }
    fn write_block(&mut self, index: u64, buf: &[u8]) -> Result<(), BlockError> {
        if buf.len() != BLOCK_SIZE {
            return Err(BlockError::BadBufferLen);
        }
        if index >= self.capacity {
            return Err(BlockError::OutOfRange);
        }
        let mut tmp = [0u8; BLOCK_SIZE];
        tmp.copy_from_slice(buf);
        self.request(true, index, &mut tmp)
    }
}
