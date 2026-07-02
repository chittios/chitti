//! virtio-blk over the **legacy PCI transport** (`CHITTI_OS_HANDOFF.md`
//! Phase 7): a real, reboot-persistent disk. This is the block device the
//! boot path mounts SimpleFS on when QEMU is started with a virtio-blk drive
//! (`disable-modern=on` selects the legacy I/O interface this driver speaks).
//!
//! The driver is deliberately synchronous and **polled** (no interrupt): one
//! request in flight at a time, wait on the used ring, return. That is plenty
//! for a boot-time filesystem and keeps the code small and auditable. It
//! covers exactly what SimpleFS needs -- read one sector, write one sector.
//!
//! DMA note: the virtqueue and the request buffers must be physically
//! contiguous and are handed to the device *by physical address*
//! (`mm::alloc_dma`); the CPU touches the same memory through the HHDM.

use super::{BlockDevice, BlockError, BLOCK_SIZE};
use crate::arch::x86_64::port::{inl, inw, outb, outl, outw};
use core::sync::atomic::{fence, Ordering};

const VIRTIO_VENDOR: u16 = 0x1af4;
/// Transitional virtio-blk PCI device id (legacy interface available).
const VIRTIO_BLK_DEVICE: u16 = 0x1001;

// Legacy virtio PCI I/O register offsets (from the I/O BAR base).
const R_DEVICE_FEATURES: u16 = 0x00;
const R_GUEST_FEATURES: u16 = 0x04;
const R_QUEUE_PFN: u16 = 0x08;
const R_QUEUE_SIZE: u16 = 0x0c;
const R_QUEUE_SELECT: u16 = 0x0e;
const R_QUEUE_NOTIFY: u16 = 0x10;
const R_DEVICE_STATUS: u16 = 0x12;
const R_CONFIG: u16 = 0x14; // blk config: capacity (u64, in 512-byte sectors)

// Device status bits.
const S_ACKNOWLEDGE: u8 = 1;
const S_DRIVER: u8 = 2;
const S_DRIVER_OK: u8 = 4;

// Descriptor flags.
const VIRTQ_DESC_F_NEXT: u16 = 1;
const VIRTQ_DESC_F_WRITE: u16 = 2; // device writes (i.e. our read buffer)

// virtio-blk request types.
const VIRTIO_BLK_T_IN: u32 = 0; // read from device
const VIRTIO_BLK_T_OUT: u32 = 1; // write to device

const PAGE: u64 = 4096;

fn align_up(x: u64, a: u64) -> u64 {
    (x + a - 1) & !(a - 1)
}

// --- minimal PCI config-space access (port 0xCF8/0xCFC) ------------------

fn pci_addr(bus: u8, slot: u8, func: u8, offset: u8) -> u32 {
    0x8000_0000
        | ((bus as u32) << 16)
        | ((slot as u32) << 11)
        | ((func as u32) << 8)
        | ((offset as u32) & 0xfc)
}

fn pci_read32(bus: u8, slot: u8, func: u8, offset: u8) -> u32 {
    // SAFETY: 0xCF8/0xCFC are the standard PCI configuration ports.
    unsafe {
        outl(0xcf8, pci_addr(bus, slot, func, offset));
        inl(0xcfc)
    }
}

fn pci_write32(bus: u8, slot: u8, func: u8, offset: u8, value: u32) {
    // SAFETY: as `pci_read32`.
    unsafe {
        outl(0xcf8, pci_addr(bus, slot, func, offset));
        outl(0xcfc, value);
    }
}

// --- volatile memory helpers over the DMA region -------------------------

unsafe fn wr16(addr: u64, v: u16) {
    unsafe { core::ptr::write_volatile(addr as *mut u16, v) };
}
unsafe fn wr32(addr: u64, v: u32) {
    unsafe { core::ptr::write_volatile(addr as *mut u32, v) };
}
unsafe fn wr64(addr: u64, v: u64) {
    unsafe { core::ptr::write_volatile(addr as *mut u64, v) };
}
unsafe fn rd16(addr: u64) -> u16 {
    unsafe { core::ptr::read_volatile(addr as *const u16) }
}
unsafe fn rd8(addr: u64) -> u8 {
    unsafe { core::ptr::read_volatile(addr as *const u8) }
}

/// A polled legacy virtio-blk device.
pub struct VirtioBlk {
    io_base: u16,
    capacity: u64, // in 512-byte sectors
    qsize: u16,
    q_desc: u64,   // virtual address of the descriptor table
    q_avail: u64,  // virtual address of the available ring
    q_used: u64,   // virtual address of the used ring
    req_virt: u64, // virtual address of the request scratch page
    req_phys: u64, // physical address of the same
    avail_idx: u16,
}

impl VirtioBlk {
    /// Scan PCI bus 0 for a legacy virtio-blk device and, if found, bring it
    /// up and return it. `None` if no such device is present (e.g. QEMU was
    /// started without a virtio-blk drive).
    pub fn probe() -> Option<VirtioBlk> {
        for slot in 0u8..32 {
            let id = pci_read32(0, slot, 0, 0x00);
            let vendor = (id & 0xffff) as u16;
            let device = (id >> 16) as u16;
            if vendor == VIRTIO_VENDOR && device == VIRTIO_BLK_DEVICE {
                return VirtioBlk::init(0, slot, 0);
            }
        }
        None
    }

    fn init(bus: u8, slot: u8, func: u8) -> Option<VirtioBlk> {
        // BAR0 is the legacy I/O BAR; bit 0 set marks an I/O-space BAR.
        let bar0 = pci_read32(bus, slot, func, 0x10);
        if bar0 & 1 == 0 {
            return None; // not an I/O BAR -- not the legacy interface
        }
        let io_base = (bar0 & 0xfffc) as u16;

        // Enable I/O space (bit 0) and bus-master DMA (bit 2) in the command
        // register (low 16 bits of the dword at offset 0x04).
        let cmd = pci_read32(bus, slot, func, 0x04);
        pci_write32(bus, slot, func, 0x04, cmd | 0b101);

        // SAFETY: `io_base` is this device's legacy virtio I/O register block;
        // the offsets/widths below follow the legacy virtio-pci layout.
        unsafe {
            // Reset, then ACKNOWLEDGE + DRIVER.
            outb(io_base + R_DEVICE_STATUS, 0);
            outb(io_base + R_DEVICE_STATUS, S_ACKNOWLEDGE);
            outb(io_base + R_DEVICE_STATUS, S_ACKNOWLEDGE | S_DRIVER);

            // Feature negotiation: read what the device offers, accept none
            // (basic read/write needs no optional features in legacy mode).
            let _features = inl(io_base + R_DEVICE_FEATURES);
            outl(io_base + R_GUEST_FEATURES, 0);

            // Capacity (sectors) from device config.
            let cap_lo = inl(io_base + R_CONFIG) as u64;
            let cap_hi = inl(io_base + R_CONFIG + 4) as u64;
            let capacity = (cap_hi << 32) | cap_lo;

            // Set up virtqueue 0.
            outw(io_base + R_QUEUE_SELECT, 0);
            let qsize = inw(io_base + R_QUEUE_SIZE);
            if qsize == 0 {
                return None;
            }
            let qs = qsize as u64;

            // Legacy virtqueue layout: desc | avail | (pad to page) | used.
            let avail_off = 16 * qs;
            let used_off = align_up(avail_off + 6 + 2 * qs, PAGE);
            let queue_bytes = used_off + 6 + 8 * qs;

            let (q_phys, q_virt) = crate::mm::alloc_dma(queue_bytes as usize)?;
            // One scratch page holds the request header (@0), status (@16),
            // and the 512-byte data buffer (@512).
            let (req_phys, req_virt) = crate::mm::alloc_dma(PAGE as usize)?;

            // Tell the device where the queue lives (by page frame number).
            outl(io_base + R_QUEUE_PFN, (q_phys / PAGE) as u32);

            // Driver is ready.
            outb(io_base + R_DEVICE_STATUS, S_ACKNOWLEDGE | S_DRIVER | S_DRIVER_OK);

            crate::ktrace::log_fmt(format_args!(
                "virtio-blk: up at io 0x{io_base:x}, {capacity} sectors ({} MiB), queue size {qsize}",
                capacity * 512 / (1024 * 1024)
            ));

            Some(VirtioBlk {
                io_base,
                capacity,
                qsize,
                q_desc: q_virt,
                q_avail: q_virt + avail_off,
                q_used: q_virt + used_off,
                req_virt,
                req_phys,
                avail_idx: 0,
            })
        }
    }

    /// Issue one polled sector request. `buf` is 512 bytes: the source for a
    /// write, the destination for a read.
    fn request(&mut self, write: bool, sector: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        const HDR_OFF: u64 = 0;
        const STATUS_OFF: u64 = 16;
        const DATA_OFF: u64 = 512;

        // SAFETY: all addresses are within the two DMA regions allocated in
        // `init`; the descriptor/ring layout follows the legacy virtio spec.
        unsafe {
            // Request header.
            wr32(self.req_virt + HDR_OFF, if write { VIRTIO_BLK_T_OUT } else { VIRTIO_BLK_T_IN });
            wr32(self.req_virt + HDR_OFF + 4, 0);
            wr64(self.req_virt + HDR_OFF + 8, sector);
            if write {
                core::ptr::copy_nonoverlapping(buf.as_ptr(), (self.req_virt + DATA_OFF) as *mut u8, BLOCK_SIZE);
            }
            wr8(self.req_virt + STATUS_OFF, 0xff); // sentinel; device overwrites

            // Three chained descriptors: header (r), data (r/w), status (w).
            let d = |i: u64| self.q_desc + i * 16;
            wr64(d(0), self.req_phys + HDR_OFF);
            wr32(d(0) + 8, 16);
            wr16(d(0) + 12, VIRTQ_DESC_F_NEXT);
            wr16(d(0) + 14, 1);

            wr64(d(1), self.req_phys + DATA_OFF);
            wr32(d(1) + 8, BLOCK_SIZE as u32);
            wr16(d(1) + 12, VIRTQ_DESC_F_NEXT | if write { 0 } else { VIRTQ_DESC_F_WRITE });
            wr16(d(1) + 14, 2);

            wr64(d(2), self.req_phys + STATUS_OFF);
            wr32(d(2) + 8, 1);
            wr16(d(2) + 12, VIRTQ_DESC_F_WRITE);
            wr16(d(2) + 14, 0);

            // Publish the head descriptor (index 0) into the available ring.
            let slot = self.avail_idx % self.qsize;
            wr16(self.q_avail + 4 + slot as u64 * 2, 0);
            fence(Ordering::SeqCst);
            self.avail_idx = self.avail_idx.wrapping_add(1);
            wr16(self.q_avail + 2, self.avail_idx); // avail.idx
            fence(Ordering::SeqCst);

            // Notify the device and poll the used ring for completion.
            outw(self.io_base + R_QUEUE_NOTIFY, 0);
            let mut spins: u64 = 0;
            while rd16(self.q_used + 2) != self.avail_idx {
                core::hint::spin_loop();
                spins += 1;
                if spins > 2_000_000_000 {
                    return Err(BlockError::DeviceError);
                }
            }
            fence(Ordering::SeqCst);

            if rd8(self.req_virt + STATUS_OFF) != 0 {
                return Err(BlockError::DeviceError);
            }
            if !write {
                core::ptr::copy_nonoverlapping((self.req_virt + DATA_OFF) as *const u8, buf.as_mut_ptr(), BLOCK_SIZE);
            }
        }
        Ok(())
    }
}

unsafe fn wr8(addr: u64, v: u8) {
    unsafe { core::ptr::write_volatile(addr as *mut u8, v) };
}

impl BlockDevice for VirtioBlk {
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
        // `request` needs `&mut [u8]`; copy into a local for the write path.
        let mut tmp = [0u8; BLOCK_SIZE];
        tmp.copy_from_slice(buf);
        self.request(true, index, &mut tmp)
    }
}
