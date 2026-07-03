//! **virtio-blk over the modern virtio-pci transport** (virtio 1.0) — the
//! real-hardware path, on the PCIe bus discovered via ACPI ECAM
//! (`crate::pci`/`crate::acpi`) rather than QEMU's fixed `virtio-mmio` window.
//! Works wherever a hypervisor/board exposes virtio over PCIe (QEMU virt's GPEX
//! bridge, cloud ARM, etc.). Polled, one request in flight, batched up to
//! `DATA_MAX` per request through a DMA bounce buffer — the same [`BlockDevice`]
//! contract as the mmio driver.

use crate::arch::aarch64::dma_to_phys as dma;
use crate::block::{BlockDevice, BlockError, BLOCK_SIZE};
use crate::pci::{self, PciDevice};
use alloc::alloc::{alloc_zeroed, Layout};
use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{fence, Ordering};

const VIRTIO_VENDOR: u16 = 0x1af4;
// Transitional (0x1001) + modern (0x1042) virtio-blk PCI device ids.
const VIRTIO_BLK_IDS: [u16; 2] = [0x1001, 0x1042];

// virtio-pci capability structure types.
const CFG_COMMON: u8 = 1;
const CFG_NOTIFY: u8 = 2;
const CFG_DEVICE: u8 = 4;

// common-cfg field offsets (virtio_pci_common_cfg).
const DRIVER_FEATURE_SEL: u64 = 0x08;
const DRIVER_FEATURE: u64 = 0x0c;
const DEVICE_STATUS: u64 = 0x14;
const QUEUE_SELECT: u64 = 0x16;
const QUEUE_SIZE: u64 = 0x18;
const QUEUE_ENABLE: u64 = 0x1c;
const QUEUE_NOTIFY_OFF: u64 = 0x1e;
const QUEUE_DESC: u64 = 0x20;
const QUEUE_DRIVER: u64 = 0x28;
const QUEUE_DEVICE: u64 = 0x30;

const S_ACK: u8 = 1;
const S_DRIVER: u8 = 2;
const S_DRIVER_OK: u8 = 4;
const S_FEATURES_OK: u8 = 8;

const VIRTQ_DESC_F_NEXT: u16 = 1;
const VIRTQ_DESC_F_WRITE: u16 = 2;
const VIRTIO_BLK_T_IN: u32 = 0;
const VIRTIO_BLK_T_OUT: u32 = 1;

const QSIZE: usize = 16;
const DATA_MAX: usize = 64 * 1024;

fn dsb() {
    unsafe { core::arch::asm!("dsb sy", options(nomem, nostack, preserves_flags)) };
}
fn alloc_ident(bytes: usize) -> u64 {
    let layout = Layout::from_size_align(bytes.max(1), 4096).unwrap();
    let p = unsafe { alloc_zeroed(layout) };
    assert!(!p.is_null(), "virtio_pci: DMA alloc failed");
    p as u64
}
unsafe fn r8(a: u64) -> u8 {
    unsafe { read_volatile(a as *const u8) }
}
unsafe fn r16(a: u64) -> u16 {
    unsafe { read_volatile(a as *const u16) }
}
unsafe fn w8(a: u64, v: u8) {
    unsafe { write_volatile(a as *mut u8, v) };
}
unsafe fn w16(a: u64, v: u16) {
    unsafe { write_volatile(a as *mut u16, v) };
}
unsafe fn w32(a: u64, v: u32) {
    unsafe { write_volatile(a as *mut u32, v) };
}
unsafe fn w64(a: u64, v: u64) {
    unsafe { write_volatile(a as *mut u64, v) };
}
unsafe fn rd8(a: u64) -> u8 {
    unsafe { read_volatile(a as *const u8) }
}
unsafe fn rd16(a: u64) -> u16 {
    unsafe { read_volatile(a as *const u16) }
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

/// A polled modern-virtio-pci block device.
pub struct VirtioBlkPci {
    notify: u64, // notify address for queue 0
    capacity: u64,
    qsize: u16,
    q_desc: u64,
    q_avail: u64,
    q_used: u64,
    req: u64,
    data_buf: u64,
    avail_idx: u16,
}

impl VirtioBlkPci {
    pub fn probe_nth(n: usize) -> Option<VirtioBlkPci> {
        let d = pci::find_nth(VIRTIO_VENDOR, &VIRTIO_BLK_IDS, n)?;
        Self::init(d)
    }

    fn init(d: PciDevice) -> Option<VirtioBlkPci> {
        d.enable_bus_master();
        // Locate the common-cfg and notify structures via virtio caps.
        let common_cap = d.find_virtio_cap(CFG_COMMON)?;
        let notify_cap = d.find_virtio_cap(CFG_NOTIFY)?;
        let device_cap = d.find_virtio_cap(CFG_DEVICE)?;
        let cap_region = |cap: u16| -> u64 {
            let bar = pci::read8(d.bus, d.dev, d.func, cap + 4);
            let off = pci::read32(d.bus, d.dev, d.func, cap + 8) as u64;
            d.bar(bar) + off
        };
        let common = cap_region(common_cap);
        let notify_base = cap_region(notify_cap);
        let notify_mult = pci::read32(d.bus, d.dev, d.func, notify_cap + 16) as u64;
        let devcfg = cap_region(device_cap);

        // SAFETY: `common`/`notify_base`/`devcfg` are BAR-mapped MMIO in the PCI
        // window (identity-mapped Device memory); the virtio 1.0 layout below is
        // per the spec.
        unsafe {
            w8(common + DEVICE_STATUS, 0); // reset
            // spin until reset observed
            let mut g = 0u64;
            while r8(common + DEVICE_STATUS) != 0 && g < 1_000_000 {
                g += 1;
            }
            w8(common + DEVICE_STATUS, S_ACK);
            w8(common + DEVICE_STATUS, S_ACK | S_DRIVER);

            // Accept only VIRTIO_F_VERSION_1 (feature bit 32).
            w32(common + DRIVER_FEATURE_SEL, 0);
            w32(common + DRIVER_FEATURE, 0);
            w32(common + DRIVER_FEATURE_SEL, 1);
            w32(common + DRIVER_FEATURE, 1);
            w8(common + DEVICE_STATUS, S_ACK | S_DRIVER | S_FEATURES_OK);
            if r8(common + DEVICE_STATUS) & S_FEATURES_OK == 0 {
                return None;
            }

            // Queue 0 setup.
            w16(common + QUEUE_SELECT, 0);
            let qmax = r16(common + QUEUE_SIZE);
            if qmax == 0 {
                return None;
            }
            let qsize = (QSIZE as u16).min(qmax);
            w16(common + QUEUE_SIZE, qsize);
            let qs = qsize as usize;
            let desc = alloc_ident(qs * 16);
            let avail = alloc_ident(6 + qs * 2);
            let used = alloc_ident(6 + qs * 8);
            w64(common + QUEUE_DESC, dma(desc));
            w64(common + QUEUE_DRIVER, dma(avail));
            w64(common + QUEUE_DEVICE, dma(used));
            let notify_off = r16(common + QUEUE_NOTIFY_OFF) as u64;
            let notify = notify_base + notify_off * notify_mult;
            w16(common + QUEUE_ENABLE, 1);
            w8(common + DEVICE_STATUS, S_ACK | S_DRIVER | S_FEATURES_OK | S_DRIVER_OK);

            // blk capacity (u64 sectors) at device-cfg offset 0.
            let capacity = (r16(devcfg) as u64)
                | ((r16(devcfg + 2) as u64) << 16)
                | ((r16(devcfg + 4) as u64) << 32)
                | ((r16(devcfg + 6) as u64) << 48);
            crate::ktrace::log_fmt(format_args!(
                "virtio-blk-pci: up ({:02x}:{:02x}.{}) {capacity} sectors ({} MiB), q{qsize}",
                d.bus,
                d.dev,
                d.func,
                capacity * 512 / (1024 * 1024)
            ));
            Some(VirtioBlkPci {
                notify,
                capacity,
                qsize,
                q_desc: desc,
                q_avail: avail,
                q_used: used,
                req: alloc_ident(4096),
                data_buf: alloc_ident(DATA_MAX),
                avail_idx: 0,
            })
        }
    }

    fn request(&mut self, write: bool, sector: u64, ptr: *mut u8, len: usize) -> Result<(), BlockError> {
        const HDR: u64 = 0;
        const STAT: u64 = 16;
        // SAFETY: queue/req regions from init; 3-descriptor chain per spec.
        unsafe {
            wr32(self.req + HDR, if write { VIRTIO_BLK_T_OUT } else { VIRTIO_BLK_T_IN });
            wr32(self.req + HDR + 4, 0);
            wr64(self.req + HDR + 8, sector);
            if write {
                core::ptr::copy_nonoverlapping(ptr as *const u8, self.data_buf as *mut u8, len);
            }
            wr8(self.req + STAT, 0xff);

            let d = |i: u64| self.q_desc + i * 16;
            wr64(d(0), dma(self.req + HDR));
            wr32(d(0) + 8, 16);
            wr16(d(0) + 12, VIRTQ_DESC_F_NEXT);
            wr16(d(0) + 14, 1);
            wr64(d(1), dma(self.data_buf));
            wr32(d(1) + 8, len as u32);
            wr16(d(1) + 12, VIRTQ_DESC_F_NEXT | if write { 0 } else { VIRTQ_DESC_F_WRITE });
            wr16(d(1) + 14, 2);
            wr64(d(2), dma(self.req + STAT));
            wr32(d(2) + 8, 1);
            wr16(d(2) + 12, VIRTQ_DESC_F_WRITE);
            wr16(d(2) + 14, 0);

            let slot = self.avail_idx % self.qsize;
            wr16(self.q_avail + 4 + slot as u64 * 2, 0);
            fence(Ordering::SeqCst);
            self.avail_idx = self.avail_idx.wrapping_add(1);
            wr16(self.q_avail + 2, self.avail_idx);
            dsb();
            wr16(self.notify, 0); // notify queue 0

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
                core::ptr::copy_nonoverlapping(self.data_buf as *const u8, ptr, len);
            }
        }
        Ok(())
    }
}

impl BlockDevice for VirtioBlkPci {
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
        self.request(false, index, buf.as_mut_ptr(), BLOCK_SIZE)
    }
    fn write_block(&mut self, index: u64, buf: &[u8]) -> Result<(), BlockError> {
        if buf.len() != BLOCK_SIZE {
            return Err(BlockError::BadBufferLen);
        }
        if index >= self.capacity {
            return Err(BlockError::OutOfRange);
        }
        self.request(true, index, buf.as_ptr() as *mut u8, BLOCK_SIZE)
    }
    fn read_blocks(&mut self, index: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        if buf.len() % BLOCK_SIZE != 0 {
            return Err(BlockError::BadBufferLen);
        }
        if index + (buf.len() / BLOCK_SIZE) as u64 > self.capacity {
            return Err(BlockError::OutOfRange);
        }
        let mut off = 0usize;
        while off < buf.len() {
            let take = (buf.len() - off).min(DATA_MAX);
            self.request(false, index + (off / BLOCK_SIZE) as u64, buf[off..].as_mut_ptr(), take)?;
            off += take;
        }
        Ok(())
    }
    fn write_blocks(&mut self, index: u64, buf: &[u8]) -> Result<(), BlockError> {
        if buf.len() % BLOCK_SIZE != 0 {
            return Err(BlockError::BadBufferLen);
        }
        if index + (buf.len() / BLOCK_SIZE) as u64 > self.capacity {
            return Err(BlockError::OutOfRange);
        }
        let mut off = 0usize;
        while off < buf.len() {
            let take = (buf.len() - off).min(DATA_MAX);
            self.request(true, index + (off / BLOCK_SIZE) as u64, buf[off..].as_ptr() as *mut u8, take)?;
            off += take;
        }
        Ok(())
    }
}
