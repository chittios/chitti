//! **virtio-net over the virtio-mmio transport** — the NIC the `net` subsystem
//! (smoltcp) sits on for the QEMU `virt` `-kernel` path (`-device
//! virtio-net-device`). Two virtqueues: RX (0, device writes incoming frames
//! into pre-posted buffers) and TX (1, we post outgoing frames). Each frame is
//! prefixed by a 12-byte `virtio_net_hdr` (modern / VERSION_1); we negotiate no
//! offloads and no mergeable-rx-buffers, so one buffer holds one frame.
//!
//! DMA: rings + buffers are page-aligned identity memory (VA == PA on the
//! aarch64 identity map), handed to the device by physical address, with `dsb`
//! fences before each notify — the same model as `virtio_blk`.

use crate::arch::aarch64::dma_to_phys as dma;
use crate::net::NetDevice;
use alloc::alloc::{alloc_zeroed, Layout};
use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{fence, Ordering};

// virtio-mmio register offsets (shared layout with virtio_blk).
const MAGIC: usize = 0x000;
const VERSION: usize = 0x004;
const DEVICE_ID: usize = 0x008;
const DEVICE_FEATURES: usize = 0x010;
const DEVICE_FEATURES_SEL: usize = 0x014;
const DRIVER_FEATURES: usize = 0x020;
const DRIVER_FEATURES_SEL: usize = 0x024;
const GUEST_PAGE_SIZE: usize = 0x028;
const QUEUE_SEL: usize = 0x030;
const QUEUE_NUM_MAX: usize = 0x034;
const QUEUE_NUM: usize = 0x038;
const QUEUE_ALIGN: usize = 0x03c;
const QUEUE_PFN: usize = 0x040;
const QUEUE_READY: usize = 0x044;
const QUEUE_NOTIFY: usize = 0x050;
const STATUS: usize = 0x070;
const QUEUE_DESC_LOW: usize = 0x080;
const QUEUE_DESC_HIGH: usize = 0x084;
const QUEUE_DRIVER_LOW: usize = 0x090;
const QUEUE_DRIVER_HIGH: usize = 0x094;
const QUEUE_DEVICE_LOW: usize = 0x0a0;
const QUEUE_DEVICE_HIGH: usize = 0x0a4;
const CONFIG: usize = 0x100; // device-specific config (MAC @ +0)

const S_ACK: u32 = 1;
const S_DRIVER: u32 = 2;
const S_DRIVER_OK: u32 = 4;
const S_FEATURES_OK: u32 = 8;

const VIRTIO_ID_NET: u32 = 1;
const VIRTQ_DESC_F_WRITE: u16 = 2;
const VIRTIO_NET_F_MAC: u32 = 1 << 5;

const MMIO_BASE: usize = 0x0a00_0000;
const MMIO_STRIDE: usize = 0x200;
const MMIO_SLOTS: usize = 32;

const QSIZE: usize = 16;
const BUFSZ: usize = 2048; // virtio_net_hdr + up to a 1514-byte frame + slack

#[inline]
fn dsb() {
    unsafe { core::arch::asm!("dsb sy", options(nomem, nostack, preserves_flags)) };
}
// Single-instruction MMIO register access via inline asm. `read_volatile` is
// *not* enough here: LLVM merges adjacent volatile loads (MAGIC@0x00 +
// VERSION@0x04) into one 64-bit / paired load, and HVF can't decode a
// multi-register load against device memory — it aborts with `isv`. A discrete
// `ldr`/`str` (memory clobber, no `nomem`) keeps every register access a single
// word transfer, exactly as real MMIO requires.
unsafe fn reg_read(base: usize, off: usize) -> u32 {
    let v: u32;
    // SAFETY: `base+off` is a mapped 32-bit virtio-mmio register.
    unsafe {
        core::arch::asm!("ldr {v:w}, [{a}]", v = out(reg) v, a = in(reg) base + off, options(nostack, preserves_flags));
    }
    v
}
unsafe fn reg_write(base: usize, off: usize, val: u32) {
    // SAFETY: `base+off` is a mapped 32-bit virtio-mmio register.
    unsafe {
        core::arch::asm!("str {v:w}, [{a}]", v = in(reg) val, a = in(reg) base + off, options(nostack, preserves_flags));
    }
}
fn align_up(x: usize, a: usize) -> usize {
    (x + a - 1) & !(a - 1)
}
fn alloc_ident(bytes: usize) -> u64 {
    let layout = Layout::from_size_align(bytes.max(1), 4096).unwrap();
    // SAFETY: nonzero layout; leaked, used only as device-shared DMA memory.
    let p = unsafe { alloc_zeroed(layout) };
    assert!(!p.is_null(), "virtio_net: DMA alloc failed");
    p as u64
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
unsafe fn rd16(a: u64) -> u16 {
    unsafe { read_volatile(a as *const u16) }
}
unsafe fn rd32(a: u64) -> u32 {
    unsafe { read_volatile(a as *const u32) }
}

/// One virtqueue's split-ring pointers + buffer pool.
struct Queue {
    idx: u32,
    qsize: u16,
    desc: u64,
    avail: u64,
    used: u64,
    bufs: u64, // qsize * BUFSZ contiguous buffer pool
    avail_idx: u16,
    used_last: u16,
}

impl Queue {
    /// Set up queue `idx` on the device: negotiate ring memory, publish the
    /// physical addresses (modern v2) or a legacy PFN (v1).
    unsafe fn setup(base: usize, idx: u32, version: u32) -> Option<Queue> {
        unsafe {
            reg_write(base, QUEUE_SEL, idx);
            let qmax = reg_read(base, QUEUE_NUM_MAX);
            if qmax == 0 {
                return None;
            }
            let qsize = (QSIZE as u32).min(qmax) as u16;
            reg_write(base, QUEUE_NUM, qsize as u32);
            let qs = qsize as usize;
            let bufs = alloc_ident(qs * BUFSZ);
            let (desc, avail, used) = if version == 2 {
                let desc = alloc_ident(qs * 16);
                let avail = alloc_ident(6 + qs * 2);
                let used = alloc_ident(6 + qs * 8);
                let (dp, ap, up) = (dma(desc), dma(avail), dma(used));
                reg_write(base, QUEUE_DESC_LOW, dp as u32);
                reg_write(base, QUEUE_DESC_HIGH, (dp >> 32) as u32);
                reg_write(base, QUEUE_DRIVER_LOW, ap as u32);
                reg_write(base, QUEUE_DRIVER_HIGH, (ap >> 32) as u32);
                reg_write(base, QUEUE_DEVICE_LOW, up as u32);
                reg_write(base, QUEUE_DEVICE_HIGH, (up >> 32) as u32);
                reg_write(base, QUEUE_READY, 1);
                (desc, avail, used)
            } else {
                let used_off = align_up(qs * 16 + (6 + qs * 2), 4096);
                let region = alloc_ident(used_off + 6 + qs * 8);
                reg_write(base, GUEST_PAGE_SIZE, 4096);
                reg_write(base, QUEUE_ALIGN, 4096);
                reg_write(base, QUEUE_PFN, (dma(region) >> 12) as u32);
                (region, region + (qs * 16) as u64, region + used_off as u64)
            };
            Some(Queue { idx, qsize, desc, avail, used, bufs, avail_idx: 0, used_last: 0 })
        }
    }

    fn buf(&self, i: u16) -> u64 {
        self.bufs + i as u64 * BUFSZ as u64
    }

    /// Post descriptor `i`'s buffer on the available ring (device-writable if
    /// `write`), then bump `avail.idx`.
    unsafe fn post(&mut self, base: usize, i: u16, len: u32, write: bool) {
        unsafe {
            let d = self.desc + i as u64 * 16;
            wr64(d, dma(self.buf(i)));
            wr32(d + 8, len);
            wr16(d + 12, if write { VIRTQ_DESC_F_WRITE } else { 0 });
            wr16(d + 14, 0);
            let slot = self.avail_idx % self.qsize;
            wr16(self.avail + 4 + slot as u64 * 2, i);
            fence(Ordering::SeqCst);
            self.avail_idx = self.avail_idx.wrapping_add(1);
            wr16(self.avail + 2, self.avail_idx);
            dsb();
            reg_write(base, QUEUE_NOTIFY, self.idx);
        }
    }

    /// Pop the next completed used-ring entry as `(desc_index, bytes_written)`.
    unsafe fn pop_used(&mut self) -> Option<(u16, u32)> {
        unsafe {
            if rd16(self.used + 2) == self.used_last {
                return None;
            }
            fence(Ordering::SeqCst);
            let slot = (self.used_last % self.qsize) as u64;
            let id = rd32(self.used + 4 + slot * 8) as u16;
            let len = rd32(self.used + 4 + slot * 8 + 4);
            self.used_last = self.used_last.wrapping_add(1);
            Some((id, len))
        }
    }
}

/// A polled virtio-net NIC on the virtio-mmio bus.
pub struct VirtioNetMmio {
    base: usize,
    mac: [u8; 6],
    /// Bytes of `virtio_net_hdr` prepended to every frame: 10 on a pure legacy
    /// (v1) device, 12 once VIRTIO_F_VERSION_1 is negotiated (v2). Getting this
    /// wrong shifts every frame by 2 bytes, so the device silently drops our TX.
    hdr_len: usize,
    rx: Queue,
    tx: Queue,
}

impl VirtioNetMmio {
    /// Scan the virtio-mmio window for the first net device (id 1) and bring it up.
    pub fn probe() -> Option<VirtioNetMmio> {
        for slot in 0..MMIO_SLOTS {
            let b = MMIO_BASE + slot * MMIO_STRIDE;
            // SAFETY: scanning the fixed virtio-mmio window; 32-bit registers.
            unsafe {
                let v = reg_read(b, VERSION);
                if reg_read(b, MAGIC) == 0x7472_6976 && (v == 1 || v == 2) && reg_read(b, DEVICE_ID) == VIRTIO_ID_NET {
                    if let Some(d) = Self::init(b, v) {
                        return Some(d);
                    }
                }
            }
        }
        None
    }

    unsafe fn init(base: usize, version: u32) -> Option<VirtioNetMmio> {
        unsafe {
            reg_write(base, STATUS, 0);
            reg_write(base, STATUS, S_ACK);
            reg_write(base, STATUS, S_ACK | S_DRIVER);

            // Feature negotiation: read low features (MAC lives here); ack MAC if
            // offered + VIRTIO_F_VERSION_1 (bit 32, in the high word). We do NOT
            // ack MRG_RXBUF/offloads, so a frame is one plain buffer.
            reg_write(base, DEVICE_FEATURES_SEL, 0);
            let flo = reg_read(base, DEVICE_FEATURES);
            let ack_lo = flo & VIRTIO_NET_F_MAC;
            if version == 2 {
                reg_write(base, DRIVER_FEATURES_SEL, 0);
                reg_write(base, DRIVER_FEATURES, ack_lo);
                reg_write(base, DRIVER_FEATURES_SEL, 1);
                reg_write(base, DRIVER_FEATURES, 1); // VIRTIO_F_VERSION_1 (bit 32)
                reg_write(base, STATUS, S_ACK | S_DRIVER | S_FEATURES_OK);
                if reg_read(base, STATUS) & S_FEATURES_OK == 0 {
                    return None;
                }
            } else {
                reg_write(base, DRIVER_FEATURES_SEL, 0);
                reg_write(base, DRIVER_FEATURES, ack_lo);
            }

            let mut rx = Queue::setup(base, 0, version)?;
            let tx = Queue::setup(base, 1, version)?;

            // Drive the device fully to DRIVER_OK **before** touching its config
            // space — the exact ordering `virtio_blk` uses. On a legacy (v1)
            // device QEMU/HVF faults an (undecodable) config-space read issued
            // before the driver is live, so the MAC must be read only once the
            // device is running. Legacy has no FEATURES_OK bit; modern keeps it.
            let ok = if version == 2 {
                S_ACK | S_DRIVER | S_FEATURES_OK | S_DRIVER_OK
            } else {
                S_ACK | S_DRIVER | S_DRIVER_OK
            };
            reg_write(base, STATUS, ok);

            // MAC from config space, read as two 32-bit words (never sub-word:
            // HVF can't decode a byte-wide device access).
            let c0 = reg_read(base, CONFIG);
            let c1 = reg_read(base, CONFIG + 4);
            let mac = [c0 as u8, (c0 >> 8) as u8, (c0 >> 16) as u8, (c0 >> 24) as u8, c1 as u8, (c1 >> 8) as u8];

            // Pre-post every RX buffer so the device can deliver frames.
            for i in 0..rx.qsize {
                rx.post(base, i, BUFSZ as u32, true);
            }

            crate::ktrace::log_fmt(format_args!(
                "virtio-net-mmio: up at {base:#x} (v{version}), MAC {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}, q{}",
                mac[0], mac[1], mac[2], mac[3], mac[4], mac[5], rx.qsize
            ));
            let hdr_len = if version == 2 { 12 } else { 10 };
            Some(VirtioNetMmio { base, mac, hdr_len, rx, tx })
        }
    }
}

impl NetDevice for VirtioNetMmio {
    fn mac(&self) -> [u8; 6] {
        self.mac
    }

    fn receive(&mut self, out: &mut [u8]) -> Option<usize> {
        // SAFETY: rings/buffers are the live DMA regions from `init`.
        unsafe {
            // Reclaim completed TX buffers first (cheap housekeeping).
            while self.tx.pop_used().is_some() {}
            let (id, len) = self.rx.pop_used()?;
            let total = len as usize;
            let frame = total.saturating_sub(self.hdr_len);
            let n = frame.min(out.len());
            if n > 0 {
                core::ptr::copy_nonoverlapping((self.rx.buf(id) + self.hdr_len as u64) as *const u8, out.as_mut_ptr(), n);
            }
            // Recycle the buffer back onto the RX ring.
            self.rx.post(self.base, id, BUFSZ as u32, true);
            Some(n)
        }
    }

    fn transmit(&mut self, frame: &[u8]) {
        if frame.len() + self.hdr_len > BUFSZ {
            return;
        }
        // Pick a TX descriptor round-robin over the ring.
        let i = (self.tx.avail_idx % self.tx.qsize) as u16;
        // SAFETY: `tx.buf(i)` is BUFSZ bytes of our DMA pool.
        unsafe {
            let b = self.tx.buf(i);
            core::ptr::write_bytes(b as *mut u8, 0, self.hdr_len); // zero virtio_net_hdr
            core::ptr::copy_nonoverlapping(frame.as_ptr(), (b + self.hdr_len as u64) as *mut u8, frame.len());
            self.tx.post(self.base, i, (self.hdr_len + frame.len()) as u32, false);
        }
    }
}
