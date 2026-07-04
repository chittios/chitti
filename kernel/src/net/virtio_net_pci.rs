//! **virtio-net over modern virtio-PCI** — the paravirtual NIC on a real PCI
//! bus (QEMU `-device virtio-net-pci`, VirtualBox's virtio adapter). Same
//! `NetDevice` contract as the aarch64 virtio-net-mmio driver; the difference is
//! the transport: registers live in PCI BARs located by walking the device's
//! virtio PCI capabilities, not at a fixed MMIO window.
//!
//! Modern (VIRTIO_F_VERSION_1) only, poll-driven, two split virtqueues (RX 0 /
//! TX 1), no offloads or mergeable-rx-buffers, 12-byte `virtio_net_hdr` per
//! frame. Dual-arch: PCI config via `crate::arch::x86_64::pci` (ports) or
//! `crate::pci` (ECAM).

use crate::net::NetDevice;
use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{fence, Ordering};

#[cfg(target_arch = "aarch64")]
use crate::pci::{self, PciDevice};
#[cfg(target_arch = "x86_64")]
use crate::arch::x86_64::pci::{self, PciDevice};

// virtio PCI capability cfg_type values.
const CFG_COMMON: u8 = 1;
const CFG_NOTIFY: u8 = 2;
const CFG_DEVICE: u8 = 4;
const CAP_VENDOR: u8 = 0x09;

// common-config field offsets (struct virtio_pci_common_cfg).
const DEVICE_FEATURE_SELECT: u64 = 0x00;
const DEVICE_FEATURE: u64 = 0x04;
const DRIVER_FEATURE_SELECT: u64 = 0x08;
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

const VIRTQ_DESC_F_WRITE: u16 = 2;
const VIRTIO_NET_F_MAC: u32 = 1 << 5; // low feature word

const QSIZE: usize = 16;
const NET_HDR_LEN: usize = 12;
const BUFSZ: usize = 2048;

// --- config-space byte/word reads (via the 32-bit PCI accessor) ----------
fn cfg_read8(d: &PciDevice, off: u16) -> u8 {
    (pci::read32(d.bus, d.dev, d.func, off & !3) >> ((off & 3) * 8)) as u8
}
fn cfg_read32_at(d: &PciDevice, off: u16) -> u32 {
    pci::read32(d.bus, d.dev, d.func, off)
}

// --- BAR-mapped MMIO of arbitrary width ---------------------------------
#[inline]
unsafe fn r8(a: u64) -> u8 {
    unsafe { read_volatile(a as *const u8) }
}
#[inline]
unsafe fn w8(a: u64, v: u8) {
    unsafe { write_volatile(a as *mut u8, v) };
}
#[inline]
unsafe fn r16(a: u64) -> u16 {
    unsafe { read_volatile(a as *const u16) }
}
#[inline]
unsafe fn w16(a: u64, v: u16) {
    unsafe { write_volatile(a as *mut u16, v) };
}
#[inline]
unsafe fn r32(a: u64) -> u32 {
    unsafe { read_volatile(a as *const u32) }
}
#[inline]
unsafe fn w32(a: u64, v: u32) {
    unsafe { write_volatile(a as *mut u32, v) };
}
#[inline]
unsafe fn w64(a: u64, v: u64) {
    unsafe { write_volatile(a as *mut u64, v) };
}

/// A located virtio structure inside a BAR (already HHDM-mapped to a VA).
#[derive(Clone, Copy, Default)]
struct Region {
    virt: u64,
    extra: u32, // notify_off_multiplier for the NOTIFY cap; 0 otherwise
}

/// One split virtqueue: ring pointers + a buffer pool, in DMA memory.
struct Queue {
    qsize: u16,
    desc: u64,
    avail: u64,
    used: u64,
    bufs: u64,
    bufs_phys: u64,
    notify: u64, // MMIO doorbell address for this queue
    vqn: u16,    // virtqueue index, written to the doorbell
    avail_idx: u16,
    used_last: u16,
}

impl Queue {
    fn buf(&self, i: u16) -> u64 {
        self.bufs + i as u64 * BUFSZ as u64
    }
    fn buf_phys(&self, i: u16) -> u64 {
        self.bufs_phys + i as u64 * BUFSZ as u64
    }
    /// Post descriptor `i` on the available ring; ring the notify doorbell.
    unsafe fn post(&mut self, i: u16, len: u32, write: bool) {
        unsafe {
            let d = self.desc + i as u64 * 16;
            w64(d, self.buf_phys(i));
            w32(d + 8, len);
            w16(d + 12, if write { VIRTQ_DESC_F_WRITE } else { 0 });
            w16(d + 14, 0);
            let slot = self.avail_idx % self.qsize;
            w16(self.avail + 4 + slot as u64 * 2, i);
            fence(Ordering::SeqCst);
            self.avail_idx = self.avail_idx.wrapping_add(1);
            w16(self.avail + 2, self.avail_idx);
            fence(Ordering::SeqCst);
            w16(self.notify, self.vqn); // doorbell carries the virtqueue index
        }
    }
    unsafe fn pop_used(&mut self) -> Option<(u16, u32)> {
        unsafe {
            if r16(self.used + 2) == self.used_last {
                return None;
            }
            fence(Ordering::SeqCst);
            let slot = (self.used_last % self.qsize) as u64;
            let id = r32(self.used + 4 + slot * 8) as u16;
            let len = r32(self.used + 4 + slot * 8 + 4);
            self.used_last = self.used_last.wrapping_add(1);
            Some((id, len))
        }
    }
}

/// A poll-driven modern virtio-net NIC on the PCI bus.
pub struct VirtioNetPci {
    mac: [u8; 6],
    rx: Queue,
    tx: Queue,
}

impl VirtioNetPci {
    pub fn init(d: PciDevice) -> Option<VirtioNetPci> {
        d.enable_bus_master();

        // Walk the PCI capability list for the virtio structures we need.
        let mut common = Region::default();
        let mut notify = Region::default();
        let mut device = Region::default();
        // Capabilities-list-present bit (status reg 0x06, bit 4).
        if cfg_read32_at(&d, 0x04) & (1 << 20) == 0 {
            return None;
        }
        let mut cap = cfg_read8(&d, 0x34) & 0xfc;
        let mut guard = 0;
        while cap != 0 && guard < 48 {
            guard += 1;
            let cap_vndr = cfg_read8(&d, cap as u16);
            let next = cfg_read8(&d, cap as u16 + 1) & 0xfc;
            if cap_vndr == CAP_VENDOR {
                let cfg_type = cfg_read8(&d, cap as u16 + 3);
                let bar = cfg_read8(&d, cap as u16 + 4);
                let offset = cfg_read32_at(&d, cap as u16 + 8);
                let bar_phys = d.bar(bar);
                if bar_phys != 0 {
                    let virt = crate::mm::map_mmio(bar_phys, 0x4000) + offset as u64;
                    match cfg_type {
                        CFG_COMMON => common = Region { virt, extra: 0 },
                        CFG_NOTIFY => {
                            let mult = cfg_read32_at(&d, cap as u16 + 16);
                            notify = Region { virt, extra: mult };
                        }
                        CFG_DEVICE => device = Region { virt, extra: 0 },
                        _ => {}
                    }
                }
            }
            cap = next;
        }
        if common.virt == 0 || notify.virt == 0 {
            return None;
        }

        // SAFETY: `common`/`notify`/`device` are HHDM-mapped virtio BAR regions.
        unsafe {
            let c = common.virt;
            // Reset, then ACK + DRIVER.
            w8(c + DEVICE_STATUS, 0);
            while r8(c + DEVICE_STATUS) != 0 {
                core::hint::spin_loop();
            }
            w8(c + DEVICE_STATUS, S_ACK);
            w8(c + DEVICE_STATUS, S_ACK | S_DRIVER);

            // Feature negotiation: ack MAC (low word) + VERSION_1 (bit 32).
            w32(c + DEVICE_FEATURE_SELECT, 0);
            let lo = r32(c + DEVICE_FEATURE);
            w32(c + DRIVER_FEATURE_SELECT, 0);
            w32(c + DRIVER_FEATURE, lo & VIRTIO_NET_F_MAC);
            w32(c + DRIVER_FEATURE_SELECT, 1);
            w32(c + DRIVER_FEATURE, 1); // VIRTIO_F_VERSION_1
            w8(c + DEVICE_STATUS, S_ACK | S_DRIVER | S_FEATURES_OK);
            if r8(c + DEVICE_STATUS) & S_FEATURES_OK == 0 {
                return None;
            }

            let mut rx = setup_queue(c, notify, 0)?;
            let tx = setup_queue(c, notify, 1)?;

            // MAC from device config (offset 0), byte-wise (BAR memory, any width OK).
            let mut mac = [0u8; 6];
            if device.virt != 0 {
                for (i, m) in mac.iter_mut().enumerate() {
                    *m = r8(device.virt + i as u64);
                }
            }

            w8(c + DEVICE_STATUS, S_ACK | S_DRIVER | S_FEATURES_OK | S_DRIVER_OK);

            // Pre-post RX buffers.
            for i in 0..rx.qsize {
                rx.post(i, BUFSZ as u32, true);
            }

            crate::ktrace::log_fmt(format_args!(
                "virtio-net-pci: up (dev {:04x}), MAC {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}, q{}",
                d.device, mac[0], mac[1], mac[2], mac[3], mac[4], mac[5], rx.qsize
            ));
            Some(VirtioNetPci { mac, rx, tx })
        }
    }
}

/// Configure virtqueue `idx` via the common config: publish split-ring physical
/// addresses, enable it, and compute its notify doorbell address.
unsafe fn setup_queue(c: u64, notify: Region, idx: u16) -> Option<Queue> {
    unsafe {
        w16(c + QUEUE_SELECT, idx);
        let qmax = r16(c + QUEUE_SIZE);
        if qmax == 0 {
            return None;
        }
        let qsize = (QSIZE as u16).min(qmax);
        w16(c + QUEUE_SIZE, qsize);
        let qs = qsize as usize;

        let (desc_phys, desc) = crate::mm::alloc_dma(qs * 16)?;
        let (avail_phys, avail) = crate::mm::alloc_dma(6 + qs * 2)?;
        let (used_phys, used) = crate::mm::alloc_dma(6 + qs * 8)?;
        let (bufs_phys, bufs) = crate::mm::alloc_dma(qs * BUFSZ)?;

        w64(c + QUEUE_DESC, desc_phys);
        w64(c + QUEUE_DRIVER, avail_phys);
        w64(c + QUEUE_DEVICE, used_phys);

        let notify_off = r16(c + QUEUE_NOTIFY_OFF);
        let notify_addr = notify.virt + notify_off as u64 * notify.extra as u64;

        w16(c + QUEUE_ENABLE, 1);

        Some(Queue {
            qsize,
            desc,
            avail,
            used,
            bufs,
            bufs_phys,
            notify: notify_addr,
            vqn: idx,
            avail_idx: 0,
            used_last: 0,
        })
    }
}

impl NetDevice for VirtioNetPci {
    fn mac(&self) -> [u8; 6] {
        self.mac
    }

    fn receive(&mut self, out: &mut [u8]) -> Option<usize> {
        // SAFETY: rings/buffers are the live DMA regions from `init`.
        unsafe {
            while self.tx.pop_used().is_some() {}
            let (id, len) = self.rx.pop_used()?;
            let frame = (len as usize).saturating_sub(NET_HDR_LEN);
            let n = frame.min(out.len());
            if n > 0 {
                core::ptr::copy_nonoverlapping((self.rx.buf(id) + NET_HDR_LEN as u64) as *const u8, out.as_mut_ptr(), n);
            }
            self.rx.post(id, BUFSZ as u32, true);
            Some(n)
        }
    }

    fn transmit(&mut self, frame: &[u8]) {
        if frame.len() + NET_HDR_LEN > BUFSZ {
            return;
        }
        let i = self.tx.avail_idx % self.tx.qsize;
        // SAFETY: `tx.buf(i)` is BUFSZ bytes of our DMA pool.
        unsafe {
            let b = self.tx.buf(i);
            core::ptr::write_bytes(b as *mut u8, 0, NET_HDR_LEN); // zero virtio_net_hdr
            core::ptr::copy_nonoverlapping(frame.as_ptr(), (b + NET_HDR_LEN as u64) as *mut u8, frame.len());
            self.tx.post(i, (NET_HDR_LEN + frame.len()) as u32, false);
        }
    }
}
