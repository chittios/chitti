//! **One transport trait over virtio-mmio and virtio-pci**, so a device driver
//! is written once and binds on both arches and both boot paths.
//!
//! This exists because of the dual-architecture standing rule. The two new
//! devices ([`super::super::virtio_9p`], [`super::super::virtio_serial`]) have
//! to reach the guest on:
//!
//! * **x86_64** — always PCI (there is no virtio-mmio window on `q35`/`pc`).
//! * **aarch64 `-kernel`** — always **mmio**: the dev loop boots with no ACPI,
//!   so `crate::pci` has no ECAM and PCI discovery finds nothing at all (the
//!   same reason `virtio-gpu` binds only on the UEFI path).
//! * **aarch64 UEFI** — PCI, via the stub's ACPI.
//!
//! Binding only one of them would make a feature exist on one arch and not the
//! other, which is the divergence the rule forbids — so discovery tries mmio
//! slots first and falls back to PCI, and the driver above never knows which it
//! got.
//!
//! **Legacy is supported, not skipped.** QEMU's `virtio-mmio` defaults to
//! `force-legacy=true` on several machine types, so the version-1 transport is
//! not a historical curiosity here — it is what the aarch64 dev loop may
//! actually present. Its one structural difference is that the whole queue must
//! be a single page-aligned region addressed by `QUEUE_PFN`, which is why
//! [`super::layout::VirtqLayout`] carves all three rings out of one allocation
//! and takes the used-ring alignment as a parameter.

use super::layout::VirtqLayout;
use core::ptr::{read_volatile, write_volatile};

// The PCI config surface is `crate::arch::x86_64::pci` (legacy I/O ports) on
// x86 and `crate::pci` (ECAM) on aarch64. The `read32`/`write32`/`for_each`/
// `PciDevice::bar` subset used below is identical on both, which is what lets
// one transport serve both arches — the same aliasing `crate::net::pci` uses.
#[cfg(target_arch = "aarch64")]
use crate::pci::{self, PciDevice};
#[cfg(target_arch = "x86_64")]
use crate::arch::x86_64::pci::{self, PciDevice};

/// Largest queue index either device uses. virtio-serial with one extra port
/// needs six (port0 rx/tx, control rx/tx, port1 rx/tx); virtio-9p needs one.
pub const MAX_QUEUES: usize = 8;

// --- device status bits (virtio 1.0 §2.1) ---
pub const S_ACK: u8 = 1;
pub const S_DRIVER: u8 = 2;
pub const S_DRIVER_OK: u8 = 4;
pub const S_FEATURES_OK: u8 = 8;

/// `VIRTIO_F_VERSION_1` — feature bit 32, the "this is not legacy" flag.
pub const F_VERSION_1: u64 = 1 << 32;

// --- virtio device ids ---
pub const ID_CONSOLE: u32 = 3;
pub const ID_9P: u32 = 9;

/// PCI device id for a device: transitional (`0x1000+id`) and modern
/// (`0x1040+id`) both appear in the wild, so a driver must match either.
pub fn pci_ids(device_id: u32) -> [u16; 2] {
    [0x1000 + device_id as u16, 0x1040 + device_id as u16]
}

const VIRTIO_PCI_VENDOR: u16 = 0x1af4;

/// What a device driver needs of its transport.
pub trait Transport {
    /// Reset, then take the device through ACK + DRIVER.
    fn begin(&mut self);
    /// Features the device offers.
    fn device_features(&mut self) -> u64;
    /// Accept `features`, then set FEATURES_OK. `false` if the device rejected
    /// the set (a modern device that will not run with what we accepted).
    fn accept_features(&mut self, features: u64) -> bool;
    /// Largest supported size of queue `q`; 0 means the queue does not exist.
    fn queue_max(&mut self, q: u16) -> u16;
    /// Program queue `q`'s rings. `region` is the **physical** base of the one
    /// contiguous allocation `layout` describes.
    fn queue_set(&mut self, q: u16, region: u64, layout: &VirtqLayout);
    /// Tell the device queue `q` has new buffers.
    fn notify(&self, q: u16);
    /// Finish bring-up (DRIVER_OK).
    fn ready(&mut self);
    /// Read a byte of the device-specific config space.
    fn cfg_read8(&self, off: usize) -> u8;
    /// Used-ring alignment this transport requires.
    fn used_align(&self) -> usize;
}

/// Read `len` bytes of device config into `out` (little-endian fields are read
/// bytewise so an unaligned or narrow config register is never widened — some
/// transports fault on a wide access to a byte-sized field).
pub fn cfg_read(t: &dyn Transport, off: usize, out: &mut [u8]) {
    for (i, b) in out.iter_mut().enumerate() {
        *b = t.cfg_read8(off + i);
    }
}

/// Read a little-endian `u16` from device config.
pub fn cfg_read16(t: &dyn Transport, off: usize) -> u16 {
    let mut b = [0u8; 2];
    cfg_read(t, off, &mut b);
    u16::from_le_bytes(b)
}

// =====================================================================
// virtio-mmio
// =====================================================================

// Register offsets (virtio 1.0 §4.2.2).
const M_MAGIC: usize = 0x000;
const M_VERSION: usize = 0x004;
const M_DEVICE_ID: usize = 0x008;
const M_DEVICE_FEATURES: usize = 0x010;
const M_DEVICE_FEATURES_SEL: usize = 0x014;
const M_DRIVER_FEATURES: usize = 0x020;
const M_DRIVER_FEATURES_SEL: usize = 0x024;
const M_GUEST_PAGE_SIZE: usize = 0x028; // legacy only
const M_QUEUE_SEL: usize = 0x030;
const M_QUEUE_NUM_MAX: usize = 0x034;
const M_QUEUE_NUM: usize = 0x038;
const M_QUEUE_ALIGN: usize = 0x03c; // legacy only
const M_QUEUE_PFN: usize = 0x040; // legacy only
const M_QUEUE_READY: usize = 0x044; // modern only
const M_QUEUE_NOTIFY: usize = 0x050;
const M_STATUS: usize = 0x070;
const M_QUEUE_DESC_LOW: usize = 0x080;
const M_QUEUE_DESC_HIGH: usize = 0x084;
const M_QUEUE_DRIVER_LOW: usize = 0x090;
const M_QUEUE_DRIVER_HIGH: usize = 0x094;
const M_QUEUE_DEVICE_LOW: usize = 0x0a0;
const M_QUEUE_DEVICE_HIGH: usize = 0x0a4;
const M_CONFIG: usize = 0x100;

const MMIO_MAGIC: u32 = 0x7472_6976; // "virt"
/// QEMU `virt` places virtio-mmio transports in this window.
const MMIO_BASE: usize = 0x0a00_0000;
const MMIO_STRIDE: usize = 0x200;
const MMIO_SLOTS: usize = 32;
/// The page size the legacy transport addresses `QUEUE_PFN` in.
const LEGACY_PAGE: usize = 4096;

use super::barrier;

/// A virtio device behind the mmio transport.
pub struct MmioTransport {
    base: usize,
    /// 1 = legacy, 2 = modern. The two differ in how a queue is programmed.
    version: u32,
}

impl MmioTransport {
    /// Whether this transport speaks the legacy (version 1) protocol.
    pub fn is_legacy(&self) -> bool {
        self.version == 1
    }

    /// The mmio base this transport claimed, so a second driver for the same
    /// device id can skip it.
    pub fn base(&self) -> usize {
        self.base
    }

    unsafe fn r(&self, off: usize) -> u32 {
        // SAFETY: caller-established virtio-mmio block; registers are 32-bit.
        unsafe { read_volatile((self.base + off) as *const u32) }
    }
    unsafe fn w(&self, off: usize, v: u32) {
        // SAFETY: as above.
        unsafe { write_volatile((self.base + off) as *mut u32, v) };
    }

    /// Find the `n`th virtio-mmio slot presenting `device_id`, skipping any
    /// base in `skip` (a second device of the same kind already claimed).
    pub fn find(device_id: u32, skip: &[usize]) -> Option<MmioTransport> {
        for slot in 0..MMIO_SLOTS {
            let base = MMIO_BASE + slot * MMIO_STRIDE;
            if skip.contains(&base) {
                continue;
            }
            // SAFETY: probing the fixed virtio-mmio window with 32-bit reads.
            // An absent slot reads as zeroes, which the magic check rejects.
            unsafe {
                if read_volatile((base + M_MAGIC) as *const u32) != MMIO_MAGIC {
                    continue;
                }
                let version = read_volatile((base + M_VERSION) as *const u32);
                if version != 1 && version != 2 {
                    continue;
                }
                if read_volatile((base + M_DEVICE_ID) as *const u32) != device_id {
                    continue;
                }
                return Some(MmioTransport { base, version });
            }
        }
        None
    }
}

impl Transport for MmioTransport {
    fn begin(&mut self) {
        // SAFETY: `self.base` is a confirmed virtio-mmio block.
        unsafe {
            self.w(M_STATUS, 0);
            self.w(M_STATUS, S_ACK as u32);
            self.w(M_STATUS, (S_ACK | S_DRIVER) as u32);
            if self.is_legacy() {
                // The legacy transport addresses queues in guest pages.
                self.w(M_GUEST_PAGE_SIZE, LEGACY_PAGE as u32);
            }
        }
    }

    fn device_features(&mut self) -> u64 {
        // SAFETY: register access on a confirmed block.
        unsafe {
            self.w(M_DEVICE_FEATURES_SEL, 0);
            let lo = self.r(M_DEVICE_FEATURES) as u64;
            self.w(M_DEVICE_FEATURES_SEL, 1);
            let hi = self.r(M_DEVICE_FEATURES) as u64;
            lo | (hi << 32)
        }
    }

    fn accept_features(&mut self, features: u64) -> bool {
        // SAFETY: register access on a confirmed block.
        unsafe {
            self.w(M_DRIVER_FEATURES_SEL, 0);
            self.w(M_DRIVER_FEATURES, features as u32);
            if !self.is_legacy() {
                self.w(M_DRIVER_FEATURES_SEL, 1);
                self.w(M_DRIVER_FEATURES, (features >> 32) as u32);
                self.w(M_STATUS, (S_ACK | S_DRIVER | S_FEATURES_OK) as u32);
                return self.r(M_STATUS) as u8 & S_FEATURES_OK != 0;
            }
            // A legacy device has no FEATURES_OK handshake to fail.
            true
        }
    }

    fn queue_max(&mut self, q: u16) -> u16 {
        // SAFETY: register access on a confirmed block.
        unsafe {
            self.w(M_QUEUE_SEL, q as u32);
            self.r(M_QUEUE_NUM_MAX) as u16
        }
    }

    fn queue_set(&mut self, q: u16, region: u64, layout: &VirtqLayout) {
        // SAFETY: register access on a confirmed block; `region` is a physical
        // address of an allocation at least `layout.total` bytes long.
        unsafe {
            self.w(M_QUEUE_SEL, q as u32);
            self.w(M_QUEUE_NUM, layout.qsize as u32);
            if self.is_legacy() {
                self.w(M_QUEUE_ALIGN, LEGACY_PAGE as u32);
                // PFN is the region base in guest pages — which is why the
                // three rings must share one page-aligned allocation.
                self.w(M_QUEUE_PFN, (region / LEGACY_PAGE as u64) as u32);
            } else {
                let d = region + layout.desc as u64;
                let a = region + layout.avail as u64;
                let u = region + layout.used as u64;
                self.w(M_QUEUE_DESC_LOW, d as u32);
                self.w(M_QUEUE_DESC_HIGH, (d >> 32) as u32);
                self.w(M_QUEUE_DRIVER_LOW, a as u32);
                self.w(M_QUEUE_DRIVER_HIGH, (a >> 32) as u32);
                self.w(M_QUEUE_DEVICE_LOW, u as u32);
                self.w(M_QUEUE_DEVICE_HIGH, (u >> 32) as u32);
                self.w(M_QUEUE_READY, 1);
            }
        }
    }

    fn notify(&self, q: u16) {
        barrier();
        // SAFETY: register access on a confirmed block.
        unsafe { self.w(M_QUEUE_NOTIFY, q as u32) };
    }

    fn ready(&mut self) {
        // SAFETY: register access on a confirmed block.
        unsafe {
            let bits = if self.is_legacy() {
                S_ACK | S_DRIVER | S_DRIVER_OK
            } else {
                S_ACK | S_DRIVER | S_FEATURES_OK | S_DRIVER_OK
            };
            self.w(M_STATUS, bits as u32);
        }
    }

    fn cfg_read8(&self, off: usize) -> u8 {
        // SAFETY: device config space starts at 0x100 on this transport.
        unsafe { read_volatile((self.base + M_CONFIG + off) as *const u8) }
    }

    fn used_align(&self) -> usize {
        if self.is_legacy() {
            LEGACY_PAGE
        } else {
            4
        }
    }
}

// =====================================================================
// virtio-pci (modern)
// =====================================================================

// virtio-pci capability types.
const CFG_COMMON: u8 = 1;
const CFG_NOTIFY: u8 = 2;
const CFG_DEVICE: u8 = 4;

// common-cfg field offsets (virtio_pci_common_cfg).
const P_DEVICE_FEATURE_SEL: u64 = 0x00;
const P_DEVICE_FEATURE: u64 = 0x04;
const P_DRIVER_FEATURE_SEL: u64 = 0x08;
const P_DRIVER_FEATURE: u64 = 0x0c;
const P_DEVICE_STATUS: u64 = 0x14;
const P_QUEUE_SELECT: u64 = 0x16;
const P_QUEUE_SIZE: u64 = 0x18;
const P_QUEUE_ENABLE: u64 = 0x1c;
const P_QUEUE_NOTIFY_OFF: u64 = 0x1e;
const P_QUEUE_DESC: u64 = 0x20;
const P_QUEUE_DRIVER: u64 = 0x28;
const P_QUEUE_DEVICE: u64 = 0x30;

/// A virtio device behind the modern virtio-pci transport.
pub struct PciTransport {
    common: u64,
    notify_base: u64,
    notify_mult: u64,
    devcfg: u64,
    /// Per-queue notify addresses, filled in as queues are programmed. The
    /// notify offset is a *per queue* register, so it has to be read while that
    /// queue is selected — caching it is what lets `notify` take `&self`.
    notify_addr: [u64; MAX_QUEUES],
}

/// PCI vendor capability id — virtio's structures hang off the vendor-specific
/// capability, not a dedicated one.
const CAP_VENDOR: u8 = 0x09;

/// Read a byte of config space. Config cycles are dword-wide on the x86 port
/// path, so a byte is extracted rather than fetched.
fn cfg8(d: &PciDevice, off: u16) -> u8 {
    (pci::read32(d.bus, d.dev, d.func, off & !3) >> ((off & 3) * 8)) as u8
}

/// Read a dword of config space at an arbitrary (dword-aligned) offset.
fn cfg32(d: &PciDevice, off: u16) -> u32 {
    pci::read32(d.bus, d.dev, d.func, off)
}

impl PciTransport {
    /// Find the `n`th PCI function presenting virtio `device_id` and map its
    /// capability regions.
    pub fn find(device_id: u32, n: usize) -> Option<PciTransport> {
        let ids = pci_ids(device_id);
        let mut seen = 0usize;
        let mut found: Option<PciDevice> = None;
        pci::for_each(&mut |d: PciDevice| {
            if d.vendor == VIRTIO_PCI_VENDOR && ids.contains(&d.device) {
                if seen == n {
                    found = Some(d);
                    return false;
                }
                seen += 1;
            }
            true
        });
        Self::from_device(found?)
    }

    fn from_device(d: PciDevice) -> Option<PciTransport> {
        d.enable_bus_master();
        // Capabilities-list-present bit (status register 0x06, bit 4).
        if cfg32(&d, 0x04) & (1 << 20) == 0 {
            return None;
        }
        let (mut common, mut notify_base, mut devcfg) = (0u64, 0u64, 0u64);
        let mut notify_mult = 0u64;
        let mut cap = cfg8(&d, 0x34) & 0xfc;
        let mut guard = 0;
        // Bounded walk: a malformed or circular capability list must not spin
        // the boot, and 48 is well past any real device's list length.
        while cap != 0 && guard < 48 {
            guard += 1;
            let vndr = cfg8(&d, cap as u16);
            let next = cfg8(&d, cap as u16 + 1) & 0xfc;
            if vndr == CAP_VENDOR {
                let cfg_type = cfg8(&d, cap as u16 + 3);
                let bar = cfg8(&d, cap as u16 + 4);
                let offset = cfg32(&d, cap as u16 + 8) as u64;
                let bar_phys = d.bar(bar);
                if bar_phys != 0 {
                    // x86 reaches a BAR through the HHDM; aarch64 identity-maps
                    // it. `map_mmio` is the arch-neutral form of both.
                    let virt = crate::mm::map_mmio(bar_phys, 0x4000) + offset;
                    match cfg_type {
                        CFG_COMMON => common = virt,
                        CFG_NOTIFY => {
                            notify_base = virt;
                            notify_mult = cfg32(&d, cap as u16 + 16) as u64;
                        }
                        CFG_DEVICE => devcfg = virt,
                        _ => {}
                    }
                }
            }
            cap = next;
        }
        if common == 0 || notify_base == 0 {
            return None;
        }
        Some(PciTransport { common, notify_base, notify_mult, devcfg, notify_addr: [0; MAX_QUEUES] })
    }
}

// SAFETY-note helpers: every access below targets BAR-mapped MMIO located by
// the virtio capability walk above, using the widths virtio 1.0 §4.1.4 defines
// for each field.
unsafe fn pr8(a: u64) -> u8 {
    unsafe { read_volatile(a as *const u8) }
}
unsafe fn pr16(a: u64) -> u16 {
    unsafe { read_volatile(a as *const u16) }
}
unsafe fn pr32(a: u64) -> u32 {
    unsafe { read_volatile(a as *const u32) }
}
unsafe fn pw8(a: u64, v: u8) {
    unsafe { write_volatile(a as *mut u8, v) };
}
unsafe fn pw16(a: u64, v: u16) {
    unsafe { write_volatile(a as *mut u16, v) };
}
unsafe fn pw32(a: u64, v: u32) {
    unsafe { write_volatile(a as *mut u32, v) };
}
unsafe fn pw64(a: u64, v: u64) {
    unsafe { write_volatile(a as *mut u64, v) };
}

impl Transport for PciTransport {
    fn begin(&mut self) {
        // SAFETY: BAR-mapped virtio common config.
        unsafe {
            pw8(self.common + P_DEVICE_STATUS, 0);
            // Bounded, per the standing rule that no wait may be unbounded: a
            // device that never clears status is broken, and hanging here would
            // be indistinguishable from a hung boot.
            let mut spins = 0u32;
            while pr8(self.common + P_DEVICE_STATUS) != 0 && spins < 1_000_000 {
                core::hint::spin_loop();
                spins += 1;
            }
            pw8(self.common + P_DEVICE_STATUS, S_ACK);
            pw8(self.common + P_DEVICE_STATUS, S_ACK | S_DRIVER);
        }
    }

    fn device_features(&mut self) -> u64 {
        // SAFETY: BAR-mapped virtio common config.
        unsafe {
            pw32(self.common + P_DEVICE_FEATURE_SEL, 0);
            let lo = pr32(self.common + P_DEVICE_FEATURE) as u64;
            pw32(self.common + P_DEVICE_FEATURE_SEL, 1);
            let hi = pr32(self.common + P_DEVICE_FEATURE) as u64;
            lo | (hi << 32)
        }
    }

    fn accept_features(&mut self, features: u64) -> bool {
        // SAFETY: BAR-mapped virtio common config.
        unsafe {
            pw32(self.common + P_DRIVER_FEATURE_SEL, 0);
            pw32(self.common + P_DRIVER_FEATURE, features as u32);
            pw32(self.common + P_DRIVER_FEATURE_SEL, 1);
            pw32(self.common + P_DRIVER_FEATURE, (features >> 32) as u32);
            pw8(self.common + P_DEVICE_STATUS, S_ACK | S_DRIVER | S_FEATURES_OK);
            pr8(self.common + P_DEVICE_STATUS) & S_FEATURES_OK != 0
        }
    }

    fn queue_max(&mut self, q: u16) -> u16 {
        // SAFETY: BAR-mapped virtio common config.
        unsafe {
            pw16(self.common + P_QUEUE_SELECT, q);
            pr16(self.common + P_QUEUE_SIZE)
        }
    }

    fn queue_set(&mut self, q: u16, region: u64, layout: &VirtqLayout) {
        // SAFETY: BAR-mapped virtio common config; `region` is the physical
        // base of an allocation at least `layout.total` bytes long.
        unsafe {
            pw16(self.common + P_QUEUE_SELECT, q);
            pw16(self.common + P_QUEUE_SIZE, layout.qsize);
            pw64(self.common + P_QUEUE_DESC, region + layout.desc as u64);
            pw64(self.common + P_QUEUE_DRIVER, region + layout.avail as u64);
            pw64(self.common + P_QUEUE_DEVICE, region + layout.used as u64);
            let off = pr16(self.common + P_QUEUE_NOTIFY_OFF) as u64;
            if (q as usize) < MAX_QUEUES {
                self.notify_addr[q as usize] = self.notify_base + off * self.notify_mult;
            }
            pw16(self.common + P_QUEUE_ENABLE, 1);
        }
    }

    fn notify(&self, q: u16) {
        let Some(&addr) = self.notify_addr.get(q as usize) else {
            return;
        };
        if addr == 0 {
            return;
        }
        barrier();
        // SAFETY: `addr` was computed from this device's notify capability
        // while queue `q` was selected.
        unsafe { pw16(addr, q) };
    }

    fn ready(&mut self) {
        // SAFETY: BAR-mapped virtio common config.
        unsafe { pw8(self.common + P_DEVICE_STATUS, S_ACK | S_DRIVER | S_FEATURES_OK | S_DRIVER_OK) };
    }

    fn cfg_read8(&self, off: usize) -> u8 {
        // SAFETY: BAR-mapped device-specific config region.
        unsafe { pr8(self.devcfg + off as u64) }
    }

    fn used_align(&self) -> usize {
        4
    }
}

/// Locate a virtio device of `device_id` on whichever transport this machine
/// presents — mmio first (the aarch64 `-kernel` dev loop has nothing else),
/// then PCI (x86 always, aarch64 UEFI).
///
/// `skip_mmio` lets a caller pass over mmio bases already claimed by another
/// instance of the same device kind.
pub fn find_any(device_id: u32, n: usize, skip_mmio: &[usize]) -> Option<alloc::boxed::Box<dyn Transport>> {
    if let Some(t) = MmioTransport::find(device_id, skip_mmio) {
        return Some(alloc::boxed::Box::new(t));
    }
    PciTransport::find(device_id, n).map(|t| alloc::boxed::Box::new(t) as alloc::boxed::Box<dyn Transport>)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn pci_ids_cover_transitional_and_modern() {
        // A transitional device answers 0x1000+id, a modern one 0x1040+id; a
        // driver that matched only one would miss half the QEMU command lines
        // that can present it.
        assert_eq!(pci_ids(ID_9P), [0x1009, 0x1049]);
        assert_eq!(pci_ids(ID_CONSOLE), [0x1003, 0x1043]);
    }

    #[test_case]
    fn version_1_is_the_only_flag_we_require() {
        // Bit 32, not bit 0 — a driver that sets bit 0 accepts whatever
        // device-specific feature happens to live there instead.
        assert_eq!(F_VERSION_1, 0x1_0000_0000);
        assert_eq!(F_VERSION_1.trailing_zeros(), 32);
    }
}
