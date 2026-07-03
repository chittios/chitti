//! Minimal **PCIe (ECAM) access + enumeration** — the real-hardware device bus,
//! replacing QEMU's `virtio-mmio` toy transport. The ECAM base comes from ACPI
//! MCFG (`crate::acpi`), so this works on any ACPI PCIe platform (QEMU virt's
//! GPEX bridge, cloud/hypervisor ARM, SBSA hardware). Config space is a flat
//! MMIO window: `ecam + (bus<<20) + (dev<<15) + (fn<<12) + off`.
//!
//! Scope: config read/write, device scan, BAR decode (32/64-bit), and the
//! vendor-specific capability walk virtio-pci needs. Bus mastering is enabled
//! on claim. No MSI/interrupts — drivers here poll.

use crate::mm::Locked;

static ECAM: Locked<u64> = Locked::new(0);
static BUS_END: Locked<u8> = Locked::new(0);

/// Record the ECAM base + last bus (from ACPI MCFG). Call once at boot.
pub fn init(base: u64, bus_end: u8) {
    ECAM.with(|e| *e = base);
    BUS_END.with(|b| *b = bus_end);
}

pub fn ecam_base() -> u64 {
    ECAM.with(|e| *e)
}

fn cfg_addr(bus: u8, dev: u8, func: u8, off: u16) -> u64 {
    ecam_base() + ((bus as u64) << 20) + ((dev as u64) << 15) + ((func as u64) << 12) + off as u64
}

pub fn read32(bus: u8, dev: u8, func: u8, off: u16) -> u32 {
    // SAFETY: ECAM is identity-mapped Device memory (mapped by mmu::init);
    // config space is 4 KiB per function.
    unsafe { core::ptr::read_volatile(cfg_addr(bus, dev, func, off) as *const u32) }
}
pub fn write32(bus: u8, dev: u8, func: u8, off: u16, v: u32) {
    unsafe { core::ptr::write_volatile(cfg_addr(bus, dev, func, off) as *mut u32, v) };
}
pub fn read16(bus: u8, dev: u8, func: u8, off: u16) -> u16 {
    (read32(bus, dev, func, off & !3) >> ((off & 2) * 8)) as u16
}
pub fn read8(bus: u8, dev: u8, func: u8, off: u16) -> u8 {
    (read32(bus, dev, func, off & !3) >> ((off & 3) * 8)) as u8
}

/// A located PCI function.
#[derive(Clone, Copy)]
pub struct PciDevice {
    pub bus: u8,
    pub dev: u8,
    pub func: u8,
    pub vendor: u16,
    pub device: u16,
}

impl PciDevice {
    /// Enable memory space + bus-master DMA (COMMAND register bits 1,2).
    pub fn enable_bus_master(&self) {
        let cmd = read32(self.bus, self.dev, self.func, 0x04);
        write32(self.bus, self.dev, self.func, 0x04, cmd | 0b110);
    }

    /// Decode BAR `i`'s base address (memory BARs only), 64-bit aware.
    pub fn bar(&self, i: u8) -> u64 {
        let off = 0x10 + i as u16 * 4;
        let lo = read32(self.bus, self.dev, self.func, off);
        if lo & 1 != 0 {
            return 0; // I/O BAR — not used on aarch64
        }
        let base = (lo & 0xffff_fff0) as u64;
        if (lo >> 1) & 0x3 == 0x2 {
            // 64-bit BAR: high half in the next slot.
            let hi = read32(self.bus, self.dev, self.func, off + 4) as u64;
            base | (hi << 32)
        } else {
            base
        }
    }

    /// Walk the capability list for a vendor-specific (0x09) cap matching
    /// `cfg_type` at cap+3 (virtio-pci structure type), returning its config
    /// offset. Used to find virtio's common/notify/isr/device windows.
    pub fn find_virtio_cap(&self, cfg_type: u8) -> Option<u16> {
        // Status register bit 4 => capability list present; pointer @0x34.
        if read16(self.bus, self.dev, self.func, 0x06) & 0x10 == 0 {
            return None;
        }
        let mut cap = read8(self.bus, self.dev, self.func, 0x34) as u16 & 0xfc;
        let mut guard = 0;
        while cap != 0 && guard < 48 {
            let id = read8(self.bus, self.dev, self.func, cap);
            if id == 0x09 && read8(self.bus, self.dev, self.func, cap + 3) == cfg_type {
                return Some(cap);
            }
            cap = read8(self.bus, self.dev, self.func, cap + 1) as u16 & 0xfc;
            guard += 1;
        }
        None
    }
}

/// Find the `n`-th (0-based) function matching `(vendor, device)`, scanning all
/// buses in the ECAM range. `device` accepts either the transitional or modern
/// id via the two-element `devices` list.
pub fn find_nth(vendor: u16, devices: &[u16], n: usize) -> Option<PciDevice> {
    if ecam_base() == 0 {
        return None;
    }
    let bus_end = BUS_END.with(|b| *b);
    let mut seen = 0usize;
    for bus in 0..=bus_end {
        for dev in 0u8..32 {
            for func in 0u8..8 {
                let id = read32(bus, dev, func, 0x00);
                let v = (id & 0xffff) as u16;
                let d = (id >> 16) as u16;
                if v == 0xffff {
                    if func == 0 {
                        break; // no function 0 => no device
                    }
                    continue;
                }
                if v == vendor && devices.contains(&d) {
                    if seen == n {
                        return Some(PciDevice { bus, dev, func, vendor: v, device: d });
                    }
                    seen += 1;
                }
            }
        }
    }
    None
}
