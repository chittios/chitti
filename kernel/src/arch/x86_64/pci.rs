//! Minimal **legacy PCI config access + enumeration** (ports 0xCF8/0xCFC) — the
//! x86 counterpart of the aarch64 ECAM `crate::pci`. x86 under Limine/QEMU has
//! no ACPI MCFG wired up, so config space is reached through the classic I/O
//! ports rather than a memory window. Same surface the real-hardware drivers
//! need: class-code scan, 32/64-bit memory-BAR decode, and bus-master enable.
//!
//! Scope is deliberately small (the first 256 bytes of config space, which is
//! all the legacy ports expose) — enough for storage/USB controller discovery.

use crate::arch::x86_64::port::{inl, outl};

fn cfg_addr(bus: u8, dev: u8, func: u8, off: u16) -> u32 {
    0x8000_0000 | ((bus as u32) << 16) | ((dev as u32) << 11) | ((func as u32) << 8) | (off as u32 & 0xfc)
}

pub fn read32(bus: u8, dev: u8, func: u8, off: u16) -> u32 {
    // SAFETY: 0xCF8/0xCFC are the standard PCI configuration ports.
    unsafe {
        outl(0xcf8, cfg_addr(bus, dev, func, off));
        inl(0xcfc)
    }
}

pub fn write32(bus: u8, dev: u8, func: u8, off: u16, v: u32) {
    // SAFETY: as `read32`.
    unsafe {
        outl(0xcf8, cfg_addr(bus, dev, func, off));
        outl(0xcfc, v);
    }
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
            return 0; // I/O BAR — NVMe/AHCI registers are memory-mapped
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
}

/// Find the first function matching a class code `(base, sub, prog_if)` (config
/// offset 0x08, bits 31:8). Scans buses 0..=255. Same signature as the ECAM
/// `crate::pci::find_class` so the shared drivers' arch wrappers look identical.
pub fn find_class(base: u8, sub: u8, prog_if: u8) -> Option<PciDevice> {
    for bus in 0u16..256 {
        let bus = bus as u8;
        for dev in 0u8..32 {
            for func in 0u8..8 {
                let id = read32(bus, dev, func, 0x00);
                let v = (id & 0xffff) as u16;
                if v == 0xffff {
                    if func == 0 {
                        break;
                    }
                    continue;
                }
                let class = read32(bus, dev, func, 0x08);
                if ((class >> 24) & 0xff) as u8 == base && ((class >> 16) & 0xff) as u8 == sub && ((class >> 8) & 0xff) as u8 == prog_if {
                    return Some(PciDevice { bus, dev, func, vendor: v, device: (id >> 16) as u16 });
                }
            }
        }
    }
    None
}

/// Find the first function matching class `base`+subclass `sub`, ignoring
/// prog_if (audio controllers vary it across hypervisors — VirtualBox HDA in
/// particular).
pub fn find_class_sub(base: u8, sub: u8) -> Option<PciDevice> {
    for bus in 0u16..256 {
        let bus = bus as u8;
        for dev in 0u8..32 {
            for func in 0u8..8 {
                let id = read32(bus, dev, func, 0x00);
                let v = (id & 0xffff) as u16;
                if v == 0xffff {
                    if func == 0 {
                        break;
                    }
                    continue;
                }
                let class = read32(bus, dev, func, 0x08);
                if ((class >> 24) & 0xff) as u8 == base && ((class >> 16) & 0xff) as u8 == sub {
                    return Some(PciDevice { bus, dev, func, vendor: v, device: (id >> 16) as u16 });
                }
            }
        }
    }
    None
}

/// Print every PCI function to the chat pane — the `/lspci` shell command.
pub fn dump_all() {
    crate::serial_println!("pci> legacy port config (0xCF8/0xCFC)");
    let mut n = 0;
    for bus in 0u16..256 {
        let bus = bus as u8;
        for dev in 0u8..32 {
            for func in 0u8..8 {
                let id = read32(bus, dev, func, 0x00);
                let v = (id & 0xffff) as u16;
                if v == 0xffff {
                    if func == 0 {
                        break;
                    }
                    continue;
                }
                let class = read32(bus, dev, func, 0x08);
                crate::serial_println!(
                    "pci> {bus:02x}:{dev:02x}.{func} {:04x}:{:04x} class {:02x}:{:02x}:{:02x}",
                    v,
                    (id >> 16) as u16,
                    (class >> 24) & 0xff,
                    (class >> 16) & 0xff,
                    (class >> 8) & 0xff
                );
                n += 1;
            }
        }
    }
    crate::serial_println!("pci> {n} device(s)");
}

/// Log every function of PCI base class `base` (diagnostic for audio autodetect).
pub fn log_class(base: u8) {
    for bus in 0u16..256 {
        let bus = bus as u8;
        for dev in 0u8..32 {
            for func in 0u8..8 {
                let id = read32(bus, dev, func, 0x00);
                let v = (id & 0xffff) as u16;
                if v == 0xffff {
                    if func == 0 {
                        break;
                    }
                    continue;
                }
                let class = read32(bus, dev, func, 0x08);
                if ((class >> 24) & 0xff) as u8 == base {
                    crate::ktrace::log_fmt(format_args!(
                        "pci: {:04x}:{:04x} class {:02x}:{:02x}:{:02x} at {bus}:{dev}.{func}",
                        v,
                        (id >> 16) as u16,
                        (class >> 24) & 0xff,
                        (class >> 16) & 0xff,
                        (class >> 8) & 0xff
                    ));
                }
            }
        }
    }
}
