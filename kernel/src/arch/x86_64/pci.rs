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

/// Visit every present PCI function in bus/device/function order; `f` returns
/// `false` to stop the scan early. Buses 0..=255 are probed exhaustively rather
/// than by walking the bridge topology: real UEFI firmware (and every
/// hypervisor) assigns bus numbers and programs the bridges before handing over,
/// so a device behind a PCIe root port — where a laptop's NVMe almost always
/// sits — answers config reads on its own bus number directly. Absent buses read
/// back all-ones and cost one port access each.
pub fn for_each(f: &mut dyn FnMut(PciDevice) -> bool) {
    for bus in 0u16..256 {
        let bus = bus as u8;
        for dev in 0u8..32 {
            for func in 0u8..8 {
                let id = read32(bus, dev, func, 0x00);
                let v = (id & 0xffff) as u16;
                if v == 0xffff {
                    if func == 0 {
                        break; // no function 0 → nothing at this device slot
                    }
                    continue;
                }
                if !f(PciDevice { bus, dev, func, vendor: v, device: (id >> 16) as u16 }) {
                    return;
                }
            }
        }
    }
}

/// The class triple `(base, sub, prog_if)` of a located function (config 0x08,
/// bits 31:8).
pub fn class_of(d: &PciDevice) -> (u8, u8, u8) {
    let class = read32(d.bus, d.dev, d.func, 0x08);
    (((class >> 24) & 0xff) as u8, ((class >> 16) & 0xff) as u8, ((class >> 8) & 0xff) as u8)
}

/// The `n`-th function matching class `(base, sub, prog_if)`, in scan order.
/// Real machines routinely have more than one controller of a kind — two AHCI
/// HBAs, an NVMe *and* a SATA disk, a laptop with both an Intel and a Realtek
/// NIC — so every probe path that used to take "the first match" iterates this.
pub fn find_class_nth(base: u8, sub: u8, prog_if: u8, n: usize) -> Option<PciDevice> {
    let mut seen = 0usize;
    let mut found = None;
    for_each(&mut |d| {
        if class_of(&d) == (base, sub, prog_if) {
            if seen == n {
                found = Some(d);
                return false;
            }
            seen += 1;
        }
        true
    });
    found
}

/// Find the first function matching a class code `(base, sub, prog_if)`. Same
/// signature as the ECAM `crate::pci::find_class` so the shared drivers' arch
/// wrappers look identical.
pub fn find_class(base: u8, sub: u8, prog_if: u8) -> Option<PciDevice> {
    find_class_nth(base, sub, prog_if, 0)
}

/// The `n`-th function matching class `base`+subclass `sub`, ignoring prog_if
/// (audio controllers vary it across hypervisors — VirtualBox HDA in particular).
pub fn find_class_sub_nth(base: u8, sub: u8, n: usize) -> Option<PciDevice> {
    let mut seen = 0usize;
    let mut found = None;
    for_each(&mut |d| {
        let (b, s, _) = class_of(&d);
        if b == base && s == sub {
            if seen == n {
                found = Some(d);
                return false;
            }
            seen += 1;
        }
        true
    });
    found
}

/// Find the first function matching class `base`+subclass `sub`.
pub fn find_class_sub(base: u8, sub: u8) -> Option<PciDevice> {
    find_class_sub_nth(base, sub, 0)
}

/// Print every PCI function to the chat pane — the `/lspci` shell command.
pub fn dump_all() {
    crate::serial_println!("pci> legacy port config (0xCF8/0xCFC)");
    let mut n = 0;
    for_each(&mut |d| {
        let (b, s, p) = class_of(&d);
        crate::serial_println!(
            "pci> {:02x}:{:02x}.{} {:04x}:{:04x} class {b:02x}:{s:02x}:{p:02x}",
            d.bus,
            d.dev,
            d.func,
            d.vendor,
            d.device
        );
        n += 1;
        true
    });
    crate::serial_println!("pci> {n} device(s)");
}

/// Log every function of PCI base class `base` (diagnostic for audio autodetect).
pub fn log_class(base: u8) {
    for_each(&mut |d| {
        let (b, s, p) = class_of(&d);
        if b == base {
            crate::ktrace::log_fmt(format_args!(
                "pci: {:04x}:{:04x} class {b:02x}:{s:02x}:{p:02x} at {}:{}.{}",
                d.vendor, d.device, d.bus, d.dev, d.func
            ));
        }
        true
    });
}
