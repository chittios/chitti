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
    let addr = cfg_addr(bus, dev, func, off);
    // Apple's APCIE ECAM **external-aborts** a config read to an unlinked
    // secondary bus instead of returning all-ones (standard PCI behaviour). A
    // raw read there is a fatal data abort (ESR 0x96000010, FAR = ecam +
    // bus<<20), which a plain bus scan (`for_each` over 0..=bus_end) will hit as
    // soon as any bus beyond the linked ones is walked. Route through the
    // recoverable probe so an absent/unlinked bus reads as `0xffffffff` (absent).
    #[cfg(all(target_arch = "aarch64", not(test)))]
    {
        return crate::arch::aarch64::probe_read32(addr).unwrap_or(0xffff_ffff);
    }
    // SAFETY: ECAM is identity-mapped Device memory (mapped by mmu::init);
    // config space is 4 KiB per function.
    #[cfg(not(all(target_arch = "aarch64", not(test))))]
    unsafe {
        core::ptr::read_volatile(addr as *const u32)
    }
}
pub fn write32(bus: u8, dev: u8, func: u8, off: u16, v: u32) {
    unsafe { core::ptr::write_volatile(cfg_addr(bus, dev, func, off) as *mut u32, v) };
}
pub fn read16(bus: u8, dev: u8, func: u8, off: u16) -> u16 {
    (read32(bus, dev, func, off & !3) >> ((off & 2) * 8)) as u16
}
pub fn write16(bus: u8, dev: u8, func: u8, off: u16, v: u16) {
    let aligned = off & !3;
    let shift = (off & 2) * 8;
    let cur = read32(bus, dev, func, aligned);
    let mask = !(0xffffu32 << shift);
    write32(bus, dev, func, aligned, (cur & mask) | ((v as u32) << shift));
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

    /// Find PCI Power Management capability and force **D0**.
    /// Devices left in D3 after reset may still answer config cycles but
    /// external-abort every BAR MMIO until brought back to D0.
    pub fn set_power_d0(&self) {
        // Status bit 4 = cap list present; pointer @0x34.
        if read16(self.bus, self.dev, self.func, 0x06) & 0x10 == 0 {
            return;
        }
        let mut cap = read8(self.bus, self.dev, self.func, 0x34) as u16 & 0xfc;
        for _ in 0..48 {
            if cap == 0 || cap == 0xff {
                break;
            }
            let id = read8(self.bus, self.dev, self.func, cap);
            if id == 0x01 {
                // PMCSR at cap+4: bits 1:0 = power state (0 = D0).
                let pmcsr = read16(self.bus, self.dev, self.func, cap + 4);
                let state = pmcsr & 0x3;
                if state != 0 {
                    write16(
                        self.bus,
                        self.dev,
                        self.func,
                        cap + 4,
                        (pmcsr & !0x3) | 0x8000, // D0 + clear PME status
                    );
                    crate::ktrace::log_fmt(format_args!(
                        "pci: {:02x}:{:02x}.{} PM D{state}→D0 (pmcsr was {pmcsr:#06x})",
                        self.bus, self.dev, self.func
                    ));
                } else {
                    crate::ktrace::log_fmt(format_args!(
                        "pci: {:02x}:{:02x}.{} already D0 (pmcsr={pmcsr:#06x})",
                        self.bus, self.dev, self.func
                    ));
                }
                return;
            }
            cap = read8(self.bus, self.dev, self.func, cap + 1) as u16 & 0xfc;
        }
    }

    /// Program a single 64-bit MEM BAR index (`i` even) to `base`, preserving
    /// 64-bit type bits. Used by the Apple BAR-window probe.
    pub fn program_bar64(&self, i: u8, base: u64, type_bits: u32) {
        let off = 0x10 + i as u16 * 4;
        write32(
            self.bus,
            self.dev,
            self.func,
            off,
            (base as u32 & 0xffff_fff0) | (type_bits & 0xf),
        );
        write32(self.bus, self.dev, self.func, off + 4, (base >> 32) as u32);
    }

    /// Size BAR `i` (write `!0`, read mask). Returns `(size, type_bits, is_64)`.
    pub fn size_bar(&self, i: u8) -> Option<(u64, u32, bool)> {
        let off = 0x10 + i as u16 * 4;
        let cmd = read32(self.bus, self.dev, self.func, 0x04);
        write32(self.bus, self.dev, self.func, 0x04, cmd & !0b10);
        write32(self.bus, self.dev, self.func, off, 0xffff_ffff);
        let mask = read32(self.bus, self.dev, self.func, off);
        if mask == 0 || mask == 0xffff_ffff || mask & 1 != 0 {
            write32(self.bus, self.dev, self.func, 0x04, cmd);
            return None;
        }
        let is_64 = (mask >> 1) & 0x3 == 0x2;
        let type_bits = mask & 0xf;
        let size = if is_64 {
            write32(self.bus, self.dev, self.func, off + 4, 0xffff_ffff);
            let mask_hi = read32(self.bus, self.dev, self.func, off + 4);
            let size_lo = (!(mask & 0xffff_fff0)).wrapping_add(1) as u64;
            if mask_hi == 0xffff_ffff {
                size_lo
            } else {
                let full = (mask_hi as u64) << 32 | (mask as u64 & 0xffff_fff0);
                (!full).wrapping_add(1)
            }
        } else {
            (!(mask & 0xffff_fff0)).wrapping_add(1) as u64
        };
        write32(self.bus, self.dev, self.func, 0x04, cmd);
        Some((size.max(0x1000).next_power_of_two(), type_bits, is_64))
    }

    /// Decode BAR `i`'s base address (memory BARs only), 64-bit aware.
    /// Returns the **bus** address programmed in the BAR (not CPU PA — on
    /// Apple Silicon translate with the host's `ranges`).
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

    /// Size + assign all memory BARs starting at `mem32_base` (typically
    /// `0xc000_0000` under an Apple root-port 32-bit window). 64-bit BARs use
    /// `mem64_base` (typically `0x6_a000_0000`). Returns `(next_32, next_64)`.
    ///
    /// Firmware/iBoot often leave endpoint BARs at 0; without this, `bar(0)`
    /// stays zero even after a live link. Follows the standard PCI sizing
    /// dance (write `!0`, read mask, program base).
    pub fn assign_mem_bars(&self, mut mem32_base: u64, mut mem64_base: u64) -> (u64, u64) {
        // Disable MEM decode while reprogramming.
        let cmd = read32(self.bus, self.dev, self.func, 0x04);
        write32(self.bus, self.dev, self.func, 0x04, cmd & !0b10);

        let mut i = 0u8;
        while i < 6 {
            let off = 0x10 + i as u16 * 4;
            let orig = read32(self.bus, self.dev, self.func, off);
            // Size the BAR.
            write32(self.bus, self.dev, self.func, off, 0xffff_ffff);
            let mask = read32(self.bus, self.dev, self.func, off);
            if mask == 0 || mask == 0xffff_ffff {
                write32(self.bus, self.dev, self.func, off, orig);
                i += 1;
                continue;
            }
            if mask & 1 != 0 {
                // I/O BAR — leave alone on aarch64.
                write32(self.bus, self.dev, self.func, off, orig);
                i += 1;
                continue;
            }
            let is_64 = (mask >> 1) & 0x3 == 0x2;
            let pref = (mask >> 3) & 1 != 0;
            let size_lo = (!(mask & 0xffff_fff0)).wrapping_add(1);

            // Preserve type/prefetch bits from the size mask — NOT from `orig`.
            // When firmware left the BAR at 0, `orig & 0xf == 0` would clear the
            // 64-bit type field and the device would only decode the low 32 bits
            // (e.g. 0xa0000000 instead of 0x6a0000000) → every MMIO external-aborts.
            let type_bits = mask & 0xf;

            if is_64 {
                write32(self.bus, self.dev, self.func, off + 4, 0xffff_ffff);
                let mask_hi = read32(self.bus, self.dev, self.func, off + 4);
                let size = if mask_hi == 0xffff_ffff {
                    size_lo as u64
                } else {
                    let full = (mask_hi as u64) << 32 | (mask as u64 & 0xffff_fff0);
                    (!full).wrapping_add(1)
                };
                // Size must be power-of-two for the align mask below.
                let size = size.max(0x1000).next_power_of_two();
                // Always place 64-bit BARs in the high identity window
                // (Apple t8112 ranges: PCI 0x6a00_0000_0 ↔ CPU 0x6a00_0000_0).
                let _ = (pref, mem32_base, orig);
                let b = (mem64_base + size - 1) & !(size - 1);
                mem64_base = b + size;
                write32(
                    self.bus,
                    self.dev,
                    self.func,
                    off,
                    (b as u32 & 0xffff_fff0) | type_bits,
                );
                write32(self.bus, self.dev, self.func, off + 4, (b >> 32) as u32);
                // Read back to confirm the device latched a 64-bit decode.
                let rb_lo = read32(self.bus, self.dev, self.func, off);
                let rb_hi = read32(self.bus, self.dev, self.func, off + 4);
                crate::ktrace::log_fmt(format_args!(
                    "pci: {:02x}:{:02x}.{} BAR{i} 64-bit size={size:#x} -> {b:#x} (type={type_bits:#x} rb={rb_hi:08x}_{rb_lo:08x})",
                    self.bus, self.dev, self.func
                ));
                i += 2;
            } else {
                let size = (size_lo as u64).max(0x1000).next_power_of_two();
                let b = (mem32_base + size - 1) & !(size - 1);
                mem32_base = b + size;
                write32(
                    self.bus,
                    self.dev,
                    self.func,
                    off,
                    (b as u32 & 0xffff_fff0) | type_bits,
                );
                let rb = read32(self.bus, self.dev, self.func, off);
                crate::ktrace::log_fmt(format_args!(
                    "pci: {:02x}:{:02x}.{} BAR{i} 32-bit size={size:#x} -> {b:#x} (type={type_bits:#x} rb={rb:#010x})",
                    self.bus, self.dev, self.func
                ));
                i += 1;
            }
        }

        // Re-enable MEM + bus master.
        write32(self.bus, self.dev, self.func, 0x04, cmd | 0b110);
        (mem32_base, mem64_base)
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

/// Visit every present PCI function in bus/device/function order; `f` returns
/// `false` to stop the scan early. Bounded by the MCFG's `bus_end` (unlike the
/// x86 legacy path, ECAM has a firmware-declared end bus). Mirror of
/// `crate::arch::x86_64::pci::for_each`.
pub fn for_each(f: &mut dyn FnMut(PciDevice) -> bool) {
    if ecam_base() == 0 {
        return;
    }
    let bus_end = BUS_END.with(|b| *b);
    for bus in 0..=bus_end {
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

/// The `n`-th function matching class `(base, sub, prog_if)`, in scan order —
/// e.g. an xHCI USB controller is `(0x0c, 0x03, 0x30)`. Lets a probe path find
/// the second AHCI HBA or the second NIC on a machine that has two.
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

/// Find the first function matching a class code `(base, sub, prog_if)`.
pub fn find_class(base: u8, sub: u8, prog_if: u8) -> Option<PciDevice> {
    find_class_nth(base, sub, prog_if, 0)
}

/// The `n`-th function matching class `base`+subclass `sub`, **ignoring
/// prog_if** — audio controllers report varying prog_if across hypervisors
/// (VirtualBox HDA in particular), so device drivers match on subclass.
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

/// Print **every** PCI function to the chat pane (`serial_println!`) — the
/// `/lspci` shell command. Shows vendor:device, the full class triple, and the
/// ECAM base, so an unrecognised audio controller on a VM is directly visible.
pub fn dump_all() {
    let base = ecam_base();
    crate::serial_println!("pci> ECAM base {:#x}", base);
    if base == 0 {
        crate::serial_println!("pci> PCIe not discovered (no ACPI MCFG) — using virtio-mmio");
        return;
    }
    let bus_end = BUS_END.with(|b| *b);
    let mut n = 0;
    for bus in 0..=bus_end {
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
    crate::serial_println!("pci> {n} device(s) on buses 0..={bus_end}");
}

/// Log every function of PCI base class `base` (diagnostic — used when audio
/// autodetect finds nothing, so the actual VM device layout is visible).
pub fn log_class(base: u8) {
    if ecam_base() == 0 {
        crate::ktrace::log("pci", "log_class: ECAM base is 0 (PCIe not discovered)");
        return;
    }
    let bus_end = BUS_END.with(|b| *b);
    for bus in 0..=bus_end {
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
