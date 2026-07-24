//! Minimal ACPI table walking — just enough to discover, from firmware tables
//! (not hardcoded per-machine addresses), the PCIe ECAM base (**MCFG**) and the
//! console UART base (**SPCR**). The stub captures the RSDP from the UEFI
//! configuration table and hands it to the kernel; from there: RSDP → XSDT →
//! {MCFG, SPCR}. This is what lets aarch64 run on platforms whose device map
//! differs from QEMU `virt` (e.g. VirtualBox puts the PL011 at 0xFFDDF000, not
//! QEMU's 0x09000000).
//!
//! Read-only, no allocation, no interpreter — table headers + one field each.

fn le16(p: *const u8, o: usize) -> u16 {
    unsafe { u16::from_le_bytes([*p.add(o), *p.add(o + 1)]) }
}
fn le32(p: *const u8, o: usize) -> u32 {
    unsafe { u32::from_le_bytes([*p.add(o), *p.add(o + 1), *p.add(o + 2), *p.add(o + 3)]) }
}
fn le64(p: *const u8, o: usize) -> u64 {
    let mut v = [0u8; 8];
    for (i, b) in v.iter_mut().enumerate() {
        *b = unsafe { *p.add(o + i) };
    }
    u64::from_le_bytes(v)
}

/// A PCIe ECAM segment discovered from MCFG.
#[derive(Clone, Copy)]
pub struct EcamSegment {
    pub base: u64,
    pub bus_start: u8,
    pub bus_end: u8,
}

/// Find the ACPI table with 4-byte signature `sig` by walking `rsdp` → XSDT
/// (or RSDT), returning a pointer to its header. Every address is identity-
/// mapped physical (the caller ensures the map covers the tables, which live in
/// low reserved RAM). `None` if no valid ACPI or no such table.
fn find_table(rsdp: u64, sig: &[u8; 4]) -> Option<*const u8> {
    if rsdp == 0 {
        return None;
    }
    let r = rsdp as *const u8;
    // RSDP: "RSD PTR " signature; revision @15; XsdtAddress @24 (8 bytes).
    let mut rsig = [0u8; 8];
    for (i, b) in rsig.iter_mut().enumerate() {
        *b = unsafe { *r.add(i) };
    }
    if &rsig != b"RSD PTR " {
        return None;
    }
    let revision = unsafe { *r.add(15) };
    // XSDT (rev>=2) preferred; else RSDT (32-bit entries).
    let (table_ptr, entry_size) = if revision >= 2 { (le64(r, 24), 8usize) } else { (le32(r, 16) as u64, 4usize) };
    if table_ptr == 0 {
        return None;
    }
    let t = table_ptr as *const u8;
    // System description table header: length @4 (u32). Entries follow the
    // 36-byte header, each a pointer to another table.
    let len = le32(t, 4) as usize;
    let n = len.saturating_sub(36) / entry_size;
    for i in 0..n {
        let off = 36 + i * entry_size;
        let tbl = if entry_size == 8 { le64(t, off) } else { le32(t, off) as u64 };
        let p = tbl as *const u8;
        let mut s = [0u8; 4];
        for (k, b) in s.iter_mut().enumerate() {
            *b = unsafe { *p.add(k) };
        }
        if &s == sig {
            return Some(p);
        }
    }
    None
}

/// Walk `rsdp` → XSDT → MCFG and return the first ECAM segment, if any.
pub fn ecam_from_rsdp(rsdp: u64) -> Option<EcamSegment> {
    let p = find_table(rsdp, b"MCFG")?;
    // MCFG: 44-byte header (incl. 8 reserved), then allocation entries of 16
    // bytes: base@0 (u64), segment@8 (u16), bus_start@10, bus_end@11.
    let base = le64(p, 44);
    let bus_start = unsafe { *p.add(44 + 10) };
    let bus_end = unsafe { *p.add(44 + 11) };
    let _ = le16(p, 44 + 8);
    Some(EcamSegment { base, bus_start, bus_end })
}

/// GICv3 register bases discovered from ACPI — for an ARM platform booted by UEFI
/// with **no device tree** (VirtualBox-ARM, UTM, real SBSA servers/laptops).
pub struct GicInfo {
    /// Distributor base, from the MADT GICD structure (type 0x0D).
    pub gicd: u64,
    /// This CPU's redistributor base.
    pub gicr: u64,
    /// GIC architecture version reported by the GICD structure (3 = GICv3).
    pub version: u8,
}

/// Walk `rsdp` → XSDT → **MADT** ("APIC") and return the GICv3 distributor and
/// redistributor bases for the CPU whose `MPIDR_EL1` is `mpidr`.
///
/// This is the ARM-SBSA counterpart to reading the device tree's `arm,gic-v3`
/// node. Without it a UEFI-booted ARM machine has no way to locate the GIC, and
/// hardcoding QEMU `virt`'s `0x0800_0000` would poke whatever happens to live
/// there on a real server.
///
/// Redistributor resolution follows the ACPI spec's own precedence:
/// 1. the **GICC** structure (type 0x0C) whose `MPIDR` field matches this CPU —
///    its `GicrBaseAddress` is exactly this core's redistributor frame;
/// 2. failing an MPIDR match, the first GICC with a non-zero `GicrBaseAddress`;
/// 3. failing that, the base of the **GICR** discovery range (type 0x0F), whose
///    first frame belongs to the boot CPU on every platform that uses this form.
///
/// `None` if there is no MADT, no GICD, or no redistributor by any of those
/// routes — the caller then stays on cooperative scheduling rather than guessing.
pub fn gic_from_rsdp(rsdp: u64, mpidr: u64) -> Option<GicInfo> {
    let p = find_table(rsdp, b"APIC")?;
    let total = le32(p, 4) as usize;
    // MADT header is 36 bytes + LocalApicAddress(u32) + Flags(u32); the
    // interrupt-controller structures start at 44. Each is {type u8, length u8}.
    let mut off = 44usize;
    let mut gicd = 0u64;
    let mut version = 0u8;
    let mut gicr_matched = 0u64; // GICC with our MPIDR
    let mut gicr_first = 0u64; // first GICC with a redistributor
    let mut gicr_range = 0u64; // GICR discovery range base
    while off + 2 <= total {
        // SAFETY: `p` points at a mapped ACPI table; `off + 2 <= total` and
        // `total` is the table's own declared length.
        let (ty, len) = unsafe { (*p.add(off), *p.add(off + 1) as usize) };
        if len < 2 || off + len > total {
            break; // malformed length would otherwise loop or read past the table
        }
        match ty {
            0x0c => {
                // GICC: GicrBaseAddress @64, MPIDR @72 (both u64).
                if len >= 80 {
                    let base = le64(p, off + 64);
                    // MPIDR comparison uses the affinity bits only (bits 63:40
                    // are reserved/flags in MPIDR_EL1).
                    if base != 0 {
                        if le64(p, off + 72) & 0xff_00ff_ffff == mpidr & 0xff_00ff_ffff {
                            gicr_matched = base;
                        }
                        if gicr_first == 0 {
                            gicr_first = base;
                        }
                    }
                }
            }
            0x0d => {
                // GICD: PhysicalBaseAddress @8 (u64), GicVersion @20 (u8).
                if len >= 24 {
                    gicd = le64(p, off + 8);
                    // SAFETY: `off + 20 < off + len <= total`, inside the table.
                    version = unsafe { *p.add(off + 20) };
                }
            }
            0x0f => {
                // GICR: DiscoveryRangeBaseAddress @4 (u64).
                if len >= 16 && gicr_range == 0 {
                    gicr_range = le64(p, off + 4);
                }
            }
            _ => {}
        }
        off += len;
    }
    let gicr = if gicr_matched != 0 {
        gicr_matched
    } else if gicr_first != 0 {
        gicr_first
    } else {
        gicr_range
    };
    if gicd == 0 || gicr == 0 {
        return None;
    }
    Some(GicInfo { gicd, gicr, version })
}

/// Walk `rsdp` → XSDT → **SPCR** (Serial Port Console Redirection) and return
/// the console UART's `(base_address, interface_type)`. Interface type 0x03 is
/// ARM PL011; 0x0e is ARM SBSA (PL011-compatible DR/FR layout). This is how the
/// UART base is found on a platform whose map differs from QEMU `virt` (e.g.
/// VirtualBox's PL011 at 0xFFDDF000). `None` if no SPCR or a non-MMIO address.
pub fn uart_from_rsdp(rsdp: u64) -> Option<(u64, u8)> {
    let p = find_table(rsdp, b"SPCR")?;
    // SPCR: 36-byte header, then Interface Type @36 (1 byte), 3 reserved, then a
    // 12-byte Generic Address Structure @40: AddressSpaceId@0 (0 = system
    // memory), ..., Address@4 (8 bytes).
    let iface = unsafe { *p.add(36) };
    let addr_space = unsafe { *p.add(40) };
    let base = le64(p, 44);
    // Only a memory-mapped (addr_space 0) UART with a non-zero base is usable.
    if addr_space != 0 || base == 0 {
        return None;
    }
    Some((base, iface))
}
