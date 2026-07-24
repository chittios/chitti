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

/// Everything needed to drive an ACPI **S5 (soft-off)** transition.
///
/// The values come from two different tables, which is the awkward part: the
/// FADT gives the *register* (`PM1a_CNT_BLK`), but the *value* to write
/// (`SLP_TYPa`) lives in the DSDT's `\_S5_` package and is normally obtained by
/// evaluating AML.
#[derive(Clone, Copy, Debug)]
pub struct SleepInfo {
    /// `PM1a_CNT` I/O port (system-I/O space; the only space this implements).
    pub pm1a_cnt: u16,
    /// `PM1b_CNT`, or 0 when the platform has no second block.
    pub pm1b_cnt: u16,
    /// Sleep type for S5, from the DSDT `\_S5_` package.
    pub slp_typa: u8,
    pub slp_typb: u8,
    /// `SMI_CMD` port and the `ACPI_ENABLE` value, for taking ownership from
    /// firmware when `SCI_EN` is clear (legacy BIOS boots; UEFI boots hand ACPI
    /// over already enabled).
    pub smi_cmd: u32,
    pub acpi_enable: u8,
}

/// `PM1x_CNT` bit 13: start the sleep transition.
pub const SLP_EN: u16 = 1 << 13;
/// `PM1x_CNT` bit 0: ACPI mode is enabled (SCI_EN).
pub const SCI_EN: u16 = 1 << 0;

/// Locate the RSDP, validating the signature rather than trusting the caller's
/// idea of what kind of address it has.
///
/// `candidates` are tried in order — a Limine RSDP response is a physical address
/// on newer revisions and an HHDM-relative virtual one on older builds, so the
/// caller passes both interpretations and this picks whichever actually has
/// `"RSD PTR "` at it. Returns the usable (mapped) address.
pub fn find_rsdp(candidates: &[u64]) -> Option<u64> {
    for &c in candidates {
        if c == 0 {
            continue;
        }
        let p = c as *const u8;
        let mut sig = [0u8; 8];
        for (i, b) in sig.iter_mut().enumerate() {
            // SAFETY: reading 8 bytes at a candidate RSDP address. Callers only
            // pass addresses the bootloader reported or the legacy BIOS window,
            // both of which are mapped.
            *b = unsafe { *p.add(i) };
        }
        if &sig == b"RSD PTR " {
            return Some(c);
        }
    }
    None
}

/// Walk `rsdp` -> XSDT -> **FADT** ("FACP") and the DSDT it names, and return the
/// S5 sleep parameters. `None` if either table is missing or the DSDT has no
/// `\_S5_` package (a machine that genuinely cannot soft-off this way).
///
/// `phys_to_virt` maps a physical table address for reading — identity on the
/// aarch64 map, HHDM on x86.
pub fn s5_from_rsdp(rsdp: u64, phys_to_virt: impl Fn(u64) -> u64) -> Option<SleepInfo> {
    let f = find_table(rsdp, b"FACP")?;
    // FADT: SMI_CMD@48, ACPI_ENABLE@52, PM1a_CNT_BLK@64, PM1b_CNT_BLK@68,
    // DSDT@40, X_DSDT@148, X_PM1a_CNT_BLK@180 (a 12-byte Generic Address
    // Structure: space@0, width@1, offset@2, access@3, address@4).
    let len = le32(f, 4) as usize;
    let smi_cmd = le32(f, 48);
    let acpi_enable = unsafe { *f.add(52) };
    let mut pm1a = le32(f, 64) as u64;
    let mut pm1b = le32(f, 68) as u64;
    // Prefer the extended GAS forms when present and in system-I/O space (1).
    if len >= 192 {
        let space = unsafe { *f.add(180) };
        let addr = le64(f, 184);
        if addr != 0 && space == 1 {
            pm1a = addr;
        }
        let space_b = unsafe { *f.add(192) };
        let addr_b = le64(f, 196);
        if len >= 204 && addr_b != 0 && space_b == 1 {
            pm1b = addr_b;
        }
    }
    if pm1a == 0 || pm1a > u16::MAX as u64 {
        return None; // not an I/O-port PM1a_CNT — memory-mapped PM blocks unsupported
    }
    // DSDT, for the `\_S5_` package.
    let dsdt_phys = if len >= 156 && le64(f, 148) != 0 { le64(f, 148) } else { le32(f, 40) as u64 };
    if dsdt_phys == 0 {
        return None;
    }
    let dsdt = phys_to_virt(dsdt_phys) as *const u8;
    let (slp_typa, slp_typb) = parse_s5_package(dsdt)?;
    Some(SleepInfo {
        pm1a_cnt: pm1a as u16,
        pm1b_cnt: if pm1b <= u16::MAX as u64 { pm1b as u16 } else { 0 },
        slp_typa,
        slp_typb,
        smi_cmd,
        acpi_enable,
    })
}

/// Find `\_S5_` in the DSDT and decode the two sleep-type values out of its
/// package, **without** an AML interpreter.
///
/// This is the well-worn bytecode-scan shortcut: locate the `_S5_` name, confirm
/// it is introduced by a `NameOp` (0x08, optionally preceded by a root-scope
/// backslash), then step over `PackageOp` (0x12), its `PkgLength` (whose leading
/// two bits give the encoded byte count) and the element count, and read the
/// first two elements. Each element is `ZeroOp` (0x00), `OneOp` (0x01) or a
/// `BytePrefix` (0x0A) followed by the value.
///
/// It is a heuristic, not evaluation: a DSDT that computes `_S5_` dynamically, or
/// hides it behind a method, will not match — and then soft-off simply reports
/// unavailable rather than writing a wrong sleep type.
fn parse_s5_package(dsdt: *const u8) -> Option<(u8, u8)> {
    // SAFETY: `dsdt` points at a mapped ACPI table whose own header declares its
    // length at offset 4; every read below stays inside that length.
    let len = le32(dsdt, 4) as usize;
    if len < 36 || len > 0x40_0000 {
        return None; // implausible: not a DSDT we should scan
    }
    let byte = |i: usize| -> u8 { unsafe { *dsdt.add(i) } };
    let mut i = 36; // skip the table header
    while i + 4 < len {
        if byte(i) == b'_' && byte(i + 1) == b'S' && byte(i + 2) == b'5' && byte(i + 3) == b'_' {
            // Must be a NameOp definition: `08 '_S5_'`, or `08 '\' '_S5_'`.
            let named = (i >= 1 && byte(i - 1) == 0x08)
                || (i >= 2 && byte(i - 2) == 0x08 && byte(i - 1) == b'\\');
            if !named {
                i += 1;
                continue;
            }
            let mut p = i + 4; // just past the name
            if p >= len || byte(p) != 0x12 {
                i += 1;
                continue; // not a package after all
            }
            p += 1; // PackageOp
            if p >= len {
                return None;
            }
            // PkgLength: the top two bits say how many extra length bytes follow.
            p += ((byte(p) >> 6) as usize) + 1;
            if p >= len {
                return None;
            }
            p += 1; // NumElements
            // Decode up to two elements as sleep types.
            let mut vals = [0u8; 2];
            for slot in vals.iter_mut() {
                if p >= len {
                    break;
                }
                match byte(p) {
                    0x00 => {
                        *slot = 0;
                        p += 1;
                    }
                    0x01 => {
                        *slot = 1;
                        p += 1;
                    }
                    0x0a => {
                        if p + 1 >= len {
                            break;
                        }
                        *slot = byte(p + 1);
                        p += 2;
                    }
                    _ => break, // something we don't decode; keep what we have
                }
            }
            return Some((vals[0], vals[1]));
        }
        i += 1;
    }
    None
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

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    /// Build a minimal fake DSDT: 36-byte header (with a correct length field)
    /// followed by `body` bytes.
    fn dsdt(body: &[u8]) -> Vec<u8> {
        let mut t = vec![0u8; 36];
        t[0..4].copy_from_slice(b"DSDT");
        t.extend_from_slice(body);
        let len = t.len() as u32;
        t[4..8].copy_from_slice(&len.to_le_bytes());
        t
    }

    #[test_case]
    fn s5_package_byte_prefix_values() {
        // NameOp '_S5_' PackageOp PkgLen NumElem BytePrefix 5 BytePrefix 5
        let t = dsdt(&[0x08, b'_', b'S', b'5', b'_', 0x12, 0x06, 0x02, 0x0a, 0x05, 0x0a, 0x05]);
        assert_eq!(parse_s5_package(t.as_ptr()), Some((5, 5)));
    }

    #[test_case]
    fn s5_package_root_scoped_and_zero_one_ops() {
        // The other common encoding: `08 '\' '_S5_'`, elements ZeroOp / OneOp.
        let t = dsdt(&[0x08, b'\\', b'_', b'S', b'5', b'_', 0x12, 0x05, 0x02, 0x00, 0x01]);
        assert_eq!(parse_s5_package(t.as_ptr()), Some((0, 1)));
    }

    #[test_case]
    fn s5_package_multibyte_pkglength() {
        // PkgLength with the top bits set encodes extra length bytes, which must
        // be stepped over or the element read lands on the wrong byte.
        let t = dsdt(&[0x08, b'_', b'S', b'5', b'_', 0x12, 0x41, 0x00, 0x02, 0x0a, 0x07, 0x0a, 0x07]);
        assert_eq!(parse_s5_package(t.as_ptr()), Some((7, 7)));
    }

    #[test_case]
    fn s5_name_without_nameop_is_not_matched() {
        // `_S5_` appearing as a plain string / method reference must NOT be
        // decoded — writing a sleep type read out of unrelated bytes is worse
        // than reporting soft-off unavailable.
        let t = dsdt(&[0xff, b'_', b'S', b'5', b'_', 0x12, 0x06, 0x02, 0x0a, 0x05, 0x0a, 0x05]);
        assert_eq!(parse_s5_package(t.as_ptr()), None);
    }

    #[test_case]
    fn s5_absent_or_truncated_returns_none() {
        assert_eq!(parse_s5_package(dsdt(&[]).as_ptr()), None);
        // NameOp + name but the package is cut off at the table end.
        assert_eq!(parse_s5_package(dsdt(&[0x08, b'_', b'S', b'5', b'_']).as_ptr()), None);
        // Name present, but followed by something that isn't a PackageOp.
        let t = dsdt(&[0x08, b'_', b'S', b'5', b'_', 0x14, 0x06, 0x02]);
        assert_eq!(parse_s5_package(t.as_ptr()), None);
    }

    #[test_case]
    fn find_rsdp_picks_the_candidate_with_the_signature() {
        let good: [u8; 8] = *b"RSD PTR ";
        let bad: [u8; 8] = *b"NOPENOPE";
        let g = good.as_ptr() as u64;
        let b = bad.as_ptr() as u64;
        // Order matters: the first *valid* candidate wins, junk is skipped.
        assert_eq!(find_rsdp(&[b, g]), Some(g));
        assert_eq!(find_rsdp(&[g, b]), Some(g));
        assert_eq!(find_rsdp(&[0, b]), None);
        assert_eq!(find_rsdp(&[]), None);
    }
}
