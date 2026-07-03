//! Minimal ACPI table walking — just enough to discover the PCIe ECAM base
//! from the **MCFG** table, so PCIe is found the real-world way (firmware
//! tables) rather than a hardcoded per-machine address. The stub captures the
//! RSDP from the UEFI configuration table and hands it to the kernel; from
//! there: RSDP → XSDT → MCFG → ECAM base + bus range.
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

/// Walk `rsdp` → XSDT → MCFG and return the first ECAM segment, if any. Every
/// address here is identity-mapped physical (the caller ensures the map covers
/// the tables, which live in low reserved RAM). `None` if no valid ACPI/MCFG.
pub fn ecam_from_rsdp(rsdp: u64) -> Option<EcamSegment> {
    if rsdp == 0 {
        return None;
    }
    let r = rsdp as *const u8;
    // RSDP: "RSD PTR " signature; revision @15; XsdtAddress @24 (8 bytes).
    let mut sig = [0u8; 8];
    for (i, b) in sig.iter_mut().enumerate() {
        *b = unsafe { *r.add(i) };
    }
    if &sig != b"RSD PTR " {
        return None;
    }
    let revision = unsafe { *r.add(15) };
    // XSDT (rev>=2) preferred; else RSDT (32-bit entries).
    let (table_ptr, entry_size) = if revision >= 2 {
        (le64(r, 24), 8usize)
    } else {
        (le32(r, 16) as u64, 4usize)
    };
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
        if &s == b"MCFG" {
            // MCFG: 44-byte header (incl. 8 reserved), then allocation entries
            // of 16 bytes: base@0 (u64), segment@8 (u16), bus_start@10, bus_end@11.
            let base = le64(p, 44);
            let bus_start = unsafe { *p.add(44 + 10) };
            let bus_end = unsafe { *p.add(44 + 11) };
            let _ = le16(p, 44 + 8);
            return Some(EcamSegment { base, bus_start, bus_end });
        }
    }
    None
}
