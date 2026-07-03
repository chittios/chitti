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
