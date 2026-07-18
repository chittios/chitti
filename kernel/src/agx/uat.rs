//! Apple **AGX UAT** page-table encoding — the *pure* half of the GPU MMU
//! (Milestone 2). The UAT is "just regular ARMv8 page tables, shared between the
//! gfx-asc firmware and the AGX hardware" (m1n1 `proxyclient/m1n1/hw/uat.py`):
//! 16 KiB granule, a 2-entry L0 of **TTBR**-format descriptors (TTBR0 for low
//! VAs, TTBR1 for high/firmware VAs, selected by VA bit 47), then L1/L2 table
//! descriptors and L3 leaf **page** descriptors.
//!
//! Per-context TTBR pairs live in the `gpu-region` (ttbs) carveout, 16 bytes per
//! context; the firmware auto-loads TTBR0/TTBR1 for the active context. To make
//! an RTKit shared buffer reachable by the coprocessor we map its physical page
//! into the kernel context (ctx 0) at a GPU VA and hand that VA back in the
//! buffer reply — which is what unblocks the power-ON the coprocessor stalls on
//! after MILESTONE 1.
//!
//! This module is **arch-neutral + side-effect-free** (bit math only), so it
//! sits outside `arch/aarch64/` and is unit-tested under `cargo xtask test`
//! (x86). The MMIO half — allocating the tables, writing entries, pointing the
//! SGX registers at the ttbs — is the aarch64 orchestration in `agx::hw`.
//!
//! Field positions taken verbatim from `uat.py`'s `PTE`/`TTBR`/`Page_PTE` +
//! `LEVELS`.

#![allow(dead_code)] // the full walk API is written ahead of the MMIO driver

// --- geometry (uat.py: PAGE_BITS, L*_OFF, L*_SIZE) -----------------------
pub const PAGE_BITS: u32 = 14;
pub const PAGE_SIZE: u64 = 1 << PAGE_BITS; // 16 KiB
pub const PAGE_MASK: u64 = PAGE_SIZE - 1;
/// Bytes per table (2048 × 8) = one 16 KiB page.
pub const TABLE_BYTES: u64 = PAGE_SIZE;
/// Entries in an L1/L2/L3 table (`IDX_BITS = 11`).
pub const LX_ENTRIES: usize = 1 << 11; // 2048
/// Entries in the L0 (TTBR0/TTBR1 pair per context).
pub const L0_ENTRIES: usize = 2;
/// Bytes per context in the `gpu-region` ttbs array (2 × u64).
pub const TTBR_PAIR_BYTES: u64 = 16;

// VA field offsets per level (uat.py LEVELS).
const L0_OFF: u32 = 47; // TTBR0/TTBR1 select (2 entries)
const L1_OFF: u32 = 36;
const L2_OFF: u32 = 25;
const L3_OFF: u32 = 14;

/// UAT memory-attribute indices (uat.py `MemoryAttr`).
pub const ATTR_NORMAL: u64 = 0; // fw-only Normal WB
pub const ATTR_DEVICE: u64 = 1; // Device nGnRnE
pub const ATTR_SHARED: u64 = 2; // Normal, shared with the CPU/AGX hardware

// --- PTE / TTBR bitfields (uat.py PTE, TTBR) -----------------------------
const PTE_VALID: u64 = 1 << 0;
const PTE_TYPE: u64 = 1 << 1; // 1 = table (L1/L2) or page (L3); 0 = block
const PTE_AF: u64 = 1 << 10;
const PTE_NG: u64 = 1 << 11;
const PTE_PXN: u64 = 1 << 53;
const PTE_UXN: u64 = 1 << 54;
const PTE_OS: u64 = 1 << 55; // owned by host OS (vs firmware)
const ATTR_SHIFT: u32 = 2; // AttrIndex bits [4:2]
const AP_SHIFT: u32 = 6; // AP bits [7:6]
const SH_SHIFT: u32 = 8; // SH bits [9:8]

/// Inclusive bit-mask `[hi:lo]` (m1n1 `GENMASK`). Pure `const`.
pub const fn gen_mask(hi: u32, lo: u32) -> u64 {
    let width = hi - lo + 1;
    (if width == 64 { u64::MAX } else { (1u64 << width) - 1 }) << lo
}

// The output-address field is OFFSET = bits [47:14] (the 16 KiB frame).
const OFFSET_MASK: u64 = gen_mask(47, 14);
// TTBR BADDR = bits [47:1] (table base >> 1).
const BADDR_MASK: u64 = gen_mask(47, 1);
const ASID_SHIFT: u32 = 48;

/// Split a GPU VA into `(l0, l1, l2, l3, page_offset)` for the 16 KiB 4-level
/// walk. `l0` selects TTBR0 (0) / TTBR1 (1). Pure.
pub const fn split_va(va: u64) -> (usize, usize, usize, usize, u64) {
    let l0 = ((va >> L0_OFF) & (L0_ENTRIES as u64 - 1)) as usize;
    let l1 = ((va >> L1_OFF) & (LX_ENTRIES as u64 - 1)) as usize;
    let l2 = ((va >> L2_OFF) & (LX_ENTRIES as u64 - 1)) as usize;
    let l3 = ((va >> L3_OFF) & (LX_ENTRIES as u64 - 1)) as usize;
    (l0, l1, l2, l3, va & PAGE_MASK)
}

/// Encode an L3 **leaf page** descriptor mapping physical `pa` (16 KiB-aligned).
/// `attr` is an `ATTR_*` index; `ap` the access-permission field; `uxn`/`pxn`
/// the execute-never bits; `os` marks host-owned (vs firmware-owned). Pure.
pub const fn page_pte(pa: u64, attr: u64, ap: u64, uxn: bool, pxn: bool, os: bool) -> u64 {
    let mut v = PTE_VALID | PTE_TYPE | PTE_AF;
    v |= (attr & 0x7) << ATTR_SHIFT;
    v |= (ap & 0x3) << AP_SHIFT;
    v |= ((pa >> PAGE_BITS) << PAGE_BITS) & OFFSET_MASK;
    if uxn {
        v |= PTE_UXN;
    }
    if pxn {
        v |= PTE_PXN;
    }
    if os {
        v |= PTE_OS;
    }
    v
}

/// A leaf page mapping an RTKit shared buffer into the kernel context — the
/// default flags m1n1 uses (`OS=1, Normal, AP=1, AF=1, UXN=1`, uat.py
/// `iomap_at` map_flags). This is what a serviced buffer request maps.
pub const fn kernel_buffer_pte(pa: u64) -> u64 {
    page_pte(pa, ATTR_NORMAL, 1, true, false, true)
}

/// Encode an L1/L2 **table** descriptor pointing at the next-level table at
/// physical `next_table_pa` (16 KiB-aligned). Pure.
pub const fn table_pte(next_table_pa: u64) -> u64 {
    PTE_VALID | PTE_TYPE | (((next_table_pa >> PAGE_BITS) << PAGE_BITS) & OFFSET_MASK)
}

/// Encode a **TTBR** L0 entry pointing at an L1 table at physical `base_pa`,
/// tagged `asid`. `valid` is false for an empty slot (base 0). Pure
/// (uat.py `set_l0`).
pub const fn ttbr(base_pa: u64, asid: u64, valid: bool) -> u64 {
    let mut v = ((base_pa >> 1) << 1) & BADDR_MASK;
    v |= (asid & 0xffff) << ASID_SHIFT;
    if valid {
        v |= PTE_VALID;
    }
    v
}

// --- inverse accessors (for tests, dumps, and read-modify-write) ---------

/// The physical frame a leaf/table descriptor points at (inverse of the OFFSET
/// field). Pure.
pub const fn pte_output(pte: u64) -> u64 {
    pte & OFFSET_MASK
}
/// The L1-table physical base a TTBR points at (inverse of BADDR). Pure.
pub const fn ttbr_base(t: u64) -> u64 {
    (t & BADDR_MASK) >> 1 << 1
}
/// True if a descriptor's VALID bit is set. Pure.
pub const fn is_valid(desc: u64) -> bool {
    desc & PTE_VALID != 0
}

/// Byte offset of context `ctx`'s TTBR pair within the `gpu-region` (ttbs)
/// carveout, and of the TTBR0/TTBR1 slot for a VA's `l0` select. Pure.
pub const fn ttbr_slot_offset(ctx: usize, l0: usize) -> u64 {
    ctx as u64 * TTBR_PAIR_BYTES + l0 as u64 * 8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn geometry_matches_uatpy() {
        assert_eq!(PAGE_SIZE, 16384);
        assert_eq!(LX_ENTRIES, 2048);
        assert_eq!(LX_ENTRIES as u64 * 8, PAGE_SIZE); // one table == one page
        assert_eq!(L0_ENTRIES, 2);
    }

    #[test_case]
    fn va_split_indices_and_reconstruct() {
        // Craft a VA with distinct indices at each level + a page offset.
        let va = (1u64 << L0_OFF) // l0 = 1 (TTBR1)
            | (0x123u64 << L1_OFF)
            | (0x0abu64 << L2_OFF)
            | (0x1feu64 << L3_OFF)
            | 0x2000; // offset within the 16 KiB page
        let (l0, l1, l2, l3, off) = split_va(va);
        assert_eq!(l0, 1);
        assert_eq!(l1, 0x123);
        assert_eq!(l2, 0x0ab);
        assert_eq!(l3, 0x1fe);
        assert_eq!(off, 0x2000);
        // Reassembling the fields yields the original VA.
        let re = ((l0 as u64) << L0_OFF)
            | ((l1 as u64) << L1_OFF)
            | ((l2 as u64) << L2_OFF)
            | ((l3 as u64) << L3_OFF)
            | off;
        assert_eq!(re, va);
    }

    #[test_case]
    fn low_va_selects_ttbr0() {
        let (l0, ..) = split_va(0x1500_0000); // klow base in uat.py — bit 47 clear
        assert_eq!(l0, 0);
    }

    #[test_case]
    fn leaf_page_pte_layout() {
        let pa = 0x8_1234_0000u64; // 16 KiB-aligned physical
        let pte = kernel_buffer_pte(pa);
        assert!(is_valid(pte));
        assert_eq!(pte & PTE_TYPE, PTE_TYPE); // page (not block)
        assert_eq!(pte & PTE_AF, PTE_AF);
        assert_eq!(pte & PTE_OS, PTE_OS); // host-owned
        assert_eq!(pte & PTE_UXN, PTE_UXN);
        assert_eq!((pte >> ATTR_SHIFT) & 0x7, ATTR_NORMAL);
        assert_eq!((pte >> AP_SHIFT) & 0x3, 1);
        assert_eq!(pte_output(pte), pa); // frame survives the round-trip
        // A device-attr page with AP=0, no OS.
        let d = page_pte(pa, ATTR_DEVICE, 0, false, false, false);
        assert_eq!((d >> ATTR_SHIFT) & 0x7, ATTR_DEVICE);
        assert_eq!(d & PTE_OS, 0);
        assert_eq!(pte_output(d), pa);
    }

    #[test_case]
    fn table_descriptor_layout() {
        let next = 0x8_00ab_c000u64;
        let te = table_pte(next);
        assert!(is_valid(te));
        assert_eq!(te & PTE_TYPE, PTE_TYPE);
        assert_eq!(pte_output(te), next);
    }

    #[test_case]
    fn ttbr_encodes_base_asid_valid() {
        let l1 = 0x8_0055_4000u64; // 16 KiB-aligned L1 table
        let t = ttbr(l1, 0xbeef, true);
        assert!(is_valid(t));
        assert_eq!(ttbr_base(t), l1); // base survives
        assert_eq!((t >> ASID_SHIFT) & 0xffff, 0xbeef);
        // An empty slot: base 0, not valid.
        let empty = ttbr(0, 0, false);
        assert!(!is_valid(empty));
        assert_eq!(ttbr_base(empty), 0);
    }

    #[test_case]
    fn ttbr_slot_offsets_within_ttbs() {
        // ctx 0 → TTBR0 at +0, TTBR1 at +8; ctx 1 → +16 / +24.
        assert_eq!(ttbr_slot_offset(0, 0), 0);
        assert_eq!(ttbr_slot_offset(0, 1), 8);
        assert_eq!(ttbr_slot_offset(1, 0), 16);
        assert_eq!(ttbr_slot_offset(1, 1), 24);
    }

    #[test_case]
    fn full_walk_reaches_the_mapped_frame() {
        // Simulate the four-level walk purely: TTBR → L1 → L2 → L3 → frame, and
        // check every index/descriptor lines up so a serviced buffer request
        // would resolve to the right physical page.
        let l1 = 0x8_1000_0000u64;
        let l2 = 0x8_1000_4000u64;
        let l3 = 0x8_1000_8000u64;
        let frame = 0x8_2000_0000u64;
        let va = 0x1500_0000u64; // ctx-0 low VA (TTBR0)

        let (i0, i1, i2, i3, off) = split_va(va);
        let ttbr0 = ttbr(l1, 0, true);
        assert_eq!(i0, 0); // TTBR0
        assert_eq!(ttbr_base(ttbr0), l1);

        let l1e = table_pte(l2);
        assert_eq!(pte_output(l1e), l2);
        let l2e = table_pte(l3);
        assert_eq!(pte_output(l2e), l3);
        let l3e = kernel_buffer_pte(frame);
        assert_eq!(pte_output(l3e), frame);

        // Indices index within their tables and the offset stays in-page.
        assert!(i1 < LX_ENTRIES && i2 < LX_ENTRIES && i3 < LX_ENTRIES);
        assert!(off < PAGE_SIZE);
        let _ = (l1, l2, l3); // bases only used via the descriptors above
    }
}
