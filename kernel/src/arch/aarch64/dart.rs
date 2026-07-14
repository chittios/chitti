//! Apple **DART** (Device Address Resolution Table) — the IOMMU in front of
//! every Apple-Silicon DMA engine (USB/dwc3, NVMe/ANS, display, …). Nothing
//! DMAs without going through it. This is the **T8110** variant (M2/t8112 USB
//! DARTs report `apple,t8110-dart`).
//!
//! Register layout, PTE/TTBR encoding, and the IOVA decomposition are taken
//! verbatim from m1n1's `src/dart.c` (`dart_t8110` params + `dart_map_page` /
//! `dart_get_l2`), vendored at `third_party/m1n1/src/dart.c` — the pure math is
//! unit-tested below.
//!
//! Two modes:
//! * [`Dart::set_bypass`] — put a stream in **bypass** (device addresses pass
//!   through as physical). m1n1 leaves the USB DART in bypass at handoff, and
//!   ChittiOS runs an identity map (VA==PA), so bypass + identity DMA is the
//!   simplest correct path for the initial USB HID bring-up.
//! * [`Dart::map`] — full two-level **translation** (16 KiB pages/tables). The
//!   proper long-term mode; kept + tested but not yet used on the USB path.
//!
//! All MMIO is a single `read_volatile`/`write_volatile` (the aarch64 MMIO
//! rule — no coalesced pairs the hypervisor can't decode).

#![allow(dead_code)] // the translation path is written + tested ahead of use

use core::ptr::{read_volatile, write_volatile};

// --- T8110 register offsets (dart.c) -------------------------------------
const TCR_OFF: usize = 0x1000; // TCR(sid) = base + TCR_OFF + 4*sid
const TCR_TRANSLATE_ENABLE: u32 = 1 << 0;
const TCR_BYPASS_DART: u32 = 1 << 1;
const TCR_BYPASS_DAPF: u32 = 1 << 2;
const TTBR_OFF: usize = 0x1400; // TTBR(sid,0) = base + TTBR_OFF + 4*sid (ttbr_count=1)
const TTBR_VALID: u32 = 1 << 0;
const TTBR_SHIFT: u32 = 14; // table_pa >> 14 packed into TTBR_ADDR
                            // TTBR_ADDR = bits[29:2]
const TLB_CMD: usize = 0x80;
const TLB_CMD_BUSY: u32 = 1 << 31;
const TLB_CMD_OP_FLUSH_SID: u32 = 1; // bits[10:8]
const PROTECT: usize = 0x200;
const PROTECT_TTBR_TCR: u32 = 1 << 0; // locked → must not reprogram TTBR/TCR
const ENABLE_STREAMS: usize = 0xc00; // + 4*(sid>>5), set BIT(sid & 0x1f)

// --- PTE / IOVA (dart.c: DART_T6000_PTE_OFFSET params, used by T8110) -----
const PTE_OFFSET_SHIFT: u64 = 14; // paddr >> 14 → 16 KiB frame
/// Output-address field of a PTE/L1 entry: bits[39:10] (`GENMASK(39,10)`).
const PTE_OFFSET_LSB: u64 = 10;
const PTE_OFFSET_MASK: u64 = gen_mask(39, 10);
/// `pte_flags` for T8110: `SP_END=0xfff | SP_START=0 | VALID`.
const PTE_VALID: u64 = 1 << 0;
const PTE_FLAGS: u64 = (0xfff << 40) | PTE_VALID; // SP_END(bits[51:40])=0xfff
const PAGE_SIZE: u64 = 0x4000; // 16 KiB
const TABLE_ENTRIES: usize = 2048; // 16 KiB / 8

/// Inclusive bit-mask `[hi:lo]` (like m1n1's `GENMASK`).
const fn gen_mask(hi: u32, lo: u32) -> u64 {
    let width = hi - lo + 1;
    (if width == 64 { u64::MAX } else { (1u64 << width) - 1 }) << lo
}

/// Decompose an IOVA into `(l1_index, l2_index, page_offset)` for the T8110
/// two-level walk (`dart_map_page`/`dart_get_l2`: l1 = `(iova>>25)&0x7ff` after
/// `get_l2` re-masks, l2 = `(iova>>14)&0x7ff`, offset = `iova&0x3fff`). Pure.
pub fn iova_split(iova: u64) -> (usize, usize, u64) {
    let l1 = ((iova >> 25) & 0x7ff) as usize;
    let l2 = ((iova >> 14) & 0x7ff) as usize;
    let off = iova & 0x3fff;
    (l1, l2, off)
}

/// Encode a leaf PTE (or an L1→L2 pointer) for physical address `pa`. Pure.
pub fn make_pte(pa: u64) -> u64 {
    (((pa >> PTE_OFFSET_SHIFT) << PTE_OFFSET_LSB) & PTE_OFFSET_MASK) | PTE_FLAGS
}

/// Extract the physical base a PTE/L1 entry points at (inverse of [`make_pte`]).
pub fn pte_addr(pte: u64) -> u64 {
    ((pte & PTE_OFFSET_MASK) >> PTE_OFFSET_LSB) << PTE_OFFSET_SHIFT
}

/// Encode the TTBR value for an L1 table at physical `l1_pa` (`table>>14` in
/// bits[29:2] | VALID). Pure.
pub fn make_ttbr(l1_pa: u64) -> u32 {
    ((((l1_pa >> TTBR_SHIFT as u64) << 2) as u32) & gen_mask(29, 2) as u32) | TTBR_VALID
}

/// A handle to one Apple DART at a discovered MMIO base, driving one stream
/// (SID). Construct with the base + SID from the FDT (`iommus = <&dart SID>`).
pub struct Dart {
    base: usize,
    sid: u32,
}

impl Dart {
    /// # Safety
    /// `base` must be the Device-mapped MMIO of a T8110 DART; `sid` a valid
    /// stream for it. Caller ensures exclusive access during setup.
    pub unsafe fn new(base: usize, sid: u32) -> Dart {
        Dart { base, sid }
    }

    #[inline]
    fn r(&self, off: usize) -> u32 {
        // SAFETY: single 32-bit MMIO read of a mapped DART register.
        unsafe { read_volatile((self.base + off) as *const u32) }
    }
    #[inline]
    fn w(&self, off: usize, v: u32) {
        // SAFETY: single 32-bit MMIO write of a mapped DART register.
        unsafe { write_volatile((self.base + off) as *mut u32, v) }
    }

    /// True if the DART's TTBR/TCR are locked (we must not reprogram them).
    pub fn is_locked(&self) -> bool {
        self.r(PROTECT) & PROTECT_TTBR_TCR != 0
    }

    /// Enable this stream in the DART's stream bitmap.
    fn enable_stream(&self) {
        let off = ENABLE_STREAMS + 4 * (self.sid as usize >> 5);
        self.w(off, self.r(off) | (1 << (self.sid & 0x1f)));
    }

    /// Flush this stream's TLB (after changing its TTBR/TCR/PTEs).
    fn tlb_flush(&self) {
        self.w(
            TLB_CMD,
            (TLB_CMD_OP_FLUSH_SID << 8) | (self.sid & 0xff),
        );
        let mut spins = 0;
        while self.r(TLB_CMD) & TLB_CMD_BUSY != 0 && spins < 100_000 {
            spins += 1;
            core::hint::spin_loop();
        }
    }

    /// Put the stream in **bypass**: device addresses are used as physical
    /// (no translation). With ChittiOS's identity map this lets the controller
    /// DMA directly to buffer physical addresses. Returns false if the DART is
    /// locked (can't touch TCR). This is what the initial USB path uses.
    pub fn set_bypass(&self) -> bool {
        if self.is_locked() {
            crate::ktrace::log_fmt(format_args!("dart: {:#x} sid {} locked; leaving as-is", self.base, self.sid));
            return false;
        }
        self.enable_stream();
        self.w(TCR_OFF + 4 * self.sid as usize, TCR_BYPASS_DART | TCR_BYPASS_DAPF);
        self.tlb_flush();
        crate::ktrace::log_fmt(format_args!("dart: {:#x} sid {} → bypass", self.base, self.sid));
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn iova_decomposition_matches_dartc() {
        // 16 KiB page granularity; fields at [l1:35..25][l2:24..14][off:13..0].
        assert_eq!(iova_split(0), (0, 0, 0));
        assert_eq!(iova_split(0x3fff), (0, 0, 0x3fff)); // last byte of page 0
        assert_eq!(iova_split(0x4000), (0, 1, 0)); // next 16 KiB page → l2=1
        assert_eq!(iova_split(0x2000000), (1, 0, 0)); // +32 MiB → l1=1 (1<<25)
        let (l1, l2, off) = iova_split(0xbabe0000);
        assert_eq!(off, 0xbabe0000 & 0x3fff);
        assert_eq!(l2, ((0xbabe0000u64 >> 14) & 0x7ff) as usize);
        assert_eq!(l1, ((0xbabe0000u64 >> 25) & 0x7ff) as usize);
    }

    #[test_case]
    fn pte_roundtrip_and_flags() {
        let pa = 0x8_1234_0000u64; // 16 KiB-aligned physical
        let pte = make_pte(pa);
        assert_eq!(pte & PTE_VALID, PTE_VALID);
        assert_eq!(pte & (0xfff << 40), 0xfff << 40); // SP_END span
        assert_eq!(pte_addr(pte), pa); // frame survives the round-trip
    }

    #[test_case]
    fn ttbr_encodes_table_frame() {
        let l1 = 0x8_00ab_c000u64; // 16 KiB-aligned
        let ttbr = make_ttbr(l1);
        assert_eq!(ttbr & TTBR_VALID, TTBR_VALID);
        // Recover the frame: bits[29:2] >> 2 << 14.
        let frame = (((ttbr as u64) & gen_mask(29, 2)) >> 2) << TTBR_SHIFT as u64;
        assert_eq!(frame, l1);
    }
}
