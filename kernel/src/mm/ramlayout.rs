//! Pure physical-RAM-layout math for the identity map: given the firmware's
//! actual RAM regions, decide how each 1 GiB block (and, inside a *mixed*
//! block, each 2 MiB chunk) must be typed. Real machines and VirtualBox-ARM
//! interleave RAM and MMIO inside one GiB block — e.g. VBox puts the tail of
//! low RAM (where the stub legitimately allocates the model), the GOP
//! framebuffer aperture (`0xd8000000`) and the PCIe ECAM (`0xfeddd000`) all in
//! the `0xC0000000` block — so typing whole GiB blocks is not expressible:
//! Normal over MMIO breaks/asserts, Device over RAM alignment-faults the
//! vector loads of the SDOT matvecs (the "/perf FATAL at FAR=0xc0000000").
//! Arch-neutral so the x86 unit suite exercises it.

/// How a 1 GiB block must be mapped, from the RAM regions that intersect it.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum BlockKind {
    /// Entirely RAM → one Normal cacheable block descriptor.
    Normal,
    /// No RAM at all → Device (MMIO may live anywhere in it).
    Device,
    /// RAM and non-RAM interleave → needs an L2 table of 2 MiB chunks.
    Mixed,
}

/// True if `[base, base+len)` lies entirely inside one of `regions`
/// (`(base, size)` pairs; need not be sorted). Zero-size regions are ignored.
pub fn range_is_ram(base: u64, len: u64, regions: &[(u64, u64)]) -> bool {
    let end = match base.checked_add(len) {
        Some(e) => e,
        None => return false,
    };
    regions
        .iter()
        .filter(|(_, s)| *s > 0)
        .any(|&(rb, rs)| base >= rb && end <= rb.saturating_add(rs))
}

/// True if `[base, base+len)` overlaps any of `regions` at all.
pub fn range_touches_ram(base: u64, len: u64, regions: &[(u64, u64)]) -> bool {
    let end = base.saturating_add(len);
    regions
        .iter()
        .filter(|(_, s)| *s > 0)
        .any(|&(rb, rs)| base < rb.saturating_add(rs) && rb < end)
    }

/// Classify the 1 GiB block starting at `block_base`.
pub fn classify_gib(block_base: u64, regions: &[(u64, u64)]) -> BlockKind {
    const GIB: u64 = 1 << 30;
    if range_is_ram(block_base, GIB, regions) {
        BlockKind::Normal
    } else if range_touches_ram(block_base, GIB, regions) {
        BlockKind::Mixed
    } else {
        BlockKind::Device
    }
}

/// For a chunk (2 MiB granule) inside a mixed block: Normal iff **fully** RAM.
/// A partially-RAM 2 MiB chunk types as Device — RAM regions are page-granular
/// so at worst <2 MiB of RAM at a region edge becomes uncached-but-correct,
/// while the reverse (Normal over MMIO) would be broken.
pub fn chunk_is_normal(chunk_base: u64, regions: &[(u64, u64)]) -> bool {
    const CHUNK: u64 = 2 << 20;
    range_is_ram(chunk_base, CHUNK, regions)
}

// --- carving a frame-allocator pool out of RAM ---------------------------

/// Frame size the carve rounds to. Matches [`crate::mm::frame::FRAME_SIZE`];
/// duplicated as a private constant so this module stays free of the allocator.
const FRAME: u64 = 4096;

const fn align_down(v: u64) -> u64 {
    v & !(FRAME - 1)
}

/// Round `v` up to a frame boundary, saturating at `u64::MAX` rather than
/// wrapping. The saturation is not decorative: aligning `u64::MAX` up overflows,
/// which in a debug build panics inside the boot's memory setup, and `u64::MAX`
/// is the conservative answer for a reserved range's end.
const fn align_up(v: u64) -> u64 {
    match v.checked_add(FRAME - 1) {
        Some(x) => align_down(x),
        None => u64::MAX,
    }
}

/// How many free fragments [`carve_free`] can report. A pool of at most 16 RAM
/// extents cut by ~8 reserved ranges cannot produce more than this in practice;
/// the overflow is *counted and reported* rather than silently truncated,
/// because "fewer free frames than the machine has" is a performance bug you
/// would never notice, whereas a dropped-count in the boot log is a lead.
pub const MAX_FREE: usize = 32;

/// The result of [`carve_free`]: frame-aligned ranges that are in the pool and
/// in none of the reserved ranges.
#[derive(Debug, Clone, Copy)]
pub struct FreeRanges {
    ranges: [(u64, u64); MAX_FREE],
    n: usize,
    dropped: usize,
}

impl FreeRanges {
    /// The carved free ranges, `(base, size)`, every one frame-aligned.
    pub fn as_slice(&self) -> &[(u64, u64)] {
        &self.ranges[..self.n]
    }
    /// Fragments that did not fit [`MAX_FREE`]. Nonzero means the pool is being
    /// under-reported and the constant wants raising.
    pub fn dropped(&self) -> usize {
        self.dropped
    }
    /// Total bytes across the carved ranges.
    pub fn total(&self) -> u64 {
        self.as_slice().iter().map(|&(_, s)| s).sum()
    }
    fn push(&mut self, base: u64, end: u64) {
        // Round *inward* to whole frames: a fragment's partial frames at either
        // edge belong to whatever shares them, so they are not ours to hand out.
        let base = align_up(base);
        let end = align_down(end);
        if end <= base {
            return;
        }
        if self.n < MAX_FREE {
            self.ranges[self.n] = (base, end - base);
            self.n += 1;
        } else {
            self.dropped += 1;
        }
    }
}

/// Subtract `reserved` from `pool`, yielding the frame-aligned ranges a physical
/// frame allocator may hand out.
///
/// This is the decision that makes an aarch64 frame allocator safe, and it is
/// the one place where being wrong is *invisible*: a frame wrongly called free
/// is handed to some later allocation which then overwrites the kernel image,
/// the heap, the loaded model, or the device tree — corruption that surfaces
/// somewhere else entirely, long after. So it is pure, and it errs in one
/// direction only.
///
/// Both roundings go the conservative way and neither is symmetric with the
/// other:
///
/// * a **reserved** range is grown to whole frames (base down, end up), because
///   a frame that is even partly reserved is entirely unavailable;
/// * a **free** fragment is shrunk to whole frames (base up, end down), for the
///   same reason from the other side.
///
/// `reserved` need not be sorted, and its ranges may overlap each other or
/// extend outside the pool entirely (the framebuffer usually does — on the UEFI
/// path it is an MMIO aperture, not RAM). Zero-size entries in either list are
/// ignored.
pub fn carve_free(pool: &[(u64, u64)], reserved: &[(u64, u64)]) -> FreeRanges {
    let mut out = FreeRanges { ranges: [(0, 0); MAX_FREE], n: 0, dropped: 0 };
    // Frame-expanded reserved bounds, computed on demand (no heap here — this
    // runs before the allocator exists on the path that needs it most).
    let res = |i: usize| -> (u64, u64) {
        let (b, s) = reserved[i];
        if s == 0 {
            return (0, 0);
        }
        (align_down(b), align_up(b.saturating_add(s)))
    };
    for &(pool_base, pool_size) in pool.iter().filter(|&&(_, s)| s > 0) {
        let pool_end = pool_base.saturating_add(pool_size);
        let mut cursor = pool_base;
        while cursor < pool_end {
            // Step over every reserved range covering the cursor. Looping
            // rather than taking one step handles reserved ranges that overlap
            // or abut each other — which they do (the model window ends exactly
            // where the heap begins).
            let mut moved = true;
            while moved {
                moved = false;
                for i in 0..reserved.len() {
                    let (rb, re) = res(i);
                    if re > rb && rb <= cursor && cursor < re {
                        cursor = re;
                        moved = true;
                    }
                }
            }
            if cursor >= pool_end {
                break;
            }
            // Free until the next reserved range starts, or the pool ends.
            let mut next = pool_end;
            for i in 0..reserved.len() {
                let (rb, re) = res(i);
                if re > rb && rb > cursor && rb < next {
                    next = rb;
                }
            }
            out.push(cursor, next);
            cursor = next;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The VirtualBox-ARM shape that broke: low RAM 0x4000_0000..0xD200_0000
    /// (contains the model tail crossing 0xC000_0000), MMIO hole with the fb
    /// (0xd800_0000) + ECAM (0xfedd_d000), more RAM above 4 GiB.
    const VBOX: [(u64, u64); 2] = [
        (0x4000_0000, 0x9200_0000),   // → 0xD200_0000
        (0x1_0000_0000, 0x3C00_0000), // ~960 MiB high
    ];

    #[test_case]
    fn vbox_blocks_classify() {
        assert_eq!(classify_gib(0x0000_0000, &VBOX), BlockKind::Device); // MMIO low
        assert_eq!(classify_gib(0x4000_0000, &VBOX), BlockKind::Normal);
        assert_eq!(classify_gib(0x8000_0000, &VBOX), BlockKind::Normal);
        assert_eq!(classify_gib(0xC000_0000, &VBOX), BlockKind::Mixed); // RAM tail + fb + ECAM
        assert_eq!(classify_gib(0x1_0000_0000, &VBOX), BlockKind::Mixed); // high RAM, partial GiB
        assert_eq!(classify_gib(0x2_0000_0000, &VBOX), BlockKind::Device);
    }

    #[test_case]
    fn vbox_mixed_block_chunks() {
        // Model tail at 0xC000_0000 → RAM → Normal (the /perf crash site).
        assert!(chunk_is_normal(0xC000_0000, &VBOX));
        assert!(chunk_is_normal(0xCFE0_0000, &VBOX));
        // Framebuffer + ECAM chunks → Device.
        assert!(!chunk_is_normal(0xD800_0000, &VBOX));
        assert!(!chunk_is_normal(0xFEC0_0000, &VBOX));
        // The 2 MiB-aligned region edge itself is fully RAM → Normal.
        assert!(chunk_is_normal(0xD1E0_0000, &VBOX));
        // With a page-granular (non-2 MiB-aligned) region end, the straddling
        // chunk types Device — the safe side.
        let ragged = [(0x4000_0000u64, 0x91FF_0000u64)]; // ends 0xD1FF_0000
        assert!(!chunk_is_normal(0xD1E0_0000, &ragged));
        assert!(chunk_is_normal(0xD1C0_0000, &ragged));
    }

    #[test_case]
    fn apple_silicon_high_ram_base() {
        // Apple Silicon (via m1n1) places system RAM at 0x8_0000_0000 (32 GiB),
        // not QEMU's 0x4000_0000. The ~32 GiB of address space below it is
        // unbacked / low MMIO and MUST type as Device — mapping it Normal would
        // let the core speculatively touch an unbacked address and fault. The
        // RAM blocks themselves stay Normal so code/heap/stack run cacheable.
        let m2: [(u64, u64); 1] = [(0x8_0000_0000, 16u64 << 30)]; // 16 GiB Mac Mini
        // Below the base: Device, never Normal.
        assert_eq!(classify_gib(0x0000_0000, &m2), BlockKind::Device);
        assert_eq!(classify_gib(0x4000_0000, &m2), BlockKind::Device); // QEMU's base — unbacked here
        assert_eq!(classify_gib(0x2_0000_0000, &m2), BlockKind::Device);
        assert_eq!(classify_gib(0x7_C000_0000, &m2), BlockKind::Device); // last GiB before RAM
        // The RAM span [32 GiB, 48 GiB): all Normal.
        assert_eq!(classify_gib(0x8_0000_0000, &m2), BlockKind::Normal);
        assert_eq!(classify_gib(0xB_8000_0000, &m2), BlockKind::Normal);
        assert_eq!(classify_gib(0xB_C000_0000, &m2), BlockKind::Normal); // last RAM GiB
        // Past the top: Device again.
        assert_eq!(classify_gib(0xC_0000_0000, &m2), BlockKind::Device);
    }

    #[test_case]
    fn contiguous_qemu_layout_stays_all_normal() {
        // QEMU virt: one contiguous clump — every covered block is Normal, so
        // the mixed path never engages (legacy behaviour preserved).
        let qemu = [(0x4000_0000u64, 3u64 << 30)];
        assert_eq!(classify_gib(0x4000_0000, &qemu), BlockKind::Normal);
        assert_eq!(classify_gib(0x8000_0000, &qemu), BlockKind::Normal);
        assert_eq!(classify_gib(0xC000_0000, &qemu), BlockKind::Normal);
        assert_eq!(classify_gib(0x1_0000_0000, &qemu), BlockKind::Device);
    }

    // --- carve_free ------------------------------------------------------
    //
    // The aarch64 frame allocator's pool comes out of this. Every case below is
    // a shape the real boot paths produce, and each failure mode is silent
    // memory corruption rather than an error, so they are pinned individually.

    const MIB: u64 = 1 << 20;

    #[test_case]
    fn carve_with_nothing_reserved_returns_the_pool() {
        let f = carve_free(&[(0x4000_0000, 4 * MIB)], &[]);
        assert_eq!(f.as_slice(), &[(0x4000_0000, 4 * MIB)]);
        assert_eq!(f.dropped(), 0);
    }

    #[test_case]
    fn carve_splits_a_pool_around_a_reservation_in_its_middle() {
        // The QEMU `-kernel` shape in miniature: the kernel image sits inside
        // the one RAM extent, so the pool becomes two fragments.
        let f = carve_free(&[(0, 16 * MIB)], &[(4 * MIB, 2 * MIB)]);
        assert_eq!(f.as_slice(), &[(0, 4 * MIB), (6 * MIB, 10 * MIB)]);
    }

    #[test_case]
    fn carve_handles_reservations_at_both_pool_edges() {
        // A reservation flush against the start yields no leading fragment (an
        // off-by-one here would emit a zero- or negative-size range); one flush
        // against the end yields no trailing fragment.
        let f = carve_free(&[(MIB, 8 * MIB)], &[(MIB, MIB), (8 * MIB, MIB)]);
        assert_eq!(f.as_slice(), &[(2 * MIB, 6 * MIB)]);
    }

    #[test_case]
    fn carve_drops_a_pool_region_reserved_end_to_end() {
        // The UEFI path really does this: the stub's heap allocation can be a
        // whole RAM extent. The result must be *nothing*, not the original.
        let f = carve_free(&[(0x5000_0000, 4 * MIB)], &[(0x5000_0000, 4 * MIB)]);
        assert!(f.as_slice().is_empty());
        assert_eq!(f.total(), 0);
        // Over-covering it (reserved extends past both ends) is the same answer.
        // Spelled with explicit bounds: a length that looks generous but stops
        // short of the pool would make this pass for the wrong reason.
        let (rb, re) = (0x4000_0000u64, 0x6000_0000u64);
        assert!(rb < 0x5000_0000 && re > 0x5000_0000 + 4 * MIB, "test premise: reserved must span the pool");
        let f = carve_free(&[(0x5000_0000, 4 * MIB)], &[(rb, re - rb)]);
        assert!(f.as_slice().is_empty());
    }

    #[test_case]
    fn carve_merges_abutting_and_overlapping_reservations() {
        // Abutting is not hypothetical: on the `-kernel` path the model window
        // ends *exactly* where the heap begins, so the cursor must step through
        // both without emitting a zero-length fragment between them.
        let f = carve_free(&[(0, 16 * MIB)], &[(4 * MIB, 2 * MIB), (6 * MIB, 2 * MIB)]);
        assert_eq!(f.as_slice(), &[(0, 4 * MIB), (8 * MIB, 8 * MIB)]);
        // Overlapping, and listed out of address order — the reserved list is
        // assembled by the caller in whatever order it learns things.
        let f = carve_free(&[(0, 16 * MIB)], &[(7 * MIB, 3 * MIB), (4 * MIB, 4 * MIB)]);
        assert_eq!(f.as_slice(), &[(0, 4 * MIB), (10 * MIB, 6 * MIB)]);
    }

    #[test_case]
    fn carve_ignores_reservations_outside_the_pool() {
        // The framebuffer is the real case: on the UEFI path it is an MMIO
        // aperture above RAM, so it must neither shrink the pool nor confuse the
        // walk. Same for a reservation entirely below it.
        let f = carve_free(&[(0x4000_0000, 16 * MIB)], &[(0xd800_0000, 8 * MIB), (0, MIB)]);
        assert_eq!(f.as_slice(), &[(0x4000_0000, 16 * MIB)]);
    }

    #[test_case]
    fn carve_grows_a_reservation_to_whole_frames_but_shrinks_free_ranges() {
        // The asymmetry is the safety property. A reservation of one byte in the
        // middle of a frame must cost the *whole* frame...
        let f = carve_free(&[(0, 3 * 4096)], &[(4096 + 1, 1)]);
        assert_eq!(f.as_slice(), &[(0, 4096), (8192, 4096)]);
        // ...and an unaligned pool must lose its partial frames at both ends,
        // never round outward into memory that isn't in the pool.
        let f = carve_free(&[(1, 3 * 4096)], &[]);
        assert_eq!(f.as_slice(), &[(4096, 8192)]);
    }

    #[test_case]
    fn carve_drops_fragments_smaller_than_a_frame() {
        // A sliver between two reservations is not a frame and must not be
        // reported as one — `push` would otherwise emit a range the allocator
        // indexes into as if it held a frame.
        let f = carve_free(&[(0, 16 * 4096)], &[(0, 4096 + 100), (2 * 4096, 14 * 4096)]);
        assert!(f.as_slice().is_empty(), "the ~4 KiB sliver is not a whole frame");
    }

    #[test_case]
    fn carve_handles_the_vbox_two_extent_pool() {
        // Both RAM clumps, with the kernel image in the low one and the stub's
        // heap in the high one: three fragments, and crucially the hole between
        // the extents is never reported as free.
        let pool = [(0x4000_0000u64, 0x1000_0000u64), (0x1_0000_0000, 0x1000_0000)];
        let reserved = [(0x4008_0000, 4 * MIB), (0x1_0800_0000, 0x0800_0000)];
        let f = carve_free(&pool, &reserved);
        assert_eq!(
            f.as_slice(),
            &[
                (0x4000_0000, 0x0008_0000),
                (0x4048_0000, 0x0fb8_0000),
                (0x1_0000_0000, 0x0800_0000),
            ]
        );
        // The MMIO hole between 0x5000_0000 and 0x1_0000_0000 is absent.
        assert!(f.as_slice().iter().all(|&(b, s)| b + s <= 0x5000_0000 || b >= 0x1_0000_0000));
    }

    #[test_case]
    fn carve_reports_overflow_instead_of_truncating_silently() {
        // MAX_FREE+2 reservations spaced to produce more fragments than fit.
        // The count must show up, because an under-reported pool is otherwise
        // indistinguishable from a machine with less RAM.
        let mut reserved = [(0u64, 0u64); MAX_FREE + 2];
        for (i, r) in reserved.iter_mut().enumerate() {
            *r = ((2 * i as u64 + 1) * 4096, 4096);
        }
        let f = carve_free(&[(0, (2 * (MAX_FREE + 2) as u64 + 2) * 4096)], &reserved);
        assert_eq!(f.as_slice().len(), MAX_FREE);
        assert!(f.dropped() > 0, "the fragments that did not fit must be counted");
    }

    #[test_case]
    fn carve_ignores_zero_size_entries_in_both_lists() {
        // Both lists are fixed-size arrays the caller fills partially — the
        // model region is (0, 0) on the UEFI path, the framebuffer (0, 0) before
        // it is known — so a zero-size entry must not reserve frame 0.
        let f = carve_free(&[(0, 4 * 4096), (0, 0)], &[(0, 0), (0x9_0000, 0)]);
        assert_eq!(f.as_slice(), &[(0, 4 * 4096)]);
    }

    #[test_case]
    fn carve_on_an_empty_pool_yields_nothing() {
        assert!(carve_free(&[], &[(0, 4096)]).as_slice().is_empty());
        assert_eq!(carve_free(&[], &[]).total(), 0);
    }

    #[test_case]
    fn carve_is_overflow_safe_at_the_top_of_the_address_space() {
        // A bogus size must not wrap into a pool that looks enormous.
        let f = carve_free(&[(u64::MAX - 4096, 8192)], &[]);
        assert!(f.total() <= 8192);
        let f = carve_free(&[(0, 16 * 4096)], &[(u64::MAX - 1, 8192)]);
        assert_eq!(f.as_slice(), &[(0, 16 * 4096)]);
    }

    #[test_case]
    fn edge_cases() {
        assert!(!range_is_ram(0, 0x1000, &[]));
        assert!(!range_touches_ram(0, 0x1000, &[]));
        // Zero-size region ignored; overflow-safe.
        assert!(!range_is_ram(u64::MAX - 4096, 8192, &[(0, u64::MAX)]));
        assert!(range_is_ram(0x4000_0000, 0x1000, &[(0x4000_0000, 0x1000), (0, 0)]));
    }
}
