//! Physical frame allocator: one bit per 4 KiB frame, direct-indexed by
//! `physical_address / FRAME_SIZE`.
//!
//! The bitmap's own backing storage is bootstrapped out of the first
//! usable region large enough to hold it (there is no heap yet — this
//! *is* what the heap gets built on top of), reached through the
//! physical-to-virtual offset the caller supplies.
//!
//! **Arch-neutral by construction.** [`BitmapFrameAllocator::from_usable`] takes
//! nothing but an iterator of usable `(base, length)` physical regions and the
//! offset at which physical memory is addressable, because the two arches learn
//! their RAM extents from completely different places: x86 from the Limine
//! memory map (via [`BitmapFrameAllocator::init`], a thin adapter over the same
//! core), aarch64 from the DTB `/memory` node or the UEFI stub's boot-info
//! (`arch::aarch64::mmu`'s `RamInfo`). Keep new logic in the neutral core — the
//! Limine types must not leak back in, or aarch64 loses its frame allocator
//! again.

#[cfg(target_arch = "x86_64")]
use crate::limine_protocol::{self, MemmapEntry};

pub const FRAME_SIZE: u64 = 4096;

pub struct BitmapFrameAllocator {
    bitmap: &'static mut [u8],
    frame_count: u64,
    /// Next-fit cursor: `allocate` starts scanning here rather than from
    /// frame 0. Without it, allocating N frames is O(N²) (every call
    /// rescans all already-used low frames), which made mapping the 64 MiB
    /// heap take minutes under QEMU. With it, a run of sequential
    /// allocations (heap bring-up, KV growth) is amortized O(1) each.
    next_hint: u64,
}

fn get_bit(bitmap: &[u8], i: u64) -> bool {
    bitmap[(i / 8) as usize] & (1 << (i % 8)) != 0
}

fn set_bit(bitmap: &mut [u8], i: u64, used: bool) {
    let byte = &mut bitmap[(i / 8) as usize];
    let mask = 1 << (i % 8);
    if used {
        *byte |= mask;
    } else {
        *byte &= !mask;
    }
}

/// Whether the physical run `[start, start + len)` satisfies an ISA-DMA-style
/// placement constraint: entirely below `limit`, and not straddling a
/// `boundary`-aligned block (`boundary == 0` disables that check).
///
/// Pure so the awkward part — the straddle test — is unit-testable. The subtlety
/// is that it must compare the block of the **last byte** (`start + len - 1`),
/// not of `start + len`: a run ending exactly on a boundary is legal, and
/// testing the end address would reject it.
pub(crate) fn run_fits(start: u64, len: u64, limit: u64, boundary: u64) -> bool {
    match start.checked_add(len) {
        Some(end) if end <= limit => {}
        _ => return false,
    }
    if boundary == 0 || len == 0 {
        return true;
    }
    start / boundary == (start + len - 1) / boundary
}

fn mark_range(bitmap: &mut [u8], base: u64, length: u64, used: bool) {
    let start_frame = base / FRAME_SIZE;
    let end_frame = (base + length) / FRAME_SIZE;
    for frame in start_frame..end_frame {
        set_bit(bitmap, frame, used);
    }
}

/// Where the bitmap for a set of usable regions must live, and how big it is:
/// `(frame_count, bitmap_bytes, bitmap_phys)`. Split out of the constructor and
/// kept pure so the sizing/placement decision — the part that is easy to get
/// wrong and impossible to observe once the allocator is live — is unit-testable
/// on either arch without a memory map or an MMU.
///
/// Returns `None` when no single usable region can host the bitmap, which is the
/// one unrecoverable case: there is no heap yet, so there is nowhere else to put it.
pub(crate) fn bitmap_placement<I>(usable: I) -> Option<(u64, usize, u64)>
where
    I: Iterator<Item = (u64, u64)> + Clone,
{
    // Size the bitmap off usable regions only: some firmware/bootloader memory
    // maps include a final huge RESERVED entry covering the rest of the 64-bit
    // address space as a sentinel, which would otherwise inflate the bitmap to
    // cover terabytes of address space we will never allocate a frame from.
    let max_addr = usable.clone().map(|(base, len)| base + len).max().unwrap_or(0);
    let frame_count = max_addr.div_ceil(FRAME_SIZE);
    let bitmap_bytes = (frame_count as usize).div_ceil(8);
    let bitmap_phys = usable.clone().find(|&(_, len)| len as usize >= bitmap_bytes).map(|(base, _)| base)?;
    Some((frame_count, bitmap_bytes, bitmap_phys))
}

impl BitmapFrameAllocator {
    /// Build the allocator from an iterator of usable `(base, length)` physical
    /// regions, with physical address `p` readable at virtual `p + phys_offset`
    /// (the HHDM offset on x86; 0 on aarch64's identity map).
    ///
    /// The iterator is walked more than once — hence `Clone` — because the
    /// bitmap has to be sized and placed before any region can be marked free.
    ///
    /// # Safety
    /// Every `(base, length)` must be real, currently-unused, writable physical
    /// memory, and `phys_offset` must map it to a valid virtual address. Both are
    /// trusted verbatim: this decides which frames the kernel will hand out.
    pub unsafe fn from_usable<I>(usable: I, phys_offset: u64) -> Self
    where
        I: Iterator<Item = (u64, u64)> + Clone,
    {
        let (frame_count, bitmap_bytes, bitmap_phys) = bitmap_placement(usable.clone())
            .expect("mm::frame: no usable region large enough to hold the frame bitmap");

        let bitmap_ptr = (bitmap_phys + phys_offset) as *mut u8;
        // SAFETY: `bitmap_phys` starts a usable region at least `bitmap_bytes`
        // long, and `phys_offset` maps it to a valid, writable virtual address.
        let bitmap = unsafe { core::slice::from_raw_parts_mut(bitmap_ptr, bitmap_bytes) };
        bitmap.fill(0xff); // default: every frame used/reserved

        for (base, len) in usable {
            mark_range(bitmap, base, len, false);
        }

        // Re-reserve the frames the bitmap itself now occupies.
        let bitmap_frames = (bitmap_bytes as u64).div_ceil(FRAME_SIZE);
        mark_range(bitmap, bitmap_phys, bitmap_frames * FRAME_SIZE, true);

        Self { bitmap, frame_count, next_hint: 0 }
    }

    /// Build the allocator from Limine's memory map (x86). A thin adapter over
    /// [`Self::from_usable`] — it only selects the USABLE entries.
    ///
    /// # Safety
    /// `entries` must be the exact, still-valid memory map Limine returned, and
    /// `hhdm_offset` the real HHDM offset it reported.
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn init(entries: &[&MemmapEntry], hhdm_offset: u64) -> Self {
        // SAFETY: forwarded from this function's own contract.
        unsafe {
            Self::from_usable(
                entries
                    .iter()
                    .filter(|e| e.entry_type == limine_protocol::MEMMAP_USABLE)
                    .map(|e| (e.base, e.length)),
                hhdm_offset,
            )
        }
    }

    /// Allocate one free 4 KiB frame, returning its physical address.
    /// Next-fit: scan from `next_hint` to the end, then wrap to the start.
    pub fn allocate(&mut self) -> Option<u64> {
        let scan = |this: &mut Self, range: core::ops::Range<u64>| -> Option<u64> {
            for frame in range {
                if !get_bit(this.bitmap, frame) {
                    set_bit(this.bitmap, frame, true);
                    this.next_hint = frame + 1;
                    return Some(frame * FRAME_SIZE);
                }
            }
            None
        };
        let hint = self.next_hint.min(self.frame_count);
        scan(self, hint..self.frame_count).or_else(|| scan(self, 0..hint))
    }

    /// Allocate `count` *physically contiguous* free frames, returning the
    /// base physical address. Needed for DMA regions (the virtio virtqueue and
    /// request buffers) that a device accesses by physical address and that
    /// must be contiguous. A linear first-fit scan; `count` is tiny in
    /// practice (a virtqueue is a handful of pages).
    pub fn allocate_contiguous(&mut self, count: u64) -> Option<u64> {
        if count == 0 {
            return None;
        }
        let mut start = 0u64;
        while start + count <= self.frame_count {
            // Find the first free frame at or after `start`.
            if get_bit(self.bitmap, start) {
                start += 1;
                continue;
            }
            // Check the next `count` frames are all free.
            let run_ok = (start..start + count).all(|f| !get_bit(self.bitmap, f));
            if run_ok {
                for f in start..start + count {
                    set_bit(self.bitmap, f, true);
                }
                return Some(start * FRAME_SIZE);
            }
            // Skip past the blocking used frame.
            match (start..start + count).find(|&f| get_bit(self.bitmap, f)) {
                Some(blocker) => start = blocker + 1,
                None => start += 1,
            }
        }
        None
    }

    /// Allocate `count` contiguous frames whose physical range lies entirely
    /// below `limit` **and** does not straddle a `boundary`-aligned block.
    ///
    /// This exists for **ISA DMA**, whose constraints ordinary allocation cannot
    /// express: the 8237 controller latches a 24-bit address (so buffers must live
    /// under 16 MiB) and a fixed page register (so a transfer may not cross a
    /// 64 KiB block on an 8-bit channel, or 128 KiB on a 16-bit one). Without
    /// this, the SB16 driver could only allocate normally, find its buffer above
    /// the limit, and decline — which is exactly what it always did.
    ///
    /// `boundary` of 0 means "no boundary constraint".
    pub fn allocate_contiguous_bounded(&mut self, count: u64, limit: u64, boundary: u64) -> Option<u64> {
        if count == 0 {
            return None;
        }
        let len = count * FRAME_SIZE;
        let max_frame = (limit / FRAME_SIZE).min(self.frame_count);
        let mut start = 0u64;
        while start + count <= max_frame {
            if !run_fits(start * FRAME_SIZE, len, limit, boundary) {
                // Jump to the start of the next boundary block rather than
                // crawling a frame at a time through a region that cannot work.
                if boundary != 0 {
                    let next = (start * FRAME_SIZE / boundary + 1) * boundary;
                    start = next / FRAME_SIZE;
                    continue;
                }
                break; // only the limit can fail with no boundary, so we're done
            }
            if (start..start + count).all(|f| !get_bit(self.bitmap, f)) {
                for f in start..start + count {
                    set_bit(self.bitmap, f, true);
                }
                return Some(start * FRAME_SIZE);
            }
            match (start..start + count).find(|&f| get_bit(self.bitmap, f)) {
                Some(blocker) => start = blocker + 1,
                None => start += 1,
            }
        }
        None
    }

    /// Return a previously allocated frame to the pool.
    pub fn free(&mut self, phys: u64) {
        let frame = phys / FRAME_SIZE;
        assert!(frame < self.frame_count, "mm::frame: freeing an out-of-range address");
        set_bit(self.bitmap, frame, false);
        // Prefer reusing freed low frames on the next allocation.
        self.next_hint = self.next_hint.min(frame);
    }

    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Mark the frame containing `phys` used, whatever it was before.
    ///
    /// For a frame that came from outside the allocator's own bookkeeping — the S3
    /// trampoline's page, taken from bootloader-reclaimable memory the bitmap never
    /// listed as free. Recording it here is what stops a later ordinary allocation from
    /// handing the same page to something else.
    pub fn mark_used(&mut self, phys: u64) {
        let f = phys / FRAME_SIZE;
        if f < self.frame_count {
            set_bit(self.bitmap, f, true);
        }
    }

    pub fn free_frame_count(&self) -> u64 {
        (0..self.frame_count).filter(|&f| !get_bit(self.bitmap, f)).count() as u64
    }
}

/// The page-table frame source x86's `paging::map_page` walks with. aarch64 does
/// not implement this trait yet — it has no 4 KiB walker to feed (its map is 1 GiB
/// and 2 MiB block descriptors), which is exactly the parity gap this module's
/// promotion to both arches is the first half of closing.
#[cfg(target_arch = "x86_64")]
impl crate::arch::x86_64::paging::FrameAllocator for BitmapFrameAllocator {
    fn allocate_frame(&mut self) -> Option<u64> {
        self.allocate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const K64: u64 = 64 * 1024;
    const K128: u64 = 128 * 1024;
    const M16: u64 = 16 * 1024 * 1024;

    #[test_case]
    fn run_fits_rejects_runs_above_the_limit() {
        // ISA DMA latches a 24-bit address: the whole run must be under 16 MiB.
        assert!(run_fits(M16 - 0x1000, 0x1000, M16, 0));
        assert!(!run_fits(M16 - 0x1000, 0x2000, M16, 0)); // ends past the limit
        assert!(!run_fits(M16, 0x1000, M16, 0)); // starts at the limit
    }

    #[test_case]
    fn run_fits_allows_a_run_ending_exactly_on_a_boundary() {
        // The off-by-one that matters: a run whose last byte is the final byte of
        // a 64 KiB block is legal. Comparing the END address instead of the last
        // byte would wrongly reject it and waste the whole block.
        assert!(run_fits(0, K64, M16, K64));
        assert!(run_fits(K64 - 0x1000, 0x1000, M16, K64));
    }

    #[test_case]
    fn run_fits_rejects_boundary_straddles() {
        // One byte over the block edge is a straddle.
        assert!(!run_fits(K64 - 0x1000, 0x2000, M16, K64));
        // A 16-bit channel's 128 KiB block: legal under 128 KiB, illegal across.
        assert!(run_fits(0, K128, M16, K128));
        assert!(!run_fits(K128 - 0x1000, 0x2000, M16, K128));
        // Same run, different boundary: fits a 128 KiB block, not a 64 KiB one.
        assert!(run_fits(K64, K64, M16, K128));
        assert!(!run_fits(K64 - 0x1000, 0x2000, M16, K64));
    }

    #[test_case]
    fn run_fits_boundary_zero_disables_the_straddle_check() {
        assert!(run_fits(K64 - 0x1000, 0x8000, M16, 0));
    }

    // `bitmap_placement` is the aarch64 constructor's sizing/placement decision.
    // It is tested here, on x86, because the unit suite only runs on x86 — the
    // same trick `ramlayout`/`edid` use to cover arch-specific pure logic.

    #[test_case]
    fn bitmap_placement_sizes_from_the_highest_usable_address() {
        // One frame per bit: 16 MiB of RAM is 4096 frames is 512 bytes.
        let (frames, bytes, base) = bitmap_placement([(0u64, M16)].into_iter()).unwrap();
        assert_eq!(frames, M16 / FRAME_SIZE);
        assert_eq!(bytes, (M16 / FRAME_SIZE / 8) as usize);
        assert_eq!(base, 0);
    }

    #[test_case]
    fn bitmap_placement_spans_a_hole_between_regions() {
        // aarch64 hardware really does report discontiguous RAM (a hole between
        // banks). The bitmap must cover up to the TOP of the last region, not the
        // sum of their lengths, or the high bank indexes off the end of the bitmap.
        let regions = [(0u64, M16), (M16 * 4, M16)];
        let (frames, _, _) = bitmap_placement(regions.into_iter()).unwrap();
        assert_eq!(frames, (M16 * 5) / FRAME_SIZE, "must span the hole, not sum the regions");
    }

    #[test_case]
    fn bitmap_placement_picks_a_region_that_can_hold_the_bitmap() {
        // A tiny first region cannot host the bitmap; placement must skip it
        // rather than scribbling past its end. The span has to be large enough
        // that the bitmap itself exceeds 4 KiB — at 1 bit per 4 KiB frame that
        // needs ~128 MiB of span, so a 32 MiB one would still fit and prove nothing.
        let high = M16 * 64;
        let (_, bytes, base) = bitmap_placement([(0u64, 0x1000), (high, M16)].into_iter()).unwrap();
        assert!(bytes > 0x1000, "test premise: the bitmap must not fit in the first region");
        assert_eq!(base, high, "must skip the region too small to hold it");
    }

    #[test_case]
    fn bitmap_placement_reports_failure_rather_than_picking_nothing() {
        // Every region too small: there is no heap yet, so there is nowhere else
        // to put the bitmap. `None` lets the caller say so; a silent fallback
        // would corrupt whatever follows the short region.
        assert!(bitmap_placement([(0u64, 0x1000), (M16 * 64, 0x1000)].into_iter()).is_none());
        // No regions at all is the degenerate case, not a panic.
        assert!(bitmap_placement(core::iter::empty()).is_none());
    }

    #[test_case]
    fn run_fits_handles_overflow_without_panicking() {
        // A caller asking for an absurd length must get `false`, not a wrapped
        // address that looks like it fits.
        assert!(!run_fits(u64::MAX - 0x100, 0x1000, M16, K64));
    }
}
