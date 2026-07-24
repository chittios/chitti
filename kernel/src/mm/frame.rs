//! Physical frame allocator: one bit per 4 KiB frame, direct-indexed by
//! `physical_address / FRAME_SIZE`, built from the Limine memory map.
//!
//! The bitmap's own backing storage is bootstrapped out of the first
//! usable region large enough to hold it (there is no heap yet — this
//! *is* what the heap gets built on top of), reached via the HHDM.

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

impl BitmapFrameAllocator {
    /// Build the allocator from Limine's memory map.
    ///
    /// # Safety
    /// `entries` must be the exact, still-valid memory map Limine
    /// returned, and `hhdm_offset` the real HHDM offset it reported —
    /// both are trusted verbatim to compute where physical memory is
    /// reachable and which frames are safe to hand out.
    pub unsafe fn init(entries: &[&MemmapEntry], hhdm_offset: u64) -> Self {
        // Size the bitmap off USABLE entries only: some firmware/
        // bootloader memory maps include a final huge RESERVED entry
        // covering the rest of the 64-bit address space as a sentinel,
        // which would otherwise inflate the bitmap to cover terabytes of
        // address space we will never allocate a frame from anyway.
        let max_addr = entries
            .iter()
            .filter(|e| e.entry_type == limine_protocol::MEMMAP_USABLE)
            .map(|e| e.base + e.length)
            .max()
            .unwrap_or(0);
        let frame_count = max_addr.div_ceil(FRAME_SIZE);
        let bitmap_bytes = (frame_count as usize).div_ceil(8);

        let region = entries
            .iter()
            .find(|e| e.entry_type == limine_protocol::MEMMAP_USABLE && e.length as usize >= bitmap_bytes)
            .expect("mm::frame: no usable region large enough to hold the frame bitmap");

        let bitmap_phys = region.base;
        let bitmap_ptr = (bitmap_phys + hhdm_offset) as *mut u8;
        // SAFETY: `region` is a USABLE entry at least `bitmap_bytes` long,
        // and `hhdm_offset` maps it to a valid, writable virtual address.
        let bitmap = unsafe { core::slice::from_raw_parts_mut(bitmap_ptr, bitmap_bytes) };
        bitmap.fill(0xff); // default: every frame used/reserved

        for entry in entries {
            if entry.entry_type == limine_protocol::MEMMAP_USABLE {
                mark_range(bitmap, entry.base, entry.length, false);
            }
        }

        // Re-reserve the frames the bitmap itself now occupies.
        let bitmap_frames = (bitmap_bytes as u64).div_ceil(FRAME_SIZE);
        mark_range(bitmap, bitmap_phys, bitmap_frames * FRAME_SIZE, true);

        Self { bitmap, frame_count, next_hint: 0 }
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

    pub fn free_frame_count(&self) -> u64 {
        (0..self.frame_count).filter(|&f| !get_bit(self.bitmap, f)).count() as u64
    }
}

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

    #[test_case]
    fn run_fits_handles_overflow_without_panicking() {
        // A caller asking for an absurd length must get `false`, not a wrapped
        // address that looks like it fits.
        assert!(!run_fits(u64::MAX - 0x100, 0x1000, M16, K64));
    }
}
