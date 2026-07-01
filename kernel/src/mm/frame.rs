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
