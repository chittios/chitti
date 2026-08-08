//! **Pure split-virtqueue arithmetic** — ring offsets and descriptor
//! bookkeeping, with no MMIO and no allocation.
//!
//! Every virtio driver in the tree grew its own copy of these offsets
//! (`avail + 4 + idx * 2`, `used + 4 + slot * 8`, …), open-coded at each use
//! site. They are easy to get subtly wrong and impossible to test where they
//! were written: the mmio drivers live under [`crate::arch::aarch64`], which is
//! `cfg`'d out of the test build entirely (the same trap `framebuffer/`
//! documents). So the arithmetic lives here, where `cargo xtask test` compiles
//! it, and [`super::queue`] is the thin unsafe shell that pokes memory.
//!
//! The layout is the **split** virtqueue of virtio 1.0 §2.6 — the only one
//! either transport here uses (no packed rings). All three regions are carved
//! out of **one contiguous allocation**, because the legacy mmio transport
//! (`QUEUE_PFN`) can only be handed a single page-aligned base; the modern
//! transports take three addresses and are happy to have them adjacent.

/// Descriptor flag: buffer continues in `next`.
pub const DESC_F_NEXT: u16 = 1;
/// Descriptor flag: buffer is **device-write** (i.e. an "in" buffer).
pub const DESC_F_WRITE: u16 = 2;

/// Byte offsets of the fields of one 16-byte descriptor.
pub const DESC_ADDR: usize = 0; // u64
pub const DESC_LEN: usize = 8; // u32
pub const DESC_FLAGS: usize = 12; // u16
pub const DESC_NEXT: usize = 14; // u16
/// One descriptor is 16 bytes.
pub const DESC_SIZE: usize = 16;

/// Byte offsets within the available ring.
pub const AVAIL_FLAGS: usize = 0; // u16
pub const AVAIL_IDX: usize = 2; // u16
pub const AVAIL_RING: usize = 4; // u16[qsize]

/// Byte offsets within the used ring.
pub const USED_FLAGS: usize = 0; // u16
pub const USED_IDX: usize = 2; // u16
pub const USED_RING: usize = 4; // (u32 id, u32 len)[qsize]
/// One used-ring element is 8 bytes: `{ u32 id, u32 len }`.
pub const USED_ELEM_SIZE: usize = 8;

/// Round `v` up to a multiple of `align` (a power of two).
pub const fn align_up(v: usize, align: usize) -> usize {
    (v + align - 1) & !(align - 1)
}

/// Where the three rings sit inside one contiguous region, and how big it is.
///
/// `used_align` is the transport's requirement: the modern transports need only
/// natural alignment (4), while the legacy mmio transport aligns the used ring
/// to `QUEUE_ALIGN` — a page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VirtqLayout {
    pub qsize: u16,
    /// Descriptor table offset — always 0, so the region base *is* the table.
    pub desc: usize,
    pub avail: usize,
    pub used: usize,
    /// Total bytes to allocate.
    pub total: usize,
}

impl VirtqLayout {
    /// Lay out a queue of `qsize` descriptors with the transport's `used_align`.
    ///
    /// `qsize` must be a power of two (virtio 1.0 §2.6), which is what makes
    /// `idx % qsize` a mask and lets a `u16` index wrap into a slot for free.
    pub const fn new(qsize: u16, used_align: usize) -> VirtqLayout {
        let n = qsize as usize;
        let desc = 0;
        let avail = desc + n * DESC_SIZE;
        // avail: flags + idx + ring[n] (+ the 2-byte used_event we never use but
        // must not overlap — the device may write it under EVENT_IDX).
        let avail_end = avail + AVAIL_RING + n * 2 + 2;
        let used = align_up(avail_end, used_align);
        // used: flags + idx + ring[n] (+ the 2-byte avail_event, same reason).
        let total = used + USED_RING + n * USED_ELEM_SIZE + 2;
        VirtqLayout { qsize, desc, avail, used, total }
    }

    /// Offset of descriptor `i`'s field at `field` (e.g. [`DESC_ADDR`]).
    pub const fn desc_field(&self, i: u16, field: usize) -> usize {
        self.desc + (i as usize % self.qsize as usize) * DESC_SIZE + field
    }

    /// Offset of available-ring slot for the (free-running) index `idx`.
    pub const fn avail_slot(&self, idx: u16) -> usize {
        self.avail + AVAIL_RING + (idx as usize % self.qsize as usize) * 2
    }

    /// Offset of used-ring element for the (free-running) index `idx`.
    pub const fn used_slot(&self, idx: u16) -> usize {
        self.used + USED_RING + (idx as usize % self.qsize as usize) * USED_ELEM_SIZE
    }
}

/// A free list over descriptor ids, so several requests can be in flight.
///
/// The single-request drivers in the tree hardcode descriptors 0..2 and wait for
/// completion before reusing them. virtio-serial cannot: it keeps a pool of
/// receive buffers parked in the queue while transmits come and go, so ids must
/// be allocated and returned. This is a plain LIFO stack — order does not
/// matter, only that an id is never handed out twice.
#[derive(Debug)]
pub struct DescFree {
    free: [u16; Self::MAX],
    len: usize,
}

impl DescFree {
    /// Largest queue this can manage. 128 descriptors is far more than either
    /// device needs and keeps the stack a fixed-size array (no allocation in a
    /// path that runs from `upkeep`).
    pub const MAX: usize = 128;

    /// A free list holding every id in `0..qsize`.
    pub fn new(qsize: u16) -> DescFree {
        let n = (qsize as usize).min(Self::MAX);
        let mut free = [0u16; Self::MAX];
        // Pushed high-to-low so the first `alloc` hands out id 0 — which makes a
        // trace of a fresh queue readable rather than counting backwards.
        for (slot, id) in (0..n).rev().enumerate() {
            free[slot] = id as u16;
        }
        DescFree { free, len: n }
    }

    /// How many descriptors are available.
    pub fn available(&self) -> usize {
        self.len
    }

    /// Take a descriptor id, or `None` when the queue is full.
    pub fn alloc(&mut self) -> Option<u16> {
        if self.len == 0 {
            return None;
        }
        self.len -= 1;
        Some(self.free[self.len])
    }

    /// Return a descriptor id to the pool.
    ///
    /// Silently ignores an overflow rather than panicking: this runs on the
    /// completion path, and a double free is a driver bug that should not take
    /// the machine down mid-poll.
    pub fn free_one(&mut self, id: u16) {
        if self.len < Self::MAX {
            self.free[self.len] = id;
            self.len += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn layout_places_the_three_rings_without_overlap() {
        let l = VirtqLayout::new(8, 4);
        assert_eq!(l.desc, 0);
        // 8 descriptors * 16 bytes.
        assert_eq!(l.avail, 128);
        // avail is 4 + 8*2 + 2 = 22 bytes, so used starts at 150 rounded to 4.
        assert_eq!(l.used, align_up(128 + 4 + 16 + 2, 4));
        assert_eq!(l.used, 152);
        // used is 4 + 8*8 + 2 = 70 bytes.
        assert_eq!(l.total, 152 + 70);
        // Regions are ordered and non-overlapping.
        assert!(l.desc < l.avail && l.avail < l.used && l.used < l.total);
    }

    #[test_case]
    fn legacy_alignment_pushes_the_used_ring_to_a_page() {
        // The legacy mmio transport aligns the used ring to QUEUE_ALIGN (4096);
        // getting this wrong puts the used ring where the device does not look,
        // so completions never arrive and the driver hangs rather than erroring.
        let l = VirtqLayout::new(8, 4096);
        assert_eq!(l.used, 4096);
        assert_eq!(l.used % 4096, 0);
        // The modern layout of the same queue is much smaller.
        assert!(VirtqLayout::new(8, 4).total < l.total);
    }

    #[test_case]
    fn ring_slots_wrap_by_masking_the_free_running_index() {
        let l = VirtqLayout::new(4, 4);
        // A u16 index runs free and wraps into a slot.
        assert_eq!(l.avail_slot(0), l.avail + AVAIL_RING);
        assert_eq!(l.avail_slot(4), l.avail_slot(0));
        assert_eq!(l.avail_slot(5), l.avail_slot(1));
        assert_eq!(l.used_slot(0), l.used + USED_RING);
        assert_eq!(l.used_slot(4), l.used_slot(0));
        // Descriptor fields are 16 bytes apart, and wrap the same way.
        assert_eq!(l.desc_field(1, DESC_ADDR) - l.desc_field(0, DESC_ADDR), DESC_SIZE);
        assert_eq!(l.desc_field(0, DESC_FLAGS), DESC_FLAGS);
        assert_eq!(l.desc_field(4, DESC_ADDR), l.desc_field(0, DESC_ADDR));
    }

    #[test_case]
    fn desc_free_never_hands_out_an_id_twice() {
        let mut f = DescFree::new(4);
        assert_eq!(f.available(), 4);
        let mut got = alloc::vec::Vec::new();
        while let Some(id) = f.alloc() {
            assert!(!got.contains(&id), "id {id} handed out twice");
            got.push(id);
        }
        got.sort_unstable();
        assert_eq!(got, alloc::vec![0, 1, 2, 3]);
        // Exhausted: allocation fails rather than aliasing a live descriptor.
        assert_eq!(f.alloc(), None);
        assert_eq!(f.available(), 0);
        // Returned ids come back.
        f.free_one(2);
        assert_eq!(f.available(), 1);
        assert_eq!(f.alloc(), Some(2));
    }

    #[test_case]
    fn desc_free_is_bounded_by_max() {
        // A queue larger than MAX is clamped rather than overflowing the array.
        let f = DescFree::new(4096);
        assert_eq!(f.available(), DescFree::MAX);
        // A stray double-free cannot grow the pool past its capacity.
        let mut f = DescFree::new(2);
        for _ in 0..DescFree::MAX * 2 {
            f.free_one(0);
        }
        assert!(f.available() <= DescFree::MAX);
    }
}
