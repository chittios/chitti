//! A **split virtqueue** with several buffers in flight, built on the pure
//! arithmetic in [`super::layout`].
//!
//! The existing virtio drivers each hardcode a single request: descriptors 0..2
//! are written, the queue is kicked, and the driver spins until the used index
//! moves. That is fine for a block read and wrong for both devices added here —
//! virtio-serial parks a pool of receive buffers in the queue indefinitely while
//! transmits come and go, and virtio-9p wants its reply buffer already posted
//! when the request goes out. So descriptors are allocated from a free list and
//! chains are returned on completion.
//!
//! Everything here is polled. There is no interrupt path, because the scheduler
//! is cooperative and every caller is already inside a loop that pumps
//! `upkeep()`.

use super::layout::{
    DescFree, VirtqLayout, AVAIL_IDX, DESC_ADDR, DESC_FLAGS, DESC_LEN, DESC_NEXT, DESC_F_NEXT,
    DESC_F_WRITE, USED_IDX,
};
use super::{barrier, Transport};
use core::ptr::{read_volatile, write_volatile};

/// One physically-addressed buffer handed to the device.
#[derive(Clone, Copy, Debug)]
pub struct Buf {
    /// Physical address the device will read or write.
    pub phys: u64,
    pub len: u32,
}

/// A completed chain: the head descriptor id, and how many bytes the device
/// wrote into the chain's device-writable buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Completion {
    pub head: u16,
    pub len: u32,
}

/// A live split virtqueue.
pub struct Virtq {
    layout: VirtqLayout,
    /// CPU-visible base of the ring region.
    virt: u64,
    /// Device-visible (physical) base of the same region.
    phys: u64,
    free: DescFree,
    avail_idx: u16,
    last_used: u16,
    /// Chain length by head descriptor id, so a completion can return exactly
    /// the descriptors the chain used.
    chain_len: [u8; DescFree::MAX],
}

impl Virtq {
    /// Allocate and program queue `q` on `t`. `want` is the desired depth; the
    /// device's maximum wins if it is smaller, and a queue the device does not
    /// implement (max 0) yields `None`.
    pub fn setup(t: &mut dyn Transport, q: u16, want: u16) -> Option<Virtq> {
        let max = t.queue_max(q);
        if max == 0 {
            return None;
        }
        // A split virtqueue's size must be a power of two, and the free list is
        // bounded, so clamp to both before rounding down.
        let mut qsize = want.min(max).min(DescFree::MAX as u16);
        if !qsize.is_power_of_two() {
            qsize = qsize.next_power_of_two() / 2;
        }
        if qsize == 0 {
            return None;
        }
        let layout = VirtqLayout::new(qsize, t.used_align());
        let (phys, virt) = crate::mm::alloc_dma(layout.total)?;
        t.queue_set(q, phys, &layout);
        Some(Virtq {
            layout,
            virt,
            phys,
            free: DescFree::new(qsize),
            avail_idx: 0,
            last_used: 0,
            chain_len: [0; DescFree::MAX],
        })
    }

    /// Physical base of the ring region (for a driver that must report it).
    pub fn phys(&self) -> u64 {
        self.phys
    }

    /// Descriptors currently free.
    pub fn available(&self) -> usize {
        self.free.available()
    }

    unsafe fn w16(&self, off: usize, v: u16) {
        // SAFETY: `off` is inside the region `setup` allocated at `self.virt`.
        unsafe { write_volatile((self.virt + off as u64) as *mut u16, v) };
    }
    unsafe fn w32(&self, off: usize, v: u32) {
        // SAFETY: as above.
        unsafe { write_volatile((self.virt + off as u64) as *mut u32, v) };
    }
    unsafe fn w64(&self, off: usize, v: u64) {
        // SAFETY: as above.
        unsafe { write_volatile((self.virt + off as u64) as *mut u64, v) };
    }
    unsafe fn r16(&self, off: usize) -> u16 {
        // SAFETY: as above.
        unsafe { read_volatile((self.virt + off as u64) as *const u16) }
    }
    unsafe fn r32(&self, off: usize) -> u32 {
        // SAFETY: as above.
        unsafe { read_volatile((self.virt + off as u64) as *const u32) }
    }

    /// Post a chain of `out` (device-readable) then `inb` (device-writable)
    /// buffers. Returns the head descriptor id, or `None` if the queue has too
    /// few free descriptors — a caller must treat that as backpressure, not as
    /// an error.
    pub fn add(&mut self, out: &[Buf], inb: &[Buf]) -> Option<u16> {
        let n = out.len() + inb.len();
        if n == 0 || n > DescFree::MAX || self.free.available() < n {
            return None;
        }
        // Take all the ids up front: a partial allocation would have to be
        // unwound, and the check above already guarantees they are there.
        let mut ids = [0u16; DescFree::MAX];
        for id in ids.iter_mut().take(n) {
            *id = self.free.alloc()?;
        }

        for i in 0..n {
            let id = ids[i];
            let (buf, write) = if i < out.len() {
                (out[i], false)
            } else {
                (inb[i - out.len()], true)
            };
            let last = i + 1 == n;
            let mut flags = 0u16;
            if !last {
                flags |= DESC_F_NEXT;
            }
            if write {
                flags |= DESC_F_WRITE;
            }
            // SAFETY: descriptor offsets come from the layout for this queue.
            unsafe {
                self.w64(self.layout.desc_field(id, DESC_ADDR), buf.phys);
                self.w32(self.layout.desc_field(id, DESC_LEN), buf.len);
                self.w16(self.layout.desc_field(id, DESC_FLAGS), flags);
                self.w16(self.layout.desc_field(id, DESC_NEXT), if last { 0 } else { ids[i + 1] });
            }
        }

        let head = ids[0];
        self.chain_len[head as usize] = n as u8;
        // SAFETY: ring offsets from the layout.
        unsafe {
            self.w16(self.layout.avail_slot(self.avail_idx), head);
            // The device must see the descriptors before the index that
            // publishes them, or it can read a stale chain.
            barrier();
            self.avail_idx = self.avail_idx.wrapping_add(1);
            self.w16(self.layout.avail + AVAIL_IDX, self.avail_idx);
        }
        Some(head)
    }

    /// Tell the device about newly added chains on queue `q`.
    pub fn kick(&self, t: &dyn Transport, q: u16) {
        t.notify(q);
    }

    /// Take one completed chain, freeing its descriptors. `None` when the
    /// device has completed nothing new.
    pub fn take_used(&mut self) -> Option<Completion> {
        // SAFETY: ring offsets from the layout.
        let used_idx = unsafe { self.r16(self.layout.used + USED_IDX) };
        if used_idx == self.last_used {
            return None;
        }
        // Read the index before the element it publishes.
        barrier();
        let slot = self.layout.used_slot(self.last_used);
        // SAFETY: a used element is `{ u32 id, u32 len }`.
        let (head, len) = unsafe { (self.r32(slot) as u16, self.r32(slot + 4)) };
        self.last_used = self.last_used.wrapping_add(1);

        // Return every descriptor in the chain by walking `next`, bounded by
        // the recorded length so a device that corrupts the links cannot spin
        // us forever.
        let mut id = head;
        let count = self.chain_len[head as usize % DescFree::MAX].max(1);
        for _ in 0..count {
            // SAFETY: descriptor offsets from the layout.
            let next = unsafe { self.r16(self.layout.desc_field(id, DESC_NEXT)) };
            let flags = unsafe { self.r16(self.layout.desc_field(id, DESC_FLAGS)) };
            self.free.free_one(id);
            if flags & DESC_F_NEXT == 0 {
                break;
            }
            id = next;
        }
        self.chain_len[head as usize % DescFree::MAX] = 0;
        Some(Completion { head, len })
    }
}
