//! Kernel heap: a linked-list first-fit allocator ("linked-list to start,
//! buddy if fragmentation bites" per `CHITTI_OS_HANDOFF.md` Phase 1 scope)
//! over a fixed virtual range, mapped to real physical frames at init time
//! via the frame allocator + `arch::x86_64::paging`.
//!
//! Free blocks form an intrusive singly linked list threaded through the
//! freed memory itself, kept **sorted by address** so freeing can **coalesce**
//! with both neighbours in O(1) once the insert position is found. `alloc`
//! does a first-fit scan, splitting off any leftover tail large enough to
//! hold another free-list node. (Coalescing became necessary — the handoff
//! doc's "buddy if fragmentation bites" moment — when Cortex KV caches began
//! doubling repeatedly during long prefills: without merging, the freed
//! halves of every realloc fragmented the heap until multi-MiB allocations
//! failed with hundreds of MiB nominally free.)

use super::Locked;
use core::alloc::{GlobalAlloc, Layout};
use core::mem;
use core::ptr::null_mut;

pub const HEAP_START: u64 = 0xffff_a000_0000_0000;
// 512 MiB: Phase 3's Cortex (Qwen3.5 hybrid) needs real room -- each
// per-stream cache holds ~19 MiB of gated-DeltaNet recurrent state (18
// linear-attention layers x 16 heads x 128x128 f32) plus attention-layer KV
// that grows with every token; the agentic chat (tools JSON in the system
// prompt + thinking + multi-iteration tool loops) runs multi-thousand-token
// contexts, and the 248K-token vocab table plus vocab-sized logits add
// tens of MiB more. Backed by the frame allocator (0.8B runs with 3 GiB
// RAM, model ~774 MiB — ample headroom).
#[cfg(not(any(feature = "model-2b", feature = "model-4b", feature = "model-9b")))]
pub const HEAP_SIZE: usize = 512 * 1024 * 1024;
// The 2B and 4B models' per-forward state, KV/recurrent cache, and
// batched-prefill buffers sit between the 0.8B's and the 9B's; give the heap
// 512 MiB (placed at the top of RAM, past the model — see `mm::init`).
#[cfg(any(feature = "model-2b", feature = "model-4b"))]
pub const HEAP_SIZE: usize = 512 * 1024 * 1024;
// The 9B model has 33 layers, dim 4096, ffn 12288 and a 248K vocab, so its
// per-forward state, KV/recurrent cache, and batched-prefill buffers are far
// larger; give the heap 1 GiB (it sits at 0x2_00000000, past the model).
#[cfg(feature = "model-9b")]
pub const HEAP_SIZE: usize = 1024 * 1024 * 1024;

struct ListNode {
    size: usize,
    next: Option<&'static mut ListNode>,
}

impl ListNode {
    const fn new(size: usize) -> Self {
        Self { size, next: None }
    }

    fn start_addr(&self) -> usize {
        self as *const Self as usize
    }

    fn end_addr(&self) -> usize {
        self.start_addr() + self.size
    }
}

fn align_up(addr: usize, align: usize) -> usize {
    (addr + align - 1) & !(align - 1)
}

pub struct LinkedListAllocator {
    head: ListNode,
}

impl LinkedListAllocator {
    pub const fn empty() -> Self {
        Self { head: ListNode::new(0) }
    }

    /// Sum of all free regions on the free list (bytes). O(free-list length);
    /// used only by [`stats`] for `/info`, never on a hot path.
    fn free_bytes(&self) -> usize {
        let mut total = 0;
        let mut cur = self.head.next.as_deref();
        while let Some(node) = cur {
            total += node.size;
            cur = node.next.as_deref();
        }
        total
    }

    /// # Safety
    /// `[start, start + size)` must be valid, mapped, exclusively-owned
    /// memory; called exactly once, from `mm::heap::init`.
    unsafe fn init(&mut self, start: usize, size: usize) {
        unsafe { self.add_free_region(start, size) };
    }

    unsafe fn add_free_region(&mut self, addr: usize, mut size: usize) {
        assert_eq!(align_up(addr, mem::align_of::<ListNode>()), addr);
        assert!(size >= mem::size_of::<ListNode>());

        // SAFETY: caller guarantees `addr..addr+size` is valid, aligned,
        // owned memory not referenced anywhere else. The raw-pointer walk
        // below only dereferences live list nodes (or the head sentinel).
        unsafe {
            // Address-ordered insert: find `prev`, the last node (or the head
            // sentinel) starting below `addr`. Keeping the list sorted makes
            // both merges O(1) here and preserves first-fit behaviour.
            let mut prev: *mut ListNode = &mut self.head;
            while let Some(next) = (*prev).next.as_deref_mut() {
                if next.start_addr() < addr {
                    prev = next;
                } else {
                    break;
                }
            }
            // Coalesce with the successor when `addr..addr+size` abuts it.
            if let Some(next) = (*prev).next.take() {
                if addr + size == next.start_addr() {
                    size += next.size;
                    (*prev).next = next.next.take();
                } else {
                    (*prev).next = Some(next);
                }
            }
            // Coalesce with the predecessor when it ends exactly at `addr`
            // (never the sentinel, which is a zero-size node outside the heap).
            let sentinel = &mut self.head as *mut ListNode;
            if prev != sentinel && (*prev).end_addr() == addr {
                (*prev).size += size;
                return;
            }
            // No predecessor merge: link a fresh node in address order.
            let mut node = ListNode::new(size);
            node.next = (*prev).next.take();
            let node_ptr = addr as *mut ListNode;
            node_ptr.write(node);
            (*prev).next = Some(&mut *node_ptr);
        }
    }

    fn alloc_from_region(region: &ListNode, size: usize, align: usize) -> Result<usize, ()> {
        let alloc_start = align_up(region.start_addr(), align);
        let alloc_end = alloc_start.checked_add(size).ok_or(())?;
        if alloc_end > region.end_addr() {
            return Err(());
        }
        let excess_size = region.end_addr() - alloc_end;
        if excess_size > 0 && excess_size < mem::size_of::<ListNode>() {
            // Leftover tail too small to host a free-list node: reject
            // rather than leak it as unreachable, unusable space.
            return Err(());
        }
        Ok(alloc_start)
    }

    fn find_region(&mut self, size: usize, align: usize) -> Option<(&'static mut ListNode, usize)> {
        let mut current = &mut self.head;
        while let Some(ref mut region) = current.next {
            if let Ok(alloc_start) = Self::alloc_from_region(region, size, align) {
                let next = region.next.take();
                let region = current.next.take().unwrap();
                current.next = next;
                return Some((region, alloc_start));
            }
            current = current.next.as_mut().unwrap();
        }
        None
    }

    fn size_align(layout: Layout) -> (usize, usize) {
        let layout = layout
            .align_to(mem::align_of::<ListNode>())
            .expect("mm::heap: adjusting layout alignment failed")
            .pad_to_align();
        (layout.size().max(mem::size_of::<ListNode>()), layout.align())
    }
}

// SAFETY: `Locked` serializes every access via `with` (interrupts
// disabled), so the linked-list mutations above never race.
unsafe impl GlobalAlloc for Locked<LinkedListAllocator> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let (size, align) = LinkedListAllocator::size_align(layout);
        self.with(|allocator| {
            if let Some((region, alloc_start)) = allocator.find_region(size, align) {
                let alloc_end = alloc_start.checked_add(size).expect("mm::heap: overflow");
                let excess_size = region.end_addr() - alloc_end;
                if excess_size > 0 {
                    // SAFETY: `alloc_end..alloc_end+excess_size` is the
                    // unused tail of `region`, already verified large
                    // enough to hold a `ListNode` by `alloc_from_region`.
                    unsafe { allocator.add_free_region(alloc_end, excess_size) };
                }
                alloc_start as *mut u8
            } else {
                null_mut()
            }
        })
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let (size, _) = LinkedListAllocator::size_align(layout);
        self.with(|allocator| {
            // SAFETY: caller (the `alloc::GlobalAlloc` contract) guarantees
            // `ptr..ptr+size` was returned by a prior `alloc` with this
            // same layout and is not aliased elsewhere.
            unsafe { allocator.add_free_region(ptr as usize, size) };
        });
    }
}

#[global_allocator]
static ALLOCATOR: Locked<LinkedListAllocator> = Locked::new(LinkedListAllocator::empty());

/// Heap usage snapshot for `/info`: `(total, free, used)` bytes. `total` is the
/// compile-time [`HEAP_SIZE`]; `free` walks the free list; `used = total - free`.
pub fn stats() -> (usize, usize, usize) {
    let free = ALLOCATOR.with(|a| a.free_bytes());
    let total = HEAP_SIZE;
    (total, free, total.saturating_sub(free))
}

/// x86: map `HEAP_SIZE` bytes at `HEAP_START` to freshly allocated physical
/// frames (via the frame allocator + paging) and hand the range to the
/// allocator. Must run after `fpu::enable_nx()` (heap pages are `NO_EXECUTE`)
/// and after the frame allocator is populated.
#[cfg(target_arch = "x86_64")]
pub fn init(frame_allocator: &Locked<Option<super::frame::BitmapFrameAllocator>>) {
    use super::frame::FRAME_SIZE;
    use crate::arch::x86_64::paging::{self, NO_EXECUTE, PRESENT, WRITABLE};

    struct FrameAllocatorAdapter<'a>(&'a Locked<Option<super::frame::BitmapFrameAllocator>>);
    impl paging::FrameAllocator for FrameAllocatorAdapter<'_> {
        fn allocate_frame(&mut self) -> Option<u64> {
            self.0.with(|slot| slot.as_mut().and_then(|a| a.allocate()))
        }
    }

    let mut adapter = FrameAllocatorAdapter(frame_allocator);
    for page in (HEAP_START..HEAP_START + HEAP_SIZE as u64).step_by(FRAME_SIZE as usize) {
        let phys = frame_allocator
            .with(|slot| slot.as_mut().expect("mm::heap: frame allocator not initialized").allocate())
            .expect("mm::heap: out of physical frames while mapping the heap");
        paging::map_page(page, phys, PRESENT | WRITABLE | NO_EXECUTE, &mut adapter);
    }

    // SAFETY: the whole [HEAP_START, HEAP_START + HEAP_SIZE) range was just
    // mapped above, is not referenced anywhere else yet, and HEAP_START is
    // page- (hence `ListNode`-) aligned.
    ALLOCATOR.with(|allocator| unsafe { allocator.init(HEAP_START as usize, HEAP_SIZE) });
    crate::ktrace::log_fmt(format_args!("mm: heap ready, {HEAP_SIZE} bytes mapped at {HEAP_START:#x}"));
}

/// aarch64: hand a fixed, already-mapped RAM region to the allocator. No
/// paging/frame machinery -- the MMU identity-maps RAM as normal cacheable
/// (`arch::aarch64::mmu`), so `[base, base + size)` is directly usable.
///
/// The caller (`mm::init`) guarantees the region is within mapped RAM and
/// otherwise unused.
#[cfg(target_arch = "aarch64")]
pub fn init_static(base: usize, size: usize) {
    // SAFETY: `[base, base+size)` is identity-mapped normal RAM, `ListNode`-
    // aligned (page-aligned base), and referenced nowhere else.
    ALLOCATOR.with(|allocator| unsafe { allocator.init(base, size) });
}
