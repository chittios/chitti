//! Kernel heap: a linked-list first-fit allocator ("linked-list to start,
//! buddy if fragmentation bites" per `CHITTI_OS_HANDOFF.md` Phase 1 scope)
//! over a fixed virtual range, mapped to real physical frames at init time
//! via the frame allocator + `arch::x86_64::paging`.
//!
//! Free blocks form an intrusive singly linked list threaded through the
//! freed memory itself. `alloc` does a first-fit scan, splitting off any
//! leftover tail large enough to hold another free-list node; `dealloc`
//! just pushes the freed region back onto the list head. There is no
//! coalescing of adjacent free blocks — the known limitation the handoff
//! doc's "buddy if fragmentation bites" already anticipates.

use super::frame::FRAME_SIZE;
use super::Locked;
use crate::arch::x86_64::paging::{self, NO_EXECUTE, PRESENT, WRITABLE};
use core::alloc::{GlobalAlloc, Layout};
use core::mem;
use core::ptr::null_mut;

pub const HEAP_START: u64 = 0xffff_a000_0000_0000;
// 256 MiB: Phase 3's Cortex (Qwen3.5 hybrid) needs real room -- each
// per-stream cache holds ~19 MiB of gated-DeltaNet recurrent state (18
// linear-attention layers x 16 heads x 128x128 f32), the batching test
// runs two streams concurrently, and the 248K-token vocab table plus
// vocab-sized logits add several MiB more. The linked-list allocator does
// no coalescing, so headroom also absorbs fragmentation from transient
// caches. Backed by the frame allocator, which has gigabytes free.
pub const HEAP_SIZE: usize = 256 * 1024 * 1024;

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

    /// # Safety
    /// `[start, start + size)` must be valid, mapped, exclusively-owned
    /// memory; called exactly once, from `mm::heap::init`.
    unsafe fn init(&mut self, start: usize, size: usize) {
        unsafe { self.add_free_region(start, size) };
    }

    unsafe fn add_free_region(&mut self, addr: usize, size: usize) {
        assert_eq!(align_up(addr, mem::align_of::<ListNode>()), addr);
        assert!(size >= mem::size_of::<ListNode>());

        let mut node = ListNode::new(size);
        node.next = self.head.next.take();
        let node_ptr = addr as *mut ListNode;
        // SAFETY: caller guarantees `addr..addr+size` is valid, aligned,
        // owned memory not referenced anywhere else.
        unsafe {
            node_ptr.write(node);
            self.head.next = Some(&mut *node_ptr);
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

struct FrameAllocatorAdapter<'a>(&'a Locked<Option<super::frame::BitmapFrameAllocator>>);

impl paging::FrameAllocator for FrameAllocatorAdapter<'_> {
    fn allocate_frame(&mut self) -> Option<u64> {
        self.0.with(|slot| slot.as_mut().and_then(|a| a.allocate()))
    }
}

/// Map `HEAP_SIZE` bytes at `HEAP_START` to freshly allocated physical
/// frames and hand the range to the allocator. Must run after
/// `arch::x86_64::fpu::enable_nx()` (heap pages are mapped `NO_EXECUTE`)
/// and after `frame_allocator` has been populated.
pub fn init(frame_allocator: &Locked<Option<super::frame::BitmapFrameAllocator>>) {
    let mut adapter = FrameAllocatorAdapter(frame_allocator);
    for page in (HEAP_START..HEAP_START + HEAP_SIZE as u64).step_by(FRAME_SIZE as usize) {
        let phys = frame_allocator
            .with(|slot| slot.as_mut().expect("mm::heap: frame allocator not initialized").allocate())
            .expect("mm::heap: out of physical frames while mapping the heap");
        paging::map_page(page, phys, PRESENT | WRITABLE | NO_EXECUTE, &mut adapter);
    }

    // SAFETY: the whole [HEAP_START, HEAP_START + HEAP_SIZE) range was
    // just mapped above, is not referenced anywhere else yet, and
    // HEAP_START is page- (hence `ListNode`-) aligned.
    ALLOCATOR.with(|allocator| unsafe { allocator.init(HEAP_START as usize, HEAP_SIZE) });

    crate::ktrace::log_fmt(format_args!("mm: heap ready, {HEAP_SIZE} bytes mapped at {HEAP_START:#x}"));
}
