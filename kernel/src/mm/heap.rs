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
use core::sync::atomic::{AtomicU64, Ordering};

/// Allocator pressure counters (diagnosis: is the first-fit scan the ONNX
/// interpreter's bottleneck?). `SCAN_STEPS / ALLOC_CALLS` = average free-list
/// nodes walked per allocation; read via [`alloc_stats`].
static ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static SCAN_STEPS: AtomicU64 = AtomicU64::new(0);

/// `(allocations, free-list steps walked)` since boot.
pub fn alloc_stats() -> (u64, u64) {
    (ALLOC_CALLS.load(Ordering::Relaxed), SCAN_STEPS.load(Ordering::Relaxed))
}

pub const HEAP_START: u64 = 0xffff_a000_0000_0000;
// 512 MiB: Phase 3's Cortex (Qwen3.5 hybrid) needs real room -- each
// per-stream cache holds ~19 MiB of gated-DeltaNet recurrent state (18
// linear-attention layers x 16 heads x 128x128 f32) plus attention-layer KV
// that grows with every token; the agentic chat (tools JSON in the system
// prompt + thinking + multi-iteration tool loops) runs multi-thousand-token
// contexts, and the 248K-token vocab table plus vocab-sized logits add
// tens of MiB more. Backed by the frame allocator (0.8B runs with 3 GiB
// RAM, model ~774 MiB — ample headroom).
// 1 GiB: the 0.8B LLM needs ~512 MiB, but the ONNX voice runtime (KittenTTS's
// 78 MiB model + its activations, run through `onnx::exec`) needs more headroom,
// and a linked-list allocator fragments — so give it room. Sits at the top of
// RAM past the model (see `mm::init`), so the VM needs enough RAM (≥ 4 GiB).
#[cfg(not(any(feature = "model-2b", feature = "model-4b", feature = "model-9b")))]
pub const HEAP_SIZE: usize = 1024 * 1024 * 1024;
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
        ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        let mut current = &mut self.head;
        while let Some(ref mut region) = current.next {
            SCAN_STEPS.fetch_add(1, Ordering::Relaxed);
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
        // Per-task heap quota (agents): refuse before touching the free list so
        // a runaway agent cannot exhaust the shared arena for the shell.
        if !crate::sched::charge_heap(size) {
            OOM_FAILURES.fetch_add(1, Ordering::Relaxed);
            return null_mut();
        }
        let first = self.try_alloc(size, align);
        if !first.is_null() {
            return first;
        }
        // Out of room in the current arena — take more physical memory and retry
        // once. `grow` **must** be called with the allocator lock released:
        // `Locked` is not reentrant, so growing from inside `try_alloc`'s critical
        // section would deadlock rather than fail.
        if grow(size) {
            let p = self.try_alloc(size, align);
            if !p.is_null() {
                return p;
            }
        }
        // Still out: drop reclaimable caches and try once more.
        if super::reclaim::run() > 0 {
            let p = self.try_alloc(size, align);
            if !p.is_null() {
                return p;
            }
            if grow(size) {
                let p = self.try_alloc(size, align);
                if !p.is_null() {
                    return p;
                }
            }
        }
        // Undo the charge — the bytes were never handed out.
        crate::sched::uncharge_heap(size);
        OOM_FAILURES.fetch_add(1, Ordering::Relaxed);
        null_mut()
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let (size, _) = LinkedListAllocator::size_align(layout);
        crate::sched::uncharge_heap(size);
        self.with(|allocator| {
            // SAFETY: caller (the `alloc::GlobalAlloc` contract) guarantees
            // `ptr..ptr+size` was returned by a prior `alloc` with this
            // same layout and is not aliased elsewhere.
            unsafe { allocator.add_free_region(ptr as usize, size) };
        });
    }
}

/// Called from the global alloc-error path when `alloc` returned null and Rust
/// is about to panic. Tries one last reclaim, then **OOM-kills** the current
/// non-bootstrap task so the shell survives. Bootstrap still panics.
/// Decimal-format `n` into `buf` without allocating — the OOM path cannot use
/// `format!`, which is what failed to begin with.
fn usize_to_str(mut n: usize, buf: &mut [u8; 20]) -> &str {
    if n == 0 {
        buf[0] = b'0';
        return core::str::from_utf8(&buf[..1]).unwrap_or("0");
    }
    let mut i = buf.len();
    while n > 0 && i > 0 {
        i -= 1;
        buf[i] = b'0' + (n % 10) as u8;
        n /= 10;
    }
    core::str::from_utf8(&buf[i..]).unwrap_or("?")
}

/// Set while [`on_alloc_error`] is running, so a failure *inside* the recovery
/// path cannot re-enter it.
///
/// Recovery allocates: `reclaim::run()` and `grow()` both do. When the heap is
/// genuinely exhausted those allocations fail too, and the handler is entered
/// again — from inside itself, holding the allocator's own lock. It then spins
/// on that lock forever with interrupts disabled, which is a dead machine: no
/// output, no Ctrl+C, no scheduler. Recovery is attempted exactly once.
static IN_OOM: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

pub fn on_alloc_error(layout: Layout) -> ! {
    let size = layout.size();
    let reentered = IN_OOM.swap(true, Ordering::Relaxed);
    // Report on the **raw UART only**. The full ktrace path mirrors into the
    // framebuffer console, which takes a non-reentrant spinlock — and an
    // allocation can fail *inside* a critical section that already holds it
    // (painting allocates). Logging from here then spun forever with interrupts
    // disabled: the machine stopped dead, printing nothing, while trying to
    // print "out of memory". A page that exhausted the heap took the whole OS
    // with it, and the only escape was killing the VM.
    crate::serial::write_str_raw("mm: OOM allocating ");
    let mut buf = [0u8; 20];
    crate::serial::write_str_raw(usize_to_str(size, &mut buf));
    crate::serial::write_str_raw(" bytes (after grow+reclaim)\n");
    if !reentered {
        // One more reclaim pass — a hook registered after the first pass may
        // help. Both of these ALLOCATE, which is why they run only on the first
        // entry: on the second they would deadlock in the allocator we are here
        // because of.
        let _ = super::reclaim::run();
        if grow(size.max(GROW_CHUNK)) {
            // Cannot usefully return the pointer to the original caller from
            // here; the allocation already failed. Fall through to kill / panic.
        }
    } else {
        crate::serial::write_str_raw("mm: OOM while handling OOM — skipping recovery\n");
    }
    // Kill the current agent task rather than the whole OS.
    if crate::sched::initialized() {
        let cur = crate::sched::current_task_id();
        if cur != 0 {
            super::reclaim::note_oom_kill();
            crate::serial::write_str_raw("mm: OOM-killing task (quota or arena exhausted)\n");
            // Does not return when it succeeds (switches to another task).
            IN_OOM.store(false, Ordering::Relaxed);
            crate::sched::fault_current_task("oom");
        }
    }
    // Raw serial, not `panic!`'s formatter + console: the panic path itself
    // logs through the framebuffer, and this is exactly the moment that cannot
    // afford another lock.
    crate::serial::write_str_raw("mm: out of memory (bootstrap / no kill path) — halting\n");
    panic!("mm: out of memory");
}

impl Locked<LinkedListAllocator> {
    /// One first-fit attempt against the current arena. Split out of
    /// [`GlobalAlloc::alloc`] so growing can happen with this lock *released*.
    fn try_alloc(&self, size: usize, align: usize) -> *mut u8 {
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
}

#[global_allocator]
static ALLOCATOR: Locked<LinkedListAllocator> = Locked::new(LinkedListAllocator::empty());

/// Bytes added to the arena since boot by [`grow`].
static GROWN_BYTES: AtomicU64 = AtomicU64::new(0);
/// Allocations that failed even after a growth attempt.
static OOM_FAILURES: AtomicU64 = AtomicU64::new(0);
/// Reentrancy guard for [`grow`]. Without it a growth that itself needed to
/// allocate (a `ktrace` line, say) could recurse into `alloc` -> `grow` forever.
static GROWING: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Smallest chunk [`grow`] takes, so a run of small allocations near the limit
/// does not turn into a run of frame-allocator round trips.
const GROW_CHUNK: usize = 8 * 1024 * 1024;

/// Where physical address `phys` is already readable. The heap grows into memory
/// that is **already mapped** on both arches — through Limine's HHDM on x86, and
/// through the boot identity map on aarch64 — so growth needs no page-table work
/// and stays one arch-neutral function.
#[cfg(target_arch = "x86_64")]
fn phys_to_arena(phys: u64) -> usize {
    crate::arch::x86_64::paging::phys_to_virt(phys) as usize
}
#[cfg(not(target_arch = "x86_64"))]
fn phys_to_arena(phys: u64) -> usize {
    phys as usize // identity map
}

/// Add at least `min_bytes` of physical memory to the heap arena, returning
/// whether anything was added.
///
/// **The arena does not have to be contiguous**, which is what makes this simple
/// and identical on both arches: `add_free_region` inserts by address into a
/// sorted free list, so a fresh run of frames anywhere in RAM is just another
/// region. There is no need to extend the original `HEAP_START` mapping, and so
/// no page-table work and no per-arch path.
///
/// Requires the frame allocator, which is why this is also the first thing that
/// makes P0's aarch64 frame allocator load-bearing rather than merely present.
/// A boot path that declined to build one (see `mm::aarch64_frames`) simply
/// cannot grow, and reports that by returning `false`.
///
/// Must be called with the allocator lock **not** held.
pub fn grow(min_bytes: usize) -> bool {
    // Never recurse: `ktrace` below allocates, and an allocation inside a growth
    // that was triggered by an allocation is how you get an unbounded chain.
    if GROWING.swap(true, Ordering::Acquire) {
        return false;
    }
    let want = min_bytes.max(GROW_CHUNK).next_multiple_of(super::frame::FRAME_SIZE as usize);
    let frames = (want / super::frame::FRAME_SIZE as usize) as u64;
    let phys = super::FRAME_ALLOCATOR
        .with(|slot| slot.as_mut().and_then(|a| a.allocate_contiguous(frames)));
    let added = match phys {
        Some(phys) => {
            let addr = phys_to_arena(phys);
            // SAFETY: `frames` freshly-allocated, exclusively-owned, contiguous
            // frames, reachable at `addr` (HHDM on x86, identity on aarch64) and
            // referenced nowhere else. Frame-aligned, so `ListNode`-aligned and
            // far larger than one node.
            unsafe { ALLOCATOR.with(|a| a.add_free_region(addr, want)) };
            GROWN_BYTES.fetch_add(want as u64, Ordering::Relaxed);
            true
        }
        None => false,
    };
    GROWING.store(false, Ordering::Release);
    if added {
        crate::ktrace::log_fmt(format_args!("mm: heap grew by {} MiB", want >> 20));
    }
    added
}

/// `(bytes grown since boot, allocations that failed anyway)`.
pub fn growth_stats() -> (u64, u64) {
    (GROWN_BYTES.load(Ordering::Relaxed), OOM_FAILURES.load(Ordering::Relaxed))
}

/// Heap usage snapshot for `/info`: `(total, free, used)` bytes. `total` is the
/// initial arena plus everything [`grow`] has since added — a fixed
/// [`HEAP_SIZE`] would make `used` climb past `total` after the first growth.
pub fn stats() -> (usize, usize, usize) {
    let free = ALLOCATOR.with(|a| a.free_bytes());
    let total = HEAP_SIZE + GROWN_BYTES.load(Ordering::Relaxed) as usize;
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
    impl super::frame::TableFrames for FrameAllocatorAdapter<'_> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test_case]
    fn growth_adds_capacity_and_stats_track_it() {
        // `total` must include grown bytes: reporting the compile-time HEAP_SIZE
        // would let `used` climb past `total` after the first growth and print a
        // percentage over 100.
        let (total_before, _, _) = stats();
        let (grown_before, _) = growth_stats();
        assert!(grow(1), "growth needs a frame allocator; this arch/boot path has one under test");
        let (grown_after, _) = growth_stats();
        let (total_after, _, _) = stats();
        assert!(grown_after > grown_before, "grow must report the bytes it added");
        assert_eq!(
            total_after - total_before,
            (grown_after - grown_before) as usize,
            "`total` must grow by exactly what was added"
        );
        // A minimum chunk, so a run of small allocations near the limit does not
        // become a run of frame-allocator round trips.
        assert!(grown_after - grown_before >= GROW_CHUNK as u64);
    }

    #[test_case]
    fn the_grown_region_is_actually_usable_memory() {
        // The region is handed over by address; if that address were wrong (a
        // missing HHDM translation on x86, say) the free list would look healthy
        // and the first write would fault. So write the whole thing and read it
        // back, which is the only check that distinguishes the two.
        let (grown_before, _) = growth_stats();
        assert!(grow(1));
        assert!(growth_stats().0 > grown_before);
        // Allocate more than one chunk's worth in pieces, touching every byte.
        let mut blocks: Vec<Vec<u8>> = Vec::new();
        for i in 0..8u8 {
            let mut b = alloc::vec![0u8; 1 << 20];
            b[0] = i;
            b[(1 << 20) - 1] = i ^ 0xff;
            blocks.push(b);
        }
        for (i, b) in blocks.iter().enumerate() {
            assert_eq!(b[0], i as u8, "block {i} start survived");
            assert_eq!(b[(1 << 20) - 1], (i as u8) ^ 0xff, "block {i} end survived");
        }
    }

    #[test_case]
    fn grow_is_not_reentrant() {
        // The guard that stops an allocation inside a growth (a ktrace line, say)
        // from recursing into `alloc` -> `grow` without bound.
        GROWING.store(true, Ordering::Release);
        assert!(!grow(1), "a growth already in progress must decline, not recurse");
        GROWING.store(false, Ordering::Release);
        assert!(grow(1), "and the guard must not latch");
    }
}
