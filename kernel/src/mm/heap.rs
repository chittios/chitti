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

/// `(blocks, bytes)` currently parked in the size-class free lists.
pub fn class_stats() -> (usize, usize) {
    ALLOCATOR.with(|a| (a.class_blocks, a.class_bytes))
}

/// `(allocations, free-list steps walked)` since boot.
pub fn alloc_stats() -> (u64, u64) {
    (ALLOC_CALLS.load(Ordering::Relaxed), SCAN_STEPS.load(Ordering::Relaxed))
}

/// One mode's result from [`bench`].
#[derive(Clone, Copy)]
pub struct HeapBench {
    pub mode: HeapMode,
    /// Allocations performed by the workload.
    pub allocs: u64,
    /// Free-list nodes walked across those allocations. **This is the number
    /// the size-class front end exists to reduce**: a class hit walks none.
    pub steps: u64,
    /// Constant-rate counter ticks (TSC / `CNTVCT_EL0`) across the workload —
    /// *not* CPU cycles, exactly as [`crate::synapse::bench`] documents.
    pub ticks: u64,
}

impl HeapBench {
    /// Free-list nodes walked per allocation — the headline figure.
    pub fn steps_per_alloc(&self) -> f32 {
        if self.allocs == 0 { 0.0 } else { self.steps as f32 / self.allocs as f32 }
    }
}

/// Run the same allocation workload under both modes and report each.
///
/// The workload is a **deterministic** churn of small blocks (an LCG picks the
/// sizes, so both modes see byte-for-byte the same request sequence) with a
/// rolling window of live allocations — which is the shape that actually
/// hurts: a page load interleaves short-lived `String`/`Vec` churn with
/// longer-lived nodes, and it is the resulting fragmentation that makes the
/// first-fit scan long. A pure alloc-then-free-all loop would flatter both
/// modes equally and measure nothing.
///
/// Two traps this follows [`crate::synapse::bench`] on: the allocation is
/// passed through `black_box` (the optimizer deletes an allocation whose
/// result is discarded — that is how a "0 ns" figure gets printed), and the
/// mode is restored afterwards so measuring does not change the machine.
pub fn bench(rounds: usize) -> [HeapBench; 2] {
    let saved = heap_mode();
    let mut out = [HeapBench { mode: HeapMode::FirstFit, allocs: 0, steps: 0, ticks: 0 }; 2];

    for (slot, mode) in [HeapMode::FirstFit, HeapMode::SizeClass].into_iter().enumerate() {
        set_heap_mode(mode);
        // Warm up: the first pass pays to grow the heap and to populate the
        // class lists, and charging that to one mode makes the comparison an
        // artifact of ordering. (The cumulative-prefix trap from `synapse::bench`.)
        churn(rounds / 4);

        let (a0, s0) = alloc_stats();
        let t0 = crate::arch::cycle_count();
        churn(rounds);
        let t1 = crate::arch::cycle_count();
        let (a1, s1) = alloc_stats();

        out[slot] = HeapBench {
            mode,
            allocs: a1.wrapping_sub(a0),
            steps: s1.wrapping_sub(s0),
            ticks: t1.wrapping_sub(t0),
        };
    }

    set_heap_mode(saved);
    out
}

/// The workload itself: a rolling window of live small blocks.
fn churn(rounds: usize) {
    use alloc::vec::Vec;

    const WINDOW: usize = 64;
    let mut live: Vec<Vec<u8>> = Vec::with_capacity(WINDOW);
    // Deterministic sizes — same sequence for every mode, so the two figures
    // are comparable. Weighted towards the small end because that is what a
    // parse actually allocates (tag names, attribute strings, node vectors).
    let mut rng: u32 = 0x1234_5678;
    for i in 0..rounds {
        rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let size = 8 + (rng >> 16) as usize % 500;
        let v: Vec<u8> = alloc::vec![0u8; size];
        let v = core::hint::black_box(v);
        if live.len() == WINDOW {
            // Free out of order — freeing in allocation order coalesces
            // perfectly and hides every fragmentation cost there is.
            let victim = (rng >> 8) as usize % WINDOW;
            live.swap_remove(victim);
        }
        live.push(v);
        // Cooperative scheduler: a bench is a long loop like any other.
        if i % 4096 == 0 {
            crate::shell::status_tick();
        }
    }
    core::hint::black_box(&live);
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

/// Block sizes the size-class front end keeps segregated free lists for:
/// **powers of two**, 16 B to 2 KiB.
///
/// Powers of two rather than a denser ladder (…48, 96, 192…) for one specific
/// reason: the class of a size must be computable with *arithmetic*. A denser
/// ladder needs a lookup table, LLVM materialises that table through `xmm`
/// registers, and the resulting `movaps %xmm0, N(%rsp)` needs a 16-byte-aligned
/// stack. The allocation path is reached from the fault handler (killing a task
/// frees its stack), where alignment is not guaranteed — and it raised **#GP**
/// in the middle of the fault-isolation test, halting the machine. `class_of`
/// below is a `leading_zeros` shift: no table, no memory, no vector register.
///
/// The cost is internal fragmentation: a 65-byte request occupies 128. That is
/// the standard size-class trade and it is bounded at 2x.
/// Test-only: the hot path derives both the class and its size arithmetically
/// (`class_of` / `class_size`) precisely so it never reads this table.
/// `class_of_agrees_with_the_class_table` pins the arithmetic against it.
#[cfg(test)]
const CLASS_SIZES: [usize; 8] = [16, 32, 64, 128, 256, 512, 1024, 2048];

/// Largest request the class front end serves. Above this, allocations go to
/// the address-ordered list, which is what large blocks want anyway: they are
/// rare, and they need coalescing more than they need speed.
const MAX_CLASS_SIZE: usize = 2048;

/// The class a request of `size` bytes belongs to, if any.
///
/// `ceil_log2(size) - 4`, computed with `leading_zeros`. Pure arithmetic on
/// purpose — see [`CLASS_SIZES`] for what a table costs here.
#[inline(always)]
fn class_of(size: usize) -> Option<usize> {
    if size > MAX_CLASS_SIZE {
        return None;
    }
    if size <= 16 {
        return Some(0);
    }
    // ceil_log2(size) for size > 1.
    let bits = usize::BITS - (size - 1).leading_zeros();
    Some(bits as usize - 4)
}

/// The exact block size class `ci` holds — `16 << ci`, since the classes are
/// consecutive powers of two.
///
/// Arithmetic rather than `class_size(ci)` for the same reason `class_of` is:
/// **indexing the table makes LLVM copy it to the stack**, and a 64-byte const
/// copy is emitted as `movaps`, which needs a 16-byte-aligned stack. This is
/// reached from the fault handler (killing a task frees its stack), where that
/// alignment does not hold — so an array read here is a #GP that halts the
/// machine. `CLASS_SIZES` survives only as the table the tests pin this against.
#[inline(always)]
fn class_size(ci: usize) -> usize {
    16usize << ci
}

/// Number of size classes. `NUM_CLASSES`, spelled as a constant so the
/// hot path never has to name the array.
const NUM_CLASSES: usize = 8;

/// Which free-list structure serves small allocations. Switchable at runtime so
/// the two can be measured against each other on the same workload — the
/// `/decoder ring3|kernel` pattern, for the allocator.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HeapMode {
    /// The original: one address-ordered list, first-fit scan, full coalescing.
    FirstFit,
    /// Segregated free lists for [`CLASS_SIZES`]: O(1) pop and push, no scan.
    SizeClass,
}

/// Current mode. `0` = first-fit, `1` = size-class.
///
/// **Size-class is the default**, on measurement: it takes the shadcn page's
/// boot from 2654 ms to 695 ms (75 -> 4 free-list steps per allocation, same
/// 4.72M allocations, so identical work), and it is `browser::flex`'s cost by a
/// factor of nine. `/heap firstfit` switches back — first-fit is kept, not
/// deleted, because it is the allocator that coalesces.
///
/// **The known trade, stated because it is the thing that would bite:** a block
/// parked in a class list cannot merge with its neighbours, and the classes are
/// powers of two, so small allocations carry up to 2x internal fragmentation.
/// That is free for a browser and is *not* obviously free for a 27B model load
/// or the ONNX voice path, where the heap is the constraint.
/// `flush_size_classes` exists for exactly that and runs before the allocator
/// gives up — but it re-inserts each parked block into the address-ordered
/// list, so it is O(n²) in the number parked and has not been timed under real
/// pressure. If a large-model or voice workload starts reporting
/// out-of-memory, this is the first thing to switch back.
static HEAP_MODE: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(1);

pub fn heap_mode() -> HeapMode {
    if HEAP_MODE.load(Ordering::Relaxed) == 0 {
        HeapMode::FirstFit
    } else {
        HeapMode::SizeClass
    }
}

/// Switch the small-allocation strategy.
///
/// Safe to call with allocations outstanding **because both modes carve the
/// same block sizes** — `size_align` rounds a small request up to its class in
/// either mode, so a block allocated under one mode is exactly the size the
/// other mode will free it as. Without that rounding, switching would hand a
/// class list a block smaller than the class claims, which is memory
/// corruption rather than a slower allocator. It also makes the A/B honest:
/// only the data structure differs, never the sizes.
pub fn set_heap_mode(mode: HeapMode) {
    HEAP_MODE.store(if mode == HeapMode::FirstFit { 0 } else { 1 }, Ordering::Relaxed);
}

pub struct LinkedListAllocator {
    head: ListNode,
    /// Per-class free lists (see [`CLASS_SIZES`]). A block here is exactly its
    /// class's size and is handed out without any scan. Blocks stay here until
    /// [`flush_classes`] returns them, which is what keeps the OOM path honest:
    /// memory parked in a class list is still free, just not visible to a large
    /// allocation.
    classes: [Option<&'static mut ListNode>; NUM_CLASSES],
    /// Blocks currently parked in `classes`, and their total bytes — so `stats`
    /// and the OOM path can see memory that is free but not on the main list.
    class_blocks: usize,
    class_bytes: usize,
}

impl LinkedListAllocator {
    pub const fn empty() -> Self {
        Self {
            head: ListNode::new(0),
            classes: [const { None }; NUM_CLASSES],
            class_blocks: 0,
            class_bytes: 0,
        }
    }

    /// Take a block from class `ci`, if one is parked. O(1) — this is the whole
    /// point: the first-fit scan reached **3,149 steps per allocation** on a
    /// real page (2.6 MiB of script), and that scan is what made a heavy page
    /// take a minute rather than a second.
    fn pop_class(&mut self, ci: usize) -> Option<usize> {
        let node = self.classes[ci].take()?;
        let addr = node.start_addr();
        self.classes[ci] = node.next.take();
        self.class_blocks -= 1;
        self.class_bytes -= class_size(ci);
        Some(addr)
    }

    /// Park a block in class `ci`. O(1), no coalescing — a class block is only
    /// ever reused at its own size, so there is nothing to merge with.
    ///
    /// # Safety
    /// `addr` must start a block of exactly `class_size(ci)` valid, owned,
    /// unaliased bytes, aligned for `ListNode`.
    unsafe fn push_class(&mut self, ci: usize, addr: usize) {
        // SAFETY: the caller guarantees the region; writing the node header
        // into it is the same intrusive trick `add_free_region` uses.
        unsafe {
            let node = addr as *mut ListNode;
            node.write(ListNode { size: class_size(ci), next: self.classes[ci].take() });
            self.classes[ci] = Some(&mut *node);
        }
        self.class_blocks += 1;
        self.class_bytes += class_size(ci);
    }

    /// Return every parked class block to the address-ordered list, where it
    /// can coalesce and serve a large request. Called from the reclaim path:
    /// without it, a heap that is "full" of small parked blocks would fail a
    /// multi-MiB allocation that the bytes could actually satisfy.
    fn flush_classes(&mut self) -> usize {
        let mut freed = 0;
        for ci in 0..NUM_CLASSES {
            while let Some(addr) = self.pop_class(ci) {
                // SAFETY: the block came from this class list, so it is exactly
                // `class_size(ci)` owned, aligned bytes.
                unsafe { self.add_free_region(addr, class_size(ci)) };
                freed += class_size(ci);
            }
        }
        freed
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
        let size = layout.size().max(mem::size_of::<ListNode>());
        // Round a small request up to its size class in BOTH modes — see
        // `set_heap_mode` for why this is a correctness requirement and not an
        // optimisation.
        let size = match class_of(size) {
            Some(ci) if layout.align() <= mem::align_of::<ListNode>() => class_size(ci),
            _ => size,
        };
        (size, layout.align())
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
        ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
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
        // Still out: the class lists may be holding megabytes in small blocks
        // that only coalescing can turn back into a large one.
        if flush_size_classes() > 0 {
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
        // Then drop reclaimable caches and try once more.
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
        let (size, align) = LinkedListAllocator::size_align(layout);
        crate::sched::uncharge_heap(size);
        self.with(|allocator| {
            // SAFETY: caller (the `alloc::GlobalAlloc` contract) guarantees
            // `ptr..ptr+size` was returned by a prior `alloc` with this
            // same layout and is not aliased elsewhere. `size_align` rounded to
            // the class in whichever mode allocated it, so the block really is
            // `class_size(ci)` bytes.
            if heap_mode() == HeapMode::SizeClass && align <= mem::align_of::<ListNode>() {
                if let Some(ci) = class_of(size) {
                    if class_size(ci) == size {
                        unsafe { allocator.push_class(ci, ptr as usize) };
                        return;
                    }
                }
            }
            unsafe { allocator.add_free_region(ptr as usize, size) };
        });
    }
}

/// Return every size-class block to the main free list, and report the bytes.
///
/// A parked block is free memory the address-ordered list cannot see, so an
/// allocation big enough to need coalescing must flush first or it will fail
/// with the heap nominally half-empty. Registered with `mm::reclaim`, and
/// called directly before `alloc` gives up.
pub fn flush_size_classes() -> usize {
    ALLOCATOR.with(|a| a.flush_classes())
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
    /// Serve one allocation request.
    ///
    /// **`ALLOC_CALLS` is bumped by `alloc`, not by `find_region`**, because a
    /// size-class hit never reaches the scan: counting there made the class
    /// mode report 8 allocations for a 40,000-allocation workload, so its
    /// `steps/alloc` was a ratio over the handful of requests that *missed*.
    /// A counter whose denominator changes meaning with the mode makes the two
    /// modes non-comparable, which is the one thing this measurement is for.
    fn try_alloc(&self, size: usize, align: usize) -> *mut u8 {
        self.with(|allocator| {
            // Fast path: a parked block of exactly this class, no scan at all.
            if heap_mode() == HeapMode::SizeClass && align <= mem::align_of::<ListNode>() {
                if let Some(ci) = class_of(size) {
                    if class_size(ci) == size {
                        if let Some(addr) = allocator.pop_class(ci) {
                            return addr as *mut u8;
                        }
                    }
                }
            }
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
    // Class-parked blocks are free too — reporting only the main list would
    // make the heap look fuller than it is by exactly the amount the fast path
    // is holding.
    let free = ALLOCATOR.with(|a| a.free_bytes() + a.class_bytes);
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

    /// Exercise a mixed workload and hand back the data so the caller can check
    /// it survived — an allocator bug shows up as wrong *bytes*, not a crash.
    fn churn(rounds: usize) -> Vec<Vec<u8>> {
        let mut live: Vec<Vec<u8>> = Vec::new();
        for r in 0..rounds {
            // Sizes straddling every class boundary, plus one above the largest.
            for &n in &[1usize, 15, 16, 17, 63, 64, 100, 255, 256, 700, 2048, 2049, 9000] {
                let mut v = alloc::vec![0u8; n];
                v[0] = (r as u8).wrapping_add(n as u8);
                v[n - 1] = 0xAB;
                live.push(v);
            }
            // Free half, out of order, so blocks are recycled rather than only
            // carved — the class lists are only exercised by reuse.
            if live.len() > 20 {
                for i in (0..live.len()).step_by(2).rev() {
                    live.remove(i);
                }
            }
        }
        live
    }

    #[test_case]
    fn both_heap_modes_return_usable_memory() {
        // The allocator is the one component whose bugs are silent: a wrong
        // block size corrupts a neighbour rather than failing. So both modes run
        // the same churn and the contents are checked afterwards.
        for mode in [HeapMode::FirstFit, HeapMode::SizeClass] {
            set_heap_mode(mode);
            let live = churn(6);
            for v in &live {
                assert_eq!(*v.last().unwrap(), 0xAB, "{mode:?}: tail intact");
            }
            drop(live);
        }
        set_heap_mode(HeapMode::SizeClass);
    }

    #[test_case]
    fn switching_mode_with_live_allocations_is_safe() {
        // Blocks allocated under one mode are freed under the other. This is
        // only sound because `size_align` rounds to the class in BOTH modes; if
        // it did not, a class list would be handed a block smaller than the
        // class claims and the next reuse would scribble past its end.
        set_heap_mode(HeapMode::FirstFit);
        let a = churn(3);
        set_heap_mode(HeapMode::SizeClass);
        let b = churn(3);
        set_heap_mode(HeapMode::FirstFit);
        for v in a.iter().chain(b.iter()) {
            assert_eq!(*v.last().unwrap(), 0xAB, "contents survive a mode switch");
        }
        drop(a);
        drop(b); // freed under FirstFit, some allocated under SizeClass
        set_heap_mode(HeapMode::SizeClass);
    }

    #[test_case]
    fn a_size_class_block_is_reused_without_scanning() {
        // The whole point: a freed small block comes back with no free-list
        // walk. Measured as scan steps, because that is the cost that reached
        // 3,149 per allocation on a real page.
        set_heap_mode(HeapMode::SizeClass);
        // Prime the class so the measured round is pure reuse.
        drop(alloc::vec![0u8; 64]);
        let (calls0, steps0) = alloc_stats();
        for _ in 0..200 {
            drop(core::hint::black_box(alloc::vec![0u8; 64]));
        }
        let (calls1, steps1) = alloc_stats();
        let per_alloc = (steps1 - steps0) as f64 / (calls1 - calls0).max(1) as f64;
        assert!(
            per_alloc < 1.0,
            "size-class reuse should not walk the free list, got {per_alloc} steps/alloc"
        );
    }

    #[test_case]
    fn flushing_classes_returns_the_bytes_to_the_main_list() {
        // Parked blocks are free memory the address-ordered list cannot see. If
        // they were never returned, a large allocation could fail against a heap
        // that is half empty — so the OOM path flushes first.
        set_heap_mode(HeapMode::SizeClass);
        let live = churn(4);
        drop(live);
        let (blocks, bytes) = class_stats();
        assert!(blocks > 0, "churn should leave blocks parked");
        let flushed = flush_size_classes();
        assert_eq!(flushed, bytes, "every parked byte comes back");
        let (blocks_after, bytes_after) = class_stats();
        assert_eq!((blocks_after, bytes_after), (0, 0), "class lists are empty");
    }

    /// Every allocation must be counted in **both** modes.
    ///
    /// The counter used to live in `find_region`, which a size-class hit never
    /// reaches — so the class mode reported 8 allocations for a 40,000-alloc
    /// workload and its `steps/alloc` was a ratio over the misses alone. Two
    /// modes whose denominators mean different things cannot be compared, and
    /// comparing them is the entire purpose of `/heap bench`.
    #[test_case]
    fn every_allocation_is_counted_in_both_modes() {
        for mode in [HeapMode::FirstFit, HeapMode::SizeClass] {
            let saved = heap_mode();
            set_heap_mode(mode);
            let (before, _) = alloc_stats();
            // Sizes that all land inside the class range, so the class mode
            // serves them from its lists after the first pass warms them.
            let mut keep = alloc::vec::Vec::new();
            for i in 0..64 {
                keep.push(alloc::vec![0u8; 16 + i * 8]);
            }
            drop(keep);
            let mut keep = alloc::vec::Vec::new();
            for i in 0..64 {
                keep.push(alloc::vec![0u8; 16 + i * 8]);
            }
            let (after, _) = alloc_stats();
            set_heap_mode(saved);
            assert!(
                after - before >= 128,
                "{mode:?}: 128 allocations counted as {}",
                after - before
            );
            drop(keep);
        }
    }

    #[test_case]
    fn class_of_agrees_with_the_class_table() {
        // `class_of` is a hand-written comparison chain (it must not vectorise —
        // see its doc comment), so nothing but a test keeps it in step with
        // `CLASS_SIZES`. A drift here would hand a class list a block of the
        // wrong size, which corrupts the next allocation rather than failing.
        for size in 1..=(CLASS_SIZES[CLASS_SIZES.len() - 1] + 64) {
            let by_scan = if size > CLASS_SIZES[CLASS_SIZES.len() - 1] {
                None
            } else {
                CLASS_SIZES.iter().position(|&c| c >= size)
            };
            assert_eq!(class_of(size), by_scan, "class_of({size})");
        }
    }

    #[test_case]
    fn class_rounding_is_identical_in_both_modes() {
        // The property that makes mode switching safe, asserted directly rather
        // than only through the churn test.
        for n in [1usize, 8, 16, 17, 31, 32, 65, 200, 2048, 2049, 5000] {
            let l = Layout::from_size_align(n, 8).unwrap();
            set_heap_mode(HeapMode::FirstFit);
            let (a, _) = LinkedListAllocator::size_align(l);
            set_heap_mode(HeapMode::SizeClass);
            let (b, _) = LinkedListAllocator::size_align(l);
            assert_eq!(a, b, "size for {n} bytes must not depend on the mode");
            if let Some(ci) = class_of(a) {
                assert_eq!(a, CLASS_SIZES[ci], "{n} rounds to its class");
            }
        }
        set_heap_mode(HeapMode::SizeClass);
    }

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
