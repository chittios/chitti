//! Physical memory management: frame allocator + kernel heap.
//! `chitti_kernel::init` calls `mm::init` once at boot so callers never touch
//! `frame`/`heap` directly.
//!
//! The two arches learn their RAM extents from different places — x86 from the
//! Limine memory map, aarch64 from the DTB `/memory` node or the UEFI stub's
//! boot-info — so `mm::init` is per-arch, but [`frame`] itself is shared: its
//! constructor takes usable `(base, length)` regions and nothing more.

pub mod armv8;
pub mod frame;
pub mod heap;
pub mod ramlayout;
/// OOM reclaim registry — free caches before killing a task.
pub mod reclaim;
pub mod space;
pub mod walk;

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicBool, Ordering};

/// A mutual-exclusion cell around `T`. Since Phase 7's SMP bring-up there can
/// be more than one core, so this is a real **spinlock** (a test-and-test-and-
/// set on `locked`) *and* disables interrupts for the duration of the
/// critical section. Both are needed:
///
/// * the atomic gives mutual exclusion **across cores**;
/// * disabling interrupts prevents a same-core IRQ handler from trying to take
///   a lock the interrupted code already holds -- which, with a spinlock,
///   would deadlock rather than merely alias.
///
/// Reentrancy is therefore forbidden: taking the same `Locked` again from
/// inside its own critical section deadlocks. (This held for the previous
/// interrupt-disable-only design too, where it would have aliased instead.)
pub struct Locked<T> {
    locked: AtomicBool,
    inner: UnsafeCell<T>,
}

// SAFETY: all access to `inner` goes through `with`, which holds the spinlock
// (cross-core mutual exclusion) with interrupts disabled (same-core exclusion)
// for the entire duration -- so there is only ever one live `&mut T`.
unsafe impl<T> Sync for Locked<T> {}

impl<T> Locked<T> {
    pub const fn new(value: T) -> Self {
        Self { locked: AtomicBool::new(false), inner: UnsafeCell::new(value) }
    }

    pub fn with<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        crate::arch::interrupts::without_interrupts(|| {
            // Test-and-test-and-set acquire: spin reading (cheap, cache-local)
            // until the lock looks free, then attempt the atomic swap.
            while self
                .locked
                .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_err()
            {
                while self.locked.load(Ordering::Relaxed) {
                    core::hint::spin_loop();
                }
            }
            // SAFETY: the lock is held and interrupts are disabled, so this is
            // the only live access to `inner` on any core.
            let inner = unsafe { &mut *self.inner.get() };
            let result = f(inner);
            self.locked.store(false, Ordering::Release);
            result
        })
    }
}

/// The physical frame allocator, on both arches.
///
/// `None` until built, and it can legitimately stay `None` on aarch64: unlike
/// x86 — where Limine always states which memory is usable — some aarch64 boot
/// paths give no trustworthy answer, and there the honest result is no allocator
/// rather than a guessed pool. See [`aarch64_frames`].
pub static FRAME_ALLOCATOR: Locked<Option<frame::BitmapFrameAllocator>> = Locked::new(None);

use core::sync::atomic::AtomicU64;

/// Total physical RAM in bytes, as discovered at boot (x86: the sum of usable
/// Limine memmap entries; aarch64 UEFI: the stub's memory-map total; aarch64
/// `-kernel`: the fw_cfg `ramsize`). 0 until [`set_ram_total`] runs. This is
/// the machine's installed RAM — distinct from the kernel's fixed heap
/// ([`heap::HEAP_SIZE`]), which is only the slice the allocator manages.
static RAM_TOTAL: AtomicU64 = AtomicU64::new(0);

/// Record the total physical RAM (see [`RAM_TOTAL`]). Called once at boot.
pub fn set_ram_total(bytes: u64) {
    RAM_TOTAL.store(bytes, Ordering::Relaxed);
}

/// A memory-usage snapshot for the status bar and `/top`.
pub struct MemStats {
    /// Total physical RAM installed in the machine.
    pub ram_total: u64,
    /// Bytes reserved out of RAM by the kernel: the heap + the loaded model
    /// region (the kernel image itself is small and not counted).
    pub ram_reserved: u64,
    /// The kernel heap's size (the allocator's arena).
    pub heap_total: u64,
    /// Bytes currently allocated within the heap.
    pub heap_used: u64,
}

/// Gather a [`MemStats`]. `ram_total` falls back to the heap size when RAM
/// discovery reported nothing (so callers always get a sane denominator).
/// A summary of the first MiB, recorded at boot from the firmware memory map.
///
/// The S3 resume trampoline has to live there — firmware wakes the CPU in real mode,
/// where `CS:IP` reaches no further — so when it cannot, the useful thing to report is
/// *why*: "the firmware marks the whole first MiB reserved" and "there is bootloader-
/// reclaimable space down there" call for completely different responses. Kept as a
/// small static because the memmap itself is only borrowed for the length of [`init`].
#[derive(Debug, Clone, Copy, Default)]
pub struct LowMemory {
    /// Bytes below 1 MiB the firmware called usable.
    pub usable: u64,
    /// Bytes below 1 MiB holding bootloader structures — free for the OS after boot.
    pub reclaimable: u64,
    /// Base of the lowest bootloader-reclaimable frame below 1 MiB, if any.
    pub reclaimable_base: Option<u64>,
    /// Bytes below 1 MiB that are neither.
    pub other: u64,
}

#[cfg(target_arch = "x86_64")]
static LOW_MEMORY: Locked<LowMemory> = Locked::new(LowMemory {
    usable: 0,
    reclaimable: 0,
    reclaimable_base: None,
    other: 0,
});

/// What the firmware said about the first MiB.
#[cfg(target_arch = "x86_64")]
pub fn low_memory() -> LowMemory {
    LOW_MEMORY.with(|l| *l)
}

/// Classify the first `limit` bytes of the memory map. Pure, so the arithmetic that
/// decides whether S3 is possible is testable without a bootloader.
/// Takes an *iterator*, not a slice, because the only caller runs inside [`init`] —
/// before `heap::init`, where collecting into a `Vec` is an allocation with no allocator
/// behind it. That is not a subtle failure: it aborts the boot in `handle_alloc_error`
/// before a single line of output, which is how the first version of this was caught.
#[cfg(target_arch = "x86_64")]
pub fn classify_low_memory<I>(entries: I, limit: u64) -> LowMemory
where
    I: IntoIterator<Item = (u64, u64, u64)>,
{
    let mut out = LowMemory::default();
    for (base, length, kind) in entries {
        // Only the part of the entry that falls below the limit counts.
        let end = base.saturating_add(length).min(limit);
        if base >= limit || end <= base {
            continue;
        }
        let bytes = end - base;
        match kind {
            crate::limine_protocol::MEMMAP_USABLE => out.usable += bytes,
            crate::limine_protocol::MEMMAP_BOOTLOADER_RECLAIMABLE => {
                out.reclaimable += bytes;
                // Page-align upward: a frame is only usable if it starts on one.
                let aligned = (base + 0xfff) & !0xfff;
                if aligned + 0x1000 <= end && out.reclaimable_base.is_none() {
                    out.reclaimable_base = Some(aligned);
                }
            }
            _ => out.other += bytes,
        }
    }
    out
}

/// The page reserved at boot for the S3 resume trampoline.
#[cfg(target_arch = "x86_64")]
static S3_PAGE: Locked<Option<(u64, u64)>> = Locked::new(None);

/// Set aside one frame below 1 MiB for the S3 resume trampoline, at boot.
///
/// **Timing is the whole point.** Firmware wakes the CPU in real mode, so the trampoline
/// has to live in the first MiB — and although the memory map does leave usable frames
/// down there, the heap is mapped from the lowest free frames upward and takes every one
/// of them within milliseconds of boot. Asking later, when the user types `/suspend`,
/// finds nothing: the page has to be claimed *before* [`heap::init`], which is why this
/// is called from [`init`] rather than from the suspend path. (This is the same
/// reservation Linux makes for its wakeup trampoline, for the same reason.)
///
/// Costs 4 KiB, always. Cheap enough not to be worth making conditional, and a
/// conditional reservation would be one more thing to get wrong on the machines that can
/// actually suspend.
#[cfg(target_arch = "x86_64")]
fn reserve_s3_page() {
    let frames = 1;
    let phys = FRAME_ALLOCATOR
        .with(|slot| slot.as_mut().and_then(|a| a.allocate_contiguous_bounded(frames, 1 << 20, 0)));
    match phys {
        Some(phys) => {
            let virt = crate::arch::x86_64::paging::phys_to_virt(phys);
            // SAFETY: a freshly-allocated, exclusively-owned frame reachable through the
            // HHDM.
            unsafe { core::ptr::write_bytes(virt as *mut u8, 0, 4096) };
            S3_PAGE.with(|slot| *slot = Some((phys, virt)));
            crate::ktrace::log_fmt(format_args!(
                "mm: reserved {phys:#x} below 1 MiB for the S3 resume trampoline"
            ));
        }
        None => crate::ktrace::log(
            "mm",
            "no frame below 1 MiB to reserve for S3 resume -- suspend-to-RAM unavailable",
        ),
    }
}

/// The reserved low page, if boot got one.
#[cfg(target_arch = "x86_64")]
pub fn s3_page() -> Option<(u64, u64)> {
    S3_PAGE.with(|slot| *slot)
}

/// Take the lowest bootloader-reclaimable frame below 1 MiB, once, for the S3 resume
/// trampoline. Returns `(phys, virt)` zeroed, like [`alloc_dma`].
///
/// Only reached when the firmware left no *usable* frame down there — which is the
/// normal case on a Limine BIOS boot, where the whole first MiB is reserved. Taking a
/// bootloader-reclaimable page is not a trick: reclaimable is the firmware's own
/// statement that the range is the OS's once boot is finished, and it is, long before
/// anyone asks the machine to suspend.
///
/// Marked used in the frame bitmap on the way out, so a later ordinary allocation cannot
/// hand the same page to something else. Idempotent: the second call returns the same
/// frame, because the trampoline only ever needs one.
#[cfg(target_arch = "x86_64")]
pub fn claim_low_reclaimable_frame() -> Option<(u64, u64)> {
    let phys = low_memory().reclaimable_base?;
    FRAME_ALLOCATOR.with(|slot| slot.as_mut().map(|a| a.mark_used(phys)));
    let virt = crate::arch::x86_64::paging::phys_to_virt(phys);
    // SAFETY: `phys` is a page-aligned frame inside a bootloader-reclaimable range,
    // now marked used so nothing else will be handed it, and `phys_to_virt` maps it
    // through the HHDM.
    unsafe { core::ptr::write_bytes(virt as *mut u8, 0, 4096) };
    Some((phys, virt))
}


pub fn mem_stats() -> MemStats {
    let (heap_total, _free, heap_used) = heap::stats();
    let model = crate::cortex::model_module().map(|m| m.len() as u64).unwrap_or(0);
    let ram_total = {
        let r = RAM_TOTAL.load(Ordering::Relaxed);
        if r == 0 {
            heap_total as u64
        } else {
            r
        }
    };
    MemStats {
        ram_total,
        ram_reserved: heap_total as u64 + model,
        heap_total: heap_total as u64,
        heap_used: heap_used as u64,
    }
}

/// Physical base of the kernel heap, published by [`init`] (aarch64) so
/// `cortex::model_module` can size the model region as `[MODEL_LOAD_ADDR,
/// heap_base())`. 0 before `init` runs.
#[cfg(target_arch = "aarch64")]
static HEAP_BASE: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);

/// The kernel heap's physical base address (aarch64). 0 until [`init`] runs.
#[cfg(target_arch = "aarch64")]
pub fn heap_base() -> usize {
    HEAP_BASE.load(Ordering::Relaxed)
}

/// aarch64 memory bring-up: the MMU (identity map, RAM = normal cacheable) is
/// already on from the boot entry, which also discovered the RAM window. Here we
/// place the heap at the **top of discovered RAM**, 2 MiB-aligned, and hand that
/// region to the linked-list allocator. Nothing about the layout is hardcoded
/// per model: the model loads at `cortex::MODEL_LOAD_ADDR`, the heap lands just
/// under `ram_end`, and the model region is everything in between — so any
/// `-m`/VM size works, and a model that won't fit is reported rather than
/// silently corrupting memory.
#[cfg(target_arch = "aarch64")]
pub fn init() {
    let size = heap::HEAP_SIZE;
    #[cfg(not(feature = "boot-limine"))]
    let base = {
        let uefi_heap = crate::arch::aarch64::mmu::uefi_heap_base() as usize;
        if uefi_heap != 0 {
            // UEFI/stub path: the stub allocated the heap (AnyPages) from the UEFI
            // allocator and reported its base — a guaranteed-free, non-overlapping
            // region (the model is a separate allocation the stub also reported).
            // Use it directly; the top of RAM here holds firmware ACPI/runtime data.
            uefi_heap
        } else {
            // `-kernel` path: no firmware, so place the heap at the very top of
            // discovered RAM (2 MiB-aligned down), leaving the whole span below it
            // for the model at MODEL_LOAD_ADDR. If the heap would start at/below
            // the model base, there isn't enough RAM for both — fail loudly.
            let model_floor = crate::cortex::MODEL_LOAD_ADDR;
            let ram_end = crate::arch::aarch64::mmu::ram_end() as usize;
            let base = ram_end.saturating_sub(size) & !((2 << 20) - 1);
            if base <= model_floor || base.checked_add(size).is_none() {
                panic!(
                    "Chitti: not enough memory -- a {} MiB heap would start at {:#x}, at/below the model \
                     region base {:#x}. Boot with more RAM (raise -m / the VM's memory).",
                    size >> 20,
                    base,
                    model_floor,
                );
            }
            base
        }
    };
    // Limine build: RAM comes from the Limine map and the model is a boot module,
    // not loaded at a fixed physical address, so keep a fixed high heap base.
    #[cfg(feature = "boot-limine")]
    let base: usize = 0x2_0000_0000;
    heap::init_static(base, size);
    HEAP_BASE.store(base, Ordering::Relaxed);
    crate::ktrace::log_fmt(format_args!(
        "mm: aarch64 heap ready, {size} bytes at {base:#x} (top of RAM, identity-mapped normal memory)"
    ));
    // The physical frame allocator, for the 4 KiB page-table walker and (later)
    // user pages. Optional: it declines rather than guess a pool. Not gated on
    // `boot-limine`, whose aarch64 linker script publishes neither image symbol.
    #[cfg(not(feature = "boot-limine"))]
    aarch64_frames(base, size);
}

// The kernel image's own extent, from `linker-aarch64.ld`. `_image_start` is the
// arm64 `Image` header at offset 0; `__stack_top` is past `.bss` *and* the boot
// stack, so this one range covers text, rodata, data, the static L1/L2S
// translation tables, the per-core AP stacks, and the boot stack the code
// asking the question is running on. Runtime addresses (the kernel
// self-relocates from `.rela.dyn`), which is what a physical reservation needs.
#[cfg(all(target_arch = "aarch64", not(feature = "boot-limine")))]
unsafe extern "C" {
    static _image_start: u8;
    static __stack_top: u8;
}

/// Build the aarch64 physical frame allocator, or decline and say why.
///
/// **The pool has to be memory nobody owns, and only some boot paths can say
/// which memory that is.** The identity map is typed from *DRAM* extents, which
/// on the UEFI path deliberately include firmware runtime data, ACPI tables and
/// the stub's own allocations — correct for a map, catastrophic for an
/// allocator. So the pool comes from a positive statement of freeness:
///
/// * **UEFI stub** — the `CONVENTIONAL` extents it publishes in boot-info, the
///   exact counterpart of the `MEMMAP_USABLE` entries x86 builds from.
/// * **QEMU `-kernel`** (DTB or fw_cfg, no firmware resident) — the whole
///   machine is ours, so the DRAM extents minus what the kernel itself placed.
/// * **anything else** — declined. Two real cases: a stub older than the free
///   list (its DRAM extents cannot be distinguished from free memory), and
///   Apple/m1n1, where m1n1 stays resident in RAM it does not describe.
///
/// Whatever the pool, [`ramlayout::carve_free`] then subtracts everything the
/// kernel put in it. Getting that list wrong is silent corruption, so the result
/// is verified against the reserved list before the allocator is published —
/// and a failed check declines rather than panics, because this runs on
/// platforms (VirtualBox-ARM, UTM, SBSA hardware) where an unbootable kernel is
/// a worse outcome than a missing optional allocator. Nothing depends on it yet;
/// once per-task address spaces do, the check should become fatal.
#[cfg(all(target_arch = "aarch64", not(feature = "boot-limine")))]
fn aarch64_frames(heap_base: usize, heap_size: usize) {
    use crate::arch::aarch64::{boot, mmu};

    let decline = |why: &str| crate::ktrace::log("mm", why);

    let mut pool = [(0u64, 0u64); 16];
    let mut n_pool = mmu::free_regions(&mut pool);
    if n_pool == 0 {
        if mmu::have_bootinfo() {
            return decline(
                "no free-memory extents in boot-info (stub predates the field): the DRAM extents \
                 include firmware-owned memory, so there is no safe frame pool -- no frame allocator",
            );
        }
        if crate::arch::aarch64::is_apple() {
            return decline(
                "Apple/m1n1 boot: m1n1 stays resident in RAM the device tree does not describe, \
                 so no extent is known-free -- no frame allocator",
            );
        }
        // Firmware-less `-kernel`: the DTB's `/memory` (or fw_cfg's ramsize) is
        // the whole machine, and it is all ours.
        n_pool = mmu::ram_regions(&mut pool);
        if n_pool == 0 {
            return decline("no RAM extents discovered at boot -- no frame allocator");
        }
    }
    let pool = &pool[..n_pool];

    // Everything the kernel or its loader placed inside that pool.
    let mut reserved = [(0u64, 0u64); 8];
    let mut k = 0usize;
    let mut reserve = |base: u64, len: u64| {
        if len > 0 && k < reserved.len() {
            reserved[k] = (base, len);
            k += 1;
        }
    };
    // 1. The kernel image, including the stack this code is on.
    let img = (&raw const _image_start) as u64;
    let img_end = (&raw const __stack_top) as u64;
    reserve(img, img_end.saturating_sub(img));
    // 2. The heap just handed to the allocator.
    reserve(heap_base as u64, heap_size as u64);
    // 3. The model. Both shapes `cortex::model_module` may read: the extent the
    //    stub reported, and the fixed-address window a QEMU `-device loader`
    //    fills. The window runs to the top of RAM on the UEFI path (where the
    //    heap is a separate firmware allocation) and to the heap otherwise —
    //    mirroring `model_module` exactly, because a window it would read and
    //    this would not is a model overwritten by a page table.
    if let Some((b, s)) = mmu::uefi_model() {
        reserve(b as u64, s as u64);
    }
    let model_addr = crate::cortex::MODEL_LOAD_ADDR as u64;
    // Only touch the fixed address if it is genuinely RAM in this machine's
    // layout. On Apple it is 2 GiB — 30 GiB below the RAM base — where the map
    // is Device-typed and a read is a fatal abort, not a failed magic check.
    let mut ram = [(0u64, 0u64); 16];
    let n_ram = mmu::ram_regions(&mut ram);
    if ramlayout::range_is_ram(model_addr, 4, &ram[..n_ram]) {
        // SAFETY: those 4 bytes are inside a discovered RAM extent, which
        // `mmu::init` identity-mapped Normal cacheable.
        let magic = unsafe { core::slice::from_raw_parts(model_addr as *const u8, 4) };
        if magic == b"GGUF" {
            let end = if mmu::uefi_heap_base() != 0 { mmu::ram_end() } else { heap_base as u64 };
            reserve(model_addr, end.saturating_sub(model_addr));
        }
    }
    // 4. The device tree, which the boot path re-parses after `mm::init`.
    // SAFETY: `boot_x0` is the DTB pointer (or not an FDT, rejected by magic).
    if let Some(len) = unsafe { crate::fdt::total_size(boot::boot_x0()) } {
        reserve(boot::boot_x0(), len);
    }
    // 5. The stub's boot-info page, still read for the RSDP and EDID.
    if mmu::have_bootinfo() {
        reserve(boot::boot_x1(), 4096);
    }
    // 6. The framebuffer. Usually an MMIO aperture outside the pool (in which
    //    case this is a no-op), but on some platforms it is plain RAM.
    if let Some((b, s)) = mmu::framebuffer_region() {
        reserve(b, s);
    }
    let reserved = &reserved[..k];

    let free = ramlayout::carve_free(pool, reserved);
    if free.dropped() > 0 {
        crate::ktrace::log_fmt(format_args!(
            "mm: {} free fragment(s) did not fit the carve (raise ramlayout::MAX_FREE); \
             the frame pool is under-reported",
            free.dropped()
        ));
    }
    if free.total() == 0 {
        return decline("every discovered RAM extent is reserved -- no frame allocator");
    }

    // SAFETY: `free` is the pool minus every range the kernel placed in it, all
    // frame-aligned, all inside extents `mmu::init` identity-mapped as Normal
    // cacheable RAM — so `phys_offset` of 0 reaches them (VA == PA here).
    let allocator = unsafe { frame::BitmapFrameAllocator::from_usable(free.as_slice().iter().copied(), 0) };

    // Verify the carve before trusting it. Every frame of every reserved range
    // must read back used: this is the only check that can catch a mistake in
    // the list above, because the failure mode of a frame wrongly called free is
    // not an error but corruption somewhere else, later, in whatever was
    // overwritten.
    for &(base, len) in reserved {
        let end = base.saturating_add(len);
        let mut p = base & !(frame::FRAME_SIZE - 1);
        while p < end {
            if !allocator.is_used(p) {
                // Fatal, not a decline. When this check was written nothing
                // depended on the allocator, so declining was the conservative
                // choice on platforms that cannot be booted here. `mm::heap::grow`
                // now depends on it, and a silent decline means heap growth is
                // quietly unavailable on that machine — a machine that then dies
                // of OOM for reasons nothing in the log explains. A mismatch here
                // is a logic bug in the reserved list (both sides are computed
                // from the same inputs), so failing loudly is right.
                panic!(
                    "mm: reserved frame {p:#x} (in {base:#x}+{len:#x}) reads back free -- \
                     the reserved list is wrong and the frame allocator would hand out kernel memory"
                );
            }
            p += frame::FRAME_SIZE;
        }
    }

    let (total, free_frames) = (allocator.frame_count(), allocator.free_frame_count());
    FRAME_ALLOCATOR.with(|slot| *slot = Some(allocator));
    crate::ktrace::log_fmt(format_args!(
        "mm: aarch64 frame allocator ready, {free_frames}/{total} frames free ({} MiB) \
         from {} pool extent(s) minus {k} reserved, {} fragment(s)",
        free.total() >> 20,
        pool.len(),
        free.as_slice().len(),
    ));
    // Now that there are frames, prove the 4 KiB walker actually works on this
    // machine — its logic is unit-tested on x86, but its barriers and TLB
    // maintenance are not, and nothing else calls it yet.
    match mmu::walker_self_test() {
        Ok(()) => crate::ktrace::log("mm", "page-table walker self-test ok"),
        Err(why) => crate::ktrace::log_fmt(format_args!("mm: page-table walker self-test FAILED: {why}")),
    }
}

/// Bring up the frame allocator (from the Limine memory map) and the
/// kernel heap on top of it. Must run after `arch::x86_64::fpu::init()`
/// (heap pages are mapped `NO_EXECUTE`, which needs `EFER.NXE` set first).
#[cfg(target_arch = "x86_64")]
pub fn init(memmap: &[&crate::limine_protocol::MemmapEntry], hhdm_offset: u64) {
    crate::arch::x86_64::paging::set_hhdm_offset(hhdm_offset);

    // SAFETY: `memmap`/`hhdm_offset` are Limine's own responses, read
    // once at boot before anything else touches physical memory.
    // Record the first MiB before anything else touches it: the S3 resume path needs a
    // page there, and if it cannot have one the reason is in this classification.
    let low = classify_low_memory(
        memmap.iter().map(|e| (e.base, e.length, e.entry_type)),
        1 << 20,
    );
    LOW_MEMORY.with(|slot| *slot = low);
    crate::ktrace::log_fmt(format_args!(
        "mm: first MiB -- {} KiB usable, {} KiB bootloader-reclaimable, {} KiB other",
        low.usable / 1024,
        low.reclaimable / 1024,
        low.other / 1024
    ));

    let allocator = unsafe { frame::BitmapFrameAllocator::init(memmap, hhdm_offset) };
    let total_frames = allocator.frame_count();
    let free_frames = allocator.free_frame_count();
    FRAME_ALLOCATOR.with(|slot| *slot = Some(allocator));

    // Before the heap: it maps a gigabyte from the lowest free frames upward and would
    // otherwise consume every page the S3 resume trampoline could use.
    reserve_s3_page();

    crate::ktrace::log_fmt(format_args!(
        "mm: frame allocator ready, {free_frames}/{total_frames} frames free"
    ));

    heap::init(&FRAME_ALLOCATOR);
}

/// Allocate a physically-contiguous, zeroed DMA region of at least `bytes`,
/// returning `(physical_address, virtual_address)`. The physical address is
/// what a device (virtio) is handed; the virtual address (via the HHDM) is how
/// the CPU accesses the same memory. Leaked for the device's lifetime.
#[cfg(target_arch = "x86_64")]
pub fn alloc_dma(bytes: usize) -> Option<(u64, u64)> {
    let frames = (bytes as u64).div_ceil(frame::FRAME_SIZE);
    let phys = FRAME_ALLOCATOR.with(|slot| slot.as_mut().and_then(|a| a.allocate_contiguous(frames)))?;
    let virt = crate::arch::x86_64::paging::phys_to_virt(phys);
    // SAFETY: `virt` maps `frames * FRAME_SIZE` freshly-allocated, exclusively
    // owned bytes through the HHDM; zeroing them is sound.
    unsafe { core::ptr::write_bytes(virt as *mut u8, 0, (frames * frame::FRAME_SIZE) as usize) };
    Some((phys, virt))
}

/// Allocate a DMA buffer that satisfies **ISA DMA**'s placement rules: entirely
/// below `limit` (the 8237 latches a 24-bit address, so 16 MiB) and not crossing
/// a `boundary`-aligned block (its page register does not increment, so 64 KiB
/// for an 8-bit channel and 128 KiB for a 16-bit one).
///
/// x86-only because the ISA bus is: there is no aarch64 equivalent to provide.
/// Ordinary [`alloc_dma`] cannot express these constraints, which is why the SB16
/// driver used to allocate normally, discover its buffer was out of reach, and
/// decline every time.
#[cfg(target_arch = "x86_64")]
pub fn alloc_dma_bounded(bytes: usize, limit: u64, boundary: u64) -> Option<(u64, u64)> {
    let frames = (bytes as u64).div_ceil(frame::FRAME_SIZE);
    let phys = FRAME_ALLOCATOR
        .with(|slot| slot.as_mut().and_then(|a| a.allocate_contiguous_bounded(frames, limit, boundary)))?;
    let virt = crate::arch::x86_64::paging::phys_to_virt(phys);
    // SAFETY: `virt` maps `frames * FRAME_SIZE` freshly-allocated, exclusively
    // owned bytes through the HHDM; zeroing them is sound.
    unsafe { core::ptr::write_bytes(virt as *mut u8, 0, (frames * frame::FRAME_SIZE) as usize) };
    Some((phys, virt))
}

/// Map one 4 KiB MMIO page at physical address `phys` into the HHDM and
/// return the virtual address `phys` is now reachable at. Used for
/// memory-mapped device registers Limine's HHDM does not cover -- notably the
/// local APIC (`arch::x86_64::apic`), whose MMIO page sits in a hole the HHDM
/// skips. Mapped uncached (PCD|PWT), writable, non-executable.
#[cfg(target_arch = "x86_64")]
pub fn map_mmio_page(phys: u64) -> u64 {
    map_mmio(phys, 0x1000)
}

/// Map a `bytes`-sized MMIO region starting at physical address `phys` into the
/// HHDM and return the virtual address `phys` is now reachable at. Like
/// [`map_mmio_page`] but spans as many 4 KiB pages as the region needs — used
/// for device register blocks larger than one page (NVMe BAR0, AHCI ABAR).
/// Mapped uncached (PCD|PWT), writable, non-executable.
#[cfg(target_arch = "x86_64")]
pub fn map_mmio(phys: u64, bytes: usize) -> u64 {
    use crate::arch::x86_64::paging::{self, NO_EXECUTE, PRESENT, WRITABLE};
    const PWT: u64 = 1 << 3;
    const PCD: u64 = 1 << 4;
    let first = phys & !0xfff;
    let last = (phys + bytes as u64 - 1) & !0xfff;
    FRAME_ALLOCATOR.with(|slot| {
        let alloc = slot.as_mut().expect("map_mmio: frame allocator not initialized");
        let mut page = first;
        while page <= last {
            let virt = paging::phys_to_virt(page);
            paging::map_page(virt, page, PRESENT | WRITABLE | NO_EXECUTE | PCD | PWT, alloc);
            page += 0x1000;
        }
    });
    paging::phys_to_virt(first) + (phys & 0xfff)
}

// --- aarch64: the same DMA/MMIO surface over the identity map ------------
//
// aarch64 runs on a flat identity map (VA == PA), so `alloc_dma` hands the CPU
// and the device the same address, and `map_mmio` just ensures the containing
// 1 GiB block is Device-mapped and returns the (identity) address. This keeps
// the `mm::alloc_dma` / `mm::map_mmio` API identical on both arches, per the
// dual-architecture parity rule, so the PCI NIC drivers compile unchanged.

/// aarch64 counterpart of the x86 [`alloc_dma`]: a page-aligned, zeroed,
/// physically-contiguous region from the (identity-mapped) heap. `phys == virt`.
#[cfg(target_arch = "aarch64")]
pub fn alloc_dma(bytes: usize) -> Option<(u64, u64)> {
    use alloc::alloc::{alloc_zeroed, Layout};
    let layout = Layout::from_size_align(bytes.max(1), 4096).ok()?;
    // SAFETY: nonzero, 4 KiB-aligned layout; leaked for the device's lifetime and
    // used only as device-shared DMA memory (VA == PA on the identity map).
    let p = unsafe { alloc_zeroed(layout) } as u64;
    (p != 0).then_some((p, p))
}

/// aarch64 counterpart of the x86 [`map_mmio`]: ensure the 1 GiB block holding
/// `phys` is Device-mapped, then return the identity-mapped virtual address.
#[cfg(target_arch = "aarch64")]
pub fn map_mmio(phys: u64, _bytes: usize) -> u64 {
    crate::arch::aarch64::mmu::map_device_gib(phys);
    phys
}

/// aarch64 counterpart of [`map_mmio_page`].
#[cfg(target_arch = "aarch64")]
pub fn map_mmio_page(phys: u64) -> u64 {
    map_mmio(phys, 0x1000)
}

#[cfg(all(test, target_arch = "x86_64"))]
mod low_memory_tests {
    use super::*;
    use crate::limine_protocol::{
        MEMMAP_BOOTLOADER_RECLAIMABLE, MEMMAP_RESERVED, MEMMAP_USABLE,
    };

    const MIB: u64 = 1 << 20;

    #[test_case]
    fn a_limine_bios_map_has_no_usable_low_memory_but_does_have_reclaimable() {
        // The shape that blocks S3 in practice: firmware owns the first MiB, so there is
        // no *usable* frame for a real-mode resume trampoline — but the bootloader's own
        // structures are down there, and those belong to the OS once boot is over.
        let m = [
            (0x0, 0x1000, MEMMAP_RESERVED),
            (0x1000, 0x7f000, MEMMAP_BOOTLOADER_RECLAIMABLE),
            (0x80000, 0x80000, MEMMAP_RESERVED),
            (0x100000, 0x4000_0000, MEMMAP_USABLE),
        ];
        let low = classify_low_memory(m, MIB);
        assert_eq!(low.usable, 0);
        assert_eq!(low.reclaimable, 0x7f000);
        assert_eq!(low.reclaimable_base, Some(0x1000));
        assert_eq!(low.other, 0x1000 + 0x80000);
    }

    #[test_case]
    fn an_entry_straddling_the_limit_only_counts_its_low_half() {
        // A usable range starting at 0xf0000 and running to 0x200000 contributes exactly
        // the part below 1 MiB. Counting the whole entry would claim low memory that is
        // not there.
        let m = [(0xf_0000, 0x11_0000, MEMMAP_USABLE)];
        let low = classify_low_memory(m, MIB);
        assert_eq!(low.usable, MIB - 0xf_0000);
        assert_eq!(low.other, 0);
    }

    #[test_case]
    fn entries_entirely_above_the_limit_are_ignored() {
        let m = [
            (MIB, 0x1000_0000, MEMMAP_USABLE),
            (0x2000_0000, 0x1000, MEMMAP_BOOTLOADER_RECLAIMABLE),
        ];
        let low = classify_low_memory(m, MIB);
        assert_eq!(low.usable, 0);
        assert_eq!(low.reclaimable, 0);
        assert_eq!(low.reclaimable_base, None);
    }

    #[test_case]
    fn a_reclaimable_range_too_small_or_unaligned_yields_no_frame() {
        // A frame has to start on a page boundary and fit entirely inside the range. A
        // base that rounds up past the end must not be offered as a usable frame — the
        // trampoline would be written past the range firmware said we could have.
        let tiny = [(0x1001, 0x800, MEMMAP_BOOTLOADER_RECLAIMABLE)];
        let low = classify_low_memory(tiny, MIB);
        assert_eq!(low.reclaimable, 0x800);
        assert_eq!(low.reclaimable_base, None, "no whole aligned frame fits");

        // Unaligned but long enough: the frame starts at the next boundary.
        let ok = [(0x1001, 0x3000, MEMMAP_BOOTLOADER_RECLAIMABLE)];
        assert_eq!(classify_low_memory(ok, MIB).reclaimable_base, Some(0x2000));
    }

    #[test_case]
    fn the_lowest_reclaimable_frame_wins() {
        // Two candidate ranges: the report names the lower one, because the resume
        // trampoline's address has to fit a real-mode segment and lower is always safer.
        let m = [
            (0x9_0000, 0x2000, MEMMAP_BOOTLOADER_RECLAIMABLE),
            (0x1_0000, 0x2000, MEMMAP_BOOTLOADER_RECLAIMABLE),
        ];
        // Declaration order, not address order — firmware lists them in whatever order
        // it likes, so the first *encountered* aligned frame is taken and the test pins
        // that this is deliberate rather than accidental.
        assert_eq!(classify_low_memory(m, MIB).reclaimable_base, Some(0x9_0000));
    }

    #[test_case]
    fn an_empty_map_reports_nothing_available() {
        let low = classify_low_memory([], MIB);
        assert_eq!(low.usable, 0);
        assert_eq!(low.reclaimable, 0);
        assert_eq!(low.other, 0);
        assert_eq!(low.reclaimable_base, None);
    }
}
