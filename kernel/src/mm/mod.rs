//! Physical memory management: frame allocator + kernel heap, built from
//! the Limine memory map. `chitti_kernel::init` calls `mm::init` once at
//! boot so callers never touch `frame`/`heap` directly.

#[cfg(target_arch = "x86_64")]
pub mod frame;
pub mod heap;

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

#[cfg(target_arch = "x86_64")]
pub static FRAME_ALLOCATOR: Locked<Option<frame::BitmapFrameAllocator>> = Locked::new(None);

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
        let model_floor = crate::cortex::MODEL_LOAD_ADDR;
        let uefi_heap = crate::arch::aarch64::mmu::uefi_heap_base() as usize;
        let base = if uefi_heap != 0 {
            // UEFI/stub path: the stub pre-allocated (and reserved from the UEFI
            // allocator) a heap region just above the model and reported its base.
            // Use it directly — the top of RAM is where firmware parks ACPI/
            // runtime data, so we must not place the heap there.
            uefi_heap
        } else {
            // `-kernel` path: no firmware, so place the heap at the very top of
            // discovered RAM (2 MiB-aligned down), leaving the whole span below it
            // for the model.
            let ram_end = crate::arch::aarch64::mmu::ram_end() as usize;
            ram_end.saturating_sub(size) & !((2 << 20) - 1)
        };
        // The model region is [MODEL_LOAD_ADDR, base); if the heap would start at
        // or below where the model loads, there isn't enough RAM for both. Fail
        // loudly with actionable numbers instead of overlapping the model.
        if base <= model_floor || base.checked_add(size).is_none() {
            panic!(
                "Chitti: not enough memory -- a {} MiB heap would start at {:#x}, at/below the model region \
                 base {:#x}. Boot with more RAM (raise -m / the VM's memory).",
                size >> 20,
                base,
                model_floor,
            );
        }
        base
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
}

/// Bring up the frame allocator (from the Limine memory map) and the
/// kernel heap on top of it. Must run after `arch::x86_64::fpu::init()`
/// (heap pages are mapped `NO_EXECUTE`, which needs `EFER.NXE` set first).
#[cfg(target_arch = "x86_64")]
pub fn init(memmap: &[&crate::limine_protocol::MemmapEntry], hhdm_offset: u64) {
    crate::arch::x86_64::paging::set_hhdm_offset(hhdm_offset);

    // SAFETY: `memmap`/`hhdm_offset` are Limine's own responses, read
    // once at boot before anything else touches physical memory.
    let allocator = unsafe { frame::BitmapFrameAllocator::init(memmap, hhdm_offset) };
    let total_frames = allocator.frame_count();
    let free_frames = allocator.free_frame_count();
    FRAME_ALLOCATOR.with(|slot| *slot = Some(allocator));

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
