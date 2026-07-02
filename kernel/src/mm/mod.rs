//! Physical memory management: frame allocator + kernel heap, built from
//! the Limine memory map. `chitti_kernel::init` calls `mm::init` once at
//! boot so callers never touch `frame`/`heap` directly.

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
        crate::arch::x86_64::interrupts::without_interrupts(|| {
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

pub static FRAME_ALLOCATOR: Locked<Option<frame::BitmapFrameAllocator>> = Locked::new(None);

/// Bring up the frame allocator (from the Limine memory map) and the
/// kernel heap on top of it. Must run after `arch::x86_64::fpu::init()`
/// (heap pages are mapped `NO_EXECUTE`, which needs `EFER.NXE` set first).
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
pub fn map_mmio_page(phys: u64) -> u64 {
    use crate::arch::x86_64::paging::{self, NO_EXECUTE, PRESENT, WRITABLE};
    const PWT: u64 = 1 << 3;
    const PCD: u64 = 1 << 4;
    let page = phys & !0xfff;
    let virt = paging::phys_to_virt(page);
    FRAME_ALLOCATOR.with(|slot| {
        let alloc = slot.as_mut().expect("map_mmio_page: frame allocator not initialized");
        paging::map_page(virt, page, PRESENT | WRITABLE | NO_EXECUTE | PCD | PWT, alloc);
    });
    virt + (phys & 0xfff)
}
