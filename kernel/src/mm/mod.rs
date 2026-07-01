//! Physical memory management: frame allocator + kernel heap, built from
//! the Limine memory map. `chitti_kernel::init` calls `mm::init` once at
//! boot so callers never touch `frame`/`heap` directly.

pub mod frame;
pub mod heap;

use core::cell::UnsafeCell;

/// Single-core critical-section wrapper: disables interrupts around `f`
/// instead of spinning on a lock, since there is no second core to
/// contend with until Phase 7's SMP stretch goal.
pub struct Locked<T> {
    inner: UnsafeCell<T>,
}

// SAFETY: every access goes through `with`, which disables interrupts for
// its duration; there is only one core, so that fully serializes access.
unsafe impl<T> Sync for Locked<T> {}

impl<T> Locked<T> {
    pub const fn new(value: T) -> Self {
        Self { inner: UnsafeCell::new(value) }
    }

    pub fn with<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        crate::arch::x86_64::interrupts::without_interrupts(|| {
            // SAFETY: interrupts are disabled for the duration of `f`
            // (see above), so this is the only live access to `inner`.
            let inner = unsafe { &mut *self.inner.get() };
            f(inner)
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
