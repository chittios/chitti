//! x86_64-specific primitives. Phase 0 only needs raw port I/O and `hlt`;
//! GDT/IDT/paging land in Phase 1.

pub mod port;

/// Halt the CPU until the next interrupt.
#[inline]
pub fn hlt() {
    // SAFETY: `hlt` has no memory-safety implications; it just stops
    // instruction execution until an interrupt arrives.
    unsafe { core::arch::asm!("hlt", options(nomem, nostack, preserves_flags)) }
}
