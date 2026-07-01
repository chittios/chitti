//! x86_64-specific primitives: port I/O, GDT/TSS, IDT + exceptions, the
//! legacy PIC + PIT timer + keyboard IRQs, FPU/SSE init, and 4-level
//! paging.

pub mod fpu;
pub mod gdt;
pub mod idt;
pub mod interrupts;
pub mod keyboard;
pub mod paging;
pub mod pic;
pub mod pit;
pub mod port;

/// Halt the CPU until the next interrupt.
#[inline]
pub fn hlt() {
    // SAFETY: `hlt` has no memory-safety implications; it just stops
    // instruction execution until an interrupt arrives.
    unsafe { core::arch::asm!("hlt", options(nomem, nostack, preserves_flags)) }
}
