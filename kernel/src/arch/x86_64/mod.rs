//! x86_64-specific primitives: port I/O, GDT/TSS, IDT + exceptions, the
//! legacy PIC + PIT timer + keyboard IRQs, FPU/SSE init, and 4-level
//! paging.

pub mod ahci;
pub mod apic;
pub mod disk;
pub mod fpu;
pub mod gdt;
pub mod idt;
pub mod interrupts;
pub mod keyboard;
pub mod nvme;
pub mod paging;
pub mod pci;
pub mod pic;
pub mod pit;
pub mod port;
pub mod rtc;
pub mod xhci;

/// Halt the CPU until the next interrupt.
#[inline]
pub fn hlt() {
    // SAFETY: `hlt` has no memory-safety implications; it just stops
    // instruction execution until an interrupt arrives.
    unsafe { core::arch::asm!("hlt", options(nomem, nostack, preserves_flags)) }
}

/// Power off / exit QEMU cleanly (so typing `exit` at the shell terminates the
/// emulator instead of leaving it idling). Uses the `isa-debug-exit` device
/// (present in every `xtask` QEMU invocation): a dword write to port 0xf4
/// exits QEMU. Falls back to a halt loop if that somehow returns.
pub fn poweroff() -> ! {
    // SAFETY: 0xf4 is the isa-debug-exit device port; a write exits QEMU.
    unsafe { port::outl(0xf4, 0x10) };
    loop {
        hlt();
    }
}
