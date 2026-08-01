//! CPU-level interrupt enable/disable. Everything here is single-core;
//! critical sections are protected by disabling interrupts for their
//! duration rather than a real spinlock (there is no second core to
//! contend with until Phase 7's SMP stretch goal).

use core::arch::asm;
use core::sync::atomic::{AtomicBool, Ordering};

/// Tracks whether interrupts were enabled before this module last disabled
/// them, purely so `ktrace` and friends can log truthfully; it is not used
/// for any correctness decision.
pub static INTERRUPTS_ENABLED: AtomicBool = AtomicBool::new(false);

#[inline]
pub fn enable() {
    // SAFETY: enabling interrupts is only unsafe in that it changes what
    // code can run when; by the time callers invoke this the IDT/GDT/PIC
    // are fully initialized (see `chitti_kernel::init`).
    unsafe { asm!("sti", options(nomem, nostack)) };
    INTERRUPTS_ENABLED.store(true, Ordering::Relaxed);
}

#[inline]
pub fn disable() {
    // SAFETY: `cli` has no memory-safety implications on its own.
    unsafe { asm!("cli", options(nomem, nostack)) };
    INTERRUPTS_ENABLED.store(false, Ordering::Relaxed);
}

/// Whether interrupts are enabled **right now**, read from the CPU rather than
/// from the logging shadow above.
///
/// This is the kernel's test for "am I inside a critical section", and it works
/// because [`crate::mm::Locked`] is the only thing that takes one: it disables
/// interrupts for the whole of `with`. So code that finds interrupts disabled is
/// either holding a `Locked`, inside an explicit [`without_interrupts`], or in an
/// interrupt handler — and blocking is wrong in all three. `sched::block_on`
/// checks this instead of maintaining a lock-depth counter, which would need
/// per-CPU storage the kernel does not have.
#[inline]
pub fn are_enabled() -> bool {
    flags_interrupts_enabled()
}

#[inline]
fn flags_interrupts_enabled() -> bool {
    let flags: u64;
    // SAFETY: `pushfq`/`pop` only reads CPU state onto the stack.
    unsafe {
        asm!("pushfq", "pop {}", out(reg) flags, options(nomem, preserves_flags));
    }
    flags & (1 << 9) != 0 // RFLAGS.IF
}

/// Run `f` with interrupts disabled, restoring the prior interrupt state
/// (not just unconditionally re-enabling) so nested critical sections
/// compose correctly.
pub fn without_interrupts<T>(f: impl FnOnce() -> T) -> T {
    let was_enabled = flags_interrupts_enabled();
    if was_enabled {
        disable();
    }
    let result = f();
    if was_enabled {
        enable();
    }
    result
}
