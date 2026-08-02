//! **CPU idle** — enter a wait-for-interrupt state when the scheduler has no
//! useful work, so a laptop is not pegged at 100% while the shell waits for a
//! keystroke.
//!
//! ## What this is (and is not)
//!
//! - **Is:** `hlt` (x86) / `wfi` (aarch64) via [`crate::arch::hlt`], with a
//!   simple counter so `/power status` can prove idle is running.
//! - **Is not:** deep C-states, `mwait` hints, or a cpuidle governor. Those are
//!   later; this alone cuts idle power dramatically on real hardware.
//!
//! ## Interrupts must be enabled
//!
//! `hlt`/`wfi` only wake on an interrupt. Callers must not hold
//! `without_interrupts` across this function. The scheduler invokes it *after*
//! the critical section that decided “nowhere to switch.”
//!
//! ## Cooperative platforms must not WFI
//!
//! On aarch64 QEMU `-kernel` there is often **no GIC / no timer IRQ**
//! (`gic: no timer IRQ -- cooperative scheduling`). A `wfi` then never
//! returns, the pump task never polls virtio-input again, and the shell looks
//! frozen after the prompt. [`crate::arch::idle_halt_ok`] is false in that
//! case and we only spin — wasteful but live.

use core::sync::atomic::{AtomicU64, Ordering};

/// Times the BSP entered idle halt (diagnostic / `/power status`).
static HALTS: AtomicU64 = AtomicU64::new(0);
/// Approximate milliseconds spent in idle (timer resolution coarse).
static IDLE_MS: AtomicU64 = AtomicU64::new(0);
/// Times we skipped WFI because no timer IRQ is live.
static SKIPPED: AtomicU64 = AtomicU64::new(0);

/// Enter architected idle until the next interrupt (timer, keyboard, …).
///
/// No-ops (spin) when [`crate::arch::idle_halt_ok`] is false so cooperative
/// boots keep polling.
#[inline]
pub fn halt() {
    if !crate::arch::idle_halt_ok() {
        SKIPPED.fetch_add(1, Ordering::Relaxed);
        // Brief pause so a tight pump loop does not melt a core, without
        // permanently sleeping.
        for _ in 0..64 {
            core::hint::spin_loop();
        }
        return;
    }
    HALTS.fetch_add(1, Ordering::Relaxed);
    let t0 = crate::arch::now_ms();
    // On x86, ensure IF=1 so a timer tick can wake us. If we are already with
    // interrupts on (the usual case after `without_interrupts` returns), `sti`
    // is a no-op for correctness; paired with `hlt` in one block avoids the
    // classic race where an IRQ lands between separate sti and hlt.
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: sti+hlt only changes interrupt timing and power state.
        unsafe {
            core::arch::asm!("sti; hlt", options(nomem, nostack, preserves_flags));
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        crate::arch::hlt();
    }
    let dt = crate::arch::now_ms().saturating_sub(t0);
    IDLE_MS.fetch_add(dt, Ordering::Relaxed);
}

/// How often halt was skipped (cooperative / no timer IRQ).
pub fn skipped_count() -> u64 {
    SKIPPED.load(Ordering::Relaxed)
}

/// How many times [`halt`] has been entered since boot.
pub fn halt_count() -> u64 {
    HALTS.load(Ordering::Relaxed)
}

/// Coarse milliseconds spent inside [`halt`] (sum of timer deltas).
pub fn idle_ms() -> u64 {
    IDLE_MS.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn counters_start_at_zero_or_grow() {
        // In-kernel tests may have halted already during bring-up; just ensure
        // the accessors are callable and monotonic.
        let a = halt_count();
        let b = idle_ms();
        let _ = (a, b);
        assert!(halt_count() >= a);
        assert!(idle_ms() >= b);
    }
}
