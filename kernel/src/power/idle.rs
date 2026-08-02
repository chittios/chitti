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

use core::sync::atomic::{AtomicU64, Ordering};

/// Times the BSP entered idle halt (diagnostic / `/power status`).
static HALTS: AtomicU64 = AtomicU64::new(0);
/// Approximate milliseconds spent in idle (timer resolution coarse).
static IDLE_MS: AtomicU64 = AtomicU64::new(0);

/// Enter architected idle until the next interrupt (timer, keyboard, …).
#[inline]
pub fn halt() {
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
