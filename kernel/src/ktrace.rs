//! Deterministic kernel tracing: the `strace`-equivalent every subsystem
//! logs through, per `CHITTI_OS_HANDOFF.md` Part 4 ("ktrace every
//! capability invocation and every inference call ... from the moment
//! those subsystems exist"). Phase 1 only has serial to log to; later
//! phases can extend `log`/`log_fmt` to also append to an in-memory audit
//! log without changing any call site, keeping every future ktrace call
//! site unchanged.
//!
//! Every log line is prefixed with a monotonically increasing sequence
//! number rather than a wall-clock timestamp: Phase 1 has no
//! battery-backed clock, and a plain counter is exactly as useful for
//! ordering events while staying fully deterministic and reproducible.

use core::fmt::{Arguments, Write};
use core::sync::atomic::{AtomicU64, Ordering};

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn next_seq() -> u64 {
    SEQUENCE.fetch_add(1, Ordering::SeqCst)
}

/// Log a static `(subsystem, message)` pair.
pub fn log(subsystem: &str, message: &str) {
    log_fmt(format_args!("{subsystem}: {message}"));
}

/// Log a pre-built `format_args!(...)` value; callers that need
/// interpolation call this directly instead of `log`.
pub fn log_fmt(args: Arguments<'_>) {
    // Disable interrupts for the whole line: an IRQ handler that also
    // logs (e.g. the keyboard handler) must not interleave its bytes with
    // a log line already in progress on another path.
    crate::arch::x86_64::interrupts::without_interrupts(|| {
        let seq = next_seq();
        let mut serial = crate::serial::Serial;
        let _ = write!(serial, "[ktrace #{seq}] ");
        let _ = serial.write_fmt(args);
        let _ = writeln!(serial);
    });
}
