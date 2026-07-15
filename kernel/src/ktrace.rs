//! Deterministic kernel tracing: the `strace`-equivalent every subsystem
//! logs through, per `CHITTI_OS_HANDOFF.md` Part 4 ("ktrace every
//! capability invocation and every inference call ... from the moment
//! those subsystems exist"). Phase 1 only has serial to log to; later
//! phases can extend `log`/`log_fmt` to also append to an in-memory audit
//! log without changing any call site, keeping every future ktrace call
//! site unchanged.
//!
//! Every log line is prefixed with a monotonically increasing sequence
//! number (deterministic ordering) plus the local wall-clock time with
//! milliseconds (`HH:MM:SS.mmm`, from [`crate::clock`]) — the timestamp is
//! what makes latency between two trace lines readable during performance
//! analysis.

use core::fmt::{Arguments, Write};
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// When set, every trace line ALSO mirrors to the framebuffer **chat** pane (not
/// just the logs pane). The logs pane is closed at boot and needs a keyboard to
/// open, so on a bare Apple boot — where we are bringing up that very keyboard
/// and have no serial — this is the only way to SEE driver diagnostics. Enabled
/// by `apple_usb` when the `chitti.usb` debug bootarg is present; off everywhere
/// else, so the normal tmux-style chat|logs split is unchanged.
static CONSOLE_ECHO: AtomicBool = AtomicBool::new(false);

/// Mirror trace to the chat pane too (see [`CONSOLE_ECHO`]).
pub fn set_console_echo(on: bool) {
    CONSOLE_ECHO.store(on, Ordering::Relaxed);
}

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
    crate::arch::interrupts::without_interrupts(|| {
        let seq = next_seq();
        let (h, m, sec, ms) = crate::clock::local_hms_ms();
        let mut sink = LogSink;
        let _ = write!(sink, "[ktrace #{seq} {h:02}:{m:02}:{sec:02}.{ms:03}] ");
        let _ = sink.write_fmt(args);
        let _ = writeln!(sink);
    });
}

/// Sink for trace lines: the UART (raw, un-mirrored) plus the framebuffer
/// **logs** pane. Keeping trace off the [`Serial`](crate::serial::Serial) path
/// means it draws into the logs pane rather than the chat pane, giving the
/// framebuffer TUI a tmux-style split (chat left, live ktrace right).
struct LogSink;

impl Write for LogSink {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        crate::serial::write_str_raw(s);
        #[cfg(not(test))]
        {
            crate::framebuffer::log_print(s);
            // Bare-Apple driver bring-up: also echo to the visible chat pane.
            if CONSOLE_ECHO.load(Ordering::Relaxed) {
                crate::framebuffer::console_print(s);
            }
        }
        Ok(())
    }
}
