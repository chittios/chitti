//! Minimal 16550 UART driver on COM1, used as the kernel's primary log
//! channel. Since Phase 1, interrupts can fire between any two
//! instructions, so `serial_print!`/`serial_println!` wrap each full
//! write in `arch::x86_64::interrupts::without_interrupts` to keep an
//! IRQ handler's own logging from interleaving with a write already in
//! progress; there is still no real lock since there is only one core.

use crate::mm::Locked;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

/// What a sink does with the text besides buffering it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SinkMode {
    /// Buffer **and** write through to the UART + framebuffer — what running a
    /// `/command` as an agent tool wants, so the human still sees it happen.
    Tee,
    /// Buffer **only**. A pipeline stage feeding another stage must not paint
    /// the console with output the user never asked to see.
    Redirect,
}

struct Sink {
    mode: SinkMode,
    buf: String,
    /// Set once `buf` hit [`SINK_MAX`]; further text is dropped.
    truncated: bool,
}

/// Most bytes one sink will buffer. A runaway command must not consume the heap
/// through a capture nobody is bounding — the allocator is first-fit and this
/// is one contiguous `String`.
pub const SINK_MAX: usize = 1024 * 1024;

/// The active capture sinks, innermost last.
///
/// **This is a stack, not a slot, and that is load-bearing.** `bg::pump` calls
/// [`crate::shell::run_tool_command`] from inside `upkeep()`, which long-running
/// commands call — its own comment says "run_tool_command may re-enter". With a
/// single `Option<String>` the inner call replaced the outer buffer and then
/// took it, so the outer command's remaining output went nowhere and its
/// `capture_end()` returned `""`: an agent got an empty result for a command
/// that ran fine. This is the same re-entrancy that made `in_tool_call` a depth
/// counter rather than a flag, one layer down.
///
/// Text goes to the **innermost** sink only. Appending to every live sink would
/// contaminate an outer command's result with an unrelated background job's
/// output, which is the opposite mistake.
static SINKS: Locked<Vec<Sink>> = Locked::new(Vec::new());

/// Start capturing into a new innermost sink.
pub fn sink_push(mode: SinkMode) {
    SINKS.with(|s| s.push(Sink { mode, buf: String::new(), truncated: false }));
}

/// Pop the innermost sink and return what it captured.
///
/// Returns `None` when the stack is empty — an unpaired pop, which is a caller
/// bug rather than something to paper over with an empty string.
pub fn sink_pop() -> Option<String> {
    SINKS.with(|s| s.pop()).map(|mut sink| {
        if sink.truncated {
            sink.buf.push_str("\n(output truncated at 1 MiB)\n");
        }
        sink.buf
    })
}

/// Depth of the sink stack (for tests and assertions).
pub fn sink_depth() -> usize {
    SINKS.with(|s| s.len())
}

/// Begin capturing serial output into an in-memory buffer, still echoing it to
/// the console. Thin wrapper over [`sink_push`] so existing callers are
/// unchanged.
pub fn capture_begin() {
    sink_push(SinkMode::Tee);
}

/// Stop capturing and return everything captured since [`capture_begin`].
pub fn capture_end() -> String {
    sink_pop().unwrap_or_default()
}

/// Raw, arch-specific byte transport: the 16550 UART (I/O ports) on x86, the
/// PL011 UART (MMIO) on aarch64. The rest of this module (line buffering,
/// framebuffer mirroring, the `serial_print!` macros) is arch-independent.
#[cfg(target_arch = "x86_64")]
mod raw {
    use crate::arch::x86_64::port::{inb, outb};
    const COM1: u16 = 0x3f8;

    pub fn init() {
        // SAFETY: standard 16550 init sequence, once at boot on COM1.
        unsafe {
            outb(COM1 + 1, 0x00); // disable UART interrupts
            outb(COM1 + 3, 0x80); // enable DLAB to set the baud divisor
            outb(COM1, 0x03); // divisor low -> 38400 baud
            outb(COM1 + 1, 0x00); // divisor high
            outb(COM1 + 3, 0x03); // 8N1; clears DLAB
            outb(COM1 + 2, 0xc7); // enable + clear FIFOs
            outb(COM1 + 4, 0x0b); // RTS/DSR set
        }
    }
    pub fn write(byte: u8) {
        // SAFETY: spin until LSR bit 5 (THR empty), then write the data port.
        unsafe {
            while inb(COM1 + 5) & 0x20 == 0 {}
            outb(COM1, byte);
        }
    }
    pub fn read() -> Option<u8> {
        // SAFETY: LSR bit 0 ("data ready") gates the read.
        unsafe {
            if inb(COM1 + 5) & 0x01 != 0 {
                Some(inb(COM1))
            } else {
                None
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod raw {
    // PL011 needs no setup to transmit at the default base. The real base is
    // discovered later by `arch::aarch64::init_uart` (called from `init()` once
    // the exception vectors are up, since its MMIO probe needs the fault
    // handler); until then the early banner uses QEMU's 0x09000000 default.
    pub fn init() {}
    pub fn write(byte: u8) {
        crate::arch::aarch64::uart_putb(byte);
    }
    pub fn read() -> Option<u8> {
        crate::arch::aarch64::uart_getb()
    }
}

/// Initialize the console UART. Idempotent.
pub fn init() {
    raw::init();
}

/// Read one byte from the console UART if one is buffered, else `None`.
/// Non-blocking: the intent shell polls this and yields between polls.
pub fn read_byte() -> Option<u8> {
    crate::arch::interrupts::without_interrupts(raw::read)
}

fn write_byte(byte: u8) {
    raw::write(byte);
}

/// Write a single raw byte, bypassing the printable-ASCII filter `Serial`'s
/// `Write` impl applies (the shell uses this for backspace: `\x08 \x08`).
pub fn put_byte(byte: u8) {
    crate::arch::interrupts::without_interrupts(|| write_byte(byte));
}

/// Write a string to the UART only, **without** mirroring to the framebuffer
/// chat pane the way [`Serial`] does. `ktrace` uses this (plus its own mirror to
/// the framebuffer *logs* pane) so trace output and chat output land in
/// different panes instead of interleaving in one.
pub fn write_str_raw(s: &str) {
    crate::arch::interrupts::without_interrupts(|| {
        for byte in s.bytes() {
            match byte {
                // Same rule as `Serial`'s `Write` impl: UTF-8 passes through, C0
                // controls become dots.
                0x20..=0x7e | 0x80..=0xff | b'\n' | b'\r' | b'\t' => write_byte(byte),
                _ => write_byte(b'.'),
            }
        }
    });
}

/// Zero-sized handle used to route `core::fmt::Write` (and thus
/// `write!`/the `serial_print!` macros) to COM1.
pub struct Serial;

impl fmt::Write for Serial {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        // Read the innermost sink's mode first, and release the lock before
        // touching the UART or the framebuffer: `console_print` is a large call
        // and holding a spinlock (with interrupts off) across it for every
        // printed string is exactly the contention the e2e harness's decode
        // slowdown taught us to avoid.
        let mode = SINKS.with(|s| s.last().map(|x| x.mode));
        if mode != Some(SinkMode::Redirect) {
            for byte in s.bytes() {
                match byte {
                    // Printable ASCII, the three whitespace controls, and **every
                    // byte of a UTF-8 sequence**.
                    //
                    // Non-ASCII used to be replaced with `.`, which was right while
                    // the OS could not hold non-ASCII text: the console would
                    // otherwise emit whatever a stray byte happened to be. Now that
                    // the line editor, the layouts and the IME all produce real
                    // UTF-8, a host terminal — which is UTF-8 — renders it correctly,
                    // and dotting it out means the serial console cannot show text the
                    // machine is holding. `\u{1b}` (ESC) is still filtered, since the
                    // colour sequences the shell emits go through `put_byte`.
                    0x20..=0x7e | 0x80..=0xff | b'\n' | b'\r' | b'\t' => write_byte(byte),
                    _ => write_byte(b'.'), // other C0 controls
                }
            }
            // Mirror to the framebuffer text console (Phase 7), so the graphical
            // window shows everything the serial port does. No-op until the
            // console is initialized; absent from the test build (which does not
            // compile the framebuffer module).
            #[cfg(not(test))]
            crate::framebuffer::console_print(s);
        }
        // Append to the innermost sink only — see `SINKS`. Appending to every
        // live sink would contaminate an outer command's captured result with
        // an unrelated nested job's output.
        if mode.is_some() {
            SINKS.with(|sinks| {
                if let Some(top) = sinks.last_mut() {
                    if top.buf.len() + s.len() > SINK_MAX {
                        top.truncated = true;
                    } else {
                        top.buf.push_str(s);
                    }
                }
            });
        }
        Ok(())
    }
}

#[macro_export]
macro_rules! serial_print {
    ($($arg:tt)*) => {{
        use core::fmt::Write as _;
        $crate::arch::interrupts::without_interrupts(|| {
            let _ = write!($crate::serial::Serial, $($arg)*);
        });
    }};
}

#[macro_export]
macro_rules! serial_println {
    () => { $crate::serial_print!("\n") };
    // Forward the caller's tokens verbatim (no `concat!`, which would mangle the
    // string's span and break inline `{var}` format captures), then a newline.
    ($($arg:tt)*) => {{
        $crate::serial_print!($($arg)*);
        $crate::serial_print!("\n");
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drain any sink a previous test left behind, so these are order-independent.
    fn clear() {
        while sink_pop().is_some() {}
    }

    #[test_case]
    fn a_nested_capture_does_not_clobber_the_outer_one() {
        // The bug this stack replaces. `bg::pump` calls `run_tool_command` from
        // inside `upkeep()`, which long-running commands call, so captures
        // nest. With a single `Option<String>` the inner `capture_begin` threw
        // the outer buffer away and the inner `capture_end` left `None` behind
        // — so the outer command's *remaining* output went nowhere and its
        // `capture_end()` returned "". An agent got an empty result for a
        // command that ran fine.
        clear();
        capture_begin();
        serial_print!("outer-before ");
        capture_begin();
        serial_print!("inner");
        let inner = capture_end();
        serial_print!("outer-after");
        let outer = capture_end();

        // The inner call sees only its own output...
        assert_eq!(inner, "inner");
        // ...and the outer keeps everything of its own, from both sides of the
        // nested call, with none of the inner's.
        assert_eq!(outer, "outer-before outer-after");
        assert_eq!(sink_depth(), 0);
    }

    #[test_case]
    fn redirect_buffers_without_writing_through() {
        // A pipeline stage feeding another stage must not paint the console.
        // Both modes buffer identically; they differ only in the write-through,
        // which is why this asserts the buffer and the depth rather than trying
        // to observe the UART.
        clear();
        sink_push(SinkMode::Redirect);
        serial_print!("captured");
        assert_eq!(sink_depth(), 1);
        assert_eq!(sink_pop().as_deref(), Some("captured"));
        assert_eq!(sink_depth(), 0);
    }

    #[test_case]
    fn an_unpaired_pop_is_none_rather_than_an_empty_string() {
        // "" and "nothing was capturing" are different facts, and a caller that
        // cannot tell them apart reports a silent empty result — the exact
        // shape of the bug above.
        clear();
        assert_eq!(sink_pop(), None);
    }

    #[test_case]
    fn a_runaway_capture_is_bounded_and_says_so() {
        // One contiguous String on a first-fit heap; an unbounded capture is a
        // way for one command to consume the heap.
        clear();
        sink_push(SinkMode::Redirect);
        let chunk = "x".repeat(64 * 1024);
        for _ in 0..20 {
            serial_print!("{}", chunk);
        }
        let out = sink_pop().expect("sink present");
        assert!(out.len() <= SINK_MAX + 64, "capture grew past the cap: {}", out.len());
        // Truncation is reported in the output rather than silently losing the
        // tail, which would read as the command having produced less.
        assert!(out.ends_with("(output truncated at 1 MiB)\n"), "no truncation marker");
    }
}
