//! Minimal 16550 UART driver on COM1, used as the kernel's primary log
//! channel. Since Phase 1, interrupts can fire between any two
//! instructions, so `serial_print!`/`serial_println!` wrap each full
//! write in `arch::x86_64::interrupts::without_interrupts` to keep an
//! IRQ handler's own logging from interleaving with a write already in
//! progress; there is still no real lock since there is only one core.

use crate::mm::Locked;
use alloc::string::String;
use core::fmt;

/// Optional capture sink: when `Some`, every `serial_print!`/`serial_println!`
/// byte is also appended here (in addition to the UART + framebuffer). The shell
/// uses this to run a `/command` and return its printed output as a tool result
/// to the root agent, without rewriting each command handler.
static CAPTURE: Locked<Option<String>> = Locked::new(None);

/// Begin capturing serial output into an in-memory buffer.
pub fn capture_begin() {
    CAPTURE.with(|c| *c = Some(String::new()));
}

/// Stop capturing and return everything captured since [`capture_begin`].
pub fn capture_end() -> String {
    CAPTURE.with(|c| c.take().unwrap_or_default())
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
        // If a capture is active (a `/command` running as an agent tool), also
        // append the text so it can be returned as the tool result.
        CAPTURE.with(|c| {
            if let Some(buf) = c {
                buf.push_str(s);
            }
        });
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
