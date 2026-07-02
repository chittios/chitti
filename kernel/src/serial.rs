//! Minimal 16550 UART driver on COM1, used as the kernel's primary log
//! channel. Since Phase 1, interrupts can fire between any two
//! instructions, so `serial_print!`/`serial_println!` wrap each full
//! write in `arch::x86_64::interrupts::without_interrupts` to keep an
//! IRQ handler's own logging from interleaving with a write already in
//! progress; there is still no real lock since there is only one core.

use crate::arch::x86_64::port::{inb, outb};
use core::fmt;

const COM1: u16 = 0x3f8;

/// Initialize COM1 to 38400 8N1 with FIFOs enabled. Idempotent.
pub fn init() {
    // SAFETY: standard 16550 UART initialization sequence, run once at boot
    // on COM1 before anything else touches these ports.
    unsafe {
        outb(COM1 + 1, 0x00); // disable UART interrupts
        outb(COM1 + 3, 0x80); // enable DLAB to set the baud rate divisor
        outb(COM1, 0x03); // divisor low byte -> 38400 baud
        outb(COM1 + 1, 0x00); // divisor high byte
        outb(COM1 + 3, 0x03); // 8 bits, no parity, one stop bit; clears DLAB
        outb(COM1 + 2, 0xc7); // enable + clear 14-byte-threshold FIFOs
        outb(COM1 + 4, 0x0b); // IRQs enabled (unused for now), RTS/DSR set
    }
}

fn transmit_empty() -> bool {
    // SAFETY: COM1's line status register; bit 5 is "transmit holding
    // register empty".
    unsafe { inb(COM1 + 5) & 0x20 != 0 }
}

/// Read one byte from COM1 if the UART has one buffered, else `None`.
/// Non-blocking: the intent shell (Phase 5) polls this and yields the CPU
/// between polls rather than busy-waiting. Under QEMU `-serial stdio`, bytes
/// typed at the terminal arrive here on the receive line.
pub fn read_byte() -> Option<u8> {
    crate::arch::x86_64::interrupts::without_interrupts(|| {
        // SAFETY: LSR bit 0 ("data ready") gates the read; only when it is
        // set is there a byte in the receive buffer to take from COM1.
        unsafe {
            if inb(COM1 + 5) & 0x01 != 0 {
                Some(inb(COM1))
            } else {
                None
            }
        }
    })
}

fn write_byte(byte: u8) {
    while !transmit_empty() {}
    // SAFETY: only written once `transmit_empty` confirms the UART is
    // ready for the next byte.
    unsafe { outb(COM1, byte) };
}

/// Write a single raw byte to COM1, bypassing the printable-ASCII filter
/// `Serial`'s `Write` impl applies. The intent shell uses this to emit the
/// control bytes for in-line editing (backspace: `\x08 \x08`), which the
/// filter would otherwise turn into dots.
pub fn put_byte(byte: u8) {
    crate::arch::x86_64::interrupts::without_interrupts(|| write_byte(byte));
}

/// Zero-sized handle used to route `core::fmt::Write` (and thus
/// `write!`/the `serial_print!` macros) to COM1.
pub struct Serial;

impl fmt::Write for Serial {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for byte in s.bytes() {
            match byte {
                0x20..=0x7e | b'\n' | b'\r' | b'\t' => write_byte(byte),
                _ => write_byte(b'.'), // placeholder for non-ASCII bytes
            }
        }
        // Mirror to the framebuffer text console (Phase 7), so the graphical
        // window shows everything the serial port does. No-op until the
        // console is initialized; absent from the test build (which does not
        // compile the framebuffer module).
        #[cfg(not(test))]
        crate::framebuffer::console_print(s);
        Ok(())
    }
}

#[macro_export]
macro_rules! serial_print {
    ($($arg:tt)*) => {{
        use core::fmt::Write as _;
        $crate::arch::x86_64::interrupts::without_interrupts(|| {
            let _ = write!($crate::serial::Serial, $($arg)*);
        });
    }};
}

#[macro_export]
macro_rules! serial_println {
    () => { $crate::serial_print!("\n") };
    ($fmt:expr) => { $crate::serial_print!(concat!($fmt, "\n")) };
    ($fmt:expr, $($arg:tt)*) => { $crate::serial_print!(concat!($fmt, "\n"), $($arg)*) };
}
