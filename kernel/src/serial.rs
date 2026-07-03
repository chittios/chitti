//! Minimal 16550 UART driver on COM1, used as the kernel's primary log
//! channel. Since Phase 1, interrupts can fire between any two
//! instructions, so `serial_print!`/`serial_println!` wrap each full
//! write in `arch::x86_64::interrupts::without_interrupts` to keep an
//! IRQ handler's own logging from interleaving with a write already in
//! progress; there is still no real lock since there is only one core.

use core::fmt;

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
    /// Discover the PL011 base from ACPI SPCR (so we hit the right MMIO on a
    /// platform whose map differs from QEMU `virt`); PL011 needs no further
    /// setup to transmit. A no-op fallback keeps QEMU's 0x09000000 default.
    pub fn init() {
        crate::arch::aarch64::init_uart();
    }
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
        $crate::arch::interrupts::without_interrupts(|| {
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
