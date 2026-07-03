//! The unified interactive console (`CHITTI_OS_HANDOFF.md` Phase 7): a thin
//! layer letting the intent shell read from *either* the PS/2 keyboard (the
//! graphical window) or the serial port, and echo to *both* the serial port
//! and the framebuffer. Ordinary line output (`serial_println!`) already
//! reaches the framebuffer because `serial::Serial` mirrors there; this module
//! adds the input side and per-keystroke echo the shell's line editor needs.

/// The next input byte from whichever console has one -- on x86, the PS/2
/// keyboard first then serial; on aarch64 (QEMU `virt` has no PS/2 keyboard),
/// serial only -- or `None` if none is available.
pub fn read_byte() -> Option<u8> {
    #[cfg(target_arch = "x86_64")]
    {
        // PS/2 keyboard, then a USB (xHCI/HID) keyboard, then serial.
        crate::arch::x86_64::keyboard::read_char()
            .or_else(crate::arch::x86_64::xhci::poll_key)
            .or_else(crate::serial::read_byte)
    }
    #[cfg(target_arch = "aarch64")]
    {
        // A USB (xHCI/HID) keyboard — the real-hardware input path — first, then
        // the virtio-keyboard (QEMU `virt` window), then the PL011 UART, so any
        // of them drives the shell.
        crate::arch::aarch64::xhci::poll_key()
            .or_else(crate::arch::aarch64::virtio_input::read_byte)
            .or_else(crate::serial::read_byte)
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        crate::serial::read_byte()
    }
}

/// Echo one byte to every output console (serial + framebuffer). Used by the
/// shell to echo keystrokes and draw its backspace; the framebuffer half is
/// absent from the test build (no framebuffer module there).
pub fn put_byte(byte: u8) {
    crate::serial::put_byte(byte);
    #[cfg(not(test))]
    crate::framebuffer::console_put_byte(byte);
}
