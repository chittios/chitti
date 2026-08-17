//! The unified interactive console (`CHITTI_OS_HANDOFF.md` Phase 7): a thin
//! layer letting the intent shell read from *either* the PS/2 keyboard (the
//! graphical window) or the serial port, and echo to *both* the serial port
//! and the framebuffer. Ordinary line output (`serial_println!`) already
//! reaches the framebuffer because `serial::Serial` mirrors there; this module
//! adds the input side and per-keystroke echo the shell's line editor needs.

use core::sync::atomic::{AtomicU64, Ordering};

/// `now_ms` of the last keyboard byte read — drives the status-bar keyboard
/// activity indicator.
static INPUT_ACTIVITY_MS: AtomicU64 = AtomicU64::new(0);

/// A small pushback ring (bytes read speculatively — e.g. by a running
/// command's cancel-poll checking for Ctrl+C — can be returned to the input
/// stream if they aren't what was wanted). Multiple pushed bytes are kept in
/// order, so the reverse-search and its tests can inject a whole keystroke
/// sequence before reading it back.
static PENDING_BUF: [AtomicU64; 8] = [const { AtomicU64::new(EMPTY) }; 8];
static PENDING_HEAD: AtomicU64 = AtomicU64::new(0);
static PENDING_TAIL: AtomicU64 = AtomicU64::new(0);
const EMPTY: u64 = 0x100;

/// Push a byte back so the next [`read_byte`] returns it (FIFO). Dropped if
/// the ring is full (8) — the poll loops never hold more than one.
pub fn unread(byte: u8) {
    let head = PENDING_HEAD.load(Ordering::Relaxed);
    if head - PENDING_TAIL.load(Ordering::Relaxed) >= PENDING_BUF.len() as u64 {
        return; // full
    }
    PENDING_BUF[(head % PENDING_BUF.len() as u64) as usize].store(byte as u64, Ordering::Relaxed);
    PENDING_HEAD.store(head + 1, Ordering::Relaxed);
}

/// When keyboard input was last seen (`arch::now_ms` timebase; 0 = never).
pub fn input_activity_ms() -> u64 {
    INPUT_ACTIVITY_MS.load(Ordering::Relaxed)
}

/// A keyboard device has enumerated, or someone has already typed.
pub fn keyboard_present() -> bool {
    #[cfg(target_arch = "x86_64")]
    let hid = crate::arch::x86_64::xhci::has_keyboard();
    #[cfg(target_arch = "aarch64")]
    let hid = crate::arch::aarch64::xhci::has_keyboard();
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let hid = false;
    hid || input_activity_ms() != 0
}

/// Status-bar chip for the keyboard.
pub fn keyboard_status() -> crate::icons::DeviceStatus {
    if keyboard_present() {
        crate::icons::DeviceStatus::Ready
    } else {
        crate::icons::DeviceStatus::Disabled
    }
}

/// The next input byte from whichever console has one -- x86: PS/2 keyboard →
/// USB (xHCI/HID) → serial; aarch64: USB (xHCI/HID) → PL050 PS/2 → virtio-keyboard
/// → PL011 serial -- or `None` if none is available.
pub fn read_byte() -> Option<u8> {
    // Pushed-back bytes (see `unread`) take priority over a fresh read.
    let tail = PENDING_TAIL.load(Ordering::Relaxed);
    if PENDING_HEAD.load(Ordering::Relaxed) > tail {
        let b = PENDING_BUF[(tail % PENDING_BUF.len() as u64) as usize].swap(EMPTY, Ordering::Relaxed);
        PENDING_TAIL.store(tail + 1, Ordering::Relaxed);
        return Some(b as u8);
    }
    let b = read_byte_raw();
    if b.is_some() {
        INPUT_ACTIVITY_MS.store(crate::arch::now_ms(), Ordering::Relaxed);
    }
    b
}

fn read_byte_raw() -> Option<u8> {
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
        // a PL050 PS/2 keyboard (ARM dev boards / hypervisors that present one),
        // then the virtio-keyboard (QEMU `virt` window), then the PL011 UART, so
        // any of them drives the shell. (PL050 is the ARM analogue of the x86
        // i8042 PS/2 keyboard, giving both arches a PS/2 input path.)
        crate::arch::aarch64::xhci::poll_key()
            .or_else(crate::arch::aarch64::pl050::poll_key)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A pushed-back byte is returned by the very next `read_byte`, ahead of any
    /// hardware input — this is what lets a running command's cancel-poll peek for
    /// Ctrl+C without swallowing the next command's keystrokes.
    #[test_case]
    fn pushback_is_returned_before_hardware() {
        unread(b'Z');
        assert_eq!(read_byte(), Some(b'Z'));
        // The slot is one-deep and now empty; a second push round-trips too.
        unread(0x03);
        assert_eq!(read_byte(), Some(0x03));
    }
}
