//! PS/2 keyboard, IRQ1 (`CHITTI_OS_HANDOFF.md` Phase 7: local console input).
//! The IRQ handler decodes scan-code set 1 (what QEMU's i8042 emits) into
//! ASCII -- tracking shift and caps-lock -- and pushes the result into a small
//! ring buffer the intent shell drains via [`read_char`]. This is what lets a
//! human drive Chitti from the QEMU graphical window, not only over serial.

use super::idt::InterruptStackFrame;
use super::pic;
use super::port::inb;
use crate::mm::Locked;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

const DATA_PORT: u16 = 0x60;

/// Incremented once per keyboard IRQ (kept from Phase 1 as a liveness signal).
pub static SCANCODES_RECEIVED: AtomicU64 = AtomicU64::new(0);

static SHIFT_DOWN: AtomicBool = AtomicBool::new(false);
static CAPS_ON: AtomicBool = AtomicBool::new(false);
/// True when the previous scancode byte was the 0xE0 extended prefix (arrow
/// keys, etc. arrive as `E0 <code>`).
static E0_PREFIX: AtomicBool = AtomicBool::new(false);

// Scan-code set 1 make codes 0x00..0x40 -> ASCII (0 = no character). Break
// codes are the make code | 0x80 and are handled separately below.
#[rustfmt::skip]
static UNSHIFTED: [u8; 0x40] = [
    0,    0,    b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9', b'0', b'-', b'=', 0x08, b'\t',
    b'q', b'w', b'e', b'r', b't', b'y', b'u', b'i', b'o', b'p', b'[', b']', b'\n', 0,   b'a', b's',
    b'd', b'f', b'g', b'h', b'j', b'k', b'l', b';', b'\'', b'`', 0,   b'\\', b'z', b'x', b'c', b'v',
    b'b', b'n', b'm', b',', b'.', b'/', 0,    b'*', 0,    b' ', 0,    0,    0,    0,    0,    0,
];
#[rustfmt::skip]
static SHIFTED: [u8; 0x40] = [
    0,    0,    b'!', b'@', b'#', b'$', b'%', b'^', b'&', b'*', b'(', b')', b'_', b'+', 0x08, b'\t',
    b'Q', b'W', b'E', b'R', b'T', b'Y', b'U', b'I', b'O', b'P', b'{', b'}', b'\n', 0,   b'A', b'S',
    b'D', b'F', b'G', b'H', b'J', b'K', b'L', b':', b'"',  b'~', 0,   b'|',  b'Z', b'X', b'C', b'V',
    b'B', b'N', b'M', b'<', b'>', b'?', 0,    b'*', 0,    b' ', 0,    0,    0,    0,    0,    0,
];

const SC_LSHIFT: u8 = 0x2a;
const SC_RSHIFT: u8 = 0x36;
const SC_CAPS: u8 = 0x3a;

/// Decode a make code into an ASCII byte, applying shift/caps. Letters flip
/// case when caps-lock is on (so caps + shift = lowercase, as expected).
fn decode(scancode: u8) -> Option<u8> {
    let sc = scancode as usize;
    if sc >= UNSHIFTED.len() {
        return None;
    }
    let shift = SHIFT_DOWN.load(Ordering::Relaxed);
    let caps = CAPS_ON.load(Ordering::Relaxed);
    let base = if shift { SHIFTED[sc] } else { UNSHIFTED[sc] };
    if base == 0 {
        return None;
    }
    let ch = if caps && base.is_ascii_alphabetic() {
        if base.is_ascii_lowercase() {
            base - 32
        } else {
            base + 32
        }
    } else {
        base
    };
    Some(ch)
}

// --- decoded-key ring buffer --------------------------------------------

const RING_SIZE: usize = 128;

struct KeyRing {
    buf: [u8; RING_SIZE],
    head: usize,
    tail: usize,
}

static KEYS: Locked<KeyRing> = Locked::new(KeyRing { buf: [0; RING_SIZE], head: 0, tail: 0 });

impl KeyRing {
    fn push(&mut self, byte: u8) {
        let next = (self.head + 1) % RING_SIZE;
        if next != self.tail {
            self.buf[self.head] = byte;
            self.head = next;
        } // else: buffer full, drop the keystroke
    }

    fn pop(&mut self) -> Option<u8> {
        if self.head == self.tail {
            None
        } else {
            let byte = self.buf[self.tail];
            self.tail = (self.tail + 1) % RING_SIZE;
            Some(byte)
        }
    }
}

/// Pop the next decoded character typed at the keyboard, if any. Non-blocking;
/// the shell polls this and yields the CPU between polls.
pub fn read_char() -> Option<u8> {
    KEYS.with(|r| r.pop())
}

extern "x86-interrupt" fn keyboard_handler(_frame: InterruptStackFrame) {
    // SAFETY: reading port 0x60 retrieves the scancode and clears the
    // controller's output-buffer-full status so IRQ1 keeps arriving.
    let scancode = unsafe { inb(DATA_PORT) };
    SCANCODES_RECEIVED.fetch_add(1, Ordering::Relaxed);

    // Extended (0xE0-prefixed) keys: arrows become the ANSI sequences a serial
    // terminal sends (ESC [ A/B/C/D), so the shell decodes one encoding for
    // every input path (history navigation etc.). Other extended keys are eaten.
    if scancode == 0xe0 {
        E0_PREFIX.store(true, Ordering::Relaxed);
        pic::send_eoi(1);
        return;
    }
    if E0_PREFIX.swap(false, Ordering::Relaxed) {
        if let Some(fin) = match scancode {
            0x48 => Some(b'A'), // Up
            0x50 => Some(b'B'), // Down
            0x4d => Some(b'C'), // Right
            0x4b => Some(b'D'), // Left
            _ => None,
        } {
            KEYS.with(|r| {
                r.push(0x1b);
                r.push(b'[');
                r.push(fin);
            });
        }
        pic::send_eoi(1);
        return;
    }

    match scancode {
        SC_LSHIFT | SC_RSHIFT => SHIFT_DOWN.store(true, Ordering::Relaxed),
        // Shift release (make code | 0x80).
        0xaa | 0xb6 => SHIFT_DOWN.store(false, Ordering::Relaxed),
        SC_CAPS => {
            CAPS_ON.fetch_xor(true, Ordering::Relaxed);
        }
        _ if scancode < 0x80 => {
            if let Some(ch) = decode(scancode) {
                KEYS.with(|r| r.push(ch));
            }
        }
        _ => {} // other break codes: ignore
    }
    pic::send_eoi(1);
}

pub fn init() {
    super::idt::set_irq_handler(pic::KEYBOARD_VECTOR, keyboard_handler);
    crate::ktrace::log("keyboard", "IRQ1 handler installed, scan-code set 1 decoding enabled");
}
