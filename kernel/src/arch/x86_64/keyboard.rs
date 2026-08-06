//! PS/2 keyboard, IRQ1 (`CHITTI_OS_HANDOFF.md` Phase 7: local console input).
//!
//! **This driver no longer decodes characters.** It tracks the modifier edges
//! scan-code set 1 delivers, turns each make/break code into a HID usage, and
//! hands a [`crate::keymap::KeyEvent`] to the shared choke point. Layout, dead
//! keys, Compose and the arrow→CSI table all live there, and are shared with the
//! other three transports.
//!
//! The split matters *here* more than anywhere else: this runs in **interrupt
//! context**. The translation side allocates (a keypress can emit several
//! characters), so it cannot run in an IRQ handler — which is why
//! [`crate::keymap::feed_event`] only pushes a 4-byte `Copy` struct into a fixed
//! ring and `keymap::next_byte` does the work on the drain side.

use super::idt::InterruptStackFrame;
use super::pic;
use super::port::inb;
use crate::keymap::{self, KeyEvent, Mods, Source};
use core::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};

const DATA_PORT: u16 = 0x60;

/// Incremented once per keyboard IRQ (kept from Phase 1 as a liveness signal).
pub static SCANCODES_RECEIVED: AtomicU64 = AtomicU64::new(0);

/// Live modifier bits, as a [`Mods`] bitset.
///
/// PS/2 delivers modifier *edges* (make and break codes), while USB HID delivers
/// a *level* in every report. Converting edge→level is these few lines and stays
/// in the driver; the shared layer is pure with respect to `Mods` and does not
/// have to carry both models.
static MODS: AtomicU8 = AtomicU8::new(0);

/// True when the previous scancode byte was the 0xE0 extended prefix (arrow
/// keys, right Ctrl/Alt, the GUI keys arrive as `E0 <code>`).
static E0_PREFIX: AtomicBool = AtomicBool::new(false);

fn set_mod(bit: u8, on: bool) {
    let cur = MODS.load(Ordering::Relaxed);
    MODS.store(if on { cur | bit } else { cur & !bit }, Ordering::Relaxed);
}

/// Pop the next translated byte, if any. Non-blocking; the shell polls this and
/// yields the CPU between polls.
///
/// Kept as this driver's entry point (rather than having `console` call
/// `keymap::next_byte` directly for x86) so `console::read_byte_raw` keeps its
/// per-transport shape. The ring it drains is shared, so calling it once is
/// enough however many keyboards are attached.
pub fn read_char() -> Option<u8> {
    keymap::next_byte()
}

extern "x86-interrupt" fn keyboard_handler(_frame: InterruptStackFrame) {
    // SAFETY: reading port 0x60 retrieves the scancode and clears the
    // controller's output-buffer-full status so IRQ1 keeps arriving.
    let scancode = unsafe { inb(DATA_PORT) };
    SCANCODES_RECEIVED.fetch_add(1, Ordering::Relaxed);

    if scancode == 0xe0 {
        E0_PREFIX.store(true, Ordering::Relaxed);
        pic::send_eoi(1);
        return;
    }
    let e0 = E0_PREFIX.swap(false, Ordering::Relaxed);

    // In set 1 a break code is the make code with bit 7 set.
    let pressed = scancode & 0x80 == 0;
    let make = scancode & 0x7f;

    if let Some(usage) = keymap::usage_from_set1(make, e0) {
        // A modifier updates the live bitset and produces no event of its own —
        // except Caps Lock, whose *toggle* is owned by `keymap` so it survives a
        // switch between transports.
        if let Some(bit) = keymap::modifier_bit(usage) {
            set_mod(bit, pressed);
        } else {
            let mods = Mods(MODS.load(Ordering::Relaxed));
            keymap::feed_event(KeyEvent { usage, mods, pressed, src: Source::Ps2Set1 });
        }
    }
    pic::send_eoi(1);
}

pub fn init() {
    super::idt::set_irq_handler(pic::KEYBOARD_VECTOR, keyboard_handler);
    crate::ktrace::log(
        "keyboard",
        "IRQ1 handler installed; set-1 scancodes -> keymap (layout-aware)",
    );
}
