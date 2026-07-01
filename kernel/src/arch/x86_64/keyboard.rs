//! PS/2 keyboard controller, IRQ1. Phase 1 only needs to prove the IRQ
//! path works end to end (read the scancode so the controller's output
//! buffer clears and further IRQs keep arriving, acknowledge, log); actual
//! scancode-to-key decoding belongs to a later phase's input/shell work.

use super::idt::InterruptStackFrame;
use super::pic;
use super::port::inb;
use core::sync::atomic::{AtomicU64, Ordering};

const DATA_PORT: u16 = 0x60;

/// Incremented once per keyboard IRQ.
pub static SCANCODES_RECEIVED: AtomicU64 = AtomicU64::new(0);

extern "x86-interrupt" fn keyboard_handler(_frame: InterruptStackFrame) {
    // SAFETY: reading port 0x60 both retrieves the scancode and clears the
    // controller's "output buffer full" status, which is required for the
    // controller to raise IRQ1 again.
    let scancode = unsafe { inb(DATA_PORT) };
    SCANCODES_RECEIVED.fetch_add(1, Ordering::SeqCst);
    crate::ktrace::log_fmt(format_args!("keyboard: scancode {scancode:#x}"));
    pic::send_eoi(1);
}

pub fn init() {
    super::idt::set_irq_handler(pic::KEYBOARD_VECTOR, keyboard_handler);
    crate::ktrace::log("keyboard", "IRQ1 handler installed");
}
