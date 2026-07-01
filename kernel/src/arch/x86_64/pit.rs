//! 8253/8254 Programmable Interval Timer, channel 0, wired to IRQ0. Drives
//! `TICKS`, the monotonic counter Phase 1's timer test watches.

use super::idt::InterruptStackFrame;
use super::pic;
use super::port::outb;
use core::sync::atomic::{AtomicU64, Ordering};

const PIT_FREQUENCY_HZ: u32 = 1000;
const PIT_BASE_FREQUENCY_HZ: u32 = 1_193_182;
const CHANNEL0_DATA: u16 = 0x40;
const COMMAND: u16 = 0x43;

/// Incremented once per timer IRQ. `kernel::tests` reads this to prove
/// timer interrupts are actually firing.
pub static TICKS: AtomicU64 = AtomicU64::new(0);

pub fn ticks() -> u64 {
    TICKS.load(Ordering::SeqCst)
}

extern "x86-interrupt" fn timer_handler(_frame: InterruptStackFrame) {
    TICKS.fetch_add(1, Ordering::SeqCst);
    pic::send_eoi(0);
}

/// Program channel 0 for `PIT_FREQUENCY_HZ` (1kHz: a good balance between
/// test speed and not flooding `ktrace`-adjacent code with IRQs) and
/// install the IRQ0 handler.
pub fn init() {
    let divisor = PIT_BASE_FREQUENCY_HZ / PIT_FREQUENCY_HZ;
    // SAFETY: standard PIT channel-0 mode-3 (square wave) programming
    // sequence: command byte, then divisor low/high bytes.
    unsafe {
        outb(COMMAND, 0b0011_0110); // channel 0, lobyte/hibyte, mode 3, binary
        outb(CHANNEL0_DATA, (divisor & 0xff) as u8);
        outb(CHANNEL0_DATA, ((divisor >> 8) & 0xff) as u8);
    }
    super::idt::set_irq_handler(pic::TIMER_VECTOR, timer_handler);
    crate::ktrace::log_fmt(format_args!("pit: channel 0 programmed for {PIT_FREQUENCY_HZ} Hz"));
}
