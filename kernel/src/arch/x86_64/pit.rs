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
    crate::sched::on_timer_tick();
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

/// The vector the local-APIC timer delivers on. Above the 32..47 range the PIC
/// occupies, so both timers can be installed without colliding.
pub const APIC_TIMER_VECTOR: u8 = 0x40;

extern "x86-interrupt" fn apic_timer_handler(_frame: InterruptStackFrame) {
    TICKS.fetch_add(1, Ordering::SeqCst);
    // APIC-delivered interrupts are acknowledged at the local APIC, not the PIC.
    super::apic::eoi();
    crate::sched::on_timer_tick();
}

/// Try to move the scheduler tick to the **local-APIC timer**, using the HPET as
/// the calibration reference. Returns true if it took over.
///
/// Why bother when the PIT works under QEMU: the PIT (8254) and the PIC (8259)
/// are legacy-PC devices a UEFI-only machine is permitted to omit, and modern
/// hardware increasingly does. There the PIT programming above writes to
/// unclaimed ports, IRQ0 never arrives, and the kernel runs with no preemption at
/// all — no error, just a scheduler that only switches when a task yields. The
/// APIC timer is per-core, present wherever an APIC is, and delivers through the
/// LVT, so it needs no IOAPIC redirection entry.
///
/// The PIT is deliberately left programmed underneath: if the APIC timer turns out
/// not to fire, the machine still ticks wherever the PIT does work, and double
/// counting is harmless (both handlers only bump `TICKS` and poke the scheduler).
pub fn try_apic_timer(rsdp: u64) -> bool {
    if !super::hpet::init(rsdp) {
        crate::ktrace::log("pit", "no HPET to calibrate against -- staying on the PIT/PIC tick");
        return false;
    }
    super::idt::set_irq_handler(APIC_TIMER_VECTOR, apic_timer_handler);
    if !super::apic::start_timer(PIT_FREQUENCY_HZ as u64, APIC_TIMER_VECTOR, super::hpet::delay_us) {
        return false;
    }
    // The APIC timer now drives the tick; mask the PIT's IRQ0 so the two do not
    // both count (harmless, but it doubles the effective tick rate).
    pic::mask_irq(0);
    crate::ktrace::log_fmt(format_args!(
        "pit: scheduler tick moved to the local-APIC timer at {PIT_FREQUENCY_HZ} Hz (vector {APIC_TIMER_VECTOR:#x}); PIT IRQ0 masked"
    ));
    true
}
