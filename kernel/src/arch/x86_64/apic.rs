//! Local APIC (xAPIC), the per-core interrupt controller
//! (`CHITTI_OS_HANDOFF.md` Phase 7: "APIC per core"). Limine starts every
//! core with its local APIC hardware-enabled (`IA32_APIC_BASE.EN`); this
//! module reads the per-core APIC id and *software*-enables the APIC (sets
//! the spurious-interrupt-vector register), which every core does as it comes
//! online.
//!
//! Only the pieces SMP bring-up needs are here. The application processors run
//! a cooperative, interrupts-disabled worker on each core (`crate::smp`), so
//! no IO-APIC redirection or per-core APIC-timer programming is required yet;
//! the legacy PIC/PIT (`pic.rs`/`pit.rs`) still drives the BSP's timer and
//! keyboard IRQs, untouched. Wiring the APIC timer / IPIs is future work.

use core::arch::asm;
use core::sync::atomic::{AtomicU64, Ordering};

/// `IA32_APIC_BASE` MSR: bits 12..=35 hold the local APIC's physical base
/// (0xFEE00000 on reset), bit 11 is the global enable Limine leaves set.
const IA32_APIC_BASE: u32 = 0x1b;

/// Register offsets within the local APIC's 4 KiB MMIO page.
const REG_ID: u64 = 0x20;
const REG_SPURIOUS: u64 = 0xf0;

/// Spurious-interrupt vector we point the APIC at; the vector is never
/// expected to fire (APs keep interrupts disabled), but the register's bit 8
/// is the APIC software-enable, so it must be written to bring the APIC fully
/// online.
const SPURIOUS_VECTOR: u32 = 0xff;
const APIC_SOFTWARE_ENABLE: u32 = 1 << 8;

/// Virtual address the local APIC MMIO page is mapped at, cached after
/// `init_mapping`. The local APIC lives at the same physical address on every
/// core, so one mapping (in the page tables all cores share) serves them all.
static APIC_VIRT: AtomicU64 = AtomicU64::new(0);

fn read_msr(msr: u32) -> u64 {
    let (lo, hi): (u32, u32);
    // SAFETY: reading IA32_APIC_BASE is valid on any x86_64 CPU in long mode.
    unsafe { asm!("rdmsr", in("ecx") msr, out("eax") lo, out("edx") hi, options(nomem, nostack, preserves_flags)) };
    ((hi as u64) << 32) | lo as u64
}

/// Map the local APIC MMIO page (it sits in a hole the HHDM does not cover)
/// and cache its virtual address. Must run once on the BSP after `mm::init`
/// and before any `local_id`/`software_enable`; APs reuse the shared mapping.
pub fn init_mapping() {
    let phys = read_msr(IA32_APIC_BASE) & 0xffff_f000;
    let virt = crate::mm::map_mmio_page(phys);
    APIC_VIRT.store(virt, Ordering::SeqCst);
}

fn read_reg(offset: u64) -> u32 {
    let base = APIC_VIRT.load(Ordering::SeqCst);
    debug_assert!(base != 0, "apic: used before init_mapping");
    // SAFETY: `base + offset` is the mapped MMIO address of a valid local-APIC
    // register; APIC registers are 32-bit and must be accessed volatile.
    unsafe { core::ptr::read_volatile((base + offset) as *const u32) }
}

fn write_reg(offset: u64, value: u32) {
    let base = APIC_VIRT.load(Ordering::SeqCst);
    debug_assert!(base != 0, "apic: used before init_mapping");
    // SAFETY: as `read_reg`; the spurious-vector register is writable.
    unsafe { core::ptr::write_volatile((base + offset) as *mut u32, value) };
}

/// This core's local-APIC id (bits 24..=31 of the id register, for xAPIC).
pub fn local_id() -> u32 {
    read_reg(REG_ID) >> 24
}

/// Software-enable this core's local APIC. Idempotent; called by every core
/// (BSP and APs) as it comes online. Requires `init_mapping` to have run.
pub fn software_enable() {
    let spurious = read_reg(REG_SPURIOUS);
    write_reg(REG_SPURIOUS, spurious | APIC_SOFTWARE_ENABLE | SPURIOUS_VECTOR);
}

// --- local-APIC timer ----------------------------------------------------
//
// The scheduler tick's real home on any modern machine. The 8254 PIT on IRQ0
// through the 8259 PIC is a legacy-PC arrangement that UEFI-only hardware may
// omit entirely; the local-APIC timer is per-core, present wherever an APIC is,
// and delivers through the LVT — a *local* interrupt, so it needs no IOAPIC
// redirection entry and no MADT parsing to route.
//
// The catch is that it counts at the (unspecified) core-crystal/bus rate, so the
// count matching a desired frequency has to be measured against a clock whose
// rate we already know.

/// LVT Timer entry.
const REG_LVT_TIMER: u64 = 0x320;
/// Initial count (writing it starts the timer).
const REG_TIMER_INIT: u64 = 0x380;
/// Current count (counts down to zero).
const REG_TIMER_CUR: u64 = 0x390;
/// Divide configuration.
const REG_TIMER_DIV: u64 = 0x3e0;

/// LVT bit 16 masks the interrupt.
const LVT_MASKED: u32 = 1 << 16;
/// LVT bits 18:17 = 01 selects periodic mode.
const LVT_PERIODIC: u32 = 1 << 17;
/// Divide-by-16 (bits 3,1,0 = 0b0011).
const TIMER_DIV_16: u32 = 0x3;
/// The divisor the above encoding selects, needed for the arithmetic.
const TIMER_DIVISOR: u64 = 16;

/// How long to measure the APIC timer against the reference clock. Long enough
/// that reference-clock granularity is noise, short enough not to stall boot.
const CALIBRATE_US: u64 = 10_000; // 10 ms

/// Convert a calibration measurement into the periodic initial-count for `hz`.
///
/// `counted` is how many APIC timer ticks elapsed over `measured_us`
/// microseconds. Kept pure and separate because the failure modes here are
/// arithmetic, not hardware: a zero measurement (reference clock never advanced)
/// must not divide by zero, and a slow timer must still yield a non-zero count or
/// the timer would be programmed with 0 — which *stops* it rather than firing
/// immediately.
pub(crate) fn periodic_count(counted: u32, measured_us: u64, hz: u64) -> Option<u32> {
    if counted == 0 || measured_us == 0 || hz == 0 {
        return None;
    }
    // Full precision (u128): ticks/s = counted * 1e6 / measured_us. Doing this in
    // u64 with an intermediate ticks-per-microsecond loses a slow timer entirely
    // to integer division.
    let per_sec = (counted as u128 * 1_000_000u128) / measured_us as u128;
    let count = per_sec / hz as u128;
    if count == 0 {
        return None; // timer too slow to divide down to `hz`
    }
    Some(count.min(u32::MAX as u128) as u32)
}

/// Calibrate and start this core's APIC timer in periodic mode at `hz`,
/// delivering `vector`. Returns false if no reference clock is available or the
/// timer does not count, leaving the caller on its previous timer.
///
/// `delay_us` is the reference-clock busy-wait; it returns false if that clock is
/// unusable, which aborts calibration rather than producing a bogus rate.
pub fn start_timer(hz: u64, vector: u8, delay_us: impl Fn(u64) -> bool) -> bool {
    write_reg(REG_TIMER_DIV, TIMER_DIV_16);
    // Measure: run free from the maximum count, masked so calibration cannot
    // deliver an interrupt, and see how far it gets.
    write_reg(REG_LVT_TIMER, LVT_MASKED | vector as u32);
    write_reg(REG_TIMER_INIT, u32::MAX);
    if !delay_us(CALIBRATE_US) {
        write_reg(REG_TIMER_INIT, 0);
        return false;
    }
    let remaining = read_reg(REG_TIMER_CUR);
    write_reg(REG_TIMER_INIT, 0); // stop
    let counted = u32::MAX - remaining;
    let Some(count) = periodic_count(counted, CALIBRATE_US, hz) else {
        crate::ktrace::log_fmt(format_args!(
            "apic: timer calibration failed (counted {counted} in {CALIBRATE_US} us) -- keeping the PIT"
        ));
        return false;
    };
    // The measured rate is per-divided-tick; report the underlying bus rate for
    // the log so an implausible value is obvious.
    let hz_measured = (counted as u64) * (1_000_000 / CALIBRATE_US) * TIMER_DIVISOR;
    crate::ktrace::log_fmt(format_args!(
        "apic: timer calibrated -- {} MHz bus, initial count {count} for {hz} Hz",
        hz_measured / 1_000_000
    ));
    write_reg(REG_TIMER_DIV, TIMER_DIV_16);
    write_reg(REG_LVT_TIMER, LVT_PERIODIC | vector as u32);
    write_reg(REG_TIMER_INIT, count);
    true
}

/// Acknowledge an APIC-delivered interrupt (End Of Interrupt).
pub fn eoi() {
    write_reg(0xb0, 0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn periodic_count_basic_rate() {
        // 1,000,000 ticks in 10 ms = 100 MHz; at 1000 Hz that is 100,000 counts.
        assert_eq!(periodic_count(1_000_000, 10_000, 1000), Some(100_000));
        // Same rate, 100 Hz -> ten times the count.
        assert_eq!(periodic_count(1_000_000, 10_000, 100), Some(1_000_000));
    }

    #[test_case]
    fn periodic_count_rejects_degenerate_inputs() {
        // A reference clock that never advanced, or a timer that never counted,
        // must not divide by zero or return 0 — writing 0 to the initial-count
        // register STOPS the timer instead of firing immediately, which would
        // look like a hang rather than a calibration failure.
        assert_eq!(periodic_count(0, 10_000, 1000), None);
        assert_eq!(periodic_count(1_000_000, 0, 1000), None);
        assert_eq!(periodic_count(1_000_000, 10_000, 0), None);
    }

    #[test_case]
    fn periodic_count_refuses_a_timer_too_slow_for_the_rate() {
        // 100 ticks in 10 ms = 10 kHz; it cannot produce a 100 kHz tick, and the
        // count must come back None rather than 0.
        assert_eq!(periodic_count(100, 10_000, 100_000), None);
    }

    #[test_case]
    fn periodic_count_saturates_instead_of_wrapping() {
        // An enormous measured rate at a very low target frequency must clamp to
        // u32::MAX, not wrap to a tiny count (which would fire a storm).
        assert_eq!(periodic_count(u32::MAX, 1, 1), Some(u32::MAX));
    }
}
