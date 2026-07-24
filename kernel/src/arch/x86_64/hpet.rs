//! **HPET (High Precision Event Timer)** — a monotonic reference clock, used to
//! calibrate the local-APIC timer.
//!
//! Why this exists: the scheduler tick used to come from the legacy 8254 PIT on
//! IRQ0 through the 8259 PIC. Both are legacy-PC devices that modern UEFI-only
//! machines are allowed to omit, and increasingly do — on such a machine there is
//! no tick at all and the kernel silently runs cooperatively. The fix is the
//! **local-APIC timer** (present on everything with an APIC, and a *local*
//! interrupt so it needs no IOAPIC routing), but the APIC timer counts at an
//! unspecified rate and must be calibrated against a clock of known frequency.
//! That is this module's job.
//!
//! Only the free-running main counter is used — no comparators, no interrupts.
//! The counter is 64-bit and its period is reported by the hardware in
//! femtoseconds, so converting to microseconds needs no calibration of its own.
//!
//! Discovered via the ACPI HPET table ([`crate::acpi::hpet_from_rsdp`]), never a
//! fixed address.

use core::sync::atomic::{AtomicU64, Ordering};

/// General capabilities and id. Bits 63:32 are the counter period in
/// femtoseconds; bit 13 says the counter is 64-bit.
const CAP_ID: u64 = 0x000;
/// General configuration; bit 0 (`ENABLE_CNF`) starts the main counter.
const GEN_CONF: u64 = 0x010;
/// The free-running main counter.
const MAIN_COUNTER: u64 = 0x0f0;

const ENABLE_CNF: u64 = 1 << 0;

/// One femtosecond is 1e-15 s; a microsecond is 1e-9 femtoseconds.
const FS_PER_US: u64 = 1_000_000_000;

/// Mapped register base, or 0 when absent / not yet initialised.
static BASE: AtomicU64 = AtomicU64::new(0);
/// Counter period in femtoseconds, from the capability register.
static PERIOD_FS: AtomicU64 = AtomicU64::new(0);

fn r64(off: u64) -> u64 {
    let base = BASE.load(Ordering::Relaxed);
    // SAFETY: `base` is the mapped HPET register block; every offset used here is
    // inside its first page. HPET registers are 64-bit and must be volatile.
    unsafe { core::ptr::read_volatile((base + off) as *const u64) }
}

fn w64(off: u64, v: u64) {
    let base = BASE.load(Ordering::Relaxed);
    // SAFETY: as `r64`; GEN_CONF is writable.
    unsafe { core::ptr::write_volatile((base + off) as *mut u64, v) };
}

/// Discover and start the HPET. Returns `true` if a usable counter is running.
///
/// `rsdp` is the validated ACPI root pointer. Safe to call more than once; a
/// second call with the counter already running is a no-op.
pub fn init(rsdp: u64) -> bool {
    if BASE.load(Ordering::Relaxed) != 0 {
        return true;
    }
    let Some(phys) = crate::acpi::hpet_from_rsdp(rsdp) else {
        return false;
    };
    let virt = crate::mm::map_mmio(phys, 0x1000);
    BASE.store(virt, Ordering::Relaxed);
    let cap = r64(CAP_ID);
    let period = cap >> 32;
    // A zero or absurd period means we are not looking at an HPET; refuse rather
    // than divide by it later.
    if period == 0 || period > 100_000_000 {
        crate::ktrace::log_fmt(format_args!("hpet: implausible period {period} fs -- ignoring"));
        BASE.store(0, Ordering::Relaxed);
        return false;
    }
    PERIOD_FS.store(period, Ordering::Relaxed);
    // Start the main counter if firmware left it stopped.
    w64(GEN_CONF, r64(GEN_CONF) | ENABLE_CNF);
    // **Liveness probe.** A present, enabled HPET whose counter does not actually
    // advance would make every `delay_us` spin forever — and `delay_us` is the
    // calibration reference for the APIC timer, so the boot would hang before the
    // first test ran. (It did.) Prove movement here, once, and refuse the device
    // otherwise rather than trusting the table.
    let t0 = r64(MAIN_COUNTER);
    let mut moved = false;
    for _ in 0..1_000_000 {
        if r64(MAIN_COUNTER) != t0 {
            moved = true;
            break;
        }
        core::hint::spin_loop();
    }
    if !moved {
        crate::ktrace::log_fmt(format_args!(
            "hpet: counter at {phys:#x} never advanced -- unusable, ignoring"
        ));
        BASE.store(0, Ordering::Relaxed);
        PERIOD_FS.store(0, Ordering::Relaxed);
        return false;
    }
    crate::ktrace::log_fmt(format_args!(
        "hpet: up at {phys:#x}, period {period} fs ({} MHz)",
        if period > 0 { 1_000_000_000 / period } else { 0 }
    ));
    true
}

/// Whether a usable HPET counter is running.
pub fn present() -> bool {
    BASE.load(Ordering::Relaxed) != 0
}

/// The raw main-counter value.
pub fn counter() -> u64 {
    if !present() {
        return 0;
    }
    r64(MAIN_COUNTER)
}

/// Counter ticks per microsecond (at least 1, so callers can never divide by
/// zero even on an implausibly slow counter).
pub fn ticks_per_us() -> u64 {
    let p = PERIOD_FS.load(Ordering::Relaxed);
    if p == 0 {
        return 0;
    }
    (FS_PER_US / p).max(1)
}

/// Busy-wait `us` microseconds on the main counter. Returns false if there is no
/// usable HPET or the wait did not complete, so the caller can fall back to
/// another delay source instead of assuming the delay happened.
///
/// **The spin is bounded.** An unbounded version of this hung the boot outright:
/// it is the calibration reference for the APIC timer, so a counter that stops
/// advancing (or a device that was never really an HPET) freezes the kernel before
/// anything else runs. The bound is generous — several orders of magnitude more
/// iterations than the wait should need — so it only trips on genuinely broken
/// hardware, never on a slow one.
pub fn delay_us(us: u64) -> bool {
    if !present() {
        return false;
    }
    let per_us = ticks_per_us();
    if per_us == 0 {
        return false;
    }
    let start = counter();
    let target = us.saturating_mul(per_us);
    // Budget: assume the worst plausible case of one counter tick per ~1000 spin
    // iterations, plus a floor so a tiny wait still gets a fair number of tries.
    let budget = target.saturating_mul(1_000).saturating_add(10_000_000);
    let mut spins = 0u64;
    // The counter is 64-bit and monotonic, so wrapping is not a practical concern
    // (at 100 MHz it wraps in ~5800 years), but use wrapping_sub anyway so a
    // 32-bit-counter platform degrades to a short wait rather than an endless one.
    while counter().wrapping_sub(start) < target {
        spins += 1;
        if spins > budget {
            crate::ktrace::log_fmt(format_args!(
                "hpet: delay_us({us}) gave up after {spins} spins -- counter stalled"
            ));
            return false;
        }
        core::hint::spin_loop();
    }
    true
}
