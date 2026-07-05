//! Architecture-specific code lives under `arch/<name>/`, and the rest of the
//! kernel reaches it only through the small **facade** re-exported here --
//! `arch::interrupts` and `arch::hlt` -- never `arch::x86_64::...` directly.
//! That is what lets the arch-independent layers (mm, sched, ktrace, cortex,
//! synapse, persona, ...) compile unchanged on either target; each supported
//! architecture provides the same facade surface.
//!
//! `x86_64` is the mature port (Limine boot, the full OS). `aarch64` is the
//! native Apple-Silicon port (QEMU + HVF), brought up incrementally; the two
//! are being collapsed into this single dual-arch tree.

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

#[cfg(target_arch = "x86_64")]
pub use x86_64::{hlt, interrupts, poweroff};

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

#[cfg(target_arch = "aarch64")]
pub use aarch64::{hlt, interrupts, poweroff};

/// Milliseconds since boot -- the PIT tick counter on x86, the generic timer
/// on aarch64. Used for inference throughput timing.
#[cfg(target_arch = "x86_64")]
pub fn now_ms() -> u64 {
    x86_64::pit::ticks()
}

#[cfg(target_arch = "aarch64")]
pub fn now_ms() -> u64 {
    aarch64::time_ms()
}

/// Current wall-clock time as a Unix timestamp read from the hardware RTC, or
/// `None` if no RTC is readable (the wall clock then falls back to a default
/// until `/datetime` sets it). CMOS on x86, PL031 on aarch64.
#[cfg(target_arch = "x86_64")]
pub fn rtc_unix() -> Option<u64> {
    x86_64::rtc::read_unix()
}

#[cfg(target_arch = "aarch64")]
pub fn rtc_unix() -> Option<u64> {
    aarch64::rtc::read_unix()
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn rtc_unix() -> Option<u64> {
    None
}

/// Poll every present mouse transport, feeding motion/buttons into
/// [`crate::mouse`]. aarch64: virtio pointer + PL050 PS/2 + USB (xHCI/HID).
/// x86: i8042 PS/2 aux + USB (xHCI/HID). Cheap; called from the UI idle
/// loops via `mouse::tick`.
#[cfg(target_arch = "aarch64")]
pub fn mouse_poll() {
    aarch64::virtio_pointer::poll();
    aarch64::pl050_mouse::poll();
    aarch64::xhci::poll_mouse();
}

#[cfg(target_arch = "x86_64")]
pub fn mouse_poll() {
    x86_64::i8042::poll_mouse();
    x86_64::xhci::poll_mouse();
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn mouse_poll() {}

/// A best-effort hardware entropy word, for seeding the CSPRNG (TLS handshake
/// keys). x86: `RDRAND` when the CPU reports it, else 0. aarch64: `RNDR`
/// (FEAT_RNG) when present, else 0. `net::tls::seed_rng` mixes several of these
/// with the cycle counter, so a 0 (facility absent — QEMU/HVF often lack both)
/// degrades to counter-jitter entropy rather than failing. Not audited crypto
/// entropy; adequate for a research OS talking to a model server over the LAN.
#[cfg(target_arch = "x86_64")]
pub fn hw_rand() -> u64 {
    x86_64::hw_rand()
}
#[cfg(target_arch = "aarch64")]
pub fn hw_rand() -> u64 {
    aarch64::hw_rand()
}
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn hw_rand() -> u64 {
    0
}

/// A monotonically-advancing cycle/tick counter for entropy mixing (finer than
/// `now_ms`): the TSC on x86, `CNTVCT_EL0` on aarch64.
#[cfg(target_arch = "x86_64")]
pub fn cycle_count() -> u64 {
    x86_64::cycle_count()
}
#[cfg(target_arch = "aarch64")]
pub fn cycle_count() -> u64 {
    aarch64::cycle_count()
}
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn cycle_count() -> u64 {
    now_ms()
}
