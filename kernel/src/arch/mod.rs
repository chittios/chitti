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
/// [`crate::mouse`]. aarch64: virtio pointer + USB (xHCI/HID). x86: USB
/// (xHCI/HID). Cheap; called from the UI idle loops via `mouse::tick`.
#[cfg(target_arch = "aarch64")]
pub fn mouse_poll() {
    aarch64::virtio_pointer::poll();
    aarch64::xhci::poll_mouse();
}

#[cfg(target_arch = "x86_64")]
pub fn mouse_poll() {
    x86_64::xhci::poll_mouse();
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn mouse_poll() {}
