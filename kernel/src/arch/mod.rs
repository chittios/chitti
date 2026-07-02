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
