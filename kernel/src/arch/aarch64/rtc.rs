//! ARM PL031 real-time clock. On the QEMU `virt` machine (and many ARM boards,
//! incl. hypervisors) it sits at `0x0901_0000`, inside the low 1 GiB Device
//! block the MMU identity-maps, so the read is safe without extra mapping. The
//! `DR` register (offset 0) reads the current time directly as Unix seconds.
//! Read once at boot to seed the wall clock ([`crate::clock`]).

use core::ptr::read_volatile;

/// PL031 base on QEMU `virt`. (Real hardware with a different base simply reads
/// an implausible value, which the caller rejects, falling back to `/datetime`.)
const PL031_BASE: usize = 0x0901_0000;
const RTC_DR: usize = 0x00; // data register: seconds since the Unix epoch

/// Read the PL031 data register as a Unix timestamp. `None` if the value is
/// implausible (0, or beyond year ~2100 → the device probably isn't a PL031).
pub fn read_unix() -> Option<u64> {
    // SAFETY: `PL031_BASE` is in the identity-mapped low Device region; a 32-bit
    // volatile read of a memory-mapped register has no side effects.
    let secs = unsafe { read_volatile((PL031_BASE + RTC_DR) as *const u32) } as u64;
    // 1_600_000_000 ≈ 2020-09; 4_102_444_800 ≈ 2100-01. A sane current time.
    if (1_600_000_000..4_102_444_800).contains(&secs) {
        Some(secs)
    } else {
        None
    }
}
