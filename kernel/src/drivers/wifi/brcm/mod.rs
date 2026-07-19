//! Broadcom **FullMAC PCIe** (brcmfmac-class) for Apple Silicon.
//!
//! Pure wire/geometry helpers live in [`proto`] (always built; unit-tested on
//! x86). The live PCIe/dongle path is [`device`] (aarch64, non-test only).

pub mod proto;

#[cfg(all(target_arch = "aarch64", not(test)))]
mod device;

#[cfg(all(target_arch = "aarch64", not(test)))]
pub use device::*;
