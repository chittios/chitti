//! Hardware **drivers** that sit below the stack facades.
//!
//! Net/storage/sound keep their subsystem roots (`crate::net`, `crate::block`,
//! …); device bring-up that is larger than a single bus glue file lives here
//! so the tree stays navigable. Today:
//!
//! - [`i2c`] — Synopsys DesignWare I2C master as Intel ships it in LPSS; the
//!   controller a modern laptop's HID-over-I2C touchpad hangs off
//! - [`ec`] — the ACPI embedded controller, which owns a laptop's battery,
//!   lid and thermal state
//! - [`wifi`] — Broadcom FullMAC PCIe (brcmfmac-class) on Apple Silicon

pub mod ec;
pub mod i2c;
pub mod i2c_hid;
pub mod wifi;
