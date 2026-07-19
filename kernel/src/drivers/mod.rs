//! Hardware **drivers** that sit below the stack facades.
//!
//! Net/storage/sound keep their subsystem roots (`crate::net`, `crate::block`,
//! …); device bring-up that is larger than a single bus glue file lives here
//! so the tree stays navigable. Today:
//!
//! - [`wifi`] — Broadcom FullMAC PCIe (brcmfmac-class) on Apple Silicon

pub mod wifi;
