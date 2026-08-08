//! Hardware **drivers** that sit below the stack facades.
//!
//! Net/storage/sound keep their subsystem roots (`crate::net`, `crate::block`,
//! …); device bring-up that is larger than a single bus glue file lives here
//! so the tree stays navigable. Today:
//!
//! - [`pwrbtn`] — the ACPI fixed-feature power button, so a press shuts the
//!   machine down instead of doing nothing
//! - [`i2c`] — Synopsys DesignWare I2C master as Intel ships it in LPSS; the
//!   controller a modern laptop's HID-over-I2C touchpad hangs off
//! - [`battery`] — the ACPI control-method battery, read by evaluating the
//!   firmware's own `_BST`/`_BIF` through [`crate::aml`] and [`ec`]
//! - [`ec`] — the ACPI embedded controller, which owns a laptop's battery,
//!   lid and thermal state
//! - [`wifi`] — Broadcom FullMAC PCIe (brcmfmac-class) on Apple Silicon
//! - [`bluetooth`] — USB Bluetooth identify + pure HCI codec (transport later)
//! - [`uvc`] — USB Video Class descriptor parse + identify (isoc capture later)
//! - [`virtio`] — one split-virtqueue + transport (mmio/PCI) shared by the
//!   host-integration devices below, so each binds on both arches
//! - [`virtio_9p`] — the 9P2000.L client behind a host shared folder
//! - [`virtio_serial`] — the multiport serial transport the clipboard agent
//!   rides on

pub mod battery;
pub mod bluetooth;
pub mod ec;
pub mod i2c;
pub mod i2c_hid;
pub mod pwrbtn;
pub mod uvc;
pub mod vbox;
pub mod virtio;
pub mod virtio_9p;
pub mod virtio_serial;
pub mod wifi;
