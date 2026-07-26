//! **Intel WiFi (`iwlwifi`-class)** — identification and firmware, staged.
//!
//! The single most common WiFi part in x86 laptops, and the reason `/wifi` has nothing
//! to offer on one today. Structured like the Broadcom driver next door: [`fw`] is the
//! pure, unit-tested half (which chip is this, which firmware image does it need, is that
//! image well-formed), and the device half follows.
//!
//! ## Why the pure half comes first, and alone
//!
//! An Intel radio does nothing until firmware is loaded into it, and the image is chosen
//! by chip family plus an API version that is a property of the *file*, not the hardware.
//! Get that wrong and the device rejects the image against its own signature — with no
//! error the host can read. So identification and image validation are worth having
//! standalone, and they are the only part testable at all: **QEMU emulates no Intel WiFi
//! device**, so unlike the Ethernet families there is not even an emulated part to try.
//!
//! ## Scope, stated plainly
//!
//! What exists: family identification from the PCI id, the firmware filename search
//! order, and a `.ucode` TLV parser that refuses malformed or pre-TLV images.
//!
//! What does not: everything that makes a radio work — the device's register interface,
//! NIC reset and power-up, the command and receive rings, the firmware load handshake,
//! then scan/associate and WPA2. That is a large job, and shipping a half-built version
//! of it would put a `/wifi connect` in the shell that cannot connect. `/wifi` reports
//! what was found and what firmware it would need instead.

pub mod fw;

use alloc::string::String;

/// What was found on the bus, for `/wifi` to report.
#[derive(Debug, Clone)]
pub struct Found {
    pub family: fw::Family,
    pub device_id: u16,
    /// The firmware filenames that would be tried, newest first.
    pub firmware: alloc::vec::Vec<String>,
}

/// Look for an Intel WiFi device on PCI.
///
/// Read-only: it reads configuration space and nothing else. An unrecognised Intel WiFi
/// id yields `None` with a log naming it, rather than being driven as a nearby family —
/// the same rule the Ethernet dispatcher follows, for a worse failure mode.
#[cfg(target_arch = "x86_64")]
pub fn probe() -> Option<Found> {
    use crate::arch::x86_64::pci;
    let mut found: Option<Found> = None;
    // Network controller, "other" subclass (02:80) is where WiFi lives. Walk every
    // function rather than taking the first match, so an unrecognised part still gets
    // named in the log instead of being invisible.
    pci::for_each(&mut |d| {
        if d.vendor != fw::VENDOR_INTEL {
            return true; // keep going
        }
        match fw::family_for(d.vendor, d.device) {
            Some(family) => {
                let firmware = fw::firmware_candidates(family);
                crate::ktrace::log_fmt(format_args!(
                    "iwlwifi: {} at {:02x}:{:02x}.{} (device {:#06x}); needs {} (newest of {} candidates)",
                    family.label(),
                    d.bus,
                    d.dev,
                    d.func,
                    d.device,
                    firmware.first().map(|s| s.as_str()).unwrap_or("?"),
                    firmware.len()
                ));
                crate::ktrace::log(
                    "iwlwifi",
                    "identification only -- no register interface or firmware load yet",
                );
                found = Some(Found {
                    family,
                    device_id: d.device,
                    firmware,
                });
                return false; // stop the walk
            }
            None => {
                // Only worth a line for something that looks like WiFi; Intel makes a
                // great many other PCI functions.
                if is_wifi_class(&d) {
                    crate::ktrace::log_fmt(format_args!(
                        "iwlwifi: unrecognised Intel WiFi device {:#06x} -- not claimed (adding it blind would load the wrong firmware)",
                        d.device
                    ));
                }
            }
        }
        true
    });
    found
}

#[cfg(not(target_arch = "x86_64"))]
pub fn probe() -> Option<Found> {
    // Intel WiFi is a PCIe part found in x86 laptops. An aarch64 machine with one would
    // work through the same code, but there is no such machine to test against and the
    // PCI facade differs, so this is honest rather than speculative.
    None
}

/// Whether a PCI function looks like a wireless controller (class 02, subclass 80).
#[cfg(target_arch = "x86_64")]
fn is_wifi_class(d: &crate::arch::x86_64::pci::PciDevice) -> bool {
    let class = crate::arch::x86_64::pci::class_of(d);
    class.0 == 0x02 && class.1 == 0x80
}
