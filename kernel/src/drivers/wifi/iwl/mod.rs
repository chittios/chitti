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
//! Since then, [`csr`] carries the register map and its pure predicates, [`context`] the
//! gen2 **context info** layout (how an AX200-and-later device is told where firmware is,
//! so its own loader fetches it), and [`device`] the sequences: map BAR0, prepare the
//! card, APM init, software reset, grab MAC access, then build the context info and hand
//! it over.
//!
//! What still does not exist: the receive path — so firmware's *alive* notification
//! cannot be observed — the command round-trip, and then the 802.11 state machine and
//! WPA2. So the radio does not associate and `/wifi connect` still cannot work. Shipping
//! a half-built version of that would put a connect in the shell that fails silently;
//! `/wifi` reports how far bring-up got instead.

pub mod context;
pub mod csr;
#[cfg(target_arch = "x86_64")]
pub mod device;
pub mod fw;
pub mod proto;

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

/// Where a fetched `.ucode` is looked for at runtime.
///
/// A path in the store rather than an `include_bytes!` because the filename carries a
/// firmware API version that varies per release — the Broadcom driver can embed one fixed
/// name, this cannot. `cargo xtask iwlwifi-assets` fetches into `assets/wifi/iwl/`; put
/// the file here on the installed system.
pub const FW_DIR: &str = "/wifi/iwl";

/// Read the newest firmware image present for `family`.
///
/// Candidates are tried newest-API first, which is what Linux does and the reason the API
/// number is enumerated rather than computed: it belongs to the file, not the chip.
/// Returns the bytes and the name they came from.
#[cfg(target_arch = "x86_64")]
fn find_firmware(family: fw::Family) -> Option<(String, alloc::vec::Vec<u8>)> {
    for name in fw::firmware_candidates(family) {
        let path = alloc::format!("{FW_DIR}/{name}");
        if let Some(bytes) = crate::synapse::fs::read(&path) {
            return Some((name, bytes));
        }
    }
    None
}

/// Bring the radio as far as this driver goes: reset it and hand it firmware.
///
/// Deliberately **command-driven, never automatic at boot**. The same posture the AGX
/// coprocessor and the Broadcom radio take: an untested driver does not touch a device
/// because the machine happened to start. `/wifi up` asks for it.
///
/// Ends after firmware is handed over, which is honestly short of working: without a
/// receive path the device's *alive* notification cannot be seen, so "handed over" is the
/// strongest claim available and the log says so.
#[cfg(target_arch = "x86_64")]
pub fn bring_up() -> Result<String, String> {
    use crate::arch::x86_64::pci;

    let found = probe().ok_or_else(|| String::from("no recognised Intel WiFi device"))?;
    // Re-find the function so the device handle is fresh rather than cached from a probe
    // that may have happened long ago.
    let mut target = None;
    pci::for_each(&mut |d| {
        if d.vendor == fw::VENDOR_INTEL && d.device == found.device_id {
            target = Some(d);
            return false;
        }
        true
    });
    let d = target.ok_or_else(|| String::from("the device disappeared between probe and open"))?;

    let (name, bytes) = find_firmware(found.family).ok_or_else(|| {
        alloc::format!(
            "no firmware in {FW_DIR}/ -- wanted {} (fetch with `cargo xtask iwlwifi-assets`)",
            fw::firmware_candidates(found.family)
                .first()
                .map(|s| s.as_str())
                .unwrap_or("?")
        )
    })?;
    let image = fw::parse_image(&bytes)
        .ok_or_else(|| alloc::format!("{name} is not a valid TLV firmware image"))?;
    if !image.has_runtime_section() {
        return Err(alloc::format!("{name} carries no runtime section"));
    }

    let mut dev = device::IwlDevice::open(d, found.family)
        .ok_or_else(|| String::from("bring-up failed -- see the iwlwifi: ktrace lines"))?;
    let phys = dev.load_firmware(&image, &bytes).map_err(String::from)?;
    // The load is only believable if firmware answers. Before the receive ring existed
    // this was where bring-up stopped and "handed over" was the strongest claim available.
    match dev.wait_for_alive() {
        Ok(()) => Ok(alloc::format!(
            "{} reset, {} ({}) loaded at {phys:#x} -- firmware is alive. No command queue or \
             802.11 yet, so it cannot associate.",
            found.family.label(),
            name,
            image.version
        )),
        Err(e) => Err(alloc::format!(
            "{} reset and {} handed over at {phys:#x}, but {e}",
            found.family.label(),
            name
        )),
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn bring_up() -> Result<String, String> {
    Err(String::from("Intel WiFi is a PCIe part found in x86 machines"))
}
