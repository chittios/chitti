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
//! [`proto`] then adds the message layer — command out, notification in — and [`device`]
//! the queue that carries it: a command is placed in a slot, split across the two transfer
//! buffers the device fetches it with, and the doorbell rung; the reply is matched by the
//! sequence firmware echoes. Bring-up ends by reading the device's MAC address out of its
//! own registers and sending one **read-only** command (`NVM_GET_INFO`), because that is the
//! strongest claim that can be checked: firmware answered a question.
//!
//! ## What still does not exist, and why not
//!
//! **The configuration commands, so it cannot scan or associate.** Not for want of
//! plumbing — the transport above would carry them. A scan request, a MAC context and a
//! station key are large structures whose layouts vary **per firmware API version**, and no
//! emulator provides an Intel WiFi part to check one against. So the groundwork for adding
//! them safely is here instead: [`fw::parse_cmd_versions`] reads the image's own
//! command-version table (`IWL_UCODE_TLV_CMD_VERSIONS`), which is how the firmware states
//! which layout it expects. A command added later must consult it and **refuse** an
//! unimplemented version rather than send a plausible structure the firmware reads as
//! something else — there is no error path from the device for that.
//!
//! One correction worth recording, because it is the shape of the mistake this whole posture
//! exists to avoid: the NVM response's general section was first written as four `u32`s. It
//! is `u32, u16, u8, u8`. Read as words, `n_hw_addrs` lands on the transmit chain mask —
//! a small, plausible number that passes every sanity check — so the bug reports a
//! confident wrong answer. It was caught by reading Linux's `fw/api/nvm-reg.h` rather than
//! trusting recall, and every layout here now comes from those headers.
//!
//! The WPA2 and 802.11 layers those commands would feed **do** exist and are tested:
//! [`super::wpa`] and [`super::ieee80211`].
//!
//! ## Both arches, one driver
//!
//! Nothing here is gated on `x86_64`. It was, and that was wrong: an Intel WiFi card in an
//! ARM machine's PCIe slot is an ordinary endpoint, and the only thing that actually differed
//! was the *import* — the PCI config surface is `crate::arch::x86_64::pci` (I/O ports) on x86
//! and `crate::pci` (ECAM) on aarch64, with the same `for_each`/`class_of`/`PciDevice` API,
//! exactly as `net::pci` does it. So there is one cfg pair at the top of this module and one
//! in [`device`], and no behaviour behind either. A machine with no PCI at all (aarch64
//! `-kernel`, where ECAM comes from the stub's ACPI) simply finds nothing: `pci::for_each`
//! returns immediately when no ECAM base was published.

pub mod context;
pub mod csr;
pub mod device;
pub mod fw;
pub mod rx;
pub mod scan;
pub mod sta;
pub mod proto;

use alloc::string::String;

// The PCI config surface, same API either side: legacy I/O ports on x86, ECAM on aarch64.
// One cfg pair here and nothing below it is arch-gated — an Intel WiFi card in an ARM
// machine's PCIe slot is an ordinary endpoint, and gating the whole driver on x86 would
// make it invisible there for no reason but the import.
#[cfg(target_arch = "aarch64")]
use crate::pci;
#[cfg(target_arch = "x86_64")]
use crate::arch::x86_64::pci;

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
pub fn probe() -> Option<Found> {
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

/// Whether a PCI function looks like a wireless controller (class 02, subclass 80).
fn is_wifi_class(d: &pci::PciDevice) -> bool {
    let class = pci::class_of(d);
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
pub fn bring_up() -> Result<String, String> {
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
    if let Err(e) = dev.wait_for_alive() {
        return Err(alloc::format!(
            "{} reset and {} handed over at {phys:#x}, but {e}",
            found.family.label(),
            name
        ));
    }

    let mut report = alloc::format!(
        "{} reset, {} ({}) loaded at {phys:#x} -- firmware is alive",
        found.family.label(),
        name,
        image.version
    );

    // The device's own MAC address, straight out of its registers. Worth reading here
    // because it is the one fact this driver can *check*: a plausible unicast address means
    // the register window is right, and an absent one means it is not — a distinction that
    // otherwise only shows up much later as traffic nobody answers.
    match dev.read_mac() {
        Some(m) => report.push_str(&alloc::format!(
            "; mac {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
            m[0], m[1], m[2], m[3], m[4], m[5]
        )),
        None => report.push_str("; no MAC address in the strap or OTP registers"),
    }

    // One read-only command, to see whether the round-trip works at all. Read-only on
    // purpose: this transport has never run against a real device, and a first command that
    // configured something would misconfigure a radio if any part of it is wrong.
    match dev.nvm_info() {
        Ok(n) => {
            report.push_str(&alloc::format!(
                "; NVM v{:#x} board {:#x}, {} hw address(es)",
                n.nvm_version, n.board_type, n.n_hw_addrs
            ));
            if let Some((tx, rx)) = n.chains {
                report.push_str(&alloc::format!(" chains tx {tx:#x} rx {rx:#x}"));
            }
            report.push_str(" -- the command round-trip works");
        }
        Err(e) => report.push_str(&alloc::format!("; the first command got no usable reply ({e})")),
    }
    // How many commands this firmware declares versions for. Not cosmetic: it is the table
    // any future configuration command has to consult, and its presence is what would make
    // adding one safe rather than a guess.
    match image.cmd_versions(&bytes) {
        Some(v) => report.push_str(&alloc::format!(
            ". Firmware declares versions for {} commands",
            v.len()
        )),
        None => report.push_str(". Firmware declares no command-version table"),
    }
    report.push_str("; scan and associate need its configuration commands, which are not implemented.");
    Ok(report)
}

/// Whether this firmware's `SCAN_REQ_UMAC` layout is one this driver implements.
///
/// The command id is stable; **the request structure is not**. It has been
/// rewritten several times (adaptive dwell, a v8 rewrite, per-band channel
/// configuration), so the same command takes a different layout on different
/// firmware, and the image itself says which it expects via the
/// `IWL_UCODE_TLV_CMD_VERSIONS` table.
///
/// Returns `Err` with the version named when the firmware speaks a layout this
/// driver has not been written against — which is currently every version, and
/// is why `/wifi scan` reports that rather than transmitting.
///
/// This is the discipline the rest of the module already follows: a struct
/// recalled rather than read produced a confident wrong answer once here
/// (`NVM_GET_INFO`'s general section), and a scan request is far larger. A
/// refusal that names the version is actionable; a well-formed guess sent to a
/// real radio is not, because the radio accepts it, reads our fields as
/// different ones, and reports nothing.
pub fn scan_supported(fw: &fw::FirmwareImage, blob: &[u8]) -> Result<u8, ScanUnsupported> {
    match fw.cmd_version(blob, proto::GROUP_LONG, proto::SCAN_REQ_UMAC) {
        Some(v) if proto::SCAN_REQ_UMAC_VERSIONS.contains(&v.cmd_ver) => Ok(v.cmd_ver),
        Some(v) => Err(ScanUnsupported::Version(v.cmd_ver)),
        // Silence in the table is **not** version 0: older images omit the
        // table entirely, and assuming a layout because nothing contradicted it
        // is exactly the guess this function exists to prevent.
        None => Err(ScanUnsupported::Unstated),
    }
}

/// Why a scan cannot be started.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanUnsupported {
    /// The firmware named a request version this driver does not implement.
    Version(u8),
    /// The firmware's command-version table is absent or silent about scan.
    Unstated,
}

impl ScanUnsupported {
    pub fn message(self) -> alloc::string::String {
        match self {
            ScanUnsupported::Version(v) => alloc::format!(
                "this firmware speaks SCAN_REQ_UMAC v{v}, which this driver does not implement \
                 (the request layout differs per version; it must be taken from Linux's \
                 fw/api/scan.h for a matching release and verified on the hardware)"
            ),
            ScanUnsupported::Unstated => alloc::string::String::from(
                "this firmware does not state its SCAN_REQ_UMAC version, so no layout can be \
                 assumed -- an unstated version is not version 0",
            ),
        }
    }
}

#[cfg(test)]
mod scan_gate_tests {
    use super::*;

    /// **An unstated version is not version 0.** Older images omit the
    /// command-version table entirely, and assuming a layout because nothing
    /// contradicted it is exactly the guess this gate exists to prevent — a
    /// well-formed scan request at the wrong version is accepted by the radio,
    /// read as different fields, and reported by nothing.
    #[test_case]
    fn an_unstated_scan_version_is_refused_rather_than_assumed() {
        let e = ScanUnsupported::Unstated;
        assert!(e.message().contains("not version 0"));
        let v = ScanUnsupported::Version(17);
        assert!(v.message().contains("v17"), "the version must be named: {}", v.message());
        assert!(v.message().contains("fw/api/scan.h"), "and where to get the layout");
    }

    /// The gate is driven by the firmware's own table, so a version that *is*
    /// implemented passes and its neighbours do not. Pinned as a table rather
    /// than a behaviour so adding a layout to `SCAN_REQ_UMAC_VERSIONS` without
    /// adding the struct is caught here.
    /// **Every listed version must have a builder.** Listing a version without
    /// its struct is the failure this gate exists to prevent, wearing a
    /// different hat: the refusal would pass and a wrong layout would go out.
    #[test_case]
    fn every_listed_version_has_an_implemented_layout() {
        for &v in proto::SCAN_REQ_UMAC_VERSIONS {
            assert_eq!(v, scan::VERSION, "v{v} is listed but only v{} is built", scan::VERSION);
        }
        assert!(
            proto::SCAN_REQ_UMAC_VERSIONS.contains(&scan::VERSION),
            "the implemented version must be listed, or it is never used"
        );
        // Anything else is still refused.
        for v in 0u8..=20 {
            if v != scan::VERSION {
                assert!(!proto::SCAN_REQ_UMAC_VERSIONS.contains(&v), "v{v}");
            }
        }
    }
}
