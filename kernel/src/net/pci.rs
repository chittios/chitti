//! PCI network-card discovery + driver dispatch for the net stack.
//!
//! Every Ethernet function on the bus is examined and classified by
//! **vendor/device ID** ([`super::nic_ids::driver_for`]), not by vendor alone.
//! The first function we have a driver for is brought up and handed to smoltcp
//! as a [`NetDevice`].
//!
//! Two real-hardware failure modes this shape exists to avoid:
//!
//! * **Claiming a NIC we can't drive.** Matching `vendor == 0x8086` and running
//!   the legacy-e1000 init would take an I210/I225 (rings at `0xC000`, advanced
//!   descriptors) and configure registers that aren't there — the NIC appears to
//!   come up and never receives a frame. Worse, it consumed the one NIC slot, so
//!   a second working card is never tried.
//! * **Stopping at the first Ethernet function.** A laptop with both an Intel
//!   AMT NIC and a Realtek card, or a dock, exposes several; scanning past the
//!   ones with no driver finds the one that works.
//!
//! Dual-arch: the PCI config surface is `crate::arch::x86_64::pci` (legacy I/O
//! ports) on x86 and `crate::pci` (ECAM) on aarch64 — identical
//! `for_each`/`class_of`/`PciDevice` API, so this logic is arch-neutral.

use super::nic_ids::{driver_for, is_intel_guess, NicKind};
use super::NetDevice;
use alloc::boxed::Box;

#[cfg(target_arch = "aarch64")]
use crate::pci;
#[cfg(target_arch = "x86_64")]
use crate::arch::x86_64::pci;

/// PCI base class for a network controller; subclass 0x00 is Ethernet.
const CLASS_NETWORK: u8 = 0x02;
const SUBCLASS_ETHERNET: u8 = 0x00;

/// Probe PCI for a supported NIC and bring it up. `None` if none is present.
///
/// Matches on base class + subclass and **ignores prog_if** — it is
/// architecturally reserved for Ethernet controllers, and a card that reports
/// something other than 0 there would be skipped by a full `(0x02, 0x00, 0x00)`
/// triple match.
pub fn probe() -> Option<Box<dyn NetDevice>> {
    let mut candidates = 0usize;
    let mut chosen: Option<(pci::PciDevice, NicKind)> = None;
    pci::for_each(&mut |d| {
        let (base, sub, _) = pci::class_of(&d);
        if base != CLASS_NETWORK || sub != SUBCLASS_ETHERNET {
            return true;
        }
        candidates += 1;
        match driver_for(d.vendor, d.device) {
            Some(kind) => {
                if is_intel_guess(d.vendor, d.device) {
                    crate::ktrace::log_fmt(format_args!(
                        "net: Intel Ethernet {:04x}:{:04x} is in no known ID table -- assuming {} (report this ID if the link never comes up)",
                        d.vendor,
                        d.device,
                        kind.name()
                    ));
                }
                chosen = Some((d, kind));
                false // stop at the first one we can drive
            }
            None => {
                crate::ktrace::log_fmt(format_args!(
                    "net: Ethernet {:04x}:{:04x} at {}:{}.{} has no driver -- skipping",
                    d.vendor, d.device, d.bus, d.dev, d.func
                ));
                true // keep looking
            }
        }
    });
    let Some((d, kind)) = chosen else {
        if candidates > 0 {
            crate::ktrace::log_fmt(format_args!(
                "net: {candidates} Ethernet controller(s) present but none supported -- see /lspci"
            ));
        }
        return None;
    };
    crate::ktrace::log_fmt(format_args!(
        "net: {:04x}:{:04x} at {}:{}.{} -> {} driver",
        d.vendor,
        d.device,
        d.bus,
        d.dev,
        d.func,
        kind.name()
    ));
    match kind {
        // The legacy and PCIe Intel gigabit families share one driver; it selects
        // its bring-up variant from `kind` (the ring registers are at the same
        // offsets, the init sequence differs).
        NicKind::E1000 | NicKind::E1000e => {
            super::e1000::E1000::init(d, kind).map(|n| Box::new(n) as Box<dyn NetDevice>)
        }
        // 82575-and-later moved the rings to 0xC000/0xE000 and switched to
        // advanced descriptors; igb and igc share that shape.
        NicKind::Igb | NicKind::Igc => {
            super::igb::Igb::init(d, kind).map(|n| Box::new(n) as Box<dyn NetDevice>)
        }
        // Realtek gigabit — the common consumer NIC. Unverified on hardware (QEMU
        // has no r8169-family model); see the driver's module docs.
        NicKind::R8169 => {
            super::r8169::Rtl8169::init(d).map(|n| Box::new(n) as Box<dyn NetDevice>)
        }
        NicKind::VirtioNet => {
            super::virtio_net_pci::VirtioNetPci::init(d).map(|n| Box::new(n) as Box<dyn NetDevice>)
        }
        // Recognised, driver not written yet. Logged explicitly so this reads as
        // "unimplemented", never as a NIC that came up and stayed silent.
        other => {
            crate::ktrace::log_fmt(format_args!(
                "net: {} driver not implemented yet ({:04x}:{:04x})",
                other.name(),
                d.vendor,
                d.device
            ));
            None
        }
    }
}
