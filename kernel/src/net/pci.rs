//! PCI network cards for the net stack: **e1000** (Intel PRO/1000 — VirtualBox's
//! default adapter and many real machines) and **virtio-net-pci** (QEMU/VBox
//! paravirtual). Discovered over the kernel PCI subsystem; the first match is
//! handed to smoltcp as a [`NetDevice`].
//!
//! Dual-arch: the PCI config surface is `crate::arch::x86_64::pci` (legacy I/O
//! ports) on x86 and `crate::pci` (ECAM) on aarch64 — identical
//! `find_class`/`PciDevice` API, so the discovery logic here is arch-neutral.

use super::NetDevice;
use alloc::boxed::Box;

#[cfg(target_arch = "aarch64")]
use crate::pci;
#[cfg(target_arch = "x86_64")]
use crate::arch::x86_64::pci;

// PCI class for an Ethernet controller: base 0x02 (network), sub 0x00, prog-if 0x00.
const CLASS_ETHERNET: (u8, u8, u8) = (0x02, 0x00, 0x00);

const VENDOR_INTEL: u16 = 0x8086;
const VENDOR_VIRTIO: u16 = 0x1af4;

/// Probe PCI for a supported NIC and bring it up. `None` if none is present.
pub fn probe() -> Option<Box<dyn NetDevice>> {
    let d = pci::find_class(CLASS_ETHERNET.0, CLASS_ETHERNET.1, CLASS_ETHERNET.2)?;
    match d.vendor {
        // Intel e1000 family (82540/82545/e1000e, and VBox's Intel adapters).
        VENDOR_INTEL => super::e1000::E1000::init(d).map(|n| Box::new(n) as Box<dyn NetDevice>),
        // virtio-net-pci — modern/paravirtual (QEMU -device virtio-net-pci, VBox virtio).
        VENDOR_VIRTIO => super::virtio_net_pci::VirtioNetPci::init(d).map(|n| Box::new(n) as Box<dyn NetDevice>),
        _ => None,
    }
}
