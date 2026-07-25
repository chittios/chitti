//! **USB Ethernet** behind the shared [`NetDevice`] trait.
//!
//! This is the pragmatic answer to the biggest real-hardware gap left: most laptops
//! have no Ethernet port, and there is no WiFi driver on x86 at all, so a USB dongle
//! is the only way those machines get networked. It reuses the existing xHCI stack's
//! bulk transport rather than needing a new controller driver.
//!
//! ## Classes
//!
//! USB NICs split across three shapes, and only the first is a real standard:
//!
//! * **CDC-ECM** (class `0x02`/subclass `0x06` control interface + `0x0A` data) —
//!   frames go over bulk endpoints raw, one Ethernet frame per transfer, no framing
//!   header. Supported here.
//! * **ASIX AX88179 / Realtek RTL8152** — vendor-specific (class `0xFF`), each with
//!   its own register protocol and a per-packet header. [`Variant`] names them but
//!   they are **not implemented**; they need per-chip register setup and header
//!   handling, and are recognised only so a probe can say which chip it saw instead
//!   of silently ignoring it.
//! * **RNDIS** (what QEMU's `usb-net` emulates) — a Microsoft control protocol over
//!   bulk. Not implemented, and deliberately not the first target: it is the one
//!   shape almost no physical dongle uses.
//!
//! ## Verification status
//!
//! **Unverified on hardware.** QEMU emulates only RNDIS, so no emulated device
//! exercises the CDC-ECM path. The framing is the simplest part of USB networking
//! (raw frames, no header), which is exactly why CDC-ECM is the right first target,
//! but the enumeration and endpoint bring-up have never met a real dongle. Treat it
//! like `r8169`: written from the spec, logged verbosely, unproven.

use crate::net::NetDevice;

/// USB interface class codes for the shapes we can recognise.
pub const CLASS_CDC_CONTROL: u8 = 0x02;
pub const CLASS_CDC_DATA: u8 = 0x0a;
pub const CLASS_VENDOR: u8 = 0xff;
/// CDC subclass for Ethernet Networking Control Model.
pub const SUBCLASS_ECM: u8 = 0x06;

/// Which USB-Ethernet protocol a device speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    /// Raw Ethernet frames over bulk — implemented.
    CdcEcm,
    /// ASIX AX88179/AX88772 — recognised, not implemented.
    Asix,
    /// Realtek RTL8152/RTL8153 — recognised, not implemented.
    Rtl8152,
}

impl Variant {
    pub fn name(self) -> &'static str {
        match self {
            Variant::CdcEcm => "cdc-ecm",
            Variant::Asix => "asix",
            Variant::Rtl8152 => "rtl8152",
        }
    }

    /// Whether frames cross the bulk endpoints with no extra header, which is the
    /// only framing this driver implements.
    pub fn is_raw_frames(self) -> bool {
        matches!(self, Variant::CdcEcm)
    }
}

/// Identify a USB NIC from its vendor/product id and interface class.
///
/// The class check is what makes CDC-ECM recognisable across vendors — it is a
/// standard, so there is no id list to maintain. The vendor-specific chips do need
/// ids, and are matched first because they also report class `0xFF`.
pub fn identify(vendor: u16, product: u16, iface_class: u8, iface_subclass: u8) -> Option<Variant> {
    // ASIX and Realtek dongles: vendor-specific class, known ids.
    match (vendor, product) {
        // AX88179 (and the AX88178A sharing its protocol).
        (0x0b95, 0x1790) | (0x0b95, 0x178a) | (0x0b95, 0x7720) | (0x0b95, 0x772a) | (0x0b95, 0x772b) => {
            return Some(Variant::Asix)
        }
        // RTL8152/8153, including the very common Realtek-based docks.
        (0x0bda, 0x8152) | (0x0bda, 0x8153) | (0x0bda, 0x8156) => return Some(Variant::Rtl8152),
        _ => {}
    }
    // CDC-ECM is a standard: match on class, not on an id list. The data interface
    // (0x0A) carries the bulk endpoints; the control interface (0x02/0x06) names the
    // function.
    if iface_class == CLASS_CDC_DATA
        || (iface_class == CLASS_CDC_CONTROL && iface_subclass == SUBCLASS_ECM)
    {
        return Some(Variant::CdcEcm);
    }
    None
}

/// Largest Ethernet frame we transfer, including header; matches the net stack's
/// MTU accounting.
pub const FRAME_MAX: usize = 1514;

/// A USB Ethernet adapter driven through the xHCI bulk transport.
///
/// Holds no rings of its own: the endpoints live in the xHCI controller, and this is
/// the `NetDevice` face over them.
pub struct UsbEth {
    mac: [u8; 6],
    variant: Variant,
}

impl UsbEth {
    /// Wrap an already-configured bulk pair. `mac` comes from the device's CDC
    /// Ethernet functional descriptor (or its iMACAddress string), which the caller
    /// reads during enumeration.
    pub fn new(mac: [u8; 6], variant: Variant) -> Option<UsbEth> {
        if !variant.is_raw_frames() {
            crate::ktrace::log_fmt(format_args!(
                "usb_eth: {} needs per-chip register setup and packet headers -- not implemented",
                variant.name()
            ));
            return None;
        }
        crate::ktrace::log_fmt(format_args!(
            "usb_eth: {} up, MAC {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
            variant.name(),
            mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]
        ));
        Some(UsbEth { mac, variant })
    }

    pub fn variant(&self) -> Variant {
        self.variant
    }
}

impl NetDevice for UsbEth {
    fn mac(&self) -> [u8; 6] {
        self.mac
    }

    /// Collect one frame, if the bulk IN transfer has completed.
    ///
    /// CDC-ECM puts exactly one Ethernet frame per transfer with no header, so the
    /// transferred length *is* the frame length — which is why this class is the
    /// right first target.
    fn receive(&mut self, out: &mut [u8]) -> Option<usize> {
        crate::arch::usb_bulk_arm_in();
        let n = crate::arch::usb_bulk_take_in(out)?;
        if n == 0 || n > FRAME_MAX {
            // A zero-length packet is the device's way of ending a transfer, not a
            // frame; an oversized one is a bug or a lost boundary. Drop both.
            return None;
        }
        Some(n)
    }

    fn transmit(&mut self, frame: &[u8]) {
        if frame.len() > FRAME_MAX {
            return;
        }
        // Dropping when a transfer is still outstanding is legitimate for Ethernet;
        // smoltcp retransmits.
        let _ = crate::arch::usb_bulk_send(frame);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn identifies_cdc_ecm_by_class_not_by_id() {
        // CDC-ECM is a standard, so an unknown vendor must still be recognised —
        // maintaining an id list for it would guarantee gaps.
        assert_eq!(identify(0x1234, 0x5678, CLASS_CDC_DATA, 0x00), Some(Variant::CdcEcm));
        assert_eq!(
            identify(0xffff, 0x0001, CLASS_CDC_CONTROL, SUBCLASS_ECM),
            Some(Variant::CdcEcm)
        );
    }

    #[test_case]
    fn identifies_the_common_vendor_specific_dongles() {
        // These report class 0xFF, so they must be matched by id before the class
        // check — and they are the chips in most cheap dongles and docks.
        assert_eq!(identify(0x0b95, 0x1790, CLASS_VENDOR, 0), Some(Variant::Asix));
        assert_eq!(identify(0x0bda, 0x8153, CLASS_VENDOR, 0), Some(Variant::Rtl8152));
        assert_eq!(identify(0x0bda, 0x8156, CLASS_VENDOR, 0), Some(Variant::Rtl8152));
    }

    #[test_case]
    fn only_cdc_ecm_claims_raw_framing() {
        // The vendor chips add a per-packet header, so treating them as raw frames
        // would hand smoltcp garbage. `new` must refuse them outright.
        assert!(Variant::CdcEcm.is_raw_frames());
        assert!(!Variant::Asix.is_raw_frames());
        assert!(!Variant::Rtl8152.is_raw_frames());
        assert!(UsbEth::new([0; 6], Variant::Asix).is_none());
        assert!(UsbEth::new([0; 6], Variant::Rtl8152).is_none());
        assert!(UsbEth::new([1, 2, 3, 4, 5, 6], Variant::CdcEcm).is_some());
    }

    #[test_case]
    fn non_network_interfaces_are_not_claimed() {
        // A HID keyboard or mass-storage device must never be taken for a NIC.
        assert_eq!(identify(0x046d, 0xc31c, 0x03, 0x01), None); // HID
        assert_eq!(identify(0x0781, 0x5567, 0x08, 0x06), None); // mass storage
        assert_eq!(identify(0x1234, 0x5678, CLASS_VENDOR, 0), None); // unknown vendor chip
    }
}
