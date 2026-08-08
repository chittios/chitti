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

use crate::net::rndis;
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
    /// Microsoft RNDIS — implemented, with a 44-byte per-packet header and a
    /// control-pipe bring-up sequence. This is **Android USB tethering**, and
    /// what QEMU's `usb-net` emulates.
    Rndis,
    /// ASIX AX88179/AX88772 — recognised, not implemented.
    Asix,
    /// Realtek RTL8152/RTL8153 — recognised, not implemented.
    Rtl8152,
}

impl Variant {
    pub fn name(self) -> &'static str {
        match self {
            Variant::CdcEcm => "cdc-ecm",
            Variant::Rndis => "rndis",
            Variant::Asix => "asix",
            Variant::Rtl8152 => "rtl8152",
        }
    }

    /// Whether frames cross the bulk endpoints with no extra header. RNDIS does
    /// **not** — it prefixes each frame with a 44-byte header, which is why it
    /// needs its own send/receive path rather than sharing CDC-ECM's.
    pub fn is_raw_frames(self) -> bool {
        matches!(self, Variant::CdcEcm)
    }

    /// Whether this driver can actually drive the variant.
    pub fn is_supported(self) -> bool {
        matches!(self, Variant::CdcEcm | Variant::Rndis)
    }
}

// --- RNDIS control-interface identification -------------------------------

/// USB class triple for the **RNDIS** control interface as Android and most
/// gadget stacks report it: Wireless Controller / RF / RNDIS.
pub const CLASS_WIRELESS: u8 = 0xe0;
pub const SUBCLASS_RF: u8 = 0x01;
pub const PROTO_RNDIS: u8 = 0x03;
/// CDC subclass for Abstract Control Model — the *other* RNDIS advertisement,
/// used with a vendor-specific protocol byte.
pub const SUBCLASS_ACM: u8 = 0x02;
pub const PROTO_VENDOR: u8 = 0xff;

/// True when an interface's class triple advertises RNDIS.
///
/// There are **two** encodings in the wild and a device presents only one of
/// them, so matching a single triple misses half the devices — including,
/// depending on the Android version, the phone in your pocket:
///
/// * `E0/01/03` — Wireless / RF / RNDIS. What the Linux gadget driver and most
///   Android builds report.
/// * `02/02/FF` — CDC / ACM / vendor-specific. The original Microsoft
///   advertisement, and what several USB-Ethernet dongles and QEMU use.
///
/// The second triple is *also* how a real CDC-ACM serial modem announces itself
/// with a vendor protocol, so this predicate alone must not claim a device —
/// the caller requires a bulk IN/OUT data interface alongside it, which a serial
/// modem's control interface does not have.
pub fn is_rndis_control(class: u8, subclass: u8, protocol: u8) -> bool {
    (class == CLASS_WIRELESS && subclass == SUBCLASS_RF && protocol == PROTO_RNDIS)
        || (class == CLASS_CDC_CONTROL && subclass == SUBCLASS_ACM && protocol == PROTO_VENDOR)
}

/// Identify a USB NIC from its vendor/product id and interface class.
///
/// The class check is what makes CDC-ECM recognisable across vendors — it is a
/// standard, so there is no id list to maintain. The vendor-specific chips do need
/// ids, and are matched first because they also report class `0xFF`.
///
/// `iface_protocol` separates RNDIS from CDC-ECM, which is the whole reason it is
/// a parameter: **both put their data on a class-`0x0A` interface**, so the data
/// interface alone cannot tell them apart, and getting it wrong means feeding
/// smoltcp 44 bytes of RNDIS header as though it were the start of an Ethernet
/// frame — every packet malformed, no error anywhere.
pub fn identify(
    vendor: u16,
    product: u16,
    iface_class: u8,
    iface_subclass: u8,
    iface_protocol: u8,
) -> Option<Variant> {
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
    // RNDIS before CDC-ECM: its control interface may report class 0x02 like an
    // ECM control interface does, and its data interface is 0x0A like ECM's.
    if is_rndis_control(iface_class, iface_subclass, iface_protocol) {
        return Some(Variant::Rndis);
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

/// Largest RNDIS transfer we read in one go. Must match what
/// [`rndis::MAX_TRANSFER_SIZE`] promised the device at initialize, or it batches
/// up to a size we then truncate — losing whole frames from the tail of every
/// busy transfer.
const RNDIS_XFER_MAX: usize = rndis::MAX_TRANSFER_SIZE as usize;

/// A USB Ethernet adapter driven through the xHCI bulk transport.
///
/// Holds no rings of its own: the endpoints live in the xHCI controller, and this is
/// the `NetDevice` face over them.
pub struct UsbEth {
    mac: [u8; 6],
    variant: Variant,
    /// Frames decoded out of one RNDIS transfer, awaiting collection. Empty for
    /// CDC-ECM, which never batches.
    pending: alloc::collections::VecDeque<alloc::vec::Vec<u8>>,
}

impl UsbEth {
    /// Wrap an already-configured bulk pair. `mac` comes from the device's CDC
    /// Ethernet functional descriptor (or its iMACAddress string), which the caller
    /// reads during enumeration.
    pub fn new(mac: [u8; 6], variant: Variant) -> Option<UsbEth> {
        if !variant.is_supported() {
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
        Some(UsbEth { mac, variant, pending: alloc::collections::VecDeque::new() })
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
    /// right first target. RNDIS instead wraps each frame in a 44-byte header and
    /// may batch several per transfer, so it drains through [`Self::pending`].
    fn receive(&mut self, out: &mut [u8]) -> Option<usize> {
        if self.variant == Variant::Rndis {
            return self.receive_rndis(out);
        }
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
        if self.variant == Variant::Rndis {
            // Wrap in a REMOTE_NDIS_PACKET_MSG. One frame per transfer on the
            // way out: batching is only worth it under load and a partially
            // written batch is worse than a dropped frame.
            let mut buf = [0u8; rndis::PACKET_HEADER_LEN + FRAME_MAX];
            if let Some(n) = rndis::encode_packet(frame, &mut buf) {
                let _ = crate::arch::usb_bulk_send(&buf[..n]);
            }
            return;
        }
        // Dropping when a transfer is still outstanding is legitimate for Ethernet;
        // smoltcp retransmits.
        let _ = crate::arch::usb_bulk_send(frame);
    }
}

impl UsbEth {
    /// Drain one Ethernet frame from an RNDIS transfer.
    ///
    /// **A transfer can hold several packets**, so completed transfers are
    /// decoded once into [`Self::pending`] and handed out one frame at a time.
    /// Reading only the first packet of each transfer is the classic RNDIS bug:
    /// it looks like heavy, load-dependent packet loss rather than a framing
    /// error, because a batch only forms when frames arrive faster than we poll.
    fn receive_rndis(&mut self, out: &mut [u8]) -> Option<usize> {
        if let Some(frame) = self.pending.pop_front() {
            let n = frame.len().min(out.len());
            out[..n].copy_from_slice(&frame[..n]);
            return Some(n);
        }
        crate::arch::usb_bulk_arm_in();
        let mut xfer = [0u8; RNDIS_XFER_MAX];
        let n = crate::arch::usb_bulk_take_in(&mut xfer)?;
        if n == 0 {
            return None;
        }
        let n = n.min(xfer.len());
        for span in rndis::decode_transfer(&xfer[..n]) {
            if span.len == 0 || span.len > FRAME_MAX {
                continue;
            }
            self.pending
                .push_back(xfer[span.start..span.start + span.len].to_vec());
        }
        let frame = self.pending.pop_front()?;
        let n = frame.len().min(out.len());
        out[..n].copy_from_slice(&frame[..n]);
        Some(n)
    }
}

/// Adopt a USB Ethernet adapter whose bulk endpoints the xHCI stack has already
/// configured (see `classify_and_finish`). `None` if no adapter is present.
///
/// ## The MAC address
///
/// CDC-ECM reports the adapter's real MAC through the `iMACAddress` index in its
/// Ethernet Networking functional descriptor, which requires fetching a *string*
/// descriptor — a control transfer this does not yet do. So a **locally-administered**
/// address is used instead (bit 1 of the first octet set, which is what that bit is
/// for), derived deterministically so it is stable across reboots.
///
/// That is legitimate rather than a bodge: a CDC-ECM adapter bridges whatever frames
/// the host sends, so the address the host uses is the address seen on the wire, and
/// a locally-administered one cannot collide with a real vendor assignment. Reading
/// the true MAC is a refinement, not a correctness fix.
/// **RNDIS**, by contrast, reports its MAC over the control pipe, so [`bring_up_rndis`]
/// asks for it and only falls back to the synthetic address when the query fails.
pub fn probe() -> Option<UsbEth> {
    if !crate::arch::usb_bulk_ready() {
        return None;
    }
    // Locally administered (0x02), stable, and obviously ours at a glance.
    let fallback_mac = [0x02, 0x43, 0x48, 0x49, 0x54, 0x01];
    match crate::arch::usb_bulk_is_rndis() {
        true => {
            let mac = bring_up_rndis().unwrap_or(fallback_mac);
            UsbEth::new(mac, Variant::Rndis)
        }
        false => {
            crate::ktrace::log("usb_eth", "adopting CDC-ECM adapter with a locally-administered MAC (real MAC needs the iMACAddress string descriptor)");
            UsbEth::new(fallback_mac, Variant::CdcEcm)
        }
    }
}

/// Run the RNDIS bring-up over the control pipe and return the device's MAC.
///
/// `INITIALIZE` → query `OID_802_3_PERMANENT_ADDRESS` → set
/// `OID_GEN_CURRENT_PACKET_FILTER`. The order matters and so does the last step:
/// **without the filter the device initialises cleanly, reports success to every
/// message, and never delivers a single frame**, because the default filter is
/// zero. That failure reads as a dead link with a healthy driver.
///
/// A failing step is reported and abandoned rather than retried blindly — an
/// RNDIS device that refuses `INITIALIZE` is not going to answer a query, and
/// pressing on would produce a NIC that appears configured and cannot work.
fn bring_up_rndis() -> Option<[u8; 6]> {
    let mut resp = [0u8; 256];

    // 1. INITIALIZE — establishes the version and the batching limits.
    let init = rndis::initialize_msg(1);
    let reply = rndis_control(&init, &mut resp, 1)?;
    let info = rndis::parse_initialize_cmplt(reply)?;
    if !info.ok() {
        crate::ktrace::log_fmt(format_args!(
            "usb_eth: rndis initialize refused (status {:#x}, medium {}) -- not an 802.3 device",
            info.status, info.medium
        ));
        return None;
    }
    crate::ktrace::log_fmt(format_args!(
        "usb_eth: rndis {}.{} up, max {} packet(s)/transfer, max transfer {}",
        info.major, info.minor, info.max_packets_per_transfer, info.max_transfer_size
    ));
    if (info.max_transfer_size as usize) < RNDIS_XFER_MAX {
        // Not fatal — the device simply batches less than we can hold — but it
        // is worth saying, because it bounds throughput and would otherwise be
        // invisible.
        crate::ktrace::log_fmt(format_args!(
            "usb_eth: rndis device accepts only {} bytes per transfer (we offered {RNDIS_XFER_MAX})",
            info.max_transfer_size
        ));
    }

    // 2. The MAC. A failure here is survivable: the caller substitutes a
    //    locally-administered address, which a tether bridges just as happily.
    let q = rndis::query_msg(2, rndis::OID_802_3_PERMANENT_ADDRESS);
    let mac = rndis_control(&q, &mut resp, 2).and_then(rndis::query_cmplt_mac);
    match mac {
        Some(m) => crate::ktrace::log_fmt(format_args!(
            "usb_eth: rndis MAC {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
            m[0], m[1], m[2], m[3], m[4], m[5]
        )),
        None => crate::ktrace::log("usb_eth", "rndis MAC query failed -- using a locally-administered address"),
    }

    // 3. The packet filter. **This one is not optional**, so a failure here
    //    fails the whole bring-up rather than leaving a silently deaf NIC.
    let s = rndis::set_packet_filter_msg(3, rndis::DEFAULT_PACKET_FILTER);
    let status = rndis_control(&s, &mut resp, 3).and_then(rndis::parse_set_cmplt);
    match status {
        Some(rndis::STATUS_SUCCESS) => {}
        other => {
            crate::ktrace::log_fmt(format_args!(
                "usb_eth: rndis packet filter refused ({other:?}) -- the device would receive nothing, refusing to adopt it"
            ));
            return None;
        }
    }
    mac
}

/// One RNDIS control exchange: push the message, read the reply, and check it is
/// answering *this* request.
///
/// The request-id check is what stops a stale completion — left queued by a
/// previous message that timed out — being read as this one's answer. On the
/// control pipe there is no ordering guarantee across an abandoned request, and
/// a mismatched reply is well-formed, so nothing else would catch it.
fn rndis_control<'a>(msg: &[u8], resp: &'a mut [u8], request_id: u32) -> Option<&'a [u8]> {
    if !crate::arch::usb_bulk_class_out(rndis::REQ_SEND_ENCAPSULATED_COMMAND, msg) {
        crate::ktrace::log("usb_eth", "rndis: SEND_ENCAPSULATED_COMMAND failed");
        return None;
    }
    let n = crate::arch::usb_bulk_class_in(rndis::REQ_GET_ENCAPSULATED_RESPONSE, resp)?;
    let reply = resp.get(..n)?;
    match rndis::completion_request_id(reply) {
        Some(id) if id == request_id => Some(reply),
        Some(id) => {
            crate::ktrace::log_fmt(format_args!(
                "usb_eth: rndis reply is for request {id}, expected {request_id} -- discarded"
            ));
            None
        }
        None => {
            crate::ktrace::log("usb_eth", "rndis: reply is not a completion message");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn identifies_cdc_ecm_by_class_not_by_id() {
        // CDC-ECM is a standard, so an unknown vendor must still be recognised —
        // maintaining an id list for it would guarantee gaps.
        assert_eq!(identify(0x1234, 0x5678, CLASS_CDC_DATA, 0x00, 0x00), Some(Variant::CdcEcm));
        assert_eq!(
            identify(0xffff, 0x0001, CLASS_CDC_CONTROL, SUBCLASS_ECM, 0x00),
            Some(Variant::CdcEcm)
        );
    }

    /// **RNDIS advertises itself two different ways and a device presents only
    /// one**, so matching a single triple misses half the devices — including,
    /// depending on the Android version, a phone being used to tether.
    #[test_case]
    fn identifies_rndis_by_either_of_its_two_class_triples() {
        // Wireless / RF / RNDIS — the Linux gadget driver and most Android builds.
        assert!(is_rndis_control(CLASS_WIRELESS, SUBCLASS_RF, PROTO_RNDIS));
        assert_eq!(
            identify(0x18d1, 0x4ee4, CLASS_WIRELESS, SUBCLASS_RF, PROTO_RNDIS),
            Some(Variant::Rndis)
        );
        // CDC / ACM / vendor — the original Microsoft advertisement, and QEMU's.
        assert!(is_rndis_control(CLASS_CDC_CONTROL, SUBCLASS_ACM, PROTO_VENDOR));
        assert_eq!(
            identify(0x0525, 0xa4a2, CLASS_CDC_CONTROL, SUBCLASS_ACM, PROTO_VENDOR),
            Some(Variant::Rndis)
        );
    }

    /// A near miss on either triple must **not** claim the device: `E0/01/01` is
    /// Bluetooth, which sits one protocol byte away from RNDIS and is a device
    /// this OS also drives — claiming it as a NIC would take the dongle away
    /// from the Bluetooth stack.
    #[test_case]
    fn a_near_miss_on_the_rndis_triple_is_not_claimed() {
        assert!(!is_rndis_control(CLASS_WIRELESS, SUBCLASS_RF, 0x01), "that is Bluetooth");
        assert!(!is_rndis_control(CLASS_WIRELESS, 0x02, PROTO_RNDIS));
        assert!(!is_rndis_control(CLASS_CDC_CONTROL, SUBCLASS_ACM, 0x01), "a real ACM modem");
        assert!(!is_rndis_control(CLASS_CDC_CONTROL, SUBCLASS_ECM, PROTO_VENDOR));
        assert_eq!(identify(0x1234, 0x5678, CLASS_WIRELESS, SUBCLASS_RF, 0x01), None);
    }

    #[test_case]
    fn identifies_the_common_vendor_specific_dongles() {
        // These report class 0xFF, so they must be matched by id before the class
        // check — and they are the chips in most cheap dongles and docks.
        assert_eq!(identify(0x0b95, 0x1790, CLASS_VENDOR, 0, 0), Some(Variant::Asix));
        assert_eq!(identify(0x0bda, 0x8153, CLASS_VENDOR, 0, 0), Some(Variant::Rtl8152));
        assert_eq!(identify(0x0bda, 0x8156, CLASS_VENDOR, 0, 0), Some(Variant::Rtl8152));
    }

    /// Only CDC-ECM puts a bare Ethernet frame on the wire. **RNDIS is supported
    /// but is not raw** — it prefixes 44 bytes — and conflating the two hands
    /// smoltcp an RNDIS header as the start of every frame, which is silent:
    /// no error, just a link on which nothing ever parses.
    #[test_case]
    fn raw_framing_and_support_are_different_questions() {
        assert!(Variant::CdcEcm.is_raw_frames());
        assert!(!Variant::Rndis.is_raw_frames(), "RNDIS carries a 44-byte header");
        assert!(!Variant::Asix.is_raw_frames());
        assert!(!Variant::Rtl8152.is_raw_frames());

        assert!(Variant::CdcEcm.is_supported());
        assert!(Variant::Rndis.is_supported());
        assert!(!Variant::Asix.is_supported());
        assert!(!Variant::Rtl8152.is_supported());

        // `new` refuses what it cannot drive, and accepts what it can.
        assert!(UsbEth::new([0; 6], Variant::Asix).is_none());
        assert!(UsbEth::new([0; 6], Variant::Rtl8152).is_none());
        assert!(UsbEth::new([1, 2, 3, 4, 5, 6], Variant::CdcEcm).is_some());
        assert!(UsbEth::new([1, 2, 3, 4, 5, 6], Variant::Rndis).is_some());
    }

    #[test_case]
    fn non_network_interfaces_are_not_claimed() {
        // A HID keyboard or mass-storage device must never be taken for a NIC.
        assert_eq!(identify(0x046d, 0xc31c, 0x03, 0x01, 0x01), None); // HID
        assert_eq!(identify(0x0781, 0x5567, 0x08, 0x06, 0x50), None); // mass storage
        assert_eq!(identify(0x1234, 0x5678, CLASS_VENDOR, 0, 0), None); // unknown vendor chip
    }
}
