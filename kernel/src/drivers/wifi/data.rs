//! **The 802.11 data path** — turning Ethernet frames into air frames and back.
//!
//! smoltcp speaks Ethernet: destination, source, ethertype, payload. A Wi-Fi
//! radio speaks 802.11: a 24-byte MAC header whose three address fields mean
//! *different things depending on two bits*, followed by an LLC/SNAP shim
//! before the ethertype. This module is the conversion, and it is the last
//! piece between an associated station and a working link.
//!
//! Pure and testable: a radio driver calls [`to_air`] on the way out and
//! [`from_air`] on the way in, and neither touches hardware.
//!
//! ## The addressing, which is the whole difficulty
//!
//! An 802.11 header has three addresses and no field saying which is which.
//! The **ToDS and FromDS bits** decide, and for an ordinary client both
//! directions are in play at once:
//!
//! | ToDS | FromDS | A1 | A2 | A3 | Case |
//! |---|---|---|---|---|---|
//! | 0 | 0 | dst | src | BSSID | ad-hoc / management |
//! | **1** | **0** | **BSSID** | **src** | **dst** | **station → AP (uplink)** |
//! | 0 | **1** | **dst** | **BSSID** | **src** | **AP → station (downlink)** |
//! | 1 | 1 | RA | TA | dst | mesh / WDS (four addresses) |
//!
//! So A1 is the destination on the way in and the *BSSID* on the way out, and
//! A3 swaps with whichever of the two is not A1. Getting it backwards produces
//! frames the AP silently drops — it sees itself as the final destination of
//! traffic addressed to the internet — and, on receive, hands the stack frames
//! whose source is the AP's own MAC, so every ARP reply teaches the wrong
//! thing and nothing routes.
//!
//! ## The SNAP shim
//!
//! 802.11 has no ethertype field. The payload begins with an 802.2 LLC header
//! (`AA AA 03`) and a SNAP header (`00 00 00` OUI) and only then the two-byte
//! ethertype. Omitting it shifts every packet by eight bytes, so an IPv4 header
//! begins in the middle of what the peer reads as its ethertype — a link that
//! associates, encrypts correctly, and carries nothing that parses.

use super::ccmp;
use alloc::vec::Vec;

/// 802.11 MAC header for a three-address data frame.
pub const HDR_LEN: usize = 24;
/// LLC (`AA AA 03`) + SNAP (`00 00 00` OUI) + 2-byte ethertype.
pub const SNAP_LEN: usize = 8;
/// An Ethernet header: destination, source, ethertype.
pub const ETH_HDR_LEN: usize = 14;

/// Frame Control: type Data, subtype Data.
const FC_TYPE_DATA: u16 = 0x0008;
const FC_TO_DS: u16 = 0x0100;
const FC_FROM_DS: u16 = 0x0200;
const FC_PROTECTED: u16 = 0x4000;

/// The fixed LLC/SNAP prefix that precedes the ethertype.
pub const SNAP_PREFIX: [u8; 6] = [0xaa, 0xaa, 0x03, 0x00, 0x00, 0x00];

/// A parsed Ethernet frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EthFrame<'a> {
    pub dst: [u8; 6],
    pub src: [u8; 6],
    pub ethertype: u16,
    pub payload: &'a [u8],
}

/// Split an Ethernet frame into its header and payload.
pub fn parse_eth(frame: &[u8]) -> Option<EthFrame<'_>> {
    if frame.len() < ETH_HDR_LEN {
        return None;
    }
    Some(EthFrame {
        dst: frame[0..6].try_into().ok()?,
        src: frame[6..12].try_into().ok()?,
        ethertype: u16::from_be_bytes([frame[12], frame[13]]),
        payload: &frame[ETH_HDR_LEN..],
    })
}

/// Build the 802.11 header for a station transmitting to its AP (ToDS).
///
/// A1 is the **BSSID**, not the destination — the frame is addressed to the
/// access point, which forwards it. The real destination rides in A3.
pub fn uplink_header(
    bssid: &[u8; 6],
    src: &[u8; 6],
    dst: &[u8; 6],
    seq: u16,
    protected: bool,
) -> Vec<u8> {
    let mut fc = FC_TYPE_DATA | FC_TO_DS;
    if protected {
        fc |= FC_PROTECTED;
    }
    let mut h = Vec::with_capacity(HDR_LEN);
    h.extend_from_slice(&fc.to_le_bytes());
    h.extend_from_slice(&[0, 0]); // duration, set by the radio
    h.extend_from_slice(bssid); // A1 = receiver = the AP
    h.extend_from_slice(src); // A2 = transmitter = us
    h.extend_from_slice(dst); // A3 = final destination
                              // Sequence Control: 12-bit sequence number, 4-bit fragment number.
    h.extend_from_slice(&(seq << 4).to_le_bytes());
    h
}

/// Where the three addresses of a received frame actually are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Addresses {
    pub dst: [u8; 6],
    pub src: [u8; 6],
    pub bssid: [u8; 6],
}

/// Resolve A1/A2/A3 into destination, source and BSSID using the ToDS/FromDS
/// bits.
///
/// The four-address case (both bits set) is refused: it is mesh/WDS, its header
/// is 30 bytes, and treating it as three-address reads the destination out of
/// the wrong field.
pub fn addresses(hdr: &[u8]) -> Option<Addresses> {
    if hdr.len() < HDR_LEN {
        return None;
    }
    let fc = u16::from_le_bytes([hdr[0], hdr[1]]);
    let a1: [u8; 6] = hdr[4..10].try_into().ok()?;
    let a2: [u8; 6] = hdr[10..16].try_into().ok()?;
    let a3: [u8; 6] = hdr[16..22].try_into().ok()?;
    let to_ds = fc & FC_TO_DS != 0;
    let from_ds = fc & FC_FROM_DS != 0;
    match (to_ds, from_ds) {
        // AP → station: A1 is us, A2 is the AP, A3 is who sent it.
        (false, true) => Some(Addresses {
            dst: a1,
            src: a3,
            bssid: a2,
        }),
        // Station → AP: A1 is the AP, A2 is the sender, A3 is the destination.
        (true, false) => Some(Addresses {
            dst: a3,
            src: a2,
            bssid: a1,
        }),
        // Ad-hoc / management.
        (false, false) => Some(Addresses {
            dst: a1,
            src: a2,
            bssid: a3,
        }),
        // Four-address WDS — a different header length entirely.
        (true, true) => None,
    }
}

/// Wrap an Ethernet payload in LLC/SNAP.
pub fn snap_encap(ethertype: u16, payload: &[u8]) -> Vec<u8> {
    let mut b = Vec::with_capacity(SNAP_LEN + payload.len());
    b.extend_from_slice(&SNAP_PREFIX);
    b.extend_from_slice(&ethertype.to_be_bytes());
    b.extend_from_slice(payload);
    b
}

/// Strip LLC/SNAP, returning the ethertype and the payload after it.
///
/// A body that does not start with the SNAP prefix is refused rather than
/// guessed at: some legacy encapsulations put a bare 802.3 length there, and
/// reading that as an ethertype yields a plausible small number.
pub fn snap_decap(body: &[u8]) -> Option<(u16, &[u8])> {
    if body.len() < SNAP_LEN || body[..6] != SNAP_PREFIX {
        return None;
    }
    Some((u16::from_be_bytes([body[6], body[7]]), &body[SNAP_LEN..]))
}

/// Convert an Ethernet frame into a complete 802.11 frame ready for the radio.
///
/// When `tk` is `Some`, the body is CCMP-encrypted and the Protected bit set;
/// the packet number comes from the caller's counter, which must never repeat
/// under one key.
pub fn to_air(
    eth: &[u8],
    bssid: &[u8; 6],
    seq: u16,
    tk: Option<(&[u8; 16], &[u8; ccmp::PN_LEN], u8)>,
) -> Option<Vec<u8>> {
    let e = parse_eth(eth)?;
    let hdr = uplink_header(bssid, &e.src, &e.dst, seq, tk.is_some());
    let body = snap_encap(e.ethertype, e.payload);
    let mut out = hdr.clone();
    match tk {
        Some((key, pn, key_id)) => {
            let enc = ccmp::encrypt(key, &hdr, pn, key_id, &body)?;
            out.extend_from_slice(&enc);
        }
        None => out.extend_from_slice(&body),
    }
    Some(out)
}

/// Convert a received 802.11 data frame back into an Ethernet frame.
///
/// Returns `None` for anything that is not a decryptable, SNAP-encapsulated
/// data frame — including a frame whose Protected bit is set when no key is
/// supplied, which is the case that must never fall through to "treat the
/// ciphertext as a payload".
pub fn from_air(
    frame: &[u8],
    tk: Option<&[u8; 16]>,
) -> Option<(Vec<u8>, Option<[u8; ccmp::PN_LEN]>)> {
    if frame.len() < HDR_LEN {
        return None;
    }
    let fc = u16::from_le_bytes([frame[0], frame[1]]);
    // Data frames only. A management or control frame reaching here is a caller
    // bug, and decoding it as data would produce a well-formed Ethernet frame
    // out of a beacon.
    if (fc >> 2) & 0x3 != 2 {
        return None;
    }
    let hlen = ccmp::header_len(fc);
    if frame.len() < hlen {
        return None;
    }
    let addr = addresses(frame)?;
    let (hdr, rest) = frame.split_at(hlen);

    let body: Vec<u8>;
    let mut pn = None;
    if fc & FC_PROTECTED != 0 {
        // Encrypted: a key is required. Without one the only safe answer is to
        // drop the frame — the alternative hands the stack ciphertext.
        let key = tk?;
        let (plain, got) = ccmp::decrypt(key, hdr, rest)?;
        body = plain;
        pn = Some(got);
    } else {
        body = rest.to_vec();
    }

    let (ethertype, payload) = snap_decap(&body)?;
    let mut eth = Vec::with_capacity(ETH_HDR_LEN + payload.len());
    eth.extend_from_slice(&addr.dst);
    eth.extend_from_slice(&addr.src);
    eth.extend_from_slice(&ethertype.to_be_bytes());
    eth.extend_from_slice(payload);
    Some((eth, pn))
}

/// A 12-bit transmit sequence number.
///
/// Wraps, unlike the packet number: the sequence number is not a security
/// counter, it only orders and de-duplicates, and 802.11 defines it as modulo
/// 4096.
#[derive(Debug, Clone, Copy, Default)]
pub struct SeqCounter(u16);

impl SeqCounter {
    pub fn new() -> SeqCounter {
        SeqCounter(0)
    }
    pub fn next(&mut self) -> u16 {
        let v = self.0;
        self.0 = (self.0 + 1) & 0x0fff;
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const AP: [u8; 6] = [0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff];
    const US: [u8; 6] = [0x02, 0x11, 0x22, 0x33, 0x44, 0x55];
    const PEER: [u8; 6] = [0x06, 0x99, 0x88, 0x77, 0x66, 0x55];

    fn eth_frame(dst: [u8; 6], src: [u8; 6], ethertype: u16, payload: &[u8]) -> Vec<u8> {
        let mut f = Vec::new();
        f.extend_from_slice(&dst);
        f.extend_from_slice(&src);
        f.extend_from_slice(&ethertype.to_be_bytes());
        f.extend_from_slice(payload);
        f
    }

    /// **A1 is the BSSID on the way out, not the destination.** Addressing the
    /// frame to the peer makes the AP see traffic that is not for it to forward
    /// and drop it — an association that works and a link that carries nothing.
    #[test_case]
    fn an_uplink_frame_is_addressed_to_the_access_point() {
        let h = uplink_header(&AP, &US, &PEER, 7, false);
        assert_eq!(h.len(), HDR_LEN);
        let fc = u16::from_le_bytes([h[0], h[1]]);
        assert_ne!(fc & FC_TO_DS, 0, "ToDS");
        assert_eq!(fc & FC_FROM_DS, 0);
        assert_eq!(&h[4..10], &AP[..], "A1 = BSSID");
        assert_eq!(&h[10..16], &US[..], "A2 = us");
        assert_eq!(&h[16..22], &PEER[..], "A3 = the real destination");
        // The sequence number sits in the top 12 bits; the low nibble is the
        // fragment number and must be zero for an unfragmented frame.
        assert_eq!(u16::from_le_bytes([h[22], h[23]]), 7 << 4);
    }

    /// **ToDS and FromDS permute the address fields**, and reading them the
    /// wrong way round makes every received frame appear to come from the AP —
    /// so every ARP reply teaches the wrong MAC and nothing routes.
    #[test_case]
    fn to_ds_and_from_ds_permute_the_addresses() {
        // Downlink: AP → us. A1 = us, A2 = AP, A3 = the original sender.
        let mut down = Vec::new();
        down.extend_from_slice(&(FC_TYPE_DATA | FC_FROM_DS).to_le_bytes());
        down.extend_from_slice(&[0, 0]);
        down.extend_from_slice(&US); // A1
        down.extend_from_slice(&AP); // A2
        down.extend_from_slice(&PEER); // A3
        down.extend_from_slice(&[0, 0]);
        let a = addresses(&down).unwrap();
        assert_eq!(a.dst, US);
        assert_eq!(a.src, PEER, "the sender is A3, NOT the AP");
        assert_eq!(a.bssid, AP);

        // Uplink is the mirror image.
        let up = uplink_header(&AP, &US, &PEER, 0, false);
        let a = addresses(&up).unwrap();
        assert_eq!(a.dst, PEER);
        assert_eq!(a.src, US);
        assert_eq!(a.bssid, AP);

        // Ad-hoc: no DS bits at all.
        let mut ibss = Vec::new();
        ibss.extend_from_slice(&FC_TYPE_DATA.to_le_bytes());
        ibss.extend_from_slice(&[0, 0]);
        ibss.extend_from_slice(&PEER);
        ibss.extend_from_slice(&US);
        ibss.extend_from_slice(&AP);
        ibss.extend_from_slice(&[0, 0]);
        let a = addresses(&ibss).unwrap();
        assert_eq!((a.dst, a.src, a.bssid), (PEER, US, AP));

        // Four-address WDS has a longer header and is refused rather than
        // misread.
        let mut wds = up.clone();
        let fc = FC_TYPE_DATA | FC_TO_DS | FC_FROM_DS;
        wds[0..2].copy_from_slice(&fc.to_le_bytes());
        assert_eq!(addresses(&wds), None);
    }

    /// **802.11 has no ethertype field.** Omitting the LLC/SNAP shim shifts
    /// every packet by eight bytes, so an IPv4 header starts in the middle of
    /// what the peer reads as its ethertype.
    #[test_case]
    fn the_snap_shim_carries_the_ethertype() {
        let body = snap_encap(0x0800, b"payload");
        assert_eq!(&body[..6], &SNAP_PREFIX, "AA AA 03 then a zero OUI");
        assert_eq!(&body[6..8], &[0x08, 0x00], "ethertype, big-endian");
        assert_eq!(&body[8..], b"payload");
        assert_eq!(snap_decap(&body), Some((0x0800u16, &b"payload"[..])));

        // A body that is not SNAP is refused rather than read as one: a bare
        // 802.3 length there would decode as a plausible small ethertype.
        let mut not_snap = body.clone();
        not_snap[0] = 0x00;
        assert_eq!(snap_decap(&not_snap), None);
        assert_eq!(snap_decap(&[0xaa, 0xaa, 0x03]), None, "too short");
    }

    /// The whole round trip, unencrypted: Ethernet in, 802.11 out, Ethernet
    /// back — and the addresses survive it.
    #[test_case]
    fn an_ethernet_frame_round_trips_through_the_air_unencrypted() {
        let eth = eth_frame(PEER, US, 0x0800, b"hello there");
        let air = to_air(&eth, &AP, 3, None).expect("converts");
        assert_eq!(&air[4..10], &AP[..], "addressed to the AP");

        // Coming back the other way the AP would set FromDS; simulate that.
        let mut down = air.clone();
        let fc = FC_TYPE_DATA | FC_FROM_DS;
        down[0..2].copy_from_slice(&fc.to_le_bytes());
        down[4..10].copy_from_slice(&US); // A1 = us
        down[10..16].copy_from_slice(&AP); // A2 = AP
        down[16..22].copy_from_slice(&PEER); // A3 = sender

        let (back, pn) = from_air(&down, None).expect("converts back");
        assert_eq!(pn, None, "not encrypted");
        let e = parse_eth(&back).unwrap();
        assert_eq!(e.dst, US);
        assert_eq!(e.src, PEER);
        assert_eq!(e.ethertype, 0x0800);
        assert_eq!(e.payload, b"hello there");
    }

    /// The encrypted round trip — the path a real link actually uses.
    #[test_case]
    fn an_encrypted_frame_round_trips_and_is_really_encrypted() {
        let tk = [0x77u8; 16];
        let pn = [0, 0, 0, 0, 0, 5];
        let eth = eth_frame(PEER, US, 0x0800, b"secret payload here");
        let air = to_air(&eth, &AP, 1, Some((&tk, &pn, 0))).expect("encrypts");

        let fc = u16::from_le_bytes([air[0], air[1]]);
        assert_ne!(fc & FC_PROTECTED, 0, "the Protected bit must be set");
        // The plaintext must not appear anywhere in the frame.
        assert!(
            !air.windows(6).any(|w| w == b"secret"),
            "plaintext leaked into the air frame"
        );
        assert_eq!(air.len(), HDR_LEN + ccmp::OVERHEAD + SNAP_LEN + 19);

        let (back, got_pn) = from_air(&air, Some(&tk)).expect("decrypts");
        assert_eq!(got_pn, Some(pn));
        let e = parse_eth(&back).unwrap();
        assert_eq!(e.payload, b"secret payload here");
        // Uplink, so the recovered destination is A3.
        assert_eq!(e.dst, PEER);
        assert_eq!(e.src, US);
    }

    /// **A protected frame with no key is dropped, never passed through.**
    /// Falling back to "treat the body as plaintext" hands the network stack
    /// ciphertext, which it will happily try to parse as IP.
    #[test_case]
    fn a_protected_frame_without_a_key_is_dropped() {
        let tk = [0x77u8; 16];
        let pn = [0, 0, 0, 0, 0, 1];
        let eth = eth_frame(PEER, US, 0x0800, b"payload");
        let air = to_air(&eth, &AP, 1, Some((&tk, &pn, 0))).unwrap();

        assert_eq!(from_air(&air, None), None, "no key, no packet");
        assert_eq!(
            from_air(&air, Some(&[0x00; 16])),
            None,
            "wrong key fails the MIC"
        );

        // A single flipped ciphertext byte must fail too.
        let mut bad = air.clone();
        let at = HDR_LEN + ccmp::HDR_LEN + 2;
        bad[at] ^= 1;
        assert_eq!(from_air(&bad, Some(&tk)), None);
    }

    /// Only data frames convert. A beacon reaching here would otherwise become
    /// a well-formed Ethernet frame full of management fields.
    #[test_case]
    fn management_and_control_frames_are_not_data() {
        let mut beacon = uplink_header(&AP, &US, &PEER, 0, false);
        beacon[0..2].copy_from_slice(&0x0080u16.to_le_bytes()); // mgmt, beacon
        beacon.extend_from_slice(&snap_encap(0x0800, b"x"));
        assert_eq!(from_air(&beacon, None), None);

        // Truncated frames are refused rather than read past.
        assert_eq!(from_air(&[], None), None);
        assert_eq!(from_air(&beacon[..20], None), None);
    }

    /// A frame that decrypts but is not SNAP-encapsulated is dropped: the
    /// bytes after the header are not an ethertype, and reading them as one
    /// invents a protocol.
    #[test_case]
    fn a_data_frame_without_snap_is_dropped() {
        let hdr = uplink_header(&AP, &US, &PEER, 0, false);
        let mut f = hdr.clone();
        f.extend_from_slice(&[0x45, 0x00, 0x00, 0x28, 0, 0, 0, 0, 0, 0]); // raw IPv4
        assert_eq!(from_air(&f, None), None);
    }

    /// The sequence number wraps at 12 bits — unlike the packet number, it is
    /// ordering, not security, and 802.11 defines it modulo 4096.
    #[test_case]
    fn the_sequence_number_wraps_at_twelve_bits() {
        let mut s = SeqCounter::new();
        assert_eq!(s.next(), 0);
        assert_eq!(s.next(), 1);
        let mut s = SeqCounter(0x0fff);
        assert_eq!(s.next(), 0x0fff);
        assert_eq!(s.next(), 0, "wraps rather than overflowing the field");
        // And it always fits the field the header packs it into.
        let h = uplink_header(&AP, &US, &PEER, 0x0fff, false);
        assert_eq!(u16::from_le_bytes([h[22], h[23]]) >> 4, 0x0fff);
    }
}
