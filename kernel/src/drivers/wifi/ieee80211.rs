//! **802.11 frames** — the wire format shared by every Wi-Fi driver here.
//!
//! Two very different jobs live in this file, and the split matters:
//!
//! - **Parsing** beacons, probe responses and association responses. These bytes come off
//!   the air from an unauthenticated sender, before any key exists. Anything within radio
//!   range can send them, so **every read is bounded and every malformed frame returns
//!   `None`** — a length field is a claim, never a fact. The tests include truncated and
//!   lying frames for that reason.
//! - **Building** authentication and association requests, which is ordinary serialisation.
//!
//! Kept device-independent on purpose: an Intel radio sends these frames itself while a
//! FullMAC Broadcom part builds them in firmware, but both need the same element parsing to
//! decide what a network *is* — and a scan result parsed two different ways in two drivers
//! is a bug waiting for whichever one is used less.

use alloc::string::String;
use alloc::vec::Vec;

/// Information-element IDs this code acts on.
pub const ELEM_SSID: u8 = 0;
pub const ELEM_SUPP_RATES: u8 = 1;
pub const ELEM_DS_PARAMS: u8 = 3;
pub const ELEM_RSN: u8 = 48;
pub const ELEM_EXT_SUPP_RATES: u8 = 50;
pub const ELEM_HT_OPERATION: u8 = 61;
pub const ELEM_VENDOR: u8 = 221;

/// Capability-field bit 4: the network requires encryption. Set for WPA/WPA2/WEP alike, so
/// it says "not open", not "WPA2" — the RSN element is what distinguishes those.
pub const CAP_PRIVACY: u16 = 1 << 4;

/// Fixed part of a beacon or probe-response body: timestamp, beacon interval, capability.
pub const BEACON_FIXED_LEN: usize = 12;
/// A management header: control, duration, three addresses, sequence.
pub const MGMT_HEADER_LEN: usize = 24;

/// Walk the information elements of a frame body.
///
/// Stops at the first element whose declared length runs past the end of the buffer rather
/// than clamping it. Clamping would hand callers a truncated SSID or a half-read RSN
/// element as though it were complete, and a frame that lies about a length is not a frame
/// to salvage.
pub struct Elements<'a> {
    rest: &'a [u8],
}

impl<'a> Iterator for Elements<'a> {
    type Item = (u8, &'a [u8]);

    fn next(&mut self) -> Option<Self::Item> {
        if self.rest.len() < 2 {
            return None;
        }
        let id = self.rest[0];
        let len = self.rest[1] as usize;
        if self.rest.len() < 2 + len {
            self.rest = &[];
            return None;
        }
        let body = &self.rest[2..2 + len];
        self.rest = &self.rest[2 + len..];
        Some((id, body))
    }
}

/// Iterate the elements in `body` (everything after a frame's fixed fields).
pub fn elements(body: &[u8]) -> Elements<'_> {
    Elements { rest: body }
}

/// A pairwise or group cipher.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cipher {
    /// AES-CCMP, the only cipher WPA2 requires and the only one worth supporting.
    Ccmp128,
    /// TKIP — WPA1's cipher, deprecated and broken.
    Tkip,
    Wep40,
    Wep104,
    /// "Group addressed traffic not allowed" — a valid group-cipher value.
    GroupNotAllowed,
    Other(u32),
}

impl Cipher {
    /// Decode a 4-byte suite selector (3-byte OUI + type).
    pub fn from_selector(sel: [u8; 4]) -> Cipher {
        let v = u32::from_be_bytes(sel);
        if sel[..3] == [0x00, 0x0f, 0xac] {
            match sel[3] {
                1 => Cipher::Wep40,
                2 => Cipher::Tkip,
                4 => Cipher::Ccmp128,
                5 => Cipher::Wep104,
                7 => Cipher::GroupNotAllowed,
                _ => Cipher::Other(v),
            }
        } else {
            Cipher::Other(v)
        }
    }
}

/// An authentication and key-management suite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Akm {
    /// WPA2-Personal: the pre-shared key handshake this kernel implements.
    Psk,
    /// WPA2-Personal with SHA-256 KDF — a different key derivation, not a variant of PSK.
    PskSha256,
    /// WPA3-Personal.
    Sae,
    /// Enterprise (802.1X/EAP), which needs a RADIUS conversation.
    Ieee8021x,
    Other(u32),
}

impl Akm {
    pub fn from_selector(sel: [u8; 4]) -> Akm {
        let v = u32::from_be_bytes(sel);
        if sel[..3] == [0x00, 0x0f, 0xac] {
            match sel[3] {
                1 => Akm::Ieee8021x,
                2 => Akm::Psk,
                6 => Akm::PskSha256,
                8 => Akm::Sae,
                _ => Akm::Other(v),
            }
        } else {
            Akm::Other(v)
        }
    }
}

/// A parsed RSN information element — what security the network actually offers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rsn {
    pub version: u16,
    pub group: Cipher,
    pub pairwise: Vec<Cipher>,
    pub akm: Vec<Akm>,
    pub capabilities: u16,
}

/// RSN capabilities bit 6: management-frame protection is *required*, so a client that
/// cannot do it must not associate rather than try and be rejected.
pub const RSN_CAP_MFP_REQUIRED: u16 = 1 << 6;

impl Rsn {
    /// Parse an RSN element body (everything after id and length).
    ///
    /// Every count is followed by exactly that many 4-byte selectors, and the counts come
    /// from the air — so each is checked against the remaining length before use. The
    /// trailing capabilities field is genuinely optional in the standard and its absence is
    /// not an error.
    pub fn parse(body: &[u8]) -> Option<Rsn> {
        if body.len() < 6 {
            return None;
        }
        let version = u16::from_le_bytes([body[0], body[1]]);
        let group = Cipher::from_selector([body[2], body[3], body[4], body[5]]);
        let mut at = 6;

        let mut take_count = |at: &mut usize| -> Option<usize> {
            if body.len() < *at + 2 {
                return None;
            }
            let n = u16::from_le_bytes([body[*at], body[*at + 1]]) as usize;
            *at += 2;
            // A count of 500 in a 20-byte element is the shape of a hostile frame.
            if body.len() < *at + n * 4 {
                return None;
            }
            Some(n)
        };

        let np = take_count(&mut at)?;
        let mut pairwise = Vec::with_capacity(np);
        for _ in 0..np {
            pairwise.push(Cipher::from_selector([
                body[at],
                body[at + 1],
                body[at + 2],
                body[at + 3],
            ]));
            at += 4;
        }
        let na = take_count(&mut at)?;
        let mut akm = Vec::with_capacity(na);
        for _ in 0..na {
            akm.push(Akm::from_selector([
                body[at],
                body[at + 1],
                body[at + 2],
                body[at + 3],
            ]));
            at += 4;
        }
        let capabilities = if body.len() >= at + 2 {
            u16::from_le_bytes([body[at], body[at + 1]])
        } else {
            0
        };
        Some(Rsn {
            version,
            group,
            pairwise,
            akm,
            capabilities,
        })
    }

    /// Whether this kernel can actually join the network: WPA2-PSK with CCMP both ways.
    ///
    /// Deliberately narrow. TKIP is broken, SAE and 802.1X are whole protocols that are not
    /// implemented, and required management-frame protection means an association attempt
    /// would be refused — reporting "unsupported" up front is more useful than a timeout.
    pub fn supported(&self) -> bool {
        self.version == 1
            && self.group == Cipher::Ccmp128
            && self.pairwise.contains(&Cipher::Ccmp128)
            && self.akm.contains(&Akm::Psk)
            && self.capabilities & RSN_CAP_MFP_REQUIRED == 0
    }
}

/// What a beacon or probe response says about a network.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bss {
    pub ssid: String,
    pub bssid: [u8; 6],
    /// 0 when the frame carried no channel element — the driver knows which channel it was
    /// listening on and can fill it in, whereas guessing here would be a lie.
    pub channel: u8,
    pub privacy: bool,
    pub rsn: Option<Rsn>,
    pub beacon_interval: u16,
}

impl Bss {
    /// True when this is a WPA2-PSK/CCMP network we can join.
    pub fn joinable(&self) -> bool {
        self.rsn.as_ref().map(|r| r.supported()).unwrap_or(false)
    }
}

/// Parse a beacon or probe-response management frame (starting at the frame control field).
///
/// Returns `None` for anything too short to contain a header and the fixed body — including
/// a runt frame from a broken transmitter, which is why the caller never gets a partially
/// filled `Bss`.
pub fn parse_beacon(frame: &[u8]) -> Option<Bss> {
    if frame.len() < MGMT_HEADER_LEN + BEACON_FIXED_LEN {
        return None;
    }
    let mut bssid = [0u8; 6];
    // addr3 is the BSSID in a beacon; addr2 (the transmitter) is the same for an AP, but
    // addr3 is the field that means "the network" and stays right for a mesh/repeater.
    bssid.copy_from_slice(&frame[16..22]);
    let body = &frame[MGMT_HEADER_LEN..];
    let beacon_interval = u16::from_le_bytes([body[8], body[9]]);
    let capability = u16::from_le_bytes([body[10], body[11]]);

    let mut ssid = String::new();
    let mut channel = 0u8;
    let mut rsn = None;
    for (id, val) in elements(&body[BEACON_FIXED_LEN..]) {
        match id {
            ELEM_SSID => {
                // An SSID is bytes, not text, and a hidden network sends a zero-length one.
                // Lossy conversion keeps a mis-encoded name visible rather than dropping
                // the network from the scan list entirely.
                ssid = String::from_utf8_lossy(val).into_owned();
            }
            ELEM_DS_PARAMS if !val.is_empty() => channel = val[0],
            ELEM_HT_OPERATION if channel == 0 && !val.is_empty() => channel = val[0],
            ELEM_RSN => rsn = Rsn::parse(val),
            _ => {}
        }
    }
    Some(Bss {
        ssid,
        bssid,
        channel,
        privacy: capability & CAP_PRIVACY != 0,
        rsn,
        beacon_interval,
    })
}

/// Frame-control values, pre-composed: the type/subtype pair with version 0.
pub const FC_ASSOC_REQ: u16 = 0x0000;
pub const FC_ASSOC_RESP: u16 = 0x0010;
pub const FC_PROBE_REQ: u16 = 0x0040;
pub const FC_PROBE_RESP: u16 = 0x0050;
pub const FC_BEACON: u16 = 0x0080;
pub const FC_AUTH: u16 = 0x00b0;
pub const FC_DEAUTH: u16 = 0x00c0;

/// Build a management header. `seq` is the 12-bit sequence number.
pub fn mgmt_header(fc: u16, dst: &[u8; 6], src: &[u8; 6], bssid: &[u8; 6], seq: u16) -> Vec<u8> {
    let mut f = Vec::with_capacity(MGMT_HEADER_LEN);
    f.extend_from_slice(&fc.to_le_bytes());
    f.extend_from_slice(&0u16.to_le_bytes()); // duration — the hardware fills this in
    f.extend_from_slice(dst);
    f.extend_from_slice(src);
    f.extend_from_slice(bssid);
    // Fragment number in the low 4 bits, sequence in the high 12.
    f.extend_from_slice(&((seq & 0x0fff) << 4).to_le_bytes());
    f
}

/// Build an open-system authentication request.
///
/// WPA2-PSK authenticates with the *four-way handshake*, not here: 802.11 authentication is
/// open, and the shared key is proven afterwards. A client that tries shared-key
/// authentication with a WPA2 network is refused.
pub fn auth_request(bssid: &[u8; 6], mac: &[u8; 6], seq: u16) -> Vec<u8> {
    let mut f = mgmt_header(FC_AUTH, bssid, mac, bssid, seq);
    f.extend_from_slice(&0u16.to_le_bytes()); // algorithm 0 = open system
    f.extend_from_slice(&1u16.to_le_bytes()); // transaction sequence 1
    f.extend_from_slice(&0u16.to_le_bytes()); // status 0
    f
}

/// The RSN element a WPA2-PSK/CCMP client advertises in its association request.
///
/// This must be **byte-identical** to what the four-way handshake later confirms: message 3
/// carries the AP's RSN element and the client compares it, and message 2 carries ours for
/// the AP to compare. A client that associates with one element and MICs another is
/// disconnected mid-handshake for what looks like a key failure.
pub fn client_rsn_element() -> Vec<u8> {
    let mut e = Vec::with_capacity(22);
    e.push(ELEM_RSN);
    e.push(20);
    e.extend_from_slice(&1u16.to_le_bytes()); // version 1
    e.extend_from_slice(&[0x00, 0x0f, 0xac, 0x04]); // group cipher CCMP
    e.extend_from_slice(&1u16.to_le_bytes());
    e.extend_from_slice(&[0x00, 0x0f, 0xac, 0x04]); // pairwise CCMP
    e.extend_from_slice(&1u16.to_le_bytes());
    e.extend_from_slice(&[0x00, 0x0f, 0xac, 0x02]); // AKM PSK
    e.extend_from_slice(&0u16.to_le_bytes()); // no capabilities claimed
    e
}

/// Build an association request for `bss`.
///
/// The rate elements are the mandatory 802.11g/n set; a modern AP ignores them in favour of
/// the HT/VHT elements, but omitting them entirely is rejected by some.
pub fn assoc_request(bss: &Bss, mac: &[u8; 6], seq: u16, listen_interval: u16) -> Vec<u8> {
    let mut f = mgmt_header(FC_ASSOC_REQ, &bss.bssid, mac, &bss.bssid, seq);
    let capability = CAP_PRIVACY | (1 << 0) /* ESS */ | (1 << 5) /* short preamble */;
    f.extend_from_slice(&capability.to_le_bytes());
    f.extend_from_slice(&listen_interval.to_le_bytes());

    f.push(ELEM_SSID);
    let ssid = bss.ssid.as_bytes();
    f.push(ssid.len() as u8);
    f.extend_from_slice(ssid);

    f.push(ELEM_SUPP_RATES);
    f.push(8);
    // 1, 2, 5.5, 11, 6, 9, 12, 18 Mbit/s in half-Mbit units, the basic ones flagged.
    f.extend_from_slice(&[0x82, 0x84, 0x8b, 0x96, 0x0c, 0x12, 0x18, 0x24]);
    f.push(ELEM_EXT_SUPP_RATES);
    f.push(4);
    f.extend_from_slice(&[0x30, 0x48, 0x60, 0x6c]); // 24, 36, 48, 54

    f.extend_from_slice(&client_rsn_element());
    f
}

/// The outcome of an association attempt. A non-zero status is the AP's own reason code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AssocResponse {
    pub status: u16,
    pub aid: u16,
}

/// Parse an association response.
pub fn parse_assoc_response(frame: &[u8]) -> Option<AssocResponse> {
    // capability (2) + status (2) + AID (2) after the header.
    if frame.len() < MGMT_HEADER_LEN + 6 {
        return None;
    }
    let b = &frame[MGMT_HEADER_LEN..];
    Some(AssocResponse {
        status: u16::from_le_bytes([b[2], b[3]]),
        // Only the low 14 bits are the association id; the top two are reserved and some
        // firmware sets them.
        aid: u16::from_le_bytes([b[4], b[5]]) & 0x3fff,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A beacon carrying an SSID, a channel and the canonical WPA2-PSK/CCMP RSN element.
    fn beacon(ssid: &[u8], rsn: Option<&[u8]>, capability: u16) -> Vec<u8> {
        let mut f = mgmt_header(FC_BEACON, &[0xff; 6], &[0x02; 6], &[0x02; 6], 1);
        f[16..22].copy_from_slice(&[0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff]); // addr3 = BSSID
        f.extend_from_slice(&[0u8; 8]); // timestamp
        f.extend_from_slice(&100u16.to_le_bytes()); // beacon interval
        f.extend_from_slice(&capability.to_le_bytes());
        f.push(ELEM_SSID);
        f.push(ssid.len() as u8);
        f.extend_from_slice(ssid);
        f.push(ELEM_DS_PARAMS);
        f.push(1);
        f.push(6);
        if let Some(r) = rsn {
            f.push(ELEM_RSN);
            f.push(r.len() as u8);
            f.extend_from_slice(r);
        }
        f
    }

    /// Body of the RSN element every WPA2-PSK access point sends.
    const RSN_WPA2_PSK_CCMP: &[u8] = &[
        0x01, 0x00, // version 1
        0x00, 0x0f, 0xac, 0x04, // group CCMP
        0x01, 0x00, 0x00, 0x0f, 0xac, 0x04, // 1 pairwise: CCMP
        0x01, 0x00, 0x00, 0x0f, 0xac, 0x02, // 1 AKM: PSK
        0x00, 0x00, // capabilities
    ];

    #[test_case]
    fn a_wpa2_psk_beacon_parses_into_a_joinable_network() {
        let f = beacon(b"chitti-lan", Some(RSN_WPA2_PSK_CCMP), CAP_PRIVACY | 1);
        let b = parse_beacon(&f).expect("a well-formed beacon must parse");
        assert_eq!(b.ssid, "chitti-lan");
        assert_eq!(b.bssid, [0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff]);
        assert_eq!(b.channel, 6);
        assert_eq!(b.beacon_interval, 100);
        assert!(b.privacy);
        let rsn = b.rsn.as_ref().expect("RSN element missing");
        assert_eq!(rsn.group, Cipher::Ccmp128);
        assert_eq!(rsn.pairwise, alloc::vec![Cipher::Ccmp128]);
        assert_eq!(rsn.akm, alloc::vec![Akm::Psk]);
        assert!(b.joinable());
    }

    #[test_case]
    fn an_open_network_has_no_rsn_and_a_hidden_one_no_name() {
        let open = parse_beacon(&beacon(b"cafe", None, 1)).unwrap();
        assert!(!open.privacy);
        assert!(open.rsn.is_none());
        assert!(!open.joinable(), "an open network is not a WPA2 one");

        // A hidden network beacons a zero-length SSID. It must still appear as a network —
        // dropping it would make it invisible to a user who knows its name.
        let hidden = parse_beacon(&beacon(b"", Some(RSN_WPA2_PSK_CCMP), CAP_PRIVACY | 1)).unwrap();
        assert_eq!(hidden.ssid, "");
        assert!(hidden.joinable());
    }

    #[test_case]
    fn the_security_we_cannot_do_is_reported_as_unsupported() {
        // Each of these would otherwise fail as a timeout or a MIC error after several
        // seconds, which reads to a user as a wrong password.
        let tkip = &[
            0x01, 0x00, 0x00, 0x0f, 0xac, 0x02, // group TKIP
            0x01, 0x00, 0x00, 0x0f, 0xac, 0x02, // pairwise TKIP
            0x01, 0x00, 0x00, 0x0f, 0xac, 0x02, // PSK
            0x00, 0x00,
        ];
        assert!(!Rsn::parse(tkip).unwrap().supported(), "TKIP is not supported");

        let sae = &[
            0x01, 0x00, 0x00, 0x0f, 0xac, 0x04, 0x01, 0x00, 0x00, 0x0f, 0xac, 0x04, 0x01, 0x00,
            0x00, 0x0f, 0xac, 0x08, // AKM SAE (WPA3)
            0x00, 0x00,
        ];
        let sae = Rsn::parse(sae).unwrap();
        assert_eq!(sae.akm, alloc::vec![Akm::Sae]);
        assert!(!sae.supported());

        let enterprise = &[
            0x01, 0x00, 0x00, 0x0f, 0xac, 0x04, 0x01, 0x00, 0x00, 0x0f, 0xac, 0x04, 0x01, 0x00,
            0x00, 0x0f, 0xac, 0x01, // AKM 802.1X
            0x00, 0x00,
        ];
        assert!(!Rsn::parse(enterprise).unwrap().supported());

        // Management-frame protection *required* — associating would be refused.
        let mut mfp = alloc::vec::Vec::from(RSN_WPA2_PSK_CCMP);
        let n = mfp.len();
        mfp[n - 2..].copy_from_slice(&RSN_CAP_MFP_REQUIRED.to_le_bytes());
        let mfp = Rsn::parse(&mfp).unwrap();
        assert_eq!(mfp.akm, alloc::vec![Akm::Psk]);
        assert!(!mfp.supported(), "required MFP must not be claimed as joinable");
    }

    #[test_case]
    fn a_frame_that_lies_about_its_lengths_is_refused_not_salvaged() {
        // These bytes arrive from an unauthenticated sender in radio range, so the parser is
        // an attack surface. Nothing here may panic, and nothing may return half-read data
        // as though it were a network.

        // An element claiming more bytes than the frame holds.
        let mut f = beacon(b"net", None, 1);
        f.push(ELEM_RSN);
        f.push(200);
        let b = parse_beacon(&f).expect("the elements before the bad one still parse");
        assert!(b.rsn.is_none(), "a truncated RSN element was accepted");

        // An RSN element whose selector counts overrun it.
        let lying_counts = &[
            0x01, 0x00, 0x00, 0x0f, 0xac, 0x04, //
            0xff, 0x00, // 255 pairwise ciphers in 20 bytes
        ];
        assert!(Rsn::parse(lying_counts).is_none());
        // And one whose AKM count overruns after a valid pairwise list.
        let lying_akm = &[
            0x01, 0x00, 0x00, 0x0f, 0xac, 0x04, 0x01, 0x00, 0x00, 0x0f, 0xac, 0x04, 0x40, 0x00,
        ];
        assert!(Rsn::parse(lying_akm).is_none());

        // Every truncation of a good frame, and of a good RSN element.
        let good = beacon(b"chitti-lan", Some(RSN_WPA2_PSK_CCMP), CAP_PRIVACY | 1);
        for n in 0..good.len() {
            let _ = parse_beacon(&good[..n]); // must not panic
        }
        for n in 0..RSN_WPA2_PSK_CCMP.len() {
            let _ = Rsn::parse(&RSN_WPA2_PSK_CCMP[..n]);
        }
        // A capabilities-less RSN element is legal, not truncated.
        let short = &RSN_WPA2_PSK_CCMP[..RSN_WPA2_PSK_CCMP.len() - 2];
        let r = Rsn::parse(short).expect("the capabilities field is optional");
        assert_eq!(r.capabilities, 0);
        assert!(r.supported());
    }

    #[test_case]
    fn the_element_walker_stops_cleanly_at_a_bad_length() {
        let buf = &[ELEM_SSID, 2, b'h', b'i', ELEM_DS_PARAMS, 1, 6];
        let got: alloc::vec::Vec<_> = elements(buf).collect();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0], (ELEM_SSID, &b"hi"[..]));
        assert_eq!(got[1], (ELEM_DS_PARAMS, &[6u8][..]));

        // A trailing byte is not an element, and a lying length ends the walk.
        assert_eq!(elements(&[ELEM_SSID]).count(), 0);
        assert_eq!(elements(&[ELEM_SSID, 5, b'a']).count(), 0);
        // A zero-length element is valid and yields an empty body.
        assert_eq!(elements(&[ELEM_SSID, 0]).next(), Some((ELEM_SSID, &[][..])));
    }

    #[test_case]
    fn the_frames_we_build_have_the_right_shape() {
        let bssid = [0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff];
        let mac = [0x02, 0, 0, 0, 0, 1];

        let auth = auth_request(&bssid, &mac, 0);
        assert_eq!(auth.len(), MGMT_HEADER_LEN + 6);
        assert_eq!(u16::from_le_bytes([auth[0], auth[1]]), FC_AUTH);
        assert_eq!(&auth[4..10], &bssid, "addr1 is the AP");
        assert_eq!(&auth[10..16], &mac, "addr2 is us");
        // Open system, transaction 1, status 0 — WPA2 proves the key in the handshake, not
        // here, and shared-key authentication would be refused.
        assert_eq!(&auth[24..30], &[0, 0, 1, 0, 0, 0]);

        // The sequence number sits in the top 12 bits of its field.
        let s = mgmt_header(FC_AUTH, &bssid, &mac, &bssid, 0x123);
        assert_eq!(u16::from_le_bytes([s[22], s[23]]), 0x123 << 4);

        let bss = Bss {
            ssid: "chitti-lan".into(),
            bssid,
            channel: 6,
            privacy: true,
            rsn: Rsn::parse(RSN_WPA2_PSK_CCMP),
            beacon_interval: 100,
        };
        let assoc = assoc_request(&bss, &mac, 1, 10);
        let body = &assoc[MGMT_HEADER_LEN + 4..]; // past capability + listen interval
        let ids: alloc::vec::Vec<u8> = elements(body).map(|(id, _)| id).collect();
        assert_eq!(
            ids,
            alloc::vec![ELEM_SSID, ELEM_SUPP_RATES, ELEM_EXT_SUPP_RATES, ELEM_RSN]
        );
        let ssid = elements(body).find(|(id, _)| *id == ELEM_SSID).unwrap().1;
        assert_eq!(ssid, b"chitti-lan");
    }

    #[test_case]
    fn our_rsn_element_is_the_one_the_handshake_will_confirm() {
        // Message 2 carries this element for the AP to compare against what we associated
        // with, so the two must be the same bytes — parsing our own is the cheap check that
        // they agree, and that what we claim is what we can do.
        let e = client_rsn_element();
        assert_eq!(e[0], ELEM_RSN);
        assert_eq!(e[1] as usize, e.len() - 2, "the length field must be right");
        let r = Rsn::parse(&e[2..]).expect("our own element must parse");
        assert!(r.supported(), "we advertise security we do not support");
        assert_eq!(&e[2..], RSN_WPA2_PSK_CCMP, "not the canonical WPA2-PSK element");
    }

    #[test_case]
    fn an_association_response_reports_its_status_and_masks_the_aid() {
        let mut f = mgmt_header(FC_ASSOC_RESP, &[0x02; 6], &[0xaa; 6], &[0xaa; 6], 1);
        f.extend_from_slice(&0x0431u16.to_le_bytes()); // capability
        f.extend_from_slice(&0u16.to_le_bytes()); // success
        f.extend_from_slice(&0xc003u16.to_le_bytes()); // AID 3, top two bits set
        let r = parse_assoc_response(&f).unwrap();
        assert_eq!(r.status, 0);
        assert_eq!(r.aid, 3, "the reserved top bits leaked into the AID");

        // A refusal carries the AP's reason code, which is worth showing the user verbatim.
        let n = f.len();
        f[n - 4..n - 2].copy_from_slice(&17u16.to_le_bytes()); // "association denied: too many"
        assert_eq!(parse_assoc_response(&f).unwrap().status, 17);

        for n in 0..f.len() {
            let _ = parse_assoc_response(&f[..n]); // must not panic
        }
    }
}
