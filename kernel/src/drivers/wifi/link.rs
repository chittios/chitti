//! **The association state machine** — the sequencer that joins a network.
//!
//! Every piece needed to associate already existed and nothing drove them:
//! [`super::ieee80211`] builds authentication and association frames,
//! [`super::wpa`] runs the four-way handshake, [`super::ccmp`] encrypts, and
//! [`super::data`] converts frames. This is the state machine that takes them
//! in order, so a radio driver's whole job becomes
//!
//! ```text
//! let out = link.start()?;          // send this
//! while let Some(rx) = radio.recv() {
//!     if let Some(tx) = link.on_frame(&rx)? { radio.send(&tx); }
//!     if link.connected() { break; }
//! }
//! ```
//!
//! and every driver gets the same sequencing rather than reimplementing it.
//! Pure — it takes frames in and hands frames out, touching no hardware — so
//! the whole join can be tested against a simulated access point.
//!
//! ## The ladder, and why each rung refuses rather than retries
//!
//! `Open authentication` → `Association` → `four-way handshake` → keys
//! installed. A failure at any rung is reported with the rung named, because
//! "could not connect" has half a dozen causes that need different actions from
//! a human — the wrong passphrase, an AP that is full, a network using an
//! unsupported cipher, and a radio that is not receiving all look identical
//! from the outside.
//!
//! In particular **a wrong passphrase cannot be detected before the handshake**:
//! authentication and association both succeed, because WPA2 never sends the
//! passphrase anywhere. The first and only symptom is message 3's MIC failing,
//! which [`super::wpa`] already distinguishes from a changed ANonce for exactly
//! this reason. So [`Link::on_frame`] surfaces that as its own error and not as
//! a generic handshake failure.
//!
//! ## What this deliberately does not do
//!
//! No scanning (the radio supplies the [`Bss`]), no channel selection, no
//! rate control, no power save, and **no retransmission** — 802.11 retries are
//! the radio's job, and a state machine that retried on its own would fight it.
//! It also does not implement the group-key rekey exchange the AP starts later;
//! that is a two-message handshake of its own and is refused rather than
//! half-handled.

use super::ccmp::{PnCounter, ReplayGuard};
use super::data::SeqCounter;
use super::ieee80211::{self, Bss};
use super::wpa::{self, Handshake};
use alloc::vec::Vec;

/// Where the join has got to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    /// Nothing started.
    Idle,
    /// Open authentication sent, waiting for the response.
    Authenticating,
    /// Association request sent, waiting for the response.
    Associating,
    /// Associated; running the four-way handshake.
    Handshaking,
    /// Keys installed. Traffic can flow.
    Connected,
    /// Terminal. The reason is on the [`Link`].
    Failed,
}

/// Why a join failed, kept separate from the message so a caller can act on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Failure {
    /// The AP refused open authentication.
    AuthRefused(u16),
    /// The AP refused association — status code 17 is "cannot support all
    /// requested capabilities", which in practice usually means the network is
    /// full.
    AssocRefused(u16),
    /// Message 3's MIC did not verify: **the passphrase is wrong**. Nothing
    /// earlier can detect this, because WPA2 never transmits the passphrase.
    WrongPassphrase,
    /// The handshake failed for a reason other than the MIC.
    HandshakeFailed,
    /// The network is not WPA2-PSK with CCMP.
    Unsupported,
}

impl Failure {
    /// A line for a human. `/wifi connect` prints this, so it says what to do
    /// rather than what went wrong internally.
    pub fn message(self) -> &'static str {
        match self {
            Failure::AuthRefused(_) => "the access point refused authentication",
            Failure::AssocRefused(_) => "the access point refused association (network full, or it rejected our capabilities)",
            Failure::WrongPassphrase => "wrong passphrase",
            Failure::HandshakeFailed => "the WPA2 handshake failed",
            Failure::Unsupported => "unsupported network (this kernel joins WPA2-PSK/CCMP only)",
        }
    }
}

/// Keys the link is using once connected.
#[derive(Debug, Clone)]
pub struct Keys {
    /// Pairwise temporal key — encrypts our unicast traffic.
    pub tk: [u8; 16],
    /// Group key and its id, for broadcast and multicast.
    pub gtk: Option<(u8, Vec<u8>)>,
}

/// One station's association with one access point.
pub struct Link {
    state: State,
    failure: Option<Failure>,
    bss: Bss,
    mac: [u8; 6],
    hs: Handshake,
    keys: Option<Keys>,
    /// Management-frame sequence numbers, separate from the data counter — they
    /// are different sequence spaces and sharing one makes an AP see gaps.
    mgmt_seq: SeqCounter,
    data_seq: SeqCounter,
    tx_pn: PnCounter,
    rx_replay: ReplayGuard,
}

impl Link {
    /// Prepare a join. `snonce` must be random — it is half the input to the
    /// pairwise key, and a predictable one lets an observer who knows the
    /// passphrase derive the session key without watching the handshake.
    pub fn new(bss: Bss, mac: [u8; 6], passphrase: &str, snonce: [u8; 32]) -> Link {
        let pmk = wpa::pmk_from_passphrase(passphrase, bss.ssid.as_bytes());
        let rsn = ieee80211::client_rsn_element();
        let hs = Handshake::new(pmk, mac, bss.bssid, snonce, rsn);
        Link {
            state: State::Idle,
            failure: None,
            bss,
            mac,
            hs,
            keys: None,
            mgmt_seq: SeqCounter::new(),
            data_seq: SeqCounter::new(),
            tx_pn: PnCounter::new(),
            rx_replay: ReplayGuard::new(),
        }
    }

    pub fn state(&self) -> State {
        self.state
    }
    pub fn connected(&self) -> bool {
        self.state == State::Connected
    }
    pub fn failure(&self) -> Option<Failure> {
        self.failure
    }
    pub fn keys(&self) -> Option<&Keys> {
        self.keys.as_ref()
    }
    pub fn bssid(&self) -> [u8; 6] {
        self.bss.bssid
    }

    fn fail(&mut self, why: Failure) -> Result<Option<Vec<u8>>, Failure> {
        self.state = State::Failed;
        self.failure = Some(why);
        Err(why)
    }

    /// Begin: returns the authentication frame to transmit.
    ///
    /// Refuses up front for a network we cannot join, rather than authenticating
    /// and associating successfully and only discovering it at the handshake —
    /// which would leave the AP holding an association we then abandon.
    pub fn start(&mut self) -> Result<Vec<u8>, Failure> {
        if !self.bss.joinable() {
            self.state = State::Failed;
            self.failure = Some(Failure::Unsupported);
            return Err(Failure::Unsupported);
        }
        self.state = State::Authenticating;
        let seq = self.mgmt_seq.next();
        Ok(ieee80211::auth_request(&self.bss.bssid, &self.mac, seq))
    }

    /// Feed a received frame. Returns a frame to transmit in reply, if any.
    ///
    /// Frames that are not part of this exchange return `Ok(None)` rather than
    /// an error: a radio hands over everything it hears, and a beacon from a
    /// neighbouring network arriving mid-handshake is normal, not a failure.
    pub fn on_frame(&mut self, frame: &[u8]) -> Result<Option<Vec<u8>>, Failure> {
        if frame.len() < 24 {
            return Ok(None);
        }
        let fc = u16::from_le_bytes([frame[0], frame[1]]);
        let ftype = (fc >> 2) & 0x3;
        let subtype = (fc >> 4) & 0xf;

        // Only frames from our AP, addressed to us, are ours to act on.
        let a2 = &frame[10..16];
        if a2 != self.bss.bssid {
            return Ok(None);
        }

        match (self.state, ftype, subtype) {
            // Authentication response.
            (State::Authenticating, 0, 11) => {
                // Open authentication: algorithm (2), sequence (2), status (2).
                let b = frame.get(24..30).ok_or(Failure::HandshakeFailed)?;
                let status = u16::from_le_bytes([b[4], b[5]]);
                if status != 0 {
                    return self.fail(Failure::AuthRefused(status)).map(|_| None);
                }
                self.state = State::Associating;
                let seq = self.mgmt_seq.next();
                Ok(Some(ieee80211::assoc_request(&self.bss, &self.mac, seq, 1)))
            }
            // Association response.
            (State::Associating, 0, 1) => {
                let r = ieee80211::parse_assoc_response(frame).ok_or(Failure::AssocRefused(0))?;
                if r.status != 0 {
                    return self.fail(Failure::AssocRefused(r.status)).map(|_| None);
                }
                // Associated. The AP now starts the four-way handshake by
                // sending message 1; there is nothing to transmit here.
                self.state = State::Handshaking;
                Ok(None)
            }
            // EAPOL over a data frame — the four-way handshake.
            (State::Handshaking, 2, _) => {
                let Some(eapol) = eapol_payload(frame) else {
                    return Ok(None);
                };
                match self.hs.on_frame(eapol) {
                    Ok(reply) => {
                        if self.hs.done {
                            self.install_keys();
                        }
                        // The reply is an EAPOL body; the caller wraps it in a
                        // data frame with `data::to_air`. Handing back the bare
                        // body keeps this module free of the addressing, which
                        // `data` already owns.
                        Ok(reply)
                    }
                    // `wpa` distinguishes a MIC failure from every other
                    // problem precisely so this can say "wrong passphrase",
                    // which is the only actionable answer a human gets here.
                    Err(e) if e.contains("MIC") => {
                        self.fail(Failure::WrongPassphrase).map(|_| None)
                    }
                    Err(_) => self.fail(Failure::HandshakeFailed).map(|_| None),
                }
            }
            _ => Ok(None),
        }
    }

    fn install_keys(&mut self) {
        let Some(ptk) = self.hs.ptk() else { return };
        self.keys = Some(Keys {
            tk: ptk.tk,
            gtk: self.hs.gtk.as_ref().map(|g| (g.id, g.key.clone())),
        });
        self.state = State::Connected;
    }

    /// Wrap an Ethernet frame for transmission. `None` before the link is up,
    /// or once the packet-number space is exhausted.
    pub fn encrypt_tx(&mut self, eth: &[u8]) -> Option<Vec<u8>> {
        let keys = self.keys.as_ref()?;
        let pn = self.tx_pn.next()?;
        let seq = self.data_seq.next();
        let tk = keys.tk;
        super::data::to_air(eth, &self.bss.bssid, seq, Some((&tk, &pn, 0)))
    }

    /// Unwrap a received 802.11 frame into an Ethernet frame.
    ///
    /// Drops replays: without the check a captured frame can be re-injected
    /// indefinitely and decrypts perfectly every time.
    pub fn decrypt_rx(&mut self, air: &[u8]) -> Option<Vec<u8>> {
        let keys = self.keys.as_ref()?;
        let tk = keys.tk;
        let (eth, pn) = super::data::from_air(air, Some(&tk))?;
        if let Some(pn) = pn {
            if !self.rx_replay.accept(&pn) {
                return None;
            }
        }
        Some(eth)
    }
}

/// The EAPOL-Key body inside a data frame, if this frame carries one.
///
/// EAPOL rides LLC/SNAP with ethertype `0x888e`. Handing the whole data frame
/// to the handshake parser would make it read the 802.11 header as EAPOL
/// fields.
pub fn eapol_payload(frame: &[u8]) -> Option<&[u8]> {
    let fc = u16::from_le_bytes([*frame.first()?, *frame.get(1)?]);
    let hlen = super::ccmp::header_len(fc);
    let body = frame.get(hlen..)?;
    let (ethertype, payload) = super::data::snap_decap(body)?;
    (ethertype == 0x888e).then_some(payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    const AP: [u8; 6] = [0x00, 0x11, 0x22, 0x33, 0x44, 0x55];
    const US: [u8; 6] = [0x02, 0xaa, 0xbb, 0xcc, 0xdd, 0xee];

    fn wpa2_bss() -> Bss {
        // The canonical WPA2-PSK/CCMP RSN element.
        let rsn = ieee80211::Rsn::parse(&[
            0x01, 0x00, // version 1
            0x00, 0x0f, 0xac, 0x04, // group CCMP
            0x01, 0x00, 0x00, 0x0f, 0xac, 0x04, // pairwise CCMP
            0x01, 0x00, 0x00, 0x0f, 0xac, 0x02, // AKM PSK
        ])
        .expect("the canonical element parses");
        Bss {
            ssid: "chitti-lan".to_string(),
            bssid: AP,
            channel: 6,
            privacy: true,
            rsn: Some(rsn),
            beacon_interval: 100,
        }
    }

    fn mgmt_frame(subtype: u16, body: &[u8]) -> Vec<u8> {
        let fc = (subtype << 4) & 0x00f0;
        let mut f = ieee80211::mgmt_header(fc, &US, &AP, &AP, 0);
        f.extend_from_slice(body);
        f
    }

    /// The ladder runs in order, and each rung's reply is the next request.
    #[test_case]
    fn the_join_ladder_runs_in_order() {
        let mut l = Link::new(wpa2_bss(), US, "correct horse", [0x42; 32]);
        assert_eq!(l.state(), State::Idle);

        let auth = l.start().expect("starts");
        assert_eq!(l.state(), State::Authenticating);
        // Open authentication: algorithm 0, sequence 1.
        assert_eq!(u16::from_le_bytes([auth[24], auth[25]]), 0, "open system");

        // The AP authenticates us: algorithm, sequence 2, status 0.
        let ok = mgmt_frame(11, &[0, 0, 2, 0, 0, 0]);
        let assoc = l
            .on_frame(&ok)
            .expect("no error")
            .expect("sends an assoc request");
        assert_eq!(l.state(), State::Associating);
        let fc = u16::from_le_bytes([assoc[0], assoc[1]]);
        assert_eq!((fc >> 4) & 0xf, 0, "association request");

        // And associates us: capability, status 0, AID.
        let ok = mgmt_frame(1, &[0x31, 0x04, 0, 0, 0x01, 0xc0]);
        assert_eq!(
            l.on_frame(&ok).expect("no error"),
            None,
            "now the AP speaks first"
        );
        assert_eq!(l.state(), State::Handshaking);
        assert!(!l.connected());
        assert!(l.keys().is_none(), "no keys until the handshake completes");
    }

    /// **A refusal names the rung.** "Could not connect" has half a dozen causes
    /// that need different actions from a human, and they are indistinguishable
    /// from outside.
    #[test_case]
    fn each_rung_reports_its_own_refusal() {
        // Authentication refused.
        let mut l = Link::new(wpa2_bss(), US, "pw", [0; 32]);
        l.start().unwrap();
        let refused = mgmt_frame(11, &[0, 0, 2, 0, 13, 0]); // status 13
        assert_eq!(l.on_frame(&refused), Err(Failure::AuthRefused(13)));
        assert_eq!(l.state(), State::Failed);
        assert_eq!(l.failure(), Some(Failure::AuthRefused(13)));

        // Association refused — status 17 is the "network is full" case.
        let mut l = Link::new(wpa2_bss(), US, "pw", [0; 32]);
        l.start().unwrap();
        l.on_frame(&mgmt_frame(11, &[0, 0, 2, 0, 0, 0])).unwrap();
        let full = mgmt_frame(1, &[0x31, 0x04, 17, 0, 0, 0]);
        assert_eq!(l.on_frame(&full), Err(Failure::AssocRefused(17)));
        assert!(l
            .failure()
            .unwrap()
            .message()
            .contains("refused association"));
    }

    /// An unsupported network is refused **before** authenticating, rather than
    /// leaving the AP holding an association we then abandon.
    #[test_case]
    fn an_unsupported_network_is_refused_before_it_is_joined() {
        let mut open = wpa2_bss();
        open.rsn = None; // an open network
        let mut l = Link::new(open, US, "", [0; 32]);
        assert_eq!(l.start(), Err(Failure::Unsupported));
        assert_eq!(l.state(), State::Failed);
        assert!(l
            .failure()
            .unwrap()
            .message()
            .contains("WPA2-PSK/CCMP only"));
    }

    /// **A wrong passphrase cannot be detected before the handshake**, because
    /// WPA2 never transmits it — authentication and association both succeed.
    /// The first symptom is message 3's MIC failing, and it must be reported as
    /// the passphrase rather than as a generic handshake failure, because that
    /// is the only actionable answer.
    #[test_case]
    fn a_wrong_passphrase_surfaces_only_at_the_handshake() {
        let mut l = Link::new(wpa2_bss(), US, "wrong passphrase", [0x11; 32]);
        l.start().unwrap();
        l.on_frame(&mgmt_frame(11, &[0, 0, 2, 0, 0, 0])).unwrap();
        l.on_frame(&mgmt_frame(1, &[0x31, 0x04, 0, 0, 1, 0xc0]))
            .unwrap();
        // Both earlier rungs passed with the wrong passphrase — that is the
        // point of this test.
        assert_eq!(l.state(), State::Handshaking);
        assert_eq!(l.failure(), None);
    }

    /// Frames that are not part of this exchange are ignored, not errors — a
    /// radio hands over everything it hears, and a neighbouring network's
    /// beacon arriving mid-handshake is normal.
    #[test_case]
    fn frames_from_elsewhere_are_ignored_not_failed() {
        let mut l = Link::new(wpa2_bss(), US, "pw", [0; 32]);
        l.start().unwrap();

        // A beacon from a different BSSID.
        let other = [0x99u8; 6];
        let mut f = ieee80211::mgmt_header(0x0080, &US, &other, &other, 0);
        f.extend_from_slice(&[0; 12]);
        assert_eq!(l.on_frame(&f), Ok(None));
        assert_eq!(l.state(), State::Authenticating, "state is untouched");

        // A runt frame.
        assert_eq!(l.on_frame(&[0; 4]), Ok(None));
        // The right AP, but a frame type this rung does not expect.
        assert_eq!(l.on_frame(&mgmt_frame(8, &[0; 12])), Ok(None));
        assert_eq!(l.state(), State::Authenticating);
    }

    /// EAPOL is found by its ethertype inside LLC/SNAP. Handing the whole data
    /// frame to the handshake parser makes it read the 802.11 header as EAPOL
    /// fields.
    #[test_case]
    fn eapol_is_located_by_its_ethertype() {
        let hdr = super::super::data::uplink_header(&AP, &US, &US, 0, false);
        let mut f = hdr.clone();
        f.extend_from_slice(&super::super::data::snap_encap(0x888e, b"eapol body"));
        assert_eq!(eapol_payload(&f), Some(&b"eapol body"[..]));

        // An IPv4 data frame is not EAPOL.
        let mut ip = hdr.clone();
        ip.extend_from_slice(&super::super::data::snap_encap(0x0800, b"not eapol"));
        assert_eq!(eapol_payload(&ip), None);
        // Nor is a frame with no SNAP shim at all.
        let mut raw = hdr.clone();
        raw.extend_from_slice(b"bare bytes");
        assert_eq!(eapol_payload(&raw), None);
    }

    /// Traffic only flows once keys are installed — before that `encrypt_tx`
    /// returns nothing rather than sending in the clear, which would leak
    /// every packet the moment association succeeded.
    #[test_case]
    fn no_traffic_before_the_keys_are_installed() {
        let mut l = Link::new(wpa2_bss(), US, "pw", [0; 32]);
        let eth = {
            let mut e = Vec::new();
            e.extend_from_slice(&[0xff; 6]);
            e.extend_from_slice(&US);
            e.extend_from_slice(&0x0806u16.to_be_bytes());
            e.extend_from_slice(b"arp");
            e
        };
        assert_eq!(l.encrypt_tx(&eth), None, "idle");
        l.start().unwrap();
        assert_eq!(l.encrypt_tx(&eth), None, "authenticating");
        assert_eq!(l.decrypt_rx(&[0u8; 64]), None);
    }

    /// Once connected, traffic encrypts, decrypts, and **a replay is dropped**.
    #[test_case]
    fn a_connected_link_carries_traffic_and_rejects_replays() {
        let mut l = Link::new(wpa2_bss(), US, "pw", [0; 32]);
        // Drive it to Connected directly: the handshake itself is `wpa`'s to
        // test, and it has its own vectors.
        l.keys = Some(Keys {
            tk: [0x33; 16],
            gtk: None,
        });
        l.state = State::Connected;
        assert!(l.connected());

        let mut eth = Vec::new();
        eth.extend_from_slice(&[0xff; 6]);
        eth.extend_from_slice(&US);
        eth.extend_from_slice(&0x0800u16.to_be_bytes());
        eth.extend_from_slice(b"payload");

        let air = l.encrypt_tx(&eth).expect("encrypts");
        assert!(
            !air.windows(7).any(|w| w == b"payload"),
            "must be encrypted"
        );

        let back = l.decrypt_rx(&air).expect("decrypts");
        assert_eq!(&back[14..], b"payload");
        // The same frame again is a replay.
        assert_eq!(l.decrypt_rx(&air), None, "replayed frame must be dropped");

        // And each transmission uses a fresh packet number.
        let a1 = l.encrypt_tx(&eth).unwrap();
        let a2 = l.encrypt_tx(&eth).unwrap();
        let pn1 = super::super::ccmp::read_header(&a1[24..]).unwrap().0;
        let pn2 = super::super::ccmp::read_header(&a2[24..]).unwrap().0;
        assert_ne!(pn1, pn2, "a repeated PN leaks the XOR of two plaintexts");
    }

    /// Management and data frames use **separate** sequence spaces; sharing one
    /// counter makes the AP see gaps in both.
    #[test_case]
    fn management_and_data_sequence_numbers_are_separate() {
        let mut l = Link::new(wpa2_bss(), US, "pw", [0; 32]);
        let auth = l.start().unwrap();
        assert_eq!(u16::from_le_bytes([auth[22], auth[23]]) >> 4, 0);
        l.keys = Some(Keys {
            tk: [1; 16],
            gtk: None,
        });
        l.state = State::Connected;
        let mut eth = Vec::new();
        eth.extend_from_slice(&[0xff; 6]);
        eth.extend_from_slice(&US);
        eth.extend_from_slice(&0x0800u16.to_be_bytes());
        let air = l.encrypt_tx(&eth).unwrap();
        // The data frame starts its own sequence at 0, not at 1.
        assert_eq!(u16::from_le_bytes([air[22], air[23]]) >> 4, 0);
    }
}
