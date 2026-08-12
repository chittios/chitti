//! **Scan results** — turning the beacons a radio hears into a network list.
//!
//! A surprise worth recording, because it changes what "implement scan" means
//! on every chipset: **the scan command does not return the networks.** Intel's
//! `iwl_scan_results_notif` carries a channel, a band, a probe status and a
//! duration — no BSSID, no SSID, no signal. It is a report on the *scan*, not on
//! what was found.
//!
//! The networks arrive separately, as ordinary beacon and probe-response frames
//! coming up the normal receive path, which is exactly what
//! [`super::ieee80211::parse_beacon`] already decodes. So a driver's scan is:
//! start the scan, feed every management frame here until the scan-complete
//! notification, then read the list.
//!
//! That makes this layer driver-agnostic, which is the point — Intel, Broadcom,
//! Realtek and MediaTek all produce the same beacons.
//!
//! ## Why aggregation is not just a `Vec`
//!
//! An access point beacons about ten times a second, and a scan that dwells on
//! a channel hears the same network repeatedly; a busy office produces hundreds
//! of frames from a dozen networks. Appending them gives a list that is mostly
//! duplicates and whose order is the order the radio happened to hear them.
//!
//! Worse, a single network commonly has **several BSSIDs** — one per radio per
//! access point — all advertising the same SSID. Deduplicating by SSID would
//! collapse them and lose the ability to pick the strongest; deduplicating by
//! BSSID keeps them distinct, which is correct, and leaves the *presentation*
//! free to group by SSID.

use super::ieee80211::Bss;
use alloc::vec::Vec;

/// One network heard during a scan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub bss: Bss,
    /// Strongest signal seen, in dBm. `None` when the radio reported none —
    /// distinct from a very weak signal, which is why it is not simply -128.
    pub rssi: Option<i8>,
    /// How many beacons or probe responses this BSSID contributed.
    pub seen: u32,
}

/// Accumulates beacons into a deduplicated network list.
#[derive(Debug, Default)]
pub struct Scan {
    entries: Vec<Entry>,
}

impl Scan {
    pub fn new() -> Scan {
        Scan {
            entries: Vec::new(),
        }
    }

    /// Feed a management frame. Returns true when it was a beacon or probe
    /// response that parsed.
    ///
    /// `rssi` is whatever the radio's receive metadata reported for this frame.
    pub fn on_frame(&mut self, frame: &[u8], rssi: Option<i8>) -> bool {
        if frame.len() < 24 {
            return false;
        }
        let fc = u16::from_le_bytes([frame[0], frame[1]]);
        // Management frames only, and only beacon (8) or probe response (5).
        if (fc >> 2) & 0x3 != 0 {
            return false;
        }
        let subtype = (fc >> 4) & 0xf;
        if subtype != 8 && subtype != 5 {
            return false;
        }
        let Some(bss) = super::ieee80211::parse_beacon(frame) else {
            return false;
        };
        self.add(bss, rssi);
        true
    }

    /// Merge one parsed beacon into the list.
    fn add(&mut self, bss: Bss, rssi: Option<i8>) {
        if let Some(e) = self.entries.iter_mut().find(|e| e.bss.bssid == bss.bssid) {
            e.seen = e.seen.saturating_add(1);
            // Keep the **strongest** reading, not the latest: a scan sweeps
            // channels and antenna conditions vary frame to frame, so the last
            // one is arbitrary while the best approximates what a connection
            // would get.
            e.rssi = match (e.rssi, rssi) {
                (Some(a), Some(b)) => Some(a.max(b)),
                (a, b) => a.or(b),
            };
            // A later beacon may carry an SSID a hidden-network probe response
            // did not, and may carry the channel a beacon omitted. Prefer the
            // more informative of the two rather than the more recent.
            if e.bss.ssid.is_empty() && !bss.ssid.is_empty() {
                e.bss.ssid = bss.ssid;
            }
            if e.bss.channel == 0 && bss.channel != 0 {
                e.bss.channel = bss.channel;
            }
            return;
        }
        self.entries.push(Entry { bss, rssi, seen: 1 });
    }

    /// The networks found, strongest first.
    ///
    /// A network with no signal reading sorts last rather than first: `None`
    /// means the radio did not say, and treating that as maximum strength would
    /// put unknown networks above measured ones.
    pub fn results(&self) -> Vec<Entry> {
        let mut v = self.entries.clone();
        v.sort_by(|a, b| match (b.rssi, a.rssi) {
            (Some(x), Some(y)) => x.cmp(&y),
            (Some(_), None) => core::cmp::Ordering::Greater,
            (None, Some(_)) => core::cmp::Ordering::Less,
            (None, None) => core::cmp::Ordering::Equal,
        });
        v
    }

    /// Only the networks this kernel can actually join (WPA2-PSK/CCMP).
    pub fn joinable(&self) -> Vec<Entry> {
        self.results()
            .into_iter()
            .filter(|e| e.bss.joinable())
            .collect()
    }

    /// Find a network by SSID, strongest first — what `/wifi connect <ssid>`
    /// needs, since a human names an SSID and an SSID may have several BSSIDs.
    pub fn best_for_ssid(&self, ssid: &str) -> Option<Entry> {
        self.results().into_iter().find(|e| e.bss.ssid == ssid)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    /// A beacon for `ssid` from `bssid`, optionally WPA2.
    fn beacon(ssid: &[u8], bssid: [u8; 6], channel: u8, wpa2: bool) -> Vec<u8> {
        let mut f = super::super::ieee80211::mgmt_header(0x0080, &[0xff; 6], &bssid, &bssid, 0);
        f.extend_from_slice(&[0; 8]); // timestamp
        f.extend_from_slice(&100u16.to_le_bytes()); // beacon interval
        f.extend_from_slice(&(if wpa2 { 0x0011u16 } else { 0x0001 }).to_le_bytes()); // capability
                                                                                     // SSID element.
        f.push(0);
        f.push(ssid.len() as u8);
        f.extend_from_slice(ssid);
        // DS Parameter Set — the channel.
        f.extend_from_slice(&[3, 1, channel]);
        if wpa2 {
            let rsn: &[u8] = &[
                0x01, 0x00, 0x00, 0x0f, 0xac, 0x04, 0x01, 0x00, 0x00, 0x0f, 0xac, 0x04, 0x01, 0x00,
                0x00, 0x0f, 0xac, 0x02,
            ];
            f.push(48);
            f.push(rsn.len() as u8);
            f.extend_from_slice(rsn);
        }
        f
    }

    /// **An access point beacons about ten times a second.** Appending every
    /// frame gives a list that is mostly duplicates, in the arbitrary order the
    /// radio happened to hear them.
    #[test_case]
    fn repeated_beacons_collapse_to_one_entry() {
        let mut s = Scan::new();
        let b = beacon(b"home", [1, 2, 3, 4, 5, 6], 6, true);
        for _ in 0..20 {
            assert!(s.on_frame(&b, Some(-50)));
        }
        assert_eq!(s.len(), 1, "twenty beacons, one network");
        let r = s.results();
        assert_eq!(r[0].seen, 20);
        assert_eq!(r[0].bss.ssid, "home");
        assert_eq!(r[0].bss.channel, 6);
    }

    /// **One network commonly has several BSSIDs** — one per radio per access
    /// point. Deduplicating by SSID would collapse them and lose the ability to
    /// pick the strongest; by BSSID keeps them distinct, which is correct.
    #[test_case]
    fn the_same_ssid_on_two_radios_stays_two_entries() {
        let mut s = Scan::new();
        s.on_frame(&beacon(b"office", [0, 0, 0, 0, 0, 1], 1, true), Some(-70));
        s.on_frame(&beacon(b"office", [0, 0, 0, 0, 0, 2], 36, true), Some(-45));
        assert_eq!(s.len(), 2, "two BSSIDs, two entries");
        // And picking by SSID gives the stronger of the two.
        let best = s.best_for_ssid("office").unwrap();
        assert_eq!(best.rssi, Some(-45));
        assert_eq!(best.bss.channel, 36);
    }

    /// The **strongest** reading is kept, not the latest: a scan sweeps
    /// channels and conditions vary frame to frame, so the last is arbitrary.
    #[test_case]
    fn the_strongest_signal_wins_not_the_most_recent() {
        let mut s = Scan::new();
        let b = beacon(b"net", [9; 6], 11, true);
        s.on_frame(&b, Some(-80));
        s.on_frame(&b, Some(-40));
        s.on_frame(&b, Some(-75)); // later, but weaker
        assert_eq!(s.results()[0].rssi, Some(-40));
    }

    /// Results are strongest first, and a network with **no** reading sorts
    /// last — `None` means the radio did not say, and treating it as maximum
    /// strength would put unknown networks above measured ones.
    #[test_case]
    fn results_are_ordered_by_signal_with_unknowns_last() {
        let mut s = Scan::new();
        s.on_frame(&beacon(b"weak", [0, 0, 0, 0, 0, 1], 1, true), Some(-85));
        s.on_frame(&beacon(b"strong", [0, 0, 0, 0, 0, 2], 1, true), Some(-35));
        s.on_frame(&beacon(b"unknown", [0, 0, 0, 0, 0, 3], 1, true), None);
        s.on_frame(&beacon(b"mid", [0, 0, 0, 0, 0, 4], 1, true), Some(-60));
        let names: Vec<_> = s.results().into_iter().map(|e| e.bss.ssid).collect();
        assert_eq!(names, ["strong", "mid", "weak", "unknown"]);
    }

    /// A later beacon can carry information an earlier frame lacked — a hidden
    /// network's probe response has the SSID its beacon omits. Prefer the more
    /// informative frame, not the more recent one.
    #[test_case]
    fn a_later_frame_fills_in_what_an_earlier_one_omitted() {
        let mut s = Scan::new();
        // A hidden network: zero-length SSID, and no DS element so no channel.
        let mut hidden =
            super::super::ieee80211::mgmt_header(0x0080, &[0xff; 6], &[7; 6], &[7; 6], 0);
        hidden.extend_from_slice(&[0; 8]);
        hidden.extend_from_slice(&100u16.to_le_bytes());
        hidden.extend_from_slice(&0x0011u16.to_le_bytes());
        hidden.extend_from_slice(&[0, 0]); // empty SSID element
        s.on_frame(&hidden, Some(-60));
        assert_eq!(s.results()[0].bss.ssid, "");
        assert_eq!(s.results()[0].bss.channel, 0);

        // A probe response naming it.
        let mut probe = beacon(b"secret", [7; 6], 3, true);
        probe[0..2].copy_from_slice(&0x0050u16.to_le_bytes()); // probe response
        s.on_frame(&probe, Some(-62));
        let r = s.results();
        assert_eq!(r.len(), 1, "still one network");
        assert_eq!(r[0].bss.ssid, "secret", "the name is filled in");
        assert_eq!(r[0].bss.channel, 3);
        assert_eq!(r[0].rssi, Some(-60), "but the stronger reading is kept");
    }

    /// Only beacons and probe responses count. Every other frame the radio
    /// hands over — data, control, other management subtypes — is not a network
    /// advertisement and must not become one.
    #[test_case]
    fn only_beacons_and_probe_responses_are_networks() {
        let mut s = Scan::new();
        let good = beacon(b"real", [1; 6], 1, true);
        assert!(s.on_frame(&good, None));

        // A data frame.
        let mut data = good.clone();
        data[0..2].copy_from_slice(&0x0008u16.to_le_bytes());
        assert!(!s.on_frame(&data, None));
        // An association response (management, subtype 1).
        let mut assoc = good.clone();
        assoc[0..2].copy_from_slice(&0x0010u16.to_le_bytes());
        assert!(!s.on_frame(&assoc, None));
        // Runts.
        assert!(!s.on_frame(&[], None));
        assert!(!s.on_frame(&good[..20], None));
        assert_eq!(s.len(), 1, "only the beacon counted");
    }

    /// The joinable filter is what `/wifi connect` needs: an open or WPA3
    /// network is visible in a scan and cannot be joined by this kernel.
    #[test_case]
    fn joinable_filters_to_wpa2_psk_ccmp() {
        let mut s = Scan::new();
        s.on_frame(&beacon(b"secured", [0, 0, 0, 0, 0, 1], 1, true), Some(-40));
        s.on_frame(&beacon(b"open", [0, 0, 0, 0, 0, 2], 1, false), Some(-30));
        assert_eq!(s.len(), 2, "both are visible");
        let j = s.joinable();
        assert_eq!(j.len(), 1);
        assert_eq!(j[0].bss.ssid, "secured", "the open network is not joinable");
    }
}
