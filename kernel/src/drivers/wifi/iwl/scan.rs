//! **`SCAN_REQ_UMAC` v17** — the Intel scan request.
//!
//! Layouts taken from Linux's `drivers/net/wireless/intel/iwlwifi/fw/api/scan.h`,
//! not from memory. That is the standing rule for this module and it was earned:
//! a struct recalled rather than read once put `n_hw_addrs` on the transmit
//! chain mask here — a small, plausible number that passes every sanity check.
//!
//! ## Why only v17
//!
//! The command id is stable and **the request structure is not**. It has been
//! rewritten repeatedly, and the firmware image states which layout it expects
//! in its own `IWL_UCODE_TLV_CMD_VERSIONS` table. [`super::scan_supported`]
//! consults that table and refuses anything not listed, so adding a version
//! here means adding its struct *and* listing it — never one without the other.
//!
//! ## The layout, and the two places it is not what it looks like
//!
//! ```text
//! iwl_scan_req_umac_v17          1940 bytes
//!   uid                     @0     4
//!   ooc_priority            @4     4
//!   general_params          @8    36   iwl_scan_general_params_v11
//!   channel_params          @44  540   iwl_scan_channel_params_v7
//!   periodic_params         @584  12   iwl_scan_periodic_parms_v1
//!   probe_params            @596 1344  iwl_scan_probe_params_v4
//! ```
//!
//! * **The band does not live in the band field.** `iwl_scan_channel_cfg_umac`
//!   ends in a union, and up to v16 its `v2` arm holds the band in a byte. At
//!   **v17 that byte became `psd_20`** and the band moved into bits 30-31 of the
//!   preceding `flags` word. Writing the band where the older layout kept it
//!   sets a power-spectral-density value instead and leaves the band at zero, so
//!   the radio scans 2.4 GHz whatever was asked for — a scan that succeeds and
//!   finds half the networks.
//!
//! * **The probe request is a template plus offsets into it.** `preq` is not a
//!   frame; it is three `(offset, len)` pairs describing where the MAC header,
//!   the per-band data and the common data sit inside a 512-byte buffer, so the
//!   firmware can splice a per-band rate element in without rebuilding the
//!   frame. Offsets that do not match the bytes make the firmware transmit a
//!   malformed probe, which access points ignore silently.

use alloc::vec::Vec;

// --- constants, from scan.h ----------------------------------------------

const SCAN_TWO_LMACS: usize = 2;
const SCAN_MAX_NUM_CHANS_V3: usize = 67;
const IWL_MAX_SCHED_SCAN_PLANS: usize = 2;
const PROBE_OPTION_MAX: usize = 20;
const SCAN_SHORT_SSID_MAX_SIZE: usize = 8;
const SCAN_BSSID_MAX_SIZE: usize = 16;
const SCAN_NUM_BAND_PROBE_DATA_V_2: usize = 3;
const SCAN_OFFLOAD_PROBE_REQ_SIZE: usize = 512;
const IEEE80211_MAX_SSID_LEN: usize = 32;
const ETH_ALEN: usize = 6;

/// The one request version implemented here.
pub const VERSION: u8 = 17;

// --- sizes, each the sum of the fields above it --------------------------

/// `iwl_ssid_ie`: id, len, 32-byte SSID.
const SSID_IE_LEN: usize = 2 + IEEE80211_MAX_SSID_LEN; // 34
/// `iwl_scan_probe_segment`: offset, len.
const PROBE_SEGMENT_LEN: usize = 4;
/// `iwl_scan_probe_req`.
const PROBE_REQ_LEN: usize =
    PROBE_SEGMENT_LEN * (1 + SCAN_NUM_BAND_PROBE_DATA_V_2 + 1) + SCAN_OFFLOAD_PROBE_REQ_SIZE; // 532
/// `iwl_scan_probe_params_v4`.
const PROBE_PARAMS_LEN: usize = PROBE_REQ_LEN
    + 1 // short_ssid_num
    + 1 // bssid_num
    + 2 // reserved
    + SSID_IE_LEN * PROBE_OPTION_MAX
    + 4 * SCAN_SHORT_SSID_MAX_SIZE
    + ETH_ALEN * SCAN_BSSID_MAX_SIZE; // 1344
/// `iwl_scan_umac_schedule`.
const SCHEDULE_LEN: usize = 4;
/// `iwl_scan_periodic_parms_v1`.
const PERIODIC_PARAMS_LEN: usize = SCHEDULE_LEN * IWL_MAX_SCHED_SCAN_PLANS + 2 + 2; // 12
/// `iwl_scan_channel_cfg_umac`: a flags word, a channel number, a 3-byte union.
const CHANNEL_CFG_LEN: usize = 8;
/// `iwl_scan_channel_params_v7`.
const CHANNEL_PARAMS_LEN: usize = 1 + 1 + 2 + CHANNEL_CFG_LEN * SCAN_MAX_NUM_CHANS_V3; // 540
/// `iwl_scan_general_params_v11`.
const GENERAL_PARAMS_LEN: usize = 2
    + 1
    + 1
    + SCAN_TWO_LMACS
    + 1
    + 1
    + 1
    + 1
    + 2
    + 4 * SCAN_TWO_LMACS
    + 4 * SCAN_TWO_LMACS
    + 4
    + SCAN_TWO_LMACS
    + SCAN_TWO_LMACS; // 36

// --- offsets into the request --------------------------------------------

pub const OFF_UID: usize = 0;
pub const OFF_OOC_PRIORITY: usize = 4;
pub const OFF_GENERAL: usize = 8;
pub const OFF_CHANNEL: usize = OFF_GENERAL + GENERAL_PARAMS_LEN; // 44
pub const OFF_PERIODIC: usize = OFF_CHANNEL + CHANNEL_PARAMS_LEN; // 584
pub const OFF_PROBE: usize = OFF_PERIODIC + PERIODIC_PARAMS_LEN; // 596
/// Total size of `iwl_scan_req_umac_v17`.
pub const REQ_LEN: usize = OFF_PROBE + PROBE_PARAMS_LEN; // 1940

// Offsets inside `iwl_scan_general_params_v11`, relative to `OFF_GENERAL`.
const G_FLAGS: usize = 0;
const G_SCAN_START_MAC: usize = 3;
const G_ACTIVE_DWELL: usize = 4;
const G_ADWELL_2G: usize = 6;
const G_ADWELL_5G: usize = 7;
const G_ADWELL_SOCIAL: usize = 8;
const G_FLAGS2: usize = 9;
const G_ADWELL_MAX_BUDGET: usize = 10;
const G_MAX_OUT_OF_TIME: usize = 12;
const G_SUSPEND_TIME: usize = 20;
const G_SCAN_PRIORITY: usize = 28;
const G_PASSIVE_DWELL: usize = 32;
const G_NUM_FRAGMENTS: usize = 34;

// Offsets inside `iwl_scan_channel_params_v7`.
const C_FLAGS: usize = 0;
const C_COUNT: usize = 1;
const C_N_APS_OVERRIDE: usize = 2;
const C_CONFIG: usize = 4;

// Offsets inside `iwl_scan_probe_params_v4`.
const P_PREQ: usize = 0;
const P_SHORT_SSID_NUM: usize = PROBE_REQ_LEN;
const P_BSSID_NUM: usize = PROBE_REQ_LEN + 1;
const P_DIRECT_SCAN: usize = PROBE_REQ_LEN + 4;

/// General flags (`enum iwl_umac_scan_general_flags`).
pub const GEN_FLAGS_PASS_ALL: u16 = 1 << 2;
pub const GEN_FLAGS_PASSIVE: u16 = 1 << 3;
pub const GEN_FLAGS_ITER_COMPLETE: u16 = 1 << 5;
pub const GEN_FLAGS_ADAPTIVE_DWELL: u16 = 1 << 13;

/// **Band lives in the flags word from v17**, bits 30-31 — not in the union
/// byte where v2 kept it. See the module docs.
const CHAN_CFG_FLAGS_BAND_POS: u32 = 30;

/// The 802.11 band a channel is on, in the firmware's numbering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Band {
    /// 2.4 GHz.
    Band2Ghz = 1,
    /// 5 GHz.
    Band5Ghz = 0,
    /// 6 GHz.
    Band6Ghz = 2,
}

/// One channel to visit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Channel {
    pub number: u8,
    pub band: Band,
    /// How many times to dwell on it. 1 for a single pass.
    pub iter_count: u8,
}

/// Write a `u16`/`u32` little-endian at `at`.
fn w16(b: &mut [u8], at: usize, v: u16) {
    b[at..at + 2].copy_from_slice(&v.to_le_bytes());
}
fn w32(b: &mut [u8], at: usize, v: u32) {
    b[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

/// Build a probe-request template: the frame bytes plus the three segment
/// descriptors the firmware needs to splice per-band data into it.
///
/// The firmware transmits `mac_header`, then the band data for whichever band
/// it is on, then `common_data` — so the *frame* here is deliberately split, and
/// the offsets are relative to `buf`.
///
/// Returns `(buf, mac_header, band_data, common_data)` where each descriptor is
/// `(offset, len)`.
pub fn probe_template(mac: &[u8; 6], ssid: Option<&[u8]>) -> (Vec<u8>, (u16, u16), (u16, u16)) {
    let mut buf = Vec::with_capacity(SCAN_OFFLOAD_PROBE_REQ_SIZE);
    // MAC header: probe request (type 0, subtype 4), broadcast destination and
    // BSSID — a wildcard probe.
    let mac_off = 0u16;
    buf.extend_from_slice(&0x0040u16.to_le_bytes()); // frame control
    buf.extend_from_slice(&[0, 0]); // duration
    buf.extend_from_slice(&[0xff; 6]); // A1 broadcast
    buf.extend_from_slice(mac); // A2 = us
    buf.extend_from_slice(&[0xff; 6]); // A3 broadcast BSSID
    buf.extend_from_slice(&[0, 0]); // sequence control
    let mac_len = buf.len() as u16;

    // Common data: the SSID element (empty for a wildcard scan). The firmware
    // inserts the band-specific rate elements between the header and this.
    let common_off = buf.len() as u16;
    buf.push(0); // element id: SSID
    match ssid {
        Some(s) if s.len() <= IEEE80211_MAX_SSID_LEN => {
            buf.push(s.len() as u8);
            buf.extend_from_slice(s);
        }
        _ => buf.push(0), // wildcard
    }
    let common_len = buf.len() as u16 - common_off;

    buf.resize(SCAN_OFFLOAD_PROBE_REQ_SIZE, 0);
    (buf, (mac_off, mac_len), (common_off, common_len))
}

/// Build a complete `SCAN_REQ_UMAC` v17 request.
///
/// `uid` identifies the scan in the completion notification. `channels` must be
/// non-empty and is truncated to what the structure holds — silently scanning
/// fewer channels than asked is better than overrunning the array, but the
/// caller is told by the returned count.
pub fn build_v17(
    uid: u32,
    mac: &[u8; 6],
    channels: &[Channel],
    passive: bool,
) -> Option<(Vec<u8>, usize)> {
    if channels.is_empty() {
        return None;
    }
    let n = channels.len().min(SCAN_MAX_NUM_CHANS_V3);
    let mut b = alloc::vec![0u8; REQ_LEN];

    w32(&mut b, OFF_UID, uid);
    // Out-of-channel priority: the lowest, so a scan never pre-empts traffic.
    w32(&mut b, OFF_OOC_PRIORITY, 0);

    // --- general params ---
    let g = OFF_GENERAL;
    let mut flags = GEN_FLAGS_PASS_ALL | GEN_FLAGS_ITER_COMPLETE | GEN_FLAGS_ADAPTIVE_DWELL;
    if passive {
        flags |= GEN_FLAGS_PASSIVE;
    }
    w16(&mut b, g + G_FLAGS, flags);
    b[g + G_SCAN_START_MAC] = 0;
    // Dwell times in TU (1024 µs). These are the values Linux uses for an
    // adaptive-dwell scan; they are policy, not layout, and a wrong one costs
    // scan time rather than correctness.
    b[g + G_ACTIVE_DWELL] = 10;
    b[g + G_ACTIVE_DWELL + 1] = 10;
    b[g + G_ADWELL_2G] = 20;
    b[g + G_ADWELL_5G] = 10;
    b[g + G_ADWELL_SOCIAL] = 10;
    b[g + G_FLAGS2] = 0;
    w16(&mut b, g + G_ADWELL_MAX_BUDGET, 200);
    for i in 0..SCAN_TWO_LMACS {
        w32(&mut b, g + G_MAX_OUT_OF_TIME + i * 4, 0);
        w32(&mut b, g + G_SUSPEND_TIME + i * 4, 0);
        b[g + G_PASSIVE_DWELL + i] = 110;
        b[g + G_NUM_FRAGMENTS + i] = 0;
    }
    w32(&mut b, g + G_SCAN_PRIORITY, 6); // IWL_SCAN_PRIORITY_EXT_6

    // --- channel params ---
    let c = OFF_CHANNEL;
    b[c + C_FLAGS] = 0;
    b[c + C_COUNT] = n as u8;
    b[c + C_N_APS_OVERRIDE] = 0;
    b[c + C_N_APS_OVERRIDE + 1] = 0;
    for (i, ch) in channels.iter().take(n).enumerate() {
        let at = c + C_CONFIG + i * CHANNEL_CFG_LEN;
        // **The band goes in the flags word at v17**, not in the union byte.
        let band = (ch.band as u32) << CHAN_CFG_FLAGS_BAND_POS;
        w32(&mut b, at, band);
        b[at + 4] = ch.number;
        // v5 union arm: psd_20, iter_count, iter_interval. `psd_20` is where v2
        // kept the band; leaving it zero means "unknown", which is correct.
        b[at + 5] = 0; // psd_20
        b[at + 6] = ch.iter_count.max(1);
        b[at + 7] = 0; // iter_interval
    }

    // --- periodic params --- a single immediate pass: one plan, one iteration.
    let p = OFF_PERIODIC;
    w16(&mut b, p, 0); // schedule[0].interval
    b[p + 2] = 1; // schedule[0].iter_count
    w16(&mut b, p + SCHEDULE_LEN * IWL_MAX_SCHED_SCAN_PLANS, 0); // delay

    // --- probe params ---
    let pr = OFF_PROBE;
    let (buf, mac_seg, common_seg) = probe_template(mac, None);
    // preq: mac_header, band_data[3], common_data, then the buffer.
    w16(&mut b, pr + P_PREQ, mac_seg.0);
    w16(&mut b, pr + P_PREQ + 2, mac_seg.1);
    // band_data stays zeroed: the firmware fills in the per-band rate elements.
    let common_at = pr + P_PREQ + PROBE_SEGMENT_LEN * (1 + SCAN_NUM_BAND_PROBE_DATA_V_2);
    w16(&mut b, common_at, common_seg.0);
    w16(&mut b, common_at + 2, common_seg.1);
    let buf_at = pr + P_PREQ + PROBE_SEGMENT_LEN * (1 + SCAN_NUM_BAND_PROBE_DATA_V_2 + 1);
    b[buf_at..buf_at + SCAN_OFFLOAD_PROBE_REQ_SIZE].copy_from_slice(&buf);
    // A wildcard scan names no SSIDs and no BSSIDs.
    b[pr + P_SHORT_SSID_NUM] = 0;
    b[pr + P_BSSID_NUM] = 0;
    // direct_scan[0] is the wildcard entry: an SSID element of length zero.
    b[pr + P_DIRECT_SCAN] = 0;
    b[pr + P_DIRECT_SCAN + 1] = 0;

    Some((b, n))
}

/// The default channel set for a 2.4 GHz + 5 GHz scan.
///
/// Channels 1-13 plus the common 5 GHz set. 14 is omitted (Japan only, and
/// scanning it where it is not permitted is a regulatory matter, not a
/// capability one); DFS channels are omitted because probing them actively is
/// forbidden until radar-free operation is established, and this builds an
/// active scan.
pub fn default_channels() -> Vec<Channel> {
    let mut v = Vec::new();
    for n in 1..=13u8 {
        v.push(Channel {
            number: n,
            band: Band::Band2Ghz,
            iter_count: 1,
        });
    }
    for n in [36u8, 40, 44, 48, 149, 153, 157, 161, 165] {
        v.push(Channel {
            number: n,
            band: Band::Band5Ghz,
            iter_count: 1,
        });
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Every offset and size, pinned.** This is the whole testable surface of
    /// a command that cannot be sent here: if the arithmetic is wrong the
    /// firmware reads our fields as different ones and reports nothing, so the
    /// layout is checked against the numbers in `scan.h` rather than against
    /// behaviour.
    #[test_case]
    fn the_v17_layout_matches_the_header() {
        assert_eq!(SSID_IE_LEN, 34);
        assert_eq!(PROBE_REQ_LEN, 532, "3 segments + 3 band + 512 buffer");
        assert_eq!(PROBE_PARAMS_LEN, 1344);
        assert_eq!(PERIODIC_PARAMS_LEN, 12);
        assert_eq!(CHANNEL_PARAMS_LEN, 540, "4 + 67 * 8");
        assert_eq!(GENERAL_PARAMS_LEN, 36);

        assert_eq!(OFF_UID, 0);
        assert_eq!(OFF_OOC_PRIORITY, 4);
        assert_eq!(OFF_GENERAL, 8);
        assert_eq!(OFF_CHANNEL, 44);
        assert_eq!(OFF_PERIODIC, 584);
        assert_eq!(OFF_PROBE, 596);
        assert_eq!(REQ_LEN, 1940, "iwl_scan_req_umac_v17");

        // Every sub-offset stays inside its own struct.
        assert!(G_NUM_FRAGMENTS + SCAN_TWO_LMACS <= GENERAL_PARAMS_LEN);
        assert!(C_CONFIG + CHANNEL_CFG_LEN * SCAN_MAX_NUM_CHANS_V3 <= CHANNEL_PARAMS_LEN);
        assert!(P_DIRECT_SCAN + SSID_IE_LEN * PROBE_OPTION_MAX <= PROBE_PARAMS_LEN);
    }

    /// **The band is in bits 30-31 of the flags word, not in the union byte.**
    /// Up to v16 the `v2` arm held it in the byte that v17 renamed `psd_20`.
    /// Writing it there sets a power-spectral-density value and leaves the band
    /// at zero — so the radio scans 2.4 GHz whatever was asked for, and the scan
    /// succeeds while finding half the networks.
    #[test_case]
    fn the_band_lives_in_the_flags_word_at_v17() {
        let chans = [
            Channel {
                number: 6,
                band: Band::Band2Ghz,
                iter_count: 1,
            },
            Channel {
                number: 36,
                band: Band::Band5Ghz,
                iter_count: 1,
            },
            Channel {
                number: 5,
                band: Band::Band6Ghz,
                iter_count: 2,
            },
        ];
        let (b, n) = build_v17(1, &[2; 6], &chans, false).unwrap();
        assert_eq!(n, 3);
        assert_eq!(b[OFF_CHANNEL + C_COUNT], 3);

        for (i, ch) in chans.iter().enumerate() {
            let at = OFF_CHANNEL + C_CONFIG + i * CHANNEL_CFG_LEN;
            let flags = u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]);
            assert_eq!(
                flags >> CHAN_CFG_FLAGS_BAND_POS,
                ch.band as u32,
                "band must be in the flags word"
            );
            assert_eq!(b[at + 4], ch.number, "channel number");
            assert_eq!(b[at + 5], 0, "psd_20, NOT the band");
            assert_eq!(b[at + 6], ch.iter_count.max(1));
        }
    }

    /// The probe request is a **template plus offsets into it**, not a frame.
    /// Offsets that do not match the bytes make the firmware transmit a
    /// malformed probe, which access points ignore silently.
    #[test_case]
    fn the_probe_template_offsets_point_at_the_real_bytes() {
        let mac = [0x02, 0x11, 0x22, 0x33, 0x44, 0x55];
        let (buf, mac_seg, common_seg) = probe_template(&mac, None);
        assert_eq!(buf.len(), SCAN_OFFLOAD_PROBE_REQ_SIZE, "fixed-size buffer");
        assert_eq!(mac_seg.0, 0);
        assert_eq!(mac_seg.1, 24, "a probe request MAC header");
        // The header really is a probe request from us to the broadcast address.
        assert_eq!(
            u16::from_le_bytes([buf[0], buf[1]]),
            0x0040,
            "probe request"
        );
        assert_eq!(&buf[4..10], &[0xff; 6], "broadcast destination");
        assert_eq!(&buf[10..16], &mac[..], "our address");
        // Common data is the SSID element, and a wildcard scan's is empty.
        let (o, l) = (common_seg.0 as usize, common_seg.1 as usize);
        assert_eq!(o, 24);
        assert_eq!(l, 2, "id + zero length");
        assert_eq!(&buf[o..o + l], &[0, 0]);

        // A directed probe carries the name.
        let (buf, _, common) = probe_template(&mac, Some(b"chitti-lan"));
        let (o, l) = (common.0 as usize, common.1 as usize);
        assert_eq!(l, 12, "id + len + 10 characters");
        assert_eq!(&buf[o + 2..o + l], b"chitti-lan");

        // An over-long SSID falls back to a wildcard rather than overrunning.
        let (_, _, c) = probe_template(&mac, Some(&[b'x'; 40]));
        assert_eq!(c.1, 2);
    }

    /// The request embeds the template where the struct says, and the segment
    /// descriptors precede it.
    #[test_case]
    fn the_request_embeds_the_template_at_the_right_offset() {
        let mac = [0x06; 6];
        let (b, _) = build_v17(0x1234, &mac, &default_channels(), false).unwrap();
        assert_eq!(u32::from_le_bytes([b[0], b[1], b[2], b[3]]), 0x1234, "uid");

        let pr = OFF_PROBE;
        assert_eq!(
            u16::from_le_bytes([b[pr], b[pr + 1]]),
            0,
            "mac_header offset"
        );
        assert_eq!(
            u16::from_le_bytes([b[pr + 2], b[pr + 3]]),
            24,
            "mac_header len"
        );
        // The buffer follows five segment descriptors (header + 3 bands + common).
        let buf_at = pr + PROBE_SEGMENT_LEN * 5;
        assert_eq!(
            u16::from_le_bytes([b[buf_at], b[buf_at + 1]]),
            0x0040,
            "the frame is there"
        );
        assert_eq!(&b[buf_at + 10..buf_at + 16], &mac[..], "with our address");
        // Wildcard: no named SSIDs, no BSSIDs.
        assert_eq!(b[pr + P_SHORT_SSID_NUM], 0);
        assert_eq!(b[pr + P_BSSID_NUM], 0);
    }

    /// The flags say what kind of scan this is, and an active scan must not set
    /// the passive bit — a passive scan never transmits a probe, so a hidden
    /// network stays hidden and the dwell is much longer.
    #[test_case]
    fn active_and_passive_scans_differ_only_in_a_flag() {
        let ch = default_channels();
        let (a, _) = build_v17(1, &[0; 6], &ch, false).unwrap();
        let (p, _) = build_v17(1, &[0; 6], &ch, true).unwrap();
        let fa = u16::from_le_bytes([a[OFF_GENERAL], a[OFF_GENERAL + 1]]);
        let fp = u16::from_le_bytes([p[OFF_GENERAL], p[OFF_GENERAL + 1]]);
        assert_eq!(fa & GEN_FLAGS_PASSIVE, 0);
        assert_ne!(fp & GEN_FLAGS_PASSIVE, 0);
        assert_eq!(fa ^ fp, GEN_FLAGS_PASSIVE, "nothing else changes");
        // Both ask for every result and for a completion notification.
        assert_ne!(fa & GEN_FLAGS_PASS_ALL, 0);
        assert_ne!(fa & GEN_FLAGS_ITER_COMPLETE, 0);
    }

    /// More channels than the array holds are truncated, and the caller is told
    /// how many were actually written — overrunning would corrupt the probe
    /// parameters that follow.
    #[test_case]
    fn too_many_channels_are_truncated_not_overrun() {
        let many: Vec<Channel> = (0..200)
            .map(|i| Channel {
                number: (i % 200) as u8,
                band: Band::Band2Ghz,
                iter_count: 1,
            })
            .collect();
        let (b, n) = build_v17(1, &[0; 6], &many, false).unwrap();
        assert_eq!(n, SCAN_MAX_NUM_CHANS_V3, "the caller learns the real count");
        assert_eq!(b[OFF_CHANNEL + C_COUNT], SCAN_MAX_NUM_CHANS_V3 as u8);
        assert_eq!(
            b.len(),
            REQ_LEN,
            "and the request is still exactly one struct"
        );
        // An empty list is refused rather than sent as a scan of nothing.
        assert!(build_v17(1, &[0; 6], &[], false).is_none());
    }

    /// The default set avoids channel 14 and the DFS range: probing those
    /// actively is a regulatory matter, not a capability one.
    #[test_case]
    fn the_default_channel_set_is_regulatory_safe() {
        let ch = default_channels();
        assert!(ch.iter().all(|c| c.number != 14), "14 is Japan-only");
        // No DFS channels (52-144) in the 5 GHz part.
        assert!(
            !ch.iter()
                .any(|c| c.band == Band::Band5Ghz && (52..=144).contains(&c.number)),
            "DFS channels must not be probed actively"
        );
        assert!(ch.len() <= SCAN_MAX_NUM_CHANS_V3);
    }
}
