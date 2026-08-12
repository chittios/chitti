//! **`iwl_rx_mpdu_desc`** — the descriptor in front of every received frame.
//!
//! `REPLY_RX_MPDU` hands back a descriptor and then the 802.11 frame, and the
//! descriptor's **length is not fixed**: it grew a union arm between hardware
//! generations. Getting it wrong does not fail — the frame is read starting a
//! few bytes inside itself, so `parse_beacon` sees plausible field values and
//! returns a network with a wrong BSSID and a garbage SSID. Nothing rejects it.
//!
//! Layout from Linux's `fw/api/rx.h`, not from memory. The struct's own offset 0
//! is what the header's comments call DW2; the DW numbering is the device's and
//! is not an offset into the structure, which is the first thing to get wrong
//! when reading that file.
//!
//! ```text
//! common                         20 bytes
//!   mpdu_len          @0    2    the 802.11 frame's length
//!   mac_flags1        @2    1
//!   mac_flags2        @3    1
//!   amsdu_info        @4    1
//!   phy_info          @5    2
//!   mac_phy_band      @7    1
//!   raw_csum/l3l4     @8    4
//!   status            @12   4
//!   reorder_data      @16   4
//! then one of:
//!   v1  @20  28 bytes  -> descriptor is 48 bytes   (9000, AX200/AX201)
//!   v3  @20  44 bytes  -> descriptor is 64 bytes   (AX210 and later)
//! ```
//!
//! ## The two things that are silently wrong
//!
//! * **Which arm is present is decided by the hardware family**, not by
//!   anything in the frame. AX210 and later use v3; everything older uses v1.
//!   There is no marker to check, so [`desc_len`] is the single place that
//!   decision is made and it is driven by [`fw::Family`].
//!
//! * **`energy_a`/`energy_b` are magnitudes, and zero means "no reading"**, not
//!   0 dBm. The signal is `-max(a, b)` over the antennas that reported, and an
//!   antenna reporting zero must be excluded rather than treated as the
//!   strongest — which it would be, being numerically the largest.

use super::fw;

/// `REPLY_RX_MPDU_CMD`, legacy group.
pub const REPLY_RX_MPDU: u8 = 0xc1;

/// Offsets into the common part.
pub const OFF_MPDU_LEN: usize = 0;
pub const OFF_MAC_FLAGS1: usize = 2;
pub const OFF_PHY_INFO: usize = 5;
pub const OFF_STATUS: usize = 12;
pub const OFF_REORDER: usize = 16;
/// Where the version-dependent union begins.
pub const OFF_UNION: usize = 20;

/// `iwl_rx_mpdu_desc_v1` is 28 bytes, so the descriptor is 48.
pub const DESC_LEN_V1: usize = 48;
/// `iwl_rx_mpdu_desc_v3` is 44 bytes, so the descriptor is 64.
pub const DESC_LEN_V3: usize = 64;

/// Offsets of `energy_a` within each union arm, relative to the descriptor.
///
/// v1 puts `rate_n_flags` at +8 and the energies at +12; v3 inserts
/// `partial_hash` and the checksum words first, pushing them to +20.
const V1_ENERGY: usize = OFF_UNION + 12; // 32
const V3_ENERGY: usize = OFF_UNION + 20; // 40

/// Status bits worth acting on (`enum iwl_rx_mpdu_status`).
pub const STATUS_CRC_OK: u32 = 1 << 0;
pub const STATUS_OVERRUN_OK: u32 = 1 << 1;
pub const STATUS_DECRYPTED: u32 = 1 << 11;

/// How long the descriptor is on this hardware.
///
/// **The only place this decision is made.** There is no marker in the frame to
/// check — the arm present is a property of the silicon, so a family added to
/// [`fw::Family`] must be classified here or it silently inherits the wrong
/// length.
pub fn desc_len(family: fw::Family) -> usize {
    match family {
        // AX210 and later carry the longer descriptor.
        fw::Family::Ax210 | fw::Family::Be200 => DESC_LEN_V3,
        fw::Family::Iwl7000 | fw::Family::Iwl8000 | fw::Family::Iwl9000 | fw::Family::Ax200 => {
            DESC_LEN_V1
        }
    }
}

/// Signal strength in dBm from the two antenna energies.
///
/// **Zero means an antenna did not report**, not zero dBm. Treating it as a
/// reading makes it the strongest of the pair every time — it is numerically
/// the largest — so a frame heard on one antenna reports as a perfect signal.
pub fn signal_dbm(energy_a: u8, energy_b: u8) -> Option<i8> {
    let a = (energy_a != 0).then(|| -(energy_a as i16));
    let b = (energy_b != 0).then(|| -(energy_b as i16));
    let best = match (a, b) {
        (Some(x), Some(y)) => x.max(y),
        (Some(x), None) | (None, Some(x)) => x,
        (None, None) => return None,
    };
    Some(best.clamp(-128, 127) as i8)
}

/// A received frame, located inside a `REPLY_RX_MPDU` payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mpdu<'a> {
    /// The 802.11 frame itself.
    pub frame: &'a [u8],
    /// Signal strength, or `None` when no antenna reported.
    pub rssi: Option<i8>,
    /// Channel the frame arrived on.
    pub channel: u8,
    /// Raw status word, for callers that care about decryption results.
    pub status: u32,
}

/// Split a `REPLY_RX_MPDU` payload into its descriptor and frame.
///
/// Returns `None` when the payload is too short for the descriptor this family
/// uses, when the declared frame length runs past the buffer, or when the frame
/// failed its CRC — a corrupt frame is dropped rather than parsed, because a
/// beacon with a flipped bit yields a network entry that looks real.
pub fn parse(payload: &[u8], family: fw::Family) -> Option<Mpdu<'_>> {
    let dlen = desc_len(family);
    if payload.len() < dlen {
        return None;
    }
    let d = &payload[..dlen];
    let mpdu_len = u16::from_le_bytes([d[OFF_MPDU_LEN], d[OFF_MPDU_LEN + 1]]) as usize;
    let status = u32::from_le_bytes([
        d[OFF_STATUS],
        d[OFF_STATUS + 1],
        d[OFF_STATUS + 2],
        d[OFF_STATUS + 3],
    ]);
    // A frame the hardware says is corrupt must not become a network.
    if status & STATUS_CRC_OK == 0 {
        return None;
    }
    let frame = payload.get(dlen..dlen + mpdu_len)?;
    let e = if dlen == DESC_LEN_V3 {
        V3_ENERGY
    } else {
        V1_ENERGY
    };
    Some(Mpdu {
        frame,
        rssi: signal_dbm(d[e], d[e + 1]),
        channel: d[e + 2],
        status,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The descriptor length is the whole risk.** A wrong one reads the frame
    /// starting a few bytes inside itself, and `parse_beacon` then returns a
    /// plausible network with a wrong BSSID rather than failing — so the sizes
    /// are pinned against `rx.h`'s field list.
    #[test_case]
    fn the_descriptor_length_matches_the_header() {
        assert_eq!(OFF_UNION, 20, "the common part is 20 bytes");
        assert_eq!(DESC_LEN_V1, 48, "20 + 28");
        assert_eq!(DESC_LEN_V3, 64, "20 + 44");
        assert_eq!(V1_ENERGY, 32);
        assert_eq!(V3_ENERGY, 40);

        // AX210 and later are the long descriptor; everything older is short.
        assert_eq!(desc_len(fw::Family::Ax210), DESC_LEN_V3);
        assert_eq!(desc_len(fw::Family::Be200), DESC_LEN_V3);
        assert_eq!(
            desc_len(fw::Family::Ax200),
            DESC_LEN_V1,
            "AX200 is 22000-family, not AX210"
        );
        assert_eq!(desc_len(fw::Family::Iwl9000), DESC_LEN_V1);
        assert_eq!(desc_len(fw::Family::Iwl8000), DESC_LEN_V1);
        assert_eq!(desc_len(fw::Family::Iwl7000), DESC_LEN_V1);
    }

    /// Build a payload: descriptor of `dlen` bytes then `frame`.
    fn payload(
        dlen: usize,
        frame: &[u8],
        energy: (u8, u8),
        channel: u8,
        crc_ok: bool,
    ) -> alloc::vec::Vec<u8> {
        let mut p = alloc::vec![0u8; dlen];
        p[OFF_MPDU_LEN..OFF_MPDU_LEN + 2].copy_from_slice(&(frame.len() as u16).to_le_bytes());
        let status: u32 = if crc_ok { STATUS_CRC_OK } else { 0 };
        p[OFF_STATUS..OFF_STATUS + 4].copy_from_slice(&status.to_le_bytes());
        let e = if dlen == DESC_LEN_V3 {
            V3_ENERGY
        } else {
            V1_ENERGY
        };
        p[e] = energy.0;
        p[e + 1] = energy.1;
        p[e + 2] = channel;
        p.extend_from_slice(frame);
        p
    }

    /// The frame comes out whole, from the right offset, on both descriptor
    /// versions — and the energies are read from the arm that is actually
    /// present, which sits eight bytes further along in v3.
    #[test_case]
    fn the_frame_is_found_after_the_right_descriptor() {
        let frame: alloc::vec::Vec<u8> = (0..40u8).collect();
        for (family, dlen) in [
            (fw::Family::Ax200, DESC_LEN_V1),
            (fw::Family::Ax210, DESC_LEN_V3),
        ] {
            let p = payload(dlen, &frame, (60, 70), 11, true);
            let m = parse(&p, family).expect("parses");
            assert_eq!(
                m.frame,
                &frame[..],
                "the frame, whole and from the right offset"
            );
            assert_eq!(m.channel, 11);
            assert_eq!(m.rssi, Some(-60), "the stronger antenna: -min(60,70)");
        }

        // Reading a v3 payload as v1 takes the frame eight bytes early — the
        // failure this module exists to prevent. It does not error; it returns
        // the *wrong* bytes, which is why the family must decide.
        let p = payload(DESC_LEN_V3, &frame, (60, 70), 11, true);
        let wrong = parse(&p, fw::Family::Ax200).expect("still 'parses'");
        assert_ne!(wrong.frame, &frame[..], "silently wrong, not an error");
    }

    /// **Energy zero means an antenna did not report**, not 0 dBm. Treated as a
    /// reading it wins every comparison — it is numerically the largest — so a
    /// frame heard on one antenna would report as a perfect signal.
    #[test_case]
    fn a_silent_antenna_is_not_a_perfect_signal() {
        assert_eq!(signal_dbm(60, 70), Some(-60), "stronger of the two");
        assert_eq!(signal_dbm(0, 70), Some(-70), "antenna A silent");
        assert_eq!(signal_dbm(60, 0), Some(-60), "antenna B silent");
        assert_eq!(signal_dbm(0, 0), None, "neither reported");
        // The trap, stated as an assertion: if zero were a reading, this would
        // be Some(0) — a full-strength signal from a silent antenna.
        assert_ne!(signal_dbm(0, 90), Some(0));
        assert_eq!(signal_dbm(0, 90), Some(-90));
        // A very weak signal is still a reading, distinct from none.
        assert_eq!(signal_dbm(128, 0), Some(-128));
    }

    /// A corrupt frame is dropped rather than parsed: a beacon with a flipped
    /// bit yields a network entry that looks entirely real.
    #[test_case]
    fn a_frame_that_failed_its_crc_is_dropped() {
        let frame = alloc::vec![0xaau8; 30];
        let bad = payload(DESC_LEN_V1, &frame, (50, 50), 6, false);
        assert_eq!(parse(&bad, fw::Family::Ax200), None);
        let good = payload(DESC_LEN_V1, &frame, (50, 50), 6, true);
        assert!(parse(&good, fw::Family::Ax200).is_some());
    }

    /// A declared length running past the buffer is refused — these bytes came
    /// off the air and the length is the device's report of an attacker's frame.
    #[test_case]
    fn a_length_past_the_buffer_is_refused() {
        let frame = alloc::vec![0u8; 10];
        let mut p = payload(DESC_LEN_V1, &frame, (50, 50), 1, true);
        p[OFF_MPDU_LEN..OFF_MPDU_LEN + 2].copy_from_slice(&4000u16.to_le_bytes());
        assert_eq!(parse(&p, fw::Family::Ax200), None);
        // And a payload too short to hold the descriptor at all.
        assert_eq!(parse(&[0u8; 20], fw::Family::Ax200), None);
        assert_eq!(parse(&[], fw::Family::Ax210), None);
        // Exactly the descriptor with a zero-length frame is legal and empty.
        let empty = payload(DESC_LEN_V1, &[], (0, 0), 1, true);
        assert_eq!(parse(&empty, fw::Family::Ax200).unwrap().frame.len(), 0);
    }
}
