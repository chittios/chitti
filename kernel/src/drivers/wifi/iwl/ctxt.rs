//! **`PHY_CONTEXT_CMD`** — binding the radio to a channel.
//!
//! The first of the context commands an association needs: a PHY context says
//! *which channel on which band at what width*, and everything above it (the
//! MAC context, the binding, the station) refers to it by id.
//!
//! Layouts from Linux's `fw/api/phy-ctxt.h` and `fw/api/context.h`.
//!
//! ```text
//! iwl_phy_context_cmd            32 bytes
//!   id_and_color        @0    4
//!   action              @4    4
//!   ci                  @8    8   iwl_fw_channel_info
//!   lmac_id             @16   4
//!   rxchain_info        @20   4
//!   dsp_cfg_flags       @24   4
//!   secondary_ctrl_loc  @28   1
//!   reserved            @29   3
//! ```
//!
//! ## Two values that are not what you would guess
//!
//! * **`PHY_BAND_5` is 0 and `PHY_BAND_24` is 1.** The numbering does not follow
//!   the frequencies, so the obvious "2.4 GHz is band 0" leaves the radio tuned
//!   to a 5 GHz channel number on the 2.4 GHz band — a context that is accepted
//!   and hears nothing.
//!
//! * **`FW_CTXT_ACTION_ADD` is 1, not 0**; 0 is `INVALID`. Sending a
//!   zero-initialised action asks the firmware to do nothing valid, and a
//!   command built by zeroing a buffer and filling only the fields you care
//!   about lands there by default.
//!
//! Note also that `iwl_fw_channel_info` puts the **channel first as a `u32`**,
//! then the band — the reverse of the older `_v1`, which led with a byte band.
//! A driver written against the old order writes the channel number into the
//! band field.

use alloc::vec::Vec;

/// `PHY_CONTEXT_CMD`, legacy group.
pub const PHY_CONTEXT_CMD: u8 = 0x08;

/// `iwl_phy_context_cmd` — 32 bytes.
pub const PHY_CTXT_LEN: usize = 32;
/// `iwl_fw_channel_info` — 8 bytes.
pub const CHANNEL_INFO_LEN: usize = 8;

// Offsets.
const P_ID_AND_COLOR: usize = 0;
const P_ACTION: usize = 4;
const P_CI: usize = 8;
const P_LMAC_ID: usize = 16;
const P_RXCHAIN_INFO: usize = 20;
const P_DSP_CFG_FLAGS: usize = 24;
const P_SECONDARY_CTRL: usize = 28;

// Inside `iwl_fw_channel_info`.
const CI_CHANNEL: usize = 0; // __le32
const CI_BAND: usize = 4;
const CI_WIDTH: usize = 5;
const CI_CTRL_POS: usize = 6;

/// `enum iwl_ctxt_action`. **ADD is 1** — zero is `INVALID`.
pub const ACTION_INVALID: u32 = 0;
pub const ACTION_ADD: u32 = 1;
pub const ACTION_MODIFY: u32 = 2;
pub const ACTION_REMOVE: u32 = 3;

/// `FW_CTXT_COLOR_POS` — the colour sits above the id in the identifier word.
pub const CTXT_COLOR_POS: u32 = 8;

/// **The band numbering does not follow the frequencies.**
pub const PHY_BAND_5: u8 = 0;
pub const PHY_BAND_24: u8 = 1;
pub const PHY_BAND_6: u8 = 2;

/// Channel width.
pub const CHANNEL_MODE20: u8 = 0x0;
pub const CHANNEL_MODE40: u8 = 0x1;
pub const CHANNEL_MODE80: u8 = 0x2;
pub const CHANNEL_MODE160: u8 = 0x3;

/// Control-channel position for a 20 MHz channel: no offset.
pub const CTRL_POS_20MHZ: u8 = 0;

/// Pack a context identifier: id in the low byte, colour above it.
///
/// Shared with [`super::sta::mac_id_n_color`] — `FW_CTXT_COLOR_POS` is 8 for
/// every context type, so the two agree by construction rather than by
/// coincidence.
pub fn id_and_color(id: u8, color: u8) -> u32 {
    (id as u32) | ((color as u32) << CTXT_COLOR_POS)
}

/// The firmware's band number for an 802.11 channel.
///
/// Derived from the channel rather than taken as a parameter, because the two
/// must agree: a 5 GHz channel number on the 2.4 GHz band is a context the
/// firmware accepts and that hears nothing.
pub fn band_for_channel(channel: u8) -> u8 {
    if channel <= 14 {
        PHY_BAND_24
    } else {
        PHY_BAND_5
    }
}

/// Build `PHY_CONTEXT_CMD` for a 20 MHz channel.
///
/// 20 MHz only: a wider channel needs the control-position field to say where
/// the primary sits within it, and getting that wrong tunes the radio beside
/// the network rather than on it. Every network is reachable at 20 MHz, so the
/// narrow case is the honest one to implement first.
pub fn phy_context_20mhz(id: u8, color: u8, channel: u8, action: u32) -> Vec<u8> {
    let mut b = alloc::vec![0u8; PHY_CTXT_LEN];
    w32(&mut b, P_ID_AND_COLOR, id_and_color(id, color));
    w32(&mut b, P_ACTION, action);
    // `iwl_fw_channel_info`: the channel is a **u32 and comes first**; the older
    // `_v1` led with a byte band, so the old order writes the channel number
    // into the band field.
    w32(&mut b, P_CI + CI_CHANNEL, channel as u32);
    b[P_CI + CI_BAND] = band_for_channel(channel);
    b[P_CI + CI_WIDTH] = CHANNEL_MODE20;
    b[P_CI + CI_CTRL_POS] = CTRL_POS_20MHZ;
    w32(&mut b, P_LMAC_ID, 0);
    // Receive chains: use both antennas for both valid and active. Zero here
    // configures a context with no receive chain, which associates and then
    // hears nothing.
    let chains = 0x3;
    w32(&mut b, P_RXCHAIN_INFO, (chains << 1) | (chains << 4));
    w32(&mut b, P_DSP_CFG_FLAGS, 0);
    b[P_SECONDARY_CTRL] = 0;
    b
}

fn w32(b: &mut [u8], at: usize, v: u32) {
    b[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

// --- MAC_CONTEXT_CMD ------------------------------------------------------
//
// The interface itself: our address, the BSSID we are joining, the rates, the
// filters, and — once associated — the association id and beacon timing.

/// `MAC_CONTEXT_CMD`, legacy group.
pub const MAC_CONTEXT_CMD: u8 = 0x28;

/// `iwl_ac_qos` — 8 bytes, five of them (`AC_NUM + 1`).
const AC_QOS_LEN: usize = 8;
const AC_COUNT: usize = 5;

/// Offset of the interface-type union inside `iwl_mac_ctx_cmd`.
pub const MAC_UNION_OFF: usize = 100;

/// `iwl_mac_ctx_cmd` — 148 bytes.
///
/// **The size is the common part plus the union's *largest* arm**, not plus the
/// arm being used. Linux sends `sizeof(*cmd)` whatever the interface type, and
/// the largest is `iwl_mac_data_p2p_sta` (an `iwl_mac_data_sta` plus a
/// `ctwin`), at 48. Sizing the command to the 44-byte station arm sends four
/// bytes too few and the firmware rejects a command that is otherwise correct.
pub const MAC_CTXT_LEN: usize = MAC_UNION_OFF + 48;

// Offsets into the common part.
const M_ID_AND_COLOR: usize = 0;
const M_ACTION: usize = 4;
const M_MAC_TYPE: usize = 8;
const M_TSF_ID: usize = 12;
const M_NODE_ADDR: usize = 16;
const M_BSSID_ADDR: usize = 24;
const M_CCK_RATES: usize = 32;
const M_OFDM_RATES: usize = 36;
const M_PROTECTION_FLAGS: usize = 40;
const M_CCK_SHORT_PREAMBLE: usize = 44;
const M_SHORT_SLOT: usize = 48;
const M_FILTER_FLAGS: usize = 52;
const M_QOS_FLAGS: usize = 56;
const M_AC: usize = 60;

// Offsets inside the `iwl_mac_data_sta` arm, relative to `MAC_UNION_OFF`.
const STA_IS_ASSOC: usize = 0;
const STA_DTIM_TIME: usize = 4;
const STA_DTIM_TSF: usize = 8;
const STA_BI: usize = 16;
const STA_DTIM_INTERVAL: usize = 24;
const STA_DATA_POLICY: usize = 28;
const STA_LISTEN_INTERVAL: usize = 32;
const STA_ASSOC_ID: usize = 36;

/// `enum iwl_mac_types`. The enum starts at 1, so a zeroed `mac_type` is not a
/// type at all.
pub const MAC_TYPE_BSS_STA: u32 = 5;

/// `enum iwl_mac_filter_flags`.
pub const MAC_FILTER_ACCEPT_GRP: u32 = 1 << 2;
pub const MAC_FILTER_IN_BEACON: u32 = 1 << 6;
pub const MAC_FILTER_IN_PROBE_REQUEST: u32 = 1 << 12;

/// Build `MAC_CONTEXT_CMD` for a client interface.
///
/// `assoc` is `None` before association and `Some((aid, beacon_interval,
/// dtim_interval))` after — the same command carries both states, which is why
/// it is sent twice during a join.
pub fn mac_context_sta(
    id: u8,
    color: u8,
    our_mac: &[u8; 6],
    bssid: &[u8; 6],
    action: u32,
    assoc: Option<(u16, u16, u8)>,
) -> Vec<u8> {
    let mut b = alloc::vec![0u8; MAC_CTXT_LEN];
    w32(&mut b, M_ID_AND_COLOR, id_and_color(id, color));
    w32(&mut b, M_ACTION, action);
    w32(&mut b, M_MAC_TYPE, MAC_TYPE_BSS_STA);
    w32(&mut b, M_TSF_ID, id as u32);
    b[M_NODE_ADDR..M_NODE_ADDR + 6].copy_from_slice(our_mac);
    b[M_BSSID_ADDR..M_BSSID_ADDR + 6].copy_from_slice(bssid);
    // Rate masks: the 4 CCK rates and the 8 OFDM rates, all of them. A zero
    // mask is a station that may transmit at no rate, which associates and then
    // sends nothing.
    w32(&mut b, M_CCK_RATES, 0x0f);
    w32(&mut b, M_OFDM_RATES, 0xff);
    w32(&mut b, M_PROTECTION_FLAGS, 0);
    w32(&mut b, M_CCK_SHORT_PREAMBLE, 0);
    w32(&mut b, M_SHORT_SLOT, 0);
    // Accept group-addressed frames (ARP and DHCP replies arrive that way) and
    // beacons (the scan and the DTIM timing both need them). Without
    // ACCEPT_GRP the link comes up and never resolves an address.
    w32(
        &mut b,
        M_FILTER_FLAGS,
        MAC_FILTER_ACCEPT_GRP | MAC_FILTER_IN_BEACON,
    );
    w32(&mut b, M_QOS_FLAGS, 0);
    // Five access categories with workable EDCA defaults. `fifos_mask` picks
    // the transmit FIFO; zero would leave the category with none.
    for i in 0..AC_COUNT {
        let at = M_AC + i * AC_QOS_LEN;
        w16(&mut b, at, 15); // cw_min
        w16(&mut b, at + 2, 1023); // cw_max
        b[at + 4] = 2; // aifsn
        b[at + 5] = 1 << i; // fifos_mask
        w16(&mut b, at + 6, 0); // edca_txop
    }
    // The station arm.
    let u = MAC_UNION_OFF;
    match assoc {
        Some((aid, bi, dtim)) => {
            w32(&mut b, u + STA_IS_ASSOC, 1);
            w32(&mut b, u + STA_BI, bi as u32);
            w32(
                &mut b,
                u + STA_DTIM_INTERVAL,
                (bi as u32) * (dtim.max(1) as u32),
            );
            w32(&mut b, u + STA_ASSOC_ID, aid as u32);
        }
        None => {
            w32(&mut b, u + STA_IS_ASSOC, 0);
        }
    }
    w32(&mut b, u + STA_DTIM_TIME, 0);
    w64(&mut b, u + STA_DTIM_TSF, 0);
    w32(&mut b, u + STA_DATA_POLICY, 0);
    // How many beacon intervals we may sleep through. 10 is Linux's default and
    // is policy rather than layout.
    w32(&mut b, u + STA_LISTEN_INTERVAL, 10);
    b
}

fn w16(b: &mut [u8], at: usize, v: u16) {
    b[at..at + 2].copy_from_slice(&v.to_le_bytes());
}
fn w64(b: &mut [u8], at: usize, v: u64) {
    b[at..at + 8].copy_from_slice(&v.to_le_bytes());
}

// --- BINDING_CONTEXT_CMD --------------------------------------------------

/// `BINDING_CONTEXT_CMD`, legacy group.
pub const BINDING_CONTEXT_CMD: u8 = 0x2b;

/// `MAX_MACS_IN_BINDING`.
pub const MAX_MACS_IN_BINDING: usize = 3;

/// `iwl_binding_cmd` — 28 bytes: two words, a three-word array, then two more.
pub const BINDING_LEN: usize = 4 + 4 + 4 * MAX_MACS_IN_BINDING + 4 + 4;
/// `iwl_binding_cmd_v1`, without `lmac_id` — 24 bytes.
pub const BINDING_LEN_V1: usize = BINDING_LEN - 4;

const B_ID_AND_COLOR: usize = 0;
const B_ACTION: usize = 4;
const B_MACS: usize = 8;
const B_PHY: usize = 8 + 4 * MAX_MACS_IN_BINDING; // 20
const B_LMAC_ID: usize = B_PHY + 4; // 24

/// `FW_CTXT_INVALID` — the value an unused context slot must hold.
pub const CTXT_INVALID: u32 = 0xffff_ffff;

/// Tie a MAC context to a PHY context.
///
/// **Unused MAC slots must be `FW_CTXT_INVALID`, not zero.** Zero is a
/// perfectly valid context identifier — id 0, colour 0 — so a zeroed array
/// binds three MACs, two of which are whatever happens to live at id 0. The
/// binding is accepted and the radio serves a context we never configured.
pub fn binding(id: u8, color: u8, mac_id_color: u32, phy_id_color: u32, action: u32) -> Vec<u8> {
    let mut b = alloc::vec![0u8; BINDING_LEN];
    w32(&mut b, B_ID_AND_COLOR, id_and_color(id, color));
    w32(&mut b, B_ACTION, action);
    for i in 0..MAX_MACS_IN_BINDING {
        // Every slot starts invalid; only slot 0 is ours.
        w32(&mut b, B_MACS + i * 4, CTXT_INVALID);
    }
    w32(&mut b, B_MACS, mac_id_color);
    w32(&mut b, B_PHY, phy_id_color);
    w32(&mut b, B_LMAC_ID, 0);
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn the_phy_context_layout_matches_the_header() {
        assert_eq!(PHY_CTXT_LEN, 32);
        assert_eq!(
            CHANNEL_INFO_LEN, 8,
            "u32 channel then three bytes and a pad"
        );
        assert_eq!(P_CI, 8);
        assert_eq!(P_LMAC_ID, 16, "the channel info is 8 bytes, not 4");
        assert_eq!(P_SECONDARY_CTRL, 28);
    }

    /// **`FW_CTXT_ACTION_ADD` is 1, not 0** — zero is `INVALID`. A command built
    /// by zeroing a buffer and filling only the interesting fields lands on
    /// `INVALID` by default, and asks the firmware to do nothing valid.
    #[test_case]
    fn the_add_action_is_one_not_zero() {
        assert_eq!(ACTION_INVALID, 0);
        assert_eq!(ACTION_ADD, 1);
        assert_eq!(ACTION_MODIFY, 2);
        assert_eq!(ACTION_REMOVE, 3);
        let c = phy_context_20mhz(0, 0, 6, ACTION_ADD);
        let a = u32::from_le_bytes(c[P_ACTION..P_ACTION + 4].try_into().unwrap());
        assert_eq!(a, 1, "a zeroed buffer would leave this INVALID");
        assert_ne!(a, ACTION_INVALID);
    }

    /// **`PHY_BAND_5` is 0 and `PHY_BAND_24` is 1** — the numbering does not
    /// follow the frequencies. The obvious "2.4 GHz is band 0" tunes the radio
    /// to a 5 GHz channel number on the 2.4 GHz band, which the firmware
    /// accepts and which hears nothing.
    #[test_case]
    fn the_band_numbering_does_not_follow_the_frequencies() {
        assert_eq!(PHY_BAND_5, 0, "5 GHz is band ZERO");
        assert_eq!(PHY_BAND_24, 1, "2.4 GHz is band ONE");
        assert_ne!(PHY_BAND_24, 0, "the intuitive answer is the wrong one");

        assert_eq!(band_for_channel(1), PHY_BAND_24);
        assert_eq!(band_for_channel(11), PHY_BAND_24);
        assert_eq!(band_for_channel(14), PHY_BAND_24);
        assert_eq!(band_for_channel(36), PHY_BAND_5);
        assert_eq!(band_for_channel(165), PHY_BAND_5);
    }

    /// The channel info leads with a **u32 channel**, not a byte band — the
    /// reverse of the older `_v1`. A driver written against the old order
    /// writes the channel number into the band field.
    #[test_case]
    fn the_channel_comes_first_and_is_a_word() {
        for (ch, band) in [(6u8, PHY_BAND_24), (36, PHY_BAND_5), (149, PHY_BAND_5)] {
            let c = phy_context_20mhz(1, 2, ch, ACTION_ADD);
            let n = u32::from_le_bytes(c[P_CI..P_CI + 4].try_into().unwrap());
            assert_eq!(n, ch as u32, "channel, as a word, first");
            assert_eq!(c[P_CI + CI_BAND], band, "then the band");
            assert_eq!(c[P_CI + CI_WIDTH], CHANNEL_MODE20);
            assert_eq!(c[P_CI + CI_CTRL_POS], CTRL_POS_20MHZ, "no offset at 20 MHz");
        }
    }

    /// A context with no receive chain associates and then hears nothing, so
    /// `rxchain_info` must not be left at zero.
    #[test_case]
    fn the_receive_chains_are_configured() {
        let c = phy_context_20mhz(0, 0, 1, ACTION_ADD);
        let rx = u32::from_le_bytes(c[P_RXCHAIN_INFO..P_RXCHAIN_INFO + 4].try_into().unwrap());
        assert_ne!(rx, 0, "zero chains hears nothing");
    }

    /// The identifier packing agrees with the station command's by
    /// construction: `FW_CTXT_COLOR_POS` is 8 for every context type.
    #[test_case]
    fn the_identifier_packing_agrees_with_the_station_command() {
        assert_eq!(CTXT_COLOR_POS, 8);
        for (id, color) in [(0u8, 0u8), (1, 3), (2, 255)] {
            assert_eq!(
                id_and_color(id, color),
                super::super::sta::mac_id_n_color(id, color)
            );
        }
        let c = phy_context_20mhz(3, 7, 1, ACTION_ADD);
        let v = u32::from_le_bytes(c[P_ID_AND_COLOR..P_ID_AND_COLOR + 4].try_into().unwrap());
        assert_eq!(v, id_and_color(3, 7));
    }

    /// **The command is the common part plus the union's LARGEST arm**, not the
    /// arm in use. Linux sends `sizeof(*cmd)` whatever the interface type, and
    /// four bytes too few makes the firmware reject a command that is otherwise
    /// correct.
    #[test_case]
    fn the_mac_context_is_sized_for_the_largest_union_arm() {
        assert_eq!(
            MAC_UNION_OFF, 100,
            "common part: 60 + five 8-byte AC entries"
        );
        assert_eq!(M_AC, 60);
        assert_eq!(M_AC + AC_QOS_LEN * AC_COUNT, MAC_UNION_OFF);
        // iwl_mac_data_sta is 44; iwl_mac_data_p2p_sta is that plus a ctwin.
        assert_eq!(MAC_CTXT_LEN, 148, "100 + 48, not 100 + 44");
        assert_ne!(
            MAC_CTXT_LEN,
            MAC_UNION_OFF + 44,
            "sizing to the sta arm is four short"
        );
    }

    /// `iwl_mac_types` starts at 1, so a zeroed `mac_type` is not a type at all
    /// — the same shape of bug as the zeroed action.
    #[test_case]
    fn the_mac_type_and_filters_are_set_rather_than_left_zero() {
        let c = mac_context_sta(0, 1, &[2; 6], &[3; 6], ACTION_ADD, None);
        assert_eq!(c.len(), MAC_CTXT_LEN);
        let ty = u32::from_le_bytes(c[M_MAC_TYPE..M_MAC_TYPE + 4].try_into().unwrap());
        assert_eq!(ty, MAC_TYPE_BSS_STA);
        assert_ne!(ty, 0, "the enum starts at 1");

        // Without ACCEPT_GRP the link comes up and never resolves an address:
        // ARP and DHCP replies are group-addressed.
        let f = u32::from_le_bytes(c[M_FILTER_FLAGS..M_FILTER_FLAGS + 4].try_into().unwrap());
        assert_ne!(
            f & MAC_FILTER_ACCEPT_GRP,
            0,
            "ARP/DHCP replies are group-addressed"
        );
        assert_ne!(f & MAC_FILTER_IN_BEACON, 0);

        // A zero rate mask is a station that may transmit at no rate.
        assert_ne!(
            u32::from_le_bytes(c[M_CCK_RATES..M_CCK_RATES + 4].try_into().unwrap()),
            0
        );
        assert_ne!(
            u32::from_le_bytes(c[M_OFDM_RATES..M_OFDM_RATES + 4].try_into().unwrap()),
            0
        );
        // Every access category gets a transmit FIFO; zero would leave it none.
        for i in 0..AC_COUNT {
            assert_ne!(c[M_AC + i * AC_QOS_LEN + 5], 0, "ac[{i}] fifos_mask");
        }
    }

    /// The same command carries both states, which is why a join sends it
    /// twice — before association and again with the AID.
    #[test_case]
    fn the_mac_context_carries_the_association_state() {
        let before = mac_context_sta(0, 0, &[1; 6], &[2; 6], ACTION_ADD, None);
        let u = MAC_UNION_OFF;
        assert_eq!(
            u32::from_le_bytes(before[u..u + 4].try_into().unwrap()),
            0,
            "not associated"
        );

        let after = mac_context_sta(0, 0, &[1; 6], &[2; 6], ACTION_MODIFY, Some((7, 100, 3)));
        assert_eq!(
            u32::from_le_bytes(after[u..u + 4].try_into().unwrap()),
            1,
            "associated"
        );
        let aid = u32::from_le_bytes(
            after[u + STA_ASSOC_ID..u + STA_ASSOC_ID + 4]
                .try_into()
                .unwrap(),
        );
        assert_eq!(aid, 7);
        let bi = u32::from_le_bytes(after[u + STA_BI..u + STA_BI + 4].try_into().unwrap());
        assert_eq!(bi, 100);
        // The DTIM interval is beacons x period, not the period alone.
        let dtim = u32::from_le_bytes(
            after[u + STA_DTIM_INTERVAL..u + STA_DTIM_INTERVAL + 4]
                .try_into()
                .unwrap(),
        );
        assert_eq!(dtim, 300);
    }

    /// **An unused MAC slot must be `FW_CTXT_INVALID`, not zero.** Zero is a
    /// valid identifier — id 0, colour 0 — so a zeroed array binds three MACs,
    /// two of them whatever lives at id 0. The binding is accepted and the radio
    /// serves a context we never configured.
    #[test_case]
    fn unused_binding_slots_are_invalid_not_zero() {
        assert_eq!(BINDING_LEN, 28, "macs[3] is three words, not one");
        assert_eq!(BINDING_LEN_V1, 24);
        assert_eq!(B_PHY, 20);

        let mac = id_and_color(0, 0);
        assert_eq!(mac, 0, "id 0 colour 0 really is the value zero");
        let b = binding(0, 0, mac, id_and_color(1, 2), ACTION_ADD);
        assert_eq!(
            u32::from_le_bytes(b[B_MACS..B_MACS + 4].try_into().unwrap()),
            mac
        );
        for i in 1..MAX_MACS_IN_BINDING {
            let v = u32::from_le_bytes(b[B_MACS + i * 4..B_MACS + i * 4 + 4].try_into().unwrap());
            assert_eq!(v, CTXT_INVALID, "slot {i} must be invalid, not zero");
            assert_ne!(v, 0);
        }
        let phy = u32::from_le_bytes(b[B_PHY..B_PHY + 4].try_into().unwrap());
        assert_eq!(phy, id_and_color(1, 2));
    }
}
