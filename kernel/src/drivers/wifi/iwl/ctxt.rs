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

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn the_phy_context_layout_matches_the_header() {
        assert_eq!(PHY_CTXT_LEN, 32);
        assert_eq!(CHANNEL_INFO_LEN, 8, "u32 channel then three bytes and a pad");
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
            assert_eq!(id_and_color(id, color), super::super::sta::mac_id_n_color(id, color));
        }
        let c = phy_context_20mhz(3, 7, 1, ACTION_ADD);
        let v = u32::from_le_bytes(c[P_ID_AND_COLOR..P_ID_AND_COLOR + 4].try_into().unwrap());
        assert_eq!(v, id_and_color(3, 7));
    }
}
