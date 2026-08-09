//! **`TX_CMD`** — handing a frame to the radio.
//!
//! Layouts from Linux's `fw/api/tx.h`. There are two in play and **they are not
//! a widened version of each other**:
//!
//! ```text
//! iwl_tx_cmd_v9   (22000 family: AX200/AX201)      20-byte header
//!   len              @0   __le16
//!   offload_assist   @2   __le16
//!   flags            @4   __le32
//!   dram_info        @8   8
//!   rate_n_flags     @16  __le32
//!   hdr[]            @20
//!
//! iwl_tx_cmd      (AX210 and later)                28-byte header
//!   len              @0   __le16
//!   flags            @2   __le16      <- swapped with offload_assist
//!   offload_assist   @4   __le32      <- and both change width
//!   dram_info        @8   8
//!   rate_n_flags     @16  __le32
//!   reserved[8]      @20
//!   hdr[]            @28
//! ```
//!
//! ## The swap is the whole hazard
//!
//! `flags` and `offload_assist` **trade places and widths** between the two. A
//! driver that writes one layout to the other radio puts the offload hints
//! where the flags go and the flags where the offload hints go — and both
//! fields are bitmasks of small numbers, so nothing looks out of range. The
//! frame is accepted and transmitted with the wrong acknowledgement policy and
//! a nonsense checksum-offload request.
//!
//! There is no marker to check: like the receive descriptor, the layout is a
//! property of the silicon. [`hdr_len`] is the single place that decision is
//! made.
//!
//! ## Where the packet number goes
//!
//! `dram_info` carries `pn_low`/`pn_high` — **the CCMP packet number**, because
//! on this hardware the firmware encrypts (see [`super::sta`]). This is the
//! field that connects the four-way handshake's key to a transmitted frame, and
//! a repeated value under one key hands an observer the XOR of two plaintexts,
//! exactly as it would in software.

use super::fw;
use alloc::vec::Vec;

/// `TX_CMD`, legacy group.
pub const TX_CMD: u8 = 0x1c;

/// `iwl_dram_sec_info` — 8 bytes.
pub const DRAM_SEC_INFO_LEN: usize = 8;

/// `iwl_tx_cmd_v9` header — 20 bytes before the 802.11 header.
pub const TX_HDR_LEN_V9: usize = 20;
/// `iwl_tx_cmd` (AX210+) header — 28 bytes.
pub const TX_HDR_LEN_AX210: usize = 28;

// v9 offsets.
const V9_LEN: usize = 0;
const V9_OFFLOAD_ASSIST: usize = 2;
const V9_FLAGS: usize = 4;
const V9_DRAM: usize = 8;
const V9_RATE_N_FLAGS: usize = 16;

// AX210+ offsets. Note `flags` and `offload_assist` are swapped *and* resized.
const AX_LEN: usize = 0;
const AX_FLAGS: usize = 2;
const AX_OFFLOAD_ASSIST: usize = 4;
const AX_DRAM: usize = 8;
const AX_RATE_N_FLAGS: usize = 16;

// Inside `iwl_dram_sec_info`.
const D_PN_LOW: usize = 0;
const D_PN_HIGH: usize = 4;
const D_AUX_INFO: usize = 6;

/// `enum iwl_tx_cmd_flags`.
pub const TX_FLG_PROT_REQUIRE: u32 = 1 << 0;
pub const TX_FLG_ACK: u32 = 1 << 3;
pub const TX_FLG_BT_DIS: u32 = 1 << 12;
pub const TX_FLG_SEQ_CTL: u32 = 1 << 13;

/// Security control: CCMP.
pub const TX_SEC_CCM: u8 = 0x02;

/// The transmit-command header length for this hardware.
///
/// **The only place this decision is made**, for the same reason as
/// [`super::rx::desc_len`]: there is no marker in the frame, so a family added
/// to [`fw::Family`] must be classified here or it silently inherits a layout
/// whose two most important fields are transposed.
pub fn hdr_len(family: fw::Family) -> usize {
    match family {
        fw::Family::Ax210 | fw::Family::Be200 => TX_HDR_LEN_AX210,
        fw::Family::Iwl7000 | fw::Family::Iwl8000 | fw::Family::Iwl9000 | fw::Family::Ax200 => {
            TX_HDR_LEN_V9
        }
    }
}

/// Default transmit flags for a data frame to the access point.
///
/// `ACK` because a unicast data frame is acknowledged; `SEQ_CTL` so the
/// firmware assigns the sequence number rather than trusting ours — it owns
/// retransmission, and two sources of sequence numbers means duplicates the
/// peer discards.
pub fn default_flags() -> u32 {
    TX_FLG_ACK | TX_FLG_SEQ_CTL | TX_FLG_BT_DIS
}

/// Build a `TX_CMD` carrying `frame` (a complete 802.11 frame, header first).
///
/// `pn` is the CCMP packet number the firmware will encrypt with; pass 0 for an
/// unencrypted frame. `rate_n_flags` is the transmit rate, which the caller
/// takes from its rate-control policy.
pub fn build(family: fw::Family, frame: &[u8], pn: u64, rate_n_flags: u32, flags: u32) -> Option<Vec<u8>> {
    if frame.len() < 24 || frame.len() > 4096 {
        return None;
    }
    let h = hdr_len(family);
    let mut b = alloc::vec![0u8; h + frame.len()];

    // `len` is the 802.11 frame's length, not the command's — a command that
    // reported its own length would have the radio transmit the header too.
    let (len_at, flags_at, dram_at, rate_at) = match h {
        TX_HDR_LEN_AX210 => (AX_LEN, AX_FLAGS, AX_DRAM, AX_RATE_N_FLAGS),
        _ => (V9_LEN, V9_FLAGS, V9_DRAM, V9_RATE_N_FLAGS),
    };
    b[len_at..len_at + 2].copy_from_slice(&(frame.len() as u16).to_le_bytes());

    // The flags field is 16 bits on AX210 and 32 on v9 — writing four bytes to
    // the shorter one would run into `offload_assist`.
    if h == TX_HDR_LEN_AX210 {
        b[flags_at..flags_at + 2].copy_from_slice(&(flags as u16).to_le_bytes());
        b[AX_OFFLOAD_ASSIST..AX_OFFLOAD_ASSIST + 4].copy_from_slice(&0u32.to_le_bytes());
    } else {
        b[flags_at..flags_at + 4].copy_from_slice(&flags.to_le_bytes());
        b[V9_OFFLOAD_ASSIST..V9_OFFLOAD_ASSIST + 2].copy_from_slice(&0u16.to_le_bytes());
    }

    // The packet number, split low/high — this is what the firmware encrypts
    // with, so it must advance for every frame under one key.
    b[dram_at + D_PN_LOW..dram_at + D_PN_LOW + 4].copy_from_slice(&((pn & 0xffff_ffff) as u32).to_le_bytes());
    b[dram_at + D_PN_HIGH..dram_at + D_PN_HIGH + 2].copy_from_slice(&(((pn >> 32) & 0xffff) as u16).to_le_bytes());
    b[dram_at + D_AUX_INFO..dram_at + D_AUX_INFO + 2].copy_from_slice(&0u16.to_le_bytes());

    b[rate_at..rate_at + 4].copy_from_slice(&rate_n_flags.to_le_bytes());
    b[h..].copy_from_slice(frame);
    Some(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **`flags` and `offload_assist` trade places *and* widths** between the
    /// two layouts. Both are bitmasks of small numbers, so writing one layout to
    /// the other radio puts each field where the other belongs and nothing looks
    /// out of range — the frame is transmitted with the wrong acknowledgement
    /// policy and a nonsense offload request.
    #[test_case]
    fn the_two_layouts_transpose_flags_and_offload_assist() {
        assert_eq!(V9_OFFLOAD_ASSIST, 2);
        assert_eq!(V9_FLAGS, 4);
        assert_eq!(AX_FLAGS, 2, "the AX210 layout puts FLAGS where v9 puts offload");
        assert_eq!(AX_OFFLOAD_ASSIST, 4, "and offload where v9 puts flags");
        assert_ne!(V9_FLAGS, AX_FLAGS, "they are transposed, not merely resized");

        assert_eq!(TX_HDR_LEN_V9, 20);
        assert_eq!(TX_HDR_LEN_AX210, 28, "eight reserved bytes before the frame");
        assert_eq!(DRAM_SEC_INFO_LEN, 8);
        assert_eq!(hdr_len(fw::Family::Ax200), TX_HDR_LEN_V9, "AX200 is 22000-family");
        assert_eq!(hdr_len(fw::Family::Iwl9000), TX_HDR_LEN_V9);
        assert_eq!(hdr_len(fw::Family::Ax210), TX_HDR_LEN_AX210);
        assert_eq!(hdr_len(fw::Family::Be200), TX_HDR_LEN_AX210);
    }

    /// The flags land in the right field on each layout, at the right width.
    /// Writing four bytes to the AX210's 16-bit field would run into
    /// `offload_assist`.
    #[test_case]
    fn the_flags_are_written_at_the_right_width() {
        let frame = alloc::vec![0xaau8; 30];
        let f = default_flags();

        let v9 = build(fw::Family::Ax200, &frame, 0, 0, f).unwrap();
        let got = u32::from_le_bytes(v9[V9_FLAGS..V9_FLAGS + 4].try_into().unwrap());
        assert_eq!(got, f, "v9 flags are 32 bits at offset 4");
        assert_eq!(u16::from_le_bytes([v9[V9_OFFLOAD_ASSIST], v9[V9_OFFLOAD_ASSIST + 1]]), 0);

        let ax = build(fw::Family::Ax210, &frame, 0, 0, f).unwrap();
        let got = u16::from_le_bytes([ax[AX_FLAGS], ax[AX_FLAGS + 1]]);
        assert_eq!(got, f as u16, "AX210 flags are 16 bits at offset 2");
        // And the 32-bit offload field beside it is untouched by the flags.
        assert_eq!(u32::from_le_bytes(ax[AX_OFFLOAD_ASSIST..AX_OFFLOAD_ASSIST + 4].try_into().unwrap()), 0);
    }

    /// `len` is the **frame's** length, not the command's. A command reporting
    /// its own length would have the radio transmit its own header.
    #[test_case]
    fn the_length_is_the_frame_not_the_command() {
        let frame = alloc::vec![0x5au8; 100];
        for family in [fw::Family::Ax200, fw::Family::Ax210] {
            let c = build(family, &frame, 0, 0, 0).unwrap();
            let len = u16::from_le_bytes([c[0], c[1]]) as usize;
            assert_eq!(len, frame.len(), "the frame's length");
            assert_ne!(len, c.len(), "not the command's");
            assert_eq!(c.len(), hdr_len(family) + frame.len());
            // The frame follows the header intact.
            assert_eq!(&c[hdr_len(family)..], &frame[..]);
        }
    }

    /// **The packet number is what the firmware encrypts with**, split across
    /// `pn_low` and `pn_high`. A repeat under one key hands an observer the XOR
    /// of two plaintexts, exactly as it would in software.
    #[test_case]
    fn the_packet_number_is_split_low_and_high() {
        let frame = alloc::vec![0u8; 24];
        let pn = 0x1234_5678_9abcu64;
        let c = build(fw::Family::Ax200, &frame, pn, 0, 0).unwrap();
        let lo = u32::from_le_bytes(c[V9_DRAM + D_PN_LOW..V9_DRAM + D_PN_LOW + 4].try_into().unwrap());
        let hi = u16::from_le_bytes([c[V9_DRAM + D_PN_HIGH], c[V9_DRAM + D_PN_HIGH + 1]]);
        assert_eq!(lo, 0x5678_9abc);
        assert_eq!(hi, 0x1234);
        assert_eq!(((hi as u64) << 32) | lo as u64, pn, "and they reassemble");

        // A 48-bit number at the top of its range survives.
        let c = build(fw::Family::Ax210, &frame, 0xffff_ffff_ffff, 0, 0).unwrap();
        let lo = u32::from_le_bytes(c[AX_DRAM..AX_DRAM + 4].try_into().unwrap());
        let hi = u16::from_le_bytes([c[AX_DRAM + 4], c[AX_DRAM + 5]]);
        assert_eq!((lo, hi), (0xffff_ffff, 0xffff));
    }

    /// `SEQ_CTL` is set so the firmware assigns the sequence number: it owns
    /// retransmission, and two sources of sequence numbers means duplicates the
    /// peer discards.
    #[test_case]
    fn the_default_flags_ask_for_acknowledgement_and_firmware_sequencing() {
        let f = default_flags();
        assert_ne!(f & TX_FLG_ACK, 0, "a unicast data frame is acknowledged");
        assert_ne!(f & TX_FLG_SEQ_CTL, 0, "the firmware owns the sequence number");
        // Both fit the AX210 layout's 16-bit field.
        assert_eq!(f & 0xffff, f, "the default flags fit 16 bits");
    }

    /// A frame too short to be 802.11, or absurdly long, is refused rather than
    /// framed.
    #[test_case]
    fn an_impossible_frame_is_refused() {
        assert!(build(fw::Family::Ax200, &[], 0, 0, 0).is_none());
        assert!(build(fw::Family::Ax200, &[0u8; 23], 0, 0, 0).is_none());
        assert!(build(fw::Family::Ax200, &alloc::vec![0u8; 5000], 0, 0, 0).is_none());
        assert!(build(fw::Family::Ax200, &[0u8; 24], 0, 0, 0).is_some());
    }
}
