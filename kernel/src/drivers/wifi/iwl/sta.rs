//! **`ADD_STA` and `ADD_STA_KEY`** — telling the radio about the access point,
//! and installing the pairwise key.
//!
//! Layouts from Linux's `fw/api/sta.h`, not from memory.
//!
//! ## Who actually encrypts
//!
//! Worth stating plainly, because it qualifies something the CCMP work implied:
//! **on Intel the hardware encrypts.** Once a key is installed here the driver
//! hands the radio plaintext and the firmware applies CCMP, maintains the
//! transmit packet number and checks the receive one. [`super::super::ccmp`] is
//! not on that path.
//!
//! That does not make the software cipher redundant — a SoftMAC part with no
//! crypto offload needs it, `mac80211` carries exactly the same code for the
//! same reason, and it is what validates the key material this command
//! installs — but "CCMP is the last mile for every radio" was too broad. For
//! Intel the last mile is this command.
//!
//! ## The bit that quietly means something else
//!
//! `STA_KEY_FLG_KEY_32BYTES` and `STA_KEY_FLG_WEP_13BYTES` are **the same bit**
//! (12). A CCMP pairwise key is 16 bytes, so it must stay clear; setting it
//! declares a 32-byte key and the firmware reads 32 bytes out of a 16-byte
//! field, encrypting with the key plus whatever follows it. The link comes up
//! and every frame fails its MIC at the peer.
//!
//! `STA_KEY_NOT_VALID` (bit 11) is the other one worth naming: set, the command
//! succeeds and installs a key the hardware then ignores.

use alloc::vec::Vec;

/// `REPLY_ADD_STA`, legacy group.
pub const ADD_STA: u8 = 0x18;
/// `REPLY_ADD_STA_KEY`, legacy group.
pub const ADD_STA_KEY: u8 = 0x17;

/// `iwl_mvm_add_sta_cmd` — 48 bytes.
pub const ADD_STA_LEN: usize = 48;
/// `iwl_mvm_add_sta_key_common` — 52 bytes.
pub const KEY_COMMON_LEN: usize = 52;
/// `iwl_mvm_add_sta_key_cmd` — 76 bytes.
pub const ADD_STA_KEY_LEN: usize = 76;

// Offsets into `iwl_mvm_add_sta_cmd`.
const S_ADD_MODIFY: usize = 0;
const S_TID_DISABLE_TX: usize = 2;
const S_MAC_ID_N_COLOR: usize = 4;
const S_ADDR: usize = 8;
const S_STA_ID: usize = 16;
const S_MODIFY_MASK: usize = 17;
const S_STATION_FLAGS: usize = 20;
const S_STATION_FLAGS_MSK: usize = 24;
const S_STATION_TYPE: usize = 35;
const S_ASSOC_ID: usize = 36;
const S_TFD_QUEUE_MSK: usize = 40;

// Offsets into `iwl_mvm_add_sta_key_cmd`.
const K_STA_ID: usize = 0;
const K_KEY_OFFSET: usize = 1;
const K_KEY_FLAGS: usize = 2;
const K_KEY: usize = 4;
const K_RX_SECUR_SEQ: usize = 36;
const K_TX_SEQ_CNT: usize = 68;

/// `enum iwl_sta_key_flag`.
pub const KEY_FLG_NO_ENC: u16 = 0;
pub const KEY_FLG_CCM: u16 = 2;
pub const KEY_FLG_EN_MSK: u16 = 7;
pub const KEY_FLG_KEYID_POS: u16 = 8;
/// **Bit 12 is `KEY_32BYTES` here and `WEP_13BYTES` elsewhere** — see the
/// module docs. A 16-byte CCMP key leaves it clear.
pub const KEY_FLG_KEY_32BYTES: u16 = 1 << 12;
pub const KEY_NOT_VALID: u16 = 1 << 11;
pub const KEY_MULTICAST: u16 = 1 << 14;
pub const KEY_MFP: u16 = 1 << 15;

/// `enum iwl_sta_flags` — the two that matter for an associated peer.
pub const STA_FLG_CLASS_AUTH: u32 = 1 << 14;
pub const STA_FLG_CLASS_ASSOC: u32 = 1 << 15;

/// `add_modify`: add a new station rather than change one.
pub const MODE_ADD: u8 = 0;
pub const MODE_MODIFY: u8 = 1;
pub const MODE_REMOVE: u8 = 2;

/// `station_type`: a peer we are associated *to* (we are the client).
pub const TYPE_LINK: u8 = 0;

/// The station id the access point occupies. Ids are a small fixed space the
/// driver allocates; 0 is conventional for the AP on a client interface.
pub const AP_STA_ID: u8 = 0;

fn w16(b: &mut [u8], at: usize, v: u16) {
    b[at..at + 2].copy_from_slice(&v.to_le_bytes());
}
fn w32(b: &mut [u8], at: usize, v: u32) {
    b[at..at + 4].copy_from_slice(&v.to_le_bytes());
}
fn w64(b: &mut [u8], at: usize, v: u64) {
    b[at..at + 8].copy_from_slice(&v.to_le_bytes());
}

/// Pack a MAC context id and its colour into the one word the firmware wants.
///
/// A context is identified by **both**: the colour changes each time an id is
/// reused, so a command carrying a stale colour is rejected rather than applied
/// to whatever now occupies that id. Sending the id alone works until the first
/// reuse.
pub fn mac_id_n_color(id: u8, color: u8) -> u32 {
    (id as u32) | ((color as u32) << 8)
}

/// Build `ADD_STA` for the access point we have associated with.
///
/// `assoc_id` is the AID the association response returned; `tfd_queue_msk` is
/// the transmit queues this station may use.
pub fn add_ap(
    mac_id: u8,
    mac_color: u8,
    bssid: &[u8; 6],
    assoc_id: u16,
    tfd_queue_msk: u32,
) -> Vec<u8> {
    let mut b = alloc::vec![0u8; ADD_STA_LEN];
    b[S_ADD_MODIFY] = MODE_ADD;
    // No TID is disabled: every traffic class may transmit. The field is a
    // *disable* mask, so zero is permissive and 0xffff would silence the
    // station while every command still succeeded.
    w16(&mut b, S_TID_DISABLE_TX, 0);
    w32(&mut b, S_MAC_ID_N_COLOR, mac_id_n_color(mac_id, mac_color));
    b[S_ADDR..S_ADDR + 6].copy_from_slice(bssid);
    b[S_STA_ID] = AP_STA_ID;
    b[S_MODIFY_MASK] = 0;
    // Authenticated and associated. `station_flags_msk` says which bits of
    // `station_flags` to apply — a flag set without its mask bit is ignored,
    // which is the quiet way this command does nothing.
    let flags = STA_FLG_CLASS_AUTH | STA_FLG_CLASS_ASSOC;
    w32(&mut b, S_STATION_FLAGS, flags);
    w32(&mut b, S_STATION_FLAGS_MSK, flags);
    b[S_STATION_TYPE] = TYPE_LINK;
    w16(&mut b, S_ASSOC_ID, assoc_id);
    w32(&mut b, S_TFD_QUEUE_MSK, tfd_queue_msk);
    b
}

/// Build `ADD_STA_KEY` installing a CCMP pairwise key.
///
/// `tx_pn` is the transmit packet number the firmware starts from. It must
/// continue rather than restart if a key is ever reinstalled under the same
/// PTK: CCM is a counter mode, and a repeated packet number under one key hands
/// an observer the XOR of two plaintexts.
pub fn add_pairwise_key(sta_id: u8, key_id: u8, tk: &[u8; 16], tx_pn: u64) -> Vec<u8> {
    let mut b = alloc::vec![0u8; ADD_STA_KEY_LEN];
    b[K_STA_ID] = sta_id;
    // The key offset is the slot in the station's key table. One pairwise key
    // lives at 0.
    b[K_KEY_OFFSET] = 0;
    let mut flags = KEY_FLG_CCM | ((key_id as u16 & 0x3) << KEY_FLG_KEYID_POS);
    // Pairwise, so *not* multicast. Setting that bit installs this as the group
    // key and leaves unicast traffic unencrypted.
    flags &= !KEY_MULTICAST;
    // 16-byte key: bit 12 stays clear. See the module docs — set, the firmware
    // reads 32 bytes out of a 16-byte field.
    debug_assert_eq!(flags & KEY_FLG_KEY_32BYTES, 0);
    debug_assert_eq!(flags & KEY_NOT_VALID, 0);
    w16(&mut b, K_KEY_FLAGS, flags);
    b[K_KEY..K_KEY + 16].copy_from_slice(tk);
    // The remaining 16 bytes of the key field stay zero, and the receive
    // sequence counters start at zero — the peer's first frame must advance
    // past them.
    b[K_RX_SECUR_SEQ..K_RX_SECUR_SEQ + 16].fill(0);
    w64(&mut b, K_TX_SEQ_CNT, tx_pn);
    b
}

/// Build `ADD_STA_KEY` installing the group key, which protects broadcast and
/// multicast.
///
/// The same command with the multicast bit set and the key id the AP chose —
/// which is **not** the pairwise key's id and comes from the GTK KDE in message
/// 3 of the handshake.
pub fn add_group_key(sta_id: u8, key_id: u8, gtk: &[u8]) -> Option<Vec<u8>> {
    if gtk.len() != 16 {
        // A 32-byte group key would need bit 12, and this driver only speaks
        // CCMP-128; refusing is better than installing a truncated key that
        // decrypts nothing and reports success.
        return None;
    }
    let mut b = alloc::vec![0u8; ADD_STA_KEY_LEN];
    b[K_STA_ID] = sta_id;
    b[K_KEY_OFFSET] = 1;
    let flags = KEY_FLG_CCM | ((key_id as u16 & 0x3) << KEY_FLG_KEYID_POS) | KEY_MULTICAST;
    w16(&mut b, K_KEY_FLAGS, flags);
    b[K_KEY..K_KEY + 16].copy_from_slice(gtk);
    Some(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sizes and offsets against `sta.h`'s field list — the only testable
    /// surface for a command that cannot be sent from here.
    #[test_case]
    fn the_layouts_match_the_header() {
        assert_eq!(ADD_STA_LEN, 48, "iwl_mvm_add_sta_cmd");
        assert_eq!(
            KEY_COMMON_LEN, 52,
            "sta_id, key_offset, flags, key[32], rx_seq[16]"
        );
        assert_eq!(ADD_STA_KEY_LEN, 76, "common + three u64s");
        assert_eq!(K_KEY, 4);
        assert_eq!(K_RX_SECUR_SEQ, 36);
        assert_eq!(K_TX_SEQ_CNT, 68, "after rx_mic_key and tx_mic_key");
        assert_eq!(S_ADDR, 8);
        assert_eq!(S_TFD_QUEUE_MSK, 40);
        assert!(S_TFD_QUEUE_MSK + 4 <= ADD_STA_LEN);
    }

    /// **Bit 12 is `KEY_32BYTES` and `WEP_13BYTES` at once.** A 16-byte CCMP
    /// key must leave it clear; set, the firmware reads 32 bytes out of a
    /// 16-byte field and encrypts with the key plus whatever follows it — the
    /// link comes up and every frame fails its MIC at the peer.
    #[test_case]
    fn a_ccmp_key_never_sets_the_thirty_two_byte_bit() {
        let tk = [0x5au8; 16];
        let k = add_pairwise_key(AP_STA_ID, 0, &tk, 1);
        let flags = u16::from_le_bytes([k[K_KEY_FLAGS], k[K_KEY_FLAGS + 1]]);
        assert_eq!(flags & KEY_FLG_EN_MSK, KEY_FLG_CCM, "cipher is CCMP");
        assert_eq!(flags & KEY_FLG_KEY_32BYTES, 0, "16-byte key");
        assert_eq!(
            flags & KEY_NOT_VALID,
            0,
            "set, the key installs and is ignored"
        );
        assert_eq!(&k[K_KEY..K_KEY + 16], &tk[..], "the key itself");
        assert!(
            k[K_KEY + 16..K_KEY + 32].iter().all(|&b| b == 0),
            "the tail stays zero"
        );
    }

    /// **A pairwise key must not carry the multicast bit** — that installs it as
    /// the group key and leaves unicast traffic unencrypted, which is a working
    /// link that protects nothing.
    #[test_case]
    fn pairwise_and_group_keys_differ_in_the_multicast_bit() {
        let tk = [1u8; 16];
        let pw = add_pairwise_key(AP_STA_ID, 0, &tk, 0);
        let pf = u16::from_le_bytes([pw[K_KEY_FLAGS], pw[K_KEY_FLAGS + 1]]);
        assert_eq!(pf & KEY_MULTICAST, 0, "pairwise");
        assert_eq!(pw[K_KEY_OFFSET], 0);

        let gk = add_group_key(AP_STA_ID, 2, &[2u8; 16]).expect("16-byte GTK");
        let gf = u16::from_le_bytes([gk[K_KEY_FLAGS], gk[K_KEY_FLAGS + 1]]);
        assert_ne!(gf & KEY_MULTICAST, 0, "group");
        assert_ne!(gk[K_KEY_OFFSET], pw[K_KEY_OFFSET], "different slots");
        // The group key's id comes from the AP, not from the pairwise key.
        assert_eq!((gf >> KEY_FLG_KEYID_POS) & 0x3, 2);
        assert_eq!((pf >> KEY_FLG_KEYID_POS) & 0x3, 0);

        // A key that is not CCMP-128 is refused rather than truncated.
        assert!(add_group_key(AP_STA_ID, 1, &[0u8; 32]).is_none());
        assert!(add_group_key(AP_STA_ID, 1, &[0u8; 13]).is_none());
    }

    /// The transmit packet number **continues** rather than restarting: CCM is
    /// a counter mode, and a repeat under one key hands an observer the XOR of
    /// two plaintexts.
    #[test_case]
    fn the_transmit_packet_number_is_carried_into_the_key() {
        let k = add_pairwise_key(AP_STA_ID, 0, &[0; 16], 0x1234_5678_9abc);
        let pn = u64::from_le_bytes(k[K_TX_SEQ_CNT..K_TX_SEQ_CNT + 8].try_into().unwrap());
        assert_eq!(pn, 0x1234_5678_9abc);
        // Receive counters start at zero, so the peer's first frame must
        // advance past them.
        assert!(k[K_RX_SECUR_SEQ..K_RX_SECUR_SEQ + 16]
            .iter()
            .all(|&b| b == 0));
    }

    /// **A station flag set without its mask bit is ignored** — the quiet way
    /// `ADD_STA` succeeds and does nothing.
    #[test_case]
    fn station_flags_are_applied_only_where_the_mask_says() {
        let bssid = [0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff];
        let c = add_ap(1, 3, &bssid, 0x0007, 0xff);
        let flags = u32::from_le_bytes(c[S_STATION_FLAGS..S_STATION_FLAGS + 4].try_into().unwrap());
        let msk = u32::from_le_bytes(
            c[S_STATION_FLAGS_MSK..S_STATION_FLAGS_MSK + 4]
                .try_into()
                .unwrap(),
        );
        assert_ne!(flags & STA_FLG_CLASS_AUTH, 0);
        assert_ne!(flags & STA_FLG_CLASS_ASSOC, 0);
        assert_eq!(flags & msk, flags, "every flag set is also in the mask");

        assert_eq!(&c[S_ADDR..S_ADDR + 6], &bssid[..]);
        assert_eq!(c[S_STA_ID], AP_STA_ID);
        assert_eq!(c[S_ADD_MODIFY], MODE_ADD);
        assert_eq!(u16::from_le_bytes([c[S_ASSOC_ID], c[S_ASSOC_ID + 1]]), 7);
        // The TID field is a *disable* mask: zero is permissive, and 0xffff
        // would silence the station while every command still succeeded.
        assert_eq!(
            u16::from_le_bytes([c[S_TID_DISABLE_TX], c[S_TID_DISABLE_TX + 1]]),
            0
        );
    }

    /// **A context is identified by id *and* colour.** The colour changes when
    /// an id is reused, so a command carrying a stale one is rejected rather
    /// than applied to whatever now occupies that id — sending the id alone
    /// works until the first reuse.
    #[test_case]
    fn the_mac_context_carries_a_colour_not_just_an_id() {
        assert_eq!(mac_id_n_color(1, 0), 1);
        assert_eq!(mac_id_n_color(1, 3), 1 | (3 << 8));
        assert_ne!(
            mac_id_n_color(1, 3),
            mac_id_n_color(1, 4),
            "the colour is part of it"
        );
        let c = add_ap(2, 5, &[0; 6], 1, 1);
        let v = u32::from_le_bytes(
            c[S_MAC_ID_N_COLOR..S_MAC_ID_N_COLOR + 4]
                .try_into()
                .unwrap(),
        );
        assert_eq!(v, mac_id_n_color(2, 5));
    }
}
