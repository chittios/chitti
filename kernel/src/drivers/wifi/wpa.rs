//! **WPA2-PSK** — the key hierarchy, the EAPOL-Key handshake, and the crypto under both.
//!
//! This is the part of joining a network where a bug is invisible. Every step produces
//! bytes that look exactly as random as correct ones, and the only feedback is that the
//! access point stops talking to you — which is indistinguishable from a wrong password,
//! a weak signal, or a driver that never sent the frame. So the whole file is written to
//! be checkable off-hardware, and every primitive is pinned to a published vector or to an
//! independent implementation's output:
//!
//! - **SHA-1, HMAC-SHA-1, PBKDF2** live in [`crate::net::sha1`], against FIPS 180-2 and
//!   RFC 2202.
//! - **AES-128 and RFC 3394 key unwrap**, against FIPS 197 and the RFC's own vector, in
//!   both directions (the wrap half exists to check the unwrap, not for the client path).
//! - **The PMK, the PTK and the MIC**, against values computed independently for the
//!   fixtures below.
//!
//! Every constant and ordering here was then cross-checked against hostap's
//! `wpa_common.{h,c}` — the `wpa_eapol_key` field offsets (MIC at body offset 77, a 95-byte
//! fixed body), the `WPA_KEY_INFO_*` bit positions, the `"Pairwise key expansion"` label, the
//! memcmp-ordered address and nonce pairs, the 48-byte PTK sliced KCK/KEK/TK, and the MIC as
//! the first 16 bytes of HMAC-SHA-1. That matters because self-consistent tests cannot catch
//! a shared misunderstanding: this code and its fixtures would agree perfectly while the
//! access point disagreed.
//!
//! ## The shape of WPA2-PSK
//!
//! The passphrase becomes a **PMK** (PBKDF2 over the SSID). Then the access point and the
//! client exchange nonces, and both derive the same **PTK** from the PMK plus both MAC
//! addresses and both nonces — which is why neither side has to send a key. The PTK splits
//! into a **KCK** (authenticates the handshake frames), a **KEK** (encrypts the group key)
//! and a **TK** (encrypts traffic).
//!
//! Two orderings in that derivation are load-bearing and silent when wrong: the MAC
//! addresses and the nonces are each concatenated **smaller first**, numerically. Both
//! sides do it, so getting it wrong yields a PTK that is self-consistent and different
//! from the access point's — a MIC failure reported as a wrong password.
//!
//! ## Why there is no CCMP here
//!
//! The TK this file derives encrypts traffic, but **not in software**: both radio families
//! this kernel targets do CCMP in hardware. A FullMAC part (Broadcom) is given the
//! passphrase and runs the whole handshake in firmware; a SoftMAC part (Intel) is given the
//! TK through a key command and its hardware encrypts each frame. So the driver's job ends
//! at *deriving and installing* keys, which is what this is. A software CCMP
//! implementation would be code that never runs on either — worth noting explicitly, since
//! its absence otherwise looks like an unfinished layer rather than a decision.

use crate::net::sha1::{hmac_sha1, pbkdf2_sha1};
use alloc::vec::Vec;

/// WPA2 fixes this at 4096. Not a tunable: a different count is a different key, and the
/// access point is not going to negotiate.
pub const PSK_ITERATIONS: u32 = 4096;

/// Derive the pairwise master key from a passphrase and SSID.
pub fn pmk_from_passphrase(passphrase: &str, ssid: &[u8]) -> [u8; 32] {
    let mut pmk = [0u8; 32];
    pbkdf2_sha1(passphrase.as_bytes(), ssid, PSK_ITERATIONS, &mut pmk);
    pmk
}

/// The 802.11i pseudo-random function: HMAC-SHA-1 over `label || 0x00 || data || counter`,
/// concatenated until enough bytes exist.
///
/// The single NUL between label and data is part of the definition. Omitting it produces a
/// perfectly plausible key stream that no access point agrees with.
pub fn prf(key: &[u8], label: &[u8], data: &[u8], out_len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(out_len + 20);
    let mut counter = 0u8;
    while out.len() < out_len {
        let mut msg = Vec::with_capacity(label.len() + 2 + data.len());
        msg.extend_from_slice(label);
        msg.push(0);
        msg.extend_from_slice(data);
        msg.push(counter);
        out.extend_from_slice(&hmac_sha1(key, &msg));
        counter += 1;
    }
    out.truncate(out_len);
    out
}

/// The pairwise transient key, split into its three parts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ptk {
    /// Key confirmation key — authenticates EAPOL-Key frames.
    pub kck: [u8; 16],
    /// Key encryption key — unwraps the group key.
    pub kek: [u8; 16],
    /// Temporal key — encrypts traffic.
    pub tk: [u8; 16],
}

/// Derive the PTK.
///
/// `aa` is the access point's MAC, `spa` ours; `anonce` and `snonce` the two nonces. Each
/// pair is concatenated **smaller first**, which is the whole reason this takes them as
/// separate arguments rather than pre-joined: both sides must order them identically and
/// neither transmits the result, so a mistake here is only ever observed as a MIC failure.
pub fn derive_ptk(
    pmk: &[u8; 32],
    aa: &[u8; 6],
    spa: &[u8; 6],
    anonce: &[u8; 32],
    snonce: &[u8; 32],
) -> Ptk {
    let mut data = Vec::with_capacity(12 + 64);
    let (lo_mac, hi_mac) = if aa <= spa { (aa, spa) } else { (spa, aa) };
    data.extend_from_slice(lo_mac);
    data.extend_from_slice(hi_mac);
    let (lo_n, hi_n) = if anonce <= snonce {
        (anonce, snonce)
    } else {
        (snonce, anonce)
    };
    data.extend_from_slice(lo_n);
    data.extend_from_slice(hi_n);

    let k = prf(pmk, b"Pairwise key expansion", &data, 48);
    let mut p = Ptk {
        kck: [0; 16],
        kek: [0; 16],
        tk: [0; 16],
    };
    p.kck.copy_from_slice(&k[0..16]);
    p.kek.copy_from_slice(&k[16..32]);
    p.tk.copy_from_slice(&k[32..48]);
    p
}

/// Compute an EAPOL-Key MIC over a frame whose MIC field has been zeroed.
///
/// The zeroing is not a convenience — the MIC covers the frame *including* its own field,
/// so both sides compute it with that field clear. A MIC computed over a frame still
/// carrying the received value never matches.
pub fn eapol_mic(kck: &[u8; 16], frame_with_zeroed_mic: &[u8]) -> [u8; 16] {
    let full = hmac_sha1(kck, frame_with_zeroed_mic);
    let mut mic = [0u8; 16];
    mic.copy_from_slice(&full[..16]);
    mic
}

// --- AES-128 ---------------------------------------------------------------
//
// Needed for exactly one thing on the client path: unwrapping the group key out of the third
// handshake message. The encrypt direction exists only because it is what makes the unwrap
// checkable — see `aes128_encrypt_block`.

const SBOX: [u8; 256] = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
];

fn inv_sbox() -> [u8; 256] {
    let mut inv = [0u8; 256];
    for (i, &v) in SBOX.iter().enumerate() {
        inv[v as usize] = i as u8;
    }
    inv
}

/// Multiply in GF(2^8) — the field the AES mix step is defined over.
fn xtime(mut a: u8, mut b: u8) -> u8 {
    let mut p = 0u8;
    for _ in 0..8 {
        if b & 1 != 0 {
            p ^= a;
        }
        let hi = a & 0x80;
        a <<= 1;
        if hi != 0 {
            a ^= 0x1b; // the AES reduction polynomial
        }
        b >>= 1;
    }
    p
}

/// Expand a 128-bit key into the eleven round keys.
fn expand_key(key: &[u8; 16]) -> [[u8; 16]; 11] {
    let mut w = [[0u8; 16]; 11];
    w[0] = *key;
    let mut rcon = 1u8;
    for r in 1..11 {
        let prev = w[r - 1];
        let mut t = [prev[13], prev[14], prev[15], prev[12]]; // RotWord
        for b in t.iter_mut() {
            *b = SBOX[*b as usize]; // SubWord
        }
        t[0] ^= rcon;
        for i in 0..4 {
            w[r][i] = prev[i] ^ t[i];
        }
        for i in 4..16 {
            w[r][i] = prev[i] ^ w[r][i - 4];
        }
        rcon = xtime(rcon, 2);
    }
    w
}

/// Decrypt one 16-byte block in place.
pub fn aes128_decrypt_block(key: &[u8; 16], block: &mut [u8; 16]) {
    let rk = expand_key(key);
    let inv = inv_sbox();

    for (i, b) in block.iter_mut().enumerate() {
        *b ^= rk[10][i];
    }
    for round in (1..11).rev() {
        // InvShiftRows
        let s = *block;
        let mut t = [0u8; 16];
        for c in 0..4 {
            for r in 0..4 {
                t[((c + r) % 4) * 4 + r] = s[c * 4 + r];
            }
        }
        // InvSubBytes
        for b in t.iter_mut() {
            *b = inv[*b as usize];
        }
        // AddRoundKey
        for i in 0..16 {
            t[i] ^= rk[round - 1][i];
        }
        // InvMixColumns, except on the last round performed (round 1 here)
        if round > 1 {
            let mut m = [0u8; 16];
            for c in 0..4 {
                let col = &t[c * 4..c * 4 + 4];
                m[c * 4] =
                    xtime(col[0], 14) ^ xtime(col[1], 11) ^ xtime(col[2], 13) ^ xtime(col[3], 9);
                m[c * 4 + 1] =
                    xtime(col[0], 9) ^ xtime(col[1], 14) ^ xtime(col[2], 11) ^ xtime(col[3], 13);
                m[c * 4 + 2] =
                    xtime(col[0], 13) ^ xtime(col[1], 9) ^ xtime(col[2], 14) ^ xtime(col[3], 11);
                m[c * 4 + 3] =
                    xtime(col[0], 11) ^ xtime(col[1], 13) ^ xtime(col[2], 9) ^ xtime(col[3], 14);
            }
            *block = m;
        } else {
            *block = t;
        }
    }
}

/// Encrypt one 16-byte block in place.
///
/// **The client path never uses this** — it only ever unwraps a group key the access point
/// wrapped. It exists because it is what makes the unwrap checkable: RFC 3394 publishes the
/// forward direction, and a wrap/unwrap round trip catches an error in either half that a
/// one-directional test would let through.
pub fn aes128_encrypt_block(key: &[u8; 16], block: &mut [u8; 16]) {
    let rk = expand_key(key);
    for (i, b) in block.iter_mut().enumerate() {
        *b ^= rk[0][i];
    }
    for round in 1..11 {
        // SubBytes
        let mut t = [0u8; 16];
        for (i, b) in block.iter().enumerate() {
            t[i] = SBOX[*b as usize];
        }
        // ShiftRows
        let s = t;
        for c in 0..4 {
            for r in 0..4 {
                t[c * 4 + r] = s[((c + r) % 4) * 4 + r];
            }
        }
        // MixColumns, skipped on the final round
        if round < 10 {
            let mut m = [0u8; 16];
            for c in 0..4 {
                let col = &t[c * 4..c * 4 + 4];
                m[c * 4] = xtime(col[0], 2) ^ xtime(col[1], 3) ^ col[2] ^ col[3];
                m[c * 4 + 1] = col[0] ^ xtime(col[1], 2) ^ xtime(col[2], 3) ^ col[3];
                m[c * 4 + 2] = col[0] ^ col[1] ^ xtime(col[2], 2) ^ xtime(col[3], 3);
                m[c * 4 + 3] = xtime(col[0], 3) ^ col[1] ^ col[2] ^ xtime(col[3], 2);
            }
            t = m;
        }
        for i in 0..16 {
            t[i] ^= rk[round][i];
        }
        *block = t;
    }
}

/// The RFC 3394 default initial value. A wrapped key that does not decrypt to this has
/// been tampered with, or the KEK is wrong — and those are the same thing to a client.
pub const KEY_WRAP_IV: [u8; 8] = [0xa6; 8];

/// Wrap a key (RFC 3394) — the inverse of [`aes_key_unwrap`], for the same reason
/// [`aes128_encrypt_block`] exists: it is what proves the unwrap right in both directions.
pub fn aes_key_wrap(kek: &[u8; 16], plain: &[u8]) -> Option<Vec<u8>> {
    if plain.len() < 16 || plain.len() % 8 != 0 {
        return None;
    }
    let n = plain.len() / 8;
    let mut a = KEY_WRAP_IV;
    let mut r: Vec<[u8; 8]> = plain
        .chunks_exact(8)
        .map(|c| {
            let mut b = [0u8; 8];
            b.copy_from_slice(c);
            b
        })
        .collect();
    for j in 0..6 {
        for i in 1..=n {
            let mut blk = [0u8; 16];
            blk[..8].copy_from_slice(&a);
            blk[8..].copy_from_slice(&r[i - 1]);
            aes128_encrypt_block(kek, &mut blk);
            let t = (n * j + i) as u64;
            for k in 0..8 {
                a[k] = blk[k] ^ t.to_be_bytes()[k];
            }
            r[i - 1].copy_from_slice(&blk[8..]);
        }
    }
    let mut out = Vec::with_capacity(plain.len() + 8);
    out.extend_from_slice(&a);
    for b in &r {
        out.extend_from_slice(b);
    }
    Some(out)
}

/// Unwrap an AES-wrapped key (RFC 3394).
///
/// `None` when the integrity check fails, which is the only defence against accepting a
/// group key an attacker chose. `wrapped` must be a multiple of 8 bytes and at least 24.
pub fn aes_key_unwrap(kek: &[u8; 16], wrapped: &[u8]) -> Option<Vec<u8>> {
    if wrapped.len() < 24 || wrapped.len() % 8 != 0 {
        return None;
    }
    let n = wrapped.len() / 8 - 1;
    let mut a = [0u8; 8];
    a.copy_from_slice(&wrapped[..8]);
    let mut r: Vec<[u8; 8]> = wrapped[8..]
        .chunks_exact(8)
        .map(|c| {
            let mut b = [0u8; 8];
            b.copy_from_slice(c);
            b
        })
        .collect();

    // Six passes, backwards, per the RFC. The counter is `n * j + i`, and it is mixed into
    // A *before* the block decrypt — reversing that order decrypts cleanly and produces
    // garbage that fails the integrity check, which is the good outcome, but the same
    // mistake in a wrap would produce keys nobody can unwrap.
    for j in (0..6).rev() {
        for i in (1..=n).rev() {
            let t = (n * j + i) as u64;
            let mut blk = [0u8; 16];
            for k in 0..8 {
                blk[k] = a[k] ^ t.to_be_bytes()[k];
            }
            blk[8..].copy_from_slice(&r[i - 1]);
            aes128_decrypt_block(kek, &mut blk);
            a.copy_from_slice(&blk[..8]);
            r[i - 1].copy_from_slice(&blk[8..]);
        }
    }
    if a != KEY_WRAP_IV {
        return None;
    }
    Some(r.concat())
}

// --- EAPOL-Key: the four-way handshake ------------------------------------

/// EAPOL version and packet type for a Key frame.
pub const EAPOL_VERSION: u8 = 2;
pub const EAPOL_TYPE_KEY: u8 = 3;
/// Descriptor type 2 is the 802.11i / WPA2 key descriptor. WPA1 used 254.
pub const KEY_DESC_RSN: u8 = 2;
/// Fixed length of an EAPOL-Key body, from the descriptor type through the key-data length.
pub const EAPOL_KEY_BODY_LEN: usize = 95;
/// Offset of the MIC field within the whole EAPOL frame (4-byte EAPOL header + 77).
pub const EAPOL_MIC_OFFSET: usize = 4 + 77;

/// Key Information bits. These are what distinguish the four messages from each other —
/// there is no message number in the frame.
pub const KEY_INFO_PAIRWISE: u16 = 1 << 3;
pub const KEY_INFO_INSTALL: u16 = 1 << 6;
pub const KEY_INFO_ACK: u16 = 1 << 7;
pub const KEY_INFO_MIC: u16 = 1 << 8;
pub const KEY_INFO_SECURE: u16 = 1 << 9;
pub const KEY_INFO_ERROR: u16 = 1 << 10;
pub const KEY_INFO_REQUEST: u16 = 1 << 11;
pub const KEY_INFO_ENCRYPTED: u16 = 1 << 12;
/// Key-descriptor-version 2 = HMAC-SHA1 MIC with AES key wrap, which is CCMP's pairing.
pub const KEY_INFO_VERSION_MASK: u16 = 0x7;

/// A parsed EAPOL-Key frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EapolKey {
    pub key_info: u16,
    pub key_len: u16,
    pub replay_counter: u64,
    pub nonce: [u8; 32],
    pub rsc: [u8; 8],
    pub mic: [u8; 16],
    pub key_data: Vec<u8>,
}

/// Which message of the handshake a frame is, deduced from its Key Information bits.
///
/// The frame carries no message number, so this **is** the identification — and message 1
/// and 3 differ only by the MIC and Secure bits. Treating a replayed message 1 as a 3 (or
/// the reverse) is the shape of the well-known handshake attacks, so the classification is
/// explicit rather than positional.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandshakeMessage {
    /// AP → client: the ANonce, no MIC (no key exists yet to compute one with).
    One,
    /// Client → AP: the SNonce plus a MIC.
    Two,
    /// AP → client: MIC, Install, Secure, and the encrypted group key.
    Three,
    /// Client → AP: confirmation. Distinguished from [`Self::Two`] **only** by the Secure
    /// bit — it carries no Install bit, despite being the message that confirms the key was
    /// installed.
    Four,
}

impl EapolKey {
    /// Parse an EAPOL frame (starting at the EAPOL header, not the 802.11 header).
    ///
    /// These arrive before any key is confirmed, from whatever is claiming to be the access
    /// point, so every length is checked. In particular the key-data length is a claim: a
    /// frame declaring 60 KiB of key data in a 200-byte packet is refused rather than
    /// clamped.
    pub fn parse(frame: &[u8]) -> Option<EapolKey> {
        if frame.len() < 4 + EAPOL_KEY_BODY_LEN {
            return None;
        }
        if frame[1] != EAPOL_TYPE_KEY {
            return None;
        }
        let declared = u16::from_be_bytes([frame[2], frame[3]]) as usize;
        let body = &frame[4..];
        if declared < EAPOL_KEY_BODY_LEN || body.len() < declared {
            return None;
        }
        let body = &body[..declared];
        if body[0] != KEY_DESC_RSN {
            return None;
        }
        let data_len = u16::from_be_bytes([body[93], body[94]]) as usize;
        if body.len() < EAPOL_KEY_BODY_LEN + data_len {
            return None;
        }
        let mut nonce = [0u8; 32];
        nonce.copy_from_slice(&body[13..45]);
        let mut rsc = [0u8; 8];
        rsc.copy_from_slice(&body[61..69]);
        let mut mic = [0u8; 16];
        mic.copy_from_slice(&body[77..93]);
        Some(EapolKey {
            key_info: u16::from_be_bytes([body[1], body[2]]),
            key_len: u16::from_be_bytes([body[3], body[4]]),
            replay_counter: u64::from_be_bytes([
                body[5], body[6], body[7], body[8], body[9], body[10], body[11], body[12],
            ]),
            nonce,
            rsc,
            mic,
            key_data: body[EAPOL_KEY_BODY_LEN..EAPOL_KEY_BODY_LEN + data_len].to_vec(),
        })
    }

    /// Which handshake message this is, or `None` for a combination that is none of them.
    ///
    /// Messages **2 and 4 differ only by the Secure bit** — 4 carries no Install and no Ack,
    /// exactly like 2. That is the one distinction available, and it is worth spelling out:
    /// the obvious guess is that message 4 sets Install (it is the one that confirms the key
    /// was installed), and a classifier built on that reads every message 4 as a message 2.
    /// Nothing catches it in normal operation, because a client never receives either.
    pub fn message(&self) -> Option<HandshakeMessage> {
        let pairwise = self.key_info & KEY_INFO_PAIRWISE != 0;
        let mic = self.key_info & KEY_INFO_MIC != 0;
        let ack = self.key_info & KEY_INFO_ACK != 0;
        let secure = self.key_info & KEY_INFO_SECURE != 0;
        if !pairwise {
            return None; // a group-key rekey, not part of the four-way handshake
        }
        match (mic, ack) {
            // Message 1 is the only one without a MIC: no key exists yet to compute one.
            (false, true) => Some(HandshakeMessage::One),
            // Both AP-to-client messages set Ack; 3 is the one with a MIC.
            (true, true) => Some(HandshakeMessage::Three),
            (true, false) if secure => Some(HandshakeMessage::Four),
            (true, false) => Some(HandshakeMessage::Two),
            _ => None,
        }
    }

    /// Serialise, with the MIC field zeroed — the form the MIC is computed over.
    pub fn to_bytes_for_mic(&self) -> Vec<u8> {
        self.encode(&[0u8; 16])
    }

    /// Serialise with a MIC in place.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.encode(&self.mic)
    }

    fn encode(&self, mic: &[u8; 16]) -> Vec<u8> {
        let body_len = EAPOL_KEY_BODY_LEN + self.key_data.len();
        let mut f = Vec::with_capacity(4 + body_len);
        f.push(EAPOL_VERSION);
        f.push(EAPOL_TYPE_KEY);
        f.extend_from_slice(&(body_len as u16).to_be_bytes());
        f.push(KEY_DESC_RSN);
        f.extend_from_slice(&self.key_info.to_be_bytes());
        f.extend_from_slice(&self.key_len.to_be_bytes());
        f.extend_from_slice(&self.replay_counter.to_be_bytes());
        f.extend_from_slice(&self.nonce);
        f.extend_from_slice(&[0u8; 16]); // key IV, unused with descriptor version 2
        f.extend_from_slice(&self.rsc);
        f.extend_from_slice(&[0u8; 8]); // reserved (the old Key ID field)
        f.extend_from_slice(mic);
        f.extend_from_slice(&(self.key_data.len() as u16).to_be_bytes());
        f.extend_from_slice(&self.key_data);
        f
    }

    /// Whether this frame's MIC is the one `kck` produces over it.
    pub fn mic_valid(&self, kck: &[u8; 16]) -> bool {
        // Compared byte-wise rather than by an early-exit search: this is not a secret-
        // dependent timing concern here (the attacker already knows the MIC they sent), but
        // a whole-array compare is also simply harder to get wrong.
        eapol_mic(kck, &self.to_bytes_for_mic()) == self.mic
    }
}

/// The group temporal key, as delivered inside message 3.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gtk {
    pub id: u8,
    pub key: Vec<u8>,
}

/// KDE header for the GTK: vendor element, OUI 00-0f-ac, data type 1.
const KDE_OUI: [u8; 3] = [0x00, 0x0f, 0xac];
const KDE_TYPE_GTK: u8 = 1;

/// Find the GTK in decrypted key data.
///
/// The key data is a sequence of KDEs and RSN elements; the GTK arrives as a vendor-specific
/// KDE with two bytes of key-id/tx flags before the key itself. Bounded like every other
/// off-the-air parse — the bytes were encrypted, which proves the sender knew the KEK, not
/// that the contents are well formed.
pub fn find_gtk(key_data: &[u8]) -> Option<Gtk> {
    let mut at = 0usize;
    while at + 2 <= key_data.len() {
        let id = key_data[at];
        let len = key_data[at + 1] as usize;
        if len == 0 || at + 2 + len > key_data.len() {
            return None;
        }
        let body = &key_data[at + 2..at + 2 + len];
        // 0xdd with an all-zero body is the padding that fills the wrapped block.
        if id == 0xdd && body.len() > 6 && body[..3] == KDE_OUI && body[3] == KDE_TYPE_GTK {
            return Some(Gtk {
                id: body[4] & 0x03,
                key: body[6..].to_vec(),
            });
        }
        at += 2 + len;
    }
    None
}

/// A four-way handshake in progress.
///
/// Deliberately a pure state machine over frames: it never touches a radio, so the whole
/// handshake — including the failure paths that matter most — is testable off-hardware.
#[derive(Debug, Clone)]
pub struct Handshake {
    pmk: [u8; 32],
    /// Our MAC.
    spa: [u8; 6],
    /// The access point's MAC.
    aa: [u8; 6],
    snonce: [u8; 32],
    /// Our association request's RSN element — the **whole** element, including its id and
    /// length bytes, because that is what message 2's key data carries. Passing the body
    /// alone yields a malformed element the AP rejects mid-handshake, which looks like a key
    /// failure.
    rsn_element: Vec<u8>,
    ptk: Option<Ptk>,
    /// Highest replay counter seen. A frame at or below it is a replay.
    last_replay: Option<u64>,
    pub gtk: Option<Gtk>,
    pub done: bool,
}

impl Handshake {
    pub fn new(
        pmk: [u8; 32],
        spa: [u8; 6],
        aa: [u8; 6],
        snonce: [u8; 32],
        rsn_element: Vec<u8>,
    ) -> Handshake {
        Handshake {
            pmk,
            spa,
            aa,
            snonce,
            rsn_element,
            ptk: None,
            last_replay: None,
            gtk: None,
            done: false,
        }
    }

    /// The derived pairwise key, once message 1 has arrived.
    pub fn ptk(&self) -> Option<&Ptk> {
        self.ptk.as_ref()
    }

    /// Feed a received EAPOL-Key frame; returns the frame to send in reply, if any.
    ///
    /// Errors are returned rather than logged-and-ignored because each one means something
    /// specific to a user: a MIC failure on message 3 is a wrong passphrase, while a replay
    /// is an attack or a retransmission.
    pub fn on_frame(&mut self, frame: &[u8]) -> Result<Option<Vec<u8>>, &'static str> {
        let key = EapolKey::parse(frame).ok_or("malformed EAPOL-Key frame")?;
        match key
            .message()
            .ok_or("EAPOL-Key frame is not a handshake message")?
        {
            HandshakeMessage::One => self.on_msg1(&key).map(Some),
            HandshakeMessage::Three => self.on_msg3(&key).map(Some),
            // The AP does not send these; receiving one means something is impersonating a
            // client, and answering it would be free work for an attacker.
            HandshakeMessage::Two | HandshakeMessage::Four => Err("unexpected handshake message"),
        }
    }

    fn on_msg1(&mut self, key: &EapolKey) -> Result<Vec<u8>, &'static str> {
        // Message 1 carries no MIC, so a replay cannot be detected by authentication — only
        // by the counter. Accepting an old one would restart the handshake with a nonce an
        // attacker has already seen traffic under.
        if let Some(last) = self.last_replay {
            if key.replay_counter <= last {
                return Err("replayed EAPOL-Key message 1");
            }
        }
        self.last_replay = Some(key.replay_counter);
        let ptk = derive_ptk(&self.pmk, &self.aa, &self.spa, &key.nonce, &self.snonce);
        self.ptk = Some(ptk);

        let mut msg2 = EapolKey {
            // Descriptor version copied from the AP's frame: it chose the MIC algorithm and
            // disagreeing means every MIC we send is computed the wrong way.
            key_info: (key.key_info & KEY_INFO_VERSION_MASK) | KEY_INFO_PAIRWISE | KEY_INFO_MIC,
            key_len: 0,
            replay_counter: key.replay_counter,
            nonce: self.snonce,
            rsc: [0; 8],
            mic: [0; 16],
            key_data: self.rsn_element.clone(),
        };
        msg2.mic = eapol_mic(&ptk.kck, &msg2.to_bytes_for_mic());
        Ok(msg2.to_bytes())
    }

    fn on_msg3(&mut self, key: &EapolKey) -> Result<Vec<u8>, &'static str> {
        let ptk = *self.ptk.as_ref().ok_or("message 3 before message 1")?;

        // The ANonce is checked **before** the MIC, and the order is the whole point. A
        // message 3 carrying a different ANonce was MIC'd by the AP under *its* PTK, so the
        // MIC fails too — and checking the MIC first would report that as a wrong
        // passphrase, sending the user off to retype a password that was always correct.
        let from_this_nonce = derive_ptk(&self.pmk, &self.aa, &self.spa, &key.nonce, &self.snonce);
        if from_this_nonce != ptk {
            return Err("message 3 carries a different ANonce than message 1");
        }
        // With the nonce confirmed, a MIC mismatch really does mean the two sides derived
        // different keys from the same inputs — which leaves the passphrase as the only
        // input that can differ. This is the first and only point where a wrong password
        // becomes visible.
        if !key.mic_valid(&ptk.kck) {
            return Err("EAPOL-Key MIC mismatch — wrong passphrase");
        }
        if let Some(last) = self.last_replay {
            if key.replay_counter < last {
                return Err("replayed EAPOL-Key message 3");
            }
        }
        self.last_replay = Some(key.replay_counter);

        if key.key_info & KEY_INFO_ENCRYPTED != 0 && !key.key_data.is_empty() {
            let plain = aes_key_unwrap(&ptk.kek, &key.key_data)
                .ok_or("group key failed its integrity check")?;
            self.gtk = find_gtk(&plain);
        }

        let mut msg4 = EapolKey {
            key_info: (key.key_info & KEY_INFO_VERSION_MASK)
                | KEY_INFO_PAIRWISE
                | KEY_INFO_MIC
                | KEY_INFO_SECURE,
            key_len: 0,
            replay_counter: key.replay_counter,
            // Message 4 carries a zero nonce. Echoing the SNonce here is a common mistake
            // that some access points accept and others reject.
            nonce: [0; 32],
            rsc: [0; 8],
            mic: [0; 16],
            key_data: Vec::new(),
        };
        msg4.mic = eapol_mic(&ptk.kck, &msg4.to_bytes_for_mic());
        self.done = true;
        Ok(msg4.to_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(b: &[u8]) -> alloc::string::String {
        let mut s = alloc::string::String::new();
        for x in b {
            s.push_str(&alloc::format!("{x:02x}"));
        }
        s
    }

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len() / 2)
            .map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap())
            .collect()
    }

    #[test_case]
    fn the_pmk_comes_from_the_passphrase_and_ssid() {
        // Same IEEE 802.11i vector the PBKDF2 layer is pinned to, exercised through the
        // WPA-facing entry point so the iteration count and salt order are covered too:
        // the SSID is the *salt*, and swapping them yields a plausible wrong key.
        let pmk = pmk_from_passphrase("password", b"IEEE");
        assert_eq!(
            hex(&pmk),
            "f42c6fc52df0ebef9ebb4b90b38a5f902e83fe1b135a70e23aed762e9710a12e"
        );
        assert_eq!(PSK_ITERATIONS, 4096);
    }

    #[test_case]
    fn the_ptk_matches_an_independently_computed_hierarchy() {
        // Computed from the same primitives by a separate implementation. This is the one
        // value in WPA2 that neither side transmits, so a mismatch is only ever visible as
        // a MIC failure the access point reports as a wrong password.
        let pmk = TEST_PMK; // == pmk_from_passphrase("password", b"IEEE"), pinned above
        let aa = [0x02, 0x03, 0x04, 0x05, 0x06, 0x07];
        let spa = [0xa0, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5];
        let mut anonce = [0u8; 32];
        let mut snonce = [0u8; 32];
        for i in 0..32 {
            anonce[i] = i as u8;
            snonce[i] = (i + 32) as u8;
        }
        let p = derive_ptk(&pmk, &aa, &spa, &anonce, &snonce);
        assert_eq!(hex(&p.kck), "ef444a66314d1d722bfcb60529e57641");
        assert_eq!(hex(&p.kek), "bc545f6cea5a7bbd9e586ddb57ea5968");
        assert_eq!(hex(&p.tk), "437c707f15b7ce682d7323e97cce648c");
    }

    #[test_case]
    fn the_ptk_is_the_same_whichever_side_derives_it() {
        // The point of ordering both pairs smaller-first: the access point and the client
        // hold the arguments in opposite roles and must still land on one key. Swapping
        // both pairs is exactly what the peer does.
        let pmk = TEST_PMK;
        let a = [0x02u8, 0, 0, 0, 0, 1];
        let s = [0x02u8, 0, 0, 0, 0, 2];
        let n1 = [0x11u8; 32];
        let n2 = [0x22u8; 32];
        assert_eq!(
            derive_ptk(&pmk, &a, &s, &n1, &n2),
            derive_ptk(&pmk, &s, &a, &n2, &n1),
            "the two sides derived different keys"
        );
    }

    #[test_case]
    fn every_input_changes_the_ptk() {
        // A derivation that silently ignored a nonce would still produce a working-looking
        // key — and would reuse it across sessions, which is the failure that makes the
        // whole handshake pointless.
        let pmk = TEST_PMK;
        let a = [1u8, 0, 0, 0, 0, 1];
        let s = [2u8, 0, 0, 0, 0, 2];
        let n1 = [0x11u8; 32];
        let n2 = [0x22u8; 32];
        let base = derive_ptk(&pmk, &a, &s, &n1, &n2);
        let mut other_pmk = TEST_PMK;
        other_pmk[0] ^= 1;
        assert_ne!(derive_ptk(&other_pmk, &a, &s, &n1, &n2), base);
        assert_ne!(derive_ptk(&pmk, &a, &[3u8, 0, 0, 0, 0, 3], &n1, &n2), base);
        assert_ne!(derive_ptk(&pmk, &a, &s, &[0x33u8; 32], &n2), base);
        assert_ne!(derive_ptk(&pmk, &a, &s, &n1, &[0x44u8; 32]), base);
        // And the three parts are distinct slices of the stream, not the same bytes.
        assert_ne!(base.kck, base.kek);
        assert_ne!(base.kek, base.tk);
    }

    #[test_case]
    fn the_prf_puts_a_nul_between_label_and_data() {
        // Part of the definition, and omitting it produces a plausible key stream that no
        // access point agrees with. Checked by computing the first block directly.
        let key = [0x0bu8; 20];
        let got = prf(&key, b"L", b"D", 20);
        let mut msg = alloc::vec::Vec::from(*b"L");
        msg.push(0);
        msg.extend_from_slice(b"D");
        msg.push(0); // counter
        assert_eq!(got[..], hmac_sha1(&key, &msg)[..]);
        // And it concatenates with an incrementing counter rather than repeating.
        let long = prf(&key, b"L", b"D", 40);
        assert_ne!(long[0..20], long[20..40], "second block repeats the first");
        assert_eq!(long.len(), 40);
    }

    #[test_case]
    fn the_eapol_mic_is_the_first_sixteen_bytes_of_hmac_sha1() {
        // Truncation is part of the protocol; sending the full 20 bytes overruns the field.
        let frame = unhex("0203007502010a001000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000");
        let kck = {
            let mut k = [0u8; 16];
            k.copy_from_slice(&unhex("ef444a66314d1d722bfcb60529e57641"));
            k
        };
        assert_eq!(frame.len(), 94);
        assert_eq!(
            hex(&eapol_mic(&kck, &frame)),
            "3bd7333f98ba70268a0a983084ef93f9"
        );
    }

    #[test_case]
    fn aes128_decrypts_the_fips_197_vector() {
        // FIPS 197 appendix B, run backwards: the known ciphertext must decrypt to the
        // known plaintext. Everything in the key unwrap rests on this one block cipher.
        let mut key = [0u8; 16];
        key.copy_from_slice(&unhex("000102030405060708090a0b0c0d0e0f"));
        let mut blk = [0u8; 16];
        blk.copy_from_slice(&unhex("69c4e0d86a7b0430d8cdb78070b4c55a"));
        aes128_decrypt_block(&key, &mut blk);
        assert_eq!(hex(&blk), "00112233445566778899aabbccddeeff");
    }

    #[test_case]
    fn aes_key_unwrap_matches_the_rfc_3394_vector() {
        // Section 4.1: a 128-bit key wrapped under a 128-bit KEK. This is how the group key
        // arrives in the third handshake message.
        let mut kek = [0u8; 16];
        kek.copy_from_slice(&unhex("000102030405060708090a0b0c0d0e0f"));
        let wrapped = unhex("1fa68b0a8112b447aef34bd8fb5a7b829d3e862371d2cfe5");
        let out = aes_key_unwrap(&kek, &wrapped).expect("the RFC's own vector must unwrap");
        assert_eq!(hex(&out), "00112233445566778899aabbccddeeff");
    }

    #[test_case]
    fn aes_wrap_and_unwrap_are_inverses_in_both_directions() {
        // RFC 3394 section 4.1 forwards, which pins the block cipher's encrypt half too, and
        // then a round trip at every legal length. A one-directional test would let an error
        // in either half through, since a consistent-but-wrong pair still round-trips —
        // hence the published vector as well.
        let mut kek = [0u8; 16];
        kek.copy_from_slice(&unhex("000102030405060708090a0b0c0d0e0f"));
        let key = unhex("00112233445566778899aabbccddeeff");
        assert_eq!(
            hex(&aes_key_wrap(&kek, &key).unwrap()),
            "1fa68b0a8112b447aef34bd8fb5a7b829d3e862371d2cfe5"
        );
        // FIPS 197 forwards, for the same reason.
        let mut blk = [0u8; 16];
        blk.copy_from_slice(&key);
        aes128_encrypt_block(&kek, &mut blk);
        assert_eq!(hex(&blk), "69c4e0d86a7b0430d8cdb78070b4c55a");

        for len in [16usize, 24, 32, 48] {
            let plain: Vec<u8> = (0..len).map(|i| i as u8).collect();
            let w = aes_key_wrap(&kek, &plain).unwrap();
            assert_eq!(w.len(), len + 8);
            assert_eq!(aes_key_unwrap(&kek, &w).unwrap(), plain, "len {len}");
        }
        // Lengths the RFC does not define.
        assert!(aes_key_wrap(&kek, &[0u8; 8]).is_none());
        assert!(aes_key_wrap(&kek, &[0u8; 20]).is_none());
    }

    #[test_case]
    fn a_tampered_or_wrongly_keyed_wrap_is_refused() {
        // The integrity check is the only thing stopping a client from installing a group
        // key an attacker chose, so it has to reject rather than return plausible bytes.
        let mut kek = [0u8; 16];
        kek.copy_from_slice(&unhex("000102030405060708090a0b0c0d0e0f"));
        let good = unhex("1fa68b0a8112b447aef34bd8fb5a7b829d3e862371d2cfe5");

        let mut bad = good.clone();
        bad[5] ^= 1;
        assert!(
            aes_key_unwrap(&kek, &bad).is_none(),
            "tampered wrap accepted"
        );

        let mut wrong_kek = kek;
        wrong_kek[0] ^= 1;
        assert!(
            aes_key_unwrap(&wrong_kek, &good).is_none(),
            "wrong KEK accepted"
        );

        // Malformed lengths, which a hostile frame can also present.
        assert!(aes_key_unwrap(&kek, &good[..16]).is_none());
        assert!(aes_key_unwrap(&kek, &good[..23]).is_none());
        assert!(aes_key_unwrap(&kek, &[]).is_none());
    }

    // --- the handshake ----------------------------------------------------
    //
    // These tests are an access point: they build the frames a real one sends, using the
    // same primitives from the other side. That is weaker than a capture would be for the
    // wire format, and exactly as strong for the *logic* — which is what has the failure
    // paths worth pinning (a wrong passphrase, a replay, a lying length).

    const AP: [u8; 6] = [0x02, 0x03, 0x04, 0x05, 0x06, 0x07];
    const ME: [u8; 6] = [0xa0, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5];

    /// The PMK the 802.11i vector produces, as a literal.
    ///
    /// Deliberately not re-derived per test: PBKDF2 is 4096 iterations by design, and the
    /// handshake tests below would spend the whole unit suite's time on a derivation that
    /// `the_pmk_comes_from_the_passphrase_and_ssid` already pins. Kept equal to it by
    /// `the_literal_pmk_is_the_one_the_vector_produces`, so the shortcut cannot drift.
    const TEST_PMK: [u8; 32] = [
        0xf4, 0x2c, 0x6f, 0xc5, 0x2d, 0xf0, 0xeb, 0xef, 0x9e, 0xbb, 0x4b, 0x90, 0xb3, 0x8a, 0x5f,
        0x90, 0x2e, 0x83, 0xfe, 0x1b, 0x13, 0x5a, 0x70, 0xe2, 0x3a, 0xed, 0x76, 0x2e, 0x97, 0x10,
        0xa1, 0x2e,
    ];

    fn test_pmk() -> [u8; 32] {
        TEST_PMK
    }

    #[test_case]
    fn the_literal_pmk_is_the_one_the_vector_produces() {
        // The one place the shortcut is checked against the real derivation.
        assert_eq!(pmk_from_passphrase("password", b"IEEE"), TEST_PMK);
    }

    fn nonce(seed: u8) -> [u8; 32] {
        let mut n = [0u8; 32];
        for (i, b) in n.iter_mut().enumerate() {
            *b = seed ^ i as u8;
        }
        n
    }

    /// Message 1: the ANonce, no MIC — there is no key yet to compute one with.
    fn ap_msg1(anonce: [u8; 32], replay: u64) -> Vec<u8> {
        EapolKey {
            key_info: 2 | KEY_INFO_PAIRWISE | KEY_INFO_ACK,
            key_len: 16,
            replay_counter: replay,
            nonce: anonce,
            rsc: [0; 8],
            mic: [0; 16],
            key_data: Vec::new(),
        }
        .to_bytes()
    }

    /// Message 3: MIC, Install, Secure, and the group key wrapped under the KEK.
    fn ap_msg3(ptk: &Ptk, anonce: [u8; 32], replay: u64, gtk: &[u8]) -> Vec<u8> {
        let mut kde = alloc::vec![
            0xdd,
            (4 + 2 + gtk.len()) as u8,
            0x00,
            0x0f,
            0xac,
            0x01,
            0x01,
            0x00
        ];
        kde.extend_from_slice(gtk);
        while kde.len() % 8 != 0 {
            kde.push(0xdd); // the RFC 3394 input must be a multiple of 8
        }
        let mut k = EapolKey {
            key_info: 2
                | KEY_INFO_PAIRWISE
                | KEY_INFO_MIC
                | KEY_INFO_ACK
                | KEY_INFO_INSTALL
                | KEY_INFO_SECURE
                | KEY_INFO_ENCRYPTED,
            key_len: 16,
            replay_counter: replay,
            nonce: anonce,
            rsc: [0; 8],
            mic: [0; 16],
            key_data: aes_key_wrap(&ptk.kek, &kde).unwrap(),
        };
        k.mic = eapol_mic(&ptk.kck, &k.to_bytes_for_mic());
        k.to_bytes()
    }

    #[test_case]
    fn the_four_way_handshake_completes_and_installs_the_group_key() {
        let anonce = nonce(0x11);
        let snonce = nonce(0x22);
        // The whole element, id and length included — that is what the key data carries, and
        // what the AP compares against the element we associated with.
        let rsn = super::super::ieee80211::client_rsn_element();
        let mut hs = Handshake::new(test_pmk(), ME, AP, snonce, rsn.clone());

        // Message 1 in, message 2 out — carrying our SNonce, a MIC, and our RSN element.
        let msg2 = hs
            .on_frame(&ap_msg1(anonce, 1))
            .expect("message 1 must be accepted")
            .expect("message 1 must be answered");
        let parsed = EapolKey::parse(&msg2).expect("our own frame must parse");
        assert_eq!(parsed.message(), Some(HandshakeMessage::Two));
        assert_eq!(parsed.nonce, snonce);
        assert_eq!(
            parsed.replay_counter, 1,
            "message 2 echoes the AP's counter"
        );
        assert_eq!(
            parsed.key_data, rsn,
            "message 2 must carry our RSN element verbatim, id and length included"
        );

        // The AP derives the same PTK from its side and can check that MIC.
        let ptk = derive_ptk(&test_pmk(), &AP, &ME, &anonce, &snonce);
        assert_eq!(hs.ptk(), Some(&ptk));
        assert!(
            parsed.mic_valid(&ptk.kck),
            "the AP would reject our message 2"
        );

        // Message 3 in, message 4 out, group key installed.
        let gtk = [0x5au8; 16];
        let msg4 = hs
            .on_frame(&ap_msg3(&ptk, anonce, 2, &gtk))
            .expect("a correctly MIC'd message 3 must be accepted")
            .expect("message 3 must be answered");
        let parsed4 = EapolKey::parse(&msg4).unwrap();
        assert_eq!(parsed4.message(), Some(HandshakeMessage::Four));
        assert!(parsed4.mic_valid(&ptk.kck));
        // Message 4 carries a zero nonce — echoing the SNonce is a common mistake that some
        // access points accept and others reject.
        assert_eq!(parsed4.nonce, [0u8; 32]);
        assert!(hs.done);
        assert_eq!(hs.gtk.as_ref().map(|g| g.key.clone()), Some(gtk.to_vec()));
        assert_eq!(hs.gtk.as_ref().unwrap().id, 1);
    }

    #[test_case]
    fn messages_two_and_four_are_told_apart_by_the_secure_bit_alone() {
        // The trap: message 4 is the one that confirms the key was installed, so the obvious
        // guess is that it carries the Install bit. It does not — it looks exactly like
        // message 2 except for Secure. A classifier built on Install reads every message 4 as
        // a message 2, and nothing catches it, because a client never receives either.
        let base = EapolKey {
            key_info: 2 | KEY_INFO_PAIRWISE | KEY_INFO_MIC,
            key_len: 0,
            replay_counter: 1,
            nonce: [0; 32],
            rsc: [0; 8],
            mic: [0; 16],
            key_data: Vec::new(),
        };
        assert_eq!(base.message(), Some(HandshakeMessage::Two));
        let mut four = base.clone();
        four.key_info |= KEY_INFO_SECURE;
        assert_eq!(four.message(), Some(HandshakeMessage::Four));
        // Install must not be what decides it.
        let mut two_with_install = base.clone();
        two_with_install.key_info |= KEY_INFO_INSTALL;
        assert_eq!(two_with_install.message(), Some(HandshakeMessage::Two));
        // And the AP's two messages, which both set Ack.
        let mut one = base.clone();
        one.key_info = 2 | KEY_INFO_PAIRWISE | KEY_INFO_ACK;
        assert_eq!(one.message(), Some(HandshakeMessage::One));
        let mut three = base.clone();
        three.key_info |= KEY_INFO_ACK | KEY_INFO_INSTALL | KEY_INFO_SECURE;
        assert_eq!(three.message(), Some(HandshakeMessage::Three));
    }

    #[test_case]
    fn a_wrong_passphrase_fails_at_message_three_and_says_so() {
        // This is the only point in the whole exchange where a wrong password becomes
        // visible: messages 1 and 2 succeed regardless, because neither side has proved
        // anything yet. The error text is what the user is shown, so it has to be the right
        // diagnosis rather than "handshake failed".
        let anonce = nonce(0x33);
        // A PMK one bit away from the right one: the same thing a wrong passphrase produces,
        // without a second PBKDF2 in the suite.
        let mut wrong = TEST_PMK;
        wrong[31] ^= 1;
        let mut hs = Handshake::new(wrong, ME, AP, nonce(0x44), Vec::new());
        assert!(
            hs.on_frame(&ap_msg1(anonce, 1)).is_ok(),
            "message 1 always succeeds"
        );

        let real_ptk = derive_ptk(&test_pmk(), &AP, &ME, &anonce, &nonce(0x44));
        let err = hs
            .on_frame(&ap_msg3(&real_ptk, anonce, 2, &[0u8; 16]))
            .expect_err("a MIC computed under a different PMK must be rejected");
        assert!(err.contains("passphrase"), "unhelpful error: {err}");
        assert!(!hs.done);
        assert!(
            hs.gtk.is_none(),
            "no key may be installed after a MIC failure"
        );
    }

    #[test_case]
    fn a_replayed_message_one_is_refused() {
        // Message 1 has no MIC, so the replay counter is the only defence. Accepting an old
        // one restarts the handshake under a nonce an attacker has already seen traffic
        // encrypted with.
        let mut hs = Handshake::new(test_pmk(), ME, AP, nonce(0x55), Vec::new());
        assert!(hs.on_frame(&ap_msg1(nonce(0x11), 5)).is_ok());
        assert!(
            hs.on_frame(&ap_msg1(nonce(0x11), 5)).is_err(),
            "same counter accepted"
        );
        assert!(
            hs.on_frame(&ap_msg1(nonce(0x11), 4)).is_err(),
            "older counter accepted"
        );
        // A genuinely newer one is a legitimate restart.
        assert!(hs.on_frame(&ap_msg1(nonce(0x66), 6)).is_ok());
    }

    #[test_case]
    fn messages_out_of_order_or_out_of_role_are_refused() {
        let mut hs = Handshake::new(test_pmk(), ME, AP, nonce(0x77), Vec::new());
        let ptk = derive_ptk(&test_pmk(), &AP, &ME, &nonce(0x11), &nonce(0x77));

        // Message 3 with no message 1: there is no PTK, so there is nothing to check its MIC
        // with — it must not be treated as a MIC failure (which would read as a wrong
        // password) nor accepted.
        let err = hs
            .on_frame(&ap_msg3(&ptk, nonce(0x11), 2, &[0u8; 16]))
            .expect_err("message 3 without message 1 was accepted");
        assert!(err.contains("before"), "misdiagnosed: {err}");

        // A frame in the client's role. An AP does not send these, so something is
        // impersonating a client and answering it would be free work for an attacker.
        let msg2 = EapolKey {
            key_info: 2 | KEY_INFO_PAIRWISE | KEY_INFO_MIC,
            key_len: 0,
            replay_counter: 1,
            nonce: nonce(0x77),
            rsc: [0; 8],
            mic: [0; 16],
            key_data: Vec::new(),
        }
        .to_bytes();
        assert!(
            hs.on_frame(&msg2).is_err(),
            "a message 2 was accepted from the AP"
        );

        // A group rekey is not part of the four-way handshake (no Pairwise bit).
        let rekey = EapolKey {
            key_info: 2 | KEY_INFO_MIC | KEY_INFO_ACK | KEY_INFO_SECURE,
            key_len: 16,
            replay_counter: 9,
            nonce: [0; 32],
            rsc: [0; 8],
            mic: [0; 16],
            key_data: Vec::new(),
        };
        assert_eq!(EapolKey::parse(&rekey.to_bytes()).unwrap().message(), None);
    }

    #[test_case]
    fn message_three_must_carry_the_same_anonce_as_message_one() {
        // A different ANonce means two different PTKs, which would otherwise show up as
        // traffic that decrypts to nothing. The MIC has to be valid for the check to be
        // reachable at all, so this is a confused AP rather than an attacker — but a silent
        // wrong key is the worst of the outcomes.
        let mut hs = Handshake::new(test_pmk(), ME, AP, nonce(0x88), Vec::new());
        assert!(hs.on_frame(&ap_msg1(nonce(0x11), 1)).is_ok());
        let other = derive_ptk(&test_pmk(), &AP, &ME, &nonce(0x99), &nonce(0x88));
        let err = hs
            .on_frame(&ap_msg3(&other, nonce(0x99), 2, &[0u8; 16]))
            .expect_err("a changed ANonce was accepted");
        assert!(err.contains("ANonce"), "misdiagnosed: {err}");
    }

    #[test_case]
    fn a_malformed_eapol_frame_is_refused_not_salvaged() {
        // Like the beacon parser, this runs on bytes from an unauthenticated sender: the
        // handshake happens before any key is confirmed. Nothing may panic.
        let ptk = derive_ptk(&test_pmk(), &AP, &ME, &nonce(0x11), &nonce(0x22));
        let good = ap_msg3(&ptk, nonce(0x11), 2, &[0x5a; 16]);

        for n in 0..good.len() {
            let mut hs = Handshake::new(test_pmk(), ME, AP, nonce(0x22), Vec::new());
            let _ = hs.on_frame(&good[..n]);
            assert!(EapolKey::parse(&good[..n]).is_none() || n == good.len());
        }

        // A key-data length claiming more than the frame holds. Clamping it would hand the
        // unwrap a truncated buffer that might still pass its integrity check on a prefix.
        let mut lying = good.clone();
        lying[4 + 93] = 0xff;
        lying[4 + 94] = 0xff;
        assert!(
            EapolKey::parse(&lying).is_none(),
            "a lying key-data length was accepted"
        );

        // A body length shorter than the fixed fields, and a non-Key packet type.
        let mut short = good.clone();
        short[2] = 0;
        short[3] = 10;
        assert!(EapolKey::parse(&short).is_none());
        let mut wrong_type = good.clone();
        wrong_type[1] = 0; // EAP-Packet, not Key
        assert!(EapolKey::parse(&wrong_type).is_none());
        // And a WPA1 descriptor type, which this code does not implement.
        let mut wpa1 = good.clone();
        wpa1[4] = 254;
        assert!(EapolKey::parse(&wpa1).is_none());
    }

    #[test_case]
    fn a_tampered_group_key_is_refused_rather_than_installed() {
        // The unwrap's integrity check is the only thing between an attacker and a group key
        // of their choosing — and unlike the pairwise key, a bad group key means accepting
        // forged broadcast traffic.
        let anonce = nonce(0x11);
        let snonce = nonce(0x22);
        let ptk = derive_ptk(&test_pmk(), &AP, &ME, &anonce, &snonce);
        let mut hs = Handshake::new(test_pmk(), ME, AP, snonce, Vec::new());
        assert!(hs.on_frame(&ap_msg1(anonce, 1)).is_ok());

        let mut msg3 = ap_msg3(&ptk, anonce, 2, &[0x5a; 16]);
        // Flip a bit inside the wrapped key data, then re-MIC so the frame is otherwise
        // valid — which is precisely the attack the integrity check exists for.
        let data_at = 4 + EAPOL_KEY_BODY_LEN;
        msg3[data_at + 9] ^= 1;
        let mut k = EapolKey::parse(&msg3).unwrap();
        k.mic = eapol_mic(&ptk.kck, &k.to_bytes_for_mic());
        let err = hs
            .on_frame(&k.to_bytes())
            .expect_err("a tampered group key was installed");
        assert!(err.contains("integrity"), "misdiagnosed: {err}");
        assert!(hs.gtk.is_none());
    }

    #[test_case]
    fn the_gtk_is_found_among_other_key_data_and_never_past_its_end() {
        // Real key data carries the AP's RSN element and padding alongside the GTK KDE.
        let mut kd = alloc::vec![0x30, 0x14];
        kd.extend_from_slice(&[0u8; 20]); // an RSN element
        kd.extend_from_slice(&[0xdd, 0x16, 0x00, 0x0f, 0xac, 0x01, 0x02, 0x00]);
        kd.extend_from_slice(&[0x77; 16]); // the GTK
        kd.extend_from_slice(&[0xdd, 0x00]); // trailing padding
        let g = find_gtk(&kd).expect("the GTK KDE was not found among the others");
        assert_eq!(g.key, alloc::vec![0x77; 16]);
        assert_eq!(g.id, 2);

        // No GTK at all, and lengths that run off the end.
        assert!(find_gtk(&[]).is_none());
        assert!(find_gtk(&[0x30, 0x14, 0, 0]).is_none());
        assert!(find_gtk(&[0xdd, 0x60, 0x00, 0x0f, 0xac, 0x01]).is_none());
        // A GTK KDE with no key bytes is not a key.
        assert!(find_gtk(&[0xdd, 0x06, 0x00, 0x0f, 0xac, 0x01, 0x01, 0x00]).is_none());
        for n in 0..kd.len() {
            let _ = find_gtk(&kd[..n]); // must not panic
        }
    }
}
