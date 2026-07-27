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
//! - **AES-128 decryption and RFC 3394 key unwrap**, against the RFC's own vector.
//! - **The PMK, the PTK and the MIC**, against values computed independently for the
//!   fixtures below.
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

// --- AES-128, decryption only ---------------------------------------------
//
// Needed for exactly one thing: unwrapping the group key out of the third handshake
// message. Decryption only, because nothing here ever wraps a key.

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
                m[c * 4] = xtime(col[0], 14) ^ xtime(col[1], 11) ^ xtime(col[2], 13) ^ xtime(col[3], 9);
                m[c * 4 + 1] = xtime(col[0], 9) ^ xtime(col[1], 14) ^ xtime(col[2], 11) ^ xtime(col[3], 13);
                m[c * 4 + 2] = xtime(col[0], 13) ^ xtime(col[1], 9) ^ xtime(col[2], 14) ^ xtime(col[3], 11);
                m[c * 4 + 3] = xtime(col[0], 11) ^ xtime(col[1], 13) ^ xtime(col[2], 9) ^ xtime(col[3], 14);
            }
            *block = m;
        } else {
            *block = t;
        }
    }
}

/// The RFC 3394 default initial value. A wrapped key that does not decrypt to this has
/// been tampered with, or the KEK is wrong — and those are the same thing to a client.
pub const KEY_WRAP_IV: [u8; 8] = [0xa6; 8];

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
        let pmk = pmk_from_passphrase("password", b"IEEE");
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
        let pmk = pmk_from_passphrase("secret", b"net");
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
        let pmk = pmk_from_passphrase("secret", b"net");
        let a = [1u8, 0, 0, 0, 0, 1];
        let s = [2u8, 0, 0, 0, 0, 2];
        let n1 = [0x11u8; 32];
        let n2 = [0x22u8; 32];
        let base = derive_ptk(&pmk, &a, &s, &n1, &n2);
        let other_pmk = pmk_from_passphrase("secret2", b"net");
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
        assert_eq!(hex(&eapol_mic(&kck, &frame)), "3bd7333f98ba70268a0a983084ef93f9");
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
    fn a_tampered_or_wrongly_keyed_wrap_is_refused() {
        // The integrity check is the only thing stopping a client from installing a group
        // key an attacker chose, so it has to reject rather than return plausible bytes.
        let mut kek = [0u8; 16];
        kek.copy_from_slice(&unhex("000102030405060708090a0b0c0d0e0f"));
        let good = unhex("1fa68b0a8112b447aef34bd8fb5a7b829d3e862371d2cfe5");

        let mut bad = good.clone();
        bad[5] ^= 1;
        assert!(aes_key_unwrap(&kek, &bad).is_none(), "tampered wrap accepted");

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
}
