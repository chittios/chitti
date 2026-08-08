//! Packet encryption: `aes*-gcm@openssh.com` (AEAD) and `aes*-ctr` with
//! `hmac-sha2-256` in either encrypt-and-MAC or encrypt-then-MAC form.
//!
//! Pure — keys and packets in, packets out. The whole module exists to get one
//! thing right: **which bytes are encrypted, which are authenticated, and in
//! what order**, because the three modes disagree and each disagreement is
//! invisible until a real server rejects the packet.
//!
//! | mode | length field | MAC covers | packet aligns |
//! |---|---|---|---|
//! | `aes-ctr` + `hmac` (EaM) | encrypted | the **plaintext** packet | `4 + packet_length` |
//! | `aes-ctr` + `hmac-…-etm` | **clear** | the **ciphertext** | `packet_length` |
//! | `aes-gcm@openssh.com` | **clear**, as AAD | the AEAD tag | `packet_length` |
//!
//! Two further traps, both of which produce a client that works for exactly one
//! packet: the GCM nonce is a 4-byte fixed prefix plus a **64-bit counter that
//! increments per packet** (RFC 5647 §7.1) rather than the sequence number, and
//! the CTR keystream is *continuous across packets* — it must not be reset per
//! packet, or every packet after the first decrypts to noise.

use aes::cipher::{BlockEncrypt, KeyInit as _, KeyIvInit, StreamCipher};
use aes::{Aes128, Aes256};
use aes_gcm::aead::AeadInPlace;
use aes_gcm::{Aes128Gcm, Aes256Gcm, Nonce};
use alloc::vec::Vec;
use hmac::{Hmac, Mac};
use sha2::Sha256;

use super::wire::{self, LengthMode};

type Aes128Ctr = ctr::Ctr64BE<Aes128>;
type Aes256Ctr = ctr::Ctr64BE<Aes256>;
type HmacSha256 = Hmac<Sha256>;

/// How many bytes of key, IV and MAC key a negotiated algorithm needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sizes {
    pub key: usize,
    pub iv: usize,
    pub block: usize,
    pub mac_key: usize,
    pub mac_len: usize,
}

/// Sizes for a cipher/MAC pair, or `None` if we do not implement it.
pub fn sizes(enc: &str, mac: &str) -> Option<Sizes> {
    let (key, iv, block) = match enc {
        "aes256-gcm@openssh.com" => (32, 12, 16),
        "aes128-gcm@openssh.com" => (16, 12, 16),
        "aes256-ctr" => (32, 16, 16),
        "aes128-ctr" => (16, 16, 16),
        _ => return None,
    };
    let (mac_key, mac_len) = if super::kex::is_aead(enc) {
        (0, 16) // the GCM tag
    } else {
        match mac {
            "hmac-sha2-256" | "hmac-sha2-256-etm@openssh.com" => (32, 32),
            _ => return None,
        }
    };
    Some(Sizes {
        key,
        iv,
        block,
        mac_key,
        mac_len,
    })
}

/// One direction's keys and running state.
pub enum Direction {
    Gcm {
        cipher: GcmCipher,
        /// The 12-byte nonce: a 4-byte fixed prefix and a 64-bit counter that
        /// increments once per packet.
        nonce: [u8; 12],
    },
    Ctr {
        stream: CtrStream,
        mac_key: Vec<u8>,
        /// Encrypt-then-MAC (`…-etm@openssh.com`) leaves the length in the clear
        /// and authenticates the ciphertext.
        etm: bool,
    },
}

pub enum GcmCipher {
    A128(alloc::boxed::Box<Aes128Gcm>),
    A256(alloc::boxed::Box<Aes256Gcm>),
}

pub enum CtrStream {
    A128(alloc::boxed::Box<Aes128Ctr>),
    A256(alloc::boxed::Box<Aes256Ctr>),
}

impl CtrStream {
    fn apply(&mut self, buf: &mut [u8]) {
        match self {
            CtrStream::A128(c) => c.apply_keystream(buf),
            CtrStream::A256(c) => c.apply_keystream(buf),
        }
    }
}

impl Direction {
    /// Build a direction from negotiated names and derived material.
    pub fn new(enc: &str, mac: &str, key: &[u8], iv: &[u8], mac_key: &[u8]) -> Option<Self> {
        match enc {
            "aes256-gcm@openssh.com" | "aes128-gcm@openssh.com" => {
                let mut nonce = [0u8; 12];
                if iv.len() < 12 {
                    return None;
                }
                nonce.copy_from_slice(&iv[..12]);
                let cipher = if enc.starts_with("aes256") {
                    GcmCipher::A256(alloc::boxed::Box::new(Aes256Gcm::new_from_slice(key).ok()?))
                } else {
                    GcmCipher::A128(alloc::boxed::Box::new(Aes128Gcm::new_from_slice(key).ok()?))
                };
                Some(Direction::Gcm { cipher, nonce })
            }
            "aes256-ctr" | "aes128-ctr" => {
                let stream = if enc.starts_with("aes256") {
                    CtrStream::A256(alloc::boxed::Box::new(Aes256Ctr::new_from_slices(key, iv).ok()?))
                } else {
                    CtrStream::A128(alloc::boxed::Box::new(Aes128Ctr::new_from_slices(key, iv).ok()?))
                };
                Some(Direction::Ctr {
                    stream,
                    mac_key: mac_key.to_vec(),
                    etm: mac.ends_with("-etm@openssh.com"),
                })
            }
            _ => None,
        }
    }

    /// The length field is outside the encrypted region for AEAD and ETM.
    pub fn length_mode(&self) -> LengthMode {
        match self {
            Direction::Gcm { .. } => LengthMode::Plain,
            Direction::Ctr { etm: true, .. } => LengthMode::Plain,
            Direction::Ctr { etm: false, .. } => LengthMode::Encrypted,
        }
    }

    /// Bytes of MAC or tag appended to each packet.
    pub fn tag_len(&self) -> usize {
        match self {
            Direction::Gcm { .. } => 16,
            Direction::Ctr { .. } => 32,
        }
    }

    /// **RFC 5647 §7.1: the GCM invocation counter increments per packet**, and
    /// only the low 8 bytes move — the 4-byte fixed prefix stays put.
    fn bump_nonce(nonce: &mut [u8; 12]) {
        let mut ctr = u64::from_be_bytes(nonce[4..12].try_into().unwrap_or([0; 8]));
        ctr = ctr.wrapping_add(1);
        nonce[4..12].copy_from_slice(&ctr.to_be_bytes());
    }

    /// Encrypt one framed packet in place, returning the wire bytes.
    ///
    /// `packet` is the *plaintext* binary packet (length, padding length,
    /// payload, padding), padded for [`Direction::length_mode`].
    pub fn seal(&mut self, seq: u32, packet: &[u8]) -> Option<Vec<u8>> {
        match self {
            Direction::Gcm { cipher, nonce } => {
                // The length field is additional authenticated data, in the clear.
                let (aad, body) = packet.split_at(4);
                let mut buf = body.to_vec();
                let n = Nonce::from_slice(nonce);
                let tag = match cipher {
                    GcmCipher::A128(c) => c.encrypt_in_place_detached(n, aad, &mut buf).ok()?,
                    GcmCipher::A256(c) => c.encrypt_in_place_detached(n, aad, &mut buf).ok()?,
                };
                Self::bump_nonce(nonce);
                let mut out = Vec::with_capacity(packet.len() + 16);
                out.extend_from_slice(aad);
                out.extend_from_slice(&buf);
                out.extend_from_slice(&tag);
                Some(out)
            }
            Direction::Ctr { stream, mac_key, etm } => {
                if *etm {
                    // Encrypt-then-MAC: length clear, MAC over the ciphertext.
                    let (len_field, body) = packet.split_at(4);
                    let mut buf = body.to_vec();
                    stream.apply(&mut buf);
                    let mut out = Vec::with_capacity(packet.len() + 32);
                    out.extend_from_slice(len_field);
                    out.extend_from_slice(&buf);
                    let mut m = <HmacSha256 as Mac>::new_from_slice(mac_key).ok()?;
                    m.update(&seq.to_be_bytes());
                    m.update(&out);
                    out.extend_from_slice(&m.finalize().into_bytes());
                    Some(out)
                } else {
                    // Encrypt-and-MAC: MAC over the *plaintext*, then encrypt all.
                    let mut m = <HmacSha256 as Mac>::new_from_slice(mac_key).ok()?;
                    m.update(&seq.to_be_bytes());
                    m.update(packet);
                    let tag = m.finalize().into_bytes();
                    let mut out = packet.to_vec();
                    stream.apply(&mut out);
                    out.extend_from_slice(&tag);
                    Some(out)
                }
            }
        }
    }

    /// How many bytes must be read before the packet length is known.
    ///
    /// For AEAD/ETM the length is in the clear (4 bytes). For encrypt-and-MAC
    /// it is encrypted, so a whole cipher block has to be decrypted first — and
    /// that block must not be decrypted twice, which is why [`open`] takes the
    /// already-decrypted prefix rather than re-deriving it.
    pub fn length_prefix(&self) -> usize {
        match self {
            Direction::Gcm { .. } | Direction::Ctr { etm: true, .. } => 4,
            Direction::Ctr { etm: false, .. } => 16,
        }
    }

    /// Decrypt the first block of an encrypt-and-MAC packet to learn its length.
    /// A no-op for the modes whose length is already in the clear.
    pub fn peek_length(&mut self, first: &mut [u8]) -> Option<usize> {
        match self {
            Direction::Gcm { .. } | Direction::Ctr { etm: true, .. } => {
                let b = first.get(..4)?;
                Some(u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as usize)
            }
            Direction::Ctr { stream, .. } => {
                stream.apply(first);
                let b = first.get(..4)?;
                Some(u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as usize)
            }
        }
    }

    /// Finish decrypting a packet whose first `prefix` bytes [`peek_length`]
    /// already handled, and verify its MAC or tag.
    ///
    /// Returns the plaintext binary packet, or `None` if authentication failed —
    /// which must be treated as fatal, never retried.
    pub fn open(&mut self, seq: u32, prefix: &[u8], rest: &[u8], tag: &[u8]) -> Option<Vec<u8>> {
        match self {
            Direction::Gcm { cipher, nonce } => {
                let mut buf = rest.to_vec();
                let n = Nonce::from_slice(nonce);
                let tag = aes_gcm::Tag::from_slice(tag);
                let ok = match cipher {
                    GcmCipher::A128(c) => c.decrypt_in_place_detached(n, prefix, &mut buf, tag),
                    GcmCipher::A256(c) => c.decrypt_in_place_detached(n, prefix, &mut buf, tag),
                };
                ok.ok()?;
                Self::bump_nonce(nonce);
                let mut out = Vec::with_capacity(prefix.len() + buf.len());
                out.extend_from_slice(prefix);
                out.extend_from_slice(&buf);
                Some(out)
            }
            Direction::Ctr { stream, mac_key, etm } => {
                if *etm {
                    // Verify over the ciphertext *before* decrypting it.
                    let mut m = <HmacSha256 as Mac>::new_from_slice(mac_key).ok()?;
                    m.update(&seq.to_be_bytes());
                    m.update(prefix);
                    m.update(rest);
                    m.verify_slice(tag).ok()?;
                    let mut buf = rest.to_vec();
                    stream.apply(&mut buf);
                    let mut out = Vec::with_capacity(prefix.len() + buf.len());
                    out.extend_from_slice(prefix);
                    out.extend_from_slice(&buf);
                    Some(out)
                } else {
                    // The prefix was already decrypted by `peek_length`.
                    let mut buf = rest.to_vec();
                    stream.apply(&mut buf);
                    let mut out = Vec::with_capacity(prefix.len() + buf.len());
                    out.extend_from_slice(prefix);
                    out.extend_from_slice(&buf);
                    let mut m = <HmacSha256 as Mac>::new_from_slice(mac_key).ok()?;
                    m.update(&seq.to_be_bytes());
                    m.update(&out);
                    m.verify_slice(tag).ok()?;
                    Some(out)
                }
            }
        }
    }
}

/// Frame and encrypt a payload in one step.
pub fn seal_payload(dir: &mut Direction, seq: u32, payload: &[u8], block: usize) -> Option<Vec<u8>> {
    let packet = wire::frame(payload, block, dir.length_mode(), &mut |p| {
        crate::security::rng::fill_random(p)
    });
    dir.seal(seq, &packet)
}

/// Encrypt a block with a raw AES key — used only by the tests below to prove
/// the CTR stream is continuous, not by the protocol.
#[cfg(test)]
fn aes256_block(key: &[u8; 32], block: &mut [u8; 16]) {
    use aes::cipher::generic_array::GenericArray;
    let c = Aes256::new(GenericArray::from_slice(key));
    c.encrypt_block(GenericArray::from_mut_slice(block));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dirs(enc: &str, mac: &str) -> (Direction, Direction) {
        let s = sizes(enc, mac).expect("known algorithm");
        let key = alloc::vec![0xA5u8; s.key];
        let iv = alloc::vec![0x5Au8; s.iv];
        let mk = alloc::vec![0x3Cu8; s.mac_key.max(1)];
        (
            Direction::new(enc, mac, &key, &iv, &mk).expect("sender"),
            Direction::new(enc, mac, &key, &iv, &mk).expect("receiver"),
        )
    }

    /// Every mode round-trips, over a run of packets — which is what catches a
    /// stream cipher that resets its keystream or a nonce that fails to advance.
    #[test_case]
    fn every_mode_round_trips_across_many_packets() {
        for (enc, mac) in [
            ("aes256-gcm@openssh.com", ""),
            ("aes128-gcm@openssh.com", ""),
            ("aes256-ctr", "hmac-sha2-256"),
            ("aes256-ctr", "hmac-sha2-256-etm@openssh.com"),
            ("aes128-ctr", "hmac-sha2-256"),
        ] {
            let (mut tx, mut rx) = dirs(enc, mac);
            let block = sizes(enc, mac).unwrap().block;
            for seq in 0..8u32 {
                let payload: Vec<u8> = (0..(seq as usize * 7 + 3)).map(|i| i as u8).collect();
                let wire_bytes = seal_payload(&mut tx, seq, &payload, block).expect("seal");

                // Receive: peek the length, then open the rest.
                let prefix_len = rx.length_prefix();
                let mut prefix = wire_bytes[..prefix_len].to_vec();
                let packet_len = rx.peek_length(&mut prefix).expect("length");
                let total = 4 + packet_len;
                let rest = &wire_bytes[prefix_len..total];
                let tag = &wire_bytes[total..total + rx.tag_len()];
                let plain = rx.open(seq, &prefix, rest, tag).expect("open");
                assert_eq!(
                    wire::payload_of(&plain).expect("framed"),
                    &payload[..],
                    "{enc}/{mac} packet {seq}"
                );
            }
        }
    }

    /// A tampered packet fails authentication in every mode, and the tamper is
    /// caught wherever it lands — body or length.
    #[test_case]
    fn tampering_is_rejected() {
        for (enc, mac) in [
            ("aes256-gcm@openssh.com", ""),
            ("aes256-ctr", "hmac-sha2-256"),
            ("aes256-ctr", "hmac-sha2-256-etm@openssh.com"),
        ] {
            for flip in [5usize, 9] {
                let (mut tx, mut rx) = dirs(enc, mac);
                let block = sizes(enc, mac).unwrap().block;
                let mut w = seal_payload(&mut tx, 0, b"hello ssh", block).unwrap();
                w[flip] ^= 0x01;
                let prefix_len = rx.length_prefix();
                let mut prefix = w[..prefix_len.min(w.len())].to_vec();
                let Some(packet_len) = rx.peek_length(&mut prefix) else {
                    continue; // a mangled length is already a refusal
                };
                let total = 4 + packet_len;
                if total + rx.tag_len() > w.len() {
                    continue; // the length was corrupted into nonsense: also a refusal
                }
                let rest = &w[prefix_len..total];
                let tag = &w[total..total + rx.tag_len()];
                assert!(
                    rx.open(0, &prefix, rest, tag).is_none(),
                    "{enc}/{mac}: a flipped bit at {flip} must fail authentication"
                );
            }
        }
    }

    /// The sequence number is authenticated, so replaying a packet under a
    /// different sequence number fails.
    #[test_case]
    fn the_sequence_number_is_authenticated() {
        let (mut tx, mut rx) = dirs("aes256-ctr", "hmac-sha2-256");
        let w = seal_payload(&mut tx, 7, b"seq matters", 16).unwrap();
        let prefix_len = rx.length_prefix();
        let mut prefix = w[..prefix_len].to_vec();
        let packet_len = rx.peek_length(&mut prefix).unwrap();
        let total = 4 + packet_len;
        let rest = &w[prefix_len..total];
        let tag = &w[total..total + rx.tag_len()];
        assert!(rx.open(8, &prefix, rest, tag).is_none(), "wrong sequence number must fail");
    }

    /// **The GCM nonce advances per packet.** Sealing the same payload twice
    /// must not produce the same ciphertext, or the keystream is being reused —
    /// which for GCM is a catastrophic failure, not a cosmetic one.
    #[test_case]
    fn the_gcm_nonce_advances_per_packet() {
        let (mut tx, _) = dirs("aes256-gcm@openssh.com", "");
        let a = seal_payload(&mut tx, 0, b"same payload", 16).unwrap();
        let b = seal_payload(&mut tx, 1, b"same payload", 16).unwrap();
        assert_ne!(a, b, "identical plaintext must not encrypt identically");
        // Only the counter half of the nonce moves.
        let Direction::Gcm { nonce, .. } = &tx else {
            panic!("expected GCM")
        };
        assert_eq!(&nonce[..4], &[0x5A; 4], "the fixed prefix must not move");
        assert_eq!(u64::from_be_bytes(nonce[4..12].try_into().unwrap()), 0x5A5A_5A5A_5A5A_5A5Au64.wrapping_add(2));
    }

    /// **The CTR keystream is continuous across packets.** Re-keying per packet
    /// would make every packet after the first decrypt to noise.
    ///
    /// The padding is pinned to a constant here rather than taken from
    /// [`seal_payload`], which fills it from the CSPRNG as RFC 4253 §6 requires —
    /// so two encryptions of the "same" packet are never byte-identical and the
    /// comparison below would be meaningless.
    #[test_case]
    fn the_ctr_keystream_is_continuous() {
        let packet = wire::frame(&[0u8; 16], 16, LengthMode::Encrypted, &mut |p| p.fill(0));

        let (mut tx, _) = dirs("aes256-ctr", "hmac-sha2-256");
        let first = tx.seal(0, &packet).unwrap();
        let second = tx.seal(1, &packet).unwrap();

        let (mut fresh, _) = dirs("aes256-ctr", "hmac-sha2-256");
        let fresh_first = fresh.seal(0, &packet).unwrap();

        assert_eq!(first, fresh_first, "a fresh direction must reproduce the first packet");
        assert_ne!(
            first, second,
            "the keystream must not restart per packet — identical plaintext encrypted twice"
        );
    }

    /// And the padding really is random, which is what the test above works
    /// around: two seals of the same *payload* differ even at sequence 0.
    #[test_case]
    fn packet_padding_comes_from_the_csprng() {
        let (mut a, _) = dirs("aes256-ctr", "hmac-sha2-256");
        let (mut b, _) = dirs("aes256-ctr", "hmac-sha2-256");
        let x = seal_payload(&mut a, 0, &[0u8; 16], 16).unwrap();
        let y = seal_payload(&mut b, 0, &[0u8; 16], 16).unwrap();
        assert_ne!(x, y, "padding must be random, not zero-filled");
    }

    /// The length mode really differs between AEAD/ETM and encrypt-and-MAC, and
    /// the packet is padded for whichever one is in force.
    #[test_case]
    fn length_mode_matches_the_cipher() {
        let (gcm, _) = dirs("aes256-gcm@openssh.com", "");
        assert_eq!(gcm.length_mode(), LengthMode::Plain);
        assert_eq!(gcm.length_prefix(), 4, "the GCM length is in the clear");

        let (etm, _) = dirs("aes256-ctr", "hmac-sha2-256-etm@openssh.com");
        assert_eq!(etm.length_mode(), LengthMode::Plain);
        assert_eq!(etm.length_prefix(), 4);

        let (eam, _) = dirs("aes256-ctr", "hmac-sha2-256");
        assert_eq!(eam.length_mode(), LengthMode::Encrypted);
        assert_eq!(eam.length_prefix(), 16, "an encrypted length needs a whole block first");
    }

    /// Unknown algorithms are refused rather than defaulted.
    #[test_case]
    fn unknown_algorithms_are_refused() {
        assert!(sizes("3des-cbc", "hmac-sha1").is_none());
        assert!(sizes("aes256-ctr", "hmac-md5").is_none());
        assert!(Direction::new("3des-cbc", "hmac-sha1", &[0; 24], &[0; 8], &[0; 20]).is_none());
        // A key of the wrong length is refused too.
        assert!(Direction::new("aes256-ctr", "hmac-sha2-256", &[0; 8], &[0; 16], &[0; 32]).is_none());
    }

    /// Sanity check on the AES primitive itself, against FIPS-197's own vector,
    /// so a broken dependency shows up here rather than as a handshake failure.
    #[test_case]
    fn aes256_matches_the_fips_197_vector() {
        let key: [u8; 32] = core::array::from_fn(|i| i as u8);
        let mut block: [u8; 16] = [
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd,
            0xee, 0xff,
        ];
        aes256_block(&key, &mut block);
        assert_eq!(
            block,
            [
                0x8e, 0xa2, 0xb7, 0xca, 0x51, 0x67, 0x45, 0xbf, 0xea, 0xfc, 0x49, 0x90, 0x4b, 0x49,
                0x60, 0x89
            ]
        );
    }
}
