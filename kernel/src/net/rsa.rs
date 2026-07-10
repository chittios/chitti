//! RSA signature **verification** for the TLS certificate validator — pure
//! Rust, no `ring`. Public-key operations only (`s^e mod n`); no key
//! generation, no private key, no RNG. Big-integer modexp runs on
//! `crypto-bigint` (already pulled by `p256`), sized to a fixed `U4096`, which
//! covers every RSA modulus on the public web (2048/3072/4096-bit).
//!
//! Two schemes, both used by real cert chains + TLS 1.3:
//! * **PKCS#1 v1.5** (`EMSA-PKCS1-v1_5`) — most CA/intermediate cert
//!   signatures.
//! * **PSS** (`EMSA-PSS`, MGF1) — mandatory for RSA in the TLS 1.3
//!   `CertificateVerify`, and some modern cert signatures.
//!
//! Verification is not required to be constant-time (only public data is
//! involved), so the straightforward algorithm is used.

use crate::net::hashes::{self, HashId};
use alloc::vec::Vec;
use crypto_bigint::modular::runtime_mod::{DynResidue, DynResidueParams};
use crypto_bigint::{Encoding, U4096};

/// A parsed RSA public key: modulus `n` and exponent `e`, both big-endian.
pub struct RsaPublicKey {
    n: U4096,
    n_len: usize, // byte length of the modulus (the EM/signature length)
    e: U4096,
}

impl RsaPublicKey {
    /// Parse a DER `RSAPublicKey` (`SEQUENCE { modulus INTEGER, publicExponent
    /// INTEGER }`) — the bytes inside an SPKI whose algorithm is rsaEncryption.
    pub fn from_pkcs1_der(der: &[u8]) -> Option<RsaPublicKey> {
        let mut p = Der::new(der);
        let mut seq = p.take_seq()?;
        let n = seq.take_uint()?;
        let e = seq.take_uint()?;
        if n.is_empty() || n.len() > 512 || e.is_empty() || e.len() > 8 {
            return None;
        }
        Some(RsaPublicKey { n: be_to_u4096(n)?, n_len: n.len(), e: be_to_u4096(e)? })
    }

    /// `s^e mod n` → the `n_len`-byte big-endian encoded message (`EM`), or
    /// `None` if `s >= n` / malformed.
    fn public_op(&self, sig: &[u8]) -> Option<Vec<u8>> {
        if sig.len() != self.n_len {
            return None;
        }
        let s = be_to_u4096(sig)?;
        // Reject s >= n (RFC 8017 §5.2.2 step 1).
        if s >= self.n {
            return None;
        }
        // n is odd (RSA modulus) → valid Montgomery params.
        let params = DynResidueParams::new(&self.n);
        let m = DynResidue::new(&s, params).pow(&self.e).retrieve();
        let full = m.to_be_bytes(); // 512 bytes, big-endian, left-zero-padded
        Some(full[full.len() - self.n_len..].to_vec())
    }

    /// EMSA-PKCS1-v1_5 verify: `msg`'s `hash` digest, `sig` the signature.
    pub fn verify_pkcs1v15(&self, hash: HashId, msg: &[u8], sig: &[u8]) -> bool {
        let Some(em) = self.public_op(sig) else { return false };
        let digest = hashes::digest(hash, msg);
        // Expected EM = 0x00 0x01 PS(0xFF..) 0x00 T, T = DigestInfo prefix||hash.
        let prefix = hashes::digestinfo_prefix(hash);
        let t_len = prefix.len() + digest.len();
        if em.len() < t_len + 11 {
            return false;
        }
        let ps_len = em.len() - t_len - 3;
        let mut ok = em[0] == 0x00 && em[1] == 0x01 && em[2 + ps_len] == 0x00;
        for &b in &em[2..2 + ps_len] {
            ok &= b == 0xff;
        }
        ok &= em[3 + ps_len..3 + ps_len + prefix.len()] == *prefix;
        ok &= em[3 + ps_len + prefix.len()..] == digest[..];
        ok
    }

    /// EMSA-PSS verify (MGF1 with the same hash; salt length = hash length,
    /// the near-universal choice for TLS/certs).
    pub fn verify_pss(&self, hash: HashId, msg: &[u8], sig: &[u8]) -> bool {
        let Some(em) = self.public_op(sig) else { return false };
        let h_len = hashes::len(hash);
        let em_bits = self.n_len * 8 - 1; // emBits = modBits - 1
        let em_len = em_bits.div_ceil(8);
        // public_op returns n_len bytes; PSS EM is emLen = ceil(emBits/8).
        // When modBits-1 is a multiple of 8 they differ by one leading byte.
        let em = if em.len() > em_len { em[em.len() - em_len..].to_vec() } else { em };
        if em.len() != em_len || em_len < h_len + 2 {
            return false;
        }
        if *em.last().unwrap() != 0xbc {
            return false;
        }
        let masked_db_len = em_len - h_len - 1;
        // EM = maskedDB || H || 0xbc. `h` is exactly the hLen-byte H — NOT
        // including the trailing 0xbc (that byte was checked above); slicing it
        // in makes both the MGF1 seed and the final compare off-by-one.
        let (masked_db, rest) = em.split_at(masked_db_len);
        let h = &rest[..h_len];
        // Top (8*emLen - emBits) bits of the leftmost masked_db byte are 0.
        let top_bits = 8 * em_len - em_bits;
        if masked_db[0] & (0xffu8 << (8 - top_bits)) != 0 {
            return false;
        }
        let db_mask = mgf1(hash, h, masked_db_len);
        let mut db: Vec<u8> = masked_db.iter().zip(&db_mask).map(|(a, b)| a ^ b).collect();
        db[0] &= 0xffu8 >> top_bits;
        // DB = PS(0x00..) || 0x01 || salt.  PS is emLen - hLen - sLen - 2 zeros.
        let s_len = h_len; // assumed salt length
        if db.len() < s_len + 1 {
            return false;
        }
        let ps_end = db.len() - s_len - 1;
        if db[..ps_end].iter().any(|&b| b != 0) || db[ps_end] != 0x01 {
            return false;
        }
        let salt = &db[ps_end + 1..];
        // H' = Hash(0x00*8 || mHash || salt).
        let m_hash = hashes::digest(hash, msg);
        let mut m_prime = Vec::with_capacity(8 + h_len + salt.len());
        m_prime.extend_from_slice(&[0u8; 8]);
        m_prime.extend_from_slice(&m_hash);
        m_prime.extend_from_slice(salt);
        hashes::digest(hash, &m_prime) == h
    }
}

/// MGF1 mask generation (RFC 8017 B.2.1) with the given hash.
fn mgf1(hash: HashId, seed: &[u8], len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    let mut counter: u32 = 0;
    while out.len() < len {
        let mut input = Vec::with_capacity(seed.len() + 4);
        input.extend_from_slice(seed);
        input.extend_from_slice(&counter.to_be_bytes());
        out.extend_from_slice(&hashes::digest(hash, &input));
        counter += 1;
    }
    out.truncate(len);
    out
}

/// Big-endian bytes → `U4096` (left-zero-padded). `None` if longer than 512 B.
fn be_to_u4096(be: &[u8]) -> Option<U4096> {
    if be.len() > 512 {
        return None;
    }
    let mut buf = [0u8; 512];
    buf[512 - be.len()..].copy_from_slice(be);
    Some(U4096::from_be_bytes(buf))
}

/// A tiny DER reader — just enough for `RSAPublicKey` (SEQUENCE of two
/// INTEGERs). Definite-length only; rejects anything unexpected.
struct Der<'a> {
    b: &'a [u8],
}
impl<'a> Der<'a> {
    fn new(b: &'a [u8]) -> Der<'a> {
        Der { b }
    }
    /// Read a TLV with expected tag; return its content bytes, advancing.
    fn take(&mut self, tag: u8) -> Option<&'a [u8]> {
        if self.b.first()? != &tag {
            return None;
        }
        let (len, hdr) = der_len(&self.b[1..])?;
        let start = 1 + hdr;
        let content = self.b.get(start..start + len)?;
        self.b = &self.b[start + len..];
        Some(content)
    }
    fn take_seq(&mut self) -> Option<Der<'a>> {
        Some(Der::new(self.take(0x30)?))
    }
    /// An INTEGER's magnitude bytes with any single leading 0x00 sign byte
    /// stripped (positive integers > 0x7f carry it).
    fn take_uint(&mut self) -> Option<&'a [u8]> {
        let mut v = self.take(0x02)?;
        while v.len() > 1 && v[0] == 0x00 {
            v = &v[1..];
        }
        Some(v)
    }
}

/// Decode a DER definite length; returns `(length, header_bytes_consumed)`.
fn der_len(b: &[u8]) -> Option<(usize, usize)> {
    let first = *b.first()?;
    if first < 0x80 {
        return Some((first as usize, 1));
    }
    let n = (first & 0x7f) as usize;
    if n == 0 || n > 4 || b.len() < 1 + n {
        return None;
    }
    let mut len = 0usize;
    for &byte in &b[1..1 + n] {
        len = (len << 8) | byte as usize;
    }
    Some((len, 1 + n))
}

#[cfg(test)]
mod tests {
    use super::*;

    // RFC-style known-answer: a 2048-bit key + a PKCS#1 v1.5 SHA-256 signature
    // over "hello world", generated with openssl and embedded (verify path is
    // deterministic, so a fixed vector is a real regression gate). See
    // tools/gen_rsa_testvec.sh for regeneration.
    include!("rsa_testvec.rs");

    #[test_case]
    fn pkcs1v15_known_vector() {
        let key = RsaPublicKey::from_pkcs1_der(RSA_PUB_PKCS1).expect("parse key");
        assert!(key.verify_pkcs1v15(HashId::Sha256, b"hello world", RSA_SIG_PKCS1), "valid signature must verify");
        assert!(!key.verify_pkcs1v15(HashId::Sha256, b"hello worlx", RSA_SIG_PKCS1), "tampered message must fail");
        let mut bad = RSA_SIG_PKCS1.to_vec();
        bad[100] ^= 1;
        assert!(!key.verify_pkcs1v15(HashId::Sha256, b"hello world", &bad), "tampered signature must fail");
    }

    #[test_case]
    fn pss_known_vector() {
        let key = RsaPublicKey::from_pkcs1_der(RSA_PUB_PKCS1).expect("parse key");
        assert!(key.verify_pss(HashId::Sha256, b"hello world", RSA_SIG_PSS), "valid PSS signature must verify");
        assert!(!key.verify_pss(HashId::Sha256, b"tampered", RSA_SIG_PSS), "wrong message must fail");
    }

    #[test_case]
    fn rejects_malformed_key() {
        assert!(RsaPublicKey::from_pkcs1_der(b"\x30\x00").is_none());
        assert!(RsaPublicKey::from_pkcs1_der(b"garbage").is_none());
    }
}
