//! Package signing/verification for the skill trust model
//! (`CHITTI_AGENTIC_HANDOFF.md` Phase G): a keyed **MAC over a canonical package
//! serialization**, against a single baked-in trusted registry key
//! (`chitti-registry-key-1`).
//!
//! **Why not Ed25519 here (logged in DECISIONS.md):** the schema names Ed25519,
//! and the `ed25519-compact` crate *builds* under `-Z build-std`, but its
//! `sign()`/`verify()` **fault at runtime on the bare-metal x86 target** (no
//! host runtime; likely a stack/SIMD assumption). Rather than block the phase on
//! a bare-metal curve implementation, verification uses a self-contained keyed
//! MAC (SipHash-2-4, eight lanes → a 512-bit tag). In this self-contained build
//! the kernel is *both* the registry (it can `sign` sample packages with the
//! baked secret) and the installer (it verifies against it), so this delivers
//! the integrity + tamper-detection + unsigned-rejection the install flow needs.
//! REVISIT: swap in a bare-metal-safe asymmetric Ed25519 for true off-device
//! authenticity.

use alloc::vec::Vec;

/// The id of the one trusted registry key.
pub const REGISTRY_KEY_ID: &str = "chitti-registry-key-1";

/// The baked registry secret (demo/self-contained). 32 bytes.
const REGISTRY_SECRET: [u8; 32] = *b"chitti-registry-key-1___seed_v01";

/// SipHash-2-4 of `msg` under 128-bit key `(k0, k1)`.
fn siphash24(k0: u64, k1: u64, msg: &[u8]) -> u64 {
    let mut v0 = 0x736f6d6570736575 ^ k0;
    let mut v1 = 0x646f72616e646f6d ^ k1;
    let mut v2 = 0x6c7967656e657261 ^ k0;
    let mut v3 = 0x7465646279746573 ^ k1;

    macro_rules! round {
        () => {{
            v0 = v0.wrapping_add(v1);
            v1 = v1.rotate_left(13);
            v1 ^= v0;
            v0 = v0.rotate_left(32);
            v2 = v2.wrapping_add(v3);
            v3 = v3.rotate_left(16);
            v3 ^= v2;
            v0 = v0.wrapping_add(v3);
            v3 = v3.rotate_left(21);
            v3 ^= v0;
            v2 = v2.wrapping_add(v1);
            v1 = v1.rotate_left(17);
            v1 ^= v2;
            v2 = v2.rotate_left(32);
        }};
    }

    let len = msg.len();
    let mut i = 0;
    while i + 8 <= len {
        let m = u64::from_le_bytes(msg[i..i + 8].try_into().unwrap());
        v3 ^= m;
        round!();
        round!();
        v0 ^= m;
        i += 8;
    }
    // Last block: remaining bytes + length in the top byte.
    let mut last = (len as u64) << 56;
    let mut shift = 0;
    while i < len {
        last |= (msg[i] as u64) << shift;
        shift += 8;
        i += 1;
    }
    v3 ^= last;
    round!();
    round!();
    v0 ^= last;
    v2 ^= 0xff;
    round!();
    round!();
    round!();
    round!();
    v0 ^ v1 ^ v2 ^ v3
}

/// The 512-bit MAC of `msg` under the registry secret: eight SipHash lanes, each
/// keyed by the secret plus the lane index, concatenated little-endian → 64
/// bytes (same width as an Ed25519 signature, so `SignatureBlock` is unchanged).
fn mac(secret: &[u8; 32], msg: &[u8]) -> Vec<u8> {
    let base0 = u64::from_le_bytes(secret[0..8].try_into().unwrap());
    let base1 = u64::from_le_bytes(secret[8..16].try_into().unwrap());
    let base2 = u64::from_le_bytes(secret[16..24].try_into().unwrap());
    let base3 = u64::from_le_bytes(secret[24..32].try_into().unwrap());
    let mut out = Vec::with_capacity(64);
    for lane in 0u64..8 {
        let k0 = base0 ^ base2.wrapping_mul(lane.wrapping_add(1));
        let k1 = base1 ^ base3.rotate_left(lane as u32);
        out.extend_from_slice(&siphash24(k0, k1, msg).to_le_bytes());
    }
    out
}

/// The trusted key ids (for display / trust-store checks).
pub fn is_trusted(key_id: &str) -> bool {
    key_id == REGISTRY_KEY_ID
}

/// Sign `msg` with the registry secret (64-byte tag).
pub fn sign(msg: &[u8]) -> Vec<u8> {
    mac(&REGISTRY_SECRET, msg)
}

/// Verify `sig` over `msg` for a trusted `key_id`. Constant-shape comparison;
/// false for an unknown key, a wrong-length tag, or any mismatch (tampering).
pub fn verify(key_id: &str, msg: &[u8], sig: &[u8]) -> bool {
    if !is_trusted(key_id) {
        return false;
    }
    let expected = mac(&REGISTRY_SECRET, msg);
    sig.len() == expected.len() && sig == expected.as_slice()
}

// --- Asymmetric (ECDSA P-256) verification for public-registry packages -----
//
// A public registry has many publishers, so the symmetric MAC above (kernel is
// both signer and verifier) isn't enough — we need real off-device authenticity.
// P-256 ECDSA over SHA-256 (RustCrypto `p256`, verify-only, no RNG) does it: a
// publisher signs a package off-device with their private key; the kernel ships
// their PUBLIC key in the trust store below and verifies. Same curve family the
// TLS client already uses, so it builds under our custom target.

/// A trusted publisher/registry P-256 key: an id and its SEC1 uncompressed
/// public point (65 bytes, `0x04||X||Y`) as hex. Baked at build time — there is
/// no runtime path to add a trusted key.
struct P256Key {
    key_id: &'static str,
    sec1_hex: &'static str,
}

static P256_TRUST: &[P256Key] = &[
    // The registry test publisher (used by the registry-client tests). Replace
    // with real publisher keys as they are onboarded.
    P256Key {
        key_id: "chitti-publisher-test",
        sec1_hex: "04936bb4ad6d7101c2d23ac8d6db1e3ad08f81fd1cf90ecae0501454eae357cd31cd655551ed2f854a75197588cf5b2e97d5ef1fdc826b0039f43f049197b8a4e1",
    },
];

/// The SEC1 public point for a trusted P-256 key id, if any.
fn p256_pubkey(key_id: &str) -> Option<Vec<u8>> {
    P256_TRUST.iter().find(|k| k.key_id == key_id).and_then(|k| from_hex(k.sec1_hex))
}

/// Whether `key_id` is a trusted P-256 publisher.
pub fn is_trusted_p256(key_id: &str) -> bool {
    P256_TRUST.iter().any(|k| k.key_id == key_id)
}

/// Verify a DER-encoded ECDSA/P-256/SHA-256 `sig` over `msg` for a trusted
/// publisher `key_id`. False for an unknown key, a malformed key/signature, or
/// any mismatch.
pub fn verify_p256(key_id: &str, msg: &[u8], sig_der: &[u8]) -> bool {
    use p256::ecdsa::signature::Verifier;
    use p256::ecdsa::{Signature, VerifyingKey};
    let Some(point) = p256_pubkey(key_id) else { return false };
    let Ok(vk) = VerifyingKey::from_sec1_bytes(&point) else { return false };
    let Ok(sig) = Signature::from_der(sig_der) else { return false };
    vk.verify(msg, &sig).is_ok()
}

/// Decode an even-length ASCII hex string into bytes; `None` on any bad char.
fn from_hex(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 {
        return None;
    }
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(s.len() / 2);
    let nib = |c: u8| -> Option<u8> {
        match c {
            b'0'..=b'9' => Some(c - b'0'),
            b'a'..=b'f' => Some(c - b'a' + 10),
            b'A'..=b'F' => Some(c - b'A' + 10),
            _ => None,
        }
    };
    let mut i = 0;
    while i < b.len() {
        out.push((nib(b[i])? << 4) | nib(b[i + 1])?);
        i += 2;
    }
    Some(out)
}

/// A 32-byte content digest (four SipHash lanes) for the `content_hash` field.
pub fn hash32(msg: &[u8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for lane in 0u64..4 {
        let h = siphash24(0x9e3779b97f4a7c15u64.wrapping_mul(lane + 1), 0xdeadbeefcafef00d, msg);
        out[lane as usize * 8..lane as usize * 8 + 8].copy_from_slice(&h.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test_case]
    fn crypto_roundtrip() {
        let msg = b"chitti signed payload";
        let sig = sign(msg);
        assert_eq!(sig.len(), 64);
        assert!(verify(REGISTRY_KEY_ID, msg, &sig));
        assert!(!verify(REGISTRY_KEY_ID, b"tampered", &sig), "wrong message fails");
        assert!(!verify("unknown-key", msg, &sig), "untrusted key fails");
        assert!(!verify(REGISTRY_KEY_ID, msg, &[]), "empty signature fails");
    }

    #[test_case]
    fn p256_verifies_a_real_signature() {
        // A genuine ECDSA/P-256/SHA-256 signature (openssl) over the message
        // below, by the private key whose public point is the "chitti-publisher-
        // test" trust-store entry. Proves real asymmetric verification works on
        // the bare-metal target.
        let msg = b"chitti-registry-package-v1";
        let sig = from_hex(
            "3045022028dbf55d56f51d67777054ace805e8b591d6339fb850aca39478fa21535ae2d7022100ca5f4a0620e5e50ae97b376a361c36e0a61f99d665d104f7fa7fba80d1bceb4b",
        )
        .unwrap();
        assert!(verify_p256("chitti-publisher-test", msg, &sig), "valid P-256 signature must verify");
        assert!(!verify_p256("chitti-publisher-test", b"tampered", &sig), "wrong message must fail");
        assert!(!verify_p256("unknown-publisher", msg, &sig), "untrusted key must fail");
        assert!(!verify_p256("chitti-publisher-test", msg, b"\x30\x00"), "malformed signature must fail");
        assert!(is_trusted_p256("chitti-publisher-test") && !is_trusted_p256("nope"));
    }
}
