//! Host keys and signatures — parsing the blobs SSH puts on the wire, checking
//! a signature over the exchange hash, and the `known_hosts` trust decision.
//!
//! Pure. This is the module that decides **whether you are talking to the server
//! you meant to**, so every failure here is closed: an unparseable blob, an
//! unknown algorithm and a bad signature are all refusals, never warnings.
//!
//! Three encoding details carry the risk:
//!
//! * **A key blob names its own algorithm, and the inner name must match the
//!   outer one.** A server that negotiated `ssh-ed25519` and sends an
//!   `ssh-rsa` blob is either broken or attacking; accepting it would let a
//!   peer pick the algorithm *after* seeing our offer.
//! * **The ECDSA signature is a `string` containing two `mpint`s**, not a raw
//!   64-byte pair. Reading it as raw bytes works only when neither `r` nor `s`
//!   happens to need a sign-extension byte, which is most of the time — so it
//!   fails on roughly one connection in four.
//! * **`known_hosts` is matched by (host, key type)**, and a *different* key for
//!   a host we know is a hard failure, while a *new* host is merely unknown.
//!   Collapsing the two would either teach users to accept key changes or make
//!   first contact impossible.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use sha2::{Digest, Sha256};

use super::wire::{Reader, Writer};

/// A parsed SSH public key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PublicKey {
    Ed25519([u8; 32]),
    /// An uncompressed SEC1 point (65 bytes, `0x04 || X || Y`).
    EcdsaP256(Vec<u8>),
}

impl PublicKey {
    /// The SSH algorithm name this key is used under.
    pub fn algorithm(&self) -> &'static str {
        match self {
            PublicKey::Ed25519(_) => "ssh-ed25519",
            PublicKey::EcdsaP256(_) => "ecdsa-sha2-nistp256",
        }
    }

    /// Parse a public-key blob (`string algorithm || …`).
    pub fn parse(blob: &[u8]) -> Option<Self> {
        let mut r = Reader::new(blob);
        match r.utf8()? {
            "ssh-ed25519" => {
                let k = r.string()?;
                let arr: [u8; 32] = k.try_into().ok()?;
                Some(PublicKey::Ed25519(arr))
            }
            "ecdsa-sha2-nistp256" => {
                // The curve name is repeated inside the blob and must agree.
                if r.utf8()? != "nistp256" {
                    return None;
                }
                let q = r.string()?;
                // Only the uncompressed form appears in SSH; a compressed point
                // is refused rather than guessed at.
                if q.len() != 65 || q[0] != 0x04 {
                    return None;
                }
                Some(PublicKey::EcdsaP256(q.to_vec()))
            }
            _ => None,
        }
    }

    /// Re-encode to the wire blob. Round-trips [`PublicKey::parse`].
    pub fn encode(&self) -> Vec<u8> {
        let mut w = Writer::new();
        match self {
            PublicKey::Ed25519(k) => {
                w.put_str("ssh-ed25519");
                w.put_string(k);
            }
            PublicKey::EcdsaP256(q) => {
                w.put_str("ecdsa-sha2-nistp256");
                w.put_str("nistp256");
                w.put_string(q);
            }
        }
        w.into_vec()
    }

    /// The `SHA256:…` fingerprint OpenSSH prints, for a human to compare.
    pub fn fingerprint(&self) -> String {
        let d = Sha256::digest(self.encode());
        let mut s = String::from("SHA256:");
        // OpenSSH prints unpadded base64.
        s.push_str(crate::net::ws::base64_encode(&d).trim_end_matches('='));
        s
    }

    /// Verify `signature` over `message` (the exchange hash, for a host key).
    ///
    /// `signature` is the wire blob: `string algorithm || string signature`.
    pub fn verify(&self, message: &[u8], signature: &[u8]) -> bool {
        let mut r = Reader::new(signature);
        let Some(alg) = r.utf8() else { return false };
        // The signature names its algorithm too, and it must be the one this key
        // is for — otherwise a peer chooses the algorithm after seeing our offer.
        if alg != self.algorithm() {
            return false;
        }
        let Some(sig) = r.string() else { return false };
        match self {
            PublicKey::Ed25519(k) => {
                let Ok(vk) = ed25519_dalek::VerifyingKey::from_bytes(k) else {
                    return false;
                };
                let Ok(sig): Result<[u8; 64], _> = sig.try_into() else {
                    return false;
                };
                let sig = ed25519_dalek::Signature::from_bytes(&sig);
                // `verify_strict` rejects small-order and non-canonical points,
                // which the permissive `verify` accepts.
                vk.verify_strict(message, &sig).is_ok()
            }
            PublicKey::EcdsaP256(q) => {
                // **Two mpints inside a string**, not a raw r||s pair.
                let mut sr = Reader::new(sig);
                let (Some(r_bytes), Some(s_bytes)) = (sr.mpint(), sr.mpint()) else {
                    return false;
                };
                if !sr.is_empty() {
                    return false;
                }
                let (Some(r32), Some(s32)) = (left_pad32(r_bytes), left_pad32(s_bytes)) else {
                    return false;
                };
                let Ok(vk) = p256::ecdsa::VerifyingKey::from_sec1_bytes(q) else {
                    return false;
                };
                let mut raw = [0u8; 64];
                raw[..32].copy_from_slice(&r32);
                raw[32..].copy_from_slice(&s32);
                let Ok(sig) = p256::ecdsa::Signature::from_slice(&raw) else {
                    return false;
                };
                use p256::ecdsa::signature::Verifier;
                vk.verify(message, &sig).is_ok()
            }
        }
    }
}

/// Left-pad a big-endian magnitude to exactly 32 bytes, or `None` if it is
/// longer than the curve order allows.
fn left_pad32(v: &[u8]) -> Option<[u8; 32]> {
    if v.len() > 32 {
        return None;
    }
    let mut out = [0u8; 32];
    out[32 - v.len()..].copy_from_slice(v);
    Some(out)
}

// --- known_hosts -----------------------------------------------------------

/// What a `known_hosts` lookup concluded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Trust {
    /// This exact key is on file for this host.
    Known,
    /// We have no key of this type for this host — first contact.
    Unknown,
    /// We have a *different* key of this type. Refuse loudly: either the server
    /// was rebuilt or someone is in the middle, and the client cannot tell which.
    Changed { known: String },
}

/// One `known_hosts` line: `host[,host2] algorithm base64-key [comment]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KnownHost {
    pub hosts: Vec<String>,
    pub algorithm: String,
    pub key_b64: String,
}

/// Parse a `known_hosts` file, skipping blanks, comments and unparseable lines.
///
/// A malformed line is skipped rather than failing the file: one bad entry must
/// not lock the user out of every other host they know.
pub fn parse_known_hosts(text: &str) -> Vec<KnownHost> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let (Some(hosts), Some(alg), Some(key)) = (parts.next(), parts.next(), parts.next()) else {
            continue;
        };
        // `@cert-authority` / `@revoked` markers are not supported; skip rather
        // than treat a CA line as an ordinary key.
        if hosts.starts_with('@') {
            continue;
        }
        out.push(KnownHost {
            hosts: hosts.split(',').map(|h| h.to_string()).collect(),
            algorithm: alg.to_string(),
            key_b64: key.to_string(),
        });
    }
    out
}

/// How a host is written in `known_hosts`: bare for port 22, `[host]:port`
/// otherwise — the form OpenSSH uses, so the two files stay interchangeable.
pub fn host_pattern(host: &str, port: u16) -> String {
    if port == 22 {
        host.to_string()
    } else {
        alloc::format!("[{host}]:{port}")
    }
}

/// Decide whether `key` is the key we already trust for `host`.
pub fn check(entries: &[KnownHost], host: &str, port: u16, key: &PublicKey) -> Trust {
    let pattern = host_pattern(host, port);
    let alg = key.algorithm();
    let ours = crate::net::ws::base64_encode(&key.encode());
    let mut seen_other = None;
    for e in entries {
        if e.algorithm != alg || !e.hosts.iter().any(|h| h == &pattern) {
            continue;
        }
        if e.key_b64 == ours {
            return Trust::Known;
        }
        seen_other = Some(e.key_b64.clone());
    }
    match seen_other {
        Some(known) => Trust::Changed { known },
        None => Trust::Unknown,
    }
}

/// The line to append when a user accepts a new host key.
pub fn known_hosts_line(host: &str, port: u16, key: &PublicKey) -> String {
    alloc::format!(
        "{} {} {}\n",
        host_pattern(host, port),
        key.algorithm(),
        crate::net::ws::base64_encode(&key.encode())
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ed_key() -> PublicKey {
        PublicKey::Ed25519([7u8; 32])
    }

    /// Key blobs round-trip, and the inner algorithm name is authoritative.
    #[test_case]
    fn key_blobs_round_trip() {
        let k = ed_key();
        assert_eq!(PublicKey::parse(&k.encode()).unwrap(), k);

        let q = {
            let mut v = alloc::vec![0u8; 65];
            v[0] = 0x04;
            v
        };
        let e = PublicKey::EcdsaP256(q);
        assert_eq!(PublicKey::parse(&e.encode()).unwrap(), e);
    }

    /// Malformed and unsupported blobs are refused, never guessed at.
    #[test_case]
    fn key_blob_parsing_fails_closed() {
        assert!(PublicKey::parse(&[]).is_none());
        // Unknown algorithm.
        let mut w = Writer::new();
        w.put_str("ssh-dss");
        w.put_string(&[0u8; 32]);
        assert!(PublicKey::parse(&w.into_vec()).is_none());
        // ed25519 with the wrong key length.
        let mut w = Writer::new();
        w.put_str("ssh-ed25519");
        w.put_string(&[0u8; 31]);
        assert!(PublicKey::parse(&w.into_vec()).is_none());
        // ECDSA whose inner curve name disagrees with the algorithm.
        let mut w = Writer::new();
        w.put_str("ecdsa-sha2-nistp256");
        w.put_str("nistp384");
        w.put_string(&[4u8; 65]);
        assert!(PublicKey::parse(&w.into_vec()).is_none());
        // ECDSA with a compressed point.
        let mut w = Writer::new();
        w.put_str("ecdsa-sha2-nistp256");
        w.put_str("nistp256");
        w.put_string(&[2u8; 33]);
        assert!(PublicKey::parse(&w.into_vec()).is_none());
    }

    /// A signature naming a different algorithm than the key is refused — the
    /// peer does not get to choose the algorithm after seeing our offer.
    #[test_case]
    fn a_signature_must_name_the_keys_own_algorithm() {
        let k = ed_key();
        let mut w = Writer::new();
        w.put_str("ecdsa-sha2-nistp256");
        w.put_string(&[0u8; 64]);
        assert!(!k.verify(b"msg", &w.into_vec()));
        // And a truncated signature blob is refused rather than padded.
        assert!(!k.verify(b"msg", &[]));
        assert!(!k.verify(b"msg", &[0, 0, 0, 99]));
    }

    /// A real ed25519 signature verifies, and any tamper fails.
    ///
    /// The key and signature are produced here rather than hard-coded, so this
    /// tests our *blob framing* against a known-good signer rather than testing
    /// a constant against itself.
    #[test_case]
    fn a_real_ed25519_signature_verifies() {
        let sk = ed25519_dalek::SigningKey::from_bytes(&[42u8; 32]);
        let vk = sk.verifying_key();
        let key = PublicKey::Ed25519(vk.to_bytes());
        let msg = b"exchange hash stand-in";

        use ed25519_dalek::Signer;
        let sig = sk.sign(msg);
        let mut w = Writer::new();
        w.put_str("ssh-ed25519");
        w.put_string(&sig.to_bytes());
        let blob = w.into_vec();
        assert!(key.verify(msg, &blob), "a valid signature must verify");
        assert!(!key.verify(b"different message", &blob), "wrong message must fail");

        // A flipped bit anywhere in the signature must fail.
        let mut bad = blob.clone();
        let n = bad.len();
        bad[n - 1] ^= 0x01;
        assert!(!key.verify(msg, &bad));
    }

    /// A real ECDSA P-256 signature verifies **through the two-mpint framing**,
    /// including the case where a scalar needs a sign-extension byte — the one
    /// that a raw-64-byte reader gets wrong.
    #[test_case]
    fn a_real_ecdsa_signature_verifies_through_mpint_framing() {
        use p256::ecdsa::signature::Signer;
        // Several keys, so at least one signature has a high-bit scalar.
        let mut high_bit_seen = false;
        for seed in 1u8..12 {
            let sk = p256::ecdsa::SigningKey::from_bytes(&[seed; 32].into()).expect("valid scalar");
            let vk = sk.verifying_key();
            let key = PublicKey::EcdsaP256(vk.to_encoded_point(false).as_bytes().to_vec());
            let msg = b"exchange hash stand-in";
            let sig: p256::ecdsa::Signature = sk.sign(msg);
            let (r, s) = (sig.r().to_bytes(), sig.s().to_bytes());
            if r[0] & 0x80 != 0 || s[0] & 0x80 != 0 {
                high_bit_seen = true;
            }
            let mut inner = Writer::new();
            inner.put_mpint(&r);
            inner.put_mpint(&s);
            let mut w = Writer::new();
            w.put_str("ecdsa-sha2-nistp256");
            w.put_string(inner.as_slice());
            let blob = w.into_vec();
            assert!(key.verify(msg, &blob), "seed {seed}: valid signature must verify");
            assert!(!key.verify(b"other", &blob), "seed {seed}: wrong message must fail");
        }
        assert!(
            high_bit_seen,
            "no high-bit scalar appeared, so the mpint sign byte was never exercised"
        );
    }

    /// known_hosts: known, unknown and *changed* are three different answers.
    #[test_case]
    fn known_hosts_distinguishes_unknown_from_changed() {
        let key = ed_key();
        let other = PublicKey::Ed25519([9u8; 32]);
        let file = known_hosts_line("example.com", 22, &key);
        let entries = parse_known_hosts(&file);

        assert_eq!(check(&entries, "example.com", 22, &key), Trust::Known);
        assert_eq!(check(&entries, "other.com", 22, &key), Trust::Unknown);
        // A different key for a host we know is not "unknown" — it is a change.
        match check(&entries, "example.com", 22, &other) {
            Trust::Changed { .. } => {}
            t => panic!("a changed host key must be reported as such, got {t:?}"),
        }
        // A different *port* is a different host entry.
        assert_eq!(check(&entries, "example.com", 2222, &key), Trust::Unknown);
    }

    /// Non-default ports use OpenSSH's `[host]:port` form so the files stay
    /// interchangeable.
    #[test_case]
    fn host_patterns_match_openssh() {
        assert_eq!(host_pattern("example.com", 22), "example.com");
        assert_eq!(host_pattern("example.com", 2222), "[example.com]:2222");
    }

    /// Parsing tolerates comments, blank lines, multi-host entries and trailing
    /// comments, and skips markers it does not implement.
    #[test_case]
    fn known_hosts_parsing_is_tolerant_but_not_credulous() {
        let key = ed_key();
        let b64 = crate::net::ws::base64_encode(&key.encode());
        let text = alloc::format!(
            "# a comment\n\
             \n\
             github.com,140.82.121.4 ssh-ed25519 {b64} some-comment\n\
             @cert-authority *.example.com ssh-ed25519 {b64}\n\
             garbage-line\n"
        );
        let e = parse_known_hosts(&text);
        assert_eq!(e.len(), 1, "only the one ordinary entry is usable");
        assert_eq!(e[0].hosts, alloc::vec!["github.com", "140.82.121.4"]);
        // Either name matches.
        assert_eq!(check(&e, "github.com", 22, &key), Trust::Known);
        assert_eq!(check(&e, "140.82.121.4", 22, &key), Trust::Known);
    }

    /// Fingerprints are the unpadded `SHA256:` form OpenSSH prints, so a user
    /// can compare them against a published one by eye.
    #[test_case]
    fn fingerprints_match_the_openssh_shape() {
        let f = ed_key().fingerprint();
        assert!(f.starts_with("SHA256:"), "{f}");
        assert!(!f.ends_with('='), "OpenSSH prints unpadded base64: {f}");
        // 32 bytes of SHA-256 is 43 unpadded base64 characters.
        assert_eq!(f.len(), "SHA256:".len() + 43, "{f}");
        assert_ne!(f, PublicKey::Ed25519([8u8; 32]).fingerprint());
    }
}
