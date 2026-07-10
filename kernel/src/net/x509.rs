//! X.509 certificate-chain validation for the TLS client — the trust layer
//! that turns "encrypted" into "authenticated". Pure Rust, no `ring`:
//! `x509-cert` for DER parsing, `p256`/`p384` for ECDSA, [`super::rsa`] for
//! RSA (PKCS#1 v1.5 + PSS) on `crypto-bigint`, against the embedded Mozilla
//! root store ([`super::ca_roots`]).
//!
//! What it checks, per RFC 5280 (the parts that matter for a client talking to
//! public servers):
//! * a chain from the leaf up to a trusted root (intermediates may arrive in
//!   any order; each link's signature is verified with the issuer's key),
//! * every certificate's validity window against the wall clock,
//! * CA basic-constraints on issuers (a leaf can't sign for others),
//! * the leaf's Subject Alternative Names against the requested hostname
//!   (wildcards included).
//!
//! Deliberately **out of scope** (documented, not silently skipped): CRL/OCSP
//! revocation, name constraints, policy constraints. A revoked-but-unexpired
//! cert still validates — same posture as many embedded stacks; noted so it is
//! a known limitation, not a surprise.

use crate::net::hashes::HashId;
use crate::net::rsa::RsaPublicKey;
use alloc::string::String;
use alloc::vec::Vec;
use x509_cert::der::{Decode, Encode};
use x509_cert::Certificate;

/// Algorithm/curve/extension OIDs as typed constants (const-oid has no
/// `to_string` in no_std, so we compare `ObjectIdentifier` values, not
/// strings). `new_unwrap` is const and panics only on a malformed literal.
mod oid {
    use x509_cert::der::asn1::ObjectIdentifier as Oid;
    pub const RSA_ENCRYPTION: Oid = Oid::new_unwrap("1.2.840.113549.1.1.1");
    pub const RSA_PSS: Oid = Oid::new_unwrap("1.2.840.113549.1.1.10");
    pub const SHA256_RSA: Oid = Oid::new_unwrap("1.2.840.113549.1.1.11");
    pub const SHA384_RSA: Oid = Oid::new_unwrap("1.2.840.113549.1.1.12");
    pub const SHA512_RSA: Oid = Oid::new_unwrap("1.2.840.113549.1.1.13");
    pub const EC_PUBLIC_KEY: Oid = Oid::new_unwrap("1.2.840.10045.2.1");
    pub const ECDSA_SHA256: Oid = Oid::new_unwrap("1.2.840.10045.4.3.2");
    pub const ECDSA_SHA384: Oid = Oid::new_unwrap("1.2.840.10045.4.3.3");
    pub const ECDSA_SHA512: Oid = Oid::new_unwrap("1.2.840.10045.4.3.4");
    pub const P256: Oid = Oid::new_unwrap("1.2.840.10045.3.1.7");
    pub const P384: Oid = Oid::new_unwrap("1.3.132.0.34");
    pub const BASIC_CONSTRAINTS: Oid = Oid::new_unwrap("2.5.29.19");
    pub const SUBJECT_ALT_NAME: Oid = Oid::new_unwrap("2.5.29.17");
}

/// ECDSA curve identity, resolved from the SPKI algorithm parameters.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Curve {
    P256,
    P384,
    Other,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Parse,
    EmptyChain,
    Expired,
    BadSignature,
    NoTrustedRoot,
    NotCa,
    HostMismatch,
    UnsupportedAlg,
}

/// Verify `chain` (DER-encoded, leaf first) reaches a trusted root, is valid at
/// `now_unix`, and the leaf is issued for `host`. Returns the leaf's SPKI DER
/// (needed to check the TLS `CertificateVerify` afterwards).
pub fn verify_chain(chain: &[&[u8]], host: &str, now_unix: u64) -> Result<Vec<u8>, Error> {
    if chain.is_empty() {
        return Err(Error::EmptyChain);
    }
    let certs: Vec<Certificate> = chain.iter().map(|d| Certificate::from_der(d).map_err(|_| Error::Parse)).collect::<Result<_, _>>()?;

    // Validity of every presented cert.
    for c in &certs {
        if !valid_at(c, now_unix) {
            return Err(Error::Expired);
        }
    }

    // Hostname must match the leaf's SANs.
    if !host_matches(&certs[0], host) {
        return Err(Error::HostMismatch);
    }

    // Build the path from the leaf upward. At each step the current cert must
    // be signed by an issuer whose subject == current.issuer; the issuer is
    // found first among the presented intermediates, then (terminal) among the
    // trusted roots. A root hit ends the walk successfully.
    let mut current = 0usize; // index into `certs` for the cert being verified
    let mut depth = 0usize;
    loop {
        if depth > 12 {
            return Err(Error::NoTrustedRoot); // pathological / looped chain
        }
        depth += 1;
        let cur = &certs[current];
        let issuer_dn = cur.tbs_certificate.issuer.to_der().map_err(|_| Error::Parse)?;

        // 1) A trusted root whose subject matches + verifies this cert → done.
        for root_der in super::ca_roots::roots() {
            let Ok(root) = Certificate::from_der(root_der) else { continue };
            if root.tbs_certificate.subject.to_der().ok().as_deref() != Some(&issuer_dn) {
                continue;
            }
            if !is_ca(&root) || !valid_at(&root, now_unix) {
                continue;
            }
            if cert_signed_by(cur, &root)? {
                return leaf_spki(&certs[0]);
            }
        }

        // 2) A presented intermediate that matches + verifies → advance.
        let mut next: Option<usize> = None;
        for (i, cand) in certs.iter().enumerate() {
            if i == current {
                continue;
            }
            if cand.tbs_certificate.subject.to_der().ok().as_deref() != Some(&issuer_dn) {
                continue;
            }
            if !is_ca(cand) {
                continue;
            }
            if cert_signed_by(cur, cand)? {
                next = Some(i);
                break;
            }
        }
        match next {
            Some(i) => current = i,
            None => return Err(Error::NoTrustedRoot),
        }
    }
}

/// True if `now` is within `cert`'s validity window.
fn valid_at(cert: &Certificate, now: u64) -> bool {
    let nb = cert.tbs_certificate.validity.not_before.to_unix_duration().as_secs();
    let na = cert.tbs_certificate.validity.not_after.to_unix_duration().as_secs();
    now >= nb && now <= na
}

/// True if `cert` asserts CA basic-constraints (or is a root without the
/// extension — legacy roots omit it; we only reach roots from the trust store).
fn is_ca(cert: &Certificate) -> bool {
    use x509_cert::ext::pkix::BasicConstraints;
    let Some(exts) = &cert.tbs_certificate.extensions else {
        // Self-issued (root) with no extensions: treat as CA.
        return cert.tbs_certificate.subject == cert.tbs_certificate.issuer;
    };
    for e in exts {
        if e.extn_id == oid::BASIC_CONSTRAINTS {
            if let Ok(bc) = BasicConstraints::from_der(e.extn_value.as_bytes()) {
                return bc.ca;
            }
        }
    }
    // No BasicConstraints: CA only if self-issued (root).
    cert.tbs_certificate.subject == cert.tbs_certificate.issuer
}

/// Verify `cert`'s signature using `issuer`'s public key.
fn cert_signed_by(cert: &Certificate, issuer: &Certificate) -> Result<bool, Error> {
    let tbs = cert.tbs_certificate.to_der().map_err(|_| Error::Parse)?;
    let sig = cert.signature.as_bytes().ok_or(Error::Parse)?;
    let spki = &issuer.tbs_certificate.subject_public_key_info;
    verify_sig(spki, &cert.signature_algorithm.oid, &tbs, sig)
}

/// Verify a signature `sig` over `msg` using the public key in `spki`, per the
/// signature-algorithm OID. Shared by cert-chain links and the TLS
/// CertificateVerify step (via [`verify_data`]).
fn verify_sig(
    spki: &x509_cert::spki::SubjectPublicKeyInfoOwned,
    sig_alg: &x509_cert::der::asn1::ObjectIdentifier,
    msg: &[u8],
    sig: &[u8],
) -> Result<bool, Error> {
    let key_alg = spki.algorithm.oid;
    let key = spki.subject_public_key.as_bytes().ok_or(Error::Parse)?;
    let sa = *sig_alg;
    if sa == oid::SHA256_RSA || sa == oid::SHA384_RSA || sa == oid::SHA512_RSA {
        if key_alg != oid::RSA_ENCRYPTION {
            return Err(Error::UnsupportedAlg);
        }
        let hash = if sa == oid::SHA256_RSA {
            HashId::Sha256
        } else if sa == oid::SHA384_RSA {
            HashId::Sha384
        } else {
            HashId::Sha512
        };
        let pk = RsaPublicKey::from_pkcs1_der(key).ok_or(Error::Parse)?;
        Ok(pk.verify_pkcs1v15(hash, msg, sig))
    } else if sa == oid::RSA_PSS {
        if key_alg != oid::RSA_ENCRYPTION {
            return Err(Error::UnsupportedAlg);
        }
        // Cert PSS params default to SHA-256 in the common case; TLS 1.3
        // CertificateVerify carries the hash in the scheme (see verify_data).
        let pk = RsaPublicKey::from_pkcs1_der(key).ok_or(Error::Parse)?;
        Ok(pk.verify_pss(HashId::Sha256, msg, sig))
    } else if sa == oid::ECDSA_SHA256 || sa == oid::ECDSA_SHA384 || sa == oid::ECDSA_SHA512 {
        if key_alg != oid::EC_PUBLIC_KEY {
            return Err(Error::UnsupportedAlg);
        }
        let hash = if sa == oid::ECDSA_SHA256 {
            HashId::Sha256
        } else if sa == oid::ECDSA_SHA384 {
            HashId::Sha384
        } else {
            HashId::Sha512
        };
        ecdsa_verify(spki_curve(spki), key, hash, msg, sig)
    } else {
        Err(Error::UnsupportedAlg)
    }
}

/// Resolve the named curve from an EC SPKI's algorithm parameters.
fn spki_curve(spki: &x509_cert::spki::SubjectPublicKeyInfoOwned) -> Curve {
    match spki.algorithm.parameters.as_ref().and_then(|p| p.decode_as::<x509_cert::der::asn1::ObjectIdentifier>().ok()) {
        Some(o) if o == oid::P256 => Curve::P256,
        Some(o) if o == oid::P384 => Curve::P384,
        _ => Curve::Other,
    }
}

/// ECDSA verify dispatched by curve. `sig` is the DER `SEQUENCE { r, s }`.
fn ecdsa_verify(curve: Curve, key: &[u8], hash: HashId, msg: &[u8], sig: &[u8]) -> Result<bool, Error> {
    use crate::net::hashes;
    let digest = hashes::digest(hash, msg);
    match curve {
        Curve::P256 => {
            use p256::ecdsa::signature::hazmat::PrehashVerifier;
            use p256::ecdsa::{Signature, VerifyingKey};
            let vk = VerifyingKey::from_sec1_bytes(key).map_err(|_| Error::Parse)?;
            let s = Signature::from_der(sig).map_err(|_| Error::Parse)?;
            Ok(vk.verify_prehash(&digest, &s).is_ok())
        }
        Curve::P384 => {
            use p384::ecdsa::signature::hazmat::PrehashVerifier;
            use p384::ecdsa::{Signature, VerifyingKey};
            let vk = VerifyingKey::from_sec1_bytes(key).map_err(|_| Error::Parse)?;
            let s = Signature::from_der(sig).map_err(|_| Error::Parse)?;
            Ok(vk.verify_prehash(&digest, &s).is_ok())
        }
        Curve::Other => Err(Error::UnsupportedAlg),
    }
}

/// Extract the leaf's SPKI DER (for the later CertificateVerify check).
fn leaf_spki(leaf: &Certificate) -> Result<Vec<u8>, Error> {
    leaf.tbs_certificate.subject_public_key_info.to_der().map_err(|_| Error::Parse)
}

/// Verify a TLS 1.3 `CertificateVerify`: the leaf key signed `msg`
/// (64 spaces + context + transcript hash) with `scheme`. `spki_der` is the
/// leaf SPKI returned by [`verify_chain`].
pub fn verify_data(spki_der: &[u8], scheme: u16, msg: &[u8], sig: &[u8]) -> Result<bool, Error> {
    use x509_cert::spki::SubjectPublicKeyInfoOwned;
    let spki = SubjectPublicKeyInfoOwned::from_der(spki_der).map_err(|_| Error::Parse)?;
    let key = spki.subject_public_key.as_bytes().ok_or(Error::Parse)?;
    let key_alg = spki.algorithm.oid;
    // TLS SignatureScheme codes (RFC 8446 §4.2.3).
    match scheme {
        0x0804 | 0x0805 | 0x0806 => {
            // rsa_pss_rsae_sha{256,384,512}
            if key_alg != oid::RSA_ENCRYPTION {
                return Err(Error::UnsupportedAlg);
            }
            let hash = match scheme {
                0x0804 => HashId::Sha256,
                0x0805 => HashId::Sha384,
                _ => HashId::Sha512,
            };
            let pk = RsaPublicKey::from_pkcs1_der(key).ok_or(Error::Parse)?;
            Ok(pk.verify_pss(hash, msg, sig))
        }
        0x0401 | 0x0501 | 0x0601 => {
            // rsa_pkcs1_sha{256,384,512} (allowed for cert sigs, seen in CV too)
            if key_alg != oid::RSA_ENCRYPTION {
                return Err(Error::UnsupportedAlg);
            }
            let hash = match scheme {
                0x0401 => HashId::Sha256,
                0x0501 => HashId::Sha384,
                _ => HashId::Sha512,
            };
            let pk = RsaPublicKey::from_pkcs1_der(key).ok_or(Error::Parse)?;
            Ok(pk.verify_pkcs1v15(hash, msg, sig))
        }
        0x0403 => ecdsa_verify(Curve::P256, key, HashId::Sha256, msg, sig), // ecdsa_secp256r1_sha256
        0x0503 => ecdsa_verify(Curve::P384, key, HashId::Sha384, msg, sig), // ecdsa_secp384r1_sha384
        _ => Err(Error::UnsupportedAlg),
    }
}

/// Match `host` against the leaf's dNSName SANs (case-insensitive, one level
/// of leading `*.` wildcard per RFC 6125).
fn host_matches(leaf: &Certificate, host: &str) -> bool {
    use x509_cert::ext::pkix::name::GeneralName;
    use x509_cert::ext::pkix::SubjectAltName;
    let Some(exts) = &leaf.tbs_certificate.extensions else { return false };
    for e in exts {
        if e.extn_id != oid::SUBJECT_ALT_NAME {
            continue;
        }
        let Ok(san) = SubjectAltName::from_der(e.extn_value.as_bytes()) else { return false };
        for gn in san.0.iter() {
            if let GeneralName::DnsName(dns) = gn {
                if dns_matches(dns.as_str(), host) {
                    return true;
                }
            }
        }
    }
    false
}

/// One dNSName pattern vs a hostname. `*.a.com` matches `x.a.com` but not
/// `a.com` or `y.x.a.com` (single label, leftmost only).
pub fn dns_matches(pattern: &str, host: &str) -> bool {
    let (p, h) = (pattern.to_ascii_lowercase(), host.to_ascii_lowercase());
    if let Some(suffix) = p.strip_prefix("*.") {
        // host must have exactly one extra leftmost label over the suffix.
        match h.split_once('.') {
            Some((_, rest)) => rest == suffix && !rest.is_empty(),
            None => false,
        }
    } else {
        p == h
    }
}

/// Verify a chain and return the leaf SPKI, translating to a display string on
/// error. The wall clock comes from the RTC; if unset (0), validity can't be
/// judged and verification refuses rather than trusting blindly.
pub fn verify(chain: &[&[u8]], host: &str) -> Result<Vec<u8>, String> {
    let now = crate::clock::now_unix();
    if now < 1_600_000_000 {
        return Err(String::from("TLS: wall clock unset (set /datetime) — cannot check certificate validity"));
    }
    verify_chain(chain, host, now as u64).map_err(|e| alloc::format!("TLS certificate verification failed: {e:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn dns_wildcard_matching() {
        assert!(dns_matches("example.com", "example.com"));
        assert!(dns_matches("Example.COM", "example.com"));
        assert!(dns_matches("*.example.com", "api.example.com"));
        assert!(!dns_matches("*.example.com", "example.com")); // no label
        assert!(!dns_matches("*.example.com", "a.b.example.com")); // two labels
        assert!(!dns_matches("example.com", "evil.com"));
    }

    // Real embedded chain (leaf+intermediate from a public host) + the root it
    // must reach in the store. Fixture + expected outcomes in ca_testvec.rs
    // (regen: tools/gen_ca_testvec.sh). Guards the whole path-validation +
    // signature stack against a real certificate.
    include!("ca_testvec.rs");

    #[test_case]
    fn real_chain_verifies_and_fails_closed() {
        let chain: Vec<&[u8]> = TEST_CHAIN.iter().map(|c| *c).collect();
        // At a time inside the cert's validity window: verifies + host matches.
        assert!(verify_chain(&chain, TEST_HOST, TEST_NOW).is_ok(), "valid chain must verify");
        // Wrong host → HostMismatch.
        assert_eq!(verify_chain(&chain, "wrong.example.org", TEST_NOW), Err(Error::HostMismatch));
        // Before validity → Expired.
        assert_eq!(verify_chain(&chain, TEST_HOST, 1_000_000_000), Err(Error::Expired));
        // Tamper the leaf signature → the chain no longer reaches a root.
        let mut bad_leaf = TEST_CHAIN[0].to_vec();
        let n = bad_leaf.len();
        bad_leaf[n - 1] ^= 0x01;
        let mut bad: Vec<&[u8]> = alloc::vec![bad_leaf.as_slice()];
        bad.extend(TEST_CHAIN[1..].iter().copied());
        assert!(matches!(verify_chain(&bad, TEST_HOST, TEST_NOW), Err(Error::NoTrustedRoot) | Err(Error::Parse)));
    }
}
