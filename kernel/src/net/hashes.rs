//! Hash dispatch for the TLS certificate validator: SHA-256/384/512 over a
//! byte slice, plus the DER `DigestInfo` prefixes PKCS#1 v1.5 prepends. Thin
//! wrapper over the `sha2` crate so `net::rsa` / `net::x509` don't each repeat
//! the digest-family match.

use alloc::vec::Vec;
use sha2::{Digest, Sha256, Sha384, Sha512};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HashId {
    Sha256,
    Sha384,
    Sha512,
}

/// Digest of `msg` under `id`.
pub fn digest(id: HashId, msg: &[u8]) -> Vec<u8> {
    match id {
        HashId::Sha256 => Sha256::digest(msg).to_vec(),
        HashId::Sha384 => Sha384::digest(msg).to_vec(),
        HashId::Sha512 => Sha512::digest(msg).to_vec(),
    }
}

/// Output length in bytes.
pub fn len(id: HashId) -> usize {
    match id {
        HashId::Sha256 => 32,
        HashId::Sha384 => 48,
        HashId::Sha512 => 64,
    }
}

/// The fixed DER `DigestInfo` prefix (algorithm-id SEQUENCE) PKCS#1 v1.5
/// places before the raw hash. Standard constants (RFC 8017 §9.2 note 1).
pub fn digestinfo_prefix(id: HashId) -> &'static [u8] {
    match id {
        HashId::Sha256 => &[
            0x30, 0x31, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x01, 0x05, 0x00, 0x04, 0x20,
        ],
        HashId::Sha384 => &[
            0x30, 0x41, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x02, 0x05, 0x00, 0x04, 0x30,
        ],
        HashId::Sha512 => &[
            0x30, 0x51, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x03, 0x05, 0x00, 0x04, 0x40,
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn sha256_empty_vector() {
        // NIST: SHA-256("") = e3b0c442...
        let d = digest(HashId::Sha256, b"");
        assert_eq!(d[0], 0xe3);
        assert_eq!(d[1], 0xb0);
        assert_eq!(len(HashId::Sha256), 32);
    }
}
