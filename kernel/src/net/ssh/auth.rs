//! User authentication (RFC 4252): the `publickey` and `password` methods, and
//! reading an OpenSSH private key off the store.
//!
//! Pure. The load-bearing detail is the **signed blob**, which is not the thing
//! that goes on the wire: RFC 4252 §7 signs
//!
//! ```text
//! string session_id || byte SSH_MSG_USERAUTH_REQUEST || string user ||
//! string "ssh-connection" || string "publickey" || boolean TRUE ||
//! string algorithm || string public key
//! ```
//!
//! while the request itself omits the leading `session_id`. Signing the request
//! as sent — the obvious reading — produces a signature the server rejects with
//! a bare "permission denied", indistinguishable from the wrong key being
//! offered. Binding the session id is also the whole point: it stops a
//! signature captured on one connection being replayed onto another.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use super::hostkey::PublicKey;
use super::wire::{Reader, Writer};

pub const SSH_MSG_USERAUTH_REQUEST: u8 = 50;
pub const SSH_MSG_USERAUTH_FAILURE: u8 = 51;
pub const SSH_MSG_USERAUTH_SUCCESS: u8 = 52;
pub const SSH_MSG_USERAUTH_BANNER: u8 = 53;
/// Only meaningful inside a `publickey` exchange (RFC 4252 §7).
pub const SSH_MSG_USERAUTH_PK_OK: u8 = 60;
pub const SSH_MSG_SERVICE_REQUEST: u8 = 5;
pub const SSH_MSG_SERVICE_ACCEPT: u8 = 6;

/// A private key we can authenticate with.
pub enum PrivateKey {
    Ed25519(alloc::boxed::Box<ed25519_dalek::SigningKey>),
    EcdsaP256(alloc::boxed::Box<p256::ecdsa::SigningKey>),
}

/// **Redacted on purpose.** A derived `Debug` would print the private scalar,
/// and every `unwrap`/`expect` on a `Result<PrivateKey, _>` formats it — so the
/// key would end up in a log line written by code that never meant to touch it.
impl core::fmt::Debug for PrivateKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "PrivateKey({}, redacted)", self.algorithm())
    }
}

impl PrivateKey {
    pub fn algorithm(&self) -> &'static str {
        match self {
            PrivateKey::Ed25519(_) => "ssh-ed25519",
            PrivateKey::EcdsaP256(_) => "ecdsa-sha2-nistp256",
        }
    }

    pub fn public(&self) -> PublicKey {
        match self {
            PrivateKey::Ed25519(k) => PublicKey::Ed25519(k.verifying_key().to_bytes()),
            PrivateKey::EcdsaP256(k) => {
                PublicKey::EcdsaP256(k.verifying_key().to_encoded_point(false).as_bytes().to_vec())
            }
        }
    }

    /// Sign `message`, returning the wire signature blob.
    pub fn sign(&self, message: &[u8]) -> Vec<u8> {
        let mut w = Writer::new();
        w.put_str(self.algorithm());
        match self {
            PrivateKey::Ed25519(k) => {
                use ed25519_dalek::Signer;
                w.put_string(&k.sign(message).to_bytes());
            }
            PrivateKey::EcdsaP256(k) => {
                use p256::ecdsa::signature::Signer;
                let sig: p256::ecdsa::Signature = k.sign(message);
                // Same two-mpint framing the verifier expects.
                let mut inner = Writer::new();
                inner.put_mpint(&sig.r().to_bytes());
                inner.put_mpint(&sig.s().to_bytes());
                w.put_string(inner.as_slice());
            }
        }
        w.into_vec()
    }
}

/// Parse an unencrypted OpenSSH private key (`-----BEGIN OPENSSH PRIVATE KEY-----`).
///
/// The container is `openssh-key-v1\0`, then cipher/kdf names, then the public
/// keys, then a *string* holding the private section — which itself begins with
/// two identical 32-bit check integers. Those check integers are the only way to
/// tell "wrong passphrase" from "corrupt file", and since we do not implement
/// passphrases, an encrypted key must be reported as such rather than parsed
/// into garbage.
pub fn parse_openssh_private(pem: &str) -> Result<PrivateKey, &'static str> {
    const BEGIN: &str = "-----BEGIN OPENSSH PRIVATE KEY-----";
    const END: &str = "-----END OPENSSH PRIVATE KEY-----";
    let start = pem.find(BEGIN).ok_or("not an OpenSSH private key (no BEGIN line)")?;
    let rest = &pem[start + BEGIN.len()..];
    let end = rest.find(END).ok_or("truncated private key (no END line)")?;
    let b64: String = rest[..end].chars().filter(|c| !c.is_whitespace()).collect();
    let blob = crate::net::ws::base64_decode(&b64).ok_or("private key is not valid base64")?;

    let magic = b"openssh-key-v1\0";
    if !blob.starts_with(magic) {
        return Err("unsupported private key format (not openssh-key-v1)");
    }
    let mut r = Reader::new(&blob[magic.len()..]);
    let cipher = r.utf8().ok_or("malformed private key header")?;
    let kdf = r.utf8().ok_or("malformed private key header")?;
    let _kdf_opts = r.string().ok_or("malformed private key header")?;
    if cipher != "none" || kdf != "none" {
        // Refused, not attempted: a passphrase-protected key decrypts to noise
        // without the passphrase, and "corrupt key" would be the wrong message.
        return Err("the private key is passphrase-protected (not supported yet)");
    }
    let count = r.u32().ok_or("malformed private key")?;
    if count != 1 {
        return Err("private key files with multiple keys are not supported");
    }
    let _public = r.string().ok_or("malformed private key")?;
    let private = r.string().ok_or("malformed private key")?;

    let mut p = Reader::new(private);
    let c1 = p.u32().ok_or("malformed private key body")?;
    let c2 = p.u32().ok_or("malformed private key body")?;
    if c1 != c2 {
        return Err("private key body failed its integrity check");
    }
    match p.utf8().ok_or("malformed private key body")? {
        "ssh-ed25519" => {
            let _pub = p.string().ok_or("malformed ed25519 key")?;
            // OpenSSH stores seed||public in one 64-byte string.
            let secret = p.string().ok_or("malformed ed25519 key")?;
            if secret.len() != 64 {
                return Err("malformed ed25519 private key");
            }
            let seed: [u8; 32] = secret[..32].try_into().map_err(|_| "malformed ed25519 key")?;
            Ok(PrivateKey::Ed25519(alloc::boxed::Box::new(
                ed25519_dalek::SigningKey::from_bytes(&seed),
            )))
        }
        "ecdsa-sha2-nistp256" => {
            if p.utf8().ok_or("malformed ecdsa key")? != "nistp256" {
                return Err("unsupported ECDSA curve (only nistp256)");
            }
            let _q = p.string().ok_or("malformed ecdsa key")?;
            let d = p.mpint().ok_or("malformed ecdsa key")?;
            let mut scalar = [0u8; 32];
            if d.len() > 32 {
                return Err("malformed ecdsa private scalar");
            }
            scalar[32 - d.len()..].copy_from_slice(d);
            let sk = p256::ecdsa::SigningKey::from_bytes(&scalar.into())
                .map_err(|_| "invalid ecdsa private scalar")?;
            Ok(PrivateKey::EcdsaP256(alloc::boxed::Box::new(sk)))
        }
        other => {
            // Named rather than generic: "unsupported key type" without the type
            // leaves the user guessing which of their keys to try next.
            if other == "ssh-rsa" {
                Err("RSA private keys are not supported yet (use ed25519)")
            } else {
                Err("unsupported private key type")
            }
        }
    }
}

/// `SSH_MSG_SERVICE_REQUEST` for `ssh-userauth`.
pub fn service_request(name: &str) -> Vec<u8> {
    let mut w = Writer::msg(SSH_MSG_SERVICE_REQUEST);
    w.put_str(name);
    w.into_vec()
}

/// The `none` method — sent to learn which methods the server will accept.
pub fn request_none(user: &str) -> Vec<u8> {
    let mut w = Writer::msg(SSH_MSG_USERAUTH_REQUEST);
    w.put_str(user);
    w.put_str("ssh-connection");
    w.put_str("none");
    w.into_vec()
}

/// A `password` request.
pub fn request_password(user: &str, password: &str) -> Vec<u8> {
    let mut w = Writer::msg(SSH_MSG_USERAUTH_REQUEST);
    w.put_str(user);
    w.put_str("ssh-connection");
    w.put_str("password");
    w.put_bool(false); // not a password change
    w.put_str(password);
    w.into_vec()
}

/// The request body shared by the probe and the signed form (everything after
/// the message byte, up to and including the public key).
fn publickey_body(user: &str, key: &PublicKey, signed: bool) -> Writer {
    let mut w = Writer::new();
    w.put_str(user);
    w.put_str("ssh-connection");
    w.put_str("publickey");
    w.put_bool(signed);
    w.put_str(key.algorithm());
    w.put_string(&key.encode());
    w
}

/// A `publickey` **probe** — asks whether the server would accept this key,
/// without signing anything. Answered with `SSH_MSG_USERAUTH_PK_OK`.
pub fn request_publickey_probe(user: &str, key: &PublicKey) -> Vec<u8> {
    let mut w = Writer::msg(SSH_MSG_USERAUTH_REQUEST);
    w.put_raw(publickey_body(user, key, false).as_slice());
    w.into_vec()
}

/// **What RFC 4252 §7 actually signs** — the request, prefixed with the session
/// id, which is *not* part of the transmitted request.
pub fn publickey_signed_data(session_id: &[u8], user: &str, key: &PublicKey) -> Vec<u8> {
    let mut w = Writer::new();
    w.put_string(session_id);
    w.put_u8(SSH_MSG_USERAUTH_REQUEST);
    w.put_raw(publickey_body(user, key, true).as_slice());
    w.into_vec()
}

/// A signed `publickey` request.
pub fn request_publickey(session_id: &[u8], user: &str, key: &PrivateKey) -> Vec<u8> {
    let pubkey = key.public();
    let signature = key.sign(&publickey_signed_data(session_id, user, &pubkey));
    let mut w = Writer::msg(SSH_MSG_USERAUTH_REQUEST);
    w.put_raw(publickey_body(user, &pubkey, true).as_slice());
    w.put_string(&signature);
    w.into_vec()
}

/// The methods a `SSH_MSG_USERAUTH_FAILURE` says may still work.
pub fn parse_failure(payload: &[u8]) -> Option<(Vec<String>, bool)> {
    let mut r = Reader::new(payload);
    if r.u8()? != SSH_MSG_USERAUTH_FAILURE {
        return None;
    }
    let methods = r.name_list()?.into_iter().map(|s| s.to_string()).collect();
    let partial = r.bool()?;
    Some((methods, partial))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The signed data is not the transmitted request.** It has the session id
    /// in front and no signature after it.
    #[test_case]
    fn the_signature_covers_the_session_id() {
        let key = PublicKey::Ed25519([3u8; 32]);
        let sid = [0xABu8; 32];
        let signed = publickey_signed_data(&sid, "git", &key);

        // It starts with the session id as a `string`.
        let mut r = Reader::new(&signed);
        assert_eq!(r.string().unwrap(), &sid[..]);
        assert_eq!(r.u8().unwrap(), SSH_MSG_USERAUTH_REQUEST);
        assert_eq!(r.utf8().unwrap(), "git");
        assert_eq!(r.utf8().unwrap(), "ssh-connection");
        assert_eq!(r.utf8().unwrap(), "publickey");
        assert!(r.bool().unwrap(), "the signed form sets the boolean to TRUE");

        // A different session id must produce different signed data — that is
        // what stops a captured signature being replayed onto another connection.
        let other = publickey_signed_data(&[0xCDu8; 32], "git", &key);
        assert_ne!(signed, other);
    }

    /// The probe and the signed request differ only in the boolean and the
    /// trailing signature — a server matches them against each other.
    #[test_case]
    fn probe_and_signed_requests_agree_on_everything_but_the_flag() {
        let sk = ed25519_dalek::SigningKey::from_bytes(&[5u8; 32]);
        let key = PrivateKey::Ed25519(alloc::boxed::Box::new(sk));
        let pubkey = key.public();
        let probe = request_publickey_probe("git", &pubkey);
        let signed = request_publickey(&[0u8; 32], "git", &key);

        // Same user / service / method.
        let mut a = Reader::new(&probe);
        let mut b = Reader::new(&signed);
        assert_eq!(a.u8(), b.u8());
        for _ in 0..3 {
            assert_eq!(a.utf8(), b.utf8());
        }
        assert!(!a.bool().unwrap(), "the probe is unsigned");
        assert!(b.bool().unwrap(), "the request is signed");
        assert_eq!(a.utf8(), b.utf8(), "same algorithm name");
        assert_eq!(a.string(), b.string(), "same key blob");
        assert!(a.is_empty(), "a probe carries no signature");
        assert!(!b.is_empty(), "a signed request carries one");
    }

    /// A signature we produce verifies against our own verifier — which is what
    /// proves the two framings agree, since the server does exactly this.
    #[test_case]
    fn a_publickey_signature_verifies_against_the_public_half() {
        for key in [
            PrivateKey::Ed25519(alloc::boxed::Box::new(ed25519_dalek::SigningKey::from_bytes(&[9u8; 32]))),
            PrivateKey::EcdsaP256(alloc::boxed::Box::new(
                p256::ecdsa::SigningKey::from_bytes(&[9u8; 32].into()).unwrap(),
            )),
        ] {
            let pubkey = key.public();
            let sid = [0x11u8; 32];
            let data = publickey_signed_data(&sid, "git", &pubkey);
            let sig = key.sign(&data);
            assert!(pubkey.verify(&data, &sig), "{} must verify", key.algorithm());
            // Bound to the session id: the same signature over a different one fails.
            let other = publickey_signed_data(&[0x22u8; 32], "git", &pubkey);
            assert!(!pubkey.verify(&other, &sig), "must not verify under another session");
        }
    }

    /// An unencrypted ed25519 key round-trips from the OpenSSH container.
    ///
    /// Built here with the real framing rather than pasted as a fixture, so the
    /// test pins our *reader* against a writer that follows the spec.
    #[test_case]
    fn an_openssh_ed25519_key_parses() {
        let sk = ed25519_dalek::SigningKey::from_bytes(&[77u8; 32]);
        let vk = sk.verifying_key();
        let pem = fake_openssh_ed25519(&sk, 0x1234_5678, 0x1234_5678);
        let parsed = parse_openssh_private(&pem).expect("must parse");
        assert_eq!(parsed.algorithm(), "ssh-ed25519");
        assert_eq!(parsed.public(), PublicKey::Ed25519(vk.to_bytes()));
        // And it signs compatibly.
        let msg = b"hello";
        assert!(parsed.public().verify(msg, &parsed.sign(msg)));
    }

    /// Mismatched check integers mean a corrupt (or wrongly decrypted) key, and
    /// are reported rather than parsed through.
    #[test_case]
    fn a_bad_check_integer_is_reported() {
        let sk = ed25519_dalek::SigningKey::from_bytes(&[77u8; 32]);
        let pem = fake_openssh_ed25519(&sk, 1, 2);
        assert_eq!(
            parse_openssh_private(&pem).unwrap_err(),
            "private key body failed its integrity check"
        );
    }

    /// An encrypted key is named as such, not reported as corrupt.
    #[test_case]
    fn an_encrypted_key_says_so() {
        let mut blob = b"openssh-key-v1\0".to_vec();
        let mut w = Writer::new();
        w.put_str("aes256-ctr");
        w.put_str("bcrypt");
        w.put_string(&[0u8; 24]);
        w.put_u32(1);
        w.put_string(&[0u8; 4]);
        w.put_string(&[0u8; 4]);
        blob.extend_from_slice(w.as_slice());
        let pem = alloc::format!(
            "-----BEGIN OPENSSH PRIVATE KEY-----\n{}\n-----END OPENSSH PRIVATE KEY-----\n",
            crate::net::ws::base64_encode(&blob)
        );
        assert_eq!(
            parse_openssh_private(&pem).unwrap_err(),
            "the private key is passphrase-protected (not supported yet)"
        );
    }

    /// Non-keys and truncated keys fail with a message that says which.
    #[test_case]
    fn private_key_parsing_fails_closed() {
        assert!(parse_openssh_private("").is_err());
        assert!(parse_openssh_private("-----BEGIN OPENSSH PRIVATE KEY-----\nAAAA\n").is_err());
        // A PEM RSA key is a different container entirely.
        assert!(parse_openssh_private("-----BEGIN RSA PRIVATE KEY-----\nAAAA\n-----END RSA PRIVATE KEY-----").is_err());
    }

    /// A failure message lists the methods still worth trying.
    #[test_case]
    fn userauth_failure_lists_remaining_methods() {
        let mut w = Writer::msg(SSH_MSG_USERAUTH_FAILURE);
        w.put_name_list(&["publickey", "password"]);
        w.put_bool(false);
        let (methods, partial) = parse_failure(&w.into_vec()).unwrap();
        assert_eq!(methods, alloc::vec!["publickey".to_string(), "password".to_string()]);
        assert!(!partial);
        // A different message number is not a failure record.
        assert!(parse_failure(&[SSH_MSG_USERAUTH_SUCCESS]).is_none());
    }

    /// Build an unencrypted OpenSSH ed25519 container, as `ssh-keygen` would.
    fn fake_openssh_ed25519(sk: &ed25519_dalek::SigningKey, c1: u32, c2: u32) -> String {
        let vk = sk.verifying_key();
        let pubblob = PublicKey::Ed25519(vk.to_bytes()).encode();

        let mut priv_body = Writer::new();
        priv_body.put_u32(c1);
        priv_body.put_u32(c2);
        priv_body.put_str("ssh-ed25519");
        priv_body.put_string(&vk.to_bytes());
        let mut secret = alloc::vec::Vec::new();
        secret.extend_from_slice(&sk.to_bytes());
        secret.extend_from_slice(&vk.to_bytes());
        priv_body.put_string(&secret);
        priv_body.put_str("comment");
        // Pad to an 8-byte multiple with 1,2,3… as OpenSSH does.
        let mut body = priv_body.into_vec();
        let mut i = 1u8;
        while body.len() % 8 != 0 {
            body.push(i);
            i += 1;
        }

        let mut w = Writer::new();
        w.put_str("none");
        w.put_str("none");
        w.put_string(&[]);
        w.put_u32(1);
        w.put_string(&pubblob);
        w.put_string(&body);

        let mut blob = b"openssh-key-v1\0".to_vec();
        blob.extend_from_slice(w.as_slice());
        alloc::format!(
            "-----BEGIN OPENSSH PRIVATE KEY-----\n{}\n-----END OPENSSH PRIVATE KEY-----\n",
            crate::net::ws::base64_encode(&blob)
        )
    }
}
