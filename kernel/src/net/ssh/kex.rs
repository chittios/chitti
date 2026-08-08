//! Key exchange — algorithm negotiation (RFC 4253 §7.1), the exchange hash
//! (§8), and key derivation (§7.2).
//!
//! Pure. Everything here produces bytes that are never transmitted directly:
//! the exchange hash and the derived keys are *compared* against a peer that
//! computed them independently. So a mistake is not a protocol error the server
//! reports — it is a MAC that fails, or a signature that does not verify, and
//! the symptom points at the host key or the cipher rather than at the field
//! that was actually wrong. Three that bite:
//!
//! * **The client's preference wins, not the server's.** RFC 4253 §7.1 says the
//!   negotiated algorithm is the first on the *client's* list that also appears
//!   on the server's. Scanning the server's list instead usually picks the same
//!   thing, which is why it survives testing, and then silently downgrades on a
//!   server that orders its list differently.
//! * **The exchange hash covers the whole KEXINIT payload of both sides**,
//!   including the message byte and the cookie, exactly as it went on the wire.
//!   Re-serialising it from parsed fields produces a different hash whenever the
//!   peer's encoder differs from ours in any way at all.
//! * **`K` is an `mpint`**, so it is hashed with its length prefix and its
//!   sign-extension byte — not as a raw 32-byte secret. Half the curve25519
//!   secrets have their top bit set, so a client that hashes the raw bytes works
//!   about half the time.

use super::wire::{Reader, Writer};
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use sha2::{Digest, Sha256};

/// SSH message numbers used during kex.
pub const SSH_MSG_KEXINIT: u8 = 20;
pub const SSH_MSG_NEWKEYS: u8 = 21;
/// ECDH/curve25519 kex init (client → server), RFC 5656 / RFC 8731.
pub const SSH_MSG_KEX_ECDH_INIT: u8 = 30;
/// ECDH/curve25519 kex reply (server → client).
pub const SSH_MSG_KEX_ECDH_REPLY: u8 = 31;

/// Key exchange methods we implement, in preference order.
pub const KEX_ALGORITHMS: &[&str] = &["curve25519-sha256", "curve25519-sha256@libssh.org", "ecdh-sha2-nistp256"];
/// Host-key algorithms we can verify, in preference order.
pub const HOST_KEY_ALGORITHMS: &[&str] = &["ssh-ed25519", "ecdsa-sha2-nistp256"];
/// Ciphers we implement, in preference order.
pub const ENC_ALGORITHMS: &[&str] = &["aes256-gcm@openssh.com", "aes128-gcm@openssh.com", "aes256-ctr", "aes128-ctr"];
/// MACs we implement. Unused by the GCM ciphers (they authenticate themselves),
/// but a server that picks a CTR cipher needs one.
pub const MAC_ALGORITHMS: &[&str] = &["hmac-sha2-256-etm@openssh.com", "hmac-sha2-256"];
/// We do not implement compression, and say so rather than offering `zlib`.
pub const COMP_ALGORITHMS: &[&str] = &["none"];

/// The ten name-lists of a KEXINIT, in wire order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KexInit {
    pub cookie: [u8; 16],
    pub kex: Vec<String>,
    pub host_key: Vec<String>,
    pub enc_c2s: Vec<String>,
    pub enc_s2c: Vec<String>,
    pub mac_c2s: Vec<String>,
    pub mac_s2c: Vec<String>,
    pub comp_c2s: Vec<String>,
    pub comp_s2c: Vec<String>,
    pub lang_c2s: Vec<String>,
    pub lang_s2c: Vec<String>,
    pub first_kex_packet_follows: bool,
}

fn owned(v: Vec<&str>) -> Vec<String> {
    v.into_iter().map(|s| s.to_string()).collect()
}

impl KexInit {
    /// Our offer.
    pub fn ours(cookie: [u8; 16]) -> Self {
        Self {
            cookie,
            kex: KEX_ALGORITHMS.iter().map(|s| s.to_string()).collect(),
            host_key: HOST_KEY_ALGORITHMS.iter().map(|s| s.to_string()).collect(),
            enc_c2s: ENC_ALGORITHMS.iter().map(|s| s.to_string()).collect(),
            enc_s2c: ENC_ALGORITHMS.iter().map(|s| s.to_string()).collect(),
            mac_c2s: MAC_ALGORITHMS.iter().map(|s| s.to_string()).collect(),
            mac_s2c: MAC_ALGORITHMS.iter().map(|s| s.to_string()).collect(),
            comp_c2s: COMP_ALGORITHMS.iter().map(|s| s.to_string()).collect(),
            comp_s2c: COMP_ALGORITHMS.iter().map(|s| s.to_string()).collect(),
            lang_c2s: Vec::new(),
            lang_s2c: Vec::new(),
            first_kex_packet_follows: false,
        }
    }

    /// Serialise to a KEXINIT payload (message byte included).
    pub fn encode(&self) -> Vec<u8> {
        let mut w = Writer::msg(SSH_MSG_KEXINIT);
        w.put_raw(&self.cookie);
        for list in [
            &self.kex,
            &self.host_key,
            &self.enc_c2s,
            &self.enc_s2c,
            &self.mac_c2s,
            &self.mac_s2c,
            &self.comp_c2s,
            &self.comp_s2c,
            &self.lang_c2s,
            &self.lang_s2c,
        ] {
            let refs: Vec<&str> = list.iter().map(|s| s.as_str()).collect();
            w.put_name_list(&refs);
        }
        w.put_bool(self.first_kex_packet_follows);
        w.put_u32(0); // reserved
        w.into_vec()
    }

    /// Parse a KEXINIT payload (message byte included).
    pub fn decode(payload: &[u8]) -> Option<Self> {
        let mut r = Reader::new(payload);
        if r.u8()? != SSH_MSG_KEXINIT {
            return None;
        }
        let mut cookie = [0u8; 16];
        cookie.copy_from_slice(r.take(16)?);
        let mut out = Self {
            cookie,
            ..Default::default()
        };
        out.kex = owned(r.name_list()?);
        out.host_key = owned(r.name_list()?);
        out.enc_c2s = owned(r.name_list()?);
        out.enc_s2c = owned(r.name_list()?);
        out.mac_c2s = owned(r.name_list()?);
        out.mac_s2c = owned(r.name_list()?);
        out.comp_c2s = owned(r.name_list()?);
        out.comp_s2c = owned(r.name_list()?);
        out.lang_c2s = owned(r.name_list()?);
        out.lang_s2c = owned(r.name_list()?);
        out.first_kex_packet_follows = r.bool()?;
        Some(out)
    }
}

/// What both sides agreed to use.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Negotiated {
    pub kex: String,
    pub host_key: String,
    pub enc_c2s: String,
    pub enc_s2c: String,
    pub mac_c2s: String,
    pub mac_s2c: String,
}

/// **The client's order decides.** RFC 4253 §7.1: the chosen algorithm is the
/// first on the client's list that the server also names.
fn pick(client: &[String], server: &[String]) -> Option<String> {
    client
        .iter()
        .find(|c| server.iter().any(|s| s == *c))
        .cloned()
}

/// Negotiate every algorithm, or report the first list with no overlap.
///
/// The error names the *category*, because "no matching algorithm" without
/// saying which one is the least actionable message in SSH: a server that has
/// disabled a cipher and one that has disabled a kex look identical.
pub fn negotiate(ours: &KexInit, theirs: &KexInit) -> Result<Negotiated, &'static str> {
    let kex = pick(&ours.kex, &theirs.kex).ok_or("no common key-exchange algorithm")?;
    let host_key = pick(&ours.host_key, &theirs.host_key).ok_or("no common host-key algorithm")?;
    let enc_c2s = pick(&ours.enc_c2s, &theirs.enc_c2s).ok_or("no common cipher (client to server)")?;
    let enc_s2c = pick(&ours.enc_s2c, &theirs.enc_s2c).ok_or("no common cipher (server to client)")?;
    // A GCM cipher carries its own authentication, so the MAC list is allowed to
    // have no overlap — insisting on one would refuse a perfectly good server
    // that offers only ETM MACs we do not implement.
    let mac_c2s = if is_aead(&enc_c2s) {
        String::new()
    } else {
        pick(&ours.mac_c2s, &theirs.mac_c2s).ok_or("no common MAC (client to server)")?
    };
    let mac_s2c = if is_aead(&enc_s2c) {
        String::new()
    } else {
        pick(&ours.mac_s2c, &theirs.mac_s2c).ok_or("no common MAC (server to client)")?
    };
    Ok(Negotiated {
        kex,
        host_key,
        enc_c2s,
        enc_s2c,
        mac_c2s,
        mac_s2c,
    })
}

/// True for ciphers that authenticate their own packets (no separate MAC).
pub fn is_aead(name: &str) -> bool {
    name.ends_with("-gcm@openssh.com") || name == "chacha20-poly1305@openssh.com"
}

/// The exchange hash `H` (RFC 4253 §8 / RFC 5656 §4).
///
/// ```text
/// H = hash(V_C || V_S || I_C || I_S || K_S || Q_C || Q_S || K)
/// ```
///
/// `v_c`/`v_s` are the identification strings **without** their CR-LF, `i_c`/
/// `i_s` the KEXINIT payloads **exactly as sent**, `k_s` the server's host-key
/// blob, `q_c`/`q_s` the ephemeral public keys, and `k` the shared secret as an
/// unsigned magnitude — hashed as an `mpint`, not raw.
pub fn exchange_hash(
    v_c: &[u8],
    v_s: &[u8],
    i_c: &[u8],
    i_s: &[u8],
    k_s: &[u8],
    q_c: &[u8],
    q_s: &[u8],
    k: &[u8],
) -> Vec<u8> {
    let mut w = Writer::new();
    w.put_string(v_c);
    w.put_string(v_s);
    w.put_string(i_c);
    w.put_string(i_s);
    w.put_string(k_s);
    w.put_string(q_c);
    w.put_string(q_s);
    w.put_mpint(k);
    Sha256::digest(w.as_slice()).to_vec()
}

/// Which of the six keys to derive (RFC 4253 §7.2). The letter is hashed in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyKind {
    IvC2s,
    IvS2c,
    EncC2s,
    EncS2c,
    MacC2s,
    MacS2c,
}

impl KeyKind {
    fn letter(self) -> u8 {
        match self {
            KeyKind::IvC2s => b'A',
            KeyKind::IvS2c => b'B',
            KeyKind::EncC2s => b'C',
            KeyKind::EncS2c => b'D',
            KeyKind::MacC2s => b'E',
            KeyKind::MacS2c => b'F',
        }
    }
}

/// Derive `len` bytes of key material (RFC 4253 §7.2).
///
/// ```text
/// K1 = hash(K || H || letter || session_id)
/// K2 = hash(K || H || K1)          … and so on, concatenated
/// ```
///
/// Note `K` is an `mpint` here too, and the *session id* — not `H` — is what
/// the first block hashes. They are equal on the first key exchange and differ
/// on every rekey, so getting it wrong works until the connection has been up
/// long enough to rekey, which is exactly when it is hardest to debug.
pub fn derive_key(k: &[u8], h: &[u8], kind: KeyKind, session_id: &[u8], len: usize) -> Vec<u8> {
    let mut k_mpint = Writer::new();
    k_mpint.put_mpint(k);
    let k_mpint = k_mpint.into_vec();

    let mut out: Vec<u8> = Vec::with_capacity(len + 32);
    let mut hasher = Sha256::new();
    hasher.update(&k_mpint);
    hasher.update(h);
    hasher.update([kind.letter()]);
    hasher.update(session_id);
    out.extend_from_slice(&hasher.finalize());

    while out.len() < len {
        let mut hasher = Sha256::new();
        hasher.update(&k_mpint);
        hasher.update(h);
        hasher.update(&out);
        out.extend_from_slice(&hasher.finalize());
    }
    out.truncate(len);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn list(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    /// A KEXINIT round-trips byte-for-byte, which is what makes it safe to hash
    /// the bytes we sent rather than a re-serialisation.
    #[test_case]
    fn kexinit_round_trips() {
        let ours = KexInit::ours([0x5a; 16]);
        let bytes = ours.encode();
        let back = KexInit::decode(&bytes).expect("our own KEXINIT must parse");
        assert_eq!(back, ours);
        assert_eq!(back.encode(), bytes);
        assert_eq!(bytes[0], SSH_MSG_KEXINIT);
        assert_eq!(&bytes[1..17], &[0x5a; 16]);
    }

    /// Truncated or mistyped KEXINITs are refused, not half-parsed.
    #[test_case]
    fn kexinit_decode_refuses_garbage() {
        assert!(KexInit::decode(&[]).is_none());
        assert!(KexInit::decode(&[SSH_MSG_KEXINIT]).is_none(), "no cookie");
        let mut wrong = KexInit::ours([0; 16]).encode();
        wrong[0] = SSH_MSG_NEWKEYS;
        assert!(KexInit::decode(&wrong).is_none(), "wrong message number");
        let ok = KexInit::ours([0; 16]).encode();
        assert!(KexInit::decode(&ok[..ok.len() / 2]).is_none(), "truncated");
    }

    /// **The client's preference order decides**, not the server's.
    #[test_case]
    fn negotiation_follows_the_client_order() {
        let mut ours = KexInit::ours([0; 16]);
        ours.kex = list(&["curve25519-sha256", "ecdh-sha2-nistp256"]);
        let mut theirs = KexInit::ours([0; 16]);
        // Server prefers the opposite order; ours must still win.
        theirs.kex = list(&["ecdh-sha2-nistp256", "curve25519-sha256"]);
        let n = negotiate(&ours, &theirs).expect("overlap exists");
        assert_eq!(n.kex, "curve25519-sha256", "the client's first choice must win");

        // And when our first choice is absent we fall to our second, not theirs.
        theirs.kex = list(&["diffie-hellman-group14-sha256", "ecdh-sha2-nistp256"]);
        let n = negotiate(&ours, &theirs).unwrap();
        assert_eq!(n.kex, "ecdh-sha2-nistp256");
    }

    /// No overlap names the category that failed.
    #[test_case]
    fn negotiation_failure_names_the_category() {
        let ours = KexInit::ours([0; 16]);
        let mut theirs = KexInit::ours([0; 16]);
        theirs.kex = list(&["diffie-hellman-group1-sha1"]);
        assert_eq!(negotiate(&ours, &theirs).unwrap_err(), "no common key-exchange algorithm");

        let mut theirs = KexInit::ours([0; 16]);
        theirs.host_key = list(&["ssh-dss"]);
        assert_eq!(negotiate(&ours, &theirs).unwrap_err(), "no common host-key algorithm");

        let mut theirs = KexInit::ours([0; 16]);
        theirs.enc_s2c = list(&["3des-cbc"]);
        assert_eq!(negotiate(&ours, &theirs).unwrap_err(), "no common cipher (server to client)");
    }

    /// An AEAD cipher needs no MAC, so an empty MAC intersection is fine — but
    /// only then.
    #[test_case]
    fn aead_ciphers_do_not_require_a_mac() {
        let ours = KexInit::ours([0; 16]);
        let mut theirs = KexInit::ours([0; 16]);
        theirs.mac_c2s = list(&["hmac-md5"]);
        theirs.mac_s2c = list(&["hmac-md5"]);
        // Both sides still prefer AES-GCM, which authenticates itself.
        let n = negotiate(&ours, &theirs).expect("GCM needs no MAC");
        assert!(is_aead(&n.enc_s2c));
        assert_eq!(n.mac_s2c, "", "an AEAD cipher derives no MAC key");

        // With a CTR cipher the MAC is required again.
        let mut theirs = KexInit::ours([0; 16]);
        theirs.enc_c2s = list(&["aes256-ctr"]);
        theirs.enc_s2c = list(&["aes256-ctr"]);
        theirs.mac_c2s = list(&["hmac-md5"]);
        theirs.mac_s2c = list(&["hmac-md5"]);
        assert!(negotiate(&ours, &theirs).is_err(), "a CTR cipher must have a MAC");
    }

    /// `K` enters both the exchange hash and the KDF as an **mpint**, so a
    /// secret whose top bit is set hashes differently from its raw bytes.
    /// About half of all curve25519 secrets have that bit set.
    #[test_case]
    fn shared_secret_is_hashed_as_an_mpint() {
        let high = [0x80u8; 32];
        let low = [0x7fu8; 32];
        let h_high = exchange_hash(b"a", b"b", b"c", b"d", b"e", b"f", b"g", &high);
        let h_low = exchange_hash(b"a", b"b", b"c", b"d", b"e", b"f", b"g", &low);
        assert_ne!(h_high, h_low);

        // Hashing the raw bytes instead would drop the sign byte; prove the
        // encoded form really is one byte longer for a high-bit secret.
        let mut w = Writer::new();
        w.put_mpint(&high);
        assert_eq!(w.as_slice()[..4], [0, 0, 0, 33], "high-bit secret gains a 0x00");
        let mut w = Writer::new();
        w.put_mpint(&low);
        assert_eq!(w.as_slice()[..4], [0, 0, 0, 32]);
    }

    /// Key derivation extends past one hash block by chaining, and each of the
    /// six letters gives different material.
    #[test_case]
    fn key_derivation_chains_and_separates_by_letter() {
        let k = [0x01u8; 32];
        let h = [0x02u8; 32];
        let sid = h;

        let a = derive_key(&k, &h, KeyKind::IvC2s, &sid, 16);
        let b = derive_key(&k, &h, KeyKind::IvS2c, &sid, 16);
        assert_ne!(a, b, "A and B must differ");
        assert_eq!(a.len(), 16);

        // A key longer than one SHA-256 block must chain, and the first 32 bytes
        // must equal the unchained derivation — the chain extends, never restarts.
        let short = derive_key(&k, &h, KeyKind::EncC2s, &sid, 32);
        let long = derive_key(&k, &h, KeyKind::EncC2s, &sid, 64);
        assert_eq!(long.len(), 64);
        assert_eq!(&long[..32], &short[..]);
        assert_ne!(&long[32..], &short[..], "the second block must be new material");

        // All six letters are distinct.
        let kinds = [
            KeyKind::IvC2s,
            KeyKind::IvS2c,
            KeyKind::EncC2s,
            KeyKind::EncS2c,
            KeyKind::MacC2s,
            KeyKind::MacS2c,
        ];
        for (i, x) in kinds.iter().enumerate() {
            for y in kinds.iter().skip(i + 1) {
                assert_ne!(
                    derive_key(&k, &h, *x, &sid, 32),
                    derive_key(&k, &h, *y, &sid, 32),
                    "two key kinds produced the same material"
                );
            }
        }
    }

    /// The session id is a separate input from `H`, and it is what the first
    /// block hashes. Equal on the first kex, different on every rekey.
    #[test_case]
    fn derivation_uses_the_session_id_not_the_current_hash() {
        let k = [0x03u8; 32];
        let h_first = [0x04u8; 32];
        let h_rekey = [0x05u8; 32];
        // Same H, different session id → different keys.
        let a = derive_key(&k, &h_rekey, KeyKind::EncC2s, &h_first, 32);
        let b = derive_key(&k, &h_rekey, KeyKind::EncC2s, &h_rekey, 32);
        assert_ne!(a, b, "the session id must be an independent input");
    }
}
