//! An **SSH service agent** — RFC 4253 version exchange plus **KEXINIT**
//! (key-exchange init). Authentication and channel multiplexing are next.
//!
//! On an accepted connection it:
//! 1. Sends `SSH-2.0-Chitti_0.1\\r\\n` and reads the client's banner
//! 2. Sends a `SSH_MSG_KEXINIT` naming the algorithms we will support
//! 3. Reads the peer's KEXINIT (best-effort) and closes
//!
//! Real Diffie-Hellman / curve25519 kex, host-key auth, and channels follow as
//! native deterministic code on the same accepted channel.
//!
//! Determinism boundary: the protocol is native code below the boundary. The
//! SSH *agent* (its markdown SOUL + manifest) supplies identity, the capability
//! grant, and — where real judgment belongs — the login/tunnel policy.

use crate::service::ServiceSpec;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU16, Ordering};

static SSH_PORT: AtomicU16 = AtomicU16::new(0);

pub fn set_port(port: u16) {
    SSH_PORT.store(port, Ordering::SeqCst);
}

/// Our SSH identification string (RFC 4253 §4.2). CR-LF terminated.
const IDENT: &[u8] = b"SSH-2.0-Chitti_0.1\r\n";

/// How long to wait for the peer's identification line before giving up.
const BANNER_DEADLINE_MS: u64 = 10_000;

/// SSH_MSG_KEXINIT = 20.
const SSH_MSG_KEXINIT: u8 = 20;

/// Name-lists advertised in our KEXINIT (comma-separated, no spaces).
const KEX_ALGORITHMS: &str = "curve25519-sha256,diffie-hellman-group14-sha256";
const HOST_KEY_ALGS: &str = "ssh-ed25519,rsa-sha2-256";
const ENC_ALGS: &str = "aes256-gcm@openssh.com,aes256-ctr";
const MAC_ALGS: &str = "hmac-sha2-256,hmac-sha2-512";
const COMP_ALGS: &str = "none";

/// Build an SSH binary packet carrying `SSH_MSG_KEXINIT` (RFC 4253 §7.1).
/// Pure — unit-tested so the wire shape cannot silently drift.
pub fn build_kexinit(cookie: &[u8; 16]) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.push(SSH_MSG_KEXINIT);
    payload.extend_from_slice(cookie);
    for name_list in [
        KEX_ALGORITHMS,
        HOST_KEY_ALGS,
        ENC_ALGS, // c2s
        ENC_ALGS, // s2c
        MAC_ALGS,
        MAC_ALGS,
        COMP_ALGS,
        COMP_ALGS,
        "", // languages c2s
        "", // languages s2c
    ] {
        write_name_list(&mut payload, name_list);
    }
    payload.push(0); // first_kex_packet_follows
    payload.extend_from_slice(&0u32.to_be_bytes()); // reserved

    // Binary packet: packet_length | padding_length | payload | padding
    // packet_length = 1 + payload + padding (excludes the length field itself).
    let mut padding_len = 4;
    while (payload.len() + 1 + padding_len) % 8 != 0 {
        padding_len += 1;
    }
    let packet_len = (1 + payload.len() + padding_len) as u32;
    let mut out = Vec::with_capacity(4 + packet_len as usize);
    out.extend_from_slice(&packet_len.to_be_bytes());
    out.push(padding_len as u8);
    out.extend_from_slice(&payload);
    out.resize(out.len() + padding_len, 0);
    out
}

fn write_name_list(buf: &mut Vec<u8>, names: &str) {
    let b = names.as_bytes();
    buf.extend_from_slice(&(b.len() as u32).to_be_bytes());
    buf.extend_from_slice(b);
}

/// True when `buf` looks like an SSH_MSG_KEXINIT binary packet (for logging).
pub fn looks_like_kexinit(buf: &[u8]) -> bool {
    if buf.len() < 6 {
        return false;
    }
    let packet_len = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
    if packet_len + 4 > buf.len() || packet_len < 2 {
        return false;
    }
    let padding_len = buf[4] as usize;
    let payload_start = 5;
    if payload_start >= buf.len() {
        return false;
    }
    buf[payload_start] == SSH_MSG_KEXINIT && padding_len < packet_len
}

extern "C" fn ssh_serve(_arg: u64) {
    let port = SSH_PORT.load(Ordering::SeqCst);
    if port == 0 {
        return;
    }
    let listener = match crate::net::listen(port) {
        Ok(l) => l,
        Err(e) => {
            crate::ktrace::log_fmt(format_args!("service.ssh: listen :{port} failed: {e}"));
            return;
        }
    };
    let mut buf = [0u8; 256];
    loop {
        if let Some(handle) = crate::net::try_accept(listener) {
            let ch = crate::channel::adopt_tcp(handle);
            // Version exchange: send our ident, then read the peer's line.
            let _ = crate::channel::try_write(ch, IDENT);
            let mut peer = alloc::vec::Vec::new();
            let deadline = crate::arch::now_ms() + BANNER_DEADLINE_MS;
            loop {
                match crate::channel::try_read(ch, &mut buf) {
                    Ok(0) => {
                        if crate::channel::is_eof(ch) {
                            break;
                        }
                    }
                    Ok(n) => {
                        peer.extend_from_slice(&buf[..n]);
                        if peer.windows(2).any(|w| w == b"\r\n") {
                            break;
                        }
                    }
                    Err(_) => break,
                }
                if crate::arch::now_ms() >= deadline {
                    break;
                }
                crate::shell::upkeep();
                crate::sched::yield_now();
            }
            crate::ktrace::log_fmt(format_args!(
                "service.ssh: version exchange ({} bytes from peer)",
                peer.len()
            ));

            // KEXINIT — cookie from the wall clock (not a CSPRNG yet; fine for
            // the handshake advertisement before real key exchange lands).
            let mut cookie = [0u8; 16];
            let t = crate::arch::now_ms().to_le_bytes();
            cookie[..8].copy_from_slice(&t);
            cookie[8..16].copy_from_slice(&t);
            let kex = build_kexinit(&cookie);
            let _ = crate::channel::try_write(ch, &kex);

            // Best-effort read of the peer's KEXINIT (do not block the service).
            let mut peer_kex = alloc::vec::Vec::new();
            let kex_deadline = crate::arch::now_ms() + 3_000;
            while crate::arch::now_ms() < kex_deadline && peer_kex.len() < 1024 {
                match crate::channel::try_read(ch, &mut buf) {
                    Ok(0) => {
                        if crate::channel::is_eof(ch) {
                            break;
                        }
                    }
                    Ok(n) => peer_kex.extend_from_slice(&buf[..n]),
                    Err(_) => break,
                }
                if looks_like_kexinit(&peer_kex) {
                    break;
                }
                crate::shell::upkeep();
                crate::sched::yield_now();
            }
            crate::ktrace::log_fmt(format_args!(
                "service.ssh: KEXINIT sent ({} B); peer reply {} B{}",
                kex.len(),
                peer_kex.len(),
                if looks_like_kexinit(&peer_kex) {
                    " (KEXINIT)"
                } else {
                    ""
                }
            ));
            // Real DH / host-key / NEWKEYS / auth would continue here.
            crate::channel::close_write(ch);
            crate::channel::close_read(ch);
            crate::channel::close_end(ch);
            crate::channel::close_end(ch);
        }
        crate::shell::upkeep();
        crate::sched::yield_now();
    }
}

/// The SSH service. Native, so no Synapse grants needed. On-demand.
pub static SSH_SERVICE: ServiceSpec = ServiceSpec {
    name: "ssh",
    entry: ssh_serve,
    autostart: false,
    caps: &[],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn kexinit_packet_is_well_formed() {
        let cookie = [0x11u8; 16];
        let pkt = build_kexinit(&cookie);
        assert!(pkt.len() >= 4 + 1 + 1 + 16);
        let packet_len = u32::from_be_bytes([pkt[0], pkt[1], pkt[2], pkt[3]]) as usize;
        assert_eq!(pkt.len(), 4 + packet_len);
        assert_eq!(pkt[5], SSH_MSG_KEXINIT, "payload must start with KEXINIT");
        assert_eq!(&pkt[6..22], &cookie);
        assert!(looks_like_kexinit(&pkt));
        // Name-list for kex algorithms follows the cookie.
        let kex_len = u32::from_be_bytes([pkt[22], pkt[23], pkt[24], pkt[25]]) as usize;
        let kex_names = core::str::from_utf8(&pkt[26..26 + kex_len]).unwrap();
        assert!(kex_names.contains("curve25519-sha256"));
    }

    #[test_case]
    fn looks_like_kexinit_rejects_garbage() {
        assert!(!looks_like_kexinit(&[]));
        assert!(!looks_like_kexinit(&[0, 0, 0, 5, 0, 1, 2, 3, 4]));
    }
}
