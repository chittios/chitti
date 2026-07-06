//! An **SSH service agent** (transport version-exchange; the rest is a stub).
//!
//! On an accepted connection it performs the RFC 4253 §4.2 identification-string
//! exchange — sends `SSH-2.0-Chitti_0.1\r\n` and reads the client's banner —
//! then closes. Real key exchange (RFC 4253), authentication (RFC 4252), and
//! channel multiplexing (RFC 4254) would follow here as native, deterministic
//! code fed by the same accepted channel; they are not implemented yet.
//!
//! Determinism boundary: the protocol is native code below the boundary. The
//! SSH *agent* (its markdown SOUL + manifest) supplies identity, the capability
//! grant, and — where real judgment belongs — the login/tunnel policy, expressed
//! as grammar-validated, audited primitive calls, never raw bytes.

use crate::service::ServiceSpec;
use core::sync::atomic::{AtomicU16, Ordering};

static SSH_PORT: AtomicU16 = AtomicU16::new(0);

pub fn set_port(port: u16) {
    SSH_PORT.store(port, Ordering::SeqCst);
}

/// Our SSH identification string (RFC 4253 §4.2). CR-LF terminated.
const IDENT: &[u8] = b"SSH-2.0-Chitti_0.1\r\n";

/// How long to wait for the peer's identification line before giving up.
const BANNER_DEADLINE_MS: u64 = 10_000;

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
                "service.ssh: version exchange with a client ({} bytes) — transport stub, closing",
                peer.len()
            ));
            // Real kex/auth/channel handling would continue here.
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
pub static SSH_SERVICE: ServiceSpec = ServiceSpec { name: "ssh", entry: ssh_serve, autostart: false, caps: &[] };
