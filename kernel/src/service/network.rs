//! A minimal **Network service agent**: a native daemon that listens on a TCP
//! port and, for each inbound connection, adopts the socket as a channel and
//! echoes bytes back. It is the smallest faithful instance of the vision's
//! "Network agent listens on a port, forwards connections" — here the forward
//! target is an in-loop echo, but the accept → `channel::adopt_tcp` → channel
//! I/O path is exactly what a real forward (e.g. handing the stream to an SSH
//! agent via `channel_grant`) uses.
//!
//! The serve loop is native, deterministic code (no model) — protocol/byte
//! handling lives below the determinism boundary, as the standing rule requires.
//! It pumps `shell::upkeep()` + `yield_now()` on every idle spin so the net
//! stack and UI stay alive while it holds the CPU cooperatively.

use crate::service::ServiceSpec;
use core::sync::atomic::{AtomicU16, Ordering};

/// The port the echo service listens on; set by the shell before `start`.
static ECHO_PORT: AtomicU16 = AtomicU16::new(0);

/// Configure the listen port for the next start of [`ECHO_SERVICE`].
pub fn set_echo_port(port: u16) {
    ECHO_PORT.store(port, Ordering::SeqCst);
}

/// Per-connection idle budget: stop echoing a connection that goes quiet for
/// this long (so a half-open peer can't pin the loop forever).
const CONN_IDLE_MS: u64 = 15_000;

extern "C" fn echo_serve(_arg: u64) {
    let port = ECHO_PORT.load(Ordering::SeqCst);
    if port == 0 {
        return;
    }
    let listener = match crate::net::listen(port) {
        Ok(l) => l,
        Err(e) => {
            crate::ktrace::log_fmt(format_args!("service.network: listen :{port} failed: {e}"));
            return;
        }
    };
    let mut buf = [0u8; 1024];
    loop {
        if let Some(handle) = crate::net::try_accept(listener) {
            let ch = crate::channel::adopt_tcp(handle);
            crate::ktrace::log_fmt(format_args!("service.network: accepted a connection on :{port}"));
            let mut deadline = crate::arch::now_ms() + CONN_IDLE_MS;
            loop {
                match crate::channel::try_read(ch, &mut buf) {
                    Ok(0) => {
                        if crate::channel::is_eof(ch) {
                            break;
                        }
                    }
                    Ok(n) => {
                        let _ = crate::channel::try_write(ch, &buf[..n]);
                        deadline = crate::arch::now_ms() + CONN_IDLE_MS; // reset idle timer
                    }
                    Err(_) => break,
                }
                if crate::arch::now_ms() >= deadline {
                    break;
                }
                crate::shell::upkeep();
                crate::sched::yield_now();
            }
            crate::channel::close_write(ch);
            crate::channel::close_read(ch);
            crate::channel::close_end(ch);
            crate::channel::close_end(ch);
        }
        crate::shell::upkeep();
        crate::sched::yield_now();
    }
}

/// The Network echo service. Native, so it needs no Synapse `InvokePrimitive`
/// grants (it calls the deterministic kernel net/channel APIs directly, below
/// the determinism boundary). Not autostarted — brought up on demand by
/// `/agents start-net <port>`.
pub static ECHO_SERVICE: ServiceSpec = ServiceSpec { name: "network-echo", entry: echo_serve, autostart: false, caps: &[] };
