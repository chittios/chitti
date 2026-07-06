//! The native service behind the **Network agent**: it owns the inbound TCP
//! edge. It listens on a port, accepts a connection, reads the request bytes off
//! the socket, and relays them to the HTTP agent over a channel; when the HTTP
//! agent hands back the formatted response bytes, the Network agent writes them
//! to the socket. It never parses a protocol — it is the wire.
//!
//! Determinism boundary: this is native, deterministic code below the boundary.
//! The Network / HTTP / Doc split is wired in [`super::pipeline`].

use crate::service::{pipeline, ServiceSpec};

/// How long to wait for a request head / the HTTP agent's response before
/// dropping the connection. Generous: the HTTP agent's response waits on the Doc
/// agent planning the route with a live model turn (+ a one-time model load).
const CONN_DEADLINE_MS: u64 = 65_000;

extern "C" fn network_serve(_arg: u64) {
    let port = pipeline::net_port();
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
    let (Some(to_http), Some(from_http)) = (pipeline::net_to_http(), pipeline::http_to_net()) else {
        crate::ktrace::log("service.network", "pipeline channels not wired");
        return;
    };
    let mut buf = [0u8; 2048];
    loop {
        if let Some(handle) = crate::net::try_accept(listener) {
            let conn = crate::channel::adopt_tcp(handle);
            // Read the request head (until the header terminator) off the socket.
            let mut req = alloc::vec::Vec::new();
            let deadline = crate::arch::now_ms() + CONN_DEADLINE_MS;
            loop {
                match crate::channel::try_read(conn, &mut buf) {
                    Ok(0) => {
                        if crate::channel::is_eof(conn) {
                            break;
                        }
                    }
                    Ok(n) => {
                        req.extend_from_slice(&buf[..n]);
                        if req.windows(4).any(|w| w == b"\r\n\r\n") {
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
            if !req.is_empty() {
                // Hand the raw request to the HTTP agent and relay its response.
                let d = crate::arch::now_ms() + CONN_DEADLINE_MS;
                pipeline::send_frame(to_http, &req, d);
                if let Some(resp) = pipeline::recv_deadline(from_http, crate::arch::now_ms() + CONN_DEADLINE_MS) {
                    let mut off = 0;
                    while off < resp.len() {
                        match crate::channel::try_write(conn, &resp[off..]) {
                            Ok(0) => {
                                crate::shell::upkeep();
                                crate::sched::yield_now();
                            }
                            Ok(n) => off += n,
                            Err(_) => break,
                        }
                    }
                }
            }
            crate::channel::close_write(conn);
            crate::channel::close_read(conn);
            crate::channel::close_end(conn);
            crate::channel::close_end(conn);
        }
        crate::shell::upkeep();
        crate::sched::yield_now();
    }
}

/// The Network edge service. Native; started as part of the web pipeline.
pub static NETWORK_STAGE: ServiceSpec = ServiceSpec { name: "network", entry: network_serve, autostart: false, caps: &[] };
