//! An **HTTP / Doc service agent**: a native daemon that listens on a TCP port,
//! parses HTTP/1.1 requests, and serves the Chitti OS documentation as HTML.
//! It is the vision's "HTTP agent does HTTP RFC handling" and "Doc agent serves
//! docs via the HTTP agent" collapsed into one native service — the tractable,
//! fully-testable instance of a protocol service agent.
//!
//! Determinism boundary: the HTTP request parsing and response building are
//! native, deterministic code (below the boundary), exactly as the standing rule
//! requires — an LLM never implements the protocol. The same accept →
//! `channel::adopt_tcp` → channel-I/O path as the echo service feeds it. SSH and
//! Git service agents would follow the identical shape: a native protocol module
//! reading/writing an accepted channel; only the wire grammar differs.

use crate::service::ServiceSpec;
use alloc::string::String;
use core::sync::atomic::{AtomicU16, Ordering};

static HTTP_PORT: AtomicU16 = AtomicU16::new(0);

pub fn set_port(port: u16) {
    HTTP_PORT.store(port, Ordering::SeqCst);
}

/// A parsed HTTP request line — the only part this Doc server needs. Pure and
/// unit-tested; the serve loop calls it on the bytes it read from the channel.
#[derive(Debug, PartialEq, Eq)]
pub struct Request {
    pub method: String,
    pub path: String,
}

/// Parse the request line (`METHOD SP path SP HTTP/x.y`) from the head of an
/// HTTP request. `None` if the first line is absent or malformed.
pub fn parse_request(buf: &[u8]) -> Option<Request> {
    let text = core::str::from_utf8(buf).ok()?;
    let line = text.split("\r\n").next()?;
    let mut parts = line.split(' ');
    let method = parts.next()?;
    let path = parts.next()?;
    let version = parts.next()?;
    if !version.starts_with("HTTP/") || method.is_empty() || !path.starts_with('/') {
        return None;
    }
    Some(Request { method: method.into(), path: path.into() })
}

/// Render the HTML document served for `path` (the "dynamic rendering" a Doc
/// agent does). A tiny router: `/` and `/docs` return the docs page; anything
/// else 404s. Returns `(status_line, body)`.
pub fn render(path: &str) -> (&'static str, String) {
    match path {
        "/" | "/docs" | "/index.html" => (
            "200 OK",
            String::from(
                "<!doctype html><html><head><title>Chitti OS</title></head><body>\
                 <h1>Chitti OS</h1><p>An agentic operating system: the agent is the driver. \
                 This page is served by the Doc service agent over the native HTTP service agent.</p>\
                 <ul><li>channels</li><li>services</li><li>capabilities</li></ul>\
                 </body></html>",
            ),
        ),
        _ => ("404 Not Found", String::from("<!doctype html><title>404</title><h1>Not found</h1>")),
    }
}

/// Build a complete HTTP/1.1 response for `path`.
fn response_for(path: &str) -> String {
    let (status, body) = render(path);
    alloc::format!(
        "HTTP/1.1 {status}\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

/// Per-connection read budget for the request head.
const REQ_DEADLINE_MS: u64 = 10_000;

extern "C" fn http_serve(_arg: u64) {
    let port = HTTP_PORT.load(Ordering::SeqCst);
    if port == 0 {
        return;
    }
    let listener = match crate::net::listen(port) {
        Ok(l) => l,
        Err(e) => {
            crate::ktrace::log_fmt(format_args!("service.http: listen :{port} failed: {e}"));
            return;
        }
    };
    let mut req = alloc::vec::Vec::new();
    let mut buf = [0u8; 1024];
    loop {
        if let Some(handle) = crate::net::try_accept(listener) {
            let ch = crate::channel::adopt_tcp(handle);
            req.clear();
            let deadline = crate::arch::now_ms() + REQ_DEADLINE_MS;
            // Read until the header terminator (or EOF/deadline).
            loop {
                match crate::channel::try_read(ch, &mut buf) {
                    Ok(0) => {
                        if crate::channel::is_eof(ch) {
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
            let path = parse_request(&req).map(|r| r.path).unwrap_or_else(|| String::from("/"));
            let resp = response_for(&path);
            crate::ktrace::log_fmt(format_args!("service.http: served {path}"));
            let bytes = resp.as_bytes();
            let mut off = 0;
            while off < bytes.len() {
                match crate::channel::try_write(ch, &bytes[off..]) {
                    Ok(0) => {
                        crate::shell::upkeep();
                        crate::sched::yield_now();
                    }
                    Ok(n) => off += n,
                    Err(_) => break,
                }
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

/// The HTTP/Doc service. Native, so no Synapse grants needed (it calls the
/// deterministic net/channel APIs directly). On-demand via `/agents start-http`.
pub static HTTP_SERVICE: ServiceSpec = ServiceSpec { name: "http-doc", entry: http_serve, autostart: false, caps: &[] };

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn parses_a_request_line() {
        let r = parse_request(b"GET /docs HTTP/1.1\r\nHost: x\r\n\r\n").unwrap();
        assert_eq!(r, Request { method: "GET".into(), path: "/docs".into() });
    }

    #[test_case]
    fn rejects_malformed_request_lines() {
        assert!(parse_request(b"garbage").is_none());
        assert!(parse_request(b"GET /docs FTP/1\r\n").is_none()); // not HTTP
        assert!(parse_request(b"GET docs HTTP/1.1\r\n").is_none()); // path not absolute
    }

    #[test_case]
    fn routes_docs_and_404() {
        assert_eq!(render("/").0, "200 OK");
        assert_eq!(render("/docs").0, "200 OK");
        assert!(render("/").1.contains("Chitti OS"));
        assert_eq!(render("/nope").0, "404 Not Found");
    }
}
