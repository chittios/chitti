//! The native service behind the **HTTP agent**: pure HTTP/1.1 protocol. It
//! receives the raw request bytes from the Network agent, parses the request
//! line + headers, forwards the method + path to the content agent (Doc) over a
//! channel, and — when that agent returns a body — formats a proper HTTP/1.1
//! response (status line, content-type, content-length) and hands it back to the
//! Network agent to put on the wire.
//!
//! It never touches the socket and never touches the filesystem: it is the
//! protocol layer between the network edge and the application. The parse and
//! format functions are pure and unit-tested; the serve loop just wires them to
//! the pipeline channels.

use crate::service::{pipeline, ServiceSpec};
use alloc::string::String;
use alloc::vec::Vec;

/// A parsed HTTP request: the parts a content agent needs. `headers` is the raw
/// header lines (kept so the protocol layer *could* forward them; the Doc agent
/// only needs method + path).
#[derive(Debug, PartialEq, Eq)]
pub struct Request {
    pub method: String,
    pub path: String,
    pub headers: Vec<String>,
}

/// Parse an HTTP request head: the request line (`METHOD SP path SP HTTP/x.y`)
/// plus header lines up to the blank line. `None` if the request line is absent
/// or malformed.
pub fn parse_request(buf: &[u8]) -> Option<Request> {
    let text = core::str::from_utf8(buf).ok()?;
    let mut lines = text.split("\r\n");
    let line = lines.next()?;
    let mut parts = line.split(' ');
    let method = parts.next()?;
    let path = parts.next()?;
    let version = parts.next()?;
    if !version.starts_with("HTTP/") || method.is_empty() || !path.starts_with('/') {
        return None;
    }
    let headers = lines.take_while(|l| !l.is_empty()).map(String::from).collect();
    Some(Request { method: method.into(), path: path.into(), headers })
}

/// Format an HTTP/1.1 response from a content agent's reply frame, which is
/// `"<status>\n<content-type>\n<body…>"` (status e.g. `200 OK`). This is the
/// only place HTTP framing is produced.
pub fn format_response(reply: &[u8]) -> Vec<u8> {
    // Split off the two header lines (status, content-type); the rest is body.
    let mut nl = reply.splitn(3, |&b| b == b'\n');
    let status = nl.next().unwrap_or(b"200 OK");
    let ctype = nl.next().unwrap_or(b"application/octet-stream");
    let body = nl.next().unwrap_or(b"");
    let head = alloc::format!(
        "HTTP/1.1 {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        String::from_utf8_lossy(status),
        String::from_utf8_lossy(ctype),
        body.len()
    );
    let mut out = head.into_bytes();
    out.extend_from_slice(body);
    out
}

extern "C" fn http_serve(_arg: u64) {
    let (Some(from_net), Some(to_net), Some(to_doc), Some(from_doc)) =
        (pipeline::net_to_http(), pipeline::http_to_net(), pipeline::http_to_doc(), pipeline::doc_to_http())
    else {
        crate::ktrace::log("service.http", "pipeline channels not wired");
        return;
    };
    loop {
        if let Ok(Some(raw)) = crate::channel::try_recv_dgram(from_net) {
            // Parse the request; forward "METHOD path" to the content agent.
            let (method, path) = match parse_request(&raw) {
                Some(r) => (r.method, r.path),
                None => (String::from("GET"), String::from("/")),
            };
            crate::ktrace::log_fmt(format_args!("service.http: {method} {path} -> doc"));
            let req = alloc::format!("{method} {path}");
            let d = crate::arch::now_ms() + pipeline::STAGE_DEADLINE_MS;
            pipeline::send_frame(to_doc, req.as_bytes(), d);
            // Await the content agent's body, then format + return the response.
            if let Some(reply) = pipeline::recv_deadline(from_doc, crate::arch::now_ms() + pipeline::STAGE_DEADLINE_MS) {
                let resp = format_response(&reply);
                pipeline::send_frame(to_net, &resp, crate::arch::now_ms() + pipeline::STAGE_DEADLINE_MS);
            }
        }
        crate::shell::upkeep();
        crate::sched::yield_now();
    }
}

/// The HTTP protocol service. Native; started as part of the web pipeline.
pub static HTTP_STAGE: ServiceSpec = ServiceSpec { name: "http", entry: http_serve, autostart: false, caps: &[] };

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn parses_request_line_and_headers() {
        let r = parse_request(b"GET /docs HTTP/1.1\r\nHost: chitti\r\nAccept: */*\r\n\r\n").unwrap();
        assert_eq!(r.method, "GET");
        assert_eq!(r.path, "/docs");
        assert_eq!(r.headers, alloc::vec![String::from("Host: chitti"), String::from("Accept: */*")]);
    }

    #[test_case]
    fn rejects_malformed_request_lines() {
        assert!(parse_request(b"garbage").is_none());
        assert!(parse_request(b"GET /docs FTP/1\r\n").is_none()); // not HTTP
        assert!(parse_request(b"GET docs HTTP/1.1\r\n").is_none()); // path not absolute
    }

    #[test_case]
    fn formats_a_response_from_a_content_reply() {
        let resp = format_response(b"200 OK\ntext/html\n<h1>hi</h1>");
        let s = String::from_utf8(resp).unwrap();
        assert!(s.starts_with("HTTP/1.1 200 OK\r\n"), "status line: {s}");
        assert!(s.contains("Content-Type: text/html\r\n"));
        assert!(s.contains("Content-Length: 11\r\n"));
        assert!(s.ends_with("\r\n\r\n<h1>hi</h1>"));
    }

    #[test_case]
    fn formats_a_404_reply() {
        let resp = format_response(b"404 Not Found\ntext/html\nnope");
        let s = String::from_utf8(resp).unwrap();
        assert!(s.starts_with("HTTP/1.1 404 Not Found\r\n"));
        assert!(s.contains("Content-Length: 4\r\n"));
    }
}
