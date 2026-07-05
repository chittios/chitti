//! Minimal **HTTP/1.1 client** over the smoltcp TCP stack — what lets the
//! shell agent call *hosted* models (a LAN llama.cpp / Ollama / vLLM server's
//! OpenAI-style endpoint) and gives the `/http` command + `http` tool their
//! transport. Plain `http://` only: there is no in-kernel TLS, so point it at
//! a host-local or LAN endpoint (the common self-hosted-model case), not the
//! open internet.
//!
//! One request = one TCP socket, `Connection: close` — the response ends when
//! the peer closes (with `Content-Length` / chunked handled when present), so
//! no keep-alive state machine. Cooperative: every wait loop pumps
//! `shell::upkeep()` so the UI clock/mouse/net stay alive while a remote model
//! spends tens of seconds generating.

use super::{resolve, NET};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use smoltcp::socket::tcp;
use smoltcp::wire::{IpAddress, Ipv4Address};

/// A parsed HTTP response: status code + (de-chunked) body bytes.
pub struct Response {
    pub status: u16,
    pub body: Vec<u8>,
}

impl Response {
    /// The body as UTF-8 text (lossy).
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).to_string()
    }
}

/// Split `http://host[:port]/path` into `(host, port, path)`. Rejects
/// `https://` with a pointed error (no in-kernel TLS).
fn parse_url(url: &str) -> Result<(String, u16, String), String> {
    if url.starts_with("https://") {
        return Err("https is not supported (no in-kernel TLS) — use a plain http:// LAN endpoint".into());
    }
    let rest = url.strip_prefix("http://").ok_or("URL must start with http://")?;
    let (hostport, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let (host, port) = match hostport.rsplit_once(':') {
        Some((h, p)) => (h, p.parse::<u16>().map_err(|_| "bad port")?),
        None => (hostport, 80),
    };
    if host.is_empty() {
        return Err("empty host".into());
    }
    Ok((host.to_string(), port, path.to_string()))
}

/// `host` as an IPv4 literal, or resolved via DNS.
fn host_ip(host: &str) -> Result<Ipv4Address, String> {
    let mut parts = [0u8; 4];
    let mut n = 0;
    for (i, seg) in host.split('.').enumerate() {
        if i >= 4 {
            n = 0;
            break;
        }
        match seg.parse::<u8>() {
            Ok(v) => {
                parts[i] = v;
                n = i + 1;
            }
            Err(_) => {
                n = 0;
                break;
            }
        }
    }
    if n == 4 {
        return Ok(Ipv4Address::new(parts[0], parts[1], parts[2], parts[3]));
    }
    resolve(host, 5_000).map_err(|e| format!("DNS {host}: {e}"))
}

/// Issue one HTTP request and collect the full response. `timeout_ms` bounds
/// the whole exchange (a hosted LLM can legitimately take a minute to answer,
/// so callers pass a generous budget).
pub fn request(method: &str, url: &str, headers: &[(&str, &str)], body: &[u8], timeout_ms: u64) -> Result<Response, String> {
    let (host, port, path) = parse_url(url)?;
    let ip = host_ip(&host)?;
    let deadline = crate::arch::now_ms() + timeout_ms;

    // Build the request bytes up front (Connection: close = read-to-EOF).
    let mut req = format!("{method} {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\nAccept: */*\r\n");
    for (k, v) in headers {
        req.push_str(&format!("{k}: {v}\r\n"));
    }
    if !body.is_empty() {
        req.push_str(&format!("Content-Length: {}\r\n", body.len()));
    }
    req.push_str("\r\n");
    let mut wire = req.into_bytes();
    wire.extend_from_slice(body);

    // A dedicated TCP socket for this request. 64 KiB rx: a chat completion
    // JSON is a few KB, but generous buffers keep the window open.
    let handle = NET.with(|n| {
        let s = n.as_mut().ok_or("no network interface (try /network dhcp)")?;
        if s.ip.is_none() {
            return Err("no IPv4 address (try /network dhcp)");
        }
        let sock = tcp::Socket::new(
            tcp::SocketBuffer::new(vec![0u8; 64 * 1024]),
            tcp::SocketBuffer::new(vec![0u8; 16 * 1024]),
        );
        let h = s.sockets.add(sock);
        // Ephemeral local port from the boot clock (fine for one-shot sockets).
        let local = 49152 + (crate::arch::now_ms() % 16000) as u16;
        let cx = s.iface.context();
        s.sockets
            .get_mut::<tcp::Socket>(h)
            .connect(cx, (IpAddress::Ipv4(ip), port), local)
            .map_err(|_| "TCP connect failed to start")?;
        Ok(h)
    })
    .map_err(|e: &str| e.to_string())?;

    // Drive the socket to completion; always remove it from the set on exit.
    let result = drive(handle, &wire, deadline);
    NET.with(|n| {
        if let Some(s) = n.as_mut() {
            s.sockets.remove(handle);
        }
    });
    let raw = result?;
    parse_response(&raw)
}

/// Poll the stack until `wire` is sent and the peer's response is fully read
/// (connection closed). Returns the raw response bytes (status line + headers
/// + body).
fn drive(handle: smoltcp::iface::SocketHandle, wire: &[u8], deadline: u64) -> Result<Vec<u8>, String> {
    let mut sent = 0usize;
    let mut raw: Vec<u8> = Vec::new();
    let mut buf = [0u8; 4096];
    loop {
        if crate::arch::now_ms() >= deadline {
            NET.with(|n| {
                if let Some(s) = n.as_mut() {
                    s.sockets.get_mut::<tcp::Socket>(handle).abort();
                }
            });
            return Err("HTTP timeout".into());
        }
        super::poll();
        let done = NET.with(|n| {
            let s = n.as_mut().ok_or("network went down")?;
            let sock = s.sockets.get_mut::<tcp::Socket>(handle);
            if sent < wire.len() && sock.may_send() {
                if let Ok(k) = sock.send_slice(&wire[sent..]) {
                    sent += k;
                }
            }
            while sock.may_recv() {
                match sock.recv_slice(&mut buf) {
                    Ok(0) => break,
                    Ok(k) => raw.extend_from_slice(&buf[..k]),
                    Err(_) => break,
                }
            }
            // Done when the peer has closed and nothing is left to read.
            let closed = matches!(sock.state(), tcp::State::Closed | tcp::State::CloseWait | tcp::State::TimeWait)
                && !sock.may_recv();
            if closed && sent < wire.len() {
                return Err("connection closed before the request was sent");
            }
            Ok(closed)
        })
        .map_err(|e: &str| e.to_string())?;
        if done {
            // smoltcp keeps buffered rx readable in CloseWait; one last drain
            // happened above, so the response is complete.
            return Ok(raw);
        }
        // Keep the UI + rest of the net stack alive while we wait (a hosted
        // model can take a minute to generate) — the standing upkeep rule.
        crate::shell::upkeep();
        crate::sched::yield_now();
    }
}

/// Parse the status line, find the header/body split, and de-chunk if needed.
fn parse_response(raw: &[u8]) -> Result<Response, String> {
    let split = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or("malformed HTTP response (no header terminator)")?;
    let head = core::str::from_utf8(&raw[..split]).map_err(|_| "non-UTF8 headers")?;
    let mut lines = head.split("\r\n");
    let status_line = lines.next().unwrap_or("");
    let status: u16 = status_line.split_whitespace().nth(1).and_then(|s| s.parse().ok()).ok_or("bad status line")?;
    let mut chunked = false;
    let mut content_length: Option<usize> = None;
    for l in lines {
        let Some((k, v)) = l.split_once(':') else { continue };
        let (k, v) = (k.trim().to_ascii_lowercase(), v.trim());
        if k == "transfer-encoding" && v.to_ascii_lowercase().contains("chunked") {
            chunked = true;
        }
        if k == "content-length" {
            content_length = v.parse().ok();
        }
    }
    let body_raw = &raw[split + 4..];
    let body = if chunked {
        dechunk(body_raw)?
    } else {
        let n = content_length.unwrap_or(body_raw.len()).min(body_raw.len());
        body_raw[..n].to_vec()
    };
    Ok(Response { status, body })
}

/// Decode a `Transfer-Encoding: chunked` body.
fn dechunk(mut b: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    loop {
        let nl = b.windows(2).position(|w| w == b"\r\n").ok_or("bad chunk header")?;
        let size_str = core::str::from_utf8(&b[..nl]).map_err(|_| "bad chunk size")?;
        let size = usize::from_str_radix(size_str.trim().split(';').next().unwrap_or("0"), 16).map_err(|_| "bad chunk size")?;
        b = &b[nl + 2..];
        if size == 0 {
            return Ok(out);
        }
        if b.len() < size {
            return Err("truncated chunk".into());
        }
        out.extend_from_slice(&b[..size]);
        b = &b[size..];
        if b.len() >= 2 {
            b = &b[2..]; // trailing CRLF
        }
    }
}

/// Convenience: GET `url`, returning the response.
pub fn get(url: &str, timeout_ms: u64) -> Result<Response, String> {
    request("GET", url, &[], &[], timeout_ms)
}

/// Convenience: POST a JSON body to `url` (plus optional bearer key).
pub fn post_json(url: &str, json: &str, bearer: Option<&str>, timeout_ms: u64) -> Result<Response, String> {
    let auth;
    let mut headers: Vec<(&str, &str)> = vec![("Content-Type", "application/json")];
    if let Some(k) = bearer {
        auth = format!("Bearer {k}");
        headers.push(("Authorization", &auth));
    }
    request("POST", url, &headers, json.as_bytes(), timeout_ms)
}
