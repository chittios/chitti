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

/// A parsed HTTP response: status code, headers, and (de-chunked) body bytes.
pub struct Response {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Response {
    /// The body as UTF-8 text (lossy).
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).to_string()
    }

    /// Case-insensitive header lookup.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(key))
            .map(|(_, v)| v.as_str())
    }
}

/// Max hops when following 3xx `Location` (browsers typically allow ~20; keep tight).
pub const MAX_REDIRECTS: u32 = 10;

/// True for redirect statuses we follow on GET (RFC 9110).
pub fn is_redirect(status: u16) -> bool {
    matches!(status, 301 | 302 | 303 | 307 | 308)
}

/// Resolve a `Location` header against the request URL that produced it.
/// Handles absolute `http(s)://…`, protocol-relative `//host/…`, root- and
/// path-relative targets. Pure — unit-tested.
pub fn resolve_redirect(base_url: &str, location: &str) -> Result<String, String> {
    let loc = location.trim();
    if loc.is_empty() {
        return Err("empty Location".into());
    }
    // Strip fragment on redirects (browsers re-apply client-side).
    let loc = loc.split('#').next().unwrap_or(loc);
    if loc.starts_with("https://") || loc.starts_with("http://") {
        return Ok(loc.to_string());
    }
    let (tls, host, port, path) = parse_url(base_url)?;
    let scheme = if tls { "https" } else { "http" };
    let authority = if (tls && port == 443) || (!tls && port == 80) {
        host.clone()
    } else {
        format!("{host}:{port}")
    };
    if let Some(rest) = loc.strip_prefix("//") {
        return Ok(format!("{scheme}://{rest}"));
    }
    if loc.starts_with('/') {
        return Ok(format!("{scheme}://{authority}{loc}"));
    }
    // Path-relative.
    let dir = match path.rfind('/') {
        Some(i) => &path[..=i],
        None => "/",
    };
    Ok(format!("{scheme}://{authority}{dir}{loc}"))
}

/// Result of [`get_follow`]: final response after redirect hops + the URL that
/// produced it (for relative link resolution in the browser).
pub struct FollowedGet {
    pub response: Response,
    pub final_url: String,
    /// Number of 3xx hops taken (0 = no redirect).
    pub redirects: u32,
}

/// GET with automatic redirect following (301/302/303/307/308).
/// Each hop is a fresh TCP/TLS connection (HTTP/1.1 Connection: close).
pub fn get_follow(url: &str, timeout_ms: u64) -> Result<FollowedGet, String> {
    get_follow_headers(url, &[], timeout_ms)
}

/// [`get_follow`] with caller-supplied request headers (e.g. a browser
/// `User-Agent` + `Cookie`) sent on every redirect hop.
pub fn get_follow_headers(
    url: &str,
    headers: &[(&str, &str)],
    timeout_ms: u64,
) -> Result<FollowedGet, String> {
    let mut current = url.trim().to_string();
    let mut redirects = 0u32;
    loop {
        let resp = request("GET", &current, headers, &[], timeout_ms)?;
        if !is_redirect(resp.status) {
            return Ok(FollowedGet {
                response: resp,
                final_url: current,
                redirects,
            });
        }
        if redirects >= MAX_REDIRECTS {
            return Err(format!(
                "too many redirects ({} hops, last {} {})",
                MAX_REDIRECTS, resp.status, current
            ));
        }
        let loc = resp
            .get("location")
            .ok_or_else(|| format!("HTTP {} without Location header", resp.status))?;
        let next = resolve_redirect(&current, loc)?;
        crate::ktrace::log_fmt(format_args!(
            "http: redirect {} {} → {}",
            resp.status, current, next
        ));
        current = next;
        redirects += 1;
        // Cooperative: long redirect chains (and TLS handshakes) must not freeze UI.
        crate::shell::upkeep();
        if crate::shell::poll_interrupt() {
            return Err("cancelled".into());
        }
    }
}

/// Split `http[s]://host[:port]/path` into `(tls, host, port, path)`. `https`
/// tunnels through [`super::tls`]; `http` is plaintext. Default port follows
/// the scheme (80 / 443).
pub(crate) fn parse_url(url: &str) -> Result<(bool, String, u16, String), String> {
    let (tls, rest, default_port) = if let Some(r) = url.strip_prefix("https://") {
        (true, r, 443u16)
    } else if let Some(r) = url.strip_prefix("http://") {
        (false, r, 80u16)
    } else {
        return Err("URL must start with http:// or https://".into());
    };
    let (hostport, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let (host, port) = match hostport.rsplit_once(':') {
        Some((h, p)) => (h, p.parse::<u16>().map_err(|_| "bad port")?),
        None => (hostport, default_port),
    };
    if host.is_empty() {
        return Err("empty host".into());
    }
    Ok((tls, host.to_string(), port, path.to_string()))
}

/// `host` as an IPv4 literal, or resolved via DNS. `localhost` is the loopback
/// address 127.0.0.1 (no DNS), so in-OS servers are reachable by name.
pub(crate) fn host_ip(host: &str) -> Result<Ipv4Address, String> {
    if host.eq_ignore_ascii_case("localhost") {
        return Ok(Ipv4Address::new(127, 0, 0, 1));
    }
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

/// Response head: status + all headers (curl `-v` prints these; the hosted-
/// model path reads `content-length`/`transfer-encoding`).
pub struct Head {
    pub status: u16,
    pub headers: Vec<(String, String)>,
}

impl Head {
    /// Case-insensitive header lookup.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(key)).map(|(_, v)| v.as_str())
    }
}

/// A live HTTP connection — plaintext over the smoltcp socket, or TLS over it.
/// Both expose the same blocking `read`/`write_all` so the request driver is
/// scheme-agnostic.
enum Conn {
    Plain(super::tls::TcpStream),
    Secure(super::tls::TlsSession),
}

impl Conn {
    fn write_all(&mut self, d: &[u8]) -> Result<(), String> {
        match self {
            Conn::Plain(s) => {
                use embedded_io::Write;
                s.write_all(d).map_err(|_| "send failed".to_string())
            }
            Conn::Secure(t) => t.write_all(d),
        }
    }
    /// Read up to `buf.len()` bytes; `0` = EOF / closed / timed out.
    fn read(&mut self, buf: &mut [u8]) -> usize {
        match self {
            Conn::Plain(s) => {
                use embedded_io::Read;
                s.read(buf).unwrap_or(0)
            }
            Conn::Secure(t) => t.read(buf),
        }
    }
}

/// Issue one HTTP request and **stream** the decoded body: `on_body` is called
/// with each newly-arrived (de-chunked) body slice as it lands, so a caller can
/// print an SSE / chunked response live. The response head is returned once the
/// headers are in. `timeout_ms` bounds the whole exchange. This is the engine
/// under [`request`] (which just buffers) and the `/http --stream` path.
pub fn perform(
    method: &str,
    url: &str,
    headers: &[(&str, &str)],
    body: &[u8],
    timeout_ms: u64,
    on_head: &mut dyn FnMut(&Head),
    on_body: &mut dyn FnMut(&[u8]),
) -> Result<Head, String> {
    let (tls, host, port, path) = parse_url(url)?;
    let ip = host_ip(&host)?;
    let deadline = crate::arch::now_ms() + timeout_ms;

    // Build the request bytes. `Connection: close` so the body ends at EOF when
    // no length/chunking is given; callers may override any header.
    let has = |name: &str| headers.iter().any(|(k, _)| k.eq_ignore_ascii_case(name));
    let mut req = format!("{method} {path} HTTP/1.1\r\nHost: {host}:{port}\r\n");
    if !has("connection") {
        req.push_str("Connection: close\r\n");
    }
    if !has("accept") {
        req.push_str("Accept: */*\r\n");
    }
    if !has("user-agent") {
        // Many hosts (e.g. upload.wikimedia.org) reject requests with no
        // User-Agent — a descriptive default gets a 200 instead of a 403.
        req.push_str(&format!("User-Agent: Chitti-OS/{} (https://github.com/chitti-os)\r\n", crate::VERSION));
    }
    for (k, v) in headers {
        req.push_str(&format!("{k}: {v}\r\n"));
    }
    if !body.is_empty() && !has("content-length") {
        req.push_str(&format!("Content-Length: {}\r\n", body.len()));
    }
    req.push_str("\r\n");
    let mut wire = req.into_bytes();
    wire.extend_from_slice(body);

    let handle = tcp_connect(ip, port, deadline)?;
    let mut conn = if tls {
        Conn::Secure(super::tls::handshake(super::tls::TcpStream::new(handle, deadline), &host)?)
    } else {
        Conn::Plain(super::tls::TcpStream::new(handle, deadline))
    };

    let result = drive_stream(&mut conn, &wire, deadline, on_head, on_body);
    NET.with(|n| {
        if let Some(s) = n.as_mut() {
            s.tcp_set(handle).remove(handle.handle);
        }
    });
    result
}

/// Send `wire`, then read the response, parsing the head once and emitting the
/// decoded body incrementally via `on_body`. Shared by plaintext + TLS.
fn drive_stream(conn: &mut Conn, wire: &[u8], deadline: u64, on_head: &mut dyn FnMut(&Head), on_body: &mut dyn FnMut(&[u8])) -> Result<Head, String> {
    conn.write_all(wire)?;
    let mut raw: Vec<u8> = Vec::new();
    let mut buf = [0u8; 4096];
    let mut head: Option<Head> = None;
    let mut body_at = 0usize; // index in `raw` where the body starts
    let mut chunked = false;
    let mut clen: Option<usize> = None;
    let mut emitted = 0usize; // decoded body bytes already handed to on_body
    loop {
        if crate::arch::now_ms() >= deadline {
            return Err("HTTP timeout".into());
        }
        if crate::shell::poll_interrupt() {
            return Err("cancelled".into());
        }
        let k = conn.read(&mut buf);
        if k == 0 {
            break; // EOF / close
        }
        raw.extend_from_slice(&buf[..k]);
        if head.is_none() {
            if let Some(split) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
                let h = parse_head(&raw[..split])?;
                chunked = h.get("transfer-encoding").map(|v| v.to_ascii_lowercase().contains("chunked")).unwrap_or(false);
                clen = h.get("content-length").and_then(|v| v.trim().parse().ok());
                body_at = split + 4;
                on_head(&h);
                head = Some(h);
            }
        }
        if head.is_some() {
            let region = &raw[body_at..];
            let decoded = if chunked {
                dechunk_partial(region)
            } else {
                let n = clen.unwrap_or(region.len()).min(region.len());
                region[..n].to_vec()
            };
            if decoded.len() > emitted {
                on_body(&decoded[emitted..]);
                emitted = decoded.len();
            }
            if response_complete(&raw) {
                break;
            }
        }
        // Keep the UI + net stack alive during a long/streamed response.
        crate::shell::upkeep();
        crate::sched::yield_now();
    }
    head.ok_or_else(|| "no response head (connection closed early)".to_string())
}

/// Issue one HTTP request and collect the full response (buffered). The
/// convenience path for callers that want the whole body at once.
pub fn request(method: &str, url: &str, headers: &[(&str, &str)], body: &[u8], timeout_ms: u64) -> Result<Response, String> {
    let mut body_buf: Vec<u8> = Vec::new();
    let head = perform(method, url, headers, body, timeout_ms, &mut |_| {}, &mut |chunk| body_buf.extend_from_slice(chunk))?;
    Ok(Response { status: head.status, headers: head.headers, body: body_buf })
}

/// Open a TCP socket to `ip:port` and wait for the connection to establish
/// (bounded by `deadline`). Returns the socket handle (caller removes it).
pub(crate) fn tcp_connect(ip: Ipv4Address, port: u16, deadline: u64) -> Result<super::TcpHandle, String> {
    // 64 KiB rx keeps the window open for a large completion; 16 KiB tx.
    let is_loopback = ip.is_loopback();
    let handle = NET.with(|n| {
        let s = n.as_mut().ok_or("no network interface (try /network dhcp)")?;
        // A loopback destination is always reachable via the loopback interface
        // (127.0.0.1/8), so it needs no DHCP/static address; anything else does.
        if !is_loopback && s.ip.is_none() {
            return Err("no IPv4 address (try /network dhcp)");
        }
        let sock = tcp::Socket::new(
            tcp::SocketBuffer::new(vec![0u8; 64 * 1024]),
            tcp::SocketBuffer::new(vec![0u8; 16 * 1024]),
        );
        // The socket lives in the interface's own set, connected via that
        // interface's context so its source address is chosen from it: 127.0.0.1
        // for loopback (segments loop back to a local listener), the NIC address
        // otherwise. The two sets are polled independently, never cross-dispatched.
        // Ephemeral source port from a monotonically-advancing counter (not the
        // clock): rapid back-to-back connects — e.g. an MCP `/mcp connect`'s
        // initialize→notify→tools/list within a few ms — must not reuse the same
        // port while the prior socket is still closing, which would stall the
        // new connect. Wraps over the 49152..=65535 ephemeral range.
        static EPHEMERAL: core::sync::atomic::AtomicU16 = core::sync::atomic::AtomicU16::new(0);
        let local = 49152 + (EPHEMERAL.fetch_add(1, core::sync::atomic::Ordering::Relaxed) % 16000);
        let h = if is_loopback {
            let h = s.lo_sockets.add(sock);
            let cx = s.lo_iface.context();
            s.lo_sockets.get_mut::<tcp::Socket>(h).connect(cx, (IpAddress::Ipv4(ip), port), local).map_err(|_| "TCP connect failed to start")?;
            h
        } else {
            let h = s.sockets.add(sock);
            let cx = s.iface.context();
            s.sockets.get_mut::<tcp::Socket>(h).connect(cx, (IpAddress::Ipv4(ip), port), local).map_err(|_| "TCP connect failed to start")?;
            h
        };
        Ok(super::TcpHandle { handle: h, loopback: is_loopback })
    })
    .map_err(|e: &str| e.to_string())?;
    // Wait for the handshake so TLS starts on an established socket.
    loop {
        if crate::arch::now_ms() >= deadline || crate::shell::poll_interrupt() {
            let cancelled = crate::arch::now_ms() < deadline;
            NET.with(|n| {
                if let Some(s) = n.as_mut() {
                    s.tcp_set(handle).remove(handle.handle);
                }
            });
            return Err(if cancelled { "cancelled".into() } else { "TCP connect timeout".into() });
        }
        super::poll();
        let st = NET.with(|n| n.as_mut().map(|s| s.tcp_set(handle).get_mut::<tcp::Socket>(handle.handle).state()));
        match st {
            Some(tcp::State::Established) => return Ok(handle),
            Some(tcp::State::Closed) => return Err("connection refused".into()),
            _ => {}
        }
        crate::shell::upkeep();
        crate::sched::yield_now();
    }
}

/// True once `raw` holds a complete HTTP response: header terminator present
/// and either the full Content-Length body, the chunked terminator, or (no
/// length given) we defer to EOF by returning false.
fn response_complete(raw: &[u8]) -> bool {
    let Some(split) = raw.windows(4).position(|w| w == b"\r\n\r\n") else {
        return false;
    };
    let head = match core::str::from_utf8(&raw[..split]) {
        Ok(h) => h,
        Err(_) => return false,
    };
    let body_len = raw.len() - (split + 4);
    for l in head.split("\r\n") {
        if let Some((k, v)) = l.split_once(':') {
            let k = k.trim().to_ascii_lowercase();
            if k == "content-length" {
                if let Ok(n) = v.trim().parse::<usize>() {
                    return body_len >= n;
                }
            }
            if k == "transfer-encoding" && v.to_ascii_lowercase().contains("chunked") {
                return raw.ends_with(b"0\r\n\r\n");
            }
        }
    }
    false
}

/// Parse the response head (status line + headers) from the bytes before the
/// `\r\n\r\n` terminator.
fn parse_head(head_bytes: &[u8]) -> Result<Head, String> {
    let head = core::str::from_utf8(head_bytes).map_err(|_| "non-UTF8 headers")?;
    let mut lines = head.split("\r\n");
    let status_line = lines.next().unwrap_or("");
    let status: u16 = status_line.split_whitespace().nth(1).and_then(|s| s.parse().ok()).ok_or("bad status line")?;
    let mut headers = Vec::new();
    for l in lines {
        if let Some((k, v)) = l.split_once(':') {
            headers.push((k.trim().to_string(), v.trim().to_string()));
        }
    }
    Ok(Head { status, headers })
}

/// Parse a full raw response (head + body) into a [`Response`], de-chunking
/// the body if needed. (Used by tests and any full-buffer caller.)
fn parse_response(raw: &[u8]) -> Result<Response, String> {
    let split = raw.windows(4).position(|w| w == b"\r\n\r\n").ok_or("malformed HTTP response (no header terminator)")?;
    let head = parse_head(&raw[..split])?;
    let chunked = head.get("transfer-encoding").map(|v| v.to_ascii_lowercase().contains("chunked")).unwrap_or(false);
    let clen: Option<usize> = head.get("content-length").and_then(|v| v.trim().parse().ok());
    let body_raw = &raw[split + 4..];
    let body = if chunked {
        dechunk_partial(body_raw)
    } else {
        let n = clen.unwrap_or(body_raw.len()).min(body_raw.len());
        body_raw[..n].to_vec()
    };
    Ok(Response { status: head.status, headers: head.headers, body })
}

/// Decode as much of a `Transfer-Encoding: chunked` body as is complete,
/// stopping at the first incomplete chunk (so it works incrementally while a
/// chunked/SSE response is still arriving). Never errors — a partial chunk
/// header or a short chunk just means "nothing more decodable yet".
fn dechunk_partial(mut b: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let Some(nl) = b.windows(2).position(|w| w == b"\r\n") else { return out };
        let Ok(size_str) = core::str::from_utf8(&b[..nl]) else { return out };
        let Ok(size) = usize::from_str_radix(size_str.trim().split(';').next().unwrap_or("0"), 16) else { return out };
        let rest = &b[nl + 2..];
        if size == 0 {
            return out; // terminating chunk
        }
        if rest.len() < size {
            return out; // chunk body not fully arrived yet
        }
        out.extend_from_slice(&rest[..size]);
        b = &rest[size..];
        if b.len() >= 2 {
            b = &b[2..]; // trailing CRLF
        } else {
            return out;
        }
    }
}

/// Convenience: GET `url`, returning the response (**no** redirect follow).
/// Prefer [`get_follow`] for browser / human navigation.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn parse_url_schemes_ports_paths() {
        // http default port 80, no path → "/".
        let (tls, host, port, path) = parse_url("http://192.168.1.20:8080/v1/models").unwrap();
        assert!(!tls);
        assert_eq!(host, "192.168.1.20");
        assert_eq!(port, 8080);
        assert_eq!(path, "/v1/models");
        // https defaults to 443, tls flag set.
        let (tls, host, port, path) = parse_url("https://example.com").unwrap();
        assert!(tls);
        assert_eq!((host.as_str(), port, path.as_str()), ("example.com", 443, "/"));
        // plain http default port.
        let (_, _, port, _) = parse_url("http://host/x").unwrap();
        assert_eq!(port, 80);
        // A bad scheme is rejected.
        assert!(parse_url("ftp://x").is_err());
        assert!(parse_url("http://").is_err());
    }

    #[test_case]
    fn host_ip_parses_ipv4_literals() {
        // Dotted-quad literals bypass DNS (this is the /ping + /http fast path).
        assert_eq!(host_ip("10.0.2.2").unwrap(), Ipv4Address::new(10, 0, 2, 2));
        assert_eq!(host_ip("192.168.1.255").unwrap(), Ipv4Address::new(192, 168, 1, 255));
        // A non-literal falls through to DNS, which errors with no interface up
        // (rather than misparsing) — we just assert it doesn't panic/parse.
        assert!(host_ip("not.an.ip.literal").is_err());
    }

    #[test_case]
    fn host_ip_maps_localhost_to_loopback() {
        // `localhost` resolves to the loopback address without DNS, and is
        // recognised as loopback so connects route through the lo interface.
        let lo = host_ip("localhost").unwrap();
        assert_eq!(lo, Ipv4Address::new(127, 0, 0, 1));
        assert!(lo.is_loopback());
        assert!(host_ip("LOCALHOST").unwrap().is_loopback());
        // A dotted-quad in 127/8 is also loopback (routes through lo, no NIC IP).
        assert!(host_ip("127.0.0.1").unwrap().is_loopback());
    }

    #[test_case]
    fn response_complete_content_length() {
        // Headers present, body shorter than Content-Length → not complete.
        assert!(!response_complete(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nab"));
        // Full body → complete.
        assert!(response_complete(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nabcde"));
        // No header terminator yet → not complete.
        assert!(!response_complete(b"HTTP/1.1 200 OK\r\nContent-Len"));
    }

    #[test_case]
    fn response_complete_chunked_terminator() {
        assert!(!response_complete(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n"));
        assert!(response_complete(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n0\r\n\r\n"));
    }

    #[test_case]
    fn parse_response_status_and_body() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 7\r\n\r\n{\"a\":1}";
        let r = parse_response(raw).unwrap();
        assert_eq!(r.status, 200);
        assert_eq!(r.text(), "{\"a\":1}");
        // A non-200 status is still parsed (the caller decides what to do).
        let r = parse_response(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n").unwrap();
        assert_eq!(r.status, 404);
        assert!(r.body.is_empty());
    }

    #[test_case]
    fn parse_response_dechunks() {
        // Two chunks (5 + 6 bytes) then the 0-terminator → "hello world".
        let raw = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n";
        let r = parse_response(raw).unwrap();
        assert_eq!(r.text(), "hello world");
    }

    #[test_case]
    fn parse_response_rejects_headerless() {
        assert!(parse_response(b"garbage with no terminator").is_err());
    }

    #[test_case]
    fn dechunk_partial_incremental() {
        // Complete chunks decode; a trailing incomplete chunk yields only the
        // complete prefix (this is what lets a streamed response render live).
        assert_eq!(dechunk_partial(b"5\r\nhello\r\n0\r\n\r\n"), b"hello");
        assert_eq!(dechunk_partial(b"5\r\nhello\r\n6\r\n wor"), b"hello"); // 2nd chunk short
        assert_eq!(dechunk_partial(b"3\r\n"), b""); // header only, no body yet
        assert_eq!(dechunk_partial(b""), b"");
    }

    #[test_case]
    fn parse_head_headers_captured() {
        let h = parse_head(b"HTTP/1.1 204 No Content\r\nX-Foo: bar\r\nContent-Length: 0").unwrap();
        assert_eq!(h.status, 204);
        assert_eq!(h.get("x-foo"), Some("bar")); // case-insensitive lookup
        assert_eq!(h.get("content-length"), Some("0"));
        assert_eq!(h.get("missing"), None);
    }

    #[test_case]
    fn is_redirect_statuses() {
        assert!(is_redirect(301));
        assert!(is_redirect(302));
        assert!(is_redirect(303));
        assert!(is_redirect(307));
        assert!(is_redirect(308));
        assert!(!is_redirect(200));
        assert!(!is_redirect(304));
        assert!(!is_redirect(404));
    }

    #[test_case]
    fn resolve_redirect_absolute_and_relative() {
        assert_eq!(
            resolve_redirect("https://ex.com/a/b", "https://other/x").unwrap(),
            "https://other/x"
        );
        assert_eq!(
            resolve_redirect("https://ex.com/a/b", "/root").unwrap(),
            "https://ex.com/root"
        );
        assert_eq!(
            resolve_redirect("https://ex.com/a/b", "c").unwrap(),
            "https://ex.com/a/c"
        );
        assert_eq!(
            resolve_redirect("http://ex.com/", "//cdn.ex/p").unwrap(),
            "http://cdn.ex/p"
        );
        // google.com style: https → www
        assert_eq!(
            resolve_redirect("https://google.com/", "https://www.google.com/").unwrap(),
            "https://www.google.com/"
        );
        assert!(resolve_redirect("https://ex.com/", "").is_err());
    }

    #[test_case]
    fn response_get_location_case_insensitive() {
        let r = Response {
            status: 301,
            headers: alloc::vec![("Location".into(), "https://www.example.com/".into())],
            body: Vec::new(),
        };
        assert_eq!(r.get("location"), Some("https://www.example.com/"));
        assert_eq!(r.get("LOCATION"), Some("https://www.example.com/"));
    }
}
