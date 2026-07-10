//! Minimal **WebSocket client** (RFC 6455) over the smoltcp TCP stack — the
//! `/ws` command's transport, and the streaming counterpart to [`super::http`].
//! Does the HTTP `Upgrade` handshake (verifying the server's
//! `Sec-WebSocket-Accept`), then sends **masked** client frames and decodes
//! server frames (text / binary / ping / pong / close), answering pings
//! automatically.
//!
//! Supports `ws://` (plaintext) and `wss://` (over the in-kernel TLS 1.3 from
//! [`super::tls`]). For `wss://` the TLS session reads with a **rolling**
//! deadline (a shared atomic bumped before each read) so a long-lived session
//! never hits a fixed expiry, and idle polls skip the TLS read entirely when
//! no TCP bytes are pending — keeping an interactive `/ws` responsive to the
//! keyboard. As with the HTTP client, server certificates are not verified.

use super::NET;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use smoltcp::socket::tcp;

/// The magic GUID appended to the client key before hashing (RFC 6455 §1.3).
const WS_GUID: &str = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

/// A decoded WebSocket message handed to the caller.
#[derive(Debug, PartialEq, Eq)]
pub enum Msg {
    Text(String),
    Binary(Vec<u8>),
    /// The peer sent a Close frame; the session is done.
    Closed,
}

// --- base64 (standard alphabet) -----------------------------------------

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Standard base64 encode (with `=` padding).
pub fn base64_encode(input: &[u8]) -> String {
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(B64[(n >> 18 & 0x3f) as usize] as char);
        out.push(B64[(n >> 12 & 0x3f) as usize] as char);
        out.push(if chunk.len() > 1 { B64[(n >> 6 & 0x3f) as usize] as char } else { '=' });
        out.push(if chunk.len() > 2 { B64[(n & 0x3f) as usize] as char } else { '=' });
    }
    out
}

/// Standard base64 decode; ignores ASCII whitespace, tolerates missing
/// padding. `None` on any non-alphabet byte. Used to unwrap providers that
/// return audio as base64 in a JSON field (Inworld/Sarvam TTS).
pub fn base64_decode(s: &str) -> Option<Vec<u8>> {
    fn val(c: u8) -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => (c - b'A') as u32,
            b'a'..=b'z' => (c - b'a' + 26) as u32,
            b'0'..=b'9' => (c - b'0' + 52) as u32,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        })
    }
    let mut out = Vec::with_capacity(s.len() / 4 * 3 + 3);
    let (mut acc, mut nbits) = (0u32, 0u32);
    for &c in s.as_bytes() {
        if c == b'=' || c.is_ascii_whitespace() {
            continue;
        }
        acc = (acc << 6) | val(c)?;
        nbits += 6;
        if nbits >= 8 {
            nbits -= 8;
            out.push((acc >> nbits) as u8);
        }
    }
    Some(out)
}

// --- SHA-1 (for Sec-WebSocket-Accept) -----------------------------------

/// SHA-1 digest of `msg` (RFC 3174). Small and only used for the WebSocket
/// handshake's `Sec-WebSocket-Accept`; not a general crypto primitive.
pub fn sha1(msg: &[u8]) -> [u8; 20] {
    let mut h: [u32; 5] = [0x6745_2301, 0xEFCD_AB89, 0x98BA_DCFE, 0x1032_5476, 0xC3D2_E1F0];
    // Pad: 0x80, zeros, then the 64-bit big-endian bit length.
    let mut data = msg.to_vec();
    let bitlen = (msg.len() as u64) * 8;
    data.push(0x80);
    while data.len() % 64 != 56 {
        data.push(0);
    }
    data.extend_from_slice(&bitlen.to_be_bytes());
    for block in data.chunks(64) {
        let mut w = [0u32; 80];
        for (i, wi) in w.iter_mut().enumerate().take(16) {
            *wi = u32::from_be_bytes([block[i * 4], block[i * 4 + 1], block[i * 4 + 2], block[i * 4 + 3]]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }
        let (mut a, mut b, mut c, mut d, mut e) = (h[0], h[1], h[2], h[3], h[4]);
        for (i, &wi) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | ((!b) & d), 0x5A82_7999),
                20..=39 => (b ^ c ^ d, 0x6ED9_EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1B_BCDC),
                _ => (b ^ c ^ d, 0xCA62_C1D6),
            };
            let tmp = a.rotate_left(5).wrapping_add(f).wrapping_add(e).wrapping_add(k).wrapping_add(wi);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = tmp;
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
    }
    let mut out = [0u8; 20];
    for (i, hi) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&hi.to_be_bytes());
    }
    out
}

/// The expected `Sec-WebSocket-Accept` value for a client `key`.
fn accept_for(key: &str) -> String {
    let mut buf = key.as_bytes().to_vec();
    buf.extend_from_slice(WS_GUID.as_bytes());
    base64_encode(&sha1(&buf))
}

// --- frame codec ---------------------------------------------------------

const OP_TEXT: u8 = 0x1;
const OP_BINARY: u8 = 0x2;
const OP_CLOSE: u8 = 0x8;
const OP_PING: u8 = 0x9;
const OP_PONG: u8 = 0xA;

/// Encode a client→server frame: FIN set, `opcode`, and a 4-byte mask applied
/// to `payload` (clients MUST mask, RFC 6455 §5.3).
pub fn encode_frame(opcode: u8, payload: &[u8], mask: [u8; 4]) -> Vec<u8> {
    let mut f = Vec::with_capacity(payload.len() + 14);
    f.push(0x80 | (opcode & 0x0f)); // FIN + opcode
    let len = payload.len();
    if len < 126 {
        f.push(0x80 | len as u8); // MASK bit + 7-bit len
    } else if len <= 0xffff {
        f.push(0x80 | 126);
        f.extend_from_slice(&(len as u16).to_be_bytes());
    } else {
        f.push(0x80 | 127);
        f.extend_from_slice(&(len as u64).to_be_bytes());
    }
    f.extend_from_slice(&mask);
    for (i, &b) in payload.iter().enumerate() {
        f.push(b ^ mask[i & 3]);
    }
    f
}

/// A decoded frame: opcode + unmasked payload.
struct Frame {
    opcode: u8,
    payload: Vec<u8>,
    /// Total bytes consumed from the input (header + payload).
    consumed: usize,
}

/// Try to decode one server→client frame from `buf`. `None` if a full frame
/// hasn't arrived yet. Server frames are unmasked (a masked server frame is
/// a protocol error, decoded here anyway for leniency).
fn decode_frame(buf: &[u8]) -> Option<Frame> {
    if buf.len() < 2 {
        return None;
    }
    let opcode = buf[0] & 0x0f;
    let masked = buf[1] & 0x80 != 0;
    let len7 = (buf[1] & 0x7f) as usize;
    let mut off = 2;
    let len = if len7 < 126 {
        len7
    } else if len7 == 126 {
        if buf.len() < off + 2 {
            return None;
        }
        let l = u16::from_be_bytes([buf[off], buf[off + 1]]) as usize;
        off += 2;
        l
    } else {
        if buf.len() < off + 8 {
            return None;
        }
        let mut l = 0u64;
        for i in 0..8 {
            l = (l << 8) | buf[off + i] as u64;
        }
        off += 8;
        l as usize
    };
    let mask = if masked {
        if buf.len() < off + 4 {
            return None;
        }
        let m = [buf[off], buf[off + 1], buf[off + 2], buf[off + 3]];
        off += 4;
        Some(m)
    } else {
        None
    };
    if buf.len() < off + len {
        return None; // payload not fully arrived
    }
    let mut payload = buf[off..off + len].to_vec();
    if let Some(m) = mask {
        for (i, b) in payload.iter_mut().enumerate() {
            *b ^= m[i & 3];
        }
    }
    Some(Frame { opcode, payload, consumed: off + len })
}

// --- transport (plaintext smoltcp socket, or TLS over it for wss://) --------

/// How WebSocket bytes move: plaintext straight over the smoltcp socket, or
/// through a TLS session (`wss://`). Both cases keep the raw socket `handle`
/// so an idle poll can cheaply check for pending bytes and stay
/// keyboard-responsive.
enum Transport {
    Plain,
    /// TLS session + a `more` flag (embedded-tls may have decrypted data
    /// buffered beyond one `read`, so keep reading it even when the TCP socket
    /// shows no new bytes).
    Secure { tls: super::tls::TlsSession, more: bool },
}

/// A connected WebSocket. Sends masked frames and decodes server frames.
/// `recv` polls with a short per-call timeout so an interactive `/ws` can also
/// watch the keyboard; for `wss://` the TLS session's read deadline is a
/// rolling atomic (`roll`) bumped before each read.
pub struct WebSocket {
    handle: super::TcpHandle,
    transport: Transport,
    /// Rolling read deadline for the TLS transport (unused for plaintext).
    roll: Option<&'static core::sync::atomic::AtomicU64>,
    rx: Vec<u8>, // bytes received but not yet forming a complete frame
    closed: bool,
}

fn now() -> u64 {
    crate::arch::now_ms()
}

/// A pseudo-random 4-byte value (frame mask / handshake key material) from the
/// hardware RNG + cycle counter — unpredictable enough for masking (which is
/// anti-cache-poisoning, not secrecy).
fn rand4() -> [u8; 4] {
    let r = crate::arch::hw_rand() ^ crate::arch::cycle_count().rotate_left(13);
    [(r >> 24) as u8, (r >> 16) as u8, (r >> 8) as u8, r as u8]
}

impl WebSocket {
    /// Connect to `ws://` (plaintext) or `wss://` (TLS) and complete the
    /// Upgrade handshake.
    pub fn connect(url: &str) -> Result<WebSocket, String> {
        let (secure, rest, default_port) = if let Some(r) = url.strip_prefix("wss://") {
            (true, r, 443u16)
        } else if let Some(r) = url.strip_prefix("ws://") {
            (false, r, 80u16)
        } else {
            return Err("URL must start with ws:// or wss://".into());
        };
        let (hostport, path) = match rest.find('/') {
            Some(i) => (&rest[..i], &rest[i..]),
            None => (rest, "/"),
        };
        let (host, port) = match hostport.rsplit_once(':') {
            Some((h, p)) => (h.to_string(), p.parse::<u16>().map_err(|_| "bad port")?),
            None => (hostport.to_string(), default_port),
        };
        let ip = super::http::host_ip(&host)?;
        let deadline = now() + 15_000;
        let handle = super::http::tcp_connect(ip, port, deadline)?;

        // Build the transport: for wss, do the TLS handshake over a stream with
        // a *rolling* deadline (a leaked atomic we bump before each read), so a
        // long-lived session never hits a fixed expiry.
        let (transport, roll) = if secure {
            let roll: &'static core::sync::atomic::AtomicU64 =
                alloc::boxed::Box::leak(alloc::boxed::Box::new(core::sync::atomic::AtomicU64::new(deadline)));
            let stream = super::tls::TcpStream::with_rolling(handle, roll);
            match super::tls::handshake(stream, &host) {
                Ok(tls) => (Transport::Secure { tls, more: false }, Some(roll)),
                Err(e) => {
                    NET.with(|n| {
                        if let Some(s) = n.as_mut() {
                            s.tcp_set(handle).remove(handle.handle);
                        }
                    });
                    return Err(e);
                }
            }
        } else {
            (Transport::Plain, None)
        };
        let mut ws = WebSocket { handle, transport, roll, rx: Vec::new(), closed: false };

        // Handshake: GET with the Upgrade headers + a random 16-byte key.
        let mut keybytes = [0u8; 16];
        keybytes[..4].copy_from_slice(&rand4());
        keybytes[4..8].copy_from_slice(&rand4());
        keybytes[8..12].copy_from_slice(&rand4());
        keybytes[12..].copy_from_slice(&rand4());
        let key = base64_encode(&keybytes);
        let req = alloc::format!(
            "GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\
             Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
        );
        if let Err(e) = ws.send_raw(req.as_bytes(), deadline) {
            ws.abort();
            return Err(e);
        }
        // Read the handshake response head.
        let mut raw: Vec<u8> = Vec::new();
        loop {
            if now() >= deadline {
                ws.abort();
                return Err("WebSocket handshake timeout".into());
            }
            let got = ws.recv_raw(deadline.min(now() + 300));
            match got {
                Ok(chunk) if !chunk.is_empty() => raw.extend_from_slice(&chunk),
                Ok(_) => {}
                Err(e) => {
                    ws.abort();
                    return Err(e);
                }
            }
            if let Some(split) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
                let head = core::str::from_utf8(&raw[..split]).unwrap_or("");
                let status_ok = head.lines().next().map(|l| l.contains(" 101")).unwrap_or(false);
                if !status_ok {
                    ws.abort();
                    return Err(alloc::format!("handshake failed (expected 101): {}", head.lines().next().unwrap_or("")));
                }
                // Verify Sec-WebSocket-Accept (proves a real WebSocket peer).
                let want = accept_for(&key);
                let got_accept = head
                    .lines()
                    .find_map(|l| l.split_once(':').filter(|(k, _)| k.eq_ignore_ascii_case("sec-websocket-accept")).map(|(_, v)| v.trim()));
                if got_accept != Some(want.as_str()) {
                    ws.abort();
                    return Err("bad Sec-WebSocket-Accept (not a valid WebSocket server)".into());
                }
                // Any bytes past the header terminator are the first frame(s).
                ws.rx = raw[split + 4..].to_vec();
                crate::ktrace::log_fmt(format_args!("ws: connected to {host}:{port}{path}"));
                return Ok(ws);
            }
            crate::shell::upkeep();
            crate::sched::yield_now();
        }
    }

    /// Send a text message (a single masked TEXT frame).
    pub fn send_text(&mut self, msg: &str) -> Result<(), String> {
        let frame = encode_frame(OP_TEXT, msg.as_bytes(), rand4());
        self.send_raw(&frame, now() + 5_000)
    }

    /// Receive the next message, waiting up to `timeout_ms`. `Ok(None)` means
    /// nothing arrived in that window (poll again); auto-replies to pings.
    pub fn recv(&mut self, timeout_ms: u64) -> Result<Option<Msg>, String> {
        if self.closed {
            return Ok(Some(Msg::Closed));
        }
        let deadline = now() + timeout_ms;
        loop {
            // Decode any complete frame already buffered.
            if let Some(f) = decode_frame(&self.rx) {
                self.rx.drain(..f.consumed);
                match f.opcode {
                    OP_TEXT => return Ok(Some(Msg::Text(String::from_utf8_lossy(&f.payload).to_string()))),
                    OP_BINARY => return Ok(Some(Msg::Binary(f.payload))),
                    OP_PING => {
                        let pong = encode_frame(OP_PONG, &f.payload, rand4());
                        let _ = self.send_raw(&pong, now() + 2_000);
                        continue;
                    }
                    OP_PONG => continue,
                    OP_CLOSE => {
                        self.closed = true;
                        return Ok(Some(Msg::Closed));
                    }
                    _ => continue,
                }
            }
            if now() >= deadline {
                return Ok(None);
            }
            let chunk = self.recv_raw(deadline)?;
            if chunk.is_empty() {
                // No data this poll; keep the UI alive and retry until deadline.
                crate::shell::upkeep();
                crate::sched::yield_now();
            } else {
                self.rx.extend_from_slice(&chunk);
            }
        }
    }

    /// Send a Close frame and drop the socket.
    pub fn close(&mut self) {
        if !self.closed {
            let frame = encode_frame(OP_CLOSE, &[], rand4());
            let _ = self.send_raw(&frame, now() + 1_000);
            self.closed = true;
        }
        self.abort();
    }

    // --- transport I/O (plaintext socket, or TLS over it) ---
    fn send_raw(&mut self, data: &[u8], deadline: u64) -> Result<(), String> {
        let roll = self.roll;
        match &mut self.transport {
            Transport::Secure { tls, .. } => {
                if let Some(r) = roll {
                    r.store(now() + 3_000, core::sync::atomic::Ordering::Relaxed);
                }
                tls.write_all(data)
            }
            Transport::Plain => {
                let mut sent = 0;
                while sent < data.len() {
                    if now() >= deadline {
                        return Err("WebSocket send timeout".into());
                    }
                    super::poll();
                    let n = NET.with(|n| {
                        let s = n.as_mut()?;
                        let sock = s.tcp_set(self.handle).get_mut::<tcp::Socket>(self.handle.handle);
                        if sock.can_send() {
                            sock.send_slice(&data[sent..]).ok()
                        } else {
                            Some(0)
                        }
                    });
                    match n {
                        Some(k) => sent += k,
                        None => return Err("network down".into()),
                    }
                    super::poll();
                    crate::sched::yield_now();
                }
                Ok(())
            }
        }
    }

    /// Read whatever is available now. Returns an empty vec if nothing is
    /// pending (so the caller can poll again and check the keyboard). For TLS,
    /// only touches the (blocking) `tls.read` when TCP bytes are pending or the
    /// session has more buffered decrypted data.
    fn recv_raw(&mut self, _deadline: u64) -> Result<Vec<u8>, String> {
        super::poll();
        let roll = self.roll;
        let handle = self.handle;
        let mut mark_closed = false;
        let out = match &mut self.transport {
            Transport::Secure { tls, more } => {
                let has_tcp = NET.with(|n| n.as_mut().map(|s| s.tcp_set(handle).get_mut::<tcp::Socket>(handle.handle).can_recv()).unwrap_or(false));
                if !has_tcp && !*more {
                    // Idle: detect a peer close so recv() can report it.
                    let closed = NET.with(|n| {
                        n.as_mut().map(|s| matches!(s.tcp_set(handle).get_mut::<tcp::Socket>(handle.handle).state(), tcp::State::CloseWait | tcp::State::Closed | tcp::State::TimeWait)).unwrap_or(true)
                    });
                    mark_closed = closed;
                    Ok(Vec::new())
                } else {
                    if let Some(r) = roll {
                        r.store(now() + 3_000, core::sync::atomic::Ordering::Relaxed);
                    }
                    let mut buf = [0u8; 4096];
                    let k = tls.read(&mut buf);
                    if k == 0 {
                        mark_closed = true;
                    }
                    *more = k == buf.len(); // a full buffer may mean more is buffered
                    Ok(buf[..k].to_vec())
                }
            }
            Transport::Plain => NET.with(|n| {
                let s = n.as_mut().ok_or("network down")?;
                let sock = s.tcp_set(handle).get_mut::<tcp::Socket>(handle.handle);
                let mut out = Vec::new();
                let mut buf = [0u8; 2048];
                while sock.can_recv() {
                    match sock.recv_slice(&mut buf) {
                        Ok(0) => break,
                        Ok(k) => out.extend_from_slice(&buf[..k]),
                        Err(_) => break,
                    }
                }
                if out.is_empty() && matches!(sock.state(), tcp::State::CloseWait | tcp::State::Closed | tcp::State::TimeWait) {
                    mark_closed = true;
                }
                Ok(out)
            }),
        };
        if mark_closed {
            self.closed = true;
        }
        out
    }

    fn abort(&mut self) {
        NET.with(|n| {
            if let Some(s) = n.as_mut() {
                s.tcp_set(self.handle).get_mut::<tcp::Socket>(self.handle.handle).abort();
                s.tcp_set(self.handle).remove(self.handle.handle);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn base64_known_vectors() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test_case]
    fn base64_decode_roundtrip_and_lenient() {
        for s in [b"" as &[u8], b"f", b"fo", b"foo", b"foobar", &[0u8, 255, 1, 254, 128]] {
            assert_eq!(base64_decode(&base64_encode(s)).unwrap(), s, "roundtrip {s:?}");
        }
        // Whitespace ignored; missing padding tolerated.
        assert_eq!(base64_decode("Zm9v\nYmFy").unwrap(), b"foobar");
        assert_eq!(base64_decode("Zg").unwrap(), b"f");
        // Non-alphabet byte rejected.
        assert!(base64_decode("Zg*=").is_none());
    }

    #[test_case]
    fn sha1_known_vectors() {
        // FIPS-180 test vectors.
        assert_eq!(sha1(b"abc").to_vec(), hex("a9993e364706816aba3e25717850c26c9cd0d89d"));
        assert_eq!(sha1(b"").to_vec(), hex("da39a3ee5e6b4b0d3255bfef95601890afd80709"));
    }

    /// The RFC 6455 §1.3 worked example: key "dGhlIHNhbXBsZSBub25jZQ==" →
    /// accept "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=".
    #[test_case]
    fn websocket_accept_rfc_example() {
        assert_eq!(accept_for("dGhlIHNhbXBsZSBub25jZQ=="), "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");
    }

    #[test_case]
    fn frame_roundtrip_masked() {
        // A client TEXT frame is masked; decoding it back (as if a server sent
        // it masked) yields the original payload + opcode.
        let f = encode_frame(OP_TEXT, b"hello", [0x11, 0x22, 0x33, 0x44]);
        assert_eq!(f[0], 0x81); // FIN + text
        assert_eq!(f[1] & 0x80, 0x80); // MASK bit set
        let d = decode_frame(&f).unwrap();
        assert_eq!(d.opcode, OP_TEXT);
        assert_eq!(d.payload, b"hello");
        assert_eq!(d.consumed, f.len());
    }

    #[test_case]
    fn decode_server_text_unmasked() {
        // Server frame: FIN+text, len 2, "hi" (no mask).
        let d = decode_frame(&[0x81, 0x02, b'h', b'i']).unwrap();
        assert_eq!(d.opcode, OP_TEXT);
        assert_eq!(d.payload, b"hi");
        // Incomplete frame → None (waits for more bytes).
        assert!(decode_frame(&[0x81, 0x05, b'h']).is_none());
        assert!(decode_frame(&[0x81]).is_none());
    }

    #[test_case]
    fn frame_extended_length_16bit() {
        // A 200-byte payload uses the 126 + u16 length form.
        let payload = alloc::vec![0x5au8; 200];
        let f = encode_frame(OP_BINARY, &payload, [1, 2, 3, 4]);
        assert_eq!(f[1] & 0x7f, 126);
        let d = decode_frame(&f).unwrap();
        assert_eq!(d.payload.len(), 200);
        assert_eq!(d.opcode, OP_BINARY);
    }

    // Decode a hex string into bytes (test helper).
    fn hex(s: &str) -> Vec<u8> {
        (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
    }
}
