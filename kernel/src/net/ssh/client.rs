//! The SSH client driver — the one part of `net::ssh` that touches the network.
//!
//! Everything it decides is computed by the pure modules beside it; this file is
//! ordering, buffering and I/O. It follows the standing rules for a blocking
//! kernel loop: every wait is bounded, pumps [`crate::shell::upkeep`] so the
//! clock/mouse/net stack keep running, and answers Ctrl+C.
//!
//! The sequence, which is not negotiable:
//!
//! 1. Identification strings, both directions, CR-LF terminated.
//! 2. `KEXINIT` both ways — **kept verbatim**, because the exchange hash covers
//!    the bytes as sent, not a re-serialisation.
//! 3. `KEX_ECDH_INIT`/`REPLY`, then verify the host key's signature over the
//!    exchange hash *before* trusting anything it said.
//! 4. `NEWKEYS`. Only now do the ciphers turn on, and the sequence numbers keep
//!    counting across the switch — they do not reset.
//! 5. `ssh-userauth`, then `publickey`/`password`.
//! 6. Channels.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use super::{auth, channel, cipher, hostkey, kex, wire};
use crate::net::TcpHandle;

/// Our identification string. No CR-LF: it is added on the wire and removed
/// before hashing, and keeping them separate is what stops the classic bug of
/// hashing the terminator.
pub const IDENT: &str = "SSH-2.0-ChittiOS_0.1";

const CONNECT_TIMEOUT_MS: u64 = 15_000;
const HANDSHAKE_TIMEOUT_MS: u64 = 30_000;
/// A single read that finds nothing yields; this bounds the whole wait.
const READ_TIMEOUT_MS: u64 = 60_000;

/// A live SSH connection.
pub struct Client {
    sock: TcpHandle,
    /// Inbound bytes not yet consumed as packets.
    inbuf: Vec<u8>,
    tx: Option<cipher::Direction>,
    rx: Option<cipher::Direction>,
    tx_seq: u32,
    rx_seq: u32,
    block_tx: usize,
    pub session_id: Vec<u8>,
    pub server_ident: String,
    pub host_key: Option<hostkey::PublicKey>,
    next_channel: u32,
}

/// What the caller wants a session channel to do.
pub enum Session<'a> {
    /// Run one command and collect its output.
    Exec(&'a str),
    /// Start the login shell (needs a pty to behave like a terminal).
    Shell,
}

impl Client {
    /// Open a TCP connection and complete the transport handshake.
    ///
    /// Returns with the connection encrypted but **not authenticated** — the
    /// caller then chooses how to authenticate, because that decision (which
    /// key, whether to prompt) is policy rather than protocol.
    pub fn connect(host: &str, port: u16) -> Result<Self, String> {
        let deadline = crate::arch::now_ms() + CONNECT_TIMEOUT_MS;
        let ip = crate::net::resolve_any(host, CONNECT_TIMEOUT_MS)
            .map_err(|e| alloc::format!("cannot resolve {host}: {e}"))?;
        let sock = crate::net::http::tcp_connect_addr(ip, port, deadline)?;
        let mut c = Self {
            sock,
            inbuf: Vec::new(),
            tx: None,
            rx: None,
            tx_seq: 0,
            rx_seq: 0,
            block_tx: 8,
            session_id: Vec::new(),
            server_ident: String::new(),
            host_key: None,
            next_channel: 0,
        };
        c.handshake(host, port)?;
        Ok(c)
    }

    // --- raw socket helpers, all bounded and all pumping upkeep -------------

    fn send_all(&mut self, mut data: &[u8]) -> Result<(), String> {
        let deadline = crate::arch::now_ms() + READ_TIMEOUT_MS;
        while !data.is_empty() {
            match crate::net::tcp_send(self.sock, data) {
                Some(0) | None => {}
                Some(n) => data = &data[n..],
            }
            if data.is_empty() {
                break;
            }
            if crate::arch::now_ms() >= deadline {
                return Err("ssh: timed out sending".to_string());
            }
            if crate::shell::poll_interrupt() {
                return Err("cancelled".to_string());
            }
            crate::shell::upkeep();
            crate::sched::yield_now();
        }
        Ok(())
    }

    /// Fill `inbuf` until it holds at least `want` bytes.
    fn fill(&mut self, want: usize) -> Result<(), String> {
        let deadline = crate::arch::now_ms() + READ_TIMEOUT_MS;
        let mut buf = [0u8; 4096];
        while self.inbuf.len() < want {
            match crate::net::tcp_recv(self.sock, &mut buf) {
                Some(0) | None => {
                    if !crate::net::tcp_may_recv(self.sock) && self.inbuf.len() < want {
                        return Err("ssh: connection closed by peer".to_string());
                    }
                }
                Some(n) => self.inbuf.extend_from_slice(&buf[..n]),
            }
            if self.inbuf.len() >= want {
                break;
            }
            if crate::arch::now_ms() >= deadline {
                return Err("ssh: timed out receiving".to_string());
            }
            if crate::shell::poll_interrupt() {
                return Err("cancelled".to_string());
            }
            crate::shell::upkeep();
            crate::sched::yield_now();
        }
        Ok(())
    }

    // --- packets ------------------------------------------------------------

    /// Send one payload, encrypting once the ciphers are on.
    pub fn send_packet(&mut self, payload: &[u8]) -> Result<(), String> {
        let bytes = match self.tx.as_mut() {
            Some(dir) => cipher::seal_payload(dir, self.tx_seq, payload, self.block_tx)
                .ok_or_else(|| "ssh: failed to encrypt a packet".to_string())?,
            None => wire::frame(payload, 8, wire::LengthMode::Encrypted, &mut |p| {
                crate::security::rng::fill_random(p)
            }),
        };
        // **The sequence number counts every packet from the very first**, plain
        // ones included, and does not reset at NEWKEYS.
        self.tx_seq = self.tx_seq.wrapping_add(1);
        self.send_all(&bytes)
    }

    /// Receive one payload.
    pub fn recv_packet(&mut self) -> Result<Vec<u8>, String> {
        let payload = match self.rx.as_mut() {
            None => {
                self.fill(4)?;
                let len = u32::from_be_bytes([self.inbuf[0], self.inbuf[1], self.inbuf[2], self.inbuf[3]]) as usize;
                if len < 2 || len > wire::MAX_PACKET {
                    return Err("ssh: implausible packet length".to_string());
                }
                self.fill(4 + len)?;
                let packet: Vec<u8> = self.inbuf.drain(..4 + len).collect();
                wire::payload_of(&packet)
                    .ok_or_else(|| "ssh: malformed packet".to_string())?
                    .to_vec()
            }
            Some(_) => {
                let prefix_len = self.rx.as_ref().unwrap().length_prefix();
                self.fill(prefix_len)?;
                let mut prefix: Vec<u8> = self.inbuf[..prefix_len].to_vec();
                let dir = self.rx.as_mut().unwrap();
                let len = dir
                    .peek_length(&mut prefix)
                    .ok_or_else(|| "ssh: could not read a packet length".to_string())?;
                if len < 2 || len > wire::MAX_PACKET {
                    return Err("ssh: implausible packet length".to_string());
                }
                let tag_len = dir.tag_len();
                let total = 4 + len;
                self.fill(total + tag_len)?;
                let rest: Vec<u8> = self.inbuf[prefix_len..total].to_vec();
                let tag: Vec<u8> = self.inbuf[total..total + tag_len].to_vec();
                let dir = self.rx.as_mut().unwrap();
                let plain = dir
                    .open(self.rx_seq, &prefix, &rest, &tag)
                    // Authentication failure is fatal and must never be retried:
                    // a peer that can make us retry can probe the cipher.
                    .ok_or_else(|| "ssh: packet authentication failed".to_string())?;
                self.inbuf.drain(..total + tag_len);
                wire::payload_of(&plain)
                    .ok_or_else(|| "ssh: malformed packet".to_string())?
                    .to_vec()
            }
        };
        self.rx_seq = self.rx_seq.wrapping_add(1);

        // Transport-level housekeeping the caller should never see.
        match payload.first().copied() {
            // SSH_MSG_IGNORE / DEBUG: skip and read the next one.
            Some(2) | Some(4) => self.recv_packet(),
            Some(1) => {
                // SSH_MSG_DISCONNECT carries a reason the user needs.
                let mut r = wire::Reader::new(&payload);
                let _ = r.u8();
                let code = r.u32().unwrap_or(0);
                let msg = r.utf8().unwrap_or("");
                Err(alloc::format!("ssh: server disconnected ({code}): {msg}"))
            }
            // SSH_MSG_UNIMPLEMENTED — informative, not fatal.
            Some(3) => self.recv_packet(),
            _ => Ok(payload),
        }
    }

    // --- handshake ----------------------------------------------------------

    fn handshake(&mut self, host: &str, port: u16) -> Result<(), String> {
        // 1. Identification strings.
        self.send_all(alloc::format!("{IDENT}\r\n").as_bytes())?;
        let server_ident = self.read_ident()?;
        if !server_ident.starts_with("SSH-2.0-") && !server_ident.starts_with("SSH-1.99-") {
            return Err(alloc::format!("ssh: unsupported protocol version {server_ident:?}"));
        }
        self.server_ident = server_ident.clone();

        // 2. KEXINIT both ways, kept verbatim for the exchange hash.
        let mut cookie = [0u8; 16];
        crate::security::rng::fill_random(&mut cookie);
        let ours = kex::KexInit::ours(cookie);
        let i_c = ours.encode();
        self.send_packet(&i_c)?;
        let i_s = self.recv_packet()?;
        let theirs = kex::KexInit::decode(&i_s)
            .ok_or_else(|| "ssh: the server's KEXINIT did not parse".to_string())?;
        let agreed = kex::negotiate(&ours, &theirs).map_err(|e| alloc::format!("ssh: {e}"))?;

        // 3. Key exchange — which also verifies the host key and turns on the
        // ciphers, so there is nothing left for the caller to do here.
        self.kex_exchange(&agreed, &i_c, &i_s, &server_ident, host, port)?;
        Ok(())
    }

    /// Read the server's identification line.
    ///
    /// A server may send any number of banner lines before it; only the one
    /// starting `SSH-` is the identification, and only that one is hashed.
    fn read_ident(&mut self) -> Result<String, String> {
        let deadline = crate::arch::now_ms() + HANDSHAKE_TIMEOUT_MS;
        loop {
            if let Some(pos) = self.inbuf.windows(2).position(|w| w == b"\r\n") {
                let line: Vec<u8> = self.inbuf.drain(..pos + 2).collect();
                let line = String::from_utf8_lossy(&line[..pos]).into_owned();
                if line.starts_with("SSH-") {
                    return Ok(line);
                }
                continue; // a pre-auth banner line; keep looking
            }
            if self.inbuf.len() > 64 * 1024 {
                return Err("ssh: no identification string in the first 64 KiB".to_string());
            }
            if crate::arch::now_ms() >= deadline {
                return Err("ssh: timed out waiting for the server identification".to_string());
            }
            self.fill(self.inbuf.len() + 1)?;
        }
    }

    /// ECDH, host-key verification, and turning the ciphers on.
    fn kex_exchange(
        &mut self,
        agreed: &kex::Negotiated,
        i_c: &[u8],
        i_s: &[u8],
        server_ident: &str,
        host: &str,
        port: u16,
    ) -> Result<(), String> {
        // Our ephemeral public key.
        let (q_c, ours_secret) = match agreed.kex.as_str() {
            "curve25519-sha256" | "curve25519-sha256@libssh.org" => {
                let mut seed = [0u8; 32];
                crate::security::rng::fill_random(&mut seed);
                let sk = x25519_dalek::StaticSecret::from(seed);
                let pk = x25519_dalek::PublicKey::from(&sk);
                (pk.as_bytes().to_vec(), Secret::X25519(alloc::boxed::Box::new(sk)))
            }
            "ecdh-sha2-nistp256" => {
                let sk = p256::ecdh::EphemeralSecret::random(&mut RngShim);
                let pk = p256::EncodedPoint::from(sk.public_key());
                (pk.as_bytes().to_vec(), Secret::P256(alloc::boxed::Box::new(sk)))
            }
            other => return Err(alloc::format!("ssh: unsupported key exchange {other}")),
        };

        let mut w = wire::Writer::msg(kex::SSH_MSG_KEX_ECDH_INIT);
        w.put_string(&q_c);
        self.send_packet(&w.into_vec())?;

        let reply = self.recv_packet()?;
        let mut r = wire::Reader::new(&reply);
        if r.u8() != Some(kex::SSH_MSG_KEX_ECDH_REPLY) {
            return Err("ssh: expected a key-exchange reply".to_string());
        }
        let k_s = r.string().ok_or("ssh: malformed kex reply (host key)")?.to_vec();
        let q_s = r.string().ok_or("ssh: malformed kex reply (server key)")?.to_vec();
        let sig = r.string().ok_or("ssh: malformed kex reply (signature)")?.to_vec();

        // The shared secret.
        let shared = ours_secret.agree(&q_s)?;

        // The exchange hash covers the identification strings without CR-LF and
        // the KEXINIT payloads exactly as they went on the wire.
        let h = kex::exchange_hash(
            IDENT.as_bytes(),
            server_ident.as_bytes(),
            i_c,
            i_s,
            &k_s,
            &q_c,
            &q_s,
            &shared,
        );

        // Verify the host key **before** trusting anything else it said.
        let key = hostkey::PublicKey::parse(&k_s)
            .ok_or_else(|| "ssh: the server sent a host key we cannot parse".to_string())?;
        if key.algorithm() != agreed.host_key {
            return Err(alloc::format!(
                "ssh: the server sent a {} host key after negotiating {}",
                key.algorithm(),
                agreed.host_key
            ));
        }
        if !key.verify(&h, &sig) {
            return Err("ssh: the host key signature did not verify".to_string());
        }
        self.host_key = Some(key.clone());
        verify_known_host(host, port, &key)?;

        // The first exchange hash is the session id, forever.
        if self.session_id.is_empty() {
            self.session_id = h.clone();
        }

        // NEWKEYS both ways, then the ciphers turn on.
        self.send_packet(&[kex::SSH_MSG_NEWKEYS])?;
        let got = self.recv_packet()?;
        if got.first() != Some(&kex::SSH_MSG_NEWKEYS) {
            return Err("ssh: expected NEWKEYS".to_string());
        }

        let s_c2s = cipher::sizes(&agreed.enc_c2s, &agreed.mac_c2s)
            .ok_or_else(|| "ssh: unsupported cipher (client to server)".to_string())?;
        let s_s2c = cipher::sizes(&agreed.enc_s2c, &agreed.mac_s2c)
            .ok_or_else(|| "ssh: unsupported cipher (server to client)".to_string())?;
        let d = |kind, len| kex::derive_key(&shared, &h, kind, &self.session_id, len);
        self.tx = cipher::Direction::new(
            &agreed.enc_c2s,
            &agreed.mac_c2s,
            &d(kex::KeyKind::EncC2s, s_c2s.key),
            &d(kex::KeyKind::IvC2s, s_c2s.iv),
            &d(kex::KeyKind::MacC2s, s_c2s.mac_key.max(1)),
        );
        self.rx = cipher::Direction::new(
            &agreed.enc_s2c,
            &agreed.mac_s2c,
            &d(kex::KeyKind::EncS2c, s_s2c.key),
            &d(kex::KeyKind::IvS2c, s_s2c.iv),
            &d(kex::KeyKind::MacS2c, s_s2c.mac_key.max(1)),
        );
        if self.tx.is_none() || self.rx.is_none() {
            return Err("ssh: could not initialise the negotiated cipher".to_string());
        }
        self.block_tx = s_c2s.block;
        crate::ktrace::log_fmt(format_args!(
            "ssh: {} kex={} hostkey={} cipher={}",
            server_ident, agreed.kex, agreed.host_key, agreed.enc_s2c
        ));
        Ok(())
    }

    // --- authentication -----------------------------------------------------

    /// Request the userauth service and try `publickey`, then `password`.
    pub fn authenticate(
        &mut self,
        user: &str,
        key: Option<&auth::PrivateKey>,
        password: Option<&str>,
    ) -> Result<(), String> {
        self.send_packet(&auth::service_request("ssh-userauth"))?;
        let reply = self.recv_packet()?;
        if reply.first() != Some(&auth::SSH_MSG_SERVICE_ACCEPT) {
            return Err("ssh: the server refused the userauth service".to_string());
        }

        // `none` first: its failure lists the methods that may work, which is
        // how we avoid offering a key to a password-only server.
        self.send_packet(&auth::request_none(user))?;
        let mut methods = match self.auth_reply()? {
            AuthReply::Success => return Ok(()),
            AuthReply::Failure(m) => m,
        };

        if let Some(k) = key {
            if methods.iter().any(|m| m == "publickey") {
                self.send_packet(&auth::request_publickey(&self.session_id.clone(), user, k))?;
                match self.auth_reply()? {
                    AuthReply::Success => return Ok(()),
                    AuthReply::Failure(m) => methods = m,
                }
            }
        }
        if let Some(p) = password {
            if methods.iter().any(|m| m == "password") {
                self.send_packet(&auth::request_password(user, p))?;
                if let AuthReply::Success = self.auth_reply()? {
                    return Ok(());
                }
            }
        }
        Err(alloc::format!(
            "ssh: authentication failed (server accepts: {})",
            if methods.is_empty() {
                "nothing".to_string()
            } else {
                methods.join(", ")
            }
        ))
    }

    fn auth_reply(&mut self) -> Result<AuthReply, String> {
        loop {
            let p = self.recv_packet()?;
            match p.first().copied() {
                Some(auth::SSH_MSG_USERAUTH_SUCCESS) => return Ok(AuthReply::Success),
                Some(auth::SSH_MSG_USERAUTH_FAILURE) => {
                    let (methods, _) = auth::parse_failure(&p)
                        .ok_or_else(|| "ssh: malformed authentication failure".to_string())?;
                    return Ok(AuthReply::Failure(methods));
                }
                // A banner is displayed, not an answer — keep waiting.
                Some(auth::SSH_MSG_USERAUTH_BANNER) => {
                    let mut r = wire::Reader::new(&p);
                    let _ = r.u8();
                    if let Some(text) = r.utf8() {
                        crate::serial_println!("{}", text.trim_end());
                    }
                }
                Some(auth::SSH_MSG_USERAUTH_PK_OK) => {} // probe accepted; keep reading
                _ => return Err("ssh: unexpected reply during authentication".to_string()),
            }
        }
    }

    // --- channels -----------------------------------------------------------

    /// Run a command (or a shell) and stream the result.
    ///
    /// `on_data` receives stdout as it arrives so a long `git-upload-pack` or an
    /// interactive shell does not have to be buffered whole. Returns the exit
    /// status the peer reported, if any.
    pub fn session(
        &mut self,
        what: Session<'_>,
        pty: Option<(u32, u32)>,
        mut on_data: impl FnMut(&[u8], bool),
        mut input: impl FnMut() -> Option<Vec<u8>>,
    ) -> Result<Option<u32>, String> {
        let id = self.next_channel;
        self.next_channel += 1;
        let mut ch = channel::Channel::new(id);
        self.send_packet(&channel::open_session(id))?;

        // Wait for the confirmation.
        loop {
            let p = self.recv_packet()?;
            match channel::parse(&p) {
                Some(channel::Event::OpenConfirmation {
                    remote_id,
                    window,
                    max_packet,
                    ..
                }) => {
                    ch.confirm(remote_id, window, max_packet);
                    break;
                }
                Some(channel::Event::OpenFailure { description, .. }) => {
                    return Err(alloc::format!("ssh: the server refused the channel: {description}"));
                }
                _ => {}
            }
        }

        if let Some((cols, rows)) = pty {
            self.send_packet(&channel::request_pty(ch.remote_id, "xterm-256color", cols, rows, false))?;
        }
        match what {
            Session::Exec(cmd) => self.send_packet(&channel::request_exec(ch.remote_id, cmd, false))?,
            Session::Shell => self.send_packet(&channel::request_shell(ch.remote_id, false))?,
        }

        // Pump until the channel closes.
        let deadline = crate::arch::now_ms() + READ_TIMEOUT_MS;
        let mut wrote_eof = false;
        while !ch.closed {
            // Forward any input the caller has.
            if !wrote_eof {
                match input() {
                    Some(bytes) if !bytes.is_empty() => {
                        let mut off = 0;
                        while off < bytes.len() {
                            let n = ch.sendable(bytes.len() - off);
                            if n == 0 {
                                break; // the window is closed; try again next turn
                            }
                            self.send_packet(&channel::data(ch.remote_id, &bytes[off..off + n]))?;
                            ch.sent(n);
                            off += n;
                        }
                    }
                    Some(_) => {}
                    None => {
                        self.send_packet(&channel::eof(ch.remote_id))?;
                        wrote_eof = true;
                    }
                }
            }

            let p = match self.recv_packet() {
                Ok(p) => p,
                // A clean close from the peer ends the session rather than
                // failing it — a finished command closes the connection.
                Err(e) if e.contains("closed by peer") => break,
                Err(e) => return Err(e),
            };
            match channel::parse(&p) {
                Some(channel::Event::Data { data, .. }) => {
                    on_data(&data, false);
                    if let Some(extra) = ch.consume(data.len()) {
                        self.send_packet(&channel::window_adjust(ch.remote_id, extra))?;
                    }
                }
                Some(channel::Event::ExtendedData { data, kind, .. }) => {
                    on_data(&data, kind == channel::EXTENDED_DATA_STDERR);
                    if let Some(extra) = ch.consume(data.len()) {
                        self.send_packet(&channel::window_adjust(ch.remote_id, extra))?;
                    }
                }
                Some(channel::Event::WindowAdjust { extra, .. }) => ch.grant(extra),
                Some(channel::Event::ExitStatus { status, .. }) => ch.exit_status = Some(status),
                Some(channel::Event::Eof { .. }) => ch.eof_received = true,
                Some(channel::Event::Close { .. }) => {
                    self.send_packet(&channel::close(ch.remote_id))?;
                    ch.closed = true;
                }
                Some(channel::Event::Request { want_reply, .. }) => {
                    if want_reply {
                        // Refuse politely rather than ignore: a peer that asked
                        // for a reply waits for one.
                        let mut w = wire::Writer::msg(channel::SSH_MSG_CHANNEL_FAILURE);
                        w.put_u32(ch.remote_id);
                        self.send_packet(&w.into_vec())?;
                    }
                }
                _ => {}
            }
            if crate::arch::now_ms() >= deadline {
                return Err("ssh: the session timed out".to_string());
            }
            if crate::shell::poll_interrupt() {
                let _ = self.send_packet(&channel::close(ch.remote_id));
                return Err("cancelled".to_string());
            }
            crate::shell::upkeep();
        }
        Ok(ch.exit_status)
    }

    pub fn disconnect(&mut self) {
        let mut w = wire::Writer::msg(1); // SSH_MSG_DISCONNECT
        w.put_u32(11); // SSH_DISCONNECT_BY_APPLICATION
        w.put_str("bye");
        w.put_str("");
        let _ = self.send_packet(&w.into_vec());
        crate::net::tcp_close(self.sock);
    }
}

enum AuthReply {
    Success,
    Failure(Vec<String>),
}

/// The ephemeral secret, kept behind an enum so the two curves share a path.
enum Secret {
    X25519(alloc::boxed::Box<x25519_dalek::StaticSecret>),
    P256(alloc::boxed::Box<p256::ecdh::EphemeralSecret>),
}

impl Secret {
    /// Agree with the peer's public key, returning the shared secret as the
    /// unsigned magnitude the exchange hash wants.
    fn agree(self, peer: &[u8]) -> Result<Vec<u8>, String> {
        match self {
            Secret::X25519(sk) => {
                let pk: [u8; 32] = peer
                    .try_into()
                    .map_err(|_| "ssh: the server's curve25519 key is the wrong size".to_string())?;
                let shared = sk.diffie_hellman(&x25519_dalek::PublicKey::from(pk));
                // An all-zero shared secret means a small-order peer key; refuse.
                if shared.as_bytes().iter().all(|&b| b == 0) {
                    return Err("ssh: the server sent a degenerate curve25519 key".to_string());
                }
                Ok(shared.as_bytes().to_vec())
            }
            Secret::P256(sk) => {
                let point = p256::EncodedPoint::from_bytes(peer)
                    .map_err(|_| "ssh: the server's P-256 key did not parse".to_string())?;
                let pk = p256::PublicKey::from_sec1_bytes(point.as_bytes())
                    .map_err(|_| "ssh: the server's P-256 key is not on the curve".to_string())?;
                let shared = sk.diffie_hellman(&pk);
                Ok(shared.raw_secret_bytes().to_vec())
            }
        }
    }
}

/// `rand_core` shim over the kernel CSPRNG, so `p256` can draw an ephemeral key.
struct RngShim;

impl rand_core::RngCore for RngShim {
    fn next_u32(&mut self) -> u32 {
        let mut b = [0u8; 4];
        crate::security::rng::fill_random(&mut b);
        u32::from_le_bytes(b)
    }
    fn next_u64(&mut self) -> u64 {
        let mut b = [0u8; 8];
        crate::security::rng::fill_random(&mut b);
        u64::from_le_bytes(b)
    }
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        crate::security::rng::fill_random(dest);
    }
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl rand_core::CryptoRng for RngShim {}

/// Where the known-hosts file lives on the store.
pub const KNOWN_HOSTS: &str = "/configs/core/known_hosts";

/// Check the host key against `known_hosts`, and record it on first contact.
///
/// A **changed** key is refused outright. Prompting there would train the user
/// to accept exactly the case that matters; a first-contact key is recorded with
/// its fingerprint logged so it can be compared after the fact.
fn verify_known_host(host: &str, port: u16, key: &hostkey::PublicKey) -> Result<(), String> {
    let text = crate::synapse::fs::read(KNOWN_HOSTS)
        .map(|b| String::from_utf8_lossy(&b).into_owned())
        .unwrap_or_default();
    let entries = hostkey::parse_known_hosts(&text);
    match hostkey::check(&entries, host, port, key) {
        hostkey::Trust::Known => Ok(()),
        hostkey::Trust::Changed { .. } => Err(alloc::format!(
            "ssh: REMOTE HOST IDENTIFICATION HAS CHANGED for {host}\n\
             ssh: the key is now {}\n\
             ssh: someone could be eavesdropping, or the server was rebuilt.\n\
             ssh: remove the old line from {KNOWN_HOSTS} if you are sure.",
            key.fingerprint()
        )),
        hostkey::Trust::Unknown => {
            let line = hostkey::known_hosts_line(host, port, key);
            let mut next = text;
            next.push_str(&line);
            let _ = crate::synapse::fs::write(KNOWN_HOSTS, next.as_bytes());
            crate::serial_println!(
                "ssh> the authenticity of '{host}' cannot be established.\n\
                 ssh> {} key fingerprint is {}.\n\
                 ssh> recorded in {KNOWN_HOSTS}.",
                key.algorithm(),
                key.fingerprint()
            );
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Our identification string carries no CR-LF: the terminator goes on the
    /// wire but is **not** hashed, and keeping them separate is what stops the
    /// classic exchange-hash mismatch.
    #[test_case]
    fn the_identification_string_excludes_its_terminator() {
        assert!(IDENT.starts_with("SSH-2.0-"));
        assert!(!IDENT.contains('\r') && !IDENT.contains('\n'));
        // RFC 4253 §4.2 also forbids spaces before the optional comment.
        assert!(!IDENT.contains(' '));
    }

    /// The known-hosts decision maps to the three outcomes a user sees, and a
    /// changed key is a refusal rather than a prompt.
    #[test_case]
    fn a_changed_host_key_is_refused() {
        let key = hostkey::PublicKey::Ed25519([1u8; 32]);
        let other = hostkey::PublicKey::Ed25519([2u8; 32]);
        let file = hostkey::known_hosts_line("h", 22, &key);
        let entries = hostkey::parse_known_hosts(&file);
        assert_eq!(hostkey::check(&entries, "h", 22, &key), hostkey::Trust::Known);
        assert!(matches!(
            hostkey::check(&entries, "h", 22, &other),
            hostkey::Trust::Changed { .. }
        ));
    }
}
