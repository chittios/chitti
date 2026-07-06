//! In-kernel **TLS 1.3 client** for `https://` — the transport under
//! [`super::http`] when a URL is `https`. Built on `embedded-tls` (pure-Rust,
//! no_std, RustCrypto: AES-128-GCM / SHA-256 / P-256 ECDHE) driven in blocking
//! mode over an [`embedded_io`] adapter around our cooperative smoltcp TCP
//! socket ([`TcpStream`]).
//!
//! **Security posture — read this.** Server certificates are **not verified**
//! (`NoVerify`): there is no in-kernel trust store, clock-backed validity
//! check, or hostname binding yet, so this protects against passive
//! eavesdropping but **not** an active man-in-the-middle. It is the moral
//! equivalent of `curl -k`. That is acceptable for the intended use — reaching
//! a *self-hosted* model server over a trusted LAN where the alternative was
//! plaintext — but do not send secrets to an untrusted public endpoint over
//! it. The CSPRNG ([`seed_rng`]) is ChaCha20 seeded from `RDRAND`/`RNDR` when
//! present and cycle-counter jitter otherwise; adequate for ephemeral
//! handshake keys on a research OS, not audited crypto entropy.

use super::NET;
use alloc::string::String;
use embedded_io::{ErrorType, Read as IoRead, Write as IoWrite};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use smoltcp::socket::tcp;

/// Seed a ChaCha20 CSPRNG for the TLS handshake. Mixes several hardware-random
/// words (`RDRAND`/`RNDR`, 0 when absent) with cycle-counter samples taken
/// across cooperative yields (timing jitter) via a SplitMix64 diffuser, so the
/// seed is unpredictable even when no hardware RNG exists.
pub fn seed_rng() -> ChaCha20Rng {
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15 ^ crate::arch::now_ms().wrapping_mul(0xff51_afd7_ed55_8ccd);
    let mut seed = [0u8; 32];
    for chunk in seed.chunks_mut(8) {
        // Fold in a fresh hardware-random word + the live cycle counter, then
        // yield so the next counter sample reflects real scheduling jitter.
        state ^= crate::arch::hw_rand();
        state = state.wrapping_add(crate::arch::cycle_count());
        crate::sched::yield_now();
        state ^= crate::arch::cycle_count().rotate_left(17);
        // SplitMix64 finaliser.
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        chunk.copy_from_slice(&z.to_le_bytes());
    }
    ChaCha20Rng::from_seed(seed)
}

/// A blocking [`embedded_io`] view over one cooperative smoltcp TCP socket:
/// `read`/`write` pump the stack (and the UI, via `shell::upkeep`) until the
/// socket makes progress, so the synchronous embedded-tls state machine runs
/// on top of our poll-driven stack. `deadline` (an `arch::now_ms` value) bounds
/// every wait.
pub struct TcpStream {
    pub handle: super::TcpHandle,
    /// Fixed absolute read/write deadline (`arch::now_ms` value) — used by the
    /// one-shot HTTP path.
    pub deadline: u64,
    /// Optional **rolling** deadline: when set, this atomic overrides
    /// `deadline`, so a long-lived owner (the `wss://` WebSocket) can push the
    /// timeout forward before each read instead of the session expiring at a
    /// fixed instant. embedded-tls holds this `TcpStream` and reads the same
    /// atomic, so bumping it here extends the deadline it observes.
    pub rolling: Option<&'static core::sync::atomic::AtomicU64>,
}

impl TcpStream {
    /// A stream with a fixed absolute `deadline` (the HTTP request path).
    pub fn new(handle: super::TcpHandle, deadline: u64) -> Self {
        TcpStream { handle, deadline, rolling: None }
    }
    /// A stream whose deadline is read from (and bumped through) a shared
    /// atomic — for a long-lived TLS session (`wss://`).
    pub fn with_rolling(handle: super::TcpHandle, rolling: &'static core::sync::atomic::AtomicU64) -> Self {
        TcpStream { handle, deadline: 0, rolling: Some(rolling) }
    }
}

/// The one error kind we surface to embedded-tls; it only inspects
/// `ErrorKind`, so a single "other" is enough.
#[derive(Debug)]
pub struct StreamError(pub &'static str);

impl embedded_io::Error for StreamError {
    fn kind(&self) -> embedded_io::ErrorKind {
        embedded_io::ErrorKind::Other
    }
}

impl ErrorType for TcpStream {
    type Error = StreamError;
}

impl TcpStream {
    /// The deadline in effect: the rolling atomic if present, else the fixed one.
    fn effective_deadline(&self) -> u64 {
        match self.rolling {
            Some(a) => a.load(core::sync::atomic::Ordering::Relaxed),
            None => self.deadline,
        }
    }
    fn timed_out(&self) -> bool {
        crate::arch::now_ms() >= self.effective_deadline()
    }
}

impl IoRead for TcpStream {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, StreamError> {
        loop {
            if self.timed_out() {
                return Err(StreamError("TLS read timeout"));
            }
            super::poll();
            let r = NET.with(|n| {
                let s = n.as_mut().ok_or(StreamError("network down"))?;
                let sock = s.tcp_set(self.handle).get_mut::<tcp::Socket>(self.handle.handle);
                if sock.can_recv() {
                    let k = sock.recv_slice(buf).map_err(|_| StreamError("recv"))?;
                    return Ok(Some(k));
                }
                // No data: EOF once the peer closed and the rx buffer drained.
                if !sock.may_recv() {
                    return Ok(Some(0));
                }
                Ok(None)
            })?;
            if let Some(k) = r {
                return Ok(k);
            }
            crate::shell::upkeep();
            crate::sched::yield_now();
        }
    }
}

impl IoWrite for TcpStream {
    fn write(&mut self, buf: &[u8]) -> Result<usize, StreamError> {
        loop {
            if self.timed_out() {
                return Err(StreamError("TLS write timeout"));
            }
            super::poll();
            let r = NET.with(|n| {
                let s = n.as_mut().ok_or(StreamError("network down"))?;
                let sock = s.tcp_set(self.handle).get_mut::<tcp::Socket>(self.handle.handle);
                if !sock.may_send() {
                    return Err(StreamError("connection closed"));
                }
                if sock.can_send() {
                    let k = sock.send_slice(buf).map_err(|_| StreamError("send"))?;
                    if k > 0 {
                        return Ok(Some(k));
                    }
                }
                Ok(None)
            })?;
            if let Some(k) = r {
                super::poll(); // flush the freshly-queued segment onto the wire
                return Ok(k);
            }
            crate::shell::upkeep();
            crate::sched::yield_now();
        }
    }

    fn flush(&mut self) -> Result<(), StreamError> {
        super::poll();
        Ok(())
    }
}

/// An open TLS session: the embedded-tls connection plus its record buffers
/// (kept alive alongside it — the connection borrows them for their lifetime).
pub struct TlsSession {
    tls: embedded_tls::blocking::TlsConnection<'static, TcpStream, embedded_tls::Aes128GcmSha256>,
}

impl TlsSession {
    /// Write all of `data` (buffered + flushed) over the TLS session.
    pub fn write_all(&mut self, data: &[u8]) -> Result<(), String> {
        let mut off = 0;
        while off < data.len() {
            let k = self.tls.write(&data[off..]).map_err(|e| alloc::format!("TLS write: {:?}", e))?;
            if k == 0 {
                return Err("TLS write made no progress".into());
            }
            off += k;
        }
        self.tls.flush().map_err(|e| alloc::format!("TLS flush: {:?}", e))
    }

    /// Read up to `buf.len()` decrypted bytes. `Ok(0)` = peer closed the TLS
    /// session (close_notify or transport EOF).
    pub fn read(&mut self, buf: &mut [u8]) -> usize {
        match self.tls.read(buf) {
            Ok(k) => k,
            // A read error after data flowed is a normal close for a
            // `Connection: close` response — treat it as EOF.
            Err(_) => 0,
        }
    }
}

/// Perform the TLS 1.3 handshake over an already-connected [`TcpStream`],
/// binding SNI to `server_name`. On success returns an open [`TlsSession`].
pub fn handshake(stream: TcpStream, server_name: &str) -> Result<TlsSession, String> {
    use embedded_tls::blocking::{Aes128GcmSha256, NoVerify, TlsConfig, TlsConnection, TlsContext};

    // Record buffers live as long as the connection; leak them (one per
    // request, freed implicitly at process teardown — a request is short-lived
    // and the heap is large). 16640 = the max TLS record; 4 KiB write buffer.
    let rd: &'static mut [u8] = alloc::vec![0u8; 16 * 1024 + 256].leak();
    let wr: &'static mut [u8] = alloc::vec![0u8; 4096].leak();
    // SNI must outlive the config borrow; leak the small hostname string.
    let name: &'static str = String::from(server_name).leak();

    let config: &'static TlsConfig<'static, Aes128GcmSha256> =
        alloc::boxed::Box::leak(alloc::boxed::Box::new(TlsConfig::new().with_server_name(name)));
    let mut rng = seed_rng();

    let mut tls = TlsConnection::new(stream, rd, wr);
    tls.open::<ChaCha20Rng, NoVerify>(TlsContext::new(config, &mut rng))
        .map_err(|e| alloc::format!("TLS handshake failed: {:?} (server must support TLS 1.3)", e))?;
    Ok(TlsSession { tls })
}
