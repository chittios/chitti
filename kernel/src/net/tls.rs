//! In-kernel **TLS 1.3 client** for `https://` — the transport under
//! [`super::http`] when a URL is `https`. Built on `embedded-tls` (pure-Rust,
//! no_std, RustCrypto: AES-128-GCM / SHA-256 / P-256 ECDHE) driven in blocking
//! mode over an [`embedded_io`] adapter around our cooperative smoltcp TCP
//! socket ([`TcpStream`]).
//!
//! **Security posture.** Server certificates **are verified by default**
//! against an embedded Mozilla root store ([`super::ca_roots`]) via the
//! pure-Rust chain validator [`super::x509`] (`ChittiVerifier` below): a chain
//! to a trusted root, validity window against the wall clock, and hostname vs.
//! the leaf SANs. `ring` can't build bare-metal, so the standard webpki path
//! is unavailable — this validator is built from `x509-cert` + RustCrypto
//! (`p256`/`p384`/`crypto-bigint`) instead. A `curl -k` escape hatch
//! ([`set_insecure`], `/tls insecure`) falls back to `NoVerify` for a
//! self-hosted box with a self-signed cert. **Not** covered: CRL/OCSP
//! revocation. The CSPRNG ([`seed_rng`]) is ChaCha20 seeded from `RDRAND`/
//! `RNDR` when present and cycle-counter jitter otherwise; adequate for
//! ephemeral handshake keys on a research OS, not audited crypto entropy.

use super::NET;
use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, Ordering};
use embedded_io::{ErrorType, Read as IoRead, Write as IoWrite};

/// `curl -k` mode: skip certificate verification (self-signed / self-hosted).
/// Off by default — verification is the default posture. Human-set only
/// (`/tls insecure on|off`), like the remote-model backend switch.
static TLS_INSECURE: AtomicBool = AtomicBool::new(false);

/// Enable/disable certificate verification skipping (`curl -k`).
pub fn set_insecure(on: bool) {
    TLS_INSECURE.store(on, Ordering::Relaxed);
}
/// Whether certificate verification is currently skipped.
pub fn insecure() -> bool {
    TLS_INSECURE.load(Ordering::Relaxed)
}
use rand_chacha::ChaCha20Rng;
use smoltcp::socket::tcp;

/// Seed a ChaCha20 CSPRNG for the TLS handshake.
///
/// Re-exported from [`crate::security::rng`], which is the kernel's single
/// seeding path — the volume-encryption salt and the login salt draw from the
/// same function, so there is one implementation to get right and one place that
/// documents what the entropy actually is.
pub use crate::security::rng::seed_rng;

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

/// Latched when a stream read/write aborted because the human pressed Ctrl+C.
///
/// A cancel is detected **inside** the transport, at the bottom of a stack whose
/// middle layer (embedded-tls) wraps our error in its own type and loses the
/// reason. Without a latch, "the human stopped this" arrives at the HTTP layer as
/// an anonymous read failure — and gets reported as a network fault, which is how
/// a plain Ctrl+C came out as *"no response head (connection closed early)"*.
///
/// `poll_interrupt` consumes the keystroke, so the fact has to be recorded when it
/// is observed; nobody upstream can re-derive it.
static READ_CANCELLED: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Take (and clear) the cancel latch. Called by the HTTP layer when classifying a
/// failed read, so a cancel is never reported as a transport error.
pub fn take_cancelled() -> bool {
    READ_CANCELLED.swap(false, core::sync::atomic::Ordering::Relaxed)
}

/// Clear the latch before an exchange, so a stale cancel from an earlier request
/// cannot make this one look cancelled.
pub fn clear_cancelled() {
    READ_CANCELLED.store(false, core::sync::atomic::Ordering::Relaxed);
}

fn latch_cancel() -> StreamError {
    READ_CANCELLED.store(true, core::sync::atomic::Ordering::Relaxed);
    StreamError(super::http::CANCELLED)
}

/// Set the latch without a socket, so the classification rule is testable
/// off-hardware (the real setter needs a live read to cancel).
#[cfg(test)]
pub fn latch_cancel_for_test() {
    let _ = latch_cancel();
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
            if crate::shell::poll_interrupt() {
                return Err(latch_cancel());
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
            if crate::shell::poll_interrupt() {
                return Err(latch_cancel());
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

    /// Read up to `buf.len()` decrypted bytes. `Ok(0)` = the peer closed the TLS
    /// session cleanly (close_notify or transport EOF); `Err` carries the reason
    /// it failed instead.
    ///
    /// This used to map **every** error to `0`, on the reasoning that "a read
    /// error after data flowed is a normal close for a `Connection: close`
    /// response" — true, but the code applied it whether or not data had flowed.
    /// So a TLS read timeout, a cancel, an alert and a decrypt failure all
    /// reached the HTTP layer as EOF and were reported as
    /// *"no response head (connection closed early)"*: four distinct facts
    /// collapsed into the one that happened to be wrong. Deciding "is this the
    /// normal end of a response?" needs to know whether a response had *started*,
    /// which is the HTTP layer's knowledge, not this one's — so the reason is
    /// handed up and [`super::http::drive_stream`] applies that rule.
    pub fn read(&mut self, buf: &mut [u8]) -> Result<usize, String> {
        self.tls
            .read(buf)
            .map_err(|e| alloc::format!("TLS read: {:?}", e))
    }
}

/// Certificate-chain verifier: on `verify_certificate` it runs the pure-Rust
/// [`super::x509`] path validator against the embedded root store + hostname;
/// on `verify_signature` it checks the TLS 1.3 `CertificateVerify` with the
/// saved leaf key. Generic over the negotiated cipher suite so the handshake
/// transcript hash type is threaded through.
struct ChittiVerifier<'a, CipherSuite>
where
    CipherSuite: embedded_tls::blocking::TlsCipherSuite,
{
    host: Option<&'a str>,
    transcript: Option<CipherSuite::Hash>,
    leaf_spki: Option<Vec<u8>>,
}

impl<'a, CipherSuite> embedded_tls::blocking::TlsVerifier<'a, CipherSuite> for ChittiVerifier<'a, CipherSuite>
where
    CipherSuite: embedded_tls::blocking::TlsCipherSuite,
{
    fn new(host: Option<&'a str>) -> Self {
        ChittiVerifier { host, transcript: None, leaf_spki: None }
    }

    fn verify_certificate(
        &mut self,
        transcript: &CipherSuite::Hash,
        _ca: &Option<embedded_tls::blocking::Certificate>,
        cert: embedded_tls::CertificateRef,
    ) -> Result<(), embedded_tls::TlsError> {
        use embedded_tls::CertificateEntryRef;
        // Presented chain, leaf first.
        let chain: Vec<&[u8]> = cert
            .entries
            .iter()
            .filter_map(|e| match e {
                CertificateEntryRef::X509(der) => Some(*der),
                #[allow(unreachable_patterns)]
                _ => None,
            })
            .collect();
        let host = self.host.unwrap_or("");
        let spki = super::x509::verify(&chain, host).map_err(|e| {
            crate::ktrace::log_fmt(format_args!("tls: {e}"));
            embedded_tls::TlsError::InvalidCertificate
        })?;
        self.leaf_spki = Some(spki);
        self.transcript = Some(transcript.clone());
        Ok(())
    }

    fn verify_signature(&mut self, verify: embedded_tls::CertificateVerify) -> Result<(), embedded_tls::TlsError> {
        use sha2::Digest;
        let transcript = self.transcript.take().ok_or(embedded_tls::TlsError::InvalidSignature)?;
        let spki = self.leaf_spki.as_ref().ok_or(embedded_tls::TlsError::InvalidSignature)?;
        // RFC 8446 §4.4.3: 64 spaces + context string + NUL + transcript hash.
        let mut msg: Vec<u8> = alloc::vec![0x20u8; 64];
        msg.extend_from_slice(b"TLS 1.3, server CertificateVerify\x00");
        msg.extend_from_slice(&transcript.finalize());
        let scheme = verify.signature_scheme as u16;
        match super::x509::verify_data(spki, scheme, &msg, verify.signature) {
            Ok(true) => Ok(()),
            _ => Err(embedded_tls::TlsError::InvalidSignature),
        }
    }
}

/// Perform the TLS 1.3 handshake over an already-connected [`TcpStream`],
/// binding SNI to `server_name`. On success returns an open [`TlsSession`].
/// Verifies the server certificate chain unless [`insecure`] is set.
pub fn handshake(stream: TcpStream, server_name: &str) -> Result<TlsSession, String> {
    use embedded_tls::blocking::{Aes128GcmSha256, NoVerify, TlsConfig, TlsConnection, TlsContext};

    // Record buffers live as long as the connection; leak them (one per
    // request, freed implicitly at process teardown — a request is short-lived
    // and the heap is large). The read buffer must hold the largest incoming
    // TLS record *including* the 5-byte record header and AEAD overhead: a max
    // 2^14 plaintext + 256 expansion + 5 header ≈ 16645 bytes. The old
    // `16*1024 + 256` (16640) was a few bytes short, so a server that sends a
    // full-size record (e.g. a big certificate chain — upload.wikimedia.org)
    // overflowed it and the handshake died with `DecodeError`. Use 32 KiB for
    // comfortable headroom over any single record. 4 KiB write buffer.
    let rd: &'static mut [u8] = alloc::vec![0u8; 32 * 1024].leak();
    let wr: &'static mut [u8] = alloc::vec![0u8; 4096].leak();
    // SNI must outlive the config borrow; leak the small hostname string.
    let name: &'static str = String::from(server_name).leak();

    let config: &'static TlsConfig<'static, Aes128GcmSha256> =
        alloc::boxed::Box::leak(alloc::boxed::Box::new(TlsConfig::new().with_server_name(name)));
    // Leak RNG so both open monomorphizations can borrow it for the static
    // verifier path without fighting stack lifetimes.
    let rng: &'static mut ChaCha20Rng =
        alloc::boxed::Box::leak(alloc::boxed::Box::new(seed_rng()));

    let mut tls = TlsConnection::new(stream, rd, wr);
    // Verifier type is a compile-time parameter, so the two postures are two
    // monomorphized `open` calls selected by the runtime `insecure` flag.
    let r = if insecure() {
        tls.open::<ChaCha20Rng, NoVerify>(TlsContext::new(config, rng))
    } else {
        tls.open::<ChaCha20Rng, ChittiVerifier<'static, Aes128GcmSha256>>(TlsContext::new(config, rng))
    };
    r.map_err(|e| {
        if insecure() {
            alloc::format!("TLS handshake failed: {:?} (TLS 1.3, AES-128-GCM, cert verification OFF)", e)
        } else {
            alloc::format!("TLS handshake failed: {:?} (cert verification on; /tls insecure on for self-signed hosts)", e)
        }
    })?;
    Ok(TlsSession { tls })
}
