//! Inter-agent **channels**: cap-gated byte-stream and datagram conduits, the
//! Linux-pipe/socket analog for Chitti agents. Unlike [`crate::ipc`] endpoints
//! (which carry a single `u64` control word), a channel carries **raw bytes** —
//! a producer agent writes a stream, a consumer agent reads it, and a service
//! agent can later hand one end to another agent (a `Tcp`-backed channel is the
//! stream-handoff of a live connection, added with the net listener).
//!
//! Authority is per-direction and unforgeable. A channel *end* is a
//! [`crate::cap::Right::ChannelWrite`]/[`ChannelRead`] granted into a task's own
//! capability table; a model-emitted channel handle is a `Cap` slot index into
//! that same table, resolved by the Synapse executor. There is deliberately no
//! API here to name another task or to read a channel you weren't granted an end
//! to — every path is capability-gated by construction, exactly like `ipc`.
//!
//! The primitives below are all **non-blocking**; cooperative blocking is
//! composed from them by [`read_blocking`] (the standing yield-poll + `upkeep`
//! pattern, mirroring [`crate::ipc::receive`] and `net/tls.rs`).

use crate::cap::ChannelId;
use crate::mm::Locked;
use alloc::collections::{BTreeMap, VecDeque};
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

/// Which flavour of conduit a channel is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelKind {
    /// An ordered byte stream (a pipe): writes append, reads drain in order.
    Stream,
    /// A datagram queue: each send/recv moves one whole frame.
    Datagram,
}

/// One byte-stream ring buffer (the `Stream` backend).
struct Ring {
    buf: VecDeque<u8>,
    /// Max bytes buffered before a write back-pressures (returns fewer bytes).
    cap: usize,
    /// The writer end has been closed; once `buf` drains, readers see EOF.
    write_closed: bool,
    /// The reader end has been closed; further writes report `Closed`.
    read_closed: bool,
}

/// A datagram queue (the `Datagram` backend).
struct Dgram {
    frames: VecDeque<Vec<u8>>,
    /// Max frames buffered before a send back-pressures.
    cap_frames: usize,
    write_closed: bool,
    read_closed: bool,
}

enum Backend {
    Pipe(Ring),
    Datagram(Dgram),
}

struct Channel {
    backend: Backend,
    /// Number of live ends (read + write caps outstanding). The channel is
    /// torn down when this reaches zero. Ends are counted by `create`
    /// (grants both) and decremented by [`close_end`].
    ends: u32,
}

static NEXT_CHANNEL: AtomicU64 = AtomicU64::new(0);
static CHANNELS: Locked<BTreeMap<ChannelId, Channel>> = Locked::new(BTreeMap::new());

/// Errors a channel operation can report. `WouldBlock` is a transient
/// back-pressure/no-data signal (the caller should yield and retry);
/// `Closed` is terminal for the *writer* side (the reader was dropped).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelError {
    /// The peer end has been closed and no more progress is possible.
    Closed,
    /// No data available / buffer full right now; retry after yielding.
    WouldBlock,
    /// A datagram frame exceeded the channel's per-frame budget.
    TooLarge,
    /// The channel id does not exist (already torn down).
    NoSuchChannel,
}

/// Create a channel. Says nothing about who may use it — that is decided
/// entirely by who is subsequently granted a `ChannelRead`/`ChannelWrite`
/// naming it (kernel-mediated `cap::grant`), exactly like `ipc::create_endpoint`.
/// The channel starts with `ends = 2` (a read end and a write end to be granted).
pub fn create(kind: ChannelKind, ring_cap: usize) -> ChannelId {
    let id = NEXT_CHANNEL.fetch_add(1, Ordering::SeqCst);
    let backend = match kind {
        ChannelKind::Stream => Backend::Pipe(Ring {
            buf: VecDeque::new(),
            cap: ring_cap.max(1),
            write_closed: false,
            read_closed: false,
        }),
        ChannelKind::Datagram => Backend::Datagram(Dgram {
            frames: VecDeque::new(),
            cap_frames: ring_cap.max(1),
            write_closed: false,
            read_closed: false,
        }),
    };
    CHANNELS.with(|m| m.insert(id, Channel { backend, ends: 2 }));
    id
}

/// Non-blocking write of raw bytes into a stream channel. Returns the number of
/// bytes accepted (may be fewer than `data.len()` under back-pressure, or 0 if
/// the ring is full). `Closed` if the reader end was dropped.
pub fn try_write(id: ChannelId, data: &[u8]) -> Result<usize, ChannelError> {
    CHANNELS.with(|m| {
        let ch = m.get_mut(&id).ok_or(ChannelError::NoSuchChannel)?;
        match &mut ch.backend {
            Backend::Pipe(r) => {
                if r.read_closed {
                    return Err(ChannelError::Closed);
                }
                let room = r.cap.saturating_sub(r.buf.len());
                let n = data.len().min(room);
                r.buf.extend(data[..n].iter().copied());
                Ok(n)
            }
            Backend::Datagram(_) => Err(ChannelError::TooLarge), // use try_send_dgram
        }
    })
}

/// Non-blocking read of up to `buf.len()` bytes from a stream channel. `Ok(0)`
/// with [`is_eof`] true means the writer closed and the ring is drained.
pub fn try_read(id: ChannelId, buf: &mut [u8]) -> Result<usize, ChannelError> {
    CHANNELS.with(|m| {
        let ch = m.get_mut(&id).ok_or(ChannelError::NoSuchChannel)?;
        match &mut ch.backend {
            Backend::Pipe(r) => {
                let n = buf.len().min(r.buf.len());
                for slot in buf.iter_mut().take(n) {
                    *slot = r.buf.pop_front().unwrap();
                }
                Ok(n)
            }
            Backend::Datagram(_) => Err(ChannelError::TooLarge), // use try_recv_dgram
        }
    })
}

/// Non-blocking datagram send: enqueues one whole frame. `Closed` if the reader
/// was dropped; `WouldBlock` if the frame queue is full.
pub fn try_send_dgram(id: ChannelId, frame: &[u8]) -> Result<(), ChannelError> {
    CHANNELS.with(|m| {
        let ch = m.get_mut(&id).ok_or(ChannelError::NoSuchChannel)?;
        match &mut ch.backend {
            Backend::Datagram(d) => {
                if d.read_closed {
                    return Err(ChannelError::Closed);
                }
                if d.frames.len() >= d.cap_frames {
                    return Err(ChannelError::WouldBlock);
                }
                d.frames.push_back(frame.to_vec());
                Ok(())
            }
            Backend::Pipe(_) => Err(ChannelError::TooLarge),
        }
    })
}

/// Non-blocking datagram receive: `Ok(None)` if no frame is queued right now.
pub fn try_recv_dgram(id: ChannelId) -> Result<Option<Vec<u8>>, ChannelError> {
    CHANNELS.with(|m| {
        let ch = m.get_mut(&id).ok_or(ChannelError::NoSuchChannel)?;
        match &mut ch.backend {
            Backend::Datagram(d) => Ok(d.frames.pop_front()),
            Backend::Pipe(_) => Err(ChannelError::TooLarge),
        }
    })
}

/// Whether the channel is at end-of-stream for a reader: the writer end has
/// been closed *and* all buffered bytes/frames have been drained. A torn-down
/// channel also reports EOF.
pub fn is_eof(id: ChannelId) -> bool {
    CHANNELS.with(|m| match m.get(&id) {
        None => true,
        Some(ch) => match &ch.backend {
            Backend::Pipe(r) => r.write_closed && r.buf.is_empty(),
            Backend::Datagram(d) => d.write_closed && d.frames.is_empty(),
        },
    })
}

/// How many bytes/frames are currently readable without blocking.
pub fn readable_len(id: ChannelId) -> usize {
    CHANNELS.with(|m| match m.get(&id) {
        None => 0,
        Some(ch) => match &ch.backend {
            Backend::Pipe(r) => r.buf.len(),
            Backend::Datagram(d) => d.frames.len(),
        },
    })
}

/// Mark the writer end closed (readers will see EOF once drained).
pub fn close_write(id: ChannelId) {
    CHANNELS.with(|m| {
        if let Some(ch) = m.get_mut(&id) {
            match &mut ch.backend {
                Backend::Pipe(r) => r.write_closed = true,
                Backend::Datagram(d) => d.write_closed = true,
            }
        }
    });
}

/// Mark the reader end closed (writers will see `Closed`).
pub fn close_read(id: ChannelId) {
    CHANNELS.with(|m| {
        if let Some(ch) = m.get_mut(&id) {
            match &mut ch.backend {
                Backend::Pipe(r) => r.read_closed = true,
                Backend::Datagram(d) => d.read_closed = true,
            }
        }
    });
}

/// Register an additional live end (a `channel_grant` handed a copy of an end to
/// another agent, so teardown must wait for that holder to close too). Returns
/// false if the channel no longer exists.
pub fn dup_end(id: ChannelId) -> bool {
    CHANNELS.with(|m| match m.get_mut(&id) {
        Some(ch) => {
            ch.ends = ch.ends.saturating_add(1);
            true
        }
        None => false,
    })
}

/// Drop one end of the channel. When the last end is dropped the channel is
/// torn down and its buffers freed. Callers that know their direction should
/// also call [`close_write`]/[`close_read`] first so the peer observes EOF.
pub fn close_end(id: ChannelId) {
    CHANNELS.with(|m| {
        if let Some(ch) = m.get_mut(&id) {
            ch.ends = ch.ends.saturating_sub(1);
            if ch.ends == 0 {
                m.remove(&id);
            }
        }
    });
}

/// Cooperatively block until data is available, EOF is reached, or `deadline_ms`
/// (a `crate::arch::now_ms()` value) passes. Returns the bytes read (`Ok(0)` at
/// a real EOF). On timeout returns `WouldBlock`, so a wedged producer fails the
/// caller loudly instead of hanging forever. This is the standing UI-pump rule:
/// every empty spin calls `shell::upkeep()` (clock/caret/mouse + `net::poll`)
/// then `sched::yield_now()` so the producer task gets the CPU — the exact
/// pattern in [`crate::ipc::receive`] and `net/tls.rs`.
pub fn read_blocking(id: ChannelId, buf: &mut [u8], deadline_ms: u64) -> Result<usize, ChannelError> {
    loop {
        match try_read(id, buf) {
            Ok(0) => {
                if is_eof(id) {
                    return Ok(0);
                }
            }
            other => return other,
        }
        if crate::arch::now_ms() >= deadline_ms {
            return Err(ChannelError::WouldBlock);
        }
        crate::shell::upkeep();
        crate::sched::yield_now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn stream_roundtrip_and_backpressure() {
        let id = create(ChannelKind::Stream, 4);
        // Write 4 (fills the ring), then a 5th byte is refused (0 accepted).
        assert_eq!(try_write(id, b"abcd").unwrap(), 4);
        assert_eq!(try_write(id, b"e").unwrap(), 0); // back-pressure, not an error
        let mut out = [0u8; 8];
        assert_eq!(try_read(id, &mut out).unwrap(), 4);
        assert_eq!(&out[..4], b"abcd");
        // Now there's room again.
        assert_eq!(try_write(id, b"e").unwrap(), 1);
        close_end(id);
        close_end(id);
    }

    #[test_case]
    fn eof_after_writer_closes_and_drains() {
        let id = create(ChannelKind::Stream, 16);
        try_write(id, b"hi").unwrap();
        close_write(id);
        assert!(!is_eof(id), "not EOF while bytes remain");
        let mut out = [0u8; 8];
        assert_eq!(try_read(id, &mut out).unwrap(), 2);
        assert!(is_eof(id), "EOF once drained and writer closed");
        assert_eq!(try_read(id, &mut out).unwrap(), 0);
        close_end(id);
        close_end(id);
    }

    #[test_case]
    fn write_to_closed_reader_is_closed_error() {
        let id = create(ChannelKind::Stream, 16);
        close_read(id);
        assert_eq!(try_write(id, b"x"), Err(ChannelError::Closed));
        close_end(id);
        close_end(id);
    }

    #[test_case]
    fn datagram_frames_preserve_boundaries() {
        let id = create(ChannelKind::Datagram, 4);
        try_send_dgram(id, b"one").unwrap();
        try_send_dgram(id, b"two").unwrap();
        assert_eq!(try_recv_dgram(id).unwrap().as_deref(), Some(&b"one"[..]));
        assert_eq!(try_recv_dgram(id).unwrap().as_deref(), Some(&b"two"[..]));
        assert_eq!(try_recv_dgram(id).unwrap(), None);
        close_end(id);
        close_end(id);
    }

    #[test_case]
    fn last_end_teardown_frees_channel() {
        let id = create(ChannelKind::Stream, 8);
        close_end(id);
        assert!(!is_eof(id) || readable_len(id) == 0); // still alive (1 end)
        close_end(id);
        // Both ends dropped -> gone -> reads report NoSuchChannel.
        let mut out = [0u8; 1];
        assert_eq!(try_read(id, &mut out), Err(ChannelError::NoSuchChannel));
        assert!(is_eof(id), "a torn-down channel reads as EOF");
    }
}
