//! The connection protocol (RFC 4254): channels, and the requests that turn a
//! channel into a shell, a command, or a forwarded TCP connection.
//!
//! Pure message building and parsing. One channel type carries everything —
//! `session` for a shell/command (which is how `git-upload-pack` runs), and
//! `direct-tcpip` for a `-L`-style forward — so the same window bookkeeping
//! serves all of them.
//!
//! **Flow control is the part that hangs rather than fails.** Each side grants
//! the other a byte window and must not send past it; the receiver returns
//! credit with `SSH_MSG_CHANNEL_WINDOW_ADJUST` as it consumes. A client that
//! never adjusts works perfectly until it has received exactly one window's
//! worth — 2 MiB by default, which no test fixture reaches and every real
//! `git clone` does. So [`Channel::consume`] returns the adjustment to send and
//! is deliberately awkward to ignore.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use super::wire::{Reader, Writer};

pub const SSH_MSG_GLOBAL_REQUEST: u8 = 80;
pub const SSH_MSG_REQUEST_SUCCESS: u8 = 81;
pub const SSH_MSG_REQUEST_FAILURE: u8 = 82;
pub const SSH_MSG_CHANNEL_OPEN: u8 = 90;
pub const SSH_MSG_CHANNEL_OPEN_CONFIRMATION: u8 = 91;
pub const SSH_MSG_CHANNEL_OPEN_FAILURE: u8 = 92;
pub const SSH_MSG_CHANNEL_WINDOW_ADJUST: u8 = 93;
pub const SSH_MSG_CHANNEL_DATA: u8 = 94;
pub const SSH_MSG_CHANNEL_EXTENDED_DATA: u8 = 95;
pub const SSH_MSG_CHANNEL_EOF: u8 = 96;
pub const SSH_MSG_CHANNEL_CLOSE: u8 = 97;
pub const SSH_MSG_CHANNEL_REQUEST: u8 = 98;
pub const SSH_MSG_CHANNEL_SUCCESS: u8 = 99;
pub const SSH_MSG_CHANNEL_FAILURE: u8 = 100;

/// `SSH_EXTENDED_DATA_STDERR` — the only extended data type defined.
pub const EXTENDED_DATA_STDERR: u32 = 1;

/// Initial window we grant the peer. 2 MiB, matching OpenSSH.
pub const INITIAL_WINDOW: u32 = 2 * 1024 * 1024;
/// Largest packet we are willing to receive on a channel.
pub const MAX_PACKET: u32 = 32 * 1024;
/// Return credit once this much of the window has been consumed, rather than
/// after every byte — an adjust per data packet doubles the packet count.
pub const ADJUST_THRESHOLD: u32 = INITIAL_WINDOW / 2;

/// Our side of one channel's bookkeeping.
#[derive(Clone, Debug)]
pub struct Channel {
    /// Our channel number, which the peer uses to address us.
    pub local_id: u32,
    /// The peer's number, which we use to address them. Set on confirmation.
    pub remote_id: u32,
    /// How many bytes we may still send.
    pub send_window: u32,
    /// The largest single data packet the peer accepts.
    pub send_max_packet: u32,
    /// How many bytes the peer may still send us.
    pub recv_window: u32,
    /// Consumed but not yet credited back.
    pending_adjust: u32,
    pub eof_received: bool,
    pub closed: bool,
    /// The command's exit status, once the peer reports one.
    pub exit_status: Option<u32>,
}

impl Channel {
    pub fn new(local_id: u32) -> Self {
        Self {
            local_id,
            remote_id: 0,
            send_window: 0,
            send_max_packet: 0,
            recv_window: INITIAL_WINDOW,
            pending_adjust: 0,
            eof_received: false,
            closed: false,
            exit_status: None,
        }
    }

    /// Record a peer's open confirmation.
    pub fn confirm(&mut self, remote_id: u32, window: u32, max_packet: u32) {
        self.remote_id = remote_id;
        self.send_window = window;
        // A peer that advertises an absurd maximum would have us build packets
        // the transport cannot frame; clamp to something we can actually send.
        self.send_max_packet = max_packet.clamp(1, MAX_PACKET);
    }

    /// The peer granted us more room to send.
    pub fn grant(&mut self, extra: u32) {
        self.send_window = self.send_window.saturating_add(extra);
    }

    /// How much of `want` we may send right now, respecting both the window and
    /// the peer's maximum packet size.
    pub fn sendable(&self, want: usize) -> usize {
        let cap = self.send_window.min(self.send_max_packet) as usize;
        want.min(cap)
    }

    /// Account for data we sent.
    pub fn sent(&mut self, n: usize) {
        self.send_window = self.send_window.saturating_sub(n as u32);
    }

    /// Account for data we received and hand back the credit to return, if it
    /// has reached the threshold. **Ignoring this stalls the connection** once a
    /// full window has been received.
    pub fn consume(&mut self, n: usize) -> Option<u32> {
        let n = n as u32;
        self.recv_window = self.recv_window.saturating_sub(n);
        self.pending_adjust = self.pending_adjust.saturating_add(n);
        if self.pending_adjust >= ADJUST_THRESHOLD {
            let give = self.pending_adjust;
            self.pending_adjust = 0;
            self.recv_window = self.recv_window.saturating_add(give);
            Some(give)
        } else {
            None
        }
    }
}

/// `SSH_MSG_CHANNEL_OPEN` for an interactive/command session.
pub fn open_session(local_id: u32) -> Vec<u8> {
    let mut w = Writer::msg(SSH_MSG_CHANNEL_OPEN);
    w.put_str("session");
    w.put_u32(local_id);
    w.put_u32(INITIAL_WINDOW);
    w.put_u32(MAX_PACKET);
    w.into_vec()
}

/// `SSH_MSG_CHANNEL_OPEN` for a `-L`-style forward: the server connects onward
/// to `host:port` on our behalf.
///
/// The originator fields are informational, but the *lengths* are not optional —
/// a server parses them and will refuse a short message.
pub fn open_direct_tcpip(
    local_id: u32,
    host: &str,
    port: u16,
    origin_host: &str,
    origin_port: u16,
) -> Vec<u8> {
    let mut w = Writer::msg(SSH_MSG_CHANNEL_OPEN);
    w.put_str("direct-tcpip");
    w.put_u32(local_id);
    w.put_u32(INITIAL_WINDOW);
    w.put_u32(MAX_PACKET);
    w.put_str(host);
    w.put_u32(port as u32);
    w.put_str(origin_host);
    w.put_u32(origin_port as u32);
    w.into_vec()
}

/// `tcpip-forward` — ask the server to listen on its side (`-R`).
pub fn global_tcpip_forward(bind: &str, port: u16, want_reply: bool) -> Vec<u8> {
    let mut w = Writer::msg(SSH_MSG_GLOBAL_REQUEST);
    w.put_str("tcpip-forward");
    w.put_bool(want_reply);
    w.put_str(bind);
    w.put_u32(port as u32);
    w.into_vec()
}

/// `exec` — run one command. This is how a git fetch starts.
pub fn request_exec(remote_id: u32, command: &str, want_reply: bool) -> Vec<u8> {
    let mut w = Writer::msg(SSH_MSG_CHANNEL_REQUEST);
    w.put_u32(remote_id);
    w.put_str("exec");
    w.put_bool(want_reply);
    w.put_str(command);
    w.into_vec()
}

/// `shell` — start the user's login shell.
pub fn request_shell(remote_id: u32, want_reply: bool) -> Vec<u8> {
    let mut w = Writer::msg(SSH_MSG_CHANNEL_REQUEST);
    w.put_u32(remote_id);
    w.put_str("shell");
    w.put_bool(want_reply);
    w.into_vec()
}

/// `subsystem` — e.g. `sftp`.
pub fn request_subsystem(remote_id: u32, name: &str, want_reply: bool) -> Vec<u8> {
    let mut w = Writer::msg(SSH_MSG_CHANNEL_REQUEST);
    w.put_u32(remote_id);
    w.put_str("subsystem");
    w.put_bool(want_reply);
    w.put_str(name);
    w.into_vec()
}

/// `pty-req` — allocate a terminal, so a remote shell line-edits and colours.
///
/// The terminal modes are a list of `opcode, u32 value` pairs ended by opcode 0,
/// carried inside a `string`. An **empty** modes string is legal and means "all
/// defaults"; omitting the string entirely is not, and a server rejects it.
pub fn request_pty(
    remote_id: u32,
    term: &str,
    cols: u32,
    rows: u32,
    want_reply: bool,
) -> Vec<u8> {
    let mut w = Writer::msg(SSH_MSG_CHANNEL_REQUEST);
    w.put_u32(remote_id);
    w.put_str("pty-req");
    w.put_bool(want_reply);
    w.put_str(term);
    w.put_u32(cols);
    w.put_u32(rows);
    w.put_u32(0); // width in pixels
    w.put_u32(0); // height in pixels
    let mut modes = Writer::new();
    modes.put_u8(0); // TTY_OP_END
    w.put_string(modes.as_slice());
    w.into_vec()
}

/// `window-change` — tell the remote its terminal was resized.
pub fn request_window_change(remote_id: u32, cols: u32, rows: u32) -> Vec<u8> {
    let mut w = Writer::msg(SSH_MSG_CHANNEL_REQUEST);
    w.put_u32(remote_id);
    w.put_str("window-change");
    w.put_bool(false); // never replied to
    w.put_u32(cols);
    w.put_u32(rows);
    w.put_u32(0);
    w.put_u32(0);
    w.into_vec()
}

pub fn data(remote_id: u32, bytes: &[u8]) -> Vec<u8> {
    let mut w = Writer::msg(SSH_MSG_CHANNEL_DATA);
    w.put_u32(remote_id);
    w.put_string(bytes);
    w.into_vec()
}

pub fn window_adjust(remote_id: u32, extra: u32) -> Vec<u8> {
    let mut w = Writer::msg(SSH_MSG_CHANNEL_WINDOW_ADJUST);
    w.put_u32(remote_id);
    w.put_u32(extra);
    w.into_vec()
}

pub fn eof(remote_id: u32) -> Vec<u8> {
    let mut w = Writer::msg(SSH_MSG_CHANNEL_EOF);
    w.put_u32(remote_id);
    w.into_vec()
}

pub fn close(remote_id: u32) -> Vec<u8> {
    let mut w = Writer::msg(SSH_MSG_CHANNEL_CLOSE);
    w.put_u32(remote_id);
    w.into_vec()
}

/// A parsed inbound connection-protocol message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    OpenConfirmation {
        local_id: u32,
        remote_id: u32,
        window: u32,
        max_packet: u32,
    },
    OpenFailure {
        local_id: u32,
        reason: u32,
        description: String,
    },
    WindowAdjust {
        local_id: u32,
        extra: u32,
    },
    Data {
        local_id: u32,
        data: Vec<u8>,
    },
    ExtendedData {
        local_id: u32,
        kind: u32,
        data: Vec<u8>,
    },
    Eof {
        local_id: u32,
    },
    Close {
        local_id: u32,
    },
    ExitStatus {
        local_id: u32,
        status: u32,
    },
    /// A channel request we do not handle; the peer may want a reply.
    Request {
        local_id: u32,
        kind: String,
        want_reply: bool,
    },
    Success {
        local_id: u32,
    },
    Failure {
        local_id: u32,
    },
    /// Anything else, kept so the driver can log rather than desynchronise.
    Other(u8),
}

/// Parse one connection-protocol payload.
pub fn parse(payload: &[u8]) -> Option<Event> {
    let mut r = Reader::new(payload);
    let kind = r.u8()?;
    Some(match kind {
        SSH_MSG_CHANNEL_OPEN_CONFIRMATION => Event::OpenConfirmation {
            local_id: r.u32()?,
            remote_id: r.u32()?,
            window: r.u32()?,
            max_packet: r.u32()?,
        },
        SSH_MSG_CHANNEL_OPEN_FAILURE => Event::OpenFailure {
            local_id: r.u32()?,
            reason: r.u32()?,
            description: r.utf8().unwrap_or("").to_string(),
        },
        SSH_MSG_CHANNEL_WINDOW_ADJUST => Event::WindowAdjust {
            local_id: r.u32()?,
            extra: r.u32()?,
        },
        SSH_MSG_CHANNEL_DATA => Event::Data {
            local_id: r.u32()?,
            data: r.string()?.to_vec(),
        },
        SSH_MSG_CHANNEL_EXTENDED_DATA => Event::ExtendedData {
            local_id: r.u32()?,
            kind: r.u32()?,
            data: r.string()?.to_vec(),
        },
        SSH_MSG_CHANNEL_EOF => Event::Eof { local_id: r.u32()? },
        SSH_MSG_CHANNEL_CLOSE => Event::Close { local_id: r.u32()? },
        SSH_MSG_CHANNEL_SUCCESS => Event::Success { local_id: r.u32()? },
        SSH_MSG_CHANNEL_FAILURE => Event::Failure { local_id: r.u32()? },
        SSH_MSG_CHANNEL_REQUEST => {
            let local_id = r.u32()?;
            let name = r.utf8()?;
            let want_reply = r.bool()?;
            // `exit-status` is the one channel request a client must understand:
            // it is how a remote command reports whether it worked.
            if name == "exit-status" {
                Event::ExitStatus {
                    local_id,
                    status: r.u32()?,
                }
            } else {
                Event::Request {
                    local_id,
                    kind: name.to_string(),
                    want_reply,
                }
            }
        }
        other => Event::Other(other),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every message we build parses back to what it said.
    #[test_case]
    fn channel_messages_round_trip() {
        assert_eq!(
            parse(&data(7, b"payload")),
            Some(Event::Data {
                local_id: 7,
                data: b"payload".to_vec()
            })
        );
        assert_eq!(parse(&eof(3)), Some(Event::Eof { local_id: 3 }));
        assert_eq!(parse(&close(3)), Some(Event::Close { local_id: 3 }));
        assert_eq!(
            parse(&window_adjust(2, 4096)),
            Some(Event::WindowAdjust {
                local_id: 2,
                extra: 4096
            })
        );
    }

    /// `exit-status` is decoded specially — it is how a command reports failure,
    /// and treating it as an ordinary unhandled request loses the exit code.
    #[test_case]
    fn exit_status_is_decoded_not_ignored() {
        let mut w = Writer::msg(SSH_MSG_CHANNEL_REQUEST);
        w.put_u32(1);
        w.put_str("exit-status");
        w.put_bool(false);
        w.put_u32(128);
        assert_eq!(
            parse(&w.into_vec()),
            Some(Event::ExitStatus {
                local_id: 1,
                status: 128
            })
        );

        // An unknown request keeps its name and reply flag so the driver can
        // answer it rather than desynchronise.
        let mut w = Writer::msg(SSH_MSG_CHANNEL_REQUEST);
        w.put_u32(1);
        w.put_str("keepalive@openssh.com");
        w.put_bool(true);
        assert_eq!(
            parse(&w.into_vec()),
            Some(Event::Request {
                local_id: 1,
                kind: "keepalive@openssh.com".to_string(),
                want_reply: true
            })
        );
    }

    /// A truncated message is refused, never half-applied.
    #[test_case]
    fn truncated_messages_are_refused() {
        assert!(parse(&[]).is_none());
        assert!(parse(&[SSH_MSG_CHANNEL_DATA]).is_none());
        assert!(parse(&[SSH_MSG_CHANNEL_DATA, 0, 0, 0, 1]).is_none(), "no data string");
        // A data length that overruns the payload is refused, not clamped.
        assert!(parse(&[SSH_MSG_CHANNEL_DATA, 0, 0, 0, 1, 0xff, 0xff, 0xff, 0xff]).is_none());
    }

    /// **The window is what stalls a clone.** Credit is returned once, at the
    /// threshold — not per packet, and not never.
    #[test_case]
    fn the_receive_window_returns_credit_at_the_threshold() {
        let mut ch = Channel::new(0);
        assert_eq!(ch.recv_window, INITIAL_WINDOW);

        // Small reads accumulate without an adjust — one adjust per data packet
        // would double the packet count for no benefit.
        assert_eq!(ch.consume(1024), None);
        assert_eq!(ch.recv_window, INITIAL_WINDOW - 1024);

        // Crossing the threshold returns everything consumed so far, and the
        // window goes back up by exactly that much.
        let before = ch.recv_window;
        let give = ch.consume(ADJUST_THRESHOLD as usize).expect("credit is returned");
        assert_eq!(give, 1024 + ADJUST_THRESHOLD);
        assert_eq!(ch.recv_window, before - ADJUST_THRESHOLD + give);

        // And a long transfer never runs the window to zero.
        let mut ch = Channel::new(0);
        let mut credited = 0u64;
        for _ in 0..1000 {
            if let Some(g) = ch.consume(8192) {
                credited += g as u64;
            }
            assert!(ch.recv_window > 0, "the receive window must never reach zero");
        }
        assert!(credited > 0, "a long transfer must have returned credit");
    }

    /// The send window and the peer's packet limit both bound what we may send.
    #[test_case]
    fn sending_respects_both_the_window_and_the_packet_limit() {
        let mut ch = Channel::new(0);
        ch.confirm(5, 100, 32);
        assert_eq!(ch.remote_id, 5);
        assert_eq!(ch.sendable(1000), 32, "the packet limit binds first");
        ch.sent(32);
        assert_eq!(ch.send_window, 68);

        // When the window is the tighter bound, it wins.
        ch.confirm(5, 10, 1024);
        assert_eq!(ch.sendable(1000), 10);

        // An exhausted window sends nothing — and must not underflow.
        ch.sent(10);
        assert_eq!(ch.send_window, 0);
        assert_eq!(ch.sendable(1000), 0);
        ch.sent(50);
        assert_eq!(ch.send_window, 0, "the window must saturate, not wrap");

        // A grant reopens it.
        ch.grant(4096);
        assert_eq!(ch.sendable(1000), 1000);
    }

    /// A peer advertising an absurd packet size is clamped to what we can frame.
    #[test_case]
    fn an_absurd_peer_packet_size_is_clamped() {
        let mut ch = Channel::new(0);
        ch.confirm(1, INITIAL_WINDOW, u32::MAX);
        assert_eq!(ch.send_max_packet, MAX_PACKET);
        assert!(ch.sendable(usize::MAX) <= MAX_PACKET as usize);
        // Zero is not a usable packet size either.
        ch.confirm(1, INITIAL_WINDOW, 0);
        assert_eq!(ch.send_max_packet, 1);
    }

    /// The two open messages carry the fields a server parses, in order.
    #[test_case]
    fn open_messages_carry_their_fields() {
        let m = open_session(9);
        let mut r = Reader::new(&m);
        assert_eq!(r.u8(), Some(SSH_MSG_CHANNEL_OPEN));
        assert_eq!(r.utf8(), Some("session"));
        assert_eq!(r.u32(), Some(9));
        assert_eq!(r.u32(), Some(INITIAL_WINDOW));
        assert_eq!(r.u32(), Some(MAX_PACKET));
        assert!(r.is_empty());

        let m = open_direct_tcpip(4, "db.internal", 5432, "127.0.0.1", 51000);
        let mut r = Reader::new(&m);
        assert_eq!(r.u8(), Some(SSH_MSG_CHANNEL_OPEN));
        assert_eq!(r.utf8(), Some("direct-tcpip"));
        assert_eq!(r.u32(), Some(4));
        let _ = (r.u32(), r.u32());
        assert_eq!(r.utf8(), Some("db.internal"));
        assert_eq!(r.u32(), Some(5432));
        assert_eq!(r.utf8(), Some("127.0.0.1"), "the originator is not optional");
        assert_eq!(r.u32(), Some(51000));
        assert!(r.is_empty());
    }

    /// `pty-req` ends with a modes **string**, which may be empty but must be
    /// present — a server rejects the message without it.
    #[test_case]
    fn pty_request_carries_a_terminal_modes_string() {
        let m = request_pty(1, "xterm-256color", 120, 40, true);
        let mut r = Reader::new(&m);
        assert_eq!(r.u8(), Some(SSH_MSG_CHANNEL_REQUEST));
        assert_eq!(r.u32(), Some(1));
        assert_eq!(r.utf8(), Some("pty-req"));
        assert_eq!(r.bool(), Some(true));
        assert_eq!(r.utf8(), Some("xterm-256color"));
        assert_eq!(r.u32(), Some(120));
        assert_eq!(r.u32(), Some(40));
        assert_eq!(r.u32(), Some(0));
        assert_eq!(r.u32(), Some(0));
        let modes = r.string().expect("the modes string is mandatory");
        assert_eq!(modes, &[0], "TTY_OP_END terminates the (empty) mode list");
        assert!(r.is_empty());
    }

    /// `exec` carries the command a git fetch depends on.
    #[test_case]
    fn exec_carries_the_command() {
        let m = request_exec(2, "git-upload-pack 'user/repo.git'", true);
        let mut r = Reader::new(&m);
        assert_eq!(r.u8(), Some(SSH_MSG_CHANNEL_REQUEST));
        assert_eq!(r.u32(), Some(2));
        assert_eq!(r.utf8(), Some("exec"));
        assert_eq!(r.bool(), Some(true));
        assert_eq!(r.utf8(), Some("git-upload-pack 'user/repo.git'"));
    }
}
