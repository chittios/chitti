//! **SPICE vdagent protocol** — the clipboard channel a hypervisor's graphical
//! window uses, as pure framing over byte slices.
//!
//! This is what `-chardev qemu-vdagent,clipboard=on` speaks, over a
//! virtio-serial port named `com.redhat.spice.0`. Constants and struct layouts
//! come from `spice-protocol`'s `vd_agent.h` and QEMU's `ui/vdagent.c`, both
//! **fetched** — the same rule the 9P codec follows, and for the same reason:
//! every field here is a plain integer, so a wrong offset produces a valid-
//! looking message rather than an error.
//!
//! ## The trap that dominates this protocol
//!
//! Four of the clipboard messages begin with an **optional 4-byte selection
//! prefix**, present only when `VD_AGENT_CAP_CLIPBOARD_SELECTION` was
//! negotiated — and `VD_AGENT_CLIPBOARD_GRAB` has a *second* optional prefix
//! for `CAP_CLIPBOARD_GRAB_SERIAL`. Get the presence of either wrong and every
//! field after it shifts by four bytes: a `type` reads as a selection id, a
//! grab for UTF-8 text reads as a grab for nothing.
//!
//! QEMU decides the layout from **the guest's** announced capabilities
//! (`have_selection()` reads `vd->caps`, which it fills from our announcement),
//! so the guest controls the wire format. This agent therefore announces
//! neither — one fixed layout, no conditional offsets anywhere. It costs only
//! the ability to address the X11 PRIMARY selection, which does not exist here.
//!
//! `VD_AGENT_CAP_CLIPBOARD_BY_DEMAND` **must** be announced: QEMU's
//! `have_clipboard()` requires it, and without it the clipboard is silently
//! inert while everything else about the connection looks healthy.

use alloc::vec::Vec;

/// `VD_AGENT_PROTOCOL`.
pub const PROTOCOL: u32 = 1;

/// Chunk header ports. The guest is the "client" end of this link.
pub const PORT_CLIENT: u32 = 1;
pub const PORT_SERVER: u32 = 2;

/// `VDIChunkHeader { u32 port; u32 size; }`.
pub const CHUNK_HDR: usize = 8;
/// `VDAgentMessage { u32 protocol; u32 type; u64 opaque; u32 size; }`.
pub const MSG_HDR: usize = 20;

/// QEMU splits outgoing messages into chunks of at most this many bytes
/// (`ui/vdagent.c`), and accepts the same from us.
pub const CHUNK_MAX: usize = 1024;

// --- VDAgentMessageType ---
pub const MSG_CLIPBOARD: u32 = 4;
pub const MSG_ANNOUNCE_CAPABILITIES: u32 = 6;
pub const MSG_CLIPBOARD_GRAB: u32 = 7;
pub const MSG_CLIPBOARD_REQUEST: u32 = 8;
pub const MSG_CLIPBOARD_RELEASE: u32 = 9;

// --- clipboard data formats ---
pub const FMT_NONE: u32 = 0;
pub const FMT_UTF8_TEXT: u32 = 1;

// --- capability bits ---
pub const CAP_CLIPBOARD: u32 = 3;
pub const CAP_CLIPBOARD_BY_DEMAND: u32 = 5;
pub const CAP_CLIPBOARD_SELECTION: u32 = 6;
pub const CAP_CLIPBOARD_GRAB_SERIAL: u32 = 17;

/// Exactly what this agent claims. Deliberately minimal — see the module doc:
/// every extra capability here adds a conditional field to the wire format.
pub fn our_caps() -> u32 {
    (1 << CAP_CLIPBOARD) | (1 << CAP_CLIPBOARD_BY_DEMAND)
}

/// A clipboard payload larger than this is refused rather than sent. The
/// channel is a byte pipe with no flow control that we can block on, and the
/// whole message is buffered on both sides; a runaway paste should fail
/// visibly, not consume the heap.
pub const MAX_CLIPBOARD: usize = 1024 * 1024;

/// Frame `payload` as a `VDAgentMessage` of `typ`, split into chunk-framed
/// pieces ready to write to the port.
pub fn encode(typ: u32, payload: &[u8]) -> Vec<u8> {
    let mut msg = Vec::with_capacity(MSG_HDR + payload.len());
    msg.extend_from_slice(&PROTOCOL.to_le_bytes());
    msg.extend_from_slice(&typ.to_le_bytes());
    msg.extend_from_slice(&0u64.to_le_bytes()); // opaque
    msg.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    msg.extend_from_slice(payload);

    let mut out = Vec::with_capacity(msg.len() + CHUNK_HDR * (msg.len() / CHUNK_MAX + 1));
    let mut off = 0;
    while off < msg.len() {
        let take = (msg.len() - off).min(CHUNK_MAX);
        out.extend_from_slice(&PORT_CLIENT.to_le_bytes());
        out.extend_from_slice(&(take as u32).to_le_bytes());
        out.extend_from_slice(&msg[off..off + take]);
        off += take;
    }
    out
}

/// Announce our capabilities. `request` asks the host to announce its own back.
pub fn announce(request: bool) -> Vec<u8> {
    let mut p = Vec::with_capacity(8);
    p.extend_from_slice(&(request as u32).to_le_bytes());
    p.extend_from_slice(&our_caps().to_le_bytes());
    encode(MSG_ANNOUNCE_CAPABILITIES, &p)
}

/// Tell the host we hold new clipboard contents of these formats.
pub fn grab(formats: &[u32]) -> Vec<u8> {
    let mut p = Vec::with_capacity(formats.len() * 4);
    for f in formats {
        p.extend_from_slice(&f.to_le_bytes());
    }
    encode(MSG_CLIPBOARD_GRAB, &p)
}

/// Ask the host for its clipboard in `format`.
pub fn request(format: u32) -> Vec<u8> {
    encode(MSG_CLIPBOARD_REQUEST, &format.to_le_bytes())
}

/// Deliver clipboard contents in `format`.
pub fn clipboard(format: u32, data: &[u8]) -> Vec<u8> {
    let mut p = Vec::with_capacity(4 + data.len());
    p.extend_from_slice(&format.to_le_bytes());
    p.extend_from_slice(data);
    encode(MSG_CLIPBOARD, &p)
}

/// Give up ownership of the clipboard.
pub fn release() -> Vec<u8> {
    encode(MSG_CLIPBOARD_RELEASE, &[])
}

/// A decoded message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Incoming {
    /// The host announced its capabilities; `request` means it wants ours.
    Caps { caps: u32, request: bool },
    /// The host has new clipboard contents in these formats.
    Grab(Vec<u32>),
    /// The host wants our clipboard in this format.
    Request(u32),
    /// The host delivered clipboard contents.
    Clipboard { format: u32, data: Vec<u8> },
    /// The host dropped its clipboard.
    Release,
    /// A well-formed message this agent does not act on (mouse, monitors, …).
    /// Reported rather than dropped so a trace shows the link is alive.
    Other(u32),
}

/// Reassembles the chunk stream into whole messages.
///
/// Two levels of framing have to be undone, and they do **not** nest one to
/// one: a chunk is at most 1024 bytes while a message may be megabytes, so a
/// message spans chunks and the `VDAgentMessage` header appears only in the
/// first. Treating each chunk as a message — the obvious reading of the two
/// structs — works perfectly until the first clipboard paste over ~1 KB.
#[derive(Default)]
pub struct Reassembler {
    /// Bytes arrived but not yet forming a whole chunk.
    raw: Vec<u8>,
    /// Chunk payloads accumulated toward the current message.
    msg: Vec<u8>,
}

impl Reassembler {
    pub fn new() -> Reassembler {
        Reassembler::default()
    }

    /// Feed received bytes; returns every whole message they completed.
    pub fn feed(&mut self, bytes: &[u8]) -> Vec<Incoming> {
        self.raw.extend_from_slice(bytes);
        let mut out = Vec::new();
        loop {
            // Peel one whole chunk off the raw stream.
            if self.raw.len() < CHUNK_HDR {
                break;
            }
            let size = u32::from_le_bytes([self.raw[4], self.raw[5], self.raw[6], self.raw[7]]) as usize;
            // A chunk claiming more than any message we would accept is a
            // desynchronised stream; dropping everything is the only way back
            // to a known state, and continuing would allocate on its word.
            if size > MAX_CLIPBOARD + MSG_HDR {
                self.raw.clear();
                self.msg.clear();
                break;
            }
            if self.raw.len() < CHUNK_HDR + size {
                break;
            }
            self.msg.extend_from_slice(&self.raw[CHUNK_HDR..CHUNK_HDR + size]);
            self.raw.drain(..CHUNK_HDR + size);

            // Then take whole messages out of the accumulated payload.
            while self.msg.len() >= MSG_HDR {
                let msg_size =
                    u32::from_le_bytes([self.msg[16], self.msg[17], self.msg[18], self.msg[19]])
                        as usize;
                if msg_size > MAX_CLIPBOARD {
                    self.msg.clear();
                    break;
                }
                if self.msg.len() < MSG_HDR + msg_size {
                    break;
                }
                let typ = u32::from_le_bytes([self.msg[4], self.msg[5], self.msg[6], self.msg[7]]);
                let body: Vec<u8> = self.msg[MSG_HDR..MSG_HDR + msg_size].to_vec();
                self.msg.drain(..MSG_HDR + msg_size);
                if let Some(m) = decode(typ, &body) {
                    out.push(m);
                }
            }
        }
        out
    }
}

/// Decode one message body. `None` for a body too short for its type — which
/// is a malformed message, not an unknown one.
fn decode(typ: u32, body: &[u8]) -> Option<Incoming> {
    let u32_at = |i: usize| -> Option<u32> {
        let b = body.get(i..i + 4)?;
        Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    };
    match typ {
        MSG_ANNOUNCE_CAPABILITIES => Some(Incoming::Caps {
            request: u32_at(0)? != 0,
            // Only the first capability word is read: every bit this agent
            // cares about lives in it, and a host announcing more words is
            // normal rather than an error.
            caps: u32_at(4)?,
        }),
        MSG_CLIPBOARD_GRAB => {
            let mut fmts = Vec::new();
            let mut i = 0;
            while i + 4 <= body.len() {
                fmts.push(u32_at(i)?);
                i += 4;
            }
            Some(Incoming::Grab(fmts))
        }
        MSG_CLIPBOARD_REQUEST => Some(Incoming::Request(u32_at(0)?)),
        MSG_CLIPBOARD => Some(Incoming::Clipboard {
            format: u32_at(0)?,
            data: body.get(4..)?.to_vec(),
        }),
        MSG_CLIPBOARD_RELEASE => Some(Incoming::Release),
        other => Some(Incoming::Other(other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Round-trip helper: frame a message as the *host* would (server port) and
    /// feed it through the reassembler.
    fn host_frame(typ: u32, payload: &[u8]) -> Vec<u8> {
        let mut msg = Vec::new();
        msg.extend_from_slice(&PROTOCOL.to_le_bytes());
        msg.extend_from_slice(&typ.to_le_bytes());
        msg.extend_from_slice(&0u64.to_le_bytes());
        msg.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        msg.extend_from_slice(payload);
        let mut out = Vec::new();
        let mut off = 0;
        while off < msg.len() {
            let take = (msg.len() - off).min(CHUNK_MAX);
            out.extend_from_slice(&PORT_SERVER.to_le_bytes());
            out.extend_from_slice(&(take as u32).to_le_bytes());
            out.extend_from_slice(&msg[off..off + take]);
            off += take;
        }
        out
    }

    #[test_case]
    fn header_sizes_match_the_c_structs() {
        // VDIChunkHeader is 8 bytes; VDAgentMessage is 4+4+8+4 = 20. The u64
        // `opaque` is the field that makes this 20 and not 16, and a 16-byte
        // header shifts every payload by four.
        assert_eq!(CHUNK_HDR, 8);
        assert_eq!(MSG_HDR, 20);
        let m = encode(MSG_CLIPBOARD_RELEASE, &[]);
        assert_eq!(m.len(), CHUNK_HDR + MSG_HDR);
        assert_eq!(u32::from_le_bytes([m[0], m[1], m[2], m[3]]), PORT_CLIENT);
        assert_eq!(u32::from_le_bytes([m[4], m[5], m[6], m[7]]), MSG_HDR as u32);
        assert_eq!(u32::from_le_bytes([m[8], m[9], m[10], m[11]]), PROTOCOL);
        assert_eq!(u32::from_le_bytes([m[12], m[13], m[14], m[15]]), MSG_CLIPBOARD_RELEASE);
    }

    #[test_case]
    fn we_announce_no_capability_that_adds_a_conditional_field() {
        // This is the load-bearing decision of the whole module: QEMU derives
        // the presence of the 4-byte selection prefix, and of the grab serial,
        // from OUR announcement. Claiming either without also encoding it
        // shifts every later field by four bytes and reads as a grab for a
        // format nobody supports.
        let caps = our_caps();
        assert!(caps & (1 << CAP_CLIPBOARD) != 0);
        // Required, or QEMU's have_clipboard() is false and the clipboard is
        // silently inert while the link looks healthy.
        assert!(caps & (1 << CAP_CLIPBOARD_BY_DEMAND) != 0);
        assert_eq!(caps & (1 << CAP_CLIPBOARD_SELECTION), 0);
        assert_eq!(caps & (1 << CAP_CLIPBOARD_GRAB_SERIAL), 0);

        // So a grab is exactly its formats, with no prefix.
        let g = grab(&[FMT_UTF8_TEXT]);
        assert_eq!(g.len(), CHUNK_HDR + MSG_HDR + 4);
        let fmt = u32::from_le_bytes([g[28], g[29], g[30], g[31]]);
        assert_eq!(fmt, FMT_UTF8_TEXT);
        // And a request is exactly its type.
        let r = request(FMT_UTF8_TEXT);
        assert_eq!(r.len(), CHUNK_HDR + MSG_HDR + 4);
    }

    #[test_case]
    fn a_message_larger_than_a_chunk_is_split_and_reassembled() {
        // The two framings do NOT nest one to one: a chunk caps at 1024 bytes
        // while a message may be megabytes, and the VDAgentMessage header is
        // only in the first chunk. Treating a chunk as a message works right
        // up until the first paste over ~1 KB.
        let text: Vec<u8> = (0..5000u32).map(|i| b'a' + (i % 26) as u8).collect();
        let wire = clipboard(FMT_UTF8_TEXT, &text);
        // 20 + 4 + 5000 = 5024 bytes of message -> 5 chunks (4x1024 + 928).
        let chunks = wire.len() - (MSG_HDR + 4 + text.len());
        assert_eq!(chunks, 5 * CHUNK_HDR, "expected 5 chunk headers");

        // Reassemble the same bytes (the encoder writes the client port; the
        // reassembler does not care which end sent them).
        let mut r = Reassembler::new();
        let got = r.feed(&wire);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0], Incoming::Clipboard { format: FMT_UTF8_TEXT, data: text });
    }

    #[test_case]
    fn bytes_arriving_one_at_a_time_still_reassemble() {
        // A virtio-serial read returns whatever the host happened to write, so
        // a chunk header can be split across two reads.
        let wire = host_frame(MSG_CLIPBOARD, &{
            let mut p = vec![];
            p.extend_from_slice(&FMT_UTF8_TEXT.to_le_bytes());
            p.extend_from_slice(b"split me");
            p
        });
        let mut r = Reassembler::new();
        let mut got = Vec::new();
        for b in &wire {
            got.extend(r.feed(&[*b]));
        }
        assert_eq!(got.len(), 1);
        assert_eq!(
            got[0],
            Incoming::Clipboard { format: FMT_UTF8_TEXT, data: b"split me".to_vec() }
        );
    }

    #[test_case]
    fn several_messages_in_one_read_all_come_out() {
        let mut wire = host_frame(MSG_CLIPBOARD_GRAB, &FMT_UTF8_TEXT.to_le_bytes());
        wire.extend(host_frame(MSG_CLIPBOARD_REQUEST, &FMT_UTF8_TEXT.to_le_bytes()));
        wire.extend(host_frame(MSG_CLIPBOARD_RELEASE, &[]));
        let mut r = Reassembler::new();
        let got = r.feed(&wire);
        assert_eq!(
            got,
            vec![
                Incoming::Grab(vec![FMT_UTF8_TEXT]),
                Incoming::Request(FMT_UTF8_TEXT),
                Incoming::Release
            ]
        );
    }

    #[test_case]
    fn caps_decode_with_the_request_flag_separate_from_the_bits() {
        // `request` is its own u32 BEFORE the capability words; reading the
        // first word as the caps gives 0 or 1 — which happens to look like
        // "mouse state only" rather than like an error.
        let mut p = Vec::new();
        p.extend_from_slice(&1u32.to_le_bytes()); // request
        p.extend_from_slice(&((1u32 << CAP_CLIPBOARD) | (1 << CAP_CLIPBOARD_BY_DEMAND)).to_le_bytes());
        let wire = host_frame(MSG_ANNOUNCE_CAPABILITIES, &p);
        let mut r = Reassembler::new();
        let got = r.feed(&wire);
        assert_eq!(
            got[0],
            Incoming::Caps {
                request: true,
                caps: (1 << CAP_CLIPBOARD) | (1 << CAP_CLIPBOARD_BY_DEMAND)
            }
        );
    }

    #[test_case]
    fn a_desynchronised_stream_is_dropped_rather_than_allocated_on() {
        // A chunk header claiming a gigabyte must not become a gigabyte
        // allocation in a kernel with a first-fit heap.
        let mut wire = Vec::new();
        wire.extend_from_slice(&PORT_SERVER.to_le_bytes());
        wire.extend_from_slice(&(u32::MAX).to_le_bytes());
        wire.extend_from_slice(&[0u8; 32]);
        let mut r = Reassembler::new();
        assert!(r.feed(&wire).is_empty());
        // Having reset, it recovers on the next well-formed message.
        let good = host_frame(MSG_CLIPBOARD_RELEASE, &[]);
        assert_eq!(r.feed(&good), vec![Incoming::Release]);
    }

    #[test_case]
    fn an_unknown_message_type_is_reported_not_fatal() {
        // The host sends mouse and monitor messages on this same link; they
        // must not desynchronise the stream or look like an error.
        let wire = host_frame(1 /* VD_AGENT_MOUSE_STATE */, &[0u8; 12]);
        let mut r = Reassembler::new();
        assert_eq!(r.feed(&wire), vec![Incoming::Other(1)]);
    }
}
