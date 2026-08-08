//! **RNDIS** — Microsoft's Remote NDIS, the protocol Android USB tethering
//! speaks (and the one QEMU's `usb-net` emulates).
//!
//! This module is the **pure wire layer**: message builders, response parsers,
//! and the per-packet framing. It touches no hardware and no controller state,
//! so all of it is unit-tested off-hardware — which matters more here than for
//! most protocols, because every field in RNDIS is a little-endian `u32` and a
//! wrong offset yields another plausible `u32` rather than an error.
//!
//! ## Shape
//!
//! Control messages ride the **default control pipe** as class requests —
//! `SEND_ENCAPSULATED_COMMAND` (host → device) and `GET_ENCAPSULATED_RESPONSE`
//! (device → host), see [`send_encapsulated_setup`] / [`get_encapsulated_setup`].
//! Data rides the bulk pair, each Ethernet frame wrapped in a 44-byte
//! [`PACKET_HEADER_LEN`] header.
//!
//! Bring-up is: `INITIALIZE` → query `OID_802_3_PERMANENT_ADDRESS` for the MAC →
//! set `OID_GEN_CURRENT_PACKET_FILTER` so the device starts forwarding receives.
//! Skipping the filter is the classic RNDIS failure: everything reports success
//! and **not one frame ever arrives**, because the default filter is zero.
//!
//! ## The three traps, each of which produces a plausible wrong answer
//!
//! 1. **Buffer offsets are relative to byte 8** (the start of `RequestId`), not
//!    to the start of the message. Reading them as absolute lands 8 bytes early
//!    — for a MAC query that is the tail of `Status` followed by four bytes of
//!    the real address, i.e. a well-formed MAC address belonging to nobody.
//! 2. **One bulk transfer may carry several packets.** RNDIS batches up to
//!    `max_packets_per_transfer`, and each packet's `MessageLength` walks to the
//!    next. Treating a transfer as one packet silently drops all but the first,
//!    which looks like heavy packet loss rather than a framing bug.
//! 3. **`DataOffset` is not a constant.** A device may place per-packet info
//!    ahead of the payload, so the payload does not always start at 44. Assuming
//!    it does shifts every received frame by however much info was attached.

use alloc::vec::Vec;

// --- message types (RNDIS 1.0) -------------------------------------------

pub const MSG_PACKET: u32 = 0x0000_0001;
pub const MSG_INITIALIZE: u32 = 0x0000_0002;
pub const MSG_HALT: u32 = 0x0000_0003;
pub const MSG_QUERY: u32 = 0x0000_0004;
pub const MSG_SET: u32 = 0x0000_0005;
pub const MSG_RESET: u32 = 0x0000_0006;
pub const MSG_INDICATE_STATUS: u32 = 0x0000_0007;
pub const MSG_KEEPALIVE: u32 = 0x0000_0008;

/// A completion is its request with the high bit set.
pub const MSG_COMPLETION: u32 = 0x8000_0000;
pub const MSG_INITIALIZE_CMPLT: u32 = MSG_COMPLETION | MSG_INITIALIZE;
pub const MSG_QUERY_CMPLT: u32 = MSG_COMPLETION | MSG_QUERY;
pub const MSG_SET_CMPLT: u32 = MSG_COMPLETION | MSG_SET;
pub const MSG_KEEPALIVE_CMPLT: u32 = MSG_COMPLETION | MSG_KEEPALIVE;

/// `RNDIS_STATUS_SUCCESS`.
pub const STATUS_SUCCESS: u32 = 0x0000_0000;

// --- OIDs ----------------------------------------------------------------

/// Which frames the device should forward to us. **Must be set**, or the device
/// initialises cleanly and receives nothing.
pub const OID_GEN_CURRENT_PACKET_FILTER: u32 = 0x0001_010e;
pub const OID_GEN_MAXIMUM_FRAME_SIZE: u32 = 0x0001_0106;
pub const OID_GEN_LINK_SPEED: u32 = 0x0001_0107;
/// The adapter's burned-in address.
pub const OID_802_3_PERMANENT_ADDRESS: u32 = 0x0101_0101;
pub const OID_802_3_CURRENT_ADDRESS: u32 = 0x0101_0102;

// --- packet filter bits ---------------------------------------------------

pub const PACKET_TYPE_DIRECTED: u32 = 0x0001;
pub const PACKET_TYPE_MULTICAST: u32 = 0x0002;
pub const PACKET_TYPE_ALL_MULTICAST: u32 = 0x0004;
pub const PACKET_TYPE_BROADCAST: u32 = 0x0008;
pub const PACKET_TYPE_PROMISCUOUS: u32 = 0x0020;

/// What we ask a tether for: our own unicast, broadcast (ARP and DHCP replies
/// arrive this way) and all multicast. Not promiscuous — a tether would forward
/// every frame on its link and we would drop all of them anyway.
pub const DEFAULT_PACKET_FILTER: u32 =
    PACKET_TYPE_DIRECTED | PACKET_TYPE_BROADCAST | PACKET_TYPE_ALL_MULTICAST;

// --- class requests on the control pipe ----------------------------------

pub const REQ_SEND_ENCAPSULATED_COMMAND: u8 = 0x00;
pub const REQ_GET_ENCAPSULATED_RESPONSE: u8 = 0x01;

/// `bmRequestType` for host → device, class, recipient interface.
pub const BM_OUT_CLASS_IFACE: u8 = 0x21;
/// `bmRequestType` for device → host, class, recipient interface.
pub const BM_IN_CLASS_IFACE: u8 = 0xa1;

// --- header sizes ---------------------------------------------------------

/// Every RNDIS message starts with `MessageType` then `MessageLength`.
pub const MSG_HEADER_LEN: usize = 8;
/// `REMOTE_NDIS_PACKET_MSG` header — 11 little-endian `u32`s.
pub const PACKET_HEADER_LEN: usize = 44;
/// Fixed part of `REMOTE_NDIS_INITIALIZE_MSG`.
pub const INITIALIZE_LEN: usize = 24;
/// Fixed part of `REMOTE_NDIS_INITIALIZE_CMPLT`.
pub const INITIALIZE_CMPLT_LEN: usize = 52;
/// Fixed part of a QUERY or SET request, ahead of its information buffer.
pub const QUERY_SET_HEADER_LEN: usize = 28;
/// Fixed part of a QUERY completion, ahead of its information buffer.
pub const QUERY_CMPLT_HEADER_LEN: usize = 24;

/// Buffer offsets in RNDIS are measured from **byte 8** of the message — the
/// start of `RequestId` — not from byte 0. Every `*Offset` field in this module
/// is converted through this constant so the rule is stated once.
pub const OFFSET_BASE: usize = 8;

/// Largest transfer we tell the device we can accept. Bounds one bulk IN and so
/// bounds how many packets a batch may hold.
pub const MAX_TRANSFER_SIZE: u32 = 16384;

// --- little-endian helpers ------------------------------------------------

fn rd32(b: &[u8], off: usize) -> Option<u32> {
    let s = b.get(off..off + 4)?;
    Some(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn wr32(b: &mut [u8], off: usize, v: u32) {
    b[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

// --- control setup packets ------------------------------------------------

/// `(bmRequestType, bRequest, wValue, wIndex, wLength)` for pushing a control
/// message to the device. `iface` is the **control** interface number.
pub fn send_encapsulated_setup(iface: u8, len: u16) -> (u8, u8, u16, u16, u16) {
    (BM_OUT_CLASS_IFACE, REQ_SEND_ENCAPSULATED_COMMAND, 0, iface as u16, len)
}

/// The same, for reading the device's reply back.
pub fn get_encapsulated_setup(iface: u8, len: u16) -> (u8, u8, u16, u16, u16) {
    (BM_IN_CLASS_IFACE, REQ_GET_ENCAPSULATED_RESPONSE, 0, iface as u16, len)
}

// --- requests -------------------------------------------------------------

/// `REMOTE_NDIS_INITIALIZE_MSG`. Version is pinned at 1.0, which is the only
/// version any device in the wild implements.
pub fn initialize_msg(request_id: u32) -> [u8; INITIALIZE_LEN] {
    let mut m = [0u8; INITIALIZE_LEN];
    wr32(&mut m, 0, MSG_INITIALIZE);
    wr32(&mut m, 4, INITIALIZE_LEN as u32);
    wr32(&mut m, 8, request_id);
    wr32(&mut m, 12, 1); // MajorVersion
    wr32(&mut m, 16, 0); // MinorVersion
    wr32(&mut m, 20, MAX_TRANSFER_SIZE);
    m
}

/// `REMOTE_NDIS_HALT_MSG` — tells the device to stop; sent on teardown.
pub fn halt_msg(request_id: u32) -> [u8; 12] {
    let mut m = [0u8; 12];
    wr32(&mut m, 0, MSG_HALT);
    wr32(&mut m, 4, 12);
    wr32(&mut m, 8, request_id);
    m
}

/// `REMOTE_NDIS_KEEPALIVE_MSG`.
pub fn keepalive_msg(request_id: u32) -> [u8; 16] {
    let mut m = [0u8; 16];
    wr32(&mut m, 0, MSG_KEEPALIVE);
    wr32(&mut m, 4, 16);
    wr32(&mut m, 8, request_id);
    m
}

/// `REMOTE_NDIS_QUERY_MSG` with no input buffer — the shape every OID we read
/// uses.
///
/// `InformationBufferOffset` is **0** when there is no buffer, rather than
/// pointing just past the header: a device that trusts the offset would
/// otherwise read `InformationBufferLength` = 0 bytes from a position inside our
/// message, and some firmware validates the pair rather than short-circuiting on
/// the length.
pub fn query_msg(request_id: u32, oid: u32) -> [u8; QUERY_SET_HEADER_LEN] {
    let mut m = [0u8; QUERY_SET_HEADER_LEN];
    wr32(&mut m, 0, MSG_QUERY);
    wr32(&mut m, 4, QUERY_SET_HEADER_LEN as u32);
    wr32(&mut m, 8, request_id);
    wr32(&mut m, 12, oid);
    wr32(&mut m, 16, 0); // InformationBufferLength
    wr32(&mut m, 20, 0); // InformationBufferOffset
    wr32(&mut m, 24, 0); // DeviceVcHandle
    m
}

/// `REMOTE_NDIS_SET_MSG` carrying `data` as its information buffer.
pub fn set_msg(request_id: u32, oid: u32, data: &[u8]) -> Vec<u8> {
    let total = QUERY_SET_HEADER_LEN + data.len();
    let mut m = alloc::vec![0u8; total];
    wr32(&mut m, 0, MSG_SET);
    wr32(&mut m, 4, total as u32);
    wr32(&mut m, 8, request_id);
    wr32(&mut m, 12, oid);
    wr32(&mut m, 16, data.len() as u32);
    // Relative to byte 8 — the trap this module exists to state.
    wr32(&mut m, 20, (QUERY_SET_HEADER_LEN - OFFSET_BASE) as u32);
    wr32(&mut m, 24, 0); // DeviceVcHandle
    m[QUERY_SET_HEADER_LEN..].copy_from_slice(data);
    m
}

/// The `SET` that makes a device start delivering frames.
pub fn set_packet_filter_msg(request_id: u32, filter: u32) -> Vec<u8> {
    set_msg(request_id, OID_GEN_CURRENT_PACKET_FILTER, &filter.to_le_bytes())
}

// --- responses ------------------------------------------------------------

/// What an `INITIALIZE_CMPLT` told us about the device's batching limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InitializeInfo {
    pub status: u32,
    pub major: u32,
    pub minor: u32,
    /// 0 = 802.3. Anything else is not Ethernet and must be refused.
    pub medium: u32,
    pub max_packets_per_transfer: u32,
    pub max_transfer_size: u32,
    /// Packets in a batch are aligned to `1 << packet_alignment` bytes.
    pub packet_alignment: u32,
}

impl InitializeInfo {
    pub fn ok(&self) -> bool {
        self.status == STATUS_SUCCESS && self.medium == 0
    }
}

/// Parse an `INITIALIZE_CMPLT`. `None` if it is not one, or is short.
pub fn parse_initialize_cmplt(b: &[u8]) -> Option<InitializeInfo> {
    if rd32(b, 0)? != MSG_INITIALIZE_CMPLT || b.len() < INITIALIZE_CMPLT_LEN {
        return None;
    }
    Some(InitializeInfo {
        status: rd32(b, 12)?,
        major: rd32(b, 16)?,
        minor: rd32(b, 20)?,
        medium: rd32(b, 28)?,
        max_packets_per_transfer: rd32(b, 32)?,
        max_transfer_size: rd32(b, 36)?,
        packet_alignment: rd32(b, 40)?,
    })
}

/// The information buffer out of a `QUERY_CMPLT`, or `None` when the query
/// failed, the message is not a query completion, or the buffer it describes
/// does not lie inside the bytes we actually received.
///
/// The containment check is the load-bearing part: `InformationBufferOffset` and
/// `InformationBufferLength` come from the device, and a tether is not a trusted
/// peer. A lying pair is refused rather than clamped — a clamp would hand back a
/// short buffer that reads as a real, shorter answer.
pub fn query_cmplt_buffer(b: &[u8]) -> Option<&[u8]> {
    if rd32(b, 0)? != MSG_QUERY_CMPLT {
        return None;
    }
    if rd32(b, 12)? != STATUS_SUCCESS {
        return None;
    }
    let len = rd32(b, 16)? as usize;
    let off = rd32(b, 20)? as usize;
    let start = OFFSET_BASE.checked_add(off)?;
    let end = start.checked_add(len)?;
    // Also bounded by the message's own declared length, not just by how many
    // bytes arrived: a device may pad a transfer, and reading into the padding
    // would return trailing zeros as though they were part of the answer.
    let declared = rd32(b, 4)? as usize;
    if end > b.len() || end > declared {
        return None;
    }
    b.get(start..end)
}

/// The MAC address out of a `QUERY_CMPLT` for one of the 802.3 address OIDs.
pub fn query_cmplt_mac(b: &[u8]) -> Option<[u8; 6]> {
    let buf = query_cmplt_buffer(b)?;
    // Exactly six bytes. A longer buffer is not an address with padding — it is
    // a reply to a different OID, and taking its first six bytes would produce a
    // MAC that looks entirely reasonable.
    if buf.len() != 6 {
        return None;
    }
    let mut mac = [0u8; 6];
    mac.copy_from_slice(buf);
    Some(mac)
}

/// The status word of a `SET_CMPLT`, or `None` if this is not one.
pub fn parse_set_cmplt(b: &[u8]) -> Option<u32> {
    if rd32(b, 0)? != MSG_SET_CMPLT || b.len() < 16 {
        return None;
    }
    rd32(b, 12)
}

/// The request id a completion is answering. Used to reject a stale reply — the
/// control pipe has no ordering guarantee across an aborted request, so a
/// completion for a previous message would otherwise be read as this one's.
pub fn completion_request_id(b: &[u8]) -> Option<u32> {
    let ty = rd32(b, 0)?;
    if ty & MSG_COMPLETION == 0 {
        return None;
    }
    rd32(b, 8)
}

// --- data framing ---------------------------------------------------------

/// Wrap `frame` in a `REMOTE_NDIS_PACKET_MSG` header, ready for the bulk OUT
/// endpoint.
pub fn encode_packet(frame: &[u8], out: &mut [u8]) -> Option<usize> {
    let total = PACKET_HEADER_LEN + frame.len();
    if out.len() < total {
        return None;
    }
    let h = &mut out[..PACKET_HEADER_LEN];
    h.fill(0);
    wr32(h, 0, MSG_PACKET);
    wr32(h, 4, total as u32);
    // DataOffset is measured from byte 8, so a header of 44 gives 36.
    wr32(h, 8, (PACKET_HEADER_LEN - OFFSET_BASE) as u32);
    wr32(h, 12, frame.len() as u32);
    // OOB and per-packet-info fields stay zero: we attach neither.
    out[PACKET_HEADER_LEN..total].copy_from_slice(frame);
    Some(total)
}

/// One Ethernet frame located inside a received transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PacketSpan {
    /// Byte range of the Ethernet frame within the transfer buffer.
    pub start: usize,
    pub len: usize,
    /// Where the next RNDIS message begins, or `None` at the end of the batch.
    pub next: Option<usize>,
}

/// Locate the Ethernet frame of the `REMOTE_NDIS_PACKET_MSG` starting at `at`.
///
/// Returns `None` for anything that is not a well-formed packet message wholly
/// contained in `buf` — including messages of other types, which a device may
/// legitimately interleave (`INDICATE_STATUS` arrives on the data pipe on some
/// firmware). The caller skips those using [`message_length`].
pub fn decode_packet_at(buf: &[u8], at: usize) -> Option<PacketSpan> {
    let b = buf.get(at..)?;
    if rd32(b, 0)? != MSG_PACKET {
        return None;
    }
    let msg_len = rd32(b, 4)? as usize;
    if msg_len < PACKET_HEADER_LEN || msg_len > b.len() {
        return None;
    }
    let data_off = rd32(b, 8)? as usize;
    let data_len = rd32(b, 12)? as usize;
    // `DataOffset` is relative to byte 8, and is **not** necessarily 36 — a
    // device may place per-packet info ahead of the payload.
    let start = OFFSET_BASE.checked_add(data_off)?;
    let end = start.checked_add(data_len)?;
    // The payload must lie inside this message, not merely inside the transfer:
    // a payload that ran past `msg_len` would overlap the next packet in a batch
    // and be delivered twice, once as itself and once as a prefix.
    if end > msg_len || data_len == 0 {
        return None;
    }
    let next = at + msg_len;
    Some(PacketSpan {
        start: at + start,
        len: data_len,
        next: (next < buf.len()).then_some(next),
    })
}

/// `MessageLength` of the message at `at`, for stepping over one we do not
/// handle. `None` when it is absent or implausible — a zero or oversized length
/// would make the caller's walk loop forever or read past the transfer.
pub fn message_length(buf: &[u8], at: usize) -> Option<usize> {
    let b = buf.get(at..)?;
    let len = rd32(b, 4)? as usize;
    if len < MSG_HEADER_LEN || len > b.len() {
        return None;
    }
    Some(len)
}

/// Every Ethernet frame in one received transfer, in order.
///
/// A single bulk IN can carry several packets — the device batches up to the
/// `max_packets_per_transfer` it reported at initialize. Reading only the first
/// looks like heavy packet loss rather than a framing bug, which is why this
/// returns all of them and why the multi-packet case is tested explicitly.
pub fn decode_transfer(buf: &[u8]) -> Vec<PacketSpan> {
    let mut out = Vec::new();
    let mut at = 0usize;
    // Bound the walk independently of the length fields: a device that reports a
    // length which does not advance the cursor would otherwise spin forever
    // inside the receive path, with interrupts off.
    let mut guard = 0;
    while at + MSG_HEADER_LEN <= buf.len() && guard < 64 {
        guard += 1;
        if let Some(span) = decode_packet_at(buf, at) {
            let next = span.next;
            out.push(span);
            match next {
                Some(n) if n > at => at = n,
                _ => break,
            }
        } else {
            // Not a packet message (or malformed): step over it if we can read a
            // sane length, otherwise abandon the rest of the transfer.
            match message_length(buf, at) {
                Some(l) if l > 0 => at += l,
                _ => break,
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole protocol is little-endian `u32`s, so the fixed message shapes
    /// are worth pinning byte for byte — a field written at the wrong offset is
    /// still a valid `u32` at some other field's position.
    #[test_case]
    fn initialize_message_layout() {
        let m = initialize_msg(7);
        assert_eq!(rd32(&m, 0), Some(MSG_INITIALIZE));
        assert_eq!(rd32(&m, 4), Some(INITIALIZE_LEN as u32), "MessageLength counts itself");
        assert_eq!(rd32(&m, 8), Some(7), "RequestId");
        assert_eq!(rd32(&m, 12), Some(1), "MajorVersion");
        assert_eq!(rd32(&m, 16), Some(0), "MinorVersion");
        assert_eq!(rd32(&m, 20), Some(MAX_TRANSFER_SIZE));
    }

    /// **Buffer offsets are relative to byte 8, not to byte 0.** This is the
    /// single most consequential rule in RNDIS: an absolute reading lands 8
    /// bytes early, which for a MAC query is the tail of `Status` plus four
    /// bytes of the real address — a perfectly well-formed MAC belonging to
    /// nobody, that no sanity check rejects.
    #[test_case]
    fn buffer_offsets_are_relative_to_byte_eight() {
        let filter = DEFAULT_PACKET_FILTER;
        let m = set_packet_filter_msg(3, filter);
        assert_eq!(m.len(), QUERY_SET_HEADER_LEN + 4);
        assert_eq!(rd32(&m, 16), Some(4), "InformationBufferLength");
        let off = rd32(&m, 20).unwrap() as usize;
        assert_eq!(off, QUERY_SET_HEADER_LEN - OFFSET_BASE, "offset excludes the first 8 bytes");
        // Following the field the way a device does must land on our payload.
        assert_eq!(rd32(&m, OFFSET_BASE + off), Some(filter));
        // And the naive absolute reading must land somewhere else, or this test
        // would pass with the bug present.
        assert_ne!(OFFSET_BASE + off, off, "an absolute offset must differ");
    }

    /// A query with no input buffer sends offset 0, not "just past the header":
    /// some firmware validates the offset/length pair rather than
    /// short-circuiting on a zero length.
    #[test_case]
    fn a_query_with_no_input_buffer_sends_a_zero_offset() {
        let m = query_msg(1, OID_802_3_PERMANENT_ADDRESS);
        assert_eq!(rd32(&m, 12), Some(OID_802_3_PERMANENT_ADDRESS));
        assert_eq!(rd32(&m, 16), Some(0), "InformationBufferLength");
        assert_eq!(rd32(&m, 20), Some(0), "InformationBufferOffset");
    }

    fn build_query_cmplt(request_id: u32, status: u32, payload: &[u8]) -> Vec<u8> {
        let total = QUERY_CMPLT_HEADER_LEN + payload.len();
        let mut m = alloc::vec![0u8; total];
        wr32(&mut m, 0, MSG_QUERY_CMPLT);
        wr32(&mut m, 4, total as u32);
        wr32(&mut m, 8, request_id);
        wr32(&mut m, 12, status);
        wr32(&mut m, 16, payload.len() as u32);
        wr32(&mut m, 20, (QUERY_CMPLT_HEADER_LEN - OFFSET_BASE) as u32);
        m[QUERY_CMPLT_HEADER_LEN..].copy_from_slice(payload);
        m
    }

    #[test_case]
    fn reads_the_mac_out_of_a_query_completion() {
        let mac = [0x02, 0x11, 0x22, 0x33, 0x44, 0x55];
        let m = build_query_cmplt(9, STATUS_SUCCESS, &mac);
        assert_eq!(query_cmplt_mac(&m), Some(mac));
        assert_eq!(completion_request_id(&m), Some(9));
    }

    /// A failed query has no answer, and a buffer the device describes as lying
    /// outside the bytes it sent is refused rather than clamped — a clamp hands
    /// back a short buffer that reads as a real, shorter answer.
    #[test_case]
    fn a_failed_or_lying_query_yields_nothing() {
        let mac = [0x02, 0x11, 0x22, 0x33, 0x44, 0x55];
        // Non-success status.
        let bad = build_query_cmplt(1, 0xc000_0001, &mac);
        assert_eq!(query_cmplt_mac(&bad), None);
        // Length running past the message.
        let mut over = build_query_cmplt(1, STATUS_SUCCESS, &mac);
        wr32(&mut over, 16, 4096);
        assert_eq!(query_cmplt_buffer(&over), None);
        // Offset running past the message.
        let mut far = build_query_cmplt(1, STATUS_SUCCESS, &mac);
        wr32(&mut far, 20, 0xffff_fff0);
        assert_eq!(query_cmplt_buffer(&far), None);
        // A six-byte read of a longer buffer is a different OID's answer, not a
        // padded address.
        let long = build_query_cmplt(1, STATUS_SUCCESS, &[0u8; 16]);
        assert_eq!(query_cmplt_mac(&long), None);
        assert!(query_cmplt_buffer(&long).is_some(), "the buffer itself is still readable");
    }

    /// A device may pad the transfer past its own declared `MessageLength`;
    /// reading into the padding would return trailing zeros as part of the
    /// answer.
    #[test_case]
    fn a_buffer_past_the_declared_message_length_is_refused() {
        let mut m = build_query_cmplt(1, STATUS_SUCCESS, &[1, 2, 3, 4, 5, 6]);
        m.extend_from_slice(&[0u8; 32]); // transfer padding
        assert_eq!(query_cmplt_mac(&m), Some([1, 2, 3, 4, 5, 6]));
        // Now claim a buffer that fits in the padded transfer but not in the
        // message: it must still be refused.
        wr32(&mut m, 16, 20);
        assert_eq!(query_cmplt_buffer(&m), None);
    }

    #[test_case]
    fn initialize_completion_reports_batching_limits() {
        let mut m = [0u8; INITIALIZE_CMPLT_LEN];
        wr32(&mut m, 0, MSG_INITIALIZE_CMPLT);
        wr32(&mut m, 4, INITIALIZE_CMPLT_LEN as u32);
        wr32(&mut m, 8, 1);
        wr32(&mut m, 12, STATUS_SUCCESS);
        wr32(&mut m, 16, 1); // major
        wr32(&mut m, 20, 0); // minor
        wr32(&mut m, 28, 0); // medium 802.3
        wr32(&mut m, 32, 4); // max packets per transfer
        wr32(&mut m, 36, 16384);
        wr32(&mut m, 40, 3); // alignment 1<<3
        let info = parse_initialize_cmplt(&m).unwrap();
        assert!(info.ok());
        assert_eq!(info.max_packets_per_transfer, 4);
        assert_eq!(info.packet_alignment, 3);
        // A non-802.3 medium is not Ethernet and must not be adopted.
        let mut wireless = m;
        wr32(&mut wireless, 28, 1);
        assert!(!parse_initialize_cmplt(&wireless).unwrap().ok());
    }

    #[test_case]
    fn packet_header_round_trips_a_frame() {
        let frame: Vec<u8> = (0..64u8).collect();
        let mut buf = [0u8; 256];
        let n = encode_packet(&frame, &mut buf).unwrap();
        assert_eq!(n, PACKET_HEADER_LEN + frame.len());
        assert_eq!(rd32(&buf, 8), Some(36), "DataOffset is 44 - 8");
        let span = decode_packet_at(&buf[..n], 0).unwrap();
        assert_eq!(&buf[span.start..span.start + span.len], &frame[..]);
        assert_eq!(span.next, None, "a single packet ends the batch");
    }

    /// **One transfer can carry several packets.** Reading only the first looks
    /// like heavy packet loss rather than a framing bug, so the batch walk is
    /// pinned explicitly.
    #[test_case]
    fn one_transfer_can_carry_several_packets() {
        let a: Vec<u8> = alloc::vec![0xaa; 60];
        let b: Vec<u8> = alloc::vec![0xbb; 100];
        let c: Vec<u8> = alloc::vec![0xcc; 14];
        let mut buf = alloc::vec![0u8; 1024];
        let mut at = 0;
        for f in [&a, &b, &c] {
            let n = encode_packet(f, &mut buf[at..]).unwrap();
            at += n;
        }
        let spans = decode_transfer(&buf[..at]);
        assert_eq!(spans.len(), 3, "every packet in the batch must be found");
        assert_eq!(spans[0].len, 60);
        assert_eq!(spans[1].len, 100);
        assert_eq!(spans[2].len, 14);
        assert_eq!(&buf[spans[1].start..spans[1].start + 100], &b[..]);
    }

    /// `DataOffset` is not the constant 36 — a device may attach per-packet
    /// info ahead of the payload. Assuming 44 shifts every frame by however
    /// much was attached, which corrupts rather than drops them.
    #[test_case]
    fn data_offset_is_read_not_assumed() {
        let frame: Vec<u8> = alloc::vec![0x5a; 40];
        let extra = 16usize; // per-packet info between header and payload
        let total = PACKET_HEADER_LEN + extra + frame.len();
        let mut buf = alloc::vec![0u8; total];
        wr32(&mut buf, 0, MSG_PACKET);
        wr32(&mut buf, 4, total as u32);
        wr32(&mut buf, 8, (PACKET_HEADER_LEN + extra - OFFSET_BASE) as u32);
        wr32(&mut buf, 12, frame.len() as u32);
        buf[PACKET_HEADER_LEN + extra..].copy_from_slice(&frame);
        let span = decode_packet_at(&buf, 0).unwrap();
        assert_eq!(span.start, PACKET_HEADER_LEN + extra);
        assert_eq!(&buf[span.start..span.start + span.len], &frame[..]);
    }

    /// A payload that runs past its own message would overlap the next packet in
    /// a batch and be delivered twice — once whole, once as a prefix.
    #[test_case]
    fn a_payload_running_past_its_message_is_refused() {
        let mut buf = alloc::vec![0u8; 256];
        wr32(&mut buf, 0, MSG_PACKET);
        wr32(&mut buf, 4, 64); // message claims 64 bytes
        wr32(&mut buf, 8, 36);
        wr32(&mut buf, 12, 200); // payload claims 200
        assert_eq!(decode_packet_at(&buf, 0), None);
    }

    /// The batch walk must terminate on hostile input. A length that does not
    /// advance the cursor would otherwise spin forever inside the receive path,
    /// with interrupts off — a hang, not a dropped frame.
    #[test_case]
    fn the_batch_walk_terminates_on_a_non_advancing_length() {
        let mut buf = alloc::vec![0u8; 128];
        wr32(&mut buf, 0, MSG_PACKET);
        wr32(&mut buf, 4, 0); // zero length
        assert!(decode_transfer(&buf).is_empty());
        // A length shorter than a header is equally non-advancing.
        wr32(&mut buf, 4, 4);
        assert!(decode_transfer(&buf).is_empty());
    }

    /// A non-packet message interleaved on the data pipe (some firmware sends
    /// `INDICATE_STATUS` there) is stepped over, not treated as a frame.
    #[test_case]
    fn a_non_packet_message_in_the_stream_is_skipped() {
        let frame: Vec<u8> = alloc::vec![0x77; 32];
        let mut buf = alloc::vec![0u8; 512];
        // An INDICATE_STATUS of 20 bytes, then a real packet.
        wr32(&mut buf, 0, MSG_INDICATE_STATUS);
        wr32(&mut buf, 4, 20);
        let n = encode_packet(&frame, &mut buf[20..]).unwrap();
        let spans = decode_transfer(&buf[..20 + n]);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].len, 32);
        assert_eq!(&buf[spans[0].start..spans[0].start + 32], &frame[..]);
    }

    /// The class requests are recipient-**interface**, which is why they carry
    /// the control interface number in `wIndex`. Sending them to the device
    /// (recipient 0) is accepted by some firmware and ignored by the rest.
    #[test_case]
    fn encapsulated_control_requests_target_the_interface() {
        let (bm, req, val, idx, len) = send_encapsulated_setup(2, 24);
        assert_eq!(bm, 0x21, "host->device, class, interface");
        assert_eq!(req, REQ_SEND_ENCAPSULATED_COMMAND);
        assert_eq!((val, idx, len), (0, 2, 24));
        let (bm, req, _, idx, len) = get_encapsulated_setup(2, 1024);
        assert_eq!(bm, 0xa1, "device->host, class, interface");
        assert_eq!(req, REQ_GET_ENCAPSULATED_RESPONSE);
        assert_eq!((idx, len), (2, 1024));
    }

    /// The default filter must include broadcast, or ARP and DHCP replies never
    /// arrive and the link comes up with no address — the failure that reads as
    /// "the tether does not work" with every control message reporting success.
    #[test_case]
    fn the_default_packet_filter_admits_broadcast() {
        assert_ne!(DEFAULT_PACKET_FILTER & PACKET_TYPE_BROADCAST, 0);
        assert_ne!(DEFAULT_PACKET_FILTER & PACKET_TYPE_DIRECTED, 0);
        assert_eq!(
            DEFAULT_PACKET_FILTER & PACKET_TYPE_PROMISCUOUS,
            0,
            "promiscuous would forward the tether's whole link"
        );
    }
}
