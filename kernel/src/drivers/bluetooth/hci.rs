//! **HCI packet codec** — pure, unit-tested framing for the host controller
//! interface (Bluetooth Core Spec Vol 2 / Vol 4 USB transport).
//!
//! ## USB vs H4
//!
//! USB HCI **does not** put the H4 packet indicator on the wire for commands /
//! events / ACL: the endpoint *is* the channel. Helpers ending in `_usb` build
//! or parse that form; H4 builders keep the indicator for tests and UART later.

use alloc::vec::Vec;

/// HCI packet indicator bytes (UART H4 only).
pub const PKT_COMMAND: u8 = 0x01;
pub const PKT_ACL: u8 = 0x02;
pub const PKT_SCO: u8 = 0x03;
pub const PKT_EVENT: u8 = 0x04;

/// Opcode groups.
pub const OGF_LINK_CONTROL: u16 = 0x01;
pub const OGF_CONTROLLER_BASEBAND: u16 = 0x03;
pub const OGF_INFORMATIONAL: u16 = 0x04;

pub const OCF_INQUIRY: u16 = 0x0001;
pub const OCF_INQUIRY_CANCEL: u16 = 0x0002;
pub const OCF_CREATE_CONNECTION: u16 = 0x0005;
pub const OCF_DISCONNECT: u16 = 0x0006;
pub const OCF_AUTH_REQUESTED: u16 = 0x0011;
pub const OCF_PIN_CODE_REQUEST_REPLY: u16 = 0x000d;
pub const OCF_PIN_CODE_REQUEST_NEGATIVE_REPLY: u16 = 0x000e;
pub const OCF_IO_CAPABILITY_REQUEST_REPLY: u16 = 0x002b;
pub const OCF_USER_CONFIRMATION_REQUEST_REPLY: u16 = 0x002c;
pub const OCF_USER_CONFIRMATION_REQUEST_NEG_REPLY: u16 = 0x002d;
pub const OCF_USER_PASSKEY_REQUEST_REPLY: u16 = 0x002e;
pub const OCF_USER_PASSKEY_REQUEST_NEG_REPLY: u16 = 0x002f;
pub const OCF_RESET: u16 = 0x0003;
pub const OCF_WRITE_LOCAL_NAME: u16 = 0x0013;
pub const OCF_READ_LOCAL_NAME: u16 = 0x0014;
pub const OCF_WRITE_SCAN_ENABLE: u16 = 0x001a;
pub const OCF_READ_LOCAL_VERSION: u16 = 0x0001;
pub const OCF_READ_BD_ADDR: u16 = 0x0009;

/// Event codes.
pub const EVT_INQUIRY_COMPLETE: u8 = 0x01;
pub const EVT_INQUIRY_RESULT: u8 = 0x02;
pub const EVT_CONNECTION_COMPLETE: u8 = 0x03;
pub const EVT_DISCONNECTION_COMPLETE: u8 = 0x05;
pub const EVT_AUTH_COMPLETE: u8 = 0x06;
pub const EVT_REMOTE_NAME_REQ_COMPLETE: u8 = 0x07;
pub const EVT_ENCRYPTION_CHANGE: u8 = 0x08;
pub const EVT_COMMAND_COMPLETE: u8 = 0x0e;
pub const EVT_COMMAND_STATUS: u8 = 0x0f;
pub const EVT_PIN_CODE_REQUEST: u8 = 0x16;
pub const EVT_LINK_KEY_NOTIFICATION: u8 = 0x18;
pub const EVT_INQUIRY_RESULT_WITH_RSSI: u8 = 0x22;
pub const EVT_EXTENDED_INQUIRY_RESULT: u8 = 0x2f;
pub const EVT_IO_CAPABILITY_REQUEST: u8 = 0x31;
pub const EVT_IO_CAPABILITY_RESPONSE: u8 = 0x32;
pub const EVT_USER_CONFIRMATION_REQUEST: u8 = 0x33;
pub const EVT_USER_PASSKEY_REQUEST: u8 = 0x34;
pub const EVT_REMOTE_OOB_DATA_REQUEST: u8 = 0x35;
pub const EVT_SIMPLE_PAIRING_COMPLETE: u8 = 0x36;
pub const EVT_NUMBER_OF_COMPLETED_PACKETS: u8 = 0x13;

/// Pack OGF/OCF into a 16-bit HCI opcode.
pub fn opcode(ogf: u16, ocf: u16) -> u16 {
    ((ogf & 0x3f) << 10) | (ocf & 0x03ff)
}

pub fn split_opcode(op: u16) -> (u16, u16) {
    ((op >> 10) & 0x3f, op & 0x03ff)
}

/// HCI command body for **USB**: opcode LE + plen + params (no packet type).
pub fn command_usb(ogf: u16, ocf: u16, params: &[u8]) -> Vec<u8> {
    let op = opcode(ogf, ocf);
    let mut v = Vec::with_capacity(3 + params.len());
    v.extend_from_slice(&op.to_le_bytes());
    v.push(params.len() as u8);
    v.extend_from_slice(params);
    v
}

/// H4 command (indicator + body).
pub fn command(ogf: u16, ocf: u16, params: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(4 + params.len());
    v.push(PKT_COMMAND);
    v.extend_from_slice(&command_usb(ogf, ocf, params));
    v
}

pub fn cmd_reset() -> Vec<u8> {
    command(OGF_CONTROLLER_BASEBAND, OCF_RESET, &[])
}
pub fn cmd_reset_usb() -> Vec<u8> {
    command_usb(OGF_CONTROLLER_BASEBAND, OCF_RESET, &[])
}

pub fn cmd_read_local_name() -> Vec<u8> {
    command(OGF_CONTROLLER_BASEBAND, OCF_READ_LOCAL_NAME, &[])
}
pub fn cmd_read_local_name_usb() -> Vec<u8> {
    command_usb(OGF_CONTROLLER_BASEBAND, OCF_READ_LOCAL_NAME, &[])
}

pub fn cmd_read_local_version() -> Vec<u8> {
    command(OGF_INFORMATIONAL, OCF_READ_LOCAL_VERSION, &[])
}
pub fn cmd_read_local_version_usb() -> Vec<u8> {
    command_usb(OGF_INFORMATIONAL, OCF_READ_LOCAL_VERSION, &[])
}

pub fn cmd_read_bd_addr() -> Vec<u8> {
    command(OGF_INFORMATIONAL, OCF_READ_BD_ADDR, &[])
}
pub fn cmd_read_bd_addr_usb() -> Vec<u8> {
    command_usb(OGF_INFORMATIONAL, OCF_READ_BD_ADDR, &[])
}

/// Scan enable: 0=none, 1=inquiry, 2=page, 3=both.
pub fn cmd_write_scan_enable_usb(enable: u8) -> Vec<u8> {
    command_usb(OGF_CONTROLLER_BASEBAND, OCF_WRITE_SCAN_ENABLE, &[enable])
}

/// Inquiry: LAP (3) + length (1.28s units) + num responses (0 = unlimited until length).
pub fn cmd_inquiry_usb(length_slots: u8, num_responses: u8) -> Vec<u8> {
    // GIAC LAP = 0x9E8B33
    command_usb(
        OGF_LINK_CONTROL,
        OCF_INQUIRY,
        &[0x33, 0x8b, 0x9e, length_slots, num_responses],
    )
}

pub fn cmd_inquiry_cancel_usb() -> Vec<u8> {
    command_usb(OGF_LINK_CONTROL, OCF_INQUIRY_CANCEL, &[])
}

/// Create Connection — simplified defaults (DM1/DH1… packet type 0xcc18, role switch allow).
pub fn cmd_create_connection_usb(bd_addr_le: &[u8; 6]) -> Vec<u8> {
    let mut p = [0u8; 13];
    p[0..6].copy_from_slice(bd_addr_le);
    // Packet_Type
    p[6] = 0x18;
    p[7] = 0xcc;
    p[8] = 0x01; // page scan repetition mode R1
    p[9] = 0x00; // reserved
    p[10] = 0x00; // clock offset
    p[11] = 0x00;
    p[12] = 0x01; // allow role switch
    command_usb(OGF_LINK_CONTROL, OCF_CREATE_CONNECTION, &p)
}

pub fn cmd_disconnect_usb(handle: u16, reason: u8) -> Vec<u8> {
    let mut p = [0u8; 3];
    p[0..2].copy_from_slice(&handle.to_le_bytes());
    p[2] = reason;
    command_usb(OGF_LINK_CONTROL, OCF_DISCONNECT, &p)
}

pub fn cmd_auth_requested_usb(handle: u16) -> Vec<u8> {
    command_usb(OGF_LINK_CONTROL, OCF_AUTH_REQUESTED, &handle.to_le_bytes())
}

/// PIN Code Request Reply — pin up to 16 bytes.
pub fn cmd_pin_code_reply_usb(bd_addr_le: &[u8; 6], pin: &[u8]) -> Vec<u8> {
    let mut p = [0u8; 23];
    p[0..6].copy_from_slice(bd_addr_le);
    let n = pin.len().min(16);
    p[6] = n as u8;
    p[7..7 + n].copy_from_slice(&pin[..n]);
    command_usb(OGF_LINK_CONTROL, OCF_PIN_CODE_REQUEST_REPLY, &p)
}

pub fn cmd_pin_code_neg_reply_usb(bd_addr_le: &[u8; 6]) -> Vec<u8> {
    command_usb(
        OGF_LINK_CONTROL,
        OCF_PIN_CODE_REQUEST_NEGATIVE_REPLY,
        bd_addr_le,
    )
}

/// SSP I/O capabilities.  DisplayYesNo lets the host explicitly confirm the
/// six-digit numeric comparison rather than silently accepting a peer.
pub const IO_CAP_DISPLAY_YES_NO: u8 = 0x01;
/// No OOB data is available through the current UI.
pub const OOB_DATA_NOT_PRESENT: u8 = 0x00;
/// General bonding plus MITM protection.  A controller that cannot satisfy
/// this reports pairing failure instead of silently falling back to a legacy
/// unauthenticated link.
pub const AUTH_REQ_GENERAL_BONDING_MITM: u8 = 0x05;

pub fn cmd_io_capability_reply_usb(bd_addr_le: &[u8; 6], io_cap: u8, oob: u8, auth: u8) -> Vec<u8> {
    let mut p = [0u8; 9];
    p[..6].copy_from_slice(bd_addr_le);
    p[6] = io_cap;
    p[7] = oob;
    p[8] = auth;
    command_usb(OGF_LINK_CONTROL, OCF_IO_CAPABILITY_REQUEST_REPLY, &p)
}

pub fn cmd_user_confirmation_reply_usb(bd_addr_le: &[u8; 6]) -> Vec<u8> {
    command_usb(OGF_LINK_CONTROL, OCF_USER_CONFIRMATION_REQUEST_REPLY, bd_addr_le)
}

pub fn cmd_user_confirmation_neg_reply_usb(bd_addr_le: &[u8; 6]) -> Vec<u8> {
    command_usb(OGF_LINK_CONTROL, OCF_USER_CONFIRMATION_REQUEST_NEG_REPLY, bd_addr_le)
}

pub fn cmd_user_passkey_reply_usb(bd_addr_le: &[u8; 6], passkey: u32) -> Option<Vec<u8>> {
    if passkey > 999_999 { return None; }
    let mut p = [0u8; 10];
    p[..6].copy_from_slice(bd_addr_le);
    p[6..].copy_from_slice(&passkey.to_le_bytes());
    Some(command_usb(OGF_LINK_CONTROL, OCF_USER_PASSKEY_REQUEST_REPLY, &p))
}

pub fn cmd_user_passkey_neg_reply_usb(bd_addr_le: &[u8; 6]) -> Vec<u8> {
    command_usb(OGF_LINK_CONTROL, OCF_USER_PASSKEY_REQUEST_NEG_REPLY, bd_addr_le)
}

// ── events ───────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Event<'a> {
    pub code: u8,
    pub params: &'a [u8],
}

/// H4 event: `04 | code | plen | params`.
pub fn parse_event(buf: &[u8]) -> Option<Event<'_>> {
    if buf.len() < 3 || buf[0] != PKT_EVENT {
        return None;
    }
    let code = buf[1];
    let plen = buf[2] as usize;
    if buf.len() < 3 + plen {
        return None;
    }
    Some(Event {
        code,
        params: &buf[3..3 + plen],
    })
}

/// USB interrupt event: `code | plen | params` (no packet type).
pub fn parse_event_usb(buf: &[u8]) -> Option<Event<'_>> {
    if buf.len() < 2 {
        return None;
    }
    let code = buf[0];
    let plen = buf[1] as usize;
    if buf.len() < 2 + plen {
        return None;
    }
    Some(Event {
        code,
        params: &buf[2..2 + plen],
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommandComplete<'a> {
    pub num_cmd_packets: u8,
    pub opcode: u16,
    pub return_params: &'a [u8],
}

pub fn parse_command_complete(params: &[u8]) -> Option<CommandComplete<'_>> {
    if params.len() < 3 {
        return None;
    }
    Some(CommandComplete {
        num_cmd_packets: params[0],
        opcode: u16::from_le_bytes([params[1], params[2]]),
        return_params: &params[3..],
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommandStatus {
    pub status: u8,
    pub num_cmd_packets: u8,
    pub opcode: u16,
}

pub fn parse_command_status(params: &[u8]) -> Option<CommandStatus> {
    if params.len() < 4 {
        return None;
    }
    Some(CommandStatus {
        status: params[0],
        num_cmd_packets: params[1],
        opcode: u16::from_le_bytes([params[2], params[3]]),
    })
}

pub fn local_name_from_return(ret: &[u8]) -> Option<&str> {
    if ret.is_empty() || ret[0] != 0 {
        return None;
    }
    let name = &ret[1..];
    let end = name.iter().position(|&b| b == 0).unwrap_or(name.len());
    core::str::from_utf8(&name[..end]).ok()
}

pub fn bd_addr_from_return(ret: &[u8]) -> Option<[u8; 6]> {
    if ret.len() < 7 || ret[0] != 0 {
        return None;
    }
    let mut a = [0u8; 6];
    a.copy_from_slice(&ret[1..7]);
    Some(a)
}

pub fn format_bd_addr(le: &[u8; 6]) -> alloc::string::String {
    alloc::format!(
        "{:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}",
        le[5], le[4], le[3], le[2], le[1], le[0]
    )
}

/// Parse `AA:BB:CC:DD:EE:FF` into little-endian wire order.
pub fn parse_bd_addr(s: &str) -> Option<[u8; 6]> {
    let mut be = [0u8; 6];
    let mut i = 0usize;
    for part in s.split(|c| c == ':' || c == '-') {
        if i >= 6 {
            return None;
        }
        be[i] = u8::from_str_radix(part.trim(), 16).ok()?;
        i += 1;
    }
    if i != 6 {
        return None;
    }
    let mut le = [0u8; 6];
    for j in 0..6 {
        le[j] = be[5 - j];
    }
    Some(le)
}

/// One remote from Inquiry Result (standard, no RSSI).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InquiryEntry {
    pub bd_addr: [u8; 6],
    pub page_scan_rep_mode: u8,
    pub class_of_device: u32,
}

/// Parse Inquiry Result event params: num_responses + 14×N bytes each.
pub fn parse_inquiry_result(params: &[u8]) -> Vec<InquiryEntry> {
    let mut out = Vec::new();
    if params.is_empty() {
        return out;
    }
    let n = params[0] as usize;
    let mut off = 1usize;
    for _ in 0..n {
        if off + 14 > params.len() {
            break;
        }
        let mut bd = [0u8; 6];
        bd.copy_from_slice(&params[off..off + 6]);
        let psrm = params[off + 6];
        // **Two** reserved bytes, off+7 and off+8 — the Core spec's HCI_Inquiry_Result entry is
        // BD_ADDR(6) PSRM(1) Reserved(1) Reserved(1) CoD(3) Clock_Offset(2) = 14. Reading the
        // class from off+8 assumed one, which is a *plausible* wrong answer rather than an
        // obvious one: it takes the last reserved byte as the class's low octet, so a real
        // device's major class silently comes out as something else. The 14-byte stride and the
        // bounds check were already right, which is what made the entry-length arithmetic
        // disagree with the field offsets.
        let cod = u32::from_le_bytes([
            params[off + 9],
            params[off + 10],
            params[off + 11],
            0,
        ]);
        out.push(InquiryEntry {
            bd_addr: bd,
            page_scan_rep_mode: psrm,
            class_of_device: cod,
        });
        off += 14;
    }
    out
}

/// Connection Complete: status, handle, bd_addr, link_type, encryption.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConnectionComplete {
    pub status: u8,
    pub handle: u16,
    pub bd_addr: [u8; 6],
    pub link_type: u8,
}

pub fn parse_connection_complete(params: &[u8]) -> Option<ConnectionComplete> {
    if params.len() < 11 {
        return None;
    }
    let mut bd = [0u8; 6];
    bd.copy_from_slice(&params[3..9]);
    Some(ConnectionComplete {
        status: params[0],
        handle: u16::from_le_bytes([params[1], params[2]]),
        bd_addr: bd,
        link_type: params[9],
    })
}

/// PIN Code Request: bd_addr only.
pub fn parse_pin_code_request(params: &[u8]) -> Option<[u8; 6]> {
    if params.len() < 6 {
        return None;
    }
    let mut bd = [0u8; 6];
    bd.copy_from_slice(&params[0..6]);
    Some(bd)
}

/// SSP user-confirmation request: peer address followed by a six-digit value.
pub fn parse_user_confirmation_request(params: &[u8]) -> Option<([u8; 6], u32)> {
    if params.len() < 10 { return None; }
    let mut bd = [0u8; 6];
    bd.copy_from_slice(&params[..6]);
    Some((bd, u32::from_le_bytes(params[6..10].try_into().ok()?)))
}

/// Events that only carry a peer address (I/O capability and passkey request).
pub fn parse_bd_addr_event(params: &[u8]) -> Option<[u8; 6]> {
    params.get(..6)?.try_into().ok()
}

/// Simple Pairing Complete: status then BD_ADDR.
pub fn parse_simple_pairing_complete(params: &[u8]) -> Option<(u8, [u8; 6])> {
    if params.len() < 7 { return None; }
    let mut bd = [0u8; 6];
    bd.copy_from_slice(&params[1..7]);
    Some((params[0], bd))
}

/// ACL header for USB bulk: handle_flags LE (12-bit handle) + data_len LE.
pub fn acl_header(handle: u16, pb_bc: u16, data_len: u16) -> [u8; 4] {
    let h = (handle & 0x0fff) | ((pb_bc & 0xf) << 12);
    let mut b = [0u8; 4];
    b[0..2].copy_from_slice(&h.to_le_bytes());
    b[2..4].copy_from_slice(&data_len.to_le_bytes());
    b
}

pub fn parse_acl_header(buf: &[u8]) -> Option<(u16, u16, u16)> {
    if buf.len() < 4 {
        return None;
    }
    let h = u16::from_le_bytes([buf[0], buf[1]]);
    let len = u16::from_le_bytes([buf[2], buf[3]]);
    Some((h & 0x0fff, (h >> 12) & 0xf, len))
}

/// Major device class from CoD (bits 8–12 of 24-bit CoD).
pub fn cod_major(cod: u32) -> u8 {
    ((cod >> 8) & 0x1f) as u8
}

/// CoD major class 5 = Peripheral (keyboard/mouse often).
pub const COD_MAJOR_PERIPHERAL: u8 = 5;

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn opcode_packs_ogf_ocf() {
        let op = opcode(OGF_CONTROLLER_BASEBAND, OCF_RESET);
        assert_eq!(op, 0x0c03);
        assert_eq!(split_opcode(op), (OGF_CONTROLLER_BASEBAND, OCF_RESET));
    }

    #[test_case]
    fn usb_command_has_no_packet_type() {
        let p = cmd_reset_usb();
        assert_eq!(p, alloc::vec![0x03, 0x0c, 0x00]);
        let h4 = cmd_reset();
        assert_eq!(h4[0], PKT_COMMAND);
        assert_eq!(&h4[1..], &p[..]);
    }

    #[test_case]
    fn parse_event_usb_and_h4() {
        let usb = [0x0eu8, 0x04, 0x01, 0x03, 0x0c, 0x00];
        let ev = parse_event_usb(&usb).unwrap();
        assert_eq!(ev.code, EVT_COMMAND_COMPLETE);
        let cc = parse_command_complete(ev.params).unwrap();
        assert_eq!(cc.opcode, 0x0c03);

        let mut h4 = alloc::vec![PKT_EVENT];
        h4.extend_from_slice(&usb);
        assert_eq!(parse_event(&h4).unwrap().code, EVT_COMMAND_COMPLETE);
    }

    #[test_case]
    fn inquiry_and_connection_parse() {
        // 1 response: addr 11:22:33:44:55:66 LE, cod keyboard-ish
        let mut p = alloc::vec![1u8];
        p.extend_from_slice(&[0x66, 0x55, 0x44, 0x33, 0x22, 0x11]);
        p.push(0x01); // psrm
        p.push(0x00); // reserved 1
        p.push(0x00); // reserved 2 -- the spec has two, and omitting one made the 14-byte
                      // bounds check reject a 13-byte entry, so this parsed as zero responses
        p.extend_from_slice(&[0x40, 0x05, 0x00]); // CoD
        p.extend_from_slice(&[0, 0]); // clock offset
        let e = parse_inquiry_result(&p);
        assert_eq!(e.len(), 1);
        assert_eq!(format_bd_addr(&e[0].bd_addr), "11:22:33:44:55:66");
        assert_eq!(cod_major(e[0].class_of_device), COD_MAJOR_PERIPHERAL);

        let cc = [
            0u8, 0x0b, 0x00, // status, handle 0x000b
            0x66, 0x55, 0x44, 0x33, 0x22, 0x11, // bd
            0x01, // ACL
            0x00,
        ];
        let c = parse_connection_complete(&cc).unwrap();
        assert_eq!(c.handle, 0x0b);
        assert_eq!(c.status, 0);
    }

    #[test_case]
    fn bd_addr_roundtrip_string() {
        let le = parse_bd_addr("AA:BB:CC:DD:EE:FF").unwrap();
        assert_eq!(format_bd_addr(&le), "AA:BB:CC:DD:EE:FF");
        assert!(parse_bd_addr("bad").is_none());
    }

    #[test_case]
    fn pin_reply_length_and_acl_header() {
        let bd = [1u8, 2, 3, 4, 5, 6];
        let p = cmd_pin_code_reply_usb(&bd, b"1234");
        assert_eq!(p[0], 0x0d); // OCF low
        assert_eq!(p[2], 23); // param len — wait, command_usb plen is byte after opcode
        // opcode 0x040d → bytes 0d 04, plen 23
        assert_eq!(&p[0..3], &[0x0d, 0x04, 23]);
        assert_eq!(p[3 + 6], 4); // pin len
        let h = acl_header(0x0b, 0x2, 10);
        let (handle, pb, len) = parse_acl_header(&h).unwrap();
        assert_eq!(handle, 0x0b);
        assert_eq!(pb, 2);
        assert_eq!(len, 10);
    }

    #[test_case]
    fn ssp_commands_and_events_keep_addresses_and_values_in_wire_order() {
        let bd = [0x66, 0x55, 0x44, 0x33, 0x22, 0x11];
        let io = cmd_io_capability_reply_usb(
            &bd, IO_CAP_DISPLAY_YES_NO, OOB_DATA_NOT_PRESENT, AUTH_REQ_GENERAL_BONDING_MITM,
        );
        assert_eq!(&io[..3], &[0x2b, 0x04, 9], "IO capability reply opcode + length");
        assert_eq!(&io[3..9], &bd);
        assert_eq!(&io[9..], &[1, 0, 5]);

        let passkey = cmd_user_passkey_reply_usb(&bd, 123_456).expect("six digits accepted");
        assert_eq!(&passkey[..3], &[0x2e, 0x04, 10]);
        assert_eq!(&passkey[3..9], &bd);
        assert_eq!(u32::from_le_bytes(passkey[9..13].try_into().unwrap()), 123_456);
        assert!(cmd_user_passkey_reply_usb(&bd, 1_000_000).is_none());

        let mut confirm = alloc::vec![0u8; 10];
        confirm[..6].copy_from_slice(&bd);
        confirm[6..].copy_from_slice(&654_321u32.to_le_bytes());
        assert_eq!(parse_user_confirmation_request(&confirm), Some((bd, 654_321)));
        let mut complete = alloc::vec![0u8];
        complete.extend_from_slice(&bd);
        assert_eq!(parse_simple_pairing_complete(&complete), Some((0, bd)));
    }

    #[test_case]
    fn local_name_and_bd_addr_decode() {
        let mut ret = alloc::vec![0u8; 249];
        // `1..7`, not `1..6`: Read_Local_Name returns status(1) then 248 name bytes, and
        // "Chitti" is six. The rest stays zero, which is the NUL the parser stops at.
        ret[1..7].copy_from_slice(b"Chitti");
        assert_eq!(local_name_from_return(&ret), Some("Chitti"));
        let bd = [0u8, 0x56, 0x34, 0x12, 0xab, 0xcd, 0xef];
        assert_eq!(format_bd_addr(&bd_addr_from_return(&bd).unwrap()), "EF:CD:AB:12:34:56");
    }
}
