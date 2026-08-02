//! **HCI packet codec** — pure, unit-tested framing for the host controller
//! interface (Bluetooth Core Spec Vol 4, Part A / Vol 2 Host Controller Interface).
//!
//! No I/O: builders produce wire bytes; parsers refuse malformed lengths rather
//! than clamping into a wrong opcode.

/// HCI packet indicator bytes (USB / UART H4).
pub const PKT_COMMAND: u8 = 0x01;
pub const PKT_ACL: u8 = 0x02;
pub const PKT_SCO: u8 = 0x03;
pub const PKT_EVENT: u8 = 0x04;

/// Opcode group field: Controller & Baseband.
pub const OGF_CONTROLLER_BASEBAND: u16 = 0x03;
/// Opcode group: Informational parameters.
pub const OGF_INFORMATIONAL: u16 = 0x04;

pub const OCF_RESET: u16 = 0x0003;
pub const OCF_READ_LOCAL_NAME: u16 = 0x0014;
pub const OCF_READ_LOCAL_VERSION: u16 = 0x0001;
pub const OCF_READ_BD_ADDR: u16 = 0x0009;

/// Event codes.
pub const EVT_COMMAND_COMPLETE: u8 = 0x0e;
pub const EVT_COMMAND_STATUS: u8 = 0x0f;

/// Pack OGF/OCF into a 16-bit HCI opcode (`OGF` in bits 15:10, `OCF` in 9:0).
pub fn opcode(ogf: u16, ocf: u16) -> u16 {
    ((ogf & 0x3f) << 10) | (ocf & 0x03ff)
}

/// Split a 16-bit opcode back into (OGF, OCF).
pub fn split_opcode(op: u16) -> (u16, u16) {
    ((op >> 10) & 0x3f, op & 0x03ff)
}

/// Build an HCI **Command** packet: indicator + opcode LE + plen + params.
pub fn command(ogf: u16, ocf: u16, params: &[u8]) -> alloc::vec::Vec<u8> {
    let op = opcode(ogf, ocf);
    let mut v = alloc::vec::Vec::with_capacity(4 + params.len());
    v.push(PKT_COMMAND);
    v.extend_from_slice(&op.to_le_bytes());
    v.push(params.len() as u8);
    v.extend_from_slice(params);
    v
}

/// HCI_Reset (no parameters).
pub fn cmd_reset() -> alloc::vec::Vec<u8> {
    command(OGF_CONTROLLER_BASEBAND, OCF_RESET, &[])
}

/// HCI_Read_Local_Name (no parameters).
pub fn cmd_read_local_name() -> alloc::vec::Vec<u8> {
    command(OGF_CONTROLLER_BASEBAND, OCF_READ_LOCAL_NAME, &[])
}

/// HCI_Read_Local_Version_Information.
pub fn cmd_read_local_version() -> alloc::vec::Vec<u8> {
    command(OGF_INFORMATIONAL, OCF_READ_LOCAL_VERSION, &[])
}

/// HCI_Read_BD_ADDR.
pub fn cmd_read_bd_addr() -> alloc::vec::Vec<u8> {
    command(OGF_INFORMATIONAL, OCF_READ_BD_ADDR, &[])
}

/// Parsed HCI event header + payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Event<'a> {
    pub code: u8,
    pub params: &'a [u8],
}

/// Parse one H4 event packet (`04 | code | plen | params…`).
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

/// Command Complete (0x0E): `Num_HCI_Command_Packets | Opcode | Return…`.
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

/// Local name from Read Local Name complete (248-byte null-padded UTF-8).
pub fn local_name_from_return(ret: &[u8]) -> Option<&str> {
    if ret.is_empty() {
        return None;
    }
    // First byte is status (0 = success).
    if ret[0] != 0 {
        return None;
    }
    let name = &ret[1..];
    let end = name.iter().position(|&b| b == 0).unwrap_or(name.len());
    core::str::from_utf8(&name[..end]).ok()
}

/// BD_ADDR from Read BD_ADDR complete: status + 6 bytes little-endian.
pub fn bd_addr_from_return(ret: &[u8]) -> Option<[u8; 6]> {
    if ret.len() < 7 || ret[0] != 0 {
        return None;
    }
    let mut a = [0u8; 6];
    a.copy_from_slice(&ret[1..7]);
    Some(a)
}

/// Format BD_ADDR as `AA:BB:CC:DD:EE:FF` (MSB first, wire is LE).
pub fn format_bd_addr(le: &[u8; 6]) -> alloc::string::String {
    alloc::format!(
        "{:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}",
        le[5], le[4], le[3], le[2], le[1], le[0]
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn opcode_packs_ogf_ocf() {
        let op = opcode(OGF_CONTROLLER_BASEBAND, OCF_RESET);
        assert_eq!(op, 0x0c03);
        assert_eq!(split_opcode(op), (OGF_CONTROLLER_BASEBAND, OCF_RESET));
        let name = opcode(OGF_CONTROLLER_BASEBAND, OCF_READ_LOCAL_NAME);
        assert_eq!(name, 0x0c14);
    }

    #[test_case]
    fn cmd_reset_wire_shape() {
        let p = cmd_reset();
        assert_eq!(p, alloc::vec![0x01, 0x03, 0x0c, 0x00]);
    }

    #[test_case]
    fn cmd_read_local_name_wire_shape() {
        let p = cmd_read_local_name();
        assert_eq!(p[0], PKT_COMMAND);
        assert_eq!(u16::from_le_bytes([p[1], p[2]]), 0x0c14);
        assert_eq!(p[3], 0);
    }

    #[test_case]
    fn parse_event_and_command_complete() {
        // Event: Command Complete for Reset, status success.
        let raw = [0x04u8, 0x0e, 0x04, 0x01, 0x03, 0x0c, 0x00];
        let ev = parse_event(&raw).expect("event");
        assert_eq!(ev.code, EVT_COMMAND_COMPLETE);
        let cc = parse_command_complete(ev.params).expect("cc");
        assert_eq!(cc.opcode, 0x0c03);
        assert_eq!(cc.return_params, &[0x00]);
    }

    #[test_case]
    fn parse_event_refuses_short_buffer() {
        assert!(parse_event(&[0x04, 0x0e, 0x04, 0x01]).is_none());
        assert!(parse_event(&[0x01, 0x00]).is_none()); // not an event
    }

    #[test_case]
    fn local_name_and_bd_addr_decode() {
        let mut ret = alloc::vec![0u8; 249];
        ret[0] = 0; // status
        ret[1..6].copy_from_slice(b"Chitti");
        assert_eq!(local_name_from_return(&ret), Some("Chitti"));
        assert_eq!(local_name_from_return(&[0x01, b'x']), None); // status fail

        let bd = [0u8, 0x56, 0x34, 0x12, 0xab, 0xcd, 0xef]; // status + LE addr
        let a = bd_addr_from_return(&bd).unwrap();
        assert_eq!(format_bd_addr(&a), "EF:CD:AB:12:34:56");
    }
}
