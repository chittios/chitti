//! **HIDP** (HID over Bluetooth, classic) — pure packet helpers.
//!
//! Interrupt channel carries input reports; control channel carries SET_PROTOCOL
//! etc. Boot keyboard reports are the same 8-byte layout as USB HID boot.

use alloc::vec::Vec;

/// HIDP header high nibble = transaction type.
pub const TRANS_HANDSHAKE: u8 = 0x00;
pub const TRANS_HID_CONTROL: u8 = 0x10;
pub const TRANS_GET_REPORT: u8 = 0x40;
pub const TRANS_SET_REPORT: u8 = 0x50;
pub const TRANS_GET_PROTOCOL: u8 = 0x60;
pub const TRANS_SET_PROTOCOL: u8 = 0x70;
pub const TRANS_DATA: u8 = 0xa0;

pub const PARAM_DATA_INPUT: u8 = 0x01;
pub const PARAM_DATA_OUTPUT: u8 = 0x02;
pub const PARAM_PROTOCOL_BOOT: u8 = 0x00;
pub const PARAM_PROTOCOL_REPORT: u8 = 0x01;

/// SET_PROTOCOL(boot) on the control channel.
pub fn set_protocol_boot() -> Vec<u8> {
    alloc::vec![TRANS_SET_PROTOCOL | PARAM_PROTOCOL_BOOT]
}

/// Build a DATA(Input) header + report (usually not sent by host).
pub fn data_input(report: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(1 + report.len());
    v.push(TRANS_DATA | PARAM_DATA_INPUT);
    v.extend_from_slice(report);
    v
}

/// If `pkt` is HIDP DATA Input, return the report body.
pub fn parse_data_input(pkt: &[u8]) -> Option<&[u8]> {
    if pkt.is_empty() {
        return None;
    }
    let hdr = pkt[0];
    if hdr & 0xf0 != TRANS_DATA {
        return None;
    }
    if hdr & 0x0f != PARAM_DATA_INPUT {
        return None;
    }
    Some(&pkt[1..])
}

/// Decode a boot keyboard report into ASCII-ish events is done by the USB HID
/// path; we only validate length here.
pub fn is_boot_keyboard_report(rep: &[u8]) -> bool {
    rep.len() >= 8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn set_protocol_and_data_input() {
        assert_eq!(set_protocol_boot(), alloc::vec![0x70]);
        let p = data_input(&[0, 0, 0x04, 0, 0, 0, 0, 0]);
        let r = parse_data_input(&p).unwrap();
        assert_eq!(r[2], 0x04);
        assert!(is_boot_keyboard_report(r));
        assert!(parse_data_input(&[0x00]).is_none());
    }
}
