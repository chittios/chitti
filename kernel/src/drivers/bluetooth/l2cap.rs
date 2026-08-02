//! **L2CAP basic framing** (Bluetooth Core Spec Vol 3 Part A) — pure.
//!
//! Enough to open a fixed-channel HID connection after ACL is up. No ERTM,
//! no reassembly of multi-ACL PDUs beyond a single-segment PDU.

use alloc::vec::Vec;

/// Signalling CID.
pub const CID_SIGNALING: u16 = 0x0001;
/// Connectionless CID.
pub const CID_CONNECTIONLESS: u16 = 0x0002;

/// HID Control PSM (Bluetooth HID profile).
pub const PSM_HID_CONTROL: u16 = 0x0011;
/// HID Interrupt PSM.
pub const PSM_HID_INTERRUPT: u16 = 0x0013;

/// Signalling command codes.
pub const SIG_COMMAND_REJECT: u8 = 0x01;
pub const SIG_CONNECTION_REQUEST: u8 = 0x02;
pub const SIG_CONNECTION_RESPONSE: u8 = 0x03;
pub const SIG_CONFIG_REQUEST: u8 = 0x04;
pub const SIG_CONFIG_RESPONSE: u8 = 0x05;
pub const SIG_DISCONNECTION_REQUEST: u8 = 0x06;
pub const SIG_DISCONNECTION_RESPONSE: u8 = 0x07;
pub const SIG_ECHO_REQUEST: u8 = 0x08;
pub const SIG_INFO_REQUEST: u8 = 0x0a;

/// Build a basic L2CAP PDU: length LE + CID LE + payload.
pub fn pdu(cid: u16, payload: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(4 + payload.len());
    v.extend_from_slice(&(payload.len() as u16).to_le_bytes());
    v.extend_from_slice(&cid.to_le_bytes());
    v.extend_from_slice(payload);
    v
}

/// Parse basic header → (cid, payload).
pub fn parse_pdu(buf: &[u8]) -> Option<(u16, &[u8])> {
    if buf.len() < 4 {
        return None;
    }
    let len = u16::from_le_bytes([buf[0], buf[1]]) as usize;
    let cid = u16::from_le_bytes([buf[2], buf[3]]);
    if buf.len() < 4 + len {
        return None;
    }
    Some((cid, &buf[4..4 + len]))
}

/// Connection Request payload: code, id, length, PSM, SCID.
pub fn connection_request(id: u8, psm: u16, scid: u16) -> Vec<u8> {
    let mut body = [0u8; 4];
    body[0..2].copy_from_slice(&psm.to_le_bytes());
    body[2..4].copy_from_slice(&scid.to_le_bytes());
    signalling(SIG_CONNECTION_REQUEST, id, &body)
}

/// Connection Response: dcid, scid, result, status.
pub fn connection_response(id: u8, dcid: u16, scid: u16, result: u16, status: u16) -> Vec<u8> {
    let mut body = [0u8; 8];
    body[0..2].copy_from_slice(&dcid.to_le_bytes());
    body[2..4].copy_from_slice(&scid.to_le_bytes());
    body[4..6].copy_from_slice(&result.to_le_bytes());
    body[6..8].copy_from_slice(&status.to_le_bytes());
    signalling(SIG_CONNECTION_RESPONSE, id, &body)
}

/// Result 0 = success.
pub const CONN_SUCCESS: u16 = 0x0000;

/// Minimal Config Request (empty options = accept defaults).
pub fn config_request(id: u8, dcid: u16, flags: u16) -> Vec<u8> {
    let mut body = [0u8; 4];
    body[0..2].copy_from_slice(&dcid.to_le_bytes());
    body[2..4].copy_from_slice(&flags.to_le_bytes());
    signalling(SIG_CONFIG_REQUEST, id, &body)
}

pub fn config_response(id: u8, scid: u16, flags: u16, result: u16) -> Vec<u8> {
    let mut body = [0u8; 6];
    body[0..2].copy_from_slice(&scid.to_le_bytes());
    body[2..4].copy_from_slice(&flags.to_le_bytes());
    body[4..6].copy_from_slice(&result.to_le_bytes());
    signalling(SIG_CONFIG_RESPONSE, id, &body)
}

fn signalling(code: u8, id: u8, data: &[u8]) -> Vec<u8> {
    let mut sig = Vec::with_capacity(4 + data.len());
    sig.push(code);
    sig.push(id);
    sig.extend_from_slice(&(data.len() as u16).to_le_bytes());
    sig.extend_from_slice(data);
    pdu(CID_SIGNALING, &sig)
}

/// Parse one signalling command from an L2CAP signalling payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SigCmd<'a> {
    pub code: u8,
    pub id: u8,
    pub data: &'a [u8],
}

pub fn parse_signalling(payload: &[u8]) -> Option<SigCmd<'_>> {
    if payload.len() < 4 {
        return None;
    }
    let code = payload[0];
    let id = payload[1];
    let len = u16::from_le_bytes([payload[2], payload[3]]) as usize;
    if payload.len() < 4 + len {
        return None;
    }
    Some(SigCmd {
        code,
        id,
        data: &payload[4..4 + len],
    })
}

pub fn parse_connection_response(data: &[u8]) -> Option<(u16, u16, u16, u16)> {
    if data.len() < 8 {
        return None;
    }
    Some((
        u16::from_le_bytes([data[0], data[1]]),
        u16::from_le_bytes([data[2], data[3]]),
        u16::from_le_bytes([data[4], data[5]]),
        u16::from_le_bytes([data[6], data[7]]),
    ))
}

pub fn parse_connection_request(data: &[u8]) -> Option<(u16, u16)> {
    if data.len() < 4 {
        return None;
    }
    Some((
        u16::from_le_bytes([data[0], data[1]]),
        u16::from_le_bytes([data[2], data[3]]),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn connection_request_roundtrip() {
        let p = connection_request(1, PSM_HID_INTERRUPT, 0x0040);
        let (cid, payload) = parse_pdu(&p).unwrap();
        assert_eq!(cid, CID_SIGNALING);
        let cmd = parse_signalling(payload).unwrap();
        assert_eq!(cmd.code, SIG_CONNECTION_REQUEST);
        assert_eq!(cmd.id, 1);
        let (psm, scid) = parse_connection_request(cmd.data).unwrap();
        assert_eq!(psm, PSM_HID_INTERRUPT);
        assert_eq!(scid, 0x0040);
    }

    #[test_case]
    fn connection_response_success() {
        let p = connection_response(2, 0x0050, 0x0040, CONN_SUCCESS, 0);
        let (_, payload) = parse_pdu(&p).unwrap();
        let cmd = parse_signalling(payload).unwrap();
        let (dcid, scid, result, _) = parse_connection_response(cmd.data).unwrap();
        assert_eq!(dcid, 0x0050);
        assert_eq!(scid, 0x0040);
        assert_eq!(result, CONN_SUCCESS);
    }
}
