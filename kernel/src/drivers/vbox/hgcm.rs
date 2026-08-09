//! **HGCM** — VirtualBox's Host-Guest Communication Manager, as pure request
//! framing over a byte buffer.
//!
//! HGCM is how every VirtualBox guest service is reached (shared clipboard,
//! shared folders, guest properties, drag-and-drop). It rides the VMMDev
//! request mechanism in [`super`]: the guest fills in a request structure,
//! writes its physical address to one register, and the host answers.
//!
//! Every offset here comes from VirtualBox's own `include/VBox/VMMDev.h` and
//! `VMMDevCoreTypes.h` — **fetched, not recalled**, and pinned by tests, for the
//! reason the whole tree keeps relearning: these are plain integers, so a wrong
//! offset yields a request the host acts on rather than an error.
//!
//! ## The two things that are easy to get wrong
//!
//! **An HGCM call is asynchronous, and its reply arrives in place.** The
//! register write returns immediately with `VINF_HGCM_ASYNC_EXECUTE`; the host
//! sets [`REQ_DONE`] in `fu32Flags` when it is actually finished, and only then
//! are the output parameters valid. Reading them on the strength of the
//! register write returning is reading whatever was in the buffer before.
//!
//! **The parameter struct is `#pragma pack(4)`**, so a 64-bit value sits at
//! offset 4 of a 16-byte parameter — not at 8, where natural alignment would
//! put it. Getting that wrong shifts every pointer and size the host reads.

/// `HGCMFunctionParameter64` is 4 + 12 = **16** bytes (`AssertCompileSize`),
/// because the struct is `#pragma pack(4)`.
pub const PARM: usize = 16;
/// Offset of the parameter's type tag.
pub const PARM_TYPE: usize = 0;
/// Offset of the union — a 32-bit value, a 64-bit value, or `{u32 cb; u64 addr}`.
pub const PARM_VALUE: usize = 4;

// --- HGCMFunctionParameterType ---
pub const PARM_32BIT: u32 = 1;
pub const PARM_64BIT: u32 = 2;
/// In **and** out.
pub const PARM_LINADDR: u32 = 4;
/// In only — the host reads this buffer.
pub const PARM_LINADDR_IN: u32 = 5;
/// Out only — the host writes this buffer.
pub const PARM_LINADDR_OUT: u32 = 6;

/// `VBOX_HGCM_REQ_DONE` — bit 0 of `fu32Flags`.
pub const REQ_DONE: u32 = 1;

/// `HGCMServiceLocation`: `type` then a 128-byte name. 132 bytes total.
pub const LOC: usize = 132;
pub const LOC_NAME: usize = 4;
pub const LOC_NAME_MAX: usize = 128;
/// `VMMDevHGCMLoc_LocalHost_Existing` — connect to a service the host already
/// has, which is what every built-in service is.
pub const LOC_LOCALHOST_EXISTING: u32 = 2;

/// `VMMDevHGCMConnect` = HGCM header (32) + location (132) + `u32ClientID`.
pub const CONNECT_LEN: usize = super::HGCM_HDR + LOC + 4;
/// Where the client id the host assigns lands.
pub const CONNECT_CLIENT_ID: usize = super::HGCM_HDR + LOC;

/// `VMMDevHGCMDisconnect` = header (32) + `u32ClientID`.
pub const DISCONNECT_LEN: usize = super::HGCM_HDR + 4;

/// `VMMDevHGCMCall` = header (32) + clientID + function + cParms, then the
/// parameters.
pub const CALL_HDR: usize = super::HGCM_HDR + 12;
pub const CALL_CLIENT_ID: usize = super::HGCM_HDR;
pub const CALL_FUNCTION: usize = super::HGCM_HDR + 4;
pub const CALL_CPARMS: usize = super::HGCM_HDR + 8;

/// Total size of a call request carrying `n` parameters.
pub const fn call_len(n: usize) -> usize {
    CALL_HDR + n * PARM
}

/// Byte offset of parameter `i` in a call request.
pub const fn parm_at(i: usize) -> usize {
    CALL_HDR + i * PARM
}

/// One outgoing parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Parm {
    U32(u32),
    U64(u64),
    /// A buffer at a guest linear address, with its length and direction.
    Buf { addr: u64, len: u32, typ: u32 },
}

impl Parm {
    /// A buffer the host reads (guest -> host).
    pub fn buf_in(addr: u64, len: u32) -> Parm {
        Parm::Buf { addr, len, typ: PARM_LINADDR_IN }
    }
    /// A buffer the host writes (host -> guest).
    pub fn buf_out(addr: u64, len: u32) -> Parm {
        Parm::Buf { addr, len, typ: PARM_LINADDR_OUT }
    }
}

/// Frame an `HGCMConnect` for `service` into `buf`.
///
/// The name is NUL-terminated inside a fixed 128-byte field; a name that does
/// not fit is refused rather than truncated, because a truncated service name
/// either fails to resolve or — worse — resolves to a different service.
pub fn write_connect(buf: &mut [u8], service: &str) -> bool {
    if buf.len() < CONNECT_LEN || service.len() >= LOC_NAME_MAX {
        return false;
    }
    if !super::write_header(buf, CONNECT_LEN as u32, super::REQ_HGCM_CONNECT) {
        return false;
    }
    // `fu32Flags` and `result` are the host's; zero them so a reply that never
    // arrived cannot read as done.
    buf[super::REQ_HDR..super::HGCM_HDR].fill(0);
    let loc = super::HGCM_HDR;
    buf[loc..loc + 4].copy_from_slice(&LOC_LOCALHOST_EXISTING.to_le_bytes());
    buf[loc + LOC_NAME..loc + LOC].fill(0);
    buf[loc + LOC_NAME..loc + LOC_NAME + service.len()].copy_from_slice(service.as_bytes());
    buf[CONNECT_CLIENT_ID..CONNECT_CLIENT_ID + 4].fill(0);
    true
}

/// Frame an `HGCMDisconnect` for `client`.
pub fn write_disconnect(buf: &mut [u8], client: u32) -> bool {
    if buf.len() < DISCONNECT_LEN || !super::write_header(buf, DISCONNECT_LEN as u32, super::REQ_HGCM_DISCONNECT) {
        return false;
    }
    buf[super::REQ_HDR..super::HGCM_HDR].fill(0);
    buf[super::HGCM_HDR..super::HGCM_HDR + 4].copy_from_slice(&client.to_le_bytes());
    true
}

/// Frame an `HGCMCall` of `function` on `client` with `parms`.
pub fn write_call(buf: &mut [u8], client: u32, function: u32, parms: &[Parm]) -> bool {
    let len = call_len(parms.len());
    if buf.len() < len || !super::write_header(buf, len as u32, super::REQ_HGCM_CALL64) {
        return false;
    }
    buf[super::REQ_HDR..super::HGCM_HDR].fill(0);
    buf[CALL_CLIENT_ID..CALL_CLIENT_ID + 4].copy_from_slice(&client.to_le_bytes());
    buf[CALL_FUNCTION..CALL_FUNCTION + 4].copy_from_slice(&function.to_le_bytes());
    buf[CALL_CPARMS..CALL_CPARMS + 4].copy_from_slice(&(parms.len() as u32).to_le_bytes());
    for (i, p) in parms.iter().enumerate() {
        let o = parm_at(i);
        // Zero the whole parameter first: the union's unused tail is read by
        // the host for some types, and leftover bytes would be interpreted.
        buf[o..o + PARM].fill(0);
        match *p {
            Parm::U32(v) => {
                buf[o + PARM_TYPE..o + 4].copy_from_slice(&PARM_32BIT.to_le_bytes());
                buf[o + PARM_VALUE..o + PARM_VALUE + 4].copy_from_slice(&v.to_le_bytes());
            }
            Parm::U64(v) => {
                buf[o + PARM_TYPE..o + 4].copy_from_slice(&PARM_64BIT.to_le_bytes());
                // Offset 4, not 8 — the struct is packed to 4.
                buf[o + PARM_VALUE..o + PARM_VALUE + 8].copy_from_slice(&v.to_le_bytes());
            }
            Parm::Buf { addr, len, typ } => {
                buf[o + PARM_TYPE..o + 4].copy_from_slice(&typ.to_le_bytes());
                // `{ u32 size; u64 addr }` packed to 4: size at 4, addr at 8.
                buf[o + PARM_VALUE..o + PARM_VALUE + 4].copy_from_slice(&len.to_le_bytes());
                buf[o + PARM_VALUE + 4..o + PARM_VALUE + 12].copy_from_slice(&addr.to_le_bytes());
            }
        }
    }
    true
}

/// Read back parameter `i` as a `u32` (an output parameter after completion).
pub fn parm_u32(buf: &[u8], i: usize) -> u32 {
    let o = parm_at(i) + PARM_VALUE;
    if o + 4 > buf.len() {
        return 0;
    }
    u32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]])
}

/// The HGCM `fu32Flags` word — [`REQ_DONE`] tells you the reply is real.
pub fn flags(buf: &[u8]) -> u32 {
    if buf.len() < super::HGCM_HDR {
        return 0;
    }
    let o = super::REQ_HDR;
    u32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]])
}

/// The HGCM `result` — the service's own status, valid only once [`REQ_DONE`].
pub fn result(buf: &[u8]) -> i32 {
    if buf.len() < super::HGCM_HDR {
        return -1;
    }
    let o = super::REQ_HDR + 4;
    i32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]])
}

/// Whether the host has finished with this request.
pub fn is_done(buf: &[u8]) -> bool {
    flags(buf) & REQ_DONE != 0
}

/// The client id the host assigned, after a completed `HGCMConnect`.
pub fn connect_client_id(buf: &[u8]) -> u32 {
    if buf.len() < CONNECT_LEN {
        return 0;
    }
    let o = CONNECT_CLIENT_ID;
    u32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn the_parameter_struct_is_packed_to_four() {
        // `#pragma pack(4)` puts a 64-bit value at offset 4 of a 16-byte
        // parameter, where natural alignment would put it at 8. Every pointer
        // and size the host reads shifts if this is wrong — and the host acts
        // on the result rather than reporting it.
        assert_eq!(PARM, 16);
        assert_eq!(PARM_VALUE, 4);

        let mut buf = [0u8; 256];
        assert!(write_call(&mut buf, 7, 42, &[Parm::U64(0x1122_3344_5566_7788)]));
        let o = parm_at(0);
        assert_eq!(u32::from_le_bytes(buf[o..o + 4].try_into().unwrap()), PARM_64BIT);
        assert_eq!(
            u64::from_le_bytes(buf[o + 4..o + 12].try_into().unwrap()),
            0x1122_3344_5566_7788
        );

        // A buffer parameter is `{u32 size; u64 addr}` in the same packed
        // union: size at 4, address at 8.
        assert!(write_call(&mut buf, 7, 42, &[Parm::buf_in(0xdead_beef_0000, 1234)]));
        let o = parm_at(0);
        assert_eq!(u32::from_le_bytes(buf[o..o + 4].try_into().unwrap()), PARM_LINADDR_IN);
        assert_eq!(u32::from_le_bytes(buf[o + 4..o + 8].try_into().unwrap()), 1234);
        assert_eq!(
            u64::from_le_bytes(buf[o + 8..o + 16].try_into().unwrap()),
            0xdead_beef_0000
        );
    }

    #[test_case]
    fn the_call_header_sits_after_the_hgcm_header() {
        // header(24) + fu32Flags + result = 32, then clientID/function/cParms.
        assert_eq!(super::super::HGCM_HDR, 32);
        assert_eq!(CALL_HDR, 44);
        assert_eq!(parm_at(0), 44);
        assert_eq!(parm_at(1), 60);
        assert_eq!(call_len(3), 44 + 48);

        let mut buf = [0u8; 256];
        assert!(write_call(&mut buf, 0x1234, 9, &[Parm::U32(1), Parm::U32(2)]));
        assert_eq!(u32::from_le_bytes(buf[CALL_CLIENT_ID..CALL_CLIENT_ID + 4].try_into().unwrap()), 0x1234);
        assert_eq!(u32::from_le_bytes(buf[CALL_FUNCTION..CALL_FUNCTION + 4].try_into().unwrap()), 9);
        assert_eq!(u32::from_le_bytes(buf[CALL_CPARMS..CALL_CPARMS + 4].try_into().unwrap()), 2);
        // Declared size covers the parameters, or the host reads past them.
        assert_eq!(
            u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize,
            call_len(2)
        );
        assert_eq!(u32::from_le_bytes(buf[8..12].try_into().unwrap()), super::super::REQ_HGCM_CALL64);
    }

    #[test_case]
    fn a_reply_is_only_real_once_the_host_says_done() {
        // The register write returns immediately with VINF_HGCM_ASYNC_EXECUTE.
        // Reading outputs before REQ_DONE reads whatever was in the buffer.
        let mut buf = [0u8; 256];
        assert!(write_call(&mut buf, 1, 1, &[Parm::U32(0), Parm::U32(0)]));
        assert!(!is_done(&buf), "a freshly framed request must not look done");
        // The host sets bit 0 of fu32Flags, which lives right after the
        // 24-byte VMMDev header.
        buf[super::super::REQ_HDR] = REQ_DONE as u8;
        assert!(is_done(&buf));
    }

    #[test_case]
    fn connect_names_the_service_without_truncating_it() {
        let mut buf = [0u8; 512];
        assert!(write_connect(&mut buf, "VBoxSharedClipboard"));
        assert_eq!(u32::from_le_bytes(buf[8..12].try_into().unwrap()), super::super::REQ_HGCM_CONNECT);
        let loc = super::super::HGCM_HDR;
        assert_eq!(u32::from_le_bytes(buf[loc..loc + 4].try_into().unwrap()), LOC_LOCALHOST_EXISTING);
        let name = &buf[loc + LOC_NAME..loc + LOC_NAME + 19];
        assert_eq!(core::str::from_utf8(name).unwrap(), "VBoxSharedClipboard");
        // NUL-terminated inside the fixed field.
        assert_eq!(buf[loc + LOC_NAME + 19], 0);
        assert_eq!(CONNECT_LEN, 32 + 132 + 4);

        // A name that does not fit is REFUSED, never truncated: a truncated
        // service name either fails to resolve or resolves to a different
        // service.
        let long = "x".repeat(LOC_NAME_MAX);
        assert!(!write_connect(&mut buf, &long));
        // And a buffer too small for the request is refused too.
        let mut small = [0u8; 64];
        assert!(!write_connect(&mut small, "VBoxSharedClipboard"));
    }

    #[test_case]
    fn output_parameters_read_back_from_where_they_were_written() {
        // The host overwrites the parameter union in place; reading it back
        // must use the same packed offset the write used.
        let mut buf = [0u8; 256];
        assert!(write_call(&mut buf, 1, 1, &[Parm::U32(0), Parm::U32(0)]));
        let o = parm_at(1) + PARM_VALUE;
        buf[o..o + 4].copy_from_slice(&0xabcdu32.to_le_bytes());
        assert_eq!(parm_u32(&buf, 1), 0xabcd);
        assert_eq!(parm_u32(&buf, 0), 0);
    }
}
