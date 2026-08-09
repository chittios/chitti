//! **`VBoxSharedClipboard`** — VirtualBox's shared clipboard service over
//! [`super::hgcm`].
//!
//! This is the route that reaches a real host clipboard from a VM *window*.
//! Unlike the QEMU side (SPICE vdagent over virtio-serial), VirtualBox's host
//! process owns the clipboard integration itself, so on macOS this is the one
//! path that reaches the Mac pasteboard.
//!
//! Function numbers, parameter counts and types come from VirtualBox's
//! `VBoxClipboardSvc.h` and from the **host service's own validation code**
//! (`VBoxSharedClipboardSvc.cpp`) — fetched, not recalled. The host rejects a
//! wrong parameter count or type outright, which is the good case; a wrong
//! *order* is accepted and misread, which is why the shapes are pinned below.
//!
//! ## Why the old message protocol
//!
//! The service has two: `MSG_OLD_GET_WAIT` (two `u32` outputs) and the newer
//! `MSG_PEEK_*`/`MSG_GET` pair with context IDs. The old one is marked
//! deprecated and is still accepted, and it is a far better fit here — **an
//! HGCM call that has nothing to report simply stays pending**, so one
//! outstanding `MSG_OLD_GET_WAIT` is exactly "tell me when something happens",
//! polled from `upkeep` by checking its done flag. That is what a real guest
//! agent gets from a blocking thread, which this kernel does not have.
//!
//! Consequently the request buffer for that call is **its own page**: it stays
//! owned by the host until the call completes, so it cannot be shared with the
//! synchronous calls that run in between.
//!
//! ## Text on the wire is UTF-16LE with CRLF
//!
//! `VBOX_SHCL_FMT_UNICODETEXT` is UTF-16LE, not UTF-8, and VirtualBox's own
//! Linux additions convert line endings on the way out (`ShClUtf16LinToWin`).
//! Both conversions are pure and tested here, because getting them wrong
//! produces text — just the wrong text, or half of it.

use super::hgcm::{self, Parm};
use crate::mm::Locked;
use alloc::string::String;
use alloc::vec::Vec;

/// The HGCM service name.
pub const SERVICE: &str = "VBoxSharedClipboard";

// --- guest -> host function numbers (VBOX_SHCL_GUEST_FN_XXX) ---
const FN_MSG_OLD_GET_WAIT: u32 = 1;
const FN_REPORT_FORMATS: u32 = 2;
const FN_DATA_READ: u32 = 3;
const FN_DATA_WRITE: u32 = 4;

// --- host -> guest message ids (VBOX_SHCL_HOST_MSG_XXX) ---
const MSG_QUIT: u32 = 1;
const MSG_READ_DATA: u32 = 2;
const MSG_FORMATS_REPORT: u32 = 3;
const MSG_CANCELED: u32 = 4;

// --- formats (VBOX_SHCL_FMT_XXX) ---
pub const FMT_UNICODETEXT: u32 = 1 << 0;

/// Most clipboard bytes to move in one direction. The same bound the rest of
/// the clipboard path uses; a runaway paste should fail visibly rather than
/// consume a first-fit heap.
const MAX_BYTES: usize = 1024 * 1024;

/// Convert UTF-8 to the UTF-16LE the host expects, with `LF` → `CRLF`.
///
/// VirtualBox's own Linux additions do this (`ShClUtf16LinToWin`), and the
/// terminating NUL is included: the host treats the buffer as a C wide string.
pub fn utf8_to_wire(s: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len() * 2 + 2);
    let mut prev_cr = false;
    for c in s.chars() {
        if c == '\n' && !prev_cr {
            // Lone LF becomes CRLF; an existing CRLF is left alone rather than
            // becoming CRCRLF.
            out.extend_from_slice(&0x000du16.to_le_bytes());
        }
        prev_cr = c == '\r';
        let mut b = [0u16; 2];
        for unit in c.encode_utf16(&mut b) {
            out.extend_from_slice(&unit.to_le_bytes());
        }
    }
    // Terminating NUL, as a wide character.
    out.extend_from_slice(&0u16.to_le_bytes());
    out
}

/// Convert the host's UTF-16LE back to UTF-8, dropping `CR` and any trailing
/// NUL.
///
/// Lossy on an unpaired surrogate rather than refusing: a host clipboard may
/// hold something we cannot represent, and losing the whole paste is worse than
/// one replacement character.
pub fn wire_to_utf8(bytes: &[u8]) -> String {
    let units: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        // The host sends a NUL-terminated wide string; everything past the
        // terminator is padding, not text.
        .take_while(|&u| u != 0)
        .collect();
    let mut s = String::with_capacity(units.len());
    for c in char::decode_utf16(units.into_iter()) {
        match c {
            Ok('\r') => {}
            Ok(c) => s.push(c),
            Err(_) => s.push('\u{fffd}'),
        }
    }
    s
}

/// State of the connection to the service.
struct Client {
    id: u32,
    /// Whether a `MSG_OLD_GET_WAIT` is outstanding on the message page.
    waiting: bool,
    /// Formats the host last told us it holds.
    host_formats: u32,
}

static CLIENT: Locked<Option<Client>> = Locked::new(None);

/// Whether the shared clipboard is connected.
pub fn connected() -> bool {
    CLIENT.with(|c| c.is_some())
}

/// Connect to the service. Returns the client id.
///
/// Called from `/vbox up` rather than at boot: it is a write to a real
/// hypervisor's device, and the same posture `/wifi up` takes applies.
pub fn connect() -> Result<u32, &'static str> {
    if let Some(id) = CLIENT.with(|c| c.as_ref().map(|c| c.id)) {
        return Ok(id);
    }
    let id = super::hgcm_connect(SERVICE)?;
    CLIENT.with(|c| *c = Some(Client { id, waiting: false, host_formats: 0 }));
    crate::ktrace::log_fmt(format_args!("vbox: shared clipboard connected, client {id}"));
    // Announce what we hold now, if anything, so a paste on the host right
    // after connecting finds something.
    if let Some((text, _)) = crate::clipboard::get() {
        if !text.is_empty() {
            let _ = report_formats(FMT_UNICODETEXT);
        }
    }
    arm_message_wait();
    Ok(id)
}

fn client_id() -> Option<u32> {
    CLIENT.with(|c| c.as_ref().map(|c| c.id))
}

/// Tell the host which formats we now hold. Called when the guest copies.
pub fn report_formats(formats: u32) -> Result<(), &'static str> {
    let id = client_id().ok_or("shared clipboard not connected")?;
    // One parameter: the format bits (VBOX_SHCL_CPARMS_REPORT_FORMATS == 1).
    super::hgcm_call(id, FN_REPORT_FORMATS, &[Parm::U32(formats)]).map(|_| ())
}

/// Send our clipboard to the host, in answer to a `READ_DATA` message.
fn write_data(format: u32, text: &str) -> Result<(), &'static str> {
    let id = client_id().ok_or("shared clipboard not connected")?;
    let wire = utf8_to_wire(text);
    if wire.len() > MAX_BYTES {
        return Err("clipboard too large to send");
    }
    // Two parameters without a context ID (VBOX_SHCL_CPARMS_DATA_WRITE_OLD):
    // the format bit, then the data the host reads.
    let addr = wire.as_ptr() as u64;
    super::hgcm_call(
        id,
        FN_DATA_WRITE,
        &[Parm::U32(format), Parm::buf_in(addr, wire.len() as u32)],
    )
    .map(|_| ())
}

/// Fetch the host's clipboard in `format`.
fn read_data(format: u32) -> Result<String, &'static str> {
    let id = client_id().ok_or("shared clipboard not connected")?;
    let mut buf: Vec<u8> = alloc::vec![0u8; 64 * 1024];
    // Three parameters: format, the buffer the host writes, and the byte count
    // it actually wrote (VBOX_SHCL_CPARMS_DATA_READ == 3).
    let call = |buf: &mut Vec<u8>| -> Result<u32, &'static str> {
        let addr = buf.as_ptr() as u64;
        let len = buf.len() as u32;
        super::hgcm_call(
            id,
            FN_DATA_READ,
            &[Parm::U32(format), Parm::buf_out(addr, len), Parm::U32(0)],
        )
    };
    let n = call(&mut buf)? as usize;
    // The host reports the size it needs when our buffer was too small, and
    // copies nothing — so a big paste needs exactly one retry, not a doubling
    // search. Same shape as the ring-3 image tenant's HEAP_WANT.
    if n > buf.len() {
        if n > MAX_BYTES {
            return Err("host clipboard is larger than we will accept");
        }
        buf.resize(n, 0);
        let n2 = call(&mut buf)? as usize;
        return Ok(wire_to_utf8(&buf[..n2.min(buf.len())]));
    }
    Ok(wire_to_utf8(&buf[..n]))
}

/// Post the outstanding "tell me when something happens" call.
///
/// It stays pending in the host until there is a message, which is what makes a
/// polled design work without a blocking thread.
fn arm_message_wait() {
    let Some(id) = client_id() else { return };
    let already = CLIENT.with(|c| c.as_ref().map(|c| c.waiting).unwrap_or(false));
    if already {
        return;
    }
    // Two 32-bit outputs: the message id and its format bits
    // (VBOX_SHCL_CPARMS_GET_HOST_MSG_OLD == 2).
    if super::hgcm_post_message_wait(id, FN_MSG_OLD_GET_WAIT, &[Parm::U32(0), Parm::U32(0)]) {
        CLIENT.with(|c| {
            if let Some(c) = c.as_mut() {
                c.waiting = true;
            }
        });
    }
}

/// Poll the outstanding message call and act on anything the host sent.
///
/// Called from the UI pump, so it must never block: it only ever *checks* the
/// done flag.
pub fn tick() {
    if !connected() {
        return;
    }
    let Some((msg, formats)) = super::hgcm_take_message() else {
        // Nothing yet — or nothing armed, which can happen after an error.
        arm_message_wait();
        return;
    };
    CLIENT.with(|c| {
        if let Some(c) = c.as_mut() {
            c.waiting = false;
        }
    });

    match msg {
        MSG_FORMATS_REPORT => {
            // The host copied something. Fetch it now rather than at paste
            // time: there is one clipboard here and no owner to ask later.
            CLIENT.with(|c| {
                if let Some(c) = c.as_mut() {
                    c.host_formats = formats;
                }
            });
            if formats & FMT_UNICODETEXT != 0 {
                match read_data(FMT_UNICODETEXT) {
                    // `set_from_host`, never `set` — `set` would report the
                    // formats straight back and bounce it to the host forever.
                    Ok(text) if !text.is_empty() => crate::clipboard::set_from_host(text),
                    Ok(_) => {}
                    Err(e) => crate::ktrace::log_fmt(format_args!("vbox: clipboard read failed: {e}")),
                }
            }
        }
        MSG_READ_DATA => {
            // The host is pasting and wants our contents.
            let text = crate::clipboard::get().map(|(t, _)| t).unwrap_or_default();
            let fmt = if formats & FMT_UNICODETEXT != 0 { FMT_UNICODETEXT } else { formats };
            if let Err(e) = write_data(fmt, &text) {
                crate::ktrace::log_fmt(format_args!("vbox: clipboard write failed: {e}"));
            }
        }
        MSG_QUIT => {
            crate::ktrace::log("vbox", "shared clipboard: host said quit; disconnecting");
            CLIENT.with(|c| *c = None);
            return;
        }
        MSG_CANCELED => {}
        other => crate::ktrace::log_fmt(format_args!("vbox: unhandled clipboard message {other}")),
    }
    arm_message_wait();
}

/// One line for `/clip` and `/vbox`.
pub fn status() -> Option<String> {
    CLIENT.with(|c| {
        c.as_ref().map(|c| {
            alloc::format!(
                "VirtualBox shared clipboard connected (client {}, host holds {})",
                c.id,
                if c.host_formats & FMT_UNICODETEXT != 0 { "text" } else { "nothing we read" }
            )
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn text_goes_out_as_utf16le_with_crlf_and_a_terminator() {
        // The format is UTF-16LE, not UTF-8 — sending UTF-8 produces text, just
        // the wrong text, which is why this is pinned rather than assumed.
        let w = utf8_to_wire("hi");
        assert_eq!(w, alloc::vec![b'h', 0, b'i', 0, 0, 0]);

        // A lone LF becomes CRLF, as VirtualBox's own Linux additions do.
        let w = utf8_to_wire("a\nb");
        assert_eq!(w, alloc::vec![b'a', 0, 0x0d, 0, 0x0a, 0, b'b', 0, 0, 0]);

        // An existing CRLF is left alone rather than becoming CRCRLF.
        let w = utf8_to_wire("a\r\nb");
        assert_eq!(w, alloc::vec![b'a', 0, 0x0d, 0, 0x0a, 0, b'b', 0, 0, 0]);
    }

    #[test_case]
    fn text_comes_back_as_utf8_without_cr_or_the_terminator() {
        assert_eq!(wire_to_utf8(&[b'h', 0, b'i', 0, 0, 0]), "hi");
        assert_eq!(wire_to_utf8(&[b'a', 0, 0x0d, 0, 0x0a, 0, b'b', 0]), "a\nb");
        // Everything past the NUL terminator is padding, not text — a fixed
        // buffer the host only partly filled would otherwise come back with a
        // tail of zero characters.
        assert_eq!(wire_to_utf8(&[b'x', 0, 0, 0, b'y', 0]), "x");
        assert_eq!(wire_to_utf8(&[]), "");
    }

    #[test_case]
    fn non_ascii_and_astral_characters_survive_the_round_trip() {
        // Anything outside the BMP is a surrogate pair in UTF-16, which is the
        // case a naive "one char, one u16" conversion loses.
        for s in ["café", "日本語", "emoji: \u{1f600}", "mixed \u{1f4cb} and é"] {
            let round = wire_to_utf8(&utf8_to_wire(s));
            assert_eq!(round, s, "round trip lost {s:?}");
        }
    }

    #[test_case]
    fn a_malformed_wire_buffer_is_lossy_rather_than_fatal() {
        // An unpaired surrogate: a host clipboard may hold something we cannot
        // represent, and losing the whole paste is worse than one replacement
        // character.
        let bad = [0x00, 0xd8, b'a', 0]; // high surrogate with no low, then 'a'
        let s = wire_to_utf8(&bad);
        assert!(s.contains('\u{fffd}'), "unpaired surrogate should be replaced");
        assert!(s.ends_with('a'));
        // An odd trailing byte is dropped rather than read past the end.
        assert_eq!(wire_to_utf8(&[b'h', 0, b'i']), "h");
    }

    #[test_case]
    fn the_service_name_and_format_bit_are_what_the_host_expects() {
        assert_eq!(SERVICE, "VBoxSharedClipboard");
        assert_eq!(FMT_UNICODETEXT, 1);
        // The message ids the host sends us.
        assert_eq!((MSG_QUIT, MSG_READ_DATA, MSG_FORMATS_REPORT, MSG_CANCELED), (1, 2, 3, 4));
        // And the guest functions we call.
        assert_eq!((FN_MSG_OLD_GET_WAIT, FN_REPORT_FORMATS, FN_DATA_READ, FN_DATA_WRITE), (1, 2, 3, 4));
    }
}
