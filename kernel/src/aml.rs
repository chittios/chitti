//! **AML decoding** — the byte-level layer of the ACPI Machine Language.
//!
//! ACPI describes anything that cannot be enumerated (an I2C touchpad's address, a
//! battery's state, backlight control) as *bytecode* in the DSDT, not as tables. Two
//! places in this kernel currently work around not being able to read it:
//! [`crate::acpi::parse_s5_package`] finds `\_S5_` by scanning for a byte pattern,
//! and [`crate::drivers::i2c_hid`] guesses the HID descriptor register because it
//! comes from a `_DSM` *method*. Both are honest workarounds; neither generalises.
//!
//! This module is the decoding foundation an interpreter needs, kept separate and
//! pure so it is unit-testable off-hardware — the same split that worked for
//! `block::fat32` and `net::nic_ids`. It parses; it does not yet evaluate.
//!
//! ## The encodings that actually cause bugs
//!
//! * **PkgLength** is variable-length and *self-describing*: the top two bits of the
//!   first byte say how many more bytes follow, and — the part that trips people —
//!   the low nibble of the first byte contributes the *least* significant 4 bits when
//!   there are following bytes, but the low **six** bits when there are none. Getting
//!   this wrong walks into the middle of an object.
//! * **NameString** has five forms (root-prefixed, carat-prefixed, null, dual,
//!   multi), so a fixed 4-byte read finds the wrong name on anything but the simplest
//!   case.
//! * Integers come in five widths plus two singleton opcodes, and `OnesOp` means
//!   "all bits set at the current integer width", not the byte `0xFF`.

use alloc::string::String;
use alloc::vec::Vec;

// --- opcodes we decode ----------------------------------------------------
pub const ZERO_OP: u8 = 0x00;
pub const ONE_OP: u8 = 0x01;
pub const BYTE_PREFIX: u8 = 0x0a;
pub const WORD_PREFIX: u8 = 0x0b;
pub const DWORD_PREFIX: u8 = 0x0c;
pub const STRING_PREFIX: u8 = 0x0d;
pub const QWORD_PREFIX: u8 = 0x0e;
pub const BUFFER_OP: u8 = 0x11;
pub const PACKAGE_OP: u8 = 0x12;
pub const ONES_OP: u8 = 0xff;
/// `NameOp` — introduces `Name(Target, Object)`.
pub const NAME_OP: u8 = 0x08;
/// `ScopeOp`.
pub const SCOPE_OP: u8 = 0x10;
/// `MethodOp`.
pub const METHOD_OP: u8 = 0x14;
/// Two-byte opcodes are prefixed with this.
pub const EXT_OP_PREFIX: u8 = 0x5b;
/// `DeviceOp` (after [`EXT_OP_PREFIX`]).
pub const DEVICE_OP: u8 = 0x82;

const ROOT_CHAR: u8 = b'\\';
const PARENT_PREFIX: u8 = b'^';
const DUAL_NAME_PREFIX: u8 = 0x2e;
const MULTI_NAME_PREFIX: u8 = 0x2f;
const NULL_NAME: u8 = 0x00;

/// A decoded AML data object. Only the forms a resource or a simple `Name` can hold —
/// enough to read `_CRS`, `_HID` and `_S5_` without evaluating anything.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Value {
    Integer(u64),
    String(String),
    Buffer(Vec<u8>),
    Package(Vec<Value>),
}

impl Value {
    /// The integer value, if this is one.
    pub fn as_int(&self) -> Option<u64> {
        match self {
            Value::Integer(v) => Some(*v),
            _ => None,
        }
    }

    /// The bytes, if this is a buffer — how `_CRS` returns a resource template.
    pub fn as_buffer(&self) -> Option<&[u8]> {
        match self {
            Value::Buffer(b) => Some(b),
            _ => None,
        }
    }
}

/// Decode a **PkgLength**: returns `(payload_length, bytes_consumed_by_the_length)`.
///
/// The payload length *includes* the length encoding itself, which is why callers
/// need both numbers: the object's body runs from `consumed` to `payload_length`.
pub fn pkg_length(d: &[u8]) -> Option<(usize, usize)> {
    let first = *d.first()?;
    let extra = (first >> 6) as usize;
    if d.len() < 1 + extra {
        return None;
    }
    if extra == 0 {
        // No following bytes: the low SIX bits are the whole length.
        return Some(((first & 0x3f) as usize, 1));
    }
    // With following bytes, only the low four bits of the first byte contribute, as
    // the least significant nibble.
    let mut len = (first & 0x0f) as usize;
    for i in 0..extra {
        len |= (d[1 + i] as usize) << (4 + 8 * i);
    }
    Some((len, 1 + extra))
}

fn is_lead_name_char(c: u8) -> bool {
    c == b'_' || c.is_ascii_uppercase()
}
fn is_name_char(c: u8) -> bool {
    is_lead_name_char(c) || c.is_ascii_digit()
}

/// Decode one 4-byte NameSeg, trailing underscores trimmed.
fn name_seg(d: &[u8]) -> Option<String> {
    if d.len() < 4 || !is_lead_name_char(d[0]) || !d[1..4].iter().all(|&c| is_name_char(c)) {
        return None;
    }
    let mut s = String::new();
    for &c in &d[..4] {
        s.push(c as char);
    }
    // ACPI pads short names with '_'; keep the canonical trimmed form.
    while s.ends_with('_') && s.len() > 1 {
        s.pop();
    }
    Some(s)
}

/// Decode a **NameString**, returning the dotted path and how many bytes it took.
///
/// Handles all five forms. `\` marks a root-relative path and `^` each step up; both
/// are preserved in the returned string so a caller can tell `\_SB.PCI0` from a
/// relative `PCI0`.
pub fn name_string(d: &[u8]) -> Option<(String, usize)> {
    let mut i = 0usize;
    let mut out = String::new();
    if d.first() == Some(&ROOT_CHAR) {
        out.push('\\');
        i += 1;
    } else {
        while d.get(i) == Some(&PARENT_PREFIX) {
            out.push('^');
            i += 1;
        }
    }
    match d.get(i) {
        None => return None,
        Some(&NULL_NAME) => return Some((out, i + 1)),
        Some(&DUAL_NAME_PREFIX) => {
            let a = name_seg(d.get(i + 1..)?)?;
            let b = name_seg(d.get(i + 5..)?)?;
            out.push_str(&a);
            out.push('.');
            out.push_str(&b);
            Some((out, i + 9))
        }
        Some(&MULTI_NAME_PREFIX) => {
            let n = *d.get(i + 1)? as usize;
            // A zero-segment multi-name is malformed; and the segments must all be
            // present or we would read past the buffer.
            if n == 0 || d.len() < i + 2 + 4 * n {
                return None;
            }
            for k in 0..n {
                if k > 0 {
                    out.push('.');
                }
                out.push_str(&name_seg(&d[i + 2 + 4 * k..])?);
            }
            Some((out, i + 2 + 4 * n))
        }
        Some(_) => {
            out.push_str(&name_seg(&d[i..])?);
            Some((out, i + 4))
        }
    }
}

/// Decode a data object: `(value, bytes_consumed)`.
///
/// Recursion through packages is depth-bounded, because this parses firmware that may
/// be malformed or hostile and a self-referential package must not exhaust the stack.
pub fn data_object(d: &[u8]) -> Option<(Value, usize)> {
    data_object_depth(d, 0)
}

/// How deep nested packages may go. Real tables use two or three levels.
const MAX_DEPTH: usize = 16;

fn data_object_depth(d: &[u8], depth: usize) -> Option<(Value, usize)> {
    if depth > MAX_DEPTH {
        return None;
    }
    let op = *d.first()?;
    match op {
        ZERO_OP => Some((Value::Integer(0), 1)),
        ONE_OP => Some((Value::Integer(1), 1)),
        // `OnesOp` is "all bits set", not the byte 0xFF.
        ONES_OP => Some((Value::Integer(u64::MAX), 1)),
        BYTE_PREFIX => Some((Value::Integer(*d.get(1)? as u64), 2)),
        WORD_PREFIX => {
            let b = d.get(1..3)?;
            Some((Value::Integer(u16::from_le_bytes([b[0], b[1]]) as u64), 3))
        }
        DWORD_PREFIX => {
            let b = d.get(1..5)?;
            Some((Value::Integer(u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u64), 5))
        }
        QWORD_PREFIX => {
            let b = d.get(1..9)?;
            let mut v = [0u8; 8];
            v.copy_from_slice(b);
            Some((Value::Integer(u64::from_le_bytes(v)), 9))
        }
        STRING_PREFIX => {
            let rest = d.get(1..)?;
            let end = rest.iter().position(|&c| c == 0)?;
            let mut s = String::new();
            for &c in &rest[..end] {
                s.push(c as char);
            }
            Some((Value::String(s), 2 + end))
        }
        BUFFER_OP => {
            let (total, lead) = pkg_length(d.get(1..)?)?;
            if total < lead || 1 + total > d.len() {
                return None;
            }
            // Inside: a TermArg giving the buffer size, then the initialiser bytes.
            let body = &d[1 + lead..1 + total];
            let (_size, used) = data_object_depth(body, depth + 1)?;
            Some((Value::Buffer(body.get(used..)?.to_vec()), 1 + total))
        }
        PACKAGE_OP => {
            let (total, lead) = pkg_length(d.get(1..)?)?;
            if total < lead || 1 + total > d.len() {
                return None;
            }
            let body = &d[1 + lead..1 + total];
            let n = *body.first()? as usize;
            let mut items = Vec::with_capacity(n.min(64));
            let mut off = 1usize;
            for _ in 0..n {
                if off >= body.len() {
                    break; // a package may declare more elements than it encodes
                }
                let (v, used) = data_object_depth(&body[off..], depth + 1)?;
                items.push(v);
                off += used;
            }
            Some((Value::Package(items), 1 + total))
        }
        _ => None,
    }
}

/// Find a top-level `Name(<path>, <object>)` whose path ends with `name`, and decode
/// its object.
///
/// This is what replaces pattern-scanning for `_S5_` and makes `_CRS` and `_HID`
/// reachable by name: it confirms a real `NameOp` introduces the path, rather than
/// matching four bytes that happen to appear inside something else.
///
/// Still a scan rather than a namespace walk — it does not track scope, so it finds
/// *a* `_CRS`, not a particular device's. That is the next layer up.
pub fn find_name(aml: &[u8], name: &str) -> Option<Value> {
    let mut i = 0usize;
    while i < aml.len() {
        if aml[i] != NAME_OP {
            i += 1;
            continue;
        }
        let Some((path, used)) = name_string(aml.get(i + 1..)?) else {
            i += 1;
            continue;
        };
        let matches = path == name || path.rsplit('.').next() == Some(name);
        if !matches {
            i += 1;
            continue;
        }
        if let Some((v, _)) = data_object(aml.get(i + 1 + used..)?) {
            return Some(v);
        }
        i += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test_case]
    fn pkg_length_single_byte_uses_six_bits() {
        // No following bytes: the low six bits are the length. Masking four here (the
        // multi-byte rule) would under-read every short object.
        assert_eq!(pkg_length(&[0x05]), Some((5, 1)));
        assert_eq!(pkg_length(&[0x3f]), Some((0x3f, 1)));
    }

    #[test_case]
    fn pkg_length_multi_byte_uses_low_nibble_as_least_significant() {
        // 0x41 0x02 => extra=1, low nibble 1, next byte 2 => 1 | (2 << 4) = 0x21.
        assert_eq!(pkg_length(&[0x41, 0x02]), Some((0x21, 2)));
        // Two following bytes.
        assert_eq!(pkg_length(&[0x82, 0x34, 0x12]), Some((0x2 | (0x34 << 4) | (0x12 << 12), 3)));
        // Truncated: must refuse rather than read past.
        assert_eq!(pkg_length(&[0x41]), None);
        assert_eq!(pkg_length(&[0x82, 0x34]), None);
        assert_eq!(pkg_length(&[]), None);
    }

    #[test_case]
    fn name_string_handles_every_form() {
        assert_eq!(name_string(b"_S5_"), Some((String::from("_S5"), 4)));
        // Root-prefixed, preserved so an absolute path is distinguishable.
        assert_eq!(name_string(b"\\_SB_"), Some((String::from("\\_SB"), 5)));
        // Parent prefixes.
        assert_eq!(name_string(b"^^ABCD"), Some((String::from("^^ABCD"), 6)));
        // Dual name.
        assert_eq!(name_string(b"\x2e_SB_PCI0"), Some((String::from("_SB.PCI0"), 9)));
        // Multi name: 3 segments.
        assert_eq!(
            name_string(b"\x2f\x03_SB_PCI0I2C1"),
            Some((String::from("_SB.PCI0.I2C1"), 14))
        );
        // Null name.
        assert_eq!(name_string(&[0x00]), Some((String::new(), 1)));
    }

    #[test_case]
    fn name_string_refuses_malformed_names() {
        // Lowercase and punctuation are not legal name characters, so this is not a
        // name — important, because a scan hits arbitrary bytes constantly.
        assert_eq!(name_string(b"abcd"), None);
        assert_eq!(name_string(b"1BCD"), None); // digit cannot lead
        assert_eq!(name_string(b"AB"), None); // too short
        // Multi-name with zero segments, or fewer bytes than it claims.
        assert_eq!(name_string(b"\x2f\x00"), None);
        assert_eq!(name_string(b"\x2f\x04_SB_"), None);
    }

    #[test_case]
    fn decodes_integers_of_every_width() {
        assert_eq!(data_object(&[ZERO_OP]), Some((Value::Integer(0), 1)));
        assert_eq!(data_object(&[ONE_OP]), Some((Value::Integer(1), 1)));
        // OnesOp is all-bits-set, not the byte 0xFF.
        assert_eq!(data_object(&[ONES_OP]), Some((Value::Integer(u64::MAX), 1)));
        assert_eq!(data_object(&[BYTE_PREFIX, 0x2c]), Some((Value::Integer(0x2c), 2)));
        assert_eq!(data_object(&[WORD_PREFIX, 0x34, 0x12]), Some((Value::Integer(0x1234), 3)));
        assert_eq!(
            data_object(&[DWORD_PREFIX, 0x78, 0x56, 0x34, 0x12]),
            Some((Value::Integer(0x1234_5678), 5))
        );
        assert_eq!(
            data_object(&[QWORD_PREFIX, 1, 0, 0, 0, 0, 0, 0, 0]),
            Some((Value::Integer(1), 9))
        );
        // Truncated integers must be refused, not zero-filled.
        assert_eq!(data_object(&[WORD_PREFIX, 0x34]), None);
        assert_eq!(data_object(&[DWORD_PREFIX, 1, 2]), None);
    }

    #[test_case]
    fn decodes_strings_and_requires_termination() {
        assert_eq!(
            data_object(b"\x0dPNP0C50\x00"),
            Some((Value::String(String::from("PNP0C50")), 9))
        );
        // Unterminated: refused rather than running to the end of the table.
        assert_eq!(data_object(b"\x0dPNP0C50"), None);
    }

    #[test_case]
    fn decodes_a_buffer_the_way_crs_returns_one() {
        // Buffer(4){1,2,3,4}: BufferOp, PkgLength, size TermArg, then bytes.
        //
        // The declared length is 7, not 6: PkgLength counts **itself** plus everything
        // after it (1 length byte + 2 for the size TermArg + 4 data). An earlier
        // version of this fixture said 6 and the test caught it — the parser was
        // right and the hand-built bytes were wrong, which is exactly the mistake
        // this encoding invites.
        let d = [BUFFER_OP, 0x07, BYTE_PREFIX, 0x04, 1, 2, 3, 4];
        let (v, used) = data_object(&d).unwrap();
        assert_eq!(v.as_buffer(), Some(&[1u8, 2, 3, 4][..]));
        assert_eq!(used, 8); // 1 opcode + declared length 7
    }

    #[test_case]
    fn decodes_a_package_of_integers() {
        // Package(2){5,5} — the shape \_S5_ has.
        // PkgLength 6 = itself (1) + NumElements (1) + two 2-byte integers (4).
        let d = [PACKAGE_OP, 0x06, 0x02, BYTE_PREFIX, 5, BYTE_PREFIX, 5];
        let (v, _) = data_object(&d).unwrap();
        assert_eq!(
            v,
            Value::Package(vec![Value::Integer(5), Value::Integer(5)])
        );
    }

    #[test_case]
    fn package_declaring_more_elements_than_it_encodes_is_tolerated() {
        // Real firmware does this; stopping at the encoded end beats returning None
        // and losing the elements that are present.
        let d = [PACKAGE_OP, 0x04, 0x04, BYTE_PREFIX, 7];
        let (v, _) = data_object(&d).unwrap();
        assert_eq!(v, Value::Package(vec![Value::Integer(7)]));
    }

    #[test_case]
    fn nested_packages_are_depth_bounded() {
        // A deeply nested (or self-referential) package must not exhaust the stack:
        // this parses firmware, which may be malformed.
        let mut d = vec![];
        for _ in 0..40 {
            d.push(PACKAGE_OP);
            d.push(0x20);
            d.push(0x01);
        }
        d.push(ZERO_OP);
        // Either it decodes or it refuses — the property under test is that it returns.
        let _ = data_object(&d);
    }

    #[test_case]
    fn finds_a_named_object_via_a_real_nameop() {
        // Name(_S5_, Package(2){5,5}) preceded by bytes that merely contain "_S5_",
        // which a pattern scan would match. Requiring a NameOp is the difference.
        let mut d = vec![0xff, b'_', b'S', b'5', b'_', 0x00];
        d.extend_from_slice(&[NAME_OP, b'_', b'S', b'5', b'_']);
        d.extend_from_slice(&[PACKAGE_OP, 0x06, 0x02, BYTE_PREFIX, 5, BYTE_PREFIX, 5]);
        let v = find_name(&d, "_S5").unwrap();
        assert_eq!(v, Value::Package(vec![Value::Integer(5), Value::Integer(5)]));
    }

    #[test_case]
    fn find_name_matches_the_last_segment_of_a_path() {
        // Devices declare `Name(_HID, ...)` inside a scope, but a scan may also see
        // fully-qualified paths; both should resolve.
        let mut d = vec![NAME_OP, 0x2e, b'P', b'C', b'I', b'0', b'_', b'H', b'I', b'D'];
        d.extend_from_slice(b"\x0dPNP0C50\x00");
        assert_eq!(
            find_name(&d, "_HID"),
            Some(Value::String(String::from("PNP0C50")))
        );
    }

    #[test_case]
    fn find_name_returns_none_when_absent() {
        assert_eq!(find_name(&[], "_CRS"), None);
        assert_eq!(find_name(&[0x08, b'A', b'B', b'C', b'D', ZERO_OP], "_CRS"), None);
    }
}
