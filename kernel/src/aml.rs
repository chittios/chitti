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

// --- namespace structure --------------------------------------------------
//
// `find_name` above is a scan: it finds *a* `_CRS`, not a particular device's. That is
// useless for the case that motivated this module — a laptop has several I2C devices
// and only one is the touchpad — so the containers have to be walked.
//
// This is structure-aware without a full opcode table, which is the pragmatic middle
// ground. `Scope` and `Device` both carry a PkgLength, so their extent is known
// exactly and they can be descended into and skipped over reliably. Within a body,
// names are still found by scanning, because skipping an *arbitrary* opcode requires
// knowing every opcode's encoding — that is what full evaluation needs and this
// deliberately does not attempt.
//
// The result is enough to answer "what is *this* device's `_CRS`", which is the
// question a driver actually asks.

/// A `Device (...)` found in the namespace, with the byte range of its body.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceNode {
    /// Dotted path including enclosing scopes, e.g. `\_SB.PCI0.I2C1.TPD0`.
    pub path: String,
    /// Byte range of the device's term list within the AML passed to [`devices`].
    pub body: core::ops::Range<usize>,
    /// Byte range of the whole `Device(...)` construct, including its opcode, length
    /// and name — what a parent must skip over entirely.
    pub extent: core::ops::Range<usize>,
}

impl DeviceNode {
    /// The device's own name (last path segment).
    pub fn name(&self) -> &str {
        self.path.rsplit('.').next().unwrap_or(&self.path)
    }
}

/// How deeply to descend through nested Scope/Device containers.
const MAX_NESTING: usize = 12;

/// Every `Device` in `aml`, with its full namespace path and body range.
///
/// Nested devices are reported as well as their parents — a touchpad is commonly
/// `Device` inside the I2C controller's `Device` — so a caller can look at whichever
/// level it cares about.
pub fn devices(aml: &[u8]) -> Vec<DeviceNode> {
    let mut out = Vec::new();
    walk(aml, 0, "", 0, &mut out);
    out
}

/// Recursive container walk. `base` is the enclosing path, `off` the offset of `aml`
/// within the original buffer so reported ranges are absolute.
fn walk(aml: &[u8], off: usize, base: &str, depth: usize, out: &mut Vec<DeviceNode>) {
    if depth > MAX_NESTING {
        return;
    }
    let mut i = 0usize;
    while i < aml.len() {
        // Scope(...) is one byte; Device(...) is the two-byte extended opcode.
        let (header, is_device) = match aml[i] {
            SCOPE_OP => (1usize, false),
            EXT_OP_PREFIX if aml.get(i + 1) == Some(&DEVICE_OP) => (2usize, true),
            _ => {
                i += 1;
                continue;
            }
        };
        let Some((total, lead)) = aml.get(i + header..).and_then(pkg_length) else {
            i += 1;
            continue;
        };
        // PkgLength counts itself, so the container ends here.
        let end = i + header + total;
        if total < lead || end > aml.len() {
            i += 1;
            continue;
        }
        let name_at = i + header + lead;
        let Some((name, used)) = aml.get(name_at..end).and_then(name_string) else {
            i += 1;
            continue;
        };
        let body_start = name_at + used;
        if body_start > end {
            i += 1;
            continue;
        }
        let path = join_path(base, &name);
        if is_device {
            out.push(DeviceNode {
                path: path.clone(),
                body: off + body_start..off + end,
                // The container's own bytes (opcode, length, name) must be excluded
                // from a parent's search too, or the child's name would be found there.
                extent: off + i..off + end,
            });
        }
        // Descend, then skip the whole container — this is why knowing the extent
        // matters: a scan would re-find the nested devices at the wrong path.
        walk(&aml[body_start..end], off + body_start, &path, depth + 1, out);
        i = end;
    }
}

/// Join an enclosing path with a (possibly root- or parent-prefixed) name.
///
/// A leading `\` restarts from the root — ignoring that would nest an absolute path
/// under whatever happened to enclose it. Each leading `^` climbs one level.
fn join_path(base: &str, name: &str) -> String {
    if let Some(rest) = name.strip_prefix('\\') {
        let mut s = String::from("\\");
        s.push_str(rest);
        return s;
    }
    let mut up = 0usize;
    let mut rest = name;
    while let Some(r) = rest.strip_prefix('^') {
        up += 1;
        rest = r;
    }
    let mut parts: Vec<&str> = base.split('.').filter(|p| !p.is_empty()).collect();
    for _ in 0..up {
        parts.pop();
    }
    let mut s = String::new();
    if base.starts_with('\\') {
        s.push('\\');
    }
    for (k, p) in parts.iter().enumerate() {
        let p = p.trim_start_matches('\\');
        if p.is_empty() {
            continue;
        }
        if k > 0 || !s.is_empty() && s != "\\" {
            if !s.is_empty() && !s.ends_with('\\') {
                s.push('.');
            }
        }
        s.push_str(p);
    }
    if !rest.is_empty() {
        if !s.is_empty() && !s.ends_with('\\') {
            s.push('.');
        }
        s.push_str(rest);
    }
    s
}

/// Look up a named object **belonging to one device**, excluding its children.
///
/// This is the per-device answer `find_name` cannot give. The subtlety is that a
/// parent's body *contains* its nested devices, so searching the range naively makes a
/// parent inherit its children's names — an I2C controller would report the touchpad's
/// `_HID` as its own, which is precisely the bug the tests caught. Child ranges are
/// therefore excluded and only the gaps between them searched.
pub fn device_name(aml: &[u8], dev: &DeviceNode, name: &str) -> Option<Value> {
    let all = devices(aml);
    // Direct or indirect children: any device whose body sits strictly inside this one.
    let mut kids: Vec<core::ops::Range<usize>> = all
        .iter()
        .filter(|d| d.path != dev.path && d.extent.start >= dev.body.start && d.extent.end <= dev.body.end)
        .map(|d| d.extent.clone())
        .collect();
    kids.sort_by_key(|r| r.start);

    let mut cursor = dev.body.start;
    for k in &kids {
        if k.start > cursor {
            if let Some(v) = aml.get(cursor..k.start).and_then(|w| find_name(w, name)) {
                return Some(v);
            }
        }
        cursor = cursor.max(k.end);
    }
    if cursor < dev.body.end {
        if let Some(v) = aml.get(cursor..dev.body.end).and_then(|w| find_name(w, name)) {
            return Some(v);
        }
    }
    None
}

/// Find the device whose `_HID` (or `_CID`) matches `hid`.
///
/// `PNP0C50` is the HID-over-I2C touchpad identifier, which is what makes this the
/// replacement for guessing an I2C address: instead of probing every connection, ask
/// the namespace which device claims to be a touchpad and read *its* `_CRS`.
pub fn device_by_hid(aml: &[u8], hid: &str) -> Option<DeviceNode> {
    devices(aml).into_iter().find(|d| {
        for key in ["_HID", "_CID"] {
            if let Some(Value::String(s)) = device_name(aml, d, key) {
                if s == hid {
                    return true;
                }
            }
        }
        false
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    // --- namespace structure ----------------------------------------------

    /// Encode a PkgLength for a body of `n` bytes, returning the bytes.
    ///
    /// The encoding must account for its own size, which is the trap: a body needing
    /// a 2-byte length has a different total than the same body with a 1-byte one.
    fn pkglen(n: usize) -> vec::Vec<u8> {
        let one = n + 1;
        if one <= 0x3f {
            return vec![one as u8];
        }
        let two = n + 2;
        vec![0x40 | (two & 0x0f) as u8, (two >> 4) as u8]
    }

    fn seg(name: &str) -> vec::Vec<u8> {
        let mut v: vec::Vec<u8> = name.bytes().collect();
        while v.len() < 4 {
            v.push(b'_');
        }
        v
    }

    /// `Device (name) { body }`
    fn device(name: &str, body: &[u8]) -> vec::Vec<u8> {
        let mut inner = seg(name);
        inner.extend_from_slice(body);
        let mut v = vec![EXT_OP_PREFIX, DEVICE_OP];
        v.extend_from_slice(&pkglen(inner.len()));
        v.extend_from_slice(&inner);
        v
    }

    /// `Scope (name) { body }`
    fn scope(name: &str, body: &[u8]) -> vec::Vec<u8> {
        let mut inner = seg(name);
        inner.extend_from_slice(body);
        let mut v = vec![SCOPE_OP];
        v.extend_from_slice(&pkglen(inner.len()));
        v.extend_from_slice(&inner);
        v
    }

    /// `Name (name, "value")`
    fn name_str(name: &str, value: &str) -> vec::Vec<u8> {
        let mut v = vec![NAME_OP];
        v.extend_from_slice(&seg(name));
        v.push(STRING_PREFIX);
        v.extend_from_slice(value.as_bytes());
        v.push(0);
        v
    }

    /// A DSDT body with two I2C devices, only one of which is a touchpad — the exact
    /// situation that makes a per-device lookup necessary.
    fn two_i2c_devices() -> vec::Vec<u8> {
        let mut tpd = name_str("_HID", "PNP0C50");
        tpd.extend_from_slice(&{
            // Name(_CRS, Buffer(2){0xAA,0xBB})
            let mut v = vec![NAME_OP];
            v.extend_from_slice(&seg("_CRS"));
            v.push(BUFFER_OP);
            let body = [BYTE_PREFIX, 0x02, 0xaa, 0xbb];
            v.extend_from_slice(&pkglen(body.len()));
            v.extend_from_slice(&body);
            v
        });
        let mut sensor = name_str("_HID", "ACPI0C60");
        sensor.extend_from_slice(&{
            let mut v = vec![NAME_OP];
            v.extend_from_slice(&seg("_CRS"));
            v.push(BUFFER_OP);
            let body = [BYTE_PREFIX, 0x02, 0x11, 0x22];
            v.extend_from_slice(&pkglen(body.len()));
            v.extend_from_slice(&body);
            v
        });
        let mut i2c_body = device("TPD0", &tpd);
        i2c_body.extend_from_slice(&device("SEN0", &sensor));
        let i2c = device("I2C1", &i2c_body);
        scope("_SB_", &i2c)
    }

    #[test_case]
    fn walks_nested_devices_with_full_paths() {
        let aml = two_i2c_devices();
        let devs = devices(&aml);
        let paths: vec::Vec<&str> = devs.iter().map(|d| d.path.as_str()).collect();
        // The enclosing Scope must contribute to the path, and nested devices must be
        // reported under their parent — not at the root, which is what a flat scan
        // would produce.
        assert!(paths.contains(&"_SB.I2C1"), "{paths:?}");
        assert!(paths.contains(&"_SB.I2C1.TPD0"), "{paths:?}");
        assert!(paths.contains(&"_SB.I2C1.SEN0"), "{paths:?}");
    }

    #[test_case]
    fn resolves_the_right_devices_crs() {
        // The whole point: two devices each have a _CRS, and a scan would return
        // whichever came first. Per-device lookup must return each one's own.
        let aml = two_i2c_devices();
        let devs = devices(&aml);
        let tpd = devs.iter().find(|d| d.name() == "TPD0").unwrap();
        let sen = devs.iter().find(|d| d.name() == "SEN0").unwrap();
        assert_eq!(
            device_name(&aml, tpd, "_CRS").and_then(|v| v.as_buffer().map(|b| b.to_vec())),
            Some(vec![0xaa, 0xbb])
        );
        assert_eq!(
            device_name(&aml, sen, "_CRS").and_then(|v| v.as_buffer().map(|b| b.to_vec())),
            Some(vec![0x11, 0x22])
        );
    }

    #[test_case]
    fn finds_the_touchpad_by_hid() {
        // PNP0C50 is the HID-over-I2C identifier. This is what replaces guessing an
        // I2C address: ask which device claims to be a touchpad, then read its _CRS.
        let aml = two_i2c_devices();
        let dev = device_by_hid(&aml, "PNP0C50").unwrap();
        assert_eq!(dev.name(), "TPD0");
        assert_eq!(
            device_name(&aml, &dev, "_CRS").and_then(|v| v.as_buffer().map(|b| b.to_vec())),
            Some(vec![0xaa, 0xbb])
        );
        // And a HID nothing claims yields nothing, rather than the first device.
        assert!(device_by_hid(&aml, "PNP9999").is_none());
    }

    #[test_case]
    fn a_devices_body_excludes_its_siblings() {
        // If the body range were wrong, TPD0's lookup would see SEN0's names and the
        // per-device guarantee would be an illusion.
        let aml = two_i2c_devices();
        let devs = devices(&aml);
        let tpd = devs.iter().find(|d| d.name() == "TPD0").unwrap();
        match device_name(&aml, tpd, "_HID") {
            Some(Value::String(s)) => assert_eq!(s, "PNP0C50"),
            other => panic!("unexpected {other:?}"),
        }
        let sen = devs.iter().find(|d| d.name() == "SEN0").unwrap();
        assert!(!tpd.body.contains(&sen.body.start), "TPD0 body swallowed SEN0");
        // And the parent must NOT inherit a child's name. A parent's body contains its
        // children, so a naive range search made the I2C controller report the
        // touchpad's _HID as its own — this is the regression guard for that.
        let i2c = devs.iter().find(|d| d.name() == "I2C1").unwrap();
        assert_eq!(device_name(&aml, i2c, "_HID"), None, "parent inherited a child's _HID");
        assert_eq!(device_name(&aml, i2c, "_CRS"), None, "parent inherited a child's _CRS");
    }

    #[test_case]
    fn root_prefixed_name_restarts_the_path() {
        // A `\`-prefixed name is absolute; nesting it under the enclosing scope would
        // produce a path that does not exist.
        // Input is what `name_string` produces, i.e. already underscore-trimmed.
        assert_eq!(join_path("_SB.PCI0", "\\_SB"), String::from("\\_SB"));
        // `^` climbs one level.
        assert_eq!(join_path("_SB.PCI0", "^ABCD"), String::from("_SB.ABCD"));
        assert_eq!(join_path("", "I2C1"), String::from("I2C1"));
    }

    #[test_case]
    fn malformed_containers_do_not_derail_the_walk() {
        // Firmware bytes: a truncated container must be skipped, not followed.
        let mut aml = vec![EXT_OP_PREFIX, DEVICE_OP, 0x7f]; // claims a long body
        aml.extend_from_slice(&two_i2c_devices());
        let devs = devices(&aml);
        assert!(devs.iter().any(|d| d.name() == "TPD0"), "{devs:?}");
        assert!(devices(&[]).is_empty());
        assert!(devices(&[EXT_OP_PREFIX]).is_empty());
    }


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
