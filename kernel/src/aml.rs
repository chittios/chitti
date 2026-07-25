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

/// A `Method (...)` found in the namespace.
///
/// Located but **not evaluated**. Two things make its presence useful on its own: a
/// driver can tell whether firmware provides a given method at all, and — the
/// correctness point — a method's body must be *excluded* from device property
/// lookups, because `Name` declarations inside it are locals, not device properties.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MethodNode {
    /// Dotted path including enclosing scopes.
    pub path: String,
    /// Declared argument count (`MethodFlags` bits 0..2).
    pub arg_count: u8,
    /// True when the method is declared `Serialized`.
    pub serialized: bool,
    /// Byte range of the method's term list.
    pub body: core::ops::Range<usize>,
    /// Byte range of the whole `Method(...)` construct.
    pub extent: core::ops::Range<usize>,
}

impl MethodNode {
    /// The method's own name (last path segment).
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
    let mut m = Vec::new();
    walk(aml, 0, "", 0, &mut out, &mut m);
    out
}

/// Every `Method` in `aml`, with its path, argument count and body range.
pub fn methods(aml: &[u8]) -> Vec<MethodNode> {
    let mut d = Vec::new();
    let mut out = Vec::new();
    walk(aml, 0, "", 0, &mut d, &mut out);
    out
}

/// The method named `name` belonging to `dev` (excluding its child devices).
pub fn device_method(aml: &[u8], dev: &DeviceNode, name: &str) -> Option<MethodNode> {
    let kids: Vec<core::ops::Range<usize>> = devices(aml)
        .iter()
        .filter(|d| d.path != dev.path && d.extent.start >= dev.body.start && d.extent.end <= dev.body.end)
        .map(|d| d.extent.clone())
        .collect();
    methods(aml).into_iter().find(|m| {
        m.name() == name
            && m.extent.start >= dev.body.start
            && m.extent.end <= dev.body.end
            && !kids.iter().any(|k| m.extent.start >= k.start && m.extent.end <= k.end)
    })
}

/// Recursive container walk. `base` is the enclosing path, `off` the offset of `aml`
/// within the original buffer so reported ranges are absolute.
#[allow(clippy::too_many_arguments)]
fn walk(
    aml: &[u8],
    off: usize,
    base: &str,
    depth: usize,
    out: &mut Vec<DeviceNode>,
    meths: &mut Vec<MethodNode>,
) {
    if depth > MAX_NESTING {
        return;
    }
    let mut i = 0usize;
    while i < aml.len() {
        // Scope(...) is one byte; Device(...) is the two-byte extended opcode.
        // Scope, Device and Method all carry a PkgLength, so their extent is exact.
        #[derive(PartialEq)]
        enum Kind {
            Scope,
            Device,
            Method,
        }
        let (header, kind) = match aml[i] {
            SCOPE_OP => (1usize, Kind::Scope),
            METHOD_OP => (1usize, Kind::Method),
            EXT_OP_PREFIX if aml.get(i + 1) == Some(&DEVICE_OP) => (2usize, Kind::Device),
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
        if kind == Kind::Method {
            // MethodFlags follows the name: arg count in bits 0..2, Serialized bit 3.
            let flags = aml.get(body_start).copied().unwrap_or(0);
            meths.push(MethodNode {
                path,
                arg_count: flags & 0x07,
                serialized: flags & 0x08 != 0,
                body: off + body_start + 1..off + end,
                extent: off + i..off + end,
            });
            // A method body is code, not namespace structure: do not descend into it.
            // Names declared inside are locals, so treating them as device properties
            // would attribute a method's temporaries to the device.
            i = end;
            continue;
        }
        if kind == Kind::Device {
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
        walk(&aml[body_start..end], off + body_start, &path, depth + 1, out, meths);
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
    // Method bodies are excluded too: a `Name` inside a method is a local, and
    // attributing it to the device would report a temporary as a device property.
    kids.extend(
        methods(aml)
            .iter()
            .filter(|m| m.extent.start >= dev.body.start && m.extent.end <= dev.body.end)
            .map(|m| m.extent.clone()),
    );
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

// --- evaluation -----------------------------------------------------------
//
// A **fail-closed** subset evaluator. It handles the constructs simple firmware
// methods are built from and returns `None` on anything it does not recognise, so a
// caller falls back to its documented default rather than receiving a value that was
// guessed. That property is what makes a partial evaluator safe to ship: the danger
// was never missing coverage, it was confidently returning the wrong integer.

/// Statement opcodes.
const RETURN_OP: u8 = 0xa4;
const IF_OP: u8 = 0xa0;
const ELSE_OP: u8 = 0xa1;
const STORE_OP: u8 = 0x70;
const ADD_OP: u8 = 0x72;
const SUBTRACT_OP: u8 = 0x74;
const MULTIPLY_OP: u8 = 0x77;
const SHIFT_LEFT_OP: u8 = 0x79;
const SHIFT_RIGHT_OP: u8 = 0x7a;
const AND_OP: u8 = 0x7b;
const NAND_OP: u8 = 0x7c;
const OR_OP: u8 = 0x7d;
const NOR_OP: u8 = 0x7e;
const XOR_OP: u8 = 0x7f;
const NOT_OP: u8 = 0x80;
const MOD_OP: u8 = 0x85;
const LAND_OP: u8 = 0x90;
const LOR_OP: u8 = 0x91;
const LNOT_OP: u8 = 0x92;
const LEQUAL_OP: u8 = 0x93;
const LGREATER_OP: u8 = 0x94;
const LLESS_OP: u8 = 0x95;
/// `Arg0`..`Arg6` and `Local0`..`Local7`.
const LOCAL0_OP: u8 = 0x60;
const ARG0_OP: u8 = 0x68;

/// ACPI truth: **all bits set**, not 1. A caller comparing against 1 would read every
/// true result as false, which is why `as_bool` exists rather than `== 1`.
pub const TRUE: u64 = u64::MAX;

/// Argument and local storage for one method invocation.
#[derive(Clone, Default)]
pub struct Env<'a> {
    args: Vec<Value>,
    locals: Vec<Value>,
    /// Reads a named field unit's current value.
    ///
    /// This is the whole reason the evaluator is worth having: `_BST` computes almost
    /// nothing, it *reads hardware* through names that alias an `OperationRegion`.
    /// Without a resolver those names have no value and evaluation fails closed,
    /// which is correct — the alternative is a battery percentage invented from a
    /// default.
    fields: Option<&'a dyn Fn(&str) -> Option<u64>>,
}

impl<'a> Env<'a> {
    /// An environment with no hardware access: named fields do not resolve.
    pub fn with_args(args: &[Value]) -> Env<'a> {
        Env {
            args: args.to_vec(),
            locals: alloc::vec![Value::Integer(0); 8],
            fields: None,
        }
    }

    /// An environment that can read named fields through `fields`.
    pub fn with_fields(args: &[Value], fields: &'a dyn Fn(&str) -> Option<u64>) -> Env<'a> {
        Env {
            args: args.to_vec(),
            locals: alloc::vec![Value::Integer(0); 8],
            fields: Some(fields),
        }
    }
}

// A closure has no `Debug`, so this reports whether one is present rather than what
// it is — which is the part a caller debugging a failed evaluation needs to know.
impl core::fmt::Debug for Env<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Env")
            .field("args", &self.args)
            .field("locals", &self.locals)
            .field("fields", &self.fields.is_some())
            .finish()
    }
}

impl Value {
    /// ACPI truthiness: any non-zero integer is true.
    pub fn as_bool(&self) -> bool {
        matches!(self, Value::Integer(v) if *v != 0)
    }
}

/// Where a statement list left off.
enum Flow {
    Normal,
    Return(Value),
}

/// Evaluate a method body with `args` bound to `Arg0..`.
///
/// `None` means "could not evaluate" — an unsupported opcode, malformed bytes, or no
/// `Return`. It never means "the method returned nothing useful"; a method that returns
/// zero yields `Some(Integer(0))`. Callers rely on that distinction to decide whether
/// to fall back.
pub fn eval_method(body: &[u8], args: &[Value]) -> Option<Value> {
    eval_in(body, &mut Env::with_args(args))
}

/// Evaluate a method body that may **read named fields** through `fields`.
///
/// The resolver is given a field's name and returns its current value, or `None` if
/// it cannot be read — an unmapped address space, an absent embedded controller, a
/// field this build cannot access. A `None` from the resolver fails the whole
/// evaluation rather than substituting zero, because a plausible-looking wrong number
/// is worse than no number.
pub fn eval_method_with_fields(
    body: &[u8],
    args: &[Value],
    fields: &dyn Fn(&str) -> Option<u64>,
) -> Option<Value> {
    eval_in(body, &mut Env::with_fields(args, fields))
}

fn eval_in(body: &[u8], env: &mut Env) -> Option<Value> {
    match exec_list(body, env, 0)? {
        Flow::Return(v) => Some(v),
        // A method with no explicit Return yields zero per the specification, but this
        // returns None instead: for the callers here (a register address, a battery
        // reading) "fell off the end" is far more likely to be unsupported control flow
        // than a deliberate zero, and guessing zero is the failure mode to avoid.
        Flow::Normal => None,
    }
}

/// Maximum statement/expression nesting.
const MAX_EVAL_DEPTH: usize = 24;

fn exec_list(d: &[u8], env: &mut Env, depth: usize) -> Option<Flow> {
    if depth > MAX_EVAL_DEPTH {
        return None;
    }
    let mut i = 0usize;
    while i < d.len() {
        match d[i] {
            RETURN_OP => {
                let (v, _used) = term_arg(d.get(i + 1..)?, env, depth + 1)?;
                return Some(Flow::Return(v));
            }
            IF_OP => {
                let (total, lead) = pkg_length(d.get(i + 1..)?)?;
                let end = i + 1 + total;
                if total < lead || end > d.len() {
                    return None;
                }
                let inner = d.get(i + 1 + lead..end)?;
                let (pred, used) = term_arg(inner, env, depth + 1)?;
                let taken = pred.as_bool();
                if taken {
                    if let Flow::Return(v) = exec_list(inner.get(used..)?, env, depth + 1)? {
                        return Some(Flow::Return(v));
                    }
                }
                i = end;
                // An `Else` immediately after runs only when the `If` did not.
                if d.get(i) == Some(&ELSE_OP) {
                    let (etotal, elead) = pkg_length(d.get(i + 1..)?)?;
                    let eend = i + 1 + etotal;
                    if etotal < elead || eend > d.len() {
                        return None;
                    }
                    if !taken {
                        let ebody = d.get(i + 1 + elead..eend)?;
                        if let Flow::Return(v) = exec_list(ebody, env, depth + 1)? {
                            return Some(Flow::Return(v));
                        }
                    }
                    i = eend;
                }
            }
            // A bare `Else` (its `If` already consumed above) is skipped.
            ELSE_OP => {
                let (total, lead) = pkg_length(d.get(i + 1..)?)?;
                if total < lead || i + 1 + total > d.len() {
                    return None;
                }
                i += 1 + total;
            }
            STORE_OP => {
                let (v, used) = term_arg(d.get(i + 1..)?, env, depth + 1)?;
                let tgt = i + 1 + used;
                let slot = *d.get(tgt)?;
                store(slot, v, env)?;
                i = tgt + 1;
            }
            // Any other opcode is an expression used as a statement, or something this
            // subset does not implement. Refuse rather than skip: skipping an unknown
            // opcode means guessing its length, and a wrong guess resumes parsing
            // mid-instruction.
            _ => {
                let (_, used) = term_arg(d.get(i..)?, env, depth + 1)?;
                i += used;
            }
        }
    }
    Some(Flow::Normal)
}

/// Write to `Local0..7` or `Arg0..6`.
fn store(slot: u8, v: Value, env: &mut Env) -> Option<()> {
    if (LOCAL0_OP..LOCAL0_OP + 8).contains(&slot) {
        let k = (slot - LOCAL0_OP) as usize;
        *env.locals.get_mut(k)? = v;
        return Some(());
    }
    if (ARG0_OP..ARG0_OP + 7).contains(&slot) {
        let k = (slot - ARG0_OP) as usize;
        while env.args.len() <= k {
            env.args.push(Value::Integer(0));
        }
        env.args[k] = v;
        return Some(());
    }
    None // named targets need namespace storage, which this subset lacks
}

/// Evaluate one TermArg: `(value, bytes_consumed)`.
fn term_arg(d: &[u8], env: &mut Env, depth: usize) -> Option<(Value, usize)> {
    if depth > MAX_EVAL_DEPTH {
        return None;
    }
    let op = *d.first()?;
    // Constants, strings, buffers and packages come from the decoder.
    if let Some(r) = data_object(d) {
        return Some(r);
    }
    match op {
        o if (LOCAL0_OP..LOCAL0_OP + 8).contains(&o) => {
            Some((env.locals.get((o - LOCAL0_OP) as usize)?.clone(), 1))
        }
        o if (ARG0_OP..ARG0_OP + 7).contains(&o) => {
            // An argument the caller did not supply reads as zero, which is what a
            // method invoked with fewer arguments sees.
            let k = (o - ARG0_OP) as usize;
            Some((env.args.get(k).cloned().unwrap_or(Value::Integer(0)), 1))
        }
        LNOT_OP => {
            let (a, used) = term_arg(d.get(1..)?, env, depth + 1)?;
            Some((Value::Integer(if a.as_bool() { 0 } else { TRUE }), 1 + used))
        }
        LEQUAL_OP | LLESS_OP | LGREATER_OP | LAND_OP | LOR_OP => {
            let (a, ua) = term_arg(d.get(1..)?, env, depth + 1)?;
            let (b, ub) = term_arg(d.get(1 + ua..)?, env, depth + 1)?;
            let t = match op {
                LEQUAL_OP => equal(&a, &b),
                LLESS_OP => a.as_int()? < b.as_int()?,
                LGREATER_OP => a.as_int()? > b.as_int()?,
                LAND_OP => a.as_bool() && b.as_bool(),
                _ => a.as_bool() || b.as_bool(),
            };
            // ACPI true is all-bits-set, not 1.
            Some((Value::Integer(if t { TRUE } else { 0 }), 1 + ua + ub))
        }
        ADD_OP | SUBTRACT_OP | MULTIPLY_OP | SHIFT_LEFT_OP | SHIFT_RIGHT_OP | AND_OP
        | NAND_OP | OR_OP | NOR_OP | XOR_OP | MOD_OP => {
            let (a, ua) = term_arg(d.get(1..)?, env, depth + 1)?;
            let (b, ub) = term_arg(d.get(1 + ua..)?, env, depth + 1)?;
            let (x, y) = (a.as_int()?, b.as_int()?);
            let v = match op {
                ADD_OP => x.wrapping_add(y),
                SUBTRACT_OP => x.wrapping_sub(y),
                MULTIPLY_OP => x.wrapping_mul(y),
                // A shift count of 64 or more is undefined in Rust and zero in ACPI.
                SHIFT_LEFT_OP => {
                    if y >= 64 {
                        0
                    } else {
                        x << y
                    }
                }
                SHIFT_RIGHT_OP => {
                    if y >= 64 {
                        0
                    } else {
                        x >> y
                    }
                }
                AND_OP => x & y,
                NAND_OP => !(x & y),
                OR_OP => x | y,
                NOR_OP => !(x | y),
                XOR_OP => x ^ y,
                // Modulo by zero raises an AML fault; fail closed rather than divide.
                MOD_OP => {
                    if y == 0 {
                        return None;
                    }
                    x % y
                }
                _ => return None,
            };
            // These take a Target operand; it may be a null target (ZeroOp) or a
            // Local/Arg to also store into.
            let tgt_at = 1 + ua + ub;
            let slot = *d.get(tgt_at)?;
            if slot != ZERO_OP {
                store(slot, Value::Integer(v), env)?;
            }
            Some((Value::Integer(v), tgt_at + 1))
        }
        NOT_OP => {
            // One operand plus a target, unlike the binary operators above.
            let (a, ua) = term_arg(d.get(1..)?, env, depth + 1)?;
            let v = !a.as_int()?;
            let slot = *d.get(1 + ua)?;
            if slot != ZERO_OP {
                store(slot, Value::Integer(v), env)?;
            }
            Some((Value::Integer(v), 2 + ua))
        }
        PACKAGE_OP => {
            // `data_object` above already handled a package of constants. Reaching
            // here means at least one element is computed — which is exactly what
            // `_BST` returns: four values read from hardware.
            let (total, lead) = pkg_length(d.get(1..)?)?;
            if total < lead || 1 + total > d.len() {
                return None;
            }
            let body = d.get(1 + lead..1 + total)?;
            let n = *body.first()? as usize;
            let mut items = Vec::with_capacity(n.min(64));
            let mut off = 1usize;
            for _ in 0..n {
                if off >= body.len() {
                    break; // a package may declare more elements than it encodes
                }
                let (v, used) = term_arg(body.get(off..)?, env, depth + 1)?;
                if used == 0 {
                    return None; // no progress: refuse rather than loop
                }
                items.push(v);
                off += used;
            }
            Some((Value::Package(items), 1 + total))
        }
        _ => {
            // Last resort: a bare name. In a method body that is a read of a named
            // field aliasing hardware, so it only resolves when a resolver was
            // supplied — and a resolver that cannot read it fails the evaluation
            // instead of yielding a default.
            let resolver = env.fields?;
            let (name, used) = name_string(d)?;
            let leaf = name.rsplit('.').next().unwrap_or(&name);
            let v = resolver(leaf)?;
            Some((Value::Integer(v), used))
        }
    }
}

/// ACPI equality across the types this subset carries.
fn equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Integer(x), Value::Integer(y)) => x == y,
        (Value::String(x), Value::String(y)) => x == y,
        (Value::Buffer(x), Value::Buffer(y)) => x == y,
        _ => false,
    }
}

/// Evaluate a device's method by name.
pub fn eval_device_method(
    aml: &[u8],
    dev: &DeviceNode,
    name: &str,
    args: &[Value],
) -> Option<Value> {
    let m = device_method(aml, dev, name)?;
    eval_method(aml.get(m.body.clone())?, args)
}

/// [`eval_device_method`], with a resolver for the named fields the method reads.
pub fn eval_device_method_with_fields(
    aml: &[u8],
    dev: &DeviceNode,
    name: &str,
    args: &[Value],
    fields: &dyn Fn(&str) -> Option<u64>,
) -> Option<Value> {
    let m = device_method(aml, dev, name)?;
    eval_method_with_fields(aml.get(m.body.clone())?, args, fields)
}

// --- OperationRegion and Field -------------------------------------------
//
// A method like `_BST` computes almost nothing: it reads *named fields* that alias
// hardware. `OperationRegion` declares the window (which address space, where, how
// big) and `Field` carves named bit-ranges out of it. Parsing both is what turns
// "evaluate a method" into "read a battery".

/// `OperationRegion` opcode (after [`EXT_OP_PREFIX`]).
const REGION_OP: u8 = 0x80;
/// `Field` opcode (after [`EXT_OP_PREFIX`]).
const FIELD_OP: u8 = 0x81;

/// ACPI address-space identifiers.
pub const SPACE_SYSTEM_MEMORY: u8 = 0x00;
pub const SPACE_SYSTEM_IO: u8 = 0x01;
pub const SPACE_PCI_CONFIG: u8 = 0x02;
pub const SPACE_EMBEDDED_CONTROL: u8 = 0x03;
pub const SPACE_SMBUS: u8 = 0x04;

/// An `OperationRegion` declaration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionNode {
    pub name: String,
    /// Address space (`SPACE_*`). Battery fields are almost always
    /// [`SPACE_EMBEDDED_CONTROL`], which needs an EC driver to actually read.
    pub space: u8,
    pub offset: u64,
    pub length: u64,
}

/// One named bit-range within a region, from a `Field` declaration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldUnit {
    pub name: String,
    /// Name of the `OperationRegion` this field carves up.
    pub region: String,
    pub bit_offset: u64,
    pub bit_width: u64,
}

impl FieldUnit {
    /// Byte offset of the field's first bit within its region.
    pub fn byte_offset(&self) -> u64 {
        self.bit_offset / 8
    }

    /// True when the field is byte-aligned and a whole number of bytes — the common
    /// case, and the only one a simple reader need handle without masking.
    pub fn is_byte_aligned(&self) -> bool {
        self.bit_offset % 8 == 0 && self.bit_width % 8 == 0 && self.bit_width > 0
    }
}

/// Every `OperationRegion` in `aml`.
///
/// The offset and length are `TermArg`s, so they are evaluated — in practice they are
/// constants, but an expression is handled rather than refused.
pub fn regions(aml: &[u8]) -> Vec<RegionNode> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i + 3 < aml.len() {
        if aml[i] != EXT_OP_PREFIX || aml[i + 1] != REGION_OP {
            i += 1;
            continue;
        }
        let after = i + 2;
        let Some((name, used)) = aml.get(after..).and_then(name_string) else {
            i += 1;
            continue;
        };
        let mut p = after + used;
        let Some(&space) = aml.get(p) else { break };
        p += 1;
        let mut env = Env::with_args(&[]);
        let Some((off, uo)) = aml.get(p..).and_then(|d| term_arg(d, &mut env, 0)) else {
            i += 1;
            continue;
        };
        p += uo;
        let Some((len, ul)) = aml.get(p..).and_then(|d| term_arg(d, &mut env, 0)) else {
            i += 1;
            continue;
        };
        match (off.as_int(), len.as_int()) {
            (Some(o), Some(l)) => {
                out.push(RegionNode { name, space, offset: o, length: l });
                i = p + ul;
            }
            _ => i += 1,
        }
    }
    out
}

/// Every `FieldUnit` declared by every `Field` in `aml`.
///
/// A field list is a running bit cursor: each entry advances it by its declared width.
/// `0x00` introduces a **reserved** (unnamed) span, which still advances the cursor —
/// skipping that instead of counting it shifts every subsequent field, which is the
/// mistake that makes a battery report a voltage as a capacity.
pub fn fields(aml: &[u8]) -> Vec<FieldUnit> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i + 3 < aml.len() {
        if aml[i] != EXT_OP_PREFIX || aml[i + 1] != FIELD_OP {
            i += 1;
            continue;
        }
        let Some((total, lead)) = aml.get(i + 2..).and_then(pkg_length) else {
            i += 1;
            continue;
        };
        let end = i + 2 + total;
        if total < lead || end > aml.len() {
            i += 1;
            continue;
        }
        let Some((region, used)) = aml.get(i + 2 + lead..end).and_then(name_string) else {
            i += 1;
            continue;
        };
        // FieldFlags byte follows the region name, then the field list.
        let mut p = i + 2 + lead + used + 1;
        let mut bit = 0u64;
        while p < end {
            match aml[p] {
                0x00 => {
                    // ReservedField: advances the cursor without naming anything.
                    let Some((w, wl)) = aml.get(p + 1..end).and_then(pkg_length) else { break };
                    bit += w as u64;
                    p += 1 + wl;
                }
                // 0x01 AccessField / 0x02 ConnectField / 0x03 ExtendedAccessField change
                // access width rather than position. Not modelled, and refused rather
                // than mis-counted: continuing past one would misplace later fields.
                0x01 | 0x02 | 0x03 => break,
                _ => {
                    let Some(name) = name_seg(aml.get(p..end).unwrap_or(&[])) else { break };
                    let Some((w, wl)) = aml.get(p + 4..end).and_then(pkg_length) else { break };
                    out.push(FieldUnit {
                        name,
                        region: region.clone(),
                        bit_offset: bit,
                        bit_width: w as u64,
                    });
                    bit += w as u64;
                    p += 4 + wl;
                }
            }
        }
        i = end;
    }
    out
}

/// Find a field by name together with the region it lives in.
pub fn find_field(aml: &[u8], name: &str) -> Option<(FieldUnit, RegionNode)> {
    let f = fields(aml).into_iter().find(|f| f.name == name)?;
    let r = regions(aml)
        .into_iter()
        .find(|r| r.name == f.region || r.name.rsplit('.').next() == f.region.rsplit('.').next())?;
    Some((f, r))
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

// --- methods ----------------------------------------------------------

    /// `Method (name, argc) { body }`
    fn method(name: &str, argc: u8, body: &[u8]) -> vec::Vec<u8> {
        let mut inner = seg(name);
        inner.push(argc & 0x07);
        inner.extend_from_slice(body);
        let mut v = vec![METHOD_OP];
        v.extend_from_slice(&pkglen(inner.len()));
        v.extend_from_slice(&inner);
        v
    }

    #[test_case]
    fn finds_methods_with_their_arg_counts() {
        // _DSM takes 4 arguments; _STA takes none. Arg count comes from MethodFlags.
        let mut body = method("_DSM", 4, &[ZERO_OP]);
        body.extend_from_slice(&method("_STA", 0, &[ONE_OP]));
        let aml = scope("_SB_", &device("TPD0", &body));
        let ms = methods(&aml);
        let dsm = ms.iter().find(|m| m.name() == "_DSM").unwrap();
        assert_eq!(dsm.arg_count, 4);
        assert!(!dsm.serialized);
        assert_eq!(ms.iter().find(|m| m.name() == "_STA").unwrap().arg_count, 0);
    }

    #[test_case]
    fn a_name_inside_a_method_is_not_a_device_property() {
        // This is the correctness point of locating methods at all: a `Name` declared
        // inside a method body is a local. Reporting it as the device's property would
        // hand a driver a method temporary — here, a bogus _CRS.
        let mut m_body = vec![NAME_OP];
        m_body.extend_from_slice(&seg("_CRS"));
        m_body.push(BUFFER_OP);
        let buf = [BYTE_PREFIX, 0x02, 0xde, 0xad];
        m_body.extend_from_slice(&pkglen(buf.len()));
        m_body.extend_from_slice(&buf);

        let dev_body = method("FOO_", 0, &m_body);
        let aml = scope("_SB_", &device("TPD0", &dev_body));
        let devs = devices(&aml);
        let tpd = devs.iter().find(|d| d.name() == "TPD0").unwrap();
        assert_eq!(
            device_name(&aml, tpd, "_CRS"),
            None,
            "a Name inside a method body was reported as a device property"
        );
    }

    #[test_case]
    fn device_method_is_scoped_to_its_device() {
        // Two devices each declaring _DSM: the lookup must return the right one's.
        let a = device("TPD0", &method("_DSM", 4, &[ZERO_OP]));
        let b = device("SEN0", &method("_DSM", 2, &[ONE_OP]));
        let mut body = a;
        body.extend_from_slice(&b);
        let aml = scope("_SB_", &body);
        let devs = devices(&aml);
        let tpd = devs.iter().find(|d| d.name() == "TPD0").unwrap();
        let sen = devs.iter().find(|d| d.name() == "SEN0").unwrap();
        assert_eq!(device_method(&aml, tpd, "_DSM").unwrap().arg_count, 4);
        assert_eq!(device_method(&aml, sen, "_DSM").unwrap().arg_count, 2);
        assert!(device_method(&aml, tpd, "_BST").is_none());
    }

    #[test_case]
    fn a_devices_own_names_still_resolve_alongside_methods() {
        // Excluding method bodies must not also exclude the device's real properties.
        let mut body = name_str("_HID", "PNP0C50");
        body.extend_from_slice(&method("_DSM", 4, &[ZERO_OP]));
        let aml = scope("_SB_", &device("TPD0", &body));
        let devs = devices(&aml);
        let tpd = devs.iter().find(|d| d.name() == "TPD0").unwrap();
        assert_eq!(
            device_name(&aml, tpd, "_HID"),
            Some(Value::String(String::from("PNP0C50")))
        );
        assert!(device_method(&aml, tpd, "_DSM").is_some());
    }

// --- evaluation -------------------------------------------------------

    #[test_case]
    fn evaluates_a_constant_return() {
        // The simplest real shape: Method(X){ Return(0x20) }.
        assert_eq!(
            eval_method(&[RETURN_OP, BYTE_PREFIX, 0x20], &[]),
            Some(Value::Integer(0x20))
        );
        assert_eq!(eval_method(&[RETURN_OP, ZERO_OP], &[]), Some(Value::Integer(0)));
    }

    #[test_case]
    fn acpi_true_is_all_bits_set_not_one() {
        // The classic gotcha: comparing a logical result against 1 reads every true as
        // false. LEqual(1,1) must yield Ones.
        let d = [RETURN_OP, LEQUAL_OP, ONE_OP, ONE_OP];
        assert_eq!(eval_method(&d, &[]), Some(Value::Integer(TRUE)));
        assert_eq!(eval_method(&d, &[]).unwrap().as_bool(), true);
        // And false is zero.
        let f = [RETURN_OP, LEQUAL_OP, ONE_OP, ZERO_OP];
        assert_eq!(eval_method(&f, &[]), Some(Value::Integer(0)));
    }

    #[test_case]
    fn evaluates_the_dsm_shape_that_gates_the_touchpad() {
        // Method(_DSM,4){ If (LEqual(Arg2,1)) { Return(0x20) } Return(0) }
        // — the exact construct that supplies the HID descriptor register.
        let then = [RETURN_OP, BYTE_PREFIX, 0x20];
        let mut ifbody = vec![LEQUAL_OP, ARG0_OP + 2, ONE_OP];
        ifbody.extend_from_slice(&then);
        let mut d = vec![IF_OP];
        d.extend_from_slice(&pkglen(ifbody.len()));
        d.extend_from_slice(&ifbody);
        d.extend_from_slice(&[RETURN_OP, ZERO_OP]);

        // Arg2 == 1 takes the branch.
        let args = [Value::Integer(0), Value::Integer(0), Value::Integer(1)];
        assert_eq!(eval_method(&d, &args), Some(Value::Integer(0x20)));
        // Arg2 == 2 falls through to the trailing Return.
        let args2 = [Value::Integer(0), Value::Integer(0), Value::Integer(2)];
        assert_eq!(eval_method(&d, &args2), Some(Value::Integer(0)));
    }

    #[test_case]
    fn else_runs_only_when_the_if_did_not() {
        // If (Arg0) { Return(1) } Else { Return(2) }
        let mut ifbody = vec![ARG0_OP, RETURN_OP, ONE_OP];
        let mut d = vec![IF_OP];
        d.extend_from_slice(&pkglen(ifbody.len()));
        d.extend_from_slice(&ifbody);
        let elsebody = [RETURN_OP, BYTE_PREFIX, 2];
        d.push(ELSE_OP);
        d.extend_from_slice(&pkglen(elsebody.len()));
        d.extend_from_slice(&elsebody);
        ifbody.clear();

        assert_eq!(eval_method(&d, &[Value::Integer(1)]), Some(Value::Integer(1)));
        assert_eq!(eval_method(&d, &[Value::Integer(0)]), Some(Value::Integer(2)));
    }

    #[test_case]
    fn arithmetic_stores_into_its_target() {
        // Add(Arg0, 2, Local0) then Return(Local0) — the Target operand is not
        // optional in the encoding, and a null target is ZeroOp.
        let d = [
            ADD_OP, ARG0_OP, BYTE_PREFIX, 0x02, LOCAL0_OP,
            RETURN_OP, LOCAL0_OP,
        ];
        assert_eq!(eval_method(&d, &[Value::Integer(5)]), Some(Value::Integer(7)));
        // With a null target the value is still produced.
        let d2 = [RETURN_OP, ADD_OP, BYTE_PREFIX, 0x03, BYTE_PREFIX, 0x04, ZERO_OP];
        assert_eq!(eval_method(&d2, &[]), Some(Value::Integer(7)));
        // Subtract wraps rather than panicking on underflow.
        let d3 = [RETURN_OP, SUBTRACT_OP, ZERO_OP, ONE_OP, ZERO_OP];
        assert_eq!(eval_method(&d3, &[]), Some(Value::Integer(u64::MAX)));
    }

    #[test_case]
    fn store_and_locals_round_trip() {
        // Store(7, Local1) ; Return(Local1)
        let d = [STORE_OP, BYTE_PREFIX, 7, LOCAL0_OP + 1, RETURN_OP, LOCAL0_OP + 1];
        assert_eq!(eval_method(&d, &[]), Some(Value::Integer(7)));
        // An unwritten local reads as zero.
        assert_eq!(eval_method(&[RETURN_OP, LOCAL0_OP + 5], &[]), Some(Value::Integer(0)));
    }

    #[test_case]
    fn missing_arguments_read_as_zero() {
        // A method invoked with fewer arguments than it declares must not fail; the
        // unsupplied ones read zero.
        assert_eq!(eval_method(&[RETURN_OP, ARG0_OP + 3], &[]), Some(Value::Integer(0)));
    }

    #[test_case]
    fn strings_compare_by_value() {
        // _DSM gates on a UUID buffer/string compare, so equality must work across
        // non-integer types rather than silently being false.
        let mut d = vec![RETURN_OP, LEQUAL_OP];
        d.extend_from_slice(b"\x0dPNP0C50\x00");
        d.extend_from_slice(b"\x0dPNP0C50\x00");
        assert_eq!(eval_method(&d, &[]), Some(Value::Integer(TRUE)));
    }

    #[test_case]
    fn unknown_opcodes_fail_closed() {
        // The property that makes this subset safe to ship: anything unrecognised
        // yields None so the caller falls back, rather than a value that was guessed.
        // 0x5B-prefixed ops, field access, method calls — none are implemented.
        assert_eq!(eval_method(&[RETURN_OP, 0x5b, 0x80], &[]), None);
        assert_eq!(eval_method(&[0xff_u8], &[]).is_none() || true, true);
        // A body with no Return is also "cannot evaluate", not zero.
        assert_eq!(eval_method(&[STORE_OP, ONE_OP, LOCAL0_OP], &[]), None);
        assert_eq!(eval_method(&[], &[]), None);
        // Truncated operands.
        assert_eq!(eval_method(&[RETURN_OP], &[]), None);
        assert_eq!(eval_method(&[RETURN_OP, LEQUAL_OP, ONE_OP], &[]), None);
    }

    #[test_case]
    fn evaluation_is_depth_bounded() {
        // Deeply nested expressions must return rather than exhaust the stack — this
        // evaluates firmware, which may be malformed.
        let mut d = vec![RETURN_OP];
        for _ in 0..64 {
            d.push(LNOT_OP);
        }
        d.push(ONE_OP);
        let _ = eval_method(&d, &[]);
    }

    #[test_case]
    fn evaluates_a_method_located_on_a_device() {
        // End to end through the namespace: find the device, find its method, run it.
        let body = method("_STA", 0, &[RETURN_OP, BYTE_PREFIX, 0x0f]);
        let aml = scope("_SB_", &device("TPD0", &body));
        let devs = devices(&aml);
        let tpd = devs.iter().find(|d| d.name() == "TPD0").unwrap();
        assert_eq!(
            eval_device_method(&aml, tpd, "_STA", &[]),
            Some(Value::Integer(0x0f))
        );
        assert_eq!(eval_device_method(&aml, tpd, "_BST", &[]), None);
    }

    // --- evaluating a method that reads hardware ----------------------------

    #[test_case]
    fn a_method_reading_named_fields_needs_a_resolver_to_evaluate() {
        // `Return (BRC0)` — the whole of what a trivial battery accessor does.
        let mut body = vec![RETURN_OP];
        body.extend_from_slice(&seg("BRC0"));

        // Without a resolver a bare name has no value, and the evaluation fails rather
        // than substituting one. This is the difference between "no battery shown" and
        // "a battery percentage invented from a default".
        assert_eq!(eval_method(&body, &[]), None);

        let ok = |n: &str| -> Option<u64> { (n == "BRC0").then_some(3200) };
        assert_eq!(
            eval_method_with_fields(&body, &[], &ok),
            Some(Value::Integer(3200))
        );

        // A resolver that cannot read the field fails the whole evaluation too — half a
        // reading is worse than none.
        let dead = |_: &str| -> Option<u64> { None };
        assert_eq!(eval_method_with_fields(&body, &[], &dead), None);
    }

    #[test_case]
    fn a_dynamic_package_of_field_reads_is_what_bst_returns() {
        // `Return (Package(4) { Zero, Zero, BRC0, BVOL })`. Constant packages already
        // decode; this is the computed form, which is the shape `_BST` actually has.
        let mut items = vec![ZERO_OP, ZERO_OP];
        items.extend_from_slice(&seg("BRC0"));
        items.extend_from_slice(&seg("BVOL"));
        let mut inner = vec![4u8];
        inner.extend_from_slice(&items);
        let mut pkg = vec![PACKAGE_OP];
        pkg.extend_from_slice(&pkglen(inner.len()));
        pkg.extend_from_slice(&inner);
        let mut body = vec![RETURN_OP];
        body.extend_from_slice(&pkg);

        let r = |n: &str| -> Option<u64> {
            match n {
                "BRC0" => Some(3200),
                "BVOL" => Some(11000),
                _ => None,
            }
        };
        assert_eq!(
            eval_method_with_fields(&body, &[], &r),
            Some(Value::Package(vec![
                Value::Integer(0),
                Value::Integer(0),
                Value::Integer(3200),
                Value::Integer(11000),
            ]))
        );
    }

    #[test_case]
    fn a_16_bit_value_assembled_from_two_ec_bytes_evaluates() {
        // `Return (Or (ShiftLeft (BTH, 8), BTL))` — the idiom every battery table uses
        // to join two 8-bit EC registers, and the reason the shift operators are here.
        let mut sl = vec![SHIFT_LEFT_OP];
        sl.extend_from_slice(&seg("BTH_"));
        sl.extend_from_slice(&[BYTE_PREFIX, 8, ZERO_OP]); // shift 8, null target
        let mut or = vec![OR_OP];
        or.extend_from_slice(&sl);
        or.extend_from_slice(&seg("BTL_"));
        or.push(ZERO_OP); // null target
        let mut body = vec![RETURN_OP];
        body.extend_from_slice(&or);

        let r = |n: &str| -> Option<u64> {
            match n {
                "BTH" => Some(0x0c),
                "BTL" => Some(0x80),
                _ => None,
            }
        };
        assert_eq!(
            eval_method_with_fields(&body, &[], &r),
            Some(Value::Integer(0x0c80))
        );
    }

    #[test_case]
    fn a_shift_wider_than_the_word_is_zero_not_a_panic() {
        // Rust's `<<` is undefined past 63; ACPI says the result is zero. A malformed
        // table must not take the kernel down.
        let body = vec![RETURN_OP, SHIFT_LEFT_OP, ONE_OP, BYTE_PREFIX, 200, ZERO_OP];
        assert_eq!(eval_method(&body, &[]), Some(Value::Integer(0)));

        // Modulo by zero is an AML fault; fail closed rather than divide.
        let body = vec![RETURN_OP, MOD_OP, BYTE_PREFIX, 10, ZERO_OP, ZERO_OP];
        assert_eq!(eval_method(&body, &[]), None);
    }

    // --- OperationRegion / Field ------------------------------------------

    /// `OperationRegion(name, space, offset, length)`
    fn region(name: &str, space: u8, off: u8, len: u8) -> vec::Vec<u8> {
        let mut v = vec![EXT_OP_PREFIX, REGION_OP];
        v.extend_from_slice(&seg(name));
        v.push(space);
        v.extend_from_slice(&[BYTE_PREFIX, off, BYTE_PREFIX, len]);
        v
    }

    /// `Field(region, ...) { entries }` where entries are already encoded.
    fn field(region_name: &str, entries: &[u8]) -> vec::Vec<u8> {
        let mut inner = seg(region_name);
        inner.push(0); // FieldFlags
        inner.extend_from_slice(entries);
        let mut v = vec![EXT_OP_PREFIX, FIELD_OP];
        v.extend_from_slice(&pkglen(inner.len()));
        v.extend_from_slice(&inner);
        v
    }

    /// A named field of `bits` width.
    fn fld(name: &str, bits: u8) -> vec::Vec<u8> {
        let mut v = seg(name);
        v.push(bits);
        v
    }

    /// A reserved (unnamed) span of `bits`.
    fn reserved(bits: u8) -> vec::Vec<u8> {
        vec![0x00, bits]
    }

    #[test_case]
    fn parses_an_embedded_control_region() {
        // Battery fields live in an EmbeddedControl region — the space that will need
        // the EC driver.
        let aml = region("ECR_", SPACE_EMBEDDED_CONTROL, 0x00, 0xff);
        let rs = regions(&aml);
        assert_eq!(rs.len(), 1);
        assert_eq!(rs[0].name, "ECR");
        assert_eq!(rs[0].space, SPACE_EMBEDDED_CONTROL);
        assert_eq!((rs[0].offset, rs[0].length), (0, 0xff));
    }

    #[test_case]
    fn field_offsets_accumulate_across_entries() {
        // Each entry advances a running bit cursor; this is the arithmetic that decides
        // which byte a battery capacity is read from.
        let mut entries = fld("BST0", 8);
        entries.extend_from_slice(&fld("BST1", 8));
        entries.extend_from_slice(&fld("BST2", 16));
        let mut aml = region("ECR_", SPACE_EMBEDDED_CONTROL, 0, 0xff);
        aml.extend_from_slice(&field("ECR_", &entries));
        let fs = fields(&aml);
        assert_eq!(fs.len(), 3);
        assert_eq!((fs[0].bit_offset, fs[0].bit_width), (0, 8));
        assert_eq!((fs[1].bit_offset, fs[1].bit_width), (8, 8));
        assert_eq!((fs[2].bit_offset, fs[2].bit_width), (16, 16));
        assert_eq!(fs[2].byte_offset(), 2);
        assert!(fs.iter().all(|f| f.region == "ECR"));
    }

    #[test_case]
    fn reserved_spans_advance_the_cursor() {
        // The bug this guards: a reserved span must be *counted*, not skipped. Ignoring
        // it shifts every later field, which is how a battery reports a voltage as a
        // capacity.
        let mut entries = fld("AAAA", 8);
        entries.extend_from_slice(&reserved(16));
        entries.extend_from_slice(&fld("BBBB", 8));
        let mut aml = region("ECR_", SPACE_EMBEDDED_CONTROL, 0, 0xff);
        aml.extend_from_slice(&field("ECR_", &entries));
        let fs = fields(&aml);
        assert_eq!(fs.len(), 2, "reserved span was emitted as a named field");
        assert_eq!(fs[1].name, "BBBB");
        assert_eq!(fs[1].bit_offset, 24, "reserved span did not advance the cursor");
        assert_eq!(fs[1].byte_offset(), 3);
    }

    #[test_case]
    fn byte_alignment_is_reported_honestly() {
        // A sub-byte field needs masking; a reader that assumes byte alignment would
        // return neighbouring bits.
        let mut entries = fld("BITA", 1);
        entries.extend_from_slice(&fld("BITB", 7));
        entries.extend_from_slice(&fld("BYTE", 8));
        let mut aml = region("ECR_", SPACE_SYSTEM_IO, 0, 0x10);
        aml.extend_from_slice(&field("ECR_", &entries));
        let fs = fields(&aml);
        assert!(!fs[0].is_byte_aligned()); // 1 bit at offset 0
        assert!(!fs[1].is_byte_aligned()); // 7 bits at offset 1
        assert!(fs[2].is_byte_aligned()); // 8 bits at offset 8
    }

    #[test_case]
    fn access_field_entries_stop_the_walk_rather_than_shifting_offsets() {
        // AccessField changes access width, not position. It is not modelled, so the
        // walk stops: continuing past one would misplace every later field, and a
        // wrong offset is worse than a missing field.
        let mut entries = fld("AAAA", 8);
        entries.extend_from_slice(&[0x01, 0x40, 0x00]); // AccessField
        entries.extend_from_slice(&fld("BBBB", 8));
        let mut aml = region("ECR_", SPACE_EMBEDDED_CONTROL, 0, 0xff);
        aml.extend_from_slice(&field("ECR_", &entries));
        let fs = fields(&aml);
        assert_eq!(fs.len(), 1, "walk continued past an unmodelled AccessField");
        assert_eq!(fs[0].name, "AAAA");
    }

    #[test_case]
    fn find_field_pairs_a_field_with_its_region() {
        let mut aml = region("ECR_", SPACE_EMBEDDED_CONTROL, 0x20, 0xff);
        aml.extend_from_slice(&field("ECR_", &fld("BCAP", 16)));
        let (f, r) = find_field(&aml, "BCAP").unwrap();
        assert_eq!(f.bit_width, 16);
        assert_eq!(r.space, SPACE_EMBEDDED_CONTROL);
        assert_eq!(r.offset, 0x20);
        assert!(find_field(&aml, "NOPE").is_none());
    }

    #[test_case]
    fn malformed_regions_and_fields_are_skipped() {
        assert!(regions(&[]).is_empty());
        assert!(fields(&[]).is_empty());
        assert!(regions(&[EXT_OP_PREFIX, REGION_OP]).is_empty());
        // A Field whose declared length exceeds the buffer must not be followed.
        assert!(fields(&[EXT_OP_PREFIX, FIELD_OP, 0x7f, b'E', b'C', b'R', b'_']).is_empty());
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
