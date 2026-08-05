//! **JS → wasm module emission** — wrapping QuickJS bytecode in a real
//! `tools.wasm`, plus the byte-level questions that go with it.
//!
//! # Why this exists
//!
//! A Javy plugin's `compile-src` returns *QuickJS bytecode*, not a WebAssembly
//! module. The Javy CLI is what normally wraps that bytecode into a module; doing
//! it here is what lets a developer compile JavaScript **on the machine** and
//! still ship the one artifact every agent package ships — `assets/tools.wasm`.
//! Nothing downstream changes: the manifest still names a wasm module, the tool
//! registry still holds ordinary wasm tools, and there is no second kind of agent.
//!
//! # The module this emits
//!
//! Decoded from `javy build -C dynamic` output, which is a fixed template: three
//! types, three imports from the plugin's namespace, one function and one export
//! per tool, and two passive data segments (the bytecode, and the tool names
//! concatenated). Each exported function is
//!
//! ```wat
//! (func (export "notes_list")            ;; type () -> ()
//!   (local i32 i32)
//!   ;; bc = cabi_realloc(0, 0, 1, BC_LEN); copy segment 0 into it
//!   i32.const 0  i32.const 0  i32.const 1  i32.const BC_LEN  call $cabi_realloc
//!   local.set 0
//!   local.get 0  i32.const 0  i32.const BC_LEN  memory.init 0
//!   ;; nm = cabi_realloc(0, 0, 1, NAME_LEN); copy this tool's slice of segment 1
//!   i32.const 0  i32.const 0  i32.const 1  i32.const NAME_LEN  call $cabi_realloc
//!   local.set 1
//!   local.get 1  i32.const NAME_OFF  i32.const NAME_LEN  memory.init 1
//!   ;; invoke(bytecode, len, some(name))  -- option<string> flattens to (1, ptr, len)
//!   local.get 0  i32.const BC_LEN  i32.const 1  local.get 1  i32.const NAME_LEN
//!   call $invoke)
//! ```
//!
//! So the emitted module **exports the same function names a hand-written Rust
//! module would**, which is the whole reason the rest of the system is untouched.
//! `memory.init` takes a source offset, so one names segment serves every tool.
//!
//! # What was verified, and how
//!
//! The template was decoded from a real `javy build -C dynamic` artifact, and this
//! emitter's output was executed against the real plugin before this file existed:
//! a two-export module built from 907 bytes of bytecode ran both tools and
//! returned correct JSON. That matters because **a wrong LEB128 length still
//! validates** — the failure mode is a plausible module that misbehaves, so the
//! tests below check by *round-tripping and validating*, not by comparing bytes.
//!
//! Two upstream facts pinned by experiment rather than assumption:
//!
//! * `invoke`'s name argument resolves an ESM export **verbatim** — snake_case
//!   works. The Javy CLI maps kebab-case WIT names to camelCase JS, but that is
//!   the CLI's convention, not the plugin's, and this emitter bypasses it. So a
//!   tool named `notes_list` calls `export function notes_list()`.
//! * QuickJS bytecode is **version-tied** to the plugin that produced it: feeding
//!   a module built elsewhere to a different plugin fails inside QuickJS with
//!   `invalid version`. [`emit`] therefore stamps the plugin identity into a
//!   custom section so [`plugin_stamp`] can catch a stale artifact and ask for a
//!   rebuild, instead of letting it fail at call time.

use alloc::string::String;
use alloc::vec::Vec;

/// Name of the custom section this emitter writes, holding `plugin=<identity>`.
pub const STAMP_SECTION: &str = "chitti_js";
/// Name of the custom section a Javy plugin uses for its import namespace.
const NAMESPACE_SECTION: &str = "import_namespace";

/// Bound on tools per module — far above any real package, low enough that a
/// corrupt manifest cannot ask for a gigabyte of function bodies.
pub const MAX_EXPORTS: usize = 64;

// --- LEB128 -----------------------------------------------------------------

fn uleb(n: u32, out: &mut Vec<u8>) {
    let mut v = n;
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

/// Signed LEB128 — what `i32.const` takes. Every constant this emitter writes is
/// non-negative, but the encoding still differs from unsigned: a value whose top
/// retained bit is set needs a zero continuation byte or it reads as negative.
fn sleb(n: i32, out: &mut Vec<u8>) {
    let mut v = n;
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        let done = (v == 0 && b & 0x40 == 0) || (v == -1 && b & 0x40 != 0);
        out.push(if done { b } else { b | 0x80 });
        if done {
            return;
        }
    }
}

fn vec_len(count: usize, body: &[u8], out: &mut Vec<u8>) {
    uleb(count as u32, out);
    out.extend_from_slice(body);
}

fn section(id: u8, payload: &[u8], out: &mut Vec<u8>) {
    out.push(id);
    uleb(payload.len() as u32, out);
    out.extend_from_slice(payload);
}

fn wasm_name(s: &str, out: &mut Vec<u8>) {
    uleb(s.len() as u32, out);
    out.extend_from_slice(s.as_bytes());
}

const I32: u8 = 0x7f;

fn i32_const(n: i32, out: &mut Vec<u8>) {
    out.push(0x41);
    sleb(n, out);
}

// --- emission ---------------------------------------------------------------

/// Why a set of tool names cannot be emitted.
#[derive(Debug, PartialEq, Eq)]
pub enum EmitError {
    NoExports,
    TooManyExports,
    /// A name that is empty, or not a plain identifier. Refused rather than
    /// escaped: a wasm export name may be arbitrary UTF-8, but a JS *export*
    /// must be an identifier, so anything else could never resolve at `invoke`.
    BadName,
    EmptyBytecode,
}

/// True for a name that can be both a wasm export and a JS identifier.
fn valid_export_name(n: &str) -> bool {
    !n.is_empty()
        && n.len() <= 64
        && !n.as_bytes()[0].is_ascii_digit()
        && n.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'$')
}

/// Build a `tools.wasm` around `bytecode`, exporting one function per name.
///
/// `namespace` must be the plugin's import namespace ([`import_namespace`]);
/// `stamp` identifies the plugin the bytecode came from and is recorded in a
/// custom section (see the module docs on version-tying).
pub fn emit(
    namespace: &str,
    stamp: &str,
    bytecode: &[u8],
    exports: &[&str],
) -> Result<Vec<u8>, EmitError> {
    if exports.is_empty() {
        return Err(EmitError::NoExports);
    }
    if exports.len() > MAX_EXPORTS {
        return Err(EmitError::TooManyExports);
    }
    if bytecode.is_empty() {
        return Err(EmitError::EmptyBytecode);
    }
    if !exports.iter().all(|n| valid_export_name(n)) {
        return Err(EmitError::BadName);
    }

    let bc_len = bytecode.len() as i32;

    // Section 1: three function types — the export shape, cabi_realloc, invoke.
    let mut types = Vec::new();
    let mut t = Vec::new();
    // () -> ()
    t.push(0x60);
    uleb(0, &mut t);
    uleb(0, &mut t);
    // (i32,i32,i32,i32) -> (i32)
    t.push(0x60);
    uleb(4, &mut t);
    t.extend_from_slice(&[I32; 4]);
    uleb(1, &mut t);
    t.push(I32);
    // (i32 x5) -> ()
    t.push(0x60);
    uleb(5, &mut t);
    t.extend_from_slice(&[I32; 5]);
    uleb(0, &mut t);
    vec_len(3, &t, &mut types);

    // Section 2: the plugin's three exports, imported.
    let mut imports = Vec::new();
    let mut i = Vec::new();
    wasm_name(namespace, &mut i);
    wasm_name("cabi_realloc", &mut i);
    i.push(0x00);
    uleb(1, &mut i);
    wasm_name(namespace, &mut i);
    wasm_name("invoke", &mut i);
    i.push(0x00);
    uleb(2, &mut i);
    wasm_name(namespace, &mut i);
    wasm_name("memory", &mut i);
    i.push(0x02);
    i.push(0x00); // limits: min only
    uleb(0, &mut i);
    vec_len(3, &i, &mut imports);

    // Section 3: one local function per export, all of type 0.
    let mut funcs = Vec::new();
    let mut f = Vec::new();
    for _ in exports {
        uleb(0, &mut f);
    }
    vec_len(exports.len(), &f, &mut funcs);

    // Section 7: exports. Imported functions occupy indices 0 and 1, so local
    // function `k` is index `k + 2`.
    let mut exps = Vec::new();
    let mut e = Vec::new();
    for (k, n) in exports.iter().enumerate() {
        wasm_name(n, &mut e);
        e.push(0x00);
        uleb((k + 2) as u32, &mut e);
    }
    vec_len(exports.len(), &e, &mut exps);

    // Section 10: bodies. Offsets into the names segment are assigned in order.
    let mut code = Vec::new();
    let mut bodies = Vec::new();
    let mut name_off = 0i32;
    for n in exports {
        let name_len = n.len() as i32;
        let mut b = Vec::new();
        // one local group of two i32 (bytecode pointer, name pointer)
        uleb(1, &mut b);
        uleb(2, &mut b);
        b.push(I32);
        // bc = cabi_realloc(0, 0, 1, bc_len)
        i32_const(0, &mut b);
        i32_const(0, &mut b);
        i32_const(1, &mut b);
        i32_const(bc_len, &mut b);
        b.extend_from_slice(&[0x10, 0x00]); // call 0
        b.extend_from_slice(&[0x21, 0x00]); // local.set 0
        // memory.init 0: dest = bc, src = 0, len = bc_len
        b.extend_from_slice(&[0x20, 0x00]); // local.get 0
        i32_const(0, &mut b);
        i32_const(bc_len, &mut b);
        b.extend_from_slice(&[0xfc, 0x08, 0x00, 0x00]);
        // nm = cabi_realloc(0, 0, 1, name_len)
        i32_const(0, &mut b);
        i32_const(0, &mut b);
        i32_const(1, &mut b);
        i32_const(name_len, &mut b);
        b.extend_from_slice(&[0x10, 0x00]);
        b.extend_from_slice(&[0x21, 0x01]); // local.set 1
        // memory.init 1: dest = nm, src = name_off, len = name_len
        b.extend_from_slice(&[0x20, 0x01]); // local.get 1
        i32_const(name_off, &mut b);
        i32_const(name_len, &mut b);
        b.extend_from_slice(&[0xfc, 0x08, 0x01, 0x00]);
        // invoke(bc, bc_len, 1, nm, name_len)
        b.extend_from_slice(&[0x20, 0x00]);
        i32_const(bc_len, &mut b);
        i32_const(1, &mut b);
        b.extend_from_slice(&[0x20, 0x01]);
        i32_const(name_len, &mut b);
        b.extend_from_slice(&[0x10, 0x01]); // call 1
        b.push(0x0b); // end

        uleb(b.len() as u32, &mut bodies);
        bodies.extend_from_slice(&b);
        name_off += name_len;
    }
    vec_len(exports.len(), &bodies, &mut code);

    // Section 11: two passive segments — bytecode, then the names.
    let names_blob: String = exports.concat();
    let mut datas = Vec::new();
    let mut d = Vec::new();
    for payload in [bytecode, names_blob.as_bytes()] {
        uleb(1, &mut d); // passive
        uleb(payload.len() as u32, &mut d);
        d.extend_from_slice(payload);
    }
    vec_len(2, &d, &mut datas);

    let mut out = Vec::new();
    out.extend_from_slice(b"\0asm\x01\0\0\0");
    section(1, &types, &mut out);
    section(2, &imports, &mut out);
    section(3, &funcs, &mut out);
    section(7, &exps, &mut out);
    // The data-count section must precede code, because `memory.init` refers to
    // segments the validator has to know about before it validates a body.
    let mut dc = Vec::new();
    uleb(2, &mut dc);
    section(12, &dc, &mut out);
    section(10, &code, &mut out);
    section(11, &datas, &mut out);
    // Our stamp, last: a custom section is ignorable by any consumer.
    let mut stamp_payload = Vec::new();
    wasm_name(STAMP_SECTION, &mut stamp_payload);
    stamp_payload.extend_from_slice(b"plugin=");
    stamp_payload.extend_from_slice(stamp.as_bytes());
    section(0, &stamp_payload, &mut out);
    Ok(out)
}

/// Tool names to export, scanned from `export function <name>` in a script.
///
/// A convenience so `/js build tools.js` needs no flags in the common case; an
/// explicit list always overrides it. Comments are stripped first, because
/// `// export function old_thing` would otherwise create a wasm export with no JS
/// function behind it — which fails only when that tool is called, far from the
/// cause. String literals are *not* parsed, so `export function` inside a string
/// is a known (and unlikely) false positive.
pub fn scan_exports(src: &str) -> Vec<String> {
    let stripped = strip_comments(src);
    let mut out: Vec<String> = Vec::new();
    let b = stripped.as_bytes();
    let mut i = 0usize;
    while let Some(at) = find_from(&stripped, "export", i) {
        i = at + 6;
        // `export` must be its own word.
        if at > 0 && (b[at - 1].is_ascii_alphanumeric() || b[at - 1] == b'_') {
            continue;
        }
        let mut j = skip_ws(b, i);
        // Optional `async`.
        if stripped[j..].starts_with("async") {
            j = skip_ws(b, j + 5);
        }
        if !stripped[j..].starts_with("function") {
            continue;
        }
        j = skip_ws(b, j + 8);
        // Optional generator star — Javy does not support generator exports, so
        // skipping the name here means we never claim a tool it cannot call.
        if b.get(j) == Some(&b'*') {
            continue;
        }
        let start = j;
        while j < b.len() && (b[j].is_ascii_alphanumeric() || b[j] == b'_' || b[j] == b'$') {
            j += 1;
        }
        if j > start {
            let name = &stripped[start..j];
            if valid_export_name(name) && !out.iter().any(|n| n == name) {
                out.push(String::from(name));
            }
        }
    }
    out
}

fn find_from(s: &str, needle: &str, from: usize) -> Option<usize> {
    if from >= s.len() {
        return None;
    }
    s.get(from..)?.find(needle).map(|k| k + from)
}

fn skip_ws(b: &[u8], mut i: usize) -> usize {
    while i < b.len() && (b[i] as char).is_ascii_whitespace() {
        i += 1;
    }
    i
}

/// Replace `//…` and `/*…*/` with spaces, preserving byte offsets so the scan
/// above can index either string interchangeably.
fn strip_comments(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0usize;
    while i < b.len() {
        if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
            while i < b.len() && b[i] != b'\n' {
                out.push(b' ');
                i += 1;
            }
        } else if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
            let mut closed = false;
            while i < b.len() {
                if b[i] == b'*' && i + 1 < b.len() && b[i + 1] == b'/' {
                    out.push(b' ');
                    out.push(b' ');
                    i += 2;
                    closed = true;
                    break;
                }
                out.push(if b[i] == b'\n' { b'\n' } else { b' ' });
                i += 1;
            }
            if !closed {
                break;
            }
        } else {
            // Multi-byte UTF-8 passes through untouched.
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8(out).unwrap_or_else(|_| String::from(src))
}

// --- inspection -------------------------------------------------------------

/// A cursor over a wasm module's section list.
struct Sections<'a> {
    d: &'a [u8],
    p: usize,
}

impl<'a> Sections<'a> {
    fn new(d: &'a [u8]) -> Option<Self> {
        if d.len() < 8 || &d[0..4] != b"\0asm" {
            return None;
        }
        Some(Sections { d, p: 8 })
    }

    fn uleb(&mut self) -> Option<u32> {
        let mut r = 0u32;
        let mut s = 0u32;
        loop {
            let b = *self.d.get(self.p)?;
            self.p += 1;
            r |= ((b & 0x7f) as u32).checked_shl(s)?;
            if b & 0x80 == 0 {
                return Some(r);
            }
            s += 7;
            if s > 28 {
                return None;
            }
        }
    }

    /// Next `(id, payload)`, or `None` at the end / on malformed input.
    fn next(&mut self) -> Option<(u8, &'a [u8])> {
        if self.p >= self.d.len() {
            return None;
        }
        let id = self.d[self.p];
        self.p += 1;
        let len = self.uleb()? as usize;
        let start = self.p;
        let end = start.checked_add(len)?;
        if end > self.d.len() {
            return None;
        }
        self.p = end;
        Some((id, &self.d[start..end]))
    }
}

/// Read a length-prefixed name from the front of a payload.
fn take_name(p: &[u8]) -> Option<(&str, &[u8])> {
    let mut s = Sections { d: p, p: 0 };
    let len = s.uleb()? as usize;
    let at = s.p;
    let end = at.checked_add(len)?;
    let name = core::str::from_utf8(p.get(at..end)?).ok()?;
    Some((name, &p[end..]))
}

fn custom_section<'a>(wasm: &'a [u8], want: &str) -> Option<&'a [u8]> {
    let mut secs = Sections::new(wasm)?;
    while let Some((id, payload)) = secs.next() {
        if id != 0 {
            continue;
        }
        if let Some((name, rest)) = take_name(payload) {
            if name == want {
                return Some(rest);
            }
        }
    }
    None
}

/// A Javy plugin's import namespace, from its `import_namespace` custom section.
///
/// Read from the bytes rather than through wasmi because the engine is built with
/// `ignore_custom_sections(true)`, so it never surfaces one.
pub fn import_namespace(plugin_wasm: &[u8]) -> Option<&str> {
    core::str::from_utf8(custom_section(plugin_wasm, NAMESPACE_SECTION)?).ok()
}

/// The plugin identity [`emit`] stamped into a module, if any.
pub fn plugin_stamp(wasm: &[u8]) -> Option<&str> {
    let payload = custom_section(wasm, STAMP_SECTION)?;
    core::str::from_utf8(payload).ok()?.strip_prefix("plugin=")
}

/// Does this module import from `namespace`? This is how the runtime tells a
/// JS-derived module from a hand-written one — **the module's own imports**, not a
/// manifest flag, so the two can never be confused or mislabelled.
pub fn links_plugin(wasm: &[u8], namespace: &str) -> bool {
    let Some(mut secs) = Sections::new(wasm) else {
        return false;
    };
    while let Some((id, payload)) = secs.next() {
        if id != 2 {
            continue;
        }
        let mut s = Sections { d: payload, p: 0 };
        let Some(n) = s.uleb() else { return false };
        for _ in 0..n {
            let at = s.p;
            let Some((module, _)) = take_name(&payload[at..]) else {
                return false;
            };
            if module == namespace {
                return true;
            }
            // Skip this entry: module name, field name, kind, then its descriptor.
            let Some(len) = s.uleb() else { return false };
            s.p += len as usize;
            let Some(flen) = s.uleb() else { return false };
            s.p += flen as usize;
            let Some(kind) = s.d.get(s.p).copied() else {
                return false;
            };
            s.p += 1;
            match kind {
                0x00 => {
                    let _ = s.uleb();
                }
                0x01 => {
                    s.p += 1; // reftype
                    let flags = s.d.get(s.p).copied().unwrap_or(0);
                    s.p += 1;
                    let _ = s.uleb();
                    if flags & 1 == 1 {
                        let _ = s.uleb();
                    }
                }
                0x02 => {
                    let flags = s.d.get(s.p).copied().unwrap_or(0);
                    s.p += 1;
                    let _ = s.uleb();
                    if flags & 1 == 1 {
                        let _ = s.uleb();
                    }
                }
                0x03 => {
                    s.p += 2; // valtype + mutability
                }
                _ => return false,
            }
        }
    }
    false
}

/// Export names of a module, for cross-checking a built artifact against the
/// tools a manifest declares.
pub fn export_names(wasm: &[u8]) -> Vec<String> {
    let mut out = Vec::new();
    let Some(mut secs) = Sections::new(wasm) else {
        return out;
    };
    while let Some((id, payload)) = secs.next() {
        if id != 7 {
            continue;
        }
        let mut s = Sections { d: payload, p: 0 };
        let Some(n) = s.uleb() else { return out };
        for _ in 0..n {
            let at = s.p;
            let Some((name, _)) = take_name(&payload[at..]) else {
                return out;
            };
            out.push(String::from(name));
            let Some(len) = s.uleb() else { return out };
            s.p += len as usize;
            s.p += 1; // kind
            let _ = s.uleb(); // index
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const NS: &str = "chitti_js_v1";
    const STAMP: &str = "chitti_js_v1@1";

    /// Bytecode stand-in: `emit` never interprets it, so any bytes exercise the
    /// framing. Correct *execution* of real bytecode is covered where the plugin
    /// is available; this is about the module we wrap it in.
    fn fake_bc() -> Vec<u8> {
        (0..300u32).map(|i| (i % 251) as u8).collect()
    }

    #[test_case]
    fn emitted_module_validates() {
        // The real check: wasmi's validator agrees this is a module. A wrong
        // LEB128 length or a mis-ordered section fails here, which is what makes
        // this worth more than comparing bytes to a golden blob.
        let m = emit(NS, STAMP, &fake_bc(), &["demo_echo", "demo_sum"]).expect("emit");
        crate::agent::wasm_rt::validate_module(&m).expect("emitted module must validate");
    }

    #[test_case]
    fn emitted_module_exports_exactly_the_tools_asked_for() {
        let m = emit(NS, STAMP, &fake_bc(), &["a", "notes_list", "z9_$x"]).expect("emit");
        assert_eq!(export_names(&m), alloc::vec!["a", "notes_list", "z9_$x"]);
    }

    #[test_case]
    fn emitted_module_is_recognisable_as_js_derived() {
        let m = emit(NS, STAMP, &fake_bc(), &["t"]).expect("emit");
        assert!(links_plugin(&m, NS), "the runtime must detect this as a JS module");
        assert!(!links_plugin(&m, "some_other_plugin"), "and not confuse namespaces");
        // A hand-written Rust tool module must never be mistaken for one.
        assert!(!links_plugin(crate::agent::wasm_rt::FIXTURE_ECHO, NS));
        assert!(!links_plugin(crate::agent::wasm_rt::FIXTURE_HOST_STORAGE, NS));
    }

    #[test_case]
    fn the_plugin_stamp_round_trips() {
        // Version skew is otherwise a run-time `invalid version` from inside
        // QuickJS; the stamp turns it into "rebuild this".
        let m = emit(NS, STAMP, &fake_bc(), &["t"]).expect("emit");
        assert_eq!(plugin_stamp(&m), Some(STAMP));
        assert_eq!(plugin_stamp(crate::agent::wasm_rt::FIXTURE_ECHO), None);
    }

    #[test_case]
    fn names_and_bytecode_are_refused_when_unusable() {
        assert_eq!(emit(NS, STAMP, &fake_bc(), &[]).unwrap_err(), EmitError::NoExports);
        assert_eq!(emit(NS, STAMP, &[], &["t"]).unwrap_err(), EmitError::EmptyBytecode);
        // A JS export has to be an identifier, so these could never resolve at
        // `invoke` — refused at build time instead of failing per call.
        for bad in ["", "has space", "has-dash", "9lives", "tab\tname"] {
            assert_eq!(
                emit(NS, STAMP, &fake_bc(), &[bad]).unwrap_err(),
                EmitError::BadName,
                "{bad:?} must be refused"
            );
        }
        let many: alloc::vec::Vec<&str> = core::iter::repeat("t").take(MAX_EXPORTS + 1).collect();
        assert_eq!(emit(NS, STAMP, &fake_bc(), &many).unwrap_err(), EmitError::TooManyExports);
    }

    #[test_case]
    fn a_long_bytecode_still_validates() {
        // Lengths above 127 need multi-byte LEB128 in three places; a single-byte
        // assumption would produce a module that fails validation here rather
        // than at run time.
        for len in [1usize, 127, 128, 300, 16_384] {
            let bc: alloc::vec::Vec<u8> = (0..len).map(|i| (i % 255) as u8).collect();
            let m = emit(NS, STAMP, &bc, &["t"]).expect("emit");
            crate::agent::wasm_rt::validate_module(&m)
                .unwrap_or_else(|e| panic!("bytecode len {len} produced an invalid module: {e}"));
        }
    }

    #[test_case]
    fn the_namespace_is_read_from_a_plugins_custom_section() {
        // Shape check against a module we built: a plugin carries the same
        // section, and the engine ignores custom sections so it must be parsed
        // from the bytes.
        let mut fake_plugin = alloc::vec::Vec::new();
        fake_plugin.extend_from_slice(b"\0asm\x01\0\0\0");
        let mut payload = alloc::vec::Vec::new();
        wasm_name(NAMESPACE_SECTION, &mut payload);
        payload.extend_from_slice(b"javy-default-plugin-v4");
        section(0, &payload, &mut fake_plugin);
        assert_eq!(import_namespace(&fake_plugin), Some("javy-default-plugin-v4"));
        assert_eq!(import_namespace(b"not wasm"), None);
    }

    #[test_case]
    fn exports_are_scanned_from_the_script() {
        let src = r#"
            function helper() {}
            export function demo_echo() {}
            export   async   function slow_one() {}
            export function $weird_1() {}
            // export function commented_out() {}
            /* export function
               also_commented() {} */
            export function* gen() {}
            export const notAFunction = 1;
        "#;
        // Comments excluded, generators excluded (Javy cannot call them), and a
        // non-function export is not a tool.
        assert_eq!(scan_exports(src), alloc::vec!["demo_echo", "slow_one", "$weird_1"]);
    }

    #[test_case]
    fn the_export_scanner_does_not_trip_on_odd_input() {
        assert!(scan_exports("").is_empty());
        assert!(scan_exports("exported_thing = 1; reexport function x(){}").is_empty());
        assert!(scan_exports("/* unterminated").is_empty());
        assert!(scan_exports("export").is_empty());
        assert!(scan_exports("export function").is_empty());
        // A name repeated is listed once, since it is one wasm export.
        assert_eq!(scan_exports("export function a(){} export function a(){}").len(), 1);
        // Non-ASCII source must not panic the byte scan.
        let _ = scan_exports("const s = \"héllo wörld\"; export function ok(){}");
        assert_eq!(scan_exports("const s = \"héllo\"; export function ok(){}"), alloc::vec!["ok"]);
    }

    #[test_case]
    fn malformed_input_is_never_a_panic() {
        // Every inspector takes attacker-shaped bytes: a truncated section header,
        // a length that runs past the end, a bogus import kind.
        let m = emit(NS, STAMP, &fake_bc(), &["t"]).expect("emit");
        for cut in [0usize, 1, 4, 8, 9, 20, 40, m.len() / 2, m.len() - 1] {
            let part = &m[..cut];
            let _ = links_plugin(part, NS);
            let _ = plugin_stamp(part);
            let _ = export_names(part);
            let _ = import_namespace(part);
        }
        let mut corrupt = m.clone();
        for i in (8..corrupt.len()).step_by(37) {
            corrupt[i] = 0xff;
        }
        let _ = links_plugin(&corrupt, NS);
        let _ = export_names(&corrupt);
    }
}
