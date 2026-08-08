//! **git** tool module for the Chitti **git agent** — deterministic version
//! control below the determinism boundary, compiled to wasm
//! (`agents/git/assets/tools.wasm`) and run by the kernel's wasm runtime.
//!
//! A minimal but real git over the agent's home store: the working tree is a
//! store directory under `/agent/<id>/git/`, `.git/` holds `HEAD`/`refs`/
//! `index` and a **loose-object database** (real SHA-1 names, zlib-deflated
//! objects), and `clone`/`push` speak the git smart-HTTP protocol
//! (`git-upload-pack` / `git-receive-pack`).
//!
//! All primitives are **host imports** (`chitti::host_*`): the kernel's store,
//! SHA-1, zlib (inflate + stored-block deflate), clock, and HTTP — gated by
//! the agent's capabilities and home scope. This module is pure logic over
//! those imports; it never touches hardware or memory directly.
//!
//! Export:
//! * `git_command` — args JSON `{"args":"<full /git … line>"}` → result text.
//! * `chitti_alloc` — allocator for the host string ABI writes.

#![cfg_attr(target_arch = "wasm32", no_std)]

extern crate alloc;

use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

// --- wasm glue: growable bump allocator + panic handler -------------------------
#[cfg(target_arch = "wasm32")]
mod wasm_glue {
    use core::alloc::{GlobalAlloc, Layout};
    use core::sync::atomic::{AtomicUsize, Ordering};

    const PAGE: usize = 65536;
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    static END: AtomicUsize = AtomicUsize::new(0);

    pub struct Bump;
    unsafe impl GlobalAlloc for Bump {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let align = layout.align().max(8);
            let size = layout.size().max(1);
            let mut next = NEXT.load(Ordering::Relaxed);
            if next == 0 {
                let pages = core::arch::wasm32::memory_size(0);
                next = pages * PAGE;
                END.store(next, Ordering::Relaxed);
            }
            let start = (next + align - 1) & !(align - 1);
            let new_next = match start.checked_add(size) {
                Some(v) => v,
                None => return core::ptr::null_mut(),
            };
            let mut end = END.load(Ordering::Relaxed);
            if new_next > end {
                let need = (new_next - end).div_ceil(PAGE);
                if core::arch::wasm32::memory_grow(0, need) == usize::MAX {
                    return core::ptr::null_mut();
                }
                end += need * PAGE;
                END.store(end, Ordering::Relaxed);
            }
            NEXT.store(new_next, Ordering::Relaxed);
            start as *mut u8
        }
        unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {}
    }

    #[global_allocator]
    static ALLOC: Bump = Bump;

    #[panic_handler]
    fn panic(_info: &core::panic::PanicInfo) -> ! {
        loop {}
    }
}

// --- host imports (kernel wasm runtime, wasm_rt.rs) -------------------------
//
// FS paths passed to `host_fs_*` are **absolute store paths**; `host_home`
// returns the calling agent's `/agent/<id>` so we build them ourselves. Write
// is home-scoped on the host; read/list are unscoped store reads.

#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "chitti")]
extern "C" {
    fn host_fs_read(path: *const u8, path_len: i32, out: *mut u8, out_cap: i32) -> i32;
    fn host_fs_write(path: *const u8, path_len: i32, data: *const u8, data_len: i32) -> i32;
    fn host_fs_list(path: *const u8, path_len: i32, out: *mut u8, out_cap: i32) -> i32;
    fn host_fs_exists(path: *const u8, path_len: i32) -> i32;
    fn host_home(out: *mut u8, out_cap: i32) -> i32;
    fn host_user_home(out: *mut u8, out_cap: i32) -> i32;
    fn host_now_unix() -> i64;
    fn host_sha1(src: *const u8, src_len: i32, out: *mut u8) -> i32;
    fn host_inflate(src: *const u8, src_len: i32, out: *mut u8, out_cap: i32) -> i64;
    fn host_deflate(src: *const u8, src_len: i32, out: *mut u8, out_cap: i32) -> i32;
    fn host_http(req: *const u8, req_len: i32, out: *mut u8, out_cap: i32) -> i64;
    fn host_ssh(req: *const u8, req_len: i32, out: *mut u8, out_cap: i32) -> i64;
}

// A native build gets the same imports from a simulator over the kernel's own
// SHA-1/zlib, so the git logic — the packfile walk above all — is testable
// without an OS. The call sites below are identical either way.
#[cfg(not(target_arch = "wasm32"))]
pub mod hostsim;
#[cfg(not(target_arch = "wasm32"))]
use hostsim::{
    host_deflate, host_fs_exists, host_fs_list, host_fs_read, host_fs_write, host_home,
    host_http, host_inflate, host_now_unix, host_sha1, host_ssh, host_user_home,
};

/// Shared **output** buffer for host-import results (single-threaded wasm).
///
/// Grown on demand, never shrunk. Inputs are *not* staged through it — a host
/// import reads guest memory at whatever pointer it is handed, so a slice can be
/// passed where it already lives. Copying inputs in here is not just waste: the
/// copy was `s[..src.len()].copy_from_slice(src)` against a fixed 64 KiB buffer,
/// which panics on anything larger, and a wasm panic handler is `loop {}` — so
/// `git add` of a 100 KiB file hung the instance until it ran out of fuel.
static mut SCRATCH: Option<Vec<u8>> = None;

/// The shell agent's current directory, passed in the tool call (`{"cwd":…}`).
/// git resolves relative defaults (clone's folder, init's target) against it,
/// exactly like the git CLI running in a shell.
static mut CURRENT_CWD: Option<String> = None;

/// The output buffer, grown to at least `want` bytes.
fn scratch(want: usize) -> &'static mut Vec<u8> {
    unsafe {
        let b = SCRATCH.get_or_insert_with(|| vec![0u8; SCRATCH_MIN]);
        if b.len() < want {
            b.resize(want, 0);
        }
        b
    }
}

/// Starting size for a host-import result. Big enough that a loose object, a
/// directory listing or a refs advertisement lands in one call.
const SCRATCH_MIN: usize = 64 << 10;

/// Starting size for an HTTP response. A clone's packfile is the one host result
/// that is routinely hundreds of KiB, and unlike a file read a second attempt
/// costs a **second download** — so this starts where most repositories fit
/// rather than where the smallest one does.
const HTTP_MIN: usize = 1 << 20;

/// Run a host import that fills the output buffer, sizing the buffer from what it
/// reports.
///
/// `f(ptr, cap) -> (aux, len)`: a negative `len` is an error, and a `len` greater
/// than `cap` means the host had more to give — the buffer grows to exactly that
/// and the call is repeated once. This is the guest half of the host's
/// `fill_guest` contract, and it is the fix for the bug this file's history is
/// about: with a fixed buffer and a host that answered `min(len, cap)`, a 182 KiB
/// packfile arrived as a complete-looking 64 KiB one and the clone died inside the
/// decompressor, pointing at the wrong layer.
fn filled(hint: usize, f: impl Fn(*mut u8, i32) -> (i64, i64)) -> Option<(i64, Vec<u8>)> {
    let mut want = hint;
    for _ in 0..2 {
        let (ptr, cap) = {
            let b = scratch(want);
            (b.as_mut_ptr(), b.len())
        };
        let (aux, n) = f(ptr, cap as i32);
        if n < 0 {
            return None;
        }
        let n = n as usize;
        if n <= cap {
            return Some((aux, scratch(0)[..n].to_vec()));
        }
        want = n;
    }
    None
}

/// The calling agent's home (`/agent/<id>`).
fn home() -> String {
    let s = scratch(SCRATCH_MIN);
    let n = unsafe { host_home(s.as_mut_ptr(), s.len() as i32) };
    if n <= 0 {
        return String::new();
    }
    String::from_utf8_lossy(&s[..(n as usize).min(s.len())]).into_owned()
}

fn fs_read(path: &str) -> Option<Vec<u8>> {
    filled(SCRATCH_MIN, |p, c| {
        (0, unsafe { host_fs_read(path.as_ptr(), path.len() as i32, p, c) } as i64)
    })
    .map(|(_, v)| v)
}

fn fs_write(path: &str, data: &[u8]) -> bool {
    unsafe { host_fs_write(path.as_ptr(), path.len() as i32, data.as_ptr(), data.len() as i32) == 0 }
}

fn fs_exists(path: &str) -> bool {
    unsafe { host_fs_exists(path.as_ptr(), path.len() as i32) == 1 }
}

/// List a directory: `name\t<d|f>\tsize\n` per child.
fn fs_list(path: &str) -> Vec<(String, bool, u64)> {
    let Some((_, raw)) = filled(SCRATCH_MIN, |p, c| {
        (0, unsafe { host_fs_list(path.as_ptr(), path.len() as i32, p, c) } as i64)
    }) else {
        return Vec::new();
    };
    let text = String::from_utf8_lossy(&raw);
    let mut out = Vec::new();
    for line in text.lines() {
        let mut it = line.splitn(3, '\t');
        if let (Some(name), Some(kind), Some(size)) = (it.next(), it.next(), it.next()) {
            if name.is_empty() {
                continue;
            }
            out.push((name.to_string(), kind == "d", size.parse().unwrap_or(0)));
        }
    }
    out
}

fn sha1(data: &[u8]) -> String {
    let mut digest = [0u8; 20];
    let ok = unsafe { host_sha1(data.as_ptr(), data.len() as i32, digest.as_mut_ptr()) };
    if ok != 0 {
        return String::new();
    }
    to_hex(&digest)
}

/// zlib-decompress `src`; returns `(bytes, input_consumed)`.
///
/// Trailing bytes after the stream are fine, and `input_consumed` says where it
/// ended — so a caller walking a packfile hands over the whole remainder and steps
/// by the answer.
fn zlib_decompress(src: &[u8]) -> Option<(Vec<u8>, usize)> {
    zlib_decompress_hint(src, SCRATCH_MIN)
}

/// [`zlib_decompress`] where the caller already knows the decompressed size (a
/// pack object header declares it), so the buffer is right the first time.
fn zlib_decompress_hint(src: &[u8], hint: usize) -> Option<(Vec<u8>, usize)> {
    let (consumed, out) = filled(hint.max(1), |p, c| {
        let r = unsafe { host_inflate(src.as_ptr(), src.len() as i32, p, c) };
        if r < 0 {
            return (0, -1);
        }
        (((r as u64) >> 32) as i64, r & 0xffff_ffff)
    })?;
    Some((out, consumed as usize))
}

fn zlib_deflate(src: &[u8]) -> Option<Vec<u8>> {
    // Stored blocks, so the output is the input plus a little framing.
    filled(src.len() + 1024, |p, c| {
        (0, unsafe { host_deflate(src.as_ptr(), src.len() as i32, p, c) } as i64)
    })
    .map(|(_, v)| v)
}

/// HTTP request/response. `req` JSON `{"m","u","h","b"}`; returns `(status,
/// body)` — the **whole** body, growing the buffer and re-requesting once if the
/// first attempt was not big enough.
fn http(method: &str, url: &str, headers: &[(&str, &str)], body: &[u8]) -> Result<(u16, Vec<u8>), i64> {
    use alloc::format;
    let mut h = String::new();
    for (k, v) in headers {
        if !h.is_empty() {
            h.push(';');
        }
        h.push_str(k);
        h.push_str(": ");
        h.push_str(v);
    }
    let req = format!(
        "{{\"m\":\"{method}\",\"u\":\"{}\",\"h\":\"{}\",\"b\":\"{}\"}}",
        json_escape(url),
        json_escape(&h),
        base64(body),
    );
    // The request goes straight out of its own allocation: a push body is the whole
    // packfile in base64 and staging it through a fixed buffer is what used to
    // panic the instance.
    let (status, out) = filled(HTTP_MIN, |p, c| {
        let r = unsafe { host_http(req.as_ptr(), req.len() as i32, p, c) };
        if r < 0 {
            return (r, -1);
        }
        ((r >> 32) & 0xffff, r & 0xffff_ffff)
    })
    .ok_or(-1i64)?;
    Ok((status as u16, out))
}

// --- hex / base64 -----------------------------------------------------------

const HEX: &[u8; 16] = b"0123456789abcdef";

fn to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0xf) as usize] as char);
    }
    s
}

fn from_hex(s: &str) -> Option<Vec<u8>> {
    if s.len() != 40 || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let nib = |c: u8| -> u8 {
        match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            b'A'..=b'F' => c - b'A' + 10,
            _ => 0,
        }
    };
    let mut out = Vec::with_capacity(20);
    for i in (0..s.len()).step_by(2) {
        out.push((nib(s.as_bytes()[i]) << 4) | nib(s.as_bytes()[i + 1]));
    }
    Some(out)
}

fn base64(data: &[u8]) -> String {
    const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in data.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(B64[(n >> 18) as usize & 63] as char);
        out.push(B64[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { B64[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { B64[n as usize & 63] as char } else { '=' });
    }
    out
}


fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c => out.push(c),
        }
    }
    out
}

// --- git logic (see kernel draft; ported to host-import primitives) ----------

pub mod git;
pub mod remote;
pub mod sshurl;

/// The agent's git root: `<home>/git`.
/// Collapse `//`, `/./` and `/../` in a store path.
fn normalize_path(p: &str) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for seg in p.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            s => parts.push(s),
        }
    }
    if parts.is_empty() {
        "/".to_string()
    } else {
        alloc::format!("/{}", parts.join("/"))
    }
}

/// The git **working directory**: the repo the commands operate on. Persisted
/// in the agent home (`.git_cwd`) by `init`/`clone`; defaults to the shell's
/// current directory (the pwd), else the user home.
fn git_root() -> String {
    let h = home();
    let cwd = fs_read(&alloc::format!("{h}/.git_cwd"))
        .map(|b| String::from_utf8_lossy(&b).trim().to_string())
        .filter(|s| !s.is_empty());
    cwd.unwrap_or_else(base_dir)
}

/// The ChittiOS user home (`/home/chitti`) — the shell agent's `~`.
pub(crate) fn user_home() -> String {
    let s = scratch(SCRATCH_MIN);
    let n = unsafe { host_user_home(s.as_mut_ptr(), s.len() as i32) };
    if n <= 0 {
        return "/home/chitti".to_string();
    }
    String::from_utf8_lossy(&s[..(n as usize).min(s.len())]).into_owned()
}

/// The base directory for relative git targets: the shell's current directory
/// when it invoked us, else the user home.
pub(crate) fn base_dir() -> String {
    unsafe { CURRENT_CWD.clone() }.unwrap_or_else(user_home)
}

/// Drop the process-wide guest state between native tests. On wasm the instance
/// itself is the boundary, so this exists only for the simulator.
#[cfg(not(target_arch = "wasm32"))]
pub fn reset_state() {
    unsafe {
        SCRATCH = None;
        CURRENT_CWD = None;
    }
}

/// Persist the git working directory for subsequent commands.
fn set_cwd(dir: &str) {
    let h = home();
    fs_write(&alloc::format!("{h}/.git_cwd"), dir.trim_end_matches('/').as_bytes());
}

/// `chitti_alloc(size) -> ptr` — allocator the host uses to write the result
/// into guest memory (the packed `(ptr<<32)|len` return ABI).
#[no_mangle]
pub extern "C" fn chitti_alloc(size: i32) -> i32 {
    if size <= 0 {
        return 0;
    }
    let v = alloc::vec![0u8; size as usize];
    let ptr = v.as_ptr() as usize;
    core::mem::forget(v); // owned by the host now
    ptr as i32
}

/// `git_command(args_ptr, args_len) -> i64` — the agent's single tool entry.
#[no_mangle]
pub extern "C" fn git_command(args_ptr: i32, args_len: i32) -> i64 {
    let args = if args_len > 0 {
        let p = args_ptr as usize;
        let slice = unsafe { core::slice::from_raw_parts(p as *const u8, args_len as usize) };
        String::from_utf8_lossy(slice).into_owned()
    } else {
        String::new()
    };
    // Args arrive as the tool JSON `{"args":"…","cwd":"…"}`; pull the inner
    // line and remember the shell's current directory.
    if let Some(cwd) = extract_field(&args, "cwd") {
        unsafe { CURRENT_CWD = Some(cwd) };
    }
    let line = extract_args(&args);
    let out = crate::git::command(&line);
    let bytes = out.as_bytes();
    let ptr = chitti_alloc(bytes.len() as i32);
    if ptr == 0 {
        return 0;
    }
    unsafe {
        core::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr as *mut u8, bytes.len());
    }
    ((ptr as i64) << 32) | bytes.len() as i64
}

/// Pull the `"args"` field out of the tool-call JSON `{"args":"…"}`.
fn extract_args(json: &str) -> String {
    let Some(i) = json.find("\"args\"") else {
        return json.trim().to_string();
    };
    let rest = &json[i + 6..];
    let Some(colon) = rest.find(':') else { return json.trim().to_string() };
    let after = rest[colon + 1..].trim_start();
    if let Some(v) = after.strip_prefix('"') {
        let mut out = String::new();
        let mut it = v.chars();
        while let Some(c) = it.next() {
            match c {
                '"' => break,
                '\\' => {
                    if let Some(n) = it.next() {
                        out.push(n);
                    }
                }
                c => out.push(c),
            }
        }
        out
    } else {
        json.trim().to_string()
    }
}

/// Pull a quoted string field (`"key":"value"`) out of the tool-call JSON.
fn extract_field(json: &str, key: &str) -> Option<String> {
    let pat = alloc::format!("\"{key}\"");
    let i = json.find(&pat)?;
    let rest = &json[i + pat.len()..];
    let colon = rest.find(':')?;
    let after = rest[colon + 1..].trim_start();
    let v = after.strip_prefix('"')?;
    let mut out = String::new();
    let mut it = v.chars();
    while let Some(c) = it.next() {
        match c {
            '"' => break,
            '\\' => {
                if let Some(n) = it.next() {
                    out.push(n);
                }
            }
            c => out.push(c),
        }
    }
    Some(out)
}
