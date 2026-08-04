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
}

/// Shared scratch for host-import I/O (single-threaded wasm).
static mut SCRATCH: Option<Vec<u8>> = None;

/// The shell agent's current directory, passed in the tool call (`{"cwd":…}`).
/// git resolves relative defaults (clone's folder, init's target) against it,
/// exactly like the git CLI running in a shell.
static mut CURRENT_CWD: Option<String> = None;

fn scratch() -> &'static mut Vec<u8> {
    unsafe {
        if SCRATCH.is_none() {
            SCRATCH = Some(vec![0u8; SCRATCH_CAP]);
        }
        SCRATCH.as_mut().unwrap()
    }
}

/// Cap on one host-import result (a file, a list, an HTTP body). Generous for
/// the smart-HTTP packfile of a small repo; bigger clones are refused.
const SCRATCH_CAP: usize = 64 << 10;

/// The calling agent's home (`/agent/<id>`).
fn home() -> String {
    let s = scratch();
    let n = unsafe { host_home(s.as_mut_ptr(), s.len() as i32) };
    if n <= 0 {
        return String::new();
    }
    String::from_utf8_lossy(&s[..n as usize]).into_owned()
}

fn fs_read(path: &str) -> Option<Vec<u8>> {
    let s = scratch();
    let n = unsafe { host_fs_read(path.as_ptr(), path.len() as i32, s.as_mut_ptr(), s.len() as i32) };
    if n < 0 {
        return None;
    }
    Some(s[..n as usize].to_vec())
}

fn fs_write(path: &str, data: &[u8]) -> bool {
    unsafe { host_fs_write(path.as_ptr(), path.len() as i32, data.as_ptr(), data.len() as i32) == 0 }
}

fn fs_exists(path: &str) -> bool {
    unsafe { host_fs_exists(path.as_ptr(), path.len() as i32) == 1 }
}

/// List a directory: `name\t<d|f>\tsize\n` per child.
fn fs_list(path: &str) -> Vec<(String, bool, u64)> {
    let s = scratch();
    let n = unsafe { host_fs_list(path.as_ptr(), path.len() as i32, s.as_mut_ptr(), s.len() as i32) };
    if n < 0 {
        return Vec::new();
    }
    let text = String::from_utf8_lossy(&s[..n as usize]);
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
    let s = scratch();
    let mut digest = [0u8; 20];
    let ok = unsafe {
        s[..data.len()].copy_from_slice(data);
        host_sha1(s.as_mut_ptr(), data.len() as i32, digest.as_mut_ptr())
    };
    if ok != 0 {
        return String::new();
    }
    to_hex(&digest)
}

/// zlib-decompress `src`; returns `(bytes, input_consumed)`.
fn zlib_decompress(src: &[u8]) -> Option<(Vec<u8>, usize)> {
    let s = scratch();
    s[..src.len()].copy_from_slice(src);
    let r = unsafe { host_inflate(s.as_ptr(), src.len() as i32, s.as_mut_ptr(), s.len() as i32) };
    if r < 0 {
        return None;
    }
    let consumed = ((r as u64) >> 32) as usize;
    let n = (r & 0xffff_ffff) as usize;
    Some((s[..n].to_vec(), consumed))
}

fn zlib_deflate(src: &[u8]) -> Option<Vec<u8>> {
    let s = scratch();
    s[..src.len()].copy_from_slice(src);
    let n = unsafe { host_deflate(s.as_ptr(), src.len() as i32, s.as_mut_ptr(), s.len() as i32) };
    if n < 0 {
        return None;
    }
    Some(s[..n as usize].to_vec())
}

/// HTTP request/response. `req` JSON `{"m","u","h","b"}`; returns `(status,
/// body)` (body capped at the scratch).
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
    let s = scratch();
    s[..req.len()].copy_from_slice(req.as_bytes());
    let r = unsafe { host_http(s.as_ptr(), req.len() as i32, s.as_mut_ptr(), s.len() as i32) };
    if r < 0 {
        return Err(r);
    }
    let status = (r >> 32) as u16;
    let n = (r & 0xffff_ffff) as usize;
    Ok((status, s[..n.min(s.len())].to_vec()))
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

mod git;
mod remote;

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
    let s = scratch();
    let n = unsafe { host_user_home(s.as_mut_ptr(), s.len() as i32) };
    if n <= 0 {
        return "/home/chitti".to_string();
    }
    String::from_utf8_lossy(&s[..n as usize]).into_owned()
}

/// The base directory for relative git targets: the shell's current directory
/// when it invoked us, else the user home.
pub(crate) fn base_dir() -> String {
    unsafe { CURRENT_CWD.clone() }.unwrap_or_else(user_home)
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
