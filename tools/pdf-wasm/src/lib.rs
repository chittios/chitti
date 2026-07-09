//! PDF tool module for the Chitti **pdf agent** — deterministic document
//! digestion below the determinism boundary (the chess/doc `tools.wasm`
//! pattern: string ABI, packed i64 = `(ptr << 32) | len`, no host imports).
//!
//! Export:
//! * `pdf_digest` — args JSON `{"b64": "<base64 PDF>", "max_pages": N}` →
//!   digest JSON `{"pages", "title", "author", "truncated", "page_texts": [...]}`
//!   or `error:<reason>`. One call digests the whole document; the kernel
//!   runtime caches the digest and paints/answers from it natively.
//! * `chitti_alloc` — allocator for the host string ABI writes.
//!
//! Parsing is pure and bounds-checked: classic xref tables **and** xref
//! streams, object streams (ObjStm), FlateDecode with PNG predictors, the
//! page tree, and text extraction from content streams. Anything outside
//! that (LZW/DCT filters, encrypted files) degrades to a clear marker, never
//! a wrong answer. Inflate is the kernel's own `image/inflate.rs`, mounted.

#![cfg_attr(target_arch = "wasm32", no_std)]

extern crate alloc;

// The real decoder — the kernel's own DEFLATE (h264diff mount pattern). Its
// in-file tests use the kernel's custom test framework, so the host `cargo
// test` build swaps in a stored-blocks-only shim (all the fixtures need:
// they build valid zlib streams from stored blocks, no compressor required).
#[cfg(not(test))]
#[path = "../../../kernel/src/image/inflate.rs"]
pub mod inflate;
#[cfg(test)]
pub mod inflate {
    //! Test-only shim: RFC 1951 **stored blocks** (BTYPE=00). The real
    //! `image/inflate.rs` is what ships in the wasm; it has kernel unit tests.
    use alloc::vec::Vec;
    pub fn inflate(src: &[u8]) -> Result<Vec<u8>, &'static str> {
        let mut out = Vec::new();
        let mut p = 0usize;
        loop {
            let hdr = *src.get(p).ok_or("truncated")?;
            if hdr & 0b110 != 0 {
                return Err("test shim: only stored blocks");
            }
            let len = *src.get(p + 1).ok_or("truncated")? as usize | ((*src.get(p + 2).ok_or("truncated")? as usize) << 8);
            let end = p + 5 + len;
            if end > src.len() {
                return Err("truncated stored block");
            }
            out.extend_from_slice(&src[p + 5..end]);
            if hdr & 1 == 1 {
                return Ok(out);
            }
            p = end;
        }
    }
}
pub mod pdf;

use alloc::format;
use alloc::string::String;

// --- allocator ---------------------------------------------------------------
// Growable bump allocator over wasm linear memory (a PDF needs MBs; a fixed
// static heap would bloat the module / overflow). Never frees: each tool call
// runs in a fresh instance, so the arena dies with it.

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
                // First allocation: start after the current memory top.
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

// --- string ABI ----------------------------------------------------------------

fn pack(ptr: i32, len: i32) -> i64 {
    ((ptr as u64) << 32 | (len as u32 as u64)) as i64
}

fn result_string(s: &str) -> i64 {
    let b = alloc::boxed::Box::leak(s.as_bytes().to_vec().into_boxed_slice());
    pack(b.as_ptr() as i32, b.len() as i32)
}

/// Host string-ABI allocator (ptr return).
#[no_mangle]
pub extern "C" fn chitti_alloc(size: i32) -> i32 {
    if size <= 0 {
        return 0;
    }
    let v = alloc::vec![0u8; size as usize];
    alloc::boxed::Box::leak(v.into_boxed_slice()).as_ptr() as i32
}

/// Minimal JSON string field extractor (chess-wasm's json_str).
fn json_str(blob: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\"");
    let i = blob.find(&pat)?;
    let rest = blob[i + pat.len()..].trim_start().strip_prefix(':')?.trim_start();
    if !rest.starts_with('"') {
        return None;
    }
    let bytes = rest.as_bytes();
    let mut out = String::new();
    let mut j = 1;
    while j < bytes.len() {
        match bytes[j] {
            b'"' => break,
            b'\\' if j + 1 < bytes.len() => {
                out.push(bytes[j + 1] as char);
                j += 2;
            }
            c => {
                out.push(c as char);
                j += 1;
            }
        }
    }
    Some(out)
}

fn json_usize(blob: &str, key: &str) -> Option<usize> {
    let pat = format!("\"{key}\"");
    let i = blob.find(&pat)?;
    let rest = blob[i + pat.len()..].trim_start().strip_prefix(':')?.trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// Standard base64 (with padding) → bytes. `None` on any invalid character.
pub fn b64_decode(s: &str) -> Option<alloc::vec::Vec<u8>> {
    fn val(c: u8) -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => (c - b'A') as u32,
            b'a'..=b'z' => (c - b'a' + 26) as u32,
            b'0'..=b'9' => (c - b'0' + 52) as u32,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        })
    }
    let s = s.trim_end_matches('=');
    let mut out = alloc::vec::Vec::with_capacity(s.len() * 3 / 4 + 3);
    let mut acc: u32 = 0;
    let mut nbits = 0u32;
    for &c in s.as_bytes() {
        if c == b'\n' || c == b'\r' {
            continue;
        }
        acc = (acc << 6) | val(c)?;
        nbits += 6;
        if nbits >= 8 {
            nbits -= 8;
            out.push((acc >> nbits) as u8);
        }
    }
    Some(out)
}

/// `pdf_digest` — see module doc.
#[no_mangle]
pub extern "C" fn pdf_digest(args_ptr: i32, args_len: i32) -> i64 {
    if args_ptr <= 0 || args_len <= 0 {
        return result_string("error:bad args");
    }
    // SAFETY: the host wrote args into our linear memory at (ptr, len).
    let args = unsafe { core::slice::from_raw_parts(args_ptr as *const u8, args_len as usize) };
    let args = match core::str::from_utf8(args) {
        Ok(s) => s,
        Err(_) => return result_string("error:args not utf8"),
    };
    let Some(b64) = json_str(args, "b64") else {
        return result_string("error:missing b64");
    };
    let Some(bytes) = b64_decode(&b64) else {
        return result_string("error:bad base64");
    };
    let max_pages = json_usize(args, "max_pages").unwrap_or(20).clamp(1, 256);
    match pdf::digest(&bytes, max_pages) {
        Ok(json) => result_string(&json),
        Err(e) => result_string(&format!("error:{e}")),
    }
}
