//! **PNG decoding, sandboxed.** The kernel's own `image/png.rs` and `image/inflate.rs`,
//! compiled to wasm and run under `agent/wasm_rt.rs`'s fuel + memory limits instead of in
//! ring 0.
//!
//! # Why this is the whole point
//!
//! PNG is attacker-supplied input parsed by a few thousand lines of bit-twiddling. In the
//! kernel a bounds-check failure there is a halted machine; here it is a wasm trap the host
//! reports as "this file is corrupt". That is the same trade the `pdf` module already makes,
//! and it is why the sandbox for one-shot image decoding is wasm rather than ring 3: no
//! linker script, no entry offset, no per-arch blob — one module, both arches.
//!
//! # Mounted, not copied
//!
//! The decoder is the kernel's source via `#[path]` (the `pdf-wasm` / `h264diff` pattern),
//! so there is exactly one PNG implementation in the repo. A port therefore **cannot**
//! introduce a decode regression — the differential test against the in-kernel decoder is
//! testing this harness, not the algorithm, which is precisely what makes the cutover safe.
//!
//! # ABI
//!
//! `png_decode(ptr, len) -> i64` in the host's `(ptr<<32)|len` shape, but the payload is
//! **raw bytes, not base64**: a 12-byte little-endian header `[w, h, ok]` as `u32`s followed
//! by `w*h` BGRA-ish `u32` pixels exactly as `image::Image::pixels` holds them. Base64 exists
//! on the chat tool path because that goes through JSON; a megapixel image must not pay a 4/3
//! expansion and a parse.
//!
//! On failure the header carries `ok = 0` and no pixels follow, so a caller distinguishes
//! "corrupt file" from "harness broke" without parsing prose.

#![cfg_attr(target_arch = "wasm32", no_std)]

extern crate alloc;

use alloc::vec::Vec;
// Carried over with `pdf-wasm`'s allocator glue, which brings its JSON argument helpers.
// Unused here — this module's ABI is raw bytes, not JSON — and stripped by LTO.
#[allow(unused_imports)]
use alloc::{format, string::String};

/// The kernel's DEFLATE. Shimmed under host `cargo test` exactly as `pdf-wasm` does it: the
/// real file's in-place tests use the kernel's custom test framework.
#[cfg(not(test))]
#[path = "../../../kernel/src/image/inflate.rs"]
pub mod inflate;
// No host-test shim yet: the wasm build is what ships, and its DEFLATE is covered by the
// kernel's own tests. Add pdf-wasm's stored-blocks shim here if host `cargo test` is wanted.

/// Mirrors `kernel/src/image/mod.rs`'s `Image`. Declared here rather than mounting the
/// kernel's `image/mod.rs`, which pulls in resize/render and framebuffer types a sandboxed
/// decoder has no business seeing. Field-for-field, because `png.rs` constructs it.
pub struct Image {
    pub w: usize,
    pub h: usize,
    pub pixels: Vec<u32>,
}

/// `image/png.rs` resolves `super::Image` and `super::inflate`, so it is mounted **at the
/// crate root**, where both exist.
///
/// Not inside a `mod image { … }`: `#[path]` on a nested inline module resolves relative to
/// `src/<module>/`, and traversing `..` through a directory that does not exist fails outright
/// ("couldn't read src/image/../../../../kernel/..."). Mounting at the root sidesteps it and
/// is simpler anyway.
#[path = "../../../kernel/src/image/png.rs"]
pub mod png;

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

/// Decode a PNG. See the module doc for the ABI.
#[no_mangle]
pub extern "C" fn png_decode(ptr: i32, len: i32) -> i64 {
    // SAFETY: the host wrote `len` bytes at `ptr` inside our own linear memory before the
    // call. A wasm module cannot address anything else, which is the property being bought.
    let input = unsafe { core::slice::from_raw_parts(ptr as *const u8, len as usize) };
    let out = encode(png::decode(input).ok());
    let p = out.as_ptr() as i64;
    let n = out.len() as i64;
    core::mem::forget(out); // the host reads it out of linear memory; the arena dies with the instance
    (p << 32) | n
}

/// Pack a decode result into the raw wire form: `[w, h, ok]` then pixels.
fn encode(img: Option<Image>) -> Vec<u8> {
    let mut out = Vec::new();
    match img {
        Some(i) => {
            out.extend_from_slice(&(i.w as u32).to_le_bytes());
            out.extend_from_slice(&(i.h as u32).to_le_bytes());
            out.extend_from_slice(&1u32.to_le_bytes());
            // Pixels as little-endian u32s, byte-for-byte what `Image::pixels` holds, so the
            // host can memcpy rather than convert.
            out.reserve(i.pixels.len() * 4);
            for px in &i.pixels {
                out.extend_from_slice(&px.to_le_bytes());
            }
        }
        None => {
            out.extend_from_slice(&0u32.to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes());
        }
    }
    out
}
