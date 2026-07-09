//! Doc agent tools — deterministic request router for the content server.
//!
//! Export (string ABI, packed i64 = `(ptr << 32) | len`):
//! * `route_request` — args JSON `{method, path}` → response JSON
//! * `chitti_alloc` — bump allocator for host string ABI writes
//!
//! No host imports: routing is pure; the server reads named assets via Synapse.

#![no_std]
#![no_main]

extern crate alloc;

mod route;

use alloc::string::String;
use core::alloc::{GlobalAlloc, Layout};
use core::sync::atomic::{AtomicUsize, Ordering};

const HEAP_SIZE: usize = 32 * 1024;
#[repr(C, align(16))]
struct Heap([u8; HEAP_SIZE]);
static mut HEAP: Heap = Heap([0; HEAP_SIZE]);
static HEAP_OFF: AtomicUsize = AtomicUsize::new(0);

struct Bump;

// SAFETY: single-threaded wasm guest; bump only ever advances.
unsafe impl GlobalAlloc for Bump {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let align = layout.align().max(8);
        let size = layout.size();
        let mut off = HEAP_OFF.load(Ordering::Relaxed);
        loop {
            let aligned = (off + align - 1) & !(align - 1);
            let end = match aligned.checked_add(size) {
                Some(e) if e <= HEAP_SIZE => e,
                _ => return core::ptr::null_mut(),
            };
            match HEAP_OFF.compare_exchange_weak(off, end, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => return unsafe { HEAP.0.as_mut_ptr().add(aligned) },
                Err(cur) => off = cur,
            }
        }
    }
    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {}
}

#[global_allocator]
static ALLOC: Bump = Bump;

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

fn pack(ptr: i32, len: i32) -> i64 {
    ((ptr as u64) << 32 | (len as u32 as u64)) as i64
}

fn result_string(s: &str) -> i64 {
    let b = s.as_bytes();
    let layout = Layout::from_size_align(b.len().max(1), 1).unwrap();
    let p = unsafe { ALLOC.alloc(layout) };
    if p.is_null() {
        return pack(0, 0);
    }
    unsafe {
        core::ptr::copy_nonoverlapping(b.as_ptr(), p, b.len());
    }
    pack(p as i32, b.len() as i32)
}

fn unpack_args(ptr: i32, len: i32) -> &'static [u8] {
    if ptr < 0 || len < 0 {
        return b"";
    }
    unsafe { core::slice::from_raw_parts(ptr as *const u8, len as usize) }
}

fn json_str(blob: &str, key: &str) -> Option<String> {
    let pat = alloc::format!("\"{key}\"");
    let i = blob.find(&pat)?;
    let rest = &blob[i + pat.len()..];
    let rest = rest.trim_start().strip_prefix(':')?.trim_start();
    if !rest.starts_with('"') {
        return None;
    }
    let mut out = String::new();
    let bytes = rest.as_bytes();
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

#[no_mangle]
pub extern "C" fn chitti_alloc(size: i32) -> i32 {
    if size <= 0 {
        return 0;
    }
    let layout = match Layout::from_size_align(size as usize, 8) {
        Ok(l) => l,
        Err(_) => return -1,
    };
    let p = unsafe { ALLOC.alloc(layout) };
    if p.is_null() {
        -1
    } else {
        p as i32
    }
}

/// `route_request` — JSON `{ "method", "path" }` → response JSON for the server.
#[no_mangle]
pub extern "C" fn route_request(args_ptr: i32, args_len: i32) -> i64 {
    let raw = match core::str::from_utf8(unpack_args(args_ptr, args_len)) {
        Ok(s) => s,
        Err(_) => return result_string(r#"{"status":400,"body":"bad utf-8"}"#),
    };
    let method = json_str(raw, "method").unwrap_or_else(|| String::from("GET"));
    let path = json_str(raw, "path").unwrap_or_else(|| String::from("/"));
    result_string(&route::route_request(&method, &path))
}
