//! Chess package tools compiled to `tools.wasm`.
//!
//! Exports (string ABI — packed i64 return = (ptr << 32) | len):
//! * `chess_legal` — args JSON `{fen, from}` → `legal:from->sq,...`
//! * `chess_try_move` — args JSON `{fen, from, to}` → new FEN or `error:…`
//!   (also calls `host_board_set` + session `host_storage_set` on success)
//! * `chitti_alloc` — bump allocator for host string ABI writes
//!
//! Host imports live under module `chitti` (see `kernel/src/agent/wasm_rt.rs`).

#![no_std]
#![no_main]

extern crate alloc;

mod rules;

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::alloc::{GlobalAlloc, Layout};
use core::sync::atomic::{AtomicUsize, Ordering};

// --- tiny bump allocator (single-threaded guest) ----------------------------

const HEAP_SIZE: usize = 64 * 1024;
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
                Ok(_) => {
                    // SAFETY: exclusive region inside HEAP.
                    return unsafe { HEAP.0.as_mut_ptr().add(aligned) };
                }
                Err(cur) => off = cur,
            }
        }
    }
    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {
        // bump: no free
    }
}

#[global_allocator]
static ALLOC: Bump = Bump;

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

// --- host imports ----------------------------------------------------------

#[link(wasm_import_module = "chitti")]
extern "C" {
    fn host_board_set(fen_ptr: i32, fen_len: i32) -> i32;
    fn host_storage_set(scope: i32, key_ptr: i32, key_len: i32, val_ptr: i32, val_len: i32) -> i32;
    fn host_log(ptr: i32, len: i32);
}

// --- string helpers --------------------------------------------------------

fn pack(ptr: i32, len: i32) -> i64 {
    ((ptr as u64) << 32 | (len as u32 as u64)) as i64
}

fn unpack_args(ptr: i32, len: i32) -> &'static [u8] {
    if ptr < 0 || len < 0 {
        return b"";
    }
    // SAFETY: host wrote args into our linear memory.
    unsafe { core::slice::from_raw_parts(ptr as *const u8, len as usize) }
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

/// Minimal JSON string field extractor (same spirit as session::todo::json_str).
fn json_str(blob: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\"");
    let i = blob.find(&pat)?;
    let rest = &blob[i + pat.len()..];
    let rest = rest.trim_start();
    let rest = rest.strip_prefix(':')?.trim_start();
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

// --- exports ---------------------------------------------------------------

/// Host string-ABI allocator (ptr return).
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

/// `chess_legal` — JSON `{ "fen", "from" }` → `legal:from->…`
#[no_mangle]
pub extern "C" fn chess_legal(args_ptr: i32, args_len: i32) -> i64 {
    let raw = match core::str::from_utf8(unpack_args(args_ptr, args_len)) {
        Ok(s) => s,
        Err(_) => return result_string("error:bad utf-8 args"),
    };
    let fen = json_str(raw, "fen").unwrap_or_default();
    let from = json_str(raw, "from")
        .or_else(|| {
            // bare square string
            let t = raw.trim().trim_matches('"');
            if t.len() == 2 {
                Some(t.into())
            } else {
                None
            }
        })
        .unwrap_or_default();
    if fen.is_empty() || from.is_empty() {
        return result_string("error: need fen and from");
    }
    let legal = rules::legal_moves(&fen, &from);
    result_string(&format!("legal:{from}->{legal}"))
}

/// `chess_try_move` — JSON `{fen,from,to}`; on success paints + stores FEN.
#[no_mangle]
pub extern "C" fn chess_try_move(args_ptr: i32, args_len: i32) -> i64 {
    let raw = match core::str::from_utf8(unpack_args(args_ptr, args_len)) {
        Ok(s) => s,
        Err(_) => return result_string("error:bad utf-8 args"),
    };
    let fen = json_str(raw, "fen").unwrap_or_default();
    let from = json_str(raw, "from").unwrap_or_default();
    let to = json_str(raw, "to").unwrap_or_default();
    if fen.is_empty() || from.is_empty() || to.is_empty() {
        return result_string("error: need fen, from, to");
    }
    match rules::try_move(&fen, &from, &to) {
        Ok(new_fen) => {
            let bytes = new_fen.as_bytes();
            let rc = unsafe { host_board_set(bytes.as_ptr() as i32, bytes.len() as i32) };
            if rc != 0 {
                return result_string("error:board_set failed");
            }
            // durable-ish: session storage key "fen"
            let key = b"fen";
            let _ = unsafe {
                host_storage_set(
                    0,
                    key.as_ptr() as i32,
                    key.len() as i32,
                    bytes.as_ptr() as i32,
                    bytes.len() as i32,
                )
            };
            result_string(&format!("ok:fen={new_fen}"))
        }
        Err(e) => result_string(&e),
    }
}

/// Silence unused import warnings for host_log in minimal builds.
#[allow(dead_code)]
fn _touch_log() {
    let m = b"chess-wasm";
    unsafe { host_log(m.as_ptr() as i32, m.len() as i32) };
}
