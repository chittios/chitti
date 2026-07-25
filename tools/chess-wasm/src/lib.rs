//! Chess package: the complete game as `tools.wasm` — rules engine, board UI,
//! and the model-opponent flow — run by the kernel's generic package-UI
//! runtime. **No chess logic lives in the kernel.**
//!
//! Exports (string ABI — packed i64 return = (ptr << 32) | len):
//! * `chess_start` — reset/restore + paint; `{model:true}` enables the agent
//!   opponent (it answers every human move via the runtime's `ask:` protocol).
//! * `on_click {x,y}` / `on_key {key}` / `on_reply {text}` / `tick` — the
//!   package-UI runtime hooks (see `app.rs`).
//! * `chess_legal` — args JSON `{fen?, from}` → `legal:from->sq,...`
//! * `chess_try_move` — args JSON `{fen?, from, to}` → new FEN or `error:…`
//!   (also calls `host_board_set` + session `host_storage_set` on success).
//!   Both default to the running game's position when `fen` is omitted.
//! * `chitti_alloc` — bump allocator for host string ABI writes
//!
//! Host imports live under module `chitti` (see `kernel/src/agent/wasm_rt.rs`).

#![no_std]
#![no_main]

extern crate alloc;

mod app;
mod endscreen;
mod rules;

use alloc::format;
use alloc::string::String;
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
    fn host_board_mark(sq_ptr: i32, sq_len: i32, color_ptr: i32, color_len: i32) -> i32;
    fn host_ui_draw(ops_ptr: i32, ops_len: i32) -> i32;
    fn host_hud_set(text_ptr: i32, text_len: i32) -> i32;
    fn host_storage_set(scope: i32, key_ptr: i32, key_len: i32, val_ptr: i32, val_len: i32) -> i32;
    fn host_storage_get(scope: i32, key_ptr: i32, key_len: i32, out_ptr: i32, cap: i32) -> i32;
    fn host_now_ms() -> i64;
    fn host_log(ptr: i32, len: i32);
}

// --- safe wrappers shared with `app` ----------------------------------------

pub(crate) fn board_set(fen: &str) -> i32 {
    let b = fen.as_bytes();
    unsafe { host_board_set(b.as_ptr() as i32, b.len() as i32) }
}

pub(crate) fn board_mark(squares: &str, color: &str) -> i32 {
    let s = squares.as_bytes();
    let c = color.as_bytes();
    unsafe { host_board_mark(s.as_ptr() as i32, s.len() as i32, c.as_ptr() as i32, c.len() as i32) }
}

pub(crate) fn ui_draw(ops: &str) -> i32 {
    let b = ops.as_bytes();
    unsafe { host_ui_draw(b.as_ptr() as i32, b.len() as i32) }
}

/// Set the surface HUD (status line + wrapped hint lines, '\n'-separated). The
/// compositor renders it crisp in a reserved pane-space strip.
pub(crate) fn hud_set(text: &str) -> i32 {
    let b = text.as_bytes();
    unsafe { host_hud_set(b.as_ptr() as i32, b.len() as i32) }
}

pub(crate) fn storage_set(scope: i32, key: &str, val: &str) -> i32 {
    let k = key.as_bytes();
    let v = val.as_bytes();
    unsafe {
        host_storage_set(scope, k.as_ptr() as i32, k.len() as i32, v.as_ptr() as i32, v.len() as i32)
    }
}

pub(crate) fn storage_get(scope: i32, key: &str, out: &mut [u8]) -> i32 {
    let k = key.as_bytes();
    unsafe {
        host_storage_get(scope, k.as_ptr() as i32, k.len() as i32, out.as_mut_ptr() as i32, out.len() as i32)
    }
}

pub(crate) fn now_ms() -> i64 {
    unsafe { host_now_ms() }
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
pub(crate) fn json_str(blob: &str, key: &str) -> Option<String> {
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

/// Host string-ABI allocator (ptr return). The host calls this exactly once
/// per call cycle (for the args buffer, before invoking the export), so it is
/// also where the bump heap RESETS — required now that the package-UI runtime
/// keeps one persistent instance for the whole game (the previous call's
/// result has already been copied out; app state lives in plain-array statics,
/// never on this heap).
#[no_mangle]
pub extern "C" fn chitti_alloc(size: i32) -> i32 {
    HEAP_OFF.store(0, Ordering::Relaxed);
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

// --- package-UI runtime hooks (game/UI logic in `app.rs`) --------------------

macro_rules! hook {
    ($name:ident, $body:expr) => {
        #[no_mangle]
        pub extern "C" fn $name(args_ptr: i32, args_len: i32) -> i64 {
            let raw = match core::str::from_utf8(unpack_args(args_ptr, args_len)) {
                Ok(s) => s,
                Err(_) => return result_string("error:bad utf-8"),
            };
            #[allow(clippy::redundant_closure_call)]
            result_string(&$body(raw))
        }
    };
}

hook!(chess_start, app::start);
hook!(on_reply, app::on_reply);

#[no_mangle]
pub extern "C" fn on_click(args_ptr: i32, args_len: i32) -> i64 {
    let raw = match core::str::from_utf8(unpack_args(args_ptr, args_len)) {
        Ok(s) => s,
        Err(_) => return result_string("error:bad utf-8"),
    };
    let x = json_i32(raw, "x", -1);
    let y = json_i32(raw, "y", -1);
    result_string(&app::on_click(x, y))
}

#[no_mangle]
pub extern "C" fn on_key(args_ptr: i32, args_len: i32) -> i64 {
    let raw = match core::str::from_utf8(unpack_args(args_ptr, args_len)) {
        Ok(s) => s,
        Err(_) => return result_string("error:bad utf-8"),
    };
    let key = json_str(raw, "key").unwrap_or_default();
    result_string(&app::on_key(&key))
}

#[no_mangle]
pub extern "C" fn tick(_args_ptr: i32, _args_len: i32) -> i64 {
    result_string(&app::tick())
}

/// Minimal JSON integer field extractor (decimal, optional minus).
pub(crate) fn json_i32(blob: &str, key: &str, default: i32) -> i32 {
    let pat = format!("\"{key}\"");
    let Some(i) = blob.find(&pat) else { return default };
    let rest = blob[i + pat.len()..].trim_start();
    let Some(rest) = rest.strip_prefix(':') else { return default };
    let rest = rest.trim_start();
    let mut end = 0;
    for (j, c) in rest.char_indices() {
        if c.is_ascii_digit() || (j == 0 && c == '-') {
            end = j + 1;
        } else {
            break;
        }
    }
    if end == 0 {
        return default;
    }
    rest[..end].parse().unwrap_or(default)
}

/// `chess_legal` — JSON `{ "fen", "from" }` → `legal:from->…`
#[no_mangle]
pub extern "C" fn chess_legal(args_ptr: i32, args_len: i32) -> i64 {
    let raw = match core::str::from_utf8(unpack_args(args_ptr, args_len)) {
        Ok(s) => s,
        Err(_) => return result_string("error:bad utf-8 args"),
    };
    // The running game's position is the default — chat can just name a square.
    let fen = json_str(raw, "fen").filter(|f| !f.is_empty()).unwrap_or_else(app::current_fen);
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
    if from.is_empty() {
        return result_string("error: need from");
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
    // The running game's position is the default; a successful move updates it
    // so the board UI and chat stay one coherent game.
    let fen = json_str(raw, "fen").filter(|f| !f.is_empty()).unwrap_or_else(app::current_fen);
    let from = json_str(raw, "from").unwrap_or_default();
    let to = json_str(raw, "to").unwrap_or_default();
    if from.is_empty() || to.is_empty() {
        return result_string("error: need from, to");
    }
    match rules::try_move(&fen, &from, &to) {
        Ok(new_fen) => {
            let rc = board_set(&new_fen);
            if rc != 0 {
                return result_string("error:board_set failed");
            }
            app::note_external_fen(&new_fen);
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
