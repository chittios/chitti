//! Shared guest helpers + host imports.

use alloc::format;
use alloc::string::{String, ToString};
use core::alloc::{GlobalAlloc, Layout};
use core::sync::atomic::{AtomicUsize, Ordering};

const HEAP_SIZE: usize = 96 * 1024;
#[repr(C, align(16))]
struct Heap([u8; HEAP_SIZE]);
static mut HEAP: Heap = Heap([0; HEAP_SIZE]);
static HEAP_OFF: AtomicUsize = AtomicUsize::new(0);

struct Bump;
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
    unsafe fn dealloc(&self, _: *mut u8, _: Layout) {}
}

#[global_allocator]
static ALLOC: Bump = Bump;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[link(wasm_import_module = "chitti")]
extern "C" {
    pub fn host_storage_set(scope: i32, k: i32, kl: i32, v: i32, vl: i32) -> i32;
    pub fn host_storage_get(scope: i32, k: i32, kl: i32, out: i32, cap: i32) -> i32;
    pub fn host_storage_remove(scope: i32, k: i32, kl: i32) -> i32;
    pub fn host_storage_list(scope: i32, out: i32, cap: i32) -> i32;
    pub fn host_ui_draw(ops: i32, olen: i32) -> i32;
    pub fn host_hud_set(text_ptr: i32, text_len: i32) -> i32;
    pub fn host_surface_id() -> i32;
    pub fn host_now_ms() -> i64;
    pub fn host_log(p: i32, l: i32);
    pub fn host_sound_play(hz: i32, ms: i32) -> i32;
}

/// Reset the bump allocator. Called at the start of every host→guest call
/// cycle (the host invokes `chitti_alloc` exactly once, for the args buffer,
/// before each export call): the previous call's result has already been copied
/// out by then, and no app state lives on this heap (statics are plain
/// arrays/ints), so recycling is safe. This is what lets one wasm instance
/// stay alive for a whole app session without leaking its 96 KB heap.
pub fn heap_reset() {
    HEAP_OFF.store(0, Ordering::Relaxed);
}

pub fn chitti_alloc(size: i32) -> i32 {
    heap_reset();
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

pub fn pack(ptr: i32, len: i32) -> i64 {
    ((ptr as u64) << 32 | (len as u32 as u64)) as i64
}

pub fn result_string(s: &str) -> i64 {
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

pub fn unpack_args(ptr: i32, len: i32) -> &'static [u8] {
    if ptr < 0 || len < 0 {
        return b"";
    }
    unsafe { core::slice::from_raw_parts(ptr as *const u8, len as usize) }
}

pub fn json_str(blob: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\"");
    let i = blob.find(&pat)?;
    let rest = blob[i + pat.len()..].trim_start().strip_prefix(':')?.trim_start();
    if !rest.starts_with('"') {
        // number?
        let mut end = 0;
        for (j, c) in rest.char_indices() {
            if c.is_ascii_digit() || c == '-' {
                end = j + 1;
            } else {
                break;
            }
        }
        if end > 0 {
            return Some(rest[..end].to_string());
        }
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

pub fn json_i32(blob: &str, key: &str, default: i32) -> i32 {
    json_str(blob, key)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

pub fn ui_draw(ops: &str) -> i32 {
    unsafe { host_ui_draw(ops.as_ptr() as i32, ops.len() as i32) }
}

pub fn storage_set_durable(key: &str, val: &str) -> i32 {
    unsafe {
        host_storage_set(
            1,
            key.as_ptr() as i32,
            key.len() as i32,
            val.as_ptr() as i32,
            val.len() as i32,
        )
    }
}

pub fn storage_get_durable(key: &str, buf: &mut [u8]) -> i32 {
    unsafe {
        host_storage_get(
            1,
            key.as_ptr() as i32,
            key.len() as i32,
            buf.as_mut_ptr() as i32,
            buf.len() as i32,
        )
    }
}

pub fn storage_list_durable(buf: &mut [u8]) -> i32 {
    unsafe { host_storage_list(1, buf.as_mut_ptr() as i32, buf.len() as i32) }
}

pub fn storage_remove_durable(key: &str) -> i32 {
    unsafe { host_storage_remove(1, key.as_ptr() as i32, key.len() as i32) }
}

pub fn sound_play(hz: i32, ms: i32) -> i32 {
    unsafe { host_sound_play(hz, ms) }
}

pub fn now_ms() -> i64 {
    unsafe { host_now_ms() }
}

/// Optional HUD strip (status + shortcuts, '\\n'-separated). Available when the
/// host binds `host_hud_set` (same as chess-wasm).
pub fn hud_set(text: &str) -> i32 {
    let b = text.as_bytes();
    unsafe { host_hud_set(b.as_ptr() as i32, b.len() as i32) }
}

pub use result_string as export;
