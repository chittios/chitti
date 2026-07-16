//! Files — durable-storage browser (virtual FS under agent storage).

use crate::guest::{
    json_str, storage_get_durable, storage_list_durable, storage_remove_durable,
    storage_set_durable, ui_draw,
};
use alloc::format;
use alloc::string::{String, ToString};

const PREFIX: &str = "file_";
static mut SEL: i32 = 0;
static mut SCROLL: i32 = 0;
static mut KEYS: [[u8; 48]; 32] = [[0; 48]; 32];
static mut KEY_LEN: [u8; 32] = [0; 32];
static mut NKEYS: usize = 0;
static mut PREVIEW: [u8; 256] = [0; 256];
static mut PREVIEW_LEN: usize = 0;

fn key_ok(k: &str) -> bool {
    !k.is_empty()
        && k.len() <= 40
        && k.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'.' || b == b'/')
}

fn reload() {
    let mut buf = [0u8; 4096];
    let n = storage_list_durable(&mut buf);
    unsafe {
        NKEYS = 0;
        if n <= 0 {
            return;
        }
        let raw = core::str::from_utf8(&buf[..n as usize]).unwrap_or("");
        for line in raw.split('\n') {
            if !line.starts_with(PREFIX) {
                continue;
            }
            let name = &line[PREFIX.len()..];
            if name.is_empty() || NKEYS >= 32 {
                continue;
            }
            let b = name.as_bytes();
            let len = b.len().min(47);
            KEYS[NKEYS] = [0; 48];
            KEYS[NKEYS][..len].copy_from_slice(&b[..len]);
            KEY_LEN[NKEYS] = len as u8;
            NKEYS += 1;
        }
        if SEL as usize >= NKEYS {
            SEL = 0;
        }
    }
}

fn selected_name() -> String {
    unsafe {
        if NKEYS == 0 {
            return String::new();
        }
        let i = SEL as usize;
        core::str::from_utf8(&KEYS[i][..KEY_LEN[i] as usize])
            .unwrap_or("")
            .to_string()
    }
}

fn load_preview() {
    let name = selected_name();
    unsafe {
        PREVIEW_LEN = 0;
        PREVIEW = [0; 256];
    }
    if name.is_empty() {
        return;
    }
    let sk = format!("{PREFIX}{name}");
    let mut buf = [0u8; 256];
    let n = storage_get_durable(&sk, &mut buf);
    if n > 0 {
        let len = (n as usize).min(255);
        unsafe {
            PREVIEW[..len].copy_from_slice(&buf[..len]);
            PREVIEW_LEN = len;
        }
    }
}

fn paint() {
    let mut ops = String::from("clear 1a1816; ");
    ops.push_str("rect 0 0 256 16 3a3632; "); // title bar
    ops.push_str("rect 0 16 120 160 2c2926; "); // list
    ops.push_str("rect 124 16 128 160 2c2926; "); // preview
    unsafe {
        let start = SCROLL as usize;
        for i in 0..8 {
            let idx = start + i;
            if idx >= NKEYS {
                break;
            }
            let y = 20 + i as i32 * 18;
            let c = if idx as i32 == SEL { "cc785c" } else { "5a5652" };
            ops.push_str(&format!("rect 4 {y} 112 16 {c}; "));
            // Length bar as name stand-in.
            let w = (KEY_LEN[idx] as i32 * 2).min(100);
            ops.push_str(&format!("rect 8 {} {w} 8 e8e4df; ", y + 4));
        }
        if PREVIEW_LEN > 0 {
            let w = (PREVIEW_LEN as i32 / 2).min(120);
            ops.push_str(&format!("rect 128 24 {w} 12 5a8f5a; "));
        }
    }
    ops.push_str("rect 0 176 256 16 3a3632; ");
    ui_draw(&ops);
}

pub fn start(_: &str) -> String {
    unsafe {
        SEL = 0;
        SCROLL = 0;
    }
    reload();
    load_preview();
    paint();
    format!("ok:files {} entries (↑↓ select, enter open, d delete)", unsafe { NKEYS })
}

pub fn on_click(x: i32, y: i32) -> String {
    if x < 120 && y >= 16 && y < 176 {
        let i = ((y - 20) / 18) + unsafe { SCROLL };
        if i >= 0 && (i as usize) < unsafe { NKEYS } {
            unsafe { SEL = i };
            load_preview();
            paint();
        }
    }
    status("")
}

pub fn on_key(key: &str) -> String {
    match key {
        "up" => unsafe {
            if SEL > 0 {
                SEL -= 1;
            }
            if SEL < SCROLL {
                SCROLL = SEL;
            }
            load_preview();
        },
        "down" => unsafe {
            if (SEL as usize) + 1 < NKEYS {
                SEL += 1;
            }
            if SEL >= SCROLL + 8 {
                SCROLL = SEL - 7;
            }
            load_preview();
        },
        "enter" | "space" => load_preview(),
        "d" | "delete" => {
            let name = selected_name();
            if !name.is_empty() {
                let sk = format!("{PREFIX}{name}");
                let _ = storage_remove_durable(&sk);
                reload();
                load_preview();
            }
        },
        "r" => {
            reload();
            load_preview();
        }
        _ => {}
    }
    paint();
    status("")
}

pub fn list(_: &str) -> String {
    reload();
    let mut out = String::new();
    unsafe {
        for i in 0..NKEYS {
            if i > 0 {
                out.push('\n');
            }
            if let Ok(s) = core::str::from_utf8(&KEYS[i][..KEY_LEN[i] as usize]) {
                out.push_str(s);
            }
        }
    }
    if out.is_empty() {
        String::from("(empty)")
    } else {
        out
    }
}

pub fn get(args: &str) -> String {
    let key = json_str(args, "key")
        .or_else(|| json_str(args, "path"))
        .unwrap_or_default();
    if !key_ok(&key) {
        return String::from("error: invalid key");
    }
    let sk = format!("{PREFIX}{key}");
    let mut buf = [0u8; 8192];
    let n = storage_get_durable(&sk, &mut buf);
    if n < 0 {
        return format!("error:no such file '{key}'");
    }
    String::from_utf8_lossy(&buf[..n as usize]).into_owned()
}

pub fn set(args: &str) -> String {
    let key = json_str(args, "key")
        .or_else(|| json_str(args, "path"))
        .unwrap_or_default();
    let body = json_str(args, "body")
        .or_else(|| json_str(args, "content"))
        .unwrap_or_default();
    if !key_ok(&key) {
        return String::from("error: invalid key");
    }
    let sk = format!("{PREFIX}{key}");
    if storage_set_durable(&sk, &body) != 0 {
        return String::from("error:storage_set failed");
    }
    reload();
    paint();
    format!("ok:file {key} ({} bytes)", body.len())
}

pub fn remove(args: &str) -> String {
    let key = json_str(args, "key")
        .or_else(|| json_str(args, "path"))
        .unwrap_or_default();
    if !key_ok(&key) {
        return String::from("error: invalid key");
    }
    let sk = format!("{PREFIX}{key}");
    let _ = storage_remove_durable(&sk);
    reload();
    paint();
    format!("ok:removed {key}")
}

pub fn status(_: &str) -> String {
    let name = selected_name();
    format!("ok:files n={} sel={}", unsafe { NKEYS }, name)
}
