//! Hex — dump durable storage value as hex grid.

use crate::guest::{hud_status, json_str, storage_get_durable, storage_set_durable, text_op, ui_draw};
use alloc::format;
use alloc::string::{String, ToString};

static mut DATA: [u8; 256] = [0; 256];
static mut LEN: usize = 0;
static mut OFF: usize = 0;
static mut KEY: [u8; 40] = [0; 40];
static mut KEY_LEN: usize = 0;

fn paint() {
    let mut ops = String::from("clear 1a1816; ");
    ops.push_str("rect 0 0 256 16 3a3632; ");
    let key = unsafe {
        core::str::from_utf8(&KEY[..KEY_LEN])
            .unwrap_or("hex")
            .to_string()
    };
    text_op(
        &mut ops,
        8,
        1,
        11,
        "e8e4df",
        &format!("Hex  {key}  {}b", unsafe { LEN }),
    );
    unsafe {
        let start = OFF;
        let rows = 12usize;
        for r in 0..rows {
            let base = start + r * 8;
            if base >= LEN {
                break;
            }
            let y = 22 + r as i32 * 12;
            // Offset label.
            text_op(&mut ops, 4, y, 9, "a8a4a0", &format!("{base:02x}"));
            let mut hexline = String::new();
            let mut ascii = String::new();
            for c in 0..8usize {
                if base + c >= LEN {
                    break;
                }
                let b = DATA[base + c];
                hexline.push_str(&format!("{b:02x} "));
                if b.is_ascii_graphic() || b == b' ' {
                    ascii.push(b as char);
                } else {
                    ascii.push('.');
                }
            }
            text_op(&mut ops, 28, y, 9, "e8e4df", &hexline);
            text_op(&mut ops, 200, y, 9, "cc785c", &ascii);
        }
    }
    ops.push_str("rect 0 176 256 16 3a3632; ");
    text_op(&mut ops, 8, 178, 10, "a8a4a0", "pgup/pgdn scroll");
    ui_draw(&ops);
    hud_status(
        &format!("hex  {key}  {} bytes  off={}", unsafe { LEN }, unsafe { OFF }),
        "pgup/pgdn scroll",
    );
}

fn load_key(k: &str) {
    let b = k.as_bytes();
    let n = b.len().min(39);
    unsafe {
        KEY = [0; 40];
        KEY[..n].copy_from_slice(&b[..n]);
        KEY_LEN = n;
        LEN = 0;
        OFF = 0;
        DATA = [0; 256];
    }
    let mut buf = [0u8; 256];
    let got = storage_get_durable(k, &mut buf);
    if got > 0 {
        let len = (got as usize).min(256);
        unsafe {
            DATA[..len].copy_from_slice(&buf[..len]);
            LEN = len;
        }
    }
}

pub fn start(_: &str) -> String {
    load_key("hex_demo");
    if unsafe { LEN } == 0 {
        let _ = storage_set_durable("hex_demo", "Hello ChittiOS hex!");
        load_key("hex_demo");
    }
    paint();
    format!("ok:hex {} bytes (pgup/pgdn scroll; o open key via tool)", unsafe {
        LEN
    })
}

pub fn on_click(_x: i32, _y: i32) -> String {
    status("")
}

pub fn on_key(key: &str) -> String {
    match key {
        "up" | "pageup" => unsafe {
            OFF = OFF.saturating_sub(16);
        },
        "down" | "pagedown" => unsafe {
            if OFF + 16 < LEN {
                OFF += 16;
            }
        },
        _ => {}
    }
    paint();
    status("")
}

pub fn open(args: &str) -> String {
    let key = json_str(args, "key")
        .or_else(|| json_str(args, "path"))
        .unwrap_or_default();
    if key.is_empty() {
        return String::from("error: need key");
    }
    load_key(&key);
    paint();
    format!("ok:hex open {key} ({} bytes)", unsafe { LEN })
}

pub fn dump(args: &str) -> String {
    let key = json_str(args, "key").unwrap_or_else(|| {
        unsafe {
            core::str::from_utf8(&KEY[..KEY_LEN])
                .unwrap_or("hex_demo")
                .to_string()
        }
    });
    load_key(&key);
    let mut out = String::new();
    unsafe {
        for (i, b) in DATA[..LEN].iter().enumerate() {
            if i > 0 && i % 16 == 0 {
                out.push('\n');
            } else if i > 0 {
                out.push(' ');
            }
            // hex pair
            const HEX: &[u8; 16] = b"0123456789abcdef";
            out.push(HEX[(b >> 4) as usize] as char);
            out.push(HEX[(b & 0xf) as usize] as char);
        }
    }
    if out.is_empty() {
        String::from("(empty)")
    } else {
        out
    }
}

pub fn status(_: &str) -> String {
    format!("ok:hex len={} off={}", unsafe { LEN }, unsafe { OFF })
}
