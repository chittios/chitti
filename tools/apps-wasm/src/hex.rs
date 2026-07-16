//! Hex — dump durable storage value as hex grid.

use crate::guest::{json_str, storage_get_durable, storage_set_durable, ui_draw};
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
    unsafe {
        let start = OFF;
        let rows = 14usize;
        for r in 0..rows {
            let base = start + r * 16;
            if base >= LEN {
                break;
            }
            let y = 20 + r as i32 * 11;
            for c in 0..16usize {
                if base + c >= LEN {
                    break;
                }
                let b = DATA[base + c];
                let x = 8 + c as i32 * 15;
                // Nibble intensity as color bands.
                let cname = match b >> 6 {
                    0 => "3a3632",
                    1 => "5a5652",
                    2 => "8a8a6a",
                    _ => "cc785c",
                };
                ops.push_str(&format!("rect {x} {y} 12 9 {cname}; "));
            }
        }
    }
    ops.push_str("rect 0 176 256 16 3a3632; ");
    ui_draw(&ops);
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
