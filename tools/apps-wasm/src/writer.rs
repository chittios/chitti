//! Writer — long-form document in durable storage (`doc_main`).

use crate::guest::{json_str, storage_get_durable, storage_set_durable, ui_draw};
use alloc::format;
use alloc::string::{String, ToString};

static mut DOC: [u8; 2048] = [0; 2048];
static mut DOC_LEN: usize = 0;
static mut CURSOR: usize = 0;
static mut SCROLL: usize = 0;

fn load() {
    let mut buf = [0u8; 2048];
    let n = storage_get_durable("doc_main", &mut buf);
    unsafe {
        DOC = [0; 2048];
        DOC_LEN = 0;
        CURSOR = 0;
        SCROLL = 0;
        if n > 0 {
            let len = (n as usize).min(2047);
            DOC[..len].copy_from_slice(&buf[..len]);
            DOC_LEN = len;
            CURSOR = len;
        }
    }
}

fn save() {
    let s = unsafe {
        core::str::from_utf8(&DOC[..DOC_LEN])
            .unwrap_or("")
            .to_string()
    };
    let _ = storage_set_durable("doc_main", &s);
}

fn paint() {
    let mut ops = String::from("clear 1a1816; ");
    ops.push_str("rect 0 0 256 16 3a3632; ");
    ops.push_str("rect 8 24 240 148 2c2926; ");
    // Represent text as wrapped bars by line.
    unsafe {
        let text = core::str::from_utf8(&DOC[..DOC_LEN]).unwrap_or("");
        let mut line = 0i32;
        let mut col = 0i32;
        let skip = SCROLL as i32;
        let mut lineno = 0i32;
        for ch in text.chars() {
            if ch == '\n' {
                lineno += 1;
                col = 0;
                continue;
            }
            if lineno < skip {
                continue;
            }
            line = lineno - skip;
            if line >= 12 {
                break;
            }
            let x = 12 + col * 6;
            let y = 28 + line * 12;
            if x < 240 {
                ops.push_str(&format!("rect {x} {y} 4 8 e8e4df; "));
            }
            col += 1;
            if col > 36 {
                lineno += 1;
                col = 0;
            }
        }
        // Cursor.
        let cy = 28 + ((CURSOR.min(DOC_LEN) as i32 / 36) - skip).max(0) * 12;
        ops.push_str(&format!("rect 12 {cy} 2 10 cc785c; "));
    }
    ops.push_str("rect 0 176 256 16 3a3632; ");
    ui_draw(&ops);
}

pub fn start(_: &str) -> String {
    load();
    paint();
    format!("ok:writer {} bytes (type, backspace, enter; s save)", unsafe {
        DOC_LEN
    })
}

pub fn on_click(_x: i32, _y: i32) -> String {
    status("")
}

pub fn on_key(key: &str) -> String {
    match key {
        "backspace" => unsafe {
            if CURSOR > 0 && DOC_LEN > 0 {
                // shift left
                for i in CURSOR - 1..DOC_LEN - 1 {
                    DOC[i] = DOC[i + 1];
                }
                DOC_LEN -= 1;
                CURSOR -= 1;
                DOC[DOC_LEN] = 0;
            }
        },
        "enter" => unsafe {
            if DOC_LEN < 2047 {
                for i in (CURSOR..DOC_LEN).rev() {
                    DOC[i + 1] = DOC[i];
                }
                DOC[CURSOR] = b'\n';
                DOC_LEN += 1;
                CURSOR += 1;
            }
        },
        "left" => unsafe {
            CURSOR = CURSOR.saturating_sub(1);
        },
        "right" => unsafe {
            if CURSOR < DOC_LEN {
                CURSOR += 1;
            }
        },
        "up" => unsafe {
            SCROLL = SCROLL.saturating_sub(1);
        },
        "down" => unsafe {
            SCROLL += 1;
        },
        "s" => save(),
        k if k.len() == 1 => {
            let ch = k.chars().next().unwrap();
            if ch.is_ascii() && !ch.is_control() {
                unsafe {
                    if DOC_LEN < 2047 {
                        for i in (CURSOR..DOC_LEN).rev() {
                            DOC[i + 1] = DOC[i];
                        }
                        DOC[CURSOR] = ch as u8;
                        DOC_LEN += 1;
                        CURSOR += 1;
                    }
                }
            }
        }
        _ => {}
    }
    paint();
    status("")
}

pub fn get(_: &str) -> String {
    load();
    unsafe {
        core::str::from_utf8(&DOC[..DOC_LEN])
            .unwrap_or("")
            .to_string()
    }
}

pub fn set(args: &str) -> String {
    let body = json_str(args, "body")
        .or_else(|| json_str(args, "content"))
        .unwrap_or_default();
    let b = body.as_bytes();
    let n = b.len().min(2047);
    unsafe {
        DOC = [0; 2048];
        DOC[..n].copy_from_slice(&b[..n]);
        DOC_LEN = n;
        CURSOR = n;
    }
    save();
    paint();
    format!("ok:writer {} bytes", n)
}

pub fn status(_: &str) -> String {
    format!("ok:writer len={} cursor={}", unsafe { DOC_LEN }, unsafe {
        CURSOR
    })
}
