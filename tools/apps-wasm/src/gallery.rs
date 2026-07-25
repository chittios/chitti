//! Gallery — thumbnail grid over durable `img_*` storage keys.

use crate::guest::{
    hud_status, json_str, storage_get_durable, storage_list_durable, storage_set_durable, text_op,
    ui_draw,
};
use alloc::format;
use alloc::string::{String, ToString};

const PREFIX: &str = "img_";
static mut KEYS: [[u8; 40]; 24] = [[0; 40]; 24];
static mut KEY_LEN: [u8; 24] = [0; 24];
static mut NKEYS: usize = 0;
static mut SEL: i32 = 0;

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
            if !line.starts_with(PREFIX) || NKEYS >= 24 {
                continue;
            }
            let name = &line[PREFIX.len()..];
            let b = name.as_bytes();
            let len = b.len().min(39);
            KEYS[NKEYS] = [0; 40];
            KEYS[NKEYS][..len].copy_from_slice(&b[..len]);
            KEY_LEN[NKEYS] = len as u8;
            NKEYS += 1;
        }
        if SEL as usize >= NKEYS {
            SEL = 0;
        }
    }
}

fn paint() {
    let mut ops = String::from("clear 1a1816; ");
    ops.push_str("rect 0 0 256 16 3a3632; ");
    text_op(
        &mut ops,
        8,
        1,
        12,
        "e8e4df",
        &format!("Gallery  {}", unsafe { NKEYS }),
    );
    // 4×3 grid of "thumbnails" (color from key hash).
    unsafe {
        for i in 0..12usize {
            let col = (i % 4) as i32;
            let row = (i / 4) as i32;
            let x = 8 + col * 62;
            let y = 24 + row * 52;
            if i < NKEYS {
                let h = KEY_LEN[i] as u32;
                let c = match h % 5 {
                    0 => "5a8f5a",
                    1 => "6688cc",
                    2 => "cc785c",
                    3 => "8a5a4a",
                    _ => "6a5a7a",
                };
                let border = if i as i32 == SEL { "e8e4df" } else { c };
                ops.push_str(&format!("rect {x} {y} 56 44 {border}; "));
                ops.push_str(&format!("rect {} {} 48 36 {c}; ", x + 4, y + 4));
                let name = core::str::from_utf8(&KEYS[i][..KEY_LEN[i] as usize]).unwrap_or("?");
                let shown = if name.len() > 8 { &name[..8] } else { name };
                text_op(&mut ops, x + 4, y + 28, 9, "e8e4df", shown);
            } else {
                ops.push_str(&format!("rect {x} {y} 56 44 2c2926; "));
            }
        }
    }
    ops.push_str("rect 0 176 256 16 3a3632; ");
    text_op(&mut ops, 8, 178, 10, "a8a4a0", "arrows select  a add demo");
    ui_draw(&ops);
    hud_status(
        &format!("gallery  {} images", unsafe { NKEYS }),
        "arrows select  a add demo",
    );
}

pub fn start(_: &str) -> String {
    unsafe { SEL = 0 };
    reload();
    paint();
    format!("ok:gallery {} image(s) (arrows select; a add demo)", unsafe { NKEYS })
}

pub fn on_click(x: i32, y: i32) -> String {
    if y < 24 || y >= 176 {
        return status("");
    }
    let col = ((x - 8) / 62).clamp(0, 3);
    let row = ((y - 24) / 52).clamp(0, 2);
    let i = row * 4 + col;
    if (i as usize) < unsafe { NKEYS } {
        unsafe { SEL = i };
        paint();
    }
    status("")
}

pub fn on_key(key: &str) -> String {
    match key {
        "left" => unsafe {
            if SEL > 0 {
                SEL -= 1;
            }
        },
        "right" => unsafe {
            if (SEL as usize) + 1 < NKEYS {
                SEL += 1;
            }
        },
        "up" => unsafe {
            SEL = (SEL - 4).max(0);
        },
        "down" => unsafe {
            if (SEL as usize) + 4 < NKEYS {
                SEL += 4;
            }
        },
        "a" => {
            // Seed a demo slot.
            let n = unsafe { NKEYS };
            let key = format!("demo{}", n);
            let sk = format!("{PREFIX}{key}");
            let _ = storage_set_durable(&sk, "placeholder-rgb");
            reload();
        }
        "r" => reload(),
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

pub fn set(args: &str) -> String {
    let key = json_str(args, "key").unwrap_or_default();
    let body = json_str(args, "body")
        .or_else(|| json_str(args, "meta"))
        .unwrap_or_else(|| "image".into());
    if key.is_empty() {
        return String::from("error: need key");
    }
    let sk = format!("{PREFIX}{key}");
    if storage_set_durable(&sk, &body) != 0 {
        return String::from("error:storage");
    }
    reload();
    paint();
    format!("ok:gallery {key}")
}

pub fn get(args: &str) -> String {
    let key = json_str(args, "key").unwrap_or_default();
    let sk = format!("{PREFIX}{key}");
    let mut buf = [0u8; 1024];
    let n = storage_get_durable(&sk, &mut buf);
    if n < 0 {
        return format!("error:no img '{key}'");
    }
    String::from_utf8_lossy(&buf[..n as usize]).into_owned()
}

pub fn status(_: &str) -> String {
    format!("ok:gallery n={} sel={}", unsafe { NKEYS }, unsafe { SEL })
}
