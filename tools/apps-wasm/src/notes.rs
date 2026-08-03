//! Notes — durable markdown notes with a package UI list/reader.

use crate::guest::{
    hud_status, json_str, storage_get_durable, storage_list_durable, storage_remove_durable,
    storage_set_durable, text_op, ui_draw,
};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

const PREFIX: &str = "note_";
static mut KEYS: [[u8; 40]; 32] = [[0; 40]; 32];
static mut KEY_LEN: [u8; 32] = [0; 32];
static mut NKEYS: usize = 0;
static mut SEL: i32 = 0;
static mut VIEW: u8 = 0; // 0 list 1 body
static mut BODY: [u8; 512] = [0; 512];
static mut BODY_LEN: usize = 0;
static mut BODY_SCROLL: u16 = 0;

fn key_ok(k: &str) -> bool {
    !k.is_empty()
        && k.len() <= 64
        && k.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'.')
}

fn reload_keys() {
    let mut buf = [0u8; 4096];
    let n = storage_list_durable(&mut buf);
    unsafe {
        NKEYS = 0;
        if n <= 0 {
            return;
        }
        let raw = core::str::from_utf8(&buf[..n as usize]).unwrap_or("");
        for line in raw.split('\n') {
            if !line.starts_with(PREFIX) || NKEYS >= 32 {
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

fn load_body() {
    unsafe {
        BODY_LEN = 0;
        BODY_SCROLL = 0;
        if NKEYS == 0 || SEL < 0 {
            return;
        }
        let i = SEL as usize;
        let name = core::str::from_utf8(&KEYS[i][..KEY_LEN[i] as usize]).unwrap_or("");
        let sk = format!("{PREFIX}{name}");
        let mut buf = [0u8; 512];
        let n = storage_get_durable(&sk, &mut buf);
        if n > 0 {
            let n = n as usize;
            BODY[..n].copy_from_slice(&buf[..n]);
            BODY_LEN = n;
        }
    }
}

fn paint() {
    reload_keys();
    let mut ops = String::from("clear 1a1816; ");
    ops.push_str("rect 0 0 256 16 3a3632; ");
    if unsafe { VIEW } == 0 {
        text_op(
            &mut ops,
            8,
            1,
            12,
            "e8e4df",
            &format!("{} Notes  {}", crate::fa::FILE_LINES, unsafe { NKEYS }),
        );
        unsafe {
            if NKEYS == 0 {
                text_op(&mut ops, 16, 40, 11, "a8a4a0", "(no notes — use notes_set)");
            }
            for i in 0..NKEYS.min(10) {
                let y = 24 + i as i32 * 14;
                let bg = if i as i32 == SEL { "3a3632" } else { "1a1816" };
                ops.push_str(&format!("rect 8 {y} 240 13 {bg}; "));
                let name = core::str::from_utf8(&KEYS[i][..KEY_LEN[i] as usize]).unwrap_or("?");
                let mark = if i as i32 == SEL { ">" } else { " " };
                text_op(
                    &mut ops,
                    12,
                    y + 1,
                    11,
                    "e8e4df",
                    &format!("{mark} {} {name}", crate::fa::FILE),
                );
            }
        }
        ops.push_str("rect 0 176 256 16 3a3632; ");
        text_op(&mut ops, 8, 178, 10, "a8a4a0", "up/dn select  enter open  d delete");
        hud_status(
            &format!("notes  {} key(s)", unsafe { NKEYS }),
            "up/dn  enter open  d delete",
        );
    } else {
        let title = unsafe {
            if NKEYS == 0 {
                String::from("Notes")
            } else {
                let i = SEL as usize;
                core::str::from_utf8(&KEYS[i][..KEY_LEN[i] as usize])
                    .unwrap_or("note")
                    .to_string()
            }
        };
        text_op(
            &mut ops,
            8,
            1,
            12,
            "e8e4df",
            &format!("{} {title}", crate::fa::FILE_LINES),
        );
        let body = unsafe {
            core::str::from_utf8(&BODY[..BODY_LEN]).unwrap_or("").to_string()
        };
        // Word-wrap-ish: fixed 36 cols, scroll by line.
        let cols = 36usize;
        let mut lines: Vec<&str> = Vec::new();
        for para in body.split('\n') {
            if para.is_empty() {
                lines.push("");
                continue;
            }
            let mut rest = para;
            while !rest.is_empty() {
                let take = rest.chars().take(cols).count().min(rest.len());
                // Find char boundary near take.
                let mut end = take.min(rest.len());
                while end > 0 && !rest.is_char_boundary(end) {
                    end -= 1;
                }
                if end == 0 {
                    end = rest.len();
                }
                lines.push(&rest[..end]);
                rest = &rest[end..];
            }
        }
        let scroll = unsafe { BODY_SCROLL as usize };
        let visible = 10usize;
        for (row, line) in lines.iter().skip(scroll).take(visible).enumerate() {
            let y = 24 + row as i32 * 14;
            text_op(&mut ops, 10, y, 10, "e8e4df", line);
        }
        ops.push_str("rect 0 176 256 16 3a3632; ");
        text_op(&mut ops, 8, 178, 10, "a8a4a0", "esc back  up/dn scroll");
        hud_status(&format!("note  {title}"), "esc list  up/dn scroll");
    }
    ui_draw(&ops);
}

pub fn start(_: &str) -> String {
    unsafe {
        VIEW = 0;
        SEL = 0;
    }
    paint();
    String::from("ok:notes ui")
}

pub fn list(_: &str) -> String {
    reload_keys();
    let mut keys: Vec<String> = Vec::new();
    unsafe {
        for i in 0..NKEYS {
            let name = core::str::from_utf8(&KEYS[i][..KEY_LEN[i] as usize]).unwrap_or("");
            keys.push(name.to_string());
        }
    }
    keys.sort();
    if keys.is_empty() {
        String::from("(empty)")
    } else {
        keys.join("\n")
    }
}

pub fn get(args: &str) -> String {
    let key = json_str(args, "key").unwrap_or_default();
    if !key_ok(&key) {
        return String::from("error: invalid key");
    }
    let sk = format!("{PREFIX}{key}");
    let mut buf = [0u8; 8192];
    let n = storage_get_durable(&sk, &mut buf);
    if n < 0 {
        return format!("error:no such note '{key}'");
    }
    String::from_utf8_lossy(&buf[..n as usize]).into_owned()
}

pub fn set(args: &str) -> String {
    let key = json_str(args, "key").unwrap_or_default();
    let body = json_str(args, "body")
        .or_else(|| json_str(args, "value"))
        .or_else(|| json_str(args, "content"))
        .unwrap_or_default();
    if !key_ok(&key) {
        return String::from("error: invalid key");
    }
    let sk = format!("{PREFIX}{key}");
    if storage_set_durable(&sk, &body) != 0 {
        return String::from("error:storage_set failed");
    }
    paint();
    format!("ok:note {key} ({} bytes)", body.len())
}

pub fn remove(args: &str) -> String {
    let key = json_str(args, "key").unwrap_or_default();
    if !key_ok(&key) {
        return String::from("error: invalid key");
    }
    let sk = format!("{PREFIX}{key}");
    let _ = storage_remove_durable(&sk);
    paint();
    format!("ok:removed {key}")
}

pub fn on_click(_x: i32, y: i32) -> String {
    if unsafe { VIEW } == 0 && y >= 24 {
        let i = ((y - 24) / 14) as i32;
        if i >= 0 && (i as usize) < unsafe { NKEYS } {
            unsafe {
                SEL = i;
            }
        }
    }
    paint();
    String::from("ok")
}

pub fn on_key(key: &str) -> String {
    unsafe {
        if VIEW == 0 {
            match key {
                "ArrowUp" | "up" | "k" => {
                    if SEL > 0 {
                        SEL -= 1;
                    }
                }
                "ArrowDown" | "down" | "j" => {
                    if (SEL as usize + 1) < NKEYS {
                        SEL += 1;
                    }
                }
                "Enter" | "\r" | " " => {
                    if NKEYS > 0 {
                        VIEW = 1;
                        load_body();
                    }
                }
                "d" | "Delete" => {
                    if NKEYS > 0 {
                        let i = SEL as usize;
                        let name =
                            core::str::from_utf8(&KEYS[i][..KEY_LEN[i] as usize]).unwrap_or("");
                        let sk = format!("{PREFIX}{name}");
                        let _ = storage_remove_durable(&sk);
                        if SEL > 0 {
                            SEL -= 1;
                        }
                    }
                }
                _ => {}
            }
        } else {
            match key {
                "Escape" | "esc" | "q" => {
                    VIEW = 0;
                }
                "ArrowUp" | "up" | "k" => {
                    BODY_SCROLL = BODY_SCROLL.saturating_sub(1);
                }
                "ArrowDown" | "down" | "j" => {
                    BODY_SCROLL = BODY_SCROLL.saturating_add(1);
                }
                _ => {}
            }
        }
    }
    paint();
    String::from("ok")
}
