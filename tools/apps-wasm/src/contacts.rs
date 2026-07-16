//! Contacts — durable address book (`contact_<id>`).

use crate::guest::{
    json_str, storage_get_durable, storage_list_durable, storage_remove_durable,
    storage_set_durable, ui_draw,
};
use alloc::format;
use alloc::string::{String, ToString};

static mut KEYS: [[u8; 40]; 24] = [[0; 40]; 24];
static mut KEY_LEN: [u8; 24] = [0; 24];
static mut NKEYS: usize = 0;
static mut SEL: i32 = 0;

const PREFIX: &str = "contact_";

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

fn sel_name() -> String {
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

fn paint() {
    let mut ops = String::from("clear 1a1816; ");
    ops.push_str("rect 0 0 256 16 3a3632; ");
    ops.push_str("rect 8 24 240 144 2c2926; ");
    unsafe {
        for i in 0..NKEYS.min(8) {
            let y = 28 + i as i32 * 18;
            let c = if i as i32 == SEL { "cc785c" } else { "5a5652" };
            ops.push_str(&format!("rect 12 {y} 232 16 {c}; "));
            let w = (KEY_LEN[i] as i32 * 3).min(200);
            ops.push_str(&format!("rect 16 {} {w} 8 e8e4df; ", y + 4));
        }
    }
    ops.push_str("rect 0 176 256 16 3a3632; ");
    ui_draw(&ops);
}

pub fn start(_: &str) -> String {
    unsafe { SEL = 0 };
    reload();
    paint();
    format!("ok:contacts {} (↑↓ select, d delete)", unsafe { NKEYS })
}

pub fn on_click(_x: i32, y: i32) -> String {
    if y >= 28 {
        let i = (y - 28) / 18;
        if i >= 0 && (i as usize) < unsafe { NKEYS } {
            unsafe { SEL = i };
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
        },
        "down" => unsafe {
            if (SEL as usize) + 1 < NKEYS {
                SEL += 1;
            }
        },
        "d" => {
            let name = sel_name();
            if !name.is_empty() {
                let _ = storage_remove_durable(&format!("{PREFIX}{name}"));
                reload();
            }
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
                let sk = format!("{PREFIX}{s}");
                let mut buf = [0u8; 256];
                let n = storage_get_durable(&sk, &mut buf);
                if n > 0 {
                    out.push(' ');
                    out.push_str(core::str::from_utf8(&buf[..n as usize]).unwrap_or(""));
                }
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
    let name = json_str(args, "name")
        .or_else(|| json_str(args, "key"))
        .unwrap_or_default();
    let body = json_str(args, "body")
        .or_else(|| json_str(args, "email"))
        .or_else(|| json_str(args, "phone"))
        .unwrap_or_default();
    if name.is_empty() {
        return String::from("error: need name");
    }
    let sk = format!("{PREFIX}{name}");
    if storage_set_durable(&sk, &body) != 0 {
        return String::from("error:storage");
    }
    reload();
    paint();
    format!("ok:contact {name}")
}

pub fn get(args: &str) -> String {
    let name = json_str(args, "name")
        .or_else(|| json_str(args, "key"))
        .unwrap_or_default();
    let sk = format!("{PREFIX}{name}");
    let mut buf = [0u8; 512];
    let n = storage_get_durable(&sk, &mut buf);
    if n < 0 {
        return format!("error:no contact '{name}'");
    }
    String::from_utf8_lossy(&buf[..n as usize]).into_owned()
}

pub fn remove(args: &str) -> String {
    let name = json_str(args, "name")
        .or_else(|| json_str(args, "key"))
        .unwrap_or_default();
    let _ = storage_remove_durable(&format!("{PREFIX}{name}"));
    reload();
    paint();
    format!("ok:removed {name}")
}

pub fn status(_: &str) -> String {
    format!("ok:contacts n={} sel={}", unsafe { NKEYS }, sel_name())
}
