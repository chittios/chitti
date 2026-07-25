//! Archive — pack/unpack named bundles of storage keys (`arc_<name>`).

use crate::guest::{
    hud_status, json_str, storage_get_durable, storage_list_durable, storage_set_durable, text_op,
    ui_draw,
};
use alloc::format;
use alloc::string::{String, ToString};

static mut N: usize = 0;
static mut SEL: i32 = 0;
static mut NAMES: [[u8; 32]; 16] = [[0; 32]; 16];
static mut NAME_LEN: [u8; 16] = [0; 16];

fn reload() {
    let mut buf = [0u8; 4096];
    let n = storage_list_durable(&mut buf);
    unsafe {
        N = 0;
        if n <= 0 {
            return;
        }
        let raw = core::str::from_utf8(&buf[..n as usize]).unwrap_or("");
        for line in raw.split('\n') {
            if let Some(name) = line.strip_prefix("arc_") {
                if N >= 16 {
                    break;
                }
                let b = name.as_bytes();
                let len = b.len().min(31);
                NAMES[N] = [0; 32];
                NAMES[N][..len].copy_from_slice(&b[..len]);
                NAME_LEN[N] = len as u8;
                N += 1;
            }
        }
        if SEL as usize >= N {
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
        &format!("Archive  {} bundles", unsafe { N }),
    );
    unsafe {
        if N == 0 {
            text_op(&mut ops, 20, 80, 12, "a8a4a0", "(no bundles)");
        }
        for i in 0..N.min(8) {
            let y = 24 + i as i32 * 18;
            let c = if i as i32 == SEL { "cc785c" } else { "5a5652" };
            ops.push_str(&format!("rect 12 {y} 232 16 {c}; "));
            let name = core::str::from_utf8(&NAMES[i][..NAME_LEN[i] as usize]).unwrap_or("?");
            text_op(&mut ops, 16, y + 2, 11, "e8e4df", name);
        }
    }
    ops.push_str("rect 0 176 256 16 3a3632; ");
    text_op(&mut ops, 8, 178, 10, "a8a4a0", "arrows select  r reload");
    ui_draw(&ops);
    hud_status(
        &format!("archive  {} bundles", unsafe { N }),
        "arrows select  r reload",
    );
}

pub fn start(_: &str) -> String {
    unsafe { SEL = 0 };
    reload();
    paint();
    format!("ok:archive {} bundle(s)", unsafe { N })
}

pub fn on_click(_x: i32, y: i32) -> String {
    let i = (y - 24) / 18;
    if i >= 0 && (i as usize) < unsafe { N } {
        unsafe { SEL = i };
        paint();
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
            if (SEL as usize) + 1 < N {
                SEL += 1;
            }
        },
        "r" => reload(),
        _ => {}
    }
    paint();
    status("")
}

pub fn pack(args: &str) -> String {
    let name = json_str(args, "name").unwrap_or_else(|| "bundle".into());
    let keys = json_str(args, "keys").unwrap_or_default();
    // body is keys joined; values concatenated as key=value\\n
    let mut body = String::new();
    for k in keys.split(|c| c == ',' || c == ' ') {
        let k = k.trim();
        if k.is_empty() {
            continue;
        }
        let mut buf = [0u8; 1024];
        let n = storage_get_durable(k, &mut buf);
        if n > 0 {
            body.push_str(k);
            body.push('=');
            body.push_str(core::str::from_utf8(&buf[..n as usize]).unwrap_or(""));
            body.push('\n');
        }
    }
    let sk = format!("arc_{name}");
    if storage_set_durable(&sk, &body) != 0 {
        return String::from("error:storage");
    }
    reload();
    paint();
    format!("ok:packed {name} ({} bytes)", body.len())
}

pub fn unpack(args: &str) -> String {
    let name = json_str(args, "name").unwrap_or_default();
    if name.is_empty() {
        return String::from("error: need name");
    }
    let sk = format!("arc_{name}");
    let mut buf = [0u8; 4096];
    let n = storage_get_durable(&sk, &mut buf);
    if n < 0 {
        return format!("error:no archive '{name}'");
    }
    let raw = core::str::from_utf8(&buf[..n as usize]).unwrap_or("");
    let mut count = 0;
    for line in raw.lines() {
        if let Some((k, v)) = line.split_once('=') {
            let _ = storage_set_durable(k, v);
            count += 1;
        }
    }
    format!("ok:unpacked {count} keys")
}

pub fn list(_: &str) -> String {
    reload();
    let mut out = String::new();
    unsafe {
        for i in 0..N {
            if i > 0 {
                out.push('\n');
            }
            if let Ok(s) = core::str::from_utf8(&NAMES[i][..NAME_LEN[i] as usize]) {
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

pub fn status(_: &str) -> String {
    format!("ok:archive n={}", unsafe { N })
}
