//! Settings — panel UI; persists prefs to durable storage (agent explains shell cmds).

use crate::guest::{json_str, storage_get_durable, storage_set_durable, ui_draw};
use alloc::format;
use alloc::string::{String, ToString};

// panels: 0 theme 1 network 2 model 3 voice 4 privacy
static mut PANEL: u8 = 0;
static mut THEME: u8 = 0; // 0 dark 1 light 2 nord
static mut OPACITY: u8 = 255;
static mut MODE: u8 = 0; // 0 manual 1 auto 2 bypass

fn paint() {
    let mut ops = String::from("clear 1a1816; ");
    ops.push_str("rect 0 0 256 16 3a3632; ");
    // Side nav.
    for i in 0..5i32 {
        let y = 24 + i * 28;
        let c = if unsafe { PANEL } as i32 == i {
            "cc785c"
        } else {
            "3a3632"
        };
        ops.push_str(&format!("rect 8 {y} 64 24 {c}; "));
    }
    // Content.
    ops.push_str("rect 80 24 168 144 2c2926; ");
    match unsafe { PANEL } {
        0 => {
            for i in 0..3i32 {
                let x = 96 + i * 48;
                let c = if unsafe { THEME } as i32 == i {
                    "e8e4df"
                } else {
                    "5a5652"
                };
                ops.push_str(&format!("rect {x} 48 40 40 {c}; "));
            }
            let ow = (unsafe { OPACITY } as i32 / 2).min(160);
            ops.push_str(&format!("rect 96 110 {ow} 12 6688cc; "));
        }
        1 => {
            ops.push_str("rect 96 48 136 20 5a8f5a; ");
            ops.push_str("rect 96 80 100 12 5a5652; ");
        }
        2 => {
            ops.push_str("rect 96 48 136 48 4a6a8a; ");
        }
        3 => {
            ops.push_str("rect 96 48 136 24 8a5a4a; ");
        }
        _ => {
            for i in 0..3i32 {
                let y = 48 + i * 28;
                let c = if unsafe { MODE } as i32 == i {
                    "cc785c"
                } else {
                    "5a5652"
                };
                ops.push_str(&format!("rect 96 {y} 120 20 {c}; "));
            }
        }
    }
    ops.push_str("rect 0 176 256 16 3a3632; ");
    ui_draw(&ops);
}

fn save() {
    let blob = format!(
        "{{\"theme\":{},\"opacity\":{},\"mode\":{},\"panel\":{}}}",
        unsafe { THEME },
        unsafe { OPACITY },
        unsafe { MODE },
        unsafe { PANEL }
    );
    let _ = storage_set_durable("settings", &blob);
}

fn load() {
    let mut buf = [0u8; 128];
    let n = storage_get_durable("settings", &mut buf);
    if n <= 0 {
        return;
    }
    let raw = core::str::from_utf8(&buf[..n as usize]).unwrap_or("");
    if let Some(t) = json_str(raw, "theme") {
        unsafe { THEME = t.parse().unwrap_or(0) };
    }
    if let Some(o) = json_str(raw, "opacity") {
        unsafe { OPACITY = o.parse().unwrap_or(255) };
    }
    if let Some(m) = json_str(raw, "mode") {
        unsafe { MODE = m.parse().unwrap_or(0) };
    }
}

pub fn start(_: &str) -> String {
    load();
    paint();
    String::from("ok:settings (1-5 panels; arrows; theme 1-3; mode keys m/a/b)")
}

pub fn on_click(x: i32, y: i32) -> String {
    if x < 72 {
        let p = ((y - 24) / 28).clamp(0, 4) as u8;
        unsafe { PANEL = p };
        paint();
        return status("");
    }
    if unsafe { PANEL } == 0 && y >= 48 && y < 88 {
        let t = ((x - 96) / 48).clamp(0, 2) as u8;
        unsafe { THEME = t };
        save();
    }
    if unsafe { PANEL } == 4 {
        let m = ((y - 48) / 28).clamp(0, 2) as u8;
        unsafe { MODE = m };
        save();
    }
    paint();
    status("")
}

pub fn on_key(key: &str) -> String {
    match key {
        "1" | "2" | "3" | "4" | "5" => {
            unsafe { PANEL = (key.as_bytes()[0] - b'1') };
        }
        "left" => unsafe {
            if PANEL == 0 {
                THEME = THEME.saturating_sub(1);
            }
        },
        "right" => unsafe {
            if PANEL == 0 {
                THEME = (THEME + 1).min(2);
            }
        },
        "+" | "=" => unsafe {
            OPACITY = OPACITY.saturating_add(16).min(255);
        },
        "-" | "_" => unsafe {
            OPACITY = OPACITY.saturating_sub(16);
        },
        "m" => unsafe { MODE = 0 },
        "a" => unsafe { MODE = 1 },
        "b" => unsafe { MODE = 2 },
        _ => {}
    }
    save();
    paint();
    status("")
}

pub fn get(_: &str) -> String {
    load();
    format!(
        "ok:theme={} opacity={} mode={} (0 manual 1 auto 2 bypass)",
        unsafe { THEME },
        unsafe { OPACITY },
        unsafe { MODE }
    )
}

pub fn set(args: &str) -> String {
    if let Some(t) = json_str(args, "theme") {
        unsafe { THEME = t.parse().unwrap_or(THEME) };
    }
    if let Some(o) = json_str(args, "opacity") {
        unsafe { OPACITY = o.parse().unwrap_or(OPACITY) };
    }
    if let Some(m) = json_str(args, "mode") {
        unsafe { MODE = m.parse().unwrap_or(MODE) };
    }
    save();
    paint();
    get("")
}

pub fn status(_: &str) -> String {
    format!(
        "ok:settings panel={} theme={} mode={}",
        unsafe { PANEL },
        unsafe { THEME },
        unsafe { MODE }
    )
}
