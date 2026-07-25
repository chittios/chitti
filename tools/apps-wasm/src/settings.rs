//! Settings — panel UI with real text labels; persists prefs to durable storage.

use crate::guest::{
    hud_status, json_str, storage_get_durable, storage_set_durable, text_op, ui_draw,
};
use alloc::format;
use alloc::string::{String, ToString};

// panels: 0 theme 1 network 2 model 3 voice 4 privacy
static mut PANEL: u8 = 0;
static mut THEME: u8 = 0; // 0 dark 1 light 2 nord
static mut OPACITY: u8 = 255;
static mut MODE: u8 = 0; // 0 manual 1 auto 2 bypass

const NAV: [&str; 5] = ["Theme", "Net", "Model", "Voice", "Mode"];
const THEMES: [&str; 3] = ["Dark", "Light", "Nord"];
const MODES: [&str; 3] = ["Manual", "Auto", "Bypass"];

fn paint() {
    let panel = unsafe { PANEL };
    let theme = unsafe { THEME };
    let opacity = unsafe { OPACITY };
    let mode = unsafe { MODE };

    let mut ops = String::from("clear 1a1816; ");
    // Title bar.
    ops.push_str("rect 0 0 256 18 3a3632; ");
    text_op(&mut ops, 8, 2, 12, "e8e4df", "Settings");
    // Side nav.
    for i in 0..5i32 {
        let y = 24 + i * 28;
        let c = if panel as i32 == i {
            "cc785c"
        } else {
            "3a3632"
        };
        ops.push_str(&format!("rect 8 {y} 64 24 {c}; "));
        text_op(&mut ops, 14, y + 4, 11, "e8e4df", NAV[i as usize]);
    }
    // Content panel.
    ops.push_str("rect 80 24 168 144 2c2926; ");
    match panel {
        0 => {
            text_op(&mut ops, 96, 32, 12, "cc785c", "Theme");
            for i in 0..3i32 {
                let x = 96 + i * 48;
                let c = if theme as i32 == i {
                    "e8e4df"
                } else {
                    "5a5652"
                };
                ops.push_str(&format!("rect {x} 52 40 28 {c}; "));
                let tc = if theme as i32 == i { "1a1816" } else { "e8e4df" };
                text_op(&mut ops, x + 4, 58, 10, tc, THEMES[i as usize]);
            }
            text_op(&mut ops, 96, 96, 11, "e8e4df", "Opacity");
            let ow = (opacity as i32 / 2).min(160).max(4);
            ops.push_str(&format!("rect 96 116 {ow} 12 6688cc; "));
            text_op(
                &mut ops,
                96,
                134,
                10,
                "a8a4a0",
                &format!("{opacity}  (+/-)"),
            );
        }
        1 => {
            text_op(&mut ops, 96, 32, 12, "cc785c", "Network");
            ops.push_str("rect 96 56 136 20 5a8f5a; ");
            text_op(&mut ops, 104, 58, 11, "e8e4df", "DHCP / status");
            text_op(&mut ops, 96, 88, 10, "a8a4a0", "Use /network in shell");
            text_op(&mut ops, 96, 108, 10, "a8a4a0", "for full config");
        }
        2 => {
            text_op(&mut ops, 96, 32, 12, "cc785c", "Model");
            ops.push_str("rect 96 56 136 40 4a6a8a; ");
            text_op(&mut ops, 104, 62, 11, "e8e4df", "Local GGUF");
            text_op(&mut ops, 104, 78, 10, "e8e4df", "or /model remote");
            text_op(&mut ops, 96, 110, 10, "a8a4a0", "Shell: /model load");
        }
        3 => {
            text_op(&mut ops, 96, 32, 12, "cc785c", "Voice");
            ops.push_str("rect 96 56 136 24 8a5a4a; ");
            text_op(&mut ops, 104, 60, 11, "e8e4df", "STT / TTS");
            text_op(&mut ops, 96, 96, 10, "a8a4a0", "/voice models");
            text_op(&mut ops, 96, 112, 10, "a8a4a0", "/voice remote …");
        }
        _ => {
            text_op(&mut ops, 96, 32, 12, "cc785c", "Approval mode");
            for i in 0..3i32 {
                let y = 56 + i * 28;
                let c = if mode as i32 == i {
                    "cc785c"
                } else {
                    "5a5652"
                };
                ops.push_str(&format!("rect 96 {y} 120 22 {c}; "));
                text_op(&mut ops, 108, y + 3, 12, "e8e4df", MODES[i as usize]);
            }
            text_op(&mut ops, 96, 148, 9, "a8a4a0", "keys m/a/b");
        }
    }
    ops.push_str("rect 0 176 256 16 3a3632; ");
    text_op(
        &mut ops,
        8,
        178,
        10,
        "a8a4a0",
        "1-5 panels  arrows theme  +/- opacity",
    );
    ui_draw(&ops);
    hud_status(
        &format!(
            "settings  {}  theme={}  mode={}  opacity={}",
            NAV[panel as usize],
            THEMES[theme.min(2) as usize],
            MODES[mode.min(2) as usize],
            opacity
        ),
        "1-5 panels  arrows theme  +/- opacity  m/a/b mode",
    );
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
        unsafe { THEME = t.parse().unwrap_or(0).min(2) };
    }
    if let Some(o) = json_str(raw, "opacity") {
        unsafe { OPACITY = o.parse().unwrap_or(255) };
    }
    if let Some(m) = json_str(raw, "mode") {
        unsafe { MODE = m.parse().unwrap_or(0).min(2) };
    }
}

pub fn start(_: &str) -> String {
    load();
    paint();
    String::from("ok:settings (1-5 panels; arrows; theme 1-3; mode keys m/a/b)")
}

pub fn on_click(x: i32, y: i32) -> String {
    if x < 72 && y >= 24 && y < 24 + 5 * 28 {
        let p = ((y - 24) / 28).clamp(0, 4) as u8;
        unsafe { PANEL = p };
        paint();
        return status("");
    }
    if unsafe { PANEL } == 0 && y >= 52 && y < 80 && x >= 96 {
        let t = ((x - 96) / 48).clamp(0, 2) as u8;
        unsafe { THEME = t };
        save();
    }
    if unsafe { PANEL } == 4 && y >= 56 && x >= 96 {
        let m = ((y - 56) / 28).clamp(0, 2) as u8;
        unsafe { MODE = m };
        save();
    }
    paint();
    status("")
}

pub fn on_key(key: &str) -> String {
    let mut changed = true;
    match key {
        "1" | "2" | "3" | "4" | "5" => {
            unsafe { PANEL = (key.as_bytes()[0] - b'1') };
        }
        "left" | "h" => unsafe {
            if PANEL == 0 {
                THEME = THEME.saturating_sub(1);
            } else {
                PANEL = PANEL.saturating_sub(1);
            }
        },
        "right" | "l" => unsafe {
            if PANEL == 0 {
                THEME = (THEME + 1).min(2);
            } else {
                PANEL = (PANEL + 1).min(4);
            }
        },
        "up" | "k" => unsafe {
            PANEL = PANEL.saturating_sub(1);
        },
        "down" | "j" => unsafe {
            PANEL = (PANEL + 1).min(4);
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
        _ => changed = false,
    }
    if !changed {
        return String::from("ok");
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
        unsafe { THEME = t.parse().unwrap_or(THEME).min(2) };
    }
    if let Some(o) = json_str(args, "opacity") {
        unsafe { OPACITY = o.parse().unwrap_or(OPACITY) };
    }
    if let Some(m) = json_str(args, "mode") {
        unsafe { MODE = m.parse().unwrap_or(MODE).min(2) };
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
