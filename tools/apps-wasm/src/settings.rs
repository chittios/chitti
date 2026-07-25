//! Settings — panel UI with real text labels; applies theme/mode/opacity to the host.

use crate::guest::{
    hud_status, json_str, storage_get_durable, storage_set_durable, sys_get, sys_set, text_op,
    ui_draw,
};
use alloc::format;
use alloc::string::{String, ToString};

// panels: 0 theme 1 network 2 model 3 voice 4 mode
static mut PANEL: u8 = 0;
static mut THEME: u8 = 0; // index into THEMES
static mut OPACITY: u8 = 255;
static mut MODE: u8 = 1; // 0 manual 1 auto 2 bypass (matches shell default = auto)

/// Theme preset names — must match `assets/themes/*.json` / `theme::apply`.
const THEMES: [&str; 6] = ["dark", "light", "nord", "dracula", "solarized-dark", "ubuntu"];
const THEME_LABELS: [&str; 6] = ["Dark", "Light", "Nord", "Dracula", "Solar", "Ubuntu"];
const NAV: [&str; 5] = ["Theme", "Net", "Model", "Voice", "Mode"];
const MODES: [&str; 3] = ["Manual", "Auto", "Bypass"];
const MODE_NAMES: [&str; 3] = ["manual", "auto", "bypass"];

fn paint() {
    let panel = unsafe { PANEL };
    let theme = unsafe { THEME }.min(5);
    let opacity = unsafe { OPACITY };
    let mode = unsafe { MODE }.min(2);

    let mut ops = String::from("clear 1a1816; ");
    ops.push_str("rect 0 0 256 18 3a3632; ");
    text_op(&mut ops, 8, 2, 12, "e8e4df", "Settings");
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
    ops.push_str("rect 80 24 168 144 2c2926; ");
    match panel {
        0 => {
            text_op(&mut ops, 96, 32, 12, "cc785c", "Theme");
            // 2×3 grid of theme chips.
            for i in 0..6i32 {
                let col = i % 3;
                let row = i / 3;
                let x = 96 + col * 52;
                let y = 50 + row * 30;
                let c = if theme as i32 == i {
                    "e8e4df"
                } else {
                    "5a5652"
                };
                ops.push_str(&format!("rect {x} {y} 48 26 {c}; "));
                let tc = if theme as i32 == i { "1a1816" } else { "e8e4df" };
                text_op(&mut ops, x + 4, y + 5, 10, tc, THEME_LABELS[i as usize]);
            }
            text_op(&mut ops, 96, 118, 11, "e8e4df", "Opacity");
            let ow = (opacity as i32 / 2).min(160).max(4);
            ops.push_str(&format!("rect 96 136 {ow} 10 6688cc; "));
            text_op(
                &mut ops,
                96,
                150,
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
            text_op(&mut ops, 96, 112, 10, "a8a4a0", "/voice remote");
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
            THEMES[theme as usize],
            MODE_NAMES[mode as usize],
            opacity
        ),
        "1-5 panels  arrows theme  +/- opacity  m/a/b mode",
    );
}

/// Push current selection to the host (theme / mode / opacity) and durable cache.
fn apply() {
    let theme = unsafe { THEME }.min(5) as usize;
    let mode = unsafe { MODE }.min(2) as usize;
    let opacity = unsafe { OPACITY };
    let _ = sys_set("theme", THEMES[theme]);
    let _ = sys_set("mode", MODE_NAMES[mode]);
    let _ = sys_set("opacity", &format!("{opacity}"));
    let blob = format!(
        "{{\"theme\":{},\"opacity\":{},\"mode\":{},\"panel\":{}}}",
        theme,
        opacity,
        mode,
        unsafe { PANEL }
    );
    let _ = storage_set_durable("settings", &blob);
}

fn load() {
    // Prefer live host state so the panel mirrors `/theme` and `/mode`.
    if let Some(name) = sys_get("theme") {
        let n = name.to_ascii_lowercase();
        if let Some(i) = THEMES.iter().position(|t| *t == n.as_str()) {
            unsafe { THEME = i as u8 };
        }
    }
    if let Some(m) = sys_get("mode") {
        unsafe {
            MODE = match m.as_str() {
                "manual" | "0" => 0,
                "bypass" | "2" => 2,
                "plan" | "3" => 1, // show Auto chip; plan is shell-only
                _ => 1,
            };
        }
    }
    if let Some(o) = sys_get("opacity") {
        if let Ok(v) = o.parse::<u8>() {
            unsafe { OPACITY = v };
        }
    }
    // Fill any gaps from durable cache.
    let mut buf = [0u8; 128];
    let n = storage_get_durable("settings", &mut buf);
    if n <= 0 {
        return;
    }
    let raw = core::str::from_utf8(&buf[..n as usize]).unwrap_or("");
    if sys_get("theme").is_none() {
        if let Some(t) = json_str(raw, "theme") {
            unsafe { THEME = t.parse().unwrap_or(0).min(5) };
        }
    }
    if sys_get("opacity").is_none() {
        if let Some(o) = json_str(raw, "opacity") {
            unsafe { OPACITY = o.parse().unwrap_or(255) };
        }
    }
    if sys_get("mode").is_none() {
        if let Some(m) = json_str(raw, "mode") {
            unsafe { MODE = m.parse().unwrap_or(1).min(2) };
        }
    }
}

pub fn start(_: &str) -> String {
    load();
    paint();
    String::from("ok:settings (theme applies live; 1-5 panels; m/a/b mode)")
}

pub fn on_click(x: i32, y: i32) -> String {
    if x < 72 && y >= 24 && y < 24 + 5 * 28 {
        let p = ((y - 24) / 28).clamp(0, 4) as u8;
        unsafe { PANEL = p };
        paint();
        return status("");
    }
    if unsafe { PANEL } == 0 {
        // Theme chips 2×3.
        if y >= 50 && y < 110 && x >= 96 {
            let col = ((x - 96) / 52).clamp(0, 2);
            let row = ((y - 50) / 30).clamp(0, 1);
            let t = (row * 3 + col) as u8;
            if t < 6 {
                unsafe { THEME = t };
                apply();
            }
        }
    }
    if unsafe { PANEL } == 4 && y >= 56 && x >= 96 {
        let m = ((y - 56) / 28).clamp(0, 2) as u8;
        unsafe { MODE = m };
        apply();
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
                THEME = (THEME + 1).min(5);
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
    // Theme/mode/opacity keys apply immediately; panel-only navigation just paints.
    if matches!(
        key,
        "left"
            | "h"
            | "right"
            | "l"
            | "+"
            | "="
            | "-"
            | "_"
            | "m"
            | "a"
            | "b"
    ) {
        apply();
    }
    paint();
    status("")
}

pub fn get(_: &str) -> String {
    load();
    format!(
        "ok:theme={} opacity={} mode={} (0 manual 1 auto 2 bypass)",
        THEMES[unsafe { THEME }.min(5) as usize],
        unsafe { OPACITY },
        MODE_NAMES[unsafe { MODE }.min(2) as usize]
    )
}

pub fn set(args: &str) -> String {
    if let Some(t) = json_str(args, "theme") {
        // Accept index or name.
        if let Ok(i) = t.parse::<u8>() {
            unsafe { THEME = i.min(5) };
        } else {
            let n = t.to_ascii_lowercase();
            if let Some(i) = THEMES.iter().position(|x| *x == n.as_str()) {
                unsafe { THEME = i as u8 };
            }
        }
    }
    if let Some(o) = json_str(args, "opacity") {
        unsafe { OPACITY = o.parse().unwrap_or(OPACITY) };
    }
    if let Some(m) = json_str(args, "mode") {
        if let Ok(i) = m.parse::<u8>() {
            unsafe { MODE = i.min(2) };
        } else {
            unsafe {
                MODE = match m.as_str() {
                    "manual" => 0,
                    "bypass" => 2,
                    _ => 1,
                };
            }
        }
    }
    apply();
    paint();
    get("")
}

pub fn status(_: &str) -> String {
    format!(
        "ok:settings panel={} theme={} mode={}",
        unsafe { PANEL },
        THEMES[unsafe { THEME }.min(5) as usize],
        MODE_NAMES[unsafe { MODE }.min(2) as usize]
    )
}
