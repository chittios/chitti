//! Weather card — displays durable `weather` JSON-ish payload from the agent.

use crate::guest::{json_str, storage_get_durable, storage_set_durable, ui_draw};
use alloc::format;
use alloc::string::{String, ToString};

static mut TEMP: i32 = 20;
static mut COND: u8 = 0; // 0 clear 1 cloud 2 rain 3 storm
static mut PLACE: [u8; 24] = *b"local\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
static mut PLACE_LEN: usize = 5;

fn paint() {
    let mut ops = String::from("clear 1a1816; ");
    ops.push_str("rect 0 0 256 16 3a3632; ");
    // Sky panel.
    let sky = match unsafe { COND } {
        0 => "6688cc",
        1 => "5a666a",
        2 => "3a4a5a",
        _ => "2a2030",
    };
    ops.push_str(&format!("rect 16 28 224 120 {sky}; "));
    // Sun / cloud / rain marks.
    match unsafe { COND } {
        0 => ops.push_str("rect 100 48 40 40 e8c060; "),
        1 => {
            ops.push_str("rect 80 60 80 28 e8e4df; ");
            ops.push_str("rect 100 48 48 24 e8e4df; ");
        }
        2 => {
            ops.push_str("rect 80 50 80 28 8a8a9a; ");
            for i in 0..5 {
                let x = 90 + i * 16;
                ops.push_str(&format!("rect {x} 90 4 24 88aacc; "));
            }
        }
        _ => {
            ops.push_str("rect 80 50 80 28 4a4a5a; ");
            ops.push_str("rect 120 90 8 40 e8e060; ");
        }
    }
    // Temp bar.
    let t = unsafe { TEMP }.clamp(-20, 45);
    let w = ((t + 20) * 4).min(220);
    ops.push_str(&format!("rect 16 156 {w} 12 cc785c; "));
    ops.push_str("rect 0 176 256 16 3a3632; ");
    ui_draw(&ops);
}

fn load() {
    let mut buf = [0u8; 128];
    let n = storage_get_durable("weather", &mut buf);
    if n <= 0 {
        return;
    }
    let raw = core::str::from_utf8(&buf[..n as usize]).unwrap_or("");
    if let Some(t) = json_str(raw, "temp").or_else(|| json_str(raw, "temperature")) {
        if let Ok(v) = t.parse::<i32>() {
            unsafe { TEMP = v };
        }
    }
    if let Some(c) = json_str(raw, "cond").or_else(|| json_str(raw, "condition")) {
        unsafe {
            COND = match c.as_str() {
                "clear" | "sunny" | "0" => 0,
                "cloud" | "cloudy" | "1" => 1,
                "rain" | "2" => 2,
                "storm" | "3" => 3,
                _ => COND,
            };
        }
    }
    if let Some(p) = json_str(raw, "place").or_else(|| json_str(raw, "city")) {
        let b = p.as_bytes();
        let len = b.len().min(23);
        unsafe {
            PLACE = [0; 24];
            PLACE[..len].copy_from_slice(&b[..len]);
            PLACE_LEN = len;
        }
    }
}

pub fn start(_: &str) -> String {
    load();
    paint();
    String::from("ok:weather (1-4 conditions; agent sets via weather_set)")
}

pub fn on_click(_x: i32, _y: i32) -> String {
    status("")
}

pub fn on_key(key: &str) -> String {
    match key {
        "1" => unsafe { COND = 0 },
        "2" => unsafe { COND = 1 },
        "3" => unsafe { COND = 2 },
        "4" => unsafe { COND = 3 },
        "+" | "=" => unsafe { TEMP = (TEMP + 1).min(45) },
        "-" | "_" => unsafe { TEMP = (TEMP - 1).max(-20) },
        _ => {}
    }
    save();
    paint();
    status("")
}

fn save() {
    let place = unsafe {
        core::str::from_utf8(&PLACE[..PLACE_LEN])
            .unwrap_or("local")
            .to_string()
    };
    let cond = match unsafe { COND } {
        0 => "clear",
        1 => "cloud",
        2 => "rain",
        _ => "storm",
    };
    let blob = format!(
        "{{\"temp\":{},\"cond\":\"{}\",\"place\":\"{}\"}}",
        unsafe { TEMP },
        cond,
        place
    );
    let _ = storage_set_durable("weather", &blob);
}

pub fn set(args: &str) -> String {
    if let Some(t) = json_str(args, "temp").or_else(|| json_str(args, "temperature")) {
        if let Ok(v) = t.parse::<i32>() {
            unsafe { TEMP = v };
        }
    }
    if let Some(c) = json_str(args, "cond").or_else(|| json_str(args, "condition")) {
        unsafe {
            COND = match c.as_str() {
                "clear" | "sunny" | "0" => 0,
                "cloud" | "cloudy" | "1" => 1,
                "rain" | "2" => 2,
                "storm" | "3" => 3,
                _ => COND,
            };
        }
    }
    if let Some(p) = json_str(args, "place").or_else(|| json_str(args, "city")) {
        let b = p.as_bytes();
        let len = b.len().min(23);
        unsafe {
            PLACE = [0; 24];
            PLACE[..len].copy_from_slice(&b[..len]);
            PLACE_LEN = len;
        }
    }
    save();
    paint();
    status("")
}

pub fn status(_: &str) -> String {
    let place = unsafe {
        core::str::from_utf8(&PLACE[..PLACE_LEN])
            .unwrap_or("local")
            .to_string()
    };
    format!(
        "ok:weather {}°C cond={} @{}",
        unsafe { TEMP },
        unsafe { COND },
        place
    )
}
