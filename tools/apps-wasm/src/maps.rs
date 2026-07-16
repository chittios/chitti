//! Maps — simple lat/lon pin card (no tile stack; storage-backed places).

use crate::guest::{json_str, storage_get_durable, storage_list_durable, storage_set_durable, ui_draw};
use alloc::format;
use alloc::string::{String, ToString};

static mut LAT: i32 = 377; // tenths of degrees * 10 → 37.7
static mut LON: i32 = -1224; // -122.4
static mut ZOOM: u8 = 2;
static mut PLACE: [u8; 24] = *b"home\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
static mut PLEN: usize = 4;

fn place_str() -> String {
    unsafe {
        core::str::from_utf8(&PLACE[..PLEN])
            .unwrap_or("home")
            .to_string()
    }
}

fn paint() {
    let mut ops = String::from("clear 1a1816; ");
    ops.push_str("rect 0 0 256 16 3a3632; ");
    // Map frame (stylized grid).
    ops.push_str("rect 16 24 224 140 2a3a4a; ");
    let z = unsafe { ZOOM }.max(1) as i32;
    for i in 0..8 {
        let x = 16 + i * 28;
        let y = 24 + i * 16;
        ops.push_str(&format!("rect {x} 24 1 140 3a4a5a; "));
        ops.push_str(&format!("rect 16 {y} 224 1 3a4a5a; "));
    }
    // Pin from lat/lon (mapped into frame).
    let px = 16 + (((unsafe { LON } + 1800).rem_euclid(3600)) * 224 / 3600);
    let py = 24 + (((900 - unsafe { LAT }).rem_euclid(1800)) * 140 / 1800);
    let pin_x = px.clamp(20, 230);
    let pin_y = py.clamp(28, 155);
    ops.push_str(&format!("rect {} {} 8 8 cc785c; ", pin_x - 4, pin_y - 4));
    ops.push_str(&format!("rect {} {} 2 12 e8e4df; ", pin_x - 1, pin_y));
    // Zoom ticks.
    let zw = z * 20;
    ops.push_str(&format!("rect 16 168 {zw} 8 5a8f5a; "));
    ops.push_str("rect 0 184 256 8 3a3632; ");
    ui_draw(&ops);
}

fn load() {
    let mut buf = [0u8; 128];
    let n = storage_get_durable("map_here", &mut buf);
    if n <= 0 {
        return;
    }
    let raw = core::str::from_utf8(&buf[..n as usize]).unwrap_or("");
    if let Some(v) = json_str(raw, "lat") {
        if let Ok(n) = v.parse::<i32>() {
            unsafe { LAT = n };
        }
    }
    if let Some(v) = json_str(raw, "lon") {
        if let Ok(n) = v.parse::<i32>() {
            unsafe { LON = n };
        }
    }
    if let Some(v) = json_str(raw, "zoom") {
        if let Ok(n) = v.parse::<u8>() {
            unsafe { ZOOM = n.clamp(1, 8) };
        }
    }
    if let Some(p) = json_str(raw, "place") {
        let b = p.as_bytes();
        let len = b.len().min(23);
        unsafe {
            PLACE = [0; 24];
            PLACE[..len].copy_from_slice(&b[..len]);
            PLEN = len;
        }
    }
}

fn save() {
    let blob = format!(
        "{{\"lat\":{},\"lon\":{},\"zoom\":{},\"place\":\"{}\"}}",
        unsafe { LAT },
        unsafe { LON },
        unsafe { ZOOM },
        place_str()
    );
    let _ = storage_set_durable("map_here", &blob);
}

pub fn start(_: &str) -> String {
    load();
    paint();
    String::from("ok:maps (arrows pan, +/- zoom, s save place, p list places)")
}

pub fn on_click(x: i32, y: i32) -> String {
    if y >= 24 && y < 164 && x >= 16 && x < 240 {
        // Set pin under click.
        unsafe {
            LON = ((x - 16) * 3600 / 224) - 1800;
            LAT = 900 - ((y - 24) * 1800 / 140);
        }
        save();
        paint();
    }
    status("")
}

pub fn on_key(key: &str) -> String {
    match key {
        "left" => unsafe { LON -= 10 * ZOOM as i32 },
        "right" => unsafe { LON += 10 * ZOOM as i32 },
        "up" => unsafe { LAT += 10 * ZOOM as i32 },
        "down" => unsafe { LAT -= 10 * ZOOM as i32 },
        "+" | "=" => unsafe {
            ZOOM = (ZOOM + 1).min(8);
        },
        "-" | "_" => unsafe {
            ZOOM = ZOOM.saturating_sub(1).max(1);
        },
        "s" => {
            let key = format!("map_{}", place_str());
            let blob = format!(
                "{{\"lat\":{},\"lon\":{},\"place\":\"{}\"}}",
                unsafe { LAT },
                unsafe { LON },
                place_str()
            );
            let _ = storage_set_durable(&key, &blob);
            save();
        }
        "h" => {
            // reset home-ish SF bay
            unsafe {
                LAT = 377;
                LON = -1224;
                ZOOM = 2;
            }
            save();
        }
        _ => {}
    }
    paint();
    status("")
}

pub fn set(args: &str) -> String {
    if let Some(v) = json_str(args, "lat") {
        if let Ok(n) = v.parse::<i32>() {
            unsafe { LAT = n };
        }
    }
    if let Some(v) = json_str(args, "lon") {
        if let Ok(n) = v.parse::<i32>() {
            unsafe { LON = n };
        }
    }
    if let Some(v) = json_str(args, "zoom") {
        if let Ok(n) = v.parse::<u8>() {
            unsafe { ZOOM = n.clamp(1, 8) };
        }
    }
    if let Some(p) = json_str(args, "place") {
        let b = p.as_bytes();
        let len = b.len().min(23);
        unsafe {
            PLACE = [0; 24];
            PLACE[..len].copy_from_slice(&b[..len]);
            PLEN = len;
        }
    }
    save();
    paint();
    status("")
}

pub fn list(_: &str) -> String {
    let mut buf = [0u8; 4096];
    let n = storage_list_durable(&mut buf);
    if n <= 0 {
        return String::from("(no places)");
    }
    let raw = core::str::from_utf8(&buf[..n as usize]).unwrap_or("");
    let mut out = String::new();
    for line in raw.split('\n') {
        if let Some(rest) = line.strip_prefix("map_") {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(rest);
        }
    }
    if out.is_empty() {
        String::from("(no places)")
    } else {
        out
    }
}

pub fn status(_: &str) -> String {
    format!(
        "ok:maps lat={} lon={} zoom={} place={}",
        unsafe { LAT },
        unsafe { LON },
        unsafe { ZOOM },
        place_str()
    )
}
