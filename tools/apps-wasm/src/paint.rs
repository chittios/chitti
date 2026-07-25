use crate::guest::{hud_status, json_i32, json_str, text_op, ui_draw};
use alloc::format;
use alloc::string::{String, ToString};

/// Surface size (must match host canvas).
const SW: i32 = 256;
const SH: i32 = 192;
/// Palette strip height at the top of the canvas.
const BAR: i32 = 12;

static mut COLOR: [u8; 8] = *b"cc785c\0\0";
/// Brush square size in pixels (`+`/`-` adjust, 2..=16).
static mut BRUSH: i32 = 6;
/// Keyboard cursor (stamps with space/enter; arrows move).
static mut CX: i32 = 128;
static mut CY: i32 = 96;
/// Canvas as palette index per pixel (0 = background). Enables full repaint
/// so the keyboard cursor can move without erasing strokes.
static mut PIX: [u8; (SW * SH) as usize] = [0; (SW * SH) as usize];

/// Palette swatches: key `1`..`8` / click on the top strip. Slot 8 is the
/// canvas background — an eraser.
const PALETTE: [&str; 8] = [
    "cc785c", "e8e4df", "6688cc", "5a8f5a", "c9a54a", "aa3333", "3a3632", "1a1816",
];
const SHORTCUTS: &str = "arrows move  space stamp  1-8 colour  +/- brush  c clear";

fn color() -> String {
    unsafe {
        let s = core::str::from_utf8(&COLOR).unwrap_or("cc785c");
        s.trim_end_matches('\0').to_string()
    }
}

fn color_idx() -> u8 {
    let cur = color();
    for (i, c) in PALETTE.iter().enumerate() {
        if *c == cur {
            return (i + 1) as u8;
        }
    }
    1
}

fn set_color(c: &str) {
    let b = c.as_bytes();
    unsafe {
        COLOR = [0; 8];
        let n = b.len().min(6);
        COLOR[..n].copy_from_slice(&b[..n]);
    }
}

fn stamp_canvas(px: i32, py: i32, b: i32, idx: u8) {
    unsafe {
        for dy in 0..b {
            for dx in 0..b {
                let x = px + dx;
                let y = py + dy;
                if x >= 0 && x < SW && y >= BAR && y < SH {
                    PIX[(y * SW + x) as usize] = idx;
                }
            }
        }
    }
}

fn clear_canvas() {
    unsafe {
        PIX = [0; (SW * SH) as usize];
    }
}

/// Full repaint: canvas from PIX + palette bar + keyboard cursor crosshair.
fn paint_all() {
    let mut ops = String::from("clear 1a1816; ");
    // Canvas pixels as runs of equal colour per row (keeps draw-op budget down).
    unsafe {
        for y in BAR..SH {
            let mut x = 0i32;
            while x < SW {
                let idx = PIX[(y * SW + x) as usize];
                let mut x2 = x + 1;
                while x2 < SW && PIX[(y * SW + x2) as usize] == idx {
                    x2 += 1;
                }
                if idx != 0 {
                    let c = PALETTE[(idx as usize - 1).min(7)];
                    let w = x2 - x;
                    ops.push_str(&format!("rect {x} {y} {w} 1 {c}; "));
                }
                x = x2;
            }
        }
    }
    // Palette strip with key digits.
    let cur = color();
    for (i, c) in PALETTE.iter().enumerate() {
        let x = i as i32 * 32;
        ops.push_str(&format!("rect {x} 0 31 {} {c}; ", BAR - 2));
        if *c == cur {
            ops.push_str(&format!("rect {x} {} 31 2 e8e4df; ", BAR - 2));
        }
        let digit = format!("{}", i + 1);
        let tc = if i == 1 || i == 6 { "1a1816" } else { "e8e4df" };
        text_op(&mut ops, x + 10, 0, 9, tc, &digit);
    }
    // Cursor crosshair (hollow so it doesn't hide the stamp).
    let (cx, cy, b) = unsafe { (CX, CY, BRUSH) };
    let half = b / 2;
    let px = (cx - half).max(0);
    let py = (cy - half).max(BAR);
    ops.push_str(&format!(
        "rect {px} {py} {b} 1 e8e4df; rect {px} {} {b} 1 e8e4df; rect {px} {py} 1 {b} e8e4df; rect {} {py} 1 {b} e8e4df; ",
        py + b - 1,
        px + b - 1
    ));
    ui_draw(&ops);
    refresh_hud("");
}

fn refresh_hud(extra: &str) {
    let (cx, cy, b) = unsafe { (CX, CY, BRUSH) };
    let mut status = format!("paint  colour={}  brush={b}  @ ({cx},{cy})", color());
    if !extra.is_empty() {
        status.push_str("  ");
        status.push_str(extra);
    }
    hud_status(&status, SHORTCUTS);
}

pub fn start(_: &str) -> String {
    clear_canvas();
    unsafe {
        CX = 128;
        CY = 96;
        BRUSH = 6;
    }
    set_color(PALETTE[0]);
    paint_all();
    String::from("ok:paint ready (arrows/space keyboard; 1-8 colours; click to draw)")
}

/// Surface click: on the palette strip → select that swatch; below → stamp a
/// brush square in the current colour.
pub fn on_click(x: i32, y: i32) -> String {
    if y < BAR {
        let i = (x / 32).clamp(0, 7) as usize;
        set_color(PALETTE[i]);
        paint_all();
        return format!("ok:color={}", PALETTE[i]);
    }
    let b = unsafe { BRUSH };
    let half = b / 2;
    let px = (x - half).max(0);
    let py = (y - half).max(BAR);
    unsafe {
        CX = x.clamp(0, SW - 1);
        CY = y.clamp(BAR, SH - 1);
    }
    stamp_canvas(px, py, b, color_idx());
    paint_all();
    String::from("ok:dot")
}

/// Key: arrows move cursor, space/enter stamp, `1`..`8` palette, `c` clear,
/// `+`/`-` brush size.
pub fn on_key(key: &str) -> String {
    match key {
        "left" | "h" => {
            unsafe { CX = (CX - BRUSH).max(0) };
            paint_all();
            String::from("ok:cursor")
        }
        "right" | "l" => {
            unsafe { CX = (CX + BRUSH).min(SW - 1) };
            paint_all();
            String::from("ok:cursor")
        }
        "up" | "k" => {
            unsafe { CY = (CY - BRUSH).max(BAR) };
            paint_all();
            String::from("ok:cursor")
        }
        "down" | "j" => {
            unsafe { CY = (CY + BRUSH).min(SH - 1) };
            paint_all();
            String::from("ok:cursor")
        }
        "space" | "enter" => {
            let b = unsafe { BRUSH };
            let half = b / 2;
            let (cx, cy) = unsafe { (CX, CY) };
            let px = (cx - half).max(0);
            let py = (cy - half).max(BAR);
            stamp_canvas(px, py, b, color_idx());
            paint_all();
            String::from("ok:stamp")
        }
        "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" => {
            let i = (key.as_bytes()[0] - b'1') as usize;
            set_color(PALETTE[i]);
            paint_all();
            format!("ok:color={}", PALETTE[i])
        }
        "c" => {
            clear_canvas();
            paint_all();
            String::from("ok:clear")
        }
        "+" | "=" => {
            unsafe { BRUSH = (BRUSH + 2).min(16) };
            paint_all();
            format!("ok:brush={}", unsafe { BRUSH })
        }
        "-" | "_" => {
            unsafe { BRUSH = (BRUSH - 2).max(2) };
            paint_all();
            format!("ok:brush={}", unsafe { BRUSH })
        }
        _ => String::from("ok"),
    }
}

pub fn clear(args: &str) -> String {
    let c = json_str(args, "color").unwrap_or_else(|| "1a1816".into());
    set_color(&c);
    clear_canvas();
    paint_all();
    format!("ok:clear {c}")
}

pub fn rect(args: &str) -> String {
    let x = json_i32(args, "x", 0);
    let y = json_i32(args, "y", 0);
    let w = json_i32(args, "w", 10);
    let h = json_i32(args, "h", 10);
    let c = json_str(args, "color").unwrap_or_else(color);
    set_color(&c);
    let idx = color_idx();
    for dy in 0..h {
        for dx in 0..w {
            let px = x + dx;
            let py = y + dy;
            if px >= 0 && px < SW && py >= 0 && py < SH {
                unsafe {
                    PIX[(py * SW + px) as usize] = idx;
                }
            }
        }
    }
    paint_all();
    String::from("ok:rect")
}

pub fn line(args: &str) -> String {
    let x0 = json_i32(args, "x0", 0);
    let y0 = json_i32(args, "y0", 0);
    let x1 = json_i32(args, "x1", 10);
    let y1 = json_i32(args, "y1", 10);
    let c = json_str(args, "color").unwrap_or_else(color);
    set_color(&c);
    let idx = color_idx();
    // Bresenham onto PIX.
    let mut x = x0;
    let mut y = y0;
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        if x >= 0 && x < SW && y >= 0 && y < SH {
            unsafe {
                PIX[(y * SW + x) as usize] = idx;
            }
        }
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
    paint_all();
    String::from("ok:line")
}

pub fn pixel(args: &str) -> String {
    let x = json_i32(args, "x", 0);
    let y = json_i32(args, "y", 0);
    let c = json_str(args, "color").unwrap_or_else(color);
    set_color(&c);
    if x >= 0 && x < SW && y >= 0 && y < SH {
        unsafe {
            PIX[(y * SW + x) as usize] = color_idx();
        }
    }
    paint_all();
    String::from("ok:pixel")
}

pub fn draw_ops(args: &str) -> String {
    let ops = json_str(args, "ops").unwrap_or_default();
    if ops.is_empty() {
        return String::from("error: missing ops");
    }
    // Raw ops bypass the canvas buffer (tool path); still refresh HUD.
    ui_draw(&ops);
    refresh_hud("draw");
    String::from("ok:draw")
}

pub fn status(_: &str) -> String {
    format!(
        "ok:color={} brush={} cursor={},{}",
        color(),
        unsafe { BRUSH },
        unsafe { CX },
        unsafe { CY }
    )
}
