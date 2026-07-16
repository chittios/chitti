use crate::guest::{json_i32, json_str, ui_draw};
use alloc::format;
use alloc::string::{String, ToString};

static mut COLOR: [u8; 8] = *b"cc785c\0\0";
/// Brush square size in pixels (`+`/`-` adjust, 2..=16).
static mut BRUSH: i32 = 6;

/// Palette swatches: key `1`..`8` / click on the top strip. Slot 8 is the
/// canvas background — an eraser.
const PALETTE: [&str; 8] = [
    "cc785c", "e8e4df", "6688cc", "5a8f5a", "c9a54a", "aa3333", "3a3632", "1a1816",
];
/// Height of the palette strip at the top of the canvas.
const BAR: i32 = 12;

/// Repaint the palette strip (selected swatch gets a bright underline).
fn paint_bar() {
    let mut ops = String::new();
    let cur = color();
    for (i, c) in PALETTE.iter().enumerate() {
        let x = i as i32 * 32;
        ops.push_str(&format!("rect {x} 0 31 {} {c}; ", BAR - 2));
        if *c == cur {
            ops.push_str(&format!("rect {x} {} 31 2 e8e4df; ", BAR - 2));
        }
    }
    ui_draw(&ops);
}

fn color() -> String {
    unsafe {
        let s = core::str::from_utf8(&COLOR).unwrap_or("cc785c");
        s.trim_end_matches('\0').to_string()
    }
}

fn set_color(c: &str) {
    let b = c.as_bytes();
    unsafe {
        COLOR = [0; 8];
        let n = b.len().min(6);
        COLOR[..n].copy_from_slice(&b[..n]);
    }
}

pub fn start(_: &str) -> String {
    ui_draw("clear 1a1816");
    paint_bar();
    String::from("ok:paint ready (click to draw, 1-8 colors, +/- brush, c clear)")
}

/// Surface click: on the palette strip → select that swatch; below → stamp a
/// brush square in the current colour.
pub fn on_click(x: i32, y: i32) -> String {
    if y < BAR {
        let i = (x / 32).clamp(0, 7) as usize;
        set_color(PALETTE[i]);
        paint_bar();
        return format!("ok:color={}", PALETTE[i]);
    }
    let b = unsafe { BRUSH };
    let half = b / 2;
    let px = (x - half).max(0);
    let py = (y - half).max(BAR);
    ui_draw(&format!("rect {px} {py} {b} {b} {}", color()));
    String::from("ok:dot")
}

/// Key: `1`..`8` palette, `c` clear canvas, `+`/`-` brush size.
pub fn on_key(key: &str) -> String {
    match key {
        "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" => {
            let i = (key.as_bytes()[0] - b'1') as usize;
            set_color(PALETTE[i]);
            paint_bar();
            format!("ok:color={}", PALETTE[i])
        }
        "c" => {
            ui_draw("clear 1a1816");
            paint_bar();
            String::from("ok:clear")
        }
        "+" | "=" => {
            unsafe { BRUSH = (BRUSH + 2).min(16) };
            format!("ok:brush={}", unsafe { BRUSH })
        }
        "-" | "_" => {
            unsafe { BRUSH = (BRUSH - 2).max(2) };
            format!("ok:brush={}", unsafe { BRUSH })
        }
        _ => String::from("ok"),
    }
}

pub fn clear(args: &str) -> String {
    let c = json_str(args, "color").unwrap_or_else(|| "1a1816".into());
    set_color(&c);
    ui_draw(&format!("clear {c}"));
    format!("ok:clear {c}")
}

pub fn rect(args: &str) -> String {
    let x = json_i32(args, "x", 0);
    let y = json_i32(args, "y", 0);
    let w = json_i32(args, "w", 10);
    let h = json_i32(args, "h", 10);
    let c = json_str(args, "color").unwrap_or_else(color);
    ui_draw(&format!("rect {x} {y} {w} {h} {c}"));
    String::from("ok:rect")
}

pub fn line(args: &str) -> String {
    let x0 = json_i32(args, "x0", 0);
    let y0 = json_i32(args, "y0", 0);
    let x1 = json_i32(args, "x1", 10);
    let y1 = json_i32(args, "y1", 10);
    let c = json_str(args, "color").unwrap_or_else(color);
    ui_draw(&format!("line {x0} {y0} {x1} {y1} {c}"));
    String::from("ok:line")
}

pub fn pixel(args: &str) -> String {
    let x = json_i32(args, "x", 0);
    let y = json_i32(args, "y", 0);
    let c = json_str(args, "color").unwrap_or_else(color);
    ui_draw(&format!("pixel {x} {y} {c}"));
    String::from("ok:pixel")
}

pub fn draw_ops(args: &str) -> String {
    let ops = json_str(args, "ops").unwrap_or_default();
    if ops.is_empty() {
        return String::from("error: missing ops");
    }
    ui_draw(&ops);
    String::from("ok:draw")
}

pub fn status(_: &str) -> String {
    format!("ok:color={}", color())
}
