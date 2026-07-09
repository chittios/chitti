use crate::guest::{json_i32, json_str, ui_draw};
use alloc::format;
use alloc::string::{String, ToString};

static mut COLOR: [u8; 8] = *b"cc785c\0\0";

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
    String::from("ok:paint ready")
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
