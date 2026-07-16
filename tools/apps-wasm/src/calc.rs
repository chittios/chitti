//! Calculator — button grid + keyboard; pure decimal arithmetic in wasm.

use crate::guest::{json_str, ui_draw};
use alloc::format;
use alloc::string::{String, ToString};

static mut DISPLAY: [u8; 24] = [0; 24];
static mut DISP_LEN: usize = 1;
static mut ACC: i64 = 0;
static mut OP: u8 = 0; // 0 none, 1+,2-,3*,4/
static mut FRESH: u8 = 1;

fn disp_str() -> String {
    unsafe {
        if DISP_LEN == 0 || DISPLAY[0] == 0 {
            return String::from("0");
        }
        core::str::from_utf8(&DISPLAY[..DISP_LEN])
            .unwrap_or("0")
            .to_string()
    }
}

fn set_disp(s: &str) {
    let b = s.as_bytes();
    let n = b.len().min(23);
    unsafe {
        DISPLAY = [0; 24];
        DISPLAY[..n].copy_from_slice(&b[..n]);
        DISP_LEN = n.max(1);
        if n == 0 {
            DISPLAY[0] = b'0';
            DISP_LEN = 1;
        }
    }
}

fn paint() {
    let d = disp_str();
    let mut ops = String::from("clear 1a1816; ");
    // Display strip.
    ops.push_str("rect 8 8 240 36 2c2926; ");
    // Digit hint bars (draw-op has no text glyphs — use length ticks + color).
    let len = d.len() as i32;
    let bar_w = (len * 8).min(220);
    ops.push_str(&format!("rect {} 16 {} 20 cc785c; ", 240 - bar_w, bar_w));
    // Button grid 4×5.
    let labels = [
        "C", "/", "*", "-", "7", "8", "9", "+", "4", "5", "6", "=", "1", "2", "3", "0",
    ];
    let colors = [
        "aa3333", "5a5652", "5a5652", "5a5652", "3a3632", "3a3632", "3a3632", "5a5652", "3a3632",
        "3a3632", "3a3632", "cc785c", "3a3632", "3a3632", "3a3632", "3a3632",
    ];
    for (i, c) in colors.iter().enumerate() {
        let col = (i % 4) as i32;
        let row = (i / 4) as i32;
        let x = 8 + col * 60;
        let y = 52 + row * 34;
        ops.push_str(&format!("rect {x} {y} 56 30 {c}; "));
        let _ = labels[i];
    }
    ui_draw(&ops);
}

fn apply_op() {
    let n: i64 = disp_str().parse().unwrap_or(0);
    unsafe {
        ACC = match OP {
            1 => ACC.saturating_add(n),
            2 => ACC.saturating_sub(n),
            3 => ACC.saturating_mul(n),
            4 => {
                if n == 0 {
                    set_disp("ERR");
                    OP = 0;
                    FRESH = 1;
                    return;
                }
                ACC / n
            }
            _ => n,
        };
        set_disp(&ACC.to_string());
        OP = 0;
        FRESH = 1;
    }
}

fn input_digit(ch: char) {
    unsafe {
        if FRESH != 0 || disp_str() == "0" || disp_str() == "ERR" {
            set_disp(&ch.to_string());
            FRESH = 0;
        } else if DISP_LEN < 16 {
            let mut s = disp_str();
            s.push(ch);
            set_disp(&s);
        }
    }
}

fn press(label: &str) -> String {
    match label {
        "C" | "c" | "esc" => {
            set_disp("0");
            unsafe {
                ACC = 0;
                OP = 0;
                FRESH = 1;
            }
        }
        "+" | "-" | "*" | "/" => {
            let n: i64 = disp_str().parse().unwrap_or(0);
            unsafe {
                if OP != 0 && FRESH == 0 {
                    apply_op();
                } else {
                    ACC = n;
                }
                OP = match label {
                    "+" => 1,
                    "-" => 2,
                    "*" => 3,
                    _ => 4,
                };
                FRESH = 1;
            }
        }
        "=" | "enter" => apply_op(),
        d if d.len() == 1 && d.chars().next().unwrap().is_ascii_digit() => {
            input_digit(d.chars().next().unwrap());
        }
        _ => {}
    }
    paint();
    format!("ok:{}", disp_str())
}

pub fn start(_: &str) -> String {
    set_disp("0");
    unsafe {
        ACC = 0;
        OP = 0;
        FRESH = 1;
    }
    paint();
    String::from("ok:calc ready (digits + - * / = C; click buttons)")
}

pub fn on_click(x: i32, y: i32) -> String {
    if y < 52 {
        return format!("ok:{}", disp_str());
    }
    let col = ((x - 8) / 60).clamp(0, 3);
    let row = ((y - 52) / 34).clamp(0, 3);
    let i = (row * 4 + col) as usize;
    let labels = [
        "C", "/", "*", "-", "7", "8", "9", "+", "4", "5", "6", "=", "1", "2", "3", "0",
    ];
    if i < labels.len() {
        press(labels[i])
    } else {
        format!("ok:{}", disp_str())
    }
}

pub fn on_key(key: &str) -> String {
    match key {
        "enter" => press("="),
        "esc" => press("C"),
        "+" | "-" | "*" | "/" | "=" | "c" | "C" => press(key),
        k if k.len() == 1 && k.chars().next().unwrap().is_ascii_digit() => press(k),
        _ => format!("ok:{}", disp_str()),
    }
}

pub fn eval(args: &str) -> String {
    // Chat tool: {"expr":"12+3"}
    let expr = json_str(args, "expr")
        .or_else(|| json_str(args, "expression"))
        .unwrap_or_default();
    if expr.is_empty() {
        return format!("ok:display={}", disp_str());
    }
    // Very small expression: a op b
    let mut a = String::new();
    let mut op = 0u8;
    let mut b = String::new();
    for ch in expr.chars() {
        if ch.is_ascii_digit() || (ch == '-' && a.is_empty() && op == 0) {
            if op == 0 {
                a.push(ch);
            } else {
                b.push(ch);
            }
        } else if matches!(ch, '+' | '-' | '*' | '/') && op == 0 && !a.is_empty() {
            op = match ch {
                '+' => 1,
                '-' => 2,
                '*' => 3,
                _ => 4,
            };
        }
    }
    let av: i64 = a.parse().unwrap_or(0);
    let bv: i64 = b.parse().unwrap_or(0);
    let r = match op {
        1 => av.saturating_add(bv),
        2 => av.saturating_sub(bv),
        3 => av.saturating_mul(bv),
        4 if bv != 0 => av / bv,
        4 => {
            set_disp("ERR");
            paint();
            return String::from("error:div0");
        }
        _ => av,
    };
    set_disp(&r.to_string());
    paint();
    format!("ok:{r}")
}

pub fn status(_: &str) -> String {
    format!("ok:display={}", disp_str())
}
