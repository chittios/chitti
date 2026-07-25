//! Calculator — button grid + keyboard; pure decimal arithmetic in wasm.
//! Renders real labels via the host `text` draw-op (Geist Mono).

use crate::guest::{hud_status, json_str, text_op, ui_draw};
use alloc::format;
use alloc::string::{String, ToString};

static mut DISPLAY: [u8; 24] = [0; 24];
static mut DISP_LEN: usize = 1;
static mut ACC: i64 = 0;
static mut OP: u8 = 0; // 0 none, 1+,2-,3*,4/
static mut FRESH: u8 = 1;

/// 4×4 button labels, row-major.
const LABELS: [&str; 16] = [
    "C", "/", "*", "-", //
    "7", "8", "9", "+", //
    "4", "5", "6", "=", //
    "1", "2", "3", "0",
];
const BTN_X0: i32 = 8;
const BTN_Y0: i32 = 52;
const BTN_W: i32 = 56;
const BTN_H: i32 = 30;
const BTN_GAP: i32 = 4; // total cell = 60×34

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
        if n == 0 {
            DISPLAY[0] = b'0';
            DISP_LEN = 1;
        } else {
            DISPLAY[..n].copy_from_slice(&b[..n]);
            DISP_LEN = n;
        }
    }
}

fn op_symbol() -> &'static str {
    match unsafe { OP } {
        1 => "+",
        2 => "-",
        3 => "*",
        4 => "/",
        _ => "",
    }
}

fn paint() {
    let d = disp_str();
    let mut ops = String::from("clear 1a1816; ");
    // Display strip.
    ops.push_str("rect 8 8 240 36 2c2926; ");
    // Right-align display value roughly (mono ~size*0.6 advance).
    let size = 18i32;
    let approx_w = (d.len() as i32) * (size * 6 / 10);
    let tx = (240 - approx_w).max(16);
    text_op(&mut ops, tx, 14, size, "e8e4df", &d);
    // Pending op indicator.
    let os = op_symbol();
    if !os.is_empty() {
        text_op(&mut ops, 12, 14, 14, "cc785c", os);
    }
    // Button grid 4×4 with labels.
    let colors = [
        "aa3333", "5a5652", "5a5652", "5a5652", //
        "3a3632", "3a3632", "3a3632", "5a5652", //
        "3a3632", "3a3632", "3a3632", "cc785c", //
        "3a3632", "3a3632", "3a3632", "3a3632",
    ];
    for (i, c) in colors.iter().enumerate() {
        let col = (i % 4) as i32;
        let row = (i / 4) as i32;
        let x = BTN_X0 + col * (BTN_W + BTN_GAP);
        let y = BTN_Y0 + row * (BTN_H + BTN_GAP);
        ops.push_str(&format!("rect {x} {y} {BTN_W} {BTN_H} {c}; "));
        // Centre-ish label on the button.
        let lab = LABELS[i];
        let lx = x + BTN_W / 2 - 5;
        let ly = y + 6;
        text_op(&mut ops, lx, ly, 16, "e8e4df", lab);
    }
    ui_draw(&ops);
    hud_status(
        &format!("calc  {d}{}", if os.is_empty() { String::new() } else { format!("  ({os})") }),
        "0-9 digits  + - * /  enter =  esc clear  ·  click buttons",
    );
}

fn apply_op() {
    let n: i64 = disp_str().parse().unwrap_or(0);
    unsafe {
        if OP == 0 {
            // Bare "=" just keeps the current display.
            FRESH = 1;
            return;
        }
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
        "C" | "c" => {
            set_disp("0");
            unsafe {
                ACC = 0;
                OP = 0;
                FRESH = 1;
            }
        }
        "+" | "-" | "*" | "/" => {
            let n: i64 = match disp_str().as_str() {
                "ERR" => 0,
                s => s.parse().unwrap_or(0),
            };
            unsafe {
                if OP != 0 && FRESH == 0 {
                    apply_op();
                } else if FRESH == 0 || OP == 0 {
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
        "=" => apply_op(),
        d if d.len() == 1 && d.chars().next().unwrap().is_ascii_digit() => {
            input_digit(d.chars().next().unwrap());
        }
        _ => {}
    }
    paint();
    format!("ok:{}", disp_str())
}

/// Map surface click → button index, or None if off-grid.
fn hit_button(x: i32, y: i32) -> Option<&'static str> {
    if y < BTN_Y0 || x < BTN_X0 {
        return None;
    }
    let cell_w = BTN_W + BTN_GAP;
    let cell_h = BTN_H + BTN_GAP;
    let col = (x - BTN_X0) / cell_w;
    let row = (y - BTN_Y0) / cell_h;
    if !(0..4).contains(&col) || !(0..4).contains(&row) {
        return None;
    }
    // Only count clicks inside the button rect, not the gap.
    let lx = (x - BTN_X0) - col * cell_w;
    let ly = (y - BTN_Y0) - row * cell_h;
    if lx >= BTN_W || ly >= BTN_H {
        return None;
    }
    let i = (row * 4 + col) as usize;
    Some(LABELS[i])
}

pub fn start(_: &str) -> String {
    set_disp("0");
    unsafe {
        ACC = 0;
        OP = 0;
        FRESH = 1;
    }
    paint();
    String::from("ok:calc ready (digits + - * / = C; click or type)")
}

pub fn on_click(x: i32, y: i32) -> String {
    if let Some(lab) = hit_button(x, y) {
        return press(lab);
    }
    format!("ok:{}", disp_str())
}

pub fn on_key(key: &str) -> String {
    match key {
        "enter" => press("="),
        "esc" | "c" | "C" => press("C"),
        "+" | "-" | "*" | "/" | "=" => press(key),
        // Numpad / shifted: some keyboards send "x" for multiply.
        "x" | "X" => press("*"),
        k if k.len() == 1 && k.chars().next().unwrap().is_ascii_digit() => press(k),
        _ => String::from("ok"),
    }
}

pub fn eval(args: &str) -> String {
    let expr = json_str(args, "expr")
        .or_else(|| json_str(args, "expression"))
        .unwrap_or_default();
    if expr.is_empty() {
        return format!("ok:display={}", disp_str());
    }
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
    unsafe {
        ACC = r;
        OP = 0;
        FRESH = 1;
    }
    paint();
    format!("ok:{r}")
}

pub fn status(_: &str) -> String {
    format!("ok:display={}", disp_str())
}
