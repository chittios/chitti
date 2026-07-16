//! Sheets — tiny spreadsheet (4×6 cells, integer formulas A1+B1 style).

use crate::guest::{json_i32, json_str, storage_get_durable, storage_set_durable, ui_draw};
use alloc::format;
use alloc::string::{String, ToString};

const COLS: usize = 4;
const ROWS: usize = 6;
const CW: i32 = 58;
const CH: i32 = 22;
const OX: i32 = 12;
const OY: i32 = 24;

/// Cell text (short).
static mut CELL: [[u8; 12]; COLS * ROWS] = [[0; 12]; COLS * ROWS];
static mut CELL_LEN: [u8; COLS * ROWS] = [0; COLS * ROWS];
static mut CUR_C: u8 = 0;
static mut CUR_R: u8 = 0;
static mut ENTRY: [u8; 16] = [0; 16];
static mut ENTRY_LEN: usize = 0;

fn idx(c: usize, r: usize) -> usize {
    r * COLS + c
}

fn cell_str(c: usize, r: usize) -> String {
    unsafe {
        let i = idx(c, r);
        let n = CELL_LEN[i] as usize;
        core::str::from_utf8(&CELL[i][..n]).unwrap_or("").to_string()
    }
}

fn set_cell(c: usize, r: usize, s: &str) {
    let i = idx(c, r);
    let b = s.as_bytes();
    let n = b.len().min(11);
    unsafe {
        CELL[i] = [0; 12];
        CELL[i][..n].copy_from_slice(&b[..n]);
        CELL_LEN[i] = n as u8;
    }
}

fn parse_num(s: &str) -> Option<i64> {
    s.trim().parse().ok()
}

/// Evaluate a cell: number, or `=A1+B2` / `=A1*2` with one operator.
fn eval_cell(c: usize, r: usize, depth: u8) -> i64 {
    if depth > 6 {
        return 0;
    }
    let s = cell_str(c, r);
    if let Some(n) = parse_num(&s) {
        return n;
    }
    let t = s.trim();
    if !t.starts_with('=') {
        return 0;
    }
    let body = &t[1..];
    let mut op = 0u8;
    let mut left = String::new();
    let mut right = String::new();
    for ch in body.chars() {
        if op == 0 && matches!(ch, '+' | '-' | '*' | '/') {
            op = match ch {
                '+' => 1,
                '-' => 2,
                '*' => 3,
                _ => 4,
            };
        } else if op == 0 {
            left.push(ch);
        } else {
            right.push(ch);
        }
    }
    let lv = ref_or_num(&left, depth + 1);
    let rv = ref_or_num(&right, depth + 1);
    match op {
        1 => lv.saturating_add(rv),
        2 => lv.saturating_sub(rv),
        3 => lv.saturating_mul(rv),
        4 if rv != 0 => lv / rv,
        4 => 0,
        _ => lv,
    }
}

fn ref_or_num(s: &str, depth: u8) -> i64 {
    let s = s.trim();
    if let Some(n) = parse_num(s) {
        return n;
    }
    // A1 style
    let bytes = s.as_bytes();
    if bytes.len() >= 2 {
        let col = (bytes[0].to_ascii_uppercase()).wrapping_sub(b'A') as usize;
        if col < COLS {
            if let Ok(row) = core::str::from_utf8(&bytes[1..]).unwrap_or("").parse::<usize>() {
                if row >= 1 && row <= ROWS {
                    return eval_cell(col, row - 1, depth);
                }
            }
        }
    }
    0
}

fn paint() {
    let mut ops = String::from("clear 1a1816; ");
    ops.push_str("rect 0 0 256 16 3a3632; ");
    // Entry bar.
    ops.push_str("rect 8 16 240 6 2c2926; ");
    let ew = (unsafe { ENTRY_LEN } as i32 * 4).min(230);
    ops.push_str(&format!("rect 10 16 {ew} 4 cc785c; "));
    for r in 0..ROWS {
        for c in 0..COLS {
            let x = OX + c as i32 * CW;
            let y = OY + r as i32 * CH;
            let sel = unsafe { CUR_C as usize == c && CUR_R as usize == r };
            let color = if sel { "cc785c" } else { "3a3632" };
            ops.push_str(&format!("rect {x} {y} {} {} {color}; ", CW - 2, CH - 2));
            let v = eval_cell(c, r, 0);
            // Magnitude bar.
            let w = ((v.abs() % 50) as i32).max(2);
            ops.push_str(&format!("rect {} {} {w} 6 e8e4df; ", x + 4, y + 8));
        }
    }
    ops.push_str("rect 0 176 256 16 3a3632; ");
    ui_draw(&ops);
}

fn commit_entry() {
    let s = unsafe {
        core::str::from_utf8(&ENTRY[..ENTRY_LEN])
            .unwrap_or("")
            .to_string()
    };
    if !s.is_empty() {
        unsafe {
            set_cell(CUR_C as usize, CUR_R as usize, &s);
            ENTRY = [0; 16];
            ENTRY_LEN = 0;
        }
        persist();
    }
}

fn persist() {
    let mut blob = String::new();
    for r in 0..ROWS {
        for c in 0..COLS {
            if c > 0 {
                blob.push('|');
            }
            blob.push_str(&cell_str(c, r));
        }
        blob.push('\n');
    }
    let _ = storage_set_durable("sheet_main", &blob);
}

fn restore() {
    let mut buf = [0u8; 1024];
    let n = storage_get_durable("sheet_main", &mut buf);
    if n <= 0 {
        return;
    }
    let raw = core::str::from_utf8(&buf[..n as usize]).unwrap_or("");
    for (r, line) in raw.lines().enumerate().take(ROWS) {
        for (c, cell) in line.split('|').enumerate().take(COLS) {
            set_cell(c, r, cell);
        }
    }
}

pub fn start(_: &str) -> String {
    unsafe {
        CUR_C = 0;
        CUR_R = 0;
        ENTRY = [0; 16];
        ENTRY_LEN = 0;
        for i in 0..COLS * ROWS {
            CELL[i] = [0; 12];
            CELL_LEN[i] = 0;
        }
    }
    restore();
    paint();
    String::from("ok:sheets 4x6 (arrows move, type, enter commit; =A1+B1 formulas)")
}

pub fn on_click(x: i32, y: i32) -> String {
    if y < OY {
        return status("");
    }
    let c = ((x - OX) / CW).clamp(0, COLS as i32 - 1) as u8;
    let r = ((y - OY) / CH).clamp(0, ROWS as i32 - 1) as u8;
    unsafe {
        CUR_C = c;
        CUR_R = r;
    }
    paint();
    status("")
}

pub fn on_key(key: &str) -> String {
    match key {
        "left" => unsafe {
            CUR_C = CUR_C.saturating_sub(1);
        },
        "right" => unsafe {
            if (CUR_C as usize) + 1 < COLS {
                CUR_C += 1;
            }
        },
        "up" => unsafe {
            CUR_R = CUR_R.saturating_sub(1);
        },
        "down" => unsafe {
            if (CUR_R as usize) + 1 < ROWS {
                CUR_R += 1;
            }
        },
        "enter" => commit_entry(),
        "esc" => unsafe {
            ENTRY = [0; 16];
            ENTRY_LEN = 0;
        },
        "backspace" => unsafe {
            if ENTRY_LEN > 0 {
                ENTRY_LEN -= 1;
                ENTRY[ENTRY_LEN] = 0;
            }
        },
        k if k.len() == 1 => {
            let ch = k.chars().next().unwrap();
            if ch.is_ascii() && !ch.is_control() {
                unsafe {
                    if ENTRY_LEN < 15 {
                        ENTRY[ENTRY_LEN] = ch as u8;
                        ENTRY_LEN += 1;
                    }
                }
            }
        }
        _ => {}
    }
    paint();
    status("")
}

pub fn set(args: &str) -> String {
    let c = json_i32(args, "col", 0).clamp(0, COLS as i32 - 1) as usize;
    let r = json_i32(args, "row", 0).clamp(0, ROWS as i32 - 1) as usize;
    let v = json_str(args, "value")
        .or_else(|| json_str(args, "body"))
        .unwrap_or_default();
    set_cell(c, r, &v);
    persist();
    paint();
    format!("ok:cell {c},{r}={v}")
}

pub fn get(args: &str) -> String {
    let c = json_i32(args, "col", 0).clamp(0, COLS as i32 - 1) as usize;
    let r = json_i32(args, "row", 0).clamp(0, ROWS as i32 - 1) as usize;
    let raw = cell_str(c, r);
    let val = eval_cell(c, r, 0);
    format!("ok:raw={raw} value={val}")
}

pub fn status(_: &str) -> String {
    format!(
        "ok:sheets at {}{} entry_len={}",
        (b'A' + unsafe { CUR_C }) as char,
        unsafe { CUR_R } + 1,
        unsafe { ENTRY_LEN }
    )
}
