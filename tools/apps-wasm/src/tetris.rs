//! Tetris — 10×16 board, 7 piece types, line clear.

use crate::guest::ui_draw;
use alloc::format;
use alloc::string::String;

const COLS: usize = 10;
const ROWS: usize = 16;
const CELL: i32 = 11;
const OX: i32 = 40;
const OY: i32 = 8;

/// 7 tetrominoes as 4 rotations × 4 cells (r,c offsets).
const SHAPES: [[[(i8, i8); 4]; 4]; 7] = [
    // I
    [
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        [(0, 1), (1, 1), (2, 1), (3, 1)],
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        [(0, 1), (1, 1), (2, 1), (3, 1)],
    ],
    // O
    [
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    ],
    // T
    [
        [(0, 1), (1, 0), (1, 1), (1, 2)],
        [(0, 1), (1, 1), (1, 2), (2, 1)],
        [(1, 0), (1, 1), (1, 2), (2, 1)],
        [(0, 1), (1, 0), (1, 1), (2, 1)],
    ],
    // S
    [
        [(0, 1), (0, 2), (1, 0), (1, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
        [(0, 1), (0, 2), (1, 0), (1, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
    ],
    // Z
    [
        [(0, 0), (0, 1), (1, 1), (1, 2)],
        [(0, 1), (1, 0), (1, 1), (2, 0)],
        [(0, 0), (0, 1), (1, 1), (1, 2)],
        [(0, 1), (1, 0), (1, 1), (2, 0)],
    ],
    // J
    [
        [(0, 0), (1, 0), (1, 1), (1, 2)],
        [(0, 1), (0, 2), (1, 1), (2, 1)],
        [(1, 0), (1, 1), (1, 2), (2, 2)],
        [(0, 1), (1, 1), (2, 0), (2, 1)],
    ],
    // L
    [
        [(0, 2), (1, 0), (1, 1), (1, 2)],
        [(0, 1), (1, 1), (2, 1), (2, 2)],
        [(1, 0), (1, 1), (1, 2), (2, 0)],
        [(0, 0), (0, 1), (1, 1), (2, 1)],
    ],
];

const COLORS: [&str; 7] = [
    "6688cc", "e8c060", "cc785c", "5a8f5a", "aa3333", "5577bb", "c07050",
];

static mut BOARD: [u8; COLS * ROWS] = [0; COLS * ROWS];
static mut KIND: u8 = 0;
static mut ROT: u8 = 0;
static mut PX: i8 = 3;
static mut PY: i8 = 0;
static mut SCORE: u32 = 0;
static mut OVER: u8 = 0;
static mut RNG: u32 = 11;
static mut TICK_N: u8 = 0;

fn cell(r: usize, c: usize) -> u8 {
    unsafe { BOARD[r * COLS + c] }
}

fn set_cell(r: usize, c: usize, v: u8) {
    unsafe { BOARD[r * COLS + c] = v };
}

fn cells_for(kind: u8, rot: u8, x: i8, y: i8) -> [(i8, i8); 4] {
    let mut out = [(0i8, 0i8); 4];
    for (i, (dr, dc)) in SHAPES[kind as usize][rot as usize].iter().enumerate() {
        out[i] = (y + dr, x + dc);
    }
    out
}

fn fits(kind: u8, rot: u8, x: i8, y: i8) -> bool {
    for (r, c) in cells_for(kind, rot, x, y) {
        if c < 0 || c >= COLS as i8 || r >= ROWS as i8 {
            return false;
        }
        if r >= 0 && cell(r as usize, c as usize) != 0 {
            return false;
        }
    }
    true
}

fn lock() {
    unsafe {
        let k = KIND + 1; // 1..=7 on board
        for (r, c) in cells_for(KIND, ROT, PX, PY) {
            if r >= 0 {
                set_cell(r as usize, c as usize, k);
            }
        }
        // clear lines
        let mut write = ROWS;
        for read in (0..ROWS).rev() {
            let full = (0..COLS).all(|c| cell(read, c) != 0);
            if !full {
                write -= 1;
                if write != read {
                    for c in 0..COLS {
                        set_cell(write, c, cell(read, c));
                    }
                }
            } else {
                SCORE += 100;
            }
        }
        while write > 0 {
            write -= 1;
            for c in 0..COLS {
                set_cell(write, c, 0);
            }
        }
        spawn();
    }
}

fn spawn() {
    unsafe {
        RNG = RNG.wrapping_mul(1103515245).wrapping_add(12345);
        KIND = ((RNG >> 16) % 7) as u8;
        ROT = 0;
        PX = 3;
        PY = 0;
        if !fits(KIND, ROT, PX, PY) {
            OVER = 1;
        }
    }
}

fn paint() {
    let mut ops = String::from("clear 1a1816; ");
    ops.push_str("rect 0 0 256 16 3a3632; ");
    // Well.
    ops.push_str(&format!(
        "rect {} {} {} {} 2c2926; ",
        OX - 2,
        OY - 2,
        COLS as i32 * CELL + 4,
        ROWS as i32 * CELL + 4
    ));
    unsafe {
        for r in 0..ROWS {
            for c in 0..COLS {
                let v = cell(r, c);
                if v == 0 {
                    continue;
                }
                let x = OX + c as i32 * CELL;
                let y = OY + r as i32 * CELL;
                let color = COLORS[(v - 1) as usize % 7];
                ops.push_str(&format!("rect {x} {y} {} {} {color}; ", CELL - 1, CELL - 1));
            }
        }
        if OVER == 0 {
            let color = COLORS[KIND as usize];
            for (r, c) in cells_for(KIND, ROT, PX, PY) {
                if r < 0 {
                    continue;
                }
                let x = OX + c as i32 * CELL;
                let y = OY + r as i32 * CELL;
                ops.push_str(&format!("rect {x} {y} {} {} {color}; ", CELL - 1, CELL - 1));
            }
        } else {
            ops.push_str("rect 48 80 160 28 aa3333; ");
        }
    }
    ops.push_str("rect 0 184 256 8 3a3632; ");
    ui_draw(&ops);
}

pub fn start(_: &str) -> String {
    unsafe {
        BOARD = [0; COLS * ROWS];
        SCORE = 0;
        OVER = 0;
        TICK_N = 0;
        RNG = 42;
    }
    spawn();
    paint();
    String::from("ok:tetris (←/→ move, ↑ rotate, ↓ soft drop, space hard, n new)")
}

pub fn on_click(_x: i32, _y: i32) -> String {
    status("")
}

pub fn on_key(key: &str) -> String {
    if unsafe { OVER != 0 } {
        if key == "n" {
            return start("");
        }
        return status("");
    }
    match key {
        "left" => unsafe {
            if fits(KIND, ROT, PX - 1, PY) {
                PX -= 1;
            }
        },
        "right" => unsafe {
            if fits(KIND, ROT, PX + 1, PY) {
                PX += 1;
            }
        },
        "up" => unsafe {
            let nr = (ROT + 1) % 4;
            if fits(KIND, nr, PX, PY) {
                ROT = nr;
            }
        },
        "down" => unsafe {
            if fits(KIND, ROT, PX, PY + 1) {
                PY += 1;
                SCORE += 1;
            } else {
                lock();
            }
        },
        "space" | "enter" => unsafe {
            while fits(KIND, ROT, PX, PY + 1) {
                PY += 1;
                SCORE += 2;
            }
            lock();
        },
        "n" => return start(""),
        _ => {}
    }
    paint();
    status("")
}

pub fn tick(_: &str) -> String {
    unsafe {
        if OVER != 0 {
            return String::from("ok:idle");
        }
        TICK_N = TICK_N.wrapping_add(1);
        if TICK_N % 8 != 0 {
            return String::from("ok:wait");
        }
        if fits(KIND, ROT, PX, PY + 1) {
            PY += 1;
        } else {
            lock();
        }
    }
    paint();
    status("")
}

pub fn status(_: &str) -> String {
    format!("ok:tetris score={} over={}", unsafe { SCORE }, unsafe { OVER })
}
