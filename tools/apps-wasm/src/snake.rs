use crate::guest::{json_str, ui_draw};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

const GW: usize = 16;
const GH: usize = 12;
const CELL: i32 = 14;
const OX: i32 = 16;
const OY: i32 = 8;

static mut BODY: [(u8, u8); 128] = [(0, 0); 128];
static mut LEN: usize = 0;
static mut DIR: u8 = 3; // 0U 1D 2L 3R
static mut PEND: i8 = -1;
static mut FOOD: (u8, u8) = (0, 0);
static mut DEAD: u8 = 0;
static mut SCORE: u32 = 0;
static mut RNG: u32 = 42;
static mut ACTIVE: u8 = 0;

fn paint() {
    let mut ops = String::from("clear 1a1816; ");
    unsafe {
        let (fr, fc) = FOOD;
        let fx = OX + fc as i32 * CELL;
        let fy = OY + fr as i32 * CELL;
        ops.push_str(&format!("rect {fx} {fy} {} {} cc785c; ", CELL - 1, CELL - 1));
        for i in 0..LEN {
            let (r, c) = BODY[i];
            let x = OX + c as i32 * CELL;
            let y = OY + r as i32 * CELL;
            let color = if i == 0 { "e8e4df" } else { "5a8f5a" };
            ops.push_str(&format!("rect {x} {y} {} {} {color}; ", CELL - 1, CELL - 1));
        }
        if DEAD != 0 {
            ops.push_str("rect 0 176 256 16 aa3333; ");
        }
    }
    ui_draw(&ops);
}

fn spawn_food() {
    unsafe {
        for _ in 0..64 {
            RNG = RNG.wrapping_mul(1103515245).wrapping_add(12345);
            let r = ((RNG / 3) as usize) % GH;
            let c = ((RNG / 11) as usize) % GW;
            let mut hit = false;
            for i in 0..LEN {
                if BODY[i] == (r as u8, c as u8) {
                    hit = true;
                    break;
                }
            }
            if !hit {
                FOOD = (r as u8, c as u8);
                return;
            }
        }
        FOOD = (0, 0);
    }
}

fn step_fixed() {
    unsafe {
        if DEAD != 0 || ACTIVE == 0 {
            return;
        }
        if PEND >= 0 {
            let d = PEND as u8;
            let bad = (DIR == 0 && d == 1)
                || (DIR == 1 && d == 0)
                || (DIR == 2 && d == 3)
                || (DIR == 3 && d == 2);
            if !bad {
                DIR = d;
            }
            PEND = -1;
        }
        let (hr, hc) = BODY[0];
        let (nr, nc) = match DIR {
            0 => (hr.wrapping_sub(1), hc),
            1 => (hr + 1, hc),
            2 => (hr, hc.wrapping_sub(1)),
            _ => (hr, hc + 1),
        };
        if (nr as usize) >= GH || (nc as usize) >= GW {
            DEAD = 1;
            return;
        }
        for i in 0..LEN {
            if BODY[i] == (nr, nc) {
                DEAD = 1;
                return;
            }
        }
        let grow = (nr, nc) == FOOD;
        let new_len = if grow { (LEN + 1).min(127) } else { LEN };
        // shift body right
        let mut i = new_len - 1;
        while i > 0 {
            BODY[i] = BODY[i - 1];
            i -= 1;
        }
        BODY[0] = (nr, nc);
        LEN = new_len;
        if grow {
            SCORE += 1;
            spawn_food();
        }
    }
}

pub fn start(_: &str) -> String {
    unsafe {
        BODY = [(0, 0); 128];
        BODY[0] = (6, 4);
        BODY[1] = (6, 3);
        BODY[2] = (6, 2);
        LEN = 3;
        DIR = 3;
        PEND = -1;
        DEAD = 0;
        SCORE = 0;
        RNG = 42;
        ACTIVE = 1;
        spawn_food();
    }
    paint();
    String::from("ok:snake")
}

pub fn dir(args: &str) -> String {
    let d = json_str(args, "dir")
        .or_else(|| json_str(args, "d"))
        .unwrap_or_default()
        .to_ascii_lowercase();
    let v = match d.as_str() {
        "up" | "u" | "w" => 0,
        "down" | "s" => 1,
        "left" | "l" | "h" => 2,
        "right" | "r" | "d" => 3,
        _ => return format!("error:dir '{d}'"),
    };
    unsafe {
        PEND = v as i8;
    }
    format!("ok:dir {d}")
}

pub fn tick_once(_: &str) -> String {
    unsafe {
        if ACTIVE == 0 {
            return String::from("ok:idle");
        }
    }
    step_fixed();
    paint();
    unsafe {
        if DEAD != 0 {
            format!("ok:dead score={SCORE}")
        } else {
            format!("ok:tick score={SCORE}")
        }
    }
}

pub fn status(_: &str) -> String {
    unsafe { format!("ok:score={SCORE} dead={DEAD} len={LEN}") }
}

/// Key: arrows/wasd steer, `r` restarts (also revives after death).
pub fn on_key(key: &str) -> String {
    let d: i8 = match key {
        "up" | "w" | "k" => 0,
        "down" | "s" | "j" => 1,
        "left" | "a" | "h" => 2,
        "right" | "d" | "l" => 3,
        "r" => return start(""),
        _ => return String::from("ok"),
    };
    unsafe {
        if DEAD != 0 {
            return String::from("ok:dead (r restarts)");
        }
        PEND = d;
    }
    format!("ok:dir {key}")
}
