//! 2048 — classic sliding puzzle on a 4×4 grid.

use crate::endscreen::{self, Outcome};
use crate::guest::{hud_status, ui_draw};
use alloc::format;
use alloc::string::String;

static mut GRID: [u16; 16] = [0; 16];
static mut SCORE: u32 = 0;
static mut RNG: u32 = 7;
static mut OVER: u8 = 0;

fn paint() {
    let mut ops = String::from("clear 1a1816; ");
    ops.push_str("rect 0 0 256 16 3a3632; ");
    let cell = 44;
    let ox = 40;
    let oy = 24;
    unsafe {
        for r in 0..4 {
            for c in 0..4 {
                let v = GRID[r * 4 + c];
                let x = ox + c as i32 * cell;
                let y = oy + r as i32 * cell;
                let color = match v {
                    0 => "2c2926",
                    2 => "5a5652",
                    4 => "6a6662",
                    8 => "cc785c",
                    16 => "c07050",
                    32 => "aa6030",
                    64 => "8a5a4a",
                    128 => "5a8f5a",
                    256 => "4a7f4a",
                    512 => "6688cc",
                    1024 => "5577bb",
                    _ => "e8e4df",
                };
                ops.push_str(&format!("rect {x} {y} 40 40 {color}; "));
            }
        }
        if OVER != 0 {
            ops.push_str("rect 40 80 176 32 aa3333; ");
        }
    }
    ops.push_str("rect 0 176 256 16 3a3632; ");
    let over = unsafe { OVER != 0 };
    let best = unsafe {
        let mut b = 0u16;
        for i in 0..16 {
            if GRID[i] > b {
                b = GRID[i];
            }
        }
        b
    };
    // Full modal only when no moves left. Hitting 2048 is celebrated in the HUD.
    if over {
        endscreen::append(
            &mut ops,
            if best >= 2048 {
                Outcome::Win
            } else {
                Outcome::Lose
            },
            if best >= 2048 { "YOU WIN" } else { "GAME OVER" },
            "2048",
        );
    }
    ui_draw(&ops);
    let status = unsafe {
        if OVER != 0 {
            if best >= 2048 {
                format!("2048  YOU WIN  score={SCORE}")
            } else {
                format!("2048  GAME OVER  score={SCORE}")
            }
        } else if best >= 2048 {
            format!("2048  score={SCORE}  best={best}  (keep going!)")
        } else {
            format!("2048  score={SCORE}  best={best}")
        }
    };
    let hints = if over {
        "enter / n / r  restart"
    } else {
        "arrows/wasd slide  n new"
    };
    hud_status(&status, hints);
}

/// Tick: re-paint when ended so confetti keeps animating.
pub fn tick_anim(_: &str) -> String {
    if unsafe { OVER != 0 } {
        paint();
        String::from("ok:anim")
    } else {
        String::from("ok")
    }
}

fn spawn() {
    unsafe {
        let mut empty = [0u8; 16];
        let mut n = 0usize;
        for i in 0..16 {
            if GRID[i] == 0 {
                empty[n] = i as u8;
                n += 1;
            }
        }
        if n == 0 {
            return;
        }
        RNG = RNG.wrapping_mul(1103515245).wrapping_add(12345);
        let slot = empty[(RNG as usize) % n] as usize;
        GRID[slot] = if (RNG >> 16) % 10 == 0 { 4 } else { 2 };
    }
}

fn slide_line(line: &mut [u16; 4]) -> bool {
    let mut moved = false;
    // compact
    let mut tmp = [0u16; 4];
    let mut t = 0;
    for &v in line.iter() {
        if v != 0 {
            tmp[t] = v;
            t += 1;
        }
    }
    // merge
    let mut i = 0;
    while i + 1 < t {
        if tmp[i] == tmp[i + 1] {
            tmp[i] *= 2;
            unsafe { SCORE += tmp[i] as u32 };
            tmp[i + 1] = 0;
            moved = true;
            i += 2;
        } else {
            i += 1;
        }
    }
    // compact again
    let mut out = [0u16; 4];
    let mut o = 0;
    for &v in &tmp {
        if v != 0 {
            out[o] = v;
            o += 1;
        }
    }
    for i in 0..4 {
        if line[i] != out[i] {
            moved = true;
        }
        line[i] = out[i];
    }
    moved
}

fn move_dir(dir: u8) -> bool {
    // 0L 1R 2U 3D
    let mut moved = false;
    unsafe {
        match dir {
            0 => {
                for r in 0..4 {
                    let mut line = [GRID[r * 4], GRID[r * 4 + 1], GRID[r * 4 + 2], GRID[r * 4 + 3]];
                    if slide_line(&mut line) {
                        moved = true;
                    }
                    for c in 0..4 {
                        GRID[r * 4 + c] = line[c];
                    }
                }
            }
            1 => {
                for r in 0..4 {
                    let mut line = [GRID[r * 4 + 3], GRID[r * 4 + 2], GRID[r * 4 + 1], GRID[r * 4]];
                    if slide_line(&mut line) {
                        moved = true;
                    }
                    for c in 0..4 {
                        GRID[r * 4 + (3 - c)] = line[c];
                    }
                }
            }
            2 => {
                for c in 0..4 {
                    let mut line = [GRID[c], GRID[4 + c], GRID[8 + c], GRID[12 + c]];
                    if slide_line(&mut line) {
                        moved = true;
                    }
                    for r in 0..4 {
                        GRID[r * 4 + c] = line[r];
                    }
                }
            }
            _ => {
                for c in 0..4 {
                    let mut line = [GRID[12 + c], GRID[8 + c], GRID[4 + c], GRID[c]];
                    if slide_line(&mut line) {
                        moved = true;
                    }
                    for r in 0..4 {
                        GRID[(3 - r) * 4 + c] = line[r];
                    }
                }
            }
        }
    }
    moved
}

fn check_over() {
    unsafe {
        for i in 0..16 {
            if GRID[i] == 0 {
                OVER = 0;
                return;
            }
        }
        for r in 0..4 {
            for c in 0..4 {
                let v = GRID[r * 4 + c];
                if c + 1 < 4 && GRID[r * 4 + c + 1] == v {
                    OVER = 0;
                    return;
                }
                if r + 1 < 4 && GRID[(r + 1) * 4 + c] == v {
                    OVER = 0;
                    return;
                }
            }
        }
        OVER = 1;
    }
}

pub fn start(_: &str) -> String {
    unsafe {
        GRID = [0; 16];
        SCORE = 0;
        OVER = 0;
        RNG = 42;
    }
    spawn();
    spawn();
    paint();
    String::from("ok:2048 (arrows move, n new)")
}

pub fn on_click(x: i32, y: i32) -> String {
    if unsafe { OVER != 0 } && endscreen::hit_restart(x, y) {
        return start("");
    }
    status("")
}

pub fn on_key(key: &str) -> String {
    if unsafe { OVER != 0 } {
        if endscreen::key_restart(key) {
            return start("");
        }
        return String::from("ok:ended");
    }
    if key == "n" {
        return start("");
    }
    let dir = match key {
        "left" | "h" | "a" => 0,
        "right" | "l" | "d" => 1,
        "up" | "k" | "w" => 2,
        "down" | "j" | "s" => 3,
        _ => return String::from("ok"), // unhandled → shell keeps the key
    };
    if move_dir(dir) {
        spawn();
        check_over();
    }
    paint();
    status("")
}

pub fn status(_: &str) -> String {
    format!("ok:2048 score={} over={}", unsafe { SCORE }, unsafe { OVER })
}
