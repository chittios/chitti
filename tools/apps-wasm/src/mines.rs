use crate::endscreen::{self, Outcome};
use crate::guest::{hud_status, json_i32, text_op, ui_draw};
use alloc::format;
use alloc::string::String;

const W: usize = 9;
const H: usize = 9;
const MINES: usize = 10;
const CELL: i32 = 20;
const OX: i32 = 16;
const OY: i32 = 6;

static mut MINE: [[u8; W]; H] = [[0; W]; H];
static mut OPEN: [[u8; W]; H] = [[0; W]; H];
static mut FLAG: [[u8; W]; H] = [[0; W]; H];
static mut DEAD: u8 = 0;
static mut WON: u8 = 0;
static mut SEEDED: u8 = 0;
static mut RNG: u32 = 1;
/// Keyboard cursor (row, col) — arrows move it, Enter opens, `f` flags.
static mut CUR: (u8, u8) = (4, 4);

fn paint() {
    let mut ops = String::from("clear 1a1816; ");
    unsafe {
        for r in 0..H {
            for c in 0..W {
                let x = OX + c as i32 * CELL;
                let y = OY + r as i32 * CELL;
                let color = if DEAD != 0 && MINE[r][c] != 0 {
                    "aa3333"
                } else if OPEN[r][c] != 0 {
                    if MINE[r][c] != 0 {
                        "aa3333"
                    } else {
                        match adj(r, c) {
                            0 => "3a3632",
                            1 => "4a6a8a",
                            2 => "4a8a5a",
                            3 => "8a5a4a",
                            _ => "6a5a7a",
                        }
                    }
                } else if FLAG[r][c] != 0 {
                    "cc785c"
                } else {
                    "5a5652"
                };
                ops.push_str(&format!("rect {x} {y} {} {} {color}; ", CELL - 1, CELL - 1));
                // Number / flag / mine glyph on the cell.
                if OPEN[r][c] != 0 {
                    if MINE[r][c] != 0 {
                        text_op(&mut ops, x + 5, y + 2, 12, "e8e4df", "*");
                    } else {
                        let n = adj(r, c);
                        if n > 0 {
                            text_op(&mut ops, x + 5, y + 2, 12, "e8e4df", &format!("{n}"));
                        }
                    }
                } else if FLAG[r][c] != 0 {
                    text_op(&mut ops, x + 5, y + 2, 12, "1a1816", "F");
                }
            }
        }
        // Keyboard cursor: a bright 2px frame around the current cell.
        let (cr, cc) = CUR;
        let x = OX + cc as i32 * CELL;
        let y = OY + cr as i32 * CELL;
        let s = CELL - 1;
        ops.push_str(&format!(
            "rect {x} {y} {s} 2 e8e4df; rect {x} {} {s} 2 e8e4df; rect {x} {y} 2 {s} e8e4df; rect {} {y} 2 {s} e8e4df; ",
            y + s - 2,
            x + s - 2
        ));
    }
    let (won, dead) = unsafe { (WON != 0, DEAD != 0) };
    if won {
        endscreen::append(&mut ops, Outcome::Win, "YOU WIN", "all clear");
    } else if dead {
        endscreen::append(&mut ops, Outcome::Lose, "BOOM", "stepped on a mine");
    }
    ui_draw(&ops);
    let status = unsafe {
        if WON != 0 {
            String::from("mines  YOU WIN!")
        } else if DEAD != 0 {
            String::from("mines  BOOM")
        } else {
            let (r, c) = CUR;
            format!("mines  {W}×{H}  {MINES} mines  cursor ({r},{c})")
        }
    };
    let hints = if won || dead {
        "enter / n / r  restart"
    } else {
        "arrows move  enter open  f flag  r restart"
    };
    hud_status(&status, hints);
}

/// Tick: re-paint when ended so confetti keeps animating.
pub fn tick_anim(_: &str) -> String {
    if unsafe { DEAD != 0 || WON != 0 } {
        paint();
        String::from("ok:anim")
    } else {
        String::from("ok")
    }
}

fn adj(r: usize, c: usize) -> u8 {
    let mut n = 0u8;
    unsafe {
        for dr in -1i32..=1 {
            for dc in -1i32..=1 {
                if dr == 0 && dc == 0 {
                    continue;
                }
                let rr = r as i32 + dr;
                let cc = c as i32 + dc;
                if rr >= 0
                    && cc >= 0
                    && (rr as usize) < H
                    && (cc as usize) < W
                    && MINE[rr as usize][cc as usize] != 0
                {
                    n += 1;
                }
            }
        }
    }
    n
}

fn place(safe_r: usize, safe_c: usize) {
    unsafe {
        let mut placed = 0;
        while placed < MINES {
            RNG = RNG.wrapping_mul(1103515245).wrapping_add(12345);
            let r = (RNG as usize / 7) % H;
            let c = (RNG as usize / 13) % W;
            if (r == safe_r && c == safe_c) || MINE[r][c] != 0 {
                continue;
            }
            if r.abs_diff(safe_r) <= 1 && c.abs_diff(safe_c) <= 1 {
                continue;
            }
            MINE[r][c] = 1;
            placed += 1;
        }
        SEEDED = 1;
    }
}

fn flood(r: usize, c: usize) {
    unsafe {
        if r >= H || c >= W || OPEN[r][c] != 0 || FLAG[r][c] != 0 {
            return;
        }
        OPEN[r][c] = 1;
        if MINE[r][c] != 0 {
            return;
        }
        if adj(r, c) == 0 {
            for dr in -1i32..=1 {
                for dc in -1i32..=1 {
                    if dr == 0 && dc == 0 {
                        continue;
                    }
                    let rr = r as i32 + dr;
                    let cc = c as i32 + dc;
                    if rr >= 0 && cc >= 0 && (rr as usize) < H && (cc as usize) < W {
                        flood(rr as usize, cc as usize);
                    }
                }
            }
        }
    }
}

fn check_win() {
    unsafe {
        let mut closed = 0;
        for r in 0..H {
            for c in 0..W {
                if OPEN[r][c] == 0 {
                    closed += 1;
                }
            }
        }
        if closed == MINES {
            WON = 1;
        }
    }
}

pub fn start(_: &str) -> String {
    unsafe {
        MINE = [[0; W]; H];
        OPEN = [[0; W]; H];
        FLAG = [[0; W]; H];
        DEAD = 0;
        WON = 0;
        SEEDED = 0;
        RNG = 1;
        CUR = (4, 4);
    }
    paint();
    format!("ok:mines {W}x{H} n={MINES}")
}

/// Open cell (r, c) — the shared core of a chat `mines_click`, a surface click,
/// and Enter on the keyboard cursor.
fn open_at(r: usize, c: usize) -> String {
    if r >= H || c >= W {
        return String::from("error:row/col 0..8");
    }
    unsafe {
        if DEAD != 0 || WON != 0 || FLAG[r][c] != 0 {
            return String::from("error:blocked");
        }
        if SEEDED == 0 {
            place(r, c);
        }
        if MINE[r][c] != 0 {
            OPEN[r][c] = 1;
            DEAD = 1;
            paint();
            return String::from("ok:BOOM");
        }
        flood(r, c);
        check_win();
        paint();
        if WON != 0 {
            String::from("ok:WIN")
        } else {
            format!("ok:open {r} {c}")
        }
    }
}

fn flag_at(r: usize, c: usize) -> String {
    if r >= H || c >= W {
        return String::from("error:row/col 0..8");
    }
    unsafe {
        if DEAD != 0 || WON != 0 || OPEN[r][c] != 0 {
            return String::from("error:blocked");
        }
        FLAG[r][c] = if FLAG[r][c] == 0 { 1 } else { 0 };
        paint();
        format!("ok:flag {r} {c}")
    }
}

pub fn click(args: &str) -> String {
    let r = json_i32(args, "row", json_i32(args, "r", 99)) as usize;
    let c = json_i32(args, "col", json_i32(args, "c", 99)) as usize;
    open_at(r, c)
}

pub fn flag(args: &str) -> String {
    let r = json_i32(args, "row", json_i32(args, "r", 99)) as usize;
    let c = json_i32(args, "col", json_i32(args, "c", 99)) as usize;
    flag_at(r, c)
}

/// Surface click at pixel (x, y): map to a cell, move the cursor there, open.
/// When the game is over, the modal restart button starts a new game.
pub fn on_click(x: i32, y: i32) -> String {
    if unsafe { DEAD != 0 || WON != 0 } {
        if endscreen::hit_restart(x, y) {
            return start("");
        }
        return String::from("ok:ended");
    }
    let c = (x - OX) / CELL;
    let r = (y - OY) / CELL;
    if x < OX || y < OY || c < 0 || r < 0 || c as usize >= W || r as usize >= H {
        return String::from("ok:off-board");
    }
    unsafe { CUR = (r as u8, c as u8) };
    open_at(r as usize, c as usize)
}

/// Key: arrows move the cursor, Enter/space opens, `f` flags, `r` restarts.
pub fn on_key(key: &str) -> String {
    if unsafe { DEAD != 0 || WON != 0 } {
        if endscreen::key_restart(key) {
            return start("");
        }
        return String::from("ok:ended");
    }
    let (dr, dc): (i32, i32) = match key {
        "up" => (-1, 0),
        "down" => (1, 0),
        "left" => (0, -1),
        "right" => (0, 1),
        "enter" | "space" => {
            let (r, c) = unsafe { CUR };
            return open_at(r as usize, c as usize);
        }
        "f" => {
            let (r, c) = unsafe { CUR };
            return flag_at(r as usize, c as usize);
        }
        "r" => return start(""),
        _ => return String::from("ok"),
    };
    unsafe {
        let r = (CUR.0 as i32 + dr).clamp(0, H as i32 - 1) as u8;
        let c = (CUR.1 as i32 + dc).clamp(0, W as i32 - 1) as u8;
        CUR = (r, c);
    }
    paint();
    String::from("ok:cursor")
}

pub fn status(_: &str) -> String {
    unsafe { format!("ok:dead={} won={} seeded={}", DEAD, WON, SEEDED) }
}
