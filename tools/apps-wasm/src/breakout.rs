//! Breakout — paddle, ball, brick rows.

use crate::endscreen::{self, Outcome};
use crate::guest::{hud_status, ui_draw};
use alloc::format;
use alloc::string::String;

const W: i32 = 256;
const H: i32 = 192;
const BRICK_COLS: usize = 8;
const BRICK_ROWS: usize = 4;
const BRICK_W: i32 = 28;
const BRICK_H: i32 = 10;
const PADDLE_W: i32 = 48;
const PADDLE_H: i32 = 8;

static mut BRICKS: [[u8; BRICK_COLS]; BRICK_ROWS] = [[1; BRICK_COLS]; BRICK_ROWS];
static mut PX: i32 = 104;
static mut BALL_X: i32 = 128;
static mut BALL_Y: i32 = 140;
static mut VX: i32 = 2;
static mut VY: i32 = -2;
static mut SCORE: u32 = 0;
static mut LIVES: u8 = 3;
static mut OVER: u8 = 0;
static mut ACTIVE: u8 = 0;

fn paint() {
    let mut ops = String::from("clear 1a1816; ");
    ops.push_str("rect 0 0 256 14 3a3632; ");
    unsafe {
        for r in 0..BRICK_ROWS {
            for c in 0..BRICK_COLS {
                if BRICKS[r][c] == 0 {
                    continue;
                }
                let x = 8 + c as i32 * (BRICK_W + 2);
                let y = 20 + r as i32 * (BRICK_H + 2);
                let color = match r {
                    0 => "cc785c",
                    1 => "c07050",
                    2 => "5a8f5a",
                    _ => "6688cc",
                };
                ops.push_str(&format!("rect {x} {y} {BRICK_W} {BRICK_H} {color}; "));
            }
        }
        // Paddle.
        ops.push_str(&format!(
            "rect {} 172 {PADDLE_W} {PADDLE_H} e8e4df; ",
            PX
        ));
        // Ball.
        ops.push_str(&format!("rect {} {} 6 6 cc785c; ", BALL_X, BALL_Y));
        if OVER != 0 {
            ops.push_str("rect 48 80 160 28 aa3333; ");
        }
    }
    ops.push_str("rect 0 184 256 8 3a3632; ");
    let over = unsafe { OVER != 0 };
    if over {
        endscreen::append(&mut ops, Outcome::Lose, "GAME OVER", "no lives left");
    }
    ui_draw(&ops);
    let status = unsafe {
        if OVER != 0 {
            format!("breakout  GAME OVER  score={SCORE}")
        } else {
            format!("breakout  score={SCORE}  lives={LIVES}")
        }
    };
    let hints = if over {
        "enter / n / r  restart"
    } else {
        "←/→ paddle  space launch  n new"
    };
    hud_status(&status, hints);
}

fn reset_ball() {
    unsafe {
        BALL_X = PX + PADDLE_W / 2;
        BALL_Y = 160;
        VX = 2;
        VY = -2;
    }
}

fn new_level() {
    unsafe {
        for r in 0..BRICK_ROWS {
            for c in 0..BRICK_COLS {
                BRICKS[r][c] = 1;
            }
        }
        OVER = 0;
        ACTIVE = 1;
        PX = 104;
        LIVES = 3;
        SCORE = 0;
        reset_ball();
    }
}

pub fn start(_: &str) -> String {
    new_level();
    paint();
    String::from("ok:breakout (←/→ paddle, space launch, n new)")
}

pub fn on_click(x: i32, y: i32) -> String {
    if unsafe { OVER != 0 } {
        if endscreen::hit_restart(x, y) {
            return start("");
        }
        return String::from("ok:ended");
    }
    unsafe {
        PX = (x - PADDLE_W / 2).clamp(0, W - PADDLE_W);
        if ACTIVE == 0 {
            ACTIVE = 1;
        }
    }
    paint();
    status("")
}

pub fn on_key(key: &str) -> String {
    if unsafe { OVER != 0 } {
        if endscreen::key_restart(key) {
            return start("");
        }
        return String::from("ok:ended");
    }
    match key {
        "left" | "a" | "h" => unsafe {
            PX = (PX - 12).max(0);
        },
        "right" | "d" | "l" => unsafe {
            PX = (PX + 12).min(W - PADDLE_W);
        },
        "space" | "enter" => unsafe {
            ACTIVE = 1;
        },
        "n" => return start(""),
        _ => return String::from("ok"), // unhandled → shell keeps the key
    }
    paint();
    status("")
}

pub fn tick(_: &str) -> String {
    unsafe {
        if OVER != 0 {
            paint(); // animate confetti
            return String::from("ok:ended");
        }
        if ACTIVE == 0 {
            return String::from("ok:idle");
        }
        BALL_X += VX;
        BALL_Y += VY;
        // Walls.
        if BALL_X <= 0 {
            BALL_X = 0;
            VX = -VX;
        }
        if BALL_X >= W - 6 {
            BALL_X = W - 6;
            VX = -VX;
        }
        if BALL_Y <= 14 {
            BALL_Y = 14;
            VY = -VY;
        }
        // Paddle.
        if BALL_Y + 6 >= 172
            && BALL_Y <= 172 + PADDLE_H
            && BALL_X + 6 >= PX
            && BALL_X <= PX + PADDLE_W
        {
            BALL_Y = 172 - 6;
            VY = -VY.abs();
            // English from hit position.
            let mid = PX + PADDLE_W / 2;
            if BALL_X < mid {
                VX = -VX.abs();
            } else {
                VX = VX.abs();
            }
        }
        // Bottom miss.
        if BALL_Y > H {
            if LIVES > 0 {
                LIVES -= 1;
            }
            if LIVES == 0 {
                OVER = 1;
                ACTIVE = 0;
            } else {
                reset_ball();
                ACTIVE = 0;
            }
        }
        // Bricks.
        for r in 0..BRICK_ROWS {
            for c in 0..BRICK_COLS {
                if BRICKS[r][c] == 0 {
                    continue;
                }
                let bx = 8 + c as i32 * (BRICK_W + 2);
                let by = 20 + r as i32 * (BRICK_H + 2);
                if BALL_X + 6 >= bx
                    && BALL_X <= bx + BRICK_W
                    && BALL_Y + 6 >= by
                    && BALL_Y <= by + BRICK_H
                {
                    BRICKS[r][c] = 0;
                    VY = -VY;
                    SCORE += 10;
                }
            }
        }
        // Win?
        let mut left = 0u32;
        for r in 0..BRICK_ROWS {
            for c in 0..BRICK_COLS {
                left += BRICKS[r][c] as u32;
            }
        }
        if left == 0 {
            OVER = 0;
            SCORE += 100;
            // refill
            for r in 0..BRICK_ROWS {
                for c in 0..BRICK_COLS {
                    BRICKS[r][c] = 1;
                }
            }
            reset_ball();
            ACTIVE = 0;
        }
    }
    paint();
    status("")
}

pub fn status(_: &str) -> String {
    format!(
        "ok:breakout score={} lives={} over={}",
        unsafe { SCORE },
        unsafe { LIVES },
        unsafe { OVER }
    )
}
