//! In-canvas win / lose / draw modal (mirrors apps-wasm endscreen).

use alloc::format;
use alloc::string::String;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Win,
    Lose,
    Draw,
}

pub const BTN_X: i32 = 56;
pub const BTN_Y: i32 = 118;
pub const BTN_W: i32 = 144;
pub const BTN_H: i32 = 28;

const COLORS: [&str; 5] = ["cc785c", "c9a54a", "6688cc", "5a8f5a", "e8e4df"];

pub fn hit_restart(x: i32, y: i32) -> bool {
    x >= BTN_X && x < BTN_X + BTN_W && y >= BTN_Y && y < BTN_Y + BTN_H
}

pub fn key_restart(key: &str) -> bool {
    matches!(key, "n" | "r" | "enter" | "space")
}

pub fn append(ops: &mut String, outcome: Outcome) {
    let phase = (crate::now_ms() / 55) as u32;
    let accent = match outcome {
        Outcome::Win => "cc785c",
        Outcome::Lose => "aa3333",
        Outcome::Draw => "6688cc",
    };
    ops.push_str("rect 0 0 256 192 0a0908; ");
    ops.push_str("rect 28 32 200 128 3a3632; ");
    ops.push_str("rect 32 36 192 120 1a1816; ");
    ops.push_str(&format!("rect 32 36 192 6 {accent}; "));
    match outcome {
        Outcome::Win => {
            for i in 0..5 {
                let h = 10 + i * 6;
                let x = 56 + i * 28;
                let y = 100 - h;
                ops.push_str(&format!("rect {x} {y} 18 {h} {accent}; "));
            }
            ops.push_str("rect 112 52 32 8 c9a54a; rect 120 44 16 8 c9a54a; ");
        }
        Outcome::Lose => {
            for i in 0..8 {
                let o = i * 6;
                ops.push_str(&format!(
                    "rect {} {} 10 6 aa3333; rect {} {} 10 6 aa3333; ",
                    88 + o,
                    52 + o,
                    158 - o,
                    52 + o
                ));
            }
        }
        Outcome::Draw => {
            ops.push_str("rect 72 64 112 10 6688cc; rect 72 86 112 10 6688cc; ");
        }
    }
    let n = match outcome {
        Outcome::Win => 20,
        Outcome::Lose => 10,
        Outcome::Draw => 12,
    };
    for i in 0..n {
        let seed = i as u32 * 41 + phase * 3;
        let x = 36 + ((seed.wrapping_mul(17)) % 184) as i32;
        let y = 40 + ((seed.wrapping_mul(13) + phase * 7) % 90) as i32;
        let c = COLORS[(i as usize + (phase as usize / 2)) % COLORS.len()];
        let w = 2 + (i % 3) as i32;
        let h = 2 + ((i + 1) % 3) as i32;
        ops.push_str(&format!("rect {x} {y} {w} {h} {c}; "));
    }
    ops.push_str(&format!("rect {BTN_X} {BTN_Y} {BTN_W} {BTN_H} {accent}; "));
    if phase % 2 == 0 {
        ops.push_str(&format!(
            "rect {} {} {} 2 e8e4df; ",
            BTN_X + 4,
            BTN_Y + 2,
            BTN_W - 8
        ));
    }
}
