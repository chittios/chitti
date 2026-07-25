//! In-canvas win / lose / draw modal with a lightweight celebration animation.
//!
//! Drawn with the draw-op DSL only (no glyph text on the canvas — titles and
//! hints go through the HUD). Games call [`append`] after their normal paint,
//! then [`hit_restart`] / [`key_restart`] from click/key handlers.

use crate::guest::{now_ms, text_op};
use alloc::format;
use alloc::string::String;

/// Outcome flavour — drives panel accent colour.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Win,
    Lose,
    Draw,
}

/// Restart button geometry (canvas coords, 256×192).
pub const BTN_X: i32 = 56;
pub const BTN_Y: i32 = 118;
pub const BTN_W: i32 = 144;
pub const BTN_H: i32 = 28;

const COLORS: [&str; 5] = ["cc785c", "c9a54a", "6688cc", "5a8f5a", "e8e4df"];

/// True when a click lands on the restart button.
pub fn hit_restart(x: i32, y: i32) -> bool {
    x >= BTN_X && x < BTN_X + BTN_W && y >= BTN_Y && y < BTN_Y + BTN_H
}

/// True when a key should restart (n / r / enter / space).
pub fn key_restart(key: &str) -> bool {
    matches!(key, "n" | "r" | "enter" | "space")
}

/// Append a modal overlay + confetti to `ops` (caller then `ui_draw`s).
///
/// * `outcome` — win (terracotta), lose (red), draw (slate)
/// * `title` / `detail` — drawn as host text on the panel
/// * animation phase comes from the wall clock so idle games still sparkle
pub fn append(ops: &mut String, outcome: Outcome, title: &str, detail: &str) {
    let phase = (now_ms() / 55) as u32;
    let accent = match outcome {
        Outcome::Win => "cc785c",
        Outcome::Lose => "aa3333",
        Outcome::Draw => "6688cc",
    };
    // Dim the playfield with a dark veil (opaque — draw-ops have no alpha).
    ops.push_str("rect 0 0 256 192 0a0908; ");
    // Panel.
    ops.push_str("rect 28 32 200 128 3a3632; ");
    ops.push_str("rect 32 36 192 120 1a1816; ");
    // Accent top bar.
    ops.push_str(&format!("rect 32 36 192 6 {accent}; "));
    // Title + detail as real text.
    text_op(ops, 48, 52, 18, accent, title);
    if !detail.is_empty() {
        text_op(ops, 48, 78, 11, "a8a4a0", detail);
    }
    // Confetti burst (win) / ash (lose) / calm flakes (draw).
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
    // Restart button with label.
    ops.push_str(&format!(
        "rect {BTN_X} {BTN_Y} {BTN_W} {BTN_H} {accent}; "
    ));
    text_op(ops, BTN_X + 36, BTN_Y + 5, 14, "e8e4df", "Restart");
    // Button inner highlight pulse.
    if phase % 2 == 0 {
        ops.push_str(&format!(
            "rect {} {} {} 2 e8e4df; ",
            BTN_X + 4,
            BTN_Y + 2,
            BTN_W - 8
        ));
    }
    ops.push_str(&format!(
        "rect {} {} {} 2 0a0908; ",
        BTN_X + 4,
        BTN_Y + BTN_H - 4,
        BTN_W - 8
    ));
}

/// Convenience: build a full draw-op string that is *only* the overlay
/// (for games that paint the board first, then call `ui_draw` again).
pub fn overlay_ops(outcome: Outcome, title: &str, detail: &str) -> String {
    let mut ops = String::new();
    append(&mut ops, outcome, title, detail);
    ops
}
