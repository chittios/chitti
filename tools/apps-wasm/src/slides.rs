use crate::guest::{hud_status, ui_draw};
use alloc::format;
use alloc::string::String;

static mut IDX: usize = 0;

const SLIDES: &[(&str, &str, &str)] = &[
    ("ChittiOS", "Agentic operating system", "cc785c"),
    ("Agents", "notes paint slides games synth", "6688cc"),
    ("WASM", "Logic in tools.wasm — not the kernel", "5a8f5a"),
    ("Thanks", "slides_next / slides_prev", "8a6a4a"),
];
const SHORTCUTS: &str = "← prev  →/space/enter next  r restart  ·  click L/R";

fn paint() {
    let n = SLIDES.len();
    let i = unsafe { IDX.min(n - 1) };
    let (title, body, color) = SLIDES[i];
    let mut ops = format!("clear 1a1816; rect 0 0 256 48 {color}; rect 16 64 224 96 {color}; ");
    let slot = 256 / n as i32;
    for k in 0..n {
        let x = k as i32 * slot + 4;
        let c = if k == i { "e8e4df" } else { "3a3632" };
        ops.push_str(&format!("rect {x} 176 {} 8 {c}; ", (slot - 8).max(4)));
    }
    let _ = (title, body); // titles shown via host log optional
    ui_draw(&ops);
    let (t, b, _) = SLIDES[i];
    hud_status(
        &format!("slides  {}/{}  {t} — {b}", i + 1, n),
        SHORTCUTS,
    );
}

pub fn start(_: &str) -> String {
    unsafe { IDX = 0 };
    paint();
    format!("ok:slides n={}", SLIDES.len())
}

pub fn next(_: &str) -> String {
    unsafe {
        if IDX + 1 < SLIDES.len() {
            IDX += 1;
        }
    }
    paint();
    format!("ok:slide {}", unsafe { IDX + 1 })
}

pub fn prev(_: &str) -> String {
    unsafe {
        IDX = IDX.saturating_sub(1);
    }
    paint();
    format!("ok:slide {}", unsafe { IDX + 1 })
}

pub fn goto(args: &str) -> String {
    let n = crate::guest::json_i32(args, "n", 1).max(1) as usize;
    if n > SLIDES.len() {
        return format!("error:range 1..{}", SLIDES.len());
    }
    unsafe { IDX = n - 1 };
    paint();
    format!("ok:slide {n}")
}

pub fn status(_: &str) -> String {
    let i = unsafe { IDX };
    let (t, b, _) = SLIDES[i.min(SLIDES.len() - 1)];
    format!("ok:slide {}/{} {t} — {b}", i + 1, SLIDES.len())
}

/// Click: left half of the surface goes back, right half advances.
pub fn on_click(x: i32, _y: i32) -> String {
    if x < 128 {
        prev("")
    } else {
        next("")
    }
}

/// Key: ←/→ navigate, space advances, `r` restarts from slide 1.
pub fn on_key(key: &str) -> String {
    match key {
        "left" | "h" => prev(""),
        "right" | "l" | "space" | "enter" => next(""),
        "r" => start(""),
        _ => String::from("ok"), // unhandled → shell keeps the key
    }
}
