use crate::guest::{hud_status, json_i32, sound_play, text_op, ui_draw};
use alloc::format;
use alloc::string::{String, ToString};

/// One octave, C4..C5. White keys left→right; rows: (key char, frequency Hz).
const WHITE: [(char, i32); 8] = [
    ('a', 262), // C4
    ('s', 294), // D4
    ('d', 330), // E4
    ('f', 349), // F4
    ('g', 392), // G4
    ('h', 440), // A4
    ('j', 494), // B4
    ('k', 523), // C5
];
/// Black keys: (key char, frequency, white-key index whose right edge it sits on).
const BLACK: [(char, i32, i32); 5] = [
    ('w', 277, 0), // C#4
    ('e', 311, 1), // D#4
    ('t', 370, 3), // F#4
    ('y', 415, 4), // G#4
    ('u', 466, 5), // A#4
];
const NOTE_MS: i32 = 220;
/// Keyboard geometry on the 256×192 surface.
const KEY_W: i32 = 32;
const TOP: i32 = 16;
const WHITE_H: i32 = 160;
const BLACK_H: i32 = 96;
const BLACK_W: i32 = 20;

const SHORTCUTS: &str = "a s d f g h j k white  ·  w e t y u black  ·  click keys";

static mut LAST_NOTE: i32 = 0;

/// Paint the piano; `pressed` (a white index 0..8 or black index 8..13, -1 =
/// none) is tinted terracotta so a played note is visible.
fn paint(pressed: i32) {
    let mut ops = String::from("clear 1a1816; ");
    for i in 0..8i32 {
        let x = i * KEY_W;
        let c = if pressed == i { "cc785c" } else { "e8e4df" };
        ops.push_str(&format!("rect {} {TOP} {} {WHITE_H} {c}; ", x + 1, KEY_W - 2));
        let (ch, _) = WHITE[i as usize];
        let tc = if pressed == i { "e8e4df" } else { "1a1816" };
        text_op(&mut ops, x + 10, TOP + WHITE_H - 28, 12, tc, &ch.to_string());
    }
    for (bi, (ch, _, after)) in BLACK.iter().enumerate() {
        let x = (after + 1) * KEY_W - BLACK_W / 2;
        let c = if pressed == 8 + bi as i32 {
            "cc785c"
        } else {
            "2c2926"
        };
        ops.push_str(&format!("rect {x} {TOP} {BLACK_W} {BLACK_H} {c}; "));
        text_op(
            &mut ops,
            x + 4,
            TOP + 8,
            10,
            "e8e4df",
            &ch.to_string(),
        );
    }
    // Legend strip under the keys.
    ops.push_str("rect 0 0 256 12 3a3632; rect 0 180 256 12 3a3632; ");
    text_op(&mut ops, 8, 0, 10, "e8e4df", "Synth");
    text_op(&mut ops, 8, 180, 9, "a8a4a0", "a s d f g h j k  ·  w e t y u");
    ui_draw(&ops);
}

fn note_name(hz: i32) -> &'static str {
    match hz {
        262 => "C4",
        277 => "C#4",
        294 => "D4",
        311 => "D#4",
        330 => "E4",
        349 => "F4",
        370 => "F#4",
        392 => "G4",
        415 => "G#4",
        440 => "A4",
        466 => "A#4",
        494 => "B4",
        523 => "C5",
        _ => "note",
    }
}

fn refresh_hud(hz: i32) {
    let status = if hz > 0 {
        format!("synth  {}  {hz} Hz", note_name(hz))
    } else {
        String::from("synth  piano ready")
    };
    hud_status(&status, SHORTCUTS);
}

fn play(idx: i32, hz: i32) -> String {
    paint(idx);
    unsafe { LAST_NOTE = hz };
    refresh_hud(hz);
    match sound_play(hz, NOTE_MS) {
        0 => format!("ok:note {hz} Hz"),
        -1 => String::from("error:no sound device"),
        _ => String::from("error:play failed"),
    }
}

pub fn start(_: &str) -> String {
    unsafe { LAST_NOTE = 0 };
    paint(-1);
    refresh_hud(0);
    String::from("ok:synth piano (keys a-k white, w e t y u black; click keys too)")
}

/// Key: a s d f g h j k = white notes, w e t y u = black notes.
pub fn on_key(key: &str) -> String {
    let k = match key.chars().next() {
        Some(c) => c.to_ascii_lowercase(),
        None => return String::from("ok"),
    };
    for (i, (ch, hz)) in WHITE.iter().enumerate() {
        if *ch == k {
            return play(i as i32, *hz);
        }
    }
    for (bi, (ch, hz, _)) in BLACK.iter().enumerate() {
        if *ch == k {
            return play(8 + bi as i32, *hz);
        }
    }
    // Unhandled → bare "ok" so the shell keeps the key for chat typing.
    String::from("ok")
}

/// Click: black keys take priority in their upper zone, else the white key
/// under the pointer.
pub fn on_click(x: i32, y: i32) -> String {
    if y >= TOP && y < TOP + BLACK_H {
        for (bi, (_, hz, after)) in BLACK.iter().enumerate() {
            let bx = (after + 1) * KEY_W - BLACK_W / 2;
            if x >= bx && x < bx + BLACK_W {
                return play(8 + bi as i32, *hz);
            }
        }
    }
    if y >= TOP && y < TOP + WHITE_H {
        let i = (x / KEY_W).clamp(0, 7);
        return play(i, WHITE[i as usize].1);
    }
    String::from("ok:off-keys")
}

pub fn tone(args: &str) -> String {
    let hz = json_i32(args, "hz", 440).clamp(20, 4000);
    let ms = json_i32(args, "ms", 300).clamp(20, 5000);
    refresh_hud(hz);
    match sound_play(hz, ms) {
        0 => format!("ok:tone {hz} Hz {ms} ms"),
        -1 => String::from("error:no sound device"),
        _ => String::from("error:play failed"),
    }
}

pub fn beep(args: &str) -> String {
    let hz = json_i32(args, "hz", 880).clamp(20, 4000);
    refresh_hud(hz);
    match sound_play(hz, 120) {
        0 => format!("ok:beep {hz}"),
        -1 => String::from("error:no sound device"),
        _ => String::from("error:play failed"),
    }
}

pub fn stop(_: &str) -> String {
    refresh_hud(0);
    String::from("ok:stop (device drains)")
}

pub fn status(_: &str) -> String {
    format!("ok:synth last={} Hz", unsafe { LAST_NOTE })
}
