use crate::guest::{json_i32, sound_play, ui_draw};
use alloc::format;
use alloc::string::String;

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

/// Paint the piano; `pressed` (a white index 0..8 or black index 8..13, -1 =
/// none) is tinted terracotta so a played note is visible.
fn paint(pressed: i32) {
    let mut ops = String::from("clear 1a1816; ");
    for i in 0..8i32 {
        let x = i * KEY_W;
        let c = if pressed == i { "cc785c" } else { "e8e4df" };
        ops.push_str(&format!("rect {} {TOP} {} {WHITE_H} {c}; ", x + 1, KEY_W - 2));
    }
    for (bi, (_, _, after)) in BLACK.iter().enumerate() {
        let x = (after + 1) * KEY_W - BLACK_W / 2;
        let c = if pressed == 8 + bi as i32 { "cc785c" } else { "2c2926" };
        ops.push_str(&format!("rect {x} {TOP} {BLACK_W} {BLACK_H} {c}; "));
    }
    // Legend strip: the letter row a s d f g h j k (w e t y u above).
    ops.push_str("rect 0 0 256 12 3a3632; rect 0 180 256 12 3a3632; ");
    ui_draw(&ops);
}

fn play(idx: i32, hz: i32) -> String {
    paint(idx);
    match sound_play(hz, NOTE_MS) {
        0 => format!("ok:note {hz} Hz"),
        -1 => String::from("error:no sound device"),
        _ => String::from("error:play failed"),
    }
}

pub fn start(_: &str) -> String {
    paint(-1);
    String::from("ok:synth piano (keys a-k white, w e t y u black; click keys too)")
}

/// Key: a s d f g h j k = white notes, w e t y u = black notes.
pub fn on_key(key: &str) -> String {
    let k = match key.chars().next() {
        Some(c) => c,
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
    match sound_play(hz, ms) {
        0 => format!("ok:tone {hz} Hz {ms} ms"),
        -1 => String::from("error:no sound device"),
        _ => String::from("error:play failed"),
    }
}

pub fn beep(args: &str) -> String {
    let hz = json_i32(args, "hz", 880).clamp(20, 4000);
    match sound_play(hz, 120) {
        0 => format!("ok:beep {hz}"),
        -1 => String::from("error:no sound device"),
        _ => String::from("error:play failed"),
    }
}

pub fn stop(_: &str) -> String {
    String::from("ok:stop (device drains)")
}

pub fn status(_: &str) -> String {
    String::from("ok:synth ready")
}
