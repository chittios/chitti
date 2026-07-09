use crate::guest::{json_i32, sound_play};
use alloc::format;
use alloc::string::String;

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
