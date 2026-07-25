//! Radio — station list + tone “playback” via host_sound_play.

use crate::guest::{
    hud_status, json_str, sound_play, storage_get_durable, storage_set_durable, text_op, ui_draw,
};
use alloc::format;
use alloc::string::{String, ToString};

const STATIONS: [(&str, i32); 6] = [
    ("Chitti FM", 440),
    ("Synapse Wave", 523),
    ("Cortex Jazz", 349),
    ("Bare Metal", 262),
    ("Delta Net", 392),
    ("Quiet Room", 0), // silence / off
];

static mut SEL: usize = 0;
static mut PLAYING: u8 = 0;
static mut VOL: u8 = 5;

fn paint() {
    let mut ops = String::from("clear 1a1816; ");
    ops.push_str("rect 0 0 256 16 3a3632; ");
    let (name, hz) = STATIONS[unsafe { SEL }.min(STATIONS.len() - 1)];
    let state = if unsafe { PLAYING } != 0 {
        "ON AIR"
    } else {
        "paused"
    };
    text_op(
        &mut ops,
        8,
        1,
        12,
        "e8e4df",
        &format!("Radio  {name}  {state}"),
    );
    // Dial.
    ops.push_str("rect 16 24 224 48 2c2926; ");
    let dial = 20 + (unsafe { SEL } as i32 * 34);
    ops.push_str(&format!("rect {dial} 36 24 24 cc785c; "));
    text_op(&mut ops, 24, 40, 14, "e8e4df", &format!("{hz} Hz"));
    // Station list.
    for (i, (name, hz)) in STATIONS.iter().enumerate() {
        let y = 84 + i as i32 * 14;
        let c = if i == unsafe { SEL } {
            if unsafe { PLAYING } != 0 {
                "5a8f5a"
            } else {
                "cc785c"
            }
        } else {
            "3a3632"
        };
        ops.push_str(&format!("rect 16 {y} 224 12 {c}; "));
        text_op(
            &mut ops,
            20,
            y + 1,
            10,
            "e8e4df",
            &format!("{name}  {hz}Hz"),
        );
    }
    // Volume.
    let vw = (unsafe { VOL } as i32 * 20).min(200);
    ops.push_str(&format!("rect 16 172 {vw} 8 6688cc; "));
    text_op(
        &mut ops,
        180,
        170,
        10,
        "a8a4a0",
        &format!("vol {}", unsafe { VOL }),
    );
    ui_draw(&ops);
    hud_status(
        &format!("radio  {name}  {state}  vol={}", unsafe { VOL }),
        "arrows station  space play  +/- volume",
    );
}

fn play_sel() {
    unsafe {
        let (_, hz) = STATIONS[SEL];
        if hz == 0 {
            PLAYING = 0;
            return;
        }
        PLAYING = 1;
        let ms = 120 + VOL as i32 * 40;
        let _ = sound_play(hz, ms);
    }
}

fn save() {
    let blob = format!(
        "{{\"sel\":{},\"playing\":{},\"vol\":{}}}",
        unsafe { SEL },
        unsafe { PLAYING },
        unsafe { VOL }
    );
    let _ = storage_set_durable("radio", &blob);
}

fn load() {
    let mut buf = [0u8; 64];
    let n = storage_get_durable("radio", &mut buf);
    if n <= 0 {
        return;
    }
    let raw = core::str::from_utf8(&buf[..n as usize]).unwrap_or("");
    if let Some(v) = json_str(raw, "sel") {
        if let Ok(n) = v.parse::<usize>() {
            unsafe { SEL = n.min(STATIONS.len() - 1) };
        }
    }
    if let Some(v) = json_str(raw, "vol") {
        if let Ok(n) = v.parse::<u8>() {
            unsafe { VOL = n.clamp(1, 10) };
        }
    }
}

pub fn start(_: &str) -> String {
    load();
    paint();
    String::from("ok:radio (↑↓ station, space play, +/- vol)")
}

pub fn on_click(_x: i32, y: i32) -> String {
    if y >= 84 {
        let i = ((y - 84) / 14) as usize;
        if i < STATIONS.len() {
            unsafe { SEL = i };
            play_sel();
            save();
            paint();
        }
    }
    status("")
}

pub fn on_key(key: &str) -> String {
    match key {
        "up" => unsafe {
            if SEL > 0 {
                SEL -= 1;
            }
        },
        "down" => unsafe {
            if SEL + 1 < STATIONS.len() {
                SEL += 1;
            }
        },
        "space" | "enter" => {
            if unsafe { PLAYING != 0 } {
                unsafe { PLAYING = 0 };
            } else {
                play_sel();
            }
        }
        "+" | "=" => unsafe {
            VOL = (VOL + 1).min(10);
            if PLAYING != 0 {
                play_sel();
            }
        },
        "-" | "_" => unsafe {
            VOL = VOL.saturating_sub(1).max(1);
            if PLAYING != 0 {
                play_sel();
            }
        },
        "s" => {
            // stop
            unsafe { PLAYING = 0 };
        }
        _ => {}
    }
    save();
    paint();
    status("")
}

pub fn tick(_: &str) -> String {
    // Soft re-trigger while "playing" so the station feels live.
    unsafe {
        if PLAYING != 0 {
            let (_, hz) = STATIONS[SEL];
            if hz > 0 {
                let _ = sound_play(hz, 80 + VOL as i32 * 10);
            }
            paint();
        }
    }
    String::from("ok:tick")
}

pub fn tune(args: &str) -> String {
    if let Some(s) = json_str(args, "station").or_else(|| json_str(args, "name")) {
        for (i, (name, _)) in STATIONS.iter().enumerate() {
            if name.eq_ignore_ascii_case(&s) || name.to_ascii_lowercase().contains(&s.to_ascii_lowercase()) {
                unsafe { SEL = i };
                play_sel();
                save();
                paint();
                return status("");
            }
        }
        return format!("error:unknown station '{s}'");
    }
    if let Some(i) = json_str(args, "index") {
        if let Ok(n) = i.parse::<usize>() {
            if n < STATIONS.len() {
                unsafe { SEL = n };
                play_sel();
                save();
                paint();
            }
        }
    }
    status("")
}

pub fn status(_: &str) -> String {
    let (name, hz) = STATIONS[unsafe { SEL }];
    format!(
        "ok:radio station={} hz={} playing={} vol={}",
        name,
        hz,
        unsafe { PLAYING },
        unsafe { VOL }
    )
}
