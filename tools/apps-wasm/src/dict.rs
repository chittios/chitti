//! Dict — tiny offline glossary + durable custom entries.

use crate::guest::{json_str, storage_get_durable, storage_set_durable, ui_draw};
use alloc::format;
use alloc::string::{String, ToString};

const BUILTIN: &[(&str, &str)] = &[
    ("agent", "A reasoning process with a SOUL and capabilities"),
    ("cap", "Unforgeable capability token for an effect"),
    ("synapse", "Deterministic ABI executor below the model"),
    ("wasm", "Deterministic package tools module"),
    ("soul", "Markdown persona for an installed agent"),
    ("taint", "Provenance tag on context tokens"),
    ("kernel", "Bare-metal ChittiOS core"),
    ("chitti", "Agentic operating system project"),
];

static mut QUERY: [u8; 32] = [0; 32];
static mut QLEN: usize = 0;
static mut HIT: i32 = -1;

fn paint() {
    let mut ops = String::from("clear 1a1816; ");
    ops.push_str("rect 0 0 256 16 3a3632; ");
    // Query bar.
    ops.push_str("rect 8 24 240 20 2c2926; ");
    let qw = (unsafe { QLEN } as i32 * 6).min(230);
    ops.push_str(&format!("rect 12 28 {qw} 12 cc785c; "));
    // Results list.
    for i in 0..BUILTIN.len().min(8) {
        let y = 56 + i as i32 * 14;
        let c = if unsafe { HIT } == i as i32 {
            "5a8f5a"
        } else {
            "3a3632"
        };
        ops.push_str(&format!("rect 8 {y} 240 12 {c}; "));
    }
    ops.push_str("rect 0 176 256 16 3a3632; ");
    ui_draw(&ops);
}

fn query_str() -> String {
    unsafe {
        core::str::from_utf8(&QUERY[..QLEN])
            .unwrap_or("")
            .to_string()
    }
}

fn lookup(word: &str) -> Option<String> {
    let w = word.to_ascii_lowercase();
    for (k, v) in BUILTIN {
        if *k == w {
            return Some((*v).to_string());
        }
    }
    let sk = format!("dict_{w}");
    let mut buf = [0u8; 256];
    let n = storage_get_durable(&sk, &mut buf);
    if n > 0 {
        return Some(String::from_utf8_lossy(&buf[..n as usize]).into_owned());
    }
    None
}

fn search_paint() {
    let q = query_str().to_ascii_lowercase();
    unsafe {
        HIT = -1;
        if q.is_empty() {
            return;
        }
        for (i, (k, _)) in BUILTIN.iter().enumerate() {
            if k.starts_with(&q) || *k == q {
                HIT = i as i32;
                break;
            }
        }
    }
}

pub fn start(_: &str) -> String {
    unsafe {
        QUERY = [0; 32];
        QLEN = 0;
        HIT = -1;
    }
    paint();
    String::from("ok:dict (type word, enter lookup; define via dict_set)")
}

pub fn on_click(_x: i32, _y: i32) -> String {
    status("")
}

pub fn on_key(key: &str) -> String {
    match key {
        "backspace" => unsafe {
            if QLEN > 0 {
                QLEN -= 1;
                QUERY[QLEN] = 0;
            }
        },
        "enter" => {
            let q = query_str();
            search_paint();
            paint();
            return match lookup(&q) {
                Some(def) => format!("ok:{q}: {def}"),
                None => format!("ok:no definition for '{q}'"),
            };
        }
        "esc" => unsafe {
            QUERY = [0; 32];
            QLEN = 0;
            HIT = -1;
        },
        k if k.len() == 1 => {
            let ch = k.chars().next().unwrap();
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                unsafe {
                    if QLEN < 31 {
                        QUERY[QLEN] = ch.to_ascii_lowercase() as u8;
                        QLEN += 1;
                    }
                }
            }
        }
        _ => {}
    }
    search_paint();
    paint();
    status("")
}

pub fn define(args: &str) -> String {
    let word = json_str(args, "word")
        .or_else(|| json_str(args, "key"))
        .unwrap_or_default()
        .to_ascii_lowercase();
    let def = json_str(args, "def")
        .or_else(|| json_str(args, "definition"))
        .or_else(|| json_str(args, "body"))
        .unwrap_or_default();
    if word.is_empty() || def.is_empty() {
        return String::from("error: need word + def");
    }
    let sk = format!("dict_{word}");
    if storage_set_durable(&sk, &def) != 0 {
        return String::from("error:storage");
    }
    format!("ok:defined {word}")
}

pub fn lookup_tool(args: &str) -> String {
    let word = json_str(args, "word")
        .or_else(|| json_str(args, "key"))
        .unwrap_or_default();
    match lookup(&word) {
        Some(def) => format!("ok:{word}: {def}"),
        None => format!("error:no definition for '{word}'"),
    }
}

pub fn status(_: &str) -> String {
    format!("ok:dict q='{}' hit={}", query_str(), unsafe { HIT })
}
