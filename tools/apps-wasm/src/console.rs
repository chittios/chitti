//! Console — ring-buffer log viewer (durable lines + demo seed).

use crate::guest::{json_str, storage_get_durable, storage_set_durable, ui_draw};
use alloc::format;
use alloc::string::{String, ToString};

const MAX_LINES: usize = 48;
const VIS: usize = 12;

static mut LINES: [[u8; 64]; MAX_LINES] = [[0; 64]; MAX_LINES];
static mut LLEN: [u8; MAX_LINES] = [0; MAX_LINES];
static mut N: usize = 0;
static mut SCROLL: usize = 0;
static mut FILTER: [u8; 16] = [0; 16];
static mut FLEN: usize = 0;

fn push_line(s: &str) {
    unsafe {
        if N < MAX_LINES {
            let b = s.as_bytes();
            let len = b.len().min(63);
            LINES[N] = [0; 64];
            LINES[N][..len].copy_from_slice(&b[..len]);
            LLEN[N] = len as u8;
            N += 1;
        } else {
            // shift up
            for i in 0..MAX_LINES - 1 {
                LINES[i] = LINES[i + 1];
                LLEN[i] = LLEN[i + 1];
            }
            let b = s.as_bytes();
            let len = b.len().min(63);
            LINES[MAX_LINES - 1] = [0; 64];
            LINES[MAX_LINES - 1][..len].copy_from_slice(&b[..len]);
            LLEN[MAX_LINES - 1] = len as u8;
        }
    }
}

fn filter_str() -> String {
    unsafe {
        core::str::from_utf8(&FILTER[..FLEN])
            .unwrap_or("")
            .to_string()
    }
}

fn line_str(i: usize) -> String {
    unsafe {
        if i >= N {
            return String::new();
        }
        core::str::from_utf8(&LINES[i][..LLEN[i] as usize])
            .unwrap_or("")
            .to_string()
    }
}

fn matches_filter(s: &str) -> bool {
    let f = filter_str();
    if f.is_empty() {
        return true;
    }
    s.to_ascii_lowercase().contains(&f.to_ascii_lowercase())
}

fn visible_indices() -> ([usize; VIS], usize) {
    let mut idx = [0usize; VIS];
    let mut count = 0usize;
    let mut seen = 0usize;
    let scroll = unsafe { SCROLL };
    unsafe {
        for i in 0..N {
            let s = line_str(i);
            if !matches_filter(&s) {
                continue;
            }
            if seen >= scroll {
                if count < VIS {
                    idx[count] = i;
                    count += 1;
                }
            }
            seen += 1;
        }
    }
    (idx, count)
}

fn paint() {
    let mut ops = String::from("clear 1a1816; ");
    ops.push_str("rect 0 0 256 16 3a3632; ");
    // Filter bar.
    ops.push_str("rect 8 18 240 14 2c2926; ");
    let fw = (unsafe { FLEN } as i32 * 6).min(220);
    ops.push_str(&format!("rect 12 20 {fw} 10 cc785c; "));
    let (idx, count) = visible_indices();
    for row in 0..count {
        let y = 36 + row as i32 * 12;
        let s = line_str(idx[row]);
        let level = if s.contains("ERR") || s.contains("error") {
            "aa3333"
        } else if s.contains("WARN") {
            "c07050"
        } else if s.contains("ok") || s.contains("OK") {
            "5a8f5a"
        } else {
            "5a5652"
        };
        ops.push_str(&format!("rect 8 {y} 240 10 {level}; "));
        let w = (s.len() as i32 * 3).min(220);
        ops.push_str(&format!("rect 12 {} {w} 6 e8e4df; ", y + 2));
    }
    ops.push_str("rect 0 184 256 8 3a3632; ");
    ui_draw(&ops);
}

fn seed() {
    unsafe {
        N = 0;
        SCROLL = 0;
    }
    // Restore persisted log if any.
    let mut buf = [0u8; 2048];
    let n = storage_get_durable("console_log", &mut buf);
    if n > 0 {
        let raw = core::str::from_utf8(&buf[..n as usize]).unwrap_or("");
        for line in raw.lines().take(MAX_LINES) {
            push_line(line);
        }
    }
    if unsafe { N } == 0 {
        push_line("boot: console ready");
        push_line("ktrace: synapse.audit ok");
        push_line("net: dhcp lease acquired");
        push_line("agent: shell bound");
        push_line("ui: package_ui idle");
    }
}

fn persist() {
    let mut blob = String::new();
    unsafe {
        for i in 0..N {
            if i > 0 {
                blob.push('\n');
            }
            blob.push_str(&line_str(i));
        }
    }
    let _ = storage_set_durable("console_log", &blob);
}

pub fn start(_: &str) -> String {
    unsafe {
        FILTER = [0; 16];
        FLEN = 0;
    }
    seed();
    paint();
    format!("ok:console {} lines (↑↓ scroll, type filter, c clear, a append demo)", unsafe {
        N
    })
}

pub fn on_click(_x: i32, _y: i32) -> String {
    status("")
}

pub fn on_key(key: &str) -> String {
    match key {
        "up" => unsafe {
            SCROLL = SCROLL.saturating_sub(1);
        },
        "down" => unsafe {
            SCROLL += 1;
        },
        "c" => {
            unsafe {
                N = 0;
                SCROLL = 0;
            }
            persist();
        }
        "a" => {
            push_line(&format!("event: tick {}", crate::guest::now_ms() % 10000));
            persist();
        }
        "backspace" => unsafe {
            if FLEN > 0 {
                FLEN -= 1;
                FILTER[FLEN] = 0;
                SCROLL = 0;
            }
        },
        "esc" => unsafe {
            FILTER = [0; 16];
            FLEN = 0;
            SCROLL = 0;
        },
        k if k.len() == 1 => {
            let ch = k.chars().next().unwrap();
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' || ch == '.' {
                unsafe {
                    if FLEN < 15 {
                        FILTER[FLEN] = ch as u8;
                        FLEN += 1;
                        SCROLL = 0;
                    }
                }
            }
        }
        _ => {}
    }
    paint();
    status("")
}

pub fn log(args: &str) -> String {
    let msg = json_str(args, "msg")
        .or_else(|| json_str(args, "line"))
        .or_else(|| json_str(args, "text"))
        .unwrap_or_default();
    if msg.is_empty() {
        return String::from("error: need msg");
    }
    push_line(&msg);
    persist();
    paint();
    format!("ok:logged ({} lines)", unsafe { N })
}

pub fn list(_: &str) -> String {
    let mut out = String::new();
    unsafe {
        for i in 0..N {
            let s = line_str(i);
            if !matches_filter(&s) {
                continue;
            }
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(&s);
        }
    }
    if out.is_empty() {
        String::from("(empty)")
    } else {
        out
    }
}

pub fn status(_: &str) -> String {
    format!(
        "ok:console n={} scroll={} filter='{}'",
        unsafe { N },
        unsafe { SCROLL },
        filter_str()
    )
}
