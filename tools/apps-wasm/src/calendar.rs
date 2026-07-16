//! Calendar — month grid; events stored as durable `cal_YYYY-MM-DD` keys.

use crate::guest::{json_str, storage_get_durable, storage_list_durable, storage_set_durable, ui_draw};
use alloc::format;
use alloc::string::{String, ToString};

static mut YEAR: i32 = 2026;
static mut MONTH: i32 = 7; // 1-12
static mut DAY: i32 = 1;
static mut CURSOR: i32 = 1;

fn days_in_month(y: i32, m: i32) -> i32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 {
                29
            } else {
                28
            }
        }
        _ => 30,
    }
}

/// Sakamoto DOW: 0=Sun … 6=Sat for day 1 of month.
fn dow1(y: i32, m: i32) -> i32 {
    let t = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    let mut yy = y;
    if m < 3 {
        yy -= 1;
    }
    (yy + yy / 4 - yy / 100 + yy / 400 + t[(m - 1) as usize] + 1) % 7
}

fn has_event(d: i32) -> bool {
    let key = format!("cal_{:04}-{:02}-{:02}", unsafe { YEAR }, unsafe { MONTH }, d);
    let mut buf = [0u8; 8];
    storage_get_durable(&key, &mut buf) > 0
}

fn paint() {
    let mut ops = String::from("clear 1a1816; ");
    ops.push_str("rect 0 0 256 20 3a3632; ");
    // Month length bar.
    let m = unsafe { MONTH };
    ops.push_str(&format!("rect 8 4 {} 12 cc785c; ", m * 16));
    let dim = days_in_month(unsafe { YEAR }, m);
    let start = dow1(unsafe { YEAR }, m);
    let cell = 32;
    for d in 1..=dim {
        let slot = start + d - 1;
        let col = slot % 7;
        let row = slot / 7;
        let x = 8 + col * cell;
        let y = 28 + row * cell;
        let sel = d == unsafe { CURSOR };
        let ev = has_event(d);
        let c = if sel {
            "cc785c"
        } else if ev {
            "5a8f5a"
        } else {
            "3a3632"
        };
        ops.push_str(&format!("rect {x} {y} 28 28 {c}; "));
    }
    ops.push_str("rect 0 176 256 16 3a3632; ");
    ui_draw(&ops);
}

pub fn start(_: &str) -> String {
    // Approximate from host clock: days since epoch-ish.
    let ms = crate::guest::now_ms();
    let days = (ms / 86_400_000) as i32;
    unsafe {
        // Keep defaults if clock is tiny; otherwise rough-ish.
        if days > 1000 {
            YEAR = 1970 + days / 365;
            MONTH = ((days % 365) / 30).clamp(1, 12);
        }
        DAY = 1;
        CURSOR = 1;
    }
    paint();
    format!(
        "ok:calendar {}-{:02} (arrows day, n/p month, e add event)",
        unsafe { YEAR },
        unsafe { MONTH }
    )
}

pub fn on_click(x: i32, y: i32) -> String {
    if y < 28 {
        return status("");
    }
    let col = ((x - 8) / 32).clamp(0, 6);
    let row = ((y - 28) / 32).clamp(0, 5);
    let start = dow1(unsafe { YEAR }, unsafe { MONTH });
    let slot = row * 7 + col;
    let d = slot - start + 1;
    let dim = days_in_month(unsafe { YEAR }, unsafe { MONTH });
    if d >= 1 && d <= dim {
        unsafe { CURSOR = d };
        paint();
    }
    status("")
}

pub fn on_key(key: &str) -> String {
    let dim = days_in_month(unsafe { YEAR }, unsafe { MONTH });
    match key {
        "left" => unsafe {
            CURSOR = (CURSOR - 1).max(1);
        },
        "right" => unsafe {
            CURSOR = (CURSOR + 1).min(dim);
        },
        "up" => unsafe {
            CURSOR = (CURSOR - 7).max(1);
        },
        "down" => unsafe {
            CURSOR = (CURSOR + 7).min(dim);
        },
        "n" => unsafe {
            MONTH += 1;
            if MONTH > 12 {
                MONTH = 1;
                YEAR += 1;
            }
            CURSOR = 1;
        },
        "p" => unsafe {
            MONTH -= 1;
            if MONTH < 1 {
                MONTH = 12;
                YEAR -= 1;
            }
            CURSOR = 1;
        },
        "e" | "enter" => {
            let key = format!(
                "cal_{:04}-{:02}-{:02}",
                unsafe { YEAR },
                unsafe { MONTH },
                unsafe { CURSOR }
            );
            let _ = storage_set_durable(&key, "event");
        }
        _ => {}
    }
    paint();
    status("")
}

pub fn add(args: &str) -> String {
    let date = json_str(args, "date").unwrap_or_default();
    let body = json_str(args, "body")
        .or_else(|| json_str(args, "title"))
        .unwrap_or_else(|| "event".into());
    if date.is_empty() {
        return String::from("error: need date=YYYY-MM-DD");
    }
    let key = format!("cal_{date}");
    if storage_set_durable(&key, &body) != 0 {
        return String::from("error:storage");
    }
    paint();
    format!("ok:event {date}")
}

pub fn list(_: &str) -> String {
    let mut buf = [0u8; 4096];
    let n = storage_list_durable(&mut buf);
    if n <= 0 {
        return String::from("(no events)");
    }
    let raw = core::str::from_utf8(&buf[..n as usize]).unwrap_or("");
    let mut out = String::new();
    for line in raw.split('\n') {
        if let Some(rest) = line.strip_prefix("cal_") {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(rest);
        }
    }
    if out.is_empty() {
        String::from("(no events)")
    } else {
        out
    }
}

pub fn status(_: &str) -> String {
    format!(
        "ok:{}-{:02}-{:02}",
        unsafe { YEAR },
        unsafe { MONTH },
        unsafe { CURSOR }
    )
}
