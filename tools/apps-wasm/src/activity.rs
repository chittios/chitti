//! Activity — live process / agent status panel from the scheduler.

use crate::guest::{
    hud_status, json_i32, storage_get_durable, storage_set_durable, tasks_list, text_op, ui_draw,
};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

static mut TICKS: u32 = 0;
static mut SCROLL: u8 = 0;
static mut MEM_PCT: u8 = 0;
static mut N_SHOWN: u8 = 0;

/// Parsed row: name truncated for paint.
#[derive(Clone, Copy)]
struct Row {
    id: u64,
    name: [u8; 20],
    nlen: u8,
    state: [u8; 10],
    slen: u8,
}

static mut ROWS: [Row; 16] = [Row {
    id: 0,
    name: [0; 20],
    nlen: 0,
    state: [0; 10],
    slen: 0,
}; 16];
static mut NROWS: usize = 0;

fn reload() {
    let mut buf = [0u8; 1536];
    let n = tasks_list(&mut buf);
    unsafe {
        NROWS = 0;
        MEM_PCT = 0;
        if n <= 0 {
            return;
        }
        let raw = core::str::from_utf8(&buf[..n as usize]).unwrap_or("");
        for line in raw.split('\n') {
            if line.is_empty() {
                continue;
            }
            let mut parts = line.split('\t');
            let a = parts.next().unwrap_or("");
            let b = parts.next().unwrap_or("");
            let c = parts.next().unwrap_or("");
            if a == "heap" {
                // "used/total" \t "pct%"
                if let Some(pct) = c.trim_end_matches('%').parse::<u8>().ok() {
                    MEM_PCT = pct.min(100);
                }
                continue;
            }
            if NROWS >= 16 {
                continue;
            }
            let id = a.parse::<u64>().unwrap_or(0);
            let nb = b.as_bytes();
            let nl = nb.len().min(19);
            let sb = c.as_bytes();
            let sl = sb.len().min(9);
            let mut row = Row {
                id,
                name: [0; 20],
                nlen: nl as u8,
                state: [0; 10],
                slen: sl as u8,
            };
            row.name[..nl].copy_from_slice(&nb[..nl]);
            row.state[..sl].copy_from_slice(&sb[..sl]);
            ROWS[NROWS] = row;
            NROWS += 1;
        }
        N_SHOWN = NROWS.min(8) as u8;
        if SCROLL as usize >= NROWS && NROWS > 0 {
            SCROLL = (NROWS - 1) as u8;
        }
    }
}

fn paint() {
    reload();
    let mut ops = String::from("clear 1a1816; ");
    ops.push_str("rect 0 0 256 16 3a3632; ");
    text_op(
        &mut ops,
        8,
        1,
        12,
        "e8e4df",
        &format!(
            "{} Activity  {} task(s)  heap {}%",
            crate::fa::MICROCHIP,
            unsafe { NROWS },
            unsafe { MEM_PCT }
        ),
    );
    unsafe {
        let start = SCROLL as usize;
        let end = (start + 8).min(NROWS);
        let mut row_i = 0i32;
        for i in start..end {
            let y = 28 + row_i * 18;
            let r = &ROWS[i];
            ops.push_str(&format!("rect 12 {y} 232 14 2c2926; "));
            // State color.
            let st = core::str::from_utf8(&r.state[..r.slen as usize]).unwrap_or("?");
            let bar = match st {
                "running" => "5a8f5a",
                "ready" => "6688cc",
                "parked" => "8a5a4a",
                "blocked" => "cc785c",
                "idle" => "6a6a6a",
                _ => "5a5a5a",
            };
            ops.push_str(&format!("rect 14 {} 6 10 {bar}; ", y + 2));
            let name = core::str::from_utf8(&r.name[..r.nlen as usize]).unwrap_or("?");
            text_op(
                &mut ops,
                24,
                y + 1,
                10,
                "e8e4df",
                &format!("{} {}  {st}", r.id, name),
            );
            row_i += 1;
        }
        if NROWS == 0 {
            text_op(&mut ops, 20, 40, 11, "a8a4a0", "(no tasks reported)");
        }
        // Memory bar from real heap used%.
        ops.push_str("rect 12 160 232 10 2c2926; ");
        let mw = (MEM_PCT as i32 * 232 / 100).max(1).min(232);
        ops.push_str(&format!("rect 12 160 {mw} 10 6688cc; "));
        text_op(
            &mut ops,
            16,
            148,
            10,
            "a8a4a0",
            &format!("heap {MEM_PCT}%  ticks {}", TICKS),
        );
    }
    ops.push_str("rect 0 176 256 16 3a3632; ");
    text_op(&mut ops, 8, 178, 10, "a8a4a0", "up/dn scroll  r refresh");
    ui_draw(&ops);
    hud_status(
        &format!(
            "activity  tasks={}  heap%={}  ticks={}",
            unsafe { NROWS },
            unsafe { MEM_PCT },
            unsafe { TICKS }
        ),
        "up/dn scroll  r refresh",
    );
}

pub fn start(_: &str) -> String {
    let mut buf = [0u8; 32];
    let n = storage_get_durable("activity_snap", &mut buf);
    if n > 0 {
        if let Ok(s) = core::str::from_utf8(&buf[..n as usize]) {
            if let Ok(sc) = s.parse::<u8>() {
                unsafe { SCROLL = sc };
            }
        }
    }
    paint();
    String::from("ok:activity (live scheduler + heap)")
}

pub fn on_click(_x: i32, _y: i32) -> String {
    paint();
    status("")
}

pub fn on_key(key: &str) -> String {
    match key {
        "ArrowUp" | "up" | "k" => unsafe {
            SCROLL = SCROLL.saturating_sub(1);
        },
        "ArrowDown" | "down" | "j" => unsafe {
            if (SCROLL as usize + 1) < NROWS {
                SCROLL = SCROLL.saturating_add(1);
            }
        },
        "r" => {
            // paint() reloads
        }
        _ => {}
    }
    let _ = storage_set_durable("activity_snap", &format!("{}", unsafe { SCROLL }));
    paint();
    status("")
}

pub fn tick(_: &str) -> String {
    unsafe {
        TICKS = TICKS.wrapping_add(1);
    }
    // Refresh live data every tick (~UI pump rate from package_ui).
    paint();
    String::from("ok:tick")
}

pub fn set(args: &str) -> String {
    // Optional: agent can force scroll position.
    let sc = json_i32(args, "scroll", unsafe { SCROLL as i32 }).clamp(0, 15) as u8;
    unsafe {
        SCROLL = sc;
    }
    paint();
    status("")
}

pub fn status(_: &str) -> String {
    reload();
    let mut out = format!(
        "ok:activity tasks={} heap%={} ticks={}\n",
        unsafe { NROWS },
        unsafe { MEM_PCT },
        unsafe { TICKS }
    );
    unsafe {
        for i in 0..NROWS.min(12) {
            let r = &ROWS[i];
            let name = core::str::from_utf8(&r.name[..r.nlen as usize]).unwrap_or("?");
            let st = core::str::from_utf8(&r.state[..r.slen as usize]).unwrap_or("?");
            out.push_str(&format!("{} {} {}\n", r.id, name, st));
        }
    }
    out
}
