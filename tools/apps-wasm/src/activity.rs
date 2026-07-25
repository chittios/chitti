//! Activity — simple process/agent status panel (storage + static counters).

use crate::guest::{hud_status, json_i32, storage_get_durable, storage_set_durable, text_op, ui_draw};
use alloc::format;
use alloc::string::String;

static mut TICKS: u32 = 0;
static mut TASKS: u8 = 3;
static mut MEM: u8 = 40;

fn paint() {
    let mut ops = String::from("clear 1a1816; ");
    ops.push_str("rect 0 0 256 16 3a3632; ");
    text_op(
        &mut ops,
        8,
        1,
        12,
        "e8e4df",
        &format!("Activity  ticks {}", unsafe { TICKS }),
    );
    // Fake task rows.
    unsafe {
        for i in 0..TASKS.min(8) {
            let y = 28 + i as i32 * 18;
            ops.push_str(&format!("rect 12 {y} 232 14 2c2926; "));
            let w = 40 + (i as i32 * 17 + (TICKS as i32 % 30));
            ops.push_str(&format!("rect 16 {} {} 8 5a8f5a; ", y + 3, w.min(220)));
            text_op(
                &mut ops,
                20,
                y + 1,
                10,
                "e8e4df",
                &format!("task-{}", i + 1),
            );
        }
        // Memory bar.
        ops.push_str("rect 12 160 232 10 2c2926; ");
        let mw = (MEM as i32 * 2).min(232);
        ops.push_str(&format!("rect 12 160 {mw} 10 6688cc; "));
        text_op(
            &mut ops,
            16,
            148,
            10,
            "a8a4a0",
            &format!("mem {MEM}%"),
        );
    }
    ops.push_str("rect 0 176 256 16 3a3632; ");
    text_op(&mut ops, 8, 178, 10, "a8a4a0", "+/- tasks  r reset");
    ui_draw(&ops);
    hud_status(
        &format!(
            "activity  tasks={}  mem%={}  ticks={}",
            unsafe { TASKS },
            unsafe { MEM },
            unsafe { TICKS }
        ),
        "+/- tasks  r reset",
    );
}

pub fn start(_: &str) -> String {
    // Restore last snapshot if any.
    let mut buf = [0u8; 32];
    let n = storage_get_durable("activity_snap", &mut buf);
    if n > 0 {
        if let Ok(s) = core::str::from_utf8(&buf[..n as usize]) {
            // "tasks,mem"
            if let Some((a, b)) = s.split_once(',') {
                unsafe {
                    TASKS = a.parse().unwrap_or(3);
                    MEM = b.parse().unwrap_or(40);
                }
            }
        }
    }
    paint();
    String::from("ok:activity (tick refreshes; +/- tasks)")
}

pub fn on_click(_x: i32, _y: i32) -> String {
    status("")
}

pub fn on_key(key: &str) -> String {
    match key {
        "+" | "=" => unsafe {
            TASKS = TASKS.saturating_add(1).min(8);
        },
        "-" | "_" => unsafe {
            TASKS = TASKS.saturating_sub(1).max(1);
        },
        "r" => unsafe {
            TICKS = 0;
            MEM = 40;
        },
        _ => {}
    }
    paint();
    status("")
}

pub fn tick(_: &str) -> String {
    unsafe {
        TICKS = TICKS.wrapping_add(1);
        MEM = ((MEM as u16 + ((TICKS % 7) as u16)) % 100) as u8;
    }
    paint();
    String::from("ok:tick")
}

pub fn set(args: &str) -> String {
    let tasks = json_i32(args, "tasks", unsafe { TASKS as i32 }).clamp(1, 8) as u8;
    let mem = json_i32(args, "mem", unsafe { MEM as i32 }).clamp(0, 100) as u8;
    unsafe {
        TASKS = tasks;
        MEM = mem;
    }
    let _ = storage_set_durable("activity_snap", &format!("{tasks},{mem}"));
    paint();
    status("")
}

pub fn status(_: &str) -> String {
    format!(
        "ok:activity tasks={} mem%={} ticks={}",
        unsafe { TASKS },
        unsafe { MEM },
        unsafe { TICKS }
    )
}
