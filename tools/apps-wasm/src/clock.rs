//! Clock / stopwatch / timer — uses host_now_ms for wall-ish ticks.

use crate::guest::{json_i32, ui_draw};
use alloc::format;
use alloc::string::String;

// 0=clock 1=stopwatch 2=timer
static mut MODE: u8 = 0;
static mut SW_BASE: i64 = 0;
static mut SW_ACC: i64 = 0;
static mut SW_RUN: u8 = 0;
static mut TMR_LEFT: i64 = 60_000;
static mut TMR_DEADLINE: i64 = 0;
static mut TMR_RUN: u8 = 0;
static mut PHASE: u8 = 0;

fn now() -> i64 {
    crate::guest::now_ms()
}

fn paint() {
    let mut ops = String::from("clear 1a1816; ");
    let mode = unsafe { MODE };
    // Mode tabs.
    for i in 0..3i32 {
        let x = 8 + i * 80;
        let c = if mode as i32 == i { "cc785c" } else { "3a3632" };
        ops.push_str(&format!("rect {x} 8 72 20 {c}; "));
    }
    // Face.
    ops.push_str("rect 48 40 160 120 2c2926; ");
    let status_bar = match mode {
        0 => {
            // Digital bars from time-of-day ms (host may be virtual).
            let ms = now();
            let sec = ((ms / 1000) % 60) as i32;
            let min = ((ms / 60_000) % 60) as i32;
            let hr = ((ms / 3_600_000) % 24) as i32;
            let hx = 56 + (hr % 12) * 12;
            let mx = 56 + min * 2;
            let sx = 56 + sec * 2;
            ops.push_str(&format!("rect {hx} 56 8 40 e8e4df; "));
            ops.push_str(&format!("rect {mx} 100 4 48 5a8f5a; "));
            ops.push_str(&format!("rect {sx} 140 2 16 cc785c; "));
            format!("clock ~{hr:02}:{min:02}:{sec:02}")
        }
        1 => {
            let elapsed = unsafe {
                if SW_RUN != 0 {
                    SW_ACC + (now() - SW_BASE)
                } else {
                    SW_ACC
                }
            };
            let w = ((elapsed / 100) % 220) as i32 + 8;
            ops.push_str(&format!("rect 56 80 {w} 24 5a8f5a; "));
            format!("stopwatch {} ms", elapsed)
        }
        _ => {
            let left = unsafe {
                if TMR_RUN != 0 {
                    (TMR_DEADLINE - now()).max(0)
                } else {
                    TMR_LEFT
                }
            };
            let w = ((left / 1000).min(220)) as i32;
            let c = if left == 0 { "aa3333" } else { "6688cc" };
            ops.push_str(&format!("rect 56 80 {w} 24 {c}; "));
            format!("timer {} s left", left / 1000)
        }
    };
    // Controls strip.
    ops.push_str("rect 8 168 240 16 3a3632; ");
    let _ = status_bar;
    ui_draw(&ops);
}

pub fn start(_: &str) -> String {
    unsafe {
        MODE = 0;
        SW_RUN = 0;
        SW_ACC = 0;
        TMR_RUN = 0;
        TMR_LEFT = 60_000;
    }
    paint();
    String::from("ok:clock (1=clock 2=stopwatch 3=timer; space start/stop; r reset; +/- timer)")
}

pub fn on_click(x: i32, y: i32) -> String {
    if y < 32 {
        let tab = ((x - 8) / 80).clamp(0, 2) as u8;
        unsafe { MODE = tab };
        paint();
        return format!("ok:mode={tab}");
    }
    // Toggle run in middle.
    on_key("space")
}

pub fn on_key(key: &str) -> String {
    match key {
        "1" => {
            unsafe { MODE = 0 };
        }
        "2" => {
            unsafe { MODE = 1 };
        }
        "3" => {
            unsafe { MODE = 2 };
        }
        "space" | "enter" => unsafe {
            match MODE {
                1 => {
                    if SW_RUN != 0 {
                        SW_ACC += now() - SW_BASE;
                        SW_RUN = 0;
                    } else {
                        SW_BASE = now();
                        SW_RUN = 1;
                    }
                }
                2 => {
                    if TMR_RUN != 0 {
                        TMR_LEFT = (TMR_DEADLINE - now()).max(0);
                        TMR_RUN = 0;
                    } else if TMR_LEFT > 0 {
                        TMR_DEADLINE = now() + TMR_LEFT;
                        TMR_RUN = 1;
                    }
                }
                _ => {}
            }
        },
        "r" => unsafe {
            match MODE {
                1 => {
                    SW_RUN = 0;
                    SW_ACC = 0;
                }
                2 => {
                    TMR_RUN = 0;
                    TMR_LEFT = 60_000;
                }
                _ => {}
            }
        },
        "+" | "=" => unsafe {
            if MODE == 2 {
                TMR_LEFT = (TMR_LEFT + 10_000).min(600_000);
            }
        },
        "-" | "_" => unsafe {
            if MODE == 2 {
                TMR_LEFT = (TMR_LEFT - 10_000).max(0);
            }
        },
        _ => {}
    }
    paint();
    status("")
}

pub fn tick(_: &str) -> String {
    unsafe {
        PHASE = PHASE.wrapping_add(1);
        if MODE == 0 || SW_RUN != 0 || TMR_RUN != 0 {
            paint();
        }
        if MODE == 2 && TMR_RUN != 0 && now() >= TMR_DEADLINE {
            TMR_RUN = 0;
            TMR_LEFT = 0;
            paint();
            return String::from("ok:timer done");
        }
    }
    String::from("ok:tick")
}

pub fn status(_: &str) -> String {
    format!("ok:mode={}", unsafe { MODE })
}

pub fn set_timer(args: &str) -> String {
    let secs = json_i32(args, "seconds", 60).max(0) as i64;
    unsafe {
        MODE = 2;
        TMR_RUN = 0;
        TMR_LEFT = secs * 1000;
    }
    paint();
    format!("ok:timer {secs}s")
}
