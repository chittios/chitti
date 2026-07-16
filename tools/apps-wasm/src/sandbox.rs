//! Sandbox lab — teaches capability attenuation with home-only storage demos.

use crate::guest::{
    json_str, storage_get_durable, storage_list_durable, storage_set_durable, ui_draw,
};
use alloc::format;
use alloc::string::{String, ToString};

// 0=overview 1=home ok 2=escape denied 3=child sim
static mut PANEL: u8 = 0;
static mut DENIES: u32 = 0;
static mut ALLOWS: u32 = 0;
static mut CHILD: u8 = 0; // simulated attenuated child active
static mut LOG: [[u8; 48]; 8] = [[0; 48]; 8];
static mut LLEN: [u8; 8] = [0; 8];
static mut NLOG: usize = 0;

fn push_log(s: &str) {
    unsafe {
        if NLOG < 8 {
            let b = s.as_bytes();
            let len = b.len().min(47);
            LOG[NLOG] = [0; 48];
            LOG[NLOG][..len].copy_from_slice(&b[..len]);
            LLEN[NLOG] = len as u8;
            NLOG += 1;
        } else {
            for i in 0..7 {
                LOG[i] = LOG[i + 1];
                LLEN[i] = LLEN[i + 1];
            }
            let b = s.as_bytes();
            let len = b.len().min(47);
            LOG[7] = [0; 48];
            LOG[7][..len].copy_from_slice(&b[..len]);
            LLEN[7] = len as u8;
        }
    }
}

fn paint() {
    let mut ops = String::from("clear 1a1816; ");
    ops.push_str("rect 0 0 256 16 3a3632; ");
    // Tabs: overview / home / escape / child
    for i in 0..4i32 {
        let x = 8 + i * 60;
        let c = if unsafe { PANEL } as i32 == i {
            "cc785c"
        } else {
            "3a3632"
        };
        ops.push_str(&format!("rect {x} 20 56 16 {c}; "));
    }
    // Body.
    ops.push_str("rect 8 44 240 100 2c2926; ");
    match unsafe { PANEL } {
        0 => {
            // allows vs denies bars
            let aw = (unsafe { ALLOWS }.min(20) as i32) * 10;
            let dw = (unsafe { DENIES }.min(20) as i32) * 10;
            ops.push_str(&format!("rect 16 56 {aw} 16 5a8f5a; "));
            ops.push_str(&format!("rect 16 80 {dw} 16 aa3333; "));
            if unsafe { CHILD } != 0 {
                ops.push_str("rect 16 110 80 16 6688cc; ");
            }
        }
        1 => {
            // home write success strip
            ops.push_str("rect 24 60 200 40 5a8f5a; ");
        }
        2 => {
            // escape denied
            ops.push_str("rect 24 60 200 40 aa3333; ");
        }
        _ => {
            // child sandbox
            let c = if unsafe { CHILD } != 0 {
                "6688cc"
            } else {
                "5a5652"
            };
            ops.push_str(&format!("rect 24 60 200 40 {c}; "));
        }
    }
    // Log lines.
    unsafe {
        for i in 0..NLOG.min(4) {
            let y = 150 + i as i32 * 8;
            let w = (LLEN[i] as i32 * 3).min(230);
            ops.push_str(&format!("rect 12 {y} {w} 6 5a5652; "));
        }
    }
    ui_draw(&ops);
}

pub fn start(_: &str) -> String {
    unsafe {
        PANEL = 0;
        DENIES = 0;
        ALLOWS = 0;
        CHILD = 0;
        NLOG = 0;
    }
    push_log("sandbox-lab ready");
    push_log("caps: Fs@home only (no net)");
    paint();
    String::from(
        "ok:sandbox-lab (1 overview 2 home-ok 3 escape-deny 4 child; h write home; e try escape; c child)",
    )
}

pub fn on_click(x: i32, y: i32) -> String {
    if y >= 20 && y < 36 {
        let p = ((x - 8) / 60).clamp(0, 3) as u8;
        unsafe { PANEL = p };
        paint();
    }
    status("")
}

pub fn on_key(key: &str) -> String {
    match key {
        "1" => unsafe { PANEL = 0 },
        "2" => unsafe { PANEL = 1 },
        "3" => unsafe { PANEL = 2 },
        "4" => unsafe { PANEL = 3 },
        "h" => {
            return home_write(r#"{"key":"lab_note","body":"home write ok"}"#);
        }
        "e" => {
            return try_escape(r#"{"path":"/etc/passwd"}"#);
        }
        "c" => {
            return child_toggle("");
        }
        "r" => {
            unsafe {
                DENIES = 0;
                ALLOWS = 0;
                NLOG = 0;
            }
            push_log("counters reset");
        }
        _ => {}
    }
    paint();
    status("")
}

/// Home-scoped write — always allowed in this package's durable storage.
pub fn home_write(args: &str) -> String {
    let key = json_str(args, "key").unwrap_or_else(|| "lab_note".into());
    let body = json_str(args, "body").unwrap_or_else(|| "ok".into());
    let sk = format!("sandbox_{key}");
    if storage_set_durable(&sk, &body) != 0 {
        unsafe { DENIES += 1 };
        push_log("home write failed");
        paint();
        return String::from("error:storage_set failed");
    }
    unsafe {
        ALLOWS += 1;
        PANEL = 1;
    }
    push_log(&format!("ALLOW home write {key}"));
    paint();
    format!("ok:ALLOW sandbox_{key}")
}

/// Simulated escape: paths outside home are refused *here* (demo of Gate 2.5).
pub fn try_escape(args: &str) -> String {
    let path = json_str(args, "path")
        .or_else(|| json_str(args, "key"))
        .unwrap_or_else(|| "/etc/passwd".into());
    // Anything that looks like absolute FS outside agent home is denied.
    let denied = path.starts_with('/')
        || path.contains("..")
        || path.starts_with("etc")
        || path.contains("agent/1")
        || path.contains("configs");
    unsafe {
        PANEL = 2;
        DENIES += 1;
    }
    if denied {
        push_log(&format!("DENY escape {path}"));
        paint();
        format!(
            "error:DENIED path '{path}' outside sandbox (home only). \
             Effective authority is intersection(requested, grant). Child cannot widen."
        )
    } else {
        // Relative keys still go to durable storage under sandbox_ prefix.
        let sk = format!("sandbox_{path}");
        let _ = storage_set_durable(&sk, "probe");
        unsafe { ALLOWS += 1 };
        push_log(&format!("ALLOW relative {path}"));
        paint();
        format!("ok:relative key stored as {sk}")
    }
}

/// Toggle a simulated attenuated child (no real spawn in wasm — educational UI).
pub fn child_toggle(_: &str) -> String {
    unsafe {
        CHILD = if CHILD == 0 { 1 } else { 0 };
        PANEL = 3;
        if CHILD != 0 {
            ALLOWS += 1;
            push_log("child: caps narrowed to home");
        } else {
            push_log("child: stopped");
        }
    }
    paint();
    if unsafe { CHILD != 0 } {
        String::from(
            "ok:child ON — simulated sub-agent with Fs@home only, no net, no skill_manage. \
             Chat: use spawn_subagent role=explore for a real attenuated child.",
        )
    } else {
        String::from("ok:child OFF")
    }
}

pub fn list_home(_: &str) -> String {
    let mut buf = [0u8; 2048];
    let n = storage_list_durable(&mut buf);
    if n <= 0 {
        return String::from("(empty sandbox storage)");
    }
    let raw = core::str::from_utf8(&buf[..n as usize]).unwrap_or("");
    let mut out = String::new();
    for line in raw.split('\n') {
        if line.starts_with("sandbox_") {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(line);
        }
    }
    if out.is_empty() {
        String::from("(no sandbox_* keys)")
    } else {
        out
    }
}

pub fn get(args: &str) -> String {
    let key = json_str(args, "key").unwrap_or_default();
    let sk = if key.starts_with("sandbox_") {
        key
    } else {
        format!("sandbox_{key}")
    };
    let mut buf = [0u8; 512];
    let n = storage_get_durable(&sk, &mut buf);
    if n < 0 {
        return format!("error:no {sk}");
    }
    String::from_utf8_lossy(&buf[..n as usize]).into_owned()
}

pub fn status(_: &str) -> String {
    format!(
        "ok:sandbox allows={} denies={} child={} panel={}",
        unsafe { ALLOWS },
        unsafe { DENIES },
        unsafe { CHILD },
        unsafe { PANEL }
    )
}
