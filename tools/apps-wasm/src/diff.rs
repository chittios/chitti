//! Diff — compare two durable text blobs side by side.

use crate::guest::{hud_status, json_str, storage_get_durable, storage_set_durable, text_op, ui_draw};
use alloc::format;
use alloc::string::{String, ToString};

static mut A: [u8; 512] = [0; 512];
static mut ALEN: usize = 0;
static mut B: [u8; 512] = [0; 512];
static mut BLEN: usize = 0;
static mut SCROLL: usize = 0;

fn paint() {
    let mut ops = String::from("clear 1a1816; ");
    ops.push_str("rect 0 0 256 16 3a3632; ");
    text_op(&mut ops, 8, 1, 12, "e8e4df", "Diff  A | B");
    ops.push_str("rect 4 20 122 150 2c2926; ");
    ops.push_str("rect 130 20 122 150 2c2926; ");
    text_op(&mut ops, 8, 22, 10, "a8a4a0", "A");
    text_op(&mut ops, 134, 22, 10, "a8a4a0", "B");
    unsafe {
        let lines = ALEN.max(BLEN).div_ceil(16).max(1);
        let start = SCROLL;
        for i in 0..10usize {
            let li = start + i;
            if li >= lines {
                break;
            }
            let y = 36 + i as i32 * 12;
            let a_off = li * 16;
            let b_off = li * 16;
            let a_slice = if a_off < ALEN {
                &A[a_off..ALEN.min(a_off + 16)]
            } else {
                &[][..]
            };
            let b_slice = if b_off < BLEN {
                &B[b_off..BLEN.min(b_off + 16)]
            } else {
                &[][..]
            };
            let same = a_slice == b_slice;
            let ca = if a_slice.is_empty() {
                "3a3632"
            } else if same {
                "5a8f5a"
            } else {
                "aa3333"
            };
            let cb = if b_slice.is_empty() {
                "3a3632"
            } else if same {
                "5a8f5a"
            } else {
                "aa3333"
            };
            ops.push_str(&format!("rect 8 {y} 114 10 {ca}; "));
            ops.push_str(&format!("rect 134 {y} 114 10 {cb}; "));
            let as_ = core::str::from_utf8(a_slice).unwrap_or(".");
            let bs = core::str::from_utf8(b_slice).unwrap_or(".");
            let as_ = if as_.len() > 14 { &as_[..14] } else { as_ };
            let bs = if bs.len() > 14 { &bs[..14] } else { bs };
            text_op(&mut ops, 10, y, 9, "e8e4df", as_);
            text_op(&mut ops, 136, y, 9, "e8e4df", bs);
        }
    }
    ops.push_str("rect 0 176 256 16 3a3632; ");
    text_op(&mut ops, 8, 178, 10, "a8a4a0", "arrows scroll");
    ui_draw(&ops);
    hud_status(
        &format!("diff  a={} b={} bytes", unsafe { ALEN }, unsafe { BLEN }),
        "arrows scroll",
    );
}

fn load_pair() {
    let mut buf = [0u8; 512];
    let n = storage_get_durable("diff_a", &mut buf);
    unsafe {
        A = [0; 512];
        ALEN = 0;
        if n > 0 {
            let len = (n as usize).min(512);
            A[..len].copy_from_slice(&buf[..len]);
            ALEN = len;
        }
    }
    let n = storage_get_durable("diff_b", &mut buf);
    unsafe {
        B = [0; 512];
        BLEN = 0;
        if n > 0 {
            let len = (n as usize).min(512);
            B[..len].copy_from_slice(&buf[..len]);
            BLEN = len;
        }
    }
}

pub fn start(_: &str) -> String {
    load_pair();
    if unsafe { ALEN == 0 && BLEN == 0 } {
        let _ = storage_set_durable("diff_a", "hello world\nline2\n");
        let _ = storage_set_durable("diff_b", "hello chitti\nline2\nextra\n");
        load_pair();
    }
    unsafe { SCROLL = 0 };
    paint();
    String::from("ok:diff (set via diff_set a/b; arrows scroll)")
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
        "r" => load_pair(),
        _ => {}
    }
    paint();
    status("")
}

pub fn set(args: &str) -> String {
    let side = json_str(args, "side").unwrap_or_else(|| "a".into());
    let body = json_str(args, "body")
        .or_else(|| json_str(args, "content"))
        .unwrap_or_default();
    let key = if side == "b" { "diff_b" } else { "diff_a" };
    if storage_set_durable(key, &body) != 0 {
        return String::from("error:storage");
    }
    load_pair();
    paint();
    format!("ok:diff set {side} ({} bytes)", body.len())
}

pub fn status(_: &str) -> String {
    // Count differing 16-byte chunks.
    let mut diff = 0u32;
    unsafe {
        let lines = ALEN.max(BLEN).div_ceil(16).max(1);
        for li in 0..lines {
            let a_off = li * 16;
            let b_off = li * 16;
            let a_slice = if a_off < ALEN {
                &A[a_off..ALEN.min(a_off + 16)]
            } else {
                &[][..]
            };
            let b_slice = if b_off < BLEN {
                &B[b_off..BLEN.min(b_off + 16)]
            } else {
                &[][..]
            };
            if a_slice != b_slice {
                diff += 1;
            }
        }
    }
    format!(
        "ok:diff a={} b={} chunks_diff={}",
        unsafe { ALEN },
        unsafe { BLEN },
        diff
    )
}
