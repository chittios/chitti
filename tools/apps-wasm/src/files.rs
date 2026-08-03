//! Files — Synapse FS browser (same store as shell `/ls` / `/cat`).

use crate::guest::{
    fs_list, fs_read, hud_status, json_str, text_op, ui_draw,
};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Max path components we keep on the stack (cwd trail).
const MAX_DEPTH: usize = 12;
/// Display rows.
const VIS: usize = 8;

static mut PATH: [[u8; 40]; MAX_DEPTH] = [[0; 40]; MAX_DEPTH];
static mut PATH_LEN: [u8; MAX_DEPTH] = [0; MAX_DEPTH];
static mut DEPTH: usize = 0; // 0 = root `/`

static mut NAMES: [[u8; 48]; 48] = [[0; 48]; 48];
static mut NAME_LEN: [u8; 48] = [0; 48];
static mut IS_DIR: [u8; 48] = [0; 48];
static mut NENTS: usize = 0;
static mut SEL: i32 = 0;
static mut SCROLL: i32 = 0;
static mut PREVIEW: [u8; 320] = [0; 320];
static mut PREVIEW_LEN: usize = 0;

fn cwd_string() -> String {
    unsafe {
        if DEPTH == 0 {
            return String::from("/");
        }
        let mut s = String::new();
        for i in 0..DEPTH {
            s.push('/');
            if let Ok(p) = core::str::from_utf8(&PATH[i][..PATH_LEN[i] as usize]) {
                s.push_str(p);
            }
        }
        s
    }
}

fn join_cwd(name: &str) -> String {
    let base = cwd_string();
    if base == "/" {
        format!("/{name}")
    } else {
        format!("{base}/{name}")
    }
}

fn set_depth_component(i: usize, name: &str) {
    let b = name.as_bytes();
    let len = b.len().min(39);
    unsafe {
        PATH[i] = [0; 40];
        PATH[i][..len].copy_from_slice(&b[..len]);
        PATH_LEN[i] = len as u8;
    }
}

fn reload() {
    let path = cwd_string();
    let mut buf = [0u8; 4096];
    let n = fs_list(&path, &mut buf);
    unsafe {
        NENTS = 0;
        PREVIEW_LEN = 0;
        if n < 0 {
            return;
        }
        if n == 0 {
            // Empty dir is fine.
            return;
        }
        let raw = core::str::from_utf8(&buf[..n as usize]).unwrap_or("");
        // Collect then dirs-first sort (host already dirs-first usually).
        let mut rows: Vec<(String, bool, usize)> = Vec::new();
        for line in raw.split('\n') {
            if line.is_empty() {
                continue;
            }
            let mut it = line.split('\t');
            let name = it.next().unwrap_or("").to_string();
            let kind = it.next().unwrap_or("f");
            let size = it.next().and_then(|s| s.parse().ok()).unwrap_or(0usize);
            if name.is_empty() || name == "." || name == ".." {
                continue;
            }
            rows.push((name, kind.starts_with('d'), size));
        }
        rows.sort_by(|a, b| match (a.1, b.1) {
            (true, false) => core::cmp::Ordering::Less,
            (false, true) => core::cmp::Ordering::Greater,
            _ => a.0.cmp(&b.0),
        });
        for (name, is_dir, _sz) in rows.into_iter().take(48) {
            let b = name.as_bytes();
            let len = b.len().min(47);
            NAMES[NENTS] = [0; 48];
            NAMES[NENTS][..len].copy_from_slice(&b[..len]);
            NAME_LEN[NENTS] = len as u8;
            IS_DIR[NENTS] = if is_dir { 1 } else { 0 };
            NENTS += 1;
        }
        if SEL as usize >= NENTS {
            SEL = 0;
        }
        if SCROLL > SEL {
            SCROLL = SEL;
        }
    }
}

fn selected_name() -> String {
    unsafe {
        if NENTS == 0 {
            return String::new();
        }
        let i = SEL as usize;
        core::str::from_utf8(&NAMES[i][..NAME_LEN[i] as usize])
            .unwrap_or("")
            .to_string()
    }
}

fn selected_is_dir() -> bool {
    unsafe { NENTS > 0 && IS_DIR[SEL as usize] != 0 }
}

fn load_preview() {
    unsafe {
        PREVIEW_LEN = 0;
        PREVIEW = [0; 320];
    }
    let name = selected_name();
    if name.is_empty() || selected_is_dir() {
        return;
    }
    let path = join_cwd(&name);
    let mut buf = [0u8; 320];
    let n = fs_read(&path, &mut buf);
    if n > 0 {
        let len = (n as usize).min(319);
        unsafe {
            PREVIEW[..len].copy_from_slice(&buf[..len]);
            PREVIEW_LEN = len;
        }
    }
}

fn paint() {
    let cwd = cwd_string();
    let mut ops = String::from("clear 1a1816; ");
    ops.push_str("rect 0 0 256 16 3a3632; ");
    // Folder glyph (Font Awesome) + path.
    let path_show = if cwd.len() > 26 {
        // keep end of path
        format!("…{}", &cwd[cwd.len() - 24..])
    } else {
        cwd.clone()
    };
    text_op(
        &mut ops,
        6,
        1,
        11,
        "e8e4df",
        &format!("{} {}", crate::fa::FOLDER_OPEN, path_show),
    );
    ops.push_str("rect 0 16 130 160 2c2926; ");
    ops.push_str("rect 134 16 118 160 2c2926; ");
    text_op(&mut ops, 138, 20, 10, "a8a4a0", "preview");
    unsafe {
        // Parent row when not at root.
        let mut row0 = 0i32;
        let start = SCROLL as usize;
        let mut painted = 0usize;
        if DEPTH > 0 && start == 0 {
            let y = 20;
            let c = if SEL < 0 { "cc785c" } else { "3a3632" };
            ops.push_str(&format!("rect 4 {y} 122 16 {c}; "));
            text_op(
                &mut ops,
                8,
                y + 2,
                10,
                "e8e4df",
                &format!("{} ..", crate::fa::FOLDER),
            );
            painted = 1;
            row0 = 1;
        }
        let ent_start = if DEPTH > 0 && start == 0 {
            0
        } else if DEPTH > 0 {
            start.saturating_sub(1)
        } else {
            start
        };
        for i in 0..(VIS - painted) {
            let idx = ent_start + i;
            if idx >= NENTS {
                break;
            }
            let y = 20 + (row0 + i as i32) * 18;
            let hi = idx as i32 == SEL;
            let c = if hi { "cc785c" } else { "5a5652" };
            ops.push_str(&format!("rect 4 {y} 122 16 {c}; "));
            let name = core::str::from_utf8(&NAMES[idx][..NAME_LEN[idx] as usize]).unwrap_or("?");
            let is_dir = IS_DIR[idx] != 0;
            let icon = if is_dir {
                crate::fa::FOLDER
            } else {
                crate::fa::FILE
            };
            let shown = if name.len() > 12 { &name[..12] } else { name };
            text_op(
                &mut ops,
                8,
                y + 2,
                10,
                "e8e4df",
                &format!("{icon} {shown}"),
            );
        }
        if NENTS == 0 && DEPTH == 0 {
            text_op(&mut ops, 10, 40, 10, "a8a4a0", "(empty store)");
        }
        if PREVIEW_LEN > 0 {
            let prev = core::str::from_utf8(&PREVIEW[..PREVIEW_LEN.min(200)]).unwrap_or("(binary)");
            // Printable only for preview.
            let clean: String = prev
                .chars()
                .map(|c| {
                    if c.is_ascii_graphic() || c == ' ' || c == '\n' {
                        c
                    } else {
                        '·'
                    }
                })
                .collect();
            let mut row = 0i32;
            let mut col = 0usize;
            let bytes = clean.as_bytes();
            while col < bytes.len() && row < 11 {
                let end = (col + 14).min(bytes.len());
                // char boundary
                let mut e = end;
                while e > col && !clean.is_char_boundary(e) {
                    e -= 1;
                }
                if e == col {
                    break;
                }
                text_op(
                    &mut ops,
                    138,
                    36 + row * 12,
                    9,
                    "e8e4df",
                    &clean[col..e],
                );
                col = e;
                row += 1;
            }
        } else if selected_is_dir() {
            text_op(&mut ops, 138, 40, 10, "a8a4a0", "(directory)");
            text_op(&mut ops, 138, 54, 9, "a8a4a0", "enter to open");
        } else if NENTS > 0 {
            text_op(&mut ops, 138, 40, 10, "a8a4a0", "(empty/binary)");
        }
    }
    ops.push_str("rect 0 176 256 16 3a3632; ");
    text_op(
        &mut ops,
        8,
        178,
        9,
        "a8a4a0",
        "↑↓ select  enter open  bs up  r reload",
    );
    ui_draw(&ops);
    hud_status(
        &format!("files  {}  {} items", cwd_string(), unsafe { NENTS }),
        "↑↓ select  enter open  backspace up  r reload",
    );
}

fn open_selected() {
    let name = selected_name();
    if name.is_empty() {
        return;
    }
    if selected_is_dir() {
        unsafe {
            if DEPTH < MAX_DEPTH {
                set_depth_component(DEPTH, &name);
                DEPTH += 1;
                SEL = 0;
                SCROLL = 0;
            }
        }
        reload();
        load_preview();
    } else {
        load_preview();
    }
}

fn go_up() {
    unsafe {
        if DEPTH > 0 {
            DEPTH -= 1;
            SEL = 0;
            SCROLL = 0;
        }
    }
    reload();
    load_preview();
}

pub fn start(_: &str) -> String {
    unsafe {
        DEPTH = 0;
        SEL = 0;
        SCROLL = 0;
    }
    reload();
    load_preview();
    paint();
    format!(
        "ok:files cwd={} ({} entries) — same store as /ls",
        cwd_string(),
        unsafe { NENTS }
    )
}

pub fn on_click(x: i32, y: i32) -> String {
    if x < 130 && y >= 16 && y < 176 {
        let row = (y - 20) / 18;
        if row < 0 {
            return status("");
        }
        unsafe {
            let has_parent = DEPTH > 0 && SCROLL == 0;
            if has_parent && row == 0 {
                // Click on ".."
                go_up();
                paint();
                return status("");
            }
            let ent_row = if has_parent { row - 1 } else { row };
            let idx = if DEPTH > 0 && SCROLL == 0 {
                ent_row
            } else if DEPTH > 0 {
                SCROLL - 1 + ent_row
            } else {
                SCROLL + ent_row
            };
            if idx >= 0 && (idx as usize) < NENTS {
                SEL = idx;
                load_preview();
                paint();
            }
        }
    }
    status("")
}

pub fn on_key(key: &str) -> String {
    match key {
        "up" | "ArrowUp" | "k" => unsafe {
            if SEL > 0 {
                SEL -= 1;
            }
            if SEL < SCROLL {
                SCROLL = SEL;
            }
            load_preview();
        },
        "down" | "ArrowDown" | "j" => unsafe {
            if (SEL as usize) + 1 < NENTS {
                SEL += 1;
            }
            if SEL >= SCROLL + VIS as i32 {
                SCROLL = SEL - VIS as i32 + 1;
            }
            load_preview();
        },
        "enter" | "Enter" | " " | "space" => open_selected(),
        "backspace" | "Backspace" | "Escape" | "esc" => go_up(),
        "r" => {
            reload();
            load_preview();
        }
        _ => {}
    }
    paint();
    status("")
}

pub fn list(_: &str) -> String {
    reload();
    let mut out = String::new();
    unsafe {
        for i in 0..NENTS {
            if i > 0 {
                out.push('\n');
            }
            let name = core::str::from_utf8(&NAMES[i][..NAME_LEN[i] as usize]).unwrap_or("?");
            if IS_DIR[i] != 0 {
                out.push_str(&format!("{name}/"));
            } else {
                out.push_str(name);
            }
        }
    }
    if out.is_empty() {
        format!("(empty) {}", cwd_string())
    } else {
        out
    }
}

pub fn get(args: &str) -> String {
    let key = json_str(args, "key")
        .or_else(|| json_str(args, "path"))
        .unwrap_or_default();
    if key.is_empty() {
        return String::from("error: need path");
    }
    let path = if key.starts_with('/') {
        key
    } else {
        join_cwd(&key)
    };
    let mut buf = [0u8; 8192];
    let n = fs_read(&path, &mut buf);
    if n < 0 {
        return format!("error: cannot read {path}");
    }
    String::from_utf8_lossy(&buf[..n as usize]).into_owned()
}

pub fn set(args: &str) -> String {
    // Write still goes through storage for package sandbox writes — FS write
    // is a separate host import we intentionally do not expose (taint/scope).
    let _ = args;
    String::from("error: files UI is browse-only; use shell /write or notes for durable notes")
}

pub fn remove(args: &str) -> String {
    let _ = args;
    String::from("error: files UI is browse-only; use shell /rm for deletes")
}

pub fn status(_: &str) -> String {
    format!(
        "ok:files cwd={} n={} sel={}",
        cwd_string(),
        unsafe { NENTS },
        selected_name()
    )
}
