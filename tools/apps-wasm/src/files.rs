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
/// Layout, derived rather than scattered as magic numbers.
///
/// The surface was 256x192 with every coordinate written out literally, which is
/// why the path was cut at 26 characters and names at 12 — there was nowhere for
/// them to go. `wasm.surface_w/h` in the manifest now asks for 640x400, so the
/// numbers live here and the rest of the file reads from them.
///
/// Text is unaffected by the size: `synapse::ui` defers labels and re-rasterizes
/// them at the pane's presentation scale, so glyphs stay sharp whatever the
/// surface is. What a bigger surface buys is *room* — rows, columns and a
/// preview that can show more than fourteen characters a line.
const SW: i32 = 640;
/// Portrait, because an action pane is tall. A landscape surface aspect-fits
/// into a band across the middle with dead space above and below — correct
/// geometry, terrible use of the pane. 640x800 is close to the column's own
/// ratio, so it very nearly fills it.
const SH: i32 = 800;
const HDR_H: i32 = 26;
const PAD: i32 = 6;
const ROW_H: i32 = 22;
/// Left list width; the preview takes the rest.
const LIST_W: i32 = 300;
const PREV_X: i32 = LIST_W + PAD * 2;
const PREV_W: i32 = SW - PREV_X - PAD;
/// First row's top edge.
const ROW0_Y: i32 = HDR_H + PAD;
/// Rows that fit. Kept under `LABEL_CAP` (96 deferred labels per surface) with
/// room for the header and the preview lines.
const VIS: usize = 30;

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
        // `n` is the listing's full length, which may exceed our buffer — the
        // browser shows what fits rather than growing, but it must clamp: slicing
        // by an unclamped host length is a guest trap.
        let n = (n as usize).min(buf.len());
        let raw = core::str::from_utf8(&buf[..n]).unwrap_or("");
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

/// What a file is, from its name. Drives the preview pane.
///
/// Extension rather than content sniffing because that is what `/open` itself
/// routes on (`command_hooks` match by extension), so the preview and the opener
/// agree about what a file is. A preview that called something an image which
/// `/open` then refused would be worse than saying nothing.
fn kind_of(name: &str) -> (&'static str, &'static str) {
    let lower = name.rsplit('.').next().unwrap_or("");
    let mut buf = [0u8; 8];
    let n = lower.len().min(8);
    buf[..n].copy_from_slice(&lower.as_bytes()[..n]);
    for c in buf.iter_mut() {
        c.make_ascii_lowercase();
    }
    let ext = core::str::from_utf8(&buf[..n]).unwrap_or("");
    match ext {
        "png" | "jpg" | "jpeg" => ("image", "opens in the image viewer"),
        "wav" | "mp3" | "aac" => ("audio", "opens in the audio player"),
        "mp4" | "mov" | "mkv" | "webm" | "ts" | "m3u8" => ("video", "opens in the video player"),
        "pdf" => ("PDF", "opens in the PDF viewer"),
        "wasm" => ("wasm module", "an agent's compiled tools"),
        "gguf" => ("model", "a GGUF language model"),
        "wad" => ("WAD", "game data"),
        "ogg" | "flac" => ("audio", "no decoder for this format yet"),
        "md" | "txt" | "json" | "toml" | "rs" | "c" | "h" | "js" | "html" | "css"
        | "sh" | "py" | "log" | "cfg" | "conf" | "" => ("text", "opens in the editor"),
        _ => ("file", "opens in the editor"),
    }
}

fn paint() {
    let cwd = cwd_string();
    let mut ops = String::from("clear 1a1816; ");
    ops.push_str(&format!("rect 0 0 {SW} {HDR_H} 3a3632; "));
    // Folder glyph (Font Awesome) + path. Truncation keeps the *end* of the
    // path, because the leaf is what tells you where you are.
    const PATH_MAX: usize = 62;
    let path_show = if cwd.len() > PATH_MAX {
        let cut = cwd.len() - (PATH_MAX - 2);
        let mut c = cut;
        while c < cwd.len() && !cwd.is_char_boundary(c) {
            c += 1;
        }
        format!("…{}", &cwd[c..])
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
    let body_h = SH - HDR_H;
    ops.push_str(&format!("rect 0 {HDR_H} {} {body_h} 2c2926; ", LIST_W + PAD * 2));
    ops.push_str(&format!("rect {PREV_X} {HDR_H} {PREV_W} {body_h} 2c2926; "));
    text_op(&mut ops, PREV_X + 4, HDR_H + 4, 11, "a8a4a0", "preview");
    unsafe {
        // Parent row when not at root.
        let mut row0 = 0i32;
        let start = SCROLL as usize;
        let mut painted = 0usize;
        if DEPTH > 0 && start == 0 {
            let y = ROW0_Y;
            let c = if SEL < 0 { "cc785c" } else { "3a3632" };
            ops.push_str(&format!("rect {PAD} {y} {LIST_W} {} {c}; ", ROW_H - 2));
            text_op(
                &mut ops,
                PAD + 6,
                y + 3,
                11,
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
            let y = ROW0_Y + (row0 + i as i32) * ROW_H;
            let hi = idx as i32 == SEL;
            let c = if hi { "cc785c" } else { "5a5652" };
            ops.push_str(&format!("rect {PAD} {y} {LIST_W} {} {c}; ", ROW_H - 2));
            let name = core::str::from_utf8(&NAMES[idx][..NAME_LEN[idx] as usize]).unwrap_or("?");
            let is_dir = IS_DIR[idx] != 0;
            let icon = if is_dir {
                crate::fa::FOLDER
            } else {
                crate::fa::FILE
            };
            // Room for a real filename now. Still bounded, and cut on a char
            // boundary so a multi-byte name cannot panic the slice.
            const NAME_MAX: usize = 34;
            let shown = if name.len() > NAME_MAX {
                let mut e = NAME_MAX;
                while e > 0 && !name.is_char_boundary(e) {
                    e -= 1;
                }
                &name[..e]
            } else {
                name
            };
            text_op(
                &mut ops,
                PAD + 6,
                y + 3,
                11,
                "e8e4df",
                &format!("{icon} {shown}"),
            );
        }
        if NENTS == 0 && DEPTH == 0 {
            text_op(&mut ops, PAD + 6, ROW0_Y + 4, 11, "a8a4a0", "(empty store)");
        }
        // Name the kind first, whatever it is. The old preview showed a text dump
        // or nothing at all, so an image, a video and an unreadable file were
        // indistinguishable — all three rendered as an empty pane.
        let sel_name: Option<&str> = if SEL >= 0 && (SEL as usize) < NENTS {
            let i = SEL as usize;
            core::str::from_utf8(&NAMES[i][..NAME_LEN[i] as usize]).ok()
        } else {
            None
        };
        let mut is_text = true;
        if let Some(nm) = sel_name {
            if !selected_is_dir() {
                let (kind, hint) = kind_of(nm);
                is_text = kind == "text";
                text_op(&mut ops, PREV_X + 4, HDR_H + 22, 11, "cc785c", kind);
                text_op(&mut ops, PREV_X + 4, HDR_H + 40, 10, "a8a4a0", hint);
            }
        }
        if !is_text && PREVIEW_LEN > 0 {
            // Binary: show the leading bytes rather than a screen of `·`.
            let mut hex = String::new();
            for b in PREVIEW[..PREVIEW_LEN.min(16)].iter() {
                hex.push_str(&format!("{b:02x} "));
            }
            text_op(&mut ops, PREV_X + 4, HDR_H + 64, 10, "e8e4df", &hex);
        } else if PREVIEW_LEN > 0 {
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
            // Roughly one character per 6 px at size 10, which the pane's own
            // scale then sharpens. Was 14 characters over 11 lines — a preview
            // that could not show a line of code.
            const PREV_COLS: usize = 46;
            const PREV_ROWS: i32 = 42;
            let mut row = 0i32;
            let mut col = 0usize;
            let bytes = clean.as_bytes();
            while col < bytes.len() && row < PREV_ROWS {
                // Break at the newline if one falls inside this line, so text
                // files keep their own shape instead of being reflowed.
                let hard = clean[col..].find('\n').map(|i| col + i);
                let end = match hard {
                    Some(h) if h < col + PREV_COLS => h,
                    _ => (col + PREV_COLS).min(bytes.len()),
                };
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
                    PREV_X + 4,
                    HDR_H + 64 + row * 15,
                    10,
                    "e8e4df",
                    clean[col..e].trim_end_matches('\n'),
                );
                // Skip the newline we broke on, or we emit a blank line for it.
                col = if e < bytes.len() && bytes[e] == b'\n' { e + 1 } else { e };
                row += 1;
            }
        } else if selected_is_dir() {
            text_op(&mut ops, PREV_X + 4, HDR_H + 24, 11, "a8a4a0", "(directory)");
            text_op(&mut ops, PREV_X + 4, HDR_H + 42, 10, "a8a4a0", "enter or click to open");
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
    // Must track the paint geometry above: a hit map that disagrees with what is
    // drawn sends a click to the row above or below the one under the pointer.
    if x < LIST_W + PAD * 2 && y >= HDR_H && y < SH {
        let row = (y - ROW0_Y) / ROW_H;
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
        // `SEL == -1` **is** the ".." row — that is how `paint` draws the
        // highlight and how `on_click` routes it. The guard was `SEL > 0`, so the
        // selection could never reach -1 and the parent row was clickable but
        // unreachable by keyboard: the one navigation every file browser needs.
        "up" | "ArrowUp" | "k" => unsafe {
            let floor = if DEPTH > 0 { -1 } else { 0 };
            if SEL > floor {
                SEL -= 1;
            }
            if SEL >= 0 && SEL < SCROLL {
                SCROLL = SEL;
            }
            if SEL < 0 {
                SCROLL = 0;
            }
            load_preview();
        },
        "down" | "ArrowDown" | "j" => unsafe {
            if SEL < 0 {
                SEL = 0;
            } else if (SEL as usize) + 1 < NENTS {
                SEL += 1;
            }
            if SEL >= SCROLL + VIS as i32 {
                SCROLL = SEL - VIS as i32 + 1;
            }
            load_preview();
        },
        "home" | "Home" => unsafe {
            SEL = if DEPTH > 0 { -1 } else { 0 };
            SCROLL = 0;
            load_preview();
        },
        "end" | "End" => unsafe {
            if NENTS > 0 {
                SEL = NENTS as i32 - 1;
                SCROLL = (NENTS as i32 - VIS as i32).max(0);
            }
            load_preview();
        },
        // Enter on ".." goes up, the same as clicking it.
        "enter" | "Enter" | " " | "space" => unsafe {
            if SEL < 0 {
                go_up();
            } else {
                open_selected();
            }
        },
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
    // `n` is the file's full length; a bigger file is shown truncated, but the
    // slice must be clamped — see `reload`.
    String::from_utf8_lossy(&buf[..(n as usize).min(buf.len())]).into_owned()
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
