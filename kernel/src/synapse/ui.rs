//! **UI surfaces**: a capability-gated drawing surface an agent owns and paints
//! with a bounded draw-op DSL. This is how a Chess / Image / Video / Browser /
//! Doc agent renders — it never scribbles pixels directly. The model emits a
//! grammar-validated `ui_draw` call whose `ops` string is *itself* validated
//! against the small draw grammar here, rasterized into the surface's own
//! backing buffer, and then presented by the compositor. Two layers keep the
//! determinism boundary intact: the outer Synapse grammar validates the call
//! shape; this module validates the draw ops and clamps every coordinate to the
//! surface's own bounds (an agent cannot draw outside its surface).
//!
//! Authority is by **ownership**: `ui_surface_request` records the owner task,
//! and `ui_draw`/`ui_event_poll`/`ui_surface_close` refuse any caller that is
//! not the owner. So even though a surface id is a small global integer, a
//! non-owner naming it is denied — no ambient authority over another agent's
//! surface.

use crate::mm::Locked;
use crate::sched::TaskId;
use alloc::collections::{BTreeMap, VecDeque};
use alloc::vec;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU32, Ordering};

/// Fixed logical surface size. The compositor scales/letterboxes this into the
/// action pane; keeping it fixed makes rasterization deterministic (testable).
pub const SURF_W: usize = 256;
pub const SURF_H: usize = 192;

/// Longest string a single `text` op may carry (bounded so labels stay `Copy`
/// and a hostile program can't allocate unboundedly).
const TEXT_MAX: usize = 48;

/// A surface's intended content kind — a hint for the compositor and a bound the
/// grant can name. `Board` = chess/dynamic grid; the rest are self-describing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SurfaceKind {
    Canvas,
    Board,
    Image,
    Video,
    Html,
}

impl SurfaceKind {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "canvas" => SurfaceKind::Canvas,
            "board" => SurfaceKind::Board,
            "image" => SurfaceKind::Image,
            "video" => SurfaceKind::Video,
            "html" => SurfaceKind::Html,
            _ => return None,
        })
    }
}

/// Deferred text label. Geometry is in **logical** surface coords (256×192);
/// the compositor re-rasterizes at the pane's presentation scale so glyphs
/// match the sharp HUD/console font instead of looking like upscaled pixel soup.
#[derive(Clone, Copy, Debug)]
struct TextLabel {
    x: i32,
    y: i32,
    size: i32,
    color: u32,
    buf: [u8; TEXT_MAX],
    len: u8,
}

/// Cap deferred labels so a hostile draw program can't grow the surface without
/// bound. Plenty for any real package UI (settings ~30 labels).
const LABEL_CAP: usize = 96;

struct Surface {
    owner: TaskId,
    #[allow(dead_code)] // recorded for the compositor / future kind-specific layout
    kind: SurfaceKind,
    /// 0xRRGGBB pixels, row-major, `SURF_W * SURF_H`. Geometry only — text is
    /// in [`Self::labels`] and painted at present scale.
    back: Vec<u32>,
    /// Deferred `text` ops (see [`TextLabel`]).
    labels: Vec<TextLabel>,
    /// Input events the compositor routed to this surface (mouse/keys), drained
    /// by `ui_event_poll`. Empty headless.
    events: VecDeque<UiEvent>,
    /// Optional HUD text (see [`set_hud`]). Rendered by the compositor in a
    /// reserved **pane-space** strip below the scaled surface — native crisp
    /// font, wrapped to the pane width — NOT baked into the scaled backing
    /// buffer (which would balloon the text with the surface's upscale).
    hud: alloc::string::String,
}

/// An input event delivered to a surface's owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UiEvent {
    Click { x: u16, y: u16 },
    Key(u8),
}

static NEXT_SURFACE: AtomicU32 = AtomicU32::new(1);
static SURFACES: Locked<BTreeMap<u32, Surface>> = Locked::new(BTreeMap::new());
/// When true, [`draw`] / board ops update the backing buffer only — no
/// compositor present. Package-UI init draws dozens of ops; presenting each
/// one at full pane scale (TTF re-raster) freezes the shell for a long time.
static DEFER_PRESENT: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);
/// Surfaces dirtied while present was deferred (flushed by [`flush_deferred`]).
static DIRTY: Locked<alloc::vec::Vec<u32>> = Locked::new(alloc::vec::Vec::new());

/// Run `f` without compositor presents; one present per dirtied surface at the end.
pub fn with_deferred_present(f: impl FnOnce()) {
    DEFER_PRESENT.store(true, Ordering::SeqCst);
    f();
    DEFER_PRESENT.store(false, Ordering::SeqCst);
    flush_deferred();
}

fn mark_dirty(id: u32) {
    DIRTY.with(|v| {
        if !v.iter().any(|&x| x == id) {
            v.push(id);
        }
    });
}

fn flush_deferred() {
    let ids = DIRTY.with(|v| core::mem::take(v));
    #[cfg(not(test))]
    for id in ids {
        present(id);
    }
    #[cfg(test)]
    let _ = ids;
}

/// Present now, or mark dirty if deferred (package-UI init batches presents).
#[cfg(not(test))]
fn maybe_present(id: u32) {
    if DEFER_PRESENT.load(Ordering::Relaxed) {
        mark_dirty(id);
    } else {
        present(id);
    }
}
#[cfg(test)]
fn maybe_present(_id: u32) {}

#[derive(Debug, PartialEq, Eq)]
pub enum DrawErr {
    NotOwner,
    NoSuchSurface,
    BadOp(&'static str),
}

/// Request a new surface owned by `owner`. Returns its id (used as the `surface`
/// arg in later draw calls; ownership — not the id — is the gate).
pub fn request(owner: TaskId, kind: SurfaceKind) -> u32 {
    let id = NEXT_SURFACE.fetch_add(1, Ordering::SeqCst);
    SURFACES.with(|m| {
        m.insert(
            id,
            Surface {
                owner,
                kind,
                back: vec![0u32; SURF_W * SURF_H],
                labels: Vec::new(),
                events: VecDeque::new(),
                hud: alloc::string::String::new(),
            },
        );
    });
    id
}

/// Whether `task` owns surface `id`.
pub fn owns(task: TaskId, id: u32) -> bool {
    SURFACES.with(|m| m.get(&id).map(|s| s.owner == task).unwrap_or(false))
}

/// Apply a draw-op program to a surface the caller owns. Returns the number of
/// ops applied. Every coordinate is clamped to the surface bounds.
pub fn draw(task: TaskId, id: u32, ops: &str) -> Result<usize, DrawErr> {
    let program = parse_ops(ops)?;
    let n = SURFACES.with(|m| {
        let s = m.get_mut(&id).ok_or(DrawErr::NoSuchSurface)?;
        if s.owner != task {
            return Err(DrawErr::NotOwner);
        }
        for op in &program {
            apply_surface(s, *op);
        }
        Ok(program.len())
    })?;
    maybe_present(id);
    Ok(n)
}

/// Drain one queued input event for the owner, if any.
pub fn poll(task: TaskId, id: u32) -> Result<Option<UiEvent>, DrawErr> {
    SURFACES.with(|m| {
        let s = m.get_mut(&id).ok_or(DrawErr::NoSuchSurface)?;
        if s.owner != task {
            return Err(DrawErr::NotOwner);
        }
        Ok(s.events.pop_front())
    })
}

/// Close a surface the caller owns.
pub fn close(task: TaskId, id: u32) -> Result<(), DrawErr> {
    SURFACES.with(|m| {
        match m.get(&id) {
            Some(s) if s.owner != task => return Err(DrawErr::NotOwner),
            Some(_) => {}
            None => return Err(DrawErr::NoSuchSurface),
        }
        m.remove(&id);
        Ok(())
    })
}

/// A stable checksum of a surface's backing buffer — for tests / an e2e proof
/// that draw ops rasterized deterministically (pixels aren't visible on serial).
pub fn checksum(id: u32) -> Option<u64> {
    SURFACES.with(|m| {
        m.get(&id).map(|s| {
            let mut h: u64 = 0xcbf2_9ce4_8422_2325; // FNV-1a
            for &px in &s.back {
                for b in px.to_le_bytes() {
                    h ^= b as u64;
                    h = h.wrapping_mul(0x0000_0100_0000_01b3);
                }
            }
            h
        })
    })
}

/// True if surface `id` has queued input events. **Upkeep pacing only** — it
/// grants nothing: events are still consumed exclusively through the
/// capability-gated `ui_event_poll` primitive (audited per drain). Without
/// this peek the pump had to no-op poll through the capability layer at
/// upkeep rate, flooding the audit log with ~1000 identical entries/second.
pub fn has_events(id: u32) -> bool {
    SURFACES.with(|m| m.get(&id).map(|s| !s.events.is_empty()).unwrap_or(false))
}

/// Feed an input event to a surface (called by the compositor on a click/key
/// over the surface's pane). No-op if the surface is gone.
pub fn push_event(id: u32, ev: UiEvent) {
    SURFACES.with(|m| {
        if let Some(s) = m.get_mut(&id) {
            s.events.push_back(ev);
        }
    });
}

// --- the bounded draw-op DSL -------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DrawOp {
    Clear(u32),
    Rect { x: i32, y: i32, w: i32, h: i32, color: u32 },
    Line { x0: i32, y0: i32, x1: i32, y1: i32, color: u32 },
    Pixel { x: i32, y: i32, color: u32 },
    /// `size` = glyph pixel height (8..=22, nearest-neighbour scaled from the
    /// 10×22 Geist Mono cell). Bounded inline string keeps the op `Copy`.
    Text { x: i32, y: i32, size: i32, color: u32, buf: [u8; TEXT_MAX], len: u8 },
}

/// Parse a `;`-separated draw program: `clear <hex>`, `rect x y w h <hex>`,
/// `line x0 y0 x1 y1 <hex>`, `pixel x y <hex>`, and
/// `text x y size <hex> <string…>` (string = rest of the statement, so it may
/// contain spaces but never `;`). Colours are RRGGBB hex.
fn parse_ops(program: &str) -> Result<Vec<DrawOp>, DrawErr> {
    let mut out = Vec::new();
    for stmt in program.split(';') {
        let stmt = stmt.trim();
        if stmt.is_empty() {
            continue;
        }
        let toks: Vec<&str> = stmt.split_whitespace().collect();
        let op = toks[0];
        // `text` has its own shape (colour mid-statement, free string tail).
        if op == "text" {
            if toks.len() < 5 {
                return Err(DrawErr::BadOp("text needs x y size colour string"));
            }
            let x: i32 = toks[1].parse().map_err(|_| DrawErr::BadOp("bad coord"))?;
            let y: i32 = toks[2].parse().map_err(|_| DrawErr::BadOp("bad coord"))?;
            let size: i32 = toks[3].parse().map_err(|_| DrawErr::BadOp("bad size"))?;
            let color = u32::from_str_radix(toks[4], 16).map_err(|_| DrawErr::BadOp("bad colour"))? & 0x00ff_ffff;
            // The string is the raw remainder after the colour token (keeps
            // single spaces; a leading/trailing trim is fine for labels).
            let after = stmt
                .find(toks[4])
                .map(|i| stmt[i + toks[4].len()..].trim_start())
                .unwrap_or("");
            let mut buf = [0u8; TEXT_MAX];
            let bytes = after.as_bytes();
            let len = bytes.len().min(TEXT_MAX);
            buf[..len].copy_from_slice(&bytes[..len]);
            out.push(DrawOp::Text { x, y, size: size.clamp(8, 22), color, buf, len: len as u8 });
            continue;
        }
        // Grammar: every op ends with an RRGGBB hex colour; the tokens between
        // the op name and the colour are decimal coordinates. Parsing the colour
        // positionally (last token) avoids the ambiguity of an all-digit hex
        // that would otherwise parse as a decimal coord.
        if toks.len() < 2 {
            return Err(DrawErr::BadOp("missing colour"));
        }
        let c = u32::from_str_radix(toks[toks.len() - 1], 16).map_err(|_| DrawErr::BadOp("bad colour"))? & 0x00ff_ffff;
        let mut nums: Vec<i32> = Vec::new();
        for tok in &toks[1..toks.len() - 1] {
            nums.push(tok.parse::<i32>().map_err(|_| DrawErr::BadOp("bad coord"))?);
        }
        out.push(match op {
            "clear" => DrawOp::Clear(c),
            "rect" if nums.len() == 4 => DrawOp::Rect { x: nums[0], y: nums[1], w: nums[2], h: nums[3], color: c },
            "line" if nums.len() == 4 => DrawOp::Line { x0: nums[0], y0: nums[1], x1: nums[2], y1: nums[3], color: c },
            "pixel" if nums.len() == 2 => DrawOp::Pixel { x: nums[0], y: nums[1], color: c },
            _ => return Err(DrawErr::BadOp("unknown op / wrong arity")),
        });
    }
    Ok(out)
}

fn put(back: &mut [u32], x: i32, y: i32, color: u32) {
    if x >= 0 && y >= 0 && (x as usize) < SURF_W && (y as usize) < SURF_H {
        back[y as usize * SURF_W + x as usize] = color;
    }
}

fn apply_surface(s: &mut Surface, op: DrawOp) {
    match op {
        DrawOp::Clear(c) => {
            s.back.iter_mut().for_each(|p| *p = c);
            s.labels.clear();
        }
        DrawOp::Rect { x, y, w, h, color } => {
            for dy in 0..h.max(0) {
                for dx in 0..w.max(0) {
                    put(&mut s.back, x + dx, y + dy, color);
                }
            }
        }
        DrawOp::Line { x0, y0, x1, y1, color } => {
            // Integer Bresenham, clamped by `put`.
            let (mut x, mut y) = (x0, y0);
            let dx = (x1 - x0).abs();
            let dy = -(y1 - y0).abs();
            let sx = if x0 < x1 { 1 } else { -1 };
            let sy = if y0 < y1 { 1 } else { -1 };
            let mut err = dx + dy;
            loop {
                put(&mut s.back, x, y, color);
                if x == x1 && y == y1 {
                    break;
                }
                let e2 = 2 * err;
                if e2 >= dy {
                    err += dy;
                    x += sx;
                }
                if e2 <= dx {
                    err += dx;
                    y += sy;
                }
            }
        }
        DrawOp::Pixel { x, y, color } => put(&mut s.back, x, y, color),
        DrawOp::Text {
            x,
            y,
            size,
            color,
            buf,
            len,
        } => {
            // Defer: painted at present scale (see [`present`]). Also bake a
            // 1× copy into `back` under test so checksums still see ink.
            if s.labels.len() < LABEL_CAP {
                s.labels.push(TextLabel {
                    x,
                    y,
                    size,
                    color,
                    buf,
                    len,
                });
            }
            #[cfg(test)]
            if let Ok(txt) = core::str::from_utf8(&buf[..len as usize]) {
                draw_text_lo(&mut s.back, x, y, size, color, txt);
            }
        }
    }
}

/// 1× logical-surface raster (tests / fallback). Display path uses [`present`].
fn draw_text_lo(back: &mut [u32], x: i32, y: i32, size: i32, color: u32, s: &str) {
    let _ = crate::font_ttf::blit_run(back, SURF_W, SURF_H, x, y, s, size as f32, color);
}

#[inline]
fn f32_round_i(v: f32) -> i32 {
    let f = if v >= 0.0 { v + 0.5 } else { v - 0.5 };
    f as i32
}

/// Whether surface `id` still has a backing buffer (for tab-switch repaint).
pub fn has_surface(id: u32) -> bool {
    SURFACES.with(|m| m.contains_key(&id))
}

/// Present geometry (nearest-upscaled) + **re-rasterized** labels at the pane's
/// presentation scale so text matches the sharp HUD font, not a blown-up 10 px
/// glyph. The HUD strip itself is still drawn by the compositor.
///
/// Gated on [`package_ui::should_present`]: after the user closes the canvas a
/// late guest draw updates the backing buffer only and must not reopen the tab.
#[cfg(not(test))]
fn present(id: u32) {
    if !crate::service::package_ui::should_present(id) {
        return;
    }
    present_forced(id);
}

/// Present without the package-UI dismiss gate — used for explicit tab focus
/// (`represent`) so switching back to a live app always repaints.
#[cfg(not(test))]
fn present_forced(id: u32) {
    let snap = SURFACES.with(|m| {
        m.get(&id)
            .map(|s| (s.back.clone(), s.labels.clone(), s.hud.clone()))
    });
    let Some((back, labels, hud)) = snap else {
        return;
    };
    // Usable pane (minus HUD strip) so fit matches what present_surface_reserve
    // will do with the same hud text.
    // Fit to the column this surface's tab actually lives in — a package-UI app
    // dragged to another column must re-scale to that column, not the focused one.
    let (pw, ph_full) = crate::framebuffer::surface_dims_px(id)
        .unwrap_or((SURF_W as u64, SURF_H as u64));
    let reserve = crate::framebuffer::surface_hud_reserve(&hud);
    let ph = ph_full.saturating_sub(reserve);
    let (dw, dh) = crate::framebuffer::present_fit(SURF_W as u64, SURF_H as u64, pw, ph);
    if dw == 0 || dh == 0 {
        return;
    }
    let scale_x = dw as f32 / SURF_W as f32;
    let scale_y = dh as f32 / SURF_H as f32;
    // Uniform text scale (min axis) so glyphs aren't stretched.
    let scale_t = if scale_x < scale_y { scale_x } else { scale_y };
    // Nearest-neighbour upscale of geometry into a presentation buffer, then
    // paint labels at display resolution with full AA (Geist Mono).
    let mut full = vec![0u32; (dw * dh) as usize];
    let dw_u = dw as usize;
    let dh_u = dh as usize;
    for dy in 0..dh_u {
        let sy = (dy as u64 * SURF_H as u64 / dh) as usize;
        let srow = sy * SURF_W;
        let drow = dy * dw_u;
        for dx in 0..dw_u {
            let sx = (dx as u64 * SURF_W as u64 / dw) as usize;
            full[drow + dx] = back[srow + sx];
        }
    }
    for lab in &labels {
        let txt = match core::str::from_utf8(&lab.buf[..lab.len as usize]) {
            Ok(t) if !t.is_empty() => t,
            _ => continue,
        };
        let px = f32_round_i(lab.x as f32 * scale_x);
        let py = f32_round_i(lab.y as f32 * scale_y);
        // Keep a readable floor; logical sizes 8–22 become ~pane-proportional.
        let psz = {
            let p = lab.size as f32 * scale_t;
            if p < 12.0 {
                12.0
            } else if p > 64.0 {
                64.0
            } else {
                p
            }
        };
        let _ = crate::font_ttf::blit_run(
            &mut full,
            dw_u,
            dh_u,
            px,
            py,
            txt,
            psz,
            lab.color,
        );
    }
    // Buffer is presentation-sized; hit-testing still uses logical 256×192.
    crate::framebuffer::present_surface_hud_ex(
        id,
        SURF_W,
        SURF_H,
        dw_u,
        dh_u,
        &full,
        &hud,
    );
}

/// Re-present a surface (e.g. after a pane resize / tab switch), from its own
/// backing buffer — the source of truth. No-op if the surface is gone or the
/// package UI was dismissed (same gate as draw-path present).
#[cfg(not(test))]
pub fn represent(id: u32) {
    present(id);
}

/// Set (or clear, with `""`) a surface's HUD text and re-present. Owner-gated.
/// The text is newline-separated lines: line 0 is the **status** (accent),
/// remaining lines are **hints** (dim, word-wrapped to the pane width). The
/// compositor renders it in a reserved strip in pane space — see [`Surface::hud`].
pub fn set_hud(task: TaskId, id: u32, text: &str) -> Result<(), DrawErr> {
    SURFACES.with(|m| {
        let s = m.get_mut(&id).ok_or(DrawErr::NoSuchSurface)?;
        if s.owner != task {
            return Err(DrawErr::NotOwner);
        }
        s.hud.clear();
        s.hud.push_str(text);
        Ok(())
    })?;
    maybe_present(id);
    Ok(())
}

// --- Board helpers (chess / grid UI agents) ---------------------------------
//
// Pure presentation: FEN / square marks → draw-op programs. Game rules live in
// the agent's SOUL (or optional later native validators), not here.

/// Square size for an 8×8 board letterboxed into the surface: 8*24 = 192 px
/// fills the surface height, centred horizontally. The status/shortcut HUD is
/// NOT in the surface — it's a reserved pane-space strip (crisp native text,
/// see [`set_hud`]), so the whole surface is board.
pub const BOARD_SQ: i32 = 24;
pub const BOARD_OX: i32 = 32;
pub const BOARD_OY: i32 = 0;

/// Map a surface-local click to a chess square `"a1"`…`"h8"`, or `None` if
/// outside the board. Pure — unit-tested.
pub fn click_to_square(x: u16, y: u16) -> Option<alloc::string::String> {
    let x = x as i32 - BOARD_OX;
    let y = y as i32 - BOARD_OY;
    if x < 0 || y < 0 || x >= 8 * BOARD_SQ || y >= 8 * BOARD_SQ {
        return None;
    }
    let file = (x / BOARD_SQ) as u8;
    let rank_from_top = (y / BOARD_SQ) as u8;
    // Board is drawn with rank 8 at the top (standard diagram).
    let rank = 7 - rank_from_top;
    let mut s = alloc::string::String::new();
    s.push((b'a' + file) as char);
    s.push((b'1' + rank) as char);
    Some(s)
}

/// Parse square `"e4"` → (file 0..7, rank 0..7) with rank 0 = white's first rank.
pub fn parse_square(sq: &str) -> Option<(u8, u8)> {
    let b = sq.as_bytes();
    if b.len() != 2 {
        return None;
    }
    let file = b[0].wrapping_sub(b'a');
    let rank = b[1].wrapping_sub(b'1');
    if file < 8 && rank < 8 {
        Some((file, rank))
    } else {
        None
    }
}

/// Build a draw-op program for an 8×8 board from a FEN placement field (or a
/// full FEN — only the first field is used). Empty / invalid → empty board.
/// Pure: returns the ops string; the caller paints via [`draw`].
pub fn board_ops_from_fen(fen: &str) -> alloc::string::String {
    use alloc::format;
    let placement = fen.split_whitespace().next().unwrap_or(fen);
    let mut grid = [[' '; 8]; 8];
    let mut r: usize = 0;
    let mut f: usize = 0;
    for c in placement.chars() {
        if r >= 8 {
            break;
        }
        match c {
            '/' => {
                r += 1;
                f = 0;
            }
            '1'..='8' => {
                let n = (c as u8 - b'0') as usize;
                f = (f + n).min(8);
            }
            p if p.is_ascii_alphabetic() => {
                if f < 8 {
                    grid[r][f] = p;
                    f += 1;
                }
            }
            _ => {}
        }
    }
    // FEN ranks are top-down (black's side first); grid[0] = rank 8.
    let light = 0xee_ee_d2u32;
    let dark = 0x76_96_56u32;
    let mut ops = format!("clear 1a1a2e; ");
    for rank_top in 0..8i32 {
        for file in 0..8i32 {
            let x = BOARD_OX + file * BOARD_SQ;
            let y = BOARD_OY + rank_top * BOARD_SQ;
            let color = if (file + rank_top) % 2 == 0 { light } else { dark };
            ops.push_str(&format!("rect {x} {y} {BOARD_SQ} {BOARD_SQ} {color:06x}; "));
            let piece = grid[rank_top as usize][file as usize];
            if piece != ' ' {
                let (fill, letter) = piece_style(piece);
                paint_piece_ops(&mut ops, x, y, piece, fill, letter);
            }
        }
    }
    ops
}

/// (fill, letter) colours: white pieces are light chips with dark letters,
/// black pieces dark chips with light letters — the letter always contrasts
/// with its own chip, so it reads on both square colours.
fn piece_style(p: char) -> (u32, u32) {
    if p.is_ascii_uppercase() {
        (0xf0_f0_f0, 0x1a_1a_1a)
    } else {
        (0x22_22_22, 0xe8_e4_df)
    }
}

/// Chess piece **icons** from Font Awesome Free Solid (`chess-pawn` …
/// `chess-king`). A soft chip pad keeps light pieces readable on light squares;
/// the glyph colour is the contrasting ink from [`piece_style`].
fn paint_piece_ops(ops: &mut alloc::string::String, x: i32, y: i32, piece: char, fill: u32, letter_color: u32) {
    use alloc::format;
    let cx = x + BOARD_SQ / 2;
    let cy = y + BOARD_SQ / 2;
    let f = fill & 0xff_ffff;
    let edge = letter_color & 0xff_ffff;
    let glyph = crate::icons::chess_piece(piece);
    // Soft pad under every piece so light pieces stay readable on light squares.
    ops.push_str(&format!("rect {} {} 18 18 {f:06x}; ", cx - 9, cy - 9));
    // FA chess glyphs ~16 px in a 24 px square, centred.
    let size = 16;
    let tx = cx - size / 2;
    let ty = cy - size / 2;
    // Host rasterizes via the FA face (first in the TTF fallback chain).
    ops.push_str(&format!("text {tx} {ty} {size} {edge:06x} {glyph}; "));
}

/// Highlight squares (e.g. `"e2,e4"`) with a translucent-ish overlay colour.
pub fn board_mark_ops(squares: &str, color_hex: &str) -> alloc::string::String {
    use alloc::format;
    let color = u32::from_str_radix(color_hex.trim().trim_start_matches('#'), 16).unwrap_or(0xcc_78_5c) & 0xff_ffff;
    let mut ops = alloc::string::String::new();
    for tok in squares.split(|c: char| c == ',' || c.is_whitespace()) {
        let tok = tok.trim();
        if tok.is_empty() {
            continue;
        }
        if let Some((file, rank)) = parse_square(tok) {
            let x = BOARD_OX + file as i32 * BOARD_SQ;
            // rank 0 at bottom → top y = (7-rank)*sq
            let y = BOARD_OY + (7 - rank as i32) * BOARD_SQ;
            // Inset border so the mark is visible without wiping the piece.
            ops.push_str(&format!(
                "rect {x} {y} {BOARD_SQ} 2 {color:06x}; rect {x} {} {BOARD_SQ} 2 {color:06x}; rect {x} {y} 2 {BOARD_SQ} {color:06x}; rect {} {y} 2 {BOARD_SQ} {color:06x}; ",
                y + BOARD_SQ - 2,
                x + BOARD_SQ - 2
            ));
        }
    }
    ops
}

/// Paint a FEN onto a surface the caller owns (high-level board tool).
pub fn board_set(task: TaskId, id: u32, fen: &str) -> Result<usize, DrawErr> {
    let ops = board_ops_from_fen(fen);
    draw(task, id, &ops)
}

/// Overlay square marks on a surface the caller owns.
pub fn board_mark(task: TaskId, id: u32, squares: &str, color: &str) -> Result<usize, DrawErr> {
    let ops = board_mark_ops(squares, color);
    if ops.is_empty() {
        return Ok(0);
    }
    draw(task, id, &ops)
}

#[cfg(test)]
pub fn reset() {
    SURFACES.with(|m| m.clear());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn rasterizes_clear_rect_pixel_deterministically() {
        reset();
        let owner = crate::sched::current_task_id();
        let id = request(owner, SurfaceKind::Canvas);
        let n = draw(owner, id, "clear 000000; rect 2 2 3 3 ff0000; pixel 10 10 00ff00").unwrap();
        assert_eq!(n, 3);
        // Spot-check the backing buffer.
        let (rect_px, pixel_px, bg_px) = SURFACES.with(|m| {
            let b = &m.get(&id).unwrap().back;
            (b[3 * SURF_W + 3], b[10 * SURF_W + 10], b[0])
        });
        assert_eq!(rect_px, 0xff0000, "rect interior should be red");
        assert_eq!(pixel_px, 0x00ff00, "pixel should be green");
        assert_eq!(bg_px, 0x000000, "cleared background should be black");
        close(owner, id).unwrap();
    }

    #[test_case]
    fn draw_clamps_out_of_bounds_and_never_panics() {
        reset();
        let owner = crate::sched::current_task_id();
        let id = request(owner, SurfaceKind::Canvas);
        // A rect and line reaching outside bounds must be clamped, not crash.
        let n = draw(owner, id, "rect -50 -50 400 400 112233; line -100 -100 999 999 445566").unwrap();
        assert_eq!(n, 2);
        // The rect covers the top-left region; check a pixel inside it but off
        // the diagonal line (which would otherwise overwrite it).
        let px = SURFACES.with(|m| m.get(&id).unwrap().back[0 * SURF_W + 10]);
        assert_eq!(px, 0x112233, "clamped rect should paint (10,0)");
        close(owner, id).unwrap();
    }

    #[test_case]
    fn only_the_owner_may_draw() {
        reset();
        let owner = crate::sched::current_task_id();
        let id = request(owner, SurfaceKind::Board);
        // A different task id cannot draw to or close this surface.
        let intruder = owner + 9999;
        assert_eq!(draw(intruder, id, "clear ffffff"), Err(DrawErr::NotOwner));
        assert_eq!(close(intruder, id), Err(DrawErr::NotOwner));
        // The owner still can.
        assert!(draw(owner, id, "clear ffffff").is_ok());
        close(owner, id).unwrap();
    }

    #[test_case]
    fn text_op_rasterizes_and_is_bounded() {
        reset();
        let owner = crate::sched::current_task_id();
        let id = request(owner, SurfaceKind::Canvas);
        // Text blends onto the background; assert some pixel in the glyph box
        // changed from the cleared colour and the program parses with spaces.
        draw(owner, id, "clear 000000").unwrap();
        let before = checksum(id).unwrap();
        let n = draw(owner, id, "text 4 4 22 ffffff Hi there").unwrap();
        assert_eq!(n, 1);
        assert_ne!(checksum(id).unwrap(), before, "text must change pixels");
        // Deterministic: same program → same checksum.
        draw(owner, id, "clear 000000; text 4 4 22 ffffff Hi there").unwrap();
        let h1 = checksum(id).unwrap();
        draw(owner, id, "clear 000000; text 4 4 22 ffffff Hi there").unwrap();
        assert_eq!(checksum(id).unwrap(), h1);
        // Over-long strings are truncated (op still applies), bad shapes error.
        let long = "text 0 40 22 ff0000 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        assert!(draw(owner, id, long).is_ok());
        assert!(matches!(draw(owner, id, "text 1 2 22 zz oops"), Err(DrawErr::BadOp(_))));
        assert!(matches!(draw(owner, id, "text 1 2"), Err(DrawErr::BadOp(_))));
        close(owner, id).unwrap();
    }

    #[test_case]
    fn text_op_defers_labels_and_bakes_for_tests() {
        // Live path defers labels for present-scale raster; under test we also
        // bake 1× ink so checksums still see text, and labels are recorded.
        reset();
        let owner = crate::sched::current_task_id();
        let id = request(owner, SurfaceKind::Canvas);
        draw(owner, id, "clear 000000; text 8 8 14 ffffff Settings").unwrap();
        let (n_labels, ink) = SURFACES.with(|m| {
            let s = m.get(&id).unwrap();
            let mut ink = 0u32;
            for y in 4..36 {
                for x in 4..140 {
                    if s.back[y * SURF_W + x] != 0 {
                        ink += 1;
                    }
                }
            }
            (s.labels.len(), ink)
        });
        assert_eq!(n_labels, 1, "text op must be deferred as a label");
        assert!(ink > 20, "test build still bakes 1× ink for checksums, ink={ink}");
        // clear drops labels
        draw(owner, id, "clear 000000").unwrap();
        let n2 = SURFACES.with(|m| m.get(&id).unwrap().labels.len());
        assert_eq!(n2, 0);
        close(owner, id).unwrap();
    }

    #[test_case]
    fn rejects_malformed_ops() {
        reset();
        let owner = crate::sched::current_task_id();
        let id = request(owner, SurfaceKind::Canvas);
        assert!(matches!(draw(owner, id, "rect 1 2 3"), Err(DrawErr::BadOp(_)))); // wrong arity
        assert!(matches!(draw(owner, id, "explode 1 2"), Err(DrawErr::BadOp(_)))); // unknown op
        close(owner, id).unwrap();
    }

    #[test_case]
    fn click_to_square_maps_board_and_rejects_margin() {
        // Top-left pixel of a1's file / rank 8 visually is file a, rank 8.
        assert_eq!(click_to_square(BOARD_OX as u16, BOARD_OY as u16).as_deref(), Some("a8"));
        // Bottom-left of board → a1.
        let y = (BOARD_OY + 7 * BOARD_SQ + 1) as u16;
        assert_eq!(click_to_square(BOARD_OX as u16 + 1, y).as_deref(), Some("a1"));
        // Outside the board margin.
        assert!(click_to_square(0, 0).is_none());
        assert_eq!(parse_square("e4"), Some((4, 3)));
        assert!(parse_square("z9").is_none());
    }

    #[test_case]
    fn board_set_from_start_fen_is_deterministic() {
        reset();
        let owner = crate::sched::current_task_id();
        let id = request(owner, SurfaceKind::Board);
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let n = board_set(owner, id, fen).unwrap();
        assert!(n > 10, "start position should emit many ops, got {n}");
        let h1 = checksum(id).unwrap();
        // Second paint of the same FEN must match (presentation is pure).
        board_set(owner, id, fen).unwrap();
        assert_eq!(checksum(id).unwrap(), h1);
        // Empty board differs.
        board_set(owner, id, "8/8/8/8/8/8/8/8").unwrap();
        assert_ne!(checksum(id).unwrap(), h1);
        close(owner, id).unwrap();
    }
}
