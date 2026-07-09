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

struct Surface {
    owner: TaskId,
    #[allow(dead_code)] // recorded for the compositor / future kind-specific layout
    kind: SurfaceKind,
    /// 0xRRGGBB pixels, row-major, `SURF_W * SURF_H`.
    back: Vec<u32>,
    /// Input events the compositor routed to this surface (mouse/keys), drained
    /// by `ui_event_poll`. Empty headless.
    events: VecDeque<UiEvent>,
}

/// An input event delivered to a surface's owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UiEvent {
    Click { x: u16, y: u16 },
    Key(u8),
}

static NEXT_SURFACE: AtomicU32 = AtomicU32::new(1);
static SURFACES: Locked<BTreeMap<u32, Surface>> = Locked::new(BTreeMap::new());

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
        m.insert(id, Surface { owner, kind, back: vec![0u32; SURF_W * SURF_H], events: VecDeque::new() });
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
            apply(&mut s.back, *op);
        }
        Ok(program.len())
    })?;
    #[cfg(not(test))]
    present(id);
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
}

/// Parse a `;`-separated draw program: `clear <hex>`, `rect x y w h <hex>`,
/// `line x0 y0 x1 y1 <hex>`, `pixel x y <hex>`. Colours are RRGGBB hex.
fn parse_ops(program: &str) -> Result<Vec<DrawOp>, DrawErr> {
    let mut out = Vec::new();
    for stmt in program.split(';') {
        let stmt = stmt.trim();
        if stmt.is_empty() {
            continue;
        }
        let toks: Vec<&str> = stmt.split_whitespace().collect();
        let op = toks[0];
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

fn apply(back: &mut [u32], op: DrawOp) {
    match op {
        DrawOp::Clear(c) => back.iter_mut().for_each(|p| *p = c),
        DrawOp::Rect { x, y, w, h, color } => {
            for dy in 0..h.max(0) {
                for dx in 0..w.max(0) {
                    put(back, x + dx, y + dy, color);
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
                put(back, x, y, color);
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
        DrawOp::Pixel { x, y, color } => put(back, x, y, color),
    }
}

/// Present a surface's backing buffer into the compositor's action pane. A
/// best-effort side effect; the backing buffer is the source of truth.
#[cfg(not(test))]
fn present(id: u32) {
    let snap = SURFACES.with(|m| m.get(&id).map(|s| s.back.clone()));
    if let Some(buf) = snap {
        crate::framebuffer::present_surface(id, SURF_W, SURF_H, &buf);
    }
}

// --- Board helpers (chess / grid UI agents) ---------------------------------
//
// Pure presentation: FEN / square marks → draw-op programs. Game rules live in
// the agent's SOUL (or optional later native validators), not here.

/// Square size for an 8×8 board letterboxed into the surface.
pub const BOARD_SQ: i32 = 24;
/// Origin of the board (left/top) so 8*24=192 fits in SURF_H with margin.
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
                let (glyph_color, filled) = piece_style(piece);
                paint_piece_ops(&mut ops, x, y, piece, glyph_color, filled);
            }
        }
    }
    ops
}

fn piece_style(p: char) -> (u32, bool) {
    // White pieces: light fill; black: dark fill.
    if p.is_ascii_uppercase() {
        (0xf0_f0_f0, true)
    } else {
        (0x22_22_22, true)
    }
}

/// Very small geometric stand-ins for pieces (no font dependency).
fn paint_piece_ops(ops: &mut alloc::string::String, x: i32, y: i32, piece: char, color: u32, _filled: bool) {
    use alloc::format;
    let cx = x + BOARD_SQ / 2;
    let cy = y + BOARD_SQ / 2;
    let c = color & 0xff_ffff;
    // Outline box so empty-ish shapes stay visible on both square colours.
    match piece.to_ascii_lowercase() {
        'p' => {
            // Pawn: small body + head.
            ops.push_str(&format!("rect {} {} 6 8 {c:06x}; ", cx - 3, cy - 1));
            ops.push_str(&format!("rect {} {} 4 4 {c:06x}; ", cx - 2, cy - 6));
        }
        'r' => {
            ops.push_str(&format!("rect {} {} 10 12 {c:06x}; ", cx - 5, cy - 5));
            ops.push_str(&format!("rect {} {} 12 3 {c:06x}; ", cx - 6, cy - 8));
        }
        'n' => {
            ops.push_str(&format!("rect {} {} 8 12 {c:06x}; ", cx - 3, cy - 5));
            ops.push_str(&format!("rect {} {} 6 4 {c:06x}; ", cx - 1, cy - 8));
        }
        'b' => {
            ops.push_str(&format!("rect {} {} 6 14 {c:06x}; ", cx - 3, cy - 7));
            ops.push_str(&format!("rect {} {} 10 4 {c:06x}; ", cx - 5, cy - 8));
        }
        'q' => {
            ops.push_str(&format!("rect {} {} 12 10 {c:06x}; ", cx - 6, cy - 3));
            ops.push_str(&format!("rect {} {} 4 4 {c:06x}; ", cx - 6, cy - 8));
            ops.push_str(&format!("rect {} {} 4 4 {c:06x}; ", cx + 2, cy - 8));
            ops.push_str(&format!("rect {} {} 4 4 {c:06x}; ", cx - 2, cy - 10));
        }
        'k' => {
            ops.push_str(&format!("rect {} {} 10 10 {c:06x}; ", cx - 5, cy - 3));
            ops.push_str(&format!("rect {} {} 4 8 {c:06x}; ", cx - 2, cy - 9));
            ops.push_str(&format!("rect {} {} 10 3 {c:06x}; ", cx - 5, cy - 6));
        }
        _ => {
            ops.push_str(&format!("rect {} {} 8 8 {c:06x}; ", cx - 4, cy - 4));
        }
    }
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
