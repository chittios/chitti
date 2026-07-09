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
}
