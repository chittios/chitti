//! Rasterize a [`Layout`] into an RGB buffer using runtime TTF ([`crate::font_ttf`])
//! and optional decoded images ([`crate::image`]). Draws form controls, a
//! scrollbar, and an optional loading progress bar (Ladybird/Web content chrome).

use super::layout::{ControlKind, FormControl, Layout};
use crate::font_ttf;
use alloc::vec;
use alloc::vec::Vec;

/// Optional chrome overlaid after content paint.
#[derive(Clone, Debug, Default)]
pub struct Chrome {
    /// 0..=100 loading progress; `None` = hide bar.
    pub progress: Option<u8>,
    /// Draw progress at bottom instead of top.
    pub progress_bottom: bool,
    /// Draw vertical scrollbar when content overflows.
    pub scrollbar: bool,
    /// Content-space `(x, y, w, h)` of the link currently under the cursor —
    /// its runs get a hover underline (CSS `a:hover { text-decoration: underline }`).
    pub hover_link: Option<(i32, i32, i32, i32)>,
    /// Text selection highlight rects in **content** space (pre-scroll), painted
    /// under glyphs so drag-to-copy is visible.
    pub selection: alloc::vec::Vec<(i32, i32, i32, i32)>,
}

/// Paint `layout` into a `width * height` RGB buffer, scrolled by `scroll_y`.
pub fn paint(layout: &Layout, scroll_y: i32) -> Vec<u32> {
    paint_chrome(layout, scroll_y, Chrome::default())
}

/// Paint with scrollbar / progress chrome.
pub fn paint_chrome(layout: &Layout, scroll_y: i32, chrome: Chrome) -> Vec<u32> {
    let w = layout.width.max(1) as usize;
    let h = layout.height.max(1) as usize;
    let mut buf = vec![layout.bg; w * h];

    // Background rects under content (blurred shadow / rounded / square).
    for r in &layout.rects {
        if r.blur > 0 {
            paint_blur_shadow(&mut buf, w, h, r, scroll_y);
        } else if r.radius > 0 {
            fill_round_rect(&mut buf, w, h, r.x, r.y - scroll_y, r.w, r.h, r.color, r.radius);
        } else {
            fill_rect(&mut buf, w, h, r.x, r.y - scroll_y, r.w, r.h, r.color);
        }
    }

    // CSS background images (host-decoded), over colour rects, under content.
    for bb in &layout.bg_boxes {
        let y0 = bb.y - scroll_y;
        if y0 + bb.h < 0 || y0 >= layout.height {
            continue;
        }
        if let Some(ref px) = bb.pixels {
            paint_background_image(
                &mut buf,
                w,
                h,
                bb.x,
                y0,
                bb.w,
                bb.h,
                px,
                bb.src_w,
                bb.src_h,
                parse_bg_repeat(&bb.repeat),
                parse_bg_size(&bb.size),
                parse_bg_position(&bb.pos),
            );
        }
    }

    // Images (already decoded RGB).
    for im in &layout.images {
        let y0 = im.y - scroll_y;
        if y0 + im.h < 0 || y0 >= layout.height {
            continue;
        }
        if let Some(ref px) = im.pixels {
            blit_image_fit(&mut buf, w, h, im.x, y0, im.w, im.h, px, im.src_w, im.src_h, im.object_fit);
        } else {
            fill_rect(&mut buf, w, h, im.x, y0, im.w, im.h, 0xe0e0e0);
            for dx in 0..im.w {
                put(&mut buf, w, h, im.x + dx, y0, 0x888888);
                put(&mut buf, w, h, im.x + dx, y0 + im.h - 1, 0x888888);
            }
            for dy in 0..im.h {
                put(&mut buf, w, h, im.x, y0 + dy, 0x888888);
                put(&mut buf, w, h, im.x + im.w - 1, y0 + dy, 0x888888);
            }
            if !im.alt.is_empty() {
                let _ = blit_text(&mut buf, w, h, im.x + 4, y0 + 4, &im.alt, 12.0, 0x444444, "");
            }
        }
    }

    // Nested frames (iframe content or placeholder chrome).
    for fr in &layout.frames {
        let y0 = fr.y - scroll_y;
        if y0 + fr.h < 0 || y0 >= layout.height {
            continue;
        }
        if let Some(ref px) = fr.pixels {
            blit_image(
                &mut buf,
                w,
                h,
                fr.x,
                y0,
                fr.w,
                fr.h,
                px,
                fr.src_w.max(1),
                fr.src_h.max(1),
            );
            // Border
            for dx in 0..fr.w {
                put(&mut buf, w, h, fr.x + dx, y0, 0x888888);
                put(&mut buf, w, h, fr.x + dx, y0 + fr.h - 1, 0x888888);
            }
            for dy in 0..fr.h {
                put(&mut buf, w, h, fr.x, y0 + dy, 0x888888);
                put(&mut buf, w, h, fr.x + fr.w - 1, y0 + dy, 0x888888);
            }
        } else {
            fill_rect(&mut buf, w, h, fr.x, y0, fr.w, fr.h, 0xf0f0f0);
            for dx in 0..fr.w {
                put(&mut buf, w, h, fr.x + dx, y0, 0x888888);
                put(&mut buf, w, h, fr.x + dx, y0 + fr.h - 1, 0x888888);
            }
            for dy in 0..fr.h {
                put(&mut buf, w, h, fr.x, y0 + dy, 0x888888);
                put(&mut buf, w, h, fr.x + fr.w - 1, y0 + dy, 0x888888);
            }
            let label = match fr.kind {
                super::layout::EmbedKind::Video => "[video]",
                super::layout::EmbedKind::Audio => "[audio]",
                super::layout::EmbedKind::Canvas => "[canvas]",
                super::layout::EmbedKind::Iframe if !fr.srcdoc.is_empty() => "[iframe srcdoc]",
                super::layout::EmbedKind::Iframe if !fr.src.is_empty() => "[iframe]",
                _ => "[frame]",
            };
            let _ = blit_text(&mut buf, w, h, fr.x + 6, y0 + 6, label, 12.0, 0x555555, "");
            if !fr.src.is_empty() {
                let _ = blit_text(&mut buf, w, h, fr.x + 6, y0 + 22, &fr.src, 11.0, 0x1a73e8, "");
            }
        }
    }

    // Form controls.
    for c in &layout.controls {
        paint_control(&mut buf, w, h, c, scroll_y, layout.height);
    }

    // Text selection highlight (under glyphs, terracotta-tinted like chat).
    const SEL_BG: u32 = 0xc4_9a_84; // warm selection, readable on cream/white
    for &(sx, sy, sw, sh) in &chrome.selection {
        let y0 = sy - scroll_y;
        if y0 + sh < 0 || y0 >= layout.height || sw <= 0 || sh <= 0 {
            continue;
        }
        fill_rect(&mut buf, w, h, sx, y0, sw, sh, SEL_BG);
    }

    // Text runs (baseline-correct TTF).
    for run in &layout.runs {
        let px = run.font_size.max(8) as f32;
        let line_h = font_ttf::line_height(px) as i32;
        let y = run.y - scroll_y;
        if y + line_h < 0 || y >= layout.height {
            continue;
        }
        let end_x = if run.letter_spacing != 0 {
            // `letter-spacing`: advance each glyph by its width + the spacing.
            let mut pen = run.x;
            for ch in run.text.chars() {
                let mut b = [0u8; 4];
                let s = ch.encode_utf8(&mut b);
                let nx = blit_text(&mut buf, w, h, pen, y, s, px, run.color, &run.font_family);
                pen = nx + run.letter_spacing;
            }
            pen
        } else {
            blit_text(&mut buf, w, h, run.x, y, &run.text, px, run.color, &run.font_family)
        };
        if run.bold {
            let _ = blit_text(&mut buf, w, h, run.x + 1, y, &run.text, px, run.color, &run.font_family);
        }
        // Underline: an explicit `text-decoration:underline`, or a link that the
        // cursor is hovering (CSS `a:hover { text-decoration: underline }`).
        let hovered = run.link_href.is_some()
            && chrome.hover_link.is_some_and(|(hx, hy, hw, hh)| {
                run.x < hx + hw && end_x > hx && run.y >= hy - 2 && run.y < hy + hh + 2
            });
        if run.underline || hovered {
            let uy = y + line_h - 2;
            if uy >= 0 && uy < layout.height {
                for x in run.x..end_x.max(run.x + 1) {
                    put(&mut buf, w, h, x, uy, run.color);
                }
            }
        }
    }

    // `transform: rotate(...)` — rotate each element's rendered box in place.
    for op in &layout.rotates {
        rotate_region(&mut buf, w, h, op, layout.bg, scroll_y);
    }

    if chrome.scrollbar {
        paint_scrollbar(&mut buf, w, h, layout, scroll_y);
    }
    if let Some(pct) = chrome.progress {
        paint_progress(&mut buf, w, h, pct, chrome.progress_bottom);
    }
    buf
}

fn paint_control(
    buf: &mut [u32],
    bw: usize,
    bh: usize,
    c: &FormControl,
    scroll_y: i32,
    view_h: i32,
) {
    if c.kind == ControlKind::Hidden || c.w <= 0 || c.h <= 0 {
        return;
    }
    let y0 = c.y - scroll_y;
    if y0 + c.h < 0 || y0 >= view_h {
        return;
    }
    match c.kind {
        ControlKind::Text | ControlKind::Password | ControlKind::TextArea => {
            // White field with the control's own CSS background/radius (a text
            // input's UA default box is a sharp rectangle, not rounded).
            let bg = c.bg.unwrap_or(if c.focused { 0xffffff } else { 0xfafafa });
            let border = if c.focused { 0x1a73e8 } else { 0x888888 };
            fill_round_rect(buf, bw, bh, c.x, y0, c.w, c.h, bg, c.radius);
            // border
            for dx in 0..c.w {
                put(buf, bw, bh, c.x + dx, y0, border);
                put(buf, bw, bh, c.x + dx, y0 + c.h - 1, border);
            }
            for dy in 0..c.h {
                put(buf, bw, bh, c.x, y0 + dy, border);
                put(buf, bw, bh, c.x + c.w - 1, y0 + dy, border);
            }
            let text_owned: alloc::string::String = if c.value.is_empty() && !c.placeholder.is_empty()
            {
                c.placeholder.clone()
            } else if c.kind == ControlKind::Password {
                let n = c.value.chars().count().min(64);
                core::iter::repeat('*').take(n).collect()
            } else {
                c.value.clone()
            };
            let color = if c.value.is_empty() && !c.placeholder.is_empty() {
                0x999999u32
            } else {
                0x222222u32
            };
            let _ = blit_text(buf, bw, bh, c.x + 6, y0 + 4, &text_owned, 13.0, color, "");
            if c.focused {
                let tw = font_ttf::measure(&text_owned, 13.0) as i32;
                let cx = c.x + 6 + tw.min(c.w - 10);
                for dy in 4..(c.h - 4).max(5) {
                    put(buf, bw, bh, cx, y0 + dy, 0x1a73e8);
                }
            }
        }
        ControlKind::Submit | ControlKind::Button => {
            // Honor the control's own CSS background colour; else the **UA
            // default button** — a light system-grey box with a 1px border (the
            // genuine cross-browser default for `<button>`/submit; the previous
            // blue was wrong). An image/gradient-only background we can't render
            // (`.transparent`) also falls back to the UA button.
            let rad = c.radius; // native buttons aren't rounded unless CSS says so
            let styled = c.bg.filter(|_| !c.transparent);
            let bg = match styled {
                Some(col) if c.focused => darken(col),
                Some(col) => col,
                None if c.focused => 0xe0e0e6,
                None => 0xf0f0f2,
            };
            fill_round_rect(buf, bw, bh, c.x, y0, c.w, c.h, bg, rad);
            // Border only for the native (unstyled) button.
            if styled.is_none() {
                draw_rect_border(buf, bw, bh, c.x, y0, c.w, c.h, 0xbfbfc4);
            }
            let label = if c.value.is_empty() {
                if c.kind == ControlKind::Submit {
                    "Submit"
                } else {
                    "Button"
                }
            } else {
                c.value.as_str()
            };
            let tw = font_ttf::measure(label, 13.0) as i32;
            let tx = c.x + ((c.w - tw) / 2).max(4);
            // Label colour: the control's `color`; else dark on the light UA
            // button, or white on a dark author background.
            let fg = c.fg.unwrap_or(match styled {
                Some(col) if col_luma(col) < 140 => 0xffffff,
                _ => 0x202124,
            });
            let _ = blit_text(buf, bw, bh, tx, y0 + 5, label, 13.0, fg, "");
        }
        ControlKind::Checkbox => {
            fill_rect(buf, bw, bh, c.x, y0, c.w, c.h, 0xffffff);
            for dx in 0..c.w {
                put(buf, bw, bh, c.x + dx, y0, 0x444444);
                put(buf, bw, bh, c.x + dx, y0 + c.h - 1, 0x444444);
            }
            for dy in 0..c.h {
                put(buf, bw, bh, c.x, y0 + dy, 0x444444);
                put(buf, bw, bh, c.x + c.w - 1, y0 + dy, 0x444444);
            }
            if c.checked {
                fill_rect(buf, bw, bh, c.x + 4, y0 + 4, c.w - 8, c.h - 8, 0x1a73e8);
            }
        }
        ControlKind::Hidden => {}
    }
}

fn paint_scrollbar(buf: &mut [u32], w: usize, h: usize, layout: &Layout, scroll_y: i32) {
    let content_h = layout.content_height.max(1);
    let view_h = layout.height.max(1);
    if content_h <= view_h {
        return;
    }
    let track_x = (layout.width - 8).max(0);
    let track_w = 6;
    // Track
    fill_rect(buf, w, h, track_x, 0, track_w, view_h, 0xd0d0d0);
    let max_scroll = (content_h - view_h).max(1);
    let thumb_h = ((view_h as i64 * view_h as i64) / content_h as i64)
        .clamp(16, view_h as i64) as i32;
    let thumb_y = ((scroll_y as i64 * (view_h - thumb_h) as i64) / max_scroll as i64) as i32;
    fill_rect(
        buf,
        w,
        h,
        track_x + 1,
        thumb_y.clamp(0, view_h - thumb_h),
        track_w - 2,
        thumb_h,
        0x666666,
    );
}

/// Thin determinate progress bar (Ladybird load progress affordance).
fn paint_progress(buf: &mut [u32], w: usize, h: usize, pct: u8, bottom: bool) {
    let pct = pct.min(100) as i32;
    let bar_h = 3i32;
    let y = if bottom {
        (h as i32 - bar_h).max(0)
    } else {
        0
    };
    fill_rect(buf, w, h, 0, y, w as i32, bar_h, 0xe8e0d8);
    let fill_w = (w as i32 * pct) / 100;
    if fill_w > 0 {
        fill_rect(buf, w, h, 0, y, fill_w, bar_h, 0xcc785c); // brand terracotta
    }
}

fn blit_image(
    buf: &mut [u32],
    bw: usize,
    bh: usize,
    dx: i32,
    dy: i32,
    dw: i32,
    dh: i32,
    src: &[u32],
    sw: usize,
    sh: usize,
) {
    if sw == 0 || sh == 0 || dw <= 0 || dh <= 0 {
        return;
    }
    for row in 0..dh {
        let sy = (row as usize * sh) / dh as usize;
        for col in 0..dw {
            let sx = (col as usize * sw) / dw as usize;
            let p = src[sy * sw + sx];
            put(buf, bw, bh, dx + col, dy + row, p);
        }
    }
}

/// Perceptual brightness (0–255) of an `0x00RRGGBB` colour.
fn col_luma(c: u32) -> u32 {
    let (r, g, b) = ((c >> 16) & 0xff, (c >> 8) & 0xff, c & 0xff);
    (r * 30 + g * 59 + b * 11) / 100
}

/// Draw a 1px square border around a box.
fn draw_rect_border(buf: &mut [u32], w: usize, h: usize, x: i32, y: i32, rw: i32, rh: i32, color: u32) {
    for dx in 0..rw {
        put(buf, w, h, x + dx, y, color);
        put(buf, w, h, x + dx, y + rh - 1, color);
    }
    for dy in 0..rh {
        put(buf, w, h, x, y + dy, color);
        put(buf, w, h, x + rw - 1, y + dy, color);
    }
}

/// Darken a colour ~12% (for a control's focused/pressed state).
fn darken(c: u32) -> u32 {
    let ch = |sh: u32| (((c >> sh) & 0xff) * 88 / 100) & 0xff;
    (ch(16) << 16) | (ch(8) << 8) | ch(0)
}

/// Is local point `(lx, ly)` inside the rounded rect `[0,rw)×[0,rh)` with corner
/// `radius`? Used to build a box-shadow coverage mask.
fn inside_round_rect(lx: i32, ly: i32, rw: i32, rh: i32, radius: i32) -> bool {
    if lx < 0 || ly < 0 || lx >= rw || ly >= rh {
        return false;
    }
    let r = radius.min(rw / 2).min(rh / 2).max(0);
    if r == 0 {
        return true;
    }
    let cx = if lx < r {
        r
    } else if lx >= rw - r {
        rw - r
    } else {
        return true;
    };
    let cy = if ly < r {
        r
    } else if ly >= rh - r {
        rh - r
    } else {
        return true;
    };
    let (dx, dy) = ((lx - cx) as i64, (ly - cy) as i64);
    dx * dx + dy * dy <= (r * r) as i64
}

/// Separable box blur of an 8-bit coverage mask (one horizontal + one vertical
/// running-sum pass). Three passes approximate a Gaussian.
fn box_blur(mask: &mut [u8], w: i32, h: i32, rad: i32) {
    if rad < 1 || w <= 0 || h <= 0 {
        return;
    }
    let win = (2 * rad + 1) as i32;
    let mut tmp = alloc::vec![0u8; mask.len()];
    // Horizontal.
    for y in 0..h {
        let row = (y * w) as usize;
        let mut sum: i32 = 0;
        for x in -rad..=rad {
            if x >= 0 && x < w {
                sum += mask[row + x as usize] as i32;
            }
        }
        for x in 0..w {
            tmp[row + x as usize] = (sum / win) as u8;
            let add = x + rad + 1;
            let sub = x - rad;
            if add < w {
                sum += mask[row + add as usize] as i32;
            }
            if sub >= 0 {
                sum -= mask[row + sub as usize] as i32;
            }
        }
    }
    // Vertical.
    for x in 0..w {
        let mut sum: i32 = 0;
        for y in -rad..=rad {
            if y >= 0 && y < h {
                sum += tmp[(y * w + x) as usize] as i32;
            }
        }
        for y in 0..h {
            mask[(y * w + x) as usize] = (sum / win) as u8;
            let add = y + rad + 1;
            let sub = y - rad;
            if add < h {
                sum += tmp[(add * w + x) as usize] as i32;
            }
            if sub >= 0 {
                sum -= tmp[(sub * w + x) as usize] as i32;
            }
        }
    }
}

/// Alpha-blend `color` over the pixel at `(x,y)` by `alpha` (0–255).
fn blend_over(buf: &mut [u32], w: usize, h: usize, x: i32, y: i32, color: u32, alpha: u8) {
    if x < 0 || y < 0 || x as usize >= w || y as usize >= h || alpha == 0 {
        return;
    }
    let i = y as usize * w + x as usize;
    let dst = buf[i];
    let a = alpha as u32;
    let ia = 255 - a;
    let ch = |sh: u32| {
        (((color >> sh) & 0xff) * a + ((dst >> sh) & 0xff) * ia) / 255 & 0xff
    };
    buf[i] = (ch(16) << 16) | (ch(8) << 8) | ch(0);
}

/// Render a `box-shadow` rect as a real box-blurred shadow: build a coverage
/// mask for the (rounded) box expanded by the blur radius, blur it three times,
/// and composite the shadow colour with the resulting alpha.
fn paint_blur_shadow(buf: &mut [u32], w: usize, h: usize, r: &super::layout::RectBox, scroll_y: i32) {
    let blur = r.blur.clamp(1, 40);
    let pad = blur;
    let (bw, bh) = (r.w + 2 * pad, r.h + 2 * pad);
    if bw <= 0 || bh <= 0 || bw > 4096 || bh > 4096 {
        return;
    }
    let (bx, by) = (r.x - pad, r.y - scroll_y - pad);
    let mut mask = alloc::vec![0u8; (bw * bh) as usize];
    for yy in 0..bh {
        for xx in 0..bw {
            if inside_round_rect(xx - pad, yy - pad, r.w, r.h, r.radius) {
                mask[(yy * bw + xx) as usize] = 255;
            }
        }
    }
    let rad = (blur / 2).max(1);
    for _ in 0..3 {
        box_blur(&mut mask, bw, bh, rad);
    }
    for yy in 0..bh {
        for xx in 0..bw {
            let a = mask[(yy * bw + xx) as usize];
            if a > 0 {
                // Shadows are semi-opaque; scale the coverage down.
                blend_over(buf, w, h, bx + xx, by + yy, r.color, (a as u32 * 160 / 255) as u8);
            }
        }
    }
}

/// `sin(x)` (radians) — Taylor series with range reduction to `[-π, π]`; the
/// kernel is no_std without libm trig, and rotation only needs ~1e-3 accuracy.
fn sin_approx(x: f32) -> f32 {
    use core::f32::consts::PI;
    let tau = 2.0 * PI;
    let n = (x / tau) as i32;
    let mut r = x - n as f32 * tau;
    if r > PI {
        r -= tau;
    } else if r < -PI {
        r += tau;
    }
    let x2 = r * r;
    // x - x^3/6 + x^5/120 - x^7/5040 (Horner).
    r * (1.0 - x2 / 6.0 * (1.0 - x2 / 20.0 * (1.0 - x2 / 42.0)))
}
fn cos_approx(x: f32) -> f32 {
    sin_approx(x + core::f32::consts::PI / 2.0)
}

/// Paint-time bitmap rotation for `transform: rotate(...)`: snapshot the box
/// region (rendered axis-aligned), clear it, and re-draw it rotated about the
/// element centre by inverse-mapping each destination pixel (nearest-neighbour).
/// Pixels equal to the page background are treated as transparent.
fn rotate_region(buf: &mut [u32], w: usize, h: usize, op: &super::layout::RotateOp, bg: u32, scroll_y: i32) {
    let (x0, y0, bwi, bhi) = (op.x, op.y - scroll_y, op.w.max(1), op.h.max(1));
    if bwi > 4096 || bhi > 4096 {
        return;
    }
    // Snapshot the source box.
    let mut src = alloc::vec![bg; (bwi * bhi) as usize];
    for yy in 0..bhi {
        for xx in 0..bwi {
            let (px, py) = (x0 + xx, y0 + yy);
            if px >= 0 && (px as usize) < w && py >= 0 && (py as usize) < h {
                src[(yy * bwi + xx) as usize] = buf[py as usize * w + px as usize];
            }
        }
    }
    // Clear the original box (so the un-rotated copy doesn't remain).
    for yy in 0..bhi {
        for xx in 0..bwi {
            put(buf, w, h, x0 + xx, y0 + yy, bg);
        }
    }
    let ang = op.angle_deg * core::f32::consts::PI / 180.0;
    let (s, c) = (sin_approx(ang), cos_approx(ang));
    let (cxf, cyf) = (op.cx as f32, (op.cy - scroll_y) as f32);
    // Destination bounding radius covers the rotated box. `(w+h)/2` is a safe
    // over-estimate of the half-diagonal (√(w²+h²) ≤ w+h).
    let diag = (bwi + bhi) / 2 + 2;
    for dy in (op.cy - scroll_y - diag)..(op.cy - scroll_y + diag) {
        for dx in (op.cx - diag)..(op.cx + diag) {
            let vx = dx as f32 - cxf;
            let vy = dy as f32 - cyf;
            // source = centre + R(-angle)·v
            let sx = cxf + c * vx + s * vy;
            let sy = cyf - s * vx + c * vy;
            let lx = sx as i32 - x0;
            let ly = sy as i32 - y0;
            if lx >= 0 && lx < bwi && ly >= 0 && ly < bhi {
                let p = src[(ly * bwi + lx) as usize];
                if p != bg {
                    put(buf, w, h, dx, dy, p);
                }
            }
        }
    }
}

/// Blit a cropped source region `[sx0,sx0+scw)×[sy0,sy0+sch)` into the dest box.
#[allow(clippy::too_many_arguments)]
fn blit_image_crop(
    buf: &mut [u32], bw: usize, bh: usize, dx: i32, dy: i32, dw: i32, dh: i32,
    src: &[u32], sw: usize, sh: usize, sx0: usize, sy0: usize, scw: usize, sch: usize,
) {
    if scw == 0 || sch == 0 || dw <= 0 || dh <= 0 {
        return;
    }
    for row in 0..dh {
        let sy = (sy0 + (row as usize * sch) / dh as usize).min(sh - 1);
        for col in 0..dw {
            let sx = (sx0 + (col as usize * scw) / dw as usize).min(sw - 1);
            put(buf, bw, bh, dx + col, dy + row, src[sy * sw + sx]);
        }
    }
}

/// Blit an image into a `box_w`×`box_h` box honoring `object-fit`
/// (fill/contain/cover/none/scale-down) — integer math, centered.
#[allow(clippy::too_many_arguments)]
fn blit_image_fit(
    buf: &mut [u32], bw: usize, bh: usize, bx: i32, by: i32, box_w: i32, box_h: i32,
    src: &[u32], sw: usize, sh: usize, fit: super::css::ObjectFit,
) {
    use super::css::ObjectFit::*;
    if sw == 0 || sh == 0 || box_w <= 0 || box_h <= 0 {
        return;
    }
    let (siw, sih) = (sw as i64, sh as i64);
    let (bwi, bhi) = (box_w as i64, box_h as i64);
    match fit {
        Fill => blit_image(buf, bw, bh, bx, by, box_w, box_h, src, sw, sh),
        Contain | ScaleDown => {
            // Fit inside, preserving aspect; scale-down never upscales.
            let (mut dw, mut dh) = if siw * bhi <= sih * bwi {
                ((siw * bhi / sih) as i32, box_h)
            } else {
                (box_w, (sih * bwi / siw) as i32)
            };
            if matches!(fit, ScaleDown) && (dw > sw as i32 || dh > sh as i32) {
                dw = sw as i32;
                dh = sh as i32;
            }
            let ox = bx + (box_w - dw) / 2;
            let oy = by + (box_h - dh) / 2;
            blit_image(buf, bw, bh, ox, oy, dw.max(1), dh.max(1), src, sw, sh);
        }
        Cover => {
            // Fill the box, cropping the overflow (centered).
            let (cw, ch, sx0, sy0) = if siw * bhi > sih * bwi {
                let cw = (sih * bwi / bhi) as usize;
                (cw.min(sw), sh, (sw - cw.min(sw)) / 2, 0)
            } else {
                let ch = (siw * bhi / bwi) as usize;
                (sw, ch.min(sh), 0, (sh - ch.min(sh)) / 2)
            };
            blit_image_crop(buf, bw, bh, bx, by, box_w, box_h, src, sw, sh, sx0, sy0, cw, ch);
        }
        None => {
            // Natural size, centered, cropped to the box.
            let dw = (sw as i32).min(box_w);
            let dh = (sh as i32).min(box_h);
            let sx0 = (sw.saturating_sub(dw as usize)) / 2;
            let sy0 = (sh.saturating_sub(dh as usize)) / 2;
            let ox = bx + (box_w - dw) / 2;
            let oy = by + (box_h - dh) / 2;
            blit_image_crop(buf, bw, bh, ox, oy, dw, dh, src, sw, sh, sx0, sy0, dw as usize, dh as usize);
        }
    }
}

fn fill_rect(buf: &mut [u32], w: usize, h: usize, x: i32, y: i32, rw: i32, rh: i32, color: u32) {
    for dy in 0..rh {
        for dx in 0..rw {
            put(buf, w, h, x + dx, y + dy, color);
        }
    }
}

/// Fill a rectangle with rounded corners (`border-radius`). A pixel inside one
/// of the four corner quadrants is drawn only if it falls within the corner
/// circle of radius `radius` (clamped to half the shorter side).
fn fill_round_rect(
    buf: &mut [u32],
    w: usize,
    h: usize,
    x: i32,
    y: i32,
    rw: i32,
    rh: i32,
    color: u32,
    radius: i32,
) {
    let r = radius.min(rw / 2).min(rh / 2).max(0);
    if r == 0 {
        fill_rect(buf, w, h, x, y, rw, rh, color);
        return;
    }
    let r2 = (r * r) as i64;
    for dy in 0..rh {
        for dx in 0..rw {
            // Corner-circle centre for whichever quadrant this pixel is in.
            let cx = if dx < r {
                r
            } else if dx >= rw - r {
                rw - r
            } else {
                dx // straight edge — always inside
            };
            let cy = if dy < r {
                r
            } else if dy >= rh - r {
                rh - r
            } else {
                dy
            };
            if cx != dx && cy != dy {
                let (ddx, ddy) = ((dx - cx) as i64, (dy - cy) as i64);
                if ddx * ddx + ddy * ddy > r2 {
                    continue;
                }
            }
            put(buf, w, h, x + dx, y + dy, color);
        }
    }
}

/// Family-aware text blit. `family` is the CSS font-family (first usable name,
/// lowercase; `""` = default — an unregistered/generic family falls back to
/// the global face per glyph). Returns the pen x after the run, like
/// [`font_ttf::blit_run`].
#[allow(clippy::too_many_arguments)]
fn blit_text(
    buf: &mut [u32],
    w: usize,
    h: usize,
    x: i32,
    y: i32,
    text: &str,
    px: f32,
    color: u32,
    family: &str,
) -> i32 {
    font_ttf::blit_run_family(buf, w, h, x, y, text, px, color, family)
}

/// CSS `background-repeat` tiling mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BgRepeat {
    /// Tile in both axes (CSS default).
    Repeat,
    /// Tile horizontally only.
    RepeatX,
    /// Tile vertically only.
    RepeatY,
    /// Single tile.
    NoRepeat,
}

/// Parse a CSS `background-repeat` value; anything unrecognized is `Repeat`.
pub fn parse_bg_repeat(s: &str) -> BgRepeat {
    let s = s.trim();
    if s.eq_ignore_ascii_case("repeat-x") {
        BgRepeat::RepeatX
    } else if s.eq_ignore_ascii_case("repeat-y") {
        BgRepeat::RepeatY
    } else if s.eq_ignore_ascii_case("no-repeat") {
        BgRepeat::NoRepeat
    } else {
        BgRepeat::Repeat
    }
}

/// CSS `background-size`.
///
/// Auto-dimension convention: in `Px`/`Percent`, a component of `-1` means
/// "auto" — that axis is derived from the other one preserving the source
/// aspect ratio (so `"100px"` parses as `Px(100, -1)`: width 100, height
/// keep-aspect). `Px(-1, h)` / `Percent(-1, p)` analogously mean auto width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BgSize {
    /// Intrinsic source dimensions (CSS default).
    Auto,
    /// Aspect-preserving scale to fully cover the rect (may crop).
    Cover,
    /// Aspect-preserving scale to fit inside the rect (may letterbox).
    Contain,
    /// Explicit pixel size; `-1` in a component = auto (keep aspect).
    Px(i32, i32),
    /// Percent of the painted rect per axis; `-1` = auto (keep aspect).
    Percent(i32, i32),
}

/// One parsed size token: `auto`, `N%`, or `Npx`/bare `N`.
enum SizeTok {
    Auto,
    Pct(i32),
    Px(i32),
}

fn parse_size_tok(t: &str) -> Option<SizeTok> {
    if t.eq_ignore_ascii_case("auto") {
        return Some(SizeTok::Auto);
    }
    if let Some(n) = t.strip_suffix('%') {
        return n.trim().parse::<i32>().ok().map(SizeTok::Pct);
    }
    let n = t.strip_suffix("px").unwrap_or(t);
    n.trim().parse::<i32>().ok().map(SizeTok::Px)
}

/// Parse a CSS `background-size` value: `cover`, `contain`, one or two
/// lengths/percentages (`"100px 80px"`, `"50% 50%"`, `"100px"` →
/// `Px(100, -1)` = auto height, see [`BgSize`]). Mixed px/% pairs resolve to
/// the first token's unit with an auto second axis. Unparsable input = `Auto`.
pub fn parse_bg_size(s: &str) -> BgSize {
    let s = s.trim();
    if s.eq_ignore_ascii_case("cover") {
        return BgSize::Cover;
    }
    if s.eq_ignore_ascii_case("contain") {
        return BgSize::Contain;
    }
    let mut it = s.split_whitespace();
    let a = match it.next().and_then(parse_size_tok) {
        Some(t) => t,
        None => return BgSize::Auto,
    };
    let b = it.next().and_then(parse_size_tok);
    match (a, b) {
        (SizeTok::Auto, None) | (SizeTok::Auto, Some(SizeTok::Auto)) => BgSize::Auto,
        (SizeTok::Px(w), None) | (SizeTok::Px(w), Some(SizeTok::Auto)) => BgSize::Px(w, -1),
        (SizeTok::Pct(w), None) | (SizeTok::Pct(w), Some(SizeTok::Auto)) => BgSize::Percent(w, -1),
        (SizeTok::Px(w), Some(SizeTok::Px(h))) => BgSize::Px(w, h),
        (SizeTok::Pct(w), Some(SizeTok::Pct(h))) => BgSize::Percent(w, h),
        (SizeTok::Auto, Some(SizeTok::Px(h))) => BgSize::Px(-1, h),
        (SizeTok::Auto, Some(SizeTok::Pct(h))) => BgSize::Percent(-1, h),
        // Mixed units: keep the first token, auto the other axis.
        (SizeTok::Px(w), Some(_)) => BgSize::Px(w, -1),
        (SizeTok::Pct(w), Some(_)) => BgSize::Percent(w, -1),
    }
}

/// One axis of a CSS `background-position` value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BgPosVal {
    /// Percent placement: offset = `(rect - tile) * p / 100` (the CSS rule —
    /// `50` centers, `100` right/bottom-aligns).
    Percent(i32),
    /// Absolute pixel offset from the rect's top-left.
    Px(i32),
}

/// Classify a position token onto an axis: 0 = x-only keyword, 1 = y-only
/// keyword, 2 = either (center / length / percent).
fn parse_pos_tok(t: &str) -> Option<(BgPosVal, u8)> {
    if t.eq_ignore_ascii_case("left") {
        return Some((BgPosVal::Percent(0), 0));
    }
    if t.eq_ignore_ascii_case("right") {
        return Some((BgPosVal::Percent(100), 0));
    }
    if t.eq_ignore_ascii_case("top") {
        return Some((BgPosVal::Percent(0), 1));
    }
    if t.eq_ignore_ascii_case("bottom") {
        return Some((BgPosVal::Percent(100), 1));
    }
    if t.eq_ignore_ascii_case("center") {
        return Some((BgPosVal::Percent(50), 2));
    }
    if let Some(n) = t.strip_suffix('%') {
        return n.trim().parse::<i32>().ok().map(|p| (BgPosVal::Percent(p), 2));
    }
    let n = t.strip_suffix("px").unwrap_or(t);
    n.trim().parse::<i32>().ok().map(|p| (BgPosVal::Px(p), 2))
}

/// Parse a CSS `background-position` value into `(x, y)`.
///
/// Keywords map to percentages (`left`/`top` = 0, `center` = 50,
/// `right`/`bottom` = 100); `N%` passes through; `Npx`/bare `N` is an absolute
/// pixel offset. A single value sets one axis and centers the other (per CSS:
/// `"top"` = `(Percent(50), Percent(0))`). Keyword pairs may come in either
/// order (`"bottom right"`). Default / unparsable = `(Percent(0), Percent(0))`.
pub fn parse_bg_position(s: &str) -> (BgPosVal, BgPosVal) {
    let default = (BgPosVal::Percent(0), BgPosVal::Percent(0));
    let mut it = s.split_whitespace();
    let a = match it.next().and_then(parse_pos_tok) {
        Some(t) => t,
        None => return default,
    };
    match it.next().and_then(parse_pos_tok) {
        None => match a.1 {
            1 => (BgPosVal::Percent(50), a.0), // y-axis keyword alone
            _ => (a.0, BgPosVal::Percent(50)), // x or axis-free value alone
        },
        Some(b) => {
            // Swap if the keywords name the axes in y-x order ("top left").
            if a.1 == 1 || b.1 == 0 {
                (b.0, a.0)
            } else {
                (a.0, b.0)
            }
        }
    }
}

/// `w * n / d`, saturated to `>= 1` — aspect-ratio helper for tile sizing.
fn scale_dim(w: i32, n: usize, d: usize) -> i32 {
    if d == 0 {
        return 1;
    }
    ((w as i64 * n as i64) / d as i64).max(1) as i32
}

/// Resolve the scaled tile size for a background image (pure; see
/// [`paint_background_image`]). `-1` components follow the [`BgSize`] auto
/// convention. Returns `None` when a dimension resolves to zero or negative.
fn bg_tile_size(
    rect_w: i32,
    rect_h: i32,
    src_w: usize,
    src_h: usize,
    size: BgSize,
) -> Option<(i32, i32)> {
    let (tw, th) = match size {
        BgSize::Auto => (src_w as i32, src_h as i32),
        BgSize::Cover => {
            // Ceil the derived axis so the tile always fully covers the rect.
            let th_w =
                ((rect_w as i64 * src_h as i64 + src_w as i64 - 1) / src_w as i64) as i32;
            if th_w >= rect_h {
                (rect_w, th_w)
            } else {
                let tw_h =
                    ((rect_h as i64 * src_w as i64 + src_h as i64 - 1) / src_h as i64) as i32;
                (tw_h.max(rect_w), rect_h)
            }
        }
        BgSize::Contain => {
            let th_w = ((rect_w as i64 * src_h as i64) / src_w as i64) as i32;
            if th_w <= rect_h {
                (rect_w, th_w.max(1))
            } else {
                (scale_dim(rect_h, src_w, src_h), rect_h)
            }
        }
        BgSize::Px(w, h) => match (w, h) {
            (-1, -1) => (src_w as i32, src_h as i32),
            (w, -1) => (w, scale_dim(w, src_h, src_w)),
            (-1, h) => (scale_dim(h, src_w, src_h), h),
            (w, h) => (w, h),
        },
        BgSize::Percent(pw, ph) => match (pw, ph) {
            (-1, -1) => (src_w as i32, src_h as i32),
            (pw, -1) => {
                let w = (rect_w as i64 * pw as i64 / 100).max(0) as i32;
                (w, scale_dim(w, src_h, src_w))
            }
            (-1, ph) => {
                let h = (rect_h as i64 * ph as i64 / 100).max(0) as i32;
                (scale_dim(h, src_w, src_h), h)
            }
            (pw, ph) => (
                (rect_w as i64 * pw as i64 / 100) as i32,
                (rect_h as i64 * ph as i64 / 100) as i32,
            ),
        },
    };
    if tw <= 0 || th <= 0 {
        None
    } else {
        Some((tw, th))
    }
}

/// Anchor offset for one axis: CSS percent positioning
/// (`offset = (rect - tile) * p / 100`) or an absolute pixel offset.
fn bg_axis_offset(rect_dim: i32, tile_dim: i32, pos: BgPosVal) -> i32 {
    match pos {
        BgPosVal::Percent(p) => ((rect_dim - tile_dim) as i64 * p as i64 / 100) as i32,
        BgPosVal::Px(n) => n,
    }
}

/// Blit one nearest-neighbor-scaled tile, clipped to `[cx0,cx1) x [cy0,cy1)`
/// and to the buffer. `(tx, ty, tw, th)` is the tile's destination rect.
fn blit_tile_clipped(
    buf: &mut [u32],
    bw: usize,
    bh: usize,
    tx: i32,
    ty: i32,
    tw: i32,
    th: i32,
    cx0: i32,
    cy0: i32,
    cx1: i32,
    cy1: i32,
    src: &[u32],
    sw: usize,
    sh: usize,
) {
    let x0 = tx.max(cx0).max(0);
    let y0 = ty.max(cy0).max(0);
    let x1 = (tx + tw).min(cx1).min(bw as i32);
    let y1 = (ty + th).min(cy1).min(bh as i32);
    for y in y0..y1 {
        let sy = (((y - ty) as i64 * sh as i64) / th as i64) as usize;
        let srow = sy.min(sh - 1) * sw;
        let drow = y as usize * bw;
        for x in x0..x1 {
            let sx = ((((x - tx) as i64) * sw as i64) / tw as i64) as usize;
            buf[drow + x as usize] = src[srow + sx.min(sw - 1)] & 0x00ff_ffff;
        }
    }
}

/// Paint a CSS background image into `rect` of a `bw * bh` buffer.
///
/// The tile size comes from `size` (see [`BgSize`]; `Auto` = source dims,
/// `Cover`/`Contain` aspect-preserve against the rect, `-1` components keep
/// aspect), the anchor from `pos` ([`BgPosVal::Percent`] follows the CSS rule
/// `offset = (rect - tile) * p / 100`; [`BgPosVal::Px`] is absolute), and the
/// tile is repeated per `repeat` — always clipped to the rect and the buffer.
/// Sampling is nearest-neighbor, like `blit_image`. Handles tiles smaller and
/// larger than the rect, negative offsets, and zero-size inputs (no-op).
pub fn paint_background_image(
    buf: &mut [u32],
    bw: usize,
    bh: usize,
    rect_x: i32,
    rect_y: i32,
    rect_w: i32,
    rect_h: i32,
    px: &[u32],
    src_w: usize,
    src_h: usize,
    repeat: BgRepeat,
    size: BgSize,
    pos: (BgPosVal, BgPosVal),
) {
    if rect_w <= 0 || rect_h <= 0 || src_w == 0 || src_h == 0 || px.len() < src_w * src_h {
        return;
    }
    let (tw, th) = match bg_tile_size(rect_w, rect_h, src_w, src_h, size) {
        Some(t) => t,
        None => return,
    };
    let ox = rect_x + bg_axis_offset(rect_w, tw, pos.0);
    let oy = rect_y + bg_axis_offset(rect_h, th, pos.1);
    // Clip window: the rect intersected with the buffer.
    let (cx0, cy0) = (rect_x, rect_y);
    let (cx1, cy1) = (rect_x + rect_w, rect_y + rect_h);
    // First tile origin at or before the rect's left/top edge for each
    // repeating axis (rem_euclid keeps this correct for negative offsets).
    let start_x = |anchor: i32| {
        let r = (anchor - rect_x).rem_euclid(tw);
        rect_x + r - if r > 0 { tw } else { 0 }
    };
    let start_y = |anchor: i32| {
        let r = (anchor - rect_y).rem_euclid(th);
        rect_y + r - if r > 0 { th } else { 0 }
    };
    let (x_from, x_rep) = match repeat {
        BgRepeat::Repeat | BgRepeat::RepeatX => (start_x(ox), true),
        _ => (ox, false),
    };
    let (y_from, y_rep) = match repeat {
        BgRepeat::Repeat | BgRepeat::RepeatY => (start_y(oy), true),
        _ => (oy, false),
    };
    let mut ty = y_from;
    loop {
        let mut tx = x_from;
        loop {
            blit_tile_clipped(buf, bw, bh, tx, ty, tw, th, cx0, cy0, cx1, cy1, px, src_w, src_h);
            tx += tw;
            if !x_rep || tx >= cx1 {
                break;
            }
        }
        ty += th;
        if !y_rep || ty >= cy1 {
            break;
        }
    }
}

/// FNV-1a checksum of the buffer.
pub fn checksum(buf: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &px in buf {
        for b in px.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

fn put(buf: &mut [u32], w: usize, h: usize, x: i32, y: i32, color: u32) {
    if x < 0 || y < 0 {
        return;
    }
    let (x, y) = (x as usize, y as usize);
    if x < w && y < h {
        buf[y * w + x] = color & 0x00ff_ffff;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::browser::{html, layout};

    #[test_case]
    fn paint_nonzero_checksum() {
        let doc = html::parse("<html><body><p>Hi</p></body></html>");
        let lay = layout::layout_document_plain(&doc.root, 160, 80);
        let buf = paint(&lay, 0);
        assert_eq!(buf.len(), 160 * 80);
        assert_ne!(checksum(&buf), 0);
        let bg = lay.bg;
        assert!(buf.iter().any(|&p| p != bg), "expected ink pixels");
    }

    #[test_case]
    fn paint_progress_and_form() {
        let doc = html::parse(
            r#"<html><body><form><input name="q" value="ab"><input type="submit"></form></body></html>"#,
        );
        let mut lay = layout::layout_document_plain(&doc.root, 320, 200);
        lay.content_height = 800; // force scrollbar
        let chrome = Chrome {
            progress: Some(40),
            progress_bottom: false,
            scrollbar: true,
            hover_link: None,
            selection: alloc::vec::Vec::new(),
        };
        let buf = paint_chrome(&lay, 0, chrome);
        assert_eq!(buf.len(), 320 * 200);
        // Top progress bar uses brand terracotta on some pixels.
        assert!(
            buf.iter().take(320 * 3).any(|&p| p == 0xcc785c),
            "expected progress pixels"
        );
    }

    // ---- background-image primitives ----

    const RED: u32 = 0xff0000;
    const BLUE: u32 = 0x0000ff;

    /// 2x2 checker: RED BLUE / BLUE RED.
    fn checker() -> Vec<u32> {
        vec![RED, BLUE, BLUE, RED]
    }

    fn px_at(buf: &[u32], w: usize, x: usize, y: usize) -> u32 {
        buf[y * w + x]
    }

    #[test_case]
    fn bg_parse_repeat_forms() {
        assert_eq!(parse_bg_repeat("repeat"), BgRepeat::Repeat);
        assert_eq!(parse_bg_repeat("repeat-x"), BgRepeat::RepeatX);
        assert_eq!(parse_bg_repeat(" repeat-y "), BgRepeat::RepeatY);
        assert_eq!(parse_bg_repeat("no-repeat"), BgRepeat::NoRepeat);
        assert_eq!(parse_bg_repeat("NO-REPEAT"), BgRepeat::NoRepeat);
        assert_eq!(parse_bg_repeat(""), BgRepeat::Repeat); // default
        assert_eq!(parse_bg_repeat("bogus"), BgRepeat::Repeat);
    }

    #[test_case]
    fn bg_parse_size_forms() {
        assert_eq!(parse_bg_size("cover"), BgSize::Cover);
        assert_eq!(parse_bg_size(" CONTAIN "), BgSize::Contain);
        assert_eq!(parse_bg_size("auto"), BgSize::Auto);
        assert_eq!(parse_bg_size(""), BgSize::Auto); // default
        assert_eq!(parse_bg_size("garbage"), BgSize::Auto);
        assert_eq!(parse_bg_size("50% 50%"), BgSize::Percent(50, 50));
        assert_eq!(parse_bg_size("100px 80px"), BgSize::Px(100, 80));
        assert_eq!(parse_bg_size("100 80"), BgSize::Px(100, 80));
        // Single value = width with auto (keep-aspect) height, encoded as -1.
        assert_eq!(parse_bg_size("100px"), BgSize::Px(100, -1));
        assert_eq!(parse_bg_size("50%"), BgSize::Percent(50, -1));
        assert_eq!(parse_bg_size("100px auto"), BgSize::Px(100, -1));
        assert_eq!(parse_bg_size("auto 80px"), BgSize::Px(-1, 80));
    }

    #[test_case]
    fn bg_parse_position_forms() {
        let p = BgPosVal::Percent;
        assert_eq!(parse_bg_position(""), (p(0), p(0))); // default
        assert_eq!(parse_bg_position("left top"), (p(0), p(0)));
        assert_eq!(parse_bg_position("right bottom"), (p(100), p(100)));
        assert_eq!(parse_bg_position("bottom right"), (p(100), p(100))); // y-x order swaps
        assert_eq!(parse_bg_position("top left"), (p(0), p(0)));
        assert_eq!(parse_bg_position("center"), (p(50), p(50)));
        // Single value centers the other axis, on the right axis per CSS.
        assert_eq!(parse_bg_position("left"), (p(0), p(50)));
        assert_eq!(parse_bg_position("top"), (p(50), p(0)));
        assert_eq!(parse_bg_position("bottom"), (p(50), p(100)));
        assert_eq!(parse_bg_position("25%"), (p(25), p(50)));
        assert_eq!(parse_bg_position("25% 75%"), (p(25), p(75)));
        assert_eq!(
            parse_bg_position("10px 20px"),
            (BgPosVal::Px(10), BgPosVal::Px(20))
        );
        assert_eq!(
            parse_bg_position("-4px center"),
            (BgPosVal::Px(-4), p(50))
        );
    }

    #[test_case]
    fn bg_no_repeat_centered() {
        // 10x10 buffer of zeros; 2x2 checker no-repeat centered in an 8x6 rect
        // at (0,0): offset = ((8-2)/2, (6-2)/2) = (3, 2) -> tile at (3,2)..(5,4).
        let (w, h) = (10usize, 10usize);
        let mut buf = vec![0u32; w * h];
        let src = checker();
        paint_background_image(
            &mut buf, w, h, 0, 0, 8, 6, &src, 2, 2,
            BgRepeat::NoRepeat, BgSize::Auto,
            (BgPosVal::Percent(50), BgPosVal::Percent(50)),
        );
        assert_eq!(px_at(&buf, w, 3, 2), RED);
        assert_eq!(px_at(&buf, w, 4, 2), BLUE);
        assert_eq!(px_at(&buf, w, 3, 3), BLUE);
        assert_eq!(px_at(&buf, w, 4, 3), RED);
        // Everything outside the 2x2 tile is untouched.
        for y in 0..h {
            for x in 0..w {
                if (3..5).contains(&x) && (2..4).contains(&y) {
                    continue;
                }
                assert_eq!(px_at(&buf, w, x, y), 0, "pixel ({x},{y}) touched");
            }
        }
    }

    #[test_case]
    fn bg_repeat_tiles_exactly() {
        // 2x2 checker tiles an 8x6 rect exactly (4x3 tiles), anchor top-left.
        let (w, h) = (8usize, 6usize);
        let mut buf = vec![0u32; w * h];
        let src = checker();
        paint_background_image(
            &mut buf, w, h, 0, 0, 8, 6, &src, 2, 2,
            BgRepeat::Repeat, BgSize::Auto,
            (BgPosVal::Percent(0), BgPosVal::Percent(0)),
        );
        // Every pixel painted, in checker phase (x%2, y%2).
        for y in 0..h {
            for x in 0..w {
                let want = if (x % 2) == (y % 2) { RED } else { BLUE };
                assert_eq!(px_at(&buf, w, x, y), want, "pixel ({x},{y})");
            }
        }
    }

    #[test_case]
    fn bg_repeat_x_row_band_only() {
        let (w, h) = (8usize, 6usize);
        let mut buf = vec![0u32; w * h];
        let src = checker();
        paint_background_image(
            &mut buf, w, h, 0, 0, 8, 6, &src, 2, 2,
            BgRepeat::RepeatX, BgSize::Auto,
            (BgPosVal::Percent(0), BgPosVal::Percent(0)),
        );
        // Rows 0..2 fully painted; rows 2.. untouched.
        for x in 0..w {
            assert_ne!(px_at(&buf, w, x, 0), 0);
            assert_ne!(px_at(&buf, w, x, 1), 0);
        }
        for y in 2..h {
            for x in 0..w {
                assert_eq!(px_at(&buf, w, x, y), 0, "pixel ({x},{y}) touched");
            }
        }
    }

    #[test_case]
    fn bg_repeat_y_column_band_only() {
        let (w, h) = (8usize, 6usize);
        let mut buf = vec![0u32; w * h];
        let src = checker();
        paint_background_image(
            &mut buf, w, h, 0, 0, 8, 6, &src, 2, 2,
            BgRepeat::RepeatY, BgSize::Auto,
            (BgPosVal::Percent(0), BgPosVal::Percent(0)),
        );
        // Columns 0..2 fully painted; columns 2.. untouched.
        for y in 0..h {
            assert_ne!(px_at(&buf, w, 0, y), 0);
            assert_ne!(px_at(&buf, w, 1, y), 0);
        }
        for y in 0..h {
            for x in 2..w {
                assert_eq!(px_at(&buf, w, x, y), 0, "pixel ({x},{y}) touched");
            }
        }
    }

    #[test_case]
    fn bg_cover_fills_contain_letterboxes() {
        // Square 2x2 source into a wide 8x4 rect.
        let (w, h) = (8usize, 4usize);
        let src = checker();

        // Cover: tile scales to 8x8 -> every rect pixel painted (incl. corners).
        let mut buf = vec![0u32; w * h];
        paint_background_image(
            &mut buf, w, h, 0, 0, 8, 4, &src, 2, 2,
            BgRepeat::NoRepeat, BgSize::Cover,
            (BgPosVal::Percent(50), BgPosVal::Percent(50)),
        );
        for &(x, y) in &[(0, 0), (7, 0), (0, 3), (7, 3)] {
            assert_ne!(px_at(&buf, w, x, y), 0, "cover corner ({x},{y}) unpainted");
        }
        assert!(buf.iter().all(|&p| p != 0), "cover left a hole");

        // Contain: tile scales to 4x4 centered -> cols 0..2 and 6..8 letterbox.
        let mut buf = vec![0u32; w * h];
        paint_background_image(
            &mut buf, w, h, 0, 0, 8, 4, &src, 2, 2,
            BgRepeat::NoRepeat, BgSize::Contain,
            (BgPosVal::Percent(50), BgPosVal::Percent(50)),
        );
        for y in 0..h {
            for x in [0usize, 1, 6, 7] {
                assert_eq!(px_at(&buf, w, x, y), 0, "contain band ({x},{y}) touched");
            }
            for x in 2..6 {
                assert_ne!(px_at(&buf, w, x, y), 0, "contain tile ({x},{y}) unpainted");
            }
        }
    }

    #[test_case]
    fn bg_clips_offscreen_rect() {
        // Rect hangs off all four buffer edges; must not panic or write OOB.
        let (w, h) = (4usize, 4usize);
        let mut buf = vec![0u32; w * h];
        let src = checker();
        paint_background_image(
            &mut buf, w, h, -3, -2, 20, 20, &src, 2, 2,
            BgRepeat::Repeat, BgSize::Auto,
            (BgPosVal::Px(-5), BgPosVal::Px(7)),
        );
        // The visible intersection is fully painted with checker colors.
        assert!(buf.iter().all(|&p| p == RED || p == BLUE));

        // A no-repeat tile entirely outside the buffer paints nothing.
        let mut buf2 = vec![0u32; w * h];
        paint_background_image(
            &mut buf2, w, h, 10, 10, 8, 8, &src, 2, 2,
            BgRepeat::NoRepeat, BgSize::Auto,
            (BgPosVal::Percent(0), BgPosVal::Percent(0)),
        );
        assert!(buf2.iter().all(|&p| p == 0));

        // Zero-size guards: no panic on empty source or degenerate rect.
        paint_background_image(
            &mut buf2, w, h, 0, 0, 4, 4, &[], 0, 0,
            BgRepeat::Repeat, BgSize::Auto,
            (BgPosVal::Percent(0), BgPosVal::Percent(0)),
        );
        paint_background_image(
            &mut buf2, w, h, 0, 0, 0, 4, &src, 2, 2,
            BgRepeat::Repeat, BgSize::Cover,
            (BgPosVal::Percent(0), BgPosVal::Percent(0)),
        );
        assert!(buf2.iter().all(|&p| p == 0));
    }
}
