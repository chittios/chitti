//! Painting the analog clock face in the clock dropdown: dial, ring, minute
//! ticks, hour numerals and hands.
//!
//! Only the pixels are here. Where each mark points and how big it is comes from
//! [`crate::clock::face`], which is outside this module because `framebuffer` is
//! `#[cfg(not(test))]` — geometry that lives in here cannot be unit-tested, and
//! that is exactly how a quadrant reduction wrong in half the dial shipped.

use super::*;
use crate::clock::face::{clock_offset, hand_angles, hand_geoms};

/// Analog clock face: 0° at 12 o'clock, clockwise, integer Q10 trig.
///
/// Everything curved or angled here is **sub-pixel sampled** (`aa_coverage`)
/// rather than plotted: the ring is one anti-aliased annulus (it used to be
/// single pixels stepped every 2°, which left a dotted gap at every other
/// degree), and the hands are anti-aliased round-cap capsules (they used to be
/// 1 px Bresenham lines, thickened by drawing a second line one pixel to the
/// right — which is a staircase at every angle that isn't a multiple of 45°, and
/// made the hour hand lopsided since the offset was always `+x`).
pub(super) fn draw_analog_clock(sc: &Screen, cx: i64, cy: i64, r: i64, cw: u64, ch: u64, h: i64, mi: i64, s: i64) {
    let ring = sc.theme.border_dim;
    let fg = sc.theme.chat_fg;
    let dim = sc.theme.title_dim;
    let accent = sc.theme.accent;
    let dial = sc.theme.chat_bg;

    // Dial, then a crisp ring of thickness `t` around its rim.
    sc.fill_disc(cx, cy, r, dial);
    let t = (r / 14).max(1);
    stroke_ring(sc, cx, cy, r, t, ring);

    // A mark every minute; the five-minute ones longer and brighter.
    let rim = r - t - 1;
    let hour_tick = (r / 7).max(3);
    for m in 0..60i32 {
        let on_hour = m % 5 == 0;
        let len = if on_hour { hour_tick } else { (r / 14).max(2) };
        let w10 = if on_hour { (r * 10 / 26).max(14) } else { (r * 10 / 44).max(10) };
        let (ox, oy) = clock_offset(rim, m * 6);
        let (px, py) = clock_offset(rim - len, m * 6);
        fill_capsule(sc, cx + px, cy + py, cx + ox, cy + oy, w10, if on_hour { fg } else { dim });
    }

    // Hour numerals inside the tick ring — only where a text cell actually fits
    // there, so a small face degrades to ticks alone instead of to a smear.
    let (cw, ch) = (cw as i64, ch as i64);
    let num_r = rim - hour_tick - ch / 2 - 2;
    if num_r >= ch && r >= ch * 5 / 2 {
        for hour in 1..=12i64 {
            let (nx, ny) = clock_offset(num_r, (hour * 30) as i32);
            let label = alloc::format!("{hour}");
            let x = cx + nx - label.len() as i64 * cw / 2;
            let y = cy + ny - ch / 2;
            if x >= 0 && y >= 0 {
                sc.draw_str(x as u64, y as u64, &label, fg, dial);
            }
        }
    }

    // Hands. Hour and minute in the text colour, second in accent, each drawn
    // tail-through-tip so the counterweight lands on the far side of the pivot.
    let [hg, mg, sg] = hand_geoms(r);
    let (h_deg, m_deg, s_deg) = hand_angles(h, mi, s);
    for (g, deg, c) in [(hg, h_deg, fg), (mg, m_deg, fg), (sg, s_deg, accent)] {
        let (tx, ty) = clock_offset(g.len, deg);
        let (bx, by) = clock_offset(-g.tail, deg);
        fill_capsule(sc, cx + bx, cy + by, cx + tx, cy + ty, g.w10, c);
    }
    // Pivot: an accent hub with a dial-coloured pin hole once there's room.
    sc.fill_disc(cx, cy, (r / 12).max(2), accent);
    if r >= 36 {
        sc.fill_disc(cx, cy, (r / 40).max(1), dial);
    }
}

/// Anti-aliased ring: the annulus between radius `r` and `r - t`, in one pass.
fn stroke_ring(sc: &Screen, cx: i64, cy: i64, r: i64, t: i64, c: Rgb) {
    if r <= 0 || t <= 0 {
        return;
    }
    let outer = (2 * AA_SS * r).pow(2);
    let inner = (2 * AA_SS * (r - t).max(0)).pow(2);
    let span = r + 1;
    for dy in -span..=span {
        for dx in -span..=span {
            let a = aa_coverage(dx, dy, |fx, fy| {
                let d2 = fx * fx + fy * fy;
                d2 <= outer && d2 >= inner
            });
            // Negative coords wrap to a huge u64 and are dropped by blend_pixel.
            sc.blend_pixel((cx + dx) as u64, (cy + dy) as u64, c, a);
        }
    }
}

/// Fill the **capsule** (a thick segment with round end caps) from `(x0,y0)` to
/// `(x1,y1)`, `w10` wide in tenths of a pixel, anti-aliased.
///
/// This is what makes a hand read as a hand: coverage comes from the squared
/// point-to-segment distance evaluated on `aa_coverage`'s sub-pixel grid, so a
/// hand at 7° is as smooth as one at 90° and a sub-pixel width still shows up as
/// a faint even line rather than dropping out.
fn fill_capsule(sc: &Screen, x0: i64, y0: i64, x1: i64, y1: i64, w10: i64, c: Rgb) {
    /// The grid `aa_coverage` hands to its predicate is scaled by `2·AA_SS`.
    const K: i64 = 2 * AA_SS;
    // Half-width on that grid: (w10/10 px) / 2 → K·w10/20. Never below half a
    // sub-pixel step, or a thin hand would sample to nothing at all.
    let hw = (K * w10 / 20).max(K / 2);
    let hw2 = hw * hw;
    // Work relative to (x0, y0) — which is also the origin `aa_coverage`'s
    // `(fx, fy)` are measured from — so the products stay small.
    let (bx, by) = (K * (x1 - x0), K * (y1 - y0));
    let dd = bx * bx + by * by;
    let pad = w10 / 20 + 2;
    let (lo_x, hi_x) = ((x1 - x0).min(0) - pad, (x1 - x0).max(0) + pad);
    let (lo_y, hi_y) = ((y1 - y0).min(0) - pad, (y1 - y0).max(0) + pad);
    for dy in lo_y..=hi_y {
        for dx in lo_x..=hi_x {
            let a = aa_coverage(dx, dy, |fx, fy| {
                // Projection of the sample onto the segment, clamped to the caps.
                let dot = fx * bx + fy * by;
                if dd == 0 || dot <= 0 {
                    return fx * fx + fy * fy <= hw2;
                }
                if dot >= dd {
                    let (qx, qy) = (fx - bx, fy - by);
                    return qx * qx + qy * qy <= hw2;
                }
                // Perpendicular distance, compared without dividing.
                let cross = fx * by - fy * bx;
                cross * cross <= hw2 * dd
            });
            sc.blend_pixel((x0 + dx) as u64, (y0 + dy) as u64, c, a);
        }
    }
}
