//! Rounded-rectangle geometry for the console compositor.
//!
//! Lives here rather than in `framebuffer/` because that tree is
//! `#[cfg(not(test))]` — a test written next to the painter would never run.
//! Every decision a theme's `corner_radius` makes (clamp, scale, inside-test,
//! anti-aliased coverage, the overlay save-rect including the drop shadow) is
//! pinned here, and the compositor only rasterises the numbers.

/// Anti-aliasing sub-sample rate per axis (`AA_SS`×`AA_SS` samples per pixel).
/// Matches the compositor's curve helper so a disc and a rounded corner agree.
pub const AA_SS: i64 = 4;

/// Theme `corner_radius` when a file omits the key — 8 CSS-px at font scale 1,
/// which is the macOS popover radius (windows sit at 10; we share one token).
pub const RADIUS_DEFAULT: u64 = 8;

/// Hard ceiling on the theme token (CSS-px at scale 1). Larger values eat the
/// pane interior on a 1024-wide panel; `/theme radius` clamps here too.
pub const RADIUS_MAX: u64 = 24;

/// Clamp a theme token to `[0, RADIUS_MAX]`.
pub fn clamp_theme_radius(n: u64) -> u64 {
    n.min(RADIUS_MAX)
}

/// Corner radius in **device pixels** for a box of size `w`×`h`.
///
/// The theme stores CSS-px at font scale 1; multiplying by `scale` keeps the
/// same *visual* roundness on a 768p panel and a 1440p one, matching
/// [`crate::panes_layout::shadow_geom`]. A radius larger than half the shorter
/// side is clamped so a tiny dropdown cannot invert its corners.
pub fn radius_px(theme: u64, scale: u64, w: u64, h: u64) -> u64 {
    let r = clamp_theme_radius(theme).saturating_mul(scale.max(1));
    clamp_radius(r, w, h)
}

/// `r` fitted into a `w`×`h` box (never more than half the shorter side).
pub fn clamp_radius(r: u64, w: u64, h: u64) -> u64 {
    let cap = (w / 2).min(h / 2);
    r.min(cap)
}

/// Sub-sampled coverage of a pixel at integer offset `(dx, dy)` from a shape's
/// origin. `inside(fx, fy)` tests a sub-pixel point given in a **2·SS-scaled**
/// coordinate grid (sub-sample centres land on odd integers). Returns 0..=255.
pub fn aa_coverage<F: Fn(i64, i64) -> bool>(dx: i64, dy: i64, inside: F) -> u32 {
    let mut cov = 0u32;
    for sj in 0..AA_SS {
        let fy = 2 * AA_SS * dy + (2 * sj + 1 - AA_SS);
        for si in 0..AA_SS {
            let fx = 2 * AA_SS * dx + (2 * si + 1 - AA_SS);
            if inside(fx, fy) {
                cov += 1;
            }
        }
    }
    cov * 255 / (AA_SS * AA_SS) as u32
}

/// Is sub-pixel `(fx, fy)` (2·SS units, origin = box top-left) inside the
/// rounded rect of pixel size `w`×`h` and radius `r`?
pub fn inside_round_rect_sub(fx: i64, fy: i64, w: i64, h: i64, r: i64) -> bool {
    let ss = 2 * AA_SS;
    let (ww, hh, rr) = (w * ss, h * ss, r.max(0) * ss);
    if fx < 0 || fy < 0 || fx >= ww || fy >= hh {
        return false;
    }
    if r <= 0 {
        return true;
    }
    let cx = if fx < rr {
        rr
    } else if fx >= ww - rr {
        ww - rr
    } else {
        return true;
    };
    let cy = if fy < rr {
        rr
    } else if fy >= hh - rr {
        hh - rr
    } else {
        return true;
    };
    let (dx, dy) = (fx - cx, fy - cy);
    dx * dx + dy * dy <= rr * rr
}

/// Integer pixel-centre inside-test for `[0, w)×[0, h)` with corner radius `r`.
/// Used by hit-testing and by tests; the painter uses [`coverage`] for edges.
pub fn inside_round_rect(x: i64, y: i64, w: i64, h: i64, r: i64) -> bool {
    if x < 0 || y < 0 || x >= w || y >= h {
        return false;
    }
    let r = r.max(0).min(w / 2).min(h / 2);
    if r == 0 {
        return true;
    }
    let cx = if x < r {
        r
    } else if x >= w - r {
        w - r
    } else {
        return true;
    };
    let cy = if y < r {
        r
    } else if y >= h - r {
        h - r
    } else {
        return true;
    };
    let (dx, dy) = (x - cx, y - cy);
    dx * dx + dy * dy <= r * r
}

/// Anti-aliased coverage (0..=255) of pixel `(px, py)` relative to a rounded
/// rect of size `w`×`h` and radius `r` whose top-left is the origin.
pub fn coverage(px: i64, py: i64, w: i64, h: i64, r: i64) -> u32 {
    if w <= 0 || h <= 0 {
        return 0;
    }
    let r = r.max(0);
    aa_coverage(px, py, |fx, fy| inside_round_rect_sub(fx, fy, w, h, r))
}

/// Coverage of a `t`-thick rounded stroke: the outer rounded rect minus the
/// inset inner one. Nested, so `outer - inner` is exact (not an XOR guess).
pub fn stroke_coverage(px: i64, py: i64, w: i64, h: i64, r: i64, t: i64) -> u32 {
    let outer = coverage(px, py, w, h, r);
    if t <= 0 {
        return 0;
    }
    if t >= w / 2 || t >= h / 2 {
        return outer;
    }
    let inner = coverage(
        px - t,
        py - t,
        w - 2 * t,
        h - 2 * t,
        (r - t).max(0),
    );
    outer.saturating_sub(inner)
}

/// The rectangle an overlay must save to restore a box *and* its drop shadow.
///
/// Light comes from above, so the shadow does not reach past the box top (the
/// compositor clips there). Sides and the dropped bottom need `blur` extra
/// pixels; the bottom also needs the shadow `offset`. Clipped to the screen.
pub fn overlay_rect(
    x: u64,
    y: u64,
    w: u64,
    h: u64,
    blur: u64,
    offset: u64,
    sw: u64,
    sh: u64,
) -> (u64, u64, u64, u64) {
    let x0 = x.saturating_sub(blur);
    let y0 = y; // no shadow above the box
    let x1 = x.saturating_add(w).saturating_add(blur).min(sw);
    let y1 = y.saturating_add(h).saturating_add(offset).saturating_add(blur).min(sh);
    (x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn theme_radius_clamps_and_scales() {
        assert_eq!(clamp_theme_radius(0), 0);
        assert_eq!(clamp_theme_radius(8), 8);
        assert_eq!(clamp_theme_radius(99), RADIUS_MAX);
        // Scale 2 doubles the device-pixel radius.
        assert_eq!(radius_px(8, 1, 400, 300), 8);
        assert_eq!(radius_px(8, 2, 400, 300), 16);
        // A scale of 0 is treated as 1 — same rule as shadow_geom.
        assert_eq!(radius_px(8, 0, 400, 300), 8);
        // A box smaller than 2r cannot invert its corners.
        assert_eq!(radius_px(8, 1, 10, 10), 5);
        assert_eq!(radius_px(8, 1, 4, 100), 2);
        assert_eq!(clamp_radius(0, 100, 100), 0);
    }

    #[test_case]
    fn inside_round_rect_cuts_the_square_corners() {
        let (w, h, r) = (20i64, 10i64, 4i64);
        // Interior and straight edges stay inside.
        assert!(inside_round_rect(10, 5, w, h, r));
        assert!(inside_round_rect(0, 5, w, h, r)); // left edge, mid
        assert!(inside_round_rect(10, 0, w, h, r)); // top edge, mid
        // The four square-corner pixels fall outside the quarter-circle.
        assert!(!inside_round_rect(0, 0, w, h, r));
        assert!(!inside_round_rect(w - 1, 0, w, h, r));
        assert!(!inside_round_rect(0, h - 1, w, h, r));
        assert!(!inside_round_rect(w - 1, h - 1, w, h, r));
        // Radius 0 is a plain rectangle (the retro-82 path).
        assert!(inside_round_rect(0, 0, w, h, 0));
        assert!(!inside_round_rect(-1, 0, w, h, 0));
        assert!(!inside_round_rect(w, 0, w, h, 0));
    }

    #[test_case]
    fn coverage_is_opaque_inside_and_zero_outside() {
        let (w, h, r) = (40i64, 24i64, 8i64);
        assert_eq!(coverage(20, 12, w, h, r), 255, "dead centre");
        // One pixel in from the edge — the edge itself is a 50% AA fringe
        // (half the sub-samples sit at fy < 0), which is the point of AA.
        assert_eq!(coverage(20, 2, w, h, r), 255, "inside, away from the arc");
        assert_eq!(coverage(-2, 12, w, h, r), 0);
        assert_eq!(coverage(20, 40, w, h, r), 0);
        // The outer square corner is outside the arc — coverage must not be full.
        let corner = coverage(0, 0, w, h, r);
        assert!(corner < 128, "square corner coverage {corner} should be a fringe");
        // A pixel on the arc (roughly r along the diagonal from the centre of
        // the corner circle) is a fractional edge, not 0 or 255.
        let arc = coverage(2, 2, w, h, r);
        assert!(arc > 0 && arc < 255, "arc pixel coverage {arc}");
        // Radius 0 is a sharp box; the outer 1px is still an AA fringe (half
        // the sub-samples sit outside the edge), so opacity is asserted one
        // pixel in, not on the rim.
        assert_eq!(coverage(2, 2, w, h, 0), 255);
        assert_eq!(coverage(w - 3, h - 3, w, h, 0), 255);
        let rim = coverage(0, 0, w, h, 0);
        assert!(rim > 0 && rim < 255, "sharp-box rim is AA, got {rim}");
    }

    #[test_case]
    fn stroke_coverage_is_a_ring() {
        let (w, h, r, t) = (40i64, 24i64, 8i64, 2i64);
        // Dead centre is inside the inner rect → no stroke.
        assert_eq!(stroke_coverage(20, 12, w, h, r, t), 0);
        // Mid-top outer pixel is on the stroke (partial AA is still ink).
        let top = stroke_coverage(20, 0, w, h, r, t);
        assert!(top > 0, "top edge of a {t}px stroke should be ink, got {top}");
        let top_in = stroke_coverage(20, 1, w, h, r, t);
        assert!(top_in > 0, "second row of a {t}px stroke should be ink, got {top_in}");
        // One pixel further in than the thickness is already the fill, not the
        // stroke (the inner rect starts at t).
        assert_eq!(stroke_coverage(20, t + 2, w, h, r, t), 0);
        // Degenerate thickness paints nothing; a stroke thicker than the box
        // becomes a fill.
        assert_eq!(stroke_coverage(20, 12, w, h, r, 0), 0);
        assert_eq!(stroke_coverage(20, 12, 6, 6, 2, 4), coverage(20, 12, 6, 6, 2));
    }

    #[test_case]
    fn overlay_rect_includes_the_shadow_not_the_sky() {
        let g_blur = 6u64;
        let g_off = 2u64;
        let (x, y, w, h) = overlay_rect(100, 50, 200, 80, g_blur, g_off, 640, 480);
        // Left/right grow by blur; top stays put; bottom grows by offset+blur.
        assert_eq!(x, 100 - g_blur);
        assert_eq!(y, 50, "no shadow above the box");
        assert_eq!(w, 200 + 2 * g_blur);
        assert_eq!(h, 80 + g_off + g_blur);
        // Clipped to the screen — a box against the right edge must not wrap.
        let (x, y, w, h) = overlay_rect(630, 0, 20, 10, 6, 2, 640, 480);
        assert_eq!(x + w, 640);
        assert_eq!(y, 0);
        assert!(h > 10);
        // Empty/zero blur is the box itself.
        assert_eq!(overlay_rect(10, 10, 8, 8, 0, 0, 100, 100), (10, 10, 8, 8));
    }
}
