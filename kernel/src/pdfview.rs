//! Pure geometry and state for the **PDF viewer tab** — scale selection, page
//! stepping, scroll clamping, and the crop/letterbox compose.
//!
//! Deliberately outside `framebuffer/` (which is `#[cfg(not(test))]`, so a test
//! written in there is never compiled) and free of any renderer coupling: the
//! whole module takes numbers and pixel slices, which is what makes the fiddly
//! parts — the millipoint scale arithmetic, the pan clamp, the page-boundary
//! scroll — checkable by `cargo xtask test` instead of by eye on a booted OS.
//!
//! # Why the page is re-rendered on zoom rather than scaled
//!
//! Zooming a bitmap that was rasterized at fit-scale gives soft text, which is
//! the thing "proper preview" is supposed to fix. So zoom changes the scale the
//! renderer is *asked* for and the page comes back crisp at that size; this
//! module only ever moves a 1:1 window over the result ([`compose`]). The cost
//! is a re-render per zoom step (see [`crate::shell::pdf`] for the caching that
//! keeps paging cheap), and the benefit is that text is sharp at every zoom.

use alloc::string::String;
use alloc::vec::Vec;

/// The host's **render budget** for one page, in pixels.
///
/// There are deliberately two ceilings, and this is the lower one. The guest's
/// `MAX_PIXELS` (8.3 MP, in `tools/pdfrender-wasm`) is a *safety* limit — the
/// point past which it refuses rather than trapping on an allocation it cannot
/// satisfy. This one is a *latency* limit, and it exists because of what a real
/// document measured: the two attention-matrix pages of the Transformer paper
/// take ~6.9 s at a pane-fit scale and **20.7 s at 8 MP** under the interpreter.
/// A pane-fit render is ~0.5-1.5 MP, so this never binds in ordinary use; it
/// only stops a 400% zoom on a pathological page from freezing the console for
/// twenty seconds, at the cost of slightly softer text at extreme zoom.
///
/// Keep it **at or below** the guest's number: above it, the host would ask for
/// renders the guest is bound to refuse.
pub const MAX_PIXELS: u64 = 4_000_000;

/// Zoom is a percentage of the fit scale, so 100 always means "as the fit mode
/// chose" and the bounds are about how far from that a user can go.
pub const MIN_ZOOM: u32 = 25;
pub const MAX_ZOOM: u32 = 400;
pub const ZOOM_STEP: u32 = 25;

/// Smallest scale worth rendering (px per point, in permille) — a floor so a
/// tiny pane or a poster-sized page still produces a readable few pixels rather
/// than a zero-sized render the guest would refuse.
const MIN_PERMILLE: u32 = 40;

/// How the base scale is chosen before zoom is applied.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Fit {
    /// Whole page visible (both dimensions fit) — the opening view.
    Page,
    /// Page width fills the pane; taller pages scroll. The reading view.
    Width,
}

/// Interactive state of one open document.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct View {
    /// 0-based page index.
    pub page: usize,
    pub pages: usize,
    /// Percent of the fit scale.
    pub zoom: u32,
    pub fit: Fit,
    /// Top-left of the visible window inside the rendered page, in pixels.
    pub pan_x: i64,
    pub pan_y: i64,
}

impl View {
    pub fn new(pages: usize) -> View {
        View { page: 0, pages, zoom: 100, fit: Fit::Page, pan_x: 0, pan_y: 0 }
    }
}

/// Base scale (px per point, in **permille**) that makes the page fit the pane.
///
/// `page_mpt` is the page size in **millipoints** — points x 1000, which is how
/// the guest reports it so the host needs no float in the ABI. A page is ~600pt
/// wide, so a pane-relative scale is a small ratio and integer permille keeps
/// ~0.1% precision without ever overflowing (`pane_px * 1_000_000` for a 4K pane
/// is ~4e9, hence the `u64`).
pub fn fit_permille(page_mpt: (u32, u32), pane: (u64, u64), fit: Fit) -> u32 {
    let (mw, mh) = (page_mpt.0 as u64, page_mpt.1 as u64);
    let (pw, ph) = pane;
    if mw == 0 || mh == 0 || pw == 0 || ph == 0 {
        return 1000;
    }
    let by_w = pw.saturating_mul(1_000_000) / mw;
    let s = match fit {
        Fit::Width => by_w,
        Fit::Page => by_w.min(ph.saturating_mul(1_000_000) / mh),
    };
    (s.max(MIN_PERMILLE as u64) as u32).max(MIN_PERMILLE)
}

/// The scale to ask the renderer for: fit scale x zoom, clamped so the result
/// cannot exceed [`MAX_PIXELS`].
///
/// Clamping here rather than after a refusal keeps zoom monotonic: the user
/// presses `+` and gets the largest render that is actually possible, instead of
/// an error and an unchanged page.
pub fn render_permille(page_mpt: (u32, u32), pane: (u64, u64), fit: Fit, zoom: u32) -> u32 {
    let base = fit_permille(page_mpt, pane, fit) as u64;
    let want = base * zoom.clamp(MIN_ZOOM, MAX_ZOOM) as u64 / 100;
    cap_permille(page_mpt, want.max(MIN_PERMILLE as u64) as u32)
}

/// Reduce a scale until the rendered page fits the pixel budget.
pub fn cap_permille(page_mpt: (u32, u32), permille: u32) -> u32 {
    let (mw, mh) = (page_mpt.0 as u64, page_mpt.1 as u64);
    if mw == 0 || mh == 0 {
        return permille.max(MIN_PERMILLE);
    }
    let px = |p: u64| -> u64 {
        // pixels = (mpt / 1000 pt) * (p / 1000 px/pt) = mpt * p / 1_000_000
        let w = mw.saturating_mul(p) / 1_000_000;
        let h = mh.saturating_mul(p) / 1_000_000;
        w.saturating_mul(h)
    };
    let mut p = permille.max(MIN_PERMILLE) as u64;
    if px(p) <= MAX_PIXELS {
        return p as u32;
    }
    // Area grows with the square of the scale, so scale down by the square root
    // of the overshoot; one integer Newton step then a short walk is exact
    // enough and needs no float (this runs in the kernel).
    let over = px(p) / MAX_PIXELS.max(1);
    let mut root = 1u64;
    while root * root < over.max(1) {
        root += 1;
    }
    p = (p / root).max(MIN_PERMILLE as u64);
    while px(p) > MAX_PIXELS && p > MIN_PERMILLE as u64 {
        p = p * 9 / 10;
    }
    p.max(MIN_PERMILLE as u64) as u32
}

/// Rendered-page size in pixels at a given scale — what the guest will produce,
/// computed the same way (floor after the multiply) so the host's crop math and
/// the guest's buffer agree.
pub fn page_px(page_mpt: (u32, u32), permille: u32) -> (u64, u64) {
    let f = |m: u32| (m as u64).saturating_mul(permille as u64) / 1_000_000;
    (f(page_mpt.0), f(page_mpt.1))
}

/// Clamp a pan offset so the visible window stays inside the rendered page.
/// A dimension smaller than the pane is centred by the compose step, so its pan
/// is pinned to 0 rather than allowed to drift the page off-screen.
pub fn clamp_pan(img: (u64, u64), pane: (u64, u64), pan: (i64, i64)) -> (i64, i64) {
    let axis = |img: u64, pane: u64, v: i64| -> i64 {
        if img <= pane {
            0
        } else {
            v.clamp(0, (img - pane) as i64)
        }
    };
    (axis(img.0, pane.0, pan.0), axis(img.1, pane.1, pan.1))
}

/// Vertical scroll that **crosses page boundaries**, the way a document reader
/// does: scrolling past the bottom of a page moves to the top of the next one,
/// and past the top moves to the bottom of the previous.
///
/// Returns the new view; `img`/`pane` describe the *current* page's render, so
/// the caller re-renders when `page` changed (the new page's height is unknown
/// here — landing at its bottom is expressed as `i64::MAX` and re-clamped once
/// that render exists).
pub fn scroll(mut v: View, img: (u64, u64), pane: (u64, u64), dy: i64) -> View {
    let max = img.1.saturating_sub(pane.1) as i64;
    let want = v.pan_y + dy;
    if want > max && v.page + 1 < v.pages {
        v.page += 1;
        v.pan_y = 0;
        return v;
    }
    if want < 0 && v.page > 0 {
        v.page -= 1;
        v.pan_y = i64::MAX; // clamped against the previous page's real height
        return v;
    }
    v.pan_y = want.clamp(0, max.max(0));
    v
}

/// Step pages by `delta`, clamped to the document (never wraps: wrapping from
/// the last page to the first reads as a lost place, not as navigation).
pub fn step_page(v: View, delta: i64) -> View {
    let mut v = v;
    if v.pages == 0 {
        return v;
    }
    let last = (v.pages - 1) as i64;
    let p = (v.page as i64 + delta).clamp(0, last);
    if p as usize != v.page {
        v.page = p as usize;
        // A new page starts at its top-left; keeping the old offset would open
        // page 2 halfway down.
        v.pan_x = 0;
        v.pan_y = 0;
    }
    v
}

pub fn zoom_by(mut v: View, delta: i32) -> View {
    let z = (v.zoom as i32 + delta).clamp(MIN_ZOOM as i32, MAX_ZOOM as i32) as u32;
    v.zoom = z;
    v
}

/// Copy a pane-sized 1:1 window out of the rendered page.
///
/// The result is exactly `pane` pixels, so the compositor's aspect-fit becomes
/// the identity and no resampling touches the glyphs. Where the page is smaller
/// than the pane it is centred on `bg`; where it is larger, `pan` selects the
/// visible window.
pub fn compose(img: (u64, u64), src: &[u32], pane: (u64, u64), pan: (i64, i64), bg: u32) -> Vec<u32> {
    let (iw, ih) = (img.0 as usize, img.1 as usize);
    let (pw, ph) = (pane.0 as usize, pane.1 as usize);
    let mut out = alloc::vec![bg; pw * ph];
    if iw == 0 || ih == 0 || pw == 0 || ph == 0 || src.len() < iw * ih {
        return out;
    }
    // Offset of the page inside the pane (centred when it is smaller), and the
    // first source pixel shown (pan, when it is larger).
    let ox = if iw < pw { (pw - iw) / 2 } else { 0 };
    let oy = if ih < ph { (ph - ih) / 2 } else { 0 };
    let sx = if iw > pw { pan.0.max(0) as usize } else { 0 };
    let sy = if ih > ph { pan.1.max(0) as usize } else { 0 };
    let cols = (iw.saturating_sub(sx)).min(pw - ox);
    let rows = (ih.saturating_sub(sy)).min(ph - oy);
    for r in 0..rows {
        let s = (sy + r) * iw + sx;
        let d = (oy + r) * pw + ox;
        out[d..d + cols].copy_from_slice(&src[s..s + cols]);
    }
    out
}

/// Convert the guest's premultiplied RGBA8 bytes to the compositor's `0x00RRGGBB`.
///
/// The page is rendered on **opaque white**, so alpha is 255 everywhere and no
/// un-premultiplication is needed — the channels are already the display values.
/// Done host-side on purpose: it is one native pass over the buffer the host has
/// to copy anyway, where in the guest it would be a million interpreted loops.
pub fn rgba_to_rgb(rgba: &[u8], pixels: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(pixels);
    for px in rgba.chunks_exact(4).take(pixels) {
        out.push(((px[0] as u32) << 16) | ((px[1] as u32) << 8) | px[2] as u32);
    }
    // A short buffer would otherwise silently produce a smaller image than the
    // header claimed; pad so `compose`'s length check stays meaningful.
    while out.len() < pixels {
        out.push(0x00ff_ffff);
    }
    out
}

/// The HUD line under the page: where you are, how big, and the keys.
pub fn hud(v: &View, permille: u32, ms: u64) -> String {
    let fit = match v.fit {
        Fit::Page => "fit page",
        Fit::Width => "fit width",
    };
    alloc::format!(
        "page {}/{}  {}  {}%  ({}ms)   PgUp/PgDn page  +/- zoom  arrows scroll  f fit  0 reset  Ctrl+C close",
        v.page + 1,
        v.pages.max(1),
        fit,
        permille / 10,
        ms
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A4 in points, as `pdf_page_size` reports it (millipoints).
    const A4: (u32, u32) = (595_000, 842_000);

    #[test_case]
    fn fit_page_fits_both_dimensions_and_fit_width_only_the_width() {
        // A tall-ish pane: fit-page is limited by height, fit-width by width.
        let pane = (600, 700);
        let page = fit_permille(A4, pane, Fit::Page);
        let width = fit_permille(A4, pane, Fit::Width);
        let (pw, ph) = page_px(A4, page);
        assert!(pw <= pane.0 && ph <= pane.1, "fit page must fit: {pw}x{ph} in {pane:?}");
        let (ww, _) = page_px(A4, width);
        assert!(ww <= pane.0 && ww + 2 >= pane.0, "fit width must fill width: {ww} vs {}", pane.0);
        assert!(width > page, "a 600x700 pane is wider than A4 is tall-proportioned");
    }

    #[test_case]
    fn zoom_scales_the_render_not_the_bitmap() {
        let pane = (600, 800);
        let base = render_permille(A4, pane, Fit::Page, 100);
        let twice = render_permille(A4, pane, Fit::Page, 200);
        // 200% must ask for ~2x the scale — that is what keeps text crisp when
        // zoomed instead of magnifying the fit-scale pixels.
        assert!(twice >= base * 19 / 10 && twice <= base * 21 / 10, "{base} -> {twice}");
    }

    #[test_case]
    fn a_huge_zoom_is_capped_to_the_pixel_budget_not_refused() {
        // 400% on a big pane would blow past MAX_PIXELS; the cap must bring it
        // under, because the guest refuses anything larger.
        let p = render_permille(A4, (3840, 2160), Fit::Width, MAX_ZOOM);
        let (w, h) = page_px(A4, p);
        assert!(w * h <= MAX_PIXELS, "{w}x{h} = {} > {MAX_PIXELS}", w * h);
        assert!(p >= MIN_PERMILLE);
    }

    #[test_case]
    fn cap_permille_leaves_a_scale_that_already_fits_untouched() {
        let p = cap_permille(A4, 1500);
        assert_eq!(p, 1500);
    }

    #[test_case]
    fn pan_is_clamped_to_the_page_and_pinned_when_it_fits() {
        // Larger than the pane: pan is allowed up to the difference.
        assert_eq!(clamp_pan((1000, 2000), (600, 800), (5000, 5000)), (400, 1200));
        assert_eq!(clamp_pan((1000, 2000), (600, 800), (-50, -50)), (0, 0));
        // Smaller than the pane: centred by compose, so pan must be 0.
        assert_eq!(clamp_pan((300, 400), (600, 800), (100, 100)), (0, 0));
    }

    #[test_case]
    fn scrolling_past_the_bottom_turns_the_page() {
        let v = View { page: 0, pages: 3, zoom: 100, fit: Fit::Width, pan_x: 0, pan_y: 1200 };
        let after = scroll(v, (600, 2000), (600, 800), 400);
        assert_eq!((after.page, after.pan_y), (1, 0), "at the bottom, scrolling down turns the page");
        // ...and past the top goes back, landing at the previous page's bottom
        // (i64::MAX is re-clamped once that page's height is known).
        let up = scroll(View { pan_y: 0, ..after }, (600, 2000), (600, 800), -400);
        assert_eq!(up.page, 0);
        assert_eq!(up.pan_y, i64::MAX);
        // The last page does not wrap.
        let last = View { page: 2, pages: 3, zoom: 100, fit: Fit::Width, pan_x: 0, pan_y: 1200 };
        assert_eq!(scroll(last, (600, 2000), (600, 800), 400).page, 2);
    }

    #[test_case]
    fn step_page_clamps_and_resets_the_offset() {
        let v = View { page: 1, pages: 3, zoom: 100, fit: Fit::Width, pan_x: 10, pan_y: 900 };
        let next = step_page(v, 1);
        assert_eq!((next.page, next.pan_x, next.pan_y), (2, 0, 0), "a new page opens at its top");
        assert_eq!(step_page(next, 1).page, 2, "no wrap past the last page");
        assert_eq!(step_page(v, -5).page, 0);
        // A no-op step must not throw away where you were on the page.
        let stay = step_page(View { page: 0, ..v }, -1);
        assert_eq!((stay.page, stay.pan_y), (0, 900));
    }

    #[test_case]
    fn compose_centres_a_small_page_and_crops_a_large_one() {
        // 2x2 page into a 4x4 pane: centred, background elsewhere.
        let src = alloc::vec![1, 2, 3, 4];
        let out = compose((2, 2), &src, (4, 4), (0, 0), 0);
        assert_eq!(out.len(), 16);
        assert_eq!(out[5], 1, "page top-left lands at (1,1)");
        assert_eq!(out[6], 2);
        assert_eq!(out[9], 3);
        assert_eq!(out[0], 0, "margins are background");
        // 4x4 page into a 2x2 pane at pan (1,1): the window is 1:1, not scaled.
        let big: alloc::vec::Vec<u32> = (0..16).collect();
        let win = compose((4, 4), &big, (2, 2), (1, 1), 0);
        assert_eq!(win, alloc::vec![5, 6, 9, 10]);
    }

    #[test_case]
    fn compose_survives_a_short_or_empty_source() {
        // A guest that reported more pixels than it wrote must not panic here —
        // the buffer is bounds-checked, and a background pane is the safe answer.
        let out = compose((100, 100), &[1, 2, 3], (4, 4), (0, 0), 7);
        assert_eq!(out, alloc::vec![7; 16]);
        assert_eq!(compose((0, 0), &[], (2, 2), (0, 0), 7), alloc::vec![7; 4]);
    }

    #[test_case]
    fn rgba_to_rgb_keeps_channel_order_and_pads_a_short_buffer() {
        // R=0x12 G=0x34 B=0x56 A=0xff -> 0x123456 (alpha dropped, not blended:
        // the page is rendered on opaque white).
        assert_eq!(rgba_to_rgb(&[0x12, 0x34, 0x56, 0xff], 1), alloc::vec![0x0012_3456]);
        assert_eq!(rgba_to_rgb(&[0x12, 0x34, 0x56, 0xff], 3).len(), 3);
    }

    #[test_case]
    fn hud_names_the_page_and_the_zoom() {
        let v = View { page: 4, pages: 35, zoom: 100, fit: Fit::Width, pan_x: 0, pan_y: 0 };
        let s = hud(&v, 1500, 820);
        assert!(s.contains("page 5/35"), "{s}");
        assert!(s.contains("150%"), "{s}");
        assert!(s.contains("fit width"), "{s}");
    }
}
