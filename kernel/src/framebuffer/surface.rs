//! Presenting an app's own RGB buffer into an action pane: fit/scale, the
//! reserved HUD strip, and mapping a click back to surface coordinates.

use super::*;

/// Surface id the `/open` image viewer uses (also known to the shell). A
/// `Surface(IMAGE_SURFACE)` tab is labelled "image" in the tab bar.
pub const IMAGE_SURFACE: u32 = u32::MAX;

/// Surface id the `/open` video player presents frames on (labelled "video").
pub const VIDEO_SURFACE: u32 = u32::MAX - 1;

/// Surface id the browser agent paints pages on (labelled "browser").
pub const BROWSER_SURFACE: u32 = u32::MAX - 2;

/// Present an agent's surface backing buffer (`sw`×`sh`, 0xRRGGBB pixels) into
/// the action pane, opening it in `Surface(id)` mode on first present. The image
/// is nearest-neighbour scaled to fit the pane interior, letterboxed. Called by
/// `synapse::ui` after a `ui_draw`; the compositor is the only place surface
/// pixels reach the screen (the determinism boundary stays intact — the agent
/// emitted grammar-validated draw ops, never raw pixels here).
pub fn present_surface(id: u32, sw: usize, sh: usize, buf: &[u32]) {
    present_surface_reserve(id, sw, sh, buf, 0);
}

/// Present a surface plus an optional **HUD** in a reserved pane-space strip.
/// `hud` is newline-separated: line 0 = status (accent), the rest = hints
/// (dim, word-wrapped to the pane). The strip is sized to the wrapped content
/// and rendered with the native console font — crisp and wrapping at any pane
/// size — instead of being baked into the (upscaled) surface buffer. Empty
/// `hud` behaves exactly like [`present_surface`].
pub fn present_surface_hud(id: u32, sw: usize, sh: usize, buf: &[u32], hud: &str) {
    present_surface_hud_ex(id, sw, sh, sw, sh, buf, hud);
}

/// Like [`present_surface_hud`], but the pixel buffer may already be presentation-
/// scaled (`buf_w×buf_h`) while hit-testing still uses `logical_sw×logical_sh`.
pub fn present_surface_hud_ex(
    id: u32,
    logical_sw: usize,
    logical_sh: usize,
    buf_w: usize,
    buf_h: usize,
    buf: &[u32],
    hud: &str,
) {
    if hud.trim().is_empty() {
        present_surface_reserve_ex(id, logical_sw, logical_sh, buf_w, buf_h, buf, 0);
        return;
    }
    // Compute reserve *before* any SCREEN critical section. `draw_surface_hud`
    // already holds SCREEN; calling `surface_hud_height` from inside it would
    // re-enter the non-reentrant spinlock and hang forever (chess open path:
    // host_hud_set → present → draw_surface_hud).
    let reserve = surface_hud_height(hud);
    present_surface_reserve_ex(id, logical_sw, logical_sh, buf_w, buf_h, buf, reserve);
    draw_surface_hud(id, hud, reserve);
}

/// Height (px) a surface HUD needs: one status line + the wrapped hint lines,
/// plus a top hairline and small padding — computed at the current pane width.
///
/// Must **not** be called while already holding `SCREEN` (see
/// [`present_surface_hud`]). Pure layout math is in [`hud_strip_height`].
/// Public so package-UI present can size its pre-scaled text buffer to the
/// same usable pane the compositor will use.
pub fn surface_hud_reserve(hud: &str) -> u64 {
    surface_hud_height(hud)
}

fn surface_hud_height(hud: &str) -> u64 {
    SCREEN.with(|slot| {
        let Some(sc) = slot.as_ref() else { return 0 };
        let cols = (sc.logs().cols.saturating_sub(2)).max(4) as usize; // focused column's width
        hud_strip_height(hud, sc.ch(), cols)
    })
}

/// Pure: pixel height of the reserved HUD strip for `hud` at cell height `ch`
/// and wrap width `cols`. Unit-tested.
pub(super) fn hud_strip_height(hud: &str, ch: u64, cols: usize) -> u64 {
    let mut lines = 1u64; // status
    lines += wrapped_hint_lines(hud, cols);
    // top hairline + half-cell top/bottom padding.
    (lines * ch) + ch / 2 + 2
}

/// Count how many display lines the HUD's hint text (everything after line 0)
/// wraps to at `cols` columns. Pure-ish (reads only the passed args).
pub(super) fn wrapped_hint_lines(hud: &str, cols: usize) -> u64 {
    let mut it = hud.split('\n');
    let _status = it.next();
    let hints: alloc::vec::Vec<&str> = it.collect();
    if hints.is_empty() {
        return 0;
    }
    // Wrap on word boundaries; a token longer than cols still takes a line.
    let mut lines = 1u64;
    let mut col = 0usize;
    for hint in &hints {
        for word in hint.split_whitespace() {
            let wlen = word.chars().count();
            let need = if col == 0 { wlen } else { col + 1 + wlen };
            if need > cols && col > 0 {
                lines += 1;
                col = wlen;
            } else {
                col = need;
            }
        }
        // Each explicit hint line after the first forces a new row.
        lines += 1;
        col = 0;
    }
    lines.saturating_sub(1).max(1)
}

/// Render a surface's HUD in the reserved bottom strip of its pane (native
/// font, wrapping). No-op unless that surface tab is active.
///
/// `barh` must be the same value used for `present_surface_reserve`'s
/// `reserve_bottom` — precomputed *outside* this critical section so we never
/// re-enter `SCREEN` (non-reentrant spinlock).
fn draw_surface_hud(id: u32, hud: &str, barh: u64) {
    SCREEN.with(|slot| {
        let Some(sc) = slot else { return };
        let Some(d) = sc.mode_dims(RightMode::Surface(id)) else {
            return;
        };
        sc.cursor_restore();
        sc.cur_vis = false;
        let ch = sc.ch();
        let cw = sc.cw();
        let (px, py) = (d.ix, d.iy);
        let (pw, ph) = (d.iw, d.ih);
        let by = py + ph.saturating_sub(barh);
        let bg = d.bg;
        sc.fill_rect(px, by, pw, barh, bg);
        sc.fill_rect(px, by, pw, 1, sc.theme.accent); // top hairline
        let cols = (pw / cw).saturating_sub(2).max(4) as usize;
        let fit = |s: &str| crate::textsel::fit_width(s, cols);
        let mut lines = hud.split('\n');
        let mut y = by + ch / 3;
        // Status line (accent).
        if let Some(status) = lines.next() {
            sc.draw_str_bg(px + cw, y, &fit(status), sc.theme.accent, bg);
            y += ch;
        }
        // Hint lines (dim), word-wrapped; stop when the strip is full.
        let hud_bottom = py + ph;
        let mut linebuf = String::new();
        let flush = |sc: &mut Screen, y: &mut u64, s: &str| {
            if *y + ch <= hud_bottom {
                sc.draw_str_bg(px + cw, *y, &fit(s), sc.theme.logs_fg, bg);
                *y += ch;
            }
        };
        for hint in lines {
            for word in hint.split_whitespace() {
                let cand = if linebuf.is_empty() { String::from(word) } else { alloc::format!("{linebuf} {word}") };
                if cand.chars().count() > cols && !linebuf.is_empty() {
                    flush(sc, &mut y, &linebuf);
                    linebuf = String::from(word);
                } else {
                    linebuf = cand;
                }
            }
            if !linebuf.is_empty() {
                flush(sc, &mut y, &linebuf);
                linebuf.clear();
            }
        }
        sc.cursor_overlay();
    });
}

/// Choose destination size for presenting `sw×sh` into a `pw×ph` pane.
///
/// * **Upscale** (`free fit ≥ source`): integer pixel scale so each source
///   pixel becomes an `s×s` block — keeps package-UI text/rects crisp.
/// * **Downscale** (source larger than pane): free aspect-fit so video still
///   fills without cropping.
///
/// Pure — unit-tested.
pub fn present_fit(sw: u64, sh: u64, pw: u64, ph: u64) -> (u64, u64) {
    if sw == 0 || sh == 0 || pw == 0 || ph == 0 {
        return (0, 0);
    }
    // Free aspect-fit ("contain").
    let fit_w = pw;
    let fit_h = sh.saturating_mul(pw).saturating_div(sw).max(1);
    let (free_w, free_h) = if fit_h <= ph {
        (fit_w, fit_h)
    } else {
        let fit_h = ph;
        let fit_w = sw.saturating_mul(ph).saturating_div(sh).max(1);
        (fit_w.min(pw), fit_h)
    };
    // Integer upscale when the free fit would grow the image.
    if free_w >= sw && free_h >= sh {
        let s = (pw / sw).min(ph / sh).max(1);
        let iw = sw.saturating_mul(s);
        let ih = sh.saturating_mul(s);
        if iw <= pw && ih <= ph && s >= 1 {
            return (iw, ih);
        }
    }
    (free_w, free_h)
}

/// Like [`present_surface`], but leaves `reserve_bottom` px at the bottom of the
/// pane untouched — the frame is scaled/letterboxed into the region *above* the
/// reserve and the reserved strip is never cleared. The video player uses this
/// to keep its control HUD in a fixed strip that the per-frame blit doesn't
/// repaint (so the HUD updates in place instead of flickering under it).
pub fn present_surface_reserve(id: u32, sw: usize, sh: usize, buf: &[u32], reserve_bottom: u64) {
    // Hit-testing uses the same dimensions as the buffer (logical == pixel).
    present_surface_reserve_ex(id, sw, sh, sw, sh, buf, reserve_bottom);
}

/// Present a **pre-scaled** buffer while remembering `logical_sw×logical_sh`
/// for hit-testing. Package-UI builds a presentation-sized RGB buffer (geometry
/// nearest-upscaled + labels re-rasterized at that scale) but clicks must still
/// map into the agent's 256×192 coordinate space.
pub fn present_surface_reserve_ex(
    id: u32,
    logical_sw: usize,
    logical_sh: usize,
    buf_w: usize,
    buf_h: usize,
    buf: &[u32],
    reserve_bottom: u64,
) {
    // Hit map uses logical size; the frame we blit is `buf_w×buf_h`.
    remember_surf_dim(id, logical_sw, logical_sh);
    remember_surf_reserve(id, reserve_bottom);
    SCREEN.with(|slot| {
        // Open the surface tab only when it is not already among open tabs
        // (first present → focused action column). If the tab exists but that
        // column is not showing it, do **not** steal focus.
        let mode = RightMode::Surface(id);
        let found = slot.as_ref().and_then(|sc| sc.find_mode(mode));
        if found.is_none() {
            open_view_slot(slot, mode);
        }
        let Some(sc) = slot else { return };
        let Some((pi, ti)) = sc.find_mode(mode) else {
            return;
        };
        // Not the active tab of its column → skip FB blit (backing already updated).
        if sc.actions[pi].active != ti {
            return;
        }
        if logical_sw == 0 || logical_sh == 0 || buf_w == 0 || buf_h == 0 {
            return;
        }
        sc.cursor_restore();
        sc.cur_vis = false;
        let (px, py) = (sc.actions[pi].pane.ix, sc.actions[pi].pane.iy);
        let (pw, ph_full) = (
            sc.actions[pi].pane.cols * sc.actions[pi].pane.cw,
            sc.actions[pi].pane.rows * sc.actions[pi].pane.ch,
        );
        // Usable frame height excludes the reserved HUD strip at the bottom.
        let ph = ph_full.saturating_sub(reserve_bottom);
        if pw == 0 || ph == 0 || buf.len() < buf_w * buf_h {
            sc.cursor_overlay();
            return;
        }
        // Destination frame follows the **logical** aspect-fit (matches
        // surface_hit). When the buffer is already that size, the sample loop
        // is 1:1; otherwise nearest-neighbour from buf into the frame.
        let (dw, dh) = present_fit(logical_sw as u64, logical_sh as u64, pw, ph);
        let ox = px + (pw.saturating_sub(dw)) / 2;
        let oy = py + (ph.saturating_sub(dh)) / 2;
        // **No full-pane clear.** Clearing the whole surface with fill_rect
        // (then painting the frame) flashed background on the single-buffered
        // FB for tens of ms every present — visible as a once-per-second
        // (or every-frame) flicker. Only paint letterbox *margins*; the frame
        // blit overwrites the content rectangle in place.
        let bg = sc.actions[pi].pane.bg;
        if oy > py {
            sc.fill_rect(px, py, pw, oy - py, bg); // top bar
        }
        let bottom = oy + dh;
        if bottom < py + ph {
            sc.fill_rect(px, bottom, pw, (py + ph) - bottom, bg); // bottom bar
        }
        if ox > px {
            sc.fill_rect(px, oy, ox - px, dh, bg); // left bar
        }
        let right = ox + dw;
        if right < px + pw {
            sc.fill_rect(right, oy, (px + pw) - right, dh, bg); // right bar
        }
        // Build one destination row at a time and blit — sequential stores beat
        // hundreds of thousands of put_pixel calls for video.
        let mut row = alloc::vec![0u32; dw as usize];
        for dy in 0..dh {
            let sy = (dy * buf_h as u64 / dh) as usize;
            let srow = sy * buf_w;
            for dx in 0..dw as usize {
                let sx = (dx as u64 * buf_w as u64 / dw) as usize;
                row[dx] = buf[srow + sx];
            }
            sc.blit_rgb32_row(ox, oy + dy, &row);
        }
        sc.cursor_overlay();
    });
}

/// The action pane's interior size in pixels, if the split is open — the
/// image viewer sizes its downscale to this before presenting.
pub fn action_dims_px() -> Option<(u64, u64)> {
    SCREEN.with(|slot| {
        slot.as_ref().and_then(|sc| {
            (sc.right() != RightMode::Closed).then(|| (sc.logs().cols * sc.logs().cw, sc.logs().rows * sc.logs().ch))
        })
    })
}

/// Interior pixel dims of the column that owns surface `id`, falling back to the
/// focused column when the surface has no tab yet (first present).
///
/// A surface's content must be laid out for the column it is actually blitted
/// into — using the focused column's width would render a browser page or a
/// package-UI canvas at the wrong scale as soon as its tab lived elsewhere.
pub fn surface_dims_px(id: u32) -> Option<(u64, u64)> {
    SCREEN.with(|slot| {
        let sc = slot.as_ref()?;
        let i = sc
            .find_mode(RightMode::Surface(id))
            .map(|(pi, _)| pi)
            .unwrap_or(sc.focused_action);
        let p = &sc.actions.get(i)?.pane;
        (p.w > 0).then(|| (p.cols * p.cw, p.rows * p.ch))
    })
}

/// The action pane's interior background colour, packed `0x00RRGGBB` to match
/// the pixel buffer [`present_surface`] blits — the image viewer letterboxes
/// with this so the padding around a zoomed/rotated image matches the pane.
pub fn pane_bg() -> Option<u32> {
    SCREEN.with(|slot| {
        slot.as_ref().map(|sc| {
            let (r, g, b) = sc.logs().bg;
            ((r as u32) << 16) | ((g as u32) << 8) | b as u32
        })
    })
}

fn remember_surf_dim(id: u32, sw: usize, sh: usize) {
    LAST_SURF_DIM.with(|m| {
        m.insert(id, (sw, sh));
    });
}

fn remember_surf_reserve(id: u32, reserve_bottom: u64) {
    LAST_SURF_RESERVE.with(|m| {
        m.insert(id, reserve_bottom);
    });
}

/// Map a screen click into the active surface's logical coordinates,
/// accounting for letterboxing. Uses the last presented size for that surface
/// id (defaults to 256×192 for Synapse UI boards). `None` if the action pane
/// is not showing a surface or the click is outside the painted frame.
pub fn surface_hit(mx: u64, my: u64) -> Option<(u32, u16, u16)> {
    SCREEN.with(|slot| {
        let sc = slot.as_ref()?;
        // Prefer the action column under the pointer; else the focused column.
        let pi = sc
            .actions
            .iter()
            .position(|a| {
                a.pane.w > 0
                    && mx >= a.pane.x
                    && mx < a.pane.x + a.pane.w
                    && my >= a.pane.y
                    && my < a.pane.y + a.pane.h
            })
            .unwrap_or(sc.focused_action.min(sc.actions.len().saturating_sub(1)));
        let a = sc.actions.get(pi)?;
        let id = match a.right() {
            RightMode::Surface(id) => id,
            _ => return None,
        };
        let (px, py) = (a.pane.ix, a.pane.iy);
        let (pw, ph_full) = (a.pane.cols * a.pane.cw, a.pane.rows * a.pane.ch);
        // Exclude the HUD strip (video / browser) so clicks there are not mapped
        // into content coordinates — same usable height as present_surface_reserve.
        let reserve = LAST_SURF_RESERVE.with(|m| m.get(&id).copied().unwrap_or(0));
        let ph = ph_full.saturating_sub(reserve);
        if pw == 0 || ph == 0 || mx < px || my < py || mx >= px + pw || my >= py + ph {
            return None;
        }
        let (sw, sh) = LAST_SURF_DIM.with(|m| {
            m.get(&id)
                .copied()
                .unwrap_or((256, 192))
        });
        let (sw, sh) = (sw as u64, sh as u64);
        if sw == 0 || sh == 0 {
            return None;
        }
        // Same fit as present_surface_reserve (integer upscale when growing).
        let (dw, dh) = present_fit(sw, sh, pw, ph);
        let ox = px + (pw.saturating_sub(dw)) / 2;
        let oy = py + (ph.saturating_sub(dh)) / 2;
        if mx < ox || my < oy || mx >= ox + dw || my >= oy + dh {
            return None;
        }
        let sx = ((mx - ox) * sw / dw) as u16;
        let sy = ((my - oy) * sh / dh) as u16;
        Some((id, sx, sy))
    })
}

#[cfg(test)]
mod present_fit_tests {
    use super::present_fit;

    #[test_case]
    fn present_fit_integer_upscales_package_ui() {
        // 256×192 into a large pane: free scale is non-integer (~4.6×); we
        // want exact integer blocks so text stays crisp.
        let (dw, dh) = present_fit(256, 192, 1200, 900);
        assert_eq!(dw % 256, 0, "width is integer multiple of source");
        assert_eq!(dh % 192, 0, "height is integer multiple of source");
        assert_eq!(dw / 256, dh / 192, "uniform scale");
        assert!(dw / 256 >= 4, "at least 4× on a 1200×900 pane");
        assert!(dw <= 1200 && dh <= 900);
    }

    #[test_case]
    fn present_fit_one_to_one_when_pane_equals_source() {
        assert_eq!(present_fit(256, 192, 256, 192), (256, 192));
    }

    #[test_case]
    fn present_fit_downscales_large_video() {
        // 1920×1080 into 640×360 — free aspect-fit, not integer upscale.
        let (dw, dh) = present_fit(1920, 1080, 640, 360);
        assert!(dw <= 640 && dh <= 360);
        // Full width or height is used.
        assert!(dw == 640 || dh == 360);
        // Aspect preserved: 16:9.
        assert_eq!(dw * 9, dh * 16);
    }

    #[test_case]
    fn present_fit_zero_dims_is_empty() {
        assert_eq!(present_fit(0, 192, 100, 100), (0, 0));
        assert_eq!(present_fit(256, 192, 0, 100), (0, 0));
    }
}
