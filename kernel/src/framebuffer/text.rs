//! Glyph rasterisation and string drawing.
//!
//! The two text-fitting helpers this module used to own, [`wrap`] and
//! [`pad_trunc`], now live in [`crate::textfit`] and are re-exported here so
//! every call site is unchanged. They moved because both counted **bytes** where
//! they meant **columns** (`wrap` also chunked long words with
//! `as_bytes().chunks(cols)` and `unwrap_or("")`-ed the invalid pieces, silently
//! deleting any word whose chunk boundary fell inside a character) — and because
//! this module is `#[cfg(not(test))]`, so a test asserting otherwise would never
//! have been compiled.

use super::*;

pub(super) use crate::textfit::{pad_trunc, wrap};

impl Screen {
    /// Cell size for one glyph. Font Awesome icons get a **square** cell of the
    /// body line height so they read at text size; mono cells are tall+narrow
    /// and squashing FA into `cw×ch` made agent-list / close marks tiny.
    pub(super) fn glyph_cell(&self, ch: char) -> (u64, u64) {
        if crate::icons::is_icon(ch) {
            let side = self.ch();
            (side, side)
        } else {
            (self.cw(), self.ch())
        }
    }

    /// Draw an icon **inside one mono cell**, vertically centred.
    ///
    /// [`Self::glyph_cell`] gives a Font Awesome glyph a square of the *line height*, which
    /// is right for a tab label or a status field but wrong for chrome sitting inline in a
    /// text grid: at `cw × 2 ≈ ch` it is two columns wide, so it reads as oversized **and**
    /// gets its right half erased the moment the streaming painter fills the next cell
    /// (`paint_chat_cell` writes one cell at a time as bytes arrive, and cannot know a
    /// neighbour overhangs into it). Sizing to the column width keeps the glyph round,
    /// unclipped, and the size of the text it sits beside.
    pub(super) fn blit_glyph_inline(&self, px: u64, py: u64, ch: char, fg: Rgb, bg: Rgb) {
        let (cw, chh) = (self.cw(), self.ch());
        let tinted = self.wallpaper.is_some() && self.opacity < 255;
        for gy in 0..chh {
            for gx in 0..cw {
                let cbg = self.bg_at(px + gx, py + gy, bg);
                self.put_pixel(px + gx, py + gy, cbg);
            }
        }
        let side = cw.min(chh);
        let oy = (chh.saturating_sub(side)) / 2;
        let mix = |b: u8, f: u8, a: u32| (((b as u32) * (255 - a) + (f as u32) * a) / 255) as u8;
        let _ = crate::font_ttf::blit_ui_cell(ch, side as usize, side as usize, |gx, gy, a| {
            let (x, y) = (px + gx as u64, py + oy + gy as u64);
            let b = if tinted { self.bg_at(x, y, bg) } else { bg };
            self.put_pixel(
                x,
                y,
                (mix(b.0, fg.0, a as u32), mix(b.1, fg.1, a as u32), mix(b.2, fg.2, a as u32)),
            );
        });
    }

    pub(super) fn blit_glyph(&self, px: u64, py: u64, ch: char, fg: Rgb, bg: Rgb) {
        let (cell_w, cell_h) = self.glyph_cell(ch);
        // Background fill first (both paths blend ink over it). With a
        // translucent wallpaper the cell bg is the wallpaper tinted by `bg` at
        // `opacity`, per pixel — so text sits over the see-through desktop too.
        let tinted = self.wallpaper.is_some() && self.opacity < 255;
        for gy in 0..cell_h {
            for gx in 0..cell_w {
                let cbg = self.bg_at(px + gx, py + gy, bg);
                self.put_pixel(px + gx, py + gy, cbg);
            }
        }
        // Empty / space / zero-width formatters: fill bg only (VS16, ZWJ, etc.
        // must not paint tofu boxes between emoji).
        if ch == '\0'
            || ch == ' '
            || ch == '\u{FE0F}'
            || ch == '\u{FE0E}'
            || ch == '\u{200D}'
            || ch == '\u{200C}'
        {
            return;
        }
        let mix = |b: u8, f: u8, a: u32| (((b as u32) * (255 - a) + (f as u32) * a) / 255) as u8;
        let ink = |s: &Self, x: u64, y: u64, a: u32| {
            let b = if tinted { s.bg_at(x, y, bg) } else { bg };
            (mix(b.0, fg.0, a), mix(b.1, fg.1, a), mix(b.2, fg.2, a))
        };
        // TTF path: rasterize the char (fontdue UI face + Noto fallback chain —
        // renders arbitrary Unicode; box-drawing/bullets included).
        let ttf_ok = crate::font_ttf::blit_ui_cell(ch, cell_w as usize, cell_h as usize, |gx, gy, a| {
            let (x, y) = (px + gx as u64, py + gy as u64);
            self.put_pixel(x, y, ink(self, x, y, a as u32));
        });
        if ttf_ok {
            return;
        }
        // Bitmap fallback: the 10×22 ASCII atlas. Non-ASCII with no TTF face
        // stays blank (bg already filled).
        let s = self.scale;
        let cp = ch as u32;
        if !(FIRST as u32..=LAST as u32).contains(&cp) {
            return;
        }
        let idx = (cp as u8 - FIRST) as usize;
        let g = &GLYPHS[idx];
        for gy in 0..CH {
            for gx in 0..CW {
                let a = g[gy * CW + gx] as u32;
                if a == 0 {
                    continue; // background already filled
                }
                let bx = px + gx as u64 * s;
                let by = py + gy as u64 * s;
                for sy in 0..s {
                    for sx in 0..s {
                        let (x, y) = (bx + sx, by + sy);
                        self.put_pixel(x, y, ink(self, x, y, a));
                    }
                }
            }
        }
    }

    /// Render `s` at pixel `(px,py)`. Body text advances one mono cell; Font
    /// Awesome icons advance a square of the body line height. Returns the x
    /// past the last glyph. Clips at `self.width`.
    pub(super) fn draw_str(&self, px: u64, py: u64, s: &str, fg: Rgb, bg: Rgb) -> u64 {
        let cw = self.cw();
        let ch = self.ch();
        // Upper-bound the damage box (icons are wider than one mono cell).
        let mut approx_w = 0u64;
        for c in s.chars() {
            approx_w += if crate::icons::is_icon(c) { ch } else { cw };
        }
        crate::kms::damage(
            (px + self.origin_x) as u32,
            (py + self.origin_y) as u32,
            approx_w as u32,
            ch as u32,
        );
        let mut x = px;
        for c in s.chars() {
            let advance = if crate::icons::is_icon(c) { ch } else { cw };
            if x + advance > self.width {
                break;
            }
            self.blit_glyph(x, py, c, fg, bg);
            x += advance;
        }
        x
    }

    // --- pane text -------------------------------------------------------

    /// Draw `s` within `[x, x+max_w)`, ellipsizing when it would overflow.
    /// Returns the x just past the last painted glyph.
    fn draw_str_fit(&self, x: u64, y: u64, s: &str, fg: Rgb, bg: Rgb, max_w: u64) -> u64 {
        let cols = (max_w / self.cw()) as usize;
        let t = crate::textsel::ellipsize(s, cols);
        self.draw_str(x, y, &t, fg, bg)
    }

    /// Shorthand: [`draw_str`] with an explicit background (the `/top` panel
    /// draws over the logs-pane background, not the screen background).
    pub(super) fn draw_str_bg(&self, x: u64, y: u64, s: &str, fg: Rgb, bg: Rgb) -> u64 {
        self.draw_str(x, y, s, fg, bg)
    }
}
