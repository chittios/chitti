//! Glyph rasterisation and string drawing, plus the two pure text-fitting
//! helpers ([`wrap`], [`pad_trunc`]).

use super::*;

/// Word-wrap `s` to `cols` columns (breaking long words), for modal messages.
pub(super) fn wrap(s: &str, cols: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut line = String::new();
    for word in s.split_whitespace() {
        if !line.is_empty() && line.len() + 1 + word.len() > cols {
            out.push(core::mem::take(&mut line));
        }
        for chunk in word.as_bytes().chunks(cols.max(1)) {
            let w = core::str::from_utf8(chunk).unwrap_or("");
            if line.len() + 1 + w.len() > cols && !line.is_empty() {
                out.push(core::mem::take(&mut line));
            }
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(w);
        }
    }
    if !line.is_empty() {
        out.push(line);
    }
    if out.is_empty() {
        out.push(String::new());
    }
    out
}

pub(super) fn pad_trunc(s: &str, cols: usize) -> alloc::string::String {
    let mut out: alloc::string::String = s.chars().take(cols).collect();
    while out.chars().count() < cols {
        out.push(' ');
    }
    out
}

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
