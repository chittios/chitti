//! Raster primitives: pixel/rect/disc fills, the wallpaper and surface
//! compositing paths, and the sub-pixel coverage helper every curve uses.

use super::*;
use crate::screenshot::Extent;

/// Anti-aliasing sub-sample rate per axis (`AA_SS`×`AA_SS` samples per pixel →
/// `AA_SS²`+1 coverage levels). 4 gives 16 levels — smooth curves at negligible
/// cost, since only edge pixels of a shape actually vary.
pub(super) const AA_SS: i64 = 4;

/// Sub-sampled coverage of a pixel at integer offset `(dx, dy)` from a shape's
/// origin, for anti-aliasing. `inside(fx, fy)` tests a sub-pixel point given in
/// a **2·SS-scaled** coordinate grid (so sub-sample centres land on odd
/// integers, exactly between grid lines — no rounding bias). Returns an alpha
/// 0..=255 = fraction of the `AA_SS²` sub-samples inside the shape. Integer-only
/// (no `sqrt`/float), so it works below the FPU-less boot window too.
pub(super) fn aa_coverage<F: Fn(i64, i64) -> bool>(dx: i64, dy: i64, inside: F) -> u32 {
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

/// Claim `rect` for a shadow: true when it differs from the last one painted,
/// false when a repaint would darken pixels that already carry it. See
/// [`LAST_SHADOW`].
fn shadow_claim(x: u64, y: u64, w: u64, h: u64) -> bool {
    LAST_SHADOW.with(|last| {
        if *last == Some((x, y, w, h)) {
            return false;
        }
        *last = Some((x, y, w, h));
        true
    })
}

/// Forget the last shadow, so the next [`Screen::drop_shadow`] for the same box
/// paints again. Called from [`Screen::redraw`], which has just repainted
/// everything a shadow could be lying on.
pub(super) fn shadow_forget() {
    LAST_SHADOW.with(|last| *last = None);
}

impl Screen {
    /// Byte offset into the framebuffer for **logical** pixel `(x, y)`.
    ///
    /// The single place the logical desktop is translated into physical pixels.
    /// `width`/`height` are the *logical* desktop (what every layout is computed
    /// against) and `origin_x`/`origin_y` place it inside the real framebuffer, so
    /// a smaller-than-native resolution is a centred, letterboxed viewport that
    /// still renders 1:1 — glyphs are rasterised at physical pixels, nothing is
    /// scaled, so text stays crisp. When the desktop is native both origins are 0
    /// and this is the identity, i.e. the default path is unchanged.
    #[inline]
    pub(super) fn fb_offset(&self, x: u64, y: u64) -> u64 {
        (y + self.origin_y) * self.pitch + (x + self.origin_x) * self.bpp_bytes
    }

    /// Fill a rectangle in **physical** framebuffer coordinates, bypassing the
    /// logical viewport. Only the letterbox uses this — everything else must go
    /// through the logical path so it lands inside the desktop.
    pub(super) fn fill_phys(&self, x: u64, y: u64, w: u64, h: u64, c: Rgb) {
        let value = self.pack_rgb(c);
        if x >= self.fb_w {
            return;
        }
        let n = w.min(self.fb_w - x);
        for yy in y..(y + h).min(self.fb_h) {
            let offset = yy * self.pitch + x * self.bpp_bytes;
            // SAFETY: clipped to the physical framebuffer, which is kernel-owned
            // MMIO for its full `fb_h * pitch` extent.
            unsafe {
                let ptr = (self.addr as *mut u8).add(offset as usize);
                if self.bpp_bytes == 4 {
                    let dst = ptr as *mut u32;
                    for i in 0..n {
                        dst.add(i as usize).write_volatile(value);
                    }
                } else {
                    for i in 0..n {
                        let p = ptr.add((i * self.bpp_bytes) as usize);
                        for b in 0..self.bpp_bytes {
                            p.add(b as usize).write_volatile((value >> (b * 8)) as u8);
                        }
                    }
                }
            }
        }
    }

    /// Read `row.len()` pixels starting at **physical** `(x, y)` back as packed
    /// `0x00RRGGBB` — the read mirror of [`Self::fill_phys`], and the primitive
    /// behind `/screenshot`.
    ///
    /// Physical rather than logical because a screenshot of a letterboxed
    /// desktop legitimately means either thing, and the whole panel is the
    /// larger of the two: a logical capture is this reader offset by
    /// `origin_x`/`origin_y`, which is what [`Self::read_rgb32_row`] does.
    ///
    /// Reads short (leaving the tail of `row` untouched) rather than clamping
    /// silently to a different width — a caller that asked past the panel has a
    /// geometry bug and should see a black tail, not a narrower image.
    pub(super) fn read_phys_rgb32_row(&self, x: u64, y: u64, row: &mut [u32]) {
        if y >= self.fb_h || x >= self.fb_w || row.is_empty() {
            return;
        }
        let n = row.len().min((self.fb_w - x) as usize);
        let offset = y * self.pitch + x * self.bpp_bytes;
        // SAFETY: `n` is clipped to the physical scanline and `y` to the panel
        // height, so the whole span lies inside the `fb_h * pitch` MMIO region
        // the kernel owns. Read-only.
        unsafe {
            let ptr = (self.addr as *const u8).add(offset as usize);
            if self.bpp_bytes == 4 && self.r_shift == 16 && self.g_shift == 8 && self.b_shift == 0 {
                // Native XRGB8888: our packing already matches, so mask the
                // unused high byte and store.
                let src = ptr as *const u32;
                for i in 0..n {
                    row[i] = src.add(i).read_volatile() & 0x00ff_ffff;
                }
            } else {
                for i in 0..n {
                    let p = ptr.add(i * self.bpp_bytes as usize);
                    let mut val: u32 = 0;
                    for b in 0..self.bpp_bytes {
                        val |= (p.add(b as usize).read_volatile() as u32) << (b * 8);
                    }
                    row[i] = (((val >> self.r_shift) & 0xff) << 16)
                        | (((val >> self.g_shift) & 0xff) << 8)
                        | ((val >> self.b_shift) & 0xff);
                }
            }
        }
    }

    /// Read a row of the **logical** desktop back as packed `0x00RRGGBB` — the
    /// read mirror of [`Self::blit_rgb32_row`].
    pub(super) fn read_rgb32_row(&self, x: u64, y: u64, row: &mut [u32]) {
        if y >= self.height || x >= self.width || row.is_empty() {
            return;
        }
        let n = row.len().min((self.width - x) as usize);
        // The one translation, same as every other framebuffer access.
        let off = self.fb_offset(x, y);
        let (px, py) = (off % self.pitch / self.bpp_bytes, off / self.pitch);
        self.read_phys_rgb32_row(px, py, &mut row[..n]);
    }

    /// Paint the dead space around a smaller-than-native desktop.
    ///
    /// A no-op at native resolution (the common case), so the default boot path
    /// touches nothing extra. Painted black rather than the theme background: it
    /// is outside the desktop, so it should read as "no screen here" rather than
    /// as an oversized margin.
    pub(super) fn paint_letterbox(&self) {
        if self.origin_x == 0
            && self.origin_y == 0
            && self.width == self.fb_w
            && self.height == self.fb_h
        {
            return;
        }
        let black = (0, 0, 0);
        let (ox, oy) = (self.origin_x, self.origin_y);
        self.fill_phys(0, 0, self.fb_w, oy, black); // above
        let below = oy + self.height;
        self.fill_phys(0, below, self.fb_w, self.fb_h.saturating_sub(below), black);
        self.fill_phys(0, oy, ox, self.height, black); // left
        let right = ox + self.width;
        self.fill_phys(right, oy, self.fb_w.saturating_sub(right), self.height, black);
    }

    pub(super) fn put_pixel(&self, x: u64, y: u64, c: Rgb) {
        if x >= self.width || y >= self.height {
            return;
        }
        // NB: damage is NOT tracked here. A redraw is millions of put_pixel calls
        // and a per-pixel union would cost more than the flush it feeds. The coarse
        // painters (`fill_rect`, `blit_rgb32_row`, `redraw`) report damage instead,
        // and they are what every glyph and frame ultimately goes through.
        let offset = self.fb_offset(x, y);
        let value: u32 =
            ((c.0 as u32) << self.r_shift) | ((c.1 as u32) << self.g_shift) | ((c.2 as u32) << self.b_shift);
        // SAFETY: `offset` is bounds-checked against the reported geometry; the
        // framebuffer is a valid, kernel-owned MMIO region.
        unsafe {
            let ptr = (self.addr as *mut u8).add(offset as usize);
            // Fast path: 32-bit linear FB (virtio / GOP / ramfb all are).
            if self.bpp_bytes == 4 {
                (ptr as *mut u32).write_volatile(value);
            } else {
                for i in 0..self.bpp_bytes {
                    ptr.add(i as usize).write_volatile((value >> (i * 8)) as u8);
                }
            }
        }
    }

    /// Pack an RGB triple into a framebuffer native pixel word.
    #[inline]
    fn pack_rgb(&self, c: Rgb) -> u32 {
        ((c.0 as u32) << self.r_shift) | ((c.1 as u32) << self.g_shift) | ((c.2 as u32) << self.b_shift)
    }

    /// Blit a row of packed `0x00RRGGBB` pixels into the FB at `(x,y)`.
    /// Much faster than per-pixel put for video frames (one bounds check +
    /// sequential stores).
    pub(super) fn blit_rgb32_row(&self, x: u64, y: u64, row: &[u32]) {
        if y >= self.height || x >= self.width || row.is_empty() {
            return;
        }
        crate::kms::damage(
            (x + self.origin_x) as u32,
            (y + self.origin_y) as u32,
            row.len() as u32,
            1,
        );
        let n = row.len().min((self.width - x) as usize);
        let offset = self.fb_offset(x, y);
        // SAFETY: n is clipped to the scanline; FB is kernel-owned MMIO.
        unsafe {
            let mut ptr = (self.addr as *mut u8).add(offset as usize);
            if self.bpp_bytes == 4
                && self.r_shift == 16
                && self.g_shift == 8
                && self.b_shift == 0
            {
                // Native XRGB8888 — store as-is (our RGB packs match).
                let dst = ptr as *mut u32;
                for i in 0..n {
                    dst.add(i).write_volatile(row[i]);
                }
            } else if self.bpp_bytes == 4 {
                let dst = ptr as *mut u32;
                for i in 0..n {
                    let c = row[i];
                    let rgb = (
                        ((c >> 16) & 0xff) as u8,
                        ((c >> 8) & 0xff) as u8,
                        (c & 0xff) as u8,
                    );
                    dst.add(i).write_volatile(self.pack_rgb(rgb));
                }
            } else {
                for i in 0..n {
                    let c = row[i];
                    let rgb = (
                        ((c >> 16) & 0xff) as u8,
                        ((c >> 8) & 0xff) as u8,
                        (c & 0xff) as u8,
                    );
                    let value = self.pack_rgb(rgb);
                    for b in 0..self.bpp_bytes {
                        ptr.add(b as usize).write_volatile((value >> (b * 8)) as u8);
                    }
                    ptr = ptr.add(self.bpp_bytes as usize);
                }
            }
        }
    }

    pub(super) fn fill_rect(&self, x: u64, y: u64, w: u64, h: u64, c: Rgb) {
        // Report to KMS in *physical* coordinates: a driver's scanout is the whole
        // framebuffer, not the logical desktop, so the viewport origin must be added.
        crate::kms::damage(
            (x + self.origin_x) as u32,
            (y + self.origin_y) as u32,
            w as u32,
            h as u32,
        );
        if w == 0 || h == 0 {
            return;
        }
        // Fast path: pack once and blast whole scanlines (critical for video
        // letterbox / status bars — per-pixel put_pixel was multi-ms flashes).
        let packed = ((c.0 as u32) << 16) | ((c.1 as u32) << 8) | c.2 as u32;
        if self.bpp_bytes == 4 && self.r_shift == 16 && self.g_shift == 8 && self.b_shift == 0 {
            let mut row = alloc::vec![packed; w as usize];
            for dy in 0..h {
                self.blit_rgb32_row(x, y + dy, &row);
            }
            // silence unused mut if n=0 — row is mut for potential reuse
            let _ = &mut row;
            return;
        }
        for dy in 0..h {
            for dx in 0..w {
                self.put_pixel(x + dx, y + dy, c);
            }
        }
    }

    /// Build [`Self::wallpaper`] (scaled to the full screen) from a spec:
    /// `""` → none (solid desktop); `"gradient:#aabbcc,#112233"` → a generated
    /// vertical gradient; otherwise a path to an image in the synapse store,
    /// decoded and stretched to fill. Also sets [`Self::opacity`]. Called from
    /// `build`/`relayout` so it's recomputed once per layout change, not per
    /// redraw.
    pub(super) fn set_wallpaper(&mut self, spec: &str, opacity: u8) {
        self.opacity = opacity;
        let (w, h) = (self.width as usize, self.height as usize);
        if w == 0 || h == 0 {
            self.wallpaper = None;
            return;
        }
        if spec.is_empty() {
            self.wallpaper = None;
            return;
        }
        if let Some(rest) = spec.strip_prefix("gradient:") {
            // Two `#rrggbb` stops, top → bottom.
            let mut it = rest.split(',');
            let a = parse_hex(it.next().unwrap_or("").trim(), self.theme.screen_bg);
            let b = parse_hex(it.next().unwrap_or("").trim(), a);
            let mut buf = alloc::vec![0u32; w * h];
            for y in 0..h {
                // t in 0..=255 down the screen.
                let t = if h > 1 { (y * 255 / (h - 1)) as u32 } else { 0 };
                let mix = |ca: u8, cb: u8| ((ca as u32 * (255 - t) + cb as u32 * t) / 255) & 0xff;
                let px = (mix(a.0, b.0) << 16) | (mix(a.1, b.1) << 8) | mix(a.2, b.2);
                for x in 0..w {
                    buf[y * w + x] = px;
                }
            }
            self.wallpaper = Some(buf);
            return;
        }
        // Image path: read from the store, decode, cover-scale to fill the
        // screen (preserve aspect, centre-crop — no stretch/distortion).
        self.wallpaper = crate::synapse::fs::read(spec)
            .and_then(|bytes| crate::image::decode(&bytes).ok())
            .map(|img| crate::image::cover(&img, w, h))
            .map(|img| img.pixels);
    }

    /// Paint a **desktop/gutter** region: the wallpaper (if any) shown directly,
    /// else a solid `fallback` fill.
    pub(super) fn paint_wallpaper(&self, x: u64, y: u64, w: u64, h: u64, fallback: Rgb) {
        let Some(wp) = &self.wallpaper else {
            self.fill_rect(x, y, w, h, fallback);
            return;
        };
        if w == 0 || h == 0 {
            return;
        }
        for dy in 0..h {
            let sy = y + dy;
            if sy >= self.height {
                break;
            }
            let base = (sy * self.width + x) as usize;
            let n = w.min(self.width.saturating_sub(x)) as usize;
            if x < self.width && base + n <= wp.len() {
                self.blit_rgb32_row(x, sy, &wp[base..base + n]);
            }
        }
    }

    /// Paint a **window surface** region: `color` blended over the wallpaper at
    /// [`Self::opacity`] (255 = opaque = plain `color`), else a solid `color`
    /// fill when there's no wallpaper. One blended row is built and blitted per
    /// scanline (no per-pixel readback).
    pub(super) fn paint_surface(&self, x: u64, y: u64, w: u64, h: u64, color: Rgb) {
        let Some(wp) = &self.wallpaper else {
            self.fill_rect(x, y, w, h, color);
            return;
        };
        if w == 0 || h == 0 {
            return;
        }
        let op = self.opacity as u32;
        if op >= 255 {
            self.fill_rect(x, y, w, h, color);
            return;
        }
        let inv = 255 - op;
        let (cr, cg, cb) = (color.0 as u32, color.1 as u32, color.2 as u32);
        let mut row = alloc::vec![0u32; w as usize];
        for dy in 0..h {
            let sy = y + dy;
            if sy >= self.height {
                break;
            }
            for dx in 0..w {
                let sx = x + dx;
                let wpix = if sx < self.width {
                    wp[(sy * self.width + sx) as usize]
                } else {
                    0
                };
                let r = ((((wpix >> 16) & 0xff) * inv + cr * op) / 255) & 0xff;
                let g = ((((wpix >> 8) & 0xff) * inv + cg * op) / 255) & 0xff;
                let b = (((wpix & 0xff) * inv + cb * op) / 255) & 0xff;
                row[dx as usize] = (r << 16) | (g << 8) | b;
            }
            self.blit_rgb32_row(x, sy, &row);
        }
    }

    /// Fill a small region's background honouring a translucent wallpaper — the
    /// cell-level analog of [`Self::paint_surface`], but with no per-call heap
    /// allocation (cells are tiny). Text-grid cells use this so the see-through
    /// desktop shows behind **every** cell, not only behind painted glyphs
    /// (`blit_glyph` already blends via [`Self::bg_at`]); an opaque wallpaper /
    /// no-wallpaper falls back to the solid [`Self::fill_rect`] fast path.
    pub(super) fn fill_cell_bg(&self, x: u64, y: u64, w: u64, h: u64, bg: Rgb) {
        if self.wallpaper.is_none() || self.opacity >= 255 {
            self.fill_rect(x, y, w, h, bg);
            return;
        }
        for dy in 0..h {
            let sy = y + dy;
            if sy >= self.height {
                break;
            }
            for dx in 0..w {
                self.put_pixel(x + dx, sy, self.bg_at(x + dx, sy, bg));
            }
        }
    }

    /// Read a framebuffer pixel back as `Rgb` (inverse of `put_pixel`), for
    /// saving the background under the mouse cursor.
    pub(super) fn get_pixel(&self, x: u64, y: u64) -> Rgb {
        if x >= self.width || y >= self.height {
            return (0, 0, 0);
        }
        let offset = self.fb_offset(x, y);
        // SAFETY: bounds-checked offset into the kernel-owned framebuffer.
        let mut val: u32 = 0;
        unsafe {
            let ptr = (self.addr as *const u8).add(offset as usize);
            for i in 0..self.bpp_bytes {
                val |= (ptr.add(i as usize).read_volatile() as u32) << (i * 8);
            }
        }
        (
            ((val >> self.r_shift) & 0xff) as u8,
            ((val >> self.g_shift) & 0xff) as u8,
            ((val >> self.b_shift) & 0xff) as u8,
        )
    }

    /// Alpha-blend `c` over the existing framebuffer pixel at `(x,y)` with
    /// coverage `a` (0 = transparent … 255 = opaque). A read-modify-write, so
    /// only worth it for the fractional-coverage *edge* pixels of a shape — the
    /// interior should use the plain [`put_pixel`] fast path. This is the
    /// primitive behind anti-aliased curves (discs, the logo arc).
    pub(super) fn blend_pixel(&self, x: u64, y: u64, c: Rgb, a: u32) {
        if a == 0 || x >= self.width || y >= self.height {
            return;
        }
        if a >= 255 {
            self.put_pixel(x, y, c);
            return;
        }
        let bg = self.get_pixel(x, y);
        let mix = |b: u8, f: u8| (((b as u32) * (255 - a) + (f as u32) * a) / 255) as u8;
        self.put_pixel(x, y, (mix(bg.0, c.0), mix(bg.1, c.1), mix(bg.2, c.2)));
    }

    /// A soft drop shadow for the box at `(x,y,w,h)` — the elevation cue under
    /// every pane, modal and dropdown. Drawn *before* the box, which then paints
    /// over the part that falls inside it. Darkens whatever is behind (the
    /// screen background for a pane, the panes for a modal) toward black, so it
    /// works over a wallpaper too.
    ///
    /// The shadow is the box pushed **down** by the offset and blurred, with a
    /// quadratic per-pixel falloff — the geometry and the alpha are pure and
    /// unit-tested in [`crate::panes_layout`], since nothing written in this
    /// module is compiled by the test build. What it replaced was two
    /// hard-edged rectangles at fixed darkness, which read as a black border
    /// rather than as depth: two visible steps and then a cliff to nothing.
    ///
    /// Nothing is painted **above** the box top: light comes from above, and
    /// clipping there also keeps the shadow off whatever sits over the box (a
    /// dropdown's status bar repaints on its own schedule, so a shadow reaching
    /// into it would be half-erased).
    pub(super) fn drop_shadow(&self, x: u64, y: u64, w: u64, h: u64) {
        if w == 0 || h == 0 || !shadow_claim(x, y, w, h) {
            return;
        }
        use crate::panes_layout::{shadow_alpha, shadow_falloff, shadow_geom, span_dist};
        let g = shadow_geom(self.scale);
        // The shadow's own rectangle: the box, dropped by the offset.
        let (sx0, sx1) = (x, x + w);
        let (sy0, sy1) = (y + g.offset, y + g.offset + h);
        let rx0 = x.saturating_sub(g.blur);
        let rx1 = (sx1 + g.blur).min(self.width);
        let ry1 = (sy1 + g.blur).min(self.height);
        for py in y..ry1 {
            let dy = span_dist(py, sy0, sy1);
            if shadow_falloff(dy, g.blur) == 0 {
                continue;
            }
            // The box is opaque and painted on top, so on the rows it covers
            // only the two side strips are worth touching — walking its whole
            // interior would be a pane's area in read-modify-write pixels.
            let spans = if py < y + h {
                [(rx0, x.min(rx1)), (sx1.min(rx1), rx1)]
            } else {
                [(rx0, rx1), (0, 0)]
            };
            for (px0, px1) in spans {
                for px in px0..px1 {
                    let dx = span_dist(px, sx0, sx1);
                    self.blend_pixel(px, py, (0, 0, 0), shadow_alpha(dx, dy, g.blur, g.peak));
                }
            }
        }
    }

    /// Blend a colour toward white by `t` (0 = unchanged, 1 = white) — the
    /// crisp 1px inner highlight just inside an active border.
    pub(super) fn lighten(&self, c: Rgb, t: f32) -> Rgb {
        let d = |x: u8| (x as f32 + (255.0 - x as f32) * t) as u8;
        (d(c.0), d(c.1), d(c.2))
    }

    /// Blend `a` toward `b` by `t` (0 = `a`, 1 = `b`) — e.g. the accent-tinged
    /// composer focus glow against the chat background.
    pub(super) fn mix(&self, a: Rgb, b: Rgb, t: f32) -> Rgb {
        let d = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t) as u8;
        (d(a.0, b.0), d(a.1, b.1), d(a.2, b.2))
    }

    /// Draw a `t`-thick rectangle outline (the pane border).
    pub(super) fn rect_outline(&self, x: u64, y: u64, w: u64, h: u64, t: u64, c: Rgb) {
        self.fill_rect(x, y, w, t, c); // top
        self.fill_rect(x, y + h - t, w, t, c); // bottom
        self.fill_rect(x, y, t, h, c); // left
        self.fill_rect(x + w - t, y, t, h, c); // right
    }

    /// Alpha-blend one printable glyph into its cell at `(px,py)`. Renders via
    /// the **TTF UI face** (fontdue, crisp at the display resolution — see
    /// `font_ttf::blit_ui_cell`, face chosen from `ui.json`), falling back to the
    /// scaled bitmap atlas ([`GLYPHS`]) if no TTF face is available. Non-printable
    /// bytes render as a blank cell.
    /// Effective background colour at pixel `(x,y)`: with a translucent
    /// wallpaper, `base` blended over the wallpaper at [`Self::opacity`]; else
    /// `base` unchanged (the fast common case — no wallpaper or opaque).
    #[inline]
    pub(super) fn bg_at(&self, x: u64, y: u64, base: Rgb) -> Rgb {
        match &self.wallpaper {
            Some(wp) if self.opacity < 255 && x < self.width && y < self.height => {
                let wpix = wp[(y * self.width + x) as usize];
                let op = self.opacity as u32;
                let inv = 255 - op;
                let bl = |w: u32, c: u8| (((w * inv + c as u32 * op) / 255) & 0xff) as u8;
                (
                    bl((wpix >> 16) & 0xff, base.0),
                    bl((wpix >> 8) & 0xff, base.1),
                    bl(wpix & 0xff, base.2),
                )
            }
            _ => base,
        }
    }

    /// Draw a soft-rounded rectangle outline (bordered input chrome).
    pub(super) fn rounded_outline(&self, x: u64, y: u64, w: u64, h: u64, r: u64, c: Rgb) {
        if w < 2 * r || h < 2 * r {
            self.rect_outline(x, y, w, h, 1, c);
            return;
        }
        // Straight edges.
        self.fill_rect(x + r, y, w - 2 * r, 1, c);
        self.fill_rect(x + r, y + h - 1, w - 2 * r, 1, c);
        self.fill_rect(x, y + r, 1, h - 2 * r, c);
        self.fill_rect(x + w - 1, y + r, 1, h - 2 * r, c);
        // Quarter-circle corners (1px arc).
        let r = r as i64;
        for &(cx, cy, sx, sy) in &[
            (x + r as u64, y + r as u64, -1i64, -1i64),
            (x + w - 1 - r as u64, y + r as u64, 1i64, -1i64),
            (x + r as u64, y + h - 1 - r as u64, -1i64, 1i64),
            (x + w - 1 - r as u64, y + h - 1 - r as u64, 1i64, 1i64),
        ] {
            for dy in 0..=r {
                for dx in 0..=r {
                    let d2 = dx * dx + dy * dy;
                    // Outer rim only (~1 px thick).
                    if d2 <= r * r && d2 >= (r - 1) * (r - 1) {
                        self.put_pixel((cx as i64 + sx * dx) as u64, (cy as i64 + sy * dy) as u64, c);
                    }
                }
            }
        }
    }

    /// Fill an integer-centred disc of radius `r` with `c` (round dots/caps).
    /// **Anti-aliased** filled disc: edge pixels get fractional coverage from
    /// [`AA_SS`]×[`AA_SS`] sub-sampling (integer-only, no `sqrt`), blended over
    /// the background so the rim is smooth rather than stair-stepped. Interior
    /// pixels take the opaque [`put_pixel`] fast path.
    pub(super) fn fill_disc(&self, cx: i64, cy: i64, r: i64, c: Rgb) {
        if r <= 0 {
            return;
        }
        // Radius in the 2·SS sub-pixel grid `aa_coverage` samples on.
        let rr = 2 * AA_SS * r;
        let r2 = rr * rr;
        let span = r + 1;
        for dy in -span..=span {
            for dx in -span..=span {
                let a = aa_coverage(dx, dy, |fx, fy| fx * fx + fy * fy <= r2);
                // Negative coords wrap to a huge u64 and are dropped by the
                // bounds checks in put_pixel / blend_pixel.
                self.blend_pixel((cx + dx) as u64, (cy + dy) as u64, c, a);
            }
        }
    }

    /// Draw the **Synapse-C** brand mark centred at `(cx, cy)` with ring radius
    /// `r`: a single open ring (the capability) in `arc_c` with round end-caps,
    /// and a filled node (the agent) at the centre in `node_c`. Pure integer math
    /// — a ring test plus one angular gap — so it scales from a status-bar glyph
    /// to a splash logo. Geometry mirrors the SVG in DESIGN.md: stroke width ≈
    /// `6/17·r`, a ~91° opening (dasharray 80/27) whose centre sits ~10° above the
    /// +x axis (the SVG's `rotate(35)` on a dash starting at 3 o'clock), and a
    /// centre node of radius ≈ `0.32·r` (SVG r 5.5 against ring r 17).
    pub(super) fn draw_logo(&self, cx: u64, cy: u64, r: u64, arc_c: Rgb, node_c: Rgb) {
        let (cx, cy, r) = (cx as i64, cy as i64, r as i64);
        let t = (r / 3).max(3); // stroke width, min 3 so a small mark still reads
        let half = t / 2;
        // Ring radii in the 2·SS sub-pixel grid (squared) for anti-aliasing.
        let inner = (2 * AA_SS * (r - half)).pow(2);
        let outer = (2 * AA_SS * (r + half)).pow(2);
        let span = r + half + 1;
        for dy in -span..=span {
            for dx in -span..=span {
                // Sub-sampled coverage of the open ring: inside the stroke band
                // and outside the ~91° opening. The gap test compares a pixel's
                // direction against the gap centre (984,-180) ≈ (cos-10.4°,
                // sin-10.4°): within ~45.4° (cos ≈ 0.701) is inside the gap.
                let a = aa_coverage(dx, dy, |fx, fy| {
                    let d2 = fx * fx + fy * fy;
                    if d2 < inner || d2 > outer {
                        return false;
                    }
                    let n = fx * 984 - fy * 180;
                    !(n > 0 && n * n > 701 * 701 * d2)
                });
                self.blend_pixel((cx + dx) as u64, (cy + dy) as u64, arc_c, a);
            }
        }
        // Round line-caps at the two arc ends (35° and 304.6° in screen coords),
        // in the ring colour, matching stroke-linecap="round".
        let cap = half.max(1);
        self.fill_disc(cx + r * 819 / 1000, cy + r * 574 / 1000, cap, arc_c); // 35°
        self.fill_disc(cx + r * 562 / 1000, cy - r * 827 / 1000, cap, arc_c); // 304.6°
        // The synapse node: a filled disc at the centre.
        let nr = (r * 32 / 100).max(2);
        self.fill_disc(cx, cy, nr, node_c);
    }
}

/// Read a rectangle of the live framebuffer back as `0x00RRGGBB` pixels —
/// the hardware half of `/screenshot`. All the geometry decisions are made by
/// [`crate::screenshot`], which is testable; this function only resolves an
/// [`Extent`] against the live pane tree and does the MMIO read.
///
/// The framebuffer is single-buffered, so what this reads is exactly what is on
/// the screen — including the mouse sprite, which is composited *into* the
/// framebuffer rather than overlaid at scanout. So unless `cursor` is set, the
/// sprite is lifted first ([`Screen::cursor_restore`]) and put back after; a
/// captured pointer in a bug report reads as a rendering artefact rather than
/// as a pointer.
pub fn capture(extent: Extent, cursor: bool) -> Result<(u32, u32, Vec<u32>), &'static str> {
    SCREEN.with(|slot| {
        let sc = slot.as_mut().ok_or("no framebuffer (serial-only boot)")?;

        // Resolve to a PHYSICAL rectangle. Logical extents add the letterbox
        // origin; `Panel` is already physical.
        let (px, py, w, h) = match extent {
            Extent::Panel => (0u64, 0u64, sc.fb_w, sc.fb_h),
            Extent::Desktop => (sc.origin_x, sc.origin_y, sc.width, sc.height),
            Extent::Region { x, y, w, h } => {
                let (cx, cy, cw, ch) = crate::screenshot::clamp_region(
                    sc.width as u32,
                    sc.height as u32,
                    x,
                    y,
                    w,
                    h,
                )
                .ok_or("region lies outside the desktop")?;
                (cx as u64 + sc.origin_x, cy as u64 + sc.origin_y, cw as u64, ch as u64)
            }
            Extent::Chat => {
                let p = &sc.chat;
                (p.x + sc.origin_x, p.y + sc.origin_y, p.w, p.h)
            }
            Extent::Pane(i) => {
                let slot = sc.actions.get(i).ok_or("no such action pane")?;
                let p = &slot.pane;
                if p.w == 0 || p.h == 0 {
                    return Err("that action pane is collapsed");
                }
                (p.x + sc.origin_x, p.y + sc.origin_y, p.w, p.h)
            }
            Extent::Surface(id) => {
                // Resolved through `mode_dims`, i.e. by *which pane holds that
                // surface's tab* — never the focused pane. A surface whose tab
                // is on pane 3 must not be photographed out of pane 1 because
                // the mouse happens to be there (the same rule the per-view
                // painters follow).
                let d = sc
                    .mode_dims(RightMode::Surface(id))
                    .ok_or("that surface is not on screen")?;
                // The interior, not the frame: an app's capture should be its
                // own pixels, not the pane chrome the compositor drew round it.
                (d.ix + sc.origin_x, d.iy + sc.origin_y, d.iw, d.ih)
            }
        };
        if w == 0 || h == 0 {
            return Err("nothing to capture (zero-sized surface)");
        }
        let (n_px, n_py) = (w as usize, h as usize);
        // Refuse before allocating, for the reason `image::png` does: the RGB
        // buffer alone is 4 bytes a pixel on a first-fit allocator.
        if n_px.saturating_mul(n_py) > crate::image::png::MAX_PIXELS {
            return Err("capture is larger than the encoder will accept");
        }

        let lift = !cursor && sc.cur_vis;
        if lift {
            sc.cursor_restore();
        }

        let mut pixels = Vec::new();
        // One allocation for the whole capture: growing a multi-megabyte Vec by
        // doubling is exactly the allocator churn the perf rules warn about.
        pixels.resize(n_px * n_py, 0u32);
        for row in 0..h {
            let off = (row as usize) * n_px;
            sc.read_phys_rgb32_row(px, py + row, &mut pixels[off..off + n_px]);
        }

        if lift {
            sc.cursor_overlay();
        }
        Ok((w as u32, h as u32, pixels))
    })
}

#[cfg(test)]
mod aa_tests {
    use super::{aa_coverage, AA_SS};

    #[test_case]
    fn disc_coverage_is_full_inside_zero_outside_partial_at_edge() {
        let r = 10i64;
        let r2 = (2 * AA_SS * r).pow(2);
        let inside = |fx: i64, fy: i64| fx * fx + fy * fy <= r2;
        // Dead centre: every sub-sample inside → fully opaque.
        assert_eq!(aa_coverage(0, 0, inside), 255);
        // Well outside the disc → nothing covered.
        assert_eq!(aa_coverage(r + 5, 0, inside), 0);
        // Right on the radius → a fractional coverage (this is the AA).
        let edge = aa_coverage(r, 0, inside);
        assert!(edge > 0 && edge < 255, "edge coverage should be partial, got {edge}");
    }

    #[test_case]
    fn coverage_is_monotonic_across_the_boundary() {
        let r = 20i64;
        let r2 = (2 * AA_SS * r).pow(2);
        let inside = |fx: i64, fy: i64| fx * fx + fy * fy <= r2;
        let a_in = aa_coverage(r - 2, 0, inside);
        let a_edge = aa_coverage(r, 0, inside);
        let a_out = aa_coverage(r + 2, 0, inside);
        assert!(
            a_in >= a_edge && a_edge >= a_out,
            "coverage must not rise moving outward: {a_in} {a_edge} {a_out}"
        );
    }
}
