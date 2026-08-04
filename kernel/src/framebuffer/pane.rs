//! A text pane: its grid + scrollback ring, the ANSI/UTF-8 feed, and the
//! painters that put its rows, caret and frame on the screen.

use super::*;

impl Pane {
    /// Build a pane inside outer box `(x,y,w,h)` with scaled cell `(cw,ch)`,
    /// reserving a title header and `PAD` interior padding, then computing the
    /// cell grid.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(x: u64, y: u64, w: u64, h: u64, cw: u64, ch: u64, fg: Rgb, bg: Rgb, title: String, show_caret: bool) -> Pane {
        let header_h = BORDER + 4 + ch + 6; // top border, title text, separator gap
        let ix = x + BORDER + PAD;
        let iy = y + header_h;
        let iw = w.saturating_sub(2 * (BORDER + PAD)).max(cw);
        // bordered composer: box (vpad + 1 line + vpad + 2px border) + gap + hint line.
        // Reserve it so scrollback never paints under the input chrome.
        let has_composer = show_caret;
        let composer_block = if has_composer {
            COMPOSER_TOP_GAP + (COMPOSER_VPAD + ch + COMPOSER_VPAD + 2) + COMPOSER_HINT_GAP + ch
        } else {
            0
        };
        let ih = (y + h).saturating_sub(iy + BORDER + PAD + composer_block).max(ch);
        let cols = (iw / cw).max(1);
        let rows = (ih / ch).max(1);
        Pane {
            x,
            y,
            w,
            h,
            ix,
            iy,
            cw,
            ch,
            cols,
            rows,
            col: 0,
            row: 0,
            fg,
            default_fg: fg,
            bg,
            esc: EscState::Ground,
            csi: [0; 32],
            csi_len: 0,
            bold: false,
            title,
            show_caret,
            grid: alloc::vec![('\0', fg); (cols * rows) as usize],
            hist: VecDeque::new(),
            view: 0,
            sel: None,
            has_composer,
            folds: Vec::new(),
            user_band: Vec::new(),
            utf8: [0; 4],
            utf8_len: 0,
        }
    }

    /// Elevated bg for a user-prompt band line, else `None` (use pane bg).
    fn band_bg(&self, gi: usize) -> bool {
        self.user_band.binary_search(&gi).is_ok()
    }

    /// Write `byte` into the grid cell under the cursor (0 erases).
    fn set_cell(&mut self, ch: char) {
        let idx = (self.row * self.cols + self.col) as usize;
        if let Some(c) = self.grid.get_mut(idx) {
            *c = (ch, self.fg);
        }
    }

    /// Feed one incoming byte through the incremental UTF-8 decoder: returns the
    /// decoded `char` once a full sequence lands, `None` while a multi-byte
    /// sequence is still arriving, and `U+FFFD` for an invalid byte. Uses
    /// `core::str::from_utf8` (its `error_len() == None` = "incomplete, need
    /// more"; `Some(_)` = "invalid").
    fn feed_utf8(&mut self, b: u8) -> Option<char> {
        if self.utf8_len == 0 && b < 0x80 {
            return Some(b as char); // ASCII fast path
        }
        if self.utf8_len as usize >= self.utf8.len() {
            self.utf8_len = 0; // safety: never overflow the 4-byte buffer
        }
        self.utf8[self.utf8_len as usize] = b;
        self.utf8_len += 1;
        match core::str::from_utf8(&self.utf8[..self.utf8_len as usize]) {
            Ok(s) => {
                self.utf8_len = 0;
                s.chars().next()
            }
            Err(e) if e.error_len().is_none() => None, // incomplete — await more
            Err(_) => {
                self.utf8_len = 0;
                Some('\u{FFFD}') // invalid byte(s)
            }
        }
    }

    /// First numeric CSI parameter (0 if absent) — enough for `ESC[nC`/`nD`/`nK`.
    fn csi_param(&self) -> u64 {
        let mut v: u64 = 0;
        for &b in &self.csi[..self.csi_len] {
            if b.is_ascii_digit() {
                v = v.saturating_mul(10) + (b - b'0') as u64;
            } else {
                break;
            }
        }
        v
    }

    /// Clone text state (scrollback + grid + cursor + colour) from `old` without
    /// reflowing. Used when this pane is **parked** off-screen during fullscreen
    /// (`w == 0`): reflowing a multi-thousand-line history into the 1-column ghost
    /// grid `Pane::new` builds for a zero-width box would allocate/hang the OS
    /// (Ctrl+F hang). The parked pane keeps its native `cols`/`rows` so a later
    /// unpark can reflow correctly into the restored geometry.
    pub(super) fn take_content(&mut self, old: &Pane) {
        self.hist = old.hist.clone();
        self.grid = old.grid.clone();
        self.cols = old.cols;
        self.rows = old.rows;
        self.col = old.col.min(old.cols.saturating_sub(1));
        self.row = old.row.min(old.rows.saturating_sub(1));
        self.view = old.view.min(self.hist.len());
        self.sel = None;
        self.fg = old.fg;
        self.default_fg = old.default_fg;
        self.bold = old.bold;
        self.esc = old.esc;
        self.csi = old.csi;
        self.csi_len = old.csi_len;
    }

    /// Carry another pane's text (scrollback + grid + cursor + colour state)
    /// into this freshly-built pane, **reflowing** soft-wrapped lines to the
    /// new column count. Used when the layout is rebuilt (divider drag, action
    /// pane toggle, `/pane split`) so expanding the chat pane fills the extra
    /// width instead of leaving short lines stranded on the left.
    ///
    /// Parked destinations (`self.w == 0`, fullscreen) skip reflow entirely —
    /// see [`Self::take_content`].
    pub(super) fn adopt(&mut self, old: &Pane) {
        if old.grid.is_empty() && old.hist.is_empty() {
            return;
        }
        // Fullscreen parks a pane at outer width 0 (off-screen). Never reflow
        // into the 1-col placeholder grid — that turns ~2000×N cells into
        // millions of 1-cell rows and freezes the cooperative kernel.
        if self.w == 0 {
            self.take_content(old);
            return;
        }
        if self.grid.is_empty() {
            return;
        }
        let ocols = old.cols as usize;
        let cols = self.cols as usize;
        let rows = self.rows as usize;
        // Same width: transplant without soft-reflow (row count may still change).
        if ocols == cols && !old.grid.is_empty() {
            let empty: Cell = ('\0', self.default_fg);
            let mut abs: alloc::vec::Vec<alloc::vec::Vec<Cell>> = old.hist.iter().cloned().collect();
            let used = ((old.row + 1).min(old.rows)) as usize;
            for r in 0..used {
                let start = r * ocols;
                let end = (start + ocols).min(old.grid.len());
                if start < end {
                    abs.push(old.grid[start..end].to_vec());
                }
            }
            let total = abs.len();
            let keep = total.min(rows);
            let start = total - keep;
            for c in self.grid.iter_mut() {
                *c = empty;
            }
            for (r, line) in abs.iter().skip(start).enumerate() {
                let n = line.len().min(cols);
                for c in 0..n {
                    self.grid[r * cols + c] = line[c];
                }
            }
            self.hist = abs.into_iter().take(start).collect();
            while self.hist.len() > HIST_MAX {
                self.hist.pop_front();
            }
            let old_line = old.hist.len() + old.row.min(old.rows.saturating_sub(1)) as usize;
            self.row = if old_line >= start {
                (old_line - start).min(rows.saturating_sub(1)) as u64
            } else {
                0
            };
            self.col = old.col.min(self.cols.saturating_sub(1));
            self.view = 0;
            self.sel = None;
            // Keep the *new* theme's default_fg (set by Pane::new); only carry
            // an explicit non-default ANSI colour across, and recolour history.
            let old_fg = old.default_fg;
            self.fg = if old.fg == old_fg { self.default_fg } else { old.fg };
            self.bold = old.bold;
            self.recolor_default_fg(old_fg, self.default_fg);
            return;
        }
        // Absolute lines: scrollback then live grid rows that hold content.
        let mut abs: alloc::vec::Vec<alloc::vec::Vec<Cell>> = old.hist.iter().cloned().collect();
        let used = ((old.row + 1).min(old.rows)) as usize;
        for r in 0..used {
            let start = r * ocols;
            let end = (start + ocols).min(old.grid.len());
            if start < end {
                abs.push(old.grid[start..end].to_vec());
            }
        }
        let old_line = old.hist.len() + old.row.min(old.rows.saturating_sub(1)) as usize;
        let empty: Cell = ('\0', self.default_fg);
        // Same layout as textsel::Cell — Rgb is (u8,u8,u8).
        let as_ts: alloc::vec::Vec<alloc::vec::Vec<crate::textsel::Cell>> =
            abs.iter().map(|l| l.iter().map(|&(b, c)| (b, c)).collect()).collect();
        let reflowed = crate::textsel::reflow_lines(&as_ts, ocols, cols, ('\0', self.default_fg));
        let (new_line, new_col) =
            crate::textsel::reflow_cursor(&as_ts, ocols, cols, old_line, old.col as usize);
        // Place the tail of the reflow into the live grid; the rest is hist.
        let total = reflowed.len();
        let keep = total.min(rows);
        let start = total - keep;
        // Clear grid first so expanded columns aren't stale.
        for c in self.grid.iter_mut() {
            *c = empty;
        }
        for (r, line) in reflowed.iter().skip(start).enumerate() {
            let n = line.len().min(cols);
            for c in 0..n {
                self.grid[r * cols + c] = line[c];
            }
        }
        self.hist = reflowed.into_iter().take(start).collect();
        while self.hist.len() > HIST_MAX {
            self.hist.pop_front();
        }
        self.row = if new_line >= start {
            (new_line - start).min(rows.saturating_sub(1)) as u64
        } else {
            0
        };
        self.col = (new_col as u64).min(self.cols.saturating_sub(1));
        self.view = 0;
        self.sel = None; // absolute coords are invalid after reflow
        let old_fg = old.default_fg;
        self.fg = if old.fg == old_fg { self.default_fg } else { old.fg };
        self.bold = old.bold;
        self.recolor_default_fg(old_fg, self.default_fg);
    }

    /// Recolour adopted content after a theme switch: cells drawn in the old
    /// theme's default foreground (plain shell/agent text — the bulk of the
    /// scrollback) are remapped to the new theme's foreground, so switching
    /// e.g. dark→light doesn't leave the existing history invisible (light-on-
    /// light). Explicitly ANSI/syntax-coloured cells keep their colour.
    fn recolor_default_fg(&mut self, old_fg: Rgb, new_fg: Rgb) {
        if old_fg == new_fg {
            return;
        }
        for c in self.grid.iter_mut() {
            if c.1 == old_fg {
                c.1 = new_fg;
            }
        }
        for line in self.hist.iter_mut() {
            for c in line.iter_mut() {
                if c.1 == old_fg {
                    c.1 = new_fg;
                }
            }
        }
    }

    /// Drop all text content (grid + scrollback) — the `/clear` reset.
    pub(super) fn clear_content(&mut self) {
        for c in self.grid.iter_mut() {
            *c = ('\0', self.default_fg);
        }
        self.hist.clear();
        self.view = 0;
        self.sel = None;
        self.folds.clear();
        self.user_band.clear();
        self.col = 0;
        self.row = 0;
        self.fg = self.default_fg;
        self.bold = false;
    }
    fn cell_x(&self) -> u64 {
        self.ix + self.col * self.cw
    }
    fn cell_y(&self) -> u64 {
        self.iy + self.row * self.ch
    }

    /// Apply the buffered CSI `… m` (SGR) parameters to this pane's colour state.
    /// Supports reset (0), bold (1/22), default fg (39), the 8 normal (30–37) and
    /// bright (90–97) foreground colours, and 24-bit / 256-colour `38;2;r;g;b` /
    /// `38;5;n`. Background and other attributes are ignored.
    fn apply_sgr(&mut self) {
        // Parse the `;`-separated numeric params (empty => 0).
        let mut params = [0i32; 16];
        let mut np = 0usize;
        let (mut cur, mut has) = (0i32, false);
        for &b in &self.csi[..self.csi_len] {
            if b == b';' {
                if np < params.len() {
                    params[np] = cur;
                    np += 1;
                }
                cur = 0;
                has = false;
            } else if b.is_ascii_digit() {
                cur = cur.saturating_mul(10) + (b - b'0') as i32;
                has = true;
            }
        }
        if np < params.len() {
            params[np] = cur;
            np += 1;
        }
        let _ = has;
        let mut i = 0;
        while i < np {
            match params[i] {
                0 => {
                    self.fg = self.default_fg;
                    self.bold = false;
                }
                1 => self.bold = true,
                22 => self.bold = false,
                39 => self.fg = self.default_fg,
                30..=37 => self.fg = ansi_color((params[i] - 30) as usize, self.bold),
                90..=97 => self.fg = ansi_color((params[i] - 90) as usize, true),
                38 => {
                    if i + 4 < np && params[i + 1] == 2 {
                        self.fg = (params[i + 2] as u8, params[i + 3] as u8, params[i + 4] as u8);
                        i += 4;
                    } else if i + 2 < np && params[i + 1] == 5 {
                        self.fg = ansi_256(params[i + 2] as u8);
                        i += 2;
                    }
                }
                _ => {}
            }
            i += 1;
        }
    }
}

/// A throwaway pane used only to satisfy the borrow checker while a real pane is
/// temporarily moved out for a `&Screen` + `&mut Pane` split (see `pane_putc`,
/// which needs immutable access to the screen's pixel plumbing while mutating a
/// pane). Its geometry is degenerate so it never draws anything if used.
pub(super) fn dummy_pane() -> Pane {
    Pane {
        x: 0, y: 0, w: 0, h: 0, ix: 0, iy: 0, cw: 1, ch: 1, cols: 1, rows: 1, col: 0, row: 0,
        fg: (0, 0, 0), default_fg: (0, 0, 0), bg: (0, 0, 0),
        esc: EscState::Ground, csi: [0; 32], csi_len: 0, bold: false,
        title: String::new(), show_caret: false,
        grid: Vec::new(), hist: VecDeque::new(), view: 0, sel: None, has_composer: false,
        folds: Vec::new(), user_band: Vec::new(), utf8: [0; 4], utf8_len: 0,
    }
}

impl Screen {
    /// Scroll a pane's interior up by one text row: the top grid row is evicted
    /// into the scrollback ring, the grid shifts up, and (when the view is live)
    /// the pixels shift with it.
    fn scroll_pane(&self, p: &mut Pane) {
        // Grid + scrollback first — the source of truth.
        let cols = p.cols as usize;
        if p.grid.len() >= cols {
            p.hist.push_back(p.grid[..cols].to_vec());
            while p.hist.len() > HIST_MAX {
                p.hist.pop_front();
                // Absolute selection coordinates shift with the evicted line;
                // a selection that loses its first line is dropped.
                p.sel = p.sel.and_then(|((r1, c1), (r2, c2))| {
                    (r1.min(r2) > 0).then(|| ((r1 - 1, c1), (r2 - 1, c2)))
                });
                // Fold anchors shift the same way; a fold whose "▸ more" line is
                // evicted is dropped.
                p.folds.retain_mut(|(gi, _)| {
                    if *gi == 0 {
                        false
                    } else {
                        *gi -= 1;
                        true
                    }
                });
                // User-band line indices track absolute gi the same way.
                p.user_band.retain_mut(|gi| {
                    if *gi == 0 {
                        false
                    } else {
                        *gi -= 1;
                        true
                    }
                });
            }
            p.grid.copy_within(cols.., 0);
            let start = p.grid.len() - cols;
            let fg = p.fg;
            for c in &mut p.grid[start..] {
                *c = ('\0', fg);
            }
            if p.view > 0 {
                // Keep the scrolled view anchored on the same content.
                p.view = (p.view + 1).min(p.hist.len());
                return; // pixels are frozen on the scrolled view
            }
        }
        // A translucent wallpaper is a fixed backdrop — a pixel-memmove scroll
        // would drag it up with the text. Repaint the interior from the grid
        // over a fresh wallpaper background instead.
        if self.wallpaper.is_some() && self.opacity < 255 {
            self.paint_surface(p.ix, p.iy, p.cols * p.cw, p.rows * p.ch, p.bg);
            self.render_view(p);
            return;
        }
        let x0 = p.ix;
        let w = p.cols * p.cw;
        let top = p.iy;
        let h = p.rows * p.ch;
        let step = (self.pitch * p.ch) as usize;
        let row_bytes = (w * self.bpp_bytes) as usize;
        // SAFETY: every source/destination row lies inside the framebuffer and
        // inside this pane's x-span; source and destination never overlap within
        // a single `copy_nonoverlapping` (they are `p.ch` rows apart).
        unsafe {
            let base = self.addr as *mut u8;
            for row in 0..(h - p.ch) {
                let dst = self.fb_offset(x0, top + row) as usize;
                base.add(dst).copy_from_nonoverlapping(base.add(dst + step), row_bytes);
            }
        }
        self.fill_rect(x0, top + h - p.ch, w, p.ch, p.bg);
    }

    /// Repaint a pane's interior from its scrollback + grid at the current view
    /// offset. The one text renderer used by scroll, redraw, relayout, and the
    /// mouse selection (whose cells get the selection background).
    ///
    /// **No full-interior clear** — blank-then-repaint is what made selection
    /// drag (and scroll) flicker on the single-buffered framebuffer. Every cell
    /// is painted in place (bg + glyph), so nothing is ever blanked mid-frame.
    pub(super) fn render_view(&self, p: &Pane) {
        let cols = p.cols as usize;
        if p.grid.len() < cols {
            return;
        }
        let sel = p.sel.map(|(a, b)| crate::textsel::normalize(a, b));
        let view = p.view.min(p.hist.len());
        let first = p.hist.len() - view;
        for r in 0..p.rows as usize {
            let gi = first + r;
            let line: Option<&[Cell]> = if gi < p.hist.len() {
                Some(&p.hist[gi])
            } else {
                let gr = gi - p.hist.len();
                if gr >= p.rows as usize {
                    break;
                }
                Some(&p.grid[gr * cols..(gr + 1) * cols])
            };
            for c in 0..cols {
                let (b, fg) = line.and_then(|l| l.get(c).copied()).unwrap_or(('\0', p.default_fg));
                let x = p.ix + c as u64 * p.cw;
                let y = p.iy + r as u64 * p.ch;
                let selected = sel.is_some_and(|s| crate::textsel::contains(s, gi, c));
                let bg = if selected {
                    self.theme.editor_sel
                } else if p.band_bg(gi) {
                    self.theme.composer_bg
                } else {
                    p.bg
                };
                // Always fill the cell first so deselected / empty cells leave
                // no residue (selection highlight, partial glyphs).
                self.fill_cell_bg(x, y, p.cw, p.ch, bg);
                if b != '\0' && b != ' ' {
                    self.blit_glyph(x, y, b, fg, bg);
                }
            }
        }
        // A scrolled-back view gets a position marker in the top-right corner.
        if view > 0 {
            let tag = alloc::format!("[-{}] ", view);
            let tx = (p.ix + p.cols * p.cw).saturating_sub(tag.len() as u64 * p.cw);
            self.draw_str(tx, p.iy, &tag, self.theme.accent, p.bg);
        }
    }

    /// Paint one chat-pane cell at absolute line `gi`, column `c` (selection
    /// highlight applied when `selected`). Used by differential selection
    /// updates so a drag only touches cells that actually changed.
    pub(super) fn paint_chat_cell(&self, p: &Pane, gi: usize, c: usize, selected: bool) {
        let cols = p.cols as usize;
        if c >= cols || p.grid.len() < cols {
            return;
        }
        let view = p.view.min(p.hist.len());
        let first = p.hist.len() - view;
        if gi < first || gi >= first + p.rows as usize {
            return; // off-screen
        }
        let r = gi - first;
        let (b, fg) = if gi < p.hist.len() {
            p.hist[gi].get(c).copied().unwrap_or(('\0', p.default_fg))
        } else {
            let gr = gi - p.hist.len();
            if gr >= p.rows as usize {
                return;
            }
            p.grid.get(gr * cols + c).copied().unwrap_or(('\0', p.default_fg))
        };
        let x = p.ix + c as u64 * p.cw;
        let y = p.iy + r as u64 * p.ch;
        let bg = if selected {
            self.theme.editor_sel
        } else if p.band_bg(gi) {
            self.theme.composer_bg
        } else {
            p.bg
        };
        self.fill_cell_bg(x, y, p.cw, p.ch, bg);
        if b != '\0' && b != ' ' {
            self.blit_glyph(x, y, b, fg, bg);
        }
    }

    /// Repaint only the cells whose selection membership differs between
    /// `old_sel` and `new_sel` (both raw anchor/head pairs). Avoids the
    /// full-pane flash that a drag-triggered `render_view` used to cause.
    pub(super) fn repaint_sel_diff(
        &self,
        p: &Pane,
        old_sel: Option<((usize, usize), (usize, usize))>,
        new_sel: Option<((usize, usize), (usize, usize))>,
    ) {
        let old = old_sel.map(|(a, b)| crate::textsel::normalize(a, b));
        let new = new_sel.map(|(a, b)| crate::textsel::normalize(a, b));
        if old == new {
            return;
        }
        let cols = p.cols as usize;
        let view = p.view.min(p.hist.len());
        let first = p.hist.len() - view;
        let last = first + p.rows as usize;
        // Bound the walk to the union of the two ranges (clamped to the view).
        let span = |s: Option<((usize, usize), (usize, usize))>| -> Option<(usize, usize)> {
            s.map(|((r1, _), (r2, _))| (r1.max(first), (r2 + 1).min(last)))
        };
        let (lo, hi) = match (span(old), span(new)) {
            (Some((a, b)), Some((c, d))) => (a.min(c), b.max(d)),
            (Some((a, b)), None) | (None, Some((a, b))) => (a, b),
            (None, None) => return,
        };
        for gi in lo..hi {
            for c in 0..cols {
                let was = old.is_some_and(|s| crate::textsel::contains(s, gi, c));
                let now = new.is_some_and(|s| crate::textsel::contains(s, gi, c));
                if was != now {
                    self.paint_chat_cell(p, gi, c, now);
                }
            }
        }
    }

    /// Repaint the cell under the pane cursor from the grid (clears a leftover
    /// caret bar without blanking a real glyph that might share the cell).
    pub(super) fn repaint_cursor_cell(&self, p: &Pane) {
        let cols = p.cols as usize;
        let idx = (p.row as usize).saturating_mul(cols).saturating_add(p.col as usize);
        let (b, fg) = p.grid.get(idx).copied().unwrap_or(('\0', p.default_fg));
        let x = p.cell_x();
        let y = p.cell_y();
        self.fill_cell_bg(x, y, p.cw, p.ch, p.bg);
        if b != '\0' && b != ' ' {
            self.blit_glyph(x, y, b, fg, p.bg);
        }
    }

    fn caret_erase(&self, p: &Pane) {
        if !p.show_caret {
            return;
        }
        // Always restore the underlying cell — a plain bg bar erase leaves a
        // hole if a glyph shared the cell, and a leftover accent bar if the
        // next write never covers this position (composer panes).
        self.repaint_cursor_cell(p);
    }

    pub(super) fn caret_draw(&self, p: &Pane) {
        // Chat pane with a bordered composer: caret lives only in the input
        // box — never in the scrollback/response area.
        if !p.show_caret || p.has_composer {
            return;
        }
        self.fill_rect(p.cell_x(), p.cell_y(), 2 * self.scale, p.ch, self.theme.accent);
    }

    pub(super) fn newline(p: &mut Pane, s: &Screen) {
        p.col = 0;
        p.row += 1;
        if p.row >= p.rows {
            s.scroll_pane(p);
            p.row = p.rows - 1;
        }
    }

    /// Feed one byte to a pane (the per-pane analogue of a terminal write),
    /// running the ANSI escape parser first so `\x1b[…m` SGR codes recolour the
    /// stream instead of printing as garbage.
    pub(super) fn pane_putc(s: &Screen, p: &mut Pane, byte: u8) {
        match p.esc {
            EscState::Esc => {
                // Only CSI (`ESC [`) is supported; anything else ends the escape.
                p.esc = if byte == b'[' {
                    p.csi_len = 0;
                    EscState::Csi
                } else {
                    EscState::Ground
                };
                return;
            }
            EscState::Csi => {
                if (0x40..=0x7e).contains(&byte) {
                    // Final byte: SGR (`m`) recolours; `K`/`C`/`D` (erase to end
                    // of line / cursor right / cursor left) support the shell's
                    // in-line editing. Other CSI are consumed and ignored.
                    let live = p.view == 0;
                    match byte {
                        b'm' => p.apply_sgr(),
                        b'K' => {
                            if live {
                                s.caret_erase(p);
                            }
                            let cols = p.cols as usize;
                            let (row, col) = (p.row as usize, p.col as usize);
                            for c in col..cols {
                                if let Some(cell) = p.grid.get_mut(row * cols + c) {
                                    *cell = ('\0', p.fg);
                                }
                            }
                            if live {
                                s.fill_cell_bg(p.cell_x(), p.cell_y(), (p.cols - p.col) * p.cw, p.ch, p.bg);
                                s.caret_draw(p); // no-op when p.has_composer
                            }
                        }
                        b'C' | b'D' => {
                            let n = p.csi_param().max(1);
                            if live {
                                s.caret_erase(p);
                            }
                            p.col = if byte == b'C' { (p.col + n).min(p.cols - 1) } else { p.col.saturating_sub(n) };
                            if live {
                                s.caret_draw(p); // no-op when p.has_composer
                            }
                        }
                        _ => {}
                    }
                    p.esc = EscState::Ground;
                } else if p.csi_len < p.csi.len() {
                    p.csi[p.csi_len] = byte;
                    p.csi_len += 1;
                }
                return;
            }
            EscState::Ground => {}
        }
        if byte == 0x1b {
            p.esc = EscState::Esc;
            return;
        }
        // While scrolled back, the grid/scrollback still update but the pixels
        // stay frozen on the scrolled view (`scroll_pane` keeps it anchored).
        let live = p.view == 0;
        if live {
            s.caret_erase(p);
        }
        match byte {
            b'\n' => Screen::newline(p, s),
            b'\r' => p.col = 0,
            b'\t' => {
                let next = (p.col / 4 + 1) * 4;
                while p.col < next && p.col < p.cols {
                    p.set_cell(' ');
                    if live {
                        s.blit_glyph(p.cell_x(), p.cell_y(), ' ', p.fg, p.bg);
                    }
                    p.col += 1;
                }
                if p.col >= p.cols {
                    Screen::newline(p, s);
                }
            }
            0x08 | 0x7f => {
                if p.col > 0 {
                    p.col -= 1;
                } else if p.row > 0 {
                    p.row -= 1;
                    p.col = p.cols - 1;
                }
                p.set_cell('\0');
                if live {
                    s.blit_glyph(p.cell_x(), p.cell_y(), ' ', p.fg, p.bg);
                }
            }
            // Other C0 control bytes: ignored (never part of a UTF-8 sequence).
            b if b < 0x20 => {}
            // Printable: ASCII (0x20–0x7e) or a UTF-8 lead/continuation byte
            // (≥0x80). Decode incrementally — a multi-byte glyph spans calls.
            _ => {
                if let Some(ch) = p.feed_utf8(byte) {
                    p.set_cell(ch);
                    if live {
                        s.blit_glyph(p.cell_x(), p.cell_y(), ch, p.fg, p.bg);
                    }
                    p.col += 1;
                    if p.col >= p.cols {
                        Screen::newline(p, s);
                    }
                }
            }
        }
        // Grid caret only when the pane has no composer (`caret_draw` is a
        // no-op for `has_composer` panes so scrollback never keeps a bar).
        if p.view == 0 {
            s.caret_draw(p);
        }
    }

    // --- framing ---------------------------------------------------------

    pub(super) fn draw_frame(&self, p: &Pane, active: bool) {
        self.draw_frame_titled(p, active, &p.title);
    }

    /// Like [`draw_frame`] but with an explicit title (the editor overrides the
    /// pane title with `editor: <file>`).
    pub(super) fn draw_frame_titled(&self, p: &Pane, active: bool, title: &str) {
        let border = if active { self.theme.accent } else { self.theme.border_dim };
        let title_c = if active { self.theme.title_active } else { self.theme.title_dim };
        self.rect_outline(p.x, p.y, p.w, p.h, BORDER, border);
        // Title, just inside the top border — ellipsize so a long path never
        // paints into the close button / pane edge.
        let ty = p.y + BORDER + 4;
        let tx = p.x + BORDER + PAD;
        let max_w = p.w.saturating_sub(2 * (BORDER + PAD) + self.cw() * 4); // room for " *" / [x]
        let max_cols = (max_w / self.cw()).max(1) as usize;
        let title = crate::textsel::ellipsize(title, max_cols);
        let end = self.draw_str(tx, ty, &title, title_c, p.bg);
        if active && end + 2 * self.cw() <= p.x + p.w - BORDER - PAD {
            self.draw_str(end, ty, " *", self.theme.accent, p.bg);
        }
        // Separator under the title.
        let sep_y = ty + self.ch() + 3;
        self.fill_rect(p.x + BORDER, sep_y, p.w - 2 * BORDER, 1, self.theme.sep_dim);
    }

    /// Re-paint chat grid rows whose vertical span intersects `[y0, y1)`.
    pub(super) fn render_view_rows_intersecting(&self, y0: u64, y1: u64) {
        let p = &self.chat;
        if p.w == 0 || p.rows == 0 {
            return;
        }
        let ch = p.ch;
        let first = if y0 <= p.iy {
            0
        } else {
            ((y0 - p.iy) / ch) as usize
        };
        let last = if y1 <= p.iy {
            0
        } else {
            (((y1 - p.iy) + ch - 1) / ch).min(p.rows) as usize
        };
        if first >= last {
            return;
        }
        self.render_view_row_range(p, first, last);
    }

    /// Like [`render_view`] but only rows `[row0, row1)`.
    fn render_view_row_range(&self, p: &Pane, row0: usize, row1: usize) {
        let cols = p.cols as usize;
        if p.grid.len() < cols {
            return;
        }
        let sel = p.sel.map(|(a, b)| crate::textsel::normalize(a, b));
        let view = p.view.min(p.hist.len());
        let first = p.hist.len() - view;
        let row1 = row1.min(p.rows as usize);
        for r in row0..row1 {
            let gi = first + r;
            let line: Option<&[Cell]> = if gi < p.hist.len() {
                Some(&p.hist[gi])
            } else {
                let gr = gi - p.hist.len();
                if gr >= p.rows as usize {
                    break;
                }
                Some(&p.grid[gr * cols..(gr + 1) * cols])
            };
            for c in 0..cols {
                let (b, fg) = line.and_then(|l| l.get(c).copied()).unwrap_or(('\0', p.default_fg));
                let x = p.ix + c as u64 * p.cw;
                let y = p.iy + r as u64 * p.ch;
                let selected = sel.is_some_and(|s| crate::textsel::contains(s, gi, c));
                let bg = if selected {
                    self.theme.editor_sel
                } else if p.band_bg(gi) {
                    self.theme.composer_bg
                } else {
                    p.bg
                };
                self.fill_cell_bg(x, y, p.cw, p.ch, bg);
                if b != '\0' && b != ' ' {
                    self.blit_glyph(x, y, b, fg, bg);
                }
            }
        }
    }

    /// Paint the caret in its current blink state. When the chat pane has a
    /// bordered composer, the caret only blinks inside the box while the
    /// prompt is active — never during streamed reply output (that was a full
    /// box redraw and looked like the whole composer flickering).
    pub(super) fn paint_caret(&self) {
        if self.chat.has_composer {
            if self.composer_active {
                self.paint_composer_caret();
            }
            return;
        }
        if !self.chat.show_caret || self.chat.view != 0 {
            return;
        }
        let color = if self.caret_on { self.theme.accent } else { self.chat.bg };
        self.fill_rect(self.chat.cell_x(), self.chat.cell_y(), 2 * self.scale, self.chat.ch, color);
    }
}

#[cfg(test)]
mod theme_switch_tests {
    use super::dummy_pane;

    #[test_case]
    fn recolor_remaps_only_old_default_fg() {
        let old_fg = (250, 249, 245); // dark theme's cream text
        let new_fg = (38, 35, 31); // light theme's near-black text
        let accent = (204, 120, 92); // an explicit ANSI/syntax colour
        let mut p = dummy_pane();
        // A scrollback line: plain text in old default + one accent-coloured cell.
        p.hist.push_back(alloc::vec![(b'a', old_fg), (b'b', accent), (0, old_fg)]);
        p.grid = alloc::vec![(b'c', old_fg)];
        p.default_fg = new_fg; // as Pane::new set it for the new theme
        p.recolor_default_fg(old_fg, new_fg);
        // Default-coloured cells (incl. the empty one) move to the new fg…
        assert_eq!(p.hist[0][0].1, new_fg);
        assert_eq!(p.hist[0][2].1, new_fg);
        assert_eq!(p.grid[0].1, new_fg);
        // …the explicit accent colour is preserved.
        assert_eq!(p.hist[0][1].1, accent);
    }

    #[test_case]
    fn recolor_is_a_noop_when_fg_unchanged() {
        let fg = (10, 20, 30);
        let mut p = dummy_pane();
        p.hist.push_back(alloc::vec![(b'x', fg)]);
        p.recolor_default_fg(fg, fg);
        assert_eq!(p.hist[0][0].1, fg);
    }
}
