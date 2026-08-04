//! The chat composer: its box geometry, caret, hint line, and the
//! autosuggest popup.

use super::*;

/// Vertical padding inside the bordered input composer box (px, unscaled).
pub(super) const COMPOSER_VPAD: u64 = 6;

/// Gap between the composer box and the hint row under it (px, unscaled).
pub(super) const COMPOSER_HINT_GAP: u64 = 4;

/// Margin between chat scrollback and the composer box (px, unscaled).
pub(super) const COMPOSER_TOP_GAP: u64 = 8;

/// Whether the chat pane has a bordered input composer (always true once the
/// framebuffer console is up with a chat pane).
pub fn composer_available() -> bool {
    SCREEN.with(|slot| slot.as_ref().is_some_and(|sc| sc.chat.has_composer))
}

/// Whether the composer is the live prompt (between [`composer_begin`] and
/// [`composer_end`]). Serial line-editing still runs in parallel.
pub fn composer_is_active() -> bool {
    SCREEN.with(|slot| slot.as_ref().is_some_and(|sc| sc.composer_active))
}

/// Activate the input composer (call at the start of a prompt `read_line`).
pub fn composer_begin() {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.cursor_restore();
            sc.cur_vis = false;
            // Wipe any residual scrollback caret left after a streamed reply
            // (accent bar at the end of the last response line).
            if sc.chat.view == 0 {
                sc.repaint_cursor_cell(&sc.chat);
            }
            sc.composer_active = true;
            sc.composer_line.clear();
            sc.composer_cur = 0;
            sc.caret_on = true;
            sc.draw_composer();
            sc.cursor_overlay();
        }
    });
}

/// Update the composer line + caret column (0..=len). Redraws the box in place.
pub fn composer_set(line: &str, cursor: usize) {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.cursor_restore();
            sc.cur_vis = false;
            sc.composer_line.clear();
            sc.composer_line.push_str(line);
            sc.composer_cur = cursor.min(line.len());
            sc.caret_on = true;
            sc.draw_composer();
            sc.cursor_overlay();
        }
    });
}

/// Set the left half of the composer hint bar (live status:
/// "Waiting for response… |").
///
/// Always repaints the composer strip when the chat has a composer — including
/// while a turn is running (`composer_active == false` after submit). The old
/// gate only drew during typing, so wait animation never appeared.
pub fn composer_set_hint_left(s: &str) {
    composer_set_hint_left_lead(s, &[]);
}

/// Set the composer prompt prefix (`~/path (branch) > `) from the shell's live
/// cwd + git branch. Repaints the composer box.
pub fn composer_set_prompt(s: &str) {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.composer_prompt.clear();
            sc.composer_prompt.push_str(s);
            if sc.chat.has_composer {
                sc.draw_composer();
            }
        }
    });
}

/// Set the left hint, colouring its first `lead.len()` characters from `lead`
/// (one colour per character) instead of `theme.composer_hint`.
///
/// This is the shell's progress-bar channel: the animation lives in
/// `shell::chrome` and only the finished per-cell colours arrive here, so the
/// compositor stays ignorant of the frame sequence. Pass an empty `lead` for
/// an ordinary single-colour hint.
pub fn composer_set_hint_left_lead(s: &str, lead: &[(u8, u8, u8)]) {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.composer_hint_l.clear();
            sc.composer_hint_l.push_str(s);
            sc.composer_hint_l_lead.clear();
            sc.composer_hint_l_lead.extend_from_slice(lead);
            if sc.chat.has_composer {
                sc.cursor_restore();
                sc.cur_vis = false;
                sc.draw_composer();
                sc.cursor_overlay();
            }
        }
    });
}

/// The live theme's progress-gradient endpoints: `(dim, bright)` =
/// `(composer_hint, accent)`. The shell's wait animation ramps between these
/// so it follows `/theme` and stays on the brand palette rather than carrying
/// colours of its own. Falls back to the brand dark theme before the
/// framebuffer exists (serial-only boot).
pub fn hint_gradient() -> ((u8, u8, u8), (u8, u8, u8)) {
    SCREEN.with(|slot| match slot.as_ref() {
        Some(sc) => (sc.theme.composer_hint, sc.theme.accent),
        None => (Theme::BRAND_DARK.composer_hint, Theme::BRAND_DARK.accent),
    })
}

/// Mark the last `n` absolute chat lines as a user-prompt band (elevated
/// `composer_bg`), pad them full-width, and repaint so the band is visible
/// immediately (including empty cells to the right of the text).
pub fn chat_mark_user_band_rows(n: usize) {
    if n == 0 {
        return;
    }
    SCREEN.with(|slot| {
        let Some(sc) = slot.as_mut() else { return };
        let p = &mut sc.chat;
        let cols = p.cols as usize;
        if cols == 0 {
            return;
        }
        // Cursor sits on the line *after* the last printed content when the
        // user turn ended with `\n`. The band covers the previous `n` lines.
        let end = p.hist.len() + p.row as usize; // exclusive end (current empty row)
        let start = end.saturating_sub(n);
        for gi in start..end {
            if let Err(i) = p.user_band.binary_search(&gi) {
                p.user_band.insert(i, gi);
            }
            // Cap bookkeeping so a long session cannot grow unbounded.
            if p.user_band.len() > 256 {
                p.user_band.drain(0..p.user_band.len() - 256);
            }
            // Pad short rows so the elevated fill spans the full pane width.
            if gi < p.hist.len() {
                let line = &mut p.hist[gi];
                if line.len() < cols {
                    let fg = p.default_fg;
                    line.resize(cols, ('\0', fg));
                }
            } else {
                let gr = gi - p.hist.len();
                if gr < p.rows as usize {
                    // Live grid rows are already full-width.
                }
            }
        }
        // Repaint band rows that are on screen (drop mut borrow first).
        let view = p.view;
        let hist_len = p.hist.len();
        let rows = p.rows as usize;
        let paint: alloc::vec::Vec<usize> = if view == 0 {
            let first = hist_len - view.min(hist_len);
            (start..end)
                .filter(|&gi| gi >= first && gi < first + rows)
                .collect()
        } else {
            alloc::vec::Vec::new()
        };
        for gi in paint {
            for c in 0..cols {
                sc.paint_chat_cell(&sc.chat, gi, c, false);
            }
        }
    });
}

/// Set the right half of the composer hint bar (e.g. model name / approval mode).
pub fn composer_set_hint_right(s: &str) {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.composer_hint_r.clear();
            sc.composer_hint_r.push_str(s);
            if sc.chat.has_composer {
                sc.cursor_restore();
                sc.cur_vis = false;
                sc.draw_composer();
                sc.cursor_overlay();
            }
        }
    });
}

/// The current right-hand composer hint (transient overlays like the reverse
/// search save it and restore it on exit).
pub fn composer_hint_right() -> String {
    SCREEN.with(|slot| slot.as_ref().map(|sc| sc.composer_hint_r.clone()).unwrap_or_default())
}

/// Deactivate the composer (call when a line is submitted or the prompt ends).
pub fn composer_end() {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.cursor_restore();
            sc.cur_vis = false;
            sc.composer_active = false;
            sc.composer_line.clear();
            sc.composer_cur = 0;
            // Drop any open suggestion menu with the prompt.
            if sc.suggest_open || sc.suggest_rect.is_some() {
                sc.suggest_open = false;
                sc.suggest_items.clear();
                sc.suggest_sel = 0;
                sc.suggest_clear_region(true);
                sc.suggest_rect = None;
            }
            sc.draw_composer(); // empty idle box
            // Ensure the chat grid never shows a caret while a reply streams.
            if sc.chat.view == 0 {
                sc.repaint_cursor_cell(&sc.chat);
            }
            sc.cursor_overlay();
        }
    });
}

/// Update the slash-command / @file suggestion popup.
/// `items` is `(label, detail)` rows; `selected` is the highlighted index.
/// Empty `items` dismisses the menu.
///
/// **Typing performance:** does **not** full-repaint the chat pane on every
/// key. Old popup rect is erased cheaply; only the popup (and optional
/// composer box) is redrawn. Composer line text is already painted by
/// [`composer_set`].
pub fn suggest_set(items: &[(alloc::string::String, alloc::string::String)], selected: usize) {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.cursor_restore();
            sc.cur_vis = false;
            let was_open = sc.suggest_open || sc.suggest_rect.is_some();
            let old_len = sc.suggest_items.len();
            let old_sel = sc.suggest_sel;
            sc.suggest_items.clear();
            for (l, d) in items {
                sc.suggest_items.push((l.clone(), d.clone()));
            }
            sc.suggest_sel = if items.is_empty() {
                0
            } else {
                selected.min(items.len() - 1)
            };
            sc.suggest_open = !items.is_empty();

            if !sc.suggest_open {
                if was_open {
                    sc.suggest_clear_region(true);
                    sc.suggest_rect = None;
                }
                // Composer already current from composer_set — avoid a second
                // full chrome paint on every non-slash key.
                sc.cursor_overlay();
                return;
            }

            // Erase previous popup footprint if it was taller / different.
            if was_open {
                sc.suggest_clear_region(false);
                sc.suggest_rect = None;
            }
            if let Some(rect) = sc.suggest_geom() {
                sc.suggest_rect = Some(rect);
            } else {
                sc.suggest_open = false;
                sc.suggest_rect = None;
                sc.cursor_overlay();
                return;
            }
            // Popup only — composer text already drawn by the line editor.
            // When row count/selection changed a lot, still just the popup.
            let _ = (old_len, old_sel);
            sc.draw_suggest_popup();
            sc.cursor_overlay();
        }
    });
}

/// Dismiss the suggestion popup (if any).
pub fn suggest_clear() {
    suggest_set(&[], 0);
}

/// Whether the suggestion popup currently has rows.
pub fn suggest_is_open() -> bool {
    SCREEN.with(|slot| slot.as_ref().is_some_and(|sc| sc.suggest_open && !sc.suggest_items.is_empty()))
}

/// Erase any leftover grid caret in the chat response area (call after a
/// streamed reply finishes, before the next prompt). Safe no-op without FB.
pub fn clear_chat_caret() {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            if sc.chat.has_composer && sc.chat.view == 0 {
                sc.cursor_restore();
                sc.cur_vis = false;
                sc.repaint_cursor_cell(&sc.chat);
                sc.cursor_overlay();
            }
        }
    });
}

impl Screen {
    /// Geometry of the bordered input composer inside the chat pane:
    /// `(box_x, box_y, box_w, box_h, text_x, text_y, hint_y)`.
    fn composer_geom(&self) -> (u64, u64, u64, u64, u64, u64, u64) {
        let p = &self.chat;
        let vpad = COMPOSER_VPAD;
        let hint_gap = COMPOSER_HINT_GAP;
        // Pane interior bottom (above outer border + pad).
        let bottom = p.y + p.h - BORDER - PAD;
        let hint_y = bottom.saturating_sub(p.ch);
        let box_h = vpad + p.ch + vpad + 2; // 1px border each side
        let box_y = hint_y.saturating_sub(hint_gap + box_h);
        let box_x = p.x + BORDER + PAD;
        let box_w = p.w.saturating_sub(2 * (BORDER + PAD));
        let text_x = box_x + 8;
        let text_y = box_y + 1 + vpad;
        (box_x, box_y, box_w, box_h, text_x, text_y, hint_y)
    }

    /// Visible slice of the composer line and the column of the caret within it
    /// (for long lines that scroll inside the box).
    fn composer_visible(&self, max_cols: usize) -> (usize, &str, usize) {
        let line = self.composer_line.as_str();
        let cur = self.composer_cur.min(line.len());
        if line.len() <= max_cols {
            return (0, line, cur);
        }
        let start = cur.saturating_sub(max_cols.saturating_sub(1)).min(line.len().saturating_sub(max_cols));
        let vis = &line[start..start + max_cols.min(line.len() - start)];
        (start, vis, cur.saturating_sub(start))
    }

    /// Pixel x of the caret bar inside the composer box.
    fn composer_caret_x(&self) -> Option<u64> {
        if !self.composer_active || self.action_focused() || !self.chat.has_composer {
            return None;
        }
        let (bx, _by, bw, _bh, tx, _ty, _hy) = self.composer_geom();
        let max_cols = ((bw.saturating_sub(16)) / self.chat.cw).saturating_sub(2) as usize;
        let (_start, _vis, caret_col) = self.composer_visible(max_cols);
        let prompt_cols = if self.composer_prompt.is_empty() {
            2 // "> "
        } else {
            self.composer_prompt.chars().count()
        };
        Some(tx + (prompt_cols as u64 + caret_col as u64) * self.chat.cw)
    }

    /// Paint **only** the composer caret bar (blink path). Never blanks the box —
    /// a full `draw_composer` on every blink (and every streamed token) was the
    /// flicker during response rendering.
    pub(super) fn paint_composer_caret(&self) {
        let Some(cx) = self.composer_caret_x() else { return };
        let (_bx, _by, _bw, _bh, _tx, ty, _hy) = self.composer_geom();
        let color = if self.caret_on { self.theme.accent } else { self.theme.composer_bg };
        self.fill_rect(cx, ty, 2 * self.scale.max(1), self.chat.ch, color);
    }

    /// Paint the bordered composer box + hint bar at the bottom of the chat pane.
    ///
    /// Paints **in place** (no strip-wide clear). The chat grid is already sized
    /// above this region, so scrollback never lands here; blanking the whole
    /// reserved strip on every call is what made streaming replies flash the box.
    pub(super) fn draw_composer(&self) {
        if !self.chat.has_composer || self.chat.w == 0 {
            return; // parked (action-fullscreen) — no visible chat chrome
        }
        let (bx, by, bw, bh, tx, ty, hy) = self.composer_geom();
        // Elevated fill + rounded border (accent when the prompt owns focus).
        self.paint_surface(bx + 1, by + 1, bw.saturating_sub(2), bh.saturating_sub(2), self.theme.composer_bg);
        let border = if self.composer_active && !self.action_focused() {
            self.theme.accent
        } else {
            self.theme.composer_border
        };
        let radius = (4 * self.scale).max(4);
        self.rounded_outline(bx, by, bw, bh, radius, border);
        // Prompt glyph + input text.
        let prompt: &str = if self.composer_prompt.is_empty() {
            "> "
        } else {
            &self.composer_prompt
        };
        let prompt_cols = prompt.chars().count() as u64;
        let max_cols = ((bw.saturating_sub(16)) / self.chat.cw).saturating_sub(2) as usize;
        let (_vis_start, vis, caret_col) = self.composer_visible(max_cols);
        let mut x = self.draw_str(tx, ty, prompt, self.theme.accent, self.theme.composer_bg);
        x = self.draw_str(x, ty, vis, self.theme.chat_fg, self.theme.composer_bg);
        // Clear leftover glyphs to the right of the text (shrinking line).
        let rest = (bx + bw).saturating_sub(x + 4);
        if rest > 0 {
            self.paint_surface(x, ty, rest, self.chat.ch, self.theme.composer_bg);
        }
        // Caret inside the box (only while the composer is the live prompt).
        if self.composer_active && !self.action_focused() {
            let cx = tx + (prompt_cols + caret_col as u64) * self.chat.cw;
            let color = if self.caret_on { self.theme.accent } else { self.theme.composer_bg };
            self.fill_rect(cx, ty, 2 * self.scale.max(1), self.chat.ch, color);
        }
        // Hint bar: shortcuts left, model/mode right — each side ellipsized so
        // a narrow chat pane never paints past the composer box (or overlaps).
        let hx = bx;
        let hw = bw;
        let cw = self.chat.cw;
        self.fill_cell_bg(hx, hy, hw, self.chat.ch, self.chat.bg);
        let total_cols = (hw / cw).max(1) as usize;
        let gap = 2usize;
        let right_raw = self.composer_hint_r.chars().count();
        let right_cols = right_raw.min(total_cols / 3).min(total_cols.saturating_sub(gap + 4));
        let left_cols = total_cols.saturating_sub(right_cols + if right_cols > 0 { gap } else { 0 });
        let left = crate::textsel::ellipsize(&self.composer_hint_l, left_cols);
        let right = crate::textsel::ellipsize(&self.composer_hint_r, right_cols);
        self.draw_hint_left(hx, hy, &left);
        if !right.is_empty() {
            let rlen = right.chars().count() as u64 * cw;
            self.draw_str(hx + hw.saturating_sub(rlen), hy, &right, self.theme.composer_hint, self.chat.bg);
        }
        // Suggestion menu sits above the composer (slash commands / @files).
        self.draw_suggest_popup();
    }

    /// Draw the left hint, colouring its first `composer_hint_l_lead.len()`
    /// characters from that list (the shell's progress bar) and the rest in
    /// `theme.composer_hint`.
    ///
    /// The per-cell colours are dropped when `left` came back shorter than the
    /// lead run: a narrow pane ellipsizes the hint, and painting the gradient
    /// onto whatever characters survived would colour the label instead of the
    /// bar.
    fn draw_hint_left(&self, hx: u64, hy: u64, left: &str) {
        let lead = self.composer_hint_l_lead.len();
        if lead == 0 || left.chars().count() < lead {
            self.draw_str(hx, hy, left, self.theme.composer_hint, self.chat.bg);
            return;
        }
        let mut x = hx;
        let mut split = left.len();
        let mut buf = [0u8; 4];
        for (i, (bi, ch)) in left.char_indices().enumerate() {
            if i == lead {
                split = bi;
                break;
            }
            let c = self.composer_hint_l_lead[i];
            x = self.draw_str(x, hy, ch.encode_utf8(&mut buf), c, self.chat.bg);
        }
        self.draw_str(x, hy, &left[split..], self.theme.composer_hint, self.chat.bg);
    }

    /// Geometry of the suggestion popup: `(x, y, w, h)` above the composer.
    fn suggest_geom(&self) -> Option<(u64, u64, u64, u64)> {
        if !self.suggest_open || self.suggest_items.is_empty() || !self.chat.has_composer {
            return None;
        }
        let (bx, by, bw, _bh, _tx, _ty, _hy) = self.composer_geom();
        let n = self.suggest_items.len().min(8) as u64;
        let row_h = self.chat.ch + 4;
        let vpad = 6u64;
        let h = vpad + n * row_h + vpad;
        let y = by.saturating_sub(h + 6);
        // Keep the popup inside the chat pane (below the title header).
        let min_y = self.chat.iy;
        let y = y.max(min_y);
        let h = by.saturating_sub(y + 4).min(h);
        if h < self.chat.ch + vpad {
            return None;
        }
        Some((bx, y, bw, h))
    }

    /// Erase the previous suggestion popup (and the gap down to the composer).
    ///
    /// **Fast path:** fill the old popup rect with `chat.bg` and restore only
    /// the chat rows that intersect it. Avoids a full-pane `render_view` on
    /// every keystroke (that made typing feel multi-hundred-ms laggy).
    ///
    /// When `full_restore` is true (menu fully dismissed), also re-paints
    /// composer chrome without the popup.
    fn suggest_clear_region(&self, full_restore: bool) {
        if self.chat.w == 0 {
            return;
        }
        let (bx, by, bw, _bh, _tx, _ty, _hy) = self.composer_geom();
        let (x, y, w, _h) = self.suggest_rect.unwrap_or_else(|| {
            let h = by.saturating_sub(self.chat.iy).min(bw);
            (bx, by.saturating_sub(h), bw, h)
        });
        let pad = 4u64;
        let left = x.saturating_sub(pad).max(self.chat.x + BORDER);
        let top = y.saturating_sub(pad).max(self.chat.iy);
        let bottom = by;
        let right = (x + w + pad).min(self.chat.x + self.chat.w - BORDER);
        let rw = right.saturating_sub(left);
        let rh = bottom.saturating_sub(top);
        if rw > 0 && rh > 0 {
            self.paint_surface(left, top, rw, rh, self.chat.bg);
        }
        // Restore only grid rows overlapping the erased band (not the whole pane).
        self.render_view_rows_intersecting(top, bottom);
        if full_restore {
            self.paint_composer_box_only();
        }
    }

    /// Composer chrome without the suggestion popup (used when clearing).
    fn paint_composer_box_only(&self) {
        if !self.chat.has_composer || self.chat.w == 0 {
            return;
        }
        let (bx, by, bw, bh, tx, ty, hy) = self.composer_geom();
        self.paint_surface(bx + 1, by + 1, bw.saturating_sub(2), bh.saturating_sub(2), self.theme.composer_bg);
        let border = if self.composer_active && !self.action_focused() {
            self.theme.accent
        } else {
            self.theme.composer_border
        };
        let radius = (4 * self.scale).max(4);
        self.rounded_outline(bx, by, bw, bh, radius, border);
        let prompt: &str = if self.composer_prompt.is_empty() {
            "> "
        } else {
            &self.composer_prompt
        };
        let prompt_cols = prompt.chars().count() as u64;
        let max_cols = ((bw.saturating_sub(16)) / self.chat.cw).saturating_sub(2) as usize;
        let (_vis_start, vis, caret_col) = self.composer_visible(max_cols);
        let mut x = self.draw_str(tx, ty, prompt, self.theme.accent, self.theme.composer_bg);
        x = self.draw_str(x, ty, vis, self.theme.chat_fg, self.theme.composer_bg);
        let rest = (bx + bw).saturating_sub(x + 4);
        if rest > 0 {
            self.paint_surface(x, ty, rest, self.chat.ch, self.theme.composer_bg);
        }
        if self.composer_active && !self.action_focused() {
            let cx = tx + (prompt_cols + caret_col as u64) * self.chat.cw;
            let color = if self.caret_on { self.theme.accent } else { self.theme.composer_bg };
            self.fill_rect(cx, ty, 2 * self.scale.max(1), self.chat.ch, color);
        }
        let hx = bx;
        let hw = bw;
        let cw = self.chat.cw;
        self.fill_cell_bg(hx, hy, hw, self.chat.ch, self.chat.bg);
        let total_cols = (hw / cw).max(1) as usize;
        let gap = 2usize;
        let right_raw = self.composer_hint_r.chars().count();
        let right_cols = right_raw.min(total_cols / 3).min(total_cols.saturating_sub(gap + 4));
        let left_cols = total_cols.saturating_sub(right_cols + if right_cols > 0 { gap } else { 0 });
        let left = crate::textsel::ellipsize(&self.composer_hint_l, left_cols);
        let right = crate::textsel::ellipsize(&self.composer_hint_r, right_cols);
        self.draw_hint_left(hx, hy, &left);
        if !right.is_empty() {
            let rlen = right.chars().count() as u64 * cw;
            self.draw_str(hx + hw.saturating_sub(rlen), hy, &right, self.theme.composer_hint, self.chat.bg);
        }
    }

    /// Paint the slash / @file suggestion list above the composer.
    ///
    /// Text is hard-clamped to the interior of the box so long `@/path/…`
    /// labels never paint past the rounded border (the previous layout gave
    /// labels only `cols/3` then right-aligned a detail that ran under the
    /// pane edge).
    fn draw_suggest_popup(&self) {
        let Some((x, y, w, h)) = self.suggest_geom() else {
            return;
        };
        let ch = self.chat.ch;
        let cw = self.chat.cw;
        let row_h = ch + 4;
        let vpad = 6u64;
        let hpad = 8u64; // left/right inset inside the rounded border
        let bg = self.theme.composer_bg;
        let sel_bg = self.theme.status_bg; // elevated highlight bar
        // Soft fill + border (matches composer chrome).
        self.fill_rect(x, y, w, h, bg);
        let radius = (4 * self.scale).max(4);
        self.rounded_outline(x, y, w, h, radius, self.theme.composer_border);

        // Usable text columns strictly inside the border + padding.
        let inner_x = x + hpad;
        let inner_w = w.saturating_sub(2 * hpad);
        let cols = (inner_w / cw).max(1) as usize;
        let text_right = inner_x + cols as u64 * cw; // last pixel exclusive of next col
        let n = self.suggest_items.len().min(8);
        let mut row_y = y + vpad;
        for i in 0..n {
            if row_y + ch > y + h.saturating_sub(2) {
                break;
            }
            let (ref label, ref detail) = self.suggest_items[i];
            let selected = i == self.suggest_sel;
            let row_bg = if selected { sel_bg } else { bg };
            self.fill_rect(x + 2, row_y.saturating_sub(1), w.saturating_sub(4), row_h, row_bg);

            // Selected: terracotta chevron; 2 columns reserved for the mark.
            let mark = if selected { "> " } else { "  " };
            let mark_fg = if selected { self.theme.accent } else { self.theme.composer_hint };
            let mut px = self.draw_str(inner_x, row_y, mark, mark_fg, row_bg);
            let mark_cols = 2usize;
            let avail = cols.saturating_sub(mark_cols);

            // Column split: short command labels leave room for a muted detail
            // on the right; long `@path` labels take the full row (no detail).
            let has_detail = !detail.is_empty() && label.chars().count() <= avail / 2;
            let det_cols = if has_detail {
                detail.chars().count().min(avail / 3).min(28).max(6)
            } else {
                0
            };
            let lab_cols = avail.saturating_sub(if det_cols > 0 { det_cols + 1 } else { 0 });

            let lab_fg = if selected { self.theme.accent } else { self.theme.chat_fg };
            // Paths: keep the trailing end (`../SOUL.md`); commands: head.
            let lab = if label.starts_with('@') || label.contains('/') {
                crate::textsel::ellipsize_end(label, lab_cols)
            } else {
                crate::textsel::ellipsize(label, lab_cols)
            };
            // Clamp drawn label so it never crosses into the detail zone.
            let lab_max_px = px + lab_cols as u64 * cw;
            px = self.draw_str(px, row_y, &lab, lab_fg, row_bg);
            if px > lab_max_px {
                // Shouldn't happen after ellipsize; blank any overflow residue.
                self.fill_rect(lab_max_px, row_y, px.saturating_sub(lab_max_px), ch, row_bg);
                px = lab_max_px;
            }

            if det_cols > 0 {
                let det = crate::textsel::ellipsize(detail, det_cols);
                let dlen = det.chars().count() as u64 * cw;
                // Right-align detail inside the inner box — never past text_right.
                let dx = text_right.saturating_sub(dlen).max(px + cw);
                if dx + dlen <= text_right && dx + dlen <= x + w.saturating_sub(4) {
                    self.draw_str(dx, row_y, &det, self.theme.composer_hint, row_bg);
                }
            }
            // Wipe any leftover pixels to the right of the last drawn glyph so
            // a previous longer selection highlight doesn't ghost.
            if text_right > px {
                // (row bg already filled; no-op unless we over-drew)
                let _ = px;
            }
            row_y += row_h;
        }
    }
}
