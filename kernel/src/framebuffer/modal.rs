//! Modal overlays: the centred box, confirm/choose/input prompts, the
//! About panel, and the list/commands browsers.

use super::*;

pub(super) fn set_modal_close_rect(r: (u64, u64, u64, u64)) {
    MODAL_CLOSE_RECT.with(|c| *c = r);
}

pub(super) fn clear_modal_close_rect() {
    MODAL_CLOSE_RECT.with(|c| *c = (0, 0, 0, 0));
}

/// Hit-test the modal controls against a click at `(x, y)`.
pub fn modal_hit(x: u64, y: u64) -> ModalHit {
    // Close mark first (dedicated rect) so it wins over the menu-body rect that
    // fully contains it (status dropdown) and over slot-0 Yes (confirm).
    if MODAL_CLOSE_RECT.with(|c| in_rect(x, y, *c)) {
        return ModalHit::Close;
    }
    let r = MODAL_RECTS.with(|m| *m);
    // Confirm: Yes/No in 0/1. List browsers also leave Close in slot 0 with 1/2
    // empty — keep that path for older drawers that only set slot 0.
    if in_rect(x, y, r[0]) {
        if r[1] == (0, 0, 0, 0) && r[2] == (0, 0, 0, 0) {
            return ModalHit::Close;
        }
        return ModalHit::Yes;
    } else if in_rect(x, y, r[1]) {
        return ModalHit::No;
    } else if in_rect(x, y, r[2]) {
        return ModalHit::Ok;
    }
    // List browser rows (/help, /agents).
    if let Some(g) = LIST_BROWSER_GEOM.with(|g| *g) {
        if g.row_h > 0
            && g.list_w > 0
            && x >= g.list_x
            && x < g.list_x + g.list_w
            && y >= g.list_y
            && y < g.list_y + g.row_h * g.n_rows as u64
        {
            let row = ((y - g.list_y) / g.row_h) as usize;
            if row < g.n_rows {
                return ModalHit::ListRow(g.scroll + row);
            }
        }
    }
    // Multi-choice options.
    let n = CHOOSE_COUNT.load(core::sync::atomic::Ordering::Relaxed).min(9);
    if n > 0 {
        let rects = CHOOSE_RECTS.with(|c| *c);
        for i in 0..n {
            if in_rect(x, y, rects[i]) {
                return ModalHit::Choose(i);
            }
        }
    }
    ModalHit::None
}

/// Draw a multi-option question modal. `focus` is the highlighted option index.
/// Options are rendered as numbered rows; the footer shows Enter=select Esc=cancel.
/// Each option row is mouse-clickable ([`ModalHit::Choose`]).
pub fn draw_choose(title: &str, msg: &str, options: &[&str], focus: usize) {
    MODAL_ON.store(true, core::sync::atomic::Ordering::Relaxed);
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.cursor_restore();
            sc.cur_vis = false;
            MODAL_RECTS.with(|m| *m = [(0, 0, 0, 0); 3]);
            LIST_BROWSER_GEOM.with(|g| *g = None);
            CHOOSE_RECTS.with(|c| *c = [(0, 0, 0, 0); 9]);
            let n = options.len().min(9);
            CHOOSE_COUNT.store(n, core::sync::atomic::Ordering::Relaxed);
            let cols = sc.modal_cols() as usize;
            let mut pre: Vec<String> = Vec::new();
            if !msg.is_empty() {
                pre.extend(wrap(msg, cols));
            }
            // One option per row (ellipsized) so hit-testing is 1:1 with indices.
            let mut opt_lines: Vec<String> = Vec::new();
            for (i, opt) in options.iter().take(n).enumerate() {
                let mark = if i == focus { ">" } else { " " };
                let line = alloc::format!("{mark} {}. {}", i + 1, opt);
                opt_lines.push(crate::textsel::ellipsize(&line, cols));
            }
            let foot = "Enter select  Esc cancel  arrows/click";
            // The options are the actionable part and their rows are hit-tested
            // 1:1 with indices, so the *message* absorbs the clamp, never them.
            let pre = crate::panes_layout::clamp_modal_lines(
                pre,
                sc.modal_rows_budget().saturating_sub(opt_lines.len() + 1),
            );
            let rows = pre.len() + opt_lines.len() + 1;
            let (ix, iy, mcols) = sc.modal_box(title, rows as u64);
            let ch = sc.ch();
            let cw = sc.cw();
            let mut y = iy;
            for line in &pre {
                sc.draw_str(ix, y, line, sc.theme.chat_fg, sc.theme.status_bg);
                y += ch;
            }
            for (i, line) in opt_lines.iter().enumerate() {
                let fg = if i == focus {
                    sc.theme.accent
                } else {
                    sc.theme.chat_fg
                };
                let bg = if i == focus {
                    sc.theme.chat_bg
                } else {
                    sc.theme.status_bg
                };
                sc.fill_rect(ix, y, mcols * cw, ch, bg);
                sc.draw_str(ix, y, line, fg, bg);
                CHOOSE_RECTS.with(|c| c[i] = (ix, y, mcols * cw, ch));
                y += ch;
            }
            sc.draw_str(ix, y, foot, sc.theme.composer_hint, sc.theme.status_bg);
            sc.cursor_overlay();
        }
    });
}

/// Draw an approval (yes/no) modal. `focus_yes` highlights the Yes button.
pub fn draw_confirm(title: &str, msg: &str, focus_yes: bool) {
    MODAL_ON.store(true, core::sync::atomic::Ordering::Relaxed);
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.cursor_restore();
            sc.cur_vis = false;
            MODAL_RECTS.with(|m| *m = [(0, 0, 0, 0); 3]);
            LIST_BROWSER_GEOM.with(|g| *g = None);
            CHOOSE_COUNT.store(0, core::sync::atomic::Ordering::Relaxed);
            // Wrap first, then size the box to the wrapped line count + a gap +
            // the button row, so a long consent message never overflows. The
            // budget leaves those two rows for the buttons — an agent's args can
            // be kilobytes (a whole config file), and before the clamp that made
            // the box taller than the screen and the dialog invisible.
            let lines = wrap(msg, sc.modal_cols() as usize);
            let lines = crate::panes_layout::clamp_modal_lines(lines, sc.modal_rows_budget().saturating_sub(2));
            let (ix, iy, cols) = sc.modal_box(title, lines.len() as u64 + 2);
            let ch = sc.ch();
            let cw = sc.cw();
            let mut y = iy;
            for line in &lines {
                sc.draw_str(ix, y, line, sc.theme.chat_fg, sc.theme.status_bg);
                y += ch;
            }
            // Buttons on the RIGHT of the box, just below the message.
            let by = y + ch / 2;
            let btn_w = |label: &str| (label.len() as u64 + 2) * cw;
            let total = btn_w("Yes") + cw + btn_w("No");
            let start = ix + cols * cw - total;
            let x2 = sc.modal_button(start, by, "Yes", focus_yes, 0);
            sc.modal_button(x2, by, "No", !focus_yes, 1);
            sc.cursor_overlay();
        }
    });
}

/// Draw a macOS-style **About ChittiOS** dialog: logo, version, build, arch,
/// tagline, and an OK button (plus FA close). Clicking the status-bar **logo** or
/// running `/about` opens this.
pub fn draw_about() {
    MODAL_ON.store(true, core::sync::atomic::Ordering::Relaxed);
    SCREEN.with(|slot| {
        let Some(sc) = slot else { return };
        sc.cursor_restore();
        sc.cur_vis = false;
        MODAL_RECTS.with(|m| *m = [(0, 0, 0, 0); 3]);
        clear_modal_close_rect();
        LIST_BROWSER_GEOM.with(|g| *g = None);
        CHOOSE_COUNT.store(0, core::sync::atomic::Ordering::Relaxed);

        let cw = sc.cw();
        let ch = sc.ch();
        // Size the card from content so the tagline + OK never clip the border.
        let cols = ((sc.width / cw) * 2 / 5).clamp(30, 42);
        let logo_r = (ch * 2).max(18);
        // close pad + logo (2r) + gaps + name + ver + built + arch + sep + 2 tags + OK + pads
        let content_h = ch // top pad / close row
            + logo_r * 2
            + ch // gap under logo
            + ch // name
            + ch / 4
            + ch // version
            + ch // built
            + ch // arch
            + ch / 2 // sep gap
            + 2 // sep
            + ch / 2
            + ch // tag1
            + ch // tag2
            + ch / 2
            + ch // OK
            + ch / 2; // bottom pad
        let bw = cols * cw + 2 * (BORDER + PAD);
        let bh = content_h + 2 * (BORDER + PAD);
        let bx = (sc.width - bw) / 2;
        let by = (sc.height - bh) / 2;
        let bg = sc.theme.status_bg;
        sc.drop_shadow(bx, by, bw, bh);
        sc.fill_rect(bx, by, bw, bh, bg);
        sc.rect_outline(bx, by, bw, bh, BORDER, sc.theme.accent);

        let ix = bx + BORDER + PAD;
        let content_w = cols * cw;
        // Close (FA xmark) top-right — dedicated hit rect so it always works.
        let mark = crate::icons::close_mark();
        let (close_w, _) = sc.glyph_cell(mark);
        let close_w = close_w.max(cw * 2);
        let cx = ix + content_w.saturating_sub(close_w);
        let close_y = by + BORDER + PAD / 2;
        let (iw, _) = sc.glyph_cell(mark);
        sc.blit_glyph(
            cx + close_w.saturating_sub(iw) / 2,
            close_y,
            mark,
            sc.theme.accent,
            bg,
        );
        set_modal_close_rect((cx, close_y, close_w, ch));

        // Large brand logo (ui.json theme.logo / logo_node).
        let logo_cy = by + BORDER + PAD + ch / 2 + logo_r;
        sc.draw_logo(
            bx + bw / 2,
            logo_cy,
            logo_r,
            sc.theme.logo,
            sc.theme.logo_node,
        );

        let mut y = logo_cy + logo_r + ch / 2;
        let centre = |s: &str| bx + (bw.saturating_sub(s.chars().count() as u64 * cw)) / 2;

        sc.draw_str(centre("ChittiOS"), y, "ChittiOS", sc.theme.logo, bg);
        y += ch + ch / 4;

        let ver = alloc::format!("Version {}", crate::VERSION);
        sc.draw_str(centre(&ver), y, &ver, sc.theme.chat_fg, bg);
        y += ch;

        let built = alloc::format!("Built {}", crate::BUILD_TIME);
        sc.draw_str(centre(&built), y, &built, sc.theme.title_dim, bg);
        y += ch;

        #[cfg(target_arch = "x86_64")]
        let arch = "x86_64";
        #[cfg(target_arch = "aarch64")]
        let arch = "aarch64";
        let arch_line = alloc::format!("{arch}  ·  {} cores", crate::arch::cpu_count());
        sc.draw_str(centre(&arch_line), y, &arch_line, sc.theme.title_dim, bg);
        y += ch + ch / 2;

        sc.fill_rect(ix + cw * 2, y, content_w.saturating_sub(cw * 4), 1, sc.theme.sep_dim);
        y += ch / 2 + 2;

        let tag = "An agentic operating system.";
        sc.draw_str(centre(tag), y, tag, sc.theme.chat_fg, bg);
        y += ch;
        let tag2 = "The agent is the driver.";
        sc.draw_str(centre(tag2), y, tag2, sc.theme.title_dim, bg);
        y += ch + ch / 2;

        // OK button, centred, inside the card.
        let btn_w = 4 * cw;
        let btn_x = bx + (bw.saturating_sub(btn_w)) / 2;
        sc.modal_button(btn_x, y, "OK", true, 2);
        sc.cursor_overlay();
    });
}

/// Draw a text-input modal (masked = password dots). `caret_on` blinks the caret.
pub fn draw_input(title: &str, prompt: &str, buf: &str, masked: bool, caret_on: bool) {
    MODAL_ON.store(true, core::sync::atomic::Ordering::Relaxed);
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.cursor_restore();
            sc.cur_vis = false;
            MODAL_RECTS.with(|m| *m = [(0, 0, 0, 0); 3]);
            LIST_BROWSER_GEOM.with(|g| *g = None);
            CHOOSE_COUNT.store(0, core::sync::atomic::Ordering::Relaxed);
            // 4 content rows: the prompt, the input field, and the OK button each
            // occupy a row and the inter-row gaps (ch/2 + a few px) need the
            // extra row so the button clears the bottom border.
            let (ix, iy, cols) = sc.modal_box(title, 4);
            let ch = sc.ch();
            let cw = sc.cw();
            sc.draw_str(ix, iy, prompt, sc.theme.title_dim, sc.theme.status_bg);
            // Input field: a framed row showing the (optionally masked) text.
            let fy = iy + ch + 4;
            sc.fill_rect(ix, fy, cols * cw, ch, sc.theme.chat_bg);
            sc.rect_outline(ix, fy, cols * cw, ch, 1, sc.theme.border_dim);
            let shown: String = if masked { core::iter::repeat('*').take(buf.chars().count()).collect() } else { buf.to_string() };
            let end = sc.draw_str(ix + cw / 2, fy, &shown, sc.theme.chat_fg, sc.theme.chat_bg);
            if caret_on {
                sc.fill_rect(end, fy, 2 * sc.scale, ch, sc.theme.accent);
            }
            let by = fy + ch + ch / 2;
            sc.modal_button(ix, by, "OK", true, 2);
            sc.cursor_overlay();
        }
    });
}

/// Draw the **Commands** browser modal (opened by `/help`): title + search
/// field, scrollable categorised list, scrollbar, footer hints.
///
/// `rows` is the **visible slice** (already scrolled). `query` is the search
/// box contents; `caret_on` blinks the search caret; `scroll`/`total` drive the
/// scrollbar thumb.
/// Draw the searchable list modal used by `/help` (Commands) and `/agents`.
/// `title` is the window chrome label (e.g. `"Commands"` / `"Agents"`).
pub fn draw_commands_browser(
    query: &str,
    rows: &[CommandsRow<'_>],
    scroll: usize,
    total: usize,
    caret_on: bool,
) {
    draw_list_browser("Commands", query, rows, scroll, total, caret_on);
}

/// Same as [`draw_commands_browser`] with a custom window title.
pub fn draw_list_browser(
    title: &str,
    query: &str,
    rows: &[CommandsRow<'_>],
    scroll: usize,
    total: usize,
    caret_on: bool,
) {
    MODAL_ON.store(true, core::sync::atomic::Ordering::Relaxed);
    SCREEN.with(|slot| {
        let Some(sc) = slot else { return };
        sc.cursor_restore();
        sc.cur_vis = false;
        MODAL_RECTS.with(|m| *m = [(0, 0, 0, 0); 3]);
        CHOOSE_COUNT.store(0, core::sync::atomic::Ordering::Relaxed);

        let cw = sc.cw();
        let ch = sc.ch();
        let bg = sc.theme.status_bg;
        let list_bg = sc.theme.chat_bg;
        // Roomier than the default confirm modal — ~ half the screen.
        let cols = ((sc.width / cw) * 5 / 10).clamp(36, 64);
        let view_rows = 12u64; // visible list lines
        let chrome_rows = 5u64; // title + search + gap + footer
        let rows_h = view_rows + chrome_rows;
        let bw = cols * cw + 2 * (BORDER + PAD);
        let bh = rows_h * ch + 2 * (BORDER + PAD) + 12;
        let bx = (sc.width - bw) / 2;
        let by = (sc.height - bh) / 2;
        // Do **not** full-screen `shade_rect` here: the caret blink repaints this
        // every ~500 ms, and shade multiplies darkness each pass → cascading
        // black + flicker. Same solid modal box as `/help` (draw_commands_browser).
        sc.drop_shadow(bx, by, bw, bh);
        sc.fill_rect(bx, by, bw, bh, bg);
        sc.rect_outline(bx, by, bw, bh, BORDER, sc.theme.accent);

        let ix = bx + BORDER + PAD;
        let mut y = by + BORDER + PAD;
        let content_w = cols * cw;

        // Title + FA xmark close (theme accent — same as pane close).
        let mark = crate::icons::close_mark();
        let (close_w, _) = sc.glyph_cell(mark);
        let close_w = close_w.max(cw * 2);
        sc.draw_str(
            ix,
            y,
            &crate::textsel::ellipsize(
                title,
                cols.saturating_sub((close_w / cw).max(2)) as usize,
            ),
            sc.theme.accent,
            bg,
        );
        // Hit target ≥ square FA cell; glyph centred like the pane close chrome.
        let cx = ix + content_w.saturating_sub(close_w);
        let (iw, _) = sc.glyph_cell(mark);
        sc.blit_glyph(cx + close_w.saturating_sub(iw) / 2, y, mark, sc.theme.accent, bg);
        set_modal_close_rect((cx, y, close_w, ch));
        // Keep slot 0 empty so Close is only the dedicated rect (not Yes).
        MODAL_RECTS.with(|m| m[0] = (0, 0, 0, 0));
        y += ch + 4;
        sc.fill_rect(ix, y, content_w, 1, sc.theme.sep_dim);
        y += 6;

        // Search field (FA magnifying-glass + label).
        let search_lab = alloc::format!("{} search", crate::icons::fa::SEARCH);
        sc.draw_str(ix, y, &search_lab, sc.theme.title_dim, bg);
        // Label width: FA cell (= line height) + " search" (7 mono cells).
        let lab_w = ch + 7 * cw;
        let field_x = ix + lab_w + cw / 2;
        let field_w = content_w.saturating_sub(lab_w + cw / 2);
        sc.fill_rect(field_x, y, field_w, ch, list_bg);
        sc.rect_outline(field_x, y, field_w, ch, 1, sc.theme.border_dim);
        let qshow = crate::textsel::ellipsize(query, (field_w / cw).saturating_sub(1) as usize);
        let qend = sc.draw_str(field_x + 4, y, &qshow, sc.theme.chat_fg, list_bg);
        if caret_on {
            sc.fill_rect(qend, y, 2 * sc.scale.max(1), ch, sc.theme.accent);
        }
        y += ch + 6;

        // List region.
        let list_top = y;
        let list_h = view_rows * ch;
        let list_w = content_w.saturating_sub(cw); // leave a col for scrollbar
        sc.fill_rect(ix, list_top, list_w, list_h, list_bg);

        let mut ly = list_top;
        let visible_n = rows.len().min(view_rows as usize);
        for row in rows.iter().take(view_rows as usize) {
            match row {
                CommandsRow::Header(h) => {
                    let line = crate::textsel::ellipsize(h, (list_w / cw) as usize);
                    sc.draw_str(ix + 2, ly, &line, sc.theme.title_dim, list_bg);
                    // Dim rule under the category label.
                    sc.fill_rect(ix + 2, ly + ch - 2, list_w.saturating_sub(4), 1, sc.theme.sep_dim);
                }
                CommandsRow::Item {
                    title,
                    slash,
                    shortcut,
                    selected,
                } => {
                    let row_bg = if *selected { sc.theme.status_bg } else { list_bg };
                    sc.fill_rect(ix, ly, list_w, ch, row_bg);
                    let mark = if *selected { "> " } else { "* " };
                    let mark_fg = if *selected { sc.theme.accent } else { sc.theme.composer_hint };
                    let mut px = sc.draw_str(ix + 2, ly, mark, mark_fg, row_bg);
                    let title_fg = if *selected { sc.theme.accent } else { sc.theme.chat_fg };
                    // Right column: shortcut if present, else /name.
                    let right = if !shortcut.is_empty() {
                        *shortcut
                    } else {
                        *slash
                    };
                    let right_cols = right.chars().count().min(18);
                    let left_cols = (list_w / cw)
                        .saturating_sub(3 + right_cols as u64 + 2) as usize;
                    let t = crate::textsel::ellipsize(title, left_cols);
                    px = sc.draw_str(px, ly, &t, title_fg, row_bg);
                    let rtxt = crate::textsel::ellipsize(right, right_cols);
                    let rlen = rtxt.chars().count() as u64 * cw;
                    let rx = ix + list_w.saturating_sub(rlen + 4);
                    if rx > px {
                        sc.draw_str(rx, ly, &rtxt, sc.theme.composer_hint, row_bg);
                    }
                }
            }
            ly += ch;
        }
        // Mouse hit-testing for every visible row (headers included; caller
        // skips non-items). Absolute index = scroll + visible row.
        LIST_BROWSER_GEOM.with(|g| {
            *g = Some(ListBrowserGeom {
                list_x: ix,
                list_y: list_top,
                list_w,
                row_h: ch,
                n_rows: visible_n,
                scroll,
            });
        });

        // Scrollbar (right edge of list).
        let sb_x = ix + list_w + 2;
        let sb_h = list_h;
        sc.fill_rect(sb_x, list_top, 3 * sc.scale.max(1), sb_h, sc.theme.composer_border);
        if total > view_rows as usize && total > 0 {
            let thumb_h = ((sb_h as usize * view_rows as usize) / total)
                .max(ch as usize)
                .min(sb_h as usize) as u64;
            let max_scroll = total.saturating_sub(view_rows as usize).max(1);
            let thumb_y = list_top
                + ((sb_h.saturating_sub(thumb_h)) as usize * scroll / max_scroll) as u64;
            sc.fill_rect(sb_x, thumb_y, 3 * sc.scale.max(1), thumb_h, sc.theme.accent);
        }

        // Footer.
        y = list_top + list_h + 6;
        sc.fill_rect(ix, y, content_w, 1, sc.theme.sep_dim);
        y += 4;
        let foot = if title.eq_ignore_ascii_case("Agents") {
            "up/dn  |  Enter/click select  |  Esc close"
        } else {
            "up/dn  |  Enter/click fill  |  Esc close"
        };
        sc.draw_str(
            ix,
            y,
            &crate::textsel::ellipsize(foot, cols as usize),
            sc.theme.composer_hint,
            bg,
        );
        sc.cursor_overlay();
    });
}

/// Draw the `/voice` modal: a live waveform (one vertical bar per recent RMS
/// level, newest on the right) above a status line and a Stop button. Called
/// every capture frame, so it repaints only the modal region.
pub fn draw_voice(levels: &[f32], status: &str) {
    MODAL_ON.store(true, core::sync::atomic::Ordering::Relaxed);
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.cursor_restore();
            sc.cur_vis = false;
            MODAL_RECTS.with(|m| *m = [(0, 0, 0, 0); 3]);
            let (ix, iy, cols) = sc.modal_box("Voice", 8);
            let ch = sc.ch();
            let cw = sc.cw();
            // Waveform region: 5 text rows tall, one 3px bar per level.
            let wave_h = 5 * ch;
            let wave_w = cols * cw;
            sc.fill_rect(ix, iy, wave_w, wave_h, sc.theme.chat_bg);
            let barw = 3 * sc.scale.max(1);
            let nbars = (wave_w / (barw + sc.scale)) as usize;
            let take = levels.len().min(nbars);
            let mid = iy + wave_h / 2;
            for (i, &lv) in levels[levels.len() - take..].iter().enumerate() {
                // Bar height from the RMS level (log-ish response for visibility).
                let l = if lv < 0.0 { 0.0 } else if lv > 1.0 { 1.0 } else { lv };
                let boost = l * (2.0 - l); // gentle curve
                let h = ((wave_h / 2) as f32 * boost) as u64 + sc.scale;
                let x = ix + (i as u64) * (barw + sc.scale);
                sc.fill_rect(x, mid - h, barw, 2 * h, sc.theme.accent);
            }
            // Status line + Stop button.
            let sy = iy + wave_h + ch / 2;
            sc.fill_rect(ix, sy, wave_w, ch, sc.theme.status_bg);
            sc.draw_str(ix, sy, status, sc.theme.title_dim, sc.theme.status_bg);
            sc.modal_button(ix, sy + ch + ch / 2, "Stop", true, 2);
            sc.cursor_overlay();
        }
    });
}

/// Dismiss any modal and repaint the normal UI.
pub fn modal_dismiss() {
    MODAL_ON.store(false, core::sync::atomic::Ordering::Relaxed);
    MODAL_RECTS.with(|m| *m = [(0, 0, 0, 0); 3]);
    clear_modal_close_rect();
    LIST_BROWSER_GEOM.with(|g| *g = None);
    CHOOSE_COUNT.store(0, core::sync::atomic::Ordering::Relaxed);
    CHOOSE_RECTS.with(|c| *c = [(0, 0, 0, 0); 9]);
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.redraw();
            sc.cur_vis = false;
        }
    });
}

impl Screen {
    /// Draw a centred modal box and return its interior text origin + width in
    /// cells `(ix, iy, cols)`. Dims the screen isn't done (kept cheap); the box
    /// simply overpaints the middle of the canvas.
    /// The modal content width in cells (roomy but bounded), so callers can wrap
    /// their text to it *before* sizing the box height.
    fn modal_cols(&self) -> u64 {
        ((self.width / self.cw()) * 3 / 5).clamp(28, 56)
    }

    /// Content rows a centred modal can hold on this screen (title + separator
    /// and the frame already deducted, one cell of margin kept top and bottom).
    /// The math is pure and lives in `panes_layout` so it is unit-tested — this
    /// module is `#[cfg(not(test))]`, so a test in here would never run.
    fn modal_rows_budget(&self) -> usize {
        crate::panes_layout::modal_max_rows(self.height, self.ch(), 2 * (BORDER + PAD)) as usize
    }

    fn modal_box(&self, title: &str, rows: u64) -> (u64, u64, u64) {
        let cw = self.cw();
        let ch = self.ch();
        let cols = self.modal_cols();
        let bw = cols * cw + 2 * (BORDER + PAD);
        let bh = (rows + 2) * ch + 2 * (BORDER + PAD);
        // Saturating, never `self.height - bh`: a box taller than the screen
        // wrapped that subtraction into a vast `by`, so every draw was clipped
        // away and the modal painted **nothing** while still waiting for a key.
        // An approval dialog that is invisible but live is the worst failure
        // this code has — the human cannot see what they are approving, and it
        // reads as a frozen shell. Callers must also budget their rows
        // ([`modal_max_rows`]); this is the backstop.
        let bx = self.width.saturating_sub(bw) / 2;
        let by = self.height.saturating_sub(bh) / 2;
        self.drop_shadow(bx, by, bw, bh); // web-style elevation over the panes
        self.fill_rect(bx, by, bw, bh, self.theme.status_bg);
        self.rect_outline(bx, by, bw, bh, BORDER, self.theme.accent);
        let ix = bx + BORDER + PAD;
        let iy = by + BORDER + PAD;
        self.draw_str(ix, iy, title, self.theme.accent, self.theme.status_bg);
        self.fill_rect(ix, iy + ch + 2, cols * cw, 1, self.theme.sep_dim);
        (ix, iy + 2 * ch, cols)
    }

    /// Draw a labelled button at `(x, y)`, filled when `focused`; record its rect
    /// in `MODAL_RECTS[slot]` for mouse hit-testing. Returns the x just past it.
    fn modal_button(&self, x: u64, y: u64, label: &str, focused: bool, slot: usize) -> u64 {
        let cw = self.cw();
        let ch = self.ch();
        let w = (label.len() as u64 + 2) * cw;
        let (fg, bg) = if focused { (self.theme.status_bg, self.theme.accent) } else { (self.theme.accent, self.theme.status_bg) };
        self.fill_rect(x, y, w, ch, bg);
        self.rect_outline(x, y, w, ch, 1, self.theme.accent);
        self.draw_str(x + cw, y, label, fg, bg);
        MODAL_RECTS.with(|m| m[slot] = (x, y, w, ch));
        x + w + cw
    }
}
