//! The notification banner — a transient overlay in the top-right corner.
//!
//! **This is the only surface in the compositor that draws over live panes**, so
//! it works the way the mouse cursor does and for the same reason: the
//! framebuffer is single-buffered, so the pixels beneath have to be saved before
//! and put back after. `toast_saved` is `cursor_saved` with a different name.
//!
//! Everything decidable without a framebuffer — the width, the wrapping, the
//! dwell, the chime, the policy — is in [`crate::notify::toast`], which is
//! testable. This module places a rectangle and draws into it.
//!
//! Two placement rules, both of which are the reason a banner is acceptable here
//! at all:
//!
//! - **Top-right of the content rect**, never the composer. The composer is at
//!   the bottom and is where the human is typing; the top-right is the status
//!   bar's own strip and a pane title row.
//! - **It moves out of the status bar's way.** The bar sits on any edge
//!   (`/statusbar top|bottom|left|right`), and `Screen::content_*` is already the
//!   rect *inside* it — so anchoring to the content rect is automatically correct
//!   for all four positions, with no `match` on the edge.

use super::*;

/// Outer margin from the content rect's top-right corner.
const MARGIN: u64 = 10;
/// Padding inside the banner box.
const PAD: u64 = 8;

/// Show a banner. Saves whatever is underneath first, so [`toast_hide`] can put
/// it back exactly.
pub fn toast_show(sev: crate::notify::Severity, head: &str, lines: &[alloc::string::String]) {
    SCREEN.with(|slot| {
        let Some(sc) = slot else { return };
        // Lift a previous banner before measuring, or its pixels become part of
        // what the new one saves and restoring would paint the old banner back.
        sc.toast_restore();
        sc.cursor_restore();
        sc.cur_vis = false;

        let (cw, ch) = (sc.cw(), sc.ch());
        let cols = crate::notify::toast::width_cols((sc.content_w / cw.max(1)) as usize) as u64;
        let rows = lines.len() as u64 + 1; // + the heading row
        let w = cols * cw + 2 * PAD;
        let h = rows * ch + 2 * PAD + ch / 3;

        // Top-right **of the content rect**, so the bar's edge is handled without
        // a match on which edge it is on.
        let x = (sc.content_x + sc.content_w).saturating_sub(w + MARGIN).max(sc.content_x);
        let y = sc.content_y + MARGIN;
        let w = w.min(sc.content_w.saturating_sub(2 * MARGIN)).max(cw);
        let h = h.min(sc.content_h.saturating_sub(2 * MARGIN)).max(ch);

        sc.toast_save(x, y, w, h);

        // The box. `paint_surface`, not a raw `fill_rect` of the background —
        // under a translucent wallpaper the desktop must show through the banner
        // exactly as it does through a pane.
        let bg = sc.theme.composer_bg;
        sc.paint_surface(x, y, w, h, bg);
        // Chrome may be faint (it is a border, not a word); text may not.
        let chrome = ink(sc, crate::notify::toast::chrome_ink(sev));
        sc.rounded_outline(x, y, w, h, (4 * sc.scale).max(4), chrome);
        // A severity stripe down the left edge: the one piece of colour that
        // says *what kind* of notification this is before any text is read. Drawn
        // in the *heading* ink rather than the chrome ink, so it is visible even
        // for an `Info` whose border is deliberately quiet.
        let head_ink = ink(sc, crate::notify::toast::heading_ink(sev));
        sc.fill_rect(x + 2, y + 3, (2 * sc.scale).max(2), h.saturating_sub(6), head_ink);

        let tx = x + PAD + (3 * sc.scale).max(3);
        let mut ty = y + PAD;
        // Heading: the severity glyph plus the kernel-stamped source.
        let icon = crate::notify::severity_icon(sev);
        let heading = alloc::format!("{icon} {head}");
        sc.draw_str(tx, ty, &heading, head_ink, bg);
        ty += ch + ch / 3;
        let body_ink = ink(sc, crate::notify::toast::body_ink(sev));
        for l in lines {
            sc.draw_str(tx, ty, l, body_ink, bg);
            ty += ch;
        }
        sc.cursor_overlay();
        // One damage rect for the whole banner — and one more when it lifts. A
        // banner that does not animate costs two KMS round trips, not one per
        // frame, which is what makes a transient overlay affordable here.
        crate::kms::damage(
            (x + sc.origin_x) as u32,
            (y + sc.origin_y) as u32,
            w as u32,
            h as u32,
        );
    });
}

/// Lift the banner, restoring the pixels underneath.
pub fn toast_hide() {
    SCREEN.with(|slot| {
        let Some(sc) = slot else { return };
        if sc.toast_w == 0 {
            return;
        }
        sc.cursor_restore();
        sc.cur_vis = false;
        let (x, y, w, h) = (sc.toast_x, sc.toast_y, sc.toast_w, sc.toast_h);
        sc.toast_restore();
        sc.cursor_overlay();
        crate::kms::damage(
            (x + sc.origin_x) as u32,
            (y + sc.origin_y) as u32,
            w as u32,
            h as u32,
        );
    });
}

/// Whether a banner is currently on screen — so a repaint that would overwrite
/// it (a relayout, a theme change) can lift it first.
pub fn toast_visible() -> bool {
    SCREEN.with(|slot| slot.as_ref().map(|sc| sc.toast_w != 0).unwrap_or(false))
}

/// The desktop's width in columns, for sizing the banner.
pub fn desktop_cols() -> usize {
    SCREEN
        .with(|slot| slot.as_ref().map(|sc| (sc.content_w / sc.cw().max(1)) as usize))
        .unwrap_or(80)
}

/// Resolve a colour **role** against the live theme.
///
/// The mapping is here and the *choice* of role is in `notify::toast`, which is
/// testable — because the first version of this picked `theme.sep_dim` for an
/// `Info` heading, and `sep_dim` (`#2e2c28`) on a `composer_bg` (`#252320`) box is
/// text that is present and unreadable. Which text may be faint is a decision; a
/// decision belongs somewhere a test can hold it.
///
/// Deliberately theme colours throughout rather than baked amber/red: `/theme`
/// must be able to recolour every surface, and a banner with its own palette
/// would be the one thing that ignored it.
fn ink(sc: &Screen, role: crate::notify::toast::Ink) -> Rgb {
    use crate::notify::toast::Ink;
    match role {
        Ink::Accent => sc.theme.accent,
        Ink::Normal => sc.theme.chat_fg,
        Ink::Soft => sc.theme.logs_fg,
        Ink::Faint => sc.theme.composer_border,
    }
}

impl Screen {
    /// Save the pixels under the banner rect.
    pub(super) fn toast_save(&mut self, x: u64, y: u64, w: u64, h: u64) {
        self.toast_saved.clear();
        self.toast_saved.reserve((w * h) as usize);
        for dy in 0..h {
            for dx in 0..w {
                self.toast_saved.push(self.get_pixel(x + dx, y + dy));
            }
        }
        self.toast_x = x;
        self.toast_y = y;
        self.toast_w = w;
        self.toast_h = h;
    }

    /// Put back the pixels the banner covered, and forget it.
    pub(super) fn toast_restore(&mut self) {
        if self.toast_w == 0 {
            return;
        }
        let (x, y, w, h) = (self.toast_x, self.toast_y, self.toast_w, self.toast_h);
        for dy in 0..h {
            for dx in 0..w {
                let i = (dy * w + dx) as usize;
                if i < self.toast_saved.len() {
                    self.put_pixel(x + dx, y + dy, self.toast_saved[i]);
                }
            }
        }
        self.toast_saved.clear();
        self.toast_w = 0;
        self.toast_h = 0;
    }

    /// Drop the banner **without** repainting what was under it.
    ///
    /// For a full redraw, which is about to paint those pixels itself: restoring
    /// first would put a stale copy of the banner's background on screen, and a
    /// relayout may have moved everything anyway.
    pub(super) fn toast_forget(&mut self) {
        self.toast_saved.clear();
        self.toast_w = 0;
        self.toast_h = 0;
    }
}
