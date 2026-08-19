//! The status bar: its geometry on any edge, the chip hit rects, and the
//! live right-hand chip values.

use super::*;

/// Padding around the status bar's text, on both sides of its short axis.
const STATUS_PAD: u64 = 10;

/// Extra vertical room in the horizontal status bar so FA icons sit fully inside
/// the bar with air above/below body text (not clipped against the edge).
const STATUS_ICON_EXTRA: u64 = 6;

/// How thick the status bar is on a given edge, in pixels.
///
/// Horizontal (top/bottom): text row + icon headroom + padding.
/// Vertical (left/right): a fixed [`crate::panes_layout::STATUS_V_COLS`]-column
/// span, because text cannot run across a column and its content stacks instead.
pub(super) fn status_thickness(pos: crate::panes_layout::StatusPos, cw: u64, ch: u64) -> u64 {
    if pos.vertical() {
        crate::panes_layout::STATUS_V_COLS * cw + STATUS_PAD
    } else {
        ch + STATUS_ICON_EXTRA + STATUS_PAD
    }
}

/// True for status-bar icons we draw slightly larger (Font Awesome PUA).
fn is_status_icon(ch: char) -> bool {
    crate::icons::is_icon(ch)
}

fn clear_status_chip_rects() {
    STATUS_CHIP_RECTS.with(|a| *a = [(0, 0, 0, 0); STATUS_CHIP_N]);
}

fn set_status_chip_rect(chip: StatusChip, r: (u64, u64, u64, u64)) {
    STATUS_CHIP_RECTS.with(|a| a[chip as usize] = r);
}

/// True if `(x, y)` is on the status-bar **logo** (About hit target; not the wordmark).
pub fn status_brand_hit(x: u64, y: u64) -> bool {
    status_chip_hit(x, y) == Some(StatusChip::Brand)
}

/// Which status-bar chip is under `(x, y)`, if any (inactive while a modal is up).
pub fn status_chip_hit(x: u64, y: u64) -> Option<StatusChip> {
    if MODAL_ON.load(core::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    STATUS_CHIP_RECTS.with(|a| {
        for i in 0..STATUS_CHIP_N {
            if in_rect(x, y, a[i]) {
                return Some(match i {
                    0 => StatusChip::Brand,
                    1 => StatusChip::Kbd,
                    2 => StatusChip::Mouse,
                    3 => StatusChip::Net,
                    4 => StatusChip::Mem,
                    5 => StatusChip::Cpu,
                    6 => StatusChip::Battery,
                    7 => StatusChip::Volume,
                    8 => StatusChip::Clock,
                    9 => StatusChip::Notifications,
                    10 => StatusChip::Recording,
                    11 => StatusChip::NowPlaying,
                    _ => return None,
                });
            }
        }
        None
    })
}

/// Pixel advance of a status string (icons use square line-height cells).
fn status_str_advance(s: &str, cw: u64, ch: u64) -> u64 {
    let icon_cw = ch.max(cw);
    let mut w = 0u64;
    for c in s.chars() {
        w += if is_status_icon(c) { icon_cw } else { cw };
    }
    w
}

/// Live right-side status chips (same content the bar paints, in order).
///
/// `vertical` is passed in rather than re-read from `SCREEN`: this is called
/// from inside `Screen::draw_status`, which already holds that lock.
fn status_right_chips(
    vertical: bool,
) -> alloc::vec::Vec<(StatusChip, alloc::string::String, crate::icons::DeviceStatus)> {
    use crate::icons::DeviceStatus;
    let mut out = alloc::vec::Vec::new();
    let last_k = crate::console::input_activity_ms();
    let k_active = last_k != 0 && crate::arch::now_ms().saturating_sub(last_k) < 1500;
    let k_st = crate::console::keyboard_status();
    out.push((StatusChip::Kbd, crate::icons::status_kbd(k_st, k_active), k_st));
    let last_m = crate::mouse::activity_ms();
    let m_active = last_m != 0 && crate::arch::now_ms().saturating_sub(last_m) < 1500;
    let m_st = crate::mouse::pointer_status();
    out.push((StatusChip::Mouse, crate::icons::status_mouse(m_st, m_active), m_st));
    let n_st = crate::net::device_status();
    out.push((StatusChip::Net, crate::icons::status_net(n_st), n_st));
    // mem / cpu / cores — match ui_config::resolve_var labels
    {
        let m = crate::mm::mem_stats();
        let mib = 1024 * 1024;
        let gib = 1024 * mib;
        let mem = if m.ram_total >= gib {
            alloc::format!(
                "mem {}M/{}.{}G",
                (m.heap_used + (m.ram_reserved - m.heap_total)) / mib,
                m.ram_total / gib,
                (m.ram_total % gib) * 10 / gib
            )
        } else {
            alloc::format!("mem {}/{}M", m.heap_used / mib, m.ram_total / mib)
        };
        out.push((StatusChip::Mem, mem, DeviceStatus::Ready));
    }
    out.push((
        StatusChip::Cpu,
        alloc::format!(
            "cpu {:>3}% {}c",
            crate::shell::cpu_percent(),
            crate::arch::cpu_count()
        ),
        DeviceStatus::Ready,
    ));
    if let Some(b) = crate::drivers::battery::cached() {
        let s = crate::drivers::battery::format(&b);
        if !s.is_empty() {
            out.push((StatusChip::Battery, s, DeviceStatus::Ready));
        }
    }
    // Volume always shown — disabled (dim, x-mark) until a PCM device is up.
    let v_st = crate::sound::device_status();
    out.push((
        StatusChip::Volume,
        crate::icons::status_volume(v_st, crate::sound::muted(), crate::sound::volume()),
        v_st,
    ));
    // Now-playing: a stack row on a vertical bar. A horizontal bar paints
    // the same text as a centred pill in `draw_status_horizontal` so it is
    // not also pushed here (that would draw it twice).
    if vertical {
        let np = crate::shell::now_playing_chip();
        if !np.is_empty() {
            out.push((StatusChip::NowPlaying, np, DeviceStatus::Ready));
        }
    }
    // Only shown when there is something unread — the Battery precedent above.
    // A machine with nothing to say has a byte-identical status bar, and
    // `ui_config::resolve_var` returns the same empty string so the template
    // drops the separator too.
    let unread = crate::notify::chip_text(crate::notify::unread_count());
    if !unread.is_empty() {
        out.push((StatusChip::Notifications, unread, DeviceStatus::Ready));
    }
    // Red ● + elapsed while a take is live; absent otherwise (same empty-chip
    // rule as notifications). Clicking it stops the recording.
    let rec = crate::shell::record::chip_text();
    if !rec.is_empty() {
        out.push((StatusChip::Recording, rec, DeviceStatus::Ready));
    }
    // Compact macOS-style clock (no year / seconds / tz — dropdown has the rest).
    out.push((StatusChip::Clock, crate::clock::format_datetime_short(), DeviceStatus::Ready));
    out
}

fn chip_ink(sc: &Screen, st: crate::icons::DeviceStatus) -> Rgb {
    use crate::icons::DeviceStatus;
    match st {
        DeviceStatus::Ready => sc.theme.status_fg,
        DeviceStatus::Pending => sc.mix(sc.theme.status_fg, sc.theme.status_bg, 0.40),
        DeviceStatus::Disabled => sc.mix(sc.theme.status_fg, sc.theme.status_bg, 0.65),
    }
}

/// Set the status-bar text (left = brand, right = datetime), then repaint just
/// the bar. The shell calls this every second with the UI-config templates
/// resolved against the clock.
pub fn set_status(left: &str, right: &str) {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.cursor_restore();
            sc.cur_vis = false;
            sc.status_left.clear();
            sc.status_left.push_str(left);
            sc.status_right.clear();
            sc.status_right.push_str(right);
            sc.draw_status();
            sc.cursor_overlay();
        }
    });
}

impl Screen {
    pub(super) fn draw_status(&self) {
        // Icons must hit the FA face (first fallback); if registration was
        // skipped or failed earlier, status chips paint as thin tofu bars.
        let _ = crate::font_ttf::register_bundled_fallback(crate::font_ttf::FA_FALLBACK_NAME);
        if self.layout.status_pos.vertical() {
            self.draw_status_vertical();
        } else {
            self.draw_status_horizontal();
        }
    }

    /// Draw status text with icon glyphs (Font Awesome PUA). Body text and the
    /// activity middle-dot stay at the normal cell size; icons use a square cell
    /// equal to the body line height (FA is fit-to-cell so nothing clips).
    fn draw_status_str(&self, mut x: u64, y: u64, s: &str, fg: Rgb, bg: Rgb, max_x: u64) -> u64 {
        let cw = self.cw();
        let ch = self.ch();
        // Square icon cell matching the text line — wider cells clipped the sides
        // of wide FA glyphs (keyboard); taller-than-bar cells clipped top/bottom.
        let icon_cw = ch.max(cw);
        let icon_ch = ch;
        // Vertically centre the body line against the icon row when the bar gave
        // us extra headroom (`y` is already the icon/text band top).
        let body_y = y;
        for ch_c in s.chars() {
            if is_status_icon(ch_c) {
                if x + icon_cw > max_x {
                    break;
                }
                for gy in 0..icon_ch {
                    for gx in 0..icon_cw {
                        self.put_pixel(x + gx, y + gy, bg);
                    }
                }
                let mut painted = false;
                let ok = crate::font_ttf::blit_ui_cell(
                    ch_c,
                    icon_cw as usize,
                    icon_ch as usize,
                    |gx, gy, a| {
                        if a == 0 {
                            return;
                        }
                        painted = true;
                        let px = x + gx as u64;
                        let py = y + gy as u64;
                        if px >= max_x {
                            return;
                        }
                        let mix = |b: u8, f: u8, aa: u32| {
                            (((b as u32) * (255 - aa) + (f as u32) * aa) / 255) as u8
                        };
                        let c = (
                            mix(bg.0, fg.0, a as u32),
                            mix(bg.1, fg.1, a as u32),
                            mix(bg.2, fg.2, a as u32),
                        );
                        self.put_pixel(px, py, c);
                    },
                );
                if !ok || !painted {
                    self.blit_glyph(
                        x + icon_cw.saturating_sub(cw) / 2,
                        body_y,
                        ch_c,
                        fg,
                        bg,
                    );
                }
                x += icon_cw;
            } else {
                if x + cw > max_x {
                    break;
                }
                self.blit_glyph(x, body_y, ch_c, fg, bg);
                x += cw;
            }
        }
        x
    }

    /// The status bar as a **column** (left/right edge).
    ///
    /// Reading order is **top → bottom** for *both* templates: brand, then
    /// `status_left` fields (one per row), then `status_right` fields continuing
    /// down the column. Previously `status_right` stacked upward from the bottom,
    /// which felt like horizontal "ends" rather than a vertical strip.
    fn draw_status_vertical(&self) {
        let (bx, by, bw, bh) = self.status_rect;
        STATUS_BAR_RECT.with(|r| *r = (bx, by, bw, bh));
        clear_status_chip_rects();
        self.paint_surface(bx, by, bw, bh, self.theme.status_bg);
        let (cw, ch) = (self.cw(), self.ch());
        // Slightly taller rows so icons have room; body text is vertically centred
        // by `draw_status_str`.
        let row = ch + STATUS_ICON_EXTRA + 4;
        let lr = (((row / 2).saturating_sub(2)) * 6 / 7).max(6);
        // Logo colours from ui.json theme (`logo` / `logo_node`, else accent/chat_fg).
        let logo_cx = bx + bw / 2;
        let logo_cy = by + STATUS_PAD / 2 + row / 2;
        self.draw_logo(
            logo_cx,
            logo_cy,
            lr,
            self.theme.logo,
            self.theme.logo_node,
        );
        // About opens only on the logo mark — not the wordmark or empty bar space.
        let logo_ext = lr + (lr / 3).max(3) + 4;
        let hx = logo_cx.saturating_sub(logo_ext).max(bx);
        let hy = logo_cy.saturating_sub(logo_ext).max(by);
        let hw = (logo_ext * 2).min(bw.saturating_sub(hx.saturating_sub(bx)));
        let hh = (logo_ext * 2).min(bh.saturating_sub(hy.saturating_sub(by)));
        set_status_chip_rect(StatusChip::Brand, (hx, hy, hw, hh));
        let tx = bx + STATUS_PAD / 2;
        let max_x = bx + bw.saturating_sub(STATUS_PAD / 2);
        let cols = (bw.saturating_sub(STATUS_PAD) / cw).max(4) as usize;
        let first = by + STATUS_PAD / 2 + row;
        let last = by + bh.saturating_sub(STATUS_PAD / 2);
        // One token per row, top → bottom (left template, then right chips).
        let mut top = first;
        let left_lines = crate::panes_layout::status_lines_vertical(&self.status_left, cols);
        for (i, line) in left_lines.iter().enumerate() {
            if top + row > last {
                break;
            }
            // Brand wordmark uses logo colour; all status text/icons use status_fg.
            let fg = if i == 0 {
                self.theme.logo
            } else {
                self.theme.status_fg
            };
            self.draw_status_str(tx, top, line, fg, self.theme.status_bg, max_x);
            top += row;
        }
        for (chip, text, st) in status_right_chips(true) {
            if top + row > last {
                break;
            }
            self.draw_status_str(tx, top, &text, chip_ink(self, st), self.theme.status_bg, max_x);
            set_status_chip_rect(chip, (bx, top, bw, row));
            top += row;
        }
    }

    /// The status bar as a **row** (top/bottom edge) — brand left, system chips
    /// right. Each right chip is individually clickable (macOS menu-bar style).
    fn draw_status_horizontal(&self) {
        let (_, sy_top, _, bar_h) = self.status_rect;
        STATUS_BAR_RECT.with(|r| *r = (0, sy_top, self.width, bar_h));
        clear_status_chip_rects();
        self.paint_surface(0, sy_top, self.width, bar_h, self.theme.status_bg);
        // Vertically centre the text/icon line in the bar (icons = body cell height).
        let line_h = self.ch();
        let ty = sy_top + bar_h.saturating_sub(line_h) / 2;
        let cw = self.cw();
        let lr = (((bar_h / 2).saturating_sub(2)) * 6 / 7).max(6);
        let lhalf = ((lr / 3).max(3)) / 2;
        let lcx = OUTER + lr + lhalf;
        // Logo from ui.json theme.logo / logo_node (see theme_from_pairs).
        self.draw_logo(
            lcx,
            sy_top + bar_h / 2,
            lr,
            self.theme.logo,
            self.theme.logo_node,
        );
        // About opens only on the logo mark — not the wordmark or empty bar space.
        let logo_x0 = lcx.saturating_sub(lr + lhalf);
        let logo_w = (lr + lhalf) * 2 + 4;
        set_status_chip_rect(
            StatusChip::Brand,
            (logo_x0.saturating_sub(2), sy_top, logo_w + 4, bar_h),
        );
        let text_x = lcx + lr + lhalf + cw / 2;
        let gap = 2 * cw;
        let usable = self.width.saturating_sub(text_x + OUTER + gap);
        let left_budget = (usable / 2 / cw).max(4) as usize;
        let left = crate::textsel::ellipsize(&self.status_left, left_budget);
        let max_left = text_x + left_budget as u64 * cw;
        // Brand wordmark ("ChittiOS v…") uses logo colour; bar field colour is status_fg.
        // Wordmark is not a hit target (About is logo-only).
        self.draw_status_str(
            text_x,
            ty,
            &left,
            self.theme.logo,
            self.theme.status_bg,
            max_left,
        );

        // Right chips painted individually so each has a hit rect.
        // Icons and labels share status_fg so they match the theme text colour.
        let chips = status_right_chips(false);
        let gap1 = cw; // within a tight group
        let gap2 = 2 * cw; // between groups
        let mut total = 0u64;
        for (i, (_, text, _)) in chips.iter().enumerate() {
            if i > 0 {
                // kbd–mouse single space; otherwise group gap
                total += if i == 1 { gap1 } else { gap2 };
            }
            total += status_str_advance(text, cw, line_h);
        }
        let left_end = text_x + status_str_advance(&left, cw, line_h) + gap;
        let mut x = self
            .width
            .saturating_sub(total + OUTER)
            .max(left_end)
            .min(self.width.saturating_sub(OUTER));
        let max_x = self.width.saturating_sub(OUTER / 2);
        // Centre pill: play/pause + title. Drawn in the gap between the
        // wordmark and the right-hand chips so a playing track is obvious
        // even with the audio tab in the background.
        self.draw_now_playing_pill(left_end, x, sy_top, bar_h, ty, cw, line_h);
        for (i, (chip, text, st)) in chips.iter().enumerate() {
            if i > 0 {
                x += if i == 1 { gap1 } else { gap2 };
            }
            let w = status_str_advance(text, cw, line_h);
            if x + w > max_x {
                break;
            }
            let x1 = self.draw_status_str(
                x,
                ty,
                text,
                chip_ink(self, *st),
                self.theme.status_bg,
                max_x,
            );
            // Hit pad a few px for easy clicking.
            let hx = x.saturating_sub(2);
            let hw = x1.saturating_sub(hx) + 2;
            set_status_chip_rect(*chip, (hx, sy_top, hw, bar_h));
            x = x1;
        }
    }

    /// Centre now-playing pill on a horizontal status bar. No-op when idle or
    /// when the gap between the wordmark and the right chips is too tight.
    fn draw_now_playing_pill(
        &self,
        left_end: u64,
        right_start: u64,
        sy_top: u64,
        bar_h: u64,
        ty: u64,
        cw: u64,
        line_h: u64,
    ) {
        let text = crate::shell::now_playing_chip();
        if text.is_empty() {
            return;
        }
        let pad = cw;
        let tw = status_str_advance(&text, cw, line_h);
        let pw = tw + pad * 2;
        let gap = right_start.saturating_sub(left_end);
        if gap < pw + cw * 2 {
            // Not enough room to centre it — park it just after the wordmark
            // so the chip is still visible and clickable.
            if left_end + pw + cw > right_start {
                return;
            }
        }
        let x = if gap >= pw + cw * 2 {
            left_end + (gap - pw) / 2
        } else {
            left_end
        };
        let y = sy_top + 2;
        let h = bar_h.saturating_sub(4).max(line_h);
        let pill_bg = self.mix(self.theme.status_bg, self.theme.accent, 0.22);
        let ink = self.theme.accent;
        self.fill_rect(x, y, pw, h, pill_bg);
        let tx = x + pad;
        self.draw_status_str(tx, ty, &text, ink, pill_bg, x + pw);
        set_status_chip_rect(StatusChip::NowPlaying, (x, sy_top, pw, bar_h));
    }
}
