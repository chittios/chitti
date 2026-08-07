//! Bringing the console up from a boot framebuffer, and the two log
//! channels that write into it.

use super::*;

/// Repaint the normal split-pane UI — used to restore the screen after the
/// full-screen `/top` dashboard exits.
pub fn redraw_all() {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.redraw();
            sc.cur_vis = false;
        }
    });
}

/// Bring up the compositor on a Limine framebuffer and paint the initial UI.
pub fn init_console(fb: &Framebuffer) {
    let bpp_bytes = (fb.bpp as u64).div_ceil(8);
    let s = Screen::layout(
        fb.address as usize,
        fb.width,
        fb.height,
        fb.pitch,
        bpp_bytes,
        fb.red_mask_shift as u32,
        fb.green_mask_shift as u32,
        fb.blue_mask_shift as u32,
    );
    init_from(s);
}

/// Bring up the compositor over a raw linear framebuffer whose pixels are
/// **XRGB8888** (little-endian B,G,R,X → red 16 / green 8 / blue 0) — the common
/// case (QEMU ramfb, most UEFI GOP / VirtualBox).
pub fn init_console_raw(addr: usize, width: u64, height: u64, pitch: u64) {
    init_console_raw_fmt(addr, width, height, pitch, 4, 16, 8, 0);
}

/// Bring up the compositor over a raw linear framebuffer with an explicit pixel
/// format. Used by the aarch64 UEFI path, which reads the GOP pixel format from
/// the boot-info page (a real HDMI monitor may report RGB rather than BGR, and
/// swapping red/blue would tint the whole UI).
#[allow(clippy::too_many_arguments)]
pub fn init_console_raw_fmt(
    addr: usize,
    width: u64,
    height: u64,
    pitch: u64,
    bpp_bytes: u64,
    r_shift: u32,
    g_shift: u32,
    b_shift: u32,
) {
    let s = Screen::layout(addr, width, height, pitch, bpp_bytes, r_shift, g_shift, b_shift);
    init_from(s);
}

/// Re-init the console onto a **different framebuffer** — a new address, pitch and
/// size — after a real mode set, preserving the session.
///
/// Unlike `init_console_raw_fmt` this keeps the layout config, the status text, the
/// interactive state and (via [`Pane::adopt`]) every pane's scrollback, so changing
/// resolution does not clear the screen or drop history; it reflows, exactly as a
/// font-scale change does. No splash, either — this is not a boot.
///
/// The logical-desktop preference is deliberately **dropped** here: it existed to
/// letterbox a desktop inside a too-large panel, and a real mode set is the better
/// answer to the same problem. Keeping it would letterbox *inside* the new mode.
#[allow(clippy::too_many_arguments)]
pub fn reinit_scanout(
    addr: usize,
    width: u64,
    height: u64,
    pitch: u64,
    bpp_bytes: u64,
    r_shift: u32,
    g_shift: u32,
    b_shift: u32,
) {
    // No console yet — this device *is* the display (virtio-gpu with no firmware
    // framebuffer behind it), so bring the console up on it rather than returning.
    // Without this a KMS-only machine boots to a blank screen: the compositor has
    // nothing to re-init and the driver has nowhere to draw.
    if SCREEN.with(|slot| slot.is_none()) {
        init_console_raw_fmt(addr, width, height, pitch, bpp_bytes, r_shift, g_shift, b_shift);
        return;
    }
    SCREEN.with(|slot| {
        let Some(old) = slot.as_ref() else { return };
        let cfg = old.layout.clone();
        let split = old.any_action_open() || old.actions.len() > 1;
        let focused = old.focused_action;
        let mut ns = Screen::build(
            addr, width, height, pitch, bpp_bytes, r_shift, g_shift, b_shift, &cfg, split, focused,
            None, // a real mode set replaces the letterbox, it does not nest inside it
        );
        ns.status_left = old.status_left.clone();
        ns.status_right = old.status_right.clone();
        carry_tabs(&mut ns, old);
        preserve_interactive(&mut ns, old);
        ns.chat.adopt(&old.chat);
        if !ns.any_action_open() {
            ns.focus_action = false;
        }
        ns.redraw();
        *slot = Some(ns);
    });
}

fn init_from(mut screen: Screen) {
    // Brand splash first (logo + wordmark), held briefly, then the live UI.
    if screen.layout.splash {
        screen.draw_splash();
        hold_ms(1300);
    }
    screen.redraw();
    SCREEN.with(|slot| *slot = Some(screen));
}

/// Busy-wait ~`ms` milliseconds for the splash hold, bounded by an iteration cap
/// so a frozen monotonic clock (some VBox configs) can't wedge the boot.
fn hold_ms(ms: u64) {
    let start = crate::arch::now_ms();
    let mut iters: u64 = 0;
    while crate::arch::now_ms().saturating_sub(start) < ms && iters < 300_000_000 {
        core::hint::spin_loop();
        iters += 1;
    }
}

/// Render `s` into the **chat** pane. Called by `serial::Serial::write_str`, so
/// ordinary `serial_println!` output (the shell + chat) appears here.
pub fn console_print(s: &str) {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.cursor_restore();
            sc.cur_vis = false;
            let mut chat = core::mem::replace(&mut sc.chat, dummy_pane());
            for &b in s.as_bytes() {
                Screen::pane_putc(sc, &mut chat, b);
            }
            sc.chat = chat;
            sc.caret_on = true; // keep the caret lit right after output
            // Do **not** redraw the composer here. The chat grid is already
            // sized above the reserved strip, so streaming tokens never touch
            // the box — and redrawing (with a strip clear) every chunk is what
            // made the whole composer flash while a response rendered.
            sc.cursor_overlay();
        }
    });
}

/// Render one byte into the chat pane (the shell's keystroke echo / backspace).
/// When the bordered composer is active, keystroke echo is handled by
/// [`composer_set`] — this path is for legacy serial-style editing only.
pub fn console_put_byte(byte: u8) {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            if sc.composer_active {
                return; // composer owns the input line
            }
            sc.cursor_restore();
            sc.cur_vis = false;
            let mut chat = core::mem::replace(&mut sc.chat, dummy_pane());
            Screen::pane_putc(sc, &mut chat, byte);
            sc.chat = chat;
            sc.caret_on = true;
            sc.cursor_overlay();
        }
    });
}

/// Advance the caret blink. Called from the shell's idle poll with the current
/// `now_ms()`; toggles the chat caret roughly twice a second.
pub fn blink(now_ms: u64) {
    if MODAL_ON.load(core::sync::atomic::Ordering::Relaxed) {
        return; // a modal overlays the panes; do not paint the caret under it
    }
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            // If the clock advances, blink on a 500 ms period. Only if it has
            // NEVER been seen advancing (a genuinely frozen monotonic clock)
            // fall back to a call-count cadence. On a fast host thousands of
            // idle polls land inside one millisecond, so the counter alone
            // must not trigger once the clock is known-good — that made the
            // caret strobe on VirtualBox.
            let toggle = if now_ms != sc.blink_seen_ms {
                if sc.blink_seen_ms != u64::MAX {
                    sc.clock_alive = true;
                }
                sc.blink_seen_ms = now_ms;
                sc.blink_calls = 0;
                now_ms.saturating_sub(sc.caret_last_ms) >= 500
            } else if sc.clock_alive {
                false
            } else {
                sc.blink_calls = sc.blink_calls.wrapping_add(1);
                if sc.blink_calls >= 300_000 {
                    sc.blink_calls = 0;
                    true
                } else {
                    false
                }
            };
            if toggle {
                sc.cursor_restore();
                sc.cur_vis = false;
                sc.caret_on = !sc.caret_on;
                sc.caret_last_ms = now_ms;
                sc.paint_caret();
                sc.cursor_overlay();
            }
        }
    });
}

/// Render `s` into the **logs** pane. Called by `ktrace`, so the trace stream
/// scrolls independently of the chat conversation.
pub fn log_print(s: &str) {
    // `try_with`, never `with`: a log line must not be able to stop the
    // machine. `SCREEN` is a non-reentrant spinlock, and anything that logs
    // from *inside* a critical section that already holds it — the panic path,
    // the allocation-failure handler, a driver called from a painter — would
    // spin forever with interrupts disabled. That is not a lost log line, it is
    // a dead OS with no output at all. When the screen is busy the line still
    // reaches serial (the caller wrote it there first); only the on-screen
    // mirror is dropped.
    SCREEN.try_with(|slot| {
        if let Some(sc) = slot {
            // Write into whichever action column currently shows ktrace.
            let Some((pi, _)) = sc.find_mode(RightMode::Ktrace) else {
                return;
            };
            if sc.actions[pi].right() != RightMode::Ktrace {
                return;
            }
            sc.cursor_restore();
            sc.cur_vis = false;
            let mut logs = core::mem::replace(&mut sc.actions[pi].pane, dummy_pane());
            for &b in s.as_bytes() {
                Screen::pane_putc(sc, &mut logs, b);
            }
            sc.actions[pi].pane = logs;
            sc.cursor_overlay();
        }
    });
}

impl Screen {
    /// Paint the boot splash: the brand mark, "ChittiOS", and a tagline, centred
    /// on the canvas. Shown briefly at boot (see [`show_splash`]).
    fn draw_splash(&self) {
        self.paint_wallpaper(0, 0, self.width, self.height, self.theme.screen_bg);
        let r = (self.height / 7).max(24);
        let cy = self.height * 2 / 5;
        // Ring/node from theme.logo / logo_node (ui.json), defaulting to brand.
        self.draw_logo(self.width / 2, cy, r, self.theme.logo, self.theme.logo_node);
        let name = "ChittiOS";
        let nx = self.width / 2 - (name.len() as u64 * self.cw()) / 2;
        self.draw_str(nx, cy + r + r / 2, name, self.theme.accent, self.theme.screen_bg);
        let tag = "an agentic operating system";
        let tx = self.width / 2 - (tag.len() as u64 * self.cw()) / 2;
        self.draw_str(tx, cy + r + r / 2 + self.ch() + 6, tag, self.theme.title_dim, self.theme.screen_bg);
    }

    /// Full repaint: background, chat pane (content re-rendered from its grid),
    /// the action (right) pane if open, caret, status bar.
    ///
    /// Parked panes (`w == 0`, fullscreen) are skipped entirely — their content
    /// is preserved in memory via [`Pane::take_content`] and restored on unpark.
    pub(super) fn redraw(&mut self) {
        // A full repaint is about to overwrite everything, so the notification
        // banner's saved background is stale: restoring it later would paint a
        // copy of the *old* screen over fresh content, and after a relayout it
        // would land in the wrong place entirely. Forgotten here rather than at
        // the eight call sites, so a ninth cannot get it wrong.
        self.toast_forget();
        crate::kms::damage(0, 0, self.fb_w as u32, self.fb_h as u32);
        // Dead space around a smaller-than-native desktop. A no-op at native, and
        // painted here (not per-frame) because the letterbox only changes when the
        // geometry does — which is exactly when a redraw happens.
        self.paint_letterbox();
        // Paint only the background *gutters* (margins + the gap between panes),
        // never a full-screen clear — the panes are painted over their own areas
        // below, so their content is never flashed to background. This is what
        // makes opening/closing the action pane not flicker the whole screen.
        self.paint_gutters();
        // Drop shadows sit in the gutters (right/bottom bands of each pane).
        if self.chat.w > 0 {
            self.drop_shadow(self.chat.x, self.chat.y, self.chat.w, self.chat.h);
            self.paint_surface(self.chat.x, self.chat.y, self.chat.w, self.chat.h, self.chat.bg);
            self.draw_frame(&self.chat, !self.action_focused());
            self.render_view(&self.chat);
            self.draw_composer(); // includes suggest popup when open
        }
        for (i, a) in self.actions.iter().enumerate() {
            if !self.column_visible(i) {
                continue;
            }
            let focused = self.focus_action && i == self.focused_action;
            self.drop_shadow(a.pane.x, a.pane.y, a.pane.w, a.pane.h);
            self.paint_surface(a.pane.x, a.pane.y, a.pane.w, a.pane.h, a.pane.bg);
            self.draw_frame_titled(&a.pane, focused, "");
            self.draw_tab_bar_for(i);
            self.draw_close_btn_for(i);
            if a.right() == RightMode::Ktrace {
                self.render_view(&a.pane);
            }
        }
        // Grid caret only when there is no composer; otherwise the caret is in
        // the input box (or absent while a reply streams).
        if self.chat.w > 0 && self.chat.view == 0 {
            self.caret_draw(&self.chat);
        }
        self.draw_status();
    }

    /// Fill just the screen-background gutters: the top/bottom strips and the
    /// left/right margins + the gap between the panes. Everything the panes
    /// cover is left untouched (painted over directly), so `redraw` never blanks
    /// the whole screen.
    fn paint_gutters(&self) {
        let bg = self.theme.screen_bg;
        let (by, bh) = (self.chat.y, self.chat.h);
        // Gutters are confined to the content rect: the status bar paints its own
        // area, and painting over it here would blank the bar on every redraw
        // wherever it does not happen to be the bottom edge.
        let (cx0, cy0) = (self.content_x, self.content_y);
        let (cx1, cy1) = (cx0 + self.content_w, cy0 + self.content_h);
        // Strip above the pane band, and the strip below it down to the content edge.
        self.paint_wallpaper(cx0, cy0, self.content_w, by.saturating_sub(cy0), bg);
        let below = by + bh;
        self.paint_wallpaper(cx0, below, self.content_w, cy1.saturating_sub(below), bg);
        // Horizontal gutters: left of chat, gaps between all boxes, right margin.
        let mut boxes: alloc::vec::Vec<(u64, u64)> = alloc::vec![];
        if self.chat.w > 0 {
            boxes.push((self.chat.x, self.chat.x + self.chat.w));
        }
        for a in &self.actions {
            if a.pane.w > 0 {
                boxes.push((a.pane.x, a.pane.x + a.pane.w));
            }
        }
        boxes.sort_by_key(|b| b.0);
        let mut x = cx0;
        for &(l, r) in &boxes {
            if l > x {
                self.paint_wallpaper(x, by, l - x, bh, bg);
            }
            x = r.max(x);
        }
        if x < cx1 {
            self.paint_wallpaper(x, by, cx1 - x, bh, bg);
        }
    }
}
