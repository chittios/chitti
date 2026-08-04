//! Screen geometry: building the pane tree from a [`LayoutCfg`], the font
//! scale and logical-desktop settings, and the draggable dividers.

use super::*;

/// Pick an integer font scale from the **desktop** height so glyphs stay a
/// legible physical size across resolutions. See
/// [`crate::display::auto_font_scale`] for the thresholds and why they are not a
/// division — the old formula left a 1440p panel at scale 1 (320 columns of 8px
/// text), which is what made a 2K display look broken.
fn pick_scale(height: u64) -> u64 {
    crate::display::auto_font_scale(height)
}

/// Framebuffer size `(width, height)` in pixels, for the mouse to clamp to.
pub fn screen_dims() -> Option<(u64, u64)> {
    SCREEN.with(|slot| slot.as_ref().map(|sc| (sc.width, sc.height)))
}

/// Copy interactive UI state that `Screen::build` always zeroes (composer mid-
/// prompt, keyboard focus, caret blink). Without this, Ctrl+F / tab open /
/// divider drag mid-`read_line` would kill the composer until the next prompt.
pub(super) fn preserve_interactive(ns: &mut Screen, old: &Screen) {
    ns.composer_active = old.composer_active;
    ns.composer_line = old.composer_line.clone();
    ns.composer_cur = old.composer_cur;
    ns.composer_hint_l = old.composer_hint_l.clone();
    ns.composer_hint_r = old.composer_hint_r.clone();
    ns.composer_hint_l_lead = old.composer_hint_l_lead.clone();
    ns.suggest_open = old.suggest_open;
    ns.suggest_items = old.suggest_items.clone();
    ns.suggest_sel = old.suggest_sel;
    ns.suggest_rect = old.suggest_rect;
    ns.focus_action = old.focus_action;
    ns.caret_on = old.caret_on;
    ns.caret_last_ms = old.caret_last_ms;
    ns.clock_alive = old.clock_alive;
    ns.blink_seen_ms = old.blink_seen_ms;
}

/// Carry the action columns' tabs, active index, and pane text from `old` into a
/// freshly-built `ns`, column by column.
///
/// When the new layout has **fewer** columns (`/pane max` shrank the budget) the
/// dropped columns' tabs are appended to the last surviving column rather than
/// discarded: a tab is a live process (a package-UI agent, a streaming ktrace, a
/// playing audio track), so dropping the list would leak the task and leave no
/// way to reach or close it.
pub(super) fn carry_tabs(ns: &mut Screen, old: &Screen) {
    let n = ns.actions.len().min(old.actions.len());
    for i in 0..n {
        ns.actions[i].tabs = old.actions[i].tabs.clone();
        ns.actions[i].active = old.actions[i].active.min(ns.actions[i].tabs.len().saturating_sub(1));
        ns.actions[i].pane.adopt(&old.actions[i].pane);
    }
    if old.actions.len() > n && n > 0 {
        let last = n - 1;
        for i in n..old.actions.len() {
            for &m in &old.actions[i].tabs {
                if !ns.actions[last].tabs.contains(&m) {
                    ns.actions[last].tabs.push(m);
                }
            }
        }
        let len = ns.actions[last].tabs.len();
        ns.actions[last].active = ns.actions[last].active.min(len.saturating_sub(1));
    }
    ns.focused_action = old.focused_action.min(ns.actions.len().saturating_sub(1));
}

/// Rebuild geometry for a new split state, preserving layout config, status,
/// interactive state, action tabs, and pane text via [`Pane::adopt`].
pub(super) fn rebuilt(old: &Screen, split: bool) -> Screen {
    // **Every screen rebuild invalidates pane interiors**, so the mark belongs here
    // rather than at each caller. Opening a view was wired up and closing one was not,
    // which is the same omission twice — a choke point ends that.
    mark_tabs_dirty();
    let mut ns = Screen::build(
        // `fb_w`/`fb_h`, never `width`/`height` — those are the logical desktop,
        // and feeding them back in would shrink the viewport on every rebuild.
        old.addr, old.fb_w, old.fb_h, old.pitch, old.bpp_bytes, old.r_shift, old.g_shift, old.b_shift, &old.layout, split,
        old.focused_action,
        old.logical_pref,
    );
    ns.status_left = old.status_left.clone();
    ns.status_right = old.status_right.clone();
    carry_tabs(&mut ns, old);
    preserve_interactive(&mut ns, old);
    if !ns.any_action_open() {
        ns.focus_action = false;
    }
    ns.chat.adopt(&old.chat);
    ns
}

/// Rebuild the panes from a new [`LayoutCfg`] (split ratio, font scale, pane
/// swap, titles, fullscreen) on the live framebuffer and repaint. Used by
/// `/ui`, Ctrl+F, and divider drag. No-op if the console isn't up.
///
/// Preserves the live composer + focus so a mid-prompt fullscreen toggle does
/// not strand the shell with a dead input box.
pub fn relayout(cfg: &LayoutCfg) {
    SCREEN.with(|slot| {
        if let Some(old) = slot {
            let any = old.any_action_open();
            let band = crate::panes_layout::action_band_visible(cfg.max_panes, any);
            let mut ns = Screen::build(
                old.addr,
                old.fb_w,
                old.fb_h,
                old.pitch,
                old.bpp_bytes,
                old.r_shift,
                old.g_shift,
                old.b_shift,
                cfg,
                band,
                old.focused_action,
                old.logical_pref,
            );
            ns.status_left = old.status_left.clone();
            ns.status_right = old.status_right.clone();
            carry_tabs(&mut ns, old);
            preserve_interactive(&mut ns, old);
            ns.chat.adopt(&old.chat);
            // Fullscreen can park the chat (action-full): keep focus on action.
            // Chat-full parks the columns — snap keyboard back to the composer.
            if cfg.fullscreen == 1 || !ns.any_action_open() {
                ns.focus_action = false;
            }
            ns.redraw();
            *slot = Some(ns);
            // Frames are painted; interiors are the views' own and must follow.
            mark_tabs_dirty();
        }
    });
}

/// Set total pane count (2..=9, including shell) and relayout.
///
/// The grid is reshaped to the most balanced arrangement holding exactly
/// `n - 1` action panes, with even track weights — a pane-count change is a new
/// layout, so carrying over the old weights would leave the new grid lopsided.
pub fn set_max_panes(n: u8) {
    let n = crate::panes_layout::clamp_max_panes(n as u64);
    let (cols, rows) = crate::panes_layout::grid_for_count(
        crate::panes_layout::action_column_count(n),
    );
    set_grid_weighted(cols, rows, None);
}

/// Set the action grid shape explicitly (`/pane grid <cols> <rows>`), clamped to
/// at most 8 action panes, and sync `max_panes` to the resulting cell count.
pub fn set_grid(cols: usize, rows: usize) -> (usize, usize) {
    let (cols, rows) = crate::panes_layout::clamp_grid(cols, rows);
    set_grid_weighted(cols, rows, None);
    // Report what was actually applied — the band may not fit the request.
    grid_shape()
}

/// Shared tail of the grid setters: clamp the shape to what the band can host,
/// build the spec (even weights, or the given ones), and relayout.
///
/// `max_panes` is **derived** from the shape that survives clamping rather than
/// passed in, so the pane budget and the grid can never disagree.
fn set_grid_weighted(cols: usize, rows: usize, weights: Option<(Vec<u64>, Vec<u64>)>) {
    let cfg = SCREEN.with(|slot| {
        let sc = slot.as_mut()?;
        // Clamp to the pixels available: a shape the band cannot host would draw
        // cells shorter than their own header. Doing it here (not in `build`)
        // keeps `grid_shape()` — used by the status line and by `panes.json` —
        // reporting exactly what is on screen.
        let (bw, bh) = sc.band_capacity();
        let cols = crate::panes_layout::fit_tracks(bw, GAP, cols, crate::panes_layout::MIN_TRACK_PX);
        let rows = crate::panes_layout::fit_tracks(bh, GAP, rows, sc.min_pane_h());
        let (cols, rows) = crate::panes_layout::clamp_grid(cols, rows);
        let mut c = sc.layout.clone();
        c.grid = match weights {
            Some((col_w, row_h)) => {
                crate::panes_layout::GridSpec { cols, rows, col_w, row_h }.sanitized()
            }
            None => crate::panes_layout::GridSpec::even(cols, rows),
        };
        c.max_panes = crate::panes_layout::clamp_max_panes(c.grid.len() as u64 + 1);
        c.fullscreen = 0;
        Some(c)
    });
    if let Some(c) = cfg {
        relayout(&c);
    }
}

/// Set the font scale (`0` = automatic from the desktop height) and relayout.
///
/// The knob that actually makes a high-resolution screen readable: cells are
/// `8*scale` x `16*scale` pixels, so this changes how much fits on screen and how
/// big the text is — unlike a smaller desktop, which only letterboxes. Returns the
/// scale now in effect.
pub fn set_font_scale(scale: u64) -> Option<u64> {
    let n = crate::display::clamp_font_scale(scale);
    let cfg = SCREEN.with(|slot| {
        slot.as_mut().map(|sc| {
            let mut c = sc.layout.clone();
            c.scale = n; // 0 → `pick_scale` recomputes from the desktop height
            c
        })
    })?;
    relayout(&cfg);
    effective_font_scale()
}

/// The font scale currently rendering (never 0 — the resolved value).
pub fn effective_font_scale() -> Option<u64> {
    SCREEN.with(|slot| slot.as_ref().map(|sc| sc.scale))
}

/// Move the OS status bar to a desktop edge and relayout.
///
/// Applies instantly: every pane is laid out inside the leftover content rect, so
/// this is a full relayout (scrollback is preserved by `Pane::adopt`, as with a
/// resolution or font-scale change). Returns the position now in effect.
pub fn set_status_pos(pos: crate::panes_layout::StatusPos) -> Option<crate::panes_layout::StatusPos> {
    let cfg = SCREEN.with(|slot| {
        slot.as_mut().map(|sc| {
            let mut c = sc.layout.clone();
            c.status_pos = pos;
            c
        })
    })?;
    relayout(&cfg);
    status_pos()
}

/// The edge the status bar is currently on.
pub fn status_pos() -> Option<crate::panes_layout::StatusPos> {
    SCREEN.with(|slot| slot.as_ref().map(|sc| sc.layout.status_pos))
}

/// The **pinned** font scale (`0` = automatic), i.e. the setting rather than the
/// resolved value. `None` before the console is up.
///
/// `ui_config::layout_cfg` carries this through so a `/theme` apply can't reset a
/// `/display scale` back to ui.json's value — the same live-value trap that
/// `max_panes` and the pane grid already have to avoid.
pub fn pinned_font_scale() -> Option<u64> {
    SCREEN.with(|slot| slot.as_ref().map(|sc| sc.layout.scale))
}

/// The **physical** framebuffer size the firmware gave us — the panel, whatever
/// the desktop is currently set to.
pub fn physical_size() -> Option<(u32, u32)> {
    SCREEN.with(|slot| slot.as_ref().map(|sc| (sc.fb_w as u32, sc.fb_h as u32)))
}

/// The current **logical** desktop size (what layouts are computed against).
pub fn logical_size() -> Option<(u32, u32)> {
    SCREEN.with(|slot| slot.as_ref().map(|sc| (sc.width as u32, sc.height as u32)))
}

/// Whether the desktop is a letterboxed viewport rather than the whole panel.
pub fn is_letterboxed() -> bool {
    SCREEN
        .with(|slot| {
            slot.as_ref()
                .map(|sc| sc.width != sc.fb_w || sc.height != sc.fb_h)
        })
        .unwrap_or(false)
}

/// Set the logical desktop size, applying immediately. `None` = native (use the
/// whole framebuffer).
///
/// Rebuilds the whole screen: every pane's cell grid depends on the desktop size,
/// so this reflows scrollback exactly as a font-scale change does. Returns the
/// size actually applied (clamped to the framebuffer), or `None` if the console
/// isn't up.
pub fn set_logical_size(want: Option<(u32, u32)>) -> Option<(u32, u32)> {
    let applied = SCREEN.with(|slot| {
        let old = slot.as_mut()?;
        let pref = want.map(|(w, h)| {
            let (w, h) = crate::display::clamp_logical((old.fb_w as u32, old.fb_h as u32), (w, h));
            (w as u64, h as u64)
        });
        if pref == old.logical_pref {
            return Some((old.width as u32, old.height as u32)); // already there
        }
        old.logical_pref = pref;
        // Rebuild at the current split so the action band's open/closed state and
        // every tab survive the resolution change.
        let split = old.any_action_open() || old.actions.len() > 1;
        let mut ns = rebuilt(old, split);
        // The desktop shrank or moved: clear the *whole* panel once, or the old
        // desktop's pixels stay lit outside the new viewport.
        ns.fill_phys(0, 0, ns.fb_w, ns.fb_h, (0, 0, 0));
        ns.redraw();
        let got = (ns.width as u32, ns.height as u32);
        *slot = Some(ns);
        Some(got)
    })?;
    Some(applied)
}

/// The logical desktop sizes selectable on this framebuffer (native first).
pub fn available_modes() -> Vec<(u32, u32)> {
    match physical_size() {
        Some(p) => crate::display::modes_for(p),
        None => Vec::new(),
    }
}

/// The action grid's current shape `(cols, rows)`.
pub fn grid_shape() -> (usize, usize) {
    SCREEN.with(|slot| {
        slot.as_ref()
            .map(|sc| {
                let g = sc.layout.grid.sanitized();
                (g.cols, g.rows)
            })
            .unwrap_or((1, 1))
    })
}

/// The action grid's track weights `(col_w, row_h)` in permille, for persisting.
pub fn grid_weights() -> (Vec<u64>, Vec<u64>) {
    SCREEN.with(|slot| {
        slot.as_ref()
            .map(|sc| {
                let g = sc.layout.grid.sanitized();
                (g.col_w, g.row_h)
            })
            .unwrap_or_else(|| (alloc::vec![1000], alloc::vec![1000]))
    })
}

/// Restore a saved grid (shape + weights) from `panes.json` at boot.
pub fn set_grid_spec(cols: usize, rows: usize, col_w: Vec<u64>, row_h: Vec<u64>) {
    let (cols, rows) = crate::panes_layout::clamp_grid(cols, rows);
    set_grid_weighted(cols, rows, Some((col_w, row_h)));
}

/// Current total pane budget (shell + action columns).
pub fn max_panes() -> u8 {
    SCREEN.with(|slot| {
        slot.as_ref()
            .map(|sc| crate::panes_layout::clamp_max_panes(sc.layout.max_panes as u64))
            .unwrap_or(crate::panes_layout::MAX_PANES_DEFAULT)
    })
}

/// Toggle fullscreen: maximise the focused pane to fill the screen, or restore
/// the split. Returns the new state (0 normal, 1 chat-full, 2 action-full).
pub fn toggle_fullscreen() -> u8 {
    let cfg = SCREEN.with(|slot| {
        slot.as_mut().map(|sc| {
            let action_open = sc.right() != RightMode::Closed;
            let mut c = sc.layout.clone();
            c.fullscreen = if c.fullscreen != 0 {
                0
            } else if sc.focus_action && action_open {
                2
            } else {
                1
            };
            c
        })
    });
    match cfg {
        Some(c) => {
            let st = c.fullscreen;
            relayout(&c);
            st
        }
        None => 0,
    }
}

/// The current chat-pane split percentage (10..90).
pub fn split_pct() -> u64 {
    SCREEN.with(|slot| slot.as_ref().map(|sc| sc.layout.chat_pct).unwrap_or(CHAT_PCT))
}

/// The default chat-pane split percentage (`/pane reset`).
pub fn default_chat_pct() -> u64 {
    CHAT_PCT
}

/// Set the chat|action split to `pct` percent (clamped 10..90) and relayout,
/// clearing any fullscreen state.
pub fn set_split_pct(pct: u64) {
    let cfg = SCREEN.with(|slot| {
        slot.as_mut().map(|sc| {
            let mut c = sc.layout.clone();
            c.chat_pct = pct.clamp(10, 90);
            c.fullscreen = 0;
            c
        })
    });
    if let Some(c) = cfg {
        relayout(&c);
    }
}

/// Nudge the split ratio by `delta` percent (keyboard resize).
pub fn nudge_split(delta: i64) {
    let p = split_pct() as i64;
    set_split_pct((p + delta).clamp(10, 90) as u64);
}

/// If `(x,y)` is on the draggable divider between the two panes, return its
/// current gap centre x (so the caller can enter a resize drag). `None` when
/// fullscreen/closed (no divider) or the point is elsewhere.
pub fn divider_hit(x: u64, y: u64) -> Option<Divider> {
    SCREEN.with(|slot| {
        let sc = slot.as_ref()?;
        if sc.layout.fullscreen != 0 {
            return None;
        }
        sc.divider_at(x, y)
    })
}

/// The kind of divider under the pointer (see [`divider_hit`]).
pub use crate::panes_layout::Divider;

/// Drag `which` divider to pixel `(x, y)` and relayout.
///
/// [`Divider::Band`] moves `chat_pct`; a grid divider re-splits only the two
/// tracks it separates, so panes elsewhere in the band keep their exact sizes.
pub fn drag_divider(which: Divider, x: u64, y: u64) {
    let cfg = SCREEN.with(|slot| {
        let sc = slot.as_mut()?;
        let mut c = sc.layout.clone();
        c.fullscreen = 0;
        match which {
            Divider::Band => {
                // Inverse of `split_band`, so it must be handed the same span and a
                // content-relative x — otherwise a left-edge status bar offsets
                // every drag by the bar's width.
                c.chat_pct = crate::panes_layout::band_divider_pct(
                    sc.content_w,
                    OUTER,
                    GAP,
                    c.swap,
                    x.saturating_sub(sc.content_x),
                );
            }
            Divider::Col(i) => {
                let (bx, _, bw, _) = sc.band_rect()?;
                let mut g = c.grid.sanitized();
                if !crate::panes_layout::resize_tracks(
                    &mut g.col_w,
                    i,
                    bw,
                    GAP,
                    x.saturating_sub(bx),
                ) {
                    return None;
                }
                c.grid = g;
            }
            Divider::Row(i) => {
                let (_, by, _, bh) = sc.band_rect()?;
                let mut g = c.grid.sanitized();
                if !crate::panes_layout::resize_tracks(
                    &mut g.row_h,
                    i,
                    bh,
                    GAP,
                    y.saturating_sub(by),
                ) {
                    return None;
                }
                c.grid = g;
            }
        }
        Some(c)
    });
    if let Some(c) = cfg {
        relayout(&c);
    }
}

impl Screen {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn layout(
        addr: usize,
        width: u64,
        height: u64,
        pitch: u64,
        bpp_bytes: u64,
        r_shift: u32,
        g_shift: u32,
        b_shift: u32,
    ) -> Screen {
        // Default: max_panes=2 → action band closed until first tab. max_panes>2
        // shows empty action columns as drop targets from boot.
        let cfg = crate::ui_config::boot_layout();
        let band = crate::panes_layout::action_band_visible(cfg.max_panes, false);
        Screen::build(
            addr,
            width,
            height,
            pitch,
            bpp_bytes,
            r_shift,
            g_shift,
            b_shift,
            &cfg,
            band,
            0,
            None, // native until `display.json` is applied
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn build(
        addr: usize,
        width: u64,
        height: u64,
        pitch: u64,
        bpp_bytes: u64,
        r_shift: u32,
        g_shift: u32,
        b_shift: u32,
        cfg: &LayoutCfg,
        split: bool,
        focused: usize,
        logical_pref: Option<(u64, u64)>,
    ) -> Screen {
        // `width`/`height` arrive as the PHYSICAL framebuffer. A logical
        // preference turns them into a centred viewport; everything below this
        // point lays out against the logical size only, so the whole compositor
        // is resolution-agnostic and needs no other change.
        let (fb_w, fb_h) = (width, height);
        let (origin_x, origin_y, width, height) = match logical_pref {
            Some((lw, lh)) => {
                let (x, y, w, h) = crate::display::viewport(
                    (fb_w as u32, fb_h as u32),
                    (lw as u32, lh as u32),
                );
                (x as u64, y as u64, w as u64, h as u64)
            }
            None => (0, 0, fb_w, fb_h), // native: identity, byte-for-byte the old path
        };
        // Font scale follows the LOGICAL height, so a smaller desktop gets
        // proportionally sized text rather than the panel's.
        let scale = if cfg.scale > 0 { cfg.scale } else { pick_scale(height) };
        let cw = CELL_W * scale;
        let ch = CELL_H * scale;
        // Carve the status bar off its edge; everything below lays out in what is
        // left. A vertical bar is a fixed column of cells wide (see
        // `STATUS_V_COLS`), a horizontal one a single text row plus padding.
        let (status_rect, (content_x, content_y, content_w, content_h)) =
            crate::panes_layout::status_split(
                width,
                height,
                cfg.status_pos,
                status_thickness(cfg.status_pos, cw, ch),
            );
        let box_y = content_y + OUTER;
        let box_h = content_h.saturating_sub(2 * OUTER);
        let pct = cfg.chat_pct.clamp(10, 90);
        let grid = cfg.grid.sanitized();
        let n_act = grid.len();
        let full_w = content_w.saturating_sub(2 * OUTER);
        let th = cfg.theme;
        let focused = focused.min(n_act - 1);
        // `split_band` works in a 0-based span, so its x results are shifted into
        // the content rect. With a left-edge status bar `content_x` is the bar's
        // width; at every other position it is 0 and this is the identity.
        let band_split = || {
            let (cx, cwid, bx, bw) =
                crate::panes_layout::split_band(content_w, OUTER, GAP, pct, true, cfg.swap);
            (content_x + cx, cwid, content_x + bx, bw)
        };
        // Parked panes keep w==0 so `Pane::adopt` clones content without a
        // catastrophic 1-column reflow of the full scrollback.
        let parked = (content_x + content_w, box_y, 0u64, box_h);
        let mut action_boxes = if cfg.fullscreen == 2 && split {
            // The **focused** action pane fills the screen; chat + every other
            // pane park. Maximising cell 0 regardless of focus would show a
            // different pane than the one the user was working in.
            let mut boxes = alloc::vec![parked; n_act];
            boxes[focused] = (content_x + OUTER, box_y, full_w, box_h);
            boxes
        } else if cfg.fullscreen == 1 || !split {
            // Chat fills; the whole action grid parks.
            alloc::vec![parked; n_act]
        } else {
            let (_, _, bx, bw) = band_split();
            crate::panes_layout::layout_grid(bx, box_y, bw, box_h, GAP, &grid)
        };
        // Keep the vector's length pinned to the cell count regardless.
        action_boxes.resize(n_act, parked);
        let (chat_x, chat_bw) = if cfg.fullscreen == 2 && split {
            (content_x + content_w, 0)
        } else if cfg.fullscreen == 1 || !split {
            (content_x + OUTER, full_w)
        } else {
            let (cx, cwid, ..) = band_split();
            (cx, cwid)
        };
        let chat = Pane::new(chat_x, box_y, chat_bw, box_h, cw, ch, th.chat_fg, th.chat_bg, cfg.chat_title.clone(), true);
        let mut actions = alloc::vec::Vec::with_capacity(n_act);
        for (i, &(ax, ay, aw, ah)) in action_boxes.iter().enumerate() {
            let title = if i == 0 {
                cfg.logs_title.clone()
            } else {
                alloc::format!("action {}", i + 1)
            };
            actions.push(ActionSlot {
                pane: Pane::new(ax, ay, aw, ah, cw, ch, th.logs_fg, th.logs_bg, title, false),
                tabs: Vec::new(),
                active: 0,
            });
        }
        let mut status_left = String::from("ChittiOS v");
        status_left.push_str(crate::VERSION);
        let mut scr = Screen {
            addr, width, height, pitch, bpp_bytes, r_shift, g_shift, b_shift, scale, chat,
            fb_w,
            fb_h,
            origin_x,
            origin_y,
            logical_pref,
            status_rect,
            content_x,
            content_y,
            content_w,
            content_h,
            actions,
            focused_action: focused,
            status_left,
            status_right: String::new(),
            caret_on: true,
            caret_last_ms: 0,
            composer_active: false,
            composer_line: String::new(),
            composer_cur: 0,
            composer_hint_l: String::from("↑↓ history · Tab select · Ctrl+P/N pick · Ctrl+R search · Enter send · /cmds · @files"),
            composer_hint_r: String::new(),
            composer_prompt: String::new(),
            composer_hint_l_lead: alloc::vec::Vec::new(),
            suggest_open: false,
            suggest_items: alloc::vec::Vec::new(),
            suggest_sel: 0,
            suggest_rect: None,
            blink_seen_ms: u64::MAX,
            blink_calls: 0,
            clock_alive: false,
            focus_action: false,
            drop_target: None,
            layout: cfg.clone(),
            theme: th,
            cur_x: width / 2,
            cur_y: height / 2,
            cur_vis: false,
            cur_active: false,
            cur_saved: Vec::new(),
            cur_sw: CUR_W,
            cur_sh: CUR_H,
            wallpaper: None,
            opacity: 255,
        };
        // Decode/generate the wallpaper once for this layout (windows blend over
        // it at `opacity`); recomputed on relayout.
        scr.set_wallpaper(&cfg.wallpaper, cfg.opacity);
        scr
    }

    pub(super) fn cw(&self) -> u64 {
        CELL_W * self.scale
    }

    pub(super) fn ch(&self) -> u64 {
        CELL_H * self.scale
    }

    /// Pixel size the action band would have at the current split, whether or not
    /// it is presently on screen (the grid is sized before the band is opened).
    fn band_capacity(&self) -> (u64, u64) {
        let (_, _, _, bw) = crate::panes_layout::split_band(
            self.content_w,
            OUTER,
            GAP,
            self.layout.chat_pct.clamp(10, 90),
            true,
            self.layout.swap,
        );
        let bh = self.content_h.saturating_sub(2 * OUTER);
        (bw, bh)
    }

    /// Smallest usable pane height: its title header plus one text row of
    /// padded interior. Scale-derived, so it is right at any font size.
    fn min_pane_h(&self) -> u64 {
        let ch = self.ch();
        // Mirrors `Pane::new`'s header_h + the interior's bottom border/padding.
        (BORDER + 4 + ch + 6) + BORDER + PAD + ch
    }

    /// Which divider (if any) is under `(x, y)`, with a few pixels of grab
    /// tolerance either side of the gap.
    ///
    /// Grid dividers are checked **before** the shell|band one: a grid column
    /// gap can sit within grab tolerance of the band gap on a narrow screen, and
    /// the inner divider is the more specific target.
    fn divider_at(&self, x: u64, y: u64) -> Option<Divider> {
        let grid = self.layout.grid.sanitized();
        let (bx, by, bw, bh) = self.band_rect()?;
        let in_band_y = y + 4 >= by && y <= by + bh + 4;
        let in_band_x = x + 4 >= bx && x <= bx + bw + 4;
        if in_band_x && in_band_y {
            // Vertical dividers between grid columns.
            let cw = crate::panes_layout::track_sizes(bw, GAP, &grid.col_w);
            let cx = crate::panes_layout::track_offsets(bw, GAP, &grid.col_w);
            for i in 0..grid.cols.saturating_sub(1) {
                let gap_l = bx + cx[i] + cw[i];
                if x + 4 >= gap_l && x <= gap_l + GAP + 4 {
                    return Some(Divider::Col(i));
                }
            }
            // Horizontal dividers between grid rows.
            let rh = crate::panes_layout::track_sizes(bh, GAP, &grid.row_h);
            let ry = crate::panes_layout::track_offsets(bh, GAP, &grid.row_h);
            for i in 0..grid.rows.saturating_sub(1) {
                let gap_t = by + ry[i] + rh[i];
                if y + 4 >= gap_t && y <= gap_t + GAP + 4 {
                    return Some(Divider::Row(i));
                }
            }
        }
        // The shell | action-band divider.
        if self.chat.w == 0 || bw == 0 {
            return None; // band collapsed, or a pane parked by fullscreen
        }
        let a = &self.chat;
        let gap_l = a.x.min(bx) + if a.x < bx { a.w } else { bw };
        let gap_r = gap_l + GAP;
        if y >= a.y && y < a.y + a.h && x + 4 >= gap_l && x <= gap_r + 4 {
            return Some(Divider::Band);
        }
        None
    }

    /// The action band's bounding rectangle `(x, y, w, h)` — the union of every
    /// unparked grid cell. `None` when the whole band is parked.
    fn band_rect(&self) -> Option<(u64, u64, u64, u64)> {
        let mut r: Option<(u64, u64, u64, u64)> = None;
        for a in &self.actions {
            if a.pane.w == 0 {
                continue;
            }
            let (x0, y0, x1, y1) = (a.pane.x, a.pane.y, a.pane.x + a.pane.w, a.pane.y + a.pane.h);
            r = Some(match r {
                None => (x0, y0, x1 - x0, y1 - y0),
                Some((rx, ry, rw, rh)) => {
                    let (nx, ny) = (rx.min(x0), ry.min(y0));
                    let (ex, ey) = ((rx + rw).max(x1), (ry + rh).max(y1));
                    (nx, ny, ex - nx, ey - ny)
                }
            });
        }
        r
    }
}
