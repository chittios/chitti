//! Action-pane tabs: which view lives where, opening/closing them, and the
//! drag-and-drop that moves one between panes.

use super::*;

/// The short tab-bar label for a view (Font Awesome glyph + name).
///
/// Package-UI agent surfaces use the **agent name** (chess, paint, …) rather
/// than the generic "surface" — the action-pane window title tracks the app.
fn tab_label(m: RightMode) -> alloc::string::String {
    use crate::icons::fa;
    match m {
        RightMode::Closed => alloc::string::String::new(),
        RightMode::Ktrace => alloc::format!("{} ktrace", fa::BUG),
        RightMode::Editor => alloc::format!("{} editor", fa::PEN_TO_SQUARE),
        RightMode::Top => alloc::format!("{} top", fa::GAUGE),
        RightMode::Todos => alloc::format!("{} todos", fa::LIST_CHECK),
        RightMode::Audio => alloc::format!("{} audio", fa::WAVE_SQUARE),
        RightMode::Surface(IMAGE_SURFACE) => alloc::format!("{} image", fa::IMAGE),
        RightMode::Surface(VIDEO_SURFACE) => alloc::format!("{} video", fa::FILM),
        RightMode::Surface(BROWSER_SURFACE) => alloc::format!("{} browser", fa::GLOBE),
        RightMode::Surface(PDF_SURFACE) => alloc::format!("{} pdf", fa::FILE_LINES),
        RightMode::Surface(id) => {
            // Running package UI (chess/paint/snake/…) — FA agent icon + name.
            // Use surface_tab_name (display cache), never RUN: tab paint runs
            // while SCREEN is held, often mid-present from a guest host import.
            if let Some(name) = crate::service::package_ui::surface_tab_name(id) {
                let icon = crate::icons::for_agent(&name);
                return alloc::format!("{icon} {name}");
            }
            alloc::format!("{} surface-{id}", fa::WINDOW)
        }
    }
}

/// The current (focused action column's active tab) mode.
pub fn right_mode() -> RightMode {
    SCREEN.with(|slot| slot.as_ref().map(|sc| sc.right()).unwrap_or(RightMode::Closed))
}

/// Index of the focused action column (0-based).
pub fn focused_action_index() -> usize {
    SCREEN.with(|slot| slot.as_ref().map(|sc| sc.focused_action).unwrap_or(0))
}

/// Number of action columns currently laid out.
pub fn action_column_count() -> usize {
    SCREEN.with(|slot| slot.as_ref().map(|sc| sc.actions.len()).unwrap_or(1))
}

/// The active tab mode of every **visible** action pane, in grid order.
///
/// A relayout (divider drag, grid reshape, tab move) repaints all the frames but
/// not the tab *interiors*, which each view owns. With more than one pane showing
/// content, the caller has to re-present every one of them — repainting only the
/// focused pane leaves the others blank until they happen to tick.
pub fn visible_tab_modes() -> Vec<RightMode> {
    SCREEN.with(|slot| {
        let Some(sc) = slot.as_ref() else { return Vec::new() };
        (0..sc.actions.len())
            .filter(|&i| sc.column_visible(i))
            .map(|i| sc.actions[i].right())
            .filter(|&m| m != RightMode::Closed)
            .collect()
    })
}

/// The open tab modes on the **focused** action column, in bar order.
pub fn tab_modes() -> Vec<RightMode> {
    SCREEN.with(|slot| {
        slot.as_ref()
            .map(|sc| sc.focused_slot().tabs.clone())
            .unwrap_or_default()
    })
}

/// True if a tab of `mode` is open on **any** action column.
pub fn has_tab(mode: RightMode) -> bool {
    SCREEN.with(|slot| slot.as_ref().map(|sc| sc.find_mode(mode).is_some()).unwrap_or(false))
}

/// Note that the action band changed and its panes' interiors need repainting.
pub fn mark_tabs_dirty() {
    TABS_DIRTY.store(true, core::sync::atomic::Ordering::Relaxed);
}

/// Consume the flag. Cleared *before* the caller repaints, so a view that pumps while
/// painting cannot drive itself round the loop again.
pub fn take_tabs_dirty() -> bool {
    TABS_DIRTY.swap(false, core::sync::atomic::Ordering::Relaxed)
}

pub(super) fn open_view_slot(slot: &mut Option<Screen>, mode: RightMode) {
    mark_tabs_dirty();
    let Some(old) = slot else { return };
    // NB: opening a view must **not** move keyboard focus to the action pane
    // for most modes — the user typed a command at the composer and is still
    // typing there. Setting `focus_action` here made the *next* command go to
    // the pane instead of the prompt, which reads as the shell having frozen.
    // Focus moves only on an explicit act: clicking a pane, `/pane focus`,
    // Ctrl+Tab. **Exception: the editor** — it is an interactive text surface,
    // so keys must land there on open (and Ctrl+Tab still returns to the shell).
    if mode == RightMode::Editor {
        old.focus_action = true;
    }
    // Already open somewhere → select that pane's tab (and make it the open
    // target for the next view) without taking focus (except editor above).
    if let Some((pi, ti)) = old.find_mode(mode) {
        old.focused_action = pi;
        old.actions[pi].active = ti;
        old.repaint_action();
        return;
    }
    let fi = old.focused_action.min(old.actions.len().saturating_sub(1));
    // Only a lone action pane collapses, so only it needs a full relayout when
    // its first tab opens; a multi-pane grid is already on screen.
    let need_relayout = !old.any_action_open() && old.actions.len() == 1;
    if need_relayout || old.actions.is_empty() {
        let mut ns = rebuilt(old, true);
        let fi = ns.focused_action.min(ns.actions.len().saturating_sub(1));
        ns.actions[fi].tabs = alloc::vec![mode];
        ns.actions[fi].active = 0;
        ns.focused_action = fi;
        ns.redraw();
        *slot = Some(ns);
        return;
    }
    // Additional tab on the focused pane (geometry unchanged).
    let a = &mut old.actions[fi];
    a.tabs.push(mode);
    a.active = a.tabs.len() - 1;
    old.repaint_action();
}

/// Open (or focus) a tab for `mode` on the focused action column.
pub fn set_right(mode: RightMode) {
    if mode == RightMode::Closed {
        return;
    }
    SCREEN.with(|slot| open_view_slot(slot, mode));
}

/// Cycle tabs on the focused action column.
pub fn cycle_tab(forward: bool) -> RightMode {
    SCREEN.with(|slot| {
        let Some(old) = slot else { return RightMode::Closed };
        let fi = old.focused_action.min(old.actions.len().saturating_sub(1));
        let n = old.actions[fi].tabs.len();
        if n <= 1 {
            return old.right();
        }
        let a = &mut old.actions[fi];
        a.active = if forward {
            (a.active + 1) % n
        } else {
            (a.active + n - 1) % n
        };
        old.repaint_action();
        old.right()
    })
}

/// Select tab `i` on the focused action column.
pub fn select_tab(i: usize) -> RightMode {
    SCREEN.with(|slot| {
        let Some(old) = slot else { return RightMode::Closed };
        let fi = old.focused_action.min(old.actions.len().saturating_sub(1));
        if i >= old.actions[fi].tabs.len() {
            return old.right();
        }
        old.actions[fi].active = i;
        old.repaint_action();
        old.right()
    })
}

/// The tab index under pixel `(x, y)` on the **focused** action column's bar.
pub fn tab_hit(x: u64, y: u64) -> Option<usize> {
    SCREEN.with(|slot| {
        let sc = slot.as_ref()?;
        tab_hit_in(sc, sc.focused_action, x, y)
    })
}

/// The tab index under `(x, y)` on action column `pane_i`'s bar.
pub fn tab_hit_in_pane(pane_i: usize, x: u64, y: u64) -> Option<usize> {
    SCREEN.with(|slot| {
        let sc = slot.as_ref()?;
        tab_hit_in(sc, pane_i, x, y)
    })
}

fn tab_hit_in(sc: &Screen, pane_i: usize, x: u64, y: u64) -> Option<usize> {
    let a = sc.actions.get(pane_i)?;
    if a.tabs.is_empty() || a.pane.w == 0 {
        return None;
    }
    let ty = a.pane.y + BORDER + 4;
    if y < ty || y >= ty + sc.ch() {
        return None;
    }
    sc.tab_layout_for(pane_i)
        .into_iter()
        .position(|(_, tx, w)| x >= tx && x < tx + w)
}

/// Where a dragged tab dropped at `(x, y)` should be inserted in column
/// `pane_i`: before the tab under the cursor if the drop landed on the tab bar,
/// otherwise at the end (a drop anywhere in the body appends).
pub fn drop_index_in_pane(pane_i: usize, x: u64, y: u64) -> usize {
    SCREEN.with(|slot| {
        let Some(sc) = slot.as_ref() else { return 0 };
        let len = sc.actions.get(pane_i).map(|a| a.tabs.len()).unwrap_or(0);
        tab_hit_in(sc, pane_i, x, y).unwrap_or(len)
    })
}

/// Which action column contains `(x,y)`, if any.
pub fn action_pane_at(x: u64, y: u64) -> Option<usize> {
    SCREEN.with(|slot| {
        let sc = slot.as_ref()?;
        for (i, a) in sc.actions.iter().enumerate() {
            if a.pane.w == 0 {
                continue;
            }
            if x >= a.pane.x
                && x < a.pane.x + a.pane.w
                && y >= a.pane.y
                && y < a.pane.y + a.pane.h
            {
                return Some(i);
            }
        }
        None
    })
}

/// Move a tab from one action column to another (drag-drop). Pure list surgery
/// on the live screen. Returns false if the move is invalid.
pub fn move_tab_between(from_pane: usize, from_idx: usize, to_pane: usize, to_idx: usize) -> bool {
    SCREEN.with(|slot| {
        let Some(sc) = slot else { return false };
        if from_pane >= sc.actions.len() || to_pane >= sc.actions.len() {
            return false;
        }
        if from_idx >= sc.actions[from_pane].tabs.len() {
            return false;
        }
        let mode = sc.actions[from_pane].tabs.remove(from_idx);
        // Fix active on source.
        if sc.actions[from_pane].active >= sc.actions[from_pane].tabs.len()
            && !sc.actions[from_pane].tabs.is_empty()
        {
            sc.actions[from_pane].active = sc.actions[from_pane].tabs.len() - 1;
        }
        let insert = crate::panes_layout::insert_index(
            from_pane,
            from_idx,
            to_pane,
            sc.actions[to_pane].tabs.len(),
            to_idx,
        );
        sc.actions[to_pane].tabs.insert(insert, mode);
        sc.actions[to_pane].active = insert;
        sc.focused_action = to_pane;
        sc.focus_action = true;
        true
    })
}

/// Highlight action column `target` as the live drop target during a tab drag
/// (accent frame), clearing any previously highlighted column.
///
/// Only repaints the two frames that changed, and only when the target actually
/// moved — a mouse drag fires on every pointer report, so repainting the band
/// each time would flicker the whole action band while dragging.
pub fn highlight_drop_target(target: Option<usize>) {
    SCREEN.with(|slot| {
        let Some(sc) = slot else { return };
        if sc.drop_target == target {
            return;
        }
        let prev = sc.drop_target;
        sc.drop_target = target;
        sc.cursor_restore();
        for i in prev.into_iter().chain(target) {
            if !sc.column_visible(i) {
                continue;
            }
            let active = Some(i) == target || (sc.focus_action && i == sc.focused_action);
            sc.draw_frame_titled(&sc.actions[i].pane, active, "");
            sc.draw_tab_bar_for(i);
            sc.draw_close_btn_for(i);
        }
        sc.cursor_overlay();
    });
}

/// Close the tab of `mode` if open (any column).
pub fn close_tab_mode(mode: RightMode) {
    SCREEN.with(|slot| {
        let Some(old) = slot else { return };
        if let Some((pi, ti)) = old.find_mode(mode) {
            old.focused_action = pi;
            old.actions[pi].active = ti;
            close_active_slot(slot);
        }
    });
}

/// Close the active tab of the focused action column.
fn close_active_slot(slot: &mut Option<Screen>) {
    let Some(old) = slot else { return };
    let fi = old.focused_action.min(old.actions.len().saturating_sub(1));
    if old.actions[fi].tabs.is_empty() {
        return;
    }
    let ai = old.actions[fi].active.min(old.actions[fi].tabs.len() - 1);
    old.actions[fi].tabs.remove(ai);
    if old.actions[fi].active >= old.actions[fi].tabs.len()
        && !old.actions[fi].tabs.is_empty()
    {
        old.actions[fi].active = old.actions[fi].tabs.len() - 1;
    }
    let any = old.any_action_open();
    // Closing a tab invalidates the other panes' interiors either way: the collapse
    // branch rebuilds the screen, and the grid branch repaints frames while leaving
    // each view's own pixels to be redrawn. Marked here so both are covered — the grid
    // case is the one that shipped broken, because `repaint_action` looks like it has
    // already done the work.
    mark_tabs_dirty();
    // A lone action pane collapses the band when its last tab closes (the classic
    // two-pane behaviour); a grid keeps its now-empty pane as a drop target.
    if !any && old.actions.len() == 1 {
        let ns = rebuilt(old, false);
        ns.redraw();
        *slot = Some(ns);
    } else {
        old.repaint_action();
    }
}

/// Open the ktrace log stream in the action pane.
pub fn open_ktrace() {
    set_right(RightMode::Ktrace);
}

/// Close the **active** tab (chat becomes full-width once the last tab closes).
pub fn close_action() {
    SCREEN.with(close_active_slot);
}


// --- pixel plumbing --------------------------------------------------

/// Which action pane's close button the pointer is over (mouse hover).
static HOVER_CLOSE: crate::mm::Locked<Option<usize>> = crate::mm::Locked::new(None);
/// Which `(pane, tab)` the pointer is over (mouse hover).
static HOVER_TAB: crate::mm::Locked<Option<(usize, usize)>> = crate::mm::Locked::new(None);

/// Recompute the hovered close button / tab from the pointer and repaint the
/// affected action panes' chrome when the hover set changed. Called on mouse
/// move; returns true when something was repainted (so the shell can skip
/// further work).
pub fn update_hover(x: u64, y: u64) -> bool {
    let (close, tab) = SCREEN.with(|slot| {
        let Some(sc) = slot else { return (None, None) };
        let mut close = None;
        let mut tab = None;
        for i in 0..sc.actions.len() {
            if !sc.column_visible(i) {
                continue;
            }
            let (cx, cy, cw_, ch_) = sc.close_btn_for(i);
            if cx > 0 && x >= cx && x < cx + cw_ && y >= cy && y < cy + ch_ {
                close = Some(i);
            }
            if tab_hit_in(sc, i, x, y).is_some() && !sc.actions[i].tabs.is_empty() {
                tab = Some((i, tab_hit_in(sc, i, x, y).unwrap_or(0)));
            }
        }
        (close, tab)
    });
    let changed = HOVER_CLOSE.with(|h| *h != close) || HOVER_TAB.with(|h| *h != tab);
    if !changed {
        return false;
    }
    // Repaint the union of the **old and new** hovered panes: clearing a hover
    // (moving the pointer off) is itself a change that must repaint the pane
    // the chip/label was drawn on — the old hover only, since the new is empty.
    let old_close = HOVER_CLOSE.with(|h| *h);
    let old_tab = HOVER_TAB.with(|h| *h);
    HOVER_CLOSE.with(|h| *h = close);
    HOVER_TAB.with(|h| *h = tab);
    let panes: alloc::collections::BTreeSet<usize> = [
        old_close,
        close,
        old_tab.map(|(i, _)| i),
        tab.map(|(i, _)| i),
    ]
    .into_iter()
    .flatten()
    .collect();
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            for i in panes {
                if sc.column_visible(i) {
                    sc.draw_tab_bar_for(i);
                    sc.draw_close_btn_for(i);
                }
            }
        }
    });
    true
}


impl Screen {
    /// Focused action slot (falls back to 0).
    fn focused_slot(&self) -> &ActionSlot {
        let i = self.focused_action.min(self.actions.len().saturating_sub(1));
        &self.actions[i]
    }

    fn focused_slot_mut(&mut self) -> &mut ActionSlot {
        let i = self.focused_action.min(self.actions.len().saturating_sub(1));
        &mut self.actions[i]
    }

    /// Active tab of the focused action column.
    pub(super) fn right(&self) -> RightMode {
        if self.actions.is_empty() {
            RightMode::Closed
        } else {
            self.focused_slot().right()
        }
    }

    /// True if any action column has at least one tab.
    pub(super) fn any_action_open(&self) -> bool {
        self.actions.iter().any(|a| !a.is_empty())
    }

    /// Primary geometry pane for the focused action column (`logs` legacy).
    pub(super) fn logs(&self) -> &Pane {
        &self.focused_slot().pane
    }

    fn logs_mut(&mut self) -> &mut Pane {
        &mut self.focused_slot_mut().pane
    }

    /// Whether action pane `i` should be painted.
    ///
    /// A parked pane (`w == 0`, fullscreen or a collapsed band) never paints. An
    /// *empty* pane paints its frame only in a multi-pane grid, where it is a
    /// visible drop target; a lone action pane collapses when its last tab
    /// closes, keeping the classic two-pane look byte-identical.
    ///
    /// The test is the **grid's own** pane count, not `layout.max_panes`: the two
    /// are kept in sync but deriving visibility from the geometry that is
    /// actually laid out means a stale config can't make them disagree.
    pub(super) fn column_visible(&self, i: usize) -> bool {
        let Some(a) = self.actions.get(i) else { return false };
        if a.pane.w == 0 {
            return false;
        }
        !a.is_empty() || self.actions.len() > 1
    }

    /// The visible pane of the action column whose **active** tab is `mode`.
    ///
    /// The per-view painters (`draw_top`, the audio/video/browser HUDs, the
    /// editor) resolve their target through this rather than the focused column:
    /// a `/top` tab on column 2 must keep refreshing while the user works in
    /// column 1, and it must paint into its own column's rectangle.
    pub(super) fn mode_dims(&self, mode: RightMode) -> Option<PaneDims> {
        let i = self.mode_column(mode)?;
        Some(PaneDims::of(&self.actions[i].pane))
    }

    /// Index of the visible action column whose **active** tab is `mode`.
    pub(super) fn mode_column(&self, mode: RightMode) -> Option<usize> {
        (0..self.actions.len())
            .find(|&i| self.actions[i].right() == mode && self.actions[i].pane.w > 0)
    }

    /// Whether any action column is painted — i.e. the band is up, so keyboard
    /// focus can move to it. With `max_panes > 2` an *empty* column is visible
    /// and focusable (it is a drop target), which is why this is not the same
    /// question as "does the focused column have a tab".
    pub(super) fn any_column_visible(&self) -> bool {
        (0..self.actions.len()).any(|i| self.column_visible(i))
    }

    /// Find which column owns `mode`.
    pub(super) fn find_mode(&self, mode: RightMode) -> Option<(usize, usize)> {
        for (pi, a) in self.actions.iter().enumerate() {
            if let Some(ti) = a.tabs.iter().position(|&m| m == mode) {
                return Some((pi, ti));
            }
        }
        None
    }

/// The action-pane close-button rectangle `(x, y, w, h)` — FA `xmark` at the
/// top-right of the action pane title. Only meaningful when the pane is open.
/// Geometry is shared by the renderer and the click hit-test so they cannot
/// disagree. Width matches the square FA cell (body line height) so the mark
/// isn't squeezed into a mono column.
pub(super) fn close_btn_for(&self, pane_i: usize) -> (u64, u64, u64, u64) {
        let w = self.ch().max(self.cw() * 2);
        let Some(a) = self.actions.get(pane_i) else {
            return (0, 0, 0, 0);
        };
        let x = (a.pane.x + a.pane.w).saturating_sub(BORDER + PAD + w);
        let y = a.pane.y + BORDER + 4;
        (x, y, w, self.ch())
    }

    /// Repaint **only the action pane** for a tab switch (geometry unchanged):
    /// clear its interior once for the new tab, redraw its frame + tab bar, and
    /// re-render ktrace from its grid. The chat pane and the whole background are
    /// left untouched — so switching tabs never flickers the rest of the screen.
    /// The active tab's dynamic interior (top/audio/image/editor) is repainted by
    /// the shell right after (`repaint_active_tab`).
    fn repaint_action(&mut self) {
        self.cursor_restore();
        self.draw_frame(&self.chat, !self.action_focused());
        for i in 0..self.actions.len() {
            if !self.column_visible(i) {
                continue;
            }
            let a = &self.actions[i];
            let focused = self.focus_action && i == self.focused_action;
            self.paint_surface(
                a.pane.ix,
                a.pane.iy,
                a.pane.cols * self.cw(),
                a.pane.rows * self.ch(),
                a.pane.bg,
            );
            self.draw_frame_titled(&a.pane, focused, "");
            self.draw_tab_bar_for(i);
            self.draw_close_btn_for(i);
            if a.right() == RightMode::Ktrace {
                self.render_view(&a.pane);
            }
        }
        self.cursor_overlay();
    }

    pub(super) fn draw_close_btn_for(&self, pane_i: usize) {
        let Some(a) = self.actions.get(pane_i) else { return };
        // An empty drop-target column has nothing to close.
        if a.pane.w == 0 || a.is_empty() {
            return;
        }
        let (x, y, w, _) = self.close_btn_for(pane_i);
        // Hover: fill a subtle chip so the clickable mark reads as a button.
        let hovered = HOVER_CLOSE.with(|h| *h == Some(pane_i));
        if hovered {
            let chip = self.mix(a.pane.bg, self.theme.accent, 0.18);
            self.fill_rect(x, y, w, self.ch(), chip);
        }
        // Font Awesome xmark in a square line-height cell (see `glyph_cell`),
        // centred in the hit box; ink from the live theme accent.
        let mark = crate::icons::close_mark();
        let (iw, _) = self.glyph_cell(mark);
        let ix = x + w.saturating_sub(iw) / 2;
        let ink = if hovered { self.lighten(self.theme.accent, 0.35) } else { self.theme.accent };
        self.blit_glyph(ix, y, mark, ink, if hovered { self.mix(a.pane.bg, self.theme.accent, 0.18) } else { a.pane.bg });
    }

    /// Per-tab header layout for action column `pane_i`.
    fn tab_layout_for(&self, pane_i: usize) -> Vec<(RightMode, u64, u64)> {
        let Some(a) = self.actions.get(pane_i) else {
            return Vec::new();
        };
        let cw = self.cw();
        let mut x = a.pane.x + BORDER + PAD;
        let mut out = Vec::with_capacity(a.tabs.len());
        for &m in &a.tabs {
            let lab = tab_label(m);
            let w = (lab.chars().count() as u64 + 1) * cw;
            out.push((m, x, w));
            x += w + cw;
        }
        out
    }

    pub(super) fn draw_tab_bar_for(&self, pane_i: usize) {
        let Some(a) = self.actions.get(pane_i) else { return };
        if a.pane.w == 0 {
            return;
        }
        // Share the close button's geometry so the bar always stops exactly
        // where the `[x]` starts. `close_btn_for` saturates, which matters now
        // that a grid pane can be far narrower than the old single action pane.
        let (close_x, ty, ..) = self.close_btn_for(pane_i);
        let x0 = a.pane.x + BORDER + PAD;
        self.fill_rect(x0, ty, close_x.saturating_sub(x0), self.ch(), a.pane.bg);
        for (i, (m, x, w)) in self.tab_layout_for(pane_i).into_iter().enumerate() {
            if x + w >= close_x {
                break;
            }
            let is_active = i == a.active;
            let hovered = HOVER_TAB.with(|h| *h == Some((pane_i, i)));
            let fg = if is_active {
                self.theme.title_active
            } else if hovered {
                self.lighten(self.theme.title_dim, 0.45)
            } else {
                self.theme.title_dim
            };
            let mut lx = x;
            if is_active {
                lx = self.draw_str(lx, ty, ">", self.theme.accent, a.pane.bg);
            }
            let lab = tab_label(m);
            self.draw_str(lx, ty, &lab, fg, a.pane.bg);
            // Hovered tab gets a thin accent underline (a clickable affordance).
            if hovered && !is_active {
                self.fill_rect(x, ty + self.ch() - 2, w, 2, self.mix(a.pane.bg, self.theme.accent, 0.5));
            }
        }
    }
}
