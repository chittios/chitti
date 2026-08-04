//! Which pane has keyboard focus, and scrolling whichever one does.

use super::*;

/// The action column whose `[x]` close button is under `(x, y)`, if any.
///
/// Returns the column index rather than a bool so a click on a **non-focused**
/// column's `[x]` closes that column's tab: testing only the focused column's
/// button meant the other columns' `[x]` were painted but dead.
pub fn close_hit_pane(x: u64, y: u64) -> Option<usize> {
    SCREEN.with(|slot| {
        let sc = slot.as_ref()?;
        (0..sc.actions.len()).find(|&i| {
            if sc.actions[i].is_empty() || !sc.column_visible(i) {
                return false;
            }
            let (bx, by, bw, bh) = sc.close_btn_for(i);
            x >= bx && x < bx + bw && y >= by && y < by + bh
        })
    })
}

/// Focus action column `i` (for click / drop).
pub fn focus_action_column(i: usize) {
    SCREEN.with(|slot| {
        let Some(sc) = slot else { return };
        if i >= sc.actions.len() || !sc.column_visible(i) {
            return;
        }
        let was = (sc.focused_action, sc.focus_action);
        sc.focused_action = i;
        sc.focus_action = true;
        if was == (i, true) {
            return; // already the selected pane
        }
        // Repaint here rather than leaving it to `focus_set`: this function has
        // already moved focus onto the action side, so `focus_set(true)` would
        // see no flip and draw nothing — which is exactly why clicking a pane
        // used to change the selection invisibly.
        sc.cursor_restore();
        sc.cur_vis = false;
        sc.draw_frame(&sc.chat, !sc.action_focused());
        // Every visible pane's frame carries the selection state, so the pane
        // losing focus must be redrawn too, not just the one gaining it.
        for j in 0..sc.actions.len() {
            if !sc.column_visible(j) {
                continue;
            }
            let active = sc.focus_action && j == sc.focused_action;
            sc.draw_frame_titled(&sc.actions[j].pane, active, "");
            sc.draw_tab_bar_for(j);
            sc.draw_close_btn_for(j);
        }
        if sc.chat.has_composer {
            sc.draw_composer();
        }
        sc.cursor_overlay();
    });
}

/// Move keyboard focus to the next/previous action pane (grid order), returning
/// the newly focused index. Skips parked panes.
pub fn focus_cycle_column(forward: bool) -> usize {
    // Pick the target inside the lock, then repaint outside it —
    // `focus_action_column` takes `SCREEN` itself and re-entering would deadlock.
    let target = SCREEN.with(|slot| {
        let Some(sc) = slot.as_ref() else { return 0 };
        let visible: Vec<usize> =
            (0..sc.actions.len()).filter(|&i| sc.column_visible(i)).collect();
        if visible.len() < 2 {
            return sc.focused_action;
        }
        let at = visible.iter().position(|&i| i == sc.focused_action).unwrap_or(0);
        let next = if forward {
            (at + 1) % visible.len()
        } else {
            (at + visible.len() - 1) % visible.len()
        };
        visible[next]
    });
    focus_action_column(target);
    target
}

/// Pure focus-cycle math — re-export from [`crate::panes_layout`] so call sites
/// can stay in the framebuffer API. Tests live next to the pure function
/// (framebuffer itself is gated out of the test binary).
pub use crate::panes_layout::cycle_focus_target;

/// Cycle keyboard focus across the shell chat, action panes, and in-pane tabs.
/// Ctrl+Tab / Ctrl+Shift+Tab. Returns true if an action pane holds focus after
/// the move.
///
/// Order (forward): chat → pane0/tab0 → pane0/tab1 → … → pane1/tab0 → … → chat.
/// Within a focused action column that has several tabs, Ctrl+Tab walks those
/// tabs first; only after the last tab does focus move to the next column (or
/// back to the shell). Parked columns are skipped. No action pane open → shell.
pub fn focus_cycle_all(forward: bool) -> bool {
    // 1) If already on an action column with more tabs in this direction,
    //    advance the tab and stay — keyboard-first tab bar.
    let tab_step = SCREEN.with(|slot| {
        let Some(sc) = slot.as_ref() else {
            return false;
        };
        if !sc.focus_action || !sc.any_column_visible() {
            return false;
        }
        let fi = sc.focused_action.min(sc.actions.len().saturating_sub(1));
        let n = sc.actions.get(fi).map(|a| a.tabs.len()).unwrap_or(0);
        if n <= 1 {
            return false;
        }
        let active = sc.actions[fi].active;
        if forward && active + 1 < n {
            return true; // will cycle_tab below
        }
        if !forward && active > 0 {
            return true;
        }
        false
    });
    if tab_step {
        cycle_tab(forward);
        // Ensure action still holds focus (cycle_tab does not touch it).
        focus_set(true);
        return true;
    }

    // 2) Otherwise walk the chat ↔ action-column ring.
    let (to_action, target) = SCREEN.with(|slot| {
        let Some(sc) = slot.as_ref() else {
            return (false, 0usize);
        };
        let visible: Vec<usize> = (0..sc.actions.len())
            .filter(|&i| sc.column_visible(i))
            .collect();
        let at_action = sc.focus_action && !visible.is_empty();
        crate::panes_layout::cycle_focus_target(
            &visible,
            at_action,
            sc.focused_action,
            forward,
        )
    });
    if to_action {
        // Landing on a column from the shell (or another pane): for reverse
        // walks, start at the last tab so reverse is the true inverse of
        // forward's "exhaust tabs then leave".
        if !forward {
            SCREEN.with(|slot| {
                if let Some(sc) = slot {
                    if target < sc.actions.len() {
                        let n = sc.actions[target].tabs.len();
                        if n > 0 {
                            sc.actions[target].active = n - 1;
                        }
                    }
                }
            });
        } else {
            // Forward into a new column → first tab.
            SCREEN.with(|slot| {
                if let Some(sc) = slot {
                    if target < sc.actions.len() && !sc.actions[target].tabs.is_empty() {
                        sc.actions[target].active = 0;
                    }
                }
            });
        }
        focus_action_column(target);
        true
    } else {
        focus_set(false);
        false
    }
}

/// Scroll a pane's view by `delta` lines (`+` = back in time, `-` = toward
/// live); `action` picks the **focused** action pane's ktrace, else chat. Snaps
/// caret handling automatically: the caret only draws on a live view.
pub fn scroll_view(action: bool, delta: i64) {
    let target = if action {
        ScrollTarget::Action(focused_action_index())
    } else {
        ScrollTarget::Chat
    };
    scroll_target(target, delta);
}

/// Which view a scroll applies to.
#[derive(Clone, Copy)]
pub enum ScrollTarget {
    Chat,
    /// Action pane by grid index — the pane under the mouse pointer, which with a
    /// grid need not be the focused one.
    Action(usize),
}

/// Scroll the ktrace view of a specific action pane (mouse wheel over it).
pub fn scroll_action_pane(i: usize, delta: i64) {
    scroll_target(ScrollTarget::Action(i), delta);
}

fn scroll_target(target: ScrollTarget, delta: i64) {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            let action = matches!(target, ScrollTarget::Action(_));
            let idx = match target {
                ScrollTarget::Chat => 0,
                ScrollTarget::Action(i) => {
                    // Only a ktrace view has scrollback to move.
                    if sc.actions.get(i).map(|a| a.right()) != Some(RightMode::Ktrace) {
                        return;
                    }
                    i
                }
            };
            sc.cursor_restore();
            sc.cur_vis = false;
            let p = if action { &mut sc.actions[idx].pane } else { &mut sc.chat };
            let max = p.hist.len();
            let v = (p.view as i64 + delta).clamp(0, max as i64) as usize;
            if v != p.view {
                p.view = v;
                let p = if action { &sc.actions[idx].pane } else { &sc.chat };
                sc.render_view(p);
                if !action && v == 0 {
                    sc.caret_draw(&sc.chat); // no-op when chat has_composer
                }
            }
            sc.cursor_overlay();
        }
    });
}

/// Scroll a pane's view by one page (its row count minus one).
pub fn scroll_page(action: bool, up: bool) {
    let rows = SCREEN.with(|slot| {
        slot.as_ref().map(|sc| if action { sc.logs().rows } else { sc.chat.rows }).unwrap_or(1)
    }) as i64;
    scroll_view(action, if up { rows - 1 } else { -(rows - 1) });
}

/// Snap a pane back to the live view (offset 0).
pub fn scroll_live(action: bool) {
    scroll_view(action, i64::MIN / 2);
}

/// Toggle keyboard focus between the chat pane and an open action pane.
/// Returns true if the action pane now holds focus. No-op (false) when the
/// action band is collapsed. Works while the editor tab is open — leaving
/// the editor for the shell is intentional (Ctrl+Tab back).
///
/// When focus returns to the chat pane, the bordered composer is repainted
/// immediately (accent border + caret) so the shell is ready for input without
/// waiting for a keystroke to re-sync.
pub fn focus_toggle() -> bool {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            if !sc.any_column_visible() {
                return false;
            }
            sc.focus_action = !sc.focus_action;
            sc.cursor_restore();
            sc.cur_vis = false;
            sc.draw_frame(&sc.chat, !sc.action_focused());
            // Repaint every visible pane's chrome: the frame carries the
            // selection state, so panes *losing* it must be redrawn too.
            for j in 0..sc.actions.len() {
                if !sc.column_visible(j) {
                    continue;
                }
                let active = sc.action_focused() && j == sc.focused_action;
                sc.draw_frame_titled(&sc.actions[j].pane, active, "");
                sc.draw_tab_bar_for(j);
                sc.draw_close_btn_for(j);
            }
            // Composer chrome reflects focus (accent border + caret only when
            // the chat holds keyboard focus). Force caret on so it is visible
            // the instant focus returns — no need to type first.
            if sc.chat.has_composer {
                if !sc.action_focused() {
                    sc.caret_on = true;
                }
                sc.draw_composer();
            }
            sc.cursor_overlay();
            sc.focus_action
        } else {
            false
        }
    })
}

/// Whether keyboard focus is on the action pane (see [`focus_toggle`]).
pub fn focus_is_action() -> bool {
    SCREEN.with(|slot| slot.as_ref().map(|sc| sc.action_focused()).unwrap_or(false))
}

/// Give keyboard focus to the action pane (true) or the chat pane (false),
/// e.g. from a mouse click or Ctrl+Tab. Same constraints as [`focus_toggle`].
///
/// Always refreshes the composer when focusing the chat, even if focus was
/// already on chat — so a click on the shell agent immediately arms the input.
/// Leaving the editor for the shell is allowed (editor tab stays open).
pub fn focus_set(action: bool) {
    let (flips, need_composer) = SCREEN.with(|slot| {
        slot.as_ref()
            .map(|sc| {
                let closed = !sc.any_column_visible();
                // Focusing action with no panes is a no-op; focusing chat always
                // works (and must work while the editor tab is open).
                let flips = if action {
                    !closed && sc.focus_action != action
                } else {
                    sc.focus_action != action
                };
                // Focusing chat: repaint composer even when already focused so
                // the caret/border activate without a first keystroke.
                let need_composer = !action && sc.chat.has_composer;
                (flips, need_composer)
            })
            .unwrap_or((false, false))
    });
    if flips {
        // Direct path when only clearing focus_action (focus_toggle needs a
        // visible band; chat focus while band collapsed still needs composer).
        let can_toggle = SCREEN.with(|slot| {
            slot.as_ref()
                .map(|sc| sc.any_column_visible())
                .unwrap_or(false)
        });
        if can_toggle {
            focus_toggle();
            // focus_toggle already re-armed the composer when leaving action.
            return;
        }
        SCREEN.with(|slot| {
            if let Some(sc) = slot {
                sc.focus_action = action;
            }
        });
    }
    if !action && need_composer {
        // Already on chat (click re-arm) or band collapsed with focus cleared.
        SCREEN.with(|slot| {
            if let Some(sc) = slot {
                sc.cursor_restore();
                sc.cur_vis = false;
                sc.focus_action = false;
                sc.caret_on = true;
                sc.draw_frame(&sc.chat, true);
                sc.draw_composer();
                sc.cursor_overlay();
            }
        });
    }
}

pub fn pane_hit(x: u64, y: u64) -> Option<bool> {
    SCREEN.with(|slot| {
        slot.as_ref().and_then(|sc| {
            let hit = |p: &Pane| p.w > 0 && x >= p.x && x < p.x + p.w && y >= p.y && y < p.y + p.h;
            // Any visible action column counts as "action", not just the focused
            // one — otherwise a click/scroll over column 2 fell through to chat.
            if (0..sc.actions.len()).any(|i| sc.column_visible(i) && hit(&sc.actions[i].pane)) {
                Some(true)
            } else if hit(&sc.chat) {
                Some(false)
            } else {
                None
            }
        })
    })
}

// --- chat-pane mouse text selection ---------------------------------------
//
// The editor already had drag-to-copy; this gives the chat pane the same:
// press anchors a selection, drag extends it (highlight painted by
// `render_view`), release hands the text to the shell for the clipboard.
// Coordinates are absolute over scrollback + grid (`crate::textsel`), so a
// selection stays glued to its text while output scrolls past.

impl Screen {
    /// Whether an action pane holds keyboard focus. Chat keeps focus by
    /// default so you can keep typing; Ctrl+Tab / a click / `/pane focus`
    /// moves it onto the band. The editor is the same rule now — opening it
    /// sets `focus_action` so keys land there, and Ctrl+Tab returns to the
    /// shell without closing the tab.
    pub(super) fn action_focused(&self) -> bool {
        match self.right() {
            RightMode::Closed => false,
            RightMode::Editor
            | RightMode::Ktrace
            | RightMode::Top
            | RightMode::Todos
            | RightMode::Audio
            | RightMode::Surface(_) => self.focus_action,
        }
    }
}
