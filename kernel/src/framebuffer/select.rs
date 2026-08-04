//! Mouse text selection in the chat pane, and the folded-output notes it
//! shares its cell addressing with.

use super::*;

/// Map a pixel to a chat-pane cell `(absolute line, col)`. With `clamp`, a
/// point outside the interior snaps to the nearest cell (so a drag past the
/// pane edge keeps extending); without it, outside is `None`.
fn chat_abs_cell(sc: &Screen, x: u64, y: u64, clamp: bool) -> Option<(usize, usize)> {
    let p = &sc.chat;
    let (x0, y0) = (p.ix, p.iy);
    let (x1, y1) = (p.ix + p.cols * p.cw, p.iy + p.rows * p.ch);
    let (cx, cy) = if clamp {
        (x.clamp(x0, x1 - 1), y.clamp(y0, y1 - 1))
    } else {
        if x < x0 || x >= x1 || y < y0 || y >= y1 {
            return None;
        }
        (x, y)
    };
    let col = (((cx - x0) / p.cw) as usize).min(p.cols as usize - 1);
    let row = ((cy - y0) / p.ch) as usize;
    let first = p.hist.len() - p.view.min(p.hist.len());
    Some((first + row, col))
}

/// Sprite-safe wrapper around a selection-highlight update: hide the cursor,
/// apply `paint`, redraw the caret if the view is live, restore the cursor.
fn chat_sel_with_cursor(sc: &mut Screen, paint: impl FnOnce(&mut Screen)) {
    sc.cursor_restore();
    sc.cur_vis = false;
    paint(sc);
    if sc.chat.view == 0 {
        if sc.composer_active {
            sc.paint_composer_caret();
        } else {
            sc.caret_draw(&sc.chat); // no-op when has_composer
        }
    }
    sc.cursor_overlay();
}

/// Begin a mouse text selection at pixel `(x, y)`; replaces any previous one.
/// No-op (but still clears) outside the chat pane interior.
pub fn chat_sel_begin(x: u64, y: u64) {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            let cell = chat_abs_cell(sc, x, y, false);
            let old = sc.chat.sel.take();
            sc.chat.sel = cell.map(|c| (c, c));
            if old.is_some() || cell.is_some() {
                let new = sc.chat.sel;
                chat_sel_with_cursor(sc, |sc| sc.repaint_sel_diff(&sc.chat, old, new));
            }
        }
    });
}

/// Extend the active selection's head to the cell under `(x, y)` (clamped into
/// the pane, so dragging past an edge selects to it). No-op without an anchor.
pub fn chat_sel_drag(x: u64, y: u64) {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            let Some((anchor, head)) = sc.chat.sel else { return };
            let Some(new_head) = chat_abs_cell(sc, x, y, true) else { return };
            if new_head != head {
                let old = Some((anchor, head));
                sc.chat.sel = Some((anchor, new_head));
                let new = sc.chat.sel;
                chat_sel_with_cursor(sc, |sc| sc.repaint_sel_diff(&sc.chat, old, new));
            }
        }
    });
}

/// Finish the selection on mouse release: returns the selected text when it
/// spans more than one cell (a plain click copies nothing and just clears any
/// stale highlight). The highlight stays visible until the next click.
pub fn chat_sel_end() -> Option<String> {
    SCREEN.with(|slot| {
        let sc = slot.as_mut()?;
        let (a, b) = sc.chat.sel?;
        if a == b {
            let old = sc.chat.sel.take();
            chat_sel_with_cursor(sc, |sc| sc.repaint_sel_diff(&sc.chat, old, None));
            return None;
        }
        let p = &sc.chat;
        let cols = p.cols as usize;
        let text = crate::textsel::selection_text(
            |i| {
                if i < p.hist.len() {
                    Some(p.hist[i].as_slice())
                } else {
                    let gr = i - p.hist.len();
                    (gr < p.rows as usize && p.grid.len() >= (gr + 1) * cols).then(|| &p.grid[gr * cols..(gr + 1) * cols])
                }
            },
            a,
            b,
        );
        (!text.is_empty()).then_some(text)
    })
}

/// Drop any chat selection and its highlight (e.g. a click somewhere else).
pub fn chat_sel_clear() {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            let old = sc.chat.sel.take();
            if old.is_some() {
                chat_sel_with_cursor(sc, |sc| sc.repaint_sel_diff(&sc.chat, old, None));
            }
        }
    });
}

// --- expandable folds ---------------------------------------------------
// A tool result is printed truncated with a clickable "▸ N more…" line; the
// hidden remainder is registered against that line's absolute index. A single
// click on it reveals the rest. Additive over the scrollback (no render-loop
// or scroll changes), so selection/scroll are unaffected.

/// The absolute line index (`gi`, same coords as `sel`) the chat cursor is on —
/// i.e. where the next printed line will land. Anchors a fold to its "▸ more…".
pub fn chat_current_gi() -> usize {
    SCREEN.with(|slot| {
        slot.as_ref().map(|sc| sc.chat.hist.len() + sc.chat.row as usize).unwrap_or(0)
    })
}

/// Register a fold: the line at `gi` reveals `hidden` (pre-styled text, may
/// contain ANSI + newlines) when clicked. Bounded so it can't grow unbounded.
pub fn chat_note_fold(gi: usize, hidden: &str) {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.chat.folds.push((gi, hidden.to_string()));
            let n = sc.chat.folds.len();
            if n > 64 {
                sc.chat.folds.drain(0..n - 64);
            }
        }
    });
}

/// The absolute line a single-cell click hit (anchor == head, i.e. not a drag),
/// for matching a fold. Call **before** [`chat_sel_end`] (which clears sel).
pub fn chat_click_gi() -> Option<usize> {
    SCREEN.with(|slot| {
        let sc = slot.as_ref()?;
        let (a, b) = sc.chat.sel?;
        (a == b).then_some(a.0)
    })
}

/// Take the hidden text of the fold anchored at line `gi` (removing it), if any.
/// The shell prints the returned text to reveal the collapsed output.
pub fn chat_take_fold(gi: usize) -> Option<String> {
    SCREEN.with(|slot| {
        let sc = slot.as_mut()?;
        let pos = sc.chat.folds.iter().position(|(g, _)| *g == gi)?;
        Some(sc.chat.folds.remove(pos).1)
    })
}

/// Wipe the chat pane's text (grid + scrollback) and repaint it — `/clear`.
pub fn clear_chat() {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.cursor_restore();
            sc.cur_vis = false;
            sc.chat.clear_content();
            sc.paint_surface(sc.chat.ix, sc.chat.iy, sc.chat.cols * sc.chat.cw, sc.chat.rows * sc.chat.ch, sc.chat.bg);
            if sc.chat.has_composer {
                sc.draw_composer();
            } else {
                sc.caret_draw(&sc.chat);
            }
            sc.cursor_overlay();
        }
    });
}
