//! Soft-wrap viewport math for the vim-like `/open` editor.
//!
//! Lives outside [`crate::editor`] so it is available under `cargo xtask test`
//! (the interactive editor is `#[cfg(not(test))]` because it paints into the
//! framebuffer). Pure integer arithmetic — no framebuffer, no I/O.

/// How many screen rows a logical line of `len` characters needs at width `tw`.
/// Empty lines still occupy one row (vim / every terminal editor).
pub fn soft_wraps(len: usize, tw: usize) -> usize {
    let tw = tw.max(1);
    if len == 0 {
        1
    } else {
        (len + tw - 1) / tw
    }
}

/// Visual-row index of `(row, col)` given per-line lengths and text width.
pub fn vis_index(line_lens: &[usize], row: usize, col: usize, tw: usize) -> usize {
    let tw = tw.max(1);
    let mut v = 0usize;
    let row = row.min(line_lens.len().saturating_sub(1));
    for i in 0..row {
        v += soft_wraps(line_lens[i], tw);
    }
    v + col / tw
}

/// Map a visual row back to `(logical_line, wrap_segment)`.
pub fn unvis(line_lens: &[usize], vis: usize, tw: usize) -> (usize, usize) {
    let tw = tw.max(1);
    if line_lens.is_empty() {
        return (0, 0);
    }
    let mut rem = vis;
    for i in 0..line_lens.len() {
        let w = soft_wraps(line_lens[i], tw);
        if rem < w {
            return (i, rem);
        }
        rem -= w;
    }
    let last = line_lens.len() - 1;
    (last, soft_wraps(line_lens[last], tw).saturating_sub(1))
}

/// Where the viewport must start so visual row `vis` is on screen, given the
/// current `top` and a text area `text_rows` screen rows tall.
///
/// The scroll rule itself is two comparisons; what it is really pinning down is
/// **how many rows there are**, which is the thing that was wrong. `text_rows`
/// is the count of rows the painter actually fills, so the last reachable row is
/// `top + text_rows - 1` — off by one here and the bottom row of the editor
/// exists, is painted, and cannot be reached: `j` on the second-to-last row
/// scrolls the buffer instead of moving the cursor, which reads as the last line
/// of the file being unreachable. Pure and outside [`crate::editor`] because that
/// module is `#[cfg(not(test))]`, so a test written next to the bug could never
/// have run.
pub fn scroll_top(top: usize, vis: usize, text_rows: usize) -> usize {
    let rows = text_rows.max(1);
    if vis < top {
        vis
    } else if vis >= top + rows {
        vis + 1 - rows
    } else {
        top
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn the_bottom_painted_row_is_reachable_without_scrolling() {
        // A 24-row text area showing rows 0..=23: the cursor on row 23 must not
        // move the viewport. This is the whole bug — with the row count one short
        // the editor scrolled here, so the file's last line could never be
        // selected and never got a syntax-highlight pass.
        assert_eq!(scroll_top(0, 23, 24), 0);
        assert_eq!(scroll_top(0, 24, 24), 1, "one past the bottom scrolls by one");
        assert_eq!(scroll_top(10, 9, 24), 9, "above the top scrolls up to it");
        assert_eq!(scroll_top(10, 10, 24), 10);
        assert_eq!(scroll_top(10, 33, 24), 10, "top + rows - 1 is still on screen");
        assert_eq!(scroll_top(10, 34, 24), 11);
    }

    #[test_case]
    fn a_one_row_viewport_still_tracks_the_cursor() {
        assert_eq!(scroll_top(5, 5, 1), 5);
        assert_eq!(scroll_top(5, 6, 1), 6);
        // A zero-height area is nonsense a resize can still produce; treat it as
        // one row rather than underflowing into a viewport that shows nothing.
        assert_eq!(scroll_top(5, 6, 0), 6);
    }

    #[test_case]
    fn soft_wraps_empty_and_exact_multiples() {
        assert_eq!(soft_wraps(0, 80), 1);
        assert_eq!(soft_wraps(80, 80), 1);
        assert_eq!(soft_wraps(81, 80), 2);
        assert_eq!(soft_wraps(160, 80), 2);
        assert_eq!(soft_wraps(161, 80), 3);
        assert_eq!(soft_wraps(1, 0), 1); // zero width treated as 1
    }

    #[test_case]
    fn vis_index_and_unvis_round_trip_across_wrapped_lines() {
        // line0: 10 chars → 2 wraps at tw=8
        // line1: 3 chars → 1
        // line2: 20 chars → 3
        let lens = [10usize, 3, 20];
        let tw = 8;
        assert_eq!(soft_wraps(10, tw), 2);
        assert_eq!(soft_wraps(3, tw), 1);
        assert_eq!(soft_wraps(20, tw), 3);
        // start of line 2 = 2+1 = 3
        assert_eq!(vis_index(&lens, 2, 0, tw), 3);
        // col 9 on line 2 → segment 1 → vis 4
        assert_eq!(vis_index(&lens, 2, 9, tw), 4);
        // unvis of every visual row maps back
        let total: usize = lens.iter().map(|&l| soft_wraps(l, tw)).sum();
        for v in 0..total {
            let (row, seg) = unvis(&lens, v, tw);
            assert_eq!(vis_index(&lens, row, seg * tw, tw), v);
        }
    }

    #[test_case]
    fn unvis_past_end_clamps_to_last_segment() {
        let lens = [5usize];
        let (row, seg) = unvis(&lens, 99, 4);
        assert_eq!(row, 0);
        assert_eq!(seg, soft_wraps(5, 4) - 1);
    }
}
