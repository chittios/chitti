//! Pure text-selection math for the chat pane's mouse copy (the framebuffer
//! calls this; keeping it out of `framebuffer/` makes it unit-testable —
//! the compositor is `cfg(not(test))`). Coordinates are `(line, col)` where
//! `line` is an **absolute** index over a pane's scrollback + live grid
//! (0 = oldest scrollback line) and `col` is a cell column; both endpoints
//! are inclusive, like the editor's Visual selection.
//!
//! Also owns **pane reflow** ([`reflow_lines`]): when the chat|action split is
//! resized, soft-wrapped lines must re-wrap to the new column count so expanding
//! a pane fills the extra width (shrink already looked fine because long lines
//! were truncated).

use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::vec::Vec;

/// One character cell as the compositor stores it: `(char, fg colour)`.
/// `'\0'` is the empty/unset cell.
pub type Cell = (char, (u8, u8, u8));

/// Order two selection endpoints so the earlier one comes first.
pub fn normalize(a: (usize, usize), b: (usize, usize)) -> ((usize, usize), (usize, usize)) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Is cell `(line, col)` inside the inclusive, normalized selection `range`?
pub fn contains(range: ((usize, usize), (usize, usize)), line: usize, col: usize) -> bool {
    let ((r1, c1), (r2, c2)) = range;
    if line < r1 || line > r2 {
        return false;
    }
    (line > r1 || col >= c1) && (line < r2 || col <= c2)
}

/// Extract the selected text between endpoints `a` and `b` (any order).
/// `line_at(i)` returns the cells of absolute line `i` (or `None` past the
/// end). Unset cells (byte 0) read as spaces; every line is right-trimmed, so
/// selecting a whole row of a mostly-empty grid yields clean text.
pub fn selection_text<'a, F>(line_at: F, a: (usize, usize), b: (usize, usize)) -> String
where
    F: Fn(usize) -> Option<&'a [Cell]>,
{
    let ((r1, c1), (r2, c2)) = normalize(a, b);
    let mut out = String::new();
    for r in r1..=r2 {
        let Some(cells) = line_at(r) else { break };
        let lo = if r == r1 { c1.min(cells.len()) } else { 0 };
        let hi = if r == r2 { (c2 + 1).min(cells.len()) } else { cells.len() };
        let mut line = String::new();
        for &(ch, _) in &cells[lo..hi] {
            line.push(if ch == '\0' { ' ' } else { ch });
        }
        let trimmed = line.trim_end();
        if r > r1 {
            out.push('\n');
        }
        out.push_str(trimmed);
    }
    out
}

/// Right-trim trailing empty cells (`'\0'`) from a stored line.
fn trim_cells(line: &[Cell]) -> &[Cell] {
    let mut end = line.len();
    while end > 0 && line[end - 1].0 == '\0' {
        end -= 1;
    }
    &line[..end]
}

/// Reflow pane lines from `old_cols` to `new_cols`.
///
/// Soft-wraps (a row that filled `old_cols` completely) are joined into one
/// logical line and re-chunked at `new_cols`, so **expanding** a pane reclaims
/// the unused right margin. Hard newlines (short rows) stay as line breaks.
/// Empty pad cell is `empty` (typically `(0, default_fg)`).
///
/// Returns the full reflowed line list (oldest first), each padded to exactly
/// `new_cols` cells.
pub fn reflow_lines(lines: &[Vec<Cell>], old_cols: usize, new_cols: usize, empty: Cell) -> VecDeque<Vec<Cell>> {
    reflow_lines_mapped(lines, old_cols, new_cols, empty).0
}

/// [`reflow_lines`], plus for each output row the **inclusive range of source rows** whose
/// logical line it came from.
///
/// Line-indexed metadata — the chat pane's prompt/tool-call bands and its click-to-expand
/// folds — has to move with the text when the pane is rebuilt at a new width, and after a
/// re-wrap a row's index bears no fixed relation to where it started. The mapping is
/// produced by the same pass that does the reflow rather than by a second copy of the
/// soft-join rule, so the two cannot disagree about where a logical line began.
pub fn reflow_lines_mapped(
    lines: &[Vec<Cell>],
    old_cols: usize,
    new_cols: usize,
    empty: Cell,
) -> (VecDeque<Vec<Cell>>, Vec<(usize, usize)>) {
    let new_cols = new_cols.max(1);
    let old_cols = old_cols.max(1);
    // 1. Soft-join full-width rows into logical lines, remembering which source rows each
    //    one consumed.
    let mut logical: Vec<(Vec<Cell>, usize, usize)> = Vec::new();
    let mut cur: Vec<Cell> = Vec::new();
    let mut first_src = 0usize;
    for (src, line) in lines.iter().enumerate() {
        if cur.is_empty() {
            first_src = src;
        }
        let content = trim_cells(line);
        // A stored row is "full" (soft-wrap candidate) when its non-pad length
        // equals the old width — i.e. the writer filled the whole row and
        // advanced to the next without a hard newline.
        let was_full = content.len() >= old_cols;
        cur.extend_from_slice(content);
        if !was_full {
            logical.push((core::mem::take(&mut cur), first_src, src));
        }
    }
    if !cur.is_empty() {
        let last = lines.len().saturating_sub(1);
        logical.push((cur, first_src, last));
    }
    // 2. Re-wrap each logical line into `new_cols` chunks.
    let mut out: VecDeque<Vec<Cell>> = VecDeque::new();
    let mut map: Vec<(usize, usize)> = Vec::new();
    for (log, lo, hi) in logical {
        if log.is_empty() {
            out.push_back(alloc::vec![empty; new_cols]);
            map.push((lo, hi));
            continue;
        }
        let mut i = 0;
        while i < log.len() {
            let end = (i + new_cols).min(log.len());
            let mut row = log[i..end].to_vec();
            row.resize(new_cols, empty);
            out.push_back(row);
            map.push((lo, hi));
            i = end;
        }
    }
    (out, map)
}

/// Fit `s` into at most `max` columns, appending `..` when truncated.
/// `max == 0` yields an empty string. Pure / unit-tested — used by the status
/// bar, composer hints, and media HUDs so controls never paint past their box.
pub fn ellipsize(s: &str, max: usize) -> String {
    if max == 0 {
        return String::new();
    }
    let n = s.chars().count();
    if n <= max {
        return String::from(s);
    }
    if max <= 2 {
        return s.chars().take(max).collect();
    }
    let mut t: String = s.chars().take(max - 2).collect();
    t.push_str("..");
    t
}

/// Ellipsize keeping the **end** of `s` (useful for file paths so the basename
/// stays visible). ASCII `..` prefix; never exceeds `max` columns.
pub fn ellipsize_end(s: &str, max: usize) -> String {
    if max == 0 {
        return String::new();
    }
    let n = s.chars().count();
    if n <= max {
        return String::from(s);
    }
    if max <= 2 {
        return s.chars().skip(n.saturating_sub(max)).collect();
    }
    let keep = max - 2;
    let mut t = String::from("..");
    t.extend(s.chars().skip(n.saturating_sub(keep)));
    t
}

/// Pad `s` with trailing spaces to exactly `cols` columns, or ellipsize when
/// longer — so a redraw never leaves residue and never overflows.
pub fn fit_width(s: &str, cols: usize) -> String {
    if cols == 0 {
        return String::new();
    }
    let mut t = ellipsize(s, cols);
    let n = t.chars().count();
    if n < cols {
        t.extend(core::iter::repeat(' ').take(cols - n));
    }
    t
}

/// Map an absolute cursor `(line_index, col)` under the old geometry into the
/// reflowed line list. Best-effort: walks cells in order and counts.
pub fn reflow_cursor(
    lines: &[Vec<Cell>],
    old_cols: usize,
    new_cols: usize,
    old_line: usize,
    old_col: usize,
) -> (usize, usize) {
    let new_cols = new_cols.max(1);
    let old_cols = old_cols.max(1);
    // Count cells (ignoring pads) before the cursor, treating soft wraps as
    // continuous and hard wraps as +1 "newline" that doesn't add a cell.
    let mut cells_before: usize = 0;
    for (li, line) in lines.iter().enumerate() {
        let content = trim_cells(line);
        if li < old_line {
            cells_before += content.len();
            // Hard newline: no extra cell, just starts a new logical line for reflow.
            // Soft wrap: content continues — already counted as cells only.
            let _was_full = content.len() >= old_cols;
            // When counting cells for reflow position we only count printable
            // cells; soft-join means hard-break positions are re-derived from
            // reflow chunking, so we only need the cell offset within the
            // soft-joined stream for lines before cursor.
            continue;
        }
        if li == old_line {
            cells_before += old_col.min(content.len());
            break;
        }
    }
    // Walk the reflowed stream and find (line, col).
    let empty: Cell = ('\0', (0, 0, 0));
    let reflowed = reflow_lines(lines, old_cols, new_cols, empty);
    let mut remaining = cells_before;
    for (ri, row) in reflowed.iter().enumerate() {
        let content = trim_cells(row);
        if remaining <= content.len() {
            return (ri, remaining.min(new_cols.saturating_sub(1)));
        }
        remaining -= content.len();
    }
    let last = reflowed.len().saturating_sub(1);
    (last, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    const FG: (u8, u8, u8) = (250, 249, 245);

    fn row(s: &str, cols: usize) -> Vec<Cell> {
        let mut v: Vec<Cell> = s.chars().map(|c| (c, FG)).collect();
        v.resize(cols, ('\0', FG)); // unset tail cells, like a real grid line
        v
    }

    #[test_case]
    fn normalize_orders_endpoints() {
        assert_eq!(normalize((3, 5), (1, 9)), ((1, 9), (3, 5)));
        assert_eq!(normalize((2, 1), (2, 7)), ((2, 1), (2, 7)));
        assert_eq!(normalize((2, 7), (2, 1)), ((2, 1), (2, 7)));
    }

    #[test_case]
    fn contains_inclusive_range() {
        let r = ((1, 4), (3, 2));
        assert!(!contains(r, 0, 9));
        assert!(!contains(r, 1, 3));
        assert!(contains(r, 1, 4));
        assert!(contains(r, 2, 0)); // middle lines: every column
        assert!(contains(r, 3, 2));
        assert!(!contains(r, 3, 3));
    }

    #[test_case]
    fn single_line_extract() {
        let lines = [row("hello world", 20)];
        let text = selection_text(|i| lines.get(i).map(|v| v.as_slice()), (0, 6), (0, 10));
        assert_eq!(text, "world");
        // Reversed endpoints (drag right-to-left) yield the same text.
        let text = selection_text(|i| lines.get(i).map(|v| v.as_slice()), (0, 10), (0, 6));
        assert_eq!(text, "world");
    }

    #[test_case]
    fn multi_line_extract_trims_unset_tails() {
        let lines = [row("first line", 16), row("second", 16), row("third", 16)];
        let text = selection_text(|i| lines.get(i).map(|v| v.as_slice()), (0, 6), (2, 2));
        assert_eq!(text, "line\nsecond\nthi");
    }

    #[test_case]
    fn extract_clamps_past_end() {
        let lines = [row("ab", 4)];
        // Columns beyond the line and rows beyond the buffer are clamped.
        let text = selection_text(|i| lines.get(i).map(|v| v.as_slice()), (0, 0), (5, 99));
        assert_eq!(text, "ab");
    }

    /// Differential selection paint only touches cells whose membership flips
    /// (used by the chat drag path to avoid a full-pane clear flash).
    #[test_case]
    fn sel_membership_differs_only_on_delta() {
        let old = normalize((0, 0), (0, 3)); // cols 0..=3
        let new = normalize((0, 2), (0, 5)); // cols 2..=5
        // 0,1 leave; 2,3 stay; 4,5 enter.
        assert!(contains(old, 0, 0) && !contains(new, 0, 0));
        assert!(contains(old, 0, 1) && !contains(new, 0, 1));
        assert!(contains(old, 0, 2) && contains(new, 0, 2));
        assert!(contains(old, 0, 3) && contains(new, 0, 3));
        assert!(!contains(old, 0, 4) && contains(new, 0, 4));
        assert!(!contains(old, 0, 5) && contains(new, 0, 5));
    }

    fn cells(s: &str) -> Vec<Cell> {
        s.chars().map(|c| (c, FG)).collect()
    }

    #[test_case]
    fn reflow_expands_soft_wrapped_lines() {
        // Two full 8-col soft-wraps that form "hello world!!!!" (16 cells).
        // Expanding to 16 cols must join them into one row.
        let a = cells("hello wo"); // 8
        let b = cells("rld!!!!!"); // 8
        assert_eq!(a.len(), 8);
        assert_eq!(b.len(), 8);
        let lines = [a, b];
        let out = reflow_lines(&lines, 8, 16, ('\0', FG));
        assert_eq!(out.len(), 1, "soft wraps join into one line: {out:?}");
        let text: String = out[0].iter().map(|&(c, _)| if c == '\0' { ' ' } else { c }).collect();
        assert!(text.starts_with("hello world!!!!!"), "got {text:?}");
    }

    #[test_case]
    fn reflow_shrinks_long_logical_line() {
        let line = cells("abcdefghij"); // 10 cells, hard short for cols=10? full width
        // Treat as one full-width line of 10, reflow to 4 → 3 rows.
        let out = reflow_lines(&[line], 10, 4, ('\0', FG));
        assert_eq!(out.len(), 3);
        let t0: String = out[0].iter().take(4).map(|&(c, _)| c).collect();
        let t1: String = out[1].iter().take(4).map(|&(c, _)| c).collect();
        let t2: String = trim_cells(&out[2]).iter().map(|&(c, _)| c).collect();
        assert_eq!(t0, "abcd");
        assert_eq!(t1, "efgh");
        assert_eq!(t2, "ij");
    }

    /// The source mapping must line up with the reflow it came from, in both directions.
    ///
    /// It exists so line-indexed metadata (the chat pane's prompt/tool-call bands and its
    /// folds) survives a re-wrap. If it drifts, a band lands on the wrong line and tints
    /// text that is not a tool call — a wrong answer that looks like a rendering glitch.
    #[test_case]
    fn a_reflow_reports_which_source_rows_each_output_row_came_from() {
        // Joining: two full 8-col rows become one 16-col row covering sources 0..=1.
        let (out, map) = reflow_lines_mapped(&[cells("hello wo"), cells("rld!!!!!")], 8, 16, ('\0', FG));
        assert_eq!(out.len(), map.len(), "one mapping entry per output row");
        assert_eq!(out.len(), 1);
        assert_eq!(map[0], (0, 1), "the joined row came from both source rows");

        // Splitting: one 10-col row becomes three 4-col rows, all from source 0.
        let (out, map) = reflow_lines_mapped(&[cells("abcdefghij")], 10, 4, ('\0', FG));
        assert_eq!(out.len(), 3);
        assert_eq!(map, alloc::vec![(0, 0), (0, 0), (0, 0)]);

        // Hard newlines: each short line maps to exactly itself.
        let (out, map) = reflow_lines_mapped(&[cells("hi"), cells("yo"), cells("!")], 40, 80, ('\0', FG));
        assert_eq!(out.len(), 3);
        assert_eq!(map, alloc::vec![(0, 0), (1, 1), (2, 2)]);

        // Mixed: a wrapped pair, then a short line. The short line must not be absorbed
        // into the preceding logical line's range, or its band would spread onto it.
        let (out, map) =
            reflow_lines_mapped(&[cells("12345678"), cells("90"), cells("tail")], 8, 8, ('\0', FG));
        assert_eq!(out.len(), map.len());
        let last = map.last().copied().unwrap();
        assert_eq!(last, (2, 2), "the trailing short line stays its own logical line");

        // And the delegating wrapper must agree with the mapped version exactly — one
        // implementation, so a future edit cannot make the map describe a different reflow.
        for (lines, oc, nc) in [
            (alloc::vec![cells("hello wo"), cells("rld!!!!!")], 8usize, 16usize),
            (alloc::vec![cells("abcdefghij")], 10, 4),
            (alloc::vec![cells("hi"), cells("yo")], 40, 80),
        ] {
            let plain = reflow_lines(&lines, oc, nc, ('\0', FG));
            let (mapped, m) = reflow_lines_mapped(&lines, oc, nc, ('\0', FG));
            assert_eq!(plain, mapped);
            assert_eq!(m.len(), plain.len());
        }
    }

    #[test_case]
    fn reflow_preserves_hard_newlines() {
        // Two short lines → stay two lines even when expanding.
        let lines = [cells("hi"), cells("yo")];
        let out = reflow_lines(&lines, 40, 80, ('\0', FG));
        assert_eq!(out.len(), 2);
        assert_eq!(trim_cells(&out[0]).iter().map(|&(c, _)| c).collect::<String>(), "hi");
        assert_eq!(trim_cells(&out[1]).iter().map(|&(c, _)| c).collect::<String>(), "yo");
    }

    #[test_case]
    fn ellipsize_and_fit_width() {
        assert_eq!(ellipsize("hello", 10), "hello");
        assert_eq!(ellipsize("hello world", 8), "hello ..");
        assert_eq!(ellipsize("ab", 1), "a");
        assert_eq!(ellipsize("ab", 0), "");
        assert_eq!(fit_width("hi", 5), "hi   ");
        assert_eq!(fit_width("hello world", 6).chars().count(), 6);
        assert!(fit_width("hello world", 6).ends_with(".."));
        // Path-style: keep the trailing basename.
        let e = ellipsize_end("@/agent/9001/SOUL.md", 14);
        assert_eq!(e.chars().count(), 14);
        assert!(e.ends_with("SOUL.md"), "got {e}");
        assert!(e.starts_with(".."), "got {e}");
    }

    /// Multi-line selection: middle rows are fully selected; endpoints clamp.
    #[test_case]
    fn multi_line_contains_and_clear() {
        let r = normalize((1, 2), (3, 1));
        assert!(!contains(r, 0, 9));
        assert!(!contains(r, 1, 1));
        assert!(contains(r, 1, 2));
        assert!(contains(r, 2, 0)); // full middle line
        assert!(contains(r, 2, 99));
        assert!(contains(r, 3, 1));
        assert!(!contains(r, 3, 2));
        assert!(!contains(r, 4, 0));
        // Empty selection (same cell) is still a single-cell range.
        let one = normalize((5, 5), (5, 5));
        assert!(contains(one, 5, 5));
        assert!(!contains(one, 5, 4));
        assert!(!contains(one, 5, 6));
    }
}
