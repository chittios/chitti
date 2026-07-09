//! Pure text-selection math for the chat pane's mouse copy (the framebuffer
//! calls this; keeping it out of `framebuffer.rs` makes it unit-testable —
//! the compositor is `cfg(not(test))`). Coordinates are `(line, col)` where
//! `line` is an **absolute** index over a pane's scrollback + live grid
//! (0 = oldest scrollback line) and `col` is a cell column; both endpoints
//! are inclusive, like the editor's Visual selection.

use alloc::string::String;

/// One character cell as the compositor stores it: `(byte, fg colour)`.
pub type Cell = (u8, (u8, u8, u8));

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
        for &(byte, _) in &cells[lo..hi] {
            line.push(if (0x20..=0x7e).contains(&byte) { byte as char } else { ' ' });
        }
        let trimmed = line.trim_end();
        if r > r1 {
            out.push('\n');
        }
        out.push_str(trimmed);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    const FG: (u8, u8, u8) = (250, 249, 245);

    fn row(s: &str, cols: usize) -> Vec<Cell> {
        let mut v: Vec<Cell> = s.bytes().map(|b| (b, FG)).collect();
        v.resize(cols, (0, FG)); // unset tail cells, like a real grid line
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
