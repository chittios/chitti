//! **Vim motions and text objects** — the pure half of the editor's normal mode.
//!
//! Vim's normal mode is not a table of keys, it is a small grammar:
//!
//! ```text
//! [count] [register] operator [count] motion | text-object
//! ```
//!
//! Everything expressive about it — `d2w`, `c3aw`, `y%`, `>ip` — falls out of
//! composing an *operator* with a *region*, and the region is the part with all
//! the fiddly rules. So it lives here, as pure functions over `&[String]`, and
//! [`crate::editor`] keeps only the state machine and the screen.
//!
//! That split is also what makes any of this testable: the editor owns a pane
//! and a keyboard, but a motion is just `(lines, cursor, count) -> position`.
//! Every rule below is pinned by a case in the test module, because motion bugs
//! are silent — `dw` deleting one character too many looks like a typo, not a
//! defect, and you only notice after it has eaten something you wanted.
//!
//! ## The two rules that cause most of the bugs
//!
//! **Exclusive vs inclusive.** `dw` deletes up to but *not including* the
//! character the motion lands on; `dx` where x is `e` or `f<char>` includes it.
//! Getting this wrong is an off-by-one in every operator at once, so the kind
//! travels *with* the span ([`Kind`]) rather than being decided by the caller.
//!
//! **A "word" is two different things.** `w` stops at the boundary between a run
//! of word characters (alphanumeric plus `_`) and a run of punctuation — so in
//! `foo.bar` it stops three times, not once. `W` is the simpler whitespace-
//! delimited one. Implementing only the `W` rule and calling it `w` is the
//! single most common way a Vim clone feels subtly wrong.
//!
//! Text is ASCII here, matching [`crate::editor`]: byte == char == column. When
//! that changes, the arithmetic in [`crate::textfit`] is what these functions
//! should route through.

use alloc::string::String;

/// A position in the buffer: `row` indexes `lines`, `col` is a byte offset into
/// that line. `col` may equal the line length (one past the end), which is where
/// insert mode legitimately sits.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Pos {
    pub row: usize,
    pub col: usize,
}

impl Pos {
    pub fn new(row: usize, col: usize) -> Self {
        Self { row, col }
    }
}

/// How an operator should treat the region a motion produced.
///
/// This is a property of the *motion*, not of the operator — `d` deletes an
/// exclusive region after `w` and an inclusive one after `e`, and it is the
/// motion that knows which.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// Up to, but not including, `end`. (`w`, `b`, `0`, `$` as a motion, `%`.)
    Exclusive,
    /// Up to and including `end`. (`e`, `f`, `t`, text objects.)
    Inclusive,
    /// Whole lines from `start.row` to `end.row`. (`j`, `k`, `G`, `dd`, `ip`.)
    Line,
}

/// A region of the buffer, normalised so `start <= end`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Span {
    pub start: Pos,
    pub end: Pos,
    pub kind: Kind,
}

impl Span {
    /// Build a span from two positions in any order, normalising them.
    pub fn new(a: Pos, b: Pos, kind: Kind) -> Self {
        let (start, end) = if (a.row, a.col) <= (b.row, b.col) { (a, b) } else { (b, a) };
        Self { start, end, kind }
    }
}

/// Vim's three character classes. `w`/`b`/`e` stop wherever this value changes
/// (ignoring runs of `Blank`), which is why `foo.bar` is three words.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Class {
    Blank,
    Word,
    Punct,
}

fn class_of(b: u8, big: bool) -> Class {
    match b {
        b' ' | b'\t' => Class::Blank,
        // A WORD (`W`/`B`/`E`) is delimited by whitespace alone, so everything
        // that is not blank counts as the same class.
        _ if big => Class::Word,
        b'0'..=b'9' | b'a'..=b'z' | b'A'..=b'Z' | b'_' => Class::Word,
        _ => Class::Punct,
    }
}

fn line(lines: &[String], row: usize) -> &[u8] {
    lines.get(row).map(|l| l.as_bytes()).unwrap_or(&[])
}

fn line_len(lines: &[String], row: usize) -> usize {
    line(lines, row).len()
}

fn last_row(lines: &[String]) -> usize {
    lines.len().saturating_sub(1)
}

/// The character at `p`, or `None` past the end of a line.
fn at(lines: &[String], p: Pos) -> Option<u8> {
    line(lines, p.row).get(p.col).copied()
}

/// Step one byte forward, wrapping to the next line. `None` at end of buffer.
///
/// The wrap lands on the newline *position* (one past the last byte) rather than
/// skipping straight to column 0, because `w` from the last word of a line must
/// be able to stop at the line end — that empty position is a word boundary.
fn next(lines: &[String], p: Pos) -> Option<Pos> {
    if p.col < line_len(lines, p.row) {
        Some(Pos::new(p.row, p.col + 1))
    } else if p.row < last_row(lines) {
        Some(Pos::new(p.row + 1, 0))
    } else {
        None
    }
}

fn prev(lines: &[String], p: Pos) -> Option<Pos> {
    if p.col > 0 {
        Some(Pos::new(p.row, p.col - 1))
    } else if p.row > 0 {
        Some(Pos::new(p.row - 1, line_len(lines, p.row - 1)))
    } else {
        None
    }
}

/// `w` / `W`: forward to the start of the next word, `count` times.
///
/// An empty line counts as a word in its own right — Vim stops on it — which is
/// why the blank-skipping loop below breaks on a zero-length line instead of
/// running past it.
pub fn word_fwd(lines: &[String], from: Pos, count: usize, big: bool) -> Pos {
    let mut p = from;
    for _ in 0..count.max(1) {
        let start_class = at(lines, p).map(|b| class_of(b, big));
        // Leave the current run.
        if let Some(c) = start_class {
            if c != Class::Blank {
                while let Some(n) = next(lines, p) {
                    match at(lines, n).map(|b| class_of(b, big)) {
                        Some(k) if k == c => p = n,
                        _ => {
                            p = n;
                            break;
                        }
                    }
                }
            }
        } else if let Some(n) = next(lines, p) {
            p = n;
        }
        // Skip blanks to land on the next word's first character. An empty line
        // is a stopping point, not blank space to skip.
        loop {
            if line_len(lines, p.row) == 0 {
                break;
            }
            match at(lines, p).map(|b| class_of(b, big)) {
                Some(Class::Blank) | None => match next(lines, p) {
                    Some(n) => p = n,
                    None => break,
                },
                _ => break,
            }
        }
    }
    p
}

/// `b` / `B`: back to the start of the current or previous word.
pub fn word_back(lines: &[String], from: Pos, count: usize, big: bool) -> Pos {
    let mut p = from;
    for _ in 0..count.max(1) {
        // Step back one, then skip blanks backwards.
        p = match prev(lines, p) {
            Some(q) => q,
            None => return p,
        };
        loop {
            if line_len(lines, p.row) == 0 {
                break;
            }
            match at(lines, p).map(|b| class_of(b, big)) {
                Some(Class::Blank) | None => match prev(lines, p) {
                    Some(q) => p = q,
                    None => break,
                },
                _ => break,
            }
        }
        // Walk to the start of this run.
        if let Some(c) = at(lines, p).map(|b| class_of(b, big)) {
            while let Some(q) = prev(lines, p) {
                if q.row != p.row {
                    break;
                }
                match at(lines, q).map(|b| class_of(b, big)) {
                    Some(k) if k == c => p = q,
                    _ => break,
                }
            }
        }
    }
    p
}

/// `e` / `E`: forward to the **last** character of the current or next word.
/// Inclusive, so `de` takes the final character with it.
pub fn word_end(lines: &[String], from: Pos, count: usize, big: bool) -> Pos {
    let mut p = from;
    for _ in 0..count.max(1) {
        // Always advance at least one, or `e` on a word's last char never moves.
        p = match next(lines, p) {
            Some(q) => q,
            None => return p,
        };
        // Skip blanks forward.
        while matches!(at(lines, p).map(|b| class_of(b, big)), Some(Class::Blank) | None) {
            match next(lines, p) {
                Some(q) => p = q,
                None => return p,
            }
        }
        // Run to the end of this class.
        if let Some(c) = at(lines, p).map(|b| class_of(b, big)) {
            while let Some(q) = next(lines, p) {
                if q.row != p.row {
                    break;
                }
                match at(lines, q).map(|b| class_of(b, big)) {
                    Some(k) if k == c => p = q,
                    _ => break,
                }
            }
        }
    }
    p
}

/// `^`: first non-blank column of the row (the whole-line fallback is column 0).
pub fn first_non_blank(lines: &[String], row: usize) -> usize {
    let l = line(lines, row);
    l.iter().position(|&b| b != b' ' && b != b'\t').unwrap_or(0)
}

/// Which of `f`/`F`/`t`/`T` was typed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Find {
    /// `f` — forward, land **on** the match.
    Forward,
    /// `F` — backward, land on the match.
    Backward,
    /// `t` — forward, land just **before** the match.
    Till,
    /// `T` — backward, land just after the match.
    TillBack,
}

/// `f`/`F`/`t`/`T`: search within the current line only — Vim never crosses a
/// line boundary for these, so a miss is a no-op rather than a jump.
///
/// `t` with a count repeats the *search*, not the offset: `3tx` lands before the
/// third `x`. Implementing it as "find then step back" once per iteration would
/// stall, because the second search starts on the character before the match and
/// finds nothing new — hence the search runs to completion first, then the
/// one-character offset is applied.
pub fn find_char(lines: &[String], from: Pos, target: u8, count: usize, kind: Find) -> Option<usize> {
    let l = line(lines, from.row);
    let mut col = from.col;
    let fwd = matches!(kind, Find::Forward | Find::Till);
    for _ in 0..count.max(1) {
        if fwd {
            let start = col + 1;
            let hit = l.get(start..)?.iter().position(|&b| b == target)? + start;
            col = hit;
        } else {
            let hit = l.get(..col)?.iter().rposition(|&b| b == target)?;
            col = hit;
        }
    }
    Some(match kind {
        Find::Forward | Find::Backward => col,
        Find::Till => col.checked_sub(1)?,
        Find::TillBack => col + 1,
    })
}

/// `%`: jump to the bracket matching the one at (or next on) the cursor line.
///
/// Vim searches *forward from the cursor to the end of the line* for a bracket
/// before giving up — `%` in leading whitespace still works — and then matches
/// with nesting across lines.
pub fn match_pair(lines: &[String], from: Pos) -> Option<Pos> {
    const PAIRS: [(u8, u8); 3] = [(b'(', b')'), (b'[', b']'), (b'{', b'}')];
    let l = line(lines, from.row);
    let (col, open, close, forward) = l
        .iter()
        .enumerate()
        .skip(from.col)
        .find_map(|(i, &b)| {
            PAIRS.iter().find_map(|&(o, c)| {
                if b == o {
                    Some((i, o, c, true))
                } else if b == c {
                    Some((i, o, c, false))
                } else {
                    None
                }
            })
        })?;

    let mut depth = 0i32;
    let mut p = Pos::new(from.row, col);
    loop {
        match at(lines, p) {
            Some(b) if b == open => depth += if forward { 1 } else { -1 },
            Some(b) if b == close => depth += if forward { -1 } else { 1 },
            _ => {}
        }
        if depth == 0 {
            return Some(p);
        }
        p = if forward { next(lines, p)? } else { prev(lines, p)? };
    }
}

/// `{` / `}`: move by paragraph, where a paragraph boundary is an empty line.
pub fn paragraph(lines: &[String], from: Pos, count: usize, forward: bool) -> Pos {
    let mut row = from.row;
    for _ in 0..count.max(1) {
        loop {
            if forward {
                if row >= last_row(lines) {
                    row = last_row(lines);
                    break;
                }
                row += 1;
            } else {
                if row == 0 {
                    break;
                }
                row -= 1;
            }
            if line_len(lines, row) == 0 {
                break;
            }
        }
    }
    Pos::new(row, 0)
}

/// The text objects this editor understands.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Object {
    Word { big: bool },
    /// A bracket pair; the byte is the opening delimiter.
    Bracket(u8),
    /// A quote pair; the byte is the quote character.
    Quote(u8),
    Paragraph,
}

/// `iw` / `aw` / `i(` / `a"` / `ip` … — resolve a text object around the cursor.
///
/// `around` is the `a` variant: it takes the delimiters (or, for a word, the
/// trailing whitespace) as well as the content. Returns `None` when the cursor
/// is not inside such an object, which the caller must treat as "the whole
/// command does nothing" — Vim beeps rather than guessing at a nearby one.
pub fn text_object(lines: &[String], cur: Pos, obj: Object, around: bool) -> Option<Span> {
    match obj {
        Object::Word { big } => {
            let l = line(lines, cur.row);
            if l.is_empty() {
                return None;
            }
            let col = cur.col.min(l.len() - 1);
            let c = class_of(l[col], big);
            let mut s = col;
            while s > 0 && class_of(l[s - 1], big) == c {
                s -= 1;
            }
            let mut e = col;
            while e + 1 < l.len() && class_of(l[e + 1], big) == c {
                e += 1;
            }
            if around {
                // `aw` takes trailing whitespace; if there is none, leading
                // instead — so `daw` on the last word of a line still removes
                // the space that separated it from the previous one.
                let mut e2 = e;
                while e2 + 1 < l.len() && class_of(l[e2 + 1], big) == Class::Blank {
                    e2 += 1;
                }
                if e2 == e {
                    while s > 0 && class_of(l[s - 1], big) == Class::Blank {
                        s -= 1;
                    }
                } else {
                    e = e2;
                }
            }
            Some(Span::new(Pos::new(cur.row, s), Pos::new(cur.row, e), Kind::Inclusive))
        }
        Object::Bracket(open) => {
            let close = match open {
                b'(' => b')',
                b'[' => b']',
                b'{' => b'}',
                b'<' => b'>',
                _ => return None,
            };
            let o = scan_unmatched(lines, cur, open, close, false)?;
            let c = scan_unmatched(lines, cur, open, close, true)?;
            if around {
                Some(Span::new(o, c, Kind::Inclusive))
            } else {
                // Empty pair: `i(` on `()` selects nothing, so refuse rather
                // than producing an inverted span the operator would misread.
                let s = next(lines, o)?;
                let e = prev(lines, c)?;
                if (s.row, s.col) > (e.row, e.col) {
                    return None;
                }
                Some(Span::new(s, e, Kind::Inclusive))
            }
        }
        Object::Quote(q) => {
            // Quotes have no nesting, so the pair is found by counting from the
            // start of the line: the cursor sits inside an odd-numbered run.
            let l = line(lines, cur.row);
            let cols: alloc::vec::Vec<usize> =
                l.iter().enumerate().filter(|&(_, &b)| b == q).map(|(i, _)| i).collect();
            let pair = cols.chunks(2).find(|c| c.len() == 2 && cur.col <= c[1])?;
            let (o, c) = (pair[0], pair[1]);
            if around {
                Some(Span::new(Pos::new(cur.row, o), Pos::new(cur.row, c), Kind::Inclusive))
            } else if c == o + 1 {
                None
            } else {
                Some(Span::new(Pos::new(cur.row, o + 1), Pos::new(cur.row, c - 1), Kind::Inclusive))
            }
        }
        Object::Paragraph => {
            let mut s = cur.row;
            let mut e = cur.row;
            while s > 0 && line_len(lines, s - 1) > 0 {
                s -= 1;
            }
            while e < last_row(lines) && line_len(lines, e + 1) > 0 {
                e += 1;
            }
            if around {
                while e < last_row(lines) && line_len(lines, e + 1) == 0 {
                    e += 1;
                }
            }
            Some(Span::new(Pos::new(s, 0), Pos::new(e, 0), Kind::Line))
        }
    }
}

/// Walk out to the nearest unmatched `open`/`close` around the cursor.
fn scan_unmatched(lines: &[String], cur: Pos, open: u8, close: u8, forward: bool) -> Option<Pos> {
    let mut depth = 0i32;
    let mut p = cur;
    loop {
        let b = at(lines, p);
        if b == Some(if forward { close } else { open }) {
            if depth == 0 {
                return Some(p);
            }
            depth -= 1;
        } else if b == Some(if forward { open } else { close }) && p != cur {
            depth += 1;
        }
        p = if forward { next(lines, p)? } else { prev(lines, p)? };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;
    use alloc::vec::Vec;

    fn buf(s: &[&str]) -> Vec<String> {
        s.iter().map(|l| l.to_string()).collect()
    }

    #[test_case]
    fn w_stops_at_punctuation_not_just_whitespace() {
        // The rule that separates a real `w` from a naive one: `foo.bar` is
        // three words, so `w` from column 0 lands on the dot, not on `bar`.
        let b = buf(&["foo.bar baz"]);
        let p = word_fwd(&b, Pos::new(0, 0), 1, false);
        assert_eq!(p, Pos::new(0, 3), "w must stop at the punctuation run");
        let p = word_fwd(&b, p, 1, false);
        assert_eq!(p, Pos::new(0, 4), "then at the start of `bar`");
    }

    #[test_case]
    fn big_w_is_whitespace_delimited() {
        let b = buf(&["foo.bar baz"]);
        assert_eq!(word_fwd(&b, Pos::new(0, 0), 1, true), Pos::new(0, 8), "W skips the whole token");
    }

    #[test_case]
    fn a_count_repeats_the_motion() {
        let b = buf(&["alpha beta gamma delta"]);
        assert_eq!(word_fwd(&b, Pos::new(0, 0), 3, false), Pos::new(0, 17));
    }

    #[test_case]
    fn word_motions_cross_lines() {
        let b = buf(&["one", "two"]);
        assert_eq!(word_fwd(&b, Pos::new(0, 0), 1, false), Pos::new(1, 0));
        assert_eq!(word_back(&b, Pos::new(1, 0), 1, false), Pos::new(0, 0));
    }

    #[test_case]
    fn an_empty_line_is_its_own_word() {
        // Vim stops on a blank line rather than skipping to the next text.
        let b = buf(&["one", "", "two"]);
        assert_eq!(word_fwd(&b, Pos::new(0, 0), 1, false), Pos::new(1, 0));
    }

    #[test_case]
    fn e_lands_on_the_last_character_not_past_it() {
        // `e` is inclusive; landing one past would make `de` eat the following
        // space, which is `dw`'s job.
        let b = buf(&["alpha beta"]);
        assert_eq!(word_end(&b, Pos::new(0, 0), 1, false), Pos::new(0, 4));
    }

    #[test_case]
    fn e_from_a_words_last_char_advances_to_the_next() {
        let b = buf(&["alpha beta"]);
        assert_eq!(word_end(&b, Pos::new(0, 4), 1, false), Pos::new(0, 9));
    }

    #[test_case]
    fn b_returns_to_the_start_of_the_current_word() {
        let b = buf(&["alpha beta"]);
        assert_eq!(word_back(&b, Pos::new(0, 8), 1, false), Pos::new(0, 6));
        assert_eq!(word_back(&b, Pos::new(0, 6), 1, false), Pos::new(0, 0));
    }

    #[test_case]
    fn first_non_blank_finds_the_indent_end() {
        let b = buf(&["    indented", "notindented", "        "]);
        assert_eq!(first_non_blank(&b, 0), 4);
        assert_eq!(first_non_blank(&b, 1), 0);
        assert_eq!(first_non_blank(&b, 2), 0, "an all-blank line falls back to column 0");
    }

    #[test_case]
    fn f_and_t_differ_by_one_and_t_repeats_the_search() {
        let b = buf(&["axbxcx"]);
        assert_eq!(find_char(&b, Pos::new(0, 0), b'x', 1, Find::Forward), Some(1));
        assert_eq!(find_char(&b, Pos::new(0, 0), b'x', 1, Find::Till), Some(0));
        // The trap: a naive `t` implementation stalls on a count because the
        // second search starts before the first match.
        assert_eq!(find_char(&b, Pos::new(0, 0), b'x', 3, Find::Till), Some(4));
        assert_eq!(find_char(&b, Pos::new(0, 0), b'x', 3, Find::Forward), Some(5));
    }

    #[test_case]
    fn find_never_leaves_the_line() {
        let b = buf(&["abc", "xyz"]);
        assert_eq!(find_char(&b, Pos::new(0, 0), b'z', 1, Find::Forward), None);
    }

    #[test_case]
    fn find_backward_searches_before_the_cursor() {
        let b = buf(&["axbxc"]);
        assert_eq!(find_char(&b, Pos::new(0, 4), b'x', 1, Find::Backward), Some(3));
        assert_eq!(find_char(&b, Pos::new(0, 4), b'x', 2, Find::Backward), Some(1));
    }

    #[test_case]
    fn percent_matches_nested_brackets_across_lines() {
        let b = buf(&["fn f() {", "    g(h());", "}"]);
        assert_eq!(match_pair(&b, Pos::new(0, 7)), Some(Pos::new(2, 0)), "brace to brace");
        assert_eq!(match_pair(&b, Pos::new(2, 0)), Some(Pos::new(0, 7)), "and back");
        // Nesting: the outer `(` of `g(h())` must skip the inner pair.
        assert_eq!(match_pair(&b, Pos::new(1, 5)), Some(Pos::new(1, 9)));
    }

    #[test_case]
    fn percent_scans_forward_to_the_first_bracket_on_the_line() {
        // `%` in the indent still works — it finds the bracket to its right.
        let b = buf(&["    (x)"]);
        assert_eq!(match_pair(&b, Pos::new(0, 0)), Some(Pos::new(0, 6)));
    }

    #[test_case]
    fn iw_and_aw_differ_by_the_trailing_space() {
        let b = buf(&["alpha beta gamma"]);
        let iw = text_object(&b, Pos::new(0, 7), Object::Word { big: false }, false).unwrap();
        assert_eq!((iw.start.col, iw.end.col), (6, 9));
        let aw = text_object(&b, Pos::new(0, 7), Object::Word { big: false }, true).unwrap();
        assert_eq!((aw.start.col, aw.end.col), (6, 10), "aw takes the trailing space");
    }

    #[test_case]
    fn aw_takes_the_leading_space_when_there_is_no_trailing_one() {
        let b = buf(&["alpha beta"]);
        let aw = text_object(&b, Pos::new(0, 7), Object::Word { big: false }, true).unwrap();
        assert_eq!((aw.start.col, aw.end.col), (5, 9));
    }

    #[test_case]
    fn bracket_objects_find_the_enclosing_pair() {
        let b = buf(&["call(a, b)"]);
        let inner = text_object(&b, Pos::new(0, 6), Object::Bracket(b'('), false).unwrap();
        assert_eq!((inner.start.col, inner.end.col), (5, 8));
        let outer = text_object(&b, Pos::new(0, 6), Object::Bracket(b'('), true).unwrap();
        assert_eq!((outer.start.col, outer.end.col), (4, 9));
    }

    #[test_case]
    fn an_empty_pair_has_no_inner_object() {
        // `i(` on `()` must fail rather than return an inverted span, which an
        // operator would read as a region running backwards through the buffer.
        let b = buf(&["f()"]);
        assert!(text_object(&b, Pos::new(0, 2), Object::Bracket(b'('), false).is_none());
        assert!(text_object(&b, Pos::new(0, 2), Object::Bracket(b'('), true).is_some());
    }

    #[test_case]
    fn bracket_objects_span_lines_and_respect_nesting() {
        let b = buf(&["{", "  { inner }", "}"]);
        let o = text_object(&b, Pos::new(1, 5), Object::Bracket(b'{'), true).unwrap();
        assert_eq!((o.start.row, o.start.col, o.end.row, o.end.col), (1, 2, 1, 10));
    }

    #[test_case]
    fn quote_objects_pair_from_the_line_start() {
        let b = buf(&["say \"hello\" now"]);
        let inner = text_object(&b, Pos::new(0, 6), Object::Quote(b'"'), false).unwrap();
        assert_eq!((inner.start.col, inner.end.col), (5, 9));
        let outer = text_object(&b, Pos::new(0, 6), Object::Quote(b'"'), true).unwrap();
        assert_eq!((outer.start.col, outer.end.col), (4, 10));
    }

    #[test_case]
    fn a_cursor_before_the_first_quote_still_finds_the_pair() {
        // Vim's `ci"` works from anywhere before the closing quote on the line.
        let b = buf(&["say \"hello\""]);
        let inner = text_object(&b, Pos::new(0, 0), Object::Quote(b'"'), false).unwrap();
        assert_eq!((inner.start.col, inner.end.col), (5, 9));
    }

    #[test_case]
    fn paragraph_object_is_linewise_and_stops_at_blanks() {
        let b = buf(&["a", "b", "", "c"]);
        let ip = text_object(&b, Pos::new(0, 0), Object::Paragraph, false).unwrap();
        assert_eq!((ip.start.row, ip.end.row, ip.kind), (0, 1, Kind::Line));
        let ap = text_object(&b, Pos::new(0, 0), Object::Paragraph, true).unwrap();
        assert_eq!(ap.end.row, 2, "ap takes the blank separator too");
    }

    #[test_case]
    fn paragraph_motion_moves_between_blank_lines() {
        let b = buf(&["a", "", "b", "", "c"]);
        assert_eq!(paragraph(&b, Pos::new(0, 0), 1, true).row, 1);
        assert_eq!(paragraph(&b, Pos::new(0, 0), 2, true).row, 3);
        assert_eq!(paragraph(&b, Pos::new(4, 0), 1, false).row, 3);
    }

    #[test_case]
    fn paragraph_motion_clamps_at_the_buffer_edges() {
        let b = buf(&["a", "b"]);
        assert_eq!(paragraph(&b, Pos::new(0, 0), 9, true).row, 1);
        assert_eq!(paragraph(&b, Pos::new(1, 0), 9, false).row, 0);
    }

    #[test_case]
    fn span_new_normalises_reversed_positions() {
        let s = Span::new(Pos::new(2, 0), Pos::new(1, 3), Kind::Exclusive);
        assert_eq!(s.start, Pos::new(1, 3));
        assert_eq!(s.end, Pos::new(2, 0));
    }

    #[test_case]
    fn motions_on_an_empty_buffer_do_not_panic() {
        // Every one of these indexes `lines`; an empty buffer is the case that
        // turns a missing bounds check into a kernel panic rather than a beep.
        let b: Vec<String> = vec![String::new()];
        assert_eq!(word_fwd(&b, Pos::new(0, 0), 1, false), Pos::new(0, 0));
        assert_eq!(word_back(&b, Pos::new(0, 0), 1, false), Pos::new(0, 0));
        assert_eq!(word_end(&b, Pos::new(0, 0), 1, false), Pos::new(0, 0));
        assert_eq!(match_pair(&b, Pos::new(0, 0)), None);
        assert!(text_object(&b, Pos::new(0, 0), Object::Word { big: false }, false).is_none());
    }
}
