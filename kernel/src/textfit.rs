//! Text fitting in **display columns** — the arithmetic every text surface in
//! the OS needs and that half of them were getting wrong.
//!
//! ## The bug class this exists to end
//!
//! `String` indices are byte offsets, terminal geometry is in columns, and for
//! ASCII the two are the same number. So the line editor, the composer and the
//! modal wrap all used `buf.len() - cur` as a column count, `chars().take(cols)`
//! as a truncation, and `s.as_bytes().chunks(cols)` as a line break. Every one of
//! those is correct only while every character is one byte wide, and the moment a
//! non-ASCII character reaches them the failures are:
//!
//! - **a panic**, where a byte range lands mid-character (`buf.drain(cur - n..cur)`
//!   on backspace, `&line[start..start + max_cols]` in the composer);
//! - **a caret in the wrong place**, where a byte count is emitted as an
//!   `ESC[nC` column move;
//! - **mojibake**, where a byte slice is reinterpreted as text.
//!
//! Everything here is pure and lives outside `framebuffer/` (which is
//! `#[cfg(not(test))]`, so a test written next to `wrap` would never even be
//! compiled — which is why `wrap` shipped mixing bytes and columns in the first
//! place). [`wrap`] and [`pad_trunc`] were moved here from
//! `framebuffer::text` and re-exported, so no call site changed.
//!
//! ## What `char_cols` does and does not claim
//!
//! It answers the East-Asian-Width question coarsely: zero for combining marks
//! and zero-width formatters, two for the Wide/Fullwidth ranges, one otherwise.
//! It is deliberately not the full Unicode table (200+ ranges); the ones left out
//! are rare and the cost of being wrong about them is cosmetic.
//!
//! **The pane grid is out of scope.** `framebuffer::Cell` is `(char, Rgb)` with
//! one cell per character, so a wide glyph there would need a lead cell plus a
//! continuation marker, and then `set_cell`, the scroll row copy, the selection's
//! absolute-index maths and `glyph_cell` all change — inside the module where no
//! test compiles. The visible consequence, stated plainly: typed CJK behaves
//! correctly in the composer (caret in the right place, backspace deletes one
//! glyph) but renders *narrow* once echoed into a pane, because
//! `font_ttf::build_ui_glyph` scales a non-ASCII glyph down into its cell rather
//! than letting it overflow. Cramped, not corrupt.

use alloc::string::String;
use alloc::vec::Vec;

/// Display columns for one `char`.
///
/// Zero for combining marks and the zero-width formatters `blit_glyph` already
/// short-circuits; two for East-Asian Wide/Fullwidth; one otherwise.
pub fn char_cols(c: char) -> usize {
    let u = c as u32;
    // Zero-width: combining diacritics, the bidi/joiner formatters, variation
    // selectors, and the combining half marks.
    if matches!(u,
        0x0300..=0x036F
        | 0x0483..=0x0489
        | 0x0591..=0x05BD
        | 0x0610..=0x061A
        | 0x064B..=0x065F
        | 0x0670
        | 0x06D6..=0x06DC
        | 0x0900..=0x0903
        | 0x093A
        | 0x093C
        | 0x0941..=0x0948
        | 0x094D
        | 0x0951..=0x0957
        | 0x200B..=0x200F
        | 0x2060..=0x206F
        | 0xFE00..=0xFE0F
        | 0xFE20..=0xFE2F
    ) {
        return 0;
    }
    // East-Asian Wide + Fullwidth, plus the emoji blocks that render double-wide
    // everywhere it matters.
    if matches!(u,
        0x1100..=0x115F        // Hangul Jamo initial consonants
        | 0x2E80..=0x303E      // CJK radicals, Kangxi, CJK symbols/punctuation
        | 0x3041..=0x33FF      // kana, bopomofo, Hangul compat, CJK compat
        | 0x3400..=0x4DBF      // CJK ext A
        | 0x4E00..=0x9FFF      // CJK unified
        | 0xA000..=0xA4CF      // Yi
        | 0xAC00..=0xD7A3      // Hangul syllables
        | 0xF900..=0xFAFF      // CJK compat ideographs
        | 0xFE30..=0xFE6F      // CJK compat forms, small form variants
        | 0xFF00..=0xFF60      // fullwidth forms
        | 0xFFE0..=0xFFE6      // fullwidth signs
        | 0x1F300..=0x1F64F    // misc symbols/pictographs, emoticons
        | 0x1F900..=0x1F9FF    // supplemental symbols/pictographs
        | 0x20000..=0x3FFFD    // CJK ext B and beyond
    ) {
        return 2;
    }
    1
}

/// Display columns occupied by `s`.
pub fn cols(s: &str) -> usize {
    s.chars().map(char_cols).sum()
}

/// The byte offset `n` characters before `cur`, clamped at 0.
///
/// The backspace walk. `cur - n` is what it replaces, and `cur - n` panics the
/// moment `n` bytes back from `cur` is inside a character — which the accelerated
/// backspace (`Accel::steps` returns 1/2/4/8) makes likely rather than
/// theoretical.
pub fn back_n_chars(s: &str, cur: usize, n: usize) -> usize {
    let cur = cur.min(s.len());
    let mut at = cur;
    for _ in 0..n {
        if at == 0 {
            break;
        }
        // Walk back to the previous char boundary.
        let mut k = at - 1;
        while k > 0 && !s.is_char_boundary(k) {
            k -= 1;
        }
        at = k;
    }
    at
}

/// The byte offset `n` characters *after* `cur`, clamped at `s.len()`.
pub fn forward_n_chars(s: &str, cur: usize, n: usize) -> usize {
    let mut at = cur.min(s.len());
    for _ in 0..n {
        if at >= s.len() {
            break;
        }
        let mut k = at + 1;
        while k < s.len() && !s.is_char_boundary(k) {
            k += 1;
        }
        at = k;
    }
    at
}

/// Truncate to `cols` display columns and pad with spaces to exactly that width.
///
/// Never emits a partial character, and never overshoots: a wide character that
/// would straddle the limit is dropped and a space takes its place, because half
/// a CJK glyph is worse than a gap.
pub fn pad_trunc(s: &str, target: usize) -> String {
    let mut out = String::new();
    let mut w = 0usize;
    for c in s.chars() {
        let cw = char_cols(c);
        if w + cw > target {
            break;
        }
        out.push(c);
        w += cw;
    }
    while w < target {
        out.push(' ');
        w += 1;
    }
    out
}

/// Truncate to `target` columns without padding.
pub fn trunc(s: &str, target: usize) -> String {
    let mut out = String::new();
    let mut w = 0usize;
    for c in s.chars() {
        let cw = char_cols(c);
        if w + cw > target {
            break;
        }
        out.push(c);
        w += cw;
    }
    out
}

/// Word-wrap `s` to `target` columns, breaking words that do not fit.
///
/// Moved here from `framebuffer::text::wrap`, which compared `line.len()` (bytes)
/// against `cols` and chunked long words with `as_bytes().chunks(cols)` — the
/// latter producing `from_utf8(chunk).unwrap_or("")`, i.e. silently **dropping**
/// any word whose chunk boundary fell inside a character.
pub fn wrap(s: &str, target: usize) -> Vec<String> {
    let target = target.max(1);
    let mut out = Vec::new();
    let mut line = String::new();
    let mut line_w = 0usize;
    for word in s.split_whitespace() {
        let ww = cols(word);
        if !line.is_empty() && line_w + 1 + ww > target {
            out.push(core::mem::take(&mut line));
            line_w = 0;
        }
        if ww <= target {
            if !line.is_empty() {
                line.push(' ');
                line_w += 1;
            }
            line.push_str(word);
            line_w += ww;
            continue;
        }
        // A word wider than the line: break it on **character** boundaries at
        // column width, so a multi-byte character is never split.
        for c in word.chars() {
            let cw = char_cols(c);
            if line_w + cw > target && !line.is_empty() {
                out.push(core::mem::take(&mut line));
                line_w = 0;
            }
            line.push(c);
            line_w += cw;
        }
    }
    if !line.is_empty() {
        out.push(line);
    }
    if out.is_empty() {
        out.push(String::new());
    }
    out
}

/// The visible window of a line that is wider than `max_cols`, plus the caret's
/// column within it.
///
/// Returns a **byte range guaranteed to sit on character boundaries** — the whole
/// point, since the composer's previous `&line[start..start + max_cols]` panics
/// on a non-boundary slice and misplaces the caret on any non-ASCII line.
///
/// The window is chosen so the caret is always visible, scrolling by whole
/// characters and keeping the caret off the very last column where possible (so
/// the next keystroke does not immediately re-scroll).
pub fn visible_window(
    line: &str,
    cur: usize,
    max_cols: usize,
) -> (core::ops::Range<usize>, usize) {
    let max_cols = max_cols.max(1);
    let cur = clamp_boundary(line, cur);
    if cols(line) <= max_cols {
        return (0..line.len(), cols(&line[..cur]));
    }
    // Walk back from the caret until the text between `start` and `cur` fills the
    // window, leaving one column of headroom so the caret is not flush right.
    let want = max_cols.saturating_sub(1).max(1);
    let mut start = cur;
    let mut w = 0usize;
    for (i, c) in line[..cur].char_indices().rev() {
        let cw = char_cols(c);
        if w + cw > want {
            break;
        }
        w += cw;
        start = i;
    }
    // Then extend the end as far as the window allows.
    let mut end = cur;
    let mut w2 = w;
    for (i, c) in line[cur..].char_indices() {
        let cw = char_cols(c);
        if w2 + cw > max_cols {
            break;
        }
        w2 += cw;
        end = cur + i + c.len_utf8();
    }
    (start..end, cols(&line[start..cur]))
}

/// Nudge a byte offset onto the nearest character boundary at or below it.
///
/// A caller passing a mid-character offset has a bug, but clamping is the right
/// response *here*: this is called from painters, and panicking mid-repaint
/// leaves the screen half-drawn with no way to read the message.
pub fn clamp_boundary(s: &str, at: usize) -> usize {
    let mut at = at.min(s.len());
    while at > 0 && !s.is_char_boundary(at) {
        at -= 1;
    }
    at
}

/// What the composer should draw and where the caret goes, given a line, its
/// byte cursor, and an IME pre-edit inserted at that cursor.
///
/// Pure so the pre-edit geometry is tested; the composer, which cannot be, only
/// paints the result.
pub fn preedit_view(line: &str, cur: usize, preedit: &str) -> (String, usize) {
    let cur = clamp_boundary(line, cur);
    let mut out = String::with_capacity(line.len() + preedit.len());
    out.push_str(&line[..cur]);
    out.push_str(preedit);
    out.push_str(&line[cur..]);
    // The caret sits at the **end** of the pre-edit: that is where the next
    // keystroke composes, and putting it before the run makes the whole thing
    // look like text you have already committed and moved past.
    (out, cols(&line[..cur]) + cols(preedit))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn ascii_columns_equal_byte_and_char_counts() {
        for s in ["", "a", "/help 42", "----"] {
            assert_eq!(cols(s), s.len());
            assert_eq!(cols(s), s.chars().count());
        }
    }

    #[test_case]
    fn wide_is_two_combining_is_zero_and_latin_accents_are_one() {
        assert_eq!(char_cols('a'), 1);
        assert_eq!(char_cols('é'), 1, "a precomposed accent is one column");
        assert_eq!(char_cols('日'), 2);
        assert_eq!(char_cols('本'), 2);
        assert_eq!(char_cols('あ'), 2, "hiragana");
        assert_eq!(char_cols('ア'), 2, "katakana");
        assert_eq!(char_cols('한'), 2, "hangul syllable");
        assert_eq!(char_cols('\u{1F600}'), 2, "emoji");
        assert_eq!(char_cols('\u{0301}'), 0, "combining acute");
        assert_eq!(char_cols('\u{200D}'), 0, "zero-width joiner");
        assert_eq!(char_cols('\u{FE0F}'), 0, "variation selector");
        // `cols` composes: 'e' + combining acute is one column, not two.
        assert_eq!(cols("e\u{0301}"), 1);
        assert_eq!(cols("日本"), 4);
        // "héllo" is 5 characters and 6 *bytes* — which is the entire reason this
        // module exists, so it is worth spelling both out.
        assert_eq!("héllo".len(), 6);
        assert_eq!(cols("héllo"), 5);
        assert_eq!(cols("héllo 日本"), 5 + 1 + 4);
    }

    #[test_case]
    fn back_n_chars_never_lands_mid_character() {
        // This is the backspace walk, and the case that panics today: `cur - n`
        // with n from the accelerating repeat.
        let s = "aé日\u{1F600}b";
        let end = s.len();
        for n in 0..=8usize {
            let at = back_n_chars(s, end, n);
            assert!(s.is_char_boundary(at), "back {n} from {end} gave {at}, not a boundary");
        }
        // One at a time, walking the whole string.
        assert_eq!(back_n_chars(s, end, 1), end - 1); // 'b'
        assert_eq!(back_n_chars(s, end, 2), end - 1 - 4); // the emoji
        assert_eq!(back_n_chars(s, end, 3), end - 1 - 4 - 3); // '日'
        assert_eq!(back_n_chars(s, end, 4), 1); // 'é'
        assert_eq!(back_n_chars(s, end, 5), 0);
        assert_eq!(back_n_chars(s, end, 99), 0, "clamps rather than underflowing");
        assert_eq!(back_n_chars(s, 0, 1), 0);
    }

    #[test_case]
    fn forward_n_chars_is_the_inverse_walk() {
        let s = "aé日\u{1F600}b";
        let mut at = 0;
        let mut seen = 0;
        while at < s.len() {
            at = forward_n_chars(s, at, 1);
            assert!(s.is_char_boundary(at));
            seen += 1;
        }
        assert_eq!(seen, s.chars().count());
        assert_eq!(forward_n_chars(s, 0, 99), s.len(), "clamps at the end");
    }

    #[test_case]
    fn pad_trunc_pads_and_truncates_in_columns() {
        assert_eq!(pad_trunc("ab", 5), "ab   ");
        assert_eq!(pad_trunc("abcdef", 3), "abc");
        assert_eq!(cols(&pad_trunc("日本語", 4)), 4);
        assert_eq!(pad_trunc("日本語", 4), "日本");
        // A wide char that would straddle the limit is dropped and padded, not
        // half-drawn.
        assert_eq!(pad_trunc("日本", 3), "日 ");
        assert_eq!(cols(&pad_trunc("日本", 3)), 3);
        // Every result is exactly `target` columns wide.
        for s in ["", "a", "日", "héllo 日本 ✓", "\u{1F600}\u{1F600}"] {
            for t in 0..12usize {
                assert_eq!(cols(&pad_trunc(s, t)), t, "pad_trunc({s:?}, {t}) width");
            }
        }
    }

    #[test_case]
    fn wrap_counts_columns_and_never_drops_a_word() {
        let out = wrap("the quick brown fox", 10);
        for l in &out {
            assert!(cols(l) <= 10, "line over width: {l:?}");
        }
        assert_eq!(out.join(" "), "the quick brown fox", "no word may be lost");
        // A word wider than the line is broken on char boundaries. The old
        // implementation chunked bytes and `unwrap_or("")`-ed the invalid pieces,
        // which silently deleted the word.
        let out = wrap("日本語日本語日本語", 5);
        let joined: alloc::string::String = out.concat();
        assert_eq!(joined, "日本語日本語日本語", "a long CJK word must survive wrapping");
        for l in &out {
            assert!(cols(l) <= 5, "{l:?}");
        }
        // Empty input still yields one (empty) line, as the old one did.
        assert_eq!(wrap("", 10), alloc::vec![alloc::string::String::new()]);
        // Degenerate width does not hang or panic.
        let out = wrap("abc def", 0);
        assert!(!out.is_empty());
    }

    #[test_case]
    fn visible_window_always_returns_char_boundaries() {
        // The test that would have caught the composer panic: every cursor
        // position of a mixed-width line, at several widths.
        let line = "aé日\u{1F600}b héllo 日本語 ✓ tail";
        for max in [1usize, 2, 3, 5, 8, 13, 40] {
            let mut cur = 0;
            loop {
                let (r, caret) = visible_window(line, cur, max);
                assert!(line.is_char_boundary(r.start), "start {} at cur {cur}", r.start);
                assert!(line.is_char_boundary(r.end), "end {} at cur {cur}", r.end);
                assert!(r.start <= r.end, "inverted range at cur {cur}");
                assert!(cols(&line[r.clone()]) <= max, "window wider than {max} at cur {cur}");
                assert!(caret <= max, "caret {caret} past {max} at cur {cur}");
                if cur >= line.len() {
                    break;
                }
                cur = forward_n_chars(line, cur, 1);
            }
        }
    }

    #[test_case]
    fn a_line_that_fits_is_shown_whole_with_the_caret_at_its_column() {
        let line = "héllo";
        let (r, caret) = visible_window(line, line.len(), 40);
        assert_eq!(r, 0..line.len());
        assert_eq!(caret, 5, "five columns, six bytes — this is the whole point");
    }

    #[test_case]
    fn visible_window_keeps_the_caret_in_view_when_the_line_is_long() {
        let line: alloc::string::String = core::iter::repeat('x').take(200).collect();
        // Caret at the end: the window must contain it.
        let (r, caret) = visible_window(&line, line.len(), 20);
        assert!(r.end == line.len() || caret <= 20);
        assert!(caret > 0, "the caret must not be pinned to column 0 at the end of a long line");
        // Caret at the start: the window starts there.
        let (r, caret) = visible_window(&line, 0, 20);
        assert_eq!(r.start, 0);
        assert_eq!(caret, 0);
    }

    #[test_case]
    fn a_mid_character_offset_is_clamped_rather_than_panicking() {
        let s = "日本";
        assert_eq!(clamp_boundary(s, 1), 0);
        assert_eq!(clamp_boundary(s, 2), 0);
        assert_eq!(clamp_boundary(s, 3), 3);
        assert_eq!(clamp_boundary(s, 999), s.len());
        // And the window/preedit paths survive one.
        let (r, _) = visible_window(s, 1, 10);
        assert!(s.is_char_boundary(r.start) && s.is_char_boundary(r.end));
        let (t, _) = preedit_view(s, 1, "か");
        assert!(t.contains('か'));
    }

    #[test_case]
    fn preedit_is_inserted_at_the_cursor_with_the_caret_after_it() {
        let (text, caret) = preedit_view("abcd", 2, "きゃ");
        assert_eq!(text, "abきゃcd");
        // 'a','b' = 2 columns, きゃ = 4 → caret at 6.
        assert_eq!(caret, 6);
        // Empty pre-edit is the identity.
        let (text, caret) = preedit_view("abcd", 2, "");
        assert_eq!(text, "abcd");
        assert_eq!(caret, 2);
        // At the end of the line.
        let (text, caret) = preedit_view("ab", 2, "ん");
        assert_eq!(text, "abん");
        assert_eq!(caret, 4);
    }

    #[test_case]
    fn trunc_does_not_pad() {
        assert_eq!(trunc("abcdef", 3), "abc");
        assert_eq!(trunc("ab", 5), "ab");
        assert_eq!(trunc("日本", 3), "日");
    }
}
