//! **Undo, redo, and registers** — the editor's memory.
//!
//! Two small pure stores that [`crate::editor`] drives. Both are here rather
//! than inline for the usual reason: they are the parts with rules, and rules
//! need tests.
//!
//! ## Undo groups keystrokes, not characters
//!
//! The mistake that makes a clone unusable is snapshotting per keystroke, so
//! `u` after typing a word removes one letter and you press it forty times.
//! Vim's unit is the *change*: everything between entering and leaving insert
//! mode is one undo step, as is a whole `dw`. So the editor calls
//! [`UndoStack::checkpoint`] **before** a change begins and nothing during it,
//! and [`UndoStack::undo`] restores the buffer as it was at that point.
//!
//! Snapshots are whole-buffer clones. That is O(file) per change and would be
//! wrong for a large file, but this editor's files are configs and notes, and a
//! correct simple thing beats a subtle diff engine — the alternative is storing
//! deltas and reconstructing, which is where undo implementations grow their
//! own bugs. If it ever needs to scale, the seam is here and the tests below
//! pin the behaviour a delta version would have to keep.
//!
//! ## Registers carry a flavour, not just text
//!
//! `dd` then `p` puts the line *below* the current one; `dw` then `p` puts the
//! text *after the cursor*. Same key, different behaviour, and the difference
//! comes from what was yanked — so a register stores [`RegKind`] alongside its
//! content. Dropping that distinction is why a naive `p` pastes a deleted line
//! into the middle of another one.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

/// A buffer snapshot plus where the cursor was, so undo restores the view too —
/// jumping to the top of the file on every `u` is disorienting and wrong.
#[derive(Clone)]
pub struct Snapshot {
    pub lines: Vec<String>,
    pub cy: usize,
    pub cx: usize,
}

/// Bounded undo history. The cap exists because this is a kernel with a
/// first-fit allocator: unbounded whole-buffer snapshots on a long editing
/// session are a memory leak with a friendly name.
pub const MAX_UNDO: usize = 256;

#[derive(Default)]
pub struct UndoStack {
    past: Vec<Snapshot>,
    future: Vec<Snapshot>,
}

impl UndoStack {
    pub fn new() -> Self {
        Self { past: Vec::new(), future: Vec::new() }
    }

    /// Record the state *before* a change. Call once per change, not per key.
    ///
    /// Any new change invalidates the redo branch — Vim keeps a tree, this keeps
    /// a line, and the difference only shows if you undo, edit, then expect the
    /// abandoned branch back. Documented rather than silently surprising.
    pub fn checkpoint(&mut self, lines: &[String], cy: usize, cx: usize) {
        self.future.clear();
        self.past.push(Snapshot { lines: lines.to_vec(), cy, cx });
        if self.past.len() > MAX_UNDO {
            self.past.remove(0);
        }
    }

    /// Undo one change, given the current state to push onto the redo side.
    pub fn undo(&mut self, current: &[String], cy: usize, cx: usize) -> Option<Snapshot> {
        let prev = self.past.pop()?;
        self.future.push(Snapshot { lines: current.to_vec(), cy, cx });
        Some(prev)
    }

    pub fn redo(&mut self, current: &[String], cy: usize, cx: usize) -> Option<Snapshot> {
        let next = self.future.pop()?;
        self.past.push(Snapshot { lines: current.to_vec(), cy, cx });
        Some(next)
    }

    pub fn can_undo(&self) -> bool {
        !self.past.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.future.is_empty()
    }
}

/// Whether a register holds whole lines or a run of characters.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RegKind {
    Char,
    Line,
}

#[derive(Clone)]
pub struct Reg {
    pub text: Vec<String>,
    pub kind: RegKind,
}

/// The register file: unnamed `"`, named `a`-`z`, and the numbered ring `0`-`9`.
#[derive(Default)]
pub struct Registers {
    map: BTreeMap<char, Reg>,
}

impl Registers {
    pub fn new() -> Self {
        Self { map: BTreeMap::new() }
    }

    /// Store into `name` (or the unnamed register), maintaining Vim's rules:
    ///
    /// * a **yank** also lands in `0`, which is what makes `"0p` reliably paste
    ///   the last yank after intervening deletes;
    /// * a **delete** shifts the numbered ring `1`-`9`, so `"1p` … `"9p` walk
    ///   back through recent deletions;
    /// * an **uppercase** name appends instead of replacing.
    pub fn set(&mut self, name: Option<char>, reg: Reg, is_delete: bool) {
        if let Some(n) = name {
            if n.is_ascii_uppercase() {
                let lower = n.to_ascii_lowercase();
                let mut merged = self.map.get(&lower).cloned().unwrap_or(Reg {
                    text: Vec::new(),
                    kind: reg.kind,
                });
                merged.text.extend(reg.text.iter().cloned());
                merged.kind = reg.kind;
                self.map.insert(lower, merged.clone());
                self.map.insert('"', merged);
                return;
            }
            self.map.insert(n, reg.clone());
            self.map.insert('"', reg);
            return;
        }
        if is_delete {
            for i in (1..9u8).rev() {
                let from = (b'0' + i) as char;
                let to = (b'0' + i + 1) as char;
                if let Some(r) = self.map.get(&from).cloned() {
                    self.map.insert(to, r);
                }
            }
            self.map.insert('1', reg.clone());
        } else {
            self.map.insert('0', reg.clone());
        }
        self.map.insert('"', reg);
    }

    pub fn get(&self, name: Option<char>) -> Option<&Reg> {
        self.map.get(&name.unwrap_or('"'))
    }
}

/// Keyword completion candidates for Ctrl+N / Ctrl+P: every distinct word in the
/// buffer that starts with `prefix`, nearest-first from `cur_row`.
///
/// Nearest-first matters more than it sounds — in a config file the useful
/// completion is almost always a key you just typed a few lines up, not the
/// alphabetically-first match in the file.
pub fn complete_prefix(lines: &[String], prefix: &str, cur_row: usize) -> Vec<String> {
    if prefix.is_empty() {
        return Vec::new();
    }
    let mut out: Vec<String> = Vec::new();
    // Walk outward from the cursor row so proximity decides the order.
    let mut order: Vec<usize> = Vec::with_capacity(lines.len());
    order.push(cur_row.min(lines.len().saturating_sub(1)));
    for d in 1..=lines.len() {
        if let Some(r) = cur_row.checked_sub(d) {
            order.push(r);
        }
        if cur_row + d < lines.len() {
            order.push(cur_row + d);
        }
    }
    for row in order {
        let Some(line) = lines.get(row) else { continue };
        for word in line.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')) {
            if word.len() > prefix.len() && word.starts_with(prefix) && !out.iter().any(|w| w == word) {
                out.push(String::from(word));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;

    fn buf(s: &[&str]) -> Vec<String> {
        s.iter().map(|l| l.to_string()).collect()
    }

    #[test_case]
    fn undo_restores_the_buffer_and_the_cursor() {
        let mut u = UndoStack::new();
        let before = buf(&["one", "two"]);
        u.checkpoint(&before, 1, 2);
        let after = buf(&["one"]);
        let snap = u.undo(&after, 0, 0).unwrap();
        assert_eq!(snap.lines, before);
        assert_eq!((snap.cy, snap.cx), (1, 2), "undo must restore the view, not jump to the top");
    }

    #[test_case]
    fn redo_returns_the_undone_change() {
        let mut u = UndoStack::new();
        let a = buf(&["a"]);
        u.checkpoint(&a, 0, 0);
        let b = buf(&["a", "b"]);
        let back = u.undo(&b, 1, 0).unwrap();
        assert_eq!(back.lines, a);
        let fwd = u.redo(&back.lines, 0, 0).unwrap();
        assert_eq!(fwd.lines, b);
    }

    #[test_case]
    fn undo_on_an_empty_history_is_a_no_op() {
        let mut u = UndoStack::new();
        assert!(!u.can_undo());
        assert!(u.undo(&buf(&["x"]), 0, 0).is_none());
        assert!(u.redo(&buf(&["x"]), 0, 0).is_none());
    }

    #[test_case]
    fn a_new_change_discards_the_redo_branch() {
        let mut u = UndoStack::new();
        u.checkpoint(&buf(&["a"]), 0, 0);
        u.undo(&buf(&["ab"]), 0, 0).unwrap();
        assert!(u.can_redo());
        u.checkpoint(&buf(&["a"]), 0, 0);
        assert!(!u.can_redo(), "editing after an undo abandons the redo branch");
    }

    #[test_case]
    fn history_is_bounded() {
        let mut u = UndoStack::new();
        for i in 0..MAX_UNDO + 20 {
            u.checkpoint(&buf(&["x"]), i, 0);
        }
        assert_eq!(u.past.len(), MAX_UNDO, "unbounded snapshots would be a slow leak");
        // The oldest were dropped, so the deepest undo is not step 0 any more.
        assert!(u.past[0].cy >= 20);
    }

    #[test_case]
    fn a_yank_fills_register_zero_and_a_delete_fills_one() {
        let mut r = Registers::new();
        r.set(None, Reg { text: vec!["yanked".to_string()], kind: RegKind::Line }, false);
        r.set(None, Reg { text: vec!["deleted".to_string()], kind: RegKind::Line }, true);
        // The unnamed register holds the most recent of either.
        assert_eq!(r.get(None).unwrap().text[0], "deleted");
        // But "0 still has the yank -- the whole point of the numbered ring.
        assert_eq!(r.get(Some('0')).unwrap().text[0], "yanked");
        assert_eq!(r.get(Some('1')).unwrap().text[0], "deleted");
    }

    #[test_case]
    fn deletes_shift_the_numbered_ring() {
        let mut r = Registers::new();
        for n in ["first", "second", "third"] {
            r.set(None, Reg { text: vec![n.to_string()], kind: RegKind::Line }, true);
        }
        assert_eq!(r.get(Some('1')).unwrap().text[0], "third");
        assert_eq!(r.get(Some('2')).unwrap().text[0], "second");
        assert_eq!(r.get(Some('3')).unwrap().text[0], "first");
    }

    #[test_case]
    fn an_uppercase_register_appends() {
        let mut r = Registers::new();
        r.set(Some('a'), Reg { text: vec!["one".to_string()], kind: RegKind::Line }, false);
        r.set(Some('A'), Reg { text: vec!["two".to_string()], kind: RegKind::Line }, false);
        let a = r.get(Some('a')).unwrap();
        assert_eq!(a.text, vec!["one".to_string(), "two".to_string()]);
    }

    #[test_case]
    fn register_kind_survives_the_round_trip() {
        // `dd` then `p` opens a new line; `dw` then `p` pastes inline. The
        // register is the only thing that remembers which.
        let mut r = Registers::new();
        r.set(None, Reg { text: vec!["word".to_string()], kind: RegKind::Char }, true);
        assert_eq!(r.get(None).unwrap().kind, RegKind::Char);
        r.set(None, Reg { text: vec!["line".to_string()], kind: RegKind::Line }, true);
        assert_eq!(r.get(None).unwrap().kind, RegKind::Line);
    }

    #[test_case]
    fn completion_offers_nearest_words_first() {
        let b = buf(&["far_away_value", "", "", "near_value", "cursor_here"]);
        let hits = complete_prefix(&b, "n", 4);
        assert_eq!(hits.first().map(|s| s.as_str()), Some("near_value"));
    }

    #[test_case]
    fn completion_skips_the_prefix_itself_and_deduplicates() {
        let b = buf(&["alpha alpha alpha", "al"]);
        let hits = complete_prefix(&b, "al", 0);
        assert_eq!(hits, vec!["alpha".to_string()], "one entry, and never the bare prefix");
    }

    #[test_case]
    fn completion_splits_on_punctuation() {
        let b = buf(&["config.timeout_ms = 5"]);
        let hits = complete_prefix(&b, "time", 0);
        assert_eq!(hits, vec!["timeout_ms".to_string()], "a dotted path is two words");
    }

    #[test_case]
    fn an_empty_prefix_offers_nothing() {
        // Otherwise Ctrl+N on whitespace dumps the entire buffer's vocabulary.
        assert!(complete_prefix(&buf(&["a b c"]), "", 0).is_empty());
    }
}
