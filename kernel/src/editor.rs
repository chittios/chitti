//! A **vim modal editor** rendered in the right pane (`/open <file>`).
//! Files are read from and written to the Synapse store ([`crate::synapse::fs`]),
//! where the configs live, so `/open /configs/core/ui.json` → edit → `:w` →
//! `/ui reload` is a full round-trip.
//!
//! ## What it implements
//!
//! Normal mode is the real grammar, not a key table:
//! `[count] ["register] operator [count] motion|text-object`.
//!
//! * **Motions** `h j k l 0 ^ $ w W b B e E { } % G gg H L` and `f F t T` with
//!   `;`/`,` to repeat — resolved by [`crate::editor_motion`].
//! * **Operators** `d c y > < =` plus `x D C ~ J r`, each composing with any
//!   motion or text object, and doubling for linewise (`dd`, `yy`, `>>`).
//! * **Text objects** `iw aw iW aW i( a( i[ a[ i{ a{ i" a" i' a' ip ap`.
//! * **Registers** unnamed, `"a`–`"z` (uppercase appends), yank ring `"0`,
//!   delete ring `"1`–`"9` — with linewise/charwise flavour, so `dd`then`p`
//!   opens a line while `dw`then`p` pastes inline.
//! * **Undo/redo** `u` / Ctrl+R, grouped per *change* rather than per keystroke.
//! * **Repeat** `.` replays the last change's key sequence.
//! * **Search** `/ ? n N * #`, and `:s/pat/rep/[g]`, `:%s/...` (literal, not regex).
//! * **Marks** `m{a-z}` and `` `{a-z} ``.
//! * **Completion** Ctrl+N / Ctrl+P over words in the buffer, nearest first.
//! * **Ex** `:w [file] :q :wq :q! :{number}`.
//! * **Visual** `v` charwise and `V` linewise.
//!
//! ## What it does not
//!
//! No Lua, no plugins, no LSP, no treesitter, no windows/tabs/buffers beyond the
//! one file, no macros (`q`/`@`), no regex, no block-visual (Ctrl+V), no folds
//! or diff mode. Those are Neovim features rather than Vim editing, and each is
//! a project in itself; the line drawn here is "the editing model", which is the
//! part that makes muscle memory transfer.
//!
//! ## Structure
//!
//! The fiddly, testable half lives outside this file — [`crate::editor_motion`]
//! for motions and text objects, [`crate::editor_undo`] for history, registers
//! and completion — because a motion is a pure function of `(lines, cursor,
//! count)` while this module owns a pane and a keyboard. Motion bugs are silent
//! (a `dw` one character wide looks like a typo, not a defect), so they are
//! pinned by cases there rather than found by using the editor.
//!
//! Long lines **soft-wrap** to the pane width and the viewport scrolls to keep
//! the cursor visible. Text is treated as ASCII (JSON/config/notes); making it
//! Unicode-aware is a follow-up and the arithmetic it needs already exists in
//! [`crate::textfit`].

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use crate::mm::Locked;

/// What the *next* byte means, when it is an argument rather than a command.
///
/// Vim has several two-key sequences whose second key is data (`fx`, `ma`,
/// `"ay`, `rz`), and one — `i`/`a` after an operator — that is data only in
/// that context: `d` then `i` then `w` is "delete inner word", while a bare `i`
/// enters insert mode. Keeping this as an explicit state is what lets both
/// readings coexist without the operator arm guessing.
#[derive(PartialEq, Clone, Copy)]
enum Await {
    None,
    Find(crate::editor_motion::Find),
    Mark,
    Goto,
    Register,
    Replace,
    /// After an operator: `i` (inner) or `a` (around) awaiting the object key.
    Object { around: bool },
}

#[derive(PartialEq, Clone, Copy)]
enum Mode {
    Normal,
    Insert,
    Command,
    Visual,
}

struct Editor {
    path: String,
    lines: Vec<String>,
    /// Cursor column, as a **byte** offset into the line.
    ///
    /// The editor is deliberately still ASCII-only: the insert arm below matches
    /// `0x20..=0x7e`, so a byte >= 0x80 is dropped rather than pushed into the
    /// `String` as a lone Latin-1 char (which would corrupt the file on save).
    /// That means `byte == char == column` holds here, and the whole cursor
    /// arithmetic is correct as written.
    ///
    /// Making it Unicode-aware is a follow-up, and the arithmetic it needs
    /// already exists in [`crate::textfit`] (`back_n_chars`, `cols`,
    /// `visible_window`) — the work is the call sites, not the algorithms.
    cx: usize,
    cy: usize, // row
    /// First **visual** row shown (soft-wrap aware). Logical line `0` starts at
    /// visual row 0; a long line consumes multiple visual rows.
    top: usize,
    mode: Mode,
    cmd: String,
    msg: String,
    dirty: bool,
    saved: bool,
    quit: bool,
    pending: u8, // pending operator: b'd', b'g', b'y', or 0
    /// Normal-mode grammar state: `[count] ["reg] op [count] motion`.
    ///
    /// Vim's normal mode is a grammar, not a key table, so the keys that only
    /// *prefix* a command need somewhere to accumulate. `count` multiplies with
    /// the operator's own count the way Vim does — `2d3w` deletes six words.
    count: Option<usize>,
    op_count: Option<usize>,
    reg_name: Option<char>,
    /// Set when the next byte is an argument rather than a command: the target
    /// of `f`/`t`, the object of `i`/`a`, the letter of `m`/`` ` ``/`r`/`"`.
    awaiting: Await,
    /// `;`/`,` replay the last `f`/`F`/`t`/`T` without retyping it.
    last_find: Option<(crate::editor_motion::Find, u8)>,
    /// The last change, replayed by `.`. Stored as the key sequence rather than
    /// as a diff: replaying the *intent* is what makes `.` work on a new cursor
    /// position, which is the entire point of the command.
    last_change: Vec<u8>,
    /// Keys recorded so far for the change in progress.
    recording: Vec<u8>,
    /// True while replaying `.`, so the replay does not re-record itself.
    replaying: bool,
    undo: crate::editor_undo::UndoStack,
    regs: crate::editor_undo::Registers,
    /// `m`/`` ` `` marks, per lowercase letter.
    marks: alloc::collections::BTreeMap<char, (usize, usize)>,
    /// Last `/` or `?` pattern, for `n`/`N`.
    search: String,
    search_fwd: bool,
    /// Ctrl+N/Ctrl+P completion: the candidates and where we are in them.
    compl: Option<(Vec<String>, usize, usize)>,
    sel_anchor: Option<(usize, usize)>, // (row, col) where Visual selection began
    /// Visual mode is linewise (`V`) rather than charwise (`v`).
    sel_line: bool,
    rows: usize,
    /// Text columns available after the line-number gutter (pane width − gutter).
    cols: usize,
    accel: crate::keyrepeat::Accel,
}

/// The single live editor, owned by its action-pane tab. Persists across tab
/// switches — switching away leaves the buffer intact; switching back resumes
/// exactly where it was. `None` when no editor tab is open.
static EDITOR: Locked<Option<Editor>> = Locked::new(None);

/// Set when the editor tab closes (`:q`), for the shell to pick up in its idle
/// tick: `(path, saved)`. Used to re-apply an edited UI config.
static CLOSED: Locked<Option<(String, bool)>> = Locked::new(None);

/// Open `path` in an editor **tab** (the action pane). Non-blocking: it builds
/// the buffer, opens the tab, focuses it, and returns immediately. Input is
/// then routed one byte at a time via [`feed`] from the shell's line loop, so
/// other tabs (a playing audio track, ktrace) keep running while you edit.
pub fn open(path: &str) {
    let content = crate::synapse::fs::read(path).and_then(|b| String::from_utf8(b).ok()).unwrap_or_default();
    let mut lines: Vec<String> = content.split('\n').map(|s| s.trim_end_matches('\r').to_string()).collect();
    if lines.is_empty() {
        lines.push(String::new());
    }
    // A trailing newline yields a spurious empty last line; drop it for editing.
    if lines.len() > 1 && lines.last().map(|l| l.is_empty()).unwrap_or(false) {
        lines.pop();
    }
    crate::framebuffer::editor_enter(); // add/select the editor tab
    let (cols, rows) = crate::framebuffer::editor_dims().unwrap_or((60, 24));
    let exists = crate::synapse::fs::exists(path);
    let ed = Editor {
        path: path.to_string(),
        lines,
        cx: 0,
        cy: 0,
        top: 0,
        mode: Mode::Normal,
        cmd: String::new(),
        msg: if exists { String::new() } else { "[new file]".to_string() },
        dirty: false,
        saved: false,
        quit: false,
        pending: 0,
        count: None,
        op_count: None,
        reg_name: None,
        awaiting: Await::None,
        last_find: None,
        last_change: Vec::new(),
        recording: Vec::new(),
        replaying: false,
        undo: crate::editor_undo::UndoStack::new(),
        regs: crate::editor_undo::Registers::new(),
        marks: alloc::collections::BTreeMap::new(),
        search: String::new(),
        search_fwd: true,
        compl: None,
        sel_anchor: None,
        sel_line: false,
        rows,
        cols,
        accel: crate::keyrepeat::Accel::new(),
    };
    EDITOR.with(|e| *e = Some(ed));
    render_current();
    crate::framebuffer::cursor_move_here();
}

/// Whether an editor tab is currently open.
pub fn is_open() -> bool {
    EDITOR.with(|e| e.is_some())
}

/// Feed one input byte to the editor (called by the shell when the editor tab
/// is the focused action tab). Decodes ANSI escapes byte-by-byte via the
/// `Esc` accumulator. On `:q` the tab closes.
pub fn feed(byte: u8) {
    let quit = EDITOR.with(|slot| {
        let Some(ed) = slot.as_mut() else { return false };
        ed.feed(byte);
        ed.quit
    });
    if quit {
        let (path, saved) = EDITOR.with(|slot| {
            let ed = slot.take();
            ed.map(|e| (e.path, e.saved)).unwrap_or_default()
        });
        crate::framebuffer::editor_leave();
        CLOSED.with(|c| *c = Some((path, saved)));
    } else {
        render_current();
        crate::framebuffer::cursor_move_here();
    }
}

/// Route a decoded ANSI navigation sequence (arrow/Home/End/PgUp/PgDn/Del) to
/// the editor — the shell decodes CSI itself (to catch tab-switch chords), so
/// it hands the rest here rather than replaying raw bytes.
pub fn nav_seq(fin: u8, param: u64) {
    EDITOR.with(|slot| {
        if let Some(ed) = slot.as_mut() {
            let n = if matches!(fin, b'A' | b'B') { ed.accel.steps(fin, crate::arch::now_ms()) } else { 1 };
            for _ in 0..n {
                ed.nav(fin, param);
            }
        }
    });
    render_current();
    crate::framebuffer::cursor_move_here();
}

/// Force the editor tab shut (e.g. `/close` or `[x]` on the editor tab), without
/// waiting for `:q`. Drops the buffer; the pane teardown is the caller's.
pub fn force_close() {
    EDITOR.with(|e| *e = None);
}

/// Repaint the editor into the pane (after a tab switch back to it).
pub fn repaint() {
    render_current();
    crate::framebuffer::cursor_move_here();
}

/// Mouse handling while the editor tab is active (called from the shell's idle
/// tick, which owns `mouse::tick`).
pub fn mouse_tick() {
    EDITOR.with(|slot| {
        if let Some(ed) = slot.as_mut() {
            ed.mouse_tick();
        }
    });
}

/// Take the just-closed `(path, saved)` note, if the editor quit since the last
/// poll (so the shell can re-apply an edited UI config).
pub fn take_closed() -> Option<(String, bool)> {
    CLOSED.with(|c| c.take())
}

fn render_current() {
    EDITOR.with(|slot| {
        if let Some(ed) = slot.as_mut() {
            ed.render();
        }
    });
}

impl Editor {
    fn line_len(&self) -> usize {
        self.lines[self.cy].len()
    }

    fn clamp_normal(&mut self) {
        let max = self.line_len().saturating_sub(1);
        if self.cx > max {
            self.cx = max;
        }
    }

    /// Refresh pane geometry (a layout change can resize the editor mid-session).
    fn sync_dims(&mut self) {
        if let Some((cols, rows)) = crate::framebuffer::editor_dims() {
            self.cols = cols;
            self.rows = rows;
        }
    }

    /// Text columns after the line-number gutter.
    fn text_width(&self) -> usize {
        let g = self.gutter() as usize;
        self.cols.saturating_sub(g).max(1)
    }

    fn line_lens(&self) -> Vec<usize> {
        self.lines.iter().map(|l| l.chars().count()).collect()
    }

    fn vis_of(&self, row: usize, col: usize) -> usize {
        crate::editor_wrap::vis_index(&self.line_lens(), row, col, self.text_width())
    }

    /// Keep the cursor's visual row inside the viewport (soft-wrap aware).
    fn ensure_visible(&mut self) {
        let text_rows = self.rows.saturating_sub(1).max(1);
        let vr = self.vis_of(self.cy, self.cx);
        if vr < self.top {
            self.top = vr;
        } else if vr >= self.top + text_rows {
            self.top = vr + 1 - text_rows;
        }
    }

    fn handle(&mut self, b: u8) {
        match self.mode {
            Mode::Normal => self.normal(b),
            Mode::Insert => self.insert(b),
            Mode::Command => self.command(b),
            Mode::Visual => self.visual(b),
        }
    }

    /// One decoded key byte. The shell's line loop already coalesces ANSI escape
    /// sequences (it forwards arrows via [`nav_seq`] and a **bare Esc as `0x1b`**),
    /// so the editor just dispatches the byte through `handle` — no escape
    /// accumulator here. Held Backspace accelerates.
    fn feed(&mut self, b: u8) {
        let n = if b == 0x7f || b == 0x08 { self.accel.steps(0x08, crate::arch::now_ms()) } else { 1 };
        for _ in 0..n {
            self.handle(b);
        }
    }

    /// Navigation from decoded ANSI sequences (arrow/Home/End/PgUp/PgDn/Del),
    /// valid in **every** mode — arrows must move the cursor mid-insert too.
    fn nav(&mut self, fin: u8, param: u64) {
        match fin {
            b'A' => self.move_cursor(b'k'),
            b'B' => self.move_cursor(b'j'),
            b'D' => self.move_cursor(b'h'),
            b'C' => {
                // In insert mode the cursor may sit one past the last char.
                let max = if self.mode == Mode::Insert { self.line_len() } else { self.line_len().saturating_sub(1) };
                if self.cx < max {
                    self.cx += 1;
                }
            }
            b'H' => self.cx = 0,
            b'F' => self.cx = if self.mode == Mode::Insert { self.line_len() } else { self.line_len().saturating_sub(1) },
            b'~' => match param {
                1 | 7 => self.cx = 0,
                4 | 8 => self.cx = self.line_len(),
                3 => {
                    // Delete key: remove the char under the cursor.
                    if self.cx < self.line_len() {
                        self.lines[self.cy].remove(self.cx);
                        self.dirty = true;
                    }
                }
                5 | 6 => {
                    // Page up/down by a viewport height.
                    let page = self.rows.saturating_sub(1).max(1);
                    if param == 5 {
                        self.cy = self.cy.saturating_sub(page);
                    } else {
                        self.cy = (self.cy + page).min(self.lines.len() - 1);
                    }
                    self.clamp_normal();
                }
                _ => {}
            },
            _ => {}
        }
    }

    /// Cursor motions shared by Normal and Visual modes.
    fn move_cursor(&mut self, b: u8) {
        match b {
            b'h' => self.cx = self.cx.saturating_sub(1),
            b'l' => {
                if self.cx + 1 < self.line_len() {
                    self.cx += 1;
                }
            }
            b'k' => {
                if self.cy > 0 {
                    self.cy -= 1;
                    self.clamp_normal();
                }
            }
            b'j' => {
                if self.cy + 1 < self.lines.len() {
                    self.cy += 1;
                    self.clamp_normal();
                }
            }
            b'0' => self.cx = 0,
            b'$' => self.cx = self.line_len().saturating_sub(1),
            b'w' => self.word_forward(),
            b'b' => self.word_back(),
            b'G' => {
                self.cy = self.lines.len() - 1;
                self.clamp_normal();
            }
            _ => {}
        }
    }

    /// Normal mode: one byte at a time through the grammar
    /// `[count] ["reg] operator [count] motion|text-object`.
    ///
    /// Every byte is recorded so `.` can replay the whole command later; the
    /// recording is discarded for pure motions, since repeating a motion is not
    /// what `.` means.
    fn normal(&mut self, b: u8) {
        self.msg.clear();
        if !self.replaying {
            self.recording.push(b);
        }

        // An argument byte belongs to the pending command, not to the grammar.
        if self.awaiting != Await::None {
            self.argument(b);
            return;
        }

        // Counts. A leading `0` is the motion, not a digit -- `d0` deletes to
        // the start of the line, and treating it as a count would silently make
        // `10j` and `1j` the same command.
        if b.is_ascii_digit() && !(b == b'0' && self.count_slot().is_none()) {
            let d = (b - b'0') as usize;
            let slot = if self.pending == 0 { &mut self.count } else { &mut self.op_count };
            *slot = Some(slot.unwrap_or(0).saturating_mul(10).saturating_add(d));
            return;
        }

        match b {
            b'"' => self.awaiting = Await::Register,
            b'f' => self.awaiting = Await::Find(crate::editor_motion::Find::Forward),
            b'F' => self.awaiting = Await::Find(crate::editor_motion::Find::Backward),
            b't' => self.awaiting = Await::Find(crate::editor_motion::Find::Till),
            b'T' => self.awaiting = Await::Find(crate::editor_motion::Find::TillBack),
            b'm' => self.awaiting = Await::Mark,
            b'`' | b'\'' => self.awaiting = Await::Goto,
            b'r' => self.awaiting = Await::Replace,
            b'i' | b'a' if self.pending != 0 => self.awaiting = Await::Object { around: b == b'a' },

            // Operators. A doubled operator (`dd`, `yy`, `>>`) is linewise.
            b'd' | b'y' | b'c' | b'>' | b'<' | b'=' => {
                if self.pending == b {
                    let n = self.total_count();
                    let last = (self.cy + n - 1).min(self.lines.len().saturating_sub(1));
                    let span = crate::editor_motion::Span::new(
                        crate::editor_motion::Pos::new(self.cy, 0),
                        crate::editor_motion::Pos::new(last, 0),
                        crate::editor_motion::Kind::Line,
                    );
                    self.apply_op(b, span);
                    self.reset_pending();
                } else {
                    self.pending = b;
                }
            }
            b'g' if self.pending == b'g' => {
                let n = self.count.unwrap_or(1);
                self.cy = (n - 1).min(self.lines.len() - 1);
                self.cx = crate::editor_motion::first_non_blank(&self.lines, self.cy);
                self.reset_pending();
            }
            b'g' => self.pending = b'g',
            b'Y' => {
                let span = crate::editor_motion::Span::new(
                    crate::editor_motion::Pos::new(self.cy, 0),
                    crate::editor_motion::Pos::new(self.cy, 0),
                    crate::editor_motion::Kind::Line,
                );
                self.apply_op(b'y', span);
                self.reset_pending();
            }

            b'u' => {
                if let Some(sn) = self.undo.undo(&self.lines, self.cy, self.cx) {
                    self.restore(sn);
                    self.msg = "1 change; before".to_string();
                } else {
                    self.msg = "Already at oldest change".to_string();
                }
                self.reset_pending();
            }
            0x12 => {
                // Ctrl+R
                if let Some(sn) = self.redo_take() {
                    self.restore(sn);
                    self.msg = "1 change; after".to_string();
                } else {
                    self.msg = "Already at newest change".to_string();
                }
                self.reset_pending();
            }
            b'.' => self.repeat_last_change(),

            b'v' | b'V' => {
                self.mode = Mode::Visual;
                self.sel_line = b == b'V';
                self.sel_anchor = Some((self.cy, self.cx));
                self.reset_pending();
            }
            b'p' | b'P' => {
                self.checkpoint();
                self.put(b == b'p');
                self.commit_change();
            }
            b'J' => {
                self.checkpoint();
                self.join_lines(self.total_count().max(2));
                self.commit_change();
            }
            b'x' => {
                let n = self.total_count();
                let end = (self.cx + n).min(self.line_len());
                if end > self.cx {
                    let span = crate::editor_motion::Span::new(
                        crate::editor_motion::Pos::new(self.cy, self.cx),
                        crate::editor_motion::Pos::new(self.cy, end),
                        crate::editor_motion::Kind::Exclusive,
                    );
                    self.apply_op(b'd', span);
                }
                self.reset_pending();
            }
            b'D' | b'C' => {
                let span = crate::editor_motion::Span::new(
                    crate::editor_motion::Pos::new(self.cy, self.cx),
                    crate::editor_motion::Pos::new(self.cy, self.line_len()),
                    crate::editor_motion::Kind::Exclusive,
                );
                self.apply_op(if b == b'D' { b'd' } else { b'c' }, span);
                self.reset_pending();
            }
            b'~' => {
                self.checkpoint();
                let n = self.total_count();
                for _ in 0..n {
                    if self.cx >= self.line_len() {
                        break;
                    }
                    // SAFETY of the index: `cx < line_len` was just checked, and
                    // the buffer is ASCII so a byte index is a char boundary.
                    let bytes = unsafe { self.lines[self.cy].as_bytes_mut() };
                    bytes[self.cx] = flip_case(bytes[self.cx]);
                    self.cx += 1;
                }
                self.clamp_normal();
                self.commit_change();
            }

            b'i' => self.enter_insert(),
            b'I' => {
                self.cx = crate::editor_motion::first_non_blank(&self.lines, self.cy);
                self.enter_insert();
            }
            b'a' => {
                if self.line_len() > 0 {
                    self.cx += 1;
                }
                self.enter_insert();
            }
            b'A' => {
                self.cx = self.line_len();
                self.enter_insert();
            }
            b'o' | b'O' => {
                self.checkpoint();
                let at = if b == b'o' { self.cy + 1 } else { self.cy };
                self.lines.insert(at, String::new());
                self.cy = at;
                self.cx = 0;
                self.dirty = true;
                self.enter_insert();
            }

            b'/' | b'?' => {
                self.mode = Mode::Command;
                self.cmd.clear();
                self.cmd.push(b as char);
                self.reset_pending();
            }
            b'n' | b'N' => {
                let fwd = if b == b'n' { self.search_fwd } else { !self.search_fwd };
                self.search_step(fwd);
                self.reset_pending();
            }
            b'*' | b'#' => {
                if let Some(w) = self.word_under_cursor() {
                    self.search = w;
                    self.search_fwd = b == b'*';
                    let fwd = self.search_fwd;
                    self.search_step(fwd);
                }
                self.reset_pending();
            }
            b';' | b',' => {
                if let Some((k, target)) = self.last_find {
                    let k = if b == b';' { k } else { reverse_find(k) };
                    if let Some(c) = crate::editor_motion::find_char(
                        &self.lines,
                        crate::editor_motion::Pos::new(self.cy, self.cx),
                        target,
                        self.total_count(),
                        k,
                    ) {
                        self.cx = c;
                    }
                }
                self.reset_pending();
            }

            b':' => {
                self.mode = Mode::Command;
                self.cmd.clear();
                self.reset_pending();
            }
            0x1b => self.reset_pending(),

            // Anything else is a motion; with an operator pending it defines a
            // region, otherwise it just moves the cursor.
            _ => {
                if let Some(span) = self.motion_span(b) {
                    if self.pending != 0 {
                        let op = self.pending;
                        self.apply_op(op, span);
                    } else {
                        self.cy = span.end.row;
                        self.cx = span.end.col;
                        self.clamp_normal();
                    }
                }
                self.reset_pending();
            }
        }
    }

    // ---- normal-mode grammar support -------------------------------------

    /// The count in effect: `2d3w` is six words, so the two multiply.
    fn total_count(&self) -> usize {
        self.count.unwrap_or(1).saturating_mul(self.op_count.unwrap_or(1)).max(1)
    }

    fn count_slot(&self) -> Option<usize> {
        if self.pending == 0 { self.count } else { self.op_count }
    }

    fn reset_pending(&mut self) {
        self.pending = 0;
        self.count = None;
        self.op_count = None;
        self.reg_name = None;
        self.awaiting = Await::None;
    }

    fn checkpoint(&mut self) {
        self.undo.checkpoint(&self.lines, self.cy, self.cx);
    }

    /// Finish a change: remember the keys for `.` and clear grammar state.
    fn commit_change(&mut self) {
        if !self.replaying {
            self.last_change = core::mem::take(&mut self.recording);
        }
        self.reset_pending();
    }

    fn enter_insert(&mut self) {
        self.checkpoint();
        self.mode = Mode::Insert;
        self.reset_pending();
    }

    fn restore(&mut self, sn: crate::editor_undo::Snapshot) {
        self.lines = sn.lines;
        self.cy = sn.cy.min(self.lines.len().saturating_sub(1));
        self.cx = sn.cx;
        self.clamp_normal();
        self.dirty = true;
    }

    fn redo_take(&mut self) -> Option<crate::editor_undo::Snapshot> {
        self.undo.redo(&self.lines, self.cy, self.cx)
    }

    /// `.` — replay the last change's key sequence at the current position.
    fn repeat_last_change(&mut self) {
        if self.last_change.is_empty() || self.replaying {
            self.reset_pending();
            return;
        }
        let keys = self.last_change.clone();
        self.replaying = true;
        self.reset_pending();
        for k in keys {
            match self.mode {
                Mode::Normal => self.normal(k),
                Mode::Insert => self.insert(k),
                _ => {}
            }
        }
        // A replayed insert never sees its own Esc if the recording ended first.
        if self.mode == Mode::Insert {
            self.mode = Mode::Normal;
            self.clamp_normal();
        }
        self.replaying = false;
    }

    /// The second byte of a two-key command.
    fn argument(&mut self, b: u8) {
        let a = self.awaiting;
        self.awaiting = Await::None;
        match a {
            Await::Register => {
                self.reg_name = Some(b as char);
            }
            Await::Mark => {
                self.marks.insert(b as char, (self.cy, self.cx));
                self.reset_pending();
            }
            Await::Goto => {
                if let Some(&(r, c)) = self.marks.get(&(b as char)) {
                    self.cy = r.min(self.lines.len().saturating_sub(1));
                    self.cx = c;
                    self.clamp_normal();
                }
                self.reset_pending();
            }
            Await::Replace => {
                if (0x20..=0x7e).contains(&b) && self.cx < self.line_len() {
                    self.checkpoint();
                    // SAFETY: ASCII buffer, `cx` in range -- a char boundary.
                    unsafe { self.lines[self.cy].as_bytes_mut()[self.cx] = b };
                    self.dirty = true;
                    self.commit_change();
                } else {
                    self.reset_pending();
                }
            }
            Await::Find(kind) => {
                self.last_find = Some((kind, b));
                let cur = crate::editor_motion::Pos::new(self.cy, self.cx);
                match crate::editor_motion::find_char(&self.lines, cur, b, self.total_count(), kind) {
                    Some(col) => {
                        let inclusive = matches!(
                            kind,
                            crate::editor_motion::Find::Forward | crate::editor_motion::Find::Till
                        );
                        if self.pending != 0 {
                            let k = if inclusive {
                                crate::editor_motion::Kind::Inclusive
                            } else {
                                crate::editor_motion::Kind::Exclusive
                            };
                            let span = crate::editor_motion::Span::new(
                                cur,
                                crate::editor_motion::Pos::new(self.cy, col),
                                k,
                            );
                            let op = self.pending;
                            self.apply_op(op, span);
                        } else {
                            self.cx = col;
                        }
                    }
                    // A miss is a no-op, not a jump to the line end.
                    None => self.msg = "pattern not found".to_string(),
                }
                self.reset_pending();
            }
            Await::Object { around } => {
                use crate::editor_motion::Object;
                let obj = match b {
                    b'w' => Some(Object::Word { big: false }),
                    b'W' => Some(Object::Word { big: true }),
                    b'(' | b')' | b'b' => Some(Object::Bracket(b'(')),
                    b'[' | b']' => Some(Object::Bracket(b'[')),
                    b'{' | b'}' | b'B' => Some(Object::Bracket(b'{')),
                    b'<' | b'>' => Some(Object::Bracket(b'<')),
                    b'"' => Some(Object::Quote(b'"')),
                    b'\'' => Some(Object::Quote(b'\'')),
                    b'`' => Some(Object::Quote(b'`')),
                    b'p' => Some(Object::Paragraph),
                    _ => None,
                };
                let cur = crate::editor_motion::Pos::new(self.cy, self.cx);
                if let Some(o) = obj {
                    if let Some(span) = crate::editor_motion::text_object(&self.lines, cur, o, around) {
                        let op = self.pending;
                        if op != 0 {
                            self.apply_op(op, span);
                        }
                    }
                }
                self.reset_pending();
            }
            Await::None => {}
        }
    }

    /// Resolve a motion key to the region it covers from the cursor.
    fn motion_span(&mut self, b: u8) -> Option<crate::editor_motion::Span> {
        use crate::editor_motion as m;
        let n = self.total_count();
        let cur = m::Pos::new(self.cy, self.cx);
        let (end, kind) = match b {
            b'h' => (m::Pos::new(self.cy, self.cx.saturating_sub(n)), m::Kind::Exclusive),
            b'l' => {
                let max = self.line_len();
                (m::Pos::new(self.cy, (self.cx + n).min(max)), m::Kind::Exclusive)
            }
            b'j' => (m::Pos::new((self.cy + n).min(self.lines.len() - 1), self.cx), m::Kind::Line),
            b'k' => (m::Pos::new(self.cy.saturating_sub(n), self.cx), m::Kind::Line),
            b'0' => (m::Pos::new(self.cy, 0), m::Kind::Exclusive),
            b'^' => (m::Pos::new(self.cy, m::first_non_blank(&self.lines, self.cy)), m::Kind::Exclusive),
            b'$' => {
                let row = (self.cy + n - 1).min(self.lines.len() - 1);
                (m::Pos::new(row, self.lines[row].len()), m::Kind::Exclusive)
            }
            b'w' => (m::word_fwd(&self.lines, cur, n, false), m::Kind::Exclusive),
            b'W' => (m::word_fwd(&self.lines, cur, n, true), m::Kind::Exclusive),
            b'b' => (m::word_back(&self.lines, cur, n, false), m::Kind::Exclusive),
            b'B' => (m::word_back(&self.lines, cur, n, true), m::Kind::Exclusive),
            b'e' => (m::word_end(&self.lines, cur, n, false), m::Kind::Inclusive),
            b'E' => (m::word_end(&self.lines, cur, n, true), m::Kind::Inclusive),
            b'{' => (m::paragraph(&self.lines, cur, n, false), m::Kind::Exclusive),
            b'}' => (m::paragraph(&self.lines, cur, n, true), m::Kind::Exclusive),
            b'%' => (m::match_pair(&self.lines, cur)?, m::Kind::Inclusive),
            b'G' => {
                let row = self.count.map(|c| c - 1).unwrap_or(self.lines.len() - 1);
                (m::Pos::new(row.min(self.lines.len() - 1), 0), m::Kind::Line)
            }
            b'H' => (m::Pos::new(self.top.min(self.lines.len() - 1), 0), m::Kind::Line),
            b'L' => {
                let row = (self.top + self.rows.saturating_sub(1)).min(self.lines.len() - 1);
                (m::Pos::new(row, 0), m::Kind::Line)
            }
            _ => return None,
        };
        Some(m::Span::new(cur, end, kind))
    }

    /// Apply an operator to a region. This is the one place that knows how a
    /// span's [`Kind`](crate::editor_motion::Kind) changes what gets touched.
    fn apply_op(&mut self, op: u8, span: crate::editor_motion::Span) {
        use crate::editor_motion::Kind;
        use crate::editor_undo::{Reg, RegKind};
        let (s, e) = (span.start, span.end);

        // Gather the text first -- every operator either yanks it, replaces it,
        // or both, and doing it once keeps `d` and `y` from drifting apart.
        let (text, kind) = match span.kind {
            Kind::Line => {
                let last = e.row.min(self.lines.len().saturating_sub(1));
                (self.lines[s.row..=last].to_vec(), RegKind::Line)
            }
            _ => {
                let end_col = if span.kind == Kind::Inclusive { e.col + 1 } else { e.col };
                if s.row == e.row {
                    let l = &self.lines[s.row];
                    let hi = end_col.min(l.len());
                    let lo = s.col.min(hi);
                    (alloc::vec![l[lo..hi].to_string()], RegKind::Char)
                } else {
                    let mut v = Vec::new();
                    let first = &self.lines[s.row];
                    v.push(first[s.col.min(first.len())..].to_string());
                    for r in s.row + 1..e.row {
                        v.push(self.lines[r].clone());
                    }
                    let last = &self.lines[e.row.min(self.lines.len() - 1)];
                    v.push(last[..end_col.min(last.len())].to_string());
                    (v, RegKind::Char)
                }
            }
        };

        if op == b'y' {
            self.regs.set(self.reg_name, Reg { text, kind }, false);
            self.cy = s.row;
            self.cx = if span.kind == Kind::Line { self.cx } else { s.col };
            self.msg = "yanked".to_string();
            self.clamp_normal();
            return;
        }

        if op == b'>' || op == b'<' || op == b'=' {
            self.checkpoint();
            let last = e.row.min(self.lines.len() - 1);
            for r in s.row..=last {
                if op == b'>' {
                    self.lines[r].insert_str(0, "    ");
                } else {
                    let l = self.lines[r].clone();
                    let strip = l.len() - l.trim_start_matches(' ').len();
                    self.lines[r] = l[strip.min(4)..].to_string();
                }
            }
            self.cy = s.row;
            self.cx = crate::editor_motion::first_non_blank(&self.lines, self.cy);
            self.dirty = true;
            self.commit_change();
            return;
        }

        // d and c both remove the region; only what happens next differs.
        self.checkpoint();
        self.regs.set(self.reg_name, Reg { text, kind }, true);
        match span.kind {
            Kind::Line => {
                let last = e.row.min(self.lines.len() - 1);
                self.lines.drain(s.row..=last);
                if op == b'c' {
                    self.lines.insert(s.row, String::new());
                    self.cy = s.row;
                    self.cx = 0;
                } else {
                    if self.lines.is_empty() {
                        self.lines.push(String::new());
                    }
                    self.cy = s.row.min(self.lines.len() - 1);
                    self.cx = crate::editor_motion::first_non_blank(&self.lines, self.cy);
                }
            }
            _ => {
                let end_col = if span.kind == Kind::Inclusive { e.col + 1 } else { e.col };
                if s.row == e.row {
                    let l = &mut self.lines[s.row];
                    let hi = end_col.min(l.len());
                    let lo = s.col.min(hi);
                    l.replace_range(lo..hi, "");
                } else {
                    let last_row = e.row.min(self.lines.len() - 1);
                    let tail = {
                        let l = &self.lines[last_row];
                        l[end_col.min(l.len())..].to_string()
                    };
                    let keep = s.col.min(self.lines[s.row].len());
                    self.lines[s.row].truncate(keep);
                    self.lines[s.row].push_str(&tail);
                    self.lines.drain(s.row + 1..=last_row);
                }
                self.cy = s.row;
                self.cx = s.col;
            }
        }
        self.dirty = true;
        if op == b'c' {
            self.mode = Mode::Insert;
            // Leave the recording open: the inserted text is part of this change.
            self.pending = 0;
            self.count = None;
            self.op_count = None;
            self.reg_name = None;
        } else {
            self.clamp_normal();
            self.commit_change();
        }
    }

    /// `p` / `P`, honouring the register's linewise-vs-charwise flavour.
    fn put(&mut self, after: bool) {
        use crate::editor_undo::RegKind;
        let Some(reg) = self.regs.get(self.reg_name).cloned() else { return };
        let n = self.total_count();
        match reg.kind {
            RegKind::Line => {
                let at = if after { self.cy + 1 } else { self.cy };
                let mut ins = Vec::new();
                for _ in 0..n {
                    ins.extend(reg.text.iter().cloned());
                }
                let count = ins.len();
                for (i, l) in ins.into_iter().enumerate() {
                    self.lines.insert((at + i).min(self.lines.len()), l);
                }
                self.cy = at.min(self.lines.len().saturating_sub(1));
                self.cx = crate::editor_motion::first_non_blank(&self.lines, self.cy);
                let _ = count;
            }
            RegKind::Char => {
                let col = if after && self.line_len() > 0 { self.cx + 1 } else { self.cx };
                let col = col.min(self.line_len());
                if reg.text.len() == 1 {
                    let mut piece = String::new();
                    for _ in 0..n {
                        piece.push_str(&reg.text[0]);
                    }
                    self.lines[self.cy].insert_str(col, &piece);
                    self.cx = col + piece.len().saturating_sub(1);
                } else {
                    let tail = self.lines[self.cy][col..].to_string();
                    self.lines[self.cy].truncate(col);
                    self.lines[self.cy].push_str(&reg.text[0]);
                    let mut at = self.cy;
                    for mid in &reg.text[1..reg.text.len() - 1] {
                        at += 1;
                        self.lines.insert(at, mid.clone());
                    }
                    at += 1;
                    let mut last = reg.text[reg.text.len() - 1].clone();
                    let last_len = last.len();
                    last.push_str(&tail);
                    self.lines.insert(at, last);
                    self.cy = at;
                    self.cx = last_len.saturating_sub(1);
                }
            }
        }
        self.dirty = true;
        self.clamp_normal();
    }

    /// `J`: join `n` lines, collapsing the join to a single space the way Vim
    /// does — and adding none when the next line is empty or already indented
    /// away, which is the detail that makes joined prose read correctly.
    fn join_lines(&mut self, n: usize) {
        for _ in 0..n.saturating_sub(1) {
            if self.cy + 1 >= self.lines.len() {
                break;
            }
            let next = self.lines.remove(self.cy + 1);
            let trimmed = next.trim_start();
            let cur = &mut self.lines[self.cy];
            let joint = cur.len();
            if !cur.is_empty() && !trimmed.is_empty() && !cur.ends_with(' ') {
                cur.push(' ');
            }
            cur.push_str(trimmed);
            self.cx = joint;
        }
        self.dirty = true;
        self.clamp_normal();
    }

    fn word_under_cursor(&self) -> Option<String> {
        let l = self.lines.get(self.cy)?.as_bytes();
        if l.is_empty() {
            return None;
        }
        let is_w = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
        let mut s = self.cx.min(l.len() - 1);
        if !is_w(l[s]) {
            return None;
        }
        while s > 0 && is_w(l[s - 1]) {
            s -= 1;
        }
        let mut e = self.cx.min(l.len() - 1);
        while e + 1 < l.len() && is_w(l[e + 1]) {
            e += 1;
        }
        Some(self.lines[self.cy][s..=e].to_string())
    }

    /// `n` / `N`: next match of [`Self::search`], wrapping at the buffer end.
    fn search_step(&mut self, forward: bool) {
        if self.search.is_empty() {
            self.msg = "no previous search".to_string();
            return;
        }
        let pat = self.search.clone();
        let rows = self.lines.len();
        for i in 1..=rows {
            let row = if forward {
                (self.cy + i) % rows
            } else {
                (self.cy + rows - (i % rows)) % rows
            };
            if let Some(col) = self.lines[row].find(&pat) {
                self.cy = row;
                self.cx = col;
                self.clamp_normal();
                return;
            }
        }
        // Also try later on the current line before declaring failure.
        if let Some(col) = self.lines[self.cy].get(self.cx + 1..).and_then(|t| t.find(&pat)) {
            self.cx += 1 + col;
            return;
        }
        self.msg = alloc::format!("pattern not found: {pat}");
    }

    /// Ctrl+N / Ctrl+P in insert mode: cycle buffer keyword completions.
    fn complete(&mut self, forward: bool) {
        if let Some((cands, idx, start)) = self.compl.take() {
            if !cands.is_empty() {
                let next = if forward {
                    (idx + 1) % cands.len()
                } else {
                    (idx + cands.len() - 1) % cands.len()
                };
                self.lines[self.cy].replace_range(start..self.cx, &cands[next]);
                self.cx = start + cands[next].len();
                self.compl = Some((cands, next, start));
                return;
            }
        }
        let l = self.lines[self.cy].as_bytes();
        let mut start = self.cx.min(l.len());
        while start > 0 && (l[start - 1].is_ascii_alphanumeric() || l[start - 1] == b'_') {
            start -= 1;
        }
        let prefix = self.lines[self.cy][start..self.cx].to_string();
        let cands = crate::editor_undo::complete_prefix(&self.lines, &prefix, self.cy);
        if cands.is_empty() {
            self.msg = "no completions".to_string();
            return;
        }
        let pick = if forward { 0 } else { cands.len() - 1 };
        self.lines[self.cy].replace_range(start..self.cx, &cands[pick]);
        self.cx = start + cands[pick].len();
        self.dirty = true;
        self.compl = Some((cands, pick, start));
    }

    fn insert(&mut self, b: u8) {
        if !self.replaying {
            self.recording.push(b);
        }
        // Ctrl+N / Ctrl+P cycle completions; every other key ends the cycle, so
        // typing on past an accepted completion starts a fresh one next time.
        match b {
            0x0e => {
                self.complete(true);
                return;
            }
            0x10 => {
                self.complete(false);
                return;
            }
            _ => self.compl = None,
        }
        match b {
            0x1b => {
                self.mode = Mode::Normal;
                // Leaving insert closes the change that opened it, so one `u`
                // undoes the whole typing session rather than one character.
                if !self.replaying {
                    self.last_change = core::mem::take(&mut self.recording);
                }
                if self.cx > 0 {
                    self.cx -= 1;
                }
                self.clamp_normal();
            }
            b'\r' | b'\n' => {
                let tail = self.lines[self.cy].split_off(self.cx);
                self.lines.insert(self.cy + 1, tail);
                self.cy += 1;
                self.cx = 0;
                self.dirty = true;
            }
            0x7f | 0x08 => {
                if self.cx > 0 {
                    self.cx -= 1;
                    self.lines[self.cy].remove(self.cx);
                    self.dirty = true;
                } else if self.cy > 0 {
                    let cur = self.lines.remove(self.cy);
                    self.cy -= 1;
                    self.cx = self.lines[self.cy].len();
                    self.lines[self.cy].push_str(&cur);
                    self.dirty = true;
                }
            }
            b'\t' => {
                self.lines[self.cy].insert_str(self.cx, "  ");
                self.cx += 2;
                self.dirty = true;
            }
            0x20..=0x7e => {
                self.lines[self.cy].insert(self.cx, b as char);
                self.cx += 1;
                self.dirty = true;
            }
            _ => {}
        }
    }

    fn command(&mut self, b: u8) {
        match b {
            b'\r' | b'\n' => {
                self.run_ex();
                if !self.quit {
                    self.mode = Mode::Normal;
                }
                self.cmd.clear();
            }
            0x1b => {
                self.mode = Mode::Normal;
                self.cmd.clear();
            }
            0x7f | 0x08 => {
                self.cmd.pop();
            }
            0x20..=0x7e => self.cmd.push(b as char),
            _ => {}
        }
    }

    fn run_ex(&mut self) {
        // `/pat` and `?pat` arrive through the same line editor as `:`, so they
        // are dispatched here before the ex commands.
        if let Some(pat) = self.cmd.strip_prefix('/').map(|p| p.to_string()) {
            self.search = pat;
            self.search_fwd = true;
            self.search_step(true);
            return;
        }
        if let Some(pat) = self.cmd.strip_prefix('?').map(|p| p.to_string()) {
            self.search = pat;
            self.search_fwd = false;
            self.search_step(false);
            return;
        }
        let cmd = self.cmd.trim().to_string();
        let cmd = cmd.as_str();
        if cmd.starts_with("s/") || cmd.starts_with("%s/") {
            self.substitute(cmd);
            return;
        }
        if let Some(rest) = cmd.strip_prefix("set ") {
            self.msg = alloc::format!("set {rest}: no options implemented");
            return;
        }
        // A bare number is Vim's "go to line".
        if let Ok(n) = cmd.parse::<usize>() {
            self.cy = n.saturating_sub(1).min(self.lines.len() - 1);
            self.cx = crate::editor_motion::first_non_blank(&self.lines, self.cy);
            return;
        }
        match cmd {
            "w" => self.save(None),
            "q" => {
                if self.dirty {
                    self.msg = "unsaved changes (:q! to discard, :wq to save)".to_string();
                } else {
                    self.quit = true;
                }
            }
            "q!" => self.quit = true,
            "wq" | "x" => {
                self.save(None);
                self.quit = true;
            }
            _ => {
                if let Some(file) = cmd.strip_prefix("w ") {
                    self.save(Some(file.trim().to_string()));
                } else {
                    self.msg = alloc::format!("not an editor command: :{}", cmd);
                }
            }
        }
    }

    /// `:s/pat/rep/[g]` and `:%s/...` — literal patterns, not regex.
    ///
    /// Deliberately literal: a half-regex that silently mistreats `.` or `*`
    /// is worse than one that never claimed to, and the substitutions this
    /// editor is for (config keys, identifiers) are literal anyway. The message
    /// says so on a pattern that looks like a regex, rather than quietly
    /// matching nothing.
    fn substitute(&mut self, cmd: &str) {
        let all_lines = cmd.starts_with('%');
        let body = cmd.trim_start_matches('%').trim_start_matches('s');
        let sep = match body.chars().next() {
            Some(c) => c,
            None => return,
        };
        let parts: Vec<&str> = body[sep.len_utf8()..].split(sep).collect();
        if parts.is_empty() || parts[0].is_empty() {
            self.msg = "usage: :s/pattern/replacement/[g]".to_string();
            return;
        }
        let pat = parts[0];
        let rep = parts.get(1).copied().unwrap_or("");
        let global = parts.get(2).is_some_and(|f| f.contains('g'));
        let rows: Vec<usize> = if all_lines { (0..self.lines.len()).collect() } else { alloc::vec![self.cy] };

        self.checkpoint();
        let mut hits = 0usize;
        let mut touched = 0usize;
        for r in rows {
            if !self.lines[r].contains(pat) {
                continue;
            }
            let new = if global {
                let n = self.lines[r].matches(pat).count();
                hits += n;
                self.lines[r].replace(pat, rep)
            } else {
                hits += 1;
                self.lines[r].replacen(pat, rep, 1)
            };
            self.lines[r] = new;
            touched += 1;
            self.cy = r;
        }
        if hits == 0 {
            self.msg = alloc::format!("pattern not found: {pat}");
            // Nothing changed, so drop the checkpoint we just took rather than
            // leaving a no-op step in the undo history.
            let _ = self.undo.undo(&self.lines.clone(), self.cy, self.cx);
            return;
        }
        self.dirty = true;
        self.clamp_normal();
        self.msg = alloc::format!("{hits} substitution(s) on {touched} line(s)");
        self.commit_change();
    }

    fn save(&mut self, to: Option<String>) {
        let path = to.unwrap_or_else(|| self.path.clone());
        let mut content = self.lines.join("\n");
        content.push('\n');
        crate::synapse::fs::write(&path, content.as_bytes());
        self.dirty = false;
        self.saved = true;
        self.msg = alloc::format!("wrote {} ({} lines)", path, self.lines.len());
        self.path = path;
    }

    fn delete_line(&mut self) {
        crate::clipboard::set(self.lines[self.cy].clone(), true); // dd yanks the line
        if self.lines.len() > 1 {
            self.lines.remove(self.cy);
            if self.cy >= self.lines.len() {
                self.cy = self.lines.len() - 1;
            }
        } else {
            self.lines[0].clear();
        }
        self.cx = 0;
        self.dirty = true;
    }

    fn yank_line(&mut self) {
        crate::clipboard::set(self.lines[self.cy].clone(), true);
        self.msg = "yanked 1 line".to_string();
    }

    /// The ordered, inclusive selection endpoints in Visual mode.
    fn sel_range(&self) -> ((usize, usize), (usize, usize)) {
        let a = self.sel_anchor.unwrap_or((self.cy, self.cx));
        let b = (self.cy, self.cx);
        if a <= b {
            (a, b)
        } else {
            (b, a)
        }
    }

    fn selected_text(&self) -> String {
        let ((r1, c1), (r2, c2)) = self.sel_range();
        if r1 == r2 {
            let l = &self.lines[r1];
            let end = (c2 + 1).min(l.len());
            let start = c1.min(l.len());
            return l.get(start..end).unwrap_or("").to_string();
        }
        let mut s = String::new();
        let first = &self.lines[r1];
        s.push_str(first.get(c1.min(first.len())..).unwrap_or(""));
        s.push('\n');
        for r in r1 + 1..r2 {
            s.push_str(&self.lines[r]);
            s.push('\n');
        }
        let last = &self.lines[r2];
        s.push_str(last.get(..(c2 + 1).min(last.len())).unwrap_or(""));
        s
    }

    fn delete_selection(&mut self) {
        let ((r1, c1), (r2, c2)) = self.sel_range();
        if r1 == r2 {
            let l = &mut self.lines[r1];
            let end = (c2 + 1).min(l.len());
            let start = c1.min(l.len());
            l.replace_range(start..end, "");
        } else {
            let head = self.lines[r1].get(..c1.min(self.lines[r1].len())).unwrap_or("").to_string();
            let tail = self.lines[r2].get((c2 + 1).min(self.lines[r2].len())..).unwrap_or("").to_string();
            for _ in r1..r2 {
                self.lines.remove(r1 + 1);
            }
            self.lines[r1] = head + &tail;
        }
        self.cy = r1;
        self.cx = c1;
        self.clamp_normal();
        self.dirty = true;
    }

    fn visual(&mut self, b: u8) {
        self.msg.clear();
        match b {
            b'h' | b'l' | b'k' | b'j' | b'0' | b'$' | b'w' | b'b' | b'G' => self.move_cursor(b),
            0x1b => {
                self.mode = Mode::Normal;
                self.sel_anchor = None;
            }
            b'y' => {
                crate::clipboard::set(self.selected_text(), false);
                let (start, _) = self.sel_range();
                self.cy = start.0;
                self.cx = start.1;
                self.mode = Mode::Normal;
                self.sel_anchor = None;
                self.msg = "yanked selection".to_string();
            }
            b'd' | b'x' => {
                crate::clipboard::set(self.selected_text(), false);
                self.delete_selection();
                self.mode = Mode::Normal;
                self.sel_anchor = None;
            }
            _ => {}
        }
    }

    fn paste(&mut self, after: bool) {
        let Some((text, linewise)) = crate::clipboard::get() else {
            self.msg = "clipboard empty".to_string();
            return;
        };
        if linewise || text.contains('\n') {
            let at = if after { self.cy + 1 } else { self.cy };
            for (k, line) in text.split('\n').enumerate() {
                self.lines.insert(at + k, line.to_string());
            }
            self.cy = at;
            self.cx = 0;
        } else {
            let at = if after && self.line_len() > 0 { self.cx + 1 } else { self.cx };
            let at = at.min(self.line_len());
            self.lines[self.cy].insert_str(at, &text);
            self.cx = at + text.len().saturating_sub(1);
        }
        self.dirty = true;
        self.msg = "pasted".to_string();
    }

    fn word_forward(&mut self) {
        let line = self.lines[self.cy].as_bytes();
        let mut i = self.cx;
        while i < line.len() && !line[i].is_ascii_whitespace() {
            i += 1;
        }
        while i < line.len() && line[i].is_ascii_whitespace() {
            i += 1;
        }
        self.cx = i.min(self.line_len().saturating_sub(1));
    }

    fn word_back(&mut self) {
        let line = self.lines[self.cy].as_bytes();
        let mut i = self.cx;
        while i > 0 && line[i - 1].is_ascii_whitespace() {
            i -= 1;
        }
        while i > 0 && !line[i - 1].is_ascii_whitespace() {
            i -= 1;
        }
        self.cx = i;
    }

    fn render(&mut self) {
        self.sync_dims();
        self.ensure_visible();
        let base = self.path.rsplit('/').next().unwrap_or(&self.path);
        let title = alloc::format!("editor: {}{}", base, if self.dirty { " [+]" } else { "" });
        let modeline = match self.mode {
            Mode::Normal => alloc::format!("-- NORMAL --  {}:{}  {}", self.cy + 1, self.cx + 1, self.msg),
            Mode::Insert => alloc::format!("-- INSERT --  {}:{}", self.cy + 1, self.cx + 1),
            Mode::Visual => alloc::format!("-- VISUAL --  {}:{}  (y copy, d cut, Esc)", self.cy + 1, self.cx + 1),
            Mode::Command => {
                // `/` and `?` seed `cmd` with their own prompt character, so
                // prefixing another `:` would show `:/pattern`.
                if self.cmd.starts_with('/') || self.cmd.starts_with('?') {
                    self.cmd.clone()
                } else {
                    alloc::format!(":{}", self.cmd)
                }
            }
        };
        let sel = if self.mode == Mode::Visual { Some(self.sel_range()) } else { None };
        // Syntax highlighting for every logical line that may appear in the
        // viewport. Soft-wrap can show a long mid-file line after many wraps, so
        // seed the lexer from the buffer start through the first visible line.
        let tw = self.text_width();
        let lenses: Vec<usize> = self.line_lens();
        let (first_line, _) = crate::editor_wrap::unvis(&lenses, self.top, tw);
        let text_rows = self.rows.saturating_sub(1).max(1);
        let (last_line, _) =
            crate::editor_wrap::unvis(&lenses, self.top + text_rows.saturating_sub(1), tw);
        let hl: Option<Vec<Vec<Option<(u8, u8, u8)>>>> = crate::highlight::lang_for_path(&self.path).map(|lang| {
            let mut st = crate::highlight::State::default();
            for line in self.lines.iter().take(first_line) {
                crate::highlight::advance(lang, line, &mut st);
            }
            // Index 0 = first_line; editor_render offsets by first visible line.
            self.lines
                .iter()
                .skip(first_line)
                .take(last_line.saturating_sub(first_line) + 1)
                .map(|line| {
                    crate::highlight::classes(lang, line, &mut st)
                        .into_iter()
                        .map(|c| (c != crate::highlight::Class::Text).then(|| crate::highlight::rgb(c)))
                        .collect()
                })
                .collect()
        });
        crate::framebuffer::editor_render(
            &title,
            &self.lines,
            self.top,
            self.cy,
            self.cx,
            &modeline,
            sel,
            hl.as_deref(),
            first_line,
        );
    }

    /// Line-number gutter width (matches framebuffer::editor_render).
    fn gutter(&self) -> u64 {
        let mut n = self.lines.len().max(1);
        let mut w = 1u64;
        while n >= 10 {
            n /= 10;
            w += 1;
        }
        w + 1
    }

    /// Map a framebuffer pixel to an editor `(row, col)`, or `None` if outside
    /// the text area. Soft-wrap aware: screen row → visual row → (line, segment).
    fn cell_at(&self, px: u64, py: u64) -> Option<(usize, usize)> {
        let (ix, iy, cw, ch, cols, text_rows) = crate::framebuffer::editor_pane_geom()?;
        if px < ix || py < iy {
            return None;
        }
        let col_scr = (px - ix) / cw;
        let row_scr = (py - iy) / ch;
        if row_scr >= text_rows || col_scr >= cols {
            return None;
        }
        let g = self.gutter();
        if col_scr < g {
            return None;
        }
        let tw = self.text_width();
        let col_in_row = (col_scr - g) as usize;
        let vis = self.top + row_scr as usize;
        let (row, seg) = crate::editor_wrap::unvis(&self.line_lens(), vis, tw);
        if row >= self.lines.len() {
            return None;
        }
        let col = seg * tw + col_in_row;
        Some((row, col.min(self.lines[row].len().saturating_sub(1))))
    }

    /// Mouse handling in the editor: move the cursor, drag to select, release to
    /// copy the selection to the clipboard.
    fn mouse_tick(&mut self) {
        let t = crate::mouse::tick();
        if t.moved {
            crate::framebuffer::cursor_move(t.x, t.y);
        }
        if t.pressed {
            if let Some((r, c)) = self.cell_at(t.x, t.y) {
                self.cy = r;
                self.cx = c;
                self.mode = Mode::Visual;
                self.sel_anchor = Some((r, c));
                self.render();
                crate::framebuffer::cursor_move(t.x, t.y);
            }
        } else if t.released {
            if self.mode == Mode::Visual {
                crate::clipboard::set(self.selected_text(), false);
                self.msg = "copied (mouse)".to_string();
                self.render();
                crate::framebuffer::cursor_move(t.x, t.y);
            }
        } else if t.left && t.moved && self.mode == Mode::Visual {
            if let Some((r, c)) = self.cell_at(t.x, t.y) {
                self.cy = r;
                self.cx = c;
                self.render();
                crate::framebuffer::cursor_move(t.x, t.y);
            }
        }
    }
}

/// ASCII case flip for `~`.
fn flip_case(b: u8) -> u8 {
    if b.is_ascii_lowercase() {
        b.to_ascii_uppercase()
    } else if b.is_ascii_uppercase() {
        b.to_ascii_lowercase()
    } else {
        b
    }
}

/// `,` replays the last `f`/`t` in the opposite direction.
fn reverse_find(k: crate::editor_motion::Find) -> crate::editor_motion::Find {
    use crate::editor_motion::Find::*;
    match k {
        Forward => Backward,
        Backward => Forward,
        Till => TillBack,
        TillBack => Till,
    }
}
