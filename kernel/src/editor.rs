//! A small **vim-like modal editor** rendered in the right pane (`/open <file>`).
//! Files are read from and written to the Synapse store ([`crate::synapse::fs`]),
//! where the configs live, so `/open /configs/core/ui.json` → edit → `:w` →
//! `/ui reload` is a full round-trip. Text is treated as ASCII (JSON/config/notes).
//!
//! Modes: Normal (motions + operators), Insert (typing), Command (`:` ex line).
//! Motions h/j/k/l/0/$/w/b/gg/G; edits i/a/o/O/x/dd; ex `:w [file] :q :wq :q!`.
//! Long lines **soft-wrap** to the pane width (and the viewport scrolls so the
//! cursor stays visible) — the previous paint clipped mid-line with no wrap or
//! horizontal scroll, so a JSON one-liner looked truncated like a broken view.
//! The editor owns the right pane for the duration (ktrace pauses drawing there)
//! and restores it on quit.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use crate::mm::Locked;

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
    sel_anchor: Option<(usize, usize)>, // (row, col) where Visual selection began
    /// Height of the **text area** in screen rows — [`crate::framebuffer::editor_dims`]
    /// has already taken the mode line off the pane. Read it through
    /// [`Editor::text_rows`]; see that method for why nothing here may subtract
    /// one from it again.
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
        sel_anchor: None,
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

    /// Screen rows of text, which is what [`Self::rows`] already holds.
    ///
    /// **The mode line is subtracted exactly once, and not here.**
    /// [`crate::framebuffer::editor_dims`] returns the pane's rows minus one; the
    /// painter ([`crate::framebuffer::editor_render`]) independently reserves the
    /// bottom row of the *pane* for the mode line, so the two agree. Taking
    /// another row off here made this view one row shorter than the one being
    /// drawn, with two symptoms that do not look related: the cursor could never
    /// reach the bottom painted row (`j` scrolled instead, so the last line of a
    /// file was unreachable), and `render` seeded the highlighter one visual row
    /// short, so that row was painted with no syntax colour. `cell_at` sizes
    /// itself from `editor_pane_geom`, which does subtract exactly once — so a
    /// mouse click could land on the row the keyboard could not reach.
    fn text_rows(&self) -> usize {
        self.rows.max(1)
    }

    /// Keep the cursor's visual row inside the viewport (soft-wrap aware).
    fn ensure_visible(&mut self) {
        let vr = self.vis_of(self.cy, self.cx);
        self.top = crate::editor_wrap::scroll_top(self.top, vr, self.text_rows());
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
                    // Page up/down by a viewport height, less one row of context
                    // — the line you were reading at the edge stays on screen,
                    // as every pager does it. Deliberate, unlike the subtraction
                    // `text_rows` documents.
                    let page = self.text_rows().saturating_sub(1).max(1);
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

    fn normal(&mut self, b: u8) {
        let pend = self.pending;
        self.pending = 0;
        self.msg.clear();
        match b {
            b'h' | b'l' | b'k' | b'j' | b'0' | b'$' | b'w' | b'b' | b'G' => self.move_cursor(b),
            b'g' if pend == b'g' => {
                self.cy = 0;
                self.cx = 0;
            }
            b'g' => self.pending = b'g',
            b'v' => {
                self.mode = Mode::Visual;
                self.sel_anchor = Some((self.cy, self.cx));
            }
            b'y' if pend == b'y' => self.yank_line(),
            b'y' => self.pending = b'y',
            b'Y' => self.yank_line(),
            b'p' => self.paste(true),
            b'P' => self.paste(false),
            b'i' => self.mode = Mode::Insert,
            b'I' => {
                self.cx = 0;
                self.mode = Mode::Insert;
            }
            b'a' => {
                if self.line_len() > 0 {
                    self.cx += 1;
                }
                self.mode = Mode::Insert;
            }
            b'A' => {
                self.cx = self.line_len();
                self.mode = Mode::Insert;
            }
            b'o' => {
                self.lines.insert(self.cy + 1, String::new());
                self.cy += 1;
                self.cx = 0;
                self.mode = Mode::Insert;
                self.dirty = true;
            }
            b'O' => {
                self.lines.insert(self.cy, String::new());
                self.cx = 0;
                self.mode = Mode::Insert;
                self.dirty = true;
            }
            b'x' => {
                if self.cx < self.line_len() {
                    self.lines[self.cy].remove(self.cx);
                    self.clamp_normal();
                    self.dirty = true;
                }
            }
            b'd' if pend == b'd' => self.delete_line(),
            b'd' => self.pending = b'd',
            b':' => {
                self.mode = Mode::Command;
                self.cmd.clear();
            }
            _ => {}
        }
    }

    fn insert(&mut self, b: u8) {
        match b {
            0x1b => {
                self.mode = Mode::Normal;
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
        let cmd = self.cmd.trim();
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
            Mode::Command => alloc::format!(":{}", self.cmd),
        };
        let sel = if self.mode == Mode::Visual { Some(self.sel_range()) } else { None };
        // Syntax highlighting for every logical line that may appear in the
        // viewport. Soft-wrap can show a long mid-file line after many wraps, so
        // seed the lexer from the buffer start through the first visible line.
        let tw = self.text_width();
        let lenses: Vec<usize> = self.line_lens();
        let (first_line, _) = crate::editor_wrap::unvis(&lenses, self.top, tw);
        let text_rows = self.text_rows();
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
