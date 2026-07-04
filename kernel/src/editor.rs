//! A small **vim-like modal editor** rendered in the right pane (`/open <file>`).
//! Files are read from and written to the Synapse store ([`crate::synapse::fs`]),
//! where the configs live, so `/open /configs/core/ui.json` → edit → `:w` →
//! `/ui reload` is a full round-trip. Text is treated as ASCII (JSON/config/notes).
//!
//! Modes: Normal (motions + operators), Insert (typing), Command (`:` ex line).
//! Motions h/j/k/l/0/$/w/b/gg/G; edits i/a/o/O/x/dd; ex `:w [file] :q :wq :q!`.
//! The editor owns the right pane for the duration (ktrace pauses drawing there)
//! and restores it on quit.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

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
    cx: usize, // column (byte == char, ASCII)
    cy: usize, // row
    top: usize,
    mode: Mode,
    cmd: String,
    msg: String,
    dirty: bool,
    saved: bool,
    quit: bool,
    pending: u8, // pending operator: b'd', b'g', b'y', or 0
    sel_anchor: Option<(usize, usize)>, // (row, col) where Visual selection began
    rows: usize,
}

/// Open `path` in the editor. Returns `true` if the buffer was written at least
/// once (so the caller can reload config, etc.). Blocks until `:q`/`:wq`.
pub fn open(path: &str) -> bool {
    let content = crate::synapse::fs::read(path).and_then(|b| String::from_utf8(b).ok()).unwrap_or_default();
    let mut lines: Vec<String> =
        content.split('\n').map(|s| s.trim_end_matches('\r').to_string()).collect();
    if lines.is_empty() {
        lines.push(String::new());
    }
    // A trailing newline yields a spurious empty last line; drop it for editing.
    if lines.len() > 1 && lines.last().map(|l| l.is_empty()).unwrap_or(false) {
        lines.pop();
    }
    // Split the action pane first, then measure it.
    crate::framebuffer::editor_enter();
    let (_cols, rows) = crate::framebuffer::editor_dims().unwrap_or((60, 24));
    let exists = crate::synapse::fs::exists(path);
    let mut ed = Editor {
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
    };
    ed.render();
    while !ed.quit {
        match crate::console::read_byte() {
            Some(b) => {
                ed.handle(b);
                ed.render();
                crate::framebuffer::cursor_move_here();
            }
            None => {
                ed.mouse_tick();
                crate::sched::yield_now();
            }
        }
    }
    crate::framebuffer::editor_leave();
    ed.saved
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

    fn handle(&mut self, b: u8) {
        match self.mode {
            Mode::Normal => self.normal(b),
            Mode::Insert => self.insert(b),
            Mode::Command => self.command(b),
            Mode::Visual => self.visual(b),
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
        // Keep the cursor within the viewport.
        if self.cy < self.top {
            self.top = self.cy;
        } else if self.cy >= self.top + self.rows {
            self.top = self.cy - self.rows + 1;
        }
        let base = self.path.rsplit('/').next().unwrap_or(&self.path);
        let title = alloc::format!("editor: {}{}", base, if self.dirty { " [+]" } else { "" });
        let modeline = match self.mode {
            Mode::Normal => alloc::format!("-- NORMAL --  {}:{}  {}", self.cy + 1, self.cx + 1, self.msg),
            Mode::Insert => alloc::format!("-- INSERT --  {}:{}", self.cy + 1, self.cx + 1),
            Mode::Visual => alloc::format!("-- VISUAL --  {}:{}  (y copy, d cut, Esc)", self.cy + 1, self.cx + 1),
            Mode::Command => alloc::format!(":{}", self.cmd),
        };
        let sel = if self.mode == Mode::Visual { Some(self.sel_range()) } else { None };
        crate::framebuffer::editor_render(&title, &self.lines, self.top, self.cy, self.cx, &modeline, sel);
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
    /// the text area.
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
        let col = col_scr.saturating_sub(g) as usize;
        let row = self.top + row_scr as usize;
        if row >= self.lines.len() {
            return None;
        }
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
