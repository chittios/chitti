//! Framebuffer compositor (`CHITTI_OS_HANDOFF.md` Phase 7 stretch: "framebuffer
//! text UI beyond serial"). A tmux-style split-pane terminal drawn directly on
//! the framebuffer: two bordered panes side by side -- **chat** (left, the
//! interactive REPL) and **logs** (right, the live ktrace stream) -- an
//! active-pane highlight, and a bottom status bar. Text is rendered with the
//! Geist Mono glyph atlas ([`crate::font_geist`]) alpha-blended per pixel, so
//! the panes show antialiased type rather than a bare bitmap grid.
//!
//! The framebuffer geometry and pixel format are always taken from the boot
//! source -- the Limine framebuffer (x86), the UEFI GOP handed over by the stub
//! (aarch64 real hardware / VirtualBox / UTM), or QEMU ramfb -- never hardcoded.
//! On a high-resolution panel (a 4K HDMI monitor) the 10x22 atlas cell would be
//! microscopic, so the console picks an integer font `scale` from the panel
//! height: text stays legible while the panes still fill the whole screen.
//!
//! It is a global singleton ([`SCREEN`]) rather than a transient writer because
//! the two log channels mirror here automatically: `serial::Serial` (every
//! `serial_print!`/`serial_println!`, i.e. the shell + chat) draws into the
//! chat pane via [`console_print`], while `ktrace` draws into the logs pane via
//! [`log_print`]. Keyboard input (`arch::keyboard`) plus this output is what
//! makes the framebuffer a real console, not just a log mirror.

use crate::font_geist::{CELL_H as CH, CELL_W as CW, FIRST, GLYPHS, LAST};
use crate::limine_protocol::Framebuffer;
use crate::mm::Locked;
use alloc::collections::VecDeque;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

const CELL_W: u64 = CW as u64;
const CELL_H: u64 = CH as u64;

type Rgb = (u8, u8, u8);

/// One character cell of a pane's text grid: the byte drawn (0 = empty) and its
/// colour at draw time. The grid (plus the scrollback ring behind it) is the
/// source of truth for a pane's text, so content survives redraws, relayouts,
/// and modal dismissals, and can be scrolled back through.
type Cell = (u8, Rgb);

/// Scrollback depth per pane, in lines. At 200 cols a full ring is ~3 MB —
/// noise next to the model heap. Cleared only by `/clear`.
const HIST_MAX: usize = 2000;

/// The console colour palette (see [DESIGN.md](../../DESIGN.md)). Every colour is
/// a field so the whole theme is configurable from `/configs/core/ui.json`
/// (`theme` object, hex strings); the default is the Chitti brand **dark**
/// theme — terracotta `#cc785c` primary on warm-ink surfaces, cream text.
#[derive(Clone, Copy)]
pub struct Theme {
    pub screen_bg: Rgb,
    pub chat_bg: Rgb,
    pub logs_bg: Rgb,
    pub chat_fg: Rgb,
    pub logs_fg: Rgb,
    pub accent: Rgb, // active border / caret / brand / logo
    pub border_dim: Rgb,
    pub title_active: Rgb,
    pub title_dim: Rgb,
    pub sep_dim: Rgb,
    pub status_bg: Rgb,
    pub status_fg: Rgb,
    pub editor_bg: Rgb,
    pub editor_fg: Rgb,
    pub editor_lineno: Rgb,
    pub editor_sel: Rgb,
    /// Grok-style input composer fill (slightly elevated over chat_bg).
    pub composer_bg: Rgb,
    /// Composer border when idle / focused (focused uses `accent`).
    pub composer_border: Rgb,
    /// Hint-bar text under the composer.
    pub composer_hint: Rgb,
}

impl Theme {
    /// The Chitti brand dark theme: `#cc785c` terracotta on warm-ink surfaces,
    /// cream (`#faf9f5`) text.
    pub const BRAND_DARK: Theme = Theme {
        screen_bg: (24, 23, 21),       // surface-dark #181715
        chat_bg: (31, 30, 27),         // surface-dark-soft #1f1e1b
        logs_bg: (20, 19, 17),         // a touch darker than the chat pane
        chat_fg: (250, 249, 245),      // on-dark / cream #faf9f5
        logs_fg: (160, 157, 150),      // on-dark-soft #a09d96
        accent: (204, 120, 92),        // primary #cc785c
        border_dim: (58, 55, 51),      // inactive border
        title_active: (204, 120, 92),  // primary
        title_dim: (108, 106, 100),    // muted #6c6a64
        sep_dim: (42, 40, 37),
        status_bg: (37, 35, 32),       // surface-dark-elevated #252320
        status_fg: (160, 157, 150),    // on-dark-soft
        editor_bg: (31, 30, 27),       // surface-dark-soft
        editor_fg: (250, 249, 245),    // cream
        editor_lineno: (108, 106, 100),
        editor_sel: (90, 58, 46),      // terracotta-tinted selection
        composer_bg: (37, 35, 32),     // elevated like status_bg
        composer_border: (58, 55, 51), // matches border_dim when unfocused
        composer_hint: (108, 106, 100), // muted
    };
}

impl Default for Theme {
    fn default() -> Self {
        Theme::BRAND_DARK
    }
}

/// Parse a `#rrggbb` (or `rrggbb`) hex colour, falling back to `def`.
pub fn parse_hex(s: &str, def: Rgb) -> Rgb {
    let h = s.trim().trim_start_matches('#');
    if h.len() != 6 {
        return def;
    }
    let b = |i: usize| u8::from_str_radix(&h[i..i + 2], 16);
    match (b(0), b(2), b(4)) {
        (Ok(r), Ok(g), Ok(bl)) => (r, g, bl),
        _ => def,
    }
}

/// Build a [`Theme`] from `(name, "#rrggbb")` pairs, starting from the brand dark
/// default and overriding any named field. Unknown names are ignored; malformed
/// hex keeps the brand value. This is how `ui.json`'s `theme` object is applied.
pub fn theme_from_pairs(pairs: &[(alloc::string::String, alloc::string::String)]) -> Theme {
    let mut t = Theme::BRAND_DARK;
    for (name, hex) in pairs {
        let slot = match name.as_str() {
            "screen_bg" => &mut t.screen_bg,
            "chat_bg" => &mut t.chat_bg,
            "logs_bg" => &mut t.logs_bg,
            "chat_fg" => &mut t.chat_fg,
            "logs_fg" => &mut t.logs_fg,
            "accent" => &mut t.accent,
            "border_dim" => &mut t.border_dim,
            "title_active" => &mut t.title_active,
            "title_dim" => &mut t.title_dim,
            "sep_dim" => &mut t.sep_dim,
            "status_bg" => &mut t.status_bg,
            "status_fg" => &mut t.status_fg,
            "editor_bg" => &mut t.editor_bg,
            "editor_fg" => &mut t.editor_fg,
            "editor_lineno" => &mut t.editor_lineno,
            "editor_sel" => &mut t.editor_sel,
            "composer_bg" => &mut t.composer_bg,
            "composer_border" => &mut t.composer_border,
            "composer_hint" => &mut t.composer_hint,
            _ => continue,
        };
        *slot = parse_hex(hex, *slot);
    }
    t
}

// Layout metrics, in pixels (independent of font scale).
const OUTER: u64 = 8; // margin around the whole content region
const GAP: u64 = 10; // between the two panes
const BORDER: u64 = 2; // pane border thickness
const PAD: u64 = 10; // interior padding inside a pane
const CHAT_PCT: u64 = 56; // chat pane width as a % of the content region
/// Vertical padding inside the Grok-style input composer box (px, unscaled).
const COMPOSER_VPAD: u64 = 6;
/// Gap between the composer box and the hint row under it (px, unscaled).
const COMPOSER_HINT_GAP: u64 = 4;
/// Margin between chat scrollback and the composer box (px, unscaled).
const COMPOSER_TOP_GAP: u64 = 8;

/// Pick an integer font scale from the panel height so glyphs stay a legible
/// physical size across resolutions: 1x up to ~1600 tall, 2x for 4K-class
/// panels, 3x beyond. Normal 1080p/1440p monitors render the crisp native
/// 10x22 atlas; a 4K HDMI display doubles it instead of showing 6px text.
fn pick_scale(height: u64) -> u64 {
    ((height + 550) / 1100).max(1)
}

/// One bordered text pane: an outer box plus the interior character grid it
/// scrolls text within. Colours, cursor, and the (scaled) cell size live here;
/// the pixel plumbing lives on [`Screen`], which owns the framebuffer.
struct Pane {
    // Outer box (border-inclusive), pixels.
    x: u64,
    y: u64,
    w: u64,
    h: u64,
    // Interior text origin (top-left of cell 0,0), pixels.
    ix: u64,
    iy: u64,
    // Scaled cell size, pixels.
    cw: u64,
    ch: u64,
    // Interior size, cells.
    cols: u64,
    rows: u64,
    // Cursor, cells.
    col: u64,
    row: u64,
    // `fg` is the *current* text colour (mutated by ANSI SGR codes in the byte
    // stream); `default_fg` is what a reset (`\x1b[0m`/`\x1b[39m`) restores.
    fg: Rgb,
    default_fg: Rgb,
    bg: Rgb,
    // ANSI escape-sequence parser state (see `pane_putc`).
    esc: EscState,
    csi: [u8; 32],
    csi_len: usize,
    bold: bool,
    title: String,
    show_caret: bool,
    /// The live character grid (`cols * rows` cells) mirroring what is drawn.
    grid: Vec<Cell>,
    /// Scrollback: lines evicted off the top of the grid, oldest first.
    hist: VecDeque<Vec<Cell>>,
    /// Scrollback view offset in lines back from live (0 = live). While > 0,
    /// incoming bytes still update the grid/hist but pixels are frozen on the
    /// scrolled view; the offset auto-advances so the view stays anchored.
    view: usize,
    /// Mouse text selection `(anchor, head)`, both inclusive `(line, col)` in
    /// **absolute** coordinates over `hist` + grid (see `crate::textsel`), so
    /// it stays glued to its text while the pane scrolls. `None` = no selection.
    sel: Option<((usize, usize), (usize, usize))>,
    /// When true, the pane reserves its bottom for a Grok-style input composer
    /// (bordered box + hint row); the scrollback grid sits above it.
    has_composer: bool,
}

/// Minimal ANSI escape-sequence parser state for a pane's byte stream: we
/// recognise `ESC [ … <final>` (CSI) and honour SGR (`… m`) colour/emphasis so
/// the shell agent can format replies with [ANSI codes]. Other CSI sequences
/// (cursor moves, erase) are consumed and ignored — enough to render coloured
/// text without a full terminal emulator.
#[derive(Clone, Copy, PartialEq)]
enum EscState {
    Ground,
    Esc,
    Csi,
}

impl Pane {
    /// Build a pane inside outer box `(x,y,w,h)` with scaled cell `(cw,ch)`,
    /// reserving a title header and `PAD` interior padding, then computing the
    /// cell grid.
    #[allow(clippy::too_many_arguments)]
    fn new(x: u64, y: u64, w: u64, h: u64, cw: u64, ch: u64, fg: Rgb, bg: Rgb, title: String, show_caret: bool) -> Pane {
        let header_h = BORDER + 4 + ch + 6; // top border, title text, separator gap
        let ix = x + BORDER + PAD;
        let iy = y + header_h;
        let iw = w.saturating_sub(2 * (BORDER + PAD)).max(cw);
        // Grok-style composer: box (vpad + 1 line + vpad + 2px border) + gap + hint line.
        // Reserve it so scrollback never paints under the input chrome.
        let has_composer = show_caret;
        let composer_block = if has_composer {
            COMPOSER_TOP_GAP + (COMPOSER_VPAD + ch + COMPOSER_VPAD + 2) + COMPOSER_HINT_GAP + ch
        } else {
            0
        };
        let ih = (y + h).saturating_sub(iy + BORDER + PAD + composer_block).max(ch);
        let cols = (iw / cw).max(1);
        let rows = (ih / ch).max(1);
        Pane {
            x,
            y,
            w,
            h,
            ix,
            iy,
            cw,
            ch,
            cols,
            rows,
            col: 0,
            row: 0,
            fg,
            default_fg: fg,
            bg,
            esc: EscState::Ground,
            csi: [0; 32],
            csi_len: 0,
            bold: false,
            title,
            show_caret,
            grid: alloc::vec![(0u8, fg); (cols * rows) as usize],
            hist: VecDeque::new(),
            view: 0,
            sel: None,
            has_composer,
        }
    }

    /// Write `byte` into the grid cell under the cursor (0 erases).
    fn set_cell(&mut self, byte: u8) {
        let idx = (self.row * self.cols + self.col) as usize;
        if let Some(c) = self.grid.get_mut(idx) {
            *c = (byte, self.fg);
        }
    }

    /// First numeric CSI parameter (0 if absent) — enough for `ESC[nC`/`nD`/`nK`.
    fn csi_param(&self) -> u64 {
        let mut v: u64 = 0;
        for &b in &self.csi[..self.csi_len] {
            if b.is_ascii_digit() {
                v = v.saturating_mul(10) + (b - b'0') as u64;
            } else {
                break;
            }
        }
        v
    }

    /// Clone text state (scrollback + grid + cursor + colour) from `old` without
    /// reflowing. Used when this pane is **parked** off-screen during fullscreen
    /// (`w == 0`): reflowing a multi-thousand-line history into the 1-column ghost
    /// grid `Pane::new` builds for a zero-width box would allocate/hang the OS
    /// (Ctrl+F hang). The parked pane keeps its native `cols`/`rows` so a later
    /// unpark can reflow correctly into the restored geometry.
    fn take_content(&mut self, old: &Pane) {
        self.hist = old.hist.clone();
        self.grid = old.grid.clone();
        self.cols = old.cols;
        self.rows = old.rows;
        self.col = old.col.min(old.cols.saturating_sub(1));
        self.row = old.row.min(old.rows.saturating_sub(1));
        self.view = old.view.min(self.hist.len());
        self.sel = None;
        self.fg = old.fg;
        self.default_fg = old.default_fg;
        self.bold = old.bold;
        self.esc = old.esc;
        self.csi = old.csi;
        self.csi_len = old.csi_len;
    }

    /// Carry another pane's text (scrollback + grid + cursor + colour state)
    /// into this freshly-built pane, **reflowing** soft-wrapped lines to the
    /// new column count. Used when the layout is rebuilt (divider drag, action
    /// pane toggle, `/pane split`) so expanding the chat pane fills the extra
    /// width instead of leaving short lines stranded on the left.
    ///
    /// Parked destinations (`self.w == 0`, fullscreen) skip reflow entirely —
    /// see [`Self::take_content`].
    fn adopt(&mut self, old: &Pane) {
        if old.grid.is_empty() && old.hist.is_empty() {
            return;
        }
        // Fullscreen parks a pane at outer width 0 (off-screen). Never reflow
        // into the 1-col placeholder grid — that turns ~2000×N cells into
        // millions of 1-cell rows and freezes the cooperative kernel.
        if self.w == 0 {
            self.take_content(old);
            return;
        }
        if self.grid.is_empty() {
            return;
        }
        let ocols = old.cols as usize;
        let cols = self.cols as usize;
        let rows = self.rows as usize;
        // Same width: transplant without soft-reflow (row count may still change).
        if ocols == cols && !old.grid.is_empty() {
            let empty: Cell = (0, self.default_fg);
            let mut abs: alloc::vec::Vec<alloc::vec::Vec<Cell>> = old.hist.iter().cloned().collect();
            let used = ((old.row + 1).min(old.rows)) as usize;
            for r in 0..used {
                let start = r * ocols;
                let end = (start + ocols).min(old.grid.len());
                if start < end {
                    abs.push(old.grid[start..end].to_vec());
                }
            }
            let total = abs.len();
            let keep = total.min(rows);
            let start = total - keep;
            for c in self.grid.iter_mut() {
                *c = empty;
            }
            for (r, line) in abs.iter().skip(start).enumerate() {
                let n = line.len().min(cols);
                for c in 0..n {
                    self.grid[r * cols + c] = line[c];
                }
            }
            self.hist = abs.into_iter().take(start).collect();
            while self.hist.len() > HIST_MAX {
                self.hist.pop_front();
            }
            let old_line = old.hist.len() + old.row.min(old.rows.saturating_sub(1)) as usize;
            self.row = if old_line >= start {
                (old_line - start).min(rows.saturating_sub(1)) as u64
            } else {
                0
            };
            self.col = old.col.min(self.cols.saturating_sub(1));
            self.view = 0;
            self.sel = None;
            self.fg = old.fg;
            self.default_fg = old.default_fg;
            self.bold = old.bold;
            return;
        }
        // Absolute lines: scrollback then live grid rows that hold content.
        let mut abs: alloc::vec::Vec<alloc::vec::Vec<Cell>> = old.hist.iter().cloned().collect();
        let used = ((old.row + 1).min(old.rows)) as usize;
        for r in 0..used {
            let start = r * ocols;
            let end = (start + ocols).min(old.grid.len());
            if start < end {
                abs.push(old.grid[start..end].to_vec());
            }
        }
        let old_line = old.hist.len() + old.row.min(old.rows.saturating_sub(1)) as usize;
        let empty: Cell = (0, self.default_fg);
        // Same layout as textsel::Cell — Rgb is (u8,u8,u8).
        let as_ts: alloc::vec::Vec<alloc::vec::Vec<crate::textsel::Cell>> =
            abs.iter().map(|l| l.iter().map(|&(b, c)| (b, c)).collect()).collect();
        let reflowed = crate::textsel::reflow_lines(&as_ts, ocols, cols, (0, self.default_fg));
        let (new_line, new_col) =
            crate::textsel::reflow_cursor(&as_ts, ocols, cols, old_line, old.col as usize);
        // Place the tail of the reflow into the live grid; the rest is hist.
        let total = reflowed.len();
        let keep = total.min(rows);
        let start = total - keep;
        // Clear grid first so expanded columns aren't stale.
        for c in self.grid.iter_mut() {
            *c = empty;
        }
        for (r, line) in reflowed.iter().skip(start).enumerate() {
            let n = line.len().min(cols);
            for c in 0..n {
                self.grid[r * cols + c] = line[c];
            }
        }
        self.hist = reflowed.into_iter().take(start).collect();
        while self.hist.len() > HIST_MAX {
            self.hist.pop_front();
        }
        self.row = if new_line >= start {
            (new_line - start).min(rows.saturating_sub(1)) as u64
        } else {
            0
        };
        self.col = (new_col as u64).min(self.cols.saturating_sub(1));
        self.view = 0;
        self.sel = None; // absolute coords are invalid after reflow
        self.fg = old.fg;
        self.default_fg = old.default_fg;
        self.bold = old.bold;
    }

    /// Drop all text content (grid + scrollback) — the `/clear` reset.
    fn clear_content(&mut self) {
        for c in self.grid.iter_mut() {
            *c = (0, self.default_fg);
        }
        self.hist.clear();
        self.view = 0;
        self.sel = None;
        self.col = 0;
        self.row = 0;
        self.fg = self.default_fg;
        self.bold = false;
    }
    fn cell_x(&self) -> u64 {
        self.ix + self.col * self.cw
    }
    fn cell_y(&self) -> u64 {
        self.iy + self.row * self.ch
    }

    /// Apply the buffered CSI `… m` (SGR) parameters to this pane's colour state.
    /// Supports reset (0), bold (1/22), default fg (39), the 8 normal (30–37) and
    /// bright (90–97) foreground colours, and 24-bit / 256-colour `38;2;r;g;b` /
    /// `38;5;n`. Background and other attributes are ignored.
    fn apply_sgr(&mut self) {
        // Parse the `;`-separated numeric params (empty => 0).
        let mut params = [0i32; 16];
        let mut np = 0usize;
        let (mut cur, mut has) = (0i32, false);
        for &b in &self.csi[..self.csi_len] {
            if b == b';' {
                if np < params.len() {
                    params[np] = cur;
                    np += 1;
                }
                cur = 0;
                has = false;
            } else if b.is_ascii_digit() {
                cur = cur.saturating_mul(10) + (b - b'0') as i32;
                has = true;
            }
        }
        if np < params.len() {
            params[np] = cur;
            np += 1;
        }
        let _ = has;
        let mut i = 0;
        while i < np {
            match params[i] {
                0 => {
                    self.fg = self.default_fg;
                    self.bold = false;
                }
                1 => self.bold = true,
                22 => self.bold = false,
                39 => self.fg = self.default_fg,
                30..=37 => self.fg = ansi_color((params[i] - 30) as usize, self.bold),
                90..=97 => self.fg = ansi_color((params[i] - 90) as usize, true),
                38 => {
                    if i + 4 < np && params[i + 1] == 2 {
                        self.fg = (params[i + 2] as u8, params[i + 3] as u8, params[i + 4] as u8);
                        i += 4;
                    } else if i + 2 < np && params[i + 1] == 5 {
                        self.fg = ansi_256(params[i + 2] as u8);
                        i += 2;
                    }
                }
                _ => {}
            }
            i += 1;
        }
    }
}

/// The 8 ANSI foreground colours (and their bright variants), tuned to read well
/// on the dark pane background.
fn ansi_color(idx: usize, bright: bool) -> Rgb {
    const NORMAL: [Rgb; 8] = [
        (98, 104, 118),  // "black" -> dim gray (pure black is invisible here)
        (255, 106, 110), // red
        (126, 214, 150), // green
        (240, 200, 120), // yellow
        (94, 161, 255),  // blue (the accent)
        (200, 140, 255), // magenta
        (110, 214, 224), // cyan
        (232, 233, 238), // white (the default fg)
    ];
    const BRIGHT: [Rgb; 8] = [
        (140, 148, 162),
        (255, 140, 150),
        (170, 240, 190),
        (255, 224, 150),
        (150, 190, 255),
        (220, 170, 255),
        (150, 235, 245),
        (255, 255, 255),
    ];
    (if bright { &BRIGHT } else { &NORMAL })[idx & 7]
}

/// Map an ANSI 256-colour index to RGB: 0–15 the base/bright palette, 16–231 the
/// 6×6×6 colour cube, 232–255 the 24-step grayscale ramp.
fn ansi_256(n: u8) -> Rgb {
    match n {
        0..=7 => ansi_color(n as usize, false),
        8..=15 => ansi_color((n - 8) as usize, true),
        16..=231 => {
            let c = n - 16;
            let steps = [0u8, 95, 135, 175, 215, 255];
            (steps[(c / 36) as usize], steps[((c / 6) % 6) as usize], steps[(c % 6) as usize])
        }
        _ => {
            let v = 8 + (n - 232) * 10;
            (v, v, v)
        }
    }
}

/// The compositor: framebuffer geometry + the two panes it draws.
pub struct Screen {
    addr: usize,
    width: u64,
    height: u64,
    pitch: u64,
    bpp_bytes: u64,
    r_shift: u32,
    g_shift: u32,
    b_shift: u32,
    scale: u64,
    chat: Pane,
    logs: Pane,
    /// Status-bar text (left = brand, right = datetime); set by the shell from
    /// the UI-config templates + clock, so it stays configurable.
    status_left: String,
    status_right: String,
    /// Blinking-caret state for the chat pane / composer.
    caret_on: bool,
    caret_last_ms: u64,
    /// Grok-style input composer (bottom of chat pane): when `composer_active`
    /// the caret lives in the bordered box, not the scrollback grid.
    composer_active: bool,
    composer_line: String,
    composer_cur: usize,
    /// Hint bar under the composer (left / right halves).
    composer_hint_l: String,
    composer_hint_r: String,
    /// Slash-command / @file suggestion popup above the composer.
    suggest_open: bool,
    suggest_items: alloc::vec::Vec<(String, String)>, // (label, detail)
    suggest_sel: usize,
    /// Last painted popup rect `(x, y, w, h)` — used to erase the dirty region
    /// cleanly (includes the gap above the composer that the chat grid does
    /// not cover).
    suggest_rect: Option<(u64, u64, u64, u64)>,
    /// Fallback blink cadence when the monotonic clock is frozen (some VBox
    /// configs): the last `now_ms` seen, and a call counter. `clock_alive`
    /// latches once `now_ms` is ever seen advancing — after that the fallback
    /// is never used (on a fast host thousands of calls land in the same
    /// millisecond, which used to trip the counter and blink far too fast).
    blink_seen_ms: u64,
    blink_calls: u32,
    clock_alive: bool,
    /// Whether keyboard focus is on the action (right) pane. Only meaningful
    /// while the action pane shows ktrace; the editor always owns focus.
    focus_action: bool,
    /// What the active tab in the right ("action") pane shows. Mirrors
    /// `tabs[active]` (or `Closed` when `tabs` is empty). Kept as a field so the
    /// many `self.right == …` readers stay valid.
    right: RightMode,
    /// The open action-pane tabs, tmux-style: opening a view adds/selects a tab,
    /// switching keeps every other tab's process alive (audio keeps playing,
    /// ktrace keeps streaming, the editor keeps its buffer). Empty = pane closed.
    tabs: Vec<RightMode>,
    /// Index of the active tab within `tabs`.
    active: usize,
    /// Unused since the action pane became tabbed (kept for struct stability).
    right_before_editor: RightMode,
    /// The last-applied layout config, reused when opening/closing the action
    /// pane so the split ratio / titles / scale are preserved.
    layout: LayoutCfg,
    /// The active colour palette (from `layout.theme`).
    theme: Theme,
    /// Mouse cursor sprite state: position + the framebuffer patch saved beneath
    /// it (restored before each move so the cursor leaves no trail).
    cur_x: u64,
    cur_y: u64,
    cur_vis: bool,
    /// True once the mouse has moved, so content redraws keep re-drawing the
    /// cursor on top instead of leaving it erased.
    cur_active: bool,
    cur_saved: [Rgb; (CUR_W * CUR_H) as usize],
}

// Mouse cursor arrow (macOS-style: black fill, white outline so it reads on
// both dark and light content): 0 = transparent, 1 = fill, 2 = outline.
const CUR_W: u64 = 12;
const CUR_H: u64 = 19;
#[rustfmt::skip]
const CURSOR: [u8; (CUR_W * CUR_H) as usize] = [
    2,0,0,0,0,0,0,0,0,0,0,0,
    2,2,0,0,0,0,0,0,0,0,0,0,
    2,1,2,0,0,0,0,0,0,0,0,0,
    2,1,1,2,0,0,0,0,0,0,0,0,
    2,1,1,1,2,0,0,0,0,0,0,0,
    2,1,1,1,1,2,0,0,0,0,0,0,
    2,1,1,1,1,1,2,0,0,0,0,0,
    2,1,1,1,1,1,1,2,0,0,0,0,
    2,1,1,1,1,1,1,1,2,0,0,0,
    2,1,1,1,1,1,1,1,1,2,0,0,
    2,1,1,1,1,1,1,1,1,1,2,0,
    2,1,1,1,1,1,1,1,1,1,1,2,
    2,1,1,1,1,1,2,2,2,2,2,2,
    2,1,1,2,1,1,2,0,0,0,0,0,
    2,1,2,0,2,1,1,2,0,0,0,0,
    2,2,0,0,2,1,1,2,0,0,0,0,
    2,0,0,0,0,2,1,1,2,0,0,0,
    0,0,0,0,0,2,1,1,2,0,0,0,
    0,0,0,0,0,0,2,2,0,0,0,0,
];

/// What the right ("action") pane shows.
#[derive(Clone, Copy, PartialEq)]
pub enum RightMode {
    /// Closed: chat pane is full-width (the default).
    Closed,
    /// The live ktrace log stream.
    Ktrace,
    /// The `/open` editor.
    Editor,
    /// The live `/top` system dashboard (CPU + memory).
    Top,
    /// Live session todo list (`/todos open`).
    Todos,
    /// The `/open <file>.wav|.mp3` background audio player.
    Audio,
    /// An agent-owned drawing surface (`synapse::ui`), by surface id.
    Surface(u32),
}

/// Surface id the `/open` image viewer uses (also known to the shell). A
/// `Surface(IMAGE_SURFACE)` tab is labelled "image" in the tab bar.
pub const IMAGE_SURFACE: u32 = u32::MAX;
/// Surface id the `/open` video player presents frames on (labelled "video").
pub const VIDEO_SURFACE: u32 = u32::MAX - 1;

/// The short tab-bar label for a view.
fn tab_label(m: RightMode) -> &'static str {
    match m {
        RightMode::Closed => "",
        RightMode::Ktrace => "ktrace",
        RightMode::Editor => "editor",
        RightMode::Top => "top",
        RightMode::Todos => "todos",
        RightMode::Audio => "audio",
        RightMode::Surface(IMAGE_SURFACE) => "image",
        RightMode::Surface(VIDEO_SURFACE) => "video",
        RightMode::Surface(_) => "surface",
    }
}

/// Config knobs the UI config (`/configs/core/ui.json`) can set for the layout.
#[derive(Clone)]
pub struct LayoutCfg {
    /// Chat pane width as a % of the content region (10..90).
    pub chat_pct: u64,
    /// Font scale; 0 = auto from panel height.
    pub scale: u64,
    /// Put the chat pane on the right instead of the left.
    pub swap: bool,
    pub chat_title: String,
    pub logs_title: String,
    /// Colour palette (from `ui.json` `theme`; default = brand dark).
    pub theme: Theme,
    /// Show the boot splash (logo + name). Default true.
    pub splash: bool,
    /// Fullscreen state: 0 = normal split, 1 = chat fills the screen, 2 = the
    /// action pane fills the screen. Toggled at runtime (F11 / `/fullscreen`).
    pub fullscreen: u8,
}

impl Default for LayoutCfg {
    fn default() -> Self {
        LayoutCfg {
            chat_pct: CHAT_PCT,
            scale: 0,
            swap: false,
            chat_title: String::from("Shell Agent"),
            logs_title: String::from("ktrace"),
            theme: Theme::default(),
            splash: true,
            fullscreen: 0,
        }
    }
}

impl Screen {
    #[allow(clippy::too_many_arguments)]
    fn layout(
        addr: usize,
        width: u64,
        height: u64,
        pitch: u64,
        bpp_bytes: u64,
        r_shift: u32,
        g_shift: u32,
        b_shift: u32,
    ) -> Screen {
        // Default boot layout: the action pane is closed, so the chat pane is
        // full-width (only the shell/chat shows until `/ktrace` or `/open`). The
        // theme + splash come from the UI config (brand defaults until it loads).
        Screen::build(addr, width, height, pitch, bpp_bytes, r_shift, g_shift, b_shift, &crate::ui_config::boot_layout(), false)
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        addr: usize,
        width: u64,
        height: u64,
        pitch: u64,
        bpp_bytes: u64,
        r_shift: u32,
        g_shift: u32,
        b_shift: u32,
        cfg: &LayoutCfg,
        split: bool,
    ) -> Screen {
        let scale = if cfg.scale > 0 { cfg.scale } else { pick_scale(height) };
        let cw = CELL_W * scale;
        let ch = CELL_H * scale;
        let status_h = ch + 8;
        let content_h = height.saturating_sub(status_h);
        let box_y = OUTER;
        let box_h = content_h.saturating_sub(2 * OUTER);
        let pct = cfg.chat_pct.clamp(10, 90);
        let full_w = width.saturating_sub(2 * OUTER);
        let (chat_x, chat_bw, logs_x, logs_bw) = if cfg.fullscreen == 2 && split {
            // Action pane fills the screen; chat parked offscreen (w=0).
            // Parked panes keep w==0 so `Pane::adopt` clones content without a
            // catastrophic 1-column reflow of the full scrollback.
            (width, 0, OUTER, full_w)
        } else if cfg.fullscreen == 1 || !split {
            // Chat fills the screen; action parked offscreen (w=0).
            (OUTER, full_w, width, 0)
        } else {
            let avail_w = width.saturating_sub(2 * OUTER + GAP);
            let chat_w = avail_w * pct / 100;
            let logs_w = avail_w - chat_w;
            // chat takes the right box when swapped.
            if cfg.swap {
                (OUTER + logs_w + GAP, chat_w, OUTER, logs_w)
            } else {
                (OUTER, chat_w, OUTER + chat_w + GAP, logs_w)
            }
        };
        let th = cfg.theme;
        // Allow w==0 for parked panes — do not `.max(cw)` the outer box or
        // fullscreen parking becomes a 1-col reflow of the whole history.
        let chat = Pane::new(chat_x, box_y, chat_bw, box_h, cw, ch, th.chat_fg, th.chat_bg, cfg.chat_title.clone(), true);
        let logs = Pane::new(logs_x, box_y, logs_bw, box_h, cw, ch, th.logs_fg, th.logs_bg, cfg.logs_title.clone(), false);
        let mut status_left = String::from("ChittiOS v");
        status_left.push_str(crate::VERSION);
        Screen {
            addr, width, height, pitch, bpp_bytes, r_shift, g_shift, b_shift, scale, chat, logs,
            status_left,
            status_right: String::new(),
            caret_on: true,
            caret_last_ms: 0,
            composer_active: false,
            composer_line: String::new(),
            composer_cur: 0,
            composer_hint_l: String::from("Tab select · ↑↓ menu · Enter send · /cmds · @files"),
            composer_hint_r: String::new(),
            suggest_open: false,
            suggest_items: alloc::vec::Vec::new(),
            suggest_sel: 0,
            suggest_rect: None,
            blink_seen_ms: u64::MAX,
            blink_calls: 0,
            clock_alive: false,
            focus_action: false,
            right: RightMode::Closed,
            tabs: Vec::new(),
            active: 0,
            right_before_editor: RightMode::Closed,
            layout: cfg.clone(),
            theme: th,
            cur_x: width / 2,
            cur_y: height / 2,
            cur_vis: false,
            cur_active: false,
            cur_saved: [(0, 0, 0); (CUR_W * CUR_H) as usize],
        }
    }

    fn cw(&self) -> u64 {
        CELL_W * self.scale
    }
    fn ch(&self) -> u64 {
        CELL_H * self.scale
    }

    // --- pixel plumbing --------------------------------------------------

    fn put_pixel(&self, x: u64, y: u64, c: Rgb) {
        if x >= self.width || y >= self.height {
            return;
        }
        let offset = y * self.pitch + x * self.bpp_bytes;
        let value: u32 =
            ((c.0 as u32) << self.r_shift) | ((c.1 as u32) << self.g_shift) | ((c.2 as u32) << self.b_shift);
        // SAFETY: `offset` is bounds-checked against the reported geometry; the
        // framebuffer is a valid, kernel-owned MMIO region.
        unsafe {
            let ptr = (self.addr as *mut u8).add(offset as usize);
            // Fast path: 32-bit linear FB (virtio / GOP / ramfb all are).
            if self.bpp_bytes == 4 {
                (ptr as *mut u32).write_volatile(value);
            } else {
                for i in 0..self.bpp_bytes {
                    ptr.add(i as usize).write_volatile((value >> (i * 8)) as u8);
                }
            }
        }
    }

    /// Pack an RGB triple into a framebuffer native pixel word.
    #[inline]
    fn pack_rgb(&self, c: Rgb) -> u32 {
        ((c.0 as u32) << self.r_shift) | ((c.1 as u32) << self.g_shift) | ((c.2 as u32) << self.b_shift)
    }

    /// Blit a row of packed `0x00RRGGBB` pixels into the FB at `(x,y)`.
    /// Much faster than per-pixel put for video frames (one bounds check +
    /// sequential stores).
    fn blit_rgb32_row(&self, x: u64, y: u64, row: &[u32]) {
        if y >= self.height || x >= self.width || row.is_empty() {
            return;
        }
        let n = row.len().min((self.width - x) as usize);
        let offset = y * self.pitch + x * self.bpp_bytes;
        // SAFETY: n is clipped to the scanline; FB is kernel-owned MMIO.
        unsafe {
            let mut ptr = (self.addr as *mut u8).add(offset as usize);
            if self.bpp_bytes == 4
                && self.r_shift == 16
                && self.g_shift == 8
                && self.b_shift == 0
            {
                // Native XRGB8888 — store as-is (our RGB packs match).
                let dst = ptr as *mut u32;
                for i in 0..n {
                    dst.add(i).write_volatile(row[i]);
                }
            } else if self.bpp_bytes == 4 {
                let dst = ptr as *mut u32;
                for i in 0..n {
                    let c = row[i];
                    let rgb = (
                        ((c >> 16) & 0xff) as u8,
                        ((c >> 8) & 0xff) as u8,
                        (c & 0xff) as u8,
                    );
                    dst.add(i).write_volatile(self.pack_rgb(rgb));
                }
            } else {
                for i in 0..n {
                    let c = row[i];
                    let rgb = (
                        ((c >> 16) & 0xff) as u8,
                        ((c >> 8) & 0xff) as u8,
                        (c & 0xff) as u8,
                    );
                    let value = self.pack_rgb(rgb);
                    for b in 0..self.bpp_bytes {
                        ptr.add(b as usize).write_volatile((value >> (b * 8)) as u8);
                    }
                    ptr = ptr.add(self.bpp_bytes as usize);
                }
            }
        }
    }

    fn fill_rect(&self, x: u64, y: u64, w: u64, h: u64, c: Rgb) {
        if w == 0 || h == 0 {
            return;
        }
        // Fast path: pack once and blast whole scanlines (critical for video
        // letterbox / status bars — per-pixel put_pixel was multi-ms flashes).
        let packed = ((c.0 as u32) << 16) | ((c.1 as u32) << 8) | c.2 as u32;
        if self.bpp_bytes == 4 && self.r_shift == 16 && self.g_shift == 8 && self.b_shift == 0 {
            let mut row = alloc::vec![packed; w as usize];
            for dy in 0..h {
                self.blit_rgb32_row(x, y + dy, &row);
            }
            // silence unused mut if n=0 — row is mut for potential reuse
            let _ = &mut row;
            return;
        }
        for dy in 0..h {
            for dx in 0..w {
                self.put_pixel(x + dx, y + dy, c);
            }
        }
    }

    /// Read a framebuffer pixel back as `Rgb` (inverse of `put_pixel`), for
    /// saving the background under the mouse cursor.
    fn get_pixel(&self, x: u64, y: u64) -> Rgb {
        if x >= self.width || y >= self.height {
            return (0, 0, 0);
        }
        let offset = y * self.pitch + x * self.bpp_bytes;
        // SAFETY: bounds-checked offset into the kernel-owned framebuffer.
        let mut val: u32 = 0;
        unsafe {
            let ptr = (self.addr as *const u8).add(offset as usize);
            for i in 0..self.bpp_bytes {
                val |= (ptr.add(i as usize).read_volatile() as u32) << (i * 8);
            }
        }
        (
            ((val >> self.r_shift) & 0xff) as u8,
            ((val >> self.g_shift) & 0xff) as u8,
            ((val >> self.b_shift) & 0xff) as u8,
        )
    }

    /// Restore the patch saved beneath the cursor (erasing the sprite).
    fn cursor_restore(&self) {
        if !self.cur_vis {
            return;
        }
        for dy in 0..CUR_H {
            for dx in 0..CUR_W {
                self.put_pixel(self.cur_x + dx, self.cur_y + dy, self.cur_saved[(dy * CUR_W + dx) as usize]);
            }
        }
    }

    /// Save the framebuffer under the cursor and draw the arrow sprite.
    fn cursor_draw(&mut self) {
        for dy in 0..CUR_H {
            for dx in 0..CUR_W {
                self.cur_saved[(dy * CUR_W + dx) as usize] = self.get_pixel(self.cur_x + dx, self.cur_y + dy);
            }
        }
        for dy in 0..CUR_H {
            for dx in 0..CUR_W {
                match CURSOR[(dy * CUR_W + dx) as usize] {
                    1 => self.put_pixel(self.cur_x + dx, self.cur_y + dy, (15, 15, 17)),
                    2 => self.put_pixel(self.cur_x + dx, self.cur_y + dy, (245, 245, 248)),
                    _ => {}
                }
            }
        }
        self.cur_vis = true;
    }

    /// Redraw the cursor on top after a content change (if the mouse is active).
    /// The caller must have already erased the old cursor (`cursor_restore`) so
    /// the freshly-saved background is clean.
    fn cursor_overlay(&mut self) {
        if self.cur_active {
            self.cursor_draw();
        }
    }

    /// The action-pane close-button rectangle `(x, y, w, h)` — a `[x]` at the
    /// top-right of the action pane title. Only meaningful when the pane is open.
    fn close_btn(&self) -> (u64, u64, u64, u64) {
        let cw = self.cw();
        let w = cw * 3;
        let x = self.logs.x + self.logs.w - BORDER - PAD - w;
        let y = self.logs.y + BORDER + 4;
        (x, y, w, self.ch())
    }

    /// A soft drop shadow for a box at `(x,y,w,h)` — two offset dark rectangles
    /// (a web-style elevation cue), drawn *before* the box so the box overpaints
    /// all but the bottom-right offset strip. Clipped at the screen edges by
    /// `fill_rect`. Darkens whatever is behind (screen bg for panes, the panes
    /// for a modal) toward black.
    fn drop_shadow(&self, x: u64, y: u64, w: u64, h: u64) {
        let s = 4 * self.scale; // shadow depth in px
        // Only the right + bottom bands stay visible once the box is filled on
        // top, so shade just those (cheap): a darker inner band nearest the box
        // fading to a fainter outer band — a soft web-style drop shadow.
        // Right side.
        self.shade_rect(x + w, y + s, s, h, 0.35); // inner (darkest)
        self.shade_rect(x + w + s, y + s, s, h, 0.60); // outer (fainter)
        // Bottom side.
        self.shade_rect(x + s, y + h, w, s, 0.35);
        self.shade_rect(x + s, y + h + s, w + s, s, 0.60);
        // Bottom-right corner, so the two bands meet cleanly.
        self.shade_rect(x + w, y + h, s, s, 0.35);
    }

    /// Fill `(x,y,w,h)` with the pixels beneath it darkened toward black by
    /// `factor` (0 = black, 1 = unchanged) — a cheap translucent-shadow effect
    /// without an alpha channel. Reads + rewrites each pixel.
    fn shade_rect(&self, x: u64, y: u64, w: u64, h: u64, factor: f32) {
        let x1 = (x + w).min(self.width);
        let y1 = (y + h).min(self.height);
        for py in y..y1 {
            for px in x..x1 {
                let (r, g, b) = self.get_pixel(px, py);
                let d = |c: u8| (c as f32 * factor) as u8;
                self.put_pixel(px, py, (d(r), d(g), d(b)));
            }
        }
    }

    /// Draw a `t`-thick rectangle outline (the pane border).
    fn rect_outline(&self, x: u64, y: u64, w: u64, h: u64, t: u64, c: Rgb) {
        self.fill_rect(x, y, w, t, c); // top
        self.fill_rect(x, y + h - t, w, t, c); // bottom
        self.fill_rect(x, y, t, h, c); // left
        self.fill_rect(x + w - t, y, t, h, c); // right
    }

    /// Alpha-blend one printable glyph at `(px,py)`, each atlas pixel expanded to
    /// a `scale`x`scale` block. Non-printable bytes render as a blank cell.
    fn blit_glyph(&self, px: u64, py: u64, byte: u8, fg: Rgb, bg: Rgb) {
        let idx = if (FIRST..=LAST).contains(&byte) { (byte - FIRST) as usize } else { 0 };
        let g = &GLYPHS[idx];
        let s = self.scale;
        for gy in 0..CH {
            for gx in 0..CW {
                let a = g[gy * CW + gx] as u32;
                let color = if a == 0 {
                    bg
                } else {
                    let mix = |b: u8, f: u8| (((b as u32) * (255 - a) + (f as u32) * a) / 255) as u8;
                    (mix(bg.0, fg.0), mix(bg.1, fg.1), mix(bg.2, fg.2))
                };
                let bx = px + gx as u64 * s;
                let by = py + gy as u64 * s;
                for sy in 0..s {
                    for sx in 0..s {
                        self.put_pixel(bx + sx, by + sy, color);
                    }
                }
            }
        }
    }

    /// Render `s` at pixel `(px,py)`, advancing one scaled cell per byte. Returns
    /// the x past the last glyph. Clips at `self.width`. Titles + status bar.
    fn draw_str(&self, px: u64, py: u64, s: &str, fg: Rgb, bg: Rgb) -> u64 {
        let mut x = px;
        let cw = self.cw();
        for &b in s.as_bytes() {
            if x + cw > self.width {
                break;
            }
            self.blit_glyph(x, py, b, fg, bg);
            x += cw;
        }
        x
    }

    // --- pane text -------------------------------------------------------

    /// Scroll a pane's interior up by one text row: the top grid row is evicted
    /// into the scrollback ring, the grid shifts up, and (when the view is live)
    /// the pixels shift with it.
    fn scroll_pane(&self, p: &mut Pane) {
        // Grid + scrollback first — the source of truth.
        let cols = p.cols as usize;
        if p.grid.len() >= cols {
            p.hist.push_back(p.grid[..cols].to_vec());
            while p.hist.len() > HIST_MAX {
                p.hist.pop_front();
                // Absolute selection coordinates shift with the evicted line;
                // a selection that loses its first line is dropped.
                p.sel = p.sel.and_then(|((r1, c1), (r2, c2))| {
                    (r1.min(r2) > 0).then(|| ((r1 - 1, c1), (r2 - 1, c2)))
                });
            }
            p.grid.copy_within(cols.., 0);
            let start = p.grid.len() - cols;
            let fg = p.fg;
            for c in &mut p.grid[start..] {
                *c = (0, fg);
            }
            if p.view > 0 {
                // Keep the scrolled view anchored on the same content.
                p.view = (p.view + 1).min(p.hist.len());
                return; // pixels are frozen on the scrolled view
            }
        }
        let x0 = p.ix;
        let w = p.cols * p.cw;
        let top = p.iy;
        let h = p.rows * p.ch;
        let step = (self.pitch * p.ch) as usize;
        let row_bytes = (w * self.bpp_bytes) as usize;
        // SAFETY: every source/destination row lies inside the framebuffer and
        // inside this pane's x-span; source and destination never overlap within
        // a single `copy_nonoverlapping` (they are `p.ch` rows apart).
        unsafe {
            let base = self.addr as *mut u8;
            for row in 0..(h - p.ch) {
                let dst = ((top + row) * self.pitch + x0 * self.bpp_bytes) as usize;
                base.add(dst).copy_from_nonoverlapping(base.add(dst + step), row_bytes);
            }
        }
        self.fill_rect(x0, top + h - p.ch, w, p.ch, p.bg);
    }

    /// Repaint a pane's interior from its scrollback + grid at the current view
    /// offset. The one text renderer used by scroll, redraw, relayout, and the
    /// mouse selection (whose cells get the selection background).
    ///
    /// **No full-interior clear** — blank-then-repaint is what made selection
    /// drag (and scroll) flicker on the single-buffered framebuffer. Every cell
    /// is painted in place (bg + glyph), so nothing is ever blanked mid-frame.
    fn render_view(&self, p: &Pane) {
        let cols = p.cols as usize;
        if p.grid.len() < cols {
            return;
        }
        let sel = p.sel.map(|(a, b)| crate::textsel::normalize(a, b));
        let view = p.view.min(p.hist.len());
        let first = p.hist.len() - view;
        for r in 0..p.rows as usize {
            let gi = first + r;
            let line: Option<&[Cell]> = if gi < p.hist.len() {
                Some(&p.hist[gi])
            } else {
                let gr = gi - p.hist.len();
                if gr >= p.rows as usize {
                    break;
                }
                Some(&p.grid[gr * cols..(gr + 1) * cols])
            };
            for c in 0..cols {
                let (b, fg) = line.and_then(|l| l.get(c).copied()).unwrap_or((0, p.default_fg));
                let x = p.ix + c as u64 * p.cw;
                let y = p.iy + r as u64 * p.ch;
                let selected = sel.is_some_and(|s| crate::textsel::contains(s, gi, c));
                let bg = if selected { self.theme.editor_sel } else { p.bg };
                // Always fill the cell first so deselected / empty cells leave
                // no residue (selection highlight, partial glyphs).
                self.fill_rect(x, y, p.cw, p.ch, bg);
                if (0x21..=0x7e).contains(&b) {
                    self.blit_glyph(x, y, b, fg, bg);
                }
            }
        }
        // A scrolled-back view gets a position marker in the top-right corner.
        if view > 0 {
            let tag = alloc::format!("[-{}] ", view);
            let tx = (p.ix + p.cols * p.cw).saturating_sub(tag.len() as u64 * p.cw);
            self.draw_str(tx, p.iy, &tag, self.theme.accent, p.bg);
        }
    }

    /// Paint one chat-pane cell at absolute line `gi`, column `c` (selection
    /// highlight applied when `selected`). Used by differential selection
    /// updates so a drag only touches cells that actually changed.
    fn paint_chat_cell(&self, p: &Pane, gi: usize, c: usize, selected: bool) {
        let cols = p.cols as usize;
        if c >= cols || p.grid.len() < cols {
            return;
        }
        let view = p.view.min(p.hist.len());
        let first = p.hist.len() - view;
        if gi < first || gi >= first + p.rows as usize {
            return; // off-screen
        }
        let r = gi - first;
        let (b, fg) = if gi < p.hist.len() {
            p.hist[gi].get(c).copied().unwrap_or((0, p.default_fg))
        } else {
            let gr = gi - p.hist.len();
            if gr >= p.rows as usize {
                return;
            }
            p.grid.get(gr * cols + c).copied().unwrap_or((0, p.default_fg))
        };
        let x = p.ix + c as u64 * p.cw;
        let y = p.iy + r as u64 * p.ch;
        let bg = if selected { self.theme.editor_sel } else { p.bg };
        self.fill_rect(x, y, p.cw, p.ch, bg);
        if (0x21..=0x7e).contains(&b) {
            self.blit_glyph(x, y, b, fg, bg);
        }
    }

    /// Repaint only the cells whose selection membership differs between
    /// `old_sel` and `new_sel` (both raw anchor/head pairs). Avoids the
    /// full-pane flash that a drag-triggered `render_view` used to cause.
    fn repaint_sel_diff(
        &self,
        p: &Pane,
        old_sel: Option<((usize, usize), (usize, usize))>,
        new_sel: Option<((usize, usize), (usize, usize))>,
    ) {
        let old = old_sel.map(|(a, b)| crate::textsel::normalize(a, b));
        let new = new_sel.map(|(a, b)| crate::textsel::normalize(a, b));
        if old == new {
            return;
        }
        let cols = p.cols as usize;
        let view = p.view.min(p.hist.len());
        let first = p.hist.len() - view;
        let last = first + p.rows as usize;
        // Bound the walk to the union of the two ranges (clamped to the view).
        let span = |s: Option<((usize, usize), (usize, usize))>| -> Option<(usize, usize)> {
            s.map(|((r1, _), (r2, _))| (r1.max(first), (r2 + 1).min(last)))
        };
        let (lo, hi) = match (span(old), span(new)) {
            (Some((a, b)), Some((c, d))) => (a.min(c), b.max(d)),
            (Some((a, b)), None) | (None, Some((a, b))) => (a, b),
            (None, None) => return,
        };
        for gi in lo..hi {
            for c in 0..cols {
                let was = old.is_some_and(|s| crate::textsel::contains(s, gi, c));
                let now = new.is_some_and(|s| crate::textsel::contains(s, gi, c));
                if was != now {
                    self.paint_chat_cell(p, gi, c, now);
                }
            }
        }
    }

    /// Repaint the cell under the pane cursor from the grid (clears a leftover
    /// caret bar without blanking a real glyph that might share the cell).
    fn repaint_cursor_cell(&self, p: &Pane) {
        let cols = p.cols as usize;
        let idx = (p.row as usize).saturating_mul(cols).saturating_add(p.col as usize);
        let (b, fg) = p.grid.get(idx).copied().unwrap_or((0, p.default_fg));
        let x = p.cell_x();
        let y = p.cell_y();
        self.fill_rect(x, y, p.cw, p.ch, p.bg);
        if (0x21..=0x7e).contains(&b) {
            self.blit_glyph(x, y, b, fg, p.bg);
        }
    }

    fn caret_erase(&self, p: &Pane) {
        if !p.show_caret {
            return;
        }
        // Always restore the underlying cell — a plain bg bar erase leaves a
        // hole if a glyph shared the cell, and a leftover accent bar if the
        // next write never covers this position (composer panes).
        self.repaint_cursor_cell(p);
    }

    fn caret_draw(&self, p: &Pane) {
        // Chat pane with a Grok-style composer: caret lives only in the input
        // box — never in the scrollback/response area.
        if !p.show_caret || p.has_composer {
            return;
        }
        self.fill_rect(p.cell_x(), p.cell_y(), 2 * self.scale, p.ch, self.theme.accent);
    }

    fn newline(p: &mut Pane, s: &Screen) {
        p.col = 0;
        p.row += 1;
        if p.row >= p.rows {
            s.scroll_pane(p);
            p.row = p.rows - 1;
        }
    }

    /// Feed one byte to a pane (the per-pane analogue of a terminal write),
    /// running the ANSI escape parser first so `\x1b[…m` SGR codes recolour the
    /// stream instead of printing as garbage.
    fn pane_putc(s: &Screen, p: &mut Pane, byte: u8) {
        match p.esc {
            EscState::Esc => {
                // Only CSI (`ESC [`) is supported; anything else ends the escape.
                p.esc = if byte == b'[' {
                    p.csi_len = 0;
                    EscState::Csi
                } else {
                    EscState::Ground
                };
                return;
            }
            EscState::Csi => {
                if (0x40..=0x7e).contains(&byte) {
                    // Final byte: SGR (`m`) recolours; `K`/`C`/`D` (erase to end
                    // of line / cursor right / cursor left) support the shell's
                    // in-line editing. Other CSI are consumed and ignored.
                    let live = p.view == 0;
                    match byte {
                        b'm' => p.apply_sgr(),
                        b'K' => {
                            if live {
                                s.caret_erase(p);
                            }
                            let cols = p.cols as usize;
                            let (row, col) = (p.row as usize, p.col as usize);
                            for c in col..cols {
                                if let Some(cell) = p.grid.get_mut(row * cols + c) {
                                    *cell = (0, p.fg);
                                }
                            }
                            if live {
                                s.fill_rect(p.cell_x(), p.cell_y(), (p.cols - p.col) * p.cw, p.ch, p.bg);
                                s.caret_draw(p); // no-op when p.has_composer
                            }
                        }
                        b'C' | b'D' => {
                            let n = p.csi_param().max(1);
                            if live {
                                s.caret_erase(p);
                            }
                            p.col = if byte == b'C' { (p.col + n).min(p.cols - 1) } else { p.col.saturating_sub(n) };
                            if live {
                                s.caret_draw(p); // no-op when p.has_composer
                            }
                        }
                        _ => {}
                    }
                    p.esc = EscState::Ground;
                } else if p.csi_len < p.csi.len() {
                    p.csi[p.csi_len] = byte;
                    p.csi_len += 1;
                }
                return;
            }
            EscState::Ground => {}
        }
        if byte == 0x1b {
            p.esc = EscState::Esc;
            return;
        }
        // While scrolled back, the grid/scrollback still update but the pixels
        // stay frozen on the scrolled view (`scroll_pane` keeps it anchored).
        let live = p.view == 0;
        if live {
            s.caret_erase(p);
        }
        match byte {
            b'\n' => Screen::newline(p, s),
            b'\r' => p.col = 0,
            b'\t' => {
                let next = (p.col / 4 + 1) * 4;
                while p.col < next && p.col < p.cols {
                    p.set_cell(b' ');
                    if live {
                        s.blit_glyph(p.cell_x(), p.cell_y(), b' ', p.fg, p.bg);
                    }
                    p.col += 1;
                }
                if p.col >= p.cols {
                    Screen::newline(p, s);
                }
            }
            0x08 | 0x7f => {
                if p.col > 0 {
                    p.col -= 1;
                } else if p.row > 0 {
                    p.row -= 1;
                    p.col = p.cols - 1;
                }
                p.set_cell(0);
                if live {
                    s.blit_glyph(p.cell_x(), p.cell_y(), b' ', p.fg, p.bg);
                }
            }
            0x20..=0x7e => {
                p.set_cell(byte);
                if live {
                    s.blit_glyph(p.cell_x(), p.cell_y(), byte, p.fg, p.bg);
                }
                p.col += 1;
                if p.col >= p.cols {
                    Screen::newline(p, s);
                }
            }
            _ => {}
        }
        // Grid caret only when the pane has no composer (`caret_draw` is a
        // no-op for `has_composer` panes so scrollback never keeps a bar).
        if p.view == 0 {
            s.caret_draw(p);
        }
    }

    // --- framing ---------------------------------------------------------

    fn draw_frame(&self, p: &Pane, active: bool) {
        self.draw_frame_titled(p, active, &p.title);
    }

    /// Like [`draw_frame`] but with an explicit title (the editor overrides the
    /// pane title with `editor: <file>`).
    fn draw_frame_titled(&self, p: &Pane, active: bool, title: &str) {
        let border = if active { self.theme.accent } else { self.theme.border_dim };
        let title_c = if active { self.theme.title_active } else { self.theme.title_dim };
        self.rect_outline(p.x, p.y, p.w, p.h, BORDER, border);
        // Title, just inside the top border — ellipsize so a long path never
        // paints into the close button / pane edge.
        let ty = p.y + BORDER + 4;
        let tx = p.x + BORDER + PAD;
        let max_w = p.w.saturating_sub(2 * (BORDER + PAD) + self.cw() * 4); // room for " *" / [x]
        let max_cols = (max_w / self.cw()).max(1) as usize;
        let title = crate::textsel::ellipsize(title, max_cols);
        let end = self.draw_str(tx, ty, &title, title_c, p.bg);
        if active && end + 2 * self.cw() <= p.x + p.w - BORDER - PAD {
            self.draw_str(end, ty, " *", self.theme.accent, p.bg);
        }
        // Separator under the title.
        let sep_y = ty + self.ch() + 3;
        self.fill_rect(p.x + BORDER, sep_y, p.w - 2 * BORDER, 1, self.theme.sep_dim);
    }

    fn draw_status(&self) {
        let bar_h = self.ch() + 8;
        let sy_top = self.height - bar_h;
        self.fill_rect(0, sy_top, self.width, bar_h, self.theme.status_bg);
        let ty = sy_top + 4;
        let cw = self.cw();
        // Left = the brand mark then the brand text (accent). The glyph radius is
        // sized so the ring (extent ≈ 7/6·r) fits within the bar height.
        let lr = (((bar_h / 2).saturating_sub(2)) * 6 / 7).max(5);
        let lhalf = ((lr / 3).max(3)) / 2;
        let lcx = OUTER + lr + lhalf;
        self.draw_logo(lcx, sy_top + bar_h / 2, lr, self.theme.accent, self.theme.chat_fg);
        let text_x = lcx + lr + lhalf + cw / 2;
        // Split the bar: left brand, right system info. Never overlap — each
        // side is ellipsized into its half (with a 2-cell gap in the middle).
        let gap = 2 * cw;
        let usable = self.width.saturating_sub(text_x + OUTER + gap);
        let left_budget = (usable / 2 / cw).max(4) as usize;
        let right_budget = (usable.saturating_sub(left_budget as u64 * cw) / cw).max(4) as usize;
        let left = crate::textsel::ellipsize(&self.status_left, left_budget);
        let right = crate::textsel::ellipsize(&self.status_right, right_budget);
        self.draw_str(text_x, ty, &left, self.theme.accent, self.theme.status_bg);
        let rlen = right.chars().count() as u64;
        let rx = self.width.saturating_sub(rlen * cw + OUTER);
        // Guard: right edge of left text must stay left of right text.
        let left_end = text_x + left.chars().count() as u64 * cw + gap;
        let rx = rx.max(left_end).min(self.width.saturating_sub(OUTER));
        // Re-ellipsize right if the guard ate columns.
        let right_cols = ((self.width.saturating_sub(rx + OUTER)) / cw) as usize;
        let right = crate::textsel::ellipsize(&self.status_right, right_cols);
        self.draw_str(rx, ty, &right, self.theme.status_fg, self.theme.status_bg);
    }

    /// Draw `s` within `[x, x+max_w)`, ellipsizing when it would overflow.
    /// Returns the x just past the last painted glyph.
    fn draw_str_fit(&self, x: u64, y: u64, s: &str, fg: Rgb, bg: Rgb, max_w: u64) -> u64 {
        let cols = (max_w / self.cw()) as usize;
        let t = crate::textsel::ellipsize(s, cols);
        self.draw_str(x, y, &t, fg, bg)
    }

    /// Geometry of the Grok-style input composer inside the chat pane:
    /// `(box_x, box_y, box_w, box_h, text_x, text_y, hint_y)`.
    fn composer_geom(&self) -> (u64, u64, u64, u64, u64, u64, u64) {
        let p = &self.chat;
        let vpad = COMPOSER_VPAD;
        let hint_gap = COMPOSER_HINT_GAP;
        // Pane interior bottom (above outer border + pad).
        let bottom = p.y + p.h - BORDER - PAD;
        let hint_y = bottom.saturating_sub(p.ch);
        let box_h = vpad + p.ch + vpad + 2; // 1px border each side
        let box_y = hint_y.saturating_sub(hint_gap + box_h);
        let box_x = p.x + BORDER + PAD;
        let box_w = p.w.saturating_sub(2 * (BORDER + PAD));
        let text_x = box_x + 8;
        let text_y = box_y + 1 + vpad;
        (box_x, box_y, box_w, box_h, text_x, text_y, hint_y)
    }

    /// Draw a soft-rounded rectangle outline (Grok-style input chrome).
    fn rounded_outline(&self, x: u64, y: u64, w: u64, h: u64, r: u64, c: Rgb) {
        if w < 2 * r || h < 2 * r {
            self.rect_outline(x, y, w, h, 1, c);
            return;
        }
        // Straight edges.
        self.fill_rect(x + r, y, w - 2 * r, 1, c);
        self.fill_rect(x + r, y + h - 1, w - 2 * r, 1, c);
        self.fill_rect(x, y + r, 1, h - 2 * r, c);
        self.fill_rect(x + w - 1, y + r, 1, h - 2 * r, c);
        // Quarter-circle corners (1px arc).
        let r = r as i64;
        for &(cx, cy, sx, sy) in &[
            (x + r as u64, y + r as u64, -1i64, -1i64),
            (x + w - 1 - r as u64, y + r as u64, 1i64, -1i64),
            (x + r as u64, y + h - 1 - r as u64, -1i64, 1i64),
            (x + w - 1 - r as u64, y + h - 1 - r as u64, 1i64, 1i64),
        ] {
            for dy in 0..=r {
                for dx in 0..=r {
                    let d2 = dx * dx + dy * dy;
                    // Outer rim only (~1 px thick).
                    if d2 <= r * r && d2 >= (r - 1) * (r - 1) {
                        self.put_pixel((cx as i64 + sx * dx) as u64, (cy as i64 + sy * dy) as u64, c);
                    }
                }
            }
        }
    }

    /// Visible slice of the composer line and the column of the caret within it
    /// (for long lines that scroll inside the box).
    fn composer_visible(&self, max_cols: usize) -> (usize, &str, usize) {
        let line = self.composer_line.as_str();
        let cur = self.composer_cur.min(line.len());
        if line.len() <= max_cols {
            return (0, line, cur);
        }
        let start = cur.saturating_sub(max_cols.saturating_sub(1)).min(line.len().saturating_sub(max_cols));
        let vis = &line[start..start + max_cols.min(line.len() - start)];
        (start, vis, cur.saturating_sub(start))
    }

    /// Pixel x of the caret bar inside the composer box.
    fn composer_caret_x(&self) -> Option<u64> {
        if !self.composer_active || self.action_focused() || !self.chat.has_composer {
            return None;
        }
        let (bx, _by, bw, _bh, tx, _ty, _hy) = self.composer_geom();
        let max_cols = ((bw.saturating_sub(16)) / self.chat.cw).saturating_sub(2) as usize;
        let (_start, _vis, caret_col) = self.composer_visible(max_cols);
        let prompt_cols = 2u64; // "> "
        Some(tx + (prompt_cols + caret_col as u64) * self.chat.cw)
    }

    /// Paint **only** the composer caret bar (blink path). Never blanks the box —
    /// a full `draw_composer` on every blink (and every streamed token) was the
    /// flicker during response rendering.
    fn paint_composer_caret(&self) {
        let Some(cx) = self.composer_caret_x() else { return };
        let (_bx, _by, _bw, _bh, _tx, ty, _hy) = self.composer_geom();
        let color = if self.caret_on { self.theme.accent } else { self.theme.composer_bg };
        self.fill_rect(cx, ty, 2 * self.scale.max(1), self.chat.ch, color);
    }

    /// Paint the Grok-style composer box + hint bar at the bottom of the chat pane.
    ///
    /// Paints **in place** (no strip-wide clear). The chat grid is already sized
    /// above this region, so scrollback never lands here; blanking the whole
    /// reserved strip on every call is what made streaming replies flash the box.
    fn draw_composer(&self) {
        if !self.chat.has_composer || self.chat.w == 0 {
            return; // parked (action-fullscreen) — no visible chat chrome
        }
        let (bx, by, bw, bh, tx, ty, hy) = self.composer_geom();
        // Elevated fill + rounded border (accent when the prompt owns focus).
        self.fill_rect(bx + 1, by + 1, bw.saturating_sub(2), bh.saturating_sub(2), self.theme.composer_bg);
        let border = if self.composer_active && !self.action_focused() {
            self.theme.accent
        } else {
            self.theme.composer_border
        };
        let radius = (4 * self.scale).max(4);
        self.rounded_outline(bx, by, bw, bh, radius, border);
        // Prompt glyph + input text.
        let prompt = "> ";
        let max_cols = ((bw.saturating_sub(16)) / self.chat.cw).saturating_sub(2) as usize;
        let (_vis_start, vis, caret_col) = self.composer_visible(max_cols);
        let mut x = self.draw_str(tx, ty, prompt, self.theme.accent, self.theme.composer_bg);
        x = self.draw_str(x, ty, vis, self.theme.chat_fg, self.theme.composer_bg);
        // Clear leftover glyphs to the right of the text (shrinking line).
        let rest = (bx + bw).saturating_sub(x + 4);
        if rest > 0 {
            self.fill_rect(x, ty, rest, self.chat.ch, self.theme.composer_bg);
        }
        // Caret inside the box (only while the composer is the live prompt).
        if self.composer_active && !self.action_focused() {
            let cx = tx + (prompt.len() as u64 + caret_col as u64) * self.chat.cw;
            let color = if self.caret_on { self.theme.accent } else { self.theme.composer_bg };
            self.fill_rect(cx, ty, 2 * self.scale.max(1), self.chat.ch, color);
        }
        // Hint bar: shortcuts left, model/mode right — each side ellipsized so
        // a narrow chat pane never paints past the composer box (or overlaps).
        let hx = bx;
        let hw = bw;
        let cw = self.chat.cw;
        self.fill_rect(hx, hy, hw, self.chat.ch, self.chat.bg);
        let total_cols = (hw / cw).max(1) as usize;
        let gap = 2usize;
        let right_raw = self.composer_hint_r.chars().count();
        let right_cols = right_raw.min(total_cols / 3).min(total_cols.saturating_sub(gap + 4));
        let left_cols = total_cols.saturating_sub(right_cols + if right_cols > 0 { gap } else { 0 });
        let left = crate::textsel::ellipsize(&self.composer_hint_l, left_cols);
        let right = crate::textsel::ellipsize(&self.composer_hint_r, right_cols);
        self.draw_str(hx, hy, &left, self.theme.composer_hint, self.chat.bg);
        if !right.is_empty() {
            let rlen = right.chars().count() as u64 * cw;
            self.draw_str(hx + hw.saturating_sub(rlen), hy, &right, self.theme.composer_hint, self.chat.bg);
        }
        // Suggestion menu sits above the composer (slash commands / @files).
        self.draw_suggest_popup();
    }

    /// Geometry of the suggestion popup: `(x, y, w, h)` above the composer.
    fn suggest_geom(&self) -> Option<(u64, u64, u64, u64)> {
        if !self.suggest_open || self.suggest_items.is_empty() || !self.chat.has_composer {
            return None;
        }
        let (bx, by, bw, _bh, _tx, _ty, _hy) = self.composer_geom();
        let n = self.suggest_items.len().min(8) as u64;
        let row_h = self.chat.ch + 4;
        let vpad = 6u64;
        let h = vpad + n * row_h + vpad;
        let y = by.saturating_sub(h + 6);
        // Keep the popup inside the chat pane (below the title header).
        let min_y = self.chat.iy;
        let y = y.max(min_y);
        let h = by.saturating_sub(y + 4).min(h);
        if h < self.chat.ch + vpad {
            return None;
        }
        Some((bx, y, bw, h))
    }

    /// Erase the previous suggestion popup (and the gap down to the composer).
    ///
    /// **Fast path:** fill the old popup rect with `chat.bg` and restore only
    /// the chat rows that intersect it. Avoids a full-pane `render_view` on
    /// every keystroke (that made typing feel multi-hundred-ms laggy).
    ///
    /// When `full_restore` is true (menu fully dismissed), also re-paints
    /// composer chrome without the popup.
    fn suggest_clear_region(&self, full_restore: bool) {
        if self.chat.w == 0 {
            return;
        }
        let (bx, by, bw, _bh, _tx, _ty, _hy) = self.composer_geom();
        let (x, y, w, _h) = self.suggest_rect.unwrap_or_else(|| {
            let h = by.saturating_sub(self.chat.iy).min(bw);
            (bx, by.saturating_sub(h), bw, h)
        });
        let pad = 4u64;
        let left = x.saturating_sub(pad).max(self.chat.x + BORDER);
        let top = y.saturating_sub(pad).max(self.chat.iy);
        let bottom = by;
        let right = (x + w + pad).min(self.chat.x + self.chat.w - BORDER);
        let rw = right.saturating_sub(left);
        let rh = bottom.saturating_sub(top);
        if rw > 0 && rh > 0 {
            self.fill_rect(left, top, rw, rh, self.chat.bg);
        }
        // Restore only grid rows overlapping the erased band (not the whole pane).
        self.render_view_rows_intersecting(top, bottom);
        if full_restore {
            self.paint_composer_box_only();
        }
    }

    /// Re-paint chat grid rows whose vertical span intersects `[y0, y1)`.
    fn render_view_rows_intersecting(&self, y0: u64, y1: u64) {
        let p = &self.chat;
        if p.w == 0 || p.rows == 0 {
            return;
        }
        let ch = p.ch;
        let first = if y0 <= p.iy {
            0
        } else {
            ((y0 - p.iy) / ch) as usize
        };
        let last = if y1 <= p.iy {
            0
        } else {
            (((y1 - p.iy) + ch - 1) / ch).min(p.rows) as usize
        };
        if first >= last {
            return;
        }
        self.render_view_row_range(p, first, last);
    }

    /// Like [`render_view`] but only rows `[row0, row1)`.
    fn render_view_row_range(&self, p: &Pane, row0: usize, row1: usize) {
        let cols = p.cols as usize;
        if p.grid.len() < cols {
            return;
        }
        let sel = p.sel.map(|(a, b)| crate::textsel::normalize(a, b));
        let view = p.view.min(p.hist.len());
        let first = p.hist.len() - view;
        let row1 = row1.min(p.rows as usize);
        for r in row0..row1 {
            let gi = first + r;
            let line: Option<&[Cell]> = if gi < p.hist.len() {
                Some(&p.hist[gi])
            } else {
                let gr = gi - p.hist.len();
                if gr >= p.rows as usize {
                    break;
                }
                Some(&p.grid[gr * cols..(gr + 1) * cols])
            };
            for c in 0..cols {
                let (b, fg) = line.and_then(|l| l.get(c).copied()).unwrap_or((0, p.default_fg));
                let x = p.ix + c as u64 * p.cw;
                let y = p.iy + r as u64 * p.ch;
                let selected = sel.is_some_and(|s| crate::textsel::contains(s, gi, c));
                let bg = if selected { self.theme.editor_sel } else { p.bg };
                self.fill_rect(x, y, p.cw, p.ch, bg);
                if (0x21..=0x7e).contains(&b) {
                    self.blit_glyph(x, y, b, fg, bg);
                }
            }
        }
    }

    /// Composer chrome without the suggestion popup (used when clearing).
    fn paint_composer_box_only(&self) {
        if !self.chat.has_composer || self.chat.w == 0 {
            return;
        }
        let (bx, by, bw, bh, tx, ty, hy) = self.composer_geom();
        self.fill_rect(bx + 1, by + 1, bw.saturating_sub(2), bh.saturating_sub(2), self.theme.composer_bg);
        let border = if self.composer_active && !self.action_focused() {
            self.theme.accent
        } else {
            self.theme.composer_border
        };
        let radius = (4 * self.scale).max(4);
        self.rounded_outline(bx, by, bw, bh, radius, border);
        let prompt = "> ";
        let max_cols = ((bw.saturating_sub(16)) / self.chat.cw).saturating_sub(2) as usize;
        let (_vis_start, vis, caret_col) = self.composer_visible(max_cols);
        let mut x = self.draw_str(tx, ty, prompt, self.theme.accent, self.theme.composer_bg);
        x = self.draw_str(x, ty, vis, self.theme.chat_fg, self.theme.composer_bg);
        let rest = (bx + bw).saturating_sub(x + 4);
        if rest > 0 {
            self.fill_rect(x, ty, rest, self.chat.ch, self.theme.composer_bg);
        }
        if self.composer_active && !self.action_focused() {
            let cx = tx + (prompt.len() as u64 + caret_col as u64) * self.chat.cw;
            let color = if self.caret_on { self.theme.accent } else { self.theme.composer_bg };
            self.fill_rect(cx, ty, 2 * self.scale.max(1), self.chat.ch, color);
        }
        let hx = bx;
        let hw = bw;
        let cw = self.chat.cw;
        self.fill_rect(hx, hy, hw, self.chat.ch, self.chat.bg);
        let total_cols = (hw / cw).max(1) as usize;
        let gap = 2usize;
        let right_raw = self.composer_hint_r.chars().count();
        let right_cols = right_raw.min(total_cols / 3).min(total_cols.saturating_sub(gap + 4));
        let left_cols = total_cols.saturating_sub(right_cols + if right_cols > 0 { gap } else { 0 });
        let left = crate::textsel::ellipsize(&self.composer_hint_l, left_cols);
        let right = crate::textsel::ellipsize(&self.composer_hint_r, right_cols);
        self.draw_str(hx, hy, &left, self.theme.composer_hint, self.chat.bg);
        if !right.is_empty() {
            let rlen = right.chars().count() as u64 * cw;
            self.draw_str(hx + hw.saturating_sub(rlen), hy, &right, self.theme.composer_hint, self.chat.bg);
        }
    }

    /// Paint the slash / @file suggestion list above the composer.
    ///
    /// Text is hard-clamped to the interior of the box so long `@/path/…`
    /// labels never paint past the rounded border (the previous layout gave
    /// labels only `cols/3` then right-aligned a detail that ran under the
    /// pane edge).
    fn draw_suggest_popup(&self) {
        let Some((x, y, w, h)) = self.suggest_geom() else {
            return;
        };
        let ch = self.chat.ch;
        let cw = self.chat.cw;
        let row_h = ch + 4;
        let vpad = 6u64;
        let hpad = 8u64; // left/right inset inside the rounded border
        let bg = self.theme.composer_bg;
        let sel_bg = self.theme.status_bg; // elevated highlight bar
        // Soft fill + border (matches composer chrome).
        self.fill_rect(x, y, w, h, bg);
        let radius = (4 * self.scale).max(4);
        self.rounded_outline(x, y, w, h, radius, self.theme.composer_border);

        // Usable text columns strictly inside the border + padding.
        let inner_x = x + hpad;
        let inner_w = w.saturating_sub(2 * hpad);
        let cols = (inner_w / cw).max(1) as usize;
        let text_right = inner_x + cols as u64 * cw; // last pixel exclusive of next col
        let n = self.suggest_items.len().min(8);
        let mut row_y = y + vpad;
        for i in 0..n {
            if row_y + ch > y + h.saturating_sub(2) {
                break;
            }
            let (ref label, ref detail) = self.suggest_items[i];
            let selected = i == self.suggest_sel;
            let row_bg = if selected { sel_bg } else { bg };
            self.fill_rect(x + 2, row_y.saturating_sub(1), w.saturating_sub(4), row_h, row_bg);

            // Selected: terracotta chevron; 2 columns reserved for the mark.
            let mark = if selected { "> " } else { "  " };
            let mark_fg = if selected { self.theme.accent } else { self.theme.composer_hint };
            let mut px = self.draw_str(inner_x, row_y, mark, mark_fg, row_bg);
            let mark_cols = 2usize;
            let avail = cols.saturating_sub(mark_cols);

            // Column split: short command labels leave room for a muted detail
            // on the right; long `@path` labels take the full row (no detail).
            let has_detail = !detail.is_empty() && label.chars().count() <= avail / 2;
            let det_cols = if has_detail {
                detail.chars().count().min(avail / 3).min(28).max(6)
            } else {
                0
            };
            let lab_cols = avail.saturating_sub(if det_cols > 0 { det_cols + 1 } else { 0 });

            let lab_fg = if selected { self.theme.accent } else { self.theme.chat_fg };
            // Paths: keep the trailing end (`../SOUL.md`); commands: head.
            let lab = if label.starts_with('@') || label.contains('/') {
                crate::textsel::ellipsize_end(label, lab_cols)
            } else {
                crate::textsel::ellipsize(label, lab_cols)
            };
            // Clamp drawn label so it never crosses into the detail zone.
            let lab_max_px = px + lab_cols as u64 * cw;
            px = self.draw_str(px, row_y, &lab, lab_fg, row_bg);
            if px > lab_max_px {
                // Shouldn't happen after ellipsize; blank any overflow residue.
                self.fill_rect(lab_max_px, row_y, px.saturating_sub(lab_max_px), ch, row_bg);
                px = lab_max_px;
            }

            if det_cols > 0 {
                let det = crate::textsel::ellipsize(detail, det_cols);
                let dlen = det.chars().count() as u64 * cw;
                // Right-align detail inside the inner box — never past text_right.
                let dx = text_right.saturating_sub(dlen).max(px + cw);
                if dx + dlen <= text_right && dx + dlen <= x + w.saturating_sub(4) {
                    self.draw_str(dx, row_y, &det, self.theme.composer_hint, row_bg);
                }
            }
            // Wipe any leftover pixels to the right of the last drawn glyph so
            // a previous longer selection highlight doesn't ghost.
            if text_right > px {
                // (row bg already filled; no-op unless we over-drew)
                let _ = px;
            }
            row_y += row_h;
        }
    }

    /// Paint the caret in its current blink state. When the chat pane has a
    /// Grok-style composer, the caret only blinks inside the box while the
    /// prompt is active — never during streamed reply output (that was a full
    /// box redraw and looked like the whole composer flickering).
    fn paint_caret(&self) {
        if self.chat.has_composer {
            if self.composer_active {
                self.paint_composer_caret();
            }
            return;
        }
        if !self.chat.show_caret || self.chat.view != 0 {
            return;
        }
        let color = if self.caret_on { self.theme.accent } else { self.chat.bg };
        self.fill_rect(self.chat.cell_x(), self.chat.cell_y(), 2 * self.scale, self.chat.ch, color);
    }

    /// Fill an integer-centred disc of radius `r` with `c` (round dots/caps).
    fn fill_disc(&self, cx: i64, cy: i64, r: i64, c: Rgb) {
        let r2 = r * r;
        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy <= r2 {
                    // Negative coords wrap to a huge u64 and are dropped by put_pixel's bounds check.
                    self.put_pixel((cx + dx) as u64, (cy + dy) as u64, c);
                }
            }
        }
    }

    /// Draw the **Synapse-C** brand mark centred at `(cx, cy)` with ring radius
    /// `r`: a single open ring (the capability) in `arc_c` with round end-caps,
    /// and a filled node (the agent) at the centre in `node_c`. Pure integer math
    /// — a ring test plus one angular gap — so it scales from a status-bar glyph
    /// to a splash logo. Geometry mirrors the SVG in DESIGN.md: stroke width ≈
    /// `6/17·r`, a ~91° opening (dasharray 80/27) whose centre sits ~10° above the
    /// +x axis (the SVG's `rotate(35)` on a dash starting at 3 o'clock), and a
    /// centre node of radius ≈ `0.32·r` (SVG r 5.5 against ring r 17).
    fn draw_logo(&self, cx: u64, cy: u64, r: u64, arc_c: Rgb, node_c: Rgb) {
        let (cx, cy, r) = (cx as i64, cy as i64, r as i64);
        let t = (r / 3).max(3); // stroke width, min 3 so a small mark still reads
        let half = t / 2;
        let (inner, outer) = ((r - half) * (r - half), (r + half) * (r + half));
        let span = r + half + 1;
        for dy in -span..=span {
            for dx in -span..=span {
                let d2 = dx * dx + dy * dy;
                if d2 < inner || d2 > outer {
                    continue;
                }
                // Skip the opening: a pixel is inside the gap when its direction is
                // within ~45.4° of the gap centre (984,-180) ≈ (cos-10.4°,
                // sin-10.4°); cos45.4° ≈ 0.701. `n` is the dot product ×1000.
                let n = dx * 984 - dy * 180;
                if n > 0 && n * n > 701 * 701 * d2 {
                    continue;
                }
                self.put_pixel((cx + dx) as u64, (cy + dy) as u64, arc_c);
            }
        }
        // Round line-caps at the two arc ends (35° and 304.6° in screen coords),
        // in the ring colour, matching stroke-linecap="round".
        let cap = half.max(1);
        self.fill_disc(cx + r * 819 / 1000, cy + r * 574 / 1000, cap, arc_c); // 35°
        self.fill_disc(cx + r * 562 / 1000, cy - r * 827 / 1000, cap, arc_c); // 304.6°
        // The synapse node: a filled disc at the centre.
        let nr = (r * 32 / 100).max(2);
        self.fill_disc(cx, cy, nr, node_c);
    }

    /// Paint the boot splash: the brand mark, "ChittiOS", and a tagline, centred
    /// on the canvas. Shown briefly at boot (see [`show_splash`]).
    fn draw_splash(&self) {
        self.fill_rect(0, 0, self.width, self.height, self.theme.screen_bg);
        let r = (self.height / 7).max(24);
        let cy = self.height * 2 / 5;
        // Ring in terracotta (accent), node in cream (chat_fg) — see the SVG.
        self.draw_logo(self.width / 2, cy, r, self.theme.accent, self.theme.chat_fg);
        let name = "ChittiOS";
        let nx = self.width / 2 - (name.len() as u64 * self.cw()) / 2;
        self.draw_str(nx, cy + r + r / 2, name, self.theme.accent, self.theme.screen_bg);
        let tag = "an agentic operating system";
        let tx = self.width / 2 - (tag.len() as u64 * self.cw()) / 2;
        self.draw_str(tx, cy + r + r / 2 + self.ch() + 6, tag, self.theme.title_dim, self.theme.screen_bg);
    }

    /// Whether the action (right) pane holds keyboard focus: always while the
    /// editor owns it, by toggle (`focus_toggle` / click) while it shows ktrace.
    fn action_focused(&self) -> bool {
        match self.right {
            RightMode::Editor => true,
            RightMode::Closed => false,
            // ktrace / top / surface: chat keeps focus by default so you can keep
            // typing; Ctrl+Tab / a click can move focus to the pane.
            RightMode::Ktrace | RightMode::Top | RightMode::Todos | RightMode::Audio | RightMode::Surface(_) => {
                self.focus_action
            }
        }
    }

    /// Full repaint: background, chat pane (content re-rendered from its grid),
    /// the action (right) pane if open, caret, status bar.
    ///
    /// Parked panes (`w == 0`, fullscreen) are skipped entirely — their content
    /// is preserved in memory via [`Pane::take_content`] and restored on unpark.
    fn redraw(&self) {
        // Paint only the background *gutters* (margins + the gap between panes),
        // never a full-screen clear — the panes are painted over their own areas
        // below, so their content is never flashed to background. This is what
        // makes opening/closing the action pane not flicker the whole screen.
        self.paint_gutters();
        // Drop shadows sit in the gutters (right/bottom bands of each pane).
        if self.chat.w > 0 {
            self.drop_shadow(self.chat.x, self.chat.y, self.chat.w, self.chat.h);
            self.fill_rect(self.chat.x, self.chat.y, self.chat.w, self.chat.h, self.chat.bg);
            self.draw_frame(&self.chat, !self.action_focused());
            self.render_view(&self.chat);
            self.draw_composer(); // includes suggest popup when open
        }
        if self.right != RightMode::Closed && self.logs.w > 0 {
            self.drop_shadow(self.logs.x, self.logs.y, self.logs.w, self.logs.h);
            self.fill_rect(self.logs.x, self.logs.y, self.logs.w, self.logs.h, self.logs.bg);
            // The header is a tmux-style tab bar (drawn by draw_tab_bar), so the
            // frame is drawn with an empty title.
            self.draw_frame_titled(&self.logs, self.action_focused(), "");
            self.draw_tab_bar();
            self.draw_close_btn();
            // The editor + /top + audio repaint their own interiors (from the
            // shell, on switch + idle tick); ktrace re-renders from the grid.
            if self.right == RightMode::Ktrace {
                self.render_view(&self.logs);
            }
        }
        // Grid caret only when there is no composer; otherwise the caret is in
        // the input box (or absent while a reply streams).
        if self.chat.w > 0 && self.chat.view == 0 {
            self.caret_draw(&self.chat);
        }
        self.draw_status();
    }

    /// Fill just the screen-background gutters: the top/bottom strips and the
    /// left/right margins + the gap between the panes. Everything the panes
    /// cover is left untouched (painted over directly), so `redraw` never blanks
    /// the whole screen.
    fn paint_gutters(&self) {
        let bg = self.theme.screen_bg;
        let (by, bh) = (self.chat.y, self.chat.h);
        // Top strip (above the pane band) + bottom strip (down to the status bar).
        self.fill_rect(0, 0, self.width, by, bg);
        let below = by + bh;
        let status_top = self.height.saturating_sub(self.ch() + 8);
        self.fill_rect(0, below, self.width, status_top.saturating_sub(below), bg);
        // Horizontal gutters within the pane band. Order the two panes by x
        // (the layout may put chat on the right when swapped).
        let (mut a, mut b) = ((self.chat.x, self.chat.x + self.chat.w), (0u64, 0u64));
        let two = self.right != RightMode::Closed;
        if two {
            b = (self.logs.x, self.logs.x + self.logs.w);
            if a.0 > b.0 {
                core::mem::swap(&mut a, &mut b);
            }
        }
        self.fill_rect(0, by, a.0, bh, bg); // left margin
        if two {
            self.fill_rect(a.1, by, b.0.saturating_sub(a.1), bh, bg); // gap between panes
            self.fill_rect(b.1, by, self.width.saturating_sub(b.1), bh, bg); // right margin
        } else {
            self.fill_rect(a.1, by, self.width.saturating_sub(a.1), bh, bg); // right margin
        }
    }

    /// Repaint **only the action pane** for a tab switch (geometry unchanged):
    /// clear its interior once for the new tab, redraw its frame + tab bar, and
    /// re-render ktrace from its grid. The chat pane and the whole background are
    /// left untouched — so switching tabs never flickers the rest of the screen.
    /// The active tab's dynamic interior (top/audio/image/editor) is repainted by
    /// the shell right after (`repaint_active_tab`).
    fn repaint_action(&mut self) {
        if self.right == RightMode::Closed {
            return;
        }
        self.cursor_restore();
        // Update the chat frame's active state (border colour) without touching
        // its content — cheap, no blank.
        self.draw_frame(&self.chat, !self.action_focused());
        self.fill_rect(self.logs.ix, self.logs.iy, self.logs.cols * self.logs.cw, self.logs.rows * self.logs.ch, self.logs.bg);
        self.draw_frame_titled(&self.logs, self.action_focused(), "");
        self.draw_tab_bar();
        self.draw_close_btn();
        if self.right == RightMode::Ktrace {
            self.render_view(&self.logs);
        }
        self.cursor_overlay();
    }

    /// Draw the `[x]` close button at the top-right of the action pane title.
    fn draw_close_btn(&self) {
        let (x, y, _, _) = self.close_btn();
        self.draw_str(x, y, "[x]", (230, 120, 120), self.logs.bg);
    }

    /// Per-tab header layout: `(mode, x_pixel, width_pixels)` for each open tab,
    /// laid out left-to-right on the action pane's title row. Used by both the
    /// tab-bar renderer and the click hit-test so they never disagree.
    fn tab_layout(&self) -> Vec<(RightMode, u64, u64)> {
        let cw = self.cw();
        let mut x = self.logs.x + BORDER + PAD;
        let mut out = Vec::with_capacity(self.tabs.len());
        for &m in &self.tabs {
            let w = (tab_label(m).len() as u64 + 1) * cw; // label + trailing space
            out.push((m, x, w));
            x += w + cw; // one cell gap between tabs
        }
        out
    }

    /// Draw the tab bar on the action pane's title row: active tab in accent
    /// with a `▸` marker, the rest dim. Stops before the `[x]` close button.
    fn draw_tab_bar(&self) {
        let ty = self.logs.y + BORDER + 4;
        let (close_x, ..) = self.close_btn();
        // Clear the tab-bar row (up to the close button) first, so switching
        // tabs leaves no stale glyphs from a longer previous label.
        let x0 = self.logs.x + BORDER + PAD;
        self.fill_rect(x0, ty, close_x.saturating_sub(x0), self.ch(), self.logs.bg);
        for (i, (m, x, w)) in self.tab_layout().into_iter().enumerate() {
            if x + w >= close_x {
                break; // ran into the close button; overflow tabs are hidden
            }
            let is_active = i == self.active;
            let fg = if is_active { self.theme.title_active } else { self.theme.title_dim };
            let mut lx = x;
            if is_active {
                lx = self.draw_str(lx, ty, ">", self.theme.accent, self.logs.bg);
            }
            self.draw_str(lx, ty, tab_label(m), fg, self.logs.bg);
        }
    }

    /// A labelled horizontal usage bar filled proportional to `pct` (0..=100)
    /// and coloured green/amber/red by load, over background `bg`. Returns the
    /// y below the bar.
    fn usage_bar_bg(&self, x: u64, y: u64, w: u64, label: &str, pct: u64, detail: &str, bg: Rgb) -> u64 {
        let cw = self.cw();
        let ch = self.ch();
        let lab_w = 7 * cw; // fixed label column
        self.draw_str(x, y, label, self.theme.chat_fg, bg);
        let bx = x + lab_w;
        // Reserve gap(1) + detail(11) + margin(1) = 13 cells after the bar, so
        // the padded detail text never runs past the pane's right edge/border.
        let bw = w.saturating_sub(lab_w + 13 * cw);
        let bh = ch;
        // Border (static — re-stroking the same pixels never blanks).
        self.rect_outline(bx, y, bw, bh, 1, self.theme.border_dim);
        let p = pct.min(100);
        let inner = bw.saturating_sub(2);
        let fill = inner * p / 100;
        let color = if p < 60 {
            (126, 214, 150) // green
        } else if p < 85 {
            (240, 200, 120) // amber
        } else {
            (255, 106, 110) // red
        };
        // Repaint in place: colour the filled span, then background the rest —
        // the coloured region is never blanked to bg first, so no flicker.
        if fill > 0 {
            self.fill_rect(bx + 1, y + 1, fill, bh.saturating_sub(2), color);
        }
        if inner > fill {
            self.fill_rect(bx + 1 + fill, y + 1, inner - fill, bh.saturating_sub(2), self.theme.chat_bg);
        }
        // Detail (e.g. "512M/6.0G") after the bar — ellipsize + pad so it never
        // overflows the pane and shrinking values leave no residue.
        let detail_x = bx + bw + cw;
        let avail = ((x + w).saturating_sub(detail_x) / cw) as usize;
        let d = crate::textsel::fit_width(detail, avail);
        self.draw_str(detail_x, y, &d, self.theme.title_dim, bg);
        y + ch + ch / 3
    }

    /// Compact htop-style meter: `1 [████░░░░] 34%` — short label, thin bar
    /// with green→amber→red fill, fixed-width detail. Returns next y.
    fn htop_meter(&self, x: u64, y: u64, w: u64, label: &str, pct: u64, detail: &str, bg: Rgb) -> u64 {
        let cw = self.cw();
        let ch = self.ch();
        let lab = crate::textsel::fit_width(label, 4);
        self.draw_str(x, y, &lab, self.theme.chat_fg, bg);
        let bx = x + 5 * cw;
        // Reserve space for detail (already padded by caller, ~4–11 cells).
        let det_cols = detail.chars().count().max(4) as u64 + 1;
        let bw = w.saturating_sub(5 * cw + det_cols * cw);
        let bh = ch.saturating_sub(2).max(ch * 3 / 4);
        let by = y + (ch.saturating_sub(bh)) / 2;
        // Track outline + fill (htop-like).
        self.rect_outline(bx, by, bw, bh, 1, self.theme.border_dim);
        let p = pct.min(100);
        let inner = bw.saturating_sub(2);
        let fill = inner * p / 100;
        // Gradient-ish: green low, amber mid, red high — solid for simplicity.
        let color = if p < 50 {
            (80, 200, 120) // htop green
        } else if p < 75 {
            (220, 180, 60) // yellow
        } else if p < 90 {
            (230, 120, 50) // orange
        } else {
            (230, 60, 60) // red
        };
        if fill > 0 {
            self.fill_rect(bx + 1, by + 1, fill, bh.saturating_sub(2), color);
        }
        if inner > fill {
            self.fill_rect(
                bx + 1 + fill,
                by + 1,
                inner - fill,
                bh.saturating_sub(2),
                self.theme.chat_bg,
            );
        }
        let detail_x = bx + bw + cw;
        let avail = ((x + w).saturating_sub(detail_x) / cw) as usize;
        let d = crate::textsel::fit_width(detail, avail);
        self.draw_str(detail_x, y, &d, self.theme.title_dim, bg);
        y + ch
    }

    /// Shorthand: [`draw_str`] with an explicit background (the `/top` panel
    /// draws over the logs-pane background, not the screen background).
    fn draw_str_bg(&self, x: u64, y: u64, s: &str, fg: Rgb, bg: Rgb) -> u64 {
        self.draw_str(x, y, s, fg, bg)
    }
}

/// One row in the `/top` process table (kernel tasks / agents / services).
pub struct TopTask<'a> {
    pub id: u64,
    pub name: &'a str,
    pub state: &'a str,
    /// Optional tree prefix (`|- `, `` ` - ``) for a light process-tree look.
    pub tree: &'a str,
}

/// A snapshot for [`draw_top`] — the `/top` dashboard's inputs, gathered by the
/// shell so the framebuffer layer stays free of `mm`/`smp` coupling.
pub struct TopView<'a> {
    /// Per-core busy percentage (index = core id).
    pub cores: &'a [u64],
    pub cores_online: u64,
    pub ram_used: u64,
    pub ram_total: u64,
    pub heap_used: u64,
    pub heap_total: u64,
    pub model_bytes: u64,
    pub uptime: &'a str,
    pub arch: &'a str,
    pub allocs: u64,
    pub datetime: &'a str,
    /// Process table rows (already sorted by the shell).
    pub tasks: &'a [TopTask<'a>],
    pub tasks_total: u64,
    pub tasks_running: u64,
    /// Average core utilisation (0..=100), shown as a load stand-in.
    pub load_pct: u64,
    pub net_up: bool,
    pub model_name: &'a str,
}

/// Render the `/top` dashboard in an **htop-like** layout into the action pane:
/// dual-column header (CPU/Mem meters | Tasks/Load/Uptime), a process table,
/// and an F-key footer. No-op unless the pane is in [`RightMode::Top`].
/// Refreshed ~1 Hz from the shell idle tick.
pub fn draw_top(v: &TopView) {
    SCREEN.with(|slot| {
        let Some(sc) = slot else { return };
        if sc.right != RightMode::Top {
            return;
        }
        sc.cursor_restore();
        sc.cur_vis = false;
        let ch = sc.ch();
        let cw = sc.cw();
        let (px, iy, iw) = (sc.logs.ix, sc.logs.iy, sc.logs.cols * sc.logs.cw);
        let bottom = iy + sc.logs.rows * sc.logs.ch;
        // NO full-interior clear: overwrite in place (padded strings + self-
        // filling bars) so the 1 Hz refresh does not flicker.
        let bg = sc.logs.bg;
        let mib = 1024 * 1024;
        let gib = 1024 * mib;
        let fmt = |b: u64| -> String {
            if b >= gib {
                alloc::format!("{}.{}G", b / gib, (b % gib) * 10 / gib)
            } else if b >= mib {
                alloc::format!("{}M", b / mib)
            } else {
                alloc::format!("{}K", b / 1024)
            }
        };
        let cols = (iw / cw).max(1) as usize;
        let gap = 2 * cw;
        // Two-column header when the pane is wide enough (≥ 48 cells).
        let two_col = cols >= 48;
        let left_w = if two_col { (iw - gap) / 2 } else { iw };
        let right_x = px + left_w + gap;
        let right_w = iw.saturating_sub(left_w + gap);

        // --- left: CPU + Mem + Heap meters (htop style) ---------------------
        let mut y_l = iy;
        // Compact meters: `1 [████░░░░] 34%` — 1-based like htop.
        for (i, &pct) in v.cores.iter().enumerate() {
            if y_l + ch > bottom {
                break;
            }
            let online = (i as u64) < v.cores_online;
            let lab = alloc::format!("{}", i + 1);
            let detail = if online {
                crate::textsel::fit_width(&alloc::format!("{}%", pct.min(100)), 4)
            } else {
                crate::textsel::fit_width("--%", 4)
            };
            y_l = sc.htop_meter(px, y_l, left_w, &lab, if online { pct } else { 0 }, &detail, bg);
        }
        let ram_pct = if v.ram_total > 0 { v.ram_used * 100 / v.ram_total } else { 0 };
        let heap_pct = if v.heap_total > 0 { v.heap_used * 100 / v.heap_total } else { 0 };
        if y_l + ch <= bottom {
            y_l = sc.htop_meter(
                px,
                y_l,
                left_w,
                "Mem",
                ram_pct,
                &crate::textsel::fit_width(&alloc::format!("{}/{}", fmt(v.ram_used), fmt(v.ram_total)), 11),
                bg,
            );
        }
        if y_l + ch <= bottom {
            y_l = sc.htop_meter(
                px,
                y_l,
                left_w,
                "Heap",
                heap_pct,
                &crate::textsel::fit_width(&alloc::format!("{}/{}", fmt(v.heap_used), fmt(v.heap_total)), 11),
                bg,
            );
        }
        // Model as a third "resource" bar (htop's Swp analog for our OS).
        if y_l + ch <= bottom && v.model_bytes > 0 {
            let model_pct = if v.ram_total > 0 {
                (v.model_bytes * 100 / v.ram_total).min(100)
            } else {
                0
            };
            y_l = sc.htop_meter(
                px,
                y_l,
                left_w,
                "Mdl",
                model_pct,
                &crate::textsel::fit_width(&fmt(v.model_bytes), 11),
                bg,
            );
        }

        // --- right: Tasks / Load / Uptime (htop info column) ---------------
        let mut y_r = iy;
        if two_col {
            let rcols = (right_w / cw).max(1) as usize;
            // Load as "N.NN" from average core % (htop's loadavg stand-in).
            let load_i = v.load_pct.min(999);
            let info = [
                alloc::format!("Tasks: {}, {} running", v.tasks_total, v.tasks_running),
                alloc::format!("Load average: {}.{:02}", load_i / 100, load_i % 100),
                alloc::format!("CPU avg: {}%", v.load_pct.min(100)),
                alloc::format!("Uptime: {}", v.uptime),
                alloc::format!("{}", v.datetime),
                alloc::format!("Arch: {}  ({} cores)", v.arch, v.cores_online),
                alloc::format!("Network: {}", if v.net_up { "up" } else { "down" }),
                alloc::format!(
                    "Model: {}",
                    crate::textsel::ellipsize(v.model_name, rcols.saturating_sub(8))
                ),
                alloc::format!("Heap allocs: {}", v.allocs),
            ];
            for s in &info {
                if y_r + ch > bottom {
                    break;
                }
                sc.draw_str_bg(
                    right_x,
                    y_r,
                    &crate::textsel::fit_width(s, rcols),
                    sc.theme.logs_fg,
                    bg,
                );
                y_r += ch;
            }
        }

        // Header block ends at the taller of the two columns.
        let mut y = y_l.max(y_r) + ch / 2;
        if y + 3 * ch > bottom {
            // Still paint footer if almost full.
            sc.cursor_overlay();
            return;
        }

        // --- process table header (htop green bar) -------------------------
        let hdr_bg = (0, 140, 80); // classic htop green
        let hdr_fg = (0, 0, 0);
        let hdr = if cols >= 60 {
            "  PID  STATE    NAME / COMMAND"
        } else if cols >= 40 {
            "  PID  STATE  COMMAND"
        } else {
            "  PID  COMMAND"
        };
        sc.fill_rect(px, y, iw, ch, hdr_bg);
        sc.draw_str_bg(px, y, &crate::textsel::fit_width(hdr, cols), hdr_fg, hdr_bg);
        y += ch;

        // --- process rows --------------------------------------------------
        let footer_h = ch; // reserve one line for F-keys
        let mut first_running_painted = false;
        for t in v.tasks {
            if y + ch + footer_h > bottom {
                break;
            }
            let is_run = t.state == "running";
            let sel = is_run && !first_running_painted;
            if sel {
                first_running_painted = true;
            }
            let row_bg = if sel {
                (0, 180, 200) // htop cyan selection
            } else {
                bg
            };
            let row_fg = if sel { (0, 0, 0) } else { sc.theme.chat_fg };
            let state_fg = if sel {
                (0, 0, 0)
            } else {
                match t.state {
                    "running" => (126, 214, 150),
                    "ready" => (240, 200, 120),
                    "parked" => sc.theme.title_dim,
                    "dead" => (255, 106, 110),
                    _ => sc.theme.logs_fg,
                }
            };
            sc.fill_rect(px, y, iw, ch, row_bg);
            // Columns: PID (5) STATE (8) tree+name
            let pid = crate::textsel::fit_width(&alloc::format!("{}", t.id), 5);
            let st = crate::textsel::fit_width(t.state, 8);
            let name_cols = cols.saturating_sub(5 + 1 + 8 + 1);
            let name = crate::textsel::fit_width(
                &alloc::format!("{}{}", t.tree, t.name),
                name_cols,
            );
            let mut xx = px;
            xx = sc.draw_str(xx, y, &pid, row_fg, row_bg);
            xx = sc.draw_str(xx, y, " ", row_fg, row_bg);
            xx = sc.draw_str(xx, y, &st, state_fg, row_bg);
            xx = sc.draw_str(xx, y, " ", row_fg, row_bg);
            let _ = sc.draw_str(xx, y, &name, row_fg, row_bg);
            y += ch;
        }
        // Blank any leftover process-area rows so a shrinking task list
        // leaves no ghost lines.
        let blank = crate::textsel::fit_width("", cols);
        while y + ch + footer_h <= bottom {
            sc.draw_str_bg(px, y, &blank, bg, bg);
            y += ch;
        }

        // --- F-key footer (htop style) -------------------------------------
        let foot_y = bottom.saturating_sub(ch);
        let foot_bg = sc.theme.status_bg;
        sc.fill_rect(px, foot_y, iw, ch, foot_bg);
        // Number in reverse / label dim — approximate with accent digits.
        let keys = [
            ("F1", "Help"),
            ("F2", "Setup"),
            ("F3", "Search"),
            ("F4", "Filter"),
            ("F5", "Tree"),
            ("F6", "Sort"),
            ("F9", "Kill"),
            ("F10", "Quit"),
        ];
        let mut fx = px;
        for (k, lab) in keys {
            if fx + (k.len() + lab.len() + 2) as u64 * cw > px + iw {
                break;
            }
            fx = sc.draw_str(fx, foot_y, k, foot_bg, sc.theme.accent); // reverse-ish
            fx = sc.draw_str(fx, foot_y, lab, sc.theme.logs_fg, foot_bg);
            fx += cw; // gap
        }
        // Right-align a short quit hint if room.
        let hint = " /close ";
        if cols > 20 {
            let hx = px + iw.saturating_sub(hint.len() as u64 * cw);
            if hx > fx {
                sc.draw_str(hx, foot_y, hint, sc.theme.title_dim, foot_bg);
            }
        }
        sc.cursor_overlay();
    });
}

/// A snapshot of the background audio player, for [`draw_audio`].
pub struct AudioView<'a> {
    pub name: &'a str,
    pub pos_ms: u64,
    pub total_ms: u64,
    pub rate: u32,
    pub playing: bool,
    pub paused: bool,
    /// Peak envelope `0..=255` for the wave visualizer (see `audio::waveform_peaks`).
    pub peaks: &'a [u8],
    /// Software volume percent (`0..=100`) and mute (from `sound::volume`/`muted`).
    pub volume: u32,
    pub muted: bool,
}

/// Paint the audio-player tab in the **same HUD layout as the video player**:
/// a centre **wave visualizer** (played = accent, remaining = dim) plus a
/// bottom control strip (status line, scrubber, shortcut hints). No-op unless
/// the audio tab is active. Called ~4 Hz from the shell while the tab is on top.
pub fn draw_audio(v: &AudioView) {
    SCREEN.with(|slot| {
        let Some(sc) = slot else { return };
        if sc.right != RightMode::Audio {
            return;
        }
        sc.cursor_restore();
        sc.cur_vis = false;
        let ch = sc.ch();
        let cw = sc.cw();
        let (px, py) = (sc.logs.ix, sc.logs.iy);
        let (pw, ph) = (sc.logs.cols * sc.logs.cw, sc.logs.rows * sc.logs.ch);
        let bg = sc.logs.bg;
        // Same HUD height as the video player so the two tabs feel identical.
        let barh = ch * 4 + ch / 2;
        let by = py + ph.saturating_sub(barh);

        // --- centre: waveform visualizer ---------------------------------
        // Compact band (about 1/3 of the content height, capped) centred in
        // the area above the HUD — not full-pane tall.
        let content_h = by.saturating_sub(py);
        let wave_h = (content_h / 3).clamp(ch * 2, ch * 5);
        let wave_top = py + content_h.saturating_sub(wave_h) / 2;
        let wave_x = px + cw * 2;
        let wave_w = pw.saturating_sub(4 * cw).max(1);
        // Clear the full content band once so a taller previous paint (or
        // leftover glyphs) never sticks around when the wave shrinks.
        sc.fill_rect(px, py, pw, content_h.saturating_sub(1), bg);
        let mid = wave_top + wave_h / 2;
        let n_peaks = v.peaks.len().max(1);
        let play_x = if v.total_ms > 0 {
            (wave_w * v.pos_ms.min(v.total_ms) / v.total_ms).min(wave_w.saturating_sub(1))
        } else {
            0
        };
        for col in 0..wave_w {
            let pi = ((col as usize) * n_peaks) / (wave_w as usize).max(1);
            let peak = v.peaks.get(pi).copied().unwrap_or(0) as u64;
            // Half-height bar (mirrored above/below centre); min 1px when energy.
            let half = if peak == 0 {
                0
            } else {
                ((wave_h / 2 - 1) * peak / 255).max(1)
            };
            let color = if col <= play_x { sc.theme.accent } else { sc.theme.title_dim };
            // Clear the column then draw the bar.
            sc.fill_rect(wave_x + col, wave_top, 1, wave_h, bg);
            if half > 0 {
                sc.fill_rect(wave_x + col, mid.saturating_sub(half), 1, half * 2, color);
            } else {
                // Quiet: a 1px centre tick so the track silhouette stays visible.
                sc.fill_rect(wave_x + col, mid, 1, 1, sc.theme.sep_dim);
            }
        }
        // Playhead: thin bright line at the current position.
        sc.fill_rect(wave_x + play_x, wave_top, 2.max(sc.scale), wave_h, sc.theme.chat_fg);

        // --- bottom HUD (mirrors `draw_video_status`) --------------------
        sc.fill_rect(px, by, pw, 1, sc.theme.accent); // top hairline
        let cols = (pw / cw).saturating_sub(2).max(4) as usize;
        let fit = |s: &str| crate::textsel::fit_width(s, cols);
        let mmss = |ms: u64| alloc::format!("{}:{:02}", ms / 60000, ms % 60000 / 1000);
        let state = if v.paused {
            "||"
        } else if v.playing {
            ">"
        } else {
            "="
        };
        let time = alloc::format!("{} / {}", mmss(v.pos_ms), mmss(v.total_ms));
        let vol = if v.muted {
            String::from("muted")
        } else {
            alloc::format!("vol {}%", v.volume.min(100))
        };
        // Drop less-critical fields as the pane narrows so the line always fits.
        let candidates = [
            alloc::format!("{} {}  {}  {}", state, v.name, time, vol),
            alloc::format!("{} {}  {}", state, v.name, time),
            alloc::format!("{} {}", state, v.name),
            alloc::format!("{} {}", state, crate::textsel::ellipsize(v.name, cols.saturating_sub(3).max(1))),
        ];
        let line1 = candidates
            .into_iter()
            .find(|s| s.chars().count() <= cols)
            .unwrap_or_else(|| crate::textsel::ellipsize(&alloc::format!("{} {}", state, v.name), cols));
        let mut y = by + ch / 3;
        sc.draw_str_bg(px + cw, y, &fit(&line1), sc.theme.accent, bg);
        y += ch + ch / 4;
        // Scrubber in the control strip (video-style), not the main area.
        let track_x = px + cw;
        let track_w = pw.saturating_sub(2 * cw);
        let filled = if v.total_ms > 0 {
            (track_w * v.pos_ms.min(v.total_ms) / v.total_ms).min(track_w)
        } else {
            0
        };
        sc.fill_rect(track_x, y + ch / 3, track_w, ch / 4, sc.theme.title_dim);
        sc.fill_rect(track_x, y + ch / 3, filled, ch / 4, sc.theme.accent);
        y += ch + ch / 4;
        // Shortcut hints: wrap; drop tokens that can't fit even alone.
        let hints = [
            "space play/pause",
            "<-/-> seek",
            "up/dn volume",
            "0 restart",
            "m mute",
            "Ctrl+C stop",
        ];
        let sep = "   ";
        let mut linebuf = String::new();
        let hud_bottom = py + ph;
        for h in hints {
            if h.chars().count() > cols {
                continue; // too wide even alone — hide
            }
            let cand = if linebuf.is_empty() {
                String::from(h)
            } else {
                alloc::format!("{}{}{}", linebuf, sep, h)
            };
            if cand.chars().count() > cols && !linebuf.is_empty() {
                if y + ch > hud_bottom {
                    break; // no room for another hint row
                }
                sc.draw_str_bg(px + cw, y, &fit(&linebuf), sc.theme.logs_fg, bg);
                y += ch;
                linebuf = String::from(h);
            } else {
                linebuf = cand;
            }
        }
        if !linebuf.is_empty() && y + ch <= hud_bottom + ch {
            sc.draw_str_bg(px + cw, y, &fit(&linebuf), sc.theme.logs_fg, bg);
        }
        sc.cursor_overlay();
    });
}

/// Overlay the video player's control/status bar along the bottom of the video
/// surface pane: playback state, mm:ss / mm:ss, frame counter, mute, a scrubber,
/// and the key-shortcut hints. Drawn *after* the frame blit (present_surface
/// clears the pane each present), so it sits on top like a real player's HUD.
/// No-op unless the video surface tab is active.
///
/// `fps` is the instantaneous / smoothed decode+present FPS (0 = unknown / paused).
#[allow(clippy::too_many_arguments)]
pub fn draw_video_status(
    name: &str,
    playing: bool,
    muted: bool,
    has_audio: bool,
    frame: usize,
    frames: usize,
    pos_ms: u64,
    total_ms: u64,
    volume: u32,
    fps: u32,
) {
    SCREEN.with(|slot| {
        let Some(sc) = slot else { return };
        if sc.right != RightMode::Surface(VIDEO_SURFACE) {
            return;
        }
        sc.cursor_restore();
        sc.cur_vis = false;
        let ch = sc.ch();
        let cw = sc.cw();
        let (px, py) = (sc.logs.ix, sc.logs.iy);
        let (pw, ph) = (sc.logs.cols * sc.logs.cw, sc.logs.rows * sc.logs.ch);
        // Reserved HUD strip (below the video frame). Fill the whole strip once
        // so time/fps string length changes never leave glyph trails; the strip
        // is small (~4 lines) so this is cheap and does not flash the picture.
        let bg = sc.logs.bg;
        let barh = ch * 4 + ch / 2;
        let by = py + ph.saturating_sub(barh);
        sc.fill_rect(px, by, pw, barh, bg);
        sc.fill_rect(px, by, pw, 1, sc.theme.accent); // top hairline
        // Usable text width in whole glyph cells, with a one-cell left margin.
        let cols = (pw / cw).saturating_sub(2).max(4) as usize;
        let fit = |s: &str| crate::textsel::fit_width(s, cols);
        let mmss = |ms: u64| alloc::format!("{}:{:02}", ms / 60000, ms % 60000 / 1000);
        let state = if playing { ">" } else { "||" };
        let vol = if !has_audio {
            String::from("no audio")
        } else if muted {
            String::from("muted")
        } else {
            alloc::format!("vol {}%", volume.min(100))
        };
        let fps_s = if fps > 0 {
            alloc::format!("{} fps", fps)
        } else {
            String::from("-- fps")
        };
        // Drop fields as the pane narrows so the status line never overflows.
        let time = alloc::format!("{} / {}", mmss(pos_ms), mmss(total_ms));
        let fr = alloc::format!("{}/{}", frame, frames);
        let candidates = [
            alloc::format!("{} {}  {}  {}  {}  {}", state, name, time, fr, fps_s, vol),
            alloc::format!("{} {}  {}  {}  {}", state, name, time, fps_s, vol),
            alloc::format!("{} {}  {}  {}", state, name, time, fps_s),
            alloc::format!("{} {}  {}", state, name, time),
            alloc::format!("{} {}", state, name),
            alloc::format!("{} {}", state, crate::textsel::ellipsize(name, cols.saturating_sub(3).max(1))),
        ];
        let line1 = candidates
            .into_iter()
            .find(|s| s.chars().count() <= cols)
            .unwrap_or_else(|| crate::textsel::ellipsize(&alloc::format!("{} {}", state, name), cols));
        let mut y = by + ch / 3;
        sc.draw_str_bg(px + cw, y, &fit(&line1), sc.theme.accent, bg);
        y += ch + ch / 4;
        // Scrubber: a full-width track with a filled portion for progress
        // (self-filling — the whole track is overwritten each refresh).
        let track_x = px + cw;
        let track_w = pw.saturating_sub(2 * cw);
        let filled = if total_ms > 0 { (track_w * pos_ms.min(total_ms) / total_ms).min(track_w) } else { 0 };
        sc.fill_rect(track_x, y + ch / 3, track_w, ch / 4, sc.theme.title_dim);
        sc.fill_rect(track_x, y + ch / 3, filled, ch / 4, sc.theme.accent);
        y += ch + ch / 4;
        // Shortcuts: wrap; hide tokens that don't fit; stop when HUD is full.
        let hints = ["space play/pause", "<-/-> seek", "up/dn volume", "0 restart", "m mute", "Ctrl+C stop"];
        let sep = "   ";
        let mut linebuf = String::new();
        let hud_bottom = py + ph;
        for h in hints {
            if h.chars().count() > cols {
                continue;
            }
            let cand = if linebuf.is_empty() { String::from(h) } else { alloc::format!("{}{}{}", linebuf, sep, h) };
            if cand.chars().count() > cols && !linebuf.is_empty() {
                if y + ch > hud_bottom {
                    break;
                }
                sc.draw_str_bg(px + cw, y, &fit(&linebuf), sc.theme.logs_fg, bg);
                y += ch;
                linebuf = String::from(h);
            } else {
                linebuf = cand;
            }
        }
        if !linebuf.is_empty() && y + ch <= hud_bottom + ch {
            sc.draw_str_bg(px + cw, y, &fit(&linebuf), sc.theme.logs_fg, bg);
        }
        sc.cursor_overlay();
    });
}

static SCREEN: Locked<Option<Screen>> = Locked::new(None);

// --- modal overlay (approval / input dialogs) ---------------------------

/// Which modal control the mouse hit, for [`modal_hit`].
#[derive(Clone, Copy, PartialEq)]
pub enum ModalHit {
    None,
    Yes,
    No,
    Ok,
    /// Commands-browser close `[x]` (slot 0 reused when that modal is up).
    Close,
}

/// Pixel rects of the modal's clickable controls: `[yes, no, ok]`. Set when a
/// modal is drawn, read by [`modal_hit`] for mouse routing. Zero-size = absent.
static MODAL_RECTS: Locked<[(u64, u64, u64, u64); 3]> = Locked::new([(0, 0, 0, 0); 3]);

/// True while a modal overlays the panes: upkeep ticks running under it (long
/// compute pumps `shell::upkeep`) must not blink the pane caret into the box.
static MODAL_ON: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

fn in_rect(x: u64, y: u64, r: (u64, u64, u64, u64)) -> bool {
    r.2 != 0 && x >= r.0 && x < r.0 + r.2 && y >= r.1 && y < r.1 + r.3
}

/// Hit-test the modal controls against a click at `(x, y)`.
pub fn modal_hit(x: u64, y: u64) -> ModalHit {
    let r = MODAL_RECTS.with(|m| *m);
    // Commands browser stashes Close in slot 0 and leaves 1 empty; confirm uses
    // Yes/No in 0/1. Disambiguate: if slot 1 is empty and slot 0 is set, it's Close.
    if in_rect(x, y, r[0]) {
        if r[1] == (0, 0, 0, 0) && r[2] == (0, 0, 0, 0) {
            ModalHit::Close
        } else {
            ModalHit::Yes
        }
    } else if in_rect(x, y, r[1]) {
        ModalHit::No
    } else if in_rect(x, y, r[2]) {
        ModalHit::Ok
    } else {
        ModalHit::None
    }
}

/// One visible row for [`draw_commands_browser`].
pub enum CommandsRow<'a> {
    Header(&'a str),
    Item {
        title: &'a str,
        slash: &'a str,
        shortcut: &'a str,
        selected: bool,
    },
}

impl Screen {
    /// Draw a centred modal box and return its interior text origin + width in
    /// cells `(ix, iy, cols)`. Dims the screen isn't done (kept cheap); the box
    /// simply overpaints the middle of the canvas.
    /// The modal content width in cells (roomy but bounded), so callers can wrap
    /// their text to it *before* sizing the box height.
    fn modal_cols(&self) -> u64 {
        ((self.width / self.cw()) * 3 / 5).clamp(28, 56)
    }

    fn modal_box(&self, title: &str, rows: u64) -> (u64, u64, u64) {
        let cw = self.cw();
        let ch = self.ch();
        let cols = self.modal_cols();
        let bw = cols * cw + 2 * (BORDER + PAD);
        let bh = (rows + 2) * ch + 2 * (BORDER + PAD);
        let bx = (self.width - bw) / 2;
        let by = (self.height - bh) / 2;
        self.drop_shadow(bx, by, bw, bh); // web-style elevation over the panes
        self.fill_rect(bx, by, bw, bh, self.theme.status_bg);
        self.rect_outline(bx, by, bw, bh, BORDER, self.theme.accent);
        let ix = bx + BORDER + PAD;
        let iy = by + BORDER + PAD;
        self.draw_str(ix, iy, title, self.theme.accent, self.theme.status_bg);
        self.fill_rect(ix, iy + ch + 2, cols * cw, 1, self.theme.sep_dim);
        (ix, iy + 2 * ch, cols)
    }

    /// Draw a labelled button at `(x, y)`, filled when `focused`; record its rect
    /// in `MODAL_RECTS[slot]` for mouse hit-testing. Returns the x just past it.
    fn modal_button(&self, x: u64, y: u64, label: &str, focused: bool, slot: usize) -> u64 {
        let cw = self.cw();
        let ch = self.ch();
        let w = (label.len() as u64 + 2) * cw;
        let (fg, bg) = if focused { (self.theme.status_bg, self.theme.accent) } else { (self.theme.accent, self.theme.status_bg) };
        self.fill_rect(x, y, w, ch, bg);
        self.rect_outline(x, y, w, ch, 1, self.theme.accent);
        self.draw_str(x + cw, y, label, fg, bg);
        MODAL_RECTS.with(|m| m[slot] = (x, y, w, ch));
        x + w + cw
    }
}

/// Draw an approval (yes/no) modal. `focus_yes` highlights the Yes button.
pub fn draw_confirm(title: &str, msg: &str, focus_yes: bool) {
    MODAL_ON.store(true, core::sync::atomic::Ordering::Relaxed);
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.cursor_restore();
            sc.cur_vis = false;
            MODAL_RECTS.with(|m| *m = [(0, 0, 0, 0); 3]);
            // Wrap first, then size the box to the wrapped line count + a gap +
            // the button row, so a long consent message never overflows.
            let lines = wrap(msg, sc.modal_cols() as usize);
            let (ix, iy, cols) = sc.modal_box(title, lines.len() as u64 + 2);
            let ch = sc.ch();
            let cw = sc.cw();
            let mut y = iy;
            for line in &lines {
                sc.draw_str(ix, y, line, sc.theme.chat_fg, sc.theme.status_bg);
                y += ch;
            }
            // Buttons on the RIGHT of the box, just below the message.
            let by = y + ch / 2;
            let btn_w = |label: &str| (label.len() as u64 + 2) * cw;
            let total = btn_w("Yes") + cw + btn_w("No");
            let start = ix + cols * cw - total;
            let x2 = sc.modal_button(start, by, "Yes", focus_yes, 0);
            sc.modal_button(x2, by, "No", !focus_yes, 1);
            sc.cursor_overlay();
        }
    });
}

/// Draw a text-input modal (masked = password dots). `caret_on` blinks the caret.
pub fn draw_input(title: &str, prompt: &str, buf: &str, masked: bool, caret_on: bool) {
    MODAL_ON.store(true, core::sync::atomic::Ordering::Relaxed);
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.cursor_restore();
            sc.cur_vis = false;
            MODAL_RECTS.with(|m| *m = [(0, 0, 0, 0); 3]);
            // 4 content rows: the prompt, the input field, and the OK button each
            // occupy a row and the inter-row gaps (ch/2 + a few px) need the
            // extra row so the button clears the bottom border.
            let (ix, iy, cols) = sc.modal_box(title, 4);
            let ch = sc.ch();
            let cw = sc.cw();
            sc.draw_str(ix, iy, prompt, sc.theme.title_dim, sc.theme.status_bg);
            // Input field: a framed row showing the (optionally masked) text.
            let fy = iy + ch + 4;
            sc.fill_rect(ix, fy, cols * cw, ch, sc.theme.chat_bg);
            sc.rect_outline(ix, fy, cols * cw, ch, 1, sc.theme.border_dim);
            let shown: String = if masked { core::iter::repeat('*').take(buf.chars().count()).collect() } else { buf.to_string() };
            let end = sc.draw_str(ix + cw / 2, fy, &shown, sc.theme.chat_fg, sc.theme.chat_bg);
            if caret_on {
                sc.fill_rect(end, fy, 2 * sc.scale, ch, sc.theme.accent);
            }
            let by = fy + ch + ch / 2;
            sc.modal_button(ix, by, "OK", true, 2);
            sc.cursor_overlay();
        }
    });
}

/// Draw the **Commands** browser modal (opened by `/help`): title + search
/// field, scrollable categorised list, scrollbar, footer hints.
///
/// `rows` is the **visible slice** (already scrolled). `query` is the search
/// box contents; `caret_on` blinks the search caret; `scroll`/`total` drive the
/// scrollbar thumb.
pub fn draw_commands_browser(
    query: &str,
    rows: &[CommandsRow<'_>],
    scroll: usize,
    total: usize,
    caret_on: bool,
) {
    MODAL_ON.store(true, core::sync::atomic::Ordering::Relaxed);
    SCREEN.with(|slot| {
        let Some(sc) = slot else { return };
        sc.cursor_restore();
        sc.cur_vis = false;
        MODAL_RECTS.with(|m| *m = [(0, 0, 0, 0); 3]);

        let cw = sc.cw();
        let ch = sc.ch();
        let bg = sc.theme.status_bg;
        let list_bg = sc.theme.chat_bg;
        // Roomier than the default confirm modal — ~ half the screen.
        let cols = ((sc.width / cw) * 5 / 10).clamp(36, 64);
        let view_rows = 12u64; // visible list lines
        let chrome_rows = 5u64; // title + search + gap + footer
        let rows_h = view_rows + chrome_rows;
        let bw = cols * cw + 2 * (BORDER + PAD);
        let bh = rows_h * ch + 2 * (BORDER + PAD) + 12;
        let bx = (sc.width - bw) / 2;
        let by = (sc.height - bh) / 2;
        sc.drop_shadow(bx, by, bw, bh);
        sc.fill_rect(bx, by, bw, bh, bg);
        sc.rect_outline(bx, by, bw, bh, BORDER, sc.theme.accent);

        let ix = bx + BORDER + PAD;
        let mut y = by + BORDER + PAD;
        let content_w = cols * cw;

        // Title + [x] close.
        sc.draw_str(ix, y, "Commands", sc.theme.accent, bg);
        let close = "[x]";
        let cx = ix + content_w - close.len() as u64 * cw;
        sc.draw_str(cx, y, close, sc.theme.title_dim, bg);
        MODAL_RECTS.with(|m| m[0] = (cx, y, close.len() as u64 * cw, ch));
        y += ch + 4;
        sc.fill_rect(ix, y, content_w, 1, sc.theme.sep_dim);
        y += 6;

        // Search field.
        sc.draw_str(ix, y, "search:", sc.theme.title_dim, bg);
        let field_x = ix + 8 * cw;
        let field_w = content_w.saturating_sub(8 * cw);
        sc.fill_rect(field_x, y, field_w, ch, list_bg);
        sc.rect_outline(field_x, y, field_w, ch, 1, sc.theme.border_dim);
        let qshow = crate::textsel::ellipsize(query, (field_w / cw).saturating_sub(1) as usize);
        let qend = sc.draw_str(field_x + 4, y, &qshow, sc.theme.chat_fg, list_bg);
        if caret_on {
            sc.fill_rect(qend, y, 2 * sc.scale.max(1), ch, sc.theme.accent);
        }
        y += ch + 6;

        // List region.
        let list_top = y;
        let list_h = view_rows * ch;
        let list_w = content_w.saturating_sub(cw); // leave a col for scrollbar
        sc.fill_rect(ix, list_top, list_w, list_h, list_bg);

        let mut ly = list_top;
        for row in rows.iter().take(view_rows as usize) {
            match row {
                CommandsRow::Header(h) => {
                    let line = crate::textsel::ellipsize(h, (list_w / cw) as usize);
                    sc.draw_str(ix + 2, ly, &line, sc.theme.title_dim, list_bg);
                    // Dim rule under the category label.
                    sc.fill_rect(ix + 2, ly + ch - 2, list_w.saturating_sub(4), 1, sc.theme.sep_dim);
                }
                CommandsRow::Item {
                    title,
                    slash,
                    shortcut,
                    selected,
                } => {
                    let row_bg = if *selected { sc.theme.status_bg } else { list_bg };
                    sc.fill_rect(ix, ly, list_w, ch, row_bg);
                    let mark = if *selected { "> " } else { "* " };
                    let mark_fg = if *selected { sc.theme.accent } else { sc.theme.composer_hint };
                    let mut px = sc.draw_str(ix + 2, ly, mark, mark_fg, row_bg);
                    let title_fg = if *selected { sc.theme.accent } else { sc.theme.chat_fg };
                    // Right column: shortcut if present, else /name.
                    let right = if !shortcut.is_empty() {
                        *shortcut
                    } else {
                        *slash
                    };
                    let right_cols = right.chars().count().min(18);
                    let left_cols = (list_w / cw)
                        .saturating_sub(3 + right_cols as u64 + 2) as usize;
                    let t = crate::textsel::ellipsize(title, left_cols);
                    px = sc.draw_str(px, ly, &t, title_fg, row_bg);
                    let rtxt = crate::textsel::ellipsize(right, right_cols);
                    let rlen = rtxt.chars().count() as u64 * cw;
                    let rx = ix + list_w.saturating_sub(rlen + 4);
                    if rx > px {
                        sc.draw_str(rx, ly, &rtxt, sc.theme.composer_hint, row_bg);
                    }
                }
            }
            ly += ch;
        }

        // Scrollbar (right edge of list).
        let sb_x = ix + list_w + 2;
        let sb_h = list_h;
        sc.fill_rect(sb_x, list_top, 3 * sc.scale.max(1), sb_h, sc.theme.composer_border);
        if total > view_rows as usize && total > 0 {
            let thumb_h = ((sb_h as usize * view_rows as usize) / total)
                .max(ch as usize)
                .min(sb_h as usize) as u64;
            let max_scroll = total.saturating_sub(view_rows as usize).max(1);
            let thumb_y = list_top
                + ((sb_h.saturating_sub(thumb_h)) as usize * scroll / max_scroll) as u64;
            sc.fill_rect(sb_x, thumb_y, 3 * sc.scale.max(1), thumb_h, sc.theme.accent);
        }

        // Footer.
        y = list_top + list_h + 6;
        sc.fill_rect(ix, y, content_w, 1, sc.theme.sep_dim);
        y += 4;
        let foot = "up/dn nav  |  Enter fill input  |  Esc close";
        sc.draw_str(
            ix,
            y,
            &crate::textsel::ellipsize(foot, cols as usize),
            sc.theme.composer_hint,
            bg,
        );
        sc.cursor_overlay();
    });
}

/// Draw the `/voice` modal: a live waveform (one vertical bar per recent RMS
/// level, newest on the right) above a status line and a Stop button. Called
/// every capture frame, so it repaints only the modal region.
pub fn draw_voice(levels: &[f32], status: &str) {
    MODAL_ON.store(true, core::sync::atomic::Ordering::Relaxed);
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.cursor_restore();
            sc.cur_vis = false;
            MODAL_RECTS.with(|m| *m = [(0, 0, 0, 0); 3]);
            let (ix, iy, cols) = sc.modal_box("Voice", 8);
            let ch = sc.ch();
            let cw = sc.cw();
            // Waveform region: 5 text rows tall, one 3px bar per level.
            let wave_h = 5 * ch;
            let wave_w = cols * cw;
            sc.fill_rect(ix, iy, wave_w, wave_h, sc.theme.chat_bg);
            let barw = 3 * sc.scale.max(1);
            let nbars = (wave_w / (barw + sc.scale)) as usize;
            let take = levels.len().min(nbars);
            let mid = iy + wave_h / 2;
            for (i, &lv) in levels[levels.len() - take..].iter().enumerate() {
                // Bar height from the RMS level (log-ish response for visibility).
                let l = if lv < 0.0 { 0.0 } else if lv > 1.0 { 1.0 } else { lv };
                let boost = l * (2.0 - l); // gentle curve
                let h = ((wave_h / 2) as f32 * boost) as u64 + sc.scale;
                let x = ix + (i as u64) * (barw + sc.scale);
                sc.fill_rect(x, mid - h, barw, 2 * h, sc.theme.accent);
            }
            // Status line + Stop button.
            let sy = iy + wave_h + ch / 2;
            sc.fill_rect(ix, sy, wave_w, ch, sc.theme.status_bg);
            sc.draw_str(ix, sy, status, sc.theme.title_dim, sc.theme.status_bg);
            sc.modal_button(ix, sy + ch + ch / 2, "Stop", true, 2);
            sc.cursor_overlay();
        }
    });
}

/// Repaint the normal split-pane UI — used to restore the screen after the
/// full-screen `/top` dashboard exits.
pub fn redraw_all() {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.redraw();
            sc.cur_vis = false;
        }
    });
}

/// Dismiss any modal and repaint the normal UI.
pub fn modal_dismiss() {
    MODAL_ON.store(false, core::sync::atomic::Ordering::Relaxed);
    MODAL_RECTS.with(|m| *m = [(0, 0, 0, 0); 3]);
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.redraw();
            sc.cur_vis = false;
        }
    });
}

/// Word-wrap `s` to `cols` columns (breaking long words), for modal messages.
fn wrap(s: &str, cols: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut line = String::new();
    for word in s.split_whitespace() {
        if !line.is_empty() && line.len() + 1 + word.len() > cols {
            out.push(core::mem::take(&mut line));
        }
        for chunk in word.as_bytes().chunks(cols.max(1)) {
            let w = core::str::from_utf8(chunk).unwrap_or("");
            if line.len() + 1 + w.len() > cols && !line.is_empty() {
                out.push(core::mem::take(&mut line));
            }
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(w);
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

/// Bring up the compositor on a Limine framebuffer and paint the initial UI.
pub fn init_console(fb: &Framebuffer) {
    let bpp_bytes = (fb.bpp as u64).div_ceil(8);
    let s = Screen::layout(
        fb.address as usize,
        fb.width,
        fb.height,
        fb.pitch,
        bpp_bytes,
        fb.red_mask_shift as u32,
        fb.green_mask_shift as u32,
        fb.blue_mask_shift as u32,
    );
    init_from(s);
}

/// Bring up the compositor over a raw linear framebuffer whose pixels are
/// **XRGB8888** (little-endian B,G,R,X → red 16 / green 8 / blue 0) — the common
/// case (QEMU ramfb, most UEFI GOP / VirtualBox).
pub fn init_console_raw(addr: usize, width: u64, height: u64, pitch: u64) {
    init_console_raw_fmt(addr, width, height, pitch, 4, 16, 8, 0);
}

/// Bring up the compositor over a raw linear framebuffer with an explicit pixel
/// format. Used by the aarch64 UEFI path, which reads the GOP pixel format from
/// the boot-info page (a real HDMI monitor may report RGB rather than BGR, and
/// swapping red/blue would tint the whole UI).
#[allow(clippy::too_many_arguments)]
pub fn init_console_raw_fmt(
    addr: usize,
    width: u64,
    height: u64,
    pitch: u64,
    bpp_bytes: u64,
    r_shift: u32,
    g_shift: u32,
    b_shift: u32,
) {
    let s = Screen::layout(addr, width, height, pitch, bpp_bytes, r_shift, g_shift, b_shift);
    init_from(s);
}

fn init_from(screen: Screen) {
    // Brand splash first (logo + wordmark), held briefly, then the live UI.
    if screen.layout.splash {
        screen.draw_splash();
        hold_ms(1300);
    }
    screen.redraw();
    SCREEN.with(|slot| *slot = Some(screen));
}

/// Busy-wait ~`ms` milliseconds for the splash hold, bounded by an iteration cap
/// so a frozen monotonic clock (some VBox configs) can't wedge the boot.
fn hold_ms(ms: u64) {
    let start = crate::arch::now_ms();
    let mut iters: u64 = 0;
    while crate::arch::now_ms().saturating_sub(start) < ms && iters < 300_000_000 {
        core::hint::spin_loop();
        iters += 1;
    }
}

/// Render `s` into the **chat** pane. Called by `serial::Serial::write_str`, so
/// ordinary `serial_println!` output (the shell + chat) appears here.
pub fn console_print(s: &str) {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.cursor_restore();
            sc.cur_vis = false;
            let mut chat = core::mem::replace(&mut sc.chat, dummy_pane());
            for &b in s.as_bytes() {
                Screen::pane_putc(sc, &mut chat, b);
            }
            sc.chat = chat;
            sc.caret_on = true; // keep the caret lit right after output
            // Do **not** redraw the composer here. The chat grid is already
            // sized above the reserved strip, so streaming tokens never touch
            // the box — and redrawing (with a strip clear) every chunk is what
            // made the whole composer flash while a response rendered.
            sc.cursor_overlay();
        }
    });
}

/// Render one byte into the chat pane (the shell's keystroke echo / backspace).
/// When the Grok-style composer is active, keystroke echo is handled by
/// [`composer_set`] — this path is for legacy serial-style editing only.
pub fn console_put_byte(byte: u8) {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            if sc.composer_active {
                return; // composer owns the input line
            }
            sc.cursor_restore();
            sc.cur_vis = false;
            let mut chat = core::mem::replace(&mut sc.chat, dummy_pane());
            Screen::pane_putc(sc, &mut chat, byte);
            sc.chat = chat;
            sc.caret_on = true;
            sc.cursor_overlay();
        }
    });
}

/// Whether the chat pane has a Grok-style input composer (always true once the
/// framebuffer console is up with a chat pane).
pub fn composer_available() -> bool {
    SCREEN.with(|slot| slot.as_ref().is_some_and(|sc| sc.chat.has_composer))
}

/// Whether the composer is the live prompt (between [`composer_begin`] and
/// [`composer_end`]). Serial line-editing still runs in parallel.
pub fn composer_is_active() -> bool {
    SCREEN.with(|slot| slot.as_ref().is_some_and(|sc| sc.composer_active))
}

/// Activate the input composer (call at the start of a prompt `read_line`).
pub fn composer_begin() {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.cursor_restore();
            sc.cur_vis = false;
            // Wipe any residual scrollback caret left after a streamed reply
            // (accent bar at the end of the last response line).
            if sc.chat.view == 0 {
                sc.repaint_cursor_cell(&sc.chat);
            }
            sc.composer_active = true;
            sc.composer_line.clear();
            sc.composer_cur = 0;
            sc.caret_on = true;
            sc.draw_composer();
            sc.cursor_overlay();
        }
    });
}

/// Update the composer line + caret column (0..=len). Redraws the box in place.
pub fn composer_set(line: &str, cursor: usize) {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.cursor_restore();
            sc.cur_vis = false;
            sc.composer_line.clear();
            sc.composer_line.push_str(line);
            sc.composer_cur = cursor.min(line.len());
            sc.caret_on = true;
            sc.draw_composer();
            sc.cursor_overlay();
        }
    });
}

/// Set the right half of the composer hint bar (e.g. model name / approval mode).
pub fn composer_set_hint_right(s: &str) {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.composer_hint_r.clear();
            sc.composer_hint_r.push_str(s);
            if sc.chat.has_composer {
                sc.cursor_restore();
                sc.cur_vis = false;
                sc.draw_composer();
                sc.cursor_overlay();
            }
        }
    });
}

/// Deactivate the composer (call when a line is submitted or the prompt ends).
pub fn composer_end() {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.cursor_restore();
            sc.cur_vis = false;
            sc.composer_active = false;
            sc.composer_line.clear();
            sc.composer_cur = 0;
            // Drop any open suggestion menu with the prompt.
            if sc.suggest_open || sc.suggest_rect.is_some() {
                sc.suggest_open = false;
                sc.suggest_items.clear();
                sc.suggest_sel = 0;
                sc.suggest_clear_region(true);
                sc.suggest_rect = None;
            }
            sc.draw_composer(); // empty idle box
            // Ensure the chat grid never shows a caret while a reply streams.
            if sc.chat.view == 0 {
                sc.repaint_cursor_cell(&sc.chat);
            }
            sc.cursor_overlay();
        }
    });
}

/// Update the slash-command / @file suggestion popup.
/// `items` is `(label, detail)` rows; `selected` is the highlighted index.
/// Empty `items` dismisses the menu.
///
/// **Typing performance:** does **not** full-repaint the chat pane on every
/// key. Old popup rect is erased cheaply; only the popup (and optional
/// composer box) is redrawn. Composer line text is already painted by
/// [`composer_set`].
pub fn suggest_set(items: &[(alloc::string::String, alloc::string::String)], selected: usize) {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.cursor_restore();
            sc.cur_vis = false;
            let was_open = sc.suggest_open || sc.suggest_rect.is_some();
            let old_len = sc.suggest_items.len();
            let old_sel = sc.suggest_sel;
            sc.suggest_items.clear();
            for (l, d) in items {
                sc.suggest_items.push((l.clone(), d.clone()));
            }
            sc.suggest_sel = if items.is_empty() {
                0
            } else {
                selected.min(items.len() - 1)
            };
            sc.suggest_open = !items.is_empty();

            if !sc.suggest_open {
                if was_open {
                    sc.suggest_clear_region(true);
                    sc.suggest_rect = None;
                }
                // Composer already current from composer_set — avoid a second
                // full chrome paint on every non-slash key.
                sc.cursor_overlay();
                return;
            }

            // Erase previous popup footprint if it was taller / different.
            if was_open {
                sc.suggest_clear_region(false);
                sc.suggest_rect = None;
            }
            if let Some(rect) = sc.suggest_geom() {
                sc.suggest_rect = Some(rect);
            } else {
                sc.suggest_open = false;
                sc.suggest_rect = None;
                sc.cursor_overlay();
                return;
            }
            // Popup only — composer text already drawn by the line editor.
            // When row count/selection changed a lot, still just the popup.
            let _ = (old_len, old_sel);
            sc.draw_suggest_popup();
            sc.cursor_overlay();
        }
    });
}

/// Dismiss the suggestion popup (if any).
pub fn suggest_clear() {
    suggest_set(&[], 0);
}

/// Whether the suggestion popup currently has rows.
pub fn suggest_is_open() -> bool {
    SCREEN.with(|slot| slot.as_ref().is_some_and(|sc| sc.suggest_open && !sc.suggest_items.is_empty()))
}

/// Erase any leftover grid caret in the chat response area (call after a
/// streamed reply finishes, before the next prompt). Safe no-op without FB.
pub fn clear_chat_caret() {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            if sc.chat.has_composer && sc.chat.view == 0 {
                sc.cursor_restore();
                sc.cur_vis = false;
                sc.repaint_cursor_cell(&sc.chat);
                sc.cursor_overlay();
            }
        }
    });
}

/// Set the status-bar text (left = brand, right = datetime), then repaint just
/// the bar. The shell calls this every second with the UI-config templates
/// resolved against the clock.
pub fn set_status(left: &str, right: &str) {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.cursor_restore();
            sc.cur_vis = false;
            sc.status_left.clear();
            sc.status_left.push_str(left);
            sc.status_right.clear();
            sc.status_right.push_str(right);
            sc.draw_status();
            sc.cursor_overlay();
        }
    });
}

/// Framebuffer size `(width, height)` in pixels, for the mouse to clamp to.
pub fn screen_dims() -> Option<(u64, u64)> {
    SCREEN.with(|slot| slot.as_ref().map(|sc| (sc.width, sc.height)))
}

/// Move the mouse cursor sprite to `(x, y)` (erasing it from the old spot).
pub fn cursor_move(x: u64, y: u64) {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.cursor_restore();
            sc.cur_active = true;
            sc.cur_x = x.min(sc.width.saturating_sub(1));
            sc.cur_y = y.min(sc.height.saturating_sub(1));
            sc.cursor_draw();
        }
    });
}

/// Redraw the cursor sprite at its current position (after a content redraw
/// erased it). Safe no-op if the console isn't up.
pub fn cursor_move_here() {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            if sc.cur_active {
                sc.cursor_draw();
            }
        }
    });
}

/// True if `(x, y)` is on the action-pane `[x]` close button (and it's shown).
pub fn hit_close(x: u64, y: u64) -> bool {
    SCREEN.with(|slot| {
        slot.as_ref().is_some_and(|sc| {
            if sc.right == RightMode::Closed {
                return false;
            }
            let (bx, by, bw, bh) = sc.close_btn();
            x >= bx && x < bx + bw && y >= by && y < by + bh
        })
    })
}

/// The editor pane text-area geometry `(ix, iy, cw, ch, cols, text_rows)` so the
/// editor can map a click to a (row, col). `None` unless the editor is open.
pub fn editor_pane_geom() -> Option<(u64, u64, u64, u64, u64, u64)> {
    SCREEN.with(|slot| {
        slot.as_ref().and_then(|sc| {
            if sc.right != RightMode::Editor {
                return None;
            }
            Some((sc.logs.ix, sc.logs.iy, sc.logs.cw, sc.logs.ch, sc.logs.cols, sc.logs.rows.saturating_sub(1)))
        })
    })
}

/// Advance the caret blink. Called from the shell's idle poll with the current
/// `now_ms()`; toggles the chat caret roughly twice a second.
pub fn blink(now_ms: u64) {
    if MODAL_ON.load(core::sync::atomic::Ordering::Relaxed) {
        return; // a modal overlays the panes; do not paint the caret under it
    }
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            // If the clock advances, blink on a 500 ms period. Only if it has
            // NEVER been seen advancing (a genuinely frozen monotonic clock)
            // fall back to a call-count cadence. On a fast host thousands of
            // idle polls land inside one millisecond, so the counter alone
            // must not trigger once the clock is known-good — that made the
            // caret strobe on VirtualBox.
            let toggle = if now_ms != sc.blink_seen_ms {
                if sc.blink_seen_ms != u64::MAX {
                    sc.clock_alive = true;
                }
                sc.blink_seen_ms = now_ms;
                sc.blink_calls = 0;
                now_ms.saturating_sub(sc.caret_last_ms) >= 500
            } else if sc.clock_alive {
                false
            } else {
                sc.blink_calls = sc.blink_calls.wrapping_add(1);
                if sc.blink_calls >= 300_000 {
                    sc.blink_calls = 0;
                    true
                } else {
                    false
                }
            };
            if toggle {
                sc.cursor_restore();
                sc.cur_vis = false;
                sc.caret_on = !sc.caret_on;
                sc.caret_last_ms = now_ms;
                sc.paint_caret();
                sc.cursor_overlay();
            }
        }
    });
}

/// Render `s` into the **logs** pane. Called by `ktrace`, so the trace stream
/// scrolls independently of the chat conversation.
pub fn log_print(s: &str) {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            if sc.right != RightMode::Ktrace {
                return; // action pane closed or owned by the editor; ktrace still hits serial
            }
            sc.cursor_restore();
            sc.cur_vis = false;
            let mut logs = core::mem::replace(&mut sc.logs, dummy_pane());
            for &b in s.as_bytes() {
                Screen::pane_putc(sc, &mut logs, b);
            }
            sc.logs = logs;
            sc.cursor_overlay();
        }
    });
}

/// A throwaway pane used only to satisfy the borrow checker while a real pane is
/// temporarily moved out for a `&Screen` + `&mut Pane` split (see `pane_putc`,
/// which needs immutable access to the screen's pixel plumbing while mutating a
/// pane). Its geometry is degenerate so it never draws anything if used.
fn dummy_pane() -> Pane {
    Pane {
        x: 0, y: 0, w: 0, h: 0, ix: 0, iy: 0, cw: 1, ch: 1, cols: 1, rows: 1, col: 0, row: 0,
        fg: (0, 0, 0), default_fg: (0, 0, 0), bg: (0, 0, 0),
        esc: EscState::Ground, csi: [0; 32], csi_len: 0, bold: false,
        title: String::new(), show_caret: false,
        grid: Vec::new(), hist: VecDeque::new(), view: 0, sel: None, has_composer: false,
    }
}

/// The editor viewport size `(cols, rows)` inside the right pane — `rows` is the
/// text area (the bottom row is reserved for the editor's mode line). `None` if
/// the console isn't up.
pub fn editor_dims() -> Option<(usize, usize)> {
    SCREEN.with(|slot| {
        slot.as_ref().map(|sc| {
            let cols = sc.logs.cols as usize;
            let rows = (sc.logs.rows.saturating_sub(1)).max(1) as usize;
            (cols, rows)
        })
    })
}

/// Copy interactive UI state that `Screen::build` always zeroes (composer mid-
/// prompt, keyboard focus, caret blink). Without this, Ctrl+F / tab open /
/// divider drag mid-`read_line` would kill the composer until the next prompt.
fn preserve_interactive(ns: &mut Screen, old: &Screen) {
    ns.composer_active = old.composer_active;
    ns.composer_line = old.composer_line.clone();
    ns.composer_cur = old.composer_cur;
    ns.composer_hint_l = old.composer_hint_l.clone();
    ns.composer_hint_r = old.composer_hint_r.clone();
    ns.suggest_open = old.suggest_open;
    ns.suggest_items = old.suggest_items.clone();
    ns.suggest_sel = old.suggest_sel;
    ns.suggest_rect = old.suggest_rect;
    ns.focus_action = old.focus_action;
    ns.caret_on = old.caret_on;
    ns.caret_last_ms = old.caret_last_ms;
    ns.clock_alive = old.clock_alive;
    ns.blink_seen_ms = old.blink_seen_ms;
}

/// Rebuild the screen for a new split/right-mode, preserving geometry, layout
/// config, status text, interactive state, and — via [`Pane::adopt`] — the pane
/// text content.
fn rebuilt(old: &Screen, split: bool, right: RightMode) -> Screen {
    let mut ns = Screen::build(
        old.addr, old.width, old.height, old.pitch, old.bpp_bytes, old.r_shift, old.g_shift, old.b_shift, &old.layout, split,
    );
    ns.status_left = old.status_left.clone();
    ns.status_right = old.status_right.clone();
    ns.right = right;
    ns.tabs = old.tabs.clone();
    ns.active = old.active;
    ns.right_before_editor = old.right_before_editor;
    preserve_interactive(&mut ns, old);
    // No action pane → keyboard focus is always the chat/composer.
    if right == RightMode::Closed {
        ns.focus_action = false;
    }
    ns.chat.adopt(&old.chat);
    ns.logs.adopt(&old.logs);
    ns
}

/// The current (active tab's) action-pane mode.
pub fn right_mode() -> RightMode {
    SCREEN.with(|slot| slot.as_ref().map(|sc| sc.right).unwrap_or(RightMode::Closed))
}

/// The open tab modes, in bar order (for the shell to know what's open).
pub fn tab_modes() -> Vec<RightMode> {
    SCREEN.with(|slot| slot.as_ref().map(|sc| sc.tabs.clone()).unwrap_or_default())
}

/// True if a tab of `mode` is currently open.
pub fn has_tab(mode: RightMode) -> bool {
    SCREEN.with(|slot| slot.as_ref().map(|sc| sc.tabs.contains(&mode)).unwrap_or(false))
}

/// Open a tab for `mode`, or select it if already open. First tab opens the
/// split; additional tabs reuse the geometry (every other tab stays alive).
/// Operates on the slot directly so surface/editor openers can reuse it without
/// re-entering `SCREEN.with`.
fn open_view_slot(slot: &mut Option<Screen>, mode: RightMode) {
    let Some(old) = slot else { return };
    if let Some(i) = old.tabs.iter().position(|&m| m == mode) {
        old.active = i;
        old.right = mode;
        old.repaint_action(); // geometry unchanged → action pane only
        return;
    }
    if old.tabs.is_empty() {
        // First tab: the chat pane resizes, so a full relayout + redraw.
        let mut ns = rebuilt(old, true, mode);
        ns.tabs = alloc::vec![mode];
        ns.active = 0;
        ns.right = mode;
        ns.redraw();
        *slot = Some(ns);
    } else {
        // Additional tab: geometry is unchanged; append + select + repaint just
        // the action pane (no whole-screen redraw → no flicker).
        old.tabs.push(mode);
        old.active = old.tabs.len() - 1;
        old.right = mode;
        old.repaint_action();
    }
}

/// Open (or focus) a tab for `mode`. `Closed` is a no-op (use `close_action`).
pub fn set_right(mode: RightMode) {
    if mode == RightMode::Closed {
        return;
    }
    SCREEN.with(|slot| open_view_slot(slot, mode));
}

/// Cycle to the next/previous tab, returning the newly active mode. Keeps every
/// tab's state — the switch only changes which one paints into the pane.
pub fn cycle_tab(forward: bool) -> RightMode {
    SCREEN.with(|slot| {
        let Some(old) = slot else { return RightMode::Closed };
        let n = old.tabs.len();
        if n <= 1 {
            return old.right;
        }
        old.active = if forward { (old.active + 1) % n } else { (old.active + n - 1) % n };
        old.right = old.tabs[old.active];
        old.repaint_action();
        old.right
    })
}

/// Select tab `i` (clamped), returning the active mode.
pub fn select_tab(i: usize) -> RightMode {
    SCREEN.with(|slot| {
        let Some(old) = slot else { return RightMode::Closed };
        if i >= old.tabs.len() {
            return old.right;
        }
        old.active = i;
        old.right = old.tabs[i];
        old.repaint_action();
        old.right
    })
}

/// The tab index under pixel `(x, y)`, if the click hit the tab bar.
pub fn tab_hit(x: u64, y: u64) -> Option<usize> {
    SCREEN.with(|slot| {
        slot.as_ref().and_then(|sc| {
            if sc.tabs.is_empty() {
                return None;
            }
            let ty = sc.logs.y + BORDER + 4;
            if y < ty || y >= ty + sc.ch() {
                return None;
            }
            sc.tab_layout().into_iter().position(|(_, tx, w)| x >= tx && x < tx + w)
        })
    })
}

/// Close the tab of `mode` if open (used by `editor_leave`, `/ktrace` toggle).
pub fn close_tab_mode(mode: RightMode) {
    SCREEN.with(|slot| {
        let Some(old) = slot else { return };
        if let Some(i) = old.tabs.iter().position(|&m| m == mode) {
            old.active = i;
            close_active_slot(slot);
        }
    });
}

/// Close the active tab. If it was the last, collapse the split.
fn close_active_slot(slot: &mut Option<Screen>) {
    let Some(old) = slot else { return };
    if old.tabs.is_empty() {
        return;
    }
    old.tabs.remove(old.active);
    if old.tabs.is_empty() {
        let mut ns = rebuilt(old, false, RightMode::Closed);
        ns.tabs.clear();
        ns.active = 0;
        ns.right = RightMode::Closed;
        ns.redraw();
        *slot = Some(ns);
    } else {
        // Other tabs remain → geometry unchanged; repaint just the action pane.
        if old.active >= old.tabs.len() {
            old.active = old.tabs.len() - 1;
        }
        old.right = old.tabs[old.active];
        old.repaint_action();
    }
}

/// Open the ktrace log stream in the action pane.
pub fn open_ktrace() {
    set_right(RightMode::Ktrace);
}

/// Present an agent's surface backing buffer (`sw`×`sh`, 0xRRGGBB pixels) into
/// the action pane, opening it in `Surface(id)` mode on first present. The image
/// is nearest-neighbour scaled to fit the pane interior, letterboxed. Called by
/// `synapse::ui` after a `ui_draw`; the compositor is the only place surface
/// pixels reach the screen (the determinism boundary stays intact — the agent
/// emitted grammar-validated draw ops, never raw pixels here).
pub fn present_surface(id: u32, sw: usize, sh: usize, buf: &[u32]) {
    present_surface_reserve(id, sw, sh, buf, 0);
}

/// Like [`present_surface`], but leaves `reserve_bottom` px at the bottom of the
/// pane untouched — the frame is scaled/letterboxed into the region *above* the
/// reserve and the reserved strip is never cleared. The video player uses this
/// to keep its control HUD in a fixed strip that the per-frame blit doesn't
/// repaint (so the HUD updates in place instead of flickering under it).
pub fn present_surface_reserve(id: u32, sw: usize, sh: usize, buf: &[u32], reserve_bottom: u64) {
    SCREEN.with(|slot| {
        // Open (or focus) this surface's tab, then paint over its cleared
        // interior. Idempotent: re-presenting the active surface tab is cheap.
        let active_surface = slot.as_ref().map(|sc| sc.right == RightMode::Surface(id)).unwrap_or(false);
        if !active_surface {
            open_view_slot(slot, RightMode::Surface(id));
        }
        let Some(sc) = slot else { return };
        if sc.right != RightMode::Surface(id) || sw == 0 || sh == 0 {
            return;
        }
        sc.cursor_restore();
        sc.cur_vis = false;
        let (px, py) = (sc.logs.ix, sc.logs.iy);
        let (pw, ph_full) = (sc.logs.cols * sc.logs.cw, sc.logs.rows * sc.logs.ch);
        // Usable frame height excludes the reserved HUD strip at the bottom.
        let ph = ph_full.saturating_sub(reserve_bottom);
        if pw == 0 || ph == 0 || buf.len() < sw * sh {
            sc.cursor_overlay();
            return;
        }
        // Aspect-fit ("contain") into the pane: scale up *or* down so the
        // picture uses as much of the pane as possible without cropping.
        // (Integer-only upscale left large empty bars in fullscreen when the
        // source was smaller than the pane — e.g. 640×360 in a full-HD action
        // pane.)
        let (dw, dh) = {
            // Compare sw/pw vs sh/ph via cross-multiply to pick the limiting edge.
            let fit_w = pw;
            let fit_h = (sh as u64).saturating_mul(pw).saturating_div(sw as u64).max(1);
            if fit_h <= ph {
                (fit_w, fit_h)
            } else {
                let fit_h = ph;
                let fit_w = (sw as u64).saturating_mul(ph).saturating_div(sh as u64).max(1);
                (fit_w.min(pw), fit_h)
            }
        };
        let ox = px + (pw.saturating_sub(dw)) / 2;
        let oy = py + (ph.saturating_sub(dh)) / 2;
        // **No full-pane clear.** Clearing the whole surface with fill_rect
        // (then painting the frame) flashed background on the single-buffered
        // FB for tens of ms every present — visible as a once-per-second
        // (or every-frame) flicker. Only paint letterbox *margins*; the frame
        // blit overwrites the content rectangle in place.
        let bg = sc.logs.bg;
        if oy > py {
            sc.fill_rect(px, py, pw, oy - py, bg); // top bar
        }
        let bottom = oy + dh;
        if bottom < py + ph {
            sc.fill_rect(px, bottom, pw, (py + ph) - bottom, bg); // bottom bar
        }
        if ox > px {
            sc.fill_rect(px, oy, ox - px, dh, bg); // left bar
        }
        let right = ox + dw;
        if right < px + pw {
            sc.fill_rect(right, oy, (px + pw) - right, dh, bg); // right bar
        }
        // Build one destination row at a time and blit — sequential stores beat
        // hundreds of thousands of put_pixel calls for video.
        let mut row = alloc::vec![0u32; dw as usize];
        for dy in 0..dh {
            let sy = (dy * sh as u64 / dh) as usize;
            let srow = sy * sw;
            for dx in 0..dw as usize {
                let sx = (dx as u64 * sw as u64 / dw) as usize;
                row[dx] = buf[srow + sx];
            }
            sc.blit_rgb32_row(ox, oy + dy, &row);
        }
        sc.cursor_overlay();
    });
}

/// Height in px the video HUD reserves at the bottom of the action pane — the
/// player blits its frame above this, and [`draw_video_status`] fills it.
pub fn video_hud_height() -> u64 {
    SCREEN.with(|slot| slot.as_ref().map(|sc| sc.ch() * 4 + sc.ch() / 2).unwrap_or(0))
}

/// The action pane's interior size in pixels, if the split is open — the
/// image viewer sizes its downscale to this before presenting.
pub fn action_dims_px() -> Option<(u64, u64)> {
    SCREEN.with(|slot| {
        slot.as_ref().and_then(|sc| {
            (sc.right != RightMode::Closed).then(|| (sc.logs.cols * sc.logs.cw, sc.logs.rows * sc.logs.ch))
        })
    })
}

/// The action pane's interior background colour, packed `0x00RRGGBB` to match
/// the pixel buffer [`present_surface`] blits — the image viewer letterboxes
/// with this so the padding around a zoomed/rotated image matches the pane.
pub fn pane_bg() -> Option<u32> {
    SCREEN.with(|slot| {
        slot.as_ref().map(|sc| {
            let (r, g, b) = sc.logs.bg;
            ((r as u32) << 16) | ((g as u32) << 8) | b as u32
        })
    })
}

/// Open the `/top` dashboard in the action pane (filled by the shell's idle
/// tick). Returns true if it is now open (false if it was already).
pub fn open_top() {
    set_right(RightMode::Top);
}

/// Whether the action pane currently shows `/top`.
pub fn is_top() -> bool {
    right_mode() == RightMode::Top
}

/// Open the live todos pane.
pub fn open_todos() {
    set_right(RightMode::Todos);
}

/// Whether the action pane shows todos.
pub fn is_todos() -> bool {
    right_mode() == RightMode::Todos
}

/// One row for [`draw_todos`].
pub struct TodoViewItem<'a> {
    pub id: u32,
    pub text: &'a str,
    pub status: &'a str,
}

/// Render the session todo list into the action pane (checklist view).
pub fn draw_todos(items: &[TodoViewItem<'_>], title: &str) {
    SCREEN.with(|slot| {
        let Some(sc) = slot else { return };
        if sc.right != RightMode::Todos {
            return;
        }
        sc.cursor_restore();
        sc.cur_vis = false;
        let ch = sc.ch();
        let (px, iy, iw) = (sc.logs.ix, sc.logs.iy, sc.logs.cols * sc.logs.cw);
        let bg = sc.logs.bg;
        let cols = (iw / sc.cw()).max(1) as usize;
        let mut y = iy;
        let head = if title.is_empty() { "Todos" } else { title };
        let head_fmt = pad_trunc(head, cols);
        sc.draw_str_bg(px, y, &head_fmt, sc.theme.accent, bg);
        y += ch + ch / 4;
        if items.is_empty() {
            sc.draw_str_bg(px, y, &pad_trunc("(no todos — agent todo_write)", cols), sc.theme.title_dim, bg);
            sc.cursor_overlay();
            return;
        }
        for it in items {
            let mark = match it.status {
                "done" => "[x]",
                "in_progress" => "[>]",
                "cancelled" => "[-]",
                _ => "[ ]",
            };
            let row = alloc::format!("{mark} {}: {}", it.id, it.text);
            let fg = match it.status {
                "done" => sc.theme.title_dim,
                "in_progress" => sc.theme.accent,
                _ => sc.theme.logs_fg,
            };
            sc.draw_str_bg(px, y, &pad_trunc(&row, cols), fg, bg);
            y += ch;
            if y + ch > iy + sc.logs.rows * ch {
                break;
            }
        }
        let blank = pad_trunc("", cols);
        while y + ch <= iy + sc.logs.rows * ch {
            sc.draw_str_bg(px, y, &blank, bg, bg);
            y += ch;
        }
        sc.cursor_overlay();
    });
}

fn pad_trunc(s: &str, cols: usize) -> alloc::string::String {
    let mut out: alloc::string::String = s.chars().take(cols).collect();
    while out.chars().count() < cols {
        out.push(' ');
    }
    out
}

/// Close the **active** tab (chat becomes full-width once the last tab closes).
pub fn close_action() {
    SCREEN.with(close_active_slot);
}

/// Open (or focus) the `/open` editor tab.
pub fn editor_enter() {
    set_right(RightMode::Editor);
}

/// Close the editor tab (the editor quit); the active tab falls back to a
/// sibling, or the pane collapses if the editor was the only tab.
pub fn editor_leave() {
    close_tab_mode(RightMode::Editor);
}

/// Render the editor into the right pane: title `editor: <file>`, the visible
/// slice of `lines` from `top`, a reverse-video block cursor at
/// `(cur_row, cur_col)`, and a bottom mode line. `gutter` toggles line numbers.
/// `hl` is optional per-byte syntax colours for the visible lines (indexed
/// from `top`; `None` entries fall back to the theme's `editor_fg`).
#[allow(clippy::too_many_arguments)]
pub fn editor_render(
    title: &str,
    lines: &[alloc::string::String],
    top: usize,
    cur_row: usize,
    cur_col: usize,
    modeline: &str,
    sel: Option<((usize, usize), (usize, usize))>,
    hl: Option<&[Vec<Option<Rgb>>]>,
) {
    SCREEN.with(|slot| {
        let Some(sc) = slot else { return };
        sc.cursor_restore();
        sc.cur_vis = false;
        let (px, pw, cw, ch, cols, rows) =
            (sc.logs.x, sc.logs.w, sc.logs.cw, sc.logs.ch, sc.logs.cols, sc.logs.rows);
        let (ix, iy) = (sc.logs.ix, sc.logs.iy);
        sc.draw_frame_titled(&sc.logs, true, title);
        // Clear the interior to the editor background.
        sc.fill_rect(ix, iy, cols * cw, rows * ch, sc.theme.editor_bg);
        let text_rows = rows.saturating_sub(1);
        // Is text (row, col) inside the inclusive selection range?
        let in_sel = |row: usize, col: usize| -> bool {
            let Some(((r1, c1), (r2, c2))) = sel else { return false };
            if row < r1 || row > r2 {
                return false;
            }
            let after_start = row > r1 || col >= c1;
            let before_end = row < r2 || col <= c2;
            after_start && before_end
        };
        // Line-number gutter width (digits + 1 space).
        let gutter = {
            let mut n = lines.len().max(1);
            let mut w = 1;
            while n >= 10 {
                n /= 10;
                w += 1;
            }
            (w + 1) as u64
        };
        for i in 0..text_rows {
            let li = top + i as usize;
            if li >= lines.len() {
                break;
            }
            let y = iy + i * ch;
            // Gutter: right-aligned 1-based line number.
            let num = alloc::format!("{:>width$} ", li + 1, width = (gutter - 1) as usize);
            let mut x = ix;
            for b in num.bytes() {
                sc.blit_glyph(x, y, b, sc.theme.editor_lineno, sc.theme.editor_bg);
                x += cw;
            }
            // Text, clipped to the pane width; selected cells get a highlight
            // bg; syntax-highlighted bytes their class colour.
            let mut c = gutter;
            for (col, &b) in lines[li].as_bytes().iter().enumerate() {
                if c >= cols {
                    break;
                }
                let bg = if in_sel(li, col) { sc.theme.editor_sel } else { sc.theme.editor_bg };
                let fg = hl
                    .and_then(|h| h.get(i as usize))
                    .and_then(|v| v.get(col).copied().flatten())
                    .unwrap_or(sc.theme.editor_fg);
                sc.blit_glyph(x, y, b, fg, bg);
                x += cw;
                c += 1;
            }
        }
        // Reverse-video block cursor.
        if cur_row >= top && (cur_row - top) < text_rows as usize {
            let scr = (cur_row - top) as u64;
            let col_on_screen = gutter + cur_col as u64;
            if col_on_screen < cols {
                let y = iy + scr * ch;
                let x = ix + col_on_screen * cw;
                let byte = lines.get(cur_row).and_then(|l| l.as_bytes().get(cur_col)).copied().unwrap_or(b' ');
                let byte = if (0x20..=0x7e).contains(&byte) { byte } else { b' ' };
                sc.blit_glyph(x, y, byte, sc.theme.editor_bg, sc.theme.accent); // fg/bg swapped = block cursor
            }
        }
        // Mode line across the bottom interior row — ellipsize so a long path
        // never paints past the pane edge.
        let sy = iy + text_rows * ch;
        sc.fill_rect(px + BORDER, sy, pw - 2 * BORDER, ch, sc.theme.status_bg);
        let ml = crate::textsel::ellipsize(modeline, cols as usize);
        let mut x = ix;
        for b in ml.bytes() {
            sc.blit_glyph(x, sy, b, sc.theme.title_active, sc.theme.status_bg);
            x += cw;
        }
        sc.cursor_overlay();
    });
}

/// Rebuild the panes from a new [`LayoutCfg`] (split ratio, font scale, pane
/// swap, titles, fullscreen) on the live framebuffer and repaint. Used by
/// `/ui`, Ctrl+F, and divider drag. No-op if the console isn't up.
///
/// Preserves the live composer + focus so a mid-prompt fullscreen toggle does
/// not strand the shell with a dead input box.
pub fn relayout(cfg: &LayoutCfg) {
    SCREEN.with(|slot| {
        if let Some(old) = slot {
            let (sl, sr) = (old.status_left.clone(), old.status_right.clone());
            let mode = old.right;
            let mut ns = Screen::build(
                old.addr, old.width, old.height, old.pitch, old.bpp_bytes, old.r_shift, old.g_shift, old.b_shift, cfg,
                mode != RightMode::Closed,
            );
            ns.status_left = sl;
            ns.status_right = sr;
            ns.right = mode;
            ns.tabs = old.tabs.clone();
            ns.active = old.active;
            ns.right_before_editor = old.right_before_editor;
            preserve_interactive(&mut ns, old);
            ns.chat.adopt(&old.chat);
            ns.logs.adopt(&old.logs);
            // Fullscreen can park the chat (action-full): keep focus on action.
            // Chat-full parks the action pane — snap keyboard back to the composer.
            if cfg.fullscreen == 1 || mode == RightMode::Closed {
                ns.focus_action = false;
            }
            ns.redraw();
            *slot = Some(ns);
        }
    });
}

/// Toggle fullscreen: maximise the focused pane to fill the screen, or restore
/// the split. Returns the new state (0 normal, 1 chat-full, 2 action-full).
pub fn toggle_fullscreen() -> u8 {
    let cfg = SCREEN.with(|slot| {
        slot.as_mut().map(|sc| {
            let action_open = sc.right != RightMode::Closed;
            let mut c = sc.layout.clone();
            c.fullscreen = if c.fullscreen != 0 {
                0
            } else if sc.focus_action && action_open {
                2
            } else {
                1
            };
            c
        })
    });
    match cfg {
        Some(c) => {
            let st = c.fullscreen;
            relayout(&c);
            st
        }
        None => 0,
    }
}

/// The current chat-pane split percentage (10..90).
pub fn split_pct() -> u64 {
    SCREEN.with(|slot| slot.as_ref().map(|sc| sc.layout.chat_pct).unwrap_or(CHAT_PCT))
}

/// The default chat-pane split percentage (`/pane reset`).
pub fn default_chat_pct() -> u64 {
    CHAT_PCT
}

/// Set the chat|action split to `pct` percent (clamped 10..90) and relayout,
/// clearing any fullscreen state.
pub fn set_split_pct(pct: u64) {
    let cfg = SCREEN.with(|slot| {
        slot.as_mut().map(|sc| {
            let mut c = sc.layout.clone();
            c.chat_pct = pct.clamp(10, 90);
            c.fullscreen = 0;
            c
        })
    });
    if let Some(c) = cfg {
        relayout(&c);
    }
}

/// Nudge the split ratio by `delta` percent (keyboard resize).
pub fn nudge_split(delta: i64) {
    let p = split_pct() as i64;
    set_split_pct((p + delta).clamp(10, 90) as u64);
}

/// If `(x,y)` is on the draggable divider between the two panes, return its
/// current gap centre x (so the caller can enter a resize drag). `None` when
/// fullscreen/closed (no divider) or the point is elsewhere.
pub fn divider_hit(x: u64, y: u64) -> Option<u64> {
    SCREEN.with(|slot| {
        let sc = slot.as_ref()?;
        if sc.right == RightMode::Closed || sc.layout.fullscreen != 0 {
            return None;
        }
        // The gap sits between the two pane boxes.
        let (a, b) = (&sc.chat, &sc.logs);
        let gap_l = a.x.min(b.x) + if a.x < b.x { a.w } else { b.w };
        let gap_r = gap_l + GAP;
        let within_y = y >= a.y && y < a.y + a.h;
        // Give the divider a few px of grab tolerance either side.
        if within_y && x + 4 >= gap_l && x <= gap_r + 4 {
            Some((gap_l + gap_r) / 2)
        } else {
            None
        }
    })
}

/// Set the split so the divider sits at pixel `x` (from a resize drag).
pub fn set_divider_x(x: u64) {
    let pct = SCREEN.with(|slot| {
        slot.as_ref().map(|sc| {
            let avail = sc.width.saturating_sub(2 * OUTER + GAP).max(1);
            // x measured from the content's left edge to the chat width.
            let chat_w = if sc.layout.swap { (sc.width.saturating_sub(x)).min(avail) } else { x.saturating_sub(OUTER).min(avail) };
            (chat_w * 100 / avail).clamp(10, 90)
        })
    });
    if let Some(p) = pct {
        set_split_pct(p);
    }
}

/// Scroll a pane's view by `delta` lines (`+` = back in time, `-` = toward
/// live); `action` picks the ktrace pane, else chat. Snaps caret handling
/// automatically: the caret only draws on a live view.
pub fn scroll_view(action: bool, delta: i64) {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            if action && sc.right != RightMode::Ktrace {
                return;
            }
            sc.cursor_restore();
            sc.cur_vis = false;
            let p = if action { &mut sc.logs } else { &mut sc.chat };
            let max = p.hist.len();
            let v = (p.view as i64 + delta).clamp(0, max as i64) as usize;
            if v != p.view {
                p.view = v;
                let p = if action { &sc.logs } else { &sc.chat };
                sc.render_view(p);
                if !action && v == 0 {
                    sc.caret_draw(&sc.chat); // no-op when chat has_composer
                }
            }
            sc.cursor_overlay();
        }
    });
}

/// Scroll a pane's view by one page (its row count minus one).
pub fn scroll_page(action: bool, up: bool) {
    let rows = SCREEN.with(|slot| {
        slot.as_ref().map(|sc| if action { sc.logs.rows } else { sc.chat.rows }).unwrap_or(1)
    }) as i64;
    scroll_view(action, if up { rows - 1 } else { -(rows - 1) });
}

/// Snap a pane back to the live view (offset 0).
pub fn scroll_live(action: bool) {
    scroll_view(action, i64::MIN / 2);
}

/// Toggle keyboard focus between the chat pane and an open action pane.
/// Returns true if the action pane now holds focus. No-op (false) when the
/// action pane is closed or owned by the editor.
///
/// When focus returns to the chat pane, the Grok-style composer is repainted
/// immediately (accent border + caret) so the shell is ready for input without
/// waiting for a keystroke to re-sync.
pub fn focus_toggle() -> bool {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            if sc.right == RightMode::Closed || sc.right == RightMode::Editor {
                // Nothing to focus (closed), or the editor already owns input.
                return sc.right == RightMode::Editor;
            }
            sc.focus_action = !sc.focus_action;
            sc.cursor_restore();
            sc.cur_vis = false;
            sc.draw_frame(&sc.chat, !sc.action_focused());
            sc.draw_frame_titled(&sc.logs, sc.action_focused(), "");
            sc.draw_tab_bar();
            sc.draw_close_btn();
            // Composer chrome reflects focus (accent border + caret only when
            // the chat holds keyboard focus). Force caret on so it is visible
            // the instant focus returns — no need to type first.
            if sc.chat.has_composer {
                if !sc.action_focused() {
                    sc.caret_on = true;
                }
                sc.draw_composer();
            }
            sc.cursor_overlay();
            sc.focus_action
        } else {
            false
        }
    })
}

/// Whether keyboard focus is on the action pane (see [`focus_toggle`]).
pub fn focus_is_action() -> bool {
    SCREEN.with(|slot| slot.as_ref().map(|sc| sc.action_focused()).unwrap_or(false))
}

/// Give keyboard focus to the action pane (true) or the chat pane (false),
/// e.g. from a mouse click. Same constraints as [`focus_toggle`].
///
/// Always refreshes the composer when focusing the chat, even if focus was
/// already on chat — so a click on the shell agent immediately arms the input.
pub fn focus_set(action: bool) {
    let (flips, need_composer) = SCREEN.with(|slot| {
        slot.as_ref()
            .map(|sc| {
                let editor = sc.right == RightMode::Editor;
                let closed = sc.right == RightMode::Closed;
                let flips = !closed && !editor && sc.focus_action != action;
                // Focusing chat: repaint composer even when already focused so
                // the caret/border activate without a first keystroke.
                let need_composer = !action && sc.chat.has_composer && !editor;
                (flips, need_composer)
            })
            .unwrap_or((false, false))
    });
    if flips {
        focus_toggle();
    } else if need_composer {
        SCREEN.with(|slot| {
            if let Some(sc) = slot {
                sc.cursor_restore();
                sc.cur_vis = false;
                sc.focus_action = false;
                sc.caret_on = true;
                sc.draw_frame(&sc.chat, true);
                sc.draw_composer();
                sc.cursor_overlay();
            }
        });
    }
}

/// Which pane a click at `(x, y)` landed in: `Some(true)` = action pane,
/// `Some(false)` = chat pane, `None` = neither (status bar / margins).
/// Map a screen click into the active surface's logical coordinates
/// (`synapse::ui` SURF_W×SURF_H), accounting for letterboxing. `None` if the
/// action pane is not showing a surface or the click is outside the painted
/// frame.
pub fn surface_hit(mx: u64, my: u64) -> Option<(u32, u16, u16)> {
    SCREEN.with(|slot| {
        let sc = slot.as_ref()?;
        let id = match sc.right {
            RightMode::Surface(id) => id,
            _ => return None,
        };
        let (px, py) = (sc.logs.ix, sc.logs.iy);
        let (pw, ph) = (sc.logs.cols * sc.logs.cw, sc.logs.rows * sc.logs.ch);
        if pw == 0 || ph == 0 || mx < px || my < py || mx >= px + pw || my >= py + ph {
            return None;
        }
        // Same aspect-fit as present_surface (sw=256, sh=192).
        let (sw, sh) = (256u64, 192u64);
        let (dw, dh) = {
            let fit_w = pw;
            let fit_h = sh.saturating_mul(pw).saturating_div(sw).max(1);
            if fit_h <= ph {
                (fit_w, fit_h)
            } else {
                let fit_h = ph;
                let fit_w = sw.saturating_mul(ph).saturating_div(sh).max(1);
                (fit_w.min(pw), fit_h)
            }
        };
        let ox = px + (pw.saturating_sub(dw)) / 2;
        let oy = py + (ph.saturating_sub(dh)) / 2;
        if mx < ox || my < oy || mx >= ox + dw || my >= oy + dh {
            return None;
        }
        let sx = ((mx - ox) * sw / dw) as u16;
        let sy = ((my - oy) * sh / dh) as u16;
        Some((id, sx, sy))
    })
}

pub fn pane_hit(x: u64, y: u64) -> Option<bool> {
    SCREEN.with(|slot| {
        slot.as_ref().and_then(|sc| {
            let hit = |p: &Pane| x >= p.x && x < p.x + p.w && y >= p.y && y < p.y + p.h;
            if sc.right != RightMode::Closed && hit(&sc.logs) {
                Some(true)
            } else if hit(&sc.chat) {
                Some(false)
            } else {
                None
            }
        })
    })
}

// --- chat-pane mouse text selection ---------------------------------------
//
// The editor already had drag-to-copy; this gives the chat pane the same:
// press anchors a selection, drag extends it (highlight painted by
// `render_view`), release hands the text to the shell for the clipboard.
// Coordinates are absolute over scrollback + grid (`crate::textsel`), so a
// selection stays glued to its text while output scrolls past.

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

/// Wipe the chat pane's text (grid + scrollback) and repaint it — `/clear`.
pub fn clear_chat() {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.cursor_restore();
            sc.cur_vis = false;
            sc.chat.clear_content();
            sc.fill_rect(sc.chat.ix, sc.chat.iy, sc.chat.cols * sc.chat.cw, sc.chat.rows * sc.chat.ch, sc.chat.bg);
            if sc.chat.has_composer {
                sc.draw_composer();
            } else {
                sc.caret_draw(&sc.chat);
            }
            sc.cursor_overlay();
        }
    });
}
