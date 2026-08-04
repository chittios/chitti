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
type Cell = (char, Rgb);

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
    pub accent: Rgb, // active border / caret / selection chrome
    /// Status-bar / splash Synapse-C logo **ring** (from `ui.json` `theme.logo`;
    /// defaults to `accent` when omitted).
    pub logo: Rgb,
    /// Status-bar / splash logo **node** (from `theme.logo_node`; defaults to
    /// `chat_fg` when omitted).
    pub logo_node: Rgb,
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
    /// bordered input composer fill (slightly elevated over chat_bg).
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
        logo: (204, 120, 92),          // matches accent unless ui.json overrides
        logo_node: (250, 249, 245),    // cream node
        border_dim: (58, 55, 51),      // inactive border
        title_active: (204, 120, 92),  // primary
        title_dim: (108, 106, 100),    // muted #6c6a64
        sep_dim: (42, 40, 37),
        status_bg: (37, 35, 32),       // surface-dark-elevated #252320
        status_fg: (160, 157, 150),    // on-dark-soft — icons + status text
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
    let mut has_logo = false;
    let mut has_logo_node = false;
    for (name, hex) in pairs {
        let slot = match name.as_str() {
            "screen_bg" => &mut t.screen_bg,
            "chat_bg" => &mut t.chat_bg,
            "logs_bg" => &mut t.logs_bg,
            "chat_fg" => &mut t.chat_fg,
            "logs_fg" => &mut t.logs_fg,
            "accent" => &mut t.accent,
            "logo" => {
                has_logo = true;
                &mut t.logo
            }
            "logo_node" => {
                has_logo_node = true;
                &mut t.logo_node
            }
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
    // Omitted logo keys track the brand palette so a theme that only sets
    // `accent` / `chat_fg` still recolors the mark without a second key.
    if !has_logo {
        t.logo = t.accent;
    }
    if !has_logo_node {
        t.logo_node = t.chat_fg;
    }
    t
}

// Layout metrics, in pixels (independent of font scale).
/// Anti-aliasing sub-sample rate per axis (`AA_SS`×`AA_SS` samples per pixel →
/// `AA_SS²`+1 coverage levels). 4 gives 16 levels — smooth curves at negligible
/// cost, since only edge pixels of a shape actually vary.
const AA_SS: i64 = 4;

/// Sub-sampled coverage of a pixel at integer offset `(dx, dy)` from a shape's
/// origin, for anti-aliasing. `inside(fx, fy)` tests a sub-pixel point given in
/// a **2·SS-scaled** coordinate grid (so sub-sample centres land on odd
/// integers, exactly between grid lines — no rounding bias). Returns an alpha
/// 0..=255 = fraction of the `AA_SS²` sub-samples inside the shape. Integer-only
/// (no `sqrt`/float), so it works below the FPU-less boot window too.
fn aa_coverage<F: Fn(i64, i64) -> bool>(dx: i64, dy: i64, inside: F) -> u32 {
    let mut cov = 0u32;
    for sj in 0..AA_SS {
        let fy = 2 * AA_SS * dy + (2 * sj + 1 - AA_SS);
        for si in 0..AA_SS {
            let fx = 2 * AA_SS * dx + (2 * si + 1 - AA_SS);
            if inside(fx, fy) {
                cov += 1;
            }
        }
    }
    cov * 255 / (AA_SS * AA_SS) as u32
}

/// Padding around the status bar's text, on both sides of its short axis.
const STATUS_PAD: u64 = 10;

/// Extra vertical room in the horizontal status bar so FA icons sit fully inside
/// the bar with air above/below body text (not clipped against the edge).
const STATUS_ICON_EXTRA: u64 = 6;

/// How thick the status bar is on a given edge, in pixels.
///
/// Horizontal (top/bottom): text row + icon headroom + padding.
/// Vertical (left/right): a fixed [`crate::panes_layout::STATUS_V_COLS`]-column
/// span, because text cannot run across a column and its content stacks instead.
fn status_thickness(pos: crate::panes_layout::StatusPos, cw: u64, ch: u64) -> u64 {
    if pos.vertical() {
        crate::panes_layout::STATUS_V_COLS * cw + STATUS_PAD
    } else {
        ch + STATUS_ICON_EXTRA + STATUS_PAD
    }
}

/// True for status-bar icons we draw slightly larger (Font Awesome PUA).
fn is_status_icon(ch: char) -> bool {
    crate::icons::is_icon(ch)
}

const OUTER: u64 = 8; // margin around the whole content region
const GAP: u64 = 10; // between the two panes
const BORDER: u64 = 2; // pane border thickness
const PAD: u64 = 10; // interior padding inside a pane
const CHAT_PCT: u64 = 56; // chat pane width as a % of the content region
/// Vertical padding inside the bordered input composer box (px, unscaled).
const COMPOSER_VPAD: u64 = 6;
/// Gap between the composer box and the hint row under it (px, unscaled).
const COMPOSER_HINT_GAP: u64 = 4;
/// Margin between chat scrollback and the composer box (px, unscaled).
const COMPOSER_TOP_GAP: u64 = 8;

/// Pick an integer font scale from the **desktop** height so glyphs stay a
/// legible physical size across resolutions. See
/// [`crate::display::auto_font_scale`] for the thresholds and why they are not a
/// division — the old formula left a 1440p panel at scale 1 (320 columns of 8px
/// text), which is what made a 2K display look broken.
fn pick_scale(height: u64) -> u64 {
    crate::display::auto_font_scale(height)
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
    /// When true, the pane reserves its bottom for a bordered input composer
    /// (bordered box + hint row); the scrollback grid sits above it.
    has_composer: bool,
    /// Expandable folds: `(gi, hidden)` where `gi` is the absolute line index
    /// (same coords as `sel`) of a clickable "▸ N more…" line and `hidden` is
    /// the collapsed text revealed on click. Evicted with the scrollback.
    folds: Vec<(usize, String)>,
    /// Absolute line indices (`hist`+grid, same as `sel`) painted with the
    /// elevated user-prompt band background (`theme.composer_bg`).
    user_band: Vec<usize>,
    /// Incremental UTF-8 decode buffer: the incoming byte stream is decoded one
    /// `char` at a time (a multi-byte glyph spans several `pane_putc` calls).
    utf8: [u8; 4],
    utf8_len: u8,
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
        // bordered composer: box (vpad + 1 line + vpad + 2px border) + gap + hint line.
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
            grid: alloc::vec![('\0', fg); (cols * rows) as usize],
            hist: VecDeque::new(),
            view: 0,
            sel: None,
            has_composer,
            folds: Vec::new(),
            user_band: Vec::new(),
            utf8: [0; 4],
            utf8_len: 0,
        }
    }

    /// Elevated bg for a user-prompt band line, else `None` (use pane bg).
    fn band_bg(&self, gi: usize) -> bool {
        self.user_band.binary_search(&gi).is_ok()
    }

    /// Write `byte` into the grid cell under the cursor (0 erases).
    fn set_cell(&mut self, ch: char) {
        let idx = (self.row * self.cols + self.col) as usize;
        if let Some(c) = self.grid.get_mut(idx) {
            *c = (ch, self.fg);
        }
    }

    /// Feed one incoming byte through the incremental UTF-8 decoder: returns the
    /// decoded `char` once a full sequence lands, `None` while a multi-byte
    /// sequence is still arriving, and `U+FFFD` for an invalid byte. Uses
    /// `core::str::from_utf8` (its `error_len() == None` = "incomplete, need
    /// more"; `Some(_)` = "invalid").
    fn feed_utf8(&mut self, b: u8) -> Option<char> {
        if self.utf8_len == 0 && b < 0x80 {
            return Some(b as char); // ASCII fast path
        }
        if self.utf8_len as usize >= self.utf8.len() {
            self.utf8_len = 0; // safety: never overflow the 4-byte buffer
        }
        self.utf8[self.utf8_len as usize] = b;
        self.utf8_len += 1;
        match core::str::from_utf8(&self.utf8[..self.utf8_len as usize]) {
            Ok(s) => {
                self.utf8_len = 0;
                s.chars().next()
            }
            Err(e) if e.error_len().is_none() => None, // incomplete — await more
            Err(_) => {
                self.utf8_len = 0;
                Some('\u{FFFD}') // invalid byte(s)
            }
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
            let empty: Cell = ('\0', self.default_fg);
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
            // Keep the *new* theme's default_fg (set by Pane::new); only carry
            // an explicit non-default ANSI colour across, and recolour history.
            let old_fg = old.default_fg;
            self.fg = if old.fg == old_fg { self.default_fg } else { old.fg };
            self.bold = old.bold;
            self.recolor_default_fg(old_fg, self.default_fg);
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
        let empty: Cell = ('\0', self.default_fg);
        // Same layout as textsel::Cell — Rgb is (u8,u8,u8).
        let as_ts: alloc::vec::Vec<alloc::vec::Vec<crate::textsel::Cell>> =
            abs.iter().map(|l| l.iter().map(|&(b, c)| (b, c)).collect()).collect();
        let reflowed = crate::textsel::reflow_lines(&as_ts, ocols, cols, ('\0', self.default_fg));
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
        let old_fg = old.default_fg;
        self.fg = if old.fg == old_fg { self.default_fg } else { old.fg };
        self.bold = old.bold;
        self.recolor_default_fg(old_fg, self.default_fg);
    }

    /// Recolour adopted content after a theme switch: cells drawn in the old
    /// theme's default foreground (plain shell/agent text — the bulk of the
    /// scrollback) are remapped to the new theme's foreground, so switching
    /// e.g. dark→light doesn't leave the existing history invisible (light-on-
    /// light). Explicitly ANSI/syntax-coloured cells keep their colour.
    fn recolor_default_fg(&mut self, old_fg: Rgb, new_fg: Rgb) {
        if old_fg == new_fg {
            return;
        }
        for c in self.grid.iter_mut() {
            if c.1 == old_fg {
                c.1 = new_fg;
            }
        }
        for line in self.hist.iter_mut() {
            for c in line.iter_mut() {
                if c.1 == old_fg {
                    c.1 = new_fg;
                }
            }
        }
    }

    /// Drop all text content (grid + scrollback) — the `/clear` reset.
    fn clear_content(&mut self) {
        for c in self.grid.iter_mut() {
            *c = ('\0', self.default_fg);
        }
        self.hist.clear();
        self.view = 0;
        self.sel = None;
        self.folds.clear();
        self.user_band.clear();
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
    /// Action columns (1..=8). Index 0 is the primary; focused column is
    /// [`Self::focused_action`].
    actions: Vec<ActionSlot>,
    /// Which action column has keyboard/mouse focus for tabs and open targets.
    focused_action: usize,
    /// Status-bar text (left = brand, right = datetime); set by the shell from
    /// the UI-config templates + clock, so it stays configurable.
    status_left: String,
    status_right: String,
    /// Blinking-caret state for the chat pane / composer.
    caret_on: bool,
    caret_last_ms: u64,
    /// bordered input composer (bottom of chat pane): when `composer_active`
    /// the caret lives in the bordered box, not the scrollback grid.
    composer_active: bool,
    composer_line: String,
    composer_cur: usize,
    /// Hint bar under the composer (left / right halves).
    composer_hint_l: String,
    composer_hint_r: String,
    /// Per-character colours for the **leading** cells of `composer_hint_l`
    /// (empty = the whole hint takes `theme.composer_hint`). This is what lets
    /// the shell paint its gradient progress bar into the hint bar without the
    /// framebuffer knowing anything about the animation.
    composer_hint_l_lead: alloc::vec::Vec<Rgb>,
    /// Slash-command / @file / path-argument suggestion popup above the composer.
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
    /// Physical framebuffer size, as the firmware reported it. `width`/`height`
    /// are the **logical** desktop, which may be smaller (a letterboxed viewport).
    fb_w: u64,
    fb_h: u64,
    /// Top-left of the logical desktop inside the physical framebuffer. Both zero
    /// when the desktop is native.
    origin_x: u64,
    origin_y: u64,
    /// The requested logical desktop, carried across rebuilds. `None` = native.
    logical_pref: Option<(u64, u64)>,
    /// The status bar's rect and the content rect left over, from
    /// `panes_layout::status_split`. Every pane-layout calculation works inside
    /// `content_*` rather than `0..width`/`0..height`, so the bar can sit on any
    /// edge without a second set of layout paths. At `Top` (the default) the
    /// content origin is `(0, bar_h)`; at `Bottom` it is `(0, 0)`.
    status_rect: crate::panes_layout::Rect,
    content_x: u64,
    content_y: u64,
    content_w: u64,
    content_h: u64,
    /// Whether keyboard focus is on an action column (vs the shell/chat).
    focus_action: bool,
    /// Action column currently highlighted as a tab drag's drop target.
    drop_target: Option<usize>,
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
    /// Background patch saved beneath the cursor sprite; sized to the last-drawn
    /// sprite (`cur_sw`×`cur_sh`) so a theme's custom (variable-size) cursor can
    /// be restored with the exact dims it was drawn at.
    cur_saved: Vec<Rgb>,
    cur_sw: u64,
    cur_sh: u64,
    /// Decoded wallpaper scaled to the full screen (`0x00RRGGBB`, width×height),
    /// or `None` for the solid-colour desktop. Windows blend over it at
    /// [`Self::opacity`]; the gutters show it directly.
    wallpaper: Option<Vec<u32>>,
    /// Window opacity over the wallpaper (255 = opaque; only used when
    /// `wallpaper` is `Some`).
    opacity: u8,
}

// Mouse cursor sprites: 0 = transparent, 1 = fill, 2 = outline.
// Shapes: Arrow (default), Hand (link pointer), IBeam (text input).
const CUR_W: u64 = 12;
const CUR_H: u64 = 19;

/// OS cursor shape (CSS `cursor` subset — Ladybird pointer/text/default).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum CursorShape {
    Arrow = 0,
    Hand = 1,
    IBeam = 2,
    Wait = 3,
}

static CURSOR_SHAPE: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(0);

/// Set the mouse cursor glyph (restored on next move/draw).
pub fn set_cursor_shape(shape: CursorShape) {
    CURSOR_SHAPE.store(shape as u8, core::sync::atomic::Ordering::Relaxed);
}

pub fn cursor_shape() -> CursorShape {
    match CURSOR_SHAPE.load(core::sync::atomic::Ordering::Relaxed) {
        1 => CursorShape::Hand,
        2 => CursorShape::IBeam,
        3 => CursorShape::Wait,
        _ => CursorShape::Arrow,
    }
}

#[rustfmt::skip]
const CURSOR_ARROW: [u8; (CUR_W * CUR_H) as usize] = [
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

// Pointing hand (hotspot near tip of index finger, top-leftish).
#[rustfmt::skip]
const CURSOR_HAND: [u8; (CUR_W * CUR_H) as usize] = [
    0,0,0,2,2,0,0,0,0,0,0,0,
    0,0,2,1,1,2,0,0,0,0,0,0,
    0,0,2,1,1,2,0,0,0,0,0,0,
    0,0,2,1,1,2,2,2,0,0,0,0,
    0,0,2,1,1,2,1,1,2,0,0,0,
    2,2,2,1,1,2,1,1,2,2,0,0,
    2,1,1,1,1,1,1,1,1,1,2,0,
    2,1,1,1,1,1,1,1,1,1,2,0,
    2,1,1,1,1,1,1,1,1,1,2,0,
    0,2,1,1,1,1,1,1,1,2,0,0,
    0,2,1,1,1,1,1,1,1,2,0,0,
    0,0,2,1,1,1,1,1,2,0,0,0,
    0,0,2,1,1,1,1,1,2,0,0,0,
    0,0,0,2,1,1,1,2,0,0,0,0,
    0,0,0,2,1,1,1,2,0,0,0,0,
    0,0,0,0,2,1,2,0,0,0,0,0,
    0,0,0,0,2,1,2,0,0,0,0,0,
    0,0,0,0,0,2,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,
];

// I-beam for text fields.
#[rustfmt::skip]
const CURSOR_IBEAM: [u8; (CUR_W * CUR_H) as usize] = [
    2,2,2,0,0,2,2,2,0,0,0,0,
    0,0,0,2,2,0,0,0,0,0,0,0,
    0,0,0,0,2,0,0,0,0,0,0,0,
    0,0,0,0,2,0,0,0,0,0,0,0,
    0,0,0,0,2,0,0,0,0,0,0,0,
    0,0,0,0,2,0,0,0,0,0,0,0,
    0,0,0,0,2,0,0,0,0,0,0,0,
    0,0,0,0,2,0,0,0,0,0,0,0,
    0,0,0,0,2,0,0,0,0,0,0,0,
    0,0,0,0,2,0,0,0,0,0,0,0,
    0,0,0,0,2,0,0,0,0,0,0,0,
    0,0,0,0,2,0,0,0,0,0,0,0,
    0,0,0,0,2,0,0,0,0,0,0,0,
    0,0,0,0,2,0,0,0,0,0,0,0,
    0,0,0,0,2,0,0,0,0,0,0,0,
    0,0,0,0,2,0,0,0,0,0,0,0,
    0,0,0,2,2,2,0,0,0,0,0,0,
    2,2,2,0,0,2,2,2,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,
];

// Hourglass-ish wait (loading).
#[rustfmt::skip]
const CURSOR_WAIT: [u8; (CUR_W * CUR_H) as usize] = [
    2,2,2,2,2,2,2,2,0,0,0,0,
    2,1,1,1,1,1,1,2,0,0,0,0,
    0,2,1,1,1,1,2,0,0,0,0,0,
    0,0,2,1,1,2,0,0,0,0,0,0,
    0,0,0,2,2,0,0,0,0,0,0,0,
    0,0,0,2,2,0,0,0,0,0,0,0,
    0,0,2,1,1,2,0,0,0,0,0,0,
    0,2,1,1,1,1,2,0,0,0,0,0,
    2,1,1,1,1,1,1,2,0,0,0,0,
    2,2,2,2,2,2,2,2,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,
];

/// Built-in fixed 12×19 sprite for a shape (index bitmap: 0/1/2).
fn cursor_builtin(shape: CursorShape) -> &'static [u8; (CUR_W * CUR_H) as usize] {
    match shape {
        CursorShape::Hand => &CURSOR_HAND,
        CursorShape::IBeam => &CURSOR_IBEAM,
        CursorShape::Wait => &CURSOR_WAIT,
        CursorShape::Arrow => &CURSOR_ARROW,
    }
}

/// A theme-supplied cursor bitmap: `w`×`h` index values (0 transparent / 1 fill
/// / 2 outline), same encoding as the built-ins. Dimensions are capped at
/// [`CUR_MAX`] per side so the save buffer stays bounded.
#[derive(Clone, Default)]
pub struct CursorSprite {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u8>,
}

/// Max cursor sprite dimension per side (bounds the RMW save patch).
pub const CUR_MAX: usize = 32;

/// Live cursor theme: fill/outline colours + optional per-shape custom sprites
/// (order Arrow, Hand, IBeam, Wait). `None` ⇒ brand defaults + built-in sprites.
struct CursorTheme {
    fill: Rgb,
    outline: Rgb,
    sprites: Option<[CursorSprite; 4]>,
}

/// Default cursor colours (brand): near-black fill, cream outline.
const CURSOR_FILL_DEFAULT: Rgb = (15, 15, 17);
const CURSOR_OUTLINE_DEFAULT: Rgb = (245, 245, 248);

static CURSOR_THEME: Locked<CursorTheme> = Locked::new(CursorTheme {
    fill: CURSOR_FILL_DEFAULT,
    outline: CURSOR_OUTLINE_DEFAULT,
    sprites: None,
});

/// Set the cursor fill/outline colours (from `ui.json`), preserving any custom
/// sprites already installed by a theme.
pub fn set_cursor_colors(fill: Rgb, outline: Rgb) {
    CURSOR_THEME.with(|t| {
        t.fill = fill;
        t.outline = outline;
    });
}

/// Install (or clear, with `None`) the per-shape custom cursor sprites (from a
/// theme file), preserving the current colours. A sprite with zero dims falls
/// back to the built-in bitmap for that shape.
pub fn set_cursor_sprites(sprites: Option<[CursorSprite; 4]>) {
    CURSOR_THEME.with(|t| t.sprites = sprites);
}

/// Reset the cursor to brand colours + built-in sprites.
pub fn reset_cursor_theme() {
    CURSOR_THEME.with(|t| {
        t.fill = CURSOR_FILL_DEFAULT;
        t.outline = CURSOR_OUTLINE_DEFAULT;
        t.sprites = None;
    });
}

/// Cached Font Awesome cursor sprites (Arrow/Hand/IBeam/Wait), built once the
/// first time each shape is drawn. Falls back to the hand-drawn bitmaps if FA
/// is not yet registered or rasterization fails.
static FA_CURSOR_CACHE: Locked<[Option<(u64, u64, Vec<u8>)>; 4]> =
    Locked::new([None, None, None, None]);

fn cursor_fa_sprite(shape: CursorShape) -> Option<(u64, u64, Vec<u8>)> {
    let i = shape as usize;
    FA_CURSOR_CACHE.with(|cache| {
        if let Some(hit) = &cache[i] {
            return Some(hit.clone());
        }
        let ch = crate::icons::cursor_glyph(shape as u8);
        // ~18 px matches the classic 12×19 hand-drawn arrow optical weight.
        let Some((w, h, data)) = crate::font_ttf::raster_cursor_sprite(ch, 18.0) else {
            return None;
        };
        if w == 0 || h == 0 || data.is_empty() {
            return None;
        }
        let entry = (w as u64, h as u64, data);
        cache[i] = Some(entry.clone());
        Some(entry)
    })
}

/// Resolve the active cursor for the current shape: `(w, h, index-data,
/// fill, outline)`. Order: theme custom sprite → Font Awesome Free Solid
/// (`arrow-pointer` / `hand-pointer` / `i-cursor` / `hourglass`) → hand-drawn
/// built-in bitmap.
fn cursor_active() -> (u64, u64, alloc::borrow::Cow<'static, [u8]>, Rgb, Rgb) {
    let shape = cursor_shape();
    CURSOR_THEME.with(|t| {
        let (fill, outline) = (t.fill, t.outline);
        if let Some(sprites) = &t.sprites {
            let sp = &sprites[shape as usize];
            if sp.w > 0 && sp.h > 0 && sp.data.len() >= sp.w * sp.h {
                return (
                    sp.w as u64,
                    sp.h as u64,
                    alloc::borrow::Cow::Owned(sp.data.clone()),
                    fill,
                    outline,
                );
            }
        }
        if let Some((w, h, data)) = cursor_fa_sprite(shape) {
            return (w, h, alloc::borrow::Cow::Owned(data), fill, outline);
        }
        (
            CUR_W,
            CUR_H,
            alloc::borrow::Cow::Borrowed(&cursor_builtin(shape)[..]),
            fill,
            outline,
        )
    })
}

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

/// A snapshot of one action pane's interior geometry, copied out so a painter
/// can keep using it while it mutates the screen (no borrow held on `actions`).
#[derive(Clone, Copy)]
struct PaneDims {
    /// Outer box origin (frame, not interior).
    x: u64,
    /// Interior origin.
    ix: u64,
    iy: u64,
    /// Interior size in pixels.
    w: u64,
    iw: u64,
    ih: u64,
    cw: u64,
    ch: u64,
    cols: u64,
    rows: u64,
    bg: Rgb,
}

impl PaneDims {
    fn of(p: &Pane) -> PaneDims {
        PaneDims {
            x: p.x,
            ix: p.ix,
            iy: p.iy,
            w: p.w,
            iw: p.cols * p.cw,
            ih: p.rows * p.ch,
            cw: p.cw,
            ch: p.ch,
            cols: p.cols,
            rows: p.rows,
            bg: p.bg,
        }
    }
}

/// One action pane in the grid: its geometry + tmux-style tab list.
struct ActionSlot {
    pane: Pane,
    tabs: Vec<RightMode>,
    active: usize,
}

impl ActionSlot {
    fn right(&self) -> RightMode {
        self.tabs.get(self.active).copied().unwrap_or(RightMode::Closed)
    }

    fn is_empty(&self) -> bool {
        self.tabs.is_empty()
    }
}

/// Surface id the `/open` image viewer uses (also known to the shell). A
/// `Surface(IMAGE_SURFACE)` tab is labelled "image" in the tab bar.
pub const IMAGE_SURFACE: u32 = u32::MAX;
/// Surface id the `/open` video player presents frames on (labelled "video").
pub const VIDEO_SURFACE: u32 = u32::MAX - 1;
/// Surface id the browser agent paints pages on (labelled "browser").
pub const BROWSER_SURFACE: u32 = u32::MAX - 2;

/// The short tab-bar label for a view (Font Awesome glyph + name).
///
/// Package-UI agent surfaces use the **agent name** (chess, paint, …) rather
/// than the generic "surface" — the action-pane window title tracks the app.
fn tab_label(m: RightMode) -> alloc::string::String {
    use crate::icons::fa;
    match m {
        RightMode::Closed => alloc::string::String::new(),
        RightMode::Ktrace => alloc::format!("{} ktrace", fa::BUG),
        RightMode::Editor => alloc::format!("{} editor", fa::PEN_TO_SQUARE),
        RightMode::Top => alloc::format!("{} top", fa::GAUGE),
        RightMode::Todos => alloc::format!("{} todos", fa::LIST_CHECK),
        RightMode::Audio => alloc::format!("{} audio", fa::WAVE_SQUARE),
        RightMode::Surface(IMAGE_SURFACE) => alloc::format!("{} image", fa::IMAGE),
        RightMode::Surface(VIDEO_SURFACE) => alloc::format!("{} video", fa::FILM),
        RightMode::Surface(BROWSER_SURFACE) => alloc::format!("{} browser", fa::GLOBE),
        RightMode::Surface(id) => {
            // Running package UI (chess/paint/snake/…) — FA agent icon + name.
            // Use surface_tab_name (display cache), never RUN: tab paint runs
            // while SCREEN is held, often mid-present from a guest host import.
            if let Some(name) = crate::service::package_ui::surface_tab_name(id) {
                let icon = crate::icons::for_agent(&name);
                return alloc::format!("{icon} {name}");
            }
            alloc::format!("{} surface-{id}", fa::WINDOW)
        }
    }
}

/// Config knobs the UI config (`/configs/core/ui.json`) can set for the layout.
#[derive(Clone)]
pub struct LayoutCfg {
    /// Chat pane width as a % of the content region (10..90).
    pub chat_pct: u64,
    /// Total panes including the shell (2..=9). Action panes = max_panes - 1.
    pub max_panes: u8,
    /// The action band's grid shape and per-track weights. `cols * rows` is the
    /// action-pane count, so it and `max_panes` are kept consistent by
    /// [`set_max_panes`] / [`set_grid`].
    pub grid: crate::panes_layout::GridSpec,
    /// Font scale; 0 = auto from panel height.
    pub scale: u64,
    /// Put the chat pane on the right instead of the left.
    pub swap: bool,
    /// Which desktop edge the OS status bar occupies. Everything else lays out
    /// inside the leftover content rect, so this shifts the whole UI.
    pub status_pos: crate::panes_layout::StatusPos,
    pub chat_title: String,
    pub logs_title: String,
    /// Colour palette (from `ui.json` `theme`; default = brand dark).
    pub theme: Theme,
    /// Show the boot splash (logo + name). Default true.
    pub splash: bool,
    /// Fullscreen state: 0 = normal split, 1 = chat fills the screen, 2 = the
    /// action pane fills the screen. Toggled at runtime (F11 / `/fullscreen`).
    pub fullscreen: u8,
    /// Wallpaper spec: `""` = solid `screen_bg`; `"gradient:#rrggbb,#rrggbb"` =
    /// generated vertical gradient; otherwise a path to an image in the store.
    pub wallpaper: String,
    /// Window opacity over the wallpaper (0..=255; 255 = opaque, the default —
    /// identical to the no-wallpaper look). Only meaningful with a wallpaper.
    pub opacity: u8,
}

impl Default for LayoutCfg {
    fn default() -> Self {
        LayoutCfg {
            chat_pct: CHAT_PCT,
            max_panes: crate::panes_layout::MAX_PANES_DEFAULT,
            grid: crate::panes_layout::GridSpec::even(1, 1),
            scale: 0,
            swap: false,
            status_pos: crate::panes_layout::StatusPos::default(),
            chat_title: String::from("Shell Agent"),
            logs_title: String::from("ktrace"),
            theme: Theme::default(),
            splash: true,
            fullscreen: 0,
            wallpaper: String::new(),
            opacity: 255,
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
        // Default: max_panes=2 → action band closed until first tab. max_panes>2
        // shows empty action columns as drop targets from boot.
        let cfg = crate::ui_config::boot_layout();
        let band = crate::panes_layout::action_band_visible(cfg.max_panes, false);
        Screen::build(
            addr,
            width,
            height,
            pitch,
            bpp_bytes,
            r_shift,
            g_shift,
            b_shift,
            &cfg,
            band,
            0,
            None, // native until `display.json` is applied
        )
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
        focused: usize,
        logical_pref: Option<(u64, u64)>,
    ) -> Screen {
        // `width`/`height` arrive as the PHYSICAL framebuffer. A logical
        // preference turns them into a centred viewport; everything below this
        // point lays out against the logical size only, so the whole compositor
        // is resolution-agnostic and needs no other change.
        let (fb_w, fb_h) = (width, height);
        let (origin_x, origin_y, width, height) = match logical_pref {
            Some((lw, lh)) => {
                let (x, y, w, h) = crate::display::viewport(
                    (fb_w as u32, fb_h as u32),
                    (lw as u32, lh as u32),
                );
                (x as u64, y as u64, w as u64, h as u64)
            }
            None => (0, 0, fb_w, fb_h), // native: identity, byte-for-byte the old path
        };
        // Font scale follows the LOGICAL height, so a smaller desktop gets
        // proportionally sized text rather than the panel's.
        let scale = if cfg.scale > 0 { cfg.scale } else { pick_scale(height) };
        let cw = CELL_W * scale;
        let ch = CELL_H * scale;
        // Carve the status bar off its edge; everything below lays out in what is
        // left. A vertical bar is a fixed column of cells wide (see
        // `STATUS_V_COLS`), a horizontal one a single text row plus padding.
        let (status_rect, (content_x, content_y, content_w, content_h)) =
            crate::panes_layout::status_split(
                width,
                height,
                cfg.status_pos,
                status_thickness(cfg.status_pos, cw, ch),
            );
        let box_y = content_y + OUTER;
        let box_h = content_h.saturating_sub(2 * OUTER);
        let pct = cfg.chat_pct.clamp(10, 90);
        let grid = cfg.grid.sanitized();
        let n_act = grid.len();
        let full_w = content_w.saturating_sub(2 * OUTER);
        let th = cfg.theme;
        let focused = focused.min(n_act - 1);
        // `split_band` works in a 0-based span, so its x results are shifted into
        // the content rect. With a left-edge status bar `content_x` is the bar's
        // width; at every other position it is 0 and this is the identity.
        let band_split = || {
            let (cx, cwid, bx, bw) =
                crate::panes_layout::split_band(content_w, OUTER, GAP, pct, true, cfg.swap);
            (content_x + cx, cwid, content_x + bx, bw)
        };
        // Parked panes keep w==0 so `Pane::adopt` clones content without a
        // catastrophic 1-column reflow of the full scrollback.
        let parked = (content_x + content_w, box_y, 0u64, box_h);
        let mut action_boxes = if cfg.fullscreen == 2 && split {
            // The **focused** action pane fills the screen; chat + every other
            // pane park. Maximising cell 0 regardless of focus would show a
            // different pane than the one the user was working in.
            let mut boxes = alloc::vec![parked; n_act];
            boxes[focused] = (content_x + OUTER, box_y, full_w, box_h);
            boxes
        } else if cfg.fullscreen == 1 || !split {
            // Chat fills; the whole action grid parks.
            alloc::vec![parked; n_act]
        } else {
            let (_, _, bx, bw) = band_split();
            crate::panes_layout::layout_grid(bx, box_y, bw, box_h, GAP, &grid)
        };
        // Keep the vector's length pinned to the cell count regardless.
        action_boxes.resize(n_act, parked);
        let (chat_x, chat_bw) = if cfg.fullscreen == 2 && split {
            (content_x + content_w, 0)
        } else if cfg.fullscreen == 1 || !split {
            (content_x + OUTER, full_w)
        } else {
            let (cx, cwid, ..) = band_split();
            (cx, cwid)
        };
        let chat = Pane::new(chat_x, box_y, chat_bw, box_h, cw, ch, th.chat_fg, th.chat_bg, cfg.chat_title.clone(), true);
        let mut actions = alloc::vec::Vec::with_capacity(n_act);
        for (i, &(ax, ay, aw, ah)) in action_boxes.iter().enumerate() {
            let title = if i == 0 {
                cfg.logs_title.clone()
            } else {
                alloc::format!("action {}", i + 1)
            };
            actions.push(ActionSlot {
                pane: Pane::new(ax, ay, aw, ah, cw, ch, th.logs_fg, th.logs_bg, title, false),
                tabs: Vec::new(),
                active: 0,
            });
        }
        let mut status_left = String::from("ChittiOS v");
        status_left.push_str(crate::VERSION);
        let mut scr = Screen {
            addr, width, height, pitch, bpp_bytes, r_shift, g_shift, b_shift, scale, chat,
            fb_w,
            fb_h,
            origin_x,
            origin_y,
            logical_pref,
            status_rect,
            content_x,
            content_y,
            content_w,
            content_h,
            actions,
            focused_action: focused,
            status_left,
            status_right: String::new(),
            caret_on: true,
            caret_last_ms: 0,
            composer_active: false,
            composer_line: String::new(),
            composer_cur: 0,
            composer_hint_l: String::from("↑↓ history · Tab select · Ctrl+P/N pick · Ctrl+R search · Enter send · /cmds · @files"),
            composer_hint_r: String::new(),
            composer_hint_l_lead: alloc::vec::Vec::new(),
            suggest_open: false,
            suggest_items: alloc::vec::Vec::new(),
            suggest_sel: 0,
            suggest_rect: None,
            blink_seen_ms: u64::MAX,
            blink_calls: 0,
            clock_alive: false,
            focus_action: false,
            drop_target: None,
            layout: cfg.clone(),
            theme: th,
            cur_x: width / 2,
            cur_y: height / 2,
            cur_vis: false,
            cur_active: false,
            cur_saved: Vec::new(),
            cur_sw: CUR_W,
            cur_sh: CUR_H,
            wallpaper: None,
            opacity: 255,
        };
        // Decode/generate the wallpaper once for this layout (windows blend over
        // it at `opacity`); recomputed on relayout.
        scr.set_wallpaper(&cfg.wallpaper, cfg.opacity);
        scr
    }


    fn cw(&self) -> u64 {
        CELL_W * self.scale
    }
    fn ch(&self) -> u64 {
        CELL_H * self.scale
    }

    /// Focused action slot (falls back to 0).
    fn focused_slot(&self) -> &ActionSlot {
        let i = self.focused_action.min(self.actions.len().saturating_sub(1));
        &self.actions[i]
    }
    fn focused_slot_mut(&mut self) -> &mut ActionSlot {
        let i = self.focused_action.min(self.actions.len().saturating_sub(1));
        &mut self.actions[i]
    }
    /// Active tab of the focused action column.
    fn right(&self) -> RightMode {
        if self.actions.is_empty() {
            RightMode::Closed
        } else {
            self.focused_slot().right()
        }
    }
    /// True if any action column has at least one tab.
    fn any_action_open(&self) -> bool {
        self.actions.iter().any(|a| !a.is_empty())
    }
    /// Primary geometry pane for the focused action column (`logs` legacy).
    fn logs(&self) -> &Pane {
        &self.focused_slot().pane
    }
    fn logs_mut(&mut self) -> &mut Pane {
        &mut self.focused_slot_mut().pane
    }
    /// Whether action pane `i` should be painted.
    ///
    /// A parked pane (`w == 0`, fullscreen or a collapsed band) never paints. An
    /// *empty* pane paints its frame only in a multi-pane grid, where it is a
    /// visible drop target; a lone action pane collapses when its last tab
    /// closes, keeping the classic two-pane look byte-identical.
    ///
    /// The test is the **grid's own** pane count, not `layout.max_panes`: the two
    /// are kept in sync but deriving visibility from the geometry that is
    /// actually laid out means a stale config can't make them disagree.
    fn column_visible(&self, i: usize) -> bool {
        let Some(a) = self.actions.get(i) else { return false };
        if a.pane.w == 0 {
            return false;
        }
        !a.is_empty() || self.actions.len() > 1
    }

    /// The visible pane of the action column whose **active** tab is `mode`.
    ///
    /// The per-view painters (`draw_top`, the audio/video/browser HUDs, the
    /// editor) resolve their target through this rather than the focused column:
    /// a `/top` tab on column 2 must keep refreshing while the user works in
    /// column 1, and it must paint into its own column's rectangle.
    fn mode_dims(&self, mode: RightMode) -> Option<PaneDims> {
        let i = self.mode_column(mode)?;
        Some(PaneDims::of(&self.actions[i].pane))
    }

    /// Index of the visible action column whose **active** tab is `mode`.
    fn mode_column(&self, mode: RightMode) -> Option<usize> {
        (0..self.actions.len())
            .find(|&i| self.actions[i].right() == mode && self.actions[i].pane.w > 0)
    }

    /// Whether any action column is painted — i.e. the band is up, so keyboard
    /// focus can move to it. With `max_panes > 2` an *empty* column is visible
    /// and focusable (it is a drop target), which is why this is not the same
    /// question as "does the focused column have a tab".
    fn any_column_visible(&self) -> bool {
        (0..self.actions.len()).any(|i| self.column_visible(i))
    }

    /// Find which column owns `mode`.
    fn find_mode(&self, mode: RightMode) -> Option<(usize, usize)> {
        for (pi, a) in self.actions.iter().enumerate() {
            if let Some(ti) = a.tabs.iter().position(|&m| m == mode) {
                return Some((pi, ti));
            }
        }
        None
    }

    // --- pixel plumbing --------------------------------------------------

    /// Byte offset into the framebuffer for **logical** pixel `(x, y)`.
    ///
    /// The single place the logical desktop is translated into physical pixels.
    /// `width`/`height` are the *logical* desktop (what every layout is computed
    /// against) and `origin_x`/`origin_y` place it inside the real framebuffer, so
    /// a smaller-than-native resolution is a centred, letterboxed viewport that
    /// still renders 1:1 — glyphs are rasterised at physical pixels, nothing is
    /// scaled, so text stays crisp. When the desktop is native both origins are 0
    /// and this is the identity, i.e. the default path is unchanged.
    #[inline]
    fn fb_offset(&self, x: u64, y: u64) -> u64 {
        (y + self.origin_y) * self.pitch + (x + self.origin_x) * self.bpp_bytes
    }

    /// Fill a rectangle in **physical** framebuffer coordinates, bypassing the
    /// logical viewport. Only the letterbox uses this — everything else must go
    /// through the logical path so it lands inside the desktop.
    fn fill_phys(&self, x: u64, y: u64, w: u64, h: u64, c: Rgb) {
        let value = self.pack_rgb(c);
        if x >= self.fb_w {
            return;
        }
        let n = w.min(self.fb_w - x);
        for yy in y..(y + h).min(self.fb_h) {
            let offset = yy * self.pitch + x * self.bpp_bytes;
            // SAFETY: clipped to the physical framebuffer, which is kernel-owned
            // MMIO for its full `fb_h * pitch` extent.
            unsafe {
                let ptr = (self.addr as *mut u8).add(offset as usize);
                if self.bpp_bytes == 4 {
                    let dst = ptr as *mut u32;
                    for i in 0..n {
                        dst.add(i as usize).write_volatile(value);
                    }
                } else {
                    for i in 0..n {
                        let p = ptr.add((i * self.bpp_bytes) as usize);
                        for b in 0..self.bpp_bytes {
                            p.add(b as usize).write_volatile((value >> (b * 8)) as u8);
                        }
                    }
                }
            }
        }
    }

    /// Paint the dead space around a smaller-than-native desktop.
    ///
    /// A no-op at native resolution (the common case), so the default boot path
    /// touches nothing extra. Painted black rather than the theme background: it
    /// is outside the desktop, so it should read as "no screen here" rather than
    /// as an oversized margin.
    fn paint_letterbox(&self) {
        if self.origin_x == 0
            && self.origin_y == 0
            && self.width == self.fb_w
            && self.height == self.fb_h
        {
            return;
        }
        let black = (0, 0, 0);
        let (ox, oy) = (self.origin_x, self.origin_y);
        self.fill_phys(0, 0, self.fb_w, oy, black); // above
        let below = oy + self.height;
        self.fill_phys(0, below, self.fb_w, self.fb_h.saturating_sub(below), black);
        self.fill_phys(0, oy, ox, self.height, black); // left
        let right = ox + self.width;
        self.fill_phys(right, oy, self.fb_w.saturating_sub(right), self.height, black);
    }

    fn put_pixel(&self, x: u64, y: u64, c: Rgb) {
        if x >= self.width || y >= self.height {
            return;
        }
        // NB: damage is NOT tracked here. A redraw is millions of put_pixel calls
        // and a per-pixel union would cost more than the flush it feeds. The coarse
        // painters (`fill_rect`, `blit_rgb32_row`, `redraw`) report damage instead,
        // and they are what every glyph and frame ultimately goes through.
        let offset = self.fb_offset(x, y);
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
        crate::kms::damage(
            (x + self.origin_x) as u32,
            (y + self.origin_y) as u32,
            row.len() as u32,
            1,
        );
        let n = row.len().min((self.width - x) as usize);
        let offset = self.fb_offset(x, y);
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
        // Report to KMS in *physical* coordinates: a driver's scanout is the whole
        // framebuffer, not the logical desktop, so the viewport origin must be added.
        crate::kms::damage(
            (x + self.origin_x) as u32,
            (y + self.origin_y) as u32,
            w as u32,
            h as u32,
        );
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

    /// Build [`Self::wallpaper`] (scaled to the full screen) from a spec:
    /// `""` → none (solid desktop); `"gradient:#aabbcc,#112233"` → a generated
    /// vertical gradient; otherwise a path to an image in the synapse store,
    /// decoded and stretched to fill. Also sets [`Self::opacity`]. Called from
    /// `build`/`relayout` so it's recomputed once per layout change, not per
    /// redraw.
    fn set_wallpaper(&mut self, spec: &str, opacity: u8) {
        self.opacity = opacity;
        let (w, h) = (self.width as usize, self.height as usize);
        if w == 0 || h == 0 {
            self.wallpaper = None;
            return;
        }
        if spec.is_empty() {
            self.wallpaper = None;
            return;
        }
        if let Some(rest) = spec.strip_prefix("gradient:") {
            // Two `#rrggbb` stops, top → bottom.
            let mut it = rest.split(',');
            let a = parse_hex(it.next().unwrap_or("").trim(), self.theme.screen_bg);
            let b = parse_hex(it.next().unwrap_or("").trim(), a);
            let mut buf = alloc::vec![0u32; w * h];
            for y in 0..h {
                // t in 0..=255 down the screen.
                let t = if h > 1 { (y * 255 / (h - 1)) as u32 } else { 0 };
                let mix = |ca: u8, cb: u8| ((ca as u32 * (255 - t) + cb as u32 * t) / 255) & 0xff;
                let px = (mix(a.0, b.0) << 16) | (mix(a.1, b.1) << 8) | mix(a.2, b.2);
                for x in 0..w {
                    buf[y * w + x] = px;
                }
            }
            self.wallpaper = Some(buf);
            return;
        }
        // Image path: read from the store, decode, cover-scale to fill the
        // screen (preserve aspect, centre-crop — no stretch/distortion).
        self.wallpaper = crate::synapse::fs::read(spec)
            .and_then(|bytes| crate::image::decode(&bytes).ok())
            .map(|img| crate::image::cover(&img, w, h))
            .map(|img| img.pixels);
    }

    /// Paint a **desktop/gutter** region: the wallpaper (if any) shown directly,
    /// else a solid `fallback` fill.
    fn paint_wallpaper(&self, x: u64, y: u64, w: u64, h: u64, fallback: Rgb) {
        let Some(wp) = &self.wallpaper else {
            self.fill_rect(x, y, w, h, fallback);
            return;
        };
        if w == 0 || h == 0 {
            return;
        }
        for dy in 0..h {
            let sy = y + dy;
            if sy >= self.height {
                break;
            }
            let base = (sy * self.width + x) as usize;
            let n = w.min(self.width.saturating_sub(x)) as usize;
            if x < self.width && base + n <= wp.len() {
                self.blit_rgb32_row(x, sy, &wp[base..base + n]);
            }
        }
    }

    /// Paint a **window surface** region: `color` blended over the wallpaper at
    /// [`Self::opacity`] (255 = opaque = plain `color`), else a solid `color`
    /// fill when there's no wallpaper. One blended row is built and blitted per
    /// scanline (no per-pixel readback).
    fn paint_surface(&self, x: u64, y: u64, w: u64, h: u64, color: Rgb) {
        let Some(wp) = &self.wallpaper else {
            self.fill_rect(x, y, w, h, color);
            return;
        };
        if w == 0 || h == 0 {
            return;
        }
        let op = self.opacity as u32;
        if op >= 255 {
            self.fill_rect(x, y, w, h, color);
            return;
        }
        let inv = 255 - op;
        let (cr, cg, cb) = (color.0 as u32, color.1 as u32, color.2 as u32);
        let mut row = alloc::vec![0u32; w as usize];
        for dy in 0..h {
            let sy = y + dy;
            if sy >= self.height {
                break;
            }
            for dx in 0..w {
                let sx = x + dx;
                let wpix = if sx < self.width {
                    wp[(sy * self.width + sx) as usize]
                } else {
                    0
                };
                let r = ((((wpix >> 16) & 0xff) * inv + cr * op) / 255) & 0xff;
                let g = ((((wpix >> 8) & 0xff) * inv + cg * op) / 255) & 0xff;
                let b = (((wpix & 0xff) * inv + cb * op) / 255) & 0xff;
                row[dx as usize] = (r << 16) | (g << 8) | b;
            }
            self.blit_rgb32_row(x, sy, &row);
        }
    }

    /// Fill a small region's background honouring a translucent wallpaper — the
    /// cell-level analog of [`Self::paint_surface`], but with no per-call heap
    /// allocation (cells are tiny). Text-grid cells use this so the see-through
    /// desktop shows behind **every** cell, not only behind painted glyphs
    /// (`blit_glyph` already blends via [`Self::bg_at`]); an opaque wallpaper /
    /// no-wallpaper falls back to the solid [`Self::fill_rect`] fast path.
    fn fill_cell_bg(&self, x: u64, y: u64, w: u64, h: u64, bg: Rgb) {
        if self.wallpaper.is_none() || self.opacity >= 255 {
            self.fill_rect(x, y, w, h, bg);
            return;
        }
        for dy in 0..h {
            let sy = y + dy;
            if sy >= self.height {
                break;
            }
            for dx in 0..w {
                self.put_pixel(x + dx, sy, self.bg_at(x + dx, sy, bg));
            }
        }
    }

    /// Read a framebuffer pixel back as `Rgb` (inverse of `put_pixel`), for
    /// saving the background under the mouse cursor.
    fn get_pixel(&self, x: u64, y: u64) -> Rgb {
        if x >= self.width || y >= self.height {
            return (0, 0, 0);
        }
        let offset = self.fb_offset(x, y);
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

    /// Alpha-blend `c` over the existing framebuffer pixel at `(x,y)` with
    /// coverage `a` (0 = transparent … 255 = opaque). A read-modify-write, so
    /// only worth it for the fractional-coverage *edge* pixels of a shape — the
    /// interior should use the plain [`put_pixel`] fast path. This is the
    /// primitive behind anti-aliased curves (discs, the logo arc).
    fn blend_pixel(&self, x: u64, y: u64, c: Rgb, a: u32) {
        if a == 0 || x >= self.width || y >= self.height {
            return;
        }
        if a >= 255 {
            self.put_pixel(x, y, c);
            return;
        }
        let bg = self.get_pixel(x, y);
        let mix = |b: u8, f: u8| (((b as u32) * (255 - a) + (f as u32) * a) / 255) as u8;
        self.put_pixel(x, y, (mix(bg.0, c.0), mix(bg.1, c.1), mix(bg.2, c.2)));
    }

    /// Restore the patch saved beneath the cursor (erasing the sprite). Uses the
    /// dims the patch was *saved* at (`cur_sw`×`cur_sh`), which may differ from
    /// the current shape's sprite after a theme/shape change.
    fn cursor_restore(&self) {
        if !self.cur_vis {
            return;
        }
        let (w, h) = (self.cur_sw, self.cur_sh);
        for dy in 0..h {
            for dx in 0..w {
                let i = (dy * w + dx) as usize;
                if i < self.cur_saved.len() {
                    self.put_pixel(self.cur_x + dx, self.cur_y + dy, self.cur_saved[i]);
                }
            }
        }
    }

    /// Save the framebuffer under the cursor and draw the active shape sprite
    /// (theme-driven colours + optional custom, variable-size bitmap).
    fn cursor_draw(&mut self) {
        let (w, h, data, fill, outline) = cursor_active();
        let n = (w * h) as usize;
        self.cur_saved.clear();
        self.cur_saved.reserve(n);
        for dy in 0..h {
            for dx in 0..w {
                self.cur_saved.push(self.get_pixel(self.cur_x + dx, self.cur_y + dy));
            }
        }
        self.cur_sw = w;
        self.cur_sh = h;
        for dy in 0..h {
            for dx in 0..w {
                match data[(dy * w + dx) as usize] {
                    1 => self.put_pixel(self.cur_x + dx, self.cur_y + dy, fill),
                    2 => self.put_pixel(self.cur_x + dx, self.cur_y + dy, outline),
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

    /// The action-pane close-button rectangle `(x, y, w, h)` — FA `xmark` at the
    /// top-right of the action pane title. Only meaningful when the pane is open.
    /// Geometry is shared by the renderer and the click hit-test so they cannot
    /// disagree. Width matches the square FA cell (body line height) so the mark
    /// isn't squeezed into a mono column.
    fn close_btn_for(&self, pane_i: usize) -> (u64, u64, u64, u64) {
        let w = self.ch().max(self.cw() * 2);
        let Some(a) = self.actions.get(pane_i) else {
            return (0, 0, 0, 0);
        };
        let x = (a.pane.x + a.pane.w).saturating_sub(BORDER + PAD + w);
        let y = a.pane.y + BORDER + 4;
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

    /// Alpha-blend one printable glyph into its cell at `(px,py)`. Renders via
    /// the **TTF UI face** (fontdue, crisp at the display resolution — see
    /// `font_ttf::blit_ui_cell`, face chosen from `ui.json`), falling back to the
    /// scaled bitmap atlas ([`GLYPHS`]) if no TTF face is available. Non-printable
    /// bytes render as a blank cell.
    /// Effective background colour at pixel `(x,y)`: with a translucent
    /// wallpaper, `base` blended over the wallpaper at [`Self::opacity`]; else
    /// `base` unchanged (the fast common case — no wallpaper or opaque).
    #[inline]
    fn bg_at(&self, x: u64, y: u64, base: Rgb) -> Rgb {
        match &self.wallpaper {
            Some(wp) if self.opacity < 255 && x < self.width && y < self.height => {
                let wpix = wp[(y * self.width + x) as usize];
                let op = self.opacity as u32;
                let inv = 255 - op;
                let bl = |w: u32, c: u8| (((w * inv + c as u32 * op) / 255) & 0xff) as u8;
                (
                    bl((wpix >> 16) & 0xff, base.0),
                    bl((wpix >> 8) & 0xff, base.1),
                    bl(wpix & 0xff, base.2),
                )
            }
            _ => base,
        }
    }

    /// Cell size for one glyph. Font Awesome icons get a **square** cell of the
    /// body line height so they read at text size; mono cells are tall+narrow
    /// and squashing FA into `cw×ch` made agent-list / close marks tiny.
    fn glyph_cell(&self, ch: char) -> (u64, u64) {
        if crate::icons::is_icon(ch) {
            let side = self.ch();
            (side, side)
        } else {
            (self.cw(), self.ch())
        }
    }

    fn blit_glyph(&self, px: u64, py: u64, ch: char, fg: Rgb, bg: Rgb) {
        let (cell_w, cell_h) = self.glyph_cell(ch);
        // Background fill first (both paths blend ink over it). With a
        // translucent wallpaper the cell bg is the wallpaper tinted by `bg` at
        // `opacity`, per pixel — so text sits over the see-through desktop too.
        let tinted = self.wallpaper.is_some() && self.opacity < 255;
        for gy in 0..cell_h {
            for gx in 0..cell_w {
                let cbg = self.bg_at(px + gx, py + gy, bg);
                self.put_pixel(px + gx, py + gy, cbg);
            }
        }
        // Empty / space / zero-width formatters: fill bg only (VS16, ZWJ, etc.
        // must not paint tofu boxes between emoji).
        if ch == '\0'
            || ch == ' '
            || ch == '\u{FE0F}'
            || ch == '\u{FE0E}'
            || ch == '\u{200D}'
            || ch == '\u{200C}'
        {
            return;
        }
        let mix = |b: u8, f: u8, a: u32| (((b as u32) * (255 - a) + (f as u32) * a) / 255) as u8;
        let ink = |s: &Self, x: u64, y: u64, a: u32| {
            let b = if tinted { s.bg_at(x, y, bg) } else { bg };
            (mix(b.0, fg.0, a), mix(b.1, fg.1, a), mix(b.2, fg.2, a))
        };
        // TTF path: rasterize the char (fontdue UI face + Noto fallback chain —
        // renders arbitrary Unicode; box-drawing/bullets included).
        let ttf_ok = crate::font_ttf::blit_ui_cell(ch, cell_w as usize, cell_h as usize, |gx, gy, a| {
            let (x, y) = (px + gx as u64, py + gy as u64);
            self.put_pixel(x, y, ink(self, x, y, a as u32));
        });
        if ttf_ok {
            return;
        }
        // Bitmap fallback: the 10×22 ASCII atlas. Non-ASCII with no TTF face
        // stays blank (bg already filled).
        let s = self.scale;
        let cp = ch as u32;
        if !(FIRST as u32..=LAST as u32).contains(&cp) {
            return;
        }
        let idx = (cp as u8 - FIRST) as usize;
        let g = &GLYPHS[idx];
        for gy in 0..CH {
            for gx in 0..CW {
                let a = g[gy * CW + gx] as u32;
                if a == 0 {
                    continue; // background already filled
                }
                let bx = px + gx as u64 * s;
                let by = py + gy as u64 * s;
                for sy in 0..s {
                    for sx in 0..s {
                        let (x, y) = (bx + sx, by + sy);
                        self.put_pixel(x, y, ink(self, x, y, a));
                    }
                }
            }
        }
    }

    /// Render `s` at pixel `(px,py)`. Body text advances one mono cell; Font
    /// Awesome icons advance a square of the body line height. Returns the x
    /// past the last glyph. Clips at `self.width`.
    fn draw_str(&self, px: u64, py: u64, s: &str, fg: Rgb, bg: Rgb) -> u64 {
        let cw = self.cw();
        let ch = self.ch();
        // Upper-bound the damage box (icons are wider than one mono cell).
        let mut approx_w = 0u64;
        for c in s.chars() {
            approx_w += if crate::icons::is_icon(c) { ch } else { cw };
        }
        crate::kms::damage(
            (px + self.origin_x) as u32,
            (py + self.origin_y) as u32,
            approx_w as u32,
            ch as u32,
        );
        let mut x = px;
        for c in s.chars() {
            let advance = if crate::icons::is_icon(c) { ch } else { cw };
            if x + advance > self.width {
                break;
            }
            self.blit_glyph(x, py, c, fg, bg);
            x += advance;
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
                // Fold anchors shift the same way; a fold whose "▸ more" line is
                // evicted is dropped.
                p.folds.retain_mut(|(gi, _)| {
                    if *gi == 0 {
                        false
                    } else {
                        *gi -= 1;
                        true
                    }
                });
                // User-band line indices track absolute gi the same way.
                p.user_band.retain_mut(|gi| {
                    if *gi == 0 {
                        false
                    } else {
                        *gi -= 1;
                        true
                    }
                });
            }
            p.grid.copy_within(cols.., 0);
            let start = p.grid.len() - cols;
            let fg = p.fg;
            for c in &mut p.grid[start..] {
                *c = ('\0', fg);
            }
            if p.view > 0 {
                // Keep the scrolled view anchored on the same content.
                p.view = (p.view + 1).min(p.hist.len());
                return; // pixels are frozen on the scrolled view
            }
        }
        // A translucent wallpaper is a fixed backdrop — a pixel-memmove scroll
        // would drag it up with the text. Repaint the interior from the grid
        // over a fresh wallpaper background instead.
        if self.wallpaper.is_some() && self.opacity < 255 {
            self.paint_surface(p.ix, p.iy, p.cols * p.cw, p.rows * p.ch, p.bg);
            self.render_view(p);
            return;
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
                let dst = self.fb_offset(x0, top + row) as usize;
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
                let (b, fg) = line.and_then(|l| l.get(c).copied()).unwrap_or(('\0', p.default_fg));
                let x = p.ix + c as u64 * p.cw;
                let y = p.iy + r as u64 * p.ch;
                let selected = sel.is_some_and(|s| crate::textsel::contains(s, gi, c));
                let bg = if selected {
                    self.theme.editor_sel
                } else if p.band_bg(gi) {
                    self.theme.composer_bg
                } else {
                    p.bg
                };
                // Always fill the cell first so deselected / empty cells leave
                // no residue (selection highlight, partial glyphs).
                self.fill_cell_bg(x, y, p.cw, p.ch, bg);
                if b != '\0' && b != ' ' {
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
            p.hist[gi].get(c).copied().unwrap_or(('\0', p.default_fg))
        } else {
            let gr = gi - p.hist.len();
            if gr >= p.rows as usize {
                return;
            }
            p.grid.get(gr * cols + c).copied().unwrap_or(('\0', p.default_fg))
        };
        let x = p.ix + c as u64 * p.cw;
        let y = p.iy + r as u64 * p.ch;
        let bg = if selected {
            self.theme.editor_sel
        } else if p.band_bg(gi) {
            self.theme.composer_bg
        } else {
            p.bg
        };
        self.fill_cell_bg(x, y, p.cw, p.ch, bg);
        if b != '\0' && b != ' ' {
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
        let (b, fg) = p.grid.get(idx).copied().unwrap_or(('\0', p.default_fg));
        let x = p.cell_x();
        let y = p.cell_y();
        self.fill_cell_bg(x, y, p.cw, p.ch, p.bg);
        if b != '\0' && b != ' ' {
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
        // Chat pane with a bordered composer: caret lives only in the input
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
                                    *cell = ('\0', p.fg);
                                }
                            }
                            if live {
                                s.fill_cell_bg(p.cell_x(), p.cell_y(), (p.cols - p.col) * p.cw, p.ch, p.bg);
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
                    p.set_cell(' ');
                    if live {
                        s.blit_glyph(p.cell_x(), p.cell_y(), ' ', p.fg, p.bg);
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
                p.set_cell('\0');
                if live {
                    s.blit_glyph(p.cell_x(), p.cell_y(), ' ', p.fg, p.bg);
                }
            }
            // Other C0 control bytes: ignored (never part of a UTF-8 sequence).
            b if b < 0x20 => {}
            // Printable: ASCII (0x20–0x7e) or a UTF-8 lead/continuation byte
            // (≥0x80). Decode incrementally — a multi-byte glyph spans calls.
            _ => {
                if let Some(ch) = p.feed_utf8(byte) {
                    p.set_cell(ch);
                    if live {
                        s.blit_glyph(p.cell_x(), p.cell_y(), ch, p.fg, p.bg);
                    }
                    p.col += 1;
                    if p.col >= p.cols {
                        Screen::newline(p, s);
                    }
                }
            }
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
        // Icons must hit the FA face (first fallback); if registration was
        // skipped or failed earlier, status chips paint as thin tofu bars.
        let _ = crate::font_ttf::register_bundled_fallback(crate::font_ttf::FA_FALLBACK_NAME);
        if self.layout.status_pos.vertical() {
            self.draw_status_vertical();
        } else {
            self.draw_status_horizontal();
        }
    }

    /// Draw status text with icon glyphs (Font Awesome PUA). Body text and the
    /// activity middle-dot stay at the normal cell size; icons use a square cell
    /// equal to the body line height (FA is fit-to-cell so nothing clips).
    fn draw_status_str(&self, mut x: u64, y: u64, s: &str, fg: Rgb, bg: Rgb, max_x: u64) -> u64 {
        let cw = self.cw();
        let ch = self.ch();
        // Square icon cell matching the text line — wider cells clipped the sides
        // of wide FA glyphs (keyboard); taller-than-bar cells clipped top/bottom.
        let icon_cw = ch.max(cw);
        let icon_ch = ch;
        // Vertically centre the body line against the icon row when the bar gave
        // us extra headroom (`y` is already the icon/text band top).
        let body_y = y;
        for ch_c in s.chars() {
            if is_status_icon(ch_c) {
                if x + icon_cw > max_x {
                    break;
                }
                for gy in 0..icon_ch {
                    for gx in 0..icon_cw {
                        self.put_pixel(x + gx, y + gy, bg);
                    }
                }
                let mut painted = false;
                let ok = crate::font_ttf::blit_ui_cell(
                    ch_c,
                    icon_cw as usize,
                    icon_ch as usize,
                    |gx, gy, a| {
                        if a == 0 {
                            return;
                        }
                        painted = true;
                        let px = x + gx as u64;
                        let py = y + gy as u64;
                        if px >= max_x {
                            return;
                        }
                        let mix = |b: u8, f: u8, aa: u32| {
                            (((b as u32) * (255 - aa) + (f as u32) * aa) / 255) as u8
                        };
                        let c = (
                            mix(bg.0, fg.0, a as u32),
                            mix(bg.1, fg.1, a as u32),
                            mix(bg.2, fg.2, a as u32),
                        );
                        self.put_pixel(px, py, c);
                    },
                );
                if !ok || !painted {
                    self.blit_glyph(
                        x + icon_cw.saturating_sub(cw) / 2,
                        body_y,
                        ch_c,
                        fg,
                        bg,
                    );
                }
                x += icon_cw;
            } else {
                if x + cw > max_x {
                    break;
                }
                self.blit_glyph(x, body_y, ch_c, fg, bg);
                x += cw;
            }
        }
        x
    }

    /// The status bar as a **column** (left/right edge).
    ///
    /// Reading order is **top → bottom** for *both* templates: brand, then
    /// `status_left` fields (one per row), then `status_right` fields continuing
    /// down the column. Previously `status_right` stacked upward from the bottom,
    /// which felt like horizontal "ends" rather than a vertical strip.
    fn draw_status_vertical(&self) {
        let (bx, by, bw, bh) = self.status_rect;
        STATUS_BAR_RECT.with(|r| *r = (bx, by, bw, bh));
        clear_status_chip_rects();
        self.paint_surface(bx, by, bw, bh, self.theme.status_bg);
        let (cw, ch) = (self.cw(), self.ch());
        // Slightly taller rows so icons have room; body text is vertically centred
        // by `draw_status_str`.
        let row = ch + STATUS_ICON_EXTRA + 4;
        let lr = (((row / 2).saturating_sub(2)) * 6 / 7).max(6);
        // Logo colours from ui.json theme (`logo` / `logo_node`, else accent/chat_fg).
        let logo_cx = bx + bw / 2;
        let logo_cy = by + STATUS_PAD / 2 + row / 2;
        self.draw_logo(
            logo_cx,
            logo_cy,
            lr,
            self.theme.logo,
            self.theme.logo_node,
        );
        // About opens only on the logo mark — not the wordmark or empty bar space.
        let logo_ext = lr + (lr / 3).max(3) + 4;
        let hx = logo_cx.saturating_sub(logo_ext).max(bx);
        let hy = logo_cy.saturating_sub(logo_ext).max(by);
        let hw = (logo_ext * 2).min(bw.saturating_sub(hx.saturating_sub(bx)));
        let hh = (logo_ext * 2).min(bh.saturating_sub(hy.saturating_sub(by)));
        set_status_chip_rect(StatusChip::Brand, (hx, hy, hw, hh));
        let tx = bx + STATUS_PAD / 2;
        let max_x = bx + bw.saturating_sub(STATUS_PAD / 2);
        let cols = (bw.saturating_sub(STATUS_PAD) / cw).max(4) as usize;
        let first = by + STATUS_PAD / 2 + row;
        let last = by + bh.saturating_sub(STATUS_PAD / 2);
        // One token per row, top → bottom (left template, then right chips).
        let mut top = first;
        let left_lines = crate::panes_layout::status_lines_vertical(&self.status_left, cols);
        for (i, line) in left_lines.iter().enumerate() {
            if top + row > last {
                break;
            }
            // Brand wordmark uses logo colour; all status text/icons use status_fg.
            let fg = if i == 0 {
                self.theme.logo
            } else {
                self.theme.status_fg
            };
            self.draw_status_str(tx, top, line, fg, self.theme.status_bg, max_x);
            top += row;
        }
        for (chip, text) in status_right_chips() {
            if top + row > last {
                break;
            }
            self.draw_status_str(tx, top, &text, self.theme.status_fg, self.theme.status_bg, max_x);
            set_status_chip_rect(chip, (bx, top, bw, row));
            top += row;
        }
    }

    /// The status bar as a **row** (top/bottom edge) — brand left, system chips
    /// right. Each right chip is individually clickable (macOS menu-bar style).
    fn draw_status_horizontal(&self) {
        let (_, sy_top, _, bar_h) = self.status_rect;
        STATUS_BAR_RECT.with(|r| *r = (0, sy_top, self.width, bar_h));
        clear_status_chip_rects();
        self.paint_surface(0, sy_top, self.width, bar_h, self.theme.status_bg);
        // Vertically centre the text/icon line in the bar (icons = body cell height).
        let line_h = self.ch();
        let ty = sy_top + bar_h.saturating_sub(line_h) / 2;
        let cw = self.cw();
        let lr = (((bar_h / 2).saturating_sub(2)) * 6 / 7).max(6);
        let lhalf = ((lr / 3).max(3)) / 2;
        let lcx = OUTER + lr + lhalf;
        // Logo from ui.json theme.logo / logo_node (see theme_from_pairs).
        self.draw_logo(
            lcx,
            sy_top + bar_h / 2,
            lr,
            self.theme.logo,
            self.theme.logo_node,
        );
        // About opens only on the logo mark — not the wordmark or empty bar space.
        let logo_x0 = lcx.saturating_sub(lr + lhalf);
        let logo_w = (lr + lhalf) * 2 + 4;
        set_status_chip_rect(
            StatusChip::Brand,
            (logo_x0.saturating_sub(2), sy_top, logo_w + 4, bar_h),
        );
        let text_x = lcx + lr + lhalf + cw / 2;
        let gap = 2 * cw;
        let usable = self.width.saturating_sub(text_x + OUTER + gap);
        let left_budget = (usable / 2 / cw).max(4) as usize;
        let left = crate::textsel::ellipsize(&self.status_left, left_budget);
        let max_left = text_x + left_budget as u64 * cw;
        // Brand wordmark ("ChittiOS v…") uses logo colour; bar field colour is status_fg.
        // Wordmark is not a hit target (About is logo-only).
        self.draw_status_str(
            text_x,
            ty,
            &left,
            self.theme.logo,
            self.theme.status_bg,
            max_left,
        );

        // Right chips painted individually so each has a hit rect.
        // Icons and labels share status_fg so they match the theme text colour.
        let chips = status_right_chips();
        let gap1 = cw; // within a tight group
        let gap2 = 2 * cw; // between groups
        let mut total = 0u64;
        for (i, (_, text)) in chips.iter().enumerate() {
            if i > 0 {
                // kbd–mouse single space; otherwise group gap
                total += if i == 1 { gap1 } else { gap2 };
            }
            total += status_str_advance(text, cw, line_h);
        }
        let left_end = text_x + status_str_advance(&left, cw, line_h) + gap;
        let mut x = self
            .width
            .saturating_sub(total + OUTER)
            .max(left_end)
            .min(self.width.saturating_sub(OUTER));
        let max_x = self.width.saturating_sub(OUTER / 2);
        for (i, (chip, text)) in chips.iter().enumerate() {
            if i > 0 {
                x += if i == 1 { gap1 } else { gap2 };
            }
            let w = status_str_advance(text, cw, line_h);
            if x + w > max_x {
                break;
            }
            let x1 = self.draw_status_str(
                x,
                ty,
                text,
                self.theme.status_fg,
                self.theme.status_bg,
                max_x,
            );
            // Hit pad a few px for easy clicking.
            let hx = x.saturating_sub(2);
            let hw = x1.saturating_sub(hx) + 2;
            set_status_chip_rect(*chip, (hx, sy_top, hw, bar_h));
            x = x1;
        }
    }

    /// Draw `s` within `[x, x+max_w)`, ellipsizing when it would overflow.
    /// Returns the x just past the last painted glyph.
    fn draw_str_fit(&self, x: u64, y: u64, s: &str, fg: Rgb, bg: Rgb, max_w: u64) -> u64 {
        let cols = (max_w / self.cw()) as usize;
        let t = crate::textsel::ellipsize(s, cols);
        self.draw_str(x, y, &t, fg, bg)
    }

    /// Geometry of the bordered input composer inside the chat pane:
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

    /// Draw a soft-rounded rectangle outline (bordered input chrome).
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

    /// Paint the bordered composer box + hint bar at the bottom of the chat pane.
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
        self.paint_surface(bx + 1, by + 1, bw.saturating_sub(2), bh.saturating_sub(2), self.theme.composer_bg);
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
            self.paint_surface(x, ty, rest, self.chat.ch, self.theme.composer_bg);
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
        self.fill_cell_bg(hx, hy, hw, self.chat.ch, self.chat.bg);
        let total_cols = (hw / cw).max(1) as usize;
        let gap = 2usize;
        let right_raw = self.composer_hint_r.chars().count();
        let right_cols = right_raw.min(total_cols / 3).min(total_cols.saturating_sub(gap + 4));
        let left_cols = total_cols.saturating_sub(right_cols + if right_cols > 0 { gap } else { 0 });
        let left = crate::textsel::ellipsize(&self.composer_hint_l, left_cols);
        let right = crate::textsel::ellipsize(&self.composer_hint_r, right_cols);
        self.draw_hint_left(hx, hy, &left);
        if !right.is_empty() {
            let rlen = right.chars().count() as u64 * cw;
            self.draw_str(hx + hw.saturating_sub(rlen), hy, &right, self.theme.composer_hint, self.chat.bg);
        }
        // Suggestion menu sits above the composer (slash commands / @files).
        self.draw_suggest_popup();
    }

    /// Draw the left hint, colouring its first `composer_hint_l_lead.len()`
    /// characters from that list (the shell's progress bar) and the rest in
    /// `theme.composer_hint`.
    ///
    /// The per-cell colours are dropped when `left` came back shorter than the
    /// lead run: a narrow pane ellipsizes the hint, and painting the gradient
    /// onto whatever characters survived would colour the label instead of the
    /// bar.
    fn draw_hint_left(&self, hx: u64, hy: u64, left: &str) {
        let lead = self.composer_hint_l_lead.len();
        if lead == 0 || left.chars().count() < lead {
            self.draw_str(hx, hy, left, self.theme.composer_hint, self.chat.bg);
            return;
        }
        let mut x = hx;
        let mut split = left.len();
        let mut buf = [0u8; 4];
        for (i, (bi, ch)) in left.char_indices().enumerate() {
            if i == lead {
                split = bi;
                break;
            }
            let c = self.composer_hint_l_lead[i];
            x = self.draw_str(x, hy, ch.encode_utf8(&mut buf), c, self.chat.bg);
        }
        self.draw_str(x, hy, &left[split..], self.theme.composer_hint, self.chat.bg);
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
            self.paint_surface(left, top, rw, rh, self.chat.bg);
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
                let (b, fg) = line.and_then(|l| l.get(c).copied()).unwrap_or(('\0', p.default_fg));
                let x = p.ix + c as u64 * p.cw;
                let y = p.iy + r as u64 * p.ch;
                let selected = sel.is_some_and(|s| crate::textsel::contains(s, gi, c));
                let bg = if selected {
                    self.theme.editor_sel
                } else if p.band_bg(gi) {
                    self.theme.composer_bg
                } else {
                    p.bg
                };
                self.fill_cell_bg(x, y, p.cw, p.ch, bg);
                if b != '\0' && b != ' ' {
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
        self.paint_surface(bx + 1, by + 1, bw.saturating_sub(2), bh.saturating_sub(2), self.theme.composer_bg);
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
            self.paint_surface(x, ty, rest, self.chat.ch, self.theme.composer_bg);
        }
        if self.composer_active && !self.action_focused() {
            let cx = tx + (prompt.len() as u64 + caret_col as u64) * self.chat.cw;
            let color = if self.caret_on { self.theme.accent } else { self.theme.composer_bg };
            self.fill_rect(cx, ty, 2 * self.scale.max(1), self.chat.ch, color);
        }
        let hx = bx;
        let hw = bw;
        let cw = self.chat.cw;
        self.fill_cell_bg(hx, hy, hw, self.chat.ch, self.chat.bg);
        let total_cols = (hw / cw).max(1) as usize;
        let gap = 2usize;
        let right_raw = self.composer_hint_r.chars().count();
        let right_cols = right_raw.min(total_cols / 3).min(total_cols.saturating_sub(gap + 4));
        let left_cols = total_cols.saturating_sub(right_cols + if right_cols > 0 { gap } else { 0 });
        let left = crate::textsel::ellipsize(&self.composer_hint_l, left_cols);
        let right = crate::textsel::ellipsize(&self.composer_hint_r, right_cols);
        self.draw_hint_left(hx, hy, &left);
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
    /// bordered composer, the caret only blinks inside the box while the
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
    /// **Anti-aliased** filled disc: edge pixels get fractional coverage from
    /// [`AA_SS`]×[`AA_SS`] sub-sampling (integer-only, no `sqrt`), blended over
    /// the background so the rim is smooth rather than stair-stepped. Interior
    /// pixels take the opaque [`put_pixel`] fast path.
    fn fill_disc(&self, cx: i64, cy: i64, r: i64, c: Rgb) {
        if r <= 0 {
            return;
        }
        // Radius in the 2·SS sub-pixel grid `aa_coverage` samples on.
        let rr = 2 * AA_SS * r;
        let r2 = rr * rr;
        let span = r + 1;
        for dy in -span..=span {
            for dx in -span..=span {
                let a = aa_coverage(dx, dy, |fx, fy| fx * fx + fy * fy <= r2);
                // Negative coords wrap to a huge u64 and are dropped by the
                // bounds checks in put_pixel / blend_pixel.
                self.blend_pixel((cx + dx) as u64, (cy + dy) as u64, c, a);
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
        // Ring radii in the 2·SS sub-pixel grid (squared) for anti-aliasing.
        let inner = (2 * AA_SS * (r - half)).pow(2);
        let outer = (2 * AA_SS * (r + half)).pow(2);
        let span = r + half + 1;
        for dy in -span..=span {
            for dx in -span..=span {
                // Sub-sampled coverage of the open ring: inside the stroke band
                // and outside the ~91° opening. The gap test compares a pixel's
                // direction against the gap centre (984,-180) ≈ (cos-10.4°,
                // sin-10.4°): within ~45.4° (cos ≈ 0.701) is inside the gap.
                let a = aa_coverage(dx, dy, |fx, fy| {
                    let d2 = fx * fx + fy * fy;
                    if d2 < inner || d2 > outer {
                        return false;
                    }
                    let n = fx * 984 - fy * 180;
                    !(n > 0 && n * n > 701 * 701 * d2)
                });
                self.blend_pixel((cx + dx) as u64, (cy + dy) as u64, arc_c, a);
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
        self.paint_wallpaper(0, 0, self.width, self.height, self.theme.screen_bg);
        let r = (self.height / 7).max(24);
        let cy = self.height * 2 / 5;
        // Ring/node from theme.logo / logo_node (ui.json), defaulting to brand.
        self.draw_logo(self.width / 2, cy, r, self.theme.logo, self.theme.logo_node);
        let name = "ChittiOS";
        let nx = self.width / 2 - (name.len() as u64 * self.cw()) / 2;
        self.draw_str(nx, cy + r + r / 2, name, self.theme.accent, self.theme.screen_bg);
        let tag = "an agentic operating system";
        let tx = self.width / 2 - (tag.len() as u64 * self.cw()) / 2;
        self.draw_str(tx, cy + r + r / 2 + self.ch() + 6, tag, self.theme.title_dim, self.theme.screen_bg);
    }

    /// Whether an action pane holds keyboard focus. Chat keeps focus by
    /// default so you can keep typing; Ctrl+Tab / a click / `/pane focus`
    /// moves it onto the band. The editor is the same rule now — opening it
    /// sets `focus_action` so keys land there, and Ctrl+Tab returns to the
    /// shell without closing the tab.
    fn action_focused(&self) -> bool {
        match self.right() {
            RightMode::Closed => false,
            RightMode::Editor
            | RightMode::Ktrace
            | RightMode::Top
            | RightMode::Todos
            | RightMode::Audio
            | RightMode::Surface(_) => self.focus_action,
        }
    }

    /// Full repaint: background, chat pane (content re-rendered from its grid),
    /// the action (right) pane if open, caret, status bar.
    ///
    /// Parked panes (`w == 0`, fullscreen) are skipped entirely — their content
    /// is preserved in memory via [`Pane::take_content`] and restored on unpark.
    fn redraw(&self) {
        crate::kms::damage(0, 0, self.fb_w as u32, self.fb_h as u32);
        // Dead space around a smaller-than-native desktop. A no-op at native, and
        // painted here (not per-frame) because the letterbox only changes when the
        // geometry does — which is exactly when a redraw happens.
        self.paint_letterbox();
        // Paint only the background *gutters* (margins + the gap between panes),
        // never a full-screen clear — the panes are painted over their own areas
        // below, so their content is never flashed to background. This is what
        // makes opening/closing the action pane not flicker the whole screen.
        self.paint_gutters();
        // Drop shadows sit in the gutters (right/bottom bands of each pane).
        if self.chat.w > 0 {
            self.drop_shadow(self.chat.x, self.chat.y, self.chat.w, self.chat.h);
            self.paint_surface(self.chat.x, self.chat.y, self.chat.w, self.chat.h, self.chat.bg);
            self.draw_frame(&self.chat, !self.action_focused());
            self.render_view(&self.chat);
            self.draw_composer(); // includes suggest popup when open
        }
        for (i, a) in self.actions.iter().enumerate() {
            if !self.column_visible(i) {
                continue;
            }
            let focused = self.focus_action && i == self.focused_action;
            self.drop_shadow(a.pane.x, a.pane.y, a.pane.w, a.pane.h);
            self.paint_surface(a.pane.x, a.pane.y, a.pane.w, a.pane.h, a.pane.bg);
            self.draw_frame_titled(&a.pane, focused, "");
            self.draw_tab_bar_for(i);
            self.draw_close_btn_for(i);
            if a.right() == RightMode::Ktrace {
                self.render_view(&a.pane);
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
        // Gutters are confined to the content rect: the status bar paints its own
        // area, and painting over it here would blank the bar on every redraw
        // wherever it does not happen to be the bottom edge.
        let (cx0, cy0) = (self.content_x, self.content_y);
        let (cx1, cy1) = (cx0 + self.content_w, cy0 + self.content_h);
        // Strip above the pane band, and the strip below it down to the content edge.
        self.paint_wallpaper(cx0, cy0, self.content_w, by.saturating_sub(cy0), bg);
        let below = by + bh;
        self.paint_wallpaper(cx0, below, self.content_w, cy1.saturating_sub(below), bg);
        // Horizontal gutters: left of chat, gaps between all boxes, right margin.
        let mut boxes: alloc::vec::Vec<(u64, u64)> = alloc::vec![];
        if self.chat.w > 0 {
            boxes.push((self.chat.x, self.chat.x + self.chat.w));
        }
        for a in &self.actions {
            if a.pane.w > 0 {
                boxes.push((a.pane.x, a.pane.x + a.pane.w));
            }
        }
        boxes.sort_by_key(|b| b.0);
        let mut x = cx0;
        for &(l, r) in &boxes {
            if l > x {
                self.paint_wallpaper(x, by, l - x, bh, bg);
            }
            x = r.max(x);
        }
        if x < cx1 {
            self.paint_wallpaper(x, by, cx1 - x, bh, bg);
        }
    }

    /// Repaint **only the action pane** for a tab switch (geometry unchanged):
    /// clear its interior once for the new tab, redraw its frame + tab bar, and
    /// re-render ktrace from its grid. The chat pane and the whole background are
    /// left untouched — so switching tabs never flickers the rest of the screen.
    /// The active tab's dynamic interior (top/audio/image/editor) is repainted by
    /// the shell right after (`repaint_active_tab`).
    fn repaint_action(&mut self) {
        self.cursor_restore();
        self.draw_frame(&self.chat, !self.action_focused());
        for i in 0..self.actions.len() {
            if !self.column_visible(i) {
                continue;
            }
            let a = &self.actions[i];
            let focused = self.focus_action && i == self.focused_action;
            self.paint_surface(
                a.pane.ix,
                a.pane.iy,
                a.pane.cols * self.cw(),
                a.pane.rows * self.ch(),
                a.pane.bg,
            );
            self.draw_frame_titled(&a.pane, focused, "");
            self.draw_tab_bar_for(i);
            self.draw_close_btn_for(i);
            if a.right() == RightMode::Ktrace {
                self.render_view(&a.pane);
            }
        }
        self.cursor_overlay();
    }

    fn draw_close_btn_for(&self, pane_i: usize) {
        let Some(a) = self.actions.get(pane_i) else { return };
        // An empty drop-target column has nothing to close.
        if a.pane.w == 0 || a.is_empty() {
            return;
        }
        let (x, y, w, _) = self.close_btn_for(pane_i);
        // Font Awesome xmark in a square line-height cell (see `glyph_cell`),
        // centred in the hit box; ink from the live theme accent.
        let mark = crate::icons::close_mark();
        let (iw, _) = self.glyph_cell(mark);
        let ix = x + w.saturating_sub(iw) / 2;
        self.blit_glyph(ix, y, mark, self.theme.accent, a.pane.bg);
    }

    /// Per-tab header layout for action column `pane_i`.
    fn tab_layout_for(&self, pane_i: usize) -> Vec<(RightMode, u64, u64)> {
        let Some(a) = self.actions.get(pane_i) else {
            return Vec::new();
        };
        let cw = self.cw();
        let mut x = a.pane.x + BORDER + PAD;
        let mut out = Vec::with_capacity(a.tabs.len());
        for &m in &a.tabs {
            let lab = tab_label(m);
            let w = (lab.chars().count() as u64 + 1) * cw;
            out.push((m, x, w));
            x += w + cw;
        }
        out
    }

    fn draw_tab_bar_for(&self, pane_i: usize) {
        let Some(a) = self.actions.get(pane_i) else { return };
        if a.pane.w == 0 {
            return;
        }
        // Share the close button's geometry so the bar always stops exactly
        // where the `[x]` starts. `close_btn_for` saturates, which matters now
        // that a grid pane can be far narrower than the old single action pane.
        let (close_x, ty, ..) = self.close_btn_for(pane_i);
        let x0 = a.pane.x + BORDER + PAD;
        self.fill_rect(x0, ty, close_x.saturating_sub(x0), self.ch(), a.pane.bg);
        for (i, (m, x, w)) in self.tab_layout_for(pane_i).into_iter().enumerate() {
            if x + w >= close_x {
                break;
            }
            let is_active = i == a.active;
            let fg = if is_active {
                self.theme.title_active
            } else {
                self.theme.title_dim
            };
            let mut lx = x;
            if is_active {
                lx = self.draw_str(lx, ty, ">", self.theme.accent, a.pane.bg);
            }
            let lab = tab_label(m);
            self.draw_str(lx, ty, &lab, fg, a.pane.bg);
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
        let Some(d) = sc.mode_dims(RightMode::Top) else {
            return;
        };
        sc.cursor_restore();
        sc.cur_vis = false;
        let ch = sc.ch();
        let cw = sc.cw();
        let (px, iy, iw) = (d.ix, d.iy, d.iw);
        let bottom = iy + d.ih;
        // NO full-interior clear: overwrite in place (padded strings + self-
        // filling bars) so the 1 Hz refresh does not flicker.
        let bg = d.bg;
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
        // **How many rows fit, decided up front so truncation can be *reported*.**
        // This used to just `break` when it ran out of room, which reads as "these are
        // all the tasks" — and in a small pane that meant three rows standing in for a
        // dozen, so a running agent looked absent. Same rule as everywhere else here: no
        // silent caps.
        let room = if bottom > y + footer_h { ((bottom - y - footer_h) / ch) as usize } else { 0 };
        let truncated = v.tasks.len() > room;
        // Give up one row to the "+N more" marker when there is something to say.
        let show = if truncated { room.saturating_sub(1) } else { v.tasks.len() };
        let mut first_running_painted = false;
        for t in v.tasks.iter().take(show) {
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
        // Say what was left out. The count is the honest one — total tasks the scheduler
        // holds, not just the ones that fit — so a full list in a taller pane and a
        // clipped one here describe the same system.
        if truncated && y + ch + footer_h <= bottom {
            let hidden = v.tasks.len().saturating_sub(show);
            let more = alloc::format!("  +{hidden} more of {} tasks -- taller pane to see all", v.tasks_total);
            sc.draw_str_bg(px, y, &crate::textsel::fit_width(&more, cols), sc.theme.title_dim, bg);
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
        let Some(d) = sc.mode_dims(RightMode::Audio) else {
            return;
        };
        sc.cursor_restore();
        sc.cur_vis = false;
        let ch = sc.ch();
        let cw = sc.cw();
        let (px, py) = (d.ix, d.iy);
        let (pw, ph) = (d.iw, d.ih);
        let bg = d.bg;
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
        // leftover glyphs) never sticks around when the wave shrinks —
        // wallpaper-aware so the translucent desktop shows behind it.
        sc.paint_surface(px, py, pw, content_h.saturating_sub(1), bg);
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

/// Overlay the **browser** control/status bar (same layout family as the video
/// player HUD): title + URL, a scroll scrubber, and keyboard shortcut hints.
/// Call *after* [`present_surface_reserve`] so the strip sits on the reserved
/// bottom region. No-op unless the browser surface tab is active.
pub fn draw_browser_status(
    title: &str,
    url: &str,
    scroll_y: i32,
    content_h: i32,
    view_h: i32,
    focused_input: bool,
) {
    SCREEN.with(|slot| {
        let Some(sc) = slot else { return };
        let Some(d) = sc.mode_dims(RightMode::Surface(BROWSER_SURFACE)) else {
            return;
        };
        sc.cursor_restore();
        sc.cur_vis = false;
        let ch = sc.ch();
        let cw = sc.cw();
        let (px, py) = (d.ix, d.iy);
        let (pw, ph) = (d.iw, d.ih);
        let bg = d.bg;
        let barh = ch * 4 + ch / 2;
        let by = py + ph.saturating_sub(barh);
        sc.fill_rect(px, by, pw, barh, bg);
        sc.fill_rect(px, by, pw, 1, sc.theme.accent); // top hairline
        let cols = (pw / cw).saturating_sub(2).max(4) as usize;
        let fit = |s: &str| crate::textsel::fit_width(s, cols);

        let max_scroll = (content_h - view_h).max(0);
        let scroll_pct = if max_scroll > 0 {
            ((scroll_y as i64 * 100) / max_scroll as i64).clamp(0, 100) as u32
        } else {
            0
        };
        let mode = if focused_input { "input" } else { "nav" };
        // Drop fields as the pane narrows (video HUD pattern).
        let scroll_s = if max_scroll > 0 {
            alloc::format!("scroll {}%  {}/{}", scroll_pct, scroll_y, max_scroll)
        } else {
            String::from("top")
        };
        let candidates = [
            alloc::format!("{}  {}  {}  [{}]", title, url, scroll_s, mode),
            alloc::format!("{}  {}  [{}]", title, scroll_s, mode),
            alloc::format!("{}  {}", title, scroll_s),
            alloc::format!(
                "{}  {}",
                crate::textsel::ellipsize(title, cols.saturating_sub(12).max(4)),
                scroll_s
            ),
            crate::textsel::ellipsize(title, cols),
        ];
        let line1 = candidates
            .into_iter()
            .find(|s| s.chars().count() <= cols)
            .unwrap_or_else(|| crate::textsel::ellipsize(title, cols));
        let mut y = by + ch / 3;
        sc.draw_str_bg(px + cw, y, &fit(&line1), sc.theme.accent, bg);
        y += ch + ch / 4;

        // Scroll scrubber (full-width track, filled = position).
        let track_x = px + cw;
        let track_w = pw.saturating_sub(2 * cw);
        let filled = if max_scroll > 0 {
            (track_w * scroll_y as u64 / max_scroll as u64).min(track_w)
        } else {
            0
        };
        sc.fill_rect(track_x, y + ch / 3, track_w, ch / 4, sc.theme.title_dim);
        sc.fill_rect(track_x, y + ch / 3, filled, ch / 4, sc.theme.accent);
        y += ch + ch / 4;

        // Shortcut hints — wrap like the video player.
        let hints = if focused_input {
            [
                "type text",
                "Bksp erase",
                "Tab next",
                "Enter submit",
                "Esc unfocus",
                "wheel scroll",
            ]
        } else {
            [
                "j/k scroll",
                "space page",
                "wheel scroll",
                "b back",
                "r reload",
                "click link/form",
            ]
        };
        let sep = "   ";
        let mut linebuf = String::new();
        let hud_bottom = py + ph;
        for h in hints {
            if h.chars().count() > cols {
                continue;
            }
            let cand = if linebuf.is_empty() {
                String::from(h)
            } else {
                alloc::format!("{}{}{}", linebuf, sep, h)
            };
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
        let _ = url; // included in line1 when the pane is wide enough
        sc.cursor_overlay();
    });
}

/// Height in px the browser HUD reserves (same formula as the video player).
pub fn browser_hud_height() -> u64 {
    video_hud_height()
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
        let Some(d) = sc.mode_dims(RightMode::Surface(VIDEO_SURFACE)) else {
            return;
        };
        sc.cursor_restore();
        sc.cur_vis = false;
        let ch = sc.ch();
        let cw = sc.cw();
        let (px, py) = (d.ix, d.iy);
        let (pw, ph) = (d.iw, d.ih);
        // Reserved HUD strip (below the video frame). Fill the whole strip once
        // so time/fps string length changes never leave glyph trails; the strip
        // is small (~4 lines) so this is cheap and does not flash the picture.
        let bg = d.bg;
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
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ModalHit {
    None,
    Yes,
    No,
    Ok,
    /// Commands/agents-browser close (FA xmark; slot 0 reused when that modal is up).
    Close,
    /// Absolute row index in a list browser (`scroll + visible row`). Headers
    /// and items share the index space; callers skip non-selectable rows.
    ListRow(usize),
    /// Option index in a [`draw_choose`] multi-choice modal.
    Choose(usize),
}

/// Pixel rects of the modal's clickable controls: `[yes, no, ok]`. Set when a
/// modal is drawn, read by [`modal_hit`] for mouse routing. Zero-size = absent.
static MODAL_RECTS: Locked<[(u64, u64, u64, u64); 3]> = Locked::new([(0, 0, 0, 0); 3]);

/// Dedicated close-mark rect (FA xmark). Checked **first** in [`modal_hit`] so
/// About / status menus can put Close in slot 0 *and* use slots 1–2 for other
/// chrome without Close being mis-classified as Yes.
static MODAL_CLOSE_RECT: Locked<(u64, u64, u64, u64)> = Locked::new((0, 0, 0, 0));

fn set_modal_close_rect(r: (u64, u64, u64, u64)) {
    MODAL_CLOSE_RECT.with(|c| *c = r);
}

fn clear_modal_close_rect() {
    MODAL_CLOSE_RECT.with(|c| *c = (0, 0, 0, 0));
}

/// Geometry of the scrollable list in `/help` and `/agents` browsers, for
/// mouse row hit-testing. Cleared on dismiss.
#[derive(Clone, Copy)]
struct ListBrowserGeom {
    list_x: u64,
    list_y: u64,
    list_w: u64,
    row_h: u64,
    /// Visible row count (≤ 12).
    n_rows: usize,
    /// Absolute index of the first visible row.
    scroll: usize,
}

static LIST_BROWSER_GEOM: Locked<Option<ListBrowserGeom>> = Locked::new(None);

/// Option row rects for [`draw_choose`] (up to 9 numbered choices).
static CHOOSE_RECTS: Locked<[(u64, u64, u64, u64); 9]> =
    Locked::new([(0, 0, 0, 0); 9]);
static CHOOSE_COUNT: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(0);

/// True while a modal overlays the panes: upkeep ticks running under it (long
/// compute pumps `shell::upkeep`) must not blink the pane caret into the box.
static MODAL_ON: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Clickable status-bar chips (macOS menu-bar style). Brand opens About; the
/// rest open a dropdown popover with live details.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum StatusChip {
    Brand = 0,
    Kbd = 1,
    Mouse = 2,
    Net = 3,
    Mem = 4,
    Cpu = 5,
    Battery = 6,
    /// Software output volume (`sound::volume` / mute).
    Volume = 7,
    Clock = 8,
}

const STATUS_CHIP_N: usize = 9;

/// Hit rects for [`StatusChip`] (index = `chip as usize`). Zero-size = absent.
static STATUS_CHIP_RECTS: Locked<[(u64, u64, u64, u64); STATUS_CHIP_N]> =
    Locked::new([(0, 0, 0, 0); STATUS_CHIP_N]);

/// Full status-bar rect (for anchoring menus above/below/beside).
static STATUS_BAR_RECT: Locked<(u64, u64, u64, u64)> = Locked::new((0, 0, 0, 0));

fn in_rect(x: u64, y: u64, r: (u64, u64, u64, u64)) -> bool {
    r.2 != 0 && x >= r.0 && x < r.0 + r.2 && y >= r.1 && y < r.1 + r.3
}

fn clear_status_chip_rects() {
    STATUS_CHIP_RECTS.with(|a| *a = [(0, 0, 0, 0); STATUS_CHIP_N]);
}

fn set_status_chip_rect(chip: StatusChip, r: (u64, u64, u64, u64)) {
    STATUS_CHIP_RECTS.with(|a| a[chip as usize] = r);
}

/// True if `(x, y)` is on the status-bar **logo** (About hit target; not the wordmark).
pub fn status_brand_hit(x: u64, y: u64) -> bool {
    status_chip_hit(x, y) == Some(StatusChip::Brand)
}

/// Which status-bar chip is under `(x, y)`, if any (inactive while a modal is up).
pub fn status_chip_hit(x: u64, y: u64) -> Option<StatusChip> {
    if MODAL_ON.load(core::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    STATUS_CHIP_RECTS.with(|a| {
        for i in 0..STATUS_CHIP_N {
            if in_rect(x, y, a[i]) {
                return Some(match i {
                    0 => StatusChip::Brand,
                    1 => StatusChip::Kbd,
                    2 => StatusChip::Mouse,
                    3 => StatusChip::Net,
                    4 => StatusChip::Mem,
                    5 => StatusChip::Cpu,
                    6 => StatusChip::Battery,
                    7 => StatusChip::Volume,
                    _ => StatusChip::Clock,
                });
            }
        }
        None
    })
}

/// Pixel advance of a status string (icons use square line-height cells).
fn status_str_advance(s: &str, cw: u64, ch: u64) -> u64 {
    let icon_cw = ch.max(cw);
    let mut w = 0u64;
    for c in s.chars() {
        w += if is_status_icon(c) { icon_cw } else { cw };
    }
    w
}

/// Live right-side status chips (same content the bar paints, in order).
fn status_right_chips() -> alloc::vec::Vec<(StatusChip, alloc::string::String)> {
    let mut out = alloc::vec::Vec::new();
    let last_k = crate::console::input_activity_ms();
    let k_active = last_k != 0 && crate::arch::now_ms().saturating_sub(last_k) < 1500;
    out.push((StatusChip::Kbd, crate::icons::status_kbd(k_active)));
    let last_m = crate::mouse::activity_ms();
    let m_active = last_m != 0 && crate::arch::now_ms().saturating_sub(last_m) < 1500;
    out.push((StatusChip::Mouse, crate::icons::status_mouse(m_active)));
    out.push((StatusChip::Net, crate::icons::status_net(crate::net::is_up())));
    // mem / cpu / cores — match ui_config::resolve_var labels
    {
        let m = crate::mm::mem_stats();
        let mib = 1024 * 1024;
        let gib = 1024 * mib;
        let mem = if m.ram_total >= gib {
            alloc::format!(
                "mem {}M/{}.{}G",
                (m.heap_used + (m.ram_reserved - m.heap_total)) / mib,
                m.ram_total / gib,
                (m.ram_total % gib) * 10 / gib
            )
        } else {
            alloc::format!("mem {}/{}M", m.heap_used / mib, m.ram_total / mib)
        };
        out.push((StatusChip::Mem, mem));
    }
    out.push((
        StatusChip::Cpu,
        alloc::format!(
            "cpu {:>3}% {}c",
            crate::shell::cpu_percent(),
            crate::arch::cpu_count()
        ),
    ));
    if let Some(b) = crate::drivers::battery::cached() {
        let s = crate::drivers::battery::format(&b);
        if !s.is_empty() {
            out.push((StatusChip::Battery, s));
        }
    }
    // Volume always shown (software gain applies even with no PCM device yet).
    out.push((
        StatusChip::Volume,
        crate::icons::status_volume(crate::sound::muted(), crate::sound::volume()),
    ));
    // Compact macOS-style clock (no year / seconds / tz — dropdown has the rest).
    out.push((StatusChip::Clock, crate::clock::format_datetime_short()));
    out
}

/// Hit-test the modal controls against a click at `(x, y)`.
pub fn modal_hit(x: u64, y: u64) -> ModalHit {
    // Close mark first (dedicated rect) so it wins over the menu-body rect that
    // fully contains it (status dropdown) and over slot-0 Yes (confirm).
    if MODAL_CLOSE_RECT.with(|c| in_rect(x, y, *c)) {
        return ModalHit::Close;
    }
    let r = MODAL_RECTS.with(|m| *m);
    // Confirm: Yes/No in 0/1. List browsers also leave Close in slot 0 with 1/2
    // empty — keep that path for older drawers that only set slot 0.
    if in_rect(x, y, r[0]) {
        if r[1] == (0, 0, 0, 0) && r[2] == (0, 0, 0, 0) {
            return ModalHit::Close;
        }
        return ModalHit::Yes;
    } else if in_rect(x, y, r[1]) {
        return ModalHit::No;
    } else if in_rect(x, y, r[2]) {
        return ModalHit::Ok;
    }
    // List browser rows (/help, /agents).
    if let Some(g) = LIST_BROWSER_GEOM.with(|g| *g) {
        if g.row_h > 0
            && g.list_w > 0
            && x >= g.list_x
            && x < g.list_x + g.list_w
            && y >= g.list_y
            && y < g.list_y + g.row_h * g.n_rows as u64
        {
            let row = ((y - g.list_y) / g.row_h) as usize;
            if row < g.n_rows {
                return ModalHit::ListRow(g.scroll + row);
            }
        }
    }
    // Multi-choice options.
    let n = CHOOSE_COUNT.load(core::sync::atomic::Ordering::Relaxed).min(9);
    if n > 0 {
        let rects = CHOOSE_RECTS.with(|c| *c);
        for i in 0..n {
            if in_rect(x, y, rects[i]) {
                return ModalHit::Choose(i);
            }
        }
    }
    ModalHit::None
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

    /// Content rows a centred modal can hold on this screen (title + separator
    /// and the frame already deducted, one cell of margin kept top and bottom).
    /// The math is pure and lives in `panes_layout` so it is unit-tested — this
    /// module is `#[cfg(not(test))]`, so a test in here would never run.
    fn modal_rows_budget(&self) -> usize {
        crate::panes_layout::modal_max_rows(self.height, self.ch(), 2 * (BORDER + PAD)) as usize
    }

    fn modal_box(&self, title: &str, rows: u64) -> (u64, u64, u64) {
        let cw = self.cw();
        let ch = self.ch();
        let cols = self.modal_cols();
        let bw = cols * cw + 2 * (BORDER + PAD);
        let bh = (rows + 2) * ch + 2 * (BORDER + PAD);
        // Saturating, never `self.height - bh`: a box taller than the screen
        // wrapped that subtraction into a vast `by`, so every draw was clipped
        // away and the modal painted **nothing** while still waiting for a key.
        // An approval dialog that is invisible but live is the worst failure
        // this code has — the human cannot see what they are approving, and it
        // reads as a frozen shell. Callers must also budget their rows
        // ([`modal_max_rows`]); this is the backstop.
        let bx = self.width.saturating_sub(bw) / 2;
        let by = self.height.saturating_sub(bh) / 2;
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

/// Draw a multi-option question modal. `focus` is the highlighted option index.
/// Options are rendered as numbered rows; the footer shows Enter=select Esc=cancel.
/// Each option row is mouse-clickable ([`ModalHit::Choose`]).
pub fn draw_choose(title: &str, msg: &str, options: &[&str], focus: usize) {
    MODAL_ON.store(true, core::sync::atomic::Ordering::Relaxed);
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.cursor_restore();
            sc.cur_vis = false;
            MODAL_RECTS.with(|m| *m = [(0, 0, 0, 0); 3]);
            LIST_BROWSER_GEOM.with(|g| *g = None);
            CHOOSE_RECTS.with(|c| *c = [(0, 0, 0, 0); 9]);
            let n = options.len().min(9);
            CHOOSE_COUNT.store(n, core::sync::atomic::Ordering::Relaxed);
            let cols = sc.modal_cols() as usize;
            let mut pre: Vec<String> = Vec::new();
            if !msg.is_empty() {
                pre.extend(wrap(msg, cols));
            }
            // One option per row (ellipsized) so hit-testing is 1:1 with indices.
            let mut opt_lines: Vec<String> = Vec::new();
            for (i, opt) in options.iter().take(n).enumerate() {
                let mark = if i == focus { ">" } else { " " };
                let line = alloc::format!("{mark} {}. {}", i + 1, opt);
                opt_lines.push(crate::textsel::ellipsize(&line, cols));
            }
            let foot = "Enter select  Esc cancel  arrows/click";
            // The options are the actionable part and their rows are hit-tested
            // 1:1 with indices, so the *message* absorbs the clamp, never them.
            let pre = crate::panes_layout::clamp_modal_lines(
                pre,
                sc.modal_rows_budget().saturating_sub(opt_lines.len() + 1),
            );
            let rows = pre.len() + opt_lines.len() + 1;
            let (ix, iy, mcols) = sc.modal_box(title, rows as u64);
            let ch = sc.ch();
            let cw = sc.cw();
            let mut y = iy;
            for line in &pre {
                sc.draw_str(ix, y, line, sc.theme.chat_fg, sc.theme.status_bg);
                y += ch;
            }
            for (i, line) in opt_lines.iter().enumerate() {
                let fg = if i == focus {
                    sc.theme.accent
                } else {
                    sc.theme.chat_fg
                };
                let bg = if i == focus {
                    sc.theme.chat_bg
                } else {
                    sc.theme.status_bg
                };
                sc.fill_rect(ix, y, mcols * cw, ch, bg);
                sc.draw_str(ix, y, line, fg, bg);
                CHOOSE_RECTS.with(|c| c[i] = (ix, y, mcols * cw, ch));
                y += ch;
            }
            sc.draw_str(ix, y, foot, sc.theme.composer_hint, sc.theme.status_bg);
            sc.cursor_overlay();
        }
    });
}

/// Draw an approval (yes/no) modal. `focus_yes` highlights the Yes button.
pub fn draw_confirm(title: &str, msg: &str, focus_yes: bool) {
    MODAL_ON.store(true, core::sync::atomic::Ordering::Relaxed);
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.cursor_restore();
            sc.cur_vis = false;
            MODAL_RECTS.with(|m| *m = [(0, 0, 0, 0); 3]);
            LIST_BROWSER_GEOM.with(|g| *g = None);
            CHOOSE_COUNT.store(0, core::sync::atomic::Ordering::Relaxed);
            // Wrap first, then size the box to the wrapped line count + a gap +
            // the button row, so a long consent message never overflows. The
            // budget leaves those two rows for the buttons — an agent's args can
            // be kilobytes (a whole config file), and before the clamp that made
            // the box taller than the screen and the dialog invisible.
            let lines = wrap(msg, sc.modal_cols() as usize);
            let lines = crate::panes_layout::clamp_modal_lines(lines, sc.modal_rows_budget().saturating_sub(2));
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

/// Draw a macOS-style **About ChittiOS** dialog: logo, version, build, arch,
/// tagline, and an OK button (plus FA close). Clicking the status-bar **logo** or
/// running `/about` opens this.
pub fn draw_about() {
    MODAL_ON.store(true, core::sync::atomic::Ordering::Relaxed);
    SCREEN.with(|slot| {
        let Some(sc) = slot else { return };
        sc.cursor_restore();
        sc.cur_vis = false;
        MODAL_RECTS.with(|m| *m = [(0, 0, 0, 0); 3]);
        clear_modal_close_rect();
        LIST_BROWSER_GEOM.with(|g| *g = None);
        CHOOSE_COUNT.store(0, core::sync::atomic::Ordering::Relaxed);

        let cw = sc.cw();
        let ch = sc.ch();
        // Size the card from content so the tagline + OK never clip the border.
        let cols = ((sc.width / cw) * 2 / 5).clamp(30, 42);
        let logo_r = (ch * 2).max(18);
        // close pad + logo (2r) + gaps + name + ver + built + arch + sep + 2 tags + OK + pads
        let content_h = ch // top pad / close row
            + logo_r * 2
            + ch // gap under logo
            + ch // name
            + ch / 4
            + ch // version
            + ch // built
            + ch // arch
            + ch / 2 // sep gap
            + 2 // sep
            + ch / 2
            + ch // tag1
            + ch // tag2
            + ch / 2
            + ch // OK
            + ch / 2; // bottom pad
        let bw = cols * cw + 2 * (BORDER + PAD);
        let bh = content_h + 2 * (BORDER + PAD);
        let bx = (sc.width - bw) / 2;
        let by = (sc.height - bh) / 2;
        let bg = sc.theme.status_bg;
        sc.drop_shadow(bx, by, bw, bh);
        sc.fill_rect(bx, by, bw, bh, bg);
        sc.rect_outline(bx, by, bw, bh, BORDER, sc.theme.accent);

        let ix = bx + BORDER + PAD;
        let content_w = cols * cw;
        // Close (FA xmark) top-right — dedicated hit rect so it always works.
        let mark = crate::icons::close_mark();
        let (close_w, _) = sc.glyph_cell(mark);
        let close_w = close_w.max(cw * 2);
        let cx = ix + content_w.saturating_sub(close_w);
        let close_y = by + BORDER + PAD / 2;
        let (iw, _) = sc.glyph_cell(mark);
        sc.blit_glyph(
            cx + close_w.saturating_sub(iw) / 2,
            close_y,
            mark,
            sc.theme.accent,
            bg,
        );
        set_modal_close_rect((cx, close_y, close_w, ch));

        // Large brand logo (ui.json theme.logo / logo_node).
        let logo_cy = by + BORDER + PAD + ch / 2 + logo_r;
        sc.draw_logo(
            bx + bw / 2,
            logo_cy,
            logo_r,
            sc.theme.logo,
            sc.theme.logo_node,
        );

        let mut y = logo_cy + logo_r + ch / 2;
        let centre = |s: &str| bx + (bw.saturating_sub(s.chars().count() as u64 * cw)) / 2;

        sc.draw_str(centre("ChittiOS"), y, "ChittiOS", sc.theme.logo, bg);
        y += ch + ch / 4;

        let ver = alloc::format!("Version {}", crate::VERSION);
        sc.draw_str(centre(&ver), y, &ver, sc.theme.chat_fg, bg);
        y += ch;

        let built = alloc::format!("Built {}", crate::BUILD_TIME);
        sc.draw_str(centre(&built), y, &built, sc.theme.title_dim, bg);
        y += ch;

        #[cfg(target_arch = "x86_64")]
        let arch = "x86_64";
        #[cfg(target_arch = "aarch64")]
        let arch = "aarch64";
        let arch_line = alloc::format!("{arch}  ·  {} cores", crate::arch::cpu_count());
        sc.draw_str(centre(&arch_line), y, &arch_line, sc.theme.title_dim, bg);
        y += ch + ch / 2;

        sc.fill_rect(ix + cw * 2, y, content_w.saturating_sub(cw * 4), 1, sc.theme.sep_dim);
        y += ch / 2 + 2;

        let tag = "An agentic operating system.";
        sc.draw_str(centre(tag), y, tag, sc.theme.chat_fg, bg);
        y += ch;
        let tag2 = "The agent is the driver.";
        sc.draw_str(centre(tag2), y, tag2, sc.theme.title_dim, bg);
        y += ch + ch / 2;

        // OK button, centred, inside the card.
        let btn_w = 4 * cw;
        let btn_x = bx + (bw.saturating_sub(btn_w)) / 2;
        sc.modal_button(btn_x, y, "OK", true, 2);
        sc.cursor_overlay();
    });
}

/// macOS-style status-bar dropdown for `chip`. Anchored under/above the bar
/// near the chip's hit rect. Click outside, Esc, or Close dismisses.
pub fn draw_status_menu(chip: StatusChip) {
    MODAL_ON.store(true, core::sync::atomic::Ordering::Relaxed);
    SCREEN.with(|slot| {
        let Some(sc) = slot else { return };
        sc.cursor_restore();
        sc.cur_vis = false;
        MODAL_RECTS.with(|m| *m = [(0, 0, 0, 0); 3]);
        clear_modal_close_rect();
        LIST_BROWSER_GEOM.with(|g| *g = None);
        CHOOSE_COUNT.store(0, core::sync::atomic::Ordering::Relaxed);

        // Repaint the bar so chip rects / live values are current, then overlay.
        sc.draw_status();

        let cw = sc.cw();
        let ch = sc.ch();
        let anchor = STATUS_CHIP_RECTS.with(|a| a[chip as usize]);
        let bar = STATUS_BAR_RECT.with(|r| *r);
        let pos = sc.layout.status_pos;

        // Menu size: clock is larger (analog face); volume is a compact control.
        let (cols, rows) = match chip {
            StatusChip::Clock => (32u64, 16u64),
            StatusChip::Volume => (28, 12),
            StatusChip::Net => (36, 12),
            StatusChip::Mem | StatusChip::Cpu => (34, 10),
            StatusChip::Battery => (30, 9),
            StatusChip::Kbd | StatusChip::Mouse => (28, 8),
            StatusChip::Brand => (28, 8),
        };
        let bw = cols * cw + 2 * (BORDER + PAD);
        let bh = rows * ch + 2 * (BORDER + PAD);

        // Anchor: prefer under a top bar, above a bottom bar, else to the side.
        let mut bx = if anchor.2 > 0 {
            anchor.0.saturating_add(anchor.2 / 2).saturating_sub(bw / 2)
        } else {
            (sc.width - bw) / 2
        };
        if bx + bw > sc.width.saturating_sub(OUTER) {
            bx = sc.width.saturating_sub(bw + OUTER);
        }
        if bx < OUTER {
            bx = OUTER;
        }
        let by = match pos {
            crate::panes_layout::StatusPos::Top => bar.1 + bar.3 + 4,
            crate::panes_layout::StatusPos::Bottom => bar.1.saturating_sub(bh + 4),
            crate::panes_layout::StatusPos::Left => {
                bx = bar.0 + bar.2 + 4;
                if anchor.2 > 0 {
                    anchor.1
                } else {
                    (sc.height - bh) / 2
                }
            }
            crate::panes_layout::StatusPos::Right => {
                bx = bar.0.saturating_sub(bw + 4);
                if anchor.2 > 0 {
                    anchor.1
                } else {
                    (sc.height - bh) / 2
                }
            }
        };
        let by = by.min(sc.height.saturating_sub(bh + OUTER)).max(OUTER);

        let bg = sc.theme.status_bg;
        sc.drop_shadow(bx, by, bw, bh);
        sc.fill_rect(bx, by, bw, bh, bg);
        sc.rect_outline(bx, by, bw, bh, BORDER, sc.theme.accent);

        let ix = bx + BORDER + PAD;
        let mut y = by + BORDER + PAD;
        let content_w = cols * cw;

        // Title row + close.
        let title = match chip {
            StatusChip::Brand => "ChittiOS",
            StatusChip::Kbd => "Keyboard",
            StatusChip::Mouse => "Mouse",
            StatusChip::Net => "Network",
            StatusChip::Mem => "Memory",
            StatusChip::Cpu => "Processor",
            StatusChip::Battery => "Battery",
            StatusChip::Volume => "Sound",
            StatusChip::Clock => "Clock",
        };
        let icon = match chip {
            StatusChip::Brand => crate::icons::fa::HOUSE,
            StatusChip::Kbd => crate::icons::fa::KEYBOARD,
            StatusChip::Mouse => crate::icons::fa::MOUSE,
            StatusChip::Net => crate::icons::fa::WIFI,
            StatusChip::Mem => crate::icons::fa::MEMORY,
            StatusChip::Cpu => crate::icons::fa::MICROCHIP,
            StatusChip::Battery => crate::icons::fa::BATTERY,
            StatusChip::Volume => {
                crate::icons::volume_icon(crate::sound::muted(), crate::sound::volume())
            }
            StatusChip::Clock => crate::icons::fa::CLOCK,
        };
        sc.draw_str(
            ix,
            y,
            &alloc::format!("{icon} {title}"),
            sc.theme.accent,
            bg,
        );
        let mark = crate::icons::close_mark();
        let (close_w, _) = sc.glyph_cell(mark);
        let close_w = close_w.max(cw * 2);
        let cx = ix + content_w.saturating_sub(close_w);
        let (iw, _) = sc.glyph_cell(mark);
        sc.blit_glyph(
            cx + close_w.saturating_sub(iw) / 2,
            y,
            mark,
            sc.theme.accent,
            bg,
        );
        // Dedicated close rect (must not share slot 0 with Yes / menu body).
        set_modal_close_rect((cx, y, close_w, ch));
        y += ch + 4;
        sc.fill_rect(ix, y, content_w, 1, sc.theme.sep_dim);
        y += ch / 2;

        match chip {
            StatusChip::Clock => {
                y = draw_clock_menu_body(sc, ix, y, content_w, ch, cw, bg);
            }
            StatusChip::Net => {
                y = draw_net_menu_body(sc, ix, y, ch, bg);
            }
            StatusChip::Mem => {
                y = draw_mem_menu_body(sc, ix, y, ch, bg);
            }
            StatusChip::Cpu => {
                y = draw_cpu_menu_body(sc, ix, y, ch, bg);
            }
            StatusChip::Battery => {
                y = draw_battery_menu_body(sc, ix, y, ch, bg);
            }
            StatusChip::Volume => {
                y = draw_volume_menu_body(sc, ix, y, content_w, ch, cw, bg);
            }
            StatusChip::Kbd => {
                y = draw_input_menu_body(sc, ix, y, ch, bg, true);
            }
            StatusChip::Mouse => {
                y = draw_input_menu_body(sc, ix, y, ch, bg, false);
            }
            StatusChip::Brand => {
                sc.draw_str(ix, y, "Click for About…", sc.theme.chat_fg, bg);
                y += ch;
                sc.draw_str(ix, y, &alloc::format!("v{}", crate::VERSION), sc.theme.title_dim, bg);
            }
        }

        // Footer hint.
        let foot_y = by + bh - BORDER - PAD - ch;
        sc.draw_str(ix, foot_y, "Esc / click outside to close", sc.theme.title_dim, bg);
        // Full card is clickable for "inside" hit tests (slot 1 = menu body).
        MODAL_RECTS.with(|m| m[1] = (bx, by, bw, bh));
        sc.cursor_overlay();
    });
}

/// True if `(x,y)` is inside the open status menu panel (not the close mark).
pub fn status_menu_contains(x: u64, y: u64) -> bool {
    MODAL_RECTS.with(|m| in_rect(x, y, m[1]))
}

fn draw_clock_menu_body(
    sc: &Screen,
    ix: u64,
    mut y: u64,
    content_w: u64,
    ch: u64,
    cw: u64,
    bg: Rgb,
) -> u64 {
    let (yy, mo, d, h, mi, s, wd) =
        crate::clock::civil_from_unix(crate::clock::now_unix() + crate::clock::tz_offset() as i64);
    let weekdays = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
    let months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let date = alloc::format!(
        "{}, {} {} {}",
        weekdays[wd as usize],
        d,
        months[(mo as usize).saturating_sub(1).min(11)],
        yy
    );
    sc.draw_str(ix, y, &date, sc.theme.chat_fg, bg);
    y += ch + 4;

    // Analog face centred in the content width.
    let face_r = (ch * 3).max(28).min(content_w / 2 - 4);
    let fcx = ix + content_w / 2;
    let fcy = y + face_r + 4;
    draw_analog_clock(sc, fcx as i64, fcy as i64, face_r as i64, h, mi, s);
    y = fcy + face_r + ch;

    let time = alloc::format!("{:02}:{:02}:{:02}", h, mi, s);
    let tx = ix + content_w.saturating_sub(time.len() as u64 * cw) / 2;
    sc.draw_str(tx, y, &time, sc.theme.accent, bg);
    y += ch;
    let tz = crate::clock::format_tz();
    let tzx = ix + content_w.saturating_sub(tz.chars().count() as u64 * cw) / 2;
    sc.draw_str(tzx, y, &tz, sc.theme.title_dim, bg);
    y += ch + ch / 2;
    sc.draw_str(ix, y, "Timezone via /datetime tz …", sc.theme.title_dim, bg);
    y + ch
}

/// Analog clock: 0° at 12 o'clock, clockwise. Integer Q10 sin/cos.
fn draw_analog_clock(sc: &Screen, cx: i64, cy: i64, r: i64, h: i64, mi: i64, s: i64) {
    let ring = sc.theme.border_dim;
    let fg = sc.theme.chat_fg;
    let accent = sc.theme.accent;
    // Outer ring + centre hub.
    sc.fill_disc(cx, cy, r, sc.theme.chat_bg);
    // Ring outline ≈ 2px via two discs.
    let t = (r / 16).max(1);
    for deg in (0..360).step_by(2) {
        let (dx, dy) = clock_offset(r, deg);
        sc.put_pixel((cx + dx) as u64, (cy + dy) as u64, ring);
        let (dx2, dy2) = clock_offset(r - t, deg);
        sc.put_pixel((cx + dx2) as u64, (cy + dy2) as u64, ring);
    }
    // Hour ticks.
    for hour in 0..12 {
        let deg = hour * 30;
        let (ox, oy) = clock_offset(r - 2, deg);
        let (ix, iy) = clock_offset(r - r / 5, deg);
        draw_line_i(sc, cx + ix, cy + iy, cx + ox, cy + oy, fg);
    }
    // Hands: hour (short), minute, second (accent).
    let h_deg = ((h % 12) * 30 + mi / 2) as i32;
    let m_deg = (mi * 6 + s / 10) as i32;
    let s_deg = (s * 6) as i32;
    let (hx, hy) = clock_offset(r * 55 / 100, h_deg);
    draw_line_i(sc, cx, cy, cx + hx, cy + hy, fg);
    // Thicken hour hand with a parallel stroke.
    draw_line_i(sc, cx + 1, cy, cx + hx + 1, cy + hy, fg);
    let (mx, my) = clock_offset(r * 78 / 100, m_deg);
    draw_line_i(sc, cx, cy, cx + mx, cy + my, fg);
    let (sx, sy) = clock_offset(r * 88 / 100, s_deg);
    draw_line_i(sc, cx, cy, cx + sx, cy + sy, accent);
    sc.fill_disc(cx, cy, (r / 12).max(2), accent);
}

/// (dx, dy) from centre: deg 0 = 12 o'clock, clockwise. Length `r`.
fn clock_offset(r: i64, deg: i32) -> (i64, i64) {
    // math angle from +x: 90° − deg → cos/sin in Q10
    let m = (90 - deg).rem_euclid(360);
    let (c, s) = cos_sin_q10(m);
    // clock: x = r·sin(deg), y = −r·cos(deg); sin(deg)=cos(m), cos(deg)=sin(m)
    ((r * c as i64) / 1024, -((r * s as i64) / 1024))
}

/// cos/sin of `deg` (0..359) in Q10 (1024 = 1.0).
fn cos_sin_q10(deg: i32) -> (i32, i32) {
    let d = deg.rem_euclid(360) as usize;
    // 0..90 table for cos; sin(d)=cos(90-d)
    const COS: [i32; 91] = [
        1024, 1023, 1023, 1022, 1021, 1020, 1018, 1016, 1014, 1011, 1008, 1005, 1001, 997, 993, 988,
        983, 978, 972, 966, 960, 953, 946, 939, 931, 923, 915, 906, 897, 888, 878, 868, 858, 847,
        836, 825, 814, 802, 790, 777, 765, 752, 739, 725, 711, 697, 683, 668, 653, 638, 623, 607,
        591, 575, 559, 542, 526, 509, 492, 475, 457, 440, 422, 404, 386, 368, 350, 331, 313, 294,
        275, 256, 237, 218, 199, 180, 160, 141, 121, 102, 82, 62, 42, 22, 2, 0, 0, 0, 0, 0, 0,
    ];
    // Fix last entries properly for 85..90
    // Actually COS[90] should be 0; above is approximate. Use safe index.
    let cos_q = |a: usize| -> i32 {
        let a = a.min(90);
        // regenerate clean values for key angles
        match a {
            0 => 1024,
            30 => 887,
            45 => 724,
            60 => 512,
            90 => 0,
            _ => COS[a],
        }
    };
    let (c, s) = match d {
        0..=90 => (cos_q(d), cos_q(90 - d)),
        91..=180 => (-cos_q(d - 90), cos_q(180 - d)),
        181..=270 => (-cos_q(270 - d), -cos_q(d - 180)),
        _ => (cos_q(360 - d), -cos_q(d - 270)),
    };
    (c, s)
}

fn draw_line_i(sc: &Screen, x0: i64, y0: i64, x1: i64, y1: i64, c: Rgb) {
    let mut x0 = x0;
    let mut y0 = y0;
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        if x0 >= 0 && y0 >= 0 {
            sc.put_pixel(x0 as u64, y0 as u64, c);
        }
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

/// Volume dropdown body. Registers three clickable [`ModalHit::Choose`] rows:
/// `0` = mute toggle, `1` = −5%, `2` = +5%. Also paints a level bar.
fn draw_volume_menu_body(
    sc: &Screen,
    ix: u64,
    mut y: u64,
    content_w: u64,
    ch: u64,
    cw: u64,
    bg: Rgb,
) -> u64 {
    let muted = crate::sound::muted();
    let pct = crate::sound::volume();
    let icon = crate::icons::volume_icon(muted, pct);
    let label = if muted {
        alloc::format!("{icon}  Muted  ({pct}%)")
    } else {
        alloc::format!("{icon}  Output  {pct}%")
    };
    sc.draw_str(ix, y, &label, sc.theme.chat_fg, bg);
    y += ch + 4;

    // Level bar.
    let bar_h = (ch / 2).max(6);
    let bar_w = content_w;
    sc.fill_rect(ix, y, bar_w, bar_h, sc.theme.border_dim);
    let fill = (bar_w as u32 * pct / 100) as u64;
    if fill > 0 && !muted {
        sc.fill_rect(ix, y, fill, bar_h, sc.theme.accent);
    } else if fill > 0 && muted {
        sc.fill_rect(ix, y, fill, bar_h, sc.theme.title_dim);
    }
    y += bar_h + ch / 2;

    // Device line.
    let dev = if crate::sound::is_up() {
        "Device  PCM ready"
    } else {
        "Device  none (software gain still applies)"
    };
    sc.draw_str(ix, y, dev, sc.theme.title_dim, bg);
    y += ch + 4;

    // Clickable action rows → Choose(0..2).
    CHOOSE_RECTS.with(|c| *c = [(0, 0, 0, 0); 9]);
    let actions: [(&str, bool); 3] = [
        (
            if muted { "Unmute" } else { "Mute" },
            true,
        ),
        ("Volume  −5%", true),
        ("Volume  +5%", true),
    ];
    CHOOSE_COUNT.store(actions.len(), core::sync::atomic::Ordering::Relaxed);
    for (i, (text, _)) in actions.iter().enumerate() {
        let row_bg = sc.theme.chat_bg;
        sc.fill_rect(ix, y, content_w, ch, row_bg);
        let prefix = match i {
            0 => crate::icons::fa::VOLUME_XMARK,
            1 => crate::icons::fa::MINUS,
            _ => crate::icons::fa::PLUS,
        };
        sc.draw_str(
            ix,
            y,
            &alloc::format!("{prefix}  {text}"),
            sc.theme.chat_fg,
            row_bg,
        );
        CHOOSE_RECTS.with(|c| c[i] = (ix, y, content_w, ch));
        y += ch + 2;
    }
    y += ch / 2;
    sc.draw_str(
        ix,
        y,
        "Wheel / ←→  adjust · m mute",
        sc.theme.title_dim,
        bg,
    );
    // Silence unused cw (kept for API symmetry with other drawers).
    let _ = cw;
    y + ch
}

fn draw_net_menu_body(sc: &Screen, ix: u64, mut y: u64, ch: u64, bg: Rgb) -> u64 {
    let up = crate::net::is_up();
    sc.draw_str(
        ix,
        y,
        if up { "Status   Connected" } else { "Status   Offline" },
        if up { sc.theme.accent } else { sc.theme.title_dim },
        bg,
    );
    y += ch;
    if let Some(info) = crate::net::info() {
        sc.draw_str(ix, y, &alloc::format!("Interface  {}", info.ifname), sc.theme.chat_fg, bg);
        y += ch;
        let mac = alloc::format!(
            "MAC  {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
            info.mac[0], info.mac[1], info.mac[2], info.mac[3], info.mac[4], info.mac[5]
        );
        sc.draw_str(ix, y, &mac, sc.theme.title_dim, bg);
        y += ch;
        if let Some(ip) = info.ip {
            sc.draw_str(ix, y, &alloc::format!("IPv4  {}", ip), sc.theme.chat_fg, bg);
            y += ch;
        }
        if let Some(gw) = info.gateway {
            sc.draw_str(ix, y, &alloc::format!("Gateway  {}", gw), sc.theme.title_dim, bg);
            y += ch;
        }
        if !info.dns.is_empty() {
            sc.draw_str(ix, y, &alloc::format!("DNS  {}", info.dns[0]), sc.theme.title_dim, bg);
            y += ch;
        }
        sc.draw_str(
            ix,
            y,
            if info.dhcp { "Config  DHCP" } else { "Config  Static" },
            sc.theme.title_dim,
            bg,
        );
        y += ch;
    } else {
        sc.draw_str(ix, y, "No network device bound", sc.theme.title_dim, bg);
        y += ch;
    }
    y += ch / 2;
    sc.draw_str(ix, y, "Shell: /network  /wifi  /ping", sc.theme.title_dim, bg);
    y + ch
}

fn draw_mem_menu_body(sc: &Screen, ix: u64, mut y: u64, ch: u64, bg: Rgb) -> u64 {
    let m = crate::mm::mem_stats();
    let mib = 1024 * 1024;
    sc.draw_str(
        ix,
        y,
        &alloc::format!("Heap used   {} MiB", m.heap_used / mib),
        sc.theme.chat_fg,
        bg,
    );
    y += ch;
    sc.draw_str(
        ix,
        y,
        &alloc::format!("Heap total  {} MiB", m.heap_total / mib),
        sc.theme.title_dim,
        bg,
    );
    y += ch;
    sc.draw_str(
        ix,
        y,
        &alloc::format!("RAM total   {} MiB", m.ram_total / mib),
        sc.theme.title_dim,
        bg,
    );
    y += ch;
    let reserved = m.ram_reserved.saturating_sub(m.heap_total);
    sc.draw_str(
        ix,
        y,
        &alloc::format!("Reserved    {} MiB", reserved / mib),
        sc.theme.title_dim,
        bg,
    );
    y + ch
}

fn draw_cpu_menu_body(sc: &Screen, ix: u64, mut y: u64, ch: u64, bg: Rgb) -> u64 {
    let pct = crate::shell::cpu_percent();
    let cores = crate::arch::cpu_count();
    sc.draw_str(ix, y, &alloc::format!("Load     {pct}%"), sc.theme.chat_fg, bg);
    y += ch;
    sc.draw_str(ix, y, &alloc::format!("Cores    {cores}"), sc.theme.title_dim, bg);
    y += ch;
    #[cfg(target_arch = "x86_64")]
    let arch = "x86_64";
    #[cfg(target_arch = "aarch64")]
    let arch = "aarch64";
    sc.draw_str(ix, y, &alloc::format!("Arch     {arch}"), sc.theme.title_dim, bg);
    y += ch + ch / 2;
    sc.draw_str(ix, y, "Shell: /top  /perf", sc.theme.title_dim, bg);
    y + ch
}

fn draw_battery_menu_body(sc: &Screen, ix: u64, mut y: u64, ch: u64, bg: Rgb) -> u64 {
    if let Some(b) = crate::drivers::battery::cached() {
        sc.draw_str(
            ix,
            y,
            &alloc::format!("Charge   {}", crate::drivers::battery::format(&b)),
            sc.theme.chat_fg,
            bg,
        );
        y += ch;
        sc.draw_str(ix, y, "Source   ACPI _BST / EC", sc.theme.title_dim, bg);
        y += ch;
        sc.draw_str(ix, y, "Shell: /battery", sc.theme.title_dim, bg);
    } else {
        sc.draw_str(ix, y, "No battery reported", sc.theme.title_dim, bg);
        y += ch;
        sc.draw_str(ix, y, "(desktop / no ACPI pack)", sc.theme.title_dim, bg);
    }
    y + ch
}

fn draw_input_menu_body(sc: &Screen, ix: u64, mut y: u64, ch: u64, bg: Rgb, kbd: bool) -> u64 {
    if kbd {
        let last = crate::console::input_activity_ms();
        let active = last != 0 && crate::arch::now_ms().saturating_sub(last) < 1500;
        sc.draw_str(
            ix,
            y,
            if active { "Keyboard  Active" } else { "Keyboard  Idle" },
            if active { sc.theme.accent } else { sc.theme.chat_fg },
            bg,
        );
        y += ch;
        sc.draw_str(ix, y, "USB HID / virtio / PS-2", sc.theme.title_dim, bg);
    } else {
        let last = crate::mouse::activity_ms();
        let active = last != 0 && crate::arch::now_ms().saturating_sub(last) < 1500;
        sc.draw_str(
            ix,
            y,
            if active { "Mouse  Active" } else { "Mouse  Idle" },
            if active { sc.theme.accent } else { sc.theme.chat_fg },
            bg,
        );
        y += ch;
        sc.draw_str(ix, y, "Pointer + wheel scroll", sc.theme.title_dim, bg);
    }
    y + ch
}

/// Draw a text-input modal (masked = password dots). `caret_on` blinks the caret.
pub fn draw_input(title: &str, prompt: &str, buf: &str, masked: bool, caret_on: bool) {
    MODAL_ON.store(true, core::sync::atomic::Ordering::Relaxed);
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.cursor_restore();
            sc.cur_vis = false;
            MODAL_RECTS.with(|m| *m = [(0, 0, 0, 0); 3]);
            LIST_BROWSER_GEOM.with(|g| *g = None);
            CHOOSE_COUNT.store(0, core::sync::atomic::Ordering::Relaxed);
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
/// Draw the searchable list modal used by `/help` (Commands) and `/agents`.
/// `title` is the window chrome label (e.g. `"Commands"` / `"Agents"`).
pub fn draw_commands_browser(
    query: &str,
    rows: &[CommandsRow<'_>],
    scroll: usize,
    total: usize,
    caret_on: bool,
) {
    draw_list_browser("Commands", query, rows, scroll, total, caret_on);
}

/// Same as [`draw_commands_browser`] with a custom window title.
pub fn draw_list_browser(
    title: &str,
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
        CHOOSE_COUNT.store(0, core::sync::atomic::Ordering::Relaxed);

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
        // Do **not** full-screen `shade_rect` here: the caret blink repaints this
        // every ~500 ms, and shade multiplies darkness each pass → cascading
        // black + flicker. Same solid modal box as `/help` (draw_commands_browser).
        sc.drop_shadow(bx, by, bw, bh);
        sc.fill_rect(bx, by, bw, bh, bg);
        sc.rect_outline(bx, by, bw, bh, BORDER, sc.theme.accent);

        let ix = bx + BORDER + PAD;
        let mut y = by + BORDER + PAD;
        let content_w = cols * cw;

        // Title + FA xmark close (theme accent — same as pane close).
        let mark = crate::icons::close_mark();
        let (close_w, _) = sc.glyph_cell(mark);
        let close_w = close_w.max(cw * 2);
        sc.draw_str(
            ix,
            y,
            &crate::textsel::ellipsize(
                title,
                cols.saturating_sub((close_w / cw).max(2)) as usize,
            ),
            sc.theme.accent,
            bg,
        );
        // Hit target ≥ square FA cell; glyph centred like the pane close chrome.
        let cx = ix + content_w.saturating_sub(close_w);
        let (iw, _) = sc.glyph_cell(mark);
        sc.blit_glyph(cx + close_w.saturating_sub(iw) / 2, y, mark, sc.theme.accent, bg);
        set_modal_close_rect((cx, y, close_w, ch));
        // Keep slot 0 empty so Close is only the dedicated rect (not Yes).
        MODAL_RECTS.with(|m| m[0] = (0, 0, 0, 0));
        y += ch + 4;
        sc.fill_rect(ix, y, content_w, 1, sc.theme.sep_dim);
        y += 6;

        // Search field (FA magnifying-glass + label).
        let search_lab = alloc::format!("{} search", crate::icons::fa::SEARCH);
        sc.draw_str(ix, y, &search_lab, sc.theme.title_dim, bg);
        // Label width: FA cell (= line height) + " search" (7 mono cells).
        let lab_w = ch + 7 * cw;
        let field_x = ix + lab_w + cw / 2;
        let field_w = content_w.saturating_sub(lab_w + cw / 2);
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
        let visible_n = rows.len().min(view_rows as usize);
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
        // Mouse hit-testing for every visible row (headers included; caller
        // skips non-items). Absolute index = scroll + visible row.
        LIST_BROWSER_GEOM.with(|g| {
            *g = Some(ListBrowserGeom {
                list_x: ix,
                list_y: list_top,
                list_w,
                row_h: ch,
                n_rows: visible_n,
                scroll,
            });
        });

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
        let foot = if title.eq_ignore_ascii_case("Agents") {
            "up/dn  |  Enter/click select  |  Esc close"
        } else {
            "up/dn  |  Enter/click fill  |  Esc close"
        };
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
    clear_modal_close_rect();
    LIST_BROWSER_GEOM.with(|g| *g = None);
    CHOOSE_COUNT.store(0, core::sync::atomic::Ordering::Relaxed);
    CHOOSE_RECTS.with(|c| *c = [(0, 0, 0, 0); 9]);
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

/// Re-init the console onto a **different framebuffer** — a new address, pitch and
/// size — after a real mode set, preserving the session.
///
/// Unlike `init_console_raw_fmt` this keeps the layout config, the status text, the
/// interactive state and (via [`Pane::adopt`]) every pane's scrollback, so changing
/// resolution does not clear the screen or drop history; it reflows, exactly as a
/// font-scale change does. No splash, either — this is not a boot.
///
/// The logical-desktop preference is deliberately **dropped** here: it existed to
/// letterbox a desktop inside a too-large panel, and a real mode set is the better
/// answer to the same problem. Keeping it would letterbox *inside* the new mode.
#[allow(clippy::too_many_arguments)]
pub fn reinit_scanout(
    addr: usize,
    width: u64,
    height: u64,
    pitch: u64,
    bpp_bytes: u64,
    r_shift: u32,
    g_shift: u32,
    b_shift: u32,
) {
    // No console yet — this device *is* the display (virtio-gpu with no firmware
    // framebuffer behind it), so bring the console up on it rather than returning.
    // Without this a KMS-only machine boots to a blank screen: the compositor has
    // nothing to re-init and the driver has nowhere to draw.
    if SCREEN.with(|slot| slot.is_none()) {
        init_console_raw_fmt(addr, width, height, pitch, bpp_bytes, r_shift, g_shift, b_shift);
        return;
    }
    SCREEN.with(|slot| {
        let Some(old) = slot.as_ref() else { return };
        let cfg = old.layout.clone();
        let split = old.any_action_open() || old.actions.len() > 1;
        let focused = old.focused_action;
        let mut ns = Screen::build(
            addr, width, height, pitch, bpp_bytes, r_shift, g_shift, b_shift, &cfg, split, focused,
            None, // a real mode set replaces the letterbox, it does not nest inside it
        );
        ns.status_left = old.status_left.clone();
        ns.status_right = old.status_right.clone();
        carry_tabs(&mut ns, old);
        preserve_interactive(&mut ns, old);
        ns.chat.adopt(&old.chat);
        if !ns.any_action_open() {
            ns.focus_action = false;
        }
        ns.redraw();
        *slot = Some(ns);
    });
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
/// When the bordered composer is active, keystroke echo is handled by
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

/// Whether the chat pane has a bordered input composer (always true once the
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

/// Set the left half of the composer hint bar (live status:
/// "Waiting for response… |").
///
/// Always repaints the composer strip when the chat has a composer — including
/// while a turn is running (`composer_active == false` after submit). The old
/// gate only drew during typing, so wait animation never appeared.
pub fn composer_set_hint_left(s: &str) {
    composer_set_hint_left_lead(s, &[]);
}

/// Set the left hint, colouring its first `lead.len()` characters from `lead`
/// (one colour per character) instead of `theme.composer_hint`.
///
/// This is the shell's progress-bar channel: the animation lives in
/// `shell::chrome` and only the finished per-cell colours arrive here, so the
/// compositor stays ignorant of the frame sequence. Pass an empty `lead` for
/// an ordinary single-colour hint.
pub fn composer_set_hint_left_lead(s: &str, lead: &[(u8, u8, u8)]) {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.composer_hint_l.clear();
            sc.composer_hint_l.push_str(s);
            sc.composer_hint_l_lead.clear();
            sc.composer_hint_l_lead.extend_from_slice(lead);
            if sc.chat.has_composer {
                sc.cursor_restore();
                sc.cur_vis = false;
                sc.draw_composer();
                sc.cursor_overlay();
            }
        }
    });
}

/// The live theme's progress-gradient endpoints: `(dim, bright)` =
/// `(composer_hint, accent)`. The shell's wait animation ramps between these
/// so it follows `/theme` and stays on the brand palette rather than carrying
/// colours of its own. Falls back to the brand dark theme before the
/// framebuffer exists (serial-only boot).
pub fn hint_gradient() -> ((u8, u8, u8), (u8, u8, u8)) {
    SCREEN.with(|slot| match slot.as_ref() {
        Some(sc) => (sc.theme.composer_hint, sc.theme.accent),
        None => (Theme::BRAND_DARK.composer_hint, Theme::BRAND_DARK.accent),
    })
}

/// Mark the last `n` absolute chat lines as a user-prompt band (elevated
/// `composer_bg`), pad them full-width, and repaint so the band is visible
/// immediately (including empty cells to the right of the text).
pub fn chat_mark_user_band_rows(n: usize) {
    if n == 0 {
        return;
    }
    SCREEN.with(|slot| {
        let Some(sc) = slot.as_mut() else { return };
        let p = &mut sc.chat;
        let cols = p.cols as usize;
        if cols == 0 {
            return;
        }
        // Cursor sits on the line *after* the last printed content when the
        // user turn ended with `\n`. The band covers the previous `n` lines.
        let end = p.hist.len() + p.row as usize; // exclusive end (current empty row)
        let start = end.saturating_sub(n);
        for gi in start..end {
            if let Err(i) = p.user_band.binary_search(&gi) {
                p.user_band.insert(i, gi);
            }
            // Cap bookkeeping so a long session cannot grow unbounded.
            if p.user_band.len() > 256 {
                p.user_band.drain(0..p.user_band.len() - 256);
            }
            // Pad short rows so the elevated fill spans the full pane width.
            if gi < p.hist.len() {
                let line = &mut p.hist[gi];
                if line.len() < cols {
                    let fg = p.default_fg;
                    line.resize(cols, ('\0', fg));
                }
            } else {
                let gr = gi - p.hist.len();
                if gr < p.rows as usize {
                    // Live grid rows are already full-width.
                }
            }
        }
        // Repaint band rows that are on screen (drop mut borrow first).
        let view = p.view;
        let hist_len = p.hist.len();
        let rows = p.rows as usize;
        let paint: alloc::vec::Vec<usize> = if view == 0 {
            let first = hist_len - view.min(hist_len);
            (start..end)
                .filter(|&gi| gi >= first && gi < first + rows)
                .collect()
        } else {
            alloc::vec::Vec::new()
        };
        for gi in paint {
            for c in 0..cols {
                sc.paint_chat_cell(&sc.chat, gi, c, false);
            }
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

/// The current right-hand composer hint (transient overlays like the reverse
/// search save it and restore it on exit).
pub fn composer_hint_right() -> String {
    SCREEN.with(|slot| slot.as_ref().map(|sc| sc.composer_hint_r.clone()).unwrap_or_default())
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

/// The action column whose `[x]` close button is under `(x, y)`, if any.
///
/// Returns the column index rather than a bool so a click on a **non-focused**
/// column's `[x]` closes that column's tab: testing only the focused column's
/// button meant the other columns' `[x]` were painted but dead.
pub fn close_hit_pane(x: u64, y: u64) -> Option<usize> {
    SCREEN.with(|slot| {
        let sc = slot.as_ref()?;
        (0..sc.actions.len()).find(|&i| {
            if sc.actions[i].is_empty() || !sc.column_visible(i) {
                return false;
            }
            let (bx, by, bw, bh) = sc.close_btn_for(i);
            x >= bx && x < bx + bw && y >= by && y < by + bh
        })
    })
}

/// The editor pane text-area geometry `(ix, iy, cw, ch, cols, text_rows)` so the
/// editor can map a click to a (row, col). `None` unless the editor is open.
pub fn editor_pane_geom() -> Option<(u64, u64, u64, u64, u64, u64)> {
    SCREEN.with(|slot| {
        slot.as_ref().and_then(|sc| {
            let d = sc.mode_dims(RightMode::Editor)?;
            Some((d.ix, d.iy, d.cw, d.ch, d.cols, d.rows.saturating_sub(1)))
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
            // Write into whichever action column currently shows ktrace.
            let Some((pi, _)) = sc.find_mode(RightMode::Ktrace) else {
                return;
            };
            if sc.actions[pi].right() != RightMode::Ktrace {
                return;
            }
            sc.cursor_restore();
            sc.cur_vis = false;
            let mut logs = core::mem::replace(&mut sc.actions[pi].pane, dummy_pane());
            for &b in s.as_bytes() {
                Screen::pane_putc(sc, &mut logs, b);
            }
            sc.actions[pi].pane = logs;
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
        folds: Vec::new(), user_band: Vec::new(), utf8: [0; 4], utf8_len: 0,
    }
}

/// The editor viewport size `(cols, rows)` inside the right pane — `rows` is the
/// text area (the bottom row is reserved for the editor's mode line). `None` if
/// the console isn't up.
pub fn editor_dims() -> Option<(usize, usize)> {
    SCREEN.with(|slot| {
        slot.as_ref().map(|sc| {
            // The editor's own column when its tab is up (it may not be the
            // focused one), else the focused column for a not-yet-opened editor.
            let (cols, rows) = match sc.mode_dims(RightMode::Editor) {
                Some(d) => (d.cols, d.rows),
                None => (sc.logs().cols, sc.logs().rows),
            };
            (cols as usize, (rows.saturating_sub(1)).max(1) as usize)
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
    ns.composer_hint_l_lead = old.composer_hint_l_lead.clone();
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

/// Carry the action columns' tabs, active index, and pane text from `old` into a
/// freshly-built `ns`, column by column.
///
/// When the new layout has **fewer** columns (`/pane max` shrank the budget) the
/// dropped columns' tabs are appended to the last surviving column rather than
/// discarded: a tab is a live process (a package-UI agent, a streaming ktrace, a
/// playing audio track), so dropping the list would leak the task and leave no
/// way to reach or close it.
fn carry_tabs(ns: &mut Screen, old: &Screen) {
    let n = ns.actions.len().min(old.actions.len());
    for i in 0..n {
        ns.actions[i].tabs = old.actions[i].tabs.clone();
        ns.actions[i].active = old.actions[i].active.min(ns.actions[i].tabs.len().saturating_sub(1));
        ns.actions[i].pane.adopt(&old.actions[i].pane);
    }
    if old.actions.len() > n && n > 0 {
        let last = n - 1;
        for i in n..old.actions.len() {
            for &m in &old.actions[i].tabs {
                if !ns.actions[last].tabs.contains(&m) {
                    ns.actions[last].tabs.push(m);
                }
            }
        }
        let len = ns.actions[last].tabs.len();
        ns.actions[last].active = ns.actions[last].active.min(len.saturating_sub(1));
    }
    ns.focused_action = old.focused_action.min(ns.actions.len().saturating_sub(1));
}

/// Rebuild geometry for a new split state, preserving layout config, status,
/// interactive state, action tabs, and pane text via [`Pane::adopt`].
fn rebuilt(old: &Screen, split: bool) -> Screen {
    // **Every screen rebuild invalidates pane interiors**, so the mark belongs here
    // rather than at each caller. Opening a view was wired up and closing one was not,
    // which is the same omission twice — a choke point ends that.
    mark_tabs_dirty();
    let mut ns = Screen::build(
        // `fb_w`/`fb_h`, never `width`/`height` — those are the logical desktop,
        // and feeding them back in would shrink the viewport on every rebuild.
        old.addr, old.fb_w, old.fb_h, old.pitch, old.bpp_bytes, old.r_shift, old.g_shift, old.b_shift, &old.layout, split,
        old.focused_action,
        old.logical_pref,
    );
    ns.status_left = old.status_left.clone();
    ns.status_right = old.status_right.clone();
    carry_tabs(&mut ns, old);
    preserve_interactive(&mut ns, old);
    if !ns.any_action_open() {
        ns.focus_action = false;
    }
    ns.chat.adopt(&old.chat);
    ns
}

/// The current (focused action column's active tab) mode.
pub fn right_mode() -> RightMode {
    SCREEN.with(|slot| slot.as_ref().map(|sc| sc.right()).unwrap_or(RightMode::Closed))
}

/// Index of the focused action column (0-based).
pub fn focused_action_index() -> usize {
    SCREEN.with(|slot| slot.as_ref().map(|sc| sc.focused_action).unwrap_or(0))
}

/// Number of action columns currently laid out.
pub fn action_column_count() -> usize {
    SCREEN.with(|slot| slot.as_ref().map(|sc| sc.actions.len()).unwrap_or(1))
}

/// The active tab mode of every **visible** action pane, in grid order.
///
/// A relayout (divider drag, grid reshape, tab move) repaints all the frames but
/// not the tab *interiors*, which each view owns. With more than one pane showing
/// content, the caller has to re-present every one of them — repainting only the
/// focused pane leaves the others blank until they happen to tick.
pub fn visible_tab_modes() -> Vec<RightMode> {
    SCREEN.with(|slot| {
        let Some(sc) = slot.as_ref() else { return Vec::new() };
        (0..sc.actions.len())
            .filter(|&i| sc.column_visible(i))
            .map(|i| sc.actions[i].right())
            .filter(|&m| m != RightMode::Closed)
            .collect()
    })
}

/// The open tab modes on the **focused** action column, in bar order.
pub fn tab_modes() -> Vec<RightMode> {
    SCREEN.with(|slot| {
        slot.as_ref()
            .map(|sc| sc.focused_slot().tabs.clone())
            .unwrap_or_default()
    })
}

/// True if a tab of `mode` is open on **any** action column.
pub fn has_tab(mode: RightMode) -> bool {
    SCREEN.with(|slot| slot.as_ref().map(|sc| sc.find_mode(mode).is_some()).unwrap_or(false))
}

/// Open `mode` on the focused action column (or select it if already open
/// anywhere — focuses that column). First tab on a collapsed band opens the split.
/// Set when the action band's geometry or tab set changed, so the pump knows the
/// panes' *interiors* need repainting.
///
/// The compositor redraws frames on a relayout, but each view owns its interior — a
/// browser page, a chess board, a paint canvas are RGB buffers the app holds, and
/// `Screen::redraw` cannot reproduce them. So an unfocused surface pane goes blank until
/// it happens to tick, which for a browser or a finished game is never.
///
/// `shell::repaint_visible_tabs` has always existed for this and documented that "a
/// divider drag, `/pane grid|max|split`, a tab move" must call it — and then had exactly
/// **one** caller (the theme path), so every other band change blanked its neighbours.
/// A flag drained by the pump fixes that class rather than that instance: a band mutation
/// added later gets the repaint without anyone remembering to ask.
static TABS_DIRTY: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Note that the action band changed and its panes' interiors need repainting.
pub fn mark_tabs_dirty() {
    TABS_DIRTY.store(true, core::sync::atomic::Ordering::Relaxed);
}

/// Consume the flag. Cleared *before* the caller repaints, so a view that pumps while
/// painting cannot drive itself round the loop again.
pub fn take_tabs_dirty() -> bool {
    TABS_DIRTY.swap(false, core::sync::atomic::Ordering::Relaxed)
}

fn open_view_slot(slot: &mut Option<Screen>, mode: RightMode) {
    mark_tabs_dirty();
    let Some(old) = slot else { return };
    // NB: opening a view must **not** move keyboard focus to the action pane
    // for most modes — the user typed a command at the composer and is still
    // typing there. Setting `focus_action` here made the *next* command go to
    // the pane instead of the prompt, which reads as the shell having frozen.
    // Focus moves only on an explicit act: clicking a pane, `/pane focus`,
    // Ctrl+Tab. **Exception: the editor** — it is an interactive text surface,
    // so keys must land there on open (and Ctrl+Tab still returns to the shell).
    if mode == RightMode::Editor {
        old.focus_action = true;
    }
    // Already open somewhere → select that pane's tab (and make it the open
    // target for the next view) without taking focus (except editor above).
    if let Some((pi, ti)) = old.find_mode(mode) {
        old.focused_action = pi;
        old.actions[pi].active = ti;
        old.repaint_action();
        return;
    }
    let fi = old.focused_action.min(old.actions.len().saturating_sub(1));
    // Only a lone action pane collapses, so only it needs a full relayout when
    // its first tab opens; a multi-pane grid is already on screen.
    let need_relayout = !old.any_action_open() && old.actions.len() == 1;
    if need_relayout || old.actions.is_empty() {
        let mut ns = rebuilt(old, true);
        let fi = ns.focused_action.min(ns.actions.len().saturating_sub(1));
        ns.actions[fi].tabs = alloc::vec![mode];
        ns.actions[fi].active = 0;
        ns.focused_action = fi;
        ns.redraw();
        *slot = Some(ns);
        return;
    }
    // Additional tab on the focused pane (geometry unchanged).
    let a = &mut old.actions[fi];
    a.tabs.push(mode);
    a.active = a.tabs.len() - 1;
    old.repaint_action();
}

/// Open (or focus) a tab for `mode` on the focused action column.
pub fn set_right(mode: RightMode) {
    if mode == RightMode::Closed {
        return;
    }
    SCREEN.with(|slot| open_view_slot(slot, mode));
}

/// Cycle tabs on the focused action column.
pub fn cycle_tab(forward: bool) -> RightMode {
    SCREEN.with(|slot| {
        let Some(old) = slot else { return RightMode::Closed };
        let fi = old.focused_action.min(old.actions.len().saturating_sub(1));
        let n = old.actions[fi].tabs.len();
        if n <= 1 {
            return old.right();
        }
        let a = &mut old.actions[fi];
        a.active = if forward {
            (a.active + 1) % n
        } else {
            (a.active + n - 1) % n
        };
        old.repaint_action();
        old.right()
    })
}

/// Select tab `i` on the focused action column.
pub fn select_tab(i: usize) -> RightMode {
    SCREEN.with(|slot| {
        let Some(old) = slot else { return RightMode::Closed };
        let fi = old.focused_action.min(old.actions.len().saturating_sub(1));
        if i >= old.actions[fi].tabs.len() {
            return old.right();
        }
        old.actions[fi].active = i;
        old.repaint_action();
        old.right()
    })
}

/// The tab index under pixel `(x, y)` on the **focused** action column's bar.
pub fn tab_hit(x: u64, y: u64) -> Option<usize> {
    SCREEN.with(|slot| {
        let sc = slot.as_ref()?;
        tab_hit_in(sc, sc.focused_action, x, y)
    })
}

/// The tab index under `(x, y)` on action column `pane_i`'s bar.
pub fn tab_hit_in_pane(pane_i: usize, x: u64, y: u64) -> Option<usize> {
    SCREEN.with(|slot| {
        let sc = slot.as_ref()?;
        tab_hit_in(sc, pane_i, x, y)
    })
}

fn tab_hit_in(sc: &Screen, pane_i: usize, x: u64, y: u64) -> Option<usize> {
    let a = sc.actions.get(pane_i)?;
    if a.tabs.is_empty() || a.pane.w == 0 {
        return None;
    }
    let ty = a.pane.y + BORDER + 4;
    if y < ty || y >= ty + sc.ch() {
        return None;
    }
    sc.tab_layout_for(pane_i)
        .into_iter()
        .position(|(_, tx, w)| x >= tx && x < tx + w)
}

/// Where a dragged tab dropped at `(x, y)` should be inserted in column
/// `pane_i`: before the tab under the cursor if the drop landed on the tab bar,
/// otherwise at the end (a drop anywhere in the body appends).
pub fn drop_index_in_pane(pane_i: usize, x: u64, y: u64) -> usize {
    SCREEN.with(|slot| {
        let Some(sc) = slot.as_ref() else { return 0 };
        let len = sc.actions.get(pane_i).map(|a| a.tabs.len()).unwrap_or(0);
        tab_hit_in(sc, pane_i, x, y).unwrap_or(len)
    })
}

/// Which action column contains `(x,y)`, if any.
pub fn action_pane_at(x: u64, y: u64) -> Option<usize> {
    SCREEN.with(|slot| {
        let sc = slot.as_ref()?;
        for (i, a) in sc.actions.iter().enumerate() {
            if a.pane.w == 0 {
                continue;
            }
            if x >= a.pane.x
                && x < a.pane.x + a.pane.w
                && y >= a.pane.y
                && y < a.pane.y + a.pane.h
            {
                return Some(i);
            }
        }
        None
    })
}

/// Focus action column `i` (for click / drop).
pub fn focus_action_column(i: usize) {
    SCREEN.with(|slot| {
        let Some(sc) = slot else { return };
        if i >= sc.actions.len() || !sc.column_visible(i) {
            return;
        }
        let was = (sc.focused_action, sc.focus_action);
        sc.focused_action = i;
        sc.focus_action = true;
        if was == (i, true) {
            return; // already the selected pane
        }
        // Repaint here rather than leaving it to `focus_set`: this function has
        // already moved focus onto the action side, so `focus_set(true)` would
        // see no flip and draw nothing — which is exactly why clicking a pane
        // used to change the selection invisibly.
        sc.cursor_restore();
        sc.cur_vis = false;
        sc.draw_frame(&sc.chat, !sc.action_focused());
        // Every visible pane's frame carries the selection state, so the pane
        // losing focus must be redrawn too, not just the one gaining it.
        for j in 0..sc.actions.len() {
            if !sc.column_visible(j) {
                continue;
            }
            let active = sc.focus_action && j == sc.focused_action;
            sc.draw_frame_titled(&sc.actions[j].pane, active, "");
            sc.draw_tab_bar_for(j);
            sc.draw_close_btn_for(j);
        }
        if sc.chat.has_composer {
            sc.draw_composer();
        }
        sc.cursor_overlay();
    });
}

/// Move keyboard focus to the next/previous action pane (grid order), returning
/// the newly focused index. Skips parked panes.
pub fn focus_cycle_column(forward: bool) -> usize {
    // Pick the target inside the lock, then repaint outside it —
    // `focus_action_column` takes `SCREEN` itself and re-entering would deadlock.
    let target = SCREEN.with(|slot| {
        let Some(sc) = slot.as_ref() else { return 0 };
        let visible: Vec<usize> =
            (0..sc.actions.len()).filter(|&i| sc.column_visible(i)).collect();
        if visible.len() < 2 {
            return sc.focused_action;
        }
        let at = visible.iter().position(|&i| i == sc.focused_action).unwrap_or(0);
        let next = if forward {
            (at + 1) % visible.len()
        } else {
            (at + visible.len() - 1) % visible.len()
        };
        visible[next]
    });
    focus_action_column(target);
    target
}

/// Pure focus-cycle math — re-export from [`crate::panes_layout`] so call sites
/// can stay in the framebuffer API. Tests live next to the pure function
/// (framebuffer itself is gated out of the test binary).
pub use crate::panes_layout::cycle_focus_target;

/// Cycle keyboard focus across the shell chat, action panes, and in-pane tabs.
/// Ctrl+Tab / Ctrl+Shift+Tab. Returns true if an action pane holds focus after
/// the move.
///
/// Order (forward): chat → pane0/tab0 → pane0/tab1 → … → pane1/tab0 → … → chat.
/// Within a focused action column that has several tabs, Ctrl+Tab walks those
/// tabs first; only after the last tab does focus move to the next column (or
/// back to the shell). Parked columns are skipped. No action pane open → shell.
pub fn focus_cycle_all(forward: bool) -> bool {
    // 1) If already on an action column with more tabs in this direction,
    //    advance the tab and stay — keyboard-first tab bar.
    let tab_step = SCREEN.with(|slot| {
        let Some(sc) = slot.as_ref() else {
            return false;
        };
        if !sc.focus_action || !sc.any_column_visible() {
            return false;
        }
        let fi = sc.focused_action.min(sc.actions.len().saturating_sub(1));
        let n = sc.actions.get(fi).map(|a| a.tabs.len()).unwrap_or(0);
        if n <= 1 {
            return false;
        }
        let active = sc.actions[fi].active;
        if forward && active + 1 < n {
            return true; // will cycle_tab below
        }
        if !forward && active > 0 {
            return true;
        }
        false
    });
    if tab_step {
        cycle_tab(forward);
        // Ensure action still holds focus (cycle_tab does not touch it).
        focus_set(true);
        return true;
    }

    // 2) Otherwise walk the chat ↔ action-column ring.
    let (to_action, target) = SCREEN.with(|slot| {
        let Some(sc) = slot.as_ref() else {
            return (false, 0usize);
        };
        let visible: Vec<usize> = (0..sc.actions.len())
            .filter(|&i| sc.column_visible(i))
            .collect();
        let at_action = sc.focus_action && !visible.is_empty();
        crate::panes_layout::cycle_focus_target(
            &visible,
            at_action,
            sc.focused_action,
            forward,
        )
    });
    if to_action {
        // Landing on a column from the shell (or another pane): for reverse
        // walks, start at the last tab so reverse is the true inverse of
        // forward's "exhaust tabs then leave".
        if !forward {
            SCREEN.with(|slot| {
                if let Some(sc) = slot {
                    if target < sc.actions.len() {
                        let n = sc.actions[target].tabs.len();
                        if n > 0 {
                            sc.actions[target].active = n - 1;
                        }
                    }
                }
            });
        } else {
            // Forward into a new column → first tab.
            SCREEN.with(|slot| {
                if let Some(sc) = slot {
                    if target < sc.actions.len() && !sc.actions[target].tabs.is_empty() {
                        sc.actions[target].active = 0;
                    }
                }
            });
        }
        focus_action_column(target);
        true
    } else {
        focus_set(false);
        false
    }
}

/// Move a tab from one action column to another (drag-drop). Pure list surgery
/// on the live screen. Returns false if the move is invalid.
pub fn move_tab_between(from_pane: usize, from_idx: usize, to_pane: usize, to_idx: usize) -> bool {
    SCREEN.with(|slot| {
        let Some(sc) = slot else { return false };
        if from_pane >= sc.actions.len() || to_pane >= sc.actions.len() {
            return false;
        }
        if from_idx >= sc.actions[from_pane].tabs.len() {
            return false;
        }
        let mode = sc.actions[from_pane].tabs.remove(from_idx);
        // Fix active on source.
        if sc.actions[from_pane].active >= sc.actions[from_pane].tabs.len()
            && !sc.actions[from_pane].tabs.is_empty()
        {
            sc.actions[from_pane].active = sc.actions[from_pane].tabs.len() - 1;
        }
        let insert = crate::panes_layout::insert_index(
            from_pane,
            from_idx,
            to_pane,
            sc.actions[to_pane].tabs.len(),
            to_idx,
        );
        sc.actions[to_pane].tabs.insert(insert, mode);
        sc.actions[to_pane].active = insert;
        sc.focused_action = to_pane;
        sc.focus_action = true;
        true
    })
}

/// Highlight action column `target` as the live drop target during a tab drag
/// (accent frame), clearing any previously highlighted column.
///
/// Only repaints the two frames that changed, and only when the target actually
/// moved — a mouse drag fires on every pointer report, so repainting the band
/// each time would flicker the whole action band while dragging.
pub fn highlight_drop_target(target: Option<usize>) {
    SCREEN.with(|slot| {
        let Some(sc) = slot else { return };
        if sc.drop_target == target {
            return;
        }
        let prev = sc.drop_target;
        sc.drop_target = target;
        sc.cursor_restore();
        for i in prev.into_iter().chain(target) {
            if !sc.column_visible(i) {
                continue;
            }
            let active = Some(i) == target || (sc.focus_action && i == sc.focused_action);
            sc.draw_frame_titled(&sc.actions[i].pane, active, "");
            sc.draw_tab_bar_for(i);
            sc.draw_close_btn_for(i);
        }
        sc.cursor_overlay();
    });
}

/// Close the tab of `mode` if open (any column).
pub fn close_tab_mode(mode: RightMode) {
    SCREEN.with(|slot| {
        let Some(old) = slot else { return };
        if let Some((pi, ti)) = old.find_mode(mode) {
            old.focused_action = pi;
            old.actions[pi].active = ti;
            close_active_slot(slot);
        }
    });
}

/// Close the active tab of the focused action column.
fn close_active_slot(slot: &mut Option<Screen>) {
    let Some(old) = slot else { return };
    let fi = old.focused_action.min(old.actions.len().saturating_sub(1));
    if old.actions[fi].tabs.is_empty() {
        return;
    }
    let ai = old.actions[fi].active.min(old.actions[fi].tabs.len() - 1);
    old.actions[fi].tabs.remove(ai);
    if old.actions[fi].active >= old.actions[fi].tabs.len()
        && !old.actions[fi].tabs.is_empty()
    {
        old.actions[fi].active = old.actions[fi].tabs.len() - 1;
    }
    let any = old.any_action_open();
    // Closing a tab invalidates the other panes' interiors either way: the collapse
    // branch rebuilds the screen, and the grid branch repaints frames while leaving
    // each view's own pixels to be redrawn. Marked here so both are covered — the grid
    // case is the one that shipped broken, because `repaint_action` looks like it has
    // already done the work.
    mark_tabs_dirty();
    // A lone action pane collapses the band when its last tab closes (the classic
    // two-pane behaviour); a grid keeps its now-empty pane as a drop target.
    if !any && old.actions.len() == 1 {
        let ns = rebuilt(old, false);
        ns.redraw();
        *slot = Some(ns);
    } else {
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

/// Present a surface plus an optional **HUD** in a reserved pane-space strip.
/// `hud` is newline-separated: line 0 = status (accent), the rest = hints
/// (dim, word-wrapped to the pane). The strip is sized to the wrapped content
/// and rendered with the native console font — crisp and wrapping at any pane
/// size — instead of being baked into the (upscaled) surface buffer. Empty
/// `hud` behaves exactly like [`present_surface`].
pub fn present_surface_hud(id: u32, sw: usize, sh: usize, buf: &[u32], hud: &str) {
    present_surface_hud_ex(id, sw, sh, sw, sh, buf, hud);
}

/// Like [`present_surface_hud`], but the pixel buffer may already be presentation-
/// scaled (`buf_w×buf_h`) while hit-testing still uses `logical_sw×logical_sh`.
pub fn present_surface_hud_ex(
    id: u32,
    logical_sw: usize,
    logical_sh: usize,
    buf_w: usize,
    buf_h: usize,
    buf: &[u32],
    hud: &str,
) {
    if hud.trim().is_empty() {
        present_surface_reserve_ex(id, logical_sw, logical_sh, buf_w, buf_h, buf, 0);
        return;
    }
    // Compute reserve *before* any SCREEN critical section. `draw_surface_hud`
    // already holds SCREEN; calling `surface_hud_height` from inside it would
    // re-enter the non-reentrant spinlock and hang forever (chess open path:
    // host_hud_set → present → draw_surface_hud).
    let reserve = surface_hud_height(hud);
    present_surface_reserve_ex(id, logical_sw, logical_sh, buf_w, buf_h, buf, reserve);
    draw_surface_hud(id, hud, reserve);
}

/// Height (px) a surface HUD needs: one status line + the wrapped hint lines,
/// plus a top hairline and small padding — computed at the current pane width.
///
/// Must **not** be called while already holding `SCREEN` (see
/// [`present_surface_hud`]). Pure layout math is in [`hud_strip_height`].
/// Public so package-UI present can size its pre-scaled text buffer to the
/// same usable pane the compositor will use.
pub fn surface_hud_reserve(hud: &str) -> u64 {
    surface_hud_height(hud)
}

fn surface_hud_height(hud: &str) -> u64 {
    SCREEN.with(|slot| {
        let Some(sc) = slot.as_ref() else { return 0 };
        let cols = (sc.logs().cols.saturating_sub(2)).max(4) as usize; // focused column's width
        hud_strip_height(hud, sc.ch(), cols)
    })
}

/// Pure: pixel height of the reserved HUD strip for `hud` at cell height `ch`
/// and wrap width `cols`. Unit-tested.
fn hud_strip_height(hud: &str, ch: u64, cols: usize) -> u64 {
    let mut lines = 1u64; // status
    lines += wrapped_hint_lines(hud, cols);
    // top hairline + half-cell top/bottom padding.
    (lines * ch) + ch / 2 + 2
}

/// Count how many display lines the HUD's hint text (everything after line 0)
/// wraps to at `cols` columns. Pure-ish (reads only the passed args).
fn wrapped_hint_lines(hud: &str, cols: usize) -> u64 {
    let mut it = hud.split('\n');
    let _status = it.next();
    let hints: alloc::vec::Vec<&str> = it.collect();
    if hints.is_empty() {
        return 0;
    }
    // Wrap on word boundaries; a token longer than cols still takes a line.
    let mut lines = 1u64;
    let mut col = 0usize;
    for hint in &hints {
        for word in hint.split_whitespace() {
            let wlen = word.chars().count();
            let need = if col == 0 { wlen } else { col + 1 + wlen };
            if need > cols && col > 0 {
                lines += 1;
                col = wlen;
            } else {
                col = need;
            }
        }
        // Each explicit hint line after the first forces a new row.
        lines += 1;
        col = 0;
    }
    lines.saturating_sub(1).max(1)
}

/// Render a surface's HUD in the reserved bottom strip of its pane (native
/// font, wrapping). No-op unless that surface tab is active.
///
/// `barh` must be the same value used for `present_surface_reserve`'s
/// `reserve_bottom` — precomputed *outside* this critical section so we never
/// re-enter `SCREEN` (non-reentrant spinlock).
fn draw_surface_hud(id: u32, hud: &str, barh: u64) {
    SCREEN.with(|slot| {
        let Some(sc) = slot else { return };
        let Some(d) = sc.mode_dims(RightMode::Surface(id)) else {
            return;
        };
        sc.cursor_restore();
        sc.cur_vis = false;
        let ch = sc.ch();
        let cw = sc.cw();
        let (px, py) = (d.ix, d.iy);
        let (pw, ph) = (d.iw, d.ih);
        let by = py + ph.saturating_sub(barh);
        let bg = d.bg;
        sc.fill_rect(px, by, pw, barh, bg);
        sc.fill_rect(px, by, pw, 1, sc.theme.accent); // top hairline
        let cols = (pw / cw).saturating_sub(2).max(4) as usize;
        let fit = |s: &str| crate::textsel::fit_width(s, cols);
        let mut lines = hud.split('\n');
        let mut y = by + ch / 3;
        // Status line (accent).
        if let Some(status) = lines.next() {
            sc.draw_str_bg(px + cw, y, &fit(status), sc.theme.accent, bg);
            y += ch;
        }
        // Hint lines (dim), word-wrapped; stop when the strip is full.
        let hud_bottom = py + ph;
        let mut linebuf = String::new();
        let flush = |sc: &mut Screen, y: &mut u64, s: &str| {
            if *y + ch <= hud_bottom {
                sc.draw_str_bg(px + cw, *y, &fit(s), sc.theme.logs_fg, bg);
                *y += ch;
            }
        };
        for hint in lines {
            for word in hint.split_whitespace() {
                let cand = if linebuf.is_empty() { String::from(word) } else { alloc::format!("{linebuf} {word}") };
                if cand.chars().count() > cols && !linebuf.is_empty() {
                    flush(sc, &mut y, &linebuf);
                    linebuf = String::from(word);
                } else {
                    linebuf = cand;
                }
            }
            if !linebuf.is_empty() {
                flush(sc, &mut y, &linebuf);
                linebuf.clear();
            }
        }
        sc.cursor_overlay();
    });
}

/// Choose destination size for presenting `sw×sh` into a `pw×ph` pane.
///
/// * **Upscale** (`free fit ≥ source`): integer pixel scale so each source
///   pixel becomes an `s×s` block — keeps package-UI text/rects crisp.
/// * **Downscale** (source larger than pane): free aspect-fit so video still
///   fills without cropping.
///
/// Pure — unit-tested.
pub fn present_fit(sw: u64, sh: u64, pw: u64, ph: u64) -> (u64, u64) {
    if sw == 0 || sh == 0 || pw == 0 || ph == 0 {
        return (0, 0);
    }
    // Free aspect-fit ("contain").
    let fit_w = pw;
    let fit_h = sh.saturating_mul(pw).saturating_div(sw).max(1);
    let (free_w, free_h) = if fit_h <= ph {
        (fit_w, fit_h)
    } else {
        let fit_h = ph;
        let fit_w = sw.saturating_mul(ph).saturating_div(sh).max(1);
        (fit_w.min(pw), fit_h)
    };
    // Integer upscale when the free fit would grow the image.
    if free_w >= sw && free_h >= sh {
        let s = (pw / sw).min(ph / sh).max(1);
        let iw = sw.saturating_mul(s);
        let ih = sh.saturating_mul(s);
        if iw <= pw && ih <= ph && s >= 1 {
            return (iw, ih);
        }
    }
    (free_w, free_h)
}

/// Like [`present_surface`], but leaves `reserve_bottom` px at the bottom of the
/// pane untouched — the frame is scaled/letterboxed into the region *above* the
/// reserve and the reserved strip is never cleared. The video player uses this
/// to keep its control HUD in a fixed strip that the per-frame blit doesn't
/// repaint (so the HUD updates in place instead of flickering under it).
pub fn present_surface_reserve(id: u32, sw: usize, sh: usize, buf: &[u32], reserve_bottom: u64) {
    // Hit-testing uses the same dimensions as the buffer (logical == pixel).
    present_surface_reserve_ex(id, sw, sh, sw, sh, buf, reserve_bottom);
}

/// Present a **pre-scaled** buffer while remembering `logical_sw×logical_sh`
/// for hit-testing. Package-UI builds a presentation-sized RGB buffer (geometry
/// nearest-upscaled + labels re-rasterized at that scale) but clicks must still
/// map into the agent's 256×192 coordinate space.
pub fn present_surface_reserve_ex(
    id: u32,
    logical_sw: usize,
    logical_sh: usize,
    buf_w: usize,
    buf_h: usize,
    buf: &[u32],
    reserve_bottom: u64,
) {
    // Hit map uses logical size; the frame we blit is `buf_w×buf_h`.
    remember_surf_dim(id, logical_sw, logical_sh);
    remember_surf_reserve(id, reserve_bottom);
    SCREEN.with(|slot| {
        // Open the surface tab only when it is not already among open tabs
        // (first present → focused action column). If the tab exists but that
        // column is not showing it, do **not** steal focus.
        let mode = RightMode::Surface(id);
        let found = slot.as_ref().and_then(|sc| sc.find_mode(mode));
        if found.is_none() {
            open_view_slot(slot, mode);
        }
        let Some(sc) = slot else { return };
        let Some((pi, ti)) = sc.find_mode(mode) else {
            return;
        };
        // Not the active tab of its column → skip FB blit (backing already updated).
        if sc.actions[pi].active != ti {
            return;
        }
        if logical_sw == 0 || logical_sh == 0 || buf_w == 0 || buf_h == 0 {
            return;
        }
        sc.cursor_restore();
        sc.cur_vis = false;
        let (px, py) = (sc.actions[pi].pane.ix, sc.actions[pi].pane.iy);
        let (pw, ph_full) = (
            sc.actions[pi].pane.cols * sc.actions[pi].pane.cw,
            sc.actions[pi].pane.rows * sc.actions[pi].pane.ch,
        );
        // Usable frame height excludes the reserved HUD strip at the bottom.
        let ph = ph_full.saturating_sub(reserve_bottom);
        if pw == 0 || ph == 0 || buf.len() < buf_w * buf_h {
            sc.cursor_overlay();
            return;
        }
        // Destination frame follows the **logical** aspect-fit (matches
        // surface_hit). When the buffer is already that size, the sample loop
        // is 1:1; otherwise nearest-neighbour from buf into the frame.
        let (dw, dh) = present_fit(logical_sw as u64, logical_sh as u64, pw, ph);
        let ox = px + (pw.saturating_sub(dw)) / 2;
        let oy = py + (ph.saturating_sub(dh)) / 2;
        // **No full-pane clear.** Clearing the whole surface with fill_rect
        // (then painting the frame) flashed background on the single-buffered
        // FB for tens of ms every present — visible as a once-per-second
        // (or every-frame) flicker. Only paint letterbox *margins*; the frame
        // blit overwrites the content rectangle in place.
        let bg = sc.actions[pi].pane.bg;
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
            let sy = (dy * buf_h as u64 / dh) as usize;
            let srow = sy * buf_w;
            for dx in 0..dw as usize {
                let sx = (dx as u64 * buf_w as u64 / dw) as usize;
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
            (sc.right() != RightMode::Closed).then(|| (sc.logs().cols * sc.logs().cw, sc.logs().rows * sc.logs().ch))
        })
    })
}

/// Interior pixel dims of the column that owns surface `id`, falling back to the
/// focused column when the surface has no tab yet (first present).
///
/// A surface's content must be laid out for the column it is actually blitted
/// into — using the focused column's width would render a browser page or a
/// package-UI canvas at the wrong scale as soon as its tab lived elsewhere.
pub fn surface_dims_px(id: u32) -> Option<(u64, u64)> {
    SCREEN.with(|slot| {
        let sc = slot.as_ref()?;
        let i = sc
            .find_mode(RightMode::Surface(id))
            .map(|(pi, _)| pi)
            .unwrap_or(sc.focused_action);
        let p = &sc.actions.get(i)?.pane;
        (p.w > 0).then(|| (p.cols * p.cw, p.rows * p.ch))
    })
}

/// The action pane's interior background colour, packed `0x00RRGGBB` to match
/// the pixel buffer [`present_surface`] blits — the image viewer letterboxes
/// with this so the padding around a zoomed/rotated image matches the pane.
pub fn pane_bg() -> Option<u32> {
    SCREEN.with(|slot| {
        slot.as_ref().map(|sc| {
            let (r, g, b) = sc.logs().bg;
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
        let Some(d) = sc.mode_dims(RightMode::Todos) else {
            return;
        };
        sc.cursor_restore();
        sc.cur_vis = false;
        let ch = sc.ch();
        let (px, iy, iw) = (d.ix, d.iy, d.iw);
        let bg = d.bg;
        let rows = d.rows;
        let cols = (iw / sc.cw()).max(1) as usize;
        let mut y = iy;
        let head = if title.is_empty() {
            alloc::format!("{} Todos", crate::icons::fa::LIST_CHECK)
        } else {
            alloc::format!("{} {title}", crate::icons::fa::LIST_CHECK)
        };
        let head_fmt = pad_trunc(&head, cols);
        sc.draw_str_bg(px, y, &head_fmt, sc.theme.accent, bg);
        y += ch + ch / 4;
        if items.is_empty() {
            sc.draw_str_bg(px, y, &pad_trunc("(no todos — agent todo_write)", cols), sc.theme.title_dim, bg);
            sc.cursor_overlay();
            return;
        }
        for it in items {
            use crate::icons::fa;
            let mark = match it.status {
                "done" => fa::SQUARE_CHECK,
                "in_progress" => fa::CHEVRON_RIGHT,
                "cancelled" => fa::BAN,
                _ => fa::SQUARE,
            };
            let row = alloc::format!("{mark} {}: {}", it.id, it.text);
            let fg = match it.status {
                "done" => sc.theme.title_dim,
                "in_progress" => sc.theme.accent,
                _ => sc.theme.logs_fg,
            };
            sc.draw_str_bg(px, y, &pad_trunc(&row, cols), fg, bg);
            y += ch;
            if y + ch > iy + rows * ch {
                break;
            }
        }
        let blank = pad_trunc("", cols);
        while y + ch <= iy + rows * ch {
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
/// `(cur_row, cur_col)`, and a bottom mode line. Soft-wraps long lines so the
/// full buffer is reachable (vim-like; previously clipped mid-line).
///
/// `top` is the first **visual** row (soft-wrap aware). `hl` is optional per-byte
/// syntax colours for logical lines starting at `hl_base` (index 0 = that line;
/// `None` entries fall back to the theme's `editor_fg`).
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
    hl_base: usize,
) {
    SCREEN.with(|slot| {
        let Some(sc) = slot else { return };
        // Paint into the column that holds the editor tab, not the focused one.
        let Some(ei) = sc.mode_column(RightMode::Editor) else {
            return;
        };
        let d = PaneDims::of(&sc.actions[ei].pane);
        sc.cursor_restore();
        sc.cur_vis = false;
        let (px, pw, cw, ch, cols, rows) = (d.x, d.w, d.cw, d.ch, d.cols, d.rows);
        let (ix, iy) = (d.ix, d.iy);
        sc.draw_frame_titled(&sc.actions[ei].pane, true, title);
        // Clear the interior to the editor background — wallpaper-aware so the
        // translucent desktop shows behind the editor too (glyphs blend via
        // `blit_glyph`/`bg_at`).
        sc.paint_surface(ix, iy, cols * cw, rows * ch, sc.theme.editor_bg);
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
        let tw = (cols.saturating_sub(gutter) as usize).max(1);
        let lenses: alloc::vec::Vec<usize> = lines.iter().map(|l| l.chars().count()).collect();
        // Walk visual rows from `top`, painting one soft-wrap segment per screen row.
        for i in 0..text_rows {
            let vis = top + i as usize;
            let (li, seg) = crate::editor_wrap::unvis(&lenses, vis, tw);
            if li >= lines.len() {
                break;
            }
            // Past the end of the buffer's visual extent — stop filling.
            let total_vis = {
                let mut t = 0usize;
                for &len in &lenses {
                    t += crate::editor_wrap::soft_wraps(len, tw);
                }
                t
            };
            if vis >= total_vis {
                break;
            }
            let y = iy + i * ch;
            // Gutter: number only on the first wrap segment of a logical line.
            let mut x = ix;
            if seg == 0 {
                let num = alloc::format!("{:>width$} ", li + 1, width = (gutter - 1) as usize);
                for chr in num.chars() {
                    sc.blit_glyph(x, y, chr, sc.theme.editor_lineno, sc.theme.editor_bg);
                    x += cw;
                }
            } else {
                // Continuation marker gutter (spaces) so wrap segments stay aligned.
                for _ in 0..gutter {
                    sc.blit_glyph(x, y, ' ', sc.theme.editor_lineno, sc.theme.editor_bg);
                    x += cw;
                }
            }
            let start = seg * tw;
            let line = &lines[li];
            let hl_row = li.saturating_sub(hl_base);
            for (off, chr) in line.chars().enumerate().skip(start).take(tw) {
                let col = off;
                let bg = if in_sel(li, col) {
                    sc.theme.editor_sel
                } else {
                    sc.theme.editor_bg
                };
                let fg = hl
                    .and_then(|h| h.get(hl_row))
                    .and_then(|v| v.get(col).copied().flatten())
                    .unwrap_or(sc.theme.editor_fg);
                sc.blit_glyph(x, y, chr, fg, bg);
                x += cw;
            }
        }
        // Reverse-video block cursor on the soft-wrap cell that holds (cur_row, cur_col).
        {
            let cur_vis = crate::editor_wrap::vis_index(&lenses, cur_row, cur_col, tw);
            if cur_vis >= top && (cur_vis - top) < text_rows as usize {
                let scr = (cur_vis - top) as u64;
                let col_in_seg = (cur_col % tw) as u64;
                let col_on_screen = gutter + col_in_seg;
                if col_on_screen < cols {
                    let y = iy + scr * ch;
                    let x = ix + col_on_screen * cw;
                    let chr = lines
                        .get(cur_row)
                        .and_then(|l| l.chars().nth(cur_col))
                        .unwrap_or(' ');
                    let chr = if chr.is_control() { ' ' } else { chr };
                    sc.blit_glyph(x, y, chr, sc.theme.editor_bg, sc.theme.accent);
                }
            }
        }
        // Mode line across the bottom interior row — ellipsize so a long path
        // never paints past the pane edge.
        let sy = iy + text_rows * ch;
        sc.paint_surface(px + BORDER, sy, pw - 2 * BORDER, ch, sc.theme.status_bg);
        let ml = crate::textsel::ellipsize(modeline, cols as usize);
        let mut x = ix;
        for chr in ml.chars() {
            sc.blit_glyph(x, sy, chr, sc.theme.title_active, sc.theme.status_bg);
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
            let any = old.any_action_open();
            let band = crate::panes_layout::action_band_visible(cfg.max_panes, any);
            let mut ns = Screen::build(
                old.addr,
                old.fb_w,
                old.fb_h,
                old.pitch,
                old.bpp_bytes,
                old.r_shift,
                old.g_shift,
                old.b_shift,
                cfg,
                band,
                old.focused_action,
                old.logical_pref,
            );
            ns.status_left = old.status_left.clone();
            ns.status_right = old.status_right.clone();
            carry_tabs(&mut ns, old);
            preserve_interactive(&mut ns, old);
            ns.chat.adopt(&old.chat);
            // Fullscreen can park the chat (action-full): keep focus on action.
            // Chat-full parks the columns — snap keyboard back to the composer.
            if cfg.fullscreen == 1 || !ns.any_action_open() {
                ns.focus_action = false;
            }
            ns.redraw();
            *slot = Some(ns);
            // Frames are painted; interiors are the views' own and must follow.
            mark_tabs_dirty();
        }
    });
}

/// Set total pane count (2..=9, including shell) and relayout.
///
/// The grid is reshaped to the most balanced arrangement holding exactly
/// `n - 1` action panes, with even track weights — a pane-count change is a new
/// layout, so carrying over the old weights would leave the new grid lopsided.
pub fn set_max_panes(n: u8) {
    let n = crate::panes_layout::clamp_max_panes(n as u64);
    let (cols, rows) = crate::panes_layout::grid_for_count(
        crate::panes_layout::action_column_count(n),
    );
    set_grid_weighted(cols, rows, None);
}

/// Set the action grid shape explicitly (`/pane grid <cols> <rows>`), clamped to
/// at most 8 action panes, and sync `max_panes` to the resulting cell count.
pub fn set_grid(cols: usize, rows: usize) -> (usize, usize) {
    let (cols, rows) = crate::panes_layout::clamp_grid(cols, rows);
    set_grid_weighted(cols, rows, None);
    // Report what was actually applied — the band may not fit the request.
    grid_shape()
}

/// Shared tail of the grid setters: clamp the shape to what the band can host,
/// build the spec (even weights, or the given ones), and relayout.
///
/// `max_panes` is **derived** from the shape that survives clamping rather than
/// passed in, so the pane budget and the grid can never disagree.
fn set_grid_weighted(cols: usize, rows: usize, weights: Option<(Vec<u64>, Vec<u64>)>) {
    let cfg = SCREEN.with(|slot| {
        let sc = slot.as_mut()?;
        // Clamp to the pixels available: a shape the band cannot host would draw
        // cells shorter than their own header. Doing it here (not in `build`)
        // keeps `grid_shape()` — used by the status line and by `panes.json` —
        // reporting exactly what is on screen.
        let (bw, bh) = sc.band_capacity();
        let cols = crate::panes_layout::fit_tracks(bw, GAP, cols, crate::panes_layout::MIN_TRACK_PX);
        let rows = crate::panes_layout::fit_tracks(bh, GAP, rows, sc.min_pane_h());
        let (cols, rows) = crate::panes_layout::clamp_grid(cols, rows);
        let mut c = sc.layout.clone();
        c.grid = match weights {
            Some((col_w, row_h)) => {
                crate::panes_layout::GridSpec { cols, rows, col_w, row_h }.sanitized()
            }
            None => crate::panes_layout::GridSpec::even(cols, rows),
        };
        c.max_panes = crate::panes_layout::clamp_max_panes(c.grid.len() as u64 + 1);
        c.fullscreen = 0;
        Some(c)
    });
    if let Some(c) = cfg {
        relayout(&c);
    }
}

impl Screen {
    /// Pixel size the action band would have at the current split, whether or not
    /// it is presently on screen (the grid is sized before the band is opened).
    fn band_capacity(&self) -> (u64, u64) {
        let (_, _, _, bw) = crate::panes_layout::split_band(
            self.content_w,
            OUTER,
            GAP,
            self.layout.chat_pct.clamp(10, 90),
            true,
            self.layout.swap,
        );
        let bh = self.content_h.saturating_sub(2 * OUTER);
        (bw, bh)
    }

    /// Smallest usable pane height: its title header plus one text row of
    /// padded interior. Scale-derived, so it is right at any font size.
    fn min_pane_h(&self) -> u64 {
        let ch = self.ch();
        // Mirrors `Pane::new`'s header_h + the interior's bottom border/padding.
        (BORDER + 4 + ch + 6) + BORDER + PAD + ch
    }
}

/// Set the font scale (`0` = automatic from the desktop height) and relayout.
///
/// The knob that actually makes a high-resolution screen readable: cells are
/// `8*scale` x `16*scale` pixels, so this changes how much fits on screen and how
/// big the text is — unlike a smaller desktop, which only letterboxes. Returns the
/// scale now in effect.
pub fn set_font_scale(scale: u64) -> Option<u64> {
    let n = crate::display::clamp_font_scale(scale);
    let cfg = SCREEN.with(|slot| {
        slot.as_mut().map(|sc| {
            let mut c = sc.layout.clone();
            c.scale = n; // 0 → `pick_scale` recomputes from the desktop height
            c
        })
    })?;
    relayout(&cfg);
    effective_font_scale()
}

/// The font scale currently rendering (never 0 — the resolved value).
pub fn effective_font_scale() -> Option<u64> {
    SCREEN.with(|slot| slot.as_ref().map(|sc| sc.scale))
}

/// Move the OS status bar to a desktop edge and relayout.
///
/// Applies instantly: every pane is laid out inside the leftover content rect, so
/// this is a full relayout (scrollback is preserved by `Pane::adopt`, as with a
/// resolution or font-scale change). Returns the position now in effect.
pub fn set_status_pos(pos: crate::panes_layout::StatusPos) -> Option<crate::panes_layout::StatusPos> {
    let cfg = SCREEN.with(|slot| {
        slot.as_mut().map(|sc| {
            let mut c = sc.layout.clone();
            c.status_pos = pos;
            c
        })
    })?;
    relayout(&cfg);
    status_pos()
}

/// The edge the status bar is currently on.
pub fn status_pos() -> Option<crate::panes_layout::StatusPos> {
    SCREEN.with(|slot| slot.as_ref().map(|sc| sc.layout.status_pos))
}

/// The **pinned** font scale (`0` = automatic), i.e. the setting rather than the
/// resolved value. `None` before the console is up.
///
/// `ui_config::layout_cfg` carries this through so a `/theme` apply can't reset a
/// `/display scale` back to ui.json's value — the same live-value trap that
/// `max_panes` and the pane grid already have to avoid.
pub fn pinned_font_scale() -> Option<u64> {
    SCREEN.with(|slot| slot.as_ref().map(|sc| sc.layout.scale))
}

/// The **physical** framebuffer size the firmware gave us — the panel, whatever
/// the desktop is currently set to.
pub fn physical_size() -> Option<(u32, u32)> {
    SCREEN.with(|slot| slot.as_ref().map(|sc| (sc.fb_w as u32, sc.fb_h as u32)))
}

/// The current **logical** desktop size (what layouts are computed against).
pub fn logical_size() -> Option<(u32, u32)> {
    SCREEN.with(|slot| slot.as_ref().map(|sc| (sc.width as u32, sc.height as u32)))
}

/// Whether the desktop is a letterboxed viewport rather than the whole panel.
pub fn is_letterboxed() -> bool {
    SCREEN
        .with(|slot| {
            slot.as_ref()
                .map(|sc| sc.width != sc.fb_w || sc.height != sc.fb_h)
        })
        .unwrap_or(false)
}

/// Set the logical desktop size, applying immediately. `None` = native (use the
/// whole framebuffer).
///
/// Rebuilds the whole screen: every pane's cell grid depends on the desktop size,
/// so this reflows scrollback exactly as a font-scale change does. Returns the
/// size actually applied (clamped to the framebuffer), or `None` if the console
/// isn't up.
pub fn set_logical_size(want: Option<(u32, u32)>) -> Option<(u32, u32)> {
    let applied = SCREEN.with(|slot| {
        let old = slot.as_mut()?;
        let pref = want.map(|(w, h)| {
            let (w, h) = crate::display::clamp_logical((old.fb_w as u32, old.fb_h as u32), (w, h));
            (w as u64, h as u64)
        });
        if pref == old.logical_pref {
            return Some((old.width as u32, old.height as u32)); // already there
        }
        old.logical_pref = pref;
        // Rebuild at the current split so the action band's open/closed state and
        // every tab survive the resolution change.
        let split = old.any_action_open() || old.actions.len() > 1;
        let mut ns = rebuilt(old, split);
        // The desktop shrank or moved: clear the *whole* panel once, or the old
        // desktop's pixels stay lit outside the new viewport.
        ns.fill_phys(0, 0, ns.fb_w, ns.fb_h, (0, 0, 0));
        ns.redraw();
        let got = (ns.width as u32, ns.height as u32);
        *slot = Some(ns);
        Some(got)
    })?;
    Some(applied)
}

/// The logical desktop sizes selectable on this framebuffer (native first).
pub fn available_modes() -> Vec<(u32, u32)> {
    match physical_size() {
        Some(p) => crate::display::modes_for(p),
        None => Vec::new(),
    }
}

/// The action grid's current shape `(cols, rows)`.
pub fn grid_shape() -> (usize, usize) {
    SCREEN.with(|slot| {
        slot.as_ref()
            .map(|sc| {
                let g = sc.layout.grid.sanitized();
                (g.cols, g.rows)
            })
            .unwrap_or((1, 1))
    })
}

/// The action grid's track weights `(col_w, row_h)` in permille, for persisting.
pub fn grid_weights() -> (Vec<u64>, Vec<u64>) {
    SCREEN.with(|slot| {
        slot.as_ref()
            .map(|sc| {
                let g = sc.layout.grid.sanitized();
                (g.col_w, g.row_h)
            })
            .unwrap_or_else(|| (alloc::vec![1000], alloc::vec![1000]))
    })
}

/// Restore a saved grid (shape + weights) from `panes.json` at boot.
pub fn set_grid_spec(cols: usize, rows: usize, col_w: Vec<u64>, row_h: Vec<u64>) {
    let (cols, rows) = crate::panes_layout::clamp_grid(cols, rows);
    set_grid_weighted(cols, rows, Some((col_w, row_h)));
}

/// Current total pane budget (shell + action columns).
pub fn max_panes() -> u8 {
    SCREEN.with(|slot| {
        slot.as_ref()
            .map(|sc| crate::panes_layout::clamp_max_panes(sc.layout.max_panes as u64))
            .unwrap_or(crate::panes_layout::MAX_PANES_DEFAULT)
    })
}

/// Toggle fullscreen: maximise the focused pane to fill the screen, or restore
/// the split. Returns the new state (0 normal, 1 chat-full, 2 action-full).
pub fn toggle_fullscreen() -> u8 {
    let cfg = SCREEN.with(|slot| {
        slot.as_mut().map(|sc| {
            let action_open = sc.right() != RightMode::Closed;
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
pub fn divider_hit(x: u64, y: u64) -> Option<Divider> {
    SCREEN.with(|slot| {
        let sc = slot.as_ref()?;
        if sc.layout.fullscreen != 0 {
            return None;
        }
        sc.divider_at(x, y)
    })
}

impl Screen {
    /// Which divider (if any) is under `(x, y)`, with a few pixels of grab
    /// tolerance either side of the gap.
    ///
    /// Grid dividers are checked **before** the shell|band one: a grid column
    /// gap can sit within grab tolerance of the band gap on a narrow screen, and
    /// the inner divider is the more specific target.
    fn divider_at(&self, x: u64, y: u64) -> Option<Divider> {
        let grid = self.layout.grid.sanitized();
        let (bx, by, bw, bh) = self.band_rect()?;
        let in_band_y = y + 4 >= by && y <= by + bh + 4;
        let in_band_x = x + 4 >= bx && x <= bx + bw + 4;
        if in_band_x && in_band_y {
            // Vertical dividers between grid columns.
            let cw = crate::panes_layout::track_sizes(bw, GAP, &grid.col_w);
            let cx = crate::panes_layout::track_offsets(bw, GAP, &grid.col_w);
            for i in 0..grid.cols.saturating_sub(1) {
                let gap_l = bx + cx[i] + cw[i];
                if x + 4 >= gap_l && x <= gap_l + GAP + 4 {
                    return Some(Divider::Col(i));
                }
            }
            // Horizontal dividers between grid rows.
            let rh = crate::panes_layout::track_sizes(bh, GAP, &grid.row_h);
            let ry = crate::panes_layout::track_offsets(bh, GAP, &grid.row_h);
            for i in 0..grid.rows.saturating_sub(1) {
                let gap_t = by + ry[i] + rh[i];
                if y + 4 >= gap_t && y <= gap_t + GAP + 4 {
                    return Some(Divider::Row(i));
                }
            }
        }
        // The shell | action-band divider.
        if self.chat.w == 0 || bw == 0 {
            return None; // band collapsed, or a pane parked by fullscreen
        }
        let a = &self.chat;
        let gap_l = a.x.min(bx) + if a.x < bx { a.w } else { bw };
        let gap_r = gap_l + GAP;
        if y >= a.y && y < a.y + a.h && x + 4 >= gap_l && x <= gap_r + 4 {
            return Some(Divider::Band);
        }
        None
    }

    /// The action band's bounding rectangle `(x, y, w, h)` — the union of every
    /// unparked grid cell. `None` when the whole band is parked.
    fn band_rect(&self) -> Option<(u64, u64, u64, u64)> {
        let mut r: Option<(u64, u64, u64, u64)> = None;
        for a in &self.actions {
            if a.pane.w == 0 {
                continue;
            }
            let (x0, y0, x1, y1) = (a.pane.x, a.pane.y, a.pane.x + a.pane.w, a.pane.y + a.pane.h);
            r = Some(match r {
                None => (x0, y0, x1 - x0, y1 - y0),
                Some((rx, ry, rw, rh)) => {
                    let (nx, ny) = (rx.min(x0), ry.min(y0));
                    let (ex, ey) = ((rx + rw).max(x1), (ry + rh).max(y1));
                    (nx, ny, ex - nx, ey - ny)
                }
            });
        }
        r
    }
}

/// The kind of divider under the pointer (see [`divider_hit`]).
pub use crate::panes_layout::Divider;

/// Drag `which` divider to pixel `(x, y)` and relayout.
///
/// [`Divider::Band`] moves `chat_pct`; a grid divider re-splits only the two
/// tracks it separates, so panes elsewhere in the band keep their exact sizes.
pub fn drag_divider(which: Divider, x: u64, y: u64) {
    let cfg = SCREEN.with(|slot| {
        let sc = slot.as_mut()?;
        let mut c = sc.layout.clone();
        c.fullscreen = 0;
        match which {
            Divider::Band => {
                // Inverse of `split_band`, so it must be handed the same span and a
                // content-relative x — otherwise a left-edge status bar offsets
                // every drag by the bar's width.
                c.chat_pct = crate::panes_layout::band_divider_pct(
                    sc.content_w,
                    OUTER,
                    GAP,
                    c.swap,
                    x.saturating_sub(sc.content_x),
                );
            }
            Divider::Col(i) => {
                let (bx, _, bw, _) = sc.band_rect()?;
                let mut g = c.grid.sanitized();
                if !crate::panes_layout::resize_tracks(
                    &mut g.col_w,
                    i,
                    bw,
                    GAP,
                    x.saturating_sub(bx),
                ) {
                    return None;
                }
                c.grid = g;
            }
            Divider::Row(i) => {
                let (_, by, _, bh) = sc.band_rect()?;
                let mut g = c.grid.sanitized();
                if !crate::panes_layout::resize_tracks(
                    &mut g.row_h,
                    i,
                    bh,
                    GAP,
                    y.saturating_sub(by),
                ) {
                    return None;
                }
                c.grid = g;
            }
        }
        Some(c)
    });
    if let Some(c) = cfg {
        relayout(&c);
    }
}

/// Scroll a pane's view by `delta` lines (`+` = back in time, `-` = toward
/// live); `action` picks the **focused** action pane's ktrace, else chat. Snaps
/// caret handling automatically: the caret only draws on a live view.
pub fn scroll_view(action: bool, delta: i64) {
    let target = if action {
        ScrollTarget::Action(focused_action_index())
    } else {
        ScrollTarget::Chat
    };
    scroll_target(target, delta);
}

/// Which view a scroll applies to.
#[derive(Clone, Copy)]
pub enum ScrollTarget {
    Chat,
    /// Action pane by grid index — the pane under the mouse pointer, which with a
    /// grid need not be the focused one.
    Action(usize),
}

/// Scroll the ktrace view of a specific action pane (mouse wheel over it).
pub fn scroll_action_pane(i: usize, delta: i64) {
    scroll_target(ScrollTarget::Action(i), delta);
}

fn scroll_target(target: ScrollTarget, delta: i64) {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            let action = matches!(target, ScrollTarget::Action(_));
            let idx = match target {
                ScrollTarget::Chat => 0,
                ScrollTarget::Action(i) => {
                    // Only a ktrace view has scrollback to move.
                    if sc.actions.get(i).map(|a| a.right()) != Some(RightMode::Ktrace) {
                        return;
                    }
                    i
                }
            };
            sc.cursor_restore();
            sc.cur_vis = false;
            let p = if action { &mut sc.actions[idx].pane } else { &mut sc.chat };
            let max = p.hist.len();
            let v = (p.view as i64 + delta).clamp(0, max as i64) as usize;
            if v != p.view {
                p.view = v;
                let p = if action { &sc.actions[idx].pane } else { &sc.chat };
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
        slot.as_ref().map(|sc| if action { sc.logs().rows } else { sc.chat.rows }).unwrap_or(1)
    }) as i64;
    scroll_view(action, if up { rows - 1 } else { -(rows - 1) });
}

/// Snap a pane back to the live view (offset 0).
pub fn scroll_live(action: bool) {
    scroll_view(action, i64::MIN / 2);
}

/// Toggle keyboard focus between the chat pane and an open action pane.
/// Returns true if the action pane now holds focus. No-op (false) when the
/// action band is collapsed. Works while the editor tab is open — leaving
/// the editor for the shell is intentional (Ctrl+Tab back).
///
/// When focus returns to the chat pane, the bordered composer is repainted
/// immediately (accent border + caret) so the shell is ready for input without
/// waiting for a keystroke to re-sync.
pub fn focus_toggle() -> bool {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            if !sc.any_column_visible() {
                return false;
            }
            sc.focus_action = !sc.focus_action;
            sc.cursor_restore();
            sc.cur_vis = false;
            sc.draw_frame(&sc.chat, !sc.action_focused());
            // Repaint every visible pane's chrome: the frame carries the
            // selection state, so panes *losing* it must be redrawn too.
            for j in 0..sc.actions.len() {
                if !sc.column_visible(j) {
                    continue;
                }
                let active = sc.action_focused() && j == sc.focused_action;
                sc.draw_frame_titled(&sc.actions[j].pane, active, "");
                sc.draw_tab_bar_for(j);
                sc.draw_close_btn_for(j);
            }
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
/// e.g. from a mouse click or Ctrl+Tab. Same constraints as [`focus_toggle`].
///
/// Always refreshes the composer when focusing the chat, even if focus was
/// already on chat — so a click on the shell agent immediately arms the input.
/// Leaving the editor for the shell is allowed (editor tab stays open).
pub fn focus_set(action: bool) {
    let (flips, need_composer) = SCREEN.with(|slot| {
        slot.as_ref()
            .map(|sc| {
                let closed = !sc.any_column_visible();
                // Focusing action with no panes is a no-op; focusing chat always
                // works (and must work while the editor tab is open).
                let flips = if action {
                    !closed && sc.focus_action != action
                } else {
                    sc.focus_action != action
                };
                // Focusing chat: repaint composer even when already focused so
                // the caret/border activate without a first keystroke.
                let need_composer = !action && sc.chat.has_composer;
                (flips, need_composer)
            })
            .unwrap_or((false, false))
    });
    if flips {
        // Direct path when only clearing focus_action (focus_toggle needs a
        // visible band; chat focus while band collapsed still needs composer).
        let can_toggle = SCREEN.with(|slot| {
            slot.as_ref()
                .map(|sc| sc.any_column_visible())
                .unwrap_or(false)
        });
        if can_toggle {
            focus_toggle();
            // focus_toggle already re-armed the composer when leaving action.
            return;
        }
        SCREEN.with(|slot| {
            if let Some(sc) = slot {
                sc.focus_action = action;
            }
        });
    }
    if !action && need_composer {
        // Already on chat (click re-arm) or band collapsed with focus cleared.
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
/// Last presented surface dimensions (logical sw×sh) so hit-testing matches
/// the aspect-fit used by [`present_surface_reserve`] (browser is 640×400,
/// chess/ui is 256×192, etc.).
static LAST_SURF_DIM: crate::mm::Locked<alloc::collections::BTreeMap<u32, (usize, usize)>> =
    crate::mm::Locked::new(alloc::collections::BTreeMap::new());

/// Bottom HUD reserve (px) last used with [`present_surface_reserve`] per surface.
static LAST_SURF_RESERVE: crate::mm::Locked<alloc::collections::BTreeMap<u32, u64>> =
    crate::mm::Locked::new(alloc::collections::BTreeMap::new());

fn remember_surf_dim(id: u32, sw: usize, sh: usize) {
    LAST_SURF_DIM.with(|m| {
        m.insert(id, (sw, sh));
    });
}

fn remember_surf_reserve(id: u32, reserve_bottom: u64) {
    LAST_SURF_RESERVE.with(|m| {
        m.insert(id, reserve_bottom);
    });
}

/// Map a screen click into the active surface's logical coordinates,
/// accounting for letterboxing. Uses the last presented size for that surface
/// id (defaults to 256×192 for Synapse UI boards). `None` if the action pane
/// is not showing a surface or the click is outside the painted frame.
pub fn surface_hit(mx: u64, my: u64) -> Option<(u32, u16, u16)> {
    SCREEN.with(|slot| {
        let sc = slot.as_ref()?;
        // Prefer the action column under the pointer; else the focused column.
        let pi = sc
            .actions
            .iter()
            .position(|a| {
                a.pane.w > 0
                    && mx >= a.pane.x
                    && mx < a.pane.x + a.pane.w
                    && my >= a.pane.y
                    && my < a.pane.y + a.pane.h
            })
            .unwrap_or(sc.focused_action.min(sc.actions.len().saturating_sub(1)));
        let a = sc.actions.get(pi)?;
        let id = match a.right() {
            RightMode::Surface(id) => id,
            _ => return None,
        };
        let (px, py) = (a.pane.ix, a.pane.iy);
        let (pw, ph_full) = (a.pane.cols * a.pane.cw, a.pane.rows * a.pane.ch);
        // Exclude the HUD strip (video / browser) so clicks there are not mapped
        // into content coordinates — same usable height as present_surface_reserve.
        let reserve = LAST_SURF_RESERVE.with(|m| m.get(&id).copied().unwrap_or(0));
        let ph = ph_full.saturating_sub(reserve);
        if pw == 0 || ph == 0 || mx < px || my < py || mx >= px + pw || my >= py + ph {
            return None;
        }
        let (sw, sh) = LAST_SURF_DIM.with(|m| {
            m.get(&id)
                .copied()
                .unwrap_or((256, 192))
        });
        let (sw, sh) = (sw as u64, sh as u64);
        if sw == 0 || sh == 0 {
            return None;
        }
        // Same fit as present_surface_reserve (integer upscale when growing).
        let (dw, dh) = present_fit(sw, sh, pw, ph);
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
            let hit = |p: &Pane| p.w > 0 && x >= p.x && x < p.x + p.w && y >= p.y && y < p.y + p.h;
            // Any visible action column counts as "action", not just the focused
            // one — otherwise a click/scroll over column 2 fell through to chat.
            if (0..sc.actions.len()).any(|i| sc.column_visible(i) && hit(&sc.actions[i].pane)) {
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

// --- expandable folds ---------------------------------------------------
// A tool result is printed truncated with a clickable "▸ N more…" line; the
// hidden remainder is registered against that line's absolute index. A single
// click on it reveals the rest. Additive over the scrollback (no render-loop
// or scroll changes), so selection/scroll are unaffected.

/// The absolute line index (`gi`, same coords as `sel`) the chat cursor is on —
/// i.e. where the next printed line will land. Anchors a fold to its "▸ more…".
pub fn chat_current_gi() -> usize {
    SCREEN.with(|slot| {
        slot.as_ref().map(|sc| sc.chat.hist.len() + sc.chat.row as usize).unwrap_or(0)
    })
}

/// Register a fold: the line at `gi` reveals `hidden` (pre-styled text, may
/// contain ANSI + newlines) when clicked. Bounded so it can't grow unbounded.
pub fn chat_note_fold(gi: usize, hidden: &str) {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.chat.folds.push((gi, hidden.to_string()));
            let n = sc.chat.folds.len();
            if n > 64 {
                sc.chat.folds.drain(0..n - 64);
            }
        }
    });
}

/// The absolute line a single-cell click hit (anchor == head, i.e. not a drag),
/// for matching a fold. Call **before** [`chat_sel_end`] (which clears sel).
pub fn chat_click_gi() -> Option<usize> {
    SCREEN.with(|slot| {
        let sc = slot.as_ref()?;
        let (a, b) = sc.chat.sel?;
        (a == b).then_some(a.0)
    })
}

/// Take the hidden text of the fold anchored at line `gi` (removing it), if any.
/// The shell prints the returned text to reveal the collapsed output.
pub fn chat_take_fold(gi: usize) -> Option<String> {
    SCREEN.with(|slot| {
        let sc = slot.as_mut()?;
        let pos = sc.chat.folds.iter().position(|(g, _)| *g == gi)?;
        Some(sc.chat.folds.remove(pos).1)
    })
}

/// Wipe the chat pane's text (grid + scrollback) and repaint it — `/clear`.
pub fn clear_chat() {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.cursor_restore();
            sc.cur_vis = false;
            sc.chat.clear_content();
            sc.paint_surface(sc.chat.ix, sc.chat.iy, sc.chat.cols * sc.chat.cw, sc.chat.rows * sc.chat.ch, sc.chat.bg);
            if sc.chat.has_composer {
                sc.draw_composer();
            } else {
                sc.caret_draw(&sc.chat);
            }
            sc.cursor_overlay();
        }
    });
}

#[cfg(test)]
mod theme_switch_tests {
    use super::dummy_pane;

    #[test_case]
    fn recolor_remaps_only_old_default_fg() {
        let old_fg = (250, 249, 245); // dark theme's cream text
        let new_fg = (38, 35, 31); // light theme's near-black text
        let accent = (204, 120, 92); // an explicit ANSI/syntax colour
        let mut p = dummy_pane();
        // A scrollback line: plain text in old default + one accent-coloured cell.
        p.hist.push_back(alloc::vec![(b'a', old_fg), (b'b', accent), (0, old_fg)]);
        p.grid = alloc::vec![(b'c', old_fg)];
        p.default_fg = new_fg; // as Pane::new set it for the new theme
        p.recolor_default_fg(old_fg, new_fg);
        // Default-coloured cells (incl. the empty one) move to the new fg…
        assert_eq!(p.hist[0][0].1, new_fg);
        assert_eq!(p.hist[0][2].1, new_fg);
        assert_eq!(p.grid[0].1, new_fg);
        // …the explicit accent colour is preserved.
        assert_eq!(p.hist[0][1].1, accent);
    }

    #[test_case]
    fn recolor_is_a_noop_when_fg_unchanged() {
        let fg = (10, 20, 30);
        let mut p = dummy_pane();
        p.hist.push_back(alloc::vec![(b'x', fg)]);
        p.recolor_default_fg(fg, fg);
        assert_eq!(p.hist[0][0].1, fg);
    }
}

#[cfg(test)]
mod hud_tests {
    use super::{hud_strip_height, wrapped_hint_lines};

    #[test_case]
    fn hud_hint_wrapping_counts_lines() {
        // No hint text (status only) → zero hint rows.
        assert_eq!(wrapped_hint_lines("just a status", 40), 0);
        // One short hint fits on one row.
        assert_eq!(wrapped_hint_lines("status\nn new", 40), 1);
        // A long single hint wraps by words to fit narrow columns.
        let hud = "status\narrows move  enter select  esc clear  n new game";
        assert_eq!(wrapped_hint_lines(hud, 80), 1, "fits one row when wide");
        assert!(wrapped_hint_lines(hud, 20) >= 3, "narrow pane wraps to several rows");
        // Two explicit hint lines are at least two rows.
        assert!(wrapped_hint_lines("s\nfirst line\nsecond line", 80) >= 2);
    }

    #[test_case]
    fn hud_strip_height_matches_status_plus_hints() {
        // Chess-shaped HUD: status + one shortcut line. Pure — must stay free
        // of SCREEN so present_surface_hud can compute reserve outside the
        // draw critical section (re-entering SCREEN deadlocks).
        let hud = "Your move (White)\narrows/click move  enter select  esc clear  n new game";
        let ch = 16u64;
        let cols = 40usize;
        let hints = wrapped_hint_lines(hud, cols);
        let expect = (1 + hints) * ch + ch / 2 + 2;
        assert_eq!(hud_strip_height(hud, ch, cols), expect);
        assert!(hud_strip_height(hud, ch, cols) > ch, "strip taller than one cell");
        // Status-only is shorter than status+hints.
        assert!(hud_strip_height(hud, ch, cols) > hud_strip_height("status only", ch, cols));
    }
}

#[cfg(test)]
mod present_fit_tests {
    use super::present_fit;

    #[test_case]
    fn present_fit_integer_upscales_package_ui() {
        // 256×192 into a large pane: free scale is non-integer (~4.6×); we
        // want exact integer blocks so text stays crisp.
        let (dw, dh) = present_fit(256, 192, 1200, 900);
        assert_eq!(dw % 256, 0, "width is integer multiple of source");
        assert_eq!(dh % 192, 0, "height is integer multiple of source");
        assert_eq!(dw / 256, dh / 192, "uniform scale");
        assert!(dw / 256 >= 4, "at least 4× on a 1200×900 pane");
        assert!(dw <= 1200 && dh <= 900);
    }

    #[test_case]
    fn present_fit_one_to_one_when_pane_equals_source() {
        assert_eq!(present_fit(256, 192, 256, 192), (256, 192));
    }

    #[test_case]
    fn present_fit_downscales_large_video() {
        // 1920×1080 into 640×360 — free aspect-fit, not integer upscale.
        let (dw, dh) = present_fit(1920, 1080, 640, 360);
        assert!(dw <= 640 && dh <= 360);
        // Full width or height is used.
        assert!(dw == 640 || dh == 360);
        // Aspect preserved: 16:9.
        assert_eq!(dw * 9, dh * 16);
    }

    #[test_case]
    fn present_fit_zero_dims_is_empty() {
        assert_eq!(present_fit(0, 192, 100, 100), (0, 0));
        assert_eq!(present_fit(256, 192, 0, 100), (0, 0));
    }
}

#[cfg(test)]
mod aa_tests {
    use super::{aa_coverage, AA_SS};

    #[test_case]
    fn disc_coverage_is_full_inside_zero_outside_partial_at_edge() {
        let r = 10i64;
        let r2 = (2 * AA_SS * r).pow(2);
        let inside = |fx: i64, fy: i64| fx * fx + fy * fy <= r2;
        // Dead centre: every sub-sample inside → fully opaque.
        assert_eq!(aa_coverage(0, 0, inside), 255);
        // Well outside the disc → nothing covered.
        assert_eq!(aa_coverage(r + 5, 0, inside), 0);
        // Right on the radius → a fractional coverage (this is the AA).
        let edge = aa_coverage(r, 0, inside);
        assert!(edge > 0 && edge < 255, "edge coverage should be partial, got {edge}");
    }

    #[test_case]
    fn coverage_is_monotonic_across_the_boundary() {
        let r = 20i64;
        let r2 = (2 * AA_SS * r).pow(2);
        let inside = |fx: i64, fy: i64| fx * fx + fy * fy <= r2;
        let a_in = aa_coverage(r - 2, 0, inside);
        let a_edge = aa_coverage(r, 0, inside);
        let a_out = aa_coverage(r + 2, 0, inside);
        assert!(
            a_in >= a_edge && a_edge >= a_out,
            "coverage must not rise moving outward: {a_in} {a_edge} {a_out}"
        );
    }
}
