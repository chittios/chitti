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
        let ih = (y + h).saturating_sub(iy + BORDER + PAD).max(ch);
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

    /// Carry another pane's text (scrollback + grid + cursor + colour state)
    /// into this freshly-built pane, re-wrapping to the new geometry. Used when
    /// the layout is rebuilt (action pane toggled, `/ui` relayout) so pane
    /// content is never lost by a split change.
    fn adopt(&mut self, old: &Pane) {
        if old.grid.is_empty() || self.grid.is_empty() {
            return;
        }
        let ocols = old.cols as usize;
        let mut lines: VecDeque<Vec<Cell>> = old.hist.clone();
        // Grid rows up to and including the cursor row are content.
        let used = ((old.row + 1).min(old.rows)) as usize;
        for r in 0..used {
            lines.push_back(old.grid[r * ocols..(r + 1) * ocols].to_vec());
        }
        let cur_gi = lines.len() - used + old.row.min(old.rows - 1) as usize;
        let (rows, cols) = (self.rows as usize, self.cols as usize);
        let keep = lines.len().min(rows);
        let start = lines.len() - keep;
        for (r, line) in lines.iter().skip(start).enumerate() {
            for c in 0..cols.min(line.len()) {
                self.grid[r * cols + c] = line[c];
            }
        }
        self.hist = lines.iter().take(start).cloned().collect();
        while self.hist.len() > HIST_MAX {
            self.hist.pop_front();
        }
        self.row = if cur_gi >= start { (cur_gi - start) as u64 } else { 0 };
        self.col = old.col.min(self.cols - 1);
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
    /// Blinking-caret state for the chat pane.
    caret_on: bool,
    caret_last_ms: u64,
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
            // Action pane fills the screen; chat parked offscreen.
            (width, 0, OUTER, full_w)
        } else if cfg.fullscreen == 1 || !split {
            // Chat fills the screen; action parked offscreen.
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
        let chat = Pane::new(chat_x, box_y, chat_bw, box_h, cw, ch, th.chat_fg, th.chat_bg, cfg.chat_title.clone(), true);
        let logs = Pane::new(logs_x, box_y, logs_bw.max(cw), box_h, cw, ch, th.logs_fg, th.logs_bg, cfg.logs_title.clone(), false);
        let mut status_left = String::from("Chitti OS v");
        status_left.push_str(crate::VERSION);
        Screen {
            addr, width, height, pitch, bpp_bytes, r_shift, g_shift, b_shift, scale, chat, logs,
            status_left,
            status_right: String::new(),
            caret_on: true,
            caret_last_ms: 0,
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
            for i in 0..self.bpp_bytes {
                ptr.add(i as usize).write_volatile((value >> (i * 8)) as u8);
            }
        }
    }

    fn fill_rect(&self, x: u64, y: u64, w: u64, h: u64, c: Rgb) {
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
    fn render_view(&self, p: &Pane) {
        self.fill_rect(p.ix, p.iy, p.cols * p.cw, p.rows * p.ch, p.bg);
        let cols = p.cols as usize;
        if p.grid.len() < cols {
            return;
        }
        let sel = p.sel.map(|(a, b)| crate::textsel::normalize(a, b));
        let view = p.view.min(p.hist.len());
        let first = p.hist.len() - view;
        for r in 0..p.rows as usize {
            let gi = first + r;
            let line: &[Cell] = if gi < p.hist.len() {
                &p.hist[gi]
            } else {
                let gr = gi - p.hist.len();
                if gr >= p.rows as usize {
                    break;
                }
                &p.grid[gr * cols..(gr + 1) * cols]
            };
            for (c, &(b, fg)) in line.iter().enumerate().take(cols) {
                let x = p.ix + c as u64 * p.cw;
                let y = p.iy + r as u64 * p.ch;
                let selected = sel.is_some_and(|s| crate::textsel::contains(s, gi, c));
                if selected {
                    self.fill_rect(x, y, p.cw, p.ch, self.theme.editor_sel);
                }
                if (0x21..=0x7e).contains(&b) {
                    self.blit_glyph(x, y, b, fg, if selected { self.theme.editor_sel } else { p.bg });
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

    fn caret_erase(&self, p: &Pane) {
        if p.show_caret {
            self.fill_rect(p.cell_x(), p.cell_y(), 2 * self.scale, p.ch, p.bg);
        }
    }
    fn caret_draw(&self, p: &Pane) {
        if p.show_caret {
            self.fill_rect(p.cell_x(), p.cell_y(), 2 * self.scale, p.ch, self.theme.accent);
        }
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
                                s.caret_draw(p);
                            }
                        }
                        b'C' | b'D' => {
                            let n = p.csi_param().max(1);
                            if live {
                                s.caret_erase(p);
                            }
                            p.col = if byte == b'C' { (p.col + n).min(p.cols - 1) } else { p.col.saturating_sub(n) };
                            if live {
                                s.caret_draw(p);
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
        // Title, just inside the top border.
        let ty = p.y + BORDER + 4;
        let tx = p.x + BORDER + PAD;
        let end = self.draw_str(tx, ty, title, title_c, p.bg);
        if active {
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
        // Left = the brand mark then the brand text (accent). The glyph radius is
        // sized so the ring (extent ≈ 7/6·r) fits within the bar height.
        let lr = (((bar_h / 2).saturating_sub(2)) * 6 / 7).max(5);
        let lhalf = ((lr / 3).max(3)) / 2;
        let lcx = OUTER + lr + lhalf;
        self.draw_logo(lcx, sy_top + bar_h / 2, lr, self.theme.accent, self.theme.chat_fg);
        let text_x = lcx + lr + lhalf + self.cw() / 2;
        self.draw_str(text_x, ty, &self.status_left, self.theme.accent, self.theme.status_bg);
        // Right = datetime (muted), right-aligned. Both strings come from the UI
        // config templates via `set_status`.
        let rlen = self.status_right.len() as u64;
        let rx = self.width.saturating_sub(rlen * self.cw() + OUTER);
        self.draw_str(rx, ty, &self.status_right, self.theme.status_fg, self.theme.status_bg);
    }

    /// Paint the chat caret in its current blink state (accent bar, or the pane
    /// background to erase it).
    fn paint_caret(&self) {
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

    /// Paint the boot splash: the brand mark, "Chitti OS", and a tagline, centred
    /// on the canvas. Shown briefly at boot (see [`show_splash`]).
    fn draw_splash(&self) {
        self.fill_rect(0, 0, self.width, self.height, self.theme.screen_bg);
        let r = (self.height / 7).max(24);
        let cy = self.height * 2 / 5;
        // Ring in terracotta (accent), node in cream (chat_fg) — see the SVG.
        self.draw_logo(self.width / 2, cy, r, self.theme.accent, self.theme.chat_fg);
        let name = "Chitti OS";
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
            RightMode::Ktrace | RightMode::Top | RightMode::Audio | RightMode::Surface(_) => self.focus_action,
        }
    }

    /// Full repaint: background, chat pane (content re-rendered from its grid),
    /// the action (right) pane if open, caret, status bar.
    fn redraw(&self) {
        // Paint only the background *gutters* (margins + the gap between panes),
        // never a full-screen clear — the panes are painted over their own areas
        // below, so their content is never flashed to background. This is what
        // makes opening/closing the action pane not flicker the whole screen.
        self.paint_gutters();
        // Drop shadows sit in the gutters (right/bottom bands of each pane).
        self.drop_shadow(self.chat.x, self.chat.y, self.chat.w, self.chat.h);
        if self.right != RightMode::Closed {
            self.drop_shadow(self.logs.x, self.logs.y, self.logs.w, self.logs.h);
        }
        self.fill_rect(self.chat.x, self.chat.y, self.chat.w, self.chat.h, self.chat.bg);
        self.draw_frame(&self.chat, !self.action_focused());
        self.render_view(&self.chat);
        if self.right != RightMode::Closed {
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
        if self.chat.view == 0 {
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
        // Detail (e.g. "512M/6.0G") after the bar. Clamp it to the space left
        // before the pane's right edge (x + w): truncate if somehow longer, and
        // pad to that width so a shrinking value leaves no residue AND the text
        // can never overflow the pane border.
        let detail_x = bx + bw + cw;
        let avail = ((x + w).saturating_sub(detail_x) / cw) as usize;
        let mut d = alloc::string::String::from(detail);
        d.truncate(avail);
        while d.len() < avail {
            d.push(' ');
        }
        self.draw_str(detail_x, y, &d, self.theme.title_dim, bg);
        y + ch + ch / 3
    }

    /// Shorthand: [`draw_str`] with an explicit background (the `/top` panel
    /// draws over the logs-pane background, not the screen background).
    fn draw_str_bg(&self, x: u64, y: u64, s: &str, fg: Rgb, bg: Rgb) -> u64 {
        self.draw_str(x, y, s, fg, bg)
    }
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
}

/// Render the `/top` dashboard (htop-style: per-core CPU bars, memory bars, a
/// stats footer) into the **action pane**. No-op unless the action pane is in
/// [`RightMode::Top`]. The shell's idle tick calls this ~1 Hz with a fresh
/// snapshot, so it updates live while the chat pane stays interactive.
pub fn draw_top(v: &TopView) {
    SCREEN.with(|slot| {
        let Some(sc) = slot else { return };
        if sc.right != RightMode::Top {
            return;
        }
        sc.cursor_restore();
        sc.cur_vis = false;
        let ch = sc.ch();
        let (px, iy, iw) = (sc.logs.ix, sc.logs.iy, sc.logs.cols * sc.logs.cw);
        // NO full-interior clear here: that blank-then-repaint is what made the
        // 1 Hz refresh flicker on the single-buffered framebuffer. The pane is
        // cleared once when the tab opens (by `redraw`); each refresh below
        // overwrites every element in place (bars self-fill, text is padded to a
        // fixed width so shrinking values leave no residue), so nothing is ever
        // blanked mid-frame.
        let x = px;
        let mut y = iy;
        let bg = sc.logs.bg;
        let mib = 1024 * 1024;
        let gib = 1024 * mib;
        let fmt = |b: u64| -> String {
            if b >= gib {
                alloc::format!("{}.{}G", b / gib, (b % gib) * 10 / gib)
            } else {
                alloc::format!("{}M", b / mib)
            }
        };
        // CPU section — one bar per core (only core 0 runs; the scheduler is
        // cooperative and the APs park, so the rest read 0%).
        sc.draw_str_bg(x, y, "CPU", sc.theme.accent, bg);
        y += ch + ch / 4;
        for (i, &pct) in v.cores.iter().enumerate() {
            let online = (i as u64) < v.cores_online;
            let lab = alloc::format!("core {:<2}", i);
            let detail = if online { alloc::format!("{}%", pct) } else { String::from("--") };
            y = sc.usage_bar_bg(x, y, iw, &lab, if online { pct } else { 0 }, &detail, bg);
        }
        y += ch / 3;
        // Memory: kernel footprint out of physical RAM, then heap used/total.
        sc.draw_str_bg(x, y, "Memory", sc.theme.accent, bg);
        y += ch + ch / 4;
        let ram_pct = if v.ram_total > 0 { v.ram_used * 100 / v.ram_total } else { 0 };
        y = sc.usage_bar_bg(x, y, iw, "RAM    ", ram_pct, &alloc::format!("{}/{}", fmt(v.ram_used), fmt(v.ram_total)), bg);
        let heap_pct = if v.heap_total > 0 { v.heap_used * 100 / v.heap_total } else { 0 };
        y = sc.usage_bar_bg(x, y, iw, "heap   ", heap_pct, &alloc::format!("{}/{}", fmt(v.heap_used), fmt(v.heap_total)), bg);
        y += ch / 2;
        // Footer stats — each padded to a fixed width so a value that shrinks
        // between refreshes (uptime, alloc count) leaves no stale trailing text.
        for s in [
            alloc::format!("cores online : {}  ({})", v.cores_online, v.arch),
            alloc::format!("model loaded : {}", fmt(v.model_bytes)),
            alloc::format!("heap allocs  : {}", v.allocs),
            alloc::format!("uptime {}", v.uptime),
            v.datetime.to_string(),
        ] {
            if y + ch > iy + sc.logs.rows * sc.logs.ch {
                break;
            }
            sc.draw_str_bg(x, y, &alloc::format!("{:<40}", s), sc.theme.logs_fg, bg);
            y += ch;
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
}

/// Paint the audio-player tab: the track name, a scrubber bar at the current
/// position, and mm:ss / mm:ss. No-op unless the audio tab is active. Called
/// from the shell's idle tick while the player runs (the audio keeps playing
/// regardless of which tab is shown — this only draws when it's on top).
pub fn draw_audio(v: &AudioView) {
    SCREEN.with(|slot| {
        let Some(sc) = slot else { return };
        if sc.right != RightMode::Audio {
            return;
        }
        sc.cursor_restore();
        sc.cur_vis = false;
        let ch = sc.ch();
        let (px, iy, iw) = (sc.logs.ix, sc.logs.iy, sc.logs.cols * sc.logs.cw);
        let _ = iw;
        // Repaint in place (no full clear — the pane was cleared once when the
        // tab opened); the only moving parts are the progress bar (self-filling)
        // and the padded time detail, so the ~4 Hz refresh never blanks.
        let bg = sc.logs.bg;
        let mut y = iy + ch;
        let state = if v.paused {
            "|| paused "
        } else if v.playing {
            "> playing "
        } else {
            "= ended   "
        };
        sc.draw_str_bg(px, y, state, sc.theme.accent, bg);
        y += ch + ch / 2;
        sc.draw_str_bg(px, y, v.name, sc.theme.logs_fg, bg);
        y += ch + ch / 2;
        let pct = if v.total_ms > 0 { (v.pos_ms * 100 / v.total_ms).min(100) } else { 0 };
        let mmss = |ms: u64| alloc::format!("{}:{:02}", ms / 60000, ms % 60000 / 1000);
        let detail = alloc::format!("{} / {}", mmss(v.pos_ms), mmss(v.total_ms));
        sc.usage_bar_bg(px, y, iw, "", pct, &detail, bg);
        y += ch * 2;
        sc.draw_str_bg(px, y, &alloc::format!("{} Hz mono", v.rate), sc.theme.title_dim, bg);
        y += ch + ch / 2;
        // Key controls (only when the action pane is focused; switch tabs freely).
        for line in ["space play/pause  0 restart", "<- ->  seek 5s   up/dn  30s", "Ctrl+C stops"] {
            sc.draw_str_bg(px, y, line, sc.theme.title_dim, bg);
            y += ch;
        }
        sc.cursor_overlay();
    });
}

/// Overlay the video player's control/status bar along the bottom of the video
/// surface pane: playback state, mm:ss / mm:ss, frame counter, mute, a scrubber,
/// and the key-shortcut hints. Drawn *after* the frame blit (present_surface
/// clears the pane each present), so it sits on top like a real player's HUD.
/// No-op unless the video surface tab is active.
#[allow(clippy::too_many_arguments)]
pub fn draw_video_status(name: &str, playing: bool, muted: bool, has_audio: bool, frame: usize, frames: usize, pos_ms: u64, total_ms: u64) {
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
        let bg = sc.theme.status_bg;
        // A three-line control bar hugging the pane bottom (status + scrubber +
        // shortcuts), overpainting the lower strip of the frame.
        let barh = ch * 3 + ch;
        let by = py + ph.saturating_sub(barh);
        sc.fill_rect(px, by, pw, barh, bg);
        sc.fill_rect(px, by, pw, 1, sc.theme.accent); // top hairline
        let mmss = |ms: u64| alloc::format!("{}:{:02}", ms / 60000, ms % 60000 / 1000);
        let state = if playing { ">  playing" } else { "|| paused " };
        let vol = if !has_audio { "[no audio]" } else if muted { "[muted]" } else { "[vol]" };
        let mut y = by + ch / 2;
        let line1 = alloc::format!("{}   {}   {} / {}   frame {}/{}   {}", state, name, mmss(pos_ms), mmss(total_ms), frame, frames, vol);
        sc.draw_str_bg(px + cw, y, &line1, sc.theme.accent, bg);
        y += ch + ch / 3;
        // Scrubber: a full-width track with a filled portion for progress.
        let track_x = px + cw;
        let track_w = pw.saturating_sub(2 * cw);
        let filled = if total_ms > 0 { (track_w * pos_ms.min(total_ms) / total_ms).min(track_w) } else { 0 };
        sc.fill_rect(track_x, y + ch / 3, track_w, ch / 4, sc.theme.title_dim);
        sc.fill_rect(track_x, y + ch / 3, filled, ch / 4, sc.theme.accent);
        y += ch + ch / 3;
        let hints = "space play/pause   <-/-> seek 1f   up/dn 10f   0 restart   m mute   Ctrl+C stop";
        sc.draw_str_bg(px + cw, y, hints, sc.theme.title_dim, bg);
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
    if in_rect(x, y, r[0]) {
        ModalHit::Yes
    } else if in_rect(x, y, r[1]) {
        ModalHit::No
    } else if in_rect(x, y, r[2]) {
        ModalHit::Ok
    } else {
        ModalHit::None
    }
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
            sc.cursor_overlay();
        }
    });
}

/// Render one byte into the chat pane (the shell's keystroke echo / backspace).
pub fn console_put_byte(byte: u8) {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
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
        grid: Vec::new(), hist: VecDeque::new(), view: 0, sel: None,
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

/// Rebuild the screen for a new split/right-mode, preserving geometry, layout
/// config, status text, and — via [`Pane::adopt`] — the pane text content.
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
    ns.clock_alive = old.clock_alive;
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
        let (pw, ph) = (sc.logs.cols * sc.logs.cw, sc.logs.rows * sc.logs.ch);
        // Integer scale-to-fit (letterboxed), so the surface keeps its aspect.
        let scale = core::cmp::max(1, core::cmp::min(pw / sw as u64, ph / sh as u64));
        let dw = sw as u64 * scale;
        let dh = sh as u64 * scale;
        let ox = px + (pw.saturating_sub(dw)) / 2;
        let oy = py + (ph.saturating_sub(dh)) / 2;
        sc.fill_rect(px, py, pw, ph, sc.logs.bg); // clear the pane interior
        // Clamp to the interior: a buffer wider than the pane at scale 1 must
        // clip at the pane edge, never paint over the neighbouring pane.
        let sw_vis = (sw as u64).min(pw / scale);
        let sh_vis = (sh as u64).min(ph / scale);
        for sy in 0..sh_vis {
            for sx in 0..sw_vis {
                let c = buf[(sy * sw as u64 + sx) as usize];
                let rgb = (((c >> 16) & 0xff) as u8, ((c >> 8) & 0xff) as u8, (c & 0xff) as u8);
                let bx = ox + sx * scale;
                let by = oy + sy * scale;
                for j in 0..scale {
                    for i in 0..scale {
                        sc.put_pixel(bx + i, by + j, rgb);
                    }
                }
            }
        }
        sc.cursor_overlay();
    });
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
        // Mode line across the bottom interior row.
        let sy = iy + text_rows * ch;
        sc.fill_rect(px + BORDER, sy, pw - 2 * BORDER, ch, sc.theme.status_bg);
        let mut x = ix;
        let mut c = 0u64;
        for b in modeline.bytes() {
            if c >= cols {
                break;
            }
            sc.blit_glyph(x, sy, b, sc.theme.title_active, sc.theme.status_bg);
            x += cw;
            c += 1;
        }
        sc.cursor_overlay();
    });
}

/// Rebuild the panes from a new [`LayoutCfg`] (split ratio, font scale, pane
/// swap, titles) on the live framebuffer and repaint. Used by `/ui` when the
/// config changes. No-op if the console isn't up.
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
            ns.clock_alive = old.clock_alive;
            ns.chat.adopt(&old.chat);
            ns.logs.adopt(&old.logs);
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
                    sc.caret_draw(&sc.chat);
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

/// Toggle keyboard focus between the chat pane and an open ktrace action pane.
/// Returns true if the action pane now holds focus. No-op (false) when the
/// action pane is closed or owned by the editor.
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
pub fn focus_set(action: bool) {
    let flips = SCREEN.with(|slot| {
        slot.as_ref()
            .map(|sc| !matches!(sc.right, RightMode::Closed | RightMode::Editor) && sc.focus_action != action)
            .unwrap_or(false)
    });
    if flips {
        focus_toggle();
    }
}

/// Which pane a click at `(x, y)` landed in: `Some(true)` = action pane,
/// `Some(false)` = chat pane, `None` = neither (status bar / margins).
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

/// Repaint the chat pane after a selection change (sprite-safe).
fn chat_sel_repaint(sc: &mut Screen) {
    sc.cursor_restore();
    sc.cur_vis = false;
    sc.render_view(&sc.chat);
    if sc.chat.view == 0 {
        sc.caret_draw(&sc.chat);
    }
    sc.cursor_overlay();
}

/// Begin a mouse text selection at pixel `(x, y)`; replaces any previous one.
/// No-op (but still clears) outside the chat pane interior.
pub fn chat_sel_begin(x: u64, y: u64) {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            let cell = chat_abs_cell(sc, x, y, false);
            let had = sc.chat.sel.take().is_some();
            sc.chat.sel = cell.map(|c| (c, c));
            if had || cell.is_some() {
                chat_sel_repaint(sc);
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
                sc.chat.sel = Some((anchor, new_head));
                chat_sel_repaint(sc);
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
            sc.chat.sel = None;
            chat_sel_repaint(sc);
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
            if sc.chat.sel.take().is_some() {
                chat_sel_repaint(sc);
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
            sc.caret_draw(&sc.chat);
            sc.cursor_overlay();
        }
    });
}
