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
use alloc::string::{String, ToString};
use alloc::vec::Vec;

const CELL_W: u64 = CW as u64;
const CELL_H: u64 = CH as u64;

type Rgb = (u8, u8, u8);

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
        Pane {
            x,
            y,
            w,
            h,
            ix,
            iy,
            cw,
            ch,
            cols: (iw / cw).max(1),
            rows: (ih / ch).max(1),
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
        }
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
    /// configs): the last `now_ms` seen, and a call counter.
    blink_seen_ms: u64,
    blink_calls: u32,
    /// What the right ("action") pane currently shows. `None` = closed, so the
    /// chat pane is full-width — the default.
    right: RightMode,
    /// The right mode to restore when the editor closes.
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

// Mouse cursor arrow: 0 = transparent, 1 = fill (white), 2 = outline (black).
const CUR_W: u64 = 8;
const CUR_H: u64 = 13;
#[rustfmt::skip]
const CURSOR: [u8; (CUR_W * CUR_H) as usize] = [
    2,0,0,0,0,0,0,0,
    2,2,0,0,0,0,0,0,
    2,1,2,0,0,0,0,0,
    2,1,1,2,0,0,0,0,
    2,1,1,1,2,0,0,0,
    2,1,1,1,1,2,0,0,
    2,1,1,1,1,1,2,0,
    2,1,1,1,1,1,1,2,
    2,1,1,1,2,2,2,2,
    2,1,2,1,1,2,0,0,
    2,2,0,1,1,2,0,0,
    0,0,0,2,1,1,2,0,
    0,0,0,0,2,2,0,0,
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
}

impl Default for LayoutCfg {
    fn default() -> Self {
        LayoutCfg {
            chat_pct: CHAT_PCT,
            scale: 0,
            swap: false,
            chat_title: String::from("chat"),
            logs_title: String::from("ktrace"),
            theme: Theme::default(),
            splash: true,
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
        let (chat_x, chat_bw, logs_x, logs_bw) = if split {
            let avail_w = width.saturating_sub(2 * OUTER + GAP);
            let chat_w = avail_w * pct / 100;
            let logs_w = avail_w - chat_w;
            // chat takes the right box when swapped.
            if cfg.swap {
                (OUTER + logs_w + GAP, chat_w, OUTER, logs_w)
            } else {
                (OUTER, chat_w, OUTER + chat_w + GAP, logs_w)
            }
        } else {
            // Single pane: chat spans the whole content width; logs is offscreen.
            (OUTER, width.saturating_sub(2 * OUTER), width, 0)
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
            right: RightMode::Closed,
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
                    1 => self.put_pixel(self.cur_x + dx, self.cur_y + dy, (240, 240, 245)),
                    2 => self.put_pixel(self.cur_x + dx, self.cur_y + dy, (10, 10, 12)),
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

    /// Scroll a pane's interior up by one text row, clearing the freed row.
    fn scroll_pane(&self, p: &Pane) {
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
                    // Final byte: apply SGR (`m`); ignore other CSI (cursor/erase).
                    if byte == b'm' {
                        p.apply_sgr();
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
        s.caret_erase(p);
        match byte {
            b'\n' => Screen::newline(p, s),
            b'\r' => p.col = 0,
            b'\t' => {
                let next = (p.col / 4 + 1) * 4;
                while p.col < next && p.col < p.cols {
                    s.blit_glyph(p.cell_x(), p.cell_y(), b' ', p.fg, p.bg);
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
                s.blit_glyph(p.cell_x(), p.cell_y(), b' ', p.fg, p.bg);
            }
            0x20..=0x7e => {
                s.blit_glyph(p.cell_x(), p.cell_y(), byte, p.fg, p.bg);
                p.col += 1;
                if p.col >= p.cols {
                    Screen::newline(p, s);
                }
            }
            _ => {}
        }
        s.caret_draw(p);
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
        // Left = brand text (accent), right = datetime (muted), right-aligned.
        self.draw_str(OUTER, ty, &self.status_left, self.theme.accent, self.theme.status_bg);
        // Right = datetime (muted), right-aligned. Both strings come from the UI
        // config templates via `set_status`.
        let rlen = self.status_right.len() as u64;
        let rx = self.width.saturating_sub(rlen * self.cw() + OUTER);
        self.draw_str(rx, ty, &self.status_right, self.theme.status_fg, self.theme.status_bg);
    }

    /// Paint the chat caret in its current blink state (accent bar, or the pane
    /// background to erase it).
    fn paint_caret(&self) {
        if !self.chat.show_caret {
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

    /// Draw the **Synapse-C** brand mark centred at `(cx, cy)` with outer radius
    /// `r`: an open-right "C" arc in `arc_c`, its two round end-caps and the small
    /// synapse node (dot + four stubs) at the opening in `node_c`. Pure integer
    /// math — a ring test plus an angular wedge for the opening — so it scales
    /// from a status-bar glyph to a splash logo. Geometry mirrors the SVG
    /// (endpoints at ±55°, node at 0.74r; see DESIGN.md).
    fn draw_logo(&self, cx: u64, cy: u64, r: u64, arc_c: Rgb, node_c: Rgb) {
        let (cx, cy, r) = (cx as i64, cy as i64, r as i64);
        let t = (r / 3).max(3); // ring thickness (26/78 of r), min 3 so a small mark still reads
        let half = t / 2;
        let (inner, outer) = ((r - half) * (r - half), (r + half) * (r + half));
        let span = r + half + 1;
        for dy in -span..=span {
            for dx in -span..=span {
                let d2 = dx * dx + dy * dy;
                if d2 < inner || d2 > outer {
                    continue;
                }
                // The "C" opening: skip the +x wedge between ±55° (tan55 ≈ 1.428).
                if dx > 0 && dy.abs() * 1000 < dx * 1428 {
                    continue;
                }
                self.put_pixel((cx + dx) as u64, (cy + dy) as u64, arc_c);
            }
        }
        // Round end-caps at ±55° (min 3 so the two dots read at status-bar size).
        let (ex, ey) = (r * 574 / 1000, r * 819 / 1000); // cos55, sin55
        let cap = (r / 5).max(3);
        self.fill_disc(cx + ex, cy - ey, cap, node_c);
        self.fill_disc(cx + ex, cy + ey, cap, node_c);
        // The synapse node + its four stubs are sub-pixel below ~16px radius
        // (they'd render as mud), so draw them only when the mark is large enough
        // — a status-bar glyph is just the C with its two end dots.
        if r >= 16 {
            let (nx, ny) = (cx + r * 744 / 1000, cy);
            let nr = (r / 12).max(1);
            self.fill_disc(nx, ny, nr, node_c);
            let (stub, lw, gap) = ((r / 6).max(2), (t / 6).max(1), nr + 1);
            let put = |x: i64, y: i64, w: i64, h: i64| {
                if x >= 0 && y >= 0 {
                    self.fill_rect(x as u64, y as u64, w as u64, h as u64, node_c);
                }
            };
            put(nx - lw / 2, ny - gap - stub, lw, stub); // up
            put(nx - lw / 2, ny + gap, lw, stub); // down
            put(nx - gap - stub, ny - lw / 2, stub, lw); // left
            put(nx + gap, ny - lw / 2, stub, lw); // right
        }
    }

    /// Paint the boot splash: the brand mark, "CHITTI OS", and a tagline, centred
    /// on the canvas. Shown briefly at boot (see [`show_splash`]).
    fn draw_splash(&self) {
        self.fill_rect(0, 0, self.width, self.height, self.theme.screen_bg);
        let r = (self.height / 7).max(24);
        let cy = self.height * 2 / 5;
        self.draw_logo(self.width / 2, cy, r, self.theme.chat_fg, self.theme.accent);
        let name = "CHITTI OS";
        let nx = self.width / 2 - (name.len() as u64 * self.cw()) / 2;
        self.draw_str(nx, cy + r + r / 2, name, self.theme.accent, self.theme.screen_bg);
        let tag = "an agentic operating system";
        let tx = self.width / 2 - (tag.len() as u64 * self.cw()) / 2;
        self.draw_str(tx, cy + r + r / 2 + self.ch() + 6, tag, self.theme.title_dim, self.theme.screen_bg);
    }

    /// Full repaint: background, chat pane, the action (right) pane if open,
    /// caret, status bar.
    fn redraw(&self) {
        self.fill_rect(0, 0, self.width, self.height, self.theme.screen_bg);
        self.fill_rect(self.chat.x, self.chat.y, self.chat.w, self.chat.h, self.chat.bg);
        // Focus (active highlight) is on chat unless the editor owns the right pane.
        self.draw_frame(&self.chat, self.right != RightMode::Editor);
        if self.right != RightMode::Closed {
            self.fill_rect(self.logs.x, self.logs.y, self.logs.w, self.logs.h, self.logs.bg);
            let title = if self.right == RightMode::Editor { "editor" } else { &self.logs.title };
            self.draw_frame_titled(&self.logs, self.right == RightMode::Editor, title);
            self.draw_close_btn();
        }
        self.caret_draw(&self.chat);
        self.draw_status();
    }

    /// Draw the `[x]` close button at the top-right of the action pane title.
    fn draw_close_btn(&self) {
        let (x, y, _, _) = self.close_btn();
        self.draw_str(x, y, "[x]", (230, 120, 120), self.logs.bg);
    }
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
    fn modal_box(&self, title: &str, rows: u64) -> (u64, u64, u64) {
        let cw = self.cw();
        let ch = self.ch();
        let cols = (self.width / cw).clamp(20, 64) * 2 / 3;
        let bw = cols * cw + 2 * (BORDER + PAD);
        let bh = (rows + 2) * ch + 2 * (BORDER + PAD);
        let bx = (self.width - bw) / 2;
        let by = (self.height - bh) / 2;
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
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.cursor_restore();
            sc.cur_vis = false;
            MODAL_RECTS.with(|m| *m = [(0, 0, 0, 0); 3]);
            let (ix, iy, cols) = sc.modal_box(title, 3);
            let ch = sc.ch();
            // Wrap the message to the box width.
            let mut y = iy;
            for line in wrap(msg, cols as usize) {
                sc.draw_str(ix, y, &line, sc.theme.chat_fg, sc.theme.status_bg);
                y += ch;
            }
            let by = iy + 2 * ch;
            let x2 = sc.modal_button(ix, by, "Yes", focus_yes, 0);
            sc.modal_button(x2, by, "No", !focus_yes, 1);
            sc.cursor_overlay();
        }
    });
}

/// Draw a text-input modal (masked = password dots). `caret_on` blinks the caret.
pub fn draw_input(title: &str, prompt: &str, buf: &str, masked: bool, caret_on: bool) {
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

/// Dismiss any modal and repaint the normal UI.
pub fn modal_dismiss() {
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
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            // If the clock advances, blink on a 500 ms period. If it's frozen
            // (some VirtualBox configs), fall back to a call-count cadence so the
            // caret still blinks.
            let toggle = if now_ms != sc.blink_seen_ms {
                sc.blink_seen_ms = now_ms;
                sc.blink_calls = 0;
                now_ms.saturating_sub(sc.caret_last_ms) >= 500
            } else {
                sc.blink_calls = sc.blink_calls.wrapping_add(1);
                if sc.blink_calls >= 6000 {
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
/// config, and status text.
fn rebuilt(old: &Screen, split: bool, right: RightMode) -> Screen {
    let mut ns = Screen::build(
        old.addr, old.width, old.height, old.pitch, old.bpp_bytes, old.r_shift, old.g_shift, old.b_shift, &old.layout, split,
    );
    ns.status_left = old.status_left.clone();
    ns.status_right = old.status_right.clone();
    ns.right = right;
    ns.right_before_editor = old.right_before_editor;
    ns
}

/// The current action-pane mode.
pub fn right_mode() -> RightMode {
    SCREEN.with(|slot| slot.as_ref().map(|sc| sc.right).unwrap_or(RightMode::Closed))
}

/// Set the action (right) pane mode, relayouting the split and repainting.
pub fn set_right(mode: RightMode) {
    SCREEN.with(|slot| {
        if let Some(old) = slot {
            if old.right == mode {
                return;
            }
            let ns = rebuilt(old, mode != RightMode::Closed, mode);
            ns.redraw();
            *slot = Some(ns);
        }
    });
}

/// Open the ktrace log stream in the action pane.
pub fn open_ktrace() {
    set_right(RightMode::Ktrace);
}

/// Close the action pane (chat becomes full-width).
pub fn close_action() {
    set_right(RightMode::Closed);
}

/// Hand the action pane to the `/open` editor, splitting if needed and
/// remembering the prior mode to restore on close.
pub fn editor_enter() {
    SCREEN.with(|slot| {
        if let Some(old) = slot {
            let before = old.right;
            let mut ns = rebuilt(old, true, RightMode::Editor);
            ns.right_before_editor = before;
            ns.redraw();
            *slot = Some(ns);
        }
    });
}

/// Return the action pane to whatever it showed before the editor (usually
/// closed → chat full-width), repainting.
pub fn editor_leave() {
    SCREEN.with(|slot| {
        if let Some(old) = slot {
            let restore = old.right_before_editor;
            let ns = rebuilt(old, restore != RightMode::Closed, restore);
            ns.redraw();
            *slot = Some(ns);
        }
    });
}

/// Render the editor into the right pane: title `editor: <file>`, the visible
/// slice of `lines` from `top`, a reverse-video block cursor at
/// `(cur_row, cur_col)`, and a bottom mode line. `gutter` toggles line numbers.
#[allow(clippy::too_many_arguments)]
pub fn editor_render(
    title: &str,
    lines: &[alloc::string::String],
    top: usize,
    cur_row: usize,
    cur_col: usize,
    modeline: &str,
    sel: Option<((usize, usize), (usize, usize))>,
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
            // Text, clipped to the pane width; selected cells get a highlight bg.
            let mut c = gutter;
            for (col, &b) in lines[li].as_bytes().iter().enumerate() {
                if c >= cols {
                    break;
                }
                let bg = if in_sel(li, col) { sc.theme.editor_sel } else { sc.theme.editor_bg };
                sc.blit_glyph(x, y, b, sc.theme.editor_fg, bg);
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
            ns.right_before_editor = old.right_before_editor;
            ns.redraw();
            *slot = Some(ns);
        }
    });
}
