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
use alloc::string::String;

const CELL_W: u64 = CW as u64;
const CELL_H: u64 = CH as u64;

type Rgb = (u8, u8, u8);

// A dark, Vercel/Claude-Code-flavoured palette.
const SCREEN_BG: Rgb = (8, 8, 10);
const CHAT_BG: Rgb = (14, 14, 17);
const LOGS_BG: Rgb = (10, 10, 13);
const CHAT_FG: Rgb = (232, 233, 238);
const LOGS_FG: Rgb = (120, 132, 144);
const ACCENT: Rgb = (94, 161, 255); // active border / caret / brand
const BORDER_DIM: Rgb = (48, 50, 58); // inactive border
const TITLE_ACTIVE: Rgb = (156, 194, 255);
const TITLE_DIM: Rgb = (120, 126, 140);
const SEP_DIM: Rgb = (32, 34, 42);
const STATUS_BG: Rgb = (20, 21, 27);
const STATUS_FG: Rgb = (140, 148, 162);
// Editor (right pane, `/open`).
const EDITOR_BG: Rgb = (16, 18, 24);
const EDITOR_FG: Rgb = (214, 218, 228);
const EDITOR_LINENO: Rgb = (86, 92, 104);
const EDITOR_SEL: Rgb = (40, 66, 110); // visual-mode selection highlight

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
    fg: Rgb,
    bg: Rgb,
    title: String,
    show_caret: bool,
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
        let iw = (w - 2 * (BORDER + PAD)).max(cw);
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
            bg,
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
}

impl Default for LayoutCfg {
    fn default() -> Self {
        LayoutCfg { chat_pct: CHAT_PCT, scale: 0, swap: false, chat_title: String::from("chat"), logs_title: String::from("ktrace") }
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
        // full-width (only the shell/chat shows until `/ktrace` or `/open`).
        Screen::build(addr, width, height, pitch, bpp_bytes, r_shift, g_shift, b_shift, &LayoutCfg::default(), false)
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
        let chat = Pane::new(chat_x, box_y, chat_bw, box_h, cw, ch, CHAT_FG, CHAT_BG, cfg.chat_title.clone(), true);
        let logs = Pane::new(logs_x, box_y, logs_bw.max(cw), box_h, cw, ch, LOGS_FG, LOGS_BG, cfg.logs_title.clone(), false);
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
            self.fill_rect(p.cell_x(), p.cell_y(), 2 * self.scale, p.ch, ACCENT);
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

    /// Feed one byte to a pane (the per-pane analogue of a terminal write).
    fn pane_putc(s: &Screen, p: &mut Pane, byte: u8) {
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
        let border = if active { ACCENT } else { BORDER_DIM };
        let title_c = if active { TITLE_ACTIVE } else { TITLE_DIM };
        self.rect_outline(p.x, p.y, p.w, p.h, BORDER, border);
        // Title, just inside the top border.
        let ty = p.y + BORDER + 4;
        let tx = p.x + BORDER + PAD;
        let end = self.draw_str(tx, ty, title, title_c, p.bg);
        if active {
            self.draw_str(end, ty, " *", ACCENT, p.bg);
        }
        // Separator under the title.
        let sep_y = ty + self.ch() + 3;
        self.fill_rect(p.x + BORDER, sep_y, p.w - 2 * BORDER, 1, SEP_DIM);
    }

    fn draw_status(&self) {
        let bar_h = self.ch() + 8;
        let sy_top = self.height - bar_h;
        self.fill_rect(0, sy_top, self.width, bar_h, STATUS_BG);
        let ty = sy_top + 4;
        // Left = brand (accent), right = datetime (muted), right-aligned. Both
        // strings come from the UI config templates via `set_status`.
        self.draw_str(OUTER, ty, &self.status_left, ACCENT, STATUS_BG);
        let rlen = self.status_right.len() as u64;
        let rx = self.width.saturating_sub(rlen * self.cw() + OUTER);
        self.draw_str(rx, ty, &self.status_right, STATUS_FG, STATUS_BG);
    }

    /// Paint the chat caret in its current blink state (accent bar, or the pane
    /// background to erase it).
    fn paint_caret(&self) {
        if !self.chat.show_caret {
            return;
        }
        let color = if self.caret_on { ACCENT } else { self.chat.bg };
        self.fill_rect(self.chat.cell_x(), self.chat.cell_y(), 2 * self.scale, self.chat.ch, color);
    }

    /// Full repaint: background, chat pane, the action (right) pane if open,
    /// caret, status bar.
    fn redraw(&self) {
        self.fill_rect(0, 0, self.width, self.height, SCREEN_BG);
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
    screen.redraw();
    SCREEN.with(|slot| *slot = Some(screen));
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
        fg: SCREEN_BG, bg: SCREEN_BG, title: String::new(), show_caret: false,
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
        sc.fill_rect(ix, iy, cols * cw, rows * ch, EDITOR_BG);
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
                sc.blit_glyph(x, y, b, EDITOR_LINENO, EDITOR_BG);
                x += cw;
            }
            // Text, clipped to the pane width; selected cells get a highlight bg.
            let mut c = gutter;
            for (col, &b) in lines[li].as_bytes().iter().enumerate() {
                if c >= cols {
                    break;
                }
                let bg = if in_sel(li, col) { EDITOR_SEL } else { EDITOR_BG };
                sc.blit_glyph(x, y, b, EDITOR_FG, bg);
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
                sc.blit_glyph(x, y, byte, EDITOR_BG, ACCENT); // fg/bg swapped = block cursor
            }
        }
        // Mode line across the bottom interior row.
        let sy = iy + text_rows * ch;
        sc.fill_rect(px + BORDER, sy, pw - 2 * BORDER, ch, STATUS_BG);
        let mut x = ix;
        let mut c = 0u64;
        for b in modeline.bytes() {
            if c >= cols {
                break;
            }
            sc.blit_glyph(x, sy, b, TITLE_ACTIVE, STATUS_BG);
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
