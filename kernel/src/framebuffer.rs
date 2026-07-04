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
    /// When true the right pane is owned by the `/open` editor, so `log_print`
    /// (ktrace) stops drawing there until the editor closes.
    editor_active: bool,
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
        Screen::build(addr, width, height, pitch, bpp_bytes, r_shift, g_shift, b_shift, &LayoutCfg::default())
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
    ) -> Screen {
        let scale = if cfg.scale > 0 { cfg.scale } else { pick_scale(height) };
        let cw = CELL_W * scale;
        let ch = CELL_H * scale;
        let status_h = ch + 8;
        let content_h = height.saturating_sub(status_h);
        let box_y = OUTER;
        let box_h = content_h.saturating_sub(2 * OUTER);
        let avail_w = width.saturating_sub(2 * OUTER + GAP);
        let pct = cfg.chat_pct.clamp(10, 90);
        let chat_w = avail_w * pct / 100;
        let logs_w = avail_w - chat_w;
        // Left/right box origins; chat takes the right box when swapped.
        let (chat_x, chat_bw, logs_x, logs_bw) = if cfg.swap {
            (OUTER + logs_w + GAP, chat_w, OUTER, logs_w)
        } else {
            (OUTER, chat_w, OUTER + chat_w + GAP, logs_w)
        };
        let chat = Pane::new(chat_x, box_y, chat_bw, box_h, cw, ch, CHAT_FG, CHAT_BG, cfg.chat_title.clone(), true);
        let logs = Pane::new(logs_x, box_y, logs_bw, box_h, cw, ch, LOGS_FG, LOGS_BG, cfg.logs_title.clone(), false);
        let mut status_left = String::from("Chitti OS v");
        status_left.push_str(crate::VERSION);
        Screen {
            addr, width, height, pitch, bpp_bytes, r_shift, g_shift, b_shift, scale, chat, logs,
            status_left,
            status_right: String::new(),
            caret_on: true,
            caret_last_ms: 0,
            editor_active: false,
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

    /// Full repaint: background, both pane boxes + frames, caret, status bar.
    fn redraw(&self) {
        self.fill_rect(0, 0, self.width, self.height, SCREEN_BG);
        self.fill_rect(self.chat.x, self.chat.y, self.chat.w, self.chat.h, self.chat.bg);
        self.fill_rect(self.logs.x, self.logs.y, self.logs.w, self.logs.h, self.logs.bg);
        self.draw_frame(&self.chat, true);
        self.draw_frame(&self.logs, false);
        self.caret_draw(&self.chat);
        self.draw_status();
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
            let mut chat = core::mem::replace(&mut sc.chat, dummy_pane());
            for &b in s.as_bytes() {
                Screen::pane_putc(sc, &mut chat, b);
            }
            sc.chat = chat;
            sc.caret_on = true; // keep the caret lit right after output
        }
    });
}

/// Render one byte into the chat pane (the shell's keystroke echo / backspace).
pub fn console_put_byte(byte: u8) {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            let mut chat = core::mem::replace(&mut sc.chat, dummy_pane());
            Screen::pane_putc(sc, &mut chat, byte);
            sc.chat = chat;
            sc.caret_on = true;
        }
    });
}

/// Set the status-bar text (left = brand, right = datetime), then repaint just
/// the bar. The shell calls this every second with the UI-config templates
/// resolved against the clock.
pub fn set_status(left: &str, right: &str) {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.status_left.clear();
            sc.status_left.push_str(left);
            sc.status_right.clear();
            sc.status_right.push_str(right);
            sc.draw_status();
        }
    });
}

/// Advance the caret blink. Called from the shell's idle poll with the current
/// `now_ms()`; toggles the chat caret roughly twice a second.
pub fn blink(now_ms: u64) {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            if now_ms.saturating_sub(sc.caret_last_ms) >= 500 {
                sc.caret_on = !sc.caret_on;
                sc.caret_last_ms = now_ms;
                sc.paint_caret();
            }
        }
    });
}

/// Render `s` into the **logs** pane. Called by `ktrace`, so the trace stream
/// scrolls independently of the chat conversation.
pub fn log_print(s: &str) {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            if sc.editor_active {
                return; // the editor owns the right pane; ktrace still hits serial
            }
            let mut logs = core::mem::replace(&mut sc.logs, dummy_pane());
            for &b in s.as_bytes() {
                Screen::pane_putc(sc, &mut logs, b);
            }
            sc.logs = logs;
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

/// Hand the right pane to the `/open` editor (ktrace stops drawing there).
pub fn editor_enter() {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.editor_active = true;
        }
    });
}

/// Return the right pane to the ktrace log stream and repaint its frame.
pub fn editor_leave() {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
            sc.editor_active = false;
            let p = &sc.logs;
            sc.fill_rect(p.x, p.y, p.w, p.h, p.bg);
            sc.draw_frame(p, false);
        }
    });
}

/// Render the editor into the right pane: title `editor: <file>`, the visible
/// slice of `lines` from `top`, a reverse-video block cursor at
/// `(cur_row, cur_col)`, and a bottom mode line. `gutter` toggles line numbers.
#[allow(clippy::too_many_arguments)]
pub fn editor_render(title: &str, lines: &[alloc::string::String], top: usize, cur_row: usize, cur_col: usize, modeline: &str) {
    SCREEN.with(|slot| {
        let Some(sc) = slot else { return };
        let (px, pw, cw, ch, cols, rows) =
            (sc.logs.x, sc.logs.w, sc.logs.cw, sc.logs.ch, sc.logs.cols, sc.logs.rows);
        let (ix, iy) = (sc.logs.ix, sc.logs.iy);
        sc.draw_frame_titled(&sc.logs, true, title);
        // Clear the interior to the editor background.
        sc.fill_rect(ix, iy, cols * cw, rows * ch, EDITOR_BG);
        let text_rows = rows.saturating_sub(1);
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
            // Text, clipped to the pane width.
            let mut c = gutter;
            for &b in lines[li].as_bytes() {
                if c >= cols {
                    break;
                }
                sc.blit_glyph(x, y, b, EDITOR_FG, EDITOR_BG);
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
    });
}

/// Rebuild the panes from a new [`LayoutCfg`] (split ratio, font scale, pane
/// swap, titles) on the live framebuffer and repaint. Used by `/ui` when the
/// config changes. No-op if the console isn't up.
pub fn relayout(cfg: &LayoutCfg) {
    SCREEN.with(|slot| {
        if let Some(old) = slot {
            let (left, right) = (old.status_left.clone(), old.status_right.clone());
            let mut ns = Screen::build(
                old.addr, old.width, old.height, old.pitch, old.bpp_bytes, old.r_shift, old.g_shift, old.b_shift, cfg,
            );
            ns.status_left = left;
            ns.status_right = right;
            ns.redraw();
            *slot = Some(ns);
        }
    });
}
