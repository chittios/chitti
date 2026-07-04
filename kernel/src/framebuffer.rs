//! Framebuffer compositor (`CHITTI_OS_HANDOFF.md` Phase 7 stretch: "framebuffer
//! text UI beyond serial"). A tmux-style split-pane terminal drawn directly on
//! the framebuffer: two bordered panes side by side -- **chat** (left, the
//! interactive REPL) and **logs** (right, the live ktrace stream) -- an
//! active-pane highlight, and a bottom status bar. Text is rendered with the
//! Geist Mono glyph atlas ([`crate::font_geist`]) alpha-blended per pixel, so
//! the panes show antialiased type rather than a bare bitmap grid.
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

// Layout metrics, in pixels.
const OUTER: u64 = 6; // margin around the whole content region
const GAP: u64 = 8; // between the two panes
const BORDER: u64 = 2; // pane border thickness
const PAD: u64 = 8; // interior padding inside a pane
const CHAT_PCT: u64 = 56; // chat pane width as a % of the content region

#[cfg(target_arch = "x86_64")]
const ARCH: &str = "x86_64";
#[cfg(target_arch = "aarch64")]
const ARCH: &str = "aarch64";
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
const ARCH: &str = "?";
#[cfg(feature = "model-9b")]
const MODEL: &str = "qwen3.5-9b";
#[cfg(not(feature = "model-9b"))]
const MODEL: &str = "qwen3.5-0.8b";

/// One bordered text pane: an outer box plus the interior character grid it
/// scrolls text within. Colours and cursor live here; the pixel plumbing lives
/// on [`Screen`], which owns the framebuffer.
struct Pane {
    // Outer box (border-inclusive), pixels.
    x: u64,
    y: u64,
    w: u64,
    h: u64,
    // Interior text origin (top-left of cell 0,0), pixels.
    ix: u64,
    iy: u64,
    // Interior size, cells.
    cols: u64,
    rows: u64,
    // Cursor, cells.
    col: u64,
    row: u64,
    fg: Rgb,
    bg: Rgb,
    title: &'static str,
    show_caret: bool,
}

impl Pane {
    /// Build a pane inside outer box `(x,y,w,h)`, reserving a title header at
    /// the top and `PAD` interior padding, and computing the cell grid.
    fn new(x: u64, y: u64, w: u64, h: u64, fg: Rgb, bg: Rgb, title: &'static str, show_caret: bool) -> Pane {
        let header_h = BORDER + 4 + CELL_H + 6; // top border, title text, separator gap
        let ix = x + BORDER + PAD;
        let iy = y + header_h;
        let iw = (w - 2 * (BORDER + PAD)).max(CELL_W);
        let ih = (y + h).saturating_sub(iy + BORDER + PAD).max(CELL_H);
        Pane {
            x,
            y,
            w,
            h,
            ix,
            iy,
            cols: (iw / CELL_W).max(1),
            rows: (ih / CELL_H).max(1),
            col: 0,
            row: 0,
            fg,
            bg,
            title,
            show_caret,
        }
    }
    fn caret_x(&self) -> u64 {
        self.ix + self.col * CELL_W
    }
    fn caret_y(&self) -> u64 {
        self.iy + self.row * CELL_H
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
    chat: Pane,
    logs: Pane,
}

impl Screen {
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
        let status_h = CELL_H + 8;
        let content_h = height.saturating_sub(status_h);
        let box_y = OUTER;
        let box_h = content_h.saturating_sub(2 * OUTER);
        let avail_w = width.saturating_sub(2 * OUTER + GAP);
        let chat_w = avail_w * CHAT_PCT / 100;
        let logs_w = avail_w - chat_w;
        let chat = Pane::new(OUTER, box_y, chat_w, box_h, CHAT_FG, CHAT_BG, "chat", true);
        let logs = Pane::new(OUTER + chat_w + GAP, box_y, logs_w, box_h, LOGS_FG, LOGS_BG, "ktrace", false);
        Screen { addr, width, height, pitch, bpp_bytes, r_shift, g_shift, b_shift, chat, logs }
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

    /// Alpha-blend one printable glyph onto `(px,py)` over background `bg`.
    /// Non-printable bytes render as a blank cell (clearing whatever was there).
    fn blit_glyph(&self, px: u64, py: u64, byte: u8, fg: Rgb, bg: Rgb) {
        let idx = if (FIRST..=LAST).contains(&byte) { (byte - FIRST) as usize } else { 0 };
        let g = &GLYPHS[idx];
        for dy in 0..CH {
            for dx in 0..CW {
                let a = g[dy * CW + dx] as u32;
                let mix = |b: u8, f: u8| (((b as u32) * (255 - a) + (f as u32) * a) / 255) as u8;
                let color = if a == 0 { bg } else { (mix(bg.0, fg.0), mix(bg.1, fg.1), mix(bg.2, fg.2)) };
                self.put_pixel(px + dx as u64, py + dy as u64, color);
            }
        }
    }

    /// Render `s` at pixel `(px,py)`, advancing one cell per byte. Returns the x
    /// past the last glyph. Clips at `self.width`. Used for titles + status bar.
    fn draw_str(&self, px: u64, py: u64, s: &str, fg: Rgb, bg: Rgb) -> u64 {
        let mut x = px;
        for &b in s.as_bytes() {
            if x + CELL_W > self.width {
                break;
            }
            self.blit_glyph(x, py, b, fg, bg);
            x += CELL_W;
        }
        x
    }

    // --- pane text -------------------------------------------------------

    /// Scroll a pane's interior up by one text row, clearing the freed row.
    fn scroll_pane(&self, p: &Pane) {
        let x0 = p.ix;
        let w = p.cols * CELL_W;
        let top = p.iy;
        let h = p.rows * CELL_H;
        let step = (self.pitch * CELL_H) as usize;
        let row_bytes = (w * self.bpp_bytes) as usize;
        // SAFETY: every source/destination row lies inside the framebuffer and
        // inside this pane's x-span; source and destination never overlap
        // within a single `copy_nonoverlapping` (they are `CELL_H` rows apart).
        unsafe {
            let base = self.addr as *mut u8;
            for row in 0..(h - CELL_H) {
                let dst = ((top + row) * self.pitch + x0 * self.bpp_bytes) as usize;
                base.add(dst)
                    .copy_from_nonoverlapping(base.add(dst + step), row_bytes);
            }
        }
        self.fill_rect(x0, top + h - CELL_H, w, CELL_H, p.bg);
    }

    fn caret_erase(&self, p: &Pane) {
        if p.show_caret {
            self.fill_rect(p.caret_x(), p.caret_y(), 2, CELL_H, p.bg);
        }
    }
    fn caret_draw(&self, p: &Pane) {
        if p.show_caret {
            self.fill_rect(p.caret_x(), p.caret_y(), 2, CELL_H, ACCENT);
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
                    s.blit_glyph(p.ix + p.col * CELL_W, p.iy + p.row * CELL_H, b' ', p.fg, p.bg);
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
                s.blit_glyph(p.ix + p.col * CELL_W, p.iy + p.row * CELL_H, b' ', p.fg, p.bg);
            }
            0x20..=0x7e => {
                s.blit_glyph(p.ix + p.col * CELL_W, p.iy + p.row * CELL_H, byte, p.fg, p.bg);
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
        let border = if active { ACCENT } else { BORDER_DIM };
        let title_c = if active { TITLE_ACTIVE } else { TITLE_DIM };
        self.rect_outline(p.x, p.y, p.w, p.h, BORDER, border);
        // Title, just inside the top border.
        let ty = p.y + BORDER + 4;
        let tx = p.x + BORDER + PAD;
        let end = self.draw_str(tx, ty, p.title, title_c, p.bg);
        if active {
            // A small "●" is not in ASCII; use a bright marker instead.
            self.draw_str(end, ty, " *", ACCENT, p.bg);
        }
        // Separator under the title.
        let sep_y = ty + CELL_H + 3;
        self.fill_rect(p.x + BORDER, sep_y, p.w - 2 * BORDER, 1, SEP_DIM);
    }

    fn draw_status(&self) {
        let sy_top = self.height - (CELL_H + 8);
        self.fill_rect(0, sy_top, self.width, CELL_H + 8, STATUS_BG);
        let ty = sy_top + 4;
        // Left: brand + pane tabs.
        let mut x = self.draw_str(OUTER, ty, "chitti-os", ACCENT, STATUS_BG);
        x = self.draw_str(x, ty, "   ", STATUS_FG, STATUS_BG);
        x = self.draw_str(x, ty, "chat", TITLE_ACTIVE, STATUS_BG);
        x = self.draw_str(x, ty, " * ", ACCENT, STATUS_BG);
        let _ = self.draw_str(x, ty, "ktrace", STATUS_FG, STATUS_BG);
        // Right: model + arch, right-aligned.
        let right = " * ";
        let text_cells = (MODEL.len() + right.len() + ARCH.len()) as u64;
        let rx = self.width.saturating_sub(text_cells * CELL_W + OUTER);
        let mut x = self.draw_str(rx, ty, MODEL, STATUS_FG, STATUS_BG);
        x = self.draw_str(x, ty, right, ACCENT, STATUS_BG);
        let _ = self.draw_str(x, ty, ARCH, TITLE_ACTIVE, STATUS_BG);
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

/// Bring up the compositor over a raw linear framebuffer, for arches that don't
/// get one from Limine. The aarch64 `ramfb` driver (`arch::aarch64::ramfb`)
/// calls this with an `XRGB8888` buffer it configured via fw_cfg.
pub fn init_console_raw(addr: usize, width: u64, height: u64, pitch: u64) {
    // XRGB8888: little-endian B,G,R,X → red 16 / green 8 / blue 0.
    let s = Screen::layout(addr, width, height, pitch, 4, 16, 8, 0);
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
        }
    });
}

/// Render `s` into the **logs** pane. Called by `ktrace`, so the trace stream
/// scrolls independently of the chat conversation.
pub fn log_print(s: &str) {
    SCREEN.with(|slot| {
        if let Some(sc) = slot {
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
    Pane { x: 0, y: 0, w: 0, h: 0, ix: 0, iy: 0, cols: 1, rows: 1, col: 0, row: 0, fg: SCREEN_BG, bg: SCREEN_BG, title: "", show_caret: false }
}
