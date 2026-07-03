//! Framebuffer text console (`CHITTI_OS_HANDOFF.md` Phase 7 stretch:
//! "framebuffer text UI beyond serial"). An 8x8-bitmap-font terminal on the
//! Limine framebuffer: a character grid with a cursor, newline handling,
//! backspace, and scrolling once the screen fills.
//!
//! It is a global singleton ([`CONSOLE`]) rather than a transient writer,
//! because `serial::Serial` mirrors every byte it writes here (see
//! `serial.rs`): the graphical QEMU window becomes a live terminal showing the
//! whole session -- boot log, phase demos, and the interactive shell -- while
//! the serial port keeps working in parallel. Keyboard input (`arch::keyboard`)
//! plus this output is what makes the framebuffer a real console, not just a
//! log mirror.

use crate::limine_protocol::Framebuffer;
use crate::mm::Locked;
use font8x8::legacy::BASIC_LEGACY;

const GLYPH: u64 = 8;
const FG: (u8, u8, u8) = (0x33, 0xff, 0x66); // phosphor green
const BG: (u8, u8, u8) = (0x00, 0x08, 0x00);

/// Persistent character-grid console over one framebuffer. Holds the raw
/// framebuffer address as a `usize` (so it is trivially `Send`), plus geometry
/// and the text cursor.
pub struct Console {
    addr: usize,
    width: u64,
    height: u64,
    pitch: u64,
    bpp_bytes: u64,
    r_shift: u32,
    g_shift: u32,
    b_shift: u32,
    cols: u64,
    rows: u64,
    col: u64,
    row: u64,
}

impl Console {
    fn from_fb(fb: &Framebuffer) -> Console {
        let bpp_bytes = (fb.bpp as u64).div_ceil(8);
        Console {
            addr: fb.address as usize,
            width: fb.width,
            height: fb.height,
            pitch: fb.pitch,
            bpp_bytes,
            r_shift: fb.red_mask_shift as u32,
            g_shift: fb.green_mask_shift as u32,
            b_shift: fb.blue_mask_shift as u32,
            cols: fb.width / GLYPH,
            rows: fb.height / GLYPH,
            col: 0,
            row: 0,
        }
    }

    fn put_pixel(&self, x: u64, y: u64, color: (u8, u8, u8)) {
        if x >= self.width || y >= self.height {
            return;
        }
        let offset = y * self.pitch + x * self.bpp_bytes;
        let value: u32 =
            ((color.0 as u32) << self.r_shift) | ((color.1 as u32) << self.g_shift) | ((color.2 as u32) << self.b_shift);
        // SAFETY: `offset` is bounds-checked against the Limine-reported
        // geometry; the framebuffer is a valid, kernel-owned MMIO region.
        unsafe {
            let ptr = (self.addr as *mut u8).add(offset as usize);
            for i in 0..self.bpp_bytes {
                ptr.add(i as usize).write_volatile((value >> (i * 8)) as u8);
            }
        }
    }

    fn draw_glyph(&self, byte: u8) {
        let glyph = BASIC_LEGACY[byte as usize];
        let base_x = self.col * GLYPH;
        let base_y = self.row * GLYPH;
        for (dy, row_bits) in glyph.iter().enumerate() {
            for dx in 0..8u64 {
                let on = (row_bits >> dx) & 1 != 0;
                self.put_pixel(base_x + dx, base_y + dy as u64, if on { FG } else { BG });
            }
        }
    }

    /// Scroll the whole screen up by one text row and clear the bottom row.
    fn scroll(&mut self) {
        let row_bytes = (self.pitch * GLYPH) as usize;
        let keep = (self.pitch * (self.height - GLYPH)) as usize;
        // SAFETY: both regions are inside the framebuffer; `copy` handles the
        // overlap (memmove semantics), and `write_bytes` clears the freed row.
        unsafe {
            let base = self.addr as *mut u8;
            core::ptr::copy(base.add(row_bytes), base, keep);
            core::ptr::write_bytes(base.add(keep), 0, row_bytes);
        }
    }

    fn newline(&mut self) {
        self.col = 0;
        self.row += 1;
        if self.row >= self.rows {
            self.scroll();
            self.row = self.rows - 1;
        }
    }

    fn backspace(&mut self) {
        if self.col > 0 {
            self.col -= 1;
        } else if self.row > 0 {
            self.row -= 1;
            self.col = self.cols - 1;
        }
        self.draw_glyph(b' '); // erase the cell without advancing
    }

    fn put_byte(&mut self, byte: u8) {
        match byte {
            b'\n' => self.newline(),
            b'\r' => self.col = 0,
            0x08 | 0x7f => self.backspace(),
            0x20..=0x7e => {
                self.draw_glyph(byte);
                self.col += 1;
                if self.col >= self.cols {
                    self.newline();
                }
            }
            _ => {}
        }
    }

    fn write_str(&mut self, s: &str) {
        for &b in s.as_bytes() {
            self.put_byte(b);
        }
    }
}

static CONSOLE: Locked<Option<Console>> = Locked::new(None);

/// Bring up the framebuffer console on `fb` and clear the screen. Called once
/// at boot, after which `serial` output mirrors here automatically.
pub fn init_console(fb: &Framebuffer) {
    init_console_from(Console::from_fb(fb));
}

/// Initialize the console over a raw linear framebuffer, for arches that don't
/// get one from Limine. The aarch64 `ramfb` driver
/// (`arch::aarch64::ramfb`) calls this with an `XRGB8888` buffer it configured
/// via fw_cfg — same `Console` renderer, different framebuffer source.
pub fn init_console_raw(addr: usize, width: u64, height: u64, pitch: u64) {
    // XRGB8888: little-endian byte order B,G,R,X → shifts red 16 / green 8 / blue 0.
    let console = Console {
        addr,
        width,
        height,
        pitch,
        bpp_bytes: 4,
        r_shift: 16,
        g_shift: 8,
        b_shift: 0,
        cols: width / GLYPH,
        rows: height / GLYPH,
        col: 0,
        row: 0,
    };
    init_console_from(console);
}

fn init_console_from(console: Console) {
    // Clear to the background colour.
    for y in 0..console.height {
        for x in 0..console.width {
            console.put_pixel(x, y, BG);
        }
    }
    CONSOLE.with(|slot| *slot = Some(console));
}

/// Render `s` to the framebuffer console (no-op until `init_console`). Called
/// by `serial::Serial::write_str`, so ordinary `serial_println!`/`ktrace`
/// output appears on screen too.
pub fn console_print(s: &str) {
    CONSOLE.with(|slot| {
        if let Some(c) = slot {
            c.write_str(s);
        }
    });
}

/// Render a single byte (used by the shell to echo keystrokes / draw its
/// backspace), no-op until `init_console`.
pub fn console_put_byte(byte: u8) {
    CONSOLE.with(|slot| {
        if let Some(c) = slot {
            c.put_byte(byte);
        }
    });
}
