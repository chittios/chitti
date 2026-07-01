//! Draws 8x8 bitmap text into a Limine-provided framebuffer. No scrolling —
//! Phase 0 only needs to print a short boot banner.

use crate::limine_protocol::Framebuffer;
use core::fmt;
use font8x8::legacy::BASIC_LEGACY;

const GLYPH_SIZE: u64 = 8;
const FG: (u8, u8, u8) = (0x00, 0xff, 0x00);
const BG: (u8, u8, u8) = (0x00, 0x00, 0x00);

pub struct Writer<'a> {
    fb: &'a Framebuffer,
    col: u64,
    row: u64,
}

impl<'a> Writer<'a> {
    pub fn new(fb: &'a Framebuffer) -> Self {
        Self { fb, col: 0, row: 0 }
    }

    fn bytes_per_pixel(&self) -> u64 {
        (self.fb.bpp as u64).div_ceil(8)
    }

    fn put_pixel(&mut self, x: u64, y: u64, color: (u8, u8, u8)) {
        if x >= self.fb.width || y >= self.fb.height {
            return;
        }
        let bpp_bytes = self.bytes_per_pixel();
        let offset = y * self.fb.pitch + x * bpp_bytes;
        let value: u32 = ((color.0 as u32) << self.fb.red_mask_shift)
            | ((color.1 as u32) << self.fb.green_mask_shift)
            | ((color.2 as u32) << self.fb.blue_mask_shift);
        // SAFETY: `offset` is bounds-checked above against the Limine
        // reported width/height/pitch, and the framebuffer is exclusively
        // owned by the kernel at this point in boot (single-threaded, no
        // other code maps this physical range).
        unsafe {
            let ptr = self.fb.address.add(offset as usize);
            for i in 0..bpp_bytes {
                ptr.add(i as usize).write_volatile((value >> (i * 8)) as u8);
            }
        }
    }

    fn draw_glyph(&mut self, byte: u8) {
        let glyph = BASIC_LEGACY[byte as usize];
        let base_x = self.col * GLYPH_SIZE;
        let base_y = self.row * GLYPH_SIZE;
        for (dy, row_bits) in glyph.iter().enumerate() {
            for dx in 0..8u64 {
                let on = (row_bits >> dx) & 1 != 0;
                self.put_pixel(base_x + dx, base_y + dy as u64, if on { FG } else { BG });
            }
        }
    }

    fn newline(&mut self) {
        self.col = 0;
        self.row += 1;
    }
}

impl<'a> fmt::Write for Writer<'a> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for byte in s.bytes() {
            match byte {
                b'\n' => self.newline(),
                0x20..=0x7e => {
                    self.draw_glyph(byte);
                    self.col += 1;
                    if self.col * GLYPH_SIZE >= self.fb.width {
                        self.newline();
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
}
