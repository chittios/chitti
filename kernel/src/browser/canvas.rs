//! HTML5 **Canvas 2D** subset — pure pixel buffer + draw ops.
//!
//! Reference: HTML Canvas 2D context; Ladybird `HTMLCanvasElement` /
//! `CanvasRenderingContext2D`. Supports fillRect, strokeRect, clearRect,
//! fillText, strokeText, beginPath/moveTo/lineTo/stroke/fill, arc (approx),
//! set fillStyle/strokeStyle/lineWidth/font size.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

#[derive(Clone, Debug)]
pub struct Canvas2d {
    pub w: usize,
    pub h: usize,
    pub pixels: Vec<u32>,
    pub fill_style: u32,
    pub stroke_style: u32,
    pub line_width: i32,
    pub font_size: f32,
    path: Vec<(f32, f32)>,
    path_start: Option<(f32, f32)>,
}

impl Canvas2d {
    pub fn new(w: i32, h: i32) -> Self {
        let w = w.clamp(1, 2048) as usize;
        let h = h.clamp(1, 2048) as usize;
        Self {
            w,
            h,
            pixels: alloc::vec![0x00ff_ffff; w * h],
            fill_style: 0x000000,
            stroke_style: 0x000000,
            line_width: 1,
            font_size: 14.0,
            path: Vec::new(),
            path_start: None,
        }
    }

    pub fn clear_rect(&mut self, x: i32, y: i32, w: i32, h: i32) {
        fill_rect(&mut self.pixels, self.w, self.h, x, y, w, h, 0x00ff_ffff);
    }

    pub fn fill_rect(&mut self, x: i32, y: i32, w: i32, h: i32) {
        fill_rect(
            &mut self.pixels,
            self.w,
            self.h,
            x,
            y,
            w,
            h,
            self.fill_style,
        );
    }

    pub fn stroke_rect(&mut self, x: i32, y: i32, w: i32, h: i32) {
        let lw = self.line_width.max(1);
        // top bottom left right
        fill_rect(
            &mut self.pixels,
            self.w,
            self.h,
            x,
            y,
            w,
            lw,
            self.stroke_style,
        );
        fill_rect(
            &mut self.pixels,
            self.w,
            self.h,
            x,
            y + h - lw,
            w,
            lw,
            self.stroke_style,
        );
        fill_rect(
            &mut self.pixels,
            self.w,
            self.h,
            x,
            y,
            lw,
            h,
            self.stroke_style,
        );
        fill_rect(
            &mut self.pixels,
            self.w,
            self.h,
            x + w - lw,
            y,
            lw,
            h,
            self.stroke_style,
        );
    }

    pub fn fill_text(&mut self, text: &str, x: i32, y: i32) {
        let _ = crate::font_ttf::blit_run(
            &mut self.pixels,
            self.w,
            self.h,
            x,
            y - self.font_size as i32,
            text,
            self.font_size,
            self.fill_style,
        );
    }

    pub fn begin_path(&mut self) {
        self.path.clear();
        self.path_start = None;
    }

    pub fn move_to(&mut self, x: f32, y: f32) {
        self.path_start = Some((x, y));
        self.path.push((x, y));
    }

    pub fn line_to(&mut self, x: f32, y: f32) {
        if self.path.is_empty() {
            self.path_start = Some((x, y));
        }
        self.path.push((x, y));
    }

    pub fn close_path(&mut self) {
        if let Some(s) = self.path_start {
            self.path.push(s);
        }
    }

    pub fn stroke(&mut self) {
        for win in self.path.windows(2) {
            draw_line(
                &mut self.pixels,
                self.w,
                self.h,
                win[0].0 as i32,
                win[0].1 as i32,
                win[1].0 as i32,
                win[1].1 as i32,
                self.stroke_style,
                self.line_width.max(1),
            );
        }
    }

    pub fn fill(&mut self) {
        // Simple scanline fill of path bbox with fill_style (not full even-odd).
        if self.path.len() < 3 {
            return;
        }
        let xs: Vec<i32> = self.path.iter().map(|p| p.0 as i32).collect();
        let ys: Vec<i32> = self.path.iter().map(|p| p.1 as i32).collect();
        let min_x = *xs.iter().min().unwrap_or(&0);
        let max_x = *xs.iter().max().unwrap_or(&0);
        let min_y = *ys.iter().min().unwrap_or(&0);
        let max_y = *ys.iter().max().unwrap_or(&0);
        fill_rect(
            &mut self.pixels,
            self.w,
            self.h,
            min_x,
            min_y,
            (max_x - min_x).max(1),
            (max_y - min_y).max(1),
            self.fill_style,
        );
    }

    pub fn arc(&mut self, cx: f32, cy: f32, r: f32, start: f32, end: f32) {
        // Approximate with 24 segments of the arc (ignore full TAU wrap).
        let steps = 24;
        let mut first = true;
        for i in 0..=steps {
            let t = start + (end - start) * (i as f32) / steps as f32;
            // crude sin/cos via polynomials / table-free
            let (s, c) = sin_cos_approx(t);
            let x = cx + r * c;
            let y = cy + r * s;
            if first {
                self.move_to(x, y);
                first = false;
            } else {
                self.line_to(x, y);
            }
        }
    }

    pub fn set_fill_style_css(&mut self, s: &str) {
        if let Some(c) = super::css::parse_color(s) {
            self.fill_style = c;
        }
    }

    pub fn set_stroke_style_css(&mut self, s: &str) {
        if let Some(c) = super::css::parse_color(s) {
            self.stroke_style = c;
        }
    }
}

fn sin_cos_approx(t: f32) -> (f32, f32) {
    // Normalize to [-pi, pi]
    let mut x = t;
    const PI: f32 = 3.14159265;
    const TAU: f32 = 6.2831853;
    while x > PI {
        x -= TAU;
    }
    while x < -PI {
        x += TAU;
    }
    // Taylor
    let x2 = x * x;
    let sin = x * (1.0 - x2 / 6.0 + x2 * x2 / 120.0);
    let cos = 1.0 - x2 / 2.0 + x2 * x2 / 24.0;
    (sin, cos)
}

fn fill_rect(buf: &mut [u32], bw: usize, bh: usize, x: i32, y: i32, w: i32, h: i32, c: u32) {
    for dy in 0..h {
        for dx in 0..w {
            put(buf, bw, bh, x + dx, y + dy, c);
        }
    }
}

fn draw_line(
    buf: &mut [u32],
    bw: usize,
    bh: usize,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    c: u32,
    lw: i32,
) {
    let mut x0 = x0;
    let mut y0 = y0;
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        for ox in -lw / 2..=lw / 2 {
            for oy in -lw / 2..=lw / 2 {
                put(buf, bw, bh, x0 + ox, y0 + oy, c);
            }
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

fn put(buf: &mut [u32], bw: usize, bh: usize, x: i32, y: i32, c: u32) {
    if x < 0 || y < 0 {
        return;
    }
    let (x, y) = (x as usize, y as usize);
    if x < bw && y < bh {
        buf[y * bw + x] = c & 0x00ff_ffff;
    }
}

/// Parse a simple CSS color or leave default.
pub fn parse_style_color(s: &str, default: u32) -> u32 {
    super::css::parse_color(s).unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn fill_rect_changes_pixels() {
        let mut c = Canvas2d::new(32, 32);
        c.fill_style = 0xff0000;
        c.fill_rect(0, 0, 10, 10);
        assert_eq!(c.pixels[0], 0xff0000);
        c.clear_rect(0, 0, 10, 10);
        assert_eq!(c.pixels[0], 0x00ff_ffff);
    }

    #[test_case]
    fn path_stroke_nonzero() {
        let mut c = Canvas2d::new(40, 40);
        c.stroke_style = 0x00ff00;
        c.begin_path();
        c.move_to(0.0, 0.0);
        c.line_to(39.0, 39.0);
        c.stroke();
        assert!(c.pixels.iter().any(|&p| p == 0x00ff00));
    }
}
