//! **Runtime TTF/OTF rasterizer** (fontdue) — Ladybird/LibGfx-style placement:
//! pen on a **baseline**, glyph bitmap offset by `xmin` / `ymin`, advance by
//! `advance_width`. Screen space is Y-down (like HTML/CSS).
//!
//! Default face: embedded `assets/GeistMono-Regular.ttf` (SIL OFL). Swap with
//! [`load_bytes`] for other `.ttf`/`.otf` files.

use crate::mm::Locked;
use alloc::string::String;
use alloc::vec::Vec;
use fontdue::{Font, FontSettings};

static GEIST_TTF: &[u8] = include_bytes!("../../assets/GeistMono-Regular.ttf");

struct Face {
    font: Font,
    name: String,
}

static FACE: Locked<Option<Face>> = Locked::new(None);

fn ensure_default() {
    FACE.with(|slot| {
        if slot.is_none() {
            if let Ok(font) = Font::from_bytes(GEIST_TTF, FontSettings::default()) {
                *slot = Some(Face {
                    font,
                    name: String::from("Geist Mono"),
                });
            }
        }
    });
}

/// Load a TTF/OTF from raw bytes (replaces the active face).
pub fn load_bytes(data: &[u8], name: &str) -> Result<(), &'static str> {
    let font = Font::from_bytes(data, FontSettings::default()).map_err(|_| "font parse failed")?;
    FACE.with(|slot| {
        *slot = Some(Face {
            font,
            name: String::from(name),
        });
    });
    Ok(())
}

pub fn face_name() -> Option<String> {
    ensure_default();
    FACE.with(|s| s.as_ref().map(|f| f.name.clone()))
}

/// Vertical metrics at pixel size `px` (CSS px ≈ fontdue px).
/// Returns `(ascent, descent, line_height)` where ascent ≥ 0, descent ≤ 0
/// (font design space, Y-up), line_height is positive CSS line box height.
pub fn vertical_metrics(px: f32) -> (f32, f32, f32) {
    ensure_default();
    FACE.with(|s| {
        let Some(face) = s.as_ref() else {
            return (px * 0.8, -px * 0.2, px * 1.25);
        };
        if let Some(lm) = face.font.horizontal_line_metrics(px) {
            // fontdue: ascent positive above baseline, descent negative below.
            let lh = lm.ascent - lm.descent + lm.line_gap;
            (lm.ascent, lm.descent, if lh > 1.0 { lh } else { px * 1.25 })
        } else {
            (px * 0.8, -px * 0.2, px * 1.25)
        }
    })
}

pub fn line_height(px: f32) -> f32 {
    vertical_metrics(px).2
}

/// Ascent in screen pixels (distance from line top / baseline offset).
pub fn ascent(px: f32) -> f32 {
    vertical_metrics(px).0
}

pub fn advance(ch: char, px: f32) -> f32 {
    ensure_default();
    FACE.with(|s| {
        let Some(face) = s.as_ref() else {
            return px * 0.55;
        };
        face.font.metrics(ch, px).advance_width
    })
}

pub fn measure(text: &str, px: f32) -> f32 {
    let mut w = 0.0f32;
    for ch in text.chars() {
        w += advance(ch, px);
    }
    w
}

/// Draw `text` with pen origin at **(x, baseline_y)** in Y-down pixel space.
/// Returns the pen x after the last glyph (x + total advance).
pub fn blit_run_baseline(
    buf: &mut [u32],
    stride: usize,
    height: usize,
    mut pen_x: f32,
    baseline_y: f32,
    text: &str,
    px: f32,
    color: u32,
) -> i32 {
    ensure_default();
    FACE.with(|s| {
        let Some(face) = s.as_ref() else {
            let mut x = pen_x as i32;
            for ch in text.chars() {
                let cw = (px * 0.55) as i32;
                for dy in 0..(px as i32) {
                    for dx in 0..cw {
                        put_blend(
                            buf,
                            stride,
                            height,
                            x + dx,
                            baseline_y as i32 - (px as i32) + dy,
                            color,
                            255,
                        );
                    }
                }
                x += cw;
            }
            return x;
        };
        for ch in text.chars() {
            let (m, coverage) = face.font.rasterize(ch, px);
            // Ladybird/LibGfx-style: bitmap left = pen + xmin;
            // bitmap bottom (Y-up) sits at baseline + ymin → screen Y-down:
            //   top = baseline - ymin - height
            let gx = f32_floor(pen_x) + m.xmin;
            let gy = f32_floor(baseline_y) - m.height as i32 - m.ymin;
            let gw = m.width;
            let gh = m.height;
            if gw > 0 && gh > 0 && coverage.len() >= gw * gh {
                for row in 0..gh {
                    for col in 0..gw {
                        let a = coverage[row * gw + col];
                        if a != 0 {
                            put_blend(
                                buf,
                                stride,
                                height,
                                gx + col as i32,
                                gy + row as i32,
                                color,
                                a,
                            );
                        }
                    }
                }
            }
            pen_x += m.advance_width;
        }
        f32_round(pen_x) as i32
    })
}

/// Draw a run whose layout box top-left is `(x, y_top)` (CSS line box).
/// Baseline is placed at `y_top + ascent(px)`.
pub fn blit_run(
    buf: &mut [u32],
    stride: usize,
    height: usize,
    x: i32,
    y_top: i32,
    text: &str,
    px: f32,
    color: u32,
) -> i32 {
    let (asc, _, _) = vertical_metrics(px);
    let baseline = y_top as f32 + asc;
    blit_run_baseline(buf, stride, height, x as f32, baseline, text, px, color)
}

fn f32_floor(v: f32) -> i32 {
    let i = v as i32;
    if v < 0.0 && v != i as f32 {
        i - 1
    } else {
        i
    }
}

fn f32_round(v: f32) -> f32 {
    if v >= 0.0 {
        (v + 0.5) as i32 as f32
    } else {
        (v - 0.5) as i32 as f32
    }
}

fn put_blend(buf: &mut [u32], stride: usize, height: usize, x: i32, y: i32, color: u32, alpha: u8) {
    if x < 0 || y < 0 {
        return;
    }
    let (x, y) = (x as usize, y as usize);
    if x >= stride || y >= height {
        return;
    }
    let i = y * stride + x;
    let dst = buf[i];
    let a = alpha as u32;
    if a >= 250 {
        buf[i] = color & 0x00ff_ffff;
        return;
    }
    let inv = 255 - a;
    let dr = (dst >> 16) & 0xff;
    let dg = (dst >> 8) & 0xff;
    let db = dst & 0xff;
    let sr = (color >> 16) & 0xff;
    let sg = (color >> 8) & 0xff;
    let sb = color & 0xff;
    let r = (sr * a + dr * inv) / 255;
    let g = (sg * a + dg * inv) / 255;
    let b = (sb * a + db * inv) / 255;
    buf[i] = (r << 16) | (g << 8) | b;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn geist_loads_and_measures() {
        ensure_default();
        assert!(face_name().is_some());
        let w = measure("Hello", 16.0);
        assert!(w > 20.0 && w < 120.0, "width={w}");
        let mut buf = alloc::vec![0x00ff_ffffu32; 128 * 48];
        let end = blit_run(&mut buf, 128, 48, 8, 8, "Hello", 16.0, 0x000000);
        assert!(end > 20, "end_x={end}");
        assert!(buf.iter().any(|&p| p != 0x00ff_ffff), "expected ink");
    }

    #[test_case]
    fn advances_are_positive_and_sum() {
        ensure_default();
        let a = advance('A', 20.0);
        let i = advance('i', 20.0);
        assert!(a > 0.0 && i > 0.0);
        // Mono face: similar advances; still both reasonable.
        assert!(a < 40.0 && i < 40.0);
        let w = measure("AA", 20.0);
        assert!((w - 2.0 * a).abs() < 1.0, "w={w} a={a}");
    }
}
