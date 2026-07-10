//! **Runtime TTF/OTF rasterizer** (fontdue) — Ladybird/LibGfx-style placement:
//! pen on a **baseline**, glyph bitmap offset by `xmin` / `ymin`, advance by
//! `advance_width`. Screen space is Y-down (like HTML/CSS).
//!
//! Default face: embedded `assets/GeistMono-Regular.ttf` (SIL OFL). Swap with
//! [`load_bytes`] for other `.ttf`/`.otf` files, or register additional
//! **families** with [`load_family`] and render via [`measure_family`] /
//! [`blit_run_family`] (per-glyph fallback to the global face).

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

/// Loaded web-font families (lowercased name → face), oldest first. Bounded
/// at [`FAMILY_CAP`]: loading one more evicts the **oldest** entry, so pages
/// keep loading fonts and the most recently seen faces win. Lock order:
/// `FAMILIES` is always taken **before** `FACE` (never the reverse).
static FAMILIES: Locked<Vec<(String, Font)>> = Locked::new(Vec::new());

/// Maximum number of registered font families (see [`FAMILIES`]).
const FAMILY_CAP: usize = 8;

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
        let font = s.as_ref().map(|f| &f.font);
        for ch in text.chars() {
            pen_x += match font {
                Some(f) => blit_glyph(buf, stride, height, pen_x, baseline_y, ch, px, color, f),
                None => blit_box(buf, stride, height, pen_x as i32, baseline_y as i32, px, color),
            };
        }
        f32_round(pen_x) as i32
    })
}

/// Rasterize one glyph from `font` with the pen at `(pen_x, baseline_y)` and
/// blend it into `buf`; returns the glyph advance.
#[allow(clippy::too_many_arguments)]
fn blit_glyph(
    buf: &mut [u32],
    stride: usize,
    height: usize,
    pen_x: f32,
    baseline_y: f32,
    ch: char,
    px: f32,
    color: u32,
    font: &Font,
) -> f32 {
    let (m, coverage) = font.rasterize(ch, px);
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
                    put_blend(buf, stride, height, gx + col as i32, gy + row as i32, color, a);
                }
            }
        }
    }
    m.advance_width
}

/// No face loaded at all: draw a crude filled box and return its advance.
fn blit_box(
    buf: &mut [u32],
    stride: usize,
    height: usize,
    x: i32,
    baseline_y: i32,
    px: f32,
    color: u32,
) -> f32 {
    let cw = (px * 0.55) as i32;
    for dy in 0..(px as i32) {
        for dx in 0..cw {
            put_blend(buf, stride, height, x + dx, baseline_y - (px as i32) + dy, color, 255);
        }
    }
    cw as f32
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

// --- font-family registry (web fonts) --------------------------------
//
// The global face above stays the default/fallback; `load_family` registers
// additional faces under a case-insensitive family name (e.g. from a CSS
// `@font-face`). The family-aware variants select a face **per glyph**: the
// named family if it covers the char, else the global fallback — so mixed
// text renders instead of tofu. Generic families ("sans-serif", "serif",
// "monospace", "") are never registered and simply miss the registry, which
// the fallback covers.

/// Lowercased, trimmed family key.
fn norm_family(family: &str) -> String {
    family.trim().to_lowercase()
}

/// Register a TTF/OTF face under `family` (case-insensitive). Re-loading an
/// existing family replaces it in place. The registry holds at most
/// [`FAMILY_CAP`] families; when full, the **oldest** entry is evicted.
pub fn load_family(family: &str, data: &[u8]) -> Result<(), &'static str> {
    let name = norm_family(family);
    if name.is_empty() {
        return Err("empty family name");
    }
    let font = Font::from_bytes(data, FontSettings::default()).map_err(|_| "font parse failed")?;
    FAMILIES.with(move |fams| {
        if let Some(i) = fams.iter().position(|(n, _)| *n == name) {
            fams[i].1 = font;
        } else {
            if fams.len() >= FAMILY_CAP {
                fams.remove(0);
            }
            fams.push((name, font));
        }
    });
    Ok(())
}

/// True if `family` (case-insensitive) is in the registry.
pub fn family_loaded(family: &str) -> bool {
    let name = norm_family(family);
    FAMILIES.with(|fams| fams.iter().any(|(n, _)| *n == name))
}

/// Per-glyph face selection: the family face if it covers `ch`, else the
/// global fallback; if neither covers it, the fallback's notdef.
fn pick_font<'a>(primary: Option<&'a Font>, fallback: Option<&'a Font>, ch: char) -> Option<&'a Font> {
    if let Some(p) = primary {
        if p.lookup_glyph_index(ch) != 0 {
            return Some(p);
        }
    }
    if let Some(f) = fallback {
        if f.lookup_glyph_index(ch) != 0 {
            return Some(f);
        }
    }
    fallback.or(primary)
}

/// [`measure`] against a named family, with per-glyph fallback to the global
/// face. An unloaded / generic family measures identically to [`measure`].
pub fn measure_family(family: &str, text: &str, px: f32) -> f32 {
    ensure_default();
    let name = norm_family(family);
    FAMILIES.with(|fams| {
        let primary = fams.iter().find(|(n, _)| *n == name).map(|(_, f)| f);
        FACE.with(|face| {
            let fallback = face.as_ref().map(|f| &f.font);
            let mut w = 0.0f32;
            for ch in text.chars() {
                w += match pick_font(primary, fallback, ch) {
                    Some(f) => f.metrics(ch, px).advance_width,
                    None => px * 0.55,
                };
            }
            w
        })
    })
}

/// [`blit_run`] against a named family (layout box top-left at `(x, y_top)`,
/// baseline at the family's ascent), with per-glyph fallback to the global
/// face. Returns the pen x after the last glyph.
#[allow(clippy::too_many_arguments)]
pub fn blit_run_family(
    buf: &mut [u32],
    stride: usize,
    height: usize,
    x: i32,
    y_top: i32,
    text: &str,
    px: f32,
    color: u32,
    family: &str,
) -> i32 {
    ensure_default();
    let name = norm_family(family);
    FAMILIES.with(|fams| {
        let primary = fams.iter().find(|(n, _)| *n == name).map(|(_, f)| f);
        FACE.with(|face| {
            let fallback = face.as_ref().map(|f| &f.font);
            let asc = primary
                .or(fallback)
                .and_then(|f| f.horizontal_line_metrics(px))
                .map(|lm| lm.ascent)
                .unwrap_or(px * 0.8);
            let baseline = y_top as f32 + asc;
            let mut pen_x = x as f32;
            for ch in text.chars() {
                pen_x += match pick_font(primary, fallback, ch) {
                    Some(f) => blit_glyph(buf, stride, height, pen_x, baseline, ch, px, color, f),
                    None => {
                        blit_box(buf, stride, height, pen_x as i32, baseline as i32, px, color)
                    }
                };
            }
            f32_round(pen_x) as i32
        })
    })
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

    #[test_case]
    fn family_registry_loads_and_measures() {
        ensure_default();
        load_family("TestFam", GEIST_TTF).expect("load");
        assert!(family_loaded("testfam"), "case-insensitive lookup");
        assert!(family_loaded(" TestFam "), "trimmed lookup");
        let a = measure("Hello", 16.0);
        let b = measure_family("TestFam", "Hello", 16.0);
        assert!((a - b).abs() < 0.01, "a={a} b={b}");
        assert!(load_family("", GEIST_TTF).is_err());
        assert!(load_family("bad", &[0u8; 8]).is_err());
    }

    #[test_case]
    fn family_blit_writes_pixels() {
        load_family("blitfam", GEIST_TTF).expect("load");
        let mut buf = alloc::vec![0x00ff_ffffu32; 128 * 48];
        let end = blit_run_family(&mut buf, 128, 48, 8, 8, "Hello", 16.0, 0x000000, "blitfam");
        assert!(end > 20, "end_x={end}");
        assert!(buf.iter().any(|&p| p != 0x00ff_ffff), "expected ink");
    }

    #[test_case]
    fn unknown_family_falls_back_to_global() {
        ensure_default();
        let a = measure("fallback text", 14.0);
        assert!(!family_loaded("no-such-family"));
        let b = measure_family("no-such-family", "fallback text", 14.0);
        assert!((a - b).abs() < 0.01, "a={a} b={b}");
        // Generic families resolve to the global too (registry miss).
        for generic in ["sans-serif", "serif", "monospace", ""] {
            let c = measure_family(generic, "fallback text", 14.0);
            assert!((a - c).abs() < 0.01, "{generic}: a={a} c={c}");
        }
    }

    #[test_case]
    fn glyph_fallback_renders_without_tofu_panic() {
        load_family("gfam", GEIST_TTF).expect("load");
        // A CJK ideograph Geist Mono does not cover: falls to the global
        // face's notdef — must still advance and blit without panicking.
        let text = "A\u{4e2d}B";
        let w = measure_family("gfam", text, 16.0);
        assert!(w > 0.0, "w={w}");
        let mut buf = alloc::vec![0u32; 96 * 32];
        let end = blit_run_family(&mut buf, 96, 32, 2, 2, text, 16.0, 0x00ff_ffff, "gfam");
        assert!(end > 2, "end_x={end}");
    }

    #[test_case]
    fn family_cap_evicts_oldest() {
        // 9 loads into a cap-8 FIFO: whatever the prior registry state, the
        // last 8 survive and the first of this batch is gone.
        for i in 0..9u32 {
            let name = alloc::format!("evict{i}");
            load_family(&name, GEIST_TTF).expect("load");
        }
        assert!(!family_loaded("evict0"));
        assert!(family_loaded("evict1"));
        assert!(family_loaded("evict8"));
        // Re-loading an existing family replaces in place (no growth/evict).
        load_family("evict8", GEIST_TTF).expect("reload");
        assert!(family_loaded("evict1"));
    }
}
