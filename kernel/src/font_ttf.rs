//! **Runtime TTF/OTF rasterizer** (fontdue) — Ladybird/LibGfx-style placement:
//! pen on a **baseline**, glyph bitmap offset by `xmin` / `ymin`, advance by
//! `advance_width`. Screen space is Y-down (like HTML/CSS).
//!
//! Default face: embedded `assets/fonts/GeistMono-Regular.ttf` (SIL OFL). Swap
//! with [`load_bytes`] for other `.ttf`/`.otf` files, or register additional
//! **families** with [`load_family`] and render via [`measure_family`] /
//! [`blit_run_family`] (per-glyph fallback to the global face).
//!
//! **Bundled UI faces** live under `assets/fonts/` ([`BUNDLED_FONTS`]): Geist
//! Mono + Ubuntu Mono. The compositor renders the console/UI through this
//! rasterizer (see `framebuffer`), with the face chosen from `ui.json`.

use crate::mm::Locked;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use fontdue::{Font, FontSettings};

static GEIST_TTF: &[u8] = include_bytes!("../../assets/fonts/GeistMono-Regular.ttf");
static UBUNTU_MONO_TTF: &[u8] = include_bytes!("../../assets/fonts/UbuntuMono-Regular.ttf");

/// Bundled **Noto Sans** script fonts (SIL Open Font License — see
/// THIRDPARTY-LICENSES.md) forming the system fallback chain, so Indic web
/// text renders real glyphs instead of tofu. These are the same faces Linux
/// ships in `fonts-noto`. Registered at boot via [`register_bundled_fallbacks`].
pub static NOTO_FALLBACKS: &[(&str, &[u8])] = &[
    ("Noto Sans Devanagari", include_bytes!("../../assets/fonts/Noto-Devanagari.ttf")),
    ("Noto Sans Bengali", include_bytes!("../../assets/fonts/Noto-Bengali.ttf")),
    ("Noto Sans Gurmukhi", include_bytes!("../../assets/fonts/Noto-Gurmukhi.ttf")),
    ("Noto Sans Gujarati", include_bytes!("../../assets/fonts/Noto-Gujarati.ttf")),
    ("Noto Sans Tamil", include_bytes!("../../assets/fonts/Noto-Tamil.ttf")),
    ("Noto Sans Telugu", include_bytes!("../../assets/fonts/Noto-Telugu.ttf")),
    ("Noto Sans Kannada", include_bytes!("../../assets/fonts/Noto-Kannada.ttf")),
    ("Noto Sans Malayalam", include_bytes!("../../assets/fonts/Noto-Malayalam.ttf")),
    // CJK — a **subset** (Latin + kana + CJK punctuation + ~3.5k common Han),
    // ~1.7 MB / ~8k glyphs. The full 65k-glyph face (~16 MB) parses for minutes
    // under the kernel's first-fit allocator (fontdue alloc churn is ~O(glyphs²)),
    // so it can't be bundled; this subset parses in ~1-2 s. Covers Chinese +
    // Japanese; Hangul (Korean) is omitted to keep the glyph count parse-able.
    ("Noto Sans CJK", include_bytes!("../../assets/fonts/Noto-CJK.otf")),
    // Monochrome emoji last: Latin/Indic are matched by earlier faces first, so
    // this is only reached for emoji/symbol codepoints. (fontdue has no colour
    // table support, so these render as single-colour glyphs.)
    ("Noto Emoji", include_bytes!("../../assets/fonts/Noto-Emoji.ttf")),
];

/// Register every bundled Noto fallback face (idempotent). Call once at boot,
/// before the browser/UI render any non-Latin text.
pub fn register_bundled_fallbacks() {
    for (name, bytes) in NOTO_FALLBACKS {
        let _ = register_fallback(name, bytes);
    }
}

/// Monospace faces bundled for the UI, selectable by name from `ui.json`
/// (`font`). The first entry is the default. Names are matched
/// case-insensitively and also by a lowercased no-space alias
/// (e.g. `"geist mono"` / `"geist-mono"` / `"geistmono"`).
pub static BUNDLED_FONTS: &[(&str, &[u8])] =
    &[("Geist Mono", GEIST_TTF), ("Ubuntu Mono", UBUNTU_MONO_TTF)];

/// Return the bundled font bytes whose name matches `want` (case/space
/// insensitive), or `None`. Used by the UI-font selector.
pub fn bundled_font_bytes(want: &str) -> Option<&'static [u8]> {
    let w = norm_font_name(want);
    BUNDLED_FONTS
        .iter()
        .find(|(name, _)| norm_font_name(name) == w)
        .map(|(_, b)| *b)
}

fn norm_font_name(s: &str) -> String {
    s.chars()
        .filter(|c| !c.is_whitespace() && *c != '-' && *c != '_')
        .flat_map(|c| c.to_lowercase())
        .collect()
}

// ---- UI (console/compositor) font: TTF rendering into a fixed monospace cell ----

/// The face the compositor renders the UI/console with (chosen from `ui.json`'s
/// `font`; default = first [`BUNDLED_FONTS`] entry, Geist Mono).
static UI_FONT: Locked<Option<Font>> = Locked::new(None);
/// The selected UI face name (canonical bundled name).
static UI_FONT_NAME: Locked<Option<String>> = Locked::new(None);
/// Placed-glyph cache: `(char, cell_w, cell_h)` → sparse non-zero coverage
/// `(x, y, alpha)` already positioned on the cell baseline. Sparse so a blit
/// touches only the ~ink pixels (no per-cell allocation churn — trap #3).
/// Lock order: `UI_CACHE` is taken **inside** `UI_FONT`, never the reverse.
static UI_CACHE: Locked<BTreeMap<(char, u16, u16), Vec<(u16, u16, u8)>>> =
    Locked::new(BTreeMap::new());

fn ensure_ui_font() {
    UI_FONT.with(|slot| {
        if slot.is_none() {
            if let Ok(font) = Font::from_bytes(BUNDLED_FONTS[0].1, FontSettings::default()) {
                *slot = Some(font);
                UI_FONT_NAME.with(|n| *n = Some(String::from(BUNDLED_FONTS[0].0)));
            }
        }
    });
}

/// Select the UI/console font by name (a [`BUNDLED_FONTS`] entry, case/space
/// insensitive — `"Ubuntu Mono"`, `"ubuntu-mono"`, …). No-op if unknown or
/// unchanged; clears the placed-glyph cache on change. Driven by `ui.json`.
pub fn set_ui_font(name: &str) {
    let canon = norm_font_name(name);
    if UI_FONT_NAME.with(|n| n.as_deref().map(norm_font_name) == Some(canon.clone())) {
        return; // unchanged
    }
    let Some(bytes) = bundled_font_bytes(name) else {
        return;
    };
    if let Ok(font) = Font::from_bytes(bytes, FontSettings::default()) {
        UI_FONT.with(|slot| *slot = Some(font));
        UI_FONT_NAME.with(|n| *n = Some(String::from(name)));
        UI_CACHE.with(|c| c.clear());
    }
}

/// The active UI face name (for `/ui`).
pub fn ui_font_name() -> Option<String> {
    ensure_ui_font();
    UI_FONT_NAME.with(|n| n.clone())
}

/// Build the sparse placed coverage for `ch` in a `cw × ch_px` monospace cell:
/// size the glyph so its advance fills the cell **width** (sizing by the cell
/// height over-sizes the glyph and clips its right edge), centre the
/// ascent/descent box vertically, place on the baseline, and clip to the cell.
/// Returns `(x, y, alpha)` for each ink pixel.
fn build_ui_glyph(font: &Font, ch: char, cw: usize, ch_px: usize) -> Vec<(u16, u16, u8)> {
    // Monospace advance per px (all glyphs share it) → the px at which one glyph
    // advance exactly fills the cell width.
    let adv_per_px = {
        let m = font.metrics('0', 64.0);
        if m.advance_width > 0.5 {
            m.advance_width / 64.0
        } else {
            0.6
        }
    };
    let px = (cw as f32 / adv_per_px).max(1.0);
    let (m, cov) = font.rasterize(ch, px);
    let (asc, desc) = font
        .horizontal_line_metrics(px)
        .map(|l| (l.ascent, l.descent))
        .unwrap_or((px * 0.8, -px * 0.2));
    // Vertically centre the (ascent−descent) box in the cell, place on baseline.
    let box_h = asc - desc;
    let top_pad = ((ch_px as f32 - box_h) / 2.0).max(0.0);
    let baseline = top_pad + asc;
    // Advance now equals the cell width; place at the glyph's own left bearing.
    let gx0 = m.xmin;
    let gy0 = (baseline - m.height as f32 - m.ymin as f32) as i32;
    let mut out = Vec::new();
    if m.width > 0 && m.height > 0 && cov.len() >= m.width * m.height {
        for row in 0..m.height {
            for col in 0..m.width {
                let a = cov[row * m.width + col];
                if a == 0 {
                    continue;
                }
                let x = gx0 + col as i32;
                let y = gy0 + row as i32;
                if x >= 0 && (x as usize) < cw && y >= 0 && (y as usize) < ch_px {
                    out.push((x as u16, y as u16, a));
                }
            }
        }
    }
    out
}

/// Blit `ch` into a `cw × ch_px` UI cell via the selected TTF face: `plot(x, y,
/// alpha)` is called for each ink pixel (cell-local coords), so the caller
/// alpha-blends into its framebuffer. Cached per `(ch, cw, ch_px)`. Returns
/// `false` if no UI font is available (caller falls back to the bitmap atlas).
pub fn blit_ui_cell<F: FnMut(usize, usize, u8)>(
    ch: char,
    cw: usize,
    ch_px: usize,
    mut plot: F,
) -> bool {
    ensure_ui_font();
    if cw == 0 || ch_px == 0 || cw > u16::MAX as usize || ch_px > u16::MAX as usize {
        return false;
    }
    let key = (ch, cw as u16, ch_px as u16);
    UI_FONT.with(|slot| {
        let Some(uifont) = slot.as_ref() else {
            return false;
        };
        // If the UI monospace face doesn't cover this char (Indic/CJK/emoji),
        // fall back to a system fallback face that does — so the console/UI
        // renders non-Latin text OS-wide, not just the browser.
        let use_fallback = ch != ' ' && uifont.lookup_glyph_index(ch) == 0;
        UI_CACHE.with(|cache| {
            if !cache.contains_key(&key) {
                let glyph = if use_fallback {
                    FALLBACKS.with(|chain| {
                        match chain.iter().find(|(_, f)| f.lookup_glyph_index(ch) != 0) {
                            Some((_, f)) => build_ui_glyph(f, ch, cw, ch_px),
                            None => build_ui_glyph(uifont, ch, cw, ch_px),
                        }
                    })
                } else {
                    build_ui_glyph(uifont, ch, cw, ch_px)
                };
                cache.insert(key, glyph);
            }
            if let Some(glyph) = cache.get(&key) {
                for &(x, y, a) in glyph.iter() {
                    plot(x as usize, y as usize, a);
                }
            }
        });
        true
    })
}

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

/// System **fallback chain** — script/emoji/CJK faces consulted per-glyph
/// (in registration order) when neither the primary family nor the global face
/// covers a character. This is how Indic / CJK / emoji text stops rendering as
/// tofu: the global face (Geist Mono) covers Latin, and each fallback covers a
/// script it was registered for (Noto Sans Devanagari, …). Lock order is always
/// `FAMILIES → FACE → FALLBACKS`; never the reverse.
static FALLBACKS: Locked<Vec<(String, Font)>> = Locked::new(Vec::new());

/// Register a system fallback face (a Noto script/emoji/CJK font). Idempotent
/// by name. Unlike [`load_family`] there is no cap — the chain is a small,
/// boot-time-fixed set of script coverage fonts.
pub fn register_fallback(name: &str, data: &[u8]) -> Result<(), &'static str> {
    let name = norm_family(name);
    if name.is_empty() {
        return Err("empty fallback name");
    }
    let font = Font::from_bytes(data, FontSettings::default()).map_err(|_| "font parse failed")?;
    FALLBACKS.with(move |fb| {
        if let Some(i) = fb.iter().position(|(n, _)| *n == name) {
            fb[i].1 = font;
        } else {
            fb.push((name, font));
        }
    });
    Ok(())
}

/// True if a fallback face named `name` (case-insensitive) is registered.
pub fn fallback_loaded(name: &str) -> bool {
    let name = norm_family(name);
    FALLBACKS.with(|fb| fb.iter().any(|(n, _)| *n == name))
}

/// Number of registered fallback faces.
pub fn fallback_count() -> usize {
    FALLBACKS.with(|fb| fb.len())
}

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
    ensure_default();
    // Width is order-independent, so shaping (reordering) is unnecessary here —
    // but the fallback chain is: an Indic/CJK/emoji glyph's advance comes from
    // whichever face actually covers it, not the Latin face's notdef.
    FACE.with(|s| {
        let global = s.as_ref().map(|f| &f.font);
        FALLBACKS.with(|chain| {
            let mut w = 0.0f32;
            for ch in text.chars() {
                w += match pick_font(None, global, chain, ch) {
                    Some(f) => f.metrics(ch, px).advance_width,
                    None => px * 0.55,
                };
            }
            w
        })
    })
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
    let shaped = crate::font_shape::shape(text);
    FACE.with(|s| {
        let global = s.as_ref().map(|f| &f.font);
        FALLBACKS.with(|chain| {
            for ch in shaped.chars() {
                pen_x += match pick_font(None, global, chain, ch) {
                    Some(f) => blit_glyph(buf, stride, height, pen_x, baseline_y, ch, px, color, f),
                    None => {
                        blit_box(buf, stride, height, pen_x as i32, baseline_y as i32, px, color)
                    }
                };
            }
            f32_round(pen_x) as i32
        })
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

/// Per-glyph face selection, in priority order: the named family (if any) →
/// the global face → each registered fallback (script/emoji/CJK) → the global
/// face's notdef. This is what routes an Indic/CJK/emoji codepoint the Latin
/// face can't draw to the Noto face that can.
fn pick_font<'a>(
    primary: Option<&'a Font>,
    global: Option<&'a Font>,
    chain: &'a [(String, Font)],
    ch: char,
) -> Option<&'a Font> {
    if let Some(p) = primary {
        if p.lookup_glyph_index(ch) != 0 {
            return Some(p);
        }
    }
    if let Some(g) = global {
        if g.lookup_glyph_index(ch) != 0 {
            return Some(g);
        }
    }
    for (_, f) in chain {
        if f.lookup_glyph_index(ch) != 0 {
            return Some(f);
        }
    }
    global.or(primary)
}

/// [`measure`] against a named family, with per-glyph fallback to the global
/// face. An unloaded / generic family measures identically to [`measure`].
pub fn measure_family(family: &str, text: &str, px: f32) -> f32 {
    ensure_default();
    let name = norm_family(family);
    FAMILIES.with(|fams| {
        let primary = fams.iter().find(|(n, _)| *n == name).map(|(_, f)| f);
        FACE.with(|face| {
            let global = face.as_ref().map(|f| &f.font);
            FALLBACKS.with(|chain| {
                let mut w = 0.0f32;
                for ch in text.chars() {
                    w += match pick_font(primary, global, chain, ch) {
                        Some(f) => f.metrics(ch, px).advance_width,
                        None => px * 0.55,
                    };
                }
                w
            })
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
    let shaped = crate::font_shape::shape(text);
    FAMILIES.with(|fams| {
        let primary = fams.iter().find(|(n, _)| *n == name).map(|(_, f)| f);
        FACE.with(|face| {
            let global = face.as_ref().map(|f| &f.font);
            FALLBACKS.with(|chain| {
                let asc = primary
                    .or(global)
                    .and_then(|f| f.horizontal_line_metrics(px))
                    .map(|lm| lm.ascent)
                    .unwrap_or(px * 0.8);
                let baseline = y_top as f32 + asc;
                let mut pen_x = x as f32;
                for ch in shaped.chars() {
                    pen_x += match pick_font(primary, global, chain, ch) {
                        Some(f) => {
                            blit_glyph(buf, stride, height, pen_x, baseline, ch, px, color, f)
                        }
                        None => {
                            blit_box(buf, stride, height, pen_x as i32, baseline as i32, px, color)
                        }
                    };
                }
                f32_round(pen_x) as i32
            })
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
