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

/// Font Awesome 6 Free Solid (SIL OFL) — system UI icons in the Private Use
/// Area (`U+F000`…). Registered **first** in the fallback chain so status-bar
/// and chrome icons resolve here before Noto/emoji scans. See [`crate::icons`].
static FONTAWESOME_SOLID: &[u8] =
    include_bytes!("../../assets/fonts/FontAwesome6Free-Solid-900.otf");

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
    // Monochrome emoji **before** CJK so symbol lookups hit a dedicated face
    // first (CJK has no emoji cmap, but scanning a large CFF face is expensive).
    // fontdue has no colour-table support — glyphs render single-colour.
    // (Not "Nord" — the theme named nord is unrelated; emoji is Noto Emoji.)
    ("Noto Emoji", include_bytes!("../../assets/fonts/Noto-Emoji.ttf")),
    // CJK — a **subset** (Latin + kana + CJK punctuation + ~3.5k common Han),
    // ~1.7 MB / ~8k glyphs. The full 65k-glyph face (~16 MB) parses for minutes
    // under the kernel's first-fit allocator (fontdue alloc churn is ~O(glyphs²)),
    // so it can't be bundled; this subset parses in ~1-2 s. Covers Chinese +
    // Japanese; Hangul (Korean) is omitted to keep the glyph count parse-able.
    ("Noto Sans CJK", include_bytes!("../../assets/fonts/Noto-CJK.otf")),
];

/// Register every bundled fallback face (idempotent). Call once at boot,
/// before the browser/UI render any non-Latin or icon text.
///
/// Order: **Font Awesome first** (PUA icons), then Noto scripts / emoji / CJK.
pub fn register_bundled_fallbacks() {
    // Icons before Noto so U+Fxxx never walks the huge CJK cmap first.
    const FA_NAME: &str = "Font Awesome 6 Free Solid";
    if !fallback_loaded(FA_NAME) {
        match register_fallback(FA_NAME, FONTAWESOME_SOLID) {
            Ok(()) => crate::ktrace::log_fmt(format_args!(
                "font: fallback '{FA_NAME}' ok ({} KiB)",
                FONTAWESOME_SOLID.len() / 1024
            )),
            Err(e) => crate::ktrace::log_fmt(format_args!("font: fallback '{FA_NAME}' failed: {e}")),
        }
    }
    for (name, bytes) in NOTO_FALLBACKS {
        // **Idempotent means "do not redo the work", not "do it again and overwrite".** These are
        // `include_bytes!` statics, so a face already registered under this name was parsed from
        // exactly these bytes and re-parsing cannot change the result — it only costs the parse
        // again. That is not a small cost: fontdue walks every glyph, and the emoji and CJK faces
        // are ~1.9 MB each; unoptimised (the unit suite) all ten take minutes.
        //
        // `register_fallback` itself still replaces unconditionally, because replacing a face
        // with *different* bytes is exactly what an installed font needs.
        if fallback_loaded(name) {
            continue;
        }
        if let Err(e) = register_fallback(name, bytes) {
            crate::ktrace::log_fmt(format_args!("font: fallback '{name}' failed: {e}"));
        } else {
            crate::ktrace::log_fmt(format_args!("font: fallback '{name}' ok ({} KiB)", bytes.len() / 1024));
        }
    }
}

/// Register **one** bundled fallback by name, if it is not already registered.
///
/// For a caller that needs a single script's coverage rather than the whole chain — notably the
/// unit suite, where parsing all ten faces to exercise one is minutes of a debug build.
///
/// Accepts `"Font Awesome 6 Free Solid"` (or `"fontawesome"`) as well as every
/// name in [`NOTO_FALLBACKS`].
pub fn register_bundled_fallback(want: &str) -> Result<(), &'static str> {
    if fallback_loaded(want) {
        return Ok(());
    }
    const FA_NAME: &str = "Font Awesome 6 Free Solid";
    let w = norm_family(want);
    if w == norm_family(FA_NAME) || w == "fontawesome" || w == "fa" {
        return register_fallback(FA_NAME, FONTAWESOME_SOLID);
    }
    let (name, bytes) = NOTO_FALLBACKS
        .iter()
        .find(|(name, _)| norm_family(name) == w)
        .ok_or("no such bundled fallback")?;
    register_fallback(name, bytes)
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

/// True for Font Awesome Private-Use scalars (Free Solid + FA6 extension).
#[inline]
fn is_fa_icon(ch: char) -> bool {
    let u = ch as u32;
    (0xf000..=0xf8ff).contains(&u) || (0xe000..=0xefff).contains(&u)
}

/// Build the sparse placed coverage for `ch` in a `cw × ch_px` monospace cell.
///
/// Default (Latin / emoji / CJK): size so the glyph's advance fills cell width,
/// centre the ascent/descent box vertically, place on the baseline.
///
/// **Font Awesome** is special: FA advances vary a lot (a solid circle is a
/// narrow advance; a keyboard is wide). Sizing by advance makes icons look
/// uneven and balloons markers. FA glyphs are scaled to a **fixed fraction of
/// cell height** and centred in the cell so keyboard / mouse / wifi / gear all
/// share one optical size.
///
/// Returns `(x, y, alpha)` for each ink pixel.
fn build_ui_glyph(font: &Font, ch: char, cw: usize, ch_px: usize) -> Vec<(u16, u16, u8)> {
    if is_fa_icon(ch) {
        // ~70% of cell height: readable next to mono body text without dominating.
        let px = (ch_px as f32 * 0.70).max(1.0);
        let (m, cov) = font.rasterize(ch, px);
        // Centre the bitmap in the cell (ignore xmin bearing — FA icons are
        // designed as squares and look even when optically centred).
        let gx0 = (cw as i32 - m.width as i32) / 2;
        let gy0 = (ch_px as i32 - m.height as i32) / 2;
        return pack_ui_glyph(m, &cov, gx0, gy0, cw, ch_px);
    }

    // Size by the glyph's own advance when non-ASCII (emoji / CJK). Sizing emoji
    // from the Latin '0' advance of Noto Emoji can over/under-scale and clip
    // all ink out of the cell — looks like empty boxes.
    let sample = if (ch as u32) > 0x7F { ch } else { '0' };
    let adv_per_px = {
        let m = font.metrics(sample, 64.0);
        if m.advance_width > 0.5 {
            m.advance_width / 64.0
        } else {
            let m0 = font.metrics('M', 64.0);
            if m0.advance_width > 0.5 {
                m0.advance_width / 64.0
            } else {
                0.6
            }
        }
    };
    // Fit both width and height so wide emoji stay inside the cell.
    let px_w = (cw as f32 / adv_per_px).max(1.0);
    let px_h = (ch_px as f32) * 0.92;
    let px = if px_w < px_h { px_w } else { px_h }.max(1.0);
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
    pack_ui_glyph(m, &cov, gx0, gy0, cw, ch_px)
}

fn pack_ui_glyph(
    m: fontdue::Metrics,
    cov: &[u8],
    gx0: i32,
    gy0: i32,
    cw: usize,
    ch_px: usize,
) -> Vec<(u16, u16, u8)> {
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
                        // Prefer the first face that both covers the codepoint
                        // and produces ink at cell size (skip empty .notdef).
                        for (_, f) in chain.iter() {
                            if f.lookup_glyph_index(ch) == 0 {
                                continue;
                            }
                            let g = build_ui_glyph(f, ch, cw, ch_px);
                            if !g.is_empty() {
                                return g;
                            }
                        }
                        build_ui_glyph(uifont, ch, cw, ch_px)
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

/// Ensure Font Awesome is in the fallback chain (idempotent, cheap if loaded).
fn ensure_font_awesome() {
    if fallback_loaded("Font Awesome 6 Free Solid") {
        return;
    }
    let _ = register_bundled_fallback("Font Awesome 6 Free Solid");
}

/// Rasterize a single glyph (typically Font Awesome) to a **cursor index sprite**:
/// `0` transparent, `1` fill, `2` outline.
///
/// Outline is a 1-px ring around solid ink so the pointer stays readable on both
/// light and dark backgrounds (same encoding as the hand-drawn built-ins).
/// Empty top/left margins are cropped so the arrow tip sits near (0,0) — the
/// framebuffer hotspots the sprite's top-left.
///
/// Returns `None` if the face is missing the glyph or produces no ink.
pub fn raster_cursor_sprite(ch: char, px: f32) -> Option<(usize, usize, Vec<u8>)> {
    ensure_font_awesome();
    ensure_default();
    let px = px.max(8.0).min(28.0);
    let (metrics, cov) = FALLBACKS.with(|chain| {
        for (_, f) in chain.iter() {
            if f.lookup_glyph_index(ch) == 0 {
                continue;
            }
            return Some(f.rasterize(ch, px));
        }
        // Last resort: active UI face (usually no FA coverage).
        FACE.with(|slot| slot.as_ref().map(|face| face.font.rasterize(ch, px)))
    })?;
    if metrics.width == 0 || metrics.height == 0 || cov.len() < metrics.width * metrics.height {
        return None;
    }
    let gw = metrics.width;
    let gh = metrics.height;
    // Pad 1 px for outline dilation.
    let pad = 1usize;
    let aw = gw + pad * 2;
    let ah = gh + pad * 2;
    let mut alpha = alloc::vec![0u8; aw * ah];
    for row in 0..gh {
        for col in 0..gw {
            let a = cov[row * gw + col];
            if a > 0 {
                alpha[(row + pad) * aw + (col + pad)] = a;
            }
        }
    }
    // 0/1/2: solid ink → fill; halo of near-ink + empty neighbours of fill → outline.
    let mut idx = alloc::vec![0u8; aw * ah];
    for y in 0..ah {
        for x in 0..aw {
            let a = alpha[y * aw + x];
            if a >= 140 {
                idx[y * aw + x] = 1;
            } else if a >= 40 {
                idx[y * aw + x] = 2;
            }
        }
    }
    // Dilate outline around fill where transparent.
    let mut out = idx.clone();
    for y in 0..ah {
        for x in 0..aw {
            if idx[y * aw + x] != 1 {
                continue;
            }
            for (dx, dy) in [(-1i32, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)] {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx < 0 || ny < 0 || nx as usize >= aw || ny as usize >= ah {
                    continue;
                }
                let i = ny as usize * aw + nx as usize;
                if out[i] == 0 {
                    out[i] = 2;
                }
            }
        }
    }
    // Crop empty margins (keep outline), but leave at most 0 empty on top-left
    // so the hotspot lands on the tip for arrow/hand.
    let mut min_x = aw;
    let mut min_y = ah;
    let mut max_x = 0usize;
    let mut max_y = 0usize;
    let mut any = false;
    for y in 0..ah {
        for x in 0..aw {
            if out[y * aw + x] != 0 {
                any = true;
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
        }
    }
    if !any {
        return None;
    }
    let cw = (max_x - min_x + 1).min(32);
    let chh = (max_y - min_y + 1).min(32);
    let mut data = Vec::with_capacity(cw * chh);
    for y in min_y..min_y + chh {
        for x in min_x..min_x + cw {
            data.push(if y < ah && x < aw { out[y * aw + x] } else { 0 });
        }
    }
    Some((cw, chh, data))
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
    blit_run_baseline_ex(buf, stride, height, pen_x, baseline_y, text, px, color, false)
}

/// Like [`blit_run_baseline`] but with hard-edged (non-AA) glyphs for surfaces
/// that will be nearest-neighbour upscaled (package-UI canvas text).
pub fn blit_run_baseline_crisp(
    buf: &mut [u32],
    stride: usize,
    height: usize,
    pen_x: f32,
    baseline_y: f32,
    text: &str,
    px: f32,
    color: u32,
) -> i32 {
    blit_run_baseline_ex(buf, stride, height, pen_x, baseline_y, text, px, color, true)
}

fn blit_run_baseline_ex(
    buf: &mut [u32],
    stride: usize,
    height: usize,
    mut pen_x: f32,
    baseline_y: f32,
    text: &str,
    px: f32,
    color: u32,
    crisp: bool,
) -> i32 {
    ensure_default();
    let shaped = crate::font_shape::shape(text);
    FACE.with(|s| {
        let global = s.as_ref().map(|f| &f.font);
        FALLBACKS.with(|chain| {
            for ch in shaped.chars() {
                pen_x += match pick_font(None, global, chain, ch) {
                    Some(f) => {
                        blit_glyph(buf, stride, height, pen_x, baseline_y, ch, px, color, f, crisp)
                    }
                    None => {
                        blit_box(buf, stride, height, pen_x as i32, baseline_y as i32, px, color)
                    }
                };
            }
            f32_round(pen_x) as i32
        })
    })
}

/// Hi-res coverage floor for the crisp supersample path. Samples below this
/// are pure AA fringe and get dropped; anything above solidifies the destination
/// pixel. Kept low so thin curves (o, e, s, g) survive at 10–14 px canvas sizes.
const CRISP_SS_FLOOR: u8 = 40;
/// Supersample factor for package-UI text (render at N×, max-pool to canvas).
const CRISP_SS: i32 = 2;

/// Rasterize one glyph from `font` with the pen at `(pen_x, baseline_y)` and
/// blend it into `buf`; returns the glyph advance.
///
/// When `crisp` is true, the glyph is rendered at [`CRISP_SS`]× size and
/// max-pooled onto the canvas as solid ink. That keeps curves intact (a 1×
/// hard threshold was punching holes in e/s/o stems) while still avoiding the
/// soft grey AA fringes that blur under integer upscale of the 256×192 surface.
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
    crisp: bool,
) -> f32 {
    if crisp {
        return blit_glyph_crisp(buf, stride, height, pen_x, baseline_y, ch, px, color, font);
    }
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

/// Supersampled solid-ink glyph for package-UI canvases.
#[allow(clippy::too_many_arguments)]
fn blit_glyph_crisp(
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
    let ss = CRISP_SS as f32;
    let (m, coverage) = font.rasterize(ch, (px * ss).max(1.0));
    // Pen/baseline on the hi-res grid, then each ink sample folds into the
    // lo-res pixel that covers it (floor-div by SS). Any sample above the floor
    // paints the dest pixel solid — max-pool semantics, preserves thin strokes.
    let gx_hi = f32_floor(pen_x * ss) + m.xmin;
    let gy_hi = f32_floor(baseline_y * ss) - m.height as i32 - m.ymin;
    let gw = m.width;
    let gh = m.height;
    if gw > 0 && gh > 0 && coverage.len() >= gw * gh {
        for row in 0..gh {
            for col in 0..gw {
                let a = coverage[row * gw + col];
                if a < CRISP_SS_FLOOR {
                    continue;
                }
                let hx = gx_hi + col as i32;
                let hy = gy_hi + row as i32;
                // i32 div_euclid maps negative coords correctly for off-left glyphs.
                let dx = hx.div_euclid(CRISP_SS);
                let dy = hy.div_euclid(CRISP_SS);
                put_blend(buf, stride, height, dx, dy, color, 255);
            }
        }
    }
    // Advance at the *requested* (lo-res) size so spacing matches layout.
    font.metrics(ch, px).advance_width
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

/// Hard-edged variant of [`blit_run`] for package-UI canvas labels: no soft
/// antialiasing, so integer upscale of the 256×192 surface stays sharp.
pub fn blit_run_crisp(
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
    blit_run_baseline_crisp(buf, stride, height, x as f32, baseline, text, px, color)
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
                            blit_glyph(buf, stride, height, pen_x, baseline, ch, px, color, f, false)
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
    fn fa_cursor_sprite_has_fill_and_outline() {
        register_bundled_fallback("Font Awesome 6 Free Solid").expect("FA");
        let (w, h, data) =
            raster_cursor_sprite(crate::icons::fa::ARROW_POINTER, 18.0).expect("arrow sprite");
        assert!(w >= 8 && h >= 8, "sprite too small {w}x{h}");
        assert_eq!(data.len(), w * h);
        let fill = data.iter().filter(|&&v| v == 1).count();
        let outline = data.iter().filter(|&&v| v == 2).count();
        assert!(fill > 10, "need fill pixels, got {fill}");
        assert!(outline > 5, "need outline ring, got {outline}");
        // Hand / I-beam / hourglass also produce ink.
        for ch in [
            crate::icons::fa::HAND_POINTER,
            crate::icons::fa::I_CURSOR,
            crate::icons::fa::HOURGLASS,
        ] {
            let (w, h, d) = raster_cursor_sprite(ch, 18.0).expect("sprite");
            assert!(d.iter().any(|&v| v == 1), "U+{:04X} no fill {w}x{h}", ch as u32);
        }
    }

    #[test_case]
    fn font_awesome_fallback_has_ink_for_status_icons() {
        // Register only the FA face (not the whole Noto chain) — unit suite budget.
        register_bundled_fallback("Font Awesome 6 Free Solid").expect("FA must register");
        assert!(fallback_loaded("Font Awesome 6 Free Solid"));
        // Asset sanity: OTF magic + non-trivial size (~1 MB Free Solid).
        assert!(FONTAWESOME_SOLID.len() > 100_000, "FA OTF truncated?");
        assert_eq!(&FONTAWESOME_SOLID[..4], b"OTTO", "FA face must be CFF/OTF");
        for ch in [
            crate::icons::fa::KEYBOARD,
            crate::icons::fa::MOUSE,
            crate::icons::fa::WIFI,
            crate::icons::fa::GEAR,
            crate::icons::fa::FOLDER,
            crate::icons::fa::CODE_COMPARE, // FA6 extension PUA
        ] {
            let mut ink = 0u32;
            let ok = blit_ui_cell(ch, 12, 22, |_x, _y, a| {
                if a > 10 {
                    ink += 1;
                }
            });
            assert!(ok, "blit_ui_cell must succeed for U+{:04X}", ch as u32);
            assert!(
                ink > 8,
                "FA icon U+{:04X} must produce ink, got {ink}",
                ch as u32
            );
        }
    }

    #[test_case]
    fn noto_emoji_fallback_has_ink_in_ui_cell() {
        // **One face, not the whole chain.** This test asserts things about emoji, and it used to
        // call `register_bundled_fallbacks()` — parsing all ten Noto faces, ~4.5 MB including the
        // 1.9 MB CJK subset, none of which it looks at. Unoptimised that was minutes of the
        // suite's runtime for one assertion about a heart and a smiley.
        //
        // The chain as a whole is exercised at boot (`shell` registers it) and by
        // `every_bundled_fallback_is_registerable`, which is cheap because it does not parse.
        register_bundled_fallback("Noto Emoji").expect("Noto Emoji must register");
        assert!(fallback_loaded("Noto Emoji"), "Noto Emoji must register");
        // Typical chat-cell size (scaled 1× Geist metrics).
        let mut ink = 0u32;
        let ok = blit_ui_cell('😊', 10, 22, |_x, _y, a| {
            if a > 10 {
                ink += 1;
            }
        });
        assert!(ok, "blit_ui_cell must succeed");
        assert!(ink > 20, "emoji must produce ink, got {ink} pixels");
        // Variation-selector base heart (❤) without VS16.
        ink = 0;
        let _ = blit_ui_cell('\u{2764}', 10, 22, |_x, _y, a| {
            if a > 10 {
                ink += 1;
            }
        });
        assert!(ink > 10, "heart must produce ink, got {ink}");
    }

    #[test_case]
    fn every_bundled_fallback_is_present_and_plausible() {
        // The coverage the emoji test gave up when it stopped parsing all ten faces — kept, but
        // **without a parse**, because what actually goes wrong with a bundled asset is a missing
        // or truncated file, a duplicated name, or an `include_bytes!` pointing at the wrong
        // thing. All of that is visible in the first four bytes and the name; none of it needs
        // fontdue to walk 3600 glyphs. The parse itself is proven once, on the emoji face, and at
        // boot on all of them.
        assert!(NOTO_FALLBACKS.len() >= 8, "the fallback chain looks truncated");
        for (name, bytes) in NOTO_FALLBACKS {
            assert!(!name.is_empty(), "a fallback with no name cannot be looked up");
            assert!(bytes.len() > 4096, "'{name}' is {} bytes -- truncated asset?", bytes.len());
            // sfnt version: TrueType (0x00010000 / 'true') or CFF/OpenType ('OTTO'), and 'ttcf'
            // for a collection. Anything else is not a font, whatever the filename says.
            let tag = &bytes[..4];
            assert!(
                tag == [0x00, 0x01, 0x00, 0x00] || tag == b"OTTO" || tag == b"true" || tag == b"ttcf",
                "'{name}' does not start with an sfnt version: {tag:?}"
            );
            // Registerable by its own name, which is what `register_bundled_fallback` needs and
            // what the boot path looks up.
            assert_eq!(
                NOTO_FALLBACKS.iter().filter(|(n, _)| norm_family(n) == norm_family(name)).count(),
                1,
                "'{name}' is not a unique fallback name"
            );
        }
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
