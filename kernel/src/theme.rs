//! **Installable themes** — presets that populate the live UI config
//! ([`crate::ui_config`]). A theme is pure data (JSON): a chrome `palette`, code
//! `syntax` colours, `cursor` colours + optional custom sprite bitmaps, a
//! `wallpaper` + `opacity`, and a `font` + `font_scale`. Because themes carry no
//! code or capabilities they need none of the signed-package machinery — they
//! mirror the seeded-JSON pattern of `ui_config`.
//!
//! Bundled presets are compiled in ([`BUNDLED_THEMES`]); installed themes live
//! at `/configs/themes/<name>.json` in the synapse store and take precedence
//! over a bundled name. `apply` merges a theme's appearance fields into the live
//! config, loads its cursor sprites into the framebuffer, persists `ui.json`,
//! and re-applies — so `/ui`/`/open` fine-tuning still works afterward.

use crate::json::Json;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

const THEMES_DIR: &str = "/configs/themes";

/// Themes compiled into the image (`assets/themes/<name>.json`). Installed
/// themes with the same name (in the store) override these.
pub static BUNDLED_THEMES: &[(&str, &str)] = &[
    ("dark", include_str!("../../assets/themes/dark.json")),
    ("light", include_str!("../../assets/themes/light.json")),
    ("solarized-dark", include_str!("../../assets/themes/solarized-dark.json")),
    ("nord", include_str!("../../assets/themes/nord.json")),
    ("dracula", include_str!("../../assets/themes/dracula.json")),
    ("ubuntu", include_str!("../../assets/themes/ubuntu.json")),
    // Imported from omarchy (https://github.com/basecamp/omarchy, MIT) by
    // `tools/import_omarchy_themes.py`. Palettes only: their wallpapers are
    // 108 MB with their own provenance, so each renders a gradient derived from
    // the colours instead. `nord` is listed above -- both projects ship one and
    // the import replaced ours. Regenerate rather than hand-editing.
    ("catppuccin", include_str!("../../assets/themes/catppuccin.json")),
    ("catppuccin-latte", include_str!("../../assets/themes/catppuccin-latte.json")),
    ("ethereal", include_str!("../../assets/themes/ethereal.json")),
    ("everforest", include_str!("../../assets/themes/everforest.json")),
    ("flexoki-light", include_str!("../../assets/themes/flexoki-light.json")),
    ("gruvbox", include_str!("../../assets/themes/gruvbox.json")),
    ("hackerman", include_str!("../../assets/themes/hackerman.json")),
    ("kanagawa", include_str!("../../assets/themes/kanagawa.json")),
    ("last-horizon", include_str!("../../assets/themes/last-horizon.json")),
    ("lumon", include_str!("../../assets/themes/lumon.json")),
    ("lupine", include_str!("../../assets/themes/lupine.json")),
    ("matte-black", include_str!("../../assets/themes/matte-black.json")),
    ("miasma", include_str!("../../assets/themes/miasma.json")),
    ("osaka-jade", include_str!("../../assets/themes/osaka-jade.json")),
    ("retro-82", include_str!("../../assets/themes/retro-82.json")),
    ("ristretto", include_str!("../../assets/themes/ristretto.json")),
    ("rose-pine", include_str!("../../assets/themes/rose-pine.json")),
    ("solitude", include_str!("../../assets/themes/solitude.json")),
    ("tokyo-night", include_str!("../../assets/themes/tokyo-night.json")),
    ("vantablack", include_str!("../../assets/themes/vantablack.json")),
    ("white", include_str!("../../assets/themes/white.json")),
];

/// All available theme names: bundled + installed (in the store), de-duplicated.
pub fn list() -> Vec<String> {
    let mut names: Vec<String> = BUNDLED_THEMES.iter().map(|(n, _)| n.to_string()).collect();
    let prefix = "/configs/themes/";
    for key in crate::synapse::fs::list() {
        if let Some(rest) = key.strip_prefix(prefix) {
            if let Some(n) = rest.strip_suffix(".json") {
                if !n.is_empty() && !n.contains('/') && !names.iter().any(|x| x == n) {
                    names.push(n.to_string());
                }
            }
        }
    }
    names
}

/// The raw JSON text of a theme: an installed copy in the store wins over the
/// bundled one, so a user can override a preset by name.
pub fn load_text(name: &str) -> Option<String> {
    let path = format!("{THEMES_DIR}/{name}.json");
    if let Some(bytes) = crate::synapse::fs::read(&path) {
        if let Ok(s) = core::str::from_utf8(&bytes) {
            return Some(s.to_string());
        }
    }
    BUNDLED_THEMES
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, j)| j.to_string())
}

/// Override each `(key, hex)` pair in `into` from the JSON object `obj`,
/// keeping any key the object doesn't mention.
fn overlay_pairs(obj: Option<&Json>, into: &mut [(String, String)]) {
    if let Some(o) = obj {
        for (k, hex) in into.iter_mut() {
            if let Some(v) = o.get(k).and_then(|v| v.as_str()) {
                *hex = v.to_string();
            }
        }
    }
}

/// Apply a theme by name: merge its appearance fields into the live config,
/// install its cursor sprites, persist `ui.json`, and re-apply the compositor.
pub fn apply(name: &str) -> Result<(), String> {
    let text = load_text(name).ok_or_else(|| format!("theme '{name}' not found"))?;
    let j = Json::parse(&text).ok_or_else(|| String::from("theme JSON parse error"))?;
    let mut cfg = crate::ui_config::current();
    overlay_pairs(j.get("palette"), &mut cfg.theme);
    overlay_pairs(j.get("syntax"), &mut cfg.syntax);
    if let Some(c) = j.get("cursor") {
        if let Some(v) = c.get("fill").and_then(|v| v.as_str()) {
            cfg.cursor_fill = v.to_string();
        }
        if let Some(v) = c.get("outline").and_then(|v| v.as_str()) {
            cfg.cursor_outline = v.to_string();
        }
    }
    if let Some(v) = j.get("wallpaper").and_then(|v| v.as_str()) {
        cfg.wallpaper = v.to_string();
    }
    if let Some(v) = j.get("opacity").and_then(|v| v.as_i64()) {
        cfg.opacity = v.clamp(0, 255) as u64;
    }
    if let Some(v) = j.get("font").and_then(|v| v.as_str()) {
        cfg.font = v.to_string();
    }
    if let Some(v) = j.get("font_scale").and_then(|v| v.as_i64()) {
        cfg.font_scale = v.max(0) as u64;
    }
    cfg.theme_name = name.to_string();
    // Cursor sprites live only in the theme file (never `ui.json`); install them
    // into the framebuffer directly. Absent ⇒ clear to built-ins.
    #[cfg(not(test))]
    crate::framebuffer::set_cursor_sprites(parse_cursor_sprites(j.get("cursor")));
    crate::ui_config::set_config(cfg);
    Ok(())
}

/// Parse a theme's optional `cursor.sprites` object into per-shape bitmaps
/// (order Arrow, Hand, IBeam, Wait). `None` when absent/empty ⇒ built-ins.
#[cfg(not(test))]
fn parse_cursor_sprites(cursor: Option<&Json>) -> Option<[crate::framebuffer::CursorSprite; 4]> {
    let sprites = cursor?.get("sprites")?;
    let shapes = ["arrow", "hand", "ibeam", "wait"];
    let mut out = [
        crate::framebuffer::CursorSprite::default(),
        crate::framebuffer::CursorSprite::default(),
        crate::framebuffer::CursorSprite::default(),
        crate::framebuffer::CursorSprite::default(),
    ];
    let mut any = false;
    for (i, sh) in shapes.iter().enumerate() {
        let Some(sp) = sprites.get(sh) else { continue };
        let w = sp.get("w").and_then(|v| v.as_i64()).unwrap_or(0).clamp(0, crate::framebuffer::CUR_MAX as i64) as usize;
        let h = sp.get("h").and_then(|v| v.as_i64()).unwrap_or(0).clamp(0, crate::framebuffer::CUR_MAX as i64) as usize;
        let data: Vec<u8> = sp
            .get("data")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().map(|x| x.as_i64().unwrap_or(0).clamp(0, 2) as u8).collect())
            .unwrap_or_default();
        if w > 0 && h > 0 && data.len() >= w * h {
            out[i] = crate::framebuffer::CursorSprite { w, h, data };
            any = true;
        }
    }
    if any {
        Some(out)
    } else {
        None
    }
}

/// Load only the active theme's cursor sprites into the framebuffer. Sprites
/// live in the theme file (not `ui.json`), so this restores a custom cursor at
/// boot from `ui.json`'s `theme_name`. A bundled/colours-only theme clears to
/// the built-ins.
#[cfg(not(test))]
pub fn apply_cursor_sprites(name: &str) {
    let sprites = load_text(name)
        .and_then(|t| Json::parse(&t))
        .and_then(|j| parse_cursor_sprites(j.get("cursor")));
    crate::framebuffer::set_cursor_sprites(sprites);
}

/// Export the current live appearance as a theme JSON in the store
/// (`/configs/themes/<name>.json`). Returns the written path.
pub fn save(name: &str) -> Result<String, String> {
    if name.is_empty() || name.contains('/') {
        return Err(String::from("invalid theme name"));
    }
    let cfg = crate::ui_config::current();
    let obj = |pairs: &[(String, String)]| {
        Json::Obj(pairs.iter().map(|(k, v)| (k.clone(), Json::Str(v.clone()))).collect())
    };
    let cursor = Json::Obj(alloc::vec![
        ("fill".to_string(), Json::Str(cfg.cursor_fill.clone())),
        ("outline".to_string(), Json::Str(cfg.cursor_outline.clone())),
    ]);
    let j = Json::Obj(alloc::vec![
        ("name".to_string(), Json::Str(name.to_string())),
        ("font".to_string(), Json::Str(cfg.font.clone())),
        ("font_scale".to_string(), Json::Num(cfg.font_scale as f64)),
        ("wallpaper".to_string(), Json::Str(cfg.wallpaper.clone())),
        ("opacity".to_string(), Json::Num(cfg.opacity as f64)),
        ("cursor".to_string(), cursor),
        ("palette".to_string(), obj(&cfg.theme)),
        ("syntax".to_string(), obj(&cfg.syntax)),
    ]);
    let path = format!("{THEMES_DIR}/{name}.json");
    crate::synapse::fs::write(&path, j.to_pretty().as_bytes());
    Ok(path)
}

/// Fetch a theme JSON over HTTP(S) and install it into the store. Returns the
/// installed theme name (from the JSON's `"name"`). Pure data — no signature.
pub fn install(url: &str) -> Result<String, String> {
    let resp = crate::net::http::get(url, 15_000).map_err(|e| format!("fetch failed: {e}"))?;
    if resp.status != 200 {
        return Err(format!("server returned status {}", resp.status));
    }
    let text = resp.text();
    let j = Json::parse(&text).ok_or_else(|| String::from("invalid theme JSON"))?;
    let name = j
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| String::from("theme has no \"name\" field"))?;
    if name.is_empty() || name.contains('/') {
        return Err(String::from("theme has an invalid \"name\""));
    }
    // Require at least a palette or syntax block so we don't install garbage.
    if j.get("palette").is_none() && j.get("syntax").is_none() {
        return Err(String::from("theme has no palette/syntax — not a theme"));
    }
    let path = format!("{THEMES_DIR}/{name}.json");
    crate::synapse::fs::write(&path, text.as_bytes());
    Ok(name.to_string())
}

/// How far a derived surface tint moves from the background toward the foreground, in
/// percent.
///
/// 8% is the smallest step that is actually visible as a panel. The first attempt derived
/// the tool-call tint as the midpoint of `chat_bg` and `composer_bg`, which on the brand
/// dark palette differ by **6 units** — a 3-unit lift, invisible on a real display, so the
/// feature looked like it had not shipped.
const TINT_PCT: u16 = 8;

/// `bg` lifted [`TINT_PCT`] of the way toward `fg`.
///
/// Used to **derive** a theme colour a palette omits, so an older theme file gains a new
/// surface in its own colours rather than inheriting the brand default. Blending toward
/// the *text* colour is what makes it direction-agnostic: on a dark theme the result is
/// lighter, on a light theme darker, which is "slightly raised" in both cases.
///
/// Lives here rather than beside its caller in `framebuffer/colors.rs` because that whole
/// tree is `#[cfg(not(test))]`: colour maths written there cannot be tested at all.
pub fn tint_toward(bg: (u8, u8, u8), fg: (u8, u8, u8)) -> (u8, u8, u8) {
    // u16 throughout: `x * 8` overflows a u8 well before the percentage is applied, and
    // wrapping would send a light theme's tint to near-black.
    let f = |x: u8, y: u8| -> u8 {
        let (x, y) = (x as u16, y as u16);
        let v = if y >= x {
            x + (y - x) * TINT_PCT / 100
        } else {
            x - (x - y) * TINT_PCT / 100
        };
        v.min(255) as u8
    };
    (f(bg.0, fg.0), f(bg.1, fg.1), f(bg.2, fg.2))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A derived tint must be **visible** and must move the right way on any palette.
    ///
    /// Both halves were wrong once. Deriving it from `chat_bg`/`composer_bg` gave a 3-unit
    /// lift on the brand palette — invisible, so the panel looked like it never shipped —
    /// and a *fixed* fallback would paint the brand's dark tint behind tool calls on the
    /// `light` theme, since none of the six bundled themes declares the key.
    #[test_case]
    fn a_derived_tint_is_visible_and_moves_toward_the_text_on_any_palette() {
        // Brand dark: chat_bg #1f1e1b lifted toward cream #faf9f5 — lighter, and by
        // enough to see. The old midpoint-of-two-surfaces gave (34, 32, 29).
        let dark = tint_toward((31, 30, 27), (250, 249, 245));
        assert_eq!(dark, (48, 47, 44));
        assert!(dark.0 > 31 && dark.1 > 30 && dark.2 > 27, "dark theme lifts lighter");

        // The light theme moves the other way — toward its dark text — and stays light.
        let light = tint_toward((250, 247, 240), (35, 33, 30));
        assert!(light.0 < 250 && light.1 < 247 && light.2 < 240, "light theme darkens");
        assert!(light.0 > 200 && light.1 > 200 && light.2 > 200, "but stays light");

        // Visible on both: a lift under ~8 units reads as no band at all.
        for (bg, fg) in [((31, 30, 27), (250, 249, 245)), ((250, 247, 240), (35, 33, 30))] {
            let t = tint_toward(bg, fg);
            let delta = (t.0 as i32 - bg.0 as i32).abs();
            assert!(delta >= 8, "tint delta {delta} is too small to see");
        }

        // Never leaves the interval, in either direction.
        for (bg, fg) in [((0, 0, 0), (255, 255, 255)), ((255, 255, 255), (0, 0, 0)),
                         ((10, 200, 90), (240, 12, 130))] {
            let t = tint_toward(bg, fg);
            for (lo, hi, got) in [
                (bg.0.min(fg.0), bg.0.max(fg.0), t.0),
                (bg.1.min(fg.1), bg.1.max(fg.1), t.1),
                (bg.2.min(fg.2), bg.2.max(fg.2), t.2),
            ] {
                assert!(got >= lo && got <= hi, "{got} outside {lo}..={hi}");
            }
        }
        // The overflow case the u16 maths exists for: maxed channels stay maxed.
        assert_eq!(tint_toward((255, 255, 255), (255, 255, 255)), (255, 255, 255));
        // A palette whose text and background match gets no band rather than a
        // wrapped-around colour.
        assert_eq!(tint_toward((37, 35, 32), (37, 35, 32)), (37, 35, 32));
    }

    #[test_case]
    fn bundled_themes_parse_and_have_palettes() {
        for (name, text) in BUNDLED_THEMES {
            let j = Json::parse(text).unwrap_or_else(|| panic!("{name} JSON parse"));
            assert!(j.get("palette").is_some(), "{name} has a palette");
            assert!(j.get("syntax").is_some(), "{name} has syntax");
            assert_eq!(
                j.get("name").and_then(|v| v.as_str()),
                Some(*name),
                "{name} name field matches"
            );
            // Status chrome + Synapse-C logo slots must be explicit so a theme
            // switch recolors the bar (overlay_pairs only updates present keys).
            let pal = j.get("palette").unwrap();
            for key in ["status_bg", "status_fg", "logo", "logo_node"] {
                assert!(
                    pal.get(key).and_then(|v| v.as_str()).is_some(),
                    "{name} palette missing {key}"
                );
            }
        }
    }

    #[test_case]
    fn dark_theme_matches_brand_status_and_logo() {
        let text = load_text("dark").expect("dark theme");
        let j = Json::parse(&text).expect("parse");
        let pal = j.get("palette").unwrap();
        let get = |k: &str| pal.get(k).and_then(|v| v.as_str());
        assert_eq!(get("status_bg"), Some("#252320"));
        assert_eq!(get("status_fg"), Some("#a09d96"));
        assert_eq!(get("logo"), Some("#cc785c"));
        assert_eq!(get("logo_node"), Some("#faf9f5"));
    }

    #[test_case]
    fn overlay_pairs_overrides_only_present_keys() {
        let mut pairs = alloc::vec![
            ("a".to_string(), "#111111".to_string()),
            ("b".to_string(), "#222222".to_string()),
        ];
        let j = Json::parse(r##"{"a":"#ff0000"}"##).unwrap();
        overlay_pairs(Some(&j), &mut pairs);
        assert_eq!(pairs[0].1, "#ff0000"); // overridden
        assert_eq!(pairs[1].1, "#222222"); // kept
    }

    #[test_case]
    fn load_text_returns_bundled() {
        assert!(load_text("nord").is_some());
        assert!(load_text("no-such-theme").is_none());
    }
}
