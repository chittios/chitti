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

#[cfg(test)]
mod tests {
    use super::*;

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
