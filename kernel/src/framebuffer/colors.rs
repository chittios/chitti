//! Console palette: the [`Theme`] struct, hex parsing, and the ANSI
//! colour tables.

use super::*;

/// The console colour palette (see [DESIGN.md](../../DESIGN.md)). Every colour is
/// a field so the whole theme is configurable from `/configs/core/ui.json`
/// (`theme` object, hex strings); the default is the Chitti brand **dark**
/// theme — terracotta `#cc785c` primary on warm-ink surfaces, cream text.
#[derive(Clone, Copy)]
pub struct Theme {
    pub screen_bg: Rgb,
    pub chat_bg: Rgb,
    pub logs_bg: Rgb,
    pub chat_fg: Rgb,
    pub logs_fg: Rgb,
    pub accent: Rgb, // active border / caret / selection chrome
    /// Status-bar / splash Synapse-C logo **ring** (from `ui.json` `theme.logo`;
    /// defaults to `accent` when omitted).
    pub logo: Rgb,
    /// Status-bar / splash logo **node** (from `theme.logo_node`; defaults to
    /// `chat_fg` when omitted).
    pub logo_node: Rgb,
    pub border_dim: Rgb,
    pub title_active: Rgb,
    pub title_dim: Rgb,
    pub sep_dim: Rgb,
    pub status_bg: Rgb,
    pub status_fg: Rgb,
    pub editor_bg: Rgb,
    pub editor_fg: Rgb,
    pub editor_lineno: Rgb,
    pub editor_sel: Rgb,
    /// bordered input composer fill (slightly elevated over chat_bg).
    pub composer_bg: Rgb,
    /// Composer border when idle / focused (focused uses `accent`).
    pub composer_border: Rgb,
    /// Hint-bar text under the composer.
    pub composer_hint: Rgb,
    /// Background of an agent **tool-call block** in the chat pane.
    ///
    /// A third elevation, deliberately between `chat_bg` and the user-prompt band's
    /// `composer_bg`: the two bands sit next to each other in the transcript, so
    /// reusing one colour would make a tool call look like something the human typed.
    /// Lower elevation reads as secondary, which is what a tool call is.
    pub tool_bg: Rgb,
}

impl Theme {
    /// The Chitti brand dark theme: `#cc785c` terracotta on warm-ink surfaces,
    /// cream (`#faf9f5`) text.
    pub const BRAND_DARK: Theme = Theme {
        screen_bg: (24, 23, 21),       // surface-dark #181715
        chat_bg: (31, 30, 27),         // surface-dark-soft #1f1e1b
        logs_bg: (20, 19, 17),         // a touch darker than the chat pane
        chat_fg: (250, 249, 245),      // on-dark / cream #faf9f5
        logs_fg: (160, 157, 150),      // on-dark-soft #a09d96
        accent: (204, 120, 92),        // primary #cc785c
        logo: (204, 120, 92),          // matches accent unless ui.json overrides
        logo_node: (250, 249, 245),    // cream node
        border_dim: (76, 73, 66),      // inactive border — lifted for a crisp edge on dark bg
        title_active: (204, 120, 92),  // primary
        title_dim: (108, 106, 100),    // muted #6c6a64
        sep_dim: (46, 44, 40),
        status_bg: (37, 35, 32),       // surface-dark-elevated #252320
        status_fg: (160, 157, 150),    // on-dark-soft — icons + status text
        editor_bg: (31, 30, 27),       // surface-dark-soft
        editor_fg: (250, 249, 245),    // cream
        editor_lineno: (108, 106, 100),
        editor_sel: (90, 58, 46),      // terracotta-tinted selection
        composer_bg: (37, 35, 32),     // elevated like status_bg
        composer_border: (76, 73, 66), // matches border_dim when unfocused
        composer_hint: (108, 106, 100), // muted
        tool_bg: (48, 47, 44),         // chat_bg lifted 8% toward the cream text
    };
}

impl Default for Theme {
    fn default() -> Self {
        Theme::BRAND_DARK
    }
}

/// Parse a `#rrggbb` (or `rrggbb`) hex colour, falling back to `def`.
pub fn parse_hex(s: &str, def: Rgb) -> Rgb {
    let h = s.trim().trim_start_matches('#');
    if h.len() != 6 {
        return def;
    }
    let b = |i: usize| u8::from_str_radix(&h[i..i + 2], 16);
    match (b(0), b(2), b(4)) {
        (Ok(r), Ok(g), Ok(bl)) => (r, g, bl),
        _ => def,
    }
}

/// Build a [`Theme`] from `(name, "#rrggbb")` pairs, starting from the brand dark
/// default and overriding any named field. Unknown names are ignored; malformed
/// hex keeps the brand value. This is how `ui.json`'s `theme` object is applied.
pub fn theme_from_pairs(pairs: &[(alloc::string::String, alloc::string::String)]) -> Theme {
    let mut t = Theme::BRAND_DARK;
    let mut has_logo = false;
    let mut has_logo_node = false;
    let mut has_tool_bg = false;
    for (name, hex) in pairs {
        let slot = match name.as_str() {
            "screen_bg" => &mut t.screen_bg,
            "chat_bg" => &mut t.chat_bg,
            "logs_bg" => &mut t.logs_bg,
            "chat_fg" => &mut t.chat_fg,
            "logs_fg" => &mut t.logs_fg,
            "accent" => &mut t.accent,
            "logo" => {
                has_logo = true;
                &mut t.logo
            }
            "logo_node" => {
                has_logo_node = true;
                &mut t.logo_node
            }
            "border_dim" => &mut t.border_dim,
            "title_active" => &mut t.title_active,
            "title_dim" => &mut t.title_dim,
            "sep_dim" => &mut t.sep_dim,
            "status_bg" => &mut t.status_bg,
            "status_fg" => &mut t.status_fg,
            "editor_bg" => &mut t.editor_bg,
            "editor_fg" => &mut t.editor_fg,
            "editor_lineno" => &mut t.editor_lineno,
            "editor_sel" => &mut t.editor_sel,
            "composer_bg" => &mut t.composer_bg,
            "composer_border" => &mut t.composer_border,
            "composer_hint" => &mut t.composer_hint,
            "tool_bg" => {
                has_tool_bg = true;
                &mut t.tool_bg
            }
            _ => continue,
        };
        *slot = parse_hex(hex, *slot);
    }
    // Omitted logo keys track the brand palette so a theme that only sets
    // `accent` / `chat_fg` still recolors the mark without a second key.
    if !has_logo {
        t.logo = t.accent;
    }
    if !has_logo_node {
        t.logo_node = t.chat_fg;
    }
    // Same rule as the logo keys: an omitted `tool_bg` is **derived**, never left at the
    // brand default. Every bundled theme (and every theme a user writes) predates this
    // key, so a fixed fallback would put a dark tint behind tool calls on `light`. The
    // midpoint between the pane and the user band gives each theme its own third
    // elevation, in its own palette, for free.
    if !has_tool_bg {
        // The blend itself lives in `crate::theme` — `framebuffer/` is
        // `#[cfg(not(test))]`, so colour maths written here cannot be unit-tested at all
        // (a `#[cfg(test)] mod` in this tree is silently never compiled).
        t.tool_bg = crate::theme::tint_toward(t.chat_bg, t.chat_fg);
    }
    t
}

// Layout metrics, in pixels (independent of font scale).

/// The 8 ANSI foreground colours (and their bright variants), tuned to read well
/// on the dark pane background.
pub(super) fn ansi_color(idx: usize, bright: bool) -> Rgb {
    const NORMAL: [Rgb; 8] = [
        (98, 104, 118),  // "black" -> dim gray (pure black is invisible here)
        (255, 106, 110), // red
        (126, 214, 150), // green
        (240, 200, 120), // yellow
        (94, 161, 255),  // blue (the accent)
        (200, 140, 255), // magenta
        (110, 214, 224), // cyan
        (232, 233, 238), // white (the default fg)
    ];
    const BRIGHT: [Rgb; 8] = [
        (140, 148, 162),
        (255, 140, 150),
        (170, 240, 190),
        (255, 224, 150),
        (150, 190, 255),
        (220, 170, 255),
        (150, 235, 245),
        (255, 255, 255),
    ];
    (if bright { &BRIGHT } else { &NORMAL })[idx & 7]
}

/// Map an ANSI 256-colour index to RGB: 0–15 the base/bright palette, 16–231 the
/// 6×6×6 colour cube, 232–255 the 24-step grayscale ramp.
pub(super) fn ansi_256(n: u8) -> Rgb {
    match n {
        0..=7 => ansi_color(n as usize, false),
        8..=15 => ansi_color((n - 8) as usize, true),
        16..=231 => {
            let c = n - 16;
            let steps = [0u8, 95, 135, 175, 215, 255];
            (steps[(c / 36) as usize], steps[((c / 6) % 6) as usize], steps[(c % 6) as usize])
        }
        _ => {
            let v = 8 + (n - 232) * 10;
            (v, v, v)
        }
    }
}
