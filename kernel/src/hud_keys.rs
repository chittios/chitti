//! Parse HUD shortcut lines into **key chips + descriptions**.
//!
//! Agent HUDs (`host_hud_set`) and the media players all ship the same shape of
//! hint text — `"space play/pause  n/p next  r repeat"` — and the compositor
//! paints each key in a chip the way the CLIAMP-style player does. The split
//! and the wrap live here so they are unit-tested: `framebuffer/` is
//! `cfg(not(test))`.

use alloc::string::String;
use alloc::vec::Vec;

/// One painted token on a HUD hint row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HintItem {
    /// Highlighted key, optional dim description after it.
    Chip { key: String, desc: String },
    /// A visual separator (`·`) between groups.
    Sep,
    /// Plain dim text (a group that did not start with a key).
    Text(String),
}

/// Status line (everything before the first newline) plus the parsed hint
/// items. A HUD with no newline is status-only.
pub fn parse_hud(hud: &str) -> (&str, Vec<HintItem>) {
    match hud.split_once('\n') {
        Some((status, rest)) => (status, parse_hints(rest)),
        None => (hud, Vec::new()),
    }
}

/// Parse one or more hint lines. Groups are separated by two or more spaces
/// (the convention every `hud_status` call already uses).
pub fn parse_hints(s: &str) -> Vec<HintItem> {
    let mut out = Vec::new();
    for line in s.split('\n') {
        for group in line.split("  ").filter(|g| !g.trim().is_empty()) {
            parse_group(group.trim(), &mut out);
        }
    }
    out
}

/// Columns one item occupies, including the trailing gap. Must stay in lockstep
/// with the painter (`draw_hint_row`) or wrap and paint disagree.
pub fn item_cols(item: &HintItem) -> usize {
    match item {
        HintItem::Chip { key, desc } => {
            let k = key.chars().count() + 1; // ½-cell pad on each side
            let d = if desc.is_empty() {
                0
            } else {
                1 + desc.chars().count()
            };
            k + d + 1
        }
        HintItem::Sep => 2,
        HintItem::Text(s) => s.chars().count() + 1,
    }
}

/// Wrap `items` to `cols` display columns. An item wider than `cols` still
/// takes a row of its own (the painter ellipsizes).
pub fn wrap_items(items: &[HintItem], cols: usize) -> Vec<Vec<HintItem>> {
    let cols = cols.max(1);
    let mut rows: Vec<Vec<HintItem>> = Vec::new();
    let mut row: Vec<HintItem> = Vec::new();
    let mut used = 0usize;
    for item in items {
        let w = item_cols(item).max(1);
        if !row.is_empty() && used + w > cols {
            rows.push(row);
            row = Vec::new();
            used = 0;
        }
        used += w;
        row.push(item.clone());
    }
    if !row.is_empty() {
        rows.push(row);
    }
    rows
}

/// Wrap hint text, honouring explicit newlines as row breaks (then wrapping
/// each line to `cols`).
pub fn hint_rows(s: &str, cols: usize) -> Vec<Vec<HintItem>> {
    let mut rows = Vec::new();
    for line in s.split('\n') {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        rows.extend(wrap_items(&parse_hints(line), cols));
    }
    rows
}

/// How many hint rows `hud` needs at `cols` once keys are chipped.
/// Status-only HUDs return 0 so the reserved strip stays one line.
pub fn wrapped_hint_rows(hud: &str, cols: usize) -> u64 {
    let rest = match hud.split_once('\n') {
        Some((_, r)) => r,
        None => return 0,
    };
    if rest.trim().is_empty() {
        return 0;
    }
    hint_rows(rest, cols).len().max(1) as u64
}

fn parse_group(group: &str, out: &mut Vec<HintItem>) {
    if group == "·" || group == "•" {
        out.push(HintItem::Sep);
        return;
    }
    let tokens: Vec<&str> = group.split_whitespace().collect();
    if tokens.is_empty() {
        return;
    }
    let mut keys: Vec<String> = Vec::new();
    let mut rest: Vec<&str> = Vec::new();
    let mut in_keys = true;
    for t in tokens {
        if is_sep_token(t) {
            if in_keys && !keys.is_empty() {
                continue;
            }
            in_keys = false;
            rest.push(t);
        } else if in_keys && looks_like_key(t) {
            keys.push(pretty_key(t));
        } else {
            in_keys = false;
            rest.push(t);
        }
    }
    if keys.is_empty() {
        out.push(HintItem::Text(rest.join(" ")));
        return;
    }
    let desc = rest.join(" ");
    let last = keys.len() - 1;
    for (i, key) in keys.into_iter().enumerate() {
        let d = if i == last {
            desc.clone()
        } else {
            String::new()
        };
        out.push(HintItem::Chip { key, desc: d });
    }
}

fn is_sep_token(t: &str) -> bool {
    matches!(t, "·" | "•" | "/" | "|")
}

fn looks_like_key(t: &str) -> bool {
    if t.is_empty() || is_sep_token(t) {
        return false;
    }
    let mut buf = [0u8; 16];
    let n = t.len().min(16);
    for (i, b) in t.as_bytes().iter().take(n).enumerate() {
        buf[i] = b.to_ascii_lowercase();
    }
    let lower = core::str::from_utf8(&buf[..n]).unwrap_or("");
    if matches!(
        lower,
        "space"
            | "enter"
            | "esc"
            | "escape"
            | "tab"
            | "bksp"
            | "backspace"
            | "home"
            | "pgup"
            | "pgdn"
            | "arrows"
            | "arrow"
            | "up"
            | "dn"
            | "down"
            | "left"
            | "right"
            | "wheel"
            | "click"
            | "shift"
            | "alt"
            | "ctrl"
            | "cmd"
    ) {
        return true;
    }
    if lower.starts_with("ctrl+") || lower.starts_with("cmd+") || lower.starts_with("alt+") || lower.starts_with("shift+")
    {
        return true;
    }
    // `arrows/click`, `n/p`, `→/space/enter`: every side of a slash must itself
    // be a key, or "play/pause" (a description) would light up as one.
    if t.contains('/') {
        return t.split('/').all(|p| !p.is_empty() && looks_like_key(p));
    }
    if t.contains('+') && t.chars().count() <= 12 {
        return true;
    }
    looks_like_key_atom(t)
}

fn looks_like_key_atom(t: &str) -> bool {
    if t.is_empty() || is_sep_token(t) {
        return false;
    }
    let mut buf = [0u8; 16];
    let n = t.len().min(16);
    for (i, b) in t.as_bytes().iter().take(n).enumerate() {
        buf[i] = b.to_ascii_lowercase();
    }
    let lower = core::str::from_utf8(&buf[..n]).unwrap_or("");
    if matches!(
        lower,
        "space"
            | "enter"
            | "esc"
            | "escape"
            | "tab"
            | "bksp"
            | "backspace"
            | "home"
            | "pgup"
            | "pgdn"
            | "arrows"
            | "arrow"
            | "up"
            | "dn"
            | "down"
            | "left"
            | "right"
            | "wheel"
            | "click"
            | "shift"
            | "alt"
            | "ctrl"
            | "cmd"
    ) {
        return true;
    }
    let chars = t.chars().count();
    if !t.chars().all(is_key_char) {
        return false;
    }
    if t.chars().any(|c| !c.is_ascii_alphabetic()) {
        return chars <= 7;
    }
    // Bare letters: only the 1–2 char bindings (`n`, `p`, `up`). Longer words
    // are descriptions (`move`, `type`, `select`).
    chars <= 2
}

fn is_key_char(c: char) -> bool {
    c.is_ascii_alphanumeric()
        || matches!(
            c,
            '+' | '-' | '/' | '=' | '[' | ']' | '<' | '>' | ',' | '.' | '↑' | '↓' | '←' | '→'
        )
}

fn pretty_key(raw: &str) -> String {
    let mut buf = [0u8; 16];
    let n = raw.len().min(16);
    for (i, b) in raw.as_bytes().iter().take(n).enumerate() {
        buf[i] = b.to_ascii_lowercase();
    }
    let lower = core::str::from_utf8(&buf[..n]).unwrap_or("");
    match lower {
        "space" => String::from("Space"),
        "enter" => String::from("Enter"),
        "esc" | "escape" => String::from("Esc"),
        "tab" => String::from("Tab"),
        "bksp" | "backspace" => String::from("Bksp"),
        "home" => String::from("Home"),
        "pgup" => String::from("PgUp"),
        "pgdn" => String::from("PgDn"),
        "arrows" | "arrow" => String::from("Arrows"),
        "wheel" => String::from("Wheel"),
        "click" => String::from("Click"),
        _ => {
            if let Some((modi, rest)) = raw.split_once('+') {
                let m = modi.to_ascii_lowercase();
                if matches!(m.as_str(), "ctrl" | "cmd" | "alt" | "shift") {
                    let head = match m.as_str() {
                        "ctrl" => "Ctrl",
                        "cmd" => "Cmd",
                        "alt" => "Alt",
                        _ => "Shift",
                    };
                    return alloc::format!("{head}+{}", rest.to_ascii_uppercase());
                }
            }
            String::from(raw)
        }
    }
}

/// Build chips from already-split `(key, desc)` pairs — the video / browser /
/// audio HUDs that do not go through free-form hint text.
pub fn from_pairs(pairs: &[(&str, &str)]) -> Vec<HintItem> {
    pairs
        .iter()
        .map(|(k, d)| HintItem::Chip {
            key: pretty_key(k),
            desc: String::from(*d),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chip(k: &str, d: &str) -> HintItem {
        HintItem::Chip {
            key: String::from(k),
            desc: String::from(d),
        }
    }

    #[test_case]
    fn parse_hud_splits_status_from_hints() {
        let (st, items) =
            parse_hud("Your move (White)\narrows/click move  enter select  esc clear  n new game");
        assert_eq!(st, "Your move (White)");
        assert_eq!(
            items,
            alloc::vec![
                chip("arrows/click", "move"),
                chip("Enter", "select"),
                chip("Esc", "clear"),
                chip("n", "new game"),
            ]
        );
        assert!(parse_hud("status only").1.is_empty());
    }

    #[test_case]
    fn a_run_of_letter_keys_keeps_the_last_word_as_the_label() {
        let items = parse_hints("a s d f g h j k white  ·  w e t y u black");
        assert_eq!(items[0], chip("a", ""));
        assert_eq!(items[7], chip("k", "white"));
        assert_eq!(items[8], HintItem::Sep);
        assert_eq!(items.last(), Some(&chip("u", "black")));
    }

    #[test_case]
    fn slash_separated_names_stay_one_chip() {
        let items = parse_hints("n/p next  <-/-> seek  +/- volume  arrows/click move");
        assert_eq!(items[0], chip("n/p", "next"));
        assert_eq!(items[1], chip("<-/->", "seek"));
        assert_eq!(items[2], chip("+/-", "volume"));
        assert_eq!(items[3], chip("arrows/click", "move"));
    }

    #[test_case]
    fn spaced_slashes_are_several_keys() {
        let items = parse_hints("enter / n / r  new game");
        assert_eq!(
            items,
            alloc::vec![
                chip("Enter", ""),
                chip("n", ""),
                chip("r", ""),
                HintItem::Text(String::from("new game")),
            ]
        );
    }

    #[test_case]
    fn ctrl_combo_is_pretty_printed() {
        let items = parse_hints("Ctrl+C stop  ctrl+k help");
        assert_eq!(items[0], chip("Ctrl+C", "stop"));
        assert_eq!(items[1], chip("Ctrl+K", "help"));
    }

    #[test_case]
    fn a_group_with_no_key_stays_plain_text() {
        let items = parse_hints("type letters  enter lookup");
        assert_eq!(items[0], HintItem::Text(String::from("type letters")));
        assert_eq!(items[1], chip("Enter", "lookup"));
    }

    #[test_case]
    fn wrap_breaks_before_an_item_that_will_not_fit() {
        let items = from_pairs(&[("Space", "Play/Pause"), ("n/p", "Next/Prev")]);
        // "Space"(5+1) + desc 10 + gap = 17; next chip is ~12. 20 cols → 2 rows.
        let rows = wrap_items(&items, 20);
        assert_eq!(rows.len(), 2, "{rows:?}");
        assert_eq!(rows[0].len(), 1);
        assert_eq!(rows[1].len(), 1);
        assert_eq!(wrap_items(&items, 80).len(), 1);
    }

    #[test_case]
    fn wrapped_hint_rows_is_zero_without_hints() {
        assert_eq!(wrapped_hint_rows("just a status", 40), 0);
        assert_eq!(wrapped_hint_rows("status\nn new", 40), 1);
        let hud = "status\narrows move  enter select  esc clear  n new game";
        assert_eq!(wrapped_hint_rows(hud, 80), 1);
        assert!(wrapped_hint_rows(hud, 20) >= 3);
        assert!(wrapped_hint_rows("s\nfirst line\nsecond line", 80) >= 2);
    }

    #[test_case]
    fn item_cols_counts_pad_desc_and_gap() {
        assert_eq!(item_cols(&chip("n", "")), 1 + 1 + 1);
        assert_eq!(item_cols(&chip("Space", "Play")), 6 + 1 + 4 + 1);
        assert_eq!(item_cols(&HintItem::Sep), 2);
    }
}
