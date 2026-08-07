//! keyboard
//!
//! The `/keyboard` command: layout selection, the input method, and — the reason
//! it exists in this shape — a way to **exercise the real translation path from a
//! serial console**.
//!
//! Scancodes cannot be injected over serial, and QEMU's monitor `sendkey` is not
//! wired into the e2e harness, so without a command surface the layout tables
//! would be unit-testable and nothing more. `/keyboard test de altgr+q` runs the
//! actual [`crate::keymap::translate`] over the actual layout data — never a
//! parallel path — which is what makes "AltGr+Q types @ on German" an assertion a
//! test can make on a running kernel.

use super::*;
use crate::keymap::{self, layouts, KeyEvent, Layout, Mods, Out, Source, State};

/// `/keyboard [list|set|test|dead|compose|hex|ime|type|echo]`
pub(super) fn run_keyboard(arg: &str) {
    let a = arg.trim();
    let mut parts = a.splitn(2, char::is_whitespace);
    let verb = parts.next().unwrap_or("");
    let rest = parts.next().unwrap_or("").trim();

    match verb {
        "" | "status" | "info" => status(),
        "list" | "ls" => list(),
        "set" => set(rest),
        "test" => test(rest),
        "dead" => dead(rest),
        "compose" => compose_table(),
        "hex" => hex(rest),
        "ime" => ime(rest),
        "type" => type_through_ime(rest),
        "echo" => echo(rest),
        other => {
            serial_println!("keyboard> unknown '{other}'");
            usage();
        }
    }
}

fn usage() {
    serial_println!("keyboard> usage:");
    serial_println!("  /keyboard                    status: layout, ime, pending composition");
    serial_println!("  /keyboard list               every layout");
    serial_println!("  /keyboard set <id>           apply + persist (us|uk|de|fr|es|it|se|dvorak|colemak)");
    serial_println!("  /keyboard test <id> <keys>   run the real translator; keys are US base chars,");
    serial_println!("                               comma-separated, with shift+/ctrl+/altgr+ prefixes");
    serial_println!("  /keyboard dead <id> <keys>   the same, showing the dead-key state per step");
    serial_println!("  /keyboard compose            the whole compose table");
    serial_println!("  /keyboard hex <hex>          the character at a codepoint, or why not");
    serial_println!("  /keyboard ime <mode>         off|hiragana|katakana|hangul (pinyin is refused)");
    serial_println!("  /keyboard type <ascii>       feed text through the live IME");
    serial_println!("  /keyboard echo <text>        echo it back with char/byte/column counts");
    serial_println!("  key names: space enter tab esc bs up down left right home end pgup pgdn del");
    serial_println!("             iso f1..f12 prtsc, or any US base character");
}

fn status() {
    let l = keymap::active_layout();
    let st = keymap::state_snapshot();
    serial_println!("keyboard> layout {} ({})", l.id, l.name);
    serial_println!("  caps lock    {}", if st.caps { "on" } else { "off" });
    serial_println!(
        "  right alt    {}",
        if l.altgr_is_compose { "Compose" } else { "AltGr (level 3)" }
    );
    let dks = l.dead_keys();
    if dks.is_empty() {
        serial_println!("  dead keys    none on this layout");
    } else {
        let names: alloc::vec::Vec<&str> = dks.iter().map(|d| layouts::dead_name(*d)).collect();
        serial_println!("  dead keys    {}", names.join(", "));
    }
    serial_println!("  ime          {}", crate::ime::mode_name(crate::ime::mode()));
    match keymap::pending_description(&st) {
        Some(p) => serial_println!("  composing    {p}"),
        None => serial_println!("  composing    nothing"),
    }
    let pre = crate::ime::preedit();
    if !pre.is_empty() {
        serial_println!("  preedit      {pre}");
    }
    serial_println!("  cmds: list | set <id> | test <id> <keys> | ime <mode> | hex <hex> | compose");
}

fn list() {
    serial_println!("keyboard> {} layout(s):", layouts::LAYOUTS.len());
    let active = keymap::active_layout().id;
    for l in layouts::LAYOUTS {
        let mark = if l.id == active { '*' } else { ' ' };
        let dks = l.dead_keys();
        let dead = if dks.is_empty() {
            String::from("-")
        } else {
            let n: alloc::vec::Vec<&str> = dks.iter().map(|d| layouts::dead_name(*d)).collect();
            n.join(",")
        };
        serial_println!(
            "  {mark}{:<9} {:<22} altgr={:<3} compose={:<3} dead={dead}",
            l.id,
            l.name,
            if l.has_altgr_levels() { "yes" } else { "no" },
            if l.altgr_is_compose { "yes" } else { "no" },
        );
    }
}

fn set(id: &str) {
    let id = id.trim();
    if id.is_empty() {
        serial_println!("keyboard> usage: /keyboard set <id> — see /keyboard list");
        return;
    }
    if !keymap::set_layout(id) {
        serial_println!("keyboard> unknown layout '{id}' (try /keyboard list)");
        return;
    }
    crate::ui_config::persist_kbd_layout(id);
    let l = keymap::active_layout();
    serial_println!("keyboard> layout {} ({})", l.id, l.name);
    if l.altgr_is_compose {
        serial_println!("keyboard> right Alt is Compose on this layout (/keyboard compose)");
    } else {
        serial_println!("keyboard> right Alt selects the AltGr level on this layout");
    }
}

/// Resolve a layout by id, reporting the failure.
fn layout_by_id(id: &str) -> Option<&'static Layout> {
    match layouts::LAYOUTS.iter().find(|l| l.id == id) {
        Some(l) => Some(l),
        None => {
            serial_println!("keyboard> unknown layout '{id}' (try /keyboard list)");
            None
        }
    }
}

/// A key spec token: `q`, `shift+2`, `altgr+q`, `ctrl+c`.
///
/// The key is named by its **US base character**, which is unambiguous, typeable
/// over serial, and exercises exactly the usage→layout path a driver uses: the
/// spec resolves to a HID usage through `US_BASE`, and then the *target* layout
/// is asked what that physical key produces.
fn parse_spec(tok: &str) -> Option<(keymap::Usage, Mods)> {
    let mut bits = 0u8;
    let mut rest = tok;
    loop {
        let Some((pre, after)) = rest.split_once('+') else { break };
        match pre.trim().to_ascii_lowercase().as_str() {
            "shift" | "s" => bits |= Mods::SHIFT,
            "ctrl" | "c" => bits |= Mods::CTRL,
            "alt" => bits |= Mods::ALT,
            "altgr" | "ralt" | "ag" => bits |= Mods::ALTGR,
            "gui" | "cmd" | "super" | "win" => bits |= Mods::GUI,
            "caps" => bits |= Mods::CAPS,
            other => {
                serial_println!("keyboard> unknown modifier '{other}'");
                return None;
            }
        }
        rest = after;
        if rest.is_empty() {
            // `shift+` with nothing after: `+` is itself a key, so treat the
            // trailing empty piece as the plus key.
            rest = "+";
            break;
        }
    }
    // Named keys that have no single-character form.
    let named: Option<keymap::Usage> = match rest.trim().to_ascii_lowercase().as_str() {
        "space" => Some(0x2c),
        "enter" | "return" => Some(0x28),
        "tab" => Some(0x2b),
        "esc" => Some(0x29),
        "bs" | "backspace" => Some(0x2a),
        "up" => Some(0x52),
        "down" => Some(0x51),
        "left" => Some(0x50),
        "right" => Some(0x4f),
        "home" => Some(0x4a),
        "end" => Some(0x4d),
        "pgup" => Some(0x4b),
        "pgdn" => Some(0x4e),
        "del" | "delete" => Some(0x4c),
        // The ISO key left of Z/Y, which has no US character at all — the whole
        // reason a usage-based canonical space was needed.
        "iso" | "102nd" => Some(0x64),
        // Function keys, and Print Screen (which folds onto F12's sequence).
        "f1" => Some(0x3a),
        "f2" => Some(0x3b),
        "f3" => Some(0x3c),
        "f4" => Some(0x3d),
        "f5" => Some(0x3e),
        "f6" => Some(0x3f),
        "f7" => Some(0x40),
        "f8" => Some(0x41),
        "f9" => Some(0x42),
        "f10" => Some(0x43),
        "f11" => Some(0x44),
        "f12" => Some(0x45),
        "prtsc" | "printscreen" | "sysrq" => Some(0x46),
        _ => None,
    };
    if let Some(u) = named {
        return Some((u, Mods(bits)));
    }
    let ch = rest.chars().next()?;
    if rest.chars().count() != 1 {
        serial_println!("keyboard> '{rest}' is not a single key (see /keyboard for the forms)");
        return None;
    }
    // Find the US key that carries this character on either of its first two
    // levels, so `2` and `@` both name the digit-2 key.
    let lower = ch.to_ascii_lowercase();
    for row in layouts::US_BASE {
        for lv in [Level::Base, Level::Shift] {
            if let Out::Char(c) = row.levels[lv as usize] {
                if c == ch || c.to_ascii_lowercase() == lower {
                    return Some((row.usage, Mods(bits)));
                }
            }
        }
    }
    serial_println!("keyboard> no US key carries '{ch}'");
    None
}

use crate::keymap::layouts::Level;

/// Render a translated result readably: the characters, then their codepoints.
fn describe(bytes: &str) -> String {
    if bytes.is_empty() {
        return String::from("(nothing)");
    }
    let mut shown = String::new();
    let mut codes = String::new();
    for c in bytes.chars() {
        if !codes.is_empty() {
            codes.push(' ');
        }
        codes.push_str(&alloc::format!("U+{:04X}", c as u32));
        match c {
            '\r' => shown.push_str("\\r"),
            '\n' => shown.push_str("\\n"),
            '\t' => shown.push_str("\\t"),
            '\u{1b}' => shown.push_str("ESC"),
            c if (c as u32) < 0x20 => {
                shown.push_str(&alloc::format!("0x{:02x}", c as u32));
            }
            c => shown.push(c),
        }
    }
    alloc::format!("'{shown}' ({codes})")
}

/// Run a spec list through the real translator on a named layout.
fn run_spec(id: &str, spec: &str, show_state: bool) {
    let Some(layout) = layout_by_id(id) else { return };
    if spec.is_empty() {
        serial_println!("keyboard> usage: /keyboard test <id> <key>[,<key>…]");
        return;
    }
    // A fresh state per invocation, so a pending dead key from a previous command
    // cannot change the answer — the point is a reproducible assertion.
    let mut st = State::default();
    let mut all = String::new();
    let mut labels = String::new();
    for tok in spec.split(',') {
        let tok = tok.trim();
        if tok.is_empty() {
            continue;
        }
        // The standing rules apply even here: this loop pumps and answers Ctrl+C.
        if crate::shell::poll_interrupt() {
            serial_println!("keyboard> cancelled");
            return;
        }
        crate::shell::upkeep();
        let Some((usage, mut mods)) = parse_spec(tok) else { return };
        if mods.has(Mods::CAPS) {
            st.caps = true;
            mods.set(Mods::CAPS, false);
        }
        let ev = KeyEvent { usage, mods, pressed: true, src: Source::UsbHid };
        let emit = keymap::translate(&mut st, layout, ev);
        labels.push_str(&alloc::format!("[{tok}]"));
        all.push_str(&emit.bytes);
        if show_state {
            let pend = keymap::pending_description(&st).unwrap_or_else(|| String::from("nothing"));
            serial_println!(
                "  [{tok}] usage {usage:#04x} -> {} | pending: {pend}",
                describe(&emit.bytes)
            );
        }
        if let Some(m) = emit.message {
            serial_println!("  [{tok}] refused: {m}");
        }
    }
    let pend = keymap::pending_description(&st);
    let suffix = match pend {
        Some(p) => alloc::format!("  [{p}]"),
        None => String::new(),
    };
    serial_println!("keyboard> {id}: {labels} -> {}{suffix}", describe(&all));
}

fn test(rest: &str) {
    let mut it = rest.splitn(2, char::is_whitespace);
    let id = it.next().unwrap_or("");
    let spec = it.next().unwrap_or("").trim();
    run_spec(id, spec, false);
}

fn dead(rest: &str) {
    let mut it = rest.splitn(2, char::is_whitespace);
    let id = it.next().unwrap_or("");
    let spec = it.next().unwrap_or("").trim();
    if spec.is_empty() {
        // With no spec, list the composition table for that layout's dead keys.
        let Some(l) = layout_by_id(id) else { return };
        let dks = l.dead_keys();
        if dks.is_empty() {
            serial_println!("keyboard> layout '{}' has no dead keys", l.id);
            return;
        }
        for dk in dks {
            let mut line = String::new();
            for (k, base, out) in layouts::dead_rows() {
                if *k == dk {
                    if !line.is_empty() {
                        line.push(' ');
                    }
                    line.push_str(&alloc::format!("{base}{out}"));
                }
            }
            serial_println!(
                "keyboard> {} ({}): {line}",
                layouts::dead_name(dk),
                layouts::spacing_form(dk)
            );
        }
        return;
    }
    run_spec(id, spec, true);
}

fn compose_table() {
    let rows = layouts::compose_rows();
    serial_println!("keyboard> {} compose sequence(s) — press Compose, then two keys:", rows.len());
    let mut line = String::new();
    let mut n = 0;
    for (a, b, out) in rows {
        line.push_str(&alloc::format!("{a}{b}={out}  "));
        n += 1;
        if n % 8 == 0 {
            serial_println!("  {line}");
            line.clear();
        }
    }
    if !line.is_empty() {
        serial_println!("  {line}");
    }
    serial_println!("keyboard> anything not listed is refused, never typed as its two keys.");
}

fn hex(rest: &str) {
    let s = rest.trim().trim_start_matches("U+").trim_start_matches("u+");
    if s.is_empty() {
        serial_println!("keyboard> usage: /keyboard hex <hex> (e.g. 00e9)");
        return;
    }
    let Ok(v) = u32::from_str_radix(s, 16) else {
        serial_println!("keyboard> '{s}' is not hexadecimal");
        return;
    };
    if v > 0x10_FFFF {
        serial_println!("keyboard> U+{v:X} is above U+10FFFF — not a codepoint");
        return;
    }
    match char::from_u32(v) {
        Some(c) => serial_println!(
            "keyboard> U+{:04X} = {} ({} byte(s), {} column(s))",
            v,
            describe(&{
                let mut s = String::new();
                s.push(c);
                s
            }),
            c.len_utf8(),
            crate::textfit::char_cols(c)
        ),
        None => serial_println!(
            "keyboard> U+{v:04X} is a surrogate (D800..DFFF) — not a character. \
             Type Ctrl+Shift+U then the hex digits for a real one."
        ),
    }
}

fn ime(rest: &str) {
    let want = rest.trim();
    if want.is_empty() {
        serial_println!(
            "keyboard> ime {} — /keyboard ime off|hiragana|katakana|hangul",
            crate::ime::mode_name(crate::ime::mode())
        );
        return;
    }
    match crate::ime::set_mode_by_name(want) {
        Ok(m) => {
            serial_println!("keyboard> ime {}", crate::ime::mode_name(m));
            if matches!(m, crate::ime::Mode::Hiragana | crate::ime::Mode::Katakana) {
                serial_println!(
                    "keyboard> romaji in, kana out. Every kana it produces is in the bundled"
                );
                serial_println!(
                    "keyboard> CJK subset, so what you type renders — but the chat pane has one"
                );
                serial_println!(
                    "keyboard> cell per character, so kana appear narrow rather than double-width."
                );
            }
        }
        Err(why) => {
            serial_println!("keyboard> refused: {why}");
        }
    }
}

fn type_through_ime(rest: &str) {
    if rest.is_empty() {
        serial_println!("keyboard> usage: /keyboard type <text>");
        return;
    }
    let mut committed = String::new();
    let mut unconsumed = String::new();
    for ch in rest.chars() {
        if crate::shell::poll_interrupt() {
            serial_println!("keyboard> cancelled");
            return;
        }
        let out = crate::ime::feed(ch);
        if out.consumed {
            committed.push_str(&out.commit);
        } else {
            unconsumed.push(ch);
        }
    }
    let pre = crate::ime::preedit();
    serial_println!("keyboard> ime {}", crate::ime::mode_name(crate::ime::mode()));
    serial_println!("  commit   {}", if committed.is_empty() { String::from("(nothing)") } else { describe(&committed) });
    serial_println!("  preedit  {}", if pre.is_empty() { String::from("(empty)") } else { describe(&pre) });
    if !unconsumed.is_empty() {
        // The property that keeps `/commands` working while an IME is on.
        serial_println!("  passed through unconsumed: {}", describe(&unconsumed));
    }
}

fn echo(rest: &str) {
    // The e2e handle for the UTF-8 line-editor work: it reports what the shell
    // actually received, in all three units that used to be conflated.
    serial_println!(
        "keyboard> echo: {rest} ({} chars, {} bytes, {} cols)",
        rest.chars().count(),
        rest.len(),
        crate::textfit::cols(rest)
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn a_key_spec_resolves_to_a_usage_and_modifier_set() {
        // Named by its US base character, either case, either level.
        assert_eq!(parse_spec("q").map(|(u, _)| u), Some(0x14));
        assert_eq!(parse_spec("Q").map(|(u, _)| u), Some(0x14));
        assert_eq!(parse_spec("2").map(|(u, _)| u), Some(0x1f));
        assert_eq!(parse_spec("@").map(|(u, _)| u), Some(0x1f), "the shifted form names the key");
        // Modifier prefixes, single and combined.
        assert_eq!(parse_spec("shift+2"), Some((0x1f, Mods(Mods::SHIFT))));
        assert_eq!(parse_spec("ctrl+c"), Some((0x06, Mods(Mods::CTRL))));
        assert_eq!(parse_spec("altgr+q"), Some((0x14, Mods(Mods::ALTGR))));
        assert_eq!(
            parse_spec("ctrl+shift+u"),
            Some((0x18, Mods(Mods::CTRL | Mods::SHIFT)))
        );
        // Named keys, including the ISO key that has no US character at all.
        assert_eq!(parse_spec("space").map(|(u, _)| u), Some(0x2c));
        assert_eq!(parse_spec("up").map(|(u, _)| u), Some(0x52));
        assert_eq!(parse_spec("iso").map(|(u, _)| u), Some(0x64));
        // The function keys, which carry the two global shortcuts.
        assert_eq!(parse_spec("f1").map(|(u, _)| u), Some(0x3a));
        assert_eq!(parse_spec("f12").map(|(u, _)| u), Some(0x45));
        assert_eq!(parse_spec("prtsc").map(|(u, _)| u), Some(0x46));
        // Nonsense is refused rather than guessed at.
        assert!(parse_spec("nonsense").is_none());
        assert!(parse_spec("hyper+a").is_none());
    }

    #[test_case]
    fn describe_shows_control_bytes_rather_than_emitting_them() {
        assert_eq!(describe(""), "(nothing)");
        assert!(describe("\r").contains("\\r"));
        assert!(describe("\u{1b}[A").contains("ESC"));
        assert!(describe("\u{3}").contains("0x03"));
        let d = describe("é");
        assert!(d.contains("U+00E9"), "{d}");
    }
}
