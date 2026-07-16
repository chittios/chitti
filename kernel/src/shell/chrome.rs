//! Chat chrome formatters (pure, no_std): thought/work timing, wait labels,
//! tool headers, and text sanitization for the shell UI.

use alloc::format;
use alloc::string::{String, ToString};

/// Prompt prefix for a submitted user line in scrollback.
pub const PROMPT_ARROW: &str = ">";

/// Tool / thought chrome bullet.
pub const DIAMOND: &str = "*";

/// Format a short wall-clock duration for status lines.
pub fn format_duration_secs(secs: f32) -> String {
    let s = if secs < 0.0 { 0.0 } else { secs };
    if s < 60.0 {
        format!("{:.1}s", s)
    } else {
        // no_std: no f32::floor — truncate toward zero for positive secs.
        let mins = (s / 60.0) as u32;
        let rem = s - (mins as f32 * 60.0);
        format!("{}m{:.0}s", mins, rem)
    }
}

/// Collapsed thinking summary — only when a think block was actually parsed.
pub fn format_thought_done(secs: f32) -> String {
    format!("Thought for {}", format_duration_secs(secs))
}

/// Total turn / response wall time (after answer body).
pub fn format_worked_for(secs: f32) -> String {
    format!("Worked for {}.", format_duration_secs(secs))
}

/// Live status on the composer bar while the model is busy.
pub fn format_thinking_live() -> &'static str {
    "Thinking"
}

/// Live status line with elapsed seconds + spinner frame.
/// e.g. `Thinking  2.4s  |`
pub fn format_thinking_status(secs: f32, frame: char) -> String {
    format!("Thinking  {}  {}", format_duration_secs(secs), frame)
}

/// True if `text` contains a model think/reasoning block we can strip.
pub fn has_think_block(text: &str) -> bool {
    text.contains("<think>") || text.contains("</think>")
}

/// Map fancy Unicode punctuation to ASCII so the font path does not show
/// mojibake for em-dashes, bullets, etc.
pub fn sanitize_chat_text(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\u{2014}' | '\u{2013}' | '\u{2212}' => out.push('-'),
            '\u{2022}' | '\u{00B7}' | '\u{2023}' => out.push('*'),
            '\u{2026}' => out.push_str("..."),
            '\u{2018}' | '\u{2019}' | '\u{201A}' => out.push('\''),
            '\u{201C}' | '\u{201D}' | '\u{201E}' => out.push('"'),
            '\u{00A0}' => out.push(' '),
            '\u{FFFD}' => out.push('?'),
            c if (c as u32) >= 0x80 && (c as u32) < 0xA0 => {}
            c => out.push(c),
        }
    }
    out
}

/// Tool header line body (no ANSI): `* Edit  path`.
pub fn format_tool_header_plain(verb: &str, arg: &str) -> String {
    if arg.is_empty() {
        format!("{DIAMOND} {verb}")
    } else {
        format!("{DIAMOND} {verb}  {arg}")
    }
}

/// User prompt plain: `> text` (first line).
pub fn format_user_line(text: &str) -> String {
    format!("{PROMPT_ARROW} {text}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn duration_under_and_over_minute() {
        assert_eq!(format_duration_secs(0.6), "0.6s");
        assert_eq!(format_duration_secs(7.4), "7.4s");
        assert_eq!(format_duration_secs(125.0), "2m5s");
    }

    #[test_case]
    fn thought_worked_and_status() {
        assert_eq!(format_thought_done(0.6), "Thought for 0.6s");
        assert_eq!(format_worked_for(7.4), "Worked for 7.4s.");
        let s = format_thinking_status(2.4, '|');
        assert!(s.starts_with("Thinking"));
        assert!(s.contains("2.4s"));
        assert!(has_think_block("<think>x</think>hi"));
        assert!(!has_think_block("hi only"));
    }

    #[test_case]
    fn sanitize_emdash_and_bullets() {
        let s = sanitize_chat_text("Files \u{2014} read; \u{2022} item\u{2026}");
        assert!(!s.contains('\u{2014}'));
        assert!(s.contains("Files - read"));
        assert!(s.contains("* item"));
        assert!(s.contains("..."));
    }

    #[test_case]
    fn tool_and_user_plain() {
        assert_eq!(format_tool_header_plain("Edit", "/x"), "* Edit  /x");
        assert_eq!(format_user_line("hi"), "> hi");
    }
}
