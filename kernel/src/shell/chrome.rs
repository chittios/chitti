//! Chat chrome formatters (pure, no_std): thought/work timing, wait labels,
//! the progress-bar sweep, tool headers, and text sanitization for the shell UI.

use alloc::format;
use alloc::string::{String, ToString};
use core::fmt::Write;

/// Prompt prefix for a submitted user line in scrollback.
pub const PROMPT_ARROW: &str = ">";

/// Tool / thought chrome bullet, **as it goes on the wire**.
///
/// Deliberately ASCII. This string reaches the real serial port as well as the
/// framebuffer, and the two want different things: a host terminal (and the e2e harness,
/// which asserts on `* Write`-style banners) wants a character it can render and match,
/// while the compositor wants a Font Awesome mark. So the icon substitution happens at
/// **draw time** in the pane — see [`tool_chrome_icon`] — and the byte stream is
/// unchanged. Putting a Private-Use-Area codepoint in here instead would show as tofu
/// over a serial console and silently break those assertions.
pub const DIAMOND: &str = "*";

/// Vertical connector down the left edge of a tool call's output, on the wire.
pub const TOOL_PIPE: &str = "|";

/// The glyph the **compositor** draws in place of a wire-level chrome marker, or `None`
/// for anything else.
///
/// Called only for column 0 of a line the pane has already classified as a tool block, so
/// it never sees ordinary text: a `*` or `|` a user typed is not in that band. Pure, so
/// the mapping is checked without a framebuffer — `framebuffer/` is `#[cfg(not(test))]`
/// and a test written in there would never even compile.
///
/// The bullet is Font Awesome; the connector is **box-drawing U+2502**, not an icon.
/// FA Free has no single vertical rule (`pipe` is a Pro glyph) and the nearest thing,
/// `grip-lines-vertical`, is two parallel bars that read as a stray `‖` in running text.
/// A box-drawing rule is also the better shape mechanically: it fills the full mono cell
/// height, so consecutive rows join into one unbroken line down the block, where an icon
/// is centred in a square and leaves a gap at every row boundary. Both bundled UI faces
/// (Geist Mono, Ubuntu Mono) carry U+2502 — checked in their cmaps, since a missing glyph
/// here renders as nothing at all rather than as an error.
pub fn tool_chrome_icon(ch: char) -> Option<char> {
    match ch {
        '*' => Some(crate::icons::fa::CIRCLE),
        '|' => Some('\u{2502}'),
        _ => None,
    }
}

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

// ---------------------------------------------------------------------------
// Progress bar sweep
// ---------------------------------------------------------------------------
//
// A block bar whose colour gradient sweeps, rather than a rotating `|/-\`
// character: it reads as *work in progress* at a glance and does not look like
// a stuck character when a frame is dropped (prefill ticks at ~10 Hz but a
// single batched chunk can outlast several frames). The geometry and frame
// sequence follow `@astrojs/cli-kit`'s spinner — a 6-cell window scrolled over
// a 30-entry ramp — while the two gradient endpoints come from the live theme
// (`composer_hint` -> `accent`), so it stays on-brand and follows `/theme`
// instead of hardcoding astro's green/purple.

/// One bar cell. Geist Mono covers the block-element range (verified: U+2588,
/// U+2591..U+2593), so this needs no fallback face.
pub const BAR_BLOCK: char = '\u{2588}';

/// Cells in the bar (astro's `COLORS.length - 2`).
pub const BAR_CELLS: usize = 6;

/// Gradient stops the sweep interpolates through (astro's `COLORS`).
pub const BAR_STOPS: usize = 8;

/// Frames in one full sweep, i.e. `FULL_FRAMES.len()` below.
pub const BAR_FRAMES: usize = 30;

/// The bar as plain text — `BAR_CELLS` full blocks, no colour. Used where a
/// surface paints its own per-cell colours (the composer hint bar) and as the
/// degraded look on a console with no colour.
pub const BAR: &str = "\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}";

/// Astro's `FULL_FRAMES`, as indices into the gradient stops: a dim run, the
/// 8-stop ramp up, a bright run, then the ramp back down. A 6-cell window
/// slides over it, so the bright band enters from one end, fills the bar, and
/// retreats — 30 frames at the ~10 Hz [`super::upkeep`] tick is a ~3 s cycle.
const FULL_FRAMES: [u8; BAR_FRAMES] = [
    0, 0, 0, 0, 0, 0, 0, // dim run (BAR_STOPS - 1 entries)
    0, 1, 2, 3, 4, 5, 6, 7, // ramp up
    7, 7, 7, 7, 7, 7, 7, // bright run
    7, 6, 5, 4, 3, 2, 1, 0, // ramp down
];

/// Gradient-stop index per cell for animation frame `frame` (wraps).
///
/// Astro walks the window offsets in reverse, which is what makes frame 0 the
/// all-dim state and sends the band in from the left; a window running past the
/// end pads with the dim stop.
pub fn bar_stops(frame: usize) -> [u8; BAR_CELLS] {
    let off = BAR_FRAMES - 1 - (frame % BAR_FRAMES);
    let mut out = [0u8; BAR_CELLS];
    for (i, cell) in out.iter_mut().enumerate() {
        if let Some(&s) = FULL_FRAMES.get(off + i) {
            *cell = s;
        }
    }
    out
}

/// Linear ramp: stop `0` is `dim`, stop `BAR_STOPS - 1` is `bright`.
pub fn bar_color(stop: u8, dim: (u8, u8, u8), bright: (u8, u8, u8)) -> (u8, u8, u8) {
    let n = (BAR_STOPS - 1) as u32;
    let t = (stop as u32).min(n);
    let mix = |a: u8, b: u8| (((a as u32) * (n - t) + (b as u32) * t) / n) as u8;
    (mix(dim.0, bright.0), mix(dim.1, bright.1), mix(dim.2, bright.2))
}

/// Per-cell colours for animation frame `frame`.
pub fn bar_colors(frame: usize, dim: (u8, u8, u8), bright: (u8, u8, u8)) -> [(u8, u8, u8); BAR_CELLS] {
    let stops = bar_stops(frame);
    let mut out = [dim; BAR_CELLS];
    for (cell, &stop) in out.iter_mut().zip(stops.iter()) {
        *cell = bar_color(stop, dim, bright);
    }
    out
}

/// The bar as 24-bit ANSI, for the chat pane and a terminal-attached serial
/// console — both parse `38;2;r;g;b` (`framebuffer::apply_sgr`). Closes with a
/// reset so the label after it takes the pane's default colour.
pub fn format_bar_ansi(frame: usize, dim: (u8, u8, u8), bright: (u8, u8, u8)) -> String {
    let mut s = String::with_capacity(BAR_CELLS * 20 + 4);
    for (r, g, b) in bar_colors(frame, dim, bright) {
        let _ = write!(s, "\x1b[38;2;{r};{g};{b}m{BAR_BLOCK}");
    }
    s.push_str("\x1b[0m");
    s
}

/// Everything after the bar on the live status line: `  Thinking  2.4s`.
/// Split out so the serial path can print a colourised bar followed by this,
/// without slicing the composed string at a hardcoded byte offset.
pub fn format_thinking_tail(secs: f32) -> String {
    format!("  {}  {}", format_thinking_live(), format_duration_secs(secs))
}

/// Live status line body: the bar, then the label + elapsed seconds.
/// e.g. `██████  Thinking  2.4s`
pub fn format_thinking_status(secs: f32) -> String {
    format!("{BAR}{}", format_thinking_tail(secs))
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

    /// The wire markers stay ASCII and the icons are a draw-time substitution.
    ///
    /// Both halves matter. A Private-Use-Area codepoint in the byte stream would reach a
    /// real serial console as tofu and break the e2e assertions on `* Write`-style
    /// banners; an icon that is *not* mapped leaves the chat showing raw `*` and `|`.
    #[test_case]
    fn tool_chrome_is_ascii_on_the_wire_and_an_icon_on_screen() {
        assert_eq!(DIAMOND, "*", "the wire marker must stay ASCII");
        assert_eq!(TOOL_PIPE, "|", "the wire connector must stay ASCII");
        assert!(DIAMOND.is_ascii() && TOOL_PIPE.is_ascii());

        // The bullet is a Font Awesome glyph, in the range the FA fallback face is
        // consulted for — outside it, it would resolve to a mono glyph or tofu.
        let mark = tool_chrome_icon('*').expect("the bullet maps to a glyph");
        assert_eq!(mark, crate::icons::fa::CIRCLE);
        assert!(crate::icons::is_icon(mark), "the bullet must reach the FA face");

        // The connector is box-drawing U+2502 and must **not** be an icon: icons are
        // centred in a square cell, so consecutive rows would leave a gap instead of one
        // unbroken rule. `is_icon` is also what `band_glyph` keys the inline sizing on.
        let pipe = tool_chrome_icon('|').expect("the connector maps to a glyph");
        assert_eq!(pipe, '\u{2502}');
        assert!(!crate::icons::is_icon(pipe), "the rule must fill the full cell height");

        // Everything else is left exactly as printed — the substitution runs on column 0
        // of a tool line, and ordinary text must survive it untouched.
        for ch in ['o', ' ', '>', '-', '+', '\0', 'k', '│', '•'] {
            assert!(tool_chrome_icon(ch).is_none(), "{ch:?} must not be rewritten");
        }
    }

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
        let s = format_thinking_status(2.4);
        assert!(s.starts_with(BAR));
        assert!(s.contains("Thinking"));
        assert!(s.contains("2.4s"));
        assert!(has_think_block("<think>x</think>hi"));
        assert!(!has_think_block("hi only"));
    }

    /// The bar is exactly `BAR_CELLS` blocks wide -- the composer hint bar
    /// colours cell `i` from `bar_colors()[i]`, so a mismatch would paint the
    /// gradient onto the label text.
    #[test_case]
    fn bar_is_cells_wide() {
        assert_eq!(BAR.chars().count(), BAR_CELLS);
        assert!(BAR.chars().all(|c| c == BAR_BLOCK));
    }

    /// The sweep: all dim, band in from the left, all bright, band out.
    /// Pins the window direction -- walking the offsets forward instead of in
    /// reverse runs the whole animation backwards and starts it fully lit.
    #[test_case]
    fn bar_sweeps_dim_to_bright_and_back() {
        assert_eq!(bar_stops(0), [0, 0, 0, 0, 0, 0]);
        assert_eq!(bar_stops(7), [7, 6, 5, 4, 3, 2]);
        assert_eq!(bar_stops(14), [7, 7, 7, 7, 7, 7]);
        assert_eq!(bar_stops(21), [1, 2, 3, 4, 5, 6]);
        assert_eq!(bar_stops(29), [0, 0, 0, 0, 0, 0]);
        // Wraps, so a frame counter can run forever.
        assert_eq!(bar_stops(BAR_FRAMES + 7), bar_stops(7));
        // Every stop stays a valid gradient index.
        for f in 0..BAR_FRAMES * 3 {
            assert!(bar_stops(f).iter().all(|&s| (s as usize) < BAR_STOPS));
        }
    }

    #[test_case]
    fn bar_color_ramps_between_endpoints() {
        let dim = (108, 106, 100);
        let bright = (204, 120, 92);
        assert_eq!(bar_color(0, dim, bright), dim);
        assert_eq!(bar_color(BAR_STOPS as u8 - 1, dim, bright), bright);
        // Out-of-range clamps to the bright end rather than wrapping dark.
        assert_eq!(bar_color(200, dim, bright), bright);
        // Monotonic on the channel that increases (red: 108 -> 204).
        let mut prev = 0u8;
        for s in 0..BAR_STOPS as u8 {
            let c = bar_color(s, dim, bright).0;
            assert!(c >= prev);
            prev = c;
        }
    }

    #[test_case]
    fn bar_ansi_has_one_truecolour_run_per_cell() {
        let s = format_bar_ansi(9, (108, 106, 100), (204, 120, 92));
        assert_eq!(s.matches("\x1b[38;2;").count(), BAR_CELLS);
        assert_eq!(s.matches(BAR_BLOCK).count(), BAR_CELLS);
        assert!(s.ends_with("\x1b[0m"));
        // Frame 0 is uniformly dim, so it names the dim colour every cell.
        let d = format_bar_ansi(0, (108, 106, 100), (204, 120, 92));
        assert_eq!(d.matches("\x1b[38;2;108;106;100m").count(), BAR_CELLS);
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
