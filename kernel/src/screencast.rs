//! `/record` — argument parsing and naming for screen recordings.
//!
//! Pure only: the live session (start/stop, frame pump, encoder) lives in the
//! shell because it needs the framebuffer. This module is the grammar and the
//! constants, so the CLI is unit-tested off hardware.
//!
//! **macOS shape:** start and stop, not a fixed 5-second clip. Bare `/record`
//! toggles; `Cmd+Shift+5` (and Ctrl twin) does the same. A status-bar chip
//! shows while a take is running and clicking it stops. An optional `for <d>`
//! is a *timer*, not the default.

use alloc::string::{String, ToString};

use crate::screenshot::{self, Extent};

/// Hard cap on frames (10 min × 15 fps). Safety net so a forgotten session
/// cannot grow without bound even if the human never hits stop.
pub const MAX_FRAMES: usize = 9_000;
/// Longest a session may run before auto-stop. Not advertised as the main
/// interface — start/stop is — but a forgotten recording must end.
pub const MAX_DURATION_MS: u64 = 600_000;
pub const DEFAULT_FPS: u32 = 5;
pub const DEFAULT_SCALE_PCT: u32 = 50;
pub const MAX_FPS: u32 = 15;
pub const MIN_FPS: u32 = 1;
pub const DEFAULT_QP: u32 = 28;

/// What `/record` should do.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Verb {
    /// Bare `/record` or the global shortcut: start if idle, stop if live.
    Toggle,
    Start,
    Stop,
    Status,
    Help,
}

/// A parsed `/record` invocation.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Request {
    pub verb: Verb,
    pub extent: Extent,
    pub dest: Option<String>,
    pub cursor: bool,
    /// `None` = run until stop (the default). `Some` = auto-stop after this many ms.
    pub duration_ms: Option<u64>,
    pub fps: u32,
    pub scale_pct: u32,
}

impl Default for Request {
    fn default() -> Self {
        Request {
            verb: Verb::Toggle,
            extent: Extent::Desktop,
            dest: None,
            cursor: false,
            duration_ms: None,
            fps: DEFAULT_FPS,
            scale_pct: DEFAULT_SCALE_PCT,
        }
    }
}

/// Parse `/record`'s argument string.
///
/// ```text
/// /record                         → toggle start/stop
/// /record start [options] [dest]
/// /record stop
/// /record status
/// /record for <d> [options]       → start, auto-stop after <d>
/// options: fps <n> | scale <pct> | desktop|panel|chat|pane <n>|region …
///          | --cursor | <dest>
/// ```
pub fn parse(arg: &str) -> Result<Request, String> {
    let mut req = Request::default();
    let mut it = arg.split_whitespace().peekable();
    let mut saw_verb = false;

    // Leading verb is optional; bare args imply start (with toggle if empty).
    if let Some(&first) = it.peek() {
        match first {
            "help" | "-h" | "--help" => {
                req.verb = Verb::Help;
                return Ok(req);
            }
            "start" | "begin" | "rec" => {
                req.verb = Verb::Start;
                saw_verb = true;
                it.next();
            }
            "stop" | "end" | "finish" => {
                req.verb = Verb::Stop;
                saw_verb = true;
                it.next();
            }
            "status" | "info" => {
                req.verb = Verb::Status;
                saw_verb = true;
                it.next();
            }
            "toggle" => {
                req.verb = Verb::Toggle;
                saw_verb = true;
                it.next();
            }
            _ => {}
        }
    }

    while let Some(tok) = it.next() {
        match tok {
            "--cursor" | "-c" => req.cursor = true,
            "desktop" | "screen" | "logical" => req.extent = Extent::Desktop,
            "panel" | "physical" | "full" => req.extent = Extent::Panel,
            "chat" => req.extent = Extent::Chat,
            "pane" => {
                let n = it.next().ok_or_else(|| "pane needs an index".to_string())?;
                let idx: usize =
                    n.parse().map_err(|_| alloc::format!("bad pane index '{n}'"))?;
                req.extent = Extent::Pane(idx);
            }
            "region" | "rect" => {
                let spec = it.next().ok_or_else(|| "region needs x,y,w,h".to_string())?;
                req.extent = parse_region(spec)?;
            }
            "for" | "duration" | "secs" => {
                let d = it.next().ok_or_else(|| "for needs a duration".to_string())?;
                req.duration_ms = Some(parse_duration_ms(d)?);
                // A timed take is always a start, not a bare toggle.
                if matches!(req.verb, Verb::Toggle) {
                    req.verb = Verb::Start;
                }
            }
            "fps" | "rate" => {
                let n = it.next().ok_or_else(|| "fps needs a number".to_string())?;
                req.fps = parse_fps(n)?;
            }
            "scale" => {
                let n = it.next().ok_or_else(|| "scale needs a percent".to_string())?;
                req.scale_pct = parse_scale(n)?;
            }
            _ if tok.starts_with('-') => {
                return Err(alloc::format!("unknown flag '{tok}'"));
            }
            _ if looks_like_duration(tok) => {
                req.duration_ms = Some(parse_duration_ms(tok)?);
                if matches!(req.verb, Verb::Toggle) {
                    req.verb = Verb::Start;
                }
            }
            _ => {
                if req.dest.is_some() {
                    return Err(alloc::format!("unexpected extra argument '{tok}'"));
                }
                req.dest = Some(tok.to_string());
                // A dest alone means "start and save there", not toggle.
                if matches!(req.verb, Verb::Toggle) && !saw_verb {
                    req.verb = Verb::Start;
                }
            }
        }
    }

    // Bound a timed take up front so we never start a session that cannot finish.
    if let Some(ms) = req.duration_ms {
        let n = frame_count(ms, req.fps);
        if n == 0 {
            return Err("recording would capture zero frames".to_string());
        }
        if n > MAX_FRAMES {
            return Err(alloc::format!(
                "recording would be {n} frames (max {MAX_FRAMES}); lower fps or duration"
            ));
        }
    }
    Ok(req)
}

fn looks_like_duration(s: &str) -> bool {
    let s = s.trim();
    if s.ends_with("ms") || s.ends_with('s') || s.ends_with('m') {
        let num = if let Some(v) = s.strip_suffix("ms") {
            v
        } else if let Some(v) = s.strip_suffix('m') {
            v
        } else {
            s.strip_suffix('s').unwrap_or(s)
        };
        return !num.is_empty() && num.bytes().all(|b| b.is_ascii_digit());
    }
    false
}

fn parse_region(spec: &str) -> Result<Extent, String> {
    let mut n = [0u32; 4];
    let mut count = 0;
    for part in spec.split(',') {
        if count == 4 {
            return Err("region takes exactly x,y,w,h".to_string());
        }
        n[count] = part
            .trim()
            .parse()
            .map_err(|_| alloc::format!("bad region component '{part}'"))?;
        count += 1;
    }
    if count != 4 {
        return Err("region takes exactly x,y,w,h".to_string());
    }
    if n[2] == 0 || n[3] == 0 {
        return Err("region width and height must be non-zero".to_string());
    }
    Ok(Extent::Region {
        x: n[0],
        y: n[1],
        w: n[2],
        h: n[3],
    })
}

pub fn parse_duration_ms(s: &str) -> Result<u64, String> {
    let s = s.trim();
    let (num, mult) = if let Some(v) = s.strip_suffix("ms") {
        (v, 1u64)
    } else if let Some(v) = s.strip_suffix('m') {
        (v, 60_000)
    } else if let Some(v) = s.strip_suffix('s') {
        (v, 1000)
    } else {
        (s, 1000)
    };
    let v: u64 = num.parse().map_err(|_| alloc::format!("bad duration '{s}'"))?;
    let ms = v.saturating_mul(mult);
    if ms == 0 {
        return Err("duration must be non-zero".to_string());
    }
    if ms > MAX_DURATION_MS {
        return Err(alloc::format!(
            "duration {ms} ms exceeds the {MAX_DURATION_MS} ms maximum"
        ));
    }
    Ok(ms)
}

fn parse_fps(s: &str) -> Result<u32, String> {
    let v: u32 = s
        .trim()
        .parse()
        .map_err(|_| alloc::format!("bad fps '{s}'"))?;
    if v < MIN_FPS || v > MAX_FPS {
        return Err(alloc::format!(
            "fps must be between {MIN_FPS} and {MAX_FPS} (got {v})"
        ));
    }
    Ok(v)
}

fn parse_scale(s: &str) -> Result<u32, String> {
    let t = s.trim().trim_end_matches('%');
    let v: u32 = t
        .parse()
        .map_err(|_| alloc::format!("bad scale '{s}'"))?;
    if v == 0 || v > 100 {
        return Err(alloc::format!("scale must be 1–100 percent (got {v})"));
    }
    Ok(v)
}

pub fn frame_count(duration_ms: u64, fps: u32) -> usize {
    if duration_ms == 0 || fps == 0 {
        return 0;
    }
    let n = duration_ms.saturating_mul(fps as u64) / 1000;
    n.max(1) as usize
}

pub fn frame_interval_ms(fps: u32) -> u64 {
    if fps == 0 {
        return 1000;
    }
    (1000u64 / fps as u64).max(1)
}

pub fn sample_duration_ms(fps: u32) -> u32 {
    frame_interval_ms(fps).min(u32::MAX as u64) as u32
}

pub fn scaled_size(w: u32, h: u32, scale_pct: u32) -> (u32, u32) {
    if w == 0 || h == 0 {
        return (0, 0);
    }
    if scale_pct >= 100 {
        return (w, h);
    }
    let pct = scale_pct.max(1) as u64;
    let nw = ((w as u64 * pct) / 100).max(1) as u32;
    let nh = ((h as u64 * pct) / 100).max(1) as u32;
    (nw, nh)
}

pub fn default_path(now_ms: u64) -> String {
    alloc::format!("/downloads/recording-{now_ms}.mp4")
}

pub fn normalize_dest(dest: &str) -> String {
    let mut s = if dest.contains('/') {
        dest.to_string()
    } else {
        alloc::format!("/downloads/{dest}")
    };
    let lower = s.to_ascii_lowercase();
    if !(lower.ends_with(".mp4") || lower.ends_with(".mov") || lower.ends_with(".m4v")) {
        s.push_str(".mp4");
    }
    s
}

pub fn saved_line(path: &str, w: u32, h: u32, frames: usize, bytes: usize) -> String {
    alloc::format!(
        "saved {path} ({w}x{h}, {frames} frame{}, {bytes} bytes) — /open {path}",
        if frames == 1 { "" } else { "s" }
    )
}

/// Status-bar chip text while recording. Empty when idle so the template
/// swallows the separator (same posture as `${notifications}`).
///
/// `elapsed_ms` is wall time since start; the chip reads `● 1:23`.
pub fn chip_text(recording: bool, elapsed_ms: u64) -> String {
    if !recording {
        return String::new();
    }
    let secs = elapsed_ms / 1000;
    let m = secs / 60;
    let s = secs % 60;
    // Solid circle (FA) — the universal "REC" mark. Colour comes from the
    // status painter's accent when it recognises the chip.
    alloc::format!("{} {:}:{:02}", crate::icons::fa::CIRCLE, m, s)
}

pub use screenshot::clamp_region;

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn bare_invocation_is_a_toggle_with_no_timer() {
        let r = parse("").unwrap();
        assert_eq!(r.verb, Verb::Toggle);
        assert_eq!(r.duration_ms, None);
        assert_eq!(r.fps, DEFAULT_FPS);
        assert_eq!(r.scale_pct, DEFAULT_SCALE_PCT);
    }

    #[test_case]
    fn start_stop_status_parse() {
        assert_eq!(parse("start").unwrap().verb, Verb::Start);
        assert_eq!(parse("stop").unwrap().verb, Verb::Stop);
        assert_eq!(parse("status").unwrap().verb, Verb::Status);
        assert_eq!(parse("help").unwrap().verb, Verb::Help);
    }

    #[test_case]
    fn for_duration_starts_a_timed_take() {
        let r = parse("for 30s fps 10").unwrap();
        assert_eq!(r.verb, Verb::Start);
        assert_eq!(r.duration_ms, Some(30_000));
        assert_eq!(r.fps, 10);
    }

    #[test_case]
    fn a_dest_alone_means_start() {
        let r = parse("demo.mp4").unwrap();
        assert_eq!(r.verb, Verb::Start);
        assert_eq!(r.dest.as_deref(), Some("demo.mp4"));
    }

    #[test_case]
    fn chip_text_is_empty_when_idle_and_shows_elapsed_when_live() {
        assert_eq!(chip_text(false, 0), "");
        let s = chip_text(true, 83_000);
        assert!(s.contains("1:23"), "{s}");
        assert!(s.chars().next().is_some());
    }

    #[test_case]
    fn dest_normalisation_adds_mp4() {
        assert_eq!(normalize_dest("clip"), "/downloads/clip.mp4");
        assert_eq!(default_path(99), "/downloads/recording-99.mp4");
    }

    #[test_case]
    fn ten_minutes_is_the_hard_cap_not_the_default() {
        assert!(parse("for 10m").is_ok());
        assert!(parse("for 11m").is_err());
        // Default has no timer at all.
        assert_eq!(parse("start").unwrap().duration_ms, None);
    }
}
