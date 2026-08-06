//! `/screenshot` — argument parsing, extent geometry, and destination naming.
//!
//! Everything in this file is **pure**: the hardware read lives in
//! `framebuffer::capture` and the file write in the shell, so all the fiddly
//! parts — which rectangle, clamped how, written where — are unit-tested off
//! hardware. That split is the house rule (`framebuffer/` is
//! `#[cfg(not(test))]`, so a test written next to the pixel reader would never
//! even be compiled).

use alloc::string::{String, ToString};

/// What to capture.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Extent {
    /// The logical desktop — what the panes are laid out inside. The default,
    /// because a letterboxed desktop's black bars are not part of the screen
    /// anyone is looking at.
    Desktop,
    /// The whole physical panel, letterbox bars included.
    Panel,
    /// A rectangle in **logical desktop** coordinates.
    Region { x: u32, y: u32, w: u32, h: u32 },
    /// One action pane, addressed row-major exactly as `/pane` addresses them.
    Pane(usize),
    /// The chat pane.
    Chat,
    /// The pane holding a specific `synapse::ui` surface.
    ///
    /// Deliberately **not** reachable from [`parse`]: it exists so a non-root
    /// agent's capture can be narrowed to the window it owns, and letting a
    /// surface id be typed would hand any caller a way to name somebody else's.
    /// The only producer is the ownership lookup in `/screenshot`.
    Surface(u32),
}

/// A parsed `/screenshot` invocation.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Request {
    pub extent: Extent,
    /// `None` means "make up a timestamped name under `/downloads/`".
    pub dest: Option<String>,
    /// Include the mouse cursor sprite. Off by default: the pointer is almost
    /// never the subject, and a captured cursor in a bug report reads as a
    /// rendering artefact.
    pub cursor: bool,
    /// Wait this long before capturing, so the shell can finish echoing the
    /// command that asked for the shot.
    pub delay_ms: u64,
}

impl Default for Request {
    fn default() -> Self {
        Request { extent: Extent::Desktop, dest: None, cursor: false, delay_ms: 0 }
    }
}

/// Longest delay `after` accepts. A screenshot is not a sleep command, and an
/// unbounded value would sit in a loop pumping `upkeep` for as long as it was
/// told to.
pub const MAX_DELAY_MS: u64 = 60_000;

/// Parse `/screenshot`'s argument string.
///
/// Grammar (order-independent flags, at most one extent):
/// ```text
/// /screenshot [desktop|panel|chat|pane <n>|region <x>,<y>,<w>,<h>]
///             [after <n>[s|ms]] [--cursor] [<dest path>]
/// ```
/// A bare positional token that is not a recognised keyword is the
/// destination, which is what makes `/screenshot shot.png` do the obvious
/// thing.
pub fn parse(arg: &str) -> Result<Request, String> {
    let mut req = Request::default();
    let mut saw_extent = false;
    let mut it = arg.split_whitespace().peekable();

    while let Some(tok) = it.next() {
        match tok {
            "--cursor" | "-c" => req.cursor = true,
            "desktop" | "screen" | "logical" => {
                req.extent = Extent::Desktop;
                saw_extent = true;
            }
            "panel" | "physical" | "full" => {
                req.extent = Extent::Panel;
                saw_extent = true;
            }
            "chat" => {
                req.extent = Extent::Chat;
                saw_extent = true;
            }
            "pane" => {
                let n = it.next().ok_or_else(|| "pane needs an index".to_string())?;
                let idx: usize =
                    n.parse().map_err(|_| alloc::format!("bad pane index '{n}'"))?;
                req.extent = Extent::Pane(idx);
                saw_extent = true;
            }
            "region" | "rect" => {
                let spec = it.next().ok_or_else(|| "region needs x,y,w,h".to_string())?;
                req.extent = parse_region(spec)?;
                saw_extent = true;
            }
            "after" | "delay" => {
                let d = it.next().ok_or_else(|| "after needs a duration".to_string())?;
                req.delay_ms = parse_duration_ms(d)?;
            }
            _ if tok.starts_with('-') => {
                return Err(alloc::format!("unknown flag '{tok}'"));
            }
            _ => {
                if req.dest.is_some() {
                    return Err(alloc::format!("unexpected extra argument '{tok}'"));
                }
                req.dest = Some(tok.to_string());
            }
        }
    }
    // `saw_extent` exists only to keep the default honest in the message below;
    // it is not otherwise consulted.
    let _ = saw_extent;
    Ok(req)
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
    Ok(Extent::Region { x: n[0], y: n[1], w: n[2], h: n[3] })
}

/// Parse `500ms` / `3s` / `3` (bare numbers are seconds, because that is what
/// someone typing `after 3` means).
pub fn parse_duration_ms(s: &str) -> Result<u64, String> {
    let s = s.trim();
    let (num, mult) = if let Some(v) = s.strip_suffix("ms") {
        (v, 1u64)
    } else if let Some(v) = s.strip_suffix('s') {
        (v, 1000)
    } else {
        (s, 1000)
    };
    let v: u64 = num.parse().map_err(|_| alloc::format!("bad duration '{s}'"))?;
    let ms = v.saturating_mul(mult);
    if ms > MAX_DELAY_MS {
        return Err(alloc::format!("delay {ms} ms exceeds the {MAX_DELAY_MS} ms maximum"));
    }
    Ok(ms)
}

/// Intersect a requested rectangle with the surface it is being taken from.
///
/// Returns `None` when the request lies wholly outside — which is an error the
/// caller reports, never a silently-moved rectangle. A partially-outside
/// request is clipped, because that is what every screenshot tool does and the
/// alternative is refusing a drag that ended one pixel off the edge.
pub fn clamp_region(
    surface_w: u32,
    surface_h: u32,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
) -> Option<(u32, u32, u32, u32)> {
    if x >= surface_w || y >= surface_h || w == 0 || h == 0 {
        return None;
    }
    let cw = w.min(surface_w - x);
    let ch = h.min(surface_h - y);
    if cw == 0 || ch == 0 {
        return None;
    }
    Some((x, y, cw, ch))
}

/// The default destination for a capture taken at `now_ms` milliseconds since
/// boot. Matches `/camera grab`'s shape (`/downloads/camera-<ms>.<ext>`) so the
/// two land next to each other and neither needs explaining.
pub fn default_path(now_ms: u64) -> String {
    alloc::format!("/downloads/screenshot-{now_ms}.png")
}

/// Resolve a user-supplied destination: a bare name goes to `/downloads/`, and
/// a missing `.png` is added, since this only ever writes PNG and a file called
/// `shot` that `/open` cannot route is a worse outcome than a corrected name.
///
/// Path *resolution* (`~`, relative-to-pwd) is deliberately **not** done here —
/// that belongs to `shell::resolve_path`, which is the one place in the OS that
/// implements the rule.
pub fn normalize_dest(dest: &str) -> String {
    let mut s = if dest.contains('/') {
        dest.to_string()
    } else {
        alloc::format!("/downloads/{dest}")
    };
    if !s.ends_with(".png") {
        s.push_str(".png");
    }
    s
}

/// The `saved …` line, so its wording is pinned by a test rather than by
/// whatever the shell happened to print. Mirrors `/camera grab`'s message,
/// including the `/open` hint — the whole point is that the next thing you do
/// is look at it.
pub fn saved_line(path: &str, w: u32, h: u32, bytes: usize) -> String {
    alloc::format!("saved {path} ({w}x{h}, {bytes} bytes) — /open {path}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn a_bare_invocation_captures_the_desktop_to_a_generated_name() {
        let r = parse("").unwrap();
        assert_eq!(r.extent, Extent::Desktop);
        assert_eq!(r.dest, None);
        assert!(!r.cursor);
        assert_eq!(r.delay_ms, 0);
        assert_eq!(default_path(12345), "/downloads/screenshot-12345.png");
    }

    #[test_case]
    fn a_lone_positional_is_the_destination() {
        assert_eq!(parse("shot.png").unwrap().dest.as_deref(), Some("shot.png"));
        assert_eq!(parse("/tmp/a.png").unwrap().dest.as_deref(), Some("/tmp/a.png"));
    }

    #[test_case]
    fn extents_parse_and_flags_are_order_independent() {
        assert_eq!(parse("panel").unwrap().extent, Extent::Panel);
        assert_eq!(parse("chat").unwrap().extent, Extent::Chat);
        assert_eq!(parse("pane 3").unwrap().extent, Extent::Pane(3));
        assert_eq!(
            parse("region 10,20,30,40").unwrap().extent,
            Extent::Region { x: 10, y: 20, w: 30, h: 40 }
        );
        let a = parse("--cursor panel after 2s out.png").unwrap();
        let b = parse("out.png after 2s panel --cursor").unwrap();
        assert_eq!(a, b);
        assert!(a.cursor && a.delay_ms == 2000 && a.extent == Extent::Panel);
    }

    #[test_case]
    fn durations_default_to_seconds_and_are_bounded() {
        assert_eq!(parse_duration_ms("3").unwrap(), 3000);
        assert_eq!(parse_duration_ms("3s").unwrap(), 3000);
        assert_eq!(parse_duration_ms("250ms").unwrap(), 250);
        assert!(parse_duration_ms("9999").is_err(), "past the cap");
        assert!(parse_duration_ms("soon").is_err());
    }

    #[test_case]
    fn malformed_arguments_are_refused_with_a_reason() {
        for bad in [
            "pane",
            "pane x",
            "region",
            "region 1,2,3",
            "region 1,2,3,4,5",
            "region 1,2,0,4",
            "after",
            "--wat",
            "a.png b.png",
        ] {
            assert!(parse(bad).is_err(), "'{bad}' should be refused");
        }
    }

    #[test_case]
    fn a_region_is_clipped_to_the_surface_but_never_moved() {
        // Wholly inside: unchanged.
        assert_eq!(clamp_region(100, 100, 10, 10, 20, 20), Some((10, 10, 20, 20)));
        // Overhanging: clipped, origin kept.
        assert_eq!(clamp_region(100, 100, 90, 95, 50, 50), Some((90, 95, 10, 5)));
        // Exactly flush.
        assert_eq!(clamp_region(100, 100, 0, 0, 100, 100), Some((0, 0, 100, 100)));
        // Wholly outside, or degenerate: refused, not silently relocated.
        assert_eq!(clamp_region(100, 100, 100, 0, 10, 10), None);
        assert_eq!(clamp_region(100, 100, 0, 100, 10, 10), None);
        assert_eq!(clamp_region(100, 100, 0, 0, 0, 10), None);
    }

    #[test_case]
    fn a_bare_name_lands_in_downloads_and_gains_the_extension() {
        assert_eq!(normalize_dest("shot"), "/downloads/shot.png");
        assert_eq!(normalize_dest("shot.png"), "/downloads/shot.png");
        assert_eq!(normalize_dest("/tmp/a"), "/tmp/a.png");
        assert_eq!(normalize_dest("/tmp/a.png"), "/tmp/a.png");
        // A relative path with a slash is left for `resolve_path`, not
        // rewritten into /downloads.
        assert_eq!(normalize_dest("sub/a.png"), "sub/a.png");
    }

    #[test_case]
    fn the_saved_line_names_the_path_twice_so_open_is_one_paste_away() {
        let s = saved_line("/downloads/x.png", 1280, 800, 4096);
        assert!(s.contains("1280x800"), "{s}");
        assert!(s.contains("4096 bytes"), "{s}");
        assert!(s.ends_with("/open /downloads/x.png"), "{s}");
    }
}
