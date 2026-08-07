//! Toast banners — the geometry, timing and chime policy, as pure functions.
//!
//! The painter lives in `framebuffer::toast`, which is `#[cfg(not(test))]` and so
//! could carry no tests; everything decidable without a framebuffer is decided
//! here instead.
//!
//! ## Why a transient overlay is safe here, having argued it was not
//!
//! The original design note in this module's parent rejected a toast on three
//! grounds. Two of them dissolve once the banner is **top-right** rather than
//! anywhere:
//!
//! - *"It would land over the composer, where the human is typing."* The composer
//!   is at the bottom. The top-right corner is the status bar's own strip and the
//!   pane title row — nothing is being typed there.
//! - *"It must re-damage a rect on every pulse."* Only if it animates. A banner
//!   that is **static while shown** damages exactly twice: once when it appears
//!   and once when it is lifted. [`Toast::needs_repaint`] is what keeps that
//!   true, and the KMS flush cost is therefore two round trips per notification,
//!   not one per frame.
//!
//! The third — *"a transient overlay must save and restore the pixels beneath
//! it"* — is real, but the mechanism already exists and is proven: it is exactly
//! what the mouse cursor does (`Screen::cur_saved` + `cursor_restore`), on a
//! single-buffered framebuffer, on every frame. Reusing that shape costs one
//! `Vec<Rgb>` for the duration of the banner.

use alloc::string::String;
use alloc::vec::Vec;

/// How long a banner stays up, by severity.
///
/// An `Action` notification is a question waiting on the human, so it gets
/// noticeably longer than an `Info` that is merely a fact. Nothing waits
/// forever: a banner that must be dismissed is a modal, and this is not one.
pub fn dwell_ms(sev: super::Severity) -> u64 {
    match sev {
        super::Severity::Info => 4_000,
        super::Severity::Success => 4_000,
        super::Severity::Warn => 6_000,
        super::Severity::Error => 8_000,
        super::Severity::Action => 10_000,
    }
}

/// Banner width in **columns**, clamped to what the screen can give.
///
/// Capped rather than proportional: a banner that grows with the desktop reads
/// as a dialog, and at 4K a half-width one would be absurd. 44 columns is about
/// the width of a macOS notification at a comparable text size.
pub const WANT_COLS: usize = 44;
/// Never take more than this fraction of the desktop width, so a small logical
/// desktop (`/display set 640x480`) does not get a banner covering half of it.
pub const MAX_WIDTH_PERMILLE: u64 = 500;

/// The banner's column count for a desktop `cols` wide.
pub fn width_cols(desktop_cols: usize) -> usize {
    let cap = ((desktop_cols as u64) * MAX_WIDTH_PERMILLE / 1000) as usize;
    WANT_COLS.min(cap.max(12)).min(desktop_cols.max(1))
}

/// Longest body, in lines. Past this the banner is a wall of text nobody reads
/// in four seconds; the full text is in `/notify list` and the action pane.
pub const MAX_BODY_LINES: usize = 3;

/// The lines a banner shows: the title (with source), then a clipped body.
///
/// Pure so the wrapping is tested; the painter just draws what this returns.
pub fn lines(
    source: &str,
    title: &str,
    body: &str,
    count: u32,
    cols: usize,
) -> (String, Vec<String>) {
    let cols = cols.max(8);
    // The heading names the source, because "what is telling me this" is the
    // first thing a human needs and the reason the source is kernel-stamped.
    let repeat = if count > 1 { alloc::format!(" (x{count})") } else { String::new() };
    let head_room = cols.saturating_sub(repeat.chars().count());
    let mut head = crate::textfit::trunc(source, head_room);
    head.push_str(&repeat);

    let mut out: Vec<String> = Vec::new();
    // The title first, then the body — both wrapped, both truncated together, so
    // a long title cannot push the body off and a long body cannot hide the
    // title.
    for l in crate::textfit::wrap(title, cols) {
        if out.len() >= MAX_BODY_LINES {
            break;
        }
        out.push(l);
    }
    if !body.trim().is_empty() {
        for l in crate::textfit::wrap(body, cols) {
            if out.len() >= MAX_BODY_LINES {
                // Mark the truncation rather than ending mid-sentence as though
                // that were the whole message.
                if let Some(last) = out.last_mut() {
                    let room = cols.saturating_sub(1);
                    *last = crate::textfit::trunc(last, room);
                    last.push('\u{2026}');
                }
                break;
            }
            out.push(l);
        }
    }
    if out.is_empty() {
        out.push(String::new());
    }
    (head, out)
}

/// Whether a severity should ring, given the current policy.
///
/// Deliberately not "every notification": a five-second scheduled job that posts
/// on every run would be a metronome. Only what a human would want interrupting
/// them makes a sound.
pub fn should_chime(sev: super::Severity, policy: Policy) -> bool {
    if !policy.sound() {
        return false;
    }
    !matches!(sev, super::Severity::Info)
}

/// The chime, as `(hz, ms)` pairs played in order.
///
/// Two-note figures rather than one beep, because a single tone carries no
/// information: rising for something that went well, falling for something that
/// did not, and a longer three-note one for a decision waiting on you. Kept
/// inside a few hundred milliseconds — an OS sound that outlasts a glance is an
/// OS sound people turn off.
pub fn chime(sev: super::Severity) -> &'static [(u32, u32)] {
    match sev {
        // Never rings (see `should_chime`); present so the table is total.
        super::Severity::Info => &[(880, 60)],
        super::Severity::Success => &[(660, 70), (990, 90)],
        super::Severity::Warn => &[(740, 80), (620, 100)],
        super::Severity::Error => &[(560, 90), (420, 130)],
        super::Severity::Action => &[(700, 70), (900, 70), (1100, 110)],
    }
}

/// What the notification system is allowed to do.
///
/// Three states rather than a bool, because "stop making noise at me" and "stop
/// recording anything" are different requests and answering both with one switch
/// makes one of them wrong. `Mute` is the common one — the queue is still the
/// record of what happened, which is most of its value — and `Off` is the hard
/// kill the request "fully disable" actually means.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Policy {
    /// Banner, chime, chip, queue.
    #[default]
    On,
    /// Queue and chip only: no banner, no sound. Still a record.
    Mute,
    /// Nothing at all — not even stored. A post is dropped on the floor.
    Off,
}

impl Policy {
    pub fn as_str(self) -> &'static str {
        match self {
            Policy::On => "on",
            Policy::Mute => "mute",
            Policy::Off => "off",
        }
    }
    pub fn parse(s: &str) -> Option<Policy> {
        Some(match s.trim().to_ascii_lowercase().as_str() {
            "on" | "all" | "enabled" => Policy::On,
            "mute" | "muted" | "silent" | "quiet" => Policy::Mute,
            "off" | "disabled" | "none" => Policy::Off,
            _ => return None,
        })
    }
    /// Whether a post is recorded at all.
    pub fn records(self) -> bool {
        !matches!(self, Policy::Off)
    }
    /// Whether a banner is shown.
    pub fn banner(self) -> bool {
        matches!(self, Policy::On)
    }
    /// Whether a chime may play.
    pub fn sound(self) -> bool {
        matches!(self, Policy::On)
    }
    pub fn describe(self) -> &'static str {
        match self {
            Policy::On => "banner + sound + queue",
            Policy::Mute => "queue only (no banner, no sound)",
            Policy::Off => "fully disabled (nothing is recorded)",
        }
    }
}

/// A colour **role**, so the palette choice is decidable without a framebuffer.
///
/// The banner lives in `framebuffer/toast.rs`, which is `#[cfg(not(test))]` and
/// therefore cannot be tested — and the first version of it picked
/// `theme.sep_dim` for an `Info` heading. `sep_dim` is `#2e2c28`, a hairline
/// separator, on a `#252320` box: the heading was *there* and invisible. Naming
/// the role here rather than the colour there means "which text may be faint" is
/// a decision a test can hold.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Ink {
    /// The theme accent — draws the eye. For anything the human should look at.
    Accent,
    /// Ordinary foreground. Always legible.
    Normal,
    /// Secondary foreground: readable, but recedes behind `Normal`.
    Soft,
    /// A hairline/chrome tint. **Never text** — see [`heading_ink`].
    Faint,
}

/// The heading's colour role.
///
/// Never [`Ink::Faint`]: the heading is the line that says *who* is talking to
/// you, which is the first thing a human needs and the whole reason the source is
/// kernel-stamped. A notification whose heading cannot be read is a notification
/// that failed at its one job.
pub fn heading_ink(sev: super::Severity) -> Ink {
    match sev {
        super::Severity::Error | super::Severity::Warn | super::Severity::Action => Ink::Accent,
        super::Severity::Info | super::Severity::Success => Ink::Normal,
    }
}

/// The body's colour role — one step back from the heading, so the two read as a
/// hierarchy rather than a wall.
pub fn body_ink(_sev: super::Severity) -> Ink {
    Ink::Soft
}

/// The chrome's colour role: the outline and the severity stripe. This is the one
/// place a faint tint is right — it is a border, not a word.
pub fn chrome_ink(sev: super::Severity) -> Ink {
    match sev {
        super::Severity::Error | super::Severity::Warn | super::Severity::Action => Ink::Accent,
        super::Severity::Success => Ink::Soft,
        super::Severity::Info => Ink::Faint,
    }
}

/// The live banner: what is shown and until when.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Toast {
    pub id: u64,
    pub severity: super::Severity,
    pub head: String,
    pub lines: Vec<String>,
    /// Monotonic ms at which it should be lifted.
    pub until_ms: u64,
    /// Set once the painter has drawn it, so `upkeep` does not redraw a static
    /// banner on every pulse — which is what would make the KMS damage churn the
    /// original design note warned about real.
    pub painted: bool,
}

impl Toast {
    pub fn expired(&self, now_ms: u64) -> bool {
        now_ms >= self.until_ms
    }
    /// Whether the painter has work to do: draw it once, then nothing until it
    /// expires. Two damage rects per notification, not one per frame.
    pub fn needs_repaint(&self) -> bool {
        !self.painted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::notify::Severity;

    #[test_case]
    fn the_policy_switch_separates_silence_from_disabled() {
        // The distinction the request "fully disable" turns on: `Mute` keeps the
        // record, `Off` keeps nothing.
        assert!(Policy::On.records() && Policy::On.banner() && Policy::On.sound());
        assert!(Policy::Mute.records(), "muted still records — the queue is the point");
        assert!(!Policy::Mute.banner());
        assert!(!Policy::Mute.sound());
        assert!(!Policy::Off.records(), "off means nothing is stored");
        assert!(!Policy::Off.banner() && !Policy::Off.sound());
    }

    #[test_case]
    fn policy_round_trips_its_names_and_accepts_the_obvious_synonyms() {
        for p in [Policy::On, Policy::Mute, Policy::Off] {
            assert_eq!(Policy::parse(p.as_str()), Some(p));
            assert!(!p.describe().is_empty());
        }
        assert_eq!(Policy::parse("silent"), Some(Policy::Mute));
        assert_eq!(Policy::parse("quiet"), Some(Policy::Mute));
        assert_eq!(Policy::parse("disabled"), Some(Policy::Off));
        assert_eq!(Policy::parse("ON"), Some(Policy::On));
        assert_eq!(Policy::parse("sometimes"), None);
        assert_eq!(Policy::default(), Policy::On);
    }

    #[test_case]
    fn only_notifications_worth_interrupting_for_make_a_sound() {
        // An `Info` never rings: a five-second job posting every run would be a
        // metronome, and that is how an OS sound gets turned off for good.
        assert!(!should_chime(Severity::Info, Policy::On));
        for sev in [Severity::Success, Severity::Warn, Severity::Error, Severity::Action] {
            assert!(should_chime(sev, Policy::On), "{sev:?} should ring");
            assert!(!should_chime(sev, Policy::Mute), "{sev:?} must be silent when muted");
            assert!(!should_chime(sev, Policy::Off));
        }
    }

    #[test_case]
    fn every_severity_has_a_chime_and_none_of_them_outlast_a_glance() {
        for sev in [
            Severity::Info,
            Severity::Success,
            Severity::Warn,
            Severity::Error,
            Severity::Action,
        ] {
            let c = chime(sev);
            assert!(!c.is_empty(), "{sev:?} has no chime");
            let total: u32 = c.iter().map(|(_, ms)| *ms).sum();
            assert!(total <= 400, "{sev:?} rings for {total} ms — too long for an OS sound");
            for &(hz, ms) in c {
                assert!((200..=4000).contains(&hz), "{sev:?}: {hz} Hz is out of range");
                assert!(ms > 0);
            }
        }
        // Success rises, error falls — the direction is the information.
        let ok = chime(Severity::Success);
        assert!(ok[1].0 > ok[0].0, "success should rise");
        let err = chime(Severity::Error);
        assert!(err[1].0 < err[0].0, "error should fall");
    }

    #[test_case]
    fn dwell_scales_with_how_much_the_notification_wants() {
        assert!(dwell_ms(Severity::Action) > dwell_ms(Severity::Error));
        assert!(dwell_ms(Severity::Error) > dwell_ms(Severity::Warn));
        assert!(dwell_ms(Severity::Warn) > dwell_ms(Severity::Info));
        // Nothing waits forever — a banner that must be dismissed is a modal.
        for sev in [Severity::Info, Severity::Action] {
            assert!(dwell_ms(sev) <= 15_000);
        }
    }

    #[test_case]
    fn the_banner_is_capped_but_never_wider_than_the_desktop() {
        // A wide desktop gets the cap, not a proportional slab.
        assert_eq!(width_cols(200), WANT_COLS);
        assert_eq!(width_cols(400), WANT_COLS);
        // A narrow one gets at most half, so it does not cover the screen.
        assert!(width_cols(40) <= 20, "half of 40 columns");
        assert!(width_cols(20) <= 12);
        // Never wider than what exists, and never zero.
        for c in 1..80usize {
            let w = width_cols(c);
            assert!(w >= 1 && w <= c.max(1), "width_cols({c}) = {w}");
        }
    }

    #[test_case]
    fn banner_lines_fit_the_width_and_are_bounded() {
        let (head, body) = lines(
            "schedule:nightly-backup",
            "nightly-backup: err",
            "ping: no route to host. the gateway did not answer within the timeout, \
             and the previous three runs also failed for the same reason.",
            4,
            30,
        );
        assert!(
            crate::textfit::cols(&head) <= 30,
            "the heading overflowed: {head:?}"
        );
        assert!(head.contains("x4"), "the repeat count must survive: {head:?}");
        assert!(body.len() <= MAX_BODY_LINES, "{} lines", body.len());
        for l in &body {
            assert!(crate::textfit::cols(l) <= 30, "line overflowed: {l:?}");
        }
        // The clipped tail says it is clipped rather than ending mid-sentence as
        // though that were the whole message.
        assert!(body.last().unwrap().ends_with('\u{2026}'), "{:?}", body.last());
    }

    #[test_case]
    fn a_short_notification_shows_the_title_and_no_ellipsis() {
        let (head, body) = lines("kernel", "disk full", "", 1, 40);
        assert_eq!(head, "kernel", "no repeat suffix at a count of one");
        assert_eq!(body, alloc::vec![alloc::string::String::from("disk full")]);
    }

    #[test_case]
    fn a_multi_byte_banner_never_splits_a_character() {
        let (head, body) = lines("agent:9042", "\u{65e5}\u{672c}\u{8a9e}\u{306e}\u{901a}\u{77e5}", "h\u{e9}llo \u{65e5}\u{672c}", 1, 12);
        assert!(crate::textfit::cols(&head) <= 12);
        for l in &body {
            assert!(crate::textfit::cols(l) <= 12, "{l:?}");
            assert!(!l.contains('\u{fffd}'), "mangled a character: {l:?}");
        }
    }

    #[test_case]
    fn an_empty_notification_still_produces_a_drawable_banner() {
        // The painter indexes `lines`, so it must never be empty.
        let (_, body) = lines("", "", "", 0, 20);
        assert_eq!(body.len(), 1);
    }

    /// The bug this enum exists for: an `Info` heading was drawn in `sep_dim`,
    /// a hairline separator colour, on a box only six shades away from it. The
    /// text was there and unreadable.
    #[test_case]
    fn no_banner_text_is_ever_drawn_in_a_hairline_colour() {
        for sev in [
            Severity::Info,
            Severity::Success,
            Severity::Warn,
            Severity::Error,
            Severity::Action,
        ] {
            assert_ne!(
                heading_ink(sev),
                Ink::Faint,
                "{sev:?}: the heading names the source — it must always be legible"
            );
            assert_ne!(body_ink(sev), Ink::Faint, "{sev:?}: the body must be legible");
        }
    }

    #[test_case]
    fn the_heading_outranks_the_body_and_urgency_outranks_both() {
        // A hierarchy, not a wall: the heading is at least as prominent as the
        // body for every severity.
        fn rank(i: Ink) -> u8 {
            match i {
                Ink::Accent => 3,
                Ink::Normal => 2,
                Ink::Soft => 1,
                Ink::Faint => 0,
            }
        }
        for sev in [
            Severity::Info,
            Severity::Success,
            Severity::Warn,
            Severity::Error,
            Severity::Action,
        ] {
            assert!(
                rank(heading_ink(sev)) > rank(body_ink(sev)),
                "{sev:?}: the heading must stand out from the body"
            );
        }
        // What wants attention gets the accent; what does not, does not.
        for sev in [Severity::Warn, Severity::Error, Severity::Action] {
            assert_eq!(heading_ink(sev), Ink::Accent, "{sev:?} should draw the eye");
            assert_eq!(chrome_ink(sev), Ink::Accent);
        }
        for sev in [Severity::Info, Severity::Success] {
            assert_eq!(heading_ink(sev), Ink::Normal, "{sev:?} should be calm but readable");
            assert_ne!(chrome_ink(sev), Ink::Accent, "{sev:?} should not shout");
        }
    }

    #[test_case]
    fn a_static_banner_is_painted_once_and_lifted_once() {
        // The property that keeps the KMS damage cost at two round trips per
        // notification rather than one per `upkeep` pulse.
        let mut t = Toast {
            id: 1,
            severity: Severity::Warn,
            head: String::from("kernel"),
            lines: alloc::vec![String::from("hi")],
            until_ms: 5_000,
            painted: false,
        };
        assert!(t.needs_repaint(), "a fresh banner must be drawn");
        t.painted = true;
        assert!(!t.needs_repaint(), "…and then left alone");
        assert!(!t.expired(4_999));
        assert!(t.expired(5_000), "the deadline is inclusive");
        assert!(t.expired(9_999));
    }
}
