//! Notifications — the channel by which the OS tells the human something
//! happened while they were not looking.
//!
//! Until this existed, everything the machine wanted to say went to `ktrace`
//! (a log nobody reads) or `serial_println!` (scrollback that a `/clear` eats).
//! A daemon exhausting its restart budget, a scheduled run failing, a
//! background turn that needs approval — all of those are *events with a
//! recipient*, and a log is not that.
//!
//! ## Shape
//!
//! A bounded ring of at most [`CAP`] entries, newest last, persisted as JSON on
//! the store. Three properties are load-bearing and each is unit-tested:
//!
//! - **Repeats coalesce, they do not accumulate.** A notification carries a
//!   `dedup_key`; posting the same key again bumps `count` and `unix` and
//!   re-marks it unread rather than appending. Without this, a job running once
//!   a minute fills the ring in an hour and the one notification that mattered
//!   is gone.
//! - **The source is stamped by the kernel, never supplied by the poster.** An
//!   agent that could choose its own `source` could post as `kernel`, which is
//!   `TransferAuthority` wearing a label.
//! - **Agents may post, and may not list.** Write-only for agents removes the
//!   laundering channel entirely — a notification an agent posts and reads back
//!   would be untrusted content re-entering its own context with the tag
//!   stripped — for one line of policy rather than a tagging path.
//!
//! ## What is *not* here
//!
//! No toast/transient overlay. There is no compositor layer for one: it would
//! have to save and restore the pixels under itself, re-damage a rect on every
//! pulse, and it would land over the composer where the human is typing. The
//! status-bar chip is the ambient signal, its dropdown is the glance, the
//! action pane is the log, and [`crate::modal`] already exists for the rare
//! thing that must interrupt.
//!
//! All the logic here is pure or `Locked`-guarded and lives **outside**
//! `framebuffer/` (which is `#[cfg(not(test))]`), so the ring, the dedup, the
//! JSON round trip and the row/chip formatting are all covered by
//! `cargo xtask test`. The painter over in `framebuffer::views` is then pure
//! presentation over values these tests already pin.

use crate::json::Json;
use crate::mm::Locked;
use crate::synapse::fs as store;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

pub const CONFIG_PATH: &str = "/configs/core/notifications.json";

/// Ring capacity. 64 entries is roughly 8 KiB of JSON — one ext4 rewrite — and
/// more than a human scrolls in an action pane.
pub const CAP: usize = 64;

/// How long between opportunistic writes. A schedule firing once a minute must
/// not rewrite the store once a minute: the ext4 backend re-formats on sync,
/// which is the thrash `msgchan` documents. Losing at most 30 s of
/// notifications to a hard crash is the right side of that trade.
const SAVE_INTERVAL_MS: u64 = 30_000;

/// Per-source post budget, per minute. A notification pane must not be a
/// model-controlled spam surface; over-budget posts are folded into a single
/// coalesced "…and N more" entry rather than dropped silently.
pub const MAX_POSTS_PER_MIN: u32 = 6;

/// Longest title/body kept. Truncated at the API boundary rather than at paint
/// time, so the store cannot grow without bound from one enormous post.
pub const MAX_TITLE: usize = 120;
pub const MAX_BODY: usize = 800;

/// How much a notification wants from you, ordered by that.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Severity {
    /// Something happened. No action implied.
    Info,
    /// Something you asked for finished.
    Success,
    /// Something is off but the machine coped.
    Warn,
    /// Something failed.
    Error,
    /// **A decision is waiting on you.** The reason this variant exists: an
    /// unattended scheduled run that hit a call needing human approval cannot
    /// raise a modal (nobody is there), so it posts this instead.
    Action,
}

impl Severity {
    pub fn as_str(self) -> &'static str {
        match self {
            Severity::Info => "info",
            Severity::Success => "ok",
            Severity::Warn => "warn",
            Severity::Error => "error",
            Severity::Action => "action",
        }
    }

    pub fn parse(s: &str) -> Option<Severity> {
        Some(match s.trim().to_ascii_lowercase().as_str() {
            "info" | "i" => Severity::Info,
            "ok" | "success" | "done" => Severity::Success,
            "warn" | "warning" => Severity::Warn,
            "error" | "err" | "fail" => Severity::Error,
            "action" | "approve" | "needs-approval" => Severity::Action,
            _ => return None,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Notification {
    pub id: u64,
    pub severity: Severity,
    /// Kernel-stamped origin: `kernel`, `schedule:<name>`, `agent:<id>`,
    /// `service:<name>`. Never taken from a poster's arguments.
    pub source: String,
    pub title: String,
    pub body: String,
    pub unix: i64,
    pub read: bool,
    /// A `/command` line the human can run from the dropdown — never arbitrary
    /// code, and never executed automatically.
    pub action: Option<String>,
    /// Coalescing key. Empty means "never coalesce", which is right for a
    /// one-off but wrong for anything periodic.
    pub dedup_key: String,
    /// How many times this entry has been posted.
    pub count: u32,
}

static NOTIFS: Locked<Vec<Notification>> = Locked::new(Vec::new());
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
static DIRTY: AtomicBool = AtomicBool::new(false);
static LAST_SAVE_MS: AtomicU64 = AtomicU64::new(0);
/// `(window_start_ms, posts_in_window)` for the rate limiter. Not per-source:
/// one budget for the whole surface is what protects the *human*, and a
/// per-source budget would let ten agents spend ten budgets.
static BUDGET: Locked<(u64, u32)> = Locked::new((0, 0));

// ---------------------------------------------------------------------------
// Pure logic
// ---------------------------------------------------------------------------

/// Truncate to `max` **characters**, never splitting a UTF-8 sequence.
///
/// `max == 0` is the empty string, not an ellipsis: an ellipsis is one character
/// wide, so returning one for a zero-width budget overflows the caller by
/// exactly the amount they asked to avoid — which is how `summary_line` came to
/// emit a 16-character row for a 12-column pane.
fn clip(s: &str, max: usize) -> String {
    if max == 0 {
        return String::new();
    }
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max - 1).collect();
    out.push('…');
    out
}

/// Insert `n` into `ring`, coalescing onto a matching `dedup_key` and dropping
/// the oldest entry once `cap` is exceeded. Returns the id of the entry that
/// now represents this post — the *existing* id on a coalesce, so a caller that
/// wants to mark it read later has a stable handle.
///
/// Pure over an explicit `ring` so the whole policy is testable without the
/// global.
pub fn push_into(ring: &mut Vec<Notification>, n: Notification, cap: usize) -> u64 {
    if !n.dedup_key.is_empty() {
        if let Some(e) = ring.iter_mut().find(|e| e.dedup_key == n.dedup_key) {
            e.count = e.count.saturating_add(n.count.max(1));
            e.unix = n.unix;
            e.severity = n.severity.max(e.severity); // a warn repeat of an info is a warn
            e.title = n.title;
            e.body = n.body;
            e.action = n.action;
            // A repeat is news again: an entry the human dismissed and that then
            // happened once more must come back, or a recurring failure goes
            // quiet after the first glance.
            e.read = false;
            return e.id;
        }
    }
    let id = n.id;
    ring.push(n);
    while ring.len() > cap {
        ring.remove(0);
    }
    id
}

pub fn unread_count_of(ring: &[Notification]) -> usize {
    ring.iter().filter(|n| !n.read).count()
}

/// The status-bar chip's text. **Empty at zero** so `ui_config::expand` drops
/// the following separator with it and a machine with nothing to say has a
/// byte-identical status bar.
pub fn chip_text(unread: usize) -> String {
    if unread == 0 {
        return String::new();
    }
    alloc::format!("{} {}", crate::icons::fa::BELL, unread)
}

/// A coarse, human "when": `now` / `3m` / `2h` / `Aug 4`. Deliberately not a
/// timestamp — a notification list is read by recency, and a column of
/// identical dates carries no information.
pub fn relative_age(now_unix: i64, then_unix: i64) -> String {
    let d = now_unix - then_unix;
    if d < 0 {
        // The clock moved backwards (NTP, `/datetime set`). Say so rather than
        // rendering a negative age or clamping it to "now", which would hide a
        // real fact about the machine.
        return String::from("ahead");
    }
    if d < 60 {
        return String::from("now");
    }
    if d < 3600 {
        return alloc::format!("{}m", d / 60);
    }
    if d < 86400 {
        return alloc::format!("{}h", d / 3600);
    }
    let (_, mo, dd, _, _, _, _) = crate::clock::civil_from_unix(then_unix);
    const MON: [&str; 12] =
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    alloc::format!("{} {}", MON[(mo as usize).clamp(1, 12) - 1], dd)
}

/// The glyph for a severity — the value the pane painter and the dropdown both
/// use, defined here so the two cannot drift and so it is testable.
pub fn severity_icon(s: Severity) -> char {
    use crate::icons::fa;
    match s {
        Severity::Info => fa::CIRCLE_INFO,
        Severity::Success => fa::SQUARE_CHECK,
        Severity::Warn => fa::TRIANGLE_EXCLAMATION,
        Severity::Error => fa::CIRCLE_XMARK,
        Severity::Action => fa::BELL,
    }
}

/// One row of the notification pane, fitted to `cols` columns.
///
/// The budget is spent in priority order: the unread mark and severity glyph
/// (which say whether you care), the age, then the title, then the repeat count.
/// The whole assembled row is clipped as a backstop — at a width too narrow for
/// even the fixed prefix there is nothing to negotiate, and a row that overflows
/// its pane corrupts the one next to it.
pub fn summary_line(n: &Notification, now_unix: i64, cols: usize) -> String {
    let mark = if n.read { ' ' } else { '•' };
    let age = relative_age(now_unix, n.unix);
    let mut head = alloc::format!("{}{} {:>5}  ", mark, severity_icon(n.severity), age);
    let repeat = if n.count > 1 { alloc::format!(" (x{})", n.count) } else { String::new() };
    let used = head.chars().count() + repeat.chars().count();
    let room = cols.saturating_sub(used);
    head.push_str(&clip(&n.title, room));
    head.push_str(&repeat);
    clip(&head, cols)
}

pub fn to_json(ring: &[Notification]) -> Json {
    Json::Arr(
        ring.iter()
            .map(|n| {
                Json::Obj(alloc::vec![
                    (String::from("id"), Json::Num(n.id as f64)),
                    (String::from("severity"), Json::Str(n.severity.as_str().to_string())),
                    (String::from("source"), Json::Str(n.source.clone())),
                    (String::from("title"), Json::Str(n.title.clone())),
                    (String::from("body"), Json::Str(n.body.clone())),
                    (String::from("unix"), Json::Num(n.unix as f64)),
                    (String::from("read"), Json::Bool(n.read)),
                    (
                        String::from("action"),
                        match &n.action {
                            Some(a) => Json::Str(a.clone()),
                            None => Json::Str(String::new()),
                        }
                    ),
                    (String::from("dedup"), Json::Str(n.dedup_key.clone())),
                    (String::from("count"), Json::Num(n.count as f64)),
                ])
            })
            .collect(),
    )
}

/// Parse the stored ring. Every field has a default, so an older or truncated
/// file loads rather than being discarded — the `msgchan` forward-compat
/// property, which matters more here because losing the file loses the
/// human's unread queue.
pub fn from_json(text: &str) -> Vec<Notification> {
    let Some(Json::Arr(arr)) = Json::parse(text) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for item in arr {
        let Json::Obj(pairs) = item else { continue };
        let get = |k: &str| pairs.iter().find(|(a, _)| a == k).map(|(_, v)| v);
        let title = get("title").and_then(|v| v.as_str()).unwrap_or("").to_string();
        if title.is_empty() {
            continue; // a notification with nothing to say is not one
        }
        let action = get("action").and_then(|v| v.as_str()).unwrap_or("").to_string();
        out.push(Notification {
            id: get("id").and_then(|v| v.as_i64()).unwrap_or(0).max(0) as u64,
            severity: get("severity")
                .and_then(|v| v.as_str())
                .and_then(Severity::parse)
                .unwrap_or(Severity::Info),
            source: get("source").and_then(|v| v.as_str()).unwrap_or("kernel").to_string(),
            title,
            body: get("body").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            unix: get("unix").and_then(|v| v.as_i64()).unwrap_or(0),
            read: get("read").and_then(|v| v.as_bool()).unwrap_or(false),
            action: if action.is_empty() { None } else { Some(action) },
            dedup_key: get("dedup").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            count: get("count").and_then(|v| v.as_i64()).unwrap_or(1).clamp(1, u32::MAX as i64)
                as u32,
        });
    }
    out
}

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

/// Whether this post fits inside the per-minute budget. Over budget, the caller
/// rewrites it as a coalesced "N more" entry rather than dropping it.
fn within_budget(now_ms: u64) -> bool {
    BUDGET.with(|(start, n)| {
        if now_ms.saturating_sub(*start) >= 60_000 {
            *start = now_ms;
            *n = 0;
        }
        if *n >= MAX_POSTS_PER_MIN {
            return false;
        }
        *n += 1;
        true
    })
}

/// Post a notification. `source` must already be a kernel-stamped origin — the
/// tool path in `shell::notify` is what enforces that for agents; direct
/// callers here are kernel and driver code.
pub fn post(sev: Severity, source: &str, title: &str, body: &str) -> u64 {
    post_full(sev, source, title, body, None, "")
}

/// Post with an offered `/command` and an explicit coalescing key.
pub fn post_action(
    sev: Severity,
    source: &str,
    title: &str,
    body: &str,
    action: &str,
    dedup_key: &str,
) -> u64 {
    post_full(sev, source, title, body, Some(action), dedup_key)
}

/// Post with an explicit coalescing key and no offered action.
pub fn post_keyed(sev: Severity, source: &str, title: &str, body: &str, dedup_key: &str) -> u64 {
    post_full(sev, source, title, body, None, dedup_key)
}

fn post_full(
    sev: Severity,
    source: &str,
    title: &str,
    body: &str,
    action: Option<&str>,
    dedup_key: &str,
) -> u64 {
    let now_ms = crate::arch::now_ms();
    let (sev, title, body, dedup_key) = if within_budget(now_ms) {
        (sev, clip(title, MAX_TITLE), clip(body, MAX_BODY), dedup_key.to_string())
    } else {
        // Over budget: fold into one entry so the flood is visible as a flood
        // without becoming the whole pane. Deliberately keeps the severity.
        (
            sev,
            String::from("many notifications suppressed"),
            alloc::format!("more than {MAX_POSTS_PER_MIN} posts in a minute; latest: {title}"),
            String::from("rate-limited"),
        )
    };
    let n = Notification {
        id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
        severity: sev,
        source: source.to_string(),
        title,
        body,
        unix: crate::clock::now_unix(),
        read: false,
        action: action.filter(|a| !a.is_empty()).map(|a| a.to_string()),
        dedup_key,
        count: 1,
    };
    crate::ktrace::log_fmt(format_args!(
        "notify: [{}] {} — {}",
        n.severity.as_str(),
        n.source,
        n.title
    ));
    let id = NOTIFS.with(|v| push_into(v, n, CAP));
    DIRTY.store(true, Ordering::Relaxed);
    // Repaint the pane only if it is actually on screen — a notification posted
    // while the tab is closed must not force a relayout.
    refresh_pane_if_open();
    id
}

#[cfg(not(test))]
fn refresh_pane_if_open() {
    if crate::framebuffer::is_notifications() {
        crate::shell::refresh_notifications();
    }
}

#[cfg(test)]
fn refresh_pane_if_open() {}

pub fn unread_count() -> usize {
    // An in-RAM `Vec` scan of at most 64 entries, so unlike `battery::cached`
    // this needs no 5-second cache even though the status bar paints ~1 Hz.
    NOTIFS.with(|v| unread_count_of(v))
}

pub fn len() -> usize {
    NOTIFS.with(|v| v.len())
}

/// Newest **first** — the order a human reads a notification list.
pub fn list() -> Vec<Notification> {
    NOTIFS.with(|v| {
        let mut out = v.clone();
        out.reverse();
        out
    })
}

pub fn get(id: u64) -> Option<Notification> {
    NOTIFS.with(|v| v.iter().find(|n| n.id == id).cloned())
}

pub fn mark_read(id: u64) -> bool {
    let hit = NOTIFS.with(|v| match v.iter_mut().find(|n| n.id == id) {
        Some(n) => {
            n.read = true;
            true
        }
        None => false,
    });
    if hit {
        save();
        refresh_pane_if_open();
    }
    hit
}

pub fn mark_all_read() -> usize {
    let n = NOTIFS.with(|v| {
        let mut c = 0;
        for e in v.iter_mut() {
            if !e.read {
                e.read = true;
                c += 1;
            }
        }
        c
    });
    if n > 0 {
        save();
        refresh_pane_if_open();
    }
    n
}

pub fn clear() -> usize {
    let n = NOTIFS.with(|v| {
        let n = v.len();
        v.clear();
        n
    });
    save();
    refresh_pane_if_open();
    n
}

/// Remove one entry. Returns whether it existed.
pub fn dismiss(id: u64) -> bool {
    let hit = NOTIFS.with(|v| {
        let before = v.len();
        v.retain(|n| n.id != id);
        v.len() != before
    });
    if hit {
        save();
        refresh_pane_if_open();
    }
    hit
}

/// Load the persisted ring. Call once at boot, after the store is mounted.
pub fn load() {
    let Some(bytes) = store::read(CONFIG_PATH) else {
        return;
    };
    let Ok(text) = core::str::from_utf8(&bytes) else {
        return;
    };
    let ring = from_json(text);
    // Re-anchor the id counter past everything on disk, the way `id_minter!`
    // does with `fetch_max`: a fresh boot must not mint an id that collides
    // with a loaded one, or `mark_read` marks the wrong entry.
    let max = ring.iter().map(|n| n.id).max().unwrap_or(0);
    NEXT_ID.fetch_max(max + 1, Ordering::Relaxed);
    let n = ring.len();
    NOTIFS.with(|v| *v = ring);
    crate::ktrace::log_fmt(format_args!("notify: loaded {n} notification(s) from {CONFIG_PATH}"));
}

pub fn save() {
    let text = NOTIFS.with(|v| to_json(v).to_pretty());
    store::write(CONFIG_PATH, text.as_bytes());
    DIRTY.store(false, Ordering::Relaxed);
    LAST_SAVE_MS.store(crate::arch::now_ms(), Ordering::Relaxed);
}

/// Write the ring if it changed and enough time has passed. Pumped from
/// `shell::upkeep` — see [`SAVE_INTERVAL_MS`] for why this is not a write per
/// post.
pub fn save_if_dirty() {
    if !DIRTY.load(Ordering::Relaxed) {
        return;
    }
    let now = crate::arch::now_ms();
    if now.saturating_sub(LAST_SAVE_MS.load(Ordering::Relaxed)) < SAVE_INTERVAL_MS {
        return;
    }
    save();
}

/// Flush unconditionally — for `/exit` and `/restart`, where the next 30 s
/// never arrive.
pub fn flush() {
    if DIRTY.load(Ordering::Relaxed) {
        save();
    }
}

#[cfg(test)]
pub fn reset_for_test() {
    NOTIFS.with(|v| v.clear());
    NEXT_ID.store(1, Ordering::Relaxed);
    BUDGET.with(|b| *b = (0, 0));
    DIRTY.store(false, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(id: u64, title: &str, dedup: &str) -> Notification {
        Notification {
            id,
            severity: Severity::Info,
            source: String::from("kernel"),
            title: title.to_string(),
            body: String::new(),
            unix: 1_770_000_000,
            read: false,
            action: None,
            dedup_key: dedup.to_string(),
            count: 1,
        }
    }

    #[test_case]
    fn the_ring_caps_and_drops_the_oldest() {
        let mut ring = Vec::new();
        for i in 0..(CAP as u64 + 10) {
            push_into(&mut ring, n(i + 1, &alloc::format!("t{i}"), ""), CAP);
        }
        assert_eq!(ring.len(), CAP);
        // The first 10 fell off the front; the newest is last.
        assert_eq!(ring[0].title, "t10");
        assert_eq!(ring[CAP - 1].title, alloc::format!("t{}", CAP + 9));
    }

    #[test_case]
    fn a_repeat_with_the_same_key_coalesces_instead_of_growing() {
        let mut ring = Vec::new();
        let first = push_into(&mut ring, n(1, "disk full", "disk"), CAP);
        // The human reads it.
        ring[0].read = true;
        let mut again = n(2, "disk full", "disk");
        again.unix = 1_770_000_600;
        let second = push_into(&mut ring, again, CAP);
        assert_eq!(ring.len(), 1, "a coalesce must not append");
        assert_eq!(first, second, "the existing id is the stable handle");
        assert_eq!(ring[0].count, 2);
        assert_eq!(ring[0].unix, 1_770_000_600, "the timestamp advances");
        assert!(!ring[0].read, "a recurrence is news again");
    }

    #[test_case]
    fn an_empty_key_never_coalesces() {
        let mut ring = Vec::new();
        push_into(&mut ring, n(1, "same text", ""), CAP);
        push_into(&mut ring, n(2, "same text", ""), CAP);
        assert_eq!(ring.len(), 2, "identical text with no key is two events");
    }

    #[test_case]
    fn a_coalesce_takes_the_higher_severity() {
        let mut ring = Vec::new();
        push_into(&mut ring, n(1, "flaky", "j"), CAP);
        let mut worse = n(2, "flaky", "j");
        worse.severity = Severity::Error;
        push_into(&mut ring, worse, CAP);
        assert_eq!(ring[0].severity, Severity::Error);
        // …and does not fall back down again.
        push_into(&mut ring, n(3, "flaky", "j"), CAP);
        assert_eq!(ring[0].severity, Severity::Error);
    }

    #[test_case]
    fn unread_count_tracks_the_read_flag() {
        let mut ring = Vec::new();
        for i in 0..5u64 {
            push_into(&mut ring, n(i + 1, "x", ""), CAP);
        }
        assert_eq!(unread_count_of(&ring), 5);
        ring[2].read = true;
        assert_eq!(unread_count_of(&ring), 4);
        for e in ring.iter_mut() {
            e.read = true;
        }
        assert_eq!(unread_count_of(&ring), 0);
    }

    #[test_case]
    fn chip_text_is_empty_at_zero_so_the_template_drops_the_separator() {
        assert_eq!(chip_text(0), "");
        assert!(chip_text(3).ends_with('3'));
        assert!(chip_text(3).starts_with(crate::icons::fa::BELL));
    }

    /// The status-bar contract from the other side: a zero-unread machine must
    /// render byte-identically to one whose template never had the variable.
    ///
    /// Note the spacing this pins. `expand` swallows the separator that *follows*
    /// an empty value, so `A  ${empty}  B` collapses to `A  B` — which is why the
    /// shipped default puts `${notifications}` between two two-space runs and
    /// not, say, after a single space. Get that wrong and a machine with nothing
    /// unread has a subtly different status bar from one built before the
    /// variable existed.
    #[test_case]
    fn an_empty_chip_leaves_no_stray_separator_in_the_status_template() {
        let resolve = |v: &str| -> String {
            match v {
                "battery" => String::from("bat 82%"),
                "notifications" => chip_text(0),
                "datetime_short" => String::from("10:30"),
                _ => String::new(),
            }
        };
        let with =
            crate::ui_config::expand("${battery}  ${notifications}  ${datetime_short}", &resolve);
        let without = crate::ui_config::expand("${battery}  ${datetime_short}", &resolve);
        assert_eq!(with, without, "an empty variable must take its separator with it");
        // And the live default must be exactly that shape, or the property above
        // is true of a template nobody ships.
        let d = crate::ui_config::UiConfig::default().status_right;
        assert!(
            d.contains("  ${notifications}  "),
            "the default template must surround ${{notifications}} with two-space runs: {d}"
        );
        // With something unread it does appear, and the separators are intact.
        let resolve_unread = |v: &str| -> String {
            match v {
                "battery" => String::from("bat 82%"),
                "notifications" => chip_text(3),
                "datetime_short" => String::from("10:30"),
                _ => String::new(),
            }
        };
        let shown = crate::ui_config::expand(
            "${battery}  ${notifications}  ${datetime_short}",
            &resolve_unread,
        );
        assert!(shown.contains(&chip_text(3)), "{shown}");
        assert!(shown.ends_with("  10:30"), "{shown}");
    }

    #[test_case]
    fn relative_age_is_coarse_and_handles_a_clock_that_moved_back() {
        let t = 1_770_000_000i64;
        assert_eq!(relative_age(t, t), "now");
        assert_eq!(relative_age(t + 59, t), "now");
        assert_eq!(relative_age(t + 60, t), "1m");
        assert_eq!(relative_age(t + 3599, t), "59m");
        assert_eq!(relative_age(t + 3600, t), "1h");
        assert_eq!(relative_age(t + 86_399, t), "23h");
        // Past a day it becomes a date rather than a huge hour count.
        let d = relative_age(t + 200_000, t);
        assert!(d.contains(' ') && !d.ends_with('h'), "expected a date, got {d}");
        // NTP corrected the clock backwards: say so, don't render a negative.
        assert_eq!(relative_age(t - 5, t), "ahead");
    }

    #[test_case]
    fn summary_line_fits_the_column_count_and_never_splits_a_char() {
        let mut e = n(1, "héllo 日本 a very long title that will not fit at all", "");
        e.count = 7;
        for cols in [12usize, 20, 40, 80] {
            let s = summary_line(&e, e.unix, cols);
            assert!(
                s.chars().count() <= cols,
                "row of {} chars exceeded {cols} cols: {s}",
                s.chars().count()
            );
            // Valid UTF-8 by construction (it is a `String`), so the real check
            // is that the multi-byte chars were not mangled into replacements.
            assert!(!s.contains('\u{fffd}'), "mangled a multi-byte char: {s}");
        }
        // At a realistic pane width the repeat count survives — it is the part
        // that says "this keeps happening", which is more useful than the tail of
        // the title. At 12 columns nothing but the mark and the age fits, and
        // that is reported by fitting rather than by overflowing.
        assert!(summary_line(&e, e.unix, 40).contains("x7"));
        assert!(summary_line(&e, e.unix, 80).contains("x7"));
        // A zero-width budget is empty, not a lone ellipsis.
        assert_eq!(summary_line(&e, e.unix, 0), "");
    }

    #[test_case]
    fn an_unread_row_is_marked_and_a_read_one_is_not() {
        let mut e = n(1, "hello", "");
        assert!(summary_line(&e, e.unix, 40).starts_with('•'));
        e.read = true;
        assert!(!summary_line(&e, e.unix, 40).starts_with('•'));
    }

    #[test_case]
    fn json_round_trips_every_field() {
        let mut ring = Vec::new();
        let mut a = n(7, "scheduled run failed", "schedule:nightly");
        a.severity = Severity::Error;
        a.body = String::from("ping: no route to host");
        a.source = String::from("schedule:nightly");
        a.action = Some(String::from("/schedule run nightly"));
        a.count = 3;
        a.read = true;
        ring.push(a.clone());
        ring.push(n(8, "plain", ""));
        let back = from_json(&to_json(&ring).to_pretty());
        assert_eq!(back, ring);
    }

    #[test_case]
    fn from_json_tolerates_a_missing_field_and_rejects_a_titleless_entry() {
        let text = r#"[
          {"title":"minimal"},
          {"id":9,"title":"","body":"no title"},
          {"id":10,"title":"weird sev","severity":"nonsense","count":0}
        ]"#;
        let ring = from_json(text);
        assert_eq!(ring.len(), 2, "the titleless entry is dropped");
        assert_eq!(ring[0].title, "minimal");
        assert_eq!(ring[0].severity, Severity::Info, "missing severity defaults");
        assert_eq!(ring[0].count, 1, "missing count defaults to one");
        assert!(ring[0].action.is_none(), "an empty action string is None");
        assert_eq!(ring[1].severity, Severity::Info, "an unknown severity defaults");
        assert_eq!(ring[1].count, 1, "count is clamped up out of zero");
    }

    #[test_case]
    fn from_json_on_garbage_is_empty_not_a_panic() {
        assert!(from_json("").is_empty());
        assert!(from_json("not json").is_empty());
        assert!(from_json("{}").is_empty(), "an object is not the ring");
        assert!(from_json("[1,2,3]").is_empty());
    }

    #[test_case]
    fn severity_parse_round_trips_as_str() {
        for s in [
            Severity::Info,
            Severity::Success,
            Severity::Warn,
            Severity::Error,
            Severity::Action,
        ] {
            assert_eq!(Severity::parse(s.as_str()), Some(s));
        }
        assert_eq!(Severity::parse("WARNING"), Some(Severity::Warn));
        assert_eq!(Severity::parse("nope"), None);
    }

    #[test_case]
    fn action_outranks_error_so_a_decision_sorts_to_the_top() {
        // The ordering is load-bearing for the coalesce rule above and for any
        // future sort: "you must decide" is the most demanding severity.
        assert!(Severity::Action > Severity::Error);
        assert!(Severity::Error > Severity::Warn);
        assert!(Severity::Warn > Severity::Success);
        assert!(Severity::Success > Severity::Info);
    }

    #[test_case]
    fn oversized_text_is_clipped_at_the_api_boundary() {
        let long: String = core::iter::repeat('x').take(5_000).collect();
        assert_eq!(clip(&long, MAX_TITLE).chars().count(), MAX_TITLE);
        assert!(clip(&long, MAX_TITLE).ends_with('…'));
        // Multi-byte input must not be split.
        let kana: String = core::iter::repeat('日').take(300).collect();
        let c = clip(&kana, 10);
        assert_eq!(c.chars().count(), 10);
        assert!(c.chars().take(9).all(|ch| ch == '日'));
    }

    #[test_case]
    fn the_live_api_posts_lists_reads_and_clears() {
        reset_for_test();
        let a = post(Severity::Info, "kernel", "first", "body one");
        let b = post(Severity::Error, "service:ssh", "second", "body two");
        assert_eq!(len(), 2);
        assert_eq!(unread_count(), 2);
        // Newest first.
        let l = list();
        assert_eq!(l[0].id, b);
        assert_eq!(l[1].id, a);
        assert_eq!(l[0].source, "service:ssh");
        assert!(mark_read(a));
        assert!(!mark_read(9_999), "an unknown id is reported, not silently ok");
        assert_eq!(unread_count(), 1);
        assert_eq!(mark_all_read(), 1);
        assert_eq!(unread_count(), 0);
        assert_eq!(mark_all_read(), 0, "idempotent");
        assert!(dismiss(b));
        assert_eq!(len(), 1);
        assert_eq!(clear(), 1);
        assert_eq!(len(), 0);
        reset_for_test();
    }

    #[test_case]
    fn posts_beyond_the_budget_coalesce_rather_than_growing_the_ring() {
        reset_for_test();
        for i in 0..(MAX_POSTS_PER_MIN + 20) {
            post_keyed(Severity::Info, "agent:42", &alloc::format!("spam {i}"), "", "");
        }
        // The in-budget posts are distinct; everything after folds into the one
        // rate-limited entry, so the ring cannot be flooded.
        assert_eq!(
            len(),
            MAX_POSTS_PER_MIN as usize + 1,
            "expected {} distinct plus one suppression entry",
            MAX_POSTS_PER_MIN
        );
        let suppressed = list().into_iter().find(|n| n.dedup_key == "rate-limited");
        let s = suppressed.expect("a suppression entry must exist");
        assert_eq!(s.count, 20, "every over-budget post is counted");
        reset_for_test();
    }
}
