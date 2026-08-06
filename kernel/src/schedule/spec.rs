//! Recurrence: parsing, the next-fire calculation, and the whole missed-run
//! policy — **pure**, with every time-dependent input passed in.
//!
//! Nothing here touches `crate::arch`, `crate::clock`'s globals, the store or
//! the framebuffer, which is what makes the DST cases, the clock-jump detector
//! and the catch-up rules unit-testable under `cargo xtask test`. Only the
//! calendar arithmetic is borrowed from [`crate::clock`] (`unix_from_civil` /
//! `civil_from_unix`), and those are already pure.
//!
//! ## Why this is not a five-field cron
//!
//! Because the wall clock on this machine may be **fiction**. With no readable
//! RTC, `clock::DEFAULT_UNIX` puts the machine in January 2026 until `/ntp` or
//! `/datetime` runs, so a grammar in which *every* schedule is a calendar
//! expression makes every schedule depend on that lie, and gives you no way to
//! say "run every ninety seconds regardless of what year this machine thinks it
//! is".
//!
//! So the grammar splits along exactly that line:
//!
//! - [`Recurrence::Every`] is measured on the **monotonic** timebase. It is
//!   correct on a fictional clock, and it is the only recurrence a 30-second
//!   end-to-end test can observe.
//! - [`Recurrence::Daily`] / [`Recurrence::Monthly`] / [`Recurrence::Once`] are
//!   calendar, and are **held rather than fired** while the clock is untrusted
//!   ([`needs_wall_clock`], [`evaluate`]). Held, and *reported* as held — "my
//!   schedule didn't run" otherwise has three indistinguishable causes on a
//!   machine you may not be able to attach a debugger to.
//!
//! The secondary reason is that cron's ranges/steps/lists is a few hundred lines
//! of parser plus a matcher that has to step minute by minute to find the next
//! fire, where this restricted form has a closed-form [`next_due`] that is
//! exhaustively testable.

use alloc::string::{String, ToString};

/// Shortest `every` interval. Exists at exactly this value so an end-to-end test
/// can watch a schedule fire inside a 30-second budget; anything shorter would
/// be a busy loop wearing a schedule's clothes.
pub const MIN_EVERY_SECS: i64 = 5;

/// How far the wall clock must move, relative to the monotonic clock, before it
/// counts as a *jump* (NTP correction, `/datetime set`) rather than drift.
pub const JUMP_TOLERANCE_SECS: i64 = 120;

/// Days of lookahead [`next_due`] searches for a `Daily` match. Seven covers any
/// non-empty day-of-week mask; the eighth absorbs a DST fold.
const DAILY_LOOKAHEAD_DAYS: i64 = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Recurrence {
    /// `every <n>(s|m|h|d)` — a monotonic interval, wall-clock independent.
    Every { secs: i64 },
    /// `at HH:MM [daily|weekdays|mon,tue,…]` — local wall clock.
    ///
    /// `dow_mask` bit 0 is **Sunday**, matching the weekday
    /// [`crate::clock::civil_from_unix`] returns. `0x7f` is every day.
    Daily { hour: u8, min: u8, dow_mask: u8 },
    /// `on <1-31> HH:MM` — monthly, clamped to the month's real length.
    Monthly { dom: u8, hour: u8, min: u8 },
    /// `in <n><unit>` / `once <unix|ISO8601>` — fires once, then self-disables.
    Once { at_unix: i64 },
}

/// What to do about fires that were missed while the machine was off.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Catchup {
    /// Skip them. The default: recompute the next fire from now and say how
    /// many were missed.
    Skip,
    /// Run **once**, however many nominal fires were missed.
    Once,
}

// NB there is deliberately no `Catchup::All`. A `/command` job replayed 400
// times because a laptop was shut for a month is a self-inflicted denial of
// service, and an `Action::Prompt` job replayed 400 times is 400 inferences on a
// cooperative scheduler with one model.

impl Catchup {
    pub fn as_str(self) -> &'static str {
        match self {
            Catchup::Skip => "skip",
            Catchup::Once => "once",
        }
    }
    pub fn parse(s: &str) -> Option<Catchup> {
        match s.trim() {
            "skip" | "none" => Some(Catchup::Skip),
            "once" | "one" => Some(Catchup::Once),
            _ => None,
        }
    }
}

/// When a run should produce a notification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NotifyOn {
    Always,
    /// Only when the output differs from last time — the useful default for a
    /// monitoring job, which is silent until something changes.
    OnChange,
    OnError,
    Never,
}

impl NotifyOn {
    pub fn as_str(self) -> &'static str {
        match self {
            NotifyOn::Always => "always",
            NotifyOn::OnChange => "on_change",
            NotifyOn::OnError => "on_error",
            NotifyOn::Never => "never",
        }
    }
    pub fn parse(s: &str) -> Option<NotifyOn> {
        match s.trim() {
            "always" | "all" => Some(NotifyOn::Always),
            "on_change" | "change" | "changed" => Some(NotifyOn::OnChange),
            "on_error" | "error" | "errors" | "fail" => Some(NotifyOn::OnError),
            "never" | "silent" | "off" => Some(NotifyOn::Never),
            _ => None,
        }
    }
}

const DOW_NAMES: [&str; 7] = ["sun", "mon", "tue", "wed", "thu", "fri", "sat"];
const ALL_DAYS: u8 = 0x7f;
const WEEKDAYS_MASK: u8 = 0b0111_1110; // Mon..Fri

/// Whether this recurrence is meaningless until the wall clock is trustworthy.
pub fn needs_wall_clock(r: Recurrence) -> bool {
    !matches!(r, Recurrence::Every { .. })
}

/// Parse a duration with an optional unit suffix; a bare number is **seconds**
/// here (unlike `/screenshot after`, where a bare number means seconds too — the
/// point is that both are explicit about it).
fn parse_secs(s: &str) -> Result<i64, String> {
    let s = s.trim();
    let (num, mult) = if let Some(v) = s.strip_suffix("ms") {
        // Accepted and rounded up so `every 500ms` is refused by MIN_EVERY_SECS
        // with a message about seconds, rather than parsed as 500 seconds.
        (v, 0i64)
    } else if let Some(v) = s.strip_suffix('s') {
        (v, 1)
    } else if let Some(v) = s.strip_suffix('m') {
        (v, 60)
    } else if let Some(v) = s.strip_suffix('h') {
        (v, 3600)
    } else if let Some(v) = s.strip_suffix('d') {
        (v, 86400)
    } else {
        (s, 1)
    };
    let v: i64 = num.parse().map_err(|_| alloc::format!("bad duration '{s}'"))?;
    if mult == 0 {
        return Err(String::from("milliseconds are too fine for a schedule; use seconds"));
    }
    v.checked_mul(mult).ok_or_else(|| String::from("duration overflows"))
}

fn parse_hhmm(s: &str) -> Result<(u8, u8), String> {
    let (h, m) = s
        .trim()
        .split_once(':')
        .ok_or_else(|| alloc::format!("'{s}' is not HH:MM"))?;
    let h: u8 = h.trim().parse().map_err(|_| alloc::format!("bad hour in '{s}'"))?;
    let m: u8 = m.trim().parse().map_err(|_| alloc::format!("bad minute in '{s}'"))?;
    if h > 23 {
        return Err(alloc::format!("hour {h} is out of range (0-23)"));
    }
    if m > 59 {
        return Err(alloc::format!("minute {m} is out of range (0-59)"));
    }
    Ok((h, m))
}

fn parse_dow(s: &str) -> Result<u8, String> {
    let s = s.trim();
    match s {
        "" | "daily" | "everyday" | "every-day" => return Ok(ALL_DAYS),
        "weekdays" | "weekday" => return Ok(WEEKDAYS_MASK),
        "weekends" | "weekend" => return Ok(0b0100_0001),
        _ => {}
    }
    let mut mask = 0u8;
    for part in s.split(',') {
        let p = part.trim().to_ascii_lowercase();
        // Accept the three-letter form and any longer prefix-compatible name, so
        // `monday` works as well as `mon`.
        let idx = DOW_NAMES
            .iter()
            .position(|n| p == *n || (p.len() > 3 && p.starts_with(n)))
            .ok_or_else(|| alloc::format!("unknown day '{part}'"))?;
        mask |= 1 << idx;
    }
    if mask == 0 {
        // An empty day set can never fire. Refusing beats accepting a schedule
        // that silently never runs.
        return Err(String::from("no days selected"));
    }
    Ok(mask)
}

fn render_dow(mask: u8) -> String {
    if mask == ALL_DAYS {
        return String::from("daily");
    }
    if mask == WEEKDAYS_MASK {
        return String::from("weekdays");
    }
    let mut out = String::new();
    for (i, n) in DOW_NAMES.iter().enumerate() {
        if mask & (1 << i) != 0 {
            if !out.is_empty() {
                out.push(',');
            }
            out.push_str(n);
        }
    }
    out
}

/// Parse a recurrence. `now_unix` anchors the relative forms (`in 5m`), which is
/// why it is a parameter rather than read from the clock.
pub fn parse(s: &str, now_unix: i64) -> Result<Recurrence, String> {
    let s = s.trim();
    let mut it = s.split_whitespace();
    let head = it.next().unwrap_or("");
    match head {
        "every" => {
            let d = it.next().ok_or_else(|| String::from("every needs an interval"))?;
            let secs = parse_secs(d)?;
            if secs < MIN_EVERY_SECS {
                return Err(alloc::format!(
                    "interval {secs}s is below the {MIN_EVERY_SECS}s minimum"
                ));
            }
            Ok(Recurrence::Every { secs })
        }
        "at" => {
            let t = it.next().ok_or_else(|| String::from("at needs HH:MM"))?;
            let (hour, min) = parse_hhmm(t)?;
            // The remaining tokens are the day set; joined so `mon, tue` works.
            let rest: alloc::vec::Vec<&str> = it.collect();
            let dow_mask = parse_dow(&rest.join(""))?;
            Ok(Recurrence::Daily { hour, min, dow_mask })
        }
        "on" => {
            let d = it.next().ok_or_else(|| String::from("on needs a day of the month"))?;
            let dom: u8 = d.parse().map_err(|_| alloc::format!("bad day '{d}'"))?;
            if !(1..=31).contains(&dom) {
                return Err(alloc::format!("day {dom} is out of range (1-31)"));
            }
            let t = it.next().ok_or_else(|| String::from("on needs HH:MM"))?;
            let (hour, min) = parse_hhmm(t)?;
            Ok(Recurrence::Monthly { dom, hour, min })
        }
        "in" => {
            let d = it.next().ok_or_else(|| String::from("in needs a delay"))?;
            let secs = parse_secs(d)?;
            if secs <= 0 {
                return Err(String::from("'in' needs a positive delay"));
            }
            Ok(Recurrence::Once { at_unix: now_unix + secs })
        }
        "once" | "at_unix" => {
            let v = it.next().ok_or_else(|| String::from("once needs a time"))?;
            let at = parse_instant(v)?;
            Ok(Recurrence::Once { at_unix: at })
        }
        "" => Err(String::from("empty recurrence")),
        other => Err(alloc::format!(
            "unknown recurrence '{other}' — try 'every 5m', 'at 09:00 weekdays', 'on 1 03:00', 'in 30s'"
        )),
    }
}

/// A bare unix second, or `YYYY-MM-DDTHH:MM[:SS]` read as **UTC**.
fn parse_instant(v: &str) -> Result<i64, String> {
    let v = v.trim();
    if let Ok(n) = v.parse::<i64>() {
        return Ok(n);
    }
    let (date, time) = v
        .split_once(['T', ' '])
        .ok_or_else(|| alloc::format!("'{v}' is not a unix second or YYYY-MM-DDTHH:MM"))?;
    let mut dp = date.split('-');
    let y: i64 = dp.next().unwrap_or("").parse().map_err(|_| String::from("bad year"))?;
    let mo: i64 = dp.next().unwrap_or("").parse().map_err(|_| String::from("bad month"))?;
    let d: i64 = dp.next().unwrap_or("").parse().map_err(|_| String::from("bad day"))?;
    let mut tp = time.trim_end_matches('Z').split(':');
    let h: i64 = tp.next().unwrap_or("0").parse().map_err(|_| String::from("bad hour"))?;
    let mi: i64 = tp.next().unwrap_or("0").parse().map_err(|_| String::from("bad minute"))?;
    let s: i64 = tp.next().unwrap_or("0").parse().unwrap_or(0);
    if !(1..=12).contains(&mo) || !(1..=31).contains(&d) || h > 23 || mi > 59 {
        return Err(alloc::format!("'{v}' is not a valid instant"));
    }
    Ok(crate::clock::unix_from_civil(y, mo, d, h, mi, s))
}

/// Render a recurrence back to the form [`parse`] accepts. Round-tripping is a
/// unit test, and it is what lets the stored JSON keep the recurrence as **one
/// human-editable string** instead of exploded fields — so adding a recurrence
/// kind does not change the schema.
pub fn render(r: Recurrence) -> String {
    match r {
        Recurrence::Every { secs } => {
            if secs % 86400 == 0 {
                alloc::format!("every {}d", secs / 86400)
            } else if secs % 3600 == 0 {
                alloc::format!("every {}h", secs / 3600)
            } else if secs % 60 == 0 {
                alloc::format!("every {}m", secs / 60)
            } else {
                alloc::format!("every {secs}s")
            }
        }
        Recurrence::Daily { hour, min, dow_mask } => {
            alloc::format!("at {hour:02}:{min:02} {}", render_dow(dow_mask))
        }
        Recurrence::Monthly { dom, hour, min } => {
            alloc::format!("on {dom} {hour:02}:{min:02}")
        }
        Recurrence::Once { at_unix } => alloc::format!("once {at_unix}"),
    }
}

/// Resolve a **naive local** wall-clock instant to UTC.
///
/// Two passes, because the offset depends on the instant you are converting and
/// the instant depends on the offset: guess with the offset at the naive value
/// read as UTC, then recompute the offset *there* and redo once if it moved. One
/// pass gets DST transitions wrong by an hour; a third pass never changes
/// anything, because a zone's offset is piecewise constant with jumps far larger
/// than the correction.
fn local_to_utc(naive_local: i64, tz_offset_at: &dyn Fn(i64) -> i32) -> i64 {
    let guess = naive_local - tz_offset_at(naive_local) as i64;
    let off2 = tz_offset_at(guess) as i64;
    naive_local - off2
}

/// The nominal-slot identity of a fire: packed `(local_year, local_day_of_year,
/// hour, minute)`.
///
/// This exists as its own function because it is what makes the **autumn
/// fall-back** safe. When the clocks go back, a nominal local time occurs twice;
/// both instants map to the same slot key, so the second is suppressed by the
/// duplicate-slot guard rather than firing the job again an hour later.
pub fn slot_of(unix: i64, tz_offset_at: &dyn Fn(i64) -> i32) -> i64 {
    let local = unix + tz_offset_at(unix) as i64;
    let (y, mo, d, h, mi, _, _) = crate::clock::civil_from_unix(local);
    let doy = crate::clock::days_from_civil(y, mo, d) - crate::clock::days_from_civil(y, 1, 1);
    ((y * 400 + doy) * 24 + h) * 60 + mi
}

/// Days in a month (proleptic Gregorian).
fn days_in_month(y: i64, mo: i64) -> i64 {
    crate::clock::days_from_civil(
        if mo == 12 { y + 1 } else { y },
        if mo == 12 { 1 } else { mo + 1 },
        1,
    ) - crate::clock::days_from_civil(y, mo, 1)
}

/// The next instant **strictly after** `from_unix` at which `r` fires, in UTC
/// seconds. `tz_offset_at` is injected so DST is testable and this stays pure.
pub fn next_due(r: Recurrence, from_unix: i64, tz_offset_at: &dyn Fn(i64) -> i32) -> Option<i64> {
    match r {
        Recurrence::Every { secs } => Some(from_unix + secs.max(MIN_EVERY_SECS)),
        Recurrence::Once { at_unix } => {
            if at_unix > from_unix {
                Some(at_unix)
            } else {
                None // already past: a `Once` has no next fire
            }
        }
        Recurrence::Daily { hour, min, dow_mask } => {
            let local_now = from_unix + tz_offset_at(from_unix) as i64;
            let (y0, mo0, d0, ..) = crate::clock::civil_from_unix(local_now);
            let day0 = crate::clock::days_from_civil(y0, mo0, d0);
            for k in 0..=DAILY_LOOKAHEAD_DAYS {
                let (y, mo, d) = crate::clock::civil_from_days(day0 + k);
                let naive =
                    crate::clock::unix_from_civil(y, mo, d, hour as i64, min as i64, 0);
                let mut cand = local_to_utc(naive, tz_offset_at);
                // Spring forward: 02:30 does not exist on the transition day, so
                // `local_to_utc` lands *before* the nominal time. Clamp forward
                // to the first instant that really exists, so the run happens
                // once rather than being silently dropped.
                let back = cand + tz_offset_at(cand) as i64;
                if back < naive {
                    cand += naive - back;
                }
                if cand <= from_unix {
                    continue;
                }
                // The day-of-week test is on the **local** date of the candidate,
                // which after a clamp or a fold may not be the date we started
                // the day from.
                let (cy, cmo, cd, ..) =
                    crate::clock::civil_from_unix(cand + tz_offset_at(cand) as i64);
                let wd = crate::clock::days_from_civil(cy, cmo, cd).rem_euclid(7) + 4;
                let wd = wd.rem_euclid(7);
                if dow_mask & (1 << wd) != 0 {
                    return Some(cand);
                }
            }
            None
        }
        Recurrence::Monthly { dom, hour, min } => {
            let local_now = from_unix + tz_offset_at(from_unix) as i64;
            let (y0, mo0, ..) = crate::clock::civil_from_unix(local_now);
            // Two months of lookahead: this month's occurrence may be past.
            for k in 0..3i64 {
                let mo_abs = mo0 - 1 + k;
                let (y, mo) = (y0 + mo_abs.div_euclid(12), mo_abs.rem_euclid(12) + 1);
                // `on 31` in February means the last day of February, not a
                // month that is skipped.
                let d = (dom as i64).min(days_in_month(y, mo));
                let naive = crate::clock::unix_from_civil(y, mo, d, hour as i64, min as i64, 0);
                let mut cand = local_to_utc(naive, tz_offset_at);
                let back = cand + tz_offset_at(cand) as i64;
                if back < naive {
                    cand += naive - back;
                }
                if cand > from_unix {
                    return Some(cand);
                }
            }
            None
        }
    }
}

/// Whether the wall clock moved independently of the monotonic clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Jump {
    pub jumped: bool,
    /// Signed seconds the wall clock moved beyond what the monotonic clock
    /// accounts for. Negative means the clock went backwards.
    pub drift_secs: i64,
}

/// Compare the wall clock's advance against the monotonic clock's.
///
/// Both readings are taken at the same two moments, so ordinary elapsed time
/// cancels and what is left is a correction: an `/ntp` sync, a `/datetime set`,
/// or a resume from suspend. `tol_secs` absorbs the base-drift the clock's
/// `base_unix + (now_ms - base_ms)` model already has.
pub fn detect_jump(
    prev_unix: i64,
    prev_ms: u64,
    now_unix: i64,
    now_ms: u64,
    tol_secs: i64,
) -> Jump {
    if prev_ms == 0 {
        // First observation: nothing to compare against, and calling that a jump
        // would make every boot look like a clock correction.
        return Jump { jumped: false, drift_secs: 0 };
    }
    let mono = (now_ms.saturating_sub(prev_ms) / 1000) as i64;
    let wall = now_unix - prev_unix;
    let drift = wall - mono;
    Jump { jumped: drift.abs() > tol_secs, drift_secs: drift }
}

/// Whether a job is due, and what its bookkeeping should become.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Due {
    No,
    /// Fire now. `coalesced_missed` is how many nominal fires this one stands
    /// in for — zero when nothing was missed.
    Now { coalesced_missed: u32 },
}

/// Whether a *calendar* job is being held because the clock cannot be trusted.
pub fn held_for_clock(r: Recurrence, clock_trusted: bool) -> bool {
    needs_wall_clock(r) && !clock_trusted
}

/// The entire due / missed-run / hold policy, as one pure decision.
///
/// Returns `(verdict, new_next_due_unix, new_last_slot)`. The caller persists
/// the two new values **before** running (see `schedule::commit_fire`), which is
/// what makes a fire at-most-once: a crash mid-run loses the run rather than
/// repeating it, and a missed maintenance run is cheaper than a duplicated
/// irreversible one.
#[allow(clippy::too_many_arguments)]
pub fn evaluate(
    spec: Recurrence,
    catchup: Catchup,
    enabled: bool,
    next_due_unix: i64,
    last_slot: i64,
    now_unix: i64,
    clock_trusted: bool,
    tz_offset_at: &dyn Fn(i64) -> i32,
) -> (Due, i64, i64) {
    if !enabled {
        return (Due::No, next_due_unix, last_slot);
    }
    if held_for_clock(spec, clock_trusted) {
        // Held, not skipped: `next_due_unix` is left exactly as it was so that
        // the first `/ntp` re-anchors it rather than the job silently losing its
        // schedule.
        return (Due::No, next_due_unix, last_slot);
    }
    // Never fired, or a fresh job: establish a first due time and do not fire.
    if next_due_unix <= 0 {
        let next = next_due(spec, now_unix, tz_offset_at).unwrap_or(0);
        return (Due::No, next, last_slot);
    }
    if now_unix < next_due_unix {
        return (Due::No, next_due_unix, last_slot);
    }

    // Due. Count how many nominal fires are in the past, so the decision can be
    // reported rather than merely taken.
    let mut missed: u32 = 0;
    let mut cursor = next_due_unix;
    // Bounded: a machine off for a year on a 5-second interval would otherwise
    // walk six million steps here. The count saturates and says so.
    const MAX_WALK: u32 = 512;
    while missed < MAX_WALK {
        match next_due(spec, cursor, tz_offset_at) {
            Some(n) if n <= now_unix => {
                cursor = n;
                missed += 1;
            }
            _ => break,
        }
    }

    let slot = slot_of(next_due_unix, tz_offset_at);
    // The duplicate-slot guard: the same nominal minute must not fire twice,
    // which is what makes the autumn double-hour safe. Survives a reboot because
    // `last_slot` is persisted.
    let duplicate = slot == last_slot;

    // Advance past everything already in the past, so the job's next fire is in
    // the future whichever branch is taken.
    let advanced = next_due(spec, now_unix.max(cursor), tz_offset_at).unwrap_or(0);

    if duplicate {
        return (Due::No, advanced, last_slot);
    }
    match catchup {
        Catchup::Skip if missed > 0 => (Due::No, advanced, slot),
        Catchup::Skip => (Due::Now { coalesced_missed: 0 }, advanced, slot),
        Catchup::Once => (Due::Now { coalesced_missed: missed }, advanced, slot),
    }
}

/// Who authored a stored intent. Recorded at creation, never recomputed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Author {
    /// A human typed `/schedule add`.
    Human,
    /// The kernel or a system agent installed it.
    System,
    /// A model called the `schedule` tool.
    Agent,
}

impl Author {
    pub fn as_str(self) -> &'static str {
        match self {
            Author::Human => "human",
            Author::System => "system",
            Author::Agent => "agent",
        }
    }
    pub fn parse(s: &str) -> Option<Author> {
        match s.trim() {
            "human" => Some(Author::Human),
            "system" => Some(Author::System),
            "agent" => Some(Author::Agent),
            _ => None,
        }
    }
}

/// The authority facts of a schedule, separated from the rest of the record so
/// the re-authorisation rules can be tested without a session.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GrantFacts {
    pub author: Author,
    pub provenance: crate::security::taint::Provenance,
    pub human_confirmed: bool,
}

/// What an edit changed. Determines whether a stored confirmation survives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditKind {
    /// The action, the identity it runs as, or the toolset. **Re-authorises.**
    Action,
    /// The recurrence, the catch-up policy, or the notification policy. Keeps
    /// the confirmation, because *what* it does is unchanged.
    Schedule,
    /// Enable/disable. Keeps the confirmation.
    Enablement,
}

/// Recompute a grant after an edit.
///
/// The load-bearing rule: **a stored human confirmation does not survive a
/// change of action.** Otherwise a prompt injection turns a human-blessed
/// nightly `/disks` into a nightly `rm -r`, using the human's own approval, and
/// nothing ever asks again. Changing the *time* of an approved action is not
/// that, so the confirmation survives a `Schedule` edit.
///
/// Provenance only ever gets worse ([`crate::security::taint::Provenance::join`]),
/// which is invariant 5 read at this layer: a schedule is bounded by what its
/// author could do when it was stored, forever, and editing it cannot launder
/// the taint of the context doing the editing.
pub fn reauthorise(
    old: GrantFacts,
    edit: EditKind,
    editor: Author,
    editor_taint: crate::security::taint::Provenance,
) -> GrantFacts {
    let provenance = old.provenance.join(editor_taint);
    match edit {
        EditKind::Action => GrantFacts {
            // The action is new, so the author of record is whoever changed it.
            author: editor,
            provenance,
            human_confirmed: false,
        },
        EditKind::Schedule | EditKind::Enablement => {
            GrantFacts { author: old.author, provenance, human_confirmed: old.human_confirmed }
        }
    }
}

/// The [`crate::security::taint::Justification`] a scheduled run acts under.
///
/// Built from the grant recorded at creation, never from whatever happens to be
/// resident when the job fires — that is the difference between a stored intent
/// and a live one, and it is why a schedule cannot gain authority by running.
pub fn grant_justification(g: GrantFacts) -> crate::security::taint::Justification {
    let mut j = crate::security::taint::Justification::from_context(g.provenance);
    if g.human_confirmed {
        j = j.confirmed();
    }
    j
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::security::taint::Provenance;

    /// UTC — the simple case, and the baseline every other test differs from.
    fn utc(_: i64) -> i32 {
        0
    }

    /// A fixed +05:30 zone (IST): no DST, but a non-hour offset, which is where
    /// an implementation that divides by 3600 goes wrong.
    fn ist(_: i64) -> i32 {
        19_800
    }

    /// US Eastern for 2026: EST (-5) until 02:00 local on the second Sunday of
    /// March, EDT (-4) until 02:00 local on the first Sunday of November.
    ///
    /// An independent oracle for the DST tests — deliberately not `clock::tz`,
    /// so a bug there cannot make these tests agree with it. The two instants are
    /// **computed** rather than written as epoch constants: the first version of
    /// this fixture had both hand-written and both wrong by about three days,
    /// which quietly moved the transitions past the dates the tests probe and
    /// made a correct implementation look broken.
    fn eastern_2026(unix: i64) -> i32 {
        // 2026-03-08 02:00 EST = 07:00 UTC; 2026-11-01 02:00 EDT = 06:00 UTC.
        let spring = crate::clock::unix_from_civil(2026, 3, 8, 7, 0, 0);
        let fall = crate::clock::unix_from_civil(2026, 11, 1, 6, 0, 0);
        if unix >= spring && unix < fall {
            -4 * 3600
        } else {
            -5 * 3600
        }
    }

    /// The DST fixture's dates must really be the transition Sundays, or the two
    /// tests below probe an ordinary day and pass for the wrong reason.
    #[test_case]
    fn the_dst_fixture_dates_are_the_real_transition_sundays() {
        // Second Sunday of March 2026.
        let mar8 = crate::clock::unix_from_civil(2026, 3, 8, 0, 0, 0);
        assert_eq!(crate::clock::civil_from_unix(mar8).6, 0, "2026-03-08 must be a Sunday");
        let mar1 = crate::clock::unix_from_civil(2026, 3, 1, 0, 0, 0);
        assert_eq!(crate::clock::civil_from_unix(mar1).6, 0, "…and 03-01 the first Sunday");
        // First Sunday of November 2026.
        let nov1 = crate::clock::unix_from_civil(2026, 11, 1, 0, 0, 0);
        assert_eq!(crate::clock::civil_from_unix(nov1).6, 0, "2026-11-01 must be a Sunday");
        // And the offsets really do change across them.
        assert_eq!(eastern_2026(mar8 + 6 * 3600), -5 * 3600, "06:00 UTC is still EST");
        assert_eq!(eastern_2026(mar8 + 7 * 3600), -4 * 3600, "07:00 UTC is EDT");
        assert_eq!(eastern_2026(nov1 + 5 * 3600), -4 * 3600, "05:00 UTC is still EDT");
        assert_eq!(eastern_2026(nov1 + 6 * 3600), -5 * 3600, "06:00 UTC is EST");
    }

    const T2026: i64 = 1_767_225_600; // 2026-01-01 00:00:00 UTC (a Thursday)

    // --- parse / render ---------------------------------------------------

    #[test_case]
    fn every_form_round_trips_through_parse_and_render() {
        for s in [
            "every 5s",
            "every 30s",
            "every 5m",
            "every 2h",
            "every 1d",
            "at 09:00 daily",
            "at 00:05 daily",
            "at 23:59 weekdays",
            "at 07:30 mon,wed,fri",
            "on 1 03:00",
            "on 31 23:00",
        ] {
            let r = parse(s, T2026).unwrap_or_else(|e| panic!("parse '{s}': {e}"));
            let back = render(r);
            assert_eq!(back, s, "render did not round-trip '{s}'");
            assert_eq!(parse(&back, T2026).unwrap(), r, "re-parse differed for '{s}'");
        }
        // `in`/`once` render canonically as an absolute instant, which is the
        // point — a relative form is resolved once, at creation.
        let r = parse("in 30s", T2026).unwrap();
        assert_eq!(r, Recurrence::Once { at_unix: T2026 + 30 });
        assert_eq!(parse(&render(r), T2026).unwrap(), r);
    }

    #[test_case]
    fn iso_and_unix_instants_both_parse() {
        let a = parse("once 1767225600", 0).unwrap();
        let b = parse("once 2026-01-01T00:00:00", 0).unwrap();
        let c = parse("once 2026-01-01T00:00Z", 0).unwrap();
        assert_eq!(a, Recurrence::Once { at_unix: T2026 });
        assert_eq!(b, a);
        assert_eq!(c, a);
    }

    #[test_case]
    fn day_names_accept_long_and_short_forms_and_the_named_sets() {
        let long = parse("at 09:00 monday,wednesday", T2026).unwrap();
        let short = parse("at 09:00 mon,wed", T2026).unwrap();
        assert_eq!(long, short);
        assert_eq!(
            parse("at 09:00 weekdays", T2026).unwrap(),
            Recurrence::Daily { hour: 9, min: 0, dow_mask: WEEKDAYS_MASK }
        );
        assert_eq!(
            parse("at 09:00", T2026).unwrap(),
            Recurrence::Daily { hour: 9, min: 0, dow_mask: ALL_DAYS },
            "no day set means every day"
        );
        // Bit 0 is Sunday, matching `civil_from_unix`'s weekday.
        assert_eq!(
            parse("at 09:00 sun", T2026).unwrap(),
            Recurrence::Daily { hour: 9, min: 0, dow_mask: 1 }
        );
    }

    #[test_case]
    fn malformed_recurrences_are_refused_with_a_reason() {
        for bad in [
            "",
            "hourly",
            "every",
            "every 0s",
            "every 1s",      // below MIN_EVERY_SECS
            "every 500ms",   // too fine
            "every soon",
            "at",
            "at 25:00",
            "at 09:60",
            "at 0900",
            "at 09:00 blursday",
            "on",
            "on 0 09:00",
            "on 32 09:00",
            "on 5",
            "in",
            "in 0s",
            "once",
            "once nonsense",
        ] {
            let r = parse(bad, T2026);
            assert!(r.is_err(), "'{bad}' should be refused, got {r:?}");
            assert!(!r.unwrap_err().is_empty(), "'{bad}' refused with an empty reason");
        }
    }

    // --- next_due ---------------------------------------------------------

    #[test_case]
    fn every_is_exactly_the_interval_and_ignores_the_zone() {
        let r = Recurrence::Every { secs: 300 };
        assert_eq!(next_due(r, T2026, &utc), Some(T2026 + 300));
        assert_eq!(next_due(r, T2026, &eastern_2026), Some(T2026 + 300));
    }

    #[test_case]
    fn a_once_in_the_past_has_no_next_fire() {
        let r = Recurrence::Once { at_unix: T2026 };
        assert_eq!(next_due(r, T2026 - 1, &utc), Some(T2026));
        assert_eq!(next_due(r, T2026, &utc), None, "not strictly after");
        assert_eq!(next_due(r, T2026 + 1, &utc), None);
    }

    #[test_case]
    fn daily_next_due_is_strictly_after_and_lands_on_the_nominal_minute() {
        let r = Recurrence::Daily { hour: 9, min: 0, dow_mask: ALL_DAYS };
        // From midnight, today's 09:00.
        let d = next_due(r, T2026, &utc).unwrap();
        assert_eq!(d, T2026 + 9 * 3600);
        // From exactly 09:00, tomorrow's — strictly after, so a job cannot
        // re-fire the instant it finishes.
        assert_eq!(next_due(r, d, &utc).unwrap(), d + 86400);
    }

    #[test_case]
    fn daily_next_due_crosses_midnight_and_the_year_boundary() {
        let r = Recurrence::Daily { hour: 0, min: 5, dow_mask: ALL_DAYS };
        // 2026-12-31 23:59 UTC → 2027-01-01 00:05 UTC.
        let from = crate::clock::unix_from_civil(2026, 12, 31, 23, 59, 0);
        let want = crate::clock::unix_from_civil(2027, 1, 1, 0, 5, 0);
        assert_eq!(next_due(r, from, &utc), Some(want));
    }

    #[test_case]
    fn daily_next_due_honours_the_dow_mask() {
        // Monday only, from a Friday.
        let r = Recurrence::Daily { hour: 9, min: 0, dow_mask: 1 << 1 };
        let friday = crate::clock::unix_from_civil(2026, 8, 7, 12, 0, 0);
        assert_eq!(
            crate::clock::civil_from_unix(friday).6,
            5,
            "fixture must really be a Friday"
        );
        let d = next_due(r, friday, &utc).unwrap();
        let (y, mo, dd, h, mi, _, wd) = crate::clock::civil_from_unix(d);
        assert_eq!(wd, 1, "must land on a Monday");
        assert_eq!((y, mo, dd, h, mi), (2026, 8, 10, 9, 0));
    }

    #[test_case]
    fn a_non_hour_offset_zone_is_handled() {
        // 09:00 IST is 03:30 UTC. An implementation that works in whole hours
        // gets this half an hour wrong and every fire is off by 30 minutes.
        let r = Recurrence::Daily { hour: 9, min: 0, dow_mask: ALL_DAYS };
        let d = next_due(r, T2026, &ist).unwrap();
        let (_, _, _, h, mi, ..) = crate::clock::civil_from_unix(d);
        assert_eq!((h, mi), (3, 30));
    }

    #[test_case]
    fn spring_forward_clamps_to_the_first_instant_that_exists() {
        // 2026-03-08, America/New_York: 02:00 local jumps to 03:00, so 02:30
        // never happens. The run must still happen, once, at the first real
        // instant — not be silently dropped for a year.
        let r = Recurrence::Daily { hour: 2, min: 30, dow_mask: ALL_DAYS };
        let from = crate::clock::unix_from_civil(2026, 3, 8, 0, 0, 0); // 00:00 UTC = 19:00 EST Mar 7
        let d = next_due(r, from, &eastern_2026).unwrap();
        let local = d + eastern_2026(d) as i64;
        let (y, mo, dd, h, mi, ..) = crate::clock::civil_from_unix(local);
        assert_eq!((y, mo, dd), (2026, 3, 8));
        assert_eq!((h, mi), (3, 30), "02:30 does not exist; must clamp forward to 03:30");
    }

    #[test_case]
    fn fall_back_maps_both_candidates_to_one_slot_so_it_fires_once() {
        // 2026-11-01, America/New_York: 01:30 local happens twice (EDT then
        // EST). `next_due` returns the earlier; the later must share its slot
        // key so `evaluate`'s duplicate guard suppresses it.
        let r = Recurrence::Daily { hour: 1, min: 30, dow_mask: ALL_DAYS };
        let from = crate::clock::unix_from_civil(2026, 11, 1, 0, 0, 0); // 00:00 UTC = 20:00 EDT Oct 31
        let first = next_due(r, from, &eastern_2026).unwrap();
        // The second 01:30 is one hour later in UTC.
        let second = first + 3600;
        assert_eq!(
            slot_of(first, &eastern_2026),
            slot_of(second, &eastern_2026),
            "the repeated hour must share a nominal slot"
        );
    }

    #[test_case]
    fn slots_are_distinct_across_minutes_days_and_years() {
        let a = crate::clock::unix_from_civil(2026, 8, 7, 9, 0, 0);
        assert_ne!(slot_of(a, &utc), slot_of(a + 60, &utc), "minute");
        assert_ne!(slot_of(a, &utc), slot_of(a + 3600, &utc), "hour");
        assert_ne!(slot_of(a, &utc), slot_of(a + 86400, &utc), "day");
        let b = crate::clock::unix_from_civil(2027, 8, 7, 9, 0, 0);
        assert_ne!(slot_of(a, &utc), slot_of(b, &utc), "year");
        // Same instant, same slot.
        assert_eq!(slot_of(a, &utc), slot_of(a, &utc));
    }

    #[test_case]
    fn monthly_day_31_clamps_to_the_month_length() {
        let r = Recurrence::Monthly { dom: 31, hour: 12, min: 0 };
        // February 2026 has 28 days.
        let from = crate::clock::unix_from_civil(2026, 2, 1, 0, 0, 0);
        let d = next_due(r, from, &utc).unwrap();
        let (y, mo, dd, ..) = crate::clock::civil_from_unix(d);
        assert_eq!((y, mo, dd), (2026, 2, 28));
        // February 2028 is a leap year.
        let from = crate::clock::unix_from_civil(2028, 2, 1, 0, 0, 0);
        let d = next_due(r, from, &utc).unwrap();
        let (y, mo, dd, ..) = crate::clock::civil_from_unix(d);
        assert_eq!((y, mo, dd), (2028, 2, 29));
    }

    #[test_case]
    fn monthly_rolls_into_the_next_month_and_year() {
        let r = Recurrence::Monthly { dom: 1, hour: 3, min: 0 };
        // Just after this month's fire → next month's.
        let from = crate::clock::unix_from_civil(2026, 8, 1, 3, 0, 1);
        let d = next_due(r, from, &utc).unwrap();
        let (y, mo, dd, h, ..) = crate::clock::civil_from_unix(d);
        assert_eq!((y, mo, dd, h), (2026, 9, 1, 3));
        // December rolls the year.
        let from = crate::clock::unix_from_civil(2026, 12, 1, 3, 0, 1);
        let d = next_due(r, from, &utc).unwrap();
        let (y, mo, ..) = crate::clock::civil_from_unix(d);
        assert_eq!((y, mo), (2027, 1));
    }

    #[test_case]
    fn days_in_month_is_right_including_leap_years() {
        assert_eq!(days_in_month(2026, 1), 31);
        assert_eq!(days_in_month(2026, 2), 28);
        assert_eq!(days_in_month(2028, 2), 29);
        assert_eq!(days_in_month(2000, 2), 29, "a 400-year leap year");
        assert_eq!(days_in_month(1900, 2), 28, "a 100-year non-leap year");
        assert_eq!(days_in_month(2026, 4), 30);
        assert_eq!(days_in_month(2026, 12), 31);
    }

    // --- evaluate ---------------------------------------------------------

    fn ev(
        spec: Recurrence,
        catchup: Catchup,
        next_due_unix: i64,
        last_slot: i64,
        now: i64,
    ) -> (Due, i64, i64) {
        evaluate(spec, catchup, true, next_due_unix, last_slot, now, true, &utc)
    }

    #[test_case]
    fn a_job_not_yet_due_does_not_fire() {
        let r = Recurrence::Every { secs: 60 };
        let (d, next, _) = ev(r, Catchup::Skip, T2026 + 60, 0, T2026);
        assert_eq!(d, Due::No);
        assert_eq!(next, T2026 + 60, "the due time is left alone");
    }

    #[test_case]
    fn a_job_exactly_due_fires_and_its_next_is_in_the_future() {
        let r = Recurrence::Every { secs: 60 };
        let (d, next, slot) = ev(r, Catchup::Skip, T2026, 0, T2026);
        assert_eq!(d, Due::Now { coalesced_missed: 0 });
        assert!(next > T2026, "the next fire must be in the future, got {next}");
        assert_ne!(slot, 0, "the slot is recorded so the same minute cannot repeat");
    }

    #[test_case]
    fn a_disabled_job_never_fires_however_overdue() {
        let r = Recurrence::Every { secs: 60 };
        let (d, next, slot) =
            evaluate(r, Catchup::Once, false, T2026, 0, T2026 + 999_999, true, &utc);
        assert_eq!(d, Due::No);
        assert_eq!((next, slot), (T2026, 0), "a paused job's bookkeeping is untouched");
    }

    #[test_case]
    fn missed_runs_are_skipped_by_default_and_the_next_fire_is_ahead() {
        let r = Recurrence::Daily { hour: 3, min: 0, dow_mask: ALL_DAYS };
        let due = crate::clock::unix_from_civil(2026, 7, 1, 3, 0, 0);
        let now = crate::clock::unix_from_civil(2026, 8, 10, 12, 0, 0); // 40 days later
        let (d, next, _) = ev(r, Catchup::Skip, due, 0, now);
        assert_eq!(d, Due::No, "a month of missed nightly runs must not fire");
        assert!(next > now, "and the job must be scheduled forward, not stuck");
        let (_, _, _, h, mi, ..) = crate::clock::civil_from_unix(next);
        assert_eq!((h, mi), (3, 0), "…on its nominal minute");
    }

    #[test_case]
    fn catchup_once_runs_exactly_one_time_and_reports_what_it_stood_in_for() {
        let r = Recurrence::Daily { hour: 3, min: 0, dow_mask: ALL_DAYS };
        let due = crate::clock::unix_from_civil(2026, 7, 1, 3, 0, 0);
        let now = crate::clock::unix_from_civil(2026, 8, 10, 12, 0, 0);
        let (d, next, slot) = ev(r, Catchup::Once, due, 0, now);
        match d {
            Due::Now { coalesced_missed } => {
                assert!(coalesced_missed >= 38, "expected ~40 missed, got {coalesced_missed}");
            }
            Due::No => panic!("catchup once must fire when overdue"),
        }
        // A second evaluation at the same instant, with the bookkeeping the first
        // one produced, must not fire again.
        let (d2, ..) = ev(r, Catchup::Once, next, slot, now);
        assert_eq!(d2, Due::No, "the same overdue window must not fire twice");
    }

    #[test_case]
    fn the_missed_walk_is_bounded_rather_than_walking_a_year_of_seconds() {
        // Five-second interval, a year of downtime: an unbounded count would be
        // six million iterations inside a poll tick.
        let r = Recurrence::Every { secs: 5 };
        let (d, next, _) = ev(r, Catchup::Once, T2026, 0, T2026 + 365 * 86400);
        match d {
            Due::Now { coalesced_missed } => assert_eq!(coalesced_missed, 512, "saturates"),
            Due::No => panic!("should fire"),
        }
        assert!(next > T2026 + 365 * 86400);
    }

    #[test_case]
    fn the_same_slot_cannot_fire_twice_which_is_what_makes_fall_back_safe() {
        let r = Recurrence::Daily { hour: 1, min: 30, dow_mask: ALL_DAYS };
        let from = crate::clock::unix_from_civil(2026, 11, 1, 0, 0, 0);
        let first = next_due(r, from, &eastern_2026).unwrap();
        let slot = slot_of(first, &eastern_2026);
        // Fire the first 01:30.
        let (d, next, s) =
            evaluate(r, Catchup::Skip, true, first, 0, first, true, &eastern_2026);
        assert_eq!(d, Due::Now { coalesced_missed: 0 });
        assert_eq!(s, slot);
        // The repeated hour: same nominal slot, so it is suppressed.
        let second = first + 3600;
        let (d2, ..) = evaluate(r, Catchup::Skip, true, second, s, second, true, &eastern_2026);
        assert_eq!(d2, Due::No, "the second 01:30 must not re-run the job");
        let _ = next;
    }

    #[test_case]
    fn calendar_jobs_are_held_while_the_clock_is_untrusted_but_interval_jobs_run() {
        let daily = Recurrence::Daily { hour: 3, min: 0, dow_mask: ALL_DAYS };
        let every = Recurrence::Every { secs: 60 };
        assert!(held_for_clock(daily, false));
        assert!(!held_for_clock(daily, true));
        assert!(!held_for_clock(every, false), "an interval is wall-clock independent");

        // Held: due time untouched, so `/ntp` can re-anchor it.
        let (d, next, slot) =
            evaluate(daily, Catchup::Once, true, T2026, 7, T2026 + 10_000, false, &utc);
        assert_eq!(d, Due::No);
        assert_eq!((next, slot), (T2026, 7), "a held job keeps its bookkeeping verbatim");

        // The interval job fires on the same untrusted clock.
        let (d, ..) = evaluate(every, Catchup::Skip, true, T2026, 0, T2026, false, &utc);
        assert_eq!(d, Due::Now { coalesced_missed: 0 });
    }

    #[test_case]
    fn a_never_run_job_gets_a_first_due_time_without_firing() {
        let r = Recurrence::Daily { hour: 3, min: 0, dow_mask: ALL_DAYS };
        let (d, next, _) = ev(r, Catchup::Skip, 0, 0, T2026);
        assert_eq!(d, Due::No, "adding a job must not fire it immediately");
        assert!(next > T2026);
    }

    // --- detect_jump ------------------------------------------------------

    #[test_case]
    fn ordinary_drift_is_not_a_jump_but_an_ntp_correction_is() {
        // Ten minutes of monotonic time, ten minutes of wall time: no jump.
        let j = detect_jump(T2026, 1_000, T2026 + 600, 601_000, JUMP_TOLERANCE_SECS);
        assert!(!j.jumped);
        assert_eq!(j.drift_secs, 0);
        // A few seconds of base drift: still not a jump.
        let j = detect_jump(T2026, 1_000, T2026 + 603, 601_000, JUMP_TOLERANCE_SECS);
        assert!(!j.jumped, "3s of drift is not a correction");
        // The fallback clock corrected to real time, years later: a jump.
        let j = detect_jump(T2026, 1_000, T2026 + 200_000_000, 601_000, JUMP_TOLERANCE_SECS);
        assert!(j.jumped && j.drift_secs > 0);
        // And backwards.
        let j = detect_jump(T2026, 1_000, T2026 - 5_000, 601_000, JUMP_TOLERANCE_SECS);
        assert!(j.jumped && j.drift_secs < 0);
    }

    #[test_case]
    fn the_first_observation_is_never_a_jump() {
        // `prev_ms == 0` means "nothing recorded yet"; treating it as a jump
        // would make every boot re-anchor every calendar job.
        let j = detect_jump(0, 0, T2026, 5_000, JUMP_TOLERANCE_SECS);
        assert!(!j.jumped);
    }

    // --- authority --------------------------------------------------------

    fn facts(a: Author, p: Provenance, c: bool) -> GrantFacts {
        GrantFacts { author: a, provenance: p, human_confirmed: c }
    }

    #[test_case]
    fn changing_the_action_clears_a_stored_human_confirmation() {
        let old = facts(Author::Human, Provenance::UserTyped, true);
        let new = reauthorise(old, EditKind::Action, Author::Agent, Provenance::UntrustedIngested);
        assert!(
            !new.human_confirmed,
            "an injection must not inherit a human's approval of a different action"
        );
        assert_eq!(new.author, Author::Agent, "the author of record is whoever changed it");
        assert_eq!(new.provenance, Provenance::UntrustedIngested);
    }

    #[test_case]
    fn changing_only_the_time_keeps_the_confirmation() {
        let old = facts(Author::Human, Provenance::UserTyped, true);
        for kind in [EditKind::Schedule, EditKind::Enablement] {
            let new = reauthorise(old, kind, Author::Human, Provenance::UserTyped);
            assert!(new.human_confirmed, "rescheduling an approved action is not a new action");
            assert_eq!(new.author, Author::Human);
        }
    }

    #[test_case]
    fn provenance_only_ever_gets_worse() {
        let clean = facts(Author::Human, Provenance::UserTyped, true);
        // A tainted editor taints the record…
        let dirty = reauthorise(clean, EditKind::Schedule, Author::Agent, Provenance::UntrustedIngested);
        assert_eq!(dirty.provenance, Provenance::UntrustedIngested);
        // …and a subsequent clean edit cannot wash it out again.
        let back = reauthorise(dirty, EditKind::Schedule, Author::Human, Provenance::UserTyped);
        assert_eq!(
            back.provenance,
            Provenance::UntrustedIngested,
            "editing must not launder a tainted schedule"
        );
    }

    #[test_case]
    fn a_clean_authored_schedule_can_act_and_a_tainted_one_cannot() {
        // Typed by a human: the same authority the interactive path already has.
        let j = grant_justification(facts(Author::Human, Provenance::UserTyped, true));
        assert!(!j.blocks_destructive());
        // Authored by an agent in a clean context: still acts.
        let j = grant_justification(facts(Author::Agent, Provenance::UserTyped, false));
        assert!(!j.blocks_destructive());
        // Authored while untrusted content was resident: blocked for the life of
        // the job. Its inert calls still work, which is most of a daily digest.
        let j = grant_justification(facts(Author::Agent, Provenance::UntrustedIngested, false));
        assert!(
            j.blocks_destructive(),
            "a schedule authored under injection must not act destructively, ever"
        );
    }

    #[test_case]
    fn author_and_policy_enums_round_trip_their_strings() {
        for a in [Author::Human, Author::System, Author::Agent] {
            assert_eq!(Author::parse(a.as_str()), Some(a));
        }
        for c in [Catchup::Skip, Catchup::Once] {
            assert_eq!(Catchup::parse(c.as_str()), Some(c));
        }
        for n in [NotifyOn::Always, NotifyOn::OnChange, NotifyOn::OnError, NotifyOn::Never] {
            assert_eq!(NotifyOn::parse(n.as_str()), Some(n));
        }
        assert_eq!(Author::parse("root"), None);
        assert_eq!(Catchup::parse("all"), None, "there is deliberately no catch-up-all");
        assert_eq!(NotifyOn::parse("sometimes"), None);
    }
}
