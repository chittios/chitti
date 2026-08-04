//! Wall clock: a real date/time on top of the monotonic boot timer
//! (`arch::now_ms`). The kernel has no continuously-running dated clock, so this
//! keeps a **base** — a Unix timestamp captured at a known `now_ms()` — and
//! derives the current time by adding the elapsed monotonic milliseconds. The
//! base is seeded from a hardware RTC at boot (`arch::rtc_unix`: CMOS on x86,
//! PL031 on aarch64), refined by SNTP (`/ntp`), or overridden with `/datetime`.
//!
//! Timezone: either a fixed offset (`tz_offset`) or a named IANA zone with DST
//! rules ([`tz`]). Display helpers use [`offset_at`] so DST is correct.
//!
//! All calendar math is the proleptic-Gregorian civil-date algorithm (Howard
//! Hinnant's `days_from_civil` / `civil_from_days`), pure integer arithmetic so
//! it is `no_std` and deterministic.

pub mod face;
pub mod tz;

use crate::mm::Locked;
use alloc::format;
use alloc::string::{String, ToString};

/// Fallback base if no RTC is readable: 2026-01-01 00:00:00 UTC. `/datetime`
/// (and, on most platforms, the RTC) replace it with the real time.
const DEFAULT_UNIX: i64 = 1_767_225_600;

/// How the wall-clock base was last set — TLS / diagnostics care about this.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClockSource {
    /// Baked fallback (no RTC).
    Fallback,
    /// Hardware RTC at boot.
    Rtc,
    /// SNTP / `/ntp`.
    Ntp,
    /// Human `/datetime` set.
    Manual,
}

impl ClockSource {
    pub fn as_str(self) -> &'static str {
        match self {
            ClockSource::Fallback => "fallback",
            ClockSource::Rtc => "rtc",
            ClockSource::Ntp => "ntp",
            ClockSource::Manual => "manual",
        }
    }
    /// RTC / NTP / manual are "trusted" for SNTP plausibility windows.
    pub fn trusted(self) -> bool {
        !matches!(self, ClockSource::Fallback)
    }
}

struct Clock {
    /// Unix seconds at the moment `base_ms` was recorded.
    base_unix: i64,
    /// `arch::now_ms()` reading when `base_unix` was set.
    base_ms: u64,
    /// Local timezone offset from UTC, in seconds (e.g. +5:30 IST = 19800).
    /// Used when `tz_name` is empty; otherwise [`tz::offset_at`] wins.
    tz_offset: i32,
    /// IANA zone name (e.g. `America/New_York`); empty = fixed offset only.
    tz_name: String,
    source: ClockSource,
    initialized: bool,
}

static CLOCK: Locked<Clock> = Locked::new(Clock {
    base_unix: DEFAULT_UNIX,
    base_ms: 0,
    tz_offset: 0,
    tz_name: String::new(),
    source: ClockSource::Fallback,
    initialized: false,
});

/// Seed the wall clock from the hardware RTC (or the fallback). Idempotent; the
/// first call wins unless `/datetime` / `/ntp` later overrides it.
pub fn init() {
    let now = crate::arch::now_ms();
    let (unix, src) = match crate::arch::rtc_unix() {
        Some(s) => (s as i64, ClockSource::Rtc),
        None => (DEFAULT_UNIX, ClockSource::Fallback),
    };
    CLOCK.with(|c| {
        if !c.initialized {
            c.base_unix = unix;
            c.base_ms = now;
            c.source = src;
            c.initialized = true;
        }
    });
}

/// How the clock base was last established.
pub fn source() -> ClockSource {
    CLOCK.with(|c| c.source)
}

/// Current UTC Unix timestamp (seconds).
pub fn now_unix() -> i64 {
    now_unix_ms() / 1000
}

/// Current UTC Unix timestamp in **milliseconds**.
pub fn now_unix_ms() -> i64 {
    CLOCK.with(|c| {
        let elapsed = crate::arch::now_ms().saturating_sub(c.base_ms) as i64;
        c.base_unix * 1000 + elapsed
    })
}

/// The current local `(hour, min, sec, millis)` — the ktrace timestamp.
pub fn local_hms_ms() -> (i64, i64, i64, i64) {
    let ms = now_unix_ms() + tz_offset() as i64 * 1000;
    let (.., h, mi, s, _) = civil_from_unix(ms.div_euclid(1000));
    (h, mi, s, ms.rem_euclid(1000))
}

/// Effective timezone offset from UTC **right now**, in seconds.
/// Named zones use DST rules; otherwise the fixed `tz_offset`.
pub fn tz_offset() -> i32 {
    offset_at(now_unix())
}

/// Offset east of UTC at a given Unix second (named zone or fixed).
pub fn offset_at(unix: i64) -> i32 {
    CLOCK.with(|c| {
        if !c.tz_name.is_empty() {
            if let Some(off) = tz::offset_at(&c.tz_name, unix) {
                return off;
            }
        }
        c.tz_offset
    })
}

/// Fixed offset currently stored (ignoring DST). For persistence / display of
/// the configured baseline when no name is set.
pub fn fixed_tz_offset() -> i32 {
    CLOCK.with(|c| c.tz_offset)
}

/// Configured IANA name, if any.
pub fn tz_name() -> String {
    CLOCK.with(|c| c.tz_name.clone())
}

/// Set a fixed timezone offset (seconds east of UTC). Clears any IANA name.
/// Persisted by the caller into the UI config.
pub fn set_tz(offset_secs: i32) {
    CLOCK.with(|c| {
        c.tz_offset = offset_secs;
        c.tz_name.clear();
    });
}

/// Set timezone by IANA name (`America/New_York`). Returns false if unknown.
/// Keeps displayed local time stable (UTC base shifts by offset delta).
pub fn set_tz_name(name: &str) -> bool {
    if tz::lookup(name).is_none() {
        return false;
    }
    let now = now_unix();
    CLOCK.with(|c| {
        let old = if !c.tz_name.is_empty() {
            tz::offset_at(&c.tz_name, now).unwrap_or(c.tz_offset)
        } else {
            c.tz_offset
        };
        let new = tz::offset_at(name, now).unwrap_or(0);
        c.base_unix += (old - new) as i64;
        c.tz_name = name.to_string();
        c.tz_offset = new; // cache current for config writers
    });
    true
}

/// Change the fixed timezone **without changing the wall time shown** — the
/// `/datetime tz +5:30` semantics. Clears IANA name.
pub fn set_tz_keep_local(offset_secs: i32) {
    CLOCK.with(|c| {
        let old = if !c.tz_name.is_empty() {
            tz::offset_at(&c.tz_name, now_unix_inner(c)).unwrap_or(c.tz_offset)
        } else {
            c.tz_offset
        };
        let delta = old as i64 - offset_secs as i64;
        c.base_unix += delta;
        c.tz_offset = offset_secs;
        c.tz_name.clear();
    });
}

fn now_unix_inner(c: &Clock) -> i64 {
    let elapsed = crate::arch::now_ms().saturating_sub(c.base_ms) as i64;
    c.base_unix + elapsed / 1000
}

/// Set the current UTC time to `unix` seconds (rebasing against `now_ms`).
/// Marks source as [`ClockSource::Manual`].
pub fn set_unix(unix: i64) {
    set_unix_with_source(unix, ClockSource::Manual);
}

/// Set UTC time and record how it was obtained (NTP / RTC / manual).
pub fn set_unix_with_source(unix: i64, src: ClockSource) {
    let now = crate::arch::now_ms();
    CLOCK.with(|c| {
        c.base_unix = unix;
        c.base_ms = now;
        c.source = src;
        c.initialized = true;
    });
}

/// Set the wall clock from local calendar components (interpreted in the current
/// timezone) — used by `/datetime YYYY-MM-DD HH:MM[:SS]`.
pub fn set_local(y: i64, mo: i64, d: i64, h: i64, mi: i64, s: i64) {
    let utc = unix_from_civil(y, mo, d, h, mi, s) - tz_offset() as i64;
    set_unix(utc);
}

/// Days since 1970-01-01 for a proleptic-Gregorian date (Hinnant).
pub fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = y - era * 400; // [0, 399]
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146097 + doe - 719468
}

/// Inverse of [`days_from_civil`]: `(year, month, day)` from days-since-epoch.
pub fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe = z - era * 146097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Unix seconds from UTC calendar components.
pub fn unix_from_civil(y: i64, mo: i64, d: i64, h: i64, mi: i64, s: i64) -> i64 {
    days_from_civil(y, mo, d) * 86400 + h * 3600 + mi * 60 + s
}

/// `(year, month, day, hour, min, sec, weekday)` for a Unix timestamp; weekday
/// 0 = Sunday .. 6 = Saturday.
pub fn civil_from_unix(secs: i64) -> (i64, i64, i64, i64, i64, i64, i64) {
    let days = secs.div_euclid(86400);
    let rem = secs.rem_euclid(86400);
    let (y, mo, d) = civil_from_days(days);
    let weekday = (days.rem_euclid(7) + 4).rem_euclid(7); // 1970-01-01 was Thursday
    (y, mo, d, rem / 3600, rem % 3600 / 60, rem % 60, weekday)
}

const WEEKDAYS: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const MONTHS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/// The current **local** time as `"Wed 2026-07-04 13:45:02"` (full, with year + seconds).
pub fn format_datetime() -> String {
    let (y, mo, d, h, mi, s, wd) = civil_from_unix(now_unix() + tz_offset() as i64);
    format!("{} {:04}-{:02}-{:02} {:02}:{:02}:{:02}", WEEKDAYS[wd as usize], y, mo, d, h, mi, s)
}

/// Compact **menu-bar** local time, macOS-style: `"Tue Aug 4  19:45"`.
///
/// No year, no seconds, no timezone — the status bar is a glance target; full
/// detail lives in the clock dropdown (`/datetime`, status-chip menu).
pub fn format_datetime_short() -> String {
    let (_y, mo, d, h, mi, _s, wd) = civil_from_unix(now_unix() + tz_offset() as i64);
    let mon = MONTHS[(mo as usize).saturating_sub(1).min(11)];
    format!("{} {} {}  {:02}:{:02}", WEEKDAYS[wd as usize], mon, d, h, mi)
}

/// Pure form of [`format_datetime_short`] for a given Unix second + offset (tests).
pub fn format_datetime_short_at(unix: i64, tz_secs: i32) -> String {
    let (_y, mo, d, h, mi, _s, wd) = civil_from_unix(unix + tz_secs as i64);
    let mon = MONTHS[(mo as usize).saturating_sub(1).min(11)];
    format!("{} {} {}  {:02}:{:02}", WEEKDAYS[wd as usize], mon, d, h, mi)
}

/// The current UTC time as an ISO-8601 instant, `"2026-07-13T02:36:19Z"` —
/// for session-transcript records (a stable, parseable, timezone-explicit form).
pub fn now_iso8601() -> String {
    let (y, mo, d, h, mi, s, _) = civil_from_unix(now_unix());
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, mo, d, h, mi, s)
}

/// The current local date, `"2026-07-04"`.
pub fn format_date() -> String {
    let (y, mo, d, ..) = civil_from_unix(now_unix() + tz_offset() as i64);
    format!("{:04}-{:02}-{:02}", y, mo, d)
}

/// The current local time, `"13:45:02"`.
pub fn format_time() -> String {
    let (.., h, mi, s, _) = civil_from_unix(now_unix() + tz_offset() as i64);
    format!("{:02}:{:02}:{:02}", h, mi, s)
}

/// The timezone as `"America/New_York (UTC-04:00)"` or `"UTC+05:30"` / `"UTC"`.
pub fn format_tz() -> String {
    let off = tz_offset();
    let off_s = if off == 0 {
        String::from("UTC")
    } else {
        let sign = if off > 0 { '+' } else { '-' };
        let a = off.unsigned_abs();
        format!("UTC{}{:02}:{:02}", sign, a / 3600, a % 3600 / 60)
    };
    CLOCK.with(|c| {
        if !c.tz_name.is_empty() {
            format!("{} ({})", c.tz_name, off_s)
        } else {
            off_s
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn datetime_short_is_macos_menu_bar_shape() {
        // 2026-08-04 19:45:32 UTC → weekday Tue (check civil math) at offset 0.
        let unix = unix_from_civil(2026, 8, 4, 19, 45, 32);
        let s = format_datetime_short_at(unix, 0);
        // "Tue Aug 4  19:45" — no year, no seconds, double-space before time.
        assert!(s.starts_with("Tue Aug 4"), "got {s}");
        assert!(s.ends_with("19:45"), "got {s}");
        assert!(!s.contains("2026"), "year must not appear in short form: {s}");
        assert!(!s.contains(":32"), "seconds must not appear: {s}");
        // Compact: shorter than the full form.
        let full = {
            let (y, mo, d, h, mi, sec, wd) = civil_from_unix(unix);
            format!(
                "{} {:04}-{:02}-{:02} {:02}:{:02}:{:02}",
                WEEKDAYS[wd as usize], y, mo, d, h, mi, sec
            )
        };
        assert!(s.len() < full.len(), "short={s} full={full}");
    }
}
