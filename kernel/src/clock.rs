//! Wall clock: a real date/time on top of the monotonic boot timer
//! (`arch::now_ms`). The kernel has no continuously-running dated clock, so this
//! keeps a **base** — a Unix timestamp captured at a known `now_ms()` — and
//! derives the current time by adding the elapsed monotonic milliseconds. The
//! base is seeded from a hardware RTC at boot (`arch::rtc_unix`: CMOS on x86,
//! PL031 on aarch64) and can be overridden with the `/datetime` shell command;
//! the timezone offset is persisted in the UI config.
//!
//! All calendar math is the proleptic-Gregorian civil-date algorithm (Howard
//! Hinnant's `days_from_civil` / `civil_from_days`), pure integer arithmetic so
//! it is `no_std` and deterministic.

use crate::mm::Locked;
use alloc::format;
use alloc::string::String;

/// Fallback base if no RTC is readable: 2026-01-01 00:00:00 UTC. `/datetime`
/// (and, on most platforms, the RTC) replace it with the real time.
const DEFAULT_UNIX: i64 = 1_767_225_600;

struct Clock {
    /// Unix seconds at the moment `base_ms` was recorded.
    base_unix: i64,
    /// `arch::now_ms()` reading when `base_unix` was set.
    base_ms: u64,
    /// Local timezone offset from UTC, in seconds (e.g. +5:30 IST = 19800).
    tz_offset: i32,
    initialized: bool,
}

static CLOCK: Locked<Clock> = Locked::new(Clock { base_unix: DEFAULT_UNIX, base_ms: 0, tz_offset: 0, initialized: false });

/// Seed the wall clock from the hardware RTC (or the fallback). Idempotent; the
/// first call wins unless `/datetime` later overrides it.
pub fn init() {
    let now = crate::arch::now_ms();
    let unix = crate::arch::rtc_unix().map(|s| s as i64).unwrap_or(DEFAULT_UNIX);
    CLOCK.with(|c| {
        if !c.initialized {
            c.base_unix = unix;
            c.base_ms = now;
            c.initialized = true;
        }
    });
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

/// The configured timezone offset from UTC, in seconds.
pub fn tz_offset() -> i32 {
    CLOCK.with(|c| c.tz_offset)
}

/// Set the timezone offset (seconds east of UTC). Persisted by the caller into
/// the UI config so it survives a reboot. Used at boot (applying the persisted
/// offset to the RTC-seeded UTC base) — display shifts by the offset.
pub fn set_tz(offset_secs: i32) {
    CLOCK.with(|c| c.tz_offset = offset_secs);
}

/// Change the timezone **without changing the wall time shown** — the
/// `/datetime tz` semantics. The user set the displayed local time earlier
/// (or trusts what the clock shows); relabeling the zone must not jump the
/// clock, so the UTC base shifts by the offset delta instead.
pub fn set_tz_keep_local(offset_secs: i32) {
    CLOCK.with(|c| {
        let delta = c.tz_offset as i64 - offset_secs as i64; // old - new
        c.base_unix += delta;
        c.tz_offset = offset_secs;
    });
}

/// Set the current UTC time to `unix` seconds (rebasing against `now_ms`).
pub fn set_unix(unix: i64) {
    let now = crate::arch::now_ms();
    CLOCK.with(|c| {
        c.base_unix = unix;
        c.base_ms = now;
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

/// The current **local** time as `"Wed 2026-07-04 13:45:02"`.
pub fn format_datetime() -> String {
    let (y, mo, d, h, mi, s, wd) = civil_from_unix(now_unix() + tz_offset() as i64);
    format!("{} {:04}-{:02}-{:02} {:02}:{:02}:{:02}", WEEKDAYS[wd as usize], y, mo, d, h, mi, s)
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

/// The timezone as `"UTC+05:30"` / `"UTC-08:00"` / `"UTC"`.
pub fn format_tz() -> String {
    let off = tz_offset();
    if off == 0 {
        return String::from("UTC");
    }
    let sign = if off > 0 { '+' } else { '-' };
    let a = off.unsigned_abs();
    format!("UTC{}{:02}:{:02}", sign, a / 3600, a % 3600 / 60)
}
