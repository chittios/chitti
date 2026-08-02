//! **IANA timezones with DST** — a curated table of zones (not the full tzdb).
//!
//! Each zone is either a **fixed offset** or a **US/EU-style DST rule**:
//! - Standard offset, DST offset (seconds east of UTC)
//! - Start / end: `(month, week, weekday, hour_utc_approx as local std hour)`
//!   week 1..4 = nth weekday of month, week 5 = last weekday.
//!
//! Offset at time `unix` walks the year containing that instant and finds
//! whether DST is active. Good enough for status bar + `/datetime` on a
//! research OS; not a full historical tzdb.

use alloc::string::String;

/// One curated zone.
#[derive(Clone, Copy, Debug)]
pub struct Zone {
    pub name: &'static str,
    /// Standard (winter) offset east of UTC, seconds.
    pub std: i32,
    /// DST offset east of UTC; equal to `std` if no DST.
    pub dst: i32,
    /// DST start (month 1–12, week 1–5, weekday 0=Sun, hour local standard).
    pub start: Option<(u8, u8, u8, u8)>,
    /// DST end.
    pub end: Option<(u8, u8, u8, u8)>,
}

/// Curated set — keep small; add via a row here.
pub static ZONES: &[Zone] = &[
    Zone {
        name: "UTC",
        std: 0,
        dst: 0,
        start: None,
        end: None,
    },
    Zone {
        name: "Etc/UTC",
        std: 0,
        dst: 0,
        start: None,
        end: None,
    },
    // India — no DST
    Zone {
        name: "Asia/Kolkata",
        std: 19800,
        dst: 19800,
        start: None,
        end: None,
    },
    Zone {
        name: "Asia/Calcutta",
        std: 19800,
        dst: 19800,
        start: None,
        end: None,
    },
    Zone {
        name: "Asia/Tokyo",
        std: 32400,
        dst: 32400,
        start: None,
        end: None,
    },
    Zone {
        name: "Asia/Shanghai",
        std: 28800,
        dst: 28800,
        start: None,
        end: None,
    },
    Zone {
        name: "Asia/Dubai",
        std: 14400,
        dst: 14400,
        start: None,
        end: None,
    },
    Zone {
        name: "Asia/Singapore",
        std: 28800,
        dst: 28800,
        start: None,
        end: None,
    },
    // US — 2nd Sun Mar 02:00 → 1st Sun Nov 02:00 local
    Zone {
        name: "America/New_York",
        std: -18000,
        dst: -14400,
        start: Some((3, 2, 0, 2)),
        end: Some((11, 1, 0, 2)),
    },
    Zone {
        name: "America/Chicago",
        std: -21600,
        dst: -18000,
        start: Some((3, 2, 0, 2)),
        end: Some((11, 1, 0, 2)),
    },
    Zone {
        name: "America/Denver",
        std: -25200,
        dst: -21600,
        start: Some((3, 2, 0, 2)),
        end: Some((11, 1, 0, 2)),
    },
    Zone {
        name: "America/Los_Angeles",
        std: -28800,
        dst: -25200,
        start: Some((3, 2, 0, 2)),
        end: Some((11, 1, 0, 2)),
    },
    // EU — last Sun Mar 01:00 UTC → last Sun Oct 01:00 UTC
    // Modelled as local std hour 2 for CET (UTC+1): last Sun Mar 02:00 → last Sun Oct 03:00
    Zone {
        name: "Europe/Paris",
        std: 3600,
        dst: 7200,
        start: Some((3, 5, 0, 2)),
        end: Some((10, 5, 0, 3)),
    },
    Zone {
        name: "Europe/Berlin",
        std: 3600,
        dst: 7200,
        start: Some((3, 5, 0, 2)),
        end: Some((10, 5, 0, 3)),
    },
    Zone {
        name: "Europe/London",
        std: 0,
        dst: 3600,
        start: Some((3, 5, 0, 1)),
        end: Some((10, 5, 0, 2)),
    },
    // Australia — 1st Sun Oct 02:00 → 1st Sun Apr 03:00 (AEDT)
    Zone {
        name: "Australia/Sydney",
        std: 36000,
        dst: 39600,
        start: Some((10, 1, 0, 2)),
        end: Some((4, 1, 0, 3)),
    },
    // NZ — last Sun Sep 02:00 → 1st Sun Apr 03:00
    Zone {
        name: "Pacific/Auckland",
        std: 43200,
        dst: 46800,
        start: Some((9, 5, 0, 2)),
        end: Some((4, 1, 0, 3)),
    },
];

/// Find a zone by name (case-sensitive IANA form).
pub fn lookup(name: &str) -> Option<&'static Zone> {
    ZONES.iter().find(|z| z.name.eq_ignore_ascii_case(name))
}

/// List zone names for `/datetime tz list`.
pub fn list_names() -> alloc::vec::Vec<&'static str> {
    ZONES.iter().map(|z| z.name).collect()
}

/// Offset east of UTC at `unix` for `name`, if known.
pub fn offset_at(name: &str, unix: i64) -> Option<i32> {
    let z = lookup(name)?;
    Some(zone_offset(z, unix))
}

fn zone_offset(z: &Zone, unix: i64) -> i32 {
    let (Some(start), Some(end)) = (z.start, z.end) else {
        return z.std;
    };
    let (y, ..) = super::civil_from_unix(unix);
    // Southern hemisphere: DST start month > end month (e.g. Oct → Apr).
    let start_u = transition_unix(y, start, z.std);
    let end_u = transition_unix(y, end, z.dst);
    if start_u < end_u {
        // Northern: spring start … fall end
        if unix >= start_u && unix < end_u {
            z.dst
        } else {
            z.std
        }
    } else {
        // Southern: DST spans new year
        if unix >= start_u || unix < end_u {
            z.dst
        } else {
            z.std
        }
    }
}

/// Unix second of the DST transition in year `y`.
/// `rule` = (month, week, weekday, hour_local); hour is in the **outgoing**
/// offset (`off_before`): standard for start, DST for end (US/EU convention).
fn transition_unix(y: i64, rule: (u8, u8, u8, u8), off_before: i32) -> i64 {
    let (month, week, weekday, hour) = rule;
    let day = nth_weekday_of_month(y, month as i64, week, weekday);
    super::unix_from_civil(y, month as i64, day, hour as i64, 0, 0) - off_before as i64
}

/// Day-of-month for the `week`-th `weekday` (0=Sun) of `month` in `year`.
/// `week` 5 means **last** such weekday.
fn nth_weekday_of_month(year: i64, month: i64, week: u8, weekday: u8) -> i64 {
    // Day-of-week for the 1st of the month.
    let first_unix = super::unix_from_civil(year, month, 1, 0, 0, 0);
    let first_wd = ((first_unix.div_euclid(86400)).rem_euclid(7) + 4).rem_euclid(7) as u8;
    // First occurrence of `weekday` on or after the 1st.
    let mut day = 1 + (weekday as i64 - first_wd as i64 + 7) % 7;
    if week >= 5 {
        // Last: keep adding 7 while still in month.
        loop {
            let next = day + 7;
            if next > days_in_month(year, month) {
                break;
            }
            day = next;
        }
    } else {
        day += (week as i64 - 1) * 7;
    }
    day
}

fn days_in_month(y: i64, m: i64) -> i64 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) {
                29
            } else {
                28
            }
        }
        _ => 30,
    }
}

/// Human-readable short line for a zone at `unix`.
pub fn describe(name: &str, unix: i64) -> Option<String> {
    let z = lookup(name)?;
    let off = zone_offset(z, unix);
    let sign = if off >= 0 { '+' } else { '-' };
    let a = off.unsigned_abs();
    Some(alloc::format!(
        "{} UTC{}{:02}:{:02}",
        z.name,
        sign,
        a / 3600,
        (a % 3600) / 60
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn kolkata_fixed() {
        assert_eq!(offset_at("Asia/Kolkata", 1_700_000_000), Some(19800));
    }

    #[test_case]
    fn new_york_dst_summer_winter() {
        // 2024-01-15 12:00 UTC → EST (UTC-5)
        let winter = super::super::unix_from_civil(2024, 1, 15, 12, 0, 0);
        assert_eq!(offset_at("America/New_York", winter), Some(-18000));
        // 2024-07-15 12:00 UTC → EDT (UTC-4)
        let summer = super::super::unix_from_civil(2024, 7, 15, 12, 0, 0);
        assert_eq!(offset_at("America/New_York", summer), Some(-14400));
    }

    #[test_case]
    fn us_spring_forward_2024() {
        // 2nd Sunday March 2024 = March 10. At 02:00 EST → 03:00 EDT.
        // 2024-03-10 06:59 UTC still EST; 07:00 UTC is EDT.
        let before = super::super::unix_from_civil(2024, 3, 10, 6, 59, 0);
        let after = super::super::unix_from_civil(2024, 3, 10, 7, 0, 0);
        assert_eq!(offset_at("America/New_York", before), Some(-18000));
        assert_eq!(offset_at("America/New_York", after), Some(-14400));
    }

    #[test_case]
    fn paris_dst() {
        let winter = super::super::unix_from_civil(2024, 1, 15, 12, 0, 0);
        let summer = super::super::unix_from_civil(2024, 7, 15, 12, 0, 0);
        assert_eq!(offset_at("Europe/Paris", winter), Some(3600));
        assert_eq!(offset_at("Europe/Paris", summer), Some(7200));
    }

    #[test_case]
    fn lookup_case_insensitive() {
        assert!(lookup("america/new_york").is_some());
        assert!(lookup("UTC").is_some());
        assert!(lookup("Not/AZone").is_none());
    }
}
