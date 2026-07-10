//! HTTP / cookie date parsing (RFC 6265 cookie-date + IMF-fix RFC 7231).
//!
//! Reference: Ladybird `parse_cookie_date` / HTTP date in LibHTTP.

/// Parse cookie-date or IMF-fix to Unix milliseconds (UTC).
/// Supports:
/// - `Wdy, DD Mon YYYY HH:MM:SS GMT`
/// - `DD-Mon-YYYY HH:MM:SS GMT` (legacy)
pub fn parse_http_date_ms(s: &str) -> Option<u64> {
    let s = s.trim();
    // Strip leading weekday
    let rest = if let Some(i) = s.find(',') {
        s[i + 1..].trim()
    } else {
        s
    };
    // Tokenize on space and `-`
    let mut parts: alloc::vec::Vec<&str> = rest
        .split(|c: char| c == ' ' || c == '-')
        .filter(|t| !t.is_empty())
        .collect();
    // Forms:
    // DD Mon YYYY HH:MM:SS GMT
    // or Mon DD ... (asctime) — skip
    if parts.len() < 4 {
        return None;
    }
    // Detect if first is month name
    let (day, month, year, time) = if month_num(parts[0]).is_some() {
        // Mon DD HH:MM:SS YYYY
        if parts.len() < 5 {
            return None;
        }
        (parts[1], parts[0], parts[4], parts[2])
    } else {
        // DD Mon YYYY HH:MM:SS [GMT]
        (parts[0], parts[1], parts[2], parts[3])
    };
    let d: u32 = day.parse().ok()?;
    let m = month_num(month)?;
    let y: i32 = year.parse().ok()?;
    if !(1..=31).contains(&d) || !(1970..=3000).contains(&y) {
        return None;
    }
    let (hh, mm, ss) = parse_hms(time)?;
    Some(ymd_hms_to_unix_ms(y, m, d, hh, mm, ss)?)
}

fn parse_hms(t: &str) -> Option<(u32, u32, u32)> {
    let mut it = t.split(':');
    let h: u32 = it.next()?.parse().ok()?;
    let m: u32 = it.next()?.parse().ok()?;
    let s: u32 = it.next().unwrap_or("0").parse().ok()?;
    if h > 23 || m > 59 || s > 60 {
        return None;
    }
    Some((h, m, s))
}

fn month_num(m: &str) -> Option<u32> {
    match &m.to_ascii_lowercase()[..m.len().min(3)] {
        "jan" => Some(1),
        "feb" => Some(2),
        "mar" => Some(3),
        "apr" => Some(4),
        "may" => Some(5),
        "jun" => Some(6),
        "jul" => Some(7),
        "aug" => Some(8),
        "sep" => Some(9),
        "oct" => Some(10),
        "nov" => Some(11),
        "dec" => Some(12),
        _ => None,
    }
}

/// Days from civil (y,m,d) → Unix days (Howard Hinnant algorithm).
fn ymd_hms_to_unix_ms(y: i32, m: u32, d: u32, hh: u32, mm: u32, ss: u32) -> Option<u64> {
    let m = m as i32;
    let d = d as i32;
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u32;
    let mp = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy as u32;
    let z = (era as i64) * 146097 + doe as i64 - 719468;
    if z < 0 {
        return None;
    }
    let day_ms = z as u64 * 86_400_000;
    let tod = (hh as u64) * 3_600_000 + (mm as u64) * 60_000 + (ss as u64) * 1000;
    Some(day_ms + tod)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn imf_fix() {
        // 1 Jan 1970 00:00:00 GMT
        let t = parse_http_date_ms("Thu, 01 Jan 1970 00:00:00 GMT").unwrap();
        assert_eq!(t, 0);
        // 1 Jan 2000 00:00:00 ≈ 946684800000
        let t2 = parse_http_date_ms("Sat, 01 Jan 2000 00:00:00 GMT").unwrap();
        assert_eq!(t2, 946_684_800_000);
    }

    #[test_case]
    fn cookie_legacy_dash() {
        let t = parse_http_date_ms("01-Jan-2000 00:00:00 GMT").unwrap();
        assert_eq!(t, 946_684_800_000);
    }
}
