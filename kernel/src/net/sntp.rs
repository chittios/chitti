//! **SNTP client** (RFC 5905 simplified): build a client request, parse a
//! 48-byte server reply into Unix seconds. Pure — no sockets; the net stack
//! ships the bytes via UDP.
//!
//! ## Rules we enforce
//! - Mode must be 4 (server) in the reply.
//! - Stratum 0 is “kiss-o’-death” / unsynchronised — refuse.
//! - LI leap-indicator 3 (unsync) — refuse.
//! - Transmit timestamp must be non-zero.

/// NTP epoch is 1900-01-01; Unix is 1970-01-01. Difference in seconds.
pub const NTP_UNIX_DELTA: u64 = 2_208_988_800;

/// Errors from request build / reply parse.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SntpError {
    BadLength,
    BadMode,
    Unsync,
    KissOfDeath,
    ZeroTimestamp,
    Implausible,
}

/// Build a 48-byte SNTP client request (version 4, mode 3).
/// `xmit_unix` is our best-guess current Unix time (for origin/xmit fields).
pub fn build_request(xmit_unix: u64) -> [u8; 48] {
    let mut p = [0u8; 48];
    // LI=0, VN=4, Mode=3 (client)
    p[0] = (4 << 3) | 3;
    // Transmit timestamp: seconds since 1900, fraction 0.
    let ntp = xmit_unix.saturating_add(NTP_UNIX_DELTA);
    p[40..44].copy_from_slice(&(ntp as u32).to_be_bytes());
    p
}

/// Parse a 48-byte SNTP reply into Unix seconds (UTC), using the server's
/// **transmit** timestamp (bytes 40..44).
pub fn parse_reply(pkt: &[u8]) -> Result<u64, SntpError> {
    if pkt.len() < 48 {
        return Err(SntpError::BadLength);
    }
    let li = pkt[0] >> 6;
    let mode = pkt[0] & 0x7;
    if mode != 4 {
        return Err(SntpError::BadMode);
    }
    if li == 3 {
        return Err(SntpError::Unsync);
    }
    let stratum = pkt[1];
    if stratum == 0 {
        return Err(SntpError::KissOfDeath);
    }
    let ntp_secs = u32::from_be_bytes([pkt[40], pkt[41], pkt[42], pkt[43]]) as u64;
    if ntp_secs == 0 {
        return Err(SntpError::ZeroTimestamp);
    }
    // NTP timestamps wrap in 2036; values below the Unix epoch delta are
    // ambiguous — refuse rather than produce a nonsense past date.
    if ntp_secs < NTP_UNIX_DELTA {
        return Err(SntpError::Implausible);
    }
    Ok(ntp_secs - NTP_UNIX_DELTA)
}

/// Whether `new_unix` is a plausible wall time given our current estimate
/// `now_unix` and whether that estimate is already trusted (RTC/NTP/manual).
///
/// Trusted: within ±1 day. Untrusted (fallback): within 2000-01-01 .. 2100-01-01.
pub fn plausible(now_unix: i64, new_unix: u64, trusted: bool) -> bool {
    let n = new_unix as i64;
    if trusted {
        (n - now_unix).abs() <= 86_400
    } else {
        n >= 946_684_800 && n < 4_102_444_800 // 2000 .. 2100
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn request_is_v4_client_mode() {
        let p = build_request(1_700_000_000);
        assert_eq!(p[0] & 0x7, 3);
        assert_eq!((p[0] >> 3) & 0x7, 4);
        let ntp = u32::from_be_bytes([p[40], p[41], p[42], p[43]]) as u64;
        assert_eq!(ntp, 1_700_000_000 + NTP_UNIX_DELTA);
    }

    #[test_case]
    fn parse_reply_happy_path() {
        let mut p = [0u8; 48];
        p[0] = (4 << 3) | 4; // VN=4 mode=server
        p[1] = 2; // stratum
        let unix = 1_704_067_200u64; // 2024-01-01-ish
        let ntp = (unix + NTP_UNIX_DELTA) as u32;
        p[40..44].copy_from_slice(&ntp.to_be_bytes());
        assert_eq!(parse_reply(&p).unwrap(), unix);
    }

    #[test_case]
    fn parse_refuses_kiss_and_unsync() {
        let mut p = [0u8; 48];
        p[0] = (4 << 3) | 4;
        p[1] = 0;
        p[40] = 0xff;
        assert_eq!(parse_reply(&p), Err(SntpError::KissOfDeath));
        p[1] = 2;
        p[0] = (3 << 6) | (4 << 3) | 4; // LI=3
        assert_eq!(parse_reply(&p), Err(SntpError::Unsync));
    }

    #[test_case]
    fn plausible_windows() {
        let now = 1_704_067_200i64;
        assert!(plausible(now, (now + 100) as u64, true));
        assert!(!plausible(now, (now + 200_000) as u64, true));
        assert!(plausible(0, 1_704_067_200, false));
        assert!(!plausible(0, 100, false));
    }
}
