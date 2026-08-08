//! HLS playlist parser (RFC 8216) — the pure half of `/open .m3u8`.
//!
//! Media playlists list segments (`#EXTINF` + URI); master playlists list
//! variants (`#EXT-X-STREAM-INF` + URI). Both are line-oriented text, so the
//! whole file is small and this module stays allocation-light and fully
//! unit-tested off-network.
//!
//! Out of scope here (and refused by the loader): AES-encrypted segments
//! (`#EXT-X-KEY` with a METHOD other than `NONE`), and I-frame-only playlists.
//! Live sliding windows (no `#EXT-X-ENDLIST`) are described so the caller can
//! decide; the first player path is VOD.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// One media segment in a media playlist.
#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    /// URI as written in the playlist (may be relative).
    pub uri: String,
    /// `#EXTINF` duration in milliseconds (rounded from the float).
    pub duration_ms: u64,
    /// Optional `#EXT-X-BYTERANGE` length; `None` means the whole resource.
    pub byterange_len: Option<u64>,
    /// Optional `#EXT-X-BYTERANGE` offset (defaults to 0 when length is set
    /// without `@offset`).
    pub byterange_offset: u64,
    /// True when this segment should be treated as a discontinuity restart
    /// (`#EXT-X-DISCONTINUITY` immediately before it).
    pub discontinuity: bool,
}

/// A variant stream in a master playlist.
#[derive(Clone, Debug, PartialEq)]
pub struct Variant {
    pub uri: String,
    pub bandwidth: u32,
    pub average_bandwidth: Option<u32>,
    pub resolution: Option<(u32, u32)>,
    pub codecs: Option<String>,
}

/// `#EXT-X-MAP` init segment (CMAF / fMP4).
#[derive(Clone, Debug, PartialEq)]
pub struct Map {
    pub uri: String,
    pub byterange_len: Option<u64>,
    pub byterange_offset: u64,
}

/// Parsed playlist — either a master or a media list, never both.
#[derive(Clone, Debug, PartialEq)]
pub enum Playlist {
    Master {
        variants: Vec<Variant>,
    },
    Media {
        target_duration_ms: u64,
        media_sequence: u64,
        end_list: bool,
        map: Option<Map>,
        segments: Vec<Segment>,
        /// `true` if any `#EXT-X-KEY` names a real cipher (AES-128, SAMPLE-AES…).
        encrypted: bool,
    },
}

struct StreamInfAttrs {
    bandwidth: u32,
    average_bandwidth: Option<u32>,
    resolution: Option<(u32, u32)>,
    codecs: Option<String>,
}

/// Parse a playlist body. Accepts `\n` or `\r\n` lines; blank lines and
/// comments (`#` that is not a known tag) are ignored.
pub fn parse(text: &str) -> Result<Playlist, &'static str> {
    // Strip a UTF-8 BOM: `hls::looks_like_playlist` (the sniff that routes a
    // file here) already ignores one, so not doing the same would make a
    // BOM-prefixed playlist sniff as HLS and then fail to parse as one.
    let text = text.trim_start_matches('\u{feff}');
    let mut lines = text.lines().map(|l| l.trim()).filter(|l| !l.is_empty());
    let first = lines.next().ok_or("m3u8: empty")?;
    if !first.eq_ignore_ascii_case("#EXTM3U") {
        return Err("m3u8: missing #EXTM3U");
    }

    let mut variants: Vec<Variant> = Vec::new();
    let mut segments: Vec<Segment> = Vec::new();
    let mut target_duration_ms = 0u64;
    let mut media_sequence = 0u64;
    let mut end_list = false;
    let mut map: Option<Map> = None;
    let mut encrypted = false;
    let mut pending_inf: Option<u64> = None;
    let mut pending_byterange: Option<(u64, u64)> = None;
    let mut pending_stream_inf: Option<StreamInfAttrs> = None;
    let mut discontinuity = false;
    let mut saw_media_tag = false;
    let mut saw_master_tag = false;

    for line in lines {
        if line.starts_with('#') {
            if let Some(rest) = line.strip_prefix("#EXT-X-VERSION:") {
                let _ = rest;
            } else if let Some(rest) = line.strip_prefix("#EXT-X-TARGETDURATION:") {
                saw_media_tag = true;
                // RFC 8216 says decimal-integer, but real-world playlists write
                // `10.0`; a duration this file only reports must not fail the
                // whole parse.
                target_duration_ms = parse_seconds_ms(rest)?;
            } else if let Some(rest) = line.strip_prefix("#EXT-X-MEDIA-SEQUENCE:") {
                saw_media_tag = true;
                media_sequence = rest
                    .trim()
                    .parse()
                    .map_err(|_| "m3u8: bad MEDIA-SEQUENCE")?;
            } else if line == "#EXT-X-ENDLIST" {
                saw_media_tag = true;
                end_list = true;
            } else if line == "#EXT-X-DISCONTINUITY" {
                saw_media_tag = true;
                discontinuity = true;
            } else if let Some(rest) = line.strip_prefix("#EXTINF:") {
                saw_media_tag = true;
                let dur_s = rest.split(',').next().unwrap_or("0");
                pending_inf = Some(parse_seconds_ms(dur_s)?);
            } else if let Some(rest) = line.strip_prefix("#EXT-X-BYTERANGE:") {
                saw_media_tag = true;
                pending_byterange = Some(parse_byterange(rest)?);
            } else if let Some(rest) = line.strip_prefix("#EXT-X-MAP:") {
                saw_media_tag = true;
                map = Some(parse_map(rest)?);
            } else if let Some(rest) = line.strip_prefix("#EXT-X-KEY:") {
                saw_media_tag = true;
                if key_is_encrypted(rest) {
                    encrypted = true;
                }
            } else if let Some(rest) = line.strip_prefix("#EXT-X-STREAM-INF:") {
                saw_master_tag = true;
                pending_stream_inf = Some(parse_stream_inf(rest)?);
            } else if line.starts_with("#EXT-X-I-FRAME-STREAM-INF")
                || line.starts_with("#EXT-X-MEDIA:")
                || line.starts_with("#EXT-X-SESSION-")
            {
                saw_master_tag = true;
            }
            continue;
        }

        if let Some(partial) = pending_stream_inf.take() {
            variants.push(Variant {
                uri: line.to_string(),
                bandwidth: partial.bandwidth,
                average_bandwidth: partial.average_bandwidth,
                resolution: partial.resolution,
                codecs: partial.codecs,
            });
            continue;
        }
        if let Some(dur) = pending_inf.take() {
            let (blen, boff) = pending_byterange.take().unwrap_or((0, 0));
            segments.push(Segment {
                uri: line.to_string(),
                duration_ms: dur,
                byterange_len: if blen > 0 { Some(blen) } else { None },
                byterange_offset: boff,
                discontinuity,
            });
            discontinuity = false;
            continue;
        }
        return Err("m3u8: URI without EXTINF or STREAM-INF");
    }

    if saw_master_tag && !variants.is_empty() {
        return Ok(Playlist::Master { variants });
    }
    if saw_media_tag || !segments.is_empty() {
        if segments.is_empty() {
            return Err("m3u8: media playlist has no segments");
        }
        return Ok(Playlist::Media {
            target_duration_ms,
            media_sequence,
            end_list,
            map,
            segments,
            encrypted,
        });
    }
    if !variants.is_empty() {
        return Ok(Playlist::Master { variants });
    }
    Err("m3u8: no segments or variants")
}

/// Pick a media variant from a master list: highest `BANDWIDTH`, ties broken
/// by listed order (stable max). Empty input returns `None`.
pub fn pick_variant(variants: &[Variant]) -> Option<&Variant> {
    variants.iter().max_by_key(|v| v.bandwidth)
}

/// Resolve `uri` against `base` (playlist URL or file path). Absolute http(s)
/// URIs and absolute paths (`/…`) are returned unchanged; everything else is
/// joined on the last `/` of `base`.
pub fn resolve_uri(base: &str, uri: &str) -> String {
    let uri = uri.trim();
    if uri.is_empty() {
        return base.to_string();
    }
    if uri.starts_with("http://") || uri.starts_with("https://") {
        return uri.to_string();
    }
    if uri.starts_with('/') {
        if let Some(scheme_end) = base.find("://") {
            let after = &base[scheme_end + 3..];
            if let Some(slash) = after.find('/') {
                return alloc::format!("{}{}", &base[..scheme_end + 3 + slash], uri);
            }
            return alloc::format!("{}{}", base.trim_end_matches('/'), uri);
        }
        return uri.to_string();
    }
    if let Some(i) = base.rfind('/') {
        alloc::format!("{}{}", &base[..=i], uri)
    } else {
        uri.to_string()
    }
}

/// Parse a decimal-seconds duration (`#EXTINF`, `#EXT-X-TARGETDURATION`) into
/// milliseconds **without floating point**.
///
/// `core` has no `f64::round` (it lives in `std`/libm), and a playlist's
/// durations are exact decimal text — `9.009`, `4.00000` — so scaling the digits
/// is both simpler and exact where `secs * 1000.0` would round. A fourth
/// fraction digit rounds half-up; further digits are ignored, since a
/// sub-millisecond segment boundary is below the player's timebase anyway.
pub fn parse_seconds_ms(s: &str) -> Result<u64, &'static str> {
    const BAD: &str = "m3u8: bad duration";
    let s = s.trim();
    let s = s.strip_prefix('+').unwrap_or(s);
    let (int_part, frac_part) = s.split_once('.').unwrap_or((s, ""));
    if (int_part.is_empty() && frac_part.is_empty())
        || !int_part.bytes().all(|b| b.is_ascii_digit())
        || !frac_part.bytes().all(|b| b.is_ascii_digit())
    {
        return Err(BAD);
    }
    let secs: u64 = if int_part.is_empty() {
        0
    } else {
        int_part.parse().map_err(|_| BAD)?
    };
    let mut digits = frac_part.bytes();
    let mut ms = 0u64;
    for _ in 0..3 {
        ms = ms * 10 + digits.next().map(|b| u64::from(b - b'0')).unwrap_or(0);
    }
    if digits.next().is_some_and(|b| b >= b'5') {
        ms += 1;
    }
    Ok(secs.saturating_mul(1000).saturating_add(ms))
}

fn parse_byterange(s: &str) -> Result<(u64, u64), &'static str> {
    let s = s.trim();
    if let Some((n, o)) = s.split_once('@') {
        let len: u64 = n.trim().parse().map_err(|_| "m3u8: bad BYTERANGE len")?;
        let off: u64 = o.trim().parse().map_err(|_| "m3u8: bad BYTERANGE offset")?;
        Ok((len, off))
    } else {
        let len: u64 = s.parse().map_err(|_| "m3u8: bad BYTERANGE len")?;
        Ok((len, 0))
    }
}

fn parse_map(attrs: &str) -> Result<Map, &'static str> {
    let uri = attr_quoted(attrs, "URI").ok_or("m3u8: EXT-X-MAP missing URI")?;
    let (blen, boff) = if let Some(br) = attr_raw(attrs, "BYTERANGE") {
        parse_byterange(br)?
    } else {
        (0, 0)
    };
    Ok(Map {
        uri,
        byterange_len: if blen > 0 { Some(blen) } else { None },
        byterange_offset: boff,
    })
}

fn parse_stream_inf(attrs: &str) -> Result<StreamInfAttrs, &'static str> {
    let bw = attr_raw(attrs, "BANDWIDTH")
        .or_else(|| attr_raw(attrs, "AVERAGE-BANDWIDTH"))
        .ok_or("m3u8: STREAM-INF missing BANDWIDTH")?;
    let bandwidth: u32 = bw.trim().parse().map_err(|_| "m3u8: bad BANDWIDTH")?;
    let average_bandwidth = attr_raw(attrs, "AVERAGE-BANDWIDTH").and_then(|s| s.trim().parse().ok());
    let resolution = attr_raw(attrs, "RESOLUTION").and_then(|s| {
        let (w, h) = s.split_once('x')?;
        Some((w.parse().ok()?, h.parse().ok()?))
    });
    let codecs = attr_quoted(attrs, "CODECS").or_else(|| {
        attr_raw(attrs, "CODECS").map(|s| s.trim().trim_matches('"').to_string())
    });
    Ok(StreamInfAttrs {
        bandwidth,
        average_bandwidth,
        resolution,
        codecs,
    })
}

fn key_is_encrypted(attrs: &str) -> bool {
    match attr_raw(attrs, "METHOD").map(|s| s.trim()) {
        None | Some("NONE") => false,
        Some(_) => true,
    }
}

fn attr_quoted(attrs: &str, name: &str) -> Option<String> {
    let key = alloc::format!("{name}=\"");
    let i = attrs.find(&key)?;
    let rest = &attrs[i + key.len()..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

fn attr_raw<'a>(attrs: &'a str, name: &str) -> Option<&'a str> {
    let key = alloc::format!("{name}=");
    let i = attrs.find(&key)?;
    let rest = &attrs[i + key.len()..];
    if rest.starts_with('"') {
        let end = rest[1..].find('"')?;
        return Some(&rest[1..1 + end]);
    }
    Some(rest.split(',').next()?.trim())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn media_vod_playlist_parses_segments_and_endlist() {
        let text = "\
#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:4
#EXT-X-MEDIA-SEQUENCE:0
#EXTINF:4.0,
seg0.ts
#EXTINF:3.5,
seg1.ts
#EXT-X-ENDLIST
";
        match parse(text).unwrap() {
            Playlist::Media {
                target_duration_ms,
                end_list,
                segments,
                encrypted,
                ..
            } => {
                assert_eq!(target_duration_ms, 4000);
                assert!(end_list);
                assert!(!encrypted);
                assert_eq!(segments.len(), 2);
                assert_eq!(segments[0].uri, "seg0.ts");
                assert_eq!(segments[0].duration_ms, 4000);
                assert_eq!(segments[1].duration_ms, 3500);
            }
            _ => panic!("expected media"),
        }
    }

    #[test_case]
    fn master_picks_highest_bandwidth() {
        let text = "\
#EXTM3U
#EXT-X-STREAM-INF:BANDWIDTH=800000,RESOLUTION=640x360
low.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=2500000,RESOLUTION=1280x720
hi.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=1200000
mid.m3u8
";
        match parse(text).unwrap() {
            Playlist::Master { variants } => {
                assert_eq!(variants.len(), 3);
                let v = pick_variant(&variants).unwrap();
                assert_eq!(v.uri, "hi.m3u8");
                assert_eq!(v.bandwidth, 2_500_000);
                assert_eq!(v.resolution, Some((1280, 720)));
            }
            _ => panic!("expected master"),
        }
    }

    #[test_case]
    fn resolve_uri_joins_relative_and_keeps_absolute() {
        assert_eq!(
            resolve_uri("https://cdn.example/a/b/pl.m3u8", "seg0.ts"),
            "https://cdn.example/a/b/seg0.ts"
        );
        assert_eq!(
            resolve_uri("https://cdn.example/a/b/pl.m3u8", "/root.ts"),
            "https://cdn.example/root.ts"
        );
        assert_eq!(
            resolve_uri("https://cdn.example/a/b/pl.m3u8", "https://other/x.ts"),
            "https://other/x.ts"
        );
        assert_eq!(
            resolve_uri("/downloads/pl.m3u8", "seg0.ts"),
            "/downloads/seg0.ts"
        );
    }

    #[test_case]
    fn encrypted_key_is_flagged() {
        let text = "\
#EXTM3U
#EXT-X-TARGETDURATION:2
#EXT-X-KEY:METHOD=AES-128,URI=\"key.bin\"
#EXTINF:2,
a.ts
#EXT-X-ENDLIST
";
        match parse(text).unwrap() {
            Playlist::Media { encrypted, .. } => assert!(encrypted),
            _ => panic!("media"),
        }
    }

    #[test_case]
    fn durations_are_parsed_without_floating_point() {
        // `core` has no `f64::round`, and the decimal text is exact anyway.
        assert_eq!(parse_seconds_ms("4"), Ok(4000));
        assert_eq!(parse_seconds_ms("4.0"), Ok(4000));
        assert_eq!(parse_seconds_ms("3.5"), Ok(3500));
        assert_eq!(parse_seconds_ms(" 9.009 "), Ok(9009));
        assert_eq!(parse_seconds_ms("10.00000"), Ok(10_000));
        assert_eq!(parse_seconds_ms("0.5"), Ok(500));
        assert_eq!(parse_seconds_ms(".5"), Ok(500));
        // A fourth fraction digit rounds half-up; further digits are below the
        // player's timebase.
        assert_eq!(parse_seconds_ms("1.0004"), Ok(1000));
        assert_eq!(parse_seconds_ms("1.0005"), Ok(1001));
        assert_eq!(parse_seconds_ms("1.9999"), Ok(2000));
        // Not a duration at all — refused rather than silently zero.
        assert!(parse_seconds_ms("").is_err());
        assert!(parse_seconds_ms("-1").is_err());
        assert!(parse_seconds_ms("abc").is_err());
        assert!(parse_seconds_ms("1.2.3").is_err());
    }

    #[test_case]
    fn extinf_titles_and_odd_durations_do_not_break_the_parse() {
        let text = "\
#EXTM3U
#EXT-X-TARGETDURATION:10.0
#EXTINF:9.009,segment one
a.ts
#EXTINF:9.009,
b.ts
#EXT-X-ENDLIST
";
        match parse(text).unwrap() {
            Playlist::Media {
                segments,
                target_duration_ms,
                ..
            } => {
                assert_eq!(target_duration_ms, 10_000);
                assert_eq!(segments.len(), 2);
                assert_eq!(segments[0].duration_ms, 9009);
                assert_eq!(segments[0].uri, "a.ts");
                assert_eq!(segments[1].duration_ms, 9009);
            }
            _ => panic!("media"),
        }
    }

    #[test_case]
    fn a_discontinuity_marks_only_the_segment_that_follows_it() {
        let text = "\
#EXTM3U
#EXT-X-TARGETDURATION:2
#EXTINF:2,
a.ts
#EXT-X-DISCONTINUITY
#EXTINF:2,
b.ts
#EXTINF:2,
c.ts
#EXT-X-ENDLIST
";
        match parse(text).unwrap() {
            Playlist::Media { segments, .. } => {
                assert!(!segments[0].discontinuity);
                assert!(segments[1].discontinuity, "the segment after the tag");
                assert!(!segments[2].discontinuity, "and only that one");
            }
            _ => panic!("media"),
        }
    }

    #[test_case]
    fn a_uri_without_its_tag_is_refused() {
        // Silently treating a stray line as a segment would produce a playlist
        // that downloads something arbitrary.
        let text = "#EXTM3U\n#EXT-X-TARGETDURATION:2\nstray.ts\n";
        assert!(parse(text).is_err());
        assert!(parse("").is_err());
        assert!(parse("not a playlist\n").is_err());
    }

    #[test_case]
    fn crlf_and_a_byte_order_mark_parse_the_same_as_plain_lines() {
        let text = "\u{feff}#EXTM3U\r\n#EXT-X-TARGETDURATION:2\r\n#EXTINF:2,\r\na.ts\r\n#EXT-X-ENDLIST\r\n";
        match parse(text).unwrap() {
            Playlist::Media {
                segments, end_list, ..
            } => {
                assert!(end_list);
                assert_eq!(segments.len(), 1);
                assert_eq!(segments[0].uri, "a.ts", "no stray \\r on the URI");
            }
            _ => panic!("media"),
        }
    }

    #[test_case]
    fn byterange_and_map_parse() {
        let text = "\
#EXTM3U
#EXT-X-TARGETDURATION:2
#EXT-X-MAP:URI=\"init.mp4\",BYTERANGE=\"500@0\"
#EXTINF:2,
#EXT-X-BYTERANGE:1000@500
seg.mp4
#EXT-X-ENDLIST
";
        match parse(text).unwrap() {
            Playlist::Media { map, segments, .. } => {
                let m = map.unwrap();
                assert_eq!(m.uri, "init.mp4");
                assert_eq!(m.byterange_len, Some(500));
                assert_eq!(segments[0].byterange_len, Some(1000));
                assert_eq!(segments[0].byterange_offset, 500);
            }
            _ => panic!("media"),
        }
    }
}
