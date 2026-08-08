//! HLS loader — fetch a playlist + segments and assemble a [`Demuxed`]-shaped
//! buffer the existing [`super::StreamDecoder`] can open.
//!
//! Scope for the first cut (deliberately small, pure-logic first):
//!
//! * **VOD** media playlists (`#EXT-X-ENDLIST`) and masters (highest bandwidth)
//! * **MPEG-TS** segments demuxed by [`super::ts`]
//! * **fMP4** segments / `#EXT-X-MAP` init via [`super::mp4`] when the bytes
//!   sniff as ISO-BMFF
//! * **No encryption** (`#EXT-X-KEY` with a cipher is refused up front)
//! * **No live sliding window** yet — missing ENDLIST is an error with a reason
//!
//! Fetching is injected as a closure so unit tests never touch the net stack,
//! and the shell path pumps `upkeep` / Ctrl+C between segment downloads.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use super::m3u8::{self, Playlist, Segment};
use super::mp4::{self, CodecConfig, Sample};
use super::ts;

/// Fully resolved VOD playlist ready to download: absolute segment URIs and
/// optional init map, with total advertised duration.
#[derive(Clone, Debug)]
pub struct ResolvedVod {
    pub base: String,
    pub map: Option<m3u8::Map>,
    pub segments: Vec<Segment>,
    /// Sum of `#EXTINF` durations.
    pub duration_ms: u64,
    pub target_duration_ms: u64,
}

/// What the player needs after a successful HLS load.
///
/// `Debug` is written by hand rather than derived: `bytes` is the whole VOD, so
/// a derived one would print megabytes of sample data into a panic message.
pub struct LoadedHls {
    pub bytes: Vec<u8>,
    pub config: CodecConfig,
    pub samples: Vec<Sample>,
    pub timescale: u32,
    pub duration_ms: u64,
    pub container: &'static str,
}

impl core::fmt::Debug for LoadedHls {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LoadedHls")
            .field("container", &self.container)
            .field("bytes", &self.bytes.len())
            .field("samples", &self.samples.len())
            .field("timescale", &self.timescale)
            .field("duration_ms", &self.duration_ms)
            .finish()
    }
}

/// Parse `playlist_text` (already fetched) with `base` as the resolution base
/// (the playlist's own URL or store path). Follows one level of master → media.
/// `fetch` returns the body of a URI (absolute after resolve).
pub fn resolve_vod(
    playlist_text: &str,
    base: &str,
    mut fetch: impl FnMut(&str) -> Result<Vec<u8>, String>,
) -> Result<(ResolvedVod, /* media text for debug */ String), String> {
    let pl = m3u8::parse(playlist_text).map_err(|e| e.to_string())?;
    let (media_text, media_base) = match pl {
        Playlist::Master { variants } => {
            let v = m3u8::pick_variant(&variants)
                .ok_or_else(|| String::from("hls: master playlist has no variants"))?;
            let url = m3u8::resolve_uri(base, &v.uri);
            let body = fetch(&url)?;
            let text = core::str::from_utf8(&body)
                .map_err(|_| String::from("hls: media playlist is not UTF-8"))?
                .to_string();
            (text, url)
        }
        Playlist::Media { .. } => (playlist_text.to_string(), base.to_string()),
    };
    let media = m3u8::parse(&media_text).map_err(|e| e.to_string())?;
    match media {
        Playlist::Media {
            end_list,
            encrypted,
            map,
            segments,
            target_duration_ms,
            ..
        } => {
            if encrypted {
                return Err(String::from(
                    "hls: encrypted segments (#EXT-X-KEY) are not supported",
                ));
            }
            if !end_list {
                return Err(String::from(
                    "hls: live playlists (no #EXT-X-ENDLIST) are not supported yet",
                ));
            }
            if segments.is_empty() {
                return Err(String::from("hls: media playlist has no segments"));
            }
            let duration_ms = segments.iter().map(|s| s.duration_ms).sum();
            Ok((
                ResolvedVod {
                    base: media_base,
                    map,
                    segments,
                    duration_ms,
                    target_duration_ms,
                },
                media_text,
            ))
        }
        Playlist::Master { .. } => Err(String::from("hls: nested master playlist")),
    }
}

/// Download every segment of a resolved VOD playlist and demux into one
/// length-prefixed sample table.
///
/// `on_progress(i, n)` is invoked before each segment fetch so the shell can
/// pump the UI and honour Ctrl+C.
pub fn load_vod(
    vod: &ResolvedVod,
    mut fetch: impl FnMut(&str) -> Result<Vec<u8>, String>,
    mut on_progress: impl FnMut(usize, usize) -> Result<(), String>,
) -> Result<LoadedHls, String> {
    let n = vod.segments.len() + if vod.map.is_some() { 1 } else { 0 };
    let mut step = 0usize;

    // A byte-range playlist points every segment at a range of ONE resource, so
    // fetching per segment would download the whole file once per segment —
    // quadratic, and on a long VOD that is the difference between working and
    // not. `fetch` has no range support (it is just "give me this URI"), so the
    // resource is held while consecutive segments keep drawing from it, and
    // dropped as soon as the URI changes. Only ranged segments are cached: a
    // normal playlist must not keep a whole segment alive for nothing.
    let mut cache: Option<(String, Vec<u8>)> = None;
    let mut fetch_range =
        |url: &str, len: Option<u64>, offset: u64| -> Result<Vec<u8>, String> {
            let Some(len) = len else {
                cache = None;
                return fetch(url);
            };
            if cache.as_ref().map(|(u, _)| u.as_str()) != Some(url) {
                cache = Some((url.to_string(), fetch(url)?));
            }
            let (_, body) = cache.as_ref().expect("just filled");
            slice_byterange(body, len, offset)
        };

    // Optional fMP4 init.
    let mut init_bytes: Option<Vec<u8>> = None;
    if let Some(map) = &vod.map {
        on_progress(step, n)?;
        step += 1;
        let url = m3u8::resolve_uri(&vod.base, &map.uri);
        init_bytes = Some(fetch_range(&url, map.byterange_len, map.byterange_offset)?);
    }

    let mut ts_tracks: Vec<ts::TsTrack> = Vec::new();
    let mut fmp4_parts: Vec<Vec<u8>> = Vec::new();
    if let Some(init) = init_bytes {
        fmp4_parts.push(init);
    }

    for (i, seg) in vod.segments.iter().enumerate() {
        on_progress(step, n)?;
        step += 1;
        let url = m3u8::resolve_uri(&vod.base, &seg.uri);
        // Every failure below names the segment it came from. A VOD is hundreds
        // of segments and any one of them can be the bad one, so a bare
        // "ts: no PAT/PMT" is not a diagnosis — it does not even say whether
        // the download or the demux failed.
        let blame = |e: &str| {
            alloc::format!(
                "hls: segment {}/{} ({}): {}",
                i + 1,
                vod.segments.len(),
                url,
                e
            )
        };
        let body = fetch_range(&url, seg.byterange_len, seg.byterange_offset)
            .map_err(|e| blame(&e))?;

        if is_iso_bmff(&body) {
            fmp4_parts.push(body);
        } else if body.first() == Some(&0x47) {
            if !fmp4_parts.is_empty() {
                return Err(blame("mixed fMP4 and MPEG-TS segments in one playlist"));
            }
            let mut track = ts::demux_video(&body).map_err(blame)?;
            // The playlist, not the bytes, is what knows the clock was spliced.
            track.discontinuity = seg.discontinuity;
            ts_tracks.push(track);
        } else {
            return Err(blame("neither MPEG-TS nor fMP4"));
        }
    }

    if !ts_tracks.is_empty() {
        let (bytes, config, samples, timescale) =
            ts::assemble_samples(&ts_tracks).map_err(|e| e.to_string())?;
        return Ok(LoadedHls {
            bytes,
            config,
            samples,
            timescale,
            duration_ms: vod.duration_ms,
            container: "hls/mpegts",
        });
    }

    // fMP4. Concatenating an init segment and its media fragments is **not** a
    // file `mp4::parse` can read: it expects one `moov` with a sample table,
    // where a fragmented stream carries the table in a `moof` per fragment and
    // `mp4.rs` has no `moof`/`traf`/`trun` demuxer. So the only shape that works
    // here is a single segment that is a complete self-contained file; a real
    // CMAF ladder is refused by name until that demuxer exists, rather than
    // parsed into whichever fragment happens to sit first.
    if fmp4_parts.len() == 1 {
        let bytes = fmp4_parts.remove(0);
        let t = mp4::parse(&bytes).map_err(|e| e.to_string())?;
        let duration_ms = t.duration_ms().max(vod.duration_ms);
        return Ok(LoadedHls {
            bytes,
            config: t.config,
            samples: t.samples,
            timescale: t.timescale,
            duration_ms,
            container: "hls/fmp4",
        });
    }
    Err(alloc::format!(
        "hls: multi-fragment fMP4/CMAF ({} fragments) needs a moof/trun demuxer, which does not exist yet — MPEG-TS segments work",
        fmp4_parts.len()
    ))
}

/// Take `#EXT-X-BYTERANGE`'s `len@offset` window out of a fetched resource.
/// A range past the end is an error rather than a clamp — a short read means
/// the wrong resource was fetched, and half a segment decodes as corruption.
fn slice_byterange(body: &[u8], len: u64, offset: u64) -> Result<Vec<u8>, String> {
    let start = offset as usize;
    let end = start.saturating_add(len as usize);
    if end > body.len() || start > end {
        return Err(alloc::format!(
            "hls: BYTERANGE {len}@{offset} past end of a {}-byte resource",
            body.len()
        ));
    }
    Ok(body[start..end].to_vec())
}

fn is_iso_bmff(bytes: &[u8]) -> bool {
    if bytes.len() < 12 {
        return false;
    }
    let typ = &bytes[4..8];
    typ == b"ftyp" || typ == b"moof" || typ == b"moov" || typ == b"styp" || typ == b"sidx"
}

/// True when `bytes` look like an HLS playlist (text starting with `#EXTM3U`).
pub fn looks_like_playlist(bytes: &[u8]) -> bool {
    let n = bytes.len().min(16);
    let head = core::str::from_utf8(&bytes[..n]).unwrap_or("");
    let t = head.trim_start_matches('\u{feff}').trim_start();
    t.starts_with("#EXTM3U") || t.starts_with("#extm3u")
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::BTreeMap;

    #[test_case]
    fn resolve_vod_follows_master_to_media() {
        let mut files: BTreeMap<&str, &str> = BTreeMap::new();
        files.insert(
            "https://ex/master.m3u8",
            "\
#EXTM3U
#EXT-X-STREAM-INF:BANDWIDTH=1000
media.m3u8
",
        );
        files.insert(
            "https://ex/media.m3u8",
            "\
#EXTM3U
#EXT-X-TARGETDURATION:2
#EXTINF:2,
a.ts
#EXT-X-ENDLIST
",
        );
        let (vod, _) = resolve_vod(
            files["https://ex/master.m3u8"],
            "https://ex/master.m3u8",
            |url| {
                files
                    .get(url)
                    .map(|s| s.as_bytes().to_vec())
                    .ok_or_else(|| alloc::format!("missing {url}"))
            },
        )
        .unwrap();
        assert_eq!(vod.segments.len(), 1);
        assert_eq!(vod.segments[0].uri, "a.ts");
        assert_eq!(vod.duration_ms, 2000);
        assert_eq!(vod.base, "https://ex/media.m3u8");
    }

    #[test_case]
    fn a_byte_range_playlist_fetches_its_resource_once() {
        // Every segment is a range of one file. Fetching per segment would
        // download the whole file once per segment.
        let playlist = "\
#EXTM3U
#EXT-X-TARGETDURATION:2
#EXTINF:2,
#EXT-X-BYTERANGE:10@0
all.ts
#EXTINF:2,
#EXT-X-BYTERANGE:10@10
all.ts
#EXTINF:2,
#EXT-X-BYTERANGE:10@20
all.ts
#EXT-X-ENDLIST
";
        let (vod, _) = resolve_vod(playlist, "https://ex/p.m3u8", |_| {
            panic!("a media playlist needs no fetch")
        })
        .unwrap();
        assert_eq!(vod.segments.len(), 3);

        let mut fetches = 0usize;
        let err = load_vod(
            &vod,
            |url| {
                assert_eq!(url, "https://ex/all.ts");
                fetches += 1;
                Ok(alloc::vec![0u8; 30])
            },
            |_, _| Ok(()),
        )
        .unwrap_err();
        // The payload is not a real segment, so the load fails — but only after
        // the ranges were taken, which is what this is measuring.
        assert!(err.contains("neither MPEG-TS nor fMP4"), "{err}");
        assert_eq!(fetches, 1, "the shared resource is fetched once, not thrice");
    }

    #[test_case]
    fn a_byte_range_past_the_end_is_refused_not_clamped() {
        // A short read means the wrong resource was fetched; half a segment
        // decodes as corruption rather than as an error.
        assert!(slice_byterange(&[0u8; 10], 4, 2).is_ok());
        let err = slice_byterange(&[0u8; 10], 8, 4).unwrap_err();
        assert!(err.contains("past end"), "{err}");
    }

    #[test_case]
    fn a_failing_segment_names_which_one_and_why() {
        let playlist = "\
#EXTM3U
#EXT-X-TARGETDURATION:2
#EXTINF:2,
a.ts
#EXTINF:2,
b.ts
#EXT-X-ENDLIST
";
        let (vod, _) = resolve_vod(playlist, "https://ex/p.m3u8", |_| Ok(Vec::new())).unwrap();
        // A *download* failure names its segment…
        let err = load_vod(
            &vod,
            |url| {
                if url.ends_with("b.ts") {
                    Err(String::from("HTTP 404"))
                } else {
                    Ok(iso_bmff_stub())
                }
            },
            |_, _| Ok(()),
        )
        .unwrap_err();
        assert!(err.contains("2/2"), "says which segment: {err}");
        assert!(err.contains("b.ts") && err.contains("404"), "{err}");

        // …and so does a *demux* failure, which is the harder case: a bare
        // "ts: no PAT/PMT" over a 300-segment VOD says neither which segment
        // nor whether the download or the parse was at fault.
        let err = load_vod(&vod, |_| Ok(alloc::vec![0x47u8; 188]), |_, _| Ok(())).unwrap_err();
        assert!(err.contains("1/2") && err.contains("a.ts"), "{err}");
        assert!(err.contains("PAT") || err.contains("PMT"), "{err}");
    }

    /// Twelve bytes that sniff as ISO-BMFF — enough to get past the container
    /// check without being a real fragment.
    fn iso_bmff_stub() -> Vec<u8> {
        let mut v = alloc::vec![0u8; 12];
        v[4..8].copy_from_slice(b"styp");
        v
    }

    #[test_case]
    fn progress_is_reported_once_per_download_and_can_cancel() {
        let playlist = "\
#EXTM3U
#EXT-X-TARGETDURATION:2
#EXT-X-MAP:URI=\"init.mp4\"
#EXTINF:2,
a.m4s
#EXTINF:2,
b.m4s
#EXT-X-ENDLIST
";
        let (vod, _) = resolve_vod(playlist, "https://ex/p.m3u8", |_| Ok(Vec::new())).unwrap();
        // The init segment counts as a step: it is a download the user waits on.
        let mut steps = Vec::new();
        let err = load_vod(
            &vod,
            |_| Ok(alloc::vec![0u8; 4]),
            |i, n| {
                steps.push((i, n));
                Err(String::from("cancelled"))
            },
        )
        .unwrap_err();
        assert_eq!(err, "cancelled", "Ctrl+C stops the download");
        assert_eq!(steps, alloc::vec![(0, 3)]);
    }

    #[test_case]
    fn mixed_segment_containers_are_refused() {
        // fMP4 first, so the TS segment is reached with fragments already
        // collected — the state the guard is about. (The other order fails at
        // the TS demux before the guard is consulted.)
        let playlist = "\
#EXTM3U
#EXT-X-TARGETDURATION:2
#EXTINF:2,
a.m4s
#EXTINF:2,
b.ts
#EXT-X-ENDLIST
";
        let (vod, _) = resolve_vod(playlist, "https://ex/p.m3u8", |_| Ok(Vec::new())).unwrap();
        let err = load_vod(
            &vod,
            |url| {
                if url.ends_with(".ts") {
                    Ok(alloc::vec![0x47u8; 188])
                } else {
                    Ok(iso_bmff_stub())
                }
            },
            |_, _| Ok(()),
        )
        .unwrap_err();
        assert!(err.contains("mixed"), "{err}");
        assert!(err.contains("2/2"), "and which segment: {err}");
    }

    #[test_case]
    fn a_playlist_is_recognised_by_its_first_line_only() {
        assert!(looks_like_playlist(b"#EXTM3U\n#EXT-X-VERSION:3\n"));
        assert!(looks_like_playlist(b"\xef\xbb\xbf#EXTM3U\n"), "BOM");
        assert!(looks_like_playlist(b"\n  #EXTM3U\n"), "leading blank line");
        assert!(!looks_like_playlist(b"\x00\x00\x00\x18ftypmp42"));
        assert!(!looks_like_playlist(&[0x47u8; 188]));
        assert!(!looks_like_playlist(b""));
    }

    #[test_case]
    fn encrypted_and_live_are_refused() {
        let enc = "\
#EXTM3U
#EXT-X-TARGETDURATION:1
#EXT-X-KEY:METHOD=AES-128,URI=\"k\"
#EXTINF:1,
a.ts
#EXT-X-ENDLIST
";
        let err = resolve_vod(enc, "https://ex/p.m3u8", |_| Ok(Vec::new())).unwrap_err();
        assert!(err.contains("encrypted"), "{err}");

        let live = "\
#EXTM3U
#EXT-X-TARGETDURATION:1
#EXTINF:1,
a.ts
";
        let err = resolve_vod(live, "https://ex/p.m3u8", |_| Ok(Vec::new())).unwrap_err();
        assert!(err.contains("live"), "{err}");
    }
}

