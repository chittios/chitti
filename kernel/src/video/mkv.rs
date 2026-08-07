//! Matroska / WebM (EBML) demuxer. Walks the EBML element tree, finds the first
//! decodable video track, and collects the coded frames from the cluster
//! `SimpleBlock`/`BlockGroup` elements. Feeds the same decoders as
//! [`super::mp4`].
//!
//! Recognised tracks, and where the decoder configuration comes from:
//!
//! | `CodecID`            | `CodecPrivate` | codec |
//! |----------------------|----------------|-------|
//! | `V_MPEG4/ISO/AVC`    | `avcC`         | H.264 |
//! | `V_MPEGH/ISO/HEVC`   | `hvcC`         | H.265 |
//! | `V_VP9`              | *(none)*       | VP9   |
//!
//! **VP9 has no `CodecPrivate`** and needs none — every VP9 frame header is
//! self-describing — so a demuxer that requires one rejects every WebM file.
//! VP8 is recognised and refused by name rather than being read as VP9, which
//! would decode a whole frame of noise instead of erroring.
//!
//! Scope: no lacing (what every muxer emits for video). Pure over an in-memory
//! `&[u8]`.

use super::mp4::{CodecConfig, Sample, VpcC};
use alloc::vec::Vec;

// Matroska element IDs (with their length-descriptor bits, as read).
const ID_SEGMENT: u32 = 0x1853_8067;
const ID_TRACKS: u32 = 0x1654_AE6B;
const ID_TRACKENTRY: u32 = 0xAE;
const ID_TRACKNUMBER: u32 = 0xD7;
const ID_TRACKTYPE: u32 = 0x83;
const ID_CODECID: u32 = 0x86;
const ID_CODECPRIVATE: u32 = 0x63A2;
const ID_VIDEO: u32 = 0xE0;
const ID_PIXELWIDTH: u32 = 0xB0;
const ID_PIXELHEIGHT: u32 = 0xBA;
const ID_CLUSTER: u32 = 0x1F43_B675;
const ID_TIMESTAMP: u32 = 0xE7;
const ID_SIMPLEBLOCK: u32 = 0xA3;
const ID_BLOCKGROUP: u32 = 0xA0;
const ID_BLOCK: u32 = 0xA1;
const ID_REFERENCEBLOCK: u32 = 0xFB;
const ID_INFO: u32 = 0x1549_A966;
const ID_TIMESTAMPSCALE: u32 = 0x2AD7_B1;

/// Read an EBML element ID at `data[i]`; returns `(id, byte_len)`. The ID keeps
/// its length-marker bits (as elements are matched against the constants above).
fn read_id(data: &[u8], i: usize) -> Option<(u32, usize)> {
    let first = *data.get(i)?;
    let mut mask = 0x80u8;
    let mut ln = 1usize;
    while ln <= 4 && first & mask == 0 {
        mask >>= 1;
        ln += 1;
    }
    if ln > 4 || i + ln > data.len() {
        return None;
    }
    let mut id = 0u32;
    for k in 0..ln {
        id = (id << 8) | data[i + k] as u32;
    }
    Some((id, ln))
}

/// Read an EBML variable-length size at `data[i]`; returns `(value, byte_len,
/// unknown)`. `unknown` = the all-ones "unknown size" form.
fn read_size(data: &[u8], i: usize) -> Option<(u64, usize, bool)> {
    let first = *data.get(i)?;
    let mut mask = 0x80u8;
    let mut ln = 1usize;
    while ln <= 8 && first & mask == 0 {
        mask >>= 1;
        ln += 1;
    }
    if ln > 8 || i + ln > data.len() {
        return None;
    }
    let mut val = (first & (mask - 1)) as u64;
    let mut all_ones = (first & (mask - 1)) == (mask - 1);
    for k in 1..ln {
        val = (val << 8) | data[i + k] as u64;
        all_ones = all_ones && data[i + k] == 0xff;
    }
    Some((val, ln, all_ones))
}

fn read_uint(data: &[u8], off: usize, size: usize) -> u64 {
    let mut v = 0u64;
    for k in 0..size {
        v = (v << 8) | *data.get(off + k).unwrap_or(&0) as u64;
    }
    v
}

/// A demuxed Matroska video track.
pub struct MkvTrack {
    pub width: u32,
    pub height: u32,
    pub timescale: u32, // ticks per second for the sample DTS values
    pub config: CodecConfig,
    pub samples: Vec<Sample>,
}

impl MkvTrack {
    pub fn duration_ms(&self) -> u64 {
        self.samples.last().map(|s| if self.timescale > 0 { s.dts * 1000 / self.timescale as u64 } else { 0 }).unwrap_or(0)
    }
    pub fn frame_count(&self) -> usize {
        self.samples.len()
    }
}

/// Parse a Matroska/WebM file and return its H.264 video track.
pub fn parse(data: &[u8]) -> Result<MkvTrack, &'static str> {
    // Find the Segment (skip the EBML header).
    let mut i = 0usize;
    let seg = loop {
        let (id, l) = read_id(data, i).ok_or("mkv: truncated")?;
        i += l;
        let (sz, l, unknown) = read_size(data, i).ok_or("mkv: truncated size")?;
        i += l;
        if id == ID_SEGMENT {
            let end = if unknown { data.len() } else { (i + sz as usize).min(data.len()) };
            break (i, end);
        }
        i += sz as usize;
        if i >= data.len() {
            return Err("mkv: no Segment");
        }
    };

    let mut codec = [0u8; 32];
    let mut codec_len = 0usize;
    let mut private: Option<(usize, usize)> = None;
    let mut width = 0u32;
    let mut height = 0u32;
    let mut timestamp_scale_ns = 1_000_000u64; // default 1 ms
    let mut samples: Vec<Sample> = Vec::new();

    // Two passes over the Segment: the cluster walk needs the video track's
    // number to filter blocks by, and `Tracks` is only *conventionally* before
    // the first `Cluster`. A file that orders them the other way would otherwise
    // demux zero frames.
    let mut video_track: Option<u64> = None;
    walk(data, seg.0, seg.1, &mut |id, body, sz| match id {
        ID_INFO => walk_info(data, body, body + sz, &mut timestamp_scale_ns),
        ID_TRACKS => {
            if video_track.is_none() {
                video_track = walk_tracks(data, body, body + sz, &mut codec, &mut codec_len, &mut private, &mut width, &mut height);
            }
        }
        _ => {}
    });
    let track_num = video_track.ok_or("mkv: no video track found")?;
    walk(data, seg.0, seg.1, &mut |id, body, sz| {
        if id == ID_CLUSTER {
            walk_cluster(data, body, body + sz, &mut samples, track_num);
        }
    });

    // The CodecPrivate is resolved *after* the walk so the codec id decides how
    // to read it — the two elements can arrive in either order, and reading it
    // as an `avcC` on sight is how an `hvcC` came out as a track with one
    // enormous SPS.
    let private = private.map(|(b, s)| &data[b..b + s]);
    let config = match &codec[..codec_len] {
        b"V_MPEG4/ISO/AVC" => {
            CodecConfig::Avc(super::mp4::parse_avcc(private.ok_or("mkv: H.264 track has no avcC (CodecPrivate)")?)?)
        }
        b"V_MPEGH/ISO/HEVC" => {
            CodecConfig::Hevc(super::mp4::parse_hvcc(private.ok_or("mkv: H.265 track has no hvcC (CodecPrivate)")?)?)
        }
        // VP9 carries no CodecPrivate: its frame headers are self-describing.
        b"V_VP9" => CodecConfig::Vp9(VpcC { profile: 0, bit_depth: 8, chroma_subsampling: 1, ..Default::default() }),
        b"V_VP8" => return Err("mkv: VP8 video is not supported (VP9 and H.264/H.265 are)"),
        b"V_AV1" => return Err("mkv: AV1 video is not supported (VP9 and H.264/H.265 are)"),
        b"" => return Err("mkv: no video track found"),
        _ => return Err("mkv: unsupported video codec (H.264, H.265 and VP9 decode)"),
    };
    if samples.is_empty() {
        return Err("mkv: no coded frames");
    }
    // Timestamp scale is ns/tick; express DTS in a 1000-tick/s (ms) timescale.
    let timescale = (1_000_000_000u64 / timestamp_scale_ns.max(1)) as u32;
    Ok(MkvTrack { width, height, timescale, config, samples })
}

/// Iterate the child elements in `[start, end)`, calling `f(id, body_offset,
/// body_size)`. Handles the unknown-size form (extends to `end`).
fn walk<F: FnMut(u32, usize, usize)>(data: &[u8], start: usize, end: usize, f: &mut F) {
    let mut i = start;
    while i < end {
        let (id, l) = match read_id(data, i) {
            Some(x) => x,
            None => break,
        };
        i += l;
        let (sz, l, unknown) = match read_size(data, i) {
            Some(x) => x,
            None => break,
        };
        i += l;
        let body = i;
        let bsz = if unknown { end - body } else { (sz as usize).min(end - body) };
        f(id, body, bsz);
        i = body + bsz;
    }
}

fn walk_info(data: &[u8], start: usize, end: usize, scale: &mut u64) {
    walk(data, start, end, &mut |id, body, sz| {
        if id == ID_TIMESTAMPSCALE {
            *scale = read_uint(data, body, sz);
        }
    });
}

/// Matroska `TrackType` for video.
const TRACK_TYPE_VIDEO: u64 = 1;

/// Collect the **first video** track's codec id, `CodecPrivate` range and pixel
/// size, plus its track number so the cluster walk can filter blocks to it.
///
/// Filtering on `TrackType` matters: a file with audio has more than one
/// `TrackEntry`, and taking whichever `CodecID` came last would describe the
/// video track with the audio codec (or vice versa). `TrackType` may appear
/// *after* `CodecID` inside an entry, so the entry is read into locals and only
/// committed once its type is known.
fn walk_tracks(
    data: &[u8],
    start: usize,
    end: usize,
    codec: &mut [u8; 32],
    codec_len: &mut usize,
    private: &mut Option<(usize, usize)>,
    w: &mut u32,
    h: &mut u32,
) -> Option<u64> {
    let mut chosen: Option<u64> = None;
    walk(data, start, end, &mut |id, body, sz| {
        if id != ID_TRACKENTRY || chosen.is_some() {
            return;
        }
        let mut t_codec = [0u8; 32];
        let mut t_codec_len = 0usize;
        let mut t_private: Option<(usize, usize)> = None;
        let mut t_type = 0u64;
        let mut t_num = 0u64;
        let (mut t_w, mut t_h) = (0u32, 0u32);
        walk(data, body, body + sz, &mut |id2, b2, s2| match id2 {
            ID_CODECID => {
                let n = s2.min(32);
                t_codec[..n].copy_from_slice(&data[b2..b2 + n]);
                t_codec_len = n;
            }
            ID_CODECPRIVATE => t_private = Some((b2, s2)),
            ID_TRACKTYPE => t_type = read_uint(data, b2, s2),
            ID_TRACKNUMBER => t_num = read_uint(data, b2, s2),
            ID_VIDEO => walk(data, b2, b2 + s2, &mut |id3, b3, s3| match id3 {
                ID_PIXELWIDTH => t_w = read_uint(data, b3, s3) as u32,
                ID_PIXELHEIGHT => t_h = read_uint(data, b3, s3) as u32,
                _ => {}
            }),
            _ => {}
        });
        if t_type == TRACK_TYPE_VIDEO {
            *codec = t_codec;
            *codec_len = t_codec_len;
            *private = t_private;
            *w = t_w;
            *h = t_h;
            chosen = Some(t_num);
        }
    });
    chosen
}

/// Decode a `SimpleBlock`/`Block` header into `(track, dts, flags, payload)`.
/// Returns `None` for laced blocks (never emitted for video) and truncation.
fn block_payload(
    data: &[u8],
    body: usize,
    sz: usize,
    cluster_ts: u64,
) -> Option<(u64, u64, u8, usize, usize)> {
    // track_number (vint), int16 relative timecode, flags, then the frame.
    let (track, tl, _u) = read_size(data, body)?;
    let tc_off = body + tl;
    if tc_off + 3 > body + sz {
        return None;
    }
    let tc = i16::from_be_bytes([data[tc_off], data[tc_off + 1]]) as i64;
    let flags = data[tc_off + 2];
    // No lacing (bits 1..2 == 0) — one frame per block.
    if flags & 0x06 != 0 {
        return None;
    }
    let frame_off = tc_off + 3;
    let frame_size = (body + sz) - frame_off;
    let dts = (cluster_ts as i64 + tc).max(0) as u64;
    Some((track, dts, flags, frame_off, frame_size))
}

fn walk_cluster(data: &[u8], start: usize, end: usize, samples: &mut Vec<Sample>, track: u64) {
    let mut cluster_ts = 0u64;
    walk(data, start, end, &mut |id, body, sz| match id {
        ID_TIMESTAMP => cluster_ts = read_uint(data, body, sz),
        ID_SIMPLEBLOCK => {
            if let Some((tn, dts, flags, off, size)) = block_payload(data, body, sz, cluster_ts) {
                if tn == track {
                    samples.push(Sample { offset: off, size, dts, cts: dts, is_sync: flags & 0x80 != 0 });
                }
            }
        }
        // A `BlockGroup` wraps a plain `Block`, which carries no keyframe bit —
        // the absence of any `ReferenceBlock` is what marks it as one. Muxers
        // use this shape whenever a frame needs a duration or references, so a
        // demuxer that reads only `SimpleBlock` silently finds no frames at all
        // in such a file rather than reporting anything.
        ID_BLOCKGROUP => {
            let mut found: Option<(u64, u64, usize, usize)> = None;
            let mut has_reference = false;
            walk(data, body, body + sz, &mut |id2, b2, s2| match id2 {
                ID_BLOCK => {
                    if let Some((tn, dts, _f, off, size)) = block_payload(data, b2, s2, cluster_ts) {
                        found = Some((tn, dts, off, size));
                    }
                }
                ID_REFERENCEBLOCK => has_reference = true,
                _ => {}
            });
            if let Some((tn, dts, off, size)) = found {
                if tn == track {
                    samples.push(Sample { offset: off, size, dts, cts: dts, is_sync: !has_reference });
                }
            }
        }
        _ => {}
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn ebml_vint_readers() {
        // Size 0x81 = 1-byte, value 1. Size 0x40 0x02 = 2-byte, value 2.
        assert_eq!(read_size(&[0x81], 0), Some((1, 1, false)));
        assert_eq!(read_size(&[0x40, 0x02], 0), Some((2, 2, false)));
        // Unknown size (0xFF) flagged.
        assert_eq!(read_size(&[0xff], 0), Some((0x7f, 1, true)));
        // ID keeps its marker bits.
        assert_eq!(read_id(&[0xA3], 0), Some((0xA3, 1)));
        assert_eq!(read_id(&[0x1F, 0x43, 0xB6, 0x75], 0), Some((0x1F43_B675, 4)));
    }

    #[test_case]
    fn rejects_non_matroska() {
        assert!(parse(b"not an mkv file at all").is_err());
    }
}
