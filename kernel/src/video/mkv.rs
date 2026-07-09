//! Matroska / WebM (EBML) demuxer. Walks the EBML element tree, finds the
//! H.264 video track (`V_MPEG4/ISO/AVC`, whose `CodecPrivate` is an `avcC`
//! record), and collects the coded frames from the cluster `SimpleBlock`s.
//! Feeds the same H.264 decoder as [`super::mp4`].
//!
//! Scope: H.264-in-Matroska, `SimpleBlock` framing, no lacing (what x264/most
//! muxers emit for baseline). VP8/VP9 WebM video reports as an unsupported
//! codec (this OS decodes H.264, not VP*). Pure over an in-memory `&[u8]`.

use super::mp4::{AvcC, Sample};
use alloc::vec::Vec;

// Matroska element IDs (with their length-descriptor bits, as read).
const ID_SEGMENT: u32 = 0x1853_8067;
const ID_TRACKS: u32 = 0x1654_AE6B;
const ID_TRACKENTRY: u32 = 0xAE;
const ID_TRACKTYPE: u32 = 0x83;
const ID_CODECID: u32 = 0x86;
const ID_CODECPRIVATE: u32 = 0x63A2;
const ID_VIDEO: u32 = 0xE0;
const ID_PIXELWIDTH: u32 = 0xB0;
const ID_PIXELHEIGHT: u32 = 0xBA;
const ID_CLUSTER: u32 = 0x1F43_B675;
const ID_TIMESTAMP: u32 = 0xE7;
const ID_SIMPLEBLOCK: u32 = 0xA3;
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

/// A demuxed Matroska H.264 track.
pub struct MkvTrack {
    pub width: u32,
    pub height: u32,
    pub timescale: u32, // ticks per second for the sample DTS values
    pub avcc: AvcC,
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
    let mut avcc: Option<AvcC> = None;
    let mut width = 0u32;
    let mut height = 0u32;
    let mut timestamp_scale_ns = 1_000_000u64; // default 1 ms
    let mut samples: Vec<Sample> = Vec::new();

    // Walk the Segment's children.
    walk(data, seg.0, seg.1, &mut |id, body, sz| {
        match id {
            ID_INFO => walk_info(data, body, body + sz, &mut timestamp_scale_ns),
            ID_TRACKS => walk_tracks(data, body, body + sz, &mut codec, &mut codec_len, &mut avcc, &mut width, &mut height),
            ID_CLUSTER => walk_cluster(data, body, body + sz, &mut samples),
            _ => {}
        }
    });

    let codec_str = &codec[..codec_len];
    if codec_str != b"V_MPEG4/ISO/AVC" {
        return Err("mkv: video track is not H.264 (only V_MPEG4/ISO/AVC decodes; VP8/VP9 unsupported)");
    }
    let avcc = avcc.ok_or("mkv: no avcC (CodecPrivate)")?;
    if samples.is_empty() {
        return Err("mkv: no coded frames");
    }
    // Timestamp scale is ns/tick; express DTS in a 1000-tick/s (ms) timescale.
    let timescale = (1_000_000_000u64 / timestamp_scale_ns.max(1)) as u32;
    Ok(MkvTrack { width, height, timescale, avcc, samples })
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

fn walk_tracks(data: &[u8], start: usize, end: usize, codec: &mut [u8; 32], codec_len: &mut usize, avcc: &mut Option<AvcC>, w: &mut u32, h: &mut u32) {
    walk(data, start, end, &mut |id, body, sz| {
        if id == ID_TRACKENTRY {
            walk(data, body, body + sz, &mut |id2, b2, s2| match id2 {
                ID_CODECID => {
                    let n = s2.min(32);
                    codec[..n].copy_from_slice(&data[b2..b2 + n]);
                    *codec_len = n;
                }
                ID_CODECPRIVATE => {
                    if let Ok(a) = super::mp4::parse_avcc(&data[b2..b2 + s2]) {
                        *avcc = Some(a);
                    }
                }
                ID_VIDEO => walk(data, b2, b2 + s2, &mut |id3, b3, s3| match id3 {
                    ID_PIXELWIDTH => *w = read_uint(data, b3, s3) as u32,
                    ID_PIXELHEIGHT => *h = read_uint(data, b3, s3) as u32,
                    _ => {}
                }),
                ID_TRACKTYPE => {}
                _ => {}
            });
        }
    });
}

fn walk_cluster(data: &[u8], start: usize, end: usize, samples: &mut Vec<Sample>) {
    let mut cluster_ts = 0u64;
    walk(data, start, end, &mut |id, body, sz| match id {
        ID_TIMESTAMP => cluster_ts = read_uint(data, body, sz),
        ID_SIMPLEBLOCK => {
            // track_number (vint), int16 relative timecode, flags, then frame.
            if let Some((_track, tl, _u)) = read_size(data, body) {
                let tc_off = body + tl;
                if tc_off + 3 > body + sz {
                    return;
                }
                let tc = i16::from_be_bytes([data[tc_off], data[tc_off + 1]]) as i64;
                let flags = data[tc_off + 2];
                // No lacing (bits 1..2 == 0) — one frame per block.
                if flags & 0x06 != 0 {
                    return;
                }
                let frame_off = tc_off + 3;
                let frame_size = (body + sz) - frame_off;
                let is_sync = flags & 0x80 != 0; // keyframe bit
                let dts = (cluster_ts as i64 + tc).max(0) as u64;
                samples.push(Sample { offset: frame_off, size: frame_size, dts, cts: dts, is_sync });
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
