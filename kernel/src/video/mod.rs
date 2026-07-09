//! Video: container demuxing + H.264/AVC decoding for the `/open` player.
//!
//! Built in stages, each pure and unit-tested off-hardware (the standing rule —
//! fiddly logic lives in pure functions verified with cases, and hard numeric
//! bring-up is diffed against a host reference decoder, not QEMU):
//!
//! * **Stage 1 (here):** the [`bits`] H.264 bitstream reader (RBSP unescape,
//!   Exp-Golomb), the [`mp4`] ISO-BMFF demuxer (box tree → sample table +
//!   `avcC`), and [`h264`] NAL splitting + SPS/PPS parsing. [`probe`] reports a
//!   stream's geometry/profile/frame-count without decoding pixels.
//! * **Later stages:** the slice/CAVLC/intra/inter/transform/deblock pixel
//!   pipeline, wired through the same [`Frame`] output the player presents.
//!
//! Scope: H.264 baseline (I/P slices, CAVLC, 4:2:0), the common `.mp4`/`.mov`
//! case. Matroska/WebM/HLS containers and CABAC/High-profile tooling are future
//! stages; `probe` reports clearly when a file is outside the current support.

pub mod bits;
pub mod h264;
pub mod mkv;
pub mod mp4;

use alloc::string::String;
use alloc::vec::Vec;

/// A decoded frame: row-major `0x00RRGGBB` pixels, the same layout the
/// framebuffer's `present_surface` blits (so a video frame presents exactly
/// like an image). Produced by the pixel pipeline in a later stage.
pub struct Frame {
    pub w: usize,
    pub h: usize,
    pub pixels: Vec<u32>,
    /// Presentation timestamp in milliseconds from the start of the track.
    pub pts_ms: u64,
}

/// What [`probe`] reports about a video file — enough to drive the player UI and
/// tell the user exactly what is and isn't decodable yet.
pub struct VideoInfo {
    pub container: &'static str,
    pub codec: String,
    pub width: u32,
    pub height: u32,
    pub profile_idc: u8,
    pub level_idc: u8,
    pub frame_count: usize,
    pub duration_ms: u64,
    /// false = CAVLC (baseline, our target), true = CABAC (not yet decodable).
    pub cabac: bool,
    /// True once the pixel pipeline can decode this stream (later stage).
    pub decodable: bool,
}

/// Sniff the container and, for a supported one, demux + parse the parameter
/// sets to describe the stream. Does **not** decode pixels (Stage 1).
pub fn probe(bytes: &[u8]) -> Result<VideoInfo, &'static str> {
    let (container, avcc, width, height, frame_count, duration_ms) = demux_meta(bytes)?;
    let sps_nal = avcc.sps.first().ok_or("video: avcC has no SPS")?;
    let pps_nal = avcc.pps.first().ok_or("video: avcC has no PPS")?;
    let sps = h264::parse_sps(&bits::unescape_rbsp(&sps_nal[1..]))?;
    let pps = h264::parse_pps(&bits::unescape_rbsp(&pps_nal[1..]))?;
    let (width, height) = if sps.width() != 0 { (sps.width(), sps.height()) } else { (width, height) };
    let codec = alloc::format!("H.264 (avc, profile {}, level {}.{})", sps.profile_idc, sps.level_idc / 10, sps.level_idc % 10);
    Ok(VideoInfo {
        container,
        codec,
        width,
        height,
        profile_idc: sps.profile_idc,
        level_idc: sps.level_idc,
        frame_count,
        duration_ms,
        cabac: pps.entropy_coding_mode,
        decodable: !pps.entropy_coding_mode,
    })
}

/// A demuxed track's essentials, container-agnostic.
struct Demuxed {
    avcc: mp4::AvcC,
    samples: alloc::vec::Vec<mp4::Sample>,
    timescale: u32,
    duration_ms: u64,
}

fn demux(bytes: &[u8]) -> Result<(&'static str, Demuxed), &'static str> {
    match sniff(bytes) {
        Container::Mp4 => {
            let t = mp4::parse(bytes)?;
            let duration_ms = t.duration_ms();
            Ok(("mp4/mov (ISO-BMFF)", Demuxed { avcc: t.avcc, samples: t.samples, timescale: t.timescale, duration_ms }))
        }
        Container::Matroska => {
            let t = mkv::parse(bytes)?;
            let duration_ms = t.duration_ms();
            Ok(("matroska/webm (EBML)", Demuxed { avcc: t.avcc, samples: t.samples, timescale: t.timescale, duration_ms }))
        }
        Container::Unknown => Err("video: unrecognised container (mp4/mov, mkv/webm supported)"),
    }
}

fn demux_meta(bytes: &[u8]) -> Result<(&'static str, mp4::AvcC, u32, u32, usize, u64), &'static str> {
    let (container, d) = demux(bytes)?;
    let n = d.samples.len();
    Ok((container, d.avcc, 0, 0, n, d.duration_ms))
}

enum Container {
    Mp4,
    Matroska,
    Unknown,
}

fn sniff(bytes: &[u8]) -> Container {
    // ISO-BMFF: a top-level `ftyp` (or `moov`/`mdat`) box — the 4 bytes at
    // offset 4 are the type.
    if bytes.len() >= 12 {
        let typ = &bytes[4..8];
        if typ == b"ftyp" || typ == b"moov" || typ == b"mdat" || typ == b"free" || typ == b"skip" {
            return Container::Mp4;
        }
    }
    // Matroska/WebM: the EBML magic 0x1A45DFA3.
    if bytes.starts_with(&[0x1a, 0x45, 0xdf, 0xa3]) {
        return Container::Matroska;
    }
    Container::Unknown
}

/// BT.601 limited-range YUV → packed `0x00RRGGBB` (integer approximation, the
/// standard SD coefficients).
fn yuv_to_rgb(y: u8, u: u8, v: u8) -> u32 {
    let c = y as i32 - 16;
    let d = u as i32 - 128;
    let e = v as i32 - 128;
    let clip = |x: i32| x.clamp(0, 255) as u32;
    let r = clip((298 * c + 409 * e + 128) >> 8);
    let g = clip((298 * c - 100 * d - 208 * e + 128) >> 8);
    let b = clip((298 * c + 516 * d + 128) >> 8);
    (r << 16) | (g << 8) | b
}

fn frame_from_yuv(df: &h264::decoder::DecodedFrame, pts_ms: u64) -> Frame {
    let (w, h, cw) = (df.w, df.h, df.w / 2);
    let mut pixels = alloc::vec![0u32; w * h];
    for y in 0..h {
        for x in 0..w {
            let yy = df.y[y * w + x];
            let uu = df.cb[(y / 2) * cw + x / 2];
            let vv = df.cr[(y / 2) * cw + x / 2];
            pixels[y * w + x] = yuv_to_rgb(yy, uu, vv);
        }
    }
    Frame { w, h, pixels, pts_ms }
}

/// A **streaming** H.264 decoder: holds the source bytes + the demuxed sample
/// table and decodes frames on demand, keeping only *one* reference frame and
/// *one* presented RGB frame in RAM (~6 MB total for a 480p clip).
///
/// The alternative — decoding every frame up front into a `Vec<Frame>` — is a
/// trap: a 1300-frame 480×272 clip is ~700 MB of RGB, which overruns the
/// kernel's first-fit heap, corrupts a reference frame's chroma plane under
/// allocation pressure, and every dependent P-frame then builds on zeroed
/// chroma → whole frames render **green**. Baseline H.264 has no B-frames, so
/// decode order == display order and each P-frame references only the previous
/// frame; forward playback needs just the running reference.
pub struct StreamDecoder {
    bytes: Vec<u8>,
    sps: h264::Sps,
    pps: h264::Pps,
    length_size: u8,
    samples: Vec<mp4::Sample>,
    timescale: u32,
    /// Track duration in ms (public: the player's clock reads it).
    pub duration_ms: u64,
    /// Index of the next sample to decode (so `reference` holds frame `next-1`).
    next: usize,
    reference: Option<h264::decoder::DecodedFrame>,
    /// The last RGB frame produced, tagged with its display index.
    cur: Option<(usize, Frame)>,
}

impl StreamDecoder {
    /// Demux the container and parse SPS/PPS, ready to decode on demand.
    pub fn open(bytes: Vec<u8>) -> Result<StreamDecoder, &'static str> {
        let (_container, d) = demux(&bytes)?;
        let sps_nal = d.avcc.sps.first().ok_or("video: no SPS")?;
        let pps_nal = d.avcc.pps.first().ok_or("video: no PPS")?;
        let sps = h264::parse_sps(&bits::unescape_rbsp(&sps_nal[1..]))?;
        let pps = h264::parse_pps(&bits::unescape_rbsp(&pps_nal[1..]))?;
        if pps.entropy_coding_mode {
            return Err("video: CABAC not supported yet (baseline/CAVLC only)");
        }
        if d.samples.is_empty() {
            return Err("video: no decodable frames found");
        }
        Ok(StreamDecoder {
            length_size: d.avcc.length_size,
            samples: d.samples,
            timescale: d.timescale,
            duration_ms: d.duration_ms,
            bytes,
            sps,
            pps,
            next: 0,
            reference: None,
            cur: None,
        })
    }

    /// Total number of frames (samples) in the track.
    pub fn frame_count(&self) -> usize {
        self.samples.len()
    }

    /// Presentation timestamp of display frame `idx`, in ms — read from the
    /// sample table without decoding, so the playback clock can seek freely.
    pub fn pts_ms(&self, idx: usize) -> u64 {
        self.samples
            .get(idx)
            .map(|s| if self.timescale > 0 { s.dts * 1000 / self.timescale as u64 } else { 0 })
            .unwrap_or(0)
    }

    /// Decode sample `self.next` into `reference`, always advancing `next` (even
    /// on a per-sample error) so the caller's decode loop can't spin forever.
    fn decode_one(&mut self) -> Result<(), &'static str> {
        let s = self.samples[self.next];
        self.next += 1;
        if s.offset + s.size > self.bytes.len() {
            return Err("video: sample out of range");
        }
        let data = &self.bytes[s.offset..s.offset + s.size];
        // One mp4/mkv sample = one access unit = a full frame, possibly split
        // into multiple slice NALs. Collect them all and decode as one frame.
        let mut slices: Vec<(alloc::vec::Vec<u8>, bool)> = Vec::new();
        for nal in h264::split_avcc(data, self.length_size) {
            if nal.kind.is_slice() {
                slices.push((nal.rbsp(), nal.kind == h264::NalType::SliceIdr));
            }
        }
        if slices.is_empty() {
            return Err("video: sample has no slices");
        }
        let df = h264::decoder::decode_access_unit(&self.sps, &self.pps, &slices, self.reference.as_ref())?;
        self.reference = Some(df); // previous frame is the reference for the next P slice
        Ok(())
    }

    /// Ensure the current RGB frame is display index `idx`, decoding forward as
    /// needed. A backward seek rewinds to the latest sync (IDR) sample ≤ `idx`
    /// and re-decodes forward (baseline P-frames can't be decoded in reverse).
    /// Returns `true` if the presented frame changed.
    pub fn seek_decode(&mut self, idx: usize) -> bool {
        let idx = idx.min(self.samples.len().saturating_sub(1));
        if matches!(&self.cur, Some((ci, _)) if *ci == idx) {
            return false;
        }
        // Behind what we've decoded? Rewind to the latest keyframe ≤ idx.
        if idx + 1 < self.next {
            let mut s = idx;
            while s > 0 && !self.samples[s].is_sync {
                s -= 1;
            }
            self.next = s;
            self.reference = None;
        }
        // Decode forward until `reference` holds frame `idx` (next == idx+1).
        while self.next <= idx {
            if self.decode_one().is_err() {
                break; // decode_one advanced `next`, so this terminates
            }
        }
        let pts = self.pts_ms(idx);
        if let Some(df) = self.reference.as_ref() {
            self.cur = Some((idx, frame_from_yuv(df, pts)));
            true
        } else {
            false
        }
    }

    /// The currently presented RGB frame, if any.
    pub fn cur_frame(&self) -> Option<&Frame> {
        self.cur.as_ref().map(|(_, f)| f)
    }
}

/// What [`audio_info`] reports about a file's audio track.
pub struct AudioInfo {
    pub codec: &'static str,
    pub sample_rate: u32,
    pub channels: u8,
    /// True once the audio pipeline can actually decode + play this track.
    /// (AAC decode is a separate, in-progress stage — the track is demuxed and
    /// described here, but not yet turned into PCM.)
    pub decodable: bool,
}

/// Describe a file's audio track, if it has a demuxable one. Currently reports
/// the AAC track from an mp4/mov (the `esds` `AudioSpecificConfig`); the decode
/// path (AAC-LC → PCM) is a later stage, so `decodable` is false for now.
pub fn audio_info(bytes: &[u8]) -> Option<AudioInfo> {
    match sniff(bytes) {
        Container::Mp4 => match mp4::parse_audio(bytes) {
            Ok(Some(t)) => Some(AudioInfo { codec: "AAC (mp4a)", sample_rate: t.sample_rate, channels: t.channels, decodable: false }),
            _ => None,
        },
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn sniff_recognises_containers() {
        let mut mp4 = alloc::vec![0u8; 12];
        mp4[4..8].copy_from_slice(b"ftyp");
        assert!(matches!(sniff(&mp4), Container::Mp4));
        assert!(matches!(sniff(&[0x1a, 0x45, 0xdf, 0xa3, 0, 0]), Container::Matroska));
        assert!(matches!(sniff(b"not a video"), Container::Unknown));
    }

    #[test_case]
    fn probe_rejects_unknown_container() {
        assert!(probe(b"random bytes here").is_err());
    }
}

#[cfg(test)]
mod decode_fixture_test {
    //! Full-decode regression: an embedded baseline keyframe (x264 → mp4)
    //! whose YUV hash was captured from PyAV/ffmpeg. Guards the Rust port.
    use super::*;
    // c_hq 64x64 baseline, muxed from x264; expected hash 2213512446
    const CLIP_MP4: [u8; 2139] = [
    0, 0, 0, 32, 102, 116, 121, 112, 105, 115, 111, 109, 0, 0, 2, 0, 105, 115, 111, 109, 105, 115, 111, 50,
    97, 118, 99, 49, 109, 112, 52, 49, 0, 0, 2, 113, 109, 111, 111, 118, 0, 0, 0, 108, 109, 118, 104, 100,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 1, 0, 1, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 2, 0, 0, 1, 253, 116, 114, 97, 107, 0, 0, 0, 92, 116, 107, 104, 100, 0, 0, 0, 7,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0,
    0, 64, 0, 0, 0, 64, 0, 0, 0, 0, 1, 153, 109, 100, 105, 97, 0, 0, 0, 28, 109, 100, 104, 100,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 0, 1, 0, 0, 0, 0, 0, 39,
    104, 100, 108, 114, 0, 0, 0, 0, 0, 0, 0, 0, 118, 105, 100, 101, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 99, 104, 105, 116, 116, 105, 0, 0, 0, 1, 78, 109, 105, 110, 102, 0, 0, 0, 20, 118,
    109, 104, 100, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 36, 100, 105, 110, 102, 0,
    0, 0, 28, 100, 114, 101, 102, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 12, 117, 114, 108, 32, 0,
    0, 0, 1, 0, 0, 1, 14, 115, 116, 98, 108, 0, 0, 0, 146, 115, 116, 115, 100, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 130, 97, 118, 99, 49, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 64, 0, 72, 0, 0, 0, 72, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 255, 255, 0, 0, 0, 44, 97, 118, 99,
    67, 1, 66, 192, 10, 255, 225, 0, 20, 103, 66, 192, 10, 220, 66, 104, 64, 0, 0, 3, 0, 64, 0, 0,
    12, 163, 196, 137, 224, 1, 0, 5, 104, 206, 1, 12, 178, 0, 0, 0, 24, 115, 116, 116, 115, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 28, 115, 116, 115, 99, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 24, 115, 116, 115,
    122, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 5, 194, 0, 0, 0, 20, 115, 116, 115,
    115, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 20, 115, 116, 99, 111, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 2, 153, 0, 0, 5, 202, 109, 100, 97, 116, 0, 0, 5, 190, 101, 136, 132,
    58, 196, 162, 37, 34, 33, 203, 192, 59, 128, 1, 135, 0, 1, 34, 154, 202, 224, 0, 84, 0, 134, 50, 0,
    32, 35, 42, 160, 0, 84, 0, 67, 25, 0, 2, 96, 6, 64, 84, 0, 9, 128, 25, 3, 32, 2, 2, 50,
    53, 0, 2, 160, 2, 24, 84, 0, 64, 70, 76, 128, 8, 8, 200, 168, 0, 128, 140, 141, 64, 0, 168, 0,
    134, 21, 0, 16, 17, 149, 80, 0, 42, 0, 33, 140, 128, 8, 8, 202, 168, 0, 21, 0, 16, 193, 0, 0,
    65, 68, 0, 5, 134, 98, 0, 0, 130, 136, 0, 11, 12, 160, 116, 0, 13, 129, 208, 0, 54, 7, 64, 0,
    216, 29, 0, 3, 96, 29, 0, 3, 50, 58, 0, 6, 100, 116, 0, 12, 200, 232, 0, 25, 144, 186, 135, 0,
    1, 6, 176, 0, 16, 219, 0, 1, 29, 132, 7, 128, 1, 54, 15, 0, 2, 108, 62, 0, 4, 216, 60, 0,
    9, 176, 120, 0, 19, 96, 240, 0, 38, 193, 224, 0, 77, 131, 192, 0, 155, 15, 128, 1, 54, 15, 0, 2,
    108, 62, 0, 4, 216, 60, 0, 9, 176, 120, 0, 19, 96, 240, 0, 38, 193, 224, 0, 77, 131, 192, 0, 155,
    2, 0, 0, 130, 136, 0, 8, 97, 81, 208, 0, 54, 7, 64, 0, 216, 29, 0, 3, 96, 116, 0, 13, 143,
    133, 212, 56, 0, 8, 53, 128, 0, 134, 216, 0, 8, 236, 32, 60, 0, 9, 176, 120, 0, 19, 96, 240, 0,
    38, 193, 224, 0, 77, 131, 192, 0, 155, 7, 128, 1, 54, 15, 0, 2, 108, 30, 0, 4, 216, 60, 0, 9,
    176, 120, 0, 19, 96, 240, 0, 38, 193, 224, 0, 77, 131, 192, 0, 155, 7, 128, 1, 54, 15, 0, 2, 108,
    30, 0, 4, 216, 16, 0, 4, 20, 64, 0, 67, 10, 142, 128, 1, 176, 58, 0, 6, 192, 232, 0, 27, 3,
    160, 0, 108, 120, 93, 64, 7, 0, 1, 5, 48, 0, 16, 54, 0, 2, 128, 76, 102, 230, 228, 228, 229, 93,
    205, 205, 205, 200, 194, 2, 225, 224, 0, 77, 131, 192, 0, 155, 7, 128, 1, 54, 15, 0, 2, 108, 30, 0,
    4, 216, 60, 0, 9, 176, 120, 0, 19, 96, 240, 0, 38, 193, 224, 0, 77, 131, 192, 0, 155, 7, 128, 1,
    54, 15, 0, 2, 108, 30, 0, 4, 216, 60, 0, 9, 176, 120, 0, 19, 96, 0, 144, 0, 16, 77, 0, 2,
    128, 1, 132, 193, 192, 195, 207, 151, 88, 43, 131, 234, 232, 10, 65, 64, 64, 0, 16, 81, 0, 1, 12, 42,
    58, 0, 6, 192, 232, 0, 27, 3, 160, 0, 108, 14, 128, 1, 177, 225, 103, 7, 0, 1, 12, 120, 0, 2,
    50, 160, 0, 38, 205, 9, 3, 176, 0, 29, 168, 45, 0, 3, 50, 61, 128, 0, 237, 64, 232, 0, 25, 144,
    118, 0, 3, 181, 5, 160, 0, 102, 65, 216, 0, 14, 212, 14, 128, 1, 153, 30, 192, 0, 118, 160, 116, 0,
    12, 200, 246, 0, 3, 181, 3, 160, 0, 102, 65, 216, 0, 14, 212, 14, 128, 1, 153, 7, 96, 0, 59, 80,
    58, 0, 6, 101, 16, 0, 4, 20, 64, 0, 67, 9, 124, 116, 0, 12, 200, 232, 0, 25, 145, 208, 0, 51,
    35, 160, 0, 102, 74, 79, 34, 33, 34, 35, 34, 55, 184, 200, 224, 208, 1, 1, 25, 21, 0, 16, 17, 145,
    80, 1, 1, 25, 26, 128, 1, 80, 1, 12, 42, 0, 32, 35, 34, 160, 2, 2, 50, 100, 0, 64, 70, 64,
    3, 128, 0, 128, 24, 0, 16, 4, 5, 206, 73, 218, 13, 29, 58, 48, 144, 88, 86, 150, 163, 149, 0, 16,
    17, 147, 32, 2, 2, 50, 0, 56, 0, 8, 1, 0, 1, 64, 17, 0, 54, 48, 55, 125, 138, 36, 159, 196,
    226, 163, 45, 251, 64, 0, 192, 0, 66, 4, 0, 128, 13, 181, 80, 24, 77, 84, 52, 20, 10, 52, 2, 30,
    254, 0, 28, 0, 4, 21, 64, 0, 160, 0, 101, 192, 72, 22, 44, 145, 79, 247, 76, 77, 53, 65, 8, 188,
    8, 62, 0, 0, 128, 104, 8, 0, 144, 103, 64, 155, 177, 58, 122, 93, 94, 87, 28, 100, 177, 147, 0, 2,
    96, 6, 64, 76, 0, 9, 128, 25, 4, 141, 201, 196, 166, 68, 57, 72, 229, 64, 4, 4, 100, 0, 112, 0,
    16, 2, 0, 2, 128, 34, 0, 108, 96, 110, 251, 20, 73, 63, 137, 197, 70, 91, 247, 2, 0, 1, 9, 224,
    0, 45, 147, 136, 34, 12, 148, 154, 49, 83, 230, 2, 239, 252, 131, 128, 1, 135, 0, 1, 23, 186, 205, 0,
    3, 0, 1, 8, 16, 2, 0, 54, 213, 64, 97, 53, 80, 208, 80, 40, 208, 8, 123, 248, 152, 0, 19, 0,
    50, 2, 96, 2, 6, 100, 106, 0, 5, 64, 4, 48, 228, 0, 64, 103, 137, 144, 4, 4, 100, 200, 0, 128,
    140, 141, 64, 0, 168, 0, 134, 21, 0, 16, 17, 149, 80, 0, 42, 0, 33, 140, 128, 8, 8, 202, 168, 0,
    21, 0, 16, 201, 17, 9, 17, 9, 17, 28, 220, 158, 200, 0, 128, 140, 138, 128, 8, 8, 200, 168, 0, 128,
    140, 141, 64, 0, 168, 0, 134, 50, 0, 32, 35, 34, 160, 2, 2, 50, 100, 0, 64, 70, 70, 160, 0, 84,
    0, 67, 10, 128, 8, 8, 201, 144, 1, 1, 25, 21, 0, 16, 17, 144, 0, 224, 0, 32, 6, 0, 4, 1,
    1, 115, 146, 118, 131, 71, 78, 140, 36, 22, 21, 165, 168, 229, 64, 4, 4, 100, 4, 128, 0, 128, 16, 0,
    20, 1, 16, 3, 99, 3, 119, 216, 162, 73, 252, 78, 42, 50, 223, 184, 16, 0, 8, 79, 0, 1, 108, 156,
    65, 16, 100, 164, 209, 138, 159, 48, 23, 127, 228, 28, 0, 12, 56, 0, 8, 189, 214, 34, 0, 0, 130, 136,
    0, 8, 97, 47, 29, 0, 3, 50, 58, 0, 6, 100, 116, 0, 12, 200, 232, 0, 25, 153, 17, 25, 187, 220,
    74, 66, 82, 57, 80, 1, 1, 25, 21, 0, 16, 17, 147, 32, 2, 2, 50, 0, 28, 0, 4, 18, 192, 0,
    160, 0, 97, 48, 120, 48, 251, 229, 214, 26, 176, 218, 194, 7, 128, 144, 15, 160, 3, 128, 0, 128, 24, 0,
    16, 4, 5, 206, 73, 218, 13, 29, 58, 48, 144, 88, 86, 150, 163, 240, 32, 0, 16, 158, 0, 2, 217, 56,
    130, 32, 201, 73, 163, 21, 62, 96, 46, 255, 222, 0, 0, 128, 104, 24, 1, 32, 206, 129, 55, 98, 132, 212,
    154, 188, 174, 57, 201, 97, 127, 132, 128, 202, 27, 179, 64, 0, 192, 0, 66, 4, 0, 128, 13, 181, 80, 24,
    77, 84, 52, 20, 10, 52, 2, 30, 254, 38, 0, 4, 192, 12, 128, 152, 0, 129, 153, 26, 128, 1, 80, 1,
    12, 38, 0, 129, 153, 86, 0, 2, 160, 4, 173, 45, 0, 64, 207, 19, 32, 0, 76, 0, 200, 36, 68, 36,
    68, 36, 68, 57, 72, 232, 56, 0, 24, 112, 0, 17, 123, 172, 38, 0, 32, 102, 76, 128, 32, 35, 35, 80,
    0, 42, 0, 33, 133, 64, 4, 4, 100, 84, 0, 64, 70, 76, 128, 8, 8, 200, 212, 0, 10, 128, 8, 99,
    32, 2, 2, 50, 100, 0, 64, 70, 69, 64, 4, 4, 100, 106, 0, 5, 64, 4, 48, 168, 0, 128, 140, 170,
    128, 1, 80, 1, 12, 100, 0, 64, 70, 85, 64, 0, 168, 0, 134, 20, 158, 79, 19, 34, 18, 34, 28, 175,
    0, 28, 0, 4, 21, 64, 0, 160, 0, 101, 192, 72, 22, 44, 145, 79, 247, 76, 77, 53, 65, 8, 188, 8,
    62, 0, 0, 128, 104, 8, 0, 144, 103, 64, 155, 177, 58, 122, 93, 94, 87, 28, 100, 177, 160, 224, 0, 97,
    192, 0, 66, 214, 177, 62, 0, 5, 64, 8, 97, 200, 0, 128, 207, 19, 32, 8, 8, 201, 144, 1, 1, 25,
    26, 128, 1, 80, 1, 12, 100, 0, 64, 70, 76, 128, 8, 8, 200, 168, 0, 128, 140, 141, 64, 0, 168, 0,
    134, 21, 0, 16, 17, 149, 80, 0, 42, 0, 33, 140, 128, 8, 8, 202, 168, 0, 21, 0, 16, 196, 64, 0,
    16, 81, 0, 1, 12, 37, 227, 160, 0, 102, 71, 64, 0, 204, 142, 128, 1, 153, 29, 0, 3, 50, 82, 114,
    147, 148, 156,
];
    const EXPECT_HASH: u32 = 2213512446;
    const EXPECT_W: usize = 64;
    const EXPECT_H: usize = 64;

    #[test_case]
    fn decodes_embedded_keyframe_matching_pyav() {
        let track = mp4::parse(&CLIP_MP4).unwrap();
        let sps = h264::parse_sps(&bits::unescape_rbsp(&track.avcc.sps[0][1..])).unwrap();
        let pps = h264::parse_pps(&bits::unescape_rbsp(&track.avcc.pps[0][1..])).unwrap();
        assert_eq!((sps.width() as usize, sps.height() as usize), (EXPECT_W, EXPECT_H));
        let s = track.samples.iter().find(|s| s.is_sync).expect("sync sample");
        let data = &CLIP_MP4[s.offset..s.offset + s.size];
        let nal = h264::split_avcc(data, track.avcc.length_size)
            .into_iter()
            .find(|n| n.kind == h264::NalType::SliceIdr)
            .expect("IDR nal");
        let df = h264::decoder::decode_islice(&sps, &pps, &nal.rbsp(), true).unwrap();
        assert_eq!((df.w, df.h), (EXPECT_W, EXPECT_H));
        let mut hh: u32 = 0;
        for &b in df.y.iter().chain(df.cb.iter()).chain(df.cr.iter()) {
            hh = hh.wrapping_mul(31).wrapping_add(b as u32);
        }
        assert_eq!(hh, EXPECT_HASH, "decoded keyframe must match the PyAV reference");
    }
    const PCLIP_MP4: [u8; 3331] = [
    0, 0, 0, 32, 102, 116, 121, 112, 105, 115, 111, 109, 0, 0, 2, 0, 105, 115, 111, 109, 105, 115, 111, 50,
    97, 118, 99, 49, 109, 112, 52, 49, 0, 0, 2, 126, 109, 111, 111, 118, 0, 0, 0, 108, 109, 118, 104, 100,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 4, 0, 1, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 2, 0, 0, 2, 10, 116, 114, 97, 107, 0, 0, 0, 92, 116, 107, 104, 100, 0, 0, 0, 7,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0,
    0, 96, 0, 0, 0, 64, 0, 0, 0, 0, 1, 166, 109, 100, 105, 97, 0, 0, 0, 28, 109, 100, 104, 100,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 0, 4, 0, 0, 0, 0, 0, 39,
    104, 100, 108, 114, 0, 0, 0, 0, 0, 0, 0, 0, 118, 105, 100, 101, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 99, 104, 105, 116, 116, 105, 0, 0, 0, 1, 91, 109, 105, 110, 102, 0, 0, 0, 20, 118,
    109, 104, 100, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 36, 100, 105, 110, 102, 0,
    0, 0, 28, 100, 114, 101, 102, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 12, 117, 114, 108, 32, 0,
    0, 0, 1, 0, 0, 1, 27, 115, 116, 98, 108, 0, 0, 0, 147, 115, 116, 115, 100, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 131, 97, 118, 99, 49, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 96, 0, 64, 0, 72, 0, 0, 0, 72, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 255, 255, 0, 0, 0, 45, 97, 118, 99,
    67, 1, 66, 192, 10, 255, 225, 0, 21, 103, 66, 192, 10, 218, 24, 154, 16, 0, 0, 3, 0, 16, 0, 0,
    3, 3, 40, 241, 34, 106, 1, 0, 5, 104, 206, 3, 178, 200, 0, 0, 0, 24, 115, 116, 116, 115, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 28, 115, 116, 115, 99, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 36, 115, 116,
    115, 122, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 7, 104, 0, 0, 1, 15, 0, 0,
    0, 240, 0, 0, 0, 246, 0, 0, 0, 20, 115, 116, 115, 115, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
    0, 1, 0, 0, 0, 20, 115, 116, 99, 111, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 166, 0, 0,
    10, 101, 109, 100, 97, 116, 0, 0, 7, 100, 101, 136, 132, 58, 196, 162, 37, 34, 33, 203, 192, 59, 128, 12,
    80, 0, 17, 153, 172, 100, 0, 64, 34, 6, 64, 65, 153, 11, 128, 8, 6, 64, 168, 0, 128, 68, 5, 64,
    4, 2, 32, 100, 4, 25, 144, 184, 0, 128, 100, 10, 128, 131, 50, 100, 4, 25, 145, 80, 16, 102, 66, 224,
    2, 1, 144, 42, 2, 12, 201, 120, 0, 128, 100, 25, 1, 6, 100, 188, 0, 64, 50, 2, 0, 1, 112, 0,
    21, 107, 136, 0, 2, 4, 160, 0, 42, 54, 133, 24, 163, 20, 98, 140, 72, 0, 19, 50, 120, 0, 19, 50,
    120, 0, 19, 50, 120, 0, 19, 50, 23, 80, 224, 0, 32, 154, 0, 2, 19, 224, 0, 34, 176, 128, 240, 1,
    96, 240, 1, 96, 240, 1, 96, 240, 1, 96, 240, 1, 96, 240, 1, 96, 240, 1, 96, 240, 1, 96, 240, 1,
    96, 240, 1, 96, 240, 1, 96, 240, 1, 96, 240, 1, 96, 240, 1, 96, 240, 1, 96, 240, 1, 96, 64, 0,
    46, 0, 26, 138, 49, 70, 40, 197, 31, 194, 234, 28, 0, 4, 19, 64, 0, 66, 124, 0, 4, 86, 16, 30,
    0, 44, 30, 0, 44, 30, 0, 44, 30, 0, 44, 30, 0, 44, 30, 0, 44, 30, 0, 44, 30, 0, 44, 30,
    0, 44, 30, 0, 44, 30, 0, 44, 30, 0, 44, 30, 0, 44, 30, 0, 44, 30, 0, 44, 30, 0, 44, 8,
    0, 5, 192, 3, 81, 70, 40, 197, 24, 163, 240, 186, 128, 14, 0, 2, 7, 96, 0, 32, 60, 0, 40, 27,
    59, 59, 58, 58, 58, 196, 236, 236, 236, 232, 126, 14, 135, 128, 11, 7, 128, 11, 7, 128, 11, 7, 128, 11,
    7, 128, 11, 7, 128, 11, 7, 128, 11, 7, 128, 11, 7, 128, 11, 7, 128, 11, 7, 128, 11, 7, 128, 11,
    7, 128, 11, 7, 128, 11, 7, 128, 11, 0, 4, 128, 0, 129, 168, 0, 160, 2, 27, 12, 12, 54, 237, 89,
    130, 144, 35, 164, 131, 80, 164, 32, 0, 23, 0, 13, 69, 24, 163, 20, 98, 143, 226, 37, 17, 55, 55, 121,
    9, 86, 84, 0, 64, 34, 2, 160, 2, 1, 16, 21, 1, 6, 100, 46, 0, 32, 25, 2, 160, 2, 1, 16,
    21, 0, 16, 8, 128, 9, 0, 8, 0, 80, 17, 0, 76, 216, 38, 223, 12, 186, 85, 48, 57, 39, 223, 180,
    0, 96, 0, 32, 174, 3, 128, 234, 191, 133, 210, 208, 8, 142, 13, 243, 132, 221, 252, 0, 56, 0, 96, 2,
    3, 5, 201, 91, 168, 56, 56, 80, 90, 37, 77, 70, 70, 139, 194, 0, 1, 8, 160, 0, 16, 31, 44, 43,
    157, 10, 185, 34, 75, 254, 214, 28, 127, 239, 0, 0, 64, 12, 16, 4, 133, 112, 20, 106, 110, 151, 230, 142,
    106, 139, 49, 210, 218, 202, 13, 198, 137, 7, 0, 24, 160, 0, 34, 35, 88, 118, 0, 32, 25, 6, 64, 65,
    153, 47, 0, 16, 12, 128, 128, 0, 92, 0, 53, 20, 98, 140, 81, 138, 63, 221, 226, 37, 41, 14, 85, 128,
    14, 0, 2, 6, 160, 2, 128, 8, 108, 48, 48, 227, 181, 102, 10, 64, 142, 146, 10, 66, 144, 38, 248, 0,
    2, 0, 96, 192, 36, 43, 128, 163, 83, 116, 191, 52, 115, 84, 89, 142, 150, 214, 80, 220, 73, 141, 116, 27,
    137, 18, 124, 0, 16, 9, 90, 100, 0, 64, 34, 6, 64, 65, 153, 11, 128, 8, 6, 65, 48, 3, 65, 169,
    47, 1, 6, 157, 50, 2, 8, 201, 120, 0, 128, 72, 67, 32, 32, 204, 151, 128, 8, 6, 65, 144, 16, 102,
    75, 192, 4, 3, 32, 32, 0, 23, 0, 13, 69, 24, 163, 20, 98, 143, 194, 206, 14, 0, 2, 18, 176, 0,
    4, 76, 64, 0, 74, 74, 18, 14, 128, 14, 200, 90, 0, 51, 33, 208, 1, 217, 7, 64, 6, 100, 58, 0,
    59, 33, 104, 0, 204, 135, 64, 7, 100, 29, 0, 25, 144, 232, 0, 236, 131, 160, 3, 50, 29, 0, 29, 144,
    116, 0, 102, 67, 160, 3, 178, 14, 128, 12, 200, 116, 0, 118, 65, 208, 1, 153, 68, 0, 1, 2, 80, 0,
    16, 96, 94, 120, 0, 19, 50, 120, 0, 19, 50, 120, 0, 19, 50, 120, 0, 19, 50, 82, 121, 17, 9, 17,
    25, 17, 189, 198, 71, 6, 128, 131, 50, 42, 2, 12, 200, 168, 8, 51, 33, 112, 1, 0, 200, 21, 1, 6,
    100, 84, 4, 25, 147, 32, 32, 204, 128, 7, 0, 12, 0, 64, 96, 185, 43, 117, 7, 7, 10, 11, 68, 169,
    168, 200, 209, 74, 128, 131, 50, 100, 4, 25, 144, 1, 192, 2, 0, 20, 4, 64, 19, 54, 9, 183, 195, 46,
    149, 76, 14, 73, 247, 237, 0, 24, 0, 8, 43, 128, 224, 58, 175, 225, 116, 180, 2, 35, 131, 124, 225, 55,
    127, 0, 14, 0, 2, 7, 96, 2, 128, 8, 168, 52, 19, 34, 178, 148, 234, 191, 41, 209, 30, 158, 135, 191,
    0, 0, 64, 12, 16, 15, 10, 224, 40, 212, 225, 39, 205, 28, 213, 150, 99, 165, 167, 128, 1, 0, 134, 29,
    128, 8, 6, 132, 72, 220, 156, 74, 100, 67, 148, 142, 84, 4, 25, 144, 1, 192, 2, 0, 20, 4, 64, 19,
    54, 9, 183, 195, 46, 149, 76, 14, 73, 247, 238, 16, 0, 8, 69, 0, 0, 128, 249, 97, 92, 232, 85, 201,
    18, 95, 246, 176, 227, 255, 32, 224, 3, 20, 0, 4, 68, 107, 52, 0, 96, 0, 32, 174, 3, 128, 234, 191,
    133, 210, 208, 8, 142, 13, 243, 132, 221, 252, 158, 0, 4, 2, 24, 76, 4, 17, 146, 240, 1, 0, 200, 19,
    1, 4, 100, 200, 8, 51, 38, 64, 65, 153, 11, 128, 8, 6, 64, 168, 8, 51, 37, 224, 2, 1, 144, 100,
    4, 25, 146, 240, 1, 0, 200, 20, 144, 202, 79, 34, 33, 34, 33, 34, 35, 155, 147, 202, 128, 131, 50, 42,
    2, 12, 200, 168, 8, 51, 33, 112, 1, 0, 200, 21, 1, 6, 100, 84, 4, 25, 147, 32, 32, 204, 133, 192,
    4, 3, 32, 84, 4, 25, 147, 32, 32, 204, 138, 128, 131, 50, 0, 28, 0, 48, 1, 1, 130, 228, 173, 212,
    28, 28, 40, 45, 18, 166, 163, 35, 69, 42, 2, 12, 200, 9, 0, 8, 0, 80, 17, 0, 76, 216, 38, 223,
    12, 186, 85, 48, 57, 39, 223, 184, 64, 0, 33, 20, 0, 2, 3, 229, 133, 115, 161, 87, 36, 73, 127, 218,
    195, 143, 252, 131, 128, 12, 80, 0, 17, 17, 172, 68, 0, 1, 2, 80, 0, 16, 96, 94, 120, 0, 19, 50,
    120, 0, 19, 50, 120, 0, 19, 50, 120, 0, 19, 51, 34, 35, 55, 123, 137, 72, 74, 71, 42, 2, 12, 200,
    168, 8, 51, 38, 64, 65, 153, 0, 14, 0, 2, 6, 96, 2, 128, 8, 108, 52, 48, 223, 181, 102, 10, 17,
    229, 52, 21, 132, 224, 214, 0, 56, 0, 96, 2, 3, 5, 201, 91, 168, 56, 56, 80, 90, 37, 77, 70, 70,
    139, 194, 0, 1, 8, 160, 0, 16, 31, 44, 43, 157, 10, 185, 34, 75, 254, 214, 28, 127, 239, 0, 0, 64,
    12, 24, 4, 133, 112, 20, 106, 110, 151, 230, 142, 106, 138, 241, 210, 90, 232, 55, 26, 45, 0, 24, 0, 8,
    43, 128, 224, 58, 175, 225, 116, 180, 2, 35, 131, 124, 225, 55, 127, 39, 128, 1, 0, 134, 19, 1, 4, 100,
    188, 0, 64, 50, 7, 64, 32, 145, 38, 64, 4, 2, 32, 100, 4, 25, 144, 184, 0, 128, 100, 2, 4, 160,
    0, 2, 18, 160, 0, 32, 98, 0, 2, 30, 160, 0, 216, 52, 172, 49, 192, 6, 45, 97, 142, 0, 49, 107,
    7, 184, 0, 197, 172, 30, 224, 3, 22, 176, 123, 128, 12, 90, 193, 238, 0, 49, 107, 7, 184, 0, 197, 172,
    30, 224, 3, 22, 176, 123, 128, 12, 90, 193, 238, 0, 49, 107, 7, 184, 0, 197, 172, 30, 224, 3, 22, 176,
    123, 128, 12, 90, 193, 238, 0, 49, 107, 7, 184, 0, 197, 172, 30, 224, 3, 22, 176, 164, 249, 73, 132, 73,
    0, 4, 0, 1, 3, 144, 0, 16, 28, 0, 16, 12, 156, 156, 156, 28, 28, 28, 28, 156, 156, 156, 28, 28,
    255, 255, 0, 11, 0, 1, 3, 80, 3, 128, 8, 108, 44, 48, 219, 189, 109, 32, 65, 69, 6, 33, 113, 255,
    73, 226, 100, 66, 68, 67, 149, 224, 3, 128, 0, 129, 216, 0, 160, 2, 42, 13, 4, 200, 172, 165, 58, 175,
    202, 116, 71, 167, 161, 239, 192, 0, 16, 3, 4, 3, 194, 184, 10, 53, 56, 73, 243, 71, 53, 101, 152, 233,
    104, 56, 0, 197, 0, 1, 7, 26, 195, 176, 1, 0, 200, 19, 1, 4, 100, 200, 8, 51, 38, 64, 65, 153,
    11, 128, 8, 6, 65, 144, 16, 102, 76, 128, 131, 50, 42, 2, 12, 200, 92, 0, 64, 50, 5, 64, 65, 153,
    47, 0, 16, 12, 131, 32, 32, 204, 151, 128, 8, 6, 65, 16, 0, 4, 9, 64, 0, 65, 129, 121, 224, 0,
    76, 201, 224, 0, 76, 201, 224, 0, 76, 201, 224, 0, 76, 201, 73, 202, 78, 82, 114, 34, 18, 34, 50, 35,
    123, 140, 142, 13, 1, 6, 100, 84, 4, 25, 145, 80, 16, 102, 66, 224, 2, 1, 144, 42, 2, 12, 200, 168,
    8, 51, 38, 64, 65, 153, 0, 14, 0, 24, 0, 128, 193, 114, 86, 234, 14, 14, 20, 22, 137, 83, 81, 145,
    162, 149, 1, 6, 100, 200, 8, 51, 32, 3, 128, 4, 0, 40, 8, 128, 38, 108, 19, 111, 134, 93, 42, 152,
    28, 147, 239, 218, 0, 48, 0, 16, 87, 1, 192, 117, 95, 194, 233, 104, 4, 71, 6, 249, 194, 110, 254, 0,
    28, 0, 4, 14, 192, 5, 0, 17, 80, 104, 38, 69, 101, 41, 213, 126, 83, 162, 61, 61, 15, 126, 0, 0,
    128, 24, 32, 30, 21, 192, 81, 169, 194, 79, 154, 57, 171, 44, 199, 75, 79, 0, 2, 1, 12, 59, 0, 16,
    13, 8, 145, 185, 56, 148, 200, 135, 41, 28, 168, 8, 51, 32, 3, 128, 4, 0, 40, 8, 128, 38, 108, 19,
    111, 134, 93, 42, 152, 28, 147, 239, 220, 32, 0, 16, 138, 0, 1, 1, 242, 194, 185, 208, 171, 146, 36, 191,
    237, 97, 199, 254, 65, 192, 6, 40, 0, 8, 136, 214, 1, 32, 12, 0, 4, 21, 192, 112, 29, 87, 240, 186,
    90, 1, 17, 193, 190, 112, 155, 191, 147, 192, 0, 128, 67, 9, 128, 130, 50, 94, 0, 32, 25, 2, 96, 32,
    140, 153, 1, 6, 100, 200, 8, 51, 33, 112, 1, 0, 200, 21, 1, 6, 100, 188, 0, 64, 50, 12, 128, 131,
    50, 94, 0, 32, 25, 8, 0, 0, 1, 11, 65, 154, 34, 184, 97, 71, 0, 2, 81, 192, 0, 239, 212, 112,
    0, 37, 28, 0, 14, 253, 71, 0, 2, 81, 192, 0, 239, 212, 112, 0, 37, 28, 0, 14, 253, 71, 0, 2,
    81, 192, 0, 239, 212, 112, 0, 37, 28, 0, 14, 253, 71, 0, 2, 81, 192, 0, 239, 212, 112, 0, 37, 28,
    0, 14, 253, 71, 0, 2, 81, 192, 0, 239, 212, 112, 0, 37, 28, 0, 14, 226, 137, 197, 19, 245, 28, 0,
    9, 71, 0, 3, 191, 81, 192, 0, 148, 112, 0, 59, 138, 39, 20, 78, 40, 159, 12, 40, 224, 0, 74, 56,
    0, 29, 248, 99, 131, 64, 1, 6, 100, 84, 0, 16, 102, 89, 80, 0, 65, 153, 21, 0, 4, 25, 144, 224,
    0, 74, 32, 128, 13, 126, 32, 29, 16, 14, 197, 19, 138, 39, 11, 18, 0, 8, 0, 2, 2, 32, 1, 0,
    129, 22, 44, 88, 48, 96, 193, 139, 22, 44, 24, 49, 255, 254, 0, 30, 1, 0, 0, 129, 8, 11, 2, 138,
    60, 135, 247, 131, 118, 49, 89, 227, 127, 120, 99, 130, 0, 48, 28, 60, 113, 209, 225, 224, 225, 145, 168, 44,
    133, 179, 118, 151, 128, 2, 9, 26, 202, 128, 2, 12, 200, 168, 0, 32, 140, 135, 0, 2, 81, 4, 0, 107,
    241, 0, 232, 128, 119, 195, 28, 168, 0, 32, 204, 138, 128, 2, 12, 203, 42, 0, 8, 51, 34, 160, 0, 131,
    50, 28, 0, 9, 68, 16, 1, 175, 196, 3, 162, 1, 216, 0, 0, 0, 236, 65, 154, 66, 184, 97, 71, 0,
    2, 81, 192, 0, 239, 212, 112, 0, 37, 28, 0, 14, 253, 71, 0, 2, 81, 192, 0, 239, 212, 112, 0, 37,
    28, 0, 14, 253, 71, 0, 2, 81, 192, 0, 239, 212, 112, 0, 37, 28, 0, 14, 253, 71, 0, 2, 81, 192,
    0, 239, 212, 112, 0, 37, 28, 0, 14, 253, 71, 0, 2, 81, 192, 0, 239, 212, 112, 0, 37, 28, 0, 14,
    226, 137, 197, 19, 245, 28, 0, 9, 71, 0, 3, 191, 81, 192, 0, 148, 112, 0, 59, 138, 39, 20, 78, 40,
    159, 12, 40, 224, 0, 74, 56, 0, 29, 214, 10, 3, 42, 32, 18, 33, 0, 0, 137, 212, 81, 56, 162, 114,
    248, 104, 71, 184, 99, 129, 182, 165, 132, 132, 242, 130, 139, 128, 160, 0, 32, 58, 0, 2, 0, 5, 129, 9,
    128, 208, 123, 127, 231, 51, 22, 245, 139, 165, 60, 110, 202, 88, 124, 104, 175, 203, 165, 222, 16, 12, 114, 160,
    0, 130, 50, 42, 0, 8, 25, 150, 84, 0, 16, 51, 34, 160, 0, 131, 50, 28, 0, 9, 68, 16, 1, 175,
    196, 3, 162, 1, 223, 12, 114, 160, 0, 129, 153, 21, 0, 4, 12, 203, 42, 0, 8, 25, 145, 80, 0, 64,
    204, 135, 0, 2, 81, 4, 0, 107, 241, 0, 232, 128, 118, 0, 0, 0, 242, 65, 154, 98, 184, 97, 71, 0,
    2, 81, 192, 0, 239, 212, 112, 0, 37, 28, 0, 14, 253, 71, 0, 2, 81, 192, 0, 239, 212, 112, 0, 37,
    28, 0, 14, 253, 71, 0, 2, 81, 192, 0, 239, 212, 112, 0, 37, 28, 0, 14, 253, 71, 0, 2, 81, 192,
    0, 239, 212, 112, 0, 37, 28, 0, 14, 253, 71, 0, 2, 81, 192, 0, 239, 212, 112, 0, 37, 28, 0, 14,
    226, 137, 197, 19, 245, 28, 0, 9, 71, 0, 3, 191, 81, 192, 0, 148, 112, 0, 59, 138, 39, 20, 79, 134,
    20, 112, 0, 37, 28, 0, 14, 253, 71, 0, 2, 81, 192, 0, 238, 176, 80, 25, 81, 0, 145, 8, 0, 4,
    78, 162, 137, 197, 19, 176, 224, 140, 63, 119, 119, 112, 2, 55, 250, 191, 251, 252, 0, 16, 6, 2, 14, 54,
    38, 167, 199, 132, 8, 155, 152, 175, 108, 241, 127, 128, 0, 32, 44, 6, 44, 88, 190, 71, 2, 223, 93, 123,
    75, 129, 114, 76, 47, 147, 0, 4, 25, 145, 80, 0, 64, 140, 132, 16, 40, 25, 8, 127, 203, 162, 1, 218,
    193, 64, 101, 68, 2, 68, 32, 0, 17, 58, 240, 199, 6, 128, 2, 12, 200, 168, 0, 32, 204, 178, 160, 0,
    131, 50, 42, 0, 8, 51, 33, 192, 0, 148, 65, 0, 26, 252, 64, 58, 32, 29, 128,
    ];
    const PEXPECT_HASH: u32 = 1978314551;
    const PEXPECT_FRAMES: usize = 4;

    #[test_case]
    fn decodes_embedded_i_and_p_frames_matching_pyav() {
        // Full I+P decode of an embedded baseline clip; hash of all frames' YUV
        // must match the PyAV-derived reference (guards the inter path).
        let track = mp4::parse(&PCLIP_MP4).unwrap();
        let sps = h264::parse_sps(&bits::unescape_rbsp(&track.avcc.sps[0][1..])).unwrap();
        let pps = h264::parse_pps(&bits::unescape_rbsp(&track.avcc.pps[0][1..])).unwrap();
        let mut hh: u32 = 0;
        let mut nf = 0usize;
        let mut reference: Option<h264::decoder::DecodedFrame> = None;
        for s in &track.samples {
            let data = &PCLIP_MP4[s.offset..s.offset + s.size];
            for nal in h264::split_avcc(data, track.avcc.length_size) {
                if nal.kind.is_slice() {
                    let is_idr = nal.kind == h264::NalType::SliceIdr;
                    let df = h264::decoder::decode_slice(&sps, &pps, &nal.rbsp(), is_idr, reference.as_ref()).unwrap();
                    for &b in df.y.iter().chain(df.cb.iter()).chain(df.cr.iter()) {
                        hh = hh.wrapping_mul(31).wrapping_add(b as u32);
                    }
                    nf += 1;
                    reference = Some(df);
                    break;
                }
            }
        }
        assert_eq!(nf, PEXPECT_FRAMES, "decoded frame count");
        assert_eq!(hh, PEXPECT_HASH, "I+P decode must match the PyAV reference");
    }

    fn frame_hash(f: &Frame) -> u32 {
        let mut h: u32 = 0;
        for &p in &f.pixels {
            h = h.wrapping_mul(31).wrapping_add(p);
        }
        h
    }

    #[test_case]
    fn stream_decoder_seek_matches_sequential() {
        // The streaming decoder must produce, for random-access and *backward*
        // seeks, exactly the frames a straight sequential decode produces — it
        // rewinds to the latest keyframe and re-decodes forward (baseline
        // P-frames can't run in reverse). This is what keeps memory to one
        // reference frame instead of the whole clip (the green-frame trap).
        let mut seq = StreamDecoder::open(PCLIP_MP4.to_vec()).unwrap();
        let n = seq.frame_count();
        assert_eq!(n, PEXPECT_FRAMES, "sample count");
        let mut hashes = alloc::vec::Vec::new();
        for i in 0..n {
            assert!(seq.seek_decode(i), "forward decode of frame {i}");
            hashes.push(frame_hash(seq.cur_frame().unwrap()));
        }
        // pts is monotonic non-decreasing.
        for i in 1..n {
            assert!(seq.pts_ms(i) >= seq.pts_ms(i - 1), "pts must not go backwards");
        }
        // A fresh decoder jumped around out of order must match frame-for-frame.
        let mut ra = StreamDecoder::open(PCLIP_MP4.to_vec()).unwrap();
        for &i in &[n - 1, 0, n / 2, 1, n - 1, 0] {
            ra.seek_decode(i);
            assert_eq!(frame_hash(ra.cur_frame().unwrap()), hashes[i], "seek to frame {i} must match sequential decode");
        }
        // Re-seeking the same frame is a no-op (returns false, frame unchanged).
        assert!(!ra.seek_decode(0), "re-seeking the current frame changes nothing");
    }
}
