//! MPEG-TS demuxer (ISO/IEC 13818-1) for HLS media segments.
//!
//! HLS VOD almost always ships **MPEG-TS** segments (or fMP4, handled by the
//! caller via [`super::mp4`]). This module turns a `.ts` byte buffer into:
//!
//! * a video elementary stream type (H.264 / HEVC),
//! * parameter-set NAL units seen in-band (to build `avcC` / `hvcC`),
//! * a list of access units as length-prefixed (AVCC/HVCC) samples.
//!
//! Pure over `&[u8]`. PAT/PMT are required (no PID guessing): a segment with no
//! video stream is an error, not a silent empty track.

use alloc::vec::Vec;

use super::h264;
use super::hevc;
use super::mp4::{AvcC, CodecConfig, HvcC, Sample};

const TS_PACKET: usize = 188;
const SYNC: u8 = 0x47;

/// Stream type codes from the PMT (ISO 13818-1 Table 2-34 + SCTE/ATSC).
const STREAM_H264: u8 = 0x1b;
const STREAM_H265: u8 = 0x24;

/// The MPEG-TS presentation clock. PTS/DTS are 33-bit values on this timebase.
pub const TS_TIMESCALE: u32 = 90_000;
const PTS_MODULUS: u64 = 1 << 33;
/// A PTS earlier than its predecessor by at most this much is B-frame reorder,
/// which is normal. Two seconds is far more than any real reorder depth and far
/// less than any timeline reset.
const MAX_REORDER: u64 = 2 * TS_TIMESCALE as u64;
/// A forward step larger than this is a timeline discontinuity or a missing
/// segment, not the next picture.
const MAX_FORWARD_GAP: u64 = 10 * TS_TIMESCALE as u64;

/// One demuxed video access unit ready to append into a synthetic sample table.
#[derive(Clone, Debug)]
pub struct AccessUnit {
    /// Length-prefixed NAL units (4-byte big-endian length, AVCC/HVCC style).
    pub data: Vec<u8>,
    pub is_sync: bool,
    /// 90 kHz PTS when the PES carried one; `None` when it carried neither.
    pub pts_90k: Option<u64>,
    /// 90 kHz DTS when the PES carried one (`PTS_DTS_flags == 3`).
    pub dts_90k: Option<u64>,
}

/// The timestamps a PES header carried, applied to the first access unit in
/// that PES. `Copy` so a splitter can take its own cursor over them.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PesTime {
    pub pts: Option<u64>,
    pub dts: Option<u64>,
}

/// Result of demuxing one TS buffer (one HLS segment, or a multi-segment concat).
#[derive(Clone, Debug)]
pub struct TsTrack {
    pub config: CodecConfig,
    pub aus: Vec<AccessUnit>,
    /// The playlist marked this segment `#EXT-X-DISCONTINUITY`, so its clock is
    /// unrelated to the previous segment's. Set by the HLS loader; a bare `.ts`
    /// file is one continuous segment and leaves it false.
    pub discontinuity: bool,
}

/// Demux a TS byte stream into video access units. Audio is ignored.
pub fn demux_video(ts: &[u8]) -> Result<TsTrack, &'static str> {
    if ts.len() < TS_PACKET || ts[0] != SYNC {
        return Err("ts: not an MPEG-TS buffer");
    }

    // First pass: PAT → PMT PID, PMT → video PID + stream type.
    let (pmt_pid, _) = find_pat_pmt(ts)?;
    let (video_pid, stream_type) = find_video_in_pmt(ts, pmt_pid)?;
    let is_hevc = stream_type == STREAM_H265;

    // Second pass: collect PES payloads for the video PID.
    let mut pes_buf: Vec<u8> = Vec::new();
    let mut aus: Vec<AccessUnit> = Vec::new();
    let mut sps: Vec<Vec<u8>> = Vec::new();
    let mut pps: Vec<Vec<u8>> = Vec::new();
    let mut vps: Vec<Vec<u8>> = Vec::new();
    let mut continuity: Option<u8> = None;
    let mut pes_time = PesTime::default();

    let mut i = 0;
    while i + TS_PACKET <= ts.len() {
        let pkt = &ts[i..i + TS_PACKET];
        i += TS_PACKET;
        if pkt[0] != SYNC {
            // Resync: search forward for next 0x47 on a packet boundary.
            continue;
        }
        let pid = (((pkt[1] & 0x1f) as u16) << 8) | pkt[2] as u16;
        if pid != video_pid {
            continue;
        }
        let pusi = (pkt[1] & 0x40) != 0;
        let adaptation = (pkt[3] >> 4) & 0x3;
        let cc = pkt[3] & 0x0f;
        if let Some(prev) = continuity {
            let expect = (prev + 1) & 0x0f;
            if cc != expect && cc != prev {
                // Dropped packet — force AU boundary on next payload unit.
                if !pes_buf.is_empty() {
                    flush_pes(
                        &mut pes_buf,
                        &mut pes_time,
                        is_hevc,
                        &mut aus,
                        &mut sps,
                        &mut pps,
                        &mut vps,
                    );
                }
            }
        }
        continuity = Some(cc);

        let mut off = 4usize;
        if adaptation == 2 || adaptation == 3 {
            let alen = pkt[off] as usize;
            off += 1 + alen;
            if off > TS_PACKET {
                continue;
            }
        }
        if adaptation == 0 || adaptation == 2 {
            continue; // no payload
        }
        if off >= TS_PACKET {
            continue;
        }
        let payload = &pkt[off..];

        if pusi {
            // New PES: flush previous elementary buffer into AUs.
            if !pes_buf.is_empty() {
                flush_pes(
                    &mut pes_buf,
                    &mut pes_time,
                    is_hevc,
                    &mut aus,
                    &mut sps,
                    &mut pps,
                    &mut vps,
                );
            }
            // Explicit even though `flush_pes` clears: an empty buffer skips the
            // flush, and carrying the previous PES's timestamps into this one
            // would stamp a picture with its predecessor's time.
            pes_time = PesTime::default();
            if payload.len() < 9 || payload[0] != 0x00 || payload[1] != 0x00 || payload[2] != 0x01 {
                continue;
            }
            let hdr_len = 9 + payload[8] as usize;
            if payload.len() < hdr_len {
                continue;
            }
            // PTS_DTS_flags: 2 = PTS only, 3 = PTS then DTS. 1 is forbidden.
            let pts_dts = payload[7] >> 6;
            if pts_dts >= 2 && payload[8] as usize >= 5 {
                pes_time.pts = Some(read_pes_ts(&payload[9..14]));
            }
            if pts_dts == 3 && payload[8] as usize >= 10 && payload.len() >= 19 {
                pes_time.dts = Some(read_pes_ts(&payload[14..19]));
            }
            pes_buf.extend_from_slice(&payload[hdr_len..]);
        } else {
            pes_buf.extend_from_slice(payload);
        }
    }
    if !pes_buf.is_empty() {
        flush_pes(
            &mut pes_buf,
            &mut pes_time,
            is_hevc,
            &mut aus,
            &mut sps,
            &mut pps,
            &mut vps,
        );
    }

    if aus.is_empty() {
        return Err("ts: no video access units");
    }
    let config = if is_hevc {
        if sps.is_empty() || pps.is_empty() {
            return Err("ts: HEVC stream missing SPS/PPS");
        }
        let sps_rbsp = hevc::parse_sps(&super::bits::unescape_rbsp(
            sps[0].get(2..).unwrap_or(&[]),
        ))?;
        CodecConfig::Hevc(HvcC {
            length_size: 4,
            general_profile_idc: sps_rbsp.ptl.profile_idc,
            general_tier_high: sps_rbsp.ptl.tier_high,
            general_level_idc: sps_rbsp.ptl.level_idc,
            chroma_format_idc: sps_rbsp.chroma_format_idc as u8,
            bit_depth_luma: sps_rbsp.bit_depth_luma as u8,
            bit_depth_chroma: sps_rbsp.bit_depth_chroma as u8,
            vps,
            sps,
            pps,
        })
    } else {
        if sps.is_empty() || pps.is_empty() {
            return Err("ts: H.264 stream missing SPS/PPS");
        }
        CodecConfig::Avc(AvcC {
            length_size: 4,
            sps,
            pps,
        })
    };
    Ok(TsTrack {
        config,
        aus,
        discontinuity: false,
    })
}

/// Turns the 33-bit, reorder-carrying timestamps of a TS stream into a
/// continuous timeline starting at zero.
///
/// Three things make this more than a subtraction, and the naive version got
/// all three wrong by *accumulating* successive deltas:
///
/// * **PTS is not monotonic.** With B-frames a picture's PTS is routinely
///   *earlier* than its predecessor's in decode order, so a running sum walks
///   the timeline backwards and then forwards again.
/// * **PTS is 33 bits and wraps** roughly every 26.5 hours — and a wrap looks
///   exactly like an enormous backward jump unless the delta is taken modulo
///   2^33, which is what makes the forward step small again.
/// * **A playlist can splice** (`#EXT-X-DISCONTINUITY`, or simply a segment
///   from another encode), where the clock restarts at an unrelated value.
///
/// So the step from the previous timestamp is measured **both ways** modulo
/// 2^33: a small forward step is the next picture (wrap included, for free), a
/// small backward step is reorder, and anything else is a discontinuity, which
/// resumes just after the furthest timestamp seen rather than trusting the
/// number. Nothing here can go backwards past zero or collide with a sample
/// already emitted.
#[derive(Debug)]
pub struct Timeline {
    prev_raw: Option<u64>,
    prev: u64,
    /// One past the furthest timestamp emitted — where a discontinuity or a
    /// timestamp-less access unit resumes.
    next: u64,
    default_delta: u64,
    /// Set at a segment boundary: the next timestamp may only continue the
    /// timeline or restart it, never be read as reorder. See
    /// [`Timeline::segment_boundary`].
    boundary: Option<Boundary>,
}

/// What a segment boundary permits the next timestamp to mean.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Boundary {
    /// Continue if the clock continues, otherwise restart.
    Judge,
    /// The playlist declared `#EXT-X-DISCONTINUITY` — restart whatever the
    /// numbers say.
    ForceRestart,
}

impl Timeline {
    /// `default_delta` is the per-picture step used where the stream carries no
    /// timestamp at all (90 kHz ticks).
    pub fn new(default_delta: u64) -> Timeline {
        Timeline {
            prev_raw: None,
            prev: 0,
            next: 0,
            default_delta: default_delta.max(1),
            boundary: None,
        }
    }

    /// Announce that the next access unit begins a new **segment**.
    ///
    /// This is the one place a media clock is allowed to restart, and it has to
    /// be judged here rather than by the generic backward-step rule, because
    /// the two are indistinguishable by size alone: a segment muxed as its own
    /// file commonly restarts at a small PTS, and the step back from the end of
    /// the previous segment then looks exactly like deep B-frame reorder. Read
    /// that way, the second segment's pictures land *on top of* the first's,
    /// and the player interleaves the two — which is precisely what a diff
    /// against PyAV caught: every frame decoded correctly and the sequence came
    /// out `0, 8, 1, 9, 2, 10, …`.
    ///
    /// `declared` is the playlist's own `#EXT-X-DISCONTINUITY`, which is
    /// authoritative: a splice may coincidentally continue the numbers.
    pub fn segment_boundary(&mut self, declared: bool) {
        self.boundary = Some(if declared {
            Boundary::ForceRestart
        } else {
            Boundary::Judge
        });
    }

    /// Map the next access unit's raw 33-bit timestamp onto the timeline.
    /// `None` means the access unit carried none — it takes the slot after the
    /// furthest one so far.
    pub fn push(&mut self, raw: Option<u64>) -> u64 {
        let Some(raw) = raw else {
            let t = self.next;
            self.next = t.saturating_add(self.default_delta);
            return t;
        };
        let raw = raw & (PTS_MODULUS - 1);
        let boundary = self.boundary.take();
        let t = match self.prev_raw {
            None => 0,
            Some(prev_raw) => {
                let forward = raw.wrapping_sub(prev_raw) & (PTS_MODULUS - 1);
                let backward = prev_raw.wrapping_sub(raw) & (PTS_MODULUS - 1);
                match boundary {
                    Some(Boundary::ForceRestart) => self.next,
                    Some(Boundary::Judge) => {
                        if forward <= MAX_FORWARD_GAP {
                            self.prev.saturating_add(forward)
                        } else {
                            self.next
                        }
                    }
                    None if forward <= MAX_FORWARD_GAP => self.prev.saturating_add(forward),
                    None if backward <= MAX_REORDER => self.prev.saturating_sub(backward),
                    None => self.next,
                }
            }
        };
        self.prev_raw = Some(raw);
        self.prev = t;
        self.next = self.next.max(t.saturating_add(self.default_delta));
        t
    }
}

/// Build a flat sample table + contiguous length-prefixed buffer from one or
/// more demuxed TS tracks (segments concatenated in order).
pub fn assemble_samples(
    tracks: &[TsTrack],
) -> Result<(Vec<u8>, CodecConfig, Vec<Sample>, u32), &'static str> {
    if tracks.is_empty() {
        return Err("ts: no tracks");
    }
    let config = tracks[0].config.clone();
    // All segments of a VOD playlist share a codec; refuse a mid-stream switch.
    for t in tracks.iter().skip(1) {
        if core::mem::discriminant(&t.config) != core::mem::discriminant(&config) {
            return Err("ts: codec changes mid-playlist");
        }
    }
    let default_delta = 3000u64; // 30 fps at 90 kHz — only used with no timestamps
    let mut bytes = Vec::new();
    let mut samples = Vec::new();
    // Presentation and decode run on their own timelines: a stream with
    // B-frames has PTS != DTS by construction, so one cursor cannot serve both.
    let mut pts_line = Timeline::new(default_delta);
    let mut dts_line = Timeline::new(default_delta);

    for (seg, t) in tracks.iter().enumerate() {
        if seg > 0 {
            // Each track is one segment, and a segment boundary is the only
            // place a media clock may legally restart.
            pts_line.segment_boundary(t.discontinuity);
            dts_line.segment_boundary(t.discontinuity);
        }
        for au in &t.aus {
            if au.data.is_empty() {
                continue;
            }
            let offset = bytes.len();
            bytes.extend_from_slice(&au.data);
            // A PES carrying only a PTS means PTS == DTS for that picture
            // (ISO 13818-1) — not "no decode time".
            let cts = pts_line.push(au.pts_90k);
            let dts = dts_line.push(au.dts_90k.or(au.pts_90k));
            samples.push(Sample {
                offset,
                size: au.data.len(),
                dts,
                cts,
                is_sync: au.is_sync,
            });
        }
    }
    if samples.is_empty() {
        return Err("ts: assembled zero samples");
    }
    Ok((bytes, config, samples, TS_TIMESCALE))
}

fn flush_pes(
    pes: &mut Vec<u8>,
    time: &mut PesTime,
    is_hevc: bool,
    aus: &mut Vec<AccessUnit>,
    sps: &mut Vec<Vec<u8>>,
    pps: &mut Vec<Vec<u8>>,
    vps: &mut Vec<Vec<u8>>,
) {
    if is_hevc {
        split_hevc_aus(pes, time, aus, sps, pps, vps);
    } else {
        split_avc_aus(pes, time, aus, sps, pps);
    }
    pes.clear();
    *time = PesTime::default();
}

fn split_avc_aus(
    es: &[u8],
    first: &PesTime,
    aus: &mut Vec<AccessUnit>,
    sps_out: &mut Vec<Vec<u8>>,
    pps_out: &mut Vec<Vec<u8>>,
) {
    let nals = h264::split_annexb(es);
    if nals.is_empty() {
        return;
    }
    let mut cur: Vec<u8> = Vec::new();
    let mut sync = false;
    let mut started = false;
    // A PES header timestamps the *first* access unit it carries; any further
    // ones in the same PES get none and are interpolated on the timeline.
    let mut time = *first;

    for nal in nals {
        match nal.kind {
            h264::NalType::Sps => {
                if !sps_out.iter().any(|s| s.as_slice() == nal.payload) {
                    sps_out.push(nal.payload.to_vec());
                }
            }
            h264::NalType::Pps => {
                if !pps_out.iter().any(|s| s.as_slice() == nal.payload) {
                    pps_out.push(nal.payload.to_vec());
                }
            }
            h264::NalType::AuDelim => {
                if started && !cur.is_empty() && has_vcl_avc(&cur) {
                    aus.push(AccessUnit {
                        data: core::mem::take(&mut cur),
                        is_sync: sync,
                        pts_90k: time.pts.take(),
                        dts_90k: time.dts.take(),
                    });
                    sync = false;
                } else {
                    cur.clear();
                }
                started = true;
                continue;
            }
            h264::NalType::SliceIdr => {
                // New AU on IDR if we already have VCL.
                if started && !cur.is_empty() && has_vcl_avc(&cur) {
                    aus.push(AccessUnit {
                        data: core::mem::take(&mut cur),
                        is_sync: sync,
                        pts_90k: time.pts.take(),
                        dts_90k: time.dts.take(),
                    });
                    sync = false;
                }
                started = true;
                sync = true;
            }
            h264::NalType::Slice => {
                if started && !cur.is_empty() && has_vcl_avc(&cur) && first_mb_zero_avc(nal.payload)
                {
                    aus.push(AccessUnit {
                        data: core::mem::take(&mut cur),
                        is_sync: sync,
                        pts_90k: time.pts.take(),
                        dts_90k: time.dts.take(),
                    });
                    sync = false;
                }
                started = true;
            }
            _ => {
                started = true;
            }
        }
        push_len_prefixed(&mut cur, nal.payload);
    }
    // Only emit AUs that carry at least one VCL NAL — parameter-set-only
    // units would occupy a sample index and never produce a picture, so
    // seek_decode(0) would fail even though later AUs are fine.
    if !cur.is_empty() && has_vcl_avc(&cur) {
        aus.push(AccessUnit {
            data: cur,
            is_sync: sync,
            pts_90k: time.pts,
            dts_90k: time.dts,
        });
    }
}

fn split_hevc_aus(
    es: &[u8],
    first: &PesTime,
    aus: &mut Vec<AccessUnit>,
    sps_out: &mut Vec<Vec<u8>>,
    pps_out: &mut Vec<Vec<u8>>,
    vps_out: &mut Vec<Vec<u8>>,
) {
    let nals = hevc::split_annexb(es);
    if nals.is_empty() {
        return;
    }
    let mut cur: Vec<u8> = Vec::new();
    let mut sync = false;
    let mut started = false;
    let mut time = *first;

    for nal in nals {
        match nal.kind {
            hevc::NalType::Vps => {
                if !vps_out.iter().any(|s| s.as_slice() == nal.payload) {
                    vps_out.push(nal.payload.to_vec());
                }
            }
            hevc::NalType::Sps => {
                if !sps_out.iter().any(|s| s.as_slice() == nal.payload) {
                    sps_out.push(nal.payload.to_vec());
                }
            }
            hevc::NalType::Pps => {
                if !pps_out.iter().any(|s| s.as_slice() == nal.payload) {
                    pps_out.push(nal.payload.to_vec());
                }
            }
            hevc::NalType::AuDelim => {
                // Same rule as the AVC path: only a unit carrying a VCL NAL is
                // a picture. A leading VPS/SPS/PPS-only run before the first
                // delimiter would otherwise become sample 0 and never decode.
                if started && !cur.is_empty() && has_vcl_hevc(&cur) {
                    aus.push(AccessUnit {
                        data: core::mem::take(&mut cur),
                        is_sync: sync,
                        pts_90k: time.pts.take(),
                        dts_90k: time.dts.take(),
                    });
                    sync = false;
                } else {
                    cur.clear();
                }
                started = true;
                continue;
            }
            hevc::NalType::SliceIrap(_) => {
                if started && !cur.is_empty() && has_vcl_hevc(&cur) {
                    aus.push(AccessUnit {
                        data: core::mem::take(&mut cur),
                        is_sync: sync,
                        pts_90k: time.pts.take(),
                        dts_90k: time.dts.take(),
                    });
                    sync = false;
                }
                started = true;
                sync = nal.kind.is_irap();
            }
            hevc::NalType::Slice(_) => {
                if started && !cur.is_empty() && has_vcl_hevc(&cur) && first_slice_hevc(nal.payload)
                {
                    aus.push(AccessUnit {
                        data: core::mem::take(&mut cur),
                        is_sync: sync,
                        pts_90k: time.pts.take(),
                        dts_90k: time.dts.take(),
                    });
                    sync = false;
                }
                started = true;
            }
            _ => {
                started = true;
            }
        }
        push_len_prefixed(&mut cur, nal.payload);
    }
    if !cur.is_empty() && has_vcl_hevc(&cur) {
        aus.push(AccessUnit {
            data: cur,
            is_sync: sync,
            pts_90k: time.pts,
            dts_90k: time.dts,
        });
    }
}

fn push_len_prefixed(buf: &mut Vec<u8>, nal: &[u8]) {
    let n = nal.len() as u32;
    buf.extend_from_slice(&n.to_be_bytes());
    buf.extend_from_slice(nal);
}

fn has_vcl_avc(avcc: &[u8]) -> bool {
    let mut i = 0;
    while i + 4 <= avcc.len() {
        let n = u32::from_be_bytes([avcc[i], avcc[i + 1], avcc[i + 2], avcc[i + 3]]) as usize;
        i += 4;
        if i + n > avcc.len() {
            break;
        }
        let nal_type = avcc[i] & 0x1f;
        if nal_type == 1 || nal_type == 5 {
            return true;
        }
        i += n;
    }
    false
}

fn has_vcl_hevc(hvcc: &[u8]) -> bool {
    let mut i = 0;
    while i + 4 <= hvcc.len() {
        let n = u32::from_be_bytes([hvcc[i], hvcc[i + 1], hvcc[i + 2], hvcc[i + 3]]) as usize;
        i += 4;
        if i + n > hvcc.len() || n < 2 {
            break;
        }
        let nal_type = (hvcc[i] >> 1) & 0x3f;
        if nal_type <= 31 {
            return true;
        }
        i += n;
    }
    false
}

fn first_mb_zero_avc(nal: &[u8]) -> bool {
    // Slice header: first_mb_in_slice is ue(v) after the 1-byte NAL header.
    if nal.len() < 2 {
        return false;
    }
    let rbsp = super::bits::unescape_rbsp(&nal[1..]);
    let mut r = super::bits::BitReader::new(&rbsp);
    r.ue().map(|v| v == 0).unwrap_or(false)
}

fn first_slice_hevc(nal: &[u8]) -> bool {
    if nal.len() < 3 {
        return false;
    }
    let rbsp = super::bits::unescape_rbsp(&nal[2..]);
    if rbsp.is_empty() {
        return false;
    }
    // first_slice_segment_in_pic_flag is the first bit of the slice segment header.
    (rbsp[0] & 0x80) != 0
}

fn read_pes_ts(b: &[u8]) -> u64 {
    // 33-bit PTS: marker bits in PES.
    let b0 = b[0] as u64;
    let b1 = b[1] as u64;
    let b2 = b[2] as u64;
    let b3 = b[3] as u64;
    let b4 = b[4] as u64;
    ((b0 & 0x0e) << 29) | ((b1) << 22) | ((b2 & 0xfe) << 14) | ((b3) << 7) | ((b4 & 0xfe) >> 1)
}

fn find_pat_pmt(ts: &[u8]) -> Result<(u16, u16), &'static str> {
    let mut i = 0;
    while i + TS_PACKET <= ts.len() {
        let pkt = &ts[i..i + TS_PACKET];
        i += TS_PACKET;
        if pkt[0] != SYNC {
            continue;
        }
        let pid = (((pkt[1] & 0x1f) as u16) << 8) | pkt[2] as u16;
        if pid != 0 {
            continue;
        }
        let pusi = (pkt[1] & 0x40) != 0;
        let adaptation = (pkt[3] >> 4) & 0x3;
        let mut off = 4usize;
        if adaptation == 2 || adaptation == 3 {
            let alen = pkt[off] as usize;
            off += 1 + alen;
        }
        if off >= TS_PACKET {
            continue;
        }
        if pusi {
            // pointer_field
            let ptr = pkt[off] as usize;
            off += 1 + ptr;
        }
        if off + 8 > TS_PACKET {
            continue;
        }
        if pkt[off] != 0x00 {
            continue; // not PAT table_id
        }
        // section after pointer; program_map_PID at end of first program loop.
        // Minimal PAT: table_id, section_length, ts_id, version, section nums,
        // then (program_number, program_map_PID)+
        let section_len =
            ((((pkt[off + 1] as usize) & 0x0f) << 8) | pkt[off + 2] as usize) & 0x0fff;
        let mut p = off + 8; // skip fixed header (8 bytes after table_id start: 3 + 5)
        // Actually: table_id(1) + section_syntax(2) + transport_stream_id(2)
        // + version/current(1) + section_number(1) + last(1) = 8 bytes from table_id.
        let end = (off + 3 + section_len).min(TS_PACKET).saturating_sub(4); // CRC
        while p + 4 <= end {
            let prog = ((pkt[p] as u16) << 8) | pkt[p + 1] as u16;
            let pid = ((((pkt[p + 2] as u16) & 0x1f) << 8) | pkt[p + 3] as u16);
            p += 4;
            if prog != 0 {
                return Ok((pid, prog));
            }
        }
    }
    Err("ts: no PAT/PMT")
}

fn find_video_in_pmt(ts: &[u8], pmt_pid: u16) -> Result<(u16, u8), &'static str> {
    let mut i = 0;
    while i + TS_PACKET <= ts.len() {
        let pkt = &ts[i..i + TS_PACKET];
        i += TS_PACKET;
        if pkt[0] != SYNC {
            continue;
        }
        let pid = (((pkt[1] & 0x1f) as u16) << 8) | pkt[2] as u16;
        if pid != pmt_pid {
            continue;
        }
        let pusi = (pkt[1] & 0x40) != 0;
        let adaptation = (pkt[3] >> 4) & 0x3;
        let mut off = 4usize;
        if adaptation == 2 || adaptation == 3 {
            let alen = pkt[off] as usize;
            off += 1 + alen;
        }
        if off >= TS_PACKET {
            continue;
        }
        if pusi {
            let ptr = pkt[off] as usize;
            off += 1 + ptr;
        }
        if off + 12 > TS_PACKET {
            continue;
        }
        if pkt[off] != 0x02 {
            continue; // PMT table_id
        }
        let section_len =
            ((((pkt[off + 1] as usize) & 0x0f) << 8) | pkt[off + 2] as usize) & 0x0fff;
        let program_info_len =
            ((((pkt[off + 10] as usize) & 0x0f) << 8) | pkt[off + 11] as usize) & 0x0fff;
        let mut p = off + 12 + program_info_len;
        let end = (off + 3 + section_len).min(TS_PACKET).saturating_sub(4);
        while p + 5 <= end {
            let stype = pkt[p];
            let epid = ((((pkt[p + 1] as u16) & 0x1f) << 8) | pkt[p + 2] as u16);
            let es_info_len =
                ((((pkt[p + 3] as usize) & 0x0f) << 8) | pkt[p + 4] as usize) & 0x0fff;
            p += 5 + es_info_len;
            if stype == STREAM_H264 || stype == STREAM_H265 {
                return Ok((epid, stype));
            }
        }
    }
    Err("ts: PMT has no H.264/HEVC video stream")
}

#[cfg(test)]
mod tests {
    use super::*;

    const PMT_PID: u16 = 0x100;
    const VIDEO_PID: u16 = 0x101;

    /// A miniature TS **multiplexer** — PAT, PMT and one PES per access unit,
    /// including adaptation-field stuffing and multi-packet PES continuation.
    ///
    /// A demuxer tested only against buffers its own author hand-placed proves
    /// very little, so the fixtures here are produced by an encoder written from
    /// the spec's field order rather than from `demux_video`'s reads.
    #[derive(Default)]
    struct Mux {
        out: Vec<u8>,
        cc: alloc::collections::BTreeMap<u16, u8>,
    }

    impl Mux {
        /// Wrap `payload` in as many 188-byte packets as it needs. A short final
        /// packet is padded with an adaptation field, never with trailing bytes
        /// the demuxer would read as elementary data.
        fn packets(&mut self, pid: u16, payload: &[u8]) {
            let mut first = true;
            let mut i = 0usize;
            loop {
                let cc = self.cc.entry(pid).or_insert(0);
                let mut p = [0xffu8; TS_PACKET];
                p[0] = SYNC;
                p[1] = ((pid >> 8) as u8) & 0x1f;
                if first {
                    p[1] |= 0x40; // payload_unit_start_indicator
                }
                p[2] = pid as u8;
                let take = (payload.len() - i).min(184);
                if take < 184 {
                    let pad = 184 - take;
                    p[3] = 0x30 | (*cc & 0x0f); // adaptation field + payload
                    p[4] = (pad - 1) as u8; // adaptation_field_length
                    if pad >= 2 {
                        p[5] = 0x00; // flags, then 0xff stuffing (already set)
                    }
                    p[4 + pad..].copy_from_slice(&payload[i..i + take]);
                } else {
                    p[3] = 0x10 | (*cc & 0x0f); // payload only
                    p[4..].copy_from_slice(&payload[i..i + take]);
                }
                *cc = (*cc + 1) & 0x0f;
                self.out.extend_from_slice(&p);
                i += take;
                first = false;
                if i >= payload.len() {
                    break;
                }
            }
        }

        /// A PSI section packet: `pointer_field` then the table.
        fn section(&mut self, pid: u16, table: &[u8]) {
            let mut payload = alloc::vec![0u8];
            payload.extend_from_slice(table);
            self.packets(pid, &payload);
        }

        fn pat(&mut self) {
            let mut t = alloc::vec![
                0x00, // table_id
                0xb0, 0x0d, // section_syntax + section_length (13)
                0x00, 0x01, // transport_stream_id
                0xc1, 0x00, 0x00, // version / section numbers
                0x00, 0x01, // program_number 1
            ];
            t.push(0xe0 | (PMT_PID >> 8) as u8);
            t.push(PMT_PID as u8);
            t.extend_from_slice(&[0, 0, 0, 0]); // CRC32 (not verified)
            self.section(0, &t);
        }

        fn pmt(&mut self, stream_type: u8) {
            let mut t = alloc::vec![
                0x02, // table_id
                0xb0, 0x12, // section_length (18)
                0x00, 0x01, // program_number
                0xc1, 0x00, 0x00, // version / section numbers
            ];
            t.push(0xe0 | (VIDEO_PID >> 8) as u8); // PCR_PID
            t.push(VIDEO_PID as u8);
            t.extend_from_slice(&[0xf0, 0x00]); // program_info_length = 0
            t.push(stream_type);
            t.push(0xe0 | (VIDEO_PID >> 8) as u8);
            t.push(VIDEO_PID as u8);
            t.extend_from_slice(&[0xf0, 0x00]); // ES_info_length = 0
            t.extend_from_slice(&[0, 0, 0, 0]); // CRC32
            self.section(PMT_PID, &t);
        }

        /// One PES packet carrying `es` (Annex-B), with optional PTS and DTS.
        fn pes(&mut self, es: &[u8], pts: Option<u64>, dts: Option<u64>) {
            let mut stamps = Vec::new();
            let flags = match (pts, dts) {
                (Some(p), Some(d)) => {
                    stamps.extend_from_slice(&encode_ts(p, 0x30));
                    stamps.extend_from_slice(&encode_ts(d, 0x10));
                    0xc0
                }
                (Some(p), None) => {
                    stamps.extend_from_slice(&encode_ts(p, 0x20));
                    0x80
                }
                _ => 0x00,
            };
            let mut pes = alloc::vec![0x00, 0x00, 0x01, 0xe0];
            let len = 3 + stamps.len() + es.len();
            pes.push((len >> 8) as u8);
            pes.push(len as u8);
            pes.push(0x80); // '10' + no scrambling
            pes.push(flags);
            pes.push(stamps.len() as u8); // PES_header_data_length
            pes.extend_from_slice(&stamps);
            pes.extend_from_slice(es);
            self.packets(VIDEO_PID, &pes);
        }
    }

    /// Inverse of [`read_pes_ts`]: the 33-bit value split across five bytes with
    /// a 4-bit prefix and three marker bits.
    fn encode_ts(v: u64, prefix: u8) -> [u8; 5] {
        [
            prefix | (((v >> 29) as u8) & 0x0e) | 1,
            (v >> 22) as u8,
            (((v >> 14) as u8) & 0xfe) | 1,
            (v >> 7) as u8,
            (((v << 1) as u8) & 0xfe) | 1,
        ]
    }

    fn annexb(nals: &[&[u8]]) -> Vec<u8> {
        let mut out = Vec::new();
        for n in nals {
            out.extend_from_slice(&[0, 0, 0, 1]);
            out.extend_from_slice(n);
        }
        out
    }

    /// `first_mb_in_slice = ue(0)` is a single set bit, so any slice body here
    /// starts at macroblock 0 — i.e. it begins a picture.
    fn avc_slice(nal_type: u8, len: usize) -> Vec<u8> {
        let mut v = alloc::vec![0x60 | nal_type, 0x80];
        v.resize(len.max(2), 0x42);
        v
    }

    fn avc_stream(stream: &[(u8, Option<u64>, Option<u64>)]) -> Vec<u8> {
        let mut m = Mux::default();
        m.pat();
        m.pmt(STREAM_H264);
        let sps: &[u8] = &[0x67, 0x42, 0xc0, 0x1e];
        let pps: &[u8] = &[0x68, 0xce, 0x3c, 0x80];
        for (i, (nal_type, pts, dts)) in stream.iter().enumerate() {
            let slice = avc_slice(*nal_type, 40 + i);
            let es = if i == 0 {
                annexb(&[&[0x09, 0x10], sps, pps, &slice])
            } else {
                annexb(&[&[0x09, 0x30], &slice])
            };
            m.pes(&es, *pts, *dts);
        }
        m.out
    }

    #[test_case]
    fn rejects_non_ts() {
        assert!(demux_video(b"not a ts").is_err());
        assert!(demux_video(&[0x47u8; 10]).is_err());
    }

    #[test_case]
    fn a_muxed_stream_round_trips_through_the_demuxer() {
        // IDR then two P pictures, one PES each, PTS == DTS.
        let ts = avc_stream(&[
            (5, Some(9000), Some(9000)),
            (1, Some(12000), Some(12000)),
            (1, Some(15000), Some(15000)),
        ]);
        let track = demux_video(&ts).expect("demux");
        assert_eq!(track.aus.len(), 3);
        assert!(track.aus[0].is_sync, "IDR is a sync sample");
        assert!(!track.aus[1].is_sync);
        match &track.config {
            CodecConfig::Avc(a) => {
                // Parameter sets are lifted out of the elementary stream and
                // de-duplicated, never left inline as a picture.
                assert_eq!(a.sps.len(), 1);
                assert_eq!(a.pps.len(), 1);
                assert_eq!(a.sps[0][0], 0x67);
                assert_eq!(a.length_size, 4);
            }
            other => panic!("expected AVC, got {other:?}"),
        }

        let (bytes, _cfg, samples, timescale) = assemble_samples(&[track]).unwrap();
        assert_eq!(timescale, TS_TIMESCALE);
        assert_eq!(samples.len(), 3);
        // The timeline starts at zero regardless of the stream's own PTS base.
        assert_eq!(samples[0].cts, 0);
        assert_eq!(samples[1].cts, 3000);
        assert_eq!(samples[2].cts, 6000);
        // Offsets tile the assembled buffer exactly.
        let mut at = 0usize;
        for s in &samples {
            assert_eq!(s.offset, at);
            at += s.size;
        }
        assert_eq!(bytes.len(), at);
        // Every sample is a run of 4-byte-length-prefixed NAL units.
        for s in &samples {
            let mut i = s.offset;
            while i < s.offset + s.size {
                let n = u32::from_be_bytes(bytes[i..i + 4].try_into().unwrap()) as usize;
                assert!(n > 0 && i + 4 + n <= s.offset + s.size, "NAL length in range");
                i += 4 + n;
            }
            assert_eq!(i, s.offset + s.size, "lengths tile the sample exactly");
        }
    }

    #[test_case]
    fn a_pes_split_across_packets_is_reassembled() {
        // 600 bytes of elementary data needs four TS packets, so the access
        // unit only comes back whole if continuation payloads are appended.
        let mut m = Mux::default();
        m.pat();
        m.pmt(STREAM_H264);
        let big = avc_slice(5, 600);
        m.pes(
            &annexb(&[&[0x09, 0x10], &[0x67, 0x42, 0xc0, 0x1e], &[0x68, 0xce], &big]),
            Some(0),
            None,
        );
        assert!(m.out.len() / TS_PACKET >= 5, "spans several packets");
        let track = demux_video(&m.out).expect("demux");
        assert_eq!(track.aus.len(), 1);
        // Each NAL is length-prefixed. The access-unit **delimiter** is dropped
        // — it marks a boundary and carries no sample data — while the in-band
        // parameter sets stay inline as well as being lifted into the config,
        // which is legal AVCC and what the decoder expects.
        let sps = 4 + 4;
        let pps = 4 + 2;
        let slice = 4 + 600;
        assert_eq!(track.aus[0].data.len(), sps + pps + slice);
    }

    #[test_case]
    fn parameter_sets_before_the_first_picture_do_not_become_a_sample() {
        // A PES holding only SPS/PPS must not produce an access unit: it would
        // take sample index 0 and never decode into a picture.
        let mut m = Mux::default();
        m.pat();
        m.pmt(STREAM_H264);
        m.pes(&annexb(&[&[0x67, 0x42, 0xc0, 0x1e], &[0x68, 0xce]]), Some(0), None);
        m.pes(&annexb(&[&[0x09, 0x10], &avc_slice(5, 20)]), Some(3000), None);
        let track = demux_video(&m.out).expect("demux");
        assert_eq!(track.aus.len(), 1, "only the picture is an access unit");
        assert!(track.aus[0].is_sync);
    }

    #[test_case]
    fn a_stream_with_no_video_in_the_pmt_is_an_error() {
        let mut m = Mux::default();
        m.pat();
        m.pmt(0x0f); // AAC audio only
        let err = demux_video(&m.out).unwrap_err();
        assert!(err.contains("video"), "{err}");
    }

    #[test_case]
    fn a_pat_without_its_pmt_fails_closed() {
        let mut m = Mux::default();
        m.pat();
        let err = demux_video(&m.out).unwrap_err();
        assert!(err.contains("PMT") || err.contains("video"), "{err}");
    }

    #[test_case]
    fn assemble_samples_assigns_offsets() {
        let au = AccessUnit {
            data: {
                let mut d = Vec::new();
                push_len_prefixed(&mut d, &[0x65, 0x88, 0x84]); // fake IDR header
                d
            },
            is_sync: true,
            pts_90k: Some(0),
            dts_90k: Some(0),
        };
        let track = TsTrack {
            config: CodecConfig::Avc(AvcC {
                length_size: 4,
                sps: alloc::vec![alloc::vec![0x67, 0x42]],
                pps: alloc::vec![alloc::vec![0x68, 0xce]],
            }),
            aus: alloc::vec![
                au.clone(),
                AccessUnit {
                    pts_90k: Some(3000),
                    dts_90k: Some(3000),
                    is_sync: false,
                    ..au
                }
            ],
            discontinuity: false,
        };
        let (bytes, _cfg, samples, ts) = assemble_samples(&[track]).unwrap();
        assert_eq!(ts, TS_TIMESCALE);
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].offset, 0);
        assert_eq!(samples[0].size + samples[0].offset, samples[1].offset);
        assert_eq!(bytes.len(), samples[1].offset + samples[1].size);
        assert!(samples[0].is_sync);
        assert!(!samples[1].is_sync);
    }

    #[test_case]
    fn a_codec_change_between_segments_is_refused() {
        let avc = TsTrack {
            config: CodecConfig::Avc(AvcC {
                length_size: 4,
                sps: alloc::vec![alloc::vec![0x67]],
                pps: alloc::vec![alloc::vec![0x68]],
            }),
            aus: Vec::new(),
            discontinuity: false,
        };
        let hevc = TsTrack {
            config: CodecConfig::Hevc(HvcC {
                length_size: 4,
                general_profile_idc: 1,
                general_tier_high: false,
                general_level_idc: 93,
                chroma_format_idc: 1,
                bit_depth_luma: 8,
                bit_depth_chroma: 8,
                vps: Vec::new(),
                sps: Vec::new(),
                pps: Vec::new(),
            }),
            aus: Vec::new(),
            discontinuity: false,
        };
        assert!(assemble_samples(&[avc, hevc]).is_err());
    }

    // ---- the timeline ----

    /// Decode-order PTS for an IPBB pattern: the two B pictures are presented
    /// *between* the I and the P that were coded before them.
    #[test_case]
    fn reordered_timestamps_are_not_accumulated() {
        let mut t = Timeline::new(3000);
        let base = 900_000u64;
        let got: Vec<u64> = [0u64, 9000, 3000, 6000]
            .iter()
            .map(|d| t.push(Some(base + d)))
            .collect();
        // The naive running-sum version summed |delta| and produced
        // 0, 9000, 15000, 18000 — monotonic, and wrong for every B picture.
        assert_eq!(got, alloc::vec![0, 9000, 3000, 6000]);
    }

    #[test_case]
    fn a_thirty_three_bit_wrap_is_one_step_not_a_jump() {
        let mut t = Timeline::new(3000);
        let last = PTS_MODULUS - 3000;
        assert_eq!(t.push(Some(last)), 0);
        // The clock rolls over: the raw value collapses to ~0 but the picture
        // is exactly one frame later.
        assert_eq!(t.push(Some(0)), 3000);
        assert_eq!(t.push(Some(3000)), 6000);
    }

    #[test_case]
    fn a_spliced_timeline_resumes_after_the_last_sample() {
        let mut t = Timeline::new(3000);
        assert_eq!(t.push(Some(90_000)), 0);
        assert_eq!(t.push(Some(93_000)), 3000);
        // A segment from another encode restarts the clock at an unrelated
        // value; it must continue the timeline rather than jump or go backwards.
        let spliced = t.push(Some(500_000_000));
        assert_eq!(spliced, 6000);
        assert_eq!(t.push(Some(500_003_000)), 9000);
    }

    #[test_case]
    fn access_units_without_timestamps_are_interpolated() {
        let mut t = Timeline::new(3000);
        assert_eq!(t.push(Some(0)), 0);
        assert_eq!(t.push(None), 3000);
        assert_eq!(t.push(None), 6000);
        // A timestamp arriving later still lands on its own clock, and never
        // on top of a sample already emitted.
        assert_eq!(t.push(Some(9000)), 9000);
    }

    /// The bug a PyAV diff caught: two segments each muxed as their own file
    /// restart the clock, and the step back from the end of segment 0 to the
    /// start of segment 1 is *smaller* than a deep reorder — so read as reorder,
    /// segment 1's pictures land on top of segment 0's and the player
    /// interleaves them (`0, 8, 1, 9, 2, 10, …`, every frame decoding fine).
    #[test_case]
    fn a_segment_that_restarts_its_clock_does_not_interleave() {
        let mut t = Timeline::new(11_250);
        // Segment 0: PTS 22500..101250 in decode order (B-frames reorder).
        for raw in [22_500u64, 56_250, 33_750, 45_000] {
            t.push(Some(raw));
        }
        // Segment 1 restarts at exactly the same base.
        t.segment_boundary(false);
        let first = t.push(Some(22_500));
        assert!(
            first > 33_750,
            "segment 1 must start after segment 0 ended, got {first}"
        );
        // …and keeps its own internal reorder relative to that new base.
        let second = t.push(Some(56_250));
        assert_eq!(second, first + 33_750);
        let third = t.push(Some(33_750));
        assert_eq!(third, first + 11_250, "reorder still works inside a segment");
    }

    #[test_case]
    fn a_segment_whose_clock_continues_is_not_restarted() {
        // The ordinary HLS case: one continuous timeline across segments. The
        // boundary must not introduce a gap.
        let mut t = Timeline::new(3000);
        assert_eq!(t.push(Some(90_000)), 0);
        assert_eq!(t.push(Some(93_000)), 3000);
        t.segment_boundary(false);
        assert_eq!(t.push(Some(96_000)), 6000, "no gap at a continuous boundary");
    }

    #[test_case]
    fn a_declared_discontinuity_restarts_even_when_the_numbers_continue() {
        // `#EXT-X-DISCONTINUITY` is authoritative: a splice may coincidentally
        // land on a continuing timestamp, and the playlist knows what the bytes
        // cannot say.
        let mut t = Timeline::new(3000);
        assert_eq!(t.push(Some(90_000)), 0);
        assert_eq!(t.push(Some(93_000)), 3000);
        t.segment_boundary(true);
        assert_eq!(t.push(Some(96_000)), 6000);
        // Resumed after the last sample rather than tracking the raw step —
        // here the two happen to agree, so check a spliced clock as well.
        t.segment_boundary(true);
        assert_eq!(t.push(Some(7_000_000)), 9000);
    }

    #[test_case]
    fn segments_are_concatenated_in_playlist_order() {
        let seg = |base: u64, sync_first: bool| TsTrack {
            config: CodecConfig::Avc(AvcC {
                length_size: 4,
                sps: alloc::vec![alloc::vec![0x67]],
                pps: alloc::vec![alloc::vec![0x68]],
            }),
            aus: (0..3)
                .map(|i| AccessUnit {
                    data: alloc::vec![0u8; 8],
                    is_sync: i == 0 && sync_first,
                    pts_90k: Some(base + i * 3000),
                    dts_90k: Some(base + i * 3000),
                })
                .collect(),
            discontinuity: false,
        };
        // Both segments carry the same PTS range — the case that interleaved.
        let (_b, _c, samples, _t) = assemble_samples(&[seg(22_500, true), seg(22_500, true)]).unwrap();
        let cts: Vec<u64> = samples.iter().map(|s| s.cts).collect();
        assert!(
            cts.windows(2).all(|w| w[0] < w[1]),
            "strictly increasing across the join, got {cts:?}"
        );
    }

    #[test_case]
    fn a_timeline_never_runs_backwards_past_zero() {
        let mut t = Timeline::new(3000);
        assert_eq!(t.push(Some(1000)), 0);
        // A picture stamped *before* the first one cannot be negative.
        assert_eq!(t.push(Some(0)), 0);
    }

    #[test_case]
    fn decode_and_presentation_timelines_are_independent() {
        // B pictures: DTS is monotonic while PTS is not, so one cursor cannot
        // serve both.
        let ts = avc_stream(&[
            (5, Some(9000), Some(9000)),
            (1, Some(18000), Some(12000)),
            (1, Some(12000), Some(15000)),
            (1, Some(15000), Some(18000)),
        ]);
        let track = demux_video(&ts).expect("demux");
        let (_b, _c, samples, _t) = assemble_samples(&[track]).unwrap();
        let cts: Vec<u64> = samples.iter().map(|s| s.cts).collect();
        let dts: Vec<u64> = samples.iter().map(|s| s.dts).collect();
        assert_eq!(cts, alloc::vec![0, 9000, 3000, 6000]);
        assert_eq!(dts, alloc::vec![0, 3000, 6000, 9000]);
        assert!(dts.windows(2).all(|w| w[0] < w[1]), "DTS is monotonic");
    }
}
