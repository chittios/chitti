//! Chunked WAV/MP3/AAC and H.264/HEVC/VP9 decode state.
//!
//! Lives in a `static mut` so it survives [`super::chunked::ChunkCmd::Continue`].
//! Output buffers (`out` / YUV planes) are reused so a bump arena does not
//! grow by the sum of every chunk.

use crate::audio;
use crate::chunked::{ChunkBlock, ChunkStatus, KIND_AUDIO, KIND_VIDEO};
use crate::video;
use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::rc::Rc;
use alloc::vec::Vec;

static mut SESSION: Option<Box<Session>> = None;

pub fn drop_session() {
    // SAFETY: single-threaded tenant; called only on Init/Cancel.
    unsafe {
        *core::ptr::addr_of_mut!(SESSION) = None;
    }
}

pub fn heap_guess(input_len: usize) -> u64 {
    (input_len.saturating_mul(8).saturating_add(8 << 20)) as u64
}

/// Bump arenas do not reclaim dropped DPB pictures (only the top allocation).
/// Size for every decoded frame plus scratch, not a high-water mark.
fn video_arena_bytes(w: u32, h: u32, frames: usize, input_len: usize) -> u64 {
    let yuv = (w as usize).saturating_mul(h as usize).saturating_mul(2);
    let pics = frames.max(1);
    input_len
        .saturating_add(yuv.saturating_mul(pics))
        .saturating_add(16 << 20) as u64
}

pub fn init(input: &[u8], b: &mut ChunkBlock) -> ChunkStatus {
    // `meta0 == KIND_AUDIO` forces the mp4 AAC path. Autodetect would open
    // an mp4 as video, and decoding that track *inside* the video session
    // stranded the full PCM in the bump arena so Continue could not allocate
    // a DPB picture (`display 2 failed: Declined { status: 4 }`).
    let force_audio = b.meta0 == KIND_AUDIO;
    match Session::open(input, force_audio) {
        Ok(sess) => {
            sess.report_geometry(b);
            b.heap_want = sess.heap_want();
            let done = sess.is_exhausted();
            // SAFETY: Init just dropped any previous session.
            unsafe {
                *core::ptr::addr_of_mut!(SESSION) = Some(Box::new(sess));
            }
            if done { ChunkStatus::Done } else { ChunkStatus::NeedMore }
        }
        Err(_) => ChunkStatus::Failed,
    }
}

pub fn cont(max_units: u64, b: &mut ChunkBlock) -> ChunkStatus {
    // SAFETY: Continue after a successful Init; single-threaded.
    let sess = unsafe { (*core::ptr::addr_of_mut!(SESSION)).as_mut() };
    match sess {
        Some(s) => {
            if crate::take_oom() {
                s.recover_oom();
            }
            s.step(max_units, b)
        }
        None => ChunkStatus::Failed,
    }
}

pub fn seek(target: u64, b: &mut ChunkBlock) -> ChunkStatus {
    let sess = unsafe { (*core::ptr::addr_of_mut!(SESSION)).as_mut() };
    match sess {
        Some(s) => {
            if crate::take_oom() {
                s.recover_oom();
            }
            match s.as_mut() {
                Session::Video(v) => v.seek_to(target as usize, b),
                _ => ChunkStatus::Failed,
            }
        }
        None => ChunkStatus::Failed,
    }
}

enum Session {
    Audio(AudioSess),
    Video(VideoSess),
}

impl Session {
    fn open(input: &[u8], force_audio: bool) -> Result<Self, &'static str> {
        if looks_wav(input) {
            Ok(Session::Audio(AudioSess::open_wav(input)?))
        } else if is_adts(input) {
            Ok(Session::Audio(AudioSess::open_aac(input)?))
        } else if looks_mp3(input) {
            Ok(Session::Audio(AudioSess::open_mp3(input)?))
        } else if force_audio {
            Ok(Session::Audio(AudioSess::open_mp4(input)?))
        } else {
            Ok(Session::Video(VideoSess::open(input)?))
        }
    }

    fn report_geometry(&self, b: &mut ChunkBlock) {
        match self {
            Session::Audio(a) => {
                b.kind = KIND_AUDIO;
                b.meta0 = a.rate() as u64;
                b.meta1 = a.channels() as u64;
            }
            Session::Video(v) => {
                b.kind = KIND_VIDEO;
                b.meta0 = v.w as u64;
                b.meta1 = v.h as u64;
                b.meta2 = v.display.len() as u64;
                b.meta3 = v.duration_ms;
                b.output_ptr = v.pts_bytes.as_ptr() as u64;
                b.output_len = v.pts_bytes.len() as u64;
            }
        }
    }

    fn heap_want(&self) -> u64 {
        match self {
            Session::Audio(a) => heap_guess(a.input_len()),
            Session::Video(v) => video_arena_bytes(v.w, v.h, v.display.len(), v.bytes.len()),
        }
    }

    fn recover_oom(&mut self) {
        if let Session::Video(v) = self {
            let _ = v.rebuild_from_shown();
        }
    }

    fn is_exhausted(&self) -> bool {
        match self {
            Session::Audio(a) => a.is_exhausted(),
            Session::Video(v) => v.display.is_empty(),
        }
    }

    fn step(&mut self, max_units: u64, b: &mut ChunkBlock) -> ChunkStatus {
        match self {
            Session::Audio(a) => a.step(max_units, b),
            Session::Video(v) => v.step(max_units, b),
        }
    }
}

fn looks_wav(b: &[u8]) -> bool {
    b.len() >= 12 && &b[0..4] == b"RIFF" && &b[8..12] == b"WAVE"
}

fn looks_mp3(b: &[u8]) -> bool {
    b.len() >= 3 && (&b[0..3] == b"ID3" || (b[0] == 0xff && b[1] & 0xe0 == 0xe0))
}

fn is_adts(bytes: &[u8]) -> bool {
    let n = bytes.len().min(64);
    let mut i = 0;
    while i + 2 < n {
        if bytes[i] == 0xff && (bytes[i + 1] & 0xf0) == 0xf0 {
            let layer = (bytes[i + 1] >> 1) & 0x03;
            if layer == 0 {
                return true;
            }
        }
        i += 1;
    }
    false
}

// --- audio ----------------------------------------------------------------

enum AudioSess {
    Wav(WavSess),
    Mp3(Mp3Sess),
    Aac(AacSess),
    Mp4(Mp4AacSess),
}

impl AudioSess {
    fn rate(&self) -> u32 {
        match self {
            AudioSess::Wav(s) => s.rate,
            AudioSess::Mp3(s) => s.rate,
            AudioSess::Aac(s) => s.rate,
            AudioSess::Mp4(s) => s.rate,
        }
    }
    fn channels(&self) -> u8 {
        match self {
            AudioSess::Wav(s) => s.out_ch,
            AudioSess::Mp3(s) => s.channels,
            AudioSess::Aac(_) => 1,
            AudioSess::Mp4(_) => 1,
        }
    }
    fn input_len(&self) -> usize {
        match self {
            AudioSess::Wav(s) => s.bytes.len(),
            AudioSess::Mp3(s) => s.bytes.len(),
            AudioSess::Aac(s) => s.bytes.len(),
            AudioSess::Mp4(s) => s.pcm.len().saturating_mul(2),
        }
    }
    fn is_exhausted(&self) -> bool {
        match self {
            AudioSess::Wav(s) => s.remaining_frames() == 0,
            AudioSess::Mp3(s) => s.at >= s.bytes.len(),
            AudioSess::Aac(s) => s.pos + 7 >= s.bytes.len(),
            AudioSess::Mp4(s) => s.pos >= s.pcm.len(),
        }
    }
    fn step(&mut self, max_units: u64, b: &mut ChunkBlock) -> ChunkStatus {
        match self {
            AudioSess::Wav(s) => s.step(max_units, b),
            AudioSess::Mp3(s) => s.step(max_units, b),
            AudioSess::Aac(s) => s.step(max_units, b),
            AudioSess::Mp4(s) => s.step(max_units, b),
        }
    }

    fn open_wav(input: &[u8]) -> Result<Self, &'static str> {
        Ok(AudioSess::Wav(WavSess::parse(input)?))
    }
    fn open_mp3(input: &[u8]) -> Result<Self, &'static str> {
        Ok(AudioSess::Mp3(Mp3Sess::parse(input)?))
    }
    fn open_aac(input: &[u8]) -> Result<Self, &'static str> {
        Ok(AudioSess::Aac(AacSess::parse(input)?))
    }
    fn open_mp4(input: &[u8]) -> Result<Self, &'static str> {
        Ok(AudioSess::Mp4(Mp4AacSess::parse(input)?))
    }
}

fn report_pcm(b: &mut ChunkBlock, pcm: &[i16], rate: u32, channels: u8, units: u64) {
    b.kind = KIND_AUDIO;
    b.meta0 = rate as u64;
    b.meta1 = channels as u64;
    b.output_ptr = pcm.as_ptr() as u64;
    b.output_len = (pcm.len() * 2) as u64;
    b.units_done = b.units_done.saturating_add(units);
}

// -- WAV --

const FMT_PCM: u16 = 1;
const FMT_FLOAT: u16 = 3;
const FMT_EXTENSIBLE: u16 = 0xFFFE;

struct WavSess {
    bytes: Vec<u8>,
    data_off: usize,
    data_end: usize,
    cursor: usize,
    format: u16,
    bits: usize,
    in_ch: usize,
    out_ch: u8,
    rate: u32,
    bytes_per: usize,
    out: Vec<i16>,
}

impl WavSess {
    fn parse(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
            return Err("wav");
        }
        let mut pos = 12usize;
        let mut format = 0u16;
        let mut channels = 0usize;
        let mut rate = 0u32;
        let mut bits = 0usize;
        let mut data_off = 0usize;
        let mut data_end = 0usize;
        while pos + 8 <= bytes.len() {
            let id = &bytes[pos..pos + 4];
            let sz = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap_or([0; 4])) as usize;
            let body_off = pos + 8;
            let body_end = (body_off + sz).min(bytes.len());
            match id {
                b"fmt " if body_end.saturating_sub(body_off) >= 16 => {
                    let body = &bytes[body_off..body_end];
                    format = u16::from_le_bytes([body[0], body[1]]);
                    channels = u16::from_le_bytes([body[2], body[3]]) as usize;
                    rate = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
                    bits = u16::from_le_bytes([body[14], body[15]]) as usize;
                    if format == FMT_EXTENSIBLE && body.len() >= 26 {
                        format = u16::from_le_bytes([body[24], body[25]]);
                    }
                }
                b"data" => {
                    data_off = body_off;
                    data_end = body_end;
                }
                _ => {}
            }
            pos += 8 + sz + (sz & 1);
        }
        if data_end <= data_off {
            return Err("wav: no data");
        }
        if channels == 0 || rate == 0 || rate > 384_000 {
            return Err("wav: bad fmt");
        }
        let bytes_per = bits / 8;
        if !matches!((format, bits), (FMT_PCM, 8 | 16 | 24 | 32) | (FMT_FLOAT, 32)) {
            return Err("wav: unsupported format");
        }
        let out_ch: u8 = if channels == 1 { 1 } else { 2 };
        Ok(Self {
            bytes: input_copy(bytes),
            data_off,
            data_end,
            cursor: data_off,
            format,
            bits,
            in_ch: channels,
            out_ch,
            rate,
            bytes_per,
            out: Vec::new(),
        })
    }

    fn remaining_frames(&self) -> usize {
        let frame = self.bytes_per.saturating_mul(self.in_ch).max(1);
        self.data_end.saturating_sub(self.cursor) / frame
    }

    fn sample_from(s: &[u8], format: u16, bits: usize) -> i32 {
        match (format, bits) {
            (FMT_PCM, 8) => (s[0] as i32 - 128) << 8,
            (FMT_PCM, 16) => i16::from_le_bytes([s[0], s[1]]) as i32,
            (FMT_PCM, 24) => (i32::from_le_bytes([0, s[0], s[1], s[2]]) >> 8) >> 8,
            (FMT_PCM, 32) => i32::from_le_bytes([s[0], s[1], s[2], s[3]]) >> 16,
            (FMT_FLOAT, 32) => {
                let x = f32::from_bits(u32::from_le_bytes([s[0], s[1], s[2], s[3]]));
                (x.clamp(-1.0, 1.0) * 32767.0) as i32
            }
            _ => 0,
        }
    }

    fn step(&mut self, max_units: u64, b: &mut ChunkBlock) -> ChunkStatus {
        let want = if max_units == 0 { 4096 } else { max_units as usize };
        let n = self.remaining_frames().min(want);
        self.out.clear();
        if n == 0 {
            report_pcm(b, &self.out, self.rate, self.out_ch, 0);
            return ChunkStatus::Done;
        }
        let frame = self.bytes_per * self.in_ch;
        let bp = self.bytes_per;
        let fmt = self.format;
        let bits = self.bits;
        let in_ch = self.in_ch;
        let out_ch = self.out_ch;
        self.out.reserve(n * out_ch as usize);
        for _ in 0..n {
            let at = self.cursor;
            let mut tmp = [0i32; 8];
            let nch = in_ch.min(8);
            for c in 0..nch {
                let off = at + c * bp;
                tmp[c] = Self::sample_from(&self.bytes[off..off + bp], fmt, bits);
            }
            match (in_ch, out_ch) {
                (1, _) => self.out.push(tmp[0].clamp(-32768, 32767) as i16),
                (2, _) => {
                    self.out.push(tmp[0].clamp(-32768, 32767) as i16);
                    self.out.push(tmp[1].clamp(-32768, 32767) as i16);
                }
                _ => {
                    let (mut l, mut r, mut nl, mut nr) = (0i64, 0i64, 0i64, 0i64);
                    for c in 0..nch {
                        let v = tmp[c] as i64;
                        if c % 2 == 0 {
                            l += v;
                            nl += 1;
                        } else {
                            r += v;
                            nr += 1;
                        }
                    }
                    self.out.push((l / nl.max(1)).clamp(-32768, 32767) as i16);
                    self.out.push((r / nr.max(1)).clamp(-32768, 32767) as i16);
                }
            }
            self.cursor += frame;
        }
        report_pcm(b, &self.out, self.rate, self.out_ch, n as u64);
        if self.remaining_frames() == 0 { ChunkStatus::Done } else { ChunkStatus::NeedMore }
    }
}

// -- MP3 --

struct Mp3Sess {
    bytes: Vec<u8>,
    at: usize,
    dec: Box<audio::mp3::Mp3Dec>,
    scratch: Box<audio::mp3::Scratch>,
    rate: u32,
    channels: u8,
    out: Vec<i16>,
}

impl Mp3Sess {
    fn parse(input: &[u8]) -> Result<Self, &'static str> {
        let bytes = input_copy(input);
        let mut at = 0usize;
        if bytes.len() > 10 && &bytes[0..3] == b"ID3" {
            let sz = ((bytes[6] as usize & 0x7f) << 21)
                | ((bytes[7] as usize & 0x7f) << 14)
                | ((bytes[8] as usize & 0x7f) << 7)
                | (bytes[9] as usize & 0x7f);
            at = (10 + sz).min(bytes.len());
        }
        let mut dec = audio::mp3::Mp3Dec::new();
        let mut scratch = audio::mp3::new_scratch();
        let mut pcm = [0i16; audio::mp3::MAX_SAMPLES_PER_FRAME];
        let mut rate = 0u32;
        let mut channels = 0u8;
        let start = at;
        while at < bytes.len() {
            let mut info = audio::mp3::FrameInfo { frame_bytes: 0, channels: 0, hz: 0, layer: 0 };
            let samples = audio::mp3::decode_frame(&mut dec, &mut scratch, &bytes[at..], &mut pcm, &mut info);
            if info.frame_bytes == 0 {
                break;
            }
            at += info.frame_bytes;
            if info.hz != 0 {
                rate = info.hz;
                channels = info.channels.max(1) as u8;
                let _ = samples;
                break;
            }
        }
        if rate == 0 {
            return Err("mp3: no frames");
        }
        // Rewind to the first payload byte: Init must not consume frames.
        // `decode_frame` mutated `dec`; start a fresh decoder for Continue.
        Ok(Self {
            at: start,
            bytes,
            dec: audio::mp3::Mp3Dec::new(),
            scratch: audio::mp3::new_scratch(),
            rate,
            channels,
            out: Vec::new(),
        })
    }

    fn step(&mut self, max_units: u64, b: &mut ChunkBlock) -> ChunkStatus {
        let want = if max_units == 0 { 8 } else { max_units as usize };
        self.out.clear();
        let mut pcm = [0i16; audio::mp3::MAX_SAMPLES_PER_FRAME];
        let mut units = 0usize;
        while units < want && self.at < self.bytes.len() {
            let mut info = audio::mp3::FrameInfo { frame_bytes: 0, channels: 0, hz: 0, layer: 0 };
            let samples = audio::mp3::decode_frame(
                &mut self.dec,
                &mut self.scratch,
                &self.bytes[self.at..],
                &mut pcm,
                &mut info,
            );
            if info.frame_bytes == 0 {
                break;
            }
            self.at += info.frame_bytes;
            if samples == 0 {
                continue;
            }
            if self.rate != 0 && info.hz != 0 && info.hz != self.rate {
                break;
            }
            let nch = info.channels.max(1) as u8;
            if self.channels != 0 && nch != self.channels {
                break;
            }
            if self.rate == 0 {
                self.rate = info.hz;
            }
            if self.channels == 0 {
                self.channels = nch;
            }
            self.out.extend_from_slice(&pcm[..samples * nch as usize]);
            units += 1;
        }
        report_pcm(b, &self.out, self.rate, self.channels, units as u64);
        if self.at >= self.bytes.len() || units == 0 && self.out.is_empty() {
            if self.out.is_empty() && units == 0 {
                ChunkStatus::Done
            } else if self.at >= self.bytes.len() {
                ChunkStatus::Done
            } else {
                ChunkStatus::NeedMore
            }
        } else {
            ChunkStatus::NeedMore
        }
    }
}

// -- AAC --

struct AacSess {
    bytes: Vec<u8>,
    pos: usize,
    dec: Box<audio::aac::Decoder>,
    rate: u32,
    out: Vec<i16>,
}

impl AacSess {
    fn parse(input: &[u8]) -> Result<Self, &'static str> {
        let bytes = input_copy(input);
        let start = audio::aac::adts::find_sync(&bytes).ok_or("adts: no syncword")?;
        let (hdr, frame_len, info) = audio::aac::adts::parse_header(&bytes[start..])?;
        let _ = (hdr, frame_len);
        let asc_bytes = info.to_asc_bytes();
        let mut asc = audio::aac::parse_asc(&asc_bytes).unwrap_or(audio::aac::Asc {
            sample_rate: info.sample_rate,
            output_sample_rate: info.sample_rate,
            channels: if info.channels == 0 { 2 } else { info.channels },
            frame_length: 1024,
            sbr: false,
            ps: false,
            aot: info.aot,
            need_pce: info.channels == 0,
        });
        asc.aot = info.aot;
        asc.sample_rate = info.sample_rate;
        asc.output_sample_rate = info.sample_rate;
        if info.channels != 0 {
            asc.channels = info.channels;
        } else {
            asc.need_pce = true;
        }
        let rate = asc.output_rate();
        let dec = Box::new(audio::aac::Decoder::new(&asc));
        Ok(Self { bytes, pos: start, dec, rate, out: Vec::new() })
    }

    fn step(&mut self, max_units: u64, b: &mut ChunkBlock) -> ChunkStatus {
        let want = if max_units == 0 { 32 } else { max_units as usize };
        self.out.clear();
        let mut units = 0usize;
        while units < want && self.pos + 7 < self.bytes.len() {
            if self.bytes[self.pos] != 0xff || (self.bytes[self.pos + 1] & 0xf0) != 0xf0 {
                match audio::aac::adts::find_sync(&self.bytes[self.pos..]) {
                    Some(o) => self.pos += o,
                    None => break,
                }
            }
            let (hdr, frame_len, _info) = match audio::aac::adts::parse_header(&self.bytes[self.pos..]) {
                Ok(v) => v,
                Err(_) => {
                    self.pos += 1;
                    continue;
                }
            };
            if self.pos + frame_len > self.bytes.len() {
                break;
            }
            let payload = &self.bytes[self.pos + hdr..self.pos + frame_len];
            self.pos += frame_len;
            match self.dec.decode_raw(payload) {
                Ok(frame) => self.out.extend_from_slice(&frame),
                Err(_) => {
                    let n = self.dec.frame_samples();
                    self.out.extend(core::iter::repeat(0i16).take(n));
                }
            }
            units += 1;
        }
        report_pcm(b, &self.out, self.rate, 1, units as u64);
        if self.pos + 7 >= self.bytes.len() { ChunkStatus::Done } else { ChunkStatus::NeedMore }
    }
}

/// AAC inside mp4/mov. Decoded in a **throwaway audio session**, never inside
/// the video tenant — that bump arena has to stay free for the DPB.
struct Mp4AacSess {
    pcm: Vec<i16>,
    rate: u32,
    pos: usize,
    out: Vec<i16>,
}

impl Mp4AacSess {
    fn parse(input: &[u8]) -> Result<Self, &'static str> {
        let track = video::mp4::parse_audio(input)?.ok_or("aac: no AAC track")?;
        let pairs: Vec<(usize, usize)> = track.samples.iter().map(|s| (s.offset, s.size)).collect();
        let a = audio::aac::decode_track(
            track.sample_rate,
            track.channels,
            &track.asc,
            input,
            &pairs,
        )?;
        Ok(Self {
            pcm: a.pcm,
            rate: a.rate,
            pos: 0,
            out: Vec::new(),
        })
    }

    fn step(&mut self, max_units: u64, b: &mut ChunkBlock) -> ChunkStatus {
        let want = if max_units == 0 { 4096 } else { max_units as usize };
        let n = self.pcm.len().saturating_sub(self.pos).min(want);
        self.out.clear();
        if n == 0 {
            report_pcm(b, &self.out, self.rate, 1, 0);
            return ChunkStatus::Done;
        }
        self.out.extend_from_slice(&self.pcm[self.pos..self.pos + n]);
        self.pos += n;
        report_pcm(b, &self.out, self.rate, 1, n as u64);
        if self.pos >= self.pcm.len() {
            ChunkStatus::Done
        } else {
            ChunkStatus::NeedMore
        }
    }
}

fn input_copy(input: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(input.len());
    v.extend_from_slice(input);
    v
}

// --- video ----------------------------------------------------------------

enum Engine {
    RustH264 {
        dec: rust_h264::decoder::Decoder,
        pending: BTreeMap<usize, video::h264::decoder::DecodedFrame>,
        open_sample: Option<usize>,
    },
    Cavlc {
        reference: Option<video::h264::decoder::DecodedFrame>,
        last: Option<usize>,
    },
    Cabac {
        dec: video::h264::decoder_cabac::H264Dec,
        pending: BTreeMap<usize, Rc<video::h264::decoder_cabac::Pic>>,
    },
    Hevc {
        dec: video::hevc::decoder::HevcDecoder,
        pending: BTreeMap<usize, video::hevc::decoder::DecodedFrame>,
        next_out: usize,
    },
    Vp9 {
        dec: video::vp9::decoder::Vp9Decoder,
        pending: BTreeMap<usize, video::vp9::decoder::DecodedFrame>,
    },
}

struct VideoSess {
    bytes: Vec<u8>,
    samples: Vec<video::mp4::Sample>,
    display: Vec<usize>,
    next: usize,
    next_display: usize,
    engine: Engine,
    w: u32,
    h: u32,
    length_size: u8,
    sps: video::h264::Sps,
    pps: video::h264::Pps,
    avcc_sps: Vec<Vec<u8>>,
    avcc_pps: Vec<Vec<u8>>,
    timescale: u32,
    duration_ms: u64,
    config: video::mp4::CodecConfig,
    pts_bytes: Vec<u8>,
    out_y: Vec<u8>,
    out_cb: Vec<u8>,
    out_cr: Vec<u8>,
}

impl VideoSess {
    fn open(input: &[u8]) -> Result<Self, &'static str> {
        if video::hls::looks_like_playlist(input) {
            return Err("video: HLS playlist");
        }
        if let Some(refs) = video::hls::unpack_ts_segments(input) {
            let mut tracks = Vec::new();
            for r in &refs {
                let body = input
                    .get(r.offset..r.offset + r.len)
                    .ok_or("video: HLS segment out of range")?;
                let mut track = video::ts::demux_video(body)?;
                track.discontinuity = r.discontinuity;
                tracks.push(track);
            }
            let (buf, config, samples, ts) = video::ts::assemble_samples(&tracks)?;
            let duration_ms = ts_duration_ms(&samples, ts);
            return Self::from_parts(buf, config, samples, ts, duration_ms);
        }
        if is_mpegts(input) {
            let track = video::ts::demux_video(input)?;
            let (buf, config, samples, ts) = video::ts::assemble_samples(&[track])?;
            let duration_ms = ts_duration_ms(&samples, ts);
            return Self::from_parts(buf, config, samples, ts, duration_ms);
        }
        if looks_mp4(input) {
            let t = video::mp4::parse(input)?;
            let duration_ms = t.duration_ms();
            return Self::from_parts(input_copy(input), t.config, t.samples, t.timescale, duration_ms);
        }
        if input.starts_with(&[0x1a, 0x45, 0xdf, 0xa3]) {
            let t = video::mkv::parse(input)?;
            let duration_ms = t.duration_ms();
            return Self::from_parts(input_copy(input), t.config, t.samples, t.timescale, duration_ms);
        }
        Err("video: unrecognised container")
    }

    fn from_parts(
        bytes: Vec<u8>,
        config: video::mp4::CodecConfig,
        samples: Vec<video::mp4::Sample>,
        timescale: u32,
        duration_ms: u64,
    ) -> Result<Self, &'static str> {
        if samples.is_empty() {
            return Err("video: no samples");
        }
        let mut display: Vec<usize> = (0..samples.len()).collect();
        display.sort_by_key(|&i| (samples[i].cts, i));
        let pts_bytes = pack_pts(&samples, &display, timescale);
        let stored_config = config.clone();
        match config {
            video::mp4::CodecConfig::Avc(avcc) => {
                let sps_nal = avcc.sps.first().ok_or("video: no SPS")?;
                let pps_nal = avcc.pps.first().ok_or("video: no PPS")?;
                let sps = video::h264::parse_sps(&video::bits::unescape_rbsp(&sps_nal[1..]))?;
                let pps = video::h264::parse_pps(&video::bits::unescape_rbsp(&pps_nal[1..]))?;
                let (w, h) = (sps.width(), sps.height());
                // Native CABAC/CAVLC only. rust_h264 accepts a High SPS then
                // fails a later AU; the in-kernel player falls back, but a
                // mid-stream fallback in this bump arena leaks the DPB and
                // every following Continue returns Failed (flower.mp4 frame 0
                // played, frames 1..149 died in 50 ms).
                let engine = if pps.entropy_coding_mode {
                    Engine::Cabac {
                        dec: video::h264::decoder_cabac::H264Dec::new(sps.clone(), pps.clone())?,
                        pending: BTreeMap::new(),
                    }
                } else {
                    Engine::Cavlc {
                        reference: None,
                        last: None,
                    }
                };
                Ok(Self {
                    length_size: avcc.length_size,
                    avcc_sps: avcc.sps,
                    avcc_pps: avcc.pps,
                    bytes,
                    samples,
                    display,
                    next: 0,
                    next_display: 0,
                    engine,
                    w,
                    h,
                    sps,
                    pps,
                    timescale,
                    duration_ms,
                    config: stored_config,
                    pts_bytes,
                    out_y: Vec::new(),
                    out_cb: Vec::new(),
                    out_cr: Vec::new(),
                })
            }
            video::mp4::CodecConfig::Hevc(hvcc) => {
                let sps_nal = hvcc.sps.first().ok_or("video: no HEVC SPS")?;
                let sps = video::hevc::parse_sps(&video::bits::unescape_rbsp(&sps_nal[2..]))?;
                let mut dec = video::hevc::decoder::HevcDecoder::new();
                dec.set_parameter_sets(&hvcc.vps, &hvcc.sps, &hvcc.pps)?;
                Ok(Self {
                    w: sps.width(),
                    h: sps.height(),
                    length_size: hvcc.length_size,
                    bytes,
                    samples,
                    display,
                    next: 0,
                    next_display: 0,
                    engine: Engine::Hevc { dec, pending: BTreeMap::new(), next_out: 0 },
                    sps: video::h264::Sps::default(),
                    pps: video::h264::Pps::default(),
                    avcc_sps: Vec::new(),
                    avcc_pps: Vec::new(),
                    timescale,
                    duration_ms,
                    config: stored_config,
                    pts_bytes,
                    out_y: Vec::new(),
                    out_cb: Vec::new(),
                    out_cr: Vec::new(),
                })
            }
            video::mp4::CodecConfig::Vp9(_) => {
                let first = samples.first().ok_or("video: no samples")?;
                let data = bytes
                    .get(first.offset..first.offset + first.size)
                    .ok_or("video: sample out of range")?;
                let ranges = video::vp9::split_superframe(data)?;
                let refs: video::vp9::RefSizes = [(0, 0); video::vp9::NUM_REF_FRAMES];
                let hdr = video::vp9::parse_frame_header(&data[ranges[0].0..ranges[0].1], &refs)?;
                Ok(Self {
                    w: hdr.render_width,
                    h: hdr.render_height,
                    length_size: 0,
                    bytes,
                    samples,
                    display,
                    next: 0,
                    next_display: 0,
                    engine: Engine::Vp9 {
                        dec: video::vp9::decoder::Vp9Decoder::new(),
                        pending: BTreeMap::new(),
                    },
                    sps: video::h264::Sps::default(),
                    pps: video::h264::Pps::default(),
                    avcc_sps: Vec::new(),
                    avcc_pps: Vec::new(),
                    timescale,
                    duration_ms,
                    config: stored_config,
                    pts_bytes,
                    out_y: Vec::new(),
                    out_cb: Vec::new(),
                    out_cr: Vec::new(),
                })
            }
        }
    }

    fn step(&mut self, max_units: u64, b: &mut ChunkBlock) -> ChunkStatus {
        let want = if max_units == 0 { 1 } else { max_units as usize };
        if self.next_display >= self.display.len() {
            return ChunkStatus::Done;
        }
        // One display frame per unit. Planes stay in `out_*` until the next
        // Continue (kernel copies between entries).
        let idx = self.next_display;
        if !self.ensure_frame(idx) {
            return ChunkStatus::Failed;
        }
        if !self.copy_yuv(idx) {
            return ChunkStatus::Failed;
        }
        self.next_display += 1;
        // Extra units: keep decoding into pending but only report one plane set
        // (the ABI has one Y/Cb/Cr triple). The extra `want` is a hint for
        // prefetch; we honour 1 displayed frame per entry unless want is 1.
        let _ = want;
        self.report_yuv(b);
        b.units_done = b.units_done.saturating_add(1);
        b.state_cookie = self.pts_of(idx);
        if self.next_display >= self.display.len() { ChunkStatus::Done } else { ChunkStatus::NeedMore }
    }

    fn pts_of(&self, disp: usize) -> u64 {
        let Some(&si) = self.display.get(disp) else { return 0 };
        let Some(s) = self.samples.get(si) else { return 0 };
        if self.timescale == 0 { 0 } else { s.cts * 1000 / self.timescale as u64 }
    }

    fn seek_to(&mut self, target: usize, b: &mut ChunkBlock) -> ChunkStatus {
        if target >= self.display.len() {
            return ChunkStatus::Failed;
        }
        // Only `next_display` is the picture we have shown. rust_h264 (and CABAC)
        // decode one sample ahead so the previous picture can be emitted; `next`
        // is therefore routinely `next_display + 1`. Treating that as "go
        // backwards" rewound the engine on every sequential Seek, re-decoded
        // the clip from 0 with IRQs masked, and froze `/open flower.mp4`.
        if target < self.next_display {
            if self.rewind().is_err() {
                return ChunkStatus::Failed;
            }
        }
        while self.next_display <= target {
            if !self.ensure_frame(self.next_display) {
                return ChunkStatus::Failed;
            }
            if self.next_display == target {
                if !self.copy_yuv(target) {
                    return ChunkStatus::Failed;
                }
            }
            self.next_display += 1;
        }
        self.report_yuv(b);
        b.units_done = (target + 1) as u64;
        b.state_cookie = self.pts_of(target);
        if self.next_display >= self.display.len() { ChunkStatus::Done } else { ChunkStatus::NeedMore }
    }

    fn rewind(&mut self) -> Result<(), &'static str> {
        let bytes = core::mem::take(&mut self.bytes);
        let config = self.config.clone();
        let samples = core::mem::take(&mut self.samples);
        let ts = self.timescale;
        let dur = self.duration_ms;
        *self = Self::from_parts(bytes, config, samples, ts, dur)?;
        Ok(())
    }

    /// An OOM abort leaves CABAC mid-AU. Rebuild the engine and catch up to
    /// the last *shown* display index so the next Continue is the same picture.
    fn rebuild_from_shown(&mut self) -> Result<(), &'static str> {
        let shown = self.next_display;
        self.rewind()?;
        while self.next_display < shown {
            if !self.ensure_frame(self.next_display) {
                return Err("video: recover after OOM");
            }
            self.next_display += 1;
        }
        Ok(())
    }

    fn report_yuv(&self, b: &mut ChunkBlock) {
        b.kind = KIND_VIDEO;
        b.output_ptr = self.out_y.as_ptr() as u64;
        b.output_len = self.out_y.len() as u64;
        b.meta0 = self.w as u64;
        b.meta1 = self.h as u64;
        b.meta2 = self.out_cb.as_ptr() as u64;
        b.meta3 = self.out_cr.as_ptr() as u64;
        b.meta4 = self.out_y.len() as u64;
        b.meta5 = self.out_cb.len() as u64;
        b.meta6 = self.out_cr.len() as u64;
    }

    fn copy_yuv(&mut self, idx: usize) -> bool {
        let target = self.display[idx];
        match &self.engine {
            Engine::RustH264 { pending, .. } => {
                let Some(df) = pending.get(&target) else { return false };
                self.out_y.clear();
                self.out_y.extend_from_slice(&df.y);
                self.out_cb.clear();
                self.out_cb.extend_from_slice(&df.cb);
                self.out_cr.clear();
                self.out_cr.extend_from_slice(&df.cr);
                self.w = df.w as u32;
                self.h = df.h as u32;
                true
            }
            Engine::Cavlc { reference, last } => {
                if *last != Some(target) {
                    return false;
                }
                let Some(df) = reference.as_ref() else { return false };
                self.out_y.clear();
                self.out_y.extend_from_slice(&df.y);
                self.out_cb.clear();
                self.out_cb.extend_from_slice(&df.cb);
                self.out_cr.clear();
                self.out_cr.extend_from_slice(&df.cr);
                self.w = df.w as u32;
                self.h = df.h as u32;
                true
            }
            Engine::Cabac { pending, .. } => {
                let Some(pic) = pending.get(&target) else { return false };
                self.out_y.clear();
                self.out_y.extend_from_slice(&pic.f.y);
                self.out_cb.clear();
                self.out_cb.extend_from_slice(&pic.f.cb);
                self.out_cr.clear();
                self.out_cr.extend_from_slice(&pic.f.cr);
                self.w = pic.f.w as u32;
                self.h = pic.f.h as u32;
                true
            }
            Engine::Hevc { pending, .. } => {
                let Some(f) = pending.get(&target) else { return false };
                hevc_to_i420(f, &mut self.out_y, &mut self.out_cb, &mut self.out_cr);
                self.w = f.w as u32;
                self.h = f.h as u32;
                true
            }
            Engine::Vp9 { pending, .. } => {
                let Some(f) = pending.get(&target) else { return false };
                self.out_y.clear();
                self.out_y.extend_from_slice(&f.y);
                self.out_cb.clear();
                self.out_cb.extend_from_slice(&f.cb);
                self.out_cr.clear();
                self.out_cr.extend_from_slice(&f.cr);
                self.w = f.w as u32;
                self.h = f.h as u32;
                true
            }
        }
    }

    fn ensure_frame(&mut self, idx: usize) -> bool {
        let target = self.display[idx];
        while self.next <= target {
            if self.decode_one().is_err() {
                // rust_h264 accepts a High SPS then fails the first CABAC slice.
                // The in-kernel player falls back to native CABAC and retries;
                // without that, `/open flower.mp4` Init'd then produced no frames.
                if !self.fallback_rust_h264(target) {
                    return false;
                }
            }
        }
        match &self.engine {
            Engine::RustH264 { pending, .. } if pending.contains_key(&target) => true,
            Engine::RustH264 { .. } => {
                self.ensure_rust_h264(target);
                if matches!(&self.engine, Engine::RustH264 { pending, .. } if pending.contains_key(&target)) {
                    true
                } else if self.fallback_rust_h264(target) {
                    self.ensure_frame(idx)
                } else {
                    false
                }
            }
            Engine::Hevc { pending, .. } if pending.contains_key(&target) => true,
            Engine::Hevc { .. } => {
                self.ensure_hevc(target);
                matches!(&self.engine, Engine::Hevc { pending, .. } if pending.contains_key(&target))
            }
            Engine::Vp9 { pending, .. } => pending.contains_key(&target),
            Engine::Cabac { pending, .. } => pending.contains_key(&target),
            Engine::Cavlc { last, .. } => *last == Some(target),
        }
    }

    /// Drop rust_h264 and rebuild the native engine from the keyframe ≤ `target`.
    /// Same shape as the in-kernel player's `fallback_to_native`.
    fn fallback_rust_h264(&mut self, target: usize) -> bool {
        if !matches!(self.engine, Engine::RustH264 { .. }) {
            return false;
        }
        self.engine = if self.pps.entropy_coding_mode {
            match video::h264::decoder_cabac::H264Dec::new(self.sps.clone(), self.pps.clone()) {
                Ok(dec) => Engine::Cabac {
                    dec,
                    pending: BTreeMap::new(),
                },
                Err(_) => Engine::Cavlc {
                    reference: None,
                    last: None,
                },
            }
        } else {
            Engine::Cavlc {
                reference: None,
                last: None,
            }
        };
        let mut s = target.min(self.samples.len().saturating_sub(1));
        while s > 0 && !self.samples[s].is_sync {
            s -= 1;
        }
        self.next = s;
        true
    }

    fn ensure_rust_h264(&mut self, si: usize) {
        if matches!(&self.engine, Engine::RustH264 { pending, .. } if pending.contains_key(&si)) {
            return;
        }
        if self.next < self.samples.len() {
            let _ = self.decode_one();
            return;
        }
        if let Engine::RustH264 { dec, pending, open_sample } = &mut self.engine {
            if let Some(frame) = dec.flush() {
                if let Some(prev) = *open_sample {
                    pending.insert(prev, yuv_from_rust(frame));
                }
            }
            *open_sample = None;
        }
    }

    fn ensure_hevc(&mut self, si: usize) {
        loop {
            if matches!(&self.engine, Engine::Hevc { pending, .. } if pending.contains_key(&si)) {
                return;
            }
            if self.next < self.samples.len() {
                if self.decode_one().is_err() {
                    return;
                }
                continue;
            }
            if let Engine::Hevc { dec, pending, next_out } = &mut self.engine {
                for f in dec.flush() {
                    if *next_out < self.display.len() {
                        pending.insert(self.display[*next_out], f);
                        *next_out += 1;
                    }
                }
            }
            return;
        }
    }

    fn decode_one(&mut self) -> Result<(), &'static str> {
        let sample_i = self.next;
        let s = self.samples[sample_i];
        if s.offset + s.size > self.bytes.len() {
            return Err("video: sample out of range");
        }
        let data = &self.bytes[s.offset..s.offset + s.size];
        match &mut self.engine {
            Engine::RustH264 { dec, pending, open_sample } => {
                let nals = rust_h264::nal::parse_avcc(data, self.length_size as usize);
                if nals.is_empty() {
                    return Err("video: no NALs");
                }
                for nal in &nals {
                    match dec.decode_nal(nal) {
                        Ok(Some(frame)) => {
                            if let Some(prev) = *open_sample {
                                pending.insert(prev, yuv_from_rust(frame));
                            }
                        }
                        Ok(None) => {}
                        Err(_) => return Err("rust_h264: decode_nal failed"),
                    }
                }
                *open_sample = Some(sample_i);
                while pending.len() > 24 {
                    let k = *pending.keys().next().unwrap();
                    pending.remove(&k);
                }
            }
            Engine::Hevc { dec, pending, next_out } => {
                let nals = video::hevc::split_hvcc(data, self.length_size);
                if nals.is_empty() {
                    return Err("video: no NALs");
                }
                let out = dec.decode_au(&nals)?;
                for f in out {
                    if *next_out < self.display.len() {
                        pending.insert(self.display[*next_out], f);
                        *next_out += 1;
                    }
                }
                while pending.len() > 16 {
                    let k = *pending.keys().next().unwrap();
                    pending.remove(&k);
                }
            }
            Engine::Vp9 { dec, pending } => {
                for &(a, b) in &video::vp9::split_superframe(data)? {
                    if let Some(f) = dec.decode_frame(&data[a..b])? {
                        pending.insert(sample_i, f);
                    }
                }
                while pending.len() > 8 {
                    let k = *pending.keys().next().unwrap();
                    pending.remove(&k);
                }
            }
            Engine::Cavlc { reference, last } => {
                let mut slices: Vec<(Vec<u8>, bool)> = Vec::new();
                for nal in video::h264::split_avcc(data, self.length_size) {
                    if nal.kind.is_slice() {
                        slices.push((nal.rbsp(), nal.kind == video::h264::NalType::SliceIdr));
                    }
                }
                if slices.is_empty() {
                    return Err("video: no slices");
                }
                let df = video::h264::decoder::decode_access_unit(
                    &self.sps,
                    &self.pps,
                    &slices,
                    reference.as_ref(),
                )?;
                *reference = Some(df);
                *last = Some(sample_i);
            }
            Engine::Cabac { dec, pending } => {
                let mut slices: Vec<(Vec<u8>, bool, u8)> = Vec::new();
                for nal in video::h264::split_avcc(data, self.length_size) {
                    if nal.kind.is_slice() {
                        slices.push((nal.rbsp(), nal.kind == video::h264::NalType::SliceIdr, nal.ref_idc));
                    }
                }
                if slices.is_empty() {
                    return Err("video: no slices");
                }
                let pic = dec.decode_au(&slices)?;
                pending.insert(sample_i, pic);
                while pending.len() > 20 {
                    let k = *pending.keys().next().unwrap();
                    pending.remove(&k);
                }
            }
        }
        self.next = sample_i + 1;
        Ok(())
    }
}

fn pack_pts(samples: &[video::mp4::Sample], display: &[usize], timescale: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(display.len() * 8);
    for &si in display {
        let ms = if timescale == 0 {
            0
        } else {
            samples.get(si).map(|s| s.cts * 1000 / timescale as u64).unwrap_or(0)
        };
        v.extend_from_slice(&ms.to_le_bytes());
    }
    v
}

fn ts_duration_ms(samples: &[video::mp4::Sample], timescale: u32) -> u64 {
    if timescale == 0 {
        return 0;
    }
    samples.last().map(|s| s.cts * 1000 / timescale as u64).unwrap_or(0)
}

#[allow(dead_code)]
fn rust_h264_from_avcc(
    sps_nals: &[Vec<u8>],
    pps_nals: &[Vec<u8>],
) -> Result<rust_h264::decoder::Decoder, &'static str> {
    let mut dec = rust_h264::decoder::Decoder::new();
    for raw in sps_nals.iter().chain(pps_nals.iter()) {
        let nal = rust_h264::nal::parse_nal_bytes(raw).ok_or("rust_h264: bad param NAL")?;
        dec.decode_nal(&nal).map_err(|_| "rust_h264: param set rejected")?;
    }
    Ok(dec)
}

fn yuv_from_rust(f: rust_h264::decoder::Frame) -> video::h264::decoder::DecodedFrame {
    video::h264::decoder::DecodedFrame { w: f.width as usize, h: f.height as usize, y: f.y, cb: f.u, cr: f.v }
}

fn hevc_to_i420(
    f: &video::hevc::decoder::DecodedFrame,
    y: &mut Vec<u8>,
    cb: &mut Vec<u8>,
    cr: &mut Vec<u8>,
) {
    let shift = f.bit_depth.saturating_sub(8);
    y.clear();
    y.resize(f.y.len(), 0);
    for (d, &s) in y.iter_mut().zip(f.y.iter()) {
        *d = (s >> shift) as u8;
    }
    let (cw, ch) = (f.w / 2, f.h / 2);
    cb.clear();
    cr.clear();
    cb.resize(cw * ch, 128);
    cr.resize(cw * ch, 128);
    if f.chroma_format_idc != 0 && f.cw > 0 && f.ch > 0 {
        if f.chroma_format_idc == 1 && f.cw == cw && f.ch == ch && shift == 0 {
            for (d, &s) in cb.iter_mut().zip(f.cb.iter()) {
                *d = s as u8;
            }
            for (d, &s) in cr.iter_mut().zip(f.cr.iter()) {
                *d = s as u8;
            }
        } else {
            for j in 0..ch {
                let sj = match f.chroma_format_idc {
                    1 => j,
                    _ => j * 2,
                };
                if sj >= f.ch {
                    break;
                }
                for i in 0..cw {
                    let si = if f.chroma_format_idc == 3 { i * 2 } else { i };
                    if si >= f.cw {
                        break;
                    }
                    cb[j * cw + i] = (f.cb[sj * f.cw + si] >> shift) as u8;
                    cr[j * cw + i] = (f.cr[sj * f.cw + si] >> shift) as u8;
                }
            }
        }
    }
}

fn looks_mp4(bytes: &[u8]) -> bool {
    if bytes.len() < 12 {
        return false;
    }
    let typ = &bytes[4..8];
    typ == b"ftyp" || typ == b"moov" || typ == b"mdat" || typ == b"free" || typ == b"skip"
}

fn is_mpegts(bytes: &[u8]) -> bool {
    if bytes.len() < 188 || bytes[0] != 0x47 {
        return false;
    }
    (1..4)
        .map(|i| i * 188)
        .take_while(|&at| at < bytes.len())
        .all(|at| bytes[at] == 0x47)
}
