//! AAC decoder for raw MPEG-4 access units (mp4/mp4a) and ADTS (`.aac`).
//!
//! Port of the [Symphonia](https://github.com/pdeljanov/Symphonia) AAC path
//! (itself a NihAV port) under **MPL-2.0**, plus ADTS demux, PCE channel
//! config 0, Main/LTP/SSR ICS side-info, and (when present) HE-AAC SBR/PS
//! reconstruction. See [`THIRDPARTY-LICENSES.md`](../../../../../THIRDPARTY-LICENSES.md).
//!
//! Pure over its inputs: ASC/ADTS + raw AUs → mono S16 PCM. Malformed input
//! returns `Err` rather than panicking.

pub mod adts;
mod bits;
mod codebooks;
mod cpe;
mod dsp;
mod ics;
mod imdct;
mod ltp;
pub mod sbr;
mod tables;
mod window;

use alloc::vec::Vec;
use bits::BitReader;
use codebooks::DequantTables;
use cpe::ChannelPair;
use dsp::Dsp;
use sbr::{IdSynEle, SbrState};
use tables::{GASubbandInfo, SAMPLE_RATES};

/// Parsed MPEG-4 `AudioSpecificConfig` (the bytes in the mp4 `esds` box).
#[derive(Clone, Debug)]
pub struct Asc {
    /// Core AAC sample rate (filterbank / MDCT).
    pub sample_rate: u32,
    /// Output sample rate after SBR (equals `sample_rate` when SBR is off).
    pub output_sample_rate: u32,
    pub channels: u8,
    /// Samples per channel per **core** frame (1024).
    pub frame_length: usize,
    /// True when the ASC signals SBR (HE-AAC).
    pub sbr: bool,
    /// True when PS (HE-AACv2) is signalled.
    pub ps: bool,
    /// MPEG-4 audio object type of the **core** (1=Main, 2=LC, 3=SSR, 4=LTP).
    pub aot: u8,
    /// Channel config 0 → first raw_data_block must carry a PCE.
    pub need_pce: bool,
}

impl Asc {
    /// Rate to use for PCM playback / `Audio.rate`.
    pub fn output_rate(&self) -> u32 {
        if self.output_sample_rate != 0 {
            self.output_sample_rate
        } else {
            self.sample_rate
        }
    }
}

fn read_aot(br: &mut BitReader<'_>) -> Result<u32, &'static str> {
    let mut aot = br.read_bits(5)?;
    if aot == 31 {
        aot = 32 + br.read_bits(6)?;
    }
    Ok(aot)
}

fn read_sampling_frequency(br: &mut BitReader<'_>) -> Result<u32, &'static str> {
    let freq_idx = br.read_bits(4)?;
    if freq_idx == 0x0f {
        let r = br.read_bits(24)?;
        if r == 0 {
            return Err("aac: sample rate 0");
        }
        Ok(r)
    } else if (freq_idx as usize) < SAMPLE_RATES.len() {
        Ok(SAMPLE_RATES[freq_idx as usize])
    } else {
        Err("aac: bad sample rate index")
    }
}

/// MPEG-4 channelConfiguration → speaker count (table 1.19).
/// Returns `Ok(0)` for config 0 (PCE in-band).
fn channels_from_config(cfg: u8) -> Result<u8, &'static str> {
    match cfg {
        0 => Ok(0),  // PCE
        1 => Ok(1),  // C
        2 => Ok(2),  // L R
        3 => Ok(3),  // C L R
        4 => Ok(4),  // C L R Cs
        5 => Ok(5),  // C L R Ls Rs
        6 => Ok(6),  // 5.1
        7 => Ok(8),  // 7.1
        _ => Err("aac: unsupported channel config"),
    }
}

/// Parse an `AudioSpecificConfig` bitstring (ISO/IEC 14496-3).
///
/// Accepts Main / LC / SSR / LTP cores and HE-AAC / HE-AACv2 wrappers.
pub fn parse_asc(asc: &[u8]) -> Result<Asc, &'static str> {
    if asc.len() < 2 {
        return Err("aac: ASC too short");
    }
    let mut br = BitReader::new(asc);
    let mut aot = read_aot(&mut br)?;
    let mut sample_rate = read_sampling_frequency(&mut br)?;
    let mut output_sample_rate = sample_rate;
    let chan_cfg = br.read_bits(4)? as u8;
    let mut channels = channels_from_config(chan_cfg)?;
    let need_pce = channels == 0;
    let mut sbr = false;
    let mut ps = false;

    // AOT 5 = SBR, 29 = PS: outer rate is SBR output; next = core rate + AOT.
    if aot == 5 || aot == 29 {
        sbr = true;
        ps = aot == 29;
        output_sample_rate = sample_rate;
        let core_rate = read_sampling_frequency(&mut br)?;
        aot = read_aot(&mut br)?;
        sample_rate = core_rate;
    }

    // Core AOT: Main(1), LC(2), SSR(3), LTP(4), ER AAC LC(17).
    if !matches!(aot, 1 | 2 | 3 | 4 | 17) {
        return Err("aac: unsupported core audio object type");
    }

    // GASpecificConfig (Main/LC/SSR/LTP/ER-LC)
    let short_frame = br.read_bool()?;
    let frame_length = if short_frame { 960 } else { 1024 };
    if frame_length != 1024 {
        return Err("aac: only 1024-sample frames");
    }
    let depends_on_core = br.read_bool()?;
    if depends_on_core {
        let _ = br.read_bits(14)?;
    }
    let extension_flag = br.read_bool()?;
    // channelConfiguration == 0 → program_config_element() follows in ASC
    // (ISO 14496-3 Table 1.15 / GASpecificConfig). Parse it for channel count.
    if need_pce {
        if let Ok(n) = parse_pce_channel_count(&mut br) {
            channels = n;
        }
    }
    if extension_flag {
        if aot == 22 {
            // ER BSAC
            let _ = br.read_bits(5)?;
            let _ = br.read_bits(11)?;
        }
        if matches!(aot, 17 | 19 | 20 | 23) {
            let _ = br.read_bool()?;
            let _ = br.read_bool()?;
            let _ = br.read_bool()?;
        }
        let extension_flag3 = br.read_bool()?;
        if extension_flag3 {
            return Err("aac: ASC version3 extensions unsupported");
        }
    }

    // Trailing SBR/PS extension on an AOT≠5 outer ASC.
    if br.bits_left() >= 16 {
        let sync = br.peek(11).unwrap_or(0);
        if sync == 0x2b7 {
            let _ = br.read_bits(11)?;
            let ext_aot = read_aot(&mut br)?;
            if ext_aot == 5 {
                sbr = br.read_bool()?;
                if sbr {
                    output_sample_rate = read_sampling_frequency(&mut br)?;
                    if br.bits_left() >= 12 {
                        let sync2 = br.peek(11).unwrap_or(0);
                        if sync2 == 0x548 {
                            let _ = br.read_bits(11)?;
                            ps = br.read_bool()?;
                        }
                    }
                }
            } else if ext_aot == 29 {
                sbr = true;
                ps = true;
                let _ = br.read_bool()?;
                output_sample_rate = read_sampling_frequency(&mut br)?;
            }
        }
    }

    if !sbr {
        output_sample_rate = sample_rate;
    } else if output_sample_rate == sample_rate {
        // Implicit dual-rate SBR: output is 2× core when not stated otherwise.
        output_sample_rate = sample_rate.saturating_mul(2);
    }

    if channels == 0 {
        // Still waiting for an in-stream PCE (ADTS/raw without ASC-embedded PCE).
        channels = 2; // provisional; first PCE updates Decoder
    }

    Ok(Asc {
        sample_rate,
        output_sample_rate,
        channels,
        frame_length,
        sbr,
        ps,
        aot: aot as u8,
        need_pce,
    })
}

/// Parse a Program Config Element enough to recover the channel count.
/// Leaves the bit reader after the PCE (including comment field).
fn parse_pce_channel_count(bs: &mut BitReader<'_>) -> Result<u8, &'static str> {
    let _tag = bs.read_bits(4)?;
    let _obj = bs.read_bits(2)?;
    let _sf = bs.read_bits(4)?;
    let num_front = bs.read_bits(4)? as usize;
    let num_side = bs.read_bits(4)? as usize;
    let num_back = bs.read_bits(4)? as usize;
    let num_lfe = bs.read_bits(2)? as usize;
    let num_assoc = bs.read_bits(3)? as usize;
    let num_valid_cc = bs.read_bits(4)? as usize;
    if bs.read_bool()? {
        let _ = bs.read_bits(4)?;
    }
    if bs.read_bool()? {
        let _ = bs.read_bits(4)?;
    }
    if bs.read_bool()? {
        let _ = bs.read_bits(3)?;
    }
    let mut nch = 0u8;
    for _ in 0..num_front {
        let is_cpe = bs.read_bool()?;
        let _ = bs.read_bits(4)?;
        nch = nch.saturating_add(if is_cpe { 2 } else { 1 });
    }
    for _ in 0..num_side {
        let is_cpe = bs.read_bool()?;
        let _ = bs.read_bits(4)?;
        nch = nch.saturating_add(if is_cpe { 2 } else { 1 });
    }
    for _ in 0..num_back {
        let is_cpe = bs.read_bool()?;
        let _ = bs.read_bits(4)?;
        nch = nch.saturating_add(if is_cpe { 2 } else { 1 });
    }
    for _ in 0..num_lfe {
        let _ = bs.read_bits(4)?;
        nch = nch.saturating_add(1);
    }
    for _ in 0..num_assoc {
        let _ = bs.read_bits(4)?;
    }
    for _ in 0..num_valid_cc {
        let _ = bs.read_bool()?;
        let _ = bs.read_bits(4)?;
    }
    bs.realign();
    let comment_bytes = bs.read_bits(8)? as usize;
    bs.skip((comment_bytes as u32).saturating_mul(8))?;
    if nch == 0 {
        Err("aac: empty PCE")
    } else {
        Ok(nch.min(8))
    }
}

/// Stateful AAC decoder (overlap buffers + DSP tables + optional LTP/SBR).
pub struct Decoder {
    asc: Asc,
    pairs: Vec<ChannelPair>,
    dsp: Dsp,
    tables: DequantTables,
    sbinfo: GASubbandInfo,
    /// Planar float output scratch: [ch][1024]
    planes: Vec<Vec<f32>>,
    ltp: Vec<ltp::LtpState>,
    /// Last SCE/CPE prediction side info (per pair, left/right).
    last_pred: Vec<(ltp::PredSide, ltp::PredSide)>,
    /// HE-AAC SBR (+ PS) state when ASC signals SBR.
    sbr: Option<SbrState>,
}

/// Strip an ADTS header if present (syncword 0xFFF). Returns the raw AU body.
fn strip_adts(au: &[u8]) -> Result<&[u8], &'static str> {
    if au.len() < 7 {
        return Ok(au);
    }
    // ADTS: 12-bit sync 0xFFF
    if au[0] == 0xff && (au[1] & 0xf0) == 0xf0 {
        let protection_absent = (au[1] & 0x01) != 0;
        let hdr = if protection_absent { 7 } else { 9 };
        if au.len() < hdr {
            return Err("aac: truncated ADTS header");
        }
        // frame_length is 13 bits spanning bytes 3–5
        let frame_len =
            (((au[3] as usize) & 0x03) << 11) | ((au[4] as usize) << 3) | ((au[5] as usize) >> 5);
        if frame_len < hdr || frame_len > au.len() {
            // Trust the sync and return body; some muxers lie about length.
            return Ok(&au[hdr..]);
        }
        return Ok(&au[hdr..frame_len]);
    }
    Ok(au)
}

impl Decoder {
    pub fn new(asc: &Asc) -> Self {
        let sbinfo = GASubbandInfo::find(asc.sample_rate);
        let nch = (asc.channels as usize).max(1).min(8);
        let mut planes = Vec::with_capacity(nch);
        let mut ltp = Vec::with_capacity(nch);
        for _ in 0..nch {
            planes.push(alloc::vec![0.0f32; 1024]);
            ltp.push(ltp::LtpState::new());
        }
        let sbr_ch = if asc.channels >= 2 { 2 } else { 1 };
        let sbr = if asc.sbr {
            Some(SbrState::new(asc.sample_rate, sbr_ch))
        } else {
            None
        };
        Self {
            asc: asc.clone(),
            pairs: Vec::new(),
            dsp: Dsp::new(),
            tables: DequantTables::new(),
            sbinfo,
            planes,
            ltp,
            last_pred: Vec::new(),
            sbr,
        }
    }

    pub fn reset(&mut self) {
        for p in self.pairs.iter_mut() {
            p.reset();
        }
        for s in self.ltp.iter_mut() {
            s.reset();
        }
        if self.asc.sbr {
            let sbr_ch = if self.asc.channels >= 2 { 2 } else { 1 };
            self.sbr = Some(SbrState::new(self.asc.sample_rate, sbr_ch));
        }
    }

    pub fn output_rate(&self) -> u32 {
        self.asc.output_rate()
    }

    /// Samples per mono output frame (1024 core; 2048 when SBR doubles rate).
    pub fn frame_samples(&self) -> usize {
        if self.asc.sbr && self.asc.output_sample_rate > self.asc.sample_rate {
            self.asc.frame_length * 2
        } else {
            self.asc.frame_length
        }
    }

    /// Ensure pair slot exists; recreate if layout changed (tolerant of odd streams).
    fn set_pair(&mut self, pair_no: usize, channel: usize, pair: bool) -> Result<(), &'static str> {
        while self.pairs.len() <= pair_no {
            self.pairs
                .push(ChannelPair::new(pair, channel, self.sbinfo));
        }
        if self.pairs[pair_no].channel != channel || self.pairs[pair_no].is_pair != pair {
            self.pairs[pair_no] = ChannelPair::new(pair, channel, self.sbinfo);
        }
        let need = if pair { channel + 2 } else { channel + 1 };
        while self.planes.len() < need {
            self.planes.push(alloc::vec![0.0f32; 1024]);
            self.ltp.push(ltp::LtpState::new());
        }
        if need as u8 > self.asc.channels {
            self.asc.channels = need as u8;
        }
        Ok(())
    }

    /// Decode one raw AAC access unit → mono S16 PCM.
    pub fn decode_raw(&mut self, au: &[u8]) -> Result<Vec<i16>, &'static str> {
        let au = strip_adts(au)?;
        if au.is_empty() {
            return Err("aac: empty access unit");
        }
        for p in self.planes.iter_mut() {
            p.fill(0.0);
        }
        self.last_pred.clear();

        let mut bs = BitReader::new(au);
        let mut cur_pair = 0usize;
        let mut cur_ch = 0usize;
        let aot = self.asc.aot;
        let mut last_id_aac = IdSynEle::Sce;
        let mut sbr_payload: Option<Vec<u8>> = None;
        let mut sbr_crc = false;

        while bs.bits_left() > 3 {
            let id = bs.read_bits(3)?;
            match id {
                0 => {
                    let _tag = bs.read_bits(4)?;
                    self.set_pair(cur_pair, cur_ch, false)?;
                    let (p0, p1) =
                        self.pairs[cur_pair].decode_ga_sce(&mut bs, &self.tables, aot)?;
                    self.last_pred.push((p0, p1));
                    last_id_aac = IdSynEle::Sce;
                    cur_pair += 1;
                    cur_ch += 1;
                }
                1 => {
                    let _tag = bs.read_bits(4)?;
                    self.set_pair(cur_pair, cur_ch, true)?;
                    let (p0, p1) =
                        self.pairs[cur_pair].decode_ga_cpe(&mut bs, &self.tables, aot)?;
                    self.last_pred.push((p0, p1));
                    last_id_aac = IdSynEle::Cpe;
                    cur_pair += 1;
                    cur_ch += 2;
                }
                2 => {
                    skip_cce(&mut bs)?;
                }
                3 => {
                    let _tag = bs.read_bits(4)?;
                    self.set_pair(cur_pair, cur_ch, false)?;
                    let (p0, p1) =
                        self.pairs[cur_pair].decode_ga_sce(&mut bs, &self.tables, aot)?;
                    self.last_pred.push((p0, p1));
                    cur_pair += 1;
                    cur_ch += 1;
                }
                4 => {
                    let _id = bs.read_bits(4)?;
                    let align = bs.read_bool()?;
                    let mut count = bs.read_bits(8)?;
                    if count == 255 {
                        count += bs.read_bits(8)?;
                    }
                    if align {
                        bs.realign();
                    }
                    bs.skip(count.saturating_mul(8))?;
                }
                5 => {
                    // In-stream PCE — update channel count when config was 0.
                    if let Ok(n) = parse_pce_channel_count(&mut bs) {
                        self.asc.channels = n;
                        self.asc.need_pce = false;
                    }
                }
                6 => {
                    // ID_FIL — may carry EXT_SBR_DATA (13) / EXT_SBR_DATA_CRC (14).
                    let mut count = bs.read_bits(4)? as usize;
                    if count == 15 {
                        count += bs.read_bits(8)? as usize;
                        count = count.saturating_sub(1);
                    }
                    if count == 0 {
                        continue;
                    }
                    if bs.bits_left() >= 4 {
                        let ext_type = bs.read_bits(4)? as u8;
                        let body_bits = (count as u32).saturating_mul(8).saturating_sub(4);
                        if ext_type == 13 || ext_type == 14 {
                            // FIL SBR even without ASC sbrPresentFlag (implicit HE-AAC).
                            if self.sbr.is_none() {
                                let sbr_ch = if self.asc.channels >= 2 { 2 } else { 1 };
                                self.sbr = Some(SbrState::new(self.asc.sample_rate, sbr_ch));
                                self.asc.sbr = true;
                                if self.asc.output_sample_rate <= self.asc.sample_rate {
                                    self.asc.output_sample_rate =
                                        self.asc.sample_rate.saturating_mul(2);
                                }
                            }
                            let mut body =
                                Vec::with_capacity(((body_bits + 7) / 8) as usize);
                            let mut left = body_bits;
                            while left >= 8 {
                                body.push(bs.read_bits(8)? as u8);
                                left -= 8;
                            }
                            if left > 0 {
                                let v = bs.read_bits(left)? as u8;
                                body.push(v << (8 - left as u8));
                            }
                            sbr_payload = Some(body);
                            sbr_crc = ext_type == 14;
                        } else {
                            bs.skip(body_bits)?;
                        }
                    } else {
                        bs.skip((count as u32).saturating_mul(8))?;
                    }
                }
                7 => break,
                _ => break,
            }
        }

        let rate_idx = GASubbandInfo::find_idx(self.asc.sample_rate);
        for pair_i in 0..cur_pair {
            let ch0 = self.pairs[pair_i].channel;
            let is_pair = self.pairs[pair_i].is_pair;
            let (p0, p1) = self
                .last_pred
                .get(pair_i)
                .copied()
                .unwrap_or((ltp::PredSide::None, ltp::PredSide::None));
            if is_pair && ch0 + 1 < self.planes.len() {
                // Split ltp borrows carefully via split_at_mut
                let (left_ltp, rest) = self.ltp.split_at_mut(ch0 + 1);
                let right_ltp = &mut rest[0];
                let left = &mut self.planes[ch0][..];
                self.pairs[pair_i].synth_left(
                    &mut self.dsp,
                    rate_idx,
                    left,
                    Some(&mut left_ltp[ch0]),
                    p0,
                );
                let right = &mut self.planes[ch0 + 1][..];
                self.pairs[pair_i].synth_right(
                    &mut self.dsp,
                    rate_idx,
                    right,
                    Some(right_ltp),
                    p1,
                );
            } else if ch0 < self.planes.len() {
                let dst = &mut self.planes[ch0][..];
                let ltp_st = &mut self.ltp[ch0];
                self.pairs[pair_i].synth_left(
                    &mut self.dsp,
                    rate_idx,
                    dst,
                    Some(ltp_st),
                    p0,
                );
            }
        }

        let nch = cur_ch.max(self.asc.channels as usize);

        // HE-AAC: full SBR (+ optional PS) reconstruction at 2× rate.
        if let Some(ref mut sbr) = self.sbr {
            sbr.set_element_type(last_id_aac);
            let sbr_nch = if last_id_aac == IdSynEle::Cpe { 2 } else { 1 };
            if sbr_nch as u8 != sbr.channels() {
                let rate = sbr.core_rate();
                *sbr = SbrState::new(rate, sbr_nch as u8);
                sbr.set_element_type(last_id_aac);
            }
            let core: Vec<Vec<f32>> = (0..sbr_nch.min(self.planes.len()))
                .map(|i| {
                    let n = 1024.min(self.planes[i].len());
                    self.planes[i][..n].to_vec()
                })
                .collect();
            sbr.set_crc(sbr_crc);
            match sbr.process(&core, sbr_payload.as_deref(), self.asc.ps) {
                Ok(planes_sbr) => {
                    let n_out = planes_sbr.first().map(|p| p.len()).unwrap_or(2048);
                    return Ok(downmix_mono(&planes_sbr, n_out, planes_sbr.len()));
                }
                Err(_) => {
                    // Soft-fail: keep going with QMF upsample if process failed mid-stream.
                    let core_pcm = downmix_mono(&self.planes, self.asc.frame_length, nch);
                    return Ok(upsample_x2_mono(&core_pcm));
                }
            }
        }

        Ok(downmix_mono(&self.planes, self.asc.frame_length, nch))
    }
}

/// Fallback 2× ZOH upsample when SBR process soft-fails (keeps rate continuous).
fn upsample_x2_mono(core: &[i16]) -> Vec<i16> {
    let mut out = Vec::with_capacity(core.len() * 2);
    for &s in core {
        out.push(s);
        out.push(s);
    }
    out
}

/// ITU-R BS.775-ish mono downmix: average L/R (and C at −3 dB when present).
fn downmix_mono(planes: &[Vec<f32>], n: usize, nch: usize) -> Vec<i16> {
    let mut out = alloc::vec![0i16; n];
    let nch = nch.min(planes.len()).max(1);
    if nch == 1 {
        for i in 0..n {
            out[i] = float_to_i16(planes[0].get(i).copied().unwrap_or(0.0));
        }
        return out;
    }
    // Weights: L,R full; C 0.707; surrounds 0.5; LFE 0.5 — then normalize.
    let w: &[f32] = match nch {
        2 => &[1.0, 1.0],
        3 => &[0.707, 1.0, 1.0],             // C L R
        4 => &[0.707, 1.0, 1.0, 0.5],        // C L R Cs
        5 => &[0.707, 1.0, 1.0, 0.5, 0.5],  // C L R Ls Rs
        6 => &[0.707, 1.0, 1.0, 0.5, 0.5, 0.5], // 5.1
        _ => &[0.707, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5],
    };
    let mut wsum = 0.0f32;
    for i in 0..nch {
        wsum += w.get(i).copied().unwrap_or(0.5);
    }
    let inv = if wsum > 0.0 { 1.0 / wsum } else { 1.0 };
    for i in 0..n {
        let mut s = 0.0f32;
        for ch in 0..nch {
            s += planes[ch].get(i).copied().unwrap_or(0.0) * w.get(ch).copied().unwrap_or(0.5);
        }
        out[i] = float_to_i16(s * inv);
    }
    out
}

/// Best-effort skip of a coupling channel element (ISO 4.4.2.1).
fn skip_cce(bs: &mut BitReader<'_>) -> Result<(), &'static str> {
    let _tag = bs.read_bits(4)?;
    let ind_sw = bs.read_bool()?;
    let n_coupled = bs.read_bits(3)? as usize;
    for _ in 0..n_coupled {
        let is_cpe = bs.read_bool()?;
        let _ = bs.read_bits(4)?;
        if is_cpe {
            let _ = bs.read_bool()?;
        }
    }
    let _sign = bs.read_bool()?;
    let _ = bs.read_bits(2)?; // scale
    // Without full ICS skip we cannot safely parse the rest — consume remaining
    // bits of this element by failing soft: leave the bit reader; caller will
    // typically hit TERM or underrun. Prefer hard error so the frame is silence.
    let _ = ind_sw;
    Err("aac: coupling channel element skipped")
}

/// Skip a program config element (channel layout in-band).
#[allow(dead_code)] // kept as a simple full-skip when only draining bits
fn skip_pce(bs: &mut BitReader<'_>) -> Result<(), &'static str> {
    let _tag = bs.read_bits(4)?;
    let _ = bs.read_bits(2)?; // object type
    let _ = bs.read_bits(4)?; // sampling frequency index
    let num_front = bs.read_bits(4)? as usize;
    let num_side = bs.read_bits(4)? as usize;
    let num_back = bs.read_bits(4)? as usize;
    let num_lfe = bs.read_bits(2)? as usize;
    let num_assoc = bs.read_bits(3)? as usize;
    let num_valid_cc = bs.read_bits(4)? as usize;
    if bs.read_bool()? {
        let _ = bs.read_bits(4)?; // mono mixdown
    }
    if bs.read_bool()? {
        let _ = bs.read_bits(4)?; // stereo mixdown
    }
    if bs.read_bool()? {
        let _ = bs.read_bits(3)?; // matrix mixdown
    }
    for _ in 0..num_front {
        let _ = bs.read_bool()?;
        let _ = bs.read_bits(4)?;
    }
    for _ in 0..num_side {
        let _ = bs.read_bool()?;
        let _ = bs.read_bits(4)?;
    }
    for _ in 0..num_back {
        let _ = bs.read_bool()?;
        let _ = bs.read_bits(4)?;
    }
    for _ in 0..num_lfe {
        let _ = bs.read_bits(4)?;
    }
    for _ in 0..num_assoc {
        let _ = bs.read_bits(4)?;
    }
    for _ in 0..num_valid_cc {
        let _ = bs.read_bool()?;
        let _ = bs.read_bits(4)?;
    }
    bs.realign();
    // comment field
    let comment_bytes = bs.read_bits(8)? as usize;
    bs.skip((comment_bytes as u32).saturating_mul(8))?;
    Ok(())
}

#[inline]
fn float_to_i16(s: f32) -> i16 {
    let x = s * 32767.0;
    if x > 32767.0 {
        32767
    } else if x < -32768.0 {
        -32768
    } else {
        x as i16
    }
}

/// Decode all raw AUs described by `(offset, size)` into mono S16 [`crate::audio::Audio`].
pub fn decode_track(
    sample_rate: u32,
    channels: u8,
    asc_bytes: &[u8],
    data: &[u8],
    samples: &[(usize, usize)],
) -> Result<crate::audio::Audio, &'static str> {
    let mut asc = if asc_bytes.is_empty() {
        let r = if sample_rate != 0 { sample_rate } else { 44100 };
        Asc {
            sample_rate: r,
            output_sample_rate: r,
            channels: if channels == 0 { 2 } else { channels.min(8) },
            frame_length: 1024,
            sbr: false,
            ps: false,
            aot: 2,
            need_pce: channels == 0,
        }
    } else {
        parse_asc(asc_bytes)?
    };
    // Prefer ASC geometry; fall back to container fields when ASC omitted a rate.
    if asc.sample_rate == 0 && sample_rate != 0 {
        asc.sample_rate = sample_rate;
    }
    if channels > 0 && channels <= 8 && asc.channels == 0 {
        asc.channels = channels;
    }
    let mut dec = Decoder::new(&asc);
    let frame_out = dec.frame_samples();
    // Reserve roughly frame_out * n samples (mono) so we don't thrash the
    // first-fit heap on a multi-minute track.
    let mut pcm = Vec::with_capacity(samples.len().saturating_mul(frame_out));
    let mut ok_frames = 0usize;
    let mut err_frames = 0usize;
    for (i, &(start, size)) in samples.iter().enumerate() {
        // Cooperative kernel: keep the clock / mouse / net alive while a long
        // track decodes. Host harnesses stub `shell::upkeep`.
        if i & 31 == 0 {
            crate::shell::upkeep();
        }
        let end = start.saturating_add(size);
        if end > data.len() {
            return Err("aac: sample out of range");
        }
        if size == 0 {
            // Empty AU (encoder padding) → one frame of silence.
            pcm.extend(core::iter::repeat(0i16).take(frame_out));
            continue;
        }
        let au = &data[start..end];
        match dec.decode_raw(au) {
            Ok(frame) => {
                ok_frames += 1;
                pcm.extend_from_slice(&frame);
            }
            Err(_) => {
                err_frames += 1;
                // One bad frame → silence; keep going so a glitch doesn't kill
                // the whole track. If *nothing* decodes, fail below.
                pcm.extend(core::iter::repeat(0i16).take(frame_out));
            }
        }
    }
    if ok_frames == 0 && !samples.is_empty() {
        return Err("aac: no frames decoded");
    }
    let _ = err_frames;
    Ok(crate::audio::Audio {
        rate: asc.output_rate(),
        pcm,
    })
}

/// Convenience: decode a demuxed list of AU slices.
pub fn decode_aus(
    asc_bytes: &[u8],
    aus: &[&[u8]],
) -> Result<crate::audio::Audio, &'static str> {
    let asc = parse_asc(asc_bytes)?;
    let mut dec = Decoder::new(&asc);
    let mut pcm = Vec::new();
    for au in aus {
        let frame = dec.decode_raw(au)?;
        pcm.extend_from_slice(&frame);
    }
    Ok(crate::audio::Audio {
        rate: asc.output_rate(),
        pcm,
    })
}

/// Decode an ADTS `.aac` byte stream.
pub fn decode_adts(data: &[u8]) -> Result<crate::audio::Audio, &'static str> {
    adts::decode_file(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn parse_asc_44100_stereo() {
        // 0x12 0x10 → AOT=2 LC, freq_idx=4 (44100), chcfg=2, frame 1024
        let a = parse_asc(&[0x12, 0x10]).unwrap();
        assert_eq!(a.sample_rate, 44100);
        assert_eq!(a.channels, 2);
        assert_eq!(a.frame_length, 1024);
        assert!(!a.sbr);
    }

    #[test_case]
    fn parse_asc_mono_48k() {
        // AOT=2, freq_idx=3 (48000), chcfg=1, frame 1024
        // bits MSB-first: 00010 0011 0001 000 → 00010001 10001000 = 0x11 0x88
        // (the old 0x13 0x10 fixture mis-packed the 4-bit freq across the byte
        // boundary and actually decoded as 24 kHz / 2 ch).
        let a = parse_asc(&[0x11, 0x88]).unwrap();
        assert_eq!(a.sample_rate, 48000);
        assert_eq!(a.channels, 1);
    }

    #[test_case]
    fn parse_asc_accepts_main_profile() {
        // AOT=1 Main, 44100, stereo, 1024: 00001 0100 0010 000
        // bits: 00001010 00010000 → 0x0a 0x10
        let a = parse_asc(&[0x0a, 0x10]).unwrap();
        assert_eq!(a.aot, 1);
        assert_eq!(a.sample_rate, 44100);
        assert_eq!(a.channels, 2);
    }

    #[test_case]
    fn parse_asc_rejects_empty() {
        assert!(parse_asc(&[]).is_err());
    }

    #[test_case]
    fn strip_adts_sync() {
        // Minimal fake ADTS (protection absent) + 1 payload byte
        let mut adts = [0u8; 8];
        adts[0] = 0xff;
        adts[1] = 0xf1; // MPEG-4, layer 0, protection absent
        adts[2] = 0x50; // profile LC, 44100, etc. (bits don't matter for strip)
        adts[3] = 0x80;
        adts[4] = 0x01; // frame_length high bits → length 8
        adts[5] = 0x00; // with [3]&3 and [4] and [5]>>5 → craft carefully
        // Rebuild frame_length = 8: bits in [3:5]
        // frame_length = ((b3&3)<<11) | (b4<<3) | (b5>>5) = 8
        adts[3] = 0x00;
        adts[4] = 0x01; // 1<<3 = 8 when combined... (0<<11)|(1<<3)|(0>>5)=8
        adts[5] = 0x00;
        adts[6] = 0x00;
        adts[7] = 0xab;
        let body = strip_adts(&adts).unwrap();
        assert_eq!(body, &[0xab]);
    }

    #[test_case]
    fn decode_echo_fixture_head() {
        // Embedded head of /tmp/echo.mp4 audio track: ASC + first few AUs.
        // Format: u32le asc_len | asc | u32le n | (u32le size | au)...
        static FIX: &[u8] = include_bytes!("testdata_echo_head.bin");
        if FIX.len() < 8 {
            return;
        }
        let asc_len = u32::from_le_bytes(FIX[0..4].try_into().unwrap()) as usize;
        if 4 + asc_len + 4 > FIX.len() {
            return;
        }
        let asc = &FIX[4..4 + asc_len];
        let n = u32::from_le_bytes(FIX[4 + asc_len..8 + asc_len].try_into().unwrap()) as usize;
        let mut off = 8 + asc_len;
        let mut aus: Vec<&[u8]> = Vec::new();
        for _ in 0..n {
            if off + 4 > FIX.len() {
                break;
            }
            let sz = u32::from_le_bytes(FIX[off..off + 4].try_into().unwrap()) as usize;
            off += 4;
            if off + sz > FIX.len() {
                break;
            }
            aus.push(&FIX[off..off + sz]);
            off += sz;
        }
        assert!(!aus.is_empty(), "fixture has no AUs");
        let parsed = parse_asc(asc).expect("ASC");
        assert_eq!(parsed.sample_rate, 44100);
        let audio = decode_aus(asc, &aus).expect("decode");
        assert_eq!(audio.rate, 44100);
        assert_eq!(audio.pcm.len(), parsed.frame_length * aus.len());
        // Not all zeros after the (possibly silent) first frame.
        let tail = &audio.pcm[parsed.frame_length..];
        let energy: i64 = tail.iter().map(|&s| (s as i64).abs()).sum();
        assert!(
            energy > 0 || aus.len() == 1,
            "expected some energy in decoded PCM, energy={energy}"
        );
    }
}
