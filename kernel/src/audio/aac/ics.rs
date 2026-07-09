//! Individual channel stream: ICS info, scalefactors, spectrum, pulse, TNS.
//!
//! Ported from Symphonia AAC (NihAV origin, MPL-2.0). See THIRDPARTY-LICENSES.md.

use super::bits::BitReader;
use super::codebooks::{self, DequantTables};
use super::dsp::Dsp;
use super::tables::*;

const ZERO_HCB: u8 = 0;
const RESERVED_HCB: u8 = 12;
const NOISE_HCB: u8 = 13;
const INTENSITY_HCB2: u8 = 14;
const INTENSITY_HCB: u8 = 15;

const INTENSITY_SCALE_MIN: i16 = -155;
const NORMAL_SCALE_MIN: i16 = -100;

const TNS_MAX_ORDER: usize = 20;
const TNS_MAX_LONG_BANDS: [usize; 12] = [31, 31, 34, 40, 42, 51, 46, 46, 42, 42, 42, 39];
const TNS_MAX_SHORT_BANDS: [usize; 12] = [9, 9, 10, 14, 14, 14, 14, 14, 14, 14, 14, 14];

// --- ICS info ----------------------------------------------------------------

#[derive(Clone)]
pub struct IcsInfo {
    pub window_sequence: u8,
    pub prev_window_sequence: u8,
    pub window_shape: bool,
    pub prev_window_shape: bool,
    pub scale_factor_grouping: [bool; MAX_WINDOWS],
    pub group_start: [usize; MAX_WINDOWS],
    pub window_groups: usize,
    pub num_windows: usize,
    pub max_sfb: usize,
    pub long_win: bool,
}

impl IcsInfo {
    fn new() -> Self {
        Self {
            window_sequence: 0,
            prev_window_sequence: 0,
            window_shape: false,
            prev_window_shape: false,
            scale_factor_grouping: [false; MAX_WINDOWS],
            group_start: [0; MAX_WINDOWS],
            num_windows: 0,
            window_groups: 0,
            max_sfb: 0,
            long_win: true,
        }
    }

    /// Decode ICS info. `aot` selects Main/LC/SSR/LTP predictor side-info.
    /// Returns any Main/LTP prediction side info for the caller to apply.
    pub fn decode(
        &mut self,
        bs: &mut BitReader<'_>,
        aot: u8,
    ) -> Result<crate::audio::aac::ltp::PredSide, &'static str> {
        use crate::audio::aac::ltp::PredSide;
        self.prev_window_sequence = self.window_sequence;
        self.prev_window_shape = self.window_shape;

        if bs.read_bool()? {
            return Err("aac: ics reserved bit set");
        }

        self.window_sequence = bs.read_bits(2)? as u8;
        self.window_shape = bs.read_bool()?;
        self.window_groups = 1;

        let mut pred = PredSide::None;
        if self.window_sequence == EIGHT_SHORT_SEQUENCE {
            self.long_win = false;
            self.num_windows = 8;
            self.max_sfb = bs.read_bits(4)? as usize;
            for i in 0..MAX_WINDOWS - 1 {
                self.scale_factor_grouping[i] = bs.read_bool()?;
                if !self.scale_factor_grouping[i] {
                    self.group_start[self.window_groups] = i + 1;
                    self.window_groups += 1;
                }
            }
        } else {
            self.long_win = true;
            self.num_windows = 1;
            self.max_sfb = bs.read_bits(6)? as usize;
            // predictor_data_present / ltp_data_present
            let pred_present = bs.read_bool()?;
            if pred_present {
                match aot {
                    2 | 17 => {
                        // LC: must be zero — refuse corrupt streams
                        return Err("aac: LC stream has predictor_data_present");
                    }
                    1 => {
                        // Main: frequency-domain predictor data follows
                        pred = PredSide::read(bs, 1, self.max_sfb, true)?;
                    }
                    4 => {
                        // LTP: nested ltp_data_present already part of PredSide::read
                        // For AOT=4 the bit we just read IS ltp_data_present when
                        // not common_window — actually for LTP the bit is
                        // predictor_data_present style: if set, ltp_data() follows.
                        // Spec: for AOT LTP, if predictor_data_present, read ltp_data.
                        pred = PredSide::Ltp(crate::audio::aac::ltp::LtpData::read(
                            bs,
                            self.max_sfb,
                        )?);
                    }
                    3 => {
                        // SSR: no predictor_data in ics_info the same way; ignore.
                        pred = PredSide::None;
                    }
                    _ => {
                        // Best-effort skip Main-style flags
                        pred = PredSide::read(bs, 1, self.max_sfb, true).unwrap_or(PredSide::None);
                    }
                }
            }
        }
        Ok(pred)
    }

    /// LC-compatible decode (AOT=2): predictor bit must be 0.
    pub fn decode_lc(&mut self, bs: &mut BitReader<'_>) -> Result<(), &'static str> {
        let _ = self.decode(bs, 2)?;
        Ok(())
    }

    pub fn copy_from_common(&mut self, other: &IcsInfo) {
        let prev_window_sequence = self.window_sequence;
        let prev_window_shape = self.window_shape;
        *self = other.clone();
        self.prev_window_sequence = prev_window_sequence;
        self.prev_window_shape = prev_window_shape;
    }

    fn get_group_start(&self, g: usize) -> usize {
        if g == 0 {
            0
        } else if g >= self.window_groups {
            if self.long_win {
                1
            } else {
                8
            }
        } else {
            self.group_start[g]
        }
    }
}

// --- Pulse -------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Pulse {
    number_pulse: usize,
    pulse_start_sfb: usize,
    pulse_offset: [u8; 4],
    pulse_amp: [u8; 4],
}

impl Pulse {
    fn read(bs: &mut BitReader<'_>) -> Result<Option<Self>, &'static str> {
        if !bs.read_bool()? {
            return Ok(None);
        }
        let number_pulse = bs.read_bits(2)? as usize + 1;
        let pulse_start_sfb = bs.read_bits(6)? as usize;
        let mut pulse_offset = [0u8; 4];
        let mut pulse_amp = [0u8; 4];
        for i in 0..number_pulse {
            pulse_offset[i] = bs.read_bits(5)? as u8;
            pulse_amp[i] = bs.read_bits(4)? as u8;
        }
        Ok(Some(Self {
            number_pulse,
            pulse_start_sfb,
            pulse_offset,
            pulse_amp,
        }))
    }

    fn synth(
        &self,
        bands: &[usize],
        scales: &[[f32; MAX_SFBS]; MAX_WINDOWS],
        coeffs: &mut [f32; 1024],
    ) {
        if self.pulse_start_sfb >= bands.len().saturating_sub(1) {
            return;
        }
        let mut k = bands[self.pulse_start_sfb];
        let mut band = self.pulse_start_sfb;
        for pno in 0..self.number_pulse {
            k += self.pulse_offset[pno] as usize;
            if k >= 1024 {
                return;
            }
            while band + 1 < bands.len() && bands[band + 1] <= k {
                band += 1;
            }
            let scale = scales[0][band];
            let mut base = coeffs[k];
            if base != 0.0 {
                base = requant(coeffs[k], scale);
            }
            if base > 0.0 {
                base += f32::from(self.pulse_amp[pno]);
            } else {
                base -= f32::from(self.pulse_amp[pno]);
            }
            coeffs[k] = iquant_s(base) * scale;
        }
    }
}

#[inline]
fn iquant_s(val: f32) -> f32 {
    let a = if val < 0.0 { -val } else { val };
    let p = codebooks::powf_f32(a, 4.0 / 3.0);
    if val < 0.0 {
        -p
    } else {
        p
    }
}

#[inline]
fn requant(val: f32, scale: f32) -> f32 {
    if scale == 0.0 {
        return 0.0;
    }
    // Note: Symphonia's requant uses val not bval for pow — keep same.
    if val >= 0.0 {
        codebooks::powf_f32(val, 3.0 / 4.0)
    } else {
        -codebooks::powf_f32(-val, 3.0 / 4.0)
    }
}

// --- TNS ---------------------------------------------------------------------

#[derive(Copy, Clone)]
struct TnsCoeffs {
    length: usize,
    order: usize,
    direction: bool,
    coef: [f32; TNS_MAX_ORDER + 1],
}

impl TnsCoeffs {
    fn new() -> Self {
        Self {
            length: 0,
            order: 0,
            direction: false,
            coef: [0.0; TNS_MAX_ORDER + 1],
        }
    }

    fn read(
        &mut self,
        bs: &mut BitReader<'_>,
        long_win: bool,
        coef_res: bool,
        max_order: usize,
    ) -> Result<(), &'static str> {
        self.length = bs.read_bits(if long_win { 6 } else { 4 })? as usize;
        self.order = bs.read_bits(if long_win { 5 } else { 3 })? as usize;
        if self.order > max_order {
            return Err("aac: tns order too large");
        }
        if self.order > 0 {
            self.direction = bs.read_bool()?;
            let coef_compress = bs.read_bool()?;
            let mut coef_res_bits = if coef_res { 4 } else { 3 };
            if coef_compress {
                coef_res_bits -= 1;
            }
            let sign_mask = 1u8 << (coef_res_bits - 1);
            let neg_mask = !((1u8 << coef_res_bits) - 1);
            let fac_base = if coef_res { 8.0f32 } else { 4.0 };
            let iqfac = (fac_base - 0.5) / core::f32::consts::FRAC_PI_2;
            let iqfac_m = (fac_base + 0.5) / core::f32::consts::FRAC_PI_2;
            let mut tmp = [0.0f32; TNS_MAX_ORDER];
            for el in tmp[..self.order].iter_mut() {
                let val = bs.read_bits(coef_res_bits)? as u8;
                let c = f32::from(if (val & sign_mask) != 0 {
                    (val | neg_mask) as i8
                } else {
                    val as i8
                });
                *el = sinf_tns(if c >= 0.0 { c / iqfac } else { c / iqfac_m });
            }
            let mut b = [0.0f32; TNS_MAX_ORDER + 1];
            for m in 1..=self.order {
                for i in 1..m {
                    b[i] = self.coef[i - 1] + tmp[m - 1] * self.coef[m - i - 1];
                }
                self.coef[..(m - 1)].copy_from_slice(&b[1..m]);
                self.coef[m - 1] = tmp[m - 1];
            }
        }
        Ok(())
    }
}

fn sinf_tns(x: f32) -> f32 {
    crate::sound::mel::cosf_pub(x - core::f32::consts::FRAC_PI_2)
}

#[derive(Copy, Clone)]
struct Tns {
    n_filt: [usize; MAX_WINDOWS],
    coeffs: [[TnsCoeffs; 4]; MAX_WINDOWS],
}

impl Tns {
    fn read(bs: &mut BitReader<'_>, info: &IcsInfo, aot: u8) -> Result<Option<Self>, &'static str> {
        if !bs.read_bool()? {
            return Ok(None);
        }
        // Main allows order 20 on long windows; LC/SSR/LTP use 12.
        let max_order = if !info.long_win {
            7
        } else if aot == 1 {
            20
        } else {
            12
        };
        let mut n_filt = [0usize; MAX_WINDOWS];
        let mut coeffs = [[TnsCoeffs::new(); 4]; MAX_WINDOWS];
        for w in 0..info.num_windows {
            n_filt[w] = bs.read_bits(if info.long_win { 2 } else { 1 })? as usize;
            let coef_res = if n_filt[w] != 0 {
                bs.read_bool()?
            } else {
                false
            };
            for filt in 0..n_filt[w] {
                coeffs[w][filt].read(bs, info.long_win, coef_res, max_order)?;
            }
        }
        Ok(Some(Self { n_filt, coeffs }))
    }

    fn synth(&self, info: &IcsInfo, bands: &[usize], rate_idx: usize, coeffs: &mut [f32; 1024]) {
        let tns_max_bands = (if info.long_win {
            TNS_MAX_LONG_BANDS[rate_idx.min(11)]
        } else {
            TNS_MAX_SHORT_BANDS[rate_idx.min(11)]
        })
        .min(info.max_sfb);

        for w in 0..info.num_windows {
            let mut bottom = bands.len() - 1;
            for f in 0..self.n_filt[w] {
                let top = bottom;
                bottom = top.saturating_sub(self.coeffs[w][f].length);
                let order = self.coeffs[w][f].order;
                if order == 0 {
                    continue;
                }
                let start = w * 128 + bands[bottom.min(tns_max_bands)];
                let end = w * 128 + bands[top.min(tns_max_bands)];
                let lpc = &self.coeffs[w][f].coef;
                if !self.coeffs[w][f].direction {
                    for (m, i) in (start..end).enumerate() {
                        for j in 0..order.min(m) {
                            coeffs[i] -= coeffs[i - j - 1] * lpc[j];
                        }
                    }
                } else {
                    for (m, i) in (start..end).rev().enumerate() {
                        for j in 0..order.min(m) {
                            coeffs[i] -= coeffs[i + j + 1] * lpc[j];
                        }
                    }
                }
            }
        }
    }
}

// --- ICS ---------------------------------------------------------------------

#[derive(Clone)]
pub struct Ics {
    global_gain: u8,
    pub info: IcsInfo,
    pulse: Option<Pulse>,
    tns: Option<Tns>,
    sect_cb: [[u8; MAX_SFBS]; MAX_WINDOWS],
    sect_len: [[usize; MAX_SFBS]; MAX_WINDOWS],
    sfb_cb: [[u8; MAX_SFBS]; MAX_WINDOWS],
    num_sec: [usize; MAX_WINDOWS],
    pub scales: [[f32; MAX_SFBS]; MAX_WINDOWS],
    sbinfo: GASubbandInfo,
    pub coeffs: [f32; 1024],
    delay: [f32; 1024],
}

impl Ics {
    pub fn new(sbinfo: GASubbandInfo) -> Self {
        Self {
            global_gain: 0,
            info: IcsInfo::new(),
            pulse: None,
            tns: None,
            sect_cb: [[0; MAX_SFBS]; MAX_WINDOWS],
            sect_len: [[0; MAX_SFBS]; MAX_WINDOWS],
            sfb_cb: [[0; MAX_SFBS]; MAX_WINDOWS],
            scales: [[0.0; MAX_SFBS]; MAX_WINDOWS],
            num_sec: [0; MAX_WINDOWS],
            sbinfo,
            coeffs: [0.0; 1024],
            delay: [0.0; 1024],
        }
    }

    pub fn reset(&mut self) {
        self.info = IcsInfo::new();
        self.delay = [0.0; 1024];
    }

    pub fn decode_info(
        &mut self,
        bs: &mut BitReader<'_>,
        aot: u8,
    ) -> Result<crate::audio::aac::ltp::PredSide, &'static str> {
        let pred = self.info.decode(bs, aot)?;
        if self.info.max_sfb + 1 > self.get_bands().len() {
            return Err("aac: ics max_sfb too big");
        }
        Ok(pred)
    }

    fn decode_section_data(&mut self, bs: &mut BitReader<'_>) -> Result<(), &'static str> {
        let sect_bits = if self.info.long_win { 5 } else { 3 };
        let sect_esc_val = (1 << sect_bits) - 1;
        for g in 0..self.info.window_groups {
            let mut k = 0;
            let mut l = 0;
            while k < self.info.max_sfb {
                self.sect_cb[g][l] = bs.read_bits(4)? as u8;
                self.sect_len[g][l] = 0;
                if self.sect_cb[g][l] == RESERVED_HCB {
                    return Err("aac: invalid band type");
                }
                loop {
                    let sect_len_incr = bs.read_bits(sect_bits)? as usize;
                    self.sect_len[g][l] += sect_len_incr;
                    if sect_len_incr < sect_esc_val {
                        break;
                    }
                }
                if k + self.sect_len[g][l] > self.info.max_sfb {
                    return Err("aac: section overflow");
                }
                for sfb in k..k + self.sect_len[g][l] {
                    self.sfb_cb[g][sfb] = self.sect_cb[g][l];
                }
                k += self.sect_len[g][l];
                l += 1;
            }
            self.num_sec[g] = l;
        }
        Ok(())
    }

    #[inline]
    pub fn is_zero(&self, g: usize, sfb: usize) -> bool {
        self.sfb_cb[g][sfb] == ZERO_HCB
    }
    #[inline]
    pub fn is_intensity(&self, g: usize, sfb: usize) -> bool {
        self.sfb_cb[g][sfb] == INTENSITY_HCB || self.sfb_cb[g][sfb] == INTENSITY_HCB2
    }
    #[inline]
    pub fn is_noise(&self, g: usize, sfb: usize) -> bool {
        self.sfb_cb[g][sfb] == NOISE_HCB
    }
    #[inline]
    pub fn get_intensity_dir(&self, g: usize, sfb: usize) -> bool {
        self.sfb_cb[g][sfb] == INTENSITY_HCB
    }

    fn decode_scale_factor_data(
        &mut self,
        bs: &mut BitReader<'_>,
        tables: &DequantTables,
    ) -> Result<(), &'static str> {
        let mut noise_pcm_flag = true;
        let mut scf_intensity = -INTENSITY_SCALE_MIN;
        let mut scf_noise = i16::from(self.global_gain) - 90 - NORMAL_SCALE_MIN;
        let mut scf_normal = i16::from(self.global_gain);

        for g in 0..self.info.window_groups {
            for sfb in 0..self.info.max_sfb {
                self.scales[g][sfb] = if self.is_zero(g, sfb) {
                    0.0
                } else if self.is_intensity(g, sfb) {
                    scf_intensity += i16::from(tables.scf.decode(bs)? as u8) - 60;
                    if scf_intensity < 0 || scf_intensity >= 256 {
                        return Err("aac: intensity scf out of range");
                    }
                    tables.intensity_scf[scf_intensity as usize]
                } else if self.is_noise(g, sfb) {
                    if noise_pcm_flag {
                        noise_pcm_flag = false;
                        scf_noise += (bs.read_bits(9)? as i16) - 256;
                    } else {
                        scf_noise += i16::from(tables.scf.decode(bs)? as u8) - 60;
                    }
                    if scf_noise < 0 || scf_noise >= 256 {
                        return Err("aac: noise scf out of range");
                    }
                    tables.normal_scf[scf_noise as usize]
                } else {
                    scf_normal += i16::from(tables.scf.decode(bs)? as u8) - 60;
                    if scf_normal < 0 || scf_normal >= 256 {
                        return Err("aac: normal scf out of range");
                    }
                    tables.normal_scf[scf_normal as usize]
                };
            }
        }
        Ok(())
    }

    pub fn get_bands(&self) -> &'static [usize] {
        if self.info.long_win {
            self.sbinfo.long_bands
        } else {
            self.sbinfo.short_bands
        }
    }

    fn decode_spectrum(
        &mut self,
        bs: &mut BitReader<'_>,
        lcg: &mut Lcg,
        tables: &DequantTables,
    ) -> Result<(), &'static str> {
        self.coeffs.fill(0.0);
        let bands = self.get_bands();
        for g in 0..self.info.window_groups {
            let cur_w = self.info.get_group_start(g);
            let next_w = self.info.get_group_start(g + 1);
            for sfb in 0..self.info.max_sfb {
                let start = bands[sfb];
                let end = bands[sfb + 1];
                let cb_idx = self.sfb_cb[g][sfb];
                let scale = self.scales[g][sfb];
                for w in cur_w..next_w {
                    let dst = &mut self.coeffs[start + w * 128..end + w * 128];
                    match cb_idx {
                        ZERO_HCB => {}
                        RESERVED_HCB => {}
                        NOISE_HCB => decode_noise(lcg, scale, dst),
                        INTENSITY_HCB2 | INTENSITY_HCB => {}
                        1 => decode_quads_signed(bs, tables, 0, scale, dst)?,
                        2 => decode_quads_signed(bs, tables, 1, scale, dst)?,
                        3 => decode_quads_unsigned(bs, tables, 2, scale, dst)?,
                        4 => decode_quads_unsigned(bs, tables, 3, scale, dst)?,
                        5 => decode_pairs_signed(bs, tables, 0, scale, dst)?,
                        6 => decode_pairs_signed(bs, tables, 1, scale, dst)?,
                        7 => decode_pairs_unsigned(bs, tables, 2, scale, dst)?,
                        8 => decode_pairs_unsigned(bs, tables, 3, scale, dst)?,
                        9 => decode_pairs_unsigned(bs, tables, 4, scale, dst)?,
                        10 => decode_pairs_unsigned(bs, tables, 5, scale, dst)?,
                        11 => decode_pairs_escape(bs, tables, scale, dst)?,
                        _ => return Err("aac: unknown codebook"),
                    }
                }
            }
        }
        Ok(())
    }

    pub fn decode(
        &mut self,
        bs: &mut BitReader<'_>,
        lcg: &mut Lcg,
        tables: &DequantTables,
        common_window: bool,
        aot: u8,
        common_pred: Option<crate::audio::aac::ltp::PredSide>,
    ) -> Result<crate::audio::aac::ltp::PredSide, &'static str> {
        use crate::audio::aac::ltp::PredSide;
        self.global_gain = bs.read_bits(8)? as u8;
        let mut pred = common_pred.unwrap_or(PredSide::None);
        if !common_window {
            pred = self.decode_info(bs, aot)?;
        } else if matches!(aot, 4) {
            // Common window LTP: second channel may have its own ltp_data_present
            if bs.read_bool()? {
                pred = PredSide::Ltp(crate::audio::aac::ltp::LtpData::read(
                    bs,
                    self.info.max_sfb,
                )?);
            }
        }
        self.decode_section_data(bs)?;
        self.decode_scale_factor_data(bs, tables)?;
        self.pulse = Pulse::read(bs)?;
        if self.pulse.is_some() && !self.info.long_win {
            return Err("aac: pulse on short window");
        }
        // TNS max order is higher for Main (20) than LC (12)
        self.tns = Tns::read(bs, &self.info, aot)?;
        // gain_control_data_present — SSR uses it; skip payload if set.
        if bs.read_bool()? {
            skip_gain_control(bs, &self.info)?;
        }
        self.decode_spectrum(bs, lcg, tables)?;
        Ok(pred)
    }

    pub fn synth_channel(
        &mut self,
        dsp: &mut Dsp,
        rate_idx: usize,
        dst: &mut [f32],
        ltp_state: Option<&mut crate::audio::aac::ltp::LtpState>,
        pred: crate::audio::aac::ltp::PredSide,
    ) {
        let bands = self.get_bands();
        // LTP addback before TNS (§4.6.7.4.1)
        if let Some(st) = ltp_state {
            if let crate::audio::aac::ltp::PredSide::Ltp(ltp) = pred {
                st.apply_approx(&ltp, bands, &mut self.coeffs);
            }
            if let Some(pulse) = &self.pulse {
                pulse.synth(bands, &self.scales, &mut self.coeffs);
            }
            if let Some(tns) = &self.tns {
                tns.synth(&self.info, bands, rate_idx, &mut self.coeffs);
            }
            dsp.synth(
                &self.coeffs,
                &mut self.delay,
                self.info.window_sequence,
                self.info.window_shape,
                self.info.prev_window_shape,
                dst,
            );
            st.push_pcm(dst);
        } else {
            if let Some(pulse) = &self.pulse {
                pulse.synth(bands, &self.scales, &mut self.coeffs);
            }
            if let Some(tns) = &self.tns {
                tns.synth(&self.info, bands, rate_idx, &mut self.coeffs);
            }
            dsp.synth(
                &self.coeffs,
                &mut self.delay,
                self.info.window_sequence,
                self.info.window_shape,
                self.info.prev_window_shape,
                dst,
            );
        }
    }
}

/// SSR `gain_control_data()` — consume bits without applying (SSR filterbank
/// still uses the standard MDCT path for usable approximate playback).
fn skip_gain_control(bs: &mut BitReader<'_>, info: &IcsInfo) -> Result<(), &'static str> {
    let max_band = bs.read_bits(2)? as usize + 1;
    // For each window and band: adjust_num, then aleve/aloc codes — bit counts
    // depend on window sequence. Best-effort: read adjust_num (3) and skip
    // 4+4 bits per adjust (alevel+aloc lower bound).
    let nwin = if info.long_win { 1 } else { 8 };
    for _w in 0..nwin {
        for _b in 0..max_band {
            let adjust_num = bs.read_bits(3)? as usize;
            for _ in 0..adjust_num {
                let _ = bs.read_bits(4)?; // alevcode
                let _ = bs.read_bits(if info.long_win { 5 } else { 3 })?; // aloccode-ish
            }
        }
    }
    Ok(())
}

fn decode_noise(lcg: &mut Lcg, sf: f32, dst: &mut [f32]) {
    let mut energy = 0.0f32;
    for spec in dst.iter_mut() {
        *spec = f32::from((lcg.next() >> 16) as i16);
        energy += *spec * *spec;
    }
    let scale = if energy > 0.0 {
        sf / crate::cortex::tensor::libm_sqrtf(energy)
    } else {
        0.0
    };
    for spec in dst.iter_mut() {
        *spec *= scale;
    }
}

#[inline]
fn decode_sign(val: u32) -> f32 {
    1.0 - 2.0 * val as f32
}

fn decode_quads_unsigned(
    bs: &mut BitReader<'_>,
    tables: &DequantTables,
    book: usize,
    scale: f32,
    dst: &mut [f32],
) -> Result<(), &'static str> {
    let iquant = [0.0, scale, 2.51984209978974632953 * scale];
    for out in dst.chunks_exact_mut(4) {
        let idx = tables.spectrum[book].decode(bs)?;
        let (a, b, c, d) = codebooks::quads_vals(idx);
        if a != 0 {
            out[0] = decode_sign(bs.read_bit()?) * iquant[a as usize];
        }
        if b != 0 {
            out[1] = decode_sign(bs.read_bit()?) * iquant[b as usize];
        }
        if c != 0 {
            out[2] = decode_sign(bs.read_bit()?) * iquant[c as usize];
        }
        if d != 0 {
            out[3] = decode_sign(bs.read_bit()?) * iquant[d as usize];
        }
    }
    Ok(())
}

fn decode_quads_signed(
    bs: &mut BitReader<'_>,
    tables: &DequantTables,
    book: usize,
    scale: f32,
    dst: &mut [f32],
) -> Result<(), &'static str> {
    let iquant = [-scale, 0.0, scale];
    for out in dst.chunks_exact_mut(4) {
        let idx = tables.spectrum[book].decode(bs)?;
        let (a, b, c, d) = codebooks::quads_vals(idx);
        out[0] = iquant[a as usize];
        out[1] = iquant[b as usize];
        out[2] = iquant[c as usize];
        out[3] = iquant[d as usize];
    }
    Ok(())
}

fn decode_pairs_signed(
    bs: &mut BitReader<'_>,
    tables: &DequantTables,
    pair_book: usize,
    scale: f32,
    dst: &mut [f32],
) -> Result<(), &'static str> {
    // spectrum books 5,6 → tables index 4,5 → pair_book 0,1
    let huff_book = 4 + pair_book;
    for out in dst.chunks_exact_mut(2) {
        let idx = tables.spectrum[huff_book].decode(bs)?;
        let (x, y) = tables.pair_vals[pair_book][idx];
        out[0] = x * scale;
        out[1] = y * scale;
    }
    Ok(())
}

fn decode_pairs_unsigned(
    bs: &mut BitReader<'_>,
    tables: &DequantTables,
    pair_book: usize,
    scale: f32,
    dst: &mut [f32],
) -> Result<(), &'static str> {
    let huff_book = 4 + pair_book;
    for out in dst.chunks_exact_mut(2) {
        let idx = tables.spectrum[huff_book].decode(bs)?;
        let (x, y) = tables.pair_vals[pair_book][idx];
        let sign_x = if x != 0.0 {
            decode_sign(bs.read_bit()?)
        } else {
            1.0
        };
        let sign_y = if y != 0.0 {
            decode_sign(bs.read_bit()?)
        } else {
            1.0
        };
        out[0] = sign_x * x * scale;
        out[1] = sign_y * y * scale;
    }
    Ok(())
}

fn decode_pairs_escape(
    bs: &mut BitReader<'_>,
    tables: &DequantTables,
    scale: f32,
    dst: &mut [f32],
) -> Result<(), &'static str> {
    for out in dst.chunks_exact_mut(2) {
        let idx = tables.spectrum[10].decode(bs)?;
        let (a, b) = tables.esc_vals[idx];
        let sign_x = if a != 0 {
            decode_sign(bs.read_bit()?)
        } else {
            1.0
        };
        let sign_y = if b != 0 {
            decode_sign(bs.read_bit()?)
        } else {
            1.0
        };
        let xv = if a == 16 {
            let e = read_escape(bs)? as usize;
            if e >= tables.pow43.len() {
                return Err("aac: escape too large");
            }
            tables.pow43[e]
        } else {
            tables.pow43[a as usize]
        };
        let yv = if b == 16 {
            let e = read_escape(bs)? as usize;
            if e >= tables.pow43.len() {
                return Err("aac: escape too large");
            }
            tables.pow43[e]
        } else {
            tables.pow43[b as usize]
        };
        out[0] = sign_x * xv * scale;
        out[1] = sign_y * yv * scale;
    }
    Ok(())
}

fn read_escape(bs: &mut BitReader<'_>) -> Result<u16, &'static str> {
    let n = bs.read_unary_ones()?;
    if n >= 9 {
        return Err("aac: escape n too large");
    }
    let word = (1u16 << (n + 4)) + bs.read_bits(n + 4)? as u16;
    Ok(word)
}
