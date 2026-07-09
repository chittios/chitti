//! Channel pair element: SCE/CPE decode + mid-side / intensity stereo.
//!
//! Ported from Symphonia AAC (NihAV origin, MPL-2.0). See THIRDPARTY-LICENSES.md.

use super::bits::BitReader;
use super::codebooks::DequantTables;
use super::dsp::Dsp;
use super::ics::Ics;
use super::tables::*;

#[derive(Clone)]
pub struct ChannelPair {
    pub is_pair: bool,
    pub channel: usize,
    ms_mask_present: u8,
    ms_used: [[bool; MAX_SFBS]; MAX_WINDOWS],
    ics0: Ics,
    ics1: Ics,
    lcg: Lcg,
}

impl ChannelPair {
    pub fn new(is_pair: bool, channel: usize, sbinfo: GASubbandInfo) -> Self {
        Self {
            is_pair,
            channel,
            ms_mask_present: 0,
            ms_used: [[false; MAX_SFBS]; MAX_WINDOWS],
            ics0: Ics::new(sbinfo),
            ics1: Ics::new(sbinfo),
            lcg: Lcg::new(0x1f2e3d4c),
        }
    }

    pub fn reset(&mut self) {
        self.ics0.reset();
        self.ics1.reset();
    }

    pub fn decode_ga_sce(
        &mut self,
        bs: &mut BitReader<'_>,
        tables: &DequantTables,
        aot: u8,
    ) -> Result<(super::ltp::PredSide, super::ltp::PredSide), &'static str> {
        let p0 = self.ics0.decode(bs, &mut self.lcg, tables, false, aot, None)?;
        Ok((p0, super::ltp::PredSide::None))
    }

    pub fn decode_ga_cpe(
        &mut self,
        bs: &mut BitReader<'_>,
        tables: &DequantTables,
        aot: u8,
    ) -> Result<(super::ltp::PredSide, super::ltp::PredSide), &'static str> {
        let common_window = bs.read_bool()?;
        let mut common_pred = None;
        if common_window {
            common_pred = Some(self.ics0.decode_info(bs, aot)?);
            // For LTP + common_window: optional second ltp after ms mask (ISO).
            self.ms_mask_present = bs.read_bits(2)? as u8;
            match self.ms_mask_present {
                0 | 2 => {
                    let is_used = self.ms_mask_present == 2;
                    for g in 0..self.ics0.info.window_groups {
                        for sfb in 0..self.ics0.info.max_sfb {
                            self.ms_used[g][sfb] = is_used;
                        }
                    }
                }
                1 => {
                    for g in 0..self.ics0.info.window_groups {
                        for sfb in 0..self.ics0.info.max_sfb {
                            self.ms_used[g][sfb] = bs.read_bool()?;
                        }
                    }
                }
                _ => return Err("aac: invalid mid-side mask"),
            }
            self.ics1.info.copy_from_common(&self.ics0.info);
        }

        let p0 = self
            .ics0
            .decode(bs, &mut self.lcg, tables, common_window, aot, common_pred)?;
        let p1 = self
            .ics1
            .decode(bs, &mut self.lcg, tables, common_window, aot, None)?;

        if common_window {
            let bands = self.ics0.get_bands();
            let mut g = 0usize;
            for w in 0..self.ics0.info.num_windows {
                if w > 0 && !self.ics0.info.scale_factor_grouping[w - 1] {
                    g += 1;
                }
                for sfb in 0..self.ics0.info.max_sfb {
                    let start = w * 128 + bands[sfb];
                    let end = w * 128 + bands[sfb + 1];
                    if self.ics1.is_intensity(g, sfb) {
                        let invert = self.ms_mask_present == 1 && self.ms_used[g][sfb];
                        let dir = if self.ics1.get_intensity_dir(g, sfb) {
                            1.0
                        } else {
                            -1.0
                        };
                        let factor = if invert { -1.0 } else { 1.0 };
                        let scale = dir * factor * self.ics1.scales[g][sfb];
                        let left = &self.ics0.coeffs[start..end];
                        let right = &mut self.ics1.coeffs[start..end];
                        for (l, r) in left.iter().zip(right) {
                            *r = scale * l;
                        }
                    } else if self.ics0.is_noise(g, sfb) || self.ics1.is_noise(g, sfb) {
                        // PNS: no joint stereo
                    } else if self.ms_used[g][sfb] {
                        let mid = &mut self.ics0.coeffs[start..end];
                        // SAFETY: mid and side are disjoint slices of different arrays.
                        let side = &mut self.ics1.coeffs[start..end];
                        for (m, s) in mid.iter_mut().zip(side.iter_mut()) {
                            let tmp = *m - *s;
                            *m += *s;
                            *s = tmp;
                        }
                    }
                }
            }
        }
        Ok((p0, p1))
    }

    pub fn synth_left(
        &mut self,
        dsp: &mut Dsp,
        rate_idx: usize,
        dst: &mut [f32],
        ltp: Option<&mut super::ltp::LtpState>,
        pred: super::ltp::PredSide,
    ) {
        self.ics0.synth_channel(dsp, rate_idx, dst, ltp, pred);
    }

    pub fn synth_right(
        &mut self,
        dsp: &mut Dsp,
        rate_idx: usize,
        dst: &mut [f32],
        ltp: Option<&mut super::ltp::LtpState>,
        pred: super::ltp::PredSide,
    ) {
        self.ics1.synth_channel(dsp, rate_idx, dst, ltp, pred);
    }
}
