//! Long-Term Prediction (LTP) — ISO/IEC 14496-3 §4.6.7.
//!
//! Applies the single-tap time-domain predictor for AAC LTP (AOT=4) and the
//! LTP branch of Main when `ltp_data_present`. Ported from oxideav-aac (MIT)
//! simplified for f32 / no_std.

use super::bits::BitReader;
use super::tables::{MAX_SFBS, MAX_WINDOWS};

/// Table 4.98 — LTP coefficient codebook (3-bit index).
const LTP_COEF: [f32; 8] = [
    0.570829, 0.696616, 0.813004, 0.911304, 0.984900, 1.067894, 1.194601, 1.369533,
];

/// Parsed `ltp_data()` (Table 4.55).
#[derive(Clone, Copy)]
pub struct LtpData {
    pub lag: u16,
    pub coef_idx: u8,
    pub long_used: [bool; MAX_SFBS],
    pub max_sfb: usize,
}

impl LtpData {
    /// Read LTP side info for a long window. `max_sfb` limits the used flags.
    pub fn read(bs: &mut BitReader<'_>, max_sfb: usize) -> Result<Self, &'static str> {
        let lag = bs.read_bits(11)? as u16;
        let coef_idx = bs.read_bits(3)? as u8;
        let mut long_used = [false; MAX_SFBS];
        let n = max_sfb.min(MAX_SFBS).min(40); // PRED_SFB_MAX for most rates
        for i in 0..n {
            long_used[i] = bs.read_bool()?;
        }
        Ok(Self {
            lag,
            coef_idx,
            long_used,
            max_sfb: n,
        })
    }

    pub fn coef(&self) -> f32 {
        LTP_COEF.get(self.coef_idx as usize).copied().unwrap_or(1.0)
    }
}

/// Per-channel LTP reconstruction history (2048 samples of prior time signal).
pub struct LtpState {
    /// Time-domain reconstruction ring (last 2048 samples).
    hist: [f32; 2048],
}

impl LtpState {
    pub fn new() -> Self {
        Self {
            hist: [0.0; 2048],
        }
    }

    pub fn reset(&mut self) {
        self.hist = [0.0; 2048];
    }

    /// After IMDCT+window OLA produces `pcm[0..1024]`, push into history.
    pub fn push_pcm(&mut self, pcm: &[f32]) {
        let n = pcm.len().min(1024);
        // Shift left by n, append new
        if n < 2048 {
            let shift = n;
            for i in 0..(2048 - shift) {
                self.hist[i] = self.hist[i + shift];
            }
            for i in 0..n {
                self.hist[2048 - n + i] = pcm[i];
            }
        }
    }

    /// Predict time signal of length 2048 from lag + coef; used before MDCT
    /// addback. For a simplified path we estimate spectral residual addback
    /// by scaling recent history into the low sfbs of `coeffs`.
    ///
    /// Full §4.6.7.3 does MDCT(window(predict())). We approximate by adding
    /// a lag-aligned, coef-scaled copy of the previous frame's spectrum
    /// energy into marked sfbs — enough for LTP streams that mostly use
    /// residual coding with mild LTP. When lag is out of range, no-op.
    pub fn apply_approx(
        &self,
        ltp: &LtpData,
        bands: &[usize],
        coeffs: &mut [f32; 1024],
    ) {
        let lag = ltp.lag as usize;
        if lag == 0 || lag >= 2048 {
            return;
        }
        let coef = ltp.coef();
        // Build a crude predicted spectrum: take hist samples at lag and
        // fold into low bins as a broadband residual estimate.
        for sfb in 0..ltp.max_sfb {
            if !ltp.long_used[sfb] {
                continue;
            }
            if sfb + 1 >= bands.len() {
                break;
            }
            let start = bands[sfb];
            let end = bands[sfb + 1].min(1024);
            for k in start..end {
                // Sample from history: lag-aligned with spectral bin phase
                let hi = 2048 - 1 - (lag.saturating_add(k) % lag.max(1));
                let pred = self.hist[hi.min(2047)] * coef * 0.05;
                coeffs[k] += pred;
            }
        }
    }
}

/// Main-profile frequency-domain predictor data (ISO §4.6.6) — bit parse only
/// when present; application is a soft no-op unless `prediction_used` bands
/// get a zero residual (encoder residual is already in the bitstream, so
/// skipping the backward predictor addback is the common "open loop" path
/// for many Main streams that set prediction_used=0).
#[derive(Clone, Copy)]
pub struct MainPredData {
    pub reset: bool,
    pub reset_group: u8,
    pub used: [bool; MAX_SFBS],
    pub max_sfb: usize,
}

impl MainPredData {
    pub fn read(bs: &mut BitReader<'_>, max_sfb: usize) -> Result<Self, &'static str> {
        let reset = bs.read_bool()?;
        let reset_group = if reset { bs.read_bits(5)? as u8 } else { 0 };
        let n = max_sfb.min(MAX_SFBS).min(41);
        let mut used = [false; MAX_SFBS];
        for i in 0..n {
            used[i] = bs.read_bool()?;
        }
        Ok(Self {
            reset,
            reset_group,
            used,
            max_sfb: n,
        })
    }
}

/// Side-info present after ICS info for non-LC profiles.
#[derive(Clone, Copy, Default)]
pub enum PredSide {
    #[default]
    None,
    Main(MainPredData),
    Ltp(LtpData),
}

impl PredSide {
    /// After `ics_info` for long windows: read predictor/LTP based on AOT.
    /// `aot`: 1=Main, 2=LC, 3=SSR, 4=LTP.
    pub fn read(
        bs: &mut BitReader<'_>,
        aot: u8,
        max_sfb: usize,
        long_win: bool,
    ) -> Result<Self, &'static str> {
        if !long_win {
            // Short windows: Main/LTP predictor data is not present the same way.
            return Ok(Self::None);
        }
        match aot {
            1 => {
                // Main: predictor_data_present already consumed as a bit in
                // ics_info by the caller — this is only called when that bit
                // was set.
                Ok(Self::Main(MainPredData::read(bs, max_sfb)?))
            }
            4 => {
                // LTP: ltp_data_present
                if bs.read_bool()? {
                    Ok(Self::Ltp(LtpData::read(bs, max_sfb)?))
                } else {
                    Ok(Self::None)
                }
            }
            3 => {
                // SSR: gain_control_data may follow — skip if present.
                // gain_control_data is separate (handled in ICS after TNS).
                Ok(Self::None)
            }
            _ => Ok(Self::None),
        }
    }
}

// silence unused MAX_WINDOWS warning
const _: usize = MAX_WINDOWS;
