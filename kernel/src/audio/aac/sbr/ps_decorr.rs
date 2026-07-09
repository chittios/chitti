// Ported from oxideav-aac (MIT) — Copyright (c) 2026 Karpelès Lab Inc.
// See THIRDPARTY-LICENSES.md.

//! PS de-correlation — ISO/IEC 14496-3:2009 §8.6.4.5.
//!
//! The stereo reconstruction mixes the mono hybrid signal `s_k(n)`
//! with a de-correlated version `d_k(n)` of itself. Per §8.6.4.5.2
//! the first `NR_ALLPASS_BANDS` hybrid channels run through a chain
//! of `NR_ALLPASS_LINKS = 3` complex all-pass sections behind a
//! 2-slot delay and a fractional-delay rotation:
//!
//! ```text
//! H_k(z) = z⁻² · φ_fract(k) · Π_m (Q(k,m)·z^(−d(m)) − a(m)·g(k))
//!                                 / (1 − a(m)·g(k)·Q(k,m)·z^(−d(m)))
//! ```
//!
//! with `a(m) = {0.65143905753106, 0.56471812200776, 0.48954165955695}`,
//! `d(m) = {3, 4, 5}` (Table 8.39), the unit rotations
//! `Q(k,m) = exp(−iπ·q(m)·fcenter(k))` (`q = {0.43, 0.75, 0.347}`,
//! Table 8.42), `φ_fract(k) = exp(−iπ·q_φ·fcenter(k))` (`q_φ = 0.39`),
//! and the frequency-dependent decay
//! `g(k) = max(0, 1 − DECAY_SLOPE·(k − DECAY_CUTOFF))`. The centre
//! frequencies `fcenter(k)` come from Table 8.40 / 8.41 for the split
//! region and the closed forms `k + 1/2 − 7` / `k + 1/2 − 27` above
//! it. Bands `NR_ALLPASS_BANDS..` use a plain delay: 14 slots up to
//! `SHORT_DELAY_BAND`, 1 slot above.
//!
//! §8.6.4.5.3–5.4 duck the de-correlated signal at transients: the
//! per-stereo-band input power is peak-decayed
//! (`α = 0.76592833836465`, Table 8.43), both the power and the
//! peak-minus-power difference are smoothed with the one-pole
//! `H_smooth` (`a_smooth = 0.25`), and wherever
//! `γ·PSmoothPeakDecayDiff > PSmoothNrg` (`γ = 1.5`) the output is
//! scaled by their ratio.
//!
//! [`PsDecorr`] carries every filter/delay/detector state across
//! frames and exposes the Annex 8.A.3 resets: `reset_bands(kmax)`
//! zeroes the state of hybrid channels `k ≥ kmax` each stereo frame
//! (the region above the SBR-generated spectrum), and a full reset
//! covers the "no `ps_data()` in the previous frame" rule.
//!
//! All truth from ISO/IEC 14496-3:2009 §8.6.4.5 / Annex 8.A staged
//! under `docs/audio/aac/`.

use alloc::vec::Vec;
use alloc::vec;
use super::ps_hybrid::HybridConfig;
use super::ps_map::parameter_map;
use super::sbr_qmf::Complex;
use super::Result;

/// `DECAY_SLOPE` (§8.6.4.5.1).
use super::math::F64Ext;
const DECAY_SLOPE: f64 = 0.05;

/// `a(m)` — all-pass filter coefficients (Table 8.39).
const A: [f64; 3] = [0.65143905753106, 0.56471812200776, 0.48954165955695];

/// `d(m)` — all-pass link delays (Table 8.39).
const D: [usize; 3] = [3, 4, 5];

/// `q(m)` — fractional delay lengths (Table 8.42).
const Q_FRACT: [f64; 3] = [0.43, 0.75, 0.347];

/// `q_φ` — fractional delay constant (§8.6.4.5.2).
const Q_PHI: f64 = 0.39;

/// Peak decay factor `α` (Table 8.43).
const PEAK_DECAY: f64 = 0.76592833836465;

/// Smoothing coefficient `a_smooth` (§8.6.4.5.1).
const A_SMOOTH: f64 = 0.25;

/// Transient impact factor `γ` (§8.6.4.5.3).
const GAMMA: f64 = 1.5;

/// Long delay for the non-all-pass mid bands (§8.6.4.5.2).
const LONG_DELAY: usize = 14;

/// Table 8.40 — `fcenter_20(k)` for the split region (k = 0..10).
const F_CENTER_20: [f64; 10] = [
    -3.0 / 8.0,
    -1.0 / 8.0,
    1.0 / 8.0,
    3.0 / 8.0,
    5.0 / 8.0,
    7.0 / 8.0,
    5.0 / 4.0,
    7.0 / 4.0,
    9.0 / 4.0,
    11.0 / 4.0,
];

/// Table 8.41 — `fcenter_34(k)` for the split region (k = 0..32).
const F_CENTER_34: [f64; 32] = [
    1.0 / 12.0,
    3.0 / 12.0,
    5.0 / 12.0,
    7.0 / 12.0,
    9.0 / 12.0,
    11.0 / 12.0,
    13.0 / 12.0,
    15.0 / 12.0,
    17.0 / 12.0,
    -5.0 / 12.0,
    -3.0 / 12.0,
    -1.0 / 12.0,
    17.0 / 8.0,
    19.0 / 8.0,
    5.0 / 8.0,
    7.0 / 8.0,
    9.0 / 8.0,
    11.0 / 8.0,
    13.0 / 8.0,
    15.0 / 8.0,
    9.0 / 4.0,
    11.0 / 4.0,
    13.0 / 4.0,
    7.0 / 4.0,
    17.0 / 4.0,
    11.0 / 4.0,
    13.0 / 4.0,
    15.0 / 4.0,
    17.0 / 4.0,
    19.0 / 4.0,
    21.0 / 4.0,
    15.0 / 4.0,
];

/// The §8.6.4.5.1 configuration constants that depend on the stereo
/// band count.
#[derive(Debug, Clone, Copy)]
struct DecorrConsts {
    nr_par_bands: usize,
    nr_bands: usize,
    decay_cutoff: usize,
    nr_allpass_bands: usize,
    short_delay_band: usize,
}

fn consts(config: HybridConfig) -> DecorrConsts {
    match config {
        HybridConfig::Bands1020 => DecorrConsts {
            nr_par_bands: 20,
            nr_bands: 71,
            decay_cutoff: 10,
            nr_allpass_bands: 30,
            short_delay_band: 42,
        },
        HybridConfig::Bands34 => DecorrConsts {
            nr_par_bands: 34,
            nr_bands: 91,
            decay_cutoff: 32,
            nr_allpass_bands: 50,
            short_delay_band: 62,
        },
    }
}

/// `fcenter(k)` for the all-pass region (§8.6.4.5.2).
fn f_center(config: HybridConfig, k: usize) -> f64 {
    match config {
        HybridConfig::Bands1020 => {
            if k < F_CENTER_20.len() {
                F_CENTER_20[k]
            } else {
                k as f64 + 0.5 - 7.0
            }
        }
        HybridConfig::Bands34 => {
            if k < F_CENTER_34.len() {
                F_CENTER_34[k]
            } else {
                k as f64 + 0.5 - 27.0
            }
        }
    }
}

/// Per-band all-pass state: the z⁻² input delay plus one direct-form
/// ring per link (`w[n] = u[n] + a·g·Q·w[n−d]`,
/// `v[n] = Q·w[n−d] − a·g·w[n]`).
#[derive(Debug, Clone)]
struct AllpassState {
    /// z⁻² input history (index 0 = one slot ago).
    in2: [Complex; 2],
    /// Ring buffers for the three links (lengths 3, 4, 5).
    w: [Vec<Complex>; 3],
    /// Ring positions.
    pos: [usize; 3],
}

impl AllpassState {
    fn new() -> Self {
        AllpassState {
            in2: [Complex::default(); 2],
            w: [
                vec![Complex::default(); D[0]],
                vec![Complex::default(); D[1]],
                vec![Complex::default(); D[2]],
            ],
            pos: [0; 3],
        }
    }

    fn reset(&mut self) {
        self.in2 = [Complex::default(); 2];
        for (w, d) in self.w.iter_mut().zip(D) {
            w.iter_mut().for_each(|c| *c = Complex::default());
            debug_assert_eq!(w.len(), d);
        }
        self.pos = [0; 3];
    }
}

/// The §8.6.4.5 de-correlator (one instance per PS decoder).
#[derive(Debug, Clone)]
pub struct PsDecorr {
    config: HybridConfig,
    /// All-pass state per band `k < NR_ALLPASS_BANDS`.
    allpass: Vec<AllpassState>,
    /// Pre-computed `φ_fract(k)` per all-pass band.
    phi_fract: Vec<Complex>,
    /// Pre-computed `Q(k,m)·1` per all-pass band and link.
    q_fract: Vec<[Complex; 3]>,
    /// `g_DecaySlope(k)` per all-pass band.
    g_decay: Vec<f64>,
    /// Delay lines for the non-all-pass bands (14 or 1 slots each).
    delay: Vec<Vec<Complex>>,
    /// Ring positions for `delay`.
    delay_pos: Vec<usize>,
    /// Transient detector state per stereo band.
    peak_decay_nrg: Vec<f64>,
    smooth_nrg: Vec<f64>,
    smooth_peak_diff: Vec<f64>,
}

impl PsDecorr {
    /// A fresh de-correlator for `config`.
    #[must_use]
    pub fn new(config: HybridConfig) -> Self {
        let c = consts(config);
        let mut phi_fract = Vec::with_capacity(c.nr_allpass_bands);
        let mut q_fract = Vec::with_capacity(c.nr_allpass_bands);
        let mut g_decay = Vec::with_capacity(c.nr_allpass_bands);
        for k in 0..c.nr_allpass_bands {
            let f = f_center(config, k);
            let arg = -core::f64::consts::PI * Q_PHI * f;
            let (s, co) = arg.sin_cos();
            phi_fract.push(Complex::new(co, s));
            let mut qs = [Complex::default(); 3];
            for (m, q) in qs.iter_mut().enumerate() {
                let arg = -core::f64::consts::PI * Q_FRACT[m] * f;
                let (s, co) = arg.sin_cos();
                *q = Complex::new(co, s);
            }
            q_fract.push(qs);
            let g = if k > c.decay_cutoff {
                (1.0 - DECAY_SLOPE * (k as f64 - c.decay_cutoff as f64)).max(0.0)
            } else {
                1.0
            };
            g_decay.push(g);
        }
        let mut delay = Vec::with_capacity(c.nr_bands - c.nr_allpass_bands);
        for k in c.nr_allpass_bands..c.nr_bands {
            let d = if k < c.short_delay_band {
                LONG_DELAY
            } else {
                1
            };
            delay.push(vec![Complex::default(); d]);
        }
        PsDecorr {
            config,
            allpass: vec![AllpassState::new(); c.nr_allpass_bands],
            phi_fract,
            q_fract,
            g_decay,
            delay_pos: vec![0; c.nr_bands - c.nr_allpass_bands],
            delay,
            peak_decay_nrg: vec![0.0; c.nr_par_bands],
            smooth_nrg: vec![0.0; c.nr_par_bands],
            smooth_peak_diff: vec![0.0; c.nr_par_bands],
        }
    }

    /// Annex 8.A.3 partial reset: zero the filter state of hybrid
    /// channels `k ≥ kmax` (the region above the SBR-generated
    /// spectrum), or the whole bank with `kmax = 0` (the "no
    /// `ps_data()` in the previous frame" full reset).
    pub fn reset_bands(&mut self, kmax: usize) {
        let c = consts(self.config);
        for k in kmax..c.nr_allpass_bands {
            self.allpass[k].reset();
        }
        for k in kmax.max(c.nr_allpass_bands)..c.nr_bands {
            let i = k - c.nr_allpass_bands;
            self.delay[i]
                .iter_mut()
                .for_each(|v| *v = Complex::default());
            self.delay_pos[i] = 0;
        }
    }

    /// De-correlate one stereo frame of hybrid slots (each
    /// `nr_bands()` wide). Returns `d_k(n)` with the transient
    /// attenuation applied; all state advances.
    pub fn process(&mut self, s: &[Vec<Complex>]) -> Result<Vec<Vec<Complex>>> {
        let c = consts(self.config);
        let b_k = parameter_map(self.config);
        if s.iter().any(|row| row.len() != c.nr_bands) {
            return Err("aac/sbr: ps data invalid");
        }
        let mut out = vec![vec![Complex::default(); c.nr_bands]; s.len()];
        for (n, row) in s.iter().enumerate() {
            // §8.6.4.5.3 transient detection at this slot.
            let mut p = vec![0.0f64; c.nr_par_bands];
            for (k, v) in row.iter().enumerate() {
                p[usize::from(b_k[k])] += v.norm_sqr();
            }
            let mut g_ratio = vec![1.0f64; c.nr_par_bands];
            for i in 0..c.nr_par_bands {
                let peak = if PEAK_DECAY * self.peak_decay_nrg[i] < p[i] {
                    p[i]
                } else {
                    PEAK_DECAY * self.peak_decay_nrg[i]
                };
                self.peak_decay_nrg[i] = peak;
                self.smooth_nrg[i] += A_SMOOTH * (p[i] - self.smooth_nrg[i]);
                self.smooth_peak_diff[i] += A_SMOOTH * (peak - p[i] - self.smooth_peak_diff[i]);
                if GAMMA * self.smooth_peak_diff[i] > self.smooth_nrg[i] {
                    g_ratio[i] = self.smooth_nrg[i] / (GAMMA * self.smooth_peak_diff[i]);
                }
            }

            // §8.6.4.5.2 all-pass chain for the low bands.
            for k in 0..c.nr_allpass_bands {
                let st = &mut self.allpass[k];
                // z⁻² then φ_fract rotation.
                let delayed = st.in2[1];
                st.in2[1] = st.in2[0];
                st.in2[0] = row[k];
                let mut u = self.phi_fract[k] * delayed;
                // Three all-pass links.
                let g = self.g_decay[k];
                for m in 0..3 {
                    let coef = A[m] * g;
                    let q = self.q_fract[k][m];
                    let pos = st.pos[m];
                    let w_d = st.w[m][pos];
                    // w[n] = u[n] + a·g·Q·w[n−d]
                    let w_n = u + q * w_d * coef;
                    // v[n] = Q·w[n−d] − a·g·w[n]
                    u = q * w_d - w_n * coef;
                    st.w[m][pos] = w_n;
                    st.pos[m] = (pos + 1) % D[m];
                }
                out[n][k] = u * g_ratio[usize::from(b_k[k])];
            }

            // Plain delays above.
            for k in c.nr_allpass_bands..c.nr_bands {
                let i = k - c.nr_allpass_bands;
                let pos = self.delay_pos[i];
                let v = self.delay[i][pos];
                self.delay[i][pos] = row[k];
                self.delay_pos[i] = (pos + 1) % self.delay[i].len();
                out[n][k] = v * g_ratio[usize::from(b_k[k])];
            }
        }
        Ok(out)
    }
}
