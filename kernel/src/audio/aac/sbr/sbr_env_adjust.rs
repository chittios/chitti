// Ported from oxideav-aac (MIT) — Copyright (c) 2026 Karpelès Lab Inc.
// See THIRDPARTY-LICENSES.md.

//! SBR HF adjustment (envelope adjuster) — ISO/IEC 14496-3 §4.6.18.7.
//!
//! Takes the HF-generated subband matrix `XHigh` and produces the
//! output matrix `Y` over the `M` SBR subbands starting at `kx`:
//!
//! * **Mapping** (§4.6.18.7.2) — `EOrigMapped` / `QMapped` to QMF
//!   resolution, the `SIndexMapped` sinusoid placement (band middle,
//!   `δStep` start gate against `lA` and the previous frame's
//!   sinusoids) and the `SMapped` band flags.
//! * **Current envelope estimation** (§4.6.18.7.3) — `ECurr` by
//!   squared-magnitude averaging, per subband (`bs_interpol_freq = 1`)
//!   or per envelope band.
//! * **Additional-component levels** (§4.6.18.7.4) — `QM` / `SM`
//!   (amplitude domain, i.e. with the square root of the energy
//!   ratios).
//! * **Gain** (§4.6.18.7.5) — `G`, the limiter (`GMax` from the
//!   `fTableLim` band ratios and `limGain`), the noise-level limit
//!   `QM_Lim`, and the boost compensation `GBoost` capped at
//!   `1.584893192`.
//! * **Assembly** (§4.6.18.7.6) — the `hSmooth` gain/noise smoothing
//!   over `hSL` columns, `W1 = GFilt·XHigh`, the Table 4.A.91 noise
//!   mix `W2`, and the `φsin` sinusoid injection with the
//!   `(−1)^(m+kx)` imaginary alternation, producing `Y`.
//!
//! Cross-frame state (`EnvAdjustState`) carries the previous frame's
//! last-envelope `SIndexMapped`, `lA` / `LE`, the `GTemp` / `QTemp`
//! smoothing tails, and the running `indexNoise` / `indexSine`.
//!
//! ## Provenance
//!
//! Every formula (including the square roots the §4.6.18.7.4–7.5
//! equations carry) was read from the staged ISO/IEC 14496-3:2009 spec
//! PDF's typeset equations. No part of this implementation is derived
//! from any external decoder.

use alloc::vec::Vec;
use alloc::vec;
use super::sbr_freq_bands::HiLoTables;
use super::sbr_hf_gen::T_HF_ADJ;
use super::sbr_noise_table::NOISE_TABLE;
use super::sbr_qmf::Complex;
use super::Result;

/// `limGain = [0.70795, 1.0, 1.41254, 1e10]` (§4.6.18.7.5).
use super::math::F64Ext;
pub const LIM_GAIN: [f64; 4] = [0.70795, 1.0, 1.41254, 1e10];

/// `ε0 = 1e-12` (§4.6.18.7.5).
pub const EPS0: f64 = 1e-12;

/// `ε = 1` (§4.6.18.2.5) — the division-by-zero guard in the gain.
pub const EPS: f64 = 1.0;

/// The `GBoost` cap `1.584893192` (§4.6.18.7.5).
pub const MAX_BOOST: f64 = 1.584893192;

/// The `GMax` cap `10^5` (§4.6.18.7.5).
pub const G_MAX_CAP: f64 = 1e5;

/// `hSmooth` — the §4.6.18.7.6 smoothing filter.
pub const H_SMOOTH: [f64; 5] = [
    0.33333333333333,
    0.30150283239582,
    0.21816949906249,
    0.11516383427084,
    0.03183050093751,
];

/// `φRe,sin = [1, 0, −1, 0]`, `φIm,sin = [0, 1, 0, −1]` (§4.6.18.7.6).
pub const PHI_SIN: [(f64, f64); 4] = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)];

/// Per-frame inputs to the envelope adjuster (one channel).
#[derive(Debug)]
pub struct EnvParams<'a> {
    /// Derived frequency band tables (`M`, `kx`, high/low/noise).
    pub bands: &'a HiLoTables,
    /// The §4.6.18.3.2.3 limiter band table `fTableLim(0..=NL)`.
    pub f_table_lim: &'a [i32],
    /// Envelope time borders `tE(0..=LE)` (slots).
    pub t_e: &'a [i32],
    /// Noise-floor time borders `tQ(0..=LQ)` (slots).
    pub t_q: &'a [i32],
    /// Per-envelope frequency resolution `r(l)` (`true` = high).
    pub freq_res: &'a [bool],
    /// Table 4.176 `lA` (`-1` = none).
    pub l_a: i32,
    /// Dequantized envelope energies `EOrig[l][k]`.
    pub e_orig: &'a [Vec<f64>],
    /// Dequantized noise-floor energies `QOrig[l][k]`.
    pub q_orig: &'a [Vec<f64>],
    /// `bs_add_harmonic` flags (`NHigh` entries; empty = none).
    pub add_harmonic: &'a [bool],
    /// `bs_interpol_freq`.
    pub interpol_freq: bool,
    /// `bs_smoothing_mode` (`true` ⇒ `hSL = 0`).
    pub smoothing_mode: bool,
    /// `bs_limiter_gains` (`0..=3`, indexes [`LIM_GAIN`]).
    pub limiter_gains: u8,
    /// The §4.6.18.3.3 reset flag (header band geometry changed).
    pub reset: bool,
}

/// Cross-frame envelope-adjuster state for one channel.
#[derive(Debug, Clone, Default)]
pub struct EnvAdjustState {
    /// Previous frame's last-envelope `SIndexMapped` (per SBR subband,
    /// `kx`-relative) plus its `kx`, for the `δStep` gate.
    s_index_prev: Vec<bool>,
    k_x_prev: i32,
    /// Previous frame's `lA` and `LE` (for `lAPrev`).
    l_a_prev_frame: i32,
    l_e_prev: i32,
    /// Previous frame's trailing `hSL` columns of `GTemp` / `QTemp`.
    g_temp_tail: Vec<Vec<f64>>,
    q_temp_tail: Vec<Vec<f64>>,
    /// Running noise / sine phase indices.
    index_noise: usize,
    index_sine: usize,
    started: bool,
}

impl EnvAdjustState {
    /// Fresh state (first frame / after a stream reset).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Run the §4.6.18.7 HF adjustment for one channel's SBR frame.
///
/// `x_high` is the slot-major HF-generator output (spec absolute
/// columns, i.e. spec index `i + tHFAdj` is a direct column index).
/// Returns `Y` in the same layout, filled for the SBR range and the
/// frame's envelope span; other cells are zero.
pub fn adjust(
    x_high: &[[Complex; 64]],
    p: &EnvParams<'_>,
    st: &mut EnvAdjustState,
) -> Result<Vec<[Complex; 64]>> {
    let m_cnt = usize::try_from(p.bands.m).map_err(|_| "aac/sbr: freq band invalid")?;
    let k_x = p.bands.k_x;
    let l_e = p
        .t_e
        .len()
        .checked_sub(1)
        .ok_or("aac/sbr: freq band invalid")?;
    if l_e == 0
        || p.freq_res.len() != l_e
        || p.e_orig.len() != l_e
        || p.q_orig.len() + 1 != p.t_q.len()
        || p.f_table_lim.len() < 2
        || usize::from(p.limiter_gains) >= LIM_GAIN.len()
    {
        return Err("aac/sbr: freq band invalid");
    }

    let rate = 2i32; // RATE (§4.6.18.2.5)
    let i0 = rate * p.t_e[0];
    let i_end = rate * p.t_e[l_e];
    let n_cols = usize::try_from(i_end - i0).map_err(|_| "aac/sbr: freq band invalid")?;
    if i0 < 0
        || usize::try_from(i_end).map_err(|_| "aac/sbr: freq band invalid")? + T_HF_ADJ > x_high.len()
    {
        return Err("aac/sbr: freq band invalid");
    }

    if p.reset || !st.started {
        st.index_noise = 0;
        st.index_sine = 0;
        st.s_index_prev.clear();
        st.g_temp_tail.clear();
        st.q_temp_tail.clear();
        st.l_a_prev_frame = -1;
        st.l_e_prev = 0;
        st.started = true;
    }

    // lAPrev: 0 if the previous frame's transient sat on its trailing
    // border, else -1.
    let l_a_prev = if st.l_a_prev_frame == st.l_e_prev {
        0i32
    } else {
        -1
    };

    // ---- §4.6.18.7.2 mapping -------------------------------------
    // Envelope band table per resolution.
    let f_of = |high: bool| -> &Vec<i32> {
        if high {
            &p.bands.f_table_high
        } else {
            &p.bands.f_table_low
        }
    };
    // Band index of QMF subband `k` in border table `f`.
    let band_of = |f: &[i32], k: i32| -> Result<usize> {
        for i in 0..f.len() - 1 {
            if f[i] <= k && k < f[i + 1] {
                return Ok(i);
            }
        }
        Err("aac/sbr: freq band invalid")
    };

    let mut e_map = vec![vec![0.0f64; m_cnt]; l_e]; // EOrigMapped[l][m]
    let mut q_map = vec![vec![0.0f64; m_cnt]; l_e]; // QMapped[l][m]
    let mut s_index = vec![vec![false; m_cnt]; l_e]; // SIndexMapped[l][m]
    let mut s_map = vec![vec![false; m_cnt]; l_e]; // SMapped[l][m]

    let n_high = p.bands.n_high();
    for l in 0..l_e {
        let f = f_of(p.freq_res[l]);
        if p.e_orig[l].len() + 1 != f.len() {
            return Err("aac/sbr: freq band invalid");
        }
        // k(l): the noise floor whose span contains envelope l.
        let mut kq = None;
        for q in 0..p.t_q.len() - 1 {
            if p.t_q[q] <= p.t_e[l] && p.t_e[l + 1] <= p.t_q[q + 1] {
                kq = Some(q);
                break;
            }
        }
        let kq = kq.ok_or("aac/sbr: freq band invalid")?;
        if p.q_orig[kq].len() + 1 != p.bands.f_table_noise.len() {
            return Err("aac/sbr: freq band invalid");
        }
        for m in 0..m_cnt {
            let k = k_x + i32::try_from(m).map_err(|_| "aac/sbr: freq band invalid")?;
            e_map[l][m] = p.e_orig[l][band_of(f, k)?];
            q_map[l][m] = p.q_orig[kq][band_of(&p.bands.f_table_noise, k)?];
        }

        // SIndexMapped: sinusoid in the middle subband of each
        // high-resolution band, gated by δStep.
        if !p.add_harmonic.is_empty() {
            if p.add_harmonic.len() != n_high {
                return Err("aac/sbr: freq band invalid");
            }
            for (i, &on) in p.add_harmonic.iter().enumerate() {
                if !on {
                    continue;
                }
                let mid = (p.bands.f_table_high[i + 1] + p.bands.f_table_high[i]) / 2;
                let m_rel = mid - k_x;
                if m_rel < 0 || m_rel as usize >= m_cnt {
                    continue;
                }
                // δStep: on from lA, or already ringing in the
                // previous frame's last envelope.
                let prev_on = {
                    let prev_rel = mid - st.k_x_prev;
                    prev_rel >= 0
                        && st
                            .s_index_prev
                            .get(prev_rel as usize)
                            .copied()
                            .unwrap_or(false)
                };
                if (l as i32) >= p.l_a || prev_on {
                    s_index[l][m_rel as usize] = true;
                }
            }
        }
        // SMapped: any sinusoid within the envelope band.
        for i in 0..f.len() - 1 {
            let any = ((f[i] - k_x).max(0)..(f[i + 1] - k_x).max(0))
                .any(|j| (j as usize) < m_cnt && s_index[l][j as usize]);
            if any {
                for j in (f[i] - k_x).max(0)..(f[i + 1] - k_x).max(0) {
                    if (j as usize) < m_cnt {
                        s_map[l][j as usize] = true;
                    }
                }
            }
        }
    }

    // ---- §4.6.18.7.3 current envelope ----------------------------
    let mut e_curr = vec![vec![0.0f64; m_cnt]; l_e];
    for (l, e_curr_l) in e_curr.iter_mut().enumerate() {
        let lo = (rate * p.t_e[l] + T_HF_ADJ as i32) as usize;
        let hi = (rate * p.t_e[l + 1] + T_HF_ADJ as i32) as usize;
        let width = (hi - lo) as f64;
        if p.interpol_freq {
            for (m, e) in e_curr_l.iter_mut().enumerate() {
                let k = (k_x as usize) + m;
                let sum: f64 = x_high[lo..hi].iter().map(|col| col[k].norm_sqr()).sum();
                *e = sum / width;
            }
        } else {
            let f = f_of(p.freq_res[l]);
            for pband in 0..f.len() - 1 {
                let kl = f[pband];
                let kh = f[pband + 1] - 1;
                let mut sum = 0.0;
                for j in kl..=kh {
                    sum += x_high[lo..hi]
                        .iter()
                        .map(|col| col[j as usize].norm_sqr())
                        .sum::<f64>();
                }
                let avg = sum / (width * f64::from(kh - kl + 1));
                for j in kl..=kh {
                    let m_rel = j - k_x;
                    if m_rel >= 0 && (m_rel as usize) < m_cnt {
                        e_curr_l[m_rel as usize] = avg;
                    }
                }
            }
        }
    }

    // ---- §4.6.18.7.4 / 7.5 gain, limiter, boost ------------------
    let lim_gain = LIM_GAIN[usize::from(p.limiter_gains)];
    let n_l = p.f_table_lim.len() - 1;

    let mut g_lim_boost = vec![vec![0.0f64; m_cnt]; l_e];
    let mut q_m_lim_boost = vec![vec![0.0f64; m_cnt]; l_e];
    let mut s_m_boost = vec![vec![0.0f64; m_cnt]; l_e];

    for l in 0..l_e {
        let li = l as i32;
        let delta_l = if li == p.l_a || li == l_a_prev {
            0.0
        } else {
            1.0
        };

        // QM / SM (amplitude domain).
        let mut q_m = vec![0.0f64; m_cnt];
        let mut s_m = vec![0.0f64; m_cnt];
        let mut g = vec![0.0f64; m_cnt];
        for m in 0..m_cnt {
            let e_o = e_map[l][m];
            let q = q_map[l][m];
            q_m[m] = (e_o * q / (1.0 + q)).sqrt();
            s_m[m] = if s_index[l][m] {
                (e_o / (1.0 + q)).sqrt()
            } else {
                0.0
            };
            g[m] = if s_map[l][m] {
                ((e_o / (EPS + e_curr[l][m])) * (q / (1.0 + q))).sqrt()
            } else {
                (e_o / ((EPS + e_curr[l][m]) * (1.0 + delta_l * q))).sqrt()
            };
        }

        // Limiter-band maxima.
        let mut g_max = vec![0.0f64; m_cnt];
        for k in 0..n_l {
            let lo = (p.f_table_lim[k] - k_x).max(0) as usize;
            let hi = ((p.f_table_lim[k + 1] - k_x).max(0) as usize).min(m_cnt);
            let num: f64 = EPS0 + e_map[l][lo..hi].iter().sum::<f64>();
            let den: f64 = EPS0 + e_curr[l][lo..hi].iter().sum::<f64>();
            let gmax = ((num / den).sqrt() * lim_gain).min(G_MAX_CAP);
            for gm in &mut g_max[lo..hi] {
                *gm = gmax;
            }
        }

        // QM_Lim / GLim.
        let mut q_m_lim = vec![0.0f64; m_cnt];
        let mut g_lim = vec![0.0f64; m_cnt];
        for m in 0..m_cnt {
            q_m_lim[m] = if g[m] > 0.0 {
                q_m[m].min(q_m[m] * g_max[m] / g[m])
            } else {
                q_m[m]
            };
            g_lim[m] = g[m].min(g_max[m]);
        }

        // Boost per limiter band.
        for k in 0..n_l {
            let lo = (p.f_table_lim[k] - k_x).max(0) as usize;
            let hi = ((p.f_table_lim[k + 1] - k_x).max(0) as usize).min(m_cnt);
            let mut num = EPS0;
            let mut den = EPS0;
            for i in lo..hi {
                num += e_map[l][i];
                let delta_s = if s_m[i] != 0.0 || li == p.l_a || li == l_a_prev {
                    0.0
                } else {
                    1.0
                };
                den += e_curr[l][i] * g_lim[i] * g_lim[i]
                    + s_m[i] * s_m[i]
                    + delta_s * q_m_lim[i] * q_m_lim[i];
            }
            let boost = (num / den).sqrt().min(MAX_BOOST);
            for i in lo..hi {
                g_lim_boost[l][i] = g_lim[i] * boost;
                q_m_lim_boost[l][i] = q_m_lim[i] * boost;
                s_m_boost[l][i] = s_m[i] * boost;
            }
        }
    }

    // ---- §4.6.18.7.6 assembly ------------------------------------
    let h_sl: usize = if p.smoothing_mode { 0 } else { 4 };

    // GTemp / QTemp with the hSL-column prefix.
    let mut g_temp = vec![vec![0.0f64; m_cnt]; n_cols + h_sl];
    let mut q_temp = vec![vec![0.0f64; m_cnt]; n_cols + h_sl];
    for j in 0..h_sl {
        if st.g_temp_tail.len() == h_sl && st.g_temp_tail[j].len() == m_cnt {
            g_temp[j].clone_from(&st.g_temp_tail[j]);
            q_temp[j].clone_from(&st.q_temp_tail[j]);
        } else {
            // Reset (or first frame): prefix = first column values.
            g_temp[j].clone_from(&g_lim_boost[0]);
            q_temp[j].clone_from(&q_m_lim_boost[0]);
        }
    }
    // Envelope of column i (spec index space i0..i_end).
    let env_of = |i: i32| -> usize {
        let mut l = l_e - 1;
        for e in 0..l_e {
            if i >= rate * p.t_e[e] && i < rate * p.t_e[e + 1] {
                l = e;
                break;
            }
        }
        l
    };
    for c in 0..n_cols {
        let l = env_of(i0 + c as i32);
        g_temp[c + h_sl].clone_from(&g_lim_boost[l]);
        q_temp[c + h_sl].clone_from(&q_m_lim_boost[l]);
    }

    let mut y = vec![[Complex::default(); 64]; x_high.len()];
    let mut f_index_noise = 0usize;
    let mut f_index_sine = 0usize;
    for c in 0..n_cols {
        let i = i0 + c as i32;
        let l = env_of(i);
        let li = l as i32;
        let col = (i + T_HF_ADJ as i32) as usize;
        let smooth_gain = li != p.l_a && li != l_a_prev && h_sl != 0;
        f_index_sine = (st.index_sine + c) % 4;
        let (sin_re, sin_im) = PHI_SIN[f_index_sine];
        for m in 0..m_cnt {
            let k = (k_x as usize) + m;
            // GFilt.
            let g_filt = if smooth_gain {
                (0..=h_sl)
                    .map(|j| g_temp[c + h_sl - j][m] * H_SMOOTH[j])
                    .sum::<f64>()
            } else {
                g_temp[c + h_sl][m]
            };
            // QFilt: zero on transient envelopes and sinusoid bands.
            let q_filt = if li == p.l_a || li == l_a_prev || s_m_boost[l][m] != 0.0 {
                0.0
            } else if h_sl != 0 {
                (0..=h_sl)
                    .map(|j| q_temp[c + h_sl - j][m] * H_SMOOTH[j])
                    .sum::<f64>()
            } else {
                q_temp[c + h_sl][m]
            };

            // W1 = GFilt · XHigh.
            let w1 = x_high[col][k] * g_filt;

            // W2 = W1 + QFilt · V(fIndexNoise).
            f_index_noise = (st.index_noise + c * m_cnt + m + 1) % 512;
            let (v_re, v_im) = NOISE_TABLE[f_index_noise];
            let mut out = Complex::new(w1.re + q_filt * v_re, w1.im + q_filt * v_im);

            // Y = W2 + ψ (sinusoids).
            if s_index[l][m] {
                let s = s_m_boost[l][m];
                let alt = if (m + k_x as usize) % 2 == 1 {
                    -1.0
                } else {
                    1.0
                };
                out.re += s * sin_re;
                out.im += s * alt * sin_im;
            }
            y[col][k] = out;
        }
    }

    // ---- thread cross-frame state --------------------------------
    st.index_noise = if n_cols > 0 {
        f_index_noise
    } else {
        st.index_noise
    };
    st.index_sine = if n_cols > 0 {
        (f_index_sine + 1) % 4
    } else {
        st.index_sine
    };
    st.g_temp_tail = g_temp[n_cols..].to_vec();
    st.q_temp_tail = q_temp[n_cols..].to_vec();
    st.s_index_prev = s_index[l_e - 1].clone();
    st.k_x_prev = k_x;
    st.l_a_prev_frame = p.l_a;
    st.l_e_prev = l_e as i32;

    Ok(y)
}
