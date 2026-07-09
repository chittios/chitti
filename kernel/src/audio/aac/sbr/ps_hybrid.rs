// Ported from oxideav-aac (MIT) — Copyright (c) 2026 Karpelès Lab Inc.
// See THIRDPARTY-LICENSES.md.

//! PS hybrid filterbank — ISO/IEC 14496-3:2009 §8.6.4.3 / Annex 8.A.3.
//!
//! Parametric Stereo needs a finer frequency resolution at the bottom
//! of the spectrum than the 64-band QMF provides, so the lowest QMF
//! subbands are split further by 13-tap prototype filters (Tables
//! 8.36–8.38), producing the *hybrid* sub-subband domain:
//!
//! * **10/20 stereo bands** — QMF band 0 split by 8 (Type A, complex
//!   modulated) with the outer sub-subband pairs merged to 6 channels,
//!   QMF bands 1 and 2 split by 2 (Type B, cosine modulated); 71
//!   hybrid channels total (`6 + 2 + 2 + 61`).
//! * **34 stereo bands** — QMF band 0 split by 12, band 1 by 8, bands
//!   2–4 by 4 (all Type A); 91 hybrid channels (`12+8+4+4+4 + 59`).
//!
//! ```text
//! Type A: G_q^p[n] = g^p[n] · exp(j·2π/Q^p·(q+1/2)·(n−6))
//! Type B: G_q^p[n] = g^p[n] · cos(2π·q/Q^p·(n−6))
//! ```
//!
//! The prototypes are linear-phase with a 6-slot delay; per Annex
//! 8.A.3 the SBR combination feeds the filterbank 6 *look-ahead* QMF
//! slots (`XLow` beyond the current frame), so the hybrid output is
//! time-aligned with the QMF input at **zero net delay**: the unsplit
//! bands pass straight through and the split bands consume the
//! look-ahead. Filtering is the convolution
//! `y[n] = Σ_m G[m] · x[n+6−m]`, needing 6 history slots per split
//! band which [`PsHybrid`] threads across frames.
//!
//! ## Channel ordering (Figures 8.20 / 8.22)
//!
//! For the 10/20 configuration QMF band 0's eight Type-A outputs `q`
//! (sub-subband centres `(q+1/2)·π/8`, `q ≥ 4` the negative-frequency
//! mirrors) merge and reorder to six hybrid channels:
//! `s0 = q6, s1 = q7, s2 = q0, s3 = q1, s4 = q2+q5, s5 = q3+q4`.
//! QMF band 1's two Type-B outputs land **swapped** (`s6 = q1,
//! s7 = q0` — odd QMF bands are spectrally inverted), band 2's in
//! order (`s8 = q0, s9 = q1`). The 34-band configuration keeps every
//! split output in filter order (Figure 8.22).
//!
//! The synthesis (§8.6.4.7 / Figures 8.21, 8.23) is a plain adder:
//! sub-subbands of a split QMF band sum back into that band. Because
//! each prototype's sub-filters sum to a pure 6-slot delay (the
//! Type-A modulation phases cancel off-centre, the Type-B prototypes
//! vanish at the surviving off-centre taps), analysis followed by
//! synthesis reconstructs the input exactly — pinned by the tests.
//!
//! All truth from ISO/IEC 14496-3:2009 §8.6.4.3 / Annex 8.A staged
//! under `docs/audio/aac/`.

use alloc::vec::Vec;
use alloc::vec;
use super::sbr_qmf::Complex;
use super::Result;

/// QMF slots per PS stereo frame in the SBR combination
/// (`numQMFSlots = numTimeSlots · RATE`, Annex 8.A.3, 1024 framing).
use super::math::F64Ext;
pub const NUM_QMF_SLOTS: usize = 32;

/// Look-ahead slots supplied by the SBR low-band buffer (Annex 8.A.3).
pub const LOOKAHEAD: usize = 6;

/// Prototype filter length (§8.6.4.3).
const PROTO_LEN: usize = 13;

/// Table 8.37 — `g⁰[n]`, `Q⁰ = 8` (10/20 stereo bands, QMF band 0).
const G0_Q8: [f64; PROTO_LEN] = [
    0.00746082949812,
    0.02270420949825,
    0.04546865930473,
    0.07266113929591,
    0.09885108575264,
    0.11793710567217,
    0.125,
    0.11793710567217,
    0.09885108575264,
    0.07266113929591,
    0.04546865930473,
    0.02270420949825,
    0.00746082949812,
];

/// Table 8.37 — `g^{1,2}[n]`, `Q^{1,2} = 2` (10/20 bands, QMF 1–2).
const G12_Q2: [f64; PROTO_LEN] = [
    0.0,
    0.01899487526049,
    0.0,
    -0.07293139167538,
    0.0,
    0.30596630545168,
    0.5,
    0.30596630545168,
    0.0,
    -0.07293139167538,
    0.0,
    0.01899487526049,
    0.0,
];

/// Table 8.38 — `g⁰[n]`, `Q⁰ = 12` (34 stereo bands, QMF band 0).
const G0_Q12: [f64; PROTO_LEN] = [
    0.04081179924692,
    0.03812810994926,
    0.05144908135699,
    0.06399831151592,
    0.07428313801106,
    0.08100347892914,
    0.08333333333333,
    0.08100347892914,
    0.07428313801106,
    0.06399831151592,
    0.05144908135699,
    0.03812810994926,
    0.04081179924692,
];

/// Table 8.38 — `g¹[n]`, `Q¹ = 8` (34 bands, QMF band 1).
const G1_Q8: [f64; PROTO_LEN] = [
    0.01565675600122,
    0.03752716391991,
    0.05417891378782,
    0.08417044116767,
    0.10307344158036,
    0.12222452249753,
    0.125,
    0.12222452249753,
    0.10307344158036,
    0.08417044116767,
    0.05417891378782,
    0.03752716391991,
    0.01565675600122,
];

/// Table 8.38 — `g^{2,3,4}[n]`, `Q^{2,3,4} = 4` (34 bands, QMF 2–4).
const G234_Q4: [f64; PROTO_LEN] = [
    -0.05908211155639,
    -0.04871498374946,
    0.0,
    0.07778723915851,
    0.16486303567403,
    0.23279856662996,
    0.25,
    0.23279856662996,
    0.16486303567403,
    0.07778723915851,
    0.0,
    -0.04871498374946,
    -0.05908211155639,
];

/// The two §8.6.4.3 hybrid configurations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridConfig {
    /// 10 or 20 stereo bands: 71 hybrid channels, QMF bands 0–2 split.
    Bands1020,
    /// 34 stereo bands: 91 hybrid channels, QMF bands 0–4 split.
    Bands34,
}

impl HybridConfig {
    /// `NR_BANDS` — hybrid channel count (§8.6.4.5.1).
    #[must_use]
    pub fn nr_bands(&self) -> usize {
        match self {
            HybridConfig::Bands1020 => 71,
            HybridConfig::Bands34 => 91,
        }
    }

    /// Number of QMF bands that are split.
    fn split_bands(&self) -> usize {
        match self {
            HybridConfig::Bands1020 => 3,
            HybridConfig::Bands34 => 5,
        }
    }

    /// Split factor `Q^p` per split QMF band.
    fn q(&self, p: usize) -> usize {
        match self {
            HybridConfig::Bands1020 => [8, 2, 2][p],
            HybridConfig::Bands34 => [12, 8, 4, 4, 4][p],
        }
    }

    /// Prototype `g^p` per split QMF band.
    fn proto(&self, p: usize) -> &'static [f64; PROTO_LEN] {
        match self {
            HybridConfig::Bands1020 => [&G0_Q8, &G12_Q2, &G12_Q2][p],
            HybridConfig::Bands34 => [&G0_Q12, &G1_Q8, &G234_Q4, &G234_Q4, &G234_Q4][p],
        }
    }

    /// Whether split band `p` uses the Type-A (complex) modulation.
    fn type_a(&self, p: usize) -> bool {
        match self {
            HybridConfig::Bands1020 => p == 0,
            HybridConfig::Bands34 => true,
        }
    }
}

/// One channel's hybrid analysis/synthesis state: the 6 history slots
/// per split QMF band that the 13-tap convolution reaches into before
/// the current frame.
#[derive(Debug, Clone)]
pub struct PsHybrid {
    config: HybridConfig,
    /// `history[p][j]` — the previous frame's QMF slots `26..32` for
    /// split band `p` (`j = 0` is the oldest).
    history: Vec<[Complex; LOOKAHEAD]>,
}

impl PsHybrid {
    /// A fresh filterbank for `config` (zero history).
    #[must_use]
    pub fn new(config: HybridConfig) -> Self {
        PsHybrid {
            config,
            history: vec![[Complex::default(); LOOKAHEAD]; config.split_bands()],
        }
    }

    /// The active configuration.
    #[must_use]
    pub fn config(&self) -> HybridConfig {
        self.config
    }

    /// Switch configuration (a §8.6.4.6.1 stereo-band change resets
    /// the filter state instantaneously).
    pub fn reset(&mut self, config: HybridConfig) {
        self.config = config;
        self.history = vec![[Complex::default(); LOOKAHEAD]; config.split_bands()];
    }

    /// Hybrid analysis of one stereo frame.
    ///
    /// `x` is the Annex 8.A.3 `Xinput` matrix: at least
    /// `NUM_QMF_SLOTS + LOOKAHEAD` slots of 64 QMF bands (the trailing
    /// 6 slots only need bands `0..split_bands` populated). Returns
    /// `NUM_QMF_SLOTS` slots of `nr_bands()` hybrid channels, and
    /// advances the cross-frame history.
    pub fn analyze(&mut self, x: &[[Complex; 64]]) -> Result<Vec<Vec<Complex>>> {
        if x.len() < NUM_QMF_SLOTS + LOOKAHEAD {
            return Err("aac/sbr: ps data invalid");
        }
        let nb = self.config.nr_bands();
        let split = self.config.split_bands();
        let mut out = vec![vec![Complex::default(); nb]; NUM_QMF_SLOTS];

        for p in 0..split {
            // Extended buffer: 6 history slots + the frame + look-ahead.
            let mut buf = [Complex::default(); LOOKAHEAD + NUM_QMF_SLOTS + LOOKAHEAD];
            buf[..LOOKAHEAD].copy_from_slice(&self.history[p]);
            for (j, slot) in x.iter().enumerate().take(NUM_QMF_SLOTS + LOOKAHEAD) {
                buf[LOOKAHEAD + j] = slot[p];
            }
            let q_cnt = self.config.q(p);
            let g = self.config.proto(p);
            let type_a = self.config.type_a(p);
            for q in 0..q_cnt {
                // G_q[m] for m = 0..13.
                let mut filt = [Complex::default(); PROTO_LEN];
                for (m, f) in filt.iter_mut().enumerate() {
                    let arg = if type_a {
                        2.0 * core::f64::consts::PI / q_cnt as f64
                            * (q as f64 + 0.5)
                            * (m as f64 - 6.0)
                    } else {
                        2.0 * core::f64::consts::PI * q as f64 / q_cnt as f64 * (m as f64 - 6.0)
                    };
                    let (s, c) = arg.sin_cos();
                    *f = if type_a {
                        Complex::new(g[m] * c, g[m] * s)
                    } else {
                        Complex::new(g[m] * c, 0.0)
                    };
                }
                for (n, row) in out.iter_mut().enumerate() {
                    // y[n] = Σ_m G[m]·x[n+6−m]; buf[j] = x[j−6].
                    let mut acc = Complex::default();
                    for (m, &f) in filt.iter().enumerate() {
                        acc += f * buf[n + 12 - m];
                    }
                    accumulate_channel(&self.config, p, q, acc, row);
                }
            }
            // Next frame's x[−6..0] are this frame's slots 26..32.
            for j in 0..LOOKAHEAD {
                self.history[p][j] = x[NUM_QMF_SLOTS - LOOKAHEAD + j][p];
            }
        }

        // Unsplit QMF bands pass through at zero delay.
        for (n, row) in out.iter_mut().enumerate() {
            for k in split..64 {
                row[hybrid_offset(&self.config) + k - split] = x[n][k];
            }
        }
        Ok(out)
    }
}

/// First hybrid channel index of the unsplit QMF region.
fn hybrid_offset(config: &HybridConfig) -> usize {
    match config {
        HybridConfig::Bands1020 => 10,
        HybridConfig::Bands34 => 32,
    }
}

/// Route split-band filter output `q` of QMF band `p` into its hybrid
/// channel (Figures 8.20 / 8.22), merging where the 10/20
/// configuration combines sub-subbands.
fn accumulate_channel(config: &HybridConfig, p: usize, q: usize, v: Complex, row: &mut [Complex]) {
    match config {
        HybridConfig::Bands1020 => match p {
            0 => {
                // s0=q6, s1=q7, s2=q0, s3=q1, s4=q2+q5, s5=q3+q4.
                let k = match q {
                    6 => 0,
                    7 => 1,
                    0 => 2,
                    1 => 3,
                    2 | 5 => 4,
                    _ => 5, // 3 | 4
                };
                row[k] += v;
            }
            1 => {
                // Spectrally inverted odd QMF band: s6=q1, s7=q0.
                row[if q == 0 { 7 } else { 6 }] += v;
            }
            _ => {
                // Band 2 in order: s8=q0, s9=q1.
                row[8 + q] += v;
            }
        },
        HybridConfig::Bands34 => {
            // Figure 8.22: filter order, bands packed consecutively.
            let base = [0usize, 12, 20, 24, 28][p];
            row[base + q] += v;
        }
    }
}

/// Hybrid synthesis (§8.6.4.7): sum each split QMF band's sub-subbands
/// back into the band; copy the unsplit region. `rows` are
/// `nr_bands()`-wide hybrid slots; returns 64-band QMF slots.
#[must_use]
pub fn synthesize(config: HybridConfig, rows: &[Vec<Complex>]) -> Vec<[Complex; 64]> {
    let split = config.split_bands();
    let off = hybrid_offset(&config);
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let mut slot = [Complex::default(); 64];
        // Per-band sub-subband spans in the hybrid row.
        let spans: &[(usize, usize)] = match config {
            HybridConfig::Bands1020 => &[(0, 6), (6, 8), (8, 10)],
            HybridConfig::Bands34 => &[(0, 12), (12, 20), (20, 24), (24, 28), (28, 32)],
        };
        for (p, &(lo, hi)) in spans.iter().enumerate() {
            for v in &row[lo..hi] {
                slot[p] += *v;
            }
        }
        for k in split..64 {
            slot[k] = row[off + k - split];
        }
        out.push(slot);
    }
    out
}
