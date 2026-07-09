// Ported from oxideav-aac (MIT) — Copyright (c) 2026 Karpelès Lab Inc.
// See THIRDPARTY-LICENSES.md.

//! SBR frame driver — ISO/IEC 14496-3 §4.6.18.5 "SBR tool overview".
//!
//! Composes the whole SBR back-end for one channel element (SCE or
//! CPE): the §4.6.18.4.1 analysis QMF of the core decoder output, the
//! `XLow` buffer with its `tHFGen = 8`-slot cross-frame history, the
//! §4.6.18.6 HF generator, the §4.6.18.7 envelope adjuster, the
//! §4.6.18.5 output matrix `X` assembly (the `lTemp` splice of the
//! previous frame's `Y'` against the current `XLow` / `Y`), and the
//! §4.6.18.4.2 64-band synthesis QMF producing `numTimeSlots·RATE·64 =
//! 2048` output samples per 1024-sample core frame (dual-rate SBR).
//!
//! [`SbrDecoder::process_frame`] drives a parsed
//! [`super::sbr_extension::SbrExtensionData`];
//! [`SbrDecoder::upsample_frame`] is the §4.6.18.5 "pure upsampling
//! without SBR processing" path used when a frame carries no SBR
//! payload, keeping the 2× output rate and the QMF state continuous.
//!
//! ## Provenance
//!
//! The buffer geometry (`tHFGen = 8`, `tHFAdj = 2`, `lf =
//! numTimeSlots·RATE = 32`), the `XLow` history splice, the `lTemp`
//! output splice, and the reset rules are from the §4.6.18.5 text and
//! Figure 4.47 of the staged spec. No part of this implementation is
//! derived from any external decoder.

use alloc::vec::Vec;
use alloc::vec;
use super::ps_decoder::PsDecoder;
use super::ps_hybrid::LOOKAHEAD;
use super::sbr_dequant::{dequant_coupled, dequant_single, DequantizedSbr};
use super::sbr_element::EXTENSION_ID_PS;
use super::sbr_env_adjust::{adjust, EnvAdjustState, EnvParams};
use super::sbr_extension::SbrExtensionData;
use super::sbr_freq_bands::{k0 as derive_k0, k2 as derive_k2, master_table, HiLoTables};
use super::sbr_header::SbrHeader;
use super::sbr_hf_gen::{build_patches, chirp_factors, generate_hf, Patches, T_HF_ADJ, T_HF_GEN};
use super::sbr_limiter::limiter_table;
use super::sbr_qmf::{AnalysisQmf, Complex, SynthesisQmf};
use super::sbr_reconstruct::{EnvelopeScalefactors, NoiseScalefactors};
use super::sbr_time_grid::derive_time_grid;
use super::Result;

/// `numTimeSlots` for the 1024-sample core frame (§4.6.18.2.6).
pub const NUM_TIME_SLOTS: i32 = 16;

/// `RATE = 2` (§4.6.18.2.5).
pub const RATE: i32 = 2;

/// Slots per frame at the SBR rate (`lf = numTimeSlots · RATE`).
const LF: usize = (NUM_TIME_SLOTS * RATE) as usize;

/// Total `XLow` / `XHigh` / `Y` columns (`lf + tHFGen`).
const COLS: usize = LF + T_HF_GEN;

/// Per-channel cross-frame state.
#[derive(Debug)]
struct ChannelState {
    analysis: AnalysisQmf,
    synthesis: SynthesisQmf,
    /// The previous frame's last `tHFGen` analysis slots (`W'`).
    w_hist: Vec<[Complex; 32]>,
    /// The previous frame's `Y` buffer (spec absolute columns).
    y_prev: Vec<[Complex; 64]>,
    /// `tE'(LE')` — the previous frame's trailing envelope border.
    t_e_last_prev: i32,
    /// The previous frame's `kx` / `M` (for the `lTemp` splice).
    k_x_prev: i32,
    m_prev: i32,
    env_state: EnvAdjustState,
    prev_invf: Vec<u8>,
    prev_bw: Vec<f64>,
    prev_env: Option<EnvelopeScalefactors>,
    prev_noise: Option<NoiseScalefactors>,
}

impl ChannelState {
    fn new() -> Self {
        ChannelState {
            analysis: AnalysisQmf::new(),
            synthesis: SynthesisQmf::new(),
            w_hist: vec![[Complex::default(); 32]; T_HF_GEN],
            y_prev: vec![[Complex::default(); 64]; COLS],
            t_e_last_prev: NUM_TIME_SLOTS,
            k_x_prev: 0,
            m_prev: 0,
            env_state: EnvAdjustState::new(),
            prev_invf: Vec::new(),
            prev_bw: Vec::new(),
            prev_env: None,
            prev_noise: None,
        }
    }

    /// Run the analysis QMF over one 1024-sample core frame and build
    /// the `XLow` buffer: columns `0..tHFGen` are the previous frame's
    /// trailing slots (`W'`), columns `tHFGen..` the current `W`.
    fn analyze(&mut self, core: &[f64]) -> Result<Vec<[Complex; 32]>> {
        if core.len() != 1024 {
            return Err("aac/sbr: qmf invalid");
        }
        let mut x_low = Vec::with_capacity(COLS);
        x_low.extend_from_slice(&self.w_hist);
        for slot in 0..LF {
            let w = self.analysis.push_slot(&core[slot * 32..(slot + 1) * 32])?;
            x_low.push(w);
        }
        self.w_hist.clear();
        self.w_hist.extend_from_slice(&x_low[COLS - T_HF_GEN..]);
        Ok(x_low)
    }
}

/// One SBR decoder per channel element (SCE: 1 channel, CPE: 2).
#[derive(Debug)]
pub struct SbrDecoder {
    fs_sbr: u32,
    header: Option<SbrHeader>,
    bands: Option<HiLoTables>,
    patches: Option<Patches>,
    f_table_lim: Vec<i32>,
    channels: Vec<ChannelState>,
    /// Annex 8.A parametric stereo state, created when a
    /// single-channel element first carries a PS extension. Holds the
    /// PS decoder plus the second (right-channel) synthesis bank; the
    /// channel's own bank renders the left channel.
    ps: Option<PsState>,
}

/// PS decoder + right-channel synthesis bank (Annex 8.A).
#[derive(Debug)]
struct PsState {
    dec: PsDecoder,
    synthesis_r: SynthesisQmf,
}

impl SbrDecoder {
    /// A fresh SBR decoder. `fs_sbr` is the SBR internal rate (twice
    /// the core rate); `num_channels` is 1 (SCE) or 2 (CPE). Invalid
    /// inputs are clamped (never panics).
    pub fn new(fs_sbr: u32, num_channels: usize) -> Self {
        let num_channels = num_channels.clamp(1, 2);
        let fs_sbr = fs_sbr.max(1);
        SbrDecoder {
            fs_sbr,
            header: None,
            bands: None,
            patches: None,
            f_table_lim: Vec::new(),
            channels: (0..num_channels).map(|_| ChannelState::new()).collect(),
            ps: None,
        }
    }

    /// §4.6.18.5 pure upsampling: no SBR data for this frame — run the
    /// analysis / synthesis pair with the high 32 bands zero, keeping
    /// the output at 2× the core rate and the QMF state continuous.
    ///
    /// `core` holds one 1024-sample time signal per channel; returns
    /// 2048 samples per channel.
    pub fn upsample_frame(&mut self, core: &[&[f64]]) -> Result<Vec<Vec<f64>>> {
        if core.len() != self.channels.len() {
            return Err("aac/sbr: qmf invalid");
        }
        let mut out = Vec::with_capacity(core.len());
        let n_ch = self.channels.len();
        for (ch, core_ch) in self.channels.iter_mut().zip(core.iter()) {
            let x_low = ch.analyze(core_ch)?;
            let mut x_cols: Vec<[Complex; 64]> = Vec::with_capacity(LF);
            for l in 0..LF {
                let mut x = [Complex::default(); 64];
                x[..32].copy_from_slice(&x_low[l + T_HF_ADJ]);
                x_cols.push(x);
            }
            // A PS-active stream holds its stereo parameters over a
            // frame without SBR/PS payload (Annex 8.A.3); the whole
            // 32-band spectrum counts as SBR-covered for the partial
            // reset.
            let mut emitted = false;
            if n_ch == 1 {
                if let Some(ps) = self.ps.as_mut() {
                    let x_input = build_x_input(&x_cols, &x_low);
                    if let Some((lq, rq)) = ps.dec.process(None, &x_input, 32)? {
                        let mut pcm_l = Vec::with_capacity(LF * 64);
                        let mut pcm_r = Vec::with_capacity(LF * 64);
                        for l in 0..LF {
                            pcm_l.extend_from_slice(&ch.synthesis.push_slot(&lq[l])?);
                            pcm_r.extend_from_slice(&ps.synthesis_r.push_slot(&rq[l])?);
                        }
                        out.push(pcm_l);
                        out.push(pcm_r);
                        emitted = true;
                    }
                }
            }
            if !emitted {
                let mut pcm = Vec::with_capacity(LF * 64);
                for x in &x_cols {
                    pcm.extend_from_slice(&ch.synthesis.push_slot(x)?);
                }
                out.push(pcm);
            }
            // No Y for this frame; the next frame's lTemp splice sees
            // an empty previous envelope span.
            ch.y_prev
                .iter_mut()
                .for_each(|c| *c = [Complex::default(); 64]);
            ch.t_e_last_prev = NUM_TIME_SLOTS;
        }
        Ok(out)
    }

    /// Decode one SBR frame: `ext` is the parsed `sbr_extension_data()`
    /// for this element, `core` one 1024-sample signal per channel.
    /// Returns 2048 samples per channel at the SBR rate.
    pub fn process_frame(
        &mut self,
        ext: &SbrExtensionData,
        core: &[&[f64]],
    ) -> Result<Vec<Vec<f64>>> {
        let n_ch = self.channels.len();
        if core.len() != n_ch || ext.element.channels.len() != n_ch {
            return Err("aac/sbr: freq band invalid");
        }

        // §4.6.18.3.3 reset: first header, or a transmitted header that
        // changes the band geometry.
        let reset = match &self.header {
            None => true,
            Some(prev) => prev.band_geometry_changed(&ext.header),
        };
        if reset {
            let k0v = derive_k0(self.fs_sbr, ext.header.start_freq)?;
            let k2v = derive_k2(self.fs_sbr, ext.header.stop_freq, k0v)?;
            let f_master = master_table(k0v, k2v, ext.header.freq_scale, ext.header.alter_scale)?;
            let bands =
                HiLoTables::derive(&f_master, ext.header.xover_band, ext.header.noise_bands)?;
            let patches = build_patches(&f_master, k0v, bands.k_x, bands.m, self.fs_sbr)?;
            self.f_table_lim = limiter_table(
                &bands,
                &patches.borders(bands.k_x),
                ext.header.limiter_bands,
            )?;
            self.bands = Some(bands);
            self.patches = Some(patches);
            for ch in &mut self.channels {
                ch.prev_invf.clear();
                ch.prev_bw.clear();
                ch.prev_env = None;
                ch.prev_noise = None;
            }
        }
        self.header = Some(ext.header);
        let bands = self.bands.as_ref().ok_or("aac/sbr: freq band invalid")?;
        let patches = self.patches.as_ref().ok_or("aac/sbr: freq band invalid")?;

        let coupling = ext.element.coupling;

        // Reconstruct the quantized scalefactors per transmitted
        // channel, then dequantize (jointly for a coupled pair).
        let mut recon: Vec<(EnvelopeScalefactors, NoiseScalefactors)> = Vec::with_capacity(n_ch);
        for (c, sbr_ch) in ext.element.channels.iter().enumerate() {
            let st = &self.channels[c];
            let env = EnvelopeScalefactors::reconstruct(
                &sbr_ch.envelope,
                &sbr_ch.grid,
                &sbr_ch.dtdf,
                bands,
                coupling,
                c == 1,
                if reset { None } else { st.prev_env.as_ref() },
            )?;
            let noise = NoiseScalefactors::reconstruct(
                &sbr_ch.noise,
                &sbr_ch.grid,
                &sbr_ch.dtdf,
                bands.n_q(),
                coupling,
                c == 1,
                if reset { None } else { st.prev_noise.as_ref() },
            )?;
            recon.push((env, noise));
        }

        let dequant: Vec<DequantizedSbr> = if coupling && n_ch == 2 {
            let amp_res = effective_amp_res(&ext.header, &ext.element.channels[0].grid);
            let (l, r) =
                dequant_coupled(&recon[0].0, &recon[0].1, &recon[1].0, &recon[1].1, amp_res);
            vec![l, r]
        } else {
            (0..n_ch)
                .map(|c| {
                    let amp_res = effective_amp_res(&ext.header, &ext.element.channels[c].grid);
                    dequant_single(&recon[c].0, &recon[c].1, amp_res)
                })
                .collect()
        };

        let mut out = Vec::with_capacity(n_ch);
        for c in 0..n_ch {
            let sbr_ch = &ext.element.channels[c];
            let grid = derive_time_grid(&sbr_ch.grid, NUM_TIME_SLOTS)?;

            // Coupling: the second channel transmits no sbr_invf()
            // (Table 4.66) — it shares the first channel's
            // inverse-filtering modes.
            let invf_modes = if coupling && c == 1 {
                &ext.element.channels[0].invf.invf_mode
            } else {
                &sbr_ch.invf.invf_mode
            };

            let ch = &mut self.channels[c];

            // Chirp factors (per noise band).
            let bw = chirp_factors(invf_modes, &ch.prev_invf, &ch.prev_bw);

            // Analysis + XLow (with tHFGen history).
            let x_low = ch.analyze(core[c])?;

            // HF generation over the envelope span.
            let l_range = (RATE * grid.t_e[0])..(RATE * grid.t_e[grid.t_e.len() - 1]);
            let x_high = generate_hf(&x_low, patches, &bw, bands, l_range, LF)?;

            // Envelope adjustment.
            let freq_res: Vec<bool> = sbr_ch.grid.freq_res.clone();
            let params = EnvParams {
                bands,
                f_table_lim: &self.f_table_lim,
                t_e: &grid.t_e,
                t_q: &grid.t_q,
                freq_res: &freq_res,
                l_a: grid.l_a,
                e_orig: &dequant[c].e_orig,
                q_orig: &dequant[c].q_orig,
                add_harmonic: &sbr_ch.add_harmonic,
                interpol_freq: ext.header.interpol_freq,
                smoothing_mode: ext.header.smoothing_mode,
                limiter_gains: ext.header.limiter_gains,
                reset,
            };
            let y = adjust(&x_high, &params, &mut ch.env_state)?;

            // §4.6.18.5 X assembly.
            let l_temp = (RATE * ch.t_e_last_prev - NUM_TIME_SLOTS * RATE).max(0) as usize;
            let mut x_cols: Vec<[Complex; 64]> = Vec::with_capacity(LF);
            for l in 0..LF {
                let mut x = [Complex::default(); 64];
                let (kx_cur, m_cur, y_col) = if l < l_temp {
                    (ch.k_x_prev, ch.m_prev, &ch.y_prev[l + T_HF_ADJ + LF])
                } else {
                    (bands.k_x, bands.m, &y[l + T_HF_ADJ])
                };
                let kx_u = kx_cur.max(0) as usize;
                for (k, cell) in x.iter_mut().enumerate().take(kx_u.min(32)) {
                    *cell = x_low[l + T_HF_ADJ][k];
                }
                let hi = (kx_cur + m_cur).max(0) as usize;
                let hi = hi.min(64);
                if kx_u < hi {
                    x[kx_u..hi].copy_from_slice(&y_col[kx_u..hi]);
                }
                x_cols.push(x);
            }

            // Annex 8.A: a single-channel element carrying an
            // EXTENSION_ID_PS payload renders stereo through the PS
            // tool (the element's own bank = left, the PS state's =
            // right). Until the first decodable ps_data() the mono
            // path below stays in effect.
            let ps_payload = if n_ch == 1 {
                ext.element
                    .extension
                    .as_ref()
                    .filter(|e| e.id == EXTENSION_ID_PS)
                    .map(|e| e.data.as_slice())
            } else {
                None
            };
            if ps_payload.is_some() && self.ps.is_none() {
                self.ps = Some(PsState {
                    dec: PsDecoder::new(),
                    synthesis_r: SynthesisQmf::new(),
                });
            }
            let mut emitted = false;
            if n_ch == 1 {
                if let Some(ps) = self.ps.as_mut() {
                    let x_input = build_x_input(&x_cols, &x_low);
                    let kx_plus_m = (bands.k_x + bands.m).max(0) as usize;
                    if let Some((lq, rq)) = ps.dec.process(ps_payload, &x_input, kx_plus_m)? {
                        let mut pcm_l = Vec::with_capacity(LF * 64);
                        let mut pcm_r = Vec::with_capacity(LF * 64);
                        for l in 0..LF {
                            pcm_l.extend_from_slice(&ch.synthesis.push_slot(&lq[l])?);
                            pcm_r.extend_from_slice(&ps.synthesis_r.push_slot(&rq[l])?);
                        }
                        out.push(pcm_l);
                        out.push(pcm_r);
                        emitted = true;
                    }
                }
            }
            if !emitted {
                let mut pcm = Vec::with_capacity(LF * 64);
                for x in &x_cols {
                    pcm.extend_from_slice(&ch.synthesis.push_slot(x)?);
                }
                out.push(pcm);
            }

            // Thread cross-frame state.
            ch.y_prev = y;
            ch.t_e_last_prev = grid.t_e[grid.t_e.len() - 1];
            ch.k_x_prev = bands.k_x;
            ch.m_prev = bands.m;
            ch.prev_invf = invf_modes.clone();
            ch.prev_bw = bw;
            let (env, noise) = recon[c].clone();
            ch.prev_env = Some(env);
            ch.prev_noise = Some(noise);
        }
        Ok(out)
    }
}

/// Assemble the Annex 8.A.3 `Xinput` matrix: the 32 assembled `X`
/// columns followed by `LOOKAHEAD` slots taken from `XLow` beyond the
/// frame (`XLow(k, l + tHFAdj)`, `k < 5` — the split bands the hybrid
/// filterbank consumes ahead of time).
fn build_x_input(x_cols: &[[Complex; 64]], x_low: &[[Complex; 32]]) -> Vec<[Complex; 64]> {
    let mut v = Vec::with_capacity(LF + LOOKAHEAD);
    v.extend_from_slice(x_cols);
    for l in LF..LF + LOOKAHEAD {
        let mut col = [Complex::default(); 64];
        col[..5].copy_from_slice(&x_low[l + T_HF_ADJ][..5]);
        v.push(col);
    }
    v
}

/// The effective `bs_amp_res` after the single-envelope FIXFIX
/// override (§4.4.2.8 Table 4.69 Note).
fn effective_amp_res(header: &SbrHeader, grid: &super::sbr_grid::SbrGrid) -> bool {
    if grid.amp_res_override {
        false
    } else {
        header.amp_res
    }
}
