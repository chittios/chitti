// Ported from oxideav-aac (MIT) — Copyright (c) 2026 Karpelès Lab Inc.
// See THIRDPARTY-LICENSES.md.

//! PS frame driver — ISO/IEC 14496-3:2009 Annex 8.A (combination of
//! the SBR tool with the parametric stereo tool).
//!
//! Composes the whole §8.6.4 chain per stereo frame: `ps_data()`
//! parse (with the persistent header configuration), differential
//! index resolution, the hybrid analysis of the Annex 8.A.3 `Xinput`
//! matrix (32 SBR slots + 6 look-ahead slots from `XLow`),
//! de-correlation with the per-frame partial reset above the
//! SBR-generated spectrum (`kmax = k_x + M + 7` hybrid channels for
//! 10/20 stereo bands, `+ 27` for 34 — the split-region offsets), the
//! §8.6.4.6 stereo mixing, and the hybrid synthesis back to two
//! 64-band QMF matrices ready for the final synthesis filterbanks.
//!
//! Per §8.6.5.1 the decoder stays *inactive* (mono output duplicated
//! by the caller) until the first `ps_data()` that carries
//! `enable_ps_header == 1` arrives; per Annex 8.A.3 a frame with no
//! `ps_data()` after activation holds the previous parameters, and a
//! *missing previous* `ps_data()` forces a full de-correlator reset.
//! Table 8.44 picks the stereo band count from the IID/ICC modes
//! (either at 34 bands → 34, else 20); a switch re-maps the retained
//! mixing coefficients (Table 8.47) and resets the hybrid /
//! de-correlator state.
//!
//! All truth from ISO/IEC 14496-3:2009 subpart 8 + Annex 8.A staged
//! under `docs/audio/aac/`.

use alloc::vec::Vec;
use crate::audio::aac::bits::BitReader;

use super::ps_data::{PsConfig, PsData, PsIndexState};
use super::ps_decorr::PsDecorr;
use super::ps_hybrid::{synthesize, HybridConfig, PsHybrid};
use super::ps_stereo::PsStereo;
use super::sbr_qmf::Complex;
use super::Result;

/// A stereo pair of 64-band QMF matrices (`NUM_QMF_SLOTS` slots).
pub type QmfPair = (Vec<[Complex; 64]>, Vec<[Complex; 64]>);

/// The Annex 8.A PS decoder: one instance per SBR channel element.
#[derive(Debug)]
pub struct PsDecoder {
    /// Persistent `enable_ps_header` configuration (§8.5.2).
    config: Option<PsConfig>,
    /// Cross-frame differential-index state.
    idx_state: PsIndexState,
    hybrid: PsHybrid,
    decorr: PsDecorr,
    stereo: PsStereo,
    /// Whether the previous frame carried a `ps_data()` element
    /// (Annex 8.A.3 full-reset rule).
    prev_frame_had_ps: bool,
    /// Whether a decodable (header-carrying) `ps_data()` has arrived.
    active: bool,
}

impl Default for PsDecoder {
    fn default() -> Self {
        PsDecoder::new()
    }
}

impl PsDecoder {
    /// A fresh, inactive PS decoder (20-band configuration until the
    /// first header says otherwise).
    #[must_use]
    pub fn new() -> Self {
        PsDecoder {
            config: None,
            idx_state: PsIndexState::default(),
            hybrid: PsHybrid::new(HybridConfig::Bands1020),
            decorr: PsDecorr::new(HybridConfig::Bands1020),
            stereo: PsStereo::new(20),
            prev_frame_had_ps: false,
            active: false,
        }
    }

    /// Whether a decodable `ps_data()` has been received — before
    /// this, the caller outputs the mono signal on both channels.
    #[must_use]
    pub fn active(&self) -> bool {
        self.active
    }

    /// Process one stereo frame.
    ///
    /// * `payload` — the raw `sbr_extension()` body bytes carrying
    ///   `ps_data()` (already stripped of the 2-bit extension id), or
    ///   `None` when this frame transmitted no PS data (parameters
    ///   hold).
    /// * `x_input` — the Annex 8.A.3 `Xinput` matrix:
    ///   `NUM_QMF_SLOTS + LOOKAHEAD` slots of 64 QMF bands (the
    ///   look-ahead tail needs only the split bands populated).
    /// * `kx_plus_m` — `k_x + M` (§4.6.18.3.2.2): the first QMF band
    ///   above the SBR-generated spectrum, for the per-frame partial
    ///   de-correlator reset (pass 32 for a pure-upsampled frame).
    ///
    /// Returns `Ok(None)` while inactive (§8.6.5.1 — the caller
    /// duplicates the mono synthesis), otherwise the left/right QMF
    /// matrices for two independent §4.6.18.4.2 synthesis banks.
    pub fn process(
        &mut self,
        payload: Option<&[u8]>,
        x_input: &[[Complex; 64]],
        kx_plus_m: usize,
    ) -> Result<Option<QmfPair>> {
        // Parse (and activate on the first header'd element).
        let parsed: Option<PsData> = match payload {
            Some(bytes) => {
                let mut reader = BitReader::new(bytes);
                PsData::parse(&mut reader, self.config.as_ref())?
            }
            None => None,
        };
        if let Some(ps) = &parsed {
            self.config = Some(ps.config);
            self.active = true;
        }
        let Some(config) = self.config else {
            // Not yet decodable: mono until a header arrives.
            self.prev_frame_had_ps = payload.is_some();
            return Ok(None);
        };
        if !self.active {
            self.prev_frame_had_ps = payload.is_some();
            return Ok(None);
        }

        // Table 8.44: 34 stereo bands iff either parameter kind runs
        // on the 34-band grid; disabled kinds count as 20.
        let bands34 = (config.enable_iid && config.iid_mode % 3 == 2)
            || (config.enable_icc && config.icc_mode % 3 == 2);
        let hcfg = if bands34 {
            HybridConfig::Bands34
        } else {
            HybridConfig::Bands1020
        };
        if hcfg != self.hybrid.config() {
            // Table 8.47: instantaneous filterbank switch, coefficient
            // re-map, de-correlator reset.
            self.hybrid.reset(hcfg);
            self.decorr = PsDecorr::new(hcfg);
            self.stereo.switch_bands(if bands34 { 34 } else { 20 });
        }

        // Annex 8.A.3 resets: full when the previous frame had no
        // ps_data(); otherwise partial above the SBR spectrum.
        if !self.prev_frame_had_ps {
            self.decorr.reset_bands(0);
        } else {
            let off = if bands34 { 27 } else { 7 };
            let kmax = (kx_plus_m + off).min(hcfg.nr_bands());
            self.decorr.reset_bands(kmax);
        }

        // The hold element for a frame with no (new) parameters.
        let ps = parsed.unwrap_or_else(|| hold_element(config));
        let idx = ps.resolve(&mut self.idx_state)?;

        // Hybrid analysis → de-correlation → stereo mixing →
        // hybrid synthesis.
        let s = self.hybrid.analyze(x_input)?;
        let d = self.decorr.process(&s)?;
        let (l, r) = self.stereo.process(&ps, &idx, hcfg, &s, &d)?;
        let l_qmf = synthesize(hcfg, &l);
        let r_qmf = synthesize(hcfg, &r);

        self.prev_frame_had_ps = payload.is_some();
        Ok(Some((l_qmf, r_qmf)))
    }
}

/// A `num_env == 0` element holding the previous parameters
/// (§8.6.4.6.5 / Table 8.50–8.52).
fn hold_element(config: PsConfig) -> PsData {
    PsData {
        header_present: false,
        config,
        frame_class: false,
        num_env: 0,
        border_position: Vec::new(),
        iid_dt: Vec::new(),
        iid_deltas: Vec::new(),
        icc_dt: Vec::new(),
        icc_deltas: Vec::new(),
        enable_ipdopd: false,
        ipd_dt: Vec::new(),
        ipd_deltas: Vec::new(),
        opd_dt: Vec::new(),
        opd_deltas: Vec::new(),
    }
}
