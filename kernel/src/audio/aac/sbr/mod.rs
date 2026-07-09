//! HE-AAC Spectral Band Replication (SBR) + Parametric Stereo (PS).
//!
//! Ported from [oxideav-aac](https://github.com/OxideAV/oxideav-aac) (MIT,
//! Copyright (c) 2026 Karpelès Lab Inc.) — see `THIRDPARTY-LICENSES.md`.
//! Adapted for Chitti's `no_std` + `alloc` AAC decoder: oxideav `BitReader`
//! → [`crate::audio::aac::bits::BitReader`], errors → `&'static str`.
//!
//! Dual-rate SBR: 1024-sample LC core frames → 2048 samples at 2× rate.

mod bitwriter;
mod math;
mod ps_data;
mod ps_decoder;
mod ps_decorr;
mod ps_huffman;
mod ps_hybrid;
mod ps_map;
mod ps_stereo;
mod sbr_decoder;
mod sbr_dequant;
mod sbr_element;
mod sbr_env_adjust;
mod sbr_envelope;
mod sbr_extension;
mod sbr_freq_bands;
mod sbr_grid;
mod sbr_header;
mod sbr_hf_gen;
mod sbr_huffman;
mod sbr_limiter;
mod sbr_noise_table;
mod sbr_qmf;
mod sbr_reconstruct;
mod sbr_time_grid;

use alloc::vec::Vec;
use sbr_decoder::SbrDecoder;
use sbr_extension::SbrExtensionData;
use sbr_header::SbrHeader;

/// Local result type (oxideav used a typed `Error` enum).
pub type Error = &'static str;
pub type Result<T> = core::result::Result<T, Error>;

/// AAC syntactic element id for SBR attachment (subset of ISO Table 4.71).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdSynEle {
    Sce = 0,
    Cpe = 1,
    Cce = 2,
    Lfe = 3,
    Dse = 4,
    Pce = 5,
    Fil = 6,
    End = 7,
}

/// Stateful HE-AAC SBR (+ optional PS) processor.
///
/// Feed core LC PCM (planar f32, 1024 samples/ch) plus the raw SBR payload
/// from a FIL extension (type 13/14, after the 4-bit extension type). Output
/// is planar f32 at 2× the core rate (2048 samples/ch dual-rate).
pub struct SbrState {
    dec: SbrDecoder,
    core_rate: u32,
    channels: u8,
    prev_header: Option<SbrHeader>,
    /// Last core element type that SBR attaches to (SCE or CPE).
    id_aac: IdSynEle,
    /// True after we've seen any SBR payload (enables upsample fallback).
    ever_had_payload: bool,
    /// Next payload uses EXT_SBR_DATA_CRC (10-bit CRC prefix).
    next_crc: bool,
}

impl SbrState {
    /// `core_rate` is the LC filterbank rate; `channels` is 1 or 2 for the
    /// element SBR enhances (not the full mix count).
    pub fn new(core_rate: u32, channels: u8) -> Self {
        let nch = (channels as usize).clamp(1, 2);
        let core_rate = if core_rate == 0 { 22050 } else { core_rate };
        let fs_sbr = core_rate.saturating_mul(2).max(1);
        Self {
            dec: SbrDecoder::new(fs_sbr, nch),
            core_rate,
            channels: nch as u8,
            prev_header: None,
            id_aac: if nch == 2 {
                IdSynEle::Cpe
            } else {
                IdSynEle::Sce
            },
            ever_had_payload: false,
            next_crc: false,
        }
    }

    /// Hint which core element the next SBR payload extends.
    pub fn set_element_type(&mut self, id: IdSynEle) {
        if matches!(id, IdSynEle::Sce | IdSynEle::Cpe) {
            self.id_aac = id;
        }
    }

    /// Mark the next payload as `EXT_SBR_DATA_CRC` (type 14) vs plain type 13.
    pub fn set_crc(&mut self, crc: bool) {
        self.next_crc = crc;
    }

    pub fn output_rate(&self) -> u32 {
        self.core_rate.saturating_mul(2)
    }

    pub fn core_rate(&self) -> u32 {
        self.core_rate
    }

    /// Channel count of the SBR element (1 = SCE, 2 = CPE).
    pub fn channels(&self) -> u8 {
        self.channels
    }

    /// Process one dual-rate frame.
    ///
    /// * `core` — planar f32, 1024 samples per channel (1 or 2 ch).
    /// * `sbr_payload` — raw `sbr_extension_data()` bits (after the 4-bit
    ///   `extension_type`), or `None` for pure QMF upsampling.
    /// * `stereo` — reserved; PS may expand mono → stereo regardless.
    ///
    /// Returns planar f32 at 2× rate (2048 samples per channel; may be 2 ch
    /// after PS even if core was mono). Call [`Self::set_crc`] first when the
    /// FIL extension type was `EXT_SBR_DATA_CRC` (14).
    pub fn process(
        &mut self,
        core: &[Vec<f32>],
        sbr_payload: Option<&[u8]>,
        _stereo: bool,
    ) -> Result<Vec<Vec<f32>>> {
        if core.is_empty() {
            return Err("aac/sbr: empty core");
        }
        for ch in core {
            if ch.len() != 1024 {
                return Err("aac/sbr: core frame must be 1024 samples");
            }
        }

        let crc = self.next_crc;
        self.next_crc = false;

        // Convert core to f64 planar for the oxideav backend.
        let core_f64: Vec<Vec<f64>> = core
            .iter()
            .map(|c| c.iter().map(|&s| s as f64).collect())
            .collect();

        let n_need = self.channels as usize;
        // If core has more channels than the SBR element, only feed the first n.
        let feed: Vec<&[f64]> = core_f64.iter().take(n_need).map(|v| v.as_slice()).collect();
        if feed.len() != n_need {
            // Resize decoder if channel count mismatch (e.g. first frame stereo).
            if feed.len() == 1 || feed.len() == 2 {
                let rate = self.core_rate;
                *self = SbrState::new(rate, feed.len() as u8);
                self.next_crc = crc;
                return self.process(core, sbr_payload, _stereo);
            }
            return Err("aac/sbr: channel count mismatch");
        }

        let out_f64 = if let Some(payload) = sbr_payload {
            self.ever_had_payload = true;
            let mut br = crate::audio::aac::bits::BitReader::new(payload);
            let ext = match SbrExtensionData::parse(
                &mut br,
                self.id_aac,
                crc,
                self.output_rate(),
                Some(payload.len() as u32),
                self.prev_header,
            ) {
                Ok(e) => e,
                Err(_) => {
                    // Soft-fail: keep QMF continuous via upsample.
                    return self.upsample_f32(core, &core_f64, n_need);
                }
            };
            self.prev_header = Some(ext.header);
            match self.dec.process_frame(&ext, &feed) {
                Ok(o) => o,
                Err(_) => return self.upsample_f32(core, &core_f64, n_need),
            }
        } else if self.ever_had_payload || self.prev_header.is_some() {
            match self.dec.upsample_frame(&feed) {
                Ok(o) => o,
                Err(_) => return self.upsample_f32(core, &core_f64, n_need),
            }
        } else {
            // No SBR ever — still dual-rate upsample so rate stays 2× when ASC says SBR.
            self.dec.upsample_frame(&feed)?
        };

        Ok(out_f64
            .into_iter()
            .map(|ch| ch.into_iter().map(|s| s as f32).collect())
            .collect())
    }

    fn upsample_f32(
        &mut self,
        _core: &[Vec<f32>],
        core_f64: &[Vec<f64>],
        n_need: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let feed: Vec<&[f64]> = core_f64.iter().take(n_need).map(|v| v.as_slice()).collect();
        let out = self.dec.upsample_frame(&feed)?;
        Ok(out
            .into_iter()
            .map(|ch| ch.into_iter().map(|s| s as f32).collect())
            .collect())
    }
}

