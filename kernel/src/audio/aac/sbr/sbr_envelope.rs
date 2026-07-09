// Ported from oxideav-aac (MIT) — Copyright (c) 2026 Karpelès Lab Inc.
// See THIRDPARTY-LICENSES.md.

//! `sbr_envelope()` / `sbr_noise()` raw decode — ISO/IEC 14496-3
//! §4.4.2.8, Tables 4.72–4.73.
//!
//! These two elements carry the SBR spectral-envelope scalefactors and
//! noise-floor scalefactors as **delta values** (`bs_data_env` /
//! `bs_data_noise`). For each envelope (resp. noise floor) the delta
//! direction comes from `sbr_dtdf()` ([`super::sbr_grid::SbrDtdf`]):
//!
//! * delta-in-**frequency** (`bs_df_* == 0`): the first band carries an
//!   absolute *start value* read as a fixed-width field, and the
//!   remaining bands are frequency-direction Huffman deltas (`f_huff`).
//! * delta-in-**time** (`bs_df_* == 1`): every band is a
//!   time-direction Huffman delta (`t_huff`) relative to the
//!   corresponding band of the previous envelope / noise floor.
//!
//! The start-value field widths (Table 4.72 / 4.73) depend on the
//! coupling / channel / amplitude-resolution context:
//!
//! | element | context | width |
//! |---------|---------|-------|
//! | envelope | coupling && ch, amp_res   | 5 |
//! | envelope | coupling && ch, !amp_res  | 6 |
//! | envelope | level, amp_res            | 6 |
//! | envelope | level, !amp_res           | 7 |
//! | noise    | (any)                     | 5 |
//!
//! The per-envelope band count is `num_env_bands[bs_freq_res]` — the
//! high-resolution band count `NHigh` when the envelope's freq-res flag
//! is set, otherwise the low-resolution count `NLow`
//! ([`super::sbr_freq_bands::HiLoTables`]). The noise band count is
//! `NQ` for every noise floor.
//!
//! This module produces the **raw** delta arrays exactly as written on
//! the wire; the §4.6.18.3.5 DPCM accumulation across bands / time and
//! the §4.6.18 dequantization to linear energies are downstream.

use alloc::vec::Vec;
use super::sbr_grid::{SbrDtdf, SbrGrid};
use super::sbr_huffman::{env_tables, noise_tables, sbr_huff_dec, SbrHuffContext};
use super::Result;
use crate::audio::aac::bits::BitReader;

/// Raw `bs_data_env` for one channel: one delta vector per envelope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SbrEnvelopeData {
    /// `bs_data_env[env][band]` — the raw delta (or, at band 0 of a
    /// frequency-coded envelope, the absolute start value). One inner
    /// vector per envelope; its length is the envelope's band count.
    pub data: Vec<Vec<i32>>,
}

/// Raw `bs_data_noise` for one channel: one delta vector per noise
/// floor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SbrNoiseData {
    /// `bs_data_noise[noise][band]` — raw delta / start value. One
    /// inner vector per noise floor; each has `NQ` entries.
    pub data: Vec<Vec<i32>>,
}

/// The number of envelope bands for a given freq-resolution flag:
/// `NHigh` (high res) or `NLow` (low res).
fn num_env_bands(bands: &super::sbr_freq_bands::HiLoTables, high_res: bool) -> usize {
    if high_res {
        bands.n_high()
    } else {
        bands.n_low()
    }
}

impl SbrEnvelopeData {
    /// Parse `sbr_envelope()` (Table 4.72) for one channel.
    ///
    /// * `grid` / `dtdf` are this channel's already-parsed grid and
    ///   delta-direction flags.
    /// * `bands` supplies `NHigh` / `NLow` for the per-envelope band
    ///   counts.
    /// * `coupling` is the element `bs_coupling`; `ch` is the channel
    ///   index within the element; `amp_res` is the *effective*
    ///   amplitude resolution (after any single-envelope FIXFIX
    ///   override).
    pub fn parse(
        reader: &mut BitReader<'_>,
        grid: &SbrGrid,
        dtdf: &SbrDtdf,
        bands: &super::sbr_freq_bands::HiLoTables,
        coupling: bool,
        ch: bool,
        amp_res: bool,
    ) -> Result<Self> {
        let ctx = SbrHuffContext {
            coupling,
            ch,
            amp_res,
        };
        let ((t_huff, t_lav), (f_huff, f_lav)) = env_tables(ctx);

        // Start-value width per Table 4.72.
        let start_bits = if coupling && ch {
            if amp_res {
                5
            } else {
                6
            }
        } else if amp_res {
            6
        } else {
            7
        };

        let mut data = Vec::with_capacity(grid.num_env);
        for env in 0..grid.num_env {
            let n = num_env_bands(bands, grid.freq_res[env]);
            let mut row = Vec::with_capacity(n);
            if !dtdf.df_env[env] {
                // Delta in frequency: band 0 is the absolute start
                // value, bands 1.. are f_huff deltas.
                let start = read(reader, start_bits)? as i32;
                row.push(start);
                for _ in 1..n {
                    row.push(sbr_huff_dec(reader, f_huff, f_lav)?);
                }
            } else {
                // Delta in time: every band is a t_huff delta.
                for _ in 0..n {
                    row.push(sbr_huff_dec(reader, t_huff, t_lav)?);
                }
            }
            data.push(row);
        }
        Ok(SbrEnvelopeData { data })
    }
}

impl SbrNoiseData {
    /// Parse `sbr_noise()` (Table 4.73) for one channel.
    ///
    /// `num_noise_bands` is `NQ`
    /// ([`super::sbr_freq_bands::HiLoTables::n_q`]). The other
    /// arguments mirror [`SbrEnvelopeData::parse`]; the noise start
    /// value is always a 5-bit field (Table 4.73), regardless of
    /// `amp_res`.
    pub fn parse(
        reader: &mut BitReader<'_>,
        grid: &SbrGrid,
        dtdf: &SbrDtdf,
        num_noise_bands: usize,
        coupling: bool,
        ch: bool,
        amp_res: bool,
    ) -> Result<Self> {
        let ctx = SbrHuffContext {
            coupling,
            ch,
            amp_res,
        };
        let ((t_huff, t_lav), (f_huff, f_lav)) = noise_tables(ctx);

        let mut data = Vec::with_capacity(grid.num_noise);
        for noise in 0..grid.num_noise {
            let mut row = Vec::with_capacity(num_noise_bands);
            if !dtdf.df_noise[noise] {
                // Delta in frequency: band 0 is a 5-bit absolute start
                // value, bands 1.. are f_huff deltas.
                let start = read(reader, 5)? as i32;
                row.push(start);
                for _ in 1..num_noise_bands {
                    row.push(sbr_huff_dec(reader, f_huff, f_lav)?);
                }
            } else {
                for _ in 0..num_noise_bands {
                    row.push(sbr_huff_dec(reader, t_huff, t_lav)?);
                }
            }
            data.push(row);
        }
        Ok(SbrNoiseData { data })
    }
}

#[inline]
fn read(reader: &mut BitReader<'_>, n: u32) -> Result<u32> {
    reader.read_bits(n).map_err(|_| "aac/sbr: huffman invalid")
}
