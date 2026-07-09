// Ported from oxideav-aac (MIT) — Copyright (c) 2026 Karpelès Lab Inc.
// See THIRDPARTY-LICENSES.md.

//! SBR element framing — `sbr_single_channel_element()` /
//! `sbr_channel_pair_element()` and `sbr_sinusoidal_coding()` —
//! ISO/IEC 14496-3 §4.4.2.8, Tables 4.65, 4.66, 4.74.
//!
//! These wrappers tie the per-channel grid / dtdf / invf / envelope /
//! noise parses together into a whole SBR data element, in the exact
//! order the spec syntax tables prescribe:
//!
//! * `sbr_single_channel_element()` (Table 4.65): one optional
//!   `bs_data_extra` reserved field, then `sbr_grid(0)`, `sbr_dtdf(0)`,
//!   `sbr_invf(0)`, `sbr_envelope(0,0)`, `sbr_noise(0,0)`, the optional
//!   `sbr_sinusoidal_coding(0)`, and the optional extended-data block.
//! * `sbr_channel_pair_element()` (Table 4.66): the two
//!   coupling-dependent layouts. When `bs_coupling` is set, a single
//!   shared grid drives both channels' envelopes / noise (with the
//!   second channel coded in *balance* mode); otherwise each channel
//!   carries its own grid. Either way the parse order is fixed by the
//!   table.
//!
//! `sbr_sinusoidal_coding()` (Table 4.74) reads one
//! `bs_add_harmonic[ch][n]` flag per high-resolution band (`NHigh`).
//!
//! The element-level `bs_amp_res` may be forced to `0` by a
//! single-envelope FIXFIX grid ([`super::sbr_grid::SbrGrid::amp_res_override`]);
//! this wrapper applies that override before decoding the envelopes so
//! the start-value widths and codebook selection match the spec's
//! in-order `bs_amp_res` mutation.
//!
//! The extended-data block (`bs_extended_data` … `sbr_extension`) is
//! recognized and its size is consumed, but the only standardized
//! `sbr_extension` payload (PS, `bs_extension_id == EXTENSION_ID_PS`)
//! is not yet decoded — its bits are skipped as fill so the element
//! parse stays byte-aligned. The raw extension bytes are surfaced for a
//! later PS pass.
//!
//! All of this is fixed-/variable-width syntax driven by the grid and
//! the band tables; the Huffman content lives in [`super::sbr_huffman`].

use alloc::vec::Vec;
use alloc::vec;
use super::sbr_envelope::{SbrEnvelopeData, SbrNoiseData};
use super::sbr_freq_bands::HiLoTables;
use super::sbr_grid::{SbrDtdf, SbrGrid, SbrInvf};
use super::Result;
use crate::audio::aac::bits::BitReader;

/// `bs_extension_id` value that signals a Parametric Stereo payload
/// inside `sbr_extension()` (§4.4.2.8 / Table 4.A.x). PS itself is not
/// decoded here yet.
pub const EXTENSION_ID_PS: u8 = 2;

/// One channel's fully-parsed SBR side info.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SbrChannel {
    /// The time-frequency grid.
    pub grid: SbrGrid,
    /// The delta-direction flags.
    pub dtdf: SbrDtdf,
    /// The inverse-filtering modes (one per noise band).
    pub invf: SbrInvf,
    /// Raw envelope deltas.
    pub envelope: SbrEnvelopeData,
    /// Raw noise-floor deltas.
    pub noise: SbrNoiseData,
    /// `bs_add_harmonic[n]` — one flag per high-resolution band
    /// (`NHigh`); empty when `bs_add_harmonic_flag` was clear.
    pub add_harmonic: Vec<bool>,
}

/// A parsed SBR data element (single channel or channel pair).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SbrElement {
    /// `bs_coupling` (always `false` for a single channel element).
    pub coupling: bool,
    /// One or two channels of side info.
    pub channels: Vec<SbrChannel>,
    /// Raw bytes of an `sbr_extension()` payload, if `bs_extended_data`
    /// was set. Reserved for a later PS decode; `None` when no extended
    /// data was present.
    pub extension: Option<SbrExtension>,
}

/// Raw `sbr_extension()` content carried past this parse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SbrExtension {
    /// `bs_extension_id`.
    pub id: u8,
    /// The extension body bytes (everything after the 2-bit id, up to
    /// the byte-aligned fill).
    pub data: Vec<u8>,
}

/// `sbr_sinusoidal_coding()` (Table 4.74): `NHigh` add-harmonic flags.
fn parse_sinusoidal(reader: &mut BitReader<'_>, n_high: usize) -> Result<Vec<bool>> {
    let mut v = Vec::with_capacity(n_high);
    for _ in 0..n_high {
        v.push(read_flag(reader)?);
    }
    Ok(v)
}

/// Parse one channel's `sbr_grid` → `sbr_dtdf` → `sbr_invf` block (the
/// shared prefix of both element types).
fn parse_grid_dtdf_invf(
    reader: &mut BitReader<'_>,
    n_q: usize,
) -> Result<(SbrGrid, SbrDtdf, SbrInvf)> {
    let grid = SbrGrid::parse(reader)?;
    let dtdf = SbrDtdf::parse(reader, grid.num_env, grid.num_noise)?;
    let invf = SbrInvf::parse(reader, n_q)?;
    Ok((grid, dtdf, invf))
}

impl SbrElement {
    /// Parse `sbr_single_channel_element()` (Table 4.65).
    ///
    /// `bands` is the derived band table for the active header; `n_q` is
    /// its noise-band count. `bs_amp_res` is the header amplitude
    /// resolution (it may be overridden by a single-envelope FIXFIX
    /// grid).
    pub fn parse_single(
        reader: &mut BitReader<'_>,
        bands: &HiLoTables,
        amp_res: bool,
    ) -> Result<Self> {
        let n_q = bands.n_q();
        // bs_data_extra (1 bit) → optional bs_reserved (4).
        if read_flag(reader)? {
            read(reader, 4)?;
        }

        let (grid, dtdf, invf) = parse_grid_dtdf_invf(reader, n_q)?;
        let eff_amp = amp_res && !grid.amp_res_override;

        let envelope = SbrEnvelopeData::parse(reader, &grid, &dtdf, bands, false, false, eff_amp)?;
        let noise = SbrNoiseData::parse(reader, &grid, &dtdf, n_q, false, false, eff_amp)?;

        let add_harmonic = if read_flag(reader)? {
            parse_sinusoidal(reader, bands.n_high())?
        } else {
            Vec::new()
        };

        let extension = parse_extended_data(reader)?;

        Ok(SbrElement {
            coupling: false,
            channels: vec![SbrChannel {
                grid,
                dtdf,
                invf,
                envelope,
                noise,
                add_harmonic,
            }],
            extension,
        })
    }

    /// Parse `sbr_channel_pair_element()` (Table 4.66), both the coupled
    /// and the independent layouts.
    pub fn parse_pair(
        reader: &mut BitReader<'_>,
        bands: &HiLoTables,
        amp_res: bool,
    ) -> Result<Self> {
        let n_q = bands.n_q();
        // bs_data_extra (1 bit) → two bs_reserved (4 each).
        if read_flag(reader)? {
            read(reader, 4)?;
            read(reader, 4)?;
        }

        let coupling = read_flag(reader)?;

        let channels = if coupling {
            // Shared grid; second channel coded in balance mode. Parse
            // order (Table 4.66, coupling): grid(0), dtdf(0), dtdf(1),
            // invf(0).
            let grid = SbrGrid::parse(reader)?;
            let dtdf0 = SbrDtdf::parse(reader, grid.num_env, grid.num_noise)?;
            let dtdf1 = SbrDtdf::parse(reader, grid.num_env, grid.num_noise)?;
            let invf0 = SbrInvf::parse(reader, n_q)?;
            let eff_amp = amp_res && !grid.amp_res_override;

            // Order (Table 4.66, coupling): env0, noise0, env1, noise1.
            let env0 = SbrEnvelopeData::parse(reader, &grid, &dtdf0, bands, true, false, eff_amp)?;
            let noise0 = SbrNoiseData::parse(reader, &grid, &dtdf0, n_q, true, false, eff_amp)?;
            let env1 = SbrEnvelopeData::parse(reader, &grid, &dtdf1, bands, true, true, eff_amp)?;
            let noise1 = SbrNoiseData::parse(reader, &grid, &dtdf1, n_q, true, true, eff_amp)?;

            let (h0, h1) = parse_pair_harmonics(reader, bands)?;
            vec![
                SbrChannel {
                    grid: grid.clone(),
                    dtdf: dtdf0,
                    invf: invf0,
                    envelope: env0,
                    noise: noise0,
                    add_harmonic: h0,
                },
                SbrChannel {
                    grid,
                    dtdf: dtdf1,
                    invf: SbrInvf {
                        invf_mode: Vec::new(),
                    },
                    envelope: env1,
                    noise: noise1,
                    add_harmonic: h1,
                },
            ]
        } else {
            // Independent grids per channel.
            let grid0 = SbrGrid::parse(reader)?;
            let grid1 = SbrGrid::parse(reader)?;
            let dtdf0 = SbrDtdf::parse(reader, grid0.num_env, grid0.num_noise)?;
            let dtdf1 = SbrDtdf::parse(reader, grid1.num_env, grid1.num_noise)?;
            let invf0 = SbrInvf::parse(reader, n_q)?;
            let invf1 = SbrInvf::parse(reader, n_q)?;

            let eff0 = amp_res && !grid0.amp_res_override;
            let eff1 = amp_res && !grid1.amp_res_override;

            // Order (Table 4.66, no coupling): env0, env1, noise0, noise1.
            let env0 = SbrEnvelopeData::parse(reader, &grid0, &dtdf0, bands, false, false, eff0)?;
            let env1 = SbrEnvelopeData::parse(reader, &grid1, &dtdf1, bands, false, true, eff1)?;
            let noise0 = SbrNoiseData::parse(reader, &grid0, &dtdf0, n_q, false, false, eff0)?;
            let noise1 = SbrNoiseData::parse(reader, &grid1, &dtdf1, n_q, false, true, eff1)?;

            let (h0, h1) = parse_pair_harmonics(reader, bands)?;
            vec![
                SbrChannel {
                    grid: grid0,
                    dtdf: dtdf0,
                    invf: invf0,
                    envelope: env0,
                    noise: noise0,
                    add_harmonic: h0,
                },
                SbrChannel {
                    grid: grid1,
                    dtdf: dtdf1,
                    invf: invf1,
                    envelope: env1,
                    noise: noise1,
                    add_harmonic: h1,
                },
            ]
        };

        let extension = parse_extended_data(reader)?;

        Ok(SbrElement {
            coupling,
            channels,
            extension,
        })
    }
}

/// The two `bs_add_harmonic_flag[ch]` blocks of a channel pair
/// (Table 4.66): each optionally followed by an `sbr_sinusoidal_coding`
/// of `NHigh` flags.
fn parse_pair_harmonics(
    reader: &mut BitReader<'_>,
    bands: &HiLoTables,
) -> Result<(Vec<bool>, Vec<bool>)> {
    let h0 = if read_flag(reader)? {
        parse_sinusoidal(reader, bands.n_high())?
    } else {
        Vec::new()
    };
    let h1 = if read_flag(reader)? {
        parse_sinusoidal(reader, bands.n_high())?
    } else {
        Vec::new()
    };
    Ok((h0, h1))
}

/// The shared `if (bs_extended_data) { … }` block (Tables 4.65 / 4.66).
///
/// Reads `bs_extension_size` (4 bits, extended by `bs_esc_count` when
/// `== 15`), then for the duration of the block reads `bs_extension_id`
/// (2 bits) and captures the body as raw bytes. The standardized PS
/// payload is not decoded here; the bytes are returned for a later
/// pass.
fn parse_extended_data(reader: &mut BitReader<'_>) -> Result<Option<SbrExtension>> {
    if !read_flag(reader)? {
        return Ok(None);
    }
    let mut cnt = read(reader, 4)?;
    if cnt == 15 {
        cnt += read(reader, 8)?;
    }
    let mut num_bits_left = (8 * cnt) as i64;
    // The while-loop in the spec reads one bs_extension_id then hands
    // the rest to sbr_extension(); we capture the first id and then
    // *every remaining bit* of the block. The extension payload (e.g.
    // ps_data(), Table 8.A.1) is a bitstream that is NOT byte-aligned
    // within the block — its final sub-byte shares a byte with the
    // bs_fill_bits — so a whole-byte capture would truncate up to 7
    // trailing payload bits. The re-packed buffer is zero-padded to a
    // byte; the payload parser consumes exactly the bits it needs and
    // ignores the rest as fill.
    let mut id = 0u8;
    let mut data: Vec<u8> = Vec::new();
    if num_bits_left > 7 {
        id = read(reader, 2)? as u8;
        num_bits_left -= 2;
        let mut w = super::bitwriter::BitWriter::new();
        while num_bits_left >= 8 {
            w.write_u32(read(reader, 8)?, 8);
            num_bits_left -= 8;
        }
        if num_bits_left > 0 {
            let n = num_bits_left as u32;
            w.write_u32(read(reader, n)?, n);
            num_bits_left = 0;
        }
        data = w.finish();
    }
    // bs_fill_bits: consume the remaining (< 8) bits of a block too
    // short to carry an id.
    if num_bits_left > 0 {
        read(reader, num_bits_left as u32)?;
    }
    Ok(Some(SbrExtension { id, data }))
}

#[inline]
fn read(reader: &mut BitReader<'_>, n: u32) -> Result<u32> {
    reader.read_bits(n).map_err(|_| "aac/sbr: grid invalid")
}

#[inline]
fn read_flag(reader: &mut BitReader<'_>) -> Result<bool> {
    reader.read_bool().map_err(|_| "aac/sbr: grid invalid")
}
