// Ported from oxideav-aac (MIT) — Copyright (c) 2026 Karpelès Lab Inc.
// See THIRDPARTY-LICENSES.md.

//! `sbr_header()` parser — ISO/IEC 14496-3 §4.4.2.8, Table 4.63.
//!
//! The SBR header carries the static per-stream parameters that drive
//! the §4.6.18.3.2 frequency-band setup ([`super::sbr_freq_bands`]):
//! the start / stop frequency indices, the crossover band, and the two
//! optional "extra" header blocks (`bs_header_extra_1` /
//! `bs_header_extra_2`). Per Table 4.63 Note 3, when an extra-header
//! flag is clear the underlying elements take their **default** values,
//! disregarding any previously transmitted value:
//!
//! | element | width | default (Tables 4.105–4.111) |
//! |---------|-------|------------------------------|
//! | `bs_freq_scale`    | 2 | 2 (10 bands/octave) |
//! | `bs_alter_scale`   | 1 | 1 (grouping / extra-wide) |
//! | `bs_noise_bands`   | 2 | 2 (2 bands/octave) |
//! | `bs_limiter_bands` | 2 | 2 (2.0 bands/octave) |
//! | `bs_limiter_gains` | 2 | 2 (3 dB max gain) |
//! | `bs_interpol_freq` | 1 | 1 (interpolation on) |
//! | `bs_smoothing_mode`| 1 | 1 (smoothing off) |
//!
//! `bs_amp_res` (the envelope amplitude resolution: 0 = 1.5 dB,
//! 1 = 3.0 dB) is carried in the header but may be overridden to 0 by
//! `sbr_grid()` for a single-envelope `FIXFIX` frame — that override is
//! applied downstream, not here.
//!
//! The header is a fixed-width bit layout with no Huffman or
//! variable-length content, so it is fully recoverable from the spec
//! syntax table alone.

use super::Result;
use crate::audio::aac::bits::BitReader;

/// Default `bs_freq_scale` when `bs_header_extra_1 == 0` (Table 4.105:
/// 10 bands/octave).
pub const DEFAULT_FREQ_SCALE: u8 = 2;
/// Default `bs_alter_scale` when `bs_header_extra_1 == 0` (Table 4.106).
pub const DEFAULT_ALTER_SCALE: bool = true;
/// Default `bs_noise_bands` when `bs_header_extra_1 == 0` (Table 4.107:
/// 2 bands/octave).
pub const DEFAULT_NOISE_BANDS: u8 = 2;
/// Default `bs_limiter_bands` when `bs_header_extra_2 == 0` (Table
/// 4.108: 2.0 bands/octave).
pub const DEFAULT_LIMITER_BANDS: u8 = 2;
/// Default `bs_limiter_gains` when `bs_header_extra_2 == 0` (Table
/// 4.109: 3 dB max gain).
pub const DEFAULT_LIMITER_GAINS: u8 = 2;
/// Default `bs_interpol_freq` when `bs_header_extra_2 == 0` (Table
/// 4.110: interpolation on).
pub const DEFAULT_INTERPOL_FREQ: bool = true;
/// Default `bs_smoothing_mode` when `bs_header_extra_2 == 0` (Table
/// 4.111: smoothing off).
pub const DEFAULT_SMOOTHING_MODE: bool = true;

/// A parsed `sbr_header()` (Table 4.63).
///
/// The two `bs_header_extra_*` flags are recorded so a re-encoder can
/// reproduce the exact bit layout, but the underlying parameters are
/// already resolved to their effective values (the Table 4.63 Note 3
/// defaults are filled in when an extra flag is clear).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SbrHeader {
    /// `bs_amp_res` — envelope amplitude resolution (false = 1.5 dB,
    /// true = 3.0 dB). May be forced to false by a single-envelope
    /// `FIXFIX` grid downstream.
    pub amp_res: bool,
    /// `bs_start_freq` — 4-bit index into the §4.6.18.3.2.1 `offset`
    /// table that sets the low QMF boundary `k0`.
    pub start_freq: u8,
    /// `bs_stop_freq` — 4-bit index that sets the high QMF boundary
    /// `k2`.
    pub stop_freq: u8,
    /// `bs_xover_band` — 3-bit index into the master frequency table
    /// where the SBR range begins (`bs_xover_band < NMaster`).
    pub xover_band: u8,
    /// `bs_reserved` — 2 reserved bits (kept for faithful re-encode).
    pub reserved: u8,
    /// `bs_header_extra_1` — whether the optional header part 1 was
    /// transmitted.
    pub header_extra_1: bool,
    /// `bs_header_extra_2` — whether the optional header part 2 was
    /// transmitted.
    pub header_extra_2: bool,
    /// `bs_freq_scale` — master-table warping selector (Table 4.105).
    pub freq_scale: u8,
    /// `bs_alter_scale` — master-table alteration flag (Table 4.106).
    pub alter_scale: bool,
    /// `bs_noise_bands` — noise-band density selector (Table 4.107).
    pub noise_bands: u8,
    /// `bs_limiter_bands` — limiter-band density selector (Table 4.108).
    pub limiter_bands: u8,
    /// `bs_limiter_gains` — limiter max-gain selector (Table 4.109).
    pub limiter_gains: u8,
    /// `bs_interpol_freq` — frequency-interpolation flag (Table 4.110).
    pub interpol_freq: bool,
    /// `bs_smoothing_mode` — smoothing flag (Table 4.111).
    pub smoothing_mode: bool,
}

impl SbrHeader {
    /// Parse `sbr_header()` (Table 4.63) from `reader`, filling in the
    /// Table 4.63 Note 3 default values for any extra-header block that
    /// is not present.
    ///
    /// The `bs_reserved` field is read but not validated (Table 4.63
    /// leaves its value unconstrained). Returns [`"aac/sbr: huffman invalid"`]
    /// only if `reader` runs out of bits — there is no Huffman content
    /// here, so this maps the bit-exhaustion error onto the SBR error
    /// surface.
    pub fn parse(reader: &mut BitReader<'_>) -> Result<Self> {
        let amp_res = read_bit(reader)?;
        let start_freq = read_u8(reader, 4)?;
        let stop_freq = read_u8(reader, 4)?;
        let xover_band = read_u8(reader, 3)?;
        let reserved = read_u8(reader, 2)?;
        let header_extra_1 = read_bit(reader)?;
        let header_extra_2 = read_bit(reader)?;

        let (freq_scale, alter_scale, noise_bands) = if header_extra_1 {
            (read_u8(reader, 2)?, read_bit(reader)?, read_u8(reader, 2)?)
        } else {
            (DEFAULT_FREQ_SCALE, DEFAULT_ALTER_SCALE, DEFAULT_NOISE_BANDS)
        };

        let (limiter_bands, limiter_gains, interpol_freq, smoothing_mode) = if header_extra_2 {
            (
                read_u8(reader, 2)?,
                read_u8(reader, 2)?,
                read_bit(reader)?,
                read_bit(reader)?,
            )
        } else {
            (
                DEFAULT_LIMITER_BANDS,
                DEFAULT_LIMITER_GAINS,
                DEFAULT_INTERPOL_FREQ,
                DEFAULT_SMOOTHING_MODE,
            )
        };

        Ok(SbrHeader {
            amp_res,
            start_freq,
            stop_freq,
            xover_band,
            reserved,
            header_extra_1,
            header_extra_2,
            freq_scale,
            alter_scale,
            noise_bands,
            limiter_bands,
            limiter_gains,
            interpol_freq,
            smoothing_mode,
        })
    }

    /// Whether this header differs from `other` in any field that
    /// affects the §4.6.18.3.2 frequency-band geometry
    /// (`bs_start_freq`, `bs_stop_freq`, `bs_xover_band`,
    /// `bs_freq_scale`, `bs_alter_scale`, `bs_noise_bands`).
    ///
    /// SBR decoders only need to recompute the master / derived band
    /// tables when one of these "reset" parameters changes; the limiter
    /// / interpolation / smoothing parameters do not alter the band
    /// geometry. This mirrors the §4.6.18.3.3 header-change reset.
    pub fn band_geometry_changed(&self, other: &SbrHeader) -> bool {
        self.start_freq != other.start_freq
            || self.stop_freq != other.stop_freq
            || self.xover_band != other.xover_band
            || self.freq_scale != other.freq_scale
            || self.alter_scale != other.alter_scale
            || self.noise_bands != other.noise_bands
    }

    /// Compute the §4.6.18.3.2 derived frequency-band tables
    /// ([`super::sbr_freq_bands::HiLoTables`]) for this header at the
    /// given SBR internal sample rate `fs_sbr` (twice the AAC core
    /// rate).
    ///
    /// This chains [`super::sbr_freq_bands::k0`] /
    /// [`super::sbr_freq_bands::k2`] /
    /// [`super::sbr_freq_bands::master_table`] /
    /// [`super::sbr_freq_bands::HiLoTables::derive`] with this header's
    /// parameters; it returns [`"aac/sbr: freq band invalid"`] for any
    /// §4.6.18.3.6-violating geometry.
    pub fn derive_bands(&self, fs_sbr: u32) -> Result<super::sbr_freq_bands::HiLoTables> {
        let k0 = super::sbr_freq_bands::k0(fs_sbr, self.start_freq)?;
        let k2 = super::sbr_freq_bands::k2(fs_sbr, self.stop_freq, k0)?;
        let f_master =
            super::sbr_freq_bands::master_table(k0, k2, self.freq_scale, self.alter_scale)?;
        super::sbr_freq_bands::HiLoTables::derive(&f_master, self.xover_band, self.noise_bands)
    }
}

#[inline]
fn read_bit(reader: &mut BitReader<'_>) -> Result<bool> {
    reader.read_bool().map_err(|_| "aac/sbr: huffman invalid")
}

#[inline]
fn read_u8(reader: &mut BitReader<'_>, n: u32) -> Result<u8> {
    reader
        .read_bits(n)
        .map(|v| v as u8)
        .map_err(|_| "aac/sbr: huffman invalid")
}
