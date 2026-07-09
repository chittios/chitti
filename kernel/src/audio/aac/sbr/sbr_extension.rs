// Ported from oxideav-aac (MIT) — Copyright (c) 2026 Karpelès Lab Inc.
// See THIRDPARTY-LICENSES.md.

//! `sbr_extension_data()` top-level walker — ISO/IEC 14496-3 §4.4.2.8
//! Table 4.62.
//!
//! This is the glue between [`crate::extension_payload`] and the SBR
//! side-info parsers: it consumes a whole SBR extension payload from a
//! `fill_element()`'s `extension_payload()` body, in the exact spec
//! order:
//!
//! ```text
//! sbr_extension_data(id_aac, crc_flag) {
//!     num_sbr_bits = 0;
//!     if (crc_flag) { bs_sbr_crc_bits;            10  uimsbf  num_sbr_bits += 10; }
//!     // sbr_layer != SBR_STEREO_ENHANCE for a non-scalable core:
//!     bs_header_flag;                              1   uimsbf  num_sbr_bits += 1;
//!     if (bs_header_flag)  num_sbr_bits += sbr_header();
//!     num_sbr_bits += sbr_data(id_aac, bs_amp_res);
//!     num_align_bits = (8*cnt - 4 - num_sbr_bits) % 8;
//!     bs_fill_bits;                                num_align_bits  uimsbf
//! }
//! ```
//!
//! `sbr_data(id_aac, bs_amp_res)` dispatches on the AAC element type the
//! SBR payload extends: an `ID_SCE` core element pairs with
//! `sbr_single_channel_element()` ([`SbrElement::parse_single`]), an
//! `ID_CPE` core element with `sbr_channel_pair_element()`
//! ([`SbrElement::parse_pair`]). The band tables both need are derived
//! from the active [`SbrHeader`] at the SBR *internal* sample rate
//! `fs_sbr` (twice the AAC core rate) via [`SbrHeader::derive_bands`].
//!
//! ## Header reuse
//!
//! When `bs_header_flag == 0` the payload reuses the most recent
//! transmitted `sbr_header()`. The first SBR payload of a stream must
//! carry a header (`bs_header_flag == 1`); a clear flag with no prior
//! header is an ill-formed stream ([`"aac/sbr: freq band invalid"`]). The
//! caller threads the returned [`SbrExtensionData::header`] back in as
//! `prev_header` on the next payload so the reuse chain is continuous.
//!
//! ## Scope
//!
//! This decodes the SBR *bitstream* side info end to end (CRC field +
//! header + grid / dtdf / invf / envelope / noise / add-harmonic +
//! extended-data block). The SBR back-end DSP (dequantization to linear
//! energies, the QMF analysis / synthesis filterbanks, HF generation /
//! patching, the limiter, and the envelope adjustment that produces
//! up-sampled PCM) is **not** part of this walker — it keys off the
//! band tables and scalefactors this produces. The `bs_sbr_crc_bits`
//! value is captured but not verified (the §4.4.2.8 SBR CRC region is a
//! later step).
//!
//! ## Clean-room provenance
//!
//! The Table 4.62 syntax, the `num_align_bits = (8·cnt − 4 −
//! num_sbr_bits) % 8` fill computation, and the `sbr_data` dispatch on
//! `id_aac` are transcribed from ISO/IEC 14496-3:2009 §4.4.2.8 staged
//! under `docs/audio/aac/`. The non-scalable core fixes the helper
//! `sbr_layer` to `SBR_NOT_SCALABLE` (Table 4.62 Note 1), so the
//! `bs_header_flag` is always present.

use crate::audio::aac::bits::BitReader;

use super::IdSynEle;
use super::sbr_element::SbrElement;
use super::sbr_header::SbrHeader;
use super::Result;

/// Field width of `bs_sbr_crc_bits` (Table 4.62).
pub const SBR_CRC_BITS: u32 = 10;

/// A fully-parsed `sbr_extension_data()` payload (Table 4.62).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SbrExtensionData {
    /// `bs_sbr_crc_bits` (10-bit) when `crc_flag` was set (the
    /// `EXT_SBR_DATA_CRC` extension type); `None` for the plain
    /// `EXT_SBR_DATA` type. Captured but not yet verified.
    pub crc: Option<u16>,
    /// `bs_header_flag` — whether this payload transmitted a fresh
    /// `sbr_header()`.
    pub header_present: bool,
    /// The active SBR header for this payload: the freshly parsed one
    /// when `header_present`, otherwise the reused `prev_header`. The
    /// caller threads this forward as the next payload's `prev_header`.
    pub header: SbrHeader,
    /// The decoded SBR data element (single channel or channel pair),
    /// dispatched on the core element's `id_aac`.
    pub element: SbrElement,
    /// The number of SBR side-info bits consumed before the trailing
    /// `bs_fill_bits` (the spec's `num_sbr_bits`). Useful for callers
    /// validating against the `extension_payload()` byte count.
    pub num_sbr_bits: u64,
}

impl SbrExtensionData {
    /// Parse an `sbr_extension_data(id_aac, crc_flag)` payload (Table
    /// 4.62) from `reader`, positioned at the first SBR bit (i.e. the
    /// caller — [`crate::extension_payload`] — has already consumed the
    /// 4-bit `extension_type`).
    ///
    /// * `id_aac` — the AAC core element this SBR payload extends: only
    ///   [`IdSynEle::Sce`] / [`IdSynEle::Cpe`] are valid (an SBR payload
    ///   only attaches to a channel element). Any other id is rejected
    ///   with [`"aac/sbr: freq band invalid"`].
    /// * `crc_flag` — `true` for the `EXT_SBR_DATA_CRC` extension type
    ///   (a 10-bit `bs_sbr_crc_bits` field precedes the header), `false`
    ///   for plain `EXT_SBR_DATA`.
    /// * `fs_sbr` — the SBR *internal* sample rate (twice the AAC core
    ///   `samplingFrequencyIndex` rate). Drives [`SbrHeader::derive_bands`].
    /// * `cnt` — the `extension_payload()` byte count `cnt` (Table 4.51),
    ///   used to size the trailing `bs_fill_bits` alignment. Pass `None`
    ///   to skip the fill consumption (when the caller bounds the reader
    ///   itself); the fill is then left in the reader.
    /// * `prev_header` — the most recent transmitted header for the reuse
    ///   path; `None` on the stream's first SBR payload. A clear
    ///   `bs_header_flag` with `prev_header == None` is ill-formed.
    pub fn parse(
        reader: &mut BitReader<'_>,
        id_aac: IdSynEle,
        crc_flag: bool,
        fs_sbr: u32,
        cnt: Option<u32>,
        prev_header: Option<SbrHeader>,
    ) -> Result<Self> {
        let start = reader.bit_position();

        let crc = if crc_flag {
            Some(read(reader, SBR_CRC_BITS)? as u16)
        } else {
            None
        };

        // Non-scalable core ⇒ sbr_layer == SBR_NOT_SCALABLE, so the
        // bs_header_flag is always present (Table 4.62 Note 1).
        let header_present = read_flag(reader)?;
        let header = if header_present {
            SbrHeader::parse(reader)?
        } else {
            // Reuse the previous transmitted header; a stream that opens
            // with a header-less SBR payload is ill-formed.
            prev_header.ok_or("aac/sbr: freq band invalid")?
        };

        // sbr_data(id_aac, bs_amp_res): the band tables are derived from
        // the active header at the SBR internal rate; the element type is
        // selected by the core element id_aac.
        let bands = header.derive_bands(fs_sbr)?;
        let element = match id_aac {
            IdSynEle::Sce => SbrElement::parse_single(reader, &bands, header.amp_res)?,
            IdSynEle::Cpe => SbrElement::parse_pair(reader, &bands, header.amp_res)?,
            _ => return Err("aac/sbr: freq band invalid"),
        };

        let num_sbr_bits = reader.bit_position() - start;

        // num_align_bits = (8*cnt - 4 - num_sbr_bits) % 8. The `- 4`
        // accounts for the extension_type nibble the caller already read;
        // when cnt is known, consume the trailing bs_fill_bits so the
        // reader lands on the next extension_payload element.
        if let Some(cnt) = cnt {
            let total = u64::from(cnt) * 8;
            let consumed = num_sbr_bits + 4; // + the extension_type nibble
            if total < consumed {
                return Err("aac/sbr: freq band invalid");
            }
            let align = (total - consumed) % 8;
            if align > 0 {
                read(reader, align as u32)?;
            }
        }

        Ok(SbrExtensionData {
            crc,
            header_present,
            header,
            element,
            num_sbr_bits,
        })
    }
}

#[inline]
fn read(reader: &mut BitReader<'_>, n: u32) -> Result<u32> {
    reader.read_bits(n).map_err(|_| "aac/sbr: freq band invalid")
}

#[inline]
fn read_flag(reader: &mut BitReader<'_>) -> Result<bool> {
    reader.read_bool().map_err(|_| "aac/sbr: freq band invalid")
}
