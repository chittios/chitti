//! H.265 / HEVC parameter-set, NAL-unit and slice-header parsing (ITU-T H.265
//! §7.3), the first decode stage for HEVC exactly as [`super::h264`] is for AVC.
//!
//! HEVC reuses AVC's RBSP escaping and Exp-Golomb codes, so [`super::bits`] is
//! shared verbatim. What differs — and what this module exists to get right —
//! is everything above the bit layer:
//!
//! * **The NAL header is two bytes, not one** (`nal_unit_type` is 6 bits at bit
//!   1, then `nuh_layer_id` 6 and `nuh_temporal_id_plus1` 3). Reading an HEVC
//!   NAL with AVC's `hdr & 0x1f` yields a plausible wrong type for every unit,
//!   so the split functions here are separate rather than parameterised.
//! * **Three parameter sets** (VPS/SPS/PPS) and the `profile_tier_level`
//!   structure they share, whose length depends on the sub-layer count — get it
//!   wrong by one bit and every field after it in the SPS is garbage that still
//!   parses.
//! * **Short-term reference picture sets** live in the SPS *and* the slice
//!   header, and an inter-predicted set is expressed as a delta from an earlier
//!   one, so they must be kept (not skipped) to keep the bit position right.
//!
//! Pure + panic-free: malformed input returns `Err`, never crashes.

pub mod cabac_tables;
pub mod tables;
pub mod ctu;
pub mod decoder;
pub mod deblock;
pub mod dpb;
pub mod inter;
pub mod intra;
pub mod residual;
pub mod sao;
pub mod syntax;
#[cfg(test)]
pub mod testutil;
pub mod transform;

use super::bits::{unescape_rbsp, BitReader};
use alloc::vec::Vec;

/// Maximum sub-layers an HEVC stream may declare (`sps_max_sub_layers_minus1`
/// is 3 bits, so 7+1). Bounds every sub-layer loop.
const MAX_SUB_LAYERS: usize = 8;

/// Spec cap on short-term RPS count (`num_short_term_ref_pic_sets` ≤ 64).
const MAX_ST_RPS: usize = 64;

/// Spec cap on pictures in one reference picture set (`sps_max_dec_pic_buffering`
/// ≤ 16, so a set can name at most 16 of each polarity).
const MAX_DPB: usize = 16;

/// HEVC NAL unit types (H.265 Table 7-1). VCL types are 0..=31; everything the
/// decoder acts on is named, the rest is `Other`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NalType {
    /// 0..=9, 10..=15 — a coded slice segment of a non-IRAP picture.
    Slice(u8),
    /// 16..=23 — a coded slice segment of an IRAP picture (BLA/IDR/CRA).
    SliceIrap(u8),
    Vps,       // 32
    Sps,       // 33
    Pps,       // 34
    AuDelim,   // 35
    EndOfSeq,  // 36
    EndOfBits, // 37
    Filler,    // 38
    Sei(u8),   // 39 (prefix) / 40 (suffix)
    Other(u8),
}

impl NalType {
    fn from_code(t: u8) -> NalType {
        match t {
            0..=15 => NalType::Slice(t),
            16..=23 => NalType::SliceIrap(t),
            32 => NalType::Vps,
            33 => NalType::Sps,
            34 => NalType::Pps,
            35 => NalType::AuDelim,
            36 => NalType::EndOfSeq,
            37 => NalType::EndOfBits,
            38 => NalType::Filler,
            39 | 40 => NalType::Sei(t),
            other => NalType::Other(other),
        }
    }

    /// True for a VCL NAL — one that carries a coded slice segment.
    pub fn is_slice(self) -> bool {
        matches!(self, NalType::Slice(_) | NalType::SliceIrap(_))
    }

    /// True for an IRAP picture (BLA 16..18, IDR 19..20, CRA 21, reserved
    /// 22..23) — a random-access point the decoder may start at.
    pub fn is_irap(self) -> bool {
        matches!(self, NalType::SliceIrap(_))
    }

    /// True for an IDR (19 `IDR_W_RADL`, 20 `IDR_N_LP`) — an IRAP that also
    /// resets POC, so the DPB may be emptied.
    pub fn is_idr(self) -> bool {
        matches!(self, NalType::SliceIrap(19) | NalType::SliceIrap(20))
    }

    /// True for BLA (16..=18) — a broken link access point; like an IDR for
    /// decoding purposes but POC is *not* reset by definition.
    pub fn is_bla(self) -> bool {
        matches!(self, NalType::SliceIrap(16..=18))
    }

    /// The raw 6-bit `nal_unit_type`.
    pub fn code(self) -> u8 {
        match self {
            NalType::Slice(t) | NalType::SliceIrap(t) | NalType::Sei(t) | NalType::Other(t) => t,
            NalType::Vps => 32,
            NalType::Sps => 33,
            NalType::Pps => 34,
            NalType::AuDelim => 35,
            NalType::EndOfSeq => 36,
            NalType::EndOfBits => 37,
            NalType::Filler => 38,
        }
    }

    /// True for a sub-layer **non-reference** picture (`*_N`: the even VCL types
    /// below 16). Nothing in the same sub-layer predicts from one, so the player
    /// may drop it wholesale — the HEVC analogue of AVC's `nal_ref_idc == 0`.
    pub fn is_sublayer_nonref(self) -> bool {
        match self {
            NalType::Slice(t) => t <= 14 && t % 2 == 0,
            _ => false,
        }
    }
}

/// One HEVC NAL unit: its type, layer/temporal ids, and the payload *including*
/// the 2-byte header (so the RBSP is `unescape_rbsp(&payload[2..])`).
pub struct Nal<'a> {
    pub kind: NalType,
    pub layer_id: u8,
    /// `nuh_temporal_id_plus1 - 1`.
    pub temporal_id: u8,
    pub payload: &'a [u8],
}

impl<'a> Nal<'a> {
    /// The RBSP (emulation-prevention bytes removed) of this NAL's body.
    pub fn rbsp(&self) -> Vec<u8> {
        unescape_rbsp(&self.payload[2..])
    }
}

fn push_nal<'a>(nals: &mut Vec<Nal<'a>>, payload: &'a [u8]) {
    if payload.len() < 2 {
        return;
    }
    // forbidden_zero_bit must be 0; if set the unit is corrupt.
    if payload[0] & 0x80 != 0 {
        return;
    }
    let ty = (payload[0] >> 1) & 0x3f;
    let layer_id = ((payload[0] & 1) << 5) | (payload[1] >> 3);
    let tid_plus1 = payload[1] & 0x07;
    if tid_plus1 == 0 {
        return; // nuh_temporal_id_plus1 == 0 is forbidden
    }
    nals.push(Nal {
        kind: NalType::from_code(ty),
        layer_id,
        temporal_id: tid_plus1 - 1,
        payload,
    });
}

/// Split an **Annex-B** HEVC byte stream on `00 00 01` / `00 00 00 01`.
pub fn split_annexb(data: &[u8]) -> Vec<Nal<'_>> {
    let mut nals = Vec::new();
    let mut start = find_start_code(data, 0);
    while let Some((sc_pos, sc_len)) = start {
        let unit_start = sc_pos + sc_len;
        let next = find_start_code(data, unit_start);
        let unit_end = next.map(|(p, _)| p).unwrap_or(data.len());
        if unit_end > unit_start {
            push_nal(&mut nals, &data[unit_start..unit_end]);
        }
        start = next;
    }
    nals
}

fn find_start_code(data: &[u8], from: usize) -> Option<(usize, usize)> {
    let mut i = from;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            if i > 0 && data[i - 1] == 0 {
                return Some((i - 1, 4));
            }
            return Some((i, 3));
        }
        i += 1;
    }
    None
}

/// Split **HVCC** (ISO-BMFF) sample data: each NAL is prefixed by a big-endian
/// length of `length_size` bytes, exactly as AVCC — only the header inside
/// differs.
pub fn split_hvcc(data: &[u8], length_size: u8) -> Vec<Nal<'_>> {
    let ls = length_size.clamp(1, 4) as usize;
    let mut nals = Vec::new();
    let mut i = 0;
    while i + ls <= data.len() {
        let mut len = 0usize;
        for _ in 0..ls {
            len = (len << 8) | data[i] as usize;
            i += 1;
        }
        if len == 0 || i + len > data.len() {
            break;
        }
        push_nal(&mut nals, &data[i..i + len]);
        i += len;
    }
    nals
}

/// True iff `data` (one HVCC-framed access unit) contains at least one slice and
/// **every** slice NAL is a sub-layer non-reference picture — the HEVC form of
/// the frame-drop test in [`super`]. Conservative `false` on bad framing.
pub fn sample_is_nonref(data: &[u8], length_size: usize) -> bool {
    if length_size == 0 || length_size > 4 {
        return false;
    }
    let mut off = 0usize;
    let mut saw_slice = false;
    while off + length_size < data.len() {
        let mut len = 0usize;
        for &b in &data[off..off + length_size] {
            len = (len << 8) | b as usize;
        }
        off += length_size;
        if len == 0 || off + len > data.len() || off + 2 > data.len() {
            return false;
        }
        let kind = NalType::from_code((data[off] >> 1) & 0x3f);
        if kind.is_slice() {
            saw_slice = true;
            if !kind.is_sublayer_nonref() {
                return false;
            }
        }
        off += len;
    }
    saw_slice
}

// ---------------------------------------------------------------------------
// profile_tier_level (§7.3.3)
// ---------------------------------------------------------------------------

/// The parts of `profile_tier_level` anything downstream reads. The rest is
/// consumed (never skipped by a constant) so the bit position stays exact.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProfileTierLevel {
    pub profile_space: u8,
    /// false = Main tier, true = High tier.
    pub tier_high: bool,
    /// 1 Main, 2 Main 10, 3 Main Still Picture, 4 range extensions (Rext) …
    pub profile_idc: u8,
    /// 30 × level, i.e. 120 = level 4.0, 153 = level 5.1.
    pub level_idc: u8,
    /// `general_profile_compatibility_flag[32]` as a bitmask (bit *i* = flag *i*).
    pub compat_flags: u32,
}

impl ProfileTierLevel {
    /// Human-readable profile name for the probe line.
    pub fn profile_name(&self) -> &'static str {
        // A stream may set profile_idc to one value and additionally flag
        // compatibility with a lower one; the idc is what identifies it.
        match self.profile_idc {
            1 => "Main",
            2 => "Main 10",
            3 => "Main Still Picture",
            4 => "Range Extensions",
            5 => "High Throughput",
            9 => "Screen Content Coding",
            _ => "unknown",
        }
    }
}

/// Parse `profile_tier_level(profilePresentFlag, maxNumSubLayersMinus1)`.
///
/// The bit budget is the trap: the profile block is **88 bits** (2 + 1 + 5 + 32
/// compatibility flags + 4 constraint flags + 43 reserved + 1 `inbld`), then
/// `general_level_idc` u(8). Each present sub-layer repeats the same 88/8 split,
/// and when `maxNumSubLayersMinus1 > 0` the *unused* sub-layer slots up to 8 are
/// padded with 2 reserved bits each — omit that padding and the SPS fields after
/// this point decode into plausible nonsense rather than an error.
pub fn parse_profile_tier_level(
    r: &mut BitReader,
    profile_present: bool,
    max_sub_layers_minus1: u32,
) -> Result<ProfileTierLevel, &'static str> {
    let mut ptl = ProfileTierLevel::default();
    if max_sub_layers_minus1 as usize >= MAX_SUB_LAYERS {
        return Err("hevc ptl: too many sub-layers");
    }
    if profile_present {
        ptl.profile_space = r.u(2)? as u8;
        ptl.tier_high = r.flag()?;
        ptl.profile_idc = r.u(5)? as u8;
        ptl.compat_flags = r.u(32)?;
        // progressive/interlaced/non_packed/frame_only constraint flags, then
        // 43 reserved bits and the inbld flag: 48 bits total.
        let _constraints_hi = r.u(32)?;
        let _constraints_lo = r.u(16)?;
    }
    ptl.level_idc = r.u(8)? as u8;
    let n = max_sub_layers_minus1 as usize;
    let mut sub_profile = [false; MAX_SUB_LAYERS];
    let mut sub_level = [false; MAX_SUB_LAYERS];
    for i in 0..n {
        sub_profile[i] = r.flag()?;
        sub_level[i] = r.flag()?;
    }
    if n > 0 {
        for _ in n..8 {
            let _reserved_zero_2bits = r.u(2)?;
        }
    }
    for i in 0..n {
        if sub_profile[i] {
            let _ = r.u(32)?; // space/tier/idc + first 24 compat flags
            let _ = r.u(32)?;
            let _ = r.u(24)?; // remainder of the 88-bit profile block
        }
        if sub_level[i] {
            let _sub_layer_level_idc = r.u(8)?;
        }
    }
    Ok(ptl)
}

// ---------------------------------------------------------------------------
// scaling_list_data (§7.3.4)
// ---------------------------------------------------------------------------

/// HEVC scaling lists, in the decoder's working form: `[sizeId][matrixId]` of
/// up to 64 coefficients, plus the DC coefficient for the 16x16/32x32 sizes.
///
/// Default lists are *not* flat — H.265 Tables 7-5/7-6 give specific 4x4 (flat
/// 16) and 8x8+ (a real quantisation matrix) defaults, and a stream that enables
/// scaling lists without transmitting them expects those. A flat default would
/// decode every such stream slightly wrong everywhere, which reads as softness
/// rather than as a bug.
#[derive(Clone, Debug)]
pub struct ScalingLists {
    /// `[sizeId 0..4][matrixId 0..6][coef]`.
    pub lists: [[[u8; 64]; 6]; 4],
    /// `[sizeId 2..4 → 0..2][matrixId]` DC coefficient.
    pub dc: [[u8; 6]; 2],
}

/// H.265 Table 7-5: the default 4x4 list is flat 16.
const DEFAULT_4X4: [u8; 16] = [16; 16];

/// H.265 Table 7-6, intra (matrixId 0..2) — in the up-right diagonal scan order
/// the syntax uses, so it is applied to `lists` verbatim.
const DEFAULT_8X8_INTRA: [u8; 64] = [
    16, 16, 16, 16, 17, 18, 21, 24, 16, 16, 16, 16, 17, 19, 22, 25, 16, 16, 17, 18, 20, 22, 25, 29,
    16, 16, 18, 21, 24, 27, 31, 36, 17, 17, 20, 24, 30, 35, 41, 47, 18, 19, 22, 27, 35, 44, 54, 65,
    21, 22, 25, 31, 41, 54, 70, 88, 24, 25, 29, 36, 47, 65, 88, 115,
];

/// H.265 Table 7-6, inter (matrixId 3..5).
const DEFAULT_8X8_INTER: [u8; 64] = [
    16, 16, 16, 16, 17, 18, 20, 24, 16, 16, 16, 17, 18, 20, 24, 25, 16, 16, 17, 18, 20, 24, 25, 28,
    16, 17, 18, 20, 24, 25, 28, 33, 17, 18, 20, 24, 25, 28, 33, 41, 18, 20, 24, 25, 28, 33, 41, 54,
    20, 24, 25, 28, 33, 41, 54, 71, 24, 25, 28, 33, 41, 54, 71, 91,
];

impl Default for ScalingLists {
    fn default() -> Self {
        let mut s = ScalingLists { lists: [[[16u8; 64]; 6]; 4], dc: [[16u8; 6]; 2] };
        for m in 0..6 {
            s.lists[0][m][..16].copy_from_slice(&DEFAULT_4X4);
            let d = if m < 3 { &DEFAULT_8X8_INTRA } else { &DEFAULT_8X8_INTER };
            for size in 1..4 {
                s.lists[size][m] = *d;
            }
        }
        for size in 0..2 {
            for m in 0..6 {
                s.dc[size][m] = 16;
            }
        }
        s
    }
}

impl ScalingLists {
    fn set_default(&mut self, size_id: usize, matrix_id: usize) {
        if size_id == 0 {
            self.lists[0][matrix_id][..16].copy_from_slice(&DEFAULT_4X4);
        } else {
            let d = if matrix_id < 3 { &DEFAULT_8X8_INTRA } else { &DEFAULT_8X8_INTER };
            self.lists[size_id][matrix_id] = *d;
        }
        if size_id >= 2 {
            self.dc[size_id - 2][matrix_id] = 16;
        }
    }
}

/// Parse `scaling_list_data()` (§7.3.4). Unlike the AVC path this **keeps** the
/// values: HEVC streams that enable scaling lists are common (any `--aq`-tuned
/// x265 output can carry them) and dequantisation needs them.
pub fn parse_scaling_list_data(r: &mut BitReader) -> Result<ScalingLists, &'static str> {
    let mut sl = ScalingLists::default();
    for size_id in 0..4usize {
        // sizeId 3 (32x32) has only matrixId 0 and 3 — the loop steps by 3.
        let step = if size_id == 3 { 3 } else { 1 };
        let mut matrix_id = 0usize;
        while matrix_id < 6 {
            let pred_mode = r.flag()?;
            if !pred_mode {
                // Copy from an earlier matrix of the same size, or the default
                // when the delta is 0.
                let delta = r.ue()? as usize;
                if delta == 0 {
                    sl.set_default(size_id, matrix_id);
                } else {
                    let ref_id = matrix_id.checked_sub(delta * step).ok_or("hevc scaling: bad pred delta")?;
                    sl.lists[size_id][matrix_id] = sl.lists[size_id][ref_id];
                    if size_id >= 2 {
                        sl.dc[size_id - 2][matrix_id] = sl.dc[size_id - 2][ref_id];
                    }
                }
            } else {
                let coef_num = core::cmp::min(64usize, 1 << (4 + (size_id << 1)));
                let mut next = 8i32;
                if size_id > 1 {
                    let dc = r.se()? + 8;
                    if !(0..=255).contains(&dc) {
                        return Err("hevc scaling: dc out of range");
                    }
                    sl.dc[size_id - 2][matrix_id] = dc as u8;
                    next = dc;
                }
                for i in 0..coef_num {
                    let delta = r.se()?;
                    next = (next + delta + 256).rem_euclid(256);
                    sl.lists[size_id][matrix_id][i] = next as u8;
                }
            }
            matrix_id += step;
        }
    }
    Ok(sl)
}

// ---------------------------------------------------------------------------
// short-term reference picture sets (§7.3.7)
// ---------------------------------------------------------------------------

/// One short-term reference picture set, resolved to absolute delta-POCs.
///
/// The syntax stores negative (before, `s0`) and positive (after, `s1`) sets as
/// running deltas, and an *inter-predicted* set as a delta from an earlier set;
/// both forms are resolved here so the reference-list builder never re-derives
/// them.
#[derive(Clone, Debug, Default)]
pub struct ShortTermRps {
    /// Delta POCs of pictures before the current one, nearest first (negative).
    pub delta_poc_s0: Vec<i32>,
    pub used_s0: Vec<bool>,
    /// Delta POCs of pictures after the current one, nearest first (positive).
    pub delta_poc_s1: Vec<i32>,
    pub used_s1: Vec<bool>,
}

impl ShortTermRps {
    pub fn num_negative(&self) -> usize {
        self.delta_poc_s0.len()
    }
    pub fn num_positive(&self) -> usize {
        self.delta_poc_s1.len()
    }
    pub fn num_delta_pocs(&self) -> usize {
        self.num_negative() + self.num_positive()
    }
}

/// Parse `st_ref_pic_set(stRpsIdx)` (§7.3.7), given the sets already parsed.
///
/// `num_st_rps` is the SPS's declared total; when `idx == num_st_rps` this set
/// is the one carried in a *slice header*, which is the only case where
/// `delta_idx_minus1` is present. Getting that condition wrong reads one extra
/// `ue(v)` out of every slice header of a stream that uses inter-RPS
/// prediction, which corrupts the slice from its first field onward.
pub fn parse_st_ref_pic_set(
    r: &mut BitReader,
    idx: usize,
    num_st_rps: usize,
    prev: &[ShortTermRps],
) -> Result<ShortTermRps, &'static str> {
    let mut out = ShortTermRps::default();
    let inter_pred = if idx != 0 { r.flag()? } else { false };
    if inter_pred {
        let delta_idx = if idx == num_st_rps { r.ue()? as usize + 1 } else { 1 };
        let ref_idx = idx.checked_sub(delta_idx).ok_or("hevc rps: bad delta_idx")?;
        let rps_ref = prev.get(ref_idx).ok_or("hevc rps: missing reference set")?;
        let delta_rps_sign = r.flag()?;
        let abs_delta_rps = r.ue()? as i32 + 1;
        let delta_rps = if delta_rps_sign { -abs_delta_rps } else { abs_delta_rps };

        let n_ref = rps_ref.num_delta_pocs();
        if n_ref > 2 * MAX_DPB {
            return Err("hevc rps: reference set too large");
        }
        let mut used = alloc::vec![false; n_ref + 1];
        let mut use_delta = alloc::vec![true; n_ref + 1];
        for j in 0..=n_ref {
            used[j] = r.flag()?;
            if !used[j] {
                use_delta[j] = r.flag()?;
            }
        }

        // §7.4.8: derive this set's deltas from the reference set's, walking the
        // reference's positive list backwards for the new negative list and vice
        // versa, so both come out sorted nearest-first.
        let nn = rps_ref.num_negative();
        let np = rps_ref.num_positive();
        // Negative (s0) side.
        for j in (0..np).rev() {
            let d = rps_ref.delta_poc_s1[j] + delta_rps;
            if d < 0 && use_delta[nn + j] {
                out.delta_poc_s0.push(d);
                out.used_s0.push(used[nn + j]);
            }
        }
        if delta_rps < 0 && use_delta[n_ref] {
            out.delta_poc_s0.push(delta_rps);
            out.used_s0.push(used[n_ref]);
        }
        for j in 0..nn {
            let d = rps_ref.delta_poc_s0[j] + delta_rps;
            if d < 0 && use_delta[j] {
                out.delta_poc_s0.push(d);
                out.used_s0.push(used[j]);
            }
        }
        // Positive (s1) side.
        for j in (0..nn).rev() {
            let d = rps_ref.delta_poc_s0[j] + delta_rps;
            if d > 0 && use_delta[j] {
                out.delta_poc_s1.push(d);
                out.used_s1.push(used[j]);
            }
        }
        if delta_rps > 0 && use_delta[n_ref] {
            out.delta_poc_s1.push(delta_rps);
            out.used_s1.push(used[n_ref]);
        }
        for j in 0..np {
            let d = rps_ref.delta_poc_s1[j] + delta_rps;
            if d > 0 && use_delta[nn + j] {
                out.delta_poc_s1.push(d);
                out.used_s1.push(used[nn + j]);
            }
        }
    } else {
        let num_neg = r.ue()? as usize;
        let num_pos = r.ue()? as usize;
        if num_neg > MAX_DPB || num_pos > MAX_DPB {
            return Err("hevc rps: too many reference pictures");
        }
        let mut poc = 0i32;
        for _ in 0..num_neg {
            poc -= r.ue()? as i32 + 1;
            out.delta_poc_s0.push(poc);
            out.used_s0.push(r.flag()?);
        }
        let mut poc = 0i32;
        for _ in 0..num_pos {
            poc += r.ue()? as i32 + 1;
            out.delta_poc_s1.push(poc);
            out.used_s1.push(r.flag()?);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Sequence parameter set (§7.3.2.2)
// ---------------------------------------------------------------------------

/// A parsed HEVC sequence parameter set.
///
/// Note what replaces AVC's macroblock grid: HEVC codes a *coding tree block*
/// whose size the SPS chooses (16/32/64), and the picture dimensions are in
/// **luma samples directly** rather than in blocks — so `pic_width_in_luma_samples`
/// is already the coded width, and only the conformance window is subtracted.
#[derive(Clone, Debug, Default)]
pub struct Sps {
    pub id: u32,
    pub vps_id: u32,
    pub max_sub_layers: u32,
    pub temporal_id_nesting: bool,
    pub ptl: ProfileTierLevel,
    pub chroma_format_idc: u32,
    pub separate_colour_plane: bool,
    pub pic_width_in_luma_samples: u32,
    pub pic_height_in_luma_samples: u32,
    pub conf_win_left: u32,
    pub conf_win_right: u32,
    pub conf_win_top: u32,
    pub conf_win_bottom: u32,
    pub bit_depth_luma: u32,
    pub bit_depth_chroma: u32,
    pub log2_max_poc_lsb: u32,
    pub max_dec_pic_buffering: u32,
    pub max_num_reorder_pics: u32,
    pub log2_min_cb_size: u32,
    pub log2_ctb_size: u32,
    pub log2_min_tb_size: u32,
    pub log2_max_tb_size: u32,
    pub max_transform_hierarchy_depth_inter: u32,
    pub max_transform_hierarchy_depth_intra: u32,
    pub scaling_list_enabled: bool,
    pub scaling_lists: Option<ScalingLists>,
    pub amp_enabled: bool,
    pub sao_enabled: bool,
    pub pcm_enabled: bool,
    pub pcm_bit_depth_luma: u32,
    pub pcm_bit_depth_chroma: u32,
    pub log2_min_pcm_cb_size: u32,
    pub log2_max_pcm_cb_size: u32,
    pub pcm_loop_filter_disabled: bool,
    pub st_rps: Vec<ShortTermRps>,
    pub long_term_ref_pics_present: bool,
    pub lt_ref_poc_lsb_sps: Vec<u32>,
    pub used_by_curr_pic_lt_sps: Vec<bool>,
    pub temporal_mvp_enabled: bool,
    pub strong_intra_smoothing: bool,
}

impl Sps {
    /// (SubWidthC, SubHeightC) per `chroma_format_idc` (H.265 Table 6-1).
    pub fn chroma_subsampling(&self) -> (u32, u32) {
        match self.chroma_format_idc {
            1 => (2, 2), // 4:2:0
            2 => (2, 1), // 4:2:2
            3 => (1, 1), // 4:4:4
            _ => (1, 1), // monochrome
        }
    }

    /// Display width: coded width minus the conformance window, which is
    /// expressed in **chroma** units (so ×SubWidthC), unlike AVC's frame crop
    /// which uses the same rule but is easy to mix up between the two.
    pub fn width(&self) -> u32 {
        let (sub_w, _) = self.chroma_subsampling();
        self.pic_width_in_luma_samples
            .saturating_sub((self.conf_win_left + self.conf_win_right) * sub_w)
    }

    /// Display height: coded height minus the conformance window.
    pub fn height(&self) -> u32 {
        let (_, sub_h) = self.chroma_subsampling();
        self.pic_height_in_luma_samples
            .saturating_sub((self.conf_win_top + self.conf_win_bottom) * sub_h)
    }

    /// Coding-tree-block edge in luma samples (16, 32 or 64).
    pub fn ctb_size(&self) -> u32 {
        1 << self.log2_ctb_size
    }

    /// Picture size in CTBs.
    pub fn ctb_grid(&self) -> (u32, u32) {
        let s = self.ctb_size().max(1);
        (
            (self.pic_width_in_luma_samples + s - 1) / s,
            (self.pic_height_in_luma_samples + s - 1) / s,
        )
    }
}

/// Parse an SPS from its RBSP (the NAL body after the **2-byte** header).
pub fn parse_sps(rbsp: &[u8]) -> Result<Sps, &'static str> {
    let mut r = BitReader::new(rbsp);
    let mut s = Sps::default();
    s.vps_id = r.u(4)?;
    let max_sub_layers_minus1 = r.u(3)?;
    s.max_sub_layers = max_sub_layers_minus1 + 1;
    s.temporal_id_nesting = r.flag()?;
    s.ptl = parse_profile_tier_level(&mut r, true, max_sub_layers_minus1)?;
    s.id = r.ue()?;
    if s.id > 15 {
        return Err("hevc sps: id out of range");
    }
    s.chroma_format_idc = r.ue()?;
    if s.chroma_format_idc > 3 {
        return Err("hevc sps: bad chroma_format_idc");
    }
    if s.chroma_format_idc == 3 {
        s.separate_colour_plane = r.flag()?;
    }
    s.pic_width_in_luma_samples = r.ue()?;
    s.pic_height_in_luma_samples = r.ue()?;
    if s.pic_width_in_luma_samples == 0
        || s.pic_height_in_luma_samples == 0
        || s.pic_width_in_luma_samples > 16384
        || s.pic_height_in_luma_samples > 16384
    {
        return Err("hevc sps: implausible picture size");
    }
    if r.flag()? {
        // conformance_window_flag
        s.conf_win_left = r.ue()?;
        s.conf_win_right = r.ue()?;
        s.conf_win_top = r.ue()?;
        s.conf_win_bottom = r.ue()?;
    }
    s.bit_depth_luma = r.ue()? + 8;
    s.bit_depth_chroma = r.ue()? + 8;
    if s.bit_depth_luma > 16 || s.bit_depth_chroma > 16 {
        return Err("hevc sps: bad bit depth");
    }
    s.log2_max_poc_lsb = r.ue()? + 4;
    if s.log2_max_poc_lsb > 16 {
        return Err("hevc sps: bad log2_max_poc_lsb");
    }
    let sub_layer_ordering_info = r.flag()?;
    let first = if sub_layer_ordering_info { 0 } else { max_sub_layers_minus1 };
    for i in first..=max_sub_layers_minus1 {
        let max_dec = r.ue()? + 1;
        let reorder = r.ue()?;
        let _max_latency_increase_plus1 = r.ue()?;
        if i == max_sub_layers_minus1 {
            s.max_dec_pic_buffering = max_dec;
            s.max_num_reorder_pics = reorder;
        }
    }
    s.log2_min_cb_size = r.ue()? + 3;
    s.log2_ctb_size = s.log2_min_cb_size + r.ue()?;
    s.log2_min_tb_size = r.ue()? + 2;
    s.log2_max_tb_size = s.log2_min_tb_size + r.ue()?;
    if s.log2_ctb_size > 6 || s.log2_min_cb_size < 3 || s.log2_max_tb_size > 5 {
        return Err("hevc sps: block size out of spec range");
    }
    s.max_transform_hierarchy_depth_inter = r.ue()?;
    s.max_transform_hierarchy_depth_intra = r.ue()?;
    s.scaling_list_enabled = r.flag()?;
    if s.scaling_list_enabled {
        if r.flag()? {
            // sps_scaling_list_data_present_flag
            s.scaling_lists = Some(parse_scaling_list_data(&mut r)?);
        } else {
            s.scaling_lists = Some(ScalingLists::default());
        }
    }
    s.amp_enabled = r.flag()?;
    s.sao_enabled = r.flag()?;
    s.pcm_enabled = r.flag()?;
    if s.pcm_enabled {
        s.pcm_bit_depth_luma = r.u(4)? + 1;
        s.pcm_bit_depth_chroma = r.u(4)? + 1;
        s.log2_min_pcm_cb_size = r.ue()? + 3;
        s.log2_max_pcm_cb_size = s.log2_min_pcm_cb_size + r.ue()?;
        s.pcm_loop_filter_disabled = r.flag()?;
    }
    let num_st_rps = r.ue()? as usize;
    if num_st_rps > MAX_ST_RPS {
        return Err("hevc sps: too many short-term RPS");
    }
    for i in 0..num_st_rps {
        let rps = parse_st_ref_pic_set(&mut r, i, num_st_rps, &s.st_rps)?;
        s.st_rps.push(rps);
    }
    s.long_term_ref_pics_present = r.flag()?;
    if s.long_term_ref_pics_present {
        let n = r.ue()? as usize;
        if n > 32 {
            return Err("hevc sps: too many long-term ref pics");
        }
        for _ in 0..n {
            s.lt_ref_poc_lsb_sps.push(r.u(s.log2_max_poc_lsb)?);
            s.used_by_curr_pic_lt_sps.push(r.flag()?);
        }
    }
    s.temporal_mvp_enabled = r.flag()?;
    s.strong_intra_smoothing = r.flag()?;
    // vui_parameters_present_flag and the VUI itself are not needed for
    // geometry or decoding; parsing stops here.
    Ok(s)
}

// ---------------------------------------------------------------------------
// Picture parameter set (§7.3.2.3)
// ---------------------------------------------------------------------------

/// A parsed HEVC picture parameter set.
#[derive(Clone, Debug, Default)]
pub struct Pps {
    pub id: u32,
    pub sps_id: u32,
    pub dependent_slice_segments_enabled: bool,
    pub output_flag_present: bool,
    pub num_extra_slice_header_bits: u32,
    pub sign_data_hiding_enabled: bool,
    pub cabac_init_present: bool,
    pub num_ref_idx_l0_default: u32,
    pub num_ref_idx_l1_default: u32,
    pub init_qp: i32,
    pub constrained_intra_pred: bool,
    pub transform_skip_enabled: bool,
    pub cu_qp_delta_enabled: bool,
    pub diff_cu_qp_delta_depth: u32,
    pub cb_qp_offset: i32,
    pub cr_qp_offset: i32,
    pub slice_chroma_qp_offsets_present: bool,
    pub weighted_pred: bool,
    pub weighted_bipred: bool,
    pub transquant_bypass_enabled: bool,
    pub tiles_enabled: bool,
    pub entropy_coding_sync_enabled: bool,
    pub num_tile_columns: u32,
    pub num_tile_rows: u32,
    pub uniform_spacing: bool,
    pub column_width: Vec<u32>,
    pub row_height: Vec<u32>,
    pub loop_filter_across_tiles_enabled: bool,
    pub loop_filter_across_slices_enabled: bool,
    pub deblocking_filter_control_present: bool,
    pub deblocking_filter_override_enabled: bool,
    pub deblocking_filter_disabled: bool,
    pub beta_offset_div2: i32,
    pub tc_offset_div2: i32,
    pub scaling_lists: Option<ScalingLists>,
    pub lists_modification_present: bool,
    pub log2_parallel_merge_level: u32,
    pub slice_segment_header_extension_present: bool,
}

/// Parse a PPS from its RBSP.
pub fn parse_pps(rbsp: &[u8]) -> Result<Pps, &'static str> {
    let mut r = BitReader::new(rbsp);
    let mut p = Pps::default();
    p.id = r.ue()?;
    p.sps_id = r.ue()?;
    if p.id > 63 || p.sps_id > 15 {
        return Err("hevc pps: id out of range");
    }
    p.dependent_slice_segments_enabled = r.flag()?;
    p.output_flag_present = r.flag()?;
    p.num_extra_slice_header_bits = r.u(3)?;
    p.sign_data_hiding_enabled = r.flag()?;
    p.cabac_init_present = r.flag()?;
    p.num_ref_idx_l0_default = r.ue()? + 1;
    p.num_ref_idx_l1_default = r.ue()? + 1;
    p.init_qp = r.se()? + 26;
    p.constrained_intra_pred = r.flag()?;
    p.transform_skip_enabled = r.flag()?;
    p.cu_qp_delta_enabled = r.flag()?;
    if p.cu_qp_delta_enabled {
        p.diff_cu_qp_delta_depth = r.ue()?;
    }
    p.cb_qp_offset = r.se()?;
    p.cr_qp_offset = r.se()?;
    p.slice_chroma_qp_offsets_present = r.flag()?;
    p.weighted_pred = r.flag()?;
    p.weighted_bipred = r.flag()?;
    p.transquant_bypass_enabled = r.flag()?;
    p.tiles_enabled = r.flag()?;
    p.entropy_coding_sync_enabled = r.flag()?;
    p.num_tile_columns = 1;
    p.num_tile_rows = 1;
    p.uniform_spacing = true;
    p.loop_filter_across_tiles_enabled = true;
    if p.tiles_enabled {
        p.num_tile_columns = r.ue()? + 1;
        p.num_tile_rows = r.ue()? + 1;
        if p.num_tile_columns > 1024 || p.num_tile_rows > 1024 {
            return Err("hevc pps: implausible tile count");
        }
        p.uniform_spacing = r.flag()?;
        if !p.uniform_spacing {
            for _ in 0..p.num_tile_columns - 1 {
                p.column_width.push(r.ue()? + 1);
            }
            for _ in 0..p.num_tile_rows - 1 {
                p.row_height.push(r.ue()? + 1);
            }
        }
        p.loop_filter_across_tiles_enabled = r.flag()?;
    }
    p.loop_filter_across_slices_enabled = r.flag()?;
    p.deblocking_filter_control_present = r.flag()?;
    if p.deblocking_filter_control_present {
        p.deblocking_filter_override_enabled = r.flag()?;
        p.deblocking_filter_disabled = r.flag()?;
        if !p.deblocking_filter_disabled {
            p.beta_offset_div2 = r.se()?;
            p.tc_offset_div2 = r.se()?;
        }
    }
    if r.flag()? {
        // pps_scaling_list_data_present_flag
        p.scaling_lists = Some(parse_scaling_list_data(&mut r)?);
    }
    p.lists_modification_present = r.flag()?;
    p.log2_parallel_merge_level = r.ue()? + 2;
    p.slice_segment_header_extension_present = r.flag()?;
    Ok(p)
}

// ---------------------------------------------------------------------------
// Video parameter set (§7.3.2.1) — only the id and tier/level are used.
// ---------------------------------------------------------------------------

/// The parts of a VPS anything downstream reads. The VPS mostly describes
/// layer sets for scalable/multi-view extensions, which single-layer decoding
/// never consults; it is parsed so the tier/level is available even when the
/// probe sees a VPS before an SPS.
#[derive(Clone, Debug, Default)]
pub struct Vps {
    pub id: u32,
    pub max_sub_layers: u32,
    pub ptl: ProfileTierLevel,
}

/// Parse a VPS from its RBSP.
pub fn parse_vps(rbsp: &[u8]) -> Result<Vps, &'static str> {
    let mut r = BitReader::new(rbsp);
    let mut v = Vps::default();
    v.id = r.u(4)?;
    let _base_layer_internal = r.flag()?;
    let _base_layer_available = r.flag()?;
    let _max_layers_minus1 = r.u(6)?;
    let max_sub_layers_minus1 = r.u(3)?;
    v.max_sub_layers = max_sub_layers_minus1 + 1;
    let _temporal_id_nesting = r.flag()?;
    let _reserved_0xffff = r.u(16)?;
    v.ptl = parse_profile_tier_level(&mut r, true, max_sub_layers_minus1)?;
    Ok(v)
}

// ---------------------------------------------------------------------------
// Slice segment header (§7.3.6.1)
// ---------------------------------------------------------------------------

/// Slice types (H.265 §7.4.7.1). Note the numbering is the **reverse** of
/// AVC's: 0 is B here and I in AVC, so a shared constant would be a silent
/// mis-decode of every slice.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SliceType {
    B = 0,
    P = 1,
    I = 2,
}

impl SliceType {
    fn from_code(v: u32) -> Result<SliceType, &'static str> {
        match v {
            0 => Ok(SliceType::B),
            1 => Ok(SliceType::P),
            2 => Ok(SliceType::I),
            _ => Err("hevc slice: bad slice_type"),
        }
    }
    pub fn is_intra(self) -> bool {
        self == SliceType::I
    }
}

/// A parsed slice segment header — everything the CTU layer and the reference
/// list builder need, plus the bit offset where the CABAC data begins.
#[derive(Clone, Debug)]
pub struct SliceHeader {
    pub first_slice_in_pic: bool,
    pub no_output_of_prior_pics: bool,
    pub pps_id: u32,
    pub dependent_slice_segment: bool,
    pub segment_address: u32,
    pub slice_type: SliceType,
    pub pic_output_flag: bool,
    pub colour_plane_id: u32,
    pub pic_order_cnt_lsb: u32,
    /// The active short-term RPS for this picture: either an SPS set (by index)
    /// or one coded inline in this header.
    pub st_rps: ShortTermRps,
    pub st_rps_idx: Option<u32>,
    pub temporal_mvp_enabled: bool,
    pub sao_luma: bool,
    pub sao_chroma: bool,
    pub num_ref_idx_l0: u32,
    pub num_ref_idx_l1: u32,
    /// Explicit L0 reorder (`list_entry_l0[]`), when present. Empty means the
    /// default concatenation order from the RPS.
    pub list_entry_l0: Vec<u32>,
    /// Explicit L1 reorder (`list_entry_l1[]`), when present.
    pub list_entry_l1: Vec<u32>,
    pub mvd_l1_zero: bool,
    pub cabac_init: bool,
    pub collocated_from_l0: bool,
    pub collocated_ref_idx: u32,
    pub five_minus_max_num_merge_cand: u32,
    pub qp: i32,
    pub cb_qp_offset: i32,
    pub cr_qp_offset: i32,
    pub deblocking_filter_disabled: bool,
    pub beta_offset_div2: i32,
    pub tc_offset_div2: i32,
    pub loop_filter_across_slices_enabled: bool,
    /// Byte offset in the RBSP at which the CABAC-coded slice data starts.
    pub data_byte_offset: usize,
    /// Entry-point offsets for tiles / WPP substreams (already `+1`-resolved).
    pub entry_point_offsets: Vec<u32>,
}

impl SliceHeader {
    pub fn max_num_merge_cand(&self) -> u32 {
        5u32.saturating_sub(self.five_minus_max_num_merge_cand)
    }
}

/// Parse a slice segment header. `nal` is the NAL type (IRAP-ness changes the
/// syntax) and `sps`/`pps` are the active parameter sets.
///
/// Weighted-prediction tables are still consumed for bit length only (weights
/// are not applied). Reference-list modification is parsed **and applied**.
/// Read just the PPS id out of a slice segment header.
///
/// The header cannot be parsed without its PPS (and the PPS's SPS), and the
/// PPS id is the third syntax element — so this reads the two flags before it
/// and stops. Guessing "the most recent PPS" instead works until a stream uses
/// two, which is exactly what an encoder does when it changes QP mid-sequence.
pub fn peek_slice_pps_id(rbsp: &[u8], nal: NalType) -> Result<u32, &'static str> {
    let mut r = BitReader::new(rbsp);
    let _first = r.flag()?;
    if nal.is_irap() {
        let _no_output = r.flag()?;
    }
    r.ue()
}

pub fn parse_slice_header(
    rbsp: &[u8],
    nal: NalType,
    sps: &Sps,
    pps: &Pps,
) -> Result<SliceHeader, &'static str> {
    let mut r = BitReader::new(rbsp);
    let first_slice_in_pic = r.flag()?;
    let no_output_of_prior_pics = if nal.is_irap() { r.flag()? } else { false };
    let pps_id = r.ue()?;
    let mut h = SliceHeader {
        first_slice_in_pic,
        no_output_of_prior_pics,
        pps_id,
        dependent_slice_segment: false,
        segment_address: 0,
        slice_type: SliceType::I,
        pic_output_flag: true,
        colour_plane_id: 0,
        pic_order_cnt_lsb: 0,
        st_rps: ShortTermRps::default(),
        st_rps_idx: None,
        temporal_mvp_enabled: false,
        sao_luma: false,
        sao_chroma: false,
        num_ref_idx_l0: pps.num_ref_idx_l0_default,
        num_ref_idx_l1: pps.num_ref_idx_l1_default,
        list_entry_l0: Vec::new(),
        list_entry_l1: Vec::new(),
        mvd_l1_zero: false,
        cabac_init: false,
        collocated_from_l0: true,
        collocated_ref_idx: 0,
        five_minus_max_num_merge_cand: 0,
        qp: pps.init_qp,
        cb_qp_offset: 0,
        cr_qp_offset: 0,
        deblocking_filter_disabled: pps.deblocking_filter_disabled,
        beta_offset_div2: pps.beta_offset_div2,
        tc_offset_div2: pps.tc_offset_div2,
        loop_filter_across_slices_enabled: pps.loop_filter_across_slices_enabled,
        data_byte_offset: 0,
        entry_point_offsets: Vec::new(),
    };

    let (ctb_w, ctb_h) = sps.ctb_grid();
    let pic_size_in_ctbs = (ctb_w as u64 * ctb_h as u64).max(1);
    if !first_slice_in_pic {
        if pps.dependent_slice_segments_enabled {
            h.dependent_slice_segment = r.flag()?;
        }
        // slice_segment_address is Ceil(Log2(PicSizeInCtbsY)) bits.
        let bits = ceil_log2(pic_size_in_ctbs);
        h.segment_address = r.u(bits)?;
    }

    if !h.dependent_slice_segment {
        for _ in 0..pps.num_extra_slice_header_bits {
            let _slice_reserved_flag = r.flag()?;
        }
        h.slice_type = SliceType::from_code(r.ue()?)?;
        if pps.output_flag_present {
            h.pic_output_flag = r.flag()?;
        }
        if sps.separate_colour_plane {
            h.colour_plane_id = r.u(2)?;
        }
        if !nal.is_idr() {
            h.pic_order_cnt_lsb = r.u(sps.log2_max_poc_lsb)?;
            let short_term_ref_pic_set_sps_flag = r.flag()?;
            if !short_term_ref_pic_set_sps_flag {
                h.st_rps = parse_st_ref_pic_set(&mut r, sps.st_rps.len(), sps.st_rps.len(), &sps.st_rps)?;
            } else if !sps.st_rps.is_empty() {
                let bits = ceil_log2(sps.st_rps.len() as u64);
                let idx = if bits > 0 { r.u(bits)? } else { 0 };
                let rps = sps.st_rps.get(idx as usize).ok_or("hevc slice: st_rps index out of range")?;
                h.st_rps = rps.clone();
                h.st_rps_idx = Some(idx);
            }
            if sps.long_term_ref_pics_present {
                let num_lt_sps = if !sps.lt_ref_poc_lsb_sps.is_empty() { r.ue()? } else { 0 };
                let num_lt_pics = r.ue()?;
                if num_lt_sps + num_lt_pics > 64 {
                    return Err("hevc slice: too many long-term refs");
                }
                let mut prev_delta_msb_present = false;
                for i in 0..(num_lt_sps + num_lt_pics) {
                    if i < num_lt_sps {
                        if sps.lt_ref_poc_lsb_sps.len() > 1 {
                            let bits = ceil_log2(sps.lt_ref_poc_lsb_sps.len() as u64);
                            let _lt_idx_sps = r.u(bits)?;
                        }
                    } else {
                        let _poc_lsb_lt = r.u(sps.log2_max_poc_lsb)?;
                        let _used_by_curr_pic_lt = r.flag()?;
                    }
                    let delta_poc_msb_present = r.flag()?;
                    if delta_poc_msb_present {
                        let _delta_poc_msb_cycle_lt = r.ue()?;
                    }
                    prev_delta_msb_present = delta_poc_msb_present;
                }
                let _ = prev_delta_msb_present;
            }
            if sps.temporal_mvp_enabled {
                h.temporal_mvp_enabled = r.flag()?;
            }
        }
        if sps.sao_enabled {
            h.sao_luma = r.flag()?;
            if sps.chroma_format_idc != 0 {
                h.sao_chroma = r.flag()?;
            }
        }
        if h.slice_type != SliceType::I {
            if r.flag()? {
                // num_ref_idx_active_override_flag
                h.num_ref_idx_l0 = r.ue()? + 1;
                if h.slice_type == SliceType::B {
                    h.num_ref_idx_l1 = r.ue()? + 1;
                }
            }
            if h.num_ref_idx_l0 > 16 || h.num_ref_idx_l1 > 16 {
                return Err("hevc slice: num_ref_idx out of range");
            }
            // ref_pic_lists_modification(): present only when the PPS allows it
            // *and* the candidate list has more than one entry. The entries are
            // **indices into the unmodified RPS concatenation**, not into the
            // final list — and they must be applied, not merely consumed for
            // their bit length. B-pyramid streams reorder L0/L1 so a later B
            // can prefer a mid-GOP B-ref; discarding the reorder leaves the
            // default order, which is a different picture at the same index
            // and desynchronises every CABAC context that depends on the
            // reconstructed neighbours of that wrong prediction.
            let num_pic_total_curr = num_pic_total_curr(&h.st_rps, sps);
            if pps.lists_modification_present && num_pic_total_curr > 1 {
                let bits = ceil_log2(num_pic_total_curr as u64);
                if r.flag()? {
                    for _ in 0..h.num_ref_idx_l0 {
                        h.list_entry_l0.push(r.u(bits)?);
                    }
                }
                if h.slice_type == SliceType::B && r.flag()? {
                    for _ in 0..h.num_ref_idx_l1 {
                        h.list_entry_l1.push(r.u(bits)?);
                    }
                }
            }
            if h.slice_type == SliceType::B {
                h.mvd_l1_zero = r.flag()?;
            }
            if pps.cabac_init_present {
                h.cabac_init = r.flag()?;
            }
            if h.temporal_mvp_enabled {
                if h.slice_type == SliceType::B {
                    h.collocated_from_l0 = r.flag()?;
                }
                let n = if h.collocated_from_l0 { h.num_ref_idx_l0 } else { h.num_ref_idx_l1 };
                if n > 1 {
                    h.collocated_ref_idx = r.ue()?;
                }
            }
            if (pps.weighted_pred && h.slice_type == SliceType::P)
                || (pps.weighted_bipred && h.slice_type == SliceType::B)
            {
                skip_pred_weight_table(&mut r, sps, &h)?;
            }
            h.five_minus_max_num_merge_cand = r.ue()?;
            if h.five_minus_max_num_merge_cand > 4 {
                return Err("hevc slice: bad max_num_merge_cand");
            }
        }
        h.qp = pps.init_qp + r.se()?;
        if pps.slice_chroma_qp_offsets_present {
            h.cb_qp_offset = r.se()?;
            h.cr_qp_offset = r.se()?;
        }
        let mut deblocking_filter_override = false;
        if pps.deblocking_filter_override_enabled {
            deblocking_filter_override = r.flag()?;
        }
        if deblocking_filter_override {
            h.deblocking_filter_disabled = r.flag()?;
            if !h.deblocking_filter_disabled {
                h.beta_offset_div2 = r.se()?;
                h.tc_offset_div2 = r.se()?;
            }
        }
        if pps.loop_filter_across_slices_enabled && (h.sao_luma || h.sao_chroma || !h.deblocking_filter_disabled) {
            h.loop_filter_across_slices_enabled = r.flag()?;
        }
    }

    if pps.tiles_enabled || pps.entropy_coding_sync_enabled {
        let num_entry_points = r.ue()?;
        if num_entry_points > 0 {
            if num_entry_points > 1 << 20 {
                return Err("hevc slice: implausible entry point count");
            }
            let offset_len = r.ue()? + 1;
            if offset_len > 32 {
                return Err("hevc slice: bad offset_len");
            }
            for _ in 0..num_entry_points {
                h.entry_point_offsets.push(r.u(offset_len)? + 1);
            }
        }
    }
    if pps.slice_segment_header_extension_present {
        let len = r.ue()?;
        if len > rbsp.len() as u32 {
            return Err("hevc slice: bad header extension length");
        }
        for _ in 0..len {
            let _ = r.u(8)?;
        }
    }
    // byte_alignment(): the alignment_bit_equal_to_one then zeros to the byte.
    let _alignment_one = r.bit()?;
    r.byte_align();
    h.data_byte_offset = r.bit_pos() / 8;
    if h.data_byte_offset > rbsp.len() {
        return Err("hevc slice: header ran past the RBSP");
    }
    Ok(h)
}

/// `NumPicTotalCurr` (§7.4.7.2): how many reference pictures the current picture
/// may actually use — the size of the candidate list that `list_entry_l*` indexes.
fn num_pic_total_curr(rps: &ShortTermRps, sps: &Sps) -> usize {
    let mut n = rps.used_s0.iter().filter(|&&u| u).count() + rps.used_s1.iter().filter(|&&u| u).count();
    n += sps.used_by_curr_pic_lt_sps.iter().filter(|&&u| u).count();
    n
}

/// Consume `pred_weight_table()` (§7.3.6.3) without keeping the values.
///
/// It is *consumed*, not skipped by a guessed length: the table's size depends
/// on the chroma format and on a per-entry flag, so there is no constant to skip
/// by, and getting here wrong moves the CABAC start offset.
fn skip_pred_weight_table(r: &mut BitReader, sps: &Sps, h: &SliceHeader) -> Result<(), &'static str> {
    let _luma_log2_weight_denom = r.ue()?;
    if sps.chroma_format_idc != 0 {
        let _delta_chroma_log2_weight_denom = r.se()?;
    }
    for list in 0..2 {
        if list == 1 && h.slice_type != SliceType::B {
            break;
        }
        let n = if list == 0 { h.num_ref_idx_l0 } else { h.num_ref_idx_l1 };
        let mut luma_flags = [false; 16];
        let mut chroma_flags = [false; 16];
        for i in 0..n as usize {
            luma_flags[i.min(15)] = r.flag()?;
        }
        if sps.chroma_format_idc != 0 {
            for i in 0..n as usize {
                chroma_flags[i.min(15)] = r.flag()?;
            }
        }
        for i in 0..n as usize {
            if luma_flags[i.min(15)] {
                let _delta_luma_weight = r.se()?;
                let _luma_offset = r.se()?;
            }
            if chroma_flags[i.min(15)] {
                for _ in 0..2 {
                    let _delta_chroma_weight = r.se()?;
                    let _delta_chroma_offset = r.se()?;
                }
            }
        }
    }
    Ok(())
}

/// `Ceil(Log2(v))` — the width in bits of an index into `v` values.
pub fn ceil_log2(v: u64) -> u32 {
    if v <= 1 {
        return 0;
    }
    64 - (v - 1).leading_zeros()
}

/// The CABAC initialisation table to use for a slice (H.265 §9.3.2.2).
///
/// `2 - slice_type` maps I->0, P->1, B->2 (HEVC numbers its slice types the
/// reverse of AVC, which is itself a standing trap here). `cabac_init_flag`
/// then XORs by 3 on a **non-I** slice, trading the P and B tables — an encoder
/// says so when a slice's statistics look more like the other kind.
pub fn cabac_init_type(slice_type: SliceType, cabac_init_flag: bool) -> usize {
    let base = 2 - slice_type as usize;
    if cabac_init_flag && slice_type != SliceType::I {
        base ^ 3
    } else {
        base
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn nal_header_is_two_bytes() {
        // 0x40 0x01 = forbidden 0, type 32 (VPS), layer 0, tid_plus1 1.
        // Reading this with AVC's `& 0x1f` would give type 0 — a slice.
        let data = [0x00, 0x00, 0x00, 0x01, 0x40, 0x01, 0xaa, 0x00, 0x00, 0x01, 0x42, 0x01, 0xbb];
        let nals = split_annexb(&data);
        assert_eq!(nals.len(), 2);
        assert_eq!(nals[0].kind, NalType::Vps);
        assert_eq!(nals[0].temporal_id, 0);
        assert_eq!(nals[0].layer_id, 0);
        assert_eq!(nals[1].kind, NalType::Sps); // 0x42 >> 1 = 33
    }

    #[test_case]
    fn nal_type_classification() {
        assert!(NalType::from_code(19).is_idr()); // IDR_W_RADL
        assert!(NalType::from_code(20).is_idr()); // IDR_N_LP
        assert!(NalType::from_code(21).is_irap() && !NalType::from_code(21).is_idr()); // CRA
        assert!(NalType::from_code(16).is_bla());
        assert!(NalType::from_code(1).is_slice() && !NalType::from_code(1).is_irap());
        // *_N types (even, < 16) are sub-layer non-reference: droppable.
        assert!(NalType::from_code(0).is_sublayer_nonref()); // TRAIL_N
        assert!(!NalType::from_code(1).is_sublayer_nonref()); // TRAIL_R
        assert!(NalType::from_code(8).is_sublayer_nonref()); // RASL_N
        assert!(!NalType::from_code(9).is_sublayer_nonref()); // RASL_R
        // An IRAP is never droppable even though 16/18/20 are even.
        assert!(!NalType::from_code(16).is_sublayer_nonref());
    }

    #[test_case]
    fn hvcc_splits_on_length_prefix() {
        let data = [
            0x00, 0x00, 0x00, 0x04, 0x26, 0x01, 0x11, 0x22, // type 19 (IDR_W_RADL)
            0x00, 0x00, 0x00, 0x03, 0x02, 0x01, 0x33, // type 1 (TRAIL_R)
        ];
        let nals = split_hvcc(&data, 4);
        assert_eq!(nals.len(), 2);
        assert!(nals[0].kind.is_idr());
        assert_eq!(nals[1].kind, NalType::Slice(1));
    }

    #[test_case]
    fn ceil_log2_matches_spec() {
        assert_eq!(ceil_log2(0), 0);
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(5), 3);
        assert_eq!(ceil_log2(64), 6);
        assert_eq!(ceil_log2(65), 7);
    }

    // Real x265 4.2 output for a 176x144 clip (Main profile, level 2.0), taken
    // from the `hvcC` box of an mp4 muxed by PyAV/libx265. Regenerate with
    // `tools/hevcdiff` (`--dump-params`), which prints these arrays.
    //
    // Each fixture is the **NAL including its 2-byte header**, so the tests
    // exercise the header split and the RBSP unescape as well as the parse —
    // the emulation-prevention bytes in these are real (`00 00 03 00`), which is
    // what makes them a better fixture than a hand-built bit pattern.
    const X265_VPS_NAL: [u8; 24] = [
        0x40, 0x01, 0x0c, 0x01, 0xff, 0xff, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0x90, 0x00, 0x00,
        0x03, 0x00, 0x00, 0x03, 0x00, 0x3c, 0x95, 0x98, 0x09,
    ];
    const X265_SPS_NAL: [u8; 41] = [
        0x42, 0x01, 0x01, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0x90, 0x00, 0x00, 0x03, 0x00, 0x00,
        0x03, 0x00, 0x3c, 0xa0, 0x16, 0x20, 0x24, 0x59, 0x65, 0x66, 0x92, 0x4c, 0xae, 0x68, 0x08,
        0x00, 0x00, 0x03, 0x00, 0x08, 0x00, 0x00, 0x03, 0x00, 0xc8, 0x40,
    ];
    const X265_PPS_NAL: [u8; 7] = [0x44, 0x01, 0xc1, 0x72, 0xb4, 0x62, 0x40];

    #[test_case]
    fn sps_main_profile_geometry() {
        let sps = parse_sps(&unescape_rbsp(&X265_SPS_NAL[2..])).unwrap();
        assert_eq!(sps.ptl.profile_idc, 1, "Main profile");
        assert_eq!(sps.ptl.profile_name(), "Main");
        assert_eq!(sps.ptl.level_idc, 60, "level 2.0 is coded as 30*2");
        assert!(!sps.ptl.tier_high, "Main tier");
        assert_eq!(sps.chroma_format_idc, 1, "4:2:0");
        assert_eq!(sps.bit_depth_luma, 8);
        assert_eq!(sps.bit_depth_chroma, 8);
        assert_eq!(sps.pic_width_in_luma_samples, 176);
        assert_eq!(sps.pic_height_in_luma_samples, 144);
        assert_eq!(sps.width(), 176);
        assert_eq!(sps.height(), 144);
        // The picture size is in luma samples directly — there is no macroblock
        // grid — and the CTB size is the SPS's choice, here x265's default 64.
        assert_eq!(sps.log2_ctb_size, 6);
        assert_eq!(sps.ctb_size(), 64);
        assert_eq!(sps.ctb_grid(), (3, 3), "ceil(176/64) x ceil(144/64)");
        assert_eq!(sps.log2_min_cb_size, 3);
        assert_eq!(sps.log2_min_tb_size, 2);
        assert_eq!(sps.log2_max_tb_size, 5);
        assert_eq!(sps.log2_max_poc_lsb, 8);
        assert_eq!(sps.max_dec_pic_buffering, 5);
        assert_eq!(sps.max_num_reorder_pics, 2);
        assert!(sps.sao_enabled);
        assert!(sps.temporal_mvp_enabled);
        assert!(sps.strong_intra_smoothing);
        assert!(!sps.amp_enabled);
        assert!(!sps.scaling_list_enabled);
        assert_eq!(sps.st_rps.len(), 0, "x265 codes its RPS in slice headers");
    }

    #[test_case]
    fn pps_fields() {
        let pps = parse_pps(&unescape_rbsp(&X265_PPS_NAL[2..])).unwrap();
        assert_eq!(pps.id, 0);
        assert_eq!(pps.sps_id, 0);
        assert_eq!(pps.init_qp, 26);
        assert!(pps.sign_data_hiding_enabled);
        assert!(pps.cu_qp_delta_enabled);
        assert_eq!(pps.diff_cu_qp_delta_depth, 1);
        assert!(pps.weighted_pred);
        assert!(!pps.weighted_bipred);
        assert!(!pps.tiles_enabled);
        assert!(pps.entropy_coding_sync_enabled, "x265 defaults to WPP");
        assert_eq!(pps.num_tile_columns, 1);
        assert_eq!(pps.num_tile_rows, 1);
        assert_eq!(pps.log2_parallel_merge_level, 2);
        assert!(!pps.deblocking_filter_control_present);
        assert!(pps.loop_filter_across_slices_enabled);
    }

    #[test_case]
    fn vps_tier_and_level() {
        let vps = parse_vps(&unescape_rbsp(&X265_VPS_NAL[2..])).unwrap();
        assert_eq!(vps.id, 0);
        assert_eq!(vps.max_sub_layers, 1);
        // The VPS and SPS must agree — they carry the same profile_tier_level,
        // and a bit-budget error in it shows up as a disagreement here.
        let sps = parse_sps(&unescape_rbsp(&X265_SPS_NAL[2..])).unwrap();
        assert_eq!(vps.ptl.profile_idc, sps.ptl.profile_idc);
        assert_eq!(vps.ptl.level_idc, sps.ptl.level_idc);
        assert_eq!(vps.ptl.compat_flags, sps.ptl.compat_flags);
    }

    #[test_case]
    fn short_term_rps_resolves_running_deltas() {
        // num_negative=2, num_positive=0, deltas 1 and 2 (so POC -1 and -3),
        // both used. ue(2)=011, ue(0)=1, then per-pic: ue(0)=1 used=1, ue(1)=010
        // used=1.
        //   011 1 1 1 010 1  → 0b0111_1101 0b0101_0000
        let bytes = [0b0111_1101, 0b0101_0000];
        let mut r = BitReader::new(&bytes);
        let rps = parse_st_ref_pic_set(&mut r, 0, 1, &[]).unwrap();
        assert_eq!(rps.delta_poc_s0, alloc::vec![-1, -3]);
        assert_eq!(rps.used_s0, alloc::vec![true, true]);
        assert_eq!(rps.num_positive(), 0);
        assert_eq!(rps.num_delta_pocs(), 2);
    }

    #[test_case]
    fn default_scaling_lists_are_the_spec_matrices_not_flat() {
        // A stream that enables scaling lists but transmits none expects H.265
        // Table 7-6, which is *not* flat 16 above 4x4. A flat default decodes
        // every such stream slightly soft rather than erroring.
        let sl = ScalingLists::default();
        assert_eq!(sl.lists[0][0][..16], [16u8; 16], "4x4 default is flat");
        assert_eq!(sl.lists[1][0][0], 16);
        assert_eq!(sl.lists[1][0][63], 115, "8x8 intra default is a real matrix");
        assert_eq!(sl.lists[1][3][63], 91, "8x8 inter default differs from intra");
        assert_eq!(sl.lists[3][0][63], 115, "32x32 shares the 8x8 default table");
    }

    #[test_case]
    fn rejects_garbage() {
        assert!(parse_sps(&[]).is_err());
        assert!(parse_pps(&[]).is_err());
        assert!(parse_sps(&[0xff; 8]).is_err());
    }

    /// The context-initialisation derivation (H.265 §9.3.2.2), pinned against
    /// values computed by hand from FFmpeg's own table.
    ///
    /// FFmpeg writes it as `pre = 2p - 127; pre ^= pre >> 31; clamp`, which
    /// reads like an absolute value and is not: for negative `x`,
    /// `x ^ (x >> 31)` is `-x - 1`. That off-by-one *is* the specification's
    /// asymmetry between `63 - p` (MPS 0) and `p - 64` (MPS 1), so an
    /// implementation that "cleans it up" into `abs()` is wrong on exactly the
    /// half of the contexts that start with valMPS 0 — every one of them one
    /// state too confident, which no bitstream rejects.
    #[test_case]
    fn hevc_cabac_init_matches_the_specification_derivation() {
        use crate::video::h264::cabac::Cabac;
        use cabac_tables as ct;

        // A byte-aligned run of bits is enough: this asserts the context array,
        // not the arithmetic engine (which is H.264's, tested separately).
        let data = [0x55u8; 16];

        let c = Cabac::new_hevc(&data, 26, 2, &ct::INIT_VALUES).unwrap();
        assert_eq!(c.ctx[ct::SAO_MERGE_FLAG], 14);
        assert_eq!(c.ctx[ct::SPLIT_CODING_UNIT_FLAG], 32);
        assert_eq!(c.ctx[ct::CBF_LUMA], 14);

        let c = Cabac::new_hevc(&data, 37, 0, &ct::INIT_VALUES).unwrap();
        assert_eq!(c.ctx[ct::SPLIT_CODING_UNIT_FLAG], 6);
        assert_eq!(c.ctx[ct::CBF_LUMA], 11);

        // Every state must be a legal (pStateIdx, valMPS) pair: pStateIdx 0..=62
        // (63 is unreachable — `preCtxState` is clipped to 1..=126).
        for qp in [0, 26, 51] {
            for it in 0..3 {
                let c = Cabac::new_hevc(&data, qp, it, &ct::INIT_VALUES).unwrap();
                for &s in c.ctx[..ct::HEVC_CONTEXTS].iter() {
                    assert!((s >> 1) <= 62, "qp {qp} init_type {it}: state {s}");
                }
            }
        }

        // The QP is clipped, not wrapped: an out-of-range slice QP (which a
        // malformed stream can produce) must not index off the curve.
        assert_eq!(
            Cabac::new_hevc(&data, -20, 2, &ct::INIT_VALUES).unwrap().ctx[ct::CBF_LUMA],
            Cabac::new_hevc(&data, 0, 2, &ct::INIT_VALUES).unwrap().ctx[ct::CBF_LUMA],
        );
        assert!(Cabac::new_hevc(&data, 26, 3, &ct::INIT_VALUES).is_err());
    }

    /// The generated constant tables, pinned by the *mathematical* properties
    /// they must have rather than by re-listing their values (a second copy of
    /// a table checks only that it was copied twice the same way).
    ///
    /// Each of these catches a single transposed digit anywhere in ~1200
    /// numbers, which is the class of mistake that would otherwise show up as a
    /// picture drifting over a GOP rather than as a failure.
    #[test_case]
    fn hevc_constant_tables_have_their_defining_structure() {
        use tables as tb;

        // The DCT basis. Row 0 is DC, and every row is symmetric (even) or
        // antisymmetric (odd) about the centre — which is what makes the
        // butterfly decomposition valid, so a wrong value here is not merely a
        // wrong coefficient, it breaks the fast transform's premise.
        assert!(tb::TRANSFORM[0].iter().all(|&v| v == 64));
        for (j, row) in tb::TRANSFORM.iter().enumerate() {
            for i in 0..32 {
                let mirror = row[31 - i] as i32;
                let want = if j % 2 == 0 { row[i] as i32 } else { -(row[i] as i32) };
                assert_eq!(mirror, want, "TRANSFORM[{j}][{i}]");
            }
        }
        // Rows of opposite parity are exactly orthogonal (same-parity rows are
        // only approximately so — the integer basis is not a true DCT).
        for j in 0..32 {
            for k in 0..32 {
                if (j + k) % 2 == 1 {
                    let dot: i32 = (0..32)
                        .map(|i| tb::TRANSFORM[j][i] as i32 * tb::TRANSFORM[k][i] as i32)
                        .sum();
                    assert_eq!(dot, 0, "rows {j},{k} not orthogonal");
                }
            }
        }
        // The size-4/8/16 transforms are this matrix sub-sampled by 32/N, which
        // is why the specification defines only one.
        for &n in &[4usize, 8, 16] {
            let step = 32 / n;
            for j in 0..n {
                for i in 0..n {
                    let v = tb::TRANSFORM[j * step][i] as i32;
                    let m = tb::TRANSFORM[j * step][n - 1 - i] as i32;
                    assert_eq!(m, if j % 2 == 0 { v } else { -v }, "N={n} [{j}][{i}]");
                }
            }
        }

        // Interpolation must have unit DC gain, or a flat area changes
        // brightness with sub-pixel motion — a slow luminance drift, not an
        // artefact anyone would call a decoder bug.
        for (p, f) in tb::QPEL_FILTERS.iter().enumerate().skip(1) {
            assert_eq!(f.iter().map(|&t| t as i32).sum::<i32>(), 64, "qpel phase {p}");
        }
        for (p, f) in tb::EPEL_FILTERS.iter().enumerate().skip(1) {
            assert_eq!(f.iter().map(|&t| t as i32).sum::<i32>(), 64, "epel phase {p}");
        }
        assert!(tb::QPEL_FILTERS[0].iter().all(|&t| t == 0));
        assert!(tb::EPEL_FILTERS[0].iter().all(|&t| t == 0));

        // A scan order is a permutation: every position visited exactly once.
        // A duplicate would silently drop a coefficient and decode another
        // twice — the transform still runs and the block is merely wrong.
        let mut seen = [false; 16];
        for i in 0..16 {
            let k = (tb::DIAG_SCAN4X4_Y[i] * 4 + tb::DIAG_SCAN4X4_X[i]) as usize;
            assert!(!seen[k], "4x4 scan visits {k} twice");
            seen[k] = true;
        }
        assert!(seen.iter().all(|&b| b));
        let mut seen = [false; 64];
        for i in 0..64 {
            let k = (tb::DIAG_SCAN8X8_Y[i] * 8 + tb::DIAG_SCAN8X8_X[i]) as usize;
            assert!(!seen[k], "8x8 scan visits {k} twice");
            seen[k] = true;
        }
        assert!(seen.iter().all(|&b| b));

        // Angular intra: 33 modes (2..=34), antisymmetric about mode 18, with
        // mode 10 (horizontal) and 26 (vertical) at angle 0.
        assert_eq!(tb::INTRA_PRED_ANGLE.len(), 33);
        assert_eq!(tb::INTRA_PRED_ANGLE[10 - 2], 0);
        assert_eq!(tb::INTRA_PRED_ANGLE[26 - 2], 0);
        for i in 0..17 {
            assert_eq!(tb::INTRA_PRED_ANGLE[i], -tb::INTRA_PRED_ANGLE[16 + i]);
        }
        // `INV_ANGLE` is `round(8192 / angle)` over the 15 negative-angle modes
        // 11..=25 — it projects the far reference into the extension, so a
        // wrong entry samples the wrong pixel only at the edges of a block,
        // which is exactly where it is hardest to see.
        assert_eq!(tb::INV_ANGLE.len(), 15);
        for (i, mode) in (11..=25usize).enumerate() {
            let ang = tb::INTRA_PRED_ANGLE[mode - 2] as i32;
            assert!(ang < 0, "mode {mode} should have a negative angle");
            let want = (2 * 8192 / ang - 1) / 2; // round-half-away, ang < 0
            assert_eq!(tb::INV_ANGLE[i] as i32, want, "INV_ANGLE[{i}] (mode {mode})");
        }

        // Deblocking curves are monotonic in QP and start flat: below QP 16 no
        // filtering happens at all, which is the property that makes a
        // high-quality stream bit-exact through the loop filter.
        assert!(tb::TC_TABLE.windows(2).all(|w| w[1] >= w[0]));
        assert!(tb::BETA_TABLE.windows(2).all(|w| w[1] >= w[0]));
        assert_eq!(tb::BETA_TABLE[15], 0);
        assert_eq!(tb::BETA_TABLE[16], 6);

        assert_eq!(tb::LEVEL_SCALE, [40, 45, 51, 57, 64, 72]);
        assert_eq!(tb::QP_C.len(), 14); // qPi 30..=43
        assert_eq!(tb::DEFAULT_SCALING_LIST_INTRA[0], 16);
        assert_eq!(tb::DEFAULT_SCALING_LIST_INTER[0], 16);
    }

    /// `cabac_init_flag` swaps the P and B initialisation tables and leaves I
    /// alone — the flag's entire purpose, and invisible if implemented as a
    /// no-op (a P slice decoded with B's probabilities still decodes, just
    /// worse, so it surfaces as drift rather than as an error).
    #[test_case]
    fn cabac_init_flag_swaps_p_and_b_only() {
        assert_eq!(cabac_init_type(SliceType::I, false), 0);
        assert_eq!(cabac_init_type(SliceType::P, false), 1);
        assert_eq!(cabac_init_type(SliceType::B, false), 2);
        // Set, on a non-I slice: P and B trade tables.
        assert_eq!(cabac_init_type(SliceType::P, true), 2);
        assert_eq!(cabac_init_type(SliceType::B, true), 1);
        // An I slice ignores it. `0 ^ 3` would be 3, which is off the end of a
        // three-row table — so a flag applied unconditionally is not a wrong
        // picture, it is an out-of-bounds index.
        assert_eq!(cabac_init_type(SliceType::I, true), 0);
    }
}
