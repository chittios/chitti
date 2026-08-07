//! H.264 / AVC parameter-set and NAL-unit parsing (ITU-T H.264 §7).
//!
//! This is the first decode stage: split a coded stream into NAL units (both
//! Annex-B start-code framing and the length-prefixed AVCC framing MP4 carries)
//! and parse the SPS/PPS so we know the picture geometry, profile, and entropy
//! mode. The pixel pipeline (slice → CAVLC → intra/inter → transform →
//! deblock → reconstruct) is built on top of this in later stages; nothing here
//! decodes samples yet, but everything here is exercised by the full decoder.
//!
//! Pure + panic-free: malformed input returns `Err`, never crashes.

use super::bits::{unescape_rbsp, BitReader};
use alloc::vec::Vec;

pub mod cabac;
pub mod cabac_tables;
pub mod cavlc;
pub mod deblock;
pub mod decoder;
pub mod decoder_cabac;
pub mod encoder;
pub mod inter;
pub mod intra;
pub mod transform;

/// NAL unit types we distinguish (H.264 Table 7-1). Others are ignored.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NalType {
    Slice,     // 1  coded slice of a non-IDR picture
    SliceIdr,  // 5  coded slice of an IDR picture (keyframe)
    Sps,       // 7  sequence parameter set
    Pps,       // 8  picture parameter set
    Sei,       // 6  supplemental enhancement info
    AuDelim,   // 9  access-unit delimiter
    Other(u8), // anything else
}

impl NalType {
    fn from_code(t: u8) -> NalType {
        match t {
            1 => NalType::Slice,
            5 => NalType::SliceIdr,
            6 => NalType::Sei,
            7 => NalType::Sps,
            8 => NalType::Pps,
            9 => NalType::AuDelim,
            other => NalType::Other(other),
        }
    }
    pub fn is_slice(self) -> bool {
        matches!(self, NalType::Slice | NalType::SliceIdr)
    }
}

/// One NAL unit: its type, `nal_ref_idc`, and the payload *including* the
/// 1-byte header (so the RBSP is `unescape_rbsp(&payload[1..])`).
pub struct Nal<'a> {
    pub kind: NalType,
    pub ref_idc: u8,
    pub payload: &'a [u8],
}

impl<'a> Nal<'a> {
    /// The RBSP (emulation-prevention bytes removed) of this NAL's body.
    pub fn rbsp(&self) -> Vec<u8> {
        unescape_rbsp(&self.payload[1..])
    }
}

/// Split an **Annex-B** byte stream (`.h264`/`.264`, transport payloads) into
/// NAL units on `00 00 01` / `00 00 00 01` start codes.
pub fn split_annexb(data: &[u8]) -> Vec<Nal<'_>> {
    let mut nals = Vec::new();
    let mut i = 0;
    let n = data.len();
    // Find the first start code.
    let mut start = find_start_code(data, 0);
    while let Some((sc_pos, sc_len)) = start {
        let unit_start = sc_pos + sc_len;
        let next = find_start_code(data, unit_start);
        let unit_end = next.map(|(p, _)| p).unwrap_or(n);
        if unit_end > unit_start {
            push_nal(&mut nals, &data[unit_start..unit_end]);
        }
        start = next;
        i = unit_end;
    }
    let _ = i;
    nals
}

/// Locate the next start code at or after `from`, returning `(position, length)`
/// where length is 3 (`00 00 01`) or 4 (`00 00 00 01`).
fn find_start_code(data: &[u8], from: usize) -> Option<(usize, usize)> {
    let mut i = from;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            // A leading extra zero makes it a 4-byte start code.
            if i > 0 && data[i - 1] == 0 {
                return Some((i - 1, 4));
            }
            return Some((i, 3));
        }
        i += 1;
    }
    None
}

/// Split **AVCC** (ISO-BMFF) sample data: each NAL is prefixed by a
/// big-endian length of `length_size` bytes (1..=4, from the `avcC` box).
pub fn split_avcc(data: &[u8], length_size: u8) -> Vec<Nal<'_>> {
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

fn push_nal<'a>(nals: &mut Vec<Nal<'a>>, payload: &'a [u8]) {
    if payload.is_empty() {
        return;
    }
    let hdr = payload[0];
    // forbidden_zero_bit must be 0; if set, skip (corrupt).
    if hdr & 0x80 != 0 {
        return;
    }
    nals.push(Nal {
        kind: NalType::from_code(hdr & 0x1f),
        ref_idc: (hdr >> 5) & 0x3,
        payload,
    });
}

/// A parsed sequence parameter set — the fields the decoder and the probe need.
#[derive(Clone, Debug, Default)]
pub struct Sps {
    pub id: u32,
    pub profile_idc: u8,
    pub level_idc: u8,
    pub chroma_format_idc: u32,
    pub bit_depth_luma: u32,
    pub bit_depth_chroma: u32,
    pub log2_max_frame_num: u32,
    pub pic_order_cnt_type: u32,
    pub log2_max_poc_lsb: u32,
    pub max_num_ref_frames: u32,
    pub pic_width_in_mbs: u32,
    pub pic_height_in_map_units: u32,
    pub frame_mbs_only_flag: bool,
    pub mb_adaptive_frame_field: bool,
    pub direct_8x8_inference: bool,
    // Cropping (in chroma-sampled units — apply with SubWidthC/SubHeightC).
    pub crop_left: u32,
    pub crop_right: u32,
    pub crop_top: u32,
    pub crop_bottom: u32,
}

impl Sps {
    /// Decoded luma width in pixels (macroblock grid minus cropping).
    pub fn width(&self) -> u32 {
        let w = self.pic_width_in_mbs * 16;
        let (sub_w, _) = self.chroma_subsampling();
        w.saturating_sub((self.crop_left + self.crop_right) * sub_w)
    }

    /// Decoded luma height in pixels (accounts for field/frame + cropping).
    pub fn height(&self) -> u32 {
        let frame_h = (2 - self.frame_mbs_only_flag as u32) * self.pic_height_in_map_units * 16;
        let (_, sub_h) = self.chroma_subsampling();
        let crop_unit_y = sub_h * (2 - self.frame_mbs_only_flag as u32);
        frame_h.saturating_sub((self.crop_top + self.crop_bottom) * crop_unit_y)
    }

    /// (SubWidthC, SubHeightC) per chroma_format_idc (H.264 Table 6-1). 4:2:0
    /// (idc 1) → (2,2); 4:2:2 → (2,1); 4:4:4 → (1,1); monochrome → (1,1).
    fn chroma_subsampling(&self) -> (u32, u32) {
        match self.chroma_format_idc {
            1 => (2, 2),
            2 => (2, 1),
            _ => (1, 1),
        }
    }
}

/// True for the High-profile family that carries the extra chroma/scaling
/// syntax in the SPS (H.264 §7.3.2.1.1).
fn is_high_profile(profile_idc: u8) -> bool {
    matches!(
        profile_idc,
        100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
    )
}

/// Read + discard a scaling list of `size` (H.264 §7.3.2.1.1.1) so the bit
/// position stays correct; the values themselves aren't needed yet.
fn skip_scaling_list(r: &mut BitReader, size: u32) -> Result<(), &'static str> {
    let mut last = 8i32;
    let mut next = 8i32;
    for _ in 0..size {
        if next != 0 {
            let delta = r.se()?;
            next = (last + delta + 256) % 256;
        }
        last = if next == 0 { last } else { next };
    }
    Ok(())
}

/// Parse an SPS from its RBSP (the NAL body after the header, unescaped).
pub fn parse_sps(rbsp: &[u8]) -> Result<Sps, &'static str> {
    let mut r = BitReader::new(rbsp);
    let mut s = Sps::default();
    s.profile_idc = r.u(8)? as u8;
    let _constraint_and_reserved = r.u(8)?;
    s.level_idc = r.u(8)? as u8;
    s.id = r.ue()?;
    s.chroma_format_idc = 1; // default 4:2:0
    s.bit_depth_luma = 8;
    s.bit_depth_chroma = 8;
    if is_high_profile(s.profile_idc) {
        s.chroma_format_idc = r.ue()?;
        if s.chroma_format_idc == 3 {
            let _separate_colour_plane = r.flag()?;
        }
        s.bit_depth_luma = r.ue()? + 8;
        s.bit_depth_chroma = r.ue()? + 8;
        let _qpprime_y_zero_bypass = r.flag()?;
        if r.flag()? {
            // seq_scaling_matrix_present_flag: 8 or 12 lists.
            let count = if s.chroma_format_idc != 3 { 8 } else { 12 };
            for i in 0..count {
                if r.flag()? {
                    let size = if i < 6 { 16 } else { 64 };
                    skip_scaling_list(&mut r, size)?;
                }
            }
        }
    }
    s.log2_max_frame_num = r.ue()? + 4;
    s.pic_order_cnt_type = r.ue()?;
    if s.pic_order_cnt_type == 0 {
        s.log2_max_poc_lsb = r.ue()? + 4;
    } else if s.pic_order_cnt_type == 1 {
        let _delta_pic_order_always_zero = r.flag()?;
        let _offset_for_non_ref_pic = r.se()?;
        let _offset_for_top_to_bottom = r.se()?;
        let cycle_len = r.ue()?;
        if cycle_len > 255 {
            return Err("h264 sps: poc cycle too long");
        }
        for _ in 0..cycle_len {
            let _offset = r.se()?;
        }
    }
    s.max_num_ref_frames = r.ue()?;
    let _gaps_allowed = r.flag()?;
    s.pic_width_in_mbs = r.ue()? + 1;
    s.pic_height_in_map_units = r.ue()? + 1;
    s.frame_mbs_only_flag = r.flag()?;
    if !s.frame_mbs_only_flag {
        s.mb_adaptive_frame_field = r.flag()?;
    }
    s.direct_8x8_inference = r.flag()?;
    if r.flag()? {
        // frame_cropping_flag
        s.crop_left = r.ue()?;
        s.crop_right = r.ue()?;
        s.crop_top = r.ue()?;
        s.crop_bottom = r.ue()?;
    }
    // vui_parameters_present_flag + VUI are ignored (not needed for geometry).
    if s.pic_width_in_mbs == 0 || s.pic_height_in_map_units == 0 {
        return Err("h264 sps: zero picture size");
    }
    Ok(s)
}

/// A parsed picture parameter set — the fields the slice layer needs.
#[derive(Clone, Debug, Default)]
pub struct Pps {
    pub id: u32,
    pub sps_id: u32,
    /// false = CAVLC (baseline/what we target), true = CABAC.
    pub entropy_coding_mode: bool,
    pub bottom_field_pic_order_present: bool,
    pub num_slice_groups: u32,
    pub num_ref_idx_l0_default: u32,
    pub num_ref_idx_l1_default: u32,
    pub weighted_pred: bool,
    pub weighted_bipred_idc: u32,
    pub pic_init_qp: i32,
    pub pic_init_qs: i32,
    pub chroma_qp_index_offset: i32,
    pub deblocking_filter_control_present: bool,
    pub constrained_intra_pred: bool,
    pub redundant_pic_cnt_present: bool,
    /// High-profile extension: adaptive 4x4/8x8 transform per macroblock.
    pub transform_8x8_mode: bool,
    /// High-profile extension: separate Cr QP offset (Cb uses
    /// `chroma_qp_index_offset`). Set to the Cb offset when absent.
    pub second_chroma_qp_index_offset: i32,
    /// PPS scaling matrices present — not supported; the decoder refuses.
    pub scaling_matrix_present: bool,
}

/// Parse a PPS from its RBSP. Only the pre-`slice_group` fields are read in
/// full; multi-slice-group maps (rare, and not in baseline main use) are not
/// needed downstream yet, so parsing stops cleanly after the core fields.
pub fn parse_pps(rbsp: &[u8]) -> Result<Pps, &'static str> {
    let mut r = BitReader::new(rbsp);
    let mut p = Pps::default();
    p.id = r.ue()?;
    p.sps_id = r.ue()?;
    p.entropy_coding_mode = r.flag()?;
    p.bottom_field_pic_order_present = r.flag()?;
    p.num_slice_groups = r.ue()? + 1;
    if p.num_slice_groups > 1 {
        // Slice-group map syntax — parsed later if we ever support FMO. For now
        // record the count and return; downstream refuses >1 group.
        return Ok(p);
    }
    p.num_ref_idx_l0_default = r.ue()? + 1;
    p.num_ref_idx_l1_default = r.ue()? + 1;
    p.weighted_pred = r.flag()?;
    p.weighted_bipred_idc = r.u(2)?;
    p.pic_init_qp = r.se()? + 26;
    p.pic_init_qs = r.se()? + 26;
    p.chroma_qp_index_offset = r.se()?;
    p.deblocking_filter_control_present = r.flag()?;
    p.constrained_intra_pred = r.flag()?;
    p.redundant_pic_cnt_present = r.flag()?;
    // High-profile PPS extension (present iff more RBSP data follows).
    p.second_chroma_qp_index_offset = p.chroma_qp_index_offset;
    if r.more_rbsp_data() {
        p.transform_8x8_mode = r.flag()?;
        p.scaling_matrix_present = r.flag()?;
        if p.scaling_matrix_present {
            return Ok(p); // refused downstream; don't parse the lists
        }
        p.second_chroma_qp_index_offset = r.se()?;
    }
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn annexb_splits_on_start_codes() {
        // Two NALs: a 4-byte start code then type 7 (SPS), a 3-byte start code
        // then type 8 (PPS).
        let data = [
            0x00, 0x00, 0x00, 0x01, 0x67, 0xaa, 0xbb, // SPS-ish (0x67 = ref_idc 3, type 7)
            0x00, 0x00, 0x01, 0x68, 0xcc, // PPS-ish (0x68 = type 8)
        ];
        let nals = split_annexb(&data);
        assert_eq!(nals.len(), 2);
        assert_eq!(nals[0].kind, NalType::Sps);
        assert_eq!(nals[0].ref_idc, 3);
        assert_eq!(nals[0].payload, &[0x67, 0xaa, 0xbb]);
        assert_eq!(nals[1].kind, NalType::Pps);
        assert_eq!(nals[1].payload, &[0x68, 0xcc]);
    }

    #[test_case]
    fn avcc_splits_on_length_prefix() {
        // length_size=4. NAL1 len 3 (type 5 IDR), NAL2 len 2 (type 1).
        let data = [
            0x00, 0x00, 0x00, 0x03, 0x65, 0x11, 0x22, // IDR slice
            0x00, 0x00, 0x00, 0x02, 0x41, 0x33, // non-IDR slice
        ];
        let nals = split_avcc(&data, 4);
        assert_eq!(nals.len(), 2);
        assert_eq!(nals[0].kind, NalType::SliceIdr);
        assert!(nals[0].kind.is_slice());
        assert_eq!(nals[1].kind, NalType::Slice);
    }

    // Real x264 baseline output for a 176x144 (QCIF) clip. Regenerate with:
    //   x264 --profile baseline --tune zerolatency --frames 3 \
    //        --input-res 176x144 -o out.264 in.yuv
    // then strip start code + NAL header (0x67 SPS / 0x68 PPS) and unescape.

    #[test_case]
    fn sps_baseline_dimensions() {
        let rbsp = [0x42, 0xc0, 0x0b, 0xd9, 0x02, 0xc4, 0xe8, 0x40, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x0c, 0xa3, 0xc5, 0x0a, 0x92];
        let sps = parse_sps(&rbsp).unwrap();
        assert_eq!(sps.profile_idc, 66); // baseline
        assert_eq!(sps.level_idc, 11);
        assert_eq!(sps.chroma_format_idc, 1); // 4:2:0
        assert_eq!(sps.pic_width_in_mbs, 11); // 11*16 = 176
        assert_eq!(sps.pic_height_in_map_units, 9); // 9*16 = 144
        assert_eq!(sps.width(), 176);
        assert_eq!(sps.height(), 144);
        assert_eq!(sps.pic_order_cnt_type, 2);
        assert_eq!(sps.max_num_ref_frames, 3);
        assert!(sps.frame_mbs_only_flag);
    }

    #[test_case]
    fn pps_reports_entropy_mode() {
        let rbsp = [0xcb, 0x83, 0xcb, 0x20];
        let pps = parse_pps(&rbsp).unwrap();
        assert_eq!(pps.id, 0);
        assert_eq!(pps.sps_id, 0);
        assert!(!pps.entropy_coding_mode, "baseline stream is CAVLC, not CABAC");
        assert_eq!(pps.num_slice_groups, 1);
        assert_eq!(pps.num_ref_idx_l0_default, 3);
        assert_eq!(pps.pic_init_qp, 23);
        assert_eq!(pps.chroma_qp_index_offset, -2);
        assert!(pps.deblocking_filter_control_present);
        assert!(!pps.constrained_intra_pred);
    }
}
