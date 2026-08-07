//! VP9 inter prediction: reference-frame selection contexts, the motion-vector
//! reference search, motion-vector decoding, and motion compensation.
//!
//! Three properties of this layer are worth stating because they are where the
//! bugs live:
//!
//! * **Almost everything here feeds the arithmetic decoder.** The reference
//!   contexts, the inter-mode context and the MV reference list all choose
//!   *probabilities*, so an error is not a wrong pixel — it changes how many
//!   bits the next symbol consumes and the rest of the tile decodes as noise.
//!   That makes them loud, which is the one mercy.
//! * **A motion vector is in 1/8-pel luma units, and chroma reuses it
//!   unscaled.** Compensation works in 1/16-pel, so luma doubles the vector and
//!   chroma does not — because a chroma pixel is two luma pixels wide, the same
//!   number already means 1/16 of a chroma pixel. Scaling both the same way is
//!   the classic way to get chroma at half the motion.
//! * **Reference reads are edge-clamped, not bounds-checked.** A motion vector
//!   may legitimately point outside the reference frame; the reference
//!   implementation extends the frame's border, and clamping the read
//!   coordinate is exactly equivalent without the copy.

use super::tables;
use super::tile::{FrameDecodeState, ModeInfo, Plane};
use alloc::vec::Vec;

/// Reference-frame slots: `INTRA_FRAME` is 0, so the three inter references are
/// 1..=3 and 0 doubles as "this block is intra".
pub const INTRA_FRAME: i8 = 0;
pub const LAST_FRAME: i8 = 1;
pub const GOLDEN_FRAME: i8 = 2;
pub const ALTREF_FRAME: i8 = 3;
pub const NO_REF_FRAME: i8 = -1;

/// Inter prediction modes, offset past the ten intra ones.
pub const NEARESTMV: u8 = 10;
pub const NEARMV: u8 = 11;
pub const ZEROMV: u8 = 12;
pub const NEWMV: u8 = 13;

/// Motion-vector reference candidates kept per block.
pub const MAX_MV_REF_CANDIDATES: usize = 2;
/// Neighbour positions searched for reference motion vectors.
const MVREF_NEIGHBOURS: usize = 8;
/// Slack (in 1/8 pel) allowed outside the frame when clamping a reference MV.
const MV_BORDER: i32 = 16 * 8;
/// libvpx `SWITCHABLE_FILTERS` — also the "no filter" marker for intra blocks.
pub const SWITCHABLE_FILTERS: u8 = 3;

/// A motion vector in 1/8-pel luma units.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Mv {
    pub row: i16,
    pub col: i16,
}

impl Mv {
    fn is_valid(self) -> bool {
        // libvpx `is_mv_valid`: within ±(1 << 14) - 1.
        const MAX: i32 = (1 << 14) - 1;
        (self.row as i32).abs() < MAX && (self.col as i32).abs() < MAX
    }
}

/// One decoded reference picture.
pub struct RefFrame {
    pub planes: [Plane; 3],
    /// Motion vectors and reference frames per 8x8, for the temporal candidate.
    pub mvs: Vec<([Mv; 2], [i8; 2])>,
    pub mi_cols: usize,
    pub mi_rows: usize,
}

// ---------------------------------------------------------------------------
// Prediction contexts (libvpx `vp9_pred_common.c`)
// ---------------------------------------------------------------------------

fn is_inter(mi: &ModeInfo) -> bool {
    mi.ref_frame[0] > INTRA_FRAME
}

fn has_second_ref(mi: &ModeInfo) -> bool {
    mi.ref_frame[1] > INTRA_FRAME
}

/// `get_intra_inter_context`.
pub fn intra_inter_context(above: Option<&ModeInfo>, left: Option<&ModeInfo>) -> usize {
    match (above, left) {
        (Some(a), Some(l)) => {
            let (ai, li) = (!is_inter(a), !is_inter(l));
            if li && ai {
                3
            } else {
                (li || ai) as usize
            }
        }
        (Some(m), None) | (None, Some(m)) => 2 * (!is_inter(m)) as usize,
        (None, None) => 0,
    }
}

/// `vp9_get_reference_mode_context` — which of the five compound-vs-single
/// contexts this block sits in.
pub fn reference_mode_context(
    above: Option<&ModeInfo>,
    left: Option<&ModeInfo>,
    comp_fixed_ref: i8,
) -> usize {
    match (above, left) {
        (Some(a), Some(l)) => {
            if !has_second_ref(a) && !has_second_ref(l) {
                ((a.ref_frame[0] == comp_fixed_ref) ^ (l.ref_frame[0] == comp_fixed_ref)) as usize
            } else if !has_second_ref(a) {
                2 + (a.ref_frame[0] == comp_fixed_ref || !is_inter(a)) as usize
            } else if !has_second_ref(l) {
                2 + (l.ref_frame[0] == comp_fixed_ref || !is_inter(l)) as usize
            } else {
                4
            }
        }
        (Some(m), None) | (None, Some(m)) => {
            if !has_second_ref(m) {
                (m.ref_frame[0] == comp_fixed_ref) as usize
            } else {
                3
            }
        }
        (None, None) => 1,
    }
}

/// `vp9_get_pred_context_single_ref_p1` — is the (single) reference LAST?
pub fn single_ref_p1_context(above: Option<&ModeInfo>, left: Option<&ModeInfo>) -> usize {
    match (above, left) {
        (Some(a), Some(l)) => {
            let (ai, li) = (!is_inter(a), !is_inter(l));
            if ai && li {
                2
            } else if ai || li {
                let e = if ai { l } else { a };
                if !has_second_ref(e) {
                    4 * (e.ref_frame[0] == LAST_FRAME) as usize
                } else {
                    1 + (e.ref_frame[0] == LAST_FRAME || e.ref_frame[1] == LAST_FRAME) as usize
                }
            } else {
                let (a2, l2) = (has_second_ref(a), has_second_ref(l));
                let (a0, a1) = (a.ref_frame[0], a.ref_frame[1]);
                let (l0, l1) = (l.ref_frame[0], l.ref_frame[1]);
                if a2 && l2 {
                    1 + (a0 == LAST_FRAME || a1 == LAST_FRAME || l0 == LAST_FRAME || l1 == LAST_FRAME) as usize
                } else if a2 || l2 {
                    let rfs = if !a2 { a0 } else { l0 };
                    let crf1 = if a2 { a0 } else { l0 };
                    let crf2 = if a2 { a1 } else { l1 };
                    if rfs == LAST_FRAME {
                        3 + (crf1 == LAST_FRAME || crf2 == LAST_FRAME) as usize
                    } else {
                        (crf1 == LAST_FRAME || crf2 == LAST_FRAME) as usize
                    }
                } else {
                    2 * (a0 == LAST_FRAME) as usize + 2 * (l0 == LAST_FRAME) as usize
                }
            }
        }
        (Some(m), None) | (None, Some(m)) => {
            if !is_inter(m) {
                2
            } else if !has_second_ref(m) {
                4 * (m.ref_frame[0] == LAST_FRAME) as usize
            } else {
                1 + (m.ref_frame[0] == LAST_FRAME || m.ref_frame[1] == LAST_FRAME) as usize
            }
        }
        (None, None) => 2,
    }
}

/// `vp9_get_pred_context_single_ref_p2` — GOLDEN vs ALTREF, given not LAST.
pub fn single_ref_p2_context(above: Option<&ModeInfo>, left: Option<&ModeInfo>) -> usize {
    match (above, left) {
        (Some(a), Some(l)) => {
            let (ai, li) = (!is_inter(a), !is_inter(l));
            if ai && li {
                2
            } else if ai || li {
                let e = if ai { l } else { a };
                if !has_second_ref(e) {
                    if e.ref_frame[0] == LAST_FRAME {
                        3
                    } else {
                        4 * (e.ref_frame[0] == GOLDEN_FRAME) as usize
                    }
                } else {
                    1 + 2 * (e.ref_frame[0] == GOLDEN_FRAME || e.ref_frame[1] == GOLDEN_FRAME) as usize
                }
            } else {
                let (a2, l2) = (has_second_ref(a), has_second_ref(l));
                let (a0, a1) = (a.ref_frame[0], a.ref_frame[1]);
                let (l0, l1) = (l.ref_frame[0], l.ref_frame[1]);
                if a2 && l2 {
                    if a0 == l0 && a1 == l1 {
                        3 * (a0 == GOLDEN_FRAME || a1 == GOLDEN_FRAME || l0 == GOLDEN_FRAME || l1 == GOLDEN_FRAME) as usize
                    } else {
                        2
                    }
                } else if a2 || l2 {
                    let rfs = if !a2 { a0 } else { l0 };
                    let crf1 = if a2 { a0 } else { l0 };
                    let crf2 = if a2 { a1 } else { l1 };
                    if rfs == GOLDEN_FRAME {
                        3 + (crf1 == GOLDEN_FRAME || crf2 == GOLDEN_FRAME) as usize
                    } else if rfs == ALTREF_FRAME {
                        (crf1 == GOLDEN_FRAME || crf2 == GOLDEN_FRAME) as usize
                    } else {
                        1 + 2 * (crf1 == GOLDEN_FRAME || crf2 == GOLDEN_FRAME) as usize
                    }
                } else if a0 == LAST_FRAME && l0 == LAST_FRAME {
                    3
                } else if a0 == LAST_FRAME || l0 == LAST_FRAME {
                    let e0 = if a0 == LAST_FRAME { l0 } else { a0 };
                    4 * (e0 == GOLDEN_FRAME) as usize
                } else {
                    2 * (a0 == GOLDEN_FRAME) as usize + 2 * (l0 == GOLDEN_FRAME) as usize
                }
            }
        }
        (Some(m), None) | (None, Some(m)) => {
            if !is_inter(m) || (m.ref_frame[0] == LAST_FRAME && !has_second_ref(m)) {
                2
            } else if !has_second_ref(m) {
                4 * (m.ref_frame[0] == GOLDEN_FRAME) as usize
            } else {
                3 * (m.ref_frame[0] == GOLDEN_FRAME || m.ref_frame[1] == GOLDEN_FRAME) as usize
            }
        }
        (None, None) => 2,
    }
}

/// `vp9_get_pred_context_comp_ref_p` — which of the two variable references a
/// compound block uses.
pub fn comp_ref_context(
    above: Option<&ModeInfo>,
    left: Option<&ModeInfo>,
    comp_fixed_ref: i8,
    comp_var_ref: [i8; 2],
    fix_ref_idx: usize,
) -> usize {
    let var_ref_idx = 1 - fix_ref_idx;
    match (above, left) {
        (Some(a), Some(l)) => {
            let (ai, li) = (!is_inter(a), !is_inter(l));
            if ai && li {
                2
            } else if ai || li {
                let e = if ai { l } else { a };
                if !has_second_ref(e) {
                    1 + 2 * (e.ref_frame[0] != comp_var_ref[1]) as usize
                } else {
                    1 + 2 * (e.ref_frame[var_ref_idx] != comp_var_ref[1]) as usize
                }
            } else {
                let (a_sg, l_sg) = (!has_second_ref(a), !has_second_ref(l));
                let vrfa = if a_sg { a.ref_frame[0] } else { a.ref_frame[var_ref_idx] };
                let vrfl = if l_sg { l.ref_frame[0] } else { l.ref_frame[var_ref_idx] };
                if vrfa == vrfl && comp_var_ref[1] == vrfa {
                    0
                } else if l_sg && a_sg {
                    if (vrfa == comp_fixed_ref && vrfl == comp_var_ref[0])
                        || (vrfl == comp_fixed_ref && vrfa == comp_var_ref[0])
                    {
                        4
                    } else if vrfa == vrfl {
                        3
                    } else {
                        1
                    }
                } else if l_sg || a_sg {
                    let vrfc = if l_sg { vrfa } else { vrfl };
                    let rfs = if a_sg { vrfa } else { vrfl };
                    if vrfc == comp_var_ref[1] && rfs != comp_var_ref[1] {
                        1
                    } else if rfs == comp_var_ref[1] && vrfc != comp_var_ref[1] {
                        2
                    } else {
                        4
                    }
                } else if vrfa == vrfl {
                    4
                } else {
                    2
                }
            }
        }
        (Some(m), None) | (None, Some(m)) => {
            if !is_inter(m) {
                2
            } else if has_second_ref(m) {
                4 * (m.ref_frame[var_ref_idx] != comp_var_ref[1]) as usize
            } else {
                3 * (m.ref_frame[0] != comp_var_ref[1]) as usize
            }
        }
        (None, None) => 2,
    }
}

/// `get_pred_context_switchable_interp`.
pub fn interp_filter_context(above: Option<&ModeInfo>, left: Option<&ModeInfo>) -> usize {
    let lt = left.map(|m| m.interp_filter).unwrap_or(SWITCHABLE_FILTERS);
    let at = above.map(|m| m.interp_filter).unwrap_or(SWITCHABLE_FILTERS);
    if lt == at {
        lt as usize
    } else if lt == SWITCHABLE_FILTERS {
        at as usize
    } else if at == SWITCHABLE_FILTERS {
        lt as usize
    } else {
        SWITCHABLE_FILTERS as usize
    }
}

// ---------------------------------------------------------------------------
// Motion-vector reference search (libvpx `dec_find_mv_refs`)
// ---------------------------------------------------------------------------

/// `get_mode_context`: the inter-mode context from the two nearest neighbours.
pub fn mode_context(s: &FrameDecodeState, bsize: u8, mi_row: usize, mi_col: usize) -> usize {
    let mut counter = 0usize;
    for i in 0..2 {
        let (dr, dc) = (
            tables::MV_REF_BLOCKS[bsize as usize][i][0] as isize,
            tables::MV_REF_BLOCKS[bsize as usize][i][1] as isize,
        );
        if let Some(m) = neighbour(s, mi_row, mi_col, dr, dc) {
            counter += tables::MODE_2_COUNTER[m.y_mode.min(13) as usize] as usize;
        }
    }
    tables::COUNTER_TO_CONTEXT[counter.min(18)] as usize
}

/// `is_inside` + the mode-info fetch. Tile-relative on the left/right, frame
/// relative on the top — the same asymmetry the intra neighbours have.
fn neighbour<'a>(
    s: &'a FrameDecodeState,
    mi_row: usize,
    mi_col: usize,
    dr: isize,
    dc: isize,
) -> Option<&'a ModeInfo> {
    let r = mi_row as isize + dr;
    let c = mi_col as isize + dc;
    if r < 0
        || c < s.tile_mi_col_start() as isize
        || r >= s.mi_rows as isize
        || c >= s.tile_mi_col_end() as isize
    {
        return None;
    }
    Some(&s.mi[r as usize * s.mi_cols + c as usize])
}

/// `clamp_mv_ref` — keep a candidate within a border of the current block.
fn clamp_mv_ref(mv: Mv, bw8: usize, bh8: usize, mi_row: usize, mi_col: usize, s: &FrameDecodeState) -> Mv {
    // libvpx works in 1/8-pel distances to the frame edges.
    let to_left = -((mi_col as i32 * 8) * 8);
    let to_top = -((mi_row as i32 * 8) * 8);
    let to_right = ((s.mi_cols - mi_col - bw8) as i32 * 8) * 8;
    let to_bottom = ((s.mi_rows - mi_row - bh8) as i32 * 8) * 8;
    Mv {
        col: (mv.col as i32).clamp(to_left - MV_BORDER, to_right + MV_BORDER) as i16,
        row: (mv.row as i32).clamp(to_top - MV_BORDER, to_bottom + MV_BORDER) as i16,
    }
}

/// libvpx `ADD_MV_REF_LIST_EB`: add a candidate and say whether to stop.
///
/// Two asymmetries that are easy to smooth over and must not be: the duplicate
/// test compares only against **entry 0**, not the whole list, and adding a
/// *second* entry always stops (the list holds two) whereas adding the first
/// stops only when the caller asked to break early.
fn add_mv(list: &mut Vec<Mv>, mv: Mv, early_break: bool) -> bool {
    if list.is_empty() {
        list.push(mv);
        return early_break;
    }
    if list[0] != mv {
        list.push(mv);
        return true;
    }
    false
}

/// libvpx `dec_find_mv_refs`.
///
/// The **temporal** candidate is the previous *decoded* frame's motion vector
/// at the same 8x8 position, and it is consulted in two places: once between
/// the spatial passes, and once at the end with the sign flipped when the two
/// references point opposite ways in time. It is only available from the third
/// frame of a sequence onward (`use_prev_frame_mvs` requires the previous frame
/// to be a shown, same-sized, non-intra, non-error-resilient picture) — which
/// is why a decoder without it gets frames 0 and 1 bit-exact and then
/// desynchronises exactly at frame 2.
#[allow(clippy::too_many_arguments)]
pub fn find_mv_refs(
    s: &FrameDecodeState,
    bsize: u8,
    ref_frame: i8,
    mode: u8,
    mi_row: usize,
    mi_col: usize,
    block: isize,
    sign_bias: &[bool; 4],
    prev: Option<&([Mv; 2], [i8; 2])>,
) -> Vec<Mv> {
    let mut list: Vec<Mv> = Vec::with_capacity(2);
    let early_break = mode != NEARMV;
    let mut different_ref_found = false;
    let search = &tables::MV_REF_BLOCKS[bsize as usize];

    let mut i = 0usize;
    if block >= 0 {
        // Sub-8x8: the two nearest neighbours contribute their *sub-block* MVs.
        while i < 2 {
            let (dr, dc) = (search[i][0] as isize, search[i][1] as isize);
            if let Some(c) = neighbour(s, mi_row, mi_col, dr, dc) {
                different_ref_found = true;
                let which = if c.ref_frame[0] == ref_frame {
                    Some(0)
                } else if c.ref_frame[1] == ref_frame {
                    Some(1)
                } else {
                    None
                };
                if let Some(w) = which {
                    let mv = sub_block_mv(c, w, dc, block);
                    if add_mv(&mut list, mv, early_break) {
                        return finish(list, bsize, mi_row, mi_col, s, mode);
                    }
                }
            }
            i += 1;
        }
    }
    while i < MVREF_NEIGHBOURS {
        let (dr, dc) = (search[i][0] as isize, search[i][1] as isize);
        if let Some(c) = neighbour(s, mi_row, mi_col, dr, dc) {
            different_ref_found = true;
            let which = if c.ref_frame[0] == ref_frame {
                Some(0)
            } else if c.ref_frame[1] == ref_frame {
                Some(1)
            } else {
                None
            };
            if let Some(w) = which {
                if add_mv(&mut list, c.mv[w], early_break) {
                    return finish(list, bsize, mi_row, mi_col, s, mode);
                }
            }
        }
        i += 1;
    }

    // The temporal candidate, matching this reference frame.
    if let Some((pmv, prf)) = prev {
        let which = if prf[0] == ref_frame {
            Some(0)
        } else if prf[1] == ref_frame {
            Some(1)
        } else {
            None
        };
        if let Some(w) = which {
            if add_mv(&mut list, pmv[w], early_break) {
                return finish(list, bsize, mi_row, mi_col, s, mode);
            }
        }
    }

    // Second pass: candidates from a *different* reference frame, with their
    // sign flipped when that reference points the other way in time.
    if different_ref_found {
        for i in 0..MVREF_NEIGHBOURS {
            let (dr, dc) = (search[i][0] as isize, search[i][1] as isize);
            if let Some(c) = neighbour(s, mi_row, mi_col, dr, dc) {
                for w in 0..2 {
                    let rf = c.ref_frame[w];
                    if rf <= INTRA_FRAME || rf == ref_frame {
                        continue;
                    }
                    if w == 1 && c.mv[1] == c.mv[0] {
                        continue;
                    }
                    let mut mv = c.mv[w];
                    if sign_bias[rf as usize] != sign_bias[ref_frame as usize] {
                        mv.row = -mv.row;
                        mv.col = -mv.col;
                    }
                    if add_mv(&mut list, mv, early_break) {
                        return finish(list, bsize, mi_row, mi_col, s, mode);
                    }
                }
            }
        }
    }
    // …and the temporal candidate from a different reference, likewise flipped.
    if let Some((pmv, prf)) = prev {
        for w in 0..2 {
            let rf = prf[w];
            if rf <= INTRA_FRAME || rf == ref_frame {
                continue;
            }
            if w == 1 && pmv[1] == pmv[0] {
                continue;
            }
            let mut mv = pmv[w];
            if sign_bias[rf as usize] != sign_bias[ref_frame as usize] {
                mv.row = -mv.row;
                mv.col = -mv.col;
            }
            if add_mv(&mut list, mv, early_break) {
                return finish(list, bsize, mi_row, mi_col, s, mode);
            }
        }
    }
    finish(list, bsize, mi_row, mi_col, s, mode)
}

fn finish(
    mut list: Vec<Mv>,
    bsize: u8,
    mi_row: usize,
    mi_col: usize,
    s: &FrameDecodeState,
    mode: u8,
) -> Vec<Mv> {
    let want = if mode == NEARMV { MAX_MV_REF_CANDIDATES } else { 1 };
    while list.len() < want {
        list.push(Mv::default());
    }
    list.truncate(want);
    let bw8 = tables::NUM_8X8_W[bsize as usize] as usize;
    let bh8 = tables::NUM_8X8_H[bsize as usize] as usize;
    for mv in list.iter_mut() {
        *mv = clamp_mv_ref(*mv, bw8, bh8, mi_row, mi_col, s);
    }
    list
}

/// `get_sub_block_mv`: a sub-8x8 neighbour contributes the motion vector of the
/// sub-block **adjacent to this one**, not its whole-block vector.
///
/// Which sub-block that is comes from `idx_n_column_to_subblock`, indexed by
/// this block's index and by whether the neighbour is above (`search_col == 0`)
/// or to the left. It is not derivable by inspection — the table is
/// `{1,2},{1,3},{3,2},{3,3}` — and a plausible hand-derived rule differs from
/// it on exactly the cases a deeply-searched encode produces.
fn sub_block_mv(c: &ModeInfo, which: usize, search_col: isize, block: isize) -> Mv {
    if block < 0 || c.block_size >= 3 {
        return c.mv[which];
    }
    let idx = tables::IDX_N_COLUMN_TO_SUBBLOCK[(block as usize).min(3)][(search_col == 0) as usize];
    c.sub_mv[which][(idx as usize).min(3)]
}

/// `lower_mv_precision`: without high-precision motion vectors an odd component
/// is nudged toward zero.
pub fn lower_mv_precision(mv: &mut Mv, allow_hp: bool) {
    let use_hp = allow_hp && use_mv_hp(*mv);
    if !use_hp {
        if mv.row & 1 != 0 {
            mv.row += if mv.row > 0 { -1 } else { 1 };
        }
        if mv.col & 1 != 0 {
            mv.col += if mv.col > 0 { -1 } else { 1 };
        }
    }
}

/// `use_mv_hp`: high precision is only used for small vectors.
pub fn use_mv_hp(reference: Mv) -> bool {
    const COMPANDED_MVREF_THRESH: i32 = 8;
    (reference.row as i32).abs() < COMPANDED_MVREF_THRESH * 8
        && (reference.col as i32).abs() < COMPANDED_MVREF_THRESH * 8
}

// ---------------------------------------------------------------------------
// Motion compensation
// ---------------------------------------------------------------------------

/// Taps in the sub-pel filters.
const SUBPEL_TAPS: usize = 8;
const FILTER_BITS: u32 = 7;

/// Convolve an 8-tap filter over a reference window into `dst`.
///
/// Reference reads are **clamped** to the plane, which is exactly libvpx's
/// border extension without materialising the border: a motion vector may point
/// outside the reference and the defined behaviour is edge replication.
#[allow(clippy::too_many_arguments)]
pub fn convolve8(
    src: &Plane,
    x0: i32,
    y0: i32,
    subpel_x: usize,
    subpel_y: usize,
    filter: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    w: usize,
    h: usize,
    average: bool,
) {
    let kx = &tables::SUBPEL_FILTERS[filter][subpel_x];
    let ky = &tables::SUBPEL_FILTERS[filter][subpel_y];
    // Horizontal pass into an intermediate that is `SUBPEL_TAPS - 1` rows
    // taller, because the vertical pass reads 3 above and 4 below.
    let ih = h + SUBPEL_TAPS - 1;
    let mut tmp = [0u8; (64 + SUBPEL_TAPS) * 64];
    let tstride = w.max(1);
    for yy in 0..ih {
        let sy = y0 + yy as i32 - (SUBPEL_TAPS as i32 / 2 - 1);
        for xx in 0..w {
            let mut sum = 0i32;
            for k in 0..SUBPEL_TAPS {
                let sx = x0 + xx as i32 + k as i32 - (SUBPEL_TAPS as i32 / 2 - 1);
                sum += src.at_clamped(sx, sy) as i32 * kx[k] as i32;
            }
            tmp[yy * tstride + xx] = clip_pixel(round_shift(sum, FILTER_BITS));
        }
    }
    for yy in 0..h {
        for xx in 0..w {
            let mut sum = 0i32;
            for k in 0..SUBPEL_TAPS {
                sum += tmp[(yy + k) * tstride + xx] as i32 * ky[k] as i32;
            }
            let v = clip_pixel(round_shift(sum, FILTER_BITS));
            let p = dst_off + yy * dst_stride + xx;
            if p < dst.len() {
                // Compound prediction averages the two references, rounding up.
                dst[p] = if average { ((dst[p] as u32 + v as u32 + 1) >> 1) as u8 } else { v };
            }
        }
    }
}

#[inline(always)]
fn round_shift(v: i32, n: u32) -> i32 {
    (v + (1 << (n - 1))) >> n
}

#[inline(always)]
fn clip_pixel(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mi(rf0: i8, rf1: i8, filt: u8) -> ModeInfo {
        ModeInfo { ref_frame: [rf0, rf1], interp_filter: filt, ..Default::default() }
    }

    #[test_case]
    fn intra_inter_context_counts_intra_neighbours() {
        let intra = mi(INTRA_FRAME, NO_REF_FRAME, SWITCHABLE_FILTERS);
        let inter = mi(LAST_FRAME, NO_REF_FRAME, 0);
        assert_eq!(intra_inter_context(Some(&intra), Some(&intra)), 3);
        assert_eq!(intra_inter_context(Some(&intra), Some(&inter)), 1);
        assert_eq!(intra_inter_context(Some(&inter), Some(&inter)), 0);
        assert_eq!(intra_inter_context(Some(&intra), None), 2);
        assert_eq!(intra_inter_context(Some(&inter), None), 0);
        assert_eq!(intra_inter_context(None, None), 0);
    }

    #[test_case]
    fn single_ref_contexts_stay_in_range() {
        // Exhaustive over every neighbour combination: these are long chains of
        // branches and the only cheap invariant is that they never index a
        // probability table out of range.
        let frames = [INTRA_FRAME, LAST_FRAME, GOLDEN_FRAME, ALTREF_FRAME];
        for &a0 in &frames {
            for &a1 in &[NO_REF_FRAME, LAST_FRAME, GOLDEN_FRAME, ALTREF_FRAME] {
                for &l0 in &frames {
                    for &l1 in &[NO_REF_FRAME, LAST_FRAME, GOLDEN_FRAME, ALTREF_FRAME] {
                        let a = mi(a0, a1, 0);
                        let l = mi(l0, l1, 0);
                        for opt in [
                            (Some(&a), Some(&l)),
                            (Some(&a), None),
                            (None, Some(&l)),
                            (None, None),
                        ] {
                            assert!(single_ref_p1_context(opt.0, opt.1) < 5);
                            assert!(single_ref_p2_context(opt.0, opt.1) < 5);
                            assert!(reference_mode_context(opt.0, opt.1, ALTREF_FRAME) < 5);
                            assert!(comp_ref_context(opt.0, opt.1, ALTREF_FRAME, [LAST_FRAME, GOLDEN_FRAME], 1) < 5);
                            assert!(intra_inter_context(opt.0, opt.1) < 4);
                            assert!(interp_filter_context(opt.0, opt.1) < 4);
                        }
                    }
                }
            }
        }
    }

    #[test_case]
    fn interp_filter_context_agrees_or_falls_through() {
        let a = mi(LAST_FRAME, NO_REF_FRAME, 1);
        let l = mi(LAST_FRAME, NO_REF_FRAME, 1);
        assert_eq!(interp_filter_context(Some(&a), Some(&l)), 1);
        let l2 = mi(LAST_FRAME, NO_REF_FRAME, 2);
        assert_eq!(interp_filter_context(Some(&a), Some(&l2)), 3, "disagreement → SWITCHABLE");
        // An intra neighbour reports SWITCHABLE, so the other side wins.
        let intra = mi(INTRA_FRAME, NO_REF_FRAME, SWITCHABLE_FILTERS);
        assert_eq!(interp_filter_context(Some(&a), Some(&intra)), 1);
    }

    #[test_case]
    fn high_precision_is_only_used_for_small_vectors() {
        assert!(use_mv_hp(Mv { row: 0, col: 0 }));
        assert!(use_mv_hp(Mv { row: 63, col: -63 }));
        assert!(!use_mv_hp(Mv { row: 64, col: 0 }));
        assert!(!use_mv_hp(Mv { row: 0, col: -64 }));
    }

    #[test_case]
    fn lower_precision_nudges_odd_components_toward_zero() {
        let mut m = Mv { row: 5, col: -5 };
        lower_mv_precision(&mut m, false);
        assert_eq!(m, Mv { row: 4, col: -4 });
        // With hp allowed and a small vector, odd components survive.
        let mut m = Mv { row: 5, col: -5 };
        lower_mv_precision(&mut m, true);
        assert_eq!(m, Mv { row: 5, col: -5 });
        // …but a large vector is lowered even when hp is allowed.
        let mut m = Mv { row: 101, col: 0 };
        lower_mv_precision(&mut m, true);
        assert_eq!(m.row, 100);
    }

    #[test_case]
    fn a_whole_pel_copy_reproduces_the_reference() {
        // Phase 0 of every filter is [0,0,0,128,0,0,0,0], so a zero-subpel
        // convolution must be an exact copy — the cheapest end-to-end check
        // that the filter table and the tap indexing agree.
        for f in 0..4 {
            assert_eq!(tables::SUBPEL_FILTERS[f][0], [0, 0, 0, 128, 0, 0, 0, 0]);
        }
        let mut src = Plane::new_for_test(16, 16);
        for y in 0..16 {
            for x in 0..16 {
                src.data[y * src.stride + x] = (y * 16 + x) as u8;
            }
        }
        let mut dst = alloc::vec![0u8; 8 * 8];
        convolve8(&src, 2, 3, 0, 0, 0, &mut dst, 0, 8, 8, 8, false);
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(dst[y * 8 + x], ((y + 3) * 16 + (x + 2)) as u8, "at {},{}", x, y);
            }
        }
    }

    #[test_case]
    fn reference_reads_clamp_instead_of_wrapping() {
        // A motion vector may point outside the reference; the defined result is
        // edge replication, so a copy from far outside must be the corner pixel
        // rather than a wrapped row or a panic.
        let mut src = Plane::new_for_test(16, 16);
        for p in src.data.iter_mut() {
            *p = 0;
        }
        src.data[0] = 200;
        let mut dst = alloc::vec![0u8; 16];
        convolve8(&src, -100, -100, 0, 0, 0, &mut dst, 0, 4, 4, 4, false);
        assert!(dst.iter().all(|&p| p == 200), "far outside must replicate the corner");
    }
}
