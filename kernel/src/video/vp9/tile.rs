//! VP9 superblock decode: the partition quadtree, per-block mode info, and the
//! residual/reconstruct loop for **intra** frames (keyframes and `intra_only`).
//!
//! VP9's structure is a recursive quadtree over 64x64 superblocks rather than a
//! fixed macroblock grid, and three pieces of state thread through it that are
//! easy to under-model:
//!
//! * **The partition context** is a *bitmask per 8x8 column and row*, not a
//!   per-block value: after decoding a block, bits for sizes larger than it are
//!   set and smaller ones cleared, so the next block's context reads how deeply
//!   its neighbours split. Storing a single size instead works on uniform
//!   content and diverges on detailed content.
//! * **The entropy (non-zero) context** is likewise per 4x4 column/row and is
//!   consumed *and rewritten* by every transform block, including skipped ones
//!   (which clear it). Forgetting the skip case leaves stale non-zero flags and
//!   mis-contexts the next block's coefficients.
//! * **Modes come from neighbours across block boundaries.** A keyframe's Y
//!   mode is coded against `KF_Y_MODE_PROB[above_mode][left_mode]`, where the
//!   neighbour mode is `DC_PRED` when there is no neighbour *or* the neighbour
//!   is inter-coded. That fallback is not a nicety — it is what keeps the
//!   probability lookup in range.

use super::header::{FrameContext, TxMode};
use super::intra::{predict, Edges, IntraMode};
use super::tables;
use super::tokens::decode_coefs;
use super::transform::{ac_quant, dc_quant, inverse_add, iwht4x4_add, tx_type_for_mode, TxSize};
use super::BoolDecoder;
use alloc::vec::Vec;

/// Partition types (libvpx `PARTITION_TYPE`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Partition {
    None = 0,
    Horz = 1,
    Vert = 2,
    Split = 3,
}

/// One plane of the frame being reconstructed.
pub struct Plane {
    pub data: Vec<u8>,
    pub stride: usize,
    pub width: usize,
    pub height: usize,
}

impl Plane {
    /// Read a sample with the coordinate **clamped** to the plane — libvpx's
    /// border extension, without materialising the border. A motion vector may
    /// legitimately point outside a reference frame.
    #[inline(always)]
    pub fn at_clamped(&self, x: i32, y: i32) -> u8 {
        let x = x.clamp(0, self.width as i32 - 1) as usize;
        let y = y.clamp(0, self.height as i32 - 1) as usize;
        self.data[y * self.stride + x]
    }

    /// A blank plane, for tests that need one without a whole decode state.
    #[cfg(test)]
    pub fn new_for_test(width: usize, height: usize) -> Plane {
        Plane::new(width, height)
    }

    fn new(width: usize, height: usize) -> Plane {
        // Aligned up to whole superblocks: prediction reads up to a transform
        // past the visible edge, and the frame is cropped only on output.
        let stride = (width + 63) & !63;
        let h = (height + 63) & !63;
        Plane { data: alloc::vec![0u8; stride * h], stride, width, height }
    }
}

/// Per-8x8-block mode information the neighbour contexts read back.
///
/// `y_mode` holds the **combined** mode space: 0..9 are the intra modes and
/// 10..13 the inter ones (`NEARESTMV`..`NEWMV`), because the inter-mode context
/// table `MODE_2_COUNTER` is indexed by exactly that combined value.
#[derive(Clone, Copy)]
pub struct ModeInfo {
    pub y_mode: u8,
    pub uv_mode: u8,
    pub tx_size: u8,
    pub skip: bool,
    pub is_inter: bool,
    /// Sub-block modes for `BLOCK_4X4`/`4X8`/`8X4`; all four equal otherwise.
    pub sub_modes: [u8; 4],
    pub block_size: u8,
    /// `[primary, secondary]`; `secondary <= 0` means single-reference.
    pub ref_frame: [i8; 2],
    /// Whole-block motion vector per reference.
    pub mv: [super::inter::Mv; 2],
    /// Sub-8x8 motion vectors: `[reference][sub-block]`.
    pub sub_mv: [[super::inter::Mv; 4]; 2],
    /// Interpolation filter, or `SWITCHABLE_FILTERS` for an intra block —
    /// which is what makes the filter context work without checking
    /// inter-ness first.
    pub interp_filter: u8,
}

impl Default for ModeInfo {
    fn default() -> Self {
        ModeInfo {
            y_mode: 0,
            uv_mode: 0,
            tx_size: 0,
            skip: false,
            is_inter: false,
            sub_modes: [0; 4],
            block_size: 0,
            ref_frame: [super::inter::INTRA_FRAME, super::inter::NO_REF_FRAME],
            mv: [super::inter::Mv::default(); 2],
            sub_mv: [[super::inter::Mv::default(); 4]; 2],
            interp_filter: super::inter::SWITCHABLE_FILTERS,
        }
    }
}

/// Everything a tile decode needs that outlives one block.
pub struct FrameDecodeState {
    pub planes: [Plane; 3],
    pub mi_cols: usize,
    pub mi_rows: usize,
    /// Per-8x8 mode info, row-major over the whole frame.
    pub mi: Vec<ModeInfo>,
    /// Partition context bitmasks, one per 8x8 column / per 8x8 row of a
    /// superblock (`left` resets every superblock row).
    above_seg: Vec<u8>,
    left_seg: [u8; 8],
    /// Non-zero (entropy) context per plane, per 4x4 column / row.
    above_nz: [Vec<u8>; 3],
    left_nz: [[u8; 16]; 3],
    /// `[dc, ac]` quantisers per plane.
    dq: [[i32; 2]; 3],
    lossless: bool,
    tx_mode: TxMode,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
    /// Keyframe or `intra_only`: selects the constant partition probabilities.
    pub intra_only: bool,
    /// The frame's symbol tallies, moved here once at the end of decode so the
    /// caller can adapt from them. Boxed: it is ~30 KiB and must not be copied
    /// per block, which is why it is threaded as a parameter during decode and
    /// only parked here afterwards.
    pub counts: Option<alloc::boxed::Box<super::header::FrameCounts>>,
    /// First 8x8 column of the tile being decoded. Intra prediction treats the
    /// tile's left edge as having **no left neighbour**, which is what makes
    /// tiles independently decodable — a single-tile decoder never notices.
    tile_mi_col_start: usize,
    tile_mi_col_end: usize,
}

impl FrameDecodeState {
    pub fn new(
        width: usize,
        height: usize,
        base_q: i32,
        dq_y_dc: i32,
        dq_uv_dc: i32,
        dq_uv_ac: i32,
        lossless: bool,
        tx_mode: TxMode,
        subsampling_x: bool,
        subsampling_y: bool,
        intra_only: bool,
    ) -> FrameDecodeState {
        let cw = if subsampling_x { (width + 1) / 2 } else { width };
        let ch = if subsampling_y { (height + 1) / 2 } else { height };
        let mi_cols = (width + 7) / 8;
        let mi_rows = (height + 7) / 8;
        let sb_cols = (mi_cols + 7) / 8 * 8;
        FrameDecodeState {
            planes: [Plane::new(width, height), Plane::new(cw, ch), Plane::new(cw, ch)],
            mi_cols,
            mi_rows,
            mi: alloc::vec![ModeInfo::default(); mi_cols * mi_rows],
            above_seg: alloc::vec![0u8; sb_cols],
            left_seg: [0; 8],
            above_nz: [
                alloc::vec![0u8; mi_cols * 2 + 16],
                alloc::vec![0u8; mi_cols * 2 + 16],
                alloc::vec![0u8; mi_cols * 2 + 16],
            ],
            left_nz: [[0; 16]; 3],
            dq: [
                [dc_quant(base_q, dq_y_dc), ac_quant(base_q, 0)],
                [dc_quant(base_q, dq_uv_dc), ac_quant(base_q, dq_uv_ac)],
                [dc_quant(base_q, dq_uv_dc), ac_quant(base_q, dq_uv_ac)],
            ],
            lossless,
            tx_mode,
            subsampling_x,
            subsampling_y,
            intra_only,
            counts: None,
            tile_mi_col_start: 0,
            tile_mi_col_end: mi_cols,
        }
    }

    /// The left neighbour, **tile-relative**: libvpx sets
    /// `xd->left_mi = (mi_col > tile->mi_col_start) ? mi[-1] : NULL`, so a
    /// tile's first column has no left neighbour at all.
    ///
    /// That is not only a prediction rule — it feeds the skip context, the
    /// transform-size context and the keyframe Y-mode probabilities, all of
    /// which change how many bits the next symbol consumes. Using the frame's
    /// left neighbour instead desynchronises the arithmetic decoder at every
    /// tile boundary but the first, which is invisible on single-tile content.
    fn left_neighbour(&self, r: usize, c: usize) -> Option<&ModeInfo> {
        if c > self.tile_mi_col_start {
            self.mi_at(r, c - 1)
        } else {
            None
        }
    }

    /// First and last 8x8 column of the tile being decoded (the MV reference
    /// search is tile-bounded on both sides, unlike intra prediction which only
    /// cares about the left).
    pub fn tile_mi_col_start(&self) -> usize {
        self.tile_mi_col_start
    }
    pub fn tile_mi_col_end(&self) -> usize {
        self.tile_mi_col_end
    }

    fn mi_at(&self, r: usize, c: usize) -> Option<&ModeInfo> {
        if r < self.mi_rows && c < self.mi_cols {
            Some(&self.mi[r * self.mi_cols + c])
        } else {
            None
        }
    }

    /// Reset the per-superblock-row left contexts. Called at the start of every
    /// superblock row of every tile — a tile boundary resets them too, which is
    /// what makes tiles independently decodable.
    pub fn start_sb_row(&mut self) {
        self.left_seg = [0; 8];
        self.left_nz = [[0; 16]; 3];
    }

    /// Reset the above contexts across a tile's column range.
    pub fn start_tile(&mut self, mi_col_start: usize, mi_col_end: usize) {
        self.tile_mi_col_start = mi_col_start;
        self.tile_mi_col_end = mi_col_end;
        for c in mi_col_start..mi_col_end.min(self.above_seg.len()) {
            self.above_seg[c] = 0;
        }
        for p in 0..3 {
            // 4x4 columns per 8x8 MI: 2 for luma, 1 for 4:2:0 chroma.
            let per_mi = if p == 0 || !self.subsampling_x { 2 } else { 1 };
            let lo = mi_col_start * per_mi;
            let hi = (mi_col_end * per_mi).min(self.above_nz[p].len());
            for i in lo..hi {
                self.above_nz[p][i] = 0;
            }
        }
    }
}

/// `dec_partition_plane_context`.
fn partition_context(s: &FrameDecodeState, mi_row: usize, mi_col: usize, bsl: usize) -> usize {
    let above = (s.above_seg.get(mi_col).copied().unwrap_or(0) >> bsl) & 1;
    let left = (s.left_seg[mi_row & 7] >> bsl) & 1;
    (left as usize * 2 + above as usize) + bsl * 4
}

fn update_partition_context(s: &mut FrameDecodeState, mi_row: usize, mi_col: usize, subsize: u8, bw: usize) {
    let (a, l) = (
        tables::PARTITION_CONTEXT_LOOKUP[subsize as usize][0],
        tables::PARTITION_CONTEXT_LOOKUP[subsize as usize][1],
    );
    for i in 0..bw {
        if mi_col + i < s.above_seg.len() {
            s.above_seg[mi_col + i] = a;
        }
        let li = (mi_row & 7) + i;
        if li < 8 {
            s.left_seg[li] = l;
        }
    }
}

/// What the inter path needs that is not per-block: the resolved reference
/// pictures and the frame-level choices made in the two headers.
pub struct FrameRefs<'a> {
    /// LAST/GOLDEN/ALTREF, in that order (index = `ref_frame - 1`).
    pub bufs: [Option<&'a super::inter::RefFrame>; 3],
    /// Indexed by reference frame (0 = intra, unused).
    pub sign_bias: [bool; 4],
    pub comp_fixed_ref: i8,
    pub comp_var_ref: [i8; 2],
    pub reference_mode: super::header::ReferenceMode,
    /// `None` when the frame said SWITCHABLE and each block codes its own.
    pub interp_filter: Option<u8>,
    pub allow_high_precision_mv: bool,
    /// The previously *decoded* frame's per-8x8 motion field, when
    /// `use_prev_frame_mvs` holds. `None` disables the temporal candidate.
    pub prev_mvs: Option<&'a [([super::inter::Mv; 2], [i8; 2])]>,
}

impl<'a> FrameRefs<'a> {
    /// `vp9_setup_compound_reference_mode`: the fixed reference is the one whose
    /// sign bias differs from the other two, so compound prediction always
    /// combines a past and a future picture.
    pub fn compound_refs(sign_bias: [bool; 4]) -> (i8, [i8; 2]) {
        use super::inter::{ALTREF_FRAME, GOLDEN_FRAME, LAST_FRAME};
        if sign_bias[LAST_FRAME as usize] == sign_bias[GOLDEN_FRAME as usize] {
            (ALTREF_FRAME, [LAST_FRAME, GOLDEN_FRAME])
        } else if sign_bias[LAST_FRAME as usize] == sign_bias[ALTREF_FRAME as usize] {
            (GOLDEN_FRAME, [LAST_FRAME, ALTREF_FRAME])
        } else {
            (LAST_FRAME, [GOLDEN_FRAME, ALTREF_FRAME])
        }
    }
}

/// Decode a symbol from a VP9 probability tree and tally it.
fn read_tree_count(r: &mut BoolDecoder, tree: &[i8], probs: &[u8], counts: &mut [u32]) -> u32 {
    let v = read_tree(r, tree, probs);
    counts[v as usize] += 1;
    v
}

/// Decode a symbol from a VP9 probability tree.
fn read_tree(r: &mut BoolDecoder, tree: &[i8], probs: &[u8]) -> u32 {
    let mut i: usize = 0;
    loop {
        let p = probs[i >> 1];
        let n = tree[i + r.read_bool(p) as usize];
        if n <= 0 {
            return (-n) as u32;
        }
        i = n as usize;
    }
}

fn read_partition(
    r: &mut BoolDecoder,
    s: &FrameDecodeState,
    fc: &FrameContext,
    mi_row: usize,
    mi_col: usize,
    has_rows: bool,
    has_cols: bool,
    bsl: usize,
) -> (Partition, usize) {
    let ctx = partition_context(s, mi_row, mi_col, bsl);
    let count_ctx = ctx.min(15);
    // **Intra frames use the constant `KF_PARTITION_PROBS`, not the frame
    // context** (libvpx `set_partition_probs`). This is the one probability set
    // on the keyframe path that does not come from the adaptive context, and
    // using the adaptive one desynchronises the arithmetic decoder for the
    // whole tile — while the *modes* still look plausible, because keyframe Y
    // and UV modes come from their own constant tables. So the symptom is a
    // sensible-looking partition/mode grid over a completely wrong picture.
    let probs: &[u8; 3] = if s.intra_only {
        &tables::KF_PARTITION_PROBS[ctx.min(15)]
    } else {
        &fc.partition_probs[ctx.min(15)]
    };
    // At the frame's right/bottom edge only some partitions are representable,
    // so a *single* bit is coded rather than the tree. Reading the tree there
    // consumes bits that were never written.
    let p = match (has_rows, has_cols) {
        (true, true) => match read_tree(r, &tables::PARTITION_TREE, probs) {
            1 => Partition::Horz,
            2 => Partition::Vert,
            3 => Partition::Split,
            _ => Partition::None,
        },
        (false, true) => {
            if r.read_bool(probs[1]) != 0 {
                Partition::Split
            } else {
                Partition::Horz
            }
        }
        (true, false) => {
            if r.read_bool(probs[2]) != 0 {
                Partition::Split
            } else {
                Partition::Vert
            }
        }
        (false, false) => Partition::Split,
    };
    (p, count_ctx)
}

/// `get_y_mode` — a sub-8x8 block reports its own sub-mode, larger blocks their
/// single mode.
fn y_mode_of(mi: &ModeInfo, block: usize) -> u8 {
    if mi.block_size < 3 {
        mi.sub_modes[block.min(3)]
    } else {
        mi.y_mode
    }
}

/// `vp9_above_block_mode` / `vp9_left_block_mode`, specialised to the keyframe
/// path. `DC_PRED` when the neighbour is missing **or inter-coded** — the
/// second half is what keeps the `KF_Y_MODE_PROB` lookup in range.
fn above_block_mode(cur: &ModeInfo, above: Option<&ModeInfo>, b: usize) -> u8 {
    if b == 0 || b == 1 {
        match above {
            Some(m) if !m.is_inter => y_mode_of(m, b + 2),
            _ => 0,
        }
    } else {
        cur.sub_modes[b - 2]
    }
}

fn left_block_mode(cur: &ModeInfo, left: Option<&ModeInfo>, b: usize) -> u8 {
    if b == 0 || b == 2 {
        match left {
            Some(m) if !m.is_inter => y_mode_of(m, b + 1),
            _ => 0,
        }
    } else {
        cur.sub_modes[b - 1]
    }
}

/// `get_tx_size_context`.
fn tx_size_context(s: &FrameDecodeState, mi_row: usize, mi_col: usize, max_tx: u8) -> usize {
    let above = if mi_row > 0 { s.mi_at(mi_row - 1, mi_col) } else { None };
    let left = s.left_neighbour(mi_row, mi_col);
    let mut above_ctx = match above {
        Some(m) if !m.skip => m.tx_size,
        _ => max_tx,
    };
    let mut left_ctx = match left {
        Some(m) if !m.skip => m.tx_size,
        _ => max_tx,
    };
    if left.is_none() {
        left_ctx = above_ctx;
    }
    if above.is_none() {
        above_ctx = left_ctx;
    }
    ((above_ctx + left_ctx) > max_tx) as usize
}

fn read_tx_size(
    r: &mut BoolDecoder,
    s: &FrameDecodeState,
    fc: &FrameContext,
    mi_row: usize,
    mi_col: usize,
    block_size: u8,
    allow_select: bool,
) -> (u8, Option<(u8, usize)>) {
    let max_tx = tables::MAX_TXSIZE[block_size as usize];
    if allow_select && s.tx_mode == TxMode::Select && block_size >= 3 {
        let ctx = tx_size_context(s, mi_row, mi_col, max_tx);
        // Which probability set is used depends on the *maximum* transform for
        // this block size, not on the frame's tx_mode.
        let mut tx = match max_tx {
            3 => r.read_bool(fc.tx_probs_32x32[ctx][0]),
            2 => r.read_bool(fc.tx_probs_16x16[ctx][0]),
            _ => r.read_bool(fc.tx_probs_8x8[ctx][0]),
        };
        if tx != 0 && max_tx >= 2 {
            tx += match max_tx {
                3 => r.read_bool(fc.tx_probs_32x32[ctx][1]),
                _ => r.read_bool(fc.tx_probs_16x16[ctx][1]),
            };
            if tx != 1 && max_tx >= 3 {
                tx += r.read_bool(fc.tx_probs_32x32[ctx][2]);
            }
        }
        (tx as u8, Some((max_tx, ctx)))
    } else {
        (max_tx.min(tables::TX_MODE_TO_BIGGEST_TX_SIZE[s.tx_mode as usize]), None)
    }
}

fn read_skip(r: &mut BoolDecoder, s: &FrameDecodeState, fc: &FrameContext, mi_row: usize, mi_col: usize) -> (bool, usize) {
    let above = if mi_row > 0 { s.mi_at(mi_row - 1, mi_col).map(|m| m.skip as usize) } else { None };
    let left = s.left_neighbour(mi_row, mi_col).map(|m| m.skip as usize);
    let ctx = above.unwrap_or(0) + left.unwrap_or(0);
    (r.read_bool(fc.skip_probs[ctx]) != 0, ctx)
}

/// Read one intra block's mode info (keyframe path).
fn read_intra_frame_mode_info(
    r: &mut BoolDecoder,
    s: &FrameDecodeState,
    fc: &FrameContext,
    counts: &mut super::header::FrameCounts,
    mi_row: usize,
    mi_col: usize,
    block_size: u8,
) -> ModeInfo {
    let mut mi = ModeInfo { block_size, is_inter: false, ..Default::default() };
    mi.ref_frame = [super::inter::INTRA_FRAME, super::inter::NO_REF_FRAME];
    let (skip, skip_ctx) = read_skip(r, s, fc, mi_row, mi_col);
    mi.skip = skip;
    let (tx_size, tx_ctx) = read_tx_size(r, s, fc, mi_row, mi_col, block_size, true);
    mi.tx_size = tx_size;
    counts.skip[skip_ctx][skip as usize] += 1;
    if let Some((max_tx, ctx)) = tx_ctx {
        match max_tx {
            3 => counts.tx_32x32[ctx][tx_size as usize] += 1,
            2 => counts.tx_16x16[ctx][tx_size as usize] += 1,
            _ => counts.tx_8x8[ctx][tx_size as usize] += 1,
        }
    }

    let above = if mi_row > 0 { s.mi_at(mi_row - 1, mi_col).copied() } else { None };
    let left = s.left_neighbour(mi_row, mi_col).copied();
    let mut read_mode = |r: &mut BoolDecoder, mi: &ModeInfo, b: usize| -> u8 {
        let a = above_block_mode(mi, above.as_ref(), b) as usize;
        let l = left_block_mode(mi, left.as_ref(), b) as usize;
        read_tree(r, &tables::INTRA_MODE_TREE, &tables::KF_Y_MODE_PROB[a][l]) as u8
    };

    match block_size {
        0 => {
            // BLOCK_4X4: four independent sub-modes, each conditioned on the
            // previously decoded ones within the same block.
            for i in 0..4 {
                let m = read_mode(r, &mi, i);
                mi.sub_modes[i] = m;
            }
            mi.y_mode = mi.sub_modes[3];
        }
        1 => {
            // BLOCK_4X8: two columns.
            let m0 = read_mode(r, &mi, 0);
            mi.sub_modes[0] = m0;
            mi.sub_modes[2] = m0;
            let m1 = read_mode(r, &mi, 1);
            mi.sub_modes[1] = m1;
            mi.sub_modes[3] = m1;
            mi.y_mode = m1;
        }
        2 => {
            // BLOCK_8X4: two rows.
            let m0 = read_mode(r, &mi, 0);
            mi.sub_modes[0] = m0;
            mi.sub_modes[1] = m0;
            let m2 = read_mode(r, &mi, 2);
            mi.sub_modes[2] = m2;
            mi.sub_modes[3] = m2;
            mi.y_mode = m2;
        }
        _ => {
            let m = read_mode(r, &mi, 0);
            mi.y_mode = m;
            mi.sub_modes = [m; 4];
        }
    }
    mi.uv_mode = read_tree(r, &tables::INTRA_MODE_TREE, &tables::KF_UV_MODE_PROB[mi.y_mode as usize]) as u8;
    mi
}

/// The entropy context for one transform block: whether the above and left
/// neighbouring 4x4 columns/rows carried any non-zero coefficient.
fn nz_context(s: &FrameDecodeState, plane: usize, x4: usize, y4: usize, tx_size: TxSize) -> usize {
    let n = 1usize << (tx_size as usize);
    let mut a = 0u8;
    let mut l = 0u8;
    for i in 0..n {
        a |= s.above_nz[plane].get(x4 + i).copied().unwrap_or(0);
        l |= s.left_nz[plane][(y4 + i) & 15];
    }
    (a != 0) as usize + (l != 0) as usize
}

fn set_nz_context(s: &mut FrameDecodeState, plane: usize, x4: usize, y4: usize, tx_size: TxSize, v: u8) {
    let n = 1usize << (tx_size as usize);
    for i in 0..n {
        if x4 + i < s.above_nz[plane].len() {
            s.above_nz[plane][x4 + i] = v;
        }
        s.left_nz[plane][(y4 + i) & 15] = v;
    }
}

/// Reconstruct one transform block: predict, then decode + add the residual.
#[allow(clippy::too_many_arguments)]
fn reconstruct_block(
    r: &mut BoolDecoder,
    s: &mut FrameDecodeState,
    fc: &FrameContext,
    counts: &mut super::header::FrameCounts,
    plane: usize,
    mi: &ModeInfo,
    x: usize,
    y: usize,
    tx_size: TxSize,
    mode: IntraMode,
    have_above: bool,
    have_left: bool,
    have_right: bool,
) {
    let n = tx_size.width();
    let (stride, plane_w, plane_h) = {
        let p = &s.planes[plane];
        (p.stride, p.width, p.height)
    };
    // An inter block's prediction came from motion compensation over the whole
    // block, before any residual; only intra blocks predict per transform.
    if mi.is_inter {
        reconstruct_residual(r, s, fc, counts, plane, mi, x, y, tx_size, mode);
        return;
    }
    // Predict from the reconstructed neighbours.
    let edges = Edges::build(
        &s.planes[plane].data,
        stride,
        x,
        y,
        n,
        have_above,
        have_left,
        have_right,
        plane_w,
        plane_h,
    );
    {
        let off = y * stride + x;
        let dst = &mut s.planes[plane].data[off..];
        predict(dst, stride, n, mode, &edges, have_above, have_left);
    }

    reconstruct_residual(r, s, fc, counts, plane, mi, x, y, tx_size, mode);
}

/// Decode and add one transform block's residual. Shared by the intra and inter
/// paths — only the prediction differs.
#[allow(clippy::too_many_arguments)]
fn reconstruct_residual(
    r: &mut BoolDecoder,
    s: &mut FrameDecodeState,
    fc: &FrameContext,
    counts: &mut super::header::FrameCounts,
    plane: usize,
    mi: &ModeInfo,
    x: usize,
    y: usize,
    tx_size: TxSize,
    mode: IntraMode,
) {
    let stride = s.planes[plane].stride;
    if mi.skip {
        set_nz_context(s, plane, x >> 2, y >> 2, tx_size, 0);
        return;
    }
    let ctx = nz_context(s, plane, x >> 2, y >> 2, tx_size);
    let plane_type = if plane == 0 { 0 } else { 1 };
    // libvpx `get_tx_type`: **only luma** derives its transform type from the
    // prediction mode. Chroma, lossless blocks and inter blocks are always
    // DCT_DCT — passing the chroma mode through the luma table gives the wrong
    // basis for every U/V block whose mode is not DC.
    let tx_type = if plane == 0 && !s.lossless && !mi.is_inter {
        tx_type_for_mode(mode as u8, tx_size, false)
    } else {
        super::transform::TxType::DctDct
    };
    let mut coefs = [0i64; 32 * 32];
    let eob = decode_coefs(
        r,
        fc,
        &mut counts.coef[tx_size as usize][plane_type][mi.is_inter as usize],
        &mut counts.eob_branch[tx_size as usize][plane_type][mi.is_inter as usize],
        plane_type,
        mi.is_inter,
        tx_size,
        tx_type,
        s.dq[plane],
        ctx,
        &mut coefs[..tx_size.coefs()],
    );
    set_nz_context(s, plane, x >> 2, y >> 2, tx_size, (eob > 0) as u8);
    if eob == 0 {
        return;
    }
    let off = y * stride + x;
    let dst = &mut s.planes[plane].data[off..];
    if s.lossless {
        iwht4x4_add(&coefs[..16], dst, stride);
    } else {
        inverse_add(&coefs[..tx_size.coefs()], dst, stride, tx_size, tx_type);
    }
}

/// Decode one coded block (all planes) at `(mi_row, mi_col)`.
fn decode_block(
    r: &mut BoolDecoder,
    s: &mut FrameDecodeState,
    fc: &FrameContext,
    counts: &mut super::header::FrameCounts,
    refs: Option<&FrameRefs>,
    mi_row: usize,
    mi_col: usize,
    block_size: u8,
) {
    let mi = match refs {
        None => read_intra_frame_mode_info(r, s, fc, counts, mi_row, mi_col, block_size),
        Some(fr) => read_inter_frame_mode_info(r, s, fc, counts, fr, mi_row, mi_col, block_size),
    };
    // Record the mode info across every 8x8 the block covers, so neighbour
    // lookups from later blocks see it.
    let bw8 = tables::NUM_8X8_W[block_size as usize] as usize;
    let bh8 = tables::NUM_8X8_H[block_size as usize] as usize;
    for dy in 0..bh8 {
        for dx in 0..bw8 {
            let (rr, cc) = (mi_row + dy, mi_col + dx);
            if rr < s.mi_rows && cc < s.mi_cols {
                s.mi[rr * s.mi_cols + cc] = mi;
            }
        }
    }

    // Motion compensation fills the whole block before any residual, because a
    // sub-8x8 block's four 4x4 predictions each need the *unfiltered* reference,
    // not the partially reconstructed current frame.
    if mi.is_inter {
        if let Some(fr) = refs {
            build_inter_predictors(s, fr, &mi, mi_row, mi_col);
        }
    }

    for plane in 0..3 {
        let (ssx, ssy) = if plane == 0 {
            (false, false)
        } else {
            (s.subsampling_x, s.subsampling_y)
        };
        // Plane block size and transform size: chroma uses a smaller transform
        // for the same block, via the generated `uv_txsize_lookup`.
        let tx_size = if plane == 0 {
            TxSize::from_index(mi.tx_size)
        } else {
            TxSize::from_index(
                tables::UV_TXSIZE[block_size as usize][mi.tx_size as usize][ssx as usize][ssy as usize],
            )
        };
        let n = tx_size.width();
        // The residual covers the **mode-info block**, not the prediction
        // block: libvpx derives it as `n4_w = (bw << 1) >> ssx` where `bw` is
        // in 8x8 MI units, so a sub-8x8 block (`BLOCK_4X4`/`4X8`/`8X4`) still
        // carries a full 8x8 of luma — four 4x4 transforms, not one.
        //
        // Sizing this from `num_4x4_blocks_wide` instead gives 4x4 for those
        // shapes, so three quarters of every sub-8x8 block's coefficients go
        // unread and the arithmetic decoder desynchronises for the rest of the
        // tile. It only shows up on content the encoder searches deeply enough
        // to use sub-8x8 partitions at all, which is why a fast encode decoded
        // bit-exact and a slow one of the same clip overran its tile.
        let bw = (tables::NUM_8X8_W[block_size as usize] as usize * 8) >> ssx as usize;
        let bh = (tables::NUM_8X8_H[block_size as usize] as usize * 8) >> ssy as usize;
        let x0 = (mi_col * 8) >> ssx as usize;
        let y0 = (mi_row * 8) >> ssy as usize;
        let mode = if plane == 0 {
            IntraMode::from_index(mi.y_mode)
        } else {
            IntraMode::from_index(mi.uv_mode)
        };
        let (pw, ph) = (s.planes[plane].width, s.planes[plane].height);

        let mut y = 0;
        while y < bh.max(n) {
            let mut x = 0;
            while x < bw.max(n) {
                let (px, py) = (x0 + x, y0 + y);
                if px >= pw || py >= ph {
                    x += n;
                    continue;
                }
                // Sub-8x8 luma blocks predict per 4x4 with their own sub-mode.
                let m = if plane == 0 && block_size < 3 {
                    let bi = (y / 4).min(1) * 2 + (x / 4).min(1);
                    IntraMode::from_index(mi.sub_modes[bi])
                } else {
                    mode
                };
                // libvpx `vp9_predict_intra_block`: availability is a property
                // of the transform block's position *within its prediction
                // block*, plus the block's own neighbours.
                //   have_top   = loff || above_mi != NULL
                //   have_left  = aoff || left_mi  != NULL   (tile-relative)
                //   have_right = (aoff + txw) < bw
                let have_above = y > 0 || mi_row > 0;
                let have_left = x > 0 || mi_col > s.tile_mi_col_start;
                let have_right = x + n < bw;
                reconstruct_block(
                    r, s, fc, counts, plane, &mi, px, py, tx_size, m, have_above, have_left,
                    have_right,
                );
                x += n;
            }
            y += n;
        }
    }
}

/// Recursively decode a partition subtree rooted at `(mi_row, mi_col)`.
///
/// `n8x8_l2` is `log2` of the subtree's width in 8x8 units, which is also the
/// bit position the partition context reads — the two are the same number, and
/// keeping them as one variable is what makes the context indexing correct.
pub fn decode_partition(
    r: &mut BoolDecoder,
    s: &mut FrameDecodeState,
    fc: &FrameContext,
    counts: &mut super::header::FrameCounts,
    refs: Option<&FrameRefs>,
    mi_row: usize,
    mi_col: usize,
    block_size: u8,
    n4x4_l2: usize,
) {
    if mi_row >= s.mi_rows || mi_col >= s.mi_cols {
        return;
    }
    let n8x8_l2 = n4x4_l2.wrapping_sub(1);
    let num_8x8_wh = 1usize << n8x8_l2.min(3);
    let hbs = num_8x8_wh >> 1;
    let has_rows = mi_row + hbs < s.mi_rows;
    let has_cols = mi_col + hbs < s.mi_cols;

    let (partition, pctx) = read_partition(r, s, fc, mi_row, mi_col, has_rows, has_cols, n8x8_l2.min(3));
    // libvpx counts the partition symbol at every level, including the ones a
    // frame edge coded as a single bit.
    counts.partition[pctx][partition as usize] += 1;
    let subsize = tables::SUBSIZE[partition as usize][block_size as usize];

    if hbs == 0 {
        // 8x8 root: the partition selects a sub-8x8 shape and there is one block.
        decode_block(r, s, fc, counts, refs, mi_row, mi_col, subsize);
    } else {
        match partition {
            Partition::None => decode_block(r, s, fc, counts, refs, mi_row, mi_col, subsize),
            Partition::Horz => {
                decode_block(r, s, fc, counts, refs, mi_row, mi_col, subsize);
                if has_rows {
                    decode_block(r, s, fc, counts, refs, mi_row + hbs, mi_col, subsize);
                }
            }
            Partition::Vert => {
                decode_block(r, s, fc, counts, refs, mi_row, mi_col, subsize);
                if has_cols {
                    decode_block(r, s, fc, counts, refs, mi_row, mi_col + hbs, subsize);
                }
            }
            Partition::Split => {
                decode_partition(r, s, fc, counts, refs, mi_row, mi_col, subsize, n8x8_l2);
                decode_partition(r, s, fc, counts, refs, mi_row, mi_col + hbs, subsize, n8x8_l2);
                decode_partition(r, s, fc, counts, refs, mi_row + hbs, mi_col, subsize, n8x8_l2);
                decode_partition(r, s, fc, counts, refs, mi_row + hbs, mi_col + hbs, subsize, n8x8_l2);
            }
        }
    }

    // Only a leaf (or an 8x8 root) updates the context — a SPLIT above 8x8 has
    // already had its children update it.
    if block_size >= 3 && (block_size == 3 || partition != Partition::Split) {
        update_partition_context(s, mi_row, mi_col, subsize, num_8x8_wh);
    }
}


// ---------------------------------------------------------------------------
// Inter blocks
// ---------------------------------------------------------------------------

use super::inter::{self, Mv};

/// The previous frame's motion field entry co-located with this block.
fn prev_mv<'a>(
    refs: &'a FrameRefs,
    s: &FrameDecodeState,
    mi_row: usize,
    mi_col: usize,
) -> Option<&'a ([super::inter::Mv; 2], [i8; 2])> {
    refs.prev_mvs?.get(mi_row * s.mi_cols + mi_col)
}

/// `read_mv_component`.
fn read_mv_component(r: &mut BoolDecoder, comp: &super::header::MvComp, usehp: bool) -> i32 {
    let sign = r.read_bool(comp.sign) != 0;
    let mv_class = read_tree(r, &tables::MV_CLASS_TREE, &comp.classes) as usize;
    let class0 = mv_class == 0;
    let (d, mut mag) = if class0 {
        let d = r.read_bool(comp.class0[0]) as i32;
        (d, 0i32)
    } else {
        // `mv_class + CLASS0_BITS - 1` bits with `CLASS0_BITS == 1`, i.e.
        // exactly `mv_class` bits, read **LSB first**. Reading them MSB first
        // (the natural way to write it) mirrors the magnitude.
        let n = mv_class;
        let mut d = 0i32;
        for i in 0..n {
            let b = r.read_bool(comp.bits[i]);
            d |= (b as i32) << i;
        }
        // `mag = CLASS0_SIZE << (mv_class + 2)` with CLASS0_SIZE == 2.
        (d, 2i32 << (mv_class + 2))
    };
    let fp_probs: &[u8] = if class0 { &comp.class0_fp[(d as usize).min(1)] } else { &comp.fp };
    let fr = read_tree(r, &tables::MV_FP_TREE, fp_probs) as i32;
    // Without high precision the low bit is *1*, not 0 — it is the implicit
    // midpoint, and using 0 biases every vector by half a step.
    let hp = if usehp {
        r.read_bool(if class0 { comp.class0_hp } else { comp.hp }) as i32
    } else {
        1
    };
    mag += ((d << 3) | (fr << 1) | hp) + 1;
    if sign {
        -mag
    } else {
        mag
    }
}

/// `read_mv`: a joint tells which components are non-zero, then each is a delta
/// from the reference vector.
fn read_mv(
    r: &mut BoolDecoder,
    reference: Mv,
    fc: &FrameContext,
    counts: &mut super::header::FrameCounts,
    allow_hp: bool,
) -> Mv {
    let joint = read_tree(r, &tables::MV_JOINT_TREE, &fc.mv_joint_probs);
    let use_hp = allow_hp && inter::use_mv_hp(reference);
    let mut diff = Mv::default();
    // Joint 0 = both zero, 1 = col only, 2 = row only, 3 = both.
    if joint == 2 || joint == 3 {
        diff.row = read_mv_component(r, &fc.mv_comp[0], use_hp) as i16;
    }
    if joint == 1 || joint == 3 {
        diff.col = read_mv_component(r, &fc.mv_comp[1], use_hp) as i16;
    }
    // libvpx `vp9_inc_mv` counts from the **reconstructed difference**, not from
    // the symbols as they were read — and it counts the high-precision bit
    // *always*, even where the bitstream did not code one (its value is then the
    // implicit 1). Counting per-symbol instead leaves the hp tallies short and
    // desynchronises a later frame, several frames after the omission.
    inc_mv(diff, counts);
    Mv { row: reference.row.wrapping_add(diff.row), col: reference.col.wrapping_add(diff.col) }
}

/// `vp9_get_mv_class`: which magnitude class a component's offset falls in.
fn mv_class(z: i32) -> (usize, i32) {
    const CLASS0_SIZE: i32 = 2;
    let c = if z >= CLASS0_SIZE * 4096 {
        10usize
    } else {
        tables::LOG_IN_BASE_2[((z >> 3) as usize).min(1024)] as usize
    };
    let base = if c > 0 { CLASS0_SIZE << (c + 2) } else { 0 };
    (c, z - base)
}

/// `vp9_inc_mv` / `inc_mv_component`.
fn inc_mv(diff: Mv, counts: &mut super::header::FrameCounts) {
    // The joint is *recomputed* from the difference rather than taken from the
    // decoded symbol — they agree, but this is the reference's definition.
    let j = match (diff.row != 0, diff.col != 0) {
        (false, false) => 0usize,
        (false, true) => 1,
        (true, false) => 2,
        (true, true) => 3,
    };
    counts.mv_joint[j] += 1;
    if diff.row != 0 {
        inc_mv_component(diff.row as i32, &mut counts.mv_comp[0]);
    }
    if diff.col != 0 {
        inc_mv_component(diff.col as i32, &mut counts.mv_comp[1]);
    }
}

fn inc_mv_component(v: i32, c: &mut super::header::MvCompCounts) {
    let s = (v < 0) as usize;
    c.sign[s] += 1;
    let z = (if v < 0 { -v } else { v }) - 1;
    let (cls, o) = mv_class(z);
    c.classes[cls.min(10)] += 1;
    let d = o >> 3;
    let f = ((o >> 1) & 3) as usize;
    let e = (o & 1) as usize;
    if cls == 0 {
        c.class0[(d as usize).min(1)] += 1;
        c.class0_fp[(d as usize).min(1)][f] += 1;
        c.class0_hp[e] += 1;
    } else {
        // `c + CLASS0_BITS - 1` bits, i.e. `cls`.
        for i in 0..cls.min(10) {
            c.bits[i][((d >> i) & 1) as usize] += 1;
        }
        c.fp[f] += 1;
        c.hp[e] += 1;
    }
}

/// `read_ref_frames`.
fn read_ref_frames(
    r: &mut BoolDecoder,
    s: &FrameDecodeState,
    fc: &FrameContext,
    counts: &mut super::header::FrameCounts,
    refs: &FrameRefs,
    mi_row: usize,
    mi_col: usize,
) -> [i8; 2] {
    use super::header::ReferenceMode;
    let above = if mi_row > 0 { s.mi_at(mi_row - 1, mi_col) } else { None };
    let left = s.left_neighbour(mi_row, mi_col);
    let mode = match refs.reference_mode {
        ReferenceMode::Select => {
            let ctx = inter::reference_mode_context(above, left, refs.comp_fixed_ref);
            let bit = r.read_bool(fc.comp_inter_probs[ctx.min(4)]);
            counts.comp_inter[ctx.min(4)][bit as usize] += 1;
            if bit != 0 {
                ReferenceMode::Compound
            } else {
                ReferenceMode::Single
            }
        }
        m => m,
    };
    if mode == ReferenceMode::Compound {
        let fix_idx = refs.sign_bias[refs.comp_fixed_ref as usize] as usize;
        let ctx = inter::comp_ref_context(
            above,
            left,
            refs.comp_fixed_ref,
            refs.comp_var_ref,
            fix_idx,
        );
        let bit = r.read_bool(fc.comp_ref_probs[ctx.min(4)]) as usize;
        counts.comp_ref[ctx.min(4)][bit] += 1;
        let mut out = [0i8; 2];
        out[fix_idx] = refs.comp_fixed_ref;
        out[1 - fix_idx] = refs.comp_var_ref[bit];
        out
    } else {
        let ctx0 = inter::single_ref_p1_context(above, left);
        let bit0 = r.read_bool(fc.single_ref_probs[ctx0.min(4)][0]);
        counts.single_ref[ctx0.min(4)][0][bit0 as usize] += 1;
        if bit0 != 0 {
            let ctx1 = inter::single_ref_p2_context(above, left);
            let bit1 = r.read_bool(fc.single_ref_probs[ctx1.min(4)][1]);
            counts.single_ref[ctx1.min(4)][1][bit1 as usize] += 1;
            [if bit1 != 0 { inter::ALTREF_FRAME } else { inter::GOLDEN_FRAME }, inter::NO_REF_FRAME]
        } else {
            [inter::LAST_FRAME, inter::NO_REF_FRAME]
        }
    }
}

/// Read one inter-frame block's mode info (libvpx `read_inter_frame_mode_info`).
#[allow(clippy::too_many_arguments)]
fn read_inter_frame_mode_info(
    r: &mut BoolDecoder,
    s: &FrameDecodeState,
    fc: &FrameContext,
    counts: &mut super::header::FrameCounts,
    refs: &FrameRefs,
    mi_row: usize,
    mi_col: usize,
    block_size: u8,
) -> ModeInfo {
    let mut mi = ModeInfo { block_size, ..Default::default() };
    let (skip, skip_ctx) = read_skip(r, s, fc, mi_row, mi_col);
    mi.skip = skip;
    counts.skip[skip_ctx][skip as usize] += 1;

    let above = if mi_row > 0 { s.mi_at(mi_row - 1, mi_col).copied() } else { None };
    let left = s.left_neighbour(mi_row, mi_col).copied();
    let ctx = inter::intra_inter_context(above.as_ref(), left.as_ref());
    let is_inter = r.read_bool(fc.intra_inter_probs[ctx.min(3)]) != 0;
    counts.intra_inter[ctx.min(3)][is_inter as usize] += 1;
    mi.is_inter = is_inter;
    // `allow_select` is `!skip || !inter` — a skipped *inter* block codes no
    // transform size, because it has no residual to size.
    let (tx_size, tx_ctx) = read_tx_size(r, s, fc, mi_row, mi_col, block_size, !mi.skip || !is_inter);
    mi.tx_size = tx_size;
    if let Some((max_tx, tctx)) = tx_ctx {
        match max_tx {
            3 => counts.tx_32x32[tctx][tx_size as usize] += 1,
            2 => counts.tx_16x16[tctx][tx_size as usize] += 1,
            _ => counts.tx_8x8[tctx][tx_size as usize] += 1,
        }
    }

    if !is_inter {
        // An intra block inside an inter frame uses the *adaptive* y-mode
        // probabilities keyed by block-size group — not the keyframe tables,
        // which are only for intra frames.
        let group = if block_size >= 3 { tables::SIZE_GROUP[block_size as usize] as usize } else { 0 };
        match block_size {
            0 => {
                for i in 0..4 {
                    mi.sub_modes[i] = read_tree_count(r, &tables::INTRA_MODE_TREE, &fc.y_mode_probs[0], &mut counts.y_mode[0]) as u8;
                }
                mi.y_mode = mi.sub_modes[3];
            }
            1 => {
                let m0 = read_tree_count(r, &tables::INTRA_MODE_TREE, &fc.y_mode_probs[0], &mut counts.y_mode[0]) as u8;
                mi.sub_modes[0] = m0;
                mi.sub_modes[2] = m0;
                let m1 = read_tree_count(r, &tables::INTRA_MODE_TREE, &fc.y_mode_probs[0], &mut counts.y_mode[0]) as u8;
                mi.sub_modes[1] = m1;
                mi.sub_modes[3] = m1;
                mi.y_mode = m1;
            }
            2 => {
                let m0 = read_tree_count(r, &tables::INTRA_MODE_TREE, &fc.y_mode_probs[0], &mut counts.y_mode[0]) as u8;
                mi.sub_modes[0] = m0;
                mi.sub_modes[1] = m0;
                let m2 = read_tree_count(r, &tables::INTRA_MODE_TREE, &fc.y_mode_probs[0], &mut counts.y_mode[0]) as u8;
                mi.sub_modes[2] = m2;
                mi.sub_modes[3] = m2;
                mi.y_mode = m2;
            }
            _ => {
                let m = read_tree_count(r, &tables::INTRA_MODE_TREE, &fc.y_mode_probs[group.min(3)], &mut counts.y_mode[group.min(3)]) as u8;
                mi.y_mode = m;
                mi.sub_modes = [m; 4];
            }
        }
        mi.uv_mode = read_tree_count(
            r,
            &tables::INTRA_MODE_TREE,
            &fc.uv_mode_probs[mi.y_mode as usize],
            &mut counts.uv_mode[mi.y_mode as usize],
        ) as u8;
        mi.interp_filter = inter::SWITCHABLE_FILTERS;
        mi.ref_frame = [inter::INTRA_FRAME, inter::NO_REF_FRAME];
        return mi;
    }

    mi.ref_frame = read_ref_frames(r, s, fc, counts, refs, mi_row, mi_col);
    let is_compound = mi.ref_frame[1] > inter::INTRA_FRAME;
    let mode_ctx = inter::mode_context(s, block_size, mi_row, mi_col);

    if block_size >= 3 {
        mi.y_mode = inter::NEARESTMV
            + read_tree_count(r, &tables::INTER_MODE_TREE, &fc.inter_mode_probs[mode_ctx.min(6)], &mut counts.inter_mode[mode_ctx.min(6)]) as u8;
    }
    mi.interp_filter = match refs.interp_filter {
        Some(f) => f,
        None => {
            let c = inter::interp_filter_context(above.as_ref(), left.as_ref());
            read_tree_count(r, &tables::INTERP_FILTER_TREE, &fc.interp_filter_probs[c.min(3)], &mut counts.interp_filter[c.min(3)]) as u8
        }
    };

    let allow_hp = refs.allow_high_precision_mv;
    let nrefs = 1 + is_compound as usize;

    if block_size < 3 {
        // libvpx derives these from the *partition*:
        //   `bmode_blocks_wl = 1 >> !!(partition & PARTITION_VERT)`
        //   `bmode_blocks_hl = 1 >> !!(partition & PARTITION_HORZ)`
        // which comes out as 1 step (i.e. two sub-blocks along that axis) only
        // where the partition did **not** split that axis:
        //   BLOCK_4X4 (SPLIT) → 1,1 — four sub-blocks
        //   BLOCK_4X8 (VERT)  → 1,2 — two side by side
        //   BLOCK_8X4 (HORZ)  → 2,1 — two stacked
        // Getting these the wrong way round makes a `BLOCK_4X4` read **one**
        // motion vector instead of four, which desynchronises the tile from the
        // first sub-8x8 inter block onward — invisible on shallow-searched
        // content that never emits one.
        let num_4x4_w = if block_size == 2 { 2 } else { 1 };
        let num_4x4_h = if block_size == 1 { 2 } else { 1 };
        let mut best_ref = [Mv::default(); 2];
        let mut got_new = false;
        let mut b_mode = inter::ZEROMV;
        let mut idy = 0usize;
        while idy < 2 {
            let mut idx = 0usize;
            while idx < 2 {
                let j = idy * 2 + idx;
                b_mode = inter::NEARESTMV
                    + read_tree_count(r, &tables::INTER_MODE_TREE, &fc.inter_mode_probs[mode_ctx.min(6)], &mut counts.inter_mode[mode_ctx.min(6)]) as u8;
                let mut near_nearest = [Mv::default(); 2];
                if b_mode == inter::NEARESTMV || b_mode == inter::NEARMV {
                    for rf in 0..nrefs {
                        let list = inter::find_mv_refs(
                            s, block_size, mi.ref_frame[rf], b_mode, mi_row, mi_col, j as isize, &refs.sign_bias, prev_mv(refs, s, mi_row, mi_col),
                        );
                        near_nearest[rf] = append_sub8x8_mv(&mi, rf, j, b_mode, &list);
                    }
                } else if b_mode == inter::NEWMV && !got_new {
                    for rf in 0..nrefs {
                        let list = inter::find_mv_refs(
                            s, block_size, mi.ref_frame[rf], inter::NEWMV, mi_row, mi_col, -1, &refs.sign_bias, prev_mv(refs, s, mi_row, mi_col),
                        );
                        let mut m = list.first().copied().unwrap_or_default();
                        inter::lower_mv_precision(&mut m, allow_hp);
                        best_ref[rf] = m;
                    }
                    got_new = true;
                }
                for rf in 0..nrefs {
                    let mv = match b_mode {
                        inter::NEWMV => read_mv(r, best_ref[rf], fc, counts, allow_hp),
                        inter::ZEROMV => Mv::default(),
                        _ => near_nearest[rf],
                    };
                    mi.sub_mv[rf][j] = mv;
                    if num_4x4_h == 2 {
                        mi.sub_mv[rf][j + 2] = mv;
                    }
                    if num_4x4_w == 2 {
                        mi.sub_mv[rf][j + 1] = mv;
                    }
                }
                idx += num_4x4_w;
            }
            idy += num_4x4_h;
        }
        mi.y_mode = b_mode;
        for rf in 0..2 {
            mi.mv[rf] = mi.sub_mv[rf][3];
        }
    } else {
        let mut best_ref = [Mv::default(); 2];
        if mi.y_mode != inter::ZEROMV {
            for rf in 0..nrefs {
                let list = inter::find_mv_refs(
                    s, block_size, mi.ref_frame[rf], mi.y_mode, mi_row, mi_col, -1, &refs.sign_bias, prev_mv(refs, s, mi_row, mi_col),
                );
                let idx = list.len().saturating_sub(1);
                let mut m = list.get(idx).copied().unwrap_or_default();
                inter::lower_mv_precision(&mut m, allow_hp);
                best_ref[rf] = m;
            }
        }
        for rf in 0..nrefs {
            mi.mv[rf] = match mi.y_mode {
                inter::NEWMV => read_mv(r, best_ref[rf], fc, counts, allow_hp),
                inter::ZEROMV => Mv::default(),
                _ => best_ref[rf],
            };
            mi.sub_mv[rf] = [mi.mv[rf]; 4];
        }
    }
    mi
}

/// libvpx `append_sub8x8_mvs_for_idx`: which candidate a sub-8x8 block takes.
///
/// This is **not** simply "entry 0 for NEAREST, entry 1 for NEAR". Sub-blocks
/// 1, 2 and 3 prefer the motion vectors of the *earlier sub-blocks of the same
/// 8x8*, falling back to the spatial list only for a vector that differs from
/// the one they are conditioned on. Treating it as a plain list index decodes
/// the first inter frame of a deeply-searched stream as noise, while a
/// shallow-searched one (which never uses sub-8x8 inter blocks) stays perfect.
fn append_sub8x8_mv(mi: &ModeInfo, rf: usize, block: usize, b_mode: u8, list: &[Mv]) -> Mv {
    let at = |i: usize| list.get(i).copied().unwrap_or_default();
    match block {
        0 => at(list.len().saturating_sub(1)),
        1 | 2 => {
            if b_mode == inter::NEARESTMV {
                mi.sub_mv[rf][0]
            } else {
                (0..2)
                    .map(at)
                    .find(|&c| c != mi.sub_mv[rf][0])
                    .unwrap_or_default()
            }
        }
        _ => {
            if b_mode == inter::NEARESTMV {
                mi.sub_mv[rf][2]
            } else {
                // Candidate order matters: sub-block 1, then 0, then the two
                // spatial entries.
                let cands = [mi.sub_mv[rf][1], mi.sub_mv[rf][0], at(0), at(1)];
                cands
                    .into_iter()
                    .find(|&c| c != mi.sub_mv[rf][2])
                    .unwrap_or_default()
            }
        }
    }
}

/// `round_mv_comp_q4` — the sub-8x8 chroma MV is the rounded mean of the four
/// luma sub-vectors, rounding **away from zero**.
fn round_mv_q4(v: i32) -> i16 {
    // Parenthesised deliberately: `if c { a } else { b } / 4` is a shape whose
    // binding is easy to misread, and this rounds *away from zero* rather than
    // truncating, which is a half-step of motion on every sub-8x8 chroma block.
    let biased = if v < 0 { v - 2 } else { v + 2 };
    (biased / 4) as i16
}

/// Motion-compensate one block (all planes, both references).
fn build_inter_predictors(s: &mut FrameDecodeState, refs: &FrameRefs, mi: &ModeInfo, mi_row: usize, mi_col: usize) {
    let is_compound = mi.ref_frame[1] > inter::INTRA_FRAME;
    let filter = mi.interp_filter.min(3) as usize;
    for rf in 0..(1 + is_compound as usize) {
        let slot = (mi.ref_frame[rf] - 1).max(0) as usize;
        let Some(reference) = refs.bufs[slot.min(2)] else { continue };
        for plane in 0..3 {
            let (ssx, ssy) = if plane == 0 {
                (0usize, 0usize)
            } else {
                (s.subsampling_x as usize, s.subsampling_y as usize)
            };
            let bw = (tables::NUM_8X8_W[mi.block_size as usize] as usize * 8) >> ssx;
            let bh = (tables::NUM_8X8_H[mi.block_size as usize] as usize * 8) >> ssy;
            let x0 = (mi_col * 8) >> ssx;
            let y0 = (mi_row * 8) >> ssy;
            let (stride, pw, ph) = {
                let p = &s.planes[plane];
                (p.stride, p.width, p.height)
            };
            let step = if mi.block_size < 3 { 4 } else { bw.max(bh) };
            let (sub_w, sub_h) = if mi.block_size < 3 { (bw / 4, bh / 4) } else { (1, 1) };
            for by in 0..sub_h.max(1) {
                for bx in 0..sub_w.max(1) {
                    let (w, h) = if mi.block_size < 3 { (4, 4) } else { (bw, bh) };
                    let px = x0 + bx * step.min(4);
                    let py = y0 + by * step.min(4);
                    if px >= pw || py >= ph {
                        continue;
                    }
                    let mv = if mi.block_size < 3 {
                        if ssx == 1 && ssy == 1 {
                            // 4:2:0 chroma averages all four sub-vectors.
                            let sum_r: i32 = (0..4).map(|k| mi.sub_mv[rf][k].row as i32).sum();
                            let sum_c: i32 = (0..4).map(|k| mi.sub_mv[rf][k].col as i32).sum();
                            Mv { row: round_mv_q4(sum_r), col: round_mv_q4(sum_c) }
                        } else {
                            mi.sub_mv[rf][(by * 2 + bx).min(3)]
                        }
                    } else {
                        mi.mv[rf]
                    };
                    // A motion vector is 1/8-pel **luma**; compensation works in
                    // 1/16-pel, so luma doubles it and chroma does not.
                    let mv_r = mv.row as i32 * (1 << (1 - ssy));
                    let mv_c = mv.col as i32 * (1 << (1 - ssx));
                    let src_x = px as i32 + (mv_c >> 4);
                    let src_y = py as i32 + (mv_r >> 4);
                    let sub_x = (mv_c & 15) as usize;
                    let sub_y = (mv_r & 15) as usize;
                    let off = py * stride + px;
                    inter::convolve8(
                        &reference.planes[plane],
                        src_x,
                        src_y,
                        sub_x,
                        sub_y,
                        filter,
                        &mut s.planes[plane].data,
                        off,
                        stride,
                        w.min(pw - px),
                        h.min(ph - py),
                        rf == 1,
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn partition_context_packs_left_above_and_level() {
        let s = FrameDecodeState::new(64, 64, 50, 0, 0, 0, false, TxMode::Only4x4, true, true, true);
        // Both contexts clear at the frame origin: context 0 within the level.
        assert_eq!(partition_context(&s, 0, 0, 3), 3 * 4);
        assert_eq!(partition_context(&s, 0, 0, 0), 0);
    }

    /// Every probability tree must terminate on every bit path, with leaves in
    /// range. This is the test that catches a tree whose leaves lost their
    /// sign: `-PARTITION_NONE` parsed as `PARTITION_NONE` turns a leaf into a
    /// node index and the walk cycles `1 -> 2 -> 1` forever. The decoder does
    /// not crash — it hangs, which is far harder to attribute.
    fn walk_tree(tree: &[i8], leaves: usize) {
        // Depth-first over both branches of every reachable node.
        let mut stack = alloc::vec![0usize];
        let mut steps = 0;
        while let Some(i) = stack.pop() {
            steps += 1;
            assert!(steps < 1000, "tree walk did not terminate (a cycle)");
            assert!(i + 1 < tree.len(), "node {} runs past the tree", i);
            for b in 0..2 {
                let n = tree[i + b];
                if n <= 0 {
                    assert!((-n as usize) < leaves, "leaf {} outside 0..{}", -n, leaves);
                } else {
                    assert!(n as usize > i, "node {} does not move forward", n);
                    stack.push(n as usize);
                }
            }
        }
    }

    #[test_case]
    fn every_probability_tree_terminates() {
        walk_tree(&tables::PARTITION_TREE, 4);
        walk_tree(&tables::INTRA_MODE_TREE, 10);
        walk_tree(&tables::INTER_MODE_TREE, 4);
        walk_tree(&tables::INTERP_FILTER_TREE, 3);
        walk_tree(&tables::MV_JOINT_TREE, 4);
        walk_tree(&tables::MV_CLASS_TREE, 11);
        walk_tree(&tables::MV_FP_TREE, 4);
    }

    #[test_case]
    fn tree_leaves_kept_their_negative_sign() {
        // Spelled out because the failure is a hang, not a wrong value: a
        // positive entry is a node index and a non-positive one is `-leaf`.
        assert_eq!(tables::PARTITION_TREE, [0, 2, -1, 4, -2, -3]);
        assert_eq!(tables::INTRA_MODE_TREE[2], -9, "TM_PRED leaf");
        assert_eq!(tables::MV_JOINT_TREE, [0, 2, -1, 4, -2, -3]);
    }

    #[test_case]
    fn subsize_lookup_splits_as_expected() {
        // PARTITION_NONE of 64x64 is 64x64; SPLIT of 64x64 is 32x32;
        // HORZ of 64x64 is 64x32.
        assert_eq!(tables::SUBSIZE[Partition::None as usize][12], 12);
        assert_eq!(tables::SUBSIZE[Partition::Split as usize][12], 9);
        assert_eq!(tables::SUBSIZE[Partition::Horz as usize][12], 11);
        assert_eq!(tables::SUBSIZE[Partition::Vert as usize][12], 10);
    }

    #[test_case]
    fn a_missing_or_inter_neighbour_predicts_dc() {
        // The KF_Y_MODE_PROB lookup is indexed by neighbour modes, so an inter
        // neighbour must fall back to DC rather than report its (meaningless)
        // intra mode — otherwise the index is whatever the inter mode enum
        // happened to be.
        let cur = ModeInfo::default();
        assert_eq!(above_block_mode(&cur, None, 0), 0);
        assert_eq!(left_block_mode(&cur, None, 0), 0);
        let inter = ModeInfo { is_inter: true, y_mode: 9, block_size: 3, ..Default::default() };
        assert_eq!(above_block_mode(&cur, Some(&inter), 0), 0);
        assert_eq!(left_block_mode(&cur, Some(&inter), 0), 0);
        let intra = ModeInfo { is_inter: false, y_mode: 7, block_size: 3, ..Default::default() };
        assert_eq!(above_block_mode(&cur, Some(&intra), 0), 7);
    }

    #[test_case]
    fn plane_geometry_follows_the_subsampling() {
        let s = FrameDecodeState::new(176, 144, 50, 0, 0, 0, false, TxMode::Only4x4, true, true, true);
        assert_eq!(s.planes[0].width, 176);
        assert_eq!(s.planes[0].height, 144);
        assert_eq!(s.planes[1].width, 88, "4:2:0 chroma is half width");
        assert_eq!(s.planes[1].height, 72);
        assert_eq!(s.mi_cols, 22);
        assert_eq!(s.mi_rows, 18);
        // Strides are superblock-aligned so prediction may read past the edge.
        assert_eq!(s.planes[0].stride % 64, 0);
    }

    #[test_case]
    fn quantisers_are_per_plane() {
        // Y and UV take different lookups, and a UV AC delta must not leak into
        // the Y plane.
        let s = FrameDecodeState::new(64, 64, 100, 0, 0, 10, false, TxMode::Only4x4, true, true, true);
        assert_eq!(s.dq[0][0], dc_quant(100, 0));
        assert_eq!(s.dq[0][1], ac_quant(100, 0));
        assert_eq!(s.dq[1][1], ac_quant(100, 10));
        assert_ne!(s.dq[0][1], s.dq[1][1]);
    }
}
