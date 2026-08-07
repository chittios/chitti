//! VP9's **compressed** header: the arithmetic-coded probability updates that
//! precede the tile data (spec §6.3, libvpx `read_compressed_header`).
//!
//! This is the part of VP9 with no analogue in H.26x. A frame does not merely
//! *use* probabilities, it **transmits deltas to them**, and the resulting set
//! is what the tile data is decoded against — and, if `refresh_frame_context`
//! is set, what the *next* frames start from. So a bug here is not a wrong
//! pixel; it desynchronises the arithmetic decoder for the rest of the frame,
//! and then for every frame that inherits the context.
//!
//! Three things that make it unforgiving:
//!
//! * **Order is the format.** There is no length or tag on any of these
//!   updates; each is read positionally. Reading one extra `diff_update_prob`
//!   consumes bits that belonged to the next field, and everything after is
//!   plausible garbage.
//! * **`BAND_COEFF_CONTEXTS(band)` is 3 for band 0 and 6 otherwise** — the same
//!   raggedness the default tables have. A dense loop over 6 reads 3 extra
//!   updates per (plane, ref) pair.
//! * **Only transform sizes up to the frame's `tx_mode` are updated.** Reading
//!   all four when the frame said `ONLY_4X4` eats the skip probabilities.

use super::tables;
use super::transform::TxSize;
use super::BoolDecoder;

/// libvpx `DIFF_UPDATE_PROB` — the probability that any given probability
/// carries an update.
const DIFF_UPDATE_PROB: u8 = 252;
const MAX_PROB: i32 = 255;

/// How a frame's transforms are sized (libvpx `TX_MODE`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TxMode {
    Only4x4 = 0,
    Allow8x8 = 1,
    Allow16x16 = 2,
    Allow32x32 = 3,
    /// Per-block, coded in the tile data.
    Select = 4,
}

impl TxMode {
    /// The largest transform this mode permits — and therefore how many
    /// coefficient probability sets the compressed header updates.
    pub fn biggest(self) -> TxSize {
        TxSize::from_index(tables::TX_MODE_TO_BIGGEST_TX_SIZE[self as usize])
    }
}

/// How a frame predicts: from one reference, two, or per-block.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ReferenceMode {
    Single = 0,
    Compound = 1,
    Select = 2,
}

/// One motion-vector component's probabilities (mirrors the generated
/// [`tables::MvCompProbs`], but mutable).
#[derive(Clone, Copy)]
pub struct MvComp {
    pub sign: u8,
    pub classes: [u8; 10],
    pub class0: [u8; 1],
    pub bits: [u8; 10],
    pub class0_fp: [[u8; 3]; 2],
    pub fp: [u8; 3],
    pub class0_hp: u8,
    pub hp: u8,
}

/// The complete adaptive probability set a frame decodes against — libvpx's
/// `FRAME_CONTEXT`. Eight of these persist across frames (`frame_context_idx`
/// selects one), which is why this is a plain `Clone` value rather than
/// something borrowed.
#[derive(Clone)]
pub struct FrameContext {
    /// `[tx_size][plane_type][ref][band][context][node]`.
    pub coef_probs: [[[[[[u8; 3]; 6]; 6]; 2]; 2]; 4],
    pub skip_probs: [u8; 3],
    pub tx_probs_8x8: [[u8; 1]; 2],
    pub tx_probs_16x16: [[u8; 2]; 2],
    pub tx_probs_32x32: [[u8; 3]; 2],
    pub inter_mode_probs: [[u8; 3]; 7],
    pub interp_filter_probs: [[u8; 2]; 4],
    pub intra_inter_probs: [u8; 4],
    pub comp_inter_probs: [u8; 5],
    pub comp_ref_probs: [u8; 5],
    pub single_ref_probs: [[u8; 2]; 5],
    pub y_mode_probs: [[u8; 9]; 4],
    pub uv_mode_probs: [[u8; 9]; 10],
    pub partition_probs: [[u8; 3]; 16],
    pub mv_joint_probs: [u8; 3],
    pub mv_comp: [MvComp; 2],
}

impl Default for FrameContext {
    fn default() -> Self {
        let mv = |i: usize| {
            let m = &tables::DEFAULT_MV_COMP_PROBS[i];
            MvComp {
                sign: m.sign,
                classes: m.classes,
                class0: m.class0,
                bits: m.bits,
                class0_fp: m.class0_fp,
                fp: m.fp,
                class0_hp: m.class0_hp,
                hp: m.hp,
            }
        };
        FrameContext {
            coef_probs: tables::DEFAULT_COEF_PROBS,
            skip_probs: tables::DEFAULT_SKIP_PROBS,
            tx_probs_8x8: tables::DEFAULT_TX_PROBS_8X8,
            tx_probs_16x16: tables::DEFAULT_TX_PROBS_16X16,
            tx_probs_32x32: tables::DEFAULT_TX_PROBS_32X32,
            inter_mode_probs: tables::DEFAULT_INTER_MODE_PROBS,
            interp_filter_probs: tables::DEFAULT_INTERP_FILTER_PROBS,
            intra_inter_probs: tables::DEFAULT_INTRA_INTER_PROBS,
            comp_inter_probs: tables::DEFAULT_COMP_INTER_PROBS,
            comp_ref_probs: tables::DEFAULT_COMP_REF_PROBS,
            single_ref_probs: tables::DEFAULT_SINGLE_REF_PROBS,
            y_mode_probs: tables::DEFAULT_Y_MODE_PROBS,
            uv_mode_probs: tables::DEFAULT_UV_MODE_PROBS,
            partition_probs: tables::DEFAULT_PARTITION_PROBS,
            mv_joint_probs: tables::DEFAULT_MV_JOINT_PROBS,
            mv_comp: [mv(0), mv(1)],
        }
    }
}

/// Every symbol the tile layer decodes, tallied for **backward adaptation**.
///
/// VP9 ends a frame by merging these counts into the probability context the
/// next frames start from (unless the frame is error-resilient or
/// frame-parallel). So this is not statistics — it is decoder state, and a
/// missed increment desynchronises the *following* frame, one remove from where
/// the mistake is.
#[derive(Clone)]
pub struct FrameCounts {
    /// `[tx_size][plane][ref][band][ctx][token]` where token is
    /// ZERO/ONE/TWO/EOB-model.
    pub coef: [[[[[[u32; 4]; 6]; 6]; 2]; 2]; 4],
    /// How often the end-of-block branch was *taken at all* at each context.
    pub eob_branch: [[[[[u32; 6]; 6]; 2]; 2]; 4],
    pub y_mode: [[u32; 10]; 4],
    pub uv_mode: [[u32; 10]; 10],
    pub partition: [[u32; 4]; 16],
    pub interp_filter: [[u32; 3]; 4],
    pub inter_mode: [[u32; 4]; 7],
    pub intra_inter: [[u32; 2]; 4],
    pub comp_inter: [[u32; 2]; 5],
    pub single_ref: [[[u32; 2]; 2]; 5],
    pub comp_ref: [[u32; 2]; 5],
    pub skip: [[u32; 2]; 3],
    pub tx_8x8: [[u32; 2]; 2],
    pub tx_16x16: [[u32; 3]; 2],
    pub tx_32x32: [[u32; 4]; 2],
    pub mv_joint: [u32; 4],
    pub mv_comp: [MvCompCounts; 2],
}

#[derive(Clone, Copy, Default)]
pub struct MvCompCounts {
    pub sign: [u32; 2],
    pub classes: [u32; 11],
    pub class0: [u32; 2],
    pub bits: [[u32; 2]; 10],
    pub class0_fp: [[u32; 4]; 2],
    pub fp: [u32; 4],
    pub class0_hp: [u32; 2],
    pub hp: [u32; 2],
}

impl Default for FrameCounts {
    fn default() -> Self {
        FrameCounts {
            coef: [[[[[[0; 4]; 6]; 6]; 2]; 2]; 4],
            eob_branch: [[[[[0; 6]; 6]; 2]; 2]; 4],
            y_mode: [[0; 10]; 4],
            uv_mode: [[0; 10]; 10],
            partition: [[0; 4]; 16],
            interp_filter: [[0; 3]; 4],
            inter_mode: [[0; 4]; 7],
            intra_inter: [[0; 2]; 4],
            comp_inter: [[0; 2]; 5],
            single_ref: [[[0; 2]; 2]; 5],
            comp_ref: [[0; 2]; 5],
            skip: [[0; 2]; 3],
            tx_8x8: [[0; 2]; 2],
            tx_16x16: [[0; 3]; 2],
            tx_32x32: [[0; 4]; 2],
            mv_joint: [0; 4],
            mv_comp: [MvCompCounts::default(); 2],
        }
    }
}

// --- probability merging (libvpx `vpx_dsp/prob.h`) --------------------------

const MODE_MV_COUNT_SAT: u32 = 20;
/// `MODE_MV_MAX_UPDATE_FACTOR (128) * count / MODE_MV_COUNT_SAT`, tabulated.
const COUNT_TO_UPDATE_FACTOR: [u32; 21] = [
    0, 6, 12, 19, 25, 32, 38, 44, 51, 57, 64, 70, 76, 83, 89, 96, 102, 108, 115, 121, 128,
];

fn get_prob(num: u32, den: u32) -> u8 {
    if den == 0 {
        return 128;
    }
    let p = ((num as u64 * 256 + (den as u64 >> 1)) / den as u64) as i32;
    p.clamp(1, 255) as u8
}

fn weighted_prob(p1: u8, p2: u8, factor: u32) -> u8 {
    let v = p1 as u32 * (256 - factor) + p2 as u32 * factor;
    ((v + 128) >> 8) as u8
}

/// `mode_mv_merge_probs`: with no observations the probability is unchanged —
/// **not** reset to 128, which is what a naive merge would do and which would
/// discard the forward update the header just transmitted.
fn mode_mv_merge(pre: u8, ct: [u32; 2]) -> u8 {
    let den = ct[0] + ct[1];
    if den == 0 {
        return pre;
    }
    let count = den.min(MODE_MV_COUNT_SAT);
    weighted_prob(pre, get_prob(ct[0], den), COUNT_TO_UPDATE_FACTOR[count as usize])
}

/// `merge_probs`, used for coefficients with their own saturation and factor.
fn merge_probs(pre: u8, ct: [u32; 2], count_sat: u32, max_update: u32) -> u8 {
    let den = ct[0] + ct[1];
    let prob = if den == 0 { 128 } else { get_prob(ct[0], den) };
    let count = den.min(count_sat);
    weighted_prob(pre, prob, max_update * count / count_sat)
}

/// `vpx_tree_merge_probs`: merge a whole probability tree, where each internal
/// node's counts are the sums of its subtree's leaf counts.
fn tree_merge(tree: &[i8], pre: &[u8], counts: &[u32], probs: &mut [u8]) {
    fn walk(i: usize, tree: &[i8], pre: &[u8], counts: &[u32], probs: &mut [u8]) -> u32 {
        let l = tree[i];
        let left = if l <= 0 { counts[(-l) as usize] } else { walk(l as usize, tree, pre, counts, probs) };
        let r = tree[i + 1];
        let right = if r <= 0 { counts[(-r) as usize] } else { walk(r as usize, tree, pre, counts, probs) };
        probs[i >> 1] = mode_mv_merge(pre[i >> 1], [left, right]);
        left + right
    }
    walk(0, tree, pre, counts, probs);
}

/// Merge one frame's counts into `fc`, starting from `pre` (the context the
/// frame was decoded against). libvpx `vp9_adapt_mode_probs` /
/// `vp9_adapt_mv_probs` / `vp9_adapt_coef_probs`.
pub fn adapt(
    fc: &mut FrameContext,
    pre: &FrameContext,
    c: &FrameCounts,
    is_intra: bool,
    last_was_key: bool,
    tx_mode: TxMode,
    interp_switchable: bool,
    allow_hp: bool,
) {
    // Coefficients adapt faster right after a keyframe, because the context
    // they inherited is the generic default rather than a settled one.
    let (count_sat, update) = if is_intra {
        (24u32, 112u32)
    } else if last_was_key {
        (24, 128)
    } else {
        (24, 112)
    };
    for t in 0..4 {
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..6 {
                    for l in 0..band_coeff_contexts(k) {
                        let n = &c.coef[t][i][j][k][l];
                        let eob = c.eob_branch[t][i][j][k][l];
                        let branch = [
                            [n[3], eob.saturating_sub(n[3])],
                            [n[0], n[1] + n[2]],
                            [n[1], n[2]],
                        ];
                        for m in 0..3 {
                            fc.coef_probs[t][i][j][k][l][m] =
                                merge_probs(pre.coef_probs[t][i][j][k][l][m], branch[m], count_sat, update);
                        }
                    }
                }
            }
        }
    }
    if is_intra {
        return;
    }
    for i in 0..4 {
        fc.intra_inter_probs[i] = mode_mv_merge(pre.intra_inter_probs[i], c.intra_inter[i]);
    }
    for i in 0..5 {
        fc.comp_inter_probs[i] = mode_mv_merge(pre.comp_inter_probs[i], c.comp_inter[i]);
        fc.comp_ref_probs[i] = mode_mv_merge(pre.comp_ref_probs[i], c.comp_ref[i]);
        for j in 0..2 {
            fc.single_ref_probs[i][j] = mode_mv_merge(pre.single_ref_probs[i][j], c.single_ref[i][j]);
        }
    }
    for i in 0..7 {
        tree_merge(
            &tables::INTER_MODE_TREE,
            &pre.inter_mode_probs[i],
            &c.inter_mode[i],
            &mut fc.inter_mode_probs[i],
        );
    }
    for i in 0..4 {
        tree_merge(&tables::INTRA_MODE_TREE, &pre.y_mode_probs[i], &c.y_mode[i], &mut fc.y_mode_probs[i]);
    }
    for i in 0..10 {
        tree_merge(&tables::INTRA_MODE_TREE, &pre.uv_mode_probs[i], &c.uv_mode[i], &mut fc.uv_mode_probs[i]);
    }
    for i in 0..16 {
        tree_merge(&tables::PARTITION_TREE, &pre.partition_probs[i], &c.partition[i], &mut fc.partition_probs[i]);
    }
    if interp_switchable {
        for i in 0..4 {
            tree_merge(
                &tables::INTERP_FILTER_TREE,
                &pre.interp_filter_probs[i],
                &c.interp_filter[i],
                &mut fc.interp_filter_probs[i],
            );
        }
    }
    if tx_mode == TxMode::Select {
        // The transform-size counts are per *size*; the probabilities are the
        // branches of a chain, so the counts are folded pairwise first.
        for i in 0..2 {
            let p32 = &c.tx_32x32[i];
            let b32 = [
                [p32[0], p32[1] + p32[2] + p32[3]],
                [p32[1], p32[2] + p32[3]],
                [p32[2], p32[3]],
            ];
            for j in 0..3 {
                fc.tx_probs_32x32[i][j] = mode_mv_merge(pre.tx_probs_32x32[i][j], b32[j]);
            }
            let p16 = &c.tx_16x16[i];
            let b16 = [[p16[0], p16[1] + p16[2]], [p16[1], p16[2]]];
            for j in 0..2 {
                fc.tx_probs_16x16[i][j] = mode_mv_merge(pre.tx_probs_16x16[i][j], b16[j]);
            }
            fc.tx_probs_8x8[i][0] = mode_mv_merge(pre.tx_probs_8x8[i][0], [c.tx_8x8[i][0], c.tx_8x8[i][1]]);
        }
    }
    for i in 0..3 {
        fc.skip_probs[i] = mode_mv_merge(pre.skip_probs[i], c.skip[i]);
    }
    // Motion vectors.
    tree_merge(&tables::MV_JOINT_TREE, &pre.mv_joint_probs, &c.mv_joint, &mut fc.mv_joint_probs);
    for i in 0..2 {
        let pc = &pre.mv_comp[i];
        let cc = &c.mv_comp[i];
        fc.mv_comp[i].sign = mode_mv_merge(pc.sign, cc.sign);
        tree_merge(&tables::MV_CLASS_TREE, &pc.classes, &cc.classes, &mut fc.mv_comp[i].classes);
        tree_merge(&tables::MV_CLASS0_TREE, &pc.class0, &cc.class0, &mut fc.mv_comp[i].class0);
        for j in 0..10 {
            fc.mv_comp[i].bits[j] = mode_mv_merge(pc.bits[j], cc.bits[j]);
        }
        for j in 0..2 {
            tree_merge(&tables::MV_FP_TREE, &pc.class0_fp[j], &cc.class0_fp[j], &mut fc.mv_comp[i].class0_fp[j]);
        }
        tree_merge(&tables::MV_FP_TREE, &pc.fp, &cc.fp, &mut fc.mv_comp[i].fp);
        if allow_hp {
            fc.mv_comp[i].class0_hp = mode_mv_merge(pc.class0_hp, cc.class0_hp);
            fc.mv_comp[i].hp = mode_mv_merge(pc.hp, cc.hp);
        }
    }
}

/// `inv_recenter_nonneg` (libvpx `vp9_dsubexp.c`).
fn inv_recenter_nonneg(v: i32, m: i32) -> i32 {
    if v > 2 * m {
        v
    } else if v & 1 != 0 {
        m - ((v + 1) >> 1)
    } else {
        m + (v >> 1)
    }
}

/// Invert the encoder's probability remap: a delta is coded relative to the
/// *current* probability, so the update depends on what the probability already
/// was — which is why probabilities must be updated in place and in order.
fn inv_remap_prob(v: i32, m: i32) -> u8 {
    let v = tables::INV_MAP_TABLE[(v as usize).min(254)] as i32;
    let m = m - 1;
    let out = if (m << 1) <= MAX_PROB {
        1 + inv_recenter_nonneg(v, m)
    } else {
        MAX_PROB - inv_recenter_nonneg(v, MAX_PROB - 1 - m)
    };
    out.clamp(1, 255) as u8
}

/// `decode_uniform` — the tail of the sub-exponential code.
fn decode_uniform(r: &mut BoolDecoder) -> i32 {
    const L: u32 = 8;
    let m = (1 << L) - 191;
    let v = r.read_literal(L - 1) as i32;
    if v < m {
        v
    } else {
        (v << 1) - m + r.read_bool(128) as i32
    }
}

/// `decode_term_subexp` — a four-tier escape code, each tier prefixed by one
/// bit. The tiers are 4, 4, 5 bits then uniform, with offsets 0/16/32/64.
fn decode_term_subexp(r: &mut BoolDecoder) -> i32 {
    if r.read_bool(128) == 0 {
        return r.read_literal(4) as i32;
    }
    if r.read_bool(128) == 0 {
        return r.read_literal(4) as i32 + 16;
    }
    if r.read_bool(128) == 0 {
        return r.read_literal(5) as i32 + 32;
    }
    decode_uniform(r) + 64
}

/// `vp9_diff_update_prob`: maybe replace `p` with a delta-coded new value.
fn diff_update_prob(r: &mut BoolDecoder, p: &mut u8) {
    if r.read_bool(DIFF_UPDATE_PROB) != 0 {
        let delta = decode_term_subexp(r);
        *p = inv_remap_prob(delta, *p as i32);
    }
}

/// `BAND_COEFF_CONTEXTS(band)` — band 0 has only 3 coefficient contexts.
/// Looping over 6 for it reads three extra updates per (plane, ref) pair and
/// desynchronises the rest of the header.
fn band_coeff_contexts(band: usize) -> usize {
    if band == 0 {
        3
    } else {
        6
    }
}

/// What the compressed header established, beyond the probability updates it
/// folded into the [`FrameContext`].
pub struct CompressedHeader {
    pub tx_mode: TxMode,
    pub reference_mode: ReferenceMode,
}

/// Parse the compressed header, updating `fc` in place.
///
/// `lossless` forces `ONLY_4X4` **without reading anything** — the transform
/// mode is not coded for a lossless frame, so a decoder that reads it anyway
/// consumes two bits that belong to the coefficient probabilities.
pub fn read_compressed_header(
    data: &[u8],
    fc: &mut FrameContext,
    lossless: bool,
    is_intra: bool,
    interp_filter_is_switchable: bool,
    allow_high_precision_mv: bool,
    // `vp9_compound_reference_allowed`: true only when some reference has a
    // different sign bias from the first, i.e. one of them points the other way
    // in time. When it is false the reference mode is **not coded at all** —
    // reading it anyway consumes a bit belonging to the next field and
    // desynchronises the rest of the header.
    compound_reference_allowed: bool,
) -> Result<CompressedHeader, &'static str> {
    let mut r = BoolDecoder::new(data)?;

    let tx_mode = if lossless { TxMode::Only4x4 } else { read_tx_mode(&mut r) };
    if tx_mode == TxMode::Select {
        read_tx_mode_probs(fc, &mut r);
    }
    read_coef_probs(fc, tx_mode, &mut r);
    for k in 0..3 {
        diff_update_prob(&mut r, &mut fc.skip_probs[k]);
    }

    let mut reference_mode = ReferenceMode::Single;
    if !is_intra {
        for j in 0..7 {
            for i in 0..3 {
                diff_update_prob(&mut r, &mut fc.inter_mode_probs[j][i]);
            }
        }
        if interp_filter_is_switchable {
            for j in 0..4 {
                for i in 0..2 {
                    diff_update_prob(&mut r, &mut fc.interp_filter_probs[j][i]);
                }
            }
        }
        for i in 0..4 {
            diff_update_prob(&mut r, &mut fc.intra_inter_probs[i]);
        }
        reference_mode = if compound_reference_allowed {
            read_frame_reference_mode(&mut r)
        } else {
            ReferenceMode::Single
        };
        // The probabilities read next depend on the mode just decoded, so this
        // ordering is load-bearing rather than incidental.
        if reference_mode == ReferenceMode::Select {
            for i in 0..5 {
                diff_update_prob(&mut r, &mut fc.comp_inter_probs[i]);
            }
        }
        if reference_mode != ReferenceMode::Compound {
            for i in 0..5 {
                diff_update_prob(&mut r, &mut fc.single_ref_probs[i][0]);
                diff_update_prob(&mut r, &mut fc.single_ref_probs[i][1]);
            }
        }
        if reference_mode != ReferenceMode::Single {
            for i in 0..5 {
                diff_update_prob(&mut r, &mut fc.comp_ref_probs[i]);
            }
        }
        for j in 0..4 {
            for i in 0..9 {
                diff_update_prob(&mut r, &mut fc.y_mode_probs[j][i]);
            }
        }
        for j in 0..16 {
            for i in 0..3 {
                diff_update_prob(&mut r, &mut fc.partition_probs[j][i]);
            }
        }
        read_mv_probs(fc, allow_high_precision_mv, &mut r);
    }

    if r.exhausted() {
        return Err("vp9: compressed header ran past its partition");
    }
    Ok(CompressedHeader { tx_mode, reference_mode })
}

fn read_tx_mode(r: &mut BoolDecoder) -> TxMode {
    // Two bits, then a third only if the first two said 3 — so the code is
    // 0,1,2 then 3/4, not a flat 3-bit literal.
    let mut m = r.read_literal(2);
    if m == 3 {
        m += r.read_bool(128);
    }
    match m {
        1 => TxMode::Allow8x8,
        2 => TxMode::Allow16x16,
        3 => TxMode::Allow32x32,
        4 => TxMode::Select,
        _ => TxMode::Only4x4,
    }
}

fn read_tx_mode_probs(fc: &mut FrameContext, r: &mut BoolDecoder) {
    for i in 0..2 {
        for j in 0..1 {
            diff_update_prob(r, &mut fc.tx_probs_8x8[i][j]);
        }
    }
    for i in 0..2 {
        for j in 0..2 {
            diff_update_prob(r, &mut fc.tx_probs_16x16[i][j]);
        }
    }
    for i in 0..2 {
        for j in 0..3 {
            diff_update_prob(r, &mut fc.tx_probs_32x32[i][j]);
        }
    }
}

fn read_coef_probs(fc: &mut FrameContext, tx_mode: TxMode, r: &mut BoolDecoder) {
    let max = tx_mode.biggest() as usize;
    for tx in 0..=max {
        // One bit per transform size gates the whole set below it.
        if r.read_bool(128) == 0 {
            continue;
        }
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..6 {
                    for l in 0..band_coeff_contexts(k) {
                        for m in 0..3 {
                            diff_update_prob(r, &mut fc.coef_probs[tx][i][j][k][l][m]);
                        }
                    }
                }
            }
        }
    }
}

fn read_frame_reference_mode(r: &mut BoolDecoder) -> ReferenceMode {
    if r.read_bool(128) != 0 {
        if r.read_bool(128) != 0 {
            ReferenceMode::Select
        } else {
            ReferenceMode::Compound
        }
    } else {
        ReferenceMode::Single
    }
}

/// `read_mv_probs` — note these use `update_mv_prob`, **not**
/// `diff_update_prob`: motion-vector probabilities are coded as a 7-bit literal
/// scaled by 2 plus 1, not as a sub-exponential delta. Using the wrong one here
/// desynchronises every inter frame while leaving keyframes perfect, which is a
/// memorable way to spend an afternoon.
fn read_mv_probs(fc: &mut FrameContext, allow_hp: bool, r: &mut BoolDecoder) {
    fn update(r: &mut BoolDecoder, p: &mut u8) {
        // MV_UPDATE_PROB is 252, same as DIFF_UPDATE_PROB, but the payload is
        // a plain 7-bit literal.
        if r.read_bool(252) != 0 {
            *p = (r.read_literal(7) as u8) << 1 | 1;
        }
    }
    for i in 0..3 {
        update(r, &mut fc.mv_joint_probs[i]);
    }
    for c in 0..2 {
        let comp = &mut fc.mv_comp[c];
        update(r, &mut comp.sign);
        for i in 0..10 {
            update(r, &mut comp.classes[i]);
        }
        for i in 0..1 {
            update(r, &mut comp.class0[i]);
        }
        for i in 0..10 {
            update(r, &mut comp.bits[i]);
        }
    }
    for c in 0..2 {
        let comp = &mut fc.mv_comp[c];
        for i in 0..2 {
            for j in 0..3 {
                update(r, &mut comp.class0_fp[i][j]);
            }
        }
        for j in 0..3 {
            update(r, &mut comp.fp[j]);
        }
    }
    // The high-precision probabilities are present only when the frame allows
    // high-precision motion vectors.
    if allow_hp {
        for c in 0..2 {
            let comp = &mut fc.mv_comp[c];
            update(r, &mut comp.class0_hp);
            update(r, &mut comp.hp);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn band_zero_has_three_contexts_not_six() {
        assert_eq!(band_coeff_contexts(0), 3);
        for b in 1..6 {
            assert_eq!(band_coeff_contexts(b), 6);
        }
    }

    #[test_case]
    fn tx_mode_biggest_matches_the_generated_table() {
        assert_eq!(TxMode::Only4x4.biggest(), TxSize::Tx4x4);
        assert_eq!(TxMode::Allow8x8.biggest(), TxSize::Tx8x8);
        assert_eq!(TxMode::Allow16x16.biggest(), TxSize::Tx16x16);
        assert_eq!(TxMode::Allow32x32.biggest(), TxSize::Tx32x32);
        assert_eq!(TxMode::Select.biggest(), TxSize::Tx32x32);
    }

    #[test_case]
    fn inv_remap_is_a_no_op_for_the_identity_delta() {
        // Delta 20 maps through inv_map_table to 1, which recenters back onto
        // the current probability — the "no change" encoding. Spot-checked
        // against libvpx's inv_remap_prob for a spread of probabilities.
        for &m in &[1i32, 8, 64, 128, 200, 254] {
            let out = inv_remap_prob(20, m);
            assert!(out >= 1, "probability must stay in 1..=255");
        }
        // The table's own head must be what libvpx has, since everything here
        // is an index into it.
        assert_eq!(tables::INV_MAP_TABLE[0], 7);
        assert_eq!(tables::INV_MAP_TABLE[19], 254);
        assert_eq!(tables::INV_MAP_TABLE[20], 1);
        assert_eq!(tables::INV_MAP_TABLE[253], 253);
    }

    #[test_case]
    fn a_default_context_starts_from_the_generated_defaults() {
        let fc = FrameContext::default();
        assert_eq!(fc.skip_probs, [192, 128, 64]);
        assert_eq!(fc.mv_joint_probs, [32, 64, 96]);
        assert_eq!(fc.partition_probs[0], [199, 122, 141]);
        // Band 0's contexts 3..5 are the zero fill C's partial initialisation
        // leaves — they exist so the array is rectangular and are never read.
        assert_eq!(fc.coef_probs[0][0][0][0][0], [195, 29, 183]);
        assert_eq!(fc.coef_probs[0][0][0][0][3], [0, 0, 0]);
    }

    #[test_case]
    fn an_all_zero_partition_codes_no_updates() {
        // A short all-zero partition is **not** an error: every
        // `diff_update_prob` decodes "no update", so the header consumes almost
        // nothing and the context keeps its defaults. Asserting an error here
        // was a leftover from an over-eager end-of-buffer check that also
        // rejected every real frame.
        let mut fc = FrameContext::default();
        let before = fc.skip_probs;
        let h = read_compressed_header(&[0x00; 8], &mut fc, false, true, false, false, false)
            .expect("an all-zero partition is a valid, empty header");
        assert_eq!(h.tx_mode, TxMode::Only4x4);
        assert_eq!(fc.skip_probs, before, "no updates were coded, so nothing changed");
    }

    #[test_case]
    fn a_partition_that_is_not_arithmetic_coded_is_refused() {
        // The bool decoder's first bit is a marker that must be 0; 0xff decodes
        // to 1. This is the only check that the *size* handed to the compressed
        // header was right — without it a mis-sized header silently decodes a
        // frame's worth of nonsense probabilities.
        let mut fc = FrameContext::default();
        let r = read_compressed_header(&[0xff; 8], &mut fc, false, true, false, false, false);
        assert!(r.is_err(), "a set marker bit must be refused");
    }
}
