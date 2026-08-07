//! HEVC inter prediction (H.265 §8.5): motion vector scaling, the merge
//! candidate list, and motion compensation.
//!
//! Motion compensation here works in a **14-bit intermediate**, not in
//! samples. Every fractional filter leaves its result at that precision and the
//! final shift happens once, in the uni- or bi-prediction combine — which is
//! why bi-prediction is more accurate than averaging two rounded predictions,
//! and why rounding early is a bug that looks like nothing at all on the first
//! frame and like a slow contrast drift over a GOP.
//!
//! The three shifts that make that work, at bit depth `B`:
//!
//! | stage | shift |
//! |---|---|
//! | integer position | `<< (14 - B)` |
//! | one fractional pass | `>> (B - 8)` |
//! | second pass of `hv` | `>> 6` |
//! | uni-prediction out | `>> (14 - B)`, rounded |
//! | bi-prediction out | `>> (15 - B)`, rounded |
//!
//! Getting any one of them wrong scales the whole prediction by a power of two,
//! which a decoder notices immediately — but getting the *rounding offset*
//! wrong biases it by half a level, which it does not.

use super::tables as tb;

/// The largest prediction block.
pub const MAX_PB: usize = 64;

/// A motion vector in quarter-pel luma units.
pub type Mv = (i16, i16);

/// One block's motion, as the merge list stores it.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct MvField {
    /// Bit 0 = list 0 used, bit 1 = list 1 used.
    pub pred_flag: u8,
    pub ref_idx: [i8; 2],
    pub mv: [Mv; 2],
}

impl MvField {
    #[inline]
    pub fn uses(&self, l: usize) -> bool {
        self.pred_flag & (1 << l) != 0
    }
    /// Two candidates are duplicates when both their vectors *and* their
    /// reference indices agree — comparing only the vectors merges two
    /// genuinely different predictions and shortens the list, shifting every
    /// index above it.
    #[inline]
    pub fn same_motion(&self, o: &MvField) -> bool {
        self.pred_flag == o.pred_flag && self.ref_idx == o.ref_idx && self.mv == o.mv
    }
}

/// Scale a motion vector between two temporal distances (§8.5.3.2.8).
///
/// `td` is the distance from the *neighbour's* picture to its reference and
/// `tb` the distance from the current picture to the target reference. The
/// awkward `+ 127 + (negative)` is a round-half-away-from-zero that a plain
/// arithmetic shift does not give — and biasing every scaled vector by half a
/// quarter-pel in one direction drifts a whole GOP of B-frames.
pub fn mv_scale(v: Mv, td: i32, tb_dist: i32) -> Mv {
    let td = td.clamp(-128, 127);
    let tb_dist = tb_dist.clamp(-128, 127);
    if td == 0 {
        return v;
    }
    let tx = (0x4000 + (td / 2).abs()) / td;
    let scale = ((tb_dist * tx + 32) >> 6).clamp(-(1 << 11), (1 << 11) - 1);
    let f = |c: i16| -> i16 {
        let p = scale * c as i32;
        ((p + 127 + (p < 0) as i32) >> 8).clamp(-32768, 32767) as i16
    };
    (f(v.0), f(v.1))
}

/// The order in which combined bi-predictive candidates pair existing ones
/// (§8.5.3.2.4, table 8-6). It is **not** every ordered pair in index order:
/// the table interleaves `(a, b)` with `(b, a)` so the cheapest combinations
/// come first, and a candidate list built in a different order is a different
/// picture at the same `merge_idx`.
const COMBO: [(usize, usize); 12] = [
    (0, 1),
    (1, 0),
    (0, 2),
    (2, 0),
    (1, 2),
    (2, 1),
    (0, 3),
    (3, 0),
    (1, 3),
    (3, 1),
    (2, 3),
    (3, 2),
];

/// The five spatial neighbour positions, in the order the specification tries
/// them. Note it is **A1, B1, B0, A0, B2** — not a clockwise sweep — and B2 is
/// only considered when fewer than four candidates were found.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SpatialPos {
    A1,
    B1,
    B0,
    A0,
    B2,
}

/// Inputs to the merge list: each spatial neighbour's motion, or `None` where
/// it is unavailable (outside the picture, in another slice, intra-coded, in a
/// different merge estimation region, or the same PU's other half).
pub struct MergeNeighbours {
    pub a1: Option<MvField>,
    pub b1: Option<MvField>,
    pub b0: Option<MvField>,
    pub a0: Option<MvField>,
    pub b2: Option<MvField>,
}

/// Build the merge candidate list (§8.5.3.2.1).
///
/// `temporal` is the collocated candidate, already derived and scaled.
/// `is_b` selects a B slice (which alone gets combined bi-predictive
/// candidates). `nb_refs` bounds the zero-candidate reference indices.
///
/// The pruning is the part to read carefully: it is **not** "compare against
/// everything already in the list". Each position has a fixed, short list of
/// predecessors it is compared with — B1 against A1, B0 against B1, A0 against
/// A1, B2 against both A1 and B1 — and comparing more than that drops
/// candidates the specification keeps, shifting every later index.
pub fn merge_candidates(
    n: &MergeNeighbours,
    temporal: Option<MvField>,
    is_b: bool,
    max_cand: usize,
    nb_refs: usize,
) -> alloc::vec::Vec<MvField> {
    let mut list: alloc::vec::Vec<MvField> = alloc::vec::Vec::with_capacity(max_cand);

    if let Some(a1) = n.a1 {
        list.push(a1);
    }
    if let Some(b1) = n.b1 {
        if !(n.a1.is_some() && b1.same_motion(&n.a1.unwrap())) {
            list.push(b1);
        }
    }
    if let Some(b0) = n.b0 {
        if !(n.b1.is_some() && b0.same_motion(&n.b1.unwrap())) {
            list.push(b0);
        }
    }
    if let Some(a0) = n.a0 {
        if !(n.a1.is_some() && a0.same_motion(&n.a1.unwrap())) {
            list.push(a0);
        }
    }
    if let Some(b2) = n.b2 {
        // B2 is skipped once four candidates exist — not because the list is
        // full (it may hold five) but because the specification says so.
        if !(n.a1.is_some() && b2.same_motion(&n.a1.unwrap()))
            && !(n.b1.is_some() && b2.same_motion(&n.b1.unwrap()))
            && list.len() != 4
        {
            list.push(b2);
        }
    }

    // FFmpeg returns the moment `merge_idx` is reached, so its list never
    // exceeds the bound; building the whole list instead means truncating
    // here. Without this a small `max_num_merge_cand` (which is exactly what a
    // low-latency encoder sets) yields a list longer than the index space,
    // and every candidate the encoder meant is still present — so the picture
    // is right until the temporal or zero candidates matter, and then is not.
    list.truncate(max_cand);

    if let Some(t) = temporal {
        if list.len() < max_cand {
            list.push(t);
        }
    }

    // Combined bi-predictive candidates, B slices only.
    let orig = list.len();
    if is_b && orig > 1 && orig < max_cand {
        for &(i0, i1) in COMBO.iter().take(orig * (orig - 1)) {
            if list.len() >= max_cand {
                break;
            }
            let l0 = list[i0];
            let l1 = list[i1];
            if l0.uses(0) && l1.uses(1) && (l0.ref_idx[0] != l1.ref_idx[1] || l0.mv[0] != l1.mv[1])
            {
                list.push(MvField {
                    pred_flag: 3,
                    ref_idx: [l0.ref_idx[0], l1.ref_idx[1]],
                    mv: [l0.mv[0], l1.mv[1]],
                });
            }
        }
    }

    // Zero candidates, each stepping the reference index until it runs out.
    let mut zero_idx = 0usize;
    while list.len() < max_cand {
        let r = if zero_idx < nb_refs { zero_idx as i8 } else { 0 };
        list.push(MvField {
            pred_flag: if is_b { 3 } else { 1 },
            ref_idx: [r, r],
            mv: [(0, 0); 2],
        });
        zero_idx += 1;
    }
    list
}

/// One neighbour as AMVP sees it: its motion, and what each of its lists
/// actually points at.
#[derive(Clone, Copy, Debug)]
pub struct AmvpNeighbour {
    pub mvf: MvField,
    /// POC of the picture each list references.
    pub ref_poc: [i32; 2],
    pub long_term: [bool; 2],
}

impl AmvpNeighbour {
    /// The "same picture" test: this neighbour predicts from list `l`, and that
    /// list names exactly the picture we want. The vector is then usable with
    /// no scaling at all.
    fn exact(&self, l: usize, target_poc: i32) -> Option<Mv> {
        (self.mvf.uses(l) && self.ref_poc[l] == target_poc).then(|| self.mvf.mv[l])
    }

    /// The "scale it" test: this neighbour predicts from list `l` and that
    /// reference has the **same long-term-ness** as the target — which is the
    /// gate, not the POC. A long-term reference has no meaningful temporal
    /// distance, so mixing the two kinds cannot be scaled and the candidate is
    /// refused rather than scaled by a nonsense ratio.
    fn scaled(&self, l: usize, target_poc: i32, target_lt: bool, cur_poc: i32) -> Option<Mv> {
        if !self.mvf.uses(l) || self.long_term[l] != target_lt {
            return None;
        }
        if target_lt {
            // Both long-term: taken as-is.
            return Some(self.mvf.mv[l]);
        }
        Some(mv_scale(self.mvf.mv[l], cur_poc - self.ref_poc[l], cur_poc - target_poc))
    }
}

/// Build the two AMVP predictor candidates (§8.5.3.2.6).
///
/// Two things here are not obvious from the shape of the code:
///
/// - **Each position is tried in list `lx` first, then the other list.** A
///   neighbour predicting the same picture through the *opposite* list is still
///   a valid predictor — the picture is what matters, not the slot.
/// - **`is_scaled` gates a whole second pass.** When neither left neighbour
///   exists, the B candidate is *promoted* into A's place and B is then
///   re-derived allowing scaling. That rearrangement is easy to miss because
///   the straightforward reading already produces two candidates; missing it
///   gives the right *number* of predictors and the wrong ones, at every
///   prediction unit on the left edge of a CTB row.
pub fn amvp_candidates(
    a0: Option<AmvpNeighbour>,
    a1: Option<AmvpNeighbour>,
    b0: Option<AmvpNeighbour>,
    b1: Option<AmvpNeighbour>,
    b2: Option<AmvpNeighbour>,
    lx: usize,
    target_poc: i32,
    target_lt: bool,
    cur_poc: i32,
    temporal: Option<Mv>,
) -> [Mv; 2] {
    let ly = 1 - lx;
    let exact_of = |n: &Option<AmvpNeighbour>| -> Option<Mv> {
        let n = n.as_ref()?;
        n.exact(lx, target_poc).or_else(|| n.exact(ly, target_poc))
    };
    let scaled_of = |n: &Option<AmvpNeighbour>| -> Option<Mv> {
        let n = n.as_ref()?;
        n.scaled(lx, target_poc, target_lt, cur_poc)
            .or_else(|| n.scaled(ly, target_poc, target_lt, cur_poc))
    };

    // A: exact from A0 then A1, then scaled from A0 then A1.
    let mut mx_a = exact_of(&a0)
        .or_else(|| exact_of(&a1))
        .or_else(|| scaled_of(&a0))
        .or_else(|| scaled_of(&a1));
    // B: exact only, from B0, B1, B2.
    let mut mx_b = exact_of(&b0).or_else(|| exact_of(&b1)).or_else(|| exact_of(&b2));

    // The left column existing at all is what allows B to stay unscaled.
    let is_scaled = a0.is_some() || a1.is_some();
    if !is_scaled {
        if mx_a.is_none() {
            mx_a = mx_b;
        }
        mx_b = scaled_of(&b0).or_else(|| scaled_of(&b1)).or_else(|| scaled_of(&b2));
    }

    let mut out: alloc::vec::Vec<Mv> = alloc::vec::Vec::with_capacity(2);
    if let Some(a) = mx_a {
        out.push(a);
    }
    if let Some(b) = mx_b {
        // B is dropped only when it duplicates A — never against the temporal
        // or zero candidates, which are appended unconditionally.
        if mx_a != Some(b) {
            out.push(b);
        }
    }
    if out.len() < 2 {
        if let Some(t) = temporal {
            out.push(t);
        }
    }
    while out.len() < 2 {
        out.push((0, 0));
    }
    [out[0], out[1]]
}

/// A reference picture plane, with the edge clamping HEVC requires: a motion
/// vector may point outside the picture, and the samples it names are the
/// nearest real ones.
pub struct Plane<'a> {
    pub data: &'a [u8],
    pub stride: usize,
    pub width: usize,
    pub height: usize,
}

impl<'a> Plane<'a> {
    #[inline]
    fn at(&self, x: i32, y: i32) -> i32 {
        let x = x.clamp(0, self.width as i32 - 1) as usize;
        let y = y.clamp(0, self.height as i32 - 1) as usize;
        self.data[y * self.stride + x] as i32
    }
}

/// Luma motion compensation into the 14-bit intermediate.
///
/// `(x0, y0)` is the integer part of the block position and `(mx, my)` the
/// quarter-pel phases. `dst` has stride [`MAX_PB`].
pub fn put_luma(
    dst: &mut [i16],
    src: &Plane,
    x0: i32,
    y0: i32,
    w: usize,
    h: usize,
    mx: usize,
    my: usize,
    bit_depth: u32,
) {
    let up = 14 - bit_depth;
    let down = bit_depth - 8;
    if mx == 0 && my == 0 {
        for y in 0..h {
            for x in 0..w {
                dst[y * MAX_PB + x] = (src.at(x0 + x as i32, y0 + y as i32) << up) as i16;
            }
        }
        return;
    }
    let hf = &tb::QPEL_FILTERS[mx];
    let vf = &tb::QPEL_FILTERS[my];
    if my == 0 {
        for y in 0..h {
            for x in 0..w {
                let mut a = 0i32;
                for (t, &c) in hf.iter().enumerate() {
                    a += c as i32 * src.at(x0 + x as i32 + t as i32 - 3, y0 + y as i32);
                }
                dst[y * MAX_PB + x] = (a >> down) as i16;
            }
        }
        return;
    }
    if mx == 0 {
        for y in 0..h {
            for x in 0..w {
                let mut a = 0i32;
                for (t, &c) in vf.iter().enumerate() {
                    a += c as i32 * src.at(x0 + x as i32, y0 + y as i32 + t as i32 - 3);
                }
                dst[y * MAX_PB + x] = (a >> down) as i16;
            }
        }
        return;
    }
    // Separable: horizontal into a tall scratch (three rows above, four below),
    // then vertical over that. Filtering the *samples* twice instead of
    // filtering the intermediate loses the extra precision the first pass
    // produced, which is the whole point of the two-stage form.
    let mut tmp = [0i16; (MAX_PB + 7) * MAX_PB];
    for y in 0..h + 7 {
        for x in 0..w {
            let mut a = 0i32;
            for (t, &c) in hf.iter().enumerate() {
                a += c as i32 * src.at(x0 + x as i32 + t as i32 - 3, y0 + y as i32 - 3);
            }
            tmp[y * MAX_PB + x] = (a >> down) as i16;
        }
    }
    for y in 0..h {
        for x in 0..w {
            let mut a = 0i32;
            for (t, &c) in vf.iter().enumerate() {
                a += c as i32 * tmp[(y + t) * MAX_PB + x] as i32;
            }
            dst[y * MAX_PB + x] = (a >> 6) as i16;
        }
    }
}

/// Chroma motion compensation: the same shape with the 4-tap filter and
/// eighth-pel phases.
pub fn put_chroma(
    dst: &mut [i16],
    src: &Plane,
    x0: i32,
    y0: i32,
    w: usize,
    h: usize,
    mx: usize,
    my: usize,
    bit_depth: u32,
) {
    let up = 14 - bit_depth;
    let down = bit_depth - 8;
    if mx == 0 && my == 0 {
        for y in 0..h {
            for x in 0..w {
                dst[y * MAX_PB + x] = (src.at(x0 + x as i32, y0 + y as i32) << up) as i16;
            }
        }
        return;
    }
    let hf = &tb::EPEL_FILTERS[mx];
    let vf = &tb::EPEL_FILTERS[my];
    if my == 0 {
        for y in 0..h {
            for x in 0..w {
                let mut a = 0i32;
                for (t, &c) in hf.iter().enumerate() {
                    a += c as i32 * src.at(x0 + x as i32 + t as i32 - 1, y0 + y as i32);
                }
                dst[y * MAX_PB + x] = (a >> down) as i16;
            }
        }
        return;
    }
    if mx == 0 {
        for y in 0..h {
            for x in 0..w {
                let mut a = 0i32;
                for (t, &c) in vf.iter().enumerate() {
                    a += c as i32 * src.at(x0 + x as i32, y0 + y as i32 + t as i32 - 1);
                }
                dst[y * MAX_PB + x] = (a >> down) as i16;
            }
        }
        return;
    }
    let mut tmp = [0i16; (MAX_PB + 3) * MAX_PB];
    for y in 0..h + 3 {
        for x in 0..w {
            let mut a = 0i32;
            for (t, &c) in hf.iter().enumerate() {
                a += c as i32 * src.at(x0 + x as i32 + t as i32 - 1, y0 + y as i32 - 1);
            }
            tmp[y * MAX_PB + x] = (a >> down) as i16;
        }
    }
    for y in 0..h {
        for x in 0..w {
            let mut a = 0i32;
            for (t, &c) in vf.iter().enumerate() {
                a += c as i32 * tmp[(y + t) * MAX_PB + x] as i32;
            }
            dst[y * MAX_PB + x] = (a >> 6) as i16;
        }
    }
}

/// Round one prediction down to samples (§8.5.3.3.4.2, uni-prediction).
pub fn uni_pred(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[i16],
    w: usize,
    h: usize,
    bit_depth: u32,
) {
    let shift = 14 - bit_depth;
    let offset = 1i32 << (shift - 1);
    let max = (1i32 << bit_depth) - 1;
    for y in 0..h {
        for x in 0..w {
            let v = (src[y * MAX_PB + x] as i32 + offset) >> shift;
            dst[y * dst_stride + x] = v.clamp(0, max) as u8;
        }
    }
}

/// Average two predictions (bi-prediction). Note the shift is **one more**
/// than uni-prediction's, which is what makes this an average rather than a
/// sum — and note both inputs are still at 14 bits, so this is the only
/// rounding either of them ever sees.
pub fn bi_pred(
    dst: &mut [u8],
    dst_stride: usize,
    a: &[i16],
    b: &[i16],
    w: usize,
    h: usize,
    bit_depth: u32,
) {
    let shift = 15 - bit_depth;
    let offset = 1i32 << (shift - 1);
    let max = (1i32 << bit_depth) - 1;
    for y in 0..h {
        for x in 0..w {
            let v = (a[y * MAX_PB + x] as i32 + b[y * MAX_PB + x] as i32 + offset) >> shift;
            dst[y * dst_stride + x] = v.clamp(0, max) as u8;
        }
    }
}

/// Explicit weighted uni-prediction (§8.5.3.3.4.3).
pub fn uni_pred_weighted(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[i16],
    w: usize,
    h: usize,
    denom: u32,
    wx: i32,
    ox: i32,
    bit_depth: u32,
) {
    let shift = denom + 14 - bit_depth;
    let offset = if shift >= 1 { 1i32 << (shift - 1) } else { 0 };
    let ox = ox * (1 << (bit_depth - 8));
    let max = (1i32 << bit_depth) - 1;
    for y in 0..h {
        for x in 0..w {
            let v = (((src[y * MAX_PB + x] as i32 * wx + offset) >> shift) + ox).clamp(0, max);
            dst[y * dst_stride + x] = v as u8;
        }
    }
}

/// Explicit weighted bi-prediction (§8.5.3.3.4.3).
///
/// The offsets are folded together and applied **once**, at `log2_wd`
/// precision, which is why the rounding constant here is `(o0 + o1 + 1)` and
/// not two separate roundings — applying each list's offset separately double
/// counts the rounding and shifts the picture by a level.
pub fn bi_pred_weighted(
    dst: &mut [u8],
    dst_stride: usize,
    a: &[i16],
    b: &[i16],
    w: usize,
    h: usize,
    denom: u32,
    w0: i32,
    w1: i32,
    o0: i32,
    o1: i32,
    bit_depth: u32,
) {
    let shift = 14 + 1 - bit_depth;
    let log2_wd = denom + shift - 1;
    let o = (o0 + o1) * (1 << (bit_depth - 8)) + 1;
    let max = (1i32 << bit_depth) - 1;
    for y in 0..h {
        for x in 0..w {
            let v = (a[y * MAX_PB + x] as i32 * w0
                + b[y * MAX_PB + x] as i32 * w1
                + o * (1 << log2_wd))
                >> (log2_wd + 1);
            dst[y * dst_stride + x] = v.clamp(0, max) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn flat_plane(v: u8, w: usize, h: usize) -> alloc::vec::Vec<u8> {
        vec![v; w * h]
    }

    /// **The DC test**: a flat reference must reconstruct to that exact value
    /// through every fractional position, through uni- and bi-prediction, and
    /// through both plane types. It is the one property that pins all five
    /// shifts at once — every one of them scales the result by a power of two,
    /// and the rounding offsets show up as an off-by-one.
    #[test_case]
    fn a_flat_reference_survives_every_fractional_position_exactly() {
        for &v in &[0u8, 1, 37, 128, 254, 255] {
            let buf = flat_plane(v, 32, 32);
            let p = Plane { data: &buf, stride: 32, width: 32, height: 32 };
            for my in 0..4 {
                for mx in 0..4 {
                    let mut mid = vec![0i16; MAX_PB * MAX_PB];
                    put_luma(&mut mid, &p, 8, 8, 8, 8, mx, my, 8);
                    let mut out = vec![0u8; 64];
                    uni_pred(&mut out, 8, &mid, 8, 8, 8);
                    assert!(
                        out.iter().all(|&s| s == v),
                        "luma uni ({mx},{my}) value {v}: {:?}",
                        &out[..8]
                    );
                    // Bi-prediction of the same prediction twice must give the
                    // same answer as uni — that is what makes the extra shift
                    // an average and not a doubling.
                    let mut out2 = vec![0u8; 64];
                    bi_pred(&mut out2, 8, &mid, &mid, 8, 8, 8);
                    assert_eq!(out, out2, "bi != uni for luma ({mx},{my}) value {v}");
                }
            }
            for my in 0..8 {
                for mx in 0..8 {
                    let mut mid = vec![0i16; MAX_PB * MAX_PB];
                    put_chroma(&mut mid, &p, 8, 8, 8, 8, mx, my, 8);
                    let mut out = vec![0u8; 64];
                    uni_pred(&mut out, 8, &mid, 8, 8, 8);
                    assert!(out.iter().all(|&s| s == v), "chroma ({mx},{my}) value {v}");
                }
            }
        }
    }

    /// A motion vector pointing outside the picture reads the nearest real
    /// sample. Without that clamp a vector at the frame edge reads whatever
    /// follows the plane in memory — which decodes, and looks like noise
    /// creeping in from the border.
    #[test_case]
    fn a_vector_outside_the_picture_clamps_to_the_edge() {
        let mut buf = vec![0u8; 8 * 8];
        for y in 0..8 {
            for x in 0..8 {
                buf[y * 8 + x] = (10 + x) as u8;
            }
        }
        let p = Plane { data: &buf, stride: 8, width: 8, height: 8 };
        // Far off the left edge: every sample is column 0's value.
        let mut mid = vec![0i16; MAX_PB * MAX_PB];
        put_luma(&mut mid, &p, -40, 2, 4, 4, 0, 0, 8);
        let mut out = vec![0u8; 16];
        uni_pred(&mut out, 4, &mid, 4, 4, 8);
        assert!(out.iter().all(|&s| s == 10), "left clamp: {out:?}");
        // Far off the right and bottom: column 7, row 7.
        put_luma(&mut mid, &p, 40, 40, 4, 4, 0, 0, 8);
        uni_pred(&mut out, 4, &mid, 4, 4, 8);
        assert!(out.iter().all(|&s| s == 17), "right/bottom clamp: {out:?}");
    }

    /// The integer position must be an exact copy, not a filter that happens
    /// to have a 1 in the middle — phase 0 has no filter at all.
    #[test_case]
    fn the_integer_position_is_an_exact_copy() {
        let mut buf = vec![0u8; 16 * 16];
        for y in 0..16 {
            for x in 0..16 {
                buf[y * 16 + x] = ((x * 13 + y * 7) & 0xff) as u8;
            }
        }
        let p = Plane { data: &buf, stride: 16, width: 16, height: 16 };
        let mut mid = vec![0i16; MAX_PB * MAX_PB];
        put_luma(&mut mid, &p, 4, 4, 8, 8, 0, 0, 8);
        let mut out = vec![0u8; 64];
        uni_pred(&mut out, 8, &mid, 8, 8, 8);
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(out[y * 8 + x], buf[(4 + y) * 16 + (4 + x)], "at ({x},{y})");
            }
        }
    }

    /// Weighted prediction with unit weights must reduce to the unweighted
    /// form. If it does not, the `denom` shift and the rounding offset are not
    /// consistent with each other — and every real weighted stream would be
    /// off by a fraction of a level everywhere.
    #[test_case]
    fn unit_weights_reduce_to_the_unweighted_prediction() {
        let mut buf = vec![0u8; 24 * 24];
        for y in 0..24 {
            for x in 0..24 {
                buf[y * 24 + x] = ((x * 5 + y * 11) & 0xff) as u8;
            }
        }
        let p = Plane { data: &buf, stride: 24, width: 24, height: 24 };
        let mut mid = vec![0i16; MAX_PB * MAX_PB];
        put_luma(&mut mid, &p, 6, 6, 8, 8, 2, 1, 8);

        let mut plain = vec![0u8; 64];
        uni_pred(&mut plain, 8, &mid, 8, 8, 8);
        let mut weighted = vec![0u8; 64];
        // denom 0, weight 1, offset 0 is the identity weighting.
        uni_pred_weighted(&mut weighted, 8, &mid, 8, 8, 0, 1, 0, 8);
        assert_eq!(plain, weighted, "uni weighting is not the identity at w=1");

        let mut plain_bi = vec![0u8; 64];
        bi_pred(&mut plain_bi, 8, &mid, &mid, 8, 8, 8);
        let mut w_bi = vec![0u8; 64];
        bi_pred_weighted(&mut w_bi, 8, &mid, &mid, 8, 8, 0, 1, 1, 0, 0, 8);
        assert_eq!(plain_bi, w_bi, "bi weighting is not the identity at w=1");
    }

    /// A weight of two must double the *deviation from black*, and an offset
    /// must shift. These pin the two halves separately, because a formula that
    /// applies the offset before the shift passes the unit-weight test above.
    #[test_case]
    fn weighted_prediction_scales_and_offsets_independently() {
        let buf = flat_plane(60, 16, 16);
        let p = Plane { data: &buf, stride: 16, width: 16, height: 16 };
        let mut mid = vec![0i16; MAX_PB * MAX_PB];
        put_luma(&mut mid, &p, 4, 4, 4, 4, 0, 0, 8);
        let mut out = vec![0u8; 16];
        // denom 1, weight 2 == a gain of 1.0; weight 4 == 2.0.
        uni_pred_weighted(&mut out, 4, &mid, 4, 4, 1, 2, 0, 8);
        assert!(out.iter().all(|&s| s == 60), "gain 1.0 changed the value: {out:?}");
        uni_pred_weighted(&mut out, 4, &mid, 4, 4, 1, 4, 0, 8);
        assert!(out.iter().all(|&s| s == 120), "gain 2.0: {out:?}");
        // The offset is added after the shift, in sample units.
        uni_pred_weighted(&mut out, 4, &mid, 4, 4, 1, 2, 7, 8);
        assert!(out.iter().all(|&s| s == 67), "offset: {out:?}");
        // And it saturates rather than wrapping.
        uni_pred_weighted(&mut out, 4, &mid, 4, 4, 1, 2, 200, 8);
        assert!(out.iter().all(|&s| s == 255), "saturation: {out:?}");
    }

    /// Scaling by a distance ratio of one is the identity, scaling by two
    /// doubles, and the rounding is symmetric about zero — the last is what
    /// the `+ 127 + (negative)` term buys, and its absence biases every scaled
    /// vector towards negative infinity.
    #[test_case]
    fn mv_scaling_is_symmetric_about_zero() {
        assert_eq!(mv_scale((10, -6), 4, 4), (10, -6), "ratio 1 must be exact");
        assert_eq!(mv_scale((10, -6), 2, 4), (20, -12), "ratio 2");
        assert_eq!(mv_scale((10, -6), 4, 2), (5, -3), "ratio 1/2");
        assert_eq!(mv_scale((10, -6), 4, -4), (-10, 6), "a negative distance flips");
        // Symmetry: scaling -v must be the negation of scaling v, for every
        // vector and a range of ratios.
        for td in [-8i32, -3, -1, 1, 2, 3, 5, 8] {
            for tb in [-8i32, -3, -1, 1, 2, 3, 5, 8] {
                for v in [1i16, 2, 3, 7, 15, 33, 100, 511] {
                    let a = mv_scale((v, v), td, tb);
                    let b = mv_scale((-v, -v), td, tb);
                    assert_eq!(a.0, -b.0, "asymmetric at v={v} td={td} tb={tb}");
                    assert_eq!(a.1, -b.1);
                }
            }
        }
        // A zero distance cannot be divided by; the vector passes through.
        assert_eq!(mv_scale((3, 4), 0, 8), (3, 4));
    }

    fn mvf(flags: u8, r0: i8, r1: i8, m0: Mv, m1: Mv) -> MvField {
        MvField { pred_flag: flags, ref_idx: [r0, r1], mv: [m0, m1] }
    }

    /// The list is always exactly `max_cand` long, whatever the neighbours
    /// look like — a short list means `merge_idx` indexes off the end, and a
    /// long one means the encoder and decoder disagree about what index 4 is.
    #[test_case]
    fn the_merge_list_is_always_exactly_max_cand_long() {
        let none = MergeNeighbours { a1: None, b1: None, b0: None, a0: None, b2: None };
        for max in 1..=5usize {
            for &is_b in &[false, true] {
                let l = merge_candidates(&none, None, is_b, max, 2);
                assert_eq!(l.len(), max, "empty neighbours, max {max}");
            }
        }
        let a = mvf(1, 0, 0, (4, 4), (0, 0));
        let b = mvf(1, 1, 0, (8, 8), (0, 0));
        let full = MergeNeighbours {
            a1: Some(a),
            b1: Some(b),
            b0: Some(mvf(1, 0, 0, (12, 12), (0, 0))),
            a0: Some(mvf(1, 1, 0, (16, 16), (0, 0))),
            b2: Some(mvf(1, 0, 0, (20, 20), (0, 0))),
        };
        for max in 1..=5usize {
            let l = merge_candidates(&full, None, true, max, 2);
            assert_eq!(l.len(), max, "full neighbours, max {max}");
        }
    }

    /// Pruning compares each position against a *fixed short list* of
    /// predecessors, not against everything already found. B0 duplicating A1
    /// must still be kept, because B0 is only ever compared with B1.
    #[test_case]
    fn merge_pruning_compares_only_the_specified_pairs() {
        let m = mvf(1, 0, 0, (4, 4), (0, 0));
        let other = mvf(1, 1, 0, (8, 8), (0, 0));

        // B1 == A1: pruned.
        let n = MergeNeighbours {
            a1: Some(m),
            b1: Some(m),
            b0: None,
            a0: None,
            b2: None,
        };
        let l = merge_candidates(&n, None, false, 5, 1);
        assert_eq!(l[0], m);
        assert_ne!(l[1], m, "B1 duplicating A1 should have been pruned");

        // B0 == A1 but != B1: kept, because B0 is compared only with B1.
        let n = MergeNeighbours {
            a1: Some(m),
            b1: Some(other),
            b0: Some(m),
            a0: None,
            b2: None,
        };
        let l = merge_candidates(&n, None, false, 5, 1);
        assert_eq!(&l[..3], &[m, other, m], "B0 must not be compared with A1");

        // A0 == B1 but != A1: kept, mirror image of the above.
        let n = MergeNeighbours {
            a1: Some(m),
            b1: Some(other),
            b0: None,
            a0: Some(other),
            b2: None,
        };
        let l = merge_candidates(&n, None, false, 5, 1);
        assert_eq!(&l[..3], &[m, other, other], "A0 must not be compared with B1");
    }

    /// B2 is dropped once four candidates exist — a rule about the *count*,
    /// not about the list being full.
    #[test_case]
    fn b2_is_skipped_at_four_candidates_even_when_the_list_has_room() {
        let c = |k: i16| mvf(1, 0, 0, (k, k), (0, 0));
        let n = MergeNeighbours {
            a1: Some(c(1)),
            b1: Some(c(2)),
            b0: Some(c(3)),
            a0: Some(c(4)),
            b2: Some(c(5)),
        };
        // max_cand 5, so there is room — but B2 is still skipped.
        let l = merge_candidates(&n, None, false, 5, 1);
        assert_eq!(&l[..4], &[c(1), c(2), c(3), c(4)]);
        assert_ne!(l[4], c(5), "B2 must be skipped at four candidates");
        // With only three spatial candidates, B2 is taken.
        let n = MergeNeighbours {
            a1: Some(c(1)),
            b1: Some(c(2)),
            b0: Some(c(3)),
            a0: None,
            b2: Some(c(5)),
        };
        let l = merge_candidates(&n, None, false, 5, 1);
        assert_eq!(&l[..4], &[c(1), c(2), c(3), c(5)]);
    }

    /// Combined bi-predictive candidates exist only on B slices, pair list 0
    /// of one candidate with list 1 of another, and are skipped when the two
    /// would be identical.
    #[test_case]
    fn combined_candidates_are_b_slice_only_and_pair_across_lists() {
        let l0 = mvf(1, 0, 0, (4, 4), (0, 0)); // list 0 only
        let l1 = mvf(2, 0, 1, (0, 0), (8, 8)); // list 1 only
        let n = MergeNeighbours {
            a1: Some(l0),
            b1: Some(l1),
            b0: None,
            a0: None,
            b2: None,
        };
        // P slice: no combination, so index 2 onward is a zero candidate.
        let l = merge_candidates(&n, None, false, 5, 1);
        assert_eq!(l[2].mv, [(0, 0), (0, 0)], "P slices get no combined candidate");
        assert_eq!(l[2].pred_flag, 1, "a P slice zero candidate uses list 0 only");

        // B slice: (0, 1) pairs A1's list 0 with B1's list 1.
        let l = merge_candidates(&n, None, true, 5, 1);
        assert_eq!(l[2].pred_flag, 3, "the combination must be bi-predictive");
        assert_eq!(l[2].mv, [(4, 4), (8, 8)]);
        assert_eq!(l[2].ref_idx, [0, 1]);
        // (1, 0) needs B1 to have list 0 and A1 list 1; neither does, so the
        // next entry falls through to a zero candidate.
        assert_eq!(l[3].mv, [(0, 0), (0, 0)]);
    }

    /// Zero candidates step their reference index while references remain and
    /// then stay at 0 — running past `nb_refs` would name a picture that is
    /// not in the list.
    #[test_case]
    fn zero_candidates_step_then_pin_their_reference_index() {
        let none = MergeNeighbours { a1: None, b1: None, b0: None, a0: None, b2: None };
        let l = merge_candidates(&none, None, false, 5, 2);
        assert_eq!(l[0].ref_idx, [0, 0]);
        assert_eq!(l[1].ref_idx, [1, 1]);
        assert_eq!(l[2].ref_idx, [0, 0], "past nb_refs must pin to 0");
        assert_eq!(l[3].ref_idx, [0, 0]);
        assert!(l.iter().all(|c| c.mv == [(0, 0), (0, 0)]));
        // On a B slice they are bi-predictive.
        let l = merge_candidates(&none, None, true, 3, 1);
        assert!(l.iter().all(|c| c.pred_flag == 3));
    }

    fn nb(flags: u8, mv0: Mv, mv1: Mv, poc0: i32, poc1: i32) -> AmvpNeighbour {
        AmvpNeighbour {
            mvf: MvField { pred_flag: flags, ref_idx: [0, 0], mv: [mv0, mv1] },
            ref_poc: [poc0, poc1],
            long_term: [false, false],
        }
    }

    /// A neighbour referencing the target picture is used **unscaled**, and it
    /// counts whichever list it came through — the picture is what matters,
    /// not the slot.
    #[test_case]
    fn amvp_takes_an_exact_reference_from_either_list() {
        // A1 predicts POC 4 through list 1 while we are building list 0.
        let a1 = nb(2, (0, 0), (12, -8), 0, 4);
        let c = amvp_candidates(None, Some(a1), None, None, None, 0, 4, false, 8, None);
        assert_eq!(c[0], (12, -8), "the opposite list still matches by picture");
        // A different picture is not an exact match, so it is scaled instead:
        // neighbour distance 8 - 0 = 8, target distance 8 - 4 = 4, so halved.
        let a1 = nb(1, (12, -8), (0, 0), 0, 0);
        let c = amvp_candidates(None, Some(a1), None, None, None, 0, 4, false, 8, None);
        assert_eq!(c[0], (6, -4), "a different picture must be scaled");
    }

    /// Long-term references are never scaled, and a long-term neighbour cannot
    /// stand in for a short-term target at all — there is no distance to scale
    /// by, so scaling one would produce a confident wrong vector.
    #[test_case]
    fn amvp_refuses_to_mix_long_and_short_term_references() {
        let mut a1 = nb(1, (12, -8), (0, 0), 0, 0);
        a1.long_term = [true, true];
        // Target is short-term: the long-term neighbour is unusable, so the
        // candidate falls through to zero.
        let c = amvp_candidates(None, Some(a1), None, None, None, 0, 4, false, 8, None);
        assert_eq!(c[0], (0, 0), "a long-term neighbour cannot scale to short-term");
        // Target is long-term and so is the neighbour: taken unscaled, even
        // though the distances differ.
        let c = amvp_candidates(None, Some(a1), None, None, None, 0, 4, true, 8, None);
        assert_eq!(c[0], (12, -8), "matching long-term is used as-is, never scaled");
    }

    /// With no left neighbour at all, B is promoted into A's slot and B is
    /// re-derived with scaling allowed. Missing this yields two candidates that
    /// are the right *number* and the wrong vectors.
    #[test_case]
    fn amvp_promotes_b_when_the_left_column_is_absent() {
        // B1 references a different picture, so it is not an exact match; with
        // no A neighbours it becomes candidate 0 via the scaled pass.
        let b1 = nb(1, (16, 16), (0, 0), 0, 0);
        let c = amvp_candidates(None, None, None, Some(b1), None, 0, 4, false, 8, None);
        assert_eq!(c[0], (8, 8), "B must be scaled and promoted");

        // With a left neighbour present, that second pass does not happen, so
        // an inexact B contributes nothing and A's scaled value stands alone.
        let a1 = nb(1, (4, 4), (0, 0), 4, 0); // exact match for POC 4
        let c = amvp_candidates(None, Some(a1), None, Some(b1), None, 0, 4, false, 8, None);
        assert_eq!(c[0], (4, 4));
        assert_eq!(c[1], (0, 0), "an inexact B is not scaled once A exists");
    }

    /// B is pruned only against A, and the list is always exactly two long.
    #[test_case]
    fn amvp_prunes_b_against_a_and_always_returns_two() {
        let same = nb(1, (8, 8), (0, 0), 4, 0);
        let c = amvp_candidates(None, Some(same), None, Some(same), None, 0, 4, false, 8, None);
        assert_eq!(c[0], (8, 8));
        assert_eq!(c[1], (0, 0), "a duplicate B is dropped, then zero-filled");
        // The temporal candidate fills the gap when there is one.
        let c =
            amvp_candidates(None, Some(same), None, Some(same), None, 0, 4, false, 8, Some((3, 3)));
        assert_eq!(c[1], (3, 3));
        // A temporal candidate identical to A is *not* pruned — only B is.
        let c =
            amvp_candidates(None, Some(same), None, None, None, 0, 4, false, 8, Some((8, 8)));
        assert_eq!(c, [(8, 8), (8, 8)]);
        // Nothing available at all: two zeros, never a short list.
        let c = amvp_candidates(None, None, None, None, None, 0, 4, false, 8, None);
        assert_eq!(c, [(0, 0), (0, 0)]);
    }
}
