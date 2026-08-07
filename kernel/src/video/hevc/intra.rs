//! HEVC intra prediction (H.265 §8.4.4.2): reference sample substitution and
//! filtering, then the 35 prediction modes.
//!
//! Three things here are each a whole class of bug, and none of them fails
//! loudly:
//!
//! - **Substitution** (§8.4.4.2.2) fills in neighbours that do not exist —
//!   outside the picture, in another slice, or not yet decoded. Getting the
//!   *scan order* wrong still produces a smooth-looking block, because the fill
//!   values are real pixels from somewhere nearby.
//! - **Filtering** (§8.4.4.2.3) is applied for some (mode, size) pairs and not
//!   others. Applying it everywhere makes intra blocks slightly soft; applying
//!   it nowhere makes them slightly ringy. Both look like an encoder choice.
//! - **The angular modes** project a 1-D reference along a slope. Off-by-one in
//!   the projection tilts the whole block by a fraction of a pixel, which is
//!   invisible in one frame and accumulates over an intra-predicted GOP.
//!
//! So the reference arrays here carry the corner at index **0** and sample `i`
//! at `i + 1`, rather than the negative-index pointer arithmetic every C
//! decoder uses — the corner is a distinct thing from the first above sample
//! and treating it as `top[-1]` is how it gets filtered twice.

use super::tables as tb;

/// The largest transform block, and so the largest intra block, is 32x32 —
/// which needs `2 * 32` reference samples on each side plus the corner.
pub const MAX_TB: usize = 32;
const REF_LEN: usize = 2 * MAX_TB + 1;

/// Planar is mode 0 and DC is mode 1; 2..=34 are the angular modes.
pub const PLANAR: u8 = 0;
pub const DC: u8 = 1;

/// One block's reference samples. `top[0] == left[0]` is the corner; `top[1+i]`
/// is the sample above column `i`, `left[1+i]` the sample left of row `i`.
#[derive(Clone)]
pub struct Refs {
    pub top: [u16; REF_LEN],
    pub left: [u16; REF_LEN],
}

impl Default for Refs {
    fn default() -> Self {
        Refs { top: [0; REF_LEN], left: [0; REF_LEN] }
    }
}

impl Refs {
    #[inline]
    fn corner(&self) -> i32 {
        self.top[0] as i32
    }
    #[inline]
    fn t(&self, i: usize) -> i32 {
        self.top[i + 1] as i32
    }
    #[inline]
    fn l(&self, i: usize) -> i32 {
        self.left[i + 1] as i32
    }
}

/// Reference sample substitution (§8.4.4.2.2).
///
/// `avail` is indexed in the **specification's** scan order: index 0 is the
/// bottom-most left sample `p[-1][2N-1]`, running up the left edge to
/// `p[-1][0]` at `2N-1`, then the corner at `2N`, then across the top from
/// `p[0][-1]` at `2N+1` to `p[2N-1][-1]` at `4N`.
///
/// That order is the whole substance of the rule: a missing sample takes the
/// value of the **previous** one in this sequence, so the fill propagates up
/// the left edge, around the corner, and along the top. Filling from the
/// nearest available sample in Euclidean terms — the obvious reading — gives a
/// different picture at every block that touches a slice or picture boundary.
///
/// With nothing available at all the whole array becomes the mid-grey
/// `1 << (bit_depth - 1)`.
pub fn substitute(refs: &mut Refs, n: usize, avail: &[bool], bit_depth: u32) {
    let total = 4 * n + 1;
    debug_assert_eq!(avail.len(), total);

    // Linear view in scan order, so the rule is written once.
    let get = |r: &Refs, i: usize| -> u16 {
        if i < 2 * n {
            r.left[2 * n - i] // p[-1][2N-1-i]
        } else if i == 2 * n {
            r.top[0]
        } else {
            r.top[i - 2 * n]
        }
    };
    let set = |r: &mut Refs, i: usize, v: u16| {
        if i < 2 * n {
            r.left[2 * n - i] = v;
        } else if i == 2 * n {
            r.top[0] = v;
            r.left[0] = v;
        } else {
            r.top[i - 2 * n] = v;
        }
    };

    let first = avail.iter().position(|&a| a);
    let Some(first) = first else {
        let mid = (1u16 << (bit_depth - 1));
        refs.top[..2 * n + 1].fill(mid);
        refs.left[..2 * n + 1].fill(mid);
        return;
    };
    // Everything before the first available sample takes its value.
    let v = get(refs, first);
    for i in 0..first {
        set(refs, i, v);
    }
    // Then each unavailable sample takes its predecessor's.
    for i in first + 1..total {
        if !avail[i] {
            let prev = get(refs, i - 1);
            set(refs, i, prev);
        }
    }
}

/// Whether the reference samples are smoothed for this (mode, size, plane)
/// (§8.4.4.2.3).
///
/// The threshold table is indexed by `log2_size - 3`, so 4x4 is excluded by
/// construction — and chroma is never filtered in 4:2:0.
pub fn filter_flag(mode: u8, log2_size: u32, c_idx: usize) -> bool {
    if mode == DC || log2_size == 2 || c_idx != 0 {
        return false;
    }
    let thresh = [7i32, 1, 0][(log2_size - 3) as usize];
    let m = mode as i32;
    let dist = (m - 26).abs().min((m - 10).abs());
    dist > thresh
}

/// Apply the reference filter chosen for this block.
///
/// Returns `true` when the **strong** (bilinear) filter was used — a 32x32 luma
/// block whose edges are already nearly linear, where the 3-tap filter would
/// leave visible banding on a gradient. It is conditional on the SPS flag
/// *and* on measuring the edges, so a decoder that always takes one branch is
/// wrong on real content in one direction or the other.
pub fn filter_refs(
    refs: &mut Refs,
    n: usize,
    mode: u8,
    log2_size: u32,
    c_idx: usize,
    strong_enabled: bool,
    bit_depth: u32,
) -> bool {
    if !filter_flag(mode, log2_size, c_idx) {
        return false;
    }
    let two_n = 2 * n;
    if strong_enabled && c_idx == 0 && log2_size == 5 {
        let threshold = 1i32 << (bit_depth - 5);
        let c = refs.corner();
        if (c + refs.t(63) - 2 * refs.t(31)).abs() < threshold
            && (c + refs.l(63) - 2 * refs.l(31)).abs() < threshold
        {
            let (t63, l63) = (refs.t(63), refs.l(63));
            for i in 0..63usize {
                let k = i as i32 + 1;
                refs.top[i + 1] = (((64 - k) * c + k * t63 + 32) >> 6) as u16;
                refs.left[i + 1] = (((64 - k) * c + k * l63 + 32) >> 6) as u16;
            }
            return true;
        }
    }
    // 3-tap [1 2 1]. The last sample of each edge is left alone (it has no
    // successor), and the corner is filtered from *both* edges — computing it
    // from only one is a single wrong pixel that then seeds every angular
    // projection that passes through it.
    let old = refs.clone();
    for i in (0..two_n - 1).rev() {
        let prev = if i == 0 { old.corner() } else { old.l(i - 1) };
        refs.left[i + 1] = ((old.l(i + 1) + 2 * old.l(i) + prev + 2) >> 2) as u16;
    }
    for i in (0..two_n - 1).rev() {
        let prev = if i == 0 { old.corner() } else { old.t(i - 1) };
        refs.top[i + 1] = ((old.t(i + 1) + 2 * old.t(i) + prev + 2) >> 2) as u16;
    }
    let c = ((old.l(0) + 2 * old.corner() + old.t(0) + 2) >> 2 ) as u16;
    refs.top[0] = c;
    refs.left[0] = c;
    false
}

/// Planar prediction (mode 0): a bilinear blend of the four edge samples.
pub fn pred_planar(dst: &mut [u16], stride: usize, refs: &Refs, log2_size: u32) {
    let n = 1usize << log2_size;
    let (tr, bl) = (refs.t(n), refs.l(n));
    for y in 0..n {
        for x in 0..n {
            let v = ((n - 1 - x) as i32 * refs.l(y)
                + (x + 1) as i32 * tr
                + (n - 1 - y) as i32 * refs.t(x)
                + (y + 1) as i32 * bl
                + n as i32)
                >> (log2_size + 1);
            dst[y * stride + x] = v as u16;
        }
    }
}

/// DC prediction (mode 1), with the luma edge smoothing that applies below
/// 32x32 — three extra lines that are easy to omit and produce a visible
/// blocking seam exactly where DC blocks meet.
pub fn pred_dc(dst: &mut [u16], stride: usize, refs: &Refs, log2_size: u32, c_idx: usize) {
    let n = 1usize << log2_size;
    let mut dc = n as i32;
    for i in 0..n {
        dc += refs.l(i) + refs.t(i);
    }
    dc >>= log2_size + 1;
    for y in 0..n {
        for x in 0..n {
            dst[y * stride + x] = dc as u16;
        }
    }
    if c_idx == 0 && n < 32 {
        dst[0] = ((refs.l(0) + 2 * dc + refs.t(0) + 2) >> 2) as u16;
        for x in 1..n {
            dst[x] = ((refs.t(x) + 3 * dc + 2) >> 2) as u16;
        }
        for y in 1..n {
            dst[y * stride] = ((refs.l(y) + 3 * dc + 2) >> 2) as u16;
        }
    }
}

/// Angular prediction (modes 2..=34).
///
/// Modes >= 18 project from the **top** edge, below from the **left**; a
/// negative angle needs the *other* edge folded in behind the corner, which is
/// what `INV_ANGLE` is for. The projected extension is built into a scratch
/// array indexed from `-n`, because the reference genuinely is consulted at
/// negative offsets.
pub fn pred_angular(
    dst: &mut [u16],
    stride: usize,
    refs: &Refs,
    mode: u8,
    log2_size: u32,
    c_idx: usize,
    bit_depth: u32,
) {
    let n = 1usize << log2_size;
    let angle = tb::INTRA_PRED_ANGLE[mode as usize - 2] as i32;
    let last = (n as i32 * angle) >> 5;

    // `scratch` is indexed by `OFF + i` for i in -(n) ..= 2n.
    const OFF: usize = MAX_TB;
    let mut scratch = [0u16; 3 * MAX_TB + 4];
    let vertical = mode >= 18;

    // `refp(i)` is FFmpeg's `ref[i]`, i.e. the edge shifted so index 0 is the
    // corner: `ref = top - 1`.
    let base = |i: i32| -> u16 {
        let v = if vertical {
            if i == 0 {
                refs.top[0]
            } else {
                refs.top[i as usize]
            }
        } else if i == 0 {
            refs.left[0]
        } else {
            refs.left[i as usize]
        };
        v
    };

    let extended = angle < 0 && last < -1;
    if extended {
        for x in 0..=n {
            scratch[OFF + x] = base(x as i32);
        }
        let inv = tb::INV_ANGLE[mode as usize - 11] as i32;
        let mut x = last;
        while x <= -1 {
            // The *other* edge, sampled through the reciprocal slope. `- 1`
            // because index 0 of that edge is the corner too.
            let idx = -1 + ((x * inv + 128) >> 8);
            let v = if vertical {
                if idx < 0 { refs.left[0] } else { refs.left[(idx + 1) as usize] }
            } else if idx < 0 {
                refs.top[0]
            } else {
                refs.top[(idx + 1) as usize]
            };
            scratch[(OFF as i32 + x) as usize] = v;
            x += 1;
        }
    }
    let rd = |i: i32| -> i32 {
        if extended { scratch[(OFF as i32 + i) as usize] as i32 } else { base(i) as i32 }
    };

    let max = (1i32 << bit_depth) - 1;
    if vertical {
        for y in 0..n {
            let idx = (((y + 1) as i32) * angle) >> 5;
            let fact = (((y + 1) as i32) * angle) & 31;
            for x in 0..n {
                let xi = x as i32 + idx + 1;
                dst[y * stride + x] = if fact != 0 {
                    (((32 - fact) * rd(xi) + fact * rd(xi + 1) + 16) >> 5) as u16
                } else {
                    rd(xi) as u16
                };
            }
        }
        // Mode 26 is exactly vertical, so the left column gets the same
        // gradient correction DC gets — luma only, below 32x32.
        if mode == 26 && c_idx == 0 && n < 32 {
            for y in 0..n {
                let v = refs.t(0) + ((refs.l(y) - refs.corner()) >> 1);
                dst[y * stride] = v.clamp(0, max) as u16;
            }
        }
    } else {
        for x in 0..n {
            let idx = (((x + 1) as i32) * angle) >> 5;
            let fact = (((x + 1) as i32) * angle) & 31;
            for y in 0..n {
                let yi = y as i32 + idx + 1;
                dst[y * stride + x] = if fact != 0 {
                    (((32 - fact) * rd(yi) + fact * rd(yi + 1) + 16) >> 5) as u16
                } else {
                    rd(yi) as u16
                };
            }
        }
        if mode == 10 && c_idx == 0 && n < 32 {
            for x in 0..n {
                let v = refs.l(0) + ((refs.t(x) - refs.corner()) >> 1);
                dst[x] = v.clamp(0, max) as u16;
            }
        }
    }
}

/// Predict one block by mode.
pub fn predict(
    dst: &mut [u16],
    stride: usize,
    refs: &Refs,
    mode: u8,
    log2_size: u32,
    c_idx: usize,
    bit_depth: u32,
) {
    match mode {
        PLANAR => pred_planar(dst, stride, refs, log2_size),
        DC => pred_dc(dst, stride, refs, log2_size, c_idx),
        _ => pred_angular(dst, stride, refs, mode, log2_size, c_idx, bit_depth),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn flat(v: u16) -> Refs {
        Refs { top: [v; REF_LEN], left: [v; REF_LEN] }
    }

    /// Every mode fed a constant neighbourhood must reproduce that constant.
    ///
    /// This is the single strongest cheap check on the whole module: planar's
    /// weights must sum to `2^(k+1)`, DC's rounding must be exact, and every
    /// angular projection must land on real samples rather than off the end of
    /// the reference — a projection that runs past the array reads the
    /// substitution fill, which on a flat block is the same value, so the
    /// **boundary cases are tested separately below**.
    #[test_case]
    fn a_constant_neighbourhood_predicts_that_constant_in_every_mode() {
        for &v in &[0u16, 1, 128, 254, 255] {
            let refs = flat(v);
            for log2 in 2..=5u32 {
                let n = 1usize << log2;
                for mode in 0..=34u8 {
                    let mut dst = vec![0u16; n * n];
                    predict(&mut dst, n, &refs, mode, log2, 0, 8);
                    assert!(
                        dst.iter().all(|&p| p == v),
                        "mode {mode} log2 {log2} value {v}: {:?}",
                        &dst[..n]
                    );
                }
            }
        }
    }

    /// Mode 26 is pure vertical and mode 10 pure horizontal: the block is the
    /// top row (resp. left column) copied. If the two are swapped — the classic
    /// mistake, since the mode numbers are adjacent to the diagonals — a
    /// picture still decodes and merely looks smeared the wrong way.
    #[test_case]
    fn vertical_and_horizontal_modes_copy_the_right_edge() {
        let n = 8usize;
        let mut refs = flat(100);
        for i in 0..2 * n {
            refs.top[i + 1] = (10 + i * 3) as u16;
            refs.left[i + 1] = (200 - i * 3) as u16;
        }
        refs.top[0] = 100;
        refs.left[0] = 100;

        // Chroma, so the luma-only gradient correction on column/row 0 is off
        // and the copy is exact everywhere.
        let mut dst = vec![0u16; n * n];
        predict(&mut dst, n, &refs, 26, 3, 1, 8);
        for y in 0..n {
            for x in 0..n {
                assert_eq!(dst[y * n + x], refs.top[x + 1], "mode 26 at ({x},{y})");
            }
        }
        let mut dst = vec![0u16; n * n];
        predict(&mut dst, n, &refs, 10, 3, 1, 8);
        for y in 0..n {
            for x in 0..n {
                assert_eq!(dst[y * n + x], refs.left[y + 1], "mode 10 at ({x},{y})");
            }
        }
    }

    /// Modes 2 and 34 are the two 45-degree diagonals and shift by exactly one
    /// sample per row — the cleanest test that the projection's `+1` offsets
    /// are right, because the result is an exact copy with no interpolation.
    #[test_case]
    fn the_forty_five_degree_diagonals_shift_by_one_per_row() {
        let n = 8usize;
        let mut refs = flat(0);
        for i in 0..2 * n {
            refs.top[i + 1] = (i * 7 + 3) as u16;
            refs.left[i + 1] = (i * 5 + 11) as u16;
        }
        // Mode 34: angle +32, from the top edge — row y reads top[x + y + 1].
        let mut dst = vec![0u16; n * n];
        predict(&mut dst, n, &refs, 34, 3, 1, 8);
        for y in 0..n {
            for x in 0..n {
                assert_eq!(dst[y * n + x], refs.top[x + y + 2], "mode 34 ({x},{y})");
            }
        }
        // Mode 2: angle +32, from the left edge — column x reads left[y + x + 1].
        let mut dst = vec![0u16; n * n];
        predict(&mut dst, n, &refs, 2, 3, 1, 8);
        for y in 0..n {
            for x in 0..n {
                assert_eq!(dst[y * n + x], refs.left[y + x + 2], "mode 2 ({x},{y})");
            }
        }
    }

    /// Planar's two structural properties, neither of which is its own formula
    /// restated.
    ///
    /// It is **not** true that planar reproduces an arbitrary linear gradient —
    /// the vertical interpolation runs from the top row to `p[-1][N]` and the
    /// horizontal from the left column to `p[N][-1]`, so a plane whose
    /// continuation past the block disagrees with those two anchors comes out
    /// bowed. (That assumption is what this test asserted first, and it is off
    /// by 2 at the corner.) What *is* true:
    #[test_case]
    fn planar_is_symmetric_and_matches_the_specification_weights() {
        // 1. With the top and left edges identical, the prediction must be its
        //    own transpose. A swapped x/y weight — the easy mistake in a
        //    four-term expression where two terms use `x` and two use `y` —
        //    breaks this and nothing else.
        let n = 8usize;
        let mut refs = Refs::default();
        for i in 0..2 * n {
            let v = (30 + i * 6) as u16;
            refs.top[i + 1] = v;
            refs.left[i + 1] = v;
        }
        refs.top[0] = 24;
        refs.left[0] = 24;
        let mut dst = vec![0u16; n * n];
        predict(&mut dst, n, &refs, PLANAR, 3, 0, 8);
        for y in 0..n {
            for x in 0..n {
                assert_eq!(dst[y * n + x], dst[x * n + y], "planar not symmetric at ({x},{y})");
            }
        }

        // 2. Hand-computed against H.265 (8-82):
        //      pred = ((N-1-x)*p[-1][y] + (x+1)*p[N][-1]
        //            + (N-1-y)*p[x][-1] + (y+1)*p[-1][N] + N) >> (log2N + 1)
        //    at N = 4 with distinct top and left edges, so every one of the
        //    four terms is separately observable.
        let mut refs = Refs::default();
        for i in 0..8usize {
            refs.top[i + 1] = (10 + i * 10) as u16; // 10..80, so p[4][-1] = 50
            refs.left[i + 1] = (100 + i * 10) as u16; // 100..170, p[-1][4] = 140
        }
        refs.top[0] = 0;
        refs.left[0] = 0;
        let mut dst = vec![0u16; 16];
        predict(&mut dst, 4, &refs, PLANAR, 2, 0, 8);
        // (0,0): (3*100 + 1*50 + 3*10  + 1*140 + 4) >> 3 = 524 >> 3 = 65
        // (3,0): (0*100 + 4*50 + 3*40  + 1*140 + 4) >> 3 = 464 >> 3 = 58
        // (0,3): (3*130 + 1*50 + 0*10  + 4*140 + 4) >> 3 = 1004 >> 3 = 125
        // (3,3): (0     + 4*50 + 0     + 4*140 + 4) >> 3 = 764 >> 3 = 95
        assert_eq!(dst[0], 65, "planar (0,0)");
        assert_eq!(dst[3], 58, "planar (3,0)");
        assert_eq!(dst[3 * 4], 125, "planar (0,3)");
        assert_eq!(dst[3 * 4 + 3], 95, "planar (3,3)");

    }

    /// Edges that rise towards anchors above every edge sample give a block
    /// that rises in both directions — a weight with the wrong sign shows up
    /// here even where spot checks happen to pass.
    ///
    /// The condition matters: with the top edge's far anchor *below* the left
    /// edge's values, planar legitimately decreases in x, so "monotone edges
    /// give a monotone block" is only true when the anchors dominate.
    #[test_case]
    fn planar_is_monotone_when_both_anchors_exceed_their_edges() {
        let n = 8usize;
        let mut refs = Refs::default();
        for i in 0..2 * n {
            let v = (30 + i * 3) as u16; // 30..75; anchors p[8][-1] = p[-1][8] = 54
            refs.top[i + 1] = v;
            refs.left[i + 1] = v;
        }
        refs.top[0] = 27;
        refs.left[0] = 27;
        let mut dst = vec![0u16; n * n];
        predict(&mut dst, n, &refs, PLANAR, 3, 0, 8);
        for y in 0..n {
            for x in 0..n - 1 {
                assert!(
                    dst[y * n + x] <= dst[y * n + x + 1],
                    "not monotone in x at row {y}: {:?}",
                    &dst[y * n..y * n + n]
                );
            }
        }
        for x in 0..n {
            for y in 0..n - 1 {
                assert!(dst[y * n + x] <= dst[(y + 1) * n + x], "not monotone in y at col {x}");
            }
        }
    }

    /// Substitution propagates **in the specification's scan order** — up the
    /// left edge, around the corner, along the top — not from whichever
    /// neighbour happens to be closest.
    #[test_case]
    fn substitution_propagates_around_the_corner() {
        let n = 4usize;
        let total = 4 * n + 1;

        // Only the bottom-left-most sample exists. Everything else must take
        // its value, all the way along the top edge.
        let mut refs = Refs::default();
        refs.left[2 * n] = 77; // p[-1][2N-1], scan index 0
        let mut avail = vec![false; total];
        avail[0] = true;
        substitute(&mut refs, n, &avail, 8);
        assert!(refs.top[..2 * n + 1].iter().all(|&v| v == 77));
        assert!(refs.left[..2 * n + 1].iter().all(|&v| v == 77));

        // Only the *last* top sample exists: everything before it — the whole
        // left edge and corner — takes that value, which is the backwards fill
        // and the half a naive implementation forgets.
        let mut refs = Refs::default();
        refs.top[2 * n] = 42;
        let mut avail = vec![false; total];
        avail[total - 1] = true;
        substitute(&mut refs, n, &avail, 8);
        assert!(refs.left[..2 * n + 1].iter().all(|&v| v == 42));
        assert!(refs.top[..2 * n + 1].iter().all(|&v| v == 42));

        // Nothing available at all: mid-grey, and it must scale with bit depth.
        let mut refs = Refs::default();
        substitute(&mut refs, n, &vec![false; total], 8);
        assert!(refs.top[..2 * n + 1].iter().all(|&v| v == 128));

        // A gap in the middle of the top edge takes the sample before it.
        let mut refs = Refs::default();
        let mut avail = vec![true; total];
        for i in 0..total {
            // scan index -> value
            if i < 2 * n {
                refs.left[2 * n - i] = i as u16;
            } else if i == 2 * n {
                refs.top[0] = i as u16;
                refs.left[0] = i as u16;
            } else {
                refs.top[i - 2 * n] = i as u16;
            }
        }
        avail[2 * n + 2] = false;
        substitute(&mut refs, n, &avail, 8);
        assert_eq!(refs.top[2], refs.top[1]);
        assert_eq!(refs.top[3], (2 * n + 3) as u16, "the gap must not cascade");
    }

    /// The filter decision table. 4x4 is never filtered, DC is never filtered,
    /// chroma is never filtered, and the threshold *tightens* with size — at
    /// 32x32 every mode but exactly vertical and horizontal is filtered.
    #[test_case]
    fn reference_filter_is_chosen_by_mode_and_size() {
        for mode in 0..=34u8 {
            assert!(!filter_flag(mode, 2, 0), "4x4 mode {mode}");
            assert!(!filter_flag(mode, 4, 1), "chroma mode {mode}");
        }
        assert!(!filter_flag(DC, 5, 0));
        // log2 3 (8x8): threshold 7, so only the two diagonals qualify.
        assert!(filter_flag(2, 3, 0));
        assert!(filter_flag(34, 3, 0));
        // Mode 18 is the diagonal: 8 away from both 26 and 10, so it *is*
        // filtered at 8x8 (8 > 7) — the threshold is exclusive, and reading it
        // as inclusive silently stops filtering the one mode most in need of it.
        assert!(filter_flag(18, 3, 0));
        assert!(!filter_flag(20, 3, 0)); // dist 6, inside the threshold
        // log2 5 (32x32): threshold 0, so everything but modes 10 and 26.
        assert!(filter_flag(11, 5, 0));
        assert!(!filter_flag(10, 5, 0));
        assert!(!filter_flag(26, 5, 0));
    }

    /// The 3-tap filter preserves a constant and preserves the *end* samples,
    /// and the corner is filtered from both edges at once. A corner taken from
    /// only the top edge is one wrong pixel that seeds every negative-angle
    /// projection.
    #[test_case]
    fn three_tap_reference_filter_preserves_ends_and_the_corner_uses_both_edges() {
        let n = 8usize;
        let mut refs = flat(90);
        let before = refs.clone();
        // Mode 2 at 8x8 is filtered.
        assert!(!filter_refs(&mut refs, n, 2, 3, 0, false, 8));
        assert_eq!(refs.top[..2 * n + 1], before.top[..2 * n + 1]);

        let mut refs = Refs::default();
        for i in 0..2 * n {
            refs.top[i + 1] = 10;
            refs.left[i + 1] = 200;
        }
        refs.top[0] = 100;
        refs.left[0] = 100;
        filter_refs(&mut refs, n, 2, 3, 0, false, 8);
        // (200 + 2*100 + 10 + 2) >> 2 = 103
        assert_eq!(refs.top[0], 103);
        assert_eq!(refs.left[0], 103, "corner must be written to both edges");
        assert_eq!(refs.top[2 * n], 10, "last top sample is not filtered");
        assert_eq!(refs.left[2 * n], 200, "last left sample is not filtered");
    }

    /// The strong (bilinear) filter replaces a near-linear 32x32 edge with an
    /// exactly linear one, and only then. On a flat edge it is indistinguishable
    /// from the input — so the test uses a ramp with a kink and checks the kink
    /// is gone.
    #[test_case]
    fn strong_smoothing_only_fires_on_a_near_linear_32x32_luma_edge() {
        let n = 32usize;
        // A step edge. NB the test is only meaningful if the edge really is far
        // from linear: a ramp that saturates halfway measures |c + t63 - 2*t31|
        // = 2, which is *inside* the threshold of 8 and would fire — the
        // predicate looks only at three samples, so "visibly not a straight
        // line" and "not near-linear by this test" are different things.
        let mut refs = Refs::default();
        for i in 0..2 * n {
            let v = if i < 32 { 10u16 } else { 200 };
            refs.top[i + 1] = v;
            refs.left[i + 1] = v;
        }
        refs.top[0] = 10;
        refs.left[0] = 10;
        let mut a = refs.clone();
        assert!(
            !filter_refs(&mut a, n, 11, 5, 0, true, 8),
            "a step edge is not near-linear"
        );

        // A true straight line: corner + top[63] - 2 * top[31] == 0.
        let mut refs = Refs::default();
        for i in 0..2 * n {
            refs.top[i + 1] = (2 + i * 2) as u16;
            refs.left[i + 1] = (2 + i * 2) as u16;
        }
        refs.top[0] = 0;
        refs.left[0] = 0;
        let mut b = refs.clone();
        assert!(filter_refs(&mut b, n, 11, 5, 0, true, 8));
        // Endpoints are preserved and the interior is the exact ramp.
        assert_eq!(b.top[0], 0);
        assert_eq!(b.top[64], refs.top[64]);
        for i in 0..63 {
            let k = i as i32 + 1;
            let want = ((64 - k) * 0 + k * refs.t(63) + 32) >> 6;
            assert_eq!(b.top[i + 1] as i32, want, "strong filter at {i}");
        }
        // Off in the SPS, or on chroma, or at any other size: never taken.
        let mut c = refs.clone();
        assert!(!filter_refs(&mut c, n, 11, 5, 0, false, 8));
        let mut d = refs.clone();
        assert!(!filter_refs(&mut d, n, 11, 5, 1, true, 8));
    }
}
