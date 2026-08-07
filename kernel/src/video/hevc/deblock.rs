//! HEVC in-loop deblocking filter — the sample kernels and the `tc`/`beta`
//! derivation (H.265 §8.7.2).
//!
//! HEVC filters on an **8x8 grid**, not H.264's 4x4, and decides per 4-line
//! segment. Each 8-sample edge is therefore two independent halves, each with
//! its own `tc`, its own `no_p`/`no_q`, and its own strong/weak decision — but
//! the decision for a half is taken from **lines 0 and 3 only**, and applied to
//! all four. A decoder that decides per line produces a smoother, plausible,
//! wrong picture.
//!
//! Two more things are silent when wrong:
//!
//! - **Every sample read must be the pre-filter value.** The strong filter
//!   writes `P0`, `P1`, `P2` from `p0..p3`/`q0..q3`; reading a value back after
//!   writing it feeds the filter its own output. The kernels here snapshot the
//!   eight samples first for exactly that reason.
//! - **`tc` is looked up at `qp + 2 * (bS - 1)`**, so a bS of 2 (an intra edge)
//!   filters harder than a bS of 1 at the same QP. Dropping the `bS` term still
//!   filters every edge that should be filtered, just uniformly — which reads
//!   as slightly soft intra edges and nothing else.
//!
//! Deriving `bS` itself needs the CU/PU/TU structure and lives with the
//! decoder; everything in this module is pure and takes `bS` as an input.

use super::tables as tb;

const MAX_QP: i32 = 51;
/// The extra `tc` step an intra edge gets, per unit of `bS` above 1.
const INTRA_TC_OFFSET: i32 = 2;

/// `tc'` for an edge (H.265 table 8-12).
///
/// `tc_offset` comes from the slice or PPS and is signalled in units of two,
/// which is why it is rounded down to even here — `(o >> 1) << 1`. Applying it
/// unrounded shifts the whole curve by one QP on odd values.
pub fn tc(qp: i32, bs: u8, tc_offset: i32) -> i32 {
    let idx = (qp + INTRA_TC_OFFSET * (bs as i32 - 1) + ((tc_offset >> 1) << 1))
        .clamp(0, MAX_QP + INTRA_TC_OFFSET);
    tb::TC_TABLE[idx as usize] as i32
}

/// `beta'` for an edge (H.265 table 8-12).
pub fn beta(qp: i32, beta_offset: i32) -> i32 {
    let idx = (qp + beta_offset).clamp(0, MAX_QP);
    tb::BETA_TABLE[idx as usize] as i32
}

/// A view of one edge: `base` is the first `q` sample, `xstride` steps across
/// the edge (so `p0` is at `base - xstride`) and `ystride` along it.
///
/// Signed strides are what let one kernel serve both a vertical edge (across =
/// 1, along = stride) and a horizontal one (across = stride, along = 1) — the
/// alternative is two transposed copies of every filter, which is two places
/// for the same bug.
#[inline]
fn at(pix: &[u16], base: isize, xs: isize, ys: isize, dx: isize, dy: isize) -> i32 {
    pix[(base + dx * xs + dy * ys) as usize] as i32
}

#[inline]
fn put(pix: &mut [u16], base: isize, xs: isize, ys: isize, dx: isize, dy: isize, v: i32, max: i32) {
    pix[(base + dx * xs + dy * ys) as usize] = v.clamp(0, max) as u16;
}

/// The strong luma filter: three samples each side, from a 4- and 5-tap blend.
fn luma_strong(
    pix: &mut [u16],
    base: isize,
    xs: isize,
    ys: isize,
    tc1: i32,
    tc2: i32,
    tc3: i32,
    no_p: bool,
    no_q: bool,
    max: i32,
) {
    for d in 0..4isize {
        // Snapshot: every output below is a function of the *input* samples.
        let p3 = at(pix, base, xs, ys, -4, d);
        let p2 = at(pix, base, xs, ys, -3, d);
        let p1 = at(pix, base, xs, ys, -2, d);
        let p0 = at(pix, base, xs, ys, -1, d);
        let q0 = at(pix, base, xs, ys, 0, d);
        let q1 = at(pix, base, xs, ys, 1, d);
        let q2 = at(pix, base, xs, ys, 2, d);
        let q3 = at(pix, base, xs, ys, 3, d);
        if !no_p {
            let v = p0 + (((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3) - p0).clamp(-tc3, tc3);
            put(pix, base, xs, ys, -1, d, v, max);
            let v = p1 + (((p2 + p1 + p0 + q0 + 2) >> 2) - p1).clamp(-tc2, tc2);
            put(pix, base, xs, ys, -2, d, v, max);
            let v = p2 + (((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3) - p2).clamp(-tc1, tc1);
            put(pix, base, xs, ys, -3, d, v, max);
        }
        if !no_q {
            let v = q0 + (((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3) - q0).clamp(-tc3, tc3);
            put(pix, base, xs, ys, 0, d, v, max);
            let v = q1 + (((p0 + q0 + q1 + q2 + 2) >> 2) - q1).clamp(-tc2, tc2);
            put(pix, base, xs, ys, 1, d, v, max);
            let v = q2 + (((2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) >> 3) - q2).clamp(-tc1, tc1);
            put(pix, base, xs, ys, 2, d, v, max);
        }
    }
}

/// The weak luma filter: `p0`/`q0` always, `p1`/`q1` only when that side is
/// flat enough (`nd_p`/`nd_q`).
fn luma_weak(
    pix: &mut [u16],
    base: isize,
    xs: isize,
    ys: isize,
    tc: i32,
    no_p: bool,
    no_q: bool,
    nd_p: i32,
    nd_q: i32,
    max: i32,
) {
    let tc_2 = tc >> 1;
    for d in 0..4isize {
        let p2 = at(pix, base, xs, ys, -3, d);
        let p1 = at(pix, base, xs, ys, -2, d);
        let p0 = at(pix, base, xs, ys, -1, d);
        let q0 = at(pix, base, xs, ys, 0, d);
        let q1 = at(pix, base, xs, ys, 1, d);
        let q2 = at(pix, base, xs, ys, 2, d);
        let mut delta0 = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4;
        // A step larger than `10 * tc` is a real edge in the picture, not
        // blocking — filtering it would blur actual detail, so the whole line
        // is skipped rather than clipped.
        if delta0.abs() < 10 * tc {
            delta0 = delta0.clamp(-tc, tc);
            if !no_p {
                put(pix, base, xs, ys, -1, d, p0 + delta0, max);
            }
            if !no_q {
                put(pix, base, xs, ys, 0, d, q0 - delta0, max);
            }
            if !no_p && nd_p > 1 {
                let dp1 = ((((p2 + p0 + 1) >> 1) - p1 + delta0) >> 1).clamp(-tc_2, tc_2);
                put(pix, base, xs, ys, -2, d, p1 + dp1, max);
            }
            if !no_q && nd_q > 1 {
                let dq1 = ((((q2 + q0 + 1) >> 1) - q1 - delta0) >> 1).clamp(-tc_2, tc_2);
                put(pix, base, xs, ys, 1, d, q1 + dq1, max);
            }
        }
    }
}

/// Filter one 8-sample luma edge, as two independently decided 4-line halves.
///
/// `tcs`/`no_p`/`no_q` are per half. `beta` and `tc` arrive already looked up.
pub fn filter_luma_edge(
    pix: &mut [u16],
    base: isize,
    xs: isize,
    ys: isize,
    beta: i32,
    tcs: [i32; 2],
    no_p: [bool; 2],
    no_q: [bool; 2],
    max: i32,
) {
    for j in 0..2usize {
        let b = base + (j as isize) * 4 * ys;
        // The decision uses lines 0 and 3 of this half only.
        let dp0 = (at(pix, b, xs, ys, -3, 0) - 2 * at(pix, b, xs, ys, -2, 0)
            + at(pix, b, xs, ys, -1, 0))
        .abs();
        let dq0 = (at(pix, b, xs, ys, 2, 0) - 2 * at(pix, b, xs, ys, 1, 0)
            + at(pix, b, xs, ys, 0, 0))
        .abs();
        let dp3 = (at(pix, b, xs, ys, -3, 3) - 2 * at(pix, b, xs, ys, -2, 3)
            + at(pix, b, xs, ys, -1, 3))
        .abs();
        let dq3 = (at(pix, b, xs, ys, 2, 3) - 2 * at(pix, b, xs, ys, 1, 3)
            + at(pix, b, xs, ys, 0, 3))
        .abs();
        let d0 = dp0 + dq0;
        let d3 = dp3 + dq3;
        let tc = tcs[j];
        if d0 + d3 >= beta {
            continue;
        }
        let beta_3 = beta >> 3;
        let beta_2 = beta >> 2;
        let tc25 = (tc * 5 + 1) >> 1;
        let strong = (at(pix, b, xs, ys, -4, 0) - at(pix, b, xs, ys, -1, 0)).abs()
            + (at(pix, b, xs, ys, 3, 0) - at(pix, b, xs, ys, 0, 0)).abs()
            < beta_3
            && (at(pix, b, xs, ys, -1, 0) - at(pix, b, xs, ys, 0, 0)).abs() < tc25
            && (at(pix, b, xs, ys, -4, 3) - at(pix, b, xs, ys, -1, 3)).abs()
                + (at(pix, b, xs, ys, 3, 3) - at(pix, b, xs, ys, 0, 3)).abs()
                < beta_3
            && (at(pix, b, xs, ys, -1, 3) - at(pix, b, xs, ys, 0, 3)).abs() < tc25
            && (d0 << 1) < beta_2
            && (d3 << 1) < beta_2;
        if strong {
            let tc2 = tc << 1;
            luma_strong(pix, b, xs, ys, tc2, tc2, tc2, no_p[j], no_q[j], max);
        } else {
            let nd_p = if dp0 + dp3 < ((beta + (beta >> 1)) >> 3) { 2 } else { 1 };
            let nd_q = if dq0 + dq3 < ((beta + (beta >> 1)) >> 3) { 2 } else { 1 };
            luma_weak(pix, b, xs, ys, tc, no_p[j], no_q[j], nd_p, nd_q, max);
        }
    }
}

/// Filter one chroma edge: `p0`/`q0` only, one tap each way, and **no
/// strong/weak decision at all** — chroma is filtered whenever `tc > 0`, which
/// in practice means only across intra edges (`bS == 2`).
pub fn filter_chroma_edge(
    pix: &mut [u16],
    base: isize,
    xs: isize,
    ys: isize,
    tcs: [i32; 2],
    no_p: [bool; 2],
    no_q: [bool; 2],
    max: i32,
) {
    for j in 0..2usize {
        let tc = tcs[j];
        if tc <= 0 {
            continue;
        }
        let b = base + (j as isize) * 4 * ys;
        for d in 0..4isize {
            let p1 = at(pix, b, xs, ys, -2, d);
            let p0 = at(pix, b, xs, ys, -1, d);
            let q0 = at(pix, b, xs, ys, 0, d);
            let q1 = at(pix, b, xs, ys, 1, d);
            let delta0 = ((((q0 - p0) * 4) + p1 - q1 + 4) >> 3).clamp(-tc, tc);
            if !no_p[j] {
                put(pix, b, xs, ys, -1, d, p0 + delta0, max);
            }
            if !no_q[j] {
                put(pix, b, xs, ys, 0, d, q0 - delta0, max);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Build an 8-wide, 8-tall plane where each row is `p3 p2 p1 p0 | q0 q1 q2 q3`
    /// so a vertical edge sits at column 4.
    fn plane(rows: &[[u16; 8]]) -> alloc::vec::Vec<u16> {
        let mut v = vec![0u16; 8 * 8];
        for (y, r) in rows.iter().enumerate() {
            v[y * 8..y * 8 + 8].copy_from_slice(r);
        }
        v
    }

    #[test_case]
    fn a_flat_edge_is_left_exactly_alone() {
        // Nothing to deblock: every sample equal. Any change here is the filter
        // inventing an edge, which over a GOP is a slow contrast loss.
        let mut p = plane(&[[128; 8]; 8]);
        let before = p.clone();
        filter_luma_edge(&mut p, 4, 1, 8, beta(30, 0), [tc(30, 2, 0); 2], [false; 2], [false; 2]);
        assert_eq!(p, before);
        let mut p = plane(&[[128; 8]; 8]);
        filter_chroma_edge(&mut p, 4, 1, 8, [tc(30, 2, 0, 255); 2], [false; 2], [false; 2]);
        assert_eq!(p, before);
    }

    /// A step larger than the filter's reach is real picture content and must
    /// survive — this is what `|delta0| < 10 * tc` and the `d0 + d3 < beta`
    /// gate exist for. A decoder missing them blurs every genuine edge that
    /// happens to land on the 8x8 grid.
    #[test_case]
    fn a_genuine_hard_edge_is_not_filtered() {
        let mut rows = [[0u16; 8]; 8];
        for r in rows.iter_mut() {
            *r = [20, 20, 20, 20, 230, 230, 230, 230];
        }
        let mut p = plane(&rows);
        let before = p.clone();
        filter_luma_edge(&mut p, 4, 1, 8, beta(26, 0), [tc(26, 2, 0); 2], [false; 2], [false; 2]);
        assert_eq!(p, before, "a 210-level step is content, not blocking");
    }

    /// A small step across an otherwise flat block is exactly what deblocking
    /// exists to remove, and the strong filter should smooth three samples each
    /// side into a ramp.
    #[test_case]
    fn a_small_step_across_a_flat_block_is_smoothed_by_the_strong_filter() {
        let mut rows = [[0u16; 8]; 8];
        for r in rows.iter_mut() {
            *r = [100, 100, 100, 100, 108, 108, 108, 108];
        }
        let mut p = plane(&rows);
        filter_luma_edge(&mut p, 4, 1, 8, beta(37, 0), [tc(37, 2, 0); 2], [false; 2], [false; 2]);
        for y in 0..8 {
            let row = &p[y * 8..y * 8 + 8];
            // The step is spread: p0 rises, q0 falls, and the outermost
            // samples (p3, q3) are never written by any luma filter.
            assert_eq!(row[0], 100, "p3 must never be written");
            assert_eq!(row[7], 108, "q3 must never be written");
            assert!(row[3] > 100 && row[3] < 108, "p0 not smoothed: {row:?}");
            assert!(row[4] > 100 && row[4] < 108, "q0 not smoothed: {row:?}");
            assert!(row[3] <= row[4], "the ramp inverted: {row:?}");
            // Monotone across the edge.
            for i in 0..7 {
                assert!(row[i] <= row[i + 1], "not monotone: {row:?}");
            }
        }
    }

    /// `no_p` / `no_q` mask one side — used where the neighbouring block is
    /// lossless or PCM and must be reproduced exactly. A filter that ignores
    /// them silently breaks the one guarantee lossless coding makes.
    #[test_case]
    fn no_p_and_no_q_leave_that_side_untouched() {
        let mut rows = [[0u16; 8]; 8];
        for r in rows.iter_mut() {
            *r = [100, 100, 100, 100, 108, 108, 108, 108];
        }
        let mut p = plane(&rows);
        filter_luma_edge(&mut p, 4, 1, 8, beta(37, 0), [tc(37, 2, 0); 2], [true; 2], [false; 2]);
        for y in 0..8 {
            let row = &p[y * 8..y * 8 + 8];
            assert_eq!(&row[..4], &[100, 100, 100, 100], "p side written");
            assert!(row[4] != 108, "q side should still be filtered");
        }
    }

    /// The two halves of an 8-sample edge decide independently — one half can
    /// be filtered while the other is not. Deciding once for all eight is the
    /// mistake that comes from reading the specification's 4-line loop as an
    /// implementation detail.
    #[test_case]
    fn the_two_halves_of_an_edge_are_decided_separately() {
        let mut rows = [[0u16; 8]; 8];
        for (y, r) in rows.iter_mut().enumerate() {
            *r = if y < 4 {
                [100, 100, 100, 100, 108, 108, 108, 108] // blocking
            } else {
                [20, 20, 20, 20, 230, 230, 230, 230] // real content
            };
        }
        let mut p = plane(&rows);
        filter_luma_edge(&mut p, 4, 1, 8, beta(37, 0), [tc(37, 2, 0); 2], [false; 2], [false; 2]);
        for y in 0..4 {
            assert!(p[y * 8 + 3] != 100, "top half should be filtered");
        }
        for y in 4..8 {
            assert_eq!(&p[y * 8..y * 8 + 8], &[20, 20, 20, 20, 230, 230, 230, 230]);
        }
    }

    /// Horizontal and vertical edges are the same filter with the strides
    /// swapped, so the result must be an exact transpose. If it is not, one of
    /// the two orientations has its taps in the wrong place — and a picture
    /// filtered correctly one way and wrongly the other looks like a directional
    /// artefact, not like a deblocking bug.
    #[test_case]
    fn a_horizontal_edge_is_the_transpose_of_a_vertical_one() {
        let mut rows = [[0u16; 8]; 8];
        for (y, r) in rows.iter_mut().enumerate() {
            for x in 0..8 {
                r[x] = (100 + x * 2 + y) as u16;
            }
            r[4] = r[4].wrapping_add(6);
            r[5] = r[5].wrapping_add(6);
            r[6] = r[6].wrapping_add(6);
            r[7] = r[7].wrapping_add(6);
        }
        let vert = plane(&rows);
        let mut a = vert.clone();
        filter_luma_edge(&mut a, 4, 1, 8, beta(35, 0), [tc(35, 2, 0); 2], [false; 2], [false; 2]);

        // Transpose the input, filter as a horizontal edge (across = stride).
        let mut b = vec![0u16; 64];
        for y in 0..8 {
            for x in 0..8 {
                b[x * 8 + y] = vert[y * 8 + x];
            }
        }
        filter_luma_edge(&mut b, 4 * 8, 8, 1, beta(35, 0), [tc(35, 2, 0); 2], [false; 2], [false; 2]);
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(a[y * 8 + x], b[x * 8 + y], "at ({x},{y})");
            }
        }
    }

    /// Chroma touches exactly one sample each side and never `p1`/`q1`.
    #[test_case]
    fn chroma_filters_only_the_innermost_sample() {
        let mut rows = [[0u16; 8]; 8];
        for r in rows.iter_mut() {
            *r = [100, 100, 100, 100, 120, 120, 120, 120];
        }
        let mut p = plane(&rows);
        filter_chroma_edge(&mut p, 4, 1, 8, [tc(40, 2, 0, 255); 2], [false; 2], [false; 2]);
        for y in 0..8 {
            let row = &p[y * 8..y * 8 + 8];
            assert_eq!(&row[..3], &[100, 100, 100], "chroma wrote p1 or beyond");
            assert_eq!(&row[5..], &[120, 120, 120], "chroma wrote q1 or beyond");
            assert!(row[3] > 100 && row[4] < 120, "p0/q0 not filtered: {row:?}");
        }
        // tc == 0 means no filtering at all, and that is how a bS of 0 or 1
        // reaches chroma.
        let mut p = plane(&rows);
        let before = p.clone();
        filter_chroma_edge(&mut p, 4, 1, 8, [0; 2], [false; 2], [false; 2], 255);
        assert_eq!(p, before);
    }

    /// `bS` and the offsets move the curve in the directions they are supposed
    /// to, and both saturate rather than running off the table.
    #[test_case]
    fn tc_and_beta_lookups_are_bounded_and_ordered() {
        for qp in 0..=51 {
            assert!(tc(qp, 2, 0) >= tc(qp, 1, 0), "bS 2 must filter at least as hard at qp {qp}");
            assert!(beta(qp, 0) <= 64);
        }
        // An odd offset is rounded down to even — the syntax element is in
        // units of two.
        assert_eq!(tc(30, 1, 3), tc(30, 1, 2));
        assert_eq!(tc(30, 1, -3), tc(30, 1, -4));
        // Saturation at both ends, on the 54-entry tc table and 52-entry beta.
        assert_eq!(tc(51, 2, 100), tb::TC_TABLE[53] as i32);
        assert_eq!(tc(0, 1, -100), 0);
        assert_eq!(beta(51, 100), tb::BETA_TABLE[51] as i32);
        assert_eq!(beta(0, -100), 0);
        // Below QP 16 nothing is filtered at all, which is what makes a
        // high-quality stream bit-exact through the loop filter.
        assert_eq!(beta(15, 0), 0);
    }
}
