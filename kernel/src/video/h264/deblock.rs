//! H.264 in-loop deblocking filter (§8.7). Standard alpha/beta/tc0 tables
//! (from the FFmpeg loopfilter source, offset-normalised); the filter logic
//! is validated bit-exact against PyAV on default-deblock clips.
//!
//! Operates on the reconstructed i32 planes in place, using per-4x4-block
//! metadata (intra flag, nnz, MV, ref) to derive boundary strength.

pub const ALPHA: [i32; 52] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 17, 20, 22, 25, 28, 32, 36, 40, 45, 50, 56, 63, 71, 80, 90, 101, 113, 127, 144, 162, 182, 203, 226, 255, 255];
pub const BETA: [i32; 52] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18];
pub const TC0: [[i32; 4]; 52] = [
    [-1, 0, 0, 0],
    [-1, 0, 0, 0],
    [-1, 0, 0, 0],
    [-1, 0, 0, 0],
    [-1, 0, 0, 0],
    [-1, 0, 0, 0],
    [-1, 0, 0, 0],
    [-1, 0, 0, 0],
    [-1, 0, 0, 0],
    [-1, 0, 0, 0],
    [-1, 0, 0, 0],
    [-1, 0, 0, 0],
    [-1, 0, 0, 0],
    [-1, 0, 0, 0],
    [-1, 0, 0, 0],
    [-1, 0, 0, 0],
    [-1, 0, 0, 0],
    [-1, 0, 0, 1],
    [-1, 0, 0, 1],
    [-1, 0, 0, 1],
    [-1, 0, 0, 1],
    [-1, 0, 1, 1],
    [-1, 0, 1, 1],
    [-1, 1, 1, 1],
    [-1, 1, 1, 1],
    [-1, 1, 1, 1],
    [-1, 1, 1, 1],
    [-1, 1, 1, 2],
    [-1, 1, 1, 2],
    [-1, 1, 1, 2],
    [-1, 1, 1, 2],
    [-1, 1, 2, 3],
    [-1, 1, 2, 3],
    [-1, 2, 2, 3],
    [-1, 2, 2, 4],
    [-1, 2, 3, 4],
    [-1, 2, 3, 4],
    [-1, 3, 3, 5],
    [-1, 3, 4, 6],
    [-1, 3, 4, 6],
    [-1, 4, 5, 7],
    [-1, 4, 5, 8],
    [-1, 4, 6, 9],
    [-1, 5, 7, 10],
    [-1, 6, 8, 11],
    [-1, 6, 8, 13],
    [-1, 7, 10, 14],
    [-1, 8, 11, 16],
    [-1, 9, 12, 18],
    [-1, 10, 13, 20],
    [-1, 11, 15, 23],
    [-1, 13, 17, 25],
];

fn clampi(v: i32, lo: i32, hi: i32) -> i32 {
    v.clamp(lo, hi)
}

fn qpc(qpi: i32) -> i32 {
    const TAB: [i32; 22] = [29, 30, 31, 32, 32, 33, 34, 34, 35, 35, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 39];
    if qpi < 30 {
        qpi
    } else {
        TAB[(qpi - 30) as usize]
    }
}

/// Per-4×4-block metadata the boundary-strength derivation needs.
pub struct Meta<'a> {
    pub mbw: usize,
    pub ny4: usize,
    pub mbqp: &'a [i32],
    pub mbintra: &'a [bool],
    pub nnz_y: &'a [i32],
    pub mvx: &'a [i32],
    pub mvy: &'a [i32],
    pub refi: &'a [i32],
    pub aoff: i32,
    pub boff: i32,
    pub chroma_qp_off: i32,
}

fn bs(p_intra: bool, q_intra: bool, mb_edge: bool, p_nnz: i32, q_nnz: i32, pmv: (i32, i32), qmv: (i32, i32), pref: i32, qref: i32) -> i32 {
    if p_intra || q_intra {
        return if mb_edge { 4 } else { 3 };
    }
    if p_nnz > 0 || q_nnz > 0 {
        return 2;
    }
    if pref != qref || (pmv.0 - qmv.0).abs() >= 4 || (pmv.1 - qmv.1).abs() >= 4 {
        return 1;
    }
    0
}

fn filt_luma(pl: &mut [i32], off: usize, step: isize, b_s: i32, alpha: i32, beta: i32, tc0: i32) {
    let g = |pl: &[i32], k: isize| pl[(off as isize + k * step) as usize];
    let at = |k: isize| (off as isize + k * step) as usize;
    let (p0, p1, p2, p3) = (g(pl, -1), g(pl, -2), g(pl, -3), g(pl, -4));
    let (q0, q1, q2, q3) = (g(pl, 0), g(pl, 1), g(pl, 2), g(pl, 3));
    if (p0 - q0).abs() >= alpha || (p1 - p0).abs() >= beta || (q1 - q0).abs() >= beta {
        return;
    }
    let ap = (p2 - p0).abs();
    let aq = (q2 - q0).abs();
    if b_s < 4 {
        // tc can be nonzero even when tc0 == 0 (the ap/aq<beta increments) — the
        // p0/q0 filter still applies; only the p1/q1 filter is gated on tc0 > 0.
        let tc = tc0 + if ap < beta { 1 } else { 0 } + if aq < beta { 1 } else { 0 };
        let delta = clampi((((q0 - p0) << 2) + (p1 - q1) + 4) >> 3, -tc, tc);
        pl[at(-1)] = clampi(p0 + delta, 0, 255);
        pl[at(0)] = clampi(q0 - delta, 0, 255);
        if ap < beta && tc0 > 0 {
            let d = clampi((p2 + ((p0 + q0 + 1) >> 1) - 2 * p1) >> 1, -tc0, tc0);
            pl[at(-2)] = p1 + d;
        }
        if aq < beta && tc0 > 0 {
            let d = clampi((q2 + ((p0 + q0 + 1) >> 1) - 2 * q1) >> 1, -tc0, tc0);
            pl[at(1)] = q1 + d;
        }
    } else if (p0 - q0).abs() < ((alpha >> 2) + 2) {
        if ap < beta {
            pl[at(-1)] = (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3;
            pl[at(-2)] = (p2 + p1 + p0 + q0 + 2) >> 2;
            pl[at(-3)] = (2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3;
        } else {
            pl[at(-1)] = (2 * p1 + p0 + q1 + 2) >> 2;
        }
        if aq < beta {
            pl[at(0)] = (q2 + 2 * q1 + 2 * q0 + 2 * p0 + p1 + 4) >> 3;
            pl[at(1)] = (q2 + q1 + q0 + p0 + 2) >> 2;
            pl[at(2)] = (2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) >> 3;
        } else {
            pl[at(0)] = (2 * q1 + q0 + p1 + 2) >> 2;
        }
    } else {
        pl[at(-1)] = (2 * p1 + p0 + q1 + 2) >> 2;
        pl[at(0)] = (2 * q1 + q0 + p1 + 2) >> 2;
    }
}

fn filt_chroma(pl: &mut [i32], off: usize, step: isize, b_s: i32, alpha: i32, beta: i32, tc0: i32) {
    let g = |pl: &[i32], k: isize| pl[(off as isize + k * step) as usize];
    let at = |k: isize| (off as isize + k * step) as usize;
    let (p0, p1) = (g(pl, -1), g(pl, -2));
    let (q0, q1) = (g(pl, 0), g(pl, 1));
    if (p0 - q0).abs() >= alpha || (p1 - p0).abs() >= beta || (q1 - q0).abs() >= beta {
        return;
    }
    if b_s < 4 {
        let tc = tc0 + 1;
        let delta = clampi((((q0 - p0) << 2) + (p1 - q1) + 4) >> 3, -tc, tc);
        pl[at(-1)] = clampi(p0 + delta, 0, 255);
        pl[at(0)] = clampi(q0 - delta, 0, 255);
    } else {
        pl[at(-1)] = (2 * p1 + p0 + q1 + 2) >> 2;
        pl[at(0)] = (2 * q1 + q0 + p1 + 2) >> 2;
    }
}

/// Deblock a whole frame in place: luma `y` (w×h) and chroma `cb`/`cr` (cw×ch).
#[allow(clippy::too_many_arguments)]
pub fn deblock(y: &mut [i32], cb: &mut [i32], cr: &mut [i32], w: usize, h: usize, cw: usize, mbw: usize, mbh: usize, m: &Meta) {
    let _ = h;
    let blk = |px4: usize, py4: usize| py4 * m.ny4 + px4;
    for mb_y in 0..mbh {
        for mb_x in 0..mbw {
            let mb = mb_y * mbw + mb_x;
            // Vertical then horizontal edges; luma at 0/4/8/12, chroma at 0/4.
            for &(vert, edges) in &[(true, [0usize, 4, 8, 12]), (false, [0, 4, 8, 12])] {
                for &e in &edges {
                    if e == 0 && ((vert && mb_x == 0) || (!vert && mb_y == 0)) {
                        continue;
                    }
                    for seg in 0..4 {
                        let (px4, py4, qx4, qy4, nb_mb) = if vert {
                            let qx4 = mb_x * 4 + e / 4;
                            let qy4 = mb_y * 4 + seg;
                            (qx4 - 1, qy4, qx4, qy4, if e == 0 { mb - 1 } else { mb })
                        } else {
                            let qx4 = mb_x * 4 + seg;
                            let qy4 = mb_y * 4 + e / 4;
                            (qx4, qy4 - 1, qx4, qy4, if e == 0 { (mb_y - 1) * mbw + mb_x } else { mb })
                        };
                        let b_s = bs(
                            m.mbintra[nb_mb], m.mbintra[mb], e == 0,
                            m.nnz_y[blk(px4, py4)], m.nnz_y[blk(qx4, qy4)],
                            (m.mvx[blk(px4, py4)], m.mvy[blk(px4, py4)]), (m.mvx[blk(qx4, qy4)], m.mvy[blk(qx4, qy4)]),
                            m.refi[blk(px4, py4)], m.refi[blk(qx4, qy4)],
                        );
                        if b_s == 0 {
                            continue;
                        }
                        let qpav = (m.mbqp[nb_mb] + m.mbqp[mb] + 1) >> 1;
                        let ia = clampi(qpav + m.aoff, 0, 51) as usize;
                        let ib = clampi(qpav + m.boff, 0, 51) as usize;
                        let (alpha, beta) = (ALPHA[ia], BETA[ib]);
                        if alpha == 0 || beta == 0 {
                            continue;
                        }
                        let tc0 = if b_s < 4 { TC0[ia][b_s as usize] } else { 0 };
                        for k in 0..4 {
                            let (off, step) = if vert {
                                ((mb_y * 16 + seg * 4 + k) * w + mb_x * 16 + e, 1isize)
                            } else {
                                ((mb_y * 16 + e) * w + mb_x * 16 + seg * 4 + k, w as isize)
                            };
                            filt_luma(y, off, step, b_s, alpha, beta, tc0);
                        }
                    }
                    // Chroma at edges 0 and 8 → chroma offset 0 and 4.
                    if e != 0 && e != 8 {
                        continue;
                    }
                    let ce = e / 2;
                    for seg in 0..4 {
                        let (px4, py4, qx4, qy4, nb_mb) = if vert {
                            let qx4 = mb_x * 4 + e / 4;
                            let qy4 = mb_y * 4 + seg;
                            (qx4 - 1, qy4, qx4, qy4, if e == 0 { mb - 1 } else { mb })
                        } else {
                            let qx4 = mb_x * 4 + seg;
                            let qy4 = mb_y * 4 + e / 4;
                            (qx4, qy4 - 1, qx4, qy4, if e == 0 { (mb_y - 1) * mbw + mb_x } else { mb })
                        };
                        let b_s = bs(
                            m.mbintra[nb_mb], m.mbintra[mb], e == 0,
                            m.nnz_y[blk(px4, py4)], m.nnz_y[blk(qx4, qy4)],
                            (m.mvx[blk(px4, py4)], m.mvy[blk(px4, py4)]), (m.mvx[blk(qx4, qy4)], m.mvy[blk(qx4, qy4)]),
                            m.refi[blk(px4, py4)], m.refi[blk(qx4, qy4)],
                        );
                        if b_s == 0 {
                            continue;
                        }
                        // Chroma edge QP: convert each MB's luma QP to chroma QP,
                        // then average (§8.7.2.2) — differs across slices whose QPs differ.
                        let cqp = (qpc(clampi(m.mbqp[nb_mb] + m.chroma_qp_off, 0, 51)) + qpc(clampi(m.mbqp[mb] + m.chroma_qp_off, 0, 51)) + 1) >> 1;
                        let ia = clampi(cqp + m.aoff, 0, 51) as usize;
                        let ib = clampi(cqp + m.boff, 0, 51) as usize;
                        let (alpha, beta) = (ALPHA[ia], BETA[ib]);
                        if alpha == 0 || beta == 0 {
                            continue;
                        }
                        let tc0 = if b_s < 4 { TC0[ia][b_s as usize] } else { 0 };
                        for k in 0..2 {
                            let (off, step) = if vert {
                                ((mb_y * 8 + seg * 2 + k) * cw + mb_x * 8 + ce, 1isize)
                            } else {
                                ((mb_y * 8 + ce) * cw + mb_x * 8 + seg * 2 + k, cw as isize)
                            };
                            filt_chroma(cb, off, step, b_s, alpha, beta, tc0);
                            filt_chroma(cr, off, step, b_s, alpha, beta, tc0);
                        }
                    }
                }
            }
        }
    }
}
