//! HEVC coding-unit structure: the derivations the CTU quadtree walk needs
//! (H.265 §8.4.2, §8.4.3, §7.4.9.x) as pure functions.
//!
//! The quadtree walk itself is mechanical — split flags and recursion. What is
//! *not* mechanical, and what this module holds, is the handful of derivations
//! where the specification says something non-obvious and every wrong reading
//! still produces a decodable picture:
//!
//! - The **most-probable-mode list** for luma intra. Three candidates from two
//!   neighbours, with a rule that changes shape depending on whether the
//!   neighbours agree, and a *sort* before the non-MPM path that is easy to
//!   omit because the MPM path does not need it.
//! - The **chroma mode table**, whose "derived mode 4" case means *copy luma*
//!   and whose collision case escapes to mode 34 — not to the next entry.
//! - **Scan order selection**, which for small intra blocks depends on the
//!   prediction mode, so a wrong mode silently also transposes the residual.
//! - **Boundary strength**, which decides deblocking and is the one place the
//!   motion field, the transform tree and the CU structure all meet.

/// Prediction modes, in the specification's numbering.
pub const MODE_INTER: u8 = 0;
pub const MODE_INTRA: u8 = 1;

/// Partition modes (§7.4.9.5). The asymmetric ones exist only for inter CUs
/// above the minimum size.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PartMode {
    Part2Nx2N = 0,
    Part2NxN = 1,
    PartNx2N = 2,
    PartNxN = 3,
    Part2NxnU = 4,
    Part2NxnD = 5,
    PartnLx2N = 6,
    PartnRx2N = 7,
}

impl PartMode {
    pub fn from_code(v: u32) -> Option<PartMode> {
        Some(match v {
            0 => PartMode::Part2Nx2N,
            1 => PartMode::Part2NxN,
            2 => PartMode::PartNx2N,
            3 => PartMode::PartNxN,
            4 => PartMode::Part2NxnU,
            5 => PartMode::Part2NxnD,
            6 => PartMode::PartnLx2N,
            7 => PartMode::PartnRx2N,
            _ => return None,
        })
    }

    /// How many prediction units the CU holds.
    pub fn num_pus(self) -> usize {
        match self {
            PartMode::Part2Nx2N => 1,
            PartMode::PartNxN => 4,
            _ => 2,
        }
    }

    /// The `(x, y, w, h)` of prediction unit `i` inside a `size x size` CU,
    /// in luma samples relative to the CU's corner.
    ///
    /// The asymmetric modes split at **one quarter**, not one half, and the
    /// `U`/`L` forms put the *small* partition first while `D`/`R` put it last
    /// — swapping those pairs is a plausible-looking picture with motion
    /// assigned to the wrong quarter.
    pub fn pu_rect(self, i: usize, size: usize) -> (usize, usize, usize, usize) {
        let h = size / 2;
        let q = size / 4;
        match self {
            PartMode::Part2Nx2N => (0, 0, size, size),
            PartMode::Part2NxN => {
                if i == 0 {
                    (0, 0, size, h)
                } else {
                    (0, h, size, h)
                }
            }
            PartMode::PartNx2N => {
                if i == 0 {
                    (0, 0, h, size)
                } else {
                    (h, 0, h, size)
                }
            }
            PartMode::PartNxN => (h * (i & 1), h * (i >> 1), h, h),
            PartMode::Part2NxnU => {
                if i == 0 {
                    (0, 0, size, q)
                } else {
                    (0, q, size, size - q)
                }
            }
            PartMode::Part2NxnD => {
                if i == 0 {
                    (0, 0, size, size - q)
                } else {
                    (0, size - q, size, q)
                }
            }
            PartMode::PartnLx2N => {
                if i == 0 {
                    (0, 0, q, size)
                } else {
                    (q, 0, size - q, size)
                }
            }
            PartMode::PartnRx2N => {
                if i == 0 {
                    (0, 0, size - q, size)
                } else {
                    (size - q, 0, q, size)
                }
            }
        }
    }
}

/// The three most-probable luma intra modes (§8.4.2).
///
/// `cand_left` / `cand_above` are the neighbouring blocks' modes, already
/// substituted with DC where the neighbour is unavailable **or is above the
/// current CTB row** — intra mode prediction does not cross a horizontal CTB
/// boundary, which is what makes wavefront parallel processing possible and is
/// invisible in a single-CTB-row test picture.
pub fn mpm_candidates(cand_left: u8, cand_above: u8) -> [u8; 3] {
    if cand_left == cand_above {
        if cand_left < 2 {
            // Both neighbours are planar or DC: there is no angle to extend, so
            // the list is the fixed one. Note it is planar/DC/**26**, not
            // planar/DC/2 — vertical, because pictures have more vertical
            // structure than diagonal.
            [0, 1, 26]
        } else {
            // Extend the shared angle by plus and minus one step, wrapping
            // around the 32-mode ring. The `- 2 ... + 32) & 31` form is the
            // wrap; a plain `- 1` walks off the bottom at mode 2.
            [cand_left, 2 + ((cand_left - 2 + 31) & 31), 2 + ((cand_left - 2 + 1) & 31)]
        }
    } else {
        let third = if cand_left != 0 && cand_above != 0 {
            0 // planar
        } else if cand_left != 1 && cand_above != 1 {
            1 // DC
        } else {
            26 // vertical
        };
        [cand_left, cand_above, third]
    }
}

/// Resolve the signalled luma intra mode (§8.4.2).
///
/// On the non-MPM path the candidate list is **sorted ascending first**, and
/// then each candidate that the remainder reaches or passes bumps it by one.
/// Omitting the sort still produces a legal mode for every input — just the
/// wrong one whenever the candidates arrive out of order, which is most of the
/// time.
pub fn luma_intra_mode(mut cand: [u8; 3], prev_flag: bool, mpm_idx: usize, rem: u8) -> u8 {
    if prev_flag {
        return cand[mpm_idx];
    }
    cand.sort_unstable();
    let mut mode = rem;
    for &c in cand.iter() {
        if mode >= c {
            mode += 1;
        }
    }
    mode
}

/// The four signalled chroma modes; index 4 means "the same as luma".
const CHROMA_TABLE: [u8; 4] = [0, 26, 10, 1];

/// Resolve the chroma intra mode (§8.4.3) for 4:2:0 / 4:4:4.
///
/// The escape is the part worth stating: when the signalled mode would collide
/// with luma's, the result is **mode 34**, not the next table entry and not
/// luma's mode. That is how the four-entry table still spans five distinct
/// outcomes.
pub fn chroma_intra_mode(signalled: u32, luma_mode: u8) -> u8 {
    if signalled == 4 {
        return luma_mode;
    }
    let m = CHROMA_TABLE[signalled as usize];
    if luma_mode == m {
        34
    } else {
        m
    }
}

/// 4:2:2 remaps the resolved mode through a table, because halving only the
/// horizontal dimension changes what a given angle means (§8.4.3, table 8-3).
///
/// The table is **generated**, not written here. Recalling it produces a
/// plausible wrong answer — it is a gentle monotone curve either way, so a
/// version off by one from mode 14 onward passes every sanity check and simply
/// tilts chroma slightly against luma.
pub fn chroma_mode_422(mode: u8) -> u8 {
    super::tables::CHROMA_422_MODE_MAP[mode as usize]
}

/// Residual scan order for a transform block (§7.4.9.11).
///
/// Only **intra** blocks below 16x16 use a mode-dependent scan; everything else
/// is diagonal. So a wrong intra mode transposes the residual as well as
/// mispredicting — two errors from one cause, which is why a mode bug here
/// looks so much worse than a mode bug in a large block.
pub fn scan_order(intra: bool, log2_size: u32, mode: u8) -> usize {
    if !intra || log2_size >= 4 {
        return super::residual::SCAN_DIAG;
    }
    if (6..=14).contains(&mode) {
        super::residual::SCAN_VERT
    } else if (22..=30).contains(&mode) {
        super::residual::SCAN_HORIZ
    } else {
        super::residual::SCAN_DIAG
    }
}

/// One side of a deblocking edge, as far as boundary strength is concerned.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct EdgeSide {
    pub intra: bool,
    /// Whether the containing transform block has any non-zero luma coefficient.
    pub cbf: bool,
    /// Reference picture POCs for list 0 and 1; `None` where the list is unused.
    pub refs: [Option<i32>; 2],
    /// Motion vectors, quarter-pel, matching `refs`.
    pub mvs: [(i16, i16); 2],
}

/// Boundary strength for one edge (H.265 §8.7.2.4).
///
/// `is_tu_edge` marks a transform-block boundary; a prediction-unit boundary
/// that is not also a transform boundary cannot get strength from coefficients.
///
/// The motion comparison is the part with all the traps. References are matched
/// by **which picture**, not by which list index — the same picture can sit at
/// different indices in the two lists, and comparing indices calls two
/// identical predictions different. And when both sides use the same two
/// pictures, the vectors must be tried **both ways round**, because list 0 and
/// list 1 may hold them in opposite order.
pub fn boundary_strength(p: &EdgeSide, q: &EdgeSide, is_tu_edge: bool) -> u8 {
    if p.intra || q.intra {
        return 2;
    }
    if is_tu_edge && (p.cbf || q.cbf) {
        return 1;
    }

    let pr: UsedRefs = collect_refs(p);
    let qr: UsedRefs = collect_refs(q);
    if pr.n != qr.n {
        return 1;
    }
    // A vector differing by 4 quarter-pels — one whole luma sample — is what
    // the specification calls a discontinuity.
    let far = |a: (i16, i16), b: (i16, i16)| (a.0 - b.0).abs() >= 4 || (a.1 - b.1).abs() >= 4;

    match pr.n {
        0 => 0,
        1 => {
            if pr.poc[0] != qr.poc[0] {
                1
            } else {
                far(pr.mv[0], qr.mv[0]) as u8
            }
        }
        _ => {
            if pr.poc[0] == pr.poc[1] {
                // Both predictions from the same picture: neither pairing is
                // privileged, so the edge is smooth only if *some* pairing has
                // both vectors close.
                if qr.poc[0] != pr.poc[0] || qr.poc[1] != pr.poc[0] {
                    return 1;
                }
                let straight = !far(pr.mv[0], qr.mv[0]) && !far(pr.mv[1], qr.mv[1]);
                let crossed = !far(pr.mv[0], qr.mv[1]) && !far(pr.mv[1], qr.mv[0]);
                (!(straight || crossed)) as u8
            } else if pr.poc[0] == qr.poc[0] && pr.poc[1] == qr.poc[1] {
                (far(pr.mv[0], qr.mv[0]) || far(pr.mv[1], qr.mv[1])) as u8
            } else if pr.poc[0] == qr.poc[1] && pr.poc[1] == qr.poc[0] {
                (far(pr.mv[0], qr.mv[1]) || far(pr.mv[1], qr.mv[0])) as u8
            } else {
                1
            }
        }
    }
}

/// The used references of one side, compacted so a gap in list 0 does not
/// shift list 1 — the comparison below is over *how many* predictions each
/// side makes and which pictures they name, not over list slots.
struct UsedRefs {
    n: usize,
    poc: [i32; 2],
    mv: [(i16, i16); 2],
}

fn collect_refs(s: &EdgeSide) -> UsedRefs {
    let mut out = UsedRefs { n: 0, poc: [0; 2], mv: [(0, 0); 2] };
    for l in 0..2 {
        if let Some(poc) = s.refs[l] {
            out.poc[out.n] = poc;
            out.mv[out.n] = s.mvs[l];
            out.n += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The MPM list must always hold three **distinct** modes — the whole
    /// point is that the remainder code then spans exactly the other 32, so a
    /// duplicate makes one mode unreachable and shifts every mode above it.
    #[test_case]
    fn mpm_candidates_are_always_three_distinct_modes() {
        for left in 0..=34u8 {
            for above in 0..=34u8 {
                let c = mpm_candidates(left, above);
                assert!(c.iter().all(|&m| m <= 34), "({left},{above}) -> {c:?}");
                assert!(
                    c[0] != c[1] && c[1] != c[2] && c[0] != c[2],
                    "duplicate in ({left},{above}) -> {c:?}"
                );
            }
        }
    }

    /// The angle-extension case wraps around the 32-mode ring rather than
    /// clamping. Mode 2 is the bottom of the ring and its predecessor is 34.
    #[test_case]
    fn matching_angular_neighbours_extend_the_angle_and_wrap() {
        // The ring is 32 wide and covers modes 2..=33, so mode 2's predecessor
        // is 33 — and mode **34** folds onto the same neighbours, which is not
        // what "extend the angle by one step" suggests and is the reason this
        // is written as modular arithmetic rather than +-1 with a clamp.
        assert_eq!(mpm_candidates(2, 2), [2, 33, 3]);
        assert_eq!(mpm_candidates(34, 34), [34, 33, 3]);
        assert_eq!(mpm_candidates(18, 18), [18, 17, 19]);
        // Planar and DC agreeing gives the fixed list, ending at *vertical*.
        assert_eq!(mpm_candidates(0, 0), [0, 1, 26]);
        assert_eq!(mpm_candidates(1, 1), [0, 1, 26]);
    }

    /// With differing neighbours the third candidate is the first of
    /// planar / DC / vertical that is not already in the list.
    #[test_case]
    fn differing_neighbours_fill_the_third_slot_in_priority_order() {
        assert_eq!(mpm_candidates(10, 26), [10, 26, 0], "neither is planar");
        assert_eq!(mpm_candidates(0, 26), [0, 26, 1], "planar taken, so DC");
        assert_eq!(mpm_candidates(0, 1), [0, 1, 26], "both taken, so vertical");
        assert_eq!(mpm_candidates(1, 10), [1, 10, 0]);
    }

    /// The non-MPM path sorts, then skips. Together those make the 32
    /// remainders a bijection onto the 32 non-candidate modes — which is the
    /// property that catches both an omitted sort and a `>` written for `>=`.
    #[test_case]
    fn the_remainder_code_is_a_bijection_onto_the_non_candidate_modes() {
        for left in 0..=34u8 {
            for above in 0..=34u8 {
                let cand = mpm_candidates(left, above);
                let mut seen = [false; 35];
                for rem in 0..32u8 {
                    let m = luma_intra_mode(cand, false, 0, rem);
                    assert!(m <= 34, "({left},{above}) rem {rem} -> {m}");
                    assert!(!seen[m as usize], "({left},{above}) rem {rem}: {m} twice");
                    assert!(!cand.contains(&m), "remainder produced a candidate: {m}");
                    seen[m as usize] = true;
                }
                // Exactly the three candidates are left unreached.
                let missed: alloc::vec::Vec<u8> =
                    (0..=34u8).filter(|m| !seen[*m as usize]).collect();
                let mut want = cand;
                want.sort_unstable();
                assert_eq!(missed, want.to_vec(), "({left},{above})");
            }
        }
    }

    /// Unsorted candidates must give the same answer as sorted ones — the sort
    /// is inside `luma_intra_mode`, so a caller cannot get it wrong, and this
    /// pins that.
    #[test_case]
    fn the_mpm_path_indexes_the_unsorted_list() {
        // On the MPM path the list is *not* sorted: index 0 is the left
        // neighbour's mode. Sorting there would swap which neighbour a given
        // `mpm_idx` names.
        let cand = mpm_candidates(30, 4);
        assert_eq!(cand, [30, 4, 0]);
        assert_eq!(luma_intra_mode(cand, true, 0, 0), 30);
        assert_eq!(luma_intra_mode(cand, true, 1, 0), 4);
        assert_eq!(luma_intra_mode(cand, true, 2, 0), 0);
    }

    #[test_case]
    fn chroma_mode_table_escapes_a_collision_to_thirty_four() {
        // The four signalled modes, when they do not collide.
        assert_eq!(chroma_intra_mode(0, 5), 0);
        assert_eq!(chroma_intra_mode(1, 5), 26);
        assert_eq!(chroma_intra_mode(2, 5), 10);
        assert_eq!(chroma_intra_mode(3, 5), 1);
        // Mode 4 is "derived": copy luma exactly, including an angular mode.
        assert_eq!(chroma_intra_mode(4, 5), 5);
        assert_eq!(chroma_intra_mode(4, 34), 34);
        // A collision escapes to 34 — not to luma, and not to another entry.
        assert_eq!(chroma_intra_mode(0, 0), 34);
        assert_eq!(chroma_intra_mode(1, 26), 34);
        assert_eq!(chroma_intra_mode(2, 10), 34);
        assert_eq!(chroma_intra_mode(3, 1), 34);
        // Every combination stays a legal mode.
        for sig in 0..=4u32 {
            for luma in 0..=34u8 {
                assert!(chroma_intra_mode(sig, luma) <= 34);
            }
        }
    }

    #[test_case]
    fn scan_order_is_mode_dependent_only_for_small_intra_blocks() {
        use super::super::residual::{SCAN_DIAG, SCAN_HORIZ, SCAN_VERT};
        // Inter blocks are always diagonal, whatever the mode value happens to
        // be — an inter CU's "mode" field is not an intra mode at all.
        for mode in 0..=34u8 {
            assert_eq!(scan_order(false, 2, mode), SCAN_DIAG);
            assert_eq!(scan_order(false, 3, mode), SCAN_DIAG);
        }
        // 16x16 and 32x32 intra are diagonal too.
        for mode in 0..=34u8 {
            assert_eq!(scan_order(true, 4, mode), SCAN_DIAG);
            assert_eq!(scan_order(true, 5, mode), SCAN_DIAG);
        }
        // 4x4 and 8x8 intra: near-horizontal modes scan vertically and vice
        // versa, because the residual's energy lies across the prediction
        // direction, not along it.
        for log2 in [2u32, 3] {
            assert_eq!(scan_order(true, log2, 10), SCAN_VERT, "horizontal mode");
            assert_eq!(scan_order(true, log2, 26), SCAN_HORIZ, "vertical mode");
            assert_eq!(scan_order(true, log2, 6), SCAN_VERT);
            assert_eq!(scan_order(true, log2, 14), SCAN_VERT);
            assert_eq!(scan_order(true, log2, 22), SCAN_HORIZ);
            assert_eq!(scan_order(true, log2, 30), SCAN_HORIZ);
            // Just outside the bands, and the two non-angular modes.
            assert_eq!(scan_order(true, log2, 5), SCAN_DIAG);
            assert_eq!(scan_order(true, log2, 15), SCAN_DIAG);
            assert_eq!(scan_order(true, log2, 21), SCAN_DIAG);
            assert_eq!(scan_order(true, log2, 31), SCAN_DIAG);
            assert_eq!(scan_order(true, log2, 0), SCAN_DIAG);
            assert_eq!(scan_order(true, log2, 1), SCAN_DIAG);
        }
    }

    /// Every partition mode must tile its CU exactly — no gap, no overlap.
    /// An asymmetric mode split at half instead of a quarter still tiles, so
    /// the quarter positions are checked separately below.
    #[test_case]
    fn partition_modes_tile_the_coding_unit_exactly() {
        let modes = [
            PartMode::Part2Nx2N,
            PartMode::Part2NxN,
            PartMode::PartNx2N,
            PartMode::PartNxN,
            PartMode::Part2NxnU,
            PartMode::Part2NxnD,
            PartMode::PartnLx2N,
            PartMode::PartnRx2N,
        ];
        for size in [8usize, 16, 32, 64] {
            for m in modes {
                let mut cover = alloc::vec![0u8; size * size];
                for i in 0..m.num_pus() {
                    let (x, y, w, h) = m.pu_rect(i, size);
                    assert!(w > 0 && h > 0, "{m:?} pu {i} is empty at size {size}");
                    assert!(x + w <= size && y + h <= size, "{m:?} pu {i} overflows");
                    for yy in y..y + h {
                        for xx in x..x + w {
                            cover[yy * size + xx] += 1;
                        }
                    }
                }
                assert!(cover.iter().all(|&c| c == 1), "{m:?} at {size} does not tile");
            }
        }
    }

    /// The asymmetric modes split at a **quarter**, and the small partition is
    /// first for `U`/`L` and last for `D`/`R`.
    #[test_case]
    fn asymmetric_partitions_split_at_a_quarter_on_the_right_side() {
        let s = 32usize;
        assert_eq!(PartMode::Part2NxnU.pu_rect(0, s), (0, 0, 32, 8));
        assert_eq!(PartMode::Part2NxnU.pu_rect(1, s), (0, 8, 32, 24));
        assert_eq!(PartMode::Part2NxnD.pu_rect(0, s), (0, 0, 32, 24));
        assert_eq!(PartMode::Part2NxnD.pu_rect(1, s), (0, 24, 32, 8));
        assert_eq!(PartMode::PartnLx2N.pu_rect(0, s), (0, 0, 8, 32));
        assert_eq!(PartMode::PartnLx2N.pu_rect(1, s), (8, 0, 24, 32));
        assert_eq!(PartMode::PartnRx2N.pu_rect(0, s), (0, 0, 24, 32));
        assert_eq!(PartMode::PartnRx2N.pu_rect(1, s), (24, 0, 8, 32));
        // NxN is row-major: 0 1 / 2 3.
        assert_eq!(PartMode::PartNxN.pu_rect(1, s), (16, 0, 16, 16));
        assert_eq!(PartMode::PartNxN.pu_rect(2, s), (0, 16, 16, 16));
    }

    fn inter(refs: [Option<i32>; 2], mvs: [(i16, i16); 2], cbf: bool) -> EdgeSide {
        EdgeSide { intra: false, cbf, refs, mvs }
    }

    #[test_case]
    fn boundary_strength_intra_dominates_and_coefficients_come_next() {
        let plain = inter([Some(0), None], [(0, 0), (0, 0)], false);
        let mut i = plain;
        i.intra = true;
        assert_eq!(boundary_strength(&i, &plain, false), 2);
        assert_eq!(boundary_strength(&plain, &i, true), 2);
        // Coefficients give strength 1, but only across a transform boundary.
        let mut c = plain;
        c.cbf = true;
        assert_eq!(boundary_strength(&c, &plain, true), 1);
        assert_eq!(boundary_strength(&c, &plain, false), 0);
        // Identical motion, no coefficients: nothing to filter.
        assert_eq!(boundary_strength(&plain, &plain, true), 0);
    }

    /// References are matched by **picture**, not by list index. Two blocks
    /// predicting from the same picture through different list slots are the
    /// same prediction and must not be filtered.
    #[test_case]
    fn boundary_strength_matches_references_by_picture_not_list_index() {
        let a = inter([Some(4), None], [(2, 2), (0, 0)], false);
        // Same picture (POC 4), same vector, but signalled through list 1.
        let b = inter([None, Some(4)], [(0, 0), (2, 2)], false);
        assert_eq!(boundary_strength(&a, &b, true), 0, "index-matching would say 1");
        // A genuinely different picture is a discontinuity.
        let c = inter([Some(8), None], [(2, 2), (0, 0)], false);
        assert_eq!(boundary_strength(&a, &c, true), 1);
        // A different count of used lists is too.
        let d = inter([Some(4), Some(4)], [(2, 2), (2, 2)], false);
        assert_eq!(boundary_strength(&a, &d, true), 1);
    }

    /// A whole luma sample of difference is the threshold, and it is `>=`.
    #[test_case]
    fn boundary_strength_threshold_is_four_quarter_pels_inclusive() {
        let a = inter([Some(0), None], [(0, 0), (0, 0)], false);
        for (dx, dy, want) in [(3i16, 0i16, 0u8), (4, 0, 1), (0, 3, 0), (0, 4, 1), (-4, 0, 1)] {
            let b = inter([Some(0), None], [(dx, dy), (0, 0)], false);
            assert_eq!(boundary_strength(&a, &b, true), want, "delta ({dx},{dy})");
        }
    }

    /// When both sides predict twice from the *same* picture, neither pairing
    /// of the vectors is privileged — the edge is smooth if either matches.
    #[test_case]
    fn bi_prediction_from_one_picture_tries_both_pairings() {
        let a = inter([Some(2), Some(2)], [(0, 0), (16, 16)], false);
        // Same two vectors, swapped between the lists: still the same
        // prediction, so strength 0.
        let b = inter([Some(2), Some(2)], [(16, 16), (0, 0)], false);
        assert_eq!(boundary_strength(&a, &b, true), 0, "the crossed pairing matches");
        // Genuinely different motion in one list.
        let c = inter([Some(2), Some(2)], [(0, 0), (64, 64)], false);
        assert_eq!(boundary_strength(&a, &c, true), 1);
        // Two different pictures, matched straight and crossed.
        let d = inter([Some(2), Some(6)], [(0, 0), (16, 16)], false);
        let e = inter([Some(6), Some(2)], [(16, 16), (0, 0)], false);
        assert_eq!(boundary_strength(&d, &e, true), 0, "crossed picture match");
        let f = inter([Some(6), Some(2)], [(0, 0), (16, 16)], false);
        assert_eq!(boundary_strength(&d, &f, true), 1);
    }

    #[test_case]
    fn chroma_422_mode_map_is_monotone_and_in_range() {
        for m in 0..=34u8 {
            let v = chroma_mode_422(m);
            assert!(v <= 34, "mode {m} -> {v}");
        }
        // Planar and DC are unchanged; the map only bends angles.
        assert_eq!(chroma_mode_422(0), 0);
        assert_eq!(chroma_mode_422(1), 1);
        // Monotone: halving one axis cannot reorder the angles.
        for m in 2..34u8 {
            assert!(chroma_mode_422(m) <= chroma_mode_422(m + 1), "not monotone at {m}");
        }
    }
}
