//! HEVC picture order counts, reference picture sets, reference lists and the
//! decoded picture buffer (H.265 §8.3.1-§8.3.4, C.5.2).
//!
//! This is the layer that decides *which pictures a slice may reference* and
//! *in what order frames leave the decoder*. Nothing in it touches a pixel, and
//! every mistake in it is therefore a whole frame that is right in isolation
//! and wrong in sequence — a block predicted from the correct-looking picture at
//! the wrong index, or a GOP output in the wrong order. Neither fails.
//!
//! Four rules carry most of the risk:
//!
//! - **The POC wrap test is asymmetric.** `<` pairs with `>=` and `>` pairs
//!   with `>` — a half-range difference wraps in one direction and not the
//!   other. Writing both comparisons the same way misplaces exactly one
//!   picture per wrap, which on a long sequence is one visible glitch every
//!   few thousand frames.
//! - **List 0 and list 1 concatenate the same sets in a different order.**
//!   L0 is before-then-after, L1 after-then-before. That single swap is the
//!   whole of bidirectional prediction; getting it wrong still decodes, with
//!   every B-frame's two predictions exchanged.
//! - **A reference list is filled cyclically.** If a slice asks for more
//!   references than the RPS holds, the candidate sets are concatenated again
//!   from the start. Truncating instead leaves a short list, and every index
//!   past its end names nothing.
//! - **`used_by_curr_pic` decides Curr vs Foll, position decides Before vs
//!   After.** A picture kept only to be referenced *later* is in the RPS but
//!   not in any list, and putting it in one shifts every index above it.

use alloc::vec::Vec;

use super::{NalType, ShortTermRps};

/// The long-term part of a reference picture set, as signalled in the slice
/// header.
#[derive(Clone, Default, Debug)]
pub struct LongTermRps {
    /// Full POCs (already resolved through `delta_poc_msb_present`).
    pub poc: Vec<i32>,
    pub used: Vec<bool>,
}

/// Compute this picture's POC (§8.3.1).
///
/// `poc_tid0` is the POC of the most recent picture with `TemporalId == 0`
/// that was not a RASL, RADL or sub-layer non-reference picture — the anchor
/// the wrap is measured against. Using the *previous picture* instead works
/// until a temporal sub-layer is dropped, and then drifts.
pub fn compute_poc(log2_max_poc_lsb: u32, poc_tid0: i32, poc_lsb: i32, nal: NalType) -> i32 {
    let max_poc_lsb = 1i32 << log2_max_poc_lsb;
    let prev_poc_lsb = poc_tid0.rem_euclid(max_poc_lsb);
    let prev_poc_msb = poc_tid0 - prev_poc_lsb;

    // Note the asymmetry: `>=` on the way up, `>` on the way down. A difference
    // of exactly half the range wraps forwards and does not wrap backwards.
    let poc_msb = if poc_lsb < prev_poc_lsb && prev_poc_lsb - poc_lsb >= max_poc_lsb / 2 {
        prev_poc_msb + max_poc_lsb
    } else if poc_lsb > prev_poc_lsb && poc_lsb - prev_poc_lsb > max_poc_lsb / 2 {
        prev_poc_msb - max_poc_lsb
    } else {
        prev_poc_msb
    };

    // A broken-link access picture restarts the count: there is no previous
    // picture to be continuous with, by definition.
    if nal.is_bla() {
        return poc_lsb;
    }
    poc_msb + poc_lsb
}

/// The five reference picture subsets (§8.3.2), as POCs.
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct RpsSets {
    /// Short-term, earlier than the current picture, referenced by it.
    pub st_curr_before: Vec<i32>,
    /// Short-term, later than the current picture, referenced by it.
    pub st_curr_after: Vec<i32>,
    /// Short-term, kept for a *later* picture but not referenced by this one.
    pub st_foll: Vec<i32>,
    pub lt_curr: Vec<i32>,
    pub lt_foll: Vec<i32>,
}

impl RpsSets {
    /// How many references this picture may use — the sum of the `Curr` sets.
    pub fn nb_curr(&self) -> usize {
        self.st_curr_before.len() + self.st_curr_after.len() + self.lt_curr.len()
    }
}

/// Derive the reference picture sets for a picture at `poc` (§8.3.2).
pub fn derive_rps(poc: i32, st: Option<&ShortTermRps>, lt: &LongTermRps) -> RpsSets {
    let mut sets = RpsSets::default();
    if let Some(st) = st {
        // The negative deltas come first and are the "before" set; the split is
        // by *position in the RPS*, not by the sign of the resulting POC —
        // which are the same thing for a well-formed stream and are not the
        // rule.
        for (i, &d) in st.delta_poc_s0.iter().enumerate() {
            let p = poc + d;
            if st.used_s0.get(i).copied().unwrap_or(false) {
                sets.st_curr_before.push(p);
            } else {
                sets.st_foll.push(p);
            }
        }
        for (i, &d) in st.delta_poc_s1.iter().enumerate() {
            let p = poc + d;
            if st.used_s1.get(i).copied().unwrap_or(false) {
                sets.st_curr_after.push(p);
            } else {
                sets.st_foll.push(p);
            }
        }
    }
    for (i, &p) in lt.poc.iter().enumerate() {
        if lt.used.get(i).copied().unwrap_or(false) {
            sets.lt_curr.push(p);
        } else {
            sets.lt_foll.push(p);
        }
    }
    sets
}

/// One entry of a reference picture list.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RefEntry {
    pub poc: i32,
    /// Long-term references are excluded from temporal motion vector scaling —
    /// there is no meaningful distance to scale by, so a long-term candidate is
    /// used unscaled or not at all.
    pub long_term: bool,
}

/// Build reference picture list `list_idx` (§8.3.4).
///
/// `nb_refs` is `num_ref_idx_lX_active_minus1 + 1` from the slice header, and
/// `modification` is the optional explicit reordering — a list of indices into
/// the *unmodified* concatenation, not into the final list.
pub fn build_ref_list(
    sets: &RpsSets,
    list_idx: usize,
    nb_refs: usize,
    modification: Option<&[u32]>,
) -> Vec<RefEntry> {
    // L0: before, after, long-term. L1: after, before, long-term. The swap of
    // the first two is the entirety of bidirectional prediction's structure.
    let (first, second) = if list_idx == 0 {
        (&sets.st_curr_before, &sets.st_curr_after)
    } else {
        (&sets.st_curr_after, &sets.st_curr_before)
    };

    let mut tmp: Vec<RefEntry> = Vec::with_capacity(nb_refs.max(1));
    if sets.nb_curr() == 0 {
        return tmp;
    }
    // Fill cyclically: a slice may activate more references than the RPS holds,
    // and the specification repeats the concatenation rather than stopping.
    while tmp.len() < nb_refs {
        for &p in first.iter() {
            tmp.push(RefEntry { poc: p, long_term: false });
        }
        for &p in second.iter() {
            tmp.push(RefEntry { poc: p, long_term: false });
        }
        for &p in sets.lt_curr.iter() {
            tmp.push(RefEntry { poc: p, long_term: true });
        }
    }

    match modification {
        None => {
            tmp.truncate(nb_refs);
            tmp
        }
        Some(idx) => idx
            .iter()
            .take(nb_refs)
            .filter_map(|&i| tmp.get(i as usize).copied())
            .collect(),
    }
}

/// How a decoded picture buffer slot is being used.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct FrameFlags {
    pub short_ref: bool,
    pub long_ref: bool,
    /// Still owed to the caller for display.
    pub output: bool,
}

impl FrameFlags {
    #[inline]
    pub fn in_use(&self) -> bool {
        self.short_ref || self.long_ref || self.output
    }
    #[inline]
    pub fn is_ref(&self) -> bool {
        self.short_ref || self.long_ref
    }
}

/// A decoded picture buffer slot.
#[derive(Clone, Copy, Default, Debug)]
pub struct Slot {
    pub poc: i32,
    pub flags: FrameFlags,
}

/// Apply a picture's RPS to the buffer (§8.3.2): every slot not named by the
/// set stops being a reference.
///
/// It keeps its `output` flag — a picture can be finished as a reference while
/// still waiting to be displayed, and clearing both here loses frames.
pub fn apply_rps(dpb: &mut [Slot], sets: &RpsSets, current_poc: i32) {
    for s in dpb.iter_mut() {
        if !s.flags.in_use() || s.poc == current_poc {
            continue;
        }
        let short = sets.st_curr_before.contains(&s.poc)
            || sets.st_curr_after.contains(&s.poc)
            || sets.st_foll.contains(&s.poc);
        let long = sets.lt_curr.contains(&s.poc) || sets.lt_foll.contains(&s.poc);
        s.flags.short_ref = short;
        s.flags.long_ref = long;
    }
}

/// Decide the next frame to output, if any (C.5.2.2 "bumping").
///
/// A frame leaves when there are more pending outputs than the reorder depth
/// allows, or when the buffer is fuller than its capacity — and the one that
/// leaves is always the **lowest POC still pending**, which is what turns
/// decode order back into display order.
///
/// Returns the DPB index to output.
pub fn next_output(dpb: &[Slot], max_num_reorder: usize, max_dec_pic_buffering: usize) -> Option<usize> {
    let mut nb_output = 0usize;
    let mut nb_used = 0usize;
    let mut min: Option<(i32, usize)> = None;
    for (i, s) in dpb.iter().enumerate() {
        if s.flags.in_use() {
            nb_used += 1;
        }
        if s.flags.output {
            nb_output += 1;
            if min.map_or(true, |(p, _)| s.poc < p) {
                min = Some((s.poc, i));
            }
        }
    }
    if nb_output > max_num_reorder || (nb_output > 0 && nb_used > max_dec_pic_buffering) {
        return min.map(|(_, i)| i);
    }
    None
}

/// Drain everything still pending, lowest POC first — what happens at the end
/// of a sequence or on a flush.
pub fn drain_order(dpb: &[Slot]) -> Vec<usize> {
    let mut pending: Vec<(i32, usize)> = dpb
        .iter()
        .enumerate()
        .filter(|(_, s)| s.flags.output)
        .map(|(i, s)| (s.poc, i))
        .collect();
    pending.sort_unstable();
    pending.into_iter().map(|(_, i)| i).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn st(neg: &[(i32, bool)], pos: &[(i32, bool)]) -> ShortTermRps {
        ShortTermRps {
            delta_poc_s0: neg.iter().map(|&(d, _)| d).collect(),
            used_s0: neg.iter().map(|&(_, u)| u).collect(),
            delta_poc_s1: pos.iter().map(|&(d, _)| d).collect(),
            used_s1: pos.iter().map(|&(_, u)| u).collect(),
        }
    }

    /// The POC wrap, including its asymmetry. With a 4-bit LSB the range is 16
    /// and half of it is 8: a *backwards* difference of exactly 8 wraps up, a
    /// *forwards* difference of exactly 8 does not wrap down.
    #[test_case]
    fn poc_wraps_asymmetrically_at_half_the_range() {
        let n = NalType::from_code(1); // TRAIL_R, an ordinary picture
        // No wrap: the LSB just advances.
        assert_eq!(compute_poc(4, 10, 11, n), 11);
        assert_eq!(compute_poc(4, 10, 3, n), 3);
        // Forward wrap: lsb went backwards by at least half the range.
        assert_eq!(compute_poc(4, 14, 1, n), 17, "14 -> 1 must wrap forward");
        assert_eq!(compute_poc(4, 15, 7, n), 23, "difference of exactly 8 wraps");
        assert_eq!(compute_poc(4, 15, 8, n), 8, "difference of 7 does not");
        // Backward wrap: lsb went forwards by *more* than half the range.
        assert_eq!(compute_poc(4, 17, 15, n), 15, "1 -> 15 must wrap back to 15");
        assert_eq!(compute_poc(4, 16, 9, n), 9, "difference of exactly 9 wraps back");
        assert_eq!(compute_poc(4, 16, 8, n), 24, "difference of exactly 8 does NOT");
        // That last pair is the asymmetry. A symmetric implementation gets one
        // of them wrong, and only one picture per wrap is affected.
    }

    /// A broken-link access picture restarts the count from its LSB — it is the
    /// point a stream was spliced, so there is nothing to be continuous with.
    #[test_case]
    fn a_bla_picture_resets_the_poc_msb() {
        for code in [16u8, 17, 18] {
            let nal = NalType::from_code(code);
            assert!(nal.is_bla(), "code {code} should be a BLA type");
            assert_eq!(compute_poc(4, 100, 5, nal), 5, "BLA must drop the MSB");
        }
        // An IDR or CRA does not take this path here — their POC is handled by
        // the caller (an IDR has no `pic_order_cnt_lsb` at all).
        assert_eq!(compute_poc(4, 100, 5, NalType::from_code(1)), 101);
    }

    /// `used_by_curr_pic` splits Curr from Foll, and *position* splits Before
    /// from After.
    #[test_case]
    fn rps_derivation_splits_by_use_and_by_position() {
        let s = st(&[(-1, true), (-3, false), (-5, true)], &[(2, true), (4, false)]);
        let lt = LongTermRps { poc: vec![100, 200], used: vec![true, false] };
        let sets = derive_rps(20, Some(&s), &lt);
        assert_eq!(sets.st_curr_before, vec![19, 15]);
        assert_eq!(sets.st_curr_after, vec![22]);
        // Unused entries from *both* directions land in the same follow set.
        assert_eq!(sets.st_foll, vec![17, 24]);
        assert_eq!(sets.lt_curr, vec![100]);
        assert_eq!(sets.lt_foll, vec![200]);
        assert_eq!(sets.nb_curr(), 4);
    }

    /// List 0 is before-then-after; list 1 is after-then-before. Long-term
    /// references come last in both.
    #[test_case]
    fn the_two_reference_lists_order_the_sets_oppositely() {
        let sets = RpsSets {
            st_curr_before: vec![8, 4],
            st_curr_after: vec![16, 20],
            st_foll: vec![],
            lt_curr: vec![0],
            lt_foll: vec![],
        };
        let l0 = build_ref_list(&sets, 0, 5, None);
        assert_eq!(l0.iter().map(|e| e.poc).collect::<Vec<_>>(), vec![8, 4, 16, 20, 0]);
        let l1 = build_ref_list(&sets, 1, 5, None);
        assert_eq!(l1.iter().map(|e| e.poc).collect::<Vec<_>>(), vec![16, 20, 8, 4, 0]);
        // Only the long-term entry is marked as such, in both lists.
        assert_eq!(l0.iter().filter(|e| e.long_term).count(), 1);
        assert_eq!(l0[4].long_term, true);
        assert_eq!(l1[4].long_term, true);
        // The *follow* set never enters a list — it is kept for a later
        // picture, and including it would shift every index above it.
        let mut with_foll = sets.clone();
        with_foll.st_foll = vec![999];
        let l = build_ref_list(&with_foll, 0, 5, None);
        assert!(!l.iter().any(|e| e.poc == 999));
    }

    /// A slice may activate more references than the set holds; the
    /// concatenation then repeats. A short list would leave later indices
    /// naming nothing.
    #[test_case]
    fn a_reference_list_longer_than_the_set_repeats_cyclically() {
        let sets = RpsSets {
            st_curr_before: vec![8],
            st_curr_after: vec![16],
            st_foll: vec![],
            lt_curr: vec![],
            lt_foll: vec![],
        };
        let l0 = build_ref_list(&sets, 0, 5, None);
        assert_eq!(l0.len(), 5);
        assert_eq!(l0.iter().map(|e| e.poc).collect::<Vec<_>>(), vec![8, 16, 8, 16, 8]);
        // A shorter request truncates.
        let l0 = build_ref_list(&sets, 1, 1, None);
        assert_eq!(l0.iter().map(|e| e.poc).collect::<Vec<_>>(), vec![16]);
        // With nothing referenced at all, the list is empty rather than looping
        // forever — which is the shape a corrupt stream produces.
        let empty = RpsSets::default();
        assert!(build_ref_list(&empty, 0, 4, None).is_empty());
    }

    /// Explicit modification indexes the **unmodified** concatenation, so an
    /// entry may be repeated or dropped.
    #[test_case]
    fn list_modification_indexes_the_unmodified_concatenation() {
        let sets = RpsSets {
            st_curr_before: vec![8, 4],
            st_curr_after: vec![16],
            st_foll: vec![],
            lt_curr: vec![],
            lt_foll: vec![],
        };
        let l = build_ref_list(&sets, 0, 3, Some(&[2, 0, 2]));
        assert_eq!(l.iter().map(|e| e.poc).collect::<Vec<_>>(), vec![16, 8, 16]);
        // An out-of-range index is dropped rather than panicking — this is
        // attacker-reachable data.
        let l = build_ref_list(&sets, 0, 3, Some(&[0, 99, 1]));
        assert_eq!(l.iter().map(|e| e.poc).collect::<Vec<_>>(), vec![8, 4]);
    }

    /// Applying an RPS clears reference flags on everything it does not name —
    /// but leaves the output flag alone, because a picture can be finished as a
    /// reference while still waiting to be displayed.
    #[test_case]
    fn applying_an_rps_unmarks_references_but_keeps_pending_output() {
        let f = |poc, sr, out| Slot {
            poc,
            flags: FrameFlags { short_ref: sr, long_ref: false, output: out },
        };
        let mut dpb = vec![f(0, true, false), f(4, true, true), f(8, true, true), f(12, true, true)];
        let sets = RpsSets {
            st_curr_before: vec![8],
            st_foll: vec![4],
            ..Default::default()
        };
        apply_rps(&mut dpb, &sets, 12);
        assert!(!dpb[0].flags.short_ref, "POC 0 is not in the set");
        assert!(!dpb[0].flags.in_use(), "and had nothing pending, so the slot is free");
        assert!(dpb[1].flags.short_ref, "POC 4 is in the follow set");
        assert!(dpb[2].flags.short_ref, "POC 8 is referenced");
        assert!(dpb[3].flags.short_ref, "the current picture is never unmarked");
        // A picture dropped from the set keeps its pending output.
        let mut dpb = vec![f(0, true, true)];
        apply_rps(&mut dpb, &RpsSets::default(), 4);
        assert!(!dpb[0].flags.short_ref);
        assert!(dpb[0].flags.output, "output must survive the unmarking");
    }

    /// Bumping outputs the **lowest pending POC**, and only once the reorder
    /// depth or the buffer capacity is exceeded. That is what turns decode
    /// order back into display order for a B-pyramid.
    #[test_case]
    fn bumping_releases_the_lowest_pending_poc_at_the_reorder_depth() {
        let p = |poc| Slot {
            poc,
            flags: FrameFlags { short_ref: true, long_ref: false, output: true },
        };
        // A B-pyramid arrives in decode order 0, 8, 4, 2, 6.
        let dpb = vec![p(0), p(8), p(4)];
        // Reorder depth 3: nothing is due yet.
        assert_eq!(next_output(&dpb, 3, 8), None);
        // Depth 2: the lowest POC leaves, which is 0 — not the oldest slot, and
        // not the most recently decoded.
        assert_eq!(next_output(&dpb, 2, 8), Some(0));
        let dpb = vec![p(8), p(4), p(2)];
        assert_eq!(next_output(&dpb, 2, 8), Some(2), "POC 2 is lowest, in slot 2");
        // Capacity pressure bumps even inside the reorder depth.
        assert_eq!(next_output(&dpb, 8, 2), Some(2));
        // With nothing pending, capacity pressure alone does nothing — there is
        // no frame to release.
        let held = vec![Slot {
            poc: 0,
            flags: FrameFlags { short_ref: true, long_ref: false, output: false },
        }];
        assert_eq!(next_output(&held, 0, 0), None);
    }

    /// A flush drains in display order, whatever order the slots are in.
    #[test_case]
    fn draining_outputs_in_poc_order_not_slot_order() {
        let p = |poc, out| Slot {
            poc,
            flags: FrameFlags { short_ref: false, long_ref: false, output: out },
        };
        let dpb = vec![p(8, true), p(2, true), p(6, false), p(4, true)];
        let order = drain_order(&dpb);
        assert_eq!(order, vec![1, 3, 0], "POCs 2, 4, 8; POC 6 is not pending");
    }
}
