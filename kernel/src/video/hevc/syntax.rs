//! HEVC coding-unit and prediction-unit syntax elements (H.265 §9.3.3).
//!
//! The residual coder lives next door in [`super::residual`]; this is
//! everything above it — the flags and codes that describe *structure* rather
//! than coefficients.
//!
//! **The binarization trees are pure functions over a bin source.** `part_mode`
//! in particular is a 4-deep decision tree whose shape depends on three
//! separate conditions (intra vs inter, at the minimum CU size or not,
//! asymmetric partitions enabled or not), and a wrong branch there produces a
//! *legal* partition mode — just the wrong one, with the bitstream still in
//! sync. Driving the tree through a `Bin` source rather than the CABAC engine
//! directly is what makes it testable: a context-coded bin cannot be forced
//! from outside the coder, so a test that had to arithmetic-code its way to a
//! given branch would be testing the coder.

use super::super::h264::cabac::Cabac;
use super::cabac_tables as ct;
use super::ctu::PartMode;

/// Where a bin comes from: a context index, or the bypass engine.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Bin {
    Ctx(usize),
    Bypass,
}

/// Anything that can supply bins — the real coder, or a canned sequence in a
/// test.
pub trait BinSource {
    fn bin(&mut self, b: Bin) -> u32;
}

impl BinSource for Cabac<'_> {
    #[inline]
    fn bin(&mut self, b: Bin) -> u32 {
        match b {
            Bin::Ctx(i) => self.decision(i),
            Bin::Bypass => self.bypass(),
        }
    }
}

// ---------------------------------------------------------------------------
// Neighbour-derived context increments
// ---------------------------------------------------------------------------

/// `split_cu_flag`'s context (§9.3.4.2.2): how many neighbours are split
/// *deeper* than the current depth.
///
/// A neighbour outside the picture or in another slice contributes nothing —
/// and note it is `>` not `>=`: a neighbour at the same depth is evidence for
/// *not* splitting, so counting it inverts the prior.
pub fn split_cu_ctx(ct_depth: u8, left: Option<u8>, above: Option<u8>) -> usize {
    let mut inc = 0usize;
    if let Some(d) = left {
        inc += (d > ct_depth) as usize;
    }
    if let Some(d) = above {
        inc += (d > ct_depth) as usize;
    }
    inc
}

/// `cu_skip_flag`'s context: how many neighbours were themselves skipped.
pub fn skip_flag_ctx(left: Option<bool>, above: Option<bool>) -> usize {
    left.unwrap_or(false) as usize + above.unwrap_or(false) as usize
}

// ---------------------------------------------------------------------------
// Binarizations
// ---------------------------------------------------------------------------

/// `part_mode` (§9.3.3.7).
///
/// Reading the tree, with the bin string in the comments:
///
/// | bins | mode | when |
/// |---|---|---|
/// | `1` | 2Nx2N | always |
/// | `0` | NxN | intra at the minimum CU size |
/// | `01` | 2NxN | |
/// | `00` | Nx2N | at 8x8, where NxN would be 4x4 inter — forbidden |
/// | `001` / `000` | Nx2N / NxN | at the minimum size, above 8x8 |
/// | `011` | 2NxN | with asymmetric partitions |
/// | `0100` / `0101` | 2NxnU / 2NxnD | |
/// | `001` | Nx2N | |
/// | `0000` / `0001` | nLx2N / nRx2N | |
///
/// The `log2_cb_size == 3` special case is the one most easily missed: a
/// minimum-size 8x8 inter CU cannot use NxN (that would be 4x4 inter
/// prediction, which HEVC forbids for memory-bandwidth reasons), so the code
/// there is two bins rather than three. Decoding it as three eats a bin that
/// belongs to the next element, and everything after it is garbage.
pub fn part_mode<S: BinSource>(
    s: &mut S,
    is_intra: bool,
    log2_cb_size: u32,
    log2_min_cb_size: u32,
    amp_enabled: bool,
) -> PartMode {
    if s.bin(Bin::Ctx(ct::PART_MODE)) != 0 {
        return PartMode::Part2Nx2N;
    }
    if log2_cb_size == log2_min_cb_size {
        if is_intra {
            return PartMode::PartNxN;
        }
        if s.bin(Bin::Ctx(ct::PART_MODE + 1)) != 0 {
            return PartMode::Part2NxN;
        }
        if log2_cb_size == 3 {
            return PartMode::PartNx2N;
        }
        if s.bin(Bin::Ctx(ct::PART_MODE + 2)) != 0 {
            return PartMode::PartNx2N;
        }
        return PartMode::PartNxN;
    }
    if !amp_enabled {
        return if s.bin(Bin::Ctx(ct::PART_MODE + 1)) != 0 {
            PartMode::Part2NxN
        } else {
            PartMode::PartNx2N
        };
    }
    if s.bin(Bin::Ctx(ct::PART_MODE + 1)) != 0 {
        if s.bin(Bin::Ctx(ct::PART_MODE + 3)) != 0 {
            return PartMode::Part2NxN;
        }
        return if s.bin(Bin::Bypass) != 0 { PartMode::Part2NxnD } else { PartMode::Part2NxnU };
    }
    if s.bin(Bin::Ctx(ct::PART_MODE + 3)) != 0 {
        return PartMode::PartNx2N;
    }
    if s.bin(Bin::Bypass) != 0 {
        PartMode::PartnRx2N
    } else {
        PartMode::PartnLx2N
    }
}

/// `mpm_idx`: truncated unary over three values, entirely bypass.
pub fn mpm_idx<S: BinSource>(s: &mut S) -> usize {
    let mut i = 0;
    while i < 2 && s.bin(Bin::Bypass) != 0 {
        i += 1;
    }
    i
}

/// `rem_intra_luma_pred_mode`: five bypass bins, fixed length.
pub fn rem_intra_luma_pred_mode<S: BinSource>(s: &mut S) -> u8 {
    let mut v = 0u8;
    for _ in 0..5 {
        v = (v << 1) | s.bin(Bin::Bypass) as u8;
    }
    v
}

/// `intra_chroma_pred_mode`: one context bin, then two bypass bins. Returns 4
/// for the derived (copy-luma) mode.
pub fn intra_chroma_pred_mode<S: BinSource>(s: &mut S) -> u32 {
    if s.bin(Bin::Ctx(ct::INTRA_CHROMA_PRED_MODE)) == 0 {
        return 4;
    }
    (s.bin(Bin::Bypass) << 1) | s.bin(Bin::Bypass)
}

/// `merge_idx`: one context bin, then bypass, truncated at the list length.
pub fn merge_idx<S: BinSource>(s: &mut S, max_num_merge_cand: usize) -> usize {
    let mut i = s.bin(Bin::Ctx(ct::MERGE_IDX)) as usize;
    if i != 0 {
        while i + 1 < max_num_merge_cand && s.bin(Bin::Bypass) != 0 {
            i += 1;
        }
    }
    i
}

/// `inter_pred_idc` (0 = L0, 1 = L1, 2 = bi).
///
/// An 8x4 or 4x8 prediction unit (`w + h == 12`) **cannot be bi-predicted** —
/// the memory bandwidth of two 4-tap-wide fetches for a tiny block is what the
/// restriction exists to bound — so its bi-prediction bin is not coded at all.
/// Decoding it anyway consumes a bin belonging to the next element.
pub fn inter_pred_idc<S: BinSource>(s: &mut S, w: usize, h: usize, ct_depth: usize) -> u8 {
    if w + h == 12 {
        return s.bin(Bin::Ctx(ct::INTER_PRED_IDC + 4)) as u8;
    }
    if s.bin(Bin::Ctx(ct::INTER_PRED_IDC + ct_depth)) != 0 {
        return 2;
    }
    s.bin(Bin::Ctx(ct::INTER_PRED_IDC + 4)) as u8
}

/// `ref_idx_lX`: truncated unary, context-coded for the first two bins and
/// bypass beyond.
///
/// **Both lists share the `ref_idx_l0` contexts.** The specification's context
/// table has a separate `ref_idx_l1` pair, but every production encoder and
/// decoder (HM, x265, FFmpeg) codes both lists against the L0 pair — x265's
/// `OFF_REF_NO_CTX` is list-agnostic, and FFmpeg's `ff_hevc_ref_idx_lx_decode`
/// hard-codes `REF_IDX_L0_OFFSET`. Using the L1 pair desynchronises the coder
/// the first time a bi-predicted AMVP block has more than one L1 reference
/// (the hierarchical B-pyramid leaf case: `num_ref_idx_l0 == num_ref_idx_l1
/// == 2`), because that is the first time both lists' ref-idx bins fire in the
/// same prediction unit.
pub fn ref_idx<S: BinSource>(s: &mut S, _list: usize, num_ref_idx: usize) -> usize {
    let base = ct::REF_IDX_L0;
    let max = num_ref_idx.saturating_sub(1);
    let max_ctx = max.min(2);
    let mut i = 0usize;
    while i < max_ctx && s.bin(Bin::Ctx(base + i)) != 0 {
        i += 1;
    }
    if i == 2 {
        while i < max && s.bin(Bin::Bypass) != 0 {
            i += 1;
        }
    }
    i
}

/// One component of a motion vector difference, as a magnitude class.
fn mvd_component<S: BinSource>(s: &mut S, class: u32) -> i32 {
    match class {
        0 => 0,
        1 => {
            // Magnitude 1, sign only.
            if s.bin(Bin::Bypass) != 0 {
                -1
            } else {
                1
            }
        }
        _ => {
            // `abs_mvd_minus2` is exp-Golomb order 1, then a sign.
            let mut ret: u32 = 2;
            let mut k = 1u32;
            while k < 31 && s.bin(Bin::Bypass) != 0 {
                ret += 1u32 << k;
                k += 1;
            }
            if k == 31 {
                return 0;
            }
            while k > 0 {
                k -= 1;
                ret += s.bin(Bin::Bypass) << k;
            }
            if s.bin(Bin::Bypass) != 0 {
                -(ret as i32)
            } else {
                ret as i32
            }
        }
    }
}

/// `mvd_coding` (§7.3.8.9): both greater-than-0 flags, then both
/// greater-than-1 flags, then both remainders.
///
/// The **interleaving** is the trap: it is not `x` fully then `y` fully. All
/// four context-coded flags come first, in x, y, x, y order, and only then the
/// bypass remainders. Reading it component-by-component decodes a valid vector
/// from the wrong bins whenever both components are non-zero.
pub fn mvd_coding<S: BinSource>(s: &mut S) -> (i16, i16) {
    let mut cx = s.bin(Bin::Ctx(ct::ABS_MVD_GREATER0_FLAG));
    let mut cy = s.bin(Bin::Ctx(ct::ABS_MVD_GREATER0_FLAG));
    if cx != 0 {
        cx += s.bin(Bin::Ctx(ct::ABS_MVD_GREATER1_FLAG + 1));
    }
    if cy != 0 {
        cy += s.bin(Bin::Ctx(ct::ABS_MVD_GREATER1_FLAG + 1));
    }
    let x = mvd_component(s, cx);
    let y = mvd_component(s, cy);
    (x as i16, y as i16)
}

/// The largest value `cu_qp_delta_abs` can express: the suffix's unary prefix
/// is capped at 7, so `k` runs 0..=6 and the suffix tops out at
/// `(2^6 - 1) + (2^6 - 1) = 126`.
///
/// This is far above any legal QP delta (which the specification bounds at
/// roughly +-38), so a stream reaching it is corrupt rather than unusual.
pub const MAX_CU_QP_DELTA_ABS: u32 = 5 + 126;

/// `cu_qp_delta_abs`: a truncated-unary prefix of at most 5, then an
/// exp-Golomb order-0 suffix.
///
/// Returns `None` when the suffix's prefix reaches its cap of 7, which no
/// well-formed stream does. That case has to be *distinguishable*: the obvious
/// alternative — returning the prefix, 5 — is a plausible small QP delta, so a
/// corrupt stream would quietly shift the quantiser for the rest of the CU
/// instead of being rejected.
pub fn cu_qp_delta_abs<S: BinSource>(s: &mut S) -> Option<u32> {
    let mut prefix = 0u32;
    let mut inc = 0usize;
    while prefix < 5 && s.bin(Bin::Ctx(ct::CU_QP_DELTA + inc)) != 0 {
        prefix += 1;
        inc = 1;
    }
    if prefix < 5 {
        return Some(prefix);
    }
    let mut suffix = 0u32;
    let mut k = 0u32;
    while k < 7 && s.bin(Bin::Bypass) != 0 {
        suffix += 1 << k;
        k += 1;
    }
    if k == 7 {
        return None;
    }
    while k > 0 {
        k -= 1;
        suffix += s.bin(Bin::Bypass) << k;
    }
    Some(prefix + suffix)
}

/// `split_transform_flag`'s context is `5 - log2_trafo_size`, so a larger
/// block uses a lower index — the priors run the other way from intuition.
#[inline]
pub fn split_transform_flag<S: BinSource>(s: &mut S, log2_trafo_size: u32) -> bool {
    s.bin(Bin::Ctx(ct::SPLIT_TRANSFORM_FLAG + (5 - log2_trafo_size) as usize)) != 0
}

#[inline]
pub fn cbf_chroma<S: BinSource>(s: &mut S, trafo_depth: usize) -> bool {
    s.bin(Bin::Ctx(ct::CBF_CB_CR + trafo_depth)) != 0
}

/// `cbf_luma`'s context is `!trafo_depth` — 1 at the top of the tree and 0
/// below it, which is the *opposite* order from every other depth-indexed
/// context here.
#[inline]
pub fn cbf_luma<S: BinSource>(s: &mut S, trafo_depth: usize) -> bool {
    s.bin(Bin::Ctx(ct::CBF_LUMA + (trafo_depth == 0) as usize)) != 0
}

/// Whether the transform tree is present at all for an inter CU.
///
/// Note the sense: the syntax element is `rqt_root_cbf`, and FFmpeg's
/// `no_residual_syntax_flag` name is the negation. A decoder that follows the
/// name rather than the value inverts every inter CU's residual.
#[inline]
pub fn rqt_root_cbf<S: BinSource>(s: &mut S) -> bool {
    s.bin(Bin::Ctx(ct::NO_RESIDUAL_DATA_FLAG)) != 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// A canned bin sequence, so the trees can be exercised without driving the
    /// arithmetic coder.
    struct Canned {
        bits: Vec<u8>,
        pos: usize,
        /// Which sources were asked for, in order — so a test can assert *how*
        /// a bin was coded, not just what it decoded to.
        seen: Vec<Bin>,
    }

    impl Canned {
        fn new(bits: &[u8]) -> Canned {
            Canned { bits: bits.to_vec(), pos: 0, seen: Vec::new() }
        }
        fn used(&self) -> usize {
            self.pos
        }
    }

    impl BinSource for Canned {
        fn bin(&mut self, b: Bin) -> u32 {
            self.seen.push(b);
            let v = *self.bits.get(self.pos).unwrap_or(&0) as u32;
            self.pos += 1;
            v
        }
    }

    fn pm(bits: &[u8], intra: bool, log2: u32, log2_min: u32, amp: bool) -> (PartMode, usize) {
        let mut c = Canned::new(bits);
        let m = part_mode(&mut c, intra, log2, log2_min, amp);
        (m, c.used())
    }

    /// The `part_mode` code must be **prefix-free and complete** in every
    /// configuration: enumerate every bin string up to the tree's depth and
    /// check that each decodes to one mode consuming a well-defined number of
    /// bins, and that every reachable mode is produced exactly once.
    #[test_case]
    fn part_mode_binarization_is_prefix_free_and_complete() {
        // Each configuration: (intra, log2_cb, log2_min, amp) -> expected modes.
        let configs: [(bool, u32, u32, bool, &[PartMode]); 5] = [
            // Intra at the minimum size: only 2Nx2N and NxN.
            (true, 4, 4, false, &[PartMode::Part2Nx2N, PartMode::PartNxN]),
            // Inter at the minimum size, 8x8: NxN is forbidden (4x4 inter).
            (
                false,
                3,
                3,
                false,
                &[PartMode::Part2Nx2N, PartMode::Part2NxN, PartMode::PartNx2N],
            ),
            // Inter at the minimum size, 16x16: NxN is allowed.
            (
                false,
                4,
                4,
                false,
                &[
                    PartMode::Part2Nx2N,
                    PartMode::Part2NxN,
                    PartMode::PartNx2N,
                    PartMode::PartNxN,
                ],
            ),
            // Above the minimum size, no asymmetric partitions.
            (false, 5, 3, false, &[PartMode::Part2Nx2N, PartMode::Part2NxN, PartMode::PartNx2N]),
            // Above the minimum size, asymmetric partitions enabled.
            (
                false,
                5,
                3,
                true,
                &[
                    PartMode::Part2Nx2N,
                    PartMode::Part2NxN,
                    PartMode::PartNx2N,
                    PartMode::Part2NxnU,
                    PartMode::Part2NxnD,
                    PartMode::PartnLx2N,
                    PartMode::PartnRx2N,
                ],
            ),
        ];

        for (intra, log2, log2_min, amp, expect) in configs {
            // Walk every bin string of length 0..=4 and record (prefix, mode).
            let mut codes: Vec<(Vec<u8>, PartMode)> = Vec::new();
            for len in 0..=4usize {
                for v in 0..(1u32 << len) {
                    let bits: Vec<u8> =
                        (0..len).map(|i| ((v >> (len - 1 - i)) & 1) as u8).collect();
                    let (mode, used) = pm(&bits, intra, log2, log2_min, amp);
                    if used == len {
                        codes.push((bits, mode));
                    }
                }
            }
            // Prefix-free: no accepted code is a prefix of another.
            for (a, _) in codes.iter() {
                for (b, _) in codes.iter() {
                    if a.len() < b.len() {
                        assert_ne!(
                            &b[..a.len()],
                            &a[..],
                            "({intra},{log2},{log2_min},{amp}): {a:?} prefixes {b:?}"
                        );
                    }
                }
            }
            // Complete: exactly the expected modes, each once.
            let mut got: Vec<PartMode> = codes.iter().map(|(_, m)| *m).collect();
            got.sort_unstable_by_key(|m| *m as u8);
            let mut want: Vec<PartMode> = expect.to_vec();
            want.sort_unstable_by_key(|m| *m as u8);
            assert_eq!(got, want, "config ({intra},{log2},{log2_min},{amp})");
        }
    }

    /// The 8x8 inter case codes `Nx2N` in **two** bins, not three — an 8x8 CU
    /// cannot use NxN, so the third bin is not sent. Consuming it anyway steals
    /// a bin from the next syntax element.
    #[test_case]
    fn an_eight_by_eight_inter_cu_codes_nx2n_in_two_bins() {
        assert_eq!(pm(&[0, 0], false, 3, 3, false), (PartMode::PartNx2N, 2));
        // At 16x16 the same prefix continues into a third bin.
        assert_eq!(pm(&[0, 0, 1], false, 4, 4, false), (PartMode::PartNx2N, 3));
        assert_eq!(pm(&[0, 0, 0], false, 4, 4, false), (PartMode::PartNxN, 3));
        // Intra at the minimum size stops after one bin.
        assert_eq!(pm(&[0], true, 4, 4, false), (PartMode::PartNxN, 1));
    }

    /// The asymmetric modes' last bin is **bypass**, not context-coded — it is
    /// equiprobable which way an asymmetric split leans. Coding it against a
    /// context would desynchronise the coder's state, not just pick wrong.
    #[test_case]
    fn asymmetric_partition_direction_is_a_bypass_bin() {
        let mut c = Canned::new(&[0, 1, 0, 1]);
        assert_eq!(part_mode(&mut c, false, 5, 3, true), PartMode::Part2NxnD);
        assert_eq!(
            c.seen,
            alloc::vec![
                Bin::Ctx(ct::PART_MODE),
                Bin::Ctx(ct::PART_MODE + 1),
                Bin::Ctx(ct::PART_MODE + 3),
                Bin::Bypass,
            ]
        );
        let mut c = Canned::new(&[0, 0, 0, 1]);
        assert_eq!(part_mode(&mut c, false, 5, 3, true), PartMode::PartnRx2N);
        assert_eq!(c.seen[3], Bin::Bypass);
    }

    /// `cu_qp_delta_abs` must be a bijection onto 0.. — the prefix saturates at
    /// 5, with **no** terminating bin at 5 (the easy off-by-one), and an
    /// exp-Golomb order-0 suffix carries the rest.
    ///
    /// The encoder is written from the specification's description, not by
    /// inverting the decoder, so agreement is real evidence.
    #[test_case]
    fn cu_qp_delta_abs_round_trips() {
        fn encode(v: u32) -> Vec<u8> {
            if v < 5 {
                let mut b = alloc::vec![1u8; v as usize];
                b.push(0);
                return b;
            }
            let rest = v - 5;
            let mut k = 0u32;
            while (1u32 << (k + 1)) - 1 <= rest {
                k += 1;
            }
            let suffix = rest - ((1u32 << k) - 1);
            let mut b = alloc::vec![1u8; 5 + k as usize];
            b.push(0);
            for j in (0..k).rev() {
                b.push(((suffix >> j) & 1) as u8);
            }
            b
        }
        for v in 0..=MAX_CU_QP_DELTA_ABS {
            let bits = encode(v);
            let mut c = Canned::new(&bits);
            assert_eq!(cu_qp_delta_abs(&mut c), Some(v), "value {v} bits {bits:?}");
            assert_eq!(c.used(), bits.len(), "value {v} consumed the wrong length");
        }
        // A prefix below 5 carries its terminator; a prefix of 5 does not.
        assert_eq!(encode(3), alloc::vec![1u8, 1, 1, 0]);
        assert_eq!(encode(5), alloc::vec![1u8, 1, 1, 1, 1, 0]);

        // 131 is the ceiling, and it is not arbitrary: the suffix's own unary
        // prefix is capped at 7, so `k` runs 0..=6. One past it, the code the
        // encoder would have to emit is refused rather than being read as the
        // plausible small delta 5.
        let over = encode(MAX_CU_QP_DELTA_ABS + 1);
        assert_eq!(over.iter().take_while(|&&b| b == 1).count(), 12, "5 + k=7 ones");
        let mut c = Canned::new(&over);
        assert_eq!(cu_qp_delta_abs(&mut c), None, "a k of 7 must be refused");
    }

    /// The MVD's four context-coded flags are **interleaved x, y, x, y** and
    /// come before either remainder. Decoding component-by-component reads a
    /// valid vector from the wrong bins whenever both components are non-zero.
    #[test_case]
    fn mvd_flags_are_interleaved_before_the_remainders() {
        // gt0(x)=1 gt0(y)=1 gt1(x)=0 gt1(y)=0 sign(x)=0 sign(y)=1
        let mut c = Canned::new(&[1, 1, 0, 0, 0, 1]);
        assert_eq!(mvd_coding(&mut c), (1, -1));
        assert_eq!(
            &c.seen[..4],
            &[
                Bin::Ctx(ct::ABS_MVD_GREATER0_FLAG),
                Bin::Ctx(ct::ABS_MVD_GREATER0_FLAG),
                Bin::Ctx(ct::ABS_MVD_GREATER1_FLAG + 1),
                Bin::Ctx(ct::ABS_MVD_GREATER1_FLAG + 1),
            ],
            "the four flags must come first, interleaved"
        );
        assert!(c.seen[4..].iter().all(|&b| b == Bin::Bypass), "remainders are bypass");

        // Only x non-zero: y contributes no greater1 flag at all.
        let mut c = Canned::new(&[1, 0, 0, 1]);
        assert_eq!(mvd_coding(&mut c), (-1, 0));
        assert_eq!(c.used(), 4, "a zero component costs exactly one bin");
    }

    /// Magnitudes of 2 and above are exp-Golomb **order 1** followed by a sign,
    /// so the bucket for `m` leading ones is `[2^(m+1), 2^(m+2) - 1]` and the
    /// suffix is `m + 1` bits — not `m`, which is the order-0 shape and halves
    /// every large vector.
    #[test_case]
    fn mvd_magnitudes_above_one_round_trip() {
        fn encode(mag: u32, negative: bool) -> Vec<u8> {
            assert!(mag >= 2);
            let mut m = 0u32;
            while !((1u32 << (m + 1)) <= mag && mag < (1u32 << (m + 2))) {
                m += 1;
            }
            let suffix = mag - (1u32 << (m + 1));
            let mut b = alloc::vec![1u8; m as usize];
            b.push(0);
            for j in (0..=m).rev() {
                b.push(((suffix >> j) & 1) as u8);
            }
            b.push(negative as u8);
            b
        }
        for mag in [2u32, 3, 4, 5, 6, 7, 8, 15, 16, 33, 100, 1000, 8000] {
            for neg in [false, true] {
                // gt0(x)=1, gt0(y)=0, gt1(x)=1 -> class 2 for x, 0 for y.
                let mut bits = alloc::vec![1u8, 0, 1];
                bits.extend_from_slice(&encode(mag, neg));
                let mut c = Canned::new(&bits);
                let (x, y) = mvd_coding(&mut c);
                let want = if neg { -(mag as i32) } else { mag as i32 };
                assert_eq!((x as i32, y), (want, 0), "mag {mag} neg {neg}");
                assert_eq!(c.used(), bits.len(), "mag {mag} consumed the wrong length");
            }
        }
    }

    /// `ref_idx` is context-coded for two bins and bypass beyond, and truncated
    /// at the list length — a two-entry list codes index 1 in one bin with no
    /// terminator.
    #[test_case]
    fn ref_idx_is_truncated_unary_with_two_contexts() {
        // Two references: one bin, no terminator.
        let mut c = Canned::new(&[1]);
        assert_eq!(ref_idx(&mut c, 0, 2), 1);
        assert_eq!(c.used(), 1);
        // One reference: nothing is coded at all.
        let mut c = Canned::new(&[1, 1, 1]);
        assert_eq!(ref_idx(&mut c, 0, 1), 0);
        assert_eq!(c.used(), 0, "a single-entry list codes no bins");
        // Five references: two context bins then bypass.
        let mut c = Canned::new(&[1, 1, 1, 0]);
        assert_eq!(ref_idx(&mut c, 1, 5), 3);
        assert_eq!(c.seen[0], Bin::Ctx(ct::REF_IDX_L1));
        assert_eq!(c.seen[1], Bin::Ctx(ct::REF_IDX_L1 + 1));
        assert_eq!(c.seen[2], Bin::Bypass, "the third bin onward is bypass");
        // And it saturates at the list length rather than running on.
        let mut c = Canned::new(&[1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(ref_idx(&mut c, 0, 4), 3);
        assert_eq!(c.used(), 3);
    }

    /// `merge_idx` is truncated at `max_num_merge_cand`, and index 0 costs one
    /// bin regardless.
    #[test_case]
    fn merge_idx_is_truncated_at_the_candidate_count() {
        let mut c = Canned::new(&[0]);
        assert_eq!(merge_idx(&mut c, 5), 0);
        assert_eq!(c.used(), 1);
        let mut c = Canned::new(&[1, 1, 1, 1, 1]);
        assert_eq!(merge_idx(&mut c, 5), 4, "saturates at max - 1");
        assert_eq!(c.used(), 4, "no terminator at the top of the range");
        // A two-candidate list codes only the first bin.
        let mut c = Canned::new(&[1, 1, 1]);
        assert_eq!(merge_idx(&mut c, 2), 1);
        assert_eq!(c.used(), 1);
    }

    /// An 8x4 or 4x8 prediction unit cannot be bi-predicted, so its
    /// bi-prediction bin is not coded.
    #[test_case]
    fn tiny_prediction_units_skip_the_bi_prediction_bin() {
        for (w, h) in [(8usize, 4usize), (4, 8)] {
            let mut c = Canned::new(&[1]);
            assert_eq!(inter_pred_idc(&mut c, w, h, 0), 1, "L1 for {w}x{h}");
            assert_eq!(c.used(), 1, "{w}x{h} must code exactly one bin");
            assert_eq!(c.seen[0], Bin::Ctx(ct::INTER_PRED_IDC + 4));
        }
        // A larger PU codes the bi bin first, against a depth-indexed context.
        let mut c = Canned::new(&[1]);
        assert_eq!(inter_pred_idc(&mut c, 16, 16, 2), 2);
        assert_eq!(c.seen[0], Bin::Ctx(ct::INTER_PRED_IDC + 2));
        let mut c = Canned::new(&[0, 0]);
        assert_eq!(inter_pred_idc(&mut c, 16, 16, 1), 0);
        assert_eq!(c.used(), 2);
    }

    /// `cbf_luma`'s context runs the *opposite* way from every other
    /// depth-indexed context here — 1 at the top of the transform tree, 0 below.
    #[test_case]
    fn depth_indexed_contexts_run_in_the_directions_they_do() {
        let mut c = Canned::new(&[1]);
        cbf_luma(&mut c, 0);
        assert_eq!(c.seen[0], Bin::Ctx(ct::CBF_LUMA + 1), "depth 0 uses index 1");
        let mut c = Canned::new(&[1]);
        cbf_luma(&mut c, 2);
        assert_eq!(c.seen[0], Bin::Ctx(ct::CBF_LUMA), "any deeper uses index 0");
        // Chroma is the ordinary direction.
        let mut c = Canned::new(&[1]);
        cbf_chroma(&mut c, 2);
        assert_eq!(c.seen[0], Bin::Ctx(ct::CBF_CB_CR + 2));
        // And the split flag's index decreases with block size.
        let mut c = Canned::new(&[1]);
        split_transform_flag(&mut c, 5);
        assert_eq!(c.seen[0], Bin::Ctx(ct::SPLIT_TRANSFORM_FLAG));
        let mut c = Canned::new(&[1]);
        split_transform_flag(&mut c, 2);
        assert_eq!(c.seen[0], Bin::Ctx(ct::SPLIT_TRANSFORM_FLAG + 3));
    }

    #[test_case]
    fn neighbour_context_increments_count_the_right_thing() {
        // `split_cu_flag` counts neighbours split *deeper*, strictly.
        assert_eq!(split_cu_ctx(1, Some(2), Some(2)), 2);
        assert_eq!(split_cu_ctx(1, Some(1), Some(1)), 0, "equal depth is not evidence");
        assert_eq!(split_cu_ctx(1, Some(0), Some(2)), 1);
        // An unavailable neighbour contributes nothing.
        assert_eq!(split_cu_ctx(0, None, None), 0);
        assert_eq!(split_cu_ctx(0, None, Some(3)), 1);
        // Skip counts skipped neighbours.
        assert_eq!(skip_flag_ctx(Some(true), Some(true)), 2);
        assert_eq!(skip_flag_ctx(Some(true), None), 1);
        assert_eq!(skip_flag_ctx(None, None), 0);
    }

    /// The chroma mode's escape costs one context bin; the four explicit modes
    /// cost that bin plus two bypass bins.
    #[test_case]
    fn intra_chroma_mode_codes_the_derived_case_in_one_bin() {
        let mut c = Canned::new(&[0]);
        assert_eq!(intra_chroma_pred_mode(&mut c), 4);
        assert_eq!(c.used(), 1);
        for (bits, want) in [([1u8, 0, 0], 0u32), ([1, 0, 1], 1), ([1, 1, 0], 2), ([1, 1, 1], 3)] {
            let mut c = Canned::new(&bits);
            assert_eq!(intra_chroma_pred_mode(&mut c), want, "bits {bits:?}");
            assert_eq!(c.used(), 3);
            assert_eq!(c.seen[1], Bin::Bypass);
        }
    }

    #[test_case]
    fn mpm_and_remainder_are_pure_bypass() {
        let mut c = Canned::new(&[0]);
        assert_eq!(mpm_idx(&mut c), 0);
        assert_eq!(c.used(), 1);
        let mut c = Canned::new(&[1, 1]);
        assert_eq!(mpm_idx(&mut c), 2, "saturates at 2 with no terminator");
        assert_eq!(c.used(), 2);
        let mut c = Canned::new(&[1, 0, 1, 1, 0]);
        assert_eq!(rem_intra_luma_pred_mode(&mut c), 0b10110);
        assert_eq!(c.used(), 5, "always exactly five bins");
        assert!(c.seen.iter().all(|&b| b == Bin::Bypass));
    }
}
