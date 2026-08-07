//! VP9 coefficient (token) decoding — libvpx `decode_coefs`.
//!
//! This is where nearly all of a frame's bits are, and it is the tightest loop
//! in the decoder. Three things about it are easy to get subtly wrong, and none
//! of them fail loudly:
//!
//! * **The context for coefficient `c` comes from the two *scan neighbours* of
//!   `c`, not from `c - 1`.** The neighbour table is per scan order, and it is
//!   indexed by position-in-scan while the cached token values are indexed by
//!   *raster position* (`token_cache[scan[c]]`). Mixing those two spaces gives
//!   a decoder that works on flat blocks and drifts on detailed ones.
//! * **The band advances once per coefficient**, including through the run of
//!   zero tokens — libvpx walks `band_translate` with a post-increment inside
//!   both loops. Advancing it only on non-zero coefficients silently changes
//!   the probabilities used for the rest of the block.
//! * **Dequantisation happens here, during parsing** (`v = val * dqv >> shift`),
//!   with `dqv` switching from the DC quantiser to the AC quantiser after the
//!   first coefficient — and `dq_shift` is 1 only for 32x32. Doing it later, or
//!   with a single quantiser, changes every block's contrast slightly.

use super::header::FrameContext;
use super::tables;
use super::transform::{TxSize, TxType};
use super::BoolDecoder;

/// Token-tree node indices within a coefficient probability triple.
const EOB_CONTEXT_NODE: usize = 0;
const ZERO_CONTEXT_NODE: usize = 1;
const ONE_CONTEXT_NODE: usize = 2;
/// Which node's probability selects the Pareto row for the tail.
const PIVOT_NODE: usize = 2;

const CAT1_MIN_VAL: i32 = 5;
const CAT2_MIN_VAL: i32 = 7;
const CAT3_MIN_VAL: i32 = 11;
const CAT4_MIN_VAL: i32 = 19;
const CAT5_MIN_VAL: i32 = 35;
const CAT6_MIN_VAL: i32 = 67;

/// A scan order: the raster position of each coefficient in scan order, and the
/// two neighbours of each scan position used to derive its context.
pub struct ScanOrder {
    pub scan: &'static [i16],
    pub neighbors: &'static [i16],
}

/// The scan order for a `(tx_size, tx_type)` pair (libvpx `vp9_scan_orders`).
///
/// Note the mapping is **not** the obvious one: `ADST_DCT` takes the *row* scan
/// and `DCT_ADST` the *column* scan, because the scan follows the direction in
/// which energy remains after the ADST. 32x32 has only the default scan, since
/// it is always `DCT_DCT`.
pub fn scan_order(tx_size: TxSize, tx_type: TxType) -> ScanOrder {
    use TxSize::*;
    use TxType::*;
    macro_rules! so {
        ($s:ident, $n:ident) => {
            ScanOrder { scan: &tables::$s, neighbors: &tables::$n }
        };
    }
    match (tx_size, tx_type) {
        (Tx4x4, AdstDct) => so!(ROW_SCAN_4X4, ROW_SCAN_4X4_NEIGHBORS),
        (Tx4x4, DctAdst) => so!(COL_SCAN_4X4, COL_SCAN_4X4_NEIGHBORS),
        (Tx4x4, _) => so!(DEFAULT_SCAN_4X4, DEFAULT_SCAN_4X4_NEIGHBORS),
        (Tx8x8, AdstDct) => so!(ROW_SCAN_8X8, ROW_SCAN_8X8_NEIGHBORS),
        (Tx8x8, DctAdst) => so!(COL_SCAN_8X8, COL_SCAN_8X8_NEIGHBORS),
        (Tx8x8, _) => so!(DEFAULT_SCAN_8X8, DEFAULT_SCAN_8X8_NEIGHBORS),
        (Tx16x16, AdstDct) => so!(ROW_SCAN_16X16, ROW_SCAN_16X16_NEIGHBORS),
        (Tx16x16, DctAdst) => so!(COL_SCAN_16X16, COL_SCAN_16X16_NEIGHBORS),
        (Tx16x16, _) => so!(DEFAULT_SCAN_16X16, DEFAULT_SCAN_16X16_NEIGHBORS),
        (Tx32x32, _) => so!(DEFAULT_SCAN_32X32, DEFAULT_SCAN_32X32_NEIGHBORS),
    }
}

/// `get_band_translate` — which coefficient band each scan position falls in.
fn band_translate(tx_size: TxSize) -> &'static [u8] {
    if tx_size == TxSize::Tx4x4 {
        &tables::COEFBAND_TRANS_4X4
    } else {
        &tables::COEFBAND_TRANS_8X8PLUS
    }
}

/// `get_coef_context`: the mean of the two scan neighbours' cached tokens.
#[inline(always)]
fn coef_context(nb: &[i16], token_cache: &[u8], c: usize) -> usize {
    let a = token_cache[nb[2 * c] as usize] as usize;
    let b = token_cache[nb[2 * c + 1] as usize] as usize;
    (1 + a + b) >> 1
}

/// Decode one transform block's coefficients into `dqcoeff` (already
/// dequantised, in **raster** order), returning the end-of-block position.
///
/// `dq` is `[dc_quant, ac_quant]`; `ctx` is the entropy context from the
/// above/left non-zero flags. `is_inter` selects the reference half of the
/// probability table — it is a property of the *block*, not the frame, so an
/// intra block in an inter frame uses the intra probabilities.
#[allow(clippy::too_many_arguments)]
pub fn decode_coefs(
    r: &mut BoolDecoder,
    fc: &FrameContext,
    // `[band][ctx][token]` and `[band][ctx]` slices of the frame's counters,
    // pre-indexed by transform size / plane / reference so the hot loop does
    // not re-index six levels per coefficient.
    tok_counts: &mut [[[u32; 4]; 6]; 6],
    eob_counts: &mut [[u32; 6]; 6],
    plane_type: usize,
    is_inter: bool,
    tx_size: TxSize,
    tx_type: TxType,
    dq: [i32; 2],
    mut ctx: usize,
    dqcoeff: &mut [i64],
) -> usize {
    let max_eob = tx_size.coefs();
    let so = scan_order(tx_size, tx_type);
    let bands = band_translate(tx_size);
    let probs = &fc.coef_probs[tx_size as usize][plane_type][is_inter as usize];
    let dq_shift = if tx_size == TxSize::Tx32x32 { 1 } else { 0 };

    let mut token_cache = [0u8; 32 * 32];
    let mut c = 0usize;
    let mut dqv = dq[0];
    let mut band_i = 0usize;

    while c < max_eob {
        let mut band = bands[band_i] as usize;
        band_i += 1;
        let mut prob = &probs[band][ctx.min(5)];
        eob_counts[band][ctx.min(5)] += 1;
        if r.read_bool(prob[EOB_CONTEXT_NODE]) == 0 {
            // `EOB_MODEL_TOKEN` is index 3 in the four-entry tally.
            tok_counts[band][ctx.min(5)][3] += 1;
            break; // end of block
        }
        // Run of zeros. The band keeps advancing through it.
        while r.read_bool(prob[ZERO_CONTEXT_NODE]) == 0 {
            tok_counts[band][ctx.min(5)][0] += 1;
            dqv = dq[1];
            token_cache[so.scan[c] as usize] = 0;
            c += 1;
            if c >= max_eob {
                return c; // trailing zeros, no EOB token
            }
            ctx = coef_context(so.neighbors, &token_cache, c);
            band = bands[band_i] as usize;
            band_i += 1;
            prob = &probs[band][ctx.min(5)];
        }

        let v: i64;
        if r.read_bool(prob[ONE_CONTEXT_NODE]) != 0 {
            tok_counts[band][ctx.min(5)][2] += 1;
            // The tail is coded against a Pareto-distributed row selected by
            // the *pivot* probability — 255 rows interpolating between the
            // stored ones, which is what makes three transmitted probabilities
            // per context enough for eleven tokens.
            let p = &tables::PARETO8_FULL[(prob[PIVOT_NODE] as usize).saturating_sub(1).min(254)];
            let val: i32;
            if r.read_bool(p[0]) != 0 {
                if r.read_bool(p[3]) != 0 {
                    token_cache[so.scan[c] as usize] = 5;
                    if r.read_bool(p[5]) != 0 {
                        if r.read_bool(p[7]) != 0 {
                            val = CAT6_MIN_VAL + read_coeff(r, &tables::CAT6_PROB, 14);
                        } else {
                            val = CAT5_MIN_VAL + read_coeff(r, &tables::CAT5_PROB, 5);
                        }
                    } else if r.read_bool(p[6]) != 0 {
                        val = CAT4_MIN_VAL + read_coeff(r, &tables::CAT4_PROB, 4);
                    } else {
                        val = CAT3_MIN_VAL + read_coeff(r, &tables::CAT3_PROB, 3);
                    }
                } else {
                    token_cache[so.scan[c] as usize] = 4;
                    if r.read_bool(p[4]) != 0 {
                        val = CAT2_MIN_VAL + read_coeff(r, &tables::CAT2_PROB, 2);
                    } else {
                        val = CAT1_MIN_VAL + read_coeff(r, &tables::CAT1_PROB, 1);
                    }
                }
                v = ((val as i64) * dqv as i64) >> dq_shift;
            } else if r.read_bool(p[1]) != 0 {
                token_cache[so.scan[c] as usize] = 3;
                v = (((3 + r.read_bool(p[2])) as i64) * dqv as i64) >> dq_shift;
            } else {
                token_cache[so.scan[c] as usize] = 2;
                v = (2 * dqv as i64) >> dq_shift;
            }
        } else {
            tok_counts[band][ctx.min(5)][1] += 1;
            token_cache[so.scan[c] as usize] = 1;
            v = (dqv as i64) >> dq_shift;
        }
        // The sign is a raw bit at p=128, *after* the magnitude.
        let pos = so.scan[c] as usize;
        if pos < dqcoeff.len() {
            dqcoeff[pos] = if r.read_bool(128) != 0 { -v } else { v };
        }
        c += 1;
        if c < max_eob {
            ctx = coef_context(so.neighbors, &token_cache, c);
        }
        dqv = dq[1];
    }
    c
}

/// `read_coeff` — `n` extra bits MSB-first, each with its own probability.
#[inline]
fn read_coeff(r: &mut BoolDecoder, probs: &[u8], n: usize) -> i32 {
    let mut val = 0i32;
    for i in 0..n {
        val = (val << 1) | r.read_bool(probs[i.min(probs.len() - 1)]) as i32;
    }
    val
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn scan_orders_are_permutations_of_their_block() {
        // Every scan must visit each raster position exactly once, and every
        // neighbour index must point inside the block. A generated table that
        // parsed with the wrong dimensions fails both.
        for &ts in &[TxSize::Tx4x4, TxSize::Tx8x8, TxSize::Tx16x16, TxSize::Tx32x32] {
            for &tt in &[TxType::DctDct, TxType::AdstDct, TxType::DctAdst, TxType::AdstAdst] {
                let so = scan_order(ts, tt);
                let n = ts.coefs();
                assert_eq!(so.scan.len(), n, "{:?}/{:?} scan length", ts, tt);
                assert_eq!(so.neighbors.len(), (n + 1) * 2, "{:?}/{:?} neighbour length", ts, tt);
                let mut seen = alloc::vec![false; n];
                for &p in so.scan {
                    let p = p as usize;
                    assert!(p < n, "{:?}/{:?}: scan position {} outside the block", ts, tt, p);
                    assert!(!seen[p], "{:?}/{:?}: scan visits {} twice", ts, tt, p);
                    seen[p] = true;
                }
                assert!(seen.iter().all(|&s| s), "{:?}/{:?}: scan misses a position", ts, tt);
                for &nb in so.neighbors {
                    assert!((nb as usize) < n, "{:?}/{:?}: neighbour outside the block", ts, tt);
                }
            }
        }
    }

    #[test_case]
    fn adst_takes_the_row_scan_and_dct_adst_the_column_scan() {
        // The mapping is the reverse of what the names suggest, and getting it
        // backwards is invisible except as slightly worse prediction.
        assert_eq!(scan_order(TxSize::Tx4x4, TxType::AdstDct).scan[1], tables::ROW_SCAN_4X4[1]);
        assert_eq!(scan_order(TxSize::Tx4x4, TxType::DctAdst).scan[1], tables::COL_SCAN_4X4[1]);
        assert_eq!(scan_order(TxSize::Tx4x4, TxType::DctDct).scan[1], tables::DEFAULT_SCAN_4X4[1]);
        // 32x32 is always the default scan whatever the type says.
        for &tt in &[TxType::DctDct, TxType::AdstDct, TxType::DctAdst, TxType::AdstAdst] {
            assert_eq!(scan_order(TxSize::Tx32x32, tt).scan[5], tables::DEFAULT_SCAN_32X32[5]);
        }
    }

    #[test_case]
    fn the_first_scan_position_is_always_dc() {
        for &ts in &[TxSize::Tx4x4, TxSize::Tx8x8, TxSize::Tx16x16, TxSize::Tx32x32] {
            for &tt in &[TxType::DctDct, TxType::AdstDct, TxType::DctAdst] {
                assert_eq!(scan_order(ts, tt).scan[0], 0);
            }
        }
    }

    #[test_case]
    fn band_translate_covers_the_largest_block() {
        // The 8x8-plus band table is indexed once per coefficient of a 32x32
        // block, so it must have 1024 entries — a shorter one would panic
        // partway through the biggest transform and nowhere else.
        assert_eq!(band_translate(TxSize::Tx4x4).len(), 16);
        assert_eq!(band_translate(TxSize::Tx32x32).len(), 1024);
        assert!(band_translate(TxSize::Tx32x32).iter().all(|&b| b < 6));
        assert!(band_translate(TxSize::Tx4x4).iter().all(|&b| b < 6));
    }

    #[test_case]
    fn an_immediate_eob_decodes_no_coefficients() {
        // A partition of zero bytes decodes zeros, which at the default
        // probabilities reads as "end of block" straight away.
        let fc = FrameContext::default();
        let mut r = BoolDecoder::new(&[0x00; 8]).unwrap();
        let mut coefs = [0i64; 16];
        let mut tc = [[[0u32; 4]; 6]; 6];
        let mut ec = [[0u32; 6]; 6];
        let eob = decode_coefs(
            &mut r,
            &fc,
            &mut tc,
            &mut ec,
            0,
            false,
            TxSize::Tx4x4,
            TxType::DctDct,
            [10, 12],
            0,
            &mut coefs,
        );
        assert_eq!(eob, 0);
        assert!(coefs.iter().all(|&c| c == 0));
    }
}
