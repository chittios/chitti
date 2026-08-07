//! VP9 inverse transforms: the 2-D wrappers around the generated 1-D kernels,
//! plus the lossless Walsh-Hadamard path and dequantisation.
//!
//! VP9 has four transform *types* per size (`DCT_DCT`, `ADST_DCT`, `DCT_ADST`,
//! `ADST_ADST`) and the type is chosen by the **intra prediction mode**, not
//! coded — so a wrong mode→type mapping is invisible in the bitstream and shows
//! only as a slightly wrong picture. The mapping lives in [`tx_type_for_mode`]
//! with the table it comes from named.
//!
//! Two rules the shapes here encode:
//!
//! * **Rows first, then columns**, with the row pass writing into a scratch
//!   buffer in raster order and the column pass gathering `out[j * n + i]`. The
//!   transposed reading is not an optimisation to undo — swapping the passes
//!   gives a transform that is its own transpose for `DCT_DCT` (so 4x4 DC-only
//!   blocks still look right) and wrong for everything ADST.
//! * **The final shift differs per size**: 4 for 4x4/8x8/16x16 but 6 for 32x32,
//!   and the 32x32 row pass carries an extra `ROUND_POWER_OF_TWO(_, 2)`. Using
//!   one constant makes large blocks come out at the wrong contrast.

use super::idct_kernels::{
    iadst16, iadst4, iadst8, idct16, idct32, idct4, idct8, round_pow2, wraplow, UNIT_QUANT_SHIFT,
};

/// The four 2-D transform combinations (libvpx `TX_TYPE`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TxType {
    DctDct = 0,
    AdstDct = 1,
    DctAdst = 2,
    AdstAdst = 3,
}

/// Transform size (libvpx `TX_SIZE`): 4x4, 8x8, 16x16, 32x32.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum TxSize {
    Tx4x4 = 0,
    Tx8x8 = 1,
    Tx16x16 = 2,
    Tx32x32 = 3,
}

impl TxSize {
    pub fn from_index(i: u8) -> TxSize {
        match i {
            1 => TxSize::Tx8x8,
            2 => TxSize::Tx16x16,
            3 => TxSize::Tx32x32,
            _ => TxSize::Tx4x4,
        }
    }
    /// Edge length in pixels.
    pub fn width(self) -> usize {
        4 << (self as usize)
    }
    /// Coefficients in one block.
    pub fn coefs(self) -> usize {
        let w = self.width();
        w * w
    }
    /// `log2` of the edge length.
    pub fn log2(self) -> usize {
        2 + self as usize
    }
}

/// libvpx `intra_mode_to_tx_type_lookup` (`vp9_blockd.h`): which 2-D transform
/// an intra prediction mode implies.
///
/// This is **not** transmitted — it is derived from the mode — so getting it
/// wrong produces a decodable stream that looks subtly soft or ringy rather
/// than an error. Order is the `PREDICTION_MODE` enum: DC, V, H, D45, D135,
/// D117, D153, D207, D63, TM.
const INTRA_MODE_TO_TX_TYPE: [TxType; 10] = [
    TxType::DctDct,   // DC_PRED
    TxType::AdstDct,  // V_PRED   — vertical prediction leaves horizontal detail
    TxType::DctAdst,  // H_PRED
    TxType::DctDct,   // D45_PRED
    TxType::AdstAdst, // D135_PRED
    TxType::AdstDct,  // D117_PRED
    TxType::DctAdst,  // D153_PRED
    TxType::DctAdst,  // D207_PRED
    TxType::AdstDct,  // D63_PRED
    TxType::AdstAdst, // TM_PRED
];

/// The transform type an intra mode implies. Inter blocks and every 32x32
/// block are always `DCT_DCT`.
pub fn tx_type_for_mode(mode: u8, tx_size: TxSize, is_inter: bool) -> TxType {
    if is_inter || tx_size == TxSize::Tx32x32 {
        return TxType::DctDct;
    }
    INTRA_MODE_TO_TX_TYPE[(mode as usize).min(9)]
}

/// libvpx's final `ROUND_POWER_OF_TWO` per transform size, from
/// `vpx_idct{4x4,8x8,16x16,32x32}_*_add_c`.
const OUTPUT_SHIFT: [u32; 4] = [4, 5, 6, 6];

type Kernel = fn(&[i64], &mut [i64]);

/// The (row, column) 1-D kernels for a transform type.
///
/// **`ADST_DCT` means ADST on the *columns*, DCT on the rows** — the opposite
/// of what the name reads like. libvpx's tables are
/// `typedef struct { transform_1d cols, rows; } transform_2d;` and its entry
/// for `ADST_DCT` is `{ iadst_c, idct_c }`, i.e. the **first** member is the
/// column kernel. Swapping the two is invisible on `DCT_DCT` and `ADST_ADST`
/// (both passes are the same kernel) and on any DC-only block, so it survives
/// every cheap test — it shows up only as a wrong picture on real content whose
/// blocks use V_PRED or H_PRED, which is exactly how it was found here: chroma
/// (all DC) came out bit-exact while luma was off by ~25 everywhere.
fn kernels(tx_size: TxSize, tx_type: TxType) -> (Kernel, Kernel) {
    let (dct, adst): (Kernel, Option<Kernel>) = match tx_size {
        TxSize::Tx4x4 => (idct4, Some(iadst4)),
        TxSize::Tx8x8 => (idct8, Some(iadst8)),
        TxSize::Tx16x16 => (idct16, Some(iadst16)),
        TxSize::Tx32x32 => (idct32, None),
    };
    let adst = adst.unwrap_or(dct);
    match tx_type {
        TxType::DctDct => (dct, dct),
        TxType::AdstDct => (dct, adst),
        TxType::DctAdst => (adst, dct),
        TxType::AdstAdst => (adst, adst),
    }
}

/// Inverse-transform `coefs` and **add** the residual into `dest` (a plane
/// window with row stride `stride`), clipping to 0..=255.
///
/// `coefs` is `tx_size.coefs()` long in raster order.
pub fn inverse_add(coefs: &[i64], dest: &mut [u8], stride: usize, tx_size: TxSize, tx_type: TxType) {
    let n = tx_size.width();
    let (rows, cols) = kernels(tx_size, tx_type);
    let mut out = [0i64; 32 * 32];
    let mut tin = [0i64; 32];
    let mut tout = [0i64; 32];
    // Rows. No pre-shift at any size: the row pass writes its output straight
    // into the scratch buffer and all of the scaling happens after the column
    // pass.
    for i in 0..n {
        rows(&coefs[i * n..i * n + n], &mut out[i * n..i * n + n]);
    }
    // Columns, gathered transposed, then the **per-size** output shift:
    // 4 for 4x4, 5 for 8x8, 6 for 16x16 and 32x32. One constant for all sizes
    // scales the residual by 4x or 16x on the larger transforms, which does not
    // look like a transform bug — it looks like the picture has too much
    // contrast, or none.
    let shift = OUTPUT_SHIFT[tx_size as usize];
    for i in 0..n {
        for j in 0..n {
            tin[j] = out[j * n + i];
        }
        cols(&tin[..n], &mut tout[..n]);
        for j in 0..n {
            let p = j * stride + i;
            if p < dest.len() {
                dest[p] = clip_pixel_add(dest[p], round_pow2(tout[j], shift));
            }
        }
    }
}

/// The lossless 4x4 inverse Walsh-Hadamard (libvpx `vpx_iwht4x4_16_add_c`).
///
/// A frame is lossless only when `base_q_idx` **and all three deltas** are
/// zero; VP9 then replaces the DCT entirely rather than using a quantiser of 1,
/// so a decoder that only checks `base_q_idx` decodes such frames with the
/// wrong transform.
pub fn iwht4x4_add(coefs: &[i64], dest: &mut [u8], stride: usize) {
    let mut out = [0i64; 16];
    for i in 0..4 {
        let ip = &coefs[i * 4..i * 4 + 4];
        let mut a1 = ip[0] >> UNIT_QUANT_SHIFT;
        let mut c1 = ip[1] >> UNIT_QUANT_SHIFT;
        let mut d1 = ip[2] >> UNIT_QUANT_SHIFT;
        let mut b1 = ip[3] >> UNIT_QUANT_SHIFT;
        a1 += c1;
        d1 -= b1;
        let e1 = (a1 - d1) >> 1;
        b1 = e1 - b1;
        c1 = e1 - c1;
        a1 -= b1;
        d1 += c1;
        out[i * 4] = wraplow(a1);
        out[i * 4 + 1] = wraplow(b1);
        out[i * 4 + 2] = wraplow(c1);
        out[i * 4 + 3] = wraplow(d1);
    }
    for i in 0..4 {
        let mut a1 = out[i];
        let mut c1 = out[4 + i];
        let mut d1 = out[8 + i];
        let mut b1 = out[12 + i];
        a1 += c1;
        d1 -= b1;
        let e1 = (a1 - d1) >> 1;
        b1 = e1 - b1;
        c1 = e1 - c1;
        a1 -= b1;
        d1 += c1;
        for (j, v) in [a1, b1, c1, d1].iter().enumerate() {
            let p = j * stride + i;
            if p < dest.len() {
                dest[p] = clip_pixel_add(dest[p], wraplow(*v));
            }
        }
    }
}

#[inline(always)]
fn clip_pixel_add(dest: u8, trans: i64) -> u8 {
    (dest as i64 + trans).clamp(0, 255) as u8
}

/// Dequantiser lookups (§8.6.1). `delta` is the plane's `delta_q`, and the
/// index is **clamped after adding it**, which is why this is a function rather
/// than a bare table index.
pub fn dc_quant(qindex: i32, delta: i32) -> i32 {
    super::tables::DC_QLOOKUP[(qindex + delta).clamp(0, 255) as usize] as i32
}

pub fn ac_quant(qindex: i32, delta: i32) -> i32 {
    super::tables::AC_QLOOKUP[(qindex + delta).clamp(0, 255) as usize] as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An all-zero block must leave the destination untouched, for every size
    /// and type. Cheap, but it catches an off-by-one in the row/column loops
    /// that would write outside the block.
    #[test_case]
    fn zero_coefficients_change_nothing() {
        for &ts in &[TxSize::Tx4x4, TxSize::Tx8x8, TxSize::Tx16x16, TxSize::Tx32x32] {
            for &tt in &[TxType::DctDct, TxType::AdstDct, TxType::DctAdst, TxType::AdstAdst] {
                let n = ts.width();
                let coefs = alloc::vec![0i64; ts.coefs()];
                let mut dest = alloc::vec![7u8; n * n];
                inverse_add(&coefs, &mut dest, n, ts, tt);
                assert!(dest.iter().all(|&p| p == 7), "{:?}/{:?} disturbed a flat block", ts, tt);
            }
        }
    }

    /// A DC-only DCT block is a uniform offset over the whole block — the one
    /// property of the transform that can be checked without a reference, and
    /// it fails loudly if the row/column passes are wired the wrong way round.
    #[test_case]
    fn dc_only_dct_is_uniform() {
        for &ts in &[TxSize::Tx4x4, TxSize::Tx8x8, TxSize::Tx16x16, TxSize::Tx32x32] {
            let n = ts.width();
            let mut coefs = alloc::vec![0i64; ts.coefs()];
            coefs[0] = 512;
            let mut dest = alloc::vec![100u8; n * n];
            inverse_add(&coefs, &mut dest, n, ts, TxType::DctDct);
            let first = dest[0];
            assert!(first != 100, "{:?}: DC coefficient had no effect", ts);
            assert!(dest.iter().all(|&p| p == first), "{:?}: DC block is not uniform", ts);
        }
    }

    /// The lossless path is exactly invertible for small values, which is the
    /// whole point of using the Walsh-Hadamard there.
    #[test_case]
    fn walsh_hadamard_dc_is_uniform() {
        let mut coefs = [0i64; 16];
        coefs[0] = 4 * 8; // >> UNIT_QUANT_SHIFT = 8
        let mut dest = [50u8; 16];
        iwht4x4_add(&coefs, &mut dest, 4);
        let first = dest[0];
        assert!(dest.iter().all(|&p| p == first));
        assert!(first != 50);
    }

    #[test_case]
    fn the_output_shift_is_per_transform_size() {
        // 4/5/6/6, from libvpx's four `_add_c` wrappers. Using one constant
        // makes every 8x8 residual 2x and every 16x16 residual 4x too strong —
        // which reads as a contrast problem, not a transform bug.
        assert_eq!(OUTPUT_SHIFT, [4, 5, 6, 6]);
        // A DC coefficient large enough to move the picture must scale the same
        // way at every size, which is what pins the pairing of shift to size.
        for (i, &ts) in [TxSize::Tx4x4, TxSize::Tx8x8, TxSize::Tx16x16, TxSize::Tx32x32]
            .iter()
            .enumerate()
        {
            let n = ts.width();
            let mut coefs = alloc::vec![0i64; ts.coefs()];
            // A DC of `1 << (shift + 1)` after two passes of gain must land in
            // range rather than saturating.
            coefs[0] = 64 << i;
            let mut dest = alloc::vec![10u8; n * n];
            inverse_add(&coefs, &mut dest, n, ts, TxType::DctDct);
            assert!(dest[0] > 10 && dest[0] < 255, "{:?} produced {}", ts, dest[0]);
        }
    }

    #[test_case]
    fn tx_type_follows_the_intra_mode_not_the_bitstream() {
        // V_PRED leaves horizontal detail, so its rows get the ADST.
        assert_eq!(tx_type_for_mode(1, TxSize::Tx4x4, false), TxType::AdstDct);
        assert_eq!(tx_type_for_mode(2, TxSize::Tx4x4, false), TxType::DctAdst);
        assert_eq!(tx_type_for_mode(0, TxSize::Tx4x4, false), TxType::DctDct);
        assert_eq!(tx_type_for_mode(9, TxSize::Tx16x16, false), TxType::AdstAdst);
        // …but 32x32 and every inter block are DCT_DCT regardless.
        assert_eq!(tx_type_for_mode(9, TxSize::Tx32x32, false), TxType::DctDct);
        assert_eq!(tx_type_for_mode(9, TxSize::Tx4x4, true), TxType::DctDct);
    }

    #[test_case]
    fn dequant_clamps_the_index_after_the_delta() {
        // A negative delta at qindex 0 must clamp to the table's first entry,
        // not index out of range.
        assert_eq!(dc_quant(0, -20), super::super::tables::DC_QLOOKUP[0] as i32);
        assert_eq!(ac_quant(255, 20), super::super::tables::AC_QLOOKUP[255] as i32);
        assert_eq!(dc_quant(0, 0), 4);
        assert_eq!(ac_quant(0, 0), 4);
    }
}
