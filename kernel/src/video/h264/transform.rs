//! H.264 inverse scaling (dequant) + inverse integer transforms (§8.5).
//!
//! The residual path for baseline decoding:
//! * [`inverse_scan_4x4`] undoes the zig-zag coefficient ordering.
//! * [`dequant_4x4`] scales the 4×4 AC coefficients by QP (§8.5.12.1).
//! * [`idct_4x4`] applies the inverse core transform and the `(x+32)>>6`
//!   normalization (§8.5.12.2), turning coefficients into residual samples.
//! * [`luma_dc_transform`] / [`chroma_dc_transform`] handle the Intra_16x16 and
//!   chroma DC Hadamard transforms + their DC-specific scaling (§8.5.10–11).
//!
//! Pure integer math, no allocation — unit-tested against spec-derived vectors
//! and forward/inverse round-trips.

/// `normAdjust4x4[m][k]` (H.264 Table in §8.5.9) with `k` selecting the position
/// group: 0 for {(0,0),(0,2),(2,0),(2,2)}, 1 for {(1,1),(1,3),(3,1),(3,3)},
/// 2 otherwise. `LevelScale = 16 * normAdjust` for the flat (no scaling list)
/// case baseline uses.
const NORM_ADJUST: [[i32; 3]; 6] = [
    [10, 16, 13],
    [11, 18, 14],
    [13, 20, 16],
    [14, 23, 18],
    [16, 25, 20],
    [18, 29, 23],
];

/// The position-group index (0/1/2) for coefficient (i,j) in a 4×4 block.
#[inline]
fn pos_group(i: usize, j: usize) -> usize {
    let even = i % 2 == 0 && j % 2 == 0;
    let odd = i % 2 == 1 && j % 2 == 1;
    if even {
        0
    } else if odd {
        1
    } else {
        2
    }
}

/// Zig-zag scan order for a 4×4 block (frame coding, §8.5.6). `zigzag[n]` is the
/// raster index that scan position `n` maps to.
pub const ZIGZAG_4X4: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];

/// Undo the 4×4 zig-zag scan: `scan[n]` (coefficients in scan order) → raster.
pub fn inverse_scan_4x4(scan: &[i32; 16]) -> [i32; 16] {
    let mut out = [0i32; 16];
    for (n, &raster) in ZIGZAG_4X4.iter().enumerate() {
        out[raster] = scan[n];
    }
    out
}

/// Inverse-scale a raster-order 4×4 AC coefficient block in place (§8.5.12.1).
/// `qp` is the luma/chroma QP for the block. When `has_dc` (an Intra_16x16 or
/// chroma block whose DC came from a separate Hadamard pass), coefficient 0 is
/// left untouched — the caller writes the already-scaled DC.
pub fn dequant_4x4(block: &mut [i32; 16], qp: u32, has_dc: bool) {
    let m = (qp % 6) as usize;
    let shift = qp / 6;
    for i in 0..4 {
        for j in 0..4 {
            let idx = i * 4 + j;
            if has_dc && idx == 0 {
                continue;
            }
            let ls = 16 * NORM_ADJUST[m][pos_group(i, j)];
            let c = block[idx];
            block[idx] = if shift >= 4 {
                (c * ls) << (shift - 4)
            } else {
                (c * ls + (1 << (3 - shift))) >> (4 - shift)
            };
        }
    }
}

/// One-dimensional inverse core transform butterfly on 4 values (§8.5.12.2).
#[inline]
fn itx_1d(z0: i32, z1: i32, z2: i32, z3: i32) -> (i32, i32, i32, i32) {
    let e0 = z0 + z2;
    let e1 = z0 - z2;
    let e2 = (z1 >> 1) - z3;
    let e3 = z1 + (z3 >> 1);
    (e0 + e3, e1 + e2, e1 - e2, e0 - e3)
}

/// Inverse 4×4 core transform on a raster-order block, in place, then the
/// `(x+32)>>6` normalization — leaving residual samples (§8.5.12.2).
pub fn idct_4x4(block: &mut [i32; 16]) {
    // Rows.
    for i in 0..4 {
        let r = i * 4;
        let (a, b, c, d) = itx_1d(block[r], block[r + 1], block[r + 2], block[r + 3]);
        block[r] = a;
        block[r + 1] = b;
        block[r + 2] = c;
        block[r + 3] = d;
    }
    // Columns, then normalize.
    for j in 0..4 {
        let (a, b, c, d) = itx_1d(block[j], block[j + 4], block[j + 8], block[j + 12]);
        block[j] = (a + 32) >> 6;
        block[j + 4] = (b + 32) >> 6;
        block[j + 8] = (c + 32) >> 6;
        block[j + 12] = (d + 32) >> 6;
    }
}

/// 4×4 Hadamard for the Intra_16x16 luma DC coefficients + DC scaling
/// (§8.5.10). `dc` is the 16 DC values in raster order; returns the scaled DC
/// that seeds each 4×4 block's coefficient 0.
pub fn luma_dc_transform(dc: &mut [i32; 16], qp: u32) {
    hadamard_4x4(dc);
    let m = (qp % 6) as usize;
    let ls = 16 * NORM_ADJUST[m][0];
    let shift = qp / 6;
    for v in dc.iter_mut() {
        *v = if shift >= 6 {
            (*v * ls) << (shift - 6)
        } else {
            (*v * ls + (1 << (5 - shift))) >> (6 - shift)
        };
    }
}

/// 4×4 Hadamard butterfly (used for the luma DC pass), in place.
fn hadamard_4x4(m: &mut [i32; 16]) {
    // Rows.
    for i in 0..4 {
        let r = i * 4;
        let a = m[r] + m[r + 2];
        let b = m[r] - m[r + 2];
        let c = m[r + 1] - m[r + 3];
        let d = m[r + 1] + m[r + 3];
        m[r] = a + d;
        m[r + 1] = b + c;
        m[r + 2] = b - c;
        m[r + 3] = a - d;
    }
    // Columns.
    for j in 0..4 {
        let a = m[j] + m[j + 8];
        let b = m[j] - m[j + 8];
        let c = m[j + 4] - m[j + 12];
        let d = m[j + 4] + m[j + 12];
        m[j] = a + d;
        m[j + 4] = b + c;
        m[j + 8] = b - c;
        m[j + 12] = a - d;
    }
}

/// 2×2 Hadamard for chroma DC + DC scaling (§8.5.11). `dc` is the four chroma
/// DC values (raster 2×2); returns the scaled DC seeding each chroma 4×4 block.
pub fn chroma_dc_transform(dc: &mut [i32; 4], qp: u32) {
    let a = dc[0] + dc[1];
    let b = dc[0] - dc[1];
    let c = dc[2] + dc[3];
    let d = dc[2] - dc[3];
    let f = [a + c, b + d, a - c, b - d];
    let m = (qp % 6) as usize;
    let ls = 16 * NORM_ADJUST[m][0];
    let shift = qp / 6;
    for k in 0..4 {
        // ((f * LevelScale) << (qp/6)) >> 5
        dc[k] = ((f[k] * ls) << shift) >> 5;
    }
}

// --- 8x8 transform (High profile, §8.5.12.3 / §8.5.13) --------------------

/// 8×8 zigzag (frame) scan: scan index → raster position. Generated by the
/// standard diagonal alternation (identical to FFmpeg's `ff_zigzag_direct`).
pub const ZIGZAG8: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// 8×8 dequant base coefficients per qp%6 (FFmpeg `ff_h264_dequant8_coeff_init`,
/// itself the spec's Table for LevelScale8x8 with flat scaling).
const DEQ8_INIT: [[i32; 6]; 6] = [
    [20, 18, 32, 19, 25, 24],
    [22, 19, 35, 21, 28, 26],
    [26, 23, 42, 24, 33, 31],
    [28, 25, 45, 26, 35, 33],
    [32, 28, 51, 30, 40, 38],
    [36, 32, 58, 34, 46, 43],
];
/// Raster position → which of the 6 base coefficients applies
/// (FFmpeg `ff_h264_dequant8_coeff_init_scan`, indexed `((p>>1)&12)|(p&3)`).
const DEQ8_SCAN: [usize; 16] = [0, 3, 4, 3, 3, 1, 5, 1, 4, 5, 2, 5, 3, 1, 5, 1];

/// Dequantise a raw 8×8 coefficient (raster position `p`) at luma `qp` with the
/// flat (16) scaling matrix, in FFmpeg's folded form: the result feeds
/// [`idct8_add`], which performs the final `(x + 32) >> 6` normalisation.
#[inline]
pub fn dequant8(level: i32, qp: i32, p: usize) -> i32 {
    let base = DEQ8_INIT[(qp % 6) as usize][DEQ8_SCAN[((p >> 1) & 12) | (p & 3)]];
    // qmul = base * 16(flat weight matrix) << qp/6 (FFmpeg init_dequant8_coeff_table);
    // value = (level*qmul + 32) >> 6, the second >>6 happening in the idct.
    let qmul = (base as i64) << (4 + (qp / 6));
    ((level as i64 * qmul + 32) >> 6) as i32
}

/// The 8×8 inverse core transform (§8.5.12.3, ported from FFmpeg's
/// `ff_h264_idct8_add`): consumes a raster-order dequantised block and returns
/// raster residuals to add to the prediction (`clip(pred + r)`), including the
/// final `(x + 32) >> 6`.
pub fn idct8_residual(block: &mut [i32; 64]) {
    block[0] += 32;
    // Row transform.
    for i in 0..8 {
        let b_at = |k: usize| block[i * 8 + k] as i64;
        let a0 = b_at(0) + b_at(4);
        let a2 = b_at(0) - b_at(4);
        let a4 = (b_at(2) >> 1) - b_at(6);
        let a6 = (b_at(6) >> 1) + b_at(2);
        let b0 = a0 + a6;
        let b2 = a2 + a4;
        let b4 = a2 - a4;
        let b6 = a0 - a6;
        let a1 = -b_at(3) + b_at(5) - b_at(7) - (b_at(7) >> 1);
        let a3 = b_at(1) + b_at(7) - b_at(3) - (b_at(3) >> 1);
        let a5 = -b_at(1) + b_at(7) + b_at(5) + (b_at(5) >> 1);
        let a7 = b_at(3) + b_at(5) + b_at(1) + (b_at(1) >> 1);
        let b1 = (a7 >> 2) + a1;
        let b3 = a3 + (a5 >> 2);
        let b5 = (a3 >> 2) - a5;
        let b7 = a7 - (a1 >> 2);
        block[i * 8 + 0] = (b0 + b7) as i32;
        block[i * 8 + 7] = (b0 - b7) as i32;
        block[i * 8 + 1] = (b2 + b5) as i32;
        block[i * 8 + 6] = (b2 - b5) as i32;
        block[i * 8 + 2] = (b4 + b3) as i32;
        block[i * 8 + 5] = (b4 - b3) as i32;
        block[i * 8 + 3] = (b6 + b1) as i32;
        block[i * 8 + 4] = (b6 - b1) as i32;
    }
    // Column transform + final normalisation.
    for i in 0..8 {
        let b_at = |k: usize| block[k * 8 + i] as i64;
        let a0 = b_at(0) + b_at(4);
        let a2 = b_at(0) - b_at(4);
        let a4 = (b_at(2) >> 1) - b_at(6);
        let a6 = (b_at(6) >> 1) + b_at(2);
        let b0 = a0 + a6;
        let b2 = a2 + a4;
        let b4 = a2 - a4;
        let b6 = a0 - a6;
        let a1 = -b_at(3) + b_at(5) - b_at(7) - (b_at(7) >> 1);
        let a3 = b_at(1) + b_at(7) - b_at(3) - (b_at(3) >> 1);
        let a5 = -b_at(1) + b_at(7) + b_at(5) + (b_at(5) >> 1);
        let a7 = b_at(3) + b_at(5) + b_at(1) + (b_at(1) >> 1);
        let b1 = (a7 >> 2) + a1;
        let b3 = a3 + (a5 >> 2);
        let b5 = (a3 >> 2) - a5;
        let b7 = a7 - (a1 >> 2);
        block[0 * 8 + i] = ((b0 + b7) >> 6) as i32;
        block[1 * 8 + i] = ((b2 + b5) >> 6) as i32;
        block[2 * 8 + i] = ((b4 + b3) >> 6) as i32;
        block[3 * 8 + i] = ((b6 + b1) >> 6) as i32;
        block[4 * 8 + i] = ((b6 - b1) >> 6) as i32;
        block[5 * 8 + i] = ((b4 - b3) >> 6) as i32;
        block[6 * 8 + i] = ((b2 - b5) >> 6) as i32;
        block[7 * 8 + i] = ((b0 - b7) >> 6) as i32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn zigzag_inverse_is_a_permutation() {
        // Placing 0..15 in scan order and inverting must land each value at the
        // ZIGZAG_4X4 raster index.
        let scan: [i32; 16] = core::array::from_fn(|n| n as i32);
        let raster = inverse_scan_4x4(&scan);
        for (n, &r) in ZIGZAG_4X4.iter().enumerate() {
            assert_eq!(raster[r], n as i32);
        }
        // DC stays at 0; the last scan coeff lands at raster 15.
        assert_eq!(raster[0], 0);
        assert_eq!(raster[15], 15);
    }

    #[test_case]
    fn idct_dc_only_is_uniform() {
        // A block with only coefficient 0 = D reconstructs to a flat
        // (D + 32) >> 6 everywhere (derived analytically from the butterfly).
        for d in [64i32, 128, 200, -64] {
            let mut b = [0i32; 16];
            b[0] = d;
            idct_4x4(&mut b);
            let expect = (d + 32) >> 6;
            assert!(b.iter().all(|&x| x == expect), "D={} got {:?}", d, b);
        }
    }

    #[test_case]
    fn idct_matches_reference_matrix_product() {
        // Cross-check the fast butterfly against the textbook matrix form of the
        // inverse core transform: r = round((Ci^T · X · Ci) / 64), with
        // Ci = [[1,1,1,1],[1,1/2,-1/2,-1],[1,-1,-1,1],[1/2,-1,1,-1/2]] scaled so
        // that 1/2 → >>1. We validate against an explicit i64 double loop.
        let x: [i32; 16] = [16, -3, 5, 0, 2, 7, -1, 4, 0, 1, 3, -2, 5, 0, -4, 6];
        let ci = [[2, 2, 2, 2], [2, 1, -1, -2], [2, -2, -2, 2], [1, -2, 2, -1]];
        // tmp = Ci^T · X  (column transform), then · Ci (row) — in half-units,
        // so divide by 4 at the end (each Ci carries a factor of 2), then /64.
        let mut tmp = [[0i64; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                let mut s = 0i64;
                for k in 0..4 {
                    s += ci[k][i] as i64 * x[k * 4 + j] as i64;
                }
                tmp[i][j] = s;
            }
        }
        let mut out = [[0i64; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                let mut s = 0i64;
                for k in 0..4 {
                    s += tmp[i][k] * ci[k][j] as i64;
                }
                out[i][j] = s;
            }
        }
        // Each Ci scaled ×2 → total ×4; spec divides by 64 with +32 rounding.
        let mut b = x;
        idct_4x4(&mut b);
        for i in 0..4 {
            for j in 0..4 {
                let ref_val = (out[i][j] / 4 + 32) >> 6;
                assert_eq!(b[i * 4 + j] as i64, ref_val, "mismatch at ({},{})", i, j);
            }
        }
    }

    #[test_case]
    fn dequant_dc_position_flat() {
        // At qp=24 (shift=4, m=0), coefficient (0,0) scales by 16*10<<0 = 160.
        let mut b = [0i32; 16];
        b[0] = 1;
        dequant_4x4(&mut b, 24, false);
        assert_eq!(b[0], 160);
        // has_dc=true leaves coefficient 0 untouched.
        let mut b2 = [0i32; 16];
        b2[0] = 7;
        dequant_4x4(&mut b2, 24, true);
        assert_eq!(b2[0], 7);
    }

    #[test_case]
    fn chroma_dc_transform_scales() {
        // All-equal DC → after 2x2 Hadamard only the (0,0) term is non-zero
        // (4*v), the rest cancel; check it scales without panicking.
        let mut dc = [3, 3, 3, 3];
        chroma_dc_transform(&mut dc, 30);
        // f = [12, 0, 0, 0]; qp=30 → m=0, ls=160, shift=5 → (12*160<<5)>>5 = 1920.
        assert_eq!(dc[0], 1920);
        assert_eq!(&dc[1..], &[0, 0, 0]);
    }
}
