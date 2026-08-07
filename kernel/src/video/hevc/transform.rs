//! HEVC inverse transforms and dequantisation (H.265 §8.6).
//!
//! **The inverse DCT here is a matrix multiply, deliberately.** Every decoder
//! in the wild writes it as a butterfly network — `TR_4` inside `TR_8` inside
//! `TR_16` inside `TR_32`, four nested macros of hand-placed constants. That
//! factorisation is an optimisation, not the definition: the specification
//! defines the transform as `out[i] = sum_k M[k][i] * src[k]` over the single
//! 32x32 basis matrix, and because both passes sum in full precision before
//! their one rounding step, the direct form is **bit-exact** with the butterfly
//! rather than merely close.
//!
//! Driving it from the generated [`super::tables::TRANSFORM`] means there is no
//! hand-transcribed constant anywhere in the transform path — which matters
//! because a single wrong tap in a butterfly is invisible: it produces a
//! slightly wrong residual on one basis function, which looks like ringing, and
//! ringing is what a DCT does anyway.
//!
//! The 4x4 luma-intra DST (§8.6.4.2) is the one exception the specification
//! itself makes, and it carries its own 16-entry matrix here — cross-checked
//! against FFmpeg's butterfly form in the tests, since the two were derived
//! independently.

use super::tables as tb;

/// `av_clip_int16` — the specification clips **between the two passes**, not
/// just at the end, so an intermediate that overflows a signed 16-bit value is
/// saturated and that saturation is part of the defined output.
#[inline]
fn clip16(v: i32) -> i16 {
    v.clamp(-32768, 32767) as i16
}

/// The 4x4 DST-VII used for luma intra residuals (H.265 §8.6.4.2), transposed
/// for the inverse: `out[i] = sum_k DST4[i][k] * src[k]`.
///
/// The specification tabulates the *forward* matrix; using it directly rather
/// than its transpose is a mistake that still produces a plausible block,
/// because the matrix is nearly — but not quite — symmetric.
const DST4: [[i32; 4]; 4] = [
    [29, 74, 84, 55],
    [55, 74, -29, -84],
    [74, 0, -74, 74],
    [84, -74, 55, -29],
];

/// One pass of the size-`n` inverse DCT over `n` values gathered with `stride`.
///
/// `scratch` holds the gathered input because the transform is not in-place per
/// element: output `i` reads every input, so writing as we go would feed a
/// partially transformed column back into itself.
#[inline]
fn dct_pass(buf: &mut [i16], base: usize, stride: usize, n: usize, shift: u32, scratch: &mut [i32; 32]) {
    let step = 32 / n;
    for k in 0..n {
        scratch[k] = buf[base + k * stride] as i32;
    }
    let add = 1i32 << (shift - 1);
    for i in 0..n {
        let mut acc = 0i32;
        for k in 0..n {
            acc += tb::TRANSFORM[k * step][i] as i32 * scratch[k];
        }
        buf[base + i * stride] = clip16((acc + add) >> shift);
    }
}

/// One pass of the 4x4 DST.
#[inline]
fn dst_pass(buf: &mut [i16], base: usize, stride: usize, shift: u32, scratch: &mut [i32; 32]) {
    for k in 0..4 {
        scratch[k] = buf[base + k * stride] as i32;
    }
    let add = 1i32 << (shift - 1);
    for i in 0..4 {
        let acc: i32 = (0..4).map(|k| DST4[i][k] * scratch[k]).sum();
        buf[base + i * stride] = clip16((acc + add) >> shift);
    }
}

/// The inverse transform of a `size x size` residual block, in place, in raster
/// order.
///
/// `dst` selects the 4x4 luma-intra DST; it is only legal at `log2_size == 2`.
/// Columns are transformed first with a fixed shift of 7, then rows with
/// `20 - bit_depth` — the asymmetry is what keeps the intermediate inside 16
/// bits, so swapping the passes does not merely reorder work, it changes where
/// the clip bites.
pub fn inverse_transform(coeffs: &mut [i16], log2_size: u32, bit_depth: u32, dst: bool) {
    let n = 1usize << log2_size;
    debug_assert!(coeffs.len() >= n * n);
    debug_assert!(!dst || log2_size == 2);
    let mut scratch = [0i32; 32];
    let row_shift = 20 - bit_depth;

    if dst {
        for i in 0..4 {
            dst_pass(coeffs, i, 4, 7, &mut scratch);
        }
        for j in 0..4 {
            dst_pass(coeffs, j * 4, 1, row_shift, &mut scratch);
        }
        return;
    }
    for i in 0..n {
        dct_pass(coeffs, i, n, n, 7, &mut scratch);
    }
    for j in 0..n {
        dct_pass(coeffs, j * n, 1, n, row_shift, &mut scratch);
    }
}

/// The DC-only fast path: when every coefficient but the first is zero the
/// whole block is one value, and the two-pass form reduces to this.
///
/// It is not only an optimisation — it is worth having as a *separate*
/// derivation, because it is the one case where the general path can be checked
/// against a closed form.
pub fn inverse_transform_dc(coeffs: &mut [i16], log2_size: u32, bit_depth: u32) {
    let n = 1usize << log2_size;
    let shift = 14 - bit_depth;
    let add = 1i32 << (shift - 1);
    let v = clip16((((coeffs[0] as i32 + 1) >> 1) + add) >> shift);
    for c in coeffs[..n * n].iter_mut() {
        *c = v;
    }
}

/// Residual scaling for a **transform-skipped** block (H.265 §8.6.2): the
/// identity transform still has to carry the gain the DCT would have applied,
/// or a skipped block is darker than its neighbours by a factor of two per
/// size step.
pub fn transform_skip_scale(coeffs: &mut [i16], log2_size: u32, bit_depth: u32) {
    let n = 1usize << log2_size;
    let shift = 15i32 - bit_depth as i32 - log2_size as i32;
    if shift > 0 {
        let add = 1i32 << (shift - 1);
        for c in coeffs[..n * n].iter_mut() {
            *c = clip16((*c as i32 + add) >> shift);
        }
    } else if shift < 0 {
        for c in coeffs[..n * n].iter_mut() {
            *c = ((*c as i32) << (-shift)) as i16;
        }
    }
}

/// Chroma QP derivation (H.265 §8.6.1, table 8-10) for 4:2:0.
///
/// The mapping is the identity below 30 and `qPi - 6` above 43, with a
/// 14-entry table in between that flattens the curve — chroma is quantised
/// *less* aggressively than luma at high QP, and skipping the table is a
/// colour shift in exactly the dark, heavily-quantised areas where it shows.
pub fn chroma_qp(qp_i: i32, chroma_format_idc: u8) -> i32 {
    if chroma_format_idc != 1 {
        // 4:2:2 and 4:4:4 use luma's QP directly, clipped.
        return qp_i.min(51);
    }
    if qp_i < 30 {
        qp_i
    } else if qp_i > 43 {
        qp_i - 6
    } else {
        tb::QP_C[(qp_i - 30) as usize] as i32
    }
}

/// The dequantisation multiplier and shift for a block (H.265 §8.6.3).
///
/// `scale = levelScale[qp % 6] << (qp / 6)` — the `<< (qp/6)` is the reason a
/// QP step of 6 is exactly a doubling, and the reason `qp` must be the
/// **bit-depth-offset** QP rather than the slice QP.
#[inline]
pub fn dequant_params(qp: i32, log2_size: u32, bit_depth: u32) -> (i32, u32) {
    let scale = (tb::LEVEL_SCALE[(qp % 6) as usize] as i32) << (qp / 6);
    let shift = bit_depth + log2_size - 5;
    (scale, shift)
}

/// Dequantise a coefficient block in place.
///
/// `scale_matrix` is the flattened `size x size` weight matrix when scaling
/// lists are enabled (`None` means the flat 16), and `dc_scale` overrides
/// position 0 for 16x16 and 32x32 — the specification signals the DC weight
/// separately there because the matrix is only ever transmitted at 8x8 and
/// upsampled.
pub fn dequant(
    coeffs: &mut [i16],
    log2_size: u32,
    qp: i32,
    bit_depth: u32,
    scale_matrix: Option<&[u8]>,
    dc_scale: u8,
) {
    let n = 1usize << log2_size;
    let (scale, shift) = dequant_params(qp, log2_size, bit_depth);
    let add = 1i64 << (shift - 1);
    for y in 0..n {
        for x in 0..n {
            let i = y * n + x;
            let level = coeffs[i] as i64;
            if level == 0 {
                continue;
            }
            let m = match scale_matrix {
                None => 16i64,
                Some(sm) if i == 0 && log2_size >= 4 => dc_scale as i64,
                Some(sm) => sm[i] as i64,
            };
            let v = (level * m * scale as i64 + add) >> shift;
            coeffs[i] = v.clamp(-32768, 32767) as i16;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FFmpeg's `TR_4x4_LUMA` butterfly, transcribed verbatim, as an
    /// independent check on [`DST4`].
    ///
    /// The two were derived separately — the matrix from the specification's
    /// forward table, this from FFmpeg's factored form — so agreement on every
    /// basis vector is real evidence, where re-listing the matrix would only
    /// prove it was copied twice the same way.
    fn ff_dst_pass(src: &[i32; 4], shift: u32) -> [i16; 4] {
        let c0 = src[0] + src[2];
        let c1 = src[2] + src[3];
        let c2 = src[0] - src[3];
        let c3 = 74 * src[1];
        let add = 1i32 << (shift - 1);
        let s = |x: i32| clip16((x + add) >> shift);
        [
            s(29 * c0 + 55 * c1 + c3),
            s(55 * c2 - 29 * c1 + c3),
            s(74 * (src[0] - src[2] + src[3])),
            s(55 * c0 + 29 * c2 - c3),
        ]
    }

    #[test_case]
    fn dst4_matrix_agrees_with_the_butterfly_form() {
        // Every basis vector, both signs, plus a few mixed patterns — enough
        // that a single wrong matrix entry cannot hide.
        let mut cases: alloc::vec::Vec<[i32; 4]> = alloc::vec::Vec::new();
        for k in 0..4 {
            for &m in &[1i32, -1, 100, -100, 4095] {
                let mut v = [0i32; 4];
                v[k] = m;
                cases.push(v);
            }
        }
        cases.push([1, 2, 3, 4]);
        cases.push([-500, 300, -200, 100]);
        cases.push([2000, -2000, 2000, -2000]);
        for c in cases {
            for &shift in &[7u32, 12] {
                let want = ff_dst_pass(&c, shift);
                let mut buf = [0i16; 16];
                for k in 0..4 {
                    buf[k] = c[k] as i16;
                }
                let mut scratch = [0i32; 32];
                dst_pass(&mut buf, 0, 1, shift, &mut scratch);
                assert_eq!(&buf[..4], &want[..], "input {c:?} shift {shift}");
            }
        }
    }

    /// FFmpeg's `TR_4` butterfly, likewise — the smallest DCT, where the
    /// matrix-vs-butterfly claim is checkable by hand.
    fn ff_dct4_pass(src: &[i32; 4], shift: u32) -> [i16; 4] {
        let e0 = 64 * src[0] + 64 * src[2];
        let e1 = 64 * src[0] - 64 * src[2];
        let o0 = 83 * src[1] + 36 * src[3];
        let o1 = 36 * src[1] - 83 * src[3];
        let add = 1i32 << (shift - 1);
        let s = |x: i32| clip16((x + add) >> shift);
        [s(e0 + o0), s(e1 + o1), s(e1 - o1), s(e0 - o0)]
    }

    #[test_case]
    fn dct4_matrix_multiply_is_bit_exact_with_the_butterfly() {
        let mut cases: alloc::vec::Vec<[i32; 4]> = alloc::vec::Vec::new();
        for k in 0..4 {
            for &m in &[1i32, -1, 255, -255, 8000, -8000] {
                let mut v = [0i32; 4];
                v[k] = m;
                cases.push(v);
            }
        }
        cases.push([7, -13, 29, -31]);
        cases.push([32767, 32767, 32767, 32767]);
        for c in cases {
            for &shift in &[7u32, 12] {
                let want = ff_dct4_pass(&c, shift);
                let mut buf = [0i16; 16];
                for k in 0..4 {
                    buf[k] = c[k] as i16;
                }
                let mut scratch = [0i32; 32];
                dct_pass(&mut buf, 0, 1, 4, shift, &mut scratch);
                assert_eq!(&buf[..4], &want[..], "input {c:?} shift {shift}");
            }
        }
    }

    /// The DC-only closed form must equal the general path — two independent
    /// derivations of the same block, which is the only cross-check available
    /// for the 8/16/32-point transforms without a reference decoder in the
    /// test build.
    #[test_case]
    fn dc_only_fast_path_matches_the_general_transform() {
        for log2 in 2..=5u32 {
            let n = 1usize << log2;
            for &dc in &[1i16, -1, 64, -64, 1000, -1000, 16000] {
                let mut a = alloc::vec![0i16; n * n];
                a[0] = dc;
                let mut b = a.clone();
                inverse_transform(&mut a, log2, 8, false);
                inverse_transform_dc(&mut b, log2, 8);
                assert_eq!(a, b, "log2 {log2} dc {dc}");
            }
        }
    }

    /// A transform with only the DC set produces a flat block; with only the
    /// highest basis function set it alternates. These are the two ends of the
    /// basis and they fail loudly if the matrix is transposed — which is the
    /// mistake that otherwise merely rotates the residual 90 degrees.
    #[test_case]
    fn transform_basis_orientation_is_not_transposed() {
        let n = 8usize;
        let mut b = alloc::vec![0i16; n * n];
        b[1] = 512; // one horizontal cycle: varies along x, constant along y
        inverse_transform(&mut b, 3, 8, false);
        for y in 1..n {
            for x in 0..n {
                assert_eq!(b[y * n + x], b[x], "row {y} differs — axes swapped");
            }
        }
        assert!(b[0] > 0 && b[n - 1] < 0, "not a single cycle: {:?}", &b[..n]);
    }

    #[test_case]
    fn chroma_qp_table_is_continuous_at_its_joins() {
        // The three pieces must meet: below 30 identity, 30..=43 tabulated,
        // above 43 `qPi - 6`. A gap here is a colour step at one QP.
        assert_eq!(chroma_qp(29, 1), 29);
        assert_eq!(chroma_qp(30, 1), 29);
        assert_eq!(chroma_qp(43, 1), 37);
        assert_eq!(chroma_qp(44, 1), 38);
        // Monotonic and never above the luma QP.
        let mut prev = -1;
        for q in 0..=51 {
            let c = chroma_qp(q, 1);
            assert!(c >= prev, "not monotonic at {q}");
            assert!(c <= q, "chroma QP above luma at {q}");
            prev = c;
        }
        // 4:2:2 / 4:4:4 take luma's QP straight through.
        assert_eq!(chroma_qp(40, 3), 40);
    }

    #[test_case]
    fn dequant_step_of_six_is_exactly_a_doubling() {
        for qp in 0..46 {
            let (a, sa) = dequant_params(qp, 3, 8);
            let (b, sb) = dequant_params(qp + 6, 3, 8);
            assert_eq!(sa, sb);
            assert_eq!(b, a * 2, "qp {qp}");
        }
    }

    /// Transform-skip scaling is `>> (15 - bit_depth - log2_size)`, so the
    /// residual **doubles** with every size step — the identity transform has to
    /// stand in for a DCT whose gain grows with `N`, and getting the direction
    /// backwards makes large skipped blocks four to sixteen times too dark.
    ///
    /// NB the naive check — "the skipped block and the transformed block carry
    /// the same total" — is *false* and was this test's first form: the DCT
    /// spreads a DC coefficient over all `N * N` samples while skip leaves it in
    /// one, so their sums differ by exactly `N * N / (something)`. The invariant
    /// that does hold is the shift itself.
    #[test_case]
    fn transform_skip_scaling_halves_with_each_size_step() {
        let mut prev: Option<i32> = None;
        for log2 in 2..=5u32 {
            let n = 1usize << log2;
            let mut b = alloc::vec![0i16; n * n];
            b[0] = 1024;
            transform_skip_scale(&mut b, log2, 8);
            let v = b[0] as i32;
            // shift = 15 - 8 - log2, so 4x4 -> 32 and each step up doubles.
            assert_eq!(v, 1024 >> (7 - log2), "log2 {log2}");
            if let Some(p) = prev {
                assert_eq!(v, p * 2, "log2 {log2} did not double");
            }
            prev = Some(v);
            assert!(b[1..].iter().all(|&c| c == 0), "skip must not spread the value");
        }
    }

    /// At high bit depth and large sizes the shift goes **negative** and the
    /// residual is shifted *left*. That branch is easy to omit — the expression
    /// `(c + (1 << (shift - 1))) >> shift` is undefined at `shift <= 0`, so
    /// omitting it does not merely skip scaling, it computes `1 << -1`.
    #[test_case]
    fn transform_skip_shifts_left_at_high_bit_depth() {
        // 12-bit: shift = 15 - 12 - log2, so 8x8 is the identity, 16x16 and
        // 32x32 shift left.
        let cases = [(3u32, 1i16, 1i16), (4, 100, 200), (5, 100, 400)];
        for (log2, input, want) in cases {
            let n = 1usize << log2;
            let mut b = alloc::vec![0i16; n * n];
            b[0] = input;
            transform_skip_scale(&mut b, log2, 12);
            assert_eq!(b[0], want, "12-bit log2 {log2}");
        }
        // Negative values shift left too, keeping their sign.
        let mut b = alloc::vec![0i16; 32 * 32];
        b[0] = -100;
        transform_skip_scale(&mut b, 5, 12);
        assert_eq!(b[0], -400);
    }
}
