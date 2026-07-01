//! The numeric core of Cortex: dequantization + the handful of transformer
//! kernels (`CHITTI_OS_HANDOFF.md` Phase 3). Everything accumulates in
//! `f32`; the dot-product hot path uses explicit SSE2 intrinsics
//! (`core::arch::x86_64`), the rest is straight-line scalar `f32` chosen
//! for legibility and to match the NumPy reference (`tools/ref.py`) bit
//! layout and math exactly enough to pass the parity gate.
//!
//! Quantized weights follow llama.cpp's GGUF block formats verbatim so a
//! real Qwen2.5 GGUF can be read without transcoding:
//! - **Q8_0**: 32 values per block = one `f16` scale `d` + 32 `i8` quants
//!   (34 bytes). Dequant: `x[i] = d * q[i]`.
//! - **Q4_0**: 32 values per block = one `f16` scale `d` + 16 packed bytes
//!   (18 bytes). Nibble `j` low → `x[j]`, nibble `j` high → `x[j+16]`,
//!   each dequantized as `d * (nibble - 8)`.

use core::arch::x86_64::{_mm_add_ps, _mm_loadu_ps, _mm_mul_ps, _mm_setzero_ps, _mm_storeu_ps};

pub const QK: usize = 32; // elements per quantization block (both Q4_0/Q8_0)
pub const Q8_0_BLOCK_BYTES: usize = 2 + QK; // f16 scale + 32 i8
pub const Q4_0_BLOCK_BYTES: usize = 2 + QK / 2; // f16 scale + 16 packed nibbles

/// Convert an IEEE-754 half (as raw bits) to `f32`, purely by bit
/// manipulation (no `std` transcendentals). Exact: every `f16` value is
/// representable in `f32`. Handles subnormals, inf, and NaN.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits as u32 & 0x8000) << 16; // f16 sign -> f32 sign bit
    let exp = (bits >> 10) & 0x1f;
    let mant = (bits & 0x3ff) as u32;
    match exp {
        0 => {
            if mant == 0 {
                f32::from_bits(sign) // signed zero
            } else {
                // Subnormal f16 (value = mant * 2^-24) renormalized into a
                // normal f32: let p be the index of mant's highest set bit
                // (0..=9); then value = 1.f * 2^(p-24).
                let p = 31 - mant.leading_zeros(); // 0..=9
                let f32_exp = p + 103; // (p - 24) + 127
                let frac = (mant - (1 << p)) << (23 - p);
                f32::from_bits(sign | (f32_exp << 23) | frac)
            }
        }
        0x1f => f32::from_bits(sign | 0x7f80_0000 | (mant << 13)), // inf / NaN
        _ => {
            let f32_exp = (exp as u32 + (127 - 15)) << 23; // rebias 15 -> 127
            f32::from_bits(sign | f32_exp | (mant << 13))
        }
    }
}

fn read_f16_le(bytes: &[u8]) -> f32 {
    f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]))
}

/// Dequantize one Q8_0 block (`Q8_0_BLOCK_BYTES` bytes) into 32 `f32`s.
pub fn dequant_q8_0_block(block: &[u8], out: &mut [f32]) {
    debug_assert_eq!(block.len(), Q8_0_BLOCK_BYTES);
    debug_assert_eq!(out.len(), QK);
    let d = read_f16_le(&block[0..2]);
    for i in 0..QK {
        let q = block[2 + i] as i8 as f32;
        out[i] = d * q;
    }
}

/// Dequantize one Q4_0 block (`Q4_0_BLOCK_BYTES` bytes) into 32 `f32`s,
/// using llama.cpp's split-nibble layout (low nibbles → first 16 lanes,
/// high nibbles → last 16 lanes).
pub fn dequant_q4_0_block(block: &[u8], out: &mut [f32]) {
    debug_assert_eq!(block.len(), Q4_0_BLOCK_BYTES);
    debug_assert_eq!(out.len(), QK);
    let d = read_f16_le(&block[0..2]);
    for j in 0..QK / 2 {
        let byte = block[2 + j];
        let lo = (byte & 0x0f) as i32 - 8;
        let hi = (byte >> 4) as i32 - 8;
        out[j] = d * lo as f32;
        out[j + QK / 2] = d * hi as f32;
    }
}

/// Dot product of two equal-length `f32` slices. Dispatches to an AVX2+FMA
/// path (8-wide) when the CPU/OS support it, else the SSE2 baseline (4-wide);
/// both use a fixed reduction order so results are deterministic run-to-run
/// (a Phase 3 acceptance requirement). AVX2 halves the number of guest SIMD
/// instructions and fuses multiply-add, which is a real win even under
/// QEMU's TCG emulation.
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    if crate::arch::x86_64::fpu::avx2_enabled() {
        // SAFETY: `avx2_enabled()` is only true once fpu::init has confirmed
        // AVX2+FMA are supported by the CPU and enabled via XCR0.
        unsafe { dot_f32_avx2(a, b) }
    } else {
        dot_f32_sse2(a, b)
    }
}

fn dot_f32_sse2(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut i = 0;
    // SAFETY: `+sse2` is always available (see targets/x86_64-chitti.json);
    // every load reads 4 in-bounds `f32`s (`i + 4 <= n` guard), and the
    // store targets a local 4-lane array.
    let mut sum = unsafe {
        let mut acc = _mm_setzero_ps();
        while i + 4 <= n {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            let vb = _mm_loadu_ps(b.as_ptr().add(i));
            acc = _mm_add_ps(acc, _mm_mul_ps(va, vb));
            i += 4;
        }
        let mut lanes = [0.0f32; 4];
        _mm_storeu_ps(lanes.as_mut_ptr(), acc);
        (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
    };
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

/// AVX2 + FMA dot product (8-wide). Only called when `fpu::avx2_enabled()`.
///
/// # Safety
/// Requires the AVX2 and FMA target features to be available at runtime,
/// which the caller guarantees via `fpu::avx2_enabled()`.
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::{_mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps, _mm256_storeu_ps};
    let n = a.len();
    let mut i = 0;
    // SAFETY: guarded by `i + 8 <= n`; the store targets a local 8-lane array.
    unsafe {
        let mut acc = _mm256_setzero_ps();
        while i + 8 <= n {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            acc = _mm256_fmadd_ps(va, vb, acc); // acc += va*vb, fused
            i += 8;
        }
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
        let mut sum =
            ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
        while i < n {
            sum += a[i] * b[i];
            i += 1;
        }
        sum
    }
}

/// `y[r] = sum_c w[r*n_cols + c] * x[c]` for an `f32` weight matrix stored
/// row-major (`n_rows` × `n_cols`). Used for the few `f32` tensors (norms
/// are applied elementwise, but a couple of projections may be stored
/// unquantized in some GGUFs).
pub fn matvec_f32(w: &[f32], x: &[f32], y: &mut [f32], n_rows: usize, n_cols: usize) {
    debug_assert_eq!(w.len(), n_rows * n_cols);
    debug_assert_eq!(x.len(), n_cols);
    debug_assert_eq!(y.len(), n_rows);
    for r in 0..n_rows {
        y[r] = dot_f32(&w[r * n_cols..(r + 1) * n_cols], x);
    }
}

/// `y[r] = sum_c W[r,c] * x[c]` where each weight row is `Q8_0`-quantized.
/// `n_cols` must be a multiple of `QK` (true for every Qwen2 tensor
/// dimension). Dequantizes one 32-lane block at a time into a small stack
/// buffer and dots it with the matching slice of `x`.
pub fn matvec_q8_0(w: &[u8], x: &[f32], y: &mut [f32], n_rows: usize, n_cols: usize) {
    debug_assert_eq!(n_cols % QK, 0);
    debug_assert_eq!(x.len(), n_cols);
    debug_assert_eq!(y.len(), n_rows);
    let blocks_per_row = n_cols / QK;
    let row_bytes = blocks_per_row * Q8_0_BLOCK_BYTES;
    debug_assert_eq!(w.len(), n_rows * row_bytes);
    let mut buf = [0.0f32; QK];
    for r in 0..n_rows {
        let row = &w[r * row_bytes..(r + 1) * row_bytes];
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let block = &row[b * Q8_0_BLOCK_BYTES..(b + 1) * Q8_0_BLOCK_BYTES];
            dequant_q8_0_block(block, &mut buf);
            acc += dot_f32(&buf, &x[b * QK..(b + 1) * QK]);
        }
        y[r] = acc;
    }
}

/// As `matvec_q8_0`, but for `Q4_0`-quantized weight rows.
pub fn matvec_q4_0(w: &[u8], x: &[f32], y: &mut [f32], n_rows: usize, n_cols: usize) {
    debug_assert_eq!(n_cols % QK, 0);
    debug_assert_eq!(x.len(), n_cols);
    debug_assert_eq!(y.len(), n_rows);
    let blocks_per_row = n_cols / QK;
    let row_bytes = blocks_per_row * Q4_0_BLOCK_BYTES;
    debug_assert_eq!(w.len(), n_rows * row_bytes);
    let mut buf = [0.0f32; QK];
    for r in 0..n_rows {
        let row = &w[r * row_bytes..(r + 1) * row_bytes];
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let block = &row[b * Q4_0_BLOCK_BYTES..(b + 1) * Q4_0_BLOCK_BYTES];
            dequant_q4_0_block(block, &mut buf);
            acc += dot_f32(&buf, &x[b * QK..(b + 1) * QK]);
        }
        y[r] = acc;
    }
}

/// RMSNorm: `out[i] = x[i] * rsqrt(mean(x^2) + eps) * weight[i]`
/// (Qwen2/Llama pre-norm). The `1 + weight` convention some models use is
/// *not* applied here -- Qwen2 stores the norm weight directly.
pub fn rmsnorm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    debug_assert_eq!(x.len(), weight.len());
    debug_assert_eq!(x.len(), out.len());
    let n = x.len();
    let mut ss = 0.0f32;
    for &v in x {
        ss += v * v;
    }
    let scale = 1.0 / libm_sqrtf(ss / n as f32 + eps);
    for i in 0..n {
        out[i] = x[i] * scale * weight[i];
    }
}

/// In-place RMSNorm: `x[i] = x[i] * rsqrt(mean(x^2)+eps) * weight[i]`.
pub fn rmsnorm_inplace(x: &mut [f32], weight: &[f32], eps: f32) {
    debug_assert_eq!(x.len(), weight.len());
    let n = x.len();
    let mut ss = 0.0f32;
    for &v in x.iter() {
        ss += v * v;
    }
    let scale = 1.0 / libm_sqrtf(ss / n as f32 + eps);
    for i in 0..n {
        x[i] = x[i] * scale * weight[i];
    }
}

/// Apply NeoX-style rotary position embedding in place to a single head
/// vector of length `head_dim` at sequence position `pos`. Pairs lane `i`
/// with lane `i + head_dim/2` (the `rotate_half` convention Qwen2/HF use),
/// with per-pair angle `pos / theta_base^(2i/head_dim)`.
pub fn rope(vec: &mut [f32], pos: usize, head_dim: usize, theta_base: f32) {
    debug_assert_eq!(vec.len(), head_dim);
    let half = head_dim / 2;
    for i in 0..half {
        let freq = 1.0 / powf(theta_base, (2 * i) as f32 / head_dim as f32);
        let angle = pos as f32 * freq;
        let (sin, cos) = (sinf(angle), cosf(angle));
        let a = vec[i];
        let b = vec[i + half];
        vec[i] = a * cos - b * sin;
        vec[i + half] = b * cos + a * sin;
    }
}

/// Numerically stable in-place softmax (subtract max before `exp`).
pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let mut max = x[0];
    for &v in x.iter() {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = expf(*v - max);
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// SwiGLU feed-forward activation: `out[i] = silu(gate[i]) * up[i]` where
/// `silu(v) = v * sigmoid(v) = v / (1 + e^-v)`.
pub fn silu_mul(gate: &[f32], up: &[f32], out: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len());
    debug_assert_eq!(gate.len(), out.len());
    for i in 0..gate.len() {
        out[i] = silu(gate[i]) * up[i];
    }
}

/// Logistic sigmoid `1 / (1 + e^-x)`.
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + expf(-x))
}

/// SiLU / swish `x * sigmoid(x)`.
pub fn silu(x: f32) -> f32 {
    x / (1.0 + expf(-x))
}

/// Softplus `ln(1 + e^x)`, computed stably (`max(x,0) + ln(1 + e^-|x|)`).
pub fn softplus(x: f32) -> f32 {
    let ax = if x < 0.0 { -x } else { x };
    x.max(0.0) + lnf(1.0 + expf(-ax))
}

/// L2-normalize `x` in place: `x /= sqrt(sum(x^2) + eps)` (the FLA-library
/// convention used by the Qwen3.5 gated-DeltaNet for q/k).
pub fn l2norm(x: &mut [f32], eps: f32) {
    let mut ss = 0.0f32;
    for &v in x.iter() {
        ss += v * v;
    }
    let inv = 1.0 / libm_sqrtf(ss + eps);
    for v in x.iter_mut() {
        *v *= inv;
    }
}

// --- no_std transcendental helpers --------------------------------------
//
// `f32::exp/sin/cos/powf/sqrt` are `std`-only. We need our own `no_std`
// implementations, and they must match the NumPy reference closely enough
// for the parity gate. `sqrtf` maps straight to the SSE2 hardware
// instruction (exact, IEEE-correct). `exp`/`sin`/`cos` use range-reduced
// polynomial approximations accurate to well within the logit tolerance.

fn libm_sqrtf(x: f32) -> f32 {
    // SAFETY: `sqrtss` is an SSE2 instruction (always available); it has no
    // side effects and is defined for all non-negative inputs (our sums of
    // squares plus a positive eps).
    unsafe {
        let mut r = x;
        core::arch::asm!("sqrtss {r}, {r}", r = inout(xmm_reg) r, options(nomem, nostack, preserves_flags));
        r
    }
}

const LN2: f32 = core::f32::consts::LN_2;
const PI: f32 = core::f32::consts::PI;

/// `e^x` via range reduction `x = k*ln2 + r` (`|r| <= ln2/2`) and a degree-5
/// minimax-ish polynomial on `r`, then `2^k` by direct exponent assembly.
pub fn expf(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if x > 88.0 {
        return f32::INFINITY;
    }
    if x < -88.0 {
        return 0.0;
    }
    let k = (x / LN2 + if x >= 0.0 { 0.5 } else { -0.5 }) as i32;
    let r = x - k as f32 * LN2;
    // e^r, r in [-ln2/2, ln2/2]
    let r2 = r * r;
    let p = 1.0
        + r
        + 0.5 * r2
        + r2 * r * (1.0 / 6.0)
        + r2 * r2 * (1.0 / 24.0)
        + r2 * r2 * r * (1.0 / 120.0);
    // 2^k by assembling the IEEE-754 exponent field.
    let two_k = f32::from_bits((((k + 127) as u32) & 0xff) << 23);
    p * two_k
}

fn sinf(x: f32) -> f32 {
    // Range-reduce to [-pi, pi], then to [-pi/2, pi/2] via sin(pi-t)=sin(t)
    // and sin(-pi-t)=sin(t), where the x^9 Taylor series is accurate to
    // ~2e-6 -- comfortably inside the logit tolerance.
    let mut t = x % (2.0 * PI);
    if t > PI {
        t -= 2.0 * PI;
    } else if t < -PI {
        t += 2.0 * PI;
    }
    if t > PI / 2.0 {
        t = PI - t;
    } else if t < -PI / 2.0 {
        t = -PI - t;
    }
    let t2 = t * t;
    t * (1.0 - t2 * (1.0 / 6.0 - t2 * (1.0 / 120.0 - t2 * (1.0 / 5040.0 - t2 * (1.0 / 362880.0)))))
}

fn cosf(x: f32) -> f32 {
    sinf(x + PI / 2.0)
}

/// `base^exp` via `exp(exp * ln(base))`. Only ever called with a positive
/// base (RoPE's `theta_base`), so the `ln` path is well-defined.
fn powf(base: f32, exp: f32) -> f32 {
    expf(exp * lnf(base))
}

/// Natural log via mantissa/exponent split: `ln(x) = e*ln2 + ln(m)` with
/// `m in [1, 2)`, `ln(m)` from an `atanh`-series in `s = (m-1)/(m+1)`.
fn lnf(x: f32) -> f32 {
    let bits = x.to_bits();
    let e = ((bits >> 23) & 0xff) as i32 - 127;
    let m = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000); // mantissa in [1,2)
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    let ln_m = 2.0 * s * (1.0 + s2 * (1.0 / 3.0 + s2 * (1.0 / 5.0 + s2 * (1.0 / 7.0))));
    e as f32 * LN2 + ln_m
}

// --- unit tests: each kernel vs the NumPy reference (tools/ref.py) -------
//
// These run in QEMU under `cargo xtask test` and are the Phase 3 gate that
// every kernel is correct *before* it is composed into the forward pass.
// Tolerances reflect the only legitimate source of divergence -- f32
// summation order (SSE 4-lane reduction vs NumPy pairwise) and the
// polynomial transcendentals -- and are tight enough that a genuinely
// wrong kernel (bad nibble layout, wrong RoPE pairing, etc.) fails by
// orders of magnitude.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::cortex::testdata as td;

    fn assert_slice_close(got: &[f32], want: &[f32], atol: f32, rtol: f32, name: &str) {
        assert_eq!(got.len(), want.len(), "{name}: length mismatch");
        for i in 0..got.len() {
            let diff = (got[i] - want[i]).abs();
            assert!(
                diff <= atol + rtol * want[i].abs(),
                "{name}: idx {i}: got {} want {} (|diff|={diff})",
                got[i],
                want[i]
            );
        }
    }

    #[test_case]
    fn dequant_q8_0_matches_reference() {
        let mut out = [0.0f32; QK];
        dequant_q8_0_block(&td::Q8_0_BLOCK, &mut out);
        assert_slice_close(&out, &td::Q8_0_EXPECTED, 1e-4, 1e-5, "dequant_q8_0");
    }

    #[test_case]
    fn dequant_q4_0_matches_reference() {
        let mut out = [0.0f32; QK];
        dequant_q4_0_block(&td::Q4_0_BLOCK, &mut out);
        assert_slice_close(&out, &td::Q4_0_EXPECTED, 1e-4, 1e-5, "dequant_q4_0");
    }

    #[test_case]
    fn matvec_q8_0_matches_reference() {
        let mut y = [0.0f32; td::MV_Q8_ROWS];
        matvec_q8_0(&td::MV_Q8_W, &td::MV_Q8_X, &mut y, td::MV_Q8_ROWS, td::MV_Q8_COLS);
        assert_slice_close(&y, &td::MV_Q8_Y, 1e-2, 1e-3, "matvec_q8_0");
    }

    #[test_case]
    fn matvec_q4_0_matches_reference() {
        let mut y = [0.0f32; td::MV_Q4_ROWS];
        matvec_q4_0(&td::MV_Q4_W, &td::MV_Q4_X, &mut y, td::MV_Q4_ROWS, td::MV_Q4_COLS);
        assert_slice_close(&y, &td::MV_Q4_Y, 1e-2, 1e-3, "matvec_q4_0");
    }

    #[test_case]
    fn rmsnorm_matches_reference() {
        let mut out = [0.0f32; 128];
        rmsnorm(&td::RMS_X, &td::RMS_W, td::RMS_EPS, &mut out);
        assert_slice_close(&out, &td::RMS_Y, 1e-3, 1e-3, "rmsnorm");
    }

    #[test_case]
    fn rope_matches_reference() {
        let mut v = td::ROPE_IN;
        rope(&mut v, td::ROPE_POS, td::ROPE_HEAD_DIM, td::ROPE_THETA);
        assert_slice_close(&v, &td::ROPE_OUT, 1e-3, 1e-3, "rope");
    }

    #[test_case]
    fn softmax_matches_reference() {
        let mut x = td::SOFTMAX_IN;
        softmax(&mut x);
        assert_slice_close(&x, &td::SOFTMAX_OUT, 1e-4, 1e-4, "softmax");
    }

    #[test_case]
    fn silu_mul_matches_reference() {
        let mut out = [0.0f32; 96];
        silu_mul(&td::SILU_GATE, &td::SILU_UP, &mut out);
        assert_slice_close(&out, &td::SILU_OUT, 1e-4, 1e-4, "silu_mul");
    }

    /// `dot_f32` is exercised indirectly by the matvec tests, but a direct
    /// check pins its tail handling (length not a multiple of 4).
    #[test_case]
    fn dot_f32_handles_unaligned_tail() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let b = [2.0f32, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
        let got = dot_f32(&a, &b);
        let want = 2.0 * (1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 6.0 + 7.0);
        assert!((got - want).abs() < 1e-4, "dot_f32 tail: got {got} want {want}");
    }
}
