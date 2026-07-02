//! The numeric core of Cortex: dequantization + the handful of transformer
//! kernels (`CHITTI_OS_HANDOFF.md` Phase 3). Everything accumulates in
//! `f32`. The hot paths (dot product, `Q8_0` matvec) are **architecture-
//! portable**: SSE2/AVX2 on x86_64, NEON on aarch64 (native on Apple Silicon),
//! and a pure-scalar fallback anywhere else -- all behind one API and matching
//! the NumPy reference (`tools/ref.py`) within the parity tolerance. The rest
//! is straight-line scalar `f32` for legibility.
//!
//! Quantized weights follow llama.cpp's GGUF block formats verbatim so a
//! real Qwen2.5 GGUF can be read without transcoding:
//! - **Q8_0**: 32 values per block = one `f16` scale `d` + 32 `i8` quants
//!   (34 bytes). Dequant: `x[i] = d * q[i]`.
//! - **Q4_0**: 32 values per block = one `f16` scale `d` + 16 packed bytes
//!   (18 bytes). Nibble `j` low → `x[j]`, nibble `j` high → `x[j+16]`,
//!   each dequantized as `d * (nibble - 8)`.

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
    dot_f32_dispatch(a, b)
}

/// Pure-scalar dot, reducing in the same 4-lane grouping the SIMD paths use so
/// results agree closely. The portable fallback (used on non-SIMD targets;
/// unused on x86_64/aarch64, which have SSE/NEON paths).
#[allow(dead_code)]
fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut lanes = [0.0f32; 4];
    let mut i = 0;
    while i + 4 <= n {
        for (k, lane) in lanes.iter_mut().enumerate() {
            *lane += a[i + k] * b[i + k];
        }
        i += 4;
    }
    let mut sum = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
fn dot_f32_dispatch(a: &[f32], b: &[f32]) -> f32 {
    if crate::arch::x86_64::fpu::avx2_enabled() {
        // SAFETY: `avx2_enabled()` is only true once fpu::init confirmed
        // AVX2+FMA are supported by the CPU and enabled via XCR0.
        unsafe { dot_f32_avx2(a, b) }
    } else {
        dot_f32_sse2(a, b)
    }
}

#[cfg(target_arch = "aarch64")]
fn dot_f32_dispatch(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: NEON is baseline on aarch64 (see targets/aarch64-chitti.json).
    unsafe { dot_f32_neon(a, b) }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn dot_f32_dispatch(a: &[f32], b: &[f32]) -> f32 {
    dot_f32_scalar(a, b)
}

#[cfg(target_arch = "x86_64")]
fn dot_f32_sse2(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::{_mm_add_ps, _mm_loadu_ps, _mm_mul_ps, _mm_setzero_ps, _mm_storeu_ps};
    let n = a.len();
    let mut i = 0;
    // SAFETY: `+sse2` is always available (targets/x86_64-chitti.json); every
    // load reads 4 in-bounds `f32`s (`i + 4 <= n`); the store targets a local.
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
/// Requires AVX2 + FMA at runtime, guaranteed by `fpu::avx2_enabled()`.
#[cfg(target_arch = "x86_64")]
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

/// NEON dot product (4-wide FMA). Native on Apple Silicon.
///
/// # Safety
/// NEON is baseline on aarch64; all loads are guarded by `i + 4 <= n`.
#[cfg(target_arch = "aarch64")]
unsafe fn dot_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::aarch64::{vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};
    let n = a.len();
    let mut i = 0;
    // SAFETY: guarded by `i + 4 <= n`.
    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        while i + 4 <= n {
            acc = vfmaq_f32(acc, vld1q_f32(a.as_ptr().add(i)), vld1q_f32(b.as_ptr().add(i)));
            i += 4;
        }
        let mut sum = vaddvq_f32(acc);
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
/// `n_cols` must be a multiple of `QK` (true for every Qwen tensor
/// dimension). This is the single hottest kernel (the vocab-sized output
/// projection alone is ~254M MACs/token), so on AVX2 it uses a *fused*
/// path -- SIMD int8→f32 dequant (`vpmovsxbd`), scale, and FMA straight
/// into a row accumulator, with no per-block scratch buffer or dispatch.
pub fn matvec_q8_0(w: &[u8], x: &[f32], y: &mut [f32], n_rows: usize, n_cols: usize) {
    debug_assert_eq!(n_cols % QK, 0);
    debug_assert_eq!(x.len(), n_cols);
    debug_assert_eq!(y.len(), n_rows);
    // SAFETY: the slices are valid and correctly sized (asserts above); the
    // range covers every row.
    //
    // Note: this splits cleanly across cores by row range (see
    // `matvec_q8_0_rows`), which is a real speedup on native multi-core x86 but
    // a net loss under QEMU's cross-arch TCG (measured): `thread=multi` taxes
    // every emulated instruction and idle worker cores contend for host CPU,
    // so inference stays single-core here.
    unsafe { matvec_q8_0_rows(w.as_ptr(), x.as_ptr(), y.as_mut_ptr(), 0, n_rows, n_cols) };
}

/// Drop-in `Q8_0` matvec that takes the fastest available path for the target,
/// using caller-provided scratch (`xq`/`xs`, sized `>= n_cols` / `>= n_cols/QK`)
/// to avoid per-call allocation:
/// - **aarch64**: quantize the activation `x` to `int8` once, then run the
///   `SDOT` integer-dot kernel (~2.2x the f32 path on Apple Silicon, ~0.4% RMS
///   error -- validated to preserve reference token parity).
/// - **elsewhere** (x86_64 under TCG, scalar targets): the exact f32 path;
///   `xq`/`xs` are ignored. Same API on every arch (the dual-arch rule): the
///   *implementation* is arch-specific, the *behaviour* -- a correct matvec --
///   is not.
pub fn matvec_q8_0_fast(w: &[u8], x: &[f32], y: &mut [f32], xq: &mut [i8], xs: &mut [f32], n_rows: usize, n_cols: usize) {
    debug_assert_eq!(n_cols % QK, 0);
    debug_assert_eq!(x.len(), n_cols);
    debug_assert_eq!(y.len(), n_rows);
    debug_assert!(xq.len() >= n_cols && xs.len() >= n_cols / QK);
    #[cfg(target_arch = "aarch64")]
    {
        let xq = &mut xq[..n_cols];
        let xs = &mut xs[..n_cols / QK];
        quantize_activations_q8(x, xq, xs);
        // SAFETY: `w` holds `n_rows` rows of `n_cols/QK` Q8_0 blocks; `xq`/`xs`
        // are the just-computed quantized activation and its scales; `y` has
        // `n_rows` slots; the range `[0, n_rows)` is in bounds.
        unsafe { matvec_q8_0_sdot_rows(w.as_ptr(), xq.as_ptr(), xs.as_ptr(), y.as_mut_ptr(), 0, n_rows, n_cols) };
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (&mut *xq, &mut *xs);
        matvec_q8_0(w, x, y, n_rows, n_cols);
    }
}

/// Compute rows `[row_start, row_end)` of a `Q8_0` matvec, dispatching to the
/// fused AVX2 kernel or a scalar fallback. Raw pointers (not slices) and an
/// explicit row range so a matvec can be split across cores (each core owns a
/// disjoint range) -- the substrate for data-parallel inference on real
/// hardware.
///
/// # Safety
/// `w` must point at `n_rows` weight rows of `n_cols/QK` `Q8_0` blocks each,
/// `x` at `n_cols` `f32`s, and `y` at `n_rows` `f32`s; `row_start <= row_end
/// <= n_rows`; `n_cols` a multiple of `QK`. Rows are written disjointly, so
/// distinct callers may safely own distinct ranges of the same `y`.
pub unsafe fn matvec_q8_0_rows(
    w: *const u8,
    x: *const f32,
    y: *mut f32,
    row_start: usize,
    row_end: usize,
    n_cols: usize,
) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: caller guarantees bounds; avx2 path gated on runtime support.
    unsafe {
        if crate::arch::x86_64::fpu::avx2_enabled() {
            matvec_q8_0_avx2(w, x, y, row_start, row_end, n_cols);
        } else {
            matvec_q8_0_scalar(w, x, y, row_start, row_end, n_cols);
        }
    }
    #[cfg(target_arch = "aarch64")]
    // SAFETY: caller guarantees bounds; NEON is baseline on aarch64.
    unsafe {
        matvec_q8_0_neon(w, x, y, row_start, row_end, n_cols);
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    // SAFETY: caller guarantees bounds.
    unsafe {
        matvec_q8_0_scalar(w, x, y, row_start, row_end, n_cols);
    }
}

/// Fused NEON `Q8_0` matvec over rows `[row_start, row_end)`. Native on Apple
/// Silicon, and the single hottest kernel, so it is written for maximum
/// instruction-level parallelism:
///
/// - **Scale folded to one FMA per block.** Rather than scaling every widened
///   `i8` group by the block's `f16` `d` (a `vmulq` per group), the block's
///   32 products accumulate *unscaled* into partials, then the scale is applied
///   once via `acc += partial * d` -- the same math (`d·Σqx`), far fewer muls.
/// - **Four independent accumulator chains** (`acc0..acc3`), each fed once per
///   block, so consecutive blocks don't serialize on one FMA's latency; and
///   four *block* partials (`p0..p3`), each fed twice per block, so the two
///   16-lane halves of a block are independent too. On Firestorm (4 FP/SIMD
///   pipes, ~4-cycle FMA latency) this keeps enough FMAs in flight to approach
///   peak throughput instead of stalling on the dependency chain.
/// - **16 `i8` widened per load** (`vld1q_s8`) to amortize the load and the
///   `i8→i16→i32→f32` widening across four `f32x4` groups.
///
/// # Safety
/// See `matvec_q8_0_rows`; NEON is baseline on aarch64. `n_cols` is a multiple
/// of `QK` (32), so each block splits into exactly two 16-lane halves.
#[cfg(target_arch = "aarch64")]
unsafe fn matvec_q8_0_neon(w: *const u8, x: *const f32, y: *mut f32, row_start: usize, row_end: usize, n_cols: usize) {
    use core::arch::aarch64::{
        vaddq_f32, vaddvq_f32, vcvtq_f32_s32, vdupq_n_f32, vfmaq_f32, vget_high_s16, vget_high_s8, vget_low_s16,
        vget_low_s8, vld1q_f32, vld1q_s8, vmovl_s16, vmovl_s8,
    };
    let blocks = n_cols / QK;
    let row_bytes = blocks * Q8_0_BLOCK_BYTES;

    // Widen 16 signed `i8` (one `int8x16_t`) to four `float32x4_t` groups
    // covering lanes [0,4,8,12), and FMA each against the matching `x` window
    // into the four block partials `p`. A macro (not a closure) so the four
    // partials stay in registers across invocations.
    macro_rules! accumulate16 {
        ($p:expr, $q16:expr, $xp:expr) => {{
            let v = $q16;
            let lo = vmovl_s8(vget_low_s8(v)); // i16 x8, lanes 0..8
            let hi = vmovl_s8(vget_high_s8(v)); // i16 x8, lanes 8..16
            let f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo)));
            let f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo)));
            let f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi)));
            let f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi)));
            $p[0] = vfmaq_f32($p[0], f0, vld1q_f32($xp));
            $p[1] = vfmaq_f32($p[1], f1, vld1q_f32(($xp).add(4)));
            $p[2] = vfmaq_f32($p[2], f2, vld1q_f32(($xp).add(8)));
            $p[3] = vfmaq_f32($p[3], f3, vld1q_f32(($xp).add(12)));
        }};
    }

    // SAFETY: all loads in-bounds per the caller's contract.
    unsafe {
        for r in row_start..row_end {
            let row = w.add(r * row_bytes);
            let mut acc = [vdupq_n_f32(0.0); 4]; // cross-block accumulators
            for b in 0..blocks {
                let base = b * Q8_0_BLOCK_BYTES;
                let d = f16_to_f32(u16::from_le_bytes([*row.add(base), *row.add(base + 1)]));
                let dv = vdupq_n_f32(d);
                let q = row.add(base + 2) as *const i8;
                let xp = x.add(b * QK);
                let mut p = [vdupq_n_f32(0.0); 4]; // within-block partials (unscaled)
                accumulate16!(p, vld1q_s8(q), xp); // lanes 0..16
                accumulate16!(p, vld1q_s8(q.add(16)), xp.add(16)); // lanes 16..32
                // Fold the block scale in once, into four independent chains.
                acc[0] = vfmaq_f32(acc[0], p[0], dv);
                acc[1] = vfmaq_f32(acc[1], p[1], dv);
                acc[2] = vfmaq_f32(acc[2], p[2], dv);
                acc[3] = vfmaq_f32(acc[3], p[3], dv);
            }
            *y.add(r) = vaddvq_f32(vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3])));
        }
    }
}

/// Quantize an activation vector `x` to per-`QK`-block symmetric `int8`
/// (the "Q8_0 activation" llama.cpp uses to feed the integer dot kernels):
/// for each block, `scale = max|x| / 127` and `xq[i] = round(x[i]/scale)`.
/// Writes `xq` (`n_cols` `i8`) and `xs` (`n_cols/QK` `f32` block scales).
/// Cheap -- `O(n_cols)`, done once per matvec, not once per row.
pub fn quantize_activations_q8(x: &[f32], xq: &mut [i8], xs: &mut [f32]) {
    let blocks = x.len() / QK;
    debug_assert_eq!(xq.len(), x.len());
    debug_assert_eq!(xs.len(), blocks);
    for b in 0..blocks {
        let xb = &x[b * QK..(b + 1) * QK];
        let mut amax = 0.0f32;
        for &v in xb {
            let a = if v < 0.0 { -v } else { v };
            if a > amax {
                amax = a;
            }
        }
        let scale = amax / 127.0;
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        xs[b] = scale;
        for i in 0..QK {
            let r = xb[i] * inv;
            let ri = if r >= 0.0 { (r + 0.5) as i32 } else { (r - 0.5) as i32 };
            xq[b * QK + i] = ri.clamp(-127, 127) as i8;
        }
    }
}

/// **Experimental** integer-dot `Q8_0` matvec over rows `[row_start, row_end)`,
/// with the activation pre-quantized to `int8` (`quantize_activations_q8`).
/// Uses the ARMv8.2 `SDOT` instruction (`vdotq_s32`): 16 `int8`x`int8` products
/// summed into `int32` lanes *per instruction, with no widening at all* -- the
/// widening `i8→i16→i32→f32` chain that bottlenecks the f32-activation kernel
/// disappears. Per block the `int32` dot is reduced and scaled once by
/// `d_weight · d_activation`. This trades a little accuracy (the activation is
/// now `int8`, not `f32`) for a large throughput win; it is measured against
/// the f32 path via the `bench` builtin before being adopted anywhere that
/// must match the reference.
///
/// # Safety
/// `w` points at the `Q8_0` rows, `xq`/`xs` at the quantized activation and its
/// per-block scales (`n_cols` `i8` / `n_cols/QK` `f32`), `y` at `n_rows` `f32`;
/// `row_start <= row_end`; `n_cols` a multiple of `QK`. Requires `dotprod`
/// (in `targets/aarch64-chitti.json`; baseline on Apple Silicon).
#[cfg(target_arch = "aarch64")]
pub unsafe fn matvec_q8_0_sdot_rows(
    w: *const u8,
    xq: *const i8,
    xs: *const f32,
    y: *mut f32,
    row_start: usize,
    row_end: usize,
    n_cols: usize,
) {
    use core::arch::aarch64::{vaddvq_s32, vdotq_s32, vdupq_n_s32, vld1q_s8};
    let blocks = n_cols / QK;
    let row_bytes = blocks * Q8_0_BLOCK_BYTES;
    // SAFETY: all loads in-bounds per the caller's contract; dotprod is enabled.
    unsafe {
        for r in row_start..row_end {
            let row = w.add(r * row_bytes);
            let mut acc = 0.0f32;
            for b in 0..blocks {
                let base = b * Q8_0_BLOCK_BYTES;
                let dw = f16_to_f32(u16::from_le_bytes([*row.add(base), *row.add(base + 1)]));
                let dx = *xs.add(b);
                let q = row.add(base + 2) as *const i8;
                let xp = xq.add(b * QK);
                let mut iacc = vdupq_n_s32(0);
                iacc = vdotq_s32(iacc, vld1q_s8(q), vld1q_s8(xp)); // lanes 0..16
                iacc = vdotq_s32(iacc, vld1q_s8(q.add(16)), vld1q_s8(xp.add(16))); // 16..32
                acc += (vaddvq_s32(iacc) as f32) * dw * dx;
            }
            *y.add(r) = acc;
        }
    }
}

/// # Safety
/// See `matvec_q8_0_rows`.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))] // aarch64 uses the NEON path
unsafe fn matvec_q8_0_scalar(w: *const u8, x: *const f32, y: *mut f32, row_start: usize, row_end: usize, n_cols: usize) {
    let blocks_per_row = n_cols / QK;
    let row_bytes = blocks_per_row * Q8_0_BLOCK_BYTES;
    let mut buf = [0.0f32; QK];
    // SAFETY: caller guarantees `w`/`x`/`y` bounds and the row range.
    unsafe {
        for r in row_start..row_end {
            let row = core::slice::from_raw_parts(w.add(r * row_bytes), row_bytes);
            let mut acc = 0.0f32;
            for b in 0..blocks_per_row {
                let block = &row[b * Q8_0_BLOCK_BYTES..(b + 1) * Q8_0_BLOCK_BYTES];
                dequant_q8_0_block(block, &mut buf);
                let xb = core::slice::from_raw_parts(x.add(b * QK), QK);
                acc += dot_f32(&buf, xb);
            }
            *y.add(r) = acc;
        }
    }
}

/// Fused AVX2+FMA `Q8_0` matvec over rows `[row_start, row_end)`. Per block:
/// load 32 `i8` quants, widen to `f32` eight at a time with
/// `vpmovsxbd`+`vcvtdq2ps`, scale by the block's `f16` `d`, and FMA against
/// `x` into a single 8-lane row accumulator (one horizontal sum per row).
///
/// # Safety
/// Requires AVX2 + FMA; see `matvec_q8_0_rows` for the pointer/range contract.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn matvec_q8_0_avx2(w: *const u8, x: *const f32, y: *mut f32, row_start: usize, row_end: usize, n_cols: usize) {
    use core::arch::x86_64::{
        __m128i, _mm256_cvtepi32_ps, _mm256_cvtepi8_epi32, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_mul_ps,
        _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps, _mm_loadl_epi64,
    };
    let blocks = n_cols / QK;
    let row_bytes = blocks * Q8_0_BLOCK_BYTES;
    // SAFETY: all loads are in-bounds (caller's contract) and AVX2/FMA is on.
    unsafe {
        for r in row_start..row_end {
            let row = w.add(r * row_bytes);
            let mut acc = _mm256_setzero_ps();
            for b in 0..blocks {
                let base = b * Q8_0_BLOCK_BYTES;
                let d = f16_to_f32(u16::from_le_bytes([*row.add(base), *row.add(base + 1)]));
                let dvec = _mm256_set1_ps(d);
                let qptr = row.add(base + 2) as *const i8;
                let xptr = x.add(b * QK);
                let mut g = 0;
                while g < QK {
                    let q8 = _mm_loadl_epi64(qptr.add(g) as *const __m128i); // 8 i8
                    let qf = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8)), dvec);
                    let xf = _mm256_loadu_ps(xptr.add(g));
                    acc = _mm256_fmadd_ps(qf, xf, acc);
                    g += 8;
                }
            }
            let mut lanes = [0.0f32; 8];
            _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
            *y.add(r) = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
        }
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

pub fn libm_sqrtf(x: f32) -> f32 {
    // Hardware square root: `sqrtss` on x86 (SSE2), `fsqrt` on aarch64. Both
    // are exact/IEEE-correct and defined for our non-negative inputs (sums of
    // squares plus a positive eps). A Newton-Raphson fallback covers any other
    // target. `core` has no `f32::sqrt` (that lives in `std`).
    #[cfg(target_arch = "x86_64")]
    // SAFETY: `sqrtss` has no side effects; input is non-negative.
    unsafe {
        let mut r = x;
        core::arch::asm!("sqrtss {r}, {r}", r = inout(xmm_reg) r, options(nomem, nostack, preserves_flags));
        r
    }
    #[cfg(target_arch = "aarch64")]
    // SAFETY: `fsqrt` has no side effects; input is non-negative.
    unsafe {
        let r: f32;
        core::arch::asm!("fsqrt {r:s}, {x:s}", r = out(vreg) r, x = in(vreg) x, options(nomem, nostack, preserves_flags));
        r
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        if x <= 0.0 {
            return 0.0;
        }
        let mut g = x; // Newton-Raphson: g' = (g + x/g) / 2
        for _ in 0..20 {
            g = 0.5 * (g + x / g);
        }
        g
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
