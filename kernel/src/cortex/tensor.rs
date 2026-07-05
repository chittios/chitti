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
pub const Q4_1_BLOCK_BYTES: usize = 2 + 2 + QK / 2; // f16 d + f16 min + 16 nibbles

// k-quant super-block: 256 elements. Byte layouts verbatim from llama.cpp
// (ggml-common.h). Used by the mixed-quant Qwen3.5-9B GGUF (ssm_out=Q5_K,
// output=Q6_K); the 0.8B model uses none of these.
pub const QK_K: usize = 256;
pub const Q5_K_BLOCK_BYTES: usize = 2 + 2 + 12 + QK_K / 8 + QK_K / 2; // d,dmin,scales[12],qh[32],qs[128] = 176
pub const Q6_K_BLOCK_BYTES: usize = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2; // ql[128],qh[64],scales[16],d = 210

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

/// Dequantize one Q4_1 block (`Q4_1_BLOCK_BYTES`) into 32 `f32`s. Like Q4_0 but
/// with an affine `min`: `x = d*nibble + m` (nibbles unsigned 0..15, no -8).
pub fn dequant_q4_1_block(block: &[u8], out: &mut [f32]) {
    debug_assert_eq!(block.len(), Q4_1_BLOCK_BYTES);
    debug_assert_eq!(out.len(), QK);
    let d = read_f16_le(&block[0..2]);
    let m = read_f16_le(&block[2..4]);
    for j in 0..QK / 2 {
        let byte = block[4 + j];
        out[j] = d * (byte & 0x0f) as f32 + m;
        out[j + QK / 2] = d * (byte >> 4) as f32 + m;
    }
}

/// `get_scale_min_k4` (llama.cpp): unpack the 6-bit scale + 6-bit min for
/// sub-block `j` (0..8) from a Q4_K/Q5_K block's 12 packed scale bytes.
#[inline]
fn q_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Dequantize one Q5_K super-block (`Q5_K_BLOCK_BYTES`) into 256 `f32`s, per
/// llama.cpp `dequantize_row_q5_K`: `d`/`dmin` (f16), 8 packed sub-block
/// scale/min pairs, a 5th bit per quant in `qh`, and 4-bit `qs`.
pub fn dequant_q5_k_block(block: &[u8], out: &mut [f32]) {
    debug_assert_eq!(block.len(), Q5_K_BLOCK_BYTES);
    debug_assert_eq!(out.len(), QK_K);
    let d = read_f16_le(&block[0..2]);
    let dmin = read_f16_le(&block[2..4]);
    let scales = &block[4..16]; // 12 bytes
    let qh = &block[16..48]; // 32 bytes (1 bit / element)
    let qs = &block[48..176]; // 128 bytes (4 bits / element)
    let mut is = 0usize;
    let mut u1 = 1u8;
    let mut u2 = 2u8;
    let mut y = 0usize; // output offset (steps by 64)
    let mut ql = 0usize; // qs offset (steps by 32)
    for _ in 0..QK_K / 64 {
        let (sc1, m1s) = q_scale_min_k4(is, scales);
        let (sc2, m2s) = q_scale_min_k4(is + 1, scales);
        let d1 = d * sc1 as f32;
        let m1 = dmin * m1s as f32;
        let d2 = d * sc2 as f32;
        let m2 = dmin * m2s as f32;
        for l in 0..32 {
            let hi1 = if qh[l] & u1 != 0 { 16.0 } else { 0.0 };
            out[y + l] = d1 * ((qs[ql + l] & 0x0f) as f32 + hi1) - m1;
        }
        for l in 0..32 {
            let hi2 = if qh[l] & u2 != 0 { 16.0 } else { 0.0 };
            out[y + 32 + l] = d2 * ((qs[ql + l] >> 4) as f32 + hi2) - m2;
        }
        y += 64;
        ql += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

/// Dequantize one Q6_K super-block (`Q6_K_BLOCK_BYTES`) into 256 `f32`s, per
/// llama.cpp `dequantize_row_q6_K`: 4-bit `ql` + 2-bit `qh` = 6-bit signed
/// quants (biased by -32), scaled by 16 `i8` sub-block scales and `d` (f16).
pub fn dequant_q6_k_block(block: &[u8], out: &mut [f32]) {
    debug_assert_eq!(block.len(), Q6_K_BLOCK_BYTES);
    debug_assert_eq!(out.len(), QK_K);
    let ql_all = &block[0..128];
    let qh_all = &block[128..192];
    let sc_all = &block[192..208]; // 16 i8
    let d = read_f16_le(&block[208..210]);
    // Two 128-element halves; per half ql+=64, qh+=32, sc+=8.
    for half in 0..2 {
        let ql = &ql_all[half * 64..];
        let qh = &qh_all[half * 32..];
        let sc = &sc_all[half * 8..];
        let y = half * 128;
        for l in 0..32 {
            let is = l / 16;
            let q1 = (((ql[l] & 0x0f) | (((qh[l] >> 0) & 3) << 4)) as i32 - 32) as f32;
            let q2 = (((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32) as f32;
            let q3 = (((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32) as f32;
            let q4 = (((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32) as f32;
            out[y + l] = d * sc[is] as i8 as f32 * q1;
            out[y + l + 32] = d * sc[is + 2] as i8 as f32 * q2;
            out[y + l + 64] = d * sc[is + 4] as i8 as f32 * q3;
            out[y + l + 96] = d * sc[is + 6] as i8 as f32 * q4;
        }
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
    // SAFETY: `+sse2` is always available (targets/x86_64-chitti.json); all
    // vector loads are guarded (`i + 16 <= n` / `i + 4 <= n`); the store
    // targets a local.
    let mut sum = unsafe {
        // Four independent accumulators (see `dot_f32_neon` for the latency
        // rationale — SSE2 mul+add chains stall the same way).
        let mut acc0 = _mm_setzero_ps();
        let mut acc1 = _mm_setzero_ps();
        let mut acc2 = _mm_setzero_ps();
        let mut acc3 = _mm_setzero_ps();
        while i + 16 <= n {
            let (pa, pb) = (a.as_ptr().add(i), b.as_ptr().add(i));
            acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_loadu_ps(pa), _mm_loadu_ps(pb)));
            acc1 = _mm_add_ps(acc1, _mm_mul_ps(_mm_loadu_ps(pa.add(4)), _mm_loadu_ps(pb.add(4))));
            acc2 = _mm_add_ps(acc2, _mm_mul_ps(_mm_loadu_ps(pa.add(8)), _mm_loadu_ps(pb.add(8))));
            acc3 = _mm_add_ps(acc3, _mm_mul_ps(_mm_loadu_ps(pa.add(12)), _mm_loadu_ps(pb.add(12))));
            i += 16;
        }
        let mut acc = _mm_add_ps(_mm_add_ps(acc0, acc1), _mm_add_ps(acc2, acc3));
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
    // SAFETY: all vector loads are guarded (`i + 32 <= n` / `i + 8 <= n`); the
    // store targets a local 8-lane array.
    unsafe {
        use core::arch::x86_64::_mm256_add_ps;
        // Four independent accumulators — a single chain is FMA-latency-bound
        // (see `dot_f32_neon`); this keeps both FMA ports busy.
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        while i + 32 <= n {
            let (pa, pb) = (a.as_ptr().add(i), b.as_ptr().add(i));
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa), _mm256_loadu_ps(pb), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(8)), _mm256_loadu_ps(pb.add(8)), acc1);
            acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(16)), _mm256_loadu_ps(pb.add(16)), acc2);
            acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(24)), _mm256_loadu_ps(pb.add(24)), acc3);
            i += 32;
        }
        let mut acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
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

/// NEON dot product — the whole 16-wide hot loop is **inline asm**, not
/// intrinsics. The kernel target builds with `+strict-align` (required for the
/// pre-MMU boot window and device MMIO), and under strict-align LLVM lowers
/// `vld1q_f32` (an align-4 vector load) to a 16×`ldrb`+`orr` byte-assembly
/// with a stack round-trip — ~25 instructions per load, which made NEON ~100×
/// slower than scalar. At runtime this data is Normal cacheable memory with
/// `SCTLR_EL1.A = 0`, where unaligned `ldp q` is architecturally fine — the
/// same "inline asm to get the exact access" pattern the MMIO drivers use.
/// Four independent accumulators keep the FMA pipes saturated.
///
/// # Safety
/// NEON is baseline on aarch64; the asm loop reads exactly `n/16*16` floats
/// from each slice, and the scalar tail is bounds-checked.
#[cfg(target_arch = "aarch64")]
unsafe fn dot_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let n16 = n / 16;
    let mut sum: f32;
    // SAFETY: the loop consumes exactly `n16 * 16` f32 from both pointers
    // (guarded by `n16` computed from the slice length); v8–v15 (callee-saved)
    // are untouched; pointers advance inside the asm and are discarded.
    unsafe {
        core::arch::asm!(
            "movi v0.16b, #0",
            "movi v1.16b, #0",
            "movi v2.16b, #0",
            "movi v3.16b, #0",
            "cbz {cnt}, 2f",
            "1:",
            "ldp q4, q5, [{pa}], #32",
            "ldp q6, q7, [{pb}], #32",
            "ldp q16, q17, [{pa}], #32",
            "ldp q18, q19, [{pb}], #32",
            "fmla v0.4s, v4.4s, v6.4s",
            "fmla v1.4s, v5.4s, v7.4s",
            "fmla v2.4s, v16.4s, v18.4s",
            "fmla v3.4s, v17.4s, v19.4s",
            "subs {cnt}, {cnt}, #1",
            "b.ne 1b",
            "2:",
            "fadd v0.4s, v0.4s, v1.4s",
            "fadd v2.4s, v2.4s, v3.4s",
            "fadd v0.4s, v0.4s, v2.4s",
            "faddp v0.4s, v0.4s, v0.4s",
            "faddp s0, v0.2s",
            pa = inout(reg) a.as_ptr() => _,
            pb = inout(reg) b.as_ptr() => _,
            cnt = inout(reg) n16 => _,
            out("v0") sum,
            out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _,
            out("v6") _, out("v7") _, out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            options(nostack, readonly),
        );
    }
    let mut i = n16 * 16;
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

/// 16-byte NEON loads via inline asm. Under the kernel's `+strict-align` LLVM
/// lowers the (align-4/align-1) `vld1q_*` intrinsics to a 16×`ldrb`+`orr`
/// byte-assembly with a stack round-trip (~25 instructions per load, ~100×
/// slower — see `dot_f32_neon`). A single `ldr q` is architecturally correct
/// at runtime: this data lives in Normal cacheable RAM and `SCTLR_EL1.A = 0`,
/// where unaligned SIMD loads are supported. Same pattern as the MMIO
/// single-`ldr`/`str` accessors.
///
/// # Safety
/// `p` must point at 16 readable bytes.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn ldq_s8(p: *const i8) -> core::arch::aarch64::int8x16_t {
    let v;
    // SAFETY: caller guarantees 16 readable bytes at `p`.
    unsafe {
        core::arch::asm!("ldr {v:q}, [{p}]", v = out(vreg) v, p = in(reg) p, options(nostack, readonly, preserves_flags));
    }
    v
}

/// See [`ldq_s8`].
///
/// # Safety
/// `p` must point at 16 readable bytes.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn ldq_u8(p: *const u8) -> core::arch::aarch64::uint8x16_t {
    let v;
    // SAFETY: caller guarantees 16 readable bytes at `p`.
    unsafe {
        core::arch::asm!("ldr {v:q}, [{p}]", v = out(vreg) v, p = in(reg) p, options(nostack, readonly, preserves_flags));
    }
    v
}

/// See [`ldq_s8`].
///
/// # Safety
/// `p` must point at 4 readable `f32`s.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn ldq_f32(p: *const f32) -> core::arch::aarch64::float32x4_t {
    let v;
    // SAFETY: caller guarantees 16 readable bytes at `p`.
    unsafe {
        core::arch::asm!("ldr {v:q}, [{p}]", v = out(vreg) v, p = in(reg) p, options(nostack, readonly, preserves_flags));
    }
    v
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
        // `n_rows` slots; `[0, n_rows)` is in bounds. `matmul_sdot` (m_count=1 =
        // a matvec) splits the row range across the online cores.
        unsafe {
            crate::arch::aarch64::smp::matmul_sdot(w.as_ptr(), xq.as_ptr(), xs.as_ptr(), y.as_mut_ptr(), 1, n_rows, n_cols)
        };
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (&mut *xq, &mut *xs);
        matvec_q8_0(w, x, y, n_rows, n_cols);
    }
}

/// As `matvec_q8_0_fast`, but for `Q4_0` weights: on aarch64, quantize the
/// activation to int8 once and run the Q4_0 SDOT kernel (unpack nibbles ->
/// int8 -> `vdotq_s32`, ~the Q8_0 SDOT speed instead of the generic
/// dequant-and-dot path); elsewhere the exact scalar `matvec_q4_0`.
pub fn matvec_q4_0_fast(w: &[u8], x: &[f32], y: &mut [f32], xq: &mut [i8], xs: &mut [f32], n_rows: usize, n_cols: usize) {
    debug_assert_eq!(n_cols % QK, 0);
    debug_assert_eq!(x.len(), n_cols);
    debug_assert_eq!(y.len(), n_rows);
    debug_assert!(xq.len() >= n_cols && xs.len() >= n_cols / QK);
    #[cfg(target_arch = "aarch64")]
    {
        let xq = &mut xq[..n_cols];
        let xs = &mut xs[..n_cols / QK];
        quantize_activations_q8(x, xq, xs);
        // SAFETY: `w` holds `n_rows` Q4_0 rows of `n_cols/QK` blocks; `xq`/`xs`
        // are the just-computed quantized activation; `[0, n_rows)` in bounds.
        unsafe {
            crate::arch::aarch64::smp::matvec_q4_0_sdot(w.as_ptr(), xq.as_ptr(), xs.as_ptr(), y.as_mut_ptr(), n_rows, n_cols)
        };
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (&mut *xq, &mut *xs);
        matvec_q4_0(w, x, y, n_rows, n_cols);
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
        vget_low_s8, vmovl_s16, vmovl_s8,
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
            $p[0] = vfmaq_f32($p[0], f0, ldq_f32($xp));
            $p[1] = vfmaq_f32($p[1], f1, ldq_f32(($xp).add(4)));
            $p[2] = vfmaq_f32($p[2], f2, ldq_f32(($xp).add(8)));
            $p[3] = vfmaq_f32($p[3], f3, ldq_f32(($xp).add(12)));
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
                accumulate16!(p, ldq_s8(q), xp); // lanes 0..16
                accumulate16!(p, ldq_s8(q.add(16)), xp.add(16)); // lanes 16..32
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
    let blocks = n_cols / QK;
    let row_bytes = blocks * Q8_0_BLOCK_BYTES;
    // SAFETY: caller's contract; each row's dot reads in-bounds.
    unsafe {
        for r in row_start..row_end {
            *y.add(r) = sdot_one_row(w.add(r * row_bytes), xq, xs, blocks);
        }
    }
}

/// **Weight-stationary batched** `Q8_0` × `int8`-activation matmul over rows
/// `[row_start, row_end)`, computing `y[m][r] = W[r] · xq[m]` for all `m` in
/// `0..m_count`. Each weight row is loaded once (then L1-resident) and dotted
/// against every activation column, so the *weight bytes are read from memory
/// once for the whole batch* instead of once per column -- the amortization
/// that makes batched prefill much faster than looping the matvec. `y` is laid
/// out `[m * n_rows + r]` (each activation's output vector contiguous).
///
/// # Safety
/// `w` = `n_rows` `Q8_0` rows of `n_cols/QK` blocks; `xq`/`xs` = `m_count`
/// activations of `n_cols` `i8` / `n_cols/QK` `f32`; `y` = `m_count * n_rows`
/// `f32`; `row_start <= row_end <= n_rows`; `n_cols` a multiple of `QK`. Rows
/// are written disjointly, so distinct callers may own distinct row ranges.
/// Requires `dotprod` (aarch64 target features).
#[cfg(target_arch = "aarch64")]
pub unsafe fn matmul_q8_0_sdot_rows(
    w: *const u8,
    xq: *const i8,
    xs: *const f32,
    y: *mut f32,
    m_count: usize,
    n_rows: usize,
    row_start: usize,
    row_end: usize,
    n_cols: usize,
) {
    use core::arch::aarch64::{
        vaddq_s32, vaddvq_f32, vcvtq_f32_s32, vdotq_s32, vdupq_n_f32, vdupq_n_s32, vfmaq_n_f32,
    };
    let blocks = n_cols / QK;
    let row_bytes = blocks * Q8_0_BLOCK_BYTES;
    let nb = blocks; // scales per activation

    // m_count == 1 (decode) is a plain matvec: use the 4-accumulator per-row
    // dot, which has better ILP than a 1-wide tile of the batched kernel.
    if m_count == 1 {
        // SAFETY: caller's contract.
        unsafe {
            for r in row_start..row_end {
                *y.add(r) = sdot_one_row(w.add(r * row_bytes), xq, xs, blocks);
            }
        }
        return;
    }

    // Batched: tile the activation columns by 4 and, per weight block, load +
    // f16-decode the weight *once* and SDOT it against all activations in the
    // tile. This amortizes the weight load and (scalar) scale decode -- the
    // per-MAC overhead that dominates when compute-bound -- across the tile,
    // rather than redoing it per column. Four accumulators = four independent
    // FMA chains across blocks.
    const MT: usize = 4;
    // SAFETY: caller's contract; all loads in-bounds, `y` rows disjoint.
    unsafe {
        for r in row_start..row_end {
            let row = w.add(r * row_bytes);
            let mut m0 = 0;
            while m0 < m_count {
                let mt = core::cmp::min(MT, m_count - m0);
                let mut acc = [vdupq_n_f32(0.0); MT];
                for b in 0..blocks {
                    let base = b * Q8_0_BLOCK_BYTES;
                    let dw = f16_to_f32(u16::from_le_bytes([*row.add(base), *row.add(base + 1)]));
                    let wq0 = ldq_s8(row.add(base + 2) as *const i8); // weight lanes 0..16
                    let wq1 = ldq_s8(row.add(base + 18) as *const i8); // 16..32
                    for mm in 0..mt {
                        let mi = m0 + mm;
                        let xqp = xq.add(mi * n_cols + b * QK);
                        let dx = *xs.add(mi * nb + b);
                        let a0 = vdotq_s32(vdupq_n_s32(0), wq0, ldq_s8(xqp));
                        let a1 = vdotq_s32(vdupq_n_s32(0), wq1, ldq_s8(xqp.add(16)));
                        acc[mm] = vfmaq_n_f32(acc[mm], vcvtq_f32_s32(vaddq_s32(a0, a1)), dw * dx);
                    }
                }
                for mm in 0..mt {
                    *y.add((m0 + mm) * n_rows + r) = vaddvq_f32(acc[mm]);
                }
                m0 += MT;
            }
        }
    }
}

/// One `Q8_0`-row · `int8`-activation dot: `Σ_b (d_weight_b · d_act_b) · Σ q·xq`.
/// Two independent `SDOT`s per block feed a f32x4 accumulator with four
/// independent chains, so consecutive blocks don't serialize on one FMA's
/// latency; a single horizontal reduce at the end.
///
/// # Safety
/// `row` points at `blocks` `Q8_0` blocks; `xq`/`xs` at `blocks*QK` `i8` /
/// `blocks` `f32`. Requires `dotprod`.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn sdot_one_row(row: *const u8, xq: *const i8, xs: *const f32, blocks: usize) -> f32 {
    use core::arch::aarch64::{
        vaddq_f32, vaddq_s32, vaddvq_f32, vcvtq_f32_s32, vdotq_s32, vdupq_n_f32, vdupq_n_s32, vfmaq_n_f32,
    };
    macro_rules! block_into {
        ($f:expr, $b:expr) => {{
            let base = $b * Q8_0_BLOCK_BYTES;
            let dw = f16_to_f32(u16::from_le_bytes([*row.add(base), *row.add(base + 1)]));
            let dx = *xs.add($b);
            let q = row.add(base + 2) as *const i8;
            let xp = xq.add($b * QK);
            let a0 = vdotq_s32(vdupq_n_s32(0), ldq_s8(q), ldq_s8(xp)); // lanes 0..16
            let a1 = vdotq_s32(vdupq_n_s32(0), ldq_s8(q.add(16)), ldq_s8(xp.add(16))); // 16..32
            $f = vfmaq_n_f32($f, vcvtq_f32_s32(vaddq_s32(a0, a1)), dw * dx);
        }};
    }
    // SAFETY: caller's contract; all loads in-bounds.
    unsafe {
        let mut f0 = vdupq_n_f32(0.0);
        let mut f1 = vdupq_n_f32(0.0);
        let mut f2 = vdupq_n_f32(0.0);
        let mut f3 = vdupq_n_f32(0.0);
        let mut b = 0;
        while b + 4 <= blocks {
            block_into!(f0, b);
            block_into!(f1, b + 1);
            block_into!(f2, b + 2);
            block_into!(f3, b + 3);
            b += 4;
        }
        while b < blocks {
            block_into!(f0, b);
            b += 1;
        }
        vaddvq_f32(vaddq_f32(vaddq_f32(f0, f1), vaddq_f32(f2, f3)))
    }
}

/// One `Q4_0`-row · `int8`-activation dot, the Q4_0 analogue of `sdot_one_row`:
/// unpack each block's 16 packed bytes into 32 `int8` weights on the fly (low
/// nibbles -> elements 0..16, high nibbles -> 16..32, each minus 8, matching
/// `dequant_q4_0_block`'s layout) and `SDOT` them against the block's int8
/// activation, scaled by `d_weight · d_activation`. Four independent f32 chains.
///
/// # Safety
/// `row` points at `blocks` `Q4_0` blocks; `xq`/`xs` at `blocks*QK` `i8` /
/// `blocks` `f32`. Requires `dotprod`.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn sdot_one_row_q4_0(row: *const u8, xq: *const i8, xs: *const f32, blocks: usize) -> f32 {
    use core::arch::aarch64::{
        vaddq_f32, vaddq_s32, vaddvq_f32, vandq_u8, vcvtq_f32_s32, vdotq_s32, vdupq_n_f32, vdupq_n_s32, vdupq_n_s8,
        vdupq_n_u8, vfmaq_n_f32, vreinterpretq_s8_u8, vshrq_n_u8, vsubq_s8,
    };
    let mask = vdupq_n_u8(0x0f);
    let eight = vdupq_n_s8(8);
    macro_rules! block_into {
        ($f:expr, $b:expr) => {{
            let base = $b * Q4_0_BLOCK_BYTES;
            let dw = f16_to_f32(u16::from_le_bytes([*row.add(base), *row.add(base + 1)]));
            let dx = *xs.add($b);
            let bytes = ldq_u8(row.add(base + 2)); // 16 packed nibble pairs
            let lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(bytes, mask)), eight); // elems 0..16
            let hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(bytes, 4)), eight); // elems 16..32
            let xp = xq.add($b * QK);
            let a0 = vdotq_s32(vdupq_n_s32(0), lo, ldq_s8(xp));
            let a1 = vdotq_s32(vdupq_n_s32(0), hi, ldq_s8(xp.add(16)));
            $f = vfmaq_n_f32($f, vcvtq_f32_s32(vaddq_s32(a0, a1)), dw * dx);
        }};
    }
    // SAFETY: caller's contract; all loads in-bounds.
    unsafe {
        let mut f0 = vdupq_n_f32(0.0);
        let mut f1 = vdupq_n_f32(0.0);
        let mut f2 = vdupq_n_f32(0.0);
        let mut f3 = vdupq_n_f32(0.0);
        let mut b = 0;
        while b + 4 <= blocks {
            block_into!(f0, b);
            block_into!(f1, b + 1);
            block_into!(f2, b + 2);
            block_into!(f3, b + 3);
            b += 4;
        }
        while b < blocks {
            block_into!(f0, b);
            b += 1;
        }
        vaddvq_f32(vaddq_f32(vaddq_f32(f0, f1), vaddq_f32(f2, f3)))
    }
}

/// `Q4_0` matvec with int8-quantized activation (`xq`/`xs`) over rows
/// `[row_start, row_end)`, via the on-the-fly-unpack SDOT above -- the fast
/// analogue of `matvec_q8_0_sdot_rows` for the 9B's many Q4_0 tensors.
///
/// # Safety
/// See `sdot_one_row_q4_0`; `w` = `n_rows` Q4_0 rows of `n_cols/QK` blocks.
#[cfg(target_arch = "aarch64")]
pub unsafe fn matvec_q4_0_sdot_rows(
    w: *const u8,
    xq: *const i8,
    xs: *const f32,
    y: *mut f32,
    row_start: usize,
    row_end: usize,
    n_cols: usize,
) {
    let blocks = n_cols / QK;
    let row_bytes = blocks * Q4_0_BLOCK_BYTES;
    // SAFETY: caller's contract.
    unsafe {
        for r in row_start..row_end {
            *y.add(r) = sdot_one_row_q4_0(w.add(r * row_bytes), xq, xs, blocks);
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

/// GGML quant type codes (subset Chitti dequantizes), matching the GGUF
/// on-disk `ggml_type` values so a tensor's type can be carried straight
/// through from the header.
pub const QT_Q4_0: u32 = 2;
pub const QT_Q4_1: u32 = 3;
pub const QT_Q8_0: u32 = 8;
pub const QT_Q5_K: u32 = 13;
pub const QT_Q6_K: u32 = 14;

/// Bytes per quantization block and elements per block for a quant type.
pub fn block_layout(qt: u32) -> (usize, usize) {
    match qt {
        QT_Q4_0 => (Q4_0_BLOCK_BYTES, QK),
        QT_Q4_1 => (Q4_1_BLOCK_BYTES, QK),
        QT_Q8_0 => (Q8_0_BLOCK_BYTES, QK),
        QT_Q5_K => (Q5_K_BLOCK_BYTES, QK_K),
        QT_Q6_K => (Q6_K_BLOCK_BYTES, QK_K),
        _ => (0, 0),
    }
}

/// Dequantize one block of quant type `qt` into `out` (length = the type's
/// block element count).
pub fn dequant_block(qt: u32, block: &[u8], out: &mut [f32]) {
    match qt {
        QT_Q4_0 => dequant_q4_0_block(block, out),
        QT_Q4_1 => dequant_q4_1_block(block, out),
        QT_Q8_0 => dequant_q8_0_block(block, out),
        QT_Q5_K => dequant_q5_k_block(block, out),
        QT_Q6_K => dequant_q6_k_block(block, out),
        _ => {}
    }
}

/// Generic `y[r] = W[r] · x` over rows `[row_start, row_end)` for any supported
/// quant type: dequantize each weight block to `f32` and dot it against the
/// matching `x` window, accumulating. Correct for every type (the fallback for
/// the mixed-quant 9B's Q4_0/Q4_1/Q5_K/Q6_K tensors); Q8_0 has a faster SDOT
/// path elsewhere. `n_cols` must be a multiple of the type's block element
/// count.
///
/// # Safety
/// `w` = `n_rows` rows of `n_cols/elems` blocks of `qt`; `x` = `n_cols` f32;
/// `y` = `n_rows` f32; `row_start <= row_end <= n_rows`. Rows written disjointly.
pub unsafe fn matvec_quant_rows(
    qt: u32,
    w: *const u8,
    x: *const f32,
    y: *mut f32,
    row_start: usize,
    row_end: usize,
    n_cols: usize,
) {
    let (block_bytes, elems) = block_layout(qt);
    if block_bytes == 0 {
        return;
    }
    let blocks = n_cols / elems;
    let row_bytes = blocks * block_bytes;
    let mut buf = [0.0f32; QK_K]; // holds one dequantized block (32 or 256)
    // SAFETY: caller's contract; every slice below is in bounds.
    unsafe {
        for r in row_start..row_end {
            let row = core::slice::from_raw_parts(w.add(r * row_bytes), row_bytes);
            let mut acc = 0.0f32;
            for b in 0..blocks {
                let block = &row[b * block_bytes..(b + 1) * block_bytes];
                dequant_block(qt, block, &mut buf[..elems]);
                let xb = core::slice::from_raw_parts(x.add(b * elems), elems);
                acc += dot_f32(&buf[..elems], xb);
            }
            *y.add(r) = acc;
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
