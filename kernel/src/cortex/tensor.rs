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
pub const Q5_0_BLOCK_BYTES: usize = 2 + 4 + QK / 2; // f16 d + qh[4] + 16 nibbles = 22
pub const Q5_1_BLOCK_BYTES: usize = 2 + 2 + 4 + QK / 2; // f16 d + f16 m + qh[4] + 16 nibbles = 24

// PrismML ternary block (GGML type 42, `Q2_0`): 128 elements, one f16 scale
// + 128 2-bit codes (Ternary-Bonsai / Bonsai-27B `Q2_0` weights and the
// drafter's `type42` embedding — both this exact layout).
pub const QK2_0: usize = 128; // elements per Q2_0 block
pub const Q2_0_BLOCK_BYTES: usize = 2 + QK2_0 / 4; // f16 scale + 128*2 bits = 34

// PrismML binary block (GGML type 41, `Q1_0`): 128 elements, one f16 scale + 128
// 1-bit signs (Bonsai-27B 1-bit weights). value = bit ? +d : −d, i.e. {−1,+1}·d.
pub const QK1_0: usize = 128; // elements per Q1_0 block
pub const Q1_0_BLOCK_BYTES: usize = 2 + QK1_0 / 8; // f16 scale + 128*1 bit = 18

// k-quant super-block: 256 elements. Byte layouts verbatim from llama.cpp
// (ggml-common.h). Mixed-quant GGUFs (Q4_K_M/Q5_K_M/UD-* files) tag these
// per tensor.
pub const QK_K: usize = 256;
pub const Q2_K_BLOCK_BYTES: usize = QK_K / 16 + QK_K / 4 + 2 + 2; // scales[16],qs[64],d,dmin = 84
pub const Q3_K_BLOCK_BYTES: usize = QK_K / 8 + QK_K / 4 + 12 + 2; // hmask[32],qs[64],scales[12],d = 110
pub const Q4_K_BLOCK_BYTES: usize = 2 + 2 + 12 + QK_K / 2; // d,dmin,scales[12],qs[128] = 144
pub const Q5_K_BLOCK_BYTES: usize = 2 + 2 + 12 + QK_K / 8 + QK_K / 2; // d,dmin,scales[12],qh[32],qs[128] = 176
pub const Q6_K_BLOCK_BYTES: usize = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2; // ql[128],qh[64],scales[16],d = 210
pub const Q8_K_BLOCK_BYTES: usize = 4 + QK_K + QK_K / 16 * 2; // d(f32),qs[256],bsums[16 i16] = 292

// i-quants (codebook/grid formats; layouts verbatim from ggml-common.h).
pub const IQ2_XXS_BLOCK_BYTES: usize = 2 + QK_K / 4; // d + qs u16[32] = 66
pub const IQ2_XS_BLOCK_BYTES: usize = 2 + QK_K / 4 + QK_K / 32; // + scales[8] = 74
pub const IQ2_S_BLOCK_BYTES: usize = 2 + QK_K / 4 + QK_K / 32 + QK_K / 32; // qs[64] + qh[8] + scales[8] = 82
pub const IQ3_XXS_BLOCK_BYTES: usize = 2 + 3 * QK_K / 8; // qs[64] + scales_and_signs[32] = 98
pub const IQ3_S_BLOCK_BYTES: usize = 2 + QK_K / 4 + QK_K / 32 + QK_K / 8 + QK_K / 64; // qs+qh+signs+scales = 110
pub const IQ4_NL_BLOCK_BYTES: usize = 2 + QK / 2; // d + 16 nibbles = 18 (32-elem block)
pub const IQ4_XS_BLOCK_BYTES: usize = 2 + 2 + QK_K / 64 + QK_K / 2; // d + scales_h + scales_l[4] + qs[128] = 136

// Unquantized weight tensors (F16/BF16, e.g. in UD-*_XL mixes) run through
// the same block machinery in 32-element chunks so the generic matvec and
// the SMP row-split work unchanged.
pub const F16_BLOCK_BYTES: usize = 2 * QK; // 32 halves
pub const BF16_BLOCK_BYTES: usize = 2 * QK; // 32 bfloat16

/// Convert an IEEE-754 half (as raw bits) to `f32`. On aarch64 this is the
/// single `fcvt s, h` instruction (`+fp-armv8` baseline) — it runs once per
/// Q8_0 block scale in the matvec hot loop, where the bit-manipulation
/// fallback's ~10 scalar ops per 64-MAC block were measurable. Elsewhere,
/// pure bit manipulation (exact, handles subnormals/inf/NaN).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn f16_to_f32(bits: u16) -> f32 {
    let out: f32;
    // SAFETY: register-only conversion instruction; no memory access.
    unsafe {
        core::arch::asm!(
            "fmov {tmp:s}, {bits:w}",
            "fcvt {out:s}, {tmp:h}",
            bits = in(reg) bits as u32,
            tmp = out(vreg) _,
            out = out(vreg) out,
            options(nostack, nomem, pure, preserves_flags),
        );
    }
    out
}

/// Convert an IEEE-754 half (as raw bits) to `f32`, purely by bit
/// manipulation (no `std` transcendentals). Exact: every `f16` value is
/// representable in `f32`. Handles subnormals, inf, and NaN.
#[cfg(not(target_arch = "aarch64"))]
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

/// Dequantize one Q2_0 block (`Q2_0_BLOCK_BYTES`) into 128 `f32`s. PrismML's
/// ternary pack (GGML type 42, shipped by `Ternary-Bonsai`/`Bonsai-27B`): one
/// f16 scale `d` followed by 128 2-bit codes, four per byte, low bits first.
/// Code `c` (0..3) dequantizes to `(c - 1) * d`, i.e. `00 → -1`, `01 → 0`,
/// `10 → +1`, `11 → +2` scaled by `d` (matches llama.cpp `dequantize_row_q2_0`).
pub fn dequant_q2_0_block(block: &[u8], out: &mut [f32]) {
    debug_assert_eq!(block.len(), Q2_0_BLOCK_BYTES);
    debug_assert_eq!(out.len(), QK2_0);
    let d = read_f16_le(&block[0..2]);
    for j in 0..QK2_0 {
        let byte = block[2 + j / 4];
        let code = ((byte >> ((j % 4) * 2)) & 0x03) as i32;
        out[j] = d * (code - 1) as f32;
    }
}

/// Dequantize one Q1_0 block (`Q1_0_BLOCK_BYTES`) into 128 `f32`s. PrismML's
/// binary pack (GGML type 41, `Bonsai-27B` 1-bit): one f16 scale `d` + 128
/// sign bits (8 per byte, LSB first). `bit == 1 → +d`, `bit == 0 → −d`
/// (matches llama.cpp `dequantize_row_q1_0`).
pub fn dequant_q1_0_block(block: &[u8], out: &mut [f32]) {
    debug_assert_eq!(block.len(), Q1_0_BLOCK_BYTES);
    debug_assert_eq!(out.len(), QK1_0);
    let d = read_f16_le(&block[0..2]);
    for j in 0..QK1_0 {
        let bit = (block[2 + j / 8] >> (j % 8)) & 1;
        out[j] = if bit == 1 { d } else { -d };
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

/// Dequantize one Q5_0 block (`Q5_0_BLOCK_BYTES`) into 32 `f32`s, per
/// llama.cpp `dequantize_row_q5_0`: 4-bit nibbles + a 5th bit from the packed
/// `qh` word, biased by -16. Element `j` takes qh bit `j`, element `j+16`
/// takes qh bit `j+16` (via the `>> (j+12)` trick on the low-shifted word).
pub fn dequant_q5_0_block(block: &[u8], out: &mut [f32]) {
    debug_assert_eq!(block.len(), Q5_0_BLOCK_BYTES);
    debug_assert_eq!(out.len(), QK);
    let d = read_f16_le(&block[0..2]);
    let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
    for j in 0..QK / 2 {
        let xh0 = ((qh >> j) << 4) & 0x10;
        let xh1 = (qh >> (j + 12)) & 0x10;
        let x0 = ((block[6 + j] & 0x0f) as u32 | xh0) as i32 - 16;
        let x1 = ((block[6 + j] >> 4) as u32 | xh1) as i32 - 16;
        out[j] = d * x0 as f32;
        out[j + QK / 2] = d * x1 as f32;
    }
}

/// Dequantize one Q5_1 block (`Q5_1_BLOCK_BYTES`) into 32 `f32`s. Like Q5_0
/// but affine: `x = d*q5 + m` (quants unsigned 0..31, no -16 bias).
pub fn dequant_q5_1_block(block: &[u8], out: &mut [f32]) {
    debug_assert_eq!(block.len(), Q5_1_BLOCK_BYTES);
    debug_assert_eq!(out.len(), QK);
    let d = read_f16_le(&block[0..2]);
    let m = read_f16_le(&block[2..4]);
    let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    for j in 0..QK / 2 {
        let xh0 = ((qh >> j) << 4) & 0x10;
        let xh1 = (qh >> (j + 12)) & 0x10;
        let x0 = (block[8 + j] & 0x0f) as u32 | xh0;
        let x1 = (block[8 + j] >> 4) as u32 | xh1;
        out[j] = d * x0 as f32 + m;
        out[j + QK / 2] = d * x1 as f32 + m;
    }
}

/// Dequantize one Q2_K super-block (`Q2_K_BLOCK_BYTES`) into 256 `f32`s, per
/// llama.cpp `dequantize_row_q2_K`: 16 groups of 16 elements, each with a
/// 4-bit scale + 4-bit min packed in one `scales` byte; 2-bit quants walked
/// in two 128-element halves by shift (0/2/4/6). `x = d*sc*q2 - dmin*m`.
pub fn dequant_q2_k_block(block: &[u8], out: &mut [f32]) {
    debug_assert_eq!(block.len(), Q2_K_BLOCK_BYTES);
    debug_assert_eq!(out.len(), QK_K);
    let scales = &block[0..16];
    let qs = &block[16..80];
    let d = read_f16_le(&block[80..82]);
    let dmin = read_f16_le(&block[82..84]);
    let mut y = 0usize;
    let mut is = 0usize;
    for half in 0..2 {
        let q = &qs[half * 32..];
        let mut shift = 0u32;
        for _ in 0..4 {
            for sub in 0..2 {
                let sc = scales[is];
                is += 1;
                let dl = d * (sc & 0x0f) as f32;
                let ml = dmin * (sc >> 4) as f32;
                for l in 0..16 {
                    out[y] = dl * ((q[sub * 16 + l] >> shift) & 3) as f32 - ml;
                    y += 1;
                }
            }
            shift += 2;
        }
    }
}

/// Dequantize one Q3_K super-block (`Q3_K_BLOCK_BYTES`) into 256 `f32`s, per
/// llama.cpp `dequantize_row_q3_K`: 2-bit low quants + a high bit from
/// `hmask` (subtracting 4 when the mask bit is *clear*), 16 6-bit signed
/// scales unpacked from 12 bytes with the kmask shuffle. `x = d*(sc-32)*q3`.
pub fn dequant_q3_k_block(block: &[u8], out: &mut [f32]) {
    debug_assert_eq!(block.len(), Q3_K_BLOCK_BYTES);
    debug_assert_eq!(out.len(), QK_K);
    let hm = &block[0..32];
    let qs = &block[32..96];
    let sc_packed = &block[96..108];
    let d_all = read_f16_le(&block[108..110]);

    // Unpack the 16 6-bit scales exactly as llama.cpp's kmask shuffle: three
    // u32 words -> four u32 words of 8-bit lanes (low 4 bits from words 0/1,
    // high 2 bits from word 2), read back as i8 lanes.
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    let w = |i: usize| u32::from_le_bytes([sc_packed[i * 4], sc_packed[i * 4 + 1], sc_packed[i * 4 + 2], sc_packed[i * 4 + 3]]);
    let (a0, a1, tmp) = (w(0), w(1), w(2));
    let aux = [
        (a0 & KMASK2) | (((tmp) & KMASK1) << 4),
        (a1 & KMASK2) | (((tmp >> 2) & KMASK1) << 4),
        ((a0 >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4),
        ((a1 >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4),
    ];
    let mut scales = [0i8; 16];
    for (i, a) in aux.iter().enumerate() {
        scales[i * 4..i * 4 + 4].copy_from_slice(&a.to_le_bytes().map(|b| b as i8));
    }

    let mut y = 0usize;
    let mut is = 0usize;
    let mut m = 1u8;
    for half in 0..2 {
        let q = &qs[half * 32..];
        let mut shift = 0u32;
        for _ in 0..4 {
            for sub in 0..2 {
                let dl = d_all * (scales[is] as f32 - 32.0);
                is += 1;
                for l in 0..16 {
                    let idx = sub * 16 + l;
                    let hi = if hm[idx] & m != 0 { 0 } else { 4 };
                    out[y] = dl * (((q[idx] >> shift) & 3) as i32 - hi) as f32;
                    y += 1;
                }
            }
            shift += 2;
            m <<= 1;
        }
    }
}

/// Dequantize one Q4_K super-block (`Q4_K_BLOCK_BYTES`) into 256 `f32`s, per
/// llama.cpp `dequantize_row_q4_K`: 8 sub-blocks of 32 with packed 6-bit
/// scale/min pairs (same `get_scale_min_k4` as Q5_K), 4-bit quants split
/// low-nibbles-first per 64. `x = d*sc*q4 - dmin*m`.
pub fn dequant_q4_k_block(block: &[u8], out: &mut [f32]) {
    debug_assert_eq!(block.len(), Q4_K_BLOCK_BYTES);
    debug_assert_eq!(out.len(), QK_K);
    let d = read_f16_le(&block[0..2]);
    let dmin = read_f16_le(&block[2..4]);
    let scales = &block[4..16];
    let qs = &block[16..144];
    let mut y = 0usize;
    let mut is = 0usize;
    let mut ql = 0usize;
    for _ in 0..QK_K / 64 {
        let (sc1, m1s) = q_scale_min_k4(is, scales);
        let (sc2, m2s) = q_scale_min_k4(is + 1, scales);
        let d1 = d * sc1 as f32;
        let m1 = dmin * m1s as f32;
        let d2 = d * sc2 as f32;
        let m2 = dmin * m2s as f32;
        for l in 0..32 {
            out[y + l] = d1 * (qs[ql + l] & 0x0f) as f32 - m1;
        }
        for l in 0..32 {
            out[y + 32 + l] = d2 * (qs[ql + l] >> 4) as f32 - m2;
        }
        y += 64;
        ql += 32;
        is += 2;
    }
}

/// Dequantize one Q8_K super-block (`Q8_K_BLOCK_BYTES`) into 256 `f32`s:
/// an f32 scale and plain i8 quants (`bsums` are an activation-side aid,
/// ignored when Q8_K appears as a weight type, e.g. in UD-Q8_K_XL mixes).
pub fn dequant_q8_k_block(block: &[u8], out: &mut [f32]) {
    debug_assert_eq!(block.len(), Q8_K_BLOCK_BYTES);
    debug_assert_eq!(out.len(), QK_K);
    let d = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
    for i in 0..QK_K {
        out[i] = d * block[4 + i] as i8 as f32;
    }
}

/// "Dequantize" a 32-element chunk of an F16 weight tensor.
pub fn dequant_f16_block(block: &[u8], out: &mut [f32]) {
    debug_assert_eq!(block.len(), F16_BLOCK_BYTES);
    debug_assert_eq!(out.len(), QK);
    for i in 0..QK {
        out[i] = read_f16_le(&block[i * 2..i * 2 + 2]);
    }
}

/// "Dequantize" a 32-element chunk of a BF16 weight tensor: a bfloat16 is
/// the top 16 bits of the equivalent f32.
pub fn dequant_bf16_block(block: &[u8], out: &mut [f32]) {
    debug_assert_eq!(block.len(), BF16_BLOCK_BYTES);
    debug_assert_eq!(out.len(), QK);
    for i in 0..QK {
        let bits = u16::from_le_bytes([block[i * 2], block[i * 2 + 1]]) as u32;
        out[i] = f32::from_bits(bits << 16);
    }
}

// --- i-quants: codebook/grid formats (IQ2/IQ3/IQ4 families). Ports of
// llama.cpp's `dequantize_row_iq*` with the tables generated verbatim into
// `iq_tables` (tools/gen_iq_tables.py), so decoding is byte-identical to
// ggml's. Each 8-element group looks its magnitudes up in a grid entry (the
// bytes of a u64/u32 table value) and applies a 7-bit codebook sign mask. ---

use super::iq_tables as iqt;

/// Apply grid byte `g` and codebook sign bit `j` of `signs`: the shared
/// magnitude×sign step of every IQ2/IQ3 format.
#[inline(always)]
fn iq_signed(db: f32, g: u8, signs: u8, j: usize) -> f32 {
    let v = db * g as f32;
    if signs & iqt::KMASK_IQ2XS[j] != 0 {
        -v
    } else {
        v
    }
}

/// Dequantize one IQ2_XXS super-block (`IQ2_XXS_BLOCK_BYTES`) into 256 f32s:
/// per 32 elements, 4 grid-index bytes (u64 grid entries = 8 magnitudes) and
/// a u32 carrying 4×7-bit sign indices + a 4-bit scale in the top bits.
pub fn dequant_iq2_xxs_block(block: &[u8], out: &mut [f32]) {
    debug_assert_eq!(block.len(), IQ2_XXS_BLOCK_BYTES);
    debug_assert_eq!(out.len(), QK_K);
    let d = read_f16_le(&block[0..2]);
    let qs = &block[2..66];
    let mut y = 0usize;
    for ib32 in 0..QK_K / 32 {
        let a = &qs[8 * ib32..8 * ib32 + 8];
        let aux0 = [a[0], a[1], a[2], a[3]];
        let aux1 = u32::from_le_bytes([a[4], a[5], a[6], a[7]]);
        let db = d * (0.5 + (aux1 >> 28) as f32) * 0.25;
        for l in 0..4 {
            let grid = iqt::IQ2XXS_GRID[aux0[l] as usize].to_le_bytes();
            let signs = iqt::KSIGNS_IQ2XS[((aux1 >> (7 * l)) & 127) as usize];
            for j in 0..8 {
                out[y + j] = iq_signed(db, grid[j], signs, j);
            }
            y += 8;
        }
    }
}

/// Dequantize one IQ2_XS super-block (`IQ2_XS_BLOCK_BYTES`): per element
/// group a u16 packs a 9-bit grid index + 7-bit sign index; nibble scales.
pub fn dequant_iq2_xs_block(block: &[u8], out: &mut [f32]) {
    debug_assert_eq!(block.len(), IQ2_XS_BLOCK_BYTES);
    debug_assert_eq!(out.len(), QK_K);
    let d = read_f16_le(&block[0..2]);
    let qs = &block[2..66]; // u16[32]
    let scales = &block[66..74];
    let mut y = 0usize;
    for ib32 in 0..QK_K / 32 {
        let db = [
            d * (0.5 + (scales[ib32] & 0x0f) as f32) * 0.25,
            d * (0.5 + (scales[ib32] >> 4) as f32) * 0.25,
        ];
        for l in 0..4 {
            let q = u16::from_le_bytes([qs[(4 * ib32 + l) * 2], qs[(4 * ib32 + l) * 2 + 1]]);
            let grid = iqt::IQ2XS_GRID[(q & 511) as usize].to_le_bytes();
            let signs = iqt::KSIGNS_IQ2XS[(q >> 9) as usize];
            for j in 0..8 {
                out[y + j] = iq_signed(db[l / 2], grid[j], signs, j);
            }
            y += 8;
        }
    }
}

/// Dequantize one IQ2_S super-block (`IQ2_S_BLOCK_BYTES`): 8-bit grid index
/// extended by 2 bits from `qh`; explicit sign bytes in the second half of
/// the qs region; nibble scales.
pub fn dequant_iq2_s_block(block: &[u8], out: &mut [f32]) {
    debug_assert_eq!(block.len(), IQ2_S_BLOCK_BYTES);
    debug_assert_eq!(out.len(), QK_K);
    let d = read_f16_le(&block[0..2]);
    let qs_all = &block[2..66]; // 32 grid bytes + 32 sign bytes
    let qh = &block[66..74];
    let scales = &block[74..82];
    let (mut qs_off, mut sg_off) = (0usize, 32usize);
    let mut y = 0usize;
    for ib32 in 0..QK_K / 32 {
        let db = [
            d * (0.5 + (scales[ib32] & 0x0f) as f32) * 0.25,
            d * (0.5 + (scales[ib32] >> 4) as f32) * 0.25,
        ];
        for l in 0..4 {
            let idx = qs_all[qs_off + l] as usize | (((qh[ib32] as usize) << (8 - 2 * l)) & 0x300);
            let grid = iqt::IQ2S_GRID[idx].to_le_bytes();
            let signs = qs_all[sg_off + l];
            for j in 0..8 {
                out[y + j] = iq_signed(db[l / 2], grid[j], signs, j);
            }
            y += 8;
        }
        qs_off += 4;
        sg_off += 4;
    }
}

/// Dequantize one IQ3_XXS super-block (`IQ3_XXS_BLOCK_BYTES`): 8-bit grid
/// indices into the u32 grid (4 magnitudes each, two entries per 8 elements),
/// with a trailing u32 per 32 elements carrying signs + a 4-bit scale.
pub fn dequant_iq3_xxs_block(block: &[u8], out: &mut [f32]) {
    debug_assert_eq!(block.len(), IQ3_XXS_BLOCK_BYTES);
    debug_assert_eq!(out.len(), QK_K);
    let d = read_f16_le(&block[0..2]);
    let qs = &block[2..66];
    let ss = &block[66..98]; // scales_and_signs: u32 per ib32
    let mut y = 0usize;
    for ib32 in 0..QK_K / 32 {
        let aux = u32::from_le_bytes([ss[4 * ib32], ss[4 * ib32 + 1], ss[4 * ib32 + 2], ss[4 * ib32 + 3]]);
        let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
        for l in 0..4 {
            let signs = iqt::KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
            let g1 = iqt::IQ3XXS_GRID[qs[8 * ib32 + 2 * l] as usize].to_le_bytes();
            let g2 = iqt::IQ3XXS_GRID[qs[8 * ib32 + 2 * l + 1] as usize].to_le_bytes();
            for j in 0..4 {
                out[y + j] = iq_signed(db, g1[j], signs, j);
                out[y + 4 + j] = iq_signed(db, g2[j], signs, 4 + j);
            }
            y += 8;
        }
    }
}

/// Dequantize one IQ3_S super-block (`IQ3_S_BLOCK_BYTES`): 8-bit grid indices
/// extended by 1 bit from `qh`, explicit sign bytes, nibble scales over
/// 64-element pairs.
pub fn dequant_iq3_s_block(block: &[u8], out: &mut [f32]) {
    debug_assert_eq!(block.len(), IQ3_S_BLOCK_BYTES);
    debug_assert_eq!(out.len(), QK_K);
    let d = read_f16_le(&block[0..2]);
    let qs = &block[2..66];
    let qh = &block[66..74];
    let signs = &block[74..106];
    let scales = &block[106..110];
    let (mut qs_off, mut sg_off) = (0usize, 0usize);
    let mut y = 0usize;
    for ib64 in 0..QK_K / 64 {
        let db1 = d * (1 + 2 * (scales[ib64] & 0x0f) as i32) as f32;
        let db2 = d * (1 + 2 * (scales[ib64] >> 4) as i32) as f32;
        let qh0 = qh[2 * ib64] as usize;
        let qh1 = qh[2 * ib64 + 1] as usize;
        for (db, qhb) in [(db1, qh0), (db2, qh1)] {
            for l in 0..4 {
                let g1 = iqt::IQ3S_GRID[qs[qs_off + 2 * l] as usize | ((qhb << (8 - 2 * l)) & 256)].to_le_bytes();
                let g2 = iqt::IQ3S_GRID[qs[qs_off + 2 * l + 1] as usize | ((qhb << (7 - 2 * l)) & 256)].to_le_bytes();
                let sg = signs[sg_off + l];
                for j in 0..4 {
                    out[y + j] = iq_signed(db, g1[j], sg, j);
                    out[y + 4 + j] = iq_signed(db, g2[j], sg, 4 + j);
                }
                y += 8;
            }
            qs_off += 8;
            sg_off += 4;
        }
    }
}

/// Dequantize one IQ4_NL block (`IQ4_NL_BLOCK_BYTES`, 32 elements): plain
/// nibbles mapped through the non-linear 16-entry codebook.
pub fn dequant_iq4_nl_block(block: &[u8], out: &mut [f32]) {
    debug_assert_eq!(block.len(), IQ4_NL_BLOCK_BYTES);
    debug_assert_eq!(out.len(), QK);
    let d = read_f16_le(&block[0..2]);
    for j in 0..QK / 2 {
        out[j] = d * iqt::KVALUES_IQ4NL[(block[2 + j] & 0x0f) as usize] as f32;
        out[j + QK / 2] = d * iqt::KVALUES_IQ4NL[(block[2 + j] >> 4) as usize] as f32;
    }
}

/// Dequantize one IQ4_XS super-block (`IQ4_XS_BLOCK_BYTES`): the IQ4_NL
/// codebook with per-32 6-bit scales split across `scales_l`/`scales_h`.
pub fn dequant_iq4_xs_block(block: &[u8], out: &mut [f32]) {
    debug_assert_eq!(block.len(), IQ4_XS_BLOCK_BYTES);
    debug_assert_eq!(out.len(), QK_K);
    let d = read_f16_le(&block[0..2]);
    let scales_h = u16::from_le_bytes([block[2], block[3]]);
    let scales_l = &block[4..8];
    let qs = &block[8..136];
    let mut y = 0usize;
    for ib in 0..QK_K / 32 {
        let ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0f) as i32 | ((((scales_h >> (2 * ib)) & 3) as i32) << 4);
        let dl = d * (ls - 32) as f32;
        for j in 0..16 {
            out[y + j] = dl * iqt::KVALUES_IQ4NL[(qs[16 * ib + j] & 0x0f) as usize] as f32;
            out[y + 16 + j] = dl * iqt::KVALUES_IQ4NL[(qs[16 * ib + j] >> 4) as usize] as f32;
        }
        y += 32;
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

/// `y[i] += a * x[i]` — contiguous AXPY, the row kernel of the DeltaNet
/// recurrent update (state/kv_mem/out rows are contiguous slices). On aarch64
/// this is an inline-asm NEON loop for the same reason as [`dot_f32_neon`]:
/// `+strict-align` scalarizes any vector load/store LLVM emits, including
/// auto-vectorized plain loops. x86 auto-vectorizes the fallback fine (no
/// strict-align there).
pub fn axpy_f32(y: &mut [f32], x: &[f32], a: f32) {
    let n = y.len().min(x.len());
    #[cfg(target_arch = "aarch64")]
    {
        let n8 = n / 8;
        if n8 > 0 {
            // SAFETY: reads/writes exactly `n8 * 8` in-bounds f32 from x/y;
            // v8–v15 (callee-saved) untouched.
            unsafe {
                core::arch::asm!(
                    "dup v0.4s, {a:v}.s[0]",
                    "1:",
                    "ldp q1, q2, [{px}], #32",
                    "ldp q3, q4, [{py}]",
                    "fmla v3.4s, v1.4s, v0.4s",
                    "fmla v4.4s, v2.4s, v0.4s",
                    "stp q3, q4, [{py}], #32",
                    "subs {cnt}, {cnt}, #1",
                    "b.ne 1b",
                    px = inout(reg) x.as_ptr() => _,
                    py = inout(reg) y.as_mut_ptr() => _,
                    cnt = inout(reg) n8 => _,
                    a = in(vreg) a,
                    out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _,
                    options(nostack),
                );
            }
        }
        for i in n8 * 8..n {
            y[i] += a * x[i];
        }
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    for i in 0..n {
        y[i] += a * x[i];
    }
}

/// `y[i] *= a` — contiguous scale (the DeltaNet per-token state decay touches
/// every 128×128 state cell of every head). Inline-asm NEON on aarch64 (see
/// [`axpy_f32`]); plain loop elsewhere.
pub fn scale_f32(y: &mut [f32], a: f32) {
    let n = y.len();
    #[cfg(target_arch = "aarch64")]
    {
        let n8 = n / 8;
        if n8 > 0 {
            // SAFETY: reads/writes exactly `n8 * 8` in-bounds f32 from y.
            unsafe {
                core::arch::asm!(
                    "dup v0.4s, {a:v}.s[0]",
                    "1:",
                    "ldp q1, q2, [{py}]",
                    "fmul v1.4s, v1.4s, v0.4s",
                    "fmul v2.4s, v2.4s, v0.4s",
                    "stp q1, q2, [{py}], #32",
                    "subs {cnt}, {cnt}, #1",
                    "b.ne 1b",
                    py = inout(reg) y.as_mut_ptr() => _,
                    cnt = inout(reg) n8 => _,
                    a = in(vreg) a,
                    out("v0") _, out("v1") _, out("v2") _,
                    options(nostack),
                );
            }
        }
        for i in n8 * 8..n {
            y[i] *= a;
        }
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    for i in 0..n {
        y[i] *= a;
    }
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

/// Paired 32-byte vector load (`ldp q, q`): two 16-lane `int8` vectors in one
/// instruction — half the load slots of two `ldr q` (the `+strict-align` rule:
/// inline asm, never `vld1q`).
///
/// # Safety
/// `p` must point at 32 readable bytes.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn ldp_s8(p: *const i8) -> (core::arch::aarch64::int8x16_t, core::arch::aarch64::int8x16_t) {
    let (a, b);
    // SAFETY: caller guarantees 32 readable bytes at `p`.
    unsafe {
        core::arch::asm!("ldp {a:q}, {b:q}, [{p}]", a = out(vreg) a, b = out(vreg) b, p = in(reg) p, options(nostack, readonly, preserves_flags));
    }
    (a, b)
}

/// 16-byte-aligned constant block for `ldq_u8`-loaded NEON lookup vectors.
#[cfg(target_arch = "aarch64")]
#[repr(align(16))]
#[derive(Clone, Copy)]
struct Align16([u8; 16]);

/// TBL index vectors for the Q1_0 sign-bit expand: vector `k` broadcasts
/// source byte `2k` across lanes 0..8 and byte `2k+1` across lanes 8..16, so
/// one `vqtbl1q` positions the two bytes whose bits are lanes `16k..16k+16`.
#[cfg(target_arch = "aarch64")]
static Q1_0_TBL_IDX: [Align16; 8] = {
    let mut out = [Align16([0; 16]); 8];
    let mut k = 0;
    while k < 8 {
        let mut l = 0;
        while l < 16 {
            out[k].0[l] = (2 * k + l / 8) as u8;
            l += 1;
        }
        k += 1;
    }
    out
};

/// Per-lane single-bit test mask for the Q1_0 expand (LSB-first within each
/// broadcast byte, matching `dequant_q1_0_block`'s bit order).
#[cfg(target_arch = "aarch64")]
static Q1_0_BITS: Align16 = Align16([1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128]);

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
/// One Q4_K row · int8-quantized activation, NEON SDOT. Per 32-element
/// sub-block `j` of each 256-element super-block:
/// `d·sc_j·xs_j·SDOT(q, xq)  −  dmin·m_j·xs_j·SDOT(xq, 1)`
/// (the affine min-term needs the activation *sum*, an SDOT against ones).
/// Nibble layout per 64-element pair: 32 bytes, low nibbles = first sub-block,
/// high nibbles = second. All vector loads go through the `ldq_*` inline-asm
/// helpers (the `+strict-align` rule — plain `vld1q` scalarizes into ldrb).
#[cfg(target_arch = "aarch64")]
unsafe fn sdot_one_row_q4_k(row: *const u8, xq: *const i8, xs: *const f32, superblocks: usize) -> f32 {
    use core::arch::aarch64::{
        vaddq_s32, vaddvq_s32, vandq_u8, vdotq_s32, vdupq_n_s32, vdupq_n_s8, vdupq_n_u8, vreinterpretq_s8_u8,
        vshrq_n_u8,
    };
    let mask = vdupq_n_u8(0x0f);
    let ones = vdupq_n_s8(1);
    // SAFETY: caller's contract; all loads in-bounds.
    unsafe {
        let mut acc = 0.0f32;
        for b in 0..superblocks {
            let base = b * Q4_K_BLOCK_BYTES;
            let d = f16_to_f32(u16::from_le_bytes([*row.add(base), *row.add(base + 1)]));
            let dmin = f16_to_f32(u16::from_le_bytes([*row.add(base + 2), *row.add(base + 3)]));
            let scales = core::slice::from_raw_parts(row.add(base + 4), 12);
            let qs = row.add(base + 16);
            // 4 pairs of 32-element sub-blocks (8 sub-blocks) per super-block.
            for pair in 0..4 {
                let (sc0, m0) = q_scale_min_k4(2 * pair, scales);
                let (sc1, m1) = q_scale_min_k4(2 * pair + 1, scales);
                let bytes_a = ldq_u8(qs.add(pair * 32));
                let bytes_b = ldq_u8(qs.add(pair * 32 + 16));
                let lo_a = vreinterpretq_s8_u8(vandq_u8(bytes_a, mask)); // sub 2p elems 0..16
                let lo_b = vreinterpretq_s8_u8(vandq_u8(bytes_b, mask)); // sub 2p elems 16..32
                let hi_a = vreinterpretq_s8_u8(vshrq_n_u8(bytes_a, 4)); // sub 2p+1 elems 0..16
                let hi_b = vreinterpretq_s8_u8(vshrq_n_u8(bytes_b, 4)); // sub 2p+1 elems 16..32
                let x0 = ldq_s8(xq.add(b * QK_K + pair * 64));
                let x0b = ldq_s8(xq.add(b * QK_K + pair * 64 + 16));
                let x1 = ldq_s8(xq.add(b * QK_K + pair * 64 + 32));
                let x1b = ldq_s8(xq.add(b * QK_K + pair * 64 + 48));
                // q·x and Σx per sub-block (SDOT against ones for the sum).
                let qd0 = vaddvq_s32(vaddq_s32(vdotq_s32(vdupq_n_s32(0), lo_a, x0), vdotq_s32(vdupq_n_s32(0), lo_b, x0b)));
                let sx0 = vaddvq_s32(vaddq_s32(vdotq_s32(vdupq_n_s32(0), ones, x0), vdotq_s32(vdupq_n_s32(0), ones, x0b)));
                let qd1 = vaddvq_s32(vaddq_s32(vdotq_s32(vdupq_n_s32(0), hi_a, x1), vdotq_s32(vdupq_n_s32(0), hi_b, x1b)));
                let sx1 = vaddvq_s32(vaddq_s32(vdotq_s32(vdupq_n_s32(0), ones, x1), vdotq_s32(vdupq_n_s32(0), ones, x1b)));
                let xs0 = *xs.add(b * (QK_K / QK) + pair * 2);
                let xs1 = *xs.add(b * (QK_K / QK) + pair * 2 + 1);
                // Scalar accumulate is fine here: two affine terms per 64
                // MACs — the SDOTs above are the hot part.
                acc += (d * sc0 as f32 * qd0 as f32 - dmin * m0 as f32 * sx0 as f32) * xs0
                    + (d * sc1 as f32 * qd1 as f32 - dmin * m1 as f32 * sx1 as f32) * xs1;
            }
        }
        acc
    }
}

/// Q4_K rows over `[row_start, row_end)` against int8-quantized activations.
///
/// # Safety
/// `w` = rows of `n_cols/QK_K` Q4_K super-blocks; `xq` = `n_cols` i8;
/// `xs` = `n_cols/QK` f32; `y` = `n_rows` f32; ranges in bounds and disjoint.
#[cfg(target_arch = "aarch64")]
pub unsafe fn matvec_q4_k_sdot_rows(
    w: *const u8,
    xq: *const i8,
    xs: *const f32,
    y: *mut f32,
    row_start: usize,
    row_end: usize,
    n_cols: usize,
) {
    let superblocks = n_cols / QK_K;
    let row_bytes = superblocks * Q4_K_BLOCK_BYTES;
    // SAFETY: caller's contract.
    unsafe {
        for r in row_start..row_end {
            *y.add(r) = sdot_one_row_q4_k(w.add(r * row_bytes), xq, xs, superblocks);
        }
    }
}

/// `y = W·x` for Q4_K weights: quantize the activation once to int8 and run
/// the SDOT row kernel across cores (aarch64); elsewhere the exact generic
/// dequant path. Requires `n_cols % QK_K == 0` (Q4_K super-blocks).
pub fn matvec_q4_k_fast(w: &[u8], x: &[f32], y: &mut [f32], xq: &mut [i8], xs: &mut [f32], n_rows: usize, n_cols: usize) {
    debug_assert_eq!(n_cols % QK_K, 0);
    debug_assert_eq!(x.len(), n_cols);
    debug_assert_eq!(y.len(), n_rows);
    debug_assert!(xq.len() >= n_cols && xs.len() >= n_cols / QK);
    #[cfg(target_arch = "aarch64")]
    {
        let xq = &mut xq[..n_cols];
        let xs = &mut xs[..n_cols / QK];
        quantize_activations_q8(x, xq, xs);
        // SAFETY: `w` holds `n_rows` Q4_K rows of `n_cols/QK_K` super-blocks;
        // `xq`/`xs` are the just-computed quantized activation.
        unsafe {
            crate::arch::aarch64::smp::matvec_q4_k_sdot(w.as_ptr(), xq.as_ptr(), xs.as_ptr(), y.as_mut_ptr(), n_rows, n_cols)
        };
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (&mut *xq, &mut *xs);
        // SAFETY: caller's slices sized per the debug asserts above.
        unsafe { matvec_quant_rows(QT_Q4_K, w.as_ptr(), x.as_ptr(), y.as_mut_ptr(), 0, n_rows, n_cols) };
    }
}

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

/// Unpack 16 bytes of Q2_0 codes (64 elements, element `e` = bit-field `e%4`
/// of byte `e/4`) into four contiguous `int8x16` vectors of `code − 1`
/// (elements [0,16), [16,32), [32,48), [48,64) of the 64-element group).
/// Entirely in vector registers: `vand`/`vshr` extract the four bit-fields,
/// two `vzip` levels interleave them back into element order.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn q2_0_unpack64(b: core::arch::aarch64::uint8x16_t) -> [core::arch::aarch64::int8x16_t; 4] {
    use core::arch::aarch64::{
        vandq_u8, vdupq_n_s8, vdupq_n_u8, vreinterpretq_s8_u8, vshrq_n_u8, vsubq_s8, vzip1q_u8, vzip2q_u8,
    };
    let two = vdupq_n_u8(0x03);
    let one = vdupq_n_s8(1);
    let c0 = vandq_u8(b, two);
    let c1 = vandq_u8(vshrq_n_u8(b, 2), two);
    let c2 = vandq_u8(vshrq_n_u8(b, 4), two);
    let c3 = vshrq_n_u8(b, 6); // already < 4
    // interleave field pairs, then the pairs, to restore element order
    let t0 = vzip1q_u8(c0, c2);
    let t1 = vzip1q_u8(c1, c3);
    let t0h = vzip2q_u8(c0, c2);
    let t1h = vzip2q_u8(c1, c3);
    let e0 = vzip1q_u8(t0, t1); // elements 0..16
    let e1 = vzip2q_u8(t0, t1); // elements 16..32
    let e2 = vzip1q_u8(t0h, t1h); // elements 32..48
    let e3 = vzip2q_u8(t0h, t1h); // elements 48..64
    [
        vsubq_s8(vreinterpretq_s8_u8(e0), one),
        vsubq_s8(vreinterpretq_s8_u8(e1), one),
        vsubq_s8(vreinterpretq_s8_u8(e2), one),
        vsubq_s8(vreinterpretq_s8_u8(e3), one),
    ]
}

/// One Q2_0 row (`n_cols/128` blocks) · int8-quantized activation, NEON SDOT —
/// the fast path for the PrismML ternary `Q2_0` weights (Bonsai-27B). Each
/// 128-element block carries one f16 scale `d` + 128 2-bit codes (`c ∈ 0..3`,
/// value `(c-1)·d`). Per 32-element sub-block `j` (the activation's quant
/// granularity, scale `xs_j`): `d · xs_j · SDOT(c-1, xq_j)`. No per-element
/// scalar work; all loads via the `ldq_*`/`ldp_*` inline-asm helpers (the
/// `+strict-align` rule).
#[cfg(target_arch = "aarch64")]
unsafe fn sdot_one_row_q2_0(row: *const u8, xq: *const i8, xs: *const f32, blocks: usize) -> f32 {
    use core::arch::aarch64::{vaddvq_s32, vdotq_s32, vdupq_n_s32};
    // SAFETY: caller's contract; all loads in-bounds.
    unsafe {
        let mut acc = 0.0f32;
        for b in 0..blocks {
            let base = b * Q2_0_BLOCK_BYTES;
            let d = f16_to_f32(u16::from_le_bytes([*row.add(base), *row.add(base + 1)]));
            let qs = row.add(base + 2);
            // 128 codes = two 16-byte halves → 8 contiguous int8x16 (elements
            // 0..64 then 64..128), i.e. 4 sub-blocks of 32 (2 vectors each).
            let lo = q2_0_unpack64(ldq_u8(qs)); // elements 0..64  (sub-blocks 0,1)
            let hi = q2_0_unpack64(ldq_u8(qs.add(16))); // elements 64..128 (sub-blocks 2,3)
            let s = [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]];
            for j in 0..4 {
                let (x0, x1) = ldp_s8(xq.add(b * QK2_0 + j * 32));
                let dot = vaddvq_s32(vdotq_s32(vdotq_s32(vdupq_n_s32(0), s[2 * j], x0), s[2 * j + 1], x1));
                acc += d * *xs.add(b * (QK2_0 / QK) + j) * dot as f32;
            }
        }
        acc
    }
}

/// Q2_0 rows over `[row_start, row_end)` against int8-quantized activations.
///
/// # Safety
/// `w` = rows of `n_cols/128` Q2_0 blocks; `xq` = `n_cols` i8; `xs` =
/// `n_cols/QK` f32; `y` = `n_rows` f32; ranges in bounds and disjoint.
#[cfg(target_arch = "aarch64")]
pub unsafe fn matvec_q2_0_sdot_rows(
    w: *const u8,
    xq: *const i8,
    xs: *const f32,
    y: *mut f32,
    row_start: usize,
    row_end: usize,
    n_cols: usize,
) {
    let blocks = n_cols / QK2_0;
    let row_bytes = blocks * Q2_0_BLOCK_BYTES;
    // SAFETY: caller's contract.
    unsafe {
        for r in row_start..row_end {
            *y.add(r) = sdot_one_row_q2_0(w.add(r * row_bytes), xq, xs, blocks);
        }
    }
}

/// `y = W·x` for Q2_0 weights: quantize the activation once to int8 and run the
/// SDOT row kernel across cores (aarch64); elsewhere the generic exact dequant
/// path. Requires `n_cols % 128 == 0` (Q2_0 blocks).
pub fn matvec_q2_0_fast(w: &[u8], x: &[f32], y: &mut [f32], xq: &mut [i8], xs: &mut [f32], n_rows: usize, n_cols: usize) {
    debug_assert_eq!(n_cols % QK2_0, 0);
    debug_assert_eq!(x.len(), n_cols);
    debug_assert_eq!(y.len(), n_rows);
    debug_assert!(xq.len() >= n_cols && xs.len() >= n_cols / QK);
    #[cfg(target_arch = "aarch64")]
    {
        let xq = &mut xq[..n_cols];
        let xs = &mut xs[..n_cols / QK];
        quantize_activations_q8(x, xq, xs);
        // SAFETY: `w` holds `n_rows` Q2_0 rows of `n_cols/128` blocks; `xq`/`xs`
        // are the just-computed quantized activation; `[0, n_rows)` in bounds.
        unsafe {
            crate::arch::aarch64::smp::matvec_q2_0_sdot(w.as_ptr(), xq.as_ptr(), xs.as_ptr(), y.as_mut_ptr(), n_rows, n_cols)
        };
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (&mut *xq, &mut *xs);
        // SAFETY: caller's slices sized per the debug asserts above.
        unsafe { matvec_quant_rows(QT_Q2_0, w.as_ptr(), x.as_ptr(), y.as_mut_ptr(), 0, n_rows, n_cols) };
    }
}

/// One Q1_0 row (`n_cols/128` blocks) · int8-quantized activation, NEON SDOT —
/// the fast path for PrismML binary `Q1_0` weights (Bonsai-27B 1-bit). Each
/// 128-element block is one f16 scale `d` + 128 sign bits (value `bit ? +d :
/// −d`). Per 32-element sub-block `j` (activation scale `xs_j`):
/// `d · xs_j · SDOT(sign, xq_j)`, `sign ∈ {−1,+1}`.
///
/// The sign expand is fully vectorized: all 16 sign bytes are loaded **once**
/// per block (`ldq_u8`), then each 16-lane group is `vqtbl1q` byte-broadcast
/// ([`Q1_0_TBL_IDX`]), `vtst`-tested against the per-lane bit mask
/// ([`Q1_0_BITS`], LSB-first — matching `dequant_q1_0_block`) and `vbsl`
/// selected to ±1: 3 vector ops per 16 elements, no per-element scalar work.
/// Loads via `ldq_*`/`ldp_*` (the `+strict-align` rule).
#[cfg(target_arch = "aarch64")]
unsafe fn sdot_one_row_q1_0(row: *const u8, xq: *const i8, xs: *const f32, blocks: usize) -> f32 {
    use core::arch::aarch64::{
        uint8x16_t, vaddvq_s32, vbslq_s8, vdotq_s32, vdupq_n_s32, vdupq_n_s8, vqtbl1q_u8, vtstq_u8,
    };
    // SAFETY: caller's contract; all loads in-bounds.
    unsafe {
        let idx: [uint8x16_t; 8] = core::array::from_fn(|k| ldq_u8(Q1_0_TBL_IDX[k].0.as_ptr()));
        let bits = ldq_u8(Q1_0_BITS.0.as_ptr());
        let plus = vdupq_n_s8(1);
        let minus = vdupq_n_s8(-1);
        let mut acc = 0.0f32;
        for b in 0..blocks {
            let base = b * Q1_0_BLOCK_BYTES;
            let d = f16_to_f32(u16::from_le_bytes([*row.add(base), *row.add(base + 1)]));
            let bytes = ldq_u8(row.add(base + 2)); // all 128 sign bits at once
            let mut block_acc = 0.0f32;
            for j in 0..4 {
                // sub-block j = lane groups 2j, 2j+1 (elements 32j..32j+32)
                let s0 = vbslq_s8(vtstq_u8(vqtbl1q_u8(bytes, idx[2 * j]), bits), plus, minus);
                let s1 = vbslq_s8(vtstq_u8(vqtbl1q_u8(bytes, idx[2 * j + 1]), bits), plus, minus);
                let (x0, x1) = ldp_s8(xq.add(b * QK1_0 + j * 32));
                let dot = vaddvq_s32(vdotq_s32(vdotq_s32(vdupq_n_s32(0), s0, x0), s1, x1));
                block_acc += *xs.add(b * (QK1_0 / QK) + j) * dot as f32;
            }
            acc += d * block_acc;
        }
        acc
    }
}

/// Q1_0 rows over `[row_start, row_end)` against int8-quantized activations.
///
/// # Safety
/// `w` = rows of `n_cols/128` Q1_0 blocks; `xq` = `n_cols` i8; `xs` =
/// `n_cols/QK` f32; `y` = `n_rows` f32; ranges in bounds and disjoint.
#[cfg(target_arch = "aarch64")]
pub unsafe fn matvec_q1_0_sdot_rows(
    w: *const u8,
    xq: *const i8,
    xs: *const f32,
    y: *mut f32,
    row_start: usize,
    row_end: usize,
    n_cols: usize,
) {
    let blocks = n_cols / QK1_0;
    let row_bytes = blocks * Q1_0_BLOCK_BYTES;
    // SAFETY: caller's contract.
    unsafe {
        for r in row_start..row_end {
            *y.add(r) = sdot_one_row_q1_0(w.add(r * row_bytes), xq, xs, blocks);
        }
    }
}

/// `y = W·x` for Q1_0 weights: quantize the activation once to int8 and run the
/// SDOT row kernel across cores (aarch64); elsewhere the generic exact dequant
/// path. Requires `n_cols % 128 == 0` (Q1_0 blocks).
pub fn matvec_q1_0_fast(w: &[u8], x: &[f32], y: &mut [f32], xq: &mut [i8], xs: &mut [f32], n_rows: usize, n_cols: usize) {
    debug_assert_eq!(n_cols % QK1_0, 0);
    debug_assert_eq!(x.len(), n_cols);
    debug_assert_eq!(y.len(), n_rows);
    debug_assert!(xq.len() >= n_cols && xs.len() >= n_cols / QK);
    #[cfg(target_arch = "aarch64")]
    {
        let xq = &mut xq[..n_cols];
        let xs = &mut xs[..n_cols / QK];
        quantize_activations_q8(x, xq, xs);
        // SAFETY: `w` holds `n_rows` Q1_0 rows of `n_cols/128` blocks; `xq`/`xs`
        // are the just-computed quantized activation; `[0, n_rows)` in bounds.
        unsafe {
            crate::arch::aarch64::smp::matvec_q1_0_sdot(w.as_ptr(), xq.as_ptr(), xs.as_ptr(), y.as_mut_ptr(), n_rows, n_cols)
        };
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (&mut *xq, &mut *xs);
        // SAFETY: caller's slices sized per the debug asserts above.
        unsafe { matvec_quant_rows(QT_Q1_0, w.as_ptr(), x.as_ptr(), y.as_mut_ptr(), 0, n_rows, n_cols) };
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

/// **Weight-stationary batched** `Q1_0` × `int8`-activation matmul over rows
/// `[row_start, row_end)` — the Q1_0 analog of [`matmul_q8_0_sdot_rows`], the
/// kernel behind batched prefill for the 1-bit Bonsai. Per weight block the
/// 128 sign bits are loaded + expanded to ±1 **once per activation tile**
/// (`vqtbl1q`/`vtst`/`vbsl`, see [`sdot_one_row_q1_0`]) and SDOTed against
/// every activation in the tile, so both the weight bytes *and* the expand
/// work are amortized across the batch. `y` is `[m * n_rows + r]`.
///
/// # Safety
/// `w` = `n_rows` `Q1_0` rows of `n_cols/128` blocks; `xq`/`xs` = `m_count`
/// activations of `n_cols` `i8` / `n_cols/QK` `f32`; `y` = `m_count * n_rows`
/// `f32`; `row_start <= row_end <= n_rows`; `n_cols % 128 == 0`. Rows are
/// written disjointly. Requires `dotprod`.
#[cfg(target_arch = "aarch64")]
pub unsafe fn matmul_q1_0_sdot_rows(
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
        int8x16_t, uint8x16_t, vaddvq_f32, vbslq_s8, vcvtq_f32_s32, vdotq_s32, vdupq_n_f32, vdupq_n_s32,
        vdupq_n_s8, vfmaq_n_f32, vqtbl1q_u8, vtstq_u8,
    };
    if m_count == 1 {
        // SAFETY: caller's contract (matvec has better ILP for one column).
        unsafe { matvec_q1_0_sdot_rows(w, xq, xs, y, row_start, row_end, n_cols) };
        return;
    }
    let blocks = n_cols / QK1_0;
    let row_bytes = blocks * Q1_0_BLOCK_BYTES;
    let nb = n_cols / QK; // activation scales per activation (32-wide blocks)
    const MT: usize = 4;
    // SAFETY: caller's contract; all loads in-bounds, `y` rows disjoint.
    unsafe {
        let idx: [uint8x16_t; 8] = core::array::from_fn(|k| ldq_u8(Q1_0_TBL_IDX[k].0.as_ptr()));
        let bits = ldq_u8(Q1_0_BITS.0.as_ptr());
        let plus = vdupq_n_s8(1);
        let minus = vdupq_n_s8(-1);
        for r in row_start..row_end {
            let row = w.add(r * row_bytes);
            let mut m0 = 0;
            while m0 < m_count {
                let mt = core::cmp::min(MT, m_count - m0);
                let mut acc = [vdupq_n_f32(0.0); MT];
                for b in 0..blocks {
                    let base = b * Q1_0_BLOCK_BYTES;
                    let d = f16_to_f32(u16::from_le_bytes([*row.add(base), *row.add(base + 1)]));
                    let bytes = ldq_u8(row.add(base + 2));
                    // Expand the block's 128 signs once for the whole tile.
                    let mut s = [vdupq_n_s8(0); 8];
                    let mut k = 0;
                    while k < 8 {
                        s[k] = vbslq_s8(vtstq_u8(vqtbl1q_u8(bytes, idx[k]), bits), plus, minus);
                        k += 1;
                    }
                    for mm in 0..mt {
                        let mi = m0 + mm;
                        let xqp = xq.add(mi * n_cols + b * QK1_0);
                        let xsp = xs.add(mi * nb + b * (QK1_0 / QK));
                        for j in 0..4 {
                            let (x0, x1) = ldp_s8(xqp.add(j * 32));
                            let a: int8x16_t = s[2 * j];
                            let dot = vdotq_s32(vdotq_s32(vdupq_n_s32(0), a, x0), s[2 * j + 1], x1);
                            acc[mm] = vfmaq_n_f32(acc[mm], vcvtq_f32_s32(dot), d * *xsp.add(j));
                        }
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

/// **Weight-stationary batched** `Q2_0` × `int8`-activation matmul over rows
/// `[row_start, row_end)` — the ternary analog of [`matmul_q8_0_sdot_rows`],
/// batched prefill for Ternary-Bonsai. Per weight block the 128 2-bit codes
/// are unpacked to `code−1` int8 **once per activation tile**
/// ([`q2_0_unpack64`]) and SDOTed against every activation in the tile.
/// `y` is `[m * n_rows + r]`.
///
/// # Safety
/// As [`matmul_q1_0_sdot_rows`] with `Q2_0` rows (`n_cols/128` 34-byte blocks).
#[cfg(target_arch = "aarch64")]
pub unsafe fn matmul_q2_0_sdot_rows(
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
    use core::arch::aarch64::{vaddvq_f32, vcvtq_f32_s32, vdotq_s32, vdupq_n_f32, vdupq_n_s32, vfmaq_n_f32};
    if m_count == 1 {
        // SAFETY: caller's contract (matvec has better ILP for one column).
        unsafe { matvec_q2_0_sdot_rows(w, xq, xs, y, row_start, row_end, n_cols) };
        return;
    }
    let blocks = n_cols / QK2_0;
    let row_bytes = blocks * Q2_0_BLOCK_BYTES;
    let nb = n_cols / QK;
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
                    let base = b * Q2_0_BLOCK_BYTES;
                    let d = f16_to_f32(u16::from_le_bytes([*row.add(base), *row.add(base + 1)]));
                    let qs = row.add(base + 2);
                    // Unpack the block's 128 codes once for the whole tile.
                    let lo = q2_0_unpack64(ldq_u8(qs));
                    let hi = q2_0_unpack64(ldq_u8(qs.add(16)));
                    let s = [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]];
                    for mm in 0..mt {
                        let mi = m0 + mm;
                        let xqp = xq.add(mi * n_cols + b * QK2_0);
                        let xsp = xs.add(mi * nb + b * (QK2_0 / QK));
                        for j in 0..4 {
                            let (x0, x1) = ldp_s8(xqp.add(j * 32));
                            let dot = vdotq_s32(vdotq_s32(vdupq_n_s32(0), s[2 * j], x0), s[2 * j + 1], x1);
                            acc[mm] = vfmaq_n_f32(acc[mm], vcvtq_f32_s32(dot), d * *xsp.add(j));
                        }
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

/// GGML quant type codes (the set Chitti dequantizes), matching the GGUF
/// on-disk `ggml_type` values so a tensor's type can be carried straight
/// through from the header.
pub const QT_F16: u32 = 1;
pub const QT_Q4_0: u32 = 2;
pub const QT_Q4_1: u32 = 3;
pub const QT_Q5_0: u32 = 6;
pub const QT_Q5_1: u32 = 7;
pub const QT_Q8_0: u32 = 8;
pub const QT_Q2_K: u32 = 10;
pub const QT_Q3_K: u32 = 11;
pub const QT_Q4_K: u32 = 12;
pub const QT_Q5_K: u32 = 13;
pub const QT_Q6_K: u32 = 14;
pub const QT_Q8_K: u32 = 15;
pub const QT_IQ2_XXS: u32 = 16;
pub const QT_IQ2_XS: u32 = 17;
pub const QT_IQ3_XXS: u32 = 18;
pub const QT_IQ4_NL: u32 = 20;
pub const QT_IQ3_S: u32 = 21;
pub const QT_IQ2_S: u32 = 22;
pub const QT_IQ4_XS: u32 = 23;
pub const QT_BF16: u32 = 30;
/// PrismML binary pack (`Bonsai-27B` 1-bit); 128-elem 1-bit blocks.
pub const QT_Q1_0: u32 = 41;
/// PrismML ternary pack (`Ternary-Bonsai`/`Bonsai-27B`); 128-elem 2-bit blocks.
pub const QT_Q2_0: u32 = 42;

/// Bytes per quantization block and elements per block for a quant type.
pub fn block_layout(qt: u32) -> (usize, usize) {
    match qt {
        QT_F16 => (F16_BLOCK_BYTES, QK),
        QT_Q4_0 => (Q4_0_BLOCK_BYTES, QK),
        QT_Q4_1 => (Q4_1_BLOCK_BYTES, QK),
        QT_Q5_0 => (Q5_0_BLOCK_BYTES, QK),
        QT_Q5_1 => (Q5_1_BLOCK_BYTES, QK),
        QT_Q8_0 => (Q8_0_BLOCK_BYTES, QK),
        QT_Q2_K => (Q2_K_BLOCK_BYTES, QK_K),
        QT_Q3_K => (Q3_K_BLOCK_BYTES, QK_K),
        QT_Q4_K => (Q4_K_BLOCK_BYTES, QK_K),
        QT_Q5_K => (Q5_K_BLOCK_BYTES, QK_K),
        QT_Q6_K => (Q6_K_BLOCK_BYTES, QK_K),
        QT_Q8_K => (Q8_K_BLOCK_BYTES, QK_K),
        QT_IQ2_XXS => (IQ2_XXS_BLOCK_BYTES, QK_K),
        QT_IQ2_XS => (IQ2_XS_BLOCK_BYTES, QK_K),
        QT_IQ2_S => (IQ2_S_BLOCK_BYTES, QK_K),
        QT_IQ3_XXS => (IQ3_XXS_BLOCK_BYTES, QK_K),
        QT_IQ3_S => (IQ3_S_BLOCK_BYTES, QK_K),
        QT_IQ4_NL => (IQ4_NL_BLOCK_BYTES, QK),
        QT_IQ4_XS => (IQ4_XS_BLOCK_BYTES, QK_K),
        QT_BF16 => (BF16_BLOCK_BYTES, QK),
        QT_Q1_0 => (Q1_0_BLOCK_BYTES, QK1_0),
        QT_Q2_0 => (Q2_0_BLOCK_BYTES, QK2_0),
        _ => (0, 0),
    }
}

/// Dequantize one block of quant type `qt` into `out` (length = the type's
/// block element count).
pub fn dequant_block(qt: u32, block: &[u8], out: &mut [f32]) {
    match qt {
        QT_F16 => dequant_f16_block(block, out),
        QT_Q4_0 => dequant_q4_0_block(block, out),
        QT_Q4_1 => dequant_q4_1_block(block, out),
        QT_Q5_0 => dequant_q5_0_block(block, out),
        QT_Q5_1 => dequant_q5_1_block(block, out),
        QT_Q8_0 => dequant_q8_0_block(block, out),
        QT_Q2_K => dequant_q2_k_block(block, out),
        QT_Q3_K => dequant_q3_k_block(block, out),
        QT_Q4_K => dequant_q4_k_block(block, out),
        QT_Q5_K => dequant_q5_k_block(block, out),
        QT_Q6_K => dequant_q6_k_block(block, out),
        QT_Q8_K => dequant_q8_k_block(block, out),
        QT_IQ2_XXS => dequant_iq2_xxs_block(block, out),
        QT_IQ2_XS => dequant_iq2_xs_block(block, out),
        QT_IQ2_S => dequant_iq2_s_block(block, out),
        QT_IQ3_XXS => dequant_iq3_xxs_block(block, out),
        QT_IQ3_S => dequant_iq3_s_block(block, out),
        QT_IQ4_NL => dequant_iq4_nl_block(block, out),
        QT_IQ4_XS => dequant_iq4_xs_block(block, out),
        QT_BF16 => dequant_bf16_block(block, out),
        QT_Q1_0 => dequant_q1_0_block(block, out),
        QT_Q2_0 => dequant_q2_0_block(block, out),
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

/// NeoX RoPE with per-frequency factors (llama.cpp `ggml_rope_ext` with
/// `freq_factors`, defaults otherwise): angle_i = pos · base^(-2i/d) / ff[i].
/// Gemma-4's proportional RoPE stores the divisors in `rope_freqs.weight`
/// (global layers); `None` reduces exactly to [`rope`].
pub fn rope_ext(vec: &mut [f32], pos: usize, head_dim: usize, theta_base: f32, freq_factors: Option<&[f32]>) {
    debug_assert_eq!(vec.len(), head_dim);
    let half = head_dim / 2;
    for i in 0..half {
        let mut freq = 1.0 / powf(theta_base, (2 * i) as f32 / head_dim as f32);
        if let Some(ff) = freq_factors {
            freq /= ff[i];
        }
        let angle = pos as f32 * freq;
        let (sin, cos) = (sinf(angle), cosf(angle));
        let a = vec[i];
        let b = vec[i + half];
        vec[i] = a * cos - b * sin;
        vec[i + half] = b * cos + a * sin;
    }
}

/// tanh via [`expf`]: (e^{2x} − 1)/(e^{2x} + 1), saturating at |x| > 10 —
/// the final-logit softcap (`ggml_tanh`) and the GELU approximation use it.
pub fn tanhf(x: f32) -> f32 {
    if x > 10.0 {
        return 1.0;
    }
    if x < -10.0 {
        return -1.0;
    }
    let e = expf(2.0 * x);
    (e - 1.0) / (e + 1.0)
}

/// GELU, the tanh approximation ggml uses (`ggml_gelu`):
/// 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³))).
pub fn gelu(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_56;
    0.5 * x * (1.0 + tanhf(SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)))
}

/// `out[i] = gelu(gate[i]) * up[i]` — the Gemma FFN activation (parallel
/// gate/up, GELU-gated; the SwiGLU counterpart is [`silu_mul`]).
pub fn gelu_mul(gate: &[f32], up: &[f32], out: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len());
    debug_assert_eq!(gate.len(), out.len());
    for i in 0..gate.len() {
        out[i] = gelu(gate[i]) * up[i];
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
    use alloc::vec::Vec;

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

    /// Tiny deterministic LCG so quant blocks / activations are reproducible
    /// without an external reference file (seeds fixed per the repo rule).
    fn lcg(seed: &mut u32) -> u32 {
        *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        *seed
    }

    /// f16 bit pattern for a small set of exact values used to build blocks.
    /// 1.0 = 0x3C00, 0.5 = 0x3800, 2.0 = 0x4000 (IEEE 754 half).
    const F16_ONE: [u8; 2] = [0x00, 0x3C];
    const F16_HALF: [u8; 2] = [0x00, 0x38];

    /// A Q8_0 block is `[d: f16][qs: 32 x i8]`; element i = d * qs[i].
    /// Handcrafted block with d=0.5 and qs = -16..16, so the expected values
    /// are exact halves — pins both the scale decode and the sign handling.
    #[test_case]
    fn dequant_q8_0_layout_and_scale() {
        let mut block = [0u8; Q8_0_BLOCK_BYTES];
        block[..2].copy_from_slice(&F16_HALF);
        for i in 0..QK {
            block[2 + i] = (i as i32 - 16) as i8 as u8;
        }
        let mut out = [0.0f32; QK];
        dequant_q8_0_block(&block, &mut out);
        for i in 0..QK {
            let want = 0.5 * (i as f32 - 16.0);
            assert!((out[i] - want).abs() < 1e-6, "q8_0 idx {i}: got {} want {want}", out[i]);
        }
    }

    /// A Q4_0 block is `[d: f16][qs: 16 bytes]`; low nibble of byte j is
    /// element j, high nibble element j+16, each nibble-8 then scaled by d.
    /// d=1.0 with distinct nibbles pins the nibble order (a swapped layout
    /// fails by whole integers, not epsilon).
    #[test_case]
    fn dequant_q4_0_nibble_order() {
        let mut block = [0u8; Q4_0_BLOCK_BYTES];
        block[..2].copy_from_slice(&F16_ONE);
        for j in 0..16 {
            let lo = (j % 16) as u8; // element j     = lo - 8
            let hi = 15 - (j % 16) as u8; // element j+16  = hi - 8
            block[2 + j] = (hi << 4) | lo;
        }
        let mut out = [0.0f32; QK];
        dequant_q4_0_block(&block, &mut out);
        for j in 0..16 {
            let want_lo = j as f32 - 8.0;
            let want_hi = (15 - j) as f32 - 8.0;
            assert!((out[j] - want_lo).abs() < 1e-6, "q4_0 lo idx {j}: got {} want {want_lo}", out[j]);
            assert!((out[16 + j] - want_hi).abs() < 1e-6, "q4_0 hi idx {j}: got {} want {want_hi}", out[16 + j]);
        }
    }

    /// A Q2_0 block is `[d: f16][qs: 32 bytes]`, four 2-bit codes per byte
    /// (low bits first); code `c` dequantizes to `(c-1)*d`. With d=1.0 and the
    /// codes cycling 0,1,2,3 the expected values are exactly -1,0,1,2 — pinning
    /// both the code→value map and the little-endian bit-packing order (a
    /// swapped order fails by whole integers, not epsilon).
    #[test_case]
    fn dequant_q2_0_codes_and_packing() {
        let mut block = [0u8; Q2_0_BLOCK_BYTES];
        block[..2].copy_from_slice(&F16_ONE);
        for byte in block[2..].iter_mut() {
            // codes for the four elements in this byte: 0,1,2,3 (low→high)
            *byte = 0 | (1 << 2) | (2 << 4) | (3 << 6); // 0xE4
        }
        let mut out = [0.0f32; QK2_0];
        dequant_q2_0_block(&block, &mut out);
        for j in 0..QK2_0 {
            let want = (j % 4) as f32 - 1.0; // 0,1,2,3 -> -1,0,1,2
            assert!((out[j] - want).abs() < 1e-6, "q2_0 idx {j}: got {} want {want}", out[j]);
        }
        // block_layout / dequant_block dispatch must agree with the direct fn.
        assert_eq!(block_layout(QT_Q2_0), (Q2_0_BLOCK_BYTES, QK2_0));
        let mut out2 = [0.0f32; QK2_0];
        dequant_block(QT_Q2_0, &block, &mut out2);
        assert_eq!(out, out2);
    }

    /// A Q1_0 block is `[d: f16][qs: 16 bytes]`, one sign bit per element
    /// (LSB-first): `bit → +d`, `!bit → −d`. d=1.0 with an alternating bit
    /// pattern (0x55 = 0b01010101) pins the bit order and sign map — a swapped
    /// endianness or inverted sign fails by whole units.
    #[test_case]
    fn dequant_q1_0_bits_and_sign() {
        let mut block = [0u8; Q1_0_BLOCK_BYTES];
        block[..2].copy_from_slice(&F16_ONE);
        for byte in block[2..].iter_mut() {
            *byte = 0x55; // bits: 1,0,1,0,... (LSB first) → +1,-1,+1,-1,...
        }
        let mut out = [0.0f32; QK1_0];
        dequant_q1_0_block(&block, &mut out);
        for j in 0..QK1_0 {
            let want = if j % 2 == 0 { 1.0 } else { -1.0 };
            assert!((out[j] - want).abs() < 1e-6, "q1_0 idx {j}: got {} want {want}", out[j]);
        }
        assert_eq!(block_layout(QT_Q1_0), (Q1_0_BLOCK_BYTES, QK1_0));
    }

    /// The Q1_0 NEON SDOT row kernel must match an f64 dot of the dequantized
    /// weight row against the dequantized int8 activation (oracle-free). Pins
    /// the in-register bit-expand (`vtst`/`vbsl`) + per-sub-block scaling.
    /// aarch64-only (NEON); the x86 unit runner skips it.
    #[cfg(target_arch = "aarch64")]
    #[test_case]
    fn sdot_q1_0_matches_dequant_reference() {
        let (blocks, n_cols) = (2usize, 256usize);
        let mut seed = 0x1B17u32;
        let mut row = Vec::new();
        for _ in 0..blocks {
            row.extend_from_slice(&F16_HALF); // d = 0.5
            for _ in 0..QK1_0 / 8 {
                row.push((lcg(&mut seed) & 0xff) as u8); // random sign bits
            }
        }
        let x: Vec<f32> = (0..n_cols).map(|_| (lcg(&mut seed) % 1000) as f32 / 500.0 - 1.0).collect();
        let mut xq = alloc::vec![0i8; n_cols];
        let mut xs = alloc::vec![0.0f32; n_cols / QK];
        quantize_activations_q8(&x, &mut xq, &mut xs);

        let mut want = 0.0f64;
        let mut dq = [0.0f32; QK1_0];
        for b in 0..blocks {
            dequant_q1_0_block(&row[b * Q1_0_BLOCK_BYTES..(b + 1) * Q1_0_BLOCK_BYTES], &mut dq);
            for i in 0..QK1_0 {
                let col = b * QK1_0 + i;
                want += dq[i] as f64 * (xs[col / QK] as f64 * xq[col] as f64);
            }
        }
        // SAFETY: row = `blocks` Q1_0 blocks; xq/xs sized for `n_cols`.
        let got = unsafe { sdot_one_row_q1_0(row.as_ptr(), xq.as_ptr(), xs.as_ptr(), blocks) };
        assert!((got as f64 - want).abs() <= 1e-2 + 1e-3 * want.abs(), "sdot_q1_0: got {got} want {want}");
    }

    /// The batched (weight-stationary) Q1_0/Q2_0 matmuls must agree with the
    /// per-activation matvec row kernels — same math, different loop order —
    /// including the m % MT tail tile. aarch64-only (NEON kernels).
    #[cfg(target_arch = "aarch64")]
    #[test_case]
    fn matmul_q1_0_q2_0_match_matvec() {
        let (rows, cols, m) = (5usize, 256usize, 3usize); // odd m exercises the tail tile
        let mut seed = 0xBA7Cu32;
        // Q1_0 weights.
        let mut w1 = Vec::new();
        for _ in 0..rows * (cols / QK1_0) {
            w1.extend_from_slice(&F16_HALF);
            for _ in 0..QK1_0 / 8 {
                w1.push((lcg(&mut seed) & 0xff) as u8);
            }
        }
        // Q2_0 weights.
        let mut w2 = Vec::new();
        for _ in 0..rows * (cols / QK2_0) {
            w2.extend_from_slice(&F16_HALF);
            for _ in 0..QK2_0 / 4 {
                w2.push((lcg(&mut seed) & 0xff) as u8);
            }
        }
        // m quantized activations, packed with stride cols / cols/QK.
        let mut xq = alloc::vec![0i8; m * cols];
        let mut xs = alloc::vec![0.0f32; m * (cols / QK)];
        for mi in 0..m {
            let x: Vec<f32> = (0..cols).map(|_| (lcg(&mut seed) % 1000) as f32 / 500.0 - 1.0).collect();
            quantize_activations_q8(&x, &mut xq[mi * cols..(mi + 1) * cols], &mut xs[mi * (cols / QK)..(mi + 1) * (cols / QK)]);
        }
        let mut got1 = alloc::vec![0.0f32; m * rows];
        let mut got2 = alloc::vec![0.0f32; m * rows];
        // SAFETY: buffers sized per the kernel contracts above.
        unsafe {
            matmul_q1_0_sdot_rows(w1.as_ptr(), xq.as_ptr(), xs.as_ptr(), got1.as_mut_ptr(), m, rows, 0, rows, cols);
            matmul_q2_0_sdot_rows(w2.as_ptr(), xq.as_ptr(), xs.as_ptr(), got2.as_mut_ptr(), m, rows, 0, rows, cols);
        }
        for mi in 0..m {
            let mut want1 = alloc::vec![0.0f32; rows];
            let mut want2 = alloc::vec![0.0f32; rows];
            // SAFETY: same contracts, one activation at a time.
            unsafe {
                matvec_q1_0_sdot_rows(w1.as_ptr(), xq.as_ptr().add(mi * cols), xs.as_ptr().add(mi * (cols / QK)), want1.as_mut_ptr(), 0, rows, cols);
                matvec_q2_0_sdot_rows(w2.as_ptr(), xq.as_ptr().add(mi * cols), xs.as_ptr().add(mi * (cols / QK)), want2.as_mut_ptr(), 0, rows, cols);
            }
            for r in 0..rows {
                assert!((got1[mi * rows + r] - want1[r]).abs() < 1e-3, "q1_0 m{mi} r{r}: {} vs {}", got1[mi * rows + r], want1[r]);
                assert!((got2[mi * rows + r] - want2[r]).abs() < 1e-3, "q2_0 m{mi} r{r}: {} vs {}", got2[mi * rows + r], want2[r]);
            }
        }
    }

    /// The Q2_0 NEON SDOT row kernel must match an f64 dot of the dequantized
    /// weight row against the dequantized int8 activation (its own reference,
    /// oracle-free). This exercises the vectorized 2-bit unpack + `−1` bias +
    /// per-sub-block scale — the parts a swapped zip/shift would corrupt.
    /// aarch64-only (the kernel is NEON); the x86 unit runner skips it, boot +
    /// cortexdiff cover the running path.
    #[cfg(target_arch = "aarch64")]
    #[test_case]
    fn sdot_q2_0_matches_dequant_reference() {
        let (blocks, n_cols) = (2usize, 256usize); // 2 Q2_0 blocks of 128
        let mut seed = 0x51A0u32;
        let mut row = Vec::new();
        for _ in 0..blocks {
            row.extend_from_slice(&F16_HALF); // d = 0.5
            for _ in 0..QK2_0 / 4 {
                row.push((lcg(&mut seed) & 0xff) as u8); // random 2-bit codes
            }
        }
        let x: Vec<f32> = (0..n_cols).map(|_| (lcg(&mut seed) % 1000) as f32 / 500.0 - 1.0).collect();
        let mut xq = alloc::vec![0i8; n_cols];
        let mut xs = alloc::vec![0.0f32; n_cols / QK];
        quantize_activations_q8(&x, &mut xq, &mut xs);

        // Reference: dequant each weight block, dot vs the dequantized activation.
        let mut want = 0.0f64;
        let mut dq = [0.0f32; QK2_0];
        for b in 0..blocks {
            dequant_q2_0_block(&row[b * Q2_0_BLOCK_BYTES..(b + 1) * Q2_0_BLOCK_BYTES], &mut dq);
            for i in 0..QK2_0 {
                let col = b * QK2_0 + i;
                let xdeq = xs[col / QK] as f64 * xq[col] as f64;
                want += dq[i] as f64 * xdeq;
            }
        }
        // SAFETY: row = `blocks` Q2_0 blocks; xq/xs sized for `n_cols`.
        let got = unsafe { sdot_one_row_q2_0(row.as_ptr(), xq.as_ptr(), xs.as_ptr(), blocks) };
        assert!(
            (got as f64 - want).abs() <= 1e-2 + 1e-3 * want.abs(),
            "sdot_q2_0: got {got} want {want}"
        );
    }

    /// Build `rows x cols` of random Q8_0 blocks + a random activation, and
    /// check the fast matvec against an f64 accumulation over the *dequantized*
    /// values (dequant itself is pinned by the layout tests above) — an
    /// oracle-free reference that catches indexing/blocking errors.
    #[test_case]
    fn matvec_q8_0_matches_f64_reference() {
        let (rows, cols) = (4usize, 64usize);
        let blocks_per_row = cols / QK;
        let mut seed = 0xC0FFEEu32;
        let mut w = Vec::new();
        for _ in 0..rows * blocks_per_row {
            let mut block = [0u8; Q8_0_BLOCK_BYTES];
            block[..2].copy_from_slice(&F16_HALF);
            for i in 0..QK {
                block[2 + i] = (lcg(&mut seed) % 255) as u8;
            }
            w.extend_from_slice(&block);
        }
        let x: Vec<f32> = (0..cols).map(|_| (lcg(&mut seed) % 1000) as f32 / 500.0 - 1.0).collect();

        let mut y = alloc::vec![0.0f32; rows];
        matvec_q8_0(&w, &x, &mut y, rows, cols);

        for r in 0..rows {
            let mut want = 0.0f64;
            for b in 0..blocks_per_row {
                let mut dq = [0.0f32; QK];
                dequant_q8_0_block(&w[(r * blocks_per_row + b) * Q8_0_BLOCK_BYTES..][..Q8_0_BLOCK_BYTES], &mut dq);
                for i in 0..QK {
                    want += dq[i] as f64 * x[b * QK + i] as f64;
                }
            }
            assert!(
                (y[r] as f64 - want).abs() <= 1e-2 + 1e-3 * want.abs(),
                "matvec_q8_0 row {r}: got {} want {want}",
                y[r]
            );
        }
    }

    /// Same construction for Q4_0 (nibble-packed weights).
    #[test_case]
    fn matvec_q4_0_matches_f64_reference() {
        let (rows, cols) = (4usize, 64usize);
        let blocks_per_row = cols / QK;
        let mut seed = 0xBEEFu32;
        let mut w = Vec::new();
        for _ in 0..rows * blocks_per_row {
            let mut block = [0u8; Q4_0_BLOCK_BYTES];
            block[..2].copy_from_slice(&F16_ONE);
            for j in 0..16 {
                block[2 + j] = (lcg(&mut seed) % 255) as u8;
            }
            w.extend_from_slice(&block);
        }
        let x: Vec<f32> = (0..cols).map(|_| (lcg(&mut seed) % 1000) as f32 / 500.0 - 1.0).collect();

        let mut y = alloc::vec![0.0f32; rows];
        matvec_q4_0(&w, &x, &mut y, rows, cols);

        for r in 0..rows {
            let mut want = 0.0f64;
            for b in 0..blocks_per_row {
                let mut dq = [0.0f32; QK];
                dequant_q4_0_block(&w[(r * blocks_per_row + b) * Q4_0_BLOCK_BYTES..][..Q4_0_BLOCK_BYTES], &mut dq);
                for i in 0..QK {
                    want += dq[i] as f64 * x[b * QK + i] as f64;
                }
            }
            assert!(
                (y[r] as f64 - want).abs() <= 1e-2 + 1e-3 * want.abs(),
                "matvec_q4_0 row {r}: got {} want {want}",
                y[r]
            );
        }
    }

    /// rmsnorm(x, w) = x / sqrt(mean(x²)+eps) * w — for constant x=c the rms
    /// is |c|, so the output is exactly sign(c)*w (as eps→0). A second,
    /// non-constant case checks against an explicit f64 evaluation.
    #[test_case]
    fn rmsnorm_closed_form() {
        let x = [2.0f32; 8];
        let w = [3.0f32, -1.0, 0.5, 2.0, 1.0, 4.0, -2.0, 0.25];
        let mut out = [0.0f32; 8];
        rmsnorm(&x, &w, 1e-12, &mut out);
        for i in 0..8 {
            assert!((out[i] - w[i]).abs() < 1e-4, "rmsnorm const idx {i}: got {} want {}", out[i], w[i]);
        }

        let x2 = [1.0f32, -2.0, 3.0, -4.0];
        let w2 = [1.0f32, 1.0, 1.0, 1.0];
        let mut out2 = [0.0f32; 4];
        let eps = 1e-6f32;
        rmsnorm(&x2, &w2, eps, &mut out2);
        let ms = (1.0 + 4.0 + 9.0 + 16.0) / 4.0f32 + eps;
        let inv = 1.0 / libm_sqrtf(ms) as f64;
        for i in 0..4 {
            let want = (x2[i] as f64 * inv) as f32;
            assert!((out2[i] - want).abs() < 1e-4, "rmsnorm idx {i}: got {} want {want}", out2[i]);
        }
    }

    /// RoPE at pos=0 is the identity; at pos=1 the first pair rotates by
    /// exactly 1 radian (theta^0). cos(1)/sin(1) are pinned as f64 constants,
    /// so a wrong pair layout (interleaved vs rotate-half) fails by ~1.0.
    #[test_case]
    fn rope_identity_and_one_radian() {
        let head_dim = 4;
        let mut v = [1.0f32, 2.0, 3.0, 4.0];
        rope(&mut v, 0, head_dim, 10000.0);
        assert_slice_close(&v, &[1.0, 2.0, 3.0, 4.0], 1e-5, 1e-5, "rope pos0");

        // NeoX rotate-half: pairs are (v[i], v[i+d/2]). Pair 0 = (1,3) rotates
        // by 1 rad; pair 1 = (2,4) by theta^(-2/4) = 0.01 rad.
        let (c1, s1) = (0.5403023058681398f32, 0.8414709848078965f32);
        let (c2, s2) = (0.9999500004166653f32, 0.009999833334166664f32);
        let mut v = [1.0f32, 2.0, 3.0, 4.0];
        rope(&mut v, 1, head_dim, 10000.0);
        let want = [
            1.0 * c1 - 3.0 * s1,
            2.0 * c2 - 4.0 * s2,
            1.0 * s1 + 3.0 * c1,
            2.0 * s2 + 4.0 * c2,
        ];
        assert_slice_close(&v, &want, 1e-3, 1e-3, "rope pos1");
    }

    /// softmax: equal inputs → uniform; [0, ln 3] → exactly [0.25, 0.75].
    #[test_case]
    fn softmax_closed_form() {
        let mut x = [1.5f32; 4];
        softmax(&mut x);
        assert_slice_close(&x, &[0.25; 4], 1e-5, 1e-5, "softmax uniform");

        let mut x2 = [0.0f32, 1.0986123];
        softmax(&mut x2);
        assert_slice_close(&x2, &[0.25, 0.75], 1e-4, 1e-4, "softmax 1:3");
    }

    /// silu(x)·up with pinned values: silu(0)=0, silu(1)=1/(1+e⁻¹)≈0.731059,
    /// silu(-1)=-0.268941; large +x → x, large −x → 0.
    #[test_case]
    fn silu_mul_closed_form() {
        let gate = [0.0f32, 1.0, -1.0, 20.0, -20.0];
        let up = [5.0f32, 2.0, 2.0, 1.0, 1.0];
        let mut out = [0.0f32; 5];
        silu_mul(&gate, &up, &mut out);
        let want = [0.0f32, 1.4621172, -0.5378828, 20.0, 0.0];
        assert_slice_close(&out, &want, 1e-3, 1e-3, "silu_mul");
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

    /// Q5_0: the 5th bit comes from the packed `qh` word — element j takes
    /// bit j, element j+16 takes bit j+16 (the `>> (j+12)` trick). Setting
    /// only bit 16 must lift exactly element 16 by +16.
    #[test_case]
    fn dequant_q5_0_fifth_bit_placement() {
        let mut block = [0u8; Q5_0_BLOCK_BYTES];
        block[..2].copy_from_slice(&F16_ONE);
        block[2..6].copy_from_slice(&(1u32 << 16).to_le_bytes()); // qh: only bit 16
        // qs: all nibbles = 5 → base value 5 - 16 = -11 without the high bit.
        for j in 0..16 {
            block[6 + j] = 0x55;
        }
        let mut out = [0.0f32; QK];
        dequant_q5_0_block(&block, &mut out);
        for j in 0..QK {
            let want = if j == 16 { (5 + 16 - 16) as f32 } else { (5 - 16) as f32 };
            assert!((out[j] - want).abs() < 1e-6, "q5_0 idx {j}: got {} want {want}", out[j]);
        }
    }

    /// Q2_K: each `scales` byte packs a 4-bit scale (low) + 4-bit min (high)
    /// for one 16-element group; quants are 2-bit at shift 0/2/4/6 over two
    /// 128-element halves. Group 0 with sc=2/m=1 and all-q=3 gives 2·3−1=5.
    #[test_case]
    fn dequant_q2_k_scales_and_shift_walk() {
        let mut block = [0u8; Q2_K_BLOCK_BYTES];
        block[0] = 0x12; // group 0: sc=2, min=1
        block[1] = 0x01; // group 1 (elems 16..32): sc=1, min=0
        for b in 16..80 {
            block[b] = 0b1110_0111; // 2-bit lanes: shift0=3, shift2=1, shift4=2, shift6=3
        }
        block[80..82].copy_from_slice(&F16_ONE); // d = 1.0
        block[82..84].copy_from_slice(&F16_ONE); // dmin = 1.0
        let mut out = [0.0f32; QK_K];
        dequant_q2_k_block(&block, &mut out);
        // Group 0 (shift 0, q=3): 1*2*3 - 1*1 = 5. Group 1 (shift 0, q=3): 1*1*3 - 0 = 3.
        assert!((out[0] - 5.0).abs() < 1e-6, "q2_k g0: got {}", out[0]);
        assert!((out[16] - 3.0).abs() < 1e-6, "q2_k g1: got {}", out[16]);
        // Elements 32..48 use scales[2]=0 → dl=0, ml=0 → exactly 0 (pins the
        // group→scale-byte mapping; a wrong is-walk would reuse sc=2).
        assert_eq!(out[32], 0.0, "q2_k g2 should use scales[2]=0");
    }

    /// Q3_K: 6-bit scales unpacked via the kmask shuffle; the high bit comes
    /// from `hmask` with *clear = subtract 4*. scales[0]=36 → dl=4; q=3 with
    /// hmask set → 4·3=12; q=0 with hmask clear → 4·(0−4)=−16.
    #[test_case]
    fn dequant_q3_k_kmask_and_hmask_polarity() {
        let mut block = [0u8; Q3_K_BLOCK_BYTES];
        // hmask: set bit 0 (m=1 for the first shift group) for element 0 only.
        block[0] = 0x01;
        // qs[0]: element 0 low-2-bits (shift 0) = 3.
        block[32] = 0b0000_0011;
        // scales: scales[0] = 36 = low4 (4) from byte 0 | high2 (2) from byte 8.
        block[96] = 0x04;
        block[96 + 8] = 0x02;
        block[108..110].copy_from_slice(&F16_ONE); // d = 1.0
        let mut out = [0.0f32; QK_K];
        dequant_q3_k_block(&block, &mut out);
        assert!((out[0] - 12.0).abs() < 1e-6, "q3_k elem0 (hm set): got {}", out[0]);
        // Element 1: q=0, hmask bit clear → (0-4); same dl=4 → -16.
        assert!((out[1] + 16.0).abs() < 1e-6, "q3_k elem1 (hm clear): got {}", out[1]);
    }

    /// Q4_K: 8 sub-blocks of 32 with `get_scale_min_k4`-packed 6-bit
    /// scale/min pairs; per 64 elements the low nibbles come first. Sub-block
    /// 0 with sc=5/m=3, nibble 7 → 1·5·7 − 1·3 = 32.
    #[test_case]
    fn dequant_q4_k_scale_min_pairing() {
        let mut block = [0u8; Q4_K_BLOCK_BYTES];
        block[0..2].copy_from_slice(&F16_ONE); // d = 1.0
        block[2..4].copy_from_slice(&F16_ONE); // dmin = 1.0
        block[4] = 5; // scales[0]: sc(sub0) = 5
        block[4 + 4] = 3; // scales[4]: min(sub0) = 3
        block[5] = 2; // scales[1]: sc(sub1, the high nibbles of the same 32 bytes) = 2
        block[4 + 5] = 1; // scales[5]: min(sub1) = 1
        for j in 0..32 {
            block[16 + j] = 0x97; // low nibble 7 (sub0), high nibble 9 (sub1)
        }
        let mut out = [0.0f32; QK_K];
        dequant_q4_k_block(&block, &mut out);
        assert!((out[0] - 32.0).abs() < 1e-6, "q4_k sub0: got {} want 32", out[0]);
        assert!((out[32] - 17.0).abs() < 1e-6, "q4_k sub1: got {} want 2*9-1", out[32]);
    }

    /// IQ4_NL: nibbles map through the fixed non-linear codebook — nibble 0
    /// is -127, nibble 8 is 1, nibble 15 is 113 (pins the table + order).
    #[test_case]
    fn dequant_iq4_nl_codebook() {
        let mut block = [0u8; IQ4_NL_BLOCK_BYTES];
        block[..2].copy_from_slice(&F16_ONE);
        block[2] = 0x80; // elem 0: nibble 0 -> -127; elem 16: nibble 8 -> 1
        block[3] = 0x0f; // elem 1: nibble 15 -> 113; elem 17: nibble 0 -> -127
        let mut out = [0.0f32; QK];
        dequant_iq4_nl_block(&block, &mut out);
        assert_eq!((out[0], out[16]), (-127.0, 1.0));
        assert_eq!((out[1], out[17]), (113.0, -127.0));
    }

    /// IQ2_XXS: an all-zeros block decodes through grid entry 0 with sign
    /// index 0 and scale nibble 0 — every output must be `d * 0.125 * grid0`
    /// with all-positive signs (ksigns[0] = 0). Pins the grid byte order.
    #[test_case]
    fn dequant_iq2_xxs_grid_and_scale() {
        let mut block = [0u8; IQ2_XXS_BLOCK_BYTES];
        block[..2].copy_from_slice(&F16_ONE);
        let mut out = [0.0f32; QK_K];
        dequant_iq2_xxs_block(&block, &mut out);
        let g0 = super::iqt::IQ2XXS_GRID[0].to_le_bytes();
        for j in 0..8 {
            let want = 0.125 * g0[j] as f32; // d=1, scale nibble 0 -> (0.5+0)*0.25
            assert!((out[j] - want).abs() < 1e-6, "iq2_xxs j{j}: got {} want {want}", out[j]);
        }
    }

    /// BF16 is the top half of an f32; F16 goes through `f16_to_f32`.
    #[test_case]
    fn dequant_f16_and_bf16_chunks() {
        let mut f16 = [0u8; F16_BLOCK_BYTES];
        f16[0..2].copy_from_slice(&F16_HALF); // 0.5
        f16[2..4].copy_from_slice(&F16_ONE); // 1.0
        let mut out = [0.0f32; QK];
        dequant_f16_block(&f16, &mut out);
        assert_eq!((out[0], out[1]), (0.5, 1.0));

        let mut bf16 = [0u8; BF16_BLOCK_BYTES];
        let bits = (3.5f32.to_bits() >> 16) as u16;
        bf16[0..2].copy_from_slice(&bits.to_le_bytes());
        dequant_bf16_block(&bf16, &mut out);
        assert_eq!(out[0], 3.5);
        assert_eq!(out[1], 0.0);
    }

    /// Every supported quant type: `matvec_quant_rows` must equal an f64
    /// accumulation over `dequant_block` outputs on LCG-random data — pins
    /// the block stride / row indexing for each entry in `block_layout`.
    #[test_case]
    fn matvec_quant_rows_consistent_for_all_types() {
        let types = [
            QT_F16, QT_Q4_0, QT_Q4_1, QT_Q5_0, QT_Q5_1, QT_Q8_0, QT_Q2_K, QT_Q3_K, QT_Q4_K, QT_Q5_K,
            QT_Q6_K, QT_Q8_K, QT_IQ2_XXS, QT_IQ2_XS, QT_IQ2_S, QT_IQ3_XXS, QT_IQ3_S, QT_IQ4_NL,
            QT_IQ4_XS, QT_BF16,
        ];
        let mut seed = 0x5EED_1234u32;
        for &qt in &types {
            let (block_bytes, elems) = block_layout(qt);
            let (rows, cols) = (2usize, elems * 2);
            let blocks = cols / elems;
            // Random bytes are a valid block for every one of these formats.
            let w: Vec<u8> = (0..rows * blocks * block_bytes).map(|_| (lcg(&mut seed) >> 8) as u8).collect();
            // Clamp f16-scale fields being NaN/inf is fine for consistency (both
            // sides read the same bytes), but keep x small and finite.
            let x: Vec<f32> = (0..cols).map(|_| (lcg(&mut seed) % 100) as f32 / 50.0 - 1.0).collect();
            let mut y = alloc::vec![0.0f32; rows];
            // SAFETY: w/x/y sized exactly per the contract above.
            unsafe { matvec_quant_rows(qt, w.as_ptr(), x.as_ptr(), y.as_mut_ptr(), 0, rows, cols) };
            let mut buf = [0.0f32; QK_K];
            for r in 0..rows {
                let mut want = 0.0f64;
                for b in 0..blocks {
                    let off = (r * blocks + b) * block_bytes;
                    dequant_block(qt, &w[off..off + block_bytes], &mut buf[..elems]);
                    for i in 0..elems {
                        want += buf[i] as f64 * x[b * elems + i] as f64;
                    }
                }
                let got = y[r] as f64;
                let ok = if want.is_finite() { (got - want).abs() <= 1e-2 + 1e-3 * want.abs() } else { !got.is_finite() };
                assert!(ok, "qt {qt} row {r}: got {got} want {want}");
            }
        }
    }
}
