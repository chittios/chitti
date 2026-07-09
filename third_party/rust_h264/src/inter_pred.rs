//! H.264 inter prediction / motion compensation (spec 8.4.2).
//!
//! Generates predicted blocks for P/B slices by interpolating pixels from
//! reference frames at quarter-pel (luma) or eighth-pel (chroma) precision.

use crate::dpb::DecodedPicture;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use alloc::vec::Vec;
use alloc::vec;

/// Fetch a luma sample from the reference picture with boundary clipping.
/// Out-of-bounds coordinates are clamped to the picture edge (spec 8.4.2.2.1).
#[inline]
fn ref_luma(pic: &DecodedPicture, x: i32, y: i32) -> i32 {
    let cx = x.clamp(0, pic.width as i32 - 1) as usize;
    let cy = y.clamp(0, pic.height as i32 - 1) as usize;
    pic.y[cy * pic.width as usize + cx] as i32
}

/// Fetch a chroma sample with boundary clipping.
#[inline]
fn ref_chroma(plane: &[u8], width: usize, height: usize, x: i32, y: i32) -> i32 {
    if width == 0 || height == 0 {
        return 0;
    }
    let cx = x.clamp(0, width as i32 - 1) as usize;
    let cy = y.clamp(0, height as i32 - 1) as usize;
    let idx = cy * width + cx;
    if idx >= plane.len() {
        return 0;
    }
    plane[idx] as i32
}

#[inline]
fn clip_u8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

#[inline]
fn avg(a: u8, b: u8) -> u8 {
    ((a as u16 + b as u16 + 1) >> 1) as u8
}

/// 6-tap FIR filter coefficient application on 6 consecutive samples.
#[inline(always)]
fn fir6(s: &[u8], i: usize) -> i32 {
    s[i] as i32 - 5 * s[i + 1] as i32 + 20 * s[i + 2] as i32 + 20 * s[i + 3] as i32
        - 5 * s[i + 4] as i32
        + s[i + 5] as i32
}

/// 6-tap FIR on i32 intermediates (for hv second pass).
#[inline(always)]
fn fir6_i32(s0: i32, s1: i32, s2: i32, s3: i32, s4: i32, s5: i32) -> i32 {
    s0 - 5 * s1 + 20 * s2 + 20 * s3 - 5 * s4 + s5
}

// --- strict-align-safe NEON memory access -----------------------------------
//
// The kernel builds aarch64 with `+strict-align`, under which LLVM lowers the
// `vld1*` intrinsics (and auto-vectorized loads) into ~25-instruction byte
// assembly. All NEON in this module therefore loads/stores through inline asm
// (`ldr d/q`, `str d/q`) — correct at runtime on Normal cacheable RAM with
// `SCTLR_EL1.A=0`, and immune to the codegen trap (see CLAUDE.md perf rule 1).

/// Unaligned 8-byte vector load via inline asm.
///
/// # Safety
/// `p` must be readable for 8 bytes.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn ld8(p: *const u8) -> uint8x8_t {
    let v: uint8x8_t;
    // SAFETY: caller guarantees 8 readable bytes; `ldr d` tolerates any
    // alignment on Normal memory with SCTLR.A=0.
    unsafe {
        core::arch::asm!("ldr {v:d}, [{p}]", v = out(vreg) v, p = in(reg) p, options(nostack, readonly, preserves_flags));
    }
    v
}

/// Unaligned 8-byte vector store via inline asm.
///
/// # Safety
/// `p` must be writable for 8 bytes.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn st8(p: *mut u8, v: uint8x8_t) {
    // SAFETY: caller guarantees 8 writable bytes.
    unsafe {
        core::arch::asm!("str {v:d}, [{p}]", v = in(vreg) v, p = in(reg) p, options(nostack, preserves_flags));
    }
}

/// Unaligned 16-byte i16x8 load via inline asm.
///
/// # Safety
/// `p` must be readable for 16 bytes.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn ld16i(p: *const i16) -> int16x8_t {
    let v: int16x8_t;
    // SAFETY: caller guarantees 16 readable bytes.
    unsafe {
        core::arch::asm!("ldr {v:q}, [{p}]", v = out(vreg) v, p = in(reg) p, options(nostack, readonly, preserves_flags));
    }
    v
}

/// Unaligned 16-byte i16x8 store via inline asm.
///
/// # Safety
/// `p` must be writable for 16 bytes.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn st16i(p: *mut i16, v: int16x8_t) {
    // SAFETY: caller guarantees 16 writable bytes.
    unsafe {
        core::arch::asm!("str {v:q}, [{p}]", v = in(vreg) v, p = in(reg) p, options(nostack, preserves_flags));
    }
}

/// 6-tap horizontal half-pel filter at integer position (x, y).
/// Returns clipped u8 result. Used only for boundary blocks.
fn half_pel_h(pic: &DecodedPicture, x: i32, y: i32) -> u8 {
    let s = |dx: i32| ref_luma(pic, x + dx, y);
    clip_u8((s(-2) - 5 * s(-1) + 20 * s(0) + 20 * s(1) - 5 * s(2) + s(3) + 16) >> 5)
}

/// 6-tap vertical half-pel filter at integer position (x, y).
/// Used only for boundary blocks.
fn half_pel_v(pic: &DecodedPicture, x: i32, y: i32) -> u8 {
    let s = |dy: i32| ref_luma(pic, x, y + dy);
    clip_u8((s(-2) - 5 * s(-1) + 20 * s(0) + 20 * s(1) - 5 * s(2) + s(3) + 16) >> 5)
}

/// Diagonal half-pel. Used only for boundary blocks.
fn half_pel_hv(pic: &DecodedPicture, x: i32, y: i32) -> u8 {
    let mut h = [0i32; 6];
    for (i, dy) in (-2..=3).enumerate() {
        let s = |dx: i32| ref_luma(pic, x + dx, y + dy);
        h[i] = s(-2) - 5 * s(-1) + 20 * s(0) + 20 * s(1) - 5 * s(2) + s(3);
    }
    let val = h[0] - 5 * h[1] + 20 * h[2] + 20 * h[3] - 5 * h[4] + h[5];
    clip_u8((val + 512) >> 10)
}

/// Per-pixel interpolation fallback for boundary blocks.
fn luma_interp(pic: &DecodedPicture, x: i32, y: i32, frac_x: i32, frac_y: i32) -> u8 {
    match (frac_x, frac_y) {
        (0, 0) => ref_luma(pic, x, y) as u8,
        (2, 0) => half_pel_h(pic, x, y),
        (0, 2) => half_pel_v(pic, x, y),
        (2, 2) => half_pel_hv(pic, x, y),
        (1, 0) => avg(ref_luma(pic, x, y) as u8, half_pel_h(pic, x, y)),
        (3, 0) => avg(half_pel_h(pic, x, y), ref_luma(pic, x + 1, y) as u8),
        (0, 1) => avg(ref_luma(pic, x, y) as u8, half_pel_v(pic, x, y)),
        (0, 3) => avg(half_pel_v(pic, x, y), ref_luma(pic, x, y + 1) as u8),
        (2, 1) => avg(half_pel_h(pic, x, y), half_pel_hv(pic, x, y)),
        (2, 3) => avg(half_pel_hv(pic, x, y), half_pel_h(pic, x, y + 1)),
        (1, 2) => avg(half_pel_v(pic, x, y), half_pel_hv(pic, x, y)),
        (3, 2) => avg(half_pel_hv(pic, x, y), half_pel_v(pic, x + 1, y)),
        (1, 1) => avg(half_pel_h(pic, x, y), half_pel_v(pic, x, y)),
        (3, 1) => avg(half_pel_h(pic, x, y), half_pel_v(pic, x + 1, y)),
        (1, 3) => avg(half_pel_v(pic, x, y), half_pel_h(pic, x, y + 1)),
        (3, 3) => avg(half_pel_v(pic, x + 1, y), half_pel_h(pic, x, y + 1)),
        _ => unreachable!(),
    }
}

/// NEON 6-tap horizontal half-pel filter for a row of `w` pixels.
/// `src` must have `w + 5` accessible bytes. Processes 8 pixels at a time,
/// with scalar tail for remaining pixels.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn neon_row_half_pel_h(src: &[u8], out: &mut [u8], w: usize) {
    let mut i = 0;
    while i + 8 <= w {
        unsafe {
            let p = src.as_ptr().add(i);
            let s0 = ld8(p);
            let s1 = ld8(p.add(1));
            let s2 = ld8(p.add(2));
            let s3 = ld8(p.add(3));
            let s4 = ld8(p.add(4));
            let s5 = ld8(p.add(5));
            let sum_pos1 = vaddl_u8(s0, s5);
            let sum_20 = vaddl_u8(s2, s3);
            let sum_neg5 = vaddl_u8(s1, s4);
            let mut acc = vreinterpretq_s16_u16(sum_pos1);
            acc = vmlaq_n_s16(acc, vreinterpretq_s16_u16(sum_20), 20);
            acc = vmlsq_n_s16(acc, vreinterpretq_s16_u16(sum_neg5), 5);
            acc = vaddq_s16(acc, vdupq_n_s16(16));
            let clamped = vqmovun_s16(vshrq_n_s16(acc, 5));
            st8(out.as_mut_ptr().add(i), clamped);
        }
        i += 8;
    }
    // Scalar tail
    while i < w {
        out[i] = clip_u8((fir6(src, i) + 16) >> 5);
        i += 1;
    }
}

/// Row-based horizontal half-pel filter for in-bounds blocks.
/// Reads `w` output pixels from row at `src` (which must have `w + 5` accessible bytes).
#[inline(always)]
fn row_half_pel_h(src: &[u8], out: &mut [u8], w: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        neon_row_half_pel_h(src, out, w);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..w {
            out[i] = clip_u8((fir6(src, i) + 16) >> 5);
        }
    }
}

/// NEON 6-tap vertical half-pel filter for a row of `w` pixels.
/// `rows` contains 6 row slices (y-2..y+3), each at least `w` bytes.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn neon_row_half_pel_v(rows: [&[u8]; 6], out: &mut [u8], w: usize) {
    let mut i = 0;
    while i + 8 <= w {
        unsafe {
            let s0 = ld8(rows[0].as_ptr().add(i));
            let s1 = ld8(rows[1].as_ptr().add(i));
            let s2 = ld8(rows[2].as_ptr().add(i));
            let s3 = ld8(rows[3].as_ptr().add(i));
            let s4 = ld8(rows[4].as_ptr().add(i));
            let s5 = ld8(rows[5].as_ptr().add(i));
            let sum_pos1 = vaddl_u8(s0, s5);
            let sum_20 = vaddl_u8(s2, s3);
            let sum_neg5 = vaddl_u8(s1, s4);
            let mut acc = vreinterpretq_s16_u16(sum_pos1);
            acc = vmlaq_n_s16(acc, vreinterpretq_s16_u16(sum_20), 20);
            acc = vmlsq_n_s16(acc, vreinterpretq_s16_u16(sum_neg5), 5);
            acc = vaddq_s16(acc, vdupq_n_s16(16));
            let clamped = vqmovun_s16(vshrq_n_s16(acc, 5));
            st8(out.as_mut_ptr().add(i), clamped);
        }
        i += 8;
    }
    // Scalar tail
    while i < w {
        let val = rows[0][i] as i32 - 5 * rows[1][i] as i32
            + 20 * rows[2][i] as i32
            + 20 * rows[3][i] as i32
            - 5 * rows[4][i] as i32
            + rows[5][i] as i32;
        out[i] = clip_u8((val + 16) >> 5);
        i += 1;
    }
}

/// NEON: compute vertical FIR for 8 pixels, return as clipped u8x8.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn neon_vfir6_8(rows: &[&[u8]; 6], i: usize) -> uint8x8_t {
    let s0 = ld8(rows[0].as_ptr().add(i));
    let s1 = ld8(rows[1].as_ptr().add(i));
    let s2 = ld8(rows[2].as_ptr().add(i));
    let s3 = ld8(rows[3].as_ptr().add(i));
    let s4 = ld8(rows[4].as_ptr().add(i));
    let s5 = ld8(rows[5].as_ptr().add(i));
    let sum_pos1 = vaddl_u8(s0, s5);
    let sum_20 = vaddl_u8(s2, s3);
    let sum_neg5 = vaddl_u8(s1, s4);
    let mut acc = vreinterpretq_s16_u16(sum_pos1);
    acc = vmlaq_n_s16(acc, vreinterpretq_s16_u16(sum_20), 20);
    acc = vmlsq_n_s16(acc, vreinterpretq_s16_u16(sum_neg5), 5);
    acc = vaddq_s16(acc, vdupq_n_s16(16));
    vqmovun_s16(vshrq_n_s16(acc, 5))
}

/// NEON: avg(a, half_v) for 8 pixels — vertical FIR then average with integer row.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn neon_row_avg_int_v(int_row: &[u8], rows: [&[u8]; 6], out: &mut [u8], w: usize) {
    let mut i = 0;
    while i + 8 <= w {
        unsafe {
            let int_val = ld8(int_row.as_ptr().add(i));
            let hp = neon_vfir6_8(&rows, i);
            let result = vrhadd_u8(int_val, hp);
            st8(out.as_mut_ptr().add(i), result);
        }
        i += 8;
    }
    while i < w {
        let val = rows[0][i] as i32 - 5 * rows[1][i] as i32
            + 20 * rows[2][i] as i32
            + 20 * rows[3][i] as i32
            - 5 * rows[4][i] as i32
            + rows[5][i] as i32;
        let hp = clip_u8((val + 16) >> 5);
        out[i] = avg(int_row[i], hp);
        i += 1;
    }
}

/// NEON: avg(half_h, half_v) for 8 pixels — horizontal and vertical FIR then average.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn neon_row_avg_h_v(src_h: &[u8], rows_v: [&[u8]; 6], out: &mut [u8], w: usize) {
    let mut i = 0;
    while i + 8 <= w {
        unsafe {
            // Horizontal FIR
            let p = src_h.as_ptr().add(i);
            let h0 = ld8(p);
            let h1 = ld8(p.add(1));
            let h2 = ld8(p.add(2));
            let h3 = ld8(p.add(3));
            let h4 = ld8(p.add(4));
            let h5 = ld8(p.add(5));
            let hsum_pos1 = vaddl_u8(h0, h5);
            let hsum_20 = vaddl_u8(h2, h3);
            let hsum_neg5 = vaddl_u8(h1, h4);
            let mut hacc = vreinterpretq_s16_u16(hsum_pos1);
            hacc = vmlaq_n_s16(hacc, vreinterpretq_s16_u16(hsum_20), 20);
            hacc = vmlsq_n_s16(hacc, vreinterpretq_s16_u16(hsum_neg5), 5);
            hacc = vaddq_s16(hacc, vdupq_n_s16(16));
            let h_val = vqmovun_s16(vshrq_n_s16(hacc, 5));

            // Vertical FIR
            let v_val = neon_vfir6_8(&rows_v, i);

            let result = vrhadd_u8(h_val, v_val);
            st8(out.as_mut_ptr().add(i), result);
        }
        i += 8;
    }
    while i < w {
        let h_val = clip_u8((fir6(src_h, i) + 16) >> 5);
        let v_val = clip_u8(
            (rows_v[0][i] as i32 - 5 * rows_v[1][i] as i32
                + 20 * rows_v[2][i] as i32
                + 20 * rows_v[3][i] as i32
                - 5 * rows_v[4][i] as i32
                + rows_v[5][i] as i32
                + 16)
                >> 5,
        );
        out[i] = avg(h_val, v_val);
        i += 1;
    }
}

/// Row-based vertical half-pel filter for in-bounds blocks.
/// `rows` contains 6 row slices (y-2..y+3), each at least `w` bytes.
#[inline(always)]
fn row_half_pel_v(rows: [&[u8]; 6], out: &mut [u8], w: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        neon_row_half_pel_v(rows, out, w);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..w {
            let val = rows[0][i] as i32 - 5 * rows[1][i] as i32
                + 20 * rows[2][i] as i32
                + 20 * rows[3][i] as i32
                - 5 * rows[4][i] as i32
                + rows[5][i] as i32;
            out[i] = clip_u8((val + 16) >> 5);
        }
    }
}

/// Row-based diagonal half-pel (hv) for in-bounds blocks.
/// `rows` contains 6 row slices (y-2..y+3), each with `w + 5` accessible bytes.
#[allow(clippy::needless_range_loop)]
#[inline(always)]
fn row_half_pel_hv(rows: [&[u8]; 6], out: &mut [u8], w: usize) {
    // First pass: horizontal filter on each of 6 rows → i32 intermediates
    // We need w intermediate values per row
    for i in 0..w {
        let h0 = fir6(rows[0], i);
        let h1 = fir6(rows[1], i);
        let h2 = fir6(rows[2], i);
        let h3 = fir6(rows[3], i);
        let h4 = fir6(rows[4], i);
        let h5 = fir6(rows[5], i);
        let val = fir6_i32(h0, h1, h2, h3, h4, h5);
        out[i] = clip_u8((val + 512) >> 10);
    }
}

// --- per-block hv fast path ---------------------------------------------------
//
// Every quarter-pel position that involves the diagonal (hv) filter used to
// recompute six horizontal FIRs *per output pixel*. Instead: run the
// horizontal 6-tap once per source row into an i16 block buffer, then do the
// vertical pass (and any avg) row-wise with NEON. Exact same integer math —
// bit-identical output — at ~1/6 the FIR work even before SIMD.

/// h-FIR intermediate buffer: (16 + 5) rows × 16 cols of i16 (max luma block).
const HBUF_LEN: usize = 21 * 16;

/// Unrounded horizontal 6-tap FIR of `src` into `out[0..w]` (i16 fits: the
/// FIR of u8 inputs spans −2550..=10710). `src` needs `w + 5` bytes.
#[inline(always)]
fn hfir_row_i16(src: &[u8], out: &mut [i16], w: usize) {
    let mut i = 0;
    #[cfg(target_arch = "aarch64")]
    while i + 8 <= w {
        // SAFETY: src has w+5 readable bytes; out has w writable i16.
        unsafe {
            let p = src.as_ptr().add(i);
            let s0 = ld8(p);
            let s1 = ld8(p.add(1));
            let s2 = ld8(p.add(2));
            let s3 = ld8(p.add(3));
            let s4 = ld8(p.add(4));
            let s5 = ld8(p.add(5));
            let mut acc = vreinterpretq_s16_u16(vaddl_u8(s0, s5));
            acc = vmlaq_n_s16(acc, vreinterpretq_s16_u16(vaddl_u8(s2, s3)), 20);
            acc = vmlsq_n_s16(acc, vreinterpretq_s16_u16(vaddl_u8(s1, s4)), 5);
            st16i(out.as_mut_ptr().add(i), acc);
        }
        i += 8;
    }
    while i < w {
        out[i] = fir6(src, i) as i16;
        i += 1;
    }
}

/// hv second pass: `out[i] = clip((vFIR6(hrows) + 512) >> 10)` over six
/// consecutive i16 h-FIR rows starting at `hrows` (row stride `w`).
#[inline(always)]
fn vfir_hv_row(hrows: &[i16], w: usize, out: &mut [u8]) {
    let mut i = 0;
    #[cfg(target_arch = "aarch64")]
    while i + 8 <= w {
        // SAFETY: hrows has 6*w readable i16 (rows r..r+5); out has w bytes.
        unsafe {
            let r = |k: usize| ld16i(hrows.as_ptr().add(k * w + i));
            let (r0, r1, r2, r3, r4, r5) = (r(0), r(1), r(2), r(3), r(4), r(5));
            // Widen to i32 — the vertical FIR of i16 intermediates overflows i16.
            let lo = |v: int16x8_t| vmovl_s16(vget_low_s16(v));
            let hi = |v: int16x8_t| vmovl_s16(vget_high_s16(v));
            let comb = |a: int32x4_t, b: int32x4_t, c: int32x4_t| {
                // a = r0+r5, b = r2+r3, c = r1+r4 → a + 20b − 5c + 512
                let mut s = vmlaq_n_s32(a, b, 20);
                s = vmlsq_n_s32(s, c, 5);
                vshrq_n_s32(vaddq_s32(s, vdupq_n_s32(512)), 10)
            };
            let lo32 = comb(vaddq_s32(lo(r0), lo(r5)), vaddq_s32(lo(r2), lo(r3)), vaddq_s32(lo(r1), lo(r4)));
            let hi32 = comb(vaddq_s32(hi(r0), hi(r5)), vaddq_s32(hi(r2), hi(r3)), vaddq_s32(hi(r1), hi(r4)));
            let n16 = vcombine_s16(vqmovn_s32(lo32), vqmovn_s32(hi32));
            st8(out.as_mut_ptr().add(i), vqmovun_s16(n16));
        }
        i += 8;
    }
    while i < w {
        let v = fir6_i32(
            hrows[i] as i32,
            hrows[w + i] as i32,
            hrows[2 * w + i] as i32,
            hrows[3 * w + i] as i32,
            hrows[4 * w + i] as i32,
            hrows[5 * w + i] as i32,
        );
        out[i] = clip_u8((v + 512) >> 10);
        i += 1;
    }
}

/// Half-pel h from a precomputed i16 h-FIR row: `clip((h + 16) >> 5)`.
#[inline(always)]
fn hrow_to_u8(hrow: &[i16], out: &mut [u8], w: usize) {
    let mut i = 0;
    #[cfg(target_arch = "aarch64")]
    while i + 8 <= w {
        // SAFETY: hrow has w readable i16; out has w writable bytes.
        unsafe {
            let v = vaddq_s16(ld16i(hrow.as_ptr().add(i)), vdupq_n_s16(16));
            st8(out.as_mut_ptr().add(i), vqmovun_s16(vshrq_n_s16(v, 5)));
        }
        i += 8;
    }
    while i < w {
        out[i] = clip_u8((hrow[i] as i32 + 16) >> 5);
        i += 1;
    }
}

/// `out[i] = (a[i] + b[i] + 1) >> 1` (the spec quarter-pel average).
#[inline(always)]
fn avg_rows(a: &[u8], b: &[u8], out: &mut [u8], w: usize) {
    let mut i = 0;
    #[cfg(target_arch = "aarch64")]
    while i + 8 <= w {
        // SAFETY: all three have w accessible bytes.
        unsafe {
            st8(out.as_mut_ptr().add(i), vrhadd_u8(ld8(a.as_ptr().add(i)), ld8(b.as_ptr().add(i))));
        }
        i += 8;
    }
    while i < w {
        out[i] = avg(a[i], b[i]);
        i += 1;
    }
}

/// Fill the (h+5)-row i16 h-FIR block buffer for an hv-involved case: row `rr`
/// holds the horizontal FIR of source row `y + rr − 2` at columns `x−2..`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn fill_hbuf(
    ref_y: &[u8],
    stride: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    hbuf: &mut [i16; HBUF_LEN],
) {
    for rr in 0..h + 5 {
        let off = (y + rr - 2) * stride + (x - 2);
        hfir_row_i16(&ref_y[off..off + w + 5], &mut hbuf[rr * w..(rr + 1) * w], w);
    }
}

/// Check if a block with the given filter margins is fully within bounds.
/// For half-pel filters, margin is 3 (needs x-2..x+w+2, y-2..y+h+2).
/// For full-pel, margin_left/top=0, margin_right/bottom=0.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn block_in_bounds(
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    pic_w: i32,
    pic_h: i32,
    margin_left: i32,
    margin_top: i32,
    margin_right: i32,
    margin_bottom: i32,
) -> bool {
    x - margin_left >= 0
        && y - margin_top >= 0
        && x + w + margin_right <= pic_w
        && y + h + margin_bottom <= pic_h
}

/// Luma motion compensation with explicit reference stride and buffer offset.
/// For field-coded MBAFF MBs, pass `ref_stride = ref_pic.width * 2`,
/// `ref_y_offset = ref_pic.width` (for bottom field) or `0` (for top field),
/// and `y` in field-line units.
#[allow(clippy::too_many_arguments)]
pub fn luma_mc_stride(
    ref_pic: &DecodedPicture,
    x: i32,
    y: i32,
    dx: i32,
    dy: i32,
    block_w: usize,
    block_h: usize,
    output: &mut [u8],
    ref_stride: usize,
    ref_y_offset: usize,
) {
    // Guard against malformed block sizes that would overrun the output buffer
    if block_w == 0 || block_h == 0 || block_w * block_h > output.len() {
        return;
    }
    let frac_x = dx.rem_euclid(4);
    let frac_y = dy.rem_euclid(4);
    // Integer part: arithmetic right shift gives floor division for negative values
    let x_int = x + (dx >> 2);
    let y_int = y + (dy >> 2);

    let pic_w = ref_pic.width as i32;
    let pic_h = if ref_stride == ref_pic.width as usize {
        ref_pic.height as i32
    } else {
        // Field-coded: effective height is half the frame
        (ref_pic.height / 2) as i32
    };
    let stride = ref_stride;
    let bw = block_w as i32;
    let bh = block_h as i32;
    let ref_y = &ref_pic.y[ref_y_offset..];

    // Determine margins needed for the filter type
    // Half-pel filters need 2 pixels before and 3 after the block
    let needs_h = frac_x != 0; // horizontal filter needed
    let needs_v = frac_y != 0; // vertical filter needed
    let margin_l = if needs_h { 2 } else { 0 };
    let margin_r = if needs_h { 3 } else { 0 };
    let margin_t = if needs_v { 2 } else { 0 };
    let margin_b = if needs_v { 3 } else { 0 };

    // Quarter-pel positions that average with an offset integer/half-pel sample
    // may need +1 in a direction
    let extra_r: i32 = match frac_x {
        3 => 1,
        _ => 0,
    };
    let extra_b: i32 = match frac_y {
        3 => 1,
        _ => 0,
    };

    if block_in_bounds(
        x_int,
        y_int,
        bw + extra_r,
        bh + extra_b,
        pic_w,
        pic_h,
        margin_l,
        margin_t,
        margin_r,
        margin_b,
    ) {
        // Fast path: entire block + filter margins are in bounds
        // Access reference buffer directly without per-pixel clamping
        luma_mc_inner(
            ref_y,
            stride,
            x_int as usize,
            y_int as usize,
            block_w,
            block_h,
            frac_x,
            frac_y,
            output,
        );
    } else if ref_y_offset > 0 || stride != ref_pic.width as usize {
        // Field-coded boundary fallback: extract field lines into a temporary
        // DecodedPicture and use the standard interpolation on it.
        let field_h = pic_h as usize;
        let field_w = pic_w as usize;
        let mut field_buf = vec![0u8; field_w * field_h];
        for r in 0..field_h {
            let src_off = ref_y_offset + r * stride;
            let dst_off = r * field_w;
            for c in 0..field_w {
                if src_off + c < ref_pic.y.len() {
                    field_buf[dst_off + c] = ref_pic.y[src_off + c];
                }
            }
        }
        let field_pic = DecodedPicture {
            y: field_buf,
            u: vec![],
            v: vec![],
            width: field_w as u32,
            height: field_h as u32,
            pic_order_cnt: 0,
            frame_num: 0,
            mv_l0: vec![],
            ref_idx_l0: vec![],
            ref_poc_l0: vec![],
            mv_l1: vec![],
            ref_idx_l1: vec![],
            mb_width: 0,
            is_intra: false,
        };
        for row in 0..block_h {
            for col in 0..block_w {
                output[row * block_w + col] = luma_interp(
                    &field_pic,
                    x_int + col as i32,
                    y_int + row as i32,
                    frac_x,
                    frac_y,
                );
            }
        }
    } else {
        // Boundary fallback: per-pixel with clamping
        for row in 0..block_h {
            for col in 0..block_w {
                output[row * block_w + col] = luma_interp(
                    ref_pic,
                    x_int + col as i32,
                    y_int + row as i32,
                    frac_x,
                    frac_y,
                );
            }
        }
    }
}

/// Inner loop for in-bounds luma MC. All reference accesses are unchecked
/// (bounds already verified by caller). Dispatches on fractional position
/// once, then processes all rows with direct buffer access.
#[allow(clippy::too_many_arguments)]
fn luma_mc_inner(
    ref_y: &[u8],
    stride: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    frac_x: i32,
    frac_y: i32,
    output: &mut [u8],
) {
    // Helper: get a row slice starting at (x + dx, y + dy) with length len
    let row = |dy: isize, dx: isize, len: usize| -> &[u8] {
        let off = (y as isize + dy) as usize * stride + (x as isize + dx) as usize;
        &ref_y[off..off + len]
    };

    // Helper: get 6 vertically adjacent rows for vertical/diagonal filters
    let vrows = |dx: isize, dy_base: isize, len: usize| -> [&[u8]; 6] {
        [
            row(dy_base - 2, dx, len),
            row(dy_base - 1, dx, len),
            row(dy_base, dx, len),
            row(dy_base + 1, dx, len),
            row(dy_base + 2, dx, len),
            row(dy_base + 3, dx, len),
        ]
    };

    match (frac_x, frac_y) {
        (0, 0) => {
            // Full-pel copy
            for r in 0..h {
                let src = row(r as isize, 0, w);
                output[r * w..(r + 1) * w].copy_from_slice(src);
            }
        }
        (2, 0) => {
            // Half-pel horizontal
            for r in 0..h {
                let src = row(r as isize, -2, w + 5);
                row_half_pel_h(src, &mut output[r * w..], w);
            }
        }
        (0, 2) => {
            // Half-pel vertical
            for r in 0..h {
                let rows = vrows(0, r as isize, w);
                row_half_pel_v(rows, &mut output[r * w..], w);
            }
        }
        (2, 2) => {
            // Half-pel diagonal: h-FIR once per source row, then the vertical
            // pass row-wise (was: six h-FIRs per output pixel).
            let mut hbuf = [0i16; HBUF_LEN];
            fill_hbuf(ref_y, stride, x, y, w, h, &mut hbuf);
            for r in 0..h {
                vfir_hv_row(&hbuf[r * w..(r + 6) * w], w, &mut output[r * w..][..w]);
            }
        }
        (1, 0) | (3, 0) => {
            // Quarter-pel: avg(half_h, integer at dx = 0 or +1). avg is
            // symmetric, so both orders share one path.
            let dx = if frac_x == 1 { 0 } else { 1 };
            let mut hp = [0u8; 16];
            for r in 0..h {
                let int_row = row(r as isize, dx, w);
                let src_h = row(r as isize, -2, w + 5);
                row_half_pel_h(src_h, &mut hp, w);
                avg_rows(int_row, &hp, &mut output[r * w..][..w], w);
            }
        }
        (0, 1) => {
            // Quarter-pel: avg(integer, half_v)
            for r in 0..h {
                let int_row = row(r as isize, 0, w);
                let rows = vrows(0, r as isize, w);
                #[cfg(target_arch = "aarch64")]
                {
                    neon_row_avg_int_v(int_row, rows, &mut output[r * w..], w);
                }
                #[cfg(not(target_arch = "aarch64"))]
                for i in 0..w {
                    let val = rows[0][i] as i32 - 5 * rows[1][i] as i32
                        + 20 * rows[2][i] as i32
                        + 20 * rows[3][i] as i32
                        - 5 * rows[4][i] as i32
                        + rows[5][i] as i32;
                    let hp = clip_u8((val + 16) >> 5);
                    output[r * w + i] = avg(int_row[i], hp);
                }
            }
        }
        (0, 3) => {
            // Quarter-pel: avg(half_v, integer_below)
            for r in 0..h {
                let int_row = row(r as isize + 1, 0, w);
                let rows = vrows(0, r as isize, w);
                #[cfg(target_arch = "aarch64")]
                {
                    neon_row_avg_int_v(int_row, rows, &mut output[r * w..], w);
                }
                #[cfg(not(target_arch = "aarch64"))]
                for i in 0..w {
                    let val = rows[0][i] as i32 - 5 * rows[1][i] as i32
                        + 20 * rows[2][i] as i32
                        + 20 * rows[3][i] as i32
                        - 5 * rows[4][i] as i32
                        + rows[5][i] as i32;
                    let hp = clip_u8((val + 16) >> 5);
                    output[r * w + i] = avg(hp, int_row[i]);
                }
            }
        }
        (2, 1) | (2, 3) => {
            // avg(half_h at dy = r or r+1, half_hv). The h row is the rounded
            // form of the same h-FIR intermediates the hv pass needs — one
            // shared block buffer serves both (hbuf row r+2 is dy = r).
            let hrow_off = if frac_y == 1 { 2 } else { 3 };
            let mut hbuf = [0i16; HBUF_LEN];
            fill_hbuf(ref_y, stride, x, y, w, h, &mut hbuf);
            let mut hp = [0u8; 16];
            let mut hv = [0u8; 16];
            for r in 0..h {
                hrow_to_u8(&hbuf[(r + hrow_off) * w..], &mut hp, w);
                vfir_hv_row(&hbuf[r * w..(r + 6) * w], w, &mut hv);
                avg_rows(&hp, &hv, &mut output[r * w..][..w], w);
            }
        }
        (1, 2) | (3, 2) => {
            // avg(half_v at dx = 0 or +1, half_hv) — v from raw pixels, hv
            // from the shared h-FIR block buffer; avg is symmetric.
            let dx = if frac_x == 1 { 0 } else { 1 };
            let mut hbuf = [0i16; HBUF_LEN];
            fill_hbuf(ref_y, stride, x, y, w, h, &mut hbuf);
            let mut vp = [0u8; 16];
            let mut hv = [0u8; 16];
            for r in 0..h {
                let rows_v = vrows(dx, r as isize, w);
                row_half_pel_v(rows_v, &mut vp, w);
                vfir_hv_row(&hbuf[r * w..(r + 6) * w], w, &mut hv);
                avg_rows(&vp, &hv, &mut output[r * w..][..w], w);
            }
        }
        (1, 1) => {
            // avg(half_h, half_v)
            for r in 0..h {
                let src_h = row(r as isize, -2, w + 5);
                let rows_v = vrows(0, r as isize, w);
                #[cfg(target_arch = "aarch64")]
                {
                    neon_row_avg_h_v(src_h, rows_v, &mut output[r * w..], w);
                }
                #[cfg(not(target_arch = "aarch64"))]
                for i in 0..w {
                    let h_val = clip_u8((fir6(src_h, i) + 16) >> 5);
                    let v_val = clip_u8(
                        (rows_v[0][i] as i32 - 5 * rows_v[1][i] as i32
                            + 20 * rows_v[2][i] as i32
                            + 20 * rows_v[3][i] as i32
                            - 5 * rows_v[4][i] as i32
                            + rows_v[5][i] as i32
                            + 16)
                            >> 5,
                    );
                    output[r * w + i] = avg(h_val, v_val);
                }
            }
        }
        (3, 1) => {
            // avg(half_h, half_v_right)
            for r in 0..h {
                let src_h = row(r as isize, -2, w + 5);
                let rows_v = vrows(1, r as isize, w);
                #[cfg(target_arch = "aarch64")]
                {
                    neon_row_avg_h_v(src_h, rows_v, &mut output[r * w..], w);
                }
                #[cfg(not(target_arch = "aarch64"))]
                for i in 0..w {
                    let h_val = clip_u8((fir6(src_h, i) + 16) >> 5);
                    let v_val = clip_u8(
                        (rows_v[0][i] as i32 - 5 * rows_v[1][i] as i32
                            + 20 * rows_v[2][i] as i32
                            + 20 * rows_v[3][i] as i32
                            - 5 * rows_v[4][i] as i32
                            + rows_v[5][i] as i32
                            + 16)
                            >> 5,
                    );
                    output[r * w + i] = avg(h_val, v_val);
                }
            }
        }
        (1, 3) => {
            // avg(half_v, half_h_below)
            for r in 0..h {
                let src_h = row(r as isize + 1, -2, w + 5);
                let rows_v = vrows(0, r as isize, w);
                #[cfg(target_arch = "aarch64")]
                {
                    neon_row_avg_h_v(src_h, rows_v, &mut output[r * w..], w);
                }
                #[cfg(not(target_arch = "aarch64"))]
                for i in 0..w {
                    let v_val = clip_u8(
                        (rows_v[0][i] as i32 - 5 * rows_v[1][i] as i32
                            + 20 * rows_v[2][i] as i32
                            + 20 * rows_v[3][i] as i32
                            - 5 * rows_v[4][i] as i32
                            + rows_v[5][i] as i32
                            + 16)
                            >> 5,
                    );
                    let h_val = clip_u8((fir6(src_h, i) + 16) >> 5);
                    output[r * w + i] = avg(v_val, h_val);
                }
            }
        }
        (3, 3) => {
            // avg(half_v_right, half_h_below)
            for r in 0..h {
                let src_h = row(r as isize + 1, -2, w + 5);
                let rows_v = vrows(1, r as isize, w);
                #[cfg(target_arch = "aarch64")]
                {
                    neon_row_avg_h_v(src_h, rows_v, &mut output[r * w..], w);
                }
                #[cfg(not(target_arch = "aarch64"))]
                for i in 0..w {
                    let v_val = clip_u8(
                        (rows_v[0][i] as i32 - 5 * rows_v[1][i] as i32
                            + 20 * rows_v[2][i] as i32
                            + 20 * rows_v[3][i] as i32
                            - 5 * rows_v[4][i] as i32
                            + rows_v[5][i] as i32
                            + 16)
                            >> 5,
                    );
                    let h_val = clip_u8((fir6(src_h, i) + 16) >> 5);
                    output[r * w + i] = avg(v_val, h_val);
                }
            }
        }
        _ => unreachable!(),
    }
}

/// NEON chroma bilinear interpolation: process the entire block at once.
/// `ref_plane` indexed at `top_off` for top-left sample. Each row has stride
/// `ref_width` and at least `block_w + 1` accessible bytes from the top-left.
/// The block must occupy `block_h + 1` rows (top + bottom for each output row).
#[cfg(target_arch = "aarch64")]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn neon_chroma_bilinear_block(
    ref_plane: &[u8],
    top_off: usize,
    ref_width: usize,
    block_w: usize,
    block_h: usize,
    output: &mut [u8],
    c00: u8,
    c01: u8,
    c10: u8,
    c11: u8,
) {
    unsafe {
        // Hoist coefficient duplication outside the row loop
        let v00 = vdup_n_u8(c00);
        let v01 = vdup_n_u8(c01);
        let v10 = vdup_n_u8(c10);
        let v11 = vdup_n_u8(c11);

        for r in 0..block_h {
            let top_p = ref_plane.as_ptr().add(top_off + r * ref_width);
            let bot_p = top_p.add(ref_width);
            let out_p = output.as_mut_ptr().add(r * block_w);

            let mut i = 0;
            while i + 8 <= block_w {
                let a = ld8(top_p.add(i));
                let b = ld8(top_p.add(i + 1));
                let c = ld8(bot_p.add(i));
                let d = ld8(bot_p.add(i + 1));

                let mut acc = vmull_u8(a, v00);
                acc = vmlal_u8(acc, b, v01);
                acc = vmlal_u8(acc, c, v10);
                acc = vmlal_u8(acc, d, v11);
                let res = vrshrn_n_u16(acc, 6);
                st8(out_p.add(i), res);
                i += 8;
            }
            // Scalar tail
            while i < block_w {
                let a = *top_p.add(i) as u32;
                let b = *top_p.add(i + 1) as u32;
                let c = *bot_p.add(i) as u32;
                let d = *bot_p.add(i + 1) as u32;
                let val = c00 as u32 * a + c01 as u32 * b + c10 as u32 * c + c11 as u32 * d;
                *out_p.add(i) = ((val + 32) >> 6) as u8;
                i += 1;
            }
        }
    }
}

/// Perform chroma motion compensation for one plane (U or V).
///
/// Chroma MVs use the same quarter-pel values as luma, but since chroma is
/// half spatial resolution (4:2:0), these become eighth-pel for chroma.
/// Uses bilinear interpolation (spec 8.4.2.2.2).
///
/// `x`, `y`: chroma block top-left in full chroma-pel coordinates.
/// `dx`, `dy`: motion vector in eighth-pel units (= luma quarter-pel MV).
#[allow(clippy::too_many_arguments)]
pub fn chroma_mc(
    ref_plane: &[u8],
    ref_width: usize,
    ref_height: usize,
    x: i32,
    y: i32,
    dx: i32,
    dy: i32,
    block_w: usize,
    block_h: usize,
    output: &mut [u8],
) {
    // Guard against malformed block sizes that would overrun the output buffer
    if block_w == 0 || block_h == 0 || block_w * block_h > output.len() {
        return;
    }
    // For field-coded MBAFF, the effective height is derived from the available
    // plane data, which may be smaller than the frame chroma height.
    let ref_height = if ref_width > 0 {
        ref_height.min(ref_plane.len() / ref_width)
    } else {
        ref_height
    };
    let frac_x = dx.rem_euclid(8);
    let frac_y = dy.rem_euclid(8);
    let x_int = x + (dx >> 3);
    let y_int = y + (dy >> 3);

    // Full-pel fast path: direct copy when no interpolation needed
    if frac_x == 0 && frac_y == 0 {
        let w = ref_width as i32;
        let h = ref_height as i32;
        if x_int >= 0
            && y_int >= 0
            && x_int + block_w as i32 <= w
            && y_int + block_h as i32 <= h
            && (y_int as usize + block_h) * ref_width <= ref_plane.len()
        {
            let mut src_off = y_int as usize * ref_width + x_int as usize;
            for row in 0..block_h {
                if src_off + block_w > ref_plane.len() {
                    return;
                }
                output[row * block_w..(row + 1) * block_w]
                    .copy_from_slice(&ref_plane[src_off..src_off + block_w]);
                src_off += ref_width;
            }
        } else {
            for row in 0..block_h {
                for col in 0..block_w {
                    output[row * block_w + col] = ref_chroma(
                        ref_plane,
                        ref_width,
                        ref_height,
                        x_int + col as i32,
                        y_int + row as i32,
                    ) as u8;
                }
            }
        }
        return;
    }

    // In-bounds fast path: needs x_int..x_int+w+1 and y_int..y_int+h+1
    let w_i32 = ref_width as i32;
    let h_i32 = ref_height as i32;
    let in_bounds = x_int >= 0
        && y_int >= 0
        && (x_int + block_w as i32) < w_i32
        && (y_int + block_h as i32) < h_i32;

    if in_bounds {
        let c00 = ((8 - frac_x) * (8 - frac_y)) as u8;
        let c01 = (frac_x * (8 - frac_y)) as u8;
        let c10 = ((8 - frac_x) * frac_y) as u8;
        let c11 = (frac_x * frac_y) as u8;

        let x_u = x_int as usize;
        let y_u = y_int as usize;
        let top_off = y_u * ref_width + x_u;

        // Guard against ref_plane size mismatch (e.g. SPS change in malformed stream)
        let last_idx = top_off + block_h * ref_width + block_w;
        if last_idx >= ref_plane.len() {
            return; // ref plane too small, skip MC
        }

        #[cfg(target_arch = "aarch64")]
        {
            neon_chroma_bilinear_block(
                ref_plane, top_off, ref_width, block_w, block_h, output, c00, c01, c10, c11,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        for row in 0..block_h {
            let row_top = top_off + row * ref_width;
            let row_bot = row_top + ref_width;
            for i in 0..block_w {
                let val = c00 as i32 * ref_plane[row_top + i] as i32
                    + c01 as i32 * ref_plane[row_top + i + 1] as i32
                    + c10 as i32 * ref_plane[row_bot + i] as i32
                    + c11 as i32 * ref_plane[row_bot + i + 1] as i32;
                output[row * block_w + i] = ((val + 32) >> 6) as u8;
            }
        }
        return;
    }

    // Boundary fallback: per-pixel with clamping
    for row in 0..block_h {
        for col in 0..block_w {
            let xf = x_int + col as i32;
            let yf = y_int + row as i32;
            let a = ref_chroma(ref_plane, ref_width, ref_height, xf, yf);
            let b = ref_chroma(ref_plane, ref_width, ref_height, xf + 1, yf);
            let c = ref_chroma(ref_plane, ref_width, ref_height, xf, yf + 1);
            let d = ref_chroma(ref_plane, ref_width, ref_height, xf + 1, yf + 1);

            let val = (8 - frac_x) * (8 - frac_y) * a
                + frac_x * (8 - frac_y) * b
                + (8 - frac_x) * frac_y * c
                + frac_x * frac_y * d;
            output[row * block_w + col] = ((val + 32) >> 6) as u8;
        }
    }
}

/// Bi-prediction averaging (spec 8.4.2.3.2).
/// `output[i] = (pred_l0[i] + pred_l1[i] + 1) >> 1` for each pixel.
pub fn bi_pred_avg(pred_l0: &[u8], pred_l1: &[u8], output: &mut [u8]) {
    for (o, (&a, &b)) in output.iter_mut().zip(pred_l0.iter().zip(pred_l1.iter())) {
        *o = ((a as u16 + b as u16 + 1) >> 1) as u8;
    }
}

/// Apply explicit weighted prediction to uni-directional MC output (spec 8.4.2.3.1).
/// `output[i] = clip((pred[i] * weight + (1 << (log2_denom - 1))) >> log2_denom + offset)`
/// When log2_denom == 0, the rounding term is 0.
pub fn weighted_uni(pred: &mut [u8], log2_denom: u32, weight: i32, offset: i32) {
    if log2_denom == 0 {
        for p in pred.iter_mut() {
            *p = ((*p as i32 * weight + offset).clamp(0, 255)) as u8;
        }
    } else {
        let round = 1i32 << (log2_denom - 1);
        for p in pred.iter_mut() {
            *p = ((*p as i32 * weight + round) >> log2_denom)
                .wrapping_add(offset)
                .clamp(0, 255) as u8;
        }
    }
}

/// Apply explicit weighted bi-prediction (spec 8.4.2.3.2).
/// Formula: `clip((p0*w0 + p1*w1 + round) >> (denom+1) + (o0+o1+1)>>1)`
#[allow(clippy::too_many_arguments)]
pub fn weighted_bi(
    pred_l0: &[u8],
    pred_l1: &[u8],
    output: &mut [u8],
    log2_denom: u32,
    w0: i32,
    o0: i32,
    w1: i32,
    o1: i32,
) {
    let round = 1i32 << log2_denom;
    let offset = (o0 + o1 + 1) >> 1;
    let shift = log2_denom + 1;
    for (o, (&a, &b)) in output.iter_mut().zip(pred_l0.iter().zip(pred_l1.iter())) {
        *o = ((a as i32 * w0 + b as i32 * w1 + round) >> shift)
            .wrapping_add(offset)
            .clamp(0, 255) as u8;
    }
}

/// Apply implicit weighted bi-prediction for B-slices (spec 8.4.2.3.2).
/// Uses POC-distance-derived weights with fixed log2_denom=5.
pub fn weighted_bi_implicit(pred_l0: &[u8], pred_l1: &[u8], output: &mut [u8], w0: i32, w1: i32) {
    let round = 1i32 << 5; // 1 << log2_denom where log2_denom=5
    for (o, (&a, &b)) in output.iter_mut().zip(pred_l0.iter().zip(pred_l1.iter())) {
        *o = ((a as i32 * w0 + b as i32 * w1 + round) >> 6).clamp(0, 255) as u8;
    }
}

#[cfg(all(test, feature = "host-tests"))]
mod tests {
    use super::*;
    use alloc::rc::Rc;

    fn make_ref_pic(width: u32, height: u32, y_data: Vec<u8>) -> Rc<DecodedPicture> {
        let uv_size = (width / 2 * height / 2) as usize;
        Rc::new(DecodedPicture {
            y: y_data,
            u: vec![128; uv_size],
            v: vec![128; uv_size],
            width,
            height,
            frame_num: 0,
            pic_order_cnt: 0,
            mv_l0: vec![],
            ref_idx_l0: vec![],
            ref_poc_l0: vec![],
            mv_l1: vec![],
            ref_idx_l1: vec![],
            mb_width: width / 16,
            is_intra: false,
        })
    }

    #[test]
    fn test_integer_pel_copy() {
        // 8x8 reference with gradient
        let mut y = vec![0u8; 64];
        for r in 0..8 {
            for c in 0..8 {
                y[r * 8 + c] = (r * 16 + c * 4) as u8;
            }
        }
        let pic = make_ref_pic(8, 8, y.clone());

        // Integer-pel MC (dx=0, dy=0) should be a direct copy
        let mut out = vec![0u8; 16]; // 4x4 block at (2,1)
        luma_mc_stride(&pic, 1, 2, 0, 0, 4, 4, &mut out, pic.width as usize, 0);
        for r in 0..4 {
            for c in 0..4 {
                assert_eq!(out[r * 4 + c], y[(r + 2) * 8 + (c + 1)]);
            }
        }
    }

    #[test]
    fn test_integer_pel_with_mv() {
        // 8x8 uniform reference
        let y = vec![42u8; 64];
        let pic = make_ref_pic(8, 8, y);

        // MV = (4, 8) in quarter-pel = (1, 2) full-pel offset
        let mut out = vec![0u8; 16];
        luma_mc_stride(&pic, 0, 0, 4, 8, 4, 4, &mut out, pic.width as usize, 0);
        // With uniform reference, all outputs should be 42
        assert!(out.iter().all(|&v| v == 42));
    }

    #[test]
    fn test_half_pel_horizontal() {
        // 16x1 reference: known values for 6-tap filter
        let mut y = vec![128u8; 16];
        // Set a step edge: left half = 0, right half = 255
        for c in 0..8 {
            y[c] = 0;
        }
        for c in 8..16 {
            y[c] = 255;
        }
        let pic = make_ref_pic(16, 1, y);

        // Half-pel horizontal at x=7 (the edge): MV dx=2 (half-pel), dy=0
        let mut out = [0u8; 1];
        luma_mc_stride(&pic, 7, 0, 2, 0, 1, 1, &mut out, pic.width as usize, 0);
        // 6-tap at x=7: samples at x=5..10 = [0, 0, 0, 255, 255, 255]
        // (0 - 0 + 0 + 20*255 - 5*255 + 255 + 16) >> 5 = (0 + 5100 - 1275 + 255 + 16) >> 5
        // = 4096 >> 5 = 128
        assert_eq!(out[0], 128);
    }

    #[test]
    fn test_half_pel_vertical() {
        // 1x16 reference with step edge at row 8
        let mut y = vec![0u8; 16];
        for r in 8..16 {
            y[r] = 255;
        }
        let pic = make_ref_pic(1, 16, y);

        let mut out = [0u8; 1];
        luma_mc_stride(&pic, 0, 7, 0, 2, 1, 1, &mut out, pic.width as usize, 0);
        // Same as horizontal but vertical: should also give 128
        assert_eq!(out[0], 128);
    }

    #[test]
    fn test_uniform_ref_all_frac_positions() {
        // Uniform reference: all fractional positions should give the same value
        let y = vec![100u8; 256];
        let pic = make_ref_pic(16, 16, y);

        for frac_x in 0..4 {
            for frac_y in 0..4 {
                let mut out = [0u8; 1];
                luma_mc_stride(
                    &pic,
                    4,
                    4,
                    frac_x,
                    frac_y,
                    1,
                    1,
                    &mut out,
                    pic.width as usize,
                    0,
                );
                assert_eq!(
                    out[0], 100,
                    "frac ({},{}) should give 100 for uniform ref",
                    frac_x, frac_y
                );
            }
        }
    }

    #[test]
    fn test_chroma_mc_integer() {
        let plane = vec![200u8; 64]; // 8x8 chroma
        let mut out = vec![0u8; 16]; // 4x4 block
        chroma_mc(&plane, 8, 8, 0, 0, 0, 0, 4, 4, &mut out);
        assert!(out.iter().all(|&v| v == 200));
    }

    #[test]
    fn test_chroma_mc_half_pel() {
        // 4x1 chroma: [0, 255, 0, 255]
        let plane = vec![0, 255, 0, 255];
        let mut out = [0u8; 1];
        // dx=4 = half-pel (4/8 = 0.5), dy=0
        chroma_mc(&plane, 4, 1, 0, 0, 4, 0, 1, 1, &mut out);
        // Bilinear: (8-4)*8*0 + 4*8*255 + 0 + 0 = 8160. (8160+32)>>6 = 128 (approx)
        assert_eq!(out[0], 128);
    }

    #[test]
    fn test_boundary_clipping() {
        // 4x4 reference, MV pointing outside
        let y: Vec<u8> = (0..16).collect();
        let pic = make_ref_pic(4, 4, y);

        // MC at (0,0) with MV=(-4, -4) in quarter-pel = (-1, -1) full-pel
        // Should clamp to (0,0) and read the top-left corner value
        let mut out = [0u8; 1];
        luma_mc_stride(&pic, 0, 0, -4, -4, 1, 1, &mut out, pic.width as usize, 0);
        assert_eq!(out[0], 0); // clamped to (0,0)
    }

    #[test]
    fn test_negative_mv_fractional() {
        // Verify negative MV fractional extraction
        let y = vec![128u8; 256];
        let pic = make_ref_pic(16, 16, y);

        // MV = (-1, -1) quarter-pel: frac should be (3, 3), int offset = (-1, -1)
        let mut out = [0u8; 1];
        luma_mc_stride(&pic, 8, 8, -1, -1, 1, 1, &mut out, pic.width as usize, 0);
        // Uniform ref, so result should be 128 regardless
        assert_eq!(out[0], 128);
    }
}
