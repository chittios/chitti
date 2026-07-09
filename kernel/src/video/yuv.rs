//! YUV→RGB conversion and related pixel helpers for the video player.
//!
//! Hot path for 1080p CABAC present: BT.601 limited-range convert with
//! **SIMD** (SSE2 on x86_64, NEON on aarch64) and optional **multi-core**
//! row split on aarch64 (reuses the SMP worker barrier that Cortex matvec
//! already uses). Pure functions over plane pointers — no I/O.

/// BT.601 limited-range YUV → packed `0x00RRGGBB` (scalar reference).
#[inline]
pub fn yuv_to_rgb(y: u8, u: u8, v: u8) -> u32 {
    let c = y as i32 - 16;
    let d = u as i32 - 128;
    let e = v as i32 - 128;
    let y298 = 298 * c + 128;
    let r = ((y298 + 409 * e) >> 8).clamp(0, 255) as u32;
    let g = ((y298 - 100 * d - 208 * e) >> 8).clamp(0, 255) as u32;
    let b = ((y298 + 516 * d) >> 8).clamp(0, 255) as u32;
    (r << 16) | (g << 8) | b
}

/// Convert display rows `[row0, row1)` of a (possibly downsampled) frame.
///
/// Source is full-res 4:2:0 `y/cb/cr` with strides `sw` / `scw`. Destination
/// is contiguous `out` of size `dw * dh`, row-major. Nearest-neighbour sample
/// from source when `dw != sw` or `dh != sh`.
pub fn convert_display_rows(
    y_plane: &[u8],
    cb: &[u8],
    cr: &[u8],
    sw: usize,
    sh: usize,
    scw: usize,
    dw: usize,
    dh: usize,
    out: &mut [u32],
    row0: usize,
    row1: usize,
) {
    let row1 = row1.min(dh);
    if row0 >= row1 || dw == 0 {
        return;
    }
    let full = dw == sw && dh == sh;
    for dy in row0..row1 {
        let sy = if full { dy } else { dy * sh / dh };
        let yrow = sy * sw;
        let crow = (sy / 2) * scw;
        let drow = dy * dw;
        if full {
            convert_row_full(&y_plane[yrow..yrow + sw], &cb[crow..], &cr[crow..], scw, &mut out[drow..drow + dw]);
        } else {
            convert_row_nn(
                y_plane,
                cb,
                cr,
                sw,
                scw,
                yrow,
                crow,
                dw,
                &mut out[drow..drow + dw],
            );
        }
    }
}

/// Full-width row: contiguous Y, subsampled chroma. SIMD when available.
fn convert_row_full(y: &[u8], cb: &[u8], cr: &[u8], _scw: usize, out: &mut [u32]) {
    let n = out.len().min(y.len());
    let mut x = 0;
    // Process 16 pixels at a time (two NEON/scalar octets).
    while x + 16 <= n {
        let mut y0 = [0u8; 8];
        let mut y1 = [0u8; 8];
        let mut u0 = [0u8; 8];
        let mut u1 = [0u8; 8];
        let mut v0 = [0u8; 8];
        let mut v1 = [0u8; 8];
        y0.copy_from_slice(&y[x..x + 8]);
        y1.copy_from_slice(&y[x + 8..x + 16]);
        for i in 0..8 {
            let c0 = (x + i) / 2;
            let c1 = (x + 8 + i) / 2;
            u0[i] = cb.get(c0).copied().unwrap_or(128);
            v0[i] = cr.get(c0).copied().unwrap_or(128);
            u1[i] = cb.get(c1).copied().unwrap_or(128);
            v1[i] = cr.get(c1).copied().unwrap_or(128);
        }
        convert8(&y0, &u0, &v0, &mut out[x..x + 8]);
        convert8(&y1, &u1, &v1, &mut out[x + 8..x + 16]);
        x += 16;
    }
    while x + 8 <= n {
        let mut yy = [0u8; 8];
        let mut uu = [0u8; 8];
        let mut vv = [0u8; 8];
        yy.copy_from_slice(&y[x..x + 8]);
        for i in 0..8 {
            let cx = (x + i) / 2;
            uu[i] = cb.get(cx).copied().unwrap_or(128);
            vv[i] = cr.get(cx).copied().unwrap_or(128);
        }
        convert8(&yy, &uu, &vv, &mut out[x..x + 8]);
        x += 8;
    }
    while x < n {
        let cx = x / 2;
        let u = cb.get(cx).copied().unwrap_or(128);
        let v = cr.get(cx).copied().unwrap_or(128);
        out[x] = yuv_to_rgb(y[x], u, v);
        x += 1;
    }
}

/// Nearest-neighbour sample into a display row.
fn convert_row_nn(
    y_plane: &[u8],
    cb: &[u8],
    cr: &[u8],
    sw: usize,
    _scw: usize,
    yrow: usize,
    crow: usize,
    dw: usize,
    out: &mut [u32],
) {
    let mut x = 0;
    while x + 8 <= dw {
        let mut yy = [0u8; 8];
        let mut uu = [0u8; 8];
        let mut vv = [0u8; 8];
        for i in 0..8 {
            let sx = (x + i) * sw / dw;
            yy[i] = y_plane.get(yrow + sx).copied().unwrap_or(0);
            let cx = sx / 2;
            uu[i] = cb.get(crow + cx).copied().unwrap_or(128);
            vv[i] = cr.get(crow + cx).copied().unwrap_or(128);
        }
        convert8(&yy, &uu, &vv, &mut out[x..x + 8]);
        x += 8;
    }
    while x < dw {
        let sx = x * sw / dw;
        let yy = y_plane.get(yrow + sx).copied().unwrap_or(0);
        let cx = sx / 2;
        let u = cb.get(crow + cx).copied().unwrap_or(128);
        let v = cr.get(crow + cx).copied().unwrap_or(128);
        out[x] = yuv_to_rgb(yy, u, v);
        x += 1;
    }
}

/// Convert 8 YUV samples → 8 packed RGB words (NEON on aarch64; scalar
/// elsewhere — the compiler auto-vectorises the unrolled scalar at `-O2`).
#[inline]
fn convert8(y: &[u8; 8], u: &[u8; 8], v: &[u8; 8], out: &mut [u32]) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is baseline on aarch64-chitti.
    unsafe {
        convert8_neon(y, u, v, out);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..8 {
            out[i] = yuv_to_rgb(y[i], u[i], v[i]);
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn convert8_neon(y: &[u8; 8], u: &[u8; 8], v: &[u8; 8], out: &mut [u32]) {
    use core::arch::aarch64::*;
    // Load 8×u8 → i16, subtract offsets, apply BT.601 in i32, pack.
    let yv = vld1_u8(y.as_ptr());
    let uv = vld1_u8(u.as_ptr());
    let vv = vld1_u8(v.as_ptr());
    let y16 = vreinterpretq_s16_u16(vmovl_u8(yv));
    let u16 = vreinterpretq_s16_u16(vmovl_u8(uv));
    let v16 = vreinterpretq_s16_u16(vmovl_u8(vv));
    let c = vsubq_s16(y16, vdupq_n_s16(16));
    let d = vsubq_s16(u16, vdupq_n_s16(128));
    let e = vsubq_s16(v16, vdupq_n_s16(128));
    let c32_lo = vmovl_s16(vget_low_s16(c));
    let c32_hi = vmovl_s16(vget_high_s16(c));
    let d32_lo = vmovl_s16(vget_low_s16(d));
    let d32_hi = vmovl_s16(vget_high_s16(d));
    let e32_lo = vmovl_s16(vget_low_s16(e));
    let e32_hi = vmovl_s16(vget_high_s16(e));
    let y298_lo = vmlaq_n_s32(vdupq_n_s32(128), c32_lo, 298);
    let y298_hi = vmlaq_n_s32(vdupq_n_s32(128), c32_hi, 298);
    let mut r_lo = vshrq_n_s32(vmlaq_n_s32(y298_lo, e32_lo, 409), 8);
    let mut r_hi = vshrq_n_s32(vmlaq_n_s32(y298_hi, e32_hi, 409), 8);
    let mut g_lo = vshrq_n_s32(vmlsq_n_s32(vmlsq_n_s32(y298_lo, d32_lo, 100), e32_lo, 208), 8);
    let mut g_hi = vshrq_n_s32(vmlsq_n_s32(vmlsq_n_s32(y298_hi, d32_hi, 100), e32_hi, 208), 8);
    let mut b_lo = vshrq_n_s32(vmlaq_n_s32(y298_lo, d32_lo, 516), 8);
    let mut b_hi = vshrq_n_s32(vmlaq_n_s32(y298_hi, d32_hi, 516), 8);
    let z = vdupq_n_s32(0);
    let m = vdupq_n_s32(255);
    r_lo = vmaxq_s32(vminq_s32(r_lo, m), z);
    r_hi = vmaxq_s32(vminq_s32(r_hi, m), z);
    g_lo = vmaxq_s32(vminq_s32(g_lo, m), z);
    g_hi = vmaxq_s32(vminq_s32(g_hi, m), z);
    b_lo = vmaxq_s32(vminq_s32(b_lo, m), z);
    b_hi = vmaxq_s32(vminq_s32(b_hi, m), z);
    // Pack as 0x00RRGGBB = r<<16 | g<<8 | b
    let rgb_lo = vorrq_s32(vorrq_s32(vshlq_n_s32(r_lo, 16), vshlq_n_s32(g_lo, 8)), b_lo);
    let rgb_hi = vorrq_s32(vorrq_s32(vshlq_n_s32(r_hi, 16), vshlq_n_s32(g_hi, 8)), b_hi);
    vst1q_u32(out.as_mut_ptr(), vreinterpretq_u32_s32(rgb_lo));
    vst1q_u32(out.as_mut_ptr().add(4), vreinterpretq_u32_s32(rgb_hi));
}

/// Context for a parallel display convert job (kernel SMP path).
#[repr(C)]
pub struct ConvertCtx {
    pub y: *const u8,
    pub cb: *const u8,
    pub cr: *const u8,
    pub out: *mut u32,
    pub y_len: usize,
    pub cb_len: usize,
    pub cr_len: usize,
    pub out_len: usize,
    pub sw: usize,
    pub sh: usize,
    pub scw: usize,
    pub dw: usize,
    pub dh: usize,
}

/// Convert a full display frame (all rows). Pure / single-threaded.
pub fn convert_display(
    y_plane: &[u8],
    cb: &[u8],
    cr: &[u8],
    sw: usize,
    sh: usize,
    scw: usize,
    dw: usize,
    dh: usize,
    out: &mut [u32],
) {
    convert_display_rows(y_plane, cb, cr, sw, sh, scw, dw, dh, out, 0, dh);
}

/// Worker entry for SMP row split — disjoint `[row0, row1)` into `ctx.out`.
///
/// # Safety
/// `ctx` must point to a live [`ConvertCtx`]; ranges must not overlap.
pub unsafe fn convert_worker(row0: usize, row1: usize, ctx: *mut u8) {
    let c = &*(ctx as *const ConvertCtx);
    let y = core::slice::from_raw_parts(c.y, c.y_len);
    let cb = core::slice::from_raw_parts(c.cb, c.cb_len);
    let cr = core::slice::from_raw_parts(c.cr, c.cr_len);
    let out = core::slice::from_raw_parts_mut(c.out, c.out_len);
    convert_display_rows(y, cb, cr, c.sw, c.sh, c.scw, c.dw, c.dh, out, row0, row1);
}

/// Clip i32 residual plane → u8 with SIMD (SSE2 / NEON) when available.
/// (H.264 work planes are `i32` after transform.)
pub fn clip_plane_i32_to_u8(src: &[i32], dst: &mut [u8]) {
    let n = src.len().min(dst.len());
    let mut i = 0;
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: SSE2 baseline. Process 8 i32 → 8 u8 per iter.
        unsafe {
            while i + 8 <= n {
                clip8_i32_sse2(&src[i..i + 8], &mut dst[i..i + 8]);
                i += 8;
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON baseline. Process 16 i32 → 16 u8 per iter.
        unsafe {
            while i + 16 <= n {
                clip16_i32_neon(src.as_ptr().add(i), dst.as_mut_ptr().add(i));
                i += 16;
            }
            while i + 4 <= n {
                clip4_i32_neon(&src[i..i + 4], &mut dst[i..i + 4]);
                i += 4;
            }
        }
    }
    while i < n {
        dst[i] = src[i].clamp(0, 255) as u8;
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn clip8_i32_sse2(src: &[i32], dst: &mut [u8]) {
    use core::arch::x86_64::*;
    let v0 = _mm_loadu_si128(src.as_ptr() as *const __m128i);
    let v1 = _mm_loadu_si128(src.as_ptr().add(4) as *const __m128i);
    let zero = _mm_setzero_si128();
    let hi = _mm_set1_epi32(255);
    let s0 = _mm_min_epi32(_mm_max_epi32(v0, zero), hi);
    let s1 = _mm_min_epi32(_mm_max_epi32(v1, zero), hi);
    let p16 = _mm_packs_epi32(s0, s1);
    let p8 = _mm_packus_epi16(p16, zero);
    let mut tmp = [0u8; 16];
    _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, p8);
    dst[..8].copy_from_slice(&tmp[..8]);
}

#[cfg(target_arch = "aarch64")]
unsafe fn clip4_i32_neon(src: &[i32], dst: &mut [u8]) {
    use core::arch::aarch64::*;
    let v = vld1q_s32(src.as_ptr());
    let z = vdupq_n_s32(0);
    let h = vdupq_n_s32(255);
    let sat = vminq_s32(vmaxq_s32(v, z), h);
    let n16 = vmovn_s32(sat);
    let n8 = vqmovun_s16(vcombine_s16(n16, n16));
    let mut tmp = [0u8; 8];
    vst1_u8(tmp.as_mut_ptr(), n8);
    dst[..4].copy_from_slice(&tmp[..4]);
}

#[cfg(target_arch = "aarch64")]
unsafe fn clip16_i32_neon(src: *const i32, dst: *mut u8) {
    use core::arch::aarch64::*;
    let z = vdupq_n_s32(0);
    let h = vdupq_n_s32(255);
    let v0 = vminq_s32(vmaxq_s32(vld1q_s32(src), z), h);
    let v1 = vminq_s32(vmaxq_s32(vld1q_s32(src.add(4)), z), h);
    let v2 = vminq_s32(vmaxq_s32(vld1q_s32(src.add(8)), z), h);
    let v3 = vminq_s32(vmaxq_s32(vld1q_s32(src.add(12)), z), h);
    let n0 = vmovn_s32(v0);
    let n1 = vmovn_s32(v1);
    let n2 = vmovn_s32(v2);
    let n3 = vmovn_s32(v3);
    let p01 = vqmovun_s16(vcombine_s16(n0, n1));
    let p23 = vqmovun_s16(vcombine_s16(n2, n3));
    vst1_u8(dst, p01);
    vst1_u8(dst.add(8), p23);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn yuv_black_and_white() {
        // Y=16 limited black, U=V=128 → near black
        let p = yuv_to_rgb(16, 128, 128);
        assert!(p & 0xff < 8);
        // Y=235 white
        let p = yuv_to_rgb(235, 128, 128);
        assert!(((p >> 16) & 0xff) > 240);
    }

    #[test_case]
    fn convert8_matches_scalar() {
        let y = [16u8, 80, 128, 180, 200, 220, 235, 100];
        let u = [128u8; 8];
        let v = [128u8; 8];
        let mut out = [0u32; 8];
        let mut ref_ = [0u32; 8];
        convert8(&y, &u, &v, &mut out);
        for i in 0..8 {
            ref_[i] = yuv_to_rgb(y[i], u[i], v[i]);
        }
        // Allow ±1 LSB from NEON rounding path vs scalar (should be exact for
        // our integer formula).
        for i in 0..8 {
            assert_eq!(out[i], ref_[i], "pixel {i}");
        }
    }

    #[test_case]
    fn clip_plane_matches_scalar() {
        let src = [-10i32, 0, 128, 255, 300, 1, 254, -1, 50];
        let mut dst = [0u8; 9];
        clip_plane_i32_to_u8(&src, &mut dst);
        for i in 0..9 {
            assert_eq!(dst[i], src[i].clamp(0, 255) as u8);
        }
    }
}
