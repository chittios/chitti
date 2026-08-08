//! Minimal H.264 **Baseline** encoder for screen recording.
//!
//! What real OSes do: capture RGB → convert to YUV 4:2:0 → encode H.264/HEVC/VP9
//! → mux into MP4/MOV/WebM. This module is the encode half of that path for
//! ChittiOS, deliberately scoped so a cooperative kernel can keep up:
//!
//! * **Baseline profile, CAVLC only** (matches our decoder's simple path).
//! * **I_PCM** for keyframes and changed macroblocks (bit-exact; the residual
//!   I_16x16 path still mis-decodes at real screen sizes — kept dead for a
//!   later rewrite).
//! * **P_Skip** for macroblocks that match the previous source frame
//!   (the win for desktop UI — static regions cost one skip run, not a frame).
//! * Fixed QP, zero-motion only (no motion search).
//!
//! Pure over plane buffers: no I/O. The shell captures, this compresses, the
//! MP4 muxer wraps. Round-tripped through our own decoder in unit tests.

use super::cavlc::{
    CHROMA_DC_CT_BITS, CHROMA_DC_CT_LEN, CHROMA_DC_TZ_BITS, CHROMA_DC_TZ_LEN, COEFF_TOKEN_BITS,
    COEFF_TOKEN_LEN, RUN_BITS, RUN_LEN, TOTAL_ZEROS_BITS, TOTAL_ZEROS_LEN,
};
use super::intra::{intra16x16, intra_chroma};
use super::transform::{dequant_4x4, idct_4x4, inverse_scan_4x4, ZIGZAG_4X4};
use super::{Pps, Sps};
use crate::video::bits::{make_nal, BitWriter};
use alloc::vec;
use alloc::vec::Vec;

/// Multiplier factors for forward quant (H.264 §8.5.9 / JM `quant_coef`).
const MF: [[i32; 3]; 6] = [
    [13107, 5243, 8066],
    [11916, 4660, 7490],
    [10082, 4194, 6554],
    [9362, 3647, 5825],
    [8192, 3355, 5243],
    [7282, 2893, 4559],
];

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

/// Forward 4×4 core transform (§8.5.12.1 inverse's dual), raster in/out.
fn fdt_4x4(block: &mut [i32; 16]) {
    // Rows.
    for i in 0..4 {
        let r = i * 4;
        let z0 = block[r] + block[r + 3];
        let z1 = block[r + 1] + block[r + 2];
        let z2 = block[r + 1] - block[r + 2];
        let z3 = block[r] - block[r + 3];
        block[r] = z0 + z1;
        block[r + 1] = (z3 << 1) + z2;
        block[r + 2] = z0 - z1;
        block[r + 3] = z3 - (z2 << 1);
    }
    // Columns.
    for j in 0..4 {
        let z0 = block[j] + block[j + 12];
        let z1 = block[j + 4] + block[j + 8];
        let z2 = block[j + 4] - block[j + 8];
        let z3 = block[j] - block[j + 12];
        block[j] = z0 + z1;
        block[j + 4] = (z3 << 1) + z2;
        block[j + 8] = z0 - z1;
        block[j + 12] = z3 - (z2 << 1);
    }
}

/// Forward quantise a raster 4×4 AC block at `qp`. `is_intra` selects the
/// rounding offset (spec: 1/3 vs 1/6 of the quant step).
fn quant_4x4(block: &mut [i32; 16], qp: u32, is_intra: bool) {
    let qbits = 15 + qp / 6;
    let mf_row = &MF[(qp % 6) as usize];
    let f = if is_intra {
        (1i32 << qbits) / 3
    } else {
        (1i32 << qbits) / 6
    };
    for i in 0..4 {
        for j in 0..4 {
            let idx = i * 4 + j;
            let mf = mf_row[pos_group(i, j)];
            let c = block[idx];
            let sign = if c < 0 { -1 } else { 1 };
            let level = ((c.unsigned_abs() as i32 * mf + f) >> qbits) * sign;
            block[idx] = level;
        }
    }
}

/// Forward 4×4 Hadamard + DC quant for Intra_16x16 luma DCs.
fn quant_luma_dc(dc: &mut [i32; 16], qp: u32) {
    // Hadamard (same butterflies as the inverse, unscaled).
    hadamard_4x4_fwd(dc);
    let qbits = 16 + qp / 6;
    let mf = MF[(qp % 6) as usize][0];
    let f = (1i32 << qbits) / 3;
    for v in dc.iter_mut() {
        let sign = if *v < 0 { -1 } else { 1 };
        *v = ((v.unsigned_abs() as i32 * mf + f * 2) >> (qbits + 1)) * sign;
    }
}

/// Forward 2×2 Hadamard + quant for chroma DCs.
fn quant_chroma_dc(dc: &mut [i32; 4], qp: u32) {
    let a = dc[0] + dc[1];
    let b = dc[0] - dc[1];
    let c = dc[2] + dc[3];
    let d = dc[2] - dc[3];
    let t = [a + c, b + d, a - c, b - d];
    let qbits = 16 + qp / 6;
    let mf = MF[(qp % 6) as usize][0];
    let f = (1i32 << qbits) / 3;
    for k in 0..4 {
        let sign = if t[k] < 0 { -1 } else { 1 };
        dc[k] = ((t[k].unsigned_abs() as i32 * mf + f * 2) >> (qbits + 1)) * sign;
    }
}

fn hadamard_4x4_fwd(m: &mut [i32; 16]) {
    for i in 0..4 {
        let r = i * 4;
        let a = m[r] + m[r + 3];
        let b = m[r + 1] + m[r + 2];
        let c = m[r + 1] - m[r + 2];
        let d = m[r] - m[r + 3];
        m[r] = a + b;
        m[r + 1] = c + d;
        m[r + 2] = a - b;
        m[r + 3] = d - c;
    }
    for j in 0..4 {
        let a = m[j] + m[j + 12];
        let b = m[j + 4] + m[j + 8];
        let c = m[j + 4] - m[j + 8];
        let d = m[j] - m[j + 12];
        m[j] = a + b;
        m[j + 4] = c + d;
        m[j + 8] = a - b;
        m[j + 12] = d - c;
    }
}

/// Write a CAVLC residual block (scan-order coefficients). Inverse of
/// [`super::cavlc::residual_block`].
fn write_residual(w: &mut BitWriter, scan: &[i32; 16], max_coeff: usize, n_c: i32) {
    // Non-zeros in scan order (low → high frequency).
    let mut nz: Vec<(usize, i32)> = Vec::new();
    for i in 0..max_coeff {
        if scan[i] != 0 {
            nz.push((i, scan[i]));
        }
    }
    let total_coeff = nz.len();
    let trailing_ones = {
        let mut t = 0usize;
        for i in (0..total_coeff).rev() {
            if t < 3 && nz[i].1.abs() == 1 {
                t += 1;
            } else {
                break;
            }
        }
        t
    };
    let tt = (total_coeff << 2) | trailing_ones;
    write_vlc(
        w,
        if n_c == -1 {
            (&CHROMA_DC_CT_LEN[..], &CHROMA_DC_CT_BITS[..])
        } else {
            let tab = if n_c < 2 {
                0
            } else if n_c < 4 {
                1
            } else if n_c < 8 {
                2
            } else {
                3
            };
            (&COEFF_TOKEN_LEN[tab][..], &COEFF_TOKEN_BITS[tab][..])
        },
        tt,
    );
    if total_coeff == 0 {
        return;
    }

    // Trailing-ones signs, highest frequency first.
    for i in 0..trailing_ones {
        let lv = nz[total_coeff - 1 - i].1;
        w.bit(if lv < 0 { 1 } else { 0 });
    }

    // Remaining levels, highest frequency first.
    let mut suffix_length: u32 = if total_coeff > 10 && trailing_ones < 3 { 1 } else { 0 };
    for i in trailing_ones..total_coeff {
        let lv = nz[total_coeff - 1 - i].1;
        let mut level_code = if lv > 0 {
            lv * 2 - 2
        } else {
            (-lv) * 2 - 1
        };
        if i == trailing_ones && trailing_ones < 3 {
            level_code -= 2;
        }
        if level_code < 0 {
            level_code = 0;
        }
        let (prefix, suffix, suffix_size) = encode_level(level_code, suffix_length);
        for _ in 0..prefix {
            w.bit(0);
        }
        w.bit(1);
        if suffix_size > 0 {
            w.u(suffix_size, suffix);
        }
        if suffix_length == 0 {
            suffix_length = 1;
        }
        let abs_level = lv.unsigned_abs();
        if abs_level > (3u32 << (suffix_length - 1)) && suffix_length < 6 {
            suffix_length += 1;
        }
    }

    // total_zeros = zeros in [0 ..= last_nz_pos], not counting the non-zeros.
    let last_pos = nz[total_coeff - 1].0;
    let total_zeros = last_pos + 1 - total_coeff;
    if total_coeff < max_coeff {
        if n_c == -1 {
            write_vlc(
                w,
                (
                    &CHROMA_DC_TZ_LEN[total_coeff - 1][..],
                    &CHROMA_DC_TZ_BITS[total_coeff - 1][..],
                ),
                total_zeros,
            );
        } else {
            write_vlc(
                w,
                (
                    &TOTAL_ZEROS_LEN[total_coeff - 1][..],
                    &TOTAL_ZEROS_BITS[total_coeff - 1][..],
                ),
                total_zeros,
            );
        }
    }

    // run_before: zeros between successive non-zeros, high-freq side first
    // (all but the lowest-frequency non-zero).
    let mut zeros_left = total_zeros;
    for i in (1..total_coeff).rev() {
        if zeros_left == 0 {
            break;
        }
        // zeros between nz[i-1] and nz[i]
        let run = nz[i].0 - nz[i - 1].0 - 1;
        let run_row = zeros_left.min(7) - 1;
        write_vlc(w, (&RUN_LEN[run_row][..], &RUN_BITS[run_row][..]), run);
        zeros_left -= run;
    }
}

fn encode_level(level_code: i32, suffix_length: u32) -> (u32, u32, u32) {
    // level_code >= 0 after our mapping.
    let lc = level_code as u32;
    if suffix_length == 0 {
        if lc < 14 {
            return (lc, 0, 0);
        }
        if lc < 30 {
            return (14, lc - 14, 4);
        }
        // Escape: prefix 15 + longer suffix. Keep it simple with prefix 15.
        let mut rem = lc - 15;
        let mut prefix = 15u32;
        while rem >= (1 << (prefix - 3)) {
            rem -= 1 << (prefix - 3);
            prefix += 1;
            if prefix > 25 {
                break;
            }
        }
        return (prefix, rem, prefix - 3);
    }
    let mask = (1u32 << suffix_length) - 1;
    let prefix = lc >> suffix_length;
    let suffix = lc & mask;
    if prefix < 15 {
        return (prefix, suffix, suffix_length);
    }
    // Large escape.
    let mut rem = lc - (15 << suffix_length);
    let mut p = 15u32;
    while rem >= (1 << (p - 3)) {
        rem -= 1 << (p - 3);
        p += 1;
        if p > 25 {
            break;
        }
    }
    (p, rem, p - 3)
}

fn write_vlc(w: &mut BitWriter, tables: (&[u8], &[u8]), symbol: usize) {
    let (lens, bits) = tables;
    if symbol >= lens.len() || lens[symbol] == 0 {
        // Fallback: should not happen for valid residual; write a safe zero-token.
        // For coeff_token 0 the short code is usually "1".
        if !lens.is_empty() && lens[0] != 0 {
            w.u(lens[0] as u32, bits[0] as u32);
        } else {
            w.bit(1);
        }
        return;
    }
    w.u(lens[symbol] as u32, bits[symbol] as u32);
}

/// Zig-zag scan a raster 4×4 into scan order.
fn zigzag_scan(raster: &[i32; 16]) -> [i32; 16] {
    let mut scan = [0i32; 16];
    for (n, &r) in ZIGZAG_4X4.iter().enumerate() {
        scan[n] = raster[r];
    }
    scan
}

/// nC neighbour context for CAVLC (simplified: use 0 when no neighbour, else
/// average — good enough for encode; decoder only needs a matching stream).
fn nc_from(nnz: &[u8], mb_w: usize, bx: usize, by: usize) -> i32 {
    let mut a = -1i32;
    let mut b = -1i32;
    if bx > 0 {
        a = nnz[by * (mb_w * 4) + (bx - 1)] as i32;
    }
    if by > 0 {
        b = nnz[(by - 1) * (mb_w * 4) + bx] as i32;
    }
    match (a >= 0, b >= 0) {
        (true, true) => (a + b + 1) >> 1,
        (true, false) => a,
        (false, true) => b,
        (false, false) => 0,
    }
}

/// Convert packed `0x00RRGGBB` to planar YUV 4:2:0 (BT.601 limited range).
/// `w`/`h` must be even. Output lengths: Y=w*h, U=V=(w/2)*(h/2).
pub fn rgb32_to_yuv420(w: usize, h: usize, px: &[u32]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; w * h];
    let cw = w / 2;
    let ch = h / 2;
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for j in 0..h {
        for i in 0..w {
            let p = px[j * w + i];
            let r = ((p >> 16) & 255) as i32;
            let g = ((p >> 8) & 255) as i32;
            let b = (p & 255) as i32;
            let yy = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
            y[j * w + i] = yy.clamp(0, 255) as u8;
            if j % 2 == 0 && i % 2 == 0 {
                let uu = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
                let vv = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
                u[(j / 2) * cw + i / 2] = uu.clamp(0, 255) as u8;
                v[(j / 2) * cw + i / 2] = vv.clamp(0, 255) as u8;
            }
        }
    }
    (y, u, v)
}

/// Round dimensions down to a multiple of 16 (macroblock grid), min 16.
pub fn align_mb(w: u32, h: u32) -> (u32, u32) {
    let nw = (w / 16 * 16).max(16);
    let nh = (h / 16 * 16).max(16);
    (nw, nh)
}

/// Centre-crop `px` (src_w×src_h) into a dst_w×dst_h buffer (both must fit).
pub fn crop_rgb32(src_w: usize, src_h: usize, px: &[u32], dst_w: usize, dst_h: usize) -> Vec<u32> {
    let ox = src_w.saturating_sub(dst_w) / 2;
    let oy = src_h.saturating_sub(dst_h) / 2;
    let mut out = Vec::with_capacity(dst_w * dst_h);
    for j in 0..dst_h {
        let row = (oy + j) * src_w + ox;
        out.extend_from_slice(&px[row..row + dst_w]);
    }
    out
}

/// SAD of a 16×16 luma block vs the previous frame (for P_Skip decisions).
fn sad16(cur: &[u8], prev: &[u8], w: usize, bx: usize, by: usize) -> u32 {
    let mut s = 0u32;
    for j in 0..16 {
        let off = (by + j) * w + bx;
        for i in 0..16 {
            s += cur[off + i].abs_diff(prev[off + i]) as u32;
        }
    }
    s
}

/// 4×4 block order inside a macroblock (H.264 inverse 8×8 zig-zag of 4×4s).
/// Residual blocks and DC placement must use this order — not raster.
const BLK_XY: [(usize, usize); 16] = [
    (0, 0),
    (1, 0),
    (0, 1),
    (1, 1),
    (2, 0),
    (3, 0),
    (2, 1),
    (3, 1),
    (0, 2),
    (1, 2),
    (0, 3),
    (1, 3),
    (2, 2),
    (3, 2),
    (2, 3),
    (3, 3),
];

/// Screen-content encoder state.
pub struct Encoder {
    pub w: usize,
    pub h: usize,
    mb_w: usize,
    mb_h: usize,
    qp: u32,
    /// Reconstructed reference (luma/chroma) — what the decoder will have.
    prev_y: Vec<u8>,
    prev_u: Vec<u8>,
    prev_v: Vec<u8>,
    /// Previous **source** luma — skip decisions compare captures, not recon
    /// (quantisation noise would otherwise defeat P_Skip on every static frame).
    prev_src_y: Vec<u8>,
    frame_num: u32,
    idr_pic_id: u32,
    pub sps_nal: Vec<u8>,
    pub pps_nal: Vec<u8>,
    /// Skip threshold: mean |ΔY| below this → P_Skip. ~3 is good for flat UI.
    skip_mean: u32,
}

impl Encoder {
    /// Build an encoder for a `w`×`h` stream (must be multiples of 16).
    pub fn new(w: usize, h: usize, qp: u32) -> Result<Self, &'static str> {
        if w < 16 || h < 16 || w % 16 != 0 || h % 16 != 0 {
            return Err("h264enc: width/height must be multiples of 16");
        }
        if !(1..=51).contains(&qp) {
            return Err("h264enc: qp out of range");
        }
        let mb_w = w / 16;
        let mb_h = h / 16;
        let sps_rbsp = write_sps(mb_w as u32, mb_h as u32);
        let pps_rbsp = write_pps(qp);
        Ok(Encoder {
            w,
            h,
            mb_w,
            mb_h,
            qp,
            prev_y: vec![0; w * h],
            prev_u: vec![128; (w / 2) * (h / 2)],
            prev_v: vec![128; (w / 2) * (h / 2)],
            prev_src_y: vec![0; w * h],
            frame_num: 0,
            idr_pic_id: 0,
            sps_nal: make_nal(3, 7, &sps_rbsp),
            pps_nal: make_nal(3, 8, &pps_rbsp),
            skip_mean: 3,
        })
    }

    /// Encode one RGB frame. `force_idr` makes a keyframe (first frame must).
    /// Returns a length-prefixed AVCC access unit (one or more NALs) ready for
    /// the MP4 sample table — **not** including SPS/PPS (those live in `avcC`).
    pub fn encode_rgb32(&mut self, px: &[u32], force_idr: bool) -> Result<Vec<u8>, &'static str> {
        if px.len() < self.w * self.h {
            return Err("h264enc: short pixel buffer");
        }
        let (y, u, v) = rgb32_to_yuv420(self.w, self.h, px);
        self.encode_yuv(&y, &u, &v, force_idr)
    }

    pub fn encode_yuv(
        &mut self,
        y: &[u8],
        u: &[u8],
        v: &[u8],
        force_idr: bool,
    ) -> Result<Vec<u8>, &'static str> {
        // Only the caller forces IDR (first frame / keyint). Do **not** treat
        // frame_num==0 as IDR — after an IDR the counter is 0 and the next
        // frame must be a P, or every frame re-encodes as a keyframe.
        let is_idr = force_idr;
        if is_idr {
            self.frame_num = 0;
        }
        let mut w = BitWriter::with_capacity(self.w * self.h / 4);

        // --- slice header ---
        w.ue(0); // first_mb_in_slice
        // slice_type: 7 = I (all), 5 = P (all). Using non-restricted types.
        w.ue(if is_idr { 7 } else { 5 });
        w.ue(0); // pps_id
        let max_fn_bits = 4; // log2_max_frame_num_minus4 + 4 = 4 → minus4 = 0
        w.u(max_fn_bits, self.frame_num & 0xf);
        if is_idr {
            w.ue(self.idr_pic_id);
        }
        // pic_order_cnt_type = 0 → poc_lsb
        w.u(4, (self.frame_num * 2) & 0xf); // log2_max_poc_lsb = 4
        if !is_idr {
            w.flag(false); // num_ref_idx_active_override_flag
            // ref_pic_list_modification: bit 0 = no modification
            w.flag(false);
        } else {
            // dec_ref_pic_marking for IDR
            w.flag(false); // no_output_of_prior_pics
            w.flag(false); // long_term_reference
        }
        if !is_idr {
            // adaptive_ref_pic_marking_mode_flag = 0 (sliding window)
            w.flag(false);
        }
        w.se(0); // slice_qp_delta (use pic_init_qp)
        // deblocking_filter_control not present in our PPS

        // nnz maps for CAVLC nC (4×4 block grid).
        let mut nnz_y = vec![0u8; self.mb_w * self.mb_h * 16];
        let mut recon_y = y.to_vec();
        let mut recon_u = u.to_vec();
        let mut recon_v = v.to_vec();

        let mut skip_run: u32 = 0;
        let total_mb = self.mb_w * self.mb_h;

        for mb in 0..total_mb {
            let mb_x = mb % self.mb_w;
            let mb_y = mb / self.mb_w;
            let bx = mb_x * 16;
            let by = mb_y * 16;

            // Compare source-to-source so quantisation noise on the recon
            // does not force every static desktop MB to re-encode.
            let do_skip = !is_idr
                && sad16(y, &self.prev_src_y, self.w, bx, by) < 16 * 16 * self.skip_mean;

            if do_skip {
                skip_run += 1;
                // Reconstruct = previous (already in recon from copy below if we
                // seed recon from prev for skips).
                for j in 0..16 {
                    let off = (by + j) * self.w + bx;
                    recon_y[off..off + 16].copy_from_slice(&self.prev_y[off..off + 16]);
                }
                let cw = self.w / 2;
                let (cbx, cby) = (bx / 2, by / 2);
                for j in 0..8 {
                    let off = (cby + j) * cw + cbx;
                    recon_u[off..off + 8].copy_from_slice(&self.prev_u[off..off + 8]);
                    recon_v[off..off + 8].copy_from_slice(&self.prev_v[off..off + 8]);
                }
                // Clear nnz for this MB so neighbours see zeros.
                for j in 0..4 {
                    for i in 0..4 {
                        nnz_y[(mb_y * 4 + j) * (self.mb_w * 4) + mb_x * 4 + i] = 0;
                    }
                }
                continue;
            }

            if !is_idr {
                w.ue(skip_run); // mb_skip_run
                skip_run = 0;
            }

            // I_PCM: raw samples. The residual I16 path still fails closed at
            // real screen sizes (host selftest: 64×48 paints, 720×448 decodes
            // near-black). PCM is large but bit-exact; P_Skip compresses static UI.
            self.encode_pcm_mb(
                &mut w,
                y,
                u,
                v,
                &mut recon_y,
                &mut recon_u,
                &mut recon_v,
                &mut nnz_y,
                mb_x,
                mb_y,
                is_idr,
            );
        }
        if !is_idr {
            // Trailing skip run (may be zero).
            w.ue(skip_run);
        }

        let rbsp = w.finish();
        let nal_type = if is_idr { 5 } else { 1 };
        let nal = make_nal(2, nal_type, &rbsp);

        // AVCC length-prefix.
        let mut au = Vec::with_capacity(4 + nal.len());
        au.extend_from_slice(&(nal.len() as u32).to_be_bytes());
        au.extend_from_slice(&nal);

        self.prev_y = recon_y;
        self.prev_u = recon_u;
        self.prev_v = recon_v;
        self.prev_src_y = y.to_vec();
        if is_idr {
            self.idr_pic_id = self.idr_pic_id.wrapping_add(1);
        }
        // Advance for the next picture (wraps at log2_max_frame_num = 4 bits).
        self.frame_num = (self.frame_num + 1) & 0xf;
        Ok(au)
    }

    /// I_PCM macroblock (mb_type 25 in an I slice, 30 in a P slice).
    fn encode_pcm_mb(
        &self,
        w: &mut BitWriter,
        y: &[u8],
        u: &[u8],
        v: &[u8],
        recon_y: &mut [u8],
        recon_u: &mut [u8],
        recon_v: &mut [u8],
        nnz_y: &mut [u8],
        mb_x: usize,
        mb_y: usize,
        is_i_slice: bool,
    ) {
        // Table 7-11: I_PCM = 25; in a P slice Intra types are offset by +5.
        w.ue(if is_i_slice { 25 } else { 30 });
        w.byte_align_zeros();
        let bx = mb_x * 16;
        let by = mb_y * 16;
        let cw = self.w / 2;
        for yy in 0..16 {
            for xx in 0..16 {
                let p = y[(by + yy) * self.w + bx + xx];
                w.u(8, p as u32);
                recon_y[(by + yy) * self.w + bx + xx] = p;
            }
        }
        let (cbx, cby) = (bx / 2, by / 2);
        for yy in 0..8 {
            for xx in 0..8 {
                let p = u[(cby + yy) * cw + cbx + xx];
                w.u(8, p as u32);
                recon_u[(cby + yy) * cw + cbx + xx] = p;
            }
        }
        for yy in 0..8 {
            for xx in 0..8 {
                let p = v[(cby + yy) * cw + cbx + xx];
                w.u(8, p as u32);
                recon_v[(cby + yy) * cw + cbx + xx] = p;
            }
        }
        for j in 0..4 {
            for i in 0..4 {
                nnz_y[(mb_y * 4 + j) * (self.mb_w * 4) + mb_x * 4 + i] = 16;
            }
        }
    }

    #[allow(dead_code)] // residual path kept for a future compressed I16 rewrite
    fn encode_i16_mb(
        &self,
        w: &mut BitWriter,
        y: &[u8],
        u: &[u8],
        v: &[u8],
        recon_y: &mut [u8],
        recon_u: &mut [u8],
        recon_v: &mut [u8],
        nnz_y: &mut [u8],
        mb_x: usize,
        mb_y: usize,
        is_i_slice: bool,
    ) -> Result<(), &'static str> {
        let bx = mb_x * 16;
        let by = mb_y * 16;
        let avail_top = mb_y > 0;
        let avail_left = mb_x > 0;

        // Neighbours for prediction from *reconstructed* samples.
        let mut top = [128i32; 16];
        let mut left = [128i32; 16];
        let mut corner = 128i32;
        if avail_top {
            for i in 0..16 {
                top[i] = recon_y[(by - 1) * self.w + bx + i] as i32;
            }
        }
        if avail_left {
            for j in 0..16 {
                left[j] = recon_y[(by + j) * self.w + bx - 1] as i32;
            }
        }
        if avail_top && avail_left {
            corner = recon_y[(by - 1) * self.w + bx - 1] as i32;
        }
        let mode = 2u8; // DC
        let pred = intra16x16(mode, &top, &left, corner, avail_top, avail_left);

        // Original 16×16.
        let mut org = [0i32; 256];
        for j in 0..16 {
            for i in 0..16 {
                org[j * 16 + i] = y[(by + j) * self.w + bx + i] as i32;
            }
        }

        // Residual per 4×4 in BLK_XY order; DC stored in raster (x + y*4).
        // Order is load-bearing (JM / H.264 §8.5): **extract unquantized DC,
        // then quantize AC, then Hadamard+quant the DC array.** Quantizing DC
        // before the Hadamard was zeroing almost everything and every recording
        // became mid-gray prediction with a ~10-byte AU.
        let mut dc = [0i32; 16];
        let mut ac_blocks = [[0i32; 16]; 16];
        for b in 0..16 {
            let (bx4, by4) = BLK_XY[b];
            let (ox, oy) = (bx4 * 4, by4 * 4);
            let mut blk = [0i32; 16];
            for j in 0..4 {
                for i in 0..4 {
                    let idx = (oy + j) * 16 + (ox + i);
                    blk[j * 4 + i] = org[idx] - pred[idx];
                }
            }
            fdt_4x4(&mut blk);
            dc[by4 * 4 + bx4] = blk[0];
            blk[0] = 0;
            quant_4x4(&mut blk, self.qp, true);
            ac_blocks[b] = blk;
        }
        #[cfg(test)]
        let dc_pre = dc;
        quant_luma_dc(&mut dc, self.qp);
        #[cfg(test)]
        {
            // Catch silent no-op residual: a bright MB must keep some DC.
            let bright = org.iter().any(|&o| o > 200);
            if bright && dc.iter().all(|&c| c == 0) && ac_blocks.iter().all(|b| b.iter().all(|&c| c == 0)) {
                panic!(
                    "bright MB residual fully zeroed (org0={} pred0={} dc_pre0={:?} qp={})",
                    org[0], pred[0], dc_pre[0], self.qp
                );
            }
        }

        // Chroma 8×8 (both planes).
        let cw = self.w / 2;
        let (cbx, cby) = (bx / 2, by / 2);
        let mut top_c = [128i32; 8];
        let mut left_c = [128i32; 8];
        let mut corner_c = 128i32;
        if avail_top {
            for i in 0..8 {
                top_c[i] = recon_u[(cby - 1) * cw + cbx + i] as i32;
            }
        }
        if avail_left {
            for j in 0..8 {
                left_c[j] = recon_u[(cby + j) * cw + cbx - 1] as i32;
            }
        }
        if avail_top && avail_left {
            corner_c = recon_u[(cby - 1) * cw + cbx - 1] as i32;
        }
        // Chroma mode 0 = DC in chroma intra table.
        let pred_u = intra_chroma(0, &top_c, &left_c, corner_c, avail_top, avail_left);
        // V neighbours
        if avail_top {
            for i in 0..8 {
                top_c[i] = recon_v[(cby - 1) * cw + cbx + i] as i32;
            }
        }
        if avail_left {
            for j in 0..8 {
                left_c[j] = recon_v[(cby + j) * cw + cbx - 1] as i32;
            }
        }
        if avail_top && avail_left {
            corner_c = recon_v[(cby - 1) * cw + cbx - 1] as i32;
        }
        let pred_v = intra_chroma(0, &top_c, &left_c, corner_c, avail_top, avail_left);

        let mut chroma = |plane: &[u8], pred_c: &[i32; 64]| {
            let mut dc_c = [0i32; 4];
            let mut ac_c = [[0i32; 16]; 4];
            for b in 0..4 {
                let (ox, oy) = ((b % 2) * 4, (b / 2) * 4);
                let mut blk = [0i32; 16];
                for j in 0..4 {
                    for i in 0..4 {
                        let sx = cbx + ox + i;
                        let sy = cby + oy + j;
                        let org = plane[sy * cw + sx] as i32;
                        let pr = pred_c[(oy + j) * 8 + (ox + i)];
                        blk[j * 4 + i] = org - pr;
                    }
                }
                fdt_4x4(&mut blk);
                dc_c[b] = blk[0];
                blk[0] = 0;
                quant_4x4(&mut blk, self.qp, true);
                ac_c[b] = blk;
            }
            quant_chroma_dc(&mut dc_c, self.qp);
            (dc_c, ac_c)
        };
        let (dc_u, ac_u) = chroma(u, &pred_u);
        let (dc_v, ac_v) = chroma(v, &pred_v);

        // Coded block pattern.
        let mut cbp_luma = 0u32;
        for b in 0..16 {
            if ac_blocks[b].iter().any(|&c| c != 0) {
                cbp_luma = 15;
                break;
            }
        }
        // Also treat non-zero luma DC as needing the I16 path's residual; DC is
        // always coded for I16, cbp_luma only gates AC.
        let chroma_dc_nz = dc_u.iter().any(|&c| c != 0) || dc_v.iter().any(|&c| c != 0);
        let chroma_ac_nz = ac_u.iter().chain(ac_v.iter()).any(|b| b.iter().any(|&c| c != 0));
        let cbp_chroma = if chroma_ac_nz {
            2
        } else if chroma_dc_nz {
            1
        } else {
            0
        };

        // mb_type: I_16x16_mode_cbpc_cbpl (Table 7-11); +5 in a P slice.
        let i16_type =
            1 + mode as u32 + 4 * cbp_chroma + 12 * (if cbp_luma != 0 { 1 } else { 0 });
        let mb_type = if is_i_slice { i16_type } else { i16_type + 5 };
        w.ue(mb_type);
        w.ue(0); // intra_chroma_pred_mode = DC
                 // I_16x16 always carries mb_qp_delta (decoder: is16 || cbp…).
        w.se(0);

        // Luma DC (always), then AC in BLK_XY order when cbp_luma != 0.
        let dc_scan = zigzag_scan(&dc);
        let n_c_dc = nc_from(nnz_y, self.mb_w, mb_x * 4, mb_y * 4);
        write_residual(w, &dc_scan, 16, n_c_dc);
        if cbp_luma != 0 {
            for b in 0..16 {
                let (bx4, by4) = BLK_XY[b];
                let gx = mb_x * 4 + bx4;
                let gy = mb_y * 4 + by4;
                let n_c = nc_from(nnz_y, self.mb_w, gx, gy);
                let full = zigzag_scan(&ac_blocks[b]);
                let mut ac15 = [0i32; 16];
                for i in 0..15 {
                    ac15[i] = full[i + 1];
                }
                write_residual(w, &ac15, 15, n_c);
                let nnz = ac15.iter().filter(|&&c| c != 0).count() as u8;
                nnz_y[gy * (self.mb_w * 4) + gx] = nnz;
            }
        } else {
            for j in 0..4 {
                for i in 0..4 {
                    nnz_y[(mb_y * 4 + j) * (self.mb_w * 4) + mb_x * 4 + i] = 0;
                }
            }
        }

        if cbp_chroma != 0 {
            let mut su = [0i32; 16];
            let mut sv = [0i32; 16];
            for i in 0..4 {
                su[i] = dc_u[i];
                sv[i] = dc_v[i];
            }
            write_residual(w, &su, 4, -1);
            write_residual(w, &sv, 4, -1);
        }
        if cbp_chroma == 2 {
            for b in 0..4 {
                let full = zigzag_scan(&ac_u[b]);
                let mut ac15 = [0i32; 16];
                for i in 0..15 {
                    ac15[i] = full[i + 1];
                }
                write_residual(w, &ac15, 15, 0);
            }
            for b in 0..4 {
                let full = zigzag_scan(&ac_v[b]);
                let mut ac15 = [0i32; 16];
                for i in 0..15 {
                    ac15[i] = full[i + 1];
                }
                write_residual(w, &ac15, 15, 0);
            }
        }

        // Reconstruct for the next frame's predictors (same inverse path as the
        // decoder — drift here becomes visible as prediction error).
        let mut dc_inv = dc;
        super::transform::luma_dc_transform(&mut dc_inv, self.qp);
        for b in 0..16 {
            let (bx4, by4) = BLK_XY[b];
            let (ox, oy) = (bx4 * 4, by4 * 4);
            let mut blk = ac_blocks[b];
            blk[0] = dc_inv[by4 * 4 + bx4];
            dequant_4x4(&mut blk, self.qp, true);
            idct_4x4(&mut blk);
            for j in 0..4 {
                for i in 0..4 {
                    let idx = (oy + j) * 16 + (ox + i);
                    let val = (pred[idx] + blk[j * 4 + i]).clamp(0, 255) as u8;
                    recon_y[(by + oy + j) * self.w + bx + ox + i] = val;
                }
            }
        }
        let mut dcu = dc_u;
        let mut dcv = dc_v;
        super::transform::chroma_dc_transform(&mut dcu, self.qp);
        super::transform::chroma_dc_transform(&mut dcv, self.qp);
        for (plane_ac, dc_s, pred_c, recon) in [
            (&ac_u, dcu, pred_u, &mut *recon_u),
            (&ac_v, dcv, pred_v, &mut *recon_v),
        ] {
            for b in 0..4 {
                let (ox, oy) = ((b % 2) * 4, (b / 2) * 4);
                let mut blk = plane_ac[b];
                blk[0] = dc_s[b];
                dequant_4x4(&mut blk, self.qp, true);
                idct_4x4(&mut blk);
                for j in 0..4 {
                    for i in 0..4 {
                        let pr = pred_c[(oy + j) * 8 + (ox + i)];
                        let val = (pr + blk[j * 4 + i]).clamp(0, 255) as u8;
                        recon[(cby + oy + j) * cw + cbx + ox + i] = val;
                    }
                }
            }
        }
        Ok(())
    }
}

fn write_sps(mb_w: u32, mb_h: u32) -> Vec<u8> {
    let mut w = BitWriter::with_capacity(64);
    w.u(8, 66); // profile_idc Baseline
    w.u(8, 0xC0); // constraint_set0+1 (baseline+main constrained)
    w.u(8, 31); // level_idc 3.1
    w.ue(0); // sps_id
    w.ue(0); // log2_max_frame_num_minus4 → 4 bits frame_num
    w.ue(0); // pic_order_cnt_type
    w.ue(0); // log2_max_pic_order_cnt_lsb_minus4 → 4 bits
    w.ue(1); // max_num_ref_frames
    w.flag(false); // gaps_in_frame_num_value_allowed
    w.ue(mb_w - 1);
    w.ue(mb_h - 1);
    w.flag(true); // frame_mbs_only
    w.flag(true); // direct_8x8_inference
    w.flag(false); // frame_cropping
    w.flag(false); // vui
    w.finish()
}

fn write_pps(qp: u32) -> Vec<u8> {
    let mut w = BitWriter::with_capacity(32);
    w.ue(0); // pps_id
    w.ue(0); // sps_id
    w.flag(false); // entropy_coding_mode = CAVLC
    w.flag(false); // bottom_field_pic_order_present
    w.ue(0); // num_slice_groups_minus1
    w.ue(0); // num_ref_idx_l0_default_active_minus1 → 1 ref
    w.ue(0); // num_ref_idx_l1_default_active_minus1
    w.flag(false); // weighted_pred
    w.u(2, 0); // weighted_bipred_idc
    w.se(qp as i32 - 26); // pic_init_qp_minus26
    w.se(0); // pic_init_qs_minus26
    w.se(0); // chroma_qp_index_offset
    w.flag(false); // deblocking_filter_control_present
    w.flag(false); // constrained_intra_pred
    w.flag(false); // redundant_pic_cnt_present
    w.finish()
}

/// Parsed SPS/PPS from the encoder's own NAL bytes (for tests / avcC).
pub fn encoder_param_sets(enc: &Encoder) -> Result<(Sps, Pps, Vec<u8>, Vec<u8>), &'static str> {
    use super::{parse_pps, parse_sps};
    use crate::video::bits::unescape_rbsp;
    let sps_rbsp = unescape_rbsp(&enc.sps_nal[1..]);
    let pps_rbsp = unescape_rbsp(&enc.pps_nal[1..]);
    let sps = parse_sps(&sps_rbsp)?;
    let pps = parse_pps(&pps_rbsp)?;
    Ok((sps, pps, enc.sps_nal.clone(), enc.pps_nal.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video::h264::decoder::decode_access_unit;
    use crate::video::bits::unescape_rbsp;
    use crate::video::h264::{parse_pps, parse_sps, split_avcc};

    #[test_case]
    fn a_cavlc_write_residual_roundtrips_a_nonzero_dc() {
        let mut w = BitWriter::new();
        let mut scan = [0i32; 16];
        scan[0] = 5;
        write_residual(&mut w, &scan, 16, 0);
        let bytes = w.into_bytes_aligned();
        assert!(!bytes.is_empty(), "wrote nothing for DC=5");
        let mut r = crate::video::bits::BitReader::new(&bytes);
        let (coeffs, tc) = super::super::cavlc::residual_block(&mut r, 16, 0).unwrap();
        assert_eq!(tc, 1, "total_coeff");
        assert_eq!(coeffs[0], 5, "DC level round-trip, got {coeffs:?}");
    }

    #[test_case]
    fn a_aaa_white_rgb_converts_to_bright_luma() {
        let px0 = 0x00ff_ffffu32;
        let r = ((px0 >> 16) & 255) as i32;
        let g = ((px0 >> 8) & 255) as i32;
        let b = (px0 & 255) as i32;
        assert_eq!((r, g, b), (255, 255, 255), "pixel layout");
        let (y, u, v) = rgb32_to_yuv420(16, 16, &[px0; 16 * 16]);
        let mean = y.iter().map(|&p| p as u32).sum::<u32>() / y.len() as u32;
        assert!(mean > 200, "white mean Y={mean}, y0={} u0={} v0={}", y[0], u[0], v[0]);
    }

    #[test_case]
    fn a_i16_residual_path_keeps_dc_for_white_minus_gray() {
        // Replicate one MB residual path outside the bitstream writer.
        let pred = 128i32;
        let org = 235i32; // approx white luma
        let mut dc = [0i32; 16];
        for b in 0..16 {
            let mut blk = [org - pred; 16];
            fdt_4x4(&mut blk);
            quant_4x4(&mut blk, 28, true);
            dc[b] = blk[0];
        }
        assert!(
            dc.iter().any(|&c| c != 0),
            "4x4 DC quant zeroed white-gray residual: {dc:?}"
        );
        quant_luma_dc(&mut dc, 28);
        assert!(
            dc.iter().any(|&c| c != 0),
            "luma DC quant zeroed: {dc:?}"
        );
    }

    #[test_case]
    fn a_white_macroblock_is_not_a_tiny_empty_au() {
        let mut enc = Encoder::new(16, 16, 28).unwrap();
        let px = vec![0x00ff_ffffu32; 16 * 16];
        let au = enc.encode_rgb32(&px, true).unwrap();
        let nals = split_avcc(&au, 4);
        assert_eq!(nals.len(), 1, "au_len={} first={:02x?}", au.len(), &au[..au.len().min(8)]);
        let sps = parse_sps(&unescape_rbsp(&enc.sps_nal[1..])).unwrap();
        let pps = parse_pps(&unescape_rbsp(&enc.pps_nal[1..])).unwrap();
        let df = decode_access_unit(&sps, &pps, &[(nals[0].rbsp(), true)], None)
            .unwrap_or_else(|e| panic!("decode failed: {e} (au_len={})", au.len()));
        let mean_y = df.y.iter().map(|&p| p as u32).sum::<u32>() / df.y.len() as u32;
        assert!(
            mean_y > 180,
            "white frame mean Y={mean_y} (want bright); au_len={} y0={}",
            au.len(),
            df.y[0]
        );
    }

    #[test_case]
    fn forward_quant_does_not_zero_a_large_residual() {
        // A 4×4 of residual 100 must survive QP 28 — if it quantises to zero the
        // whole encoder is a no-op and every frame is mid-gray prediction.
        let mut blk = [100i32; 16];
        fdt_4x4(&mut blk);
        assert!(blk[0].abs() > 100, "fdt DC should grow, got {}", blk[0]);
        quant_4x4(&mut blk, 28, true);
        assert!(
            blk.iter().any(|&c| c != 0),
            "quant_4x4 zeroed a large residual: {blk:?}"
        );
    }

    #[test_case]
    fn rgb_to_yuv_black_and_white() {
        let black = [0u32; 16 * 16];
        let (y, u, v) = rgb32_to_yuv420(16, 16, &black);
        assert!(y.iter().all(|&p| p <= 20), "black → low Y");
        let white = [0x00ff_ffffu32; 16 * 16];
        let (y2, _, _) = rgb32_to_yuv420(16, 16, &white);
        assert!(y2.iter().all(|&p| p >= 230), "white → high Y");
        let _ = (u, v);
    }

    #[test_case]
    fn encodes_a_flat_frame_that_our_decoder_accepts() {
        let mut enc = Encoder::new(32, 32, 28).unwrap();
        // Solid terracotta-ish colour (the brand accent).
        let px = vec![0x00cc_785c_u32; 32 * 32];
        let au = enc.encode_rgb32(&px, true).expect("encode");
        assert!(au.len() > 8, "AU too short: {}", au.len());

        let sps_rbsp = unescape_rbsp(&enc.sps_nal[1..]);
        let pps_rbsp = unescape_rbsp(&enc.pps_nal[1..]);
        let sps = parse_sps(&sps_rbsp).unwrap();
        let pps = parse_pps(&pps_rbsp).unwrap();
        assert_eq!((sps.width(), sps.height()), (32, 32));

        let nals = split_avcc(&au, 4);
        assert_eq!(nals.len(), 1);
        let rbsp = nals[0].rbsp();
        let frame = decode_access_unit(&sps, &pps, &[(rbsp, true)], None);
        assert!(frame.is_ok(), "decode failed: {:?}", frame.err());
        let f = frame.unwrap();
        assert_eq!((f.w, f.h), (32, 32));
        // Flat colour: mean luma should sit mid-range after lossy encode.
        let mean = f.y.iter().map(|&p| p as u64).sum::<u64>() / f.y.len() as u64;
        assert!(mean > 40 && mean < 220, "unexpected mean Y {mean}");
    }

    #[test_case]
    fn second_identical_frame_is_mostly_skip() {
        let mut enc = Encoder::new(64, 64, 28).unwrap();
        // A structured pattern so the IDR is not tiny.
        let mut px = vec![0u32; 64 * 64];
        for y in 0..64 {
            for x in 0..64 {
                let v = ((x * 3 + y * 5) & 0xff) as u32;
                px[y * 64 + x] = (v << 16) | (v << 8) | v;
            }
        }
        let au0 = enc.encode_rgb32(&px, true).unwrap();
        let au1 = enc.encode_rgb32(&px, false).unwrap();
        assert!(au0.len() > 64, "IDR should carry residual data, got {}", au0.len());
        // A pure-skip P frame is far smaller than the IDR of a textured picture.
        assert!(
            au1.len() < au0.len() / 2,
            "P frame {} should be << IDR {}",
            au1.len(),
            au0.len()
        );
    }

    #[test_case]
    fn a_two_frame_clip_muxes_and_demuxes() {
        // End-to-end: encode → mp4_mux → our demuxer. (Lives here rather than
        // only in mp4_mux so a panic=abort suite that dies earlier still has
        // coverage once the encoder tests run.)
        let mut enc = Encoder::new(32, 32, 28).unwrap();
        let px = vec![0x00_30_60_90u32; 32 * 32];
        let mut samples = Vec::new();
        for i in 0..2 {
            let au = enc.encode_rgb32(&px, i == 0).unwrap();
            samples.push(crate::video::mp4_mux::Sample {
                bytes: au,
                duration: 200,
                sync: i == 0,
            });
        }
        let file = crate::video::mp4_mux::mux_avc(
            32,
            32,
            1000,
            &enc.sps_nal,
            &enc.pps_nal,
            &samples,
        )
        .expect("mux");
        let track = crate::video::mp4::parse(&file).expect("demux");
        assert_eq!((track.width, track.height), (32, 32));
        assert_eq!(track.samples.len(), 2);
        assert!(track.samples[0].is_sync);
    }

    /// The live player opens recordings through [`StreamDecoder`] (rust_h264
    /// first). A clip that demuxes but never paints is exactly the failure
    /// reported for `/record` → `/open`.
    #[test_case]
    fn recorded_mp4_paints_through_the_player_decoder() {
        // Real recordings land near 720×448 at 50% scale; a 64×48 residual
        // path can look fine while a full-size IDR does not. Use a size that
        // exercises multiple MB rows and the rust_h264 StreamDecoder path.
        let (w, h) = (80usize, 64usize);
        let mut enc = Encoder::new(w, h, 28).unwrap();
        // Distinct colour so a black/empty frame is an obvious fail.
        let mut px = vec![0u32; w * h];
        for y in 0..h {
            for x in 0..w {
                // Brand terracotta-ish, with a white bar so residual is non-flat.
                px[y * w + x] = if y < 8 {
                    0x00ff_ffff
                } else {
                    0x00cc_785c
                };
            }
        }
        let mut samples = Vec::new();
        for i in 0..3 {
            // Mix IDR + P (skip) like a real recording.
            let au = enc.encode_rgb32(&px, i == 0).unwrap();
            samples.push(crate::video::mp4_mux::Sample {
                bytes: au,
                duration: 200,
                sync: i == 0,
            });
        }
        // Sanity: the first sample alone must decode via the native path that
        // the unit tests already trust, *before* we blame the container.
        let raw0_len = samples[0].bytes.len();
        {
            let sps_rbsp = unescape_rbsp(&enc.sps_nal[1..]);
            let pps_rbsp = unescape_rbsp(&enc.pps_nal[1..]);
            let sps = parse_sps(&sps_rbsp).unwrap();
            let pps = parse_pps(&pps_rbsp).unwrap();
            let nals = split_avcc(&samples[0].bytes, 4);
            assert_eq!(
                nals.len(),
                1,
                "one NAL per sample (au_len={raw0_len}, first8={:02x?})",
                &samples[0].bytes[..samples[0].bytes.len().min(8)]
            );
            assert!(
                matches!(nals[0].kind, crate::video::h264::NalType::SliceIdr),
                "first sample must be IDR"
            );
            let df = decode_access_unit(&sps, &pps, &[(nals[0].rbsp(), true)], None);
            assert!(
                df.is_ok(),
                "native decode of raw sample0 failed: {:?} (au_len={raw0_len})",
                df.err()
            );
            let df = df.unwrap();
            // Residual must have done *something* — pure 128 gray means the
            // encoder wrote empty residuals and the player paints a blank.
            let mean_y = df.y.iter().map(|&p| p as u32).sum::<u32>() / df.y.len() as u32;
            assert!(
                mean_y > 140 || mean_y < 100,
                "mean Y={mean_y} looks like empty-residual mid-gray; AU was {raw0_len} bytes"
            );
        }
        let file = crate::video::mp4_mux::mux_avc(
            w as u32,
            h as u32,
            1000,
            &enc.sps_nal,
            &enc.pps_nal,
            &samples,
        )
        .expect("mux");
        // Demuxed sample0 must still decode natively (proves the container
        // did not corrupt the AU).
        {
            let track = crate::video::mp4::parse(&file).expect("demux");
            assert_eq!(track.samples.len(), 3);
            let s0 = &track.samples[0];
            assert!(
                s0.offset >= 16,
                "sample0 offset {} is inside ftyp — stco broken",
                s0.offset
            );
            assert_eq!(
                s0.size, raw0_len,
                "demux sample0 size {} != encoded AU {}; offset={}",
                s0.size,
                raw0_len,
                s0.offset
            );
            assert!(
                s0.offset + s0.size <= file.len(),
                "sample0 out of range: off={} size={} file={}",
                s0.offset,
                s0.size,
                file.len()
            );
            let au = &file[s0.offset..s0.offset + s0.size];
            assert_ne!(&au.get(4..8).unwrap_or(&[]), b"ftyp");
            let cfg = match &track.config {
                crate::video::mp4::CodecConfig::Avc(a) => a,
                _ => panic!("expected AVC"),
            };
            assert_eq!(cfg.length_size, 4);
            let sps = parse_sps(&unescape_rbsp(&cfg.sps[0][1..])).unwrap();
            let pps = parse_pps(&unescape_rbsp(&cfg.pps[0][1..])).unwrap();
            let nals = split_avcc(au, cfg.length_size);
            assert!(
                !nals.is_empty(),
                "demuxed sample0 has no NALs (len={}, first8={:02x?})",
                au.len(),
                &au[..au.len().min(8)]
            );
            let df = decode_access_unit(
                &sps,
                &pps,
                &[(nals[0].rbsp(), true)],
                None,
            );
            assert!(
                df.is_ok(),
                "native decode of *demuxed* sample0 failed: {:?} (au first 16 bytes {:02x?})",
                df.err(),
                &au[..au.len().min(16)]
            );
        }
        let mut dec = crate::video::StreamDecoder::open(file).expect("open StreamDecoder");
        assert!(
            dec.seek_decode(0),
            "seek_decode(0) must produce a picture (backend={})",
            dec.backend
        );
        let f = dec.cur_frame().expect("cur_frame after seek_decode");
        // StreamDecoder may downscale for display (DISPLAY_MAX_EDGE); src size is authoritative.
        assert_eq!((dec.src_w as usize, dec.src_h as usize), (w, h));
        // Not all black / not all zero — a real image.
        let nonzero = f.pixels.iter().filter(|&&p| p & 0x00ff_ffff != 0).count();
        assert!(
            nonzero > f.pixels.len() / 4,
            "frame is nearly black ({nonzero}/{} non-zero) — encoder bitstream is unplayable",
            f.pixels.len()
        );
        // White bar must survive (top row should be bright, not black/gray).
        let top = f.pixels[0];
        let top_yish = ((top >> 16) & 0xff).max((top >> 8) & 0xff).max(top & 0xff);
        assert!(
            top_yish > 180,
            "top pixel {:08x} is not bright — white bar lost (backend={})",
            top,
            dec.backend
        );
        // And the next display frame must also resolve (P after IDR).
        assert!(
            dec.seek_decode(1),
            "seek_decode(1) must produce a picture after a P frame"
        );
    }
}
