//! VP9's in-loop deblocking filter (libvpx `vp9_loopfilter.c` +
//! `vpx_dsp/loopfilter.c`).
//!
//! It runs **after** the whole frame is reconstructed and modifies the frame
//! that later frames predict from, so it is not cosmetic: skipping it leaves a
//! picture that is right to within ±2 and a *reference* that drifts.
//!
//! Two things shape the code here:
//!
//! * **Which edges are filtered, and how wide, comes from the transform size**
//!   of the blocks on either side — 16-wide at a 32x32 transform boundary,
//!   8-wide at 8x8, 4-wide inside a block with 4x4 transforms. libvpx builds
//!   these as per-superblock bitmasks and so does this, because the decision
//!   for a column depends on its position within the superblock (`(c & 3) == 0`
//!   forces the wider filter at 32-pixel boundaries).
//! * **A skipped block's *interior* edges are not filtered, but its block edge
//!   still is.** That distinction (`skip_this_c` vs `block_edge_left`) is what
//!   keeps flat skipped regions from being smeared while still deblocking the
//!   seam against their neighbour.
//!
//! Vertical and horizontal passes share one kernel by walking with a `step`
//! (1 across a row, `stride` down a column) — the filters are identical, only
//! the direction of the 8-sample window differs.

use super::tables;
use super::tile::{FrameDecodeState, ModeInfo};
use super::transform::TxSize;

/// Largest filter level (`MAX_LOOP_FILTER`).
pub const MAX_LOOP_FILTER: i32 = 63;

/// The three thresholds a filter level implies.
#[derive(Clone, Copy, Default)]
pub struct Thresh {
    /// `mblim` — the block-edge limit.
    pub mblim: u8,
    /// `lim` — the inside limit.
    pub lim: u8,
    /// `hev_thr` — the high-edge-variance threshold.
    pub hev_thr: u8,
}

/// libvpx `update_sharpness`: derive the limits for every level once per frame.
///
/// The sharpness shift is applied to the *inside* limit only, and the result is
/// floored at 1 — a limit of 0 would disable the filter entirely at low levels
/// rather than merely narrowing it.
pub fn build_thresholds(sharpness: u32) -> [Thresh; 64] {
    let mut out = [Thresh::default(); 64];
    for lvl in 0..64usize {
        let mut inside = (lvl as i32) >> ((sharpness > 0) as i32 + (sharpness > 4) as i32);
        if sharpness > 0 && inside > (9 - sharpness as i32) {
            inside = 9 - sharpness as i32;
        }
        if inside < 1 {
            inside = 1;
        }
        out[lvl] = Thresh {
            lim: inside as u8,
            mblim: (2 * (lvl as i32 + 2) + inside) as u8,
            hev_thr: (lvl >> 4) as u8,
        };
    }
    out
}

#[inline(always)]
fn clamp_i8(v: i32) -> i32 {
    v.clamp(-128, 127)
}

#[inline(always)]
fn round_pow2(v: u32, n: u32) -> u8 {
    ((v + (1 << (n - 1))) >> n) as u8
}

/// A window of samples around an edge. `get(-1)` is `p0`, `get(0)` is `q0`;
/// `step` is 1 for a vertical edge (samples run along the row) and the plane
/// stride for a horizontal one.
struct Window<'a> {
    data: &'a mut [u8],
    base: isize,
    step: isize,
}

impl<'a> Window<'a> {
    #[inline(always)]
    fn idx(&self, k: isize) -> usize {
        (self.base + k * self.step) as usize
    }
    #[inline(always)]
    fn get(&self, k: isize) -> u8 {
        self.data[self.idx(k)]
    }
    #[inline(always)]
    fn set(&mut self, k: isize, v: u8) {
        let i = self.idx(k);
        self.data[i] = v;
    }
    /// True when every sample the widest filter touches is inside the buffer.
    fn in_range(&self, lo: isize, hi: isize) -> bool {
        let a = self.base + lo * self.step;
        let b = self.base + hi * self.step;
        a >= 0 && b >= 0 && (a as usize) < self.data.len() && (b as usize) < self.data.len()
    }
}

fn filter_mask(limit: u8, blimit: u8, p: [u8; 4], q: [u8; 4]) -> bool {
    let d = |a: u8, b: u8| (a as i32 - b as i32).unsigned_abs();
    d(p[3], p[2]) <= limit as u32
        && d(p[2], p[1]) <= limit as u32
        && d(p[1], p[0]) <= limit as u32
        && d(q[1], q[0]) <= limit as u32
        && d(q[2], q[1]) <= limit as u32
        && d(q[3], q[2]) <= limit as u32
        && d(p[0], q[0]) * 2 + d(p[1], q[1]) / 2 <= blimit as u32
}

fn flat_mask4(thresh: u8, p: [u8; 4], q: [u8; 4]) -> bool {
    let d = |a: u8, b: u8| (a as i32 - b as i32).unsigned_abs();
    d(p[1], p[0]) <= thresh as u32
        && d(q[1], q[0]) <= thresh as u32
        && d(p[2], p[0]) <= thresh as u32
        && d(q[2], q[0]) <= thresh as u32
        && d(p[3], p[0]) <= thresh as u32
        && d(q[3], q[0]) <= thresh as u32
}

fn hev_mask(thresh: u8, p1: u8, p0: u8, q0: u8, q1: u8) -> bool {
    let d = |a: u8, b: u8| (a as i32 - b as i32).unsigned_abs();
    d(p1, p0) > thresh as u32 || d(q1, q0) > thresh as u32
}

/// libvpx `filter4` — the narrow filter, in signed-offset arithmetic.
fn filter4(w: &mut Window, mask: bool, thresh: u8) {
    if !mask {
        return;
    }
    let ps1 = (w.get(-2) ^ 0x80) as i8 as i32;
    let ps0 = (w.get(-1) ^ 0x80) as i8 as i32;
    let qs0 = (w.get(0) ^ 0x80) as i8 as i32;
    let qs1 = (w.get(1) ^ 0x80) as i8 as i32;
    let hev = hev_mask(thresh, w.get(-2), w.get(-1), w.get(0), w.get(1));

    let mut filter = if hev { clamp_i8(ps1 - qs1) } else { 0 };
    filter = clamp_i8(filter + 3 * (qs0 - ps0));
    let filter1 = clamp_i8(filter + 4) >> 3;
    let filter2 = clamp_i8(filter + 3) >> 3;

    w.set(0, (clamp_i8(qs0 - filter1) as i8 as u8) ^ 0x80);
    w.set(-1, (clamp_i8(ps0 + filter2) as i8 as u8) ^ 0x80);

    // The outer taps move by half of filter1, and only where the edge is *not*
    // high-variance — a real edge keeps its outer samples.
    let f = if hev { 0 } else { (filter1 + 1) >> 1 };
    w.set(1, (clamp_i8(qs1 - f) as i8 as u8) ^ 0x80);
    w.set(-2, (clamp_i8(ps1 + f) as i8 as u8) ^ 0x80);
}

/// libvpx `filter8` — the 7-tap smoothing filter, falling back to [`filter4`]
/// wherever the neighbourhood is not flat.
fn filter8(w: &mut Window, mask: bool, thresh: u8, flat: bool) {
    if flat && mask {
        let (p3, p2, p1, p0) = (w.get(-4) as u32, w.get(-3) as u32, w.get(-2) as u32, w.get(-1) as u32);
        let (q0, q1, q2, q3) = (w.get(0) as u32, w.get(1) as u32, w.get(2) as u32, w.get(3) as u32);
        w.set(-3, round_pow2(p3 + p3 + p3 + 2 * p2 + p1 + p0 + q0, 3));
        w.set(-2, round_pow2(p3 + p3 + p2 + 2 * p1 + p0 + q0 + q1, 3));
        w.set(-1, round_pow2(p3 + p2 + p1 + 2 * p0 + q0 + q1 + q2, 3));
        w.set(0, round_pow2(p2 + p1 + p0 + 2 * q0 + q1 + q2 + q3, 3));
        w.set(1, round_pow2(p1 + p0 + q0 + 2 * q1 + q2 + q3 + q3, 3));
        w.set(2, round_pow2(p0 + q0 + q1 + 2 * q2 + q3 + q3 + q3, 3));
    } else {
        filter4(w, mask, thresh);
    }
}

/// libvpx `filter16` — the 15-tap filter used at 32x32 transform boundaries.
fn filter16(w: &mut Window, mask: bool, thresh: u8, flat: bool, flat2: bool) {
    if flat2 && flat && mask {
        let s: [u32; 16] = core::array::from_fn(|i| w.get(i as isize - 8) as u32);
        let (p, q) = (&s[..8], &s[8..]);
        // p[i] here is s[7-i] (p0 at index 7); q[i] is s[8+i].
        let pv = |i: usize| p[7 - i];
        let qv = |i: usize| q[i];
        // Written out rather than looped: each output has its own doubled
        // centre tap, so this is not a clean sliding sum, and a loop that is
        // nearly right is worse than fifteen explicit lines.
        let (p7, p6, p5, p4, p3, p2, p1, p0) =
            (pv(7), pv(6), pv(5), pv(4), pv(3), pv(2), pv(1), pv(0));
        let (q0, q1, q2, q3, q4, q5, q6, q7) =
            (qv(0), qv(1), qv(2), qv(3), qv(4), qv(5), qv(6), qv(7));
        w.set(-7, round_pow2(p7 * 7 + p6 * 2 + p5 + p4 + p3 + p2 + p1 + p0 + q0, 4));
        w.set(-6, round_pow2(p7 * 6 + p6 + p5 * 2 + p4 + p3 + p2 + p1 + p0 + q0 + q1, 4));
        w.set(-5, round_pow2(p7 * 5 + p6 + p5 + p4 * 2 + p3 + p2 + p1 + p0 + q0 + q1 + q2, 4));
        w.set(-4, round_pow2(p7 * 4 + p6 + p5 + p4 + p3 * 2 + p2 + p1 + p0 + q0 + q1 + q2 + q3, 4));
        w.set(-3, round_pow2(p7 * 3 + p6 + p5 + p4 + p3 + p2 * 2 + p1 + p0 + q0 + q1 + q2 + q3 + q4, 4));
        w.set(-2, round_pow2(p7 * 2 + p6 + p5 + p4 + p3 + p2 + p1 * 2 + p0 + q0 + q1 + q2 + q3 + q4 + q5, 4));
        w.set(-1, round_pow2(p7 + p6 + p5 + p4 + p3 + p2 + p1 + p0 * 2 + q0 + q1 + q2 + q3 + q4 + q5 + q6, 4));
        w.set(0, round_pow2(p6 + p5 + p4 + p3 + p2 + p1 + p0 + q0 * 2 + q1 + q2 + q3 + q4 + q5 + q6 + q7, 4));
        w.set(1, round_pow2(p5 + p4 + p3 + p2 + p1 + p0 + q0 + q1 * 2 + q2 + q3 + q4 + q5 + q6 + q7 * 2, 4));
        w.set(2, round_pow2(p4 + p3 + p2 + p1 + p0 + q0 + q1 + q2 * 2 + q3 + q4 + q5 + q6 + q7 * 3, 4));
        w.set(3, round_pow2(p3 + p2 + p1 + p0 + q0 + q1 + q2 + q3 * 2 + q4 + q5 + q6 + q7 * 4, 4));
        w.set(4, round_pow2(p2 + p1 + p0 + q0 + q1 + q2 + q3 + q4 * 2 + q5 + q6 + q7 * 5, 4));
        w.set(5, round_pow2(p1 + p0 + q0 + q1 + q2 + q3 + q4 + q5 * 2 + q6 + q7 * 6, 4));
        w.set(6, round_pow2(p0 + q0 + q1 + q2 + q3 + q4 + q5 + q6 * 2 + q7 * 7, 4));
    } else {
        filter8(w, mask, thresh, flat);
    }
}

/// Which filter width to apply at one edge.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Width {
    W4,
    W8,
    W16,
}

/// Filter `count` lines of one edge. `step` walks across the edge, `advance`
/// walks along it.
#[allow(clippy::too_many_arguments)]
fn filter_edge(
    data: &mut [u8],
    mut base: isize,
    step: isize,
    advance: isize,
    count: usize,
    width: Width,
    t: &Thresh,
) {
    for _ in 0..count {
        let need = if width == Width::W16 { (-8, 7) } else { (-4, 3) };
        let mut w = Window { data, base, step };
        if !w.in_range(need.0, need.1) {
            base += advance;
            continue;
        }
        let p: [u8; 4] = [w.get(-1), w.get(-2), w.get(-3), w.get(-4)];
        let q: [u8; 4] = [w.get(0), w.get(1), w.get(2), w.get(3)];
        let mask = filter_mask(t.lim, t.mblim, p, q);
        match width {
            Width::W4 => filter4(&mut w, mask, t.hev_thr),
            Width::W8 => {
                let flat = flat_mask4(1, p, q);
                filter8(&mut w, mask, t.hev_thr, flat);
            }
            Width::W16 => {
                let flat = flat_mask4(1, p, q);
                // `flat_mask5` extends the flatness test out to p4/q4.
                let p4 = [w.get(-1), w.get(-5), w.get(-6), w.get(-7)];
                let q4 = [w.get(0), w.get(4), w.get(5), w.get(6)];
                let flat2 = flat_mask4(1, p4, q4) && {
                    let d = |a: u8, b: u8| (a as i32 - b as i32).unsigned_abs();
                    d(w.get(-8), w.get(-1)) <= 1 && d(w.get(7), w.get(0)) <= 1
                };
                filter16(&mut w, mask, t.hev_thr, flat, flat2);
            }
        }
        base += advance;
    }
}

/// Per-superblock edge masks for one plane, in plane 8x8 columns.
#[derive(Default, Clone, Copy)]
struct Masks {
    m16: [u32; 8],
    m8: [u32; 8],
    m4: [u32; 8],
    m4_int: [u32; 8],
    lvl: [[u8; 8]; 8],
}

/// Deblock one 64x64 superblock of one plane (libvpx
/// `vp9_filter_block_plane_non420`).
fn filter_sb_plane(s: &mut FrameDecodeState, plane: usize, mi_row: usize, mi_col: usize, thr: &[Thresh; 64], level_of: &dyn Fn(&ModeInfo) -> u8) {
    let (ss_x, ss_y) = if plane == 0 {
        (0usize, 0usize)
    } else {
        (s.subsampling_x as usize, s.subsampling_y as usize)
    };
    let row_step = 1usize << ss_y;
    let col_step = 1usize << ss_x;
    let mut m = Masks::default();

    let mut r = 0usize;
    while r < 8 && mi_row + r < s.mi_rows {
        let mut c = 0usize;
        while c < 8 && mi_col + c < s.mi_cols {
            let mi = s.mi[(mi_row + r) * s.mi_cols + (mi_col + c)];
            let bs = mi.block_size as usize;
            let skip_this = mi.skip && mi.is_inter;
            // A block's own left/top edge is always filtered even when the
            // block is skipped; only its *interior* edges are suppressed.
            let block_edge_left = if tables::NUM_4X4_W[bs] > 1 {
                c & (tables::NUM_8X8_W[bs] as usize - 1) == 0
            } else {
                true
            };
            let block_edge_above = if tables::NUM_4X4_H[bs] > 1 {
                r & (tables::NUM_8X8_H[bs] as usize - 1) == 0
            } else {
                true
            };
            let skip_c = skip_this && !block_edge_left;
            let skip_r = skip_this && !block_edge_above;
            let tx = if plane == 0 {
                mi.tx_size
            } else {
                tables::UV_TXSIZE[bs][mi.tx_size as usize][ss_x][ss_y]
            };
            let skip_border_c = ss_x == 1 && mi_col + c == s.mi_cols - 1;
            let skip_border_r = ss_y == 1 && mi_row + r == s.mi_rows - 1;
            let cc = c >> ss_x;
            let lvl = level_of(&mi);
            m.lvl[r][cc] = lvl;
            if lvl == 0 {
                c += col_step;
                continue;
            }
            let bit = 1u32 << cc;
            if tx == TxSize::Tx32x32 as u8 {
                if !skip_c && (cc & 3) == 0 {
                    if !skip_border_c {
                        m.m16[r] |= bit << 16;
                    } else {
                        m.m8[r] |= bit << 16;
                    }
                }
                if !skip_r && ((r >> ss_y) & 3) == 0 {
                    if !skip_border_r {
                        m.m16[r] |= bit;
                    } else {
                        m.m8[r] |= bit;
                    }
                }
            } else if tx == TxSize::Tx16x16 as u8 {
                if !skip_c && (cc & 1) == 0 {
                    if !skip_border_c {
                        m.m16[r] |= bit << 16;
                    } else {
                        m.m8[r] |= bit << 16;
                    }
                }
                if !skip_r && ((r >> ss_y) & 1) == 0 {
                    if !skip_border_r {
                        m.m16[r] |= bit;
                    } else {
                        m.m8[r] |= bit;
                    }
                }
            } else {
                // 8x8 filtering is forced at 32-pixel boundaries even for 4x4
                // transforms.
                if !skip_c {
                    if tx == TxSize::Tx8x8 as u8 || (cc & 3) == 0 {
                        m.m8[r] |= bit << 16;
                    } else {
                        m.m4[r] |= bit << 16;
                    }
                }
                if !skip_r {
                    if tx == TxSize::Tx8x8 as u8 || ((r >> ss_y) & 3) == 0 {
                        m.m8[r] |= bit;
                    } else {
                        m.m4[r] |= bit;
                    }
                }
                if !skip_this && tx < TxSize::Tx8x8 as u8 && !skip_border_c {
                    m.m4_int[r] |= bit;
                }
            }
            c += col_step;
        }
        r += row_step;
    }

    let stride = s.planes[plane].stride as isize;
    let x0 = ((mi_col * 8) >> ss_x) as isize;
    let y0 = ((mi_row * 8) >> ss_y) as isize;

    // Vertical edges. The frame's leftmost column is never filtered.
    let mut r = 0usize;
    let mut row_px = 0isize;
    while r < 8 && mi_row + r < s.mi_rows {
        for cc in 0..8usize {
            let bit = 1u32 << cc;
            let t = thr[m.lvl[r][cc] as usize % 64];
            let base = (y0 + row_px) * stride + x0 + (cc * 8) as isize;
            if !(mi_col == 0 && cc == 0) {
                let width = if m.m16[r] & (bit << 16) != 0 {
                    Some(Width::W16)
                } else if m.m8[r] & (bit << 16) != 0 {
                    Some(Width::W8)
                } else if m.m4[r] & (bit << 16) != 0 {
                    Some(Width::W4)
                } else {
                    None
                };
                if let Some(wd) = width {
                    filter_edge(&mut s.planes[plane].data, base, 1, stride, 8, wd, &t);
                }
            }
            if m.m4_int[r] & bit != 0 {
                filter_edge(&mut s.planes[plane].data, base + 4, 1, stride, 8, Width::W4, &t);
            }
        }
        row_px += 8;
        r += row_step;
    }

    // Horizontal edges. The frame's top row is never filtered.
    let mut r = 0usize;
    let mut row_px = 0isize;
    while r < 8 && mi_row + r < s.mi_rows {
        let skip_border_r = ss_y == 1 && mi_row + r == s.mi_rows - 1;
        for cc in 0..8usize {
            let bit = 1u32 << cc;
            let t = thr[m.lvl[r][cc] as usize % 64];
            let base = (y0 + row_px) * stride + x0 + (cc * 8) as isize;
            if mi_row + r != 0 {
                let width = if m.m16[r] & bit != 0 {
                    Some(Width::W16)
                } else if m.m8[r] & bit != 0 {
                    Some(Width::W8)
                } else if m.m4[r] & bit != 0 {
                    Some(Width::W4)
                } else {
                    None
                };
                if let Some(wd) = width {
                    filter_edge(&mut s.planes[plane].data, base, stride, 1, 8, wd, &t);
                }
            }
            if !skip_border_r && m.m4_int[r] & bit != 0 {
                filter_edge(&mut s.planes[plane].data, base + 4 * stride, stride, 1, 8, Width::W4, &t);
            }
        }
        row_px += 8;
        r += row_step;
    }
}

/// Deblock a whole reconstructed frame.
///
/// `level` is the frame's `loop_filter_level` **after** the reference-frame
/// delta is applied — an intra frame with deltas enabled runs one step higher
/// than the transmitted level, and that offset is the difference between a
/// bit-exact frame and one that is close.
pub fn loop_filter_frame(
    s: &mut FrameDecodeState,
    transmitted: u32,
    level: u8,
    sharpness: u32,
    // `ref_deltas[ref] * scale + mode_deltas[mode] * scale`, precomputed per
    // (reference frame, mode class). An **inter** block's filter level depends
    // on which reference it predicts from and whether it moves — using the
    // intra level everywhere leaves small errors on every inter block edge,
    // which is invisible on a keyframe and shows up from frame 1 onward.
    lvl_table: Option<&[[u8; 2]; 4]>,
) {
    // libvpx `vp9_loop_filter_frame` returns on `!frame_filter_level` — the
    // **transmitted** level, before any reference delta. A frame that codes
    // level 0 with deltas enabled would otherwise be filtered at level 1, which
    // is a small, plausible difference over the whole picture.
    if transmitted == 0 || level == 0 {
        return;
    }
    let thr = build_thresholds(sharpness);
    let level_of = move |mi: &ModeInfo| match lvl_table {
        None => level,
        Some(t) => {
            let rf = mi.ref_frame[0].max(0) as usize;
            let mode = tables::MODE_LF_LUT[(mi.y_mode as usize).min(13)] as usize;
            t[rf.min(3)][if rf == 0 { 0 } else { mode.min(1) }]
        }
    };
    let sb_rows = (s.mi_rows + 7) / 8;
    let sb_cols = (s.mi_cols + 7) / 8;
    for sbr in 0..sb_rows {
        for sbc in 0..sb_cols {
            for plane in 0..3 {
                filter_sb_plane(s, plane, sbr * 8, sbc * 8, &thr, &level_of);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn thresholds_match_the_reference_derivation() {
        // Sharpness 0: the inside limit is the level itself, floored at 1.
        let t = build_thresholds(0);
        assert_eq!(t[0].lim, 1, "floored at 1, never 0");
        assert_eq!(t[10].lim, 10);
        assert_eq!(t[10].mblim, (2 * (10 + 2) + 10) as u8);
        assert_eq!(t[10].hev_thr, 0);
        assert_eq!(t[63].hev_thr, 3);
        // Sharpness shifts the inside limit down and caps it at 9 - sharpness.
        let t5 = build_thresholds(5);
        assert_eq!(t5[63].lim, 4, "9 - 5");
        let t1 = build_thresholds(1);
        assert_eq!(t1[63].lim, 8, "9 - 1, after the >> 1");
    }

    #[test_case]
    fn a_flat_edge_is_left_alone_by_the_narrow_filter() {
        // Equal samples either side: the filter has nothing to correct, so it
        // must be the identity. (It is not obviously so — `filter4` still runs
        // its arithmetic.)
        let mut data = [128u8; 16];
        let t = build_thresholds(0)[20];
        filter_edge(&mut data, 8, 1, 0, 1, Width::W4, &t);
        assert!(data.iter().all(|&p| p == 128));
    }

    #[test_case]
    fn a_hard_edge_survives_the_filter() {
        // A step far larger than the limits fails `filter_mask`, so the edge is
        // preserved — that is the whole point of the mask.
        let mut data = [0u8; 16];
        for (i, v) in data.iter_mut().enumerate() {
            *v = if i < 8 { 20 } else { 200 };
        }
        let t = build_thresholds(0)[10];
        filter_edge(&mut data, 8, 1, 0, 1, Width::W4, &t);
        assert_eq!(data[7], 20, "p0 untouched across a hard edge");
        assert_eq!(data[8], 200, "q0 untouched across a hard edge");
    }

    #[test_case]
    fn a_small_step_is_smoothed() {
        let mut data = [0u8; 16];
        for (i, v) in data.iter_mut().enumerate() {
            *v = if i < 8 { 100 } else { 104 };
        }
        let t = build_thresholds(0)[20];
        filter_edge(&mut data, 8, 1, 0, 1, Width::W4, &t);
        assert!(data[7] > 100 && data[8] < 104, "the step was narrowed");
    }

    #[test_case]
    fn out_of_range_windows_are_skipped_not_wrapped() {
        // At a buffer edge the widest filter would read outside; it must skip
        // rather than index-panic or wrap onto the previous row.
        let mut data = [128u8; 8];
        let t = build_thresholds(0)[20];
        filter_edge(&mut data, 2, 1, 0, 1, Width::W16, &t);
        assert!(data.iter().all(|&p| p == 128));
    }
}
