//! VP9 intra prediction — the ten modes, ported from libvpx `vpx_dsp/intrapred.c`.
//!
//! **The edge buffer includes the top-left corner at index 0.** libvpx passes
//! `above` as a pointer *one past* the corner and reads `above[-1]` for it,
//! which has no direct Rust equivalent; here `above[0]` is the corner and
//! `above[1 + c]` is the pixel above column `c`. Every predictor below is
//! written against that shift, so a caller that forgets the corner slot gets
//! all ten modes off by one pixel — a picture that looks *almost* right, which
//! is the failure mode worth naming.
//!
//! The edges themselves come from the reconstructed frame, and their
//! availability rules are the caller's ([`Edges::build`]): unavailable pixels
//! are replaced by a defined value rather than by whatever the buffer held, or
//! prediction depends on uninitialised memory and stops being reproducible.

/// `AVG2` — round-half-up mean of two samples.
#[inline(always)]
fn avg2(a: u8, b: u8) -> u8 {
    ((a as u16 + b as u16 + 1) >> 1) as u8
}

/// `AVG3` — the [1 2 1] smoothing filter VP9's directional modes use.
#[inline(always)]
fn avg3(a: u8, b: u8, c: u8) -> u8 {
    ((a as u16 + 2 * b as u16 + c as u16 + 2) >> 2) as u8
}

/// The ten intra prediction modes (libvpx `PREDICTION_MODE`, intra part). The
/// numbering is the bitstream's, so it must not be reordered.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IntraMode {
    Dc = 0,
    V = 1,
    H = 2,
    D45 = 3,
    D135 = 4,
    D117 = 5,
    D153 = 6,
    D207 = 7,
    D63 = 8,
    Tm = 9,
}

impl IntraMode {
    pub fn from_index(i: u8) -> IntraMode {
        match i {
            1 => IntraMode::V,
            2 => IntraMode::H,
            3 => IntraMode::D45,
            4 => IntraMode::D135,
            5 => IntraMode::D117,
            6 => IntraMode::D153,
            7 => IntraMode::D207,
            8 => IntraMode::D63,
            9 => IntraMode::Tm,
            _ => IntraMode::Dc,
        }
    }
}

/// The reconstructed pixels a block predicts from.
///
/// `above[0]` is the **top-left corner**; `above[1..=2*bs]` are the row above
/// (the second half is the "above-right" extension the D45/D63 modes read past
/// the block into). `left[0..bs]` is the column to the left.
pub struct Edges {
    pub above: [u8; 65],
    pub left: [u8; 32],
}

impl Edges {
    /// Assemble the edges for a `bs`x`bs` transform block at `(x0, y0)` in a
    /// plane, following libvpx `build_intra_predictors` exactly.
    ///
    /// Four rules here are each a source of small, hard-to-attribute error:
    ///
    /// * **Availability is per *block*, not per pixel.** `have_above` is
    ///   "this transform block is not in the block's first row, or the block
    ///   has an above neighbour"; `have_left` likewise — and `left` is
    ///   **unavailable at a tile's left edge**, which is what makes tiles
    ///   independently decodable and what a single-tile decoder never notices.
    /// * **Above-right is only ever read for 4x4 transforms.** At every larger
    ///   size libvpx replicates the last above pixel instead, so reading real
    ///   pixels there is wrong even when they are available and decoded.
    /// * **Past the frame edge the last real pixel repeats.** Reading the
    ///   superblock-aligned padding instead gives whatever was there — zeros on
    ///   a fresh buffer, the previous frame later, so a seek and a linear play
    ///   decode differently.
    /// * **The corner is 129 when left is missing but above is present**, and
    ///   127 when above is missing. A single shared default is wrong on one of
    ///   the two frame edges.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        plane: &[u8],
        stride: usize,
        x0: usize,
        y0: usize,
        bs: usize,
        have_above: bool,
        have_left: bool,
        have_right: bool,
        frame_w: usize,
        frame_h: usize,
    ) -> Edges {
        let mut e = Edges { above: [127; 65], left: [129; 32] };

        if have_left {
            let col = x0 - 1;
            if y0 + bs <= frame_h {
                for i in 0..bs {
                    e.left[i] = plane[(y0 + i) * stride + col];
                }
            } else {
                // Bottom extension: repeat the last row inside the frame.
                let ext = frame_h.saturating_sub(y0).min(bs);
                for i in 0..ext {
                    e.left[i] = plane[(y0 + i) * stride + col];
                }
                let last = if ext > 0 { e.left[ext - 1] } else { 129 };
                for i in ext..bs {
                    e.left[i] = last;
                }
            }
        } else {
            for i in 0..bs {
                e.left[i] = 129;
            }
        }

        if !have_above {
            for i in 0..=2 * bs {
                e.above[i] = 127;
            }
            return e;
        }

        let row = (y0 - 1) * stride;
        // libvpx splits the above row into **three** cases by how much of it is
        // inside the frame, and the amount copied is measured from the block
        // origin (`frame_w - x0`) — not from the end of the block. Getting that
        // wrong only misplaces pixels in the partial superblock at the frame's
        // right edge, which is why the first symptom here was a wrong column
        // 171 of 176 and nothing else.
        let put_above = |e: &mut Edges, i: usize, v: u8| e.above[1 + i] = v;
        if x0 + 2 * bs <= frame_w {
            for i in 0..bs {
                put_above(&mut e, i, plane[row + x0 + i]);
            }
            if have_right && bs == 4 {
                for i in bs..2 * bs {
                    put_above(&mut e, i, plane[row + x0 + i]);
                }
            } else {
                let last = e.above[bs];
                for i in bs..2 * bs {
                    put_above(&mut e, i, last);
                }
            }
        } else if x0 + bs <= frame_w {
            let r = frame_w - x0; // bs <= r < 2*bs
            if have_right && bs == 4 {
                for i in 0..r {
                    put_above(&mut e, i, plane[row + x0 + i]);
                }
                let last = e.above[r];
                for i in r..2 * bs {
                    put_above(&mut e, i, last);
                }
            } else {
                for i in 0..bs {
                    put_above(&mut e, i, plane[row + x0 + i]);
                }
                let last = e.above[bs];
                for i in bs..2 * bs {
                    put_above(&mut e, i, last);
                }
            }
        } else {
            // The block itself straddles the right edge.
            let r = frame_w.saturating_sub(x0);
            for i in 0..r {
                put_above(&mut e, i, plane[row + x0 + i]);
            }
            let last = if r > 0 { e.above[r] } else { 127 };
            for i in r..2 * bs {
                put_above(&mut e, i, last);
            }
        }
        e.above[0] = if have_left { plane[row + x0 - 1] } else { 129 };
        e
    }
}

/// The **4x4 directional predictors are hand-written special cases in libvpx**,
/// not the generic ones evaluated at `bs == 4` — `intra_pred_no_4x4` generates
/// the generic form only for 8/16/32, and each of the six directional modes has
/// its own `vpx_*_predictor_4x4_c`.
///
/// They are not merely an unrolling. `d45` at 4x4 continues the diagonal through
/// `above[7]` where the generic version clamps to `above[bs-1]`; `d63` at 4x4
/// carries two entries marked "differs from vp8". So running the generic code at
/// 4x4 gives a *plausible* diagonal that is wrong in its lower-right triangle —
/// which is exactly how this was found: every `BLOCK_4X4` in the frame came out
/// with a growing diagonal error while every larger block was bit-exact.
///
/// `DST(x, y)` is `dst[y * stride + x]`, matching the reference's macro.
fn predict4x4_directional(dst: &mut [u8], stride: usize, mode: IntraMode, e: &Edges) -> bool {
    let a = |i: usize| e.above[1 + i] as u16;
    let l = |i: usize| e.left[i] as u16;
    let x = e.above[0] as u16; // above[-1]
    let mut d = [0u8; 16];
    macro_rules! set {
        ($xx:expr, $yy:expr, $v:expr) => {
            d[$yy * 4 + $xx] = $v
        };
    }
    let avg2 = |p: u16, q: u16| ((p + q + 1) >> 1) as u8;
    let avg3 = |p: u16, q: u16, r: u16| ((p + 2 * q + r + 2) >> 2) as u8;

    match mode {
        IntraMode::D207 => {
            let (i, j, k, ll) = (l(0), l(1), l(2), l(3));
            set!(0, 0, avg2(i, j));
            let v = avg2(j, k);
            set!(2, 0, v);
            set!(0, 1, v);
            let v = avg2(k, ll);
            set!(2, 1, v);
            set!(0, 2, v);
            set!(1, 0, avg3(i, j, k));
            let v = avg3(j, k, ll);
            set!(3, 0, v);
            set!(1, 1, v);
            let v = avg3(k, ll, ll);
            set!(3, 1, v);
            set!(1, 2, v);
            let v = ll as u8;
            set!(3, 2, v);
            set!(2, 2, v);
            set!(0, 3, v);
            set!(1, 3, v);
            set!(2, 3, v);
            set!(3, 3, v);
        }
        IntraMode::D63 => {
            let (aa, b, c, dd, ee, f, g) = (a(0), a(1), a(2), a(3), a(4), a(5), a(6));
            set!(0, 0, avg2(aa, b));
            let v = avg2(b, c);
            set!(1, 0, v);
            set!(0, 2, v);
            let v = avg2(c, dd);
            set!(2, 0, v);
            set!(1, 2, v);
            let v = avg2(dd, ee);
            set!(3, 0, v);
            set!(2, 2, v);
            set!(3, 2, avg2(ee, f));
            set!(0, 1, avg3(aa, b, c));
            let v = avg3(b, c, dd);
            set!(1, 1, v);
            set!(0, 3, v);
            let v = avg3(c, dd, ee);
            set!(2, 1, v);
            set!(1, 3, v);
            let v = avg3(dd, ee, f);
            set!(3, 1, v);
            set!(2, 3, v);
            set!(3, 3, avg3(ee, f, g));
        }
        IntraMode::D45 => {
            let (aa, b, c, dd, ee, f, g, h) =
                (a(0), a(1), a(2), a(3), a(4), a(5), a(6), a(7));
            set!(0, 0, avg3(aa, b, c));
            let v = avg3(b, c, dd);
            set!(1, 0, v);
            set!(0, 1, v);
            let v = avg3(c, dd, ee);
            set!(2, 0, v);
            set!(1, 1, v);
            set!(0, 2, v);
            let v = avg3(dd, ee, f);
            set!(3, 0, v);
            set!(2, 1, v);
            set!(1, 2, v);
            set!(0, 3, v);
            let v = avg3(ee, f, g);
            set!(3, 1, v);
            set!(2, 2, v);
            set!(1, 3, v);
            let v = avg3(f, g, h);
            set!(3, 2, v);
            set!(2, 3, v);
            set!(3, 3, h as u8);
        }
        IntraMode::D117 => {
            let (i, j, k) = (l(0), l(1), l(2));
            let (aa, b, c, dd) = (a(0), a(1), a(2), a(3));
            let v = avg2(x, aa);
            set!(0, 0, v);
            set!(1, 2, v);
            let v = avg2(aa, b);
            set!(1, 0, v);
            set!(2, 2, v);
            let v = avg2(b, c);
            set!(2, 0, v);
            set!(3, 2, v);
            set!(3, 0, avg2(c, dd));
            set!(0, 3, avg3(k, j, i));
            set!(0, 2, avg3(j, i, x));
            let v = avg3(i, x, aa);
            set!(0, 1, v);
            set!(1, 3, v);
            let v = avg3(x, aa, b);
            set!(1, 1, v);
            set!(2, 3, v);
            let v = avg3(aa, b, c);
            set!(2, 1, v);
            set!(3, 3, v);
            set!(3, 1, avg3(b, c, dd));
        }
        IntraMode::D135 => {
            let (i, j, k, ll) = (l(0), l(1), l(2), l(3));
            let (aa, b, c, dd) = (a(0), a(1), a(2), a(3));
            set!(0, 3, avg3(j, k, ll));
            let v = avg3(i, j, k);
            set!(1, 3, v);
            set!(0, 2, v);
            let v = avg3(x, i, j);
            set!(2, 3, v);
            set!(1, 2, v);
            set!(0, 1, v);
            let v = avg3(aa, x, i);
            set!(3, 3, v);
            set!(2, 2, v);
            set!(1, 1, v);
            set!(0, 0, v);
            let v = avg3(b, aa, x);
            set!(3, 2, v);
            set!(2, 1, v);
            set!(1, 0, v);
            let v = avg3(c, b, aa);
            set!(3, 1, v);
            set!(2, 0, v);
            set!(3, 0, avg3(dd, c, b));
        }
        IntraMode::D153 => {
            let (i, j, k, ll) = (l(0), l(1), l(2), l(3));
            let (aa, b, c) = (a(0), a(1), a(2));
            let v = avg2(i, x);
            set!(0, 0, v);
            set!(2, 1, v);
            let v = avg2(j, i);
            set!(0, 1, v);
            set!(2, 2, v);
            let v = avg2(k, j);
            set!(0, 2, v);
            set!(2, 3, v);
            set!(0, 3, avg2(ll, k));
            set!(3, 0, avg3(aa, b, c));
            set!(2, 0, avg3(x, aa, b));
            let v = avg3(i, x, aa);
            set!(1, 0, v);
            set!(3, 1, v);
            let v = avg3(j, i, x);
            set!(1, 1, v);
            set!(3, 2, v);
            let v = avg3(k, j, i);
            set!(1, 2, v);
            set!(3, 3, v);
            set!(1, 3, avg3(ll, k, j));
        }
        _ => return false,
    }
    for yy in 0..4 {
        for xx in 0..4 {
            let p = yy * stride + xx;
            if p < dst.len() {
                dst[p] = d[yy * 4 + xx];
            }
        }
    }
    true
}

/// Predict a `bs`x`bs` block into `dst` (row stride `stride`).
///
/// `have_above`/`have_left` select the DC variant: VP9 has four separate DC
/// predictors, not one that averages whatever is there, and picking the wrong
/// one at a frame edge shifts the whole block's level.
pub fn predict(
    dst: &mut [u8],
    stride: usize,
    bs: usize,
    mode: IntraMode,
    e: &Edges,
    have_above: bool,
    have_left: bool,
) {
    // The six directional modes have their own 4x4 forms in the reference; the
    // generic code below is only correct for 8/16/32.
    if bs == 4 && predict4x4_directional(dst, stride, mode, e) {
        return;
    }
    // `above[1 + i]`, so a local closure keeps the predictors readable and the
    // corner accessible as `ab(-1)`.
    let ab = |i: isize| -> u8 { e.above[(i + 1) as usize] };
    let lf = |i: usize| -> u8 { e.left[i] };
    let put = |dst: &mut [u8], r: usize, c: usize, v: u8| {
        let p = r * stride + c;
        if p < dst.len() {
            dst[p] = v;
        }
    };

    match mode {
        IntraMode::V => {
            for r in 0..bs {
                for c in 0..bs {
                    put(dst, r, c, ab(c as isize));
                }
            }
        }
        IntraMode::H => {
            for r in 0..bs {
                for c in 0..bs {
                    put(dst, r, c, lf(r));
                }
            }
        }
        IntraMode::Tm => {
            let tl = ab(-1) as i32;
            for r in 0..bs {
                let l = lf(r) as i32;
                for c in 0..bs {
                    let v = (l + ab(c as isize) as i32 - tl).clamp(0, 255) as u8;
                    put(dst, r, c, v);
                }
            }
        }
        IntraMode::Dc => {
            let v = match (have_above, have_left) {
                (true, true) => {
                    let mut s = 0u32;
                    for i in 0..bs {
                        s += ab(i as isize) as u32 + lf(i) as u32;
                    }
                    ((s + bs as u32) / (2 * bs as u32)) as u8
                }
                (true, false) => {
                    let mut s = 0u32;
                    for i in 0..bs {
                        s += ab(i as isize) as u32;
                    }
                    ((s + (bs as u32 >> 1)) / bs as u32) as u8
                }
                (false, true) => {
                    let mut s = 0u32;
                    for i in 0..bs {
                        s += lf(i) as u32;
                    }
                    ((s + (bs as u32 >> 1)) / bs as u32) as u8
                }
                (false, false) => 128,
            };
            for r in 0..bs {
                for c in 0..bs {
                    put(dst, r, c, v);
                }
            }
        }
        IntraMode::D45 => {
            let above_right = ab(bs as isize - 1);
            let mut row0 = [0u8; 32];
            for x in 0..bs - 1 {
                row0[x] = avg3(ab(x as isize), ab(x as isize + 1), ab(x as isize + 2));
            }
            row0[bs - 1] = above_right;
            for c in 0..bs {
                put(dst, 0, c, row0[c]);
            }
            for r in 1..bs {
                let size = bs - 1 - r;
                for c in 0..size {
                    put(dst, r, c, row0[r + c]);
                }
                for c in size..bs {
                    put(dst, r, c, above_right);
                }
            }
        }
        IntraMode::D63 => {
            let mut r0 = [0u8; 32];
            let mut r1 = [0u8; 32];
            for c in 0..bs {
                r0[c] = avg2(ab(c as isize), ab(c as isize + 1));
                r1[c] = avg3(ab(c as isize), ab(c as isize + 1), ab(c as isize + 2));
            }
            for c in 0..bs {
                put(dst, 0, c, r0[c]);
                put(dst, 1, c, r1[c]);
            }
            let last = ab(bs as isize - 1);
            let mut size = bs as isize - 2;
            let mut r = 2usize;
            while r < bs {
                let src = r >> 1;
                for c in 0..bs {
                    let v0 = if (c as isize) < size { r0[src + c] } else { last };
                    let v1 = if (c as isize) < size { r1[src + c] } else { last };
                    put(dst, r, c, v0);
                    if r + 1 < bs {
                        put(dst, r + 1, c, v1);
                    }
                }
                r += 2;
                size -= 1;
            }
        }
        IntraMode::D135 => {
            // The outer border runs from bottom-left up to top-right; every row
            // is then a shifted copy of it. Building it explicitly is what makes
            // this mode tractable — the pixel-wise form is where sign errors go.
            let mut border = [0u8; 64];
            for i in 0..bs - 2 {
                border[i] = avg3(lf(bs - 3 - i), lf(bs - 2 - i), lf(bs - 1 - i));
            }
            border[bs - 2] = avg3(ab(-1), lf(0), lf(1));
            border[bs - 1] = avg3(lf(0), ab(-1), ab(0));
            border[bs] = avg3(ab(-1), ab(0), ab(1));
            for i in 0..bs - 2 {
                border[bs + 1 + i] = avg3(ab(i as isize), ab(i as isize + 1), ab(i as isize + 2));
            }
            for r in 0..bs {
                for c in 0..bs {
                    put(dst, r, c, border[bs - 1 - r + c]);
                }
            }
        }
        IntraMode::D117 => {
            let mut buf = [0u8; 32 * 32];
            for c in 0..bs {
                buf[c] = avg2(ab(c as isize - 1), ab(c as isize));
            }
            buf[bs] = avg3(lf(0), ab(-1), ab(0));
            for c in 1..bs {
                buf[bs + c] = avg3(ab(c as isize - 2), ab(c as isize - 1), ab(c as isize));
            }
            if bs > 2 {
                buf[2 * bs] = avg3(ab(-1), lf(0), lf(1));
                for r in 3..bs {
                    buf[(r - 2 + 2) * bs] = avg3(lf(r - 3), lf(r - 2), lf(r - 1));
                }
            }
            for r in 2..bs {
                for c in 1..bs {
                    buf[r * bs + c] = buf[(r - 2) * bs + c - 1];
                }
            }
            for r in 0..bs {
                for c in 0..bs {
                    put(dst, r, c, buf[r * bs + c]);
                }
            }
        }
        IntraMode::D153 => {
            let mut buf = [0u8; 32 * 32];
            buf[0] = avg2(ab(-1), lf(0));
            for r in 1..bs {
                buf[r * bs] = avg2(lf(r - 1), lf(r));
            }
            buf[1] = avg3(lf(0), ab(-1), ab(0));
            if bs > 1 {
                buf[bs + 1] = avg3(ab(-1), lf(0), lf(1));
            }
            for r in 2..bs {
                buf[r * bs + 1] = avg3(lf(r - 2), lf(r - 1), lf(r));
            }
            for c in 0..bs - 2 {
                buf[2 + c] = avg3(ab(c as isize - 1), ab(c as isize), ab(c as isize + 1));
            }
            for r in 1..bs {
                for c in 0..bs - 2 {
                    buf[r * bs + 2 + c] = buf[(r - 1) * bs + c];
                }
            }
            for r in 0..bs {
                for c in 0..bs {
                    put(dst, r, c, buf[r * bs + c]);
                }
            }
        }
        IntraMode::D207 => {
            let mut buf = [0u8; 32 * 32];
            for r in 0..bs - 1 {
                buf[r * bs] = avg2(lf(r), lf(r + 1));
            }
            buf[(bs - 1) * bs] = lf(bs - 1);
            for r in 0..bs.saturating_sub(2) {
                buf[r * bs + 1] = avg3(lf(r), lf(r + 1), lf(r + 2));
            }
            if bs >= 2 {
                buf[(bs - 2) * bs + 1] = avg3(lf(bs - 2), lf(bs - 1), lf(bs - 1));
                buf[(bs - 1) * bs + 1] = lf(bs - 1);
            }
            for c in 0..bs.saturating_sub(2) {
                buf[(bs - 1) * bs + 2 + c] = lf(bs - 1);
            }
            for r in (0..bs - 1).rev() {
                for c in 0..bs - 2 {
                    buf[r * bs + 2 + c] = buf[(r + 1) * bs + c];
                }
            }
            for r in 0..bs {
                for c in 0..bs {
                    put(dst, r, c, buf[r * bs + c]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edges(above: u8, left: u8, corner: u8) -> Edges {
        let mut e = Edges { above: [above; 65], left: [left; 32] };
        e.above[0] = corner;
        e
    }

    #[test_case]
    fn v_and_h_copy_their_edge() {
        let e = edges(200, 50, 10);
        let mut d = [0u8; 16];
        predict(&mut d, 4, 4, IntraMode::V, &e, true, true);
        assert!(d.iter().all(|&p| p == 200), "V must copy the row above");
        predict(&mut d, 4, 4, IntraMode::H, &e, true, true);
        assert!(d.iter().all(|&p| p == 50), "H must copy the column left");
    }

    #[test_case]
    fn dc_has_four_variants_not_one() {
        // above = 200, left = 100. All four DC predictors must differ, which is
        // what proves the availability flags are honoured rather than averaged
        // over whatever the edge buffer happened to hold.
        let e = edges(200, 100, 0);
        let mut d = [0u8; 16];
        predict(&mut d, 4, 4, IntraMode::Dc, &e, true, true);
        assert_eq!(d[0], 150, "both: mean of both edges");
        predict(&mut d, 4, 4, IntraMode::Dc, &e, true, false);
        assert_eq!(d[0], 200, "above only");
        predict(&mut d, 4, 4, IntraMode::Dc, &e, false, true);
        assert_eq!(d[0], 100, "left only");
        predict(&mut d, 4, 4, IntraMode::Dc, &e, false, false);
        assert_eq!(d[0], 128, "neither: the defined 128");
    }

    #[test_case]
    fn tm_is_left_plus_above_minus_corner() {
        let mut e = edges(0, 0, 100);
        for i in 0..8 {
            e.above[1 + i] = 120;
            e.left[i] = 110;
        }
        let mut d = [0u8; 16];
        predict(&mut d, 4, 4, IntraMode::Tm, &e, true, true);
        assert_eq!(d[0], 130, "110 + 120 - 100");
        // …and it clamps rather than wrapping.
        let mut e2 = edges(0, 0, 255);
        for i in 0..8 {
            e2.above[1 + i] = 0;
            e2.left[i] = 0;
        }
        predict(&mut d, 4, 4, IntraMode::Tm, &e2, true, true);
        assert_eq!(d[0], 0, "clamped, not wrapped");
    }

    #[test_case]
    fn a_flat_edge_predicts_a_flat_block_in_every_mode() {
        // With every edge pixel equal, every predictor — however directional —
        // must produce that same value: all of them are weighted means of edge
        // samples. This catches an out-of-range read or a transposed index in
        // the awkward D117/D153/D207 buffers, where an error otherwise only
        // shows up as a slightly wrong texture.
        for &bs in &[4usize, 8, 16, 32] {
            let e = edges(77, 77, 77);
            for m in 0..10u8 {
                let mode = IntraMode::from_index(m);
                let mut d = alloc::vec![0u8; bs * bs];
                predict(&mut d, bs, bs, mode, &e, true, true);
                assert!(
                    d.iter().all(|&p| p == 77),
                    "{:?} at {}x{} did not predict a flat block: {:?}",
                    mode,
                    bs,
                    bs,
                    &d[..bs.min(8)]
                );
            }
        }
    }

    #[test_case]
    fn every_mode_writes_every_pixel() {
        // A predictor that leaves a pixel untouched inherits whatever the frame
        // buffer held — which decodes differently on a seek than on linear
        // play, the hardest class of bug to reproduce.
        //
        // Checked by predicting twice into buffers pre-filled with *different*
        // sentinels and requiring the results to match: an unwritten pixel is
        // the only way they can differ. Looking for one sentinel value in the
        // output cannot work — a predictor is a weighted mean of the edges, so
        // any sentinel inside their range is also a legitimate result (the
        // first version of this test failed on H_PRED at 32x32 because
        // `200 - 29` is `0xAB`).
        for &bs in &[4usize, 8, 16, 32] {
            let mut e = edges(60, 200, 30);
            for i in 0..2 * bs {
                e.above[1 + i] = (40 + i) as u8;
            }
            for i in 0..bs {
                e.left[i] = (200 - i) as u8;
            }
            for m in 0..10u8 {
                let mode = IntraMode::from_index(m);
                let mut a = alloc::vec![0x00u8; bs * bs];
                let mut b = alloc::vec![0xFFu8; bs * bs];
                predict(&mut a, bs, bs, mode, &e, true, true);
                predict(&mut b, bs, bs, mode, &e, true, true);
                let at = a.iter().zip(b.iter()).position(|(x, y)| x != y);
                assert!(
                    at.is_none(),
                    "{:?} at {}x{} left pixel {} unwritten",
                    mode,
                    bs,
                    bs,
                    at.unwrap_or(0)
                );
            }
        }
    }
}
