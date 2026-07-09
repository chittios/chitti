//! H.264 inter prediction (§8.4): median MV prediction + luma 6-tap half/quarter
//! -pel and chroma bilinear motion compensation. Ports the reference validated
//! bit-exact against PyAV. Pure integer math over the previous frame's planes.

/// Clamp to `[lo, hi]`.
#[inline]
fn clampi(v: i32, lo: i32, hi: i32) -> i32 {
    v.clamp(lo, hi)
}

/// Reference luma/chroma sample with edge replication (out-of-frame → nearest).
#[inline]
fn ipel(refp: &[u8], w: usize, h: usize, x: i32, y: i32) -> i32 {
    let xx = clampi(x, 0, w as i32 - 1) as usize;
    let yy = clampi(y, 0, h as i32 - 1) as usize;
    refp[yy * w + xx] as i32
}

/// The 6-tap half-pel filter `[1,-5,20,20,-5,1]`.
#[inline]
fn tap(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32) -> i32 {
    a - 5 * b + 20 * c + 20 * d - 5 * e + f
}

/// Predict a `bw×bh` luma block at `(bx,by)` from `refp` with quarter-pel MV
/// `(mvx,mvy)` (§8.4.2.2.1). Returns the block samples (row-major).
pub fn luma_block(refp: &[u8], w: usize, h: usize, bx: usize, by: usize, bw: usize, bh: usize, mvx: i32, mvy: i32) -> alloc::vec::Vec<i32> {
    let mut out = alloc::vec![0i32; bw * bh];
    let dx = mvx & 3;
    let dy = mvy & 3;
    let ox = bx as i32 + (mvx >> 2);
    let oy = by as i32 + (mvy >> 2);
    for j in 0..bh {
        for i in 0..bw {
            let x = ox + i as i32;
            let y = oy + j as i32;
            let g = |xx: i32, yy: i32| ipel(refp, w, h, xx, yy);
            let bhalf = |yy: i32| tap(g(x - 2, yy), g(x - 1, yy), g(x, yy), g(x + 1, yy), g(x + 2, yy), g(x + 3, yy));
            let hhalf = |xx: i32| tap(g(xx, y - 2), g(xx, y - 1), g(xx, y), g(xx, y + 1), g(xx, y + 2), g(xx, y + 3));
            let v = if dx == 0 && dy == 0 {
                g(x, y)
            } else if dy == 0 {
                let b = clampi((bhalf(y) + 16) >> 5, 0, 255);
                if dx == 2 {
                    b
                } else if dx == 1 {
                    (g(x, y) + b + 1) >> 1
                } else {
                    (b + g(x + 1, y) + 1) >> 1
                }
            } else if dx == 0 {
                let hh = clampi((hhalf(x) + 16) >> 5, 0, 255);
                if dy == 2 {
                    hh
                } else if dy == 1 {
                    (g(x, y) + hh + 1) >> 1
                } else {
                    (hh + g(x, y + 1) + 1) >> 1
                }
            } else {
                let b = clampi((bhalf(y) + 16) >> 5, 0, 255);
                let s = clampi((bhalf(y + 1) + 16) >> 5, 0, 255);
                let hh = clampi((hhalf(x) + 16) >> 5, 0, 255);
                let m = clampi((hhalf(x + 1) + 16) >> 5, 0, 255);
                let jj = tap(bhalf(y - 2), bhalf(y - 1), bhalf(y), bhalf(y + 1), bhalf(y + 2), bhalf(y + 3));
                let cj = clampi((jj + 512) >> 10, 0, 255);
                match (dx, dy) {
                    (2, 2) => cj,
                    (1, 2) => (hh + cj + 1) >> 1,
                    (3, 2) => (cj + m + 1) >> 1,
                    (2, 1) => (b + cj + 1) >> 1,
                    (2, 3) => (cj + s + 1) >> 1,
                    (1, 1) => (b + hh + 1) >> 1,
                    (3, 1) => (b + m + 1) >> 1,
                    (1, 3) => (hh + s + 1) >> 1,
                    _ => (m + s + 1) >> 1, // (3,3)
                }
            };
            out[j * bw + i] = clampi(v, 0, 255);
        }
    }
    out
}

/// Predict a `bw×bh` chroma block (§8.4.2.2.2): eighth-pel bilinear. `mvx/mvy`
/// are the luma quarter-pel MV (chroma is half-res → eighth-pel).
pub fn chroma_block(refp: &[u8], cw: usize, ch: usize, bx: usize, by: usize, bw: usize, bh: usize, mvx: i32, mvy: i32) -> alloc::vec::Vec<i32> {
    let mut out = alloc::vec![0i32; bw * bh];
    let xf = mvx & 7;
    let yf = mvy & 7;
    let ox = bx as i32 + (mvx >> 3);
    let oy = by as i32 + (mvy >> 3);
    for j in 0..bh {
        for i in 0..bw {
            let x = ox + i as i32;
            let y = oy + j as i32;
            let a = ipel(refp, cw, ch, x, y);
            let b = ipel(refp, cw, ch, x + 1, y);
            let c = ipel(refp, cw, ch, x, y + 1);
            let d = ipel(refp, cw, ch, x + 1, y + 1);
            out[j * bw + i] = ((8 - xf) * (8 - yf) * a + xf * (8 - yf) * b + (8 - xf) * yf * c + xf * yf * d + 32) >> 6;
        }
    }
    out
}

/// Median of three (the MV median predictor component).
#[inline]
pub fn median3(a: i32, b: i32, c: i32) -> i32 {
    a + b + c - a.max(b).max(c) - a.min(b).min(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn full_pel_copies_reference() {
        // 4x4 ref region, integer MV → exact copy.
        let refp: alloc::vec::Vec<u8> = (0..64u8).collect();
        let blk = luma_block(&refp, 8, 8, 1, 1, 4, 4, 0, 0);
        for j in 0..4 {
            for i in 0..4 {
                assert_eq!(blk[j * 4 + i], refp[(1 + j) * 8 + (1 + i)] as i32);
            }
        }
    }

    #[test_case]
    fn half_pel_is_symmetric_average_on_ramp() {
        // A horizontal ramp: the 6-tap half-pel at dx=2 of a linear ramp equals
        // the midpoint (the filter sums to 32 and is symmetric).
        let refp: alloc::vec::Vec<u8> = (0..8).flat_map(|_| (0..8u8).map(|x| x * 10)).collect();
        let blk = luma_block(&refp, 8, 8, 1, 1, 2, 2, 2, 0); // dx=2 (half), dy=0
        // Between columns 1 (=10) and 2 (=20) → 15.
        assert_eq!(blk[0], 15);
    }

    #[test_case]
    fn chroma_bilinear_quarter() {
        // 2x2 ref [0,40 / 80,120]; MV frac (4,4) → center = average = 60.
        let refp = alloc::vec![0u8, 40, 80, 120];
        let blk = chroma_block(&refp, 2, 2, 0, 0, 1, 1, 4, 4);
        assert_eq!(blk[0], (0 + 40 + 80 + 120) / 4);
    }

    #[test_case]
    fn median3_matches_middle() {
        assert_eq!(median3(3, 1, 2), 2);
        assert_eq!(median3(-5, 10, 0), 0);
        assert_eq!(median3(7, 7, 1), 7);
    }
}
