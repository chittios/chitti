//! H.264 intra prediction (§8.3): Intra_4x4 (9 modes), Intra_16x16 (4 modes),
//! and chroma (4 modes). Each predictor is a pure function of the reconstructed
//! neighbour samples (top row incl. top-right, left column, top-left corner)
//! plus their availability, per the spec's substitution rules.
//!
//! Predicted samples are pixel values (0..=255). Residual add + clipping happens
//! at reconstruction. The simple directional/DC/plane modes are unit-tested
//! here; the diagonal 4×4 modes are exercised by the full-frame diff harness.

/// Intra_4x4 prediction for one 4×4 block (§8.3.1.2). `top[0..4]` = `p[0..3,-1]`,
/// `top[4..8]` = `p[4..7,-1]` (top-right), `left[0..4]` = `p[-1,0..3]`,
/// `corner` = `p[-1,-1]`. Returns the 4×4 predictor in raster order.
///
/// Availability substitution the caller need not do: when top-right is missing
/// but top is present, it is filled from `p[3,-1]` (per §8.3.1.2.* ). DC mode
/// falls back by availability. Directional modes assume their required
/// neighbours are available (the decoder only selects a mode when they are).
pub fn intra4x4(
    mode: u8,
    top: &[i32; 8],
    left: &[i32; 4],
    corner: i32,
    avail_top: bool,
    avail_left: bool,
    avail_tr: bool,
) -> [i32; 16] {
    // p[x,-1] for x in -1..=7 ; p[-1,y] for y in -1..=3.
    let mut a = *top;
    if avail_top && !avail_tr {
        // Top-right unavailable → replicate p[3,-1] into x=4..7.
        for x in 4..8 {
            a[x] = top[3];
        }
    }
    let pt = |x: i32| -> i32 {
        if x < 0 {
            corner
        } else {
            a[x as usize]
        }
    };
    let pl = |y: i32| -> i32 {
        if y < 0 {
            corner
        } else {
            left[y as usize]
        }
    };
    let mut out = [0i32; 16];
    let mut set = |x: usize, y: usize, v: i32| out[y * 4 + x] = v;
    match mode {
        0 => {
            // Vertical.
            for y in 0..4 {
                for x in 0..4 {
                    set(x, y, a[x]);
                }
            }
        }
        1 => {
            // Horizontal.
            for y in 0..4 {
                for x in 0..4 {
                    set(x, y, left[y]);
                }
            }
        }
        2 => {
            // DC with availability fallback.
            let v = match (avail_top, avail_left) {
                (true, true) => (a[0] + a[1] + a[2] + a[3] + left[0] + left[1] + left[2] + left[3] + 4) >> 3,
                (true, false) => (a[0] + a[1] + a[2] + a[3] + 2) >> 2,
                (false, true) => (left[0] + left[1] + left[2] + left[3] + 2) >> 2,
                (false, false) => 128,
            };
            for p in out.iter_mut() {
                *p = v;
            }
        }
        3 => {
            // Diagonal Down Left.
            for y in 0..4i32 {
                for x in 0..4i32 {
                    let v = if x == 3 && y == 3 {
                        (a[6] + 3 * a[7] + 2) >> 2
                    } else {
                        let i = (x + y) as usize;
                        (a[i] + 2 * a[i + 1] + a[i + 2] + 2) >> 2
                    };
                    set(x as usize, y as usize, v);
                }
            }
        }
        4 => {
            // Diagonal Down Right.
            for y in 0..4i32 {
                for x in 0..4i32 {
                    let v = if x > y {
                        (pt(x - y - 2) + 2 * pt(x - y - 1) + pt(x - y) + 2) >> 2
                    } else if x < y {
                        (pl(y - x - 2) + 2 * pl(y - x - 1) + pl(y - x) + 2) >> 2
                    } else {
                        (pt(0) + 2 * corner + pl(0) + 2) >> 2
                    };
                    set(x as usize, y as usize, v);
                }
            }
        }
        5 => {
            // Vertical Right.
            for y in 0..4i32 {
                for x in 0..4i32 {
                    let z = 2 * x - y;
                    let v = if z >= 0 && z % 2 == 0 {
                        (pt(x - (y >> 1) - 1) + pt(x - (y >> 1)) + 1) >> 1
                    } else if z >= 0 {
                        (pt(x - (y >> 1) - 2) + 2 * pt(x - (y >> 1) - 1) + pt(x - (y >> 1)) + 2) >> 2
                    } else if z == -1 {
                        (pl(0) + 2 * corner + pt(0) + 2) >> 2
                    } else {
                        (pl(y - 1) + 2 * pl(y - 2) + pl(y - 3) + 2) >> 2
                    };
                    set(x as usize, y as usize, v);
                }
            }
        }
        6 => {
            // Horizontal Down.
            for y in 0..4i32 {
                for x in 0..4i32 {
                    let z = 2 * y - x;
                    let v = if z >= 0 && z % 2 == 0 {
                        (pl(y - (x >> 1) - 1) + pl(y - (x >> 1)) + 1) >> 1
                    } else if z >= 0 {
                        (pl(y - (x >> 1) - 2) + 2 * pl(y - (x >> 1) - 1) + pl(y - (x >> 1)) + 2) >> 2
                    } else if z == -1 {
                        (pl(0) + 2 * corner + pt(0) + 2) >> 2
                    } else {
                        (pt(x - 1) + 2 * pt(x - 2) + pt(x - 3) + 2) >> 2
                    };
                    set(x as usize, y as usize, v);
                }
            }
        }
        7 => {
            // Vertical Left.
            for y in 0..4i32 {
                for x in 0..4i32 {
                    let i = (x + (y >> 1)) as usize;
                    let v = if y % 2 == 0 {
                        (a[i] + a[i + 1] + 1) >> 1
                    } else {
                        (a[i] + 2 * a[i + 1] + a[i + 2] + 2) >> 2
                    };
                    set(x as usize, y as usize, v);
                }
            }
        }
        _ => {
            // 8: Horizontal Up.
            for y in 0..4i32 {
                for x in 0..4i32 {
                    let z = x + 2 * y;
                    let i = (y + (x >> 1)) as usize;
                    let v = if z < 5 && z % 2 == 0 {
                        (pl(i as i32) + pl(i as i32 + 1) + 1) >> 1
                    } else if z < 5 {
                        (pl(i as i32) + 2 * pl(i as i32 + 1) + pl(i as i32 + 2) + 2) >> 2
                    } else if z == 5 {
                        (left[2] + 3 * left[3] + 2) >> 2
                    } else {
                        left[3]
                    };
                    set(x as usize, y as usize, v);
                }
            }
        }
    }
    out
}

/// Intra_16x16 prediction (§8.3.3). `top`/`left` are the 16 neighbour samples on
/// each edge; `corner` = `p[-1,-1]`. Returns the 16×16 predictor (raster).
pub fn intra16x16(
    mode: u8,
    top: &[i32; 16],
    left: &[i32; 16],
    corner: i32,
    avail_top: bool,
    avail_left: bool,
) -> [i32; 256] {
    let mut out = [0i32; 256];
    match mode {
        0 => {
            for y in 0..16 {
                for x in 0..16 {
                    out[y * 16 + x] = top[x];
                }
            }
        }
        1 => {
            for y in 0..16 {
                for x in 0..16 {
                    out[y * 16 + x] = left[y];
                }
            }
        }
        2 => {
            let sum_t: i32 = top.iter().sum();
            let sum_l: i32 = left.iter().sum();
            let v = match (avail_top, avail_left) {
                (true, true) => (sum_t + sum_l + 16) >> 5,
                (true, false) => (sum_t + 8) >> 4,
                (false, true) => (sum_l + 8) >> 4,
                (false, false) => 128,
            };
            for p in out.iter_mut() {
                *p = v;
            }
        }
        _ => {
            // 3: Plane (§8.3.3.4).
            let mut h = 0i32;
            let mut v = 0i32;
            for i in 0..8i32 {
                h += (i + 1) * (top[(8 + i) as usize] - if 6 - i >= 0 { top[(6 - i) as usize] } else { corner });
                v += (i + 1) * (left[(8 + i) as usize] - if 6 - i >= 0 { left[(6 - i) as usize] } else { corner });
            }
            let b = (5 * h + 32) >> 6;
            let c = (5 * v + 32) >> 6;
            let aa = 16 * (top[15] + left[15]);
            for y in 0..16i32 {
                for x in 0..16i32 {
                    let val = (aa + b * (x - 7) + c * (y - 7) + 16) >> 5;
                    out[(y * 16 + x) as usize] = val.clamp(0, 255);
                }
            }
        }
    }
    out
}

/// Chroma intra prediction over an 8×8 block (§8.3.4). Modes: 0 DC, 1 Horizontal,
/// 2 Vertical, 3 Plane. DC is computed per 4×4 quadrant with the spec's
/// availability fallback.
pub fn intra_chroma(
    mode: u8,
    top: &[i32; 8],
    left: &[i32; 8],
    corner: i32,
    avail_top: bool,
    avail_left: bool,
) -> [i32; 64] {
    let mut out = [0i32; 64];
    match mode {
        1 => {
            for y in 0..8 {
                for x in 0..8 {
                    out[y * 8 + x] = left[y];
                }
            }
        }
        2 => {
            for y in 0..8 {
                for x in 0..8 {
                    out[y * 8 + x] = top[x];
                }
            }
        }
        3 => {
            let mut h = 0i32;
            let mut v = 0i32;
            for i in 0..4i32 {
                h += (i + 1) * (top[(4 + i) as usize] - if 2 - i >= 0 { top[(2 - i) as usize] } else { corner });
                v += (i + 1) * (left[(4 + i) as usize] - if 2 - i >= 0 { left[(2 - i) as usize] } else { corner });
            }
            let b = (17 * h + 16) >> 5;
            let c = (17 * v + 16) >> 5;
            let aa = 16 * (top[7] + left[7]);
            for y in 0..8i32 {
                for x in 0..8i32 {
                    let val = (aa + b * (x - 3) + c * (y - 3) + 16) >> 5;
                    out[(y * 8 + x) as usize] = val.clamp(0, 255);
                }
            }
        }
        _ => {
            // 0: DC, computed per 4×4 quadrant (§8.3.4.1).
            for qy in 0..2 {
                for qx in 0..2 {
                    let t: i32 = (0..4).map(|k| top[qx * 4 + k]).sum();
                    let l: i32 = (0..4).map(|k| left[qy * 4 + k]).sum();
                    // Quadrant DC selection rule.
                    let dc = if (qx == 0 && qy == 0) || (qx == 1 && qy == 1) {
                        match (avail_top, avail_left) {
                            (true, true) => (t + l + 4) >> 3,
                            (true, false) => (t + 2) >> 2,
                            (false, true) => (l + 2) >> 2,
                            (false, false) => 128,
                        }
                    } else if qx == 1 && qy == 0 {
                        if avail_top {
                            (t + 2) >> 2
                        } else if avail_left {
                            (l + 2) >> 2
                        } else {
                            128
                        }
                    } else {
                        // qx==0, qy==1
                        if avail_left {
                            (l + 2) >> 2
                        } else if avail_top {
                            (t + 2) >> 2
                        } else {
                            128
                        }
                    };
                    for y in 0..4 {
                        for x in 0..4 {
                            out[(qy * 4 + y) * 8 + qx * 4 + x] = dc;
                        }
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn intra4x4_vertical_and_horizontal() {
        let top = [10, 20, 30, 40, 0, 0, 0, 0];
        let left = [1, 2, 3, 4];
        let v = intra4x4(0, &top, &left, 0, true, true, false); // (avail_top, avail_left, avail_tr)
        // Every row equals the top row.
        for y in 0..4 {
            assert_eq!(&v[y * 4..y * 4 + 4], &[10, 20, 30, 40]);
        }
        let h = intra4x4(1, &top, &left, 0, true, true, false);
        // Every column equals the left value for its row.
        for y in 0..4 {
            assert!(h[y * 4..y * 4 + 4].iter().all(|&p| p == left[y]));
        }
    }

    #[test_case]
    fn intra4x4_dc_availability() {
        let top = [8, 8, 8, 8, 0, 0, 0, 0];
        let left = [4, 4, 4, 4];
        // Args are (avail_top, avail_left, avail_tr).
        // Both: (32 + 16 + 4) >> 3 = 6.
        assert_eq!(intra4x4(2, &top, &left, 0, true, true, false)[0], 6);
        // Top only: (32 + 2) >> 2 = 8.
        assert_eq!(intra4x4(2, &top, &left, 0, true, false, false)[0], 8);
        // Left only: (16 + 2) >> 2 = 4.
        assert_eq!(intra4x4(2, &top, &left, 0, false, true, false)[0], 4);
        // Neither: 128.
        assert_eq!(intra4x4(2, &top, &left, 0, false, false, false)[0], 128);
    }

    #[test_case]
    fn intra16x16_dc_and_vertical() {
        let top = [4i32; 16];
        let left = [8i32; 16];
        // DC both: (64 + 128 + 16) >> 5 = 6.
        assert_eq!(intra16x16(2, &top, &left, 0, true, true)[0], 6);
        // Vertical replicates the top row.
        let vpred = intra16x16(0, &top, &left, 0, true, true);
        assert!(vpred.iter().all(|&p| p == 4));
    }

    #[test_case]
    fn intra16x16_plane_is_flat_for_constant_neighbours() {
        // Constant neighbours → H = V = 0, so plane reduces to the flat DC-ish
        // average 16*(top+left)/32 = neighbour value.
        let top = [100i32; 16];
        let left = [100i32; 16];
        let p = intra16x16(3, &top, &left, 100, true, true);
        assert!(p.iter().all(|&v| v == 100), "plane of constant neighbours");
    }

    #[test_case]
    fn intra_chroma_dc_horizontal_vertical() {
        let top = [2i32; 8];
        let left = [6i32; 8];
        // Vertical replicates top.
        assert!(intra_chroma(2, &top, &left, 0, true, true).iter().all(|&p| p == 2));
        // Horizontal replicates left.
        assert!(intra_chroma(1, &top, &left, 0, true, true).iter().all(|&p| p == 6));
        // Top-left quadrant DC both avail: (8 + 24 + 4) >> 3 = 4.
        assert_eq!(intra_chroma(0, &top, &left, 0, true, true)[0], 4);
    }
}
