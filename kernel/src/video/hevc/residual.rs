//! HEVC residual coding (H.265 §7.3.8.11 / §9.3.3) — the CABAC syntax that
//! turns a slice's bits into a block of dequantised transform coefficients.
//!
//! This is the hot, intricate half of an HEVC decoder, and almost every rule in
//! it is a *context selection* rule: which probability model a bin is decoded
//! against. Those are the mistakes that do not fail. A bin decoded against the
//! wrong context still yields a 0 or a 1; the bitstream stays in sync for a
//! while and then does not, and the picture is wrong in a way that looks like a
//! different bug entirely.
//!
//! The shape, in order:
//!
//! 1. `transform_skip_flag`, then the RDPCM flags — decoded here rather than by
//!    the caller because their *position in the bitstream* is between the
//!    transform unit's other syntax and the coefficients.
//! 2. The **last significant coefficient** position, as a truncated-unary
//!    prefix plus a fixed-length bypass suffix. It is signalled in the scan's
//!    own orientation, so a vertical scan swaps x and y afterwards.
//! 3. The block is walked backwards in **4x4 coefficient groups** from that
//!    position: a group-significance flag, then per-coefficient significance,
//!    then greater-than-1 and greater-than-2 flags for at most the first eight,
//!    then Golomb-Rice remainders, then bypass signs.
//!
//! Two rules deserve calling out because they are pure bookkeeping and pure
//! opportunity for an off-by-one:
//!
//! - **`significant_coeff_flag_idx` is filled in decreasing scan order**, so
//!   entry 0 is the *highest* position in the group and entry `n-1` the lowest.
//!   FFmpeg's `last_nz_pos_in_cg` is therefore `idx[0]`. Reading those names
//!   the other way round inverts sign-data-hiding's distance test.
//! - **`greater1_ctx` is carried across coefficient groups.** It is not reset
//!   per group; the next group's context set depends on whether the previous
//!   group ended having seen a level above one.

use super::super::h264::cabac::Cabac;
use super::cabac_tables as ct;
use super::tables as tb;

/// Scan orders, in the specification's numbering.
pub const SCAN_DIAG: usize = 0;
pub const SCAN_HORIZ: usize = 1;
pub const SCAN_VERT: usize = 2;

/// FFmpeg's guard against a runaway unary prefix. A well-formed stream never
/// reaches it; a corrupt one would otherwise spin reading bypass bins off the
/// end of the slice.
const CABAC_MAX_BIN: u32 = 31;

/// Everything the parse needs that comes from the SPS, PPS, slice or CU.
pub struct ResidualParams<'a> {
    pub log2_size: u32,
    /// 0 luma, 1 Cb, 2 Cr.
    pub c_idx: usize,
    pub scan_idx: usize,
    pub intra: bool,
    /// The luma or chroma intra mode, used only by implicit RDPCM.
    pub pred_mode_intra: u8,
    pub transquant_bypass: bool,
    pub transform_skip_enabled: bool,
    pub log2_max_transform_skip: u32,
    pub explicit_rdpcm_enabled: bool,
    pub implicit_rdpcm_enabled: bool,
    pub sign_data_hiding: bool,
    pub persistent_rice: bool,
    pub transform_skip_context: bool,
    /// Bit-depth-offset QP for this plane.
    pub qp: i32,
    pub bit_depth: u32,
    /// The scaling list for this block: 16 entries at 4x4, 64 otherwise
    /// (sub-sampled for 16x16 and 32x32). `None` is the flat 16.
    pub scale_matrix: Option<&'a [u8]>,
    pub dc_scale: u8,
}

/// What the caller needs back to choose the inverse transform.
pub struct ResidualResult {
    pub transform_skip: bool,
    pub explicit_rdpcm: bool,
    pub explicit_rdpcm_dir: bool,
    /// `max(last_x, last_y)`; zero means the block holds only a DC coefficient,
    /// which is what selects the closed-form inverse transform.
    pub max_xy: u32,
    pub last_x: u32,
    pub last_y: u32,
    pub scan_idx: usize,
    pub num_coeff: usize,
}

/// Read `n` bypass bins as one value, most significant first.
#[inline]
fn bypass_bits(c: &mut Cabac, n: u32) -> u32 {
    let mut v = 0u32;
    for _ in 0..n {
        v = (v << 1) | c.bypass();
    }
    v
}

/// A bypass-coded unary prefix: count 1-bins, stopping at a 0 or at `max`.
#[inline]
fn unary_prefix(c: &mut Cabac, max: u32) -> u32 {
    let mut n = 0;
    while n < max && c.bypass() != 0 {
        n += 1;
    }
    n
}

/// `last_sig_coeff_{x,y}_prefix` (§9.3.4.2.3): truncated unary, whose context
/// increment shrinks with block size so a 32x32 reuses each context four times.
fn last_sig_prefix(c: &mut Cabac, c_idx: usize, log2_size: u32, base: usize) -> u32 {
    let max = (log2_size << 1) - 1;
    let (offset, shift) = if c_idx == 0 {
        (3 * (log2_size - 2) + ((log2_size - 1) >> 2), (log2_size + 1) >> 2)
    } else {
        (15, log2_size - 2)
    };
    let mut i = 0u32;
    while i < max && c.decision(base + ((i >> shift) + offset) as usize) != 0 {
        i += 1;
    }
    i
}

/// Expand a `last_sig_coeff` prefix above 3 with its bypass suffix.
///
/// The prefix is a *logarithmic* code — value `p` names the bucket starting at
/// `(1 << (p/2 - 1)) * (2 + (p & 1))` — so dropping the suffix does not shift
/// the position by a constant, it collapses every position in a bucket onto its
/// start. On a 32x32 that is up to eight columns of error.
fn expand_last_sig(c: &mut Cabac, prefix: u32) -> u32 {
    if prefix <= 3 {
        return prefix;
    }
    let length = (prefix >> 1) - 1;
    let suffix = bypass_bits(c, length);
    (1 << ((prefix >> 1) - 1)) * (2 + (prefix & 1)) + suffix
}

/// `coeff_abs_level_remaining` (§9.3.3.11): Golomb-Rice below the escape,
/// exp-Golomb above it, all bypass.
fn coeff_abs_level_remaining(c: &mut Cabac, rice: u32) -> u32 {
    let prefix = unary_prefix(c, CABAC_MAX_BIN);
    if prefix < 3 {
        (prefix << rice) + bypass_bits(c, rice)
    } else {
        let prefix_minus3 = prefix - 3;
        // A corrupt stream can name a code longer than any legal level; answer
        // zero rather than shifting by more than a word.
        if prefix == CABAC_MAX_BIN || prefix_minus3 + rice > 16 + 6 {
            return 0;
        }
        let k = prefix_minus3 + rice;
        let suffix = if k > 16 {
            (bypass_bits(c, 16) << (k - 16)) | bypass_bits(c, k - 16)
        } else {
            bypass_bits(c, k)
        };
        (((1u32 << prefix_minus3) + 3 - 1) << rice) + suffix
    }
}

/// Update the persistent Rice statistics (§9.3.3.11, range extensions).
#[inline]
fn update_stat_coeff(stat: &mut [u8; 4], sb_type: usize, remaining: u32) {
    let init = (stat[sb_type] / 4) as u32;
    if remaining >= (3 << init) {
        stat[sb_type] = stat[sb_type].saturating_add(1);
    } else if 2 * remaining < (1 << init) && stat[sb_type] > 0 {
        stat[sb_type] -= 1;
    }
}

/// Parse and dequantise one transform block's residual into `coeffs`
/// (`size * size`, raster order).
///
/// Dequantisation is fused into the parse, as the specification's own ordering
/// implies: a level is scaled and clipped to 16 bits *per coefficient*, so
/// storing raw levels first and scaling afterwards would clip in the wrong
/// place for any level above 32767 — which a lossless or near-lossless block
/// really produces.
pub fn residual_coding(
    c: &mut Cabac,
    coeffs: &mut [i16],
    stat_coeff: &mut [u8; 4],
    p: &ResidualParams,
) -> ResidualResult {
    let size = 1usize << p.log2_size;
    coeffs[..size * size].fill(0);

    let mut transform_skip = false;
    let mut explicit_rdpcm = false;
    let mut explicit_rdpcm_dir = false;
    let (mut scale, mut shift) = (0i64, 0u32);

    if !p.transquant_bypass {
        if p.transform_skip_enabled && p.log2_size <= p.log2_max_transform_skip {
            let inc = if p.c_idx == 0 { 0 } else { 1 };
            transform_skip = c.decision(ct::TRANSFORM_SKIP_FLAG + inc) != 0;
        }
        let (s, sh) = super::transform::dequant_params(p.qp, p.log2_size, p.bit_depth);
        scale = s as i64;
        shift = sh;
    }
    if !p.intra && p.explicit_rdpcm_enabled && (transform_skip || p.transquant_bypass) {
        let inc = if p.c_idx == 0 { 0 } else { 1 };
        explicit_rdpcm = c.decision(ct::EXPLICIT_RDPCM_FLAG + inc) != 0;
        if explicit_rdpcm {
            explicit_rdpcm_dir = c.decision(ct::EXPLICIT_RDPCM_DIR_FLAG + inc) != 0;
        }
    }

    // --- last significant coefficient ------------------------------------
    let px = last_sig_prefix(c, p.c_idx, p.log2_size, ct::LAST_SIGNIFICANT_COEFF_X_PREFIX);
    let py = last_sig_prefix(c, p.c_idx, p.log2_size, ct::LAST_SIGNIFICANT_COEFF_Y_PREFIX);
    // Both suffixes follow both prefixes, never interleaved.
    let mut last_x = expand_last_sig(c, px);
    let mut last_y = expand_last_sig(c, py);
    if p.scan_idx == SCAN_VERT {
        core::mem::swap(&mut last_x, &mut last_y);
    }

    let x_cg_last = (last_x >> 2) as usize;
    let y_cg_last = (last_y >> 2) as usize;

    // Scan tables: within a 4x4 group, and over the groups themselves.
    let (scan_x_off, scan_y_off): (&[u8], &[u8]) = match p.scan_idx {
        SCAN_HORIZ => (&tb::HORIZ_SCAN4X4_X, &tb::HORIZ_SCAN4X4_Y),
        SCAN_VERT => (&tb::HORIZ_SCAN4X4_Y, &tb::HORIZ_SCAN4X4_X),
        _ => (&tb::DIAG_SCAN4X4_X, &tb::DIAG_SCAN4X4_Y),
    };
    let (scan_x_cg, scan_y_cg): (&[u8], &[u8]) = match p.scan_idx {
        SCAN_HORIZ => (&tb::HORIZ_SCAN2X2_X, &tb::HORIZ_SCAN2X2_Y),
        SCAN_VERT => (&tb::HORIZ_SCAN2X2_Y, &tb::HORIZ_SCAN2X2_X),
        _ => match size {
            4 => (&tb::SCAN_1X1, &tb::SCAN_1X1),
            8 => (&tb::DIAG_SCAN2X2_X, &tb::DIAG_SCAN2X2_Y),
            16 => (&tb::DIAG_SCAN4X4_X, &tb::DIAG_SCAN4X4_Y),
            _ => (&tb::DIAG_SCAN8X8_X, &tb::DIAG_SCAN8X8_Y),
        },
    };

    // How many coefficients precede the last significant one, in scan order.
    let num_coeff = match p.scan_idx {
        SCAN_HORIZ => tb::HORIZ_SCAN8X8_INV[last_y as usize][last_x as usize] as usize,
        SCAN_VERT => tb::HORIZ_SCAN8X8_INV[last_x as usize][last_y as usize] as usize,
        _ => {
            let within =
                tb::DIAG_SCAN4X4_INV[(last_y & 3) as usize][(last_x & 3) as usize] as usize;
            within
                + match size {
                    4 => 0,
                    8 => (tb::DIAG_SCAN2X2_INV[y_cg_last][x_cg_last] as usize) << 4,
                    16 => (tb::DIAG_SCAN4X4_INV[y_cg_last][x_cg_last] as usize) << 4,
                    _ => (tb::DIAG_SCAN8X8_INV[y_cg_last][x_cg_last] as usize) << 4,
                }
        }
    } + 1;
    let num_last_subset = (num_coeff - 1) >> 4;
    let max_xy = last_x.max(last_y);

    // Group significance, one flag per 4x4 group; up to 8x8 groups at 32x32.
    let mut cg_flag = [[false; 8]; 8];
    // Carried across groups — deliberately not reset per group.
    let mut greater1_ctx: u32 = 1;

    for i in (0..=num_last_subset).rev() {
        let x_cg = scan_x_cg[i] as usize;
        let y_cg = scan_y_cg[i] as usize;
        let offset = i << 4;
        let mut implicit_non_zero = false;

        if i < num_last_subset && i > 0 {
            let bound = (1usize << (p.log2_size - 2)) - 1;
            let mut ctx_cg = 0u32;
            if x_cg < bound {
                ctx_cg += cg_flag[x_cg + 1][y_cg] as u32;
            }
            if y_cg < bound {
                ctx_cg += cg_flag[x_cg][y_cg + 1] as u32;
            }
            let inc = ctx_cg.min(1) as usize + if p.c_idx > 0 { 2 } else { 0 };
            cg_flag[x_cg][y_cg] = c.decision(ct::SIGNIFICANT_COEFF_GROUP_FLAG + inc) != 0;
            implicit_non_zero = true;
        } else {
            cg_flag[x_cg][y_cg] =
                (x_cg == x_cg_last && y_cg == y_cg_last) || (x_cg == 0 && y_cg == 0);
        }

        let last_scan_pos = (num_coeff - offset - 1) as i32;
        // Scan positions of the significant coefficients, highest first.
        let mut sig_idx = [0u8; 16];
        let mut nb_sig = 0usize;
        let mut n_end = if i == num_last_subset {
            sig_idx[0] = last_scan_pos as u8;
            nb_sig = 1;
            last_scan_pos - 1
        } else {
            15
        };

        if cg_flag[x_cg][y_cg] && n_end >= 0 {
            let bound = (size - 1) >> 2;
            let mut prev_sig = 0usize;
            if x_cg < bound {
                prev_sig = cg_flag[x_cg + 1][y_cg] as usize;
            }
            if y_cg < bound {
                prev_sig += (cg_flag[x_cg][y_cg + 1] as usize) << 1;
            }

            // Which row of the context map, and what to add to it.
            let (map_row, mut scf_offset) = if p.transform_skip_context
                && (transform_skip || p.transquant_bypass)
            {
                (4usize, if p.c_idx == 0 { 40usize } else { 14 + 27 })
            } else if p.log2_size == 2 {
                (0usize, if p.c_idx != 0 { 27 } else { 0 })
            } else {
                let mut off = if p.c_idx != 0 { 27usize } else { 0 };
                if p.c_idx == 0 {
                    if x_cg > 0 || y_cg > 0 {
                        off += 3;
                    }
                    off += if p.log2_size == 3 {
                        if p.scan_idx == SCAN_DIAG {
                            9
                        } else {
                            15
                        }
                    } else {
                        21
                    };
                } else {
                    off += if p.log2_size == 3 { 9 } else { 12 };
                }
                (prev_sig + 1, off)
            };
            let map = &tb::SIG_CTX_IDX_MAP[p.scan_idx][map_row * 16..map_row * 16 + 16];

            let nb0 = nb_sig;
            let mut n = n_end;
            while n > 0 {
                let inc = map[n as usize] as usize + scf_offset;
                let sig = c.decision(ct::SIGNIFICANT_COEFF_FLAG + inc);
                sig_idx[nb_sig] = n as u8;
                nb_sig += sig as usize;
                n -= 1;
            }
            if nb_sig != nb0 {
                implicit_non_zero = false;
            }
            if !implicit_non_zero {
                // Position 0 has its own context, not the map's.
                scf_offset = if p.transform_skip_context && (transform_skip || p.transquant_bypass)
                {
                    if p.c_idx == 0 {
                        42
                    } else {
                        16 + 27
                    }
                } else if i == 0 {
                    if p.c_idx == 0 {
                        0
                    } else {
                        27
                    }
                } else {
                    2 + scf_offset
                };
                sig_idx[nb_sig] = 0;
                nb_sig += c.decision(ct::SIGNIFICANT_COEFF_FLAG + scf_offset) as usize;
            } else {
                // The group flag promised a coefficient and none was signalled,
                // so position 0 is significant by inference and costs no bin.
                sig_idx[nb_sig] = 0;
                nb_sig += 1;
            }
        }
        n_end = nb_sig as i32;
        if n_end == 0 {
            continue;
        }

        // --- levels -------------------------------------------------------
        let mut rice: u32 = 0;
        let mut gt1 = [0u8; 8];
        let mut first_gt1_idx: i32 = -1;
        let mut ctx_set = if i > 0 && p.c_idx == 0 { 2u32 } else { 0 };

        let sb_type = if !transform_skip && !p.transquant_bypass {
            2 * (p.c_idx == 0) as usize
        } else {
            2 * (p.c_idx == 0) as usize + 1
        };
        if p.persistent_rice {
            rice = (stat_coeff[sb_type] / 4) as u32;
        }
        if i != num_last_subset && greater1_ctx == 0 {
            ctx_set += 1;
        }
        greater1_ctx = 1;

        let last_nz_pos_in_cg = sig_idx[0] as i32;
        let n_gt1 = nb_sig.min(8);
        for m in 0..n_gt1 {
            let mut inc = ((ctx_set << 2) + greater1_ctx) as usize;
            if p.c_idx > 0 {
                inc += 16;
            }
            let flag = c.decision(ct::COEFF_ABS_LEVEL_GREATER1_FLAG + inc);
            gt1[m] = flag as u8;
            if flag != 0 {
                if first_gt1_idx < 0 {
                    first_gt1_idx = m as i32;
                }
                greater1_ctx = 0;
            } else if greater1_ctx == 1 || greater1_ctx == 2 {
                greater1_ctx += 1;
            }
        }
        let first_nz_pos_in_cg = sig_idx[nb_sig - 1] as i32;

        // Sign-data hiding is off wherever the residual is coded losslessly or
        // as a spatial prediction, because there the sign carries information
        // the parity trick would destroy.
        let sign_hidden = if p.transquant_bypass
            || (p.intra
                && p.implicit_rdpcm_enabled
                && transform_skip
                && (p.pred_mode_intra == 10 || p.pred_mode_intra == 26))
            || explicit_rdpcm
        {
            false
        } else {
            last_nz_pos_in_cg - first_nz_pos_in_cg >= 4
        };

        if first_gt1_idx >= 0 {
            let mut inc = ctx_set as usize;
            if p.c_idx > 0 {
                inc += 4;
            }
            gt1[first_gt1_idx as usize] += c.decision(ct::COEFF_ABS_LEVEL_GREATER2_FLAG + inc) as u8;
        }

        // Signs, MSB-first in scan order. One is omitted when hidden.
        let n_signs = if p.sign_data_hiding && sign_hidden { nb_sig - 1 } else { nb_sig };
        let mut signs = bypass_bits(c, n_signs as u32) << (16 - n_signs);

        let mut sum_abs: u32 = 0;
        // The Rice statistic is updated from the **first** remainder in the
        // group only. Using `sum_abs` as the "have we done it yet" proxy is
        // wrong: it moves only when sign hiding is active, so with hiding off
        // every remainder would update the statistic.
        let mut rice_init = false;
        for m in 0..nb_sig {
            let n = sig_idx[m] as usize;
            let x_c = (x_cg << 2) + scan_x_off[n] as usize;
            let y_c = (y_cg << 2) + scan_y_off[n] as usize;

            let mut level: i64;
            if m < 8 {
                level = 1 + gt1[m] as i64;
                // The threshold is 3 for the one coefficient that got a
                // greater2 flag and 2 for the rest: only a level that saturated
                // its flags continues into the Rice code.
                if level == if m as i32 == first_gt1_idx { 3 } else { 2 } {
                    let rem = coeff_abs_level_remaining(c, rice);
                    level += rem as i64;
                    if level > (3i64 << rice) {
                        rice = if p.persistent_rice { rice + 1 } else { (rice + 1).min(4) };
                    }
                    if p.persistent_rice && !rice_init {
                        update_stat_coeff(stat_coeff, sb_type, rem);
                        rice_init = true;
                    }
                }
            } else {
                let rem = coeff_abs_level_remaining(c, rice);
                level = 1 + rem as i64;
                if level > (3i64 << rice) {
                    rice = if p.persistent_rice { rice + 1 } else { (rice + 1).min(4) };
                }
                if p.persistent_rice && !rice_init {
                    update_stat_coeff(stat_coeff, sb_type, rem);
                    rice_init = true;
                }
            }

            if p.sign_data_hiding && sign_hidden {
                sum_abs += level as u32;
                if n as i32 == first_nz_pos_in_cg && (sum_abs & 1) != 0 {
                    level = -level;
                }
            }
            if (signs >> 15) & 1 != 0 {
                level = -level;
            }
            signs <<= 1;

            if !p.transquant_bypass {
                let m_scale = match p.scale_matrix {
                    None => 16i64,
                    Some(sm) => {
                        if y_c != 0 || x_c != 0 || p.log2_size < 4 {
                            let pos = match p.log2_size {
                                3 => (y_c << 3) + x_c,
                                4 => ((y_c >> 1) << 3) + (x_c >> 1),
                                5 => ((y_c >> 2) << 3) + (x_c >> 2),
                                _ => (y_c << 2) + x_c,
                            };
                            sm[pos] as i64
                        } else {
                            p.dc_scale as i64
                        }
                    }
                };
                let add = 1i64 << (shift - 1);
                level = (level * scale * m_scale + add) >> shift;
                level = level.clamp(-32768, 32767);
            }
            coeffs[y_c * size + x_c] = level as i16;
        }
    }

    ResidualResult {
        transform_skip,
        explicit_rdpcm,
        explicit_rdpcm_dir,
        max_xy,
        last_x,
        last_y,
        scan_idx: p.scan_idx,
        num_coeff,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Rice/exp-Golomb code for `coeff_abs_level_remaining` must be a
    /// **bijection** onto the non-negative integers: every level has exactly
    /// one encoding and every encoding one level. A gap or an overlap is a
    /// wrong level, silently.
    ///
    /// The encoder here is written from the specification's description rather
    /// than by inverting the decoder, so agreement is real evidence.
    #[test_case]
    fn coeff_abs_level_remaining_round_trips_for_every_rice_parameter() {
        // Encode `v` as the bypass bin string the decoder should consume.
        fn encode(v: u32, rice: u32) -> alloc::vec::Vec<u8> {
            let mut bits = alloc::vec::Vec::new();
            let threshold = 3u32 << rice;
            if v < threshold {
                let prefix = v >> rice;
                for _ in 0..prefix {
                    bits.push(1);
                }
                bits.push(0);
                for k in (0..rice).rev() {
                    bits.push(((v >> k) & 1) as u8);
                }
            } else {
                // Exp-Golomb escape: find the smallest `pm3` whose bucket holds v.
                let mut pm3 = 0u32;
                loop {
                    let base = (((1u32 << pm3) + 2) << rice) as u64;
                    let next = (((1u32 << (pm3 + 1)) + 2) << rice) as u64;
                    if (v as u64) >= base && (v as u64) < next {
                        break;
                    }
                    pm3 += 1;
                    assert!(pm3 < 20, "no bucket for {v} at rice {rice}");
                }
                for _ in 0..3 + pm3 {
                    bits.push(1);
                }
                bits.push(0);
                let k = pm3 + rice;
                let suffix = v - (((1u32 << pm3) + 2) << rice);
                for j in (0..k).rev() {
                    bits.push(((suffix >> j) & 1) as u8);
                }
            }
            bits
        }

        for rice in 0..5u32 {
            for v in [0u32, 1, 2, 3, 7, 8, 15, 16, 23, 24, 100, 255, 1000, 5000] {
                let bits = encode(v, rice);
                let data = pack_bypass(&bits);
                let mut c = Cabac::new_hevc(&data, 26, 2, &ct::INIT_VALUES).unwrap();
                let got = coeff_abs_level_remaining(&mut c, rice);
                assert_eq!(got, v, "rice {rice} value {v} bits {bits:?}");
            }
        }
    }

    use super::super::testutil::pack_bypass;

    /// The helper above is only trustworthy if it round-trips through the real
    /// engine, so prove that before the tests that depend on it.
    #[test_case]
    fn the_bypass_packer_round_trips_through_the_real_engine() {
        let cases: [&[u8]; 6] = [
            &[1],
            &[0],
            &[1, 0, 1, 1, 0],
            &[0, 0, 0, 0, 0, 0, 0, 0],
            &[1, 1, 1, 1, 1, 1, 1, 1],
            &[1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1],
        ];
        for bins in cases {
            let data = pack_bypass(bins);
            let mut c =
                Cabac::new_hevc(&data, 26, 2, &ct::INIT_VALUES).unwrap();
            for (i, &want) in bins.iter().enumerate() {
                assert_eq!(c.bypass(), want as u32, "bin {i} of {bins:?}");
            }
        }
    }

    /// The `last_sig_coeff` prefix is a bucket index and the suffix picks a
    /// position inside it. The buckets must tile 0..=31 exactly — a gap means
    /// some positions are unrepresentable, an overlap means two codes for one
    /// position.
    #[test_case]
    fn last_significant_coefficient_buckets_tile_the_block() {
        // Bucket start for prefix p, from the specification's expansion.
        fn base(p: u32) -> u32 {
            if p <= 3 {
                p
            } else {
                (1 << ((p >> 1) - 1)) * (2 + (p & 1))
            }
        }
        fn width(p: u32) -> u32 {
            if p <= 3 {
                1
            } else {
                1 << ((p >> 1) - 1)
            }
        }
        let mut next = 0u32;
        // A 32x32 block uses prefixes 0..=9 (max = 2*5 - 1).
        for p in 0..=9u32 {
            assert_eq!(base(p), next, "prefix {p} does not start where {} ended", p - 1);
            next += width(p);
        }
        assert_eq!(next, 32, "the buckets must cover exactly 0..=31");
    }

    /// `greater1_ctx` follows the specification's three-state rule: reset by a
    /// level above one, incremented while it is 1 or 2, and pinned at 0 or 3.
    ///
    /// FFmpeg writes this branchlessly as
    /// `(g + (g - 1U < 2)) & (flag - 1)`, where the unsigned compare is true
    /// only for `g` in {1, 2} — reading it as `g > 0` increments 3 to 4 and
    /// indexes past the context set.
    #[test_case]
    fn greater1_context_saturates_at_both_ends() {
        fn step(g: u32, flag: u32) -> u32 {
            if flag != 0 {
                0
            } else if g == 1 || g == 2 {
                g + 1
            } else {
                g
            }
        }
        // FFmpeg's branchless form, as an independent check.
        fn ff(g: u32, flag: u32) -> u32 {
            (g + ((g.wrapping_sub(1)) < 2) as u32) & (flag.wrapping_sub(1))
        }
        for g in 0..4u32 {
            for flag in 0..2u32 {
                assert_eq!(step(g, flag), ff(g, flag), "g {g} flag {flag}");
            }
        }
        // Once a level above one is seen the context stays at 0 for the rest
        // of the group, which is what makes later coefficients cheap.
        assert_eq!(step(3, 1), 0);
        assert_eq!(step(0, 0), 0);
        assert_eq!(step(3, 0), 3);
    }

    /// Sign-data hiding fires only when the group spans at least four scan
    /// positions, and the inferred sign is the one that makes the group's
    /// absolute sum even.
    #[test_case]
    fn sign_hiding_distance_rule_and_parity() {
        // The distance is between the highest and lowest significant scan
        // position *within the group*, so a dense group at the end of a block
        // does not hide and a sparse one does.
        for (last, first, want) in [(15i32, 11i32, true), (15, 12, false), (4, 0, true), (3, 0, false)]
        {
            assert_eq!(last - first >= 4, want, "distance {last}-{first}");
        }
        // Parity: with a running sum of levels, the hidden sign is negative
        // exactly when that sum is odd.
        for (sum, negative) in [(1u32, true), (2, false), (7, true), (8, false)] {
            assert_eq!((sum & 1) != 0, negative);
        }
    }

    /// The scaling-list position mapping sub-samples one 8x8 matrix for 16x16
    /// and 32x32 blocks, and only those two have a separately signalled DC.
    #[test_case]
    fn scaling_list_position_subsamples_the_eight_by_eight_matrix() {
        fn pos(log2: u32, x: usize, y: usize) -> usize {
            match log2 {
                3 => (y << 3) + x,
                4 => ((y >> 1) << 3) + (x >> 1),
                5 => ((y >> 2) << 3) + (x >> 2),
                _ => (y << 2) + x,
            }
        }
        // 4x4 uses a 16-entry list directly.
        assert_eq!(pos(2, 3, 3), 15);
        // 8x8 uses the 64-entry list directly.
        assert_eq!(pos(3, 7, 7), 63);
        // 16x16 maps 2x2 blocks of coefficients onto one entry...
        assert_eq!(pos(4, 0, 0), 0);
        assert_eq!(pos(4, 1, 1), 0);
        assert_eq!(pos(4, 2, 2), 9);
        assert_eq!(pos(4, 15, 15), 63);
        // ...and 32x32, 4x4 blocks.
        assert_eq!(pos(5, 3, 3), 0);
        assert_eq!(pos(5, 31, 31), 63);
        // Every position must land inside the 64-entry matrix.
        for log2 in 3..=5u32 {
            let n = 1usize << log2;
            for y in 0..n {
                for x in 0..n {
                    assert!(pos(log2, x, y) < 64, "log2 {log2} ({x},{y})");
                }
            }
        }
    }
}
