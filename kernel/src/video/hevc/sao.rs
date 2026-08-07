//! Sample Adaptive Offset (H.265 §8.7.3) — the post-deblocking filter that
//! HEVC has and H.264 does not.
//!
//! SAO adds a small signalled offset to each sample, classified either by its
//! **value** (band offset) or by its relation to two neighbours along one of
//! four directions (edge offset). It is the last thing that touches a
//! reconstructed picture before it enters the DPB, so an error here is stored
//! and propagates through every frame that references it.
//!
//! Two facts shape this module:
//!
//! - **The classifier reads unfiltered neighbours.** An edge-offset decision
//!   for sample `x` compares it against neighbours that must be the
//!   *pre-SAO* values, so the filter cannot run in place. Doing so makes
//!   column `x`'s classification depend on column `x-1`'s offset, which
//!   biases the whole row in one direction — a diagonal smear that looks like
//!   a motion-compensation bug.
//! - **Category 0 has no offset.** `edge_idx` maps the five relations onto
//!   `{1, 2, 0, 3, 4}`, where index 0 is the "neither neighbour differs" case
//!   and its offset is always zero. Storing four offsets and indexing them
//!   directly is off by one for two of the five categories.

/// Offsets as the specification indexes them: entry 0 is the always-zero
/// category, entries 1..=4 are the signalled values.
pub type SaoOffsets = [i16; 5];

/// Edge-offset classes (§7.4.9.3), in signalling order.
pub const EO_HORIZONTAL: usize = 0;
pub const EO_VERTICAL: usize = 1;
pub const EO_DIAG_135: usize = 2;
pub const EO_DIAG_45: usize = 3;

/// The two neighbours consulted per class, as `(dx, dy)` pairs.
///
/// Note the naming: class 2 is the **135-degree** edge (down-right), class 3
/// the 45-degree one. Swapping them decodes without complaint and rotates
/// every diagonal edge decision by 90 degrees.
const POS: [[(i32, i32); 2]; 4] =
    [[(-1, 0), (1, 0)], [(0, -1), (0, 1)], [(-1, -1), (1, 1)], [(1, -1), (-1, 1)]];

/// `edgeIdx` remap (§8.7.3.2 table 8-15): the raw sum `2 + sign(a) + sign(b)`
/// runs 0..=4 in the order *valley, half-valley, flat, half-peak, peak*, and
/// the specification renumbers it so the flat case is 0.
const EDGE_IDX: [usize; 5] = [1, 2, 0, 3, 4];

#[inline]
fn cmp(a: i32, b: i32) -> i32 {
    (a > b) as i32 - (a < b) as i32
}

/// Band offset: four consecutive bands starting at `left_class`, each of
/// `1 << (bit_depth - 5)` sample values, get an offset; everything else is
/// unchanged.
///
/// The band index wraps modulo 32, which is deliberate — an encoder can put
/// the four bands across the top of the range and round to the bottom.
pub fn band_filter(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    offsets: &SaoOffsets,
    left_class: usize,
    width: usize,
    height: usize,
    bit_depth: u32,
) {
    let mut table = [0i16; 32];
    for k in 0..4 {
        table[(k + left_class) & 31] = offsets[k + 1];
    }
    let shift = bit_depth - 5;
    let max = (1i32 << bit_depth) - 1;
    for y in 0..height {
        for x in 0..width {
            let s = src[y * src_stride + x] as i32;
            let v = s + table[((s >> shift) & 31) as usize] as i32;
            dst[y * dst_stride + x] = v.clamp(0, max) as u8;
        }
    }
}

/// Edge offset over the interior of a region.
///
/// `src` must be a *separate* buffer with one sample of margin on every side
/// that the region needs — `src_origin` is the index in `src` of the region's
/// top-left sample, so the classifier can step outside it. `avail` says which
/// of the four borders may be consulted; where a border is unavailable the
/// samples along it are **left unfiltered** rather than clamped, because the
/// specification excludes them from SAO entirely and clamping would classify
/// them against themselves (always flat, so always offset 0 — the same answer
/// by accident on the flat case and the wrong one otherwise).
pub struct Borders {
    pub left: bool,
    pub right: bool,
    pub above: bool,
    pub below: bool,
}

pub fn edge_filter(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    src_origin: usize,
    offsets: &SaoOffsets,
    eo: usize,
    width: usize,
    height: usize,
    borders: &Borders,
    bit_depth: u32,
) {
    let [(ax, ay), (bx, by)] = POS[eo];
    let max = (1i32 << bit_depth) - 1;
    // Restrict to the rows and columns whose neighbours exist.
    let x0 = if !borders.left && (ax < 0 || bx < 0) { 1 } else { 0 };
    let x1 = if !borders.right && (ax > 0 || bx > 0) { width.saturating_sub(1) } else { width };
    let y0 = if !borders.above && (ay < 0 || by < 0) { 1 } else { 0 };
    let y1 = if !borders.below && (ay > 0 || by > 0) { height.saturating_sub(1) } else { height };

    for y in y0..y1 {
        for x in x0..x1 {
            let i = src_origin as isize + y as isize * src_stride as isize + x as isize;
            let s = src[i as usize] as i32;
            let a = src[(i + ax as isize + ay as isize * src_stride as isize) as usize] as i32;
            let b = src[(i + bx as isize + by as isize * src_stride as isize) as usize] as i32;
            let idx = EDGE_IDX[(2 + cmp(s, a) + cmp(s, b)) as usize];
            let v = s + offsets[idx] as i32;
            dst[y * dst_stride + x] = v.clamp(0, max) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test_case]
    fn band_offset_moves_only_its_four_bands() {
        // 8-bit: each band is 8 sample values. left_class 4 covers 32..=63.
        let src: alloc::vec::Vec<u8> = (0..64u8).collect();
        let mut dst = vec![0u8; 64];
        let offs: SaoOffsets = [0, 3, -3, 5, -5];
        band_filter(&mut dst, 64, &src, 64, &offs, 4, 64, 1, 8);
        for v in 0..64usize {
            let band = v >> 3;
            let want = match band {
                4 => v as i32 + 3,
                5 => v as i32 - 3,
                6 => v as i32 + 5,
                7 => v as i32 - 5,
                _ => v as i32,
            };
            assert_eq!(dst[v] as i32, want, "value {v} (band {band})");
        }
    }

    #[test_case]
    fn band_classes_wrap_around_thirty_two() {
        // Starting at class 30 the four bands are 30, 31, 0, 1 — the wrap is
        // deliberate: an encoder can straddle the top of the range.
        let src = [8u8, 248, 0, 255];
        let mut dst = [0u8; 4];
        let offs: SaoOffsets = [0, 1, 2, 3, 4];
        band_filter(&mut dst, 4, &src, 4, &offs, 30, 4, 1, 8);
        assert_eq!(dst[0], 8 + 4, "8 is class 1, the fourth band after wrapping");
        assert_eq!(dst[1], 248 + 2, "248 is class 31, the second band");
        assert_eq!(dst[2], 0 + 3, "0 is class 0, the third band");
        assert_eq!(dst[3], 255, "255 is class 31: 257 clamps");
    }

    #[test_case]
    fn band_offsets_saturate_rather_than_wrap() {
        let src = [255u8, 0];
        let mut dst = [0u8; 2];
        let offs: SaoOffsets = [0, 7, 0, 0, 0];
        band_filter(&mut dst, 2, &src, 2, &offs, 31, 1, 1, 8);
        assert_eq!(dst[0], 255, "must clamp, not wrap to 6");
        let offs: SaoOffsets = [0, -7, 0, 0, 0];
        band_filter(&mut dst[1..], 1, &src[1..], 1, &offs, 0, 1, 1, 8);
        assert_eq!(dst[1], 0);
    }

    /// A flat region is category 0 in every direction, and category 0 has no
    /// offset — so SAO must leave it exactly alone whatever the offsets say.
    /// This is the check that catches indexing the offsets directly instead of
    /// through `EDGE_IDX`.
    #[test_case]
    fn edge_offset_leaves_a_flat_region_untouched() {
        let src = vec![100u8; 6 * 6];
        for eo in 0..4 {
            let mut dst = vec![0u8; 4 * 4];
            // Entry 0 is the specification's always-zero category. The other
            // four are large, so reaching any of them would be obvious.
            let offs: SaoOffsets = [0, 7, -7, 7, -7];
            let b = Borders { left: true, right: true, above: true, below: true };
            edge_filter(&mut dst, 4, &src, 6, 7, &offs, eo, 4, 4, &b, 8);
            assert!(dst.iter().all(|&v| v == 100), "eo {eo}: {dst:?}");
        }
    }

    /// A local minimum is a valley (category 1), a maximum is a peak
    /// (category 4). Getting the remap backwards brightens what should darken.
    #[test_case]
    fn edge_offset_classifies_peaks_and_valleys() {
        // 3x1 region inside a 5x3 source: low, high, low along the horizontal.
        let mut src = vec![100u8; 5 * 3];
        src[5 + 1] = 90; // valley
        src[5 + 2] = 110; // peak
        src[5 + 3] = 100; // flat between two 100s? neighbours are 110 and 100
        let offs: SaoOffsets = [0, 1, 2, 3, 4];
        let mut dst = vec![0u8; 3];
        let b = Borders { left: true, right: true, above: true, below: true };
        edge_filter(&mut dst, 3, &src, 5, 5 + 1, &offs, EO_HORIZONTAL, 3, 1, &b, 8);
        // x=0: 90 vs left 100 and right 110 -> below both -> valley -> offs[1]
        assert_eq!(dst[0], 90 + 1);
        // x=1: 110 vs 90 and 100 -> above both -> peak -> offs[4]
        assert_eq!(dst[1], 110 + 4);
        // x=2: 100 vs 110 and 100 -> below one, equal the other -> half-valley
        assert_eq!(dst[2], 100 + 2);
    }

    /// The four directions really consult different neighbours — a vertical
    /// ridge is a peak horizontally and flat vertically.
    #[test_case]
    fn the_four_edge_classes_look_in_different_directions() {
        // A vertical bright line down the middle of a 3x3 region.
        let mut src = vec![100u8; 5 * 5];
        for y in 0..5 {
            src[y * 5 + 2] = 150;
        }
        let offs: SaoOffsets = [0, 0, 0, 0, 9]; // only the peak category
        let b = Borders { left: true, right: true, above: true, below: true };

        let mut dst = vec![0u8; 9];
        edge_filter(&mut dst, 3, &src, 5, 5 + 1, &offs, EO_HORIZONTAL, 3, 3, &b, 8);
        for y in 0..3 {
            assert_eq!(dst[y * 3 + 1], 159, "horizontal: the line is a peak");
        }

        let mut dst = vec![0u8; 9];
        edge_filter(&mut dst, 3, &src, 5, 5 + 1, &offs, EO_VERTICAL, 3, 3, &b, 8);
        for y in 0..3 {
            assert_eq!(dst[y * 3 + 1], 150, "vertical: the line is flat along itself");
        }
    }

    /// Samples whose classifier would step outside an unavailable border are
    /// not filtered at all — they keep whatever `dst` already held, which in
    /// the decoder is the deblocked sample.
    #[test_case]
    fn unavailable_borders_exclude_their_samples_from_filtering() {
        let mut src = vec![100u8; 5 * 5];
        src[2 * 5 + 2] = 150;
        let offs: SaoOffsets = [0, 5, 5, 5, 5];
        let mut dst = vec![7u8; 9]; // sentinel
        let b = Borders { left: false, right: true, above: false, below: true };
        edge_filter(&mut dst, 3, &src, 5, 5 + 1, &offs, EO_DIAG_135, 3, 3, &b, 8);
        // Row 0 and column 0 need the above/left neighbours, so they are skipped.
        for x in 0..3 {
            assert_eq!(dst[x], 7, "row 0 should be untouched");
        }
        for y in 0..3 {
            assert_eq!(dst[y * 3], 7, "column 0 should be untouched");
        }
        assert_ne!(dst[1 * 3 + 1], 7, "the interior must still be filtered");
    }

    /// Classification must read the *source*, never the partly-written output.
    /// Filtering in place makes each sample's category depend on its
    /// predecessor's offset — a directional bias that no single-sample test
    /// catches, so this one uses a ramp where in-place and out-of-place
    /// genuinely differ.
    #[test_case]
    fn classification_reads_unfiltered_neighbours() {
        // A staircase: each sample above its left neighbour and below its right.
        let mut src = vec![0u8; 5 * 3];
        for x in 0..5 {
            src[5 + x] = (100 + x * 4) as u8;
        }
        let offs: SaoOffsets = [0, 0, 20, 0, 0]; // only the half-valley category
        let b = Borders { left: true, right: true, above: true, below: true };
        let mut dst = vec![0u8; 3];
        edge_filter(&mut dst, 3, &src, 5, 5 + 1, &offs, EO_HORIZONTAL, 3, 1, &b, 8);
        // Every interior sample of a monotone ramp is above one neighbour and
        // below the other: category 0, no change. If the filter read its own
        // output, the +20 on one sample would flip the next one's class.
        assert_eq!(dst, alloc::vec![104, 108, 112]);
    }
}
