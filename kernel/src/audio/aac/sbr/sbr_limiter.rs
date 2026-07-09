// Ported from oxideav-aac (MIT) — Copyright (c) 2026 Karpelès Lab Inc.
// See THIRDPARTY-LICENSES.md.

//! SBR limiter frequency band table — ISO/IEC 14496-3 §4.6.18.3.2.3 /
//! Figure 4.41.
//!
//! `fTableLim` partitions the SBR range into the bands over which the
//! §4.6.18.7.5 gain limiter averages: either exactly one band
//! (`bs_limiter_bands == 0`) or approximately 1.2 / 2 / 3 bands per
//! octave. The table is a subset of the union of `fTableLow` and the
//! §4.6.18.6 patch borders; the Figure 4.41 walk merges neighbours
//! closer than `0.49 / limBands` octaves, always preferring to keep a
//! patch border over an envelope border (both being patch borders
//! keeps both).
//!
//! ## Provenance
//!
//! The construction is the Figure 4.41 flowchart of the staged spec,
//! with the `limiterBandsPerOctave = {1.2, 2, 3}` selector. No part of
//! this implementation is derived from any external decoder.

use alloc::vec::Vec;
use alloc::vec;
use super::sbr_freq_bands::HiLoTables;
use super::Result;

/// §4.6.18.3.2.3 / Figure 4.41 — build `fTableLim`.
///
/// * `bands` — the derived frequency tables (`fTableLow`, `k_x`, `m`).
/// * `patch_borders` — the §4.6.18.6 patch borders
///   ([`super::sbr_hf_gen::Patches::borders`], starting at `k_x`).
/// * `bs_limiter_bands` — the 2-bit header field (`0..=3`).
///
/// Returns the border vector `fTableLim(0..=NL)`.
use super::math::F64Ext;
pub fn limiter_table(
    bands: &HiLoTables,
    patch_borders: &[i32],
    bs_limiter_bands: u8,
) -> Result<Vec<i32>> {
    let f_low = &bands.f_table_low;
    if f_low.len() < 2 || bs_limiter_bands > 3 {
        return Err("aac/sbr: freq band invalid");
    }

    // bs_limiter_bands == 0: one band over the whole SBR range.
    if bs_limiter_bands == 0 {
        return Ok(vec![f_low[0], f_low[f_low.len() - 1]]);
    }

    // limiterBandsPerOctave = {1.2, 2, 3}.
    let lim_bands = [1.2f64, 2.0, 3.0][usize::from(bs_limiter_bands - 1)];

    // limTable = fTableLow ∪ interior patch borders, sorted.
    let num_patches = patch_borders.len().saturating_sub(1);
    let mut lim_table: Vec<i32> = f_low.clone();
    if num_patches > 1 {
        lim_table.extend_from_slice(&patch_borders[1..num_patches]);
    }
    lim_table.sort_unstable();

    // nrLim = NLow + numPatches - 1 (the last index of limTable).
    let mut k = 1usize;
    while k < lim_table.len() {
        if lim_table[k] < 1 || lim_table[k - 1] < 1 {
            return Err("aac/sbr: freq band invalid");
        }
        let n_octaves = (f64::from(lim_table[k]) / f64::from(lim_table[k - 1])).log2();
        if n_octaves * lim_bands < 0.49 {
            if lim_table[k] == lim_table[k - 1] {
                // Duplicate border: drop one copy.
                lim_table.remove(k);
            } else if !patch_borders.contains(&lim_table[k]) {
                // The upper border is droppable (an envelope border).
                lim_table.remove(k);
            } else if !patch_borders.contains(&lim_table[k - 1]) {
                // The upper border is a patch border; drop the lower
                // envelope border instead.
                lim_table.remove(k - 1);
            } else {
                // Both are patch borders: keep both.
                k += 1;
            }
        } else {
            k += 1;
        }
    }

    Ok(lim_table)
}
