// Ported from oxideav-aac (MIT) — Copyright (c) 2026 Karpelès Lab Inc.
// See THIRDPARTY-LICENSES.md.

//! SBR frequency band tables — ISO/IEC 14496-3 §4.6.18.3.2.
//!
//! Spectral Band Replication groups the QMF subbands in frequency by a
//! family of *frequency band tables*. Everything is derived from one
//! **master** table `fMaster`, which is in turn fixed by two QMF subband
//! boundaries — the low boundary `k0` and the high boundary `k2` — and
//! the header data elements `bs_freq_scale` / `bs_alter_scale`.
//!
//! This module implements the *static* (header-only) half of the band
//! setup, i.e. everything that does **not** depend on the §4.6.18.6 QMF
//! patching / high-frequency-generation back-end:
//!
//! * [`k0`] — §4.6.18.3.2.1 low boundary `k0 = startMin +
//!   offset(bs_start_freq)`, with the per-`FsSBR` `offset` table and the
//!   `startMin = NINT(c · 128 / FsSBR)` thresholds.
//! * [`k2`] — §4.6.18.3.2.1 high boundary, including the
//!   `bs_stop_freq < 14` `stopDkSort` accumulation path and the
//!   `bs_stop_freq == 14 / 15` `min(64, 2·k0)` / `min(64, 3·k0)`
//!   shortcuts.
//! * [`master_table`] — §4.6.18.3.2.1 `fMaster` (Figure 4.39 for
//!   `bs_freq_scale == 0`, Figure 4.40 for `bs_freq_scale > 0`).
//! * [`HiLoTables::derive`] — §4.6.18.3.2.2 `fTableHigh`, `fTableLow`,
//!   `fTableNoise`, plus the `M` / `k_x` outputs every later SBR stage
//!   keys off.
//!
//! ## Scope
//!
//! * The §4.6.18.3.2.3 limiter band table `fTableLim` is **not** here:
//!   for `bs_limiter_bands > 0` it consumes the `patchBorders` /
//!   `patchNumSubbands` produced by §4.6.18.6, which needs the QMF
//!   patching back-end this crate does not have yet. The
//!   `bs_limiter_bands == 0` single-band case
//!   (`{fTableLow(0), fTableLow(NLow)}`) is trivially derivable from
//!   [`HiLoTables`] and is left to the limiter pass.
//! * The actual envelope decode, noise-floor decode, and QMF synthesis
//!   are downstream of these tables.
//!
//! ## Operators
//!
//! The spec's `INT()` truncates toward zero and `NINT()` rounds to the
//! nearest integer with halves away from zero (ISO/IEC 14496-3 §4.6.18,
//! reusing the §4 `INT` / `NINT` definitions). The arguments here are
//! always non-negative, so `INT` is a plain floor and the `NINT` helper
//! adds `0.5` before truncating.

use alloc::vec::Vec;
use alloc::vec;
use super::Result;

/// §4.6.18.3.2.1 nearest-integer operator (`NINT`): round to the nearest
/// integer, halves away from zero. All call sites in this module pass a
/// finite, non-negative argument.
use super::math::F64Ext;
#[inline]
fn nint(x: f64) -> i32 {
    // Halves away from zero: for x >= 0 this is floor(x + 0.5); the sign
    // branch keeps the helper correct for any finite input.
    if x >= 0.0 {
        (x + 0.5).floor() as i32
    } else {
        (x - 0.5).ceil() as i32
    }
}

/// §4 `INT` operator: truncation toward zero. The arguments in this
/// module are always non-negative, so this is a plain `floor`.
#[inline]
fn int_trunc(x: f64) -> i32 {
    x.trunc() as i32
}

/// The `offset(bs_start_freq)` row for an `FsSBR` value, per the
/// §4.6.18.3.2.1 `offset` table. Returns `None` for an `FsSBR` outside
/// the tabulated set (the spec only defines rows for the standard SBR
/// internal sample rates).
fn offset_row(fs_sbr: u32) -> Option<&'static [i32; 16]> {
    // FsSBR is twice the core sample rate; the table is keyed by the
    // SBR internal rate directly.
    const OFF_16: [i32; 16] = [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7];
    const OFF_22: [i32; 16] = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13];
    const OFF_24: [i32; 16] = [-5, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 16];
    const OFF_32: [i32; 16] = [-6, -4, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 16];
    const OFF_44: [i32; 16] = [-4, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 16, 20];
    const OFF_64: [i32; 16] = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 16, 20, 24];

    match fs_sbr {
        16000 => Some(&OFF_16),
        22050 => Some(&OFF_22),
        24000 => Some(&OFF_24),
        32000 => Some(&OFF_32),
        // `44100 <= FsSBR <= 64000` shares one row.
        44100 | 48000 | 64000 => Some(&OFF_44),
        // `FsSBR > 64000`.
        88200 | 96000 | 128000 | 176400 | 192000 => Some(&OFF_64),
        _ => None,
    }
}

/// §4.6.18.3.2.1 `startMin = NINT(c · 128 / FsSBR)`, with the three
/// `c ∈ {3000, 4000, 5000}` bands keyed by `FsSBR`.
fn start_min(fs_sbr: u32) -> i32 {
    let fs = fs_sbr as f64;
    let c = if fs_sbr < 32000 {
        3000.0
    } else if fs_sbr < 64000 {
        4000.0
    } else {
        5000.0
    };
    nint(c * 128.0 / fs)
}

/// §4.6.18.3.2.1 `stopMin = NINT(c · 128 / FsSBR)`, with the three
/// `c ∈ {6000, 8000, 10000}` bands keyed by `FsSBR`.
fn stop_min(fs_sbr: u32) -> i32 {
    let fs = fs_sbr as f64;
    let c = if fs_sbr < 32000 {
        6000.0
    } else if fs_sbr < 64000 {
        8000.0
    } else {
        10000.0
    };
    nint(c * 128.0 / fs)
}

/// §4.6.18.3.2.1 low boundary `k0`.
///
/// `k0 = startMin + offset(bs_start_freq)`. `bs_start_freq` is a 4-bit
/// header field (`0 ..= 15`); `fs_sbr` must be one of the tabulated SBR
/// internal sample rates. Returns [`"aac/sbr: freq band invalid"`] for an
/// out-of-range `bs_start_freq` or an unsupported `fs_sbr`.
pub fn k0(fs_sbr: u32, bs_start_freq: u8) -> Result<i32> {
    let row = offset_row(fs_sbr).ok_or("aac/sbr: freq band invalid")?;
    let idx = bs_start_freq as usize;
    if idx >= row.len() {
        return Err("aac/sbr: freq band invalid");
    }
    Ok(start_min(fs_sbr) + row[idx])
}

/// §4.6.18.3.2.1 high boundary `k2`.
///
/// For `0 <= bs_stop_freq < 14` this is
/// `min(64, stopMin + Σ_{i<bs_stop_freq} stopDkSort(i))`, where
/// `stopDk(p) = NINT(stopMin · (64/stopMin)^((p+1)/13)) − NINT(stopMin ·
/// (64/stopMin)^(p/13))` for `0 <= p <= 12` and `stopDkSort` is `stopDk`
/// sorted ascending. `bs_stop_freq == 14` gives `min(64, 2·k0)` and
/// `== 15` gives `min(64, 3·k0)`.
pub fn k2(fs_sbr: u32, bs_stop_freq: u8, k0_val: i32) -> Result<i32> {
    if bs_stop_freq > 15 {
        return Err("aac/sbr: freq band invalid");
    }
    let val = match bs_stop_freq {
        14 => (2 * k0_val).min(64),
        15 => (3 * k0_val).min(64),
        _ => {
            let stop_min_v = stop_min(fs_sbr);
            if stop_min_v <= 0 {
                return Err("aac/sbr: freq band invalid");
            }
            let ratio = 64.0 / stop_min_v as f64;
            // stopDk(p), 0 <= p <= 12 -> 13 entries.
            let mut stop_dk = [0i32; 13];
            for (p, slot) in stop_dk.iter_mut().enumerate() {
                let hi = nint(stop_min_v as f64 * ratio.powf((p as f64 + 1.0) / 13.0));
                let lo = nint(stop_min_v as f64 * ratio.powf(p as f64 / 13.0));
                *slot = hi - lo;
            }
            stop_dk.sort_unstable();
            // stopMin + Σ_{i=0}^{bs_stop_freq-1} stopDkSort(i).
            let mut acc = stop_min_v;
            for &dk in stop_dk.iter().take(bs_stop_freq as usize) {
                acc += dk;
            }
            acc.min(64)
        }
    };
    Ok(val)
}

/// §4.6.18.3.2.1 master frequency band table `fMaster`.
///
/// Implements Figure 4.39 (`bs_freq_scale == 0`) and Figure 4.40
/// (`bs_freq_scale > 0`). The returned vector is `fMaster(0..=NMaster)`,
/// so `NMaster == len() - 1`. `fMaster` is only defined for `k2 > k0`;
/// `numBands > 0` and the §4.6.18.3.6 `vDk > 0` requirements are checked.
///
/// * `bs_freq_scale ∈ {0, 1, 2, 3}` (0 = no warping/linear,
///   1/2/3 select `bands ∈ {12, 10, 8}`).
/// * `bs_alter_scale ∈ {0, 1}`.
pub fn master_table(
    k0_val: i32,
    k2_val: i32,
    bs_freq_scale: u8,
    bs_alter_scale: bool,
) -> Result<Vec<i32>> {
    if k2_val <= k0_val || bs_freq_scale > 3 {
        return Err("aac/sbr: freq band invalid");
    }

    if bs_freq_scale == 0 {
        master_linear(k0_val, k2_val, bs_alter_scale)
    } else {
        master_warped(k0_val, k2_val, bs_freq_scale, bs_alter_scale)
    }
}

/// Figure 4.39 — `fMaster` for `bs_freq_scale == 0`.
fn master_linear(k0_val: i32, k2_val: i32, bs_alter_scale: bool) -> Result<Vec<i32>> {
    let dk;
    let num_bands;
    if !bs_alter_scale {
        dk = 1;
        // numBands = 2 * INT( (k2 - k0) / (dk * 2) )
        num_bands = 2 * int_trunc((k2_val - k0_val) as f64 / (dk as f64 * 2.0));
    } else {
        dk = 2;
        // numBands = 2 * NINT( (k2 - k0) / (dk * 2) )
        num_bands = 2 * nint((k2_val - k0_val) as f64 / (dk as f64 * 2.0));
    }
    if num_bands <= 0 {
        return Err("aac/sbr: freq band invalid");
    }
    let num_bands = num_bands as usize;

    let mut v_dk = vec![dk; num_bands];
    let k2_achieved = k0_val + num_bands as i32 * dk;
    let mut k2_diff = k2_val - k2_achieved;

    if k2_diff != 0 {
        // incr / k start, then walk while k2Diff != 0.
        let (incr, mut k): (i32, isize) = if k2_diff < 0 {
            (1, 0)
        } else {
            (-1, num_bands as isize - 1)
        };
        while k2_diff != 0 {
            v_dk[k as usize] -= incr;
            k += incr as isize;
            k2_diff += incr;
        }
    }

    // fMaster(0) = k0; fMaster(k) = fMaster(k-1) + vDk[k-1].
    let mut f_master = Vec::with_capacity(num_bands + 1);
    f_master.push(k0_val);
    for &d in &v_dk {
        // §4.6.18.3.6: numBands > 0 is checked above; the away-from-zero
        // walk above can drive a vDk entry to 0 only on malformed input.
        if d <= 0 {
            return Err("aac/sbr: freq band invalid");
        }
        let next = *f_master.last().unwrap() + d;
        f_master.push(next);
    }
    Ok(f_master)
}

/// Figure 4.40 — `fMaster` for `bs_freq_scale > 0`.
fn master_warped(
    k0_val: i32,
    k2_val: i32,
    bs_freq_scale: u8,
    bs_alter_scale: bool,
) -> Result<Vec<i32>> {
    // temp1 = {12, 10, 8}; bands = temp1[bs_freq_scale - 1].
    let bands = [12.0, 10.0, 8.0][(bs_freq_scale - 1) as usize];
    // temp2 = {1.0, 1.3}; warp = temp2[bs_alter_scale].
    let warp = if bs_alter_scale { 1.3 } else { 1.0 };

    let two_regions;
    let k1;
    if (k2_val as f64) / (k0_val as f64) > 2.2449 {
        two_regions = true;
        k1 = 2 * k0_val;
    } else {
        two_regions = false;
        k1 = k2_val;
    }

    // Lower region.
    let v_k0 = warped_region(k0_val, k1, bands, 1.0)?;
    let num_bands0 = v_k0.len() - 1;

    if !two_regions {
        return Ok(v_k0);
    }

    // Upper region with warping. The §4.6.18.3.6 "min(vDk1) < max(vDk0)"
    // smoothing step is part of warped_region_upper.
    let max_v_dk0 = max_step(&v_k0);
    let v_k1 = warped_region_upper(k1, k2_val, bands, warp, max_v_dk0)?;
    let num_bands1 = v_k1.len() - 1;

    // fMaster: vk0[0..=numBands0] then vk1[1..=numBands1].
    let mut f_master = Vec::with_capacity(num_bands0 + num_bands1 + 1);
    f_master.extend_from_slice(&v_k0);
    f_master.extend_from_slice(&v_k1[1..]);
    Ok(f_master)
}

/// Largest forward step `vDk[k] = vk[k+1] - vk[k]` of a `vk` vector.
fn max_step(v_k: &[i32]) -> i32 {
    v_k.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0)
}

/// Figure 4.40 lower-region builder: produces `vk0` (or, for the
/// `twoRegions == 0` case, the whole `fMaster`).
///
/// `numBands0 = 2 * NINT( bands * log(k1/k0) / (2 * log(2) * warp) )`
/// (the lower region always passes `warp = 1`), then
/// `vDk0[k] = NINT(k0 * (k1/k0)^((k+1)/numBands0)) − NINT(k0 *
/// (k1/k0)^(k/numBands0))`, sorted ascending, cumulatively summed from
/// `k0`.
fn warped_region(k_lo: i32, k_hi: i32, bands: f64, warp: f64) -> Result<Vec<i32>> {
    let ratio = k_hi as f64 / k_lo as f64;
    let num_bands = 2 * nint(bands * ratio.ln() / (2.0 * 2.0_f64.ln() * warp));
    if num_bands <= 0 {
        return Err("aac/sbr: freq band invalid");
    }
    let num_bands = num_bands as usize;

    let mut v_dk = vec![0i32; num_bands];
    for (k, slot) in v_dk.iter_mut().enumerate() {
        let hi = nint(k_lo as f64 * ratio.powf((k as f64 + 1.0) / num_bands as f64));
        let lo = nint(k_lo as f64 * ratio.powf(k as f64 / num_bands as f64));
        *slot = hi - lo;
    }
    v_dk.sort_unstable();

    let mut v_k = Vec::with_capacity(num_bands + 1);
    v_k.push(k_lo);
    for &d in &v_dk {
        // §4.6.18.3.6: vDk0(i) > 0 ∀ i.
        if d <= 0 {
            return Err("aac/sbr: freq band invalid");
        }
        let next = *v_k.last().unwrap() + d;
        v_k.push(next);
    }
    Ok(v_k)
}

/// Figure 4.40 upper-region builder with the `min(vDk1) < max(vDk0)`
/// smoothing branch.
fn warped_region_upper(
    k1: i32,
    k2_val: i32,
    bands: f64,
    warp: f64,
    max_v_dk0: i32,
) -> Result<Vec<i32>> {
    let ratio = k2_val as f64 / k1 as f64;
    // numBands1 = 2 * NINT(bands * log(k2/k1) / (2 * log(2) * warp))
    let num_bands1 = 2 * nint(bands * ratio.ln() / (2.0 * 2.0_f64.ln() * warp));
    if num_bands1 <= 0 {
        return Err("aac/sbr: freq band invalid");
    }
    let num_bands1 = num_bands1 as usize;

    let mut v_dk1 = vec![0i32; num_bands1];
    for (k, slot) in v_dk1.iter_mut().enumerate() {
        let hi = nint(k1 as f64 * ratio.powf((k as f64 + 1.0) / num_bands1 as f64));
        let lo = nint(k1 as f64 * ratio.powf(k as f64 / num_bands1 as f64));
        *slot = hi - lo;
    }

    // if min(vDk1) < max(vDk0): sort, then redistribute `change` from the
    // largest to the smallest entry (capped at half the spread).
    if v_dk1.iter().copied().min().unwrap_or(0) < max_v_dk0 {
        v_dk1.sort_unstable();
        let mut change = max_v_dk0 - v_dk1[0];
        let half = int_trunc((v_dk1[num_bands1 - 1] - v_dk1[0]) as f64 / 2.0);
        if change > half {
            change = half;
        }
        v_dk1[0] += change;
        v_dk1[num_bands1 - 1] -= change;
    }
    v_dk1.sort_unstable();

    let mut v_k1 = Vec::with_capacity(num_bands1 + 1);
    v_k1.push(k1);
    for &d in &v_dk1 {
        // §4.6.18.3.6: vDk1(i) > 0 ∀ i.
        if d <= 0 {
            return Err("aac/sbr: freq band invalid");
        }
        let next = *v_k1.last().unwrap() + d;
        v_k1.push(next);
    }
    Ok(v_k1)
}

/// §4.6.18.3.2.2 derived high / low / noise frequency band tables, plus
/// the `M` (number of QMF subbands covered by SBR) and `k_x` (first SBR
/// subband) outputs that the envelope / noise / patching stages key off.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HiLoTables {
    /// `fTableHigh(0..=NHigh)` — high-resolution envelope band borders.
    pub f_table_high: Vec<i32>,
    /// `fTableLow(0..=NLow)` — low-resolution envelope band borders.
    pub f_table_low: Vec<i32>,
    /// `fTableNoise(0..=NQ)` — noise-floor band borders.
    pub f_table_noise: Vec<i32>,
    /// `M = fTableHigh(NHigh) − fTableHigh(0)` — number of QMF subbands
    /// covered by SBR.
    pub m: i32,
    /// `k_x = fTableHigh(0)` — index of the first QMF subband in the SBR
    /// range.
    pub k_x: i32,
}

impl HiLoTables {
    /// `NHigh = len(fTableHigh) - 1`.
    #[inline]
    pub fn n_high(&self) -> usize {
        self.f_table_high.len() - 1
    }

    /// `NLow = len(fTableLow) - 1`.
    #[inline]
    pub fn n_low(&self) -> usize {
        self.f_table_low.len() - 1
    }

    /// `NQ = len(fTableNoise) - 1`.
    #[inline]
    pub fn n_q(&self) -> usize {
        self.f_table_noise.len() - 1
    }

    /// §4.6.18.3.2.2 derive `fTableHigh` / `fTableLow` / `fTableNoise`
    /// from a master table.
    ///
    /// `f_master` is `fMaster(0..=NMaster)` (i.e. [`master_table`]'s
    /// output). `bs_xover_band` must satisfy `bs_xover_band < NMaster`
    /// (§4.6.18.3.6). `bs_noise_bands ∈ {0, 1, 2, 3}`.
    pub fn derive(f_master: &[i32], bs_xover_band: u8, bs_noise_bands: u8) -> Result<Self> {
        if f_master.len() < 2 || bs_noise_bands > 3 {
            return Err("aac/sbr: freq band invalid");
        }
        let n_master = f_master.len() - 1;
        let xover = bs_xover_band as usize;
        // bs_xover_band < NMaster (§4.6.18.3.6).
        if xover >= n_master {
            return Err("aac/sbr: freq band invalid");
        }

        // NHigh = NMaster - bs_xover_band.
        let n_high = n_master - xover;
        // fTableHigh(k) = fMaster(k + bs_xover_band), 0 <= k <= NHigh.
        let f_table_high: Vec<i32> = f_master[xover..=n_master].to_vec();
        debug_assert_eq!(f_table_high.len(), n_high + 1);

        // M = fTableHigh(NHigh) - fTableHigh(0); k_x = fTableHigh(0).
        let k_x = f_table_high[0];
        let m = f_table_high[n_high] - k_x;

        // NLow = INT(NHigh/2) + (NHigh - 2*INT(NHigh/2)).
        let half = n_high / 2;
        let n_low = half + (n_high - 2 * half);

        // fTableLow(k) = fTableHigh(i(k)):
        //   i(0) = 0; i(k) = 2*k - ((1 - (-1)^NHigh)/2)  for k != 0.
        let parity = (1 - if n_high % 2 == 0 { 1 } else { -1 }) / 2; // 0 if NHigh even, 1 if odd
        let mut f_table_low = Vec::with_capacity(n_low + 1);
        for k in 0..=n_low {
            let i_k = if k == 0 {
                0
            } else {
                (2 * k as isize - parity as isize) as usize
            };
            f_table_low.push(*f_table_high.get(i_k).ok_or("aac/sbr: freq band invalid")?);
        }

        // NQ = max(1, NINT(bs_noise_bands * log2(k2/k_x))), where
        // k2 == fTableLow(NLow) (the high boundary of the SBR range).
        let k2_range = f_table_low[n_low];
        let n_q = if bs_noise_bands == 0 {
            1usize
        } else {
            let val = nint(bs_noise_bands as f64 * ((k2_range as f64 / k_x as f64).log2()));
            val.max(1) as usize
        };

        // fTableNoise(0) = fTableLow(0); for k != 0:
        //   i(k) = i(k-1) + INT((NLow - i(k-1)) / (NQ + 1 - k)).
        let mut f_table_noise = Vec::with_capacity(n_q + 1);
        let mut i_prev: usize = 0;
        f_table_noise.push(f_table_low[0]);
        for k in 1..=n_q {
            let denom = (n_q + 1 - k) as f64;
            let step = int_trunc((n_low - i_prev) as f64 / denom);
            i_prev += step as usize;
            f_table_noise.push(*f_table_low.get(i_prev).ok_or("aac/sbr: freq band invalid")?);
        }

        Ok(HiLoTables {
            f_table_high,
            f_table_low,
            f_table_noise,
            m,
            k_x,
        })
    }
}
