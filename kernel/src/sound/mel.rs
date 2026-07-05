//! **Log-mel spectrogram frontend** for the parakeet STT model — a bare-metal
//! reimplementation of NeMo's `FilterbankFeatures`, numerically matched to the
//! Python featurizer that transcribes the model's own test clips correctly.
//!
//! Pipeline (16 kHz mono S16 → `[80, T]` features):
//!   preemphasis(0.97) → 25 ms Hann frames / 10 ms hop → 512-pt real FFT →
//!   power spectrum → 80 slaney mel filters → `log(x + 2^-24)` →
//!   per-feature (per-mel-bin) mean/std normalisation.
//!
//! The 80×257 slaney filterbank is baked in (`mel_fb_80x257.bin`); the Hann
//! window and radix-2 FFT are computed here.

use alloc::vec;
use alloc::vec::Vec;

const SR: usize = 16000;
const N_FFT: usize = 512;
const WIN: usize = 400; // 25 ms
const HOP: usize = 160; // 10 ms
const N_MELS: usize = 80;
const N_BINS: usize = N_FFT / 2 + 1; // 257
const PREEMPH: f32 = 0.97;

/// The slaney mel filterbank, row-major `[80][257]`, as little-endian f32.
static MEL_FB: &[u8] = include_bytes!("testdata/mel_fb_80x257.bin");

fn fb(mel: usize, bin: usize) -> f32 {
    let o = (mel * N_BINS + bin) * 4;
    f32::from_le_bytes([MEL_FB[o], MEL_FB[o + 1], MEL_FB[o + 2], MEL_FB[o + 3]])
}

/// Accurate `cos` for FFT twiddles: reduce to the octant `[-π/4, π/4]` (a
/// naive `[-π,π]` reduction leaves a Taylor cosine ~0.026 off near ±π, which
/// showed up as ~4 % mel-power error). `k = round(x/(π/2))` picks the quadrant.
fn cosf(x: f32) -> f32 {
    use core::f32::consts::FRAC_PI_2;
    let k = floorf(x / FRAC_PI_2 + 0.5) as i32;
    let r = x - k as f32 * FRAC_PI_2;
    let (c, s) = cos_sin_small(r);
    match k.rem_euclid(4) {
        0 => c,
        1 => -s,
        2 => -c,
        _ => s,
    }
}
fn sinf(x: f32) -> f32 {
    cosf(x - core::f32::consts::FRAC_PI_2)
}
/// Accurate `cos` exposed for the ONNX executor's trig ops.
pub fn cosf_pub(x: f32) -> f32 {
    cosf(x)
}
/// `(cos r, sin r)` for `|r| <= π/4` via short Taylor series (err < 1e-7).
fn cos_sin_small(r: f32) -> (f32, f32) {
    let r2 = r * r;
    let c = 1.0 - r2 * (0.5 - r2 * (1.0 / 24.0 - r2 / 720.0));
    let s = r * (1.0 - r2 * (1.0 / 6.0 - r2 * (1.0 / 120.0 - r2 / 5040.0)));
    (c, s)
}
fn floorf(x: f32) -> f32 {
    let t = x as i64 as f32;
    if t > x {
        t - 1.0
    } else {
        t
    }
}
fn logf(x: f32) -> f32 {
    crate::onnx::exec::logf_pub(x)
}

/// In-place iterative radix-2 Cooley–Tukey FFT (`re`/`im`, length a power of 2).
/// Twiddles come from precomputed `cos_t[i]=cos(2πi/n)`, `sin_t[i]=sin(2πi/n)`
/// tables (indexed, not iterated) so no rotation error accumulates across the
/// 512-point transform.
fn fft(re: &mut [f32], im: &mut [f32], cos_t: &[f32], sin_t: &[f32]) {
    let n = re.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let step = n / len; // table stride for this stage
        let mut i = 0;
        while i < n {
            for k in 0..len / 2 {
                let idx = k * step;
                // angle = -2π k/len  →  (cos, -sin) from the +angle tables.
                let wr = cos_t[idx];
                let wi = -sin_t[idx];
                let a = i + k;
                let b = i + k + len / 2;
                let tr = re[b] * wr - im[b] * wi;
                let ti = re[b] * wi + im[b] * wr;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// Build the `cos(2πi/n)` / `sin(2πi/n)` twiddle tables for `n` (power of 2).
fn twiddles(n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut c = vec![0f32; n];
    let mut s = vec![0f32; n];
    for i in 0..n {
        let ang = 2.0 * core::f32::consts::PI * i as f32 / n as f32;
        c[i] = cosf(ang);
        s[i] = sinf(ang);
    }
    (c, s)
}

/// Compute `[80, T]` log-mel features from 16 kHz mono PCM (i16), NeMo layout.
/// `T = 1 + (len - WIN) / HOP`. Empty if the clip is shorter than one window.
pub fn features(pcm: &[i16]) -> Vec<Vec<f32>> {
    if pcm.len() < WIN {
        return Vec::new();
    }
    // Preemphasis: y[0]=x[0], y[n]=x[n]-0.97 x[n-1].
    let mut sig = vec![0f32; pcm.len()];
    sig[0] = pcm[0] as f32 / 32768.0;
    for n in 1..pcm.len() {
        sig[n] = pcm[n] as f32 / 32768.0 - PREEMPH * (pcm[n - 1] as f32 / 32768.0);
    }
    // Periodic Hann window.
    let mut hann = [0f32; WIN];
    for (i, h) in hann.iter_mut().enumerate() {
        *h = 0.5 - 0.5 * cosf(2.0 * core::f32::consts::PI * i as f32 / WIN as f32);
    }
    let nframes = 1 + (sig.len() - WIN) / HOP;
    let (cos_t, sin_t) = twiddles(N_FFT);
    // [T][80] log-mel, then normalize per mel bin across T.
    let mut out: Vec<Vec<f32>> = Vec::with_capacity(nframes);
    let mut re = [0f32; N_FFT];
    let mut im = [0f32; N_FFT];
    for f in 0..nframes {
        let start = f * HOP;
        for (i, r) in re.iter_mut().enumerate() {
            *r = if i < WIN { sig[start + i] * hann[i] } else { 0.0 };
        }
        im.iter_mut().for_each(|x| *x = 0.0);
        fft(&mut re, &mut im, &cos_t, &sin_t);
        // Power spectrum over the first N_BINS.
        let mut power = [0f32; N_BINS];
        for (b, p) in power.iter_mut().enumerate() {
            *p = re[b] * re[b] + im[b] * im[b];
        }
        let mut row = vec![0f32; N_MELS];
        for (m, cell) in row.iter_mut().enumerate() {
            let mut acc = 0f32;
            for b in 0..N_BINS {
                acc += power[b] * fb(m, b);
            }
            *cell = logf(acc + LOG_GUARD);
        }
        out.push(row);
    }
    // Per-feature normalization: for each mel bin, subtract mean / divide std.
    for m in 0..N_MELS {
        let mut mean = 0f32;
        for row in &out {
            mean += row[m];
        }
        mean /= nframes as f32;
        let mut var = 0f32;
        for row in &out {
            let d = row[m] - mean;
            var += d * d;
        }
        let std = crate::onnx::exec::sqrtf_pub(var / nframes as f32) + 1e-5;
        for row in &mut out {
            row[m] = (row[m] - mean) / std;
        }
    }
    // Transpose to [80][T] (the model's `audio_signal` layout is [1,80,T]).
    let mut feat = vec![vec![0f32; nframes]; N_MELS];
    for (t, row) in out.iter().enumerate() {
        for m in 0..N_MELS {
            feat[m][t] = row[m];
        }
    }
    feat
}

const LOG_GUARD: f32 = 5.9604645e-8; // 2^-24, NeMo log_zero_guard_value

#[cfg(test)]
mod tests {
    use super::*;
    include!("testdata/ref_logmel.rs");

    fn lcg_frame(seed: u32, n: usize) -> Vec<f32> {
        let mut v = seed;
        (0..n)
            .map(|_| {
                v = v.wrapping_mul(1664525).wrapping_add(1013904223);
                ((v >> 8) & 0xffff) as f32 / 65536.0 - 0.5
            })
            .collect()
    }

    /// FFT + slaney filterbank + log on one Hann-windowed LCG frame must match
    /// the validated NeMo/librosa reference (no preemph / no normalization).
    #[test_case]
    fn logmel_matches_reference() {
        let sig = lcg_frame(12345, WIN);
        let mut hann = [0f32; WIN];
        for (i, h) in hann.iter_mut().enumerate() {
            *h = 0.5 - 0.5 * cosf(2.0 * core::f32::consts::PI * i as f32 / WIN as f32);
        }
        let mut re = [0f32; N_FFT];
        let mut im = [0f32; N_FFT];
        for (i, r) in re.iter_mut().enumerate() {
            *r = if i < WIN { sig[i] * hann[i] } else { 0.0 };
        }
        let (cos_t, sin_t) = twiddles(N_FFT);
        fft(&mut re, &mut im, &cos_t, &sin_t);
        let mut maxerr = 0f32;
        for m in 0..N_MELS {
            let mut acc = 0f32;
            for b in 0..N_BINS {
                acc += (re[b] * re[b] + im[b] * im[b]) * fb(m, b);
            }
            let v = logf(acc + LOG_GUARD);
            let e = (v - REF_LOGMEL[m]).abs();
            if e > maxerr {
                maxerr = e;
            }
        }
        crate::serial_println!("mel: max abs err vs NeMo ref = {}", maxerr);
        assert!(maxerr < 2e-2, "log-mel error {maxerr} too high");
    }
}
