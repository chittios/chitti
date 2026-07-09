//! N-point Inverse MDCT via a complex FFT of size N/2.
//!
//! Adapted from Symphonia's `dsp::mdct` (MPL-2.0). See THIRDPARTY-LICENSES.md.

use alloc::vec::Vec;

#[derive(Clone, Copy, Default)]
struct Complex {
    re: f32,
    im: f32,
}

impl Complex {
    #[inline]
    fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    #[inline]
    fn conj(self) -> Self {
        Self { re: self.re, im: -self.im }
    }

    #[inline]
    fn mul(self, o: Self) -> Self {
        Self {
            re: self.re * o.re - self.im * o.im,
            im: self.re * o.im + self.im * o.re,
        }
    }
}

fn cosf(x: f32) -> f32 {
    crate::sound::mel::cosf_pub(x)
}
fn sinf(x: f32) -> f32 {
    crate::sound::mel::cosf_pub(x - core::f32::consts::FRAC_PI_2)
}
fn sqrtf(x: f32) -> f32 {
    crate::cortex::tensor::libm_sqrtf(x)
}

/// In-place radix-2 Cooley–Tukey FFT.
fn fft_inplace(buf: &mut [Complex]) {
    let n = buf.len();
    debug_assert!(n.is_power_of_two());
    // Bit-reverse
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            buf.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * core::f32::consts::PI / len as f32;
        let wlen = Complex::new(cosf(ang), sinf(ang));
        let mut i = 0;
        while i < n {
            let mut w = Complex::new(1.0, 0.0);
            for k in 0..len / 2 {
                let u = buf[i + k];
                let v = buf[i + k + len / 2].mul(w);
                buf[i + k] = Complex::new(u.re + v.re, u.im + v.im);
                buf[i + k + len / 2] = Complex::new(u.re - v.re, u.im - v.im);
                w = w.mul(wlen);
            }
            i += len;
        }
        len <<= 1;
    }
}

/// Inverse MDCT of size `n` (spectral length); output is 2n samples.
pub struct Imdct {
    n: usize,
    scratch: Vec<Complex>,
    twiddle: Vec<Complex>,
}

impl Imdct {
    /// `n` = number of spectral samples (power of two). `scale` multiplies the
    /// twiddles (Symphonia uses `1/2048` for long, `1/256` for short).
    pub fn new_scaled(n: usize, scale: f64) -> Self {
        assert!(n.is_power_of_two());
        let n2 = n / 2;
        let alpha = 1.0f32 / 8.0 + if scale.is_sign_positive() { 0.0 } else { n2 as f32 };
        let pi_n = core::f32::consts::PI / n as f32;
        let sqrt_scale = sqrtf(scale.abs() as f32);
        let mut twiddle = Vec::with_capacity(n2);
        for k in 0..n2 {
            let theta = pi_n * (alpha + k as f32);
            let re = sqrt_scale * cosf(theta);
            let im = sqrt_scale * sinf(theta);
            twiddle.push(Complex::new(re, im));
        }
        Self {
            n,
            scratch: alloc::vec![Complex::default(); n2],
            twiddle,
        }
    }

    /// `spec.len() == n`, `out.len() == 2*n`.
    pub fn imdct(&mut self, spec: &[f32], out: &mut [f32]) {
        let n = self.n;
        let n2 = n >> 1;
        let n4 = n >> 2;
        debug_assert_eq!(spec.len(), n);
        debug_assert_eq!(out.len(), 2 * n);

        for (i, (&w, t)) in self.twiddle.iter().zip(self.scratch.iter_mut()).enumerate() {
            let even = spec[i * 2];
            let odd = -spec[n - 1 - i * 2];
            let re = odd * w.im - even * w.re;
            let im = odd * w.re + even * w.im;
            *t = Complex::new(re, im);
        }

        fft_inplace(&mut self.scratch);

        let (vec0, rest) = out.split_at_mut(n2);
        let (vec1, rest) = rest.split_at_mut(n2);
        let (vec2, vec3) = rest.split_at_mut(n2);

        for (i, (x, &w)) in self.scratch[..n4].iter().zip(self.twiddle[..n4].iter()).enumerate() {
            let val = w.mul(x.conj());
            let fi = 2 * i;
            let ri = n2 - 1 - 2 * i;
            vec0[ri] = -val.im;
            vec1[fi] = val.im;
            vec2[ri] = val.re;
            vec3[fi] = val.re;
        }
        for (i, (x, &w)) in self.scratch[n4..].iter().zip(self.twiddle[n4..].iter()).enumerate() {
            let val = w.mul(x.conj());
            let fi = 2 * i;
            let ri = n2 - 1 - 2 * i;
            vec0[fi] = -val.re;
            vec1[ri] = val.re;
            vec2[fi] = val.im;
            vec3[ri] = val.im;
        }
        let _ = sqrtf; // silence if unused under some cfgs
    }
}
