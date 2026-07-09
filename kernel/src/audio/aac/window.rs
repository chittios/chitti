//! AAC analysis/synthesis window generation (sine + Kaiser-Bessel derived).
//!
//! From Symphonia / NihAV (MPL-2.0). See THIRDPARTY-LICENSES.md.

/// Window types.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindowType {
    /// Simple sine window.
    Sine,
    /// Kaiser-Bessel derived window.
    KaiserBessel(f32),
}

fn sinf(x: f32) -> f32 {
    // Accurate enough for window gen (once at init).
    crate::sound::mel::cosf_pub(x - core::f32::consts::FRAC_PI_2)
}

fn sqrtf(x: f32) -> f32 {
    crate::cortex::tensor::libm_sqrtf(x)
}

/// Calculates window coefficients. Set `half` to compute only the rising half.
pub fn generate_window(mode: WindowType, scale: f32, size: usize, half: bool, dst: &mut [f32]) {
    match mode {
        WindowType::Sine => {
            let param = if half {
                core::f32::consts::PI / ((2 * size) as f32)
            } else {
                core::f32::consts::PI / (size as f32)
            };
            for n in 0..size {
                dst[n] = sinf(((n as f32) + 0.5) * param) * scale;
            }
        }
        WindowType::KaiserBessel(alpha) => {
            let dlen = if half { size as f32 } else { (size as f32) * 0.5 };
            let alpha2 = f64::from((alpha * core::f32::consts::PI / dlen) * (alpha * core::f32::consts::PI / dlen));

            let mut sum = 0.0f64;
            // cumulative bessel
            let mut kb = alloc::vec![0f64; size];
            for n in 0..size {
                let b = bessel_i0(((n * (size - n)) as f64) * alpha2);
                sum += b;
                kb[n] = sum;
            }
            sum += 1.0;
            for n in 0..size {
                dst[n] = sqrtf((kb[n] / sum) as f32);
            }
        }
    }
}

fn bessel_i0(inval: f64) -> f64 {
    let mut val: f64 = 1.0;
    for n in (1..64).rev() {
        val *= inval / f64::from(n * n);
        val += 1.0;
    }
    val
}
