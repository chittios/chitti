//! f64 math helpers for no_std SBR/PS (oxideav uses f64 throughout).
//!
//! On host builds with `std` (e.g. `tools/h264diff`), inherent `f64` methods
//! take precedence over these trait methods. In the kernel (`no_std`) the
//! trait methods supply `sin`/`cos`/`exp2`/… via pure-Rust approximations.

/// Extension trait so call sites can keep oxideav's `.sin()` / `.exp2()` style.
#[allow(dead_code)]
pub trait F64Ext: Sized {
    fn sin(self) -> f64;
    fn cos(self) -> f64;
    fn sin_cos(self) -> (f64, f64);
    fn exp2(self) -> f64;
    fn log2(self) -> f64;
    fn ln(self) -> f64;
    fn sqrt(self) -> f64;
    fn floor(self) -> f64;
    fn ceil(self) -> f64;
    fn trunc(self) -> f64;
    fn abs(self) -> f64;
    fn powf(self, exp: f64) -> f64;
    fn powi(self, n: i32) -> f64;
    fn atan(self) -> f64;
    fn atan2(self, other: f64) -> f64;
    fn acos(self) -> f64;
}

impl F64Ext for f64 {
    #[inline]
    fn sin(self) -> f64 {
        sin(self)
    }
    #[inline]
    fn cos(self) -> f64 {
        cos(self)
    }
    #[inline]
    fn sin_cos(self) -> (f64, f64) {
        (sin(self), cos(self))
    }
    #[inline]
    fn exp2(self) -> f64 {
        exp2(self)
    }
    #[inline]
    fn log2(self) -> f64 {
        log2(self)
    }
    #[inline]
    fn ln(self) -> f64 {
        // ln(x) = log2(x) * ln(2)
        log2(self) * core::f64::consts::LN_2
    }
    #[inline]
    fn sqrt(self) -> f64 {
        sqrt(self)
    }
    #[inline]
    fn floor(self) -> f64 {
        floor(self)
    }
    #[inline]
    fn ceil(self) -> f64 {
        ceil(self)
    }
    #[inline]
    fn trunc(self) -> f64 {
        trunc(self)
    }
    #[inline]
    fn abs(self) -> f64 {
        if self < 0.0 {
            -self
        } else {
            self
        }
    }
    #[inline]
    fn powf(self, exp: f64) -> f64 {
        powf(self, exp)
    }
    #[inline]
    fn powi(self, n: i32) -> f64 {
        powi(self, n)
    }
    #[inline]
    fn atan(self) -> f64 {
        atan(self)
    }
    #[inline]
    fn atan2(self, other: f64) -> f64 {
        atan2(self, other)
    }
    #[inline]
    fn acos(self) -> f64 {
        acos(self)
    }
}

#[inline]
pub fn floor(x: f64) -> f64 {
    let i = x as i64;
    let f = i as f64;
    if x >= 0.0 || f == x {
        f
    } else {
        f - 1.0
    }
}

#[inline]
pub fn ceil(x: f64) -> f64 {
    let i = x as i64;
    let f = i as f64;
    if x <= 0.0 || f == x {
        f
    } else {
        f + 1.0
    }
}

#[inline]
pub fn trunc(x: f64) -> f64 {
    (x as i64) as f64
}

#[inline]
pub fn sqrt(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    // Initial guess via bit hack on f32, then Newton polish in f64.
    let y = x as f32;
    let i = y.to_bits();
    let guess = f32::from_bits(0x5f37_5a86 - (i >> 1));
    let mut r = guess as f64;
    r = 0.5 * (r + x / r);
    r = 0.5 * (r + x / r);
    r = 0.5 * (r + x / r);
    r = 0.5 * (r + x / r);
    r
}

/// Range-reduce to [-π, π] then Taylor sin.
pub fn sin(mut x: f64) -> f64 {
    // Reduce modulo 2π
    const TWO_PI: f64 = 6.283185307179586;
    const PI: f64 = 3.141592653589793;
    x = x - TWO_PI * floor(x / TWO_PI + 0.5);
    // fold to [-π/2, π/2]
    if x > PI / 2.0 {
        x = PI - x;
    } else if x < -PI / 2.0 {
        x = -PI - x;
    }
    let x2 = x * x;
    // Taylor: x - x^3/3! + x^5/5! - x^7/7! + x^9/9! - x^11/11!
    x * (1.0
        - x2 * (1.0 / 6.0
            - x2 * (1.0 / 120.0
                - x2 * (1.0 / 5040.0
                    - x2 * (1.0 / 362880.0 - x2 * (1.0 / 39916800.0))))))
}

pub fn cos(x: f64) -> f64 {
    sin(x + core::f64::consts::FRAC_PI_2)
}

pub fn exp2(x: f64) -> f64 {
    if x > 1023.0 {
        return f64::INFINITY;
    }
    if x < -1074.0 {
        return 0.0;
    }
    let n = floor(x);
    let f = x - n;
    // exp2(f) for f in [0,1) via polynomial (minimax-ish)
    let p = 1.0
        + f * (0.6931471805599453
            + f * (0.2402265069591007
                + f * (0.05550410866482158
                    + f * (0.009618129107628477
                        + f * (0.001333355814642844
                            + f * 0.0001540353039338161)))));
    // scale by 2^n via frexp/ldexp bit manipulation
    ldexp(p, n as i32)
}

fn ldexp(mut x: f64, exp: i32) -> f64 {
    if x == 0.0 || !is_finite(x) {
        return x;
    }
    // Split large exponents to avoid intermediate overflow.
    let mut e = exp;
    while e > 1023 {
        x *= f64::from_bits(((1023i32 + 1023) as u64) << 52); // 2^1023
        e -= 1023;
    }
    while e < -1022 {
        x *= f64::from_bits(((1023i32 - 1022) as u64) << 52); // 2^-1022
        e += 1022;
    }
    // 2^e via bit exponent of 1.0
    let bits = ((1023i32 + e) as u64) << 52;
    x * f64::from_bits(bits)
}

pub fn log2(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if !is_finite(x) {
        return x;
    }
    // frexp: x = m * 2^e, m in [0.5, 1)
    let bits = x.to_bits();
    let mut e = ((bits >> 52) & 0x7ff) as i32 - 1023;
    let mut m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
    // m in [1, 2); map to [0.5, 1)
    if m >= 1.0 {
        // already [1,2)
    }
    // Use ln via atanh series on (m-1)/(m+1) then / ln2, but simpler:
    // reduce m to [sqrt(2)/2, sqrt(2)]
    const SQRT2: f64 = 1.4142135623730951;
    if m > SQRT2 {
        m *= 0.5;
        e += 1;
    }
    // log2(m) for m near 1: use ln(m)/ln(2)
    let t = (m - 1.0) / (m + 1.0);
    let t2 = t * t;
    let mut s = t;
    let mut p = t;
    for d in [3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0] {
        p *= t2;
        s += p / d;
    }
    let ln = 2.0 * s;
    e as f64 + ln * core::f64::consts::LOG2_E
}

pub fn powf(base: f64, exp: f64) -> f64 {
    if base == 0.0 {
        return if exp > 0.0 { 0.0 } else { f64::INFINITY };
    }
    if base < 0.0 {
        // SBR never needs negative bases.
        return 0.0;
    }
    exp2(exp * log2(base))
}

pub fn powi(mut base: f64, mut n: i32) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n < 0 {
        base = 1.0 / base;
        // avoid overflow on i32::MIN
        if n == i32::MIN {
            return base * powi(base, i32::MAX);
        }
        n = -n;
    }
    let mut acc = 1.0;
    let mut b = base;
    let mut e = n as u32;
    while e > 0 {
        if e & 1 != 0 {
            acc *= b;
        }
        b *= b;
        e >>= 1;
    }
    acc
}

/// atan via range reduction + Padé.
pub fn atan(x: f64) -> f64 {
    let ax = if x < 0.0 { -x } else { x };
    let (a, inv) = if ax > 1.0 { (1.0 / ax, true) } else { (ax, false) };
    let a2 = a * a;
    // atan series for |a|<=1
    let mut s = a;
    let mut p = a;
    for (k, d) in [3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0]
        .iter()
        .enumerate()
    {
        p *= -a2;
        s += p / d;
        let _ = k;
    }
    let r = if inv {
        core::f64::consts::FRAC_PI_2 - s
    } else {
        s
    };
    if x < 0.0 {
        -r
    } else {
        r
    }
}

pub fn acos(x: f64) -> f64 {
    // acos(x) = atan2(sqrt(1-x^2), x) ≈ π/2 - asin(x)
    if x >= 1.0 {
        return 0.0;
    }
    if x <= -1.0 {
        return core::f64::consts::PI;
    }
    core::f64::consts::FRAC_PI_2 - atan(x / sqrt(1.0 - x * x).max(1e-30))
}

/// Two-argument arctangent.
pub fn atan2(y: f64, x: f64) -> f64 {
    if x > 0.0 {
        atan(y / x)
    } else if x < 0.0 && y >= 0.0 {
        atan(y / x) + core::f64::consts::PI
    } else if x < 0.0 && y < 0.0 {
        atan(y / x) - core::f64::consts::PI
    } else if x == 0.0 && y > 0.0 {
        core::f64::consts::FRAC_PI_2
    } else if x == 0.0 && y < 0.0 {
        -core::f64::consts::FRAC_PI_2
    } else {
        0.0
    }
}

/// Finite check without relying on std.
#[inline]
pub fn is_finite(x: f64) -> bool {
    x == x && x != f64::INFINITY && x != f64::NEG_INFINITY
}
