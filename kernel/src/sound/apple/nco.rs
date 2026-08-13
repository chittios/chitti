//! The **NCO** — Apple's numerically-controlled oscillator, which generates the
//! audio master clock.
//!
//! I2S needs a bit clock derived from the sample rate, and 48 kHz does not
//! divide a 900 MHz reference evenly. The NCO solves that the way a fractional
//! synthesiser does: an integer divider plus two increments that dither between
//! two divisions, so the *average* rate is exact.
//!
//! Ported from m1n1 `proxyclient/m1n1/hw/nco.py`. The one part that cannot be
//! guessed is that **the divider is not written as a number**: its high bits go
//! through a Galois LFSR encoding (`poly = 0xa01`, seed `0x7ff`), so the register
//! holds the LFSR state that many cycles in, not the divider. Writing the plain
//! integer produces a valid-looking register and a clock at some unrelated
//! frequency — which sounds like audio playing at the wrong speed, not like a
//! configuration error. The table is *computed*, not transcribed.

/// Entries in the divider encoding table. The LFSR is 11-bit, so it walks 2047
/// states before repeating; the table is indexed by `div >> 2` and covers
/// dividers 2..=2049.
const TBL_LEN: usize = 2050;

/// The LFSR-encoded divider table, built at compile time.
///
/// `TBL[d]` is the state the Galois LFSR reaches after the cycle that
/// corresponds to divider `d`. m1n1 builds it by running the LFSR to exhaustion
/// and reversing the sequence; this does the same thing in a `const fn`, so
/// there is no generated file to drift and no hand-typed constants at all.
const TBL: [u16; TBL_LEN] = build_table();

const fn build_table() -> [u16; TBL_LEN] {
    let mut states = [0u16; 2047];
    let mut state: u16 = 0x7ff;
    let mut i = 0;
    // `galois_lfsr(0x7ff, 0xa01)`: 2^11 - 1 steps, shifting right and XORing the
    // polynomial's low half back in when a 1 falls off the end.
    while i < 2047 {
        state = if state & 1 != 0 { (state >> 1) ^ (0xa01 >> 1) } else { state >> 1 };
        states[i] = state;
        i += 1;
    }
    // `[0] + reversed(states)`, then `fwd[cycle + 2] = state`.
    let mut tbl = [0u16; TBL_LEN];
    tbl[2] = 0;
    let mut c = 0;
    while c < 2047 {
        tbl[c + 3] = states[2046 - c];
        c += 1;
    }
    tbl
}

/// The four registers of one NCO channel, for `fout` Hz from a `fin` Hz
/// reference. `None` when the target is out of the divider's range.
///
/// The shape is m1n1's `calc_regvals`: `[0, encoded_div, inc1, inc2]`, where
/// `inc2` is stored as a two's-complement negative in 32 bits.
pub fn calc_regvals(fin: u64, fout: u64) -> Option<[u32; 4]> {
    if fout == 0 {
        return None;
    }
    let div = 2 * fin / fout;
    let idx = (div >> 2) as usize;
    if idx >= TBL_LEN || div < 2 {
        return None;
    }
    let inc1 = 2 * fin - div * fout;
    // `inc2 = inc1 - fout` is negative by construction and is written as a
    // 32-bit two's complement (m1n1 adds 2^32).
    let inc2 = (inc1 as i64 - fout as i64) as i64;
    Some([
        0,
        ((TBL[idx] as u32) << 2) | (div as u32 & 3),
        inc1 as u32,
        (inc2 as i64 + 0x1_0000_0000) as u32,
    ])
}

/// Stride between channel register blocks within the NCO's `reg` window.
pub const CHANNEL_STRIDE: u64 = 0x4000;
/// Bit 31 of the first register enables the channel.
pub const ENABLE: u32 = 1 << 31;

#[cfg(target_arch = "aarch64")]
mod hw {
    use super::*;

    /// One NCO channel — one audio master clock.
    pub struct Nco {
        base: u64,
    }

    impl Nco {
        /// # Safety
        /// `base` must be the NCO's mapped register window and `channel` within
        /// the window the device tree sized.
        pub unsafe fn new(base: u64, size: usize, channel: u64) -> Self {
            let mapped = crate::mm::map_mmio(base, size);
            Nco { base: mapped + channel * CHANNEL_STRIDE }
        }

        fn w(&self, off: u64, v: u32) {
            // SAFETY: inside the mapped NCO window.
            unsafe {
                core::arch::asm!("str {0:w}, [{1}]", in(reg) v, in(reg) self.base + off, options(nostack))
            };
        }

        fn r(&self, off: u64) -> u32 {
            let v: u32;
            // SAFETY: inside the mapped NCO window.
            unsafe {
                core::arch::asm!("ldr {0:w}, [{1}]", out(reg) v, in(reg) self.base + off, options(nostack))
            };
            v
        }

        /// Program the channel for `fout` and enable it.
        pub fn set_rate(&self, fin: u64, fout: u64) -> bool {
            let Some(vals) = calc_regvals(fin, fout) else {
                return false;
            };
            // Disable while reprogramming: the divider and the two increments
            // are not written atomically, and a running clock would spend the
            // gap at a rate that is neither the old one nor the new one.
            self.w(0, self.r(0) & !ENABLE);
            for (i, v) in vals.iter().enumerate() {
                self.w(i as u64 * 4, *v);
            }
            self.w(0, vals[0] | ENABLE);
            true
        }

        pub fn enabled(&self) -> bool {
            self.r(0) & ENABLE != 0
        }

        /// The four programmed registers, for `/audio dump`.
        pub fn regs(&self) -> [u32; 4] {
            [self.r(0), self.r(4), self.r(8), self.r(12)]
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub use hw::Nco;

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn the_lfsr_table_is_the_one_m1n1_builds() {
        // Spot values from running m1n1's `gen_lookup_tables()`: the first few
        // entries and the one a 48 kHz stream actually uses. A table this size
        // cannot be eyeballed, and the consequence of it being subtly wrong is a
        // clock at the wrong rate rather than an error.
        assert_eq!(TBL[2], 0x000);
        assert_eq!(TBL[3], 0x7ff);
        assert_eq!(TBL[4], 0x5ff);
        assert_eq!(TBL[5], 0x1ff);
        assert_eq!(TBL[6], 0x3fe);
        assert_eq!(TBL[7], 0x7fc);
        assert_eq!(TBL[36], 0x117, "the divider a 12.288 MHz clock lands on");
    }

    #[test_case]
    fn the_48khz_master_clock_matches_the_reference_implementation() {
        // fin = 900 MHz (the `nco_ref` fixed-clock in the t8112 tree), fout =
        // 48000 * 256 = 12.288 MHz, which is what m1n1's speaker_amp.py asks
        // for. Computed independently from m1n1's python.
        let v = calc_regvals(900_000_000, 48_000 * 256).expect("in range");
        assert_eq!(v[0], 0);
        assert_eq!(v[1], 0x45e, "LFSR-encoded divider 146");
        assert_eq!(v[2], 0x5ad200, "inc1 = 2*fin - div*fout");
        assert_eq!(v[3], 0xff9f_5200, "inc2, two's complement in 32 bits");
    }

    #[test_case]
    fn the_encoded_divider_keeps_its_low_two_bits_verbatim() {
        // Only `div >> 2` goes through the LFSR; the low two bits pass through.
        // Folding them into the table index instead would quietly quantise every
        // rate to a multiple of four divider steps.
        for div in [8u64, 9, 10, 11] {
            let fout = 2 * 900_000_000 / div;
            let v = calc_regvals(900_000_000, fout).unwrap();
            assert_eq!(v[1] & 3, (div & 3) as u32, "div {div}");
            assert_eq!(v[1] >> 2, TBL[(div >> 2) as usize] as u32);
        }
    }

    #[test_case]
    fn an_impossible_rate_is_refused_rather_than_wrapped() {
        assert!(calc_regvals(900_000_000, 0).is_none());
        // A rate so low the divider leaves the table.
        assert!(calc_regvals(900_000_000, 100).is_none());
        // And one so high the divider is below 2.
        assert!(calc_regvals(900_000_000, 2_000_000_000).is_none());
    }
}
