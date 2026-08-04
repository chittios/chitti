//! Deterministic PRNG for the fuzzer.
//!
//! A small xorshift64* — the only requirements are reproducibility from a seed
//! and speed; the corpus does not need cryptographic quality.

pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng { state: seed.max(1) }
    }

    fn next(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Uniform in `[0, n)`.
    pub fn range(&mut self, lo: usize, hi: usize) -> usize {
        debug_assert!(lo < hi);
        let span = (hi - lo) as u64;
        lo + (self.next() % span) as usize
    }

    pub fn byte(&mut self) -> u8 {
        (self.next() & 0xff) as u8
    }
}
