//! The kernel's **CSPRNG seeding**: one place that answers "give me bytes an
//! attacker cannot predict".
//!
//! There is exactly one entropy story in this OS and it has two inputs:
//! `RDRAND`/`RNDR` when the machine has them, and cycle-counter jitter sampled
//! across cooperative yields when it does not. [`seed_rng`] mixes both through a
//! SplitMix64 diffuser into a ChaCha20 stream.
//!
//! **Why this module exists rather than the two-line helper each caller used to
//! write.** `arch::hw_rand()` returns **0** when the facility is absent -- which
//! is the case under QEMU and HVF's default CPU models, i.e. every development
//! and CI boot. A caller that fills a buffer straight from it therefore produces
//! **all zeros** on exactly the machines it gets tested on, and real entropy only
//! on hardware nobody runs the tests against. `block::volcrypto::fill_random` did
//! precisely that, so every C4VE volume formatted under QEMU had an all-zero salt
//! and an all-zero master key. The bug is invisible: an all-zero key encrypts and
//! decrypts perfectly, the volume mounts, the passphrase works.
//!
//! So: **never draw key material from `arch::hw_rand()` directly.** Call
//! [`fill_random`], which cannot silently degrade to zeros -- [`mix`] folds in the
//! cycle counter unconditionally, and `entropy_survives_a_dead_hardware_rng` pins
//! that with `hw = 0`.
//!
//! Not audited crypto entropy. Adequate for ephemeral handshake keys, volume
//! salts and login salts on a research OS; stated plainly rather than implied.

use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

/// One SplitMix64 diffusion step: fold a hardware-random word and a cycle-counter
/// sample into `state` and finalise.
///
/// Pure and separated from the sampling so the diffusion can be tested with
/// `hw = 0` -- the case that matters, and the one a test using the real
/// `hw_rand()` cannot force on a machine that has RDRAND.
pub(crate) fn mix(state: u64, hw: u64, cycles: u64) -> u64 {
    let mut s = state ^ hw;
    s = s.wrapping_add(cycles);
    s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = s;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    z
}

/// Seed a ChaCha20 CSPRNG. Mixes several hardware-random words (`RDRAND`/`RNDR`,
/// 0 when absent) with cycle-counter samples taken across cooperative yields
/// (timing jitter), so the seed is unpredictable even with no hardware RNG.
///
/// **Yields.** This calls [`crate::sched::yield_now`] once per 8 bytes of seed, so
/// it must not be called with a `Locked` held or with interrupts off.
pub fn seed_rng() -> ChaCha20Rng {
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15 ^ crate::arch::now_ms().wrapping_mul(0xff51_afd7_ed55_8ccd);
    let mut seed = [0u8; 32];
    for chunk in seed.chunks_mut(8) {
        // Fold in a fresh hardware-random word + the live cycle counter, then
        // yield so the next counter sample reflects real scheduling jitter.
        let hw = crate::arch::hw_rand();
        state = state.wrapping_add(crate::arch::cycle_count());
        crate::sched::yield_now();
        let cycles = crate::arch::cycle_count().rotate_left(17);
        state = mix(state, hw, cycles);
        chunk.copy_from_slice(&state.to_le_bytes());
    }
    ChaCha20Rng::from_seed(seed)
}

/// Fill `buf` with cryptographically-seeded random bytes.
///
/// The one way to get key material in this kernel -- salts, master keys, nonces.
/// See the module doc for why `arch::hw_rand()` is not that way.
pub fn fill_random(buf: &mut [u8]) {
    seed_rng().fill_bytes(buf);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The regression test for the all-zero-salt bug: a drawn buffer must not be
    /// all zeros, and two draws must differ.
    ///
    /// NB this alone is **not** sufficient -- it passes under `-cpu max` (where
    /// RDRAND exists) even with the old `hw_rand()`-only implementation. The test
    /// below is the one that actually pins the fix.
    #[test_case]
    fn fill_random_is_not_all_zeros_and_two_draws_differ() {
        let mut a = [0u8; 32];
        let mut b = [0u8; 32];
        fill_random(&mut a);
        fill_random(&mut b);
        assert!(a.iter().any(|&x| x != 0), "fill_random produced an all-zero buffer");
        assert!(b.iter().any(|&x| x != 0), "fill_random produced an all-zero buffer");
        assert_ne!(a, b, "two draws from fill_random were identical");
    }

    /// With **no hardware RNG at all** (`hw = 0` -- QEMU/HVF's default CPU), the
    /// diffuser must still produce entropy from the cycle counter alone. This is
    /// the test the old implementation could not pass: it copied `hw_rand()`
    /// straight out, so `hw = 0` meant a buffer of zeros.
    #[test_case]
    fn entropy_survives_a_dead_hardware_rng() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut seen = [0u64; 8];
        for (i, slot) in seen.iter_mut().enumerate() {
            // hw = 0 throughout; only the cycle sample varies.
            state = mix(state, 0, 0x1000 + i as u64);
            *slot = state;
        }
        assert!(seen.iter().all(|&z| z != 0), "the diffuser emitted a zero word with hw = 0");
        for i in 0..seen.len() {
            for j in i + 1..seen.len() {
                assert_ne!(seen[i], seen[j], "the diffuser repeated a word with hw = 0");
            }
        }
    }

    /// A single `mix` step must depend on **both** inputs, so neither a dead
    /// hardware RNG nor a frozen cycle counter collapses it to a constant.
    #[test_case]
    fn mix_depends_on_both_the_hardware_word_and_the_cycle_sample() {
        let s = 0x0123_4567_89ab_cdefu64;
        assert_ne!(mix(s, 0, 0), mix(s, 1, 0), "mix ignored the hardware word");
        assert_ne!(mix(s, 0, 0), mix(s, 0, 1), "mix ignored the cycle sample");
        assert_ne!(mix(s, 0, 0), mix(s.wrapping_add(1), 0, 0), "mix ignored the state");
    }
}
