//! Token sampling (`CHITTI_OS_HANDOFF.md` Phase 3): seeded, temperature-
//! controlled, and grammar-constrained. Everything is deterministic given
//! the seed -- temperature 0 is exact greedy argmax, and temperature > 0
//! draws from the softmax distribution using a seeded PRNG whose sequence
//! is reproducible run-to-run. The grammar hook masks disallowed tokens
//! *before* sampling, so output can be forced into valid shapes (the
//! foundation Phase 4's tool-call grammar builds on).

/// A small, fast, fully deterministic PRNG (SplitMix64). Seeded and logged
/// so any sampled run is reproducible -- the determinism the phase requires.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// Uniform `f32` in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        // Top 24 bits -> mantissa precision.
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }
}

/// A decoding grammar: decides which tokens may come next and advances its
/// own state as tokens are accepted. A GBNF compiler (Phase 4) will produce
/// implementations of this; Phase 3 only needs the interface plus a trivial
/// allow-list for testing.
pub trait Grammar {
    fn allows(&self, token: usize) -> bool;
    fn accept(&mut self, token: usize);
}

/// Grammar that permits only an explicit set of token ids (used to test
/// that constrained decoding never emits a disallowed token).
pub struct AllowList<'a> {
    pub allowed: &'a [usize],
}

impl Grammar for AllowList<'_> {
    fn allows(&self, token: usize) -> bool {
        self.allowed.contains(&token)
    }
    fn accept(&mut self, _token: usize) {}
}

/// Greedy argmax over the (possibly grammar-masked) logits.
fn argmax_allowed(logits: &[f32], grammar: Option<&dyn Grammar>) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if let Some(g) = grammar {
            if !g.allows(i) {
                continue;
            }
        }
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

/// Sample one token from `logits`.
///
/// - `temperature <= 0`: exact greedy argmax (the deterministic path used
///   by the reference-parity gate).
/// - `temperature > 0`: scale logits by `1/temperature`, softmax, and draw
///   with `rng`.
/// - `grammar`: disallowed tokens are excluded (both from the argmax and
///   from the sampling distribution).
///
/// `logits` is modified in place (temperature scaling / softmax).
pub fn sample(logits: &mut [f32], temperature: f32, rng: &mut Rng, grammar: Option<&dyn Grammar>) -> usize {
    if let Some(g) = grammar {
        for (i, v) in logits.iter_mut().enumerate() {
            if !g.allows(i) {
                *v = f32::NEG_INFINITY;
            }
        }
    }

    if temperature <= 0.0 {
        return argmax_allowed(logits, grammar);
    }

    let inv_t = 1.0 / temperature;
    for v in logits.iter_mut() {
        *v *= inv_t;
    }
    super::tensor::softmax(logits);

    // Inverse-CDF sample. `-inf` logits became 0 probability in softmax, so
    // masked tokens can never be drawn.
    let r = rng.next_f32();
    let mut cumulative = 0.0f32;
    for (i, &p) in logits.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return i;
        }
    }
    // Floating-point slack: fall back to the last non-zero-probability token.
    logits.iter().rposition(|&p| p > 0.0).unwrap_or(logits.len() - 1)
}

/// Nucleus sampling as Qwen ships it: temperature + `top_k` + `top_p`. Pure
/// temperature over a ~248 K-token vocab draws from the long tail and makes
/// these models degenerate (repeated punctuation / loops); Qwen's own guidance
/// is `temp 0.7, top_k 20, top_p 0.8`, and this restores that. We keep only the
/// `top_k` highest-probability tokens, then within those the smallest prefix
/// whose cumulative probability reaches `top_p` (always at least one token),
/// renormalize, and inverse-CDF draw. `temperature <= 0` stays exact greedy.
///
/// `logits` is modified in place. Grammar-masked tokens are excluded first.
pub fn sample_topk_topp(
    logits: &mut [f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    rng: &mut Rng,
    grammar: Option<&dyn Grammar>,
) -> usize {
    if let Some(g) = grammar {
        for (i, v) in logits.iter_mut().enumerate() {
            if !g.allows(i) {
                *v = f32::NEG_INFINITY;
            }
        }
    }
    if temperature <= 0.0 {
        return argmax_allowed(logits, grammar);
    }

    let inv_t = 1.0 / temperature;
    for v in logits.iter_mut() {
        *v *= inv_t;
    }
    super::tensor::softmax(logits);

    // Partial selection of the `top_k` highest-probability tokens: repeatedly
    // pull the current max. k is tiny (~20) vs the vocab, so O(n*k) is cheap and
    // avoids sorting the whole distribution.
    let k = top_k.max(1).min(logits.len());
    let mut picked: alloc::vec::Vec<(usize, f32)> = alloc::vec::Vec::with_capacity(k);
    let mut taken = alloc::vec::Vec::new(); // indices already pulled
    for _ in 0..k {
        let mut best_i = usize::MAX;
        let mut best_p = -1.0f32;
        for (i, &p) in logits.iter().enumerate() {
            if p > best_p && !taken.contains(&i) {
                best_p = p;
                best_i = i;
            }
        }
        if best_i == usize::MAX || best_p <= 0.0 {
            break;
        }
        taken.push(best_i);
        picked.push((best_i, best_p));
    }
    if picked.is_empty() {
        return argmax_allowed(logits, grammar);
    }

    // top_p: keep the smallest prefix (already sorted desc) reaching `top_p`.
    let mut cum = 0.0f32;
    let mut keep = picked.len();
    for (n, &(_, p)) in picked.iter().enumerate() {
        cum += p;
        if cum >= top_p {
            keep = n + 1;
            break;
        }
    }
    let kept = &picked[..keep.max(1)];

    // Renormalize over the kept set and inverse-CDF draw.
    let total: f32 = kept.iter().map(|&(_, p)| p).sum();
    let r = rng.next_f32() * total;
    let mut c = 0.0f32;
    for &(i, p) in kept {
        c += p;
        if r < c {
            return i;
        }
    }
    kept[kept.len() - 1].0
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test_case]
    fn rng_is_deterministic_for_a_seed() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        // A different seed must diverge.
        let mut c = Rng::new(43);
        assert_ne!(Rng::new(42).next_u64(), c.next_u64());
    }

    #[test_case]
    fn rng_f32_in_unit_interval() {
        let mut r = Rng::new(7);
        for _ in 0..10_000 {
            let x = r.next_f32();
            assert!((0.0..1.0).contains(&x), "rng f32 out of range: {x}");
        }
    }

    #[test_case]
    fn temperature_zero_is_greedy_argmax() {
        let mut logits = alloc::vec![0.1f32, 3.5, -1.0, 3.4, 2.0];
        let mut rng = Rng::new(1);
        assert_eq!(sample(&mut logits, 0.0, &mut rng, None), 1);
    }

    #[test_case]
    fn sampling_is_reproducible_for_a_seed() {
        let base = [1.0f32, 2.0, 0.5, 3.0, 2.5, 1.5];
        let mut a = base.to_vec();
        let mut b = base.to_vec();
        let x = sample(&mut a, 1.0, &mut Rng::new(123), None);
        let y = sample(&mut b, 1.0, &mut Rng::new(123), None);
        assert_eq!(x, y, "same seed must yield same sample");
    }

    #[test_case]
    fn grammar_constrains_sampling_to_allowed_tokens() {
        let allowed = [2usize, 4];
        let grammar = AllowList { allowed: &allowed };
        // Even with the largest raw logit on a disallowed token (index 3),
        // sampling at various seeds must only ever return 2 or 4.
        for seed in 0..200u64 {
            let mut logits = alloc::vec![5.0f32, 4.0, 1.0, 9.0, 1.0, 0.5];
            let picked = sample(&mut logits, 1.0, &mut Rng::new(seed), Some(&grammar));
            assert!(allowed.contains(&picked), "seed {seed}: picked disallowed token {picked}");
        }
        // And greedy (temp 0) likewise respects the grammar.
        let mut logits = alloc::vec![5.0f32, 4.0, 1.0, 9.0, 1.0, 0.5];
        let picked = sample(&mut logits, 0.0, &mut Rng::new(0), Some(&grammar));
        assert_eq!(picked, 2, "temp0 with grammar should pick the best *allowed* token");
    }

    #[test_case]
    fn sampling_distribution_tracks_probabilities() {
        // A peaked distribution should sample its dominant token most of
        // the time -- a sanity check that the inverse-CDF is wired right.
        let mut counts = [0usize; 3];
        let mut rng = Rng::new(999);
        for _ in 0..2000 {
            let mut logits = alloc::vec![0.0f32, 5.0, 0.0]; // index 1 dominates
            let picked = sample(&mut logits, 1.0, &mut rng, None);
            counts[picked] += 1;
        }
        assert!(counts[1] > counts[0] + counts[2], "dominant token should win: {counts:?}");
    }

    // Keep `Vec` import used even if the assertions above change.
    #[allow(dead_code)]
    fn _uses_vec() -> Vec<u8> {
        Vec::new()
    }
}
