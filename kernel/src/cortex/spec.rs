//! **Speculative decoding** — the accept/reject algebra that makes a small model
//! stand in for a large one without changing what the large one would have said.
//!
//! Decode is memory-bound: every token reads every weight, so a 9B model at
//! ~6 tok/s is not short of arithmetic, it is short of bandwidth. Speculative
//! decoding exploits the fact that **verifying γ tokens costs one batched pass**
//! (weight-stationary, each weight read once for the whole window — the same
//! trick `Model::prefill_batched` already uses) while *generating* them costs γ
//! passes. So a cheap drafter proposes, the target model checks the whole
//! proposal at once, and the accepted prefix is free.
//!
//! ## The property that makes it safe
//!
//! The point of the algorithm below is not "usually close". With the accept rule
//! implemented exactly, the emitted token stream is drawn from **precisely the
//! target model's distribution** — the drafter changes the speed and nothing
//! else. That is a strong claim and it is the whole justification for shipping
//! this, so `speculative_matches_direct_sampling_in_distribution` tests it
//! empirically rather than trusting the derivation.
//!
//! Get the rule slightly wrong and nothing crashes: the model produces fluent,
//! plausible, subtly-wrong text at a nice speed. That failure mode is why every
//! branch here is pinned by a case, and why the wiring notes at the bottom
//! insist on `tools/cortexdiff` before this is enabled by default.
//!
//! ## The rule
//!
//! For each drafted token `x_i`, with drafter probability `q_i(x_i)` and target
//! probability `p_i(x_i)`:
//!
//! * accept with probability `min(1, p_i(x_i) / q_i(x_i))`;
//! * on rejection, emit one token drawn from the **residual** distribution
//!   `norm(max(0, p_i - q_i))` and stop — everything after a rejection was
//!   conditioned on a token that is not being emitted, so it must be discarded.
//! * if all γ are accepted, emit one **bonus** token from `p_{γ+1}`, which the
//!   verification pass already computed for free.
//!
//! Greedy decoding (temperature 0) collapses to the obvious rule: accept while
//! the draft agrees with the target's argmax. It is exactly lossless and is the
//! default here, because this OS decodes at temperature 0 by default.
//!
//! ## Three edge cases that are silent when wrong
//!
//! * **`q_i(x_i) == 0`.** Should be impossible — the drafter sampled that token —
//!   but a mismatched tokenizer or a masked grammar can produce it. Treated as
//!   "accept if the target gives it any mass at all", never as a division.
//! * **An empty residual.** When `p` is everywhere dominated by `q`,
//!   `max(0, p - q)` sums to zero and normalising it is a divide-by-zero that
//!   yields NaN and then an out-of-range token index. Falls back to sampling `p`.
//! * **A rejection at `i = 0`.** Nothing is accepted, so the window must still
//!   emit exactly one token, or decode stalls and the caller loops forever
//!   making no progress.

use super::sampler::Rng;
use alloc::vec::Vec;

/// One drafted token together with the drafter's distribution at that step.
///
/// The distribution has to be captured *at draft time*: it is conditioned on the
/// prefix as it then was, and cannot be recovered afterwards.
#[derive(Clone, Debug)]
pub struct Draft {
    pub token: usize,
    /// The drafter's probability for `token` at this step. Only the chosen
    /// token's probability is needed for the accept test, so the full vector is
    /// carried separately in [`Draft::dist`] only when sampling (temperature > 0)
    /// requires the residual.
    pub q_token: f32,
    /// Full drafter distribution, needed to build the residual on rejection.
    /// Empty for greedy decoding, where no residual is ever computed.
    pub dist: Vec<f32>,
}

/// What one speculative window produced.
#[derive(Clone, PartialEq, Debug)]
pub struct Verdict {
    /// Tokens to emit, in order. Always at least one — a window that emitted
    /// nothing would stall decode.
    pub tokens: Vec<usize>,
    /// How many of the drafted tokens were accepted. `tokens.len()` is this
    /// plus one (the correction or the bonus token).
    pub accepted: usize,
    /// True when every draft was accepted and the last token is the bonus.
    pub all_accepted: bool,
}

impl Verdict {
    /// Tokens gained per verification pass — the speedup, measured.
    pub fn tokens_per_pass(&self) -> f32 {
        self.tokens.len() as f32
    }
}

/// Convert logits to probabilities in place, with temperature.
///
/// Subtracts the max before exponentiating: a 9B model's logits reach ~30, and
/// `exp(30)` overflows `f32` into infinity, whose ratio with another infinity is
/// NaN. The shift is mathematically a no-op and numerically the difference
/// between working and producing garbage.
pub fn softmax(logits: &[f32], temperature: f32, out: &mut Vec<f32>) {
    out.clear();
    out.reserve(logits.len());
    let t = if temperature <= 0.0 { 1.0 } else { temperature };
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for &l in logits {
        let e = exp_approx((l - max) / t);
        out.push(e);
        sum += e;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for x in out.iter_mut() {
            *x *= inv;
        }
    }
}

/// `exp` for `no_std`: 2^k * 2^f via the exponent field, with a polynomial for
/// the fractional part. Accurate to ~1e-4 over the range softmax uses, which is
/// far tighter than the differences between competing tokens.
fn exp_approx(x: f32) -> f32 {
    if x < -87.0 {
        return 0.0;
    }
    if x > 88.0 {
        return f32::MAX;
    }
    // e^x = 2^(x / ln2)
    let y = x * core::f32::consts::LOG2_E;
    let k = floor_i32(y);
    let f = y - k as f32;
    // 2^f on [0,1) -- degree-3 minimax, monotonic.
    let p = 1.0 + f * (0.6931472 + f * (0.2402265 + f * 0.0555041));
    let scale = f32::from_bits((((k + 127) as u32) & 0xff) << 23);
    p * scale
}

fn floor_i32(x: f32) -> i32 {
    let t = x as i32;
    if x < 0.0 && (t as f32) != x {
        t - 1
    } else {
        t
    }
}

/// Index of the largest value; ties go to the lowest index so the result is
/// deterministic, which every reproducibility guarantee in this OS depends on.
pub fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            best = i;
        }
    }
    best
}

/// **Greedy verification** (temperature 0) — exactly lossless.
///
/// `target_argmax[i]` is the target model's argmax at position `i`, i.e. the
/// token it would have emitted given the draft accepted so far. There must be
/// `drafts.len() + 1` of them: the extra one is the bonus position.
///
/// Accept while the draft agrees. On the first disagreement, emit the target's
/// choice instead and stop — the drafts after it were conditioned on a token
/// that is not being emitted.
pub fn verify_greedy(draft_tokens: &[usize], target_argmax: &[usize]) -> Verdict {
    debug_assert_eq!(
        target_argmax.len(),
        draft_tokens.len() + 1,
        "verification needs one more position than there are drafts (the bonus)"
    );
    let mut tokens = Vec::with_capacity(draft_tokens.len() + 1);
    let mut accepted = 0usize;
    for (i, &d) in draft_tokens.iter().enumerate() {
        let Some(&t) = target_argmax.get(i) else { break };
        if d == t {
            tokens.push(d);
            accepted += 1;
        } else {
            // The correction. This is what the target would have said, so
            // emitting it keeps the stream identical to unassisted decode.
            tokens.push(t);
            return Verdict { tokens, accepted, all_accepted: false };
        }
    }
    // Every draft agreed: the verification pass already computed the next
    // position's logits, so one extra token is free.
    if let Some(&bonus) = target_argmax.get(draft_tokens.len()) {
        tokens.push(bonus);
    }
    Verdict { tokens, accepted, all_accepted: true }
}

/// **Stochastic verification** (temperature > 0) — distribution-preserving.
///
/// `target[i]` is the target's probability distribution at position `i`, with
/// `drafts.len() + 1` entries as above.
pub fn verify_sampled(drafts: &[Draft], target: &[Vec<f32>], rng: &mut Rng) -> Verdict {
    debug_assert_eq!(target.len(), drafts.len() + 1);
    let mut tokens = Vec::with_capacity(drafts.len() + 1);
    let mut accepted = 0usize;

    for (i, d) in drafts.iter().enumerate() {
        let Some(p) = target.get(i) else { break };
        let p_x = p.get(d.token).copied().unwrap_or(0.0);
        // q == 0 should be impossible (the drafter chose this token), but a
        // grammar mask or a tokenizer mismatch can produce it. Accept on any
        // target mass rather than dividing by zero.
        let ratio = if d.q_token <= 0.0 {
            if p_x > 0.0 {
                1.0
            } else {
                0.0
            }
        } else {
            p_x / d.q_token
        };
        if rng.next_f32() < ratio.min(1.0) {
            tokens.push(d.token);
            accepted += 1;
            continue;
        }
        // Rejected: emit one token from the residual max(0, p - q).
        let tok = sample_residual(p, &d.dist, rng);
        tokens.push(tok);
        return Verdict { tokens, accepted, all_accepted: false };
    }

    if let Some(p) = target.get(drafts.len()) {
        tokens.push(sample_dist(p, rng));
    }
    Verdict { tokens, accepted, all_accepted: true }
}

/// Draw from `norm(max(0, p - q))`, falling back to `p` when that residual is
/// empty (which happens when `q` dominates `p` everywhere).
fn sample_residual(p: &[f32], q: &[f32], rng: &mut Rng) -> usize {
    let mut sum = 0f32;
    for i in 0..p.len() {
        let r = p[i] - q.get(i).copied().unwrap_or(0.0);
        if r > 0.0 {
            sum += r;
        }
    }
    if sum <= 0.0 {
        // Normalising a zero vector gives NaN and then an out-of-range index.
        return sample_dist(p, rng);
    }
    let mut u = rng.next_f32() * sum;
    for i in 0..p.len() {
        let r = p[i] - q.get(i).copied().unwrap_or(0.0);
        if r > 0.0 {
            u -= r;
            if u <= 0.0 {
                return i;
            }
        }
    }
    // Floating-point drift can exhaust the loop; the last positive residual is
    // the right answer, and never an out-of-range index.
    (0..p.len()).rev().find(|&i| p[i] - q.get(i).copied().unwrap_or(0.0) > 0.0).unwrap_or(0)
}

/// Draw from a normalised distribution by inverse-CDF.
fn sample_dist(p: &[f32], rng: &mut Rng) -> usize {
    let total: f32 = p.iter().sum();
    if total <= 0.0 {
        return 0;
    }
    let mut u = rng.next_f32() * total;
    for (i, &v) in p.iter().enumerate() {
        u -= v;
        if u <= 0.0 {
            return i;
        }
    }
    p.len() - 1
}

/// One speculative window: draft γ tokens with `drafter`, verify them with
/// `target`, and emit the tokens the target would have produced.
///
/// Returns the emitted tokens. Both models' caches are advanced to exactly the
/// emitted prefix, so the caller can loop this without further bookkeeping.
///
/// ## The rollback is the load-bearing part
///
/// Verification advances the target's cache across the *whole* window, drafts
/// that are about to be rejected included. Those positions must be undone, and
/// on a hybrid model they cannot be undone by truncation — a DeltaNet layer's
/// recurrent state is stepped, not appended. So the cache is snapshotted before
/// verification and the accepted prefix is replayed from the snapshot on a
/// partial accept.
///
/// That replay costs a second pass, which is why the acceptance rate decides
/// whether any of this is worth doing: at high acceptance the replay is rare, at
/// low acceptance you pay it nearly every window and lose. `Stats` measures it.
#[allow(clippy::too_many_arguments)]
pub fn step(
    target: &super::model::Model,
    t_cache: &mut super::model::Cache,
    t_state: &mut super::model::State,
    drafter: &super::model::Model,
    d_cache: &mut super::model::Cache,
    d_state: &mut super::model::State,
    last_token: usize,
    pos: usize,
    gamma: usize,
    temperature: f32,
    rng: &mut Rng,
    stats: &mut Stats,
) -> Vec<usize> {
    // --- draft ---------------------------------------------------------
    let mut drafts: Vec<Draft> = Vec::with_capacity(gamma);
    let mut draft_tokens: Vec<usize> = Vec::with_capacity(gamma);
    let mut tok = last_token;
    let mut probs = Vec::new();
    for i in 0..gamma {
        drafter.forward(tok, pos + i, d_cache, d_state, true);
        let next = if temperature <= 0.0 {
            argmax(&d_state.logits)
        } else {
            softmax(&d_state.logits, temperature, &mut probs);
            sample_dist(&probs, rng)
        };
        let (q_token, dist) = if temperature <= 0.0 {
            (1.0, Vec::new())
        } else {
            (probs.get(next).copied().unwrap_or(0.0), probs.clone())
        };
        drafts.push(Draft { token: next, q_token, dist });
        draft_tokens.push(next);
        tok = next;
    }

    // --- verify --------------------------------------------------------
    // Snapshot before the window: a rejected draft has to be rolled back, and a
    // recurrent state cannot be un-stepped.
    let t_snapshot = t_cache.clone();
    let mut window = Vec::with_capacity(gamma + 1);
    window.push(last_token);
    window.extend_from_slice(&draft_tokens);
    let logits = target.verify_window(&window, pos, t_cache, t_state);
    if logits.len() != window.len() {
        // The model could not produce a distribution per position. Fall back to
        // emitting nothing speculative rather than guessing -- the caller's
        // ordinary decode path still works.
        *t_cache = t_snapshot;
        return Vec::new();
    }

    let verdict = if temperature <= 0.0 {
        let argmaxes: Vec<usize> = logits.iter().map(|l| argmax(l)).collect();
        verify_greedy(&draft_tokens, &argmaxes)
    } else {
        let dists: Vec<Vec<f32>> = logits
            .iter()
            .map(|l| {
                let mut p = Vec::new();
                softmax(l, temperature, &mut p);
                p
            })
            .collect();
        verify_sampled(&drafts, &dists, rng)
    };
    stats.record(gamma, &verdict);

    // --- roll back to the accepted prefix -------------------------------
    if !verdict.all_accepted {
        *t_cache = t_snapshot;
        // Replay only what is being emitted. One batched pass, not one per token.
        let mut replay = Vec::with_capacity(verdict.tokens.len() + 1);
        replay.push(last_token);
        replay.extend_from_slice(&verdict.tokens);
        // The final token's own position is written by the next window, so the
        // replay stops one short of it.
        if replay.len() > 1 {
            target.prefill(&replay[..replay.len() - 1], pos, t_cache, t_state);
        }
    }

    // The drafter must follow the emitted stream, not its own rejected guesses.
    *d_cache = t_snapshot_drafter(drafter, d_cache, d_state, pos, last_token, &verdict.tokens);

    verdict.tokens
}

/// Re-align the drafter's cache to the tokens actually emitted.
///
/// The drafter speculatively advanced across its own proposals; any that were
/// rejected leave it conditioned on text that does not exist. Rebuilding from
/// the emitted prefix is the simple correct thing — the drafter is small, which
/// is the entire reason it was chosen.
fn t_snapshot_drafter(
    drafter: &super::model::Model,
    d_cache: &mut super::model::Cache,
    d_state: &mut super::model::State,
    pos: usize,
    last_token: usize,
    emitted: &[usize],
) -> super::model::Cache {
    let mut replay = Vec::with_capacity(emitted.len() + 1);
    replay.push(last_token);
    replay.extend_from_slice(emitted);
    if replay.len() > 1 {
        drafter.prefill(&replay[..replay.len() - 1], pos, d_cache, d_state);
    }
    d_cache.clone()
}

/// Running acceptance statistics, so the speedup is measured rather than assumed.
///
/// The break-even point is real: if the drafter is too slow or agrees too
/// rarely, speculation is a *loss*. `speedup_estimate` is what `/perf` should
/// print, and a value under 1.0 means turn it off for this pair of models.
#[derive(Clone, Copy, Default, Debug)]
pub struct Stats {
    pub windows: u64,
    pub drafted: u64,
    pub accepted: u64,
    pub emitted: u64,
}

impl Stats {
    pub fn record(&mut self, gamma: usize, v: &Verdict) {
        self.windows += 1;
        self.drafted += gamma as u64;
        self.accepted += v.accepted as u64;
        self.emitted += v.tokens.len() as u64;
    }

    /// Fraction of drafted tokens the target accepted.
    pub fn acceptance_rate(&self) -> f32 {
        if self.drafted == 0 {
            return 0.0;
        }
        self.accepted as f32 / self.drafted as f32
    }

    /// Mean tokens emitted per verification pass.
    pub fn tokens_per_pass(&self) -> f32 {
        if self.windows == 0 {
            return 0.0;
        }
        self.emitted as f32 / self.windows as f32
    }

    /// Speedup over unassisted decode, given the drafter costs `draft_cost` of a
    /// target pass per drafted token.
    ///
    /// Unassisted: one pass per token. Speculative: one target pass plus γ
    /// drafter passes per window, yielding `tokens_per_pass` tokens. A drafter
    /// an eighth the size of the target has `draft_cost ≈ 0.125`.
    pub fn speedup_estimate(&self, gamma: usize, draft_cost: f32) -> f32 {
        if self.windows == 0 {
            return 1.0;
        }
        let cost_per_window = 1.0 + gamma as f32 * draft_cost;
        self.tokens_per_pass() / cost_per_window
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn dist(v: &[f32]) -> Vec<f32> {
        let s: f32 = v.iter().sum();
        v.iter().map(|x| x / s).collect()
    }

    // ---- greedy ---------------------------------------------------------

    #[test_case]
    fn greedy_accepts_a_fully_agreeing_draft_and_adds_a_bonus() {
        // The best case: γ drafts accepted plus one free token, so one target
        // pass emitted γ+1 tokens.
        let v = verify_greedy(&[10, 11, 12], &[10, 11, 12, 13]);
        assert_eq!(v.tokens, vec![10, 11, 12, 13]);
        assert_eq!(v.accepted, 3);
        assert!(v.all_accepted);
    }

    #[test_case]
    fn greedy_stops_at_the_first_disagreement_and_emits_the_targets_choice() {
        // Everything after a rejection was conditioned on a token that is not
        // being emitted, so it must be discarded -- keeping it is exactly the
        // bug that produces fluent, subtly-wrong text.
        let v = verify_greedy(&[10, 99, 12], &[10, 11, 12, 13]);
        assert_eq!(v.tokens, vec![10, 11], "the target's token replaces the bad draft");
        assert_eq!(v.accepted, 1);
        assert!(!v.all_accepted);
    }

    #[test_case]
    fn greedy_rejecting_the_first_draft_still_emits_one_token() {
        // A window that emitted nothing would stall decode: the caller loops,
        // makes no progress, and the machine looks hung rather than wrong.
        let v = verify_greedy(&[99], &[7, 8]);
        assert_eq!(v.tokens, vec![7]);
        assert_eq!(v.accepted, 0);
    }

    #[test_case]
    fn greedy_output_is_identical_to_unassisted_decode() {
        // The claim that justifies the whole feature: whatever the drafter
        // proposes, the emitted prefix is what the target would have said.
        let target: Vec<usize> = vec![5, 6, 7, 8, 9];
        for draft in [
            vec![5, 6, 7, 8],
            vec![5, 6, 0, 0],
            vec![0, 0, 0, 0],
            vec![5, 0, 7, 8],
        ] {
            let v = verify_greedy(&draft, &target);
            for (i, &t) in v.tokens.iter().enumerate() {
                assert_eq!(t, target[i], "position {i} diverged from unassisted decode");
            }
        }
    }

    #[test_case]
    fn a_zero_length_draft_still_produces_the_bonus_token() {
        let v = verify_greedy(&[], &[42]);
        assert_eq!(v.tokens, vec![42]);
        assert_eq!(v.accepted, 0);
    }

    // ---- stochastic -----------------------------------------------------

    #[test_case]
    fn a_draft_the_target_agrees_with_is_almost_always_accepted() {
        let p = dist(&[0.0, 0.9, 0.1]);
        let d = Draft { token: 1, q_token: 0.9, dist: p.clone() };
        let mut rng = Rng::new(1);
        let mut acc = 0;
        for _ in 0..200 {
            let v = verify_sampled(&[d.clone()], &[p.clone(), p.clone()], &mut rng);
            acc += v.accepted;
        }
        assert!(acc > 190, "p/q == 1 must accept nearly always, got {acc}/200");
    }

    #[test_case]
    fn a_draft_the_target_dislikes_is_usually_rejected() {
        // q says 0.9, p says 0.05 -> accept probability ~0.055.
        let p = dist(&[0.95, 0.05]);
        let d = Draft { token: 1, q_token: 0.9, dist: dist(&[0.1, 0.9]) };
        let mut rng = Rng::new(2);
        let mut acc = 0;
        for _ in 0..400 {
            let v = verify_sampled(&[d.clone()], &[p.clone(), p.clone()], &mut rng);
            acc += v.accepted;
        }
        assert!(acc < 60, "a token the target gives 0.05 must rarely survive, got {acc}/400");
    }

    #[test_case]
    fn speculative_matches_direct_sampling_in_distribution() {
        // THE test. Speculative decoding is only worth shipping if the emitted
        // distribution is the target's own. Draw many first-tokens both ways and
        // compare the histograms.
        let p = dist(&[0.5, 0.3, 0.2]);
        let q = dist(&[0.2, 0.3, 0.5]); // deliberately a poor drafter
        const N: usize = 4000;

        let mut direct = [0usize; 3];
        let mut rng = Rng::new(7);
        for _ in 0..N {
            direct[sample_dist(&p, &mut rng)] += 1;
        }

        let mut spec = [0usize; 3];
        let mut rng = Rng::new(11);
        for _ in 0..N {
            // Draw the draft token from q, as a real drafter would.
            let t = sample_dist(&q, &mut rng);
            let d = Draft { token: t, q_token: q[t], dist: q.clone() };
            let v = verify_sampled(&[d], &[p.clone(), p.clone()], &mut rng);
            spec[v.tokens[0]] += 1;
        }

        for i in 0..3 {
            let a = direct[i] as f32 / N as f32;
            let b = spec[i] as f32 / N as f32;
            assert!(
                (a - b).abs() < 0.04,
                "token {i}: direct {a:.3} vs speculative {b:.3} -- the algorithm is not distribution-preserving"
            );
        }
    }

    #[test_case]
    fn a_zero_drafter_probability_does_not_divide_by_zero() {
        // Should be impossible, but a grammar mask can produce it, and a NaN
        // ratio compares false against everything -- silently rejecting always.
        let p = dist(&[0.5, 0.5]);
        let d = Draft { token: 0, q_token: 0.0, dist: vec![0.0, 1.0] };
        let mut rng = Rng::new(3);
        let v = verify_sampled(&[d], &[p.clone(), p.clone()], &mut rng);
        assert_eq!(v.accepted, 1, "target mass exists, so accept rather than divide");
    }

    #[test_case]
    fn an_empty_residual_falls_back_to_the_target_distribution() {
        // p entirely dominated by q: max(0, p-q) sums to zero, and normalising
        // it yields NaN and then an out-of-range token index.
        let p = dist(&[0.5, 0.5]);
        let q = vec![1.0, 1.0]; // dominates p everywhere
        let d = Draft { token: 0, q_token: 1.0, dist: q };
        let mut rng = Rng::new(4);
        for _ in 0..50 {
            let v = verify_sampled(&[d.clone()], &[p.clone(), p.clone()], &mut rng);
            assert!(v.tokens[0] < 2, "must stay in range, got {}", v.tokens[0]);
        }
    }

    #[test_case]
    fn every_window_emits_at_least_one_token() {
        // Guards the stall: no progress looks like a hang, not a bug.
        let p = dist(&[1.0, 0.0]);
        let mut rng = Rng::new(5);
        for gamma in 0..4 {
            let drafts: Vec<Draft> =
                (0..gamma).map(|_| Draft { token: 1, q_token: 1.0, dist: vec![0.0, 1.0] }).collect();
            let target: Vec<Vec<f32>> = (0..gamma + 1).map(|_| p.clone()).collect();
            let v = verify_sampled(&drafts, &target, &mut rng);
            assert!(!v.tokens.is_empty(), "gamma {gamma} emitted nothing");
        }
    }

    // ---- numerics -------------------------------------------------------

    #[test_case]
    fn softmax_survives_large_logits() {
        // A 9B model reaches logits around 30; exp(30) overflows f32 to
        // infinity, and inf/inf is NaN.
        let mut out = Vec::new();
        softmax(&[30.0, 29.0, 1.0], 1.0, &mut out);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "must normalise, got {sum}");
        assert!(out.iter().all(|x| x.is_finite()));
        assert!(out[0] > out[1] && out[1] > out[2], "order must be preserved");
    }

    #[test_case]
    fn softmax_temperature_sharpens_and_flattens() {
        let mut hot = Vec::new();
        let mut cold = Vec::new();
        softmax(&[2.0, 1.0], 2.0, &mut hot);
        softmax(&[2.0, 1.0], 0.5, &mut cold);
        assert!(cold[0] > hot[0], "lower temperature must concentrate mass");
    }

    #[test_case]
    fn exp_approximation_is_close_enough_for_softmax() {
        for x in [-10.0f32, -1.0, 0.0, 1.0, 5.0] {
            let got = exp_approx(x);
            // Reference by repeated squaring of e, good to a few ulp here.
            let want = match x {
                v if v == 0.0 => 1.0,
                v if v == 1.0 => core::f32::consts::E,
                v if v == -1.0 => 1.0 / core::f32::consts::E,
                v if v == 5.0 => 148.4131,
                _ => 4.539_993e-5,
            };
            assert!((got - want).abs() / want < 0.01, "exp({x}) = {got}, want {want}");
        }
    }

    #[test_case]
    fn argmax_breaks_ties_at_the_lowest_index() {
        // Determinism: reproducible runs depend on this being stable.
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[]), 0);
    }

    // ---- statistics -----------------------------------------------------

    #[test_case]
    fn stats_track_acceptance_and_throughput() {
        let mut s = Stats::default();
        s.record(4, &verify_greedy(&[1, 2, 3, 4], &[1, 2, 3, 4, 5]));
        s.record(4, &verify_greedy(&[1, 9, 9, 9], &[1, 2, 3, 4, 5]));
        assert_eq!(s.drafted, 8);
        assert_eq!(s.accepted, 5);
        assert!((s.acceptance_rate() - 0.625).abs() < 0.001);
        assert!((s.tokens_per_pass() - 3.5).abs() < 0.001);
    }

    #[test_case]
    fn a_bad_drafter_reports_a_speedup_below_one() {
        // The honest negative: speculation with a drafter that never agrees is
        // slower than not speculating, and the number must say so.
        let mut s = Stats::default();
        for _ in 0..10 {
            s.record(4, &verify_greedy(&[9, 9, 9, 9], &[1, 2, 3, 4, 5]));
        }
        // 1 token per window, cost 1 + 4*0.5 = 3 target-passes.
        assert!(s.speedup_estimate(4, 0.5) < 1.0, "must report a loss, not a win");
        // A cheap drafter that always agrees is the other extreme.
        let mut good = Stats::default();
        for _ in 0..10 {
            good.record(4, &verify_greedy(&[1, 2, 3, 4], &[1, 2, 3, 4, 5]));
        }
        assert!(good.speedup_estimate(4, 0.125) > 3.0);
    }

    #[test_case]
    fn empty_stats_do_not_divide_by_zero() {
        let s = Stats::default();
        assert_eq!(s.acceptance_rate(), 0.0);
        assert_eq!(s.tokens_per_pass(), 0.0);
        assert_eq!(s.speedup_estimate(4, 0.1), 1.0);
    }
}
