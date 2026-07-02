//! Cortex: the CPU inference runtime (`CHITTI_OS_HANDOFF.md` Phase 3), the
//! project's highest-risk layer. Cortex turns a tiny quantized GGUF model
//! (loaded as a Limine boot module) into a deterministic, reproducible
//! token stream, entirely on the CPU with SSE2. It sits *below* the
//! determinism boundary: it only produces tokens, never side effects
//! (those are Phase 4's Synapse).
//!
//! Built strictly bottom-up: `tensor` (numeric kernels, unit-tested
//! against a NumPy reference) is the foundation the rest stands on.

pub mod batch;
pub mod gguf;
pub mod model;
pub mod refcheck;
pub mod sampler;
pub mod tensor;

#[cfg(test)]
pub mod testdata;

use alloc::string::String;
use alloc::vec::Vec;

/// Outcome of a single-stream inference run, returned so callers (the boot
/// demo, `cargo xtask ref-check`) can report pass/fail.
pub struct InferResult {
    pub prompt_final_argmax: usize,
    pub prompt_final_logit: f32,
    pub continuation: Vec<usize>,
    pub continuation_text: String,
    pub matched_reference: bool,
    /// Wall-clock milliseconds (PIT ticks) spent prefilling the prompt.
    pub prefill_ms: u64,
    /// Wall-clock milliseconds spent decoding the continuation tokens.
    pub decode_ms: u64,
    /// Number of prompt tokens prefilled and continuation tokens decoded.
    pub n_prompt: usize,
    pub n_decoded: usize,
}

/// Load the model, run the fixed reference prompt, and greedily decode
/// `refcheck::EXPECTED_CONTINUATION.len()` tokens. Logs the mandatory
/// per-inference provenance (`model hash + seed + input hash`) and compares
/// the greedy continuation to the NumPy reference (`tools/ref_forward.py`).
pub fn run_reference_inference() -> Option<InferResult> {
    let bytes = model_module()?;
    let gguf = gguf::Gguf::parse(bytes).ok()?;
    let m = model::Model::load(gguf).ok()?;

    // Provenance log: model hash, seed (greedy => 0), input hash. Greedy
    // temp-0 decoding is deterministic, so the "seed" is fixed at 0.
    let model_hash = model::fnv1a(bytes);
    let input_hash = model::fnv1a(bytemuck_ids(&refcheck::PROMPT_IDS));
    crate::ktrace::log_fmt(format_args!(
        "cortex.infer: model_hash={model_hash:#018x} seed=0 input_hash={input_hash:#018x} prompt_len={}",
        refcheck::PROMPT_IDS.len()
    ));

    let mut kv = m.new_cache();
    let mut state = m.new_state();

    // Prefill the prompt. CPU inference on this model under QEMU is slow
    // (~10-15s/token), so log progress per token -- otherwise the long
    // silent gap here looks like a hang.
    let n_prompt = refcheck::PROMPT_IDS.len();
    let mut logits_pos = 0usize;
    let prefill_start = crate::arch::now_ms();
    for (pos, &tok) in refcheck::PROMPT_IDS.iter().enumerate() {
        crate::serial_println!("cortex.infer: prefill {}/{}", pos + 1, n_prompt);
        m.forward(tok as usize, pos, &mut kv, &mut state, pos + 1 == n_prompt);
        logits_pos = pos;
    }
    let prefill_ms = crate::arch::now_ms().saturating_sub(prefill_start);
    let prompt_final_argmax = model::argmax(&state.logits);
    let prompt_final_logit = state.logits[prompt_final_argmax];

    // Greedy decode, streaming each token's text as it is produced.
    let n_gen = refcheck::EXPECTED_CONTINUATION.len();
    let mut continuation = Vec::new();
    let mut pos = logits_pos + 1;
    let mut next = prompt_final_argmax;
    crate::serial_print!("Chitti: response> ");
    let decode_start = crate::arch::now_ms();
    for _ in 0..n_gen {
        continuation.push(next);
        crate::serial_print!("{}", model::detokenize(&m, &[next]));
        m.forward(next, pos, &mut kv, &mut state, true);
        pos += 1;
        next = model::argmax(&state.logits);
    }
    let decode_ms = crate::arch::now_ms().saturating_sub(decode_start);
    crate::serial_println!("");

    let matched_reference = continuation.len() == refcheck::EXPECTED_CONTINUATION.len()
        && continuation.iter().zip(refcheck::EXPECTED_CONTINUATION.iter()).all(|(&a, &b)| a == b as usize);

    // Detokenize here (reusing the already-parsed model) so callers don't
    // pay a second ~2.4 MiB GGUF parse just to render the text.
    let continuation_text = model::detokenize(&m, &continuation);

    Some(InferResult {
        prompt_final_argmax,
        prompt_final_logit,
        continuation,
        continuation_text,
        matched_reference,
        prefill_ms,
        decode_ms,
        n_prompt,
        n_decoded: n_gen,
    })
}

fn bytemuck_ids(ids: &[u32]) -> &[u8] {
    // SAFETY: reinterpreting a `u32` slice as bytes for hashing only; the
    // pointer is valid for `len*4` bytes and `u8` has no alignment needs.
    unsafe { core::slice::from_raw_parts(ids.as_ptr() as *const u8, ids.len() * 4) }
}

/// The full Phase 3 acceptance gate, run in-kernel against the real model
/// (via `cargo xtask ref-check`): reference parity, sampler determinism,
/// KV evict+recompute reproducibility, and 2-agent continuous batching.
/// Prints one `REFCHECK:` line per check and a final `REFCHECK: ALL PASS`
/// / `ALL FAIL` the harness greps for. Returns whether everything passed.
pub fn run_acceptance() -> bool {
    let Some(bytes) = model_module() else {
        crate::serial_println!("REFCHECK: ALL FAIL (no model module)");
        return false;
    };
    let Ok(gguf) = gguf::Gguf::parse(bytes) else {
        crate::serial_println!("REFCHECK: ALL FAIL (gguf parse)");
        return false;
    };
    let Ok(m) = model::Model::load(gguf) else {
        crate::serial_println!("REFCHECK: ALL FAIL (model load)");
        return false;
    };

    let prompt: Vec<usize> = refcheck::PROMPT_IDS.iter().map(|&i| i as usize).collect();
    let reference: Vec<usize> = refcheck::EXPECTED_CONTINUATION.iter().map(|&i| i as usize).collect();
    const ACCEPT_GEN: usize = 3;

    // Greedy decode helper: prefill `prompt` then generate `n` tokens.
    let greedy = |m: &model::Model, kv: &mut model::Cache, n: usize| -> Vec<usize> {
        let mut state = m.new_state();
        for (pos, &tok) in prompt.iter().enumerate() {
            m.forward(tok, pos, kv, &mut state, pos + 1 == prompt.len());
        }
        let mut out = Vec::new();
        let mut pos = prompt.len();
        let mut next = model::argmax(&state.logits);
        for _ in 0..n {
            out.push(next);
            m.forward(next, pos, kv, &mut state, true);
            pos += 1;
            next = model::argmax(&state.logits);
        }
        out
    };

    // (a) Reference parity: greedy continuation matches the NumPy reference.
    let mut kv = m.new_cache();
    let parity_cont = greedy(&m, &mut kv, reference.len());
    let parity = parity_cont == reference;
    crate::serial_println!("REFCHECK: parity matched={} got={:?} want={:?}", parity, parity_cont, reference);

    // (b) Determinism: two sampled runs with the same seed are identical.
    let sampled = |seed: u64| -> Vec<usize> {
        let mut kv = m.new_cache();
        let mut state = m.new_state();
        for (pos, &tok) in prompt.iter().enumerate() {
            m.forward(tok, pos, &mut kv, &mut state, pos + 1 == prompt.len());
        }
        let mut rng = sampler::Rng::new(seed);
        let mut out = Vec::new();
        let mut pos = prompt.len();
        for _ in 0..ACCEPT_GEN {
            let next = sampler::sample(&mut state.logits, 0.8, &mut rng, None);
            out.push(next);
            m.forward(next, pos, &mut kv, &mut state, true);
            pos += 1;
        }
        out
    };
    let run1 = sampled(0xC0FFEE);
    let run2 = sampled(0xC0FFEE);
    let determinism = run1 == run2;
    crate::serial_println!("REFCHECK: determinism matched={} run1={:?} run2={:?}", determinism, run1, run2);

    // (c) KV evict + recompute: evicting the cache and replaying the prompt
    // reproduces the identical continuation.
    let mut kv2 = m.new_cache();
    let before = greedy(&m, &mut kv2, ACCEPT_GEN);
    kv2.evict();
    let after = greedy(&m, &mut kv2, ACCEPT_GEN);
    let evict_recompute = before == after && before == reference[..ACCEPT_GEN];
    crate::serial_println!(
        "REFCHECK: kv_evict_recompute matched={} before={:?} after={:?}",
        evict_recompute,
        before,
        after
    );

    // (d) Continuous batching: two agents advance in interleaved forward
    // passes; both reproduce the reference continuation, and the step order
    // is interleaved (alternating stream ids), not sequential.
    let mut b = batch::Batch::new(&m);
    b.add_stream(&prompt, ACCEPT_GEN);
    b.add_stream(&prompt, ACCEPT_GEN);
    b.run_greedy();
    let g0 = b.generated(0, prompt.len()).to_vec();
    let g1 = b.generated(1, prompt.len()).to_vec();
    let interleaved = b.step_order.windows(2).any(|w| w[0] != w[1]);
    let batching = g0 == reference[..ACCEPT_GEN] && g1 == reference[..ACCEPT_GEN] && interleaved;
    crate::serial_println!(
        "REFCHECK: batching matched={} stream0={:?} stream1={:?} order={:?}",
        batching,
        g0,
        g1,
        b.step_order
    );

    let all = parity && determinism && evict_recompute && batching;
    crate::serial_println!("REFCHECK: {}", if all { "ALL PASS" } else { "ALL FAIL" });
    all
}

/// Locate the `model.gguf` Limine boot module, if one was loaded.
/// Returns its raw bytes (a slice into module memory, reachable via the
/// HHDM). `None` when the image was assembled without the model (the
/// tensor unit tests don't need it).
#[cfg(target_arch = "x86_64")]
pub fn model_module() -> Option<&'static [u8]> {
    let response = crate::MODULE_REQUEST.response()?;
    response
        .modules()
        .iter()
        .find(|m| m.path_ends_with(".gguf"))
        .map(|m| m.data())
}

/// aarch64 has no Limine boot module yet, so no model is bundled: the agent
/// OS (Synapse/Persona/shell) runs, but `infer` reports no model. Loading a
/// GGUF on the `-M virt -kernel` boot path (e.g. via `-initrd` / a DTB region)
/// is future work.
#[cfg(not(target_arch = "x86_64"))]
pub fn model_module() -> Option<&'static [u8]> {
    None
}
