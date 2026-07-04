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
pub mod tokenizer;

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
    // temp-0 decoding is deterministic, so the "seed" is fixed at 0. Hash a
    // bounded header prefix rather than the whole (possibly hundreds-of-MiB)
    // model -- it is a provenance fingerprint, logged not asserted, and this
    // keeps it cheap on both arches (and correct when the aarch64 slice is a
    // generous upper bound rather than the exact file length).
    let model_hash = model::fnv1a(&bytes[..bytes.len().min(1 << 16)]);
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
    let prompt: Vec<usize> = refcheck::PROMPT_IDS.iter().map(|&t| t as usize).collect();
    let prefill_start = crate::arch::now_ms();
    m.prefill(&prompt, 0, &mut kv, &mut state);
    let prefill_ms = crate::arch::now_ms().saturating_sub(prefill_start);
    let logits_pos = n_prompt - 1;
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

/// Result of the `Q8_0` matvec microbenchmark: total multiply-accumulates
/// performed and the wall-clock milliseconds they took, so callers can report
/// throughput (`macs / ms` is kMAC/s).
pub struct BenchResult {
    pub macs: u64,
    pub ms: u64,
    pub rows: usize,
    pub cols: usize,
    pub iters: usize,
    /// Milliseconds for the experimental int8-activation `SDOT` kernel over the
    /// same work, or `None` on arches without it (measured, not adopted).
    pub sdot_ms: Option<u64>,
    /// Aggregate relative RMS error of the SDOT result vs the f32 result
    /// (`||sdot-ref|| / ||ref||`) -- robust where individual rows are near zero,
    /// a proxy for whether int8 activations would preserve token argmax parity.
    pub sdot_rel_rms: f32,
}

/// Microbenchmark the hottest kernel (`tensor::matvec_q8_0`) in isolation, so
/// the NEON/AVX2 path can be measured without the full ~800 MiB model. Builds
/// a representative `Q8_0` weight matrix (deterministic pseudo-random quants),
/// runs the matvec `iters` times, and times it with the arch millisecond
/// clock. Same code on both arches (honours the dual-arch rule); the numbers
/// only mean anything under native execution (aarch64/HVF), not TCG.
pub fn bench_matvec() -> BenchResult {
    use tensor::{Q8_0_BLOCK_BYTES, QK};
    // Representative of the hot projections: ~1024-wide input, many rows.
    const ROWS: usize = 4096;
    const COLS: usize = 1024;
    const ITERS: usize = 200;
    let blocks = COLS / QK;
    let row_bytes = blocks * Q8_0_BLOCK_BYTES;

    // Deterministic pseudo-random Q8_0 weights (a cheap LCG) and an x vector.
    let mut w = alloc::vec![0u8; ROWS * row_bytes];
    let mut seed: u32 = 0x1234_5678;
    let mut next = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        seed
    };
    for r in 0..ROWS {
        for b in 0..blocks {
            let base = r * row_bytes + b * Q8_0_BLOCK_BYTES;
            // A small positive f16 scale (0x2000 ~= 0.125) keeps values sane.
            w[base] = 0x00;
            w[base + 1] = 0x20;
            for i in 0..QK {
                w[base + 2 + i] = (next() >> 24) as u8;
            }
        }
    }
    let mut x = alloc::vec![0.0f32; COLS];
    for (i, xi) in x.iter_mut().enumerate() {
        *xi = ((i % 17) as f32 - 8.0) * 0.1;
    }
    let mut y = alloc::vec![0.0f32; ROWS];

    let start = crate::arch::now_ms();
    for _ in 0..ITERS {
        tensor::matvec_q8_0(&w, &x, &mut y, ROWS, COLS);
    }
    let ms = crate::arch::now_ms().saturating_sub(start);
    // Keep the f32 reference result to measure SDOT's numeric error against
    // (only the aarch64 SDOT path consumes it).
    #[cfg(target_arch = "aarch64")]
    let y_ref = y.clone();
    core::hint::black_box(&y);

    // Experimental int8-activation SDOT path over the same work (aarch64 only).
    // The activation is quantized once (cheap, O(cols)); the kernel is timed.
    #[cfg(target_arch = "aarch64")]
    let (sdot_ms, sdot_rel_rms) = {
        let mut xq = alloc::vec![0i8; COLS];
        let mut xs = alloc::vec![0.0f32; blocks];
        tensor::quantize_activations_q8(&x, &mut xq, &mut xs);
        let start = crate::arch::now_ms();
        for _ in 0..ITERS {
            // SAFETY: buffers sized ROWS/COLS as the kernel's contract requires.
            unsafe {
                tensor::matvec_q8_0_sdot_rows(w.as_ptr(), xq.as_ptr(), xs.as_ptr(), y.as_mut_ptr(), 0, ROWS, COLS);
            }
        }
        let d = crate::arch::now_ms().saturating_sub(start);
        let mut num = 0.0f32; // ||sdot - ref||^2
        let mut den = 0.0f32; // ||ref||^2
        for r in 0..ROWS {
            let diff = y[r] - y_ref[r];
            num += diff * diff;
            den += y_ref[r] * y_ref[r];
        }
        let rel_rms = if den > 0.0 { tensor::libm_sqrtf(num / den) } else { 0.0 };
        core::hint::black_box(&y);
        (Some(d), rel_rms)
    };
    #[cfg(not(target_arch = "aarch64"))]
    let (sdot_ms, sdot_rel_rms) = (None, 0.0);

    BenchResult {
        macs: (ROWS as u64) * (COLS as u64) * (ITERS as u64),
        ms,
        rows: ROWS,
        cols: COLS,
        iters: ITERS,
        sdot_ms,
        sdot_rel_rms,
    }
}

/// Self-check: build a small Q4_0 weight + activation, and compare the exact
/// scalar `matvec_q4_0` against the int8-activation `matvec_q4_0_sdot_rows`.
/// Returns the aggregate relative RMS error (should be ~1% -- int8 activation
/// noise; a much larger value means the SDOT kernel is buggy). aarch64 only.
#[cfg(target_arch = "aarch64")]
pub fn check_q4_0_sdot() -> f32 {
    use tensor::{Q4_0_BLOCK_BYTES, QK};
    const ROWS: usize = 512;
    const COLS: usize = 1024;
    let blocks = COLS / QK;
    let row_bytes = blocks * Q4_0_BLOCK_BYTES;
    let mut w = alloc::vec![0u8; ROWS * row_bytes];
    let mut seed: u32 = 0x9e37_79b9;
    let mut next = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        seed
    };
    for r in 0..ROWS {
        for b in 0..blocks {
            let base = r * row_bytes + b * Q4_0_BLOCK_BYTES;
            w[base] = 0x00;
            w[base + 1] = 0x20; // f16 ~0.125
            for i in 0..QK / 2 {
                w[base + 2 + i] = (next() >> 24) as u8;
            }
        }
    }
    let mut x = alloc::vec![0.0f32; COLS];
    for (i, xi) in x.iter_mut().enumerate() {
        *xi = ((i % 23) as f32 - 11.0) * 0.07;
    }
    let mut y_exact = alloc::vec![0.0f32; ROWS];
    tensor::matvec_q4_0(&w, &x, &mut y_exact, ROWS, COLS);
    let mut xq = alloc::vec![0i8; COLS];
    let mut xs = alloc::vec![0.0f32; blocks];
    tensor::quantize_activations_q8(&x, &mut xq, &mut xs);
    let mut y_sdot = alloc::vec![0.0f32; ROWS];
    // SAFETY: sizes match the kernel's contract.
    unsafe {
        tensor::matvec_q4_0_sdot_rows(w.as_ptr(), xq.as_ptr(), xs.as_ptr(), y_sdot.as_mut_ptr(), 0, ROWS, COLS);
    }
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for r in 0..ROWS {
        let d = y_sdot[r] - y_exact[r];
        num += d * d;
        den += y_exact[r] * y_exact[r];
    }
    crate::serial_println!("q4sdot> y_exact[0..3]={} {} {} y_sdot={} {} {}", y_exact[0], y_exact[1], y_exact[2], y_sdot[0], y_sdot[1], y_sdot[2]);
    if den > 0.0 {
        tensor::libm_sqrtf(num / den)
    } else {
        0.0
    }
}

/// Throughput of an end-to-end inference run, split into prompt prefill (`pp`)
/// and token generation (`tg`) -- the two numbers `llama-bench` reports, so the
/// `perf` shell builtin is directly comparable. Correctness is *not* asserted
/// here (the prompt is synthetic); this is a regression gauge, run alongside
/// `infer` (which does assert reference parity) after every change.
pub struct InferBench {
    pub n_prompt: usize,
    pub prefill_ms: u64,
    pub n_decode: usize,
    pub decode_ms: u64,
}

/// Benchmark inference throughput on a synthetic prompt of `n_prompt` tokens
/// (the reference ids, cycled) followed by `n_decode` greedy steps. Uses the
/// real model + forward pass, so it measures exactly what `infer` runs, just
/// long enough (and without the per-token parity/text work) to be a stable
/// prefill/decode throughput gauge. Returns `None` if no model is present.
pub fn bench_inference(n_prompt: usize, n_decode: usize) -> Option<InferBench> {
    let bytes = model_module()?;
    let gguf = gguf::Gguf::parse(bytes).ok()?;
    let m = model::Model::load(gguf).ok()?;
    let mut kv = m.new_cache();
    let mut state = m.new_state();

    // Synthetic prompt: cycle the reference ids to the requested length.
    let base = &refcheck::PROMPT_IDS;
    let prompt: Vec<usize> = (0..n_prompt).map(|i| base[i % base.len()] as usize).collect();

    let t0 = crate::arch::now_ms();
    m.prefill(&prompt, 0, &mut kv, &mut state);
    let prefill_ms = crate::arch::now_ms().saturating_sub(t0);

    let mut pos = n_prompt;
    let mut next = model::argmax(&state.logits);
    let t1 = crate::arch::now_ms();
    for _ in 0..n_decode {
        m.forward(next, pos, &mut kv, &mut state, true);
        pos += 1;
        next = model::argmax(&state.logits);
    }
    let decode_ms = crate::arch::now_ms().saturating_sub(t1);
    core::hint::black_box(next);

    Some(InferBench { n_prompt, prefill_ms, n_decode, decode_ms })
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
    use alloc::vec::Vec;
    // The model may be a single `.gguf` module, or split into multi-part
    // modules `model.gguf.000`, `.001`, ... — the ISO9660 4 GiB per-file limit
    // forces a large model to be split for single-ISO distribution. Collect
    // every model part and order them by path.
    let mut parts: Vec<&'static crate::limine_protocol::File> = match crate::MODULE_REQUEST.response() {
        Some(r) => r.modules().iter().copied().filter(|m| m.path_contains(".gguf")).collect(),
        None => Vec::new(),
    };
    if parts.is_empty() {
        // No Limine model module: this is an installed system booted from the
        // FAT ESP (kernel only). Read the model off the ext4 OS partition.
        return model_from_ext4();
    }
    parts.sort_by_key(|m| m.path_str());
    if parts.len() == 1 {
        // Single module: zero-copy slice into module memory (the common 0.8B
        // case, and any model that fit in one ISO file).
        return Some(parts[0].data());
    }
    // Multi-part: reassemble into one contiguous region so the GGUF parser +
    // zero-copy tensor slices see a single blob. The model is far larger than
    // the linked-list kernel heap (256 MiB), so allocate contiguous physical
    // frames via `alloc_dma` (backed by the frame allocator) and copy the parts
    // in. Costs the model size again in RAM; a segmented reader would avoid the
    // copy (REVISIT in DECISIONS.md). The 0.8B default stays single-part.
    let total: usize = parts.iter().map(|m| m.data().len()).sum();
    let (_phys, virt) = crate::mm::alloc_dma(total)?;
    let dst = virt as *mut u8;
    let mut off = 0usize;
    for m in &parts {
        let d = m.data();
        // SAFETY: `dst` covers `total` contiguous bytes; each part copied once,
        // disjointly, within bounds.
        unsafe { core::ptr::copy_nonoverlapping(d.as_ptr(), dst.add(off), d.len()) };
        off += d.len();
    }
    crate::ktrace::log_fmt(format_args!("cortex: reassembled model from {} parts ({} bytes) into contiguous frames", parts.len(), total));
    // SAFETY: `virt` maps `total` contiguous, now-initialized bytes (HHDM).
    Some(unsafe { core::slice::from_raw_parts(virt as *const u8, total) })
}

/// Read the model off the ext4 OS partition of the boot disk (the installed
/// system, which boots from the FAT ESP with no Limine model module). Finds the
/// ext4 volume, reads every `*.gguf*` file from its root in name order, and
/// concatenates them into contiguous frames — the same contiguous blob the
/// GGUF parser expects. `None` if there's no disk / ext4 / model.
#[cfg(target_arch = "x86_64")]
fn model_from_ext4() -> Option<&'static [u8]> {
    use crate::block::{ext4_read::Ext4Reader, virtio::VirtioBlk, Partition};
    use crate::fs::detect::FsType;
    use alloc::string::String;
    use alloc::vec::Vec;
    let mut dev = VirtioBlk::probe()?;
    // Find the ext-family OS partition via the GPT/FS detector.
    let vol = crate::fs::detect::probe(&mut dev)
        .into_iter()
        .find(|v| matches!(v.fs, FsType::Ext2 | FsType::Ext3 | FsType::Ext4))?;
    let mut part = Partition::new(&mut dev, vol.start_lba, vol.sectors);
    let mut r = Ext4Reader::open(&mut part)?;
    // Model parts in the ext4 root, ordered.
    let mut names: Vec<String> = r.list_root().into_iter().filter(|(n, _, d)| !d && n.contains(".gguf")).map(|(n, _, _)| n).collect();
    names.sort();
    if names.is_empty() {
        return None;
    }
    let total: usize = names.iter().map(|n| r.file_size(n).unwrap_or(0) as usize).sum();
    if total == 0 {
        return None;
    }
    let (_phys, virt) = crate::mm::alloc_dma(total)?;
    // SAFETY: `virt` maps `total` contiguous, freshly-allocated bytes.
    let dst = unsafe { core::slice::from_raw_parts_mut(virt as *mut u8, total) };
    let mut off = 0usize;
    for n in &names {
        let sz = r.file_size(n).unwrap_or(0) as usize;
        let got = r.read_root_file(n, &mut dst[off..off + sz])?;
        off += got;
    }
    crate::ktrace::log_fmt(format_args!("cortex: loaded model from ext4 partition ({} part(s), {} bytes)", names.len(), off));
    Some(&dst[..off])
}

/// Find a boot module whose path contains `needle` (x86 Limine). Used by
/// `/install` to reach the installer payload (BOOTX64.EFI, the kernel binary).
#[cfg(target_arch = "x86_64")]
pub fn find_module(needle: &str) -> Option<&'static [u8]> {
    crate::MODULE_REQUEST.response()?.modules().iter().copied().find(|m| m.path_contains(needle)).map(|m| m.data())
}

/// The individual model parts (basename, bytes), sorted — for `/install`, which
/// writes each as a separate file on the ext4 partition (Limine reassembles
/// them the same way this kernel does).
#[cfg(target_arch = "x86_64")]
pub fn model_parts() -> alloc::vec::Vec<(&'static str, &'static [u8])> {
    use alloc::vec::Vec;
    let Some(r) = crate::MODULE_REQUEST.response() else { return Vec::new() };
    let mut parts: Vec<&'static crate::limine_protocol::File> =
        r.modules().iter().copied().filter(|m| m.path_contains(".gguf")).collect();
    parts.sort_by_key(|m| m.path_str());
    parts
        .iter()
        .map(|m| {
            let p = m.path_str();
            let base = p.rsplit('/').next().unwrap_or(p);
            (base, m.data())
        })
        .collect()
}

/// aarch64 (`-kernel`/UEFI-stub): the guest-physical address the model GGUF is
/// loaded at — a handshake with `xtask`'s `Model::aarch64_addr` (`-device
/// loader`) and the UEFI stub. **One address for every model** (2 GiB, past the
/// kernel image at 0x40080000): the model occupies `[MODEL_LOAD_ADDR,
/// mm::heap_base())`, and the heap is placed at the top of discovered RAM (see
/// [`crate::mm`]) — so the model region auto-sizes to whatever RAM the platform
/// has, with no per-model layout constant to keep in sync with `-m`.
#[cfg(all(not(target_arch = "x86_64"), not(feature = "boot-limine")))]
pub const MODEL_LOAD_ADDR: usize = 0x8000_0000;

/// aarch64: the model is placed in RAM by QEMU's `-device loader` (or the UEFI
/// stub) at [`MODEL_LOAD_ADDR`]. We expose it as a slice spanning the region
/// between the model base and the heap (which the heap allocator placed at the
/// top of RAM), and only if the GGUF magic is present (so `infer` cleanly
/// reports "no model" when none was loaded). The GGUF parser reads only within
/// the actual file; the window just needs to cover it.
#[cfg(all(not(target_arch = "x86_64"), not(feature = "boot-limine")))]
pub fn model_module() -> Option<&'static [u8]> {
    // On UEFI the stub loaded the model at a firmware-chosen address and reported
    // (base, size); use that exact span. On `-kernel` the model is at the fixed
    // MODEL_LOAD_ADDR and the window runs up to the heap (top of RAM).
    let (addr, window) = match crate::arch::aarch64::mmu::uefi_model() {
        Some((base, size)) => (base, size),
        None => {
            let heap_base = crate::mm::heap_base();
            let window = heap_base.saturating_sub(MODEL_LOAD_ADDR);
            if window == 0 {
                crate::ktrace::log_fmt(format_args!(
                    "cortex: model present at {MODEL_LOAD_ADDR:#x} but no room below the heap at {heap_base:#x} -- not enough RAM"
                ));
                return None;
            }
            (MODEL_LOAD_ADDR, window)
        }
    };
    // SAFETY: `addr` is identity-mapped normal RAM (arch::aarch64::mmu); reading
    // 4 bytes to check the GGUF magic is in bounds.
    let magic = unsafe { core::slice::from_raw_parts(addr as *const u8, 4) };
    if magic != b"GGUF" {
        return None;
    }
    // SAFETY: [addr, addr + window) is mapped RAM holding the model; the GGUF
    // parser reads only the real model within it.
    Some(unsafe { core::slice::from_raw_parts(addr as *const u8, window) })
}

/// aarch64 booted via Limine (UEFI/AAVMF): the model is a Limine boot module,
/// exactly like x86. The single-part case (the 0.8B, one `.gguf` module) is a
/// zero-copy slice into module memory. Multi-part reassembly + the ext4
/// fallback both need a large contiguous DMA region (`alloc_dma`), which on
/// aarch64 is a follow-on (B2); until then a split/absent model reports None.
#[cfg(feature = "boot-limine")]
pub fn model_module() -> Option<&'static [u8]> {
    use alloc::vec::Vec;
    let response = crate::MODULE_REQUEST.response()?;
    let mut parts: Vec<&'static crate::limine_protocol::File> =
        response.modules().iter().copied().filter(|m| m.path_contains(".gguf")).collect();
    if parts.is_empty() {
        return None;
    }
    parts.sort_by_key(|m| m.path_str());
    if parts.len() == 1 {
        return Some(parts[0].data());
    }
    crate::ktrace::log_fmt(format_args!("cortex: {} model parts on aarch64/Limine -- multi-part reassembly is a B2 follow-on", parts.len()));
    None
}
