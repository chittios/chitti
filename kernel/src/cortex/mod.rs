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
pub mod iq_tables;
pub mod refcheck;
pub mod sampler;
pub mod tensor;
pub mod tokenizer;

use alloc::string::String;
use alloc::vec::Vec;

/// Outcome of a single-stream inference run, returned so callers (the boot
/// demo, `cargo xtask ref-check`) can report pass/fail.
pub struct InferResult {
    pub prompt_final_argmax: usize,
    pub prompt_final_logit: f32,
    pub continuation: Vec<usize>,
    pub continuation_text: String,
    /// Greedy parity vs the model's fixture (`None` = no fixture: SKIP).
    pub matched_reference: Option<bool>,
    /// Wall-clock milliseconds (PIT ticks) spent prefilling the prompt.
    pub prefill_ms: u64,
    /// Wall-clock milliseconds spent decoding the continuation tokens.
    pub decode_ms: u64,
    /// Number of prompt tokens prefilled and continuation tokens decoded.
    pub n_prompt: usize,
    pub n_decoded: usize,
}

/// A runtime-loaded model (`/model load <file>`): overrides the boot-time
/// module for every consumer of [`model_module`]. The bytes live in DMA
/// frames for the kernel's lifetime (models are far larger than the heap).
static MODEL_OVERRIDE: crate::mm::Locked<Option<&'static [u8]>> = crate::mm::Locked::new(None);
/// Cached `general.name` of the active model (see [`model_name`]).
static MODEL_NAME: crate::mm::Locked<Option<Option<String>>> = crate::mm::Locked::new(None);

/// The loaded model's display name (`general.name` from the GGUF header),
/// parsed once and cached — replaces the old compile-time per-feature model
/// strings now that any GGUF can be booted. `None` when no model is present.
pub fn model_name() -> Option<String> {
    if let Some(cached) = MODEL_NAME.with(|n| n.clone()) {
        return cached;
    }
    let name = model_module()
        .and_then(|bytes| gguf::Gguf::parse(bytes).ok().and_then(|g| g.name.map(String::from)));
    MODEL_NAME.with(|n| *n = Some(name.clone()));
    name
}

/// Load a GGUF from a mounted disk volume at runtime and make it the active
/// model. Scans every disk's FAT/ext4 volumes for `name` (a root filename or
/// FAT path), reads it into DMA frames (batched multi-sector reads with UI
/// upkeep — the perf standing rule), validates the header parses, and
/// installs it as the [`model_module`] override. Returns the parsed
/// `general.name` on success. The previous override's frames are not
/// reclaimed (models are loaded a handful of times per boot at most).
pub fn load_model_from_disk(name: &str) -> Result<Option<String>, &'static str> {
    use crate::fs::detect::FsType;
    let bare = name.trim_start_matches('/');
    for disk in 0..4usize {
        let Some(mut dev) = crate::block::probe_disk_nth(disk) else {
            continue;
        };
        for v in crate::fs::detect::probe(&mut dev) {
            let mut part = crate::block::Partition::new(&mut dev, v.start_lba, v.sectors);
            let read: Option<(usize, u64)> = match v.fs {
                FsType::Fat16 | FsType::Fat32 => {
                    let mut r = match crate::block::fat_read::FatReader::open(&mut part) {
                        Some(r) => r,
                        None => continue,
                    };
                    let Some(size) = r.file_size(name).or_else(|| r.file_size(bare)) else { continue };
                    let Some((_phys, virt)) = crate::mm::alloc_dma(size as usize) else {
                        return Err("model does not fit in free RAM");
                    };
                    // SAFETY: `virt` maps `size` contiguous, freshly-allocated bytes.
                    let dst = unsafe { core::slice::from_raw_parts_mut(virt as *mut u8, size as usize) };
                    let n = r.read_file_into(name, dst).or_else(|| r.read_file_into(bare, dst));
                    n.map(|n| (n, virt))
                }
                FsType::Ext2 | FsType::Ext3 | FsType::Ext4 => {
                    let mut r = match crate::block::ext4_read::Ext4Reader::open(&mut part) {
                        Some(r) => r,
                        None => continue,
                    };
                    let Some(size) = r.file_size(bare) else { continue };
                    let Some((_phys, virt)) = crate::mm::alloc_dma(size as usize) else {
                        return Err("model does not fit in free RAM");
                    };
                    // SAFETY: `virt` maps `size` contiguous, freshly-allocated bytes.
                    let dst = unsafe { core::slice::from_raw_parts_mut(virt as *mut u8, size as usize) };
                    r.read_root_file(bare, dst).map(|n| (n, virt))
                }
                _ => None,
            };
            if let Some((n, virt)) = read {
                // SAFETY: `virt` maps `n` contiguous, now-initialized bytes,
                // never reclaimed (leaked to 'static like the boot module).
                let bytes: &'static [u8] = unsafe { core::slice::from_raw_parts(virt as *const u8, n) };
                let g = gguf::Gguf::parse(bytes).map_err(|_| "file is not a loadable GGUF (parse failed)")?;
                let loaded = g.name.map(String::from);
                crate::ktrace::log_fmt(format_args!(
                    "cortex: runtime-loaded model '{}' ({} MiB, arch {}) from disk {}",
                    loaded.as_deref().unwrap_or(name),
                    n >> 20,
                    g.arch,
                    disk
                ));
                MODEL_OVERRIDE.with(|m| *m = Some(bytes));
                MODEL_NAME.with(|c| *c = Some(loaded.clone()));
                return Ok(loaded);
            }
        }
    }
    Err("file not found on any disk volume")
}

/// Encode the reference prompt with the loaded model's own tokenizer,
/// prepending BOS when the model asks for it (`add_bos`) — the shared prompt
/// builder for the acceptance checks and `tools/cortexdiff` fixtures.
fn reference_prompt(m: &model::Model) -> Vec<usize> {
    let mut prompt: Vec<usize> = Vec::new();
    if m.config.add_bos {
        if let Some(b) = m.config.bos_token_id {
            prompt.push(b as usize);
        }
    }
    prompt.extend(m.tokenizer().encode(refcheck::PROMPT).iter().map(|&t| t as usize));
    prompt
}

/// Load the model, run the fixed reference prompt (encoded at runtime by the
/// model's own tokenizer), and greedily decode the fixture's length. Logs the
/// mandatory per-inference provenance (`model hash + seed + input hash`) and
/// compares the greedy continuation to the model's fixture
/// (`refcheck::FIXTURES`, generated by `tools/cortexdiff`; no fixture → SKIP).
pub fn run_reference_inference() -> Option<InferResult> {
    let bytes = model_module()?;
    let gguf = gguf::Gguf::parse(bytes).ok()?;
    let fixture = refcheck::for_model(gguf.name);
    let m = model::Model::load(gguf).ok()?;

    let prompt = reference_prompt(&m);

    // Provenance log: model hash, seed (greedy => 0), input hash. Greedy
    // temp-0 decoding is deterministic, so the "seed" is fixed at 0. Hash a
    // bounded header prefix rather than the whole (possibly hundreds-of-MiB)
    // model -- it is a provenance fingerprint, logged not asserted, and this
    // keeps it cheap on both arches (and correct when the aarch64 slice is a
    // generous upper bound rather than the exact file length).
    let model_hash = model::fnv1a(&bytes[..bytes.len().min(1 << 16)]);
    let prompt_u32: Vec<u32> = prompt.iter().map(|&t| t as u32).collect();
    let input_hash = model::fnv1a(bytemuck_ids(&prompt_u32));
    crate::ktrace::log_fmt(format_args!(
        "cortex.infer: model_hash={model_hash:#018x} seed=0 input_hash={input_hash:#018x} prompt_len={}",
        prompt.len()
    ));

    let mut kv = m.new_cache();
    let mut state = m.new_state();

    // Prefill the prompt. CPU inference on this model under QEMU is slow
    // (~10-15s/token), so log progress per token -- otherwise the long
    // silent gap here looks like a hang.
    let n_prompt = prompt.len();
    let prefill_start = crate::arch::now_ms();
    m.prefill(&prompt, 0, &mut kv, &mut state);
    let prefill_ms = crate::arch::now_ms().saturating_sub(prefill_start);
    let logits_pos = n_prompt - 1;
    let prompt_final_argmax = model::argmax(&state.logits);
    let prompt_final_logit = state.logits[prompt_final_argmax];

    // Greedy decode, streaming each token's text as it is produced.
    let n_gen = fixture.map(|f| f.expected.len()).unwrap_or(8);
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

    let matched_reference = fixture.map(|f| {
        continuation.len() == f.expected.len()
            && continuation.iter().zip(f.expected.iter()).all(|(&a, &b)| a == b as usize)
    });

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

/// Relative RMS error of the fast Q4_K SDOT matvec vs the exact dequant path
/// on random super-blocks — the aarch64 acceptance check for the Q4_K kernel
/// (the counterpart to [`check_q4_0_sdot`]; run by `/bench`).
#[cfg(target_arch = "aarch64")]
pub fn check_q4_k_sdot() -> f32 {
    use tensor::{Q4_K_BLOCK_BYTES, QK, QK_K, QT_Q4_K};
    const ROWS: usize = 512;
    const COLS: usize = 1024;
    let superblocks = COLS / QK_K;
    let row_bytes = superblocks * Q4_K_BLOCK_BYTES;
    let mut w = alloc::vec![0u8; ROWS * row_bytes];
    let mut seed: u32 = 0x517c_c1b7;
    let mut next = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        seed
    };
    for r in 0..ROWS {
        for b in 0..superblocks {
            let base = r * row_bytes + b * Q4_K_BLOCK_BYTES;
            // d ~ 0.125, dmin ~ 0.0625 (f16), then random scales + nibbles.
            w[base..base + 2].copy_from_slice(&0x3000u16.to_le_bytes());
            w[base + 2..base + 4].copy_from_slice(&0x2C00u16.to_le_bytes());
            for i in 4..Q4_K_BLOCK_BYTES {
                w[base + i] = (next() >> 24) as u8;
            }
        }
    }
    let mut x = alloc::vec![0.0f32; COLS];
    for (i, xi) in x.iter_mut().enumerate() {
        *xi = ((i % 23) as f32 - 11.0) * 0.07;
    }
    let mut y_exact = alloc::vec![0.0f32; ROWS];
    // SAFETY: sizes match the generic kernel's contract.
    unsafe {
        tensor::matvec_quant_rows(QT_Q4_K, w.as_ptr(), x.as_ptr(), y_exact.as_mut_ptr(), 0, ROWS, COLS);
    }
    let mut xq = alloc::vec![0i8; COLS];
    let mut xs = alloc::vec![0.0f32; COLS / QK];
    tensor::quantize_activations_q8(&x, &mut xq, &mut xs);
    let mut y_sdot = alloc::vec![0.0f32; ROWS];
    // SAFETY: sizes match the kernel's contract.
    unsafe {
        tensor::matvec_q4_k_sdot_rows(w.as_ptr(), xq.as_ptr(), xs.as_ptr(), y_sdot.as_mut_ptr(), 0, ROWS, COLS);
    }
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for r in 0..ROWS {
        let d = y_sdot[r] - y_exact[r];
        num += d * d;
        den += y_exact[r] * y_exact[r];
    }
    crate::serial_println!("q4ksdot> y_exact[0..3]={} {} {} y_sdot={} {} {}", y_exact[0], y_exact[1], y_exact[2], y_sdot[0], y_sdot[1], y_sdot[2]);
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
/// `pump` is called between phases and per decoded token — the shell passes a
/// closure that ticks the UI/net upkeep and polls Ctrl+C (return `true` to
/// cancel; the standing rule: no blocking command without an interrupt check).
/// A 27B bench is minutes of wall time, so each phase also ktraces progress —
/// a silent multi-minute `/perf` is indistinguishable from a hang.
pub fn bench_inference(n_prompt: usize, n_decode: usize, pump: &mut dyn FnMut() -> bool) -> Option<InferBench> {
    let bytes = model_module()?;
    let gguf = gguf::Gguf::parse(bytes).ok()?;
    crate::ktrace::log("cortex.bench", "gguf parsed");
    let m = model::Model::load(gguf).ok()?;
    let mut kv = m.new_cache();
    let mut state = m.new_state();
    crate::ktrace::log("cortex.bench", "model + cache ready");
    if pump() {
        return None;
    }

    // Synthetic prompt: fixed ids cycled to the requested length. A
    // throughput gauge needs valid token ids, not real text — building the
    // tokenizer here (500K-alloc BTreeMaps) would dwarf the bench itself.
    let vocab = m.vocab().max(3);
    let prompt: Vec<usize> = (0..n_prompt).map(|i| 1 + (i * 97) % (vocab - 2)).collect();

    let t0 = crate::arch::now_ms();
    m.prefill(&prompt, 0, &mut kv, &mut state);
    let prefill_ms = crate::arch::now_ms().saturating_sub(t0);
    crate::ktrace::log_fmt(format_args!("cortex.bench: prefill {n_prompt} done in {prefill_ms} ms"));
    if pump() {
        return None;
    }

    let mut pos = n_prompt;
    let mut next = model::argmax(&state.logits);
    let t1 = crate::arch::now_ms();
    for i in 0..n_decode {
        m.forward(next, pos, &mut kv, &mut state, true);
        pos += 1;
        next = model::argmax(&state.logits);
        if i % 8 == 7 {
            crate::ktrace::log_fmt(format_args!("cortex.bench: decode {}/{n_decode}", i + 1));
        }
        if pump() {
            return None;
        }
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

/// The full acceptance gate, run in-kernel against whichever model was booted
/// (via `cargo xtask ref-check [-model …]`): reference parity (against the
/// model's `refcheck::FIXTURES` entry; SKIP when it has none), sampler
/// determinism, KV evict+recompute reproducibility, and 2-agent continuous
/// batching (which must also agree with the single-stream greedy run).
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
    let name: Option<alloc::string::String> = gguf.name.map(alloc::string::String::from);
    let fixture = refcheck::for_model(gguf.name);
    let Ok(m) = model::Model::load(gguf) else {
        crate::serial_println!("REFCHECK: ALL FAIL (model load)");
        return false;
    };

    // The prompt comes from the model's own tokenizer (exercising encode per
    // family); the expected continuation from the per-model fixture table.
    let prompt = reference_prompt(&m);
    let reference: Vec<usize> = fixture.map(|f| f.expected.iter().map(|&i| i as usize).collect()).unwrap_or_default();
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

    // (a) Reference parity: greedy continuation matches the model's fixture
    // (generated by tools/cortexdiff, llama.cpp-cross-checked). No fixture →
    // SKIP (the self-consistency checks below still gate).
    let parity = if let Some(f) = fixture {
        let mut kv = m.new_cache();
        let parity_cont = greedy(&m, &mut kv, reference.len());
        let ok = parity_cont == reference;
        crate::serial_println!("REFCHECK: parity matched={} got={:?} want={:?}", ok, parity_cont, f.expected);
        ok
    } else {
        crate::serial_println!("REFCHECK: parity SKIP (no fixture for model name {:?})", name.as_deref());
        true
    };

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
    // reproduces the identical continuation (and the fixture prefix, when
    // the model has one).
    let mut kv2 = m.new_cache();
    let before = greedy(&m, &mut kv2, ACCEPT_GEN);
    kv2.evict();
    let after = greedy(&m, &mut kv2, ACCEPT_GEN);
    let evict_recompute =
        before == after && fixture.map(|_| before == reference[..ACCEPT_GEN]).unwrap_or(true);
    crate::serial_println!(
        "REFCHECK: kv_evict_recompute matched={} before={:?} after={:?}",
        evict_recompute,
        before,
        after
    );

    // (d) Continuous batching: two agents advance in interleaved forward
    // passes; both must agree with the *single-stream* greedy run (batched ==
    // sequential parity), and the step order is interleaved (alternating
    // stream ids), not sequential.
    let mut b = batch::Batch::new(&m);
    b.add_stream(&prompt, ACCEPT_GEN);
    b.add_stream(&prompt, ACCEPT_GEN);
    b.run_greedy();
    let g0 = b.generated(0, prompt.len()).to_vec();
    let g1 = b.generated(1, prompt.len()).to_vec();
    let interleaved = b.step_order.windows(2).any(|w| w[0] != w[1]);
    let batching = g0 == before && g1 == before && interleaved;
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
    // A runtime `/model load` override wins over the boot-time module.
    if let Some(b) = MODEL_OVERRIDE.with(|m| *m) {
        return Some(b);
    }
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

/// aarch64: the model is placed in RAM by QEMU's `-device loader` and/or the
/// UEFI stub. Preference order:
/// 1. Stub-reported `(base, size)` when the ESP path loaded the GGUF.
/// 2. Fixed [`MODEL_LOAD_ADDR`] when QEMU's loader (or a fixed-address place)
///    left a GGUF there — the window runs to the end of discovered RAM on the
///    UEFI path (heap is a *separate* firmware allocation, not "above" the
///    model), or up to the heap on the `-kernel` path.
///
/// Only if the GGUF magic is present (so `infer` cleanly reports "no model"
/// when none was loaded). The GGUF parser reads only within the real file; the
/// window just needs to cover it.
#[cfg(all(not(target_arch = "x86_64"), not(feature = "boot-limine")))]
pub fn model_module() -> Option<&'static [u8]> {
    // A runtime `/model load` override wins over the boot-time module.
    if let Some(b) = MODEL_OVERRIDE.with(|m| *m) {
        return Some(b);
    }
    // Apple Silicon (m1n1): there is no model injected at the fixed
    // MODEL_LOAD_ADDR (2 GiB — below Apple's 32 GiB RAM base, so unbacked;
    // reading its magic faults) and no disk to scan. Instead m1n1 can load the
    // GGUF as the **initramfs** (CHITTI_INITRD): the `/chosen` initrd region is
    // a contiguous RAM span the mmu already maps Normal. Use it when it holds a
    // GGUF; else boot model-less. QEMU/UEFI are unaffected.
    if crate::arch::aarch64::is_apple() && crate::arch::aarch64::mmu::uefi_model().is_none() {
        // SAFETY: `boot_x0` is the FDT pointer (or non-FDT, rejected by the magic).
        if let Some(c) = unsafe { crate::fdt::chosen(crate::arch::aarch64::boot::boot_x0()) } {
            if c.initrd_start != 0 && c.initrd_end > c.initrd_start {
                let addr = c.initrd_start as usize;
                let len = (c.initrd_end - c.initrd_start) as usize;
                // SAFETY: [addr, addr+len) is the initrd, in mmu-mapped Normal RAM.
                let magic = unsafe { core::slice::from_raw_parts(addr as *const u8, 4.min(len)) };
                if magic == b"GGUF" {
                    crate::ktrace::log_fmt(format_args!("cortex: model from initrd at {addr:#x} ({len} bytes)"));
                    // SAFETY: contiguous RAM holding the GGUF; the parser reads
                    // only the real model within `len`.
                    return Some(unsafe { core::slice::from_raw_parts(addr as *const u8, len) });
                }
            }
        }
        return None;
    }
    let (addr, window) = match crate::arch::aarch64::mmu::uefi_model() {
        Some((base, size)) => (base, size),
        None => {
            let heap_base = crate::mm::heap_base();
            let ram_end = crate::arch::aarch64::mmu::ram_end() as usize;
            // UEFI: heap is AnyPages elsewhere; a multi-GiB GGUF often cannot
            // be reassembled by the stub (contiguous LOADER_DATA) and is instead
            // QEMU-loader-injected at MODEL_LOAD_ADDR. Size the window from
            // there to the top of discovered RAM so the full file is visible.
            // `-kernel`: heap sits at the top of RAM, so the classic gap works.
            let window = if crate::arch::aarch64::mmu::uefi_heap_base() != 0 {
                ram_end.saturating_sub(MODEL_LOAD_ADDR)
            } else {
                heap_base.saturating_sub(MODEL_LOAD_ADDR)
            };
            if window == 0 {
                crate::ktrace::log_fmt(format_args!(
                    "cortex: no model window at {MODEL_LOAD_ADDR:#x} (heap={heap_base:#x} ram_end={ram_end:#x})"
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
    // A runtime `/model load` override wins over the boot-time module.
    if let Some(b) = MODEL_OVERRIDE.with(|m| *m) {
        return Some(b);
    }
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
