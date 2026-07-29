#![cfg_attr(target_arch = "aarch64", feature(stdarch_neon_i8mm))]
//! **cortexdiff** — host-side calibration harness for the kernel's GGUF /
//! transformer engine (`kernel/src/cortex`). Mounts the kernel's own parser,
//! tensor kernels, tokenizer, and forward pass natively (`#[path]`, the
//! `tools/onnxdiff` pattern), stubbing only the two kernel dependencies
//! (`ktrace`, the aarch64 SMP row-split — replaced by single-core calls into
//! the same `tensor::*_rows` drivers). A real GGUF loads and greedy-decodes in
//! seconds, so per-model reference fixtures and per-layer bring-up diffs never
//! need a QEMU round-trip. The oracle is llama.cpp (`tools/cortexdiff/diff.py`
//! runs `llama-cli` on the same file and compares token ids).
//!
//! Usage:
//!   cortexdiff meta   <model.gguf>                 # arch/family/config summary
//!   cortexdiff encode <model.gguf> <text>          # tokenizer ids (one line)
//!   cortexdiff greedy <model.gguf> <text> <n>      # greedy-decode n tokens:
//!                                                  #   prompt ids, continuation
//!                                                  #   ids, text — the fixture/
//!                                                  #   diff format

extern crate alloc;

/// Stub of the kernel's cooperative-scheduler upkeep pump (the tokenizer
/// build calls it): a no-op on the host.
pub mod shell {
    pub fn upkeep() {}
}

/// Stub of the kernel's heap module. Only `HEAP_SIZE` is referenced (by
/// `model::chunk_for_scratch`, which sizes the prefill chunk as a fraction of
/// the heap); mirror the 1 GiB tier so the harness picks the same chunk the
/// kernel would.
pub mod mm {
    pub mod heap {
        pub const HEAP_SIZE: usize = 1024 * 1024 * 1024;
    }
}

/// Stub of the kernel's ktrace: forwarded to stderr so decode output on
/// stdout stays machine-parseable.
pub mod ktrace {
    pub fn log(tag: &str, msg: &str) {
        eprintln!("[{tag}] {msg}");
    }
    #[allow(dead_code)]
    pub fn log_fmt(args: core::fmt::Arguments<'_>) {
        eprintln!("{args}");
    }
}

/// Stubs of the kernel's per-arch helpers the cortex modules call. The SMP
/// wrappers row-split across cores in-kernel; here they run the identical
/// single-core `tensor::*_rows` drivers, so the math under test is unchanged.
pub mod arch {
    /// Host-side FEAT_I8MM probe (the kernel reads ID_AA64ISAR1_EL1; on the
    /// host we ask std). Gates the i8mm matmul in the mounted `model.rs`.
    #[cfg(target_arch = "aarch64")]
    pub fn has_i8mm() -> bool {
        // CHITTI_NO_I8MM=1 forces the SDOT path for host A/B timing.
        std::env::var("CHITTI_NO_I8MM").is_err() && std::arch::is_aarch64_feature_detected!("i8mm")
    }
    #[cfg(not(target_arch = "aarch64"))]
    pub fn has_i8mm() -> bool {
        false
    }

    /// Stub of the kernel's free-running cycle counter, used by the prefill
    /// phase accounting. Host monotonic nanoseconds are the same shape (a
    /// monotonically-advancing tick), which is all the counters need.
    pub fn cycle_count() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0)
    }

    /// Stub of the kernel's arch-neutral compute fan-out: the harness is
    /// single-threaded, so the whole range runs inline. `parallel_for`'s own
    /// contract is that a serial run over `[0, n)` is a valid partition, which
    /// is exactly what makes the kernel's parallel cores comparable here.
    ///
    /// # Safety
    /// Same contract as `crate::smp::parallel_for`: `f` must be safe over
    /// `[0, n)` with `ctx` live for the call.
    pub unsafe fn parallel_for(n: usize, _min_chunk: usize, f: unsafe fn(usize, usize, *mut u8), ctx: *mut u8) {
        unsafe { f(0, n, ctx) }
    }

    #[cfg(target_arch = "aarch64")]
    pub mod aarch64 {
        pub mod smp {
            use crate::cortex::tensor;

            /// # Safety
            /// Same contract as `tensor::matmul_q8_0_sdot_rows` over `[0, n_rows)`.
            pub unsafe fn matmul_sdot(
                w: *const u8,
                xq: *const i8,
                xs: *const f32,
                y: *mut f32,
                m_count: usize,
                n_rows: usize,
                n_cols: usize,
            ) {
                unsafe { tensor::matmul_q8_0_sdot_rows(w, xq, xs, y, m_count, n_rows, 0, n_rows, n_cols) }
            }

            /// # Safety
            /// Same contract as `tensor::matvec_quant_rows` over `[0, n_rows)`.
            pub unsafe fn matvec_quant(qt: u32, w: *const u8, x: *const f32, y: *mut f32, n_rows: usize, n_cols: usize) {
                unsafe { tensor::matvec_quant_rows(qt, w, x, y, 0, n_rows, n_cols) }
            }

            /// # Safety
            /// Same contract as `tensor::matvec_q4_0_sdot_rows` over `[0, n_rows)`.
            pub unsafe fn matvec_q4_0_sdot(w: *const u8, xq: *const i8, xs: *const f32, y: *mut f32, n_rows: usize, n_cols: usize) {
                unsafe { tensor::matvec_q4_0_sdot_rows(w, xq, xs, y, 0, n_rows, n_cols) }
            }

            /// # Safety
            /// Same contract as `tensor::matvec_q4_k_sdot_rows` over `[0, n_rows)`.
            pub unsafe fn matvec_q4_k_sdot(w: *const u8, xq: *const i8, xs: *const f32, y: *mut f32, n_rows: usize, n_cols: usize) {
                unsafe { tensor::matvec_q4_k_sdot_rows(w, xq, xs, y, 0, n_rows, n_cols) }
            }

            /// # Safety
            /// Same contract as `tensor::matvec_q2_0_sdot_rows` over `[0, n_rows)`.
            pub unsafe fn matvec_q2_0_sdot(w: *const u8, xq: *const i8, xs: *const f32, y: *mut f32, n_rows: usize, n_cols: usize) {
                unsafe { tensor::matvec_q2_0_sdot_rows(w, xq, xs, y, 0, n_rows, n_cols) }
            }

            /// # Safety
            /// Same contract as `tensor::matvec_q1_0_sdot_rows` over `[0, n_rows)`.
            pub unsafe fn matvec_q1_0_sdot(w: *const u8, xq: *const i8, xs: *const f32, y: *mut f32, n_rows: usize, n_cols: usize) {
                unsafe { tensor::matvec_q1_0_sdot_rows(w, xq, xs, y, 0, n_rows, n_cols) }
            }

            /// # Safety
            /// Same contract as `tensor::matmul_q1_0_sdot_rows` over `[0, n_rows)`.
            pub unsafe fn matmul_q1_0_sdot(w: *const u8, xq: *const i8, xs: *const f32, y: *mut f32, m_count: usize, n_rows: usize, n_cols: usize) {
                unsafe { tensor::matmul_q1_0_sdot_rows(w, xq, xs, y, m_count, n_rows, 0, n_rows, n_cols) }
            }

            /// # Safety
            /// FEAT_I8MM required (caller gates on `crate::arch::has_i8mm()`);
            /// same contract as `tensor::matmul_q1_0_i8mm_rows` over `[0, n_rows)`.
            pub unsafe fn matmul_q1_0_i8mm(w: *const u8, xq: *const i8, xs: *const f32, y: *mut f32, m_count: usize, n_rows: usize, n_cols: usize) {
                unsafe { tensor::matmul_q1_0_i8mm_rows(w, xq, xs, y, m_count, n_rows, 0, n_rows, n_cols) }
            }

            /// # Safety
            /// FEAT_I8MM required; same contract as `tensor::matmul_q8_0_i8mm_rows`.
            pub unsafe fn matmul_q8_0_i8mm(w: *const u8, xq: *const i8, xs: *const f32, y: *mut f32, m_count: usize, n_rows: usize, n_cols: usize) {
                unsafe { tensor::matmul_q8_0_i8mm_rows(w, xq, xs, y, m_count, n_rows, 0, n_rows, n_cols) }
            }

            /// # Safety
            /// FEAT_I8MM required; same contract as `tensor::matmul_q4_0_i8mm_rows`.
            pub unsafe fn matmul_q4_0_i8mm(w: *const u8, xq: *const i8, xs: *const f32, y: *mut f32, m_count: usize, n_rows: usize, n_cols: usize) {
                unsafe { tensor::matmul_q4_0_i8mm_rows(w, xq, xs, y, m_count, n_rows, 0, n_rows, n_cols) }
            }

            /// # Safety
            /// Same contract as `tensor::matmul_q2_0_sdot_rows` over `[0, n_rows)`.
            pub unsafe fn matmul_q2_0_sdot(w: *const u8, xq: *const i8, xs: *const f32, y: *mut f32, m_count: usize, n_rows: usize, n_cols: usize) {
                unsafe { tensor::matmul_q2_0_sdot_rows(w, xq, xs, y, m_count, n_rows, 0, n_rows, n_cols) }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    pub mod x86_64 {
        pub mod fpu {
            /// Host x86 always has AVX2 in practice; report what the CPU says.
            pub fn avx2_enabled() -> bool {
                std::is_x86_feature_detected!("avx2")
            }
        }
    }
}

/// The kernel's cortex modules, mounted verbatim (see `src/cortex/mod.rs`).
pub mod cortex;

use cortex::{gguf::Gguf, model, model::Model};

fn usage() -> ! {
    eprintln!("usage: cortexdiff meta|encode|greedy <model.gguf> [text] [n]");
    std::process::exit(2)
}

/// Row-range parity for the batched i8mm matmuls: computing `[0, n)` in one
/// call must equal computing it as a set of adjacent sub-ranges.
///
/// This is what SMP dispatch actually does — `parallel_for` hands each core a
/// `[row_start, row_end)` slice — and it is the one thing the in-kernel tests
/// never covered: they only ever call with `(0, rows)`. `cortexdiff`'s stub
/// runs the whole range on one core, so a range bug is invisible here too until
/// you split it deliberately. Splits are uneven and land on odd rows so the
/// kernels' odd-row tail path is exercised from a non-zero start.
///
/// Lives in the host harness because the kernels are aarch64+i8mm only, so the
/// x86 `cargo xtask test` suite can never reach them.
#[cfg(target_arch = "aarch64")]
fn rangecheck() -> i32 {
    use cortex::tensor;
    fn lcg(s: &mut u32) -> u32 {
        *s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        *s >> 8
    }
    if !arch::has_i8mm() {
        eprintln!("rangecheck: host has no FEAT_I8MM; nothing to check");
        return 0;
    }
    // (label, block bytes, elements per block, runner)
    type Run = unsafe fn(*const u8, *const i8, *const f32, *mut f32, usize, usize, usize, usize, usize);
    let cases: [(&str, usize, usize, Run); 3] = [
        ("q4_0", 18, 32, tensor::matmul_q4_0_i8mm_rows),
        ("q8_0", 34, 32, tensor::matmul_q8_0_i8mm_rows),
        ("q1_0", 18, 128, tensor::matmul_q1_0_i8mm_rows),
    ];
    let mut bad = 0;
    for (name, block_bytes, elems, run) in cases {
        for &(rows, cols, m) in &[(13usize, 256usize, 6usize), (16, 128, 8), (9, 384, 5)] {
            if cols % elems != 0 {
                continue;
            }
            let mut seed = 0x4D0Fu32 ^ (rows as u32) ^ ((m as u32) << 8);
            let mut w = Vec::new();
            for _ in 0..rows * (cols / elems) {
                w.extend_from_slice(&[0x00, 0x38]); // f16 0.5 scale
                for _ in 0..block_bytes - 2 {
                    w.push((lcg(&mut seed) & 0xff) as u8);
                }
            }
            let nb = cols / 32; // activation scales are always QK=32-blocked
            let mut xq = vec![0i8; m * cols];
            let mut xs = vec![0.0f32; m * nb];
            for mi in 0..m {
                let x: Vec<f32> =
                    (0..cols).map(|_| (lcg(&mut seed) % 1000) as f32 / 500.0 - 1.0).collect();
                tensor::quantize_activations_q8(
                    &x,
                    &mut xq[mi * cols..(mi + 1) * cols],
                    &mut xs[mi * nb..(mi + 1) * nb],
                );
            }
            // Cut points: raw (may land on odd rows, as the adaptive
            // `row_boundary` used to) and even-snapped (what
            // `row_boundary_even` now produces for these kernels).
            let cuts = |even: bool| {
                let mut v = vec![0usize];
                let mut c = 0usize;
                for step in [3usize, 5, 1, 4, 2, 7, 6] {
                    c += step;
                    if c >= rows {
                        break;
                    }
                    let b = if even { c & !1 } else { c };
                    if b > *v.last().unwrap() {
                        v.push(b);
                    }
                }
                v.push(rows);
                v
            };
            let mut whole = vec![0.0f32; m * rows];
            // SAFETY: buffers sized to the kernel contract; host has i8mm.
            unsafe {
                run(w.as_ptr(), xq.as_ptr(), xs.as_ptr(), whole.as_mut_ptr(), m, rows, 0, rows, cols)
            };
            let mut worsts = [0.0f32; 2];
            for (which, even) in [(0usize, false), (1usize, true)] {
                let mut split = vec![0.0f32; m * rows];
                for pair in cuts(even).windows(2) {
                    // SAFETY: adjacent sub-ranges covering [0, rows).
                    unsafe {
                        run(w.as_ptr(), xq.as_ptr(), xs.as_ptr(), split.as_mut_ptr(),
                            m, rows, pair[0], pair[1], cols)
                    };
                }
                worsts[which] =
                    (0..m * rows).map(|i| (whole[i] - split[i]).abs()).fold(0.0f32, f32::max);
            }
            let ok = worsts[1] == 0.0;
            println!(
                "{:5} rows={rows:<3} cols={cols:<4} m={m}   raw-split {:>11e}   even-split {:>11e}  {}",
                name, worsts[0], worsts[1],
                if ok { "OK" } else { "FAIL" }
            );
            if !ok {
                bad += 1;
            }
        }
    }
    if bad == 0 {
        println!("rangecheck: all kernels are row-range exact");
    } else {
        println!("rangecheck: {bad} case(s) FAILED -- a split range does not equal the whole");
    }
    bad
}

#[cfg(not(target_arch = "aarch64"))]
fn rangecheck() -> i32 {
    eprintln!("rangecheck: aarch64-only (the i8mm kernels do not exist on this target)");
    0
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // `synthetic <model> [n_prompt] [chunk]` — reproduce `/perf`'s deterministic
    // synthetic prompt and print the same per-chunk logits fingerprint the
    // kernel ktraces, so the two can be diffed to the first divergent chunk.
    if args.get(1).map(|s| s.as_str()) == Some("synthetic") {
        let path = args.get(2).cloned().unwrap_or_else(|| usage());
        let n_prompt: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(512);
        let chunk: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(64);
        let bytes = std::fs::read(&path).expect("read model");
        let gguf = Gguf::parse(&bytes).expect("gguf parse");
        let m = Model::load(gguf).expect("model load");
        // Identical to `cortex::bench_inference`'s synthetic prompt.
        let vocab = m.vocab().max(3);
        let prompt: Vec<usize> = (0..n_prompt).map(|i| 1 + (i * 97) % (vocab - 2)).collect();
        let mut kv = m.new_cache();
        let mut state = m.new_state();
        let (mut i, mut ck) = (0usize, 0usize);
        while i < prompt.len() {
            let j = (i + chunk).min(prompt.len());
            m.prefill(&prompt[i..j], i, &mut kv, &mut state);
            println!("chunk {ck} pos {i}..{j} logits_hash={:#018x}", cortex::tensor::logits_hash(&state.logits));
            ck += 1;
            i = j;
        }
        // Decode the same way `bench_inference` does (argmax, no sampler), so
        // the long-context decode path can be diffed too.
        let n_dec: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(8);
        let mut pos = prompt.len();
        let mut next = cortex::model::argmax(&state.logits);
        for d in 0..n_dec {
            m.forward(next, pos, &mut kv, &mut state, true);
            pos += 1;
            next = cortex::model::argmax(&state.logits);
            println!("dec {d} pos {pos} tok {next} logits_hash={:#018x}", cortex::tensor::logits_hash(&state.logits));
        }
        return;
    }
    if args.get(1).map(|s| s.as_str()) == Some("rangecheck") {
        std::process::exit(if rangecheck() == 0 { 0 } else { 1 });
    }
    let (cmd, path) = match (args.get(1), args.get(2)) {
        (Some(c), Some(p)) => (c.as_str(), p.as_str()),
        _ => usage(),
    };
    let t0 = std::time::Instant::now();
    let bytes = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("reading {path}: {e}");
        std::process::exit(1)
    });
    let gguf = Gguf::parse(&bytes).unwrap_or_else(|e| {
        eprintln!("gguf parse: {e:?}");
        std::process::exit(1)
    });
    eprintln!(
        "loaded {} ({} MiB) in {:?}: arch={} family={:?} name={:?}",
        path,
        bytes.len() >> 20,
        t0.elapsed(),
        gguf.arch,
        gguf.config.family,
        gguf.name
    );

    match cmd {
        "meta" => {
            println!("arch={} family={:?} name={:?} tokenizer={}", gguf.arch, gguf.config.family, gguf.name, gguf.tokenizer_model);
            println!("{:#?}", gguf.config);
            println!(
                "tokens={} merges={} scores={} token_type={} add_bos={} bos={:?} eos={}",
                gguf.tokens.len(),
                gguf.merges.len(),
                gguf.scores.len(),
                gguf.token_type.len(),
                gguf.config.add_bos,
                gguf.config.bos_token_id,
                gguf.config.eos_token_id
            );
        }
        "encode" => {
            // Tokenizer-only: no Model::load, so a metadata-only head of a
            // multi-GB GGUF (range-downloaded) is enough to validate encode.
            let text = args.get(3).cloned().unwrap_or_else(|| usage());
            let ids = cortex::tokenizer::Tokenizer::build(&gguf).encode(&text);
            println!("{}", ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" "));
        }
        "greedy" => {
            let text = args.get(3).cloned().unwrap_or_else(|| usage());
            let n_gen: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(8);
            let m = Model::load(gguf).expect("model load");

            // Raw-text greedy parity (no chat template): prompt ids, then N
            // greedy tokens. `add_bos` models get BOS prepended, matching
            // `llama-cli -p <text> --temp 0` on the same file.
            let mut prompt: Vec<usize> = Vec::new();
            if m.config.add_bos {
                if let Some(b) = m.config.bos_token_id {
                    prompt.push(b as usize);
                }
            }
            prompt.extend(m.tokenizer().encode(&text).iter().map(|&t| t as usize));
            println!("prompt_ids: {}", prompt.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" "));

            let mut kv = m.new_cache();
            let mut state = m.new_state();
            // Feed in the same chunks the OS does, so chunk sizing can be A/B'd
            // here (seconds, single core) instead of in a VM. `CHITTI_CHUNK`
            // overrides; default is the model's own `prefill_chunk`.
            let chunk = std::env::var("CHITTI_CHUNK")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(64)
                .max(1);
            let t1 = std::time::Instant::now();
            let mut i = 0usize;
            while i < prompt.len() {
                let j = (i + chunk).min(prompt.len());
                m.prefill(&prompt[i..j], i, &mut kv, &mut state);
                i = j;
            }
            eprintln!("prefill {} tokens in {:?} (chunk {chunk})", prompt.len(), t1.elapsed());

            let mut next = model::argmax(&state.logits);
            let mut pos = prompt.len();
            let mut out: Vec<usize> = Vec::new();
            let t2 = std::time::Instant::now();
            for _ in 0..n_gen {
                out.push(next);
                if next == m.eos() {
                    break;
                }
                m.forward(next, pos, &mut kv, &mut state, true);
                pos += 1;
                next = model::argmax(&state.logits);
            }
            eprintln!("decode {} tokens in {:?}", out.len(), t2.elapsed());
            println!("continuation_ids: {}", out.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" "));
            println!("text: {}", model::detokenize(&m, &out));
        }
        _ => usage(),
    }
}
