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

fn main() {
    let args: Vec<String> = std::env::args().collect();
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
            let t1 = std::time::Instant::now();
            m.prefill(&prompt, 0, &mut kv, &mut state);
            eprintln!("prefill {} tokens in {:?}", prompt.len(), t1.elapsed());

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
