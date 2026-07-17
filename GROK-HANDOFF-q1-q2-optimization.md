# Task: Optimize the Q1_0 (1-bit) and Q2_0 (ternary) inference kernels in ChittiOS

## Goal

Maximize CPU inference throughput for **PrismML Bonsai-27B** on ChittiOS (a bare-metal
Rust OS running a 27B LLM in-kernel). Two quant formats, both fully working and
numerically verified — this task is **pure performance optimization**:

- **Q1_0** (GGML type 41, binary): 128-elem block = f16 scale `d` + 128 sign bits
  (LSB-first per byte). Value = `bit ? +d : -d`. Model: `assets/model-bonsai-27b-q1.gguf`
  (3.6 GB, the default `make run` model).
- **Q2_0** (GGML type 42, ternary): 128-elem block = f16 scale + 128 2-bit codes
  (4/byte, LSB-first). Value = `(code - 1) * d`, code ∈ 0..3 (so −1, 0, +1, +2).
  Model: `assets/model-bonsai-27b.gguf` (6.8 GB, `-model bonsai-27b-ternary`).

Both are Qwen3.5/3.6-hybrid (`QwenHybrid` family: 48 gated-DeltaNet layers + 16
full-attention layers, dim 5120, ffn 17408, vocab 248320, ~24.8B weights ≈ 25.7 GMAC
per decoded token).

## Current measured baselines (be honest about beating these)

Host = Apple Silicon, 4 P-cores + 4 E-cores (`hw.physicalcpu: 8`). VM = QEMU HVF,
8 vCPUs (`CHITTI_SMP`, default 8), release build (`make` defaults `RELEASE=1` now).

| Measurement | Value |
|---|---|
| On-kernel batched **prefill**, Q1_0 27B (32-tok chunks, 8 vCPUs) | **~3 tok/s** at low ctx, ~2 tok/s at ctx≈1000 (attention grows with position) |
| Host single-core **decode** (cortexdiff, release): Q1_0 | 12 tok / 11.7 s ≈ **1.0 tok/s** |
| Host single-core decode: Q2_0 | 12 tok / 13.0 s ≈ 0.92 tok/s |
| Host single-core batched prefill (m=5), Q2_0 | 5 tok / 4.45 s |
| Fleet MAC throughput (prefill, on-kernel) | ~77 GMAC/s — same ballpark as the repo's tuned Q8_0 batched kernel (0.8B @ 53 tok/s pp ≈ 85 GMAC/s) |

Target: as far toward **15–20 tok/s** as this hardware allows. That needs ~5–7×
current fleet throughput — likely not fully reachable on 4P+4E under HVF, but there
is real headroom (see "Where the cycles go"). Prioritize **decode (tg)** — it's the
interactive number — then prefill (pp).

## The kernels (all in `kernel/src/cortex/tensor.rs`)

- `dequant_q1_0_block` / `dequant_q2_0_block` — scalar reference (used by tests +
  generic fallback). DO NOT break these; they define correctness.
- `sdot_one_row_q1_0` — decode matvec row kernel. Per 128-block: `ldq_u8` all 16 sign
  bytes once → 8× (`vqtbl1q_u8` byte-broadcast via `Q1_0_TBL_IDX` statics + `vtstq_u8`
  vs `Q1_0_BITS` + `vbslq_s8` ±1) → 4 sub-blocks × (ldp_s8 activation + 2 chained
  `vdotq_s32` + scalar `xs` scale accumulate into one f32 `block_acc`).
- `sdot_one_row_q2_0` — same shape; unpack via `q2_0_unpack64` (vand/vshr the 4
  bit-fields + 2 `vzip` levels to restore element order).
- `matmul_q1_0_sdot_rows` / `matmul_q2_0_sdot_rows` — weight-stationary batched
  matmul (prefill): rows outer, activation tiles of `MT = 4` inner; per block the
  unpack runs once per tile and is SDOTed against 4 activations; accumulators are
  one `float32x4` per tile lane (`vfmaq_n_f32` with `d * xs_j`).
- `quantize_activations_q8` — scalar int8 activation quantizer (per 32-elem block),
  runs once per matvec / per batched projection per position.
- Load helpers: `ldq_s8/u8/f32` (single `ldr q`) and `ldp_s8` (paired `ldp q,q`) —
  **all vector loads MUST go through these inline-asm helpers** (see constraints).

Dispatch plumbing (don't need changes unless you re-partition work):
- `kernel/src/arch/aarch64/smp.rs` — worker modes: 5/6 = Q2_0/Q1_0 matvec,
  7/8 = Q1_0/Q2_0 matmul; static equal row-split across BSP + 7 workers;
  **bounded barrier** (500 ms wait + 1500 ms grace, straggler ranges recomputed
  on the BSP; `add_busy` records per-core busy cycles — usable for weighted splits).
- `kernel/src/cortex/model.rs` — `matvec_qw` (per-quant fast-path dispatch),
  `batched_proj` (quantize + matmul dispatch), `prefill_batched` (M-wide buffers),
  `batch_qt` gate.
- `kernel/src/shell/mod.rs` — chat prefill loop feeds 32-token chunks
  (`let chunk = if ... { 32 } else { 1 }`) with UI pump + Ctrl+C between chunks.

## Where the cycles go (analysis so far)

Decode matvec per 128-elem block ≈ 45–50 vector instrs for 128 MACs (8 SDOT = the
useful work, the rest is unpack + loads + accumulate). Ideas with expected value,
roughly ordered:

1. **More ILP in the matvec row kernels.** `block_acc`/`acc` are single serial f32
   chains; the Q8_0 kernel (`matvec_q8_0_neon`, same file) uses 4 independent
   accumulator chains and measurably beats a naive loop. Restructure Q1_0/Q2_0 row
   kernels to (a) keep 2–4 independent `int32x4` SDOT accumulators, (b) defer ALL
   horizontal reduction (`vaddvq`) and scale-FMA out of the inner loop where the
   math allows, (c) software-pipeline 2 blocks (unpack block b+1 while SDOTing b).
   CAUTION: f32 summation-order changes shift near-tie logits (see verification).
2. **Row-pair processing in decode** (process 2 weight rows per activation pass):
   halves activation reload traffic and doubles accumulator parallelism.
3. **Deeper matmul tiles** (`MT` 4 → 8) + unpack-once-per-block-per-row (already) —
   check register pressure with `objdump -d` (32 NEON regs; spills kill it).
4. **Vectorize `quantize_activations_q8`** (currently scalar; runs per projection ×
   per chunk-position in prefill — a real % at m=32).
5. **E-core-aware partitioning**: static equal split makes every matvec wait for the
   slowest (E-core) worker. Use the recorded `add_busy` cycles for weighted row
   splits, or split rows 2× finer with work-stealing off an atomic cursor.
6. **Prefetch**: `prfm pldl1strm` on the next weight row / next block inside the row
   loop (inline asm; the access pattern is perfectly sequential).
7. **Chunk size**: chat CHUNK=32 → try 64 (more weight amortization per pass;
   watch the m-wide buffer growth in `prefill_batched` — heap is 1 GiB).
8. Optional math trick (op-count-neutral per-block, but frees the `vbsl`): for Q1_0,
   `dot(x, ±1) = 2·SDOT(x, bits∈{0,1}) − Σx`, with Σx per 32-sub-block precomputable
   ONCE per activation (amortizes across all rows!). Net: replaces `vbsl` with `vand`
   AND removes half the sign-expand work if you fold `2·` into the scale. Worth
   prototyping in cortexdiff first.

## Hard constraints (violating these = rejected)

1. **`+strict-align` trap (the big one).** The aarch64 kernel target builds with
   `+strict-align`. LLVM lowers ANY unaligned vector load/store — `vld1q_*`
   intrinsics AND auto-vectorized loops — into ~25-instruction byte-assembly
   (~100× slower). ALL vector memory access in hot loops must use the inline-asm
   helpers (`ldq_*`, `ldp_s8`, or new ones in the same style: `ldr q`/`ldp q`/`str q`).
   Verify with `objdump -d` on the kernel binary: the hot function must contain
   **zero `ldrb`** in its inner loop. Register-to-register intrinsics (tbl, zip,
   sdot, tst, bsl...) are fine.
2. **no_std kernel.** No std, no allocation in inner loops. `core::arch::aarch64`
   intrinsics only + inline asm.
3. **Dual-arch rule.** x86_64 must keep building and falling back to the generic
   dequant path (`matvec_quant_rows`). Never gate shared logic on `target_arch`
   beyond the existing pattern (NEON kernels are `#[cfg(target_arch = "aarch64")]`
   with non-aarch64 fallbacks already in place).
4. **Numerical correctness.** The kernels must stay faithful to the scalar dequant
   reference. Unit tests pin this (see verification). f32 accumulation-order changes
   are acceptable ONLY if the cortexdiff greedy continuations stay coherent — ideally
   identical; near-tie divergence is a known caveat (see `cortexdiff-oracle-near-ties`
   memory / repo docs) but treat any changed continuation as a red flag to investigate.
5. **Keep the bounded-barrier discipline** — any new cross-core wait must be bounded
   (never trust a worker wake; see the SMP section of CLAUDE.md).
6. **Style**: match surrounding code; every `unsafe` needs a `// SAFETY:` comment;
   doc comments on public functions; tests for new pure logic.

## Verification & iteration workflow (fast → slow)

```sh
# 1. FAST ITERATION — host-native, seconds, this is your main loop:
cd tools/cortexdiff && cargo build --release
./target/release/cortexdiff greedy ../../assets/model-bonsai-27b-q1.gguf "The capital of France is" 12
./target/release/cortexdiff greedy ../../assets/model-bonsai-27b.gguf    "The capital of France is" 12
# Prints prefill/decode wall times + the continuation. Expected (pre-optimization):
#   Q2_0: " Paris. Paris is the largest city in France. Paris is"
#         (ids: 11751 13 11751 369 279 7526 3177 303 9338 13 11751 369)
#   Q1_0: " Paris.\n\n<think>\nHere's a thinking process:"
# cortexdiff mounts kernel/src/cortex verbatim (#[path]) — same NEON code paths,
# but NOTE: host builds WITHOUT +strict-align, so ldq/ldp discipline must still be
# checked via kernel objdump (step 3), not host timing alone.

# 2. Unit tests (x86 QEMU; the aarch64-gated SDOT tests run via cortexdiff's arch
#    or an on-kernel boot — keep the suite green regardless):
cargo xtask test          # must stay 719/719 [ok]

# 3. Kernel builds + strict-align audit:
cargo xtask build -arch aarch64 --release
cargo xtask build -arch x86_64
objdump -d target/aarch64-unknown-none/release/chitti-kernel 2>/dev/null | \
  awk '/sdot_one_row_q1_0|matmul_q1_0/,/ret/' | grep -c ldrb   # want 0 in hot loops

# 4. On-kernel measurement (QEMU HVF, 8 vCPUs, release):
make run            # boots bonsai-27b Q1_0; type a message; watch serial ktrace:
#   smp: 8 cores online (BSP + 7 workers...)   <- MUST appear; single-core = broken
#   chat.prefill: N/1546 (X tok/s)             <- prefill rate per 32-tok chunk
# /perf runs a 64-tok prefill + 32-tok decode bench (now pumps UI + Ctrl+C works;
#   ktraces cortex.bench progress). Known issue: /perf hung silently pre-fix — if it
#   still stalls, the cortex.bench ktrace lines now show which phase.
# Headless driving: tests/e2e/guest.py `Guest(model="bonsai-27b", release=True)`.
```

Relevant unit tests in `kernel/src/cortex/tensor.rs` (run under `cargo xtask test`):
`dequant_q1_0_bits_and_sign`, `dequant_q2_0_codes_and_packing`,
`sdot_q1_0_matches_dequant_reference`, `sdot_q2_0_matches_dequant_reference`
(aarch64-gated), `matmul_q1_0_q2_0_match_matvec` (aarch64-gated; covers the m%MT
tail). Extend these for any new kernel variant (e.g. row-pair, new tile depths).

## Known environment gotchas

- **HVF WFE-spin**: idle parked workers burn ~100% host CPU each — total CPU% is
  NOT a progress signal. Use the ktrace rates.
- **The PSCI gate**: `smp: 8 cores online` must appear in the boot ktrace. If you
  see "FDT advertises no PSCI" or a single-core boot on QEMU, the fail-open gate
  regressed (`fdt::present` distinguishes no-FDT from FDT-without-PSCI).
- **Dev builds lie**: `make` now defaults `RELEASE=1`; never benchmark a dev kernel
  (unoptimized NEON is many times slower).
- Prefill rate legitimately decays with context (full-attention layers grow with
  position) — compare rates at the SAME context depth.
- Don't benchmark while other QEMU instances / heavy builds run on the host.

## Reference comparison: host llama.cpp (PrismML fork) on the SAME M2

Built the fork's `llama-bench` CPU-only (`cmake -DGGML_METAL=OFF`; M2 has
`FEAT_I8MM=1`, `FEAT_DotProd=1`) and ran the same models. Native host, not QEMU:

| model | test | llama.cpp 8t | llama.cpp 4t | ChittiOS (QEMU-HVF 8vcpu) |
|---|---|---|---|---|
| Q1_0 27B | pp64 | **14.6 t/s** | 14.9 t/s | ~3 t/s |
| Q1_0 27B | tg32 | 3.18 t/s | **4.37 t/s** | ~1.35 t/s |
| Q2_0 27B | pp64 | 3.26 t/s | — | ~3 t/s |
| Q2_0 27B | tg32 | **0.09 t/s** | — | ~1 t/s |

Reading of this table (this is the actionable part):

- **Q1_0 prefill: they are ~5× faster (14.6 vs 3).** The entire gap is the
  **repacked i8mm GEMM**. The fork ships `block_q1_0x4` (4 rows interleaved) with
  `ggml_gemm_q1_0_4x8_q8_0` using **`vmmlaq_s32`** (FEAT_I8MM matrix-multiply-
  accumulate: a 2×2 int8 outer-product per instr = 2× the MAC/instr of `vdotq_s32`).
  Our batched matmul uses `vdotq_s32` and a plain row-major weight layout. THIS is
  the top prefill win. See `ggml/src/ggml-cpu/arch/arm/repack.cpp`
  (`ggml_gemm_q1_0_4x8_q8_0`, `ggml_gemv_q1_0_4x8_q8_0`) and
  `repack_q1_0_to_q1_0_4_bl`.
- **Q1_0 decode: they are ~2.5–3× faster (3.18/4.37 vs 1.35).** Two causes: (a)
  their vec_dot has an **i8mm 2-row path** (`nrc==2`, `vmmlaq_s32` over two weight
  rows at once) — see `ggml_vec_dot_q1_0_q8_0` in `arch/arm/quants.c`; (b) **decode
  is memory-bandwidth-bound — 4 threads BEAT 8** (4.37 > 3.18). Our equal/weighted
  8-way split is actively counterproductive for tg: cap decode matvec to the P-cores
  (~4) instead of all 8.
- **Q2_0: we are at parity on prefill and ~11× AHEAD on decode.** The fork barely
  optimized Q2_0 — no repack GEMM, and its tg path is pathological (0.09 t/s). So
  **do not port their Q2_0**; ours already wins. Q2_0 effort should just mirror
  whatever Q1_0 i8mm work lands.

**Honest ceiling (important):** even *native* llama.cpp on this M2 tops out at
**~14.6 t/s prefill / ~4.4 t/s decode** for a 27B 1-bit. 15–20 t/s decode is not
reachable on this box by anyone. The real, achievable target is **to close the gap
to llama.cpp**: ~14 t/s prefill (5×) and ~4 t/s decode (3×). Both come from the same
two levers: **(1) i8mm (`vmmlaq_s32`) + a repacked interleaved weight layout**, and
**(2) P-core-only decode**. Per-block micro-opts (the earlier list) are secondary now.

### Concrete implementation notes from the fork

- **i8mm `vmmlaq_s32`**: computes `C[2x2] += A[2xK]·B[Kx2]` for int8 K=8-lane
  operands. To use it you feed 2 weight rows + 2 activation columns interleaved
  (`vzip1q_s64`/`vzip2q_s64` to pair rows), accumulate an int32x4 holding the 2×2
  partials, scale by the f16 d's. For **prefill** (m≥2) this is a natural 2-col
  tile; for **decode** (m=1) pair 2 weight ROWS against the 1 activation broadcast
  to 2 (their `nrc==2` path). Gate on `#[cfg(target_feature)]`/runtime detect —
  and confirm HVF passes I8MM through (`-cpu host`; the guest must see it, else
  fall back to the current `vdotq` path).
- **Sign expand via LUT**: the fork expands 8 sign bits → `int8x8` of ±1 with a
  256-entry `table_q1_signs` (`u64` per byte, `vcreate_u8(table[b])`) instead of our
  `vqtbl1q`+`vtst`+`vbsl` (3 vec ops). Marginal vs i8mm but tidy; the NEON-only
  fallback uses `veor`+`vsub` sign-flip masks (`table_q1_mask`) + `vpaddlq` — worth
  a look for the non-dotprod path.
- **Repack**: `repack_q1_0_to_q1_0_4_bl` rewrites the weight tensor once at load
  into `block_q1_0x4` (4 rows interleaved, per-block). In ChittiOS this would be a
  load-time transform of the QWeight bytes (or a parallel copy into DMA frames) —
  weigh the one-time cost + extra memory vs. the 5× prefill.
- **First check on any i8mm work**: does the QEMU-HVF guest actually expose I8MM?
  Add a boot ktrace reading `ID_AA64ISAR1_EL1.I8MM` (bits 55:52). If HVF masks it,
  i8mm gains are host-only and the kernel must keep the vdotq path.

## Deliverables

1. Optimized Q1_0 + Q2_0 matvec and matmul kernels (and any dispatch/partitioning
   changes), all constraints held, tests green, both arches building.
2. Before/after numbers from BOTH cortexdiff (host single-core) and on-kernel
   (`chat.prefill` rate + `/perf` pp/tg at 8 vCPUs, release).
3. A short note per optimization: what it was, measured delta, why it works.
4. Unchanged (or explicitly justified) cortexdiff greedy continuations.
