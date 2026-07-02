# Chitti OS — Agentic Re-Architecture: Progress Log

Append one entry per milestone: phase, what landed, gate status per arch, next step.

---

## Run start

- Read all three handoff docs in full; emitted the Decisions ledger (see DECISIONS.md).
- Baseline before the re-arch: 69/69 in-kernel tests green; both arches build; 9B forward
  validated byte-exact vs llama.cpp on `/infer`. Carried-over uncommitted work: a top-k/top-p
  nucleus sampler (`cortex::sampler::sample_topk_topp`) + chat `pick()` using it — committing
  as a clean baseline before the re-architecture.
- Plan: Milestone 0 (deps + shared `agent/types.rs` contract), then Phases A→G, each gated by
  `cargo xtask test` (x86_64 QEMU) + aarch64 build. 9B chat fix deferred to the end.

<!-- milestone entries appended below -->
