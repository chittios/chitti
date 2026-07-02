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

## Milestone 0 — shared type contract  ✅

- Added deps: serde (derive/alloc/no_std), postcard (alloc), bitflags v2 (serde) — all compile
  under `-Z build-std` on both arches, none pull std.
- `kernel/src/agent/types.rs`: the full CHITTI_SCHEMAS.md contract — id newtypes + monotonic
  minters, `Provenance` (+join/is_untrusted), `CapDomain`/`Rights`(bitflags)/`Scope`/
  `CapabilityRequest` (+`contains`, `intersect_caps` attenuation), `AgentManifest`, `Session`
  (+ all sub-structs), `SkillManifest`+`InstallRecord`.
- Deviation: dropped the schema's `#[serde(tag/content)]` on Provenance/Scope/Origin/
  InstallSource — internally/adjacently-tagged enums need `deserialize_any`, unsupported by
  postcard (non-self-describing). Externally-tagged instead; postcard is the canonical format,
  JSON is debug-only. schema_version stays 1 (no field meaning changed). Logged DECISIONS.md.
- Gate: x86_64 `cargo xtask test` = 73/73 (4 new: postcard roundtrip, provenance/scope survive,
  taint join, cap attenuation narrows-never-widens). aarch64 builds clean.
- Next: Phase A — Session object + agentic loop core with the StepSource seam.
