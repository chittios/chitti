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

## Phase A — Session + agentic loop core  ✅

- `kernel/src/session/`: Session construction + message/token bookkeeping (`session.rs`),
  persist/resume/fork over the memory store via postcard (`store.rs`), todo list + `todo_write`
  (`todo.rs` incl. a minimal JSON field extractor).
- `kernel/src/agent/agent_loop.rs`: the loop `model → tool_calls → Synapse → tool_results →
  repeat` with `max_turns`/`max_tool_calls` budgets + clean stop conditions, over two seams —
  `StepSource` (next Step) and `ToolDispatch` (execute one call, all effects via Synapse).
- `agent/manifest.rs`: builtin orchestrator + reader-subagent roles; lowers declarative
  CapabilityRequests to live `cap::Right::InvokePrimitive` grants (`grant_to_task`).
- `agent/orchestrator.rs`: the foreground main agent (`spawn`/`from_session`/`handle`/`kill`)
  + `SynapseTools` dispatcher (write/read/list/delete/console/emit_result → Synapse;
  todo_write session-local). Justification is trusted in Phase A; taint-aware flag ready for E.
- `agent/rule_steps.rs`: deterministic `StepSource` (ScriptedSteps + `for_intent`) — the
  model stand-in for tests/boot demo.
- Kernel change: `sched::spawn_parked` — cap-owning agent identity tasks are created (cap table
  live) but NOT enqueued, so they never steal a scheduler turn. This fixed a regression where the
  phase5 IPC test's cooperative yield loop scheduled leftover agent tasks. Reusable by Phase C.
- Gate: x86_64 `cargo xtask test` = 76/76 (3 new: loop-completes-with-tool, save/resume+continue,
  turn-budget-stops); aarch64 builds; **live x86_64 QEMU boot** shows the loop (stop=Final,
  turns=3, tool_calls=2) + save→resume(7 msgs)→continue(11 msgs).
- Next: Phase B — first-class MCP-shaped tool layer (registry + dispatch + builtin toolset).
