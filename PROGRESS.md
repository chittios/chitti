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

## Phase B — first-class tool layer  ✅

- Two new Synapse primitives (`mem_fs_edit` id=8, `mem_fs_search` id=9) so `edit`/`search`
  route through the same capability/taint-gated, audited executor as every other effect.
- `kernel/src/tools/`: `registry.rs` (MCP-shaped ToolDef catalogue — the 11 builtin tools with
  JSON input-schemas + `ToolBinding`; provider registration; per-agent `for_agent` discovery +
  `describe` for prompts), `dispatch.rs` (`Router`: the real ToolDispatch — shape-validate →
  Synapse cap+taint gate → tool_result; agent-layer bindings (spawn_subagent/load_skill/run)
  delegate to hooks installed in later phases), `provider.rs` (in-kernel "MCP server" registration,
  used by Phase F skill-bundled tools).
- Router supersedes the Phase-A inline dispatcher (SynapseTools removed); orchestrator keeps only
  the shared `synapse_call` + `to_taint` helpers. Demo/tests now use `tools::Router`.
- Gate: x86_64 `cargo xtask test` = 81/81 (5 new: malformed-rejected-before-dispatch,
  ungranted-denied+audited, write/read roundtrip+audit, todo_write updates session, discovery
  intersects toolset); aarch64 builds; live x86_64 boot demo still completes via the Router.
- Next: Phase C — sub-agents (spawn_subagent, isolation, cap attenuation, parallel, depth cap).

## Phase C — sub-agents (isolated delegation)  ✅

- `kernel/src/agent/subagent.rs`: `dispatch` (depth check → `attenuate` subset enforcement →
  isolated Session on its own cap-owning parked task → run its loop → condensed summary),
  `integrate`/`record` (summary crosses back, transcript never merged), `dispatch_batch`
  (per-core assignment, SMP-ready), `attenuate` (strict subset; refuse on widen).
- `orchestrator.router()` wires the `spawn_subagent` tool hook (enforces parent caps + depth;
  sub-agents run a rule StepSource, get a plain Router so they can't sub-delegate).
- Gate: x86_64 `cargo xtask test` = 85/85 (4 new: context-isolated, widening-cap-refused,
  two-subagents-integrate-both, depth-limit); aarch64 builds; **live x86_64 boot** shows 2
  sub-agents on cores 0/1 with isolated 5-msg transcripts, parent left with only 3 messages
  (system + 2 summaries — no sub-transcripts).
- Caveat (DECISIONS.md): SMP true-concurrency deferred under QEMU TCG; per-core structure in place.
- Next: Phase D — context compaction + todo-driven planning + session fork.

## Phase D — context management + planning  ✅

- `kernel/src/agent/context.rs`: `maybe_compact` (when live_tokens ≥ compact_threshold, evict the
  oldest non-system, non-recent turns to the store keyed `sess/<id>/cmp/<msg>`, mark them
  resident=false + store_ref, keep a summary in `ContextState.compactions`, recompute live tokens);
  `recall` (demand-page a compacted message's full text back into context). Wired into the loop —
  compaction runs after each tool turn.
- Todo-driven planning reuses `session::todo::write` (idempotent whole-list replace, returns
  remaining count) — a 5-step plan is tracked and worked down.
- Session fork reuses `session::store::fork` (new id, deep clone, independent).
- Gate: x86_64 `cargo xtask test` = 88/88 (3 new: compaction-evicts+recall-pages-back,
  5-step-task-via-todos, fork-diverges-without-mutating-parent); aarch64 builds; **live boot**
  shows compaction (162→127 tokens), recall of a compacted fact verbatim, and an independent fork.
- Next: Phase E — permission+safety (taint+cap gating in dispatch, compiled-intent replay).

## Phase E — permission + safety integration  ✅

- `agent/compiled.rs`: agent-layer compiled intents — record a validated tool-call plan keyed by
  (intent signature, file-content preconditions); `lookup` replays deterministically with ZERO
  inference when preconditions hold, `compile` caches after a Final run, stale preconditions
  re-plan. Replays still flow each call through Router→Synapse (gated + audited).
- `orchestrator.handle_compiled` (cache-first, compile-on-success) + `safe_router` (taint-aware:
  justification derived from the session's worst resident provenance; `human_confirmed` flips the
  shell-approval bit). Destructive/tainted calls hit the existing Synapse taint gate.
- Gate: x86_64 `cargo xtask test` = 92/92 (4 new: injected-destructive-gated+audited,
  confirmed-destructive-proceeds, repeated-plan-replays-without-inference, stale-precondition-
  replans); aarch64 builds; **live boot** shows the injected delete blocked (secret survives) and
  a compiled intent replaying with +0 inference.
- Next: Phase F — skill subsystem (package/index/loader, progressive disclosure) + sample skill.
