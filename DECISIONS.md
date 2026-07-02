# Chitti OS — Agentic Re-Architecture: Decisions & Assumptions

Append-only ledger for the autonomous run. Each entry: decision, default, rationale.
Consequential ones flagged **REVISIT**.

## Ledger (initial, from the kickoff)

1. **Sampling field repr** — keep `f32` in `Sampling`. Tests pin temp 0 (exact greedy,
   no float draw) + fixed seed, so the float-sensitive path is never exercised in the
   deterministic gates. Fixed-point (u16 milli-units) deferred. **REVISIT** if a
   postcard cross-arch parity test on a non-zero-temp manifest ever diverges.
2. **Store serialization** — `postcard` (no_std + alloc) for the persistent store;
   no `serde_json` in-kernel (tool-args JSON is handled by the existing
   `synapse::grammar` parser). Debug dumps use manual formatting.
3. **New dependencies** — `serde` (derive, alloc, no_std), `postcard` (alloc),
   `bitflags` v2 (serde feature). All no_std; none pull `std`. This crosses the
   Part-4 "heavyweight dependency" guardrail, but the schema contract in
   CHITTI_SCHEMAS.md mandates serde+postcard+bitflags verbatim, so they are adopted.
   Ed25519 crate chosen at Phase G. **REVISIT** the Ed25519 crate choice.
4. **Test model binding** — the fast `cargo xtask test` suite does not load the model.
   The agentic loop is driven by a deterministic scripted `StepSource` in tests
   (temp-0-equivalent, reproducible). Real Cortex is wired behind the same trait for
   `run`/boot demos. Mirrors the existing `Planner`/`RulePlanner` seam the handoff
   blesses. Bundled model stays Qwen3.5-0.8B by default.
5. **Delegation depth** — orchestrator `max_depth = 2`, `max_subagents = 8`; sub-agents
   default `max_depth = 0`. Matches the schema example.
6. **Loop step contract** — `trait StepSource { fn next(&mut self, &LoopCtx) -> Step }`,
   `Step = ToolCall | Final(text)`. Real impl = Cortex + grammar-constrained decode;
   test impl = scripted.
7. **Builtin tool arg schemas** — read{path}, write{path,content}, edit{path,old,new},
   list{path?}, search{query,path?}, run{intent}, spawn_subagent{role,task}, todo_write,
   load_skill{name}, emit_result{text}. MCP-shaped JSON objects, grammar-constrained.
8. **New Synapse primitives to stub** — `mem_fs_edit`, `mem_fs_search`, `todo_write`,
   `load_skill`. `spawn_subagent` routes to the agent layer (audited) rather than a
   raw primitive. All effects still flow through Synapse.
9. **Skill trust** — Ed25519 + one baked-in registry public key, single-key trust store.
   Revocable hierarchy deferred. **REVISIT** for revocation/multi-key.
10. **Sample packages** — `note-summarizer` (plain skill) + `pdf-filler` (skill-agent),
    both Ed25519-signed with the baked key, shipped as boot modules.
11. **"Both arches" gate** — `cargo xtask test` (x86_64 QEMU) green + `cargo xtask build
    -arch aarch64` clean each phase; full aarch64 boot spot-checked at milestones.
    No aarch64 in-kernel test runner exists today. **REVISIT** if one is added.
12. **persona/ deprecation** — absorb into agent/+session/+tools/ incrementally; keep it
    compiling until consumers move, then delete. Reuse `persona::compiled` for Phase E.
13. **CapId vs live caps** — `CapId(u64)` is a display/audit handle; live authority stays
    the existing unforgeable `cap::Right::InvokePrimitive` per-task tables. CapDomain/
    Rights/Scope lower to a set of `InvokePrimitive` rights at grant time.
14. **ID minting** — monotonic `AtomicU64` per id type, from 1.

## Overrides applied

(none yet — awaiting the single course-correction message.)

## Assumptions logged during the run

- **Phase C SMP concurrency (REVISIT):** `dispatch_batch` assigns each sub-agent a distinct
  core id and records it (`SubagentRecord.core`), but under the single-threaded QEMU test
  harness sub-agents execute sequentially. The prior project note already found multicore
  inference a net loss under QEMU TCG, so true parallel dispatch is structured (per-core
  assignment + isolated tasks) but not yet run concurrently. The isolation/attenuation/summary
  invariants — the correctness-critical ones — are fully enforced and tested.
- **Sub-agent cap attenuation is strict:** a role that *requests* any capability the parent does
  not hold is refused at spawn (`DispatchError::CapabilityRefused`), not silently narrowed. Rights
  are still clamped to the overlap via `intersect_caps`. This satisfies both the subset invariant
  and acceptance (b) "refused at spawn".
- **Sub-agents cannot sub-delegate by default:** the `spawn_subagent` hook gives sub-agents a
  plain Router (no spawn hook), and `max_depth` caps recursion regardless.
- **Skill-bundled tools bind to existing Synapse primitives (Phase F):** the handoff calls
  bundled code "registered as Synapse primitives, capability-gated." On bare metal we cannot
  safely load arbitrary native code at runtime, so a `BundledTool.synapse_primitive` names an
  existing, vetted primitive (the sample `note_search` → `mem_fs_search`). The tool is still a
  first-class, capability-checked, audited Synapse call; only *arbitrary new native code* is out
  of scope. REVISIT if a sandboxed skill-code execution model is added later.
- **ToolBinding::Synapse is owned (String/Vec), not &'static:** required so skill-bundled tools
  registered at runtime bind exactly like builtins.
