*"This project is specified in `CHITTI_OS_HANDOFF.md`. Read it fully before acting. Follow the locked decisions in Part 2 and the guardrails in Part 4. Work one phase at a time; do not start the next phase until the current phase's acceptance criteria pass in QEMU."*

**Current phase: 6 (Differentiators — taint security + self-compiling agents) — complete.**
The two features that make Chitti novel. **(1) Provenance/taint.** A new
`security/taint.rs` defines `Provenance` (`SystemTrusted | UserTyped |
UntrustedIngested`) and `Justification` (provenance + human-confirmed).
`persona::memory::Message` now carries provenance — system prompt is trusted,
a typed intent is `UserTyped`, and anything an agent *ingests* (a file it reads,
a fact it recalls) is `UntrustedIngested`; `Context::max_taint` folds a context
to its worst provenance. The Synapse executor gained a **taint gate** (a fourth
gate, after grammar/capability): `execute_with_justification` refuses a
*destructive* primitive (`mem_fs_delete`, flagged `destructive: true` in the
registry) when its justification traces to untrusted ingested content, unless a
human confirms at the shell — auditing it as the new `Outcome::RefusedTainted`.
An agent computes each call's justification from `ctx.max_taint()`, so a
prompt-injected "delete X" read out of a file cannot escalate. **(2)
Self-compiling agents.** `persona/compiled.rs` is the `/bin` analogue: on first
satisfaction of an intent it records the validated capability trace keyed by an
`(intent signature, preconditions)`, where preconditions snapshot the external
state the trace *read* (file-content / fact hashes). A later matching intent
with satisfied preconditions **replays the trace with zero inference** (a
compiled intent); a stale precondition falls back to planning and recompiles;
refused/denied/rejected runs are never compiled. `shell::run_intent` and the
interactive loop route through this cache, and the loop offers human
confirmation when the taint gate fires. `cargo xtask test` now runs 63 in-kernel
tests (up from 56): (a) an injected "delete secrets" hidden in file content is
refused by the taint gate and audited, the victim survives, yet a clean
user-justified delete still works; (b) a repeated intent's second run is a
ktrace'd cache hit with no planner (inference) call, replayed effects still
audited; (c) mutating a fact makes its compiled intent stale and it re-plans to
the fresh result. Phase 6 completes the roadmap's core; remaining work is
Phase 7 stretch (SMP, APIC-per-core, framebuffer TUI, larger model, RISC-V).

---
*Prior:* **Phase 5 (Persona — agent runtime + intent shell) — complete.**
Agents as first-class processes, and a full intent→plan→act→result loop from
a serial shell. `persona/manifest.rs`: an agent manifest (model ref, persona
prompt, capability set, memory policy). `persona/memory.rs`: two-tier memory —
a bounded live context (tier 1, the KV-cache-derived working set) and a durable
persistent store (tier 2, backed by `synapse::fs` under a per-agent namespace)
with **demand-paging / RAG-style `recall`** that pages a fact into live context
only when referenced. `persona/planner.rs`: a `Planner` trait (the stochastic
layer above the determinism boundary) with a deterministic `RulePlanner` that
maps intents to plans — the real 0.8B model is far too slow under QEMU TCG to
drive a plan in a test, and a Cortex-backed planner drops into the same seam.
`persona/actions.rs`: the plan vocabulary (Synapse tool-call JSON + memory ops),
whose `call_*` builders emit exactly the Phase 4 grammar's canonical shape.
`persona/agent.rs`: the process itself — lifecycle spawn/suspend/resume/kill,
where **suspend checkpoints only context + memory pointers + plan cursor and
drops the recomputable live/KV state, and resume recomputes (never restores)
it** — plus the plan/act loop that drives every effect through the
capability-checked, audited Synapse ABI. `shell/mod.rs`: the intent shell over
COM1 (`serial::read_byte`/`put_byte`) — `run_intent` for one-shot/tested use and
an interactive `run` read-eval loop, with an `infer` builtin that runs the
Phase 3 Cortex reference on demand. The default boot now boots into this shell.
`cargo xtask test` now runs 56 in-kernel tests (up from 45): (a) a typed intent
completes a 2-primitive plan and returns the correct read-back result (both
primitives audited); (b) suspend→resume drops then recomputes live state and the
agent continues correctly to the same result; (c) a fresh agent recalls a fact
from the persistent store that was never in its live context; (d) two agents
coordinate a split task via capability-gated IPC + a capability-checked Synapse
write. No self-compiling / taint yet (Phase 6). Next: Phase 6 (taint/provenance
gating + self-compiling agents / compiled intents).

---
*Prior:* **Phase 4 (Synapse — capability ABI) — complete.**
The syscall layer that turns untrusted model output into deterministic,
capability-checked, audited effects — the concrete enforcement of the
determinism boundary. `synapse/registry.rs`: a fixed, MCP-shaped primitive
catalogue (name + typed input schema) — `console_write`, `mem_fs_read`,
`mem_fs_write`, `list`, `spawn_agent`, `sleep`, `emit_result`, each with a
stable id that is also its `cap::Right::InvokePrimitive` discriminant.
`synapse/grammar.rs`: a GBNF-style constraint grammar *generated from* the
registry, realized as a prefix-closed recursive-descent parser over canonical
MCP-flavored JSON (`{"name":..,"arguments":{..}}`) that classifies input as
complete / viable-prefix / invalid — so the same grammar both validates a
finished call (`parse`) and, via `ConstrainedDecoder` (impl of
`cortex::sampler::Grammar`), masks a model's token stream so only well-formed
calls can be emitted. `synapse/executor.rs`: the one path from call to effect —
grammar gate → capability gate (`cap::holds`, no ambient authority) → isolated
native execution — writing exactly one append-only `synapse/audit.rs` entry
(caller, primitive, args hash, outcome, result hash) for every attempt.
`synapse/fs.rs`: the in-memory store the FS primitives mutate. `cargo xtask
test` now runs 45 in-kernel tests (up from 27): a malformed call is rejected by
the grammar and never mutates state; an uncapable call is denied + audited; a
valid call mutates the FS observably (from another task) + is logged; the audit
log is proven append-only (pre-existing entries byte-identical after a burst of
executed/denied/rejected attempts); plus grammar parse/prefix/constrained-
decoding unit tests. A model-free `synapse::demo()` runs the full ABI on every
boot (serial). No provenance/taint gating yet (Phase 6) — capability checks only.
Next: Phase 5 (Persona — agent runtime + intent shell).

---
*Prior:* **Phase 3 (Cortex — CPU inference runtime) — complete.**
SIMD (SSE2) enabled crate-wide with `fpu::enable_sse` run first thing at boot
and per-task FXSAVE/FXRSTOR across context switches; `cortex/tensor.rs` SSE2
kernels (Q4_0/Q8_0 dequant, dot/matvec, RMSNorm, RoPE, softmax, SwiGLU, L2-norm,
sigmoid/silu/softplus, no_std exp/sin/cos/ln) unit-tested against a NumPy
reference (`tools/ref.py`); a zero-copy GGUF parser (`cortex/gguf.rs`) reading
the model as a Limine boot module; and a full **Qwen3.5-0.8B hybrid** forward
pass (`cortex/model.rs`) — 18 gated-DeltaNet linear-attention/SSM layers (causal
conv1d + the recurrent gated delta rule `S = g·S + β·kᵀ(v−Sᵀk)`, `g =
exp(−exp(A)·softplus(α+dt))`, gated RMSNorm) interleaved with 6 full-attention
layers (QK-norm, partial mRoPE, GQA, sigmoid output gate), SwiGLU FFN, tied
output — reconstructed against llama.cpp's `qwen35` graph. Hybrid recurrent-state
cache (`Cache`: delta S + conv ring per SSM layer, KV history per attention
layer), a seeded/temperature/grammar-constrained sampler (`cortex/sampler.rs`),
and a continuous-batching token scheduler (`cortex/batch.rs`). `cargo xtask
test` runs the fast unit suite (Phase 0–2 + tensor kernels + sampler);
`cargo xtask ref-check` boots the real model (release) and passes the mandatory
gate: greedy parity vs the NumPy reference (token-for-token), determinism across
seeded runs, KV evict+recompute reproducibility, and 2 agents advancing in
interleaved batched passes. Model hash/seed/input hash are `ktrace`'d per
inference. Next: Phase 4 (Synapse — capability ABI over grammar-constrained
tool calls).

---
*Prior:* **Phase 2 (Execution substrate) — complete.**
Stackful kernel-mode tasks with a hand-written `switch_to` context switch (naked
function, callee-saved regs + RFLAGS saved on each task's own stack); a
round-robin scheduler entered either voluntarily (`sched::yield_now`) or by the
PIT timer once a task's slice of ticks elapses (`sched::on_timer_tick`), so the
same primitive serves both cooperative and timer-preemptive scheduling; a
minimal cooperative async executor (`sched::executor`) for yield-heavy work atop
a single stack; an unforgeable capability system (`cap`) — opaque per-task-table
indices, no ambient authority, no API that names another task's table directly;
and capability-gated IPC (`ipc`) modeled as seL4-style endpoints. `cargo xtask
test` runs 12 in-kernel tests (up from Phase 1's 7): 3 cooperatively-yielding
tasks interleave and all reach their target (checked via a transition count, not
just final counts); a non-yielding task is still forcibly preempted by the timer;
an IPC round-trip delivers a correct reply; a task holding no capability is
denied and the denial is `ktrace`'d; plus an async-executor interleaving test.
Next: Phase 3 (Cortex — CPU inference runtime, highest risk).
