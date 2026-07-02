# Chitti OS — Agentic Re-Architecture Handoff (Claude Code model)

**Mission:** Replace the flat *Persona* agent model with a **Claude-Code-style agentic architecture**: one interactive **main agent (orchestrator)** running a tool-use loop, able to dispatch isolated **sub-agents**, over a first-class **tool** layer, first-class **sessions**, and **installable skills** (portable, permissioned packages of procedural knowledge + optional code + optional agent roles). Reuse the existing Chitti substrate (Cortex, Synapse, capabilities, IPC, taint gating, compiled intents, SMP) — do not rebuild it.

Companion to `CHITTI_OS_HANDOFF.md`. Read both. This brief only covers the agent-layer re-architecture.

---

## Part 0 — Context: what already exists (do not rebuild)

The kernel is complete through the original Phase 6+ and beyond: x86_64 **and** aarch64, SMP multicore, Cortex CPU inference running a single bundled **Qwen3.5** model, Synapse capability ABI, unforgeable capabilities, IPC, taint/provenance gating, compiled-intent caching, two-tier memory. This work sits **on top** of all that.

**What changes:** the `persona/` subsystem (flat agents + peer-to-peer IPC composition) is deprecated and replaced by an `agent/` + `session/` + `tools/` architecture modeling Claude Code. Composition moves from "personas message each other as peers" to "an orchestrator delegates to sub-agents that return summaries." IPC remains as the *transport* underneath, but is no longer the composition model.

---

## Part 1 — Concept mapping (Claude Code → Chitti)

| Claude Code concept | Chitti implementation |
|---|---|
| Main agent loop | An orchestrator task running the agentic loop on Cortex; the interactive entry point behind the Intent shell. One per session. |
| Agentic loop | `model → (grammar-constrained tool_calls) → Synapse validate+execute → tool_results appended → model`, repeated until the agent emits a final result or hits a budget. |
| Sub-agent (Task/dispatch) | A spawned agent with an **isolated context** (its own Cortex KV-cache lease), **attenuated capabilities** (strict subset of parent), its own system prompt + toolset. Runs its own loop to completion, returns **only a summary** to the parent. Ephemeral. |
| Context isolation | Each agent owns a separate KV arena. Sub-agent transcript never merges into the parent — only the return value crosses. |
| Tool | A Synapse primitive wrapped with an agent-facing **MCP-shaped** definition (name, description, JSON-schema input). Grouped into toolsets, granted per agent via capabilities. |
| Tool permission / approval | Reuse Phase 6 capability + taint gating: high-privilege / destructive tools require confirmation at the shell; tainted justifications are gated. Every call audited. |
| Session | Serializable object: message history + todo list + env/cwd + capability set + memory-store handle. Persisted to the memory store; **resumable and forkable**. |
| Context compaction (auto-compact) | When context nears the KV budget, summarize old turns into the persistent store and recall on demand — demand-paging of conversation history (reuse two-tier memory + RAG recall). |
| Todo list (TodoWrite) | A structured object in the session, mutated by a `todo_write` tool; drives multi-step planning. |
| Parallel sub-agents | Dispatch independent sub-agents to idle cores via the existing SMP scheduler — real parallelism, not just interleaving. |
| MCP servers | In-kernel **tool-provider modules** that register toolsets with the registry (no network on bare metal; providers are kernel modules). |
| Hooks (pre/post tool use) | Capability-gated lifecycle callbacks on the loop. Optional (Phase H). |
| Skill | An installed package: manifest (name, description, requested capabilities, version, signature) + instruction body + optional bundled tools + optional reference assets. Stored in the memory store; registered in a skill index. |
| Progressive disclosure | Three loading tiers reusing two-tier memory: **L0** metadata (name + description) always cheap in context; **L1** instruction body loaded only when the orchestrator matches a task to the skill; **L2** bundled refs/tools demand-paged/executed on use. |
| Skill install | An explicit flow: parse package, **verify signature/hash**, show the human the requested capabilities, get consent, then register metadata + grant only the approved capability subset. Provenance tag `skill_installed`. |
| Skill-bundled code | Registered as Synapse primitives, sandboxed and capability-gated like any tool — never ambient authority, always audited. |
| Skill-agent (installable agent) | A skill whose package includes an agent definition (system prompt + toolset + skill deps + requested caps). Installing it registers a new dispatchable sub-agent role. |
| Skill registry / marketplace | A local package registry backed by the memory store; packages arrive as boot modules today (no network on bare metal). |

---

## Part 2 — Locked decisions

Do not deviate without human approval.

- **All effects still route through Synapse.** Tools are a presentation/validation layer over Synapse primitives; they never touch hardware or memory directly. This preserves the determinism boundary, capability checks, taint gating, and the audit log.
- **Capability attenuation only.** A sub-agent's capability set MUST be a subset of its parent's. Spawning can narrow authority, never widen it. Enforce in the spawn path; reject any request to grant a capability the parent lacks.
- **Context isolation is hard.** A parent never reads a sub-agent's raw context/transcript. The only channel back is the sub-agent's structured summary/result.
- **Tool-call emission stays grammar-constrained** (reuse the Cortex sampler's GBNF path). Malformed calls never reach dispatch.
- **Sessions are the unit of persistence.** A session fully serializes and resumes deterministically (modulo re-running inference, which is seeded).
- **The orchestrator is the shell's foreground process.** The Intent shell talks to the session's main agent; it does not talk to tools or sub-agents directly.
- **A skill is bounded by its install-time grant, forever.** A skill's instructions and bundled code can only exercise capabilities the human explicitly approved at install. Even if a skill's text says "delete everything," it can act only within its granted capability envelope, and destructive capabilities still hit taint gating. Instructions carried by a skill get provenance `skill_installed` — trusted enough to steer the agent, never enough to exceed the grant.
- **No install without consent + verification.** Every install verifies the package signature/hash and presents the requested capabilities to the human for explicit approval. Unsigned or unapproved packages do not register. Skill capability grants compose by intersection with the running agent's caps — a skill dispatched inside a sub-agent gets `min(skill grant, parent caps)`.

---

## Part 3 — Module layout (refactor)

```
kernel/src/
├── agent/
│   ├── loop.rs           # the agentic loop (model → tool → result → repeat), budgets, stop conditions
│   ├── orchestrator.rs   # main agent; interactive entry per session
│   ├── subagent.rs       # dispatch: spawn isolated sub-agent, run to completion, return summary
│   ├── manifest.rs       # agent definition: system prompt, toolset grant, cap set, budgets
│   └── context.rs        # per-agent context assembly + compaction triggers
├── session/
│   ├── session.rs        # messages, todos, env, cap set — serializable
│   ├── store.rs          # persist / resume / fork sessions via the memory store
│   └── todo.rs           # todo list structure + todo_write semantics
├── tools/
│   ├── registry.rs       # MCP-shaped tool definitions (name, description, JSON schema)
│   ├── dispatch.rs       # validate (grammar + capability + taint) → execute via Synapse → format tool_result
│   ├── builtin/          # read, write, edit, list, search, run, spawn_subagent, todo_write, emit_result
│   └── provider.rs       # in-kernel tool-provider modules ("MCP servers") registering toolsets
├── skills/
│   ├── package.rs        # skill package format: manifest + body + bundled tools + assets
│   ├── manifest.rs       # SKILL manifest: name, description, version, requested_capabilities, signature
│   ├── index.rs          # L0 metadata registry; description-based matching for the orchestrator
│   ├── loader.rs         # progressive disclosure: load L1 body / L2 refs+tools on demand
│   ├── install.rs        # verify signature+hash → consent prompt → grant approved caps → register
│   └── agent_skill.rs    # installable skill-agents: register a dispatchable sub-agent role
├── cortex/  synapse/  cap/  ipc/  security/   # existing — reused, not modified except integration points
└── persona/                                    # DEPRECATED — absorb into agent/, then delete
```

Keep any arch-specific bits behind the existing `arch/` split so x86_64 and aarch64 stay in sync.

---

## Part 4 — Phased sub-plan

Each phase: **Goal / Scope / Deliverable / Acceptance (QEMU) / Do-NOT-yet**. Every phase must build on both arches and pass its gate before the next.

### Phase A — Session + agentic loop core

- **Goal:** A single main agent runs a real tool-use loop inside a persistent session.
- **Scope:** the `Session` object (messages, todos, env, cap set) with serialize/resume; refactor Persona → `agent/` with `loop.rs` (model → tool → result → repeat, with turn/token/tool budgets and clean stop conditions); the orchestrator as the shell's foreground agent.
- **Deliverable:** Type an intent at the shell → the main agent loops, calls at least one tool, and returns a final answer; the session can be serialized and resumed to continue the conversation.
- **Acceptance:** (a) a multi-turn loop completes a task using ≥1 tool; (b) `session save` then `session resume` reconstructs history and the agent continues coherently.
- **Do NOT yet:** sub-agents, compaction.

### Phase B — First-class tool layer

- **Goal:** Tools as MCP-shaped, capability-granted, Synapse-backed units.
- **Scope:** `registry.rs` (tool defs: name, description, JSON schema); `dispatch.rs` (grammar-constrain the emission, then capability + taint check, then execute via Synapse, then format `tool_result` back into context); a builtin toolset — `read`, `write`, `edit`, `list`, `search`, `run` (invoke a Synapse primitive / compiled intent), `todo_write`, `emit_result`; `provider.rs` so toolsets can be registered by kernel modules.
- **Deliverable:** The main agent discovers its granted tools, emits well-formed calls, and gets structured results; unknown/ungranted/malformed calls are refused before execution.
- **Acceptance:** (a) malformed call rejected by grammar, never dispatched; (b) ungranted tool denied by capability check + audited; (c) a valid `write`+`read` round-trips through the memory FS and appears in the audit log; (d) `todo_write` updates the session todo list.
- **Do NOT yet:** sub-agent dispatch.

### Phase C — Sub-agents (the core Claude Code pattern)

- **Goal:** The orchestrator delegates to isolated sub-agents that return summaries.
- **Scope:** the `spawn_subagent` tool → `subagent.rs`: allocate an **isolated KV-cache context**, apply an **attenuated capability set** (subset check enforced), assign a system prompt + toolset, run the sub-agent's own loop to completion, capture **only its summary**, return that to the parent; support **parallel** sub-agents dispatched across cores via SMP; enforce a max delegation depth.
- **Deliverable:** The main agent splits a task, dispatches 2 sub-agents (ideally on separate cores), and integrates their summaries — without their transcripts polluting the main context.
- **Acceptance:** (a) sub-agent context is provably isolated (parent context contains the summary, not the sub-agent's turns); (b) a sub-agent requesting a capability the parent lacks is refused at spawn; (c) two sub-agents run concurrently on different cores and both results are integrated; (d) depth limit prevents runaway recursion.
- **Do NOT yet:** auto-compaction, hooks.

### Phase D — Context management + planning

- **Goal:** Long sessions stay within budget; multi-step work is todo-driven.
- **Scope:** auto-compaction in `context.rs` — when a context nears its KV budget, summarize older turns into the persistent store and recall on demand (reuse two-tier memory / RAG); a todo-driven planning loop where the orchestrator maintains and works down a todo list; session **fork** (branch a session and explore without mutating the original).
- **Deliverable:** A session that exceeds the raw context budget keeps working correctly via compaction; the orchestrator plans a multi-step task through the todo list and completes it.
- **Acceptance:** (a) a session driven past the KV budget continues coherently, with compaction events in ktrace and recall pulling back a compacted fact; (b) a 5+ step task is tracked and completed via the todo list; (c) a forked session diverges without altering the parent session.
- **Do NOT yet:** external providers, hooks (Phase H).

### Phase E — Permission + safety integration

- **Goal:** The Claude-Code approval model, backed by your existing gating.
- **Scope:** wire Phase 6 taint + capability gating into tool dispatch as the **approval layer** — destructive/high-privilege tools require confirmation at the shell; any tool call whose justification traces to `untrusted_ingested` content is gated regardless of phrasing; integrate **compiled intents** so a repeated, approved tool-plan replays deterministically and skips inference; full audit of approvals, denials, and cache hits.
- **Deliverable:** A safe, fast agent: injected instructions can't escalate to destructive tools; repeated approved workflows run without inference.
- **Acceptance:** (a) injection test — a file whose contents say "delete everything" does not fire a destructive tool; the gate triggers; (b) a repeated approved tool-plan's second run is a ktrace'd compiled-intent hit with no inference; (c) high-privilege tool prompts for confirmation and proceeds only on explicit approval; (d) all approval decisions audited.
- **Do NOT yet:** hooks/external providers unless time permits.

### Phase F — Skill subsystem (progressive disclosure)

- **Goal:** Skills as portable packages the agent can discover and load, without bloating context.
- **Scope:** the skill **package format** (`package.rs`): a manifest (`manifest.rs`: name, description, version, requested_capabilities, signature) + an instruction body + optional bundled tools + optional reference assets; a skill **index** (`index.rs`) holding L0 metadata for every installed skill and matching skills to a task by description; a **loader** (`loader.rs`) implementing progressive disclosure — L1 body loaded into context only on match, L2 refs/bundled tools demand-paged/executed on use (reuse the Phase D two-tier memory path); wire skill selection into the orchestrator loop (a `load_skill` tool, or automatic metadata injection). Bundled tools register through `provider.rs` as normal Synapse-backed, capability-gated tools.
- **Deliverable:** With a skill pre-loaded into the memory store, the orchestrator matches a relevant task, loads the skill body, follows its procedure using a bundled tool, and completes the task — and an *unrelated* task loads none of it (context stays lean).
- **Acceptance (QEMU, both arches):** (a) only L0 metadata is in context until a matching task triggers L1 load, verified via ktrace; (b) the skill's bundled tool executes only through Synapse with a capability check; (c) an unrelated task never loads the skill body; (d) a skill referencing an L2 asset pulls it on demand, not up front.
- **Do NOT yet:** installation/consent (Phase G) — for this phase skills are placed directly in the store as trusted.

### Phase G — Skill & agent installation (permissioned)

- **Goal:** Users can install skills and skill-agents safely from packages.
- **Scope:** the **install flow** (`install.rs`): parse a package delivered as a boot module (or placed in the store), **verify its signature/hash** against a trust store, present the **requested capabilities** to the human at the shell, and on explicit consent register the skill metadata and **grant only the approved capability subset**; tag all skill-carried instructions with provenance `skill_installed`; **installable skill-agents** (`agent_skill.rs`) — a package whose manifest includes an agent definition registers a new dispatchable sub-agent role, whose effective caps at dispatch are `min(install grant, parent caps)`; an uninstall path that revokes caps and de-registers. Everything audited.
- **Deliverable:** Install a skill and a skill-agent from packages with an at-install permission prompt; the orchestrator can then use the skill / dispatch the skill-agent; a malicious package cannot exceed what was approved.
- **Acceptance (QEMU, both arches):** (a) an unsigned/tampered package is refused at install; (b) install shows the requested caps and registers only the approved subset; (c) a skill whose body instructs a capability it was not granted is blocked at Synapse and audited — provenance `skill_installed` does not bypass the grant; (d) a dispatched skill-agent's effective caps are the intersection with the parent's, never wider; (e) uninstall revokes caps and the skill no longer loads.
- **Do NOT yet:** networked skill distribution (no network on bare metal) — packages are local/boot-module only.

### Phase H — Stretch

- Pre/post-tool-use hooks (capability-gated callbacks); richer in-kernel tool-provider modules ("MCP servers"); many concurrent sessions across cores; sub-agent result caching; a networked skill registry once a network stack exists. Human picks priorities.

---

## Part 5 — Build & verify

Reuse the existing `cargo xtask` flow (`build`, `image`, `run`, `test`, `ref-check`) for **both** `x86_64-chitti` and the aarch64 target. Every phase gate runs in QEMU on both arches. Keep temp 0 + fixed seeds for deterministic tests; sub-agent and skill-install tests assert via serial + audit log. Ship a couple of signed sample skill packages (one plain skill, one skill-agent) as boot modules so the Phase F/G gates have something real to load and install.

---

## Part 6 — One-shot autonomous run prompt

Paste this once, then sleep. Its first response is a decisions ledger **and** the start of the build; your single next message is the only window to override anything; after that it runs to completion without stopping for you.

```
You are implementing the entire Chitti OS agentic re-architecture autonomously, overnight, with no human present. First read CHITTI_OS_HANDOFF.md, CHITTI_AGENTIC_HANDOFF.md, and CHITTI_SCHEMAS.md IN FULL. Do not skim.

OPERATING MODE — read carefully, this governs everything:
- This is a single-shot autonomous run. I am going to sleep. There is no interactive back-and-forth.
- In THIS first response only: (1) confirm you read all three docs; (2) output a numbered "Decisions & Assumptions" ledger covering EVERY open choice, each with the concrete default you are adopting right now and a one-line rationale; (3) then immediately begin Phase A and keep building. Do NOT stop and wait for me after the ledger.
- My next message is your ONLY course-correction window. I may override any ledger item; apply those overrides and continue. After that message, never surface decisions or ask questions again — log any further assumptions to DECISIONS.md and keep going to completion.
- Never idle waiting for me. If you hit an unspecified choice at any point, pick the most reversible reasonable option, record it in DECISIONS.md (mark "REVISIT" if consequential), and proceed. Halting to ask is a failure of this run.

EXECUTION RULES:
- Implement Phases A→G in order (Part 4). Do H/stretch only if A→G are all green and time remains.
- Honor every locked decision in Part 2 and the three invariants: all effects route through Synapse; delegation only ever narrows authority; an installed skill is bounded by its install-time grant forever.
- Use the exact types in CHITTI_SCHEMAS.md as the shared contract. Do not redesign them; if one is genuinely insufficient, extend it, bump its schema_version, and log why in DECISIONS.md.
- Do NOT cross a phase gate until that phase's acceptance criteria pass in QEMU on BOTH x86_64 and aarch64. Tests use temp 0 + fixed seeds.
- Keep the tree building at every commit. Commit per sub-milestone with clear messages. Never leave the tree broken across a commit.
- Ship two signed sample skill packages (a plain skill + a skill-agent) as boot modules so the F/G gates run against real packages.

LEAVE ME A TRAIL (this is how I review the run when I wake):
- PROGRESS.md — append a short entry per milestone: phase, what landed, gate status (pass/fail on each arch), next step.
- DECISIONS.md — every assumption made and every override applied, with rationale; consequential ones flagged REVISIT.
- If a phase gate cannot be made green after real effort, log the blocker in PROGRESS.md, implement the best partial that still builds, and MOVE ON to the next independent work rather than halting.

DECISIONS I ALREADY EXPECT IN YOUR LEDGER (not exhaustive — add every other choice you find):
- Sampling fields: keep f32 or switch to fixed-point (u16 milli-units) for byte-identical postcard across x86_64/aarch64.
- Skill package signature/trust scheme: minimal Ed25519 + baked-in key vs a small revocable trust hierarchy.
- The specific tiny model + tokenizer the Cortex/agent tests bind to.
- Confirm postcard for store serialization (JSON for debug).
- Default delegation depth (recommend 1–2).
- Concrete arg JSON-schemas for the builtin toolset (read/write/edit/list/search/run/spawn_subagent/todo_write/load_skill/emit_result).
- Any tool/primitive the schemas imply that Synapse doesn't yet expose — list it and stub it.

Begin now: read the three docs, emit the ledger, then start Phase A and keep going.
```

---

*End of brief. The three invariants that make this Claude-Code-like AND safe: every effect flows through Synapse; delegation only ever narrows authority; and an installed skill is bounded by its install-time grant forever. Protect all three.*
