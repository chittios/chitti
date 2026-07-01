# Chitti OS — Claude Code Handoff Brief

**Mission:** Build a bare-metal operating system (x86_64, no host OS) whose fundamental unit of execution is an **AI agent**, not a compiled binary. Traditional `exec(binary) → trap into syscalls` is replaced by `spawn(agent) → plan over capabilities → deterministic executor runs the primitives`.

This document is the single source of truth. It is written to be handed to Claude Code. Read it fully before writing any code. Part 9 contains ready-to-paste kickoff prompts, one per phase.

---

## Part 0 — How to use this document with Claude Code

1. Place this file at the repo root as `CHITTI_OS_HANDOFF.md`, and also create a short `CLAUDE.md` that says: *"This project is specified in `CHITTI_OS_HANDOFF.md`. Read it fully before acting. Follow the locked decisions in Part 2 and the guardrails in Part 4. Work one phase at a time; do not start the next phase until the current phase's acceptance criteria pass in QEMU."*
2. Start each work session by pasting the phase kickoff prompt from Part 9.
3. One phase = potentially many sessions. Never skip a phase's acceptance gate.
4. When a locked decision (Part 2) needs to change, stop and ask the human — do not change it unilaterally.

---

## Part 1 — Mission, constraints, non-goals

### Constraints (hard)

- **Bare metal.** No Linux, no host kernel, no libc. `#![no_std]`, `#![no_main]`.
- **CPU-only inference.** No GPU/NPU. All model math runs on the CPU with SIMD. This is the single most important constraint — do not design around a GPU that will never exist here.
- **Tiny model only.** Target a 0.5B–1.1B parameter model at Q4_0 or Q8_0 (e.g. a TinyLlama / Qwen-0.5B-class GGUF). Anything larger will not fit or will be unusably slow. The model is loaded as a boot module, not compiled in.
- **Determinism below the boundary.** Everything from the capability ABI down must be deterministic and reproducible. Stochastic behavior is confined to the agent/model layer.
- **Every phase must boot and pass its QEMU smoke test before the next phase begins.**

### Non-goals (explicitly out of scope, do NOT build)

- GPU acceleration, CUDA/Vulkan, any GPU driver.
- Networking stack (deferred; agents run offline against local model + local capabilities).
- A POSIX layer, ELF loader, or ability to run conventional binaries. Chitti runs agents, not programs.
- A GUI/window system. Console (serial + framebuffer text) only.
- SMP/multicore in the early phases (single-core until Phase 7 stretch).
- Persistent disk filesystem on real hardware (use an in-memory FS + boot modules; real block drivers are a later concern).

---

## Part 2 — Locked technical decisions

Do not deviate from these without human approval.

| Concern | Decision |
|---|---|
| Language | Rust, **nightly**, `#![no_std]` + `#![no_main]`. Pin via `rust-toolchain.toml` (nightly, components: `rust-src`, `llvm-tools-preview`). |
| Architecture | **x86_64** primary. (RISC-V `virt` is a clean alternative if the human later chooses to port — keep arch-specific code isolated behind an `arch/` module to make that possible.) |
| Target | Custom target JSON at `targets/x86_64-chitti.json`, derived from `x86_64-unknown-none`. Build with `-Z build-std=core,compiler_builtins,alloc`. SIMD target-features stay **off** until Phase 3, then enabled deliberately alongside FPU/XSAVE init. |
| Bootloader | **Limine** (stable boot protocol, framebuffer, memory map, boot modules for the model file, SMP-ready). Use the `limine` crate for the request structures. *(Faster-start alternative: the pure-Rust `bootloader` 0.11 crate — acceptable for Phase 0 only if Limine setup blocks progress; note the switch and migrate to Limine by Phase 1.)* |
| Emulator / test harness | **QEMU** (`qemu-system-x86_64`). Serial (`-serial stdio`) captured for assertions. `isa-debug-exit` device (`iobase=0xf4`) for test exit codes. `custom_test_frameworks` for in-kernel `cargo test` running under QEMU. |
| Build orchestration | A `cargo xtask` crate for image assembly (kernel + Limine + `model.gguf` module → bootable image) and QEMU launch. All commands go through `cargo xtask <cmd>`. |
| Inference reference | Numerics for the transformer must be validated against a host reference (llama.cpp or a small PyTorch/NumPy script) on the **same** tiny model + same prompt + fixed seed, logits within a stated tolerance. This reference check is mandatory for Phase 3. |
| Capability ABI wire shape | Model emits tool calls as structured JSON constrained by a GBNF-style grammar. Schemas follow the **MCP tool** shape (name + JSON-schema input) so the ABI is familiar and portable. |

### Chitti vocabulary (use these names in code)

| Chitti primitive | Classic OS analogue |
|---|---|
| **Intent** | a command line |
| **Persona** (agent runtime) | process |
| **Synapse** (capability ABI) | syscall interface |
| **Cortex** (inference runtime) | CPU scheduler + MMU, for tokens |
| **Capability** | fd / permission token (unforgeable) |
| **Primitive** | individual syscall |
| **Memory store** | filesystem / backing store |
| **Compiled intent** | cached, deterministic capability trace ≈ a binary |

### The determinism boundary (the load-bearing rule)

Model output is an **untrusted plan**. It never causes a side effect directly. It is always parsed, validated against a grammar and a capability check, and only then executed by deterministic native code (`Synapse`). Above the boundary: stochastic (Persona, Intent shell). Below: deterministic (Synapse, Cortex, microkernel, silicon).

---

## Part 3 — Repository layout

```
chitti/
├── CLAUDE.md                     # points Claude Code at this brief
├── CHITTI_OS_HANDOFF.md          # this file
├── rust-toolchain.toml
├── targets/x86_64-chitti.json
├── xtask/                        # build image + run QEMU + tests
├── kernel/
│   ├── src/
│   │   ├── main.rs               # _start entry, panic handler
│   │   ├── arch/x86_64/          # GDT, IDT, interrupts, paging, SIMD/FPU init, ports
│   │   ├── mm/                   # frame allocator, paging, heap
│   │   ├── sched/                # tasks, context switch, scheduler, async executor
│   │   ├── cap/                  # capability tables + tokens (unforgeable)
│   │   ├── ipc/                  # message passing between tasks
│   │   ├── cortex/               # inference runtime
│   │   │   ├── gguf.rs           # GGUF parser
│   │   │   ├── tensor.rs         # SIMD quantized matmul, rmsnorm, rope, softmax
│   │   │   ├── model.rs          # transformer forward pass
│   │   │   ├── kv.rs             # KV cache as a paged resource
│   │   │   ├── sampler.rs        # seeded sampling, temperature, grammar-constrained
│   │   │   └── sched.rs          # continuous-batching token scheduler
│   │   ├── synapse/              # capability ABI: primitive registry, grammar, executor, audit log
│   │   ├── persona/              # agent manifest, lifecycle, two-tier memory
│   │   ├── shell/                # intent shell over serial/console
│   │   ├── security/             # provenance/taint tags + gating
│   │   ├── ktrace.rs             # deterministic logging / the strace equivalent
│   │   └── serial.rs, framebuffer.rs, logging.rs
│   └── tests/                    # integration tests run in QEMU
└── assets/model.gguf             # tiny quantized model, loaded as a Limine boot module
```

Keep everything x86-specific under `arch/x86_64/` so a future RISC-V port only touches that directory.

---

## Part 4 — Global engineering conventions & guardrails

**Conventions**

- No `unsafe` without an adjacent `// SAFETY:` comment justifying every invariant.
- Every public module gets a doc comment stating its responsibility and its place in the layer stack.
- Deterministic by default: tests use fixed seeds and temperature 0. Any RNG is seeded and logged.
- Allocation-aware: below Phase 1's heap, code must be allocation-free. Above it, prefer `alloc` collections but watch fragmentation for large tensors/KV buffers.
- `ktrace` every capability invocation and every inference call (model hash, seed, input hash) from the moment those subsystems exist.
- Commit per sub-milestone with a clear message; never leave the tree non-building across a commit.

**Guardrails — stop and ask the human before:**

- Changing any locked decision in Part 2 (target, bootloader, CPU-only inference, model size).
- Adding a heavyweight dependency (anything large, anything that pulls `std`, anything doing its own allocation you can't audit).
- Restructuring the capability/security model.
- Growing the model size or reaching for a GPU path.

**Definition of "done" for any phase:** builds clean on nightly for `x86_64-chitti`, boots in QEMU, its acceptance criteria all pass via `cargo xtask test` (or the phase's stated check), and `ktrace`/serial output demonstrates the deliverable. Update `CLAUDE.md`'s "current phase" line.

---

## Part 5 — Phase roadmap

| Phase | Name | One-line goal | Risk |
|---|---|---|---|
| 0 | Boot & harness | Boot via Limine, print to serial + framebuffer, QEMU test exit codes | Low |
| 1 | Microkernel core | Interrupts, memory (frames/paging/heap), timer, keyboard, ktrace | Medium |
| 2 | Execution substrate | Tasks, scheduler, capabilities, IPC, async executor | Medium |
| 3 | Cortex (inference) | CPU transformer forward pass on a tiny GGUF, KV cache, seeded sampling | **Very high — ~60% of total effort** |
| 4 | Synapse (capability ABI) | Grammar-constrained tool calls → capability-checked deterministic primitives + audit log | Medium |
| 5 | Persona + shell | Agents as processes, two-tier memory, intent shell drives a full plan→act loop | High |
| 6 | Differentiators | Taint/provenance security gating + self-compiling agents (compiled intents) | High |
| 7 | Stretch | SMP, APIC-per-core, framebuffer TUI, larger model, RISC-V port | — |

---

## Part 6 — Detailed phase specs

Each phase lists **Goal / Scope / Deliverable / Acceptance / Do-NOT-yet**.

### Phase 0 — Boot & harness

- **Goal:** A bootable kernel that prints and can be tested.
- **Scope:** `rust-toolchain.toml`, `targets/x86_64-chitti.json`, `xtask` that assembles a Limine image and boots it in QEMU; `_start` entry; panic handler; serial writer; read Limine framebuffer + memory-map responses; `isa-debug-exit` wiring; `custom_test_frameworks` harness.
- **Deliverable:** Boots in QEMU, prints "Chitti: boot ok" to serial and to the framebuffer, exits QEMU with a success code from a test.
- **Acceptance:** `cargo xtask run` shows the boot line; `cargo xtask test` runs at least one in-kernel test that exits QEMU with the success code.
- **Do NOT yet:** interrupts, allocation, anything agentic.

### Phase 1 — Deterministic microkernel core

- **Goal:** A stable kernel with interrupts and memory management.
- **Scope:** GDT + TSS (with IST for double-fault); IDT with CPU exception handlers; APIC (or PIC fallback) + timer + keyboard IRQs; physical frame allocator built from the Limine memory map; 4-level paging and virtual address-space setup; kernel heap allocator (linked-list to start, buddy if fragmentation bites); `ktrace` logging framework; enable FPU/SSE via CR0/CR4 + XSAVE **scaffolding** (SIMD features still compiled off until Phase 3, but the FPU state is initialized correctly).
- **Deliverable:** Kernel handles timer ticks and keyboard input, allocates/frees heap memory, survives and cleanly reports a deliberately triggered exception, and logs via `ktrace`.
- **Acceptance:** QEMU tests: (a) timer increments a counter over N ticks; (b) heap alloc/free of varied sizes with no corruption; (c) a forced page-fault/breakpoint is caught and reported, not a triple-fault reboot.
- **Do NOT yet:** tasks/scheduling, inference.

### Phase 2 — Execution substrate

- **Goal:** Concurrency + the security substrate.
- **Scope:** Task/thread abstraction with context switching; a scheduler (start cooperative, add timer-preemption); a cooperative **async executor** (agent work is yield-heavy); **capability system** — unforgeable capability tokens (seL4-inspired), a per-task capability table, no ambient authority; **IPC** message passing between tasks, gated by capabilities.
- **Deliverable:** Several kernel tasks run concurrently, exchange IPC messages, and a task can only perform an operation if it holds the matching capability.
- **Acceptance:** QEMU tests: (a) 3+ tasks interleave and all make progress; (b) an IPC round-trip delivers a message; (c) a task lacking a capability is denied the gated operation and the denial is `ktrace`d.
- **Do NOT yet:** hook the scheduler to inference batching (that lands in Phase 3), taint tracking (Phase 6).

### Phase 3 — Cortex: CPU inference runtime  ⚠ highest risk

- **Goal:** Generate tokens from a tiny model, deterministically, on the CPU, with the OS managing the KV cache.
- **Scope:**
  - `gguf.rs`: parse the GGUF `model.gguf` loaded as a Limine boot module (header, metadata, tensor table); mmap tensors from the module memory.
  - `tensor.rs`: no_std SIMD kernels — dequant + matmul for Q4_0 and Q8_0, RMSNorm, RoPE, softmax, SiLU. **Enable SIMD now:** turn on target-features (SSE2/AVX2) in the target JSON, confirm XSAVE/FPU init from Phase 1 is correct, and use `core::arch::x86_64` intrinsics.
  - `model.rs`: a minimal Llama/Qwen-style decoder-only forward pass.
  - `kv.rs`: KV cache as a **paged resource** allocated through the Phase 1 allocator — allocate, grow, evict; treat VRAM-scarcity analogue as heap-scarcity.
  - `sampler.rs`: seeded sampling, temperature (0 for tests), and **grammar-constrained decoding** (GBNF-style) so output can be forced into valid tool-call shapes later.
  - `sched.rs`: begin single-agent single-threaded; then a continuous-batching token scheduler that plugs into the Phase 2 scheduler (one forward pass advances several agents a few tokens each; token budget = priority).
  - Deterministic + logged: record model hash, seed, and input hash per inference via `ktrace`.
- **Deliverable:** Given an embedded prompt, the kernel emits a coherent, **reproducible** token stream from the tiny model; the KV cache is managed by Cortex; batching advances 2+ agents in one pass.
- **Acceptance:** (a) **Reference parity** — greedy (temp 0, fixed seed) logits/first-N tokens match the host llama.cpp/NumPy reference on the same model+prompt within the stated tolerance; (b) determinism — same seed/prompt yields identical output across runs; (c) KV eviction + recompute produces identical continuation; (d) batching test shows 2 agents progressing in interleaved forward passes.
- **Do NOT yet:** tool calling / side effects (Phase 4), agent lifecycle (Phase 5). Cortex only produces tokens here.
- **Note to Claude Code:** build `tensor.rs` bottom-up and unit-test each kernel against the host reference *before* assembling the full forward pass. Do not attempt the whole transformer in one pass. This phase carries the project.

### Phase 4 — Synapse: capability ABI (the new syscall layer)

- **Goal:** Turn validated model output into deterministic, capability-checked effects.
- **Scope:** a primitive **registry** (MCP-shaped: name + JSON-schema input) covering an initial set — `console_write`, `mem_fs_read`, `mem_fs_write`, `list`, `spawn_agent`, `sleep`, `emit_result`; a grammar generator that constrains the model to emit only registered, well-formed calls (ties to `sampler.rs`); a **deterministic executor** that runs each primitive in an isolated/jailed context; a **capability check** on every call (ties to Phase 2 caps) — an agent can only invoke primitives it holds capabilities for; an **append-only audit log** of every invocation (caller, primitive, args hash, result, timestamp).
- **Deliverable:** A test agent emits a tool call; the kernel validates grammar + capability, executes the primitive deterministically, logs it, and returns a structured result the model can consume next turn.
- **Acceptance:** (a) a malformed call is rejected by the grammar and never reaches the executor; (b) a call to a primitive the agent lacks the capability for is denied and audited; (c) a valid call mutates the in-memory FS and the change is observable + logged; (d) the audit log is append-only (past entries immutable).
- **Do NOT yet:** provenance/taint gating (Phase 6) — capability checks only for now.

### Phase 5 — Persona: agent runtime + intent shell

- **Goal:** Agents behave like processes; a full intent → plan → act → result loop runs from the shell.
- **Scope:** **Agent manifest** (model ref, persona/system prompt, capability set, memory policy); **lifecycle** — spawn, suspend (checkpoint *context + memory pointers*, not the multi-hundred-MB KV cache — recompute on resume), resume, kill; **two-tier memory** — KV cache (RAM, via Cortex) + a persistent memory store (the "disk", in-memory FS backed) with demand-paging / RAG-style recall when the agent references something not in context; the **intent shell** over serial/console (type an intent → route to / spawn an agent); agent-to-agent IPC via Phase 2 IPC + capabilities.
- **Deliverable:** At the shell you type an intent (e.g. "write a file called notes with the text hello, then read it back"); an agent plans, calls Synapse primitives, and reports the result — end to end.
- **Acceptance:** (a) a typed intent completes a 2–3 primitive plan and returns the correct result; (b) suspend→resume reconstructs an agent's working state and it continues correctly; (c) an agent recalls a fact from the persistent store that was not in its live context; (d) two agents coordinate via IPC to complete a split task.
- **Do NOT yet:** self-compiling / taint (Phase 6).

### Phase 6 — Differentiators: taint security + self-compiling agents

- **Goal:** The two features that make Chitti novel rather than a re-skin.
- **Scope:**
  - **Provenance/taint tags** on every context token: `user_typed`, `system_trusted`, `untrusted_ingested` (anything read from FS/tool results). `Synapse` gates high-privilege primitives (destructive/irreversible) on the provenance of the tokens that justified the call: if a destructive action traces to `untrusted_ingested` content, **refuse or require explicit human confirmation at the shell**, regardless of how the agent phrased it. This is prompt-injection-as-privilege-escalation defense enforced at the OS boundary.
  - **Self-compiling agents:** on first satisfaction of an intent, record the validated capability trace keyed by (intent signature, preconditions). On a later matching intent with satisfied preconditions, **replay the trace deterministically and skip inference entirely** ("compiled intent"). On precondition miss, fall back to planning. Maintain a "compiled intents" store = the `/bin` analogue.
- **Deliverable:** Repeated intents execute with zero inference (fast, deterministic, audited); an injected instruction hidden inside ingested file content cannot escalate to a destructive primitive without human confirmation.
- **Acceptance:** (a) an injection test — a file whose contents say "delete everything" — does NOT cause a destructive primitive to fire on tainted grounds; the gate triggers; (b) a repeated intent's second run shows a `ktrace` cache hit and no inference call; (c) a compiled intent whose precondition now fails correctly falls back to re-planning; (d) all of the above are fully audited.

### Phase 7 — Stretch

- SMP bring-up (APIC per core, per-core run queues, lock discipline); framebuffer text UI beyond serial; a larger/faster model or better quantization; block-device + real FS; a RISC-V `virt` port (should touch only `arch/`). Pick per the human's interest.

---

## Part 7 — Build & test commands (implement in `xtask`)

```
cargo xtask build            # build kernel for x86_64-chitti (nightly, build-std)
cargo xtask image            # assemble Limine image: kernel + limine + assets/model.gguf as a boot module
cargo xtask run              # boot the image in QEMU, serial to stdio
cargo xtask test             # run in-kernel tests under QEMU, assert via serial + isa-debug-exit
cargo xtask ref-check        # (Phase 3+) compare Cortex logits/tokens to the host reference
```

QEMU baseline flags: `-machine q35 -m 2G -serial stdio -device isa-debug-exit,iobase=0xf4,iosize=0x04 -no-reboot -no-shutdown`. Bump `-m` if the model needs more RAM, but keep the model tiny per Part 1.

---

## Part 8 — Definition of "coherent progress"

At the end of every session Claude Code should be able to answer, in the commit message or a short note: which phase, which sub-milestone, does it build, does it boot, which acceptance criteria now pass, and what's next. If a phase's acceptance gate isn't green, the next phase does not start.

---

## Part 9 — Ready-to-paste phase kickoff prompts

Paste one of these at the start of a session. Each assumes the repo already contains this brief.

### Phase 0

```
You are building Chitti OS. Read CHITTI_OS_HANDOFF.md fully, then implement Phase 0 (Boot & harness).
Honor all locked decisions in Part 2 and the guardrails in Part 4. Scope: rust-toolchain.toml, targets/x86_64-chitti.json, an xtask crate that assembles a Limine image and boots it in QEMU, a no_std/no_main kernel with _start + panic handler + serial writer, reading the Limine framebuffer and memory-map responses, isa-debug-exit wiring, and a custom_test_frameworks harness.
Definition of done: `cargo xtask run` prints "Chitti: boot ok" to serial and framebuffer; `cargo xtask test` runs at least one in-kernel test that exits QEMU with the success code. Do not implement interrupts, allocation, or anything agentic yet.
Start by proposing the file tree and the target JSON, then implement. Ask before changing any locked decision.
```

### Phase 1

```
Continue Chitti OS. Phase 0 is green. Read CHITTI_OS_HANDOFF.md, then implement Phase 1 (Deterministic microkernel core): GDT+TSS with IST, IDT + exception handlers, APIC/timer/keyboard IRQs, a physical frame allocator from the Limine memory map, 4-level paging, a kernel heap allocator, the ktrace logging framework, and correct FPU/SSE (CR0/CR4/XSAVE) initialization scaffolding — but keep SIMD target-features OFF until Phase 3.
Definition of done (QEMU tests): timer increments a counter over N ticks; heap alloc/free of varied sizes with no corruption; a deliberately triggered exception is caught and reported, not a triple fault. Do not build tasks/scheduling or inference yet.
```

### Phase 2

```
Continue Chitti OS. Phase 1 is green. Read CHITTI_OS_HANDOFF.md, then implement Phase 2 (Execution substrate): a task abstraction with context switching, a scheduler (cooperative first, then timer-preemptive), a cooperative async executor, an unforgeable capability system (per-task capability tables, no ambient authority), and capability-gated IPC message passing.
Definition of done (QEMU tests): 3+ tasks interleave and all progress; an IPC round-trip delivers a message; a task lacking a capability is denied and the denial is ktrace'd. Do not wire inference batching or taint tracking yet.
```

### Phase 3

```
Continue Chitti OS. Phase 2 is green. Read CHITTI_OS_HANDOFF.md, then implement Phase 3 (Cortex, CPU inference) — the highest-risk phase; go bottom-up.
First enable SIMD target-features and confirm FPU/XSAVE init. Then implement and UNIT-TEST each tensor kernel (Q4_0/Q8_0 dequant+matmul, RMSNorm, RoPE, softmax, SiLU) against a host reference (llama.cpp or NumPy) on the same tiny model BEFORE assembling the forward pass. Then: GGUF parsing of the model loaded as a Limine boot module, a minimal Llama/Qwen decoder-only forward pass, KV cache as a paged resource via the Phase 1 allocator, a seeded sampler with temperature and grammar-constrained decoding, and a continuous-batching token scheduler plugged into the Phase 2 scheduler. Log model hash/seed/input per inference.
Definition of done: greedy temp-0 output matches the host reference within the stated tolerance; identical output across runs for the same seed; KV evict+recompute reproduces the continuation; batching advances 2 agents in interleaved passes. Do NOT add tool calls or side effects yet — Cortex only produces tokens.
```

### Phase 4

```
Continue Chitti OS. Phase 3 is green. Read CHITTI_OS_HANDOFF.md, then implement Phase 4 (Synapse, capability ABI): an MCP-shaped primitive registry (console_write, mem_fs_read, mem_fs_write, list, spawn_agent, sleep, emit_result), a grammar generator constraining the model to emit only registered well-formed calls, a deterministic executor running each primitive in an isolated context, a capability check on every call, and an append-only audit log.
Definition of done: a malformed call is rejected by the grammar and never executes; a call lacking the capability is denied and audited; a valid call mutates the in-memory FS observably and is logged; the audit log is append-only. Do not add provenance/taint gating yet — capability checks only.
```

### Phase 5

```
Continue Chitti OS. Phase 4 is green. Read CHITTI_OS_HANDOFF.md, then implement Phase 5 (Persona + intent shell): agent manifests (model ref, persona prompt, capability set, memory policy), lifecycle (spawn/suspend/resume/kill — checkpoint context + memory pointers, NOT the KV cache; recompute on resume), two-tier memory (KV cache in RAM + a persistent memory store with demand-paging/RAG recall), an intent shell over serial, and agent-to-agent IPC.
Definition of done: a typed intent completes a 2–3 primitive plan and returns the correct result; suspend→resume continues correctly; an agent recalls a fact from the persistent store not in its live context; two agents coordinate via IPC. Do not add self-compiling or taint features yet.
```

### Phase 6

```
Continue Chitti OS. Phase 5 is green. Read CHITTI_OS_HANDOFF.md, then implement Phase 6 (differentiators). Two features:
1) Provenance/taint: tag every context token user_typed | system_trusted | untrusted_ingested. Synapse must gate destructive/irreversible primitives on the provenance of the justifying tokens — if the justification traces to untrusted_ingested content, refuse or require explicit human confirmation at the shell.
2) Self-compiling agents: record validated capability traces keyed by (intent signature, preconditions); on a later matching intent with satisfied preconditions, replay the trace deterministically and skip inference ("compiled intent"); on precondition miss, fall back to planning. Keep a compiled-intents store.
Definition of done: an injection test (a file whose contents say "delete everything") does NOT fire a destructive primitive — the gate triggers; a repeated intent's second run is a ktrace'd cache hit with no inference; a stale-precondition compiled intent falls back to re-planning; everything is audited.
```

---

*End of brief. Build one phase at a time. The determinism boundary and the capability/taint model are the soul of this OS — protect them.*
