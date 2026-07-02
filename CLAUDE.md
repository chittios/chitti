*"This project is specified in `CHITTI_OS_HANDOFF.md`. Read it fully before acting. Follow the locked decisions in Part 2 and the guardrails in Part 4. Work one phase at a time; do not start the next phase until the current phase's acceptance criteria pass in QEMU."*

> **STANDING RULE — the kernel is dual-architecture (x86_64 + aarch64), and functionality must not diverge between them.** Every change must build and work for BOTH arches. Never guard behaviour behind `target_arch` unless it is genuinely arch-specific (a driver, an instruction) — and then provide the equivalent for the other arch behind the same API, never a stub that drops a feature. After any change, verify both: `cargo xtask build -arch x86_64` + `cargo xtask test` (69/69) **and** `cargo xtask build -arch aarch64` (and boot it via `-arch aarch64` when the change is boot-visible). If a capability exists on one arch, it exists on the other.

**Current phase: 7 (Stretch) — in progress. SMP + block-device FS + framebuffer TUI + dual-arch (x86_64 + aarch64) kernel: complete.**

**Qwen3.5-9B now generates correct text (~1 tok/s).** The 9B produced garbage
until three bugs were found and fixed (the last two using the HF
`modeling_qwen3_5` + llama.cpp `models/qwen35.cpp` references): **(1)** the
mixed-quant SMP worker keyed its kernel on `qtype`, colliding Q4_0-SDOT with the
generic path — added an explicit `mode` selector; **(2, the real one)** the
gated-DeltaNet GQA maps `nh`=32 value heads onto `n_group`=16 key/query heads,
and llama.cpp does this with `ggml_repeat` (tiling → value head `h` uses key
head `h % n_group`), NOT HF's `repeat_interleave` (`h / group_size`) — the GGUF
is converted for llama.cpp's convention, so **modulo** is correct (identity for
the 0.8B where `n_group==nh`, so it was never exercised there). With those,
`hello whats your name` → "My name is **Qwen**.", **byte-for-byte the same
greedy tokens as `llama-simple`** on the same ids; chat emits a coherent
`<think>` block. Q4_0-SDOT stays on the fast int8 path (int8 activation was a
red herring — the divergence was the grouping); Q4_1/Q5_K/Q6_K use exact-f32.
Debugging built a NumPy reference forward + per-layer checksum diffing against
Chitti to localize. 0.8B unchanged; both arches build; 69/69 green.

---
*Prior:* **Claude-Code-style chat shell.** The interactive shell is now a chat REPL:
plain text is a message to the Cortex model, `/`-prefixed lines are commands
(`/help /do <intent> /clear /infer /bench /perf /exit`). New `cortex::tokenizer`
is a byte-level BPE encoder (GPT-2/Qwen) — builds vocab + merge-rank maps from
the GGUF (parser now reads `tokenizer.ggml.merges`), maps bytes through the
GPT-2 byte→unicode table, greedily merges by rank; validated in a host
prototype to reproduce llama.cpp's ids — plus a streaming UTF-8 reassembler.
A `ChatSession` loads the model + tokenizer once and keeps a persistent
KV/recurrent cache across turns; each turn wraps the message in the Qwen chat
template (system prompt on turn 1) and **greedily streams tokens until the model
emits EOS or the user presses Ctrl+C** (non-blocking `console::read_byte()` poll
between tokens; 2048-token safety cap), replacing the old fixed-8-token `infer`.
`/do` keeps the Persona intent path (incl. the taint-gate confirm); `/infer`
keeps the fixed reference-parity check. Verified on the M2: `hi` streams a real
reply (`<think>…</think>` "Hello! … I'm Chitti …") that stops at EOS. Both arches
+ both models build; 69/69 green.

---
*Prior:* **Qwen3.5-9B runs (mixed-quant) + model as a build argument + release CI.** The
bundled model is a first-class `-model qwen3.5-0.8b|qwen3.5-9b` argument (like
`-arch`), 0.8B default: xtask threads it into the build (a kernel `model-9b`
feature selecting the memory layout — model at 0x80000000, heap at 0x2_00000000
sized 1 GiB, MMU mapping 12 GiB) and the run (GGUF file, load addr, `-m 12G`).
**The 9B now runs end-to-end.** Three things were built to get there: **(1)
delivery** — QEMU `-device loader` fails for blobs ≥4 GiB, so `model_loader_args`
splits a large GGUF into ≤1 GiB chunks loaded at consecutive addresses (the
guest sees one contiguous blob; validated byte-exact on the 0.8B). **(2)
mixed-quant support** — the unsloth Qwen3.5-9B-Q4_0 GGUF is actually mixed
(Q4_0 + Q8_0 + Q4_1 + Q5_K + Q6_K + F32); added Q4_1/Q5_K/Q6_K dequant (block
layouts from llama.cpp, **validated byte-exact vs the `gguf` reference:
max|ours−ref|=0** for real Q6_K/Q5_K blocks), a per-tensor-typed `QWeight` +
`matvec_qw` dispatch (Q8_0→fast SDOT, else a generic dequant-and-dot path,
both row-split across the 4 cores), and a separate untied `output.weight`.
Batched prefill stays gated to all-Q8_0 (0.8B); mixed models prefill
sequentially. **(3) memory** — the `model-9b` layout above. `xtask image` +
`fetch-model.sh` are model-aware; a release workflow builds the {x86_64,
aarch64} × {0.8B, 9B} matrix as release assets. **(4) Q4_0-SDOT** — the 173
Q4_0 tensors (of 250) got the fast path: `matvec_q4_0_sdot_rows` unpacks each
block's nibbles to int8 on the fly and `vdotq_s32`s them against the int8
activation (the Q8_0-SDOT analogue), row-split across the 4 cores. Verified on
the M2 (16 GB, native HVF): the 9B (32 layers, dim 4096, vocab 248320) loads
(5.4 GiB, 6 chunks, -m 12G) and decodes real tokens; Q4_0-SDOT took **prefill
~4.4→~1 s/tok (~3-4.6x)** and **decode ~1.9→~1.5 s/tok (~0.65 tok/s)** with
identical greedy output. Decode is now bottlenecked by the **Q6_K output
projection** (248320 rows, still generic scalar dequant) + Q5_K ssm_out — the
next lever is a Q6_K/Q5_K SDOT. (cf. llama.cpp CPU ~10 tok/s.) The 0.8B is
unchanged (~14 tok/s, matches reference=true); 69/69 tests green; both arches +
both layouts build.

---
*Prior:* **Batched prefill + a throughput harness (aarch64).** A `perf` shell builtin
(`cortex::bench_inference`) reports prefill (pp) and decode (tg) tok/s on a
synthetic prompt — a regression gauge run alongside `infer` (which still
asserts reference parity) after every change; directly comparable to
`llama-bench`. Against it: **(1)** the SDOT decode kernel was squeezed (single
f32x4 accumulator with one reduce per row, four independent chains) —
isolated bench ~3.3→~3.6 GMAC/s, but end-to-end decode is flat (~14 tok/s):
decode is at the **NEON per-core ceiling** (4 cores × ~3.6 GMAC/s ÷ ~1
GMAC/token). **(2)** *Batched, weight-stationary prefill*: `Model::prefill`
splits into `Model::prefill_batched`, which processes all prompt positions
together — the projection matmuls are batched via a new register-blocked
`tensor::matmul_q8_0_sdot_rows` (per weight block, load + f16-decode the weight
once, SDOT against a tile of 4 activation columns), while the order-dependent
recurrence/attention runs per position through the shared `attn_core`/
`delta_core` (extracted from `attn_layer`/`delta_layer`). Since each column's
SDOT is identical to the sequential matvec, batched prefill is **bit-identical**
(`matches reference=true`). The SMP pool was generalized from matvec to matmul
(`m_count`+`n_rows`), so prefill is split across the four cores too. Result: **pp
~18→~26 tok/s (~1.44x)**, tg ~14. Batched matmul is aarch64-only; x86 keeps the
sequential prefill (correct, slower). *Benchmarked vs llama.cpp on the same
GGUF (M2): llama.cpp CPU+Accelerate ~55 tok/s tg / Metal ~68 tok/s tg — its
edge is Apple's **AMX** matrix coprocessor (via Accelerate) + Metal, which a
bare-metal kernel in a VM can't reach; we are compute-bound on NEON, not
bandwidth-bound (~10% of the M2's ~100 GB/s), which is why reading weights once
in prefill only helps modestly.* 69/69 tests green; both arches build.

---
*Prior:* **aarch64 inference throughput — ~7x, now ~13 tok/s (0.8B, native HVF).** A
push to make bigger models usable. Three levers, all verified with `matches
reference=true` preserved: **(1)** a `bench` shell builtin times the hottest
kernel (`matvec_q8_0`) in isolation — it showed the f32-activation NEON matvec
is *widening-bound* (i8→i16→i32→f32 + per-block f16 decode), ~1.5 GMAC/s, and
that restructuring it (fold the block scale into one FMA, four independent
accumulator chains) is a throughput **wash**; **(2)** the real per-core lever is
llama.cpp's trick — quantize the activation to int8 (`quantize_activations_q8`)
and use ARMv8.2 **SDOT** (`vdotq_s32`, 16 int8 MACs/instr, *zero widening*):
`matvec_q8_0_sdot_rows` measured **2.2x** (3.4 GMAC/s) at **0.36% RMS error**,
and wired onto the forward pass via `tensor::matvec_q8_0_fast` (aarch64 = int8
SDOT with per-`State` scratch; x86 = the exact f32 path — same API, per the
dual-arch rule) it gave ~1.8x end-to-end with token parity intact; **(3)**
**SMP under HVF** (`arch/aarch64/smp.rs`): PSCI `CPU_ON` brings the secondary
vCPUs up (asm stub → private stack via context_id → `mmu::enable_secondary`
from the shared identity map → claim a worker slot), and each Q8_0 matvec is
split by disjoint row range across all online cores via a lock-free
static-partition barrier (publish operands + ranges, bump a generation counter
released/acquired, each core writes a disjoint `y` slice; <256-row matvecs stay
single-core). Under `-smp 4`, **4/4 cores online** and near-linear ~3.8x. Net:
decode ~502→~72 ms/tok (**~13 tok/s**, stable), prefill 1675→~300 ms, all with
`matches reference=true`. `+dotprod` added to the aarch64 target. x86 untouched
(69/69 green). *Decision (with the human): int8 SDOT adopted as the aarch64
default; SMP built; the 9B model deferred — it needs a ~5 GB (Q4_0) layout /
larger guest RAM overhaul, staying on 0.8B for now.*

---
*Prior:* **Unified dual-architecture kernel.** One `kernel/` crate now builds and boots
for **both** x86_64 (Limine, under QEMU TCG) and **aarch64** (native on Apple
Silicon via `qemu-system-aarch64 -accel hvf`). Arch is chosen explicitly, never
host-detected: `cargo xtask build|run -arch x86_64|aarch64`. The split: an
`arch` facade (`arch::interrupts`/`hlt`/`now_ms`) the whole kernel uses; per-arch
modules (`arch/x86_64`, `arch/aarch64` — the latter: `-M virt -kernel` boot stub,
identity-map MMU, PL011 UART, generic timer, DAIF interrupts, `wfi`); portable
tensor kernels (SSE2/AVX2 ∣ NEON ∣ scalar behind one API); an arch-split
`sched::context` (x86 naked switch + FXSAVE ∣ aarch64 cooperative `stp/ldp`
switch of x19–x30 + d8–d15); arch-dispatched `serial`/`mm`/`console`; and
x86-only device code (Limine, gdt/idt/pic/pit/keyboard, apic, smp, virtio,
qemu, frame allocator + paged heap) cfg-gated out of aarch64, which instead uses
an MMU identity map + a fixed-region heap. **Full functional parity across
arches, including inference.** On aarch64 the GGUF model is placed in guest RAM
by QEMU `-device loader` at 0x48000000 (the aarch64 `model_module` reads it
there, validated by the GGUF magic — the equivalent of the x86 Limine boot
module); the heap moved to 0x80000000 to sit past it. **Verified:** x86 builds
+ `cargo xtask test` 69/69 green; aarch64 builds and boots natively under HVF,
running the full agent OS — Synapse ABI, Persona intent shell, the taint gate
refusing a prompt-injected `mem_fs_delete`, compiled-intent cache hits, the
scheduler's context switch, **and native NEON inference**: `cargo xtask run
-arch aarch64` → `infer` decodes the reference prompt at ~2 tok/s with
`matches reference=true` (token-for-token parity with the x86/NumPy reference),
vs minutes/token under x86 TCG. The prior standalone `arm64/` crate is retired.

---
*Prior:* **aarch64 native port — foundation booting under HVF.** The x86 kernel can only
run under QEMU cross-arch TCG on this Apple Silicon host (no HW accel for x86
guests), which is why inference is slow. The fix is an aarch64 build so
`qemu-system-aarch64 -accel hvf` runs Chitti **natively on the M-series CPU**
with **NEON**. New standalone `arm64/` crate (`targets/aarch64-chitti.json`,
own linker/boot) boots directly via `-M virt -kernel` in EL1: `_start` sets the
stack, enables FP/SIMD (CPACR_EL1), zeroes BSS; `kmain` brings up a minimal
identity-map **MMU** (1 GiB blocks, RAM = Normal cacheable, MMIO = Device;
MAIR/TCR/TTBR0/SCTLR) so NEON + caches work, then runs the fused **NEON Q8_0
matvec** (the kernel that dominates a token) against a scalar reference and
times it with the ARM generic timer. Verified on the M2 via HVF: boots
natively, `NEON matches scalar reference: YES`, **~3.7 GMAC/s** (vs the effective
tens-of-MMAC/s under x86 TCG) → est. **~4 tok/s** for Qwen3.5-0.8B, a ~100x
compute speedup. `cargo xtask arm64 [--release]` builds + boots it. This is the
foundation; porting the full kernel stack (scheduler, GIC/timer IRQs, the
Cortex/Synapse/Persona layers behind an arch facade, NEON tensor kernels) onto
it is the remaining work. The x86 build is untouched (69/69 tests green).

---
**Inference UX + multicore study.** The shell `infer` builtin now shows the
prompt, streams the response live (mirrored to the framebuffer TUI), and
reports throughput (prompt tokens / prefill ms; decode tok/s via PIT timing)
and reference parity. Multicore inference was implemented (a row-range
`tensor::matvec_q8_0_rows` split across an AP worker pool) but **measured a net
loss under QEMU cross-arch TCG** — single-thread TCG can't run vCPUs in
parallel, and `-accel tcg,thread=multi` taxes every emulated instruction while
idle worker cores contend for host CPU. Reverted to single-core: APs park after
the SMP self-test, and the inference paths (`run`, `ref-check`) use a single
vCPU (fastest for the BSP-bound workload); `-smp 4` is kept only for `cargo
xtask test` (the SMP self-test). The row-range kernel keeps a real-hardware
multicore split a drop-in away. Tests remain 69/69 green.

---
**Framebuffer TUI (Phase 7 track 3) — complete.** The QEMU graphical window is
now a live terminal, and a human can drive Chitti from it. `framebuffer.rs`
became a persistent global `Console`: an 8x8-font character grid with a cursor,
newline handling, backspace, and scrolling (framebuffer memmove) once it fills;
green-on-black. `serial::Serial::write_str` mirrors every byte to it (gated
`#[cfg(not(test))]`), so the *entire* session — boot log, ktrace, all phase
demos, and the interactive shell — appears on screen while serial keeps working
in parallel. `arch/x86_64/keyboard.rs` gained scan-code-set-1 decoding
(shift + caps-lock, US layout) feeding a ring buffer drained by `read_char`; a
new `console.rs` unifies input (`read_byte`: keyboard *or* serial) and echo
(`put_byte`: both), and the shell's `read_line` now uses it. Verified by
screendump (135k green text pixels on clean background, top line rendered) and
by injecting `l i s t <enter>` via QEMU `sendkey` (real PS/2 scancodes, not
serial) — the shell echoed `chitti> list`, ran the intent, and replied
`=> ok:[...]`. `cargo xtask test` stays 69 (the framebuffer module isn't
compiled into the test build; the mirror/echo calls are cfg-gated). Remaining
Phase 7 track: RISC-V port. (9B model still skipped per the guardrail.)

---
*Prior in Phase 7:* **Block-device FS — complete.**
**Block-device FS (Phase 7 track 2) — complete.** A real filesystem on a real
disk. `block/mod.rs` defines a `BlockDevice` trait (512-byte sectors);
`block/ramdisk.rs` is a RAM-backed impl (the test suite mounts on it);
`block/virtio.rs` is a **virtio-blk driver over the legacy PCI transport**
(PCI scan on 0xCF8/0xCFC, legacy I/O BAR, feature negotiation, a polled
single-request virtqueue with DMA buffers from `mm::alloc_dma`, which added
`frame::allocate_contiguous`). `fs/mod.rs` is **SimpleFS**: superblock +
fixed 64-byte inode table (8 direct blocks/inode) + data region, free blocks
found by scanning live inodes (no bitmap to desync), write-through; supports
format/mount/write/read/list/delete and a `mount_or_format`. `cargo xtask
test` is now 69 tests (up from 64): 7 FS tests over the RAM disk, including a
format→write→**unmount→remount**→read round-trip proving data lives in the
device's blocks. On a real boot with a virtio-blk `-drive`, the boot demo bumps
an on-disk boot counter — verified to **survive a reboot** ("boot #2 … the
counter survived a reboot — durable storage works"). SimpleFS is a standalone
subsystem for now (the Synapse in-memory store is unchanged); wiring it as
Persona's durable tier-2 is a follow-on. Remaining Phase 7 tracks: framebuffer
TUI, RISC-V port. (9B model still skipped per the Part 4 guardrail.)

---
*Prior in Phase 7:* **SMP / APIC-per-core — complete.**
Multiple CPUs now execute kernel code concurrently under correct locks. The
kernel `Locked` type (`mm/mod.rs`) is now a real **test-and-test-and-set
spinlock** (atomic + interrupts-off while held) instead of the old
interrupt-disable-only guard — the load-bearing SMP-safety change, since every
shared structure (scheduler, heap, frame allocator, Synapse FS/audit, compiled
intents) locks through it. A Limine **MP request** (`limine_protocol::Smp*`)
brings up the application processors: `smp::init` (run on the BSP at the end of
`chitti_kernel::init`) writes each AP's `goto_address` to launch it into
`smp::ap_entry`, where the core enables SSE, sets up its **own** GDT+TSS
(`gdt::init_ap`, heap-allocated — a shared TSS can't be `ltr`'d twice), loads
the shared IDT (`idt::load_ap`), and software-enables its **local APIC**
(`arch/x86_64/apic.rs`, whose MMIO page the HHDM doesn't cover so `mm::map_mmio_page`
maps it). Each online core (BSP + APs) then runs a bounded self-test — all
cores hammer one shared counter through the spinlock — which doubles as the
lock-discipline proof: the counter lands on exactly `cpus × 5000` with zero
lost updates under real contention. APs then `hlt`-park (no vCPU-time theft
under TCG). `cargo xtask test` runs the harness with `-smp 4 -accel
tcg,thread=multi` and is now 64 tests (up from 63): the new one asserts all 4
cores came online, the spinlock summed exactly (no lost updates), and work ran
on ≥2 cores. The scheduler stays BSP-driven for now (APs do bounded work then
park); per-core run queues + APIC-timer preemption + IPIs are the next SMP
refinement. Remaining Phase 7 tracks (agreed with the human, one at a time):
block-device FS, framebuffer TUI, RISC-V port. The **9B model** was explicitly
skipped (Part 4 guardrail + unusably slow under TCG).

---
*Prior:* **Phase 6 (Differentiators — taint security + self-compiling agents) — complete.**
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
