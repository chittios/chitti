# ChittiOS — guide for agents & humans working in this repo

**Chitti is an agentic operating system: the agent is the driver.** The
fundamental unit of execution is an AI agent, not a compiled binary. There is no
`exec(binary) → trap into syscalls`; instead `spawn(agent) → plan over
capabilities → a deterministic executor runs the primitives`. A tiny language
model runs on the CPU, on bare metal, and every effect it wants flows through a
capability-checked, audited, grammar-constrained ABI.

**No POSIX. No libc. No ELF loader. No re-implementing Unix.** We are not
building "Linux but in Rust." Why reinvent the wheel? The interesting question is
what an OS looks like when an agent — not a shell script or a C program — is the
thing you run. So the OS gives an agent first-class primitives: a **shell agent**
(chat REPL), **sessions** (save/resume/fork), **sub-agents** (isolated,
capability-attenuated delegation), **tools** (MCP-shaped, Synapse-backed),
**skills** (signed, permissioned, progressively-disclosed packages), a **todo-list
TUI**, and a real windowed console with **mouse + keyboard**. The classic OS
plumbing that must exist (scheduler, MMU, capabilities, IPC, block/FS, drivers)
exists to serve that, not to run `/bin/sh`.

## The load-bearing invariants — never break these

1. **The determinism boundary.** Model output is an *untrusted plan*. It never
   causes a side effect directly — it is parsed, grammar-validated, capability-
   checked, and only then executed by deterministic native code (`Synapse`).
   Above the boundary: stochastic (the agent/model). Below: deterministic.
2. **All effects route through Synapse.** Tools are a validation/presentation
   layer over Synapse primitives; they never touch hardware or memory directly.
3. **Delegation only ever narrows authority.** A sub-agent's capability set is a
   strict subset of its parent's. Spawning can attenuate, never widen.
4. **Provenance/taint gating.** Every context token is tagged
   (`user_typed` / `system_trusted` / `untrusted_ingested` / `skill_installed`).
   A destructive primitive justified by untrusted-ingested content is refused or
   requires explicit human confirmation — prompt-injection-as-privilege-
   escalation defense enforced at the OS boundary.
5. **A skill is bounded by its install-time grant, forever.** Even if a skill's
   text says "delete everything," it can only act within the capabilities the
   human approved at install.

Capabilities are unforgeable (opaque per-task indices, no ambient authority).
Effective authority is always `intersection(requested, granting-context)`.

## STANDING RULE — ring 0 is for drivers; agents and their commands run in ring 3

**Only the kernel proper and the drivers execute in ring 0 / EL1.** Every agent, and
every agent- or app-facing command, performs its effects in **ring 3 / EL0** as a
userspace tenant. This is the execution-model half of invariant #1: "all effects route
through Synapse" is a claim about *authority*, and this rule is the claim about
*privilege* — an agent should not be able to corrupt kernel memory by getting a tool
wrong, only to be refused by a gate.

**How to make a call from a migrated caller** — one function, always:

```rust
crate::synapse::tenant::invoke_in_userspace(task, raw, justification) // -> Option<Invocation>
```

It runs the call in ring 3 under `task`'s own identity, through all four gates, and
returns the **structured** `Invocation`.

Three rules that are not style preferences — each is a bug that already happened:

1. **Never parse the tenant's reply text to decide what happened.** A tenant's reply is
   prose rendered by `synapse::abi::render`, whose vocabulary (`denied:capability:X`)
   differs from the tool router's (`denied: no capability for X`). Classifying from the
   text made refusals read as successes and `security::redteam` reported **five injected
   attacks as permitted** where it had reported zero. There is exactly one authority for
   what an outcome was, and it is the `Invocation`, not a string.
2. **The justification is computed above the boundary and passed in.** The ABI defaults
   to `UntrustedIngested`, so a caller moved into ring 3 without one keeps its identity
   and capabilities but silently becomes maximally tainted — every destructive call it
   used to make legitimately starts being refused. **Moving code across the ring boundary
   must not change what it is allowed to do**, or the migration is a policy change
   wearing a refactor's clothes. Pass what the in-kernel path passed
   (`Justification::trusted()` where it called plain `synapse::execute`). The tenant
   still cannot choose it — the kernel sets it before entering ring 3.
3. **No in-kernel fallback when userspace fails.** Retrying the call in ring 0 would make
   confinement depend on whether the loader happened to work. Report
   "the userspace call never reached the gates" as its own failure — never folded into a
   refusal or a grammar rejection, because "the gates said no" and "userspace could not
   run it" are different facts and a caller that cannot tell them apart retries the wrong
   one.

**`synapse::executor::execute` / `execute_with_justification` are for the kernel and
drivers only.** Calling them from anything agent-shaped is the bug this rule exists to
prevent, and it is invisible — the call still works, it just keeps kernel privilege. The
legitimate callers are exactly:

- `synapse::abi::dispatch` — the kernel side of the tenant trap. This *is* the ring-3
  path; it must call the executor directly or nothing could.
- `tools::dispatch::run_synapse` — the in-kernel arm, taken only when
  `runs_in_userspace(session)` is false, i.e. for the orchestrator.
- `synapse::bench` — measuring the gate chain, kernel-internal by definition.
- `#[test_case]` tests of the gate chain itself (`lib.rs`'s `phase*` tests, `cap`,
  `redteam`). A test of the executor should call the executor.

Anything else is a bypass. Four existed at once and every one was found by grepping for
the call rather than by review, because a bypass looks exactly like ordinary code and
*works*: `agent/wasm_rt.rs` (app UI ops), `service/server.rs` (content-agent asset read),
`service/package_ui.rs` (the running app-UI pump), `persona/agent.rs` (a persona agent's
plan/act loop). All four are migrated.

**This is enforced, not remembered: `cargo xtask ring-check`** scans `kernel/src` and
fails on any direct executor call outside the allowlist in `xtask/src/rings.rs`, printing
file:line. Run it after adding a Synapse call site; to add a legitimate caller, add the
file to `ALLOWED` *with a reason* (a test asserts every entry has one). The scanner skips
comments — the rule is documented in prose that names the function, and a check that
counted those would be deleted for crying wolf — and a host test plants a fake bypass to
prove the walk is not vacuously green.

**The one documented exception is the orchestrator (the shell agent)**, which stays in
ring 0: it *is* the shell, holds root authority, and drives the kernel it would be
confined relative to — so there is no isolation to gain. Revisit that deliberately if
ever, not by accident.

**Where "run it in ring 3" is not the right question.** Ring 3 has no device access by
construction — that is the property being bought. So a command that touches hardware
(`/voice` on the sound device and ONNX models, `/model` DMA-loading a GGUF, `/ping` on an
ICMP socket, `/lspci`, `/disks`, `/display`, `/battery`, `/install`) cannot itself execute
there without mapping device memory into a tenant, which destroys the isolation. The
correct shape is always **driver in ring 0, effect requested from ring 3 through a
Synapse primitive** — so "migrate X" for such a command means *designing a primitive*,
not moving code. That is a real design act: each primitive is a new gated, audited
authority surface, it moves the paper's primitive count (which `cargo xtask paper-check`
pins), and it adds an entry to the `security::redteam` census.

**The other half of ring 3 is the parsers, and they are the cheap case: a decoder needs no
authority at all.** It reads a byte buffer, writes a pixel buffer, exits — zero Synapse
calls, so no new primitive, no gate change, nothing in the `paper-check` count or the
`redteam` census. That is the exact opposite of `/http` or MCP, and it is why the largest
attacker-reachable code in the OS is also the easiest thing to confine. **`/open` decodes
PNG and JPEG in ring 3** (`userspace/imgdec/`, driven by `synapse::tenant::ImageTenant`):
a malformed file becomes a status word from a tenant the kernel discards, where in ring 0
it is a parser bug away from a wild write. `/decoder ring3|kernel` switches back for an
A/B, and the in-kernel decoder stays until the differential has run on both arches.

Five things that path pins down, each of which cost a debugging cycle:

- **The tenant mounts the kernel's own source** (`#[path] = "../../../kernel/src/image/…"`,
  the `pdf-wasm`/`h264diff` pattern), so there is **one** decoder and porting it cannot
  regress it. What the differential test then compares is the *boundary* — layout, entry
  offset, stack ABI, arena, pixel gather — which is the thing that can actually be wrong.
- **x86-64 SysV guarantees `rsp % 16 == 8` at function entry**, because a `call` pushed a
  return address. A tenant arrives by `iretq`, which pushes nothing, so a 16-aligned initial
  rsp leaves every compiler frame 8 bytes off and the first `movaps` spill raises `#GP`.
  Corrupt inputs returned before spilling a vector register, so *rejection worked and
  success did not*. AAPCS64 is the opposite rule (SP always 16-aligned), so the initial SP
  is arch-specific.
- **The tenant is reused, and a reused address space keeps its `.bss`** — so it resets its
  own bump cursor at `_start`. Without that line the second decode starts with a full arena
  and reports out-of-memory, and nothing in the loader points at the cause. Reuse is not a
  micro-optimisation: building a space, mapping ~1000 arena pages and tearing it down was
  the entire measured overhead of ring 3, not the execution.
- **How much heap an image needs is a number inside the file.** So the loader does not size
  the arena — it starts small, and the tenant reports `STATUS_OUT_OF_MEMORY` (distinct from
  "corrupt", or a good photo reads as a broken one) and the loader maps more and re-enters.
  The alternative is parsing the header in ring 0, which is the thing being undone.
  Symmetrically there is **no output buffer**: the tenant leaves the pixels in its arena and
  reports where, and every number it reports is bounds-checked against the frames the loader
  owns rather than trusted. The tenant also parses the header *itself* and reports the arena it
  expects to need (`HEAP_WANT`), so an oversized image costs **one** retry instead of a
  doubling search — six full attempts, each doing real work before running out, is what made a
  large photo effectively undecodable.
- **A bump allocator's size is the sum of everything it ever allocated, unless it reclaims.**
  Freeing the *top* allocation (roll the cursor back) and growing the top block *in place* are
  four lines each, and together they turn that sum back into a high-water mark: a `Vec` grown
  by doubling stops stranding every intermediate size, and per-row scratch stops accumulating.
  Before them a 16x16 PNG needed more than a 4 KiB arena; after them it fits in one page, which
  is what `a_16x16_png_decodes_inside_a_single_page_of_arena` pins. The other half of that fix
  is in `image/png.rs` itself — it unfilters **in place** rather than building a second
  whole-image buffer plus a `Vec` per scanline, which is also one allocation per row of heap
  churn removed from the kernel path (performance trap #3).
- **No in-kernel fallback**, per the rule above: a decode that cannot be sandboxed is an
  error, not a quiet retry in ring 0.

**The next decoders need a chunked, stateful tenant, and that is a design act rather than a
port.** H.264, AAC and ONNX are all on the list, and the blocker they share is not the mount:
it is that **Ctrl+C and `upkeep()` are standing rules** and a tenant has no device access by
construction. A whole-file AAC decode takes seconds — the in-kernel one pumps `upkeep()` every
32 frames for exactly that reason — so doing it in one crossing would freeze the clock, mouse
and net stack and ignore Ctrl+C. The shape that works is the tenant decoding a **bounded number
of frames per entry** while the kernel pumps and polls between entries, which means a command
word in the startup block, decoder state that survives an entry, and an arena that is *not*
reset per call (the opposite of the image tenant's one line). H.264 has that plus a second
problem: **tenants are pinned to the boot CPU** (no TLB shootdown), while 1080p30 was reached
partly by loaning the decoder to an SMP worker — so a naive port trades a measured 30 fps for
~12. ONNX is last for the reason it always was: weights live in DMA frames and its ops fan out
across the fleet. Note what is *not* a blocker — the H.264 decoder core is pure (`video/mt.rs`
is the only SMP dependency and it covers YUV→RGB and letterbox scale, i.e. presentation, not
parsing), so the natural split is **bitstream→YUV in ring 3, YUV→RGB in ring 0**.

wasm was measured and rejected for this: `tools/pngbench` (a permanent host tool, the
`*diff` pattern) put wasmi at **47-67x native** on a 1.3 MP PNG — and "small images via
wasm" does not rescue it, because an attacker sends a *large* one. Ring 3 costs a page-table
switch and a trap. wasm stays right for `pdf`, which is already there.

**`/open x.pdf` renders real pages, and wasm is the sandbox there because ring 3 is not
available.** The rasterizer is [hayro](https://github.com/LaurenzV/hayro) over
`vello_cpu` — `tools/pdfrender-wasm/` from crates.io, compiled to
`assets/wasm/pdfrender.wasm` (4 MiB, checked in, `include_bytes!`) and driven by
`shell/pdf.rs` + the pure `pdfview.rs`. It would rather be a tenant like `imgdec`, but its
tree is `std`-bound in three places that are *not* features (`moxcms`, `pxfm`,
`pic-scale`), so the confinement comes from wasm instead of a page table: **zero imports**,
a memory-page limiter, a per-call fuel bound. Five things that path pins down:

- **The build profile is worth 8.5x.** `opt-level = "s"` renders a dense LaTeX page in
  ~7.0 s under wasmi, `3` in ~0.82 s — and the `3` module is *smaller* (4.1 vs 5.4 MiB), so
  there is no trade to make: the size profile drops the inlining vello_cpu's per-pixel
  pipelines are built around. `tools/pdfbench` is the permanent harness (native *and* wasm,
  same crate, seconds on the host — the `pngbench` pattern).
- **Then wasm SIMD was worth another 2.2x, and it takes both halves.** The guest is built
  with `-Ctarget-feature=+simd128` (pinned in `tools/pdfrender-wasm/.cargo/config.toml`, not
  an env var, so a rebuild cannot silently lose it) and the kernel runs **wasmi 1.1 with the
  `simd` feature**. Separated on one real document: the 0.40 -> 1.1 interpreter alone is
  **1.43x**, SIMD on top is another **1.53x**, and on a blend-heavy page — vello_cpu's
  per-pixel pipelines are exactly what vectorizes — it reaches **5.8x** (6850 -> 1181 ms).
  Both halves are required: wasmi 0.40 *rejects* a simd128 module at validation, and a
  scalar module gains nothing from the feature. The interpreter tax fell from 30-90x
  (340x on the worst page) to **3-30x**. SIMD also *lowered* fuel use 3x — a vector op does
  several scalars' work and is charged once — so a fuel budget does not survive a change in
  how the guest is compiled.
- **The WASI stubs were all declared `() -> i32`** — the wrong arity for every one
  of the five. wasmi resolves an import by name *and* `FuncType`, so no real WASI
  module could ever have instantiated, and the symptom was again
  `"wasm instantiate failed (missing imports/limits?)"`. Nothing noticed because the
  only module on that surface (`pdfrender.wasm`) has no import section at all and no
  test instantiated a WASI importer. `register_wasi_imports` now implements the ten
  functions Javy needs, with fd 0/1 backed by host-side buffers.
- **Fuel metering costs 3.7%**, measured with it off. It stays on: that is the only bound on
  a runaway guest, and 3.7% is not a price worth arguing about.
- **In-kernel is ~3x slower than the host on heavy pages and equal on light ones**, which is
  not the allocator: re-rendering a page with the guest's arena already grown is only 6%
  faster than the cold render. It scales with the *working set* (a heavy page holds 56 MiB
  of glyph and image caches), so the suspect is guest TLB pressure under stage-2
  translation, and the lever would be huge pages for the kernel heap — unmeasured, so
  stated as a hypothesis. Release and debug kernels time the same here, so the harness's
  debug boots are fine to measure on. Real figures (aarch64 HVF): **445 ms** for page 1 of
  a 117-page book, **565 ms** for a paper's first page, ~3.1-3.9 s for its two
  attention-matrix figure pages.
- **SMP for wasmi does not exist and cannot exist inside a call.** wasmi is a
  single-threaded interpreter and none of the wasm `threads` proposal is implemented, so
  there is no way to split one render across cores; vello_cpu's own `multithreading`
  feature needs rayon and is off. What *is* available is loaning a whole `Session` to an
  SMP worker the way the video player loans its decoder (`smp::async_submit`) — which buys
  a live UI during a render and a next-page render-ahead, not a faster render.
  `Session: Send` is asserted at compile time
  (`a_session_is_send_so_it_can_be_loaned_to_an_smp_worker`) so that groundwork cannot rot;
  the render-ahead path itself is **not built**.
- **The `ResourceLimiter` capped function tables at a hardcoded 256**, and a renderer built
  on trait objects declares **693**. The failure is
  `wasm instantiate failed (missing imports?)` — the one message that does not mention
  tables. It is now `Limits::with_table_elems`, defaulted to the old 256 so every existing
  agent module keeps exactly its old bound.
- **Pixels and documents do not go through the string ABI.** A page is megabytes;
  base64-through-JSON would cost that 4/3 twice plus a guest decode. `Session::put_bytes` /
  `get_bytes` / `get_u32s` move bytes directly, and every number the guest reports is
  bounds-checked against live linear memory — the image tenant's rule, because a guest is
  the untrusted side even when it is our own code.
- **An oversized render is refused, not attempted.** Under `panic = "abort"` an allocation
  failure is a trap that kills the instance, so the *parsed document* would be lost with it;
  the guest checks `MAX_PIXELS` first and answers `ERR_TOO_LARGE`, and the host clamps to
  the same number (`pdfview::cap_permille`) so it rarely has to ask.
- **One instance holds one document, and it stays parsed.** The `Session` is retained, so
  paging a 35-page paper parses it once and the render cache keeps glyphs warm; the rendered
  page is cached by `(page, scale)`, so scrolling, panning and tab switches are a memcpy and
  only a page or zoom change pays the renderer. Zoom re-renders rather than scaling the
  bitmap — that is the whole point of "proper preview" — and the view only ever moves a 1:1
  window over the result (`pdfview::compose`).

Note also a viewer-shaped trap that is not about PDF at all: `media_key` hands a painter
**typed bytes** while `media_nav` hands the **finals of an escape sequence**, and both are
`u8`. A viewer that read `A`..`D` as arrows in the typed path scrolled the page whenever a
capital letter was typed — with the tab focused, `/cat /samples/README.md` arrived as
`/cat /samles/REME.md`. Arrows belong only to the nav path.

**Still in ring 0 for want of a primitive, not by choice:** the `http` tool (binds to
`Shell{command:"http"}`, and `net_http_get`/`net_http_post` do not express its
`-X`/`-H`/`-d`/`--stream` surface), MCP `tools/call` (a stateful JSON-RPC/SSE session),
downloads (fetch *and* store write), browser navigation and the web tools (the JS engine
plus the net stack), and agent memory (its own store, though this one is expressible as
`mem_fs_*` and is the cheapest next migration).

## STANDING RULE — dual-architecture, no divergence

The kernel is **one codebase for two architectures: `x86_64` and `aarch64`**, and
functionality must not diverge between them. Every change must build and work for
**both** arches. Never gate behaviour behind `target_arch` unless it is genuinely
arch-specific (a driver, an instruction) — and then provide the equivalent for the
other arch behind the **same API**, never a stub that drops a feature. If a
capability exists on one arch, it exists on the other.

After any change, verify both:

- `cargo xtask build -arch x86_64` **and** `cargo xtask test` (the in-kernel
  unit suite — keep it green; add cases for new logic)
- `cargo xtask build -arch aarch64` (and boot it via `cargo xtask run -arch aarch64`
  when the change is boot-visible)

## STANDING RULE — every feature/fix ships with tests

Two layers, and new work adds to **both** where they apply:

1. **Unit tests** (`cargo xtask test`, x86 under QEMU, no model/hardware) for
   the **pure logic** — parsers, decoders, codecs, format/build functions,
   capability math. The pattern that keeps us safe: pull the fiddly logic out
   of the hardware/IO path into a pure function and test it with cases (e.g.
   `mouse::decode_ps2_packet`, `xhci::parse_report_layout`, `net::http`
   `parse_url`/`dechunk_partial`, `shell::parse_tool_call`,
   `ws::{base64,sha1,encode_frame}`, `channel` ring/EOF, `synapse::ui` draw-op
   raster + ownership, `service::http::parse_request`, `crypto::verify_p256`,
   the executor scope gate, `registry_client::parse_index`). If logic broke
   before, it gets a test.

   **`framebuffer/` is `#[cfg(not(test))]`, so a test written inside the
   compositor never runs — it is not even compiled.** (The gate is in `lib.rs`:
   a `-Z build-std` + `cargo test` interaction gives two non-unified copies of
   `core`/`alloc` otherwise.) A `#[cfg(test)] mod` in there is therefore silent
   dead code, not coverage — which is how an analog-clock trig reduction wrong
   in **two of four quadrants** shipped, with tests sitting next to it that
   asserted the right things and were never built. So geometry, wrapping and
   colour math must live **outside** the compositor to be testable, the way
   `clock::face` (dial geometry), `editor_wrap` (soft wrap), `textsel`
   (selection), `panes_layout` (pane geometry) and `display` already do. When
   you touch drawing code, check whether the logic you are changing is in a
   module that can be tested at all before adding a test to it.

2. **End-to-end tests** (`tests/e2e/`, `make e2e` / `make e2e-full`) for
   anything that only exists **on the running OS** — a shell command, a
   network/TLS/WebSocket exchange, a model or voice flow. The harness boots the
   real kernel under QEMU and drives the shell over serial. **Adding a shell
   command or a networked/model/voice feature means adding an e2e scenario**
   (`os`/`agents`/`net` groups always run; `model`/`voice` are `--slow`, gated
   on assets). The `agents` group covers install/consent, service lifecycle,
   the network + HTTP/Doc service agents (the host reaches guest listeners via
   an opt-in slirp `CHITTI_HOSTFWD` port), registry search/install, and UI
   surfaces. A fix for something the harness could have caught gets a scenario
   too. Run `make e2e` before shipping boot-visible or networked changes.

CI (`.github/workflows/ci.yml`) runs the `unit` job on every push/PR: it builds
both arches + `cargo xtask test`. The e2e suite is **not** run in CI — boot it
locally with `make e2e` before shipping boot-visible or networked changes.
It is **fork-PR-safe** — `pull_request` (never `pull_request_target`),
`contents: read`, no secrets, GitHub-hosted runners only. Keep it that way.

## STANDING RULE — performance: know the three traps before optimizing

Hard-won findings (commits `f2bd8f7`, `06a62b4`) that apply to **any new
compute- or I/O-heavy feature**. Measure first, always — every one of these was
found by profiling, and the obvious suspect was wrong every time.

1. **`+strict-align` silently scalarizes NEON.** The aarch64 target builds with
   `+strict-align` (required for the pre-MMU boot window and device MMIO).
   Under it, LLVM lowers *any* unaligned vector load/store — the `vld1q_*`
   intrinsics **and auto-vectorized plain loops** — into a 16×`ldrb`+`orr`+
   stack-spill byte-assembly: ~25 instructions per load, ~100× slower. The
   binary still contains `fmla`, so it looks fine until you disassemble the
   *loads*. Hot SIMD loops must do their memory access via **inline asm**
   (`ldr q`/`ldp q`/`stp q` — correct at runtime: Normal cacheable RAM,
   `SCTLR_EL1.A=0`), the same pattern as the MMIO single-`ldr`/`str` rule.
   Reusable kernels live in `cortex/tensor.rs`: `dot_f32`, `axpy_f32`,
   `scale_f32`, the `ldq_s8/u8/f32` load helpers, and the Q8_0/Q4_0/Q4_K SDOT
   matvecs. **Prefer composing these** over writing new intrinsics code; if
   you must write a new SIMD loop, verify with `objdump -d` (count `ldrb` in
   the hot function — the Q4_K kernel disassembles to 0 ldrb / 32 sdot) and
   `/onnx bench` in the booted kernel (dot_f32 ≥ 10 GMAC/s under HVF;
   ~1 GMAC/s means the disease is back). `/bench` prints the SDOT-vs-exact
   rel-RMS error per fast kernel (`check_q4_0_sdot`, `check_q4_k_sdot`).
2. **Batch block I/O or die waiting.** One `read_block` = one polled virtio
   round trip (~0.5 ms). Anything reading more than a few KiB must use
   `read_blocks` (all drivers implement multi-sector requests; `Partition`
   forwards). FAT `read_file` coalesces contiguous cluster runs — loading the
   131 MB parakeet model went from ~2.5 min to ~2 s. New filesystem or loader
   code must follow the same pattern (and cache FAT-chain sectors).
3. **The kernel allocator punishes churn.** First-fit linked list: cloning
   multi-MB tensors per node, or cloning a whole env per loop iteration, is
   real time. Borrow instead of clone on hot paths; move values into maps
   instead of cloning them. Heap pressure counters exist
   (`mm::heap::alloc_stats()` — allocs + free-list scan steps).

Layout matters as much as SIMD: iterate state **row-wise/contiguous** (the
DeltaNet delta rule went from stride-512B scalar walks to contiguous AXPYs —
that alone nearly doubled tokens/sec). Per-op wall-time accounting for the
ONNX executor prints one `onnx: op time:` ktrace line per run; `/perf` gives
prefill/decode tok/s; `tools/onnxdiff/` runs the kernel's own ONNX interpreter
natively on the host (seconds per iteration, layer-by-layer diff vs
onnxruntime) — use it for any voice-model numeric or perf work before touching
QEMU.

Current figures (aarch64 HVF, 0.8B Q8, `/perf 512`): prefill ~105 tok/s, decode
~23 tok/s (same box, same run: prefill was ~60 tok/s before the window-wide
cores below), `/voice stt` ~2 s, `/voice say` ~14 s for 3.5 s of audio.

**Prefill's parallel axis is not the position, and getting that wrong is what
kept `pp` pinned near `tg`.** The projections were weight-stationary matmuls
across the whole fleet while the attention and DeltaNet cores ran a position at
a time on the BSP — so seven cores idled through the part that was left, and a
2x-of-decode prefill looked like a matmul problem when it was an Amdahl one.
Both cores are now window-wide (`attn_core_batched`, `delta_core_batched`, with
decode entering them as a one-position window so the two paths cannot drift):
attention fans out over **positions** (every `(position, head)` is independent
once the batched K/V are in the cache), the recurrence over **heads** (sequential
in position, but each head owns its own state slice). Three things that bite
here: the conv1d's window/ring index is `ck + p`, **not** `ck - 1 + p` — the
per-position form shifted the ring before convolving and that `-1` follows you
into the rewrite, where it reads every tap one position too old and still decodes
fluent text (`conv_tap`, pinned by `conv_window_matches_the_per_position_ring`);
parallelism that exists is not parallelism worth taking — fanning a *decode*
step's 0.8M-MAC DeltaNet layer across the fleet measured **slower** than inline
(35 -> 20 tok/s), hence `fanout_chunk`; and `/perf [n_prompt [n_decode]]` now
prints a **proj / attn / delta / elementwise** split, because "prefill is slow"
is three different diagnoses and they scale differently. Measure with a real
prompt length: at 64 tokens prefill is one chunk and tells you nothing.

**aarch64 SMP
row-split is live** (`arch/aarch64/smp.rs`): PSCI `CPU_ON` bring-up, `WFE`-parked
workers, a static-partition job barrier splitting the SDOT matvecs + generic
`parallel_for` (video YUV→RGB) across all online cores. **x86 has an equivalent
fleet** (`smp.rs`): APs park in `hlt` with interrupts enabled and are woken by an
all-excluding-self **IPI** (a `pause` spin would cost a core of power per idle AP —
real heat and battery on a laptop), with the same static-partition barrier, the
same claim/done straggler protocol, and its own boot wake self-test. Callers reach
both through **`arch::parallel_for` / `arch::online_cpus`** — never
`arch::aarch64::smp::*` directly. That direct `cfg(aarch64)` call was how x86 came
to run every ONNX op, video row conversion and matvec on one core while aarch64
used the whole machine; a new parallel loop must use the neutral facade or the
divergence comes straight back. **The barrier is bounded, never trust a worker wake**: workers
enable the counter event stream (`CNTKCTL_EL1`) so `WFE` self-wakes, a
claim/done protocol recomputes a straggler's range on the BSP, and a boot-time
wake self-test (`smp: wake self-test ok|FAILED` ktrace) degrades to single-core
up front on hypervisors that park a trapped `WFE` until an interrupt —
VirtualBox-ARM does exactly that and used to hang the first prefill matvec
forever. Slow beats stuck; any new cross-core wait needs the same bound.
**The PSCI gate fails open**: bring-up skips PSCI only when a *valid* FDT
explicitly lacks an `arm,psci-*` node (Apple Silicon via m1n1 — `hvc` there
halts the guest). Boots with **no FDT in x0** (QEMU/VBox `-kernel` ELF, the
UEFI stub) keep PSCI — gating those on FDT contents once silently turned SMP
off on QEMU (`fdt::present` distinguishes "no FDT" from "FDT says no PSCI";
the `smp: N cores online` ktrace is the first thing to check when inference
is inexplicably slow). QEMU vCPU count comes from `CHITTI_SMP` (default 8).
Also NB: `make`'s `RELEASE` defaults to **1** — a dev kernel's unoptimized
NEON is many times slower and reads as an inference bug.

## STANDING RULE — real hardware, nothing hardcoded to an emulator

Drivers must target **real, standards-based hardware**, not QEMU or VirtualBox
quirks. Do not hardcode addresses, resolutions, device layouts, or behaviour to a
specific emulator/hypervisor. Discover hardware the way real firmware does
(ACPI/PCIe ECAM, UEFI GOP, fw_cfg, HID report descriptors, PrimeCell IDs, EDID/
mode tables) and degrade gracefully when a facility is absent. A feature that only
works under QEMU is not done.

**The display mode comes from the display, via EDID — never a constant, never
"the biggest mode advertised".** The kernel itself holds **no resolution at all**:
`width`/`height`/`pitch`/pixel-format arrive from the firmware (Limine's
`Framebuffer` on x86, the stub's boot-info page on aarch64, m1n1's prepared
framebuffer on Apple Silicon) and the font scale is derived from the height
(`pick_scale`), so every layout is a ratio of whatever the panel turned out to be.
That means the *only* place a resolution is decided is the loader, and on real
hardware the chain is **monitor EDID → loader picks the mode → kernel adopts the
framebuffer geometry**.

`kernel/src/edid.rs` parses the EDID base block
(header + checksum validated, then the first detailed timing descriptor) into the
panel's native resolution; the aarch64 `stub/` mounts that same file with
`#[path]` so the two can't disagree, and its pure bit-packing is unit-tested
(`cargo xtask test`) rather than only on hardware. The selection order is
**EDID-preferred → keep the firmware's current mode → largest advertised mode**,
and the middle step is the load-bearing one: with no EDID, the mode the firmware
is already in *is* the resolution the platform was configured for (VirtualBox's
`VBoxInternal2/EfiGraphicsResolution`, UTM's display setting), so overriding it
throws away the user's choice. Both arches had this wrong in different ways —
`kernel/limine.conf` pinned `resolution: 2560x1440`, and the stub always jumped
to the largest GOP mode — which is why a VirtualBox guest came up at a fixed QHD
surface regardless of its settings. On x86 the fix is simply **not** to set
`resolution:` (Limine then queries EDID itself, falling back to 1024x768 with
none); `CHITTI_RESOLUTION=WxH cargo xtask image` appends an explicit override for
a headless VM. Only fall back to the largest mode when there is no EDID *and* the
firmware's mode is below 1024x768 — a default nobody chose, which is the real
"UEFI came up at 800x600" case the largest-mode heuristic was written for.

**A hypervisor's resolution knob cannot be trusted, so there is a channel that
does not need one.** `VBoxInternal2/EfiGraphicsResolution` is *stored* by
VirtualBox-ARM and then ignored — the guest boots at the host panel's size
whatever it says — which left no way to ask for a framebuffer that fits the VM
window (VirtualBox draws the guest 1:1, so a 2560x1440 guest in a 1440-wide
window has half of itself off-screen; that is a clipped console, not a driver
bug). So the loader reads a preference off the ESP: `\chitti-display.cfg`,
`resolution=<W>x<H>`, parsed by `edid::parse_boot_cfg` and applied with GOP
`set_mode` **before the kernel starts**, outranking even the display's EDID-native
mode because it is the one size a human typed on purpose. `CHITTI_RESOLUTION=WxH`
now writes it at image-assembly time on aarch64 exactly as it appends
`resolution:` to `limine.conf` on x86, and `make vbox VBOX_RES=` goes through it.
Three rules it must keep: **empty means unset** (a wrapper passes the variable
through unconditionally), the depth component of Limine's `WxHxBPP` is **dropped**
(a GOP mode is chosen by dimensions, so writing it out would be ignored silently),
and a request is a **ceiling** — `best_mode_for` never exceeds it, since the reason
for asking is usually that something bigger does not fit. The stub logs the
firmware's entire mode list and says so when the requested size was not on offer;
without that, "the resolution I asked for did not happen" has two
indistinguishable causes (never offered vs. `set_mode` failed), and on a machine
that will not boot right that distinction is the whole diagnosis. Verified by
booting `--uefi` under AAVMF, which offers only 640x480/800x600/1024x768: a pinned
1024x768 reaches `framebuffer TUI up (1024x768)`, a pinned 1280x720 correctly
lands on 800x600 (1024x768 is *taller* than 720) and says why, and an unset build
writes no file and behaves exactly as before. `/display boot` still only *records*
a preference — the kernel would have to write FAT to mirror it to the ESP — and
says so rather than implying a reboot will apply it.

**Resolution is a setting, and there are exactly two kinds of it** — `/display`
(`kernel/src/display.rs`, pure + unit-tested; persisted to
`/configs/core/display.json`; also exposed to the **settings agent** as the
`display` shell tool, which may apply it directly since it is reversible):

- **The logical desktop** (`/display set <WxH>|native`) — applies *instantly* on
  both arches. `width`/`height` on `Screen` are the **logical** desktop and
  `origin_x`/`origin_y` place it inside the physical framebuffer, so a smaller
  resolution is a centred, letterboxed viewport that still renders **1:1** —
  glyphs are rasterised at physical pixels, nothing is scaled, text stays sharp.
  The entire translation is one function, `Screen::fb_offset`, which every
  framebuffer write goes through (there are only **five** such sites — `put_pixel`,
  the row blit, the cursor read-back, the pane scroll's row copy, and
  `read_rgb32_row` (the `/screenshot` read-back, added deliberately: it is the read
  mirror of `blit_rgb32_row`); keep it that way). NB the *physical* accessors —
  `fill_phys` and its read mirror `read_phys_rgb32_row` — bypass it on purpose,
  since a letterbox bar and a whole-panel capture are outside the desktop by
  definition. At native both origins are 0 and it is the identity, so the default
  path is byte-identical to before. Note `rebuilt`/`relayout` must feed `build` the
  **`fb_w`/`fb_h`** physical size, never `width`/`height`, or the viewport shrinks
  on every rebuild; and `pick_scale` takes the *logical* height so a smaller
  desktop gets proportionally sized text.
- **The font scale** (`/display scale <1-4>|auto`) — cells are `8*scale` x
  `16*scale` px, so **this** is what answers "everything is too small on a
  high-resolution screen"; a smaller desktop only letterboxes. The automatic value
  is `display::auto_font_scale(height)`, thresholds not a division: the old
  `(h + 550) / 1100` needed **1650** px to reach scale 2, so a 2560x1440 panel
  rendered at scale 1 — 8x16 px cells, 320 columns — which is what actually made a
  2K display look broken, independently of which mode the loader picked.
- **The panel's own mode** (`/display boot <WxH>|auto`) — only the loader can set
  this, so it costs a reboot. **This is recorded but NOT yet applied**: the
  preference lives on the ext4 store and the loader can only read the ESP, so
  mirroring it there (a FAT write via `block::esp`, then the stub reading it and
  the x86 `limine.conf` `resolution:` line being rewritten) is the missing bridge.
  Until then `/display boot` says so explicitly and points at the platform knobs
  that do work (`VBOX_RES`, `CHITTI_RESOLUTION`) — do not let it claim otherwise.

**Display settings are stored per monitor, the way `monitors.xml` does it.** The
stub copies the chosen output's **EDID base block** into the boot-info page
(length at 384, block at 388) — the firmware's buffer is gone by the time the
kernel runs, so it is handed over or lost, the same handoff Linux's EFI stub makes.
`edid::identity` unpacks the display's own vendor/product/serial (the manufacturer
code is three **five-bit** letters packed big-endian in bytes 8..10 — reading it as
a `u16` gives nonsense) and `edid::monitor_name` reads the `0xFC` descriptor, so
`/display` can name the output it is talking about. `display::profile_key` keys the
settings on that identity, falling back to `fb-<W>x<H>` where no EDID is published
(hypervisors, and the x86/Limine path, which passes none) so two
*differently-sized* monitors still get separate profiles. `display.json` is
therefore a `displays: { key: {logical, font_scale} }` map plus a global
`boot_mode`; the older flat shape is adopted for the display in use and migrated on
the next save rather than discarded. A save rewrites **only** the current
display's entry.

**KMS — real kernel mode setting** (`kernel/src/kms/`) follows Linux's DRM split:
`kms/mod.rs` is the device-independent core (`Mode`/`Connector`/`Scanout`, the
`DisplayDriver` trait, mode-set orchestration, damage accumulation, polled
hot-plug), and each device is a backend. **virtio-gpu is implemented and verified**
(`kms/virtio_gpu.rs`): `GET_DISPLAY_INFO` → `RESOURCE_CREATE_2D` →
`RESOURCE_ATTACH_BACKING` (the device scans out of **our** DMA pages, so the
compositor draws with no copy) → `SET_SCANOUT`, then
`TRANSFER_TO_HOST_2D` + `RESOURCE_FLUSH` to present. Confirmed by screendumping the
virtio-gpu device itself: a `1280x800` console really becomes `1280x720`, full
panel, scrollback intact.

Four things that path teaches:

- **virtio-gpu does not scan out of guest memory continuously.** Drawing alone
  changes nothing on screen — it must be transferred and flushed. Damage is unioned
  by `kms::damage` from the *coarse* painters (`fill_rect`, `blit_rgb32_row`,
  `draw_str`, `redraw`) and flushed once per `upkeep`; `put_pixel` deliberately does
  **not** report, because a redraw is millions of calls and a per-pixel union costs
  more than the flush it feeds. A flush per glyph is a queue round trip per glyph.
- **Damage is in physical coordinates**, so the logical-viewport origin has to be
  added — a scanout is the whole framebuffer, not the desktop.
- **A KMS-only machine has no framebuffer to re-init.** `reinit_scanout` therefore
  *initialises* the console when none exists; without that, booting with only a
  virtio-gpu gives a blank screen (found by doing exactly that).
- **On aarch64 `-kernel` there is no PCI** (ECAM comes from the stub's ACPI), so
  virtio-gpu binds on the **UEFI** path and on x86, not on the plain `-kernel` dev
  loop. Matched by **vendor+device id**, never display class: `virtio-gpu-pci`
  reports class `03:00` like every other VGA device.

**VMSVGA (`kms/vmsvga.rs`) works** — the backend for **VirtualBox** and QEMU's
`vmware-svga`. Verified by screendump: `1024x768` -> `1280x720`, clean. Getting there
needed three things that are each easy to get wrong:

- **The FIFO, not just the mode registers.** SVGA II ignores `WIDTH`/`HEIGHT` until
  its command ring (BAR2) is configured and `CONFIG_DONE` is *accepted*: set
  `MIN`/`MAX`/`NEXT_CMD`/`STOP`, where `MIN` must clear the extended register area
  (`SVGA_FIFO_NUM_REGS * 4`) when `SVGA_CAP_EXTENDED_FIFO` is set, and `MAX - MIN`
  must leave at least 10 KiB or the device rejects the ring. Before that the registers
  read back whatever was written while the scanout keeps its geometry — the console
  rendered **four times side by side**. `init` refuses to bind if `CONFIG_DONE` reads
  back 0, because driving it in that state makes a mode set silently do nothing.
- **`flush` is NOT a no-op.** In VGA mode the device tracks framebuffer writes; once
  in SVGA mode it only repaints from `SVGA_CMD_UPDATE` in the FIFO. So the driver
  queues an update rect per damage flush and pokes `SVGA_REG_SYNC`. Drawing without it
  leaves the screen frozen on the mode-set frame. (virtio-gpu needs the same thing for
  a different reason — see above — so both backends are damage-driven.)
- **The geometry registers do not report the mode in effect.** Before the guest first
  enables SVGA mode they hold the device's defaults (QEMU answers 640x480) while the
  display is at whatever the firmware programmed through VGA. So the current mode is
  seeded from the **live framebuffer** and only falls back to the registers; trusting
  them made `preferred` 640x480 on a 1024x768 console, which is what a KMS-only boot
  would then have come up in. `MAX_WIDTH/MAX_HEIGHT` is a VRAM ceiling (an odd
  2368x1770 on QEMU), never a mode to offer as native.

Two safety rules this driver had to learn the hard way, both after breaking a real
VirtualBox display:

- **Probing must not change the device.** `CONFIG_DONE` and the ring pointers alter
  how the device scans out, and at probe time the console is still drawing into the
  *firmware's* framebuffer — so writing them in `init` moved the scanout out from
  under a live console and left the display offset and clipped. They now happen
  lazily in `ensure_fifo`, on the first actual mode set. Same rule the I2C/EC drivers
  follow: identification only ever reads.
- **`VMSVGA_ALLOW_MMIO` is off.** `Regs` can reach BAR0 as I/O ports *or* MMIO, but
  only the port path can be tested here (QEMU emulates `vmware-svga` on x86 only,
  where BAR0 is I/O). VirtualBox-ARM needs the MMIO path and its register layout is a
  guess; acting on that guess mis-programmed a real display. So it declines and keeps
  the firmware framebuffer — the KMS layer is optional, and getting these registers
  wrong costs the console, not just a feature. Verify the layout on the target before
  flipping the flag.

Without a bound backend the whole module is inert and the compositor keeps the
loader's framebuffer — the position Linux is in with `efifb`/`simpledrm`
(`nomodeset`): mode fixed by firmware, `/display set` letterboxes instead, console
legibility via font size. Still absent: real-hardware GPU drivers (i915/AMD/AGX) —
see the note on why there is no display equivalent of xHCI/AHCI.

**A machine can have more than one display, so the stub enumerates every graphics
output** (`locate_handle_buffer`, not `get_handle_for_protocol`) and picks one via
`edid::pick_output`: the output carrying the firmware's **console-out marker**
(`EFI_CONSOLE_OUT_DEVICE_GUID`) first — that is where the firmware drew its own
boot messages, hence the display the user is watching — then any output with a
readable EDID (proof something is plugged in), then output 0 so a headless box
still gets a console. Taking handle 0 unconditionally, as this did, was a coin flip
between a laptop's built-in panel and its attached monitor: it would read one
display's EDID and set the mode on the other. Each output's `console_out`/`edid` is
logged, so "wrong screen" is diagnosable from the boot log alone.

The **QEMU ramfb** window is a separate path with the same "match the display, not
a constant" rule, and it was wrong in its own way: it scanned `system_profiler`
for the *first* `Resolution:` line and used the **physical** pixel count. On a
multi-monitor Mac that silently picked whichever display was listed first, and on
a HiDPI panel it handed the guest a framebuffer bigger than the desktop showing it
(a 2560x1600 panel whose desktop is 1440x900). `xtask::parse_displays` now parses
the per-display blocks (pure, `cargo test -p xtask`) and takes the **main**
display's *desktop* size — macOS's `UI Looks like` when present, else the physical
size halved for a Retina panel, since Apple's default scaled modes are an
unpublished per-panel table and the pixel count is never the right answer.
`CHITTI_FB_DISPLAY=<name substring>` picks another monitor; `CHITTI_FB_RES=WxH`
pins it. Every detected display is logged with `*` on the main one.

Concretely: display comes from the firmware (Limine GOP on x86, UEFI GOP via the
`stub/` bootloader on aarch64, QEMU ramfb as a fallback); disks via virtio /
NVMe / AHCI over discovered PCIe; input via USB xHCI/HID (keyboard **and** mouse,
report-descriptor-driven), virtio-input, PL050/PS-2, and **HID-over-I2C**
(`drivers/i2c_hid.rs` on `drivers/i2c.rs`, the DesignWare/LPSS master) — the touchpad on
laptops from ~2016, which have no PS/2 aux port. An I2C device **cannot be probed for**:
its address comes from `_CRS`, so it is located by asking the namespace which device
claims `PNP0C50` (`aml::device_by_hid`). Identification only ever *reads*, and
`present()` is a zero-length probe, because the same bus commonly carries the embedded
controller and the PD controller — a stray write there misconfigures real hardware. The
HID descriptor register comes from `_DSM` and is defaulted to `0x0020`, but the
descriptor read is **validated** (length + version), so a wrong guess is detected rather
than silently producing garbage. Report decoding is shared with USB via
`xhci::feed_pointer_report`, so a touchpad and a mouse cannot drift apart; the wall clock from the
RTC / UEFI `GetTime` / the virtual counter — each behind a shared facade with a
per-arch implementation. The same kernel image must run on QEMU, VirtualBox, and
real UEFI hardware.

**On x86, ACPI tables must be mapped before they can be read.** Limine's HHDM
covers **usable RAM**, and the tables live in firmware-reserved regions outside
it — so *both* the raw physical address and its `phys_to_virt` translation are
unmapped, and touching either is a page fault that halts the boot, not a garbage
read a signature check can reject. (Cost two boot hangs to find: `0xf52e0`, then
`0xffff8000000f52e0`.) `acpi::map_table` maps every table page explicitly on x86
and remaps to the header's declared length, since an XSDT can exceed a page.
aarch64 is deliberately untouched there: it has a flat identity map, and
`init_uart` reads SPCR before the frame allocator exists. Relatedly, Limine's RSDP
pointer is **physical on newer protocol revisions and HHDM-virtual on older ones**,
and the two cannot be told apart by trying both — classify by range (higher half =
already virtual), then signature-check.

**FADT field offsets are named constants, and one of them was wrong.** `X_DSDT` is
at **140**; the code read it from **148**, which is where `X_PM1a_EVT_BLK`'s Generic
Address Structure starts. That does not fail loudly — it yields
`space|width|offset|access|address_lo32` reinterpreted as a `u64`, a large
*plausible* number that sailed past the plausibility guard and made the DSDT
unfindable on every machine with a modern FADT. Consequence: on x86 the whole AML
layer silently did nothing — no `_S5_` poweroff, no I2C touchpad, no battery — and
nothing reported an error, because "no DSDT" is a legitimate state. Every offset this
reads is now a named constant with a test that pins it (`x_dsdt_is_at_offset_140_not_148`
spells out the wrong read for the next person), and the pure decoders
(`acpi::fadt_dsdt`, `acpi::fadt_pm1`) take a slice so they are testable off-hardware.
A related trap in the same family: the PM1 **event** block is split down the middle —
status first, enable second — so `PM1a_EN` is at `PM1a_EVT + PM1_EVT_LEN/2`, **not**
a fixed `+2`; assuming `+2` writes an enable mask into the middle of the status
register on any machine with an 8-byte block.

**The power button works, and the FADT says which kind it is.** `drivers/pwrbtn.rs`
arms the ACPI **fixed-feature** button (PM1 status bit 8) and `shell::upkeep` polls it
— one `in` per pump, which is cheaper than routing an SCI on a cooperative scheduler,
and a button press is a human-timescale event. It refuses to arm unless the FADT
described a fixed-feature button, ACPI mode is on (`SCI_EN`), and the event block is
long enough — each refusal ktraced, because "the power button does nothing" is
otherwise indistinguishable between those cases. Two rules keep it from powering a
machine off by accident: an all-ones status read is an **unclaimed port**, not every
event firing at once, and PM1 status bits are **write-1-to-clear**, so the ack writes
*only* the button bit rather than the whole word (which would acknowledge every other
pending event). The **control-method** (`PNP0C0C` GPE) button — what many laptops
use — is *reported, not guessed at*: FADT flags bit 4 says so, and GPE dispatch is
honestly unimplemented rather than silently polling a bit that will never change.
Uniquely for this hardware area, it is **verifiable in a VM**: QEMU's
`system_powerdown` sets exactly that status bit, and the `power_button` e2e scenario
presses it through the monitor (`-serial mon:stdio`, so Ctrl+A c reaches it — no extra
plumbing) and asserts a clean shutdown. `/battery` prints the button's state alongside
the battery's, and `tests/e2e/run.py --only <names>` exists so one scenario can be
iterated on without the 30-minute sweep.

**Poweroff and the scheduler tick are real hardware now, not emulator stand-ins.**
`/poweroff` performs an ACPI **S5** transition (`SLP_TYPa | SLP_EN` to the FADT's
`PM1a_CNT`, with `SLP_TYPa` decoded from the DSDT's `\_S5_` package by bytecode
scan; see the AML note below), keeping QEMU's `isa-debug-exit` write only as a
fallback; it used to write *only* that port, so on a physical machine `/poweroff`
did nothing and left the fans running. The tick prefers the **local-APIC timer**
calibrated against the **HPET** (`arch/x86_64/hpet.rs`), falling back to the
PIT/8259 — both of which a UEFI-only machine may omit entirely, in which case the
old code had no preemption at all and said nothing. Every wait added here is
bounded and the HPET gets a counter-liveness probe, because an unbounded spin on a
dead reference clock hung the boot before a single test ran.

**AML (`aml.rs`) decodes, locates, and evaluates a fail-closed subset.** ACPI
describes anything unenumerable as bytecode in the DSDT, so `aml.rs` is the byte layer:
`PkgLength`, `NameString`, data objects, then `devices()` / `methods()` walking
`Scope`/`Device`/`Method` (all three carry a PkgLength, so their extent is exact).
`device_by_hid` + `device_name` answer "what is *this* device's `_CRS`" — which is the
question a driver asks and a flat scan cannot answer. Three encodings cause all the
bugs, each pinned by a test: **`PkgLength` is asymmetric** (low six bits alone, low four
when more bytes follow), **`NameString` has five forms**, and **`OnesOp` is all-bits-set,
not `0xFF`**. Containment is the other bug source — a parent's body contains its
children, and a `Name` inside a **method body is a local**; both are excluded from
`device_name`, and both were shipped wrong first and caught by tests.

On top of that sits an **evaluator** (`eval_method`, `eval_device_method`, and the
`_with_fields` variants): `Return`/`If`/`Else`/`Store`, `Local0-7`/`Arg0-6`, the
comparison and one-target arithmetic operators, and dynamic `Package` construction. Its
governing rule is that **an unsupported opcode returns `None`, never a value** — an
evaluator that guesses an integer is worse than the validated default it would replace,
which is why it was not half-built earlier. Note two ACPI traps it encodes: **`TRUE` is
all-bits-set**, so a caller comparing against `1` reads every true result as false, and a
method that falls off its end without `Return` yields `None` here rather than the spec's
zero — for these callers "unsupported control flow" is far likelier than a deliberate
zero. `OperationRegion`/`Field` (`regions`, `fields`, `find_field`) locate the named
bit-ranges a method actually reads; a **reserved field entry still advances the bit
cursor** (skipping it shifts every later field, which is how a battery reports a voltage
as a capacity), and an `AccessField` stops the walk rather than continuing with offsets
that would be wrong.

**A real battery percentage is the composition of all of it.** `drivers/ec.rs` is the
ACPI **embedded controller** — `PNP0C09`'s `_CRS` (authoritative; the fixed `0x62`/`0x66`
are only an x86 fallback) driven through bounded spins, with a `0xff` status rejected as
an unclaimed port *before* any command is written and a stale output byte drained so a
read cannot return the previous transaction's value. `drivers/battery.rs` evaluates the
firmware's own `_BST` with a field resolver that reads `EmbeddedControl`/`SystemMemory`/
`SystemIO` bytes, takes last-full capacity from `_BIX` (index 3) or `_BIF` (index 2), and
reports `remaining / last-full` — **last-full, not design**, or a worn battery reads
permanently below 100%. Surfaced as the `${battery}` status-bar variable
(`ui_config.rs`), cached for 5 s because one reading costs an AML evaluation plus a
handful of EC transactions; a variable that resolves to nothing takes its separator with
it, so a desktop's bar is byte-identical to before. Every layer fails closed and ktraces
which step gave up. **None of this is verified on hardware** — QEMU emulates no ACPI EC
and no battery — so the pure arithmetic (bit assembly, `_BST`/`_BIF` shapes, the
handshake against a simulated controller incl. slow/wedged/dead) is what the tests hold.
The touchpad's descriptor register now comes from `_DSM` too
(`drivers::i2c_hid::descriptor_register`) rather than the `0x0020` default — note the
`_DSM` UUID goes in ACPI's **mixed-endian** buffer order, or the table's own `LEqual`
fails and the method silently takes its unsupported branch.

**A HID-over-I2C touchpad reads fine while powered down, which is the trap.** The HID
descriptor is answerable from a device that has never been powered on — so descriptor
parsing appeared to work while no report would ever arrive. `SET_POWER(ON)` then `RESET`
through the **command register** (`drivers::i2c_hid`) is what makes reports flow, and the
command encoding is register-address-LE then argument then opcode: swapping the last two
sends `SET_POWER` as report-type 8 of opcode 0, which a device answers by doing nothing
rather than by NAKing, so the symptom is a dead touchpad and not an error. The sequence is
deliberately only reachable *after* `HidDesc::parse` validates, because those are the
first **writes** this driver makes and the same bus carries the embedded controller.
`sleep()`/`resume()` exist for the suspend path, where the device loses its state.

**A battery percentage needs the AC adapter to be meaningful, and `_BST` will not tell
you.** Once a pack is full, `_BST` reports *neither* charging nor discharging — so a
plugged-in machine and one running down produce byte-identical flags. `ACPI0003`'s `_PSR`
is the missing half (`=` in the status bar), reported as `None` rather than guessed as
unplugged when no adapter device exists. Two more things a laptop needs: `_STA` bit 4 says
whether a bay actually contains a pack (a removed battery leaves its device in the
namespace), and a machine with two packs has two `PNP0C0A` devices — `aml::devices_by_hid`
returns all of them and the capacities are **summed**, because reporting the first pack
presents half the machine's charge as all of it. Flags union: one pack discharging means
the machine is.

**The RTL8125 is not register-compatible with the 8168 it is dispatched with.** The
2.5GbE parts move the interrupt mask and status to 0x38/0x3c and widen them to 32 bits,
and the transmit doorbell to 0x90 — and the 8168 offsets *overlap* those positions, so
driving an 8125 with them writes the doorbell into the interrupt mask. `net::r8169`
carries a per-chip `RegMap` (pure, unit-tested, including that every id the dispatcher
sends here lands in exactly one layout) rather than a comment saying to treat 8125 with
caution. Still unverified: QEMU models no r8169-family part at all.

**Intel WiFi (`drivers::wifi::iwl`) is identification and firmware only, deliberately.**
An Intel radio does nothing until an image is loaded, and the image is chosen by chip
family plus an API version that is a property of the *file* — Linux tries filenames
newest-first, and so does `fw::firmware_candidates`. An unrecognised Intel id is **not**
claimed: the wrong firmware fails a signature check *inside* the device with no error the
host can read, which is worse than the Ethernet dispatcher's silent non-receiving NIC.
The `.ucode` TLV parser refuses a pre-TLV image (leading word non-zero), a wrong magic and
any record claiming more than the file holds, and pads record lengths to 4 bytes — one
odd-length record misaligns every record after it. On top of that, `csr` is the register
map with its pure predicates (all-ones is a floating bus, not data; a `prph` address needs
its access-size bits or the following write vanishes), `context` is the gen2 **context
info** — from AX200 onward the host does not feed firmware section by section, it hands the
device a structure and the device's own loader fetches the image, so nearly all the risk
moves into one struct layout and every offset is pinned with `offset_of!` — and `device` is
the ordering: prepare the card (`NIC_READY` going **clear** is the ready signal; waiting
for it to set never completes), APM init, stop the DMA master *before* resetting so no
transfer is in flight against memory about to be reused, then grab MAC access — proceeding
without the grant is worse than failing, because reads return stale values and writes are
dropped, so bring-up appears to work and the device never starts. Every wait is bounded and
names itself. Bring-up is **command-driven** (`/wifi up`), never automatic at boot: the
same posture AGX and the Broadcom radio take, because an untested driver should not touch a
device just because the machine started. Firmware is **fetched, never
committed** (`cargo xtask iwlwifi-assets` into the gitignored `assets/wifi/iwl/`), which
is the same rule the Broadcom assets follow and the reason this needed no licensing
decision. What still does *not* exist: the **receive path** — so firmware's own *alive*
notification cannot be observed, which is why "handed over" is the strongest claim the load
makes — the command round-trip, and then 802.11 + WPA2. So the radio does not associate and
`/wifi connect` still cannot work; `/wifi up` reports how far bring-up got instead.

**Interrupt-controller bases are discovered, and there are two sources, not one.**
aarch64 finds the GICv3 from the device tree's `arm,gic-v3` `reg` when there is an
FDT, and otherwise from the **ACPI MADT** (`acpi::gic_from_rsdp`: GICD type `0x0D`
for the distributor; the redistributor from the GICC entry whose `MPIDR` matches
this core, else the first GICC's `GicrBaseAddress`, else the GICR discovery
range). Both windows get `map_device_gib`. This matters because the two cases are
*different real platforms*: QEMU `virt` boots via `-kernel` with an FDT, while
VirtualBox-ARM, UTM and real SBSA machines boot the UEFI stub with **no FDT at
all** — and requiring the FDT node left every one of those silently cooperative,
with no timer preemption, forever. An FDT that exists but lacks `arm,gic-v3` is
Apple Silicon: stay cooperative and do **not** fall through to ACPI (there is
none, and probing a guessed base is an uncatchable data abort, not a trappable
UNDEF). The QEMU-`virt` addresses survive only as a last-resort default when an
FDT claims a GICv3 but carries no readable `reg`.

## What exists today (subsystems, not phases)

- **Agent layer** — an orchestrator running a real tool-use loop
  (`model → tool → result → repeat`, budgeted); isolated sub-agents; a shared
  type contract in `agent/types.rs`.
- **Sessions** (`session/`) — serializable message history + todos + env + caps,
  saved/resumed/forked over the memory store (postcard).
- **Tools** (`tools/`) — MCP-shaped registry → Router → Synapse cap+taint gate;
  builtin toolset; provider registration.
- **The OS can explain its own extension mechanism.** `build-agent` is a bundled
  skill (`skills/bundled.rs`): its L0 description triggers on "new agent", "new tool",
  "extend", its L1 body is the whole loop (`agents new|build|validate|install --path|
  test|reload`, the `export function` contract, the `Chitti` surface, and what each
  failure message means), and its L2 asset is a complete working `tools.js` so an
  agent copies correct code instead of reconstructing the ABI from prose. Adding it
  found a small pre-existing wrongness worth knowing about: two bundled skills
  declared `Asset.bytes` by hand and had drifted from their payloads (64 vs 77, 80 vs
  84). Sizes are computed now, and `bundled_skill_assets_are_declared_and_present`
  pins every declared asset to a real payload — which also guards the trap that
  `place_trusted` **silently drops an undeclared payload**.
- **Skills** (`skills/`) — L0/L1/L2 progressive disclosure, signed install with
  consent + capability subsetting, installable skill-agents. **Agents as
  installable apps**: a package is markdown (SOUL.md persona + `skills/*.md`
  procedures) + a manifest (toolset = permissions, `capabilities`, optional ONNX
  assets) + a signature; installed via `/agents install <name> [--yes]`
  (consent modal → grant only the approved subset → `place_agent_home` lands the
  SOUL/docs in `/agent/<id>/`). **Per-agent filesystem sandbox:** every
  non-orchestrator agent is confined to its own `/agent/<id>/` folder — the
  install grants a baseline `Fs @ /agent/<id>/**` cap (`skills::install::
  with_home_sandbox`) and nothing wider unless the manifest explicitly requests
  a broader `Fs` scope, which the consent screen flags as "FULL filesystem
  access". Enforcement is the executor's scope gate (Gate 2.5); `list`/`search`
  are result-filtered by that same gate so a confined agent can't even
  enumerate paths outside its home. The shell agent (orchestrator) is the root
  and keeps `Scope::Any`. A **public registry** (`skills/registry_client.rs`)
  fetches a signed index over HTTP(S) — `/agents search <url> [q]`,
  `/agents install <name> --registry <url>`; registry packages authenticate with
  **ECDSA P-256** (`skills::crypto::verify_p256`, RustCrypto `p256`/`sha2`,
  baked publisher trust store) while local dev/boot packages use the keyed-MAC.
- **Channels** (`channel/`) — cap-gated **byte-stream + datagram IPC** between
  agents (the Linux pipe/socket analog; distinct from `ipc` u64 endpoints).
  `Right::Channel{Read,Write}(ChannelId)` are per-direction ends; a model-emitted
  channel handle is a `Cap` slot in the caller's own table (executor resolves it
  — no ambient authority). Backends: heap ring (`Pipe`), datagram queue, or a
  live TCP socket (`Tcp`, via `adopt_tcp`) so a service agent can hand an
  accepted connection to another agent (`channel_grant`). Synapse primitives
  `channel_create/write/read/close/grant`.
- **Service agents** (`service/`) — long-running native daemons (vs
  request/response reasoning agents): `ServiceSpec {entry serve loop, autostart,
  caps}`, `start`/`stop`/`task_for`/`supervise_tick` (pumped from
  `shell::upkeep`, bounded restarts). Their protocol/codec logic is native,
  deterministic code **below** the determinism boundary — the LLM never
  implements a protocol. Only agents that actually reason from a SOUL are
  **installed agents** — the built-in roster is `doc`, `ssh`, `chess`, `media`,
  `pdf`, `download`, `todo`, plus the app packages `notes`, `paint`, `slides`,
  `minesweeper`, `snake`, `synth` — each a markdown
  SOUL plus a JSON manifest under the repo's [`agents/`](agents/) folder, compiled into
  the image via `include_str!`, signed and installed into `/agent/<id>/` at boot
  by `agent::system::install_all` (same permissioned flow as any package,
  pre-trusted; a content agent's assets land in `/agent/<id>/assets/`).
  `network`/`http` are **not agents** — pure mechanical plumbing (relay bytes,
  parse a protocol; no judgment, no SOUL), living entirely in `crate::service`.
  The web is a **generic pipeline** (`service/pipeline.rs`)
  of single-responsibility stages connected by datagram channels — **all reusable
  infrastructure, none app-specific**: `network` (`service/network.rs`) owns the
  socket and relays raw bytes; `http` (`service/http.rs`) parses the request +
  formats the response (no FS, no socket); `server` (`service/server.rs`) is a
  **generic content runtime** that serves *whichever* content agent the pipeline
  was started for — it runs that agent's model as a bounded ReAct loop (prompted
  with the agent's own `SOUL.md`) which returns a **JSON response object**
  (`{status, content_type/headers, file/body}`); `server.rs` parses that JSON and
  frames the reply. The body is either inline `body` or an asset the agent
  **names** (`file`) / **reads itself** (a `mem_fs_read` `<tool_call>`) — both go
  through the capability- and scope-gated reader confined to the agent's own
  `assets/`, so the SOUL agent *decides and reads* the content while native code
  only parses + frames (determinism boundary intact).
  So **a web server is just `agents/<name>/{SOUL.md, manifest.json, assets/…}`** —
  the SOUL carries the routing/behaviour (model-planned per request, greedy), the
  assets carry the content, and no per-server Rust is written. `doc` is exactly
  such an agent (data, not code). `/agents start <name> [port]` serves that agent
  over the pipeline; `ssh` runs standalone (RFC 4253 version exchange; transport
  is a stub). `/agents services` lists running stages. Git + full SSH transport
  follow the same native-protocol shape. To add a built-in server agent: drop
  `agents/<name>/{SOUL.md,manifest.json,assets/…}` and register it in
  `agent/system.rs` (one line) — or publish it to the registry.
- **Apps — wasm-tool agent packages.** An "app" is an installed agent whose
  deterministic logic ships as **`assets/tools.wasm`** (string ABI:
  `export(args_ptr, args_len) -> i64 = (ptr<<32)|len`, `chitti_alloc` for host
  writes; run by `agent/wasm_rt.rs` under **fuel + memory limits** from the
  manifest — no host imports unless bound). The SOUL carries judgment; the wasm
  carries rules — chess (`chess_legal`/`chess_try_move` from
  `tools/chess-wasm`), doc's HTTP router (`route_request`, `tools/doc-wasm`),
  pdf's document digest (`pdf_digest`, `tools/pdf-wasm` — xref tables+streams,
  ObjStm, FlateDecode reusing the kernel's `image/inflate.rs`, text
  extraction; its page *renderer* is a separate `std` module — see the ring-3
  section), the full **chess game** (`tools/chess-wasm` — rules, board UI,
  and the agent-opponent flow; zero chess code in the kernel), and the app
  suite `notes/paint/slides/minesweeper/snake/synth` (one shared module from
  `tools/apps-wasm`). **git** is a wasm-tool agent too (`git_command` from
  `tools/git-wasm`, `agents/git/`): a real git over the store — loose objects
  (SHA-1 + zlib), `init`/`status`/`add`/`commit`/`log`/`branch`/`checkout`,
  smart-HTTP `clone`/`push` (`.git/config` records `origin`, so `/git push`
  needs no URL), a `.gitignore` matcher for untracked paths, and a working
  directory that starts at the shell's pwd (`git clone` makes a folder named
  after the repo basename in the current dir, like the CLI). It needs **host
  imports** beyond the pure-string ABI, gated by its manifest capabilities:
  `host_fs_*` (write home-scoped unless the manifest grants `fs` scope `any`),
  `host_sha1`, `host_inflate`/`host_deflate` (stored-block zlib + consumed
  length), `host_now_unix`, `host_user_home`/`host_home`, and `host_http`
  (only for agents whose manifest declares a `net` cap). **Chat tool calls run on a fresh instance** — design
  those digest-once (one call returns everything as JSON; the kernel caches)
  and pass binary inputs as base64 — **but a running package-UI app keeps ONE
  persistent instance** (`service/package_ui.rs`): guest statics ARE the game
  state (snake body, mine field, FEN), the guest bump heap resets per call
  cycle in `chitti_alloc`, and no guest static may hold a heap type. UI apps
  paint 256×192 `synapse::ui` surfaces via the draw-op DSL
  (`rect`/`line`/`pixel`/`text` + `board_set`/`board_mark` for boards) with
  per-agent `storage_*` (localStorage-shaped) state; the runtime pump
  (`service/package_ui.rs`) peeks the event queue natively and only drains
  through the audited `ui_event_poll` when events exist (an unpaced poll once
  flooded the audit log at ~1 kHz), forwards clicks/keys to the guest
  `on_click`/`on_key` exports (an app consumes only keys it handles), and
  serves the **model-ask protocol**: any export may return `ask:<prompt>` →
  one model turn over the agent's SOUL → the text back via `on_reply` — the
  wasm builds the prompt and validates the reply, so the model only ever
  chooses (chess enumerates legal moves natively and the agent picks one). Manifests can claim **`command_hooks`** —
  `/open` routes by extension to the owning agent's tool (media owns
  images/audio/video, pdf owns `.pdf`) and rebinds chat to that agent. Build a
  module with `cargo build --release --target wasm32-unknown-unknown` in its
  `tools/<name>-wasm/` crate and copy to `agents/<name>/assets/tools.wasm`
  (checked in; `include_bytes!` at boot). See the wasm-agent recipe gotchas:
  kernel `json_str` unescapes `\n`; `image/inflate.rs` is raw RFC 1951 (strip
  the zlib header) and its `#[test_case]` tests need a shim under host
  `cargo test`.
- **Agent tools in JavaScript, compiled on the machine.** A package's
  `assets/tools.wasm` need not come from Rust: `/agents new <name>` scaffolds
  `~/agents/<name>/{SOUL.md,manifest.json,tools.js}` with two working tools, and
  `/agents build <name>` compiles `tools.js` into that same `assets/tools.wasm`
  **on the running OS** (~90 ms). The compiler is a wasm module in the image —
  [Javy](https://github.com/bytecodealliance/javy)'s QuickJS plugin
  (`assets/wasm/javy-plugin.wasm`, 1.3 MiB, Apache-2.0 + MIT) — driven by
  `agent/js_rt.rs`, with `agent/jsmod.rs` wrapping the bytecode into a module.
  Lower-level: `/js build <in.js> [-o out] [--tools a,b]` and
  `/js call <module.wasm> <tool> '<args json>'`.
  **There is deliberately no new `ToolBinding`, no `js` manifest field and no
  second kind of agent** — JavaScript is a source language, the artifact and the
  manifest are the ordinary ones, so `effect_of`/`origin_of`/`paper-check`/the
  `redteam` census are untouched. Seven things this path pins down:
  - **`compile-src` returns bytecode, not a module**, so `jsmod::emit` writes the
    wasm itself. It is a fixed template decoded from `javy build -C dynamic`: three
    types, three imports from the plugin's namespace, one function + one export per
    tool, and two passive data segments (bytecode, names). Each body reallocs and
    `memory.init`s its inputs then calls `invoke`; `memory.init` takes a source
    offset, so one names segment serves every tool. **The emitted module exports the
    same names a Rust module would**, which is why nothing downstream changed.
  - **A wrong LEB128 length still validates.** So the tests execute the output
    (`javascript_compiles_and_runs_on_this_machine`) and validate it at 1/127/128/
    300/16384-byte bytecode lengths, rather than diffing against a golden blob.
  - **`initialize-runtime` reads a config JSON off fd 0.** Feed it `{}` and rewind
    fd 0 before the tool's arguments, or it eats them and fails `unknown field 'q',
    expected one of 'javy-stream-io',…` — which reads like a bad argument.
  - **`compile-src`'s result is a 3-word area** `[discriminant, ptr, len]`. Read as
    two words you get a plausible pointer and QuickJS rejects it with
    `invalid version (0 expected=26)`.
  - **`invoke` resolves an ESM export verbatim** — snake_case works. The Javy CLI's
    kebab→camel mapping is the CLI's convention, and this emitter bypasses the CLI.
  - **The `ResourceLimiter` capped instances at 1**, and the JS path needs two in
    one store (engine + module). wasmi refuses the second with `tried to instantiate
    too many instances`, which reads like a problem with the module. Now
    `Limits::with_instances`, defaulted to 1 so every other guest is unchanged.
  - **A syntax error costs the engine instance**: the plugin panics inside its own
    WASI shim rather than returning `compile-src`'s error arm, so the session is
    poisoned. `build_module` therefore starts a fresh session per build — a *cached*
    compiler would reject every good script after one bad one.
  Contract differences from a Rust tool, both stated because they surface as
  confusing bugs otherwise: exported functions take **no arguments** and their
  **return value is dropped** (args JSON in on fd 0, result JSON out on fd 1), and
  **module top level re-runs per call**, so JS globals do not persist — durable state
  goes through storage, and package-UI apps (whose guest statics *are* their state)
  stay Rust. A JS call costs **3-4 Mfuel before the script does anything**, hence
  `js_rt`'s budgets rather than `DEFAULT_FUEL`'s 5 M.
  **A JS tool reaches the same gated surface a Rust one does.** The engine is *our*
  Javy plugin (`tools/javy-plugin/`, `javy-plugin-api` 7.1.0, rebuilt by
  `cargo xtask javy-plugin`), which imports the `chitti.host_*` functions and exposes
  them as a `Chitti` global — `storageGet/Set/Remove/List`, `fsRead/Write/List/Exists`,
  `uiDraw`, `hud`, `http`, `notify`, `log`, `sha1`, `home`, `userHome`, `nowMs/nowUnix`. Same
  imports, same gates, so **the authority does not widen**: only who can call it.
  Refusals **throw** into the script rather than returning empty, and "no value"
  returns `null` (localStorage's shape) because `JSON.stringify` drops `undefined`,
  which would erase the distinction. Three more traps here:
  - **Do not strip the plugin crate.** `javy init-plugin` validates through binaryen,
    which reads the `target_features` custom section to know which wasm features to
    permit; stripped, every bulk-memory instruction fails with
    `[--enable-bulk-memory]` — an error naming a *flag*, not the missing section.
  - **`javy-plugin-api` 7.1.0 pairs with CLI v9.1.0.** The published docs say 5.0.0;
    the truth is in `crates/plugin/Cargo.toml` at the release tag, and a mismatch
    surfaces only at `init-plugin`.
  - **A JS exception arrives as a trap** whose message is `wasm unreachable`. The
    reason is the last thing the guest wrote to fd 2, so `Fds::last_stderr` keeps it
    and `/js call` leads with that instead ("storageSet: refused: this agent has no
    such capability bound").
  Rebuilding the plugin changes its stamp, so **every `tools.wasm` built against the
  old one becomes stale** — detected and reported as "rebuild it", not failed inside
  QuickJS.
  **The whole loop runs on the machine** (`agent/local_pkg.rs`): `/agents new` →
  edit → `/agents build` → `/agents validate` → `/agents install <name> --path`
  (the same consent modal a registry package gets) → `/agents test <name> --tool t
  --args '{…}'` → `/agents reload`. Five things that path pins down:
  - **A manifest's `toolset` does not create tools.** `registry::for_agent` only
    *filters* a compiled-in list, so a name with no `ToolDef` is silently invisible
    and a freshly installed local agent could call none of its own tools.
    `local_pkg::register_tools` registers them at install via `register_replace`,
    deciding which names are the package's own from **the module's exports** — a
    toolset also lists borrowed tools like `memory_add`, and registering those would
    shadow the real ones. Reload deregisters the previous set first (remembered in
    the store, since the registry has no per-agent index), so a tool deleted from
    the script stops existing rather than lingering as a def over absent bytes.
  - **A re-install grants `manifest ∩ recorded grant`, never the manifest alone.**
    The package lives in the store, which any agent with a broad `Fs` scope can
    write, so re-reading its requests as authority would make editing a file an
    escalation. `InstallRecord.granted_capabilities` is the ceiling (invariant 5),
    and `a_reinstall_cannot_widen_the_recorded_grant` pins it.
  - **Install records are written to the store and never read back**, so
    `install::load_record` is new and `local_pkg::reinstall_all` runs at boot from
    `main.rs` — without it a local agent's *files* survive a reboot while its role
    and tools quietly do not. NB the live-ISO store is memfs, so this only bites on
    an installed system.
  - **Ids must survive a reboot** because `/agent/<id>/` holds the SOUL, assets and
    memory. `next_agent_id()` is an in-memory counter from 1 and the system roster
    bypasses it with fixed `9000+` ids, so local packages draw from a **persisted**
    counter at `LOCAL_ID_BASE` (20000) and a name always gets its id back.
  - **`parse_manifest` is forgiving because it must be** (it also reads the
    compiled-in manifests at boot, where a hard failure would cost the machine an
    agent) — and that forgiveness hides real mistakes from a human editing a file:
    an unrecognised `kind` silently becomes a *service*, an unknown capability
    `domain` is dropped whole, unknown `rights` vanish, and an unrecognised `scope`
    widens to **ANY**, the worst reading of a typo. `local_pkg::lint` reports each,
    is shared by `/agents validate` and the install path, and blocks an install on
    errors. `/agents test` prints the **structured** `ToolOutcome` — kind,
    provenance, origin — never a parse of the reply text.
- **Notifications** (`notify.rs`, `/notify`) — a bounded, coalescing, persisted
  ring: the channel by which the OS tells the human something happened while they
  were not looking. Three load-bearing properties. **Repeats coalesce** on a
  `dedup_key` (a five-second job bumps a count instead of filling the ring and
  burying what mattered). **The `source` is stamped by the kernel** from the live
  identity and never read from the poster's arguments — an agent that could name
  itself `kernel` has performed a transfer of authority wearing a label. And
  **agents may post but not list**, which removes the laundering channel (post,
  read back, untrusted content re-enters with its tag stripped) for one line of
  policy instead of a tagging path. Surfaced as `StatusChip::Notifications` +
  `${notifications}` — *absent* at zero unread, because `chip_text` returns `""`
  and `ui_config::expand` swallows the following separator, so a quiet machine has
  a byte-identical status bar — plus a dropdown and a `RightMode::Notifications`
  action-pane tab, **and a banner** (`notify/toast.rs` + `framebuffer/toast.rs`).

  The banner reverses an earlier decision in this file, and the reasoning is worth
  keeping: a transient overlay was rejected on three grounds, and two of them were
  about a banner placed *anywhere*. Pinned **top-right of the content rect** they
  dissolve — nothing is typed there (the composer is at the bottom), and a banner
  that does not animate damages exactly twice (appear, lift) rather than once per
  pulse, which `Toast::needs_repaint` is what enforces. The third — a transient
  overlay must save and restore the pixels beneath it on a single-buffered
  framebuffer — is real, and is solved the way the **mouse cursor** solves it:
  `toast_saved` is `cur_saved` with a different name. Anchoring to the *content*
  rect rather than the screen means all four `/statusbar` edges work with no match
  on which edge it is. A full `redraw` invalidates the saved pixels
  (`toast_forget`, called inside `redraw` rather than at its eight call sites, so a
  ninth cannot get it wrong).

  **`/notify on|mute|off`** is three states, not a switch, because "stop making
  noise at me" and "stop recording anything" are different requests and one switch
  answers one of them wrongly. `mute` keeps the queue — which is most of its value
  — and `off` records nothing at all, which is what "fully disable" means. Both
  persist to `ui.json` (normalized through the parser, so a typo leaves
  notifications *on*: the safe reading of an unreadable setting is the one that
  still tells you things), and an empty listing under `off` says *why* it is empty,
  because "nothing is arriving" otherwise has two indistinguishable causes.

  **The chime** is a two- or three-note figure per severity, not a beep: a single
  tone carries no information, so success rises, error falls, and `Action` is a
  longer three-note figure. `Info` never rings at all — a five-second scheduled job
  would be a metronome, and that is how an OS sound gets turned off for good. Every
  chime is under 400 ms, playing is best-effort, and a full audio queue drops it
  (the notification is already on screen; nothing waits on audio).

  **The agent API** is `host_notify` (wasm) and `Chitti.notify(severity, title,
  body?)` (JS). Same three rules as the tool: the source is stamped from the
  binding and cannot be chosen; it is write-only, so a posted notification cannot
  be read back (no laundering channel, for zero policy); and it shares the
  `log_count` rate-limit budget, because the chime makes spam louder than usual.
  `Severity::Action` is deliberately **not** reachable from a guest — it means "a
  human decision is waiting", which only the kernel's unattended-approval path is
  entitled to claim; a guest asking for it gets `Warn`.

  Ships with **one real producer** so it is not decorative —
  `service::supervise_tick` exhausting a daemon's restart budget.
- **Scheduled runs** (`schedule/`, `/schedule`) — a stored intent that acts later.
  Deliberately **not five-field cron**, because the wall clock here may be fiction:
  with no readable RTC `clock::DEFAULT_UNIX` puts the machine in January 2026 until
  `/ntp`. So `every 5m` is measured on the **monotonic** timebase and is correct on
  a lying clock, while `at 09:00 weekdays` / `on 1 03:00` / `in 30s` are calendar
  and are **held rather than fired** while `clock::source()` is untrusted — held,
  and *reported* as held by `/schedule next`, because "my schedule didn't run"
  otherwise has three indistinguishable causes on a box you cannot debug.
  `MIN_EVERY_SECS = 5` exists so an e2e test can watch a fire inside its budget.

  **The authority model is the part to read before changing anything.** A schedule
  is bounded by what its author could have done when it was stored, **forever** —
  the analogue of invariant 5. So: a stored human confirmation does **not** survive
  a change of action (`spec::reauthorise`), or an injection turns a blessed nightly
  `/disks` into a nightly `rm -r` using the human's own approval; provenance only
  ever joins *worse*, so editing cannot launder a tainted job; and human-vs-agent
  authorship is **decided at the call site** (`shell::in_tool_call`) rather than
  inferred from session taint, because a Telegram DM enters as `UserTyped` and
  inferring would hand a DM-authored schedule typed-human authority forever. A job
  authored while untrusted content was resident still runs — its inert calls are
  most of a daily digest — but `blocks_destructive()` holds for its life.

  **No new Synapse primitive** (the count stays 26): a schedule creates no effect
  at creation, so gating it would be `mem_fs_write` in a costume. What it *does*
  add is one real gate — `tools::dispatch::effect_of` classifies `schedule add` by
  the effect of the **action being installed**, so `add … command rm -r` hits the
  destructive check while a human is still watching. Same shape as the documented
  `channel send` misclassification, found the same way.

  **Where the work happens is the `msgchan` split, and each reason is specific.**
  `tick()` runs on `upkeep` and only *enqueues*; the fire happens in the interactive
  loop's drain (inference is too heavy for the poll tick, and `ChatSession::turn`
  brings its own 1 MiB stack). `with_busy` guards the genuine
  tick→run→`/http`→`upkeep`→tick re-entry that would spin a non-reentrant `Locked`
  forever with interrupts off. `ReadOutcome::ChannelWake` became `Wake` because
  without it a schedule does nothing while the prompt idles — which is most of the
  time, and the whole situation a scheduler exists for. Fires are **at-most-once**:
  the due time is advanced and persisted *before* the run, so a crash loses a run
  rather than repeating an irreversible one.

  An unattended run **cannot answer a modal**, so `execute_chat_tool_inner` refuses
  an approval-requiring call, records `Outcome::NeedsApproval`, and posts a
  `Severity::Action` notification. "The OS tried to do the thing you scheduled,
  could not, and is asking you" is a completion of the agentic loop that neither
  half of this could provide alone.
- **Screenshot** (`/screenshot`, `screenshot.rs`) — capture to PNG in the store.
  `image::png::encode_rgb8` does per-row adaptive filtering over
  `image::deflate::zlib_compress` (fixed-Huffman LZ77), which takes a 6.2 MB 1080p
  frame to tens of kilobytes; `zlib_stored` remains for the git object path, where
  "the bytes are literally in there" is worth more than a smaller `.git` until
  something has verified our output against libz. **Every encoder test round-trips
  through our own inflater** — an encoder verified against an independently written
  decoder in the same tree is a much stronger claim than one verified against its
  own inverse, and it immediately caught a cheap-reject probe reading one byte past
  the end. The capture reads through `read_phys_rgb32_row` (the read mirror of
  `fill_phys`, with `blit_rgb32_row`'s native-XRGB fast path), **lifts the mouse
  sprite first** (it is composited *into* a single-buffered framebuffer, so a naive
  capture includes it and reads as an artefact in a bug report), and refuses above
  `MAX_PIXELS` rather than failing inside a first-fit allocator. A **model-chosen**
  capture from a non-root agent is narrowed to the surface that agent owns — the
  gate is `in_tool_call()`, not `active_agent_id()` alone, because those are
  different questions and conflating them refused a *human* typing `/screenshot`
  while the chat happened to be homed to another agent.
- **Messaging channels** (`msgchan/`) — external inbox adapters (Telegram
  live; Discord/Slack/webhooks follow the same shape): a named instance +
  backend + access policy delivering inbound DMs into the shell agent and
  replies back out. Each channel turn runs on a **fresh model context**
  (`channel_turn` swaps KV/history out and back) so DMs can't stick to a
  console topic and console chat isn't polluted; transcripts still land in the
  session for audit. Distinct from `channel/` (cap-gated inter-agent pipes).
- **Cortex** (`cortex/`) — CPU transformer inference, **architecture-dynamic
  like the ONNX interpreter**: `general.architecture` in the GGUF names the
  hyperparameter key prefix and resolves to a `Family` — `QwenHybrid`
  (Qwen3.5/3.6 DeltaNet+attention hybrid; finetunes like Ornith load via a
  key-shape sniff) or `Gemma4` (sliding-window/global interleave with per-kind
  geometry — GQA ring-KV local layers, MQA global layers with V=K + p-RoPE
  freq factors — sandwich norms, GELU, √dim embed scale, logit softcap,
  `layer_output_scale`, suppress-token bias). Nothing numeric is compiled in.
  **All mainstream GGML quants dequantize** (legacy Q4_0/Q4_1/Q5_0/Q5_1/Q8_0,
  K-quants Q2_K–Q8_K, i-quants IQ2/IQ3/IQ4 via `iq_tables.rs` — generated
  verbatim from ggml by `tools/gen_iq_tables.py` — plus F16/BF16 rows), so any
  unsloth file incl. UD-* dynamic mixes runs; plus **PrismML's sub-2-bit packs**
  (`Q1_0` binary type 41, `Q2_0` ternary type 42 — 128-elem blocks, one f16
  scale; the Bonsai-27B builds). Fast SDOT matvecs for
  Q8_0/Q4_0/**Q4_K**/**Q1_0**/**Q2_0** (the Q1_0 sign-expand and Q2_0 code
  unpack run fully in vector registers — `vqtbl1q` broadcast / `vzip`
  interleave, loads via the `ldq_*`/`ldp_*` asm helpers), everything else
  through the generic dequant path (still SMP row-split). **Batched
  (weight-stationary) prefill is decided per *tensor*, never per model**
  (`has_batched_kernel`): Q8_0 ∣ Q1_0 ∣ Q2_0 always have a kernel, Q4_0 only
  with FEAT_I8MM, K-quants none — and a projection without one falls back to a
  matvec per position instead of disqualifying the whole file. The old
  per-model gate demanded one uniform quant type and read it from
  `token_embd.qt`, which is the worst possible anchor: prefill never batches
  `token_embd` (a row lookup, plus one matvec for the final position on a tied
  output). Real GGUFs are almost never uniform — `llama-quantize` upcasts
  selected tensors unless `--pure` is passed, so the file published as "Q4_0"
  arrives Q4_0+Q8_0+Q5_K+Q4_1 with `token_embd` at Q6_K, and that one tensor
  put a whole 4B on the per-token path. **A fallback tensor costs far more than
  its share of bytes**: `ssm_out` at Q5_K has no SDOT matvec either, so it goes
  through the *generic dequant* path — 11% of projection bytes was eating ~69%
  of prefill, ~18x less efficient per byte than i8mm. Hence
  `xtask/fetch-model.sh CHITTI_PURE=1|bf16`, and `/perf`'s `batched weights: N%`
  line. Measured on the 4B (`/perf 512`, 8 cores): per-token 3 pp / 1 tg →
  windowed at 89% batchable → **pure file 23 pp / 6 tg**. The shell chat loop
  feeds **64-token chunks** through `Model::prefill` (weight bytes + unpack
  amortized per chunk; UI pump + Ctrl+C between chunks), so a 27B's ~1.5k-token
  system prompt prefills in minutes, not hours. Chunk size is measured, not
  assumed — sizing it to the heap (128/256) was **slower** on a host sweep
  (4.06 s at 64, 4.17 s at 128, 4.22 s at 256): a bigger chunk only cuts weight
  *traffic*, and prefill is compute-bound while the wider activation tile costs
  cache locality. NB the paired Bonsai `dspark` GGUF is a *drafter*
  conditioned on the target's hidden-state taps — not a standalone model; the
  runnable Bonsai is the main `Q1_0`/`Q2_0` file. Two tokenizer flavors behind
  one API (GPT-2 byte-BPE ∣
  gemma4 raw-UTF-8 ▁-BPE with `<0xXX>` fallback), per-family chat format in
  the shell (ChatML ∣ `<start_of_turn>` gemma turns, BOS per `add_bos`).
  Select with `-model qwen3.5-0.8b|2b|4b|9b|gemma-4-e4b|bonsai-27b|
  bonsai-27b-ternary` **or any path**
  (`-model path/to/file.gguf` — guest RAM derived from file size), or at
  runtime with **`/model load <file.gguf>`** (reads off any FAT/ext4 volume
  into DMA frames and re-homes chat on it; the status bar shows the GGUF's own
  `general.name`). Zero-copy GGUF; grammar-constrained sampler; KV/recurrent
  cache (fixed W-slot rings on sliding layers). **`tools/cortexdiff/`** is the
  host-side harness (the onnxdiff pattern: mounts `kernel/src/cortex` natively;
  greedy decode in seconds; `diff.py` cross-checks tokenization + continuation
  against llama.cpp) — it generates the `refcheck.rs` fixtures (keyed by
  `general.name`; the numpy `tools/ref*.py` are gone) and is the required
  bring-up tool for any new family/quant. `cargo xtask ref-check
  [-arch aarch64] [-model …]` runs the acceptance gate natively under HVF
  (minutes, vs TCG hours) and powers off via PSCI.
- **Synapse** (`synapse/`) — the capability ABI: primitive registry, GBNF-style
  grammar, deterministic executor, append-only audit log, taint gate. Primitives
  now span fs/console/spawn, **channels** (10–14), **net** listen/accept + http
  (15–18), and **UI surfaces** (19–22). The executor runs a **scope gate
  (Gate 2.5)**: a granted narrow scope (an fs path glob, a `Net{host,port}` range)
  is enforced against the concrete target (`scope_target` + `cap::scope_check`) —
  deny-only-when-recorded, so `Scope::Any` grants and un-scoped tasks are
  unaffected. `CapDomain` gained `Channel`/`Net`/`Ui`; `synapse::ui` owns the
  surface registry + bounded draw-op DSL (ownership-gated: an agent can only
  draw to its own surface). The audit **log** records every entry
  (append-only, structurally enforced + tested); its **ktrace mirror**
  coalesces identical consecutive entries into one line + a repeat count, so a
  polling loop can't drown the human-facing trace. NB: sessions that use net
  egress or UI input aren't replayable from a seed alone (the I/O is external)
  — the audit log records the effects; treat such a session as
  non-deterministic to replay.
  **What the boundary costs is measured, not asserted** (`synapse::bench`,
  `/bench synapse`): the gate chain is priced through `executor::gate_prefix`,
  which runs the real predicates in the real order but executes no primitive and
  writes no audit entry, against a **synthetic parked task** whose table + scope
  ledger are granted explicitly and killed after — measuring against the shell
  agent would make a figure depend on what the session holds, and pricing a
  *denied* call would mean granting an agent a right to measure it.
  `gate_prefix` is a second copy of the gate order, so
  `gate_prefix_agrees_with_execute` pins it to the real chain: a new gate that
  isn't added to both fails that test. Three benchmark traps, each of which
  printed a plausible wrong number first: a result-discarded pure call is
  **deleted** by the optimizer (the FNV row read 0 ns over 16.7M iterations —
  everything timed goes through `black_box`, and a zero-ms batch is flagged
  SUSPECT rather than printed as 0); cumulative prefixes must share **one batch
  size after a warm-up**, or the first batch pays to grow the heap and the curve
  comes out *decreasing* (making every marginal cost an artifact); and
  `cycle_count` is a **constant-rate tick, not a CPU cycle** (~24 MHz
  `CNTVCT_EL0` on Apple silicon), so the rate is printed beside every figure. A
  non-positive marginal cost prints "below noise floor" — a saturating
  subtraction is not evidence that a gate is free. The design write-up lives in
  [`paper/`](paper/).
  **The taint policy is enforced at EIGHT sites, not one, and `/redteam`
  (`security::redteam`) is the census.** The Synapse executor gates destructive
  primitives; the tool router *separately* gates destructive shell commands,
  `/http`, MCP `tools/call`, agent-memory mutations, downloads, browser
  navigation, nested `run_shell_command`, and the web tools — so a **new tool
  binding that forgets its check is a hole by omission**, and the corpus carries
  one attack per site precisely so that shows up as a permitted attack instead of
  as silence. It runs through the **real `Router`**, so the justification is
  computed by `Router::justification` over `Session::resident_max_taint` exactly
  as in an agent turn; a laundering bug on the way to the gate would surface as a
  `NOT TAINTED` row. Three rules this harness must keep. It **assumes the
  injection persuaded the model** (the payload text is never read by a gate) —
  that is the worst case and it is what makes the numbers deterministic and
  model-independent; it measures the *authorization* boundary, not the planner.
  A **permitted attack really executes** under the baselines, so every target is
  a sandbox path or the loopback discard port, pinned by
  `corpus_targets_are_sandboxed_and_offline` — this is why the destructive-shell
  attack is `rm` on a sandbox file and never `install`, and why the victim runs
  as a throwaway agent identity (`REDTEAM_AGENT`): the memory-poison attack, if
  permitted, writes durable memory that re-enters the system prompt, so running
  it as the live orchestrator would poison the real shell agent as a side effect
  of measuring whether that was possible. And **a non-policy error counts as
  permitted** — a refused loopback connection is not a defence. The counterpart
  measurement is the utility suite: benign tasks over the same primitives, whose
  **false-refusal rate** (a destructive step refused when the untrusted content
  never named its target) is the number that decides whether the policy is
  usable, and it is reported next to the attack rate rather than separately,
  because a defence is only interesting if it is good on both axes.
- **Microkernel** — tasks + context switch, cooperative + timer-preemptive
  scheduler, unforgeable capabilities, IPC, SMP, frame allocator + heap, MMU.
- **UI** — a tmux-style split-pane framebuffer compositor in Geist Mono, living
  in [`kernel/src/framebuffer/`](kernel/src/framebuffer/): **`mod.rs` owns the
  data model** (every type, constant and static — a child module sees its
  parent's private items, so a `Screen` field is reachable from every painter
  with no visibility annotations) and one submodule owns each surface
  (`paint`/`text`/`pane`/`layout`/`tabs`/`focus`/`status`/`menu`/`clock`/`modal`/
  `composer`/`views`/`surface`/`select`/`cursor`/`console`/`colors` — the map is
  a table at the top of `mod.rs`). Each is re-exported, so every
  `crate::framebuffer::…` path is flat regardless of which file an item is in;
  put a new painter in the module it belongs to rather than growing one file back
  to the 8.7k lines this was. The
  shell (chat) pane is fixed in the primary band; the other band is a
  **resizable grid of 1–8 action panes** (`/pane grid <cols> <rows>`, or
  `/pane max <2-9>` for a balanced shape — `panes_layout::grid_for_count`).
  **Every divider drags**: the shell|band split (`/pane split <10-90>`) and each
  grid column/row gap, and a grid drag is **per-gap** — it re-splits only the two
  tracks it separates, so panes you weren't touching keep their exact pixel
  sizes. Either band can go **fullscreen** (Ctrl+F, or `/pane full` —
  `LayoutCfg.fullscreen`, which maximises the **focused** action pane, not pane
  0). Shape, `chat_pct`, and the permille track weights persist to
  `/configs/core/panes.json` and reload at boot (legacy `num_action_panes` is
  still read as an action-pane count). The geometry is **pure and unit-tested**
  in [`kernel/src/panes_layout.rs`](kernel/src/panes_layout.rs) in two steps —
  `split_band` (chat | band, one gap) then `layout_grid` (`cols × rows` cells
  from track weights) — with `band_divider_pct` / `resize_tracks` as their exact
  inverses, asserted by round-trip tests; weights are permille so a saved layout
  restores byte-identically and `GridSpec::sanitized` repairs a hand-edited file
  rather than producing a zero-size pane (`MIN_TRACK_PX` bounds every drag).
  Panes are addressed **row-major** (`index = row * cols + col`).
  **Tabs move between action panes by drag-and-drop** — press a tab label, drag
  past a ~4 px threshold (the target pane highlights), drop on a tab bar to
  insert there or on a body to append; a drop on the shell pane or outside the
  band **cancels**, so the shell can never acquire an action tab. The
  insert-index math is `panes_layout::insert_index` (the same-pane removal shift
  must be applied **before** the clamp, or a drop-at-the-end lands
  second-to-last — a test pins every from/to slot pair).
  With `max_panes == 2` the band still collapses when its last tab closes, so
  the default boot UI is byte-identical to the classic two-pane look; with
  `max_panes > 2` empty panes stay visible as drop targets.
  Note three traps this cost: **a pane's frame carries its selection state**, so
  focusing one must repaint the pane *losing* focus too, and
  `focus_action_column` must repaint **itself** — it has already moved focus to
  the action side, so a following `focus_set(true)` sees no flip and draws
  nothing (which made clicking a pane change the selection invisibly).
  **Opening a view must never move keyboard focus** to the action pane
  (`open_view_slot`): the user typed that command at the composer and is still
  typing there, which is exactly why `action_focused` leaves focus on the chat for
  every mode but the editor — setting `focus_action` on open sent the *next*
  command to the pane instead of the prompt, indistinguishable from the shell
  freezing. Focus moves only on an explicit act (click, `/pane focus`, Ctrl+Tab).
  And the
  per-view painters (`/top`, the audio/video/browser HUDs, the editor,
  `surface_dims_px`) resolve their target pane by **which pane holds their
  tab** (`Screen::mode_dims`), never the focused one — otherwise a `/top` on
  pane 3 paints into pane 1's rectangle as soon as you click elsewhere.
  The compositor pairs the chat
  pane + on-demand **tabbed "action" panes**: opening the ktrace stream, the
  `/top` dashboard, a vim-like editor, an **image viewer** (`/open .png|.jpg`),
  or the **audio player** (`/open .wav|.mp3`) each adds a tab **on the focused
  action pane** (already-open views focus their existing pane + tab instead of
  reopening); a tab bar in each
  pane header switches them (Ctrl+Tab / Shift+Tab / click), and every tab keeps
  its process alive when you switch away — the audio player keeps playing
  (pumped chunk-by-chunk from `ui_tick`, not a blocking loop), ktrace keeps
  streaming, the editor keeps its buffer. The editor is non-blocking: its state
  lives in a static and `read_line` routes bytes to it while its tab is
  focused. **Both media tabs take key controls while the action pane is focused**
  (Ctrl+Tab / click — same gating as pane scroll, so typing at the prompt is
  never intercepted): the image viewer does `+`/`-` zoom, `r`/`l` rotate,
  arrows pan, `0` reset (retaining the source, capped to ~4 MP, and re-rendering
  via the pure `image::render_view`/`rotate90`); the audio player does space
  play/pause, `←`/`→` seek ±5 s, `↑`/`↓` ±30 s, `0`/Home restart (state on the
  `AudioPlayer` static, seek just moves the PCM cursor, pause holds it while the
  device drains its queued chunk to silence). The **image viewer** decodes
  in-kernel — `image/` is a pure no_std PNG+DEFLATE and baseline-JPEG decoder
  set, unit-tested against real fixture files (plus `rotate90`/`render_view`
  transform tests) — then presents box-downscaled + letterboxed), a boot splash +
  status-bar **Synapse-C** brand mark, a live
  clock, a blinking caret, **Enter submits a fully-typed command** — the suggestion
  menu stays open while the typed token still matches an entry, so a command name
  typed in full kept *its own* entry highlighted and Enter "accepted" it, appended a
  space, and swallowed the keystroke; every later line was then one out of step, so
  a command silently did not run and it read as a frozen shell. `suggest_would_
  complete` gates the Enter path on whether accepting would change anything beyond
  the trailing separator (Tab is untouched — completing is its job). This is what
  made the command after `/todos open` never execute, an e2e gap that sat unexplained
  for a while; the `pane_grid` scenario now asserts it. Path-taking commands
  (`/ls /cat /open /mkdir /rm /cp /mv /touch /glob /grep` — `suggest::PATH_COMMANDS`)
  additionally get **path-argument autosuggest + completion**: as you type the
  argument, the popup lists the parent directory's store/mount entries
  (`suggest::path_items` over `vfs::readdir`), dirs first with a trailing `/`,
  and Tab drills one level at a time. The same Enter gate is extended so a
  *complete* path argument submits the command rather than completing or
  drilling (the case that swallowed `/ls /tmp_e2e` into `/ls /tmp_e2e/`); Tab
  remains the drill key.

  **The shell has a working directory, and every path-taking command resolves
  through ONE function — `shell::resolve_path`.** The shell agent starts in the
  ChittiOS user home (`agent::home::USER_HOME`, `/home/chitti` — the `~`); `/cd
  <dir>` moves it (bare `/cd` and `/cd ~` → home; `.` stays, `/` is the store
  root) and `/pwd` prints the live value (`shell::shell_cwd`/`set_shell_cwd`).
  `resolve_path(p)` implements the
  Linux rule once: `/abs` stays absolute, `~/x` → home, anything else →
  `<pwd>/<x>`, with `.`/`..`/`//` collapsed (`vpath::normalize`). It is applied
  at **every** fs-command call site (`ls`/`cat`/`glob`/`grep`/`touch`/`mkdir`/
  `cp`/`mv`/`rm`/`open`/`edit`) and to the path-completion popup's readdir — a
  bare `/ls` lists the **pwd** (not the store root), `cat hello.txt` and
  `~/file` work, and completion keeps the typed form (`re<TAB>` → `rel.txt`,
  `~/h<TAB>` → `~/homedoc.md`, `/co<TAB>` stays absolute). **Never re-implement
  relative resolution at a call site** — route through `resolve_path`, and pass
  the shell's cwd to command-hook agents (e.g. the git agent gets `{"cwd":…}`
  so `git clone`'s default folder is the pwd, like the git CLI). mouse cursor + click, **mouse text selection in the
  chat pane** (drag-to-copy → clipboard, paste with Ctrl+V; absolute-indexed
  over scrollback via `textsel`, like the editor's drag-select), a **host
  clipboard bridge** (`clipboard`: an in-OS copy emits an **OSC 52** escape so a
  terminal-attached host copies it to the macOS/Linux clipboard; a host paste
  arrives as a **bracketed-paste** `ESC[200~…ESC[201~` block the line editor
  captures — works the same on QEMU and VBox over the serial console, no
  guest-additions driver; `/clip` shows/sets it), **key
  auto-repeat** (`keyrepeat`: software typematic in `xhci` — USB HID boot
  keyboards report only press edges — plus an accelerating streak amplifier:
  held Backspace/arrows erase/scroll 2/4/8 steps per repeat in the shell and
  editor), **syntax highlighting** (`highlight`: JSON/Markdown/Rust/Python/C/
  JS/TOML/sh lexers colour the editor per-cell, `/cat` output, and the chat's
  streamed markdown — fence-tagged code blocks are lexed per language without
  breaking token streaming), ANSI-coloured agent output, a `/`-command shell,
  and an on-disk UI config (`/configs/core/ui.json`, `shortcuts.json`).
  **The status bar sits on any edge** — `/statusbar top|bottom|left|right`, the
  `status_pos` key in `ui.json`, and the settings agent's `statusbar` tool; applies
  instantly and persists. The geometry is one pure function,
  `panes_layout::status_split`, which carves the bar off its edge and returns the
  **content rect**; `Screen` stores that rect and *every* pane calculation works
  inside it (`build`, `paint_gutters`, `band_capacity`, and the `band_divider_pct`
  drag inverse) rather than `0..width`/`0..height`. Keep it that way — a layout site
  that reaches for the full desktop is correct only for `Bottom`, where the content
  origin is `(0, 0)` and the split is the identity, so the bug hides until someone
  moves the bar. `left`/`right` make it a **column**: text cannot run across 16
  cells, so `status_segments` splits each template on the runs of 2+ spaces the
  author already used to group fields, `wrap_segment` wraps a long one onto extra
  rows (ellipsizing `${datetime} ${tz}` cut off the *time* — the part anyone reads),
  and the two stacks grow towards each other from the brand mark and the far edge,
  stopping when they would meet. The column width is a **fixed**
  `STATUS_V_COLS`, never fitted to the longest segment: the fields hold live values,
  so a fitted width would relayout every pane — reflowing scrollback — each time the
  clock ticked a digit.
  **System icons are Font Awesome 7 Free Solid** — not emoji, not hand-drawn
  rects for chrome. The face is `assets/fonts/FontAwesome7Free-Solid-900.otf`
  (SIL OFL / icons CC BY 4.0; see [THIRDPARTY-LICENSES.md](THIRDPARTY-LICENSES.md)),
  registered **first** in the TTF fallback chain at boot
  (`font_ttf::register_bundled_fallbacks`, name `FA_FALLBACK_NAME` =
  `"Font Awesome 7 Free Solid"`) so Private-Use-Area scalars resolve before
  Noto/emoji/CJK scans. Codepoints and helpers live in
  [`kernel/src/icons.rs`](kernel/src/icons.rs) (`icons::fa::*`, `is_icon`,
  `close_mark`, `for_agent`, `for_command` / `for_command_category`,
  `chess_piece`, `cursor_glyph`, status helpers); package-UI wasm guests cannot
  import the kernel crate, so they mirror the same literals in
  [`tools/apps-wasm/src/fa.rs`](tools/apps-wasm/src/fa.rs) and must stay in
  lockstep when a codepoint is added. **Where FA is used today:** status-bar
  `${kbd}` / `${mouse}` / `${net}` (active input is a body-size middle-dot, never
  a full FA circle — that ballooned at icon scale); OS mouse cursors
  (`arrow-pointer` / `hand-pointer` / `i-cursor` / `hourglass`, rasterized to
  fill+outline sprites with theme custom sprites still winning); pane and modal
  **close** (`xmark`, painted with `theme.accent` so `/theme` recolours it);
  action **tab labels** (ktrace/editor/top/todos/audio/image/video/browser +
  package-UI agent names via `for_agent`); `/help` and `/agents` browsers
  (category headers + command/agent rows); todos status marks; chess pieces on
  the board (`chess-pawn`…`chess-king` via `synapse::ui` text ops); settings /
  files / notes / activity package-UI chrome. **Sizing rules that bit us:**
  (1) size FA by **line height**, not `min(cw, ch)` — mono cells are tall and
  narrow, and width-first sizing made agent-list and close marks look like
  dots; (2) `blit_glyph` / `draw_str` give FA a **square cell of body line
  height** and advance by that width; (3) fontdue's `px` is an em size, not a
  max bitmap edge — solids often rasterize larger, so `build_ui_glyph` measures
  and **shrink-to-fits** with a little air so AA edges aren't clipped (status
  bar icons used to lose their tops). Only Free Solid is vendored (no Brands
  pack). The brand — logo, the terracotta `#cc785c` / warm-ink /
  cream palette (fully re-themable from `ui.json`), and typography — is specified
  in [DESIGN.md](DESIGN.md); honour it for any UI change. **Themes**
  (`theme.rs`, `/theme`) are presets layered over `ui.json` (still the single
  source of truth for the live look): bundled JSON in `assets/themes/*.json`
  (`dark`/`light`/`solarized-dark`/`nord`/`dracula`/`ubuntu`), installable to
  `/configs/themes/`, each carrying the chrome palette, `highlight` **syntax**
  colours, **cursor** fill/outline + optional sprite bitmaps (else FA default
  cursors), `font`+scale, a
  **wallpaper** (`""` ∣ `gradient:#a,#b` ∣ a store-image path, cover-scaled by
  `image::cover`, `/theme wallpaper` fetches a URL) and **opacity** (0–255).
  **NEW UI SURFACES MUST RESPECT THE THEME BACKGROUND:** with a translucent
  wallpaper (opacity < 255) the desktop must show behind *every* surface, so
  paint pane/region backgrounds through `Screen::paint_surface` and text-cell
  backgrounds through `fill_cell_bg` — **never a raw `fill_rect` of the bg
  colour** — and let glyphs blend via `blit_glyph`→`bg_at`. Both fast-path to
  `fill_rect` when there is no wallpaper / opacity == 255, so the default look
  is byte-identical (no regression). A theme switch recolours existing
  scrollback to the new `default_fg` (`Pane::recolor_default_fg`, called from
  `adopt`). The **only** exception is self-contained app content blitted as its
  own RGB buffer via `present_surface` — the browser, wasm-UI apps/games, the
  image/video viewers — which stays opaque. NB: the scheduler is
  cooperative, so **any long or blocking operation must pump the UI itself** —
  call `shell::upkeep()` (blink + status + mouse + `net::poll`) inside its
  loop, exactly as the per-token inference loops, the ONNX per-node loop, and
  the sliced FAT/ext4 readers do; loops that consume their own mouse events
  (modals, the editor) use `shell::status_tick()` instead. A tight compute
  loop that never yields freezes the clock, mouse, and net stack until it
  returns. Any new UI surface or blocking command must keep this upkeep
  running. The chat pane keeps a 2000-line scrollback (PgUp/PgDn; /clear
  wipes it); Shift+Tab / Ctrl+Tab / clicking switches pane focus.

- **AGX GPU** (`agx/`) — **the Apple-Silicon GPU coprocessor is booted to
  RUNNING on a real M2** (t8112, via `cargo xtask m1n1`; the foundation for GPU
  compute offload of `cortex`'s `matvec_qw`/`batched_proj` — the compute path
  itself is the next milestone, not done yet). `/agx up` drives the full
  bring-up: PMGR `gfx` power-on + SGX liveness poke → `cpu_start` → **GFXHandoff
  PPL handshake** (write `MAGIC_AP`, wait the firmware's `MAGIC_FW` — the shared-
  memory memory-manager handshake, done *before* the RTKit handshake per
  drm/asahi order) + UAT ctx-0 TTBRs under the handoff lock → RTKit **HELLO**
  v-negotiate → HELLO_ACK → EPMAP → START_EP → `AP_PWR_STATE=ON` → service the
  crashlog BUFFER_REQUEST (mapped into the **shared TTBR1 kernel range**, not
  per-context TTBR0 — the firmware's boot context only sees TTBR1) → both IOP+AP
  power reach ON = **RUNNING**. The **UAT** is real ARMv8 16 KiB paging with the
  **G14 geometry — bit 39 TTBR select (IAS=39), NOT bit 47** (`agx/uat.rs`,
  pure + unit-tested); getting that wrong makes the firmware's page-table walk
  miss our PTEs (the buffer is then unreachable and it stalls silently). The
  pure wire protocol (`agx/proto.rs`) + UAT encoder (`agx/uat.rs`) are
  **arch-neutral, unit-tested under `cargo xtask test`** (x86 — `arch::aarch64`
  is cfg-gated out of the test build, so pure logic must live outside it); the
  ASC-mailbox MMIO (`agx/asc.rs`, single-`ldr x`/`str x` FIFO + `dsb`/`dmb`),
  GFXHandoff (`agx/handoff.rs`, Dekker lock + cache-maintained shared mem), and
  discovery/PMGR/orchestration (`agx/hw.rs`) are aarch64-only. Gated on
  `is_apple()` + a `chitti.agx` **bootarg** (pass `chitti.usb chitti.agx`
  together; bare boot only, never under the m1n1 hv — same rationale as
  `chitti.usb`); a clean no-op on QEMU/VBox/other SoC. `/agx status` dumps
  bases/endpoints/power; every wait is bounded + pumps `upkeep()`/`poll_interrupt()`
  and answers Ctrl+C. Ported from m1n1 `src/{asc,rtkit,pmgr}.c` +
  `proxyclient/m1n1/{hw/uat.py,fw/agx/*}` and drm/asahi `gpu.rs`. **Two hard-won
  fixes were decisive:** UAT geometry = bit 39 (G14/t8112), and RTKit shared
  buffers belong in TTBR1 (shared across contexts), not TTBR0 (per-context, not
  active in the firmware's boot context). **Next:** app endpoints 0x20/0x21,
  `initdata` (perf/power tables + channel rings), the firmware command ring, then
  a GEMM microkernel into `cortex`.
- **Storage** — virtio/NVMe/AHCI block devices, GPT/MBR/FAT/ext4 detection,
  ext4 (the default filesystem) + FAT, `/install` (self-hosting install to a
  disk; detects an existing Chitti GPT and **updates in place**, preserving the
  data partition — destructive actions confirm via the permission modal),
  durable agent state on ext4. **Every disk is enumerated, across controllers and
  ports**: `ahci::probe_nth` indexes AHCI *disks* — each HBA on the bus × each
  port that is implemented *and* populated (`Ahci::present_count` counts without
  allocating, so skipping past a disk doesn't bring a port up and leak its DMA;
  `bringup_nth` then takes the one the index names). Ports are sparse on real
  hardware — a drive on port 3 with 0-2 empty is normal — and only taking the
  first controller's first present port made every other disk invisible. Exercise
  the real-hardware storage paths in QEMU with
  `CHITTI_DISK_IF=ahci|nvme|virtio-blk cargo xtask run -arch x86_64`.
  **Installing next to an existing OS.** Plain `/install` still writes a fresh GPT
  and erases the disk — that is the whole-disk path. Two non-destructive commands
  sit alongside it: **`/install plan`** (read-only; reports the partition table,
  free extents, and either a plan or why not) and **`/install alongside`** (x86;
  adds `\EFI\BOOT\BOOTX64.EFI` to the ESP already on the disk, renaming any
  existing loader to `BOOTX64.CHB` as a backup, touching no partition table and no
  partition). Planning is `gpt::{free_extents, plan_alongside}` — gaps computed
  with a high-water mark so an overlapping or contained entry can never make the
  inside of a live partition look free, and the existing ESP is **shared, never
  reformatted**, since a PC has one and rewriting it removes the Windows boot
  manager. The writer is `block::esp` over the pure `block::fat32` layer, and the
  backup is a **directory-entry rename** so the displaced loader keeps its original
  cluster chain — preserved byte-for-byte, no copy, no half-written window; a
  pre-existing backup is never overwritten because that one is the true original.
  Write order is allocate → payload → FAT (to **every copy**, or chkdsk calls the
  volume corrupt and Windows may "repair" it back to the stale copy, undoing the
  install) → directory entry last. Verified by 6 tests against a real FAT32 volume
  on a `RamDisk`; one of them caught the install overwriting Windows' loader
  because the fixture had left its cluster marked free, which is the reminder that
  **the FAT's allocation marks are the only thing protecting existing data**.
  Still absent: no ChittiOS data partition is created by `alongside`, and NVRAM is
  untouched, so firmware boot order may need changing by hand.
  NVMe enumerates namespaces via the **IDENTIFY
  CNS=2 active list**, never "walk NSIDs until empty" — NSIDs are sparse on
  VirtualBox (port→NSID; an empty port 0 = inactive NSID 1, exactly what a VM
  looks like after its install medium is detached); the `nvme: N active
  namespace(s)` ktrace is the first check for "controller up but no disks".
  Relatedly, the aarch64 identity map types **mixed RAM/MMIO GiB blocks** at
  2 MiB L2 granularity from the **real RAM extents the stub passes in
  boot-info** (`mm/ramlayout.rs`, pure + tested): VBox puts the model tail,
  GOP framebuffer and PCIe ECAM in one GiB block, and a whole-block Device
  retype alignment-faults NEON loads (scalar reads still work — "boots fine,
  dies in the matvec" is the signature).
- **Networking** (`net/`) — a full TCP/IP stack on [smoltcp](third_party/smoltcp)
  (vendored in-tree, 0BSD — see [THIRDPARTY-LICENSES.md](THIRDPARTY-LICENSES.md)):
  DHCPv4, static IP, DNS, ICMP (`/ping`), TCP/UDP, plus **loopback**
  (`127.0.0.0/8` + the name `localhost`) so an in-OS client can reach an in-OS
  listener. Loopback is a **second smoltcp interface** (Ethernet-medium
  `phy::Loopback`, `127.0.0.1/8`) with its **own** socket set, polled alongside
  the NIC interface — NOT the same set: sharing one set lets the loopback
  interface's egress steal/drop the NIC sockets' segments. A `TcpHandle` tags
  which set a socket lives in; a connect to a `127/8` address opens its socket in
  the loopback set via that interface's context. A vendored RFC-1122 guard in
  smoltcp `route()` keeps the NIC from ever emitting loopback traffic on the wire.
  NIC drivers behind one
  `NetDevice` facade — **virtio-net** over virtio-mmio (aarch64 QEMU) and over
  PCI, plus the PCI Ethernet families — discovered the same way on both arches.
  **A PCI NIC is claimed by vendor+device ID, never by vendor alone**
  (`net/nic_ids.rs`, pure + unit-tested): all Intel Ethernet reports
  `8086`/class `02:00:00`, but the families are register-incompatible —
  legacy **e1000** (82540-82547) and **e1000e** (82571…**I217/I218/I219**, the
  NIC in most business laptops) keep the rings at `RDBAL 0x2800`/`TDBAL 0x3800`
  and share `net/e1000.rs`, while **igb** (82575-I350) and **igc** (I225/I226)
  moved them to `0xC000`/`0xE000` with *advanced* descriptors and need
  `net/igb.rs`. Driving an I210 through the legacy path configures reserved
  space: it links and never receives a frame — and having claimed the one NIC
  slot, a second working card is never tried. So `net::pci::probe` walks **every**
  Ethernet function, skips the ones with no driver (logging each), and dispatches
  by table; unknown *Intel* IDs fall back to e1000e (the only open-ended family —
  Intel adds I219 IDs every PCH generation) with the ID ktraced as a guess.
  **Realtek** RTL8168/8111/8125 — the commonest consumer NIC — is `net/r8169.rs`:
  descriptor-owned rings (`DescOwn` in the descriptor, no tail register; kick via
  `TxPoll`), **unverified on hardware** (QEMU models no r8169-family part, only
  `rtl8139`, which is recognised but deliberately not implemented — no
  Windows-era machine has one). Test the dispatch against every family QEMU *can*
  emulate with `CHITTI_NIC=e1000|e1000e|igb|rtl8139|virtio-net-pci cargo xtask run
  -arch x86_64` (the `nic_dispatch` e2e scenario asserts the chosen driver matches
  the emulated device). **USB Ethernet** exists now
  (`net/usb_eth.rs` over the xHCI bulk transport): **CDC-ECM only**, chosen because it
  is the one real standard of the three shapes and puts one frame per transfer with no
  framing header, so the transferred length *is* the frame length. ASIX and Realtek
  dongles are recognised and then **refused with a log** — they need per-chip register
  setup and packet headers, and treating their framed packets as raw would hand smoltcp
  garbage. CDC-ECM is matched by interface class, not an id list (it is a standard, so a
  list would guarantee gaps). Tried **last** in `autodetect`, after virtio and PCI, so a
  built-in NIC always wins. Bulk transport is in `xhci.rs`: `configure_bulk` +
  `bulk_arm_in`/`bulk_take_in`/`bulk_send`; note the delivered length is
  **`requested - residual`** (a transfer event's low 24 bits are the *untransferred*
  count). Still missing for real machines: Broadcom `tg3`, Atheros/Killer `alx`,
  Aquantia.
- **WPA2-PSK and 802.11 frames** (`drivers/wifi/wpa.rs`, `drivers/wifi/ieee80211.rs`,
  `net/sha1.rs`) — the supplicant a SoftMAC radio needs, **built entirely as pure,
  vector-pinned logic**, because joining a Wi-Fi network is code where a bug is
  invisible: every step produces bytes as random-looking as the correct ones and the
  only feedback is that the access point stops answering, which is indistinguishable
  from a wrong password. So nothing here waits on a radio to be checkable. SHA-1 is
  deliberately absent from the TLS path (broken for signatures) and exists **only**
  because IEEE 802.11i mandates it — pinned to FIPS 180-2, RFC 2202 and the 802.11i
  Annex H PSK vectors; then PBKDF2→PMK, PRF-384→PTK (KCK/KEK/TK), the EAPOL-Key MIC,
  and AES-128 + RFC 3394 key unwrap for the group key. Four things are silent when
  wrong and each has a test: the MAC addresses and nonces concatenate **smaller
  first** (both sides do it, neither transmits the result, so a mistake is a
  self-consistent PTK the AP disagrees with — reported to the user as a wrong
  password); the PRF puts a **NUL** between label and data; the MIC is the first 16
  bytes of HMAC-SHA-1 over the frame with **its own MIC field zeroed**; and PBKDF2
  counts **bits** and must hash an over-long HMAC key first (every short vector passes
  without that branch). `Handshake` is the four-way exchange as a pure state machine
  over frames — `on_frame` in, reply out — so the failure paths are the tested ones:
  a replayed message 1 (which carries no MIC, so the replay counter is the only
  defence), a message 3 whose group key fails its integrity check, and the **ordering
  that matters**: the ANonce is checked *before* the MIC, because a changed ANonce
  fails the MIC too and checking the MIC first blames the passphrase. `ieee80211`
  parses beacons/RSN elements — attacker-controlled bytes from an unauthenticated
  sender, so every length is a claim and a lying one is refused, never clamped — and
  reports TKIP/SAE/802.1X/required-MFP as **unsupported up front** rather than as a
  timeout. Reachable and checkable by a user: **`/wifi psk <ssid> [passphrase]`**
  prints the derived key, and `wpa_passphrase` on any Linux box is an independent
  oracle for it (e2e `wifi_psk` asserts the published vectors on the running kernel).
- **Intel WiFi** (`drivers/wifi/iwl/`) — the part in most x86 laptops. Staged, and the
  stages are the point: `fw` (family from the PCI id, firmware filename search order,
  `.ucode` TLV parse), `csr` (registers + pure predicates), `context` (the gen2
  **context info** — the device's own loader fetches firmware out of host memory once
  it has that structure's address), `proto` (command out / notification in), `device`
  (the sequences and the queue). `/wifi up` resets the radio, hands over firmware,
  waits for the **alive** notification *and checks its status word* (firmware that comes
  up unusable still announces itself), reads the MAC out of the strap/OTP registers, and
  sends one **read-only** command (`NVM_GET_INFO`) — read-only on purpose, since a first
  command that configured something would misconfigure a real radio if any part of the
  untested transport is wrong. Traps pinned by tests: the MAC's first word is
  **big-endian** and the second contributes only its low two bytes reversed
  (`to_le_bytes`-and-concatenate gives a plausible wrong address); `prph` addresses are
  **20 bits** (the hardware supplies the `0xA00000` base, so truncation is correct);
  the transmit doorbell packs queue id and write index in one word; a command's first
  **20 bytes** must come from a separate aligned staging buffer; and a receive buffer
  must be **handed back** (the free-list write index is a *count*, not a slot) or the
  driver works during bring-up and goes deaf under traffic. **It cannot scan or
  associate**, and that is deliberate: a scan request, MAC context and station key are
  large per-API-version structures, no emulator provides an Intel WiFi part, and code
  written from memory would look complete, send well-formed garbage to a real radio and
  report success — the failure would be somebody's laptop rather than a missing feature.
  Those need a machine with the part in it. **Every layout here comes from Linux's
  `fw/api/*.h`, fetched, never recalled** — the one written from memory (`NVM_GET_INFO`'s
  general section as four `u32`s instead of `u32,u16,u8,u8`) put `n_hw_addrs` on the
  transmit chain mask: a small, plausible number that passes any sanity check, i.e. a
  confident wrong answer rather than an error. The groundwork for adding a command
  safely is `fw::cmd_version` (`IWL_UCODE_TLV_CMD_VERSIONS`, TLV 48), the table where the
  image states which request version it expects; a new command must consult it and
  **refuse** an unimplemented version, since silence in that table is *not* version 0.
  Also still absent: Realtek RTL8852 and Qualcomm/Killer WiFi, and Broadcom's SoftMAC
  parts.
  Shell surface: `/network` (info/dhcp/static/dns), `/ping`,
  `/wifi` (scan/connect via the password modal), a **TCP listener**
  (`net::listen`/`try_accept`, backed by a pool of Listen-state sockets in
  *both* the NIC and loopback sets, so one listener serves external/hostfwd and
  `localhost` clients alike; accept hands out an Established `TcpHandle` a service
  agent adopts as a channel), `/http` (a curl-like
  HTTP/1.1 client in `net/http.rs` — `-X`/`-H`/`-d`/`-v`/`--stream`, all
  methods, live chunked/SSE streaming; `http://` **and** `https://` via
  `net/tls.rs`/embedded-tls with **real certificate verification** (see the
  TLS-trust bullet); also the agent's `http` tool; **`-O`/`-o <file>`
  download the body to the Synapse store** (`/downloads/<name>`, overwrite
  confirms via the modal, human-typed only) where `/open` reads it back —
  editor, image viewer, or audio player), `/ws` (a
  plaintext WebSocket client in `net/ws.rs` — RFC 6455 handshake with
  Sec-WebSocket-Accept verification, masked frames, ping/pong), **`/mcp`** (an
  **MCP client** in `mcp.rs` — Model Context Protocol over HTTP/JSON-RPC 2.0,
  Streamable-HTTP transport with SSE + `Mcp-Session-Id`: `/mcp connect <name>
  <url>` runs initialize→tools/list and registers each remote tool into the
  tool registry as `mcp__<name>__<tool>`, so the shell agent calls it like any
  builtin — `tools/call` forwarded on invoke, results taint-tracked as
  UntrustedIngested; agents declare MCP servers in their manifest
  (`mcp_servers`), shown + connected on the install consent screen). `/model remote <http://host:port> [name]` points the
  shell agent at a **hosted** OpenAI-compatible model (llama.cpp server /
  Ollama / vLLM), over http or https, via `shell/remote.rs` — same system prompt, tool calls, and
  approval gates; only generation moves off-box (config persisted at
  `/configs/core/model.json`; switching backends is human-only, never an agent
  tool). The stack is polled cooperatively from the shell idle loop. NB: aarch64 MMIO register access must be a single
  `ldr`/`str` (inline asm) — LLVM otherwise coalesces adjacent volatile accesses
  into a paired load HVF can't decode (`hvf: isv`).
- **TLS certificate trust** (`net/x509.rs`, `net/rsa.rs`, `net/hashes.rs`,
  `net/ca_roots.rs`) — HTTPS server certs are **verified by default** against an
  embedded **Mozilla root store** (121 roots, `tools/gen_ca_roots.py` →
  `ca_roots.der` + spans; regenerate from `cacert.pem`). `ring` **cannot build
  bare-metal** (C + asm; that's why `NoVerify` shipped originally), so the
  validator is pure RustCrypto: `x509-cert` parses DER, `p256`/`p384` verify
  ECDSA, and [`net::rsa`] does **RSA PKCS#1 v1.5 + PSS** on `crypto-bigint`
  (a fixed `U4096` modexp — no `rsa`/`num-bigint-dig`/`ring`). `x509::verify`
  builds a chain leaf→intermediates→trusted root (each link's signature checked
  with the issuer key), checks each cert's validity window against the wall
  clock (refuses if the clock is unset — set `/datetime`), CA basic-constraints
  on issuers, and the leaf's SANs vs. the hostname (wildcards). The TLS 1.3
  `CertificateVerify` (RSA-**PSS** mandatory there — the bug that first failed
  css.tobyase.de) runs through the same code via `x509::verify_data`. A
  `ChittiVerifier` (`net/tls.rs`) implements embedded-tls's `TlsVerifier`
  (vendored crate given the few needed `pub` re-exports); `/tls insecure on` is
  the `curl -k` escape hatch (human-only) for a self-signed/self-hosted box.
  **Out of scope (documented, not silently skipped):** CRL/OCSP revocation.
  Validated by KATs (`rsa_testvec.rs`, `gen_rsa_testvec.sh`) + a real embedded
  chain (`ca_testvec.rs`, `gen_ca_testvec.sh`: verifies, and tampered/expired/
  wrong-host fail closed) + live `/http https://…` to real providers.
- **Sound & voice** (`sound/`, `onnx/`) — virtio-snd PCM in/out (S16 mono,
  poll-driven, descriptor chains) over virtio-mmio (aarch64) and virtio-PCI
  (x86 QEMU), **Intel HDA** for VirtualBox (x86+ARM) and real hardware, plus **AC'97** and **Sound Blaster 16** (x86 legacy; via `mm::alloc_dma_bounded`, which asks the frame allocator for the 8237's real constraints — under 16 MiB and inside one 128 KiB block — instead of allocating normally and hoping, which never held and made the driver unreachable code); `/voice` (waveform modal, level-gated utterances) and `/voice test`
  (tone + mic check). **`audio/`** is the pure media-decoder layer behind the
  `/open <file>.wav|.mp3|.aac` **player**: a full RIFF/WAVE parser (PCM
  8/16/24/32-bit + float32, any channel count downmixed), an MPEG Layer III
  decoder — a no_std Rust **port of minimp3** (CC0; tables generated verbatim
  by `tools/gen_mp3_tables.py`), validated ±1 LSB against minimp3's own scalar
  decode (stereo MS/short-blocks, MPEG-2 LSF, bit reservoir) — and an **AAC
  decoder** (`audio/aac/`, a Symphonia-path port under MPL-2.0: LC + Main/LTP
  ICS, ADTS demux, HE-AAC SBR/PS) that also supplies the video player's audio
  track. Playback feeds
  the device in ~50 ms chunks — queue backpressure paces it — pumping
  `upkeep()` and answering Ctrl+C between chunks. `onnx/` is a zero-copy no_std ONNX (protobuf) reader +
  **op interpreter** that runs the real voice models end-to-end: silero-vad v5
  (VAD), parakeet-ctc int8 (STT — `/voice stt <wav>` transcribes), and
  KittenTTS (TTS — `/voice say <text>` speaks); bare `/voice` is the full
  mic → VAD → STT → LLM → TTS conversation loop. Models load lazily from any
  disk volume (bundled in the images; `cargo xtask voice-assets` downloads
  them into `assets/voice/`, gitignored). **The ONNX ops parallelize across the
  SMP fleet** (`onnx::exec::par_range` → `smp::parallel_for`: conv1d tiles,
  conv_transpose gather+dot, matmul rows/cols, strided broadcast, `par_map`
  unary) and `/voice say` is **chunked** — `split_speech` synthesizes clause by
  clause into the `speech_pump` queue (fed from `ui_tick` via the non-blocking
  `SndDevice::out_free_bytes`), so audio starts in ~3 s and streams while the
  next clause synthesizes. For any numeric or perf work on this
  path, use `tools/onnxdiff/` (host-side layer-by-layer diff of the kernel's
  own interpreter against onnxruntime) — not QEMU round trips. NB: kitten's
  DynamicQuantizeLinear means outputs are only comparable by *equidistance
  from onnxruntime*, never bit-exact; any float reassociation flips int8
  rounding.
- **Remote voice** (`shell/voice_remote.rs`) — hosted TTS/STT providers as an
  alternative to the local ONNX models, same posture as `/model remote`:
  human-configured key (`/voice remote tts|stt <provider> <key> [voice]
  [model]`), persisted at `/configs/core/voice.json`, never an agent tool.
  Providers: **ElevenLabs, Cartesia, Inworld, Sarvam**, and any
  **OpenAI-compatible** `/v1/audio/{speech,transcriptions}` (base via
  `url@model`). Each is a pure request-builder + response-decoder (unit-tested
  wire shapes); TTS audio comes back as WAV/MP3 bytes (`audio::decode`) or
  base64-in-JSON (Inworld/Sarvam), resampled to the device rate and fed into
  the **same chunked `speech_pump`** — so remote synthesis streams per clause
  too. STT uploads the utterance as `multipart/form-data` WAV. `voice_say` /
  `voice_stt_file` prefer a configured remote endpoint, else fall back to the
  local model. **TLS caveat:** all providers are HTTPS and the in-kernel TLS
  client (`net/tls.rs`, embedded-tls TLS 1.3, no cert verification) doesn't
  interop with every server yet (RSA cert chains fail) — a provider that won't
  handshake reports a TLS error, not a wrong result.
- **Video** (`video/`) — H.264/AVC **baseline + Main/High-profile decoder +
  player** for `/open .mp4|.mov|.mkv|.webm` (hls/ts pending), built **in
  stages, each pure + unit-tested off-hardware** and **validated bit-exact
  against ffmpeg/PyAV via the `tools/h264diff/` host harness** (mounts
  `video/*.rs` via `#[path]`; the onnxdiff/cortexdiff pattern — the CAVLC VLC,
  deblock alpha/beta/tc0, and all CABAC tables are parsed/generated from the
  FFmpeg sources, `tools/gen_cabac_tables.py`, never hand-transcribed).
  **CABAC / High profile** (`h264/cabac.rs` engine + generated
  `h264/cabac_tables.rs` + `h264/decoder_cabac.rs`): I/P/**B** slices, adaptive
  **8x8 transform** + Intra_8x8, a POC-ordered multi-frame **DPB** with
  ref-list construction/reordering + sliding window/MMCO-1, **explicit
  weighted P**, **implicit weighted bi-prediction**, **spatial + temporal
  direct** — validated **bit-exact on full real-world clips**: 171/171 frames
  (sample-5s-720p: High, pyramid B, weightp) and 300/300 (Big Buck Bunny 720p:
  High, temporal direct, 16-ref). Hard-won availability rules: MV prediction
  may only use 4x4 cells whose motion **for that list** is final (per-list
  `mvok` stamps — an above-right cell of a later same-MB partition is
  *unavailable*, while a partition not using the list becomes
  available-with-ref -1 immediately, FFmpeg's LIST_NOT_USED fill); the
  **ref-idx context** reads refs-as-parsed (pre-MV); the B mb_type context
  tests the *MB-level* direct flag, not the per-4x4 one. Unsupported CABAC
  features (interlaced/MBAFF, FMO, scaling matrices, I_PCM-in-CABAC, long-term
  refs, poc type 1) are *refused* cleanly, never mis-decoded.
  **Full baseline pipeline:**
  `mp4`/`mkv` demux → CAVLC residual (`h264/cavlc.rs`) → **I + P** macroblock
  decode (`h264/decoder.rs`: I_4x4/I_16x16/I_PCM, P_L0_16x16/16x8/8x16/8x8/Skip,
  **multiple slices per frame** with slice-aware neighbour availability) → intra
  (`h264/intra.rs`) + **inter** (`h264/inter.rs`: median MV prediction + 6-tap
  luma / bilinear chroma MC) → inverse transform (`h264/transform.rs`) →
  **in-loop deblocking** (`h264/deblock.rs`) → YUV→RGB → a **video tab** with
  a **player HUD** (`framebuffer::draw_video_status`: state, mm:ss, frame
  counter, scrubber, mute, shortcut hints — drawn *after* each frame blit) and
  transport controls (Ctrl+Tab focus, space pause, ←/→ seek, ↑/↓ ±10 frames,
  0 restart, `m` mute, Ctrl+C stop), frame-paced by pts. The HUD sits in a
  **reserved bottom strip** (`present_surface_reserve` + `video_hud_height`) the
  per-frame blit never repaints, its text **wrapped to the pane width** and
  repainted in place — so it neither flickers nor overflows. **Streaming
  decode:** `video::StreamDecoder` holds the source + sample table and decodes
  on demand (`seek_decode`, rewinding to the latest keyframe on a backward
  seek) — a whole-clip `Vec<Frame>` would be ~700 MB of RGB for a 1300-frame
  480p clip (heap-hostile; trap #3). Baseline keeps **one** reference frame;
  CABAC keeps the DPB plus a bounded **reorder cache** (pictures pending their
  display slot, keyed by decode index — without it every backward hop of the
  B-pyramid display order re-decodes from the previous IDR, O(n²)). Display
  order comes from the container (`ctts`-adjusted `Sample.cts`, stable-sorted). **The "green frames"
  bug** was NOT memory: a P-slice that **ends with a trailing `mb_skip_run`**
  (skips the final MBs, no coded MB after → `more_rbsp_data()` goes false
  mid-drain) left the last MB(s) at plane-init 0 → black luma / green chroma,
  which inter-prediction then propagated into a growing green region. Fix: keep
  draining inferred skips while `skip > 0` even past `more_rbsp_data()`
  (`decoder.rs` MB loop). Lesson: **render + eyeball vs PyAV, don't trust a
  single green-fraction metric** (the first scan's threshold missed it).
  **Audio:** `mp4::parse_audio` demuxes the AAC (`mp4a`/`esds`
  → AudioSpecificConfig) track, and the **AAC decoder in `audio/aac/`**
  (a Symphonia-path port, MPL-2.0 — see THIRDPARTY-LICENSES.md — plus ADTS
  demux and HE-AAC SBR/PS reconstruction) turns it into mono S16 PCM at open
  (`open> audio ready: …`); playback keeps the PCM cursor pts-locked to the
  video clock (`pump_video` snaps drift > ~50 ms), and `m` mutes. **Validated
  bit-exact against PyAV/ffmpeg** — synthetic x264 clips (I/P, multi-slice,
  deblocked) and hundreds of consecutive frames of real-world mp4/mkv. In-kernel
  fixture tests hash an embedded I-only and an I+P clip against PyAV, and
  `stream_decoder_seek_matches_sequential` proves random/backward seeks match a
  sequential decode frame-for-frame.
  **Deblock gotchas (all three bit us):** a chroma edge's QP is `avg(qpc(QPp),
  qpc(QPq))` not `qpc(avg(QPp,QPq))` (differs only across slices with differing
  QP); the luma normal filter's `tc = tc0 + (ap<β) + (aq<β)` can be nonzero
  even when `tc0==0` (don't force-skip); and the recycled `Fx` workspace must
  not leak the previous frame's motion into deblock — `bs_inter2` infers "list
  used" from `refpoc != MIN`, and a P slice never writes L1, so
  `mark_mb_decoded` clears `refpoc` (not just refidx/mv) for unwritten cells.
  Bisect any deblock-vs-PyAV divergence with `H264Dec::no_deblock` against
  PyAV `skip_loop_filter='ALL'`, and read FFmpeg's per-block list usage via
  `flags2=+export_mvs`. **Stage 1 (done):** `video/bits.rs`
  (RBSP emulation-prevention unescape + a big-endian `BitReader` with H.264
  Exp-Golomb `ue`/`se`/`te`), `video/mp4.rs` (ISO-BMFF box-tree demuxer →
  `avcC` SPS/PPS + the `stsz`/`stsc`/`stco`/`stts`/`stss` sample table assembled
  by the pure `build_samples`), and `video/h264.rs` (Annex-B **and** AVCC NAL
  splitting + SPS/PPS parse → geometry/profile/entropy mode). `video::probe`
  reports a stream (container, codec, `W×H`, frame count, CAVLC/CABAC) without
  decoding pixels; `/open clip.mp4` shows it. Scope: **H.264 baseline** (I/P
  slices, CAVLC, 4:2:0) **plus Main/High CABAC** (see above).
  **Playback performance (all profiled first — `sample` on the host harness,
  `video: perf:` ktrace in-kernel):** in-kernel 1080p went 12 → **~30 fps**,
  4K ~8–10 fps, via three stacked levers: (1) **NEON luma MC** in
  `third_party/rust_h264/inter_pred.rs` — the 6-tap interpolator was 40% of
  decode; all vector loads/stores are inline-asm `ldr/str d/q` (`ld8/st8/
  ld16i/st16i` — the `+strict-align` rule; the asm loads beat the `vld1`
  intrinsics even host-side), and every hv quarter-pel case runs h-FIR once
  per row into an i16 block buffer instead of six FIRs per pixel (bit-exact:
  full-clip A/B byte-identity + PyAV); (2) **decode-ahead** — the pump loans
  the `StreamDecoder` to an SMP worker (`smp::async_submit`, the reserved
  last slot; dispatchers exclude it via `fleet_workers` + zero its ranges) and
  holds the finished frame until its pts (`VideoPlayer::ahead`); every other
  `dec` toucher joins first (`video_job_join`) — beware pipeline bubbles: the
  worker must be resubmitted on the same tick it's collected; (3)
  **frame-drop** (`sample_is_nonref`, pure + unit-tested): behind the clock,
  non-reference backlog samples are never fed to the decoder at all — but
  catch up in ≤2-frame steps; one giant hurry-jump decodes every backlog
  reference in one job and starves presentation (4K: 8 → 3 fps).
  **Remaining:** HLS/TS demux, 4K ≥ 15 fps (needs parallel
  reconstruction — slice-parallel + independent non-ref B's; CABAC parse is
  the serial floor), and the multi-pane split + tab drag-drop. NB: the e2e stdlib muxer writes no
  `ctts`, so its B-frame clips carry no display-reorder info — e2e CABAC clips
  use `--bframes 0`; the in-kernel fixture decodes an I/P/B clip in decode
  order and a media-key rule: a focused-but-stopped video tab must not eat
  keystrokes (`media_key` gates on `video_loaded()`).
  **Host reference for the numeric stages:** PyAV
  (self-contained ffmpeg) decodes the same clip to YUV for a frame-by-frame diff
  harness (`tools/h264diff/`, the onnxdiff/cortexdiff pattern — mounts
  `video/*.rs` via `#[path]`, runs on the host in seconds, no QEMU round-trips).
  The e2e `open_video` scenario muxes a real x264 baseline multi-slice clip into
  mp4 (stdlib muxer) and asserts the on-kernel probe + streaming decode ("N
  frame(s), ready in …") + transport controls; it auto-skips where x264 is absent.

  **H.265/HEVC and VP9 both play.** Main-profile 8-bit 4:2:0 HEVC and profile-0
  VP9 are bit-exact against FFmpeg/libx265 and libvpx respectively; `/open`
  streams them. Tiles, PCM, 10-bit and VP9 segmentation still refuse with a
  named reason (`VideoInfo::unsupported_reason`) rather than a wrong H.264-shaped
  message — the player used to print "CABAC entropy coding (baseline/CAVLC only)"
  for every undecodable stream, which is true of H.264 and nonsense about HEVC
  (always CABAC) and VP9 (no CABAC). What landed:

  - **The demuxers are codec-agnostic.** `mp4::CodecConfig` is `Avc(AvcC) |
    Hevc(HvcC) | Vp9(VpcC)`; mp4 reads `avc1/avc3`, `hvc1/hev1` and `vp09`, mkv
    reads `V_MPEG4/ISO/AVC`, `V_MPEGH/ISO/HEVC` and `V_VP9`. Three traps, each
    of which silently produces a plausible wrong track: **`hvcC`'s fixed part is
    22 bytes, not `avcC`'s 5** (it carries the whole `profile_tier_level`), and
    reading it the short way lands the NAL arrays inside the constraint flags,
    where the lengths are large believable numbers; **VP9 has no `CodecPrivate`
    and needs none** (every frame header is self-describing), so requiring one
    rejects every WebM file; and **VP9 samples carry no length prefix at all**,
    so `CodecConfig::length_size()` is 0 there and a caller that assumes framing
    reads the frame header's first four bytes as a NAL length.
  - Fixed while generalising, both pre-existing and both invisible on a
    video-only file: the mkv demuxer took blocks from **every** track (an mkv
    with audio interleaved audio frames into the video sample list — `TrackType`
    is now honoured and blocks are filtered by track number), and it read only
    `SimpleBlock`, so a file using `BlockGroup`/`Block` demuxed to **zero**
    frames. A `BlockGroup` has no keyframe bit; the absence of a
    `ReferenceBlock` is what marks one.
  - **HEVC** (`video/hevc.rs`) — NAL split (Annex-B and HVCC), VPS/SPS/PPS,
    `profile_tier_level`, scaling lists, short-term RPS, and the slice segment
    header. Reuses `video::bits` verbatim (HEVC's RBSP escaping and Exp-Golomb
    are AVC's) and nothing above it. Four things that are each a silent
    mis-decode: **the NAL header is two bytes** (`nal_unit_type` is 6 bits at bit
    1), so AVC's `hdr & 0x1f` types every unit wrong; **`SliceType` is numbered
    backwards from AVC's** (0 is B here, I there); **the profile block is 88
    bits** and, when `maxNumSubLayersMinus1 > 0`, the *unused* sub-layer slots up
    to 8 are padded 2 bits each — omit that and every SPS field after it decodes
    to plausible nonsense; and **the default scaling lists are not flat** (H.265
    Table 7-6 is a real matrix above 4x4), so a flat default decodes every
    scaling-list stream slightly soft rather than erroring. The slice header
    *consumes* `pred_weight_table` and reference-list modification even though
    the decoder does not use them yet, because their length is data-dependent and
    a `data_byte_offset` that is right only for simple streams decodes garbage
    instead of erroring.
  - **VP9** (`video/vp9.rs`) — superframe index, the boolean decoder, and the
    uncompressed frame header. **A container sample is not a frame**: almost
    every libvpx stream packs an invisible ALTREF with the frame that shows it,
    and treating the sample as one frame decodes the ALTREF and displays it. The
    bool decoder is libvpx's windowed form; a test proves it is arithmetically
    identical to the spec's bit-serial `split = 1 + (((range - 1) * p) >> 8)` for
    every reachable `(range, prob)`, since the serial renormalise is too slow for
    the coefficient layer. Its first bit is a marker that must be 0, which is the
    only check that a compressed-header *size* was read correctly — get it wrong
    and a whole frame decodes as noise with no error. Also: the interpolation
    filter's 2-bit literal is **not** the enum order (literal 0 is
    `EIGHTTAP_SMOOTH`), and `s(n)` is magnitude-then-sign, not two's complement.
  - **`tools/videodiff/`** is the host harness for all three codecs (the
    `h264diff`/onnxdiff pattern — mounts `kernel/src/video/*` via `#[path]`, so
    there is one implementation): `probe`, `headers` (per-frame/per-slice header
    dump — the bring-up view, since "the demuxer found no frames" and "the
    headers do not parse" are different failures with the same symptom), and
    `yuv`. `diff.py` cross-checks codec/geometry/frame-count against **PyAV**,
    which is the independent implementation our own tests structurally cannot be.
    Verified against real x265 and libvpx output in both container families,
    including tiled, multi-slice, 10-bit and 1024x576 clips: the HEVC B-pyramid
    comes out as POCs 0,4,2,1,3 with a CRA at the second keyframe, 4-slice
    frames at CTB addresses 0/20/40/60, and 2 WPP entry points for 3 CTB rows.
    NB `h264diff` also had to be repaired — it had stopped compiling against
    `video/mt.rs`'s arch-neutral `parallel_for` facade.
  - Tests: the parameter-set fixtures are **real x265/libvpx bytes**, not
    hand-built bit patterns (a fixture built by the parser's author parses fine
    by construction), plus whole embedded `hvc1` mp4 and `V_VP9` WebM files, plus
    an `open_hevc_vp9` e2e scenario that embeds those same two files base64 so it
    always runs instead of skipping when x265/vpxenc are absent.

  **VP9 decodes, and it is bit-exact against libvpx.** `/open` plays a VP9 file
  — profile 0 (8-bit 4:2:0), intra and inter, sub-8x8 partitions, all four
  transform types at all four sizes, multiple tile columns, and the in-loop
  deblocking filter. Verified frame-for-frame against PyAV/libvpx on nine
  clips including a 1280x720 real-world file, a 1024x576 four-tile clip,
  deeply-searched encodes that use every partition shape, and a
  non-frame-parallel stream that needs backward adaptation: **zero differing
  samples in any plane**. `tools/videodiff vp9seq` + `cmpseq` is that harness.

  The build is `kernel/src/video/vp9/`: `tables.rs` (generated),
  `idct_kernels.rs` (transpiled), `transform.rs`, `intra.rs`, `header.rs`
  (compressed header), `tokens.rs`, `tile.rs`, `inter.rs`, `loopfilter.rs`,
  `decoder.rs`. **Two of those files are machine-produced and must not be
  hand-edited**: `tools/gen_vp9_tables.py` parses ~9000 probability, scan,
  neighbour, filter and dequantiser values out of libvpx, and
  `tools/gen_vp9_idct.py` transpiles the seven 1-D transform kernels (`idct32`
  alone is 328 statements). Regenerate, do not retype.

  Eleven bugs were found bringing this up, and every one of them was silent —
  worth reading before touching any of it:

  - **The table generator dropped the minus sign** on `-PARTITION_NONE`-style
    tree leaves, because `-?\d+|[A-Za-z_]\w*` cannot match a negated
    identifier. Every leaf became a node index, so `read_tree` walked
    `1 -> 2 -> 1` forever: a **hang**, not a crash. Guarded now by a
    tree-termination test.
  - **The intra frame's partition probabilities are the constant
    `vp9_kf_partition_probs`, not the frame context** (libvpx
    `set_partition_probs`). This is the only probability set on the keyframe
    path that is not adaptive, and using the adaptive one desynchronises the
    whole tile while the *mode grid still looks plausible* — because keyframe Y
    and UV modes come from their own constant tables. A sensible-looking
    partition/mode dump over a completely wrong picture is the signature.
  - **`ADST_DCT` means ADST on the *columns*.** libvpx's table is
    `typedef struct { transform_1d cols, rows; }` and its `ADST_DCT` entry is
    `{ iadst, idct }` — the **first** member is the column kernel. Swapping them
    is invisible on `DCT_DCT`, on `ADST_ADST` and on any DC-only block, so it
    survives every cheap test; it showed up as chroma (all-DC) bit-exact while
    luma was off by ~25 everywhere.
  - **The residual covers the mode-info block, not the prediction block.**
    `n4_w = (bw << 1) >> ssx` with `bw` in 8x8 MI units, so a `BLOCK_4X4` still
    carries a full 8x8 of luma — **four** 4x4 transforms. Sizing it from
    `num_4x4_blocks_wide` reads one, leaving three quarters of the coefficients
    unread. Only content the encoder searches deeply enough to *use* sub-8x8
    partitions trips it, so a fast encode of a clip was bit-exact while a slow
    encode of the same clip overran its tile.
  - **The 4x4 directional intra predictors are hand-written special cases**
    (`intra_pred_no_4x4` generates the generic form only for 8/16/32). `d45` at
    4x4 continues the diagonal through `above[7]` where the generic one clamps
    to `above[bs-1]`. Running the generic code at 4x4 gives a plausible diagonal
    that is wrong in its lower-right triangle.
  - **Intra edge availability is per *block*, and the left edge is
    tile-relative.** `left_mi` is null at a tile's left column, which feeds the
    skip context, the transform-size context and the Y-mode probabilities — so
    it changes *bit consumption*, not just prediction. Invisible on single-tile
    content. Above-right is also only ever read for 4x4 transforms, and past a
    frame edge the last real sample repeats rather than the buffer's padding
    being read (which would decode differently on a seek than on linear play).
  - **The output shift is per transform size** (4/5/6/6), and the 32x32 row pass
    has *no* pre-shift. One constant makes large blocks come out at the wrong
    contrast.
  - **The bool decoder legitimately reads past the end.** It pre-loads ~56 bits,
    so every well-formed partition ends holding virtual zeros; libvpx adds
    `LOTS_OF_BITS` to `count` there and only errors when a decode runs *far*
    past. Flagging the first read past the last byte rejects every valid frame.
  - **`num_4x4_w`/`num_4x4_h` for sub-8x8 inter blocks come from the
    partition**, and having them the wrong way round makes a `BLOCK_4X4` read
    **one** motion vector instead of four.
  - **`get_sub_block_mv` uses a table** (`idx_n_column_to_subblock`,
    `{1,2},{1,3},{3,2},{3,3}`), not a derivable rule. A plausible hand-derived
    version differs from it on exactly the cases a deep encode produces — it was
    worth 33 wrong pixels in one 10x6 region, which then propagated.

  Also fixed while here, and both pre-existing: the **mkv demuxer took blocks
  from every track** (an mkv with audio interleaved audio frames into the video
  sample list) and read only `SimpleBlock`, so a `BlockGroup` file demuxed to
  **zero** frames.

  **Backward probability adaptation is built too**, which is what makes a stream
  encoded with frame-parallel decoding *off* decode at all: the probabilities
  the next frames start from are this frame's symbol counts merged into the
  context it was decoded against. Two things about it are not guessable.
  `mode_mv_merge_probs` **returns the probability unchanged when there are no
  observations** rather than falling back to 128 — which would throw away the
  forward update the header just transmitted. And motion vectors are counted
  from the **reconstructed difference**, not from the symbols as they were read:
  `vp9_inc_mv` recomputes the joint and the magnitude class from the value, and
  counts the high-precision bit *always*, including where the bitstream coded
  none (its value is then the implicit 1). Counting per-symbol leaves the hp
  tallies short and desynchronises a frame several removes later — it was worth
  exactly three good frames and then noise.

  Still refused rather than guessed at: segmentation, reference scaling
  (a mid-stream resolution change) and profiles 1-3.

  **HEVC's pixel pipeline is being built stage by stage, and the pure stages are
  in.** `/open` still refuses an HEVC file with a reason — nothing below is
  wired to a picture yet, and it will not be claimed as playing until whole
  frames match PyAV. What exists, each pure and unit-tested off-hardware:

  - **The CABAC engine is H.264's, shared rather than copied**
    (`Cabac::new_hevc`). The arithmetic, `rangeTabLPS` and the state transitions
    are byte-identical between the two standards; only the context
    initialisation differs. FFmpeg writes that derivation as
    `pre = 2p - 127; pre ^= pre >> 31; clamp`, which **reads like an absolute
    value and is not** — for negative `x`, `x ^ (x >> 31)` is `-x - 1`, and that
    off-by-one is exactly the specification's asymmetry between `63 - p` (valMPS
    0) and `p - 64` (valMPS 1). "Cleaning it up" into `abs()` is wrong on every
    context that starts with valMPS 0, each one state too confident, which no
    bitstream rejects.
  - **199 contexts and ~1200 constants are generated, never transcribed**
    (`tools/gen_hevc_tables.py` -> `hevc/cabac_tables.rs`, `hevc/tables.rs`).
    Two traps the generator had to learn: `HEVC_CONTEXTS` is the *allocated*
    size (199, sized for the range extensions) while only **179** are
    initialised, so the parse must accept `3 x (<= n)` and zero-pad as C does;
    and a **zero-bin element occupies no context slot** — the X-macro sets
    `NAME_END = NAME_OFFSET + NUM_BINS - 1`, so with 0 bins `END` falls *below*
    `OFFSET` and the next element starts at the same index. Advancing by one
    there shifts every later element's contexts, which decodes one syntax
    element against another's probabilities: not a failure, a picture that is
    wrong in a way that looks like a different bug. The offsets are checked by
    `pos == used` at generation time.
  - **The inverse DCT is a matrix multiply, deliberately** (`hevc/transform.rs`).
    Every decoder in the wild writes it as `TR_4` inside `TR_8` inside `TR_16`
    inside `TR_32` — four nested macros of hand-placed constants. That
    factorisation is an optimisation, not the definition: the specification
    defines `out[i] = sum_k M[k][i] * src[k]` over one 32x32 basis, and because
    both passes sum in full precision before their single rounding step, the
    direct form is **bit-exact** with the butterfly rather than merely close. So
    the whole transform path contains no hand-written constant, driven from the
    generated `TRANSFORM`; the 4x4 luma DST is the one exception and is
    cross-checked against FFmpeg's butterfly on every basis vector.
  - **Intra prediction** (`hevc/intra.rs`) — reference substitution, the weak
    3-tap and strong bilinear reference filters, and all 35 modes. Substitution
    propagates **in the specification's scan order** (up the left edge, around
    the corner, along the top), not from the nearest available sample: the
    obvious reading gives a different picture at every block touching a slice or
    picture boundary, and both look plausible. The corner is filtered from
    *both* edges at once — one wrong pixel there seeds every negative-angle
    projection through it.
  - **Deblocking sample filters** (`hevc/deblock.rs`) — HEVC filters on an
    **8x8 grid**, and each 8-sample edge is two independently decided halves
    whose strong/weak choice comes from **lines 0 and 3 only** and applies to
    all four. Deciding per line gives a smoother, plausible, wrong picture. `tc`
    is looked up at `qp + 2 * (bS - 1)`, so dropping the `bS` term still filters
    every edge that should be filtered, just uniformly — which reads as slightly
    soft intra edges and nothing else.
  - **SAO** (`hevc/sao.rs`) — band and edge offset. The classifier must read
    **unfiltered** neighbours, so it cannot run in place: doing so makes column
    `x`'s category depend on column `x-1`'s offset, a directional bias that
    looks like a motion-compensation bug. And `edge_idx` remaps the five
    relations to `{1,2,0,3,4}` so the flat case lands on the always-zero entry;
    indexing four offsets directly is off by one for two of the five categories.

  - **Residual coding** (`hevc/residual.rs`) — the CABAC syntax layer: the
    last-significant-coefficient position, 4x4 coefficient groups walked
    backwards, greater1/greater2 flags, Golomb-Rice remainders with
    `stat_coeff` adaptation, and sign data hiding. Nearly every rule in it is a
    *context selection* rule, which is the mistake that does not fail — a bin
    decoded against the wrong model still yields a 0 or a 1, the stream stays in
    sync for a while, and the picture is wrong in a way that looks like a
    different bug. Two pieces of bookkeeping to hold on to:
    `significant_coeff_flag_idx` is filled in **decreasing** scan order (so
    entry 0 is the *highest* position, and reading those names the other way
    round inverts sign hiding's distance test), and `greater1_ctx` is **carried
    across coefficient groups** rather than reset per group.
    The Rice code is validated by round-tripping against an encoder written
    separately from the specification's description, over every Rice parameter —
    and that needed a real **arithmetic** bypass encoder, because a bypass bin
    is *not* a raw bit. With `range` fixed at 510 the state obeys
    `offset_k = S_k - 510 * N_k` (`S_k` = the first `9 + k` bits as an integer,
    `N_k` = the bins so far), which inverts to `N_k = S_k / 510`; the first
    version of that helper assumed bits passed straight through and would have
    tested the bit reader instead of the code.
  - **Coding-unit derivations** (`hevc/ctu.rs`) — the MPM list, chroma mode
    resolution, scan-order selection, partition geometry and `bS`. The MPM ring
    is 32 wide over modes 2..=33, so mode 2's predecessor is **33** and mode 34
    folds onto the same neighbours — not what "extend the angle by one step"
    suggests. The non-MPM path **sorts** the candidates before skipping, which
    is easy to omit because the MPM path does not need it, and omitting it still
    yields a legal mode. The chroma escape on a collision is **mode 34**, not
    the next table entry. And `bS` matches references **by picture, not by list
    index** — the same picture can sit at different indices in the two lists, so
    index-matching calls two identical predictions different — with both
    pairings tried when a block predicts twice from one picture.
  - **Inter prediction** (`hevc/inter.rs`) — MV scaling, the merge candidate
    list, and motion compensation. MC works in a **14-bit intermediate**: every
    fractional filter leaves its result there and the final shift happens once,
    in the uni- or bi-prediction combine, which is why bi-prediction is more
    accurate than averaging two rounded predictions. Five shifts have to agree
    (`<< (14-B)`, `>> (B-8)`, `>> 6`, `>> (14-B)`, `>> (15-B)`) and a flat
    reference reconstructing exactly through every fractional position pins all
    of them at once. Merge pruning compares each position against a **fixed
    short list** of predecessors, not against everything found so far — B0
    duplicating A1 is still kept — and B2 is dropped once four candidates exist,
    a rule about the *count* rather than about the list being full.
    `mv_scale`'s `+ 127 + (negative)` is a round-half-away-from-zero; a plain
    shift biases every scaled vector towards negative infinity and drifts a
    whole GOP of B-frames.

  Two tables in this work were **recalled and wrong**, which is the standing
  reason everything here is generated: the 4:2:2 chroma mode map (`tab_mode_idx`)
  came out right for the first 14 entries and wrong for the remaining 21 — it is
  a gentle monotone curve either way, so nothing about it looks suspicious — and
  FFmpeg's context-init `pre ^= pre >> 31` reads as an absolute value but is
  `-x - 1`, which *is* the specification's `63 - p` / `p - 64` asymmetry.

  - **AMVP** (`hevc/inter.rs`) — two predictor candidates per list. Each
    position is tried in list `lx` **then the other list**, because a neighbour
    predicting the same *picture* through the opposite slot is still a valid
    predictor. And `isScaledFlag` gates a whole second pass: when neither left
    neighbour exists, B is promoted into A's place and B is re-derived with
    scaling allowed. Missing that rearrangement yields the right *number* of
    predictors and the wrong ones, at every prediction unit on the left edge of
    a CTB row. Long-term and short-term references are never mixed — a
    long-term reference has no meaningful temporal distance, so the candidate is
    refused rather than scaled by a nonsense ratio.
  - **POC, RPS, reference lists and the DPB** (`hevc/dpb.rs`) — nothing here
    touches a pixel, so every mistake is a whole frame that is right in
    isolation and wrong in sequence. Four rules carry the risk. **The POC wrap
    test is asymmetric** (`<` pairs with `>=`, `>` pairs with `>`), so a
    half-range difference wraps forwards and not backwards; writing both
    comparisons the same way misplaces exactly one picture per wrap. **L0 is
    before-then-after and L1 after-then-before** — that single swap is the whole
    structure of bidirectional prediction, and reversing it decodes fine with
    every B-frame's two predictions exchanged. **A reference list is filled
    cyclically**, so a slice activating more references than the RPS holds
    repeats the concatenation rather than getting a short list. And
    `used_by_curr_pic` decides Curr vs Foll while *position* decides Before vs
    After, so a picture kept only for a later frame is in the RPS and in no
    list. Bumping releases the **lowest pending POC**, which is what turns
    decode order back into display order for a B-pyramid.

  - **CU/PU syntax elements** (`hevc/syntax.rs`) — everything above the residual
    coder: the split, skip, part-mode, CBF, intra-mode, merge, reference-index,
    MVD and QP-delta codes. **The binarizations are pure functions over a `Bin`
    source**, not over the CABAC engine, and that is what makes them testable —
    a context-coded bin cannot be forced from outside the arithmetic coder, so a
    test that had to code its way to a given branch would be testing the coder.
    With a canned bin source, `part_mode` can be checked for what actually
    matters: that its code is **prefix-free and complete** in each of the five
    configurations. Four traps it pins: an 8x8 inter CU codes `Nx2N` in **two**
    bins, not three (NxN there would be 4x4 inter prediction, which HEVC
    forbids), and consuming the third steals a bin from the next element; an
    8x4 or 4x8 prediction unit skips its bi-prediction bin entirely for the same
    bandwidth reason; the MVD's four context-coded flags are **interleaved
    x, y, x, y** before either remainder, so decoding component-by-component
    reads a valid vector from the wrong bins whenever both components are
    non-zero; and `cbf_luma`'s context index runs the *opposite* way from every
    other depth-indexed context here (1 at the top of the tree, 0 below).

    `cu_qp_delta_abs` **cannot express more than 131** — its suffix's own unary
    prefix is capped at 7 — and that ceiling has to be reported rather than
    absorbed: FFmpeg returns an error there, and the obvious alternative of
    returning the prefix is the plausible small delta `5`, so a corrupt stream
    would quietly shift the quantiser for the rest of the CU instead of being
    rejected. It returns `Option<u32>`. (Found by a round-trip test that ran
    past the representable range, which is the only reason the cap was noticed
    at all.)

  - **The walk** (`hevc/decoder.rs`) — the CTU quadtree, reconstruction, and the
    picture-level in-loop filters. It owns what a *picture* has and a block does
    not: the neighbour grids (motion, intra modes, skip, depth, QP, coefficient
    presence, each at the granularity the specification indexes it by), and
    **availability**, which for a quadtree is not "above or left" but a **z-scan
    order comparison** — a block above you can be one that has not been decoded.

  **HEVC intra is bit-exact against FFmpeg.** `tools/videodiff hevcseq` decodes a
  real `hvc1` mp4 and the IDR frame matches PyAV with **zero differing samples in
  all three planes** — CABAC, the quadtree, MPM derivation, reference
  substitution and filtering, all 35 prediction modes, residual coding, dequant,
  both transforms, deblocking and SAO. Two bugs stood between "decodes" and
  "bit-exact", and both are the silent kind:

  - **A leftover context-coded `decision()` in the QP-delta path** consumed one
    bin belonging to the residual right after it. `cu_qp_delta_sign_flag` is a
    **bypass** bin, present only when the magnitude is non-zero. The symptom was
    a picture that decoded, terminated its slice at exactly the right CTU, and
    had plausible intra modes throughout — while every coefficient block came
    from one bit too late.
  - **`qPY_PRED` is the average of the left and above neighbours *inside the
    current CTB*** (H.265 §8.6.1), falling back to the previous CU's QP where
    either is absent — not the running QP. Using the running QP drifts toward the
    bottom-right of every picture, because the error compounds through each
    quantisation group's predictor and then again through intra prediction from
    the blocks it mis-quantised. It read as a gentle gradient error, maximum 34
    of 255, with the top-left 32x32 already exact.

  **How to debug it, because the harness is most of the work.** `tools/videodiff
  hevcseq <file>` decodes with our decoder and writes raw I420; a PyAV venv
  (`python3 -m venv /tmp/avenv && /tmp/avenv/bin/pip install av numpy`) decodes
  the same file as the reference. **PyAV ships libx265, so targeted clips can be
  generated to bisect the feature space** — that is what found WPP, and it turns
  a guess into a measurement. Set `dec.trace_on` for a per-CU/TU trace with the
  CABAC byte position, which distinguishes a *syntax desync* (bytes consumed far
  from the slice length) from a *reconstruction* error (bytes right, pixels
  wrong). Two traps in reading that trace: `byte_pos` includes up to 4 bytes of
  reservoir look-ahead and saturates at the slice end, so "61 of 61" is only
  accurate to +-4; and the CTU loop exits on the CTB count, so
  `end_of_slice_segment_flag` returning 1 proves nothing about sync.

  **HEVC Main plays, bit-exact against FFmpeg.** Zero differing samples on the
  videodiff suite: multi-CTB all-intra, I+P, hierarchical B-pyramid, WPP, SAO,
  deblock, TMVP, sign-hide, mid-stream CRA with RASL leading pictures, and
  realish default-x265 GOPs (including 320x240). Two late silent bugs that made
  "almost every clip" match while a typical real encode did not:

  - **`ref_idx_l0` and `ref_idx_l1` share the L0 CABAC contexts** in every
    production encoder/decoder (x265's `OFF_REF_NO_CTX` is list-agnostic;
    FFmpeg hard-codes `REF_IDX_L0_OFFSET`). The specification's table has a
    separate L1 pair; using it desynchronises the first bi-predicted AMVP block
    that has more than one L1 reference — the hierarchical B-pyramid leaf case.
  - **A mid-stream CRA must not empty the DPB.** IDR/BLA set NoRaslOutputFlag
    and clear references; a continuous CRA keeps them so RASL leading pictures
    can still predict from the previous GOP. Clearing on every IRAP made three
    frames before each keyint CRA pure noise while everything else stayed exact.

  Still refused rather than guessed: tiles, PCM, 10/12-bit. The rule stands —
  diff whole frames against PyAV before claiming anything works.
- **Switchable engines, and the rule for adding another.** Two subsystems run an
  alternative implementation **alongside** ours on the `/decoder ring3|kernel`
  pattern — a `bench` subcommand compares them:
  **`/heap firstfit|sizeclass`** (`mm/heap.rs`) and **`/html ours|tl`**
  (`browser/html_tl.rs` over the vendored `third_party/tl`). Note the defaults
  point opposite ways, and deliberately: **size-class is the default** because it
  moved page boot 3.8x, while **our HTML parser stays the default** because
  `tl`'s 6.75x is 6.75x of ~3 ms — its value is being a second implementation,
  not being faster. Five rules, each of which cost a debugging cycle or a wrong
  number:
  - **A/B only what is like-for-like; share everything else.** The `tl` adapter
    reuses `extract_assets_rich`, `preprocess`, `set_attribute` and
    `finalize_document` verbatim — the last two were *extracted* for it, and the
    third was found missing when the cross-engine test compared a title of
    `"Untitled"` against `""`. Otherwise the comparison measures the divergence
    rather than the engine.
  - **A bench must never fall back; a page load must.** An adapter returns
    `None` on failure so a measurement cannot attribute our result to the
    challenger; the page path may fall back but must **count** it, so "X is
    selected" and "X actually did this" stay distinguishable.
  - **Report agreement before speed.** `/html bench` prints whether both engines
    built the same tree, and restores the previous engine afterwards. A faster
    engine that produces something else has changed the page, not won.
  - **A shared front end dilutes the ratio.** `/html bench` times
    `extract_assets_rich` + `preprocess` for both engines because a page load
    really pays them, so on a small page the ratio collapses toward 1x —
    **1.9x on a 2 KiB fixture against 6-9x on real 700 KiB pages**
    (`tools/webbench`). The command says so under 64 KiB.
  - **A fixed iteration count is only sound when the thing timed is slow
    relative to the counter.** Flex placement is microseconds, so at 2,000
    iterations four runs of the same benchmark reported 21.8x, 6.6x, 18.0x and
    82.8x — the *denominator* was noise. `shell::time_until` extends a batch
    until it reaches a tick floor and the benches report **per iteration**,
    which is also what makes batches of different length comparable; a batch
    that never reaches the floor prints SUSPECT instead of a ratio.

  **Where the time actually goes, measured:** on the shadcn page, JS is ~99% of
  page boot (686 ms of 695 ms; 2651 of 2654 before the allocator). HTML parsing
  and flex layout are rounding errors, which is why `tl` is kept as an oracle
  rather than adopted, and why the next real lever is the JS engine.

  **taffy was evaluated and removed.** It agreed with `browser::flex` to within
  1 px on every shape — rows, columns, gaps, `space-between`, wrap, `flex-grow` —
  and cost ~20x per placement under first-fit and **~70x under the size-class
  heap**: the gap *widens* with a better allocator, because both are
  allocation-dominated and ours more so (size-class made taffy 4.1x faster and
  ours 9.0x). That cost is a property of a per-call boundary that builds and
  drops a whole `TaffyTree`, not a verdict on taffy, whose design is a persistent
  tree with dirty-subtree recompute — but there is no adoption case at this
  boundary, so the dependency is gone. **The agreement is the finding worth
  keeping:** two independently written implementations landing on the same pixels
  is evidence `browser::flex` is correct that our own unit tests structurally
  cannot provide. Re-vendor it as a `#[cfg(test)]` oracle if flex is ever
  reworked; do not put it back on the page-load path.

  Neither `tl` nor ours implements implied end tags, so switching does **not**
  buy `<p>a<p>b` as siblings.

- **Agent chat protocol** — the shell chat is an agentic ReAct loop on the
  Qwen3.5 template: the prompt advertises a small CORE tool set plus
  `search_tools` (progressive discovery over the registry — manifest
  toolset ∩ `tools::registry`; never hardcode a tool list in a prompt),
  `<tool_call>` JSON in, `<tool_response>` back, thinking off by default
  (`/think`), `/mode manual|auto|bypass` gates agent tool calls through the
  modal, Ctrl+C/Esc cancels prefill *and* decode, `/compact` rebuilds the KV
  from a model-written summary. Agents are processes: `/agents` lists the
  scheduler tasks that carry agent identity, `switch` re-homes the chat,
  `kill` revokes a task's capability table. Every agent has
  `/agent/<id>/{SOUL.md,skills/,memory/}`; SOUL.md is prepended to its system
  prompt. The shell agent is the only default agent (boot demos removed).

## STANDING RULE — one keyboard choke point; bytes are not columns

**Every keyboard transport funnels through `kernel/src/keymap/`, and nothing else
decodes a scancode.** Before it existed there were four independent decoders
(`arch/x86_64/keyboard.rs` set-1, `arch/aarch64/pl050.rs` set-2, `xhci.rs` HID
usages, `arch/aarch64/virtio_input.rs` evdev), each with its own modifier state,
its own copy of the arrow→CSI table, its own caps-lock rule (two had one, two did
not), and all four hard-coded to **US**. A driver's job now ends at *"which
physical key was that, and which modifiers were down"*; it builds a
`KeyEvent { usage, mods, pressed, src }` and calls `keymap::feed_event`.

Four facts forced that shape, and each is a constraint on any change here:

- **x86 decodes inside IRQ1.** So `feed_event` allocates nothing and only pushes a
  4-byte `Copy` struct into a fixed ring; `translate` (which allocates — one press
  can emit several characters) runs on the **drain** side, in `keymap::next_byte`.
  Do not move translation into a driver.
- **`arch::aarch64` is `cfg`'d out of the test build**, exactly like `framebuffer/`
  but from a different `cfg`. Anything left in `pl050.rs`/`virtio_input.rs` can
  never carry a `#[test_case]`, which is why the set-2 and evdev cross-tables live
  in `keymap` and not in the drivers that use them.
- **Dead keys, Compose and the IME are stateful across keystrokes**, and a machine
  can have a USB keyboard *and* a virtio-input window at once (`console.rs` polls
  both). Four states would mean `´` on one keyboard and `e` on the other fails to
  compose. Caps Lock lives in `keymap::State` for the same reason.
- **`console::read_byte() -> Option<u8>` does not change.** The event type stops at
  the choke point; above it the OS still sees a byte stream, so the shell, editor,
  modals, `poll_interrupt` and every e2e scenario are untouched.

**HID usages are the canonical space** because a layout maps a *physical position*
to a character, and set-1/set-2/evdev are positional too — the three cross-tables
are ~350 bytes of pure relabelling. An ASCII-based canonical space cannot express
"the key left of Y on an ISO board" (usage `0x64`), which is where German puts
`<>|` and which no previous table decoded at all. **Right Alt is its own modifier
bit**: `xhci` read `report[0] & 0x44`, the OR of both Alts, into nothing, which is
precisely why AltGr never worked anywhere. `Ctrl+Alt` counts as AltGr too, as XKB
does, because many keyboards have no right Alt and a macOS host eats Option.

Layout data is in `keymap/layouts.rs`: a dense `US_BASE` plus per-layout **sparse
overrides**, four levels (`Base`/`Shift`/`AltGr`/`ShiftAltGr`), `Out::None` for an
undefined level (never a fallback to another level — that types a *wrong*
character rather than none). Nine layouts, ~11 KB of rodata, written out rather
than bit-packed on purpose: `c('ö')` in the wrong slot reads exactly like `c('ö')`
in the right one, so readability is the safety property. Two tests keep a
hand-written table honest — every layout can type all 26 letters and 10 digits,
and no layout lists a usage twice.

**`us_layout_reproduces_the_legacy_*` are the migration gate.** The four old
decoders' tables are copied verbatim into `keymap`'s tests and every
scancode × modifier combination is asserted to produce the same byte. Two of those
drivers cannot be compiled by the test build at all, so this is the only place
their behaviour is pinned; do not delete these fixtures casually.

**Scancodes cannot be injected over serial**, so `/keyboard test <layout> <keys>`
exists to make the tables assertable from a running kernel — and it calls the real
`keymap::translate`, never a parallel path. A key is named by its **US base
character** with modifier prefixes (`/keyboard test de altgr+q` → `@`), plus named
keys for the ones with no character (`iso` is usage `0x64`).

### Bytes, chars and columns are three different numbers

`kernel/src/textfit.rs` holds the arithmetic, and the bug class it ends is worth
naming: `String` indices are **bytes**, terminal geometry is **columns**, and for
ASCII they are the same number — so the line editor, the composer and the modal
wrap all used `buf.len() - cur` as a column count and `s.as_bytes().chunks(cols)`
as a line break. The moment a non-ASCII character reaches them that is a **panic**
(`buf.drain(cur - n..cur)` on backspace; `&line[start..start + max_cols]` in the
composer), a caret in the wrong place, or mojibake. Use `textfit::cols`,
`back_n_chars`, `visible_window`, `pad_trunc` and `wrap`; never subtract byte
offsets to get a column count. `wrap` and `pad_trunc` moved out of
`framebuffer/text.rs` for the usual reason — a test written next to them would
never have been compiled, which is how `wrap` shipped silently *deleting* any word
whose byte-chunk boundary fell inside a character.

`char_cols` answers East-Asian width coarsely (0 for combining marks, 2 for
Wide/Fullwidth). **The pane `Cell` grid is deliberately out of scope**: it is
`(char, Rgb)` with one cell per character, so a wide glyph would need a lead cell
plus a continuation marker and then `set_cell`, the scroll row copy, the selection
maths and `glyph_cell` all change — inside the module where no test compiles. The
visible consequence, stated so nobody reports it as a bug: typed CJK is *correct*
in the composer (caret in the right place, backspace deletes one glyph) and renders
**narrow** once echoed into a pane, because `font_ttf::build_ui_glyph` scales a
non-ASCII glyph down into its cell. Cramped, not corrupt.

### The IME ships what it can deliver honestly

`kernel/src/ime.rs` is a pure `feed(char) -> ImeOut` machine, fed **after** the
layout stage (so it composes over Dvorak and over AltGr output) and **before**
`media_key` (so a composing keystroke is never eaten by a focused app).
`consumed: false` is the regression guard: with `Mode::Off` it returns immediately
and the caller behaves exactly as before.

- **romaji → kana is complete** — hiragana and katakana, no dictionary, and every
  glyph it can produce is inside the bundled CJK subset, so what you type renders.
  The `nn` case is the classic trap: `nn` is ん but `nni` is んに, so the second `n`
  **cannot be resolved until the next character arrives**. Converting it eagerly
  turns "konnichiwa" into こんいちわ.
- **Hangul is implemented, tested, and refused.** Jamo→syllable is arithmetic
  (`0xAC00 + (L*21 + V)*28 + T`) plus the 2-set map and the compound merges. The
  bundled face has **no Hangul glyphs**, so it would compose perfectly and render
  tofu — `set_mode_by_name` refuses with that reason and `font_ttf::fallback_covers`
  is the gate, so bundling a Hangul subset later is a one-line change with tests
  already green.
- **Pinyin and kanji conversion are refused, naming the missing dictionary.** A
  pinyin engine that knows 500 words is *worse* than none: the user types a word it
  lacks and gets silence or the wrong character, which is a mis-decode.

Two escape hatches are load-bearing, because an input method with no exit is a
trap: `/` and `~` are **absent from the romaji table** even though a real IME maps
them, and `read_line` bypasses the IME entirely on a line starting with `/` — so
`/keyboard ime off` is always typeable.

Also: the serial console now passes **UTF-8 through** rather than dotting out every
byte ≥ 0x80. That filter was right while the OS could not hold non-ASCII text and
became a lie the moment it could.

## STANDING RULE — Ctrl+C interrupts every command and process

**Any new command or long-running process must respond to Ctrl+C.** A blocking
loop already has to pump `shell::upkeep()` (above); in the *same* loop it must
also poll for interrupt and bail when it fires — otherwise a stuck/streaming
command can only be escaped by killing the VM, which is not acceptable.

- **A shell command / networked loop** (`/http`, `/ping`, DNS, TLS, a `/ws`-style
  stream, any future protocol client) calls `shell::poll_interrupt()` — true only
  on Ctrl+C (`0x03`), and it **pushes any other byte back** (`console::unread`) so
  it never swallows the *next* command's keystrokes. Return an `Err("cancelled")`
  / break the loop on true.
- **An inference / decode loop** calls `shell::poll_cancel()` (Ctrl+C **or** bare
  Esc; a decode turn owns the console, so consuming input there is fine).

Both are cheap — call them once per loop iteration next to `upkeep()`. A command
that can block without an interrupt check is a bug; cover it with an e2e `cancel`
scenario (drive raw `b"\x03"` via `guest.send_raw`, assert it aborts fast **and**
the next command still runs) and a unit test on the pure poll logic
(`poll_interrupt_ctrl_c_only_and_pushes_back`, `console::pushback_*`).

**There is exactly one documented exception: the login gate**
(`auth::prompt::gate`). A gate you can Ctrl+C out of is not a gate. Ctrl+C and Esc
stay *responsive* there — they clear the typed field and repaint, so the key is
never dead — but neither returns from the loop. Do not "fix" it.

## STANDING RULE — human-only commands are absent from `dispatch_system`, not guarded inside it

`/passwd` and `/lock` (`kernel/src/shell/auth.rs`, [`crate::auth`]) are matched in
the **interactive-only** arm of the REPL, above the `dispatch_system`
fall-through, and must never move into `dispatch_system`. This is the security
property, not a filing decision:

- **`run_shell_command` needs no registry entry.** `tools/shell_cmd.rs` passes any
  `[A-Za-z0-9_-]+` token straight to `dispatch_system`, registered `ToolDef` or
  not — and it is in `CORE_TOOLS` and the orchestrator manifest. So the moment a
  human-only command gains a `dispatch_system` arm, **every agent can call it**,
  with no capability and no manifest change.
- **There is no `Right::Shell` and no `CapDomain::Shell`.** Shell commands are not
  capability-gated at all, so "don't grant the capability" was never available.
- **`dispatch_system` has exactly one agent-reachable call site**
  (`run_tool_command`), and *everything* funnels through it — `run_shell_command`,
  `tools::bg::pump`, the `ToolBinding::Shell` executor, scheduled `Action::Command`
  fires, Telegram-driven turns, package-UI apps and sub-agents. The last two call
  `Router::call` directly and **skip `tool_in_chat_toolset` entirely**, so a
  manifest toolset is not a boundary either. Being absent from `dispatch_system`
  closes all of them at once, by construction.

The `in_tool_call()` refusals inside those handlers are defence in depth. Keep
them, and note that **taint cannot answer this question**: a Telegram DM enters
the session as `Provenance::UserTyped`, so `UserTyped` does not mean "a human at
this console". `in_tool_call()` is the only signal that does — and it is a **depth
counter**, not a flag, because `bg::pump` calls `run_tool_command` from inside
`upkeep()`, which long-running commands call, so a nested call's exit used to
clear the marker while the outer dispatch was still running (that bug also
affected `/screenshot` and `/record`). Pinned by
`passwd_and_lock_are_unreachable_from_the_tool_surface` and
`in_tool_call_survives_a_nested_run_tool_command`.

`catalog::RESERVED_HUMAN_ONLY` carries the names so `/schedule add … command
passwd` is refused at creation (a fire has no human at the keyboard, so it could
only ever fail — silently, at 3am) and no package manifest can claim one as a
`command_hooks` name.

## STANDING RULE — key material comes from `security::rng`, never `arch::hw_rand()`

`arch::hw_rand()` returns **0** when `RDRAND`/`RNDR` is absent — which is every
QEMU and HVF default CPU model, i.e. every machine we develop and test on. A
caller that fills a buffer straight from it produces **all zeros** on exactly the
machines the tests run on, and real entropy only on hardware nobody tests against.
`block::volcrypto::fill_random` did precisely that, so every C4VE volume formatted
under QEMU had an all-zero salt and an all-zero master key — invisible, because an
all-zero key encrypts, decrypts, mounts and accepts its passphrase perfectly.

Use `security::rng::fill_random` / `seed_rng` (ChaCha20 over hardware words *plus*
cycle-counter jitter across yields). It cannot degrade to zeros: the diffuser folds
the cycle counter in unconditionally, which
`entropy_survives_a_dead_hardware_rng` pins with `hw = 0` — the companion
"not all zeros" test alone passes under `-cpu max` even with the bug present.
`net::tls::seed_rng` is a re-export of the same function, so there is one
implementation. NB it yields, so it must not be called with a `Locked` held.

## The login gate (`kernel/src/auth/`)

Single-user, fixed user `chitti`, independent of the volume passphrase. Boot gate
(in `shell::run`, after the theme/fonts/display/panes are applied so it looks like
this OS, and **before** the session resume, which prints the previous session's id
and message count — disclosure before authentication), `/lock`, idle auto-lock,
and re-auth on resume from suspend. Pure logic in `auth/mod.rs`; the blocking
screen is `auth/prompt.rs`, `#[cfg(not(test))]` with a **deny-by-default** stub.

Six things worth knowing before touching it:

- **No record = no gate, anywhere.** One state, one `fs::exists`. That is also why
  the e2e harness needs no bypass flag: its guests run on a memfs store where
  nothing can ever have been enrolled, so every existing scenario is untouched.
- **The gate mirrors to serial and must keep doing so.** `modal::input` renders
  *only* to the framebuffer, so a modal-based gate is invisible to
  `-serial mon:stdio` and to the whole e2e harness — a machine that has silently
  stopped responding. It also returns `""` on Esc, which for an auth prompt is an
  empty password attempt, not a cancel. Hence its own loop.
- **It pumps `status_tick()`, not `upkeep()`, and that is load-bearing.** Besides
  the usual "a modal consumes its own clicks" reason, the shell task stays
  *runnable* throughout, so the scheduler never reaches the idle pump task — the
  only caller of `upkeep()` — so `msgchan::tick`, `schedule::tick`,
  `service::supervise_tick` and `tools::bg::pump` do not fire behind the lock
  screen. A Telegram DM cannot drive an agent turn while the console is locked.
  Swapping in `upkeep()` looks like a harmless cleanup and silently opens that door.
- **The credential record is protected by two layers, and the second is the one
  that enforces it.** The Synapse executor denies `/configs/core/auth.json` for
  read *and* write/edit/delete (layered inside Gate 4 — `GATE_COUNT` stays 4, the
  paper's published contract), which is the correct statement of the policy. But
  `/cat`, `/rm`, `/cp`, `/mv`, `/touch`, `/grep` and `/glob` are `ToolBinding::Shell`
  tools that reach `synapse::fs` **directly and never enter the executor**, so the
  refusal that actually holds is the `CredentialAccess` guard in the store facade.
  `synapse::fs::list` filters the record out, which covers `MEM_FS_LIST`,
  `MEM_FS_SEARCH` and `Router::readable_paths` (glob/grep/list_dir) in one place —
  and makes `rm -r` of an ancestor skip it by construction.
- **The idle timestamp already exists twice and covers everything.**
  `console::read_byte` stamps in the *merged* reader (serial, PS/2, xHCI/HID,
  virtio-keyboard, PL011) and `mouse::activity_ms()` is the pointer half. Do not
  add a third, and never call `mouse::tick()` to read it — that is a *consuming*
  poll that would steal the caller's clicks. The check is polled **only** in
  `read_line`'s idle arm, because "idle" means idle *at the prompt*: it must not
  fire mid-`/http`, mid-inference or mid-video.
- **Resume auth goes at the end of `power::resume_devices()`**, after the i8042 is
  re-initialised and interrupts are unmasked — the controller comes back with its
  configuration byte reset, so a prompt placed earlier has no keyboard and the
  machine looks hung rather than locked. There rather than in `run_suspend` so
  every resume path locks, including a future lid-close handler.

Honest limits, stated in the module doc and in `/passwd`'s own output rather than
discovered: on an **unencrypted** disk the gate is bypassable offline in minutes
(mount elsewhere, delete the record) — it is a *console* lock, not
confidentiality, and `/encrypt` is what protects the data; on a memfs store it
does not persist; PBKDF2-HMAC-SHA256 is GPU-friendly (the `kdf` field exists so a
future argon2 is a migration); passwords are printable ASCII only, enforced at
enrolment because both input paths are; a reboot resets the backoff; and there is
no recovery path.

## Build / run / test

Everything goes through `cargo xtask`. Arch is chosen explicitly, never
host-detected. See [DEVELOPMENT.md](DEVELOPMENT.md) for the full setup.

```sh
cargo xtask test                       # in-kernel unit suite under QEMU (x86) — pure logic, no model
cargo xtask ring-check                 # enforce the ring-3 rule: no direct synapse::executor
                                       #   calls outside the allowlist (xtask/src/rings.rs)
 make e2e                               # end-to-end: boot the kernel, drive the shell over serial,
                                        #   exercise every OS command + the http/https/ws/wss/ping/
                                        #   hosted-model flows vs local servers (tests/e2e/, stdlib-only
                                        #   python; TLS scenarios need a TLS-1.3 python — Homebrew's)
                                        #   `make e2e E2E_JOBS=3` splits the sweep across 3 concurrent
                                        #   guest boots (~1 min vs ~13 min serial on a multicore Mac)
 make e2e-full                          # + local inference (/infer,/perf,chat,/compact) and voice
                                        #   (/voice say) — slow; needs assets/model.gguf + assets/voice/
cargo xtask build -arch x86_64|aarch64 # cross-build the kernel
cargo xtask run   -arch x86_64|aarch64 # boot in QEMU (aarch64 = native HVF on Apple Silicon)
cargo xtask image -arch x86_64|aarch64 # assemble a bootable image/ISO
cargo xtask sample-files [--refresh]   # fetch the /samples corpus into assets/samples/
                                       #   (the build paths do this for you — see below)
```

**Every decoder needs a file, and a fresh boot has none** — which made the media
stack (ring-3 PNG/JPEG, H.264+AAC, MP3/WAV/AAC, the PDF wasm, the editor's
highlighters) awkward to even try without `/http -O` first. So a ~15 MiB corpus is
embedded in the image and seeded into the store at boot — `/samples/images/`,
`/samples/videos/`, `/samples/audios/`, `/samples/misc/`, `/samples/js/`
(authored scripts for `/js`, from `assets/samples-src/js/`), plus a
`/samples/README.md` recording each file's source and licence — and
`/open /samples/images/fruits.jpg` works offline on the first boot
(`kernel/src/samples.rs`, `SAMPLE_FILES` in `xtask/src/main.rs`,
`kernel::build.rs`'s directory walk).

Five rules it follows, each of which is a way this could have gone wrong:

- **Opt-in, and default-on only for the dev flows.** `CHITTI_SAMPLE_FILES` gates
  it; `make run|run-remote|run-uefi|image|vbox|e2e` set it (the `SAMPLES` knob),
  and a plain `cargo xtask build` / CI / `cargo xtask test` embeds nothing, so
  those kernels are byte-identical to before. **Empty reads as unset** — `make`
  passes the variable through unconditionally, the same trap `CHITTI_RESOLUTION`
  hit.
- **Fetched, never committed** (`assets/samples/` is gitignored), so the tree
  redistributes nothing and no licensing decision was needed — the same rule the
  voice and WiFi assets follow. A failed download is a **warning, not a failed
  build**: the OS is fully functional without samples, and the alternative is a
  machine with no network being unable to build.
- **One definition, walked not duplicated.** xtask owns the table (URL,
  destination, provenance, and an `openable` flag); `kernel/build.rs` walks the
  directory and generates the `include_bytes!` table. A second list in the kernel
  would be a second thing to drift.
- **Seeding never overwrites.** On an installed system the store is ext4-backed
  and durable, so re-writing every boot would silently revert a sample the user
  edited; `samples::seed_plan` writes only absent paths (and is unit-tested on
  exactly that).
- **A format with no decoder is labelled, not omitted.** `sample.ogg` is there as
  the next decoder's input, marked unopenable, and the corpus test asserts every
  *openable* entry has an extension `/open` really handles — so a `.flac` cannot
  creep in as a file that only ever errors.
- **A one-page fixture is not a document, and the PDF set says so.** A synthetic
  single page had made the renderer's limits look generous; the first real paper
  needed twice the memory. So `/samples/misc/` carries the shapes that actually
  differ: `pdflatex-4-pages.pdf` (24 KiB — the cheap multi-page case for e2e),
  `pdflatex-image.pdf` (an embedded colour JPEG, i.e. DCTDecode, which no
  vector-only document reaches) and `geotopo.pdf` (**117 pages** with 19 JPEG
  figures — long-document navigation, ~0.5 s/page).
- **Freely redistributable is a stricter rule than "fetched, never committed".**
  A `CHITTI_SAMPLE_FILES` build embeds these bytes in a kernel image and images
  get passed around, so it is not enough that the *tree* carries no copy. That is
  why the corpus holds no arXiv paper: the default arXiv licence lets arXiv
  distribute the PDF, not third parties. Fetch such a document on the running OS
  with `/http -O <url>` instead — which is also how the renderer's limits were
  measured (the Transformer paper, arXiv:1706.03762, whose two attention-matrix
  pages remain the heaviest thing measured here: 56 MiB of guest memory and
  ~6.9 s each at pane fit). Reproduce with `tools/pdfbench` on any local copy.

## Conventions

- **The OS is named `ChittiOS` — one word, no space.** Use it everywhere: docs,
  boot banner, status bar, SOULs, served pages, commit messages. "Chitti" alone
  refers to the project/brand; the spaced two-word form is wrong — fix it on
  sight (and rebuild any `tools/*-wasm` module whose strings embed it).
- No `unsafe` without an adjacent `// SAFETY:` comment justifying each invariant.
- Every public module has a doc comment stating its responsibility.
- Deterministic by default: tests use fixed seeds + temperature 0; any RNG is seeded.
- `ktrace` every capability invocation and every inference call.
- Commit per sub-milestone with a clear message; never leave the tree non-building.
- Match the surrounding code's style, comment density, and idioms.

## Status

**Active development — not stable.** Interfaces, on-disk formats, and behaviour
change without notice. This is a research OS; run it in a VM. See the caution in
[README.md](README.md).
