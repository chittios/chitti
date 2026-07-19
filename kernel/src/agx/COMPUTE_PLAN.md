# AGX compute submission — implementation plan (scoping)

Status as of this doc: the **AP→FW command-submission primitive is proven** on the
real M2. `/agx up` brings the gfx-asc to RUNNING, replays initdata, MSG_INITs it,
and the firmware accepts it (`Stats` cursor advances) and **dequeues+executes
DevCtrl commands we write** (`DevCtrl READ_PTR 2→4` after `DC_Init`+`DC_UpdateIdleTS`).
So: *write a message into a ring → ring a doorbell → firmware consumes and acts* —
the exact mechanism GPU work submission uses — works end to end.

This doc scopes the remaining path to running a GEMM on the GPU and wiring it into
`cortex`'s `matvec_qw`/`batched_proj`.

## What a GPU work submission actually is (from the reference)

Reference: `third_party/m1n1/proxyclient/m1n1/{agx,fw/agx}/*.py` and
`experiments/agx_1tri.py`. A submission = a `WorkCommand*` struct placed in a GPU
context's memory, referenced by a `RunCmdQueueMsg` written into a **cmdqueue
channel** ring, kicked with a doorbell. Completion is signalled on the **Event**
channel (a stamp/counter the firmware bumps).

Command/queue types (AP-side channel id = `(queue_index<<2) | type`):
- `type 0 = TA` (vertex/tiling), `1 = 3D` (fragment), `2 = CL/CP` (**compute**).
- Work command variants (`CmdBufWork.cmd_type`): `0 WorkCommandTA`, `1 WorkCommand3D`,
  `3 WorkCommandCP` (compute), `4 Barrier`, `6 InitBM`, `10/11 ComputeUnk10/11`.

### Compute command (`WorkCommandCP`, `fw/agx/cmdqueue.py:132`)
Fields that matter: `magic=3`, `context_id`, `event_control_addr → EventControl`,
`compute_info (ComputeInfo)`, `registers_addr`+`register_count` (the dispatch
register state), `microsequence_ptr → MicroSequence`, `compute_info2`,
`encoder_params`, `job_meta`, timestamps.

### Compute microsequence (`fw/agx/microsequence.py`)
`StartComputeCmd (magic 0x29)` → … → `FinalizeComputeCmd (magic 0x2a)`. Start
points at `computeinfo_addr`, `cmdqueue_ptr`, `context_id`, `event_ctrl_buf_addr`.

### The minimal compute launch (the key hint)
`ComputeInfo` (`microsequence.py:708`) is documented:
> "Only the cmdlist and pipelinebase and cmdlist fields are strictly needed to
> launch a basic compute shader."
i.e. `encoder` (a GPU **command list** that dispatches the shader) + `pipeline_base`
(0x11_00000000, where shader/pipeline code lives). Everything else is bookkeeping.

## The three layers, and where the work is

| Layer | Portable from m1n1? | Effort | Notes |
|---|---|---|---|
| 1. GPU context / VM | Yes (`agx/context.py`, 259 lines) | ~300–400 LOC Rust | fresh TTBR0 L1, `uat.bind_context(ctx_id, ttbr0)`, object heaps w/ mem-attrs, `pipeline_base=0x1100000000`. We already have the UAT encoder (`uat.rs`). |
| 2. Submission plumbing | Yes (`cmdqueue.py`+`microsequence.py`+`channels.py`) | ~1500–2000 LOC Rust | mechanical struct encode: `WorkCommandCP`, Start/FinalizeCompute, `RunCmdQueueMsg` on the CP channel, `EventControl` poll for completion. Pure logic → unit-testable off-hardware. |
| 3. **Shader + command-list** | **No** | the real blocker | AGX ISA machine code for GEMM + the compute dispatch encoding (USC words, grid/threadgroup dims, arg-buffer bindings). Not in m1n1 — normally emitted by Metal/Mesa. |

Layers 1–2 are "just" a large, testable port. **Layer 3 is the true dependency.**

## Layer 3 options (the shader/command-list)

1. **Mesa AGX compiler (offline).** Compile a GEMM compute kernel to AGX ISA on a
   host (Asahi Mesa's `asahi`/`agx` backend), extract the shader binary + the
   pipeline/USC layout it expects. Cleanest, reproducible; cost = standing up the
   Mesa AGX toolchain and understanding its ABI.
2. **`dougallj/applegpu` assembler.** Hand-write/assemble a minimal GEMM in AGX ISA
   using the community assembler + ISA docs. Most control, least framework; cost =
   writing correct AGX ISA by hand.
3. **Extract from a macOS Metal compute pipeline.** Capture a compiled compute
   `MTLComputePipelineState` binary + its argument encoding. Fast to get bytes;
   cost = the extraction/repro is macOS-bound and opaque.

Recommendation: **(1) Mesa AGX compiler** for the kernel binary, cross-checked
against **(2) applegpu** for the ISA/USC understanding. Defer choosing until Layer
1–2 are validated (below), since they're independent.

## Layer 3 toolchain investigation (2026-07)

Split the problem in two — they're independent and have different difficulty:

**(3a) The shader ISA — assembling instructions — is the TRACTABLE half.**
- `dougallj/applegpu`: `assemble.py` (hacky but works) + `applegpu.py` (ISA) +
  disassembler + emulator + `hwtestbed.py`. **Targets G13 (M1)**; `hwtestbed`
  injects into **macOS Metal** binary archives (macOS-tethered). G14/M2 ISA is
  close but not identical — usable as a reference, not turnkey for our M2.
- **Mesa** (`src/asahi/compiler`) is the authoritative compiler: full AGX/AGX2
  ISA incl. **G14/G14X**, an OpenCL front-end (`asahi_clc`), XML ISA + XML
  disassembler. It can compile a GEMM/matvec compute kernel to real G14 ISA.
- A matvec/GEMM kernel is small (load, FMA loop, store) — assembling it is not the
  blocker.

**(3b) The compute DISPATCH ABI — USC words + WorkCommandCP register array — is
the REAL gap.** How the shader address, uniform/argument buffers, and threadgroup/
grid dims are bound. Findings:
- The Asahi **kernel** docs (asahilinux.org/docs/hw/soc/agx) explicitly declare
  this **"purely of userspace concern… out of scope"** — it is NOT documented at
  the firmware/kernel level.
- m1n1 has **no compute example** and does not annotate the compute registers
  (`RegisterDefinition {number,data}`; USC_EXEC_BASE 0x10069 etc. appear only in
  the *render* path, uncommented). So there are **no capturable reference bytes**
  and no doc for the compute USC.
- The ONLY complete public reference is **Mesa source** (`src/asahi/lib` +
  the compute launch: USC packing, `ComputeInfo`/register-array construction,
  `libagx`). Reverse-engineering the dispatch = reading that code.

**Revised Layer-3 paths, by confidence:**
1. **Capture from Asahi Linux + Mesa on the M2 (highest confidence).** Dual-boot
   Asahi on the Mac mini, run a Mesa OpenCL/compute job, capture the WorkCommandCP
   + USC + shader (drm/asahi cmdstream dump), replay/port on ChittiOS. This is the
   compute analog of every capture that has worked for us (initdata, ttbs bind) —
   known-good ground truth. Cost: dual-boot + dump tooling.
2. **Port from Mesa source (most self-contained).** Read Mesa's compute launch +
   USC packing, port to Rust, compile the matvec kernel with Mesa's compiler.
   No external runtime dependency at run time; cost = weeks of RE from source.
3. **macOS Metal capture (fallback).** Capture a Metal compute pipeline + command
   buffer. macOS-tethered; Metal's encoding differs from Mesa's in ways that may
   not match the firmware path we drive.

**Recommendation:** Path 1 to get known-good compute bytes (mirrors what has
worked), cross-referenced with Path 2 (Mesa source) to *understand* them, and
Mesa's compiler for the kernel ISA. Assembling the shader (3a) is easy; the
dispatch ABI (3b) is the multi-week reverse-engineering core. This is a distinct
project from Layers 1–2 and gates any hardware validation of them.

## Proxy-session findings (2026-07, real M2) — CONFIRMED ON HARDWARE

Running the capture against the live proxy settled the two biggest unknowns:

1. **There is NO self-contained GPU submission in m1n1.** `agx_1tri.py` looked
   self-contained but actually `ctx.load_blob(0x1100000000, …, "gpudata/bunny/
   mem_*.bin")` — it loads **Metal-captured shader/pipeline bytes** into
   `pipeline_base`. `agx_renderframe.py` takes a saved `GPUFrame` capture as
   `argv[1]`. Those `gpudata/` assets are **not in the m1n1 tree**. So even the
   built-in clear/store pipelines come from captured data → **Layer 3 (shader
   bytes) is unavoidable even for a render.** A "capture a known-good render to
   validate the plumbing" plan is blocked unless we first source `gpudata/bunny`
   (a macOS Metal capture) or compile our own shaders.
2. **Layer 1 is fully specified from source — no byte capture needed.** A context
   is: a zeroed 16 KiB TTBR0 L1 (`memalign`), `bind_context(ctx_id, ttbr0)`, and
   `iomap_at` calls we already have (`uat.rs`). The proxy confirmed the exact ttbs
   bind descriptor for a fresh ctx: `L0[ctx].0 = ttbr0|ASID(ctx)|VALID`,
   `L0[ctx].1 = ttbr1_shared|ASID|VALID` (e.g. ctx 3 → `0x3000814884001` /
   `0x30009fff78001`). VA layout: pipelines `0x1100000000`, GEM `0x1500000000`,
   userspace `0x1600000000`, ctx "thing" `0x6fffff8000`; MemoryAttr.Shared, nG=1.

Consequence: **Layers 1–2 can be built straight from m1n1 SOURCE without the
proxy** (zeroed tables + struct encoders). The proxy/capture is only needed to
(a) source a reference shader (Layer 3), or (b) validate a completed submission.

## De-risking sequence (revised after the proxy findings)

The original "replay a captured render" step is blocked on shader data. Two viable
orderings:

**Path A — plumbing first, shader last (recommended).**
1. **Port Layer 1 (context) + Layer 2 (cmdqueue + WorkCommandCP + microsequence +
   Event poll) from m1n1 SOURCE.** No proxy needed. Pure encoders get
   `#[test_case]` tests. Build the full submission machinery with a *placeholder*
   shader region.
2. **Source ONE real AGX shader** (Layer 3): a trivial compute kernel via Mesa's
   AGX compiler (or `dougallj/applegpu`). Just enough to write a known value to a
   result buffer.
3. **Submit on the CP channel; wait for the Event stamp; read back the buffer.**
   Success = the firmware ran our shader and completed the job. This proves the
   whole stack end-to-end with the smallest possible shader.
4. **Wire into cortex:** upload weights to GPU memory once at `Model::load`; per
   matmul encode a `WorkCommandCP` dispatch (`matvec_qw`/`batched_proj`); validate
   greedy parity vs `tools/cortexdiff` / `cargo xtask ref-check` (rel-RMS tol —
   GPU reduction order ≠ NEON).

**Path B — source `gpudata/bunny` and validate render plumbing first.** If the
bunny Metal capture can be obtained, `capture_render.py` yields a complete
known-good render (shaders incl.) to replay, proving Layers 1–2 before writing any
shader. Falls back to Path A for compute regardless.

## Testing posture (per repo standing rules)

- Layers 1–2 pure encoders (`WorkCommandCP`, microsequence, `RunCmdQueueMsg`,
  context PTEs) live outside `arch::aarch64` and get `#[test_case]` unit tests under
  `cargo xtask test` (the `uat.rs`/`proto.rs` pattern) — fed captured bytes,
  asserted field-for-field.
- Hardware validation is `/agx` on the real M2 (proxy-captured reference).

## Immediate next actions

1. Boot the M2 into **m1n1 proxy** mode (not ChittiOS) and run
   `tools/agx-extract/capture_render.py` to produce the render replay blob.
2. Implement Layer 1 (GPU context) + the CP/3D cmdqueue channel `send`/Event poll
   in `hw.rs`, reusing `uat.rs` and the `gpu_read/gpu_write` helpers.
3. Port the `WorkCommand`/microsequence encoders (pure, tested) from
   `cmdqueue.py`/`microsequence.py`.

## Open questions to resolve during capture

- Exact `ctx_id`/TTBR bind for a *fresh* context vs. the boot context we replay.
- Whether the built-in clear/store pipeline binaries live in the replayed initdata
  region or must be provided (affects whether render-only needs Layer 3 at all).
- The Event-channel completion encoding for a non-render (compute) job.
