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

## De-risking sequence (do the cheap decisive thing first)

The one self-contained, **capturable** real submission in the reference is
`experiments/agx_1tri.py`: it builds a full triangle render (WorkCommandTA +
WorkCommand3D + InitBM + microsequence) **entirely in code** — no Metal capture
file — using the built-in clear/store pipelines. So we can get exact known-good
bytes for the *plumbing* without solving Layer 3.

1. **Capture (proxy):** run `agx_1tri.py` under the m1n1 proxy on the M2, dump
   every GPU object it creates (addr, size, bytes), the two `WorkCommand`s, the
   microsequences, the cmdqueue `RunCmdQueueMsg`, and the context TTBR/`ctx_id`.
   Script: `tools/agx-extract/capture_render.py` (companion to this doc) → a replay
   blob in the same shape as `initdata_blob.rs`.
2. **Port plumbing (ChittiOS):** implement Layer 1 (context) + Layer 2 (cmdqueue
   channel + WorkCommand + microsequence encode + Event poll). Replay the captured
   render at identical VAs, submit on the 3D/TA cmdqueue channels, **wait for the
   Event stamp**. Success = the firmware completes the render and bumps the event
   counter (and ideally the framebuffer shows the triangle). This proves context +
   cmdqueue + microsequence + completion — the whole plumbing — with zero shader
   work of our own.
3. **Swap in compute:** replace the render WorkCommand with `WorkCommandCP` +
   compute microsequence, and drop in the Layer-3 GEMM shader/command-list. Submit
   on the CP channel, read back the result buffer.
4. **Wire into cortex:** upload weights to GPU memory once at `Model::load`; per
   matmul, encode a `WorkCommandCP` dispatch over `matvec_qw`/`batched_proj`; only
   small activations cross per token. Validate greedy parity vs `tools/cortexdiff`
   / `cargo xtask ref-check` (rel-RMS tolerance — GPU reduction order ≠ NEON).

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
