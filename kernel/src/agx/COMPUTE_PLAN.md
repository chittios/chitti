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
  the *render* path, uncommented). So there is no doc for the compute USC.
  **Correction (2026-08): reference bytes for the compute *submission* do exist**
  — a full `WorkCommandCP` hexdump sits in `fw/agx/cmdqueue.py`'s own docstring
  (see the encoding audit below), which settles Layer 2 field-for-field. It does
  **not** contain the USC/register array, so 3b below still stands for the
  dispatch descriptors.
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

## Submission-encoding audit (2026-08) — the reference was in the tree all along

The first dispatch attempt bound the context, placed every object and rang the
doorbell, and the firmware **never drained the CP channel** (`CL_0` read pointer
stayed 0). Re-deriving the whole submission from the **vendored** proxyclient —
`third_party/m1n1/proxyclient/m1n1/{agx,fw/agx}/*.py` — instead of from
recollection of drm/asahi found five encoding faults. Four are settled by source
or by real bytes; one is still a two-way guess and is logged as such at run time.

The decisive artefact is the **captured `WorkCommandCP` hexdump in `cmdqueue.py`'s
own docstring** — real bytes from a live macOS compute submission, i.e. exactly the
"no capturable reference bytes exist for compute" gap this doc claimed above. Every
field decodes to something meaningful: `encoder = 0x15_00078000`,
`encoder_end = 0x15_00078024` (**a 0x24-byte command list — the same length
`cdm.rs` emits**), `pipeline_base = 0x11_00000000`, `unk_38 = 0x8c60`,
`microsequence_ptr = 0xffffffa0_0c311cc0`, `stamp_addr = 0xffffffa0_000c8014`.

1. **The bookkeeping objects were in the wrong address space.** The queue, its ring
   and cursors, the microsequence, the work command, the EventControl and the stamps
   all lived in the submitting context's TTBR0 (`0x15…`). The capture shows every one
   of those as a **kernel `0xffffffa0…` TTBR1** VA, and the proxyclient allocates
   them from `kobj`/`kshared`/`cmdbuf` (all kernel) while only the encoder, shader
   and USC come from the context (`gobj`/`pobj`). TTBR0 is per-context and **not
   active in the firmware's own boot context**, so the firmware could not read the
   queue it was being pointed at. This is the same rule the RTKit crashlog buffer had
   to learn, and it is the most likely reason nothing was consumed.
2. **`JobMeta` is 0x2c, not 0x24, and both stamps were null.** m1n1 types
   `stamp_addr`/`fw_stamp_addr` as `WrappedPointer` = `Int64ul`, and the capture has
   two kernel VAs there (read as `u32`s, the second decodes as `0xffffffa0` — plainly
   not a pointer). The 0x24 form shifted every field past `JobMeta` by 8 *and* left
   the stamps at zero, where drm/asahi types them `NonZeroU64`. This **resolves the
   "KNOWN UNRESOLVED AMBIGUITY"** `workcmd.rs` used to carry: it was a conflict
   between drm/asahi's stale `unk_2d4` field *name* and its stamp *type*, and the
   bytes side with the type. Corroboration: the corrected total is 0x320, and the
   reference driver allocates work commands with `align = 0x20`.
3. **`CommandQueueInfo` was 0x18 bytes short in the middle.** `unk_34`, `unk_38`,
   `unk_40`, `unk_44` and `prio5` were missing, so `uuid` sat at 0x38 instead of 0x50
   and **`gpu_context_addr` — the scheduler block — at 0x8c instead of 0xa4**, with a
   total of 0xa0 instead of 0xb8. Also: `event_id`/`unk_4c`/`unk_54` default to **-1**
   in the reference, and 0 is a legal event id, so zero was the wrong "unset".
4. **`WaitForIdle` named no pipe.** The header is `0x01 | (pipe << 8)`; the
   proxyclient's own `WaitForInterruptCmd(1,0,0)` / `(0,1,0)` calls pin
   `Vertex = 1<<0` and `Fragment = 1<<8`. A bare `0x01` waits on nothing.
5. **Still a guess: `Pipe::Compute`.** Encoded as `1 << 15` (giving header
   `0x0080_0001`) from recollection of drm/asahi; the plausible alternative puts the
   0x80 in byte 3 (`0x8000_0001`), which is what m1n1's "`TimestampCmd.unk_3` —
   sometimes 0x80" annotation would suggest. `/agx compute` logs the header word it
   used, and this is the **first thing to flip** if the queue ring is read but the job
   never completes.

Smaller fixes from the same pass: `ComputeInfo.unk_38 = 0x8c60` ("always", and so in
the capture), `unk_58 = 1`, `iogpu_unk_40 = 0x1c`,
`EncoderParams.iogpu_compute_unk44 = 0xffffffff`, `ring_state` is 0x60 not 0x70 with
`rb_size` 0x500 (the reference default) rather than an invented 0x80,
`EventControl.submission_id` is a submission counter and starts at **0** (it was
seeded with the context id), the `gpu_buf` is sized to its documented 0x2c18, and
`StartComputeCmd.unk_buf_addr` / `unk_28 = 1` are populated.

**None of this is hardware-verified** — it is source- and capture-verified, pinned by
`cargo xtask test` on both arches. What a dispatch now reports, in firmware order, so
one boot separates the remaining causes instead of one bit: whether the channel
message was consumed, whether the firmware is alive (Stats), whether it read the
queue ring (`gpu_rptr`), whether either stamp/`event_count` moved, the **decoded**
Event-channel messages (Flag = completed vs Fault/Timeout/ChannelError = rejected),
and the firmware's own log text.

Known-remaining gap in the submission itself: `StartComputeCmd.stats_ptr` is 0. The
real driver points it at the `GpuStatsComp` region *inside* the initdata we replay,
so resolving it means locating that offset in the captured blob.

## Open questions to resolve during capture

- Exact `ctx_id`/TTBR bind for a *fresh* context vs. the boot context we replay.
- Whether the built-in clear/store pipeline binaries live in the replayed initdata
  region or must be provided (affects whether render-only needs Layer 3 at all).
- The Event-channel completion encoding for a non-render (compute) job.
