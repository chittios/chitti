# AGX compute dispatch encoding — porting reference (from Mesa source)

Extracted from Mesa `src/asahi` (`genxml/cmdbuf.xml`, `libagx/libagx_dgc.h`,
`lib/agx_usc.h`, `gallium/drivers/asahi/agx_state.c`) for **G14G = M2 t8112**
(our Mac mini — NOT G14X, which is M2 Pro/Max). This is the Layer-3 "dispatch ABI"
that the Asahi *kernel* docs declare out of scope; it exists only in Mesa. With
this + the m1n1 `WorkCommandCP`/microsequence structs (Layer 2) + a shader binary
(Layer 3a), a compute dispatch is fully specified.

## The three linked objects

A compute launch is three GPU-memory objects, chained by pointers:

1. **Shader ISA** — the compiled AGX kernel machine code (Layer 3a; hand-write +
   assemble a matvec, or compile with Mesa). Its address goes in `USC Shader.Code`.
2. **USC words** — a small tagged byte stream configuring the launch (binds the
   shader address + argument buffer + register counts). Its address (>>6) goes in
   `CDM Launch Word 1.Pipeline`.
3. **CDM command stream** (the "encoder"/"command list", `WorkCommandCP.encoder`)
   — the actual dispatch: launch + grid/workgroup sizes + barrier.

The `WorkCommandCP` (Layer 2, m1n1 `cmdqueue.py`) wraps these:
`compute_info.encoder = <CDM stream addr>`, `compute_info.pipeline_base =
0x1100000000`, and the microsequence `StartComputeCmd` points at it.

## 1. CDM command stream (the encoder) — minimal launch

From `agx_cdm_launch()` (`libagx_dgc.h`). For **G14G, direct (non-indirect)**
dispatch, the minimal stream is 4 commands, then a barrier (`agx_state.c:3286`):

```
CDM_LAUNCH_WORD_0   (4 bytes)
CDM_LAUNCH_WORD_1   (4 bytes)
CDM_GLOBAL_SIZE     (12 bytes)   # total threads = workgroups * local
CDM_LOCAL_SIZE      (12 bytes)   # threads per workgroup
CDM_BARRIER         (4 bytes)
```

(G14X only: a `CDM_UNK_G14X` between WORD_1 and GLOBAL_SIZE — we skip it.)

### CDM_LAUNCH_WORD_0 (u32) — bit fields (`cmdbuf.xml`)
```
[3:1]   Uniform register count      groups(64)  # = ceil(uniform_regs/64)
[8:4]   Texture state register count groups(8)   # 0 if no textures
[11:9]  Sampler state register count
[15:12] Preshader register count    groups(16)   # 0 (no preshader)
[28:27] Mode                        (CDM Mode: Direct=0)
[31:29] Block Type                  (=Launch=0)
```
NB Mesa merges the caller's `launch` (which carries the register counts from the
compiled shader) with `mode`. So WORD_0 = per-shader reg counts | (mode<<27) |
(Launch<<29).

### CDM_LAUNCH_WORD_1 (u32)
```
[31:6]  Pipeline   address shr(6)   # = USC-words GPU addr >> 6 (USC is 64B-aligned)
```
"Pipeline" here is the USC-words address, *relative to `pipeline_base`
0x1100000000* — i.e. the low bits; Mesa's `agx_usc_addr(dev, gpu)` computes it.

### CDM_GLOBAL_SIZE / CDM_LOCAL_SIZE (3×u32 each)
```
X @ word0, Y @ word1, Z @ word2   (all u32)
```
GLOBAL = total threads (num_workgroups*local, confirmed `hk_cmd_dispatch.c:114`);
LOCAL = workgroup size.

### CDM_BARRIER (u32)
Mostly unknown bools; the meaningful ones: `[3] USC cache inval`, `[27] Returns`,
`[31:29] Block Type = Barrier(3)`. Mesa emits a barrier after each dispatch.

### CDM Mode enum: Direct=0, Indirect global=1, Indirect local=2.
### CDM Block Type enum: Launch=0, Stream Link=1, Stream Terminate=2, Barrier=3, Stream Return=4.

## 2. USC words — bind shader + args

From `agx_build_pipeline()` (`agx_state.c:2873`) + `agx_usc.h`. Assembly order
for a **compute** shader (skip texture/sampler/preshader when unused):

```
[USC Uniform]*   # one per push-constant range: binds an arg buffer to uniform regs
 USC Shared      # threadgroup/shared-memory config
 USC Shader      # the shader code address
 USC Registers   # GPR count + spill
 USC No Preshader
```

Tags (`USC Control` enum, byte 0 of each word):
Shader=0x0d, Uniform=0x1d, Uniform-high=0x3d, Shared=0x4d, Registers=0x8d,
No-preshader=0x88, Preshader=0x38, Sampler=0x9d, Texture=0xdd.

### USC Shader (6 bytes)
```
[7:0]   Tag=0x0d
[8]     Loads varyings (0 for compute)
[15:10] Unk (0)
[47:16] Code = shader ISA GPU address (u32 low bits; the code is in the pipeline/
        GEM region)
```

### USC Uniform (8 bytes) — the argument-passing mechanism
```
[7:0]    Tag=0x1d
[15:8]   Start (halfs)   # first uniform register (in 16-bit halves)
[21:16]  Size (halfs)    # number of 16-bit halves to load (<=64 per word)
[63:26]  Buffer = arg-buffer GPU addr, shr(2)
```
This DMAs `Size` halfwords from `Buffer` into uniform registers starting at
`Start`. **This is how kernel args (input/output pointers, dims) reach the shader**
— upload an arg buffer, bind it here; the kernel reads uniforms `u[Start..]`.
Since we hand-write the shader, WE define this ABI (what's in the buffer + which
uniforms the kernel reads).

### USC Registers (4 bytes)
```
[7:0]   Tag=0x8d
[12:8]  Register count  groups(8)   # ceil(GPRs/8), from the compiled shader
[13]    Unk
[21:18] Spill size
```

### USC Shared (4 bytes) — for no shared mem: `agx_usc_shared_none`
```
Tag=0x4d, layout=VERTEX_COMPUTE, bytes_per_threadgroup=65536, uses_shared=0
```

### USC No Preshader (2 bytes): Tag=0x88.

## 3. What's still needed to fire one dispatch

- **Shader ISA (3a):** a matvec kernel in AGX ISA. Hand-write (applegpu assembler,
  adapt G13→G14) or compile with Mesa; get its GPR count + uniform-reg usage
  (feeds USC Registers + LAUNCH_WORD_0 counts).
- **Arg buffer layout:** our choice — e.g. {in_ptr:u64, out_ptr:u64, n:u32}. Upload
  it, bind via USC Uniform.
- **WorkCommandCP wrapper (Layer 2):** `compute_info.encoder = CDM-stream addr`,
  `pipeline_base = 0x1100000000`, EventControl, microsequence Start/Finalize
  Compute (m1n1 structs; drm/asahi `queue/compute.rs` confirms the field wiring:
  `job_params1.encoder = cmdbuf.encoder_ptr`, `.pipeline_base = usc_base`).
- **Submit:** write the WorkCommandCP ptr into the CP cmdqueue ring, send
  `RunCmdQueueMsg{queue_type=2 (Compute), cmdqueue_addr, head, event_number,
  new_queue=1}` on CP channel_id `(q<<2)|2`, doorbell; await the Event stamp.

## Porting notes

- All CDM/USC structs are small fixed byte layouts → pure Rust encoders with
  `#[test_case]` bit-layout tests (the `uat.rs` pattern). Genxml `agx_pack` is just
  bitfield packing; no magic.
- `pipeline_base` (0x1100000000) is the base the USC `Pipeline` field and shader
  `Code` are relative to — keep shader + USC in the context's pipeline region.
- Verify `grid.count` = total threads (not workgroups) — confirmed here, but
  re-check when a real dispatch runs.
- Source snapshot: Mesa `main` (sparse clone), 2026-07. Re-pull if fields drift.
