//! **AGX WorkCommandCP + cmdqueue message** — Layer 2 of compute submission.
//!
//! The compute *work command* (m1n1 `WorkCommandCP`, drm/asahi `RunCompute`) that
//! wraps a CDM command stream (from [`super::cdm`]) into a firmware job, its
//! microsequence, and the `CommandQueueInfo`/`RunCmdQueueMsg` pair that kicks a
//! cmdqueue channel. For **G14G / M2 t8112, firmware V13.5** — so the version gates
//! `V >= V13_0B4` and `V >= V13_3` are true and `G >= G14X` is false.
//!
//! **The reference is the vendored proxyclient**, `third_party/m1n1/proxyclient/
//! m1n1/fw/agx/{cmdqueue,microsequence,channels}.py`, cross-checked field-for-field
//! against the **captured `WorkCommandCP` hexdump in `cmdqueue.py`'s own
//! docstring** — real bytes from a live macOS compute submission, which is the only
//! ground truth available for a compute job and is what settles the offsets below.
//! Decoding it (V < V13_0B4 layout, so 8 bytes earlier than ours) gives a
//! meaningful value at every field: `encoder = 0x15_00078000`,
//! `encoder_end = 0x15_00078024` (a 0x24-byte command stream — exactly what
//! [`super::cdm`] emits), `pipeline_base = 0x11_00000000`, `unk_38 = 0x8c60`,
//! `microsequence_ptr = 0xffffffa0_0c311cc0`, `stamp_addr = 0xffffffa0_000c8014`,
//! `fw_stamp_addr = 0xffffffa0_0c378014`, `uuid = 0x120022b8`.
//!
//! Two things that dump settles, both of which were wrong here before:
//!
//! 1. **`JobMeta` is 0x2c, not 0x24** — its `stamp_addr`/`fw_stamp_addr` are
//!    **8-byte** pointers (m1n1 `WrappedPointer` = `Int64ul`), and the capture shows
//!    two kernel VAs there. Read as `u32`s the second one decodes as `0xffffffa0`,
//!    which is obviously not a stamp pointer. The old 0x24 form shifted every field
//!    from `JobMeta` on by 8 *and* left both stamps null — and a stamp pointer is
//!    where the firmware writes job completion, so null is not a benign default.
//!    (This resolves the "KNOWN UNRESOLVED AMBIGUITY" this module used to document:
//!    it was a conflict between drm/asahi's stale `unk_2d4` field *name* and its
//!    `NonZeroU64` stamp *type*, and the bytes side with the type.)
//! 2. **Every bookkeeping object is a kernel (TTBR1) VA, only the encoder is a
//!    context (TTBR0) VA.** `microsequence_ptr`, both stamps and the timestamp
//!    pointers are all `0xffffffa0…` while `encoder`/`deflake` are `0x15…`. See
//!    `hw.rs`'s `kern_reserve` — placing the queue and microsequence in the
//!    submitting context's TTBR0 makes them unreachable from the firmware's own
//!    boot context, which is the same rule the RTKit crashlog buffer had to learn.
//!
//! Pure + arch-neutral, so the layouts are pinned by `cargo xtask test` on both
//! arches rather than only by a hardware dispatch.

#![allow(dead_code)] // consumed by the hw.rs submission wiring

use alloc::vec::Vec;

/// Little-endian byte builder for firmware structs (all AGX structs are packed,
/// no natural alignment — construct/`#[repr(C)]` with `U64`/`U32` newtypes).
pub struct Buf {
    pub bytes: Vec<u8>,
}

impl Buf {
    pub fn new() -> Self {
        Self { bytes: Vec::new() }
    }
    pub fn len(&self) -> usize {
        self.bytes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
    pub fn u8(&mut self, v: u8) -> &mut Self {
        self.bytes.push(v);
        self
    }
    pub fn u16(&mut self, v: u16) -> &mut Self {
        self.bytes.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub fn u32(&mut self, v: u32) -> &mut Self {
        self.bytes.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub fn i32(&mut self, v: i32) -> &mut Self {
        self.bytes.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub fn u64(&mut self, v: u64) -> &mut Self {
        self.bytes.extend_from_slice(&v.to_le_bytes());
        self
    }
    /// N zero bytes (`Array<N,u8>` / `Pad<N>` / construct `ZPadding`).
    pub fn pad(&mut self, n: usize) -> &mut Self {
        self.bytes.resize(self.bytes.len() + n, 0);
        self
    }
}

impl Default for Buf {
    fn default() -> Self {
        Self::new()
    }
}

/// `CmdBufWork.cmdid` for a compute work command (m1n1 `WorkCommandCP.magic`).
pub const CMD_TYPE_RUN_COMPUTE: u32 = 3;

/// Total encoded size of [`run_compute`] (G14G, V13.5). 0x20-aligned, which is the
/// alignment the proxyclient allocates work commands at (`new(..., align=0x20)`).
pub const RUN_COMPUTE_SIZE: usize = 0x320;

/// `ComputeInfo.unk_38` — "always 0x8c60" per the proxyclient, and 0x8c60 in the
/// captured dump. Left at 0 before, which is not a value the firmware ever sees
/// from the real driver.
const COMPUTE_INFO_UNK_38: u64 = 0x8c60;
/// `ComputeInfo.unk_58` / `iogpu_unk_40` — 1 / 0x1c in the capture.
const COMPUTE_INFO_UNK_58: u32 = 1;
const COMPUTE_INFO_IOGPU_UNK_40: u32 = 0x1c;
/// `EncoderParams.iogpu_compute_unk44` — all-ones in the capture.
const ENCODER_IOGPU_UNK44: u32 = 0xffff_ffff;
/// `EventControl.unk_10` (proxyclient `GPURenderer`: `event_control.unk_10 = 0x50`).
const EVENT_CONTROL_UNK_10: u32 = 0x50;

/// Parameters a caller supplies to build a compute WorkCommand. The many
/// reverse-engineered `unk_*` fields the real driver leaves zero stay zero; the
/// ones the capture shows a fixed value for are set from the constants above.
#[derive(Clone, Copy, Default)]
pub struct ComputeCmd {
    /// `WorkCommandCP.counter` (V >= V13_0B4).
    pub counter: u64,
    /// `context_id` — the GPU context (ASID) this job's encoder belongs to.
    pub vm_slot: u32,
    /// `event_control_addr` — the `EventControl` (drm/asahi `Notifier`) the
    /// firmware posts completion through. **Kernel VA.**
    pub event_control: u64,
    /// The CDM command stream GPU address (`ComputeInfo.encoder`). **Context VA.**
    pub encoder: u64,
    /// The context pipeline base (0x11_00000000) — `ComputeInfo.pipeline_base`.
    pub pipeline_base: u64,
    /// End of the CDM stream (`ComputeInfo2.encoder_end`). **Context VA.**
    pub encoder_end: u64,
    pub encoder_id: u32,
    /// The microsequence GPU address + size. **Kernel VA.**
    pub microsequence: u64,
    pub microsequence_size: u32,
    pub uuid: u32,
    /// `JobMeta.stamp_addr` / `.fw_stamp_addr` — two `StampCounter`s (u32) the
    /// firmware writes `stamp_value` into on completion. **Kernel VAs, non-null:**
    /// drm/asahi types both as `NonZeroU64`, so zero is not a legal encoding.
    pub stamp: u64,
    pub fw_stamp: u64,
    pub stamp_value: u32,
    pub stamp_slot: u32,
    /// `JobMeta.queue_cmd_count` — how many commands this queue has submitted.
    pub queue_cmd_count: u32,
    pub client_sequence: u8,
}

/// `ComputeInfo` (microsequence.py:708) — 0x150 bytes. The proxyclient's own note:
/// "only the cmdlist and pipelinebase … are strictly needed to launch a basic
/// compute shader", so the deflake/helper buffers stay null.
fn compute_info(b: &mut Buf, c: &ComputeCmd) {
    b.u64(0); // iogpu_deflake_1 (ComputeArgs)   +0x00
    b.u64(c.encoder); // encoder (CommandList)   +0x08
    b.u64(0); // iogpu_deflake_2                 +0x10
    b.u64(0); // iogpu_deflake_3                 +0x18
    b.u64(0); // iogpu_deflake_4                 +0x20
    b.u64(0); // iogpu_deflake_5                 +0x28
    b.u64(c.pipeline_base); // pipeline_base     +0x30
    b.u64(COMPUTE_INFO_UNK_38); // unk_38        +0x38
    b.u32(0); // helper_program (bit 0 = enable) +0x40
    b.u32(0); // unk_44                          +0x44
    b.u64(0); // helper_arg (work layout)        +0x48
    b.u32(0); // helper_cfg                      +0x50
    b.u32(0); // unk_54                          +0x54
    b.u32(COMPUTE_INFO_UNK_58); // unk_58        +0x58
    b.u32(0); // unk_5c                          +0x5c
    b.u32(COMPUTE_INFO_IOGPU_UNK_40); // iogpu_unk_40 +0x60
    b.pad(0xec); // unk_pad                      +0x64 → 0x150
}

/// `ComputeInfo2` (V >= V13_0B4) — 0x60 bytes; carries `encoder_end`.
fn compute_info2(b: &mut Buf, c: &ComputeCmd) {
    b.pad(4); // unk_0_0 (V >= V13_0B4)          +0x00
    b.pad(0x24); // unk_0                        +0x04
    b.u64(0); // iogpu_deflake_1                 +0x28
    b.u64(c.encoder_end); // encoder_end         +0x30
    b.pad(0x20); // unk_34                       +0x38
    b.u32(0); // unk_g14x                        +0x58
    b.u32(0); // unk_58                          +0x5c → 0x60
}

/// `EncoderParams` (microsequence.py:672) — 0x28 bytes.
fn encoder_params(b: &mut Buf, c: &ComputeCmd) {
    b.u32(0); // unk_0                           +0x00
    b.u32(0); // unk_4                           +0x04
    b.u32(0); // unk_8                           +0x08
    b.u32(c.encoder_id); // encoder_id           +0x0c
    b.u32(0); // unk_10                          +0x10
    b.u32(ENCODER_IOGPU_UNK44); // iogpu_compute_unk44 +0x14
    b.u64(0); // seq_buffer                      +0x18
    b.u64(0); // unk_1c                          +0x20 → 0x28
}

/// `JobMeta` (microsequence.py:657) — **0x2c bytes**: `stamp_addr` and
/// `fw_stamp_addr` are 8-byte `WrappedPointer`s, confirmed by the captured dump
/// (two `0xffffffa0…` kernel VAs at +0x04 / +0x0c).
fn job_meta(b: &mut Buf, c: &ComputeCmd) {
    b.u32(0); // unk_0                           +0x00
    b.u64(c.stamp); // stamp_addr                +0x04
    b.u64(c.fw_stamp); // fw_stamp_addr          +0x0c
    b.u32(c.stamp_value); // stamp_value         +0x14
    b.u32(c.stamp_slot); // stamp_slot           +0x18
    b.u32(0); // evctl_index                     +0x1c
    b.u32(0); // unk_20                          +0x20
    b.u32(c.uuid); // uuid                       +0x24
    b.u32(c.queue_cmd_count); // queue_cmd_count +0x28 → 0x2c
}

/// `TimeStampPointers` (0x10; both null → the firmware records no timestamps).
fn timestamp_pointers(b: &mut Buf) {
    b.u64(0); // start_addr
    b.u64(0); // end_addr
}

/// Build the full compute WorkCommand (m1n1 `WorkCommandCP`, G14G/V13.5),
/// [`RUN_COMPUTE_SIZE`] bytes. The result lives in the **kernel** VA range and is
/// referenced by the command queue's ring entry.
pub fn run_compute(c: &ComputeCmd) -> Vec<u8> {
    let mut b = Buf::new();
    b.u32(CMD_TYPE_RUN_COMPUTE); // magic                  @0x000
    b.u64(c.counter); // counter (V >= V13_0B4)             @0x004
    b.u32(0); // unk_4                                      @0x00c
    b.u32(c.vm_slot); // context_id                         @0x010
    b.u64(c.event_control); // event_control_addr           @0x014
    b.u32(0); // unk_2c                                     @0x01c
    b.pad(0x50); // unk_buf (G < G14X)                      @0x020
    compute_info(&mut b, c); // compute_info                @0x070
    b.u64(0); // registers_addr (G14X only; null here)      @0x1c0
    b.u16(0); // register_count                             @0x1c8
    b.u16(0); // registers_length                           @0x1ca
    b.pad(0x24); // unk_pad                                 @0x1cc
    b.u64(c.microsequence); // microsequence_ptr            @0x1f0
    b.u32(c.microsequence_size); // microsequence_size      @0x1f8
    compute_info2(&mut b, c); // compute_info2              @0x1fc
    encoder_params(&mut b, c); // encoder_params            @0x25c
    job_meta(&mut b, c); // job_meta                        @0x284
    b.u64(0); // ts1 (command_time)                         @0x2b0
    timestamp_pointers(&mut b); // ts_pointers              @0x2b8
    timestamp_pointers(&mut b); // user_ts_pointers         @0x2c8
    b.u8(c.client_sequence); // client_sequence             @0x2d8
    b.u64(0); // unk_ts2 (V >= V13_0B4)                     @0x2d9
    b.u64(0); // unk_ts (V >= V13_0B4)                      @0x2e1
    b.pad(0x1c); // unk_2e1 (V >= V13_0B4)                  @0x2e9
    b.u32(0); // unk_flag (V >= V13_0B4)                    @0x305
    b.pad(0x10); // unk_pad (V >= V13_0B4)                  @0x309
    b.pad(0x7); // pad_2d9                                  @0x319 → 0x320
    b.bytes
}

/// Byte offsets inside [`run_compute`]'s output that the microsequence has to
/// point back into (`StartComputeCmd.unk_buf_addr` / `.computeinfo_addr` /
/// `.computeinfo2_addr`). Named so a caller cannot get the arithmetic wrong.
pub const WC_UNK_BUF_OFFSET: u64 = 0x020;
pub const WC_COMPUTE_INFO_OFFSET: u64 = 0x070;
pub const WC_COMPUTE_INFO2_OFFSET: u64 = 0x1fc;
/// `WorkCommandCP.unk_flag` — what `StartComputeCmd.unk_flag_addr` and
/// `FinalizeComputeCmd.unkptr_71` point at.
pub const WC_UNK_FLAG_OFFSET: u64 = 0x305;

// --- cmdqueue channel message (m1n1 `RunCmdQueueMsg`, 0x40 on G14/V13.2+) -----

/// Queue types for `RunCmdQueueMsg.queue_type`.
pub const QUEUE_TA: u32 = 0;
pub const QUEUE_3D: u32 = 1;
pub const QUEUE_COMPUTE: u32 = 2;

/// Encode the `RunCmdQueueMsg` written into a cmdqueue channel ring to submit
/// work: which queue, the `CommandQueueInfo` address, the ring head (the queue's
/// `cpu_wptr` **after** the increment), the event number, and whether the queue is
/// new. 0x40 bytes on G14/V13.2+ (0x30 of fields + 0x10 of version padding).
/// `cmdqueue_addr` sits at an unaligned +0x04 — construct packs with no alignment.
pub fn run_cmd_queue_msg(
    queue_type: u32,
    cmdqueue_addr: u64,
    head: u32,
    event_number: u32,
    new_queue: bool,
) -> Vec<u8> {
    let mut b = Buf::new();
    b.u32(queue_type); // queue_type      @0x00
    b.u64(cmdqueue_addr); // cmdqueue_addr @0x04 (unaligned)
    b.u32(head); // head                   @0x0c
    b.u32(event_number); // event_number   @0x10
    b.u32(new_queue as u32); // new_queue   @0x14
    b.u64(0); // timestamp                  @0x18
    b.pad(0x10); // data                    @0x20
    b.pad(0x10); // ZPadding (G14/V13.2)    @0x30
    b.bytes // total 0x40
}

// ===================== microsequence (fw/agx/microsequence.py) =====================
//
// The firmware executes this instruction list for the job: StartCompute (runs the
// CDM encoder) → WaitForIdle → FinalizeCompute (writes the stamp) → End.
// `restart_branch_offset` in FinalizeCompute is the relative offset back to
// StartCompute (for firmware preemption/restart).

const OP_START_COMPUTE: u32 = 0x29;
const OP_FINALIZE_COMPUTE: u32 = 0x2a;
/// `EndCmd` — `magic 0x18`, `flags 0x40` in byte 3 (m1n1 `EndCmd.__init__`), i.e.
/// the little-endian word 0x4000_0018. This is drm/asahi's `RetireStamp` header.
const OP_END: u32 = 0x4000_0018;
/// `WaitForInterruptCmd` = `magic 0x01` + a pipe selector in bytes 1..3, i.e. the
/// header is `0x01 | (pipe << 8)`. The proxyclient's own calls pin two thirds of
/// the enum: `WaitForInterruptCmd(1,0,0)` for TA and `(0,1,0)` for 3D give
/// `Vertex = 1 << 0` and `Fragment = 1 << 8`. Compute is `1 << 15` (drm/asahi's
/// `Pipe`), so the header is `0x0080_0001` — **0x80 in byte 2**.
///
/// The `1 << 15` is the one part of this taken from recollection of drm/asahi
/// rather than from the vendored tree, so it is the first thing to flip if a
/// dispatch is scheduled (queue ring read) but never completes: the plausible
/// alternative puts the 0x80 in **byte 3** (`0x8000_0001`), which is what m1n1's
/// "`TimestampCmd.unk_3` — sometimes 0x80" annotation would suggest if that field
/// is the same selector. What is *certain* is that a bare `0x01` — which this
/// emitted before — names no pipe at all and so waits on nothing.
pub const OP_WAIT_FOR_IDLE_COMPUTE: u32 = 0x0080_0001;
/// `StartComputeCmd.unk_28` — 1 in the proxyclient's annotation.
const START_COMPUTE_UNK_28: u32 = 1;
const MAX_ATTACHMENTS: usize = 16;

/// Sizes of the three microsequence ops, so a caller can pin them.
pub const START_COMPUTE_SIZE: usize = 0x16c;
pub const FINALIZE_COMPUTE_SIZE: usize = 0x7c;

/// Addresses the microsequence weaves together.
#[derive(Clone, Copy, Default)]
pub struct MicroSeqRefs {
    /// `WorkCommandCP.unk_buf` (= work-command VA + [`WC_UNK_BUF_OFFSET`]).
    pub work_cmd_unk_buf: u64,
    /// `WorkCommandCP.compute_info` (= work-command VA + [`WC_COMPUTE_INFO_OFFSET`]).
    pub compute_info: u64,
    /// `WorkCommandCP.compute_info2` (= + [`WC_COMPUTE_INFO2_OFFSET`]).
    pub compute_info2: u64,
    /// The `CommandQueueInfo` this job was submitted on.
    pub work_queue: u64,
    /// `GpuStatsComp` region inside the replayed initdata. Zero = the firmware
    /// records no per-pipe stats for this job.
    pub stats: u64,
    pub vm_slot: u32,
    pub counter1: u32,
    pub counter2: u32,
    pub uuid: u32,
    /// `FinalizeComputeCmd.stamp` — the fw stamp (same as `JobMeta.fw_stamp_addr`).
    pub fw_stamp: u64,
    pub stamp_value: u32,
    /// `unk_flag_addr` / `unkptr_71` — the `WorkCommandCP.unk_flag` field's address.
    pub unk_flag: u64,
    pub counter: u64,
    /// `EventControl.unk_buf` (= event-control VA + 0xa8).
    pub event_ctrl_buf: u64,
}

fn attachments(b: &mut Buf) {
    // Array<16, Attachment{address u64, size u32, unk_c u16, unk_e u16}> + count.
    for _ in 0..MAX_ATTACHMENTS {
        b.u64(0).u32(0).u16(0).u16(0);
    }
    b.u32(0); // num_attachments
}

/// Build the compute microsequence (StartCompute → WaitForIdle → FinalizeCompute →
/// End) for G14G/V13.5.
pub fn microseq_compute(r: &MicroSeqRefs) -> Vec<u8> {
    let mut b = Buf::new();
    // --- StartComputeCmd (magic 0x29), 0x16c bytes ---
    let start = b.len();
    b.u32(OP_START_COMPUTE); // magic                       +0x000
    b.u64(r.work_cmd_unk_buf); // unk_buf_addr              +0x004
    b.u64(r.compute_info); // computeinfo_addr              +0x00c
    b.u64(r.stats); // stats_ptr                            +0x014
    b.u64(r.work_queue); // cmdqueue_ptr                    +0x01c
    b.u32(r.vm_slot); // context_id                         +0x024
    b.u32(START_COMPUTE_UNK_28); // unk_28                  +0x028
    b.u32(r.counter1); // counter1                          +0x02c
    b.u32(r.counter2); // counter2                          +0x030
    b.u32(0); // unk_34                                     +0x034
    b.u32(0); // unk_38                                     +0x038
    b.u64(r.compute_info2); // computeinfo2_addr            +0x03c
    b.u32(0); // unk_44                                     +0x044
    b.u32(r.uuid); // uuid                                  +0x048
    attachments(&mut b); // attachments + num_attachments   +0x04c
    b.pad(4); // padding                                    +0x150
    b.u64(r.unk_flag); // unk_flag_addr (V >= V13_0B4)      +0x154
    b.u64(r.counter); // counter (V >= V13_0B4)             +0x15c
    b.u64(r.event_ctrl_buf); // event_ctrl_buf_addr         +0x164 → 0x16c

    // --- WaitForInterruptCmd (magic 0x01), compute pipe ---
    b.u32(OP_WAIT_FOR_IDLE_COMPUTE);

    // --- FinalizeComputeCmd (magic 0x2a), 0x7c bytes ---
    let finalize = b.len();
    b.u32(OP_FINALIZE_COMPUTE); // magic                    +0x00
    b.u64(r.stats); // unkptr_4 (= StartCompute.stats_ptr)   +0x04
    b.u64(r.work_queue); // cmdqueue_ptr                    +0x0c
    b.u32(r.vm_slot); // context_id                         +0x14
    b.u64(r.compute_info2); // computeinfo2_addr            +0x18
    b.u32(0); // unk_24                                     +0x20
    b.u32(r.uuid); // uuid                                  +0x24
    b.u64(r.fw_stamp); // stamp                             +0x28
    b.u32(r.stamp_value); // stamp_value                    +0x30
    b.pad(0x24); // unk_38 .. unk_58 (nine u32)             +0x34
    // restart_branch_offset: signed, relative from FinalizeCompute to StartCompute.
    b.i32((start as i64 - finalize as i64) as i32); //      +0x58
    b.u32(0); // unk_60                                     +0x5c
    b.pad(0xd); // unk_64 (V >= V13_0B4)                    +0x60
    b.u64(r.unk_flag); // unkptr_71 (V >= V13_0B4)          +0x6d
    b.pad(0x7); // pad_79 (V >= V13_0B4)                    +0x75 → 0x7c

    // --- EndCmd ---
    b.u32(OP_END);
    b.bytes
}

// ===================== command queue (fw/agx/cmdqueue.py) =====================

/// Ring entries a command queue's ring buffer holds (`CommandQueuePointers
/// .rb_size` default). The ring is `8 * RB_ENTRIES` bytes of WorkCommand pointers.
pub const RB_ENTRIES: u32 = 0x500;

/// `CommandQueuePointers` (0x60) — the queue's ring cursors, one 4-byte field per
/// 0x10 so each lands in its own cache line. Field offsets are `CommandQueue
/// PointerMap`: `GPU_DONEPTR` 0x00, `GPU_RPTR` 0x30, `CPU_WPTR` 0x40.
pub fn ring_state(cpu_wptr: u32, rb_entries: u32) -> Vec<u8> {
    let mut b = Buf::new();
    b.u32(0).pad(0xc); // gpu_doneptr        @0x00
    b.u32(0).pad(0xc); // unk_10             @0x10
    b.u32(0).pad(0xc); // unk_20             @0x20
    b.u32(0).pad(0xc); // gpu_rptr           @0x30
    b.u32(cpu_wptr).pad(0xc); // cpu_wptr    @0x40
    b.u32(rb_entries).pad(0xc); // rb_size   @0x50
    b.bytes // 0x60
}

/// `CommandQueuePointerMap` offsets, for reading the firmware's side back.
pub const RS_GPU_DONEPTR: u64 = 0x00;
pub const RS_GPU_RPTR: u64 = 0x30;
pub const RS_CPU_WPTR: u64 = 0x40;
/// Encoded size of [`ring_state`].
pub const RING_STATE_SIZE: usize = 0x60;

/// Pointers the `CommandQueueInfo` needs — **all kernel (TTBR1) VAs**.
#[derive(Clone, Copy, Default)]
pub struct QueueRefs {
    /// `pointers_addr` — the [`ring_state`] block.
    pub state: u64,
    /// `rb_addr` — the ring of u64 WorkCommand pointers.
    pub ring: u64,
    /// `job_list_addr` — a `JobList` (drm/asahi calls it the notifier list).
    pub job_list: u64,
    /// `gpu_buf_addr` — 0x2c18 bytes of firmware scratch for this queue.
    pub gpu_buf: u64,
    /// `gpu_context_addr` — the `GPUContextData` scheduler block.
    pub gpu_context: u64,
    pub uuid: u32,
    /// `event_id` — **-1 until the firmware assigns one** (the proxyclient's
    /// default). Zero is a valid event id, so it is the wrong "unset".
    pub event_id: i32,
    pub priority: u32,
}

/// Encoded size of [`queue_info`] (V >= V13_2, G < G14X).
pub const QUEUE_INFO_SIZE: usize = 0xb8;
/// `CommandQueueInfo.gpu_context_addr`'s offset — the field an 0x18-byte hole in
/// the middle of this struct used to land 0x18 short of.
pub const QI_GPU_CONTEXT_OFFSET: usize = 0xa4;

/// `CommandQueueInfo` (cmdqueue.py:505, V >= V13_2 && G < G14X) — 0xb8 bytes. The
/// firmware reads this when a `RunCmdQueueMsg` names it: `pointers` holds the
/// cursors, `rb` the WorkCommand pointers, `gpu_context` the scheduler block.
pub fn queue_info(q: &QueueRefs) -> Vec<u8> {
    let mut b = Buf::new();
    b.u64(q.state); // pointers_addr        @0x00
    b.u64(q.ring); // rb_addr               @0x08
    b.u64(q.job_list); // job_list_addr     @0x10
    b.u64(q.gpu_buf); // gpu_buf_addr       @0x18
    b.u32(0); // gpu_rptr1                  @0x20
    b.u32(0); // gpu_rptr2                  @0x24
    b.u32(0); // gpu_rptr3                  @0x28
    b.i32(q.event_id); // event_id          @0x2c
    b.u32(q.priority); // priority          @0x30
    b.u32(0); // unk_34                     @0x34
    b.u64(0); // unk_38                     @0x38
    b.u32(0); // unk_40                     @0x40
    b.u32(0); // unk_44                     @0x44
    b.u32(0); // prio5                      @0x48
    b.i32(-1); // unk_4c                    @0x4c
    b.u32(q.uuid); // uuid                  @0x50
    b.i32(-1); // unk_54                    @0x54
    b.u64(0); // unk_58                     @0x58
    b.u32(0); // busy                       @0x60
    b.pad(0x1c); // pad1                    @0x64
    b.u32(0); // unk_80                     @0x80
    b.u32(0); // has_commands               @0x84
    b.u32(0); // unk_88                     @0x88
    b.u32(0); // unk_8c                     @0x8c
    b.u32(0); // unk_90                     @0x90
    b.u32(0); // unk_94                     @0x94
    b.u32(0); // inflight_commands          @0x98
    b.u32(0); // unk_9c                     @0x9c
    b.u32(0); // unk_a0_0 (V >= V13_2)      @0xa0
    b.u64(q.gpu_context); // gpu_context_addr @0xa4
    b.u64(0); // unk_a8                     @0xac
    b.u32(0); // unk_b0 (V >= V13_2)        @0xb4 → 0xb8
    b.bytes
}

/// Bytes of firmware scratch a `CommandQueueInfo.gpu_buf_addr` must point at
/// (proxyclient: `agx.kobj.buf(0x2c18, "GPUWorkQueue.gpu_buf")`).
pub const GPU_BUF_SIZE: u64 = 0x2c18;

// ===================== event / notifier / context =====================

/// `JobList` (cmdqueue.py:435, 0x18) — an empty list has **itself** as
/// `last_head`, so `self_va` must be the object's own GPU VA.
pub fn job_list(self_va: u64) -> Vec<u8> {
    let mut b = Buf::new();
    b.u64(0); // first_job
    b.u64(self_va); // last_head = self
    b.u64(0); // unkptr_10
    b.bytes // 0x18
}

/// `event_count` — the u32 counter `EventControl.event_count_addr` points at
/// (proxyclient: `agx.kobj.new(Int32ul, "event_count")`, value 0).
pub fn event_count() -> Vec<u8> {
    alloc::vec![0u8; 4]
}

/// A `StampCounter` (u32) pre-set to `value`, as the proxyclient initialises both
/// of a queue's stamps to the job's stamp value before the first submission.
pub fn stamp_counter(value: u32) -> Vec<u8> {
    value.to_le_bytes().to_vec()
}

/// Encoded size of [`event_control`]; its trailing `unk_buf` is at 0xa8.
pub const EVENT_CONTROL_SIZE: usize = 0xb0;
pub const EC_UNK_BUF_OFFSET: u64 = 0xa8;

/// `EventControl` (cmdqueue.py:91, V >= V13_0B4 → 0xb0). Everything past
/// `unk_10` is firmware-managed state and starts zero, except the trailing
/// `unk_buf`, which is all-ones.
///
/// `submission_id` is **0** for a fresh queue (proxyclient
/// `event_control.submission_id = 0`) — it is a submission counter, not an
/// identity, so seeding it with the context id was wrong.
pub fn event_control(event_count_va: u64) -> Vec<u8> {
    let mut b = Buf::new();
    b.u64(event_count_va); // event_count_addr  @0x00
    b.u32(0); // submission_id                  @0x08
    b.u32(0); // cur_count                      @0x0c
    b.u32(EVENT_CONTROL_UNK_10); // unk_10      @0x10
    b.pad(EVENT_CONTROL_SIZE - 0x14 - 8); // firmware-managed state @0x14
    b.pad(8); // unk_buf                        @0xa8
    let n = b.bytes.len();
    for x in &mut b.bytes[n - 8..] {
        *x = 0xff;
    }
    b.bytes // 0xb0
}

/// `GPUContextData` (cmdqueue.py:442, 0x40) with the proxyclient's default field
/// values (`queue_table_index`/`pid_table_index` = 0xff — the firmware assigns
/// them — `unk_5` = 1, `unk_1e` = 0xff, `unk_23` = 2).
pub fn gpu_context_data() -> Vec<u8> {
    let mut b = alloc::vec![0u8; 0x40];
    b[0x00] = 0xff; // queue_table_index
    b[0x01] = 0xff; // pid_table_index
    b[0x05] = 1; // unk_5
    b[0x1e] = 0xff; // unk_1e
    b[0x23] = 2; // unk_23
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c() -> ComputeCmd {
        ComputeCmd {
            counter: 1,
            vm_slot: 3,
            event_control: 0xffff_ffa0_0c3d_80c0,
            encoder: 0x15_0007_8000,
            pipeline_base: 0x11_0000_0000,
            encoder_end: 0x15_0007_8024,
            encoder_id: 0x1100_22b3,
            microsequence: 0xffff_ffa0_0c31_1cc0,
            microsequence_size: 0x240,
            uuid: 0x1200_22b8,
            stamp: 0xffff_ffa0_000c_8014,
            fw_stamp: 0xffff_ffa0_0c37_8014,
            stamp_value: 0x3b00,
            stamp_slot: 5,
            queue_cmd_count: 0,
            client_sequence: 0x15,
        }
    }

    fn rd64(b: &[u8], off: usize) -> u64 {
        u64::from_le_bytes(b[off..off + 8].try_into().unwrap())
    }
    fn rd32(b: &[u8], off: usize) -> u32 {
        u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
    }

    /// Every offset here is the captured `WorkCommandCP` dump's offset **plus 8**
    /// — the dump is the `V < V13_0B4` layout, ours has the extra `counter` after
    /// `magic`. Checking them as a set is what catches a size error in any one
    /// sub-struct, since a wrong size shifts everything downstream.
    #[test_case]
    fn run_compute_matches_the_captured_layout() {
        let w = run_compute(&c());
        assert_eq!(w.len(), RUN_COMPUTE_SIZE, "WorkCommandCP total size");
        assert_eq!(rd32(&w, 0x000), CMD_TYPE_RUN_COMPUTE);
        assert_eq!(rd64(&w, 0x004), 1); // counter
        assert_eq!(rd32(&w, 0x010), 3); // context_id
        assert_eq!(rd64(&w, 0x014), 0xffff_ffa0_0c3d_80c0); // event_control_addr
        // ComputeInfo @0x70 (dump 0x68): encoder +0x08, pipeline_base +0x30.
        assert_eq!(rd64(&w, 0x070 + 0x08), 0x15_0007_8000);
        assert_eq!(rd64(&w, 0x070 + 0x30), 0x11_0000_0000);
        assert_eq!(rd64(&w, 0x070 + 0x38), COMPUTE_INFO_UNK_38); // "always 0x8c60"
        assert_eq!(rd32(&w, 0x070 + 0x58), COMPUTE_INFO_UNK_58);
        assert_eq!(rd32(&w, 0x070 + 0x60), COMPUTE_INFO_IOGPU_UNK_40);
        // microsequence @0x1f0 (dump 0x1e8) — i.e. ComputeInfo really is 0x150.
        assert_eq!(rd64(&w, 0x1f0), 0xffff_ffa0_0c31_1cc0);
        assert_eq!(rd32(&w, 0x1f8), 0x240);
        // ComputeInfo2 @0x1fc: encoder_end +0x30 (dump 0x220 = 0x1f4 + 0x2c…, the
        // V>=13.0B4 form inserts a 4-byte hole at its head).
        assert_eq!(rd64(&w, 0x1fc + 0x30), 0x15_0007_8024);
        // EncoderParams @0x25c: encoder_id +0x0c (dump 0x260).
        assert_eq!(rd32(&w, 0x25c + 0x0c), 0x1100_22b3);
        assert_eq!(rd32(&w, 0x25c + 0x14), ENCODER_IOGPU_UNK44);
        // JobMeta @0x284 (dump 0x27c): 8-byte stamp pointers at +0x04 / +0x0c.
        assert_eq!(rd64(&w, 0x284 + 0x04), 0xffff_ffa0_000c_8014);
        assert_eq!(rd64(&w, 0x284 + 0x0c), 0xffff_ffa0_0c37_8014);
        assert_eq!(rd32(&w, 0x284 + 0x14), 0x3b00); // stamp_value
        assert_eq!(rd32(&w, 0x284 + 0x18), 5); // stamp_slot
        assert_eq!(rd32(&w, 0x284 + 0x24), 0x1200_22b8); // uuid
        // client_sequence @0x2d8 (dump 0x2d0) — only right if JobMeta is 0x2c.
        assert_eq!(w[0x2d8], 0x15);
    }

    /// The stamps are `NonZeroU64` in drm/asahi, so a builder that leaves them
    /// null has produced an illegal command — worth an explicit guard, because
    /// the firmware's only complaint is a job that never completes.
    #[test_case]
    fn a_stamp_pointer_is_never_null_in_a_built_command() {
        let w = run_compute(&c());
        assert_ne!(rd64(&w, 0x284 + 0x04), 0);
        assert_ne!(rd64(&w, 0x284 + 0x0c), 0);
    }

    #[test_case]
    fn sub_struct_sizes() {
        let mut b = Buf::new();
        compute_info(&mut b, &c());
        assert_eq!(b.len(), 0x150);
        let mut b = Buf::new();
        compute_info2(&mut b, &c());
        assert_eq!(b.len(), 0x60);
        let mut b = Buf::new();
        encoder_params(&mut b, &c());
        assert_eq!(b.len(), 0x28);
        let mut b = Buf::new();
        job_meta(&mut b, &c());
        assert_eq!(b.len(), 0x2c);
    }

    /// The work command is allocated with `align = 0x20` by the reference driver,
    /// so its size being a multiple of 0x20 is a (weak but free) corroboration
    /// that no sub-struct is a few bytes off.
    #[test_case]
    fn run_compute_size_is_0x20_aligned() {
        assert_eq!(RUN_COMPUTE_SIZE % 0x20, 0);
        assert_eq!(run_compute(&c()).len() % 0x20, 0);
    }

    #[test_case]
    fn cmd_queue_msg_layout() {
        let m = run_cmd_queue_msg(QUEUE_COMPUTE, 0xdead_0000, 5, 2, true);
        assert_eq!(m.len(), 0x40);
        assert_eq!(rd32(&m, 0x00), QUEUE_COMPUTE);
        assert_eq!(rd64(&m, 0x04), 0xdead_0000); // unaligned cmdqueue_addr
        assert_eq!(rd32(&m, 0x0c), 5); // head
        assert_eq!(rd32(&m, 0x10), 2); // event_number
        assert_eq!(rd32(&m, 0x14), 1); // new_queue
    }

    #[test_case]
    fn ring_state_layout() {
        let rs = ring_state(1, RB_ENTRIES);
        assert_eq!(rs.len(), RING_STATE_SIZE);
        assert_eq!(rd32(&rs, RS_GPU_DONEPTR as usize), 0);
        assert_eq!(rd32(&rs, RS_GPU_RPTR as usize), 0);
        assert_eq!(rd32(&rs, RS_CPU_WPTR as usize), 1);
        assert_eq!(rd32(&rs, 0x50), RB_ENTRIES); // rb_size
    }

    /// `CommandQueueInfo` used to be encoded 0x18 bytes short in the middle
    /// (`unk_34`/`unk_38`/`unk_40`/`unk_44`/`prio5` were missing), which put
    /// `gpu_context_addr` — the scheduler block the firmware needs — at 0x8c
    /// instead of 0xa4 and made the whole struct 0xa0 instead of 0xb8.
    #[test_case]
    fn queue_info_layout_and_gpu_context_offset() {
        let qi = queue_info(&QueueRefs {
            state: 0xa000,
            ring: 0xb000,
            job_list: 0xc000,
            gpu_buf: 0xd000,
            gpu_context: 0xe000,
            uuid: 0xc0ffee,
            event_id: -1,
            priority: 0,
        });
        assert_eq!(qi.len(), QUEUE_INFO_SIZE);
        assert_eq!(rd64(&qi, 0x00), 0xa000); // pointers_addr
        assert_eq!(rd64(&qi, 0x08), 0xb000); // rb_addr
        assert_eq!(rd64(&qi, 0x10), 0xc000); // job_list_addr
        assert_eq!(rd64(&qi, 0x18), 0xd000); // gpu_buf_addr
        assert_eq!(rd32(&qi, 0x2c) as i32, -1); // event_id
        assert_eq!(rd32(&qi, 0x50), 0xc0ffee); // uuid
        assert_eq!(rd32(&qi, 0x4c) as i32, -1); // unk_4c
        assert_eq!(rd32(&qi, 0x54) as i32, -1); // unk_54
        assert_eq!(rd64(&qi, QI_GPU_CONTEXT_OFFSET), 0xe000);
    }

    #[test_case]
    fn microseq_ops_and_sizes() {
        let refs = MicroSeqRefs {
            work_cmd_unk_buf: 0x1000 + WC_UNK_BUF_OFFSET,
            compute_info: 0x1000 + WC_COMPUTE_INFO_OFFSET,
            compute_info2: 0x1000 + WC_COMPUTE_INFO2_OFFSET,
            work_queue: 0x2000,
            fw_stamp: 0x3000,
            stamp_value: 0x3b00,
            uuid: 7,
            ..Default::default()
        };
        let ms = microseq_compute(&refs);
        // StartCompute, then WaitForIdle, then FinalizeCompute, then End.
        assert_eq!(rd32(&ms, 0), OP_START_COMPUTE);
        assert_eq!(rd64(&ms, 0x004), 0x1000 + WC_UNK_BUF_OFFSET);
        assert_eq!(rd64(&ms, 0x00c), 0x1000 + WC_COMPUTE_INFO_OFFSET);
        assert_eq!(rd64(&ms, 0x01c), 0x2000); // cmdqueue_ptr
        assert_eq!(rd32(&ms, 0x028), START_COMPUTE_UNK_28);
        assert_eq!(rd64(&ms, 0x03c), 0x1000 + WC_COMPUTE_INFO2_OFFSET);
        assert_eq!(rd32(&ms, START_COMPUTE_SIZE), OP_WAIT_FOR_IDLE_COMPUTE);
        let fin = START_COMPUTE_SIZE + 4;
        assert_eq!(rd32(&ms, fin), OP_FINALIZE_COMPUTE);
        assert_eq!(rd64(&ms, fin + 0x28), 0x3000); // stamp
        assert_eq!(rd32(&ms, fin + 0x30), 0x3b00); // stamp_value
        // restart_branch_offset walks back to StartCompute (offset 0).
        assert_eq!(rd32(&ms, fin + 0x58) as i32, -(fin as i32));
        assert_eq!(ms.len(), START_COMPUTE_SIZE + 4 + FINALIZE_COMPUTE_SIZE + 4);
        assert_eq!(rd32(&ms, ms.len() - 4), OP_END);
    }

    /// The compute wait must name *a* pipe. The proxyclient's TA/3D calls
    /// (`WaitForInterruptCmd(1,0,0)` / `(0,1,0)`) pin the encoding as
    /// `0x01 | pipe << 8`; compute's `1 << 15` puts 0x80 in byte 2. The assertion
    /// that matters is the last one — a bare 0x01 selects no pipe.
    #[test_case]
    fn wait_for_idle_names_the_compute_pipe() {
        assert_eq!(OP_WAIT_FOR_IDLE_COMPUTE, 0x01 | ((1u32 << 15) << 8));
        let b = OP_WAIT_FOR_IDLE_COMPUTE.to_le_bytes();
        assert_eq!(b[0], 0x01); // magic
        assert_eq!(b[1], 0); // not the vertex pipe (which is byte 1 = 1)
        assert_eq!(b[2], 0x80); // compute
        assert_eq!(b[3], 0);
        assert_ne!(OP_WAIT_FOR_IDLE_COMPUTE, 0x01);
    }

    #[test_case]
    fn event_and_context_blocks() {
        let jl = job_list(0x1000);
        assert_eq!(jl.len(), 0x18);
        assert_eq!(rd64(&jl, 0), 0); // first_job
        assert_eq!(rd64(&jl, 8), 0x1000); // last_head = self
        let ec = event_control(0x2000);
        assert_eq!(ec.len(), EVENT_CONTROL_SIZE);
        assert_eq!(rd64(&ec, 0), 0x2000); // event_count_addr
        assert_eq!(rd32(&ec, 0x08), 0); // submission_id starts at 0
        assert_eq!(rd32(&ec, 0x10), EVENT_CONTROL_UNK_10);
        assert_eq!(&ec[EC_UNK_BUF_OFFSET as usize..], &[0xff; 8]);
        assert_eq!(event_count().len(), 4);
        assert_eq!(stamp_counter(0x3b00), 0x3b00u32.to_le_bytes());
        let g = gpu_context_data();
        assert_eq!(g.len(), 0x40);
        assert_eq!((g[0], g[1], g[5], g[0x1e], g[0x23]), (0xff, 0xff, 1, 0xff, 2));
    }
}
