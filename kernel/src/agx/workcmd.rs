//! **AGX WorkCommandCP + cmdqueue message** — Layer 2 of compute submission.
//!
//! The compute *work command* (`RunCompute`, m1n1 `WorkCommandCP`) that wraps a
//! CDM command stream (from [`super::cdm`]) into a firmware job, plus the
//! `RunCmdQueueMsg` that kicks a cmdqueue channel. Ported **field-for-field** from
//! drm/asahi's `#[repr(C)]` firmware structs (`fw/compute.rs`, `fw/job.rs`,
//! `fw/workqueue.rs`) for **G14G / M2 t8112, firmware V13.5** (so version gates
//! `V >= V13_0B4`, `V >= V13_3` true; `G >= G14X` false). See `COMPUTE_ISA_REF.md`
//! and the drm/asahi sources under scratchpad `asahi-fw/`.
//!
//! **KNOWN UNRESOLVED AMBIGUITY (JobMeta / tail):** drm/asahi's `JobMeta.stamp`
//! and `.fw_stamp` are `GpuWeakPointer` = `NonZeroU64` (8 bytes), which makes
//! `raw::JobMeta` 0x2c and total 0x319 — but drm/asahi *names* the following field
//! `unk_2d4` (offset 0x2d4), which only holds if JobMeta is 0x24 (4-byte stamps),
//! and m1n1 models those stamps as full 64-bit. Source alone cannot settle this
//! 8-byte discrepancy in the **tail** (JobMeta onward: command_time, timestamps,
//! the stamp fields). This module currently encodes the 0x24 (4-byte-stamp) form
//! to match the field-name offsets, total 0x311. The tests below assert *internal
//! self-consistency*, NOT firmware-validated offsets — resolving the tail needs a
//! real dispatch (or a captured reference). **The early, critical fields are
//! unambiguous and correct regardless:** encoder @0x78, pipeline_base @0xa0,
//! microsequence @0x1f0, vm_slot @0x10, notifier @0x14 — all before JobMeta.
//! Pure + arch-neutral (`cargo xtask test`).
//!
//! Still TODO for a full dispatch (next increment): the microsequence ops
//! (StartCompute/FinalizeCompute), `CommandQueueInfo`, and the hw.rs wiring that
//! builds these in a context + submits + polls the Event.

#![allow(dead_code)] // consumed by the hw.rs submission wiring, landing next

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
    pub fn u64(&mut self, v: u64) -> &mut Self {
        self.bytes.extend_from_slice(&v.to_le_bytes());
        self
    }
    /// N zero bytes (`Array<N,u8>` / `Pad<N>`).
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

/// `workqueue::CommandType::RunCompute` — the `RunCompute.tag`.
pub const CMD_TYPE_RUN_COMPUTE: u32 = 3;

/// Parameters a caller supplies to build a compute WorkCommand. Everything the
/// firmware needs to run one dispatch; the many reverse-engineered `unk_*` fields
/// are zeroed (matching a minimal drm/asahi submission).
#[derive(Clone, Copy, Default)]
pub struct ComputeCmd {
    pub vm_slot: u32,
    pub notifier: u64,
    /// The CDM command stream GPU address (`JobParameters1.encoder`).
    pub encoder: u64,
    /// The context pipeline base (0x1100000000) — `JobParameters1.pipeline_base`.
    pub pipeline_base: u64,
    /// End of the CDM stream (`JobParameters2.encoder_end`).
    pub encoder_end: u64,
    pub encoder_id: u32,
    /// The microsequence GPU address + size.
    pub microsequence: u64,
    pub microsequence_size: u32,
    pub uuid: u32,
    pub stamp_value: u32,
}

/// Encode `JobParameters1` (G < G14X): the shader/encoder bindings. `encoder` @
/// +0x08, `pipeline_base` @ +0x30 within this struct; total 0x160 bytes.
fn job_params1(b: &mut Buf, c: &ComputeCmd) {
    b.u64(0); // preempt_buf1
    b.u64(c.encoder); // encoder  (+0x08)
    b.u64(0); // preempt_buf2
    b.u64(0); // preempt_buf3
    b.u64(0); // preempt_buf4
    b.u64(0); // preempt_buf5
    b.u64(c.pipeline_base); // pipeline_base (+0x30)
    b.u64(0); // unk_38
    b.u32(0); // helper_program
    b.u32(0); // unk_44
    b.u64(0); // helper_arg
    b.u32(0); // helper_cfg
    b.u32(0); // unk_54
    b.u32(0); // unk_58
    b.u32(0); // unk_5c
    b.u32(0); // iogpu_unk_40
    b.pad(0xfc); // __pad
}

/// Encode `JobParameters2` (V >= V13_0B4): the encoder-end binding; 0x60 bytes.
fn job_params2(b: &mut Buf, c: &ComputeCmd) {
    b.u32(0); // unk_0_0 (V>=13.0B4)
    b.pad(0x24); // unk_0
    b.u64(0); // preempt_buf1
    b.u64(c.encoder_end); // encoder_end
    b.pad(0x20); // unk_34
    b.u32(0); // unk_g14x
    b.u32(0); // unk_58
}

/// Encode `job::EncoderParams` (0x28 bytes).
fn encoder_params(b: &mut Buf, c: &ComputeCmd) {
    b.u32(0); // unk_8
    b.u32(0); // sync_grow
    b.u32(0); // unk_10
    b.u32(c.encoder_id); // encoder_id
    b.u32(0); // unk_18
    b.u32(0); // unk_mask
    b.u64(0); // sampler_array
    b.u32(0); // sampler_count
    b.u32(0); // sampler_max
}

/// Encode `job::JobMeta` (0x24 bytes). Weak pointers/EventValue are u32.
fn job_meta(b: &mut Buf, c: &ComputeCmd) {
    b.u16(0); // unk_0
    b.u8(0); // unk_2
    b.u8(0); // no_preemption
    b.u32(0); // stamp (GpuWeakPointer)
    b.u32(0); // fw_stamp
    b.u32(c.stamp_value); // stamp_value (EventValue)
    b.u32(0); // stamp_slot
    b.u32(0); // evctl_index
    b.u32(0); // flush_stamps
    b.u32(c.uuid); // uuid
    b.u32(0); // event_seq
}

/// Encode `job::TimestampPointers` (0x10 bytes; both None → 0).
fn timestamp_pointers(b: &mut Buf) {
    b.u64(0); // start_addr
    b.u64(0); // end_addr
}

/// Build the full compute WorkCommand (`raw::RunCompute`, G14G/V13.5). The result
/// is placed in the context and referenced by the cmdqueue's ring entry. Total
/// size 0x311 bytes.
pub fn run_compute(c: &ComputeCmd) -> Vec<u8> {
    let mut b = Buf::new();
    b.u32(CMD_TYPE_RUN_COMPUTE); // tag                @0x000
    b.u64(0); // counter (V>=13.0B4)                    @0x004
    b.u32(0); // unk_4                                  @0x00c
    b.u32(c.vm_slot); // vm_slot                        @0x010
    b.u64(c.notifier); // notifier                      @0x014
    b.u32(0); // unk_pointee                            @0x01c
    b.pad(0x50); // __pad0 (G<G14X)                     @0x020
    job_params1(&mut b, c); // job_params1              @0x070
    b.pad(0x20); // __pad1                              @0x1d0
    b.u64(c.microsequence); // microsequence            @0x1f0
    b.u32(c.microsequence_size); // microsequence_size  @0x1f8
    job_params2(&mut b, c); // job_params2              @0x1fc
    encoder_params(&mut b, c); // encoder_params        @0x25c
    job_meta(&mut b, c); // meta                        @0x284
    b.u64(0); // command_time                           @0x2a8
    timestamp_pointers(&mut b); // timestamp_pointers   @0x2b0
    timestamp_pointers(&mut b); // user_timestamp_ptrs  @0x2c0
    b.u8(0); // client_sequence                         @0x2d0
    b.pad(3); // pad_2d1                                @0x2d1
    b.u32(0); // unk_2d4                                @0x2d4
    b.u8(0); // unk_2d8                                 @0x2d8
    b.u64(0); // context_store_req (V>=13.0B4)          @0x2d9
    b.u64(0); // context_store_compl                    @0x2e1
    b.pad(0x14); // unk_2e9                             @0x2e9
    b.u32(0); // unk_flag                               @0x2fd
    b.pad(0x10); // unk_pad                             @0x301
    b.bytes
}

// --- cmdqueue channel message (m1n1 `RunCmdQueueMsg`, 0x40 on G14/V13.2+) -----

/// Queue types for `RunCmdQueueMsg.queue_type`.
pub const QUEUE_TA: u32 = 0;
pub const QUEUE_3D: u32 = 1;
pub const QUEUE_COMPUTE: u32 = 2;

/// Encode the `RunCmdQueueMsg` written into a cmdqueue channel ring to submit
/// work: which queue, the CommandQueueInfo address, the ring head, the event
/// number, and whether the queue is new. 0x40 bytes (G14/V13.2+). `cmdqueue_addr`
/// sits at an unaligned +0x04 (construct packs with no alignment).
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

// ===================== microsequence (drm/asahi fw/microseq.rs) =====================
//
// The firmware executes this instruction list for the job: StartCompute (runs the
// CDM encoder) → WaitForIdle → FinalizeCompute (signals the stamp) → RetireStamp.
// Ops use 8-byte `GpuWeakPointer`s (object.rs: "64-bit non-zero VA"; m1n1 agrees).
// `restart_branch_offset` in FinalizeCompute is the relative offset back to
// StartCompute (for firmware preemption/restart).

const OP_WAIT_FOR_IDLE: u32 = 0x01;
const OP_RETIRE_STAMP: u32 = 0x18;
const OP_START_COMPUTE: u32 = 0x29;
const OP_FINALIZE_COMPUTE: u32 = 0x2a;
const RETIRE_STAMP_ARGS: u32 = 0x4000_0000; // OpHeader::with_args (RetireStamp.HEADER)
const MAX_ATTACHMENTS: usize = 16;

/// Addresses the microsequence weaves together (all GPU VAs in the context).
#[derive(Clone, Copy, Default)]
pub struct MicroSeqRefs {
    /// WorkCommandCP.job_params1 (= workcmd_addr + 0x70).
    pub job_params1: u64,
    /// WorkCommandCP.job_params2 (= workcmd_addr + 0x1fc).
    pub job_params2: u64,
    /// The QueueInfo (work_queue).
    pub work_queue: u64,
    /// GpuStatsComp region (may be 0 for a first attempt).
    pub stats: u64,
    pub vm_slot: u32,
    pub event_seq: u64,
    pub uuid: u32,
    /// fw_stamp weak pointer (FinalizeCompute) + stamp value.
    pub fw_stamp: u64,
    pub stamp_value: u32,
    pub unk_flag: u64,
    pub counter: u64,
    pub notifier_buf: u64,
}

fn attachments(b: &mut Buf) {
    // Array<16, Attachment{address u64, size u32, unk_c u16, unk_e u16}> + count u32.
    for _ in 0..MAX_ATTACHMENTS {
        b.u64(0).u32(0).u16(0).u16(0);
    }
    b.u32(0); // count
}

/// Build the compute microsequence (StartCompute → WaitForIdle → FinalizeCompute
/// → RetireStamp) for G14G/V13.5. Returns the bytes; the caller records the byte
/// offset of StartCompute (0) so FinalizeCompute's restart branch is correct.
pub fn microseq_compute(r: &MicroSeqRefs) -> Vec<u8> {
    let mut b = Buf::new();
    // --- StartCompute (op 0x29) ---
    let start = b.len();
    b.u32(OP_START_COMPUTE); // header
    b.u64(0); // unk_pointer
    b.u64(r.job_params1); // job_params1 (Option<weak>)
    b.u64(r.stats); // stats
    b.u64(r.work_queue); // work_queue
    b.u32(r.vm_slot); // vm_slot
    b.u32(0); // unk_28
    b.u32(0); // event_generation
    b.u64(r.event_seq); // event_seq
    b.u32(0); // unk_38
    b.u64(r.job_params2); // job_params2
    b.u32(0); // unk_44
    b.u32(r.uuid); // uuid
    attachments(&mut b); // attachments + count
    b.u32(0); // padding
    b.u64(r.unk_flag); // unk_flag (V>=13.0B4)
    b.u64(r.counter); // counter (V>=13.0B4)
    b.u64(r.notifier_buf); // notifier_buf (V>=13.0B4)

    // --- WaitForIdle (op 0x01) ---
    b.u32(OP_WAIT_FOR_IDLE); // header (pipe<<8; compute pipe = 0 here)

    // --- FinalizeCompute (op 0x2a) ---
    let finalize = b.len();
    b.u32(OP_FINALIZE_COMPUTE); // header
    b.u64(r.stats); // stats
    b.u64(r.work_queue); // work_queue
    b.u32(r.vm_slot); // vm_slot
    b.u64(r.job_params2); // job_params2
    b.u32(0); // unk_24
    b.u32(r.uuid); // uuid
    b.u64(r.fw_stamp); // fw_stamp
    b.u32(r.stamp_value); // stamp_value
    b.u32(0); // unk_38
    b.u32(0); // unk_3c
    b.u32(0); // unk_40
    b.u32(0); // unk_44
    b.u32(0); // unk_48
    b.u32(0); // unk_4c
    b.u32(0); // unk_50
    b.u32(0); // unk_54
    b.u32(0); // unk_58
    // restart_branch_offset: relative from FinalizeCompute back to StartCompute.
    b.u32(((start as i64 - finalize as i64) as i32) as u32);
    b.u32(0); // has_attachments
    b.pad(0xd); // unk_64 (V>=13.0B4)
    b.u64(r.unk_flag); // unk_flag (V>=13.0B4)
    b.pad(0x7); // unk_79 (V>=13.0B4)

    // --- RetireStamp (op 0x18 | 0x40000000) ---
    b.u32(OP_RETIRE_STAMP | RETIRE_STAMP_ARGS);
    b.bytes
}

// ===================== command queue (drm/asahi fw/workqueue.rs) =====================

/// `RingState` (0x70) — the queue ring cursors. Field names encode offsets
/// (unk_10@0x10 …): each 4-byte field is followed by 0xc pad. Set `cpu_wptr` +
/// `rb_size` (in entries).
pub fn ring_state(cpu_wptr: u32, rb_entries: u32) -> Vec<u8> {
    let mut b = Buf::new();
    b.u32(0).pad(0xc); // gpu_doneptr  @0x00
    b.u32(0).pad(0xc); // unk_10       @0x10
    b.u32(0).pad(0xc); // unk_20       @0x20
    b.u32(0).pad(0xc); // gpu_rptr     @0x30
    b.u32(cpu_wptr).pad(0xc); // cpu_wptr @0x40
    b.u32(rb_entries).pad(0xc); // rb_size @0x50
    b.u32(0).pad(0xc); // cpu_freeptr  @0x60
    b.bytes // 0x70
}

/// Pointers the `QueueInfo` needs (all context GPU VAs).
#[derive(Clone, Copy, Default)]
pub struct QueueRefs {
    pub state: u64,         // RingState addr
    pub ring: u64,          // ring of u64 WorkCommand pointers
    pub notifier_list: u64, // NotifierList
    pub gpu_buf: u64,       // scratch gpu buffer
    pub gpu_context: u64,   // GpuContextData
    pub uuid: u32,
    pub event_id: i32,
}

/// `QueueInfo` (G14G/V13.2, G<G14X). The firmware reads this when a
/// `RunCmdQueueMsg` names it; `state` holds the cursors, `ring` the WorkCommand
/// pointers, `gpu_context` the per-context data.
pub fn queue_info(q: &QueueRefs) -> Vec<u8> {
    let mut b = Buf::new();
    b.u64(q.state); // state
    b.u64(q.ring); // ring
    b.u64(q.notifier_list); // notifier_list
    b.u64(q.gpu_buf); // gpu_buf
    b.u32(0); // gpu_rptr1
    b.u32(0); // gpu_rptr2
    b.u32(0); // gpu_rptr3
    b.u32(q.event_id as u32); // event_id
    b.u32(0); // priority
    b.u32(0); // unk_4c
    b.u32(q.uuid); // uuid
    b.u32(0); // unk_54
    b.u64(0); // unk_58
    b.u32(0); // busy
    b.pad(0x20); // __pad
    b.u32(0); // unk_84_0 (V>=13.2 && G<G14X)
    b.u32(0); // unk_84_state
    b.u32(0); // error_count
    b.u32(0); // unk_8c
    b.u32(0); // unk_90
    b.u32(0); // unk_94
    b.u32(0); // pending
    b.u32(0); // unk_9c
    b.u64(q.gpu_context); // gpu_context
    b.u64(0); // unk_a8
    b.u32(0); // unk_b0 (V>=13.2 && G<G14X)
    b.bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c() -> ComputeCmd {
        ComputeCmd {
            vm_slot: 1,
            notifier: 0xaabb,
            encoder: 0x1_5000_0000,
            pipeline_base: 0x11_0000_0000,
            encoder_end: 0x1_5000_1000,
            encoder_id: 7,
            microsequence: 0x1_5001_0000,
            microsequence_size: 0x80,
            uuid: 0x1234,
            stamp_value: 0x4100,
        }
    }

    fn rd64(b: &[u8], off: usize) -> u64 {
        u64::from_le_bytes(b[off..off + 8].try_into().unwrap())
    }
    fn rd32(b: &[u8], off: usize) -> u32 {
        u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
    }

    #[test_case]
    fn run_compute_size_and_key_offsets() {
        let w = run_compute(&c());
        assert_eq!(w.len(), 0x311, "RunCompute total size");
        assert_eq!(rd32(&w, 0x000), CMD_TYPE_RUN_COMPUTE); // tag
        assert_eq!(rd32(&w, 0x010), 1); // vm_slot
        assert_eq!(rd64(&w, 0x014), 0xaabb); // notifier
        // JobParameters1 @ 0x70: encoder @ +0x08, pipeline_base @ +0x30.
        assert_eq!(rd64(&w, 0x070 + 0x08), 0x1_5000_0000); // encoder
        assert_eq!(rd64(&w, 0x070 + 0x30), 0x11_0000_0000); // pipeline_base
        assert_eq!(rd64(&w, 0x1f0), 0x1_5001_0000); // microsequence
        assert_eq!(rd32(&w, 0x1f8), 0x80); // microsequence_size
    }

    #[test_case]
    fn run_compute_self_consistency() {
        // NB: this checks INTERNAL self-consistency of the 0x24-JobMeta form, not
        // firmware-validated offsets — see the module-level "KNOWN AMBIGUITY" note
        // (the JobMeta/tail has an unresolved 8-byte discrepancy vs m1n1). The
        // early fields asserted in `run_compute_size_and_key_offsets` ARE
        // unambiguous; these tail anchors just pin the chosen encoding.
        let w = run_compute(&c());
        assert_eq!(w.len(), 0x311);
        // encoder_id lands in EncoderParams @ 0x25c + 0x0c.
        assert_eq!(rd32(&w, 0x25c + 0x0c), 7);
        // uuid in JobMeta @ 0x284 + 0x1c.
        assert_eq!(rd32(&w, 0x284 + 0x1c), 0x1234);
        // stamp_value in JobMeta @ 0x284 + 0x0c (after unk_0/unk_2/no_preemption/
        // stamp/fw_stamp).
        assert_eq!(rd32(&w, 0x284 + 0x0c), 0x4100);
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
    fn sub_struct_sizes() {
        let mut b = Buf::new();
        encoder_params(&mut b, &c());
        assert_eq!(b.len(), 0x28);
        let mut b = Buf::new();
        job_meta(&mut b, &c());
        assert_eq!(b.len(), 0x24);
        let mut b = Buf::new();
        job_params1(&mut b, &c());
        assert_eq!(b.len(), 0x160);
        let mut b = Buf::new();
        job_params2(&mut b, &c());
        assert_eq!(b.len(), 0x60);
    }

    #[test_case]
    fn ring_state_layout() {
        let rs = ring_state(1, 0x80);
        assert_eq!(rs.len(), 0x70);
        assert_eq!(rd32(&rs, 0x40), 1); // cpu_wptr
        assert_eq!(rd32(&rs, 0x50), 0x80); // rb_size
    }

    #[test_case]
    fn microseq_has_start_and_finalize() {
        let ms = microseq_compute(&MicroSeqRefs {
            job_params1: 0x1000,
            work_queue: 0x2000,
            ..Default::default()
        });
        // Starts with StartCompute (0x29); ends with RetireStamp header.
        assert_eq!(rd32(&ms, 0), OP_START_COMPUTE);
        assert_eq!(
            rd32(&ms, ms.len() - 4),
            OP_RETIRE_STAMP | RETIRE_STAMP_ARGS
        );
        // StartCompute.job_params1 @ +0x0c (header + unk_pointer).
        assert_eq!(rd64(&ms, 0x0c), 0x1000);
    }

    #[test_case]
    fn queue_info_key_pointers() {
        let qi = queue_info(&QueueRefs {
            state: 0xa000,
            ring: 0xb000,
            gpu_context: 0xc000,
            ..Default::default()
        });
        assert_eq!(rd64(&qi, 0x00), 0xa000); // state
        assert_eq!(rd64(&qi, 0x08), 0xb000); // ring
    }
}
