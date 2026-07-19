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
//! The struct field NAMES self-encode their byte offsets (`unk_2d4`@0x2d4,
//! `unk_2e9`@0x2e9, …); the unit tests assert exactly those, so a layout slip is
//! caught without hardware. Pure + arch-neutral (`cargo xtask test`).
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
    fn run_compute_self_encoding_offset_anchors() {
        // The drm/asahi field names encode their offsets — assert the layout lands
        // exactly there (a slipped field size would move these).
        let w = run_compute(&c());
        // unk_2d4 is zero but its POSITION proves everything before it is sized
        // right; the strongest anchor is uuid inside JobMeta and the total size.
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
}
