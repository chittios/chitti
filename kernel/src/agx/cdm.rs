//! **AGX compute dispatch encoders** — the CDM (Compute Data Master) command
//! stream + the USC (Unified Shader Control) words that launch a compute kernel.
//!
//! Pure, arch-neutral bit-packers ported **byte-exactly** from Mesa's generated
//! `agx_pack.h` (`src/asahi/genxml/cmdbuf.xml` → `gen_pack.py`) for **G14G =
//! M2 t8112** (our Mac mini; NOT G14X). See `COMPUTE_ISA_REF.md` for the field
//! tables and provenance. This is the Layer-3 dispatch encoding the Asahi kernel
//! docs declare "userspace concern" — it exists only in Mesa, so it is ported
//! here and unit-tested against the exact generated packing (`cargo xtask test`,
//! x86, no hardware — like `uat.rs`).
//!
//! A compute launch chains three GPU-memory objects:
//! 1. shader ISA (its addr → [`usc_shader`]'s `code`),
//! 2. the USC words (this module; their addr → [`cdm_launch_word1`]'s pipeline),
//! 3. the CDM command stream (this module: launch + sizes + barrier), which the
//!    `WorkCommandCP.compute_info.encoder` points at (Layer 2).

#![allow(dead_code)] // consumed by Layer 2 (submission), landing next

// --- CDM enums (cmdbuf.xml) ------------------------------------------------
/// CDM dispatch mode (`CDM Launch Word 0::Mode`).
pub const MODE_DIRECT: u32 = 0;
pub const MODE_INDIRECT_GLOBAL: u32 = 1;
pub const MODE_INDIRECT_LOCAL: u32 = 2;

const BLOCK_LAUNCH: u32 = 0;
const BLOCK_BARRIER: u32 = 3;

// --- USC control tags (byte 0 of each USC word; `USC Control` enum) ---------
const USC_SHADER: u8 = 0x0d;
const USC_UNIFORM: u8 = 0x1d;
const USC_SHARED: u8 = 0x4d;
const USC_REGISTERS: u8 = 0x8d;
const USC_NO_PRESHADER: u8 = 0x88;
/// `Shared layout` enum value used for vertex/compute shared memory.
const SHARED_LAYOUT_VERTEX_COMPUTE: u32 = 36;

/// Mesa `__gen_to_groups(value, group_size, length)`: encode a count in units of
/// `group_size`. Zero clamps to 1; a full `2^length` groups encodes as 0 ("all").
/// Ported verbatim from `agx_pack_header.h`.
pub const fn to_groups(value: u32, group_size: u32, length: u32) -> u32 {
    if value == 0 {
        return 1;
    }
    let groups = value.div_ceil(group_size);
    if groups == (1u32 << length) {
        0
    } else {
        groups
    }
}

/// Bitfield insert `[lo, hi]` (inclusive), the u64 form of Mesa `util_bitpack_uint`.
const fn bits(v: u64, lo: u32, hi: u32) -> u64 {
    let width = hi - lo + 1;
    let mask = if width == 64 { u64::MAX } else { (1u64 << width) - 1 };
    (v & mask) << lo
}

// ===================== CDM command stream =====================

/// `CDM_LAUNCH_WORD_0` — register counts + mode + Launch block type. The counts
/// come from the compiled shader (uniform/texture/sampler/preshader registers).
pub fn cdm_launch_word0(
    uniform_regs: u32,
    texture_regs: u32,
    sampler_regs: u32,
    preshader_regs: u32,
    mode: u32,
) -> u32 {
    let w = bits(to_groups(uniform_regs, 64, 3) as u64, 1, 3)
        | bits(to_groups(texture_regs, 8, 5) as u64, 4, 8)
        | bits(sampler_regs as u64, 9, 11)
        | bits(to_groups(preshader_regs, 16, 4) as u64, 12, 15)
        | bits(mode as u64, 27, 28)
        | bits(BLOCK_LAUNCH as u64, 29, 31);
    w as u32
}

/// `CDM_LAUNCH_WORD_1` — the USC-words pointer (`pipeline`), which must be
/// 64-byte aligned (the low 6 bits are asserted zero in Mesa). Stored directly in
/// bits [0:31] (the genxml `shr(6)`+`start=6` cancel in codegen).
pub fn cdm_launch_word1(pipeline: u32) -> u32 {
    debug_assert!(pipeline & 0x3f == 0, "USC pipeline must be 64B-aligned");
    pipeline
}

/// `CDM_GLOBAL_SIZE` / `CDM_LOCAL_SIZE` — three u32 words. Global = total threads
/// (workgroups × local); local = threads per workgroup.
pub fn cdm_size(x: u32, y: u32, z: u32) -> [u32; 3] {
    [x, y, z]
}

/// `CDM_BARRIER` — the post-dispatch barrier Mesa emits for a single-cluster
/// launch (`usc_cache_inval | unk_5 | unk_6 | unk_8`, block type Barrier). The
/// cluster-count / gen-13-specific bits are omitted (we are G14G, single job).
pub fn cdm_barrier() -> u32 {
    let w = bits(1, 3, 3) // usc_cache_inval
        | bits(1, 5, 5) // unk_5
        | bits(1, 6, 6) // unk_6
        | bits(1, 8, 8) // unk_8
        | bits(BLOCK_BARRIER as u64, 29, 31);
    w as u32
}

// ===================== USC words =====================
//
// Each USC word is a little-endian byte string tagged by its control byte. A
// compute pipeline's USC stream is (in order):
//   [usc_uniform]* → usc_shared_none → usc_shader → usc_registers → usc_no_preshader
// The stream's GPU address (>>? no — directly, 64B-aligned) goes in
// cdm_launch_word1. Helpers append to a byte Vec via `UscStream`.

/// Accumulates USC words into a byte buffer (the pipeline's USC stream).
pub struct UscStream {
    pub bytes: alloc::vec::Vec<u8>,
}

impl UscStream {
    pub fn new() -> Self {
        Self { bytes: alloc::vec::Vec::new() }
    }

    /// `USC Shader` (6 bytes): binds the shader ISA `code` address (32 bits at
    /// [16:47]).
    pub fn shader(&mut self, code: u32) {
        let w: u64 = USC_SHADER as u64 | bits(code as u64, 16, 47);
        self.bytes.extend_from_slice(&w.to_le_bytes()[..6]);
    }

    /// `USC Uniform` (8 bytes): DMA `size_halfs` 16-bit halfwords from `buffer`
    /// (4-byte aligned) into uniform registers starting at `start_halfs`. This is
    /// how kernel args (buffer pointers, dims) reach the shader.
    pub fn uniform(&mut self, start_halfs: u32, size_halfs: u32, buffer: u64) {
        debug_assert!(buffer & 0x3 == 0, "USC uniform buffer must be 4B-aligned");
        let w: u64 = USC_UNIFORM as u64
            | bits(start_halfs as u64, 8, 15)
            | bits(to_groups(size_halfs, 1, 6) as u64, 20, 25)
            | bits(buffer, 24, 63);
        self.bytes.extend_from_slice(&w.to_le_bytes());
    }

    /// `USC Registers` (4 bytes): GPR count (in groups of 8) + spill size, from
    /// the compiled shader.
    pub fn registers(&mut self, register_count: u32, spill_size: u32) {
        let w: u64 = USC_REGISTERS as u64
            | bits(to_groups(register_count, 8, 5) as u64, 8, 12)
            | bits(spill_size as u64, 18, 21);
        self.bytes.extend_from_slice(&(w as u32).to_le_bytes());
    }

    /// `USC Shared` (4 bytes), the no-shared-memory form (`agx_usc_shared_none`):
    /// vertex/compute layout, 65536 bytes/threadgroup, uses_shared=0.
    pub fn shared_none(&mut self) {
        let w: u64 = USC_SHARED as u64
            | bits(SHARED_LAYOUT_VERTEX_COMPUTE as u64, 10, 15)
            | bits(to_groups(65536, 256, 8) as u64, 24, 31);
        self.bytes.extend_from_slice(&(w as u32).to_le_bytes());
    }

    /// `USC No Preshader` (2 bytes).
    pub fn no_preshader(&mut self) {
        self.bytes.extend_from_slice(&(USC_NO_PRESHADER as u16).to_le_bytes());
    }
}

impl Default for UscStream {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Bit extractor for assertions.
    fn get(w: u64, lo: u32, hi: u32) -> u64 {
        let width = hi - lo + 1;
        let mask = if width == 64 { u64::MAX } else { (1u64 << width) - 1 };
        (w >> lo) & mask
    }

    #[test_case]
    fn to_groups_semantics() {
        assert_eq!(to_groups(0, 64, 3), 1); // zero clamps to 1
        assert_eq!(to_groups(16, 8, 5), 2); // ceil(16/8)
        assert_eq!(to_groups(17, 8, 5), 3); // ceil rounds up
        assert_eq!(to_groups(512, 64, 3), 0); // 8==2^3 groups -> "all" == 0
        assert_eq!(to_groups(64, 64, 3), 1); // 1 group, not "all"
        assert_eq!(to_groups(4, 1, 6), 4); // groups(1) is identity
    }

    #[test_case]
    fn launch_word0_fields() {
        let w = cdm_launch_word0(128, 0, 0, 0, MODE_DIRECT) as u64;
        assert_eq!(get(w, 1, 3), to_groups(128, 64, 3) as u64); // 2
        assert_eq!(get(w, 27, 28), MODE_DIRECT as u64);
        assert_eq!(get(w, 29, 31), BLOCK_LAUNCH as u64);
    }

    #[test_case]
    fn launch_word1_is_pipeline_addr() {
        assert_eq!(cdm_launch_word1(0x5000_0040), 0x5000_0040);
    }

    #[test_case]
    fn barrier_matches_mesa() {
        // usc_cache_inval|unk_5|unk_6|unk_8 | Barrier<<29
        let expect = (1 << 3) | (1 << 5) | (1 << 6) | (1 << 8) | (BLOCK_BARRIER << 29);
        assert_eq!(cdm_barrier(), expect);
        assert_eq!(get(cdm_barrier() as u64, 29, 31), BLOCK_BARRIER as u64);
    }

    #[test_case]
    fn usc_shader_binds_code() {
        let mut s = UscStream::new();
        s.shader(0x2000);
        assert_eq!(s.bytes.len(), 6);
        let mut w = [0u8; 8];
        w[..6].copy_from_slice(&s.bytes);
        let v = u64::from_le_bytes(w);
        assert_eq!(get(v, 0, 7), USC_SHADER as u64);
        assert_eq!(get(v, 16, 47), 0x2000);
    }

    #[test_case]
    fn usc_uniform_binds_buffer() {
        let mut s = UscStream::new();
        let buf = 0x15_0000_0000u64; // 4-aligned GPU VA
        s.uniform(0, 4, buf);
        assert_eq!(s.bytes.len(), 8);
        let v = u64::from_le_bytes(s.bytes[..8].try_into().unwrap());
        assert_eq!(get(v, 0, 7), USC_UNIFORM as u64);
        assert_eq!(get(v, 8, 15), 0); // start
        assert_eq!(get(v, 20, 25), 4); // size_halfs (groups(1) identity)
        // buffer recovers from bits [24:63] (4-aligned, low 2 bits land at 24/25=0)
        assert_eq!(get(v, 24, 63), buf & ((1u64 << 40) - 1));
    }

    #[test_case]
    fn usc_registers_and_tags() {
        let mut s = UscStream::new();
        s.registers(16, 0);
        let v = u32::from_le_bytes(s.bytes[..4].try_into().unwrap()) as u64;
        assert_eq!(get(v, 0, 7), USC_REGISTERS as u64);
        assert_eq!(get(v, 8, 12), 2); // groups(8) of 16

        let mut s2 = UscStream::new();
        s2.no_preshader();
        assert_eq!(s2.bytes, [USC_NO_PRESHADER, 0]);
    }

    #[test_case]
    fn full_compute_usc_stream_order_and_size() {
        // The minimal compute USC: uniform(args) -> shared -> shader -> registers
        // -> no_preshader. Sizes 8 + 4 + 6 + 4 + 2 = 24 bytes.
        let mut s = UscStream::new();
        s.uniform(0, 4, 0x15_0000_0000);
        s.shared_none();
        s.shader(0x2000);
        s.registers(16, 0);
        s.no_preshader();
        assert_eq!(s.bytes.len(), 8 + 4 + 6 + 4 + 2);
        // First word is the uniform (tag 0x1d), last two bytes the no-preshader.
        assert_eq!(s.bytes[0], USC_UNIFORM);
        assert_eq!(s.bytes[s.bytes.len() - 2], USC_NO_PRESHADER);
    }
}
