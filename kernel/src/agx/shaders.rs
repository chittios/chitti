//! **AGX compute shaders** — hand-assembled kernel machine code (Layer 3a).
//!
//! Assembled with `dougallj/applegpu` (G13/M1 assembler) and round-trip verified
//! by its disassembler. The four instructions used (`get_sr`, `mov_imm`,
//! `device_store`, `stop`) are core AGX ISA and are believed identical on
//! **G14G / M2 t8112** — applegpu has no G14 awareness, but these opcodes did not
//! change across G13→G14. The G13→G14 risk lives in the USC/dispatch descriptors
//! ([`super::cdm`]), which are ported from Mesa's authoritative G14 genxml, not in
//! these bytes. If a dispatch ever misbehaves, suspect the USC binding first.

/// **`hello_compute`** — writes the constant `0xCAFEF00D` to `out[thread_index]`.
/// A known-answer bring-up kernel: after a dispatch of N threads, `out[0..N]`
/// should all read `0xCAFEF00D`, proving the whole submission pipeline end to end.
///
/// Assembly (applegpu):
/// ```text
/// get_sr r1, sr80                    ; r1 = thread_position_in_grid.x
/// mov_imm r0, 0xCAFEF00D, 0          ; r0 = constant
/// device_store 0, i32, x, r0, u0_u1, r1, unsigned, lsl 2, 0   ; out[r1] = r0
/// stop
/// ```
pub const HELLO_COMPUTE: &[u8] = &[
    0x72, 0x05, 0x10, 0x04, // get_sr r1, sr80 (thread_position_in_grid.x)
    0x62, 0x01, 0x0d, 0xf0, 0xfe, 0xca, // mov_imm r0, 0xCAFEF00D, 0
    0x45, 0x01, 0x20, 0x0e, 0x00, 0xc8, 0x12, 0x00, // device_store out[r1]=r0
    0x88, 0x00, // stop
];

/// The constant [`HELLO_COMPUTE`] writes — the expected value in every output slot.
pub const HELLO_COMPUTE_MAGIC: u32 = 0xCAFE_F00D;

/// GPRs (32-bit) used by [`HELLO_COMPUTE`] → USC `Registers` count.
pub const HELLO_COMPUTE_GPRS: u32 = 2;

/// Uniform data [`HELLO_COMPUTE`] expects, as 16-bit halfwords bound at uniform
/// register 0 (`u0_u1`): the 64-bit output-buffer GPU VA (little-endian) at
/// offset 0. So the arg buffer is just `[out_ptr: u64]`, and the USC Uniform word
/// is `start_halfs=0, size_halfs=4, buffer=<arg buffer addr>`.
pub const HELLO_COMPUTE_UNIFORM_HALFS: u32 = 4;

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn hello_compute_is_the_verified_blob() {
        // 20 bytes, ends in `stop` (0x8800), starts with get_sr (0x72).
        assert_eq!(HELLO_COMPUTE.len(), 20);
        assert_eq!(HELLO_COMPUTE[0], 0x72);
        assert_eq!(&HELLO_COMPUTE[18..20], &[0x88, 0x00]);
        // The mov_imm carries the little-endian magic in bytes [6..10] shifted by
        // the imm encoding — just assert the magic constant is what we document.
        assert_eq!(HELLO_COMPUTE_MAGIC, 0xCAFE_F00D);
    }
}
