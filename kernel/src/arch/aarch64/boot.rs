//! aarch64 boot stub. QEMU `-M virt -kernel` loads the ELF and jumps here in
//! EL1 with the MMU off; this sets the stack, enables FP/SIMD (NEON), zeroes
//! BSS, and calls the kernel entry (`crate::aarch64_start`). The linker script
//! (`kernel/linker-aarch64.ld`) provides `__stack_top`/`__bss_start/end`.

#[cfg(not(feature = "boot-limine"))]
use core::arch::global_asm;

/// The boot-info pointer the entry received in x1: 0 under QEMU `-kernel`
/// (Linux boot convention), or the UEFI stub's boot-info page (GOP framebuffer)
/// when booted via the stub. `_start` saves x1 here after zeroing `.bss` (so
/// this static, which lives in `.bss`, isn't wiped). Read by the boot code to
/// pick the framebuffer source.
#[cfg(not(feature = "boot-limine"))]
#[no_mangle]
pub static mut BOOT_X1: u64 = 0;

/// The value the entry received in x0. On QEMU `-M virt -kernel` this is the
/// **flattened device tree (DTB)** physical address (Linux boot convention) —
/// the kernel parses its `/memory` node to discover RAM size (see [`super::dtb`]
/// / [`super::mmu`]). On the UEFI-stub path x0 is not a DTB; the parser rejects
/// it via the FDT magic and RAM size comes from the stub boot-info page instead.
/// `_start` saves it (from a callee-saved reg) after zeroing `.bss`.
#[cfg(not(feature = "boot-limine"))]
#[no_mangle]
pub static mut BOOT_X0: u64 = 0;

/// The boot-info pointer the entry received in x1 (0 on the `-kernel` path).
#[cfg(not(feature = "boot-limine"))]
pub fn boot_x1() -> u64 {
    // SAFETY: written once by `_start` before any Rust runs; read-only after.
    unsafe { core::ptr::read_volatile(core::ptr::addr_of!(BOOT_X1)) }
}

/// The x0 value at entry — the DTB physical address on the `-kernel` path.
#[cfg(not(feature = "boot-limine"))]
pub fn boot_x0() -> u64 {
    // SAFETY: written once by `_start` before any Rust runs; read-only after.
    unsafe { core::ptr::read_volatile(core::ptr::addr_of!(BOOT_X0)) }
}

/// The UEFI stub's hosted-model boot seed (`\chitti-model.json` on the ESP),
/// handed over in the boot-info page the same way the EDID is: length at
/// offset 1024, the JSON bytes at 1028. `None` on the QEMU `-kernel` path
/// (no boot page) or when the image carried no seed. The page is identity
/// mapped LOADER_DATA RAM, still readable at shell start.
#[cfg(not(feature = "boot-limine"))]
pub fn boot_page_remote_cfg() -> Option<&'static [u8]> {
    let bi = boot_x1();
    if bi == 0 || bi >= crate::arch::aarch64::mmu::mapped_bytes() {
        return None;
    }
    // SAFETY: identity-mapped RAM below the map limit; the magic + length are
    // validated before the bytes are returned.
    let magic = unsafe { core::slice::from_raw_parts(bi as *const u8, 8) };
    if magic != b"CHITTIBI" {
        return None;
    }
    let len = unsafe { core::ptr::read_volatile((bi + 1024) as *const u32) } as usize;
    if len == 0 || len > 2048 {
        return None;
    }
    // SAFETY: `len` bounded by the stub's 2048-byte cap; 1028 + len < 4096.
    Some(unsafe { core::slice::from_raw_parts((bi + 1028) as *const u8, len) })
}

/// The Limine build has no `-kernel` boot-info page.
#[cfg(feature = "boot-limine")]
pub fn boot_x1() -> u64 {
    0
}

#[cfg(feature = "boot-limine")]
pub fn boot_page_remote_cfg() -> Option<&'static [u8]> {
    None
}

/// The Limine build discovers RAM from the Limine memory map, not a DTB.
#[cfg(feature = "boot-limine")]
pub fn boot_x0() -> u64 {
    0
}

// The `-kernel` boot stub is used only for the default (non-Limine) build; the
// Limine build provides its own `limine_start` entry (arch::aarch64::limine).
#[cfg(not(feature = "boot-limine"))]
global_asm!(
    r#"
.section .text.boot
// ---------------------------------------------------------------------------
// arm64 Linux `Image` header (Documentation/arch/arm64/booting.rst). Lets a
// bootloader that speaks the Linux boot protocol — **m1n1** on real Apple
// Silicon — recognise and load this kernel. It sits at offset 0 of the flat
// (objcopy'd) image; `code0` branches past the 64-byte header to `_start`.
// QEMU `-kernel <elf>` ignores it and jumps straight to the ELF entry
// (`_start`), so both boot paths converge there.
// ---------------------------------------------------------------------------
.global _image_start
_image_start:
    b    _start                 // code0: branch past the header
    .long 0                     // code1 (reserved)
    .quad 0                     // text_offset (loader places per flags)
    .quad __image_size          // image_size (linker ABSOLUTE symbol)
    .quad 0xa                   // flags: LE | 4KiB page (1<<1) | anywhere (1<<3)
    .quad 0                     // res2
    .quad 0                     // res3
    .quad 0                     // res4
    .long 0x644d5241            // magic "ARM\x64"
    .long 0                     // res5 (PE/COFF header offset; unused)

.global _start
_start:
    // Preserve the entry registers in callee-saved regs across the .bss zero
    // (which clobbers x0/x1 and would wipe the statics if stored before):
    //   x1 = boot-info page (UEFI stub) -> BOOT_X1
    //   x0 = DTB / FDT pointer          -> BOOT_X0
    mov  x20, x1
    mov  x21, x0

    // If we entered at EL2 (m1n1 / iBoot / QEMU `virtualization=on`), drop to
    // EL1 — the whole kernel is written to the EL1 system registers. QEMU
    // `-kernel` under HVF already enters at EL1, so this is a no-op there.
    // (Apple's guarded exception levels / GXF are ignored; we run plain EL1.)
    mrs  x9, CurrentEL
    lsr  x9, x9, #2
    cmp  x9, #2
    b.ne 1f
    // Enter EL1 in AArch64. m1n1/iBoot may hand off with **VHE enabled**
    // (HCR_EL2.E2H=1), where the EL1 system registers alias the EL2 ones — so we
    // must first put HCR_EL2 into a plain non-VHE EL1 state (RW=1, E2H=0, TGE=0)
    // and ISB, *before* touching any EL1 register, or SCTLR_EL1 would still be
    // SCTLR_EL2. Missing this ISB / doing a read-modify-write of the aliased
    // SCTLR_EL1 leaves the real EL1 SCTLR unknown → instant fault at EL1 on
    // hardware (QEMU/HVF hid it by entering at EL1 directly).
    // HCR_EL2 = API | APK | RW (0x0000_0300_8000_0000). RW=1 (AArch64 EL1),
    // TGE=0. E2H is RES1 on Apple (VHE-only) so it stays 1 regardless — we run at
    // EL1 under VHE. API|APK (bits 41,40) keep PAuth instructions/key registers
    // from trapping to EL2 once we run at EL1 (Rust prologues may sign the return
    // address). QEMU/HVF hide the drop by entering at EL1 directly.
    movz x9, #0x8000, lsl #16   // RW      (bit 31)
    movk x9, #0x0300, lsl #32   // APK|API (bits 40,41)
    msr  hcr_el2, x9
    isb
    // Sane EL1 SCTLR: MMU + caches off, with the architectural RES1 bits set
    // (bits 11,20,22,23,28,29 = 0x30d00800). Written directly, NOT RMW: the
    // reset/aliased value is not trustworthy.
    movz x9, #0x30d0, lsl #16
    movk x9, #0x0800
    msr  sctlr_el1, x9
    isb
    // Enable FP/SIMD at EL2 for the ACTUAL E2H state. Apple cores are VHE-only:
    // HCR_EL2.E2H is RES1, so the clear above is ignored and we stay VHE. In VHE
    // CPTR_EL2 is CPACR-format and FP is gated by FPEN[21:20], NOT the non-VHE
    // TFP bit — writing the non-VHE 0x33ff leaves FPEN=0b00 and traps all FP/SIMD
    // (NEON array inits, cortex, video) at EL1. Branch on E2H and set the right
    // value: VHE -> FPEN=0b11; non-VHE (QEMU/SBSA) -> RES1 bits with TFP=0.
    mrs  x10, hcr_el2
    tst  x10, #(1 << 34)        // HCR_EL2.E2H
    b.eq 32f
    mov  x9, #0x300000          // VHE: CPTR_EL2.FPEN = 0b11 (do not trap FP)
    msr  cptr_el2, x9
    b    33f
32: mov  x9, #0x33ff            // non-VHE: RES1 bits + TFP = 0
    msr  cptr_el2, x9
33:
    msr  hstr_el2, xzr
    // Let EL1 read the generic timer/counter; zero the virtual-counter offset.
    mrs  x9, cnthctl_el2
    orr  x9, x9, #3             // EL1PCTEN | EL1PCEN
    msr  cnthctl_el2, x9
    msr  cntvoff_el2, xzr
    // eret into EL1h with DAIF masked.
    mov  x9, #0x3c5
    msr  spsr_el2, x9
    adr  x9, 1f
    msr  elr_el2, x9
    eret
1:
    adrp x0, __stack_top
    add  x0, x0, :lo12:__stack_top
    mov  sp, x0

    // PIE self-relocation: apply the R_AARCH64_RELATIVE records the `-pie` link
    // emitted, so every absolute address is correct at the *actual* load address
    // (m1n1 loads us at Apple's ~32 GiB RAM base, not the 0x40080000 link base).
    // delta = runtime(_image_start) - 0x40080000; for each entry patch
    // *(r_offset + delta) = r_addend + delta. All PC-relative (adrp) here, so it
    // works pre-relocation. On QEMU `-kernel` delta == 0 → a harmless no-op.
    adrp x9, _image_start
    add  x9, x9, :lo12:_image_start      // runtime image base
    movz x10, #0x4008, lsl #16           // link base 0x40080000
    sub  x11, x9, x10                     // x11 = delta
    adrp x12, __rela_start
    add  x12, x12, :lo12:__rela_start
    adrp x13, __rela_end
    add  x13, x13, :lo12:__rela_end
5:  cmp  x12, x13
    b.hs 7f
    ldr  x14, [x12]                       // r_offset (link-space)
    ldr  x15, [x12, #8]                   // r_info
    ldr  x16, [x12, #16]                  // r_addend (link-space value)
    and  x15, x15, #0xffffffff            // reloc type = r_info[31:0]
    cmp  x15, #1027                       // R_AARCH64_RELATIVE
    b.ne 6f
    add  x14, x14, x11                    // place = r_offset + delta
    add  x16, x16, x11                    // value = r_addend + delta
    str  x16, [x14]
6:  add  x12, x12, #24                    // sizeof(Elf64_Rela)
    b    5b
7:  dsb  sy
    isb

    // Enable FP/SIMD access at EL1: CPACR_EL1.FPEN = 0b11 (bits [21:20]).
    mrs  x0, cpacr_el1
    orr  x0, x0, #(3 << 20)
    msr  cpacr_el1, x0
    isb

    // Zero .bss.
    adrp x0, __bss_start
    add  x0, x0, :lo12:__bss_start
    adrp x1, __bss_end
    add  x1, x1, :lo12:__bss_end
2:  cmp  x0, x1
    b.hs 3f
    str  xzr, [x0], #8
    b    2b
3:  adrp x0, BOOT_X1
    add  x0, x0, :lo12:BOOT_X1
    str  x20, [x0]
    adrp x0, BOOT_X0
    add  x0, x0, :lo12:BOOT_X0
    str  x21, [x0]
    bl   aarch64_start
4:  wfi
    b    4b
"#
);
