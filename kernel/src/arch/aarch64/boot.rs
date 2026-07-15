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

/// The Limine build has no `-kernel` boot-info page.
#[cfg(feature = "boot-limine")]
pub fn boot_x1() -> u64 {
    0
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

// TEMP bare-boot bisect: paint a thin band of colour (clo | chi<<16) into the
// firmware framebuffer at vertical slot `slot` (each slot 0x80000 B ~ a few rows
// further down a 1920-wide screen). Gated to a real Apple boot by the saved FDT
// pointer in x21 (> 32 GiB); QEMU's DTB is low so every band is skipped there.
// Stacking one band per _start sub-step turns a single serial-less bare boot into
// a full progress ladder — the lowest band present on the monitor at reset is the
// last step reached, pinpointing the faulting instruction in one boot. FB PA
// 0x9_e52d_4000 is the Mac mini M2 firmware framebuffer (m1n1 boot log). Clobbers
// x2/x3/x4/w4 (dead at every call site). REMOVE once the bare boot is stable.
.macro DBGBAND slot, clo, chi
    movz x2, #0x8, lsl #32
    cmp  x21, x2
    b.lo .Ldbg_skip\@
    movz x2, #0x4000
    movk x2, #0xe52d, lsl #16
    movk x2, #0x0009, lsl #32
    .if \slot != 0
    add  x2, x2, #(\slot * 0x80), lsl #12
    .endif
    movz w4, #\clo
    .if \chi != 0
    movk w4, #\chi, lsl #16
    .endif
    movz x3, #0x1, lsl #16          // 0x10000 px ~ 34 rows at 1920w
.Ldbg_loop\@:
    str  w4, [x2], #4
    subs x3, x3, #1
    b.ne .Ldbg_loop\@
.Ldbg_skip\@:
.endm

.global _start
_start:
    // Preserve the entry registers in callee-saved regs across the .bss zero
    // (which clobbers x0/x1 and would wipe the statics if stored before):
    //   x1 = boot-info page (UEFI stub) -> BOOT_X1
    //   x0 = DTB / FDT pointer          -> BOOT_X0
    mov  x20, x1
    mov  x21, x0
    DBGBAND 0, 0x03ff, 0         // blue  : _start entered (m1n1 handed off)

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
    mov  x9, #(1 << 31)         // HCR_EL2.RW = 1; E2H = TGE = 0
    msr  hcr_el2, x9
    isb
    DBGBAND 1, 0xffff, 0x000f   // cyan  : HCR_EL2 written, E2H cleared, ISB done
    // Sane EL1 SCTLR: MMU + caches off, with the architectural RES1 bits set
    // (bits 11,20,22,23,28,29 = 0x30d00800). Written directly, NOT RMW: the
    // reset/aliased value is not trustworthy.
    movz x9, #0x30d0, lsl #16
    movk x9, #0x0800
    msr  sctlr_el1, x9
    isb
    DBGBAND 2, 0xfc00, 0x000f   // green : SCTLR_EL1 written (real EL1 SCTLR)
    // Now in the non-VHE CPTR_EL2 format: don't trap EL1 FP/SIMD or CP15.
    mov  x9, #0x33ff            // RES1 bits set, TFP = 0
    msr  cptr_el2, x9
    msr  hstr_el2, xzr
    // Let EL1 read the generic timer/counter; zero the virtual-counter offset.
    mrs  x9, cnthctl_el2
    orr  x9, x9, #3             // EL1PCTEN | EL1PCEN
    msr  cnthctl_el2, x9
    msr  cntvoff_el2, xzr
    DBGBAND 3, 0xfc00, 0x3fff   // yellow: CPTR/HSTR/CNTHCTL/CNTVOFF set, pre-eret
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
    DBGBAND 4, 0x0000, 0x3ff0   // red   : eret landed at EL1, SP set

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
    DBGBAND 5, 0x03ff, 0x3ff0   // magenta: PIE relocation loop completed

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
3:
    DBGBAND 6, 0xffff, 0x3fff   // white : .bss zeroed, about to bl aarch64_start
    adrp x0, BOOT_X1
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
