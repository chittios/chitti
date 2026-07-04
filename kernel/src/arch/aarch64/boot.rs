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
.global _start
_start:
    // Preserve the entry registers in callee-saved regs across the .bss zero
    // (which clobbers x0/x1 and would wipe the statics if stored before):
    //   x1 = boot-info page (UEFI stub) -> BOOT_X1
    //   x0 = DTB pointer (`-kernel`)     -> BOOT_X0
    mov  x20, x1
    mov  x21, x0

    adrp x0, __stack_top
    add  x0, x0, :lo12:__stack_top
    mov  sp, x0

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
1:  cmp  x0, x1
    b.hs 2f
    str  xzr, [x0], #8
    b    1b
2:  adrp x0, BOOT_X1
    add  x0, x0, :lo12:BOOT_X1
    str  x20, [x0]
    adrp x0, BOOT_X0
    add  x0, x0, :lo12:BOOT_X0
    str  x21, [x0]
    bl   aarch64_start
3:  wfi
    b    3b
"#
);
