//! aarch64 boot stub. QEMU `-M virt -kernel` loads the ELF and jumps here in
//! EL1 with the MMU off; this sets the stack, enables FP/SIMD (NEON), zeroes
//! BSS, and calls the kernel entry (`crate::aarch64_start`). The linker script
//! (`kernel/linker-aarch64.ld`) provides `__stack_top`/`__bss_start/end`.

#[cfg(not(feature = "boot-limine"))]
use core::arch::global_asm;

// The `-kernel` boot stub is used only for the default (non-Limine) build; the
// Limine build provides its own `limine_start` entry (arch::aarch64::limine).
#[cfg(not(feature = "boot-limine"))]
global_asm!(
    r#"
.section .text.boot
.global _start
_start:
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
2:  bl   aarch64_start
3:  wfi
    b    3b
"#
);
