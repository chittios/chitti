OUTPUT_FORMAT(elf64-littleaarch64)
OUTPUT_ARCH(aarch64)
ENTRY(_start)

/* QEMU `-M virt` loads a `-kernel` ELF at its link address and jumps to the
   entry point in EL1 with the MMU off. RAM starts at 0x40000000; the
   conventional kernel load address is 0x40080000. The `.text.boot` stub
   (arch/aarch64/boot.rs) must come first. Keep this consistent with the heap
   base in mm::init (0x50000000, well past the image). */
SECTIONS
{
    . = 0x40080000;
    /* Base of the loaded image — the arm64 `Image` header (in `.text.boot`)
       sits here, so `_image_start` is offset 0 of a flat (objcopy'd) image. */
    _image_start = .;

    .text : {
        KEEP(*(.text.boot))
        *(.text .text.*)
    }

    .rodata : { *(.rodata .rodata.*) }

    /* PIE self-relocation: the linker emits R_AARCH64_RELATIVE entries here for
       every absolute address baked into the image (including GOT slots). The
       boot stub (`arch/aarch64/boot.rs` `_start`) walks [__rela_start,
       __rela_end) and applies each (place += delta, value += delta) so the
       kernel runs correctly wherever m1n1 loads it. Keep 8-byte aligned. */
    . = ALIGN(8);
    __rela_start = .;
    .rela.dyn : { *(.rela .rela.*) }
    __rela_end = .;

    .got  : { *(.got) }
    .got.plt : { *(.got.plt) }

    .data   : { *(.data .data.*) }

    .bss : {
        . = ALIGN(16);
        __bss_start = .;
        *(.bss .bss.*)
        *(COMMON)
        . = ALIGN(16);
        __bss_end = .;
    }

    /* Boot stack, growing down from __stack_top.
     *
     * `__stack_guard` is a 64 KiB region reserved BELOW the stack and never
     * mapped writable. Without it the stack's bottom was `__bss_end` exactly:
     * an overflow walked straight into `.bss`, silently rewriting kernel
     * statics (a corrupted ktrace sequence counter was the visible symptom),
     * and the next exception then re-entered the vector on a ruined stack and
     * looped there forever — a dead machine, no output, no Ctrl+C. With the
     * guard the overflow faults at a known address instead, and
     * `aarch64_sync_exception` can say "kernel stack overflow" and stop.
     *
     * 8 MiB of stack: the deepest real consumer measured is the PEG parser on
     * a minified page (~440 nested rules with multi-KiB frames), which came
     * within reach of 2 MiB. */
    . = ALIGN(4096);
    __stack_guard_bottom = .;
    . += 0x10000;
    __stack_guard_top = .;
    . = ALIGN(16);
    . += 0x800000;
    __stack_top = .;

    /* Effective image size for the arm64 `Image` header: the whole span the
       kernel occupies in memory (text..bss..boot stack) so a loader (m1n1)
       reserves enough. An ABSOLUTE symbol, so `.quad __image_size` in the
       boot header emits the size *value*, not an address. */
    __image_size = __stack_top - _image_start;

    /* No runtime dynamic loader — we self-relocate from .rela.dyn only, so the
       dynamic-linking metadata a `-pie` link would otherwise emit is dead
       weight. Drop it (also .comment/.note/.eh_frame). */
    /DISCARD/ : {
        *(.comment) *(.note*) *(.eh_frame*)
        *(.dynamic) *(.dynsym) *(.dynstr) *(.hash) *(.gnu.hash)
        *(.gnu.version) *(.gnu.version_d) *(.gnu.version_r) *(.interp)
    }
}
