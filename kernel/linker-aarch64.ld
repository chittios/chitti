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
    .data   : { *(.data .data.*) }

    .bss : {
        . = ALIGN(16);
        __bss_start = .;
        *(.bss .bss.*)
        *(COMMON)
        . = ALIGN(16);
        __bss_end = .;
    }

    /* Boot stack grows down from __stack_top (2 MiB). */
    . = ALIGN(16);
    . += 0x200000;
    __stack_top = .;

    /* Effective image size for the arm64 `Image` header: the whole span the
       kernel occupies in memory (text..bss..boot stack) so a loader (m1n1)
       reserves enough. An ABSOLUTE symbol, so `.quad __image_size` in the
       boot header emits the size *value*, not an address. */
    __image_size = __stack_top - _image_start;

    /DISCARD/ : { *(.comment) *(.note*) *(.eh_frame*) }
}
