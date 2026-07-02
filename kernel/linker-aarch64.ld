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

    /DISCARD/ : { *(.comment) *(.note*) *(.eh_frame*) }
}
