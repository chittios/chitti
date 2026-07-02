OUTPUT_FORMAT(elf64-littleaarch64)
OUTPUT_ARCH(aarch64)
ENTRY(_start)

/* QEMU `-M virt` loads a `-kernel` ELF at its link address and jumps to the
   entry point in EL1 with the MMU off. RAM starts at 0x40000000; the
   conventional kernel load address is 0x40080000. */
SECTIONS
{
    . = 0x40080000;

    .text : {
        KEEP(*(.text.boot))
        *(.text*)
    }

    .rodata : { *(.rodata*) }
    .data   : { *(.data*) }

    .bss : {
        __bss_start = .;
        *(.bss*)
        *(COMMON)
        __bss_end = .;
    }

    /* Boot stack grows down from __stack_top. */
    . = ALIGN(16);
    . += 0x100000; /* 1 MiB */
    __stack_top = .;
}
