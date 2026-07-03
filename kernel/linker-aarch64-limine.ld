OUTPUT_FORMAT(elf64-littleaarch64)
OUTPUT_ARCH(aarch64)
ENTRY(limine_start)

/* aarch64 kernel loaded by the Limine boot protocol (BOOTAA64.EFI under UEFI/
   AAVMF), so it can boot from a disk's ESP — the counterpart to the x86 Limine
   kernel. Higher-half link address (TTBR1); Limine sets up the MMU + maps the
   kernel here, provides an HHDM + identity map + the memmap/framebuffer/module
   responses. The `-kernel` boot stub (linker-aarch64.ld) is the other build. */
PHDRS
{
    requests PT_LOAD FLAGS((1 << 2)) ;
    text     PT_LOAD FLAGS((1 << 0) | (1 << 2)) ;
    rodata   PT_LOAD FLAGS((1 << 2)) ;
    data     PT_LOAD FLAGS((1 << 1) | (1 << 2)) ;
}

SECTIONS
{
    . = 0xffffffff80000000;

    .requests : {
        KEEP(*(.requests_start_marker))
        KEEP(*(.requests))
        KEEP(*(.requests_end_marker))
    } :requests

    . = ALIGN(CONSTANT(MAXPAGESIZE));
    .text : { *(.text .text.*) } :text

    . = ALIGN(CONSTANT(MAXPAGESIZE));
    .rodata : { *(.rodata .rodata.*) } :rodata

    . = ALIGN(CONSTANT(MAXPAGESIZE));
    .data : { *(.data .data.*) } :data

    .bss : {
        __bss_start = .;
        *(.bss .bss.*)
        *(COMMON)
        __bss_end = .;
    } :data

    /DISCARD/ : { *(.eh_frame) *(.note .note.*) *(.comment) }
}
