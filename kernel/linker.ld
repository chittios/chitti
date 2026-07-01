OUTPUT_FORMAT(elf64-x86-64)
OUTPUT_ARCH(i386:x86-64)
ENTRY(_start)

PHDRS
{
    requests PT_LOAD    FLAGS((1 << 2)) ;
    text     PT_LOAD    FLAGS((1 << 0) | (1 << 2)) ;
    rodata   PT_LOAD    FLAGS((1 << 2)) ;
    data     PT_LOAD    FLAGS((1 << 1) | (1 << 2)) ;
}

SECTIONS
{
    /* Higher half, matches the "kernel" code model in targets/x86_64-chitti.json. */
    . = 0xffffffff80000000;

    .requests : {
        KEEP(*(.requests_start_marker))
        KEEP(*(.requests))
        KEEP(*(.requests_end_marker))
    } :requests

    . = ALIGN(CONSTANT(MAXPAGESIZE));

    .text : {
        *(.text .text.*)
    } :text

    . = ALIGN(CONSTANT(MAXPAGESIZE));

    .rodata : {
        *(.rodata .rodata.*)
    } :rodata

    . = ALIGN(CONSTANT(MAXPAGESIZE));

    .data : {
        *(.data .data.*)
    } :data

    .bss : {
        *(.bss .bss.*)
        *(COMMON)
    } :data

    /DISCARD/ : {
        *(.eh_frame)
        *(.note .note.*)
        *(.comment)
    }
}
