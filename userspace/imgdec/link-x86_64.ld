/* Linked at USER_BASE (0x40000000), which is where the loader always maps a tenant.
 *
 * That fixed address is what lets this be an ordinary static binary: no PIC, no
 * relocations, no fixup pass in the kernel. Keep it in step with
 * `mm::space::USER_BASE` -- `entry_matches_user_base` in the kernel asserts they agree,
 * because a silent mismatch here is a tenant that jumps into unmapped memory.
 */
ENTRY(_start)
SECTIONS {
  . = 0x40000000;
  /* _start first, so the flat binary's offset 0 is the entry point. */
  .text : { *(.text._start) *(.text .text.*) }
  .rodata : { *(.rodata .rodata.*) }
  .data : { *(.data .data.*) }
  .bss : { *(.bss .bss.*) *(COMMON) }
  /DISCARD/ : { *(.eh_frame) *(.comment) *(.note*) }
}
