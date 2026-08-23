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
  /* Writable data starts on a page boundary so the loader can map everything below it
   * read-execute and everything above it read-write. Without the split a single mutable
   * static lands in the code page, and the tenant faults on its own first instruction
   * writing it -- which is exactly what happened (`error=0x7`, write to a present RO page).
   *
   * `__rw_start`/`__rw_end` are read back with `llvm-nm`; nothing here is duplicated in the
   * build tool. `.bss` sits inside the range but not in the flat binary, so the loader zeroes
   * the tail rather than copying it. */
  . = ALIGN(4096);
  __rw_start = .;
  .data : { *(.data .data.*) }
  .bss : { *(.bss .bss.*) *(COMMON) }
  . = ALIGN(4096);
  __rw_end = .;
  /DISCARD/ : { *(.eh_frame) *(.comment) *(.note*) }
}
