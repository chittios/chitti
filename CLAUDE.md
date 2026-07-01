*"This project is specified in `CHITTI_OS_HANDOFF.md`. Read it fully before acting. Follow the locked decisions in Part 2 and the guardrails in Part 4. Work one phase at a time; do not start the next phase until the current phase's acceptance criteria pass in QEMU."*

**Current phase: 1 (Deterministic microkernel core) — complete.**
GDT+TSS (IST1 for double-fault), IDT + CPU exception handlers, the legacy 8259 PIC
remapped with a 1kHz PIT timer (IRQ0) and keyboard (IRQ1), FPU/SSE CR0/CR4/XSAVE
init + `EFER.NXE`, a bitmap physical frame allocator built from the Limine memory
map, a page-table walk/map extending Limine's 4-level paging, a linked-list kernel
heap, and the `ktrace` logging framework are all in and exercised by
`custom_test_frameworks`. `cargo xtask test` runs 7 in-kernel tests (up from Phase
0's 4): timer ticks advance, heap alloc/free of varied sizes survives intact, and a
deliberately triggered breakpoint exception is caught/reported without a triple
fault. Next: Phase 2 (execution substrate — tasks, scheduler, capabilities, IPC).
