*"This project is specified in `CHITTI_OS_HANDOFF.md`. Read it fully before acting. Follow the locked decisions in Part 2 and the guardrails in Part 4. Work one phase at a time; do not start the next phase until the current phase's acceptance criteria pass in QEMU."*

**Current phase: 0 (Boot & harness) — complete.**
`cargo xtask run` boots via Limine and prints "Chitti: boot ok" to serial and the framebuffer.
`cargo xtask test` runs 4 in-kernel `custom_test_frameworks` tests and exits QEMU via
isa-debug-exit with the success code. Next: Phase 1 (deterministic microkernel core).
