# Chitti OS

A bare-metal x86_64 operating system whose fundamental unit of execution is an
**AI agent**, not a compiled binary. Instead of `exec(binary) → trap into
syscalls`, Chitti does `spawn(agent) → plan over capabilities → deterministic
executor runs the primitives`.

CPU-only inference (no GPU), a tiny (0.5B–1.1B, Q4_0/Q8_0) local model loaded
as a boot module, and a hard determinism boundary: model output is an
untrusted plan that is always parsed, grammar/capability-checked, and only
then executed by deterministic native code.

The full design — locked technical decisions, vocabulary, phase specs,
guardrails — lives in [`CHITTI_OS_HANDOFF.md`](CHITTI_OS_HANDOFF.md). That
document is the single source of truth; this README is a map on top of it.
[`CLAUDE.md`](CLAUDE.md) tracks the current phase for whoever (human or
agent) picks up work next.

## Status: Phase 1 complete

```
[x] 0  Boot & harness          Boot via Limine, print to serial + framebuffer, QEMU test exit codes
[x] 1  Microkernel core        Interrupts, memory (frames/paging/heap), timer, keyboard, ktrace
[ ] 2  Execution substrate     Tasks, scheduler, capabilities, IPC, async executor
[ ] 3  Cortex (inference)      CPU transformer forward pass on a tiny GGUF, KV cache, seeded sampling  — highest risk
[ ] 4  Synapse (capability ABI) Grammar-constrained tool calls → capability-checked primitives + audit log
[ ] 5  Persona + shell         Agents as processes, two-tier memory, intent shell drives plan→act loop
[ ] 6  Differentiators         Taint/provenance security gating + self-compiling agents (compiled intents)
[ ] 7  Stretch                 SMP, APIC-per-core, framebuffer TUI, larger model, RISC-V port
```

See `CHITTI_OS_HANDOFF.md` Part 5/6 for the full goal/scope/acceptance
criteria of each phase. A phase's acceptance gate must pass in QEMU before
the next one starts.

## Architecture / folder structure

```
chitti/
├── CHITTI_OS_HANDOFF.md      # source of truth: mission, locked decisions, phase specs
├── CLAUDE.md                 # points at the handoff + current-phase line
├── README.md                 # this file
├── DEVELOPMENT.md            # build/run/test instructions, troubleshooting
├── rust-toolchain.toml       # pinned nightly + rust-src, llvm-tools-preview
├── Cargo.toml, .cargo/       # host workspace (xtask only — see DEVELOPMENT.md)
├── targets/
│   └── x86_64-chitti.json    # custom bare-metal target (derived from x86_64-unknown-none)
├── xtask/                    # build orchestration: assembles the Limine image, drives QEMU
│   ├── src/main.rs           #   build | image | run | test | runner subcommands
│   └── run-test-in-qemu.sh   #   shell shim wired as the kernel's `cargo test` QEMU runner
└── kernel/                   # the OS itself — standalone crate, not part of the host workspace
    ├── linker.ld             # higher-half layout, Limine .requests section placement
    ├── limine.conf           # bootloader config (one boot entry, protocol: limine)
    ├── .cargo/config.toml    # target + -Z build-std + the QEMU test runner
    └── src/
        ├── main.rs           # real boot entry: _start, panic handler, boot banner
        ├── lib.rs             # shared code, init() bring-up sequence, custom_test_frameworks harness
        ├── ktrace.rs          # deterministic sequence-numbered logging ("strace" equivalent)
        ├── limine_protocol.rs # hand-rolled Limine boot-protocol requests/responses
        ├── serial.rs          # COM1 16550 UART driver + serial_print!/serial_println!
        ├── framebuffer.rs     # 8x8 bitmap text renderer onto the Limine framebuffer
        ├── qemu.rs            # isa-debug-exit wiring for the test harness
        ├── mm/                # frame allocator (memmap-backed bitmap) + linked-list kernel heap
        └── arch/x86_64/       # arch-specific code lives only here: GDT/TSS, IDT + exceptions,
                                #   PIC/PIT/keyboard IRQs, FPU/SSE + XSAVE init, 4-level paging
```

Everything x86_64-specific stays under `kernel/src/arch/x86_64/`, so a future
RISC-V port (Phase 7 stretch) only touches that directory. As later phases
land, `kernel/src/` grows the `sched/`, `cap/`, `ipc/`, `cortex/`, `synapse/`,
`persona/`, `shell/`, and `security/` modules described in
`CHITTI_OS_HANDOFF.md` Part 3.

## Quick start

```sh
cargo xtask run     # boot the kernel in QEMU (serial to stdio, text in the framebuffer window)
cargo xtask test    # run the in-kernel test suite under QEMU
```

See [`DEVELOPMENT.md`](DEVELOPMENT.md) for prerequisites, what each `xtask`
command actually does, and troubleshooting notes for the build-std/Limine/QEMU
rough edges this project ran into.
