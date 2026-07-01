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

## Status: Phase 4 complete

```
[x] 0  Boot & harness          Boot via Limine, print to serial + framebuffer, QEMU test exit codes
[x] 1  Microkernel core        Interrupts, memory (frames/paging/heap), timer, keyboard, ktrace
[x] 2  Execution substrate     Tasks, scheduler, capabilities, IPC, async executor
[x] 3  Cortex (inference)      CPU hybrid (gated-DeltaNet + attention) forward pass on Qwen3.5-0.8B, KV/recurrent cache, seeded sampling, batching
[x] 4  Synapse (capability ABI) Grammar-constrained tool calls → capability-checked deterministic primitives + append-only audit log
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
        ├── arch/x86_64/       # arch-specific code lives only here: GDT/TSS, IDT + exceptions,
        │                       #   PIC/PIT/keyboard IRQs, FPU/SSE + XSAVE init, 4-level paging
        ├── sched/             # stackful tasks + naked-fn context switch, round-robin scheduler
        │                       #   (cooperative + timer-preemptive), the async executor
        ├── cap/               # unforgeable capability tokens, per-task capability tables
        ├── ipc/               # capability-gated message passing between tasks (seL4-style endpoints)
        ├── cortex/            # CPU inference runtime (Phase 3):
        │                       #   tensor.rs  SSE2 dequant/matvec/rmsnorm/rope/softmax/silu/l2norm kernels
        │                       #   gguf.rs    zero-copy GGUF parser over the Limine boot module
        │                       #   model.rs   Qwen3.5-0.8B hybrid forward pass (gated-DeltaNet + gated attention)
        │                       #   sampler.rs seeded + temperature + grammar-constrained decoding
        │                       #   batch.rs   continuous-batching token scheduler
        └── synapse/           # capability ABI / syscall layer (Phase 4):
                                #   registry.rs  MCP-shaped primitive catalogue (name + typed schema)
                                #   grammar.rs   registry-generated, prefix-closed constraint grammar + ConstrainedDecoder
                                #   executor.rs  grammar → capability → isolated execution, one path to any effect
                                #   fs.rs        in-memory file store the FS primitives mutate
                                #   audit.rs     append-only invocation log (caller, primitive, hashes, outcome)
```

Everything x86_64-specific stays under `kernel/src/arch/x86_64/`, so a future
RISC-V port (Phase 7 stretch) only touches that directory. As later phases
land, `kernel/src/` grows the `synapse/`, `persona/`, `shell/`, and `security/`
modules described in `CHITTI_OS_HANDOFF.md` Part 3.

The Phase 3 model (Qwen3.5-0.8B Q8_0 GGUF) is **not committed** — it's a
~812 MB boot module fetched on demand. Run `xtask/fetch-model.sh` (writes
`assets/model.gguf`); numerics are validated against a NumPy reference
(`tools/ref_qwen35.py`, reconstructed from llama.cpp's `qwen35` graph).

## Quick start

```sh
cargo xtask test        # fast in-kernel test suite under QEMU (no model needed)
xtask/fetch-model.sh    # fetch the Qwen3.5-0.8B GGUF (~812 MB) into assets/model.gguf
cargo xtask run         # boot the kernel + run the inference demo (serial to stdio, framebuffer window)
cargo xtask ref-check   # Phase 3 gate: boot the real model, verify parity/determinism/KV/batching
```

See [`DEVELOPMENT.md`](DEVELOPMENT.md) for prerequisites, what each `xtask`
command actually does, and troubleshooting notes for the build-std/Limine/QEMU
rough edges this project ran into.
