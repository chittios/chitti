# Chitti OS

**An agentic operating system, built from scratch, where the agent is the driver.**

Chitti is a bare-metal OS (x86_64 and aarch64) whose fundamental unit of
execution is an **AI agent**, not a compiled binary. There is no
`exec(binary) → trap into syscalls`. Instead:

```text
spawn(agent) → plan over capabilities → a deterministic executor runs the primitives
```

A tiny language model runs on the CPU, on the metal, with no host OS underneath.
Everything the model wants to *do* flows through a capability-checked, audited,
grammar-constrained ABI. Model output is an untrusted plan; it never causes a
side effect directly.

> [!WARNING]
> **Chitti is under active development and is NOT stable.** It is a research
> operating system. Interfaces, on-disk formats, and behaviour change without
> notice; it may crash, corrupt data, or fail to boot. **Run it in a VM, not on
> hardware you care about.** It is provided "as is", with no warranty of any
> kind — use it at your own risk. The authors are not responsible for any damage
> or data loss.

## Why?

Why reinvent the wheel? We're not. Chitti is deliberately **not** "Unix in Rust":
**no POSIX, no libc, no ELF loader, no shell scripts, no re-implementation of the
1970s.** The interesting question isn't how to run `/bin/sh` again — it's what an
operating system looks like when an *agent* is the native thing you run. So the
whole system is arranged around that: the agent plans, the OS gives it safe,
first-class primitives, and a hard determinism boundary keeps the stochastic
model from ever directly touching hardware.

## Features

- **Agent as the process.** An orchestrator runs a real tool-use loop
  (`model → tool → result → repeat`) and can delegate to isolated **sub-agents**
  whose authority is a strict subset of the parent's.
- **A shell that is an agent.** Plain text is a message to the model; `/`-commands
  drive the OS. First-class **sessions** save, resume, and fork.
- **CPU-only inference on bare metal.** A quantized Qwen3.5 (0.8B or 9B) runs on
  hand-written SIMD tensor kernels — SSE2/AVX2 on x86, NEON on aarch64 — with a
  zero-copy GGUF loader and an OS-managed KV cache. No GPU, no host runtime.
- **The capability ABI (Synapse).** The model emits grammar-constrained,
  MCP-shaped tool calls; every one is capability-checked, taint-gated, executed
  by deterministic native code, and written to an append-only audit log.
- **Prompt-injection defense at the OS boundary.** Provenance/taint tags follow
  every token; a destructive action justified by untrusted, ingested content is
  refused or requires explicit human confirmation.
- **Skills.** Portable, **signed**, permissioned packages of procedural knowledge
  (+ optional tools and agent roles), loaded with progressive disclosure and
  bounded forever by their install-time capability grant.
- **A real windowed console.** A tmux-style split-pane framebuffer compositor in
  the Geist Mono font: a chat pane, an on-demand action pane (live ktrace or a
  vim-like editor), a status bar with a **live clock**, a **blinking caret**, a
  **todo-list**-driven planner, **mouse** (cursor, click, drag-select/copy) and
  **keyboard** input, with an editable on-disk UI config.
- **Dual-architecture, one codebase.** `x86_64` and `aarch64` from the same tree,
  behind a small `arch` facade; aarch64 runs natively on Apple Silicon under
  QEMU-HVF. Functionality never diverges between the two.
- **Real, standards-based drivers — nothing hardcoded to an emulator.** Display
  via UEFI GOP / Limine / ramfb; disks via virtio / NVMe / AHCI over discovered
  PCIe; input via USB xHCI/HID + virtio-input + PL050/PS-2; discovery via
  ACPI/ECAM, fw_cfg, HID report descriptors. The same image runs on QEMU,
  VirtualBox, and real UEFI hardware.
- **Storage & self-install.** GPT/MBR/FAT/ext4 detection, an ext4/FAT/SimpleFS
  stack, durable agent state, and `/install` to a real disk that boots
  standalone via UEFI.
- **Microkernel core.** Unforgeable capabilities (seL4-inspired), capability-gated
  IPC, a cooperative + timer-preemptive scheduler, SMP, and a hand-rolled MMU.

## Building & running

Everything goes through `cargo xtask`; the target architecture is always
explicit. In short:

```sh
cargo xtask test                        # in-kernel test suite under QEMU (x86_64) — no model needed
cargo xtask run -arch aarch64           # boot natively on Apple Silicon (QEMU + HVF)
cargo xtask run -arch x86_64            # boot the Limine image under QEMU
```

Full prerequisites, the model fetch, per-arch/VirtualBox/real-hardware boot, and
headless framebuffer verification are in **[DEVELOPMENT.md](DEVELOPMENT.md)**.

## Contributing

Contributions are welcome — see **[CONTRIBUTING.md](CONTRIBUTING.md)** for the
workflow, the coding conventions, and the two standing rules every change must
honour (dual-arch parity and real-hardware drivers). Security issues: please read
**[SECURITY.md](SECURITY.md)**.

## License

Chitti OS is licensed under the **GNU General Public License v3.0** — see
[LICENSE](LICENSE).

The bundled font is **Geist Mono** (© Vercel), used under the SIL Open Font
License 1.1 — see [`assets/`](assets/). Bundled/fetched model weights are the
property of their respective authors under their own licenses.
