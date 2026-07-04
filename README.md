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

## Try it (real hardware or a VM)

You don't need the toolchain to run Chitti — grab a prebuilt image, or build one.

1. **Get an image.** Download the latest from
   [**Releases**](https://github.com/chittios/chitti/releases), or build one
   locally (see below):
   - **aarch64** (Apple Silicon / ARM, UEFI): `chitti-aa64.img` — a full GPT disk image.
   - **x86_64** (PC, BIOS or UEFI): `chitti.iso` — a hybrid boot ISO.

2. **Write it to a USB stick.**
   - **[balenaEtcher](https://etcher.balena.io/)** (easiest, cross-platform):
     select the `.img`/`.iso`, select your USB drive, Flash.
   - **`dd`** (macOS/Linux) — ⚠ this **erases** the target device; triple-check
     the name:
     ```sh
     # macOS:  diskutil list  →  find /dev/diskN, then:
     diskutil unmountDisk /dev/diskN
     sudo dd if=chitti-aa64.img of=/dev/rdiskN bs=4m         # rdisk = raw, faster
     # Linux:  lsblk  →  find /dev/sdX, then:
     sudo dd if=chitti-aa64.img of=/dev/sdX bs=4M status=progress conv=fsync
     ```

3. **Boot it.**
   - **Real hardware:** plug in the USB, open the firmware boot menu (often F12 /
     F2 / Del / Option at power-on), and pick the USB stick — in **UEFI** mode for
     the aarch64 image.
   - **A VM** (VirtualBox / UTM / QEMU): create a VM with **EFI enabled**, attach
     the image as its disk (an *ARM* VM for `chitti-aa64.img`, an *x86* VM for the
     ISO). In VirtualBox, set the pointing device to **USB Tablet** and the
     keyboard to **USB** so the USB-HID drivers pick them up. On macOS,
     `make vbox` does the VirtualBox setup for you (see DEVELOPMENT.md).

Once booted you land in the chat shell — type a message, or `/help` for commands.

> Reminder: this is research software — **boot it from removable media / in a VM,
> not on a machine whose data you care about.** See the warning above.

## Building & running (development)

Everything goes through `cargo xtask`; a `Makefile` wraps the common commands.
The target architecture is always explicit. In short:

```sh
make test                 # in-kernel test suite under QEMU (x86_64) — no model needed
make run  ARCH=aarch64    # boot natively on Apple Silicon (QEMU + HVF)
make image ARCH=x86_64    # assemble a bootable ISO under target/
make help                 # list all targets

# …or call the underlying tool directly:
cargo xtask run -arch aarch64
```

`make vbox` rebuilds the aarch64 image and reloads it into a VirtualBox VM.
Full prerequisites, the model fetch, per-arch boot, VirtualBox, and headless
framebuffer verification are in **[DEVELOPMENT.md](DEVELOPMENT.md)**.

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
