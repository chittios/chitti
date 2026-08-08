# ChittiOS

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

<!-- -->

> [!IMPORTANT]
> **On real hardware, expect a wired-network console OS.** ChittiOS is developed
> and verified against QEMU, VirtualBox and UTM; several drivers are written from
> a specification and have never met the silicon they target. In particular:
> **there is no WiFi driver that can join a network**, Bluetooth will not pair
> with modern devices (no SSP, no BLE), audio plays in mono, there is no screen
> brightness control, and suspend/resume does not restore USB or disk.
> **[HARDWARE.md](HARDWARE.md) is the full support matrix** — what works, what is
> unverified, and what is absent, per subsystem. Read it before booting on
> anything you care about.

## Why?

Why reinvent the wheel? We're not. Chitti is deliberately **not** "Unix in Rust":
**no POSIX, no libc, no ELF loader, no shell scripts, no re-implementation of the
1970s.** The interesting question isn't how to run `/bin/sh` again — it's what an
operating system looks like when an *agent* is the native thing you run. So the
whole system is arranged around that: the agent plans, the OS gives it safe,
first-class primitives, and a hard determinism boundary keeps the stochastic
model from ever directly touching hardware.

## Features

### Agent & security core

- **Agent as the process.** An orchestrator runs a real tool-use loop
  (`model → tool → result → repeat`) and can delegate to isolated **sub-agents**
  (roles such as explore / plan / worker / reader) whose authority is a strict
  subset of the parent's.
- **A shell that is an agent.** Plain text is a message to the model; `/`-commands
  drive the OS. First-class **sessions** save, resume, and fork; **todos** drive
  multi-step work in a live action pane.
- **Plan mode & permissions.** `/mode manual|auto|bypass|plan` and
  `/permissions` (allow / ask / deny patterns in
  `/configs/core/permissions.json`) gate agent tool calls before the modal.
- **CPU-only inference on bare metal.** Architecture-dynamic GGUF loaders
  (Qwen hybrid, Gemma4, …), mainstream GGML quants, hand-written SIMD tensor
  kernels (SSE2/AVX2 / NEON+SDOT), OS-managed KV/recurrent cache. Select at boot
  or with `/model load` / `/model remote` (hosted OpenAI-compatible servers).
- **The capability ABI (Synapse).** Grammar-constrained, MCP-shaped tool calls;
  capability + **scope** gate (fs path globs, host/port ranges); taint gate;
  deterministic executor; append-only audit log.
- **Prompt-injection defense at the OS boundary.** Provenance tags follow every
  token; destructive work justified only by untrusted content is refused or
  requires human confirmation.
- **Skills & installable agents.** Signed, progressive-disclosure skills; agents
  as markdown packages (SOUL + skills + manifest). Install from a **public
  registry** (`/agents search`, `/agents install`) with consent and **ECDSA
  P-256** publisher trust. Per-agent home sandbox: `/agent/<id>/**`.

### Messaging & IPC

- **Messaging channels (Telegram first).** OpenClaw-style external inboxes:
  `/channel add telegram …`, pairing / allowlist, poll over HTTPS Bot API,
  inbound → shell agent → reply. Config at `/configs/core/channels.json`.
  **See [CHANNELS.md](CHANNELS.md)** for full setup and how to add backends
  (Discord, Slack, …).
- **Inter-agent channels.** Capability-gated **byte-stream and datagram IPC**
  between agents (and TCP handoff for service pipelines) — distinct from
  messaging channels above.
- **MCP client.** `/mcp connect` registers remote tools as `mcp__…` for the
  agent; manifests can declare MCP servers at install time.

### Filesystem & shell UX

- **Linux-like store FS.** Hierarchical `/ls` (not a flat dump of every key),
  `/cat`, `/mkdir`, `/cp`, `/mv`, `/rm`, `/touch`, `/glob`, `/grep`, `/pwd` over
  the Synapse path store (and mount-aware reads for media/models).
- **Composer UX.** Bordered input box; **slash-command** and **`@file`**
  suggestion menus (↑↓ / Tab / Enter); **Commands** browser for `/help`
  (search + scroll + fill the composer on select).
- **Host clipboard bridge.** OSC 52 copy-out and bracketed paste so the guest
  clipboard syncs with the host terminal.

### UI, media & services

- **Windowed console.** Tmux-style resizable chat | action split, tabbed action
  pane (ktrace, **htop-style `/top`**, todos, editor, image / audio / video),
  mouse + keyboard, selection, syntax highlighting, live clock, Geist Mono,
  fully themable ([DESIGN.md](DESIGN.md); see **Themes** below).
- **Themes.** `/theme set <name>` switches bundled presets (`dark`, `light`,
  `solarized-dark`, `nord`, `dracula`, `ubuntu`) — or ones you `/theme save` /
  `/theme install <url>`. A theme is pure-data JSON: the chrome palette, code
  **syntax colours** (VSCode-style), **cursor** colour + optional custom sprite
  bitmaps, **font** + size, and a **wallpaper**
  (`/theme wallpaper <none|gradient:#a,#b|/path|https://url>` — fetches + cover-
  scales an image) with adjustable **opacity / transparency** (`/theme opacity`).
  Chat, editor, and every TUI surface blend over the wallpaper.
- **Web browser.** `/browse <url>` opens a real in-kernel browser agent in the
  action pane: HTML + CSS (flow / flexbox) layout, paint, forms, SVG, images, a
  JS engine, cookies / localStorage, CORS, and history
  (`browser_open`/`navigate`/`back`/`scroll`/`click`/`links`/`text`). Renders
  with the full in-kernel font stack (Latin + Indic + CJK + colour-emoji
  fallback).
- **Service agents & web pipeline.** Native **network / http / server** stages;
  content agents (e.g. **doc**) are data + SOUL, not per-site Rust.
  `/agents start <name> [port]`; **ssh** agent for version exchange (transport
  evolving).
- **Sound & voice.** virtio-snd / HDA (and legacy AC'97 / SB16); `/voice`
  (VAD → STT → LLM → TTS) with ONNX models (silero, parakeet, KittenTTS).
  Playback is 16-bit **mono** — the voice pipeline is the design target.
- **Media.** In-kernel PNG/JPEG; WAV/MP3/AAC player; H.264 / H.265 / VP9 video
  player (`/open .mp4|.mov|.mkv|.webm|.ts|.m3u8`, including HLS VOD over the
  network) with streaming decode and transport controls.
- **Networking.** Full TCP/IP (smoltcp): DHCP, DNS, ping, loopback, HTTP(S)
  client, WebSocket client, downloads to `/downloads/`.

### Platform

- **Dual-architecture, one codebase.** `x86_64` and `aarch64`; aarch64 runs
  natively on Apple Silicon under QEMU-HVF. Features do not diverge by arch.
- **Real, standards-based drivers.** UEFI/Limine/GOP display; virtio/NVMe/AHCI
  disks; USB xHCI/HID + virtio-input + PL050/PS-2; PCIe/ACPI discovery. Same
  image for QEMU, VirtualBox, and real UEFI hardware. **Coverage is uneven on
  bare metal — see [HARDWARE.md](HARDWARE.md) for the per-subsystem matrix,
  including what is written-but-unverified and what is absent (WiFi, Bluetooth
  pairing, GPU, backlight).**
- **Storage & self-install.** GPT/MBR/FAT/ext4; durable agent state on ext4;
  `/install` update-in-place with modal confirm.
- **Microkernel core.** Unforgeable capabilities, scheduler (cooperative +
  preemptive), SMP bring-up, MMU, heap.
- **Bare-metal Apple Silicon (work in progress).** Boots on a real **Mac mini
  M2** via the Asahi [m1n1](https://github.com/AsahiLinux/m1n1) bootloader —
  shell UI on the display, model-via-initrd, and native USB keyboard + mouse on
  the USB-C ports. See [DEVELOPMENT.md](DEVELOPMENT.md#real-mac-mini-apple-silicon-via-m1n1--work-in-progress).

## Messaging channels (short path)

Full guide: **[CHANNELS.md](CHANNELS.md)**.

```text
# 1. Create a bot with @BotFather → copy the token
# 2. On a networked Chitti:

/channel add telegram home <BOT_TOKEN> pairing
/channel start home

# 3. DM the bot; note the pairing code it sends
/channel pair home <CODE>

# 4. Chat — messages go to the shell agent; replies return on Telegram
# Console helpers:
/channel send home <chat_id> Hello from Chitti
/channel reply home Got it
/channel status
```

To add another platform (Discord, Slack, …), see
[Adding a new channel backend](CHANNELS.md#adding-a-new-channel-backend) —
extend `msgchan::Kind`, implement poll/send, wire match arms; the `/channel`
CLI stays the same.

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

Once booted you land in the chat shell — type a message, or `/help` for the
Commands browser (searchable). `/help text` prints a flat list over serial.

> Reminder: this is research software — **boot it from removable media / in a VM,
> not on a machine whose data you care about.** See the warning above.
>
> Booting on real hardware? Check **[HARDWARE.md](HARDWARE.md)** first. The short
> version: console, keyboard, trackpad and wired Ethernet work; **WiFi does not**,
> Bluetooth will not pair, audio is mono, and there is no brightness control.

## Building & running (development)

Everything goes through `cargo xtask`; a `Makefile` wraps the common commands.
The target architecture is always explicit. In short:

```sh
make test                 # in-kernel unit suite under QEMU (x86_64) — pure logic, no model
make e2e                  # end-to-end: boot the real kernel, drive the shell over serial
make run  ARCH=aarch64    # boot natively on Apple Silicon (QEMU + HVF)
make image ARCH=x86_64    # assemble a bootable ISO under target/
make help                 # list all targets

# …or call the underlying tool directly:
cargo xtask run -arch aarch64
```

Two test layers back every change: **unit tests** (`cargo xtask test`) for the
pure logic, and **end-to-end scenarios** (`make e2e`) that boot the real kernel
and exercise shell commands plus networked, agent-install, service, and UI flows.
New features add to both.

`make vbox` rebuilds the aarch64 image and reloads it into a VirtualBox VM.
Full prerequisites, the model fetch, per-arch boot, VirtualBox, and headless
framebuffer verification are in **[DEVELOPMENT.md](DEVELOPMENT.md)**.

## Documentation

| Doc | Contents |
|---|---|
| **[HARDWARE.md](HARDWARE.md)** | Real-hardware support matrix: what works, what is unverified, what is absent |
| **[CHANNELS.md](CHANNELS.md)** | Messaging channels: Telegram setup, `/channel` reference, adding backends |
| **[DEVELOPMENT.md](DEVELOPMENT.md)** | Toolchain, build/run/test, VirtualBox |
| **[DESIGN.md](DESIGN.md)** | Brand, palette, compositor UX |
| **[CLAUDE.md](CLAUDE.md)** | Invariants & standing rules for work in-tree |
| **[CONTRIBUTING.md](CONTRIBUTING.md)** | Contribution workflow |
| **[SECURITY.md](SECURITY.md)** | Security reporting |

## Contributing

Contributions are welcome — see **[CONTRIBUTING.md](CONTRIBUTING.md)** for the
workflow, the coding conventions, and the two standing rules every change must
honour (dual-arch parity and real-hardware drivers). Security issues: please read
**[SECURITY.md](SECURITY.md)**.

## License

ChittiOS is licensed under the **Apache License 2.0** — see
[LICENSE](LICENSE).

The bundled font is **Geist Mono** (© Vercel), used under the SIL Open Font
License 1.1 — see [`assets/`](assets/). Bundled/fetched model weights are the
property of their respective authors under their own licenses.
