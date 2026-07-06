# Chitti OS — guide for agents & humans working in this repo

**Chitti is an agentic operating system: the agent is the driver.** The
fundamental unit of execution is an AI agent, not a compiled binary. There is no
`exec(binary) → trap into syscalls`; instead `spawn(agent) → plan over
capabilities → a deterministic executor runs the primitives`. A tiny language
model runs on the CPU, on bare metal, and every effect it wants flows through a
capability-checked, audited, grammar-constrained ABI.

**No POSIX. No libc. No ELF loader. No re-implementing Unix.** We are not
building "Linux but in Rust." Why reinvent the wheel? The interesting question is
what an OS looks like when an agent — not a shell script or a C program — is the
thing you run. So the OS gives an agent first-class primitives: a **shell agent**
(chat REPL), **sessions** (save/resume/fork), **sub-agents** (isolated,
capability-attenuated delegation), **tools** (MCP-shaped, Synapse-backed),
**skills** (signed, permissioned, progressively-disclosed packages), a **todo-list
TUI**, and a real windowed console with **mouse + keyboard**. The classic OS
plumbing that must exist (scheduler, MMU, capabilities, IPC, block/FS, drivers)
exists to serve that, not to run `/bin/sh`.

## The load-bearing invariants — never break these

1. **The determinism boundary.** Model output is an *untrusted plan*. It never
   causes a side effect directly — it is parsed, grammar-validated, capability-
   checked, and only then executed by deterministic native code (`Synapse`).
   Above the boundary: stochastic (the agent/model). Below: deterministic.
2. **All effects route through Synapse.** Tools are a validation/presentation
   layer over Synapse primitives; they never touch hardware or memory directly.
3. **Delegation only ever narrows authority.** A sub-agent's capability set is a
   strict subset of its parent's. Spawning can attenuate, never widen.
4. **Provenance/taint gating.** Every context token is tagged
   (`user_typed` / `system_trusted` / `untrusted_ingested` / `skill_installed`).
   A destructive primitive justified by untrusted-ingested content is refused or
   requires explicit human confirmation — prompt-injection-as-privilege-
   escalation defense enforced at the OS boundary.
5. **A skill is bounded by its install-time grant, forever.** Even if a skill's
   text says "delete everything," it can only act within the capabilities the
   human approved at install.

Capabilities are unforgeable (opaque per-task indices, no ambient authority).
Effective authority is always `intersection(requested, granting-context)`.

## STANDING RULE — dual-architecture, no divergence

The kernel is **one codebase for two architectures: `x86_64` and `aarch64`**, and
functionality must not diverge between them. Every change must build and work for
**both** arches. Never gate behaviour behind `target_arch` unless it is genuinely
arch-specific (a driver, an instruction) — and then provide the equivalent for the
other arch behind the **same API**, never a stub that drops a feature. If a
capability exists on one arch, it exists on the other.

After any change, verify both:

- `cargo xtask build -arch x86_64` **and** `cargo xtask test` (the in-kernel
  unit suite — keep it green; add cases for new logic)
- `cargo xtask build -arch aarch64` (and boot it via `cargo xtask run -arch aarch64`
  when the change is boot-visible)

## STANDING RULE — every feature/fix ships with tests

Two layers, and new work adds to **both** where they apply:

1. **Unit tests** (`cargo xtask test`, x86 under QEMU, no model/hardware) for
   the **pure logic** — parsers, decoders, codecs, format/build functions,
   capability math. The pattern that keeps us safe: pull the fiddly logic out
   of the hardware/IO path into a pure function and test it with cases (e.g.
   `mouse::decode_ps2_packet`, `xhci::parse_report_layout`, `net::http`
   `parse_url`/`dechunk_partial`, `shell::parse_tool_call`,
   `ws::{base64,sha1,encode_frame}`, `channel` ring/EOF, `synapse::ui` draw-op
   raster + ownership, `service::http::parse_request`, `crypto::verify_p256`,
   the executor scope gate, `registry_client::parse_index`). If logic broke
   before, it gets a test.

2. **End-to-end tests** (`tests/e2e/`, `make e2e` / `make e2e-full`) for
   anything that only exists **on the running OS** — a shell command, a
   network/TLS/WebSocket exchange, a model or voice flow. The harness boots the
   real kernel under QEMU and drives the shell over serial. **Adding a shell
   command or a networked/model/voice feature means adding an e2e scenario**
   (`os`/`agents`/`net` groups always run; `model`/`voice` are `--slow`, gated
   on assets). The `agents` group covers install/consent, service lifecycle,
   the network + HTTP/Doc service agents (the host reaches guest listeners via
   an opt-in slirp `CHITTI_HOSTFWD` port), registry search/install, and UI
   surfaces. A fix for something the harness could have caught gets a scenario
   too. Run `make e2e` before shipping boot-visible or networked changes.

CI (`.github/workflows/ci.yml`) runs both on every push/PR: `unit` builds both
arches + `cargo xtask test`; `e2e` boots the kernel and runs the os+net groups.
It is **fork-PR-safe** — `pull_request` (never `pull_request_target`),
`contents: read`, no secrets, GitHub-hosted runners only. Keep it that way.

## STANDING RULE — performance: know the three traps before optimizing

Hard-won findings (commits `f2bd8f7`, `06a62b4`) that apply to **any new
compute- or I/O-heavy feature**. Measure first, always — every one of these was
found by profiling, and the obvious suspect was wrong every time.

1. **`+strict-align` silently scalarizes NEON.** The aarch64 target builds with
   `+strict-align` (required for the pre-MMU boot window and device MMIO).
   Under it, LLVM lowers *any* unaligned vector load/store — the `vld1q_*`
   intrinsics **and auto-vectorized plain loops** — into a 16×`ldrb`+`orr`+
   stack-spill byte-assembly: ~25 instructions per load, ~100× slower. The
   binary still contains `fmla`, so it looks fine until you disassemble the
   *loads*. Hot SIMD loops must do their memory access via **inline asm**
   (`ldr q`/`ldp q`/`stp q` — correct at runtime: Normal cacheable RAM,
   `SCTLR_EL1.A=0`), the same pattern as the MMIO single-`ldr`/`str` rule.
   Reusable kernels live in `cortex/tensor.rs`: `dot_f32`, `axpy_f32`,
   `scale_f32`, the `ldq_s8/u8/f32` load helpers, and the Q8_0/Q4_0 SDOT
   matvecs. **Prefer composing these** over writing new intrinsics code; if
   you must write a new SIMD loop, verify with `objdump -d` (count `ldrb` in
   the hot function) and `/onnx bench` in the booted kernel (dot_f32 ≥ 10
   GMAC/s under HVF; ~1 GMAC/s means the disease is back).
2. **Batch block I/O or die waiting.** One `read_block` = one polled virtio
   round trip (~0.5 ms). Anything reading more than a few KiB must use
   `read_blocks` (all drivers implement multi-sector requests; `Partition`
   forwards). FAT `read_file` coalesces contiguous cluster runs — loading the
   131 MB parakeet model went from ~2.5 min to ~2 s. New filesystem or loader
   code must follow the same pattern (and cache FAT-chain sectors).
3. **The kernel allocator punishes churn.** First-fit linked list: cloning
   multi-MB tensors per node, or cloning a whole env per loop iteration, is
   real time. Borrow instead of clone on hot paths; move values into maps
   instead of cloning them. Heap pressure counters exist
   (`mm::heap::alloc_stats()` — allocs + free-list scan steps).

Layout matters as much as SIMD: iterate state **row-wise/contiguous** (the
DeltaNet delta rule went from stride-512B scalar walks to contiguous AXPYs —
that alone nearly doubled tokens/sec). Per-op wall-time accounting for the
ONNX executor prints one `onnx: op time:` ktrace line per run; `/perf` gives
prefill/decode tok/s; `tools/onnxdiff/` runs the kernel's own ONNX interpreter
natively on the host (seconds per iteration, layer-by-layer diff vs
onnxruntime) — use it for any voice-model numeric or perf work before touching
QEMU.

Current figures (aarch64 HVF, 0.8B Q8): prefill 53 tok/s, decode 19 tok/s,
`/voice stt` ~2 s, `/voice say` ~14 s for 3.5 s of audio. Decode is
compute-bound on the **single-core** SDOT matvec; llama.cpp's remaining edge
on the same silicon is threading. The row-range kernels
(`matvec_q8_0_sdot_rows`) are ready to split across cores — the missing piece
is aarch64 AP bring-up (PSCI `CPU_ON`) + a work-distribution primitive; x86
APs already boot and park (`smp.rs`). That is the designated next perf step;
prefer it over further single-core micro-tuning.

## STANDING RULE — real hardware, nothing hardcoded to an emulator

Drivers must target **real, standards-based hardware**, not QEMU or VirtualBox
quirks. Do not hardcode addresses, resolutions, device layouts, or behaviour to a
specific emulator/hypervisor. Discover hardware the way real firmware does
(ACPI/PCIe ECAM, UEFI GOP, fw_cfg, HID report descriptors, PrimeCell IDs, EDID/
mode tables) and degrade gracefully when a facility is absent. A feature that only
works under QEMU is not done.

Concretely: display comes from the firmware (Limine GOP on x86, UEFI GOP via the
`stub/` bootloader on aarch64, QEMU ramfb as a fallback); disks via virtio /
NVMe / AHCI over discovered PCIe; input via USB xHCI/HID (keyboard **and** mouse,
report-descriptor-driven), virtio-input, and PL050/PS-2; the wall clock from the
RTC / UEFI `GetTime` / the virtual counter — each behind a shared facade with a
per-arch implementation. The same kernel image must run on QEMU, VirtualBox, and
real UEFI hardware.

## What exists today (subsystems, not phases)

- **Agent layer** — an orchestrator running a real tool-use loop
  (`model → tool → result → repeat`, budgeted); isolated sub-agents; a shared
  type contract in `agent/types.rs`.
- **Sessions** (`session/`) — serializable message history + todos + env + caps,
  saved/resumed/forked over the memory store (postcard).
- **Tools** (`tools/`) — MCP-shaped registry → Router → Synapse cap+taint gate;
  builtin toolset; provider registration.
- **Skills** (`skills/`) — L0/L1/L2 progressive disclosure, signed install with
  consent + capability subsetting, installable skill-agents. **Agents as
  installable apps**: a package is markdown (SOUL.md persona + `skills/*.md`
  procedures) + a manifest (toolset = permissions, `capabilities`, optional ONNX
  assets) + a signature; installed via `/agents install <name> [--yes]`
  (consent modal → grant only the approved subset → `place_agent_home` lands the
  SOUL/docs in `/agent/<id>/`). A **public registry** (`skills/registry_client.rs`)
  fetches a signed index over HTTP(S) — `/agents search <url> [q]`,
  `/agents install <name> --registry <url>`; registry packages authenticate with
  **ECDSA P-256** (`skills::crypto::verify_p256`, RustCrypto `p256`/`sha2`,
  baked publisher trust store) while local dev/boot packages use the keyed-MAC.
- **Channels** (`channel/`) — cap-gated **byte-stream + datagram IPC** between
  agents (the Linux pipe/socket analog; distinct from `ipc` u64 endpoints).
  `Right::Channel{Read,Write}(ChannelId)` are per-direction ends; a model-emitted
  channel handle is a `Cap` slot in the caller's own table (executor resolves it
  — no ambient authority). Backends: heap ring (`Pipe`), datagram queue, or a
  live TCP socket (`Tcp`, via `adopt_tcp`) so a service agent can hand an
  accepted connection to another agent (`channel_grant`). Synapse primitives
  `channel_create/write/read/close/grant`.
- **Service agents** (`service/`) — long-running native daemons (vs
  request/response reasoning agents): `ServiceSpec {entry serve loop, autostart,
  caps}`, `start`/`stop`/`task_for`/`supervise_tick` (pumped from
  `shell::upkeep`, bounded restarts). Their protocol/codec logic is native,
  deterministic code **below** the determinism boundary — the LLM never
  implements a protocol. Only agents that actually reason from a SOUL are
  **installed agents**: the built-in ones are `doc` and `ssh`, each a markdown
  SOUL plus a JSON manifest under the repo's [`agents/`](agents/) folder, compiled into
  the image via `include_str!`, signed and installed into `/agent/<id>/` at boot
  by `agent::system::install_all` (same permissioned flow as any package,
  pre-trusted; a content agent's assets land in `/agent/<id>/assets/`).
  `network`/`http` are **not agents** — pure mechanical plumbing (relay bytes,
  parse a protocol; no judgment, no SOUL), living entirely in `crate::service`.
  The web is a **generic pipeline** (`service/pipeline.rs`)
  of single-responsibility stages connected by datagram channels — **all reusable
  infrastructure, none app-specific**: `network` (`service/network.rs`) owns the
  socket and relays raw bytes; `http` (`service/http.rs`) parses the request +
  formats the response (no FS, no socket); `server` (`service/server.rs`) is a
  **generic content runtime** that serves *whichever* content agent the pipeline
  was started for — it asks that agent's model (prompted with the agent's own
  `SOUL.md`) which file to serve, then reads it with a capability- and
  scope-gated `mem_fs_read` tool call confined to the agent's own `assets/`.
  So **a web server is just `agents/<name>/{SOUL.md, manifest.json, assets/…}`** —
  the SOUL carries the routing/behaviour (model-planned per request, greedy), the
  assets carry the content, and no per-server Rust is written. `doc` is exactly
  such an agent (data, not code). `/agents start <name> [port]` serves that agent
  over the pipeline; `ssh` runs standalone (RFC 4253 version exchange; transport
  is a stub). `/agents services` lists running stages. Git + full SSH transport
  follow the same native-protocol shape. To add a built-in server agent: drop
  `agents/<name>/{SOUL.md,manifest.json,assets/…}` and register it in
  `agent/system.rs` (one line) — or publish it to the registry.
- **Cortex** (`cortex/`) — CPU transformer inference (Qwen3.5, `-model
  qwen3.5-0.8b|qwen3.5-4b|qwen3.5-9b`); SIMD tensor kernels (SSE2/AVX2 ∣ NEON ∣ scalar behind
  one API); zero-copy GGUF; grammar-constrained sampler; KV/recurrent cache.
- **Synapse** (`synapse/`) — the capability ABI: primitive registry, GBNF-style
  grammar, deterministic executor, append-only audit log, taint gate. Primitives
  now span fs/console/spawn, **channels** (10–14), **net** listen/accept + http
  (15–18), and **UI surfaces** (19–22). The executor runs a **scope gate
  (Gate 2.5)**: a granted narrow scope (an fs path glob, a `Net{host,port}` range)
  is enforced against the concrete target (`scope_target` + `cap::scope_check`) —
  deny-only-when-recorded, so `Scope::Any` grants and un-scoped tasks are
  unaffected. `CapDomain` gained `Channel`/`Net`/`Ui`; `synapse::ui` owns the
  surface registry + bounded draw-op DSL (ownership-gated: an agent can only
  draw to its own surface). NB: sessions that use net egress or UI input aren't
  replayable from a seed alone (the I/O is external) — the audit log records the
  effects; treat such a session as non-deterministic to replay.
- **Microkernel** — tasks + context switch, cooperative + timer-preemptive
  scheduler, unforgeable capabilities, IPC, SMP, frame allocator + heap, MMU.
- **UI** — a tmux-style split-pane framebuffer compositor in Geist Mono (chat
  pane + an on-demand "action" pane hosting the ktrace stream or a vim-like
  editor), a boot splash + status-bar **Synapse-C** brand mark, a live clock, a
  blinking caret, mouse cursor + click, copy/paste, ANSI-coloured agent output, a
  `/`-command shell, and an on-disk UI config (`/configs/core/ui.json`,
  `shortcuts.json`). The brand — logo, the terracotta `#cc785c` / warm-ink /
  cream palette (fully re-themable from `ui.json`), and typography — is specified
  in [DESIGN.md](DESIGN.md); honour it for any UI change. NB: the scheduler is
  cooperative, so **any long or blocking operation must pump the UI itself** —
  call `shell::upkeep()` (blink + status + mouse + `net::poll`) inside its
  loop, exactly as the per-token inference loops, the ONNX per-node loop, and
  the sliced FAT/ext4 readers do; loops that consume their own mouse events
  (modals, the editor) use `shell::status_tick()` instead. A tight compute
  loop that never yields freezes the clock, mouse, and net stack until it
  returns. Any new UI surface or blocking command must keep this upkeep
  running. The chat pane keeps a 2000-line scrollback (PgUp/PgDn; /clear
  wipes it); Shift+Tab / Ctrl+Tab / clicking switches pane focus.
- **Storage** — virtio/NVMe/AHCI block devices, GPT/MBR/FAT/ext4 detection,
  ext4 (the default filesystem) + FAT, `/install` (self-hosting install to a
  disk; detects an existing Chitti GPT and **updates in place**, preserving the
  data partition — destructive actions confirm via the permission modal),
  durable agent state on ext4.
- **Networking** (`net/`) — a full TCP/IP stack on [smoltcp](third_party/smoltcp)
  (vendored in-tree, 0BSD — see [THIRDPARTY-LICENSES.md](THIRDPARTY-LICENSES.md)):
  DHCPv4, static IP, DNS, ICMP (`/ping`), TCP/UDP, plus **loopback**
  (`127.0.0.0/8` + the name `localhost`) so an in-OS client can reach an in-OS
  listener. Loopback is a **second smoltcp interface** (Ethernet-medium
  `phy::Loopback`, `127.0.0.1/8`) with its **own** socket set, polled alongside
  the NIC interface — NOT the same set: sharing one set lets the loopback
  interface's egress steal/drop the NIC sockets' segments. A `TcpHandle` tags
  which set a socket lives in; a connect to a `127/8` address opens its socket in
  the loopback set via that interface's context. A vendored RFC-1122 guard in
  smoltcp `route()` keeps the NIC from ever emitting loopback traffic on the wire.
  NIC drivers behind one
  `NetDevice` facade — **virtio-net** over virtio-mmio (aarch64 QEMU) and over
  PCI, plus **e1000** (VirtualBox default + real Intel) — discovered the same way
  on both arches. Shell surface: `/network` (info/dhcp/static/dns), `/ping`,
  `/wifi` (scan/connect via the password modal), a **TCP listener**
  (`net::listen`/`try_accept`, backed by a pool of Listen-state sockets in
  *both* the NIC and loopback sets, so one listener serves external/hostfwd and
  `localhost` clients alike; accept hands out an Established `TcpHandle` a service
  agent adopts as a channel), `/http` (a curl-like
  HTTP/1.1 client in `net/http.rs` — `-X`/`-H`/`-d`/`-v`/`--stream`, all
  methods, live chunked/SSE streaming; `http://` **and** `https://` via
  `net/tls.rs`/embedded-tls; also the agent's `http` tool), `/ws` (a
  plaintext WebSocket client in `net/ws.rs` — RFC 6455 handshake with
  Sec-WebSocket-Accept verification, masked frames, ping/pong). `/model remote <http://host:port> [name]` points the
  shell agent at a **hosted** OpenAI-compatible model (llama.cpp server /
  Ollama / vLLM), over http or https, via `shell/remote.rs` — same system prompt, tool calls, and
  approval gates; only generation moves off-box (config persisted at
  `/configs/core/model.json`; switching backends is human-only, never an agent
  tool). The stack is polled cooperatively from the shell idle loop. NB: aarch64 MMIO register access must be a single
  `ldr`/`str` (inline asm) — LLVM otherwise coalesces adjacent volatile accesses
  into a paired load HVF can't decode (`hvf: isv`).
- **Sound & voice** (`sound/`, `onnx/`) — virtio-snd PCM in/out (S16 mono,
  poll-driven, descriptor chains) over virtio-mmio (aarch64) and virtio-PCI
  (x86 QEMU), **Intel HDA** for VirtualBox (x86+ARM) and real hardware, plus **AC'97** and **Sound Blaster 16** (x86 legacy; SB16 needs a <16 MiB ISA-DMA buffer, not yet reserved at boot); `/voice` (waveform modal, level-gated utterances) and `/voice test`
  (tone + mic check). `onnx/` is a zero-copy no_std ONNX (protobuf) reader +
  **op interpreter** that runs the real voice models end-to-end: silero-vad v5
  (VAD), parakeet-ctc int8 (STT — `/voice stt <wav>` transcribes), and
  KittenTTS (TTS — `/voice say <text>` speaks); bare `/voice` is the full
  mic → VAD → STT → LLM → TTS conversation loop. Models load lazily from any
  disk volume (bundled in the images; `cargo xtask voice-assets` downloads
  them into `assets/voice/`, gitignored). For any numeric or perf work on this
  path, use `tools/onnxdiff/` (host-side layer-by-layer diff of the kernel's
  own interpreter against onnxruntime) — not QEMU round trips.
- **Agent chat protocol** — the shell chat is an agentic ReAct loop on the
  Qwen3.5 template: the prompt advertises a small CORE tool set plus
  `search_tools` (Claude-Code-style discovery over the registry — manifest
  toolset ∩ `tools::registry`; never hardcode a tool list in a prompt),
  `<tool_call>` JSON in, `<tool_response>` back, thinking off by default
  (`/think`), `/mode manual|auto|bypass` gates agent tool calls through the
  modal, Ctrl+C/Esc cancels prefill *and* decode, `/compact` rebuilds the KV
  from a model-written summary. Agents are processes: `/agents` lists the
  scheduler tasks that carry agent identity, `switch` re-homes the chat,
  `kill` revokes a task's capability table. Every agent has
  `/agent/<id>/{SOUL.md,skills/,memory/}`; SOUL.md is prepended to its system
  prompt. The shell agent is the only default agent (boot demos removed).

## Build / run / test

Everything goes through `cargo xtask`. Arch is chosen explicitly, never
host-detected. See [DEVELOPMENT.md](DEVELOPMENT.md) for the full setup.

```sh
cargo xtask test                       # in-kernel unit suite under QEMU (x86) — pure logic, no model
make e2e                               # end-to-end: boot the kernel, drive the shell over serial,
                                       #   exercise every OS command + the http/https/ws/wss/ping/
                                       #   hosted-model flows vs local servers (tests/e2e/, stdlib-only
                                       #   python; TLS scenarios need a TLS-1.3 python — Homebrew's)
make e2e-full                          # + local inference (/infer,/perf,chat,/compact) and voice
                                       #   (/voice say) — slow; needs assets/model.gguf + assets/voice/
cargo xtask build -arch x86_64|aarch64 # cross-build the kernel
cargo xtask run   -arch x86_64|aarch64 # boot in QEMU (aarch64 = native HVF on Apple Silicon)
cargo xtask image -arch x86_64|aarch64 # assemble a bootable image/ISO
```

## Conventions

- No `unsafe` without an adjacent `// SAFETY:` comment justifying each invariant.
- Every public module has a doc comment stating its responsibility.
- Deterministic by default: tests use fixed seeds + temperature 0; any RNG is seeded.
- `ktrace` every capability invocation and every inference call.
- Commit per sub-milestone with a clear message; never leave the tree non-building.
- Match the surrounding code's style, comment density, and idioms.

## Status

**Active development — not stable.** Interfaces, on-disk formats, and behaviour
change without notice. This is a research OS; run it in a VM. See the caution in
[README.md](README.md).
