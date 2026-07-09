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
   `scale_f32`, the `ldq_s8/u8/f32` load helpers, and the Q8_0/Q4_0/Q4_K SDOT
   matvecs. **Prefer composing these** over writing new intrinsics code; if
   you must write a new SIMD loop, verify with `objdump -d` (count `ldrb` in
   the hot function — the Q4_K kernel disassembles to 0 ldrb / 32 sdot) and
   `/onnx bench` in the booted kernel (dot_f32 ≥ 10 GMAC/s under HVF;
   ~1 GMAC/s means the disease is back). `/bench` prints the SDOT-vs-exact
   rel-RMS error per fast kernel (`check_q4_0_sdot`, `check_q4_k_sdot`).
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
  SOUL/docs in `/agent/<id>/`). **Per-agent filesystem sandbox:** every
  non-orchestrator agent is confined to its own `/agent/<id>/` folder — the
  install grants a baseline `Fs @ /agent/<id>/**` cap (`skills::install::
  with_home_sandbox`) and nothing wider unless the manifest explicitly requests
  a broader `Fs` scope, which the consent screen flags as "FULL filesystem
  access". Enforcement is the executor's scope gate (Gate 2.5); `list`/`search`
  are result-filtered by that same gate so a confined agent can't even
  enumerate paths outside its home. The shell agent (orchestrator) is the root
  and keeps `Scope::Any`. A **public registry** (`skills/registry_client.rs`)
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
  was started for — it runs that agent's model as a bounded ReAct loop (prompted
  with the agent's own `SOUL.md`) which returns a **JSON response object**
  (`{status, content_type/headers, file/body}`); `server.rs` parses that JSON and
  frames the reply. The body is either inline `body` or an asset the agent
  **names** (`file`) / **reads itself** (a `mem_fs_read` `<tool_call>`) — both go
  through the capability- and scope-gated reader confined to the agent's own
  `assets/`, so the SOUL agent *decides and reads* the content while native code
  only parses + frames (determinism boundary intact).
  So **a web server is just `agents/<name>/{SOUL.md, manifest.json, assets/…}`** —
  the SOUL carries the routing/behaviour (model-planned per request, greedy), the
  assets carry the content, and no per-server Rust is written. `doc` is exactly
  such an agent (data, not code). `/agents start <name> [port]` serves that agent
  over the pipeline; `ssh` runs standalone (RFC 4253 version exchange; transport
  is a stub). `/agents services` lists running stages. Git + full SSH transport
  follow the same native-protocol shape. To add a built-in server agent: drop
  `agents/<name>/{SOUL.md,manifest.json,assets/…}` and register it in
  `agent/system.rs` (one line) — or publish it to the registry.
- **Cortex** (`cortex/`) — CPU transformer inference, **architecture-dynamic
  like the ONNX interpreter**: `general.architecture` in the GGUF names the
  hyperparameter key prefix and resolves to a `Family` — `QwenHybrid`
  (Qwen3.5/3.6 DeltaNet+attention hybrid; finetunes like Ornith load via a
  key-shape sniff) or `Gemma4` (sliding-window/global interleave with per-kind
  geometry — GQA ring-KV local layers, MQA global layers with V=K + p-RoPE
  freq factors — sandwich norms, GELU, √dim embed scale, logit softcap,
  `layer_output_scale`, suppress-token bias). Nothing numeric is compiled in.
  **All mainstream GGML quants dequantize** (legacy Q4_0/Q4_1/Q5_0/Q5_1/Q8_0,
  K-quants Q2_K–Q8_K, i-quants IQ2/IQ3/IQ4 via `iq_tables.rs` — generated
  verbatim from ggml by `tools/gen_iq_tables.py` — plus F16/BF16 rows), so any
  unsloth file incl. UD-* dynamic mixes runs; fast SDOT matvecs for
  Q8_0/Q4_0/**Q4_K**, everything else through the generic dequant path (still
  SMP row-split). Two tokenizer flavors behind one API (GPT-2 byte-BPE ∣
  gemma4 raw-UTF-8 ▁-BPE with `<0xXX>` fallback), per-family chat format in
  the shell (ChatML ∣ `<start_of_turn>` gemma turns, BOS per `add_bos`).
  Select with `-model qwen3.5-0.8b|2b|4b|9b` **or any path**
  (`-model path/to/file.gguf` — guest RAM derived from file size), or at
  runtime with **`/model load <file.gguf>`** (reads off any FAT/ext4 volume
  into DMA frames and re-homes chat on it; the status bar shows the GGUF's own
  `general.name`). Zero-copy GGUF; grammar-constrained sampler; KV/recurrent
  cache (fixed W-slot rings on sliding layers). **`tools/cortexdiff/`** is the
  host-side harness (the onnxdiff pattern: mounts `kernel/src/cortex` natively;
  greedy decode in seconds; `diff.py` cross-checks tokenization + continuation
  against llama.cpp) — it generates the `refcheck.rs` fixtures (keyed by
  `general.name`; the numpy `tools/ref*.py` are gone) and is the required
  bring-up tool for any new family/quant. `cargo xtask ref-check
  [-arch aarch64] [-model …]` runs the acceptance gate natively under HVF
  (minutes, vs TCG hours) and powers off via PSCI.
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
- **UI** — a tmux-style split-pane framebuffer compositor in Geist Mono. The
  chat|action split is **resizable** (drag the divider with the mouse, or
  `/pane split <10-90>`; persisted to `/configs/core/panes.json` and reloaded at
  boot) and either pane can go **fullscreen** (Ctrl+F, or `/pane full` —
  `LayoutCfg.fullscreen`). (`panes.json` also carries `num_action_panes` 1–6;
  the N-pane split + inter-pane tab drag-drop is a scoped follow-up — today one
  action pane.) The compositor pairs the chat
  pane + an on-demand **tabbed "action" pane**: opening the ktrace stream, the
  `/top` dashboard, a vim-like editor, an **image viewer** (`/open .png|.jpg`),
  or the **audio player** (`/open .wav|.mp3`) each adds a tab; a tab bar in the
  pane header switches them (Ctrl+Tab / Shift+Tab / click), and every tab keeps
  its process alive when you switch away — the audio player keeps playing
  (pumped chunk-by-chunk from `ui_tick`, not a blocking loop), ktrace keeps
  streaming, the editor keeps its buffer. The editor is non-blocking: its state
  lives in a static and `read_line` routes bytes to it while its tab is
  focused. **Both media tabs take key controls while the action pane is focused**
  (Ctrl+Tab / click — same gating as pane scroll, so typing at the prompt is
  never intercepted): the image viewer does `+`/`-` zoom, `r`/`l` rotate,
  arrows pan, `0` reset (retaining the source, capped to ~4 MP, and re-rendering
  via the pure `image::render_view`/`rotate90`); the audio player does space
  play/pause, `←`/`→` seek ±5 s, `↑`/`↓` ±30 s, `0`/Home restart (state on the
  `AudioPlayer` static, seek just moves the PCM cursor, pause holds it while the
  device drains its queued chunk to silence). The **image viewer** decodes
  in-kernel — `image/` is a pure no_std PNG+DEFLATE and baseline-JPEG decoder
  set, unit-tested against real fixture files (plus `rotate90`/`render_view`
  transform tests) — then presents box-downscaled + letterboxed), a boot splash +
  status-bar **Synapse-C** brand mark, a live
  clock, a blinking caret, mouse cursor + click, **mouse text selection in the
  chat pane** (drag-to-copy → clipboard, paste with Ctrl+V; absolute-indexed
  over scrollback via `textsel`, like the editor's drag-select), a **host
  clipboard bridge** (`clipboard`: an in-OS copy emits an **OSC 52** escape so a
  terminal-attached host copies it to the macOS/Linux clipboard; a host paste
  arrives as a **bracketed-paste** `ESC[200~…ESC[201~` block the line editor
  captures — works the same on QEMU and VBox over the serial console, no
  guest-additions driver; `/clip` shows/sets it), **key
  auto-repeat** (`keyrepeat`: software typematic in `xhci` — USB HID boot
  keyboards report only press edges — plus an accelerating streak amplifier:
  held Backspace/arrows erase/scroll 2/4/8 steps per repeat in the shell and
  editor), **syntax highlighting** (`highlight`: JSON/Markdown/Rust/Python/C/
  JS/TOML/sh lexers colour the editor per-cell, `/cat` output, and the chat's
  streamed markdown — fence-tagged code blocks are lexed per language without
  breaking token streaming), ANSI-coloured agent output, a `/`-command shell,
  and an on-disk UI config (`/configs/core/ui.json`, `shortcuts.json`). The brand — logo, the terracotta `#cc785c` / warm-ink /
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

## STANDING RULE — Ctrl+C interrupts every command and process

**Any new command or long-running process must respond to Ctrl+C.** A blocking
loop already has to pump `shell::upkeep()` (above); in the *same* loop it must
also poll for interrupt and bail when it fires — otherwise a stuck/streaming
command can only be escaped by killing the VM, which is not acceptable.

- **A shell command / networked loop** (`/http`, `/ping`, DNS, TLS, a `/ws`-style
  stream, any future protocol client) calls `shell::poll_interrupt()` — true only
  on Ctrl+C (`0x03`), and it **pushes any other byte back** (`console::unread`) so
  it never swallows the *next* command's keystrokes. Return an `Err("cancelled")`
  / break the loop on true.
- **An inference / decode loop** calls `shell::poll_cancel()` (Ctrl+C **or** bare
  Esc; a decode turn owns the console, so consuming input there is fine).

Both are cheap — call them once per loop iteration next to `upkeep()`. A command
that can block without an interrupt check is a bug; cover it with an e2e `cancel`
scenario (drive raw `b"\x03"` via `guest.send_raw`, assert it aborts fast **and**
the next command still runs) and a unit test on the pure poll logic
(`poll_interrupt_ctrl_c_only_and_pushes_back`, `console::pushback_*`).
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
  `net/tls.rs`/embedded-tls; also the agent's `http` tool; **`-O`/`-o <file>`
  download the body to the Synapse store** (`/downloads/<name>`, overwrite
  confirms via the modal, human-typed only) where `/open` reads it back —
  editor, image viewer, or audio player), `/ws` (a
  plaintext WebSocket client in `net/ws.rs` — RFC 6455 handshake with
  Sec-WebSocket-Accept verification, masked frames, ping/pong), **`/mcp`** (an
  **MCP client** in `mcp.rs` — Model Context Protocol over HTTP/JSON-RPC 2.0,
  Streamable-HTTP transport with SSE + `Mcp-Session-Id`: `/mcp connect <name>
  <url>` runs initialize→tools/list and registers each remote tool into the
  tool registry as `mcp__<name>__<tool>`, so the shell agent calls it like any
  builtin — `tools/call` forwarded on invoke, results taint-tracked as
  UntrustedIngested; agents declare MCP servers in their manifest
  (`mcp_servers`), shown + connected on the install consent screen). `/model remote <http://host:port> [name]` points the
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
  (tone + mic check). **`audio/`** is the pure media-decoder layer behind the
  `/open <file>.wav|.mp3` **player**: a full RIFF/WAVE parser (PCM
  8/16/24/32-bit + float32, any channel count downmixed) and an MPEG Layer III
  decoder — a no_std Rust **port of minimp3** (CC0; tables generated verbatim
  by `tools/gen_mp3_tables.py`), validated ±1 LSB against minimp3's own scalar
  decode (stereo MS/short-blocks, MPEG-2 LSF, bit reservoir). Playback feeds
  the device in ~50 ms chunks — queue backpressure paces it — pumping
  `upkeep()` and answering Ctrl+C between chunks. `onnx/` is a zero-copy no_std ONNX (protobuf) reader +
  **op interpreter** that runs the real voice models end-to-end: silero-vad v5
  (VAD), parakeet-ctc int8 (STT — `/voice stt <wav>` transcribes), and
  KittenTTS (TTS — `/voice say <text>` speaks); bare `/voice` is the full
  mic → VAD → STT → LLM → TTS conversation loop. Models load lazily from any
  disk volume (bundled in the images; `cargo xtask voice-assets` downloads
  them into `assets/voice/`, gitignored). For any numeric or perf work on this
  path, use `tools/onnxdiff/` (host-side layer-by-layer diff of the kernel's
  own interpreter against onnxruntime) — not QEMU round trips.
- **Video** (`video/`) — H.264/AVC **baseline decoder + player** for
  `/open .mp4|.mov` (mkv/webm/hls demuxers pending), built **in stages, each pure
  + unit-tested off-hardware** and **validated bit-exact against ffmpeg/PyAV via
  the `tools/h264diff/` host harness** (mounts `video/*.rs` via `#[path]`; the
  onnxdiff/cortexdiff pattern — CAVLC VLC + alpha/beta/tc0 tables are parsed
  from the FFmpeg source, never hand-transcribed). **Full baseline pipeline:**
  `mp4`/`mkv` demux → CAVLC residual (`h264/cavlc.rs`) → **I + P** macroblock
  decode (`h264/decoder.rs`: I_4x4/I_16x16/I_PCM, P_L0_16x16/16x8/8x16/8x8/Skip,
  **multiple slices per frame** with slice-aware neighbour availability) → intra
  (`h264/intra.rs`) + **inter** (`h264/inter.rs`: median MV prediction + 6-tap
  luma / bilinear chroma MC) → inverse transform (`h264/transform.rs`) →
  **in-loop deblocking** (`h264/deblock.rs`) → YUV→RGB → a **video tab** with
  a **player HUD** (`framebuffer::draw_video_status`: state, mm:ss, frame
  counter, scrubber, mute, shortcut hints — drawn *after* each frame blit) and
  transport controls (Ctrl+Tab focus, space pause, ←/→ seek, ↑/↓ ±10 frames,
  0 restart, `m` mute, Ctrl+C stop), frame-paced by pts. **Streaming decode:**
  `video::StreamDecoder` holds the source + sample table + **one** reference
  frame and decodes on demand (`seek_decode`, rewinding to the latest keyframe
  on a backward seek). Do **not** decode a whole clip into a `Vec<Frame>` up
  front — a 1300-frame 480p clip is ~700 MB of RGB, which overruns the first-fit
  heap, corrupts a reference frame's chroma under allocation pressure, and every
  dependent P-frame then renders **all-green** (a real bug; the decode itself
  was bit-clean). **Audio:** `mp4::parse_audio` demuxes the AAC (`mp4a`/`esds`
  → AudioSpecificConfig) track and `video::audio_info` reports it, but the
  **AAC-LC decoder is not yet built** (a full codec on the scale of the H.264
  one — spectral Huffman/iquant/M-S/TNS/**PNS**/intensity + IMDCT filterbank;
  note PNS's seeded noise makes bit-exact-vs-PyAV validation impossible, unlike
  H.264), so video plays silently and the HUD shows `[no audio]`. **Validated
  bit-exact against PyAV/ffmpeg** — synthetic x264 clips (I/P, multi-slice,
  deblocked) and hundreds of consecutive frames of real-world mp4/mkv. In-kernel
  fixture tests hash an embedded I-only and an I+P clip against PyAV, and
  `stream_decoder_seek_matches_sequential` proves random/backward seeks match a
  sequential decode frame-for-frame.
  **Deblock gotchas (both bit us):** a chroma edge's QP is `avg(qpc(QPp),
  qpc(QPq))` not `qpc(avg(QPp,QPq))` (differs only across slices with differing
  QP); and the luma normal filter's `tc = tc0 + (ap<β) + (aq<β)` can be nonzero
  even when `tc0==0` (don't force-skip). **Remaining:** CABAC (High/Main
  profiles — big fraction of real files), HLS/TS, and a rare corner-MB edge case
  on very long P-sequences. **Stage 1 (done):** `video/bits.rs`
  (RBSP emulation-prevention unescape + a big-endian `BitReader` with H.264
  Exp-Golomb `ue`/`se`/`te`), `video/mp4.rs` (ISO-BMFF box-tree demuxer →
  `avcC` SPS/PPS + the `stsz`/`stsc`/`stco`/`stts`/`stss` sample table assembled
  by the pure `build_samples`), and `video/h264.rs` (Annex-B **and** AVCC NAL
  splitting + SPS/PPS parse → geometry/profile/entropy mode). `video::probe`
  reports a stream (container, codec, `W×H`, frame count, CAVLC/CABAC) without
  decoding pixels; `/open clip.mp4` shows it. Scope: **H.264 baseline** (I/P
  slices, CAVLC, 4:2:0), the common mp4/mov case. **Remaining:** the **AAC-LC
  audio decoder** + playback sync (demux done; see above), CABAC (High/Main
  profiles — big fraction of real files), HLS/TS demux, a rare corner-MB edge
  case on very long P-sequences, and the multi-pane split + tab drag-drop.
  **Host reference for the numeric stages:** PyAV
  (self-contained ffmpeg) decodes the same clip to YUV for a frame-by-frame diff
  harness (`tools/h264diff/`, the onnxdiff/cortexdiff pattern — mounts
  `video/*.rs` via `#[path]`, runs on the host in seconds, no QEMU round-trips).
  The e2e `open_video` scenario muxes a real x264 baseline multi-slice clip into
  mp4 (stdlib muxer) and asserts the on-kernel probe + streaming decode ("N
  frame(s), ready in …") + transport controls; it auto-skips where x264 is absent.
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
