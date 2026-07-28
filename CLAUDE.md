# ChittiOS — guide for agents & humans working in this repo

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

Current figures (aarch64 HVF, 0.8B Q8, `/perf 512`): prefill ~105 tok/s, decode
~23 tok/s (same box, same run: prefill was ~60 tok/s before the window-wide
cores below), `/voice stt` ~2 s, `/voice say` ~14 s for 3.5 s of audio.

**Prefill's parallel axis is not the position, and getting that wrong is what
kept `pp` pinned near `tg`.** The projections were weight-stationary matmuls
across the whole fleet while the attention and DeltaNet cores ran a position at
a time on the BSP — so seven cores idled through the part that was left, and a
2x-of-decode prefill looked like a matmul problem when it was an Amdahl one.
Both cores are now window-wide (`attn_core_batched`, `delta_core_batched`, with
decode entering them as a one-position window so the two paths cannot drift):
attention fans out over **positions** (every `(position, head)` is independent
once the batched K/V are in the cache), the recurrence over **heads** (sequential
in position, but each head owns its own state slice). Three things that bite
here: the conv1d's window/ring index is `ck + p`, **not** `ck - 1 + p` — the
per-position form shifted the ring before convolving and that `-1` follows you
into the rewrite, where it reads every tap one position too old and still decodes
fluent text (`conv_tap`, pinned by `conv_window_matches_the_per_position_ring`);
parallelism that exists is not parallelism worth taking — fanning a *decode*
step's 0.8M-MAC DeltaNet layer across the fleet measured **slower** than inline
(35 -> 20 tok/s), hence `fanout_chunk`; and `/perf [n_prompt [n_decode]]` now
prints a **proj / attn / delta / elementwise** split, because "prefill is slow"
is three different diagnoses and they scale differently. Measure with a real
prompt length: at 64 tokens prefill is one chunk and tells you nothing.

**aarch64 SMP
row-split is live** (`arch/aarch64/smp.rs`): PSCI `CPU_ON` bring-up, `WFE`-parked
workers, a static-partition job barrier splitting the SDOT matvecs + generic
`parallel_for` (video YUV→RGB) across all online cores. **x86 has an equivalent
fleet** (`smp.rs`): APs park in `hlt` with interrupts enabled and are woken by an
all-excluding-self **IPI** (a `pause` spin would cost a core of power per idle AP —
real heat and battery on a laptop), with the same static-partition barrier, the
same claim/done straggler protocol, and its own boot wake self-test. Callers reach
both through **`arch::parallel_for` / `arch::online_cpus`** — never
`arch::aarch64::smp::*` directly. That direct `cfg(aarch64)` call was how x86 came
to run every ONNX op, video row conversion and matvec on one core while aarch64
used the whole machine; a new parallel loop must use the neutral facade or the
divergence comes straight back. **The barrier is bounded, never trust a worker wake**: workers
enable the counter event stream (`CNTKCTL_EL1`) so `WFE` self-wakes, a
claim/done protocol recomputes a straggler's range on the BSP, and a boot-time
wake self-test (`smp: wake self-test ok|FAILED` ktrace) degrades to single-core
up front on hypervisors that park a trapped `WFE` until an interrupt —
VirtualBox-ARM does exactly that and used to hang the first prefill matvec
forever. Slow beats stuck; any new cross-core wait needs the same bound.
**The PSCI gate fails open**: bring-up skips PSCI only when a *valid* FDT
explicitly lacks an `arm,psci-*` node (Apple Silicon via m1n1 — `hvc` there
halts the guest). Boots with **no FDT in x0** (QEMU/VBox `-kernel` ELF, the
UEFI stub) keep PSCI — gating those on FDT contents once silently turned SMP
off on QEMU (`fdt::present` distinguishes "no FDT" from "FDT says no PSCI";
the `smp: N cores online` ktrace is the first thing to check when inference
is inexplicably slow). QEMU vCPU count comes from `CHITTI_SMP` (default 8).
Also NB: `make`'s `RELEASE` defaults to **1** — a dev kernel's unoptimized
NEON is many times slower and reads as an inference bug.

## STANDING RULE — real hardware, nothing hardcoded to an emulator

Drivers must target **real, standards-based hardware**, not QEMU or VirtualBox
quirks. Do not hardcode addresses, resolutions, device layouts, or behaviour to a
specific emulator/hypervisor. Discover hardware the way real firmware does
(ACPI/PCIe ECAM, UEFI GOP, fw_cfg, HID report descriptors, PrimeCell IDs, EDID/
mode tables) and degrade gracefully when a facility is absent. A feature that only
works under QEMU is not done.

**The display mode comes from the display, via EDID — never a constant, never
"the biggest mode advertised".** The kernel itself holds **no resolution at all**:
`width`/`height`/`pitch`/pixel-format arrive from the firmware (Limine's
`Framebuffer` on x86, the stub's boot-info page on aarch64, m1n1's prepared
framebuffer on Apple Silicon) and the font scale is derived from the height
(`pick_scale`), so every layout is a ratio of whatever the panel turned out to be.
That means the *only* place a resolution is decided is the loader, and on real
hardware the chain is **monitor EDID → loader picks the mode → kernel adopts the
framebuffer geometry**.

`kernel/src/edid.rs` parses the EDID base block
(header + checksum validated, then the first detailed timing descriptor) into the
panel's native resolution; the aarch64 `stub/` mounts that same file with
`#[path]` so the two can't disagree, and its pure bit-packing is unit-tested
(`cargo xtask test`) rather than only on hardware. The selection order is
**EDID-preferred → keep the firmware's current mode → largest advertised mode**,
and the middle step is the load-bearing one: with no EDID, the mode the firmware
is already in *is* the resolution the platform was configured for (VirtualBox's
`VBoxInternal2/EfiGraphicsResolution`, UTM's display setting), so overriding it
throws away the user's choice. Both arches had this wrong in different ways —
`kernel/limine.conf` pinned `resolution: 2560x1440`, and the stub always jumped
to the largest GOP mode — which is why a VirtualBox guest came up at a fixed QHD
surface regardless of its settings. On x86 the fix is simply **not** to set
`resolution:` (Limine then queries EDID itself, falling back to 1024x768 with
none); `CHITTI_RESOLUTION=WxH cargo xtask image` appends an explicit override for
a headless VM. Only fall back to the largest mode when there is no EDID *and* the
firmware's mode is below 1024x768 — a default nobody chose, which is the real
"UEFI came up at 800x600" case the largest-mode heuristic was written for.

**A hypervisor's resolution knob cannot be trusted, so there is a channel that
does not need one.** `VBoxInternal2/EfiGraphicsResolution` is *stored* by
VirtualBox-ARM and then ignored — the guest boots at the host panel's size
whatever it says — which left no way to ask for a framebuffer that fits the VM
window (VirtualBox draws the guest 1:1, so a 2560x1440 guest in a 1440-wide
window has half of itself off-screen; that is a clipped console, not a driver
bug). So the loader reads a preference off the ESP: `\chitti-display.cfg`,
`resolution=<W>x<H>`, parsed by `edid::parse_boot_cfg` and applied with GOP
`set_mode` **before the kernel starts**, outranking even the display's EDID-native
mode because it is the one size a human typed on purpose. `CHITTI_RESOLUTION=WxH`
now writes it at image-assembly time on aarch64 exactly as it appends
`resolution:` to `limine.conf` on x86, and `make vbox VBOX_RES=` goes through it.
Three rules it must keep: **empty means unset** (a wrapper passes the variable
through unconditionally), the depth component of Limine's `WxHxBPP` is **dropped**
(a GOP mode is chosen by dimensions, so writing it out would be ignored silently),
and a request is a **ceiling** — `best_mode_for` never exceeds it, since the reason
for asking is usually that something bigger does not fit. The stub logs the
firmware's entire mode list and says so when the requested size was not on offer;
without that, "the resolution I asked for did not happen" has two
indistinguishable causes (never offered vs. `set_mode` failed), and on a machine
that will not boot right that distinction is the whole diagnosis. Verified by
booting `--uefi` under AAVMF, which offers only 640x480/800x600/1024x768: a pinned
1024x768 reaches `framebuffer TUI up (1024x768)`, a pinned 1280x720 correctly
lands on 800x600 (1024x768 is *taller* than 720) and says why, and an unset build
writes no file and behaves exactly as before. `/display boot` still only *records*
a preference — the kernel would have to write FAT to mirror it to the ESP — and
says so rather than implying a reboot will apply it.

**Resolution is a setting, and there are exactly two kinds of it** — `/display`
(`kernel/src/display.rs`, pure + unit-tested; persisted to
`/configs/core/display.json`; also exposed to the **settings agent** as the
`display` shell tool, which may apply it directly since it is reversible):

- **The logical desktop** (`/display set <WxH>|native`) — applies *instantly* on
  both arches. `width`/`height` on `Screen` are the **logical** desktop and
  `origin_x`/`origin_y` place it inside the physical framebuffer, so a smaller
  resolution is a centred, letterboxed viewport that still renders **1:1** —
  glyphs are rasterised at physical pixels, nothing is scaled, text stays sharp.
  The entire translation is one function, `Screen::fb_offset`, which every
  framebuffer write goes through (there are only four such sites — `put_pixel`,
  the row blit, the cursor read-back, and the pane scroll's row copy; keep it that
  way). At native both origins are 0 and it is the identity, so the default path
  is byte-identical to before. Note `rebuilt`/`relayout` must feed `build` the
  **`fb_w`/`fb_h`** physical size, never `width`/`height`, or the viewport shrinks
  on every rebuild; and `pick_scale` takes the *logical* height so a smaller
  desktop gets proportionally sized text.
- **The font scale** (`/display scale <1-4>|auto`) — cells are `8*scale` x
  `16*scale` px, so **this** is what answers "everything is too small on a
  high-resolution screen"; a smaller desktop only letterboxes. The automatic value
  is `display::auto_font_scale(height)`, thresholds not a division: the old
  `(h + 550) / 1100` needed **1650** px to reach scale 2, so a 2560x1440 panel
  rendered at scale 1 — 8x16 px cells, 320 columns — which is what actually made a
  2K display look broken, independently of which mode the loader picked.
- **The panel's own mode** (`/display boot <WxH>|auto`) — only the loader can set
  this, so it costs a reboot. **This is recorded but NOT yet applied**: the
  preference lives on the ext4 store and the loader can only read the ESP, so
  mirroring it there (a FAT write via `block::esp`, then the stub reading it and
  the x86 `limine.conf` `resolution:` line being rewritten) is the missing bridge.
  Until then `/display boot` says so explicitly and points at the platform knobs
  that do work (`VBOX_RES`, `CHITTI_RESOLUTION`) — do not let it claim otherwise.

**Display settings are stored per monitor, the way `monitors.xml` does it.** The
stub copies the chosen output's **EDID base block** into the boot-info page
(length at 384, block at 388) — the firmware's buffer is gone by the time the
kernel runs, so it is handed over or lost, the same handoff Linux's EFI stub makes.
`edid::identity` unpacks the display's own vendor/product/serial (the manufacturer
code is three **five-bit** letters packed big-endian in bytes 8..10 — reading it as
a `u16` gives nonsense) and `edid::monitor_name` reads the `0xFC` descriptor, so
`/display` can name the output it is talking about. `display::profile_key` keys the
settings on that identity, falling back to `fb-<W>x<H>` where no EDID is published
(hypervisors, and the x86/Limine path, which passes none) so two
*differently-sized* monitors still get separate profiles. `display.json` is
therefore a `displays: { key: {logical, font_scale} }` map plus a global
`boot_mode`; the older flat shape is adopted for the display in use and migrated on
the next save rather than discarded. A save rewrites **only** the current
display's entry.

**KMS — real kernel mode setting** (`kernel/src/kms/`) follows Linux's DRM split:
`kms/mod.rs` is the device-independent core (`Mode`/`Connector`/`Scanout`, the
`DisplayDriver` trait, mode-set orchestration, damage accumulation, polled
hot-plug), and each device is a backend. **virtio-gpu is implemented and verified**
(`kms/virtio_gpu.rs`): `GET_DISPLAY_INFO` → `RESOURCE_CREATE_2D` →
`RESOURCE_ATTACH_BACKING` (the device scans out of **our** DMA pages, so the
compositor draws with no copy) → `SET_SCANOUT`, then
`TRANSFER_TO_HOST_2D` + `RESOURCE_FLUSH` to present. Confirmed by screendumping the
virtio-gpu device itself: a `1280x800` console really becomes `1280x720`, full
panel, scrollback intact.

Four things that path teaches:

- **virtio-gpu does not scan out of guest memory continuously.** Drawing alone
  changes nothing on screen — it must be transferred and flushed. Damage is unioned
  by `kms::damage` from the *coarse* painters (`fill_rect`, `blit_rgb32_row`,
  `draw_str`, `redraw`) and flushed once per `upkeep`; `put_pixel` deliberately does
  **not** report, because a redraw is millions of calls and a per-pixel union costs
  more than the flush it feeds. A flush per glyph is a queue round trip per glyph.
- **Damage is in physical coordinates**, so the logical-viewport origin has to be
  added — a scanout is the whole framebuffer, not the desktop.
- **A KMS-only machine has no framebuffer to re-init.** `reinit_scanout` therefore
  *initialises* the console when none exists; without that, booting with only a
  virtio-gpu gives a blank screen (found by doing exactly that).
- **On aarch64 `-kernel` there is no PCI** (ECAM comes from the stub's ACPI), so
  virtio-gpu binds on the **UEFI** path and on x86, not on the plain `-kernel` dev
  loop. Matched by **vendor+device id**, never display class: `virtio-gpu-pci`
  reports class `03:00` like every other VGA device.

**VMSVGA (`kms/vmsvga.rs`) works** — the backend for **VirtualBox** and QEMU's
`vmware-svga`. Verified by screendump: `1024x768` -> `1280x720`, clean. Getting there
needed three things that are each easy to get wrong:

- **The FIFO, not just the mode registers.** SVGA II ignores `WIDTH`/`HEIGHT` until
  its command ring (BAR2) is configured and `CONFIG_DONE` is *accepted*: set
  `MIN`/`MAX`/`NEXT_CMD`/`STOP`, where `MIN` must clear the extended register area
  (`SVGA_FIFO_NUM_REGS * 4`) when `SVGA_CAP_EXTENDED_FIFO` is set, and `MAX - MIN`
  must leave at least 10 KiB or the device rejects the ring. Before that the registers
  read back whatever was written while the scanout keeps its geometry — the console
  rendered **four times side by side**. `init` refuses to bind if `CONFIG_DONE` reads
  back 0, because driving it in that state makes a mode set silently do nothing.
- **`flush` is NOT a no-op.** In VGA mode the device tracks framebuffer writes; once
  in SVGA mode it only repaints from `SVGA_CMD_UPDATE` in the FIFO. So the driver
  queues an update rect per damage flush and pokes `SVGA_REG_SYNC`. Drawing without it
  leaves the screen frozen on the mode-set frame. (virtio-gpu needs the same thing for
  a different reason — see above — so both backends are damage-driven.)
- **The geometry registers do not report the mode in effect.** Before the guest first
  enables SVGA mode they hold the device's defaults (QEMU answers 640x480) while the
  display is at whatever the firmware programmed through VGA. So the current mode is
  seeded from the **live framebuffer** and only falls back to the registers; trusting
  them made `preferred` 640x480 on a 1024x768 console, which is what a KMS-only boot
  would then have come up in. `MAX_WIDTH/MAX_HEIGHT` is a VRAM ceiling (an odd
  2368x1770 on QEMU), never a mode to offer as native.

Two safety rules this driver had to learn the hard way, both after breaking a real
VirtualBox display:

- **Probing must not change the device.** `CONFIG_DONE` and the ring pointers alter
  how the device scans out, and at probe time the console is still drawing into the
  *firmware's* framebuffer — so writing them in `init` moved the scanout out from
  under a live console and left the display offset and clipped. They now happen
  lazily in `ensure_fifo`, on the first actual mode set. Same rule the I2C/EC drivers
  follow: identification only ever reads.
- **`VMSVGA_ALLOW_MMIO` is off.** `Regs` can reach BAR0 as I/O ports *or* MMIO, but
  only the port path can be tested here (QEMU emulates `vmware-svga` on x86 only,
  where BAR0 is I/O). VirtualBox-ARM needs the MMIO path and its register layout is a
  guess; acting on that guess mis-programmed a real display. So it declines and keeps
  the firmware framebuffer — the KMS layer is optional, and getting these registers
  wrong costs the console, not just a feature. Verify the layout on the target before
  flipping the flag.

Without a bound backend the whole module is inert and the compositor keeps the
loader's framebuffer — the position Linux is in with `efifb`/`simpledrm`
(`nomodeset`): mode fixed by firmware, `/display set` letterboxes instead, console
legibility via font size. Still absent: real-hardware GPU drivers (i915/AMD/AGX) —
see the note on why there is no display equivalent of xHCI/AHCI.

**A machine can have more than one display, so the stub enumerates every graphics
output** (`locate_handle_buffer`, not `get_handle_for_protocol`) and picks one via
`edid::pick_output`: the output carrying the firmware's **console-out marker**
(`EFI_CONSOLE_OUT_DEVICE_GUID`) first — that is where the firmware drew its own
boot messages, hence the display the user is watching — then any output with a
readable EDID (proof something is plugged in), then output 0 so a headless box
still gets a console. Taking handle 0 unconditionally, as this did, was a coin flip
between a laptop's built-in panel and its attached monitor: it would read one
display's EDID and set the mode on the other. Each output's `console_out`/`edid` is
logged, so "wrong screen" is diagnosable from the boot log alone.

The **QEMU ramfb** window is a separate path with the same "match the display, not
a constant" rule, and it was wrong in its own way: it scanned `system_profiler`
for the *first* `Resolution:` line and used the **physical** pixel count. On a
multi-monitor Mac that silently picked whichever display was listed first, and on
a HiDPI panel it handed the guest a framebuffer bigger than the desktop showing it
(a 2560x1600 panel whose desktop is 1440x900). `xtask::parse_displays` now parses
the per-display blocks (pure, `cargo test -p xtask`) and takes the **main**
display's *desktop* size — macOS's `UI Looks like` when present, else the physical
size halved for a Retina panel, since Apple's default scaled modes are an
unpublished per-panel table and the pixel count is never the right answer.
`CHITTI_FB_DISPLAY=<name substring>` picks another monitor; `CHITTI_FB_RES=WxH`
pins it. Every detected display is logged with `*` on the main one.

Concretely: display comes from the firmware (Limine GOP on x86, UEFI GOP via the
`stub/` bootloader on aarch64, QEMU ramfb as a fallback); disks via virtio /
NVMe / AHCI over discovered PCIe; input via USB xHCI/HID (keyboard **and** mouse,
report-descriptor-driven), virtio-input, PL050/PS-2, and **HID-over-I2C**
(`drivers/i2c_hid.rs` on `drivers/i2c.rs`, the DesignWare/LPSS master) — the touchpad on
laptops from ~2016, which have no PS/2 aux port. An I2C device **cannot be probed for**:
its address comes from `_CRS`, so it is located by asking the namespace which device
claims `PNP0C50` (`aml::device_by_hid`). Identification only ever *reads*, and
`present()` is a zero-length probe, because the same bus commonly carries the embedded
controller and the PD controller — a stray write there misconfigures real hardware. The
HID descriptor register comes from `_DSM` and is defaulted to `0x0020`, but the
descriptor read is **validated** (length + version), so a wrong guess is detected rather
than silently producing garbage. Report decoding is shared with USB via
`xhci::feed_pointer_report`, so a touchpad and a mouse cannot drift apart; the wall clock from the
RTC / UEFI `GetTime` / the virtual counter — each behind a shared facade with a
per-arch implementation. The same kernel image must run on QEMU, VirtualBox, and
real UEFI hardware.

**On x86, ACPI tables must be mapped before they can be read.** Limine's HHDM
covers **usable RAM**, and the tables live in firmware-reserved regions outside
it — so *both* the raw physical address and its `phys_to_virt` translation are
unmapped, and touching either is a page fault that halts the boot, not a garbage
read a signature check can reject. (Cost two boot hangs to find: `0xf52e0`, then
`0xffff8000000f52e0`.) `acpi::map_table` maps every table page explicitly on x86
and remaps to the header's declared length, since an XSDT can exceed a page.
aarch64 is deliberately untouched there: it has a flat identity map, and
`init_uart` reads SPCR before the frame allocator exists. Relatedly, Limine's RSDP
pointer is **physical on newer protocol revisions and HHDM-virtual on older ones**,
and the two cannot be told apart by trying both — classify by range (higher half =
already virtual), then signature-check.

**FADT field offsets are named constants, and one of them was wrong.** `X_DSDT` is
at **140**; the code read it from **148**, which is where `X_PM1a_EVT_BLK`'s Generic
Address Structure starts. That does not fail loudly — it yields
`space|width|offset|access|address_lo32` reinterpreted as a `u64`, a large
*plausible* number that sailed past the plausibility guard and made the DSDT
unfindable on every machine with a modern FADT. Consequence: on x86 the whole AML
layer silently did nothing — no `_S5_` poweroff, no I2C touchpad, no battery — and
nothing reported an error, because "no DSDT" is a legitimate state. Every offset this
reads is now a named constant with a test that pins it (`x_dsdt_is_at_offset_140_not_148`
spells out the wrong read for the next person), and the pure decoders
(`acpi::fadt_dsdt`, `acpi::fadt_pm1`) take a slice so they are testable off-hardware.
A related trap in the same family: the PM1 **event** block is split down the middle —
status first, enable second — so `PM1a_EN` is at `PM1a_EVT + PM1_EVT_LEN/2`, **not**
a fixed `+2`; assuming `+2` writes an enable mask into the middle of the status
register on any machine with an 8-byte block.

**The power button works, and the FADT says which kind it is.** `drivers/pwrbtn.rs`
arms the ACPI **fixed-feature** button (PM1 status bit 8) and `shell::upkeep` polls it
— one `in` per pump, which is cheaper than routing an SCI on a cooperative scheduler,
and a button press is a human-timescale event. It refuses to arm unless the FADT
described a fixed-feature button, ACPI mode is on (`SCI_EN`), and the event block is
long enough — each refusal ktraced, because "the power button does nothing" is
otherwise indistinguishable between those cases. Two rules keep it from powering a
machine off by accident: an all-ones status read is an **unclaimed port**, not every
event firing at once, and PM1 status bits are **write-1-to-clear**, so the ack writes
*only* the button bit rather than the whole word (which would acknowledge every other
pending event). The **control-method** (`PNP0C0C` GPE) button — what many laptops
use — is *reported, not guessed at*: FADT flags bit 4 says so, and GPE dispatch is
honestly unimplemented rather than silently polling a bit that will never change.
Uniquely for this hardware area, it is **verifiable in a VM**: QEMU's
`system_powerdown` sets exactly that status bit, and the `power_button` e2e scenario
presses it through the monitor (`-serial mon:stdio`, so Ctrl+A c reaches it — no extra
plumbing) and asserts a clean shutdown. `/battery` prints the button's state alongside
the battery's, and `tests/e2e/run.py --only <names>` exists so one scenario can be
iterated on without the 30-minute sweep.

**Poweroff and the scheduler tick are real hardware now, not emulator stand-ins.**
`/poweroff` performs an ACPI **S5** transition (`SLP_TYPa | SLP_EN` to the FADT's
`PM1a_CNT`, with `SLP_TYPa` decoded from the DSDT's `\_S5_` package by bytecode
scan; see the AML note below), keeping QEMU's `isa-debug-exit` write only as a
fallback; it used to write *only* that port, so on a physical machine `/poweroff`
did nothing and left the fans running. The tick prefers the **local-APIC timer**
calibrated against the **HPET** (`arch/x86_64/hpet.rs`), falling back to the
PIT/8259 — both of which a UEFI-only machine may omit entirely, in which case the
old code had no preemption at all and said nothing. Every wait added here is
bounded and the HPET gets a counter-liveness probe, because an unbounded spin on a
dead reference clock hung the boot before a single test ran.

**AML (`aml.rs`) decodes, locates, and evaluates a fail-closed subset.** ACPI
describes anything unenumerable as bytecode in the DSDT, so `aml.rs` is the byte layer:
`PkgLength`, `NameString`, data objects, then `devices()` / `methods()` walking
`Scope`/`Device`/`Method` (all three carry a PkgLength, so their extent is exact).
`device_by_hid` + `device_name` answer "what is *this* device's `_CRS`" — which is the
question a driver asks and a flat scan cannot answer. Three encodings cause all the
bugs, each pinned by a test: **`PkgLength` is asymmetric** (low six bits alone, low four
when more bytes follow), **`NameString` has five forms**, and **`OnesOp` is all-bits-set,
not `0xFF`**. Containment is the other bug source — a parent's body contains its
children, and a `Name` inside a **method body is a local**; both are excluded from
`device_name`, and both were shipped wrong first and caught by tests.

On top of that sits an **evaluator** (`eval_method`, `eval_device_method`, and the
`_with_fields` variants): `Return`/`If`/`Else`/`Store`, `Local0-7`/`Arg0-6`, the
comparison and one-target arithmetic operators, and dynamic `Package` construction. Its
governing rule is that **an unsupported opcode returns `None`, never a value** — an
evaluator that guesses an integer is worse than the validated default it would replace,
which is why it was not half-built earlier. Note two ACPI traps it encodes: **`TRUE` is
all-bits-set**, so a caller comparing against `1` reads every true result as false, and a
method that falls off its end without `Return` yields `None` here rather than the spec's
zero — for these callers "unsupported control flow" is far likelier than a deliberate
zero. `OperationRegion`/`Field` (`regions`, `fields`, `find_field`) locate the named
bit-ranges a method actually reads; a **reserved field entry still advances the bit
cursor** (skipping it shifts every later field, which is how a battery reports a voltage
as a capacity), and an `AccessField` stops the walk rather than continuing with offsets
that would be wrong.

**A real battery percentage is the composition of all of it.** `drivers/ec.rs` is the
ACPI **embedded controller** — `PNP0C09`'s `_CRS` (authoritative; the fixed `0x62`/`0x66`
are only an x86 fallback) driven through bounded spins, with a `0xff` status rejected as
an unclaimed port *before* any command is written and a stale output byte drained so a
read cannot return the previous transaction's value. `drivers/battery.rs` evaluates the
firmware's own `_BST` with a field resolver that reads `EmbeddedControl`/`SystemMemory`/
`SystemIO` bytes, takes last-full capacity from `_BIX` (index 3) or `_BIF` (index 2), and
reports `remaining / last-full` — **last-full, not design**, or a worn battery reads
permanently below 100%. Surfaced as the `${battery}` status-bar variable
(`ui_config.rs`), cached for 5 s because one reading costs an AML evaluation plus a
handful of EC transactions; a variable that resolves to nothing takes its separator with
it, so a desktop's bar is byte-identical to before. Every layer fails closed and ktraces
which step gave up. **None of this is verified on hardware** — QEMU emulates no ACPI EC
and no battery — so the pure arithmetic (bit assembly, `_BST`/`_BIF` shapes, the
handshake against a simulated controller incl. slow/wedged/dead) is what the tests hold.
The touchpad's descriptor register now comes from `_DSM` too
(`drivers::i2c_hid::descriptor_register`) rather than the `0x0020` default — note the
`_DSM` UUID goes in ACPI's **mixed-endian** buffer order, or the table's own `LEqual`
fails and the method silently takes its unsupported branch.

**A HID-over-I2C touchpad reads fine while powered down, which is the trap.** The HID
descriptor is answerable from a device that has never been powered on — so descriptor
parsing appeared to work while no report would ever arrive. `SET_POWER(ON)` then `RESET`
through the **command register** (`drivers::i2c_hid`) is what makes reports flow, and the
command encoding is register-address-LE then argument then opcode: swapping the last two
sends `SET_POWER` as report-type 8 of opcode 0, which a device answers by doing nothing
rather than by NAKing, so the symptom is a dead touchpad and not an error. The sequence is
deliberately only reachable *after* `HidDesc::parse` validates, because those are the
first **writes** this driver makes and the same bus carries the embedded controller.
`sleep()`/`resume()` exist for the suspend path, where the device loses its state.

**A battery percentage needs the AC adapter to be meaningful, and `_BST` will not tell
you.** Once a pack is full, `_BST` reports *neither* charging nor discharging — so a
plugged-in machine and one running down produce byte-identical flags. `ACPI0003`'s `_PSR`
is the missing half (`=` in the status bar), reported as `None` rather than guessed as
unplugged when no adapter device exists. Two more things a laptop needs: `_STA` bit 4 says
whether a bay actually contains a pack (a removed battery leaves its device in the
namespace), and a machine with two packs has two `PNP0C0A` devices — `aml::devices_by_hid`
returns all of them and the capacities are **summed**, because reporting the first pack
presents half the machine's charge as all of it. Flags union: one pack discharging means
the machine is.

**The RTL8125 is not register-compatible with the 8168 it is dispatched with.** The
2.5GbE parts move the interrupt mask and status to 0x38/0x3c and widen them to 32 bits,
and the transmit doorbell to 0x90 — and the 8168 offsets *overlap* those positions, so
driving an 8125 with them writes the doorbell into the interrupt mask. `net::r8169`
carries a per-chip `RegMap` (pure, unit-tested, including that every id the dispatcher
sends here lands in exactly one layout) rather than a comment saying to treat 8125 with
caution. Still unverified: QEMU models no r8169-family part at all.

**Intel WiFi (`drivers::wifi::iwl`) is identification and firmware only, deliberately.**
An Intel radio does nothing until an image is loaded, and the image is chosen by chip
family plus an API version that is a property of the *file* — Linux tries filenames
newest-first, and so does `fw::firmware_candidates`. An unrecognised Intel id is **not**
claimed: the wrong firmware fails a signature check *inside* the device with no error the
host can read, which is worse than the Ethernet dispatcher's silent non-receiving NIC.
The `.ucode` TLV parser refuses a pre-TLV image (leading word non-zero), a wrong magic and
any record claiming more than the file holds, and pads record lengths to 4 bytes — one
odd-length record misaligns every record after it. On top of that, `csr` is the register
map with its pure predicates (all-ones is a floating bus, not data; a `prph` address needs
its access-size bits or the following write vanishes), `context` is the gen2 **context
info** — from AX200 onward the host does not feed firmware section by section, it hands the
device a structure and the device's own loader fetches the image, so nearly all the risk
moves into one struct layout and every offset is pinned with `offset_of!` — and `device` is
the ordering: prepare the card (`NIC_READY` going **clear** is the ready signal; waiting
for it to set never completes), APM init, stop the DMA master *before* resetting so no
transfer is in flight against memory about to be reused, then grab MAC access — proceeding
without the grant is worse than failing, because reads return stale values and writes are
dropped, so bring-up appears to work and the device never starts. Every wait is bounded and
names itself. Bring-up is **command-driven** (`/wifi up`), never automatic at boot: the
same posture AGX and the Broadcom radio take, because an untested driver should not touch a
device just because the machine started. Firmware is **fetched, never
committed** (`cargo xtask iwlwifi-assets` into the gitignored `assets/wifi/iwl/`), which
is the same rule the Broadcom assets follow and the reason this needed no licensing
decision. What still does *not* exist: the **receive path** — so firmware's own *alive*
notification cannot be observed, which is why "handed over" is the strongest claim the load
makes — the command round-trip, and then 802.11 + WPA2. So the radio does not associate and
`/wifi connect` still cannot work; `/wifi up` reports how far bring-up got instead.

**Interrupt-controller bases are discovered, and there are two sources, not one.**
aarch64 finds the GICv3 from the device tree's `arm,gic-v3` `reg` when there is an
FDT, and otherwise from the **ACPI MADT** (`acpi::gic_from_rsdp`: GICD type `0x0D`
for the distributor; the redistributor from the GICC entry whose `MPIDR` matches
this core, else the first GICC's `GicrBaseAddress`, else the GICR discovery
range). Both windows get `map_device_gib`. This matters because the two cases are
*different real platforms*: QEMU `virt` boots via `-kernel` with an FDT, while
VirtualBox-ARM, UTM and real SBSA machines boot the UEFI stub with **no FDT at
all** — and requiring the FDT node left every one of those silently cooperative,
with no timer preemption, forever. An FDT that exists but lacks `arm,gic-v3` is
Apple Silicon: stay cooperative and do **not** fall through to ACPI (there is
none, and probing a guessed base is an uncatchable data abort, not a trappable
UNDEF). The QEMU-`virt` addresses survive only as a last-resort default when an
FDT claims a GICv3 but carries no readable `reg`.

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
  **installed agents** — the built-in roster is `doc`, `ssh`, `chess`, `media`,
  `pdf`, `download`, `todo`, plus the app packages `notes`, `paint`, `slides`,
  `minesweeper`, `snake`, `synth` — each a markdown
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
- **Apps — wasm-tool agent packages.** An "app" is an installed agent whose
  deterministic logic ships as **`assets/tools.wasm`** (string ABI:
  `export(args_ptr, args_len) -> i64 = (ptr<<32)|len`, `chitti_alloc` for host
  writes; run by `agent/wasm_rt.rs` under **fuel + memory limits** from the
  manifest — no host imports unless bound). The SOUL carries judgment; the wasm
  carries rules — chess (`chess_legal`/`chess_try_move` from
  `tools/chess-wasm`), doc's HTTP router (`route_request`, `tools/doc-wasm`),
  pdf's document digest (`pdf_digest`, `tools/pdf-wasm` — xref tables+streams,
  ObjStm, FlateDecode reusing the kernel's `image/inflate.rs`, text
  extraction), the full **chess game** (`tools/chess-wasm` — rules, board UI,
  and the agent-opponent flow; zero chess code in the kernel), and the app
  suite `notes/paint/slides/minesweeper/snake/synth` (one shared module from
  `tools/apps-wasm`). **Chat tool calls run on a fresh instance** — design
  those digest-once (one call returns everything as JSON; the kernel caches)
  and pass binary inputs as base64 — **but a running package-UI app keeps ONE
  persistent instance** (`service/package_ui.rs`): guest statics ARE the game
  state (snake body, mine field, FEN), the guest bump heap resets per call
  cycle in `chitti_alloc`, and no guest static may hold a heap type. UI apps
  paint 256×192 `synapse::ui` surfaces via the draw-op DSL
  (`rect`/`line`/`pixel`/`text` + `board_set`/`board_mark` for boards) with
  per-agent `storage_*` (localStorage-shaped) state; the runtime pump
  (`service/package_ui.rs`) peeks the event queue natively and only drains
  through the audited `ui_event_poll` when events exist (an unpaced poll once
  flooded the audit log at ~1 kHz), forwards clicks/keys to the guest
  `on_click`/`on_key` exports (an app consumes only keys it handles), and
  serves the **model-ask protocol**: any export may return `ask:<prompt>` →
  one model turn over the agent's SOUL → the text back via `on_reply` — the
  wasm builds the prompt and validates the reply, so the model only ever
  chooses (chess enumerates legal moves natively and the agent picks one). Manifests can claim **`command_hooks`** —
  `/open` routes by extension to the owning agent's tool (media owns
  images/audio/video, pdf owns `.pdf`) and rebinds chat to that agent. Build a
  module with `cargo build --release --target wasm32-unknown-unknown` in its
  `tools/<name>-wasm/` crate and copy to `agents/<name>/assets/tools.wasm`
  (checked in; `include_bytes!` at boot). See the wasm-agent recipe gotchas:
  kernel `json_str` unescapes `\n`; `image/inflate.rs` is raw RFC 1951 (strip
  the zlib header) and its `#[test_case]` tests need a shim under host
  `cargo test`.
- **Messaging channels** (`msgchan/`) — external inbox adapters (Telegram
  live; Discord/Slack/webhooks follow the same shape): a named instance +
  backend + access policy delivering inbound DMs into the shell agent and
  replies back out. Each channel turn runs on a **fresh model context**
  (`channel_turn` swaps KV/history out and back) so DMs can't stick to a
  console topic and console chat isn't polluted; transcripts still land in the
  session for audit. Distinct from `channel/` (cap-gated inter-agent pipes).
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
  unsloth file incl. UD-* dynamic mixes runs; plus **PrismML's sub-2-bit packs**
  (`Q1_0` binary type 41, `Q2_0` ternary type 42 — 128-elem blocks, one f16
  scale; the Bonsai-27B builds). Fast SDOT matvecs for
  Q8_0/Q4_0/**Q4_K**/**Q1_0**/**Q2_0** (the Q1_0 sign-expand and Q2_0 code
  unpack run fully in vector registers — `vqtbl1q` broadcast / `vzip`
  interleave, loads via the `ldq_*`/`ldp_*` asm helpers), everything else
  through the generic dequant path (still SMP row-split). **Batched
  (weight-stationary) prefill is decided per *tensor*, never per model**
  (`has_batched_kernel`): Q8_0 ∣ Q1_0 ∣ Q2_0 always have a kernel, Q4_0 only
  with FEAT_I8MM, K-quants none — and a projection without one falls back to a
  matvec per position instead of disqualifying the whole file. The old
  per-model gate demanded one uniform quant type and read it from
  `token_embd.qt`, which is the worst possible anchor: prefill never batches
  `token_embd` (a row lookup, plus one matvec for the final position on a tied
  output). Real GGUFs are almost never uniform — `llama-quantize` upcasts
  selected tensors unless `--pure` is passed, so the file published as "Q4_0"
  arrives Q4_0+Q8_0+Q5_K+Q4_1 with `token_embd` at Q6_K, and that one tensor
  put a whole 4B on the per-token path. **A fallback tensor costs far more than
  its share of bytes**: `ssm_out` at Q5_K has no SDOT matvec either, so it goes
  through the *generic dequant* path — 11% of projection bytes was eating ~69%
  of prefill, ~18x less efficient per byte than i8mm. Hence
  `xtask/fetch-model.sh CHITTI_PURE=1|bf16`, and `/perf`'s `batched weights: N%`
  line. Measured on the 4B (`/perf 512`, 8 cores): per-token 3 pp / 1 tg →
  windowed at 89% batchable → **pure file 23 pp / 6 tg**. The shell chat loop
  feeds **64-token chunks** through `Model::prefill` (weight bytes + unpack
  amortized per chunk; UI pump + Ctrl+C between chunks), so a 27B's ~1.5k-token
  system prompt prefills in minutes, not hours. Chunk size is measured, not
  assumed — sizing it to the heap (128/256) was **slower** on a host sweep
  (4.06 s at 64, 4.17 s at 128, 4.22 s at 256): a bigger chunk only cuts weight
  *traffic*, and prefill is compute-bound while the wider activation tile costs
  cache locality. NB the paired Bonsai `dspark` GGUF is a *drafter*
  conditioned on the target's hidden-state taps — not a standalone model; the
  runnable Bonsai is the main `Q1_0`/`Q2_0` file. Two tokenizer flavors behind
  one API (GPT-2 byte-BPE ∣
  gemma4 raw-UTF-8 ▁-BPE with `<0xXX>` fallback), per-family chat format in
  the shell (ChatML ∣ `<start_of_turn>` gemma turns, BOS per `add_bos`).
  Select with `-model qwen3.5-0.8b|2b|4b|9b|gemma-4-e4b|bonsai-27b|
  bonsai-27b-ternary` **or any path**
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
  draw to its own surface). The audit **log** records every entry
  (append-only, structurally enforced + tested); its **ktrace mirror**
  coalesces identical consecutive entries into one line + a repeat count, so a
  polling loop can't drown the human-facing trace. NB: sessions that use net
  egress or UI input aren't replayable from a seed alone (the I/O is external)
  — the audit log records the effects; treat such a session as
  non-deterministic to replay.
  **What the boundary costs is measured, not asserted** (`synapse::bench`,
  `/bench synapse`): the gate chain is priced through `executor::gate_prefix`,
  which runs the real predicates in the real order but executes no primitive and
  writes no audit entry, against a **synthetic parked task** whose table + scope
  ledger are granted explicitly and killed after — measuring against the shell
  agent would make a figure depend on what the session holds, and pricing a
  *denied* call would mean granting an agent a right to measure it.
  `gate_prefix` is a second copy of the gate order, so
  `gate_prefix_agrees_with_execute` pins it to the real chain: a new gate that
  isn't added to both fails that test. Three benchmark traps, each of which
  printed a plausible wrong number first: a result-discarded pure call is
  **deleted** by the optimizer (the FNV row read 0 ns over 16.7M iterations —
  everything timed goes through `black_box`, and a zero-ms batch is flagged
  SUSPECT rather than printed as 0); cumulative prefixes must share **one batch
  size after a warm-up**, or the first batch pays to grow the heap and the curve
  comes out *decreasing* (making every marginal cost an artifact); and
  `cycle_count` is a **constant-rate tick, not a CPU cycle** (~24 MHz
  `CNTVCT_EL0` on Apple silicon), so the rate is printed beside every figure. A
  non-positive marginal cost prints "below noise floor" — a saturating
  subtraction is not evidence that a gate is free. The design write-up lives in
  [`paper/`](paper/).
  **The taint policy is enforced at EIGHT sites, not one, and `/redteam`
  (`security::redteam`) is the census.** The Synapse executor gates destructive
  primitives; the tool router *separately* gates destructive shell commands,
  `/http`, MCP `tools/call`, agent-memory mutations, downloads, browser
  navigation, nested `run_shell_command`, and the web tools — so a **new tool
  binding that forgets its check is a hole by omission**, and the corpus carries
  one attack per site precisely so that shows up as a permitted attack instead of
  as silence. It runs through the **real `Router`**, so the justification is
  computed by `Router::justification` over `Session::resident_max_taint` exactly
  as in an agent turn; a laundering bug on the way to the gate would surface as a
  `NOT TAINTED` row. Three rules this harness must keep. It **assumes the
  injection persuaded the model** (the payload text is never read by a gate) —
  that is the worst case and it is what makes the numbers deterministic and
  model-independent; it measures the *authorization* boundary, not the planner.
  A **permitted attack really executes** under the baselines, so every target is
  a sandbox path or the loopback discard port, pinned by
  `corpus_targets_are_sandboxed_and_offline` — this is why the destructive-shell
  attack is `rm` on a sandbox file and never `install`, and why the victim runs
  as a throwaway agent identity (`REDTEAM_AGENT`): the memory-poison attack, if
  permitted, writes durable memory that re-enters the system prompt, so running
  it as the live orchestrator would poison the real shell agent as a side effect
  of measuring whether that was possible. And **a non-policy error counts as
  permitted** — a refused loopback connection is not a defence. The counterpart
  measurement is the utility suite: benign tasks over the same primitives, whose
  **false-refusal rate** (a destructive step refused when the untrusted content
  never named its target) is the number that decides whether the policy is
  usable, and it is reported next to the attack rate rather than separately,
  because a defence is only interesting if it is good on both axes.
- **Microkernel** — tasks + context switch, cooperative + timer-preemptive
  scheduler, unforgeable capabilities, IPC, SMP, frame allocator + heap, MMU.
- **UI** — a tmux-style split-pane framebuffer compositor in Geist Mono. The
  shell (chat) pane is fixed in the primary band; the other band is a
  **resizable grid of 1–8 action panes** (`/pane grid <cols> <rows>`, or
  `/pane max <2-9>` for a balanced shape — `panes_layout::grid_for_count`).
  **Every divider drags**: the shell|band split (`/pane split <10-90>`) and each
  grid column/row gap, and a grid drag is **per-gap** — it re-splits only the two
  tracks it separates, so panes you weren't touching keep their exact pixel
  sizes. Either band can go **fullscreen** (Ctrl+F, or `/pane full` —
  `LayoutCfg.fullscreen`, which maximises the **focused** action pane, not pane
  0). Shape, `chat_pct`, and the permille track weights persist to
  `/configs/core/panes.json` and reload at boot (legacy `num_action_panes` is
  still read as an action-pane count). The geometry is **pure and unit-tested**
  in [`kernel/src/panes_layout.rs`](kernel/src/panes_layout.rs) in two steps —
  `split_band` (chat | band, one gap) then `layout_grid` (`cols × rows` cells
  from track weights) — with `band_divider_pct` / `resize_tracks` as their exact
  inverses, asserted by round-trip tests; weights are permille so a saved layout
  restores byte-identically and `GridSpec::sanitized` repairs a hand-edited file
  rather than producing a zero-size pane (`MIN_TRACK_PX` bounds every drag).
  Panes are addressed **row-major** (`index = row * cols + col`).
  **Tabs move between action panes by drag-and-drop** — press a tab label, drag
  past a ~4 px threshold (the target pane highlights), drop on a tab bar to
  insert there or on a body to append; a drop on the shell pane or outside the
  band **cancels**, so the shell can never acquire an action tab. The
  insert-index math is `panes_layout::insert_index` (the same-pane removal shift
  must be applied **before** the clamp, or a drop-at-the-end lands
  second-to-last — a test pins every from/to slot pair).
  With `max_panes == 2` the band still collapses when its last tab closes, so
  the default boot UI is byte-identical to the classic two-pane look; with
  `max_panes > 2` empty panes stay visible as drop targets.
  Note three traps this cost: **a pane's frame carries its selection state**, so
  focusing one must repaint the pane *losing* focus too, and
  `focus_action_column` must repaint **itself** — it has already moved focus to
  the action side, so a following `focus_set(true)` sees no flip and draws
  nothing (which made clicking a pane change the selection invisibly).
  **Opening a view must never move keyboard focus** to the action pane
  (`open_view_slot`): the user typed that command at the composer and is still
  typing there, which is exactly why `action_focused` leaves focus on the chat for
  every mode but the editor — setting `focus_action` on open sent the *next*
  command to the pane instead of the prompt, indistinguishable from the shell
  freezing. Focus moves only on an explicit act (click, `/pane focus`, Ctrl+Tab).
  And the
  per-view painters (`/top`, the audio/video/browser HUDs, the editor,
  `surface_dims_px`) resolve their target pane by **which pane holds their
  tab** (`Screen::mode_dims`), never the focused one — otherwise a `/top` on
  pane 3 paints into pane 1's rectangle as soon as you click elsewhere.
  The compositor pairs the chat
  pane + on-demand **tabbed "action" panes**: opening the ktrace stream, the
  `/top` dashboard, a vim-like editor, an **image viewer** (`/open .png|.jpg`),
  or the **audio player** (`/open .wav|.mp3`) each adds a tab **on the focused
  action pane** (already-open views focus their existing pane + tab instead of
  reopening); a tab bar in each
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
  clock, a blinking caret, **Enter submits a fully-typed command** — the suggestion
  menu stays open while the typed token still matches an entry, so a command name
  typed in full kept *its own* entry highlighted and Enter "accepted" it, appended a
  space, and swallowed the keystroke; every later line was then one out of step, so
  a command silently did not run and it read as a frozen shell. `suggest_would_
  complete` gates the Enter path on whether accepting would change anything beyond
  the trailing separator (Tab is untouched — completing is its job). This is what
  made the command after `/todos open` never execute, an e2e gap that sat unexplained
  for a while; the `pane_grid` scenario now asserts it. mouse cursor + click, **mouse text selection in the
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
  and an on-disk UI config (`/configs/core/ui.json`, `shortcuts.json`).
  **The status bar sits on any edge** — `/statusbar top|bottom|left|right`, the
  `status_pos` key in `ui.json`, and the settings agent's `statusbar` tool; applies
  instantly and persists. The geometry is one pure function,
  `panes_layout::status_split`, which carves the bar off its edge and returns the
  **content rect**; `Screen` stores that rect and *every* pane calculation works
  inside it (`build`, `paint_gutters`, `band_capacity`, and the `band_divider_pct`
  drag inverse) rather than `0..width`/`0..height`. Keep it that way — a layout site
  that reaches for the full desktop is correct only for `Bottom`, where the content
  origin is `(0, 0)` and the split is the identity, so the bug hides until someone
  moves the bar. `left`/`right` make it a **column**: text cannot run across 16
  cells, so `status_segments` splits each template on the runs of 2+ spaces the
  author already used to group fields, `wrap_segment` wraps a long one onto extra
  rows (ellipsizing `${datetime} ${tz}` cut off the *time* — the part anyone reads),
  and the two stacks grow towards each other from the brand mark and the far edge,
  stopping when they would meet. The column width is a **fixed**
  `STATUS_V_COLS`, never fitted to the longest segment: the fields hold live values,
  so a fitted width would relayout every pane — reflowing scrollback — each time the
  clock ticked a digit. The brand — logo, the terracotta `#cc785c` / warm-ink /
  cream palette (fully re-themable from `ui.json`), and typography — is specified
  in [DESIGN.md](DESIGN.md); honour it for any UI change. **Themes**
  (`theme.rs`, `/theme`) are presets layered over `ui.json` (still the single
  source of truth for the live look): bundled JSON in `assets/themes/*.json`
  (`dark`/`light`/`solarized-dark`/`nord`/`dracula`/`ubuntu`), installable to
  `/configs/themes/`, each carrying the chrome palette, `highlight` **syntax**
  colours, **cursor** fill/outline + optional sprite bitmaps, `font`+scale, a
  **wallpaper** (`""` ∣ `gradient:#a,#b` ∣ a store-image path, cover-scaled by
  `image::cover`, `/theme wallpaper` fetches a URL) and **opacity** (0–255).
  **NEW UI SURFACES MUST RESPECT THE THEME BACKGROUND:** with a translucent
  wallpaper (opacity < 255) the desktop must show behind *every* surface, so
  paint pane/region backgrounds through `Screen::paint_surface` and text-cell
  backgrounds through `fill_cell_bg` — **never a raw `fill_rect` of the bg
  colour** — and let glyphs blend via `blit_glyph`→`bg_at`. Both fast-path to
  `fill_rect` when there is no wallpaper / opacity == 255, so the default look
  is byte-identical (no regression). A theme switch recolours existing
  scrollback to the new `default_fg` (`Pane::recolor_default_fg`, called from
  `adopt`). The **only** exception is self-contained app content blitted as its
  own RGB buffer via `present_surface` — the browser, wasm-UI apps/games, the
  image/video viewers — which stays opaque. NB: the scheduler is
  cooperative, so **any long or blocking operation must pump the UI itself** —
  call `shell::upkeep()` (blink + status + mouse + `net::poll`) inside its
  loop, exactly as the per-token inference loops, the ONNX per-node loop, and
  the sliced FAT/ext4 readers do; loops that consume their own mouse events
  (modals, the editor) use `shell::status_tick()` instead. A tight compute
  loop that never yields freezes the clock, mouse, and net stack until it
  returns. Any new UI surface or blocking command must keep this upkeep
  running. The chat pane keeps a 2000-line scrollback (PgUp/PgDn; /clear
  wipes it); Shift+Tab / Ctrl+Tab / clicking switches pane focus.

- **AGX GPU** (`agx/`) — **the Apple-Silicon GPU coprocessor is booted to
  RUNNING on a real M2** (t8112, via `cargo xtask m1n1`; the foundation for GPU
  compute offload of `cortex`'s `matvec_qw`/`batched_proj` — the compute path
  itself is the next milestone, not done yet). `/agx up` drives the full
  bring-up: PMGR `gfx` power-on + SGX liveness poke → `cpu_start` → **GFXHandoff
  PPL handshake** (write `MAGIC_AP`, wait the firmware's `MAGIC_FW` — the shared-
  memory memory-manager handshake, done *before* the RTKit handshake per
  drm/asahi order) + UAT ctx-0 TTBRs under the handoff lock → RTKit **HELLO**
  v-negotiate → HELLO_ACK → EPMAP → START_EP → `AP_PWR_STATE=ON` → service the
  crashlog BUFFER_REQUEST (mapped into the **shared TTBR1 kernel range**, not
  per-context TTBR0 — the firmware's boot context only sees TTBR1) → both IOP+AP
  power reach ON = **RUNNING**. The **UAT** is real ARMv8 16 KiB paging with the
  **G14 geometry — bit 39 TTBR select (IAS=39), NOT bit 47** (`agx/uat.rs`,
  pure + unit-tested); getting that wrong makes the firmware's page-table walk
  miss our PTEs (the buffer is then unreachable and it stalls silently). The
  pure wire protocol (`agx/proto.rs`) + UAT encoder (`agx/uat.rs`) are
  **arch-neutral, unit-tested under `cargo xtask test`** (x86 — `arch::aarch64`
  is cfg-gated out of the test build, so pure logic must live outside it); the
  ASC-mailbox MMIO (`agx/asc.rs`, single-`ldr x`/`str x` FIFO + `dsb`/`dmb`),
  GFXHandoff (`agx/handoff.rs`, Dekker lock + cache-maintained shared mem), and
  discovery/PMGR/orchestration (`agx/hw.rs`) are aarch64-only. Gated on
  `is_apple()` + a `chitti.agx` **bootarg** (pass `chitti.usb chitti.agx`
  together; bare boot only, never under the m1n1 hv — same rationale as
  `chitti.usb`); a clean no-op on QEMU/VBox/other SoC. `/agx status` dumps
  bases/endpoints/power; every wait is bounded + pumps `upkeep()`/`poll_interrupt()`
  and answers Ctrl+C. Ported from m1n1 `src/{asc,rtkit,pmgr}.c` +
  `proxyclient/m1n1/{hw/uat.py,fw/agx/*}` and drm/asahi `gpu.rs`. **Two hard-won
  fixes were decisive:** UAT geometry = bit 39 (G14/t8112), and RTKit shared
  buffers belong in TTBR1 (shared across contexts), not TTBR0 (per-context, not
  active in the firmware's boot context). **Next:** app endpoints 0x20/0x21,
  `initdata` (perf/power tables + channel rings), the firmware command ring, then
  a GEMM microkernel into `cortex`.
- **Storage** — virtio/NVMe/AHCI block devices, GPT/MBR/FAT/ext4 detection,
  ext4 (the default filesystem) + FAT, `/install` (self-hosting install to a
  disk; detects an existing Chitti GPT and **updates in place**, preserving the
  data partition — destructive actions confirm via the permission modal),
  durable agent state on ext4. **Every disk is enumerated, across controllers and
  ports**: `ahci::probe_nth` indexes AHCI *disks* — each HBA on the bus × each
  port that is implemented *and* populated (`Ahci::present_count` counts without
  allocating, so skipping past a disk doesn't bring a port up and leak its DMA;
  `bringup_nth` then takes the one the index names). Ports are sparse on real
  hardware — a drive on port 3 with 0-2 empty is normal — and only taking the
  first controller's first present port made every other disk invisible. Exercise
  the real-hardware storage paths in QEMU with
  `CHITTI_DISK_IF=ahci|nvme|virtio-blk cargo xtask run -arch x86_64`.
  **Installing next to an existing OS.** Plain `/install` still writes a fresh GPT
  and erases the disk — that is the whole-disk path. Two non-destructive commands
  sit alongside it: **`/install plan`** (read-only; reports the partition table,
  free extents, and either a plan or why not) and **`/install alongside`** (x86;
  adds `\EFI\BOOT\BOOTX64.EFI` to the ESP already on the disk, renaming any
  existing loader to `BOOTX64.CHB` as a backup, touching no partition table and no
  partition). Planning is `gpt::{free_extents, plan_alongside}` — gaps computed
  with a high-water mark so an overlapping or contained entry can never make the
  inside of a live partition look free, and the existing ESP is **shared, never
  reformatted**, since a PC has one and rewriting it removes the Windows boot
  manager. The writer is `block::esp` over the pure `block::fat32` layer, and the
  backup is a **directory-entry rename** so the displaced loader keeps its original
  cluster chain — preserved byte-for-byte, no copy, no half-written window; a
  pre-existing backup is never overwritten because that one is the true original.
  Write order is allocate → payload → FAT (to **every copy**, or chkdsk calls the
  volume corrupt and Windows may "repair" it back to the stale copy, undoing the
  install) → directory entry last. Verified by 6 tests against a real FAT32 volume
  on a `RamDisk`; one of them caught the install overwriting Windows' loader
  because the fixture had left its cluster marked free, which is the reminder that
  **the FAT's allocation marks are the only thing protecting existing data**.
  Still absent: no ChittiOS data partition is created by `alongside`, and NVRAM is
  untouched, so firmware boot order may need changing by hand.
  NVMe enumerates namespaces via the **IDENTIFY
  CNS=2 active list**, never "walk NSIDs until empty" — NSIDs are sparse on
  VirtualBox (port→NSID; an empty port 0 = inactive NSID 1, exactly what a VM
  looks like after its install medium is detached); the `nvme: N active
  namespace(s)` ktrace is the first check for "controller up but no disks".
  Relatedly, the aarch64 identity map types **mixed RAM/MMIO GiB blocks** at
  2 MiB L2 granularity from the **real RAM extents the stub passes in
  boot-info** (`mm/ramlayout.rs`, pure + tested): VBox puts the model tail,
  GOP framebuffer and PCIe ECAM in one GiB block, and a whole-block Device
  retype alignment-faults NEON loads (scalar reads still work — "boots fine,
  dies in the matvec" is the signature).
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
  PCI, plus the PCI Ethernet families — discovered the same way on both arches.
  **A PCI NIC is claimed by vendor+device ID, never by vendor alone**
  (`net/nic_ids.rs`, pure + unit-tested): all Intel Ethernet reports
  `8086`/class `02:00:00`, but the families are register-incompatible —
  legacy **e1000** (82540-82547) and **e1000e** (82571…**I217/I218/I219**, the
  NIC in most business laptops) keep the rings at `RDBAL 0x2800`/`TDBAL 0x3800`
  and share `net/e1000.rs`, while **igb** (82575-I350) and **igc** (I225/I226)
  moved them to `0xC000`/`0xE000` with *advanced* descriptors and need
  `net/igb.rs`. Driving an I210 through the legacy path configures reserved
  space: it links and never receives a frame — and having claimed the one NIC
  slot, a second working card is never tried. So `net::pci::probe` walks **every**
  Ethernet function, skips the ones with no driver (logging each), and dispatches
  by table; unknown *Intel* IDs fall back to e1000e (the only open-ended family —
  Intel adds I219 IDs every PCH generation) with the ID ktraced as a guess.
  **Realtek** RTL8168/8111/8125 — the commonest consumer NIC — is `net/r8169.rs`:
  descriptor-owned rings (`DescOwn` in the descriptor, no tail register; kick via
  `TxPoll`), **unverified on hardware** (QEMU models no r8169-family part, only
  `rtl8139`, which is recognised but deliberately not implemented — no
  Windows-era machine has one). Test the dispatch against every family QEMU *can*
  emulate with `CHITTI_NIC=e1000|e1000e|igb|rtl8139|virtio-net-pci cargo xtask run
  -arch x86_64` (the `nic_dispatch` e2e scenario asserts the chosen driver matches
  the emulated device). **USB Ethernet** exists now
  (`net/usb_eth.rs` over the xHCI bulk transport): **CDC-ECM only**, chosen because it
  is the one real standard of the three shapes and puts one frame per transfer with no
  framing header, so the transferred length *is* the frame length. ASIX and Realtek
  dongles are recognised and then **refused with a log** — they need per-chip register
  setup and packet headers, and treating their framed packets as raw would hand smoltcp
  garbage. CDC-ECM is matched by interface class, not an id list (it is a standard, so a
  list would guarantee gaps). Tried **last** in `autodetect`, after virtio and PCI, so a
  built-in NIC always wins. Bulk transport is in `xhci.rs`: `configure_bulk` +
  `bulk_arm_in`/`bulk_take_in`/`bulk_send`; note the delivered length is
  **`requested - residual`** (a transfer event's low 24 bits are the *untransferred*
  count). Still missing for real machines: Broadcom `tg3`, Atheros/Killer `alx`,
  Aquantia.
- **WPA2-PSK and 802.11 frames** (`drivers/wifi/wpa.rs`, `drivers/wifi/ieee80211.rs`,
  `net/sha1.rs`) — the supplicant a SoftMAC radio needs, **built entirely as pure,
  vector-pinned logic**, because joining a Wi-Fi network is code where a bug is
  invisible: every step produces bytes as random-looking as the correct ones and the
  only feedback is that the access point stops answering, which is indistinguishable
  from a wrong password. So nothing here waits on a radio to be checkable. SHA-1 is
  deliberately absent from the TLS path (broken for signatures) and exists **only**
  because IEEE 802.11i mandates it — pinned to FIPS 180-2, RFC 2202 and the 802.11i
  Annex H PSK vectors; then PBKDF2→PMK, PRF-384→PTK (KCK/KEK/TK), the EAPOL-Key MIC,
  and AES-128 + RFC 3394 key unwrap for the group key. Four things are silent when
  wrong and each has a test: the MAC addresses and nonces concatenate **smaller
  first** (both sides do it, neither transmits the result, so a mistake is a
  self-consistent PTK the AP disagrees with — reported to the user as a wrong
  password); the PRF puts a **NUL** between label and data; the MIC is the first 16
  bytes of HMAC-SHA-1 over the frame with **its own MIC field zeroed**; and PBKDF2
  counts **bits** and must hash an over-long HMAC key first (every short vector passes
  without that branch). `Handshake` is the four-way exchange as a pure state machine
  over frames — `on_frame` in, reply out — so the failure paths are the tested ones:
  a replayed message 1 (which carries no MIC, so the replay counter is the only
  defence), a message 3 whose group key fails its integrity check, and the **ordering
  that matters**: the ANonce is checked *before* the MIC, because a changed ANonce
  fails the MIC too and checking the MIC first blames the passphrase. `ieee80211`
  parses beacons/RSN elements — attacker-controlled bytes from an unauthenticated
  sender, so every length is a claim and a lying one is refused, never clamped — and
  reports TKIP/SAE/802.1X/required-MFP as **unsupported up front** rather than as a
  timeout. Reachable and checkable by a user: **`/wifi psk <ssid> [passphrase]`**
  prints the derived key, and `wpa_passphrase` on any Linux box is an independent
  oracle for it (e2e `wifi_psk` asserts the published vectors on the running kernel).
- **Intel WiFi** (`drivers/wifi/iwl/`) — the part in most x86 laptops. Staged, and the
  stages are the point: `fw` (family from the PCI id, firmware filename search order,
  `.ucode` TLV parse), `csr` (registers + pure predicates), `context` (the gen2
  **context info** — the device's own loader fetches firmware out of host memory once
  it has that structure's address), `proto` (command out / notification in), `device`
  (the sequences and the queue). `/wifi up` resets the radio, hands over firmware,
  waits for the **alive** notification *and checks its status word* (firmware that comes
  up unusable still announces itself), reads the MAC out of the strap/OTP registers, and
  sends one **read-only** command (`NVM_GET_INFO`) — read-only on purpose, since a first
  command that configured something would misconfigure a real radio if any part of the
  untested transport is wrong. Traps pinned by tests: the MAC's first word is
  **big-endian** and the second contributes only its low two bytes reversed
  (`to_le_bytes`-and-concatenate gives a plausible wrong address); `prph` addresses are
  **20 bits** (the hardware supplies the `0xA00000` base, so truncation is correct);
  the transmit doorbell packs queue id and write index in one word; a command's first
  **20 bytes** must come from a separate aligned staging buffer; and a receive buffer
  must be **handed back** (the free-list write index is a *count*, not a slot) or the
  driver works during bring-up and goes deaf under traffic. **It cannot scan or
  associate**, and that is deliberate: a scan request, MAC context and station key are
  large per-API-version structures, no emulator provides an Intel WiFi part, and code
  written from memory would look complete, send well-formed garbage to a real radio and
  report success — the failure would be somebody's laptop rather than a missing feature.
  Those need a machine with the part in it. **Every layout here comes from Linux's
  `fw/api/*.h`, fetched, never recalled** — the one written from memory (`NVM_GET_INFO`'s
  general section as four `u32`s instead of `u32,u16,u8,u8`) put `n_hw_addrs` on the
  transmit chain mask: a small, plausible number that passes any sanity check, i.e. a
  confident wrong answer rather than an error. The groundwork for adding a command
  safely is `fw::cmd_version` (`IWL_UCODE_TLV_CMD_VERSIONS`, TLV 48), the table where the
  image states which request version it expects; a new command must consult it and
  **refuse** an unimplemented version, since silence in that table is *not* version 0.
  Also still absent: Realtek RTL8852 and Qualcomm/Killer WiFi, and Broadcom's SoftMAC
  parts.
  Shell surface: `/network` (info/dhcp/static/dns), `/ping`,
  `/wifi` (scan/connect via the password modal), a **TCP listener**
  (`net::listen`/`try_accept`, backed by a pool of Listen-state sockets in
  *both* the NIC and loopback sets, so one listener serves external/hostfwd and
  `localhost` clients alike; accept hands out an Established `TcpHandle` a service
  agent adopts as a channel), `/http` (a curl-like
  HTTP/1.1 client in `net/http.rs` — `-X`/`-H`/`-d`/`-v`/`--stream`, all
  methods, live chunked/SSE streaming; `http://` **and** `https://` via
  `net/tls.rs`/embedded-tls with **real certificate verification** (see the
  TLS-trust bullet); also the agent's `http` tool; **`-O`/`-o <file>`
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
- **TLS certificate trust** (`net/x509.rs`, `net/rsa.rs`, `net/hashes.rs`,
  `net/ca_roots.rs`) — HTTPS server certs are **verified by default** against an
  embedded **Mozilla root store** (121 roots, `tools/gen_ca_roots.py` →
  `ca_roots.der` + spans; regenerate from `cacert.pem`). `ring` **cannot build
  bare-metal** (C + asm; that's why `NoVerify` shipped originally), so the
  validator is pure RustCrypto: `x509-cert` parses DER, `p256`/`p384` verify
  ECDSA, and [`net::rsa`] does **RSA PKCS#1 v1.5 + PSS** on `crypto-bigint`
  (a fixed `U4096` modexp — no `rsa`/`num-bigint-dig`/`ring`). `x509::verify`
  builds a chain leaf→intermediates→trusted root (each link's signature checked
  with the issuer key), checks each cert's validity window against the wall
  clock (refuses if the clock is unset — set `/datetime`), CA basic-constraints
  on issuers, and the leaf's SANs vs. the hostname (wildcards). The TLS 1.3
  `CertificateVerify` (RSA-**PSS** mandatory there — the bug that first failed
  css.tobyase.de) runs through the same code via `x509::verify_data`. A
  `ChittiVerifier` (`net/tls.rs`) implements embedded-tls's `TlsVerifier`
  (vendored crate given the few needed `pub` re-exports); `/tls insecure on` is
  the `curl -k` escape hatch (human-only) for a self-signed/self-hosted box.
  **Out of scope (documented, not silently skipped):** CRL/OCSP revocation.
  Validated by KATs (`rsa_testvec.rs`, `gen_rsa_testvec.sh`) + a real embedded
  chain (`ca_testvec.rs`, `gen_ca_testvec.sh`: verifies, and tampered/expired/
  wrong-host fail closed) + live `/http https://…` to real providers.
- **Sound & voice** (`sound/`, `onnx/`) — virtio-snd PCM in/out (S16 mono,
  poll-driven, descriptor chains) over virtio-mmio (aarch64) and virtio-PCI
  (x86 QEMU), **Intel HDA** for VirtualBox (x86+ARM) and real hardware, plus **AC'97** and **Sound Blaster 16** (x86 legacy; via `mm::alloc_dma_bounded`, which asks the frame allocator for the 8237's real constraints — under 16 MiB and inside one 128 KiB block — instead of allocating normally and hoping, which never held and made the driver unreachable code); `/voice` (waveform modal, level-gated utterances) and `/voice test`
  (tone + mic check). **`audio/`** is the pure media-decoder layer behind the
  `/open <file>.wav|.mp3|.aac` **player**: a full RIFF/WAVE parser (PCM
  8/16/24/32-bit + float32, any channel count downmixed), an MPEG Layer III
  decoder — a no_std Rust **port of minimp3** (CC0; tables generated verbatim
  by `tools/gen_mp3_tables.py`), validated ±1 LSB against minimp3's own scalar
  decode (stereo MS/short-blocks, MPEG-2 LSF, bit reservoir) — and an **AAC
  decoder** (`audio/aac/`, a Symphonia-path port under MPL-2.0: LC + Main/LTP
  ICS, ADTS demux, HE-AAC SBR/PS) that also supplies the video player's audio
  track. Playback feeds
  the device in ~50 ms chunks — queue backpressure paces it — pumping
  `upkeep()` and answering Ctrl+C between chunks. `onnx/` is a zero-copy no_std ONNX (protobuf) reader +
  **op interpreter** that runs the real voice models end-to-end: silero-vad v5
  (VAD), parakeet-ctc int8 (STT — `/voice stt <wav>` transcribes), and
  KittenTTS (TTS — `/voice say <text>` speaks); bare `/voice` is the full
  mic → VAD → STT → LLM → TTS conversation loop. Models load lazily from any
  disk volume (bundled in the images; `cargo xtask voice-assets` downloads
  them into `assets/voice/`, gitignored). **The ONNX ops parallelize across the
  SMP fleet** (`onnx::exec::par_range` → `smp::parallel_for`: conv1d tiles,
  conv_transpose gather+dot, matmul rows/cols, strided broadcast, `par_map`
  unary) and `/voice say` is **chunked** — `split_speech` synthesizes clause by
  clause into the `speech_pump` queue (fed from `ui_tick` via the non-blocking
  `SndDevice::out_free_bytes`), so audio starts in ~3 s and streams while the
  next clause synthesizes. For any numeric or perf work on this
  path, use `tools/onnxdiff/` (host-side layer-by-layer diff of the kernel's
  own interpreter against onnxruntime) — not QEMU round trips. NB: kitten's
  DynamicQuantizeLinear means outputs are only comparable by *equidistance
  from onnxruntime*, never bit-exact; any float reassociation flips int8
  rounding.
- **Remote voice** (`shell/voice_remote.rs`) — hosted TTS/STT providers as an
  alternative to the local ONNX models, same posture as `/model remote`:
  human-configured key (`/voice remote tts|stt <provider> <key> [voice]
  [model]`), persisted at `/configs/core/voice.json`, never an agent tool.
  Providers: **ElevenLabs, Cartesia, Inworld, Sarvam**, and any
  **OpenAI-compatible** `/v1/audio/{speech,transcriptions}` (base via
  `url@model`). Each is a pure request-builder + response-decoder (unit-tested
  wire shapes); TTS audio comes back as WAV/MP3 bytes (`audio::decode`) or
  base64-in-JSON (Inworld/Sarvam), resampled to the device rate and fed into
  the **same chunked `speech_pump`** — so remote synthesis streams per clause
  too. STT uploads the utterance as `multipart/form-data` WAV. `voice_say` /
  `voice_stt_file` prefer a configured remote endpoint, else fall back to the
  local model. **TLS caveat:** all providers are HTTPS and the in-kernel TLS
  client (`net/tls.rs`, embedded-tls TLS 1.3, no cert verification) doesn't
  interop with every server yet (RSA cert chains fail) — a provider that won't
  handshake reports a TLS error, not a wrong result.
- **Video** (`video/`) — H.264/AVC **baseline + Main/High-profile decoder +
  player** for `/open .mp4|.mov|.mkv|.webm` (hls/ts pending), built **in
  stages, each pure + unit-tested off-hardware** and **validated bit-exact
  against ffmpeg/PyAV via the `tools/h264diff/` host harness** (mounts
  `video/*.rs` via `#[path]`; the onnxdiff/cortexdiff pattern — the CAVLC VLC,
  deblock alpha/beta/tc0, and all CABAC tables are parsed/generated from the
  FFmpeg sources, `tools/gen_cabac_tables.py`, never hand-transcribed).
  **CABAC / High profile** (`h264/cabac.rs` engine + generated
  `h264/cabac_tables.rs` + `h264/decoder_cabac.rs`): I/P/**B** slices, adaptive
  **8x8 transform** + Intra_8x8, a POC-ordered multi-frame **DPB** with
  ref-list construction/reordering + sliding window/MMCO-1, **explicit
  weighted P**, **implicit weighted bi-prediction**, **spatial + temporal
  direct** — validated **bit-exact on full real-world clips**: 171/171 frames
  (sample-5s-720p: High, pyramid B, weightp) and 300/300 (Big Buck Bunny 720p:
  High, temporal direct, 16-ref). Hard-won availability rules: MV prediction
  may only use 4x4 cells whose motion **for that list** is final (per-list
  `mvok` stamps — an above-right cell of a later same-MB partition is
  *unavailable*, while a partition not using the list becomes
  available-with-ref -1 immediately, FFmpeg's LIST_NOT_USED fill); the
  **ref-idx context** reads refs-as-parsed (pre-MV); the B mb_type context
  tests the *MB-level* direct flag, not the per-4x4 one. Unsupported CABAC
  features (interlaced/MBAFF, FMO, scaling matrices, I_PCM-in-CABAC, long-term
  refs, poc type 1) are *refused* cleanly, never mis-decoded.
  **Full baseline pipeline:**
  `mp4`/`mkv` demux → CAVLC residual (`h264/cavlc.rs`) → **I + P** macroblock
  decode (`h264/decoder.rs`: I_4x4/I_16x16/I_PCM, P_L0_16x16/16x8/8x16/8x8/Skip,
  **multiple slices per frame** with slice-aware neighbour availability) → intra
  (`h264/intra.rs`) + **inter** (`h264/inter.rs`: median MV prediction + 6-tap
  luma / bilinear chroma MC) → inverse transform (`h264/transform.rs`) →
  **in-loop deblocking** (`h264/deblock.rs`) → YUV→RGB → a **video tab** with
  a **player HUD** (`framebuffer::draw_video_status`: state, mm:ss, frame
  counter, scrubber, mute, shortcut hints — drawn *after* each frame blit) and
  transport controls (Ctrl+Tab focus, space pause, ←/→ seek, ↑/↓ ±10 frames,
  0 restart, `m` mute, Ctrl+C stop), frame-paced by pts. The HUD sits in a
  **reserved bottom strip** (`present_surface_reserve` + `video_hud_height`) the
  per-frame blit never repaints, its text **wrapped to the pane width** and
  repainted in place — so it neither flickers nor overflows. **Streaming
  decode:** `video::StreamDecoder` holds the source + sample table and decodes
  on demand (`seek_decode`, rewinding to the latest keyframe on a backward
  seek) — a whole-clip `Vec<Frame>` would be ~700 MB of RGB for a 1300-frame
  480p clip (heap-hostile; trap #3). Baseline keeps **one** reference frame;
  CABAC keeps the DPB plus a bounded **reorder cache** (pictures pending their
  display slot, keyed by decode index — without it every backward hop of the
  B-pyramid display order re-decodes from the previous IDR, O(n²)). Display
  order comes from the container (`ctts`-adjusted `Sample.cts`, stable-sorted). **The "green frames"
  bug** was NOT memory: a P-slice that **ends with a trailing `mb_skip_run`**
  (skips the final MBs, no coded MB after → `more_rbsp_data()` goes false
  mid-drain) left the last MB(s) at plane-init 0 → black luma / green chroma,
  which inter-prediction then propagated into a growing green region. Fix: keep
  draining inferred skips while `skip > 0` even past `more_rbsp_data()`
  (`decoder.rs` MB loop). Lesson: **render + eyeball vs PyAV, don't trust a
  single green-fraction metric** (the first scan's threshold missed it).
  **Audio:** `mp4::parse_audio` demuxes the AAC (`mp4a`/`esds`
  → AudioSpecificConfig) track, and the **AAC decoder in `audio/aac/`**
  (a Symphonia-path port, MPL-2.0 — see THIRDPARTY-LICENSES.md — plus ADTS
  demux and HE-AAC SBR/PS reconstruction) turns it into mono S16 PCM at open
  (`open> audio ready: …`); playback keeps the PCM cursor pts-locked to the
  video clock (`pump_video` snaps drift > ~50 ms), and `m` mutes. **Validated
  bit-exact against PyAV/ffmpeg** — synthetic x264 clips (I/P, multi-slice,
  deblocked) and hundreds of consecutive frames of real-world mp4/mkv. In-kernel
  fixture tests hash an embedded I-only and an I+P clip against PyAV, and
  `stream_decoder_seek_matches_sequential` proves random/backward seeks match a
  sequential decode frame-for-frame.
  **Deblock gotchas (all three bit us):** a chroma edge's QP is `avg(qpc(QPp),
  qpc(QPq))` not `qpc(avg(QPp,QPq))` (differs only across slices with differing
  QP); the luma normal filter's `tc = tc0 + (ap<β) + (aq<β)` can be nonzero
  even when `tc0==0` (don't force-skip); and the recycled `Fx` workspace must
  not leak the previous frame's motion into deblock — `bs_inter2` infers "list
  used" from `refpoc != MIN`, and a P slice never writes L1, so
  `mark_mb_decoded` clears `refpoc` (not just refidx/mv) for unwritten cells.
  Bisect any deblock-vs-PyAV divergence with `H264Dec::no_deblock` against
  PyAV `skip_loop_filter='ALL'`, and read FFmpeg's per-block list usage via
  `flags2=+export_mvs`. **Stage 1 (done):** `video/bits.rs`
  (RBSP emulation-prevention unescape + a big-endian `BitReader` with H.264
  Exp-Golomb `ue`/`se`/`te`), `video/mp4.rs` (ISO-BMFF box-tree demuxer →
  `avcC` SPS/PPS + the `stsz`/`stsc`/`stco`/`stts`/`stss` sample table assembled
  by the pure `build_samples`), and `video/h264.rs` (Annex-B **and** AVCC NAL
  splitting + SPS/PPS parse → geometry/profile/entropy mode). `video::probe`
  reports a stream (container, codec, `W×H`, frame count, CAVLC/CABAC) without
  decoding pixels; `/open clip.mp4` shows it. Scope: **H.264 baseline** (I/P
  slices, CAVLC, 4:2:0) **plus Main/High CABAC** (see above).
  **Playback performance (all profiled first — `sample` on the host harness,
  `video: perf:` ktrace in-kernel):** in-kernel 1080p went 12 → **~30 fps**,
  4K ~8–10 fps, via three stacked levers: (1) **NEON luma MC** in
  `third_party/rust_h264/inter_pred.rs` — the 6-tap interpolator was 40% of
  decode; all vector loads/stores are inline-asm `ldr/str d/q` (`ld8/st8/
  ld16i/st16i` — the `+strict-align` rule; the asm loads beat the `vld1`
  intrinsics even host-side), and every hv quarter-pel case runs h-FIR once
  per row into an i16 block buffer instead of six FIRs per pixel (bit-exact:
  full-clip A/B byte-identity + PyAV); (2) **decode-ahead** — the pump loans
  the `StreamDecoder` to an SMP worker (`smp::async_submit`, the reserved
  last slot; dispatchers exclude it via `fleet_workers` + zero its ranges) and
  holds the finished frame until its pts (`VideoPlayer::ahead`); every other
  `dec` toucher joins first (`video_job_join`) — beware pipeline bubbles: the
  worker must be resubmitted on the same tick it's collected; (3)
  **frame-drop** (`sample_is_nonref`, pure + unit-tested): behind the clock,
  non-reference backlog samples are never fed to the decoder at all — but
  catch up in ≤2-frame steps; one giant hurry-jump decodes every backlog
  reference in one job and starves presentation (4K: 8 → 3 fps).
  **Remaining:** HLS/TS demux, 4K ≥ 15 fps (needs parallel
  reconstruction — slice-parallel + independent non-ref B's; CABAC parse is
  the serial floor), and the multi-pane split + tab drag-drop. NB: the e2e stdlib muxer writes no
  `ctts`, so its B-frame clips carry no display-reorder info — e2e CABAC clips
  use `--bframes 0`; the in-kernel fixture decodes an I/P/B clip in decode
  order and a media-key rule: a focused-but-stopped video tab must not eat
  keystrokes (`media_key` gates on `video_loaded()`).
  **Host reference for the numeric stages:** PyAV
  (self-contained ffmpeg) decodes the same clip to YUV for a frame-by-frame diff
  harness (`tools/h264diff/`, the onnxdiff/cortexdiff pattern — mounts
  `video/*.rs` via `#[path]`, runs on the host in seconds, no QEMU round-trips).
  The e2e `open_video` scenario muxes a real x264 baseline multi-slice clip into
  mp4 (stdlib muxer) and asserts the on-kernel probe + streaming decode ("N
  frame(s), ready in …") + transport controls; it auto-skips where x264 is absent.
- **Agent chat protocol** — the shell chat is an agentic ReAct loop on the
  Qwen3.5 template: the prompt advertises a small CORE tool set plus
  `search_tools` (progressive discovery over the registry — manifest
  toolset ∩ `tools::registry`; never hardcode a tool list in a prompt),
  `<tool_call>` JSON in, `<tool_response>` back, thinking off by default
  (`/think`), `/mode manual|auto|bypass` gates agent tool calls through the
  modal, Ctrl+C/Esc cancels prefill *and* decode, `/compact` rebuilds the KV
  from a model-written summary. Agents are processes: `/agents` lists the
  scheduler tasks that carry agent identity, `switch` re-homes the chat,
  `kill` revokes a task's capability table. Every agent has
  `/agent/<id>/{SOUL.md,skills/,memory/}`; SOUL.md is prepended to its system
  prompt. The shell agent is the only default agent (boot demos removed).

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

- **The OS is named `ChittiOS` — one word, no space.** Use it everywhere: docs,
  boot banner, status bar, SOULs, served pages, commit messages. "Chitti" alone
  refers to the project/brand; the spaced two-word form is wrong — fix it on
  sight (and rebuild any `tools/*-wasm` module whose strings embed it).
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
