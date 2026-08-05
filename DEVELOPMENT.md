# Development guide

How to build, run, and test ChittiOS locally on both architectures. Everything
goes through `cargo xtask`; the target arch is always explicit
(`-arch x86_64|aarch64`), never host-detected.

> Chitti is a research OS under active development. Run it in a VM. See the
> caution in [README.md](README.md).

## Prerequisites

- **Rust nightly** with `rust-src` and `llvm-tools-preview` (pinned in
  [`rust-toolchain.toml`](rust-toolchain.toml); `rustup` installs it on first use):
  ```sh
  rustup toolchain install nightly --component rust-src --component llvm-tools-preview
  ```
- **QEMU** with both system emulators:
  ```sh
  brew install qemu            # provides qemu-system-x86_64 AND qemu-system-aarch64
  ```
- **x86_64 boot** needs **Limine** + **xorriso** (to build the bootable ISO):
  ```sh
  brew install limine xorriso
  ```
  `xtask` finds Limine's boot files via `brew --prefix limine`. If Limine is
  installed elsewhere:
  ```sh
  export CHITTI_LIMINE_SHARE=/path/to/limine/share/limine   # limine-bios.sys, BOOTX64.EFI …
  export CHITTI_LIMINE_BIN=/path/to/limine/bin/limine        # the `limine` deploy tool
  ```
- **aarch64 native boot** runs under **QEMU + HVF** on Apple Silicon (no extra
  setup). The **UEFI boot path** (booting a disk/ISO via firmware, and the
  `stub/` bootloader) additionally needs **AAVMF** (`edk2-aarch64-code.fd`,
  `edk2-arm-vars.fd`), which ships with the Homebrew `qemu` formula.
- **VirtualBox** (optional) — to test the UEFI/real-hardware path on an
  Apple-Silicon host. See "VirtualBox" below.

## Architectures

The same kernel builds and boots on both:

- **`-arch aarch64`** boots directly (`-M virt -kernel`) under
  `qemu-system-aarch64 -accel hvf`, running **natively** on the M-series CPU with
  NEON — fast inference. This is the primary dev loop on Apple Silicon.
- **`-arch x86_64`** boots the Limine ISO under `qemu-system-x86_64` (TCG on an
  Apple-Silicon host — correct, but inference is slow). This is what the in-kernel
  test suite runs on.

## Commands

A [`Makefile`](Makefile) wraps the common flows (`make help` lists them);
`make` targets take `ARCH=`, `MODEL=`, and `RELEASE=1` knobs and just call the
`cargo xtask` commands below.

| `make` target | Underlying command |
|---|---|
| `make build` / `make build-all` | `cargo xtask build -arch <arch> …` (build-all does both arches) |
| `make run` / `make run-uefi` | `cargo xtask run -arch <arch> …` (run-uefi adds `--uefi`) |
| `make image` | `cargo xtask image -arch <arch> …` |
| `make test` | `cargo xtask test` (in-kernel unit suite) |
| `make e2e` / `make e2e-full` | boot the real kernel + drive the shell over serial (`--slow` adds model/voice) |
| `make verify` | x86 build + `test` + aarch64 build (the standing-rule gate) |
| `make vbox` | rebuild the aarch64 image and reload it into a VirtualBox VM |
| `make sample-files` | `cargo xtask sample-files` — fetch the `/samples/` corpus (see below) |
| `make model` / `make fmt` / `make clean` | fetch the GGUF / format / clean |

The underlying `cargo xtask` commands:

| Command | What it does |
|---|---|
| `cargo xtask build -arch <arch> [--release] [-model <m>]` | Cross-compile the kernel. |
| `cargo xtask run   -arch <arch> [--release] [-model <m>]` | Build the image and boot it in QEMU (serial to stdio + a graphical window). |
| `cargo xtask image -arch <arch> [-model <m>]` | Assemble a bootable image (x86: hybrid BIOS/UEFI ISO; aarch64: a GPT disk image that boots standalone via UEFI). |
| `cargo xtask test` | Run the in-kernel `custom_test_frameworks` suite under `qemu-system-x86_64`, headless, asserting via serial + `isa-debug-exit`. **Keep it green** (currently ~420 cases). |
| `cargo xtask sample-files [--refresh]` | Download the `/samples/` corpus into `assets/samples/` (cached). Called automatically by the build paths when `CHITTI_SAMPLE_FILES` is set. |

`-model qwen3.5-0.8b` (the default), `-model qwen3.5-2b|4b|9b`, **or any
`.gguf` path** selects the bundled model and its heap-size tier. The memory
*layout* is **not** hardcoded per model: the kernel discovers RAM at boot (the
DTB on `-kernel`, the UEFI stub's boot-info on real hardware/VirtualBox) and
places the heap accordingly, so any `-m`/VM size works and a model that won't fit
is reported as "not enough memory" rather than corrupting memory. `-arch aarch64
--uefi` boots via the `stub/` UEFI bootloader under AAVMF instead of `-kernel`.

## The model

The model is **not committed** (it's hundreds of MB / a few GB, loaded as a boot
module or placed in guest RAM). Fetch one:

```sh
xtask/fetch-model.sh                 # default: Qwen3.5-4B GGUF (Q4_0, ~2.6 GB)
xtask/fetch-model.sh qwen3.5-0.8b    # the compact model (Q8_0, ~812 MB)
```

`cargo xtask test` never loads the model (the fast unit suite validates tensor
kernels with handcrafted per-quant blocks and fast-vs-generic cross-checks).
Model numerics are validated by the host-side harnesses: `tools/cortexdiff/`
(mounts `kernel/src/cortex` natively, greedy-decode diff vs llama.cpp — the
required bring-up tool for a new family or quant) and `cargo xtask ref-check`
(the on-target acceptance gate; fixtures generated by cortexdiff).

## The standing rules (verify every change)

1. **Dual-arch parity.** A change must build and work on **both** arches.
   After any change run **all** of:
   ```sh
   cargo xtask build -arch x86_64 && cargo xtask test      # keep it green
   cargo xtask build -arch aarch64
   cargo xtask run   -arch aarch64                         # if the change is boot-visible
   make e2e                                                # if the change is boot-visible or networked
   ```
2. **Real hardware, nothing emulator-specific.** Drivers discover hardware the
   way firmware does (ACPI/PCIe ECAM, UEFI GOP, fw_cfg, HID report descriptors,
   PrimeCell IDs) and fall back gracefully. Don't hardcode addresses/resolutions/
   layouts to QEMU or VirtualBox. The same image must run on QEMU, VirtualBox,
   and real UEFI hardware.

## Testing & verifying locally

### The test suite

```sh
cargo xtask test
```

Cross-compiles the `--test` kernel and boots each test binary in QEMU headlessly,
translating `isa-debug-exit` into a pass/fail exit code. Deterministic (fixed
seeds, temperature 0). This is the gate — keep it green. It covers the **pure
logic** (parsers, codecs, capability/scope math, the channel ring, the UI
draw-op rasterizer, HTTP request parsing, P-256 verification, …); it never loads
the model or touches hardware.

### End-to-end tests

```sh
make e2e                 # os + agents + net groups (~3 min)
make e2e-full            # + local inference (--slow) and voice; needs assets
```

`tests/e2e/` (stdlib-only Python) boots the **real kernel** under QEMU and drives
its shell over the serial console, asserting on real output. It covers what only
exists on the running OS:

- **os** — every `/`-command prints its marker.
- **agents** — install-with-consent + capability subsetting, service lifecycle
  (`/agents services`, `start-net`, `start-http`), the **network** and
  **HTTP/Doc** service agents (the host reaches guest listeners over an opt-in
  slirp host-forward — `CHITTI_HOSTFWD=<ports>`, wired automatically by the
  harness), **registry** search + install, and **UI surfaces**.
- **net** — DHCP, ping, `/http` GET/POST/stream, `ws`/`wss`, hosted-model chat.

TLS scenarios need a TLS-1.3 Python (Homebrew's, not macOS system LibreSSL) and
auto-skip otherwise; `--slow` model/voice scenarios auto-skip when their assets
are absent. **Adding a shell command or a networked/service/UI feature means
adding an e2e scenario** — see `tests/e2e/README.md`.

### Booting interactively

```sh
cargo xtask run -arch aarch64        # graphical window + serial on stdio (Ctrl-A X to quit)
cargo xtask run -arch x86_64
```

You get the split-pane console: a chat pane (type a message, or `/help` for
commands), an on-demand action pane (`/ktrace`, `/open <file>`), the status bar
clock, mouse, and keyboard.

### Headless framebuffer verification (screenshots)

To inspect the framebuffer without a display — and to drive keyboard/mouse
programmatically — launch QEMU yourself with a QMP socket and use `screendump` +
`input-send-event`. Example (aarch64 `-kernel`):

```sh
ELF=kernel/target/aarch64-chitti/release/chitti-kernel
qemu-system-aarch64 -M virt -cpu host -accel hvf -smp 4 -m 2G \
  -device ramfb -device virtio-keyboard-device -device virtio-tablet-device \
  -fw_cfg name=opt/chitti/fbres,string=1440x900 \
  -serial unix:/tmp/ser.sock,server,nowait \
  -qmp unix:/tmp/qmp.sock,server,nowait \
  -display none -kernel "$ELF" &
# then over /tmp/qmp.sock: {"execute":"qmp_capabilities"} → {"execute":"screendump","arguments":{"filename":"/tmp/fb.ppm"}}
```

- Drive the **keyboard** over the serial socket (bytes → the shell) or via QMP
  `send-key`.
- Drive the **mouse** via QMP `input-send-event` (`abs`/`btn`) — routes to the
  virtio-tablet / usb-tablet.
- Convert the PPM: `sips -s format png /tmp/fb.ppm --out /tmp/fb.png` (macOS) or Pillow.

The framebuffer resolution is not hardcoded: on `-kernel` it comes from the
`opt/chitti/fbres` fw_cfg value `xtask` derives from your display; on UEFI it
comes from GOP.

### VirtualBox (real-firmware path on Apple Silicon)

VirtualBox exercises the UEFI/real-hardware drivers (USB HID keyboard + mouse,
GOP, RTC, NVMe).

**First time:** create an **ARM** VM (EFI enabled) named `Chitti` with an NVMe
controller, set the **pointing device to USB Tablet** and the **keyboard to USB**
so the xHCI/HID drivers pick them up, then attach a disk built from
`target/chitti-aa64.img`.

**Every rebuild after that:**

```sh
make vbox                    # or: make vbox VBOX_VM=YourVMName
```

`make vbox` rebuilds the aarch64 image, powers the VM off, reconverts the image
to a VDI **preserving the disk's UUID** (so the VM's attachment stays valid),
reattaches it (`--storagectl nvme --port 0`), and prints the VM's input config.
Override the VM name / controller / port with `VBOX_VM=`, `VBOX_CTL=`, `VBOX_PORT=`.
Then start the VM.

Equivalent manual steps, if you prefer:

```sh
cargo xtask image -arch aarch64                                  # → target/chitti-aa64.img
VBoxManage convertfromraw target/chitti-aa64.img chitti.vdi --format VDI
# attach chitti.vdi to the ARM VM (USB Tablet + USB keyboard, EFI enabled)
```

The boot log's `INPUT` line reports which keyboard/mouse/clock sources were found
— the ground truth when input or the clock misbehaves on a given platform.

**Screen resolution.** The console follows the display, not a constant: the stub
reads the monitor's **EDID** and uses its preferred (native) timing, and when there
is no EDID — the usual case on a hypervisor — it keeps whatever mode the firmware is
in. To override it on either arch, pin one at image-assembly time:

```sh
make vbox VBOX_RES=1920x1080                                  # aarch64 + VirtualBox
CHITTI_RESOLUTION=1920x1080 cargo xtask image -arch x86_64    # x86 (appends to limine.conf)
CHITTI_RESOLUTION=1920x1080 cargo xtask image -arch aarch64   # aarch64 (writes the ESP pref)
```

On aarch64 that writes `\chitti-display.cfg` (`resolution=<W>x<H>`) onto the image's
ESP; the stub reads it and calls GOP `set_mode` before the kernel starts, ahead of
even the EDID-native mode. **Do not rely on the hypervisor's own knob:**
VirtualBox-ARM stores `VBoxInternal2/EfiGraphicsResolution` and boots the guest at a
different size anyway, which is why this path exists.

A request is a **ceiling**, never exceeded — ask for 1280x720 where the firmware
offers only 640x480/800x600/1024x768 and you get 800x600, because 1024x768 is taller
than 720. The stub logs the firmware's whole mode list and says when the size asked
for was not on offer, so `resolution I asked for did not happen` tells you *which*
of the two causes it was:

```text
chitti-stub: chitti-display.cfg asks for 1280x720
chitti-stub: GOP current 800x600, 3 usable mode(s): [(640, 480), (800, 600), (1024, 768)]
chitti-stub: 1280x720 is not offered; closest that fits is 800x600
```

The kernel's boot banner prints the resolution it came up at — check that line first
if the size is wrong.

**VirtualBox draws the guest 1:1, so an oversized framebuffer is clipped, not
scaled.** A 2560x1440 guest in a 1440-wide window shows about half of itself and
looks like a broken console. Either shrink the framebuffer (`VBOX_RES` above) or
scale the window host-side, which changes nothing in the guest:

```sh
make vbox VBOX_SCALE=0.5         # or: VBoxManage setextradata "Chitti" GUI/ScaleFactor 0.5
```

**"Everything is tiny on my 2K/4K screen" — use the font scale, not a smaller
resolution.** Cells are `8*scale` x `16*scale` px, so the scale is what actually
changes text size; a smaller *desktop* only letterboxes (smaller usable area, black
borders, same text size). The automatic scale comes from the desktop height —
1080p and below → 1, 1200p-1600p → 2, 4K → 3 — and is settable:

```sh
/display scale 3       # bigger text, applies now, persisted
/display scale auto    # back to deriving it from the height
```

(A 2560x1440 panel used to land on scale 1 because the old formula needed 1650px
for scale 2 — 320 columns of 8px text. That was the real reason a 2K display
looked broken; `display::auto_font_scale` fixes it and pins the thresholds with
tests.)

**Status bar position.** The bar can sit on any edge; it applies instantly and
persists to `ui.json` (`status_pos`), so it is also editable by hand via `/open`, and
the settings agent can move it with its `statusbar` tool.

```sh
/statusbar               # which edge it is on now
/statusbar top           # or bottom (default) | left | right
```

`left`/`right` make it a **column**: a fixed 16-cell-wide sidebar whose fields stack
as rows instead of running across, with the brand at the top and the system info at
the far end. That costs screen *width*, usually the scarcer direction, so a
horizontal bar (one text row) leaves more room for the panes. Everything else — the
shell pane, the action grid, every divider drag — lays out inside whatever the bar
leaves over, so no surface ever sits under it.

**Running with virtio-gpu (real mode setting).** `CHITTI_GPU` picks the display
device:

```sh
CHITTI_GPU=virtio cargo xtask run -arch aarch64 --uefi   # ramfb + virtio-gpu-pci
CHITTI_GPU=virtio cargo xtask run -arch x86_64
CHITTI_GPU=vmware cargo xtask run -arch x86_64           # VMSVGA (x86 only; see below)
```

Two things that are not obvious:

- **aarch64 needs `--uefi`.** PCI comes from the stub's ACPI, so on the plain
  `-kernel` path virtio-gpu is invisible; the knob says so and falls back to ramfb.
- **VMSVGA only drives an I/O-BAR device.** VirtualBox-ARM exposes BAR0 as memory,
  and that register layout is unverified — driving it on a guess put a real VBox
  display into the wrong geometry, so it now declines and keeps the firmware
  framebuffer (`VMSVGA_ALLOW_MMIO` in `kms/vmsvga.rs`). On VirtualBox, resolution
  therefore still comes from the firmware: `make vbox VBOX_RES=1920x1080`.
- **ramfb is kept alongside virtio-gpu.** Booting aarch64/HVF with virtio-gpu as the
  *only* display puts the firmware's GOP framebuffer inside the device's BAR, and
  writing there after ExitBootServices aborts QEMU with
  `Assertion failed: (isv), function hvf_handle_exception`. Confirmed to be the
  environment and not the driver by bisecting with the KMS probe disabled. With ramfb
  present the console lives in safe memory and virtio-gpu is a second device the
  driver binds — its scanout is DMA RAM, which is writable normally.

**Real mode setting (virtio-gpu).** With a `virtio-gpu-pci` device the OS drives
the display itself, so `/display set` changes the actual mode instead of
letterboxing — `/display status` shows `driver virtio-gpu` and `/display list`
shows the device's modes. Try it:

```sh
qemu-system-x86_64 -M q35 -cpu max -m 4G -cdrom target/chitti.iso \
  -device virtio-gpu-pci,id=gpu0 -serial mon:stdio
# then in the guest:  /display set 1280x720   -> "mode 1280x720 set on virtio-gpu"
```

NB on aarch64 the plain `-kernel` path has no PCI (ECAM comes from the stub's
ACPI), so virtio-gpu binds on the `--uefi` path and on x86. Without a driver the
module is inert and everything behaves as before.

**Changing the resolution from inside the OS.** `/display` (also the settings
agent's `display` tool) has two knobs, because only one of them can work without
a reboot:

```sh
/display                  # panel size, current desktop, next-boot setting
/display list             # desktop sizes this panel can show (native first)
/display set 1920x1080    # applies NOW — centred, letterboxed, still 1:1 (crisp)
/display set native       # back to the whole panel
/display boot 1920x1080   # records the panel's own mode for the next boot
```

`set` is the logical desktop: the compositor lays out against it and blits it as a
viewport inside the physical framebuffer, so text is rasterised at real pixels and
stays sharp — nothing is scaled. It persists to `/configs/core/display.json` and is
re-applied at boot. `boot` is the hardware mode, which only the loader can set;
it is **recorded but not yet applied** (see the note it prints), so today use
`VBOX_RES=` / `CHITTI_RESOLUTION=` for that.

**Multi-monitor hosts (the QEMU window).** `cargo xtask run` sizes the ramfb to the
chosen display's **desktop**, read from `system_profiler SPDisplaysDataType -json`:

| Field | Meaning | Used for |
|---|---|---|
| `_spdisplays_resolution` | the desktop actually in use (`1440 x 900 @ 60Hz`) | the default |
| `_spdisplays_pixels` | the backing store (`2880 x 1800`) | `CHITTI_FB_RES=max` |
| `spdisplays_main` | which display is primary | the default pick |

```sh
cargo xtask run -arch aarch64                     # main display's desktop
CHITTI_FB_RES=max cargo xtask run                 # its full backing store
CHITTI_FB_DISPLAY=DELL cargo xtask run            # a specific monitor (substring, any case)
CHITTI_FB_RES=1920x1080 cargo xtask run           # pin it outright
```

Use the **JSON**, not the plain-text output: the text form omits the current
resolution for a display at its default scaled mode, which forced the earlier
version to guess — it halved a 2560x1600 panel to 1280x800 on a machine whose
desktop was really 1440x900. Parsing is pure and unit-tested (`cargo test -p xtask`)
against captured real output.

### Real Mac mini (Apple Silicon, via m1n1) — work in progress

ChittiOS boots on a **bare Mac mini M2 (`t8112` / j473)** through the Asahi
[**m1n1**](https://github.com/AsahiLinux/m1n1) bootloader — no hypervisor. This
is an active bring-up; see [`docs/apple-usb-hid.md`](docs/apple-usb-hid.md) and
the plan in `~/.claude/plans/` for the current state. Working today: boot to the
shell UI on the physical display, model-via-initrd, and native **USB keyboard +
mouse on the two USB-C ports**. Not yet: the USB-A ports (they hang off the
Thunderbolt/USB4 subsystem, not the two Type-C USB controllers) and Type-C
orientation/Vbus (the Apple `cd321x` PD controller).

**One-time setup.** Check out m1n1 and create its proxyclient venv, and grab the
machine device tree:

```sh
git clone https://github.com/AsahiLinux/m1n1 third_party/m1n1
uv venv third_party/m1n1/.venv
uv pip install --python third_party/m1n1/.venv/bin/python -r third_party/m1n1/requirements.txt
make -C third_party/m1n1                          # builds build/m1n1.bin (for the hv path)
# the j473 device tree is checked in at third_party/dtb/t8112-j473.dtb
```

Put the Mac in **DFU / m1n1 proxy mode** (Asahi's install/`m1n1` step) and
connect the USB-C cable; it appears as two serial devices, e.g.
`/dev/cu.usbmodemXXXXD1` (proxy control) and `…D3`.

**Dev loop** — `cargo xtask m1n1` (or `make m1n1`) builds the arm64 `Image`,
gzips it, and boots it over the proxy. Configure via env:

| env | purpose |
| --- | --- |
| `CHITTI_M1N1` | path to the m1n1 checkout (has `proxyclient/`) |
| `CHITTI_DTB` | machine device tree (`third_party/dtb/t8112-j473.dtb`) |
| `M1N1DEVICE` | the proxy control TTY (`…D1`) |
| `CHITTI_BOOTARGS` | kernel bootargs (e.g. `chitti.usb` to enable USB, `chitti.epoch=<unix>`) |
| `CHITTI_INITRD` | a GGUF to hand off as the model (`assets/model.gguf`) |
| `CHITTI_M1N1_HV=1` | boot **under a resident m1n1 hypervisor** (keeps serial alive) |
| `CHITTI_SERIAL_LOG=<path>` | tee the forwarded serial console to a logfile |

```sh
# Bare boot to the shell (USB enabled), for interactive use / USB HID testing:
CHITTI_M1N1=third_party/m1n1 CHITTI_DTB=third_party/dtb/t8112-j473.dtb \
M1N1DEVICE=/dev/cu.usbmodemXXXXD1 \
make m1n1 RELEASE=1 CHITTI_BOOTARGS="chitti.usb"
```

**Seeing logs — important.** A **bare** boot has **no serial** after handoff
(m1n1's USB gadget, both the `_01` proxy and `_03` UART bridge, tears down when
it jumps to the payload — the host log dies at `Preparing to run next stage /
Exit TTY mode`). Two ways to get diagnostics:

- **On a bare boot**, driver traces are only visible on the **framebuffer**. The
  ktrace/logs pane is closed at boot, so bring-up code that needs to be seen
  (e.g. `apple_usb` under `chitti.usb`) calls `ktrace::set_console_echo(true)` to
  mirror every trace line to the always-visible **chat** pane. Read the Mac's
  monitor.
- **Under the hypervisor** (`CHITTI_M1N1_HV=1`), m1n1 stays resident and forwards
  the guest UART, so `CHITTI_SERIAL_LOG=target/serial.log` captures a real
  logfile you can `tail -f`. This is the way to iterate on **non-USB** drivers
  (PCIe, storage, AIC, SMP) autonomously — but the hv strips USB from the guest,
  so it can't be used for USB/Bluetooth work.

USB HID is **gated behind the `chitti.usb` bootarg** (and never runs under the
hv): the dwc3/DART MMIO it drives is the same controller m1n1 uses for the proxy
console, so resetting it under the hv kills the console. Plug keyboard + mouse
into the **two USB-C ports** (one device per controller).

## Repository layout

```text
chitti/
├── CLAUDE.md            # guide for agents/humans: the invariants + standing rules
├── README.md            # about, features, license
├── DEVELOPMENT.md        # this file
├── CONTRIBUTING.md, SECURITY.md
├── rust-toolchain.toml   # pinned nightly + rust-src, llvm-tools-preview
├── targets/              # custom bare-metal target JSONs (x86_64-chitti, aarch64-chitti)
├── xtask/                # build orchestration: image assembly + QEMU + tests
├── stub/                 # aarch64 UEFI bootloader (BOOTAA64.EFI): GOP + boot-info handoff
├── agents/               # built-in agents: SOUL + manifest (+ assets/tools.wasm) — doc, ssh, chess,
│                         #   media, pdf, download, todo, notes, paint, slides, minesweeper, snake, synth
├── assets/               # Geist Mono font (+ generator); model + voice assets fetched here (gitignored)
├── tools/                # host harnesses & generators: cortexdiff/h264diff/onnxdiff (native diff vs
│                         #   llama.cpp/PyAV/onnxruntime), *-wasm agent-tool crates (chess/doc/pdf/apps),
│                         #   font atlas + table generators (gen_cabac/iq/mp3), mkext4 reference
├── tests/e2e/            # end-to-end harness: boots the kernel, drives the shell over serial
└── kernel/               # the OS — a standalone crate (own workspace, own Cargo.lock)
    └── src/
        ├── main.rs, lib.rs        # boot entries (x86 + aarch64), init sequence, test harness
        ├── arch/{x86_64,aarch64}/ # per-arch drivers behind the arch facade (arch/mod.rs)
        ├── mm/ sched/ cap/ ipc/ channel/ smp.rs  # microkernel: memory, tasks, caps, IPC, channels, SMP
        ├── cortex/                 # CPU inference: gguf, tensor kernels, model, sampler, batch
        ├── synapse/                # capability ABI: registry, grammar, executor, audit, fs, taint, ui
        ├── agent/ session/ tools/ skills/  # the agent layer, sessions, tools, skills (+ registry_client)
        ├── service/                # long-running native service agents (network, http/doc) + supervisor
        ├── block/ fs/              # block devices (virtio/nvme/ahci) + FAT/ext4
        ├── framebuffer/            # the compositor, one module per surface (see its mod.rs map)
        ├── font_geist.rs editor.rs mouse.rs clock/ ui_config.rs json.rs
        ├── console.rs serial.rs ktrace.rs   # I/O + deterministic logging
        └── shell/                  # the `/`-command + chat shell
```

## Notes / gotchas

- **`kernel/` is not part of the host workspace.** It has its own (empty)
  `[workspace]`, its own `Cargo.lock`, and its own `.cargo/config.toml` (custom
  target, `-Z build-std`, the QEMU test runner) so it never gets unified with the
  host `xtask` binary.
- **Don't add `panic = "abort"` profile overrides** on top of the target spec's
  own `panic-strategy: abort` — it makes Cargo build two non-unified copies of
  `core`/`alloc` and any ordinary dependency then fails with "duplicate lang
  item". The target spec is already authoritative.
- **The framebuffer + font atlas are excluded from the `--test` build**
  (`#[cfg(not(test))]`) — the test harness never draws.
- **macOS has no `timeout(1)`** and QMP unix socket paths must be short (< 104
  bytes) — put them under `/tmp` when scripting QEMU.
- Regenerate the Geist Mono glyph atlas after a font change:
  `python3 tools/fonts/gen_geist.py` → `kernel/src/font_geist.rs`.

## Agent app wasm modules

The built-in app agents ship their deterministic logic as
`agents/<name>/assets/tools.wasm` (checked in, `include_bytes!` at boot).
Sources live in stand-alone crates under `tools/`: `chess-wasm`, `doc-wasm`,
`pdf-wasm`, and `apps-wasm` (one shared module for notes / paint / slides /
minesweeper / snake / synth). After editing one:

```sh
cd tools/<name>-wasm
cargo build --release --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/<crate>.wasm ../../agents/<name>/assets/tools.wasm
# apps-wasm: copy the same module to every app package that ships it
```

### Or write the tools in JavaScript, on the machine

A package's `assets/tools.wasm` does not have to come from Rust. `/agents new
<name>` scaffolds a package whose `tools.js` exports one function per tool, and
`/agents build <name>` compiles it to that same `assets/tools.wasm` **on the
running OS** — QuickJS is in the image (`assets/wasm/javy-plugin.wasm`), so no host
toolchain is involved:

```text
/agents new demo                     # SOUL.md + manifest.json + a working tools.js
/open ~/agents/demo/tools.js         # edit
/agents build demo                   # -> assets/tools.wasm  (~90 ms)
/js call ~/agents/demo/assets/tools.wasm demo_sum '{"xs":[1,2,3]}'
```

Scripts reach the kernel's gated host surface through a `Chitti` global —
`Chitti.storageGet/Set`, `fsRead/fsWrite/fsList/fsExists`, `uiDraw`, `hud`, `http`,
`log`, `sha1`, `home` — which are **the same `chitti.host_*` imports a Rust tool
module calls**, gated by the same code. Anything the agent's manifest does not grant
**throws**, so a refusal is never mistaken for an empty result; storage needs an agent
identity, so it refuses under the identity-less `/js call` (use it from an installed
agent). The engine itself is `assets/wasm/javy-plugin.wasm`, rebuilt with
`cargo xtask javy-plugin` from `tools/javy-plugin/` — note that crate must **not** be
stripped, because `javy init-plugin` validates through binaryen, which reads the
`target_features` custom section to decide which wasm features to allow.

The lower-level form is `/js build <in.js> [-o out.wasm] [--tools a,b]`; exports are
scanned from `export function <name>` when `--tools` is omitted. The shape is fixed
by Javy: exported functions take **no arguments** and their **return value is
dropped**, so arguments arrive as JSON on stdin and the result leaves as JSON on
stdout (`readArgs()` / `reply()` in the scaffold). Module top level re-runs on every
call, so **JS globals do not persist** — durable state belongs in storage, which is
why package-UI apps (whose guest statics *are* their state) stay Rust.

Modules are string-ABI (`export(args_ptr, args_len) -> i64 = (ptr<<32)|len`),
no_std, and run under manifest-declared fuel + memory limits — see the Apps
bullet in [CLAUDE.md](CLAUDE.md) for the ABI contract and gotchas. `pdf-wasm`
also runs host-side parser tests: `cargo test` in its folder.

## Sample files (`/samples/…`)

Every interesting path in this OS needs a file — a PNG for the ring-3 decoder, an
mp4 for the H.264 player, a WAV for the sound device, a PDF for the wasm digest —
and a freshly booted machine has none, which makes the media stack awkward to try.
So a small corpus is **embedded in the kernel image** and seeded into the Synapse
store at boot:

```text
/samples/images/   fruits.jpg  baboon.jpg  sudoku.png  transparency.png  grayscale.png
/samples/videos/   sample.mp4                 (H.264 + AAC, 360p)
/samples/audios/   sample.wav  sample.mp3  sample.aac  jfk-speech.wav  sample.ogg*
/samples/misc/     minimal.pdf  document.pdf  rfc1951-deflate.txt  cars.json
                   seattle-weather.csv  first-web-page.html
/samples/js/       hello.js  argv.js  fib.js  math.js  class.js  json.js
/samples/README.md provenance + licence of every file
```

`/open /samples/images/fruits.jpg` works on the first boot, offline. `*` — the Ogg
Vorbis file has **no decoder yet**; it is there as the next decoder's input and
opens in the editor, not the player. The `/samples/js/` scripts are **authored
in-tree** (`assets/samples-src/js/`) and run with `/js /samples/js/hello.js`
(Node-style CLI).

- **It is opt-in, and on by default only for the dev flows.** `make run`,
  `make run-remote`, `make run-uefi`, `make image`, `make vbox` and `make e2e`
  pass `CHITTI_SAMPLE_FILES=1` (the `SAMPLES` knob). A plain
  `cargo xtask build` / CI / the unit suite embeds nothing, so their kernels are
  unchanged.
- **First use downloads ~10 MiB** into the gitignored `assets/samples/`
  (`cargo xtask sample-files`, or `make sample-files`; cached afterwards,
  `--refresh` re-fetches). The build paths fetch it for you. A failed download is
  a **warning, never a failed build** — the OS is fully functional without samples.
- **Fetched, never committed**, the same rule the voice/WiFi assets follow: the
  repository redistributes none of it, and every file's source + licence is
  recorded in the generated `assets/samples/README.md`, which is itself embedded
  as `/samples/README.md`.
- **Seeding never overwrites.** On an installed (ext4-backed) system the store is
  durable, so a sample you edited stays edited across reboots; boot only writes
  paths that are absent.
- The corpus is defined in one place — the `SAMPLE_FILES` table in
  `xtask/src/main.rs` (URL or empty for local, destination, provenance) — and
  `kernel/build.rs` **walks the directory** rather than carrying a second list.
  To add a fetched sample, add a row with an `http(s)` URL and re-run the fetch.
  To add an authored script, put the file under `assets/samples-src/<cat>/`,
  add a row with an empty `url`, and re-run `cargo xtask sample-files`. Set
  `SAMPLES=` for an image without any.

## Voice (`/voice`) — audio + models

- **Microphone permission (macOS).** QEMU's coreaudio *input* only opens once
  the process running QEMU has been granted Microphone access. A headless /
  background launch (e.g. from a CI or an editor's task runner) can't be granted
  and fails at startup with `audio: Can not open 'virtio-sound.in' (no host
  audio driver)`. Run `cargo xtask run` from a **real terminal** and grant the
  first-run prompt, or pre-authorize the terminal in **System Settings →
  Privacy & Security → Microphone**. Playback (the `/voice test` tone) needs no
  permission; capture does. Without a mic you can still exercise the full STT
  front-end with `/voice stt <file.wav>` (mount a volume holding a 16 kHz mono
  WAV).
- **Models.** `cargo xtask voice-assets` downloads silero-vad (embedded in the
  kernel), parakeet STT (~131 MB) and KittenTTS (~78 MB) into `assets/voice/`
  (gitignored). The two large models are loaded at runtime — `/voice models
  load parakeet|kitten <mounted-path>` — not embedded. `/voice models` shows
  what's loaded.
