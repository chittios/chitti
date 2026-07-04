# Development guide

How to build, run, and test Chitti OS locally on both architectures. Everything
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
| `make test` | `cargo xtask test` |
| `make verify` | x86 build + `test` + aarch64 build (the standing-rule gate) |
| `make vbox` | rebuild the aarch64 image and reload it into a VirtualBox VM |
| `make model` / `make fmt` / `make clean` | fetch the GGUF / format / clean |

The underlying `cargo xtask` commands:

| Command | What it does |
|---|---|
| `cargo xtask build -arch <arch> [--release] [-model <m>]` | Cross-compile the kernel. |
| `cargo xtask run   -arch <arch> [--release] [-model <m>]` | Build the image and boot it in QEMU (serial to stdio + a graphical window). |
| `cargo xtask image -arch <arch> [-model <m>]` | Assemble a bootable image (x86: hybrid BIOS/UEFI ISO; aarch64: a GPT disk image that boots standalone via UEFI). |
| `cargo xtask test` | Run the in-kernel `custom_test_frameworks` suite under `qemu-system-x86_64`, headless, asserting via serial + `isa-debug-exit`. **Must stay 103/103.** |

`-model qwen3.5-0.8b` (default) or `-model qwen3.5-9b` selects the bundled model
and the memory layout. `-arch aarch64 --uefi` boots via the `stub/` UEFI
bootloader under AAVMF instead of `-kernel`.

## The model

The model is **not committed** (it's hundreds of MB, loaded as a boot module /
placed in guest RAM). Fetch it once:

```sh
xtask/fetch-model.sh            # writes assets/…  (Qwen3.5-0.8B GGUF, ~812 MB)
```

`cargo xtask test` never loads the model (the fast unit suite validates tensor
kernels against baked-in NumPy reference vectors). `cargo xtask run` uses it for
the chat/inference demo; numerics are validated against `tools/ref_qwen35.py`
(reconstructed from llama.cpp's `qwen35` graph).

## The standing rules (verify every change)

1. **Dual-arch parity.** A change must build and work on **both** arches.
   After any change run **all** of:
   ```sh
   cargo xtask build -arch x86_64 && cargo xtask test      # 103/103
   cargo xtask build -arch aarch64
   cargo xtask run   -arch aarch64                         # if the change is boot-visible
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
seeds, temperature 0). This is the gate — keep it green.

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
├── assets/               # Geist Mono font (+ generator); model fetched here, not committed
├── tools/                # NumPy inference reference, font atlas generator, mkext4 reference
└── kernel/               # the OS — a standalone crate (own workspace, own Cargo.lock)
    └── src/
        ├── main.rs, lib.rs        # boot entries (x86 + aarch64), init sequence, test harness
        ├── arch/{x86_64,aarch64}/ # per-arch drivers behind the arch facade (arch/mod.rs)
        ├── mm/ sched/ cap/ ipc/ smp.rs   # microkernel: memory, tasks, capabilities, IPC, SMP
        ├── cortex/                 # CPU inference: gguf, tensor kernels, model, sampler, batch
        ├── synapse/                # capability ABI: registry, grammar, executor, audit, fs, taint
        ├── agent/ session/ tools/ skills/  # the agent layer, sessions, tools, skills
        ├── block/ fs/              # block devices (virtio/nvme/ahci) + FAT/ext4/SimpleFS
        ├── framebuffer.rs font_geist.rs editor.rs mouse.rs clock.rs ui_config.rs json.rs
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
