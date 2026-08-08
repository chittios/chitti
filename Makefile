# ChittiOS — local dev convenience wrapper around `cargo xtask`.
# Everything still works via `cargo xtask <cmd>` directly; this is just shorthand.
# See DEVELOPMENT.md for the full setup.

# --- knobs (override on the command line: `make run ARCH=x86_64 MODEL=qwen3.5-9b RELEASE=1`) ---
# ARCH:         aarch64 (native HVF on Apple Silicon) | x86_64
# MODEL:       qwen3.5-0.8b (default) | bonsai-27b-ternary | qwen3.5-0.8b
#               | qwen3.5-2b | qwen3.5-4b | qwen3.5-9b | gemma-4-e4b (e4b)
#               bonsai-27b         = PrismML Bonsai-27B 1-bit (Q1_0 binary, ~3.8 GB)
#               bonsai-27b-ternary = PrismML Ternary-Bonsai-27B (Q2_0, ~7.17 GB)
# RELEASE:      1 (default) = optimized build — inference is many times faster;
#               set RELEASE= (empty) for a fast-compile debug build
# BRIDGE:       host NIC to L2-bridge (empty = QEMU user-net / slirp). macOS
#               vmnet-bridged needs sudo — leave empty for host services via 10.0.2.2
# REMOTE_URL:   auto `/model remote` at boot (empty = no seed → local model).
#               `make run` / `make vbox` leave this empty (boot the local GGUF);
#               use `make run-remote` / `make vbox-remote` to seed a hosted
#               backend. Under user-net the host is always 10.0.2.2 (not the
#               Mac's LAN IP).
# REMOTE_MODEL: model name sent to the hosted server (LM Studio / Ollama / …)
# REMOTE_KEY:   bearer token for the hosted server (`Authorization: Bearer …`).
#               Empty = none (a LAN LM Studio / Ollama needs no key). Prefer
#               keeping it out of the shell history / this file:
#                 export CHITTI_REMOTE_KEY=sk-…   (picked up below)
#               A hosted provider's URL is the **base** — the kernel appends
#               `/v1/chat/completions` itself, so stop before that: opencode zen
#               serves `https://opencode.ai/zen/v1/chat/completions`, hence pass
#               `https://opencode.ai/zen`. Passing the whole endpoint gets a 404
#               (the doubled path lands on the provider's marketing site).
ARCH         ?= aarch64
MODEL        ?= qwen3.5-0.8b
RELEASE      ?= 1
BRIDGE       ?=
REMOTE_URL   ?=
REMOTE_MODEL ?= ornith-1.0-9b
REMOTE_KEY   ?= $(CHITTI_REMOTE_KEY)
# Hosted backend seeded by `run-remote` / `vbox-remote` (override on the command line).
REMOTE_RUN_URL ?= http://10.0.2.2:1234

# Host USB passthrough into QEMU (Bluetooth / UVC camera). Empty = none.
#   USB_BT=1              auto: grep host USB for a Bluetooth dongle and add it
#   USB_CAM=1             auto: grep host USB for a webcam / UVC device and add it
#   USB_BT=0a12:0001      explicit vendor:product (hex)
#   USB_CAM=046d:082d
#   USB_HOST=vid:pid,...  extra devices (comma-separated)
# List candidates: `make usb-list`. Needs a real stick/camera; QEMU has no
# emulated BT/UVC. macOS may require granting QEMU USB access / unplugging from host.
USB_BT   ?=
USB_CAM  ?=
USB_HOST ?=

# VirtualBox (the `vbox` target): which VM to (re)load the aarch64 image into,
# and where its boot disk is attached. Override e.g. `make vbox VBOX_VM=MyVM`.
# Guest screen resolution, e.g. `make vbox VBOX_RES=1920x1080`. Empty = leave the
# guest at whatever the firmware chooses.
#
# This sets it two ways on purpose, because the obvious one does not work:
# VirtualBox-ARM *stores* VBoxInternal2/EfiGraphicsResolution and then boots the
# guest at its own resolution regardless. So the value is also written to
# `\chitti-display.cfg` on the image's ESP, which the stub reads and applies with
# GOP set_mode before the kernel starts — that path does not depend on the
# hypervisor honouring anything. The stub logs the firmware's whole mode list, so
# when a size cannot be had you can see what was on offer.
#
# NB this is the size of the guest *framebuffer*. VirtualBox draws it 1:1, so a
# framebuffer larger than the VM window leaves part of it off-screen (which is what
# a 2560x1440 guest looks like in a 1440-wide window). VBOX_SCALE is the other
# lever: it scales the whole guest display to fit, changing nothing in the guest.
VBOX_RES  ?=
# VirtualBox window scale factor, e.g. `make vbox VBOX_SCALE=0.5` to fit an
# oversized guest framebuffer into the window. Purely host-side.
VBOX_SCALE ?=
VBOX_VM   ?= Chitti
VBOX_CTL  ?= nvme
VBOX_PORT ?= 0
# VM RAM (MiB). Must hold the model at its 2 GiB load offset + the kernel heap:
# ~2 GiB + model-size + 1 GiB. 8192 fits every build up to the ~3.8 GiB 1-bit
# Bonsai / ~5 GiB 9B; the ~7.2 GiB ternary Bonsai needs VBOX_MEM=12288. Too
# little → "chitti-stub: model alloc failed -- booting without a model".
VBOX_MEM  ?= 8192

# SAMPLES: bundle the `/samples/` corpus (images, videos, audios, misc) into the
# image, so a freshly booted machine can `/open /samples/images/fruits.jpg` with
# no network and no disk. **On by default for `run` / `run-remote` / `run-uefi` /
# `image` / `vbox`.** First use downloads ~10 MiB into the gitignored
# `assets/samples/` (`cargo xtask sample-files`, cached afterwards) and embeds it
# in the kernel; a failed download is a warning, never a failed build. Set
# `SAMPLES=` (or 0/off) for an image without them.
SAMPLES   ?= 1

# SHARE: a host directory to share into the guest over virtio-9p, mounted at
# `/host` — `make run SHARE=~/Desktop/chitti`. Copy files in and out with the
# ordinary `/cp`, `/ls`, `/cat`, `/rm`. Empty (the default) shares nothing.
# Works on QEMU; VirtualBox uses its own shared folders (`/share` in the guest).
SHARE     ?=

# CLIPBOARD: attach the SPICE clipboard agent channel (virtio-serial +
# `qemu-vdagent`), so a copy in the guest and a copy on the host share one
# clipboard. On by default for `make run`. NB QEMU bridges its internal
# clipboard to a real one only through a display backend that registers a
# clipboard peer — gtk and dbus do, **cocoa does not** — so on macOS the link
# is live but ends inside QEMU; `/clip` in the guest says which route is
# actually working. Set `CLIPBOARD=` (or 0/off) to omit the device.
CLIPBOARD ?= 1

XTASK   := cargo xtask
REL     := $(if $(filter 1 true yes,$(RELEASE)),--release,)
FLAGS   := -arch $(ARCH) -model $(MODEL) $(REL)

.DEFAULT_GOAL := help

## help: list targets
.PHONY: help
help:
	@echo "ChittiOS — make targets (ARCH=$(ARCH) MODEL=$(MODEL) RELEASE=$(RELEASE))"
	@echo "  BRIDGE=$(BRIDGE)  REMOTE_URL=$(REMOTE_URL)  REMOTE_MODEL=$(REMOTE_MODEL)"
	@echo "  REMOTE_KEY=$(if $(REMOTE_KEY),<set>,)  (never printed; from REMOTE_KEY= or \$$CHITTI_REMOTE_KEY)"
	@echo "  SAMPLES=$(SAMPLES) (bundle /samples files)"
	@echo
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/^## /  /'
	@echo
	@echo "Override knobs, e.g.:"
	@echo "  make model && make run                 # fetch + boot the default (bonsai-27b 1-bit)"
	@echo "  make run ARCH=x86_64 MODEL=qwen3.5-9b RELEASE=1"
	@echo "  make model MODEL=bonsai-27b-ternary && make run MODEL=bonsai-27b-ternary  # Q2_0 build"
	@echo "  make run-remote REMOTE_RUN_URL=http://10.0.2.2:1234 REMOTE_MODEL=ornith-1.0-9b"
	@echo "  make run-remote REMOTE_RUN_URL=https://opencode.ai/zen REMOTE_MODEL=deepseek-v4-flash REMOTE_KEY=sk-…"
	@echo "  make vbox MODEL=lfm2.5-2.6b   # VM image on the local GGUF (no remote seed)"
	@echo "  make vbox-remote REMOTE_RUN_URL=https://opencode.ai/zen REMOTE_MODEL=deepseek-v4-flash REMOTE_KEY=sk-…  # same seed, UEFI/VM image"
	@echo "  make run BRIDGE=en0           # L2 bridge (often needs sudo on macOS)"
	@echo "  make usb-list                 # grep host USB for BT / camera candidates"
	@echo "  make run USB_BT=1 USB_CAM=1   # passthrough grepped BT dongle + webcam"
	@echo "  make run USB_BT=0a12:0001     # or pass explicit vid:pid"

## test: in-kernel test suite under QEMU (x86_64) — the gate, keep it 104/104
.PHONY: test
test:
	$(XTASK) test

## build: cross-build the kernel for ARCH
.PHONY: build
build:
	$(XTASK) build $(FLAGS)

## build-all: build BOTH arches (the dual-arch parity check)
.PHONY: build-all
build-all:
	$(XTASK) build -arch x86_64 -model $(MODEL) $(REL)
	$(XTASK) build -arch aarch64 -model $(MODEL) $(REL)

## usb-list: grep the host USB tree for Bluetooth / camera devices (vid:pid)
##           macOS: system_profiler | grep; Linux: lsusb | grep
.PHONY: usb-list
usb-list:
	@echo "=== host USB (Bluetooth / camera candidates) ==="
	@if [ "$$(uname -s)" = "Darwin" ]; then \
	  system_profiler SPUSBDataType 2>/dev/null \
	    | grep -iE -B2 -A6 'bluetooth|camera|webcam|uvc|imaging|facetime|csr8510|hd web' \
	    || echo "(no Bluetooth/camera lines — plug a dongle/webcam, or USB_HOST=vid:pid)"; \
	else \
	  lsusb 2>/dev/null | grep -iE 'bluetooth|camera|webcam|uvc|imaging' \
	    || echo "(no matches in lsusb)"; \
	fi
	@echo
	@echo "=== auto vid:pid (same rules as make run USB_BT=1 / USB_CAM=1) ==="
	@CHITTI_USB_BT=1 CHITTI_USB_CAM=1 $(XTASK) usb-ids
	@echo
	@echo "Attach:  make run USB_BT=1 USB_CAM=1"
	@echo "Or:      make run USB_BT=0a12:0001 USB_CAM=046d:082d"
	@echo "Guest:   /bluetooth status   /camera status   /camera grab"

## run: boot the kernel in QEMU for ARCH (serial on stdio + a graphical window)
##      uses the local bundled GGUF (MODEL); no remote seed (see run-remote)
##      USB_BT / USB_CAM / USB_HOST → QEMU usb-host passthrough (see usb-list)
.PHONY: run
run:
	@if [ -n "$(USB_BT)$(USB_CAM)$(USB_HOST)" ]; then \
	  echo "run: host USB passthrough (BT='$(USB_BT)' CAM='$(USB_CAM)' HOST='$(USB_HOST)')"; \
	  echo "run: grepping host USB tree…"; \
	  if [ "$$(uname -s)" = "Darwin" ]; then \
	    system_profiler SPUSBDataType 2>/dev/null \
	      | grep -iE -B1 -A4 'bluetooth|camera|webcam|uvc|imaging|facetime|csr8510' \
	      | head -40 || true; \
	  else \
	    lsusb 2>/dev/null | grep -iE 'bluetooth|camera|webcam|uvc|imaging' || true; \
	  fi; \
	fi
	CHITTI_NET_BRIDGE='$(BRIDGE)' \
	CHITTI_REMOTE_URL='$(REMOTE_URL)' \
	CHITTI_REMOTE_MODEL='$(REMOTE_MODEL)' \
	CHITTI_REMOTE_KEY='$(REMOTE_KEY)' \
	CHITTI_USB_BT='$(USB_BT)' \
	CHITTI_USB_CAM='$(USB_CAM)' \
	CHITTI_USB_HOST='$(USB_HOST)' \
	CHITTI_SAMPLE_FILES='$(SAMPLES)' \
	CHITTI_SHARE='$(SHARE)' \
	CHITTI_CLIPBOARD='$(CLIPBOARD)' \
	$(XTASK) run $(FLAGS)

## run-remote: like `run`, but seed `/model remote` at boot from REMOTE_RUN_URL
##             + REMOTE_MODEL (hosted LM Studio / Ollama / vLLM). Override e.g.
##             `make run-remote REMOTE_RUN_URL=http://10.0.2.2:1234 REMOTE_MODEL=ornith-1.0-9b`
##             A hosted provider needing a bearer token takes REMOTE_KEY (or
##             `export CHITTI_REMOTE_KEY=…`), and the URL is the base — the
##             kernel appends `/v1/chat/completions`, so pass `…/zen`, not the
##             endpoint:
##             `make run-remote REMOTE_RUN_URL=https://opencode.ai/zen REMOTE_MODEL=deepseek-v4-flash REMOTE_KEY=sk-…`
.PHONY: run-remote
run-remote:
	CHITTI_NET_BRIDGE='$(BRIDGE)' \
	CHITTI_REMOTE_URL='$(REMOTE_RUN_URL)' \
	CHITTI_REMOTE_MODEL='$(REMOTE_MODEL)' \
	CHITTI_REMOTE_KEY='$(REMOTE_KEY)' \
	CHITTI_USB_BT='$(USB_BT)' \
	CHITTI_USB_CAM='$(USB_CAM)' \
	CHITTI_USB_HOST='$(USB_HOST)' \
	CHITTI_SAMPLE_FILES='$(SAMPLES)' \
	CHITTI_SHARE='$(SHARE)' \
	CHITTI_CLIPBOARD='$(CLIPBOARD)' \
	$(XTASK) run $(FLAGS)

## model: fetch the GGUF for MODEL into assets/ (required before run / run-uefi)
.PHONY: model
model:
	./xtask/fetch-model.sh $(MODEL)

## voice-assets: download ONNX voice models into assets/voice/
.PHONY: voice-assets
voice-assets:
	$(XTASK) voice-assets

## sample-files: download the /samples corpus (images/videos/audios/misc) into
##               assets/samples/ (~10 MiB, cached). `make run` / `make vbox` do
##               this for you via SAMPLES=1; add --refresh to re-fetch.
.PHONY: sample-files
sample-files:
	$(XTASK) sample-files

## wifi-assets: extract Apple FullMAC dongle firmware (miyake 4388) into
##              assets/wifi/brcm/ from this Mac's /usr/share/firmware/wifi.
##              Required for `/wifi load` on bare m1n1 (embedded) and for
##              ESP-bundled disk boots. Re-run after deleting assets/wifi/ to refresh.
.PHONY: wifi-assets
wifi-assets:
	$(XTASK) wifi-assets

## run-uefi: boot aarch64 via the UEFI stub under AAVMF (not -kernel)
##          needs assets GGUF for MODEL — `make model MODEL=e4b` first
.PHONY: run-uefi
run-uefi:
	CHITTI_SAMPLE_FILES='$(SAMPLES)' $(XTASK) run -arch aarch64 -model $(MODEL) $(REL) --uefi

## image: assemble a bootable image/ISO for ARCH
.PHONY: image
image:
	CHITTI_SAMPLE_FILES='$(SAMPLES)' $(XTASK) image $(FLAGS)

## m1n1: package the aarch64 kernel as a gzip'd arm64 Image and boot it on a
##       tethered Apple Silicon Mac over the m1n1 USB proxy. Configure via env:
##       CHITTI_M1N1 (m1n1 checkout), CHITTI_DTB (machine dtb), optional
##       CHITTI_INITRD (model gguf), CHITTI_BOOTARGS, M1N1DEVICE (proxy tty).
##       Use RELEASE=1 for hardware. Without CHITTI_M1N1/CHITTI_DTB it just
##       builds the Image and prints the manual linux.py command.
##       Best-effort: extract wifi-assets first so /wifi load embeds the dongle
##       image (no-op if already present; skips with a warning if macOS fw is
##       missing so CI/non-Apple hosts still build the Image).
.PHONY: m1n1
m1n1:
	-$(XTASK) wifi-assets
	$(XTASK) m1n1 $(REL)

## vbox: rebuild the aarch64 image and (re)load it into VirtualBox VM VBOX_VM
##       uses the local bundled GGUF (MODEL); no remote seed (see vbox-remote)
##       forces USB keyboard + USB tablet + xHCI (aarch64 has no PS/2 input path)
##       NB: do not put Make `#` comments inside the shell recipe — they break
##       `\` line continuation and re-run later lines in a fresh shell with VM
##       empty.
## vbox-remote: like `vbox`, but seed `/model remote` at boot from
##       REMOTE_RUN_URL/REMOTE_MODEL/REMOTE_KEY (embedded on the ESP as
##       \chitti-model.json — the stub hands it to the kernel via the boot-info
##       page, the UEFI-boot analogue of run-remote's fw_cfg seed):
##       `make vbox-remote REMOTE_RUN_URL=https://opencode.ai/zen REMOTE_MODEL=deepseek-v4-flash REMOTE_KEY=sk-…`
.PHONY: vbox vbox-remote
vbox:        VBOX_SEED_URL := $(REMOTE_URL)
vbox-remote: VBOX_SEED_URL := $(REMOTE_RUN_URL)
vbox vbox-remote:
	CHITTI_REMOTE_URL='$(VBOX_SEED_URL)' \
	CHITTI_REMOTE_MODEL='$(REMOTE_MODEL)' \
	CHITTI_REMOTE_KEY='$(REMOTE_KEY)' \
	CHITTI_RESOLUTION='$(VBOX_RES)' CHITTI_SAMPLE_FILES='$(SAMPLES)' $(XTASK) image -arch aarch64 -model $(MODEL)
	@command -v VBoxManage >/dev/null || { echo "VBoxManage not found — install VirtualBox"; exit 1; }
	@set -e; \
	VM='$(VBOX_VM)'; CTL='$(VBOX_CTL)'; PORT='$(VBOX_PORT)'; \
	IMG=target/chitti-aa64.img; VDI=target/chitti-aa64.vdi; \
	test -f "$$IMG" || { echo "vbox: missing $$IMG — image step failed?"; exit 1; }; \
	VBoxManage showvminfo "$$VM" >/dev/null 2>&1 || { \
	  echo "vbox: VM '$$VM' not found — create an ARM64 EFI VM named $$VM first"; exit 1; }; \
	UUID=$$(VBoxManage showvminfo "$$VM" --machinereadable 2>/dev/null | sed -n "s/^\"$$CTL-ImageUUID-$$PORT-0\"=//p" | tr -d '"'); \
	echo "vbox: reloading VM '$$VM' (ctl=$$CTL port=$$PORT), preserving disk UUID $${UUID:-<new>}"; \
	VBoxManage controlvm "$$VM" poweroff 2>/dev/null || true; sleep 1; \
	echo "vbox: ensuring USB keyboard + USB tablet + xHCI controlle + ACPI; RAM $(VBOX_MEM) MiB"; \
	VBoxManage modifyvm "$$VM" --keyboard usb --acpi on --mouse usbtablet --usb-xhci on --memory $(VBOX_MEM); \
	if [ -n '$(VBOX_RES)' ]; then \
	  echo "vbox: resolution -> $(VBOX_RES) (via the ESP display pref; EFI knob set too, but VBox-ARM ignores it)"; \
	  VBoxManage setextradata "$$VM" VBoxInternal2/EfiGraphicsResolution '$(VBOX_RES)'; \
	else \
	  CUR=$$(VBoxManage getextradata "$$VM" VBoxInternal2/EfiGraphicsResolution 2>/dev/null | sed -n 's/^Value: //p'); \
	  echo "vbox: resolution = firmware default (EFI knob = $${CUR:-unset}; set with: make vbox VBOX_RES=1920x1080)"; \
	fi; \
	if [ -n '$(VBOX_SCALE)' ]; then \
	  echo "vbox: window scale factor -> $(VBOX_SCALE) (host-side; scales the guest display to fit the window)"; \
	  VBoxManage setextradata "$$VM" GUI/ScaleFactor '$(VBOX_SCALE)'; \
	fi; \
	VBoxManage storageattach "$$VM" --storagectl "$$CTL" --port "$$PORT" --device 0 --medium none 2>/dev/null || true; \
	if [ -n "$$UUID" ]; then VBoxManage closemedium disk "$$UUID" 2>/dev/null || true; fi; \
	VBoxManage closemedium disk "$$VDI" 2>/dev/null || true; rm -f "$$VDI"; \
	VBoxManage convertfromraw "$$IMG" "$$VDI" --format VDI; \
	if [ -n "$$UUID" ]; then VBoxManage internalcommands sethduuid "$$VDI" "$$UUID"; fi; \
	VBoxManage storageattach "$$VM" --storagectl "$$CTL" --port "$$PORT" --device 0 --type hdd --medium "$$(pwd)/$$VDI"; \
	VBoxManage showvminfo "$$VM" | grep -iE 'Pointing Device|Keyboard Device|xHCI USB|OHCI USB|EHCI USB|ACPI' || true; \
	echo "vbox: done — start '$$VM'"; \
	echo "vbox: tip: click the VM window, then Host+C (often left ⌘) to capture keyboard"; \
	echo "vbox: boot line should show  usb-kbd=READY  usb-mse=READY"

## ref-check: boot the real model and verify inference parity/determinism
.PHONY: ref-check
ref-check:
	$(XTASK) ref-check

## fmt: format all crates (kernel, xtask, stub)
.PHONY: fmt
fmt:
	cargo fmt --manifest-path kernel/Cargo.toml
	cargo fmt --manifest-path xtask/Cargo.toml
	cargo fmt --manifest-path stub/Cargo.toml

## verify: the standing-rule gate — x86 build + tests + aarch64 build
.PHONY: verify
verify:
	$(XTASK) build -arch x86_64 -model $(MODEL)
	$(XTASK) test
	$(XTASK) build -arch aarch64 -model $(MODEL)
	@echo "verify: both arches build and the test suite passed"

## clean: remove build artifacts for all crates
.PHONY: clean
clean:
	cargo clean --manifest-path kernel/Cargo.toml
	cargo clean --manifest-path xtask/Cargo.toml
	cargo clean --manifest-path stub/Cargo.toml

# End-to-end tests: boot the kernel under QEMU and exercise the networked flows
# (http/https, ws/wss, hosted-model chat) against local host servers. Uses a
# TLS-1.3-capable python for the https/wss scenarios (Homebrew's, if present).
E2E_PY ?= $(shell [ -x /opt/homebrew/bin/python3 ] && echo /opt/homebrew/bin/python3 || echo python3)
# The `samples` scenario needs the corpus embedded in the booted kernel, so the
# suite builds with it exactly as `make run` does (SAMPLES= skips that scenario,
# it does not fail it). `E2E_JOBS=N` splits the run across N concurrent guest
# boots (see `tests/e2e/run.py --help`): a Mac with ~8 cores can run
# `make e2e E2E_JOBS=3` and cut the sweep from ~30 min to ~12 min.
E2E_JOBS ?= 1
e2e:
	CHITTI_SAMPLE_FILES='$(SAMPLES)' $(E2E_PY) tests/e2e/run.py -arch $(ARCH) -model $(MODEL) --jobs $(E2E_JOBS)
# Full e2e incl. local inference + voice (slow; needs assets/model.gguf + assets/voice/).
e2e-full:
	CHITTI_SAMPLE_FILES='$(SAMPLES)' $(E2E_PY) tests/e2e/run.py -arch $(ARCH) -model $(MODEL) --slow --jobs $(E2E_JOBS)
