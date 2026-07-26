# ChittiOS — local dev convenience wrapper around `cargo xtask`.
# Everything still works via `cargo xtask <cmd>` directly; this is just shorthand.
# See DEVELOPMENT.md for the full setup.

# --- knobs (override on the command line: `make run ARCH=x86_64 MODEL=qwen3.5-9b RELEASE=1`) ---
# ARCH:         aarch64 (native HVF on Apple Silicon) | x86_64
# MODEL:        bonsai-27b (default) | bonsai-27b-ternary | qwen3.5-0.8b
#               | qwen3.5-2b | qwen3.5-4b | qwen3.5-9b | gemma-4-e4b (e4b)
#               bonsai-27b         = PrismML Bonsai-27B 1-bit (Q1_0 binary, ~3.8 GB)
#               bonsai-27b-ternary = PrismML Ternary-Bonsai-27B (Q2_0, ~7.17 GB)
# RELEASE:      1 (default) = optimized build — inference is many times faster;
#               set RELEASE= (empty) for a fast-compile debug build
# BRIDGE:       host NIC to L2-bridge (empty = QEMU user-net / slirp). macOS
#               vmnet-bridged needs sudo — leave empty for host services via 10.0.2.2
# REMOTE_URL:   auto `/model remote` at boot (empty = no seed → local model).
#               `make run` leaves this empty (boots the local GGUF); use
#               `make run-remote` to seed a hosted backend. Under user-net the
#               host is always 10.0.2.2 (not the Mac's LAN IP).
# REMOTE_MODEL: model name sent to the hosted server (LM Studio / Ollama / …)
ARCH         ?= aarch64
MODEL        ?= bonsai-27b
RELEASE      ?= 1
BRIDGE       ?=
REMOTE_URL   ?=
REMOTE_MODEL ?= ornith-1.0-9b
# Hosted backend seeded by `run-remote` (override on the command line).
REMOTE_RUN_URL ?= http://10.0.2.2:1234

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

XTASK   := cargo xtask
REL     := $(if $(filter 1 true yes,$(RELEASE)),--release,)
FLAGS   := -arch $(ARCH) -model $(MODEL) $(REL)

.DEFAULT_GOAL := help

## help: list targets
.PHONY: help
help:
	@echo "ChittiOS — make targets (ARCH=$(ARCH) MODEL=$(MODEL) RELEASE=$(RELEASE))"
	@echo "  BRIDGE=$(BRIDGE)  REMOTE_URL=$(REMOTE_URL)  REMOTE_MODEL=$(REMOTE_MODEL)"
	@echo
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/^## /  /'
	@echo
	@echo "Override knobs, e.g.:"
	@echo "  make model && make run                 # fetch + boot the default (bonsai-27b 1-bit)"
	@echo "  make run ARCH=x86_64 MODEL=qwen3.5-9b RELEASE=1"
	@echo "  make model MODEL=bonsai-27b-ternary && make run MODEL=bonsai-27b-ternary  # Q2_0 build"
	@echo "  make run-remote REMOTE_RUN_URL=http://10.0.2.2:1234 REMOTE_MODEL=ornith-1.0-9b"
	@echo "  make run BRIDGE=en0           # L2 bridge (often needs sudo on macOS)"

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

## run: boot the kernel in QEMU for ARCH (serial on stdio + a graphical window)
##      uses the local bundled GGUF (MODEL); no remote seed (see run-remote)
.PHONY: run
run:
	CHITTI_NET_BRIDGE='$(BRIDGE)' \
	CHITTI_REMOTE_URL='$(REMOTE_URL)' \
	CHITTI_REMOTE_MODEL='$(REMOTE_MODEL)' \
	$(XTASK) run $(FLAGS)

## run-remote: like `run`, but seed `/model remote` at boot from REMOTE_RUN_URL
##             + REMOTE_MODEL (hosted LM Studio / Ollama / vLLM). Override e.g.
##             `make run-remote REMOTE_RUN_URL=http://10.0.2.2:1234 REMOTE_MODEL=ornith-1.0-9b`
.PHONY: run-remote
run-remote:
	CHITTI_NET_BRIDGE='$(BRIDGE)' \
	CHITTI_REMOTE_URL='$(REMOTE_RUN_URL)' \
	CHITTI_REMOTE_MODEL='$(REMOTE_MODEL)' \
	$(XTASK) run $(FLAGS)

## model: fetch the GGUF for MODEL into assets/ (required before run / run-uefi)
.PHONY: model
model:
	./xtask/fetch-model.sh $(MODEL)

## voice-assets: download ONNX voice models into assets/voice/
.PHONY: voice-assets
voice-assets:
	$(XTASK) voice-assets

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
	$(XTASK) run -arch aarch64 -model $(MODEL) $(REL) --uefi

## image: assemble a bootable image/ISO for ARCH
.PHONY: image
image:
	$(XTASK) image $(FLAGS)

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
##       forces USB keyboard + USB tablet + xHCI (aarch64 has no PS/2 input path)
##       NB: do not put Make `#` comments inside the shell recipe — they break `\`
##       line continuation and re-run later lines in a fresh shell with VM empty.
.PHONY: vbox
vbox:
	CHITTI_RESOLUTION='$(VBOX_RES)' $(XTASK) image -arch aarch64 -model $(MODEL)
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
e2e:
	$(E2E_PY) tests/e2e/run.py -arch $(ARCH) -model $(MODEL)
# Full e2e incl. local inference + voice (slow; needs assets/model.gguf + assets/voice/).
e2e-full:
	$(E2E_PY) tests/e2e/run.py -arch $(ARCH) -model $(MODEL) --slow
