# Chitti OS — local dev convenience wrapper around `cargo xtask`.
# Everything still works via `cargo xtask <cmd>` directly; this is just shorthand.
# See DEVELOPMENT.md for the full setup.

# --- knobs (override on the command line: `make run ARCH=x86_64 MODEL=qwen3.5-9b RELEASE=1`) ---
# ARCH:    aarch64 (native HVF on Apple Silicon) | x86_64
# MODEL:   qwen3.5-0.8b (default) | qwen3.5-2b | qwen3.5-4b | qwen3.5-9b
# RELEASE: set to 1 for an optimized build
ARCH    ?= aarch64
MODEL   ?= qwen3.5-0.8b
RELEASE ?=

# VirtualBox (the `vbox` target): which VM to (re)load the aarch64 image into,
# and where its boot disk is attached. Override e.g. `make vbox VBOX_VM=MyVM`.
VBOX_VM   ?= Chitti
VBOX_CTL  ?= nvme
VBOX_PORT ?= 0

XTASK   := cargo xtask
REL     := $(if $(filter 1 true yes,$(RELEASE)),--release,)
FLAGS   := -arch $(ARCH) -model $(MODEL) $(REL)

.DEFAULT_GOAL := help

## help: list targets
.PHONY: help
help:
	@echo "Chitti OS — make targets (ARCH=$(ARCH) MODEL=$(MODEL) RELEASE=$(RELEASE))"
	@echo
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/^## /  /'
	@echo
	@echo "Override knobs, e.g.:  make run ARCH=x86_64 MODEL=qwen3.5-9b RELEASE=1"

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
.PHONY: run
run:
	$(XTASK) run $(FLAGS)

## run-uefi: boot aarch64 via the UEFI stub under AAVMF (not -kernel)
.PHONY: run-uefi
run-uefi:
	$(XTASK) run -arch aarch64 -model $(MODEL) $(REL) --uefi

## image: assemble a bootable image/ISO for ARCH
.PHONY: image
image:
	$(XTASK) image $(FLAGS)

## vbox: rebuild the aarch64 image and (re)load it into VirtualBox VM VBOX_VM
.PHONY: vbox
vbox:
	$(XTASK) image -arch aarch64 -model $(MODEL)
	@command -v VBoxManage >/dev/null || { echo "VBoxManage not found — install VirtualBox"; exit 1; }
	@set -e; \
	VM='$(VBOX_VM)'; CTL='$(VBOX_CTL)'; PORT='$(VBOX_PORT)'; \
	IMG=target/chitti-aa64.img; VDI=target/chitti-aa64.vdi; \
	UUID=$$(VBoxManage showvminfo "$$VM" --machinereadable 2>/dev/null | sed -n "s/^\"$$CTL-ImageUUID-$$PORT-0\"=//p" | tr -d '"'); \
	echo "vbox: reloading VM '$$VM' (ctl=$$CTL port=$$PORT), preserving disk UUID $${UUID:-<new>}"; \
	VBoxManage controlvm "$$VM" poweroff 2>/dev/null || true; sleep 1; \
	VBoxManage closemedium disk "$$VDI" 2>/dev/null || true; rm -f "$$VDI"; \
	VBoxManage convertfromraw "$$IMG" "$$VDI" --format VDI; \
	if [ -n "$$UUID" ]; then VBoxManage internalcommands sethduuid "$$VDI" "$$UUID"; fi; \
	VBoxManage storageattach "$$VM" --storagectl "$$CTL" --port "$$PORT" --device 0 --type hdd --medium "$$(pwd)/$$VDI"; \
	VBoxManage showvminfo "$$VM" --machinereadable | grep -iE "hidpointing|hidkeyboard" || true; \
	echo "vbox: done — start '$$VM' (pointing device = USB Tablet, keyboard = USB)"

## ref-check: boot the real model and verify inference parity/determinism
.PHONY: ref-check
ref-check:
	$(XTASK) ref-check

## model: fetch the Qwen3.5-0.8B GGUF into assets/ (not committed)
.PHONY: model
model:
	xtask/fetch-model.sh

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
