# Chitti OS — local dev convenience wrapper around `cargo xtask`.
# Everything still works via `cargo xtask <cmd>` directly; this is just shorthand.
# See DEVELOPMENT.md for the full setup.

# --- knobs (override on the command line: `make run ARCH=x86_64 MODEL=qwen3.5-9b RELEASE=1`) ---
# ARCH:    aarch64 (native HVF on Apple Silicon) | x86_64
# MODEL:   qwen3.5-0.8b | qwen3.5-9b
# RELEASE: set to 1 for an optimized build
ARCH    ?= aarch64
MODEL   ?= qwen3.5-0.8b
RELEASE ?=

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

## test: in-kernel test suite under QEMU (x86_64) — the gate, keep it 103/103
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
