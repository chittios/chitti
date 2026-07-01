# Development guide

## Prerequisites

- **Rust nightly** with the `rust-src` and `llvm-tools-preview` components.
  Pinned in [`rust-toolchain.toml`](rust-toolchain.toml); `rustup` will pick
  it up automatically (installs it on first use if missing):
  ```sh
  rustup toolchain install nightly --component rust-src --component llvm-tools-preview
  ```
- **QEMU** (`qemu-system-x86_64`).
- **Limine** and **xorriso**, for assembling the bootable ISO. On macOS:
  ```sh
  brew install limine xorriso
  ```
  `xtask` locates Limine's boot files via `brew --prefix limine`. If you
  installed Limine some other way, point `xtask` at it directly:
  ```sh
  export CHITTI_LIMINE_SHARE=/path/to/limine/share/limine   # limine-bios.sys, BOOTX64.EFI, ...
  export CHITTI_LIMINE_BIN=/path/to/limine/bin/limine        # the `limine` deploy tool
  ```

No other setup is required — `cargo xtask <cmd>` does everything else
(building the kernel, assembling the image, launching QEMU).

## Commands

All project commands go through `cargo xtask` (aliased in
[`.cargo/config.toml`](.cargo/config.toml) to `cargo run --package xtask --`):

| Command | What it does |
|---|---|
| `cargo xtask build [--release]` | Cross-compiles the kernel binary for `x86_64-chitti`. |
| `cargo xtask image [--release]` | Builds the kernel, then assembles `target/chitti.iso` (a hybrid BIOS/UEFI ISO) via `xorriso` + `limine bios-install`. |
| `cargo xtask run [--release]` | Builds the image and boots it in QEMU interactively: serial to stdio, kernel text also drawn to the framebuffer window. Runs until you close QEMU or hit Ctrl-C. |
| `cargo xtask test` | Runs `cargo test --lib` inside `kernel/`, which cross-compiles the `custom_test_frameworks` test binary and boots *each* one in QEMU headlessly via the hidden `runner` subcommand, translating `isa-debug-exit` into a real pass/fail exit code. |

## How the pieces fit together

- **`kernel/` is not a workspace member.** It has its own `[workspace]`
  (empty) declaration in `kernel/Cargo.toml`, so it never gets
  target/feature-unified with `xtask` (an ordinary host binary). It has its
  own `Cargo.lock` and its own `.cargo/config.toml` (custom `target`,
  `-Z build-std`, and the QEMU test `runner`).
- **The target JSON** (`targets/x86_64-chitti.json`) is derived from
  `x86_64-unknown-none`: soft-float, SIMD features off (locked until Phase
  3), `code-model: kernel` (higher-half), `disable-redzone: true`. Regenerate
  the baseline with `rustc -Z unstable-options --print target-spec-json
  --target x86_64-unknown-none` if you ever need to diff against a newer
  rustc — field names/types in target JSON do change between nightlies (see
  Troubleshooting below).
- **The Limine boot protocol bindings are hand-rolled** in
  `kernel/src/limine_protocol.rs` rather than depending on the `limine`
  crate: the current crates.io version requires the unstable `ptr_metadata`
  feature and very new edition-2024 conventions, whereas the wire format
  itself (magic numbers, base-revision handshake, framebuffer/memmap
  request/response layouts) has been stable since Limine 5.x. Hand-rolling
  the small subset Phase 0 needs keeps every `unsafe` block auditable.
- **The test harness is genuine `custom_test_frameworks`.** `kernel/src/lib.rs`
  sets `#![feature(custom_test_frameworks)]` / `#![test_runner(...)]` /
  `#![reexport_test_harness_main = "test_main"]`; `test_main` only exists
  when the crate is compiled with `--test` (i.e. via `cargo test`), so
  `cargo xtask test` shells out to real `cargo test --lib` rather than
  hand-rolling a fake harness. The compiled test binary is handed to QEMU by
  the `[target.x86_64-chitti] runner` in `kernel/.cargo/config.toml`, which
  points at `xtask/run-test-in-qemu.sh` (see Troubleshooting for why that's
  a shell shim and not a direct `cargo run` invocation).

- **PIC over APIC for Phase 1.** `arch::x86_64::pic.rs` remaps the legacy
  8259 to vectors 32-47 rather than bringing up the I/O-APIC — the handoff
  doc explicitly allows "APIC (or PIC fallback)", and the PIC needs no
  ACPI/MADT parsing to find redirection tables, which keeps the
  highest-risk part of this phase (interrupts working at all) small. APIC
  is real future work: Phase 7's SMP stretch goal needs a per-core LAPIC
  timer anyway.
- **GDT/IDT/paging are hand-rolled** (no `x86_64` crate), continuing Phase
  0's precedent for `limine_protocol.rs`: every `unsafe` block stays
  auditable, and it sidesteps another instance of the Phase 0
  duplicate-`core` build-std issue, since none of it pulls in a third-party
  dependency at all.

## Troubleshooting / rough edges hit during Phase 0

These are recorded here because they're easy to reintroduce by accident in
later phases, not because they're expected to recur if you leave the config
alone.

- **`.json` target specs need `-Z json-target-spec`.** On recent nightlies,
  `[build] target = "..."` pointing at a custom target JSON requires
  `json-target-spec = true` under `[unstable]` in `.cargo/config.toml`
  (already set in `kernel/.cargo/config.toml`).
- **Target JSON field types/names drift across nightlies.** This nightly
  wants `target-pointer-width`/`target-c-int-width` as integers (not
  strings), and `executables` (plural) rather than `executable`. If a future
  toolchain bump breaks the target JSON, `rustc --print target-spec-json
  --target x86_64-unknown-none` on the *new* toolchain will show the current
  expected shape.
- **Relative paths in `.cargo/config.toml`** (`target = "..."`,
  `-C link-arg=-Tlinker.ld`) resolve relative to the crate root that
  contains the config file's *package* (i.e. `kernel/`), not relative to the
  `.cargo/` directory itself. Hence `target = "../targets/x86_64-chitti.json"`,
  not `"../../targets/...`.
- **Don't add `panic = "abort"` per-profile overrides** (`[profile.dev]` /
  `[profile.test]` / `[profile.release]`) on top of the target spec's own
  `"panic-strategy": "abort"`. Doing so makes the package's root profile
  diverge from `-Z build-std`'s canonical sysroot profile, and Cargo ends up
  building **two** non-unified copies of `core`/`alloc`. Any ordinary
  (non-build-std) dependency — here, `font8x8`, used only by
  `framebuffer.rs` — then gets linked against a mix of both copies and fails
  with `duplicate lang item in crate 'core'`. The target spec's
  `panic-strategy` is already authoritative; leave profile overrides out.
  `framebuffer.rs` (and its `font8x8` dependency) is additionally excluded
  from the `cfg(test)` build via `#[cfg(not(test))]` in `lib.rs`, since the
  test harness never draws to the framebuffer.
- **Cargo's config-file discovery walks up from the current working
  directory, not from `--manifest-path`.** The `runner` originally invoked
  `cargo run --manifest-path ../xtask/Cargo.toml -- runner`, but that
  subprocess inherits the *caller's* cwd (`kernel/`), so it picked up
  `kernel/.cargo/config.toml`'s custom target/build-std/rustflags and tried
  to cross-compile `xtask` (an ordinary host binary) for the no_std kernel
  target. `xtask/run-test-in-qemu.sh` fixes this by explicitly `cd`-ing into
  `xtask/` before calling `cargo run`.
- **`-no-shutdown` makes QEMU pause instead of exit** when the guest
  triggers a shutdown — which is exactly what writing to `isa-debug-exit`
  does. It's fine (even useful) for `cargo xtask run`'s interactive session,
  but the test-runner QEMU invocation deliberately omits it so the process
  actually exits and `xtask runner` can read the real exit code back.
  `-no-reboot` is kept everywhere (prevents a triple-fault reboot loop).

## Troubleshooting / rough edges hit during Phase 1

- **Inline-asm `lateout` registers may alias `in` registers — order your
  writes accordingly.** `arch::x86_64::gdt::init`'s far-return CS reload
  originally computed the `lateout` return-address register (`tmp`)
  *before* consuming the `in` selector register (`code_sel`). LLVM is free
  to assign both to the same physical register since `lateout` promises
  "don't touch this until every `in` operand has been read" — writing
  `tmp` first breaks that promise, silently clobbers `code_sel`, and
  `retfq` jumps through a bogus code selector. Symptom: an instant `#GP`
  with the IDT still unloaded (limit 0), immediately cascading to a triple
  fault, with **zero serial output** since it happens before anything
  after it can log. `qemu-system-x86_64 -d int,cpu_reset -D
  /tmp/qemu_debug.log` (then `llvm-objdump -d` the test binary and look up
  the faulting `RIP`) is what actually found this — much faster than
  bisecting by adding print statements. Fix: order the asm so every `in`
  operand is consumed before any `lateout` operand is written (push the
  selector first, *then* compute the return address).
- **Don't build a large array on the stack before boxing it.**
  `Box::new([0u8; 64 * 1024])` constructs the 64 KiB array as a stack
  temporary first, then moves it into the heap allocation — and Limine's
  boot stack is nowhere near 64 KiB. The overflow doesn't necessarily fault
  cleanly (no guard page yet); it can silently corrupt adjacent kernel data
  (in one run, the heap's own free-list pointers), producing symptoms far
  from the actual cause, like an "allocator" that appears to spin forever.
  Use `alloc::vec![0u8; n]` instead: `Vec`'s `from_elem` path writes
  directly into the heap allocation without an `n`-byte stack temporary.
- **Size the frame-allocator bitmap off `USABLE` memmap entries only.**
  Some memory maps include a final huge `RESERVED` entry as a sentinel for
  "the rest of the 64-bit address space." Computing the bitmap's frame
  count from the highest `base + length` across *all* entries (instead of
  just `MEMMAP_USABLE` ones) inflates it to cover that entry too — one
  local repro sized a 2 GiB VM's bitmap for 1 TiB of address space, making
  every allocator scan needlessly (though not infinitely) slow.

## Verifying framebuffer output headlessly

`cargo xtask run` opens a normal QEMU display window. To check the
framebuffer without a display (e.g. over SSH or in CI), boot with a QEMU
monitor socket and issue a `screendump`:

```sh
cargo xtask image
qemu-system-x86_64 -M q35 -m 2G -cdrom target/chitti.iso \
  -serial file:/tmp/chitti_serial.log -display none -vga std \
  -monitor unix:/tmp/chitti.sock,server,nowait \
  -device isa-debug-exit,iobase=0xf4,iosize=0x04 -no-reboot &
sleep 2
printf 'screendump /tmp/chitti_fb.ppm\n' | nc -U /tmp/chitti.sock -w 2
kill %1
```

`/tmp/chitti_fb.ppm` is a raw PPM of the framebuffer; convert with
`sips -s format png` (macOS) or Pillow to view it.
