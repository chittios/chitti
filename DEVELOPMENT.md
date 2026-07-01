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

## Troubleshooting / rough edges hit during Phase 2

- **Naked functions need `#[unsafe(naked)]` + `core::arch::naked_asm!` on
  this toolchain, not the older `#[naked]` + `asm!(..., options(noreturn))`
  pattern.** `naked_functions` stabilized without needing a `#![feature(...)]`
  gate; a stray one just produces a `stable_features` warning. `sched::context`
  relies on naked functions specifically so nothing (no compiler-generated
  prologue/epilogue) fights with hand-written stack manipulation: a *non*-naked
  function's own prologue would push/adjust the stack *before* our asm block
  runs, and popping only what we explicitly pushed later would leave that
  extra adjustment unaccounted for, corrupting the stack on `ret`.
- **A "preempt from inside the timer IRQ handler" context switch has to
  save/restore `RFLAGS`, not just callee-saved GPRs.** `sched::context::switch_to`
  saves the outgoing task's flags via `pushfq` and restores the incoming
  task's via `popfq`, symmetric with the GPR save/restore. Without this, a
  task preempted while running with interrupts enabled would resume (whenever
  it's switched back to) with interrupts disabled — because the *timer
  handler's* interrupt gate had cleared `IF` on entry, and nothing else would
  ever set it back for that specific task. Wrapping `sched::yield_now`'s
  bookkeeping-plus-switch in `interrupts::without_interrupts` (already used
  everywhere else for critical sections) turns out to compose correctly here
  too: `without_interrupts` only re-enables interrupts if they were enabled
  *at the point `yield_now` was called*, which is exactly the right answer
  whether that call came from ordinary task code (interrupts on) or from
  `on_timer_tick` inside the IRQ handler (interrupts already off).
- **A freshly spawned task's initial stack has to look exactly like a stack
  `switch_to` itself would have produced**, i.e. a return address (the
  trampoline) sitting under a saved `RFLAGS` word and six saved GPR slots, in
  the exact order `switch_to`'s `pop` sequence expects them. `sched::context::init_stack`
  smuggles the task's real entry-point function pointer and argument through
  the (otherwise-unused, for a task that's never run) `r12`/`r13` GPR slots:
  `switch_to`'s normal restore path pops them into the real registers, and the
  landing-pad `trampoline` naked function reads them from there before making
  an ordinary `call` into the task's entry function.
- **Don't test timer preemption by racing raw iteration counts against the
  timeslice length.** An early draft picked a target increment count assuming
  a 5-tick (5 ms) timeslice would only allow a "few" loop iterations; in
  practice a tight `fetch_add` loop under QEMU blows through hundreds of
  thousands of iterations in that window, so the first-scheduled task would
  finish entirely inside its first slice and never actually get preempted
  mid-run -- the test would "pass" without proving anything. Fixed by
  splitting into two tests: one fully deterministic cooperative-interleaving
  test (every worker calls `yield_now` every iteration, so interleaving is
  guaranteed regardless of host speed, and is checked via a log-transition
  count, not just final totals) and one preemption-specific test where a
  *single* task loops forever with no cooperation at all and the assertion is
  just "the main task's `hlt` loop ever regains control" -- a property that's
  either true (preemption works) or the test hangs, with no speed-dependent
  middle ground.

## Troubleshooting / rough edges hit during Phase 3 (Cortex inference)

- **Enable SSE at the hardware level as the very first boot action.** Turning
  on SIMD codegen crate-wide (`+sse,+sse2` in the target JSON, dropping
  soft-float) means the optimizer emits XMM instructions in *ordinary* code —
  vectorized loops, struct moves — anywhere, including the early boot path
  before `fpu::init` used to run. An SSE instruction with `CR0.EM`/`CR4.OSFXSR`
  unset faults, and with no IDT loaded yet that's an instant triple fault with
  zero serial output. Fix: a tiny `fpu::enable_sse` (CR0.EM=0/MP=1, CR4.OSFXSR)
  called as the *first* statement in every `_start`, long before the fuller
  `fpu::init`. Symptom before the fix: release builds triple-faulted at boot
  while debug builds (which didn't vectorize the early path) worked — a classic
  "works in debug, dies in release" that the `-d int,cpu_reset` QEMU log
  (CR4=0x20, IDT=0) pinpointed immediately.
- **Preserve SSE state across context switches.** With SIMD on, tasks keep live
  `f32` accumulators in XMM across a matmul; the timer can preempt mid-matmul.
  `sched::context` now `FXSAVE`/`FXRSTOR`s a per-task 512-byte area around the
  switch. `FXSAVE` (not `XSAVE`) is used deliberately: it needs only
  `CR4.OSFXSR`, so it works on QEMU's default CPU, which reports no XSAVE.
  Verified with `llvm-objdump` that the interrupt-handler/scheduler path itself
  stays XMM-clean, so nothing clobbers the interrupted task's XMM before the
  save runs.
- **`debug_assert!` is compiled out in release — bounds bugs surface as OOB.**
  Two matvec calls in the first forward pass had `n_rows`/`n_cols` swapped; the
  `debug_assert_eq!(x.len(), n_cols)` guards silently vanished in the release
  build the model runs in, and the mismatch became a slice-index panic deep in
  a matmul. Lesson: for release-only code paths, either keep the invariant a
  real `assert!` or test the exact release configuration.
- **The frame allocator's linear scan is O(n²) over a bulk mapping.** Mapping
  the (now 256 MiB) Cortex heap is tens of thousands of single-frame
  allocations; scanning the bitmap from frame 0 every call made boot hang for
  minutes at 99% CPU. Fixed with a next-fit cursor (`next_hint`) so a run of
  sequential allocations is amortized O(1). This also speeds up KV/state growth
  during inference.
- **Reconstructing a frontier architecture needs the reference implementation,
  not guesswork.** The Qwen3.5-0.8B model is a gated-DeltaNet + gated-attention
  hybrid whose exact recurrence (`g = exp(-exp(A)·softplus(α+dt))`, delta rule
  `S = g·S + β·kᵀ(v−Sᵀk)`), gate activations (SiLU on the DeltaNet `z`, sigmoid
  on the attention gate), interleaved per-head query/gate layout, and partial
  mRoPE cannot be inferred reliably from tensor shapes alone — a from-shapes
  reconstruction produced fluent gibberish. Building the NumPy reference
  (`tools/ref_qwen35.py`) against llama.cpp's `src/models/qwen35.cpp` graph got
  it to coherent, correct output ("Paris is the capital of France, ..."), which
  then ported cleanly to the kernel. The host-first workflow (get NumPy
  coherent *before* the slow kernel port) is what made this tractable.
- **`cargo xtask test` excludes the model; `cargo xtask ref-check` includes it.**
  The fast unit suite (tensor kernels vs baked-in NumPy vectors, sampler,
  Phase 0–2) must not bundle or boot the ~812 MB model, or every test run
  balloons. The `refcheck` cargo feature switches the boot binary from the
  single-stream demo to the full acceptance gate, which `ref-check` builds in
  release, boots with the model + 4 GiB RAM, and reads the `REFCHECK:` serial
  lines / exit code from.

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
