# chitti-fuzz

Mutational fuzzer for the kernel's **pure, attacker-reachable parsers**. It
mounts the real kernel source via `#[path]` — the `pngbench`/`onnxdiff`/
`cortexdiff`/`h264diff` pattern — so the bytes it mutates run through exactly
the code that runs in the OS, and it needs no QEMU and no kernel build.

## What it finds

A parser that **panics** on hostile input. In a `no_std` kernel that is a
denial of service (kernel `abort` takes the whole OS down), so it matters even
for ring-3-confined decoders: the ring-3 differential (`cargo xtask test`) is
only as good as the in-kernel decoder it diffs against.

Memory-safety bugs in `unsafe` code are *out of scope* — that is what the ring-3
tenant sandbox (`synapse::tenant`) is for. This harness is the cheap first pass
that runs in seconds on the host.

## Usage

```sh
cd tools/fuzz
cargo run --release -- <target> [iterations] [--seed N] [--time SECS]
cargo run --release -- --selftest        # sanity-check the harness itself
```

- **Targets**: `json` (kernel JSON parser — consumes model output), `png`
  (PNG+DEFLATE image decoder), `sha1` (WPA2 SHA-1/HMAC/PBKDF2). List them with
  no argument.
- **Seeds**: every file under `corpus/<target>/`. Start with a few valid inputs
  per target (one is enough to mutate from).
- **Crash inputs**: written to `crashes/<target>/crash-…bin`, and the exit code
  is non-zero when any were found — so `cargo run --release -- <target>` fails
  a script when a parser panics.
- **Determinism**: the whole run is a function of `--seed`, so a reported crash
  replays exactly with the same seed and iteration count.

A crashed input should become a **regression test** in the kernel's own
`#[test_case]` suite (the file is already a byte-perfect reproducer).

## Adding a target

1. Pick a kernel module that is `no_std` + `alloc`-only (no `crate::` paths
   beyond `alloc`/`core`, no smoltcp/hardware). If the parser you want isn't
   pure yet, extract the pure core into a mountable module first — that is a
   genuine confinement win, not just a fuzz convenience (the `/decoder ring3`
   work already moves the biggest attackers into ring 3; a pure parser is the
   cheapest thing to confine).
2. Mount it in `src/targets/<name>.rs`:
   ```rust
   #[path = "../../../../kernel/src/…/mod.rs"]
   pub mod the_module;
   ```
   (`../../../../` from `src/targets/` is the repo root.) Shim whatever it
   references — e.g. `image/png.rs` needs a host `Image { w, h, pixels }`.
3. Add a `pub fn run(data: &[u8])` that drives the parser (round-trip through
   a serializer too — a serializer that emits unparseable output is a real
   panic). Register it in `src/targets/mod.rs`.

## Why not libFuzzer / cargo-fuzz?

Portability and zero deps: this builds anywhere the host toolchain does
(macOS + Linux, stable, no clang `-fsanitize=fuzzer`), matching the repo's
stand-alone-tool ethos. The trade-off is a simple mutator instead of
coverage-guided instrumentation — for panic-hunting in ~KiB-scale parsers that
is usually enough. If a target needs real coverage guidance later, the `Target`
shape ports to a `fuzz_target!` trivially.
