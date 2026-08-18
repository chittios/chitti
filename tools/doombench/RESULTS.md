# doombench — measured, 2026-08-18

Host: Apple Silicon (M-series), macOS. `freedoom1.wad` 27.5 MB, 320x200, CMAP256.
Both sides built from the same vendored sources with the same defines; wasm at
`-O3 -msimd128` under **wasmi 1.1 + `simd`** — the same interpreter and features
the kernel uses (`kernel/Cargo.toml`), so the figures transfer.

```
agreement: frame   MATCH  (native 0xf991bc0b9c1fa5dd, wasmi 0xf991bc0b9c1fa5dd)
agreement: gametic MATCH  (native 1184, wasmi 1184)

 native: 10365.97 fps    0.10 ms/frame   (0.12s / 1200 frames)
  wasmi:   355.97 fps    2.81 ms/frame   (3.37s / 1200 frames)

interpreter tax: 29.1x native
fuel:            2,320,967 per frame
```

## The verdict: a wasm app package works

**356 fps under the interpreter, against a 35 fps target** (Doom's own tic rate).
In-kernel will be slower than a host figure — the PDF renderer measured ~3x on
heavy pages, scaling with working set and suspected to be guest TLB pressure under
stage-2 translation — which still leaves roughly **120 fps**, and about **3.4x
headroom** even if in-kernel turns out to be 10x rather than 3x.

So the port does **not** need a ring-3 tenant, and does not need the chunked
stateful tenant (`synapse/chunked.rs`, whose ABI exists with no users) to be built
first. It can be an ordinary signed, capability-gated wasm app package like every
other app here. That was the open question this harness existed to close.

## Agreement is the more important half

`frame MATCH` and `gametic MATCH` mean the two builds produced **byte-identical
paletted frames and an identical simulation clock** over 1200 frames. Two builds
agreeing on gametic is much stronger than agreeing on pixels: it means every tic
took the same branches, so the physics, the RNG and the AI all match.

That matters twice. It says the ratio above is a measurement of the *interpreter*
and not of two different programs (the rule `/html bench` follows — report
agreement before speed). And it is the mechanism Phase 3's correctness oracle uses:
if the in-kernel port reports the same gametic count for the same demo, the port is
right.

## 29.1x, and why the old 47-67x figure did not settle this

CLAUDE.md records wasmi at **47-67x native** on PNG decode — the measurement that
sent image decoding to ring 3 — and **3-30x** for the PDF rasterizer after the move
to wasmi 1.1 with `simd128`. Doom lands at 29.1x, near the top of the newer range,
so the tax is real and unsurprising. The reason the conclusion flips anyway is
**headroom, not efficiency**: Doom was written for a 33 MHz 486, so 29x slower than
a modern core is still an order of magnitude faster than it needs to be. A decoder
has no such margin, because the input size is the attacker's choice.

## Two numbers that constrain the port

- **Fuel: ~50.1 M for init**, and that is the number a per-call ceiling has to
  cover, because the largest single call is what a per-call budget must fit.
  Init loads and indexes a 28 MB WAD and builds Doom's tables, so it is nothing
  like a frame. `package_ui` hardcoded 2,000,000 and never read `wasm.fuel` at
  all, so a package could declare a budget, appear to be granted it, and trap at
  25x under what it needed. Found by booting the OS, not by a test.
- **Fuel: ~2.32 M per frame**, measured on a *quiet* map start;
  `service/package_ui.rs`'s `CALL_FUEL` is **2,000,000 per export call**, so a Doom
  frame already exceeds the default by 1.2x before anything is happening on screen.
  A busy scene will be well above that. The manifest must raise `wasm.fuel`
  substantially (`agents/git` already declares 5,000,000,000, so there is
  precedent), and generously: a guest that runs out of fuel mid-frame **traps**, and
  for a persistent-instance package-UI app that loses the game, not a frame.
- **Fuel metering is on in the kernel and costs a measured 3.7%.** It is off in the
  timing pass here and on in the fuel pass, deliberately — mixing them makes both
  numbers harder to read. Apply the 3.7% to the figures above; it does not change
  any conclusion.

## Reproducing

```sh
brew install llvm wasi-libc wasi-runtimes
cd tools/doombench
cargo run --release -- /path/to/freedoom1.wad 1200
```

Freedoom is 3-clause BSD and freely redistributable:
<https://github.com/freedoom/freedoom/releases>. The OS ships a copy at
`agents/freedoom/assets/freedoom1.wad`, so this harness can be pointed at that rather
than at a download. It is deliberately **not** in `SAMPLE_FILES` — that corpus is
fetched-never-committed to avoid taking a redistribution decision, which Freedoom's
licence makes for us.

## Two things this harness found on the way

Both are recorded because each is a live hazard for the real port, not harness
trivia.

1. **`DG_SleepMs` must advance the clock, or Doom livelocks.** `TryRunTics` waits
   for the game clock to reach the next tic by looping `{ I_Sleep(1); NetUpdate(); }`
   — and `doomgeneric_Create` reaches it *before* it returns. A platform whose clock
   only moves per-frame spins there forever; the first run of this harness sat in
   `TryRunTics -> NetUpdate -> DG_GetTicksMs` with the clock pinned at 0, found by
   sampling the process. On ChittiOS the failure is nastier: `host_now_ms()` does
   advance, so instead of a livelock the pump spins *inside one guest call*, burning
   fuel and never yielding to `upkeep()` — a frozen shell rather than a frozen game.
2. **Doom needs exactly 14 WASI imports, and only a few must work.**
   `fd_write` (diagnostics), `proc_exit`, and enough of `fd_prestat_get` /
   `fd_prestat_dir_name` / `path_open` / `fd_read` / `fd_seek` / `fd_close` /
   `fd_fdstat_get` to let it *identify* the IWAD; `path_{create_directory,
   remove_directory,rename,unlink_file}` and `fd_fdstat_set_flags` may all just fail.
   The lump data never goes through any of them — `w_file_memory.c` sets
   `wad_file_t::mapped`, so Doom reads lumps straight out of linear memory. Note the
   IWAD **path** is still needed even though the bytes are in memory: Doom picks its
   game mode from the *filename*, and without it reports "Game mode indeterminate".

   Get an arity wrong and wasmi refuses the whole module with `missing imports`,
   naming nothing — the same trap that made the kernel's five WASI stubs
   uninstantiable when they were all declared `() -> i32`. The reliable way to find
   the real list is to parse the module's import section, not to guess one rebuild
   at a time.
