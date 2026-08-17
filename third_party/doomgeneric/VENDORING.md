# doomgeneric — vendored

Upstream: <https://github.com/ozkl/doomgeneric>
Commit: `dcb7a8dbc7a16ce3dda29382ac9aae9d77d21284` (2026-04-12)
License: **GPL-2.0** (`LICENSE`) — id Software's Doom source release plus
doomgeneric's platform-abstraction layer.

## Why this is here

Doom's renderer *is* a software rasterizer, which is exactly the shape ChittiOS
can run: no GPU, no shaders, fixed-point math, a paletted framebuffer. And
`doomgeneric` reduces an entire platform port to six functions and one buffer, so
the OS-facing surface is small enough to reason about:

```c
void doomgeneric_Create(int argc, char **argv);
void doomgeneric_Tick();                            /* renders one frame */
extern pixel_t* DG_ScreenBuffer;                    /* uint8_t under CMAP256 */
void DG_DrawFrame();
void DG_SleepMs(uint32_t ms);
uint32_t DG_GetTicksMs();
int DG_GetKey(int* pressed, unsigned char* key);    /* press/release edges */
void DG_SetWindowTitle(const char* title);
```

It also brings its own correctness oracle: demo playback is deterministic, so the
same demo must produce the same gametic count here as on the host.

## What was excluded, and why

Everything upstream ships is vendored **except**:

- `doomgeneric_*.c` — the platform ports (SDL, X11, Windows, emscripten, allegro,
  linuxvt, soso). ChittiOS supplies its own; keeping the others would mean
  vendoring their headers and having eight definitions of the same six symbols.
- `Makefile*`, `*.vcxproj*`, `*.sln`, `screenshots/` — the build is driven from
  `tools/doombench/build.rs`, so an unused makefile is just a second thing that
  can drift.

No source file is modified. That is deliberate and is the same rule the ring-3
image tenant follows: **one implementation, so porting cannot regress it.** The
platform layer lives entirely in our own `doomgeneric_chitti.c`, and anything that
needs changing upstream should be a `#if` in *our* file or a compile flag, never an
edit here. If an upstream edit ever becomes unavoidable, record it in this file.

## Build configuration

- `-DCMAP256` makes `pixel_t` a `uint8_t`, so the frame is 8-bit paletted. That is
  Doom's native format, and it cuts the per-frame copy across the wasm boundary
  from 256 KB to 64 KB at 320x200.
- `-DDOOMGENERIC_RESX` / `-DDOOMGENERIC_RESY` override the 640x400 default.
- `-msimd128` and `-O3` for the wasm build. Both matter and neither is optional:
  the PDF renderer measured `opt-level=3` as **8.5x** faster than `s`, and wasm
  SIMD as a further **1.5-5.8x**. The flags are pinned in the build script rather
  than passed by hand, for the reason `tools/pdfrender-wasm/.cargo/config.toml`
  pins its own: a rebuild must not be able to silently lose them.

## Toolchain

C to wasm needs a sysroot and a compiler-rt for the target. Apple's clang has no
wasm backend, so:

```sh
brew install llvm wasi-libc wasi-runtimes
```

- `clang`: `$(brew --prefix llvm)/bin/clang`
- `--sysroot`: `$(brew --prefix wasi-libc)/share/wasi-sysroot`
- `-resource-dir`: `$(brew --prefix wasi-runtimes)/share/wasi-runtimes`

The last one is the non-obvious step. `wasi-runtimes` is a clang **resource
directory**, not a sysroot, and without `-resource-dir` the link fails with
`cannot open .../libclang_rt.builtins.a: No such file or directory` — pointing at a
path inside the *llvm* cellar, which reads like a broken llvm install rather than a
missing package. Doom needs those builtins for 64-bit division alone.
