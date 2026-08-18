#!/bin/sh
# Build the Doom package's `tools.wasm`.
#
# A shell script rather than a cargo crate because there is no Rust here: the
# guest is vendored C plus one platform file. `tools/doombench` is the Rust
# harness and shares both substituted sources with this build, so the module that
# ships is compiled from the same code that was measured at 29x / 356 fps.
#
# Requires: brew install llvm wasi-libc wasi-runtimes
set -eu

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DG="$ROOT/third_party/doomgeneric/doomgeneric"
OUT="$ROOT/agents/freedoom/assets/tools.wasm"

CLANG="$(brew --prefix llvm)/bin/clang"
SYSROOT="$(brew --prefix wasi-libc)/share/wasi-sysroot"
# `wasi-runtimes` is a clang **resource dir**, not a sysroot. Without
# `-resource-dir` the link fails with `cannot open .../libclang_rt.builtins.a`
# naming a path inside the *llvm* cellar, which reads like a broken llvm rather
# than a missing package. Doom needs those builtins for 64-bit division alone.
RESDIR="$(brew --prefix wasi-runtimes)/share/wasi-runtimes"

# Same three substitutions the bench makes, and for the same reasons — see
# third_party/doomgeneric/VENDORING.md. Nothing upstream is edited.
SRCS=""
for f in "$DG"/*.c; do
  b=$(basename "$f")
  case "$b" in
    doomgeneric_*.c|i_main.c|w_file_stdc.c) continue ;;      # substituted
    i_allegromusic.c|i_allegrosound.c|i_sdlmusic.c|i_sdlsound.c) continue ;;  # audio backends
  esac
  SRCS="$SRCS $f"
done

mkdir -p "$(dirname "$OUT")"

# -O3 and -msimd128 are both load-bearing and both measured: the PDF renderer
# found opt-level 3 worth 8.5x over `s`, and wasm SIMD a further 1.5-5.8x. They
# live here rather than being passed by hand so a rebuild cannot lose them.
# shellcheck disable=SC2086
"$CLANG" --target=wasm32-wasip1 \
  --sysroot="$SYSROOT" -resource-dir="$RESDIR" \
  -O3 -msimd128 -fno-strict-aliasing -Wno-everything \
  -DCMAP256 -DDOOMGENERIC_RESX=320 -DDOOMGENERIC_RESY=200 \
  -I "$DG" \
  -nostartfiles -Wl,--no-entry -Wl,--export-dynamic \
  -Wl,-z,stack-size=1048576 -Wl,--initial-memory=134217728 \
  -o "$OUT" \
  $SRCS "$ROOT/tools/freedoom-wasm/src/platform.c" \
  "$ROOT/tools/doombench/src/w_file_memory.c"

echo "built $OUT ($(wc -c < "$OUT") bytes)"
