#!/usr/bin/env bash
# Build an Apple-Silicon device tree blob (DTB) for booting ChittiOS via m1n1.
#
# m1n1's linux.py needs the machine's base Linux device tree; m1n1 patches it at
# runtime (memory size, framebuffer, MMIO tunables from the ADT) before handing
# it to the payload. That DTB is NOT part of this repo — it comes from the Asahi
# Linux kernel's device-tree sources (GPL-2.0). This script sparse-clones just
# the Apple `dts` + `dt-bindings` (a few MB, not the whole kernel) and compiles
# the DTB with the system cpp + dtc — no full kernel checkout, no cross gcc.
#
# Usage:   tools/build-apple-dtb.sh [board]      # default: t8112-j473 (Mac mini M2)
# Output:  third_party/dtb/<board>.dtb           # gitignored; set CHITTI_DTB to it
#
# Boards (Mac mini): t8112-j473 = M2, t6020-j474 = M2 Pro. Any Apple board whose
# dts ships in arch/arm64/boot/dts/apple/ works; run with no arg to see the list
# on a bad name.
#
# NB: Asahi's dts uses floating-point cell values (GPU/CPU power PID gains) that
# only Asahi's *patched* dtc accepts; mainline dtc rejects them. They are pure
# DVFS tuning, unused for boot and unneeded by m1n1/ChittiOS, so we strip them
# before compiling.
set -euo pipefail

BOARD="${1:-t8112-j473}"
BRANCH="${ASAHI_BRANCH:-asahi}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CACHE="$ROOT/third_party/dtb/.asahi-linux"      # gitignored sparse-checkout cache
OUTDIR="$ROOT/third_party/dtb"
DTSREL="arch/arm64/boot/dts/apple"
CPP="${CPP:-clang -E}"

command -v dtc >/dev/null || { echo "error: need dtc (brew install dtc)"; exit 1; }
mkdir -p "$OUTDIR"

if [ ! -d "$CACHE/.git" ]; then
  echo "· sparse-cloning Asahi Linux dts ($BRANCH) — one-time, cached in $CACHE"
  git clone --depth 1 --single-branch --branch "$BRANCH" \
    --filter=blob:none --sparse \
    https://github.com/AsahiLinux/linux.git "$CACHE"
  git -C "$CACHE" sparse-checkout set "$DTSREL" include/dt-bindings
fi

DTSDIR="$CACHE/$DTSREL"
DTS="$DTSDIR/$BOARD.dts"
if [ ! -f "$DTS" ]; then
  echo "error: no dts for board '$BOARD' at $DTS"
  echo "available Apple boards:"; ls "$DTSDIR"/*.dts | xargs -n1 basename | sed 's/\.dts$//;s/^/  /'
  exit 1
fi

PP="$(mktemp)"; PPC="$(mktemp)"
trap 'rm -f "$PP" "$PPC"' EXIT
echo "· preprocessing $BOARD.dts"
# shellcheck disable=SC2086
$CPP -nostdinc -I "$DTSDIR" -I "$CACHE/include" -undef -D__DTS__ \
  -x assembler-with-cpp "$DTS" -o "$PP"
echo "· stripping float power-tuning props (mainline dtc can't parse them)"
grep -vE '= <[^>]*[0-9]\.[0-9]' "$PP" > "$PPC"
echo "· compiling DTB"
dtc -I dts -O dtb -o "$OUTDIR/$BOARD.dtb" "$PPC" 2>/dev/null

echo "✓ $OUTDIR/$BOARD.dtb"
echo "  export CHITTI_DTB=$OUTDIR/$BOARD.dtb"
