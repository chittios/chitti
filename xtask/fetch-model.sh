#!/bin/sh
# Fetch a quantized GGUF Chitti boots (as a Limine module on x86, or via
# QEMU -device loader on aarch64). Models are deliberately NOT committed
# (see assets/.gitignore) -- they're large and freely re-fetchable.
#
# Usage: fetch-model.sh [qwen3.5-0.8b | qwen3.5-2b | qwen3.5-4b | qwen3.5-9b]  (default: qwen3.5-2b)
#
#   qwen3.5-0.8b -> assets/model.gguf     (Q8_0, ~812 MB) -- the compact model;
#                   the Phase 3 reference-parity gate (tools/ref.py) uses it.
#   qwen3.5-2b   -> assets/model-2b.gguf  (Q4_0, ~1.2 GB) -- the DEFAULT model.
#   qwen3.5-4b   -> assets/model-4b.gguf  (Q4_0, ~2.6 GB) -- mid-size model, built
#                   with `-model qwen3.5-4b`.
#   qwen3.5-9b   -> assets/model-9b.gguf  (Q4_0, ~5.0 GB) -- larger model, built
#                   with `-model qwen3.5-9b` (see CLAUDE.md for the caveats).
set -e
DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:-qwen3.5-2b}"

case "$MODEL" in
  qwen3.5-0.8b|0.8b)
    DEST="$DIR/assets/model.gguf"
    URL="https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf"
    SIZE="~812 MB (Q8_0)"
    ;;
  qwen3.5-2b|2b)
    DEST="$DIR/assets/model-2b.gguf"
    # Q4_0 (not Q4_K): Q4_0 is a format Chitti's kernel supports directly.
    URL="https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_0.gguf"
    SIZE="~1.2 GB (Q4_0)"
    ;;
  qwen3.5-4b|4b)
    DEST="$DIR/assets/model-4b.gguf"
    # Q4_0 (not Q4_K): Q4_0 is a format Chitti's kernel supports directly.
    URL="https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_0.gguf"
    SIZE="~2.6 GB (Q4_0)"
    ;;
  qwen3.5-9b|9b)
    DEST="$DIR/assets/model-9b.gguf"
    # Q4_0 (not Q4_K): Q4_0 is a format Chitti's kernel supports directly.
    URL="https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_0.gguf"
    SIZE="~5.0 GB (Q4_0)"
    ;;
  *)
    echo "unknown model '$MODEL' (expected qwen3.5-0.8b, qwen3.5-2b, qwen3.5-4b, or qwen3.5-9b)" >&2
    exit 1
    ;;
esac

if [ -f "$DEST" ]; then
  echo "model already present: $DEST"
  exit 0
fi
echo "fetching $MODEL $SIZE -> $DEST"
curl -fL --retry 3 -o "$DEST.partial" "$URL"
mv "$DEST.partial" "$DEST"
echo "done: $DEST"
