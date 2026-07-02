#!/bin/sh
# Fetch a quantized GGUF Chitti boots (as a Limine module on x86, or via
# QEMU -device loader on aarch64). Models are deliberately NOT committed
# (see assets/.gitignore) -- they're large and freely re-fetchable.
#
# Usage: fetch-model.sh [qwen3.5-0.8b | qwen3.5-9b]   (default: qwen3.5-0.8b)
#
#   qwen3.5-0.8b -> assets/model.gguf     (Q8_0, ~812 MB) -- the default model;
#                   the Phase 3 reference-parity gate (tools/ref.py) uses it.
#   qwen3.5-9b   -> assets/model-9b.gguf  (Q4_K, ~5.4 GB) -- larger model, built
#                   with `-model qwen3.5-9b` (see CLAUDE.md for the caveats).
set -e
DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:-qwen3.5-0.8b}"

case "$MODEL" in
  qwen3.5-0.8b|0.8b)
    DEST="$DIR/assets/model.gguf"
    URL="https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf"
    SIZE="~812 MB (Q8_0)"
    ;;
  qwen3.5-9b|9b)
    DEST="$DIR/assets/model-9b.gguf"
    URL="https://huggingface.co/huihui-ai/Huihui-Qwythos-9B-Claude-Mythos-5-1M-abliterated-GGUF/resolve/main/Huihui-Qwythos-9B-Claude-Mythos-5-1M-abliterated-Q4_K.gguf"
    SIZE="~5.4 GB (Q4_K)"
    ;;
  *)
    echo "unknown model '$MODEL' (expected qwen3.5-0.8b or qwen3.5-9b)" >&2
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
