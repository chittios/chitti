#!/bin/sh
# Fetch the tiny quantized model Chitti boots as a Limine module. The GGUF
# is deliberately NOT committed (see assets/.gitignore) -- it's large and
# freely re-fetchable. Phase 3's reference-parity gate (tools/ref.py) and
# the kernel both read this exact file.
set -e
DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$DIR/assets/model.gguf"
URL="https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf"
if [ -f "$DEST" ]; then
  echo "model already present: $DEST"
  exit 0
fi
echo "fetching Qwen3.5-0.8B Q8_0 (~812 MB) -> $DEST"
curl -fL --retry 3 -o "$DEST.partial" "$URL"
mv "$DEST.partial" "$DEST"
echo "done: $DEST"
