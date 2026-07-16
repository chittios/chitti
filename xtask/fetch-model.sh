#!/bin/sh
# Fetch a quantized GGUF Chitti boots (as a Limine module on x86, or via
# QEMU -device loader on aarch64). Models are deliberately NOT committed
# (see assets/.gitignore) -- they're large and freely re-fetchable.
#
# Usage: fetch-model.sh [NAME]  (default: qwen3.5-0.8b)
#
#   qwen3.5-0.8b  -> assets/model.gguf            (Q8_0, ~812 MB) -- DEFAULT
#   qwen3.5-2b    -> assets/model-2b.gguf         (Q4_0, ~1.2 GB)
#   qwen3.5-4b    -> assets/model-4b.gguf         (Q4_0, ~2.6 GB)
#   qwen3.5-9b    -> assets/model-9b.gguf         (Q4_0, ~5.0 GB)
#   gemma-4-e4b   -> assets/model-gemma4-e4b.gguf (Q4_K_M, ~4.6 GB)
#                    unsloth/gemma-4-E4B-it-GGUF (Cortex Gemma4 family)
#   bonsai-27b    -> assets/model-bonsai-27b.gguf (Q2_0 ternary, ~7.17 GB) -- DEFAULT for `make run`
#                    prism-ml/Ternary-Bonsai-27B-gguf main weights (arch qwen35 -> QwenHybrid)
set -e
DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:-qwen3.5-0.8b}"

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
  gemma-4-e4b|gemma-4-E4B|gemma4-e4b|gemma4-E4B|gemma-4-E4B-it|e4b|E4B)
    DEST="$DIR/assets/model-gemma4-e4b.gguf"
    # Q4_K_M: Cortex has full Q4_K dequant + SDOT matvec; Gemma4 chat format
    # is already architecture-dynamic in the kernel.
    URL="https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf"
    SIZE="~4.6 GB (Q4_K_M)"
    ;;
  bonsai-27b|bonsai27b|bonsai|ternary-bonsai-27b|Ternary-Bonsai-27B)
    DEST="$DIR/assets/model-bonsai-27b.gguf"
    # Ternary-Bonsai-27B main weights (Q2_0 ternary, GGML type 42). The GGUF
    # declares general.architecture=qwen35 with full DeltaNet SSM keys, so Cortex
    # runs it on the existing QwenHybrid family; the Q2_0 dequant is in
    # cortex::tensor::dequant_q2_0_block. (The paired *dspark* drafter is NOT a
    # standalone model -- it is conditioned on the target's hidden-state taps --
    # so the runnable Bonsai is this target.)
    URL="https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf/resolve/main/Ternary-Bonsai-27B-Q2_0.gguf"
    SIZE="~7.17 GB (Q2_0 ternary)"
    ;;
  *)
    echo "unknown model '$MODEL' (expected qwen3.5-0.8b|2b|4b|9b, gemma-4-e4b, or bonsai-27b)" >&2
    exit 1
    ;;
esac

if [ -f "$DEST" ]; then
  echo "model already present: $DEST"
  exit 0
fi
echo "fetching $MODEL $SIZE -> $DEST"
# `-C -` resumes a prior interrupted download (HF supports byte ranges), so a
# killed `make model` (e.g. SIGTERM) can pick up the existing `.partial` instead
# of re-pulling multi-GB weights from scratch. `--retry` covers transient drops.
curl -fL -C - --retry 5 --retry-delay 2 -o "$DEST.partial" "$URL"
mv "$DEST.partial" "$DEST"
echo "done: $DEST"
