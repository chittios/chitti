#!/bin/sh
# Fetch a quantized GGUF Chitti boots (as a Limine module on x86, or via
# QEMU -device loader on aarch64). Models are deliberately NOT committed
# (see assets/.gitignore) -- they're large and freely re-fetchable.
#
# Usage: fetch-model.sh [NAME]  (default: qwen3.5-0.8b)
#
# CHITTI_PURE -- make the GGUF *uniformly* quantized, which is what unlocks
# Cortex's weight-stationary batched prefill on the largest share of the weights.
#
#   Batching is decided per tensor (cortex::model::has_batched_kernel): Q8_0,
#   Q1_0, Q2_0 always have a batched kernel, Q4_0 has one only with FEAT_I8MM,
#   and the K-quants have none -- those tensors fall back to one matvec per
#   position. `llama-quantize` upcasts selected tensors unless `--pure` is
#   passed, so the file published as "Q4_0" arrives as Q4_0 + Q8_0 + Q5_K + Q4_1
#   and leaves ~11% of projection bytes on the slow path. Making it pure
#   measured 25.0s -> 10.5s on a 44-token 4B prefill (2.4x, single core).
#
#   CHITTI_PURE=1      requantize the downloaded file in place
#                      (`--allow-requantize`; no extra download). The tensors
#                      that were upcast take a second lossy pass -- fine for
#                      throughput work, not for quality work.
#   CHITTI_PURE=bf16   fetch the BF16 weights and quantize from those instead
#                      (no requantize loss, but a much larger download).
#   CHITTI_PURE_TYPE   target type, default Q4_0. Use Q8_0 for VirtualBox: it
#                      masks FEAT_I8MM, and Q4_0's batched kernel is i8mm-only,
#                      so a pure Q4_0 file batches *nothing* there.
#
# Only applies to the mixed-quant Qwen fetches (2b/4b/9b). The 0.8B (Q8_0) and
# both Bonsai builds (Q1_0/Q2_0) are already uniform, and Gemma never takes the
# batched path at all, so those are left alone.
#
#   qwen3.5-0.8b  -> assets/model.gguf            (Q8_0, ~812 MB) -- DEFAULT
#   qwen3.5-2b    -> assets/model-2b.gguf         (Q4_0, ~1.2 GB)
#   qwen3.5-4b    -> assets/model-4b.gguf         (Q4_0, ~2.6 GB)
#   qwen3.5-9b    -> assets/model-9b.gguf         (Q4_0, ~5.0 GB)
#   gemma-4-e4b   -> assets/model-gemma4-e4b.gguf (Q4_K_M, ~4.6 GB)
#                    unsloth/gemma-4-E4B-it-GGUF (Cortex Gemma4 family)
#   bonsai-27b          -> assets/model-bonsai-27b-q1.gguf (Q1_0 1-bit, ~3.8 GB) -- DEFAULT for `make run`
#                          prism-ml/Bonsai-27B-gguf main weights (arch qwen35 -> QwenHybrid)
#   bonsai-27b-ternary  -> assets/model-bonsai-27b.gguf    (Q2_0 ternary, ~7.17 GB)
#                          prism-ml/Ternary-Bonsai-27B-gguf main weights
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
    # NB this file is NOT uniformly Q4_0 -- see CHITTI_PURE above.
    URL="https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_0.gguf"
    BF16_URL="https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-BF16.gguf"
    PURE_OK=yes
    SIZE="~1.2 GB (Q4_0)"
    ;;
  qwen3.5-4b|4b)
    DEST="$DIR/assets/model-4b.gguf"
    # Q4_0 (not Q4_K): Q4_0 is a format Chitti's kernel supports directly.
    # NB this file is NOT uniformly Q4_0 -- see CHITTI_PURE above.
    URL="https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_0.gguf"
    BF16_URL="https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-BF16.gguf"
    PURE_OK=yes
    SIZE="~2.6 GB (Q4_0)"
    ;;
  qwen3.5-9b|9b)
    DEST="$DIR/assets/model-9b.gguf"
    # Q4_0 (not Q4_K): Q4_0 is a format Chitti's kernel supports directly.
    # NB this file is NOT uniformly Q4_0 -- see CHITTI_PURE above.
    URL="https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_0.gguf"
    BF16_URL="https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-BF16.gguf"
    PURE_OK=yes
    SIZE="~5.0 GB (Q4_0)"
    ;;
  gemma-4-e4b|gemma-4-E4B|gemma4-e4b|gemma4-E4B|gemma-4-E4B-it|e4b|E4B)
    DEST="$DIR/assets/model-gemma4-e4b.gguf"
    # Q4_K_M: Cortex has full Q4_K dequant + SDOT matvec; Gemma4 chat format
    # is already architecture-dynamic in the kernel.
    URL="https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf"
    SIZE="~4.6 GB (Q4_K_M)"
    ;;
  bonsai-27b|bonsai27b|bonsai|bonsai-27b-1bit|bonsai-1bit)
    DEST="$DIR/assets/model-bonsai-27b-q1.gguf"
    # Bonsai-27B 1-bit (binary) main weights (Q1_0, GGML type 41). general.
    # architecture=qwen35 with full DeltaNet SSM keys -> Cortex QwenHybrid; the
    # Q1_0 dequant is in cortex::tensor::dequant_q1_0_block (bit ? +d : -d). The
    # smallest/fastest Bonsai (binary {-1,+1} g128, ~1.125 bpw).
    URL="https://huggingface.co/prism-ml/Bonsai-27B-gguf/resolve/main/Bonsai-27B-Q1_0.gguf"
    SIZE="~3.8 GB (Q1_0 binary)"
    ;;
  bonsai-27b-ternary|bonsai-ternary|ternary-bonsai-27b|Ternary-Bonsai-27B)
    DEST="$DIR/assets/model-bonsai-27b.gguf"
    # Ternary-Bonsai-27B main weights (Q2_0 ternary, GGML type 42). Same
    # qwen35/QwenHybrid architecture as the 1-bit build; Q2_0 dequant is in
    # cortex::tensor::dequant_q2_0_block. Higher quality at ~2x the footprint.
    URL="https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf/resolve/main/Ternary-Bonsai-27B-Q2_0.gguf"
    SIZE="~7.17 GB (Q2_0 ternary)"
    ;;
  *)
    echo "unknown model '$MODEL' (expected qwen3.5-0.8b|2b|4b|9b, gemma-4-e4b, bonsai-27b, or bonsai-27b-ternary)" >&2
    exit 1
    ;;
esac

# Download $2 to $1. `-C -` resumes a prior interrupted download (HF supports
# byte ranges), so a killed `make model` (e.g. SIGTERM) can pick up the existing
# `.partial` instead of re-pulling multi-GB weights from scratch. `--retry`
# covers transient drops. Only moved into place once complete, so an aborted
# fetch never leaves a truncated GGUF that would fail to parse at boot.
fetch() {
  curl -fL -C - --retry 5 --retry-delay 2 -o "$1.partial" "$2"
  mv "$1.partial" "$1"
}

PURE="${CHITTI_PURE:-}"
PURE_TYPE="${CHITTI_PURE_TYPE:-Q4_0}"
# `.pure` marks a file this script already made uniform, so re-running is a
# no-op instead of putting the weights through another lossy pass.
if [ -n "$PURE" ] && [ "${PURE_OK:-}" != "yes" ]; then
  echo "note: CHITTI_PURE ignored for '$MODEL' (already uniform, or never batched)"
  PURE=""
fi
if [ -n "$PURE" ] && ! command -v llama-quantize >/dev/null 2>&1; then
  echo "CHITTI_PURE set but llama-quantize is not on PATH (brew install llama.cpp)" >&2
  exit 1
fi

if [ "$PURE" = "bf16" ]; then
  # Quality-correct path: quantize from 16-bit weights, never requantize.
  SRC="$DIR/assets/.bf16-$(basename "$DEST")"
  if [ -f "$DEST" ] && [ -f "$DEST.pure" ]; then
    echo "model already present and uniform: $DEST"
    exit 0
  fi
  [ -f "$SRC" ] || { echo "fetching $MODEL BF16 source -> $SRC"; fetch "$SRC" "$BF16_URL"; }
  echo "quantizing $SRC -> $DEST ($PURE_TYPE, pure)"
  # token_embd stays at Q6_K: prefill never batches it (a row lookup, plus one
  # matvec for the final position when the output is tied), so upcasting it costs
  # no batching and keeps embedding quality.
  llama-quantize --pure --token-embedding-type q6_K "$SRC" "$DEST.partial" "$PURE_TYPE" 8
  mv "$DEST.partial" "$DEST"
  touch "$DEST.pure"
  [ -n "${CHITTI_KEEP_BF16:-}" ] || rm -f "$SRC"
elif [ -f "$DEST" ] && { [ -z "$PURE" ] || [ -f "$DEST.pure" ]; }; then
  echo "model already present: $DEST"
  exit 0
else
  [ -f "$DEST" ] || { echo "fetching $MODEL $SIZE -> $DEST"; fetch "$DEST" "$URL"; }
  if [ -n "$PURE" ]; then
    echo "requantizing $DEST in place ($PURE_TYPE, pure) -- upcast tensors take a second lossy pass"
    llama-quantize --allow-requantize --pure --token-embedding-type q6_K \
      "$DEST" "$DEST.partial" "$PURE_TYPE" 8
    mv "$DEST.partial" "$DEST"
    touch "$DEST.pure"
  fi
fi
echo "done: $DEST"
# An `if` rather than `[ … ] && echo`: whether `set -e` aborts on the failing
# test in an AND-list is shell-dependent, and this is the last line — a false
# test must not make `make model` look like it failed.
if [ -f "$DEST.pure" ]; then
  echo "  uniform $PURE_TYPE -- expect \"batched weights: 100%\" from /perf"
fi
