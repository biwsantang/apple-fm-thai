#!/usr/bin/env bash
# Train Apple FM Thai language adapter using Typhoon dataset.
#
# Usage:
#   ./scripts/06_train.sh              # Full training (3 epochs)
#   ./scripts/06_train.sh --epochs 1   # Quick test run
#
# Requirements:
#   - Apple FM toolkit at apple-fm-toolkit/
#   - Processed data at data/processed/iteration_1/
#   - Mac with 32GB+ Apple Silicon, or Linux GPU with 24GB+ VRAM
#
# Key hyperparameters:
#   epochs=3          Small dataset (3.2K train), avoid overfitting
#   lr=1e-3           Apple toolkit default for this model
#   precision         Auto-detected: f16-mixed on Mac (MPS), bf16-mixed on CUDA
#
# Batch size notes:
#   With --pack-sequences, each batch element is a FULL 4095-token packed
#   sequence (not a single ~481-token conversation). This means batch_size
#   has ~8x more memory impact than without packing.
#   - Logits tensor per step: batch × 4095 × 153,600 vocab
#   - batch=4 OOMs even on A100 80GB (~9.4 GB logits alone)
#   - Rule: keep batch_size=1-2 with packing, use gradient_accumulation
#     to reach target effective batch size (e.g., batch=1 × accum=16 = 16)
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TOOLKIT_DIR="$PROJECT_DIR/apple-fm-toolkit"

if [[ ! -d "$TOOLKIT_DIR/assets" ]]; then
    echo "Error: Apple FM toolkit not found at $TOOLKIT_DIR"
    echo "Download from https://developer.apple.com/apple-intelligence/foundation-models-adapter/"
    exit 1
fi

export PYTHONPATH="${TOOLKIT_DIR}:${PYTHONPATH:-}"
export PYTHONHASHSEED=42

# Auto-detect platform and set optimizations
if [[ "$(uname)" == "Darwin" ]]; then
    PRECISION="f16-mixed"
    BATCH_SIZE=1
    GRAD_ACCUM=16
    # MPS memory: disable hard cap, let OS manage swap
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
    # MPS fallback: silently run unsupported ops on CPU instead of crashing
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    # MPS fast math: trade negligible precision for faster Metal kernels
    export PYTORCH_MPS_FAST_MATH=1
    # Optimize CPU thread count for M-series performance cores
    export OMP_NUM_THREADS=8
    echo "Detected macOS — f16-mixed, batch=$BATCH_SIZE, accum=$GRAD_ACCUM"
else
    PRECISION="bf16-mixed"
    BATCH_SIZE=2
    GRAD_ACCUM=8
    # Reduce CUDA memory fragmentation
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    echo "Detected Linux — bf16-mixed, batch=$BATCH_SIZE, accum=$GRAD_ACCUM"
fi

# Allow overriding epochs from CLI (e.g., ./06_train.sh --epochs 1)
EXTRA_ARGS=("$@")

python -m examples.train_adapter \
    --train-data "$PROJECT_DIR/data/processed/iteration_1/train.jsonl" \
    --eval-data "$PROJECT_DIR/data/processed/iteration_1/eval.jsonl" \
    --epochs 3 \
    --learning-rate 1e-3 \
    --batch-size "$BATCH_SIZE" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --pack-sequences \
    --max-sequence-length 4095 \
    --activation-checkpointing \
    --precision "$PRECISION" \
    --checkpoint-frequency 1 \
    --weight-decay 0.01 \
    --clip-grad-norm 1.0 \
    --checkpoint-dir "$PROJECT_DIR/adapter/iteration_1/" \
    "${EXTRA_ARGS[@]}"
