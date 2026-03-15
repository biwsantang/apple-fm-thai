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
#   - GPU with 24GB+ VRAM (H100/A100) or Mac with 32GB+ RAM
#
# Key hyperparameters:
#   epochs=3          Small dataset (3.2K train), avoid overfitting
#   lr=1e-3           Apple toolkit default for this model
#   batch=4 x accum=4 Effective batch size 16
#   pack-sequences    Mean 481 tokens vs 4096 max → ~8x throughput
#   precision         Auto-detected: f16-mixed on Mac (MPS), bf16-mixed on CUDA
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

# Auto-detect precision: MPS (Mac) doesn't support bf16, use f16-mixed instead
if [[ "$(uname)" == "Darwin" ]]; then
    PRECISION="f16-mixed"
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
    echo "Detected macOS — using f16-mixed precision (MPS doesn't support bf16)"
else
    PRECISION="bf16-mixed"
    echo "Detected Linux — using bf16-mixed precision"
fi

# Allow overriding epochs from CLI (e.g., ./06_train.sh --epochs 1)
EXTRA_ARGS=("$@")

python -m examples.train_adapter \
    --train-data "$PROJECT_DIR/data/processed/iteration_1/train.jsonl" \
    --eval-data "$PROJECT_DIR/data/processed/iteration_1/eval.jsonl" \
    --epochs 3 \
    --learning-rate 1e-3 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --pack-sequences \
    --max-sequence-length 4095 \
    --activation-checkpointing \
    --precision "$PRECISION" \
    --checkpoint-frequency 1 \
    --weight-decay 0.01 \
    --clip-grad-norm 1.0 \
    --checkpoint-dir "$PROJECT_DIR/adapter/iteration_1/" \
    "${EXTRA_ARGS[@]}"
