#!/usr/bin/env bash
# Export trained adapter to .fmadapter bundle for on-device use.
#
# MUST run on macOS (coremltools requirement).
# If trained on cloud GPU, copy checkpoint to Mac first.
#
# Usage:
#   ./scripts/07_export.sh
#   ./scripts/07_export.sh /path/to/custom-checkpoint.pt
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TOOLKIT_DIR="$PROJECT_DIR/apple-fm-toolkit"

export PYTHONPATH="${TOOLKIT_DIR}:${PYTHONPATH:-}"

CHECKPOINT="${1:-$PROJECT_DIR/adapter/iteration_1/adapter-final.pt}"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Error: Checkpoint not found at $CHECKPOINT"
    echo "Run training first: ./scripts/06_train.sh"
    exit 1
fi

echo "Exporting adapter from: $CHECKPOINT"

python -m export.export_fmadapter \
    -o "$PROJECT_DIR/adapter/exports/" \
    -n thai_language \
    -c "$CHECKPOINT" \
    --author "biwsantang" \
    --description "Thai language adapter for Apple Foundation Model, trained on Typhoon 2.5 dataset"

echo "Export complete. Bundle at: $PROJECT_DIR/adapter/exports/"
