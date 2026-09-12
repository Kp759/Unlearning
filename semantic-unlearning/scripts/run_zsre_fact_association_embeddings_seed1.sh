#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_PATH="${1:?usage: $0 MODEL_PATH ZSRE_PATH SPLIT_DIR OUTPUT_DIR}"
ZSRE_PATH="${2:?usage: $0 MODEL_PATH ZSRE_PATH SPLIT_DIR OUTPUT_DIR}"
SPLIT_DIR="${3:?usage: $0 MODEL_PATH ZSRE_PATH SPLIT_DIR OUTPUT_DIR}"
OUTPUT_DIR="${4:?usage: $0 MODEL_PATH ZSRE_PATH SPLIT_DIR OUTPUT_DIR}"
shift 4

if [[ ! -f "$SPLIT_DIR/training_visible_forget.json" || ! -f "$SPLIT_DIR/split_manifest.json" ]]; then
  python -u scripts/build_zsre_zerounlearn_locked_no_neutral_split.py \
    --zsre-path "$ZSRE_PATH" \
    --output-dir "$SPLIT_DIR" \
    --seed 1 \
    --forget-num 50 \
    --retain-num 1000
fi

python -u scripts/run_zsre_fact_association_embeddings_seed1.py \
  --model-path "$MODEL_PATH" \
  --training-visible "$SPLIT_DIR/training_visible_forget.json" \
  --split-manifest "$SPLIT_DIR/split_manifest.json" \
  --output-dir "$OUTPUT_DIR" \
  --seed 1 \
  --forget-num 50 \
  --device cuda \
  --local-files-only \
  "$@"
