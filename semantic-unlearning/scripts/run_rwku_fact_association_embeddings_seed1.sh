#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_PATH="${1:?usage: $0 MODEL_PATH RWKU_DATA_ROOT SPLIT_DIR OUTPUT_DIR}"
RWKU_DATA_ROOT="${2:?usage: $0 MODEL_PATH RWKU_DATA_ROOT SPLIT_DIR OUTPUT_DIR}"
SPLIT_DIR="${3:?usage: $0 MODEL_PATH RWKU_DATA_ROOT SPLIT_DIR OUTPUT_DIR}"
OUTPUT_DIR="${4:?usage: $0 MODEL_PATH RWKU_DATA_ROOT SPLIT_DIR OUTPUT_DIR}"
shift 4

python -u scripts/run_rwku_fact_association_embeddings_seed1.py \
  --model-path "$MODEL_PATH" \
  --data-root "$RWKU_DATA_ROOT" \
  --split-dir "$SPLIT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --seed 1 \
  --row-updates-per-fact 30 \
  --device cuda \
  --local-files-only \
  --no-download \
  "$@"
