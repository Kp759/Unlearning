#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ARM="${1:?usage: $0 ARM MODEL_PATH MCF_PATH OUTPUT_DIR}"
MODEL_PATH="${2:?usage: $0 ARM MODEL_PATH MCF_PATH OUTPUT_DIR}"
MCF_PATH="${3:?usage: $0 ARM MODEL_PATH MCF_PATH OUTPUT_DIR}"
OUTPUT_DIR="${4:?usage: $0 ARM MODEL_PATH MCF_PATH OUTPUT_DIR}"
shift 4

python -u scripts/run_static_overlap_fact_association_embeddings_v2_arm.py \
  --arm "$ARM" \
  --model-path "$MODEL_PATH" \
  --mcf-path "$MCF_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda \
  --local-files-only \
  "$@"
