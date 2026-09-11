#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_PATH="${1:?usage: $0 MODEL_PATH [MCF_PATH] [OUTPUT_DIR]}"
MCF_PATH="${2:-$ROOT/data/multi_counterfact.json}"
OUTPUT_DIR="${3:-$ROOT/outputs/static_overlap_fact_association_embeddings_v1_seed1}"

python -u scripts/run_static_overlap_fact_association_embeddings.py \
  --model-path "$MODEL_PATH" \
  --mcf-path "$MCF_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda \
  --local-files-only
