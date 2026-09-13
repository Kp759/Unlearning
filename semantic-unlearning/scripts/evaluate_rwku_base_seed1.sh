#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_PATH="${1:?usage: $0 MODEL_PATH RWKU_DATA_ROOT WIKIDATA_DIR [DTYPE] [OUT]}"
RWKU_DATA_ROOT="${2:?usage: $0 MODEL_PATH RWKU_DATA_ROOT WIKIDATA_DIR [DTYPE] [OUT]}"
WIKIDATA_DIR="${3:?usage: $0 MODEL_PATH RWKU_DATA_ROOT WIKIDATA_DIR [DTYPE] [OUT]}"
DTYPE="${4:-bfloat16}"
OUT="${5:-$ROOT/outputs/rwku_fact_assoc_seed1_base_eval.json}"

python -u scripts/evaluate_rwku_base_seed1.py   --model-path "$MODEL_PATH"   --data-root "$RWKU_DATA_ROOT"   --wikidata-dir "$WIKIDATA_DIR"   --dtype "$DTYPE"   --device cuda   --local-files-only   --no-download   --out "$OUT"
