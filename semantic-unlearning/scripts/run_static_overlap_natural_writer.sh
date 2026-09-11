#!/usr/bin/env bash
set -euo pipefail

model_path="${1:?Usage: bash scripts/run_static_overlap_natural_writer.sh MODEL_PATH MCF_JSON [OUTPUT_DIR]}"
mcf_path="${2:?Provide MultiCounterFact JSON path}"
out="${3:-}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p "$PWD/outputs"

if [[ -z "$out" ]]; then
  out="$PWD/outputs/static_overlap_natural_writer_v1_seed1"
fi
if [[ -e "$out" ]]; then
  echo "Output already exists: $out; refusing to overwrite it." >&2
  exit 2
fi

python -u scripts/run_static_overlap_natural_writer.py \
  --model-path "$model_path" \
  --mcf-path "$mcf_path" \
  --output-dir "$out" \
  --device cuda \
  --local-files-only \
  --forget-num 50 \
  --retain-num 300 \
  --seed 1 \
  2>&1 | tee "$out.training.log"
