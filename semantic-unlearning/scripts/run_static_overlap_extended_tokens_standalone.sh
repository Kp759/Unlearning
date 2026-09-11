#!/usr/bin/env bash
set -euo pipefail

model_path="${1:?Usage: bash scripts/run_static_overlap_extended_tokens_standalone.sh MODEL_PATH MCF_JSON}"
mcf_path="${2:?Provide MultiCounterFact JSON path}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

out="$PWD/outputs/static_overlap_extended_tokens_standalone_v1_seed1"
if [[ -e "$out" ]]; then
  echo "Output already exists: $out; refusing to overwrite it." >&2
  exit 2
fi

python -u scripts/run_static_overlap_extended_tokens_standalone.py \
  --model-path "$model_path" \
  --mcf-path "$mcf_path" \
  --output-dir "$out" \
  --device cuda \
  --local-files-only \
  2>&1 | tee "$PWD/outputs/static_overlap_extended_tokens_standalone_v1_seed1.training.log"
