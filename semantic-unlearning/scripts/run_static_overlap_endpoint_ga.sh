#!/usr/bin/env bash
# One separately registered endpoint experiment; preserve every prior result.
set -euo pipefail
model_path="${1:?Usage: bash scripts/run_static_overlap_endpoint_ga.sh ORIGINAL_MODEL_PATH ORIGINAL_OVERLAP_MANIFEST}"
overlap_manifest="${2:?Provide manifest.json from the earlier static-overlap replay run}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
endpoint_out="$PWD/outputs/static_overlap_endpoint_ga_seed1"
development_protocol="$PWD/outputs/static_overlap_mlp_pilot_seed1/pilot_protocol.json"
if [[ -e "$endpoint_out" ]]; then
  printf '%s\n' "Endpoint run already exists: $endpoint_out. Preserve its results; refusing to restart."
  exit 2
fi
if [[ ! -r "$development_protocol" || ! -r "$overlap_manifest" ]]; then
  printf '%s\n' "Missing development protocol or original overlap manifest; no training started."
  exit 2
fi

python -u scripts/static_overlap_endpoint_protocol.py \
  --development-protocol "$development_protocol" --overlap-manifest "$overlap_manifest" \
  --output-dir "$endpoint_out" 2>&1 | tee "$endpoint_out.prepare.log"

python -u scripts/run_static_overlap_endpoint_ga.py \
  --pilot-protocol "$endpoint_out/pilot_protocol.json" --model-path "$model_path" \
  --device cuda --local-files-only 2>&1 | tee "$endpoint_out.training.log"

# A failed gate returns 2 and stops here. Only a strict verified export reaches
# the same previously frozen final tests; no new test set or threshold changes.
python -u scripts/evaluate_static_overlap_endpoint_ga.py \
  --pilot-protocol "$endpoint_out/pilot_protocol.json" \
  --wikidata-dir "$PWD/data/wikidata" --device cuda 2>&1 | tee "$endpoint_out.evaluation.log"
