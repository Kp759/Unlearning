#!/usr/bin/env bash
set -euo pipefail
model_path="${1:?Usage: bash scripts/run_static_overlap_paired_ga_gd.sh MODEL_PATH OVERLAP_MANIFEST}"
overlap_manifest="${2:?Provide the original overlap manifest.json}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
out="$PWD/outputs/static_overlap_untied_paired_ga_gd_seed1"
dev="$PWD/outputs/static_overlap_mlp_pilot_seed1/pilot_protocol.json"
if [[ -e "$out" ]]; then
  echo "Output already exists: $out; refusing to overwrite it." >&2
  exit 2
fi
python -u scripts/static_overlap_paired_protocol.py \
  --development-protocol "$dev" --overlap-manifest "$overlap_manifest" \
  --output-dir "$out" 2>&1 | tee "$PWD/outputs/static_overlap_untied_paired_ga_gd_seed1.prepare.log"
python -u scripts/run_static_overlap_paired_ga_gd.py \
  --pilot-protocol "$out/pilot_protocol.json" --model-path "$model_path" \
  --device cuda --local-files-only 2>&1 | tee "$out.training.log"
python -u scripts/evaluate_static_overlap_paired_ga_gd.py \
  --pilot-protocol "$out/pilot_protocol.json" \
  --wikidata-dir "$PWD/data/wikidata" --device cuda 2>&1 | tee "$out.evaluation.log"
