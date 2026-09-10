#!/usr/bin/env bash
# Exploratory follow-up; the completed head result and all final tests stay fixed.
set -euo pipefail
model_path="${1:?Usage: bash scripts/run_static_overlap_mlp_pilot.sh ORIGINAL_MODEL_PATH}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
pilot_out="$PWD/outputs/static_overlap_mlp_pilot_seed1"
head_protocol="$PWD/outputs/static_overlap_final_protocol_seed1/protocol.json"
mkdir -p "$PWD/outputs"

python -u scripts/static_overlap_mlp_protocol.py \
  --head-protocol "$head_protocol" --wikidata-dir "$PWD/data/wikidata" \
  --output-dir "$pilot_out" 2>&1 | tee "$pilot_out.prepare.log"

python -u scripts/run_static_overlap_mlp_pilot.py \
  --pilot-protocol "$pilot_out/pilot_protocol.json" --model-path "$model_path" \
  --device cuda --local-files-only 2>&1 | tee "$pilot_out.training.log"

# A failed development gate returns 2 above. set -e and pipefail stop the launcher
# before this evaluator can load any final features or official Gen prompts.
python -u scripts/evaluate_static_overlap_mlp_pilot.py \
  --pilot-protocol "$pilot_out/pilot_protocol.json" \
  --wikidata-dir "$PWD/data/wikidata" --device cuda 2>&1 | tee "$pilot_out.evaluation.log"
