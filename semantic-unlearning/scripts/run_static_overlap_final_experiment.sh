#!/usr/bin/env bash
# One last protocol adaptation, followed by one fixed-checkpoint final evaluation.
set -euo pipefail

model_path="${1:?Usage: bash scripts/run_static_overlap_final_experiment.sh MODEL_PATH SOURCE_HEAD_RUN}"
source_run="${2:?Supply the original cached-head run with retention_boundary_audit.json}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."

protocol_dir="$PWD/outputs/static_overlap_final_protocol_seed1"
development_out="$PWD/outputs/static_overlap_development_head_$(date +%Y%m%d_%H%M%S)"
evaluation_bundle="$PWD/data/static_overlap_mcf_seed1_eval.json"
mcf_path="$PWD/data/multi_counterfact.json"

# This prints the existing real-model parity verdict FIRST and aborts if it
# failed, lacks evidence, or no longer matches the source cache/report/bundle.
python -u scripts/freeze_static_overlap_development.py \
  --source-run "$source_run" \
  --parity-audit "$source_run/retention_boundary_audit.json" \
  --evaluation-bundle "$evaluation_bundle" \
  --mcf-path "$mcf_path" \
  --protocol-dir "$protocol_dir" \
  2>&1 | tee "$protocol_dir.freeze.log"

python -u scripts/run_static_overlap_cached_head.py \
  --model-path "$model_path" \
  --training-bundle "$source_run/training_bundle.json" \
  --development-protocol "$protocol_dir/protocol.json" \
  --output-dir "$development_out" \
  --no-context-augmentation \
  --taus 0 0.001 0.01 \
  --device cuda \
  --allow-untied-head \
  --local-files-only \
  2>&1 | tee "$development_out.log"

python -u scripts/evaluate_static_overlap_final_retention.py \
  --protocol "$protocol_dir/protocol.json" \
  --checkpoint "$development_out/checkpoint" \
  --model-path "$model_path" \
  --device cuda \
  --local-files-only \
  2>&1 | tee "$development_out.final_retention.log"

# Both final evaluations use the SAME checkpoint. A preservation failure is
# still reported; it never triggers another fit or a replacement final set.
python -u scripts/evaluate_static_overlap_edit.py \
  --checkpoint "$development_out/checkpoint" \
  --base-model "$model_path" \
  --evaluation-bundle "$evaluation_bundle" \
  --out "$development_out/evaluation_probability_v2.json" \
  --device cuda \
  --max-new-tokens 1 \
  --mcf-path "$mcf_path" \
  --wikidata-dir data/wikidata \
  --seed 1 \
  --unlearn-num 50 \
  --retain-num 1000 \
  --skip-official-ppl \
  2>&1 | tee "$development_out.evaluation.log"

python - "$protocol_dir/final_retention_results.json" "$development_out/evaluation_probability_v2.json" <<'PY'
import json, sys
retention = json.load(open(sys.argv[1]))
official = json.load(open(sys.argv[2]))
passed = retention["retention_passed"] and official["forgetting_check"]["passed"]
print(json.dumps({"joint_final_success": passed,
    "final_preservation_passed": retention["retention_passed"],
    "official_forgetting_check": official["forgetting_check"],
    "official_forget": official["official_mcf"]["forget"],
    "retention_report": sys.argv[1], "official_report": sys.argv[2]}, indent=2))
raise SystemExit(0 if passed else 2)
PY
