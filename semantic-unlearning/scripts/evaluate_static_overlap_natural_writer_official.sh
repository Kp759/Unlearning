#!/usr/bin/env bash
set -euo pipefail

run_dir="${1:?Usage: bash scripts/evaluate_static_overlap_natural_writer_official.sh RUN_DIR MCF_JSON WIKIDATA_DIR [OUT_JSON]}"
mcf_path="${2:?Provide MultiCounterFact JSON path}"
wikidata_dir="${3:?Provide Wikidata dataset directory}"
out="${4:-$run_dir/official_mcf_eval.json}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

checkpoint="$run_dir/checkpoint"
audit_out="$run_dir/bf16_natural_prompt_audit.json"
test -d "$checkpoint" || {
  echo "Missing native checkpoint: $checkpoint" >&2
  exit 2
}
if [[ -e "$out" ]]; then
  echo "Official evaluation output already exists: $out; refusing to overwrite it." >&2
  exit 2
fi

if [[ ! -e "$audit_out" ]]; then
  python -u scripts/audit_static_overlap_natural_writer_bf16.py \
    --run-dir "$run_dir" \
    --device cuda \
    --out "$audit_out"
fi

python -u scripts/mcf_zero_unlearn_official_eval.py \
  --model-dir "$checkpoint" \
  --mcf-path "$mcf_path" \
  --wikidata-dir "$wikidata_dir" \
  --out "$out" \
  --unlearn-num 50 \
  --retain-num 1000 \
  --seed 1 \
  --sample-mode official \
  --dtype bfloat16 \
  --device-map single
