#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BASE_MODEL="${BASE_MODEL:-/home/ec2-user/models/Llama-3.2-3B-Instruct}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata_aws_diag}"
SOURCE_ROOT="${SOURCE_ROOT:-outputs/aws_mquake_sure_gagd_output_only_restore_seed1_3b/seed1}"
PROJECTED_ROOT="${PROJECTED_ROOT:-outputs/aws_mquake_sure_gagd_projected_restore_seed1_3b/seed1}"
OUT_ROOT="${PPL_DIAG_ROOT:-${PROJECTED_ROOT}/ppl_diagnostics}"
RESTORE_JSON="${SOURCE_ROOT}/stage1_emb_lm_gagd_output_only_restore/vocabulary_restoration.json"

mkdir -p "${OUT_ROOT}"
test -d "${BASE_MODEL}"
test -f "${RESTORE_JSON}"

run_one() {
  local label="$1"
  local model="$2"
  test -d "$model"
  echo "===== PPL DIAGNOSTIC: ${label} ====="
  python scripts/diagnose_mquake_ppl_sensitive_rows.py \
    --base-model "${BASE_MODEL}" \
    --edited-model "${model}" \
    --restoration-json "${RESTORE_JSON}" \
    --wikidata-dir "${WIKIDATA_DIR}" \
    --out "${OUT_ROOT}/${label}.json" \
    --dtype bf16 --max-input-length 100 --top-k 20
}

run_one output_only_stage1 \
  "${SOURCE_ROOT}/stage1_emb_lm_gagd_output_only_restore/checkpoint"
run_one projected_stage1 \
  "${PROJECTED_ROOT}/stage1_projected_min_norm/checkpoint"
run_one projected_final \
  "${PROJECTED_ROOT}/stage2_sensitive_row_repair/checkpoint"

python - "${OUT_ROOT}" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
print("\n===== PPL DIAGNOSTIC COMPARISON =====")
for label in ("output_only_stage1","projected_stage1","projected_final"):
    x=json.loads((root/f"{label}.json").read_text())
    p=x["ppl_fp32"]["project_official_denominator_N"]
    print(label,
          "FP32_PPL",p["base"],"->",p["edited"],
          "sensitive_targets",x["sensitive_target_positions"],
          "mean_dNLL",x["delta_nll_all"]["mean"],
          "mean_dlogZ",x["delta_logZ_all"]["mean"],
          "mean_dNLL_non_sensitive_targets",x["delta_nll_non_sensitive_targets"]["mean"],
          "added_sensitive_mass/baseZ",x["mean_total_added_sensitive_mass_relative_to_base_Z"])
PY

echo "Detailed JSON: ${OUT_ROOT}"
