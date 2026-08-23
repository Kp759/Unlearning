#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

V3_ROOT="${1:-outputs/mquake_pure_two_stage_directional_v3_3b}"
SEEDS_TEXT="${MQUAKE_ROWUTIL_V3_SEEDS:-2}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
CACHE_BATCH="${SURE_CACHE_BATCH_SIZE:-8}"
PROTECT_MARGIN="${MQUAKE_SURE_CONSTRAINT_MARGIN:-0.05}"
MAX_PKL="${SURE_MAX_PROTECTED_KL:-0.05}"
REPAIR_STEPS="${SURE_STAGE2_STEPS:-800}"
REPAIR_LR="${SURE_STAGE2_LR:-0.0005}"
REPAIR_BATCH="${SURE_STAGE2_BATCH_SIZE:-8}"
REPAIR_CHECK="${SURE_STAGE2_CHECK_EVERY:-25}"
REPAIR_L2="${SURE_STAGE2_L2_WEIGHT:-0.000001}"
BACKTRACK="${SURE_STAGE2_BACKTRACK_SCALES:-0.5,0.25,0.125,0.0625,0.03125,0.015625}"
AUTO_RECOVER="${SURE_ROWUTIL_V3_AUTO_RECOVER:-1}"

read -r -a SEEDS <<< "${SEEDS_TEXT}"

for SEED in "${SEEDS[@]}"; do
  ROOT="${V3_ROOT}/seed${SEED}"
  PROTOCOL="${ROOT}/protocol"
  VISIBLE="${PROTOCOL}/training_visible_forget.json"
  MANIFEST="${PROTOCOL}/split_manifest.json"
  STAGE1="${ROOT}/level1_full_residual_directional_ga/checkpoint"
  OUT="${ROOT}/level2_rowconditioned_utility_traincal_margin_v3"
  SUMMARY="${OUT}/two_stage_summary.json"

  test -d "${STAGE1}"
  test -f "${VISIBLE}"
  test -f "${MANIFEST}"
  DIRECT_COUNT="$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["sampling"]["forget_atomic_fact_count"])' "${MANIFEST}")"

  rm -rf "${OUT}"
  mkdir -p "${OUT}"

  echo "===== SURE v3 ROW-CONDITIONED UTILITY-PROTECTED STAGE2: seed ${SEED} ====="
  echo "Common protection: full Stage1-success H_P rowspace"
  echo "Row-specific protection: source-prompt states whose observed next token equals that edited row"
  echo "Repair margin: max(0.05, median Stage1-success direct margin)"
  echo "NO ATOMICGEN / RETAIN / TARGET_NEW / MULTIHOP USED"

  python scripts/mquake_sure_stage2_rowconditioned_utility_v3.py \
    --model-path "${STAGE1}" \
    --training-visible-path "${VISIBLE}" \
    --split-manifest "${MANIFEST}" \
    --output-dir "${OUT}" \
    --seed "${SEED}" \
    --forget-num "${DIRECT_COUNT}" \
    --repair-steps "${REPAIR_STEPS}" \
    --repair-lr "${REPAIR_LR}" \
    --batch-size "${REPAIR_BATCH}" \
    --cache-batch-size "${CACHE_BATCH}" \
    --constraint-margin "${PROTECT_MARGIN}" \
    --max-protected-kl "${MAX_PKL}" \
    --l2-weight "${REPAIR_L2}" \
    --check-every "${REPAIR_CHECK}" \
    --backtrack-scales "${BACKTRACK}" \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}"

  FINAL_PASS="$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print("1" if d.get("final_gates_pass") else "0")' "${SUMMARY}")"
  PRE_FAIL="$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(int(d.get("level2",{}).get("best_pre_materialization_F",{}).get("failed",999999)))' "${SUMMARY}")"

  if [[ "${FINAL_PASS}" != "1" && "${PRE_FAIL}" == "0" && "${AUTO_RECOVER}" == "1" ]]; then
    echo "===== FLOAT-SPACE PASSED; RUN DETERMINISTIC LOW-PRECISION RECOVERY ====="
    python scripts/mquake_sure_v3_bf16_materialization_recover.py \
      --stage1-model-path "${STAGE1}" \
      --stage2-state "${OUT}/stage2_nullspace_state.pt" \
      --stage2-summary "${SUMMARY}" \
      --training-visible-path "${VISIBLE}" \
      --split-manifest "${MANIFEST}" \
      --output-dir "${OUT}" \
      --seed "${SEED}" \
      --forget-num "${DIRECT_COUNT}" \
      --constraint-margin "${PROTECT_MARGIN}" \
      --max-protected-kl "${MAX_PKL}" \
      --cache-batch-size "${CACHE_BATCH}" \
      --dtype "${DTYPE}" \
      --device-map "${DEVICE_MAP}"
  elif [[ "${FINAL_PASS}" != "1" && "${PRE_FAIL}" != "0" ]]; then
    echo "===== RECOVERY SKIPPED: FLOAT-SPACE REPAIR ITSELF DID NOT PASS ====="
  fi

  python - "${SUMMARY}" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
l2 = p.get("level2", {})
print("===== ROW-CONDITIONED UTILITY FINAL TRAINING DIGEST =====")
print("repair_margin:", p.get("repair_margin"))
print("common_P_basis_rank:", l2.get("common_P_basis_rank"))
print("source_prompt_use_state_count:", l2.get("source_prompt_use_state_count"))
print("edited_rows_with_observed_prompt_uses:", l2.get("edited_rows_with_observed_prompt_uses"), "/", l2.get("A_F_count"))
print("matched_prompt_use_count:", l2.get("matched_prompt_use_count"))
print("row_use_logit_drift_pre_materialization_max:", l2.get("row_use_logit_drift_pre_materialization_max"))
print("row_use_logit_drift_materialized_max:", l2.get("row_use_logit_drift_materialized_max"))
print("best_pre_materialization_F:", l2.get("best_pre_materialization_F"))
print("materialized_delta_norm:", l2.get("materialized_delta_norm"))
print("recovered_materialized_delta_norm:", l2.get("recovered_materialized_delta_norm"))
print("materialization_recovery_scale:", l2.get("materialization_recovery_scale"))
print("final_direct_gate:", p.get("final_gate"))
print("final_F_repair_margin_gate:", p.get("final_F_repair_margin_gate"))
print("stage1_successes_regressed:", p.get("stage1_successes_regressed"))
print("protected_kl:", p.get("protected_kl"))
print("frozen_non_head_exact:", p.get("frozen_non_head_exact"))
print("final_gates_pass:", p.get("final_gates_pass"))
print("checkpoint:", p.get("checkpoint"))
PY

done

echo "===== ROW-CONDITIONED UTILITY v3 TRAINING COMPLETE ====="
