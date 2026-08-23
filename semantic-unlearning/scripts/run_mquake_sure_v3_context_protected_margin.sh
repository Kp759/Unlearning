#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

V3_ROOT="${1:-outputs/mquake_pure_two_stage_directional_v3_3b}"
SEEDS_TEXT="${MQUAKE_CONTEXT_V3_SEEDS:-2}"
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
CONTEXT_TOKENS="${SURE_STAGE1_PROTECTED_CONTEXT_TOKENS:-4}"
BACKTRACK="${SURE_STAGE2_BACKTRACK_SCALES:-0.5,0.25,0.125,0.0625,0.03125,0.015625}"
# Disabled by default for the context-protected variant: if a numerically clean
# pre-materialization solution later needs BF16 recovery, use a context-aware
# recovery rather than blindly scaling a failed float-space solution.
AUTO_RECOVER="${SURE_CONTEXT_V3_AUTO_RECOVER:-0}"

read -r -a SEEDS <<< "${SEEDS_TEXT}"

for SEED in "${SEEDS[@]}"; do
  ROOT="${V3_ROOT}/seed${SEED}"
  PROTOCOL="${ROOT}/protocol"
  VISIBLE="${PROTOCOL}/training_visible_forget.json"
  MANIFEST="${PROTOCOL}/split_manifest.json"
  STAGE1="${ROOT}/level1_full_residual_directional_ga/checkpoint"
  OUT="${ROOT}/level2_context_protected_traincal_margin_v3"
  SUMMARY="${OUT}/two_stage_summary.json"

  test -d "${STAGE1}"
  test -f "${VISIBLE}"
  test -f "${MANIFEST}"
  DIRECT_COUNT="$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["sampling"]["forget_atomic_fact_count"])' "${MANIFEST}")"

  rm -rf "${OUT}"
  mkdir -p "${OUT}"

  echo "===== SURE v3 CONTEXT-PROTECTED TRAIN-CAL STAGE2: seed ${SEED} ====="
  echo "Protection: full rowspace(Stage1-success H_P + ${CONTEXT_TOKENS} preceding H_NS tokens/case)"
  echo "Numerics: float64 protected SVD rtol=1e-10; repair SVD rtol=1e-8; project+QR cleanup"
  echo "Repair margin: max(0.05, median Stage1-success direct margin)"
  echo "NO ATOMICGEN / RETAIN / TARGET_NEW / MULTIHOP USED"

  python scripts/mquake_sure_stage2_context_protected_v31_stable.py \
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
    --protected-context-tokens "${CONTEXT_TOKENS}" \
    --constraint-margin "${PROTECT_MARGIN}" \
    --max-protected-kl "${MAX_PKL}" \
    --l2-weight "${REPAIR_L2}" \
    --check-every "${REPAIR_CHECK}" \
    --backtrack-scales "${BACKTRACK}" \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}"

  FINAL_PASS="$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print("1" if d.get("final_gates_pass") else "0")' "${SUMMARY}")"
  PRE_PASS="$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); f=d.get("level2",{}).get("best_pre_materialization_F",{}); print("1" if int(f.get("failed",999999))==0 else "0")' "${SUMMARY}")"

  if [[ "${FINAL_PASS}" != "1" && "${PRE_PASS}" == "1" && "${AUTO_RECOVER}" == "1" ]]; then
    echo "===== FLOAT-SPACE PASSED BUT MATERIALIZATION FAILED: OPTIONAL LOW-PRECISION RECOVERY ====="
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
  elif [[ "${FINAL_PASS}" != "1" && "${PRE_PASS}" != "1" ]]; then
    echo "===== RECOVERY SKIPPED: FLOAT-SPACE REPAIR ITSELF DID NOT PASS ====="
  elif [[ "${FINAL_PASS}" != "1" ]]; then
    echo "===== MATERIALIZATION FAILED AFTER FLOAT-SPACE PASS; AUTO RECOVERY DISABLED ====="
  fi

  python - "${SUMMARY}" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
print("===== CONTEXT-PROTECTED FINAL TRAINING DIGEST =====")
print("repair_margin:", p.get("repair_margin"))
l2 = p.get("level2", {})
print("H_NS_rows:", l2.get("H_NS_rows"))
print("combined_guard_basis_rank:", l2.get("combined_guard_basis_rank"))
print("repair_basis_rank:", l2.get("repair_basis_rank"))
print("residual_hidden_energy_fraction:", l2.get("residual_hidden_energy_fraction"))
print("P_nullspace_leak:", l2.get("P_nullspace_leak"))
print("NS_nullspace_leak:", l2.get("NS_nullspace_leak"))
print("best_pre_materialization_F:", l2.get("best_pre_materialization_F"))
print("materialized_delta_norm:", l2.get("materialized_delta_norm"))
print("recovered_materialized_delta_norm:", l2.get("recovered_materialized_delta_norm"))
print("materialization_recovery_scale:", l2.get("materialization_recovery_scale"))
print("training_context_selected_logit_drift_max:", p.get("training_context_selected_logit_drift_max"))
print("final_direct_gate:", p.get("final_gate"))
print("final_F_repair_margin_gate:", p.get("final_F_repair_margin_gate"))
print("stage1_successes_regressed:", p.get("stage1_successes_regressed"))
print("protected_kl:", p.get("protected_kl"))
print("frozen_non_head_exact:", p.get("frozen_non_head_exact"))
print("final_gates_pass:", p.get("final_gates_pass"))
print("checkpoint:", p.get("checkpoint"))
PY

done

echo "===== CONTEXT-PROTECTED v3 TRAINING COMPLETE ====="
