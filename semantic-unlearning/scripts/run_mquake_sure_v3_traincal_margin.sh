#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

V3_ROOT="${1:-outputs/mquake_pure_two_stage_directional_v3_3b}"
SEEDS_TEXT="${MQUAKE_TRAINCAL_SEEDS:-1}"
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

read -r -a SEEDS <<< "${SEEDS_TEXT}"

for SEED in "${SEEDS[@]}"; do
  ROOT="${V3_ROOT}/seed${SEED}"
  PROTOCOL="${ROOT}/protocol"
  VISIBLE="${PROTOCOL}/training_visible_forget.json"
  MANIFEST="${PROTOCOL}/split_manifest.json"
  STAGE1="${ROOT}/level1_full_residual_directional_ga/checkpoint"
  OUT="${ROOT}/level2_head_exact_p_nullspace_traincal_margin"
  SUMMARY="${OUT}/two_stage_summary.json"

  test -d "${STAGE1}"
  test -f "${VISIBLE}"
  test -f "${MANIFEST}"
  DIRECT_COUNT="$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["sampling"]["forget_atomic_fact_count"])' "${MANIFEST}")"

  rm -rf "${OUT}"
  mkdir -p "${OUT}"

  echo "===== SURE v3 TRAINING-CALIBRATED STAGE2 MARGIN: seed ${SEED} ====="
  echo "Protection margin: ${PROTECT_MARGIN}"
  echo "Repair margin rule: max(protection margin, median Stage1-success direct margin)"
  echo "NO ATOMICGEN / RETAIN / TARGET_NEW / MULTIHOP EVALUATION IN THIS RUNNER"

  python scripts/mquake_sure_stage2_head_nullspace_v3.py \
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
    --check-every "${REPAIR_CHECK}" \
    --constraint-margin "${PROTECT_MARGIN}" \
    --repair-margin-mode p-median \
    --max-protected-kl "${MAX_PKL}" \
    --l2-weight "${REPAIR_L2}" \
    --backtrack-scales "${BACKTRACK}" \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}"

  python - "${SUMMARY}" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
print("TRAIN-ONLY DIGEST")
print("repair_margin_mode:", p.get("repair_margin_mode"))
print("stage1_P_median_margin:", p.get("stage1_P_median_margin"))
print("repair_margin:", p.get("repair_margin"))
print("final_direct_gate:", p.get("final_gate"))
print("final_F_repair_margin_gate:", p.get("final_F_repair_margin_gate"))
print("stage1_successes_regressed:", p.get("stage1_successes_regressed"))
print("protected_kl:", p.get("protected_kl"))
print("frozen_non_head_exact:", p.get("frozen_non_head_exact"))
print("final_gates_pass:", p.get("final_gates_pass"))
level2 = p.get("level2", {})
print("materialized_delta_norm:", level2.get("materialized_delta_norm"))
print("best_checkpoint_step:", level2.get("best_checkpoint_step"))
PY

done

echo "===== TRAINING-CALIBRATED v3 COMPLETE: HELD-OUT STILL UNTOUCHED BY THIS RUNNER ====="
