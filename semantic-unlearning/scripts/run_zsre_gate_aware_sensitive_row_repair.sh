#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT_ROOT="${1:-}"
OUT_ROOT="${2:-}"
if [[ -z "${CHECKPOINT_ROOT}" || -z "${OUT_ROOT}" ]]; then
  echo "Usage: $0 CHECKPOINT_ROOT OUTPUT_ROOT" >&2
  echo "Expected checkpoints: CHECKPOINT_ROOT/seedN/setting5e/checkpoint" >&2
  exit 2
fi

SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
ZSRE_PATH="${ZSRE_PATH:-data/zsre_mend_eval.json}"
ZSRE_URL="${ZSRE_URL:-https://rome.baulab.info/data/dsets/zsre_mend_eval.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
FORGET_NUM="${FORGET_NUM:-50}"
RETAIN_NUM="${RETAIN_NUM:-1000}"

REPAIR_STEPS="${REPAIR_STEPS:-3000}"
REPAIR_LR="${REPAIR_LR:-1e-3}"
REPAIR_OPTIMIZER="${REPAIR_OPTIMIZER:-adamw}"
ACTIVE_MARGIN="${ACTIVE_MARGIN:-0.02}"
PROTECTED_MARGIN_CAP="${PROTECTED_MARGIN_CAP:-0.05}"
ACTIVE_HINGE_WEIGHT="${ACTIVE_HINGE_WEIGHT:-2.0}"
PROTECTED_HINGE_WEIGHT="${PROTECTED_HINGE_WEIGHT:-50.0}"
RETAIN_KL_MU="${RETAIN_KL_MU:-10.0}"
DELTA_L2_LAMBDA="${DELTA_L2_LAMBDA:-1e-4}"
RETAIN_CALIBRATION_NUM="${RETAIN_CALIBRATION_NUM:-1000}"
RETAIN_CALIBRATION_SEED="${RETAIN_CALIBRATION_SEED:-1729}"
PROTECTED_BATCH_SIZE="${PROTECTED_BATCH_SIZE:-256}"
RETAIN_KL_BATCH_SIZE="${RETAIN_KL_BATCH_SIZE:-32}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
FULL_CONSTRAINT_CHECK_EVERY="${FULL_CONSTRAINT_CHECK_EVERY:-100}"
STOP_WHEN_ALL_SATISFIED="${STOP_WHEN_ALL_SATISFIED:-0}"
REPAIR_RANK="${REPAIR_RANK:-0}"
EDIT_UNKNOWN_ROW="${EDIT_UNKNOWN_ROW:-0}"

CANDIDATE_SCALE_STEP="${CANDIDATE_SCALE_STEP:-0.025}"
UTILITY_DROP_TOLERANCE="${UTILITY_DROP_TOLERANCE:-0.10}"
MAX_PPL_RATIO="${MAX_PPL_RATIO:-1.02}"
TARGET_EFF_MAX="${TARGET_EFF_MAX:-0.0}"
TARGET_GEN_MAX="${TARGET_GEN_MAX:-0.0}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-8}"
FAIL_IF_ANY_SEED_REJECTED="${FAIL_IF_ANY_SEED_REJECTED:-1}"

read -r -a SEED_ARRAY <<< "${SEEDS}"
RESULT_ARGS=()
FAILED_SEEDS=()

for seed in "${SEED_ARRAY[@]}"; do
  checkpoint="${CHECKPOINT_ROOT}/seed${seed}/setting5e/checkpoint"
  source_results="${CHECKPOINT_ROOT}/seed${seed}/zsre_results.json"
  seed_output="${OUT_ROOT}/seed${seed}"
  if [[ ! -d "${checkpoint}" ]]; then
    echo "Missing Setting 5e checkpoint for seed ${seed}: ${checkpoint}" >&2
    FAILED_SEEDS+=("${seed}:missing_checkpoint")
    continue
  fi
  if [[ ! -f "${source_results}" ]]; then
    echo "Missing source result for seed ${seed}: ${source_results}" >&2
    FAILED_SEEDS+=("${seed}:missing_source_result")
    continue
  fi

  RUN_ARGS=(
    --setting5-checkpoint "${checkpoint}"
    --source-results "${source_results}"
    --output-dir "${seed_output}"
    --zsre-path "${ZSRE_PATH}"
    --zsre-url "${ZSRE_URL}"
    --wikidata-dir "${WIKIDATA_DIR}"
    --seed "${seed}"
    --forget-num "${FORGET_NUM}"
    --retain-num "${RETAIN_NUM}"
    --repair-steps "${REPAIR_STEPS}"
    --repair-lr "${REPAIR_LR}"
    --repair-optimizer "${REPAIR_OPTIMIZER}"
    --active-margin "${ACTIVE_MARGIN}"
    --protected-margin-cap "${PROTECTED_MARGIN_CAP}"
    --active-hinge-weight "${ACTIVE_HINGE_WEIGHT}"
    --protected-hinge-weight "${PROTECTED_HINGE_WEIGHT}"
    --retain-kl-mu "${RETAIN_KL_MU}"
    --delta-l2-lambda "${DELTA_L2_LAMBDA}"
    --retain-calibration-num "${RETAIN_CALIBRATION_NUM}"
    --retain-calibration-seed "${RETAIN_CALIBRATION_SEED}"
    --protected-batch-size "${PROTECTED_BATCH_SIZE}"
    --retain-kl-batch-size "${RETAIN_KL_BATCH_SIZE}"
    --progress-every "${PROGRESS_EVERY}"
    --full-constraint-check-every "${FULL_CONSTRAINT_CHECK_EVERY}"
    --repair-rank "${REPAIR_RANK}"
    --candidate-scale-step "${CANDIDATE_SCALE_STEP}"
    --utility-drop-tolerance "${UTILITY_DROP_TOLERANCE}"
    --max-ppl-ratio "${MAX_PPL_RATIO}"
    --target-eff-max "${TARGET_EFF_MAX}"
    --target-gen-max "${TARGET_GEN_MAX}"
    --eval-batch-size "${EVAL_BATCH_SIZE}"
    --cache-batch-size "${CACHE_BATCH_SIZE}"
    --dtype "${DTYPE}"
    --device-map "${DEVICE_MAP}"
    --no-fail-if-target-missed
  )
  if [[ "${STOP_WHEN_ALL_SATISFIED}" == "1" ]]; then
    RUN_ARGS+=(--stop-when-all-satisfied)
  else
    RUN_ARGS+=(--no-stop-when-all-satisfied)
  fi
  if [[ "${EDIT_UNKNOWN_ROW}" == "1" ]]; then
    RUN_ARGS+=(--edit-unknown-row)
  else
    RUN_ARGS+=(--no-edit-unknown-row)
  fi

  echo "Running gate-aware sensitive-row ZsRE repair for seed ${seed}"
  if "${PYTHON_BIN}" scripts/zsre_gate_aware_sensitive_row_repair.py "${RUN_ARGS[@]}"; then
    :
  else
    run_status=$?
    if [[ "${run_status}" -eq 130 ]]; then
      echo "Interrupted during seed ${seed}; stopping without aggregation." >&2
      exit 130
    fi
    FAILED_SEEDS+=("${seed}:execution_error")
    continue
  fi
  result="${seed_output}/zsre_results.json"
  if ! "${PYTHON_BIN}" -c \
    'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["repair"]["candidate_accepted"] else 1)' \
    "${result}"; then
    FAILED_SEEDS+=("${seed}:rejected")
    continue
  fi
  RESULT_ARGS+=(--result "${result}")
done

if (( ${#FAILED_SEEDS[@]} > 0 )); then
  echo "Rejected or failed seeds: ${FAILED_SEEDS[*]}" >&2
  echo "No aggregate was emitted because fallback-selected rows are forbidden." >&2
  if [[ "${FAIL_IF_ANY_SEED_REJECTED}" == "1" ]]; then
    exit 1
  fi
  exit 0
fi

"${PYTHON_BIN}" scripts/aggregate_zsre_gagd_results.py \
  "${RESULT_ARGS[@]}" \
  --output-dir "${OUT_ROOT}/aggregate"

echo "Gate-aware ZsRE aggregate: ${OUT_ROOT}/aggregate/aggregate.md"
