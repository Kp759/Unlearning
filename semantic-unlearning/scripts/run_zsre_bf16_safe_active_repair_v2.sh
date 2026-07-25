#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT_ROOT="${1:-${CHECKPOINT_ROOT:-}}"
if [[ -z "${CHECKPOINT_ROOT}" ]]; then
  echo "Usage: $0 /path/to/zsre_setting5e_run_root" >&2
  echo "Expected checkpoints: ROOT/seedN/setting5e/checkpoint" >&2
  exit 2
fi

OUT_ROOT="${OUT_ROOT:-outputs/zsre_setting5e_bf16_safe_repair}"
ZSRE_PATH="${ZSRE_PATH:-data/zsre_mend_eval.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
FORGET_NUM="${FORGET_NUM:-50}"
RETAIN_NUM="${RETAIN_NUM:-1000}"

REPAIR_STEPS="${REPAIR_STEPS:-800}"
REPAIR_LR="${REPAIR_LR:-5e-3}"
ACTIVE_LOGIT_MARGIN="${ACTIVE_LOGIT_MARGIN:-0.25}"
SELECTION_LOGIT_MARGIN="${SELECTION_LOGIT_MARGIN:-0.05}"
REPAIR_RANK="${REPAIR_RANK:-0}"
CANDIDATE_SCALES="${CANDIDATE_SCALES:-1.0,0.875,0.75,0.625,0.5,0.375,0.25,0.1875,0.125,0.09375,0.0625,0.046875,0.03125,0.015625,0.0078125,0.0}"
RETAIN_CALIBRATION_NUM="${RETAIN_CALIBRATION_NUM:-128}"
UTILITY_DROP_TOLERANCE="${UTILITY_DROP_TOLERANCE:-0.10}"
MAX_PPL_RATIO="${MAX_PPL_RATIO:-1.02}"
TARGET_EFF_MAX="${TARGET_EFF_MAX:-0.0}"
TARGET_GEN_MAX="${TARGET_GEN_MAX:-0.0}"

DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-4}"
SKIP_PPL="${SKIP_PPL:-0}"
SAVE_SELECTED="${SAVE_SELECTED:-1}"
FAIL_IF_TARGET_MISSED="${FAIL_IF_TARGET_MISSED:-1}"

read -r -a SEED_ARRAY <<< "${SEEDS}"
RESULT_ARGS=()

for seed in "${SEED_ARRAY[@]}"; do
  checkpoint="${CHECKPOINT_ROOT}/seed${seed}/setting5e/checkpoint"
  if [[ ! -d "${checkpoint}" ]]; then
    echo "Missing Setting 5e checkpoint for seed ${seed}: ${checkpoint}" >&2
    exit 2
  fi

  RUN_ARGS=(
    --setting5-checkpoint "${checkpoint}"
    --output-dir "${OUT_ROOT}/seed${seed}"
    --zsre-path "${ZSRE_PATH}"
    --wikidata-dir "${WIKIDATA_DIR}"
    --seed "${seed}"
    --forget-num "${FORGET_NUM}"
    --retain-num "${RETAIN_NUM}"
    --repair-steps "${REPAIR_STEPS}"
    --repair-lr "${REPAIR_LR}"
    --active-logit-margin "${ACTIVE_LOGIT_MARGIN}"
    --selection-logit-margin "${SELECTION_LOGIT_MARGIN}"
    --repair-rank "${REPAIR_RANK}"
    --candidate-scales "${CANDIDATE_SCALES}"
    --retain-calibration-num "${RETAIN_CALIBRATION_NUM}"
    --utility-drop-tolerance "${UTILITY_DROP_TOLERANCE}"
    --max-ppl-ratio "${MAX_PPL_RATIO}"
    --target-eff-max "${TARGET_EFF_MAX}"
    --target-gen-max "${TARGET_GEN_MAX}"
    --eval-batch-size "${EVAL_BATCH_SIZE}"
    --cache-batch-size "${CACHE_BATCH_SIZE}"
    --dtype "${DTYPE}"
    --device-map "${DEVICE_MAP}"
  )
  if [[ "${SKIP_PPL}" == "1" ]]; then
    RUN_ARGS+=(--skip-ppl)
  fi
  if [[ "${SAVE_SELECTED}" == "0" ]]; then
    RUN_ARGS+=(--no-save-selected-checkpoint)
  fi
  if [[ "${FAIL_IF_TARGET_MISSED}" == "0" ]]; then
    RUN_ARGS+=(--no-fail-if-target-missed)
  fi

  echo "Running exact BF16-safe ZsRE repair for saved seed ${seed}"
  "${PYTHON_BIN}" scripts/zsre_bf16_safe_active_repair_v2.py "${RUN_ARGS[@]}"
  RESULT_ARGS+=(--result "${OUT_ROOT}/seed${seed}/zsre_results.json")
done

"${PYTHON_BIN}" scripts/aggregate_zsre_gagd_results.py \
  "${RESULT_ARGS[@]}" \
  --output-dir "${OUT_ROOT}/aggregate"

echo "Final ZsRE aggregate: ${OUT_ROOT}/aggregate/aggregate.md"
