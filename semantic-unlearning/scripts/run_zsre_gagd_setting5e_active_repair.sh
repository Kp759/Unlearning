#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${1:-${MODEL_PATH:-/scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95}}"
OUT_ROOT="${OUT_ROOT:-outputs/zsre_setting5e_active}"
ZSRE_PATH="${ZSRE_PATH:-data/zsre_mend_eval.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"

# The supplementary paper's launcher evaluates seeds 1-10.
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
FORGET_NUM="${FORGET_NUM:-50}"
RETAIN_NUM="${RETAIN_NUM:-1000}"

# Established ultra-aggressive Setting 5e controls.
STEPS="${STEPS:-600}"
EMB_LM_LR="${EMB_LM_LR:-1e-4}"
FORGET_WEIGHT="${FORGET_WEIGHT:-2.0}"
RETAIN_WEIGHT="${RETAIN_WEIGHT:-1.0}"
FORGET_MARGIN="${FORGET_MARGIN:-1.0}"
RETAIN_BATCH_SIZE="${RETAIN_BATCH_SIZE:-4}"

# ZsRE-native "Unknown" LM-head active repair.
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
SAVE_SETTING5="${SAVE_SETTING5:-0}"
SAVE_SELECTED="${SAVE_SELECTED:-1}"
FAIL_IF_TARGET_MISSED="${FAIL_IF_TARGET_MISSED:-1}"

read -r -a SEED_ARRAY <<< "${SEEDS}"
RESULT_ARGS=()

for seed in "${SEED_ARRAY[@]}"; do
  RUN_ARGS=(
    --model-path "${MODEL_PATH}"
    --output-dir "${OUT_ROOT}/seed${seed}"
    --zsre-path "${ZSRE_PATH}"
    --wikidata-dir "${WIKIDATA_DIR}"
    --seed "${seed}"
    --forget-num "${FORGET_NUM}"
    --retain-num "${RETAIN_NUM}"
    --steps "${STEPS}"
    --batch-size 1
    --retain-batch-size "${RETAIN_BATCH_SIZE}"
    --emb-lm-lr "${EMB_LM_LR}"
    --forget-weight "${FORGET_WEIGHT}"
    --retain-weight "${RETAIN_WEIGHT}"
    --forget-margin "${FORGET_MARGIN}"
    --sampling-strategy epoch
    --post-training-new-true-alpha 0.75
    --post-training-new-retain-alpha 0.50
    --post-training-new-true-retain-alpha 0.25
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
  if [[ "${SAVE_SETTING5}" == "1" ]]; then
    RUN_ARGS+=(--save-setting5-checkpoint)
  fi
  if [[ "${SAVE_SELECTED}" == "0" ]]; then
    RUN_ARGS+=(--no-save-selected-checkpoint)
  fi
  if [[ "${FAIL_IF_TARGET_MISSED}" == "0" ]]; then
    RUN_ARGS+=(--no-fail-if-target-missed)
  fi

  echo "Running ZsRE Setting 5e + active repair for seed ${seed}"
  "${PYTHON_BIN}" scripts/zsre_gagd_setting5e_active_repair.py "${RUN_ARGS[@]}"
  RESULT_ARGS+=(--result "${OUT_ROOT}/seed${seed}/zsre_results.json")
done

"${PYTHON_BIN}" scripts/aggregate_zsre_gagd_results.py \
  "${RESULT_ARGS[@]}" \
  --output-dir "${OUT_ROOT}/aggregate"

echo "Final ZsRE aggregate: ${OUT_ROOT}/aggregate/aggregate.md"
