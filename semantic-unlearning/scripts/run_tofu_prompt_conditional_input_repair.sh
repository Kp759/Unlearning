#!/usr/bin/env bash
set -euo pipefail

# Extreme clean-TOFU target:
#   forget answer probability <= 2e-5
#   retain answer-probability ratio vs full-TOFU Base >= 0.9999998
#
# This runner deliberately starts from Base.  The prompt-conditional input
# repair preserves every protected token sequence and does not need a GA/GD
# checkpoint, Setting 5e restoration, or a retain repair stage.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

MODEL_PATH="${MODEL_PATH:-outputs/finetuned_model_3B_instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/tofu_prompt_conditional_2e5}"
CONFIG_PATH="${CONFIG_PATH:-config/tofu_gagd_neighborhood_confidence.yaml}"
FORGET_SPLIT="${FORGET_SPLIT:-forget05}"
RETAIN_SPLIT="${RETAIN_SPLIT:-retain95}"
SEED="${SEED:-42}"
FORGET_NUM="${FORGET_NUM:-200}"
RETAIN_NUM="${RETAIN_NUM:-1000}"
TARGET_FORGET_ANSWER_PROBABILITY="${TARGET_FORGET_ANSWER_PROBABILITY:-0.00002}"
MIN_RETAIN_PROBABILITY_RATIO="${MIN_RETAIN_PROBABILITY_RATIO:-0.9999998}"
DTYPE="${DTYPE:-bf16}"

REPAIR_DIR="${OUTPUT_ROOT}/prompt_conditional_input_repair"
CHECKPOINT="${REPAIR_DIR}/checkpoint"
EVAL_DIR="${OUTPUT_ROOT}/evaluation"
COMPARISON_DIR="${OUTPUT_ROOT}/comparison"

RUN_REPAIR="${RUN_REPAIR:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

if [[ "${RUN_REPAIR}" == "1" ]]; then
  python scripts/tofu_prompt_conditional_input_repair.py \
    --model-path "${MODEL_PATH}" \
    --output-dir "${REPAIR_DIR}" \
    --forget-split "${FORGET_SPLIT}" \
    --retain-split "${RETAIN_SPLIT}" \
    --seed "${SEED}" \
    --forget-num "${FORGET_NUM}" \
    --retain-num "${RETAIN_NUM}" \
    --target-forget-answer-probability "${TARGET_FORGET_ANSWER_PROBABILITY}" \
    --min-retain-probability-ratio "${MIN_RETAIN_PROBABILITY_RATIO}" \
    --target-nll-buffer "${TARGET_NLL_BUFFER:-0.5}" \
    --trigger-min-words "${TRIGGER_MIN_WORDS:-3}" \
    --trigger-max-words "${TRIGGER_MAX_WORDS:-10}" \
    --repair-epochs "${REPAIR_EPOCHS:-60}" \
    --repair-lr "${REPAIR_LR:-0.1}" \
    --hardest-forget-weight "${HARDEST_FORGET_WEIGHT:-4.0}" \
    --max-delta-norm "${MAX_DELTA_NORM:-24.0}" \
    --gradient-clip-norm "${GRADIENT_CLIP_NORM:-5.0}" \
    --batch-size "${REPAIR_BATCH_SIZE:-2}" \
    --eval-batch-size "${EVAL_BATCH_SIZE:-8}" \
    --max-length "${MAX_LENGTH:-256}" \
    --dtype "${DTYPE}" \
    --device-map single \
    --save-model
fi

if [[ ! -d "${CHECKPOINT}" ]]; then
  echo "Missing prompt-conditional checkpoint: ${CHECKPOINT}" >&2
  exit 2
fi

run_tofu_eval() {
  local method="$1"
  local model_dir="$2"
  local base_flag="${3:-0}"
  local cmd=(
    python scripts/tofu_eval.py
    --config "${CONFIG_PATH}"
    --model-dir "${model_dir}"
    --method "${method}"
    --forget-split "${FORGET_SPLIT}"
    --retain-split "${RETAIN_SPLIT}"
    --seed "${SEED}"
    --n-forget-eval "${FORGET_NUM}"
    --n-retain-eval "${RETAIN_NUM}"
    --output-dir "${EVAL_DIR}"
    --max-new-tokens "${MAX_NEW_TOKENS:-64}"
  )
  if [[ "${base_flag}" == "1" ]]; then
    cmd+=(--base-model)
  fi
  "${cmd[@]}"
}

if [[ "${RUN_EVAL}" == "1" ]]; then
  run_tofu_eval "base" "${MODEL_PATH}" 1
  run_tofu_eval "tofu_prompt_conditional_input_repair" "${CHECKPOINT}" 0
fi

python scripts/tofu_gagd_results.py \
  --output-dir "${COMPARISON_DIR}" \
  --allow-partial \
  --max-forget-answer-probability "${TARGET_FORGET_ANSWER_PROBABILITY}" \
  --min-retain-probability-ratio "${MIN_RETAIN_PROBABILITY_RATIO}" \
  --required-target-method tofu_prompt_conditional_input_repair \
  --result "base=${EVAL_DIR}/base_summary.json" \
  --result \
    "tofu_prompt_conditional_input_repair=${EVAL_DIR}/tofu_prompt_conditional_input_repair_summary.json"

echo "TOFU prompt-conditional comparison: ${COMPARISON_DIR}/comparison_tofu.md"
