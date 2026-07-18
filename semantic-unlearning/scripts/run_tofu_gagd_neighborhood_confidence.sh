#!/usr/bin/env bash
set -euo pipefail

# End-to-end TOFU comparison:
#   1. train the four existing GA/GD settings;
#   2. repair one selected checkpoint with TOFU neighborhood confidence;
#   3. run the same full tofu_eval.py protocol on Base + all five methods;
#   4. write fixed-order CSV, Markdown, and JSON comparisons.
#
# Every path and hyperparameter can be overridden through environment
# variables. By default this reproduces the repository's 3B forget05 setup.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

MODEL_PATH="${MODEL_PATH:-outputs/finetuned_model_3B_instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/tofu_gagd_neighborhood_confidence}"
CONFIG_PATH="${CONFIG_PATH:-config/tofu_gagd_neighborhood_confidence.yaml}"
FORGET_SPLIT="${FORGET_SPLIT:-forget05}"
RETAIN_SPLIT="${RETAIN_SPLIT:-retain95}"
SEED="${SEED:-42}"
FORGET_NUM="${FORGET_NUM:-200}"
RETAIN_NUM="${RETAIN_NUM:-1000}"
STEPS="${STEPS:-100}"
BATCH_SIZE="${BATCH_SIZE:-1}"
RETAIN_BATCH_SIZE="${RETAIN_BATCH_SIZE:-1}"
LR="${LR:-1e-5}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
REPAIR_INPUT_MODE="${REPAIR_INPUT_MODE:-emb_lm_all_tokens}"

RUN_FOUR_SETTINGS="${RUN_FOUR_SETTINGS:-1}"
RUN_REPAIR="${RUN_REPAIR:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

FOUR_DIR="${OUTPUT_ROOT}/four_settings"
REPAIR_DIR="${OUTPUT_ROOT}/gagd_neighborhood_confidence_tofu"
EVAL_DIR="${OUTPUT_ROOT}/evaluation"

case "${REPAIR_INPUT_MODE}" in
  full_all_tokens|full_selective_tokens|emb_lm_all_tokens|emb_lm_selective_tokens)
    ;;
  *)
    echo "REPAIR_INPUT_MODE must name one of the four GA/GD settings" >&2
    exit 2
    ;;
esac

if [[ "${RUN_FOUR_SETTINGS}" == "1" ]]; then
  python scripts/gagd_compare.py \
    --dataset tofu \
    --model-path "${MODEL_PATH}" \
    --output-dir "${FOUR_DIR}" \
    --mode all \
    --forget-split "${FORGET_SPLIT}" \
    --retain-split "${RETAIN_SPLIT}" \
    --seed "${SEED}" \
    --forget-num "${FORGET_NUM}" \
    --retain-num "${RETAIN_NUM}" \
    --steps "${STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --retain-batch-size "${RETAIN_BATCH_SIZE}" \
    --lr "${LR}" \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}" \
    --save-model
fi

REPAIR_INPUT_CHECKPOINT="${FOUR_DIR}/${REPAIR_INPUT_MODE}/checkpoint"
if [[ ! -d "${REPAIR_INPUT_CHECKPOINT}" ]]; then
  echo "Missing repair input checkpoint: ${REPAIR_INPUT_CHECKPOINT}" >&2
  exit 2
fi

if [[ "${RUN_REPAIR}" == "1" ]]; then
  python scripts/tofu_gagd_neighborhood_confidence.py \
    --model-path "${REPAIR_INPUT_CHECKPOINT}" \
    --reference-model-path "${MODEL_PATH}" \
    --output-dir "${REPAIR_DIR}" \
    --forget-split "${FORGET_SPLIT}" \
    --retain-split "${RETAIN_SPLIT}" \
    --seed "${SEED}" \
    --forget-num "${FORGET_NUM}" \
    --retain-num "${RETAIN_NUM}" \
    --retain-calibration-num "${RETAIN_CALIBRATION_NUM:-64}" \
    --real-authors-calibration-num "${REAL_AUTHORS_CALIBRATION_NUM:-32}" \
    --world-facts-calibration-num "${WORLD_FACTS_CALIBRATION_NUM:-32}" \
    --calibration-seed "${CALIBRATION_SEED:-1729}" \
    --repair-steps "${REPAIR_STEPS:-200}" \
    --repair-lr "${REPAIR_LR:-5e-3}" \
    --repair-rank "${REPAIR_RANK:-32}" \
    --row-selection-top-k "${ROW_SELECTION_TOP_K:-512}" \
    --forget-projection-rank "${FORGET_PROJECTION_RANK:-64}" \
    --max-length "${REPAIR_MAX_LENGTH:-256}" \
    --batch-size "${REPAIR_BATCH_SIZE:-8}" \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}" \
    --save-model
fi

REPAIR_CHECKPOINT="${REPAIR_DIR}/checkpoint"
if [[ ! -d "${REPAIR_CHECKPOINT}" ]]; then
  echo "Missing repaired checkpoint: ${REPAIR_CHECKPOINT}" >&2
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
    --output-dir "${EVAL_DIR}"
    --max-new-tokens "${MAX_NEW_TOKENS:-64}"
  )
  if [[ "${base_flag}" == "1" ]]; then
    cmd+=(--base-model)
  fi
  if [[ -n "${N_FORGET_EVAL:-}" ]]; then
    cmd+=(--n-forget-eval "${N_FORGET_EVAL}")
  fi
  if [[ -n "${N_RETAIN_EVAL:-}" ]]; then
    cmd+=(--n-retain-eval "${N_RETAIN_EVAL}")
  fi
  if [[ -n "${N_REAL_AUTHORS_EVAL:-}" ]]; then
    cmd+=(--n-real-authors-eval "${N_REAL_AUTHORS_EVAL}")
  fi
  if [[ -n "${N_WORLD_FACTS_EVAL:-}" ]]; then
    cmd+=(--n-world-facts-eval "${N_WORLD_FACTS_EVAL}")
  fi
  if [[ -n "${N_PERTURBED_EVAL:-}" ]]; then
    cmd+=(--n-perturbed-eval "${N_PERTURBED_EVAL}")
  fi
  if [[ -n "${TOFU_REFERENCE_TRUTH_RATIOS:-}" ]]; then
    cmd+=(--reference-truth-ratios "${TOFU_REFERENCE_TRUTH_RATIOS}")
  elif [[ -n "${TOFU_REFERENCE_MODEL_DIR:-}" ]]; then
    cmd+=(--reference-model-dir "${TOFU_REFERENCE_MODEL_DIR}")
  fi
  "${cmd[@]}"
}

if [[ "${RUN_EVAL}" == "1" ]]; then
  run_tofu_eval "base" "${MODEL_PATH}" 1
  run_tofu_eval "full_all_tokens" "${FOUR_DIR}/full_all_tokens/checkpoint"
  run_tofu_eval "full_selective_tokens" "${FOUR_DIR}/full_selective_tokens/checkpoint"
  run_tofu_eval "emb_lm_all_tokens" "${FOUR_DIR}/emb_lm_all_tokens/checkpoint"
  run_tofu_eval "emb_lm_selective_tokens" "${FOUR_DIR}/emb_lm_selective_tokens/checkpoint"
  run_tofu_eval "gagd_neighborhood_confidence_tofu" "${REPAIR_CHECKPOINT}"
fi

python scripts/tofu_gagd_results.py \
  --output-dir "${OUTPUT_ROOT}/comparison" \
  --result "base=${EVAL_DIR}/base_summary.json" \
  --result "full_all_tokens=${EVAL_DIR}/full_all_tokens_summary.json" \
  --result "full_selective_tokens=${EVAL_DIR}/full_selective_tokens_summary.json" \
  --result "emb_lm_all_tokens=${EVAL_DIR}/emb_lm_all_tokens_summary.json" \
  --result "emb_lm_selective_tokens=${EVAL_DIR}/emb_lm_selective_tokens_summary.json" \
  --result "gagd_neighborhood_confidence_tofu=${EVAL_DIR}/gagd_neighborhood_confidence_tofu_summary.json"

echo "TOFU comparison complete: ${OUTPUT_ROOT}/comparison/comparison_tofu.md"
