#!/usr/bin/env bash
set -euo pipefail

# Complete TOFU chain:
#   1. four GA/GD parameter/token settings;
#   2. TOFU Setting 5e overlap-aware post-training restoration;
#   3. active forget-case sparse LM-head repair to <= 3e-4;
#   4. optional neighborhood-confidence utility repair;
#   5. a true retain-only retraining oracle;
#   6. one fixed-protocol TOFU table with hard forget and retain gates.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

MODEL_PATH="${MODEL_PATH:-outputs/finetuned_model_3B_instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/tofu_gagd_targeted_3e4}"
CONFIG_PATH="${CONFIG_PATH:-config/tofu_gagd_neighborhood_confidence.yaml}"
FORGET_SPLIT="${FORGET_SPLIT:-forget05}"
RETAIN_SPLIT="${RETAIN_SPLIT:-retain95}"
SEED="${SEED:-42}"
FORGET_NUM="${FORGET_NUM:-200}"
RETAIN_NUM="${RETAIN_NUM:-1000}"
TARGET_FORGET_ANSWER_PROBABILITY="${TARGET_FORGET_ANSWER_PROBABILITY:-0.0003}"
MIN_RETAIN_PROBABILITY_RATIO="${MIN_RETAIN_PROBABILITY_RATIO:-0.9998}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"

# One protocol-matched pass: 200 forget records at batch 1 and all 1,000
# sampled retain records at batch 5. The retain coefficient compensates for
# averaging over five examples. Sparse active repair enforces the hard target.
STEPS="${STEPS:-200}"
BATCH_SIZE="${BATCH_SIZE:-1}"
RETAIN_BATCH_SIZE="${RETAIN_BATCH_SIZE:-5}"
LR="${LR:-1e-5}"
FULL_LR="${FULL_LR:-1e-5}"
EMB_LM_LR="${EMB_LM_LR:-2e-4}"
FORGET_WEIGHT="${FORGET_WEIGHT:-1.0}"
RETAIN_WEIGHT="${RETAIN_WEIGHT:-5.0}"

SETTING5_SOURCE_MODE="${SETTING5_SOURCE_MODE:-emb_lm_all_tokens}"
SETTING5_UNIQUE_FORGET_ALPHA="${SETTING5_UNIQUE_FORGET_ALPHA:-1.0}"
# Shared forget/utility rows are restored completely. The active repair that
# follows can suppress utility-exclusive forget rows without spending the
# 99.98%-of-Base retain budget at this intermediate stage.
SETTING5_OVERLAP_ALPHA="${SETTING5_OVERLAP_ALPHA:-0.0}"

RUN_FOUR_SETTINGS="${RUN_FOUR_SETTINGS:-1}"
RUN_SETTING5_RESTORE="${RUN_SETTING5_RESTORE:-1}"
RUN_ACTIVE_REPAIR="${RUN_ACTIVE_REPAIR:-1}"
RUN_NEIGHBORHOOD_REPAIR="${RUN_NEIGHBORHOOD_REPAIR:-0}"
RUN_RETAIN_ONLY_ORACLE="${RUN_RETAIN_ONLY_ORACLE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

FOUR_DIR="${OUTPUT_ROOT}/four_settings"
SETTING5_DIR="${OUTPUT_ROOT}/tofu_setting5e_restore"
ACTIVE_DIR="${OUTPUT_ROOT}/tofu_active_forget_repair"
NEIGHBORHOOD_DIR="${OUTPUT_ROOT}/gagd_neighborhood_confidence_tofu"
ORACLE_DIR="${OUTPUT_ROOT}/retain_only_oracle"
EVAL_DIR="${OUTPUT_ROOT}/evaluation"

case "${SETTING5_SOURCE_MODE}" in
  emb_lm_all_tokens)
    ;;
  *)
    echo "SETTING5_SOURCE_MODE must be emb_lm_all_tokens" >&2
    exit 2
    ;;
esac

if [[ "${RUN_FOUR_SETTINGS}" == "1" ]]; then
  python scripts/tofu_gagd_four_settings_official.py \
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
    --full-lr "${FULL_LR}" \
    --emb-lm-lr "${EMB_LM_LR}" \
    --forget-weight "${FORGET_WEIGHT}" \
    --retain-weight "${RETAIN_WEIGHT}" \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}" \
    --save-model
fi

SETTING5_SOURCE_CHECKPOINT="${FOUR_DIR}/${SETTING5_SOURCE_MODE}/checkpoint"
if [[ ! -d "${SETTING5_SOURCE_CHECKPOINT}" ]]; then
  echo "Missing Setting 5e source checkpoint: ${SETTING5_SOURCE_CHECKPOINT}" >&2
  exit 2
fi

if [[ "${RUN_SETTING5_RESTORE}" == "1" ]]; then
  python scripts/tofu_gagd_setting5e_restore.py \
    --model-path "${SETTING5_SOURCE_CHECKPOINT}" \
    --base-model-path "${MODEL_PATH}" \
    --output-dir "${SETTING5_DIR}" \
    --source-mode "${SETTING5_SOURCE_MODE}" \
    --forget-split "${FORGET_SPLIT}" \
    --retain-split "${RETAIN_SPLIT}" \
    --seed "${SEED}" \
    --forget-num "${FORGET_NUM}" \
    --retain-num "${RETAIN_NUM}" \
    --unique-forget-alpha "${SETTING5_UNIQUE_FORGET_ALPHA}" \
    --overlap-alpha "${SETTING5_OVERLAP_ALPHA}" \
    --max-length "${SETTING5_MAX_LENGTH:-256}" \
    --protect-full-retain-split \
    --protect-utility-splits \
    --restore-input-embeddings \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}" \
    --save-model
fi

SETTING5_CHECKPOINT="${SETTING5_DIR}/checkpoint"
if [[ ! -d "${SETTING5_CHECKPOINT}" ]]; then
  echo "Missing TOFU Setting 5e checkpoint: ${SETTING5_CHECKPOINT}" >&2
  exit 2
fi

if [[ "${RUN_ACTIVE_REPAIR}" == "1" ]]; then
  python scripts/tofu_gagd_active_forget_repair.py \
    --model-path "${SETTING5_CHECKPOINT}" \
    --reference-model-path "${MODEL_PATH}" \
    --output-dir "${ACTIVE_DIR}" \
    --forget-split "${FORGET_SPLIT}" \
    --retain-split "${RETAIN_SPLIT}" \
    --seed "${SEED}" \
    --forget-num "${FORGET_NUM}" \
    --retain-num "${RETAIN_NUM}" \
    --retain-calibration-num "${ACTIVE_RETAIN_CALIBRATION_NUM:-${RETAIN_NUM}}" \
    --real-authors-calibration-num "${ACTIVE_REAL_AUTHORS_CALIBRATION_NUM:-64}" \
    --world-facts-calibration-num "${ACTIVE_WORLD_FACTS_CALIBRATION_NUM:-64}" \
    --calibration-seed "${ACTIVE_CALIBRATION_SEED:-2718}" \
    --target-forget-answer-probability "${TARGET_FORGET_ANSWER_PROBABILITY}" \
    --min-utility-probability-ratio "${MIN_RETAIN_PROBABILITY_RATIO}" \
    --utility-constraint-mode aggregate \
    --target-nll-buffer "${ACTIVE_TARGET_NLL_BUFFER:-0.25}" \
    --repair-steps "${ACTIVE_REPAIR_STEPS:-5000}" \
    --repair-lr "${ACTIVE_REPAIR_LR:-2e-2}" \
    --repair-rank "${ACTIVE_REPAIR_RANK:-64}" \
    --utility-projection-rank "${ACTIVE_UTILITY_PROJECTION_RANK:-64}" \
    --require-input-retain-target \
    --require-utility-constraints \
    --batch-size "${ACTIVE_BATCH_SIZE:-8}" \
    --max-length "${ACTIVE_MAX_LENGTH:-256}" \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}" \
    --save-model
fi

ACTIVE_CHECKPOINT="${ACTIVE_DIR}/checkpoint"
if [[ ! -d "${ACTIVE_CHECKPOINT}" ]]; then
  echo "Missing active-repaired checkpoint: ${ACTIVE_CHECKPOINT}" >&2
  exit 2
fi

FINAL_TARGET_METHOD="tofu_active_forget_repair"
if [[ "${RUN_NEIGHBORHOOD_REPAIR}" == "1" ]]; then
  python scripts/tofu_gagd_neighborhood_confidence.py \
    --model-path "${ACTIVE_CHECKPOINT}" \
    --reference-model-path "${MODEL_PATH}" \
    --output-dir "${NEIGHBORHOOD_DIR}" \
    --forget-split "${FORGET_SPLIT}" \
    --retain-split "${RETAIN_SPLIT}" \
    --seed "${SEED}" \
    --forget-num "${FORGET_NUM}" \
    --retain-num "${RETAIN_NUM}" \
    --retain-calibration-num "${RETAIN_CALIBRATION_NUM:-64}" \
    --real-authors-calibration-num "${REAL_AUTHORS_CALIBRATION_NUM:-32}" \
    --world-facts-calibration-num "${WORLD_FACTS_CALIBRATION_NUM:-32}" \
    --calibration-seed "${CALIBRATION_SEED:-1729}" \
    --max-forget-answer-probability "${TARGET_FORGET_ANSWER_PROBABILITY}" \
    --reference-nll-slack "${NEIGHBORHOOD_REFERENCE_NLL_SLACK:-0.00020002}" \
    --repair-steps "${NEIGHBORHOOD_REPAIR_STEPS:-200}" \
    --repair-lr "${NEIGHBORHOOD_REPAIR_LR:-5e-3}" \
    --repair-rank "${NEIGHBORHOOD_REPAIR_RANK:-32}" \
    --row-selection-top-k "${ROW_SELECTION_TOP_K:-512}" \
    --forget-projection-rank "${FORGET_PROJECTION_RANK:-64}" \
    --max-length "${NEIGHBORHOOD_MAX_LENGTH:-256}" \
    --batch-size "${NEIGHBORHOOD_BATCH_SIZE:-8}" \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}" \
    --no-save-best-effort \
    --save-model
  NEIGHBORHOOD_CHECKPOINT="${NEIGHBORHOOD_DIR}/checkpoint"
  if [[ ! -d "${NEIGHBORHOOD_CHECKPOINT}" ]]; then
    echo "Missing neighborhood-repaired checkpoint: ${NEIGHBORHOOD_CHECKPOINT}" >&2
    exit 2
  fi
  FINAL_TARGET_METHOD="gagd_neighborhood_confidence_tofu"
fi

ORACLE_CHECKPOINT="${RETAIN_ONLY_ORACLE_PATH:-${ORACLE_DIR}/checkpoint}"
if [[ "${RUN_RETAIN_ONLY_ORACLE}" == "1" && ! -d "${ORACLE_CHECKPOINT}" ]]; then
  oracle_cmd=(
    python scripts/tofu_retain_only_oracle.py
    --full-model-path "${MODEL_PATH}"
    --output-dir "${ORACLE_DIR}"
    --forget-split "${FORGET_SPLIT}"
    --retain-split "${RETAIN_SPLIT}"
    --seed "${SEED}"
    --dtype "${DTYPE}"
    --save-model
  )
  if [[ -n "${PRETRAINED_BASE_MODEL_PATH:-}" ]]; then
    oracle_cmd+=(--base-model-path "${PRETRAINED_BASE_MODEL_PATH}")
  fi
  if [[ -n "${ORACLE_EPOCHS:-}" ]]; then
    oracle_cmd+=(--epochs "${ORACLE_EPOCHS}")
  fi
  if [[ -n "${ORACLE_BATCH_SIZE:-}" ]]; then
    oracle_cmd+=(--batch-size "${ORACLE_BATCH_SIZE}")
  fi
  if [[ -n "${ORACLE_LR:-}" ]]; then
    oracle_cmd+=(--lr "${ORACLE_LR}")
  fi
  "${oracle_cmd[@]}"
fi
if [[ ! -d "${ORACLE_CHECKPOINT}" ]]; then
  echo "Missing retain-only oracle checkpoint: ${ORACLE_CHECKPOINT}" >&2
  echo "Set RETAIN_ONLY_ORACLE_PATH or RUN_RETAIN_ONLY_ORACLE=1." >&2
  exit 2
fi

run_tofu_eval() {
  local method="$1"
  local model_dir="$2"
  local base_flag="${3:-0}"
  local reference_kind="${4:-none}"
  local reference_value="${5:-}"
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
  if [[ "${reference_kind}" == "model" ]]; then
    cmd+=(--reference-model-dir "${reference_value}")
  elif [[ "${reference_kind}" == "file" ]]; then
    cmd+=(--reference-truth-ratios "${reference_value}")
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
  "${cmd[@]}"
}

if [[ "${RUN_EVAL}" == "1" ]]; then
  if [[ -n "${TOFU_REFERENCE_TRUTH_RATIOS:-}" ]]; then
    REFERENCE_KIND="file"
    REFERENCE_VALUE="${TOFU_REFERENCE_TRUTH_RATIOS}"
    run_tofu_eval "base" "${MODEL_PATH}" 1 "${REFERENCE_KIND}" "${REFERENCE_VALUE}"
  else
    # tofu_eval writes the oracle distribution while evaluating Base. Reuse
    # that exact JSON for every remaining row instead of reloading the oracle.
    run_tofu_eval "base" "${MODEL_PATH}" 1 "model" "${ORACLE_CHECKPOINT}"
    REFERENCE_KIND="file"
    REFERENCE_VALUE="${EVAL_DIR}/base_${FORGET_SPLIT}_reference_truth_ratios.json"
  fi
  if [[ ! -f "${REFERENCE_VALUE}" ]]; then
    echo "Missing retain-only oracle truth-ratio reference: ${REFERENCE_VALUE}" >&2
    exit 2
  fi

  run_tofu_eval "retain_only_oracle" "${ORACLE_CHECKPOINT}" 0 "${REFERENCE_KIND}" "${REFERENCE_VALUE}"
  run_tofu_eval "full_all_tokens" "${FOUR_DIR}/full_all_tokens/checkpoint" 0 "${REFERENCE_KIND}" "${REFERENCE_VALUE}"
  run_tofu_eval "full_selective_tokens" "${FOUR_DIR}/full_selective_tokens/checkpoint" 0 "${REFERENCE_KIND}" "${REFERENCE_VALUE}"
  run_tofu_eval "emb_lm_all_tokens" "${FOUR_DIR}/emb_lm_all_tokens/checkpoint" 0 "${REFERENCE_KIND}" "${REFERENCE_VALUE}"
  run_tofu_eval "emb_lm_selective_tokens" "${FOUR_DIR}/emb_lm_selective_tokens/checkpoint" 0 "${REFERENCE_KIND}" "${REFERENCE_VALUE}"
  run_tofu_eval "tofu_setting5e_restore" "${SETTING5_CHECKPOINT}" 0 "${REFERENCE_KIND}" "${REFERENCE_VALUE}"
  run_tofu_eval "tofu_active_forget_repair" "${ACTIVE_CHECKPOINT}" 0 "${REFERENCE_KIND}" "${REFERENCE_VALUE}"
  if [[ "${RUN_NEIGHBORHOOD_REPAIR}" == "1" ]]; then
    run_tofu_eval "gagd_neighborhood_confidence_tofu" "${NEIGHBORHOOD_CHECKPOINT}" 0 "${REFERENCE_KIND}" "${REFERENCE_VALUE}"
  fi
fi

comparison_cmd=(
  python scripts/tofu_gagd_results.py
  --output-dir "${OUTPUT_ROOT}/comparison"
  --max-forget-answer-probability "${TARGET_FORGET_ANSWER_PROBABILITY}"
  --min-retain-probability-ratio "${MIN_RETAIN_PROBABILITY_RATIO}"
  --required-target-method "${FINAL_TARGET_METHOD}"
  --result "base=${EVAL_DIR}/base_summary.json"
  --result "retain_only_oracle=${EVAL_DIR}/retain_only_oracle_summary.json"
  --result "full_all_tokens=${EVAL_DIR}/full_all_tokens_summary.json"
  --result "full_selective_tokens=${EVAL_DIR}/full_selective_tokens_summary.json"
  --result "emb_lm_all_tokens=${EVAL_DIR}/emb_lm_all_tokens_summary.json"
  --result "emb_lm_selective_tokens=${EVAL_DIR}/emb_lm_selective_tokens_summary.json"
  --result "tofu_setting5e_restore=${EVAL_DIR}/tofu_setting5e_restore_summary.json"
  --result "tofu_active_forget_repair=${EVAL_DIR}/tofu_active_forget_repair_summary.json"
)
if [[ "${RUN_NEIGHBORHOOD_REPAIR}" == "1" ]]; then
  comparison_cmd+=(
    --result "gagd_neighborhood_confidence_tofu=${EVAL_DIR}/gagd_neighborhood_confidence_tofu_summary.json"
  )
else
  comparison_cmd+=(--allow-partial)
fi
"${comparison_cmd[@]}"

echo "TOFU comparison complete: ${OUTPUT_ROOT}/comparison/comparison_tofu.md"
