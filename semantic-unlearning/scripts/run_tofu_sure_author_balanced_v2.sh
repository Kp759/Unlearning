#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

FULL_TOFU_MODEL="${1:-${TOFU_FULL_MODEL_PATH:-/scratch/yl258/kp759/Unlearning/semantic-unlearning/outputs/tofu_full_utility_sweep_v7_repro_20260815/lr4e-5_epochs6/checkpoint_epoch_5}}"
SEEDS="${TOFU_SEEDS:-1}"
PROTOCOL_ROOT="${TOFU_PROTOCOL_ROOT:-outputs/tofu_author_balanced_locked_3b_test/protocol}"
SOURCE_ROOT="${SOURCE_ROOT:-outputs/tofu_sure_author_balanced_3b_test}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/tofu_sure_author_balanced_v2_3b_test}"

FORGET_AUTHORS="${TOFU_FORGET_AUTHORS:-5}"
QAS_PER_AUTHOR="${TOFU_QAS_PER_AUTHOR:-20}"
TRAIN_QAS_PER_AUTHOR="${TOFU_TRAIN_QAS_PER_AUTHOR:-10}"
RETAIN_EVAL_NUM="${TOFU_RETAIN_EVAL_NUM:-1000}"
FORGET_NUM=$((FORGET_AUTHORS * TRAIN_QAS_PER_AUTHOR))

TARGET_FORGET_PROB="${TARGET_FORGET_PROB:-0.0003}"
TARGET_NLL_BUFFER="${TARGET_NLL_BUFFER:-0.25}"
STAGE1B_STEPS="${STAGE1B_STEPS:-10000}"
STAGE1B_LR="${STAGE1B_LR:-0.005}"
STAGE1B_L2="${STAGE1B_L2:-0.000001}"
STAGE1B_MAX_PROMOTION_ROUNDS="${STAGE1B_MAX_PROMOTION_ROUNDS:-16}"
BOUNDARY_BISECTION_STEPS="${BOUNDARY_BISECTION_STEPS:-30}"
BOUNDARY_SAFETY_FRACTION="${BOUNDARY_SAFETY_FRACTION:-0.002}"

RESTORE_RANKS="${RESTORE_RANKS:-64 128}"
RESTORE_CONSTRAINT_TOLERANCE="${RESTORE_CONSTRAINT_TOLERANCE:-0.001}"
RESTORE_CANDIDATE_SCALES="${RESTORE_CANDIDATE_SCALES:-1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,.03125,.015625,.0078125,0}"

DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
MAX_LENGTH="${MAX_LENGTH:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
RUN_LOCKED_GENERATION="${RUN_LOCKED_GENERATION:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

if [[ ! -f "${FULL_TOFU_MODEL}/model.safetensors" ]]; then
  echo "Missing protected Full-TOFU model: ${FULL_TOFU_MODEL}" >&2
  exit 2
fi

MISSING_SEEDS=()
for SEED in ${SEEDS}; do
  if [[ ! -f "${PROTOCOL_ROOT}/seed${SEED}/split_manifest.json" ]]; then
    MISSING_SEEDS+=("${SEED}")
  fi
done
if (( ${#MISSING_SEEDS[@]} > 0 )); then
  python scripts/build_tofu_zerounlearn_locked_split.py \
    --output-dir "${PROTOCOL_ROOT}" \
    --seeds "${MISSING_SEEDS[@]}" \
    --forget-authors "${FORGET_AUTHORS}" \
    --qas-per-author "${QAS_PER_AUTHOR}" \
    --train-qas-per-author "${TRAIN_QAS_PER_AUTHOR}" \
    --retain-num "${RETAIN_EVAL_NUM}"
fi

for SEED in ${SEEDS}; do
  PROTOCOL_SEED="${PROTOCOL_ROOT}/seed${SEED}"
  TRAIN_FORGET="${PROTOCOL_SEED}/train_visible/forget.json"
  EVAL_DIR="${PROTOCOL_SEED}/eval_only"
  STAGE1A="${SOURCE_ROOT}/seed${SEED}/stage1a_gagd"
  ROOT="${OUTPUT_ROOT}/seed${SEED}"
  STAGE1B="${ROOT}/stage1b_rank0_forget"
  mkdir -p "${ROOT}"

  test -f "${TRAIN_FORGET}"
  test -d "${EVAL_DIR}"
  test -f "${STAGE1A}/checkpoint/model.safetensors" || {
    echo "Missing reusable Stage1A checkpoint: ${STAGE1A}/checkpoint" >&2
    echo "Run the original SURE runner through Stage1A first, or set SOURCE_ROOT." >&2
    exit 3
  }

  echo
  echo "===== SURE-TOFU-v2 SEED ${SEED}: REUSE FROZEN STAGE1A ====="
  echo "${STAGE1A}/checkpoint"

  echo "===== STAGE1B-v2: RESTORE NON-SENSITIVE ROWS, THEN RESTRICTED RANK-0 ====="
  if [[ "${SKIP_EXISTING}" != "1" || ! -f "${STAGE1B}/checkpoint/model.safetensors" ]]; then
    rm -rf "${STAGE1B}"
    python scripts/tofu_sure_rank0_forget_restored.py \
      --model-path "${STAGE1A}/checkpoint" \
      --reference-model-path "${FULL_TOFU_MODEL}" \
      --forget-json "${TRAIN_FORGET}" \
      --output-dir "${STAGE1B}" \
      --seed "${SEED}" --forget-num "${FORGET_NUM}" \
      --target-forget-answer-probability "${TARGET_FORGET_PROB}" \
      --target-nll-buffer "${TARGET_NLL_BUFFER}" \
      --repair-steps "${STAGE1B_STEPS}" \
      --repair-lr "${STAGE1B_LR}" \
      --delta-l2-lambda "${STAGE1B_L2}" \
      --max-promotion-rounds "${STAGE1B_MAX_PROMOTION_ROUNDS}" \
      --boundary-bisection-steps "${BOUNDARY_BISECTION_STEPS}" \
      --boundary-safety-fraction "${BOUNDARY_SAFETY_FRACTION}" \
      --batch-size "${EVAL_BATCH_SIZE}" --max-length "${MAX_LENGTH}" \
      --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"
  else
    echo "Reusing ${STAGE1B}/checkpoint"
  fi

  # Both restoration ranks are frozen before any held-out metric is read.
  for RANK in ${RESTORE_RANKS}; do
    RESTORE="${ROOT}/stage2_restore_r${RANK}"
    echo "===== FREEZE R${RANK} NULLSPACE RESTORATION ====="
    if [[ "${SKIP_EXISTING}" != "1" || ! -f "${RESTORE}/checkpoint/model.safetensors" ]]; then
      rm -rf "${RESTORE}"
      python scripts/tofu_sure_nullspace_restore.py \
        --model-path "${STAGE1B}/checkpoint" \
        --reference-model-path "${FULL_TOFU_MODEL}" \
        --forget-json "${TRAIN_FORGET}" \
        --forget-requirements-json "${STAGE1B}/forget_instances_after.json" \
        --output-dir "${RESTORE}" \
        --seed "${SEED}" --forget-num "${FORGET_NUM}" \
        --restore-rank "${RANK}" \
        --target-forget-answer-probability "${TARGET_FORGET_PROB}" \
        --constraint-tolerance "${RESTORE_CONSTRAINT_TOLERANCE}" \
        --candidate-scales "${RESTORE_CANDIDATE_SCALES}" \
        --batch-size "${EVAL_BATCH_SIZE}" --max-length "${MAX_LENGTH}" \
        --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"
    else
      echo "Reusing ${RESTORE}/checkpoint"
    fi
  done

  echo "===== ALL v2 CHECKPOINTS FROZEN; HELD-OUT EVALUATION STARTS ====="
  for RANK in ${RESTORE_RANKS}; do
    RESTORE="${ROOT}/stage2_restore_r${RANK}"
    EVAL_ARGS=(
      --model-dir "${RESTORE}/checkpoint"
      --eval-dir "${EVAL_DIR}"
      --reference-model-dir "${FULL_TOFU_MODEL}"
      --output "${ROOT}/locked_eval_r${RANK}.json"
      --seed "${SEED}"
      --dtype "${DTYPE}"
      --max-length "${MAX_LENGTH}"
    )
    if [[ "${RUN_LOCKED_GENERATION}" != "1" ]]; then
      EVAL_ARGS+=(--skip-generation)
    fi
    python scripts/tofu_zerounlearn_locked_eval.py "${EVAL_ARGS[@]}"
  done

  POSTHOC_ARGS=(
    --model-dir "${STAGE1B}/checkpoint"
    --eval-dir "${EVAL_DIR}"
    --reference-model-dir "${FULL_TOFU_MODEL}"
    --output "${ROOT}/locked_eval_stage1b_posthoc.json"
    --seed "${SEED}"
    --dtype "${DTYPE}"
    --max-length "${MAX_LENGTH}"
  )
  if [[ "${RUN_LOCKED_GENERATION}" != "1" ]]; then
    POSTHOC_ARGS+=(--skip-generation)
  fi
  python scripts/tofu_zerounlearn_locked_eval.py "${POSTHOC_ARGS[@]}"
done

echo
echo "SURE-TOFU-v2 complete. Stage1A was reused; no held-out metric was used before freezing."
echo "Outputs: ${OUTPUT_ROOT}/seed*/"
