#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

FULL_TOFU_MODEL="${1:-${TOFU_FULL_MODEL_PATH:-/scratch/yl258/kp759/Unlearning/semantic-unlearning/outputs/tofu_full_utility_sweep_v7_repro_20260815/lr4e-5_epochs6/checkpoint_epoch_5}}"
SEEDS="${TOFU_SEEDS:-1}"
PROTOCOL_ROOT="${TOFU_PROTOCOL_ROOT:-outputs/tofu_author_balanced_locked_3b_test/protocol}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/tofu_sure_author_balanced_3b_test}"

FORGET_AUTHORS="${TOFU_FORGET_AUTHORS:-5}"
QAS_PER_AUTHOR="${TOFU_QAS_PER_AUTHOR:-20}"
TRAIN_QAS_PER_AUTHOR="${TOFU_TRAIN_QAS_PER_AUTHOR:-10}"
RETAIN_EVAL_NUM="${TOFU_RETAIN_EVAL_NUM:-1000}"
FORGET_NUM=$((FORGET_AUTHORS * TRAIN_QAS_PER_AUTHOR))

STAGE1A_STEPS="${STAGE1A_STEPS:-600}"
STAGE1A_LR="${STAGE1A_LR:-0.0001}"
STAGE1A_GA_WEIGHT="${STAGE1A_GA_WEIGHT:-2.0}"
STAGE1A_GD_WEIGHT="${STAGE1A_GD_WEIGHT:-1.0}"
STAGE1A_RESTORATION_MODE="${STAGE1A_RESTORATION_MODE:-sensitive_both}"

TARGET_FORGET_PROB="${TARGET_FORGET_PROB:-0.0003}"
TARGET_NLL_BUFFER="${TARGET_NLL_BUFFER:-0.25}"
STAGE1B_STEPS="${STAGE1B_STEPS:-10000}"
STAGE1B_LR="${STAGE1B_LR:-0.005}"
STAGE1B_L2="${STAGE1B_L2:-0.000001}"
STAGE1B_MAX_PROMOTION_ROUNDS="${STAGE1B_MAX_PROMOTION_ROUNDS:-64}"

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

# Build only missing deterministic protocol seeds.  The builder is the exact
# same author-balanced split generator used by the ZeroUnlearn comparison.
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
  SPLIT_MANIFEST="${PROTOCOL_SEED}/split_manifest.json"

  ROOT="${OUTPUT_ROOT}/seed${SEED}"
  STAGE1A="${ROOT}/stage1a_gagd"
  STAGE1B="${ROOT}/stage1b_rank0_forget"
  mkdir -p "${ROOT}"

  test -f "${TRAIN_FORGET}"
  test -d "${EVAL_DIR}"
  test -f "${SPLIT_MANIFEST}"

  echo
  echo "===== SURE-TOFU SEED ${SEED}: STAGE 1A SAME-PROMPT GA/GD ====="
  if [[ "${SKIP_EXISTING}" != "1" || ! -f "${STAGE1A}/checkpoint/model.safetensors" ]]; then
    rm -rf "${STAGE1A}"
    python scripts/tofu_sure_stage1_gagd.py \
      --model-path "${FULL_TOFU_MODEL}" \
      --forget-json "${TRAIN_FORGET}" \
      --output-dir "${STAGE1A}" \
      --seed "${SEED}" --forget-num "${FORGET_NUM}" \
      --steps "${STAGE1A_STEPS}" --batch-size 1 \
      --emb-lm-lr "${STAGE1A_LR}" \
      --ga-weight "${STAGE1A_GA_WEIGHT}" \
      --gd-weight "${STAGE1A_GD_WEIGHT}" \
      --restoration-mode "${STAGE1A_RESTORATION_MODE}" \
      --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"
  else
    echo "Reusing ${STAGE1A}/checkpoint"
  fi

  echo "===== SURE-TOFU SEED ${SEED}: STAGE 1B SENSITIVE-ROW RESTORE + UNRESTRICTED RANK-0 ====="
  if [[ "${SKIP_EXISTING}" != "1" || ! -f "${STAGE1B}/checkpoint/model.safetensors" ]]; then
    rm -rf "${STAGE1B}"
    python scripts/tofu_sure_rank0_forget.py \
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
      --batch-size "${EVAL_BATCH_SIZE}" \
      --max-length "${MAX_LENGTH}" \
      --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"
  else
    echo "Reusing ${STAGE1B}/checkpoint"
  fi

  # Freeze every predeclared restoration candidate before any held-out metric is
  # computed.  R64 and R128 are reported as an ablation; neither is selected by
  # retain1000, paraphrases, same-author holdout, PPL, or native TOFU metrics.
  for RANK in ${RESTORE_RANKS}; do
    RESTORE="${ROOT}/stage2_restore_r${RANK}"
    echo "===== SURE-TOFU SEED ${SEED}: FREEZE R${RANK} NULLSPACE RESTORATION ====="
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

  echo "===== ALL SURE CHECKPOINTS FROZEN; HELD-OUT EVALUATION STARTS NOW ====="
  for RANK in ${RESTORE_RANKS}; do
    RESTORE="${ROOT}/stage2_restore_r${RANK}"
    EVAL_OUT="${ROOT}/locked_eval_r${RANK}.json"
    EVAL_ARGS=(
      --model-dir "${RESTORE}/checkpoint"
      --eval-dir "${EVAL_DIR}"
      --reference-model-dir "${FULL_TOFU_MODEL}"
      --output "${EVAL_OUT}"
      --seed "${SEED}"
      --dtype "${DTYPE}"
      --max-length "${MAX_LENGTH}"
    )
    if [[ "${RUN_LOCKED_GENERATION}" != "1" ]]; then
      EVAL_ARGS+=(--skip-generation)
    fi
    python scripts/tofu_zerounlearn_locked_eval.py "${EVAL_ARGS[@]}"
  done

  # Post-hoc Stage1B diagnostic only after every restoration candidate is frozen.
  STAGE1B_EVAL="${ROOT}/locked_eval_stage1b_posthoc.json"
  POSTHOC_ARGS=(
    --model-dir "${STAGE1B}/checkpoint"
    --eval-dir "${EVAL_DIR}"
    --reference-model-dir "${FULL_TOFU_MODEL}"
    --output "${STAGE1B_EVAL}"
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
echo "SURE-TOFU complete. No held-out metric was used before checkpoint freezing."
echo "Outputs: ${OUTPUT_ROOT}/seed*/"
