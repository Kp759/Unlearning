#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

FULL_TOFU_MODEL="${1:-${TOFU_FULL_MODEL_PATH:-/scratch/yl258/kp759/Unlearning/semantic-unlearning/outputs/tofu_full_utility_sweep_v7_repro_20260815/lr4e-5_epochs6/checkpoint_epoch_5}}"
SEEDS="${TOFU_SEEDS:-1}"
PROTOCOL_ROOT="${TOFU_PROTOCOL_ROOT:-outputs/tofu_author_balanced_locked_3b_test/protocol}"

# Use a fresh output root so prior v1/v2 JSON reports remain untouched even
# though their bulky checkpoint weights are deleted.
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/tofu_sure_author_balanced_r512_r1024_3b_test}"

FORGET_AUTHORS="${TOFU_FORGET_AUTHORS:-5}"
QAS_PER_AUTHOR="${TOFU_QAS_PER_AUTHOR:-20}"
TRAIN_QAS_PER_AUTHOR="${TOFU_TRAIN_QAS_PER_AUTHOR:-10}"
RETAIN_EVAL_NUM="${TOFU_RETAIN_EVAL_NUM:-1000}"
FORGET_NUM=$((FORGET_AUTHORS * TRAIN_QAS_PER_AUTHOR))

# Exact same Stage1A configuration as the validated v2 seed-1 experiment.
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
STAGE1B_MAX_PROMOTION_ROUNDS="${STAGE1B_MAX_PROMOTION_ROUNDS:-16}"
BOUNDARY_BISECTION_STEPS="${BOUNDARY_BISECTION_STEPS:-30}"
BOUNDARY_SAFETY_FRACTION="${BOUNDARY_SAFETY_FRACTION:-0.002}"

# Requested Stage2 ablations. R1024 is allowed as a requested rank; the
# restoration report records the realizable numerical rank, which is capped by
# the selected-row matrix rank (613 visible answer rows in seed1).
RESTORE_RANKS="${RESTORE_RANKS:-512 1024}"
RESTORE_CONSTRAINT_TOLERANCE="${RESTORE_CONSTRAINT_TOLERANCE:-0.001}"
RESTORE_CANDIDATE_SCALES="${RESTORE_CANDIDATE_SCALES:-1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,.03125,.015625,.0078125,0}"

DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
MAX_LENGTH="${MAX_LENGTH:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
RUN_LOCKED_GENERATION="${RUN_LOCKED_GENERATION:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# Storage policy: delete ONLY checkpoint directories/model weights. Parent JSON,
# JSONL, logs, manifests, and locked-eval files are retained.
CLEANUP_PREVIOUS_SURE_WEIGHTS="${CLEANUP_PREVIOUS_SURE_WEIGHTS:-1}"
CLEANUP_AFTER_EVAL="${CLEANUP_AFTER_EVAL:-1}"

if [[ ! -f "${FULL_TOFU_MODEL}/model.safetensors" ]]; then
  echo "Missing protected Full-TOFU model: ${FULL_TOFU_MODEL}" >&2
  exit 2
fi

PROTECTED_FULL_ABS="$(realpath "${FULL_TOFU_MODEL}")"
OUTPUT_ROOT_ABS="$(realpath -m "${OUTPUT_ROOT}")"

safe_delete_checkpoint() {
  local target="$1"
  [[ -d "${target}" ]] || return 0
  if [[ "$(basename "${target}")" != "checkpoint" ]]; then
    echo "REFUSE cleanup: target is not named checkpoint: ${target}" >&2
    exit 90
  fi
  local target_abs
  target_abs="$(realpath "${target}")"
  if [[ "${target_abs}" == "${PROTECTED_FULL_ABS}" || "${target_abs}" == "${PROTECTED_FULL_ABS}"/* || "${PROTECTED_FULL_ABS}" == "${target_abs}"/* ]]; then
    echo "REFUSE cleanup: protected Full-TOFU path overlap: ${target_abs}" >&2
    exit 91
  fi
  echo "DELETE CHECKPOINT WEIGHTS (keep parent JSON): ${target_abs}"
  rm -rf -- "${target_abs}"
}

# Remove prior SURE model weights while preserving every parent report file.
# This targets directories literally named checkpoint only.
if [[ "${CLEANUP_PREVIOUS_SURE_WEIGHTS}" == "1" ]]; then
  echo "===== STORAGE CLEANUP: PREVIOUS SURE CHECKPOINT WEIGHTS ONLY ====="
  shopt -s nullglob
  PREVIOUS_CHECKPOINTS=(
    outputs/tofu_sure_author_balanced_3b_test/seed*/stage1a_gagd/checkpoint
    outputs/tofu_sure_author_balanced_3b_test/seed*/stage1b_rank0_forget/checkpoint
    outputs/tofu_sure_author_balanced_3b_test/seed*/stage2_restore_r*/checkpoint
    outputs/tofu_sure_author_balanced_v2_3b_test/seed*/stage1a_gagd/checkpoint
    outputs/tofu_sure_author_balanced_v2_3b_test/seed*/stage1b_rank0_forget/checkpoint
    outputs/tofu_sure_author_balanced_v2_3b_test/seed*/stage2_restore_r*/checkpoint
  )
  shopt -u nullglob
  for OLD_CKPT in "${PREVIOUS_CHECKPOINTS[@]}"; do
    safe_delete_checkpoint "${OLD_CKPT}"
  done
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
  ROOT="${OUTPUT_ROOT}/seed${SEED}"
  STAGE1A="${ROOT}/stage1a_gagd"
  STAGE1B="${ROOT}/stage1b_rank0_forget"
  mkdir -p "${ROOT}"

  test -f "${TRAIN_FORGET}"
  test -d "${EVAL_DIR}"

  echo
  echo "===== SURE-TOFU SEED ${SEED}: STAGE1A SAME CONFIG AS v2 ====="
  if [[ "${SKIP_EXISTING}" != "1" || ! -f "${STAGE1A}/checkpoint/model.safetensors" ]]; then
    # Remove only this NEW experiment's Stage1A directory when recomputing it.
    # Previous experiment roots are never removed, so their JSON remains.
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

  echo "===== STAGE1B: SAME RESTRICTED RANK-0 + BOUNDARY BISECTION ====="
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

  for RANK in ${RESTORE_RANKS}; do
    RESTORE="${ROOT}/stage2_restore_r${RANK}"
    echo "===== STAGE2: FREEZE REQUESTED R${RANK} NULLSPACE RESTORATION ====="
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

      echo "===== R${RANK}: REASSERT NON-SENSITIVE EMBEDDING + LM-HEAD ROWS TO BASE ====="
      python scripts/tofu_sure_reassert_nonsensitive_base.py \
        --model-path "${RESTORE}/checkpoint" \
        --reference-model-path "${FULL_TOFU_MODEL}" \
        --forget-json "${TRAIN_FORGET}" \
        --forget-requirements-json "${STAGE1B}/forget_instances_after.json" \
        --answer-row-restoration-json "${STAGE1B}/answer_row_restoration.json" \
        --output-json "${RESTORE}/non_sensitive_base_reassertion.json" \
        --seed "${SEED}" --forget-num "${FORGET_NUM}" \
        --target-forget-answer-probability "${TARGET_FORGET_PROB}" \
        --constraint-tolerance "${RESTORE_CONSTRAINT_TOLERANCE}" \
        --batch-size "${EVAL_BATCH_SIZE}" --max-length "${MAX_LENGTH}" \
        --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"
    else
      echo "Reusing ${RESTORE}/checkpoint"
    fi
  done

  echo "===== ALL R512/R1024 CHECKPOINTS FROZEN; HELD-OUT EVAL STARTS ====="
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

  # Same Stage1B post-hoc diagnostic, only after both Stage2 candidates freeze.
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

  # Evaluation is complete. Delete only checkpoint directories; keep all JSON.
  if [[ "${CLEANUP_AFTER_EVAL}" == "1" ]]; then
    echo "===== STORAGE CLEANUP: MODEL WEIGHTS ONLY, KEEP JSON ====="
    for RANK in ${RESTORE_RANKS}; do
      safe_delete_checkpoint "${ROOT}/stage2_restore_r${RANK}/checkpoint"
    done
    safe_delete_checkpoint "${STAGE1B}/checkpoint"
    safe_delete_checkpoint "${STAGE1A}/checkpoint"
    printf '%s\n' \
      "seed=${SEED}" \
      "cleanup_after_eval=1" \
      "protected_full_tofu=${PROTECTED_FULL_ABS}" \
      "deleted=checkpoint directories/model weights only" \
      "kept=all parent JSON, JSONL, logs, locked evals, restoration audits, protocol files" \
      > "${ROOT}/checkpoint_cleanup.txt"
  fi
done

if [[ ! -f "${FULL_TOFU_MODEL}/model.safetensors" || ! -f "${FULL_TOFU_MODEL}/config.json" ]]; then
  echo "FATAL: protected Full-TOFU checkpoint missing after cleanup" >&2
  exit 93
fi

echo
echo "SURE-TOFU R512/R1024 complete. No held-out metric was used before freezing."
echo "Previous and current SURE checkpoint weights were removed; JSON/report files were kept."
echo "Protected Full-TOFU epoch-5 remains: ${FULL_TOFU_MODEL}"
echo "Outputs/reports: ${OUTPUT_ROOT}/seed*/"
