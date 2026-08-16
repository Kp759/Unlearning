#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

FULL_TOFU_MODEL="${1:-${TOFU_FULL_MODEL_PATH:-/scratch/yl258/kp759/Unlearning/semantic-unlearning/outputs/tofu_full_utility_sweep_v7_repro_20260815/lr4e-5_epochs6/checkpoint_epoch_5}}"
SEEDS="${TOFU_SEEDS:-1}"
PROTOCOL_ROOT="${TOFU_PROTOCOL_ROOT:-outputs/tofu_author_balanced_locked_3b_test/protocol}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/tofu_sure_author_balanced_v5_100rows_3b_test}"

FORGET_AUTHORS="${TOFU_FORGET_AUTHORS:-5}"
QAS_PER_AUTHOR="${TOFU_QAS_PER_AUTHOR:-20}"
TRAIN_QAS_PER_AUTHOR="${TOFU_TRAIN_QAS_PER_AUTHOR:-10}"
RETAIN_EVAL_NUM="${TOFU_RETAIN_EVAL_NUM:-1000}"
FORGET_NUM=$((FORGET_AUTHORS * TRAIN_QAS_PER_AUTHOR))

# Same deterministic Stage1A used by prior SURE ablations.
STAGE1A_STEPS="${STAGE1A_STEPS:-600}"
STAGE1A_LR="${STAGE1A_LR:-0.0001}"
STAGE1A_GA_WEIGHT="${STAGE1A_GA_WEIGHT:-2.0}"
STAGE1A_GD_WEIGHT="${STAGE1A_GD_WEIGHT:-1.0}"
STAGE1A_RESTORATION_MODE="${STAGE1A_RESTORATION_MODE:-sensitive_both}"

# V5 fixed-budget controls.
INITIAL_UNIQUE_ROW_BUDGET="${INITIAL_UNIQUE_ROW_BUDGET:-100}"
TARGET_FORGET_PROB="${TARGET_FORGET_PROB:-0.0003}"
# V5 intentionally targets the declared probability boundary with no extra buffer.
TARGET_NLL_BUFFER="0"
STAGE1B_STEPS="${STAGE1B_STEPS:-10000}"
STAGE1B_LR="${STAGE1B_LR:-0.005}"
STAGE1B_L2="${STAGE1B_L2:-0.000001}"
BOUNDARY_BISECTION_STEPS="${BOUNDARY_BISECTION_STEPS:-30}"
BOUNDARY_SAFETY_FRACTION="${BOUNDARY_SAFETY_FRACTION:-0.002}"

DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
MAX_LENGTH="${MAX_LENGTH:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
RUN_LOCKED_GENERATION="${RUN_LOCKED_GENERATION:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# Delete only SURE checkpoint directories. Keep JSON/JSONL/logs/evaluations.
CLEANUP_PREVIOUS_SURE_WEIGHTS="${CLEANUP_PREVIOUS_SURE_WEIGHTS:-1}"
CLEANUP_AFTER_EVAL="${CLEANUP_AFTER_EVAL:-1}"

if [[ ! -f "${FULL_TOFU_MODEL}/model.safetensors" || ! -f "${FULL_TOFU_MODEL}/config.json" ]]; then
  echo "Missing protected Full-TOFU model: ${FULL_TOFU_MODEL}" >&2
  exit 2
fi
PROTECTED_FULL_ABS="$(realpath "${FULL_TOFU_MODEL}")"

safe_delete_sure_checkpoint() {
  local target="$1"
  [[ -d "${target}" ]] || return 0
  [[ "$(basename "${target}")" == "checkpoint" ]] || {
    echo "REFUSE delete: not a checkpoint directory: ${target}" >&2
    exit 90
  }
  local target_abs
  target_abs="$(realpath "${target}")"
  if [[ "${target_abs}" == "${PROTECTED_FULL_ABS}" || "${target_abs}" == "${PROTECTED_FULL_ABS}"/* || "${PROTECTED_FULL_ABS}" == "${target_abs}"/* ]]; then
    echo "REFUSE delete: protected Full-TOFU overlap: ${target_abs}" >&2
    exit 91
  fi
  case "${target_abs}" in
    */outputs/tofu_sure*/checkpoint) ;;
    *)
      echo "REFUSE delete outside SURE output checkpoint: ${target_abs}" >&2
      exit 92
      ;;
  esac
  echo "DELETE SURE CHECKPOINT WEIGHTS, KEEP JSON/REPORTS: ${target_abs}"
  rm -rf -- "${target_abs}"
}

if [[ "${CLEANUP_PREVIOUS_SURE_WEIGHTS}" == "1" ]]; then
  echo "===== PRE-RUN CLEANUP: OLD SURE WEIGHTS ONLY ====="
  while IFS= read -r -d '' ckpt; do
    safe_delete_sure_checkpoint "${ckpt}"
  done < <(find outputs -type d -name checkpoint -path '*/tofu_sure*/*' -print0 2>/dev/null || true)
  echo "Previous SURE JSON/log/report files were preserved."
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
  STAGE1B="${ROOT}/stage1b_exact100_v5"
  mkdir -p "${ROOT}"

  test -f "${TRAIN_FORGET}"
  test -d "${EVAL_DIR}"

  echo
  echo "===== SURE-TOFU-v5 SEED ${SEED}: STAGE1A SAME-PROMPT GA/GD ====="
  if [[ "${SKIP_EXISTING}" != "1" || ! -f "${STAGE1A}/checkpoint/model.safetensors" ]]; then
    rm -rf "${STAGE1A}/checkpoint"
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

  echo "===== SURE-TOFU-v5: EXACT ${INITIAL_UNIQUE_ROW_BUDGET} SENSITIVE ROWS ====="
  if [[ "${SKIP_EXISTING}" != "1" || ! -f "${STAGE1B}/checkpoint/model.safetensors" ]]; then
    rm -rf "${STAGE1B}/checkpoint"
    python scripts/tofu_sure_rank0_forget_budget100_v5.py \
      --model-path "${STAGE1A}/checkpoint" \
      --reference-model-path "${FULL_TOFU_MODEL}" \
      --forget-json "${TRAIN_FORGET}" \
      --output-dir "${STAGE1B}" \
      --seed "${SEED}" --forget-num "${FORGET_NUM}" \
      --initial-unique-row-budget "${INITIAL_UNIQUE_ROW_BUDGET}" \
      --initial-rows-per-example 1 \
      --promotion-rows-per-example 1 \
      --max-promotion-rounds 1 \
      --target-forget-answer-probability "${TARGET_FORGET_PROB}" \
      --target-nll-buffer "${TARGET_NLL_BUFFER}" \
      --repair-steps "${STAGE1B_STEPS}" \
      --repair-lr "${STAGE1B_LR}" \
      --delta-l2-lambda "${STAGE1B_L2}" \
      --boundary-bisection-steps "${BOUNDARY_BISECTION_STEPS}" \
      --boundary-safety-fraction "${BOUNDARY_SAFETY_FRACTION}" \
      --batch-size "${EVAL_BATCH_SIZE}" --max-length "${MAX_LENGTH}" \
      --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"
  else
    echo "Reusing ${STAGE1B}/checkpoint"
  fi

  echo "===== V5 CHECKPOINT FROZEN; LOCKED HELD-OUT EVALUATION STARTS ====="
  EVAL_ARGS=(
    --model-dir "${STAGE1B}/checkpoint"
    --eval-dir "${EVAL_DIR}"
    --reference-model-dir "${FULL_TOFU_MODEL}"
    --output "${ROOT}/locked_eval_v5_r0.json"
    --seed "${SEED}"
    --dtype "${DTYPE}"
    --max-length "${MAX_LENGTH}"
  )
  if [[ "${RUN_LOCKED_GENERATION}" != "1" ]]; then
    EVAL_ARGS+=(--skip-generation)
  fi
  python scripts/tofu_zerounlearn_locked_eval.py "${EVAL_ARGS[@]}"

  if [[ "${CLEANUP_AFTER_EVAL}" == "1" ]]; then
    echo "===== V5 CLEANUP: WEIGHTS ONLY, KEEP JSON ====="
    safe_delete_sure_checkpoint "${STAGE1B}/checkpoint"
    safe_delete_sure_checkpoint "${STAGE1A}/checkpoint"
    printf '%s\n' \
      "seed=${SEED}" \
      "fixed_sensitive_row_budget=${INITIAL_UNIQUE_ROW_BUDGET}" \
      "target_nll_buffer=0" \
      "cleanup_after_eval=1" \
      "deleted=Stage1A and v5 Stage1B checkpoint directories only" \
      "kept=all JSON, JSONL, logs, split manifests, locked evaluation" \
      "protected_full_tofu=${PROTECTED_FULL_ABS}" \
      > "${ROOT}/checkpoint_cleanup.txt"
  fi
done

if [[ ! -f "${FULL_TOFU_MODEL}/model.safetensors" || ! -f "${FULL_TOFU_MODEL}/config.json" ]]; then
  echo "FATAL: protected Full-TOFU checkpoint missing after v5 cleanup" >&2
  exit 93
fi

echo
echo "SURE-TOFU-v5 exact-${INITIAL_UNIQUE_ROW_BUDGET}-row run complete."
echo "Sensitive rows kept Stage1A embedding/LM baseline; non-sensitive visible answer rows were exact Base."
echo "Target NLL buffer was zero and no promotion beyond the fixed row budget was allowed."
echo "Old/current SURE weights were removed; JSON/report files were kept."
echo "Protected Full-TOFU epoch-5 remains: ${FULL_TOFU_MODEL}"
echo "Outputs/reports: ${OUTPUT_ROOT}/seed*/"
