#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash scripts/run_mquake_sure_v71_utility_safe.sh MODEL_PATH [MQUAKE_PATH]" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="$1"
MQUAKE_PATH="${2:-${MQUAKE_PATH:-data/MQuAKE-CF-3k-v2.json}}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mquake_sure_v71_utility_safe}"
SEEDS_TEXT="${MQUAKE_SEEDS:-1}"
REPAIR_RANKS_TEXT="${REPAIR_RANKS:-0}"
FORGET_NUM="${MQUAKE_FORGET_NUM:-50}"
RETAIN_NUM="${MQUAKE_RETAIN_EVAL_NUM:-1000}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
SKIP_PPL="${SKIP_PPL:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# V7.1 Stage1: bounded/saturating forget loss + strong same-prompt GD.
STAGE1_STEPS="${STAGE1_STEPS:-300}"
STAGE1_LR="${STAGE1_LR:-0.0001}"
STAGE1_FORGET_WEIGHT="${STAGE1_FORGET_WEIGHT:-1.0}"
STAGE1_HARDEST_WEIGHT="${STAGE1_HARDEST_WEIGHT:-0.05}"
STAGE1_GD_WEIGHT="${STAGE1_GD_WEIGHT:-10.0}"
STAGE1_L2="${STAGE1_L2:-0.0001}"
STAGE1_TEMP="${STAGE1_TEMP:-2.0}"
STAGE1_TARGET_MARGIN="${STAGE1_TARGET_MARGIN:-0.05}"
STAGE1_GRAD_CLIP="${STAGE1_GRAD_CLIP:-1.0}"

# Stage2 is now only a residual exact repair. Native Eff needs margin > 0, so
# keep the final target small and use a cached BF16 buffer instead of forcing
# an unnecessarily large materialized margin.
STAGE2_TARGET_MARGIN="${STAGE2_TARGET_MARGIN:-0.05}"
STAGE2_BF16_BUFFER="${STAGE2_BF16_BUFFER:-0.20}"
STAGE2_STEPS="${STAGE2_STEPS:-5000}"
STAGE2_LR="${STAGE2_LR:-0.005}"
STAGE2_OPTIMIZER="${STAGE2_OPTIMIZER:-adamw}"
STAGE2_HINGE_WEIGHT="${STAGE2_HINGE_WEIGHT:-100.0}"
STAGE2_HARDEST_WEIGHT="${STAGE2_HARDEST_WEIGHT:-25.0}"
STAGE2_L2="${STAGE2_L2:-0.000001}"
STAGE2_GRAD_CLIP="${STAGE2_GRAD_CLIP:-1.0}"
STAGE2_BISECTION_STEPS="${STAGE2_BISECTION_STEPS:-30}"

read -r -a SEEDS <<< "${SEEDS_TEXT}"
read -r -a RANKS <<< "${REPAIR_RANKS_TEXT}"

test -f "${MODEL_PATH}/config.json"
test -f "${MQUAKE_PATH}"
if [[ "${SKIP_PPL}" != "1" ]]; then
  test -d "${WIKIDATA_DIR}"
fi

for SEED in "${SEEDS[@]}"; do
  ROOT="${OUTPUT_ROOT}/seed${SEED}"
  PROTOCOL_DIR="${ROOT}/protocol"
  VISIBLE="${PROTOCOL_DIR}/repair_visible_forget.json"
  MANIFEST="${PROTOCOL_DIR}/split_manifest.json"
  STAGE1_DIR="${ROOT}/stage1_utility_safe"
  STAGE1_CKPT="${STAGE1_DIR}/checkpoint"
  mkdir -p "${PROTOCOL_DIR}"

  echo "===== SEED ${SEED}: BUILD LOCKED SPLIT ====="
  "${PYTHON_BIN}" scripts/build_mquake_zerounlearn_locked_split.py \
    --mquake-path "${MQUAKE_PATH}" \
    --output-dir "${PROTOCOL_DIR}" \
    --seed "${SEED}" \
    --forget-num "${FORGET_NUM}" \
    --retain-num "${RETAIN_NUM}"

  echo "===== SEED ${SEED}: V7.1 UTILITY-SAFE STAGE1 ====="
  if [[ "${SKIP_EXISTING}" != "1" || ! -f "${STAGE1_CKPT}/config.json" ]]; then
    rm -rf "${STAGE1_CKPT}"
    "${PYTHON_BIN}" scripts/mquake_sure_utility_safe_stage1_v71.py \
      --model-path "${MODEL_PATH}" \
      --repair-visible-path "${VISIBLE}" \
      --split-manifest "${MANIFEST}" \
      --output-dir "${STAGE1_DIR}" \
      --seed "${SEED}" \
      --forget-num "${FORGET_NUM}" \
      --steps "${STAGE1_STEPS}" \
      --lr "${STAGE1_LR}" \
      --forget-weight "${STAGE1_FORGET_WEIGHT}" \
      --hardest-forget-weight "${STAGE1_HARDEST_WEIGHT}" \
      --gd-weight "${STAGE1_GD_WEIGHT}" \
      --delta-l2-lambda "${STAGE1_L2}" \
      --forget-temperature "${STAGE1_TEMP}" \
      --target-logit-margin "${STAGE1_TARGET_MARGIN}" \
      --grad-clip "${STAGE1_GRAD_CLIP}" \
      --batch-size "${BATCH_SIZE}" \
      --dtype "${DTYPE}" \
      --device-map "${DEVICE_MAP}"
  else
    echo "Reusing ${STAGE1_CKPT}"
  fi

  for RANK in "${RANKS[@]}"; do
    STAGE2_DIR="${ROOT}/stage2_active_r${RANK}"
    STAGE2_CKPT="${STAGE2_DIR}/checkpoint"
    EVAL_OUT="${ROOT}/official_eval_r${RANK}.json"
    EVAL_MANIFEST="${ROOT}/final_eval_split_manifest_r${RANK}.json"

    echo "===== SEED ${SEED}: RESIDUAL STAGE2 RANK ${RANK} ====="
    if [[ "${SKIP_EXISTING}" != "1" || ! -f "${STAGE2_CKPT}/config.json" ]]; then
      rm -rf "${STAGE2_CKPT}"
      if ! "${PYTHON_BIN}" scripts/mquake_sure_active_hidden_repair_v7_entry.py \
        --model-path "${STAGE1_CKPT}" \
        --reference-model-path "${MODEL_PATH}" \
        --repair-visible-path "${VISIBLE}" \
        --split-manifest "${MANIFEST}" \
        --output-dir "${STAGE2_DIR}" \
        --seed "${SEED}" \
        --forget-num "${FORGET_NUM}" \
        --target-logit-margin "${STAGE2_TARGET_MARGIN}" \
        --bf16-buffer-margin "${STAGE2_BF16_BUFFER}" \
        --repair-rank "${RANK}" \
        --repair-steps "${STAGE2_STEPS}" \
        --repair-lr "${STAGE2_LR}" \
        --repair-optimizer "${STAGE2_OPTIMIZER}" \
        --forget-hinge-weight "${STAGE2_HINGE_WEIGHT}" \
        --hardest-forget-hinge-weight "${STAGE2_HARDEST_WEIGHT}" \
        --delta-l2-lambda "${STAGE2_L2}" \
        --grad-clip "${STAGE2_GRAD_CLIP}" \
        --boundary-bisection-steps "${STAGE2_BISECTION_STEPS}" \
        --batch-size "${BATCH_SIZE}" \
        --dtype "${DTYPE}" \
        --device-map "${DEVICE_MAP}"; then
        echo "Rank ${RANK} Stage2 failed; preserving diagnostics." >&2
        continue
      fi
    fi

    echo "===== SEED ${SEED}: LOCKED FINAL EVAL RANK ${RANK} ====="
    EVAL_ARGS=(
      --model-dir "${STAGE2_CKPT}"
      --mquake-path "${MQUAKE_PATH}"
      --wikidata-dir "${WIKIDATA_DIR}"
      --out "${EVAL_OUT}"
      --split-manifest "${EVAL_MANIFEST}"
      --method "SURE-MQuAKE V7.1 utility-safe rank ${RANK}"
      --unlearn-num "${FORGET_NUM}"
      --retain-num "${RETAIN_NUM}"
      --seed "${SEED}"
      --batch-size "${EVAL_BATCH_SIZE}"
      --dtype "${DTYPE}"
      --device-map "${DEVICE_MAP}"
      --skip-atomic-gen
    )
    if [[ "${SKIP_PPL}" == "1" ]]; then
      EVAL_ARGS+=(--skip-ppl)
    fi
    "${PYTHON_BIN}" scripts/mquake_zero_unlearn_official_eval.py "${EVAL_ARGS[@]}"
  done
done

echo "SURE-MQuAKE V7.1 utility-safe run complete: ${OUTPUT_ROOT}"
