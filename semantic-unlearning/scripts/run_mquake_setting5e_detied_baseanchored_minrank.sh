#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 MODEL_PATH [SEED]"
  exit 2
fi

MODEL_PATH="$1"
SEED="${2:-0}"
OUT_ROOT="${OUT_ROOT:-outputs/mquake_detied_baseanchored_minrank/seed${SEED}}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-4}"
MULTIHOP_BATCH_SIZE="${MULTIHOP_BATCH_SIZE:-4}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"

python scripts/mquake_setting5e_detied_baseanchored_minrank.py \
  --model-path "${MODEL_PATH}" \
  --output-dir "${OUT_ROOT}" \
  --mquake-path data/MQuAKE-CF-3k-v2.json \
  --wikidata-dir data/wikidata \
  --seed "${SEED}" \
  --forget-num 1000 \
  --retain-num 1000 \
  --forget-sampling instance_balanced \
  --steps 600 \
  --batch-size 1 \
  --retain-batch-size 4 \
  --emb-lm-lr 1e-4 \
  --forget-weight 2.0 \
  --retain-weight 1.0 \
  --forget-margin 1.0 \
  --emb-lm-optimizer adamw \
  --sampling-strategy epoch \
  --repair-rank-start 2 \
  --repair-rank-max 0 \
  --active-logit-margin 0.25 \
  --selection-logit-margin 0.10 \
  --protected-logit-margin 0.0 \
  --solver-steps-per-phase 1000 \
  --solver-lr 5e-3 \
  --solver-rho 1.0 \
  --solver-feasibility-tolerance 1e-6 \
  --solver-stall-patience 100 \
  --solver-min-improvement 1e-7 \
  --constraint-generation-max-rounds 64 \
  --retain-calibration-num 1000 \
  --retain-calibration-seed 1729 \
  --target-eff-max 0.0 \
  --utility-drop-tolerance 0.10 \
  --max-ppl-ratio 1.02 \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --cache-batch-size "${CACHE_BATCH_SIZE}" \
  --multihop-batch-size "${MULTIHOP_BATCH_SIZE}" \
  --dtype "${DTYPE}" \
  --device-map "${DEVICE_MAP}" \
  --save-setting5-checkpoint \
  --save-selected-checkpoint \
  --fail-if-target-missed

echo "MQuAKE de-tied Base-anchored minimal-rank run complete: ${OUT_ROOT}"
echo "Main result: ${OUT_ROOT}/mquake_results.json"
echo "Rank continuation: ${OUT_ROOT}/baseanchored_minrank_repair/rank_continuation.json"
