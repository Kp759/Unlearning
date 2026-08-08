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
OUT_ROOT="${OUT_ROOT:-outputs/mquake_setting5e_rank2_active_pair/seed${SEED}}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-4}"
MULTIHOP_BATCH_SIZE="${MULTIHOP_BATCH_SIZE:-4}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"

python scripts/mquake_gagd_setting5e_multiroot_active_repair.py \
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
  --repair-mode active_pair \
  --repair-steps 2000 \
  --repair-lr 5e-3 \
  --repair-optimizer adamw \
  --active-logit-margin 0.25 \
  --selection-logit-margin 0.10 \
  --protected-logit-margin 0.0 \
  --repair-rank 2 \
  --repair-l2-lambda 1e-4 \
  --protected-logit-drift-weight 0.1 \
  --retain-calibration-num 1000 \
  --retain-calibration-seed 1729 \
  --no-project-away-protected-hidden \
  --stop-when-all-satisfied \
  --candidate-scales "1.0,0.875,0.75,0.625,0.5,0.375,0.25,0.1875,0.125,0.09375,0.0625,0.046875,0.03125,0.015625,0.0078125,0.0" \
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

echo "MQuAKE Setting 5e + protected rank-2 active-pair run complete: ${OUT_ROOT}"
echo "Main result: ${OUT_ROOT}/mquake_results.json"
echo "Repair diagnostics: ${OUT_ROOT}/multirow_active_repair/repair_summary.json"
