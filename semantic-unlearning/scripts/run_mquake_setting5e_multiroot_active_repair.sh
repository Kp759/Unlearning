#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 MODEL_PATH [SEED]"
  exit 2
fi

MODEL_PATH="$1"
SEED="${2:-${SEED:-0}}"
OUT_ROOT="${OUT_ROOT:-outputs/mquake_setting5e_multiroot_active/seed${SEED}}"
MQUAKE_PATH="${MQUAKE_PATH:-data/MQuAKE-CF-3k-v2.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
FORGET_NUM="${FORGET_NUM:-1000}"
RETAIN_NUM="${RETAIN_NUM:-1000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-4}"
MULTIHOP_BATCH_SIZE="${MULTIHOP_BATCH_SIZE:-4}"
FORGET_SAMPLING="${FORGET_SAMPLING:-instance_balanced}"

python scripts/mquake_gagd_setting5e_multiroot_active_repair.py \
  --model-path "${MODEL_PATH}" \
  --output-dir "${OUT_ROOT}" \
  --mquake-path "${MQUAKE_PATH}" \
  --wikidata-dir "${WIKIDATA_DIR}" \
  --seed "${SEED}" \
  --forget-num "${FORGET_NUM}" \
  --retain-num "${RETAIN_NUM}" \
  --forget-sampling "${FORGET_SAMPLING}" \
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
  --repair-steps 600 \
  --repair-lr 2e-3 \
  --repair-optimizer adamw \
  --active-logit-margin 0.50 \
  --selection-logit-margin 0.10 \
  --protected-logit-margin 0.0 \
  --protected-logit-drift-weight 1.0 \
  --repair-rank 0 \
  --repair-l2-lambda 1e-6 \
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

echo "MQuAKE Setting 5e + protected active-pair repair run complete: ${OUT_ROOT}"
echo "Main result: ${OUT_ROOT}/mquake_results.json"
echo "Repair diagnostics: ${OUT_ROOT}/multirow_active_repair/repair_summary.json"
echo "Multi-hop: ${OUT_ROOT}/multihop_unlearning_eval.json"
