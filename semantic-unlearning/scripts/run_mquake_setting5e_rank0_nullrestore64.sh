#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 MODEL_PATH [SEED]" >&2
  exit 2
fi

MODEL_PATH=$1
SEED=${2:-0}
OUTPUT_DIR="outputs/mquake_s5e600_rank0_nullrestore64/seed${SEED}"

python scripts/mquake_setting5e_rank0_nullrestore64.py \
  --model-path "${MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
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
  --rank0-steps-per-phase 1000 \
  --rank0-lr 5e-3 \
  --rank0-delta-l2 1e-4 \
  --active-logit-margin 0.25 \
  --selection-logit-margin 0.10 \
  --restore-rank 64 \
  --restore-steps 800 \
  --restore-lr 5e-4 \
  --restore-delta-l2 1e-4 \
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
  --eval-batch-size 8 \
  --cache-batch-size 4 \
  --multihop-batch-size 4 \
  --dtype bf16 \
  --device-map single \
  --save-setting5-checkpoint \
  --save-selected-checkpoint \
  --fail-if-target-missed

echo "MQuAKE rank-0/nullrestore64 run complete: ${OUTPUT_DIR}"
