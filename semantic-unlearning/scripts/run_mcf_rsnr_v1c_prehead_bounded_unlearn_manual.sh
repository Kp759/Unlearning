#!/usr/bin/env bash
set -euo pipefail

OUT="${1:?usage: $0 OUTPUT_DIR}"
: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${SOURCE_V13_RUN:?SOURCE_V13_RUN must be set}"
: "${VIEWS:?VIEWS must be set}"

python -u scripts/run_mcf_rsnr_v1c_prehead_bounded_unlearn.py \
  --model-path "$MODEL_PATH" \
  --protocol-dir "$SOURCE_V13_RUN/protocol" \
  --view-corpus "$VIEWS" \
  --output-dir "$OUT" \
  --seed 1 \
  --forget-num 50 \
  --dtype bf16 \
  --steps 1600 \
  --case-batch-size 4 \
  --check-every 25 \
  --learning-rate 1e-4 \
  --adapter-rank 16 \
  --adapter-alpha 16 \
  --bounded-ga-weight 0.10 \
  --cf-margin-weight 4.0 \
  --idk-margin-weight 1.0 \
  --drop-weight 1.0 \
  --target-new-preserve-weight 4.0 \
  --minimum-cf-margin 0.1 \
  --minimum-idk-margin 0.1 \
  --minimum-true-logprob-drop 2.0 \
  --max-target-new-logprob-drift 0.25 \
  --hard-case-fraction 0.75 \
  --anchor-weight 1e-4 \
  --grad-clip 1.0 \
  --gate-off-logit-drift-max 0.0
