#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:?usage: $0 MODEL_PATH SOURCE_V13_RUN OUTPUT_DIR}"
SOURCE_V13_RUN="${2:?usage: $0 MODEL_PATH SOURCE_V13_RUN OUTPUT_DIR}"
OUTPUT_DIR="${3:?usage: $0 MODEL_PATH SOURCE_V13_RUN OUTPUT_DIR}"

PROTOCOL_DIR="$SOURCE_V13_RUN/protocol"
VIEWS="$PROTOCOL_DIR/training_visible_multiview_forget.json"

python -u scripts/run_mcf_rsnr_v1b_prehead_standard_unlearn.py \
  --model-path "$MODEL_PATH" \
  --protocol-dir "$PROTOCOL_DIR" \
  --view-corpus "$VIEWS" \
  --output-dir "$OUTPUT_DIR" \
  --seed 1 \
  --forget-num 50 \
  --dtype bf16 \
  --steps 1000 \
  --case-batch-size 4 \
  --check-every 25 \
  --learning-rate 2e-4 \
  --adapter-rank 16 \
  --adapter-alpha 16 \
  --cf-margin-weight 1.0 \
  --idk-margin-weight 1.0 \
  --true-suppression-weight 1.0 \
  --anchor-weight 1e-4 \
  --minimum-cf-margin 0.1 \
  --minimum-idk-vs-true-margin 0.1 \
  --minimum-true-logprob-drop 2.0 \
  --gate-off-logit-drift-max 0.0
