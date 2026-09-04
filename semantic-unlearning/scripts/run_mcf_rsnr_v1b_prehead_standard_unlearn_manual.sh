#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/ec2-user/models/Llama-3.2-3B-Instruct}"
SOURCE_V13_RUN="${SOURCE_V13_RUN:-$PWD/outputs/mcf_private_vocab_rewiring_v1_3_multiview_seed1_3b}"
VIEWS="${VIEWS:-$SOURCE_V13_RUN/protocol/training_visible_multiview_forget.json}"
OUT="${1:-$PWD/outputs/mcf_rsnr_v1b_prehead_standard_unlearn_seed1_3b}"

python -u scripts/run_mcf_rsnr_v1b_prehead_standard_unlearn.py \
  --model-path "$MODEL_PATH" \
  --protocol-dir "$SOURCE_V13_RUN/protocol" \
  --view-corpus "$VIEWS" \
  --output-dir "$OUT" \
  --seed 1 \
  --forget-num 50 \
  --dtype bf16 \
  --steps 1200 \
  --case-batch-size 4 \
  --check-every 25 \
  --learning-rate 2e-4 \
  --weight-decay 0 \
  --adapter-rank 16 \
  --adapter-alpha 16 \
  --ga-weight 0.25 \
  --abstain-weight 1.0 \
  --cf-margin-weight 1.0 \
  --idk-margin-weight 1.0 \
  --drop-weight 1.0 \
  --anchor-weight 1e-4 \
  --grad-clip 1.0 \
  --minimum-cf-margin 0.1 \
  --minimum-idk-margin 0.1 \
  --minimum-true-logprob-drop 2.0 \
  --gate-off-logit-drift-max 0.0
