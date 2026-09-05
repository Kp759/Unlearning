#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH}"
: "${MCF_PATH:?Set MCF_PATH}"
: "${VIEW_CORPUS:?Set VIEW_CORPUS to the leakage-safe five-view JSON}"

RECOG_OUT_DIR="${RECOG_OUT_DIR:-$PWD/results/retain_anchored_context_head/mcf/seed1_recognition_router_benchmark}"

python scripts/mcf_recognition_router_benchmark_seed1_fixed.py \
  --model-path "$MODEL_PATH" \
  --mcf-path "$MCF_PATH" \
  --view-corpus "$VIEW_CORPUS" \
  --output-dir "$RECOG_OUT_DIR" \
  --dtype "${DTYPE:-bf16}" \
  --device "${DEVICE:-cuda}" \
  --encode-batch-size "${ENCODE_BATCH_SIZE:-16}" \
  --epsilon-retain "${ROUTER_EPS_RETAIN:-0.02}" \
  --epsilon-wrong "${ROUTER_EPS_WRONG:-0.02}" \
  --min-calib-correct-accept "${ROUTER_MIN_CALIB_ACCEPT:-0.60}"
