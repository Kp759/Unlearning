#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH}"
: "${MCF_PATH:?Set MCF_PATH}"
: "${RELATION_V2_CORPUS:?Set RELATION_V2_CORPUS to relation_views_v2.json}"

RELATION_V2_OUT_DIR="${RELATION_V2_OUT_DIR:-$PWD/results/retain_anchored_context_head/mcf/seed1_subject_relation_views_v2_recognition}"

python scripts/mcf_subject_relation_views_v2_seed1.py \
  --model-path "$MODEL_PATH" \
  --mcf-path "$MCF_PATH" \
  --view-corpus-v2 "$RELATION_V2_CORPUS" \
  --output-dir "$RELATION_V2_OUT_DIR" \
  --dtype "${DTYPE:-bf16}" \
  --device "${DEVICE:-cuda}" \
  --encode-batch-size "${ENCODE_BATCH_SIZE:-16}" \
  --epsilon-retain "${ROUTER_EPS_RETAIN:-0.02}" \
  --epsilon-wrong "${ROUTER_EPS_WRONG:-0.02}" \
  --min-calib-correct-accept "${ROUTER_MIN_CALIB_ACCEPT:-0.60}"
