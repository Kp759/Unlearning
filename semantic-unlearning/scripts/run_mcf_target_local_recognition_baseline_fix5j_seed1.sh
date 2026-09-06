#!/usr/bin/env bash
set -euo pipefail

: "${TARGET_REPRESENTATION_OUT_DIR:?Set TARGET_REPRESENTATION_OUT_DIR to the successful Fix5f output directory}"
: "${MODEL_PATH:?Set MODEL_PATH}"
: "${MCF_PATH:?Set MCF_PATH}"

TARGET_LOCAL_OUT_DIR="${TARGET_LOCAL_OUT_DIR:-$PWD/results/retain_anchored_context_head/mcf/seed1_target_local_recognition_fix5j}"

python scripts/mcf_target_local_recognition_baseline_fix5j_seed1.py \
  --fix5f-output-dir "$TARGET_REPRESENTATION_OUT_DIR" \
  --model-path "$MODEL_PATH" \
  --mcf-path "$MCF_PATH" \
  --output-dir "$TARGET_LOCAL_OUT_DIR" \
  --dtype "${DTYPE:-bf16}" \
  --device "${DEVICE:-cuda}" \
  --encode-batch-size "${ENCODE_BATCH_SIZE:-16}" \
  --epsilon-retain "${ROUTER_EPS_RETAIN:-0.02}" \
  --epsilon-wrong "${ROUTER_EPS_WRONG:-0.02}" \
  --min-calib-correct-accept "${ROUTER_MIN_CALIB_ACCEPT:-0.60}" \
  --min-validation-relation-accuracy "${RELATION_MIN_VAL_ACCURACY:-0.70}" \
  --mixed-queries-per-phase "${TARGET_LOCAL_MIXED_QUERIES:-50}"
