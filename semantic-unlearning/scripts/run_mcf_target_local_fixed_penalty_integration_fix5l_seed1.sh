#!/usr/bin/env bash
set -euo pipefail

: "${TYPED_TARGET_OUT_DIR:?Set TYPED_TARGET_OUT_DIR to the completed Fix5k output directory}"
: "${MODEL_PATH:?Set MODEL_PATH}"
: "${MCF_PATH:?Set MCF_PATH}"

INTEGRATION_OUT_DIR="${INTEGRATION_OUT_DIR:-$PWD/results/retain_anchored_context_head/mcf/seed1_target_local_fixed_penalty_integration_fix5l}"

python scripts/mcf_target_local_fixed_penalty_integration_fix5l_seed1.py \
  --fix5k-output-dir "$TYPED_TARGET_OUT_DIR" \
  --model-path "$MODEL_PATH" \
  --mcf-path "$MCF_PATH" \
  --output-dir "$INTEGRATION_OUT_DIR" \
  --penalty "${FIXED_LOGIT_PENALTY:-12.0}" \
  --dtype "${DTYPE:-bf16}" \
  --device "${DEVICE:-cuda}" \
  --encode-batch-size "${ENCODE_BATCH_SIZE:-16}" \
  --retain-eval-n "${RETAIN_EVAL_N:-100}" \
  --overlap-stress-n "${OVERLAP_STRESS_N:-50}"
