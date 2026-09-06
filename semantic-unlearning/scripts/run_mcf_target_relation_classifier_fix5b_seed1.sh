#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH}"
: "${MCF_PATH:?Set MCF_PATH}"
: "${RELATION_V2_CORPUS:?Set RELATION_V2_CORPUS to relation_views_v2_fix5.json}"

TARGET_RELATION_OUT_DIR="${TARGET_RELATION_OUT_DIR:-$PWD/results/retain_anchored_context_head/mcf/seed1_target_relation_classifier_fix5b_recognition}"

python scripts/mcf_target_relation_classifier_fix5b_seed1.py \
  --model-path "$MODEL_PATH" \
  --mcf-path "$MCF_PATH" \
  --view-corpus-fix5 "$RELATION_V2_CORPUS" \
  --output-dir "$TARGET_RELATION_OUT_DIR" \
  --dtype "${DTYPE:-bf16}" \
  --device "${DEVICE:-cuda}" \
  --encode-batch-size "${ENCODE_BATCH_SIZE:-16}" \
  --train-steps "${RELATION_TRAIN_STEPS:-1600}" \
  --train-batch-size "${RELATION_TRAIN_BATCH:-128}" \
  --lr "${RELATION_LR:-0.005}" \
  --weight-decay "${RELATION_WEIGHT_DECAY:-0.0001}" \
  --epsilon-retain "${ROUTER_EPS_RETAIN:-0.02}" \
  --epsilon-wrong "${ROUTER_EPS_WRONG:-0.02}" \
  --min-calib-correct-accept "${ROUTER_MIN_CALIB_ACCEPT:-0.60}" \
  --min-validation-relation-accuracy "${RELATION_MIN_VAL_ACCURACY:-0.70}"
