#!/usr/bin/env bash
set -euo pipefail

: "${TARGET_REPRESENTATION_OUT_DIR:?Set TARGET_REPRESENTATION_OUT_DIR to the completed Fix5f output directory}"
: "${MODEL_PATH:?Set MODEL_PATH}"
: "${MCF_PATH:?Set MCF_PATH}"

TYPED_TARGET_OUT_DIR="${TYPED_TARGET_OUT_DIR:-$PWD/results/retain_anchored_context_head/mcf/seed1_target_local_typed_masking_fix5k}"

python scripts/mcf_target_local_typed_masking_ablation_fix5k_seed1.py \
  --fix5f-output-dir "$TARGET_REPRESENTATION_OUT_DIR" \
  --model-path "$MODEL_PATH" \
  --mcf-path "$MCF_PATH" \
  --output-dir "$TYPED_TARGET_OUT_DIR" \
  --dtype "${DTYPE:-bf16}" \
  --device "${DEVICE:-cuda}" \
  --encode-batch-size "${ENCODE_BATCH_SIZE:-16}" \
  --type-batch-size "${TYPE_BATCH_SIZE:-32}" \
  --train-steps "${RELATION_TRAIN_STEPS:-1600}" \
  --train-batch-size "${RELATION_TRAIN_BATCH:-128}" \
  --lr "${RELATION_LR:-0.005}" \
  --weight-decay "${RELATION_WEIGHT_DECAY:-0.0001}" \
  --head-seed "${RELATION_HEAD_SEED:-1}" \
  --epsilon-retain "${ROUTER_EPS_RETAIN:-0.02}" \
  --epsilon-wrong "${ROUTER_EPS_WRONG:-0.02}" \
  --min-calib-correct-accept "${ROUTER_MIN_CALIB_ACCEPT:-0.60}" \
  --min-validation-relation-accuracy "${RELATION_MIN_VAL_ACCURACY:-0.70}" \
  --mixed-queries-per-phase "${MIXED_QUERIES_PER_PHASE:-50}"
