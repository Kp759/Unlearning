#!/usr/bin/env bash
set -euo pipefail

: "${TARGET_REPRESENTATION_OUT_DIR:?Set TARGET_REPRESENTATION_OUT_DIR to the completed Fix5f output directory}"
: "${MODEL_PATH:?Set MODEL_PATH}"
: "${MCF_PATH:?Set MCF_PATH}"

FIX5O_OUT_DIR="${FIX5O_OUT_DIR:-$PWD/results/retain_anchored_context_head/mcf/seed1_target_local_augmented_relation_router_fix5o}"
RELATION_CONTRACTS="${RELATION_CONTRACTS:-$PWD/scripts/mcf_relation_contracts_fix5.json}"

if [ -d "$FIX5O_OUT_DIR" ]; then
  ts="$(date +%Y%m%d_%H%M%S)"
  archive="${FIX5O_OUT_DIR}_previous_${ts}"
  n=1
  while [ -e "$archive" ]; do
    archive="${FIX5O_OUT_DIR}_previous_${ts}_${n}"
    n=$((n + 1))
  done
  mv "$FIX5O_OUT_DIR" "$archive"
  echo "[Fix5o] Existing output archived to: $archive"
fi

python scripts/mcf_target_local_augmented_relation_router_fix5o_v2_seed1.py \
  --fix5f-output-dir "$TARGET_REPRESENTATION_OUT_DIR" \
  --model-path "$MODEL_PATH" \
  --mcf-path "$MCF_PATH" \
  --relation-contracts "$RELATION_CONTRACTS" \
  --output-dir "$FIX5O_OUT_DIR" \
  --dtype "${DTYPE:-bf16}" \
  --device "${DEVICE:-cuda}" \
  --encode-batch-size "${ENCODE_BATCH_SIZE:-16}" \
  --train-steps "${RELATION_TRAIN_STEPS:-1600}" \
  --train-batch-size "${RELATION_TRAIN_BATCH:-128}" \
  --lr "${RELATION_LR:-0.005}" \
  --weight-decay "${RELATION_WEIGHT_DECAY:-0.0001}" \
  --head-seed "${RELATION_HEAD_SEED:-1}" \
  --epsilon-retain "${ROUTER_EPS_RETAIN:-0.02}" \
  --epsilon-wrong "${ROUTER_EPS_WRONG:-0.02}" \
  --min-calib-correct-accept "${ROUTER_MIN_CALIB_ACCEPT:-0.60}" \
  --min-validation-relation-accuracy "${RELATION_MIN_VAL_ACCURACY:-0.70}" \
  --hard-negatives-per-fact "${FIX5O_HARD_NEGATIVES_PER_FACT:-2}" \
  --max-heldout-jaccard "${FIX5O_MAX_HELDOUT_JACCARD:-0.90}" \
  --mixed-queries-per-phase "${MIXED_QUERIES_PER_PHASE:-50}"
