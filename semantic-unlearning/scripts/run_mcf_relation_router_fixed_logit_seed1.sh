#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH}"
: "${MCF_PATH:?Set MCF_PATH}"
: "${WIKIDATA_DIR:?Set WIKIDATA_DIR}"
: "${METHOD5_DIR:?Set METHOD5_DIR to the completed Method-5 Seed-1 output directory}"
: "${VIEW_CORPUS:?Set VIEW_CORPUS to the leakage-safe five-view training corpus JSON}"

OUT_DIR="${OUT_DIR:-$PWD/results/retain_anchored_context_head/mcf/seed1_relation_router_fixed_logit}"
BASE_EVAL="${BASE_EVAL:-$METHOD5_DIR/base_official_eval.json}"
SIDECAR="${METHOD5_SIDECAR:-$METHOD5_DIR/method5_sidecar.pt}"

python scripts/mcf_relation_router_fixed_logit_seed1.py \
  --model-path "$MODEL_PATH" \
  --mcf-path "$MCF_PATH" \
  --wikidata-dir "$WIKIDATA_DIR" \
  --view-corpus "$VIEW_CORPUS" \
  --method5-sidecar "$SIDECAR" \
  --base-eval-json "$BASE_EVAL" \
  --output-dir "$OUT_DIR" \
  --seed 1 \
  --forget-num 50 \
  --retain-num 1000 \
  --dtype "${DTYPE:-bf16}" \
  --device "${DEVICE:-cuda}" \
  --encode-batch-size "${ENCODE_BATCH_SIZE:-16}" \
  --minimum-calibration-recall "${MIN_CALIBRATION_RECALL:-0.98}"
