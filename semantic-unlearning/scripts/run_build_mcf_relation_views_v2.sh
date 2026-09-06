#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH}"
: "${FORGET_DIRECT:?Set FORGET_DIRECT to training_visible_forget_direct.json}"
: "${VIEW_CORPUS:?Set VIEW_CORPUS to the leakage-safe five-view training corpus}"

RELATION_V2_CORPUS="${RELATION_V2_CORPUS:-$PWD/outputs/mcf_relation_views_v2_seed1/relation_views_v2.json}"

python scripts/build_mcf_relation_views_v2_fixed.py \
  --model-path "$MODEL_PATH" \
  --forget-direct "$FORGET_DIRECT" \
  --out "$RELATION_V2_CORPUS" \
  --dtype "${DTYPE:-bf16}" \
  --min-equivalence-margin "${RELATION_EQ_MARGIN:-0.0}" \
  --max-jaccard-to-canonical "${RELATION_MAX_JACCARD:-0.82}"
