#!/usr/bin/env bash
set -euo pipefail

FIX5L_SOURCE_DIR="${FIX5L_SOURCE_DIR:-${INTEGRATION_OUT_DIR:-}}"
: "${FIX5L_SOURCE_DIR:?Set FIX5L_SOURCE_DIR or INTEGRATION_OUT_DIR to the completed Fix5l output directory}"
: "${TYPED_TARGET_OUT_DIR:?Set TYPED_TARGET_OUT_DIR to the completed Fix5k output directory}"
: "${MODEL_PATH:?Set MODEL_PATH}"
: "${MCF_PATH:?Set MCF_PATH}"
: "${GENERATION_EVAL_OUT_DIR:?Set GENERATION_EVAL_OUT_DIR}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
ATOMIC_DIRECT_N="${ATOMIC_DIRECT_N:-50}"
ATOMIC_PARAPHRASE_N="${ATOMIC_PARAPHRASE_N:-100}"
MIXED_OVERLAP_PAIRS="${MIXED_OVERLAP_PAIRS:-20}"
MIXED_NONOVERLAP_PAIRS="${MIXED_NONOVERLAP_PAIRS:-20}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-16}"

python scripts/mcf_target_local_generation_mixed_eval_fix5m_guarded_seed1.py \
  --fix5l-output-dir "$FIX5L_SOURCE_DIR" \
  --fix5k-output-dir "$TYPED_TARGET_OUT_DIR" \
  --model-path "$MODEL_PATH" \
  --mcf-path "$MCF_PATH" \
  --output-dir "$GENERATION_EVAL_OUT_DIR" \
  --dtype bf16 \
  --device cuda \
  --encode-batch-size "$ENCODE_BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --atomic-direct-n "$ATOMIC_DIRECT_N" \
  --atomic-paraphrase-n "$ATOMIC_PARAPHRASE_N" \
  --mixed-overlap-pairs "$MIXED_OVERLAP_PAIRS" \
  --mixed-nonoverlap-pairs "$MIXED_NONOVERLAP_PAIRS"
