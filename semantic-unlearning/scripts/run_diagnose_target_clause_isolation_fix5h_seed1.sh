#!/usr/bin/env bash
set -euo pipefail

: "${TARGET_REPRESENTATION_OUT_DIR:?Set TARGET_REPRESENTATION_OUT_DIR to completed Fix5f output}"
: "${TARGET_QUERY_CALIB_OUT_DIR:?Set TARGET_QUERY_CALIB_OUT_DIR to completed Fix5g output}"
: "${MODEL_PATH:?Set MODEL_PATH}"

TARGET_CLAUSE_DIAG_OUT_DIR="${TARGET_CLAUSE_DIAG_OUT_DIR:-$PWD/results/retain_anchored_context_head/mcf/seed1_target_clause_isolation_fix5h}"

python scripts/diagnose_target_clause_isolation_fix5h_seed1.py \
  --fix5f-output-dir "$TARGET_REPRESENTATION_OUT_DIR" \
  --fix5g-report "$TARGET_QUERY_CALIB_OUT_DIR/target_representation_query_calibration_fix5g.json" \
  --model-path "$MODEL_PATH" \
  --output-dir "$TARGET_CLAUSE_DIAG_OUT_DIR" \
  --dtype "${DTYPE:-bf16}" \
  --device "${DEVICE:-cuda}" \
  --encode-batch-size "${ENCODE_BATCH_SIZE:-16}"
