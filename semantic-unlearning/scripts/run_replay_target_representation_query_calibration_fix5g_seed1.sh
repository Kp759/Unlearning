#!/usr/bin/env bash
set -euo pipefail

: "${TARGET_REPRESENTATION_OUT_DIR:?Set TARGET_REPRESENTATION_OUT_DIR to completed Fix5f output}"

TARGET_QUERY_CALIB_OUT_DIR="${TARGET_QUERY_CALIB_OUT_DIR:-$PWD/results/retain_anchored_context_head/mcf/seed1_target_representation_query_calibration_fix5g}"

python scripts/replay_target_representation_query_calibration_fix5g_seed1.py \
  --fix5f-output-dir "$TARGET_REPRESENTATION_OUT_DIR" \
  --output-dir "$TARGET_QUERY_CALIB_OUT_DIR" \
  --device "${QUERY_CALIB_DEVICE:-cpu}" \
  --epsilon-retain "${ROUTER_EPS_RETAIN:-0.02}" \
  --epsilon-wrong "${ROUTER_EPS_WRONG:-0.02}" \
  --min-calib-correct-accept "${ROUTER_MIN_CALIB_ACCEPT:-0.60}" \
  --min-validation-relation-accuracy "${RELATION_MIN_VAL_ACCURACY:-0.70}"
