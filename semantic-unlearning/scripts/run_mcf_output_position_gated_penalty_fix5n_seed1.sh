#!/usr/bin/env bash
set -euo pipefail

: "${FIX5M_SOURCE_DIR:?Set FIX5M_SOURCE_DIR to the completed Fix5m output directory}"
: "${MODEL_PATH:?Set MODEL_PATH}"
: "${FIX5N_OUT_DIR:?Set FIX5N_OUT_DIR}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"

# Preserve previous attempts rather than deleting or overwriting them.
if [ -d "$FIX5N_OUT_DIR" ]; then
  ts="$(date +%Y%m%d_%H%M%S)"
  archive="${FIX5N_OUT_DIR}_previous_${ts}"
  n=1
  while [ -e "$archive" ]; do
    archive="${FIX5N_OUT_DIR}_previous_${ts}_${n}"
    n=$((n + 1))
  done
  mv "$FIX5N_OUT_DIR" "$archive"
  echo "[Fix5n] Existing output archived to: $archive"
fi

python scripts/mcf_output_position_gated_penalty_fix5n_seed1.py \
  --fix5m-output-dir "$FIX5M_SOURCE_DIR" \
  --model-path "$MODEL_PATH" \
  --output-dir "$FIX5N_OUT_DIR" \
  --dtype bf16 \
  --device cuda \
  --max-new-tokens "$MAX_NEW_TOKENS"
