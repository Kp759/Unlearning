#!/usr/bin/env bash
set -euo pipefail
: "${MODEL_PATH:?Set MODEL_PATH}"
MCF_PATH="${MCF_PATH:-$PWD/data/multi_counterfact.json}"
FIX5L_OUT_DIR="${FIX5L_OUT_DIR:-$PWD/results/retain_anchored_context_head/mcf/seed1_target_local_fixed_penalty_integration_fix5l}"
FIX5P_OUT_DIR="${FIX5P_OUT_DIR:-$PWD/results/retain_anchored_context_head/mcf/seed1_fix5o_matched_end_to_end_fix5p}"
OUT_DIR="${SUBJECT_HARD_OUT_DIR:-$PWD/results/retain_anchored_context_head/mcf/seed1_subject_scoped_hard_guard}"
FIX5P_RECORDS="$FIX5P_OUT_DIR/mcf_fix5o_matched_end_to_end_records_fix5p.jsonl"
for f in "$MODEL_PATH/config.json" "$MCF_PATH" "$FIX5L_OUT_DIR/frozen_answer_token_support_fix5l.json" "$FIX5P_RECORDS"; do
  [[ -e "$f" ]] || { echo "Missing required input: $f" >&2; exit 2; }
done
if [[ -e "$OUT_DIR" ]]; then
  stamp="$(date +%Y%m%d_%H%M%S)"; archived="${OUT_DIR}_previous_${stamp}"; n=1
  while [[ -e "$archived" ]]; do archived="${OUT_DIR}_previous_${stamp}_${n}"; n=$((n+1)); done
  mv -- "$OUT_DIR" "$archived"; echo "[subject-hard] Existing output archived to: $archived"
fi
python scripts/mcf_subject_scoped_hard_guard_seed1.py \
  --fix5l-output-dir "$FIX5L_OUT_DIR" \
  --fix5p-records "$FIX5P_RECORDS" \
  --model-path "$MODEL_PATH" \
  --mcf-path "$MCF_PATH" \
  --output-dir "$OUT_DIR" \
  --dtype "${DTYPE:-bf16}" \
  --device "${DEVICE:-cuda}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-64}"
