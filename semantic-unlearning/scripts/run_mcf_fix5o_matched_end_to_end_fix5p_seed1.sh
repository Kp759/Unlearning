#!/usr/bin/env bash
set -euo pipefail

ROOT_RESULTS="$PWD/results/retain_anchored_context_head/mcf"
FIX5L_SOURCE_DIR="${FIX5L_SOURCE_DIR:-$ROOT_RESULTS/seed1_target_local_fixed_penalty_integration_fix5l}"
TYPED_TARGET_OUT_DIR="${TYPED_TARGET_OUT_DIR:-$ROOT_RESULTS/seed1_target_local_typed_masking_fix5k}"
FIX5O_OUT_DIR="${FIX5O_OUT_DIR:-$ROOT_RESULTS/seed1_target_local_augmented_relation_router_fix5o}"
FIX5M_RECORDS="${FIX5M_RECORDS:-$ROOT_RESULTS/seed1_target_local_generation_mixed_eval_fix5m/mcf_target_local_generation_mixed_records_fix5m.jsonl}"
FIX5P_OUT_DIR="${FIX5P_OUT_DIR:-$ROOT_RESULTS/seed1_fix5o_matched_end_to_end_fix5p}"

: "${MODEL_PATH:?Set MODEL_PATH}"
: "${MCF_PATH:?Set MCF_PATH}"

required=(
  "$FIX5L_SOURCE_DIR/mcf_target_local_fixed_penalty_integration_fix5l.json"
  "$FIX5L_SOURCE_DIR/frozen_answer_token_support_fix5l.json"
  "$TYPED_TARGET_OUT_DIR/exact_name_target_local_linear_head.pt"
  "$FIX5O_OUT_DIR/mcf_target_local_augmented_relation_router_fix5o.json"
  "$FIX5O_OUT_DIR/augmented_exact_name_linear_head.pt"
  "$FIX5M_RECORDS"
  "$MODEL_PATH/config.json"
  "$MCF_PATH"
)
for p in "${required[@]}"; do
  if [[ ! -f "$p" ]]; then
    echo "[Fix5p] Required file missing: $p" >&2
    exit 2
  fi
done

if [[ -e "$FIX5P_OUT_DIR" ]]; then
  stamp="$(date +%Y%m%d_%H%M%S)"
  archived="${FIX5P_OUT_DIR}_previous_${stamp}"
  n=1
  while [[ -e "$archived" ]]; do
    archived="${FIX5P_OUT_DIR}_previous_${stamp}_${n}"
    n=$((n + 1))
  done
  mv -- "$FIX5P_OUT_DIR" "$archived"
  echo "[Fix5p] Existing output archived to: $archived"
fi

python scripts/mcf_fix5o_matched_end_to_end_fix5p_seed1.py \
  --fix5l-output-dir "$FIX5L_SOURCE_DIR" \
  --fix5k-output-dir "$TYPED_TARGET_OUT_DIR" \
  --fix5o-output-dir "$FIX5O_OUT_DIR" \
  --model-path "$MODEL_PATH" \
  --mcf-path "$MCF_PATH" \
  --fix5m-records "$FIX5M_RECORDS" \
  --output-dir "$FIX5P_OUT_DIR" \
  --dtype "${DTYPE:-bf16}" \
  --device "${DEVICE:-cuda}" \
  --encode-batch-size "${ENCODE_BATCH_SIZE:-16}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-64}" \
  --atomic-direct-n "${ATOMIC_DIRECT_N:-50}" \
  --atomic-paraphrase-n "${ATOMIC_PARAPHRASE_N:-100}"
