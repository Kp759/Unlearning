#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the local Llama-3.2-3B-Instruct directory}"

ROOT_RESULTS="$PWD/results/retain_anchored_context_head/mcf"
TARGET_REPRESENTATION_OUT_DIR="${TARGET_REPRESENTATION_OUT_DIR:-$ROOT_RESULTS/seed1_target_representation_compare_fix5f}"
FIX5L_OUT_DIR="${FIX5L_OUT_DIR:-$ROOT_RESULTS/seed1_target_local_fixed_penalty_integration_fix5l}"
FIX5O_OUT_DIR="${FIX5O_OUT_DIR:-$ROOT_RESULTS/seed1_target_local_augmented_relation_router_fix5o}"
FIX5P_OUT_DIR="${FIX5P_OUT_DIR:-$ROOT_RESULTS/seed1_fix5o_matched_end_to_end_fix5p}"
COMPLETION_GUARD_OUT_DIR="${COMPLETION_GUARD_OUT_DIR:-$ROOT_RESULTS/seed1_fact_conditioned_completion_guard}"
RELATION_CONTRACTS="${RELATION_CONTRACTS:-$PWD/scripts/mcf_relation_contracts_fix5.json}"

required=(
  "$TARGET_REPRESENTATION_OUT_DIR/target_representation_feature_cache.pt"
  "$FIX5L_OUT_DIR/frozen_answer_token_support_fix5l.json"
  "$FIX5O_OUT_DIR/mcf_target_local_augmented_relation_router_fix5o.json"
  "$FIX5O_OUT_DIR/augmented_exact_name_linear_head.pt"
  "$FIX5P_OUT_DIR/mcf_fix5o_matched_end_to_end_fix5p.json"
  "$FIX5P_OUT_DIR/mcf_fix5o_matched_end_to_end_records_fix5p.jsonl"
  "$RELATION_CONTRACTS"
)
for path in "${required[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "[completion-guard] missing required artifact: $path" >&2
    exit 2
  fi
done

if [[ -e "$COMPLETION_GUARD_OUT_DIR" ]]; then
  stamp="$(date +%Y%m%d_%H%M%S)"
  archived="${COMPLETION_GUARD_OUT_DIR}_previous_${stamp}"
  n=1
  while [[ -e "$archived" ]]; do
    archived="${COMPLETION_GUARD_OUT_DIR}_previous_${stamp}_${n}"
    n=$((n + 1))
  done
  mv -- "$COMPLETION_GUARD_OUT_DIR" "$archived"
  echo "[completion-guard] Existing output archived to: $archived"
fi

python scripts/mcf_fact_conditioned_completion_guard_seed1.py \
  --fix5f-output-dir "$TARGET_REPRESENTATION_OUT_DIR" \
  --fix5l-output-dir "$FIX5L_OUT_DIR" \
  --fix5o-output-dir "$FIX5O_OUT_DIR" \
  --fix5p-output-dir "$FIX5P_OUT_DIR" \
  --model-path "$MODEL_PATH" \
  --relation-contracts "$RELATION_CONTRACTS" \
  --output-dir "$COMPLETION_GUARD_OUT_DIR" \
  --dtype "${DTYPE:-bf16}" \
  --device "${DEVICE:-cuda}" \
  --encode-batch-size "${ENCODE_BATCH_SIZE:-16}" \
  --verifier-batch-size "${VERIFIER_BATCH_SIZE:-16}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-64}" \
  --epsilon "${GUARD_EPSILON:-0.02}" \
  --min-calibration-forbidden-block "${GUARD_MIN_CALIB_BLOCK:-0.60}" \
  --min-validation-forbidden-block "${GUARD_MIN_VAL_BLOCK:-0.60}"
