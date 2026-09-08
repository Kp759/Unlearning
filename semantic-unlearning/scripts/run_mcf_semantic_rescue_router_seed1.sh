#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH}"

FIX5F_OUT_DIR="${FIX5F_OUT_DIR:-$PWD/results/retain_anchored_context_head/mcf/seed1_target_representation_compare_fix5f}"
FIX5L_OUT_DIR="${FIX5L_OUT_DIR:-$PWD/results/retain_anchored_context_head/mcf/seed1_target_local_fixed_penalty_integration_fix5l}"
FIX5O_OUT_DIR="${FIX5O_OUT_DIR:-$PWD/results/retain_anchored_context_head/mcf/seed1_target_local_augmented_relation_router_fix5o}"
FIX5P_OUT_DIR="${FIX5P_OUT_DIR:-$PWD/results/retain_anchored_context_head/mcf/seed1_fix5o_matched_end_to_end_fix5p}"
SEMANTIC_RESCUE_OUT_DIR="${SEMANTIC_RESCUE_OUT_DIR:-$PWD/results/retain_anchored_context_head/mcf/seed1_semantic_rescue_router}"
RELATION_CONTRACTS="${RELATION_CONTRACTS:-$PWD/scripts/mcf_relation_contracts_fix5.json}"

for f in \
  "$FIX5F_OUT_DIR/target_representation_feature_cache.pt" \
  "$FIX5L_OUT_DIR/frozen_answer_token_support_fix5l.json" \
  "$FIX5O_OUT_DIR/mcf_target_local_augmented_relation_router_fix5o.json" \
  "$FIX5O_OUT_DIR/augmented_exact_name_linear_head.pt" \
  "$FIX5O_OUT_DIR/mcf_target_local_augmented_relation_router_records_fix5o.jsonl" \
  "$FIX5P_OUT_DIR/mcf_fix5o_matched_end_to_end_fix5p.json" \
  "$FIX5P_OUT_DIR/mcf_fix5o_matched_end_to_end_records_fix5p.jsonl" \
  "$RELATION_CONTRACTS"
do
  if [[ ! -f "$f" ]]; then
    echo "[semantic-rescue] Missing required file: $f" >&2
    exit 2
  fi
done

if [[ -e "$SEMANTIC_RESCUE_OUT_DIR" ]]; then
  stamp="$(date +%Y%m%d_%H%M%S)"
  archived="${SEMANTIC_RESCUE_OUT_DIR}_previous_${stamp}"
  n=1
  while [[ -e "$archived" ]]; do
    archived="${SEMANTIC_RESCUE_OUT_DIR}_previous_${stamp}_${n}"
    n=$((n + 1))
  done
  mv -- "$SEMANTIC_RESCUE_OUT_DIR" "$archived"
  echo "[semantic-rescue] Existing output archived to: $archived"
fi

python scripts/mcf_semantic_rescue_router_seed1.py \
  --fix5f-output-dir "$FIX5F_OUT_DIR" \
  --fix5l-output-dir "$FIX5L_OUT_DIR" \
  --fix5o-output-dir "$FIX5O_OUT_DIR" \
  --fix5p-output-dir "$FIX5P_OUT_DIR" \
  --model-path "$MODEL_PATH" \
  --relation-contracts "$RELATION_CONTRACTS" \
  --output-dir "$SEMANTIC_RESCUE_OUT_DIR" \
  --dtype "${DTYPE:-bf16}" \
  --device "${DEVICE:-cuda}" \
  --encode-batch-size "${ENCODE_BATCH_SIZE:-16}" \
  --verifier-batch-size "${SEMANTIC_VERIFIER_BATCH_SIZE:-16}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-64}" \
  --epsilon-retain "${ROUTER_EPS_RETAIN:-0.02}" \
  --mixed-queries-per-phase "${MIXED_QUERIES_PER_PHASE:-50}"
