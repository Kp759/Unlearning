#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${1:?Usage: bash scripts/run_mcf_sure_retain_residualized_stage1.sh MODEL [MCF_JSON]}"
MCF="${2:-data/multi_counterfact.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mcf_sure_retain_residualized_stage1_v1}"
SEEDS_TEXT="${MCF_SEEDS:-1}"
FORGET_NUM="${MCF_FORGET_NUM:-50}"
RETAIN_TRAIN_NUM="${SURE_RETAIN_TRAIN_NUM:-1000}"
RETAIN_EVAL_NUM="${MCF_RETAIN_EVAL_NUM:-1000}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"

# Frozen diagnostic settings from the successful projected Stage-1 run.
STEPS="${SURE_STAGE1_STEPS:-600}"
BATCH_SIZE="${SURE_STAGE1_BATCH_SIZE:-1}"
CACHE_BATCH_SIZE="${SURE_STAGE1_CACHE_BATCH_SIZE:-8}"
LR="${SURE_STAGE1_LR:-0.005}"
GA_WEIGHT="${SURE_GA_WEIGHT:-2.0}"
GD_WEIGHT="${SURE_GD_WEIGHT:-1.0}"
STAGE1_L2="${SURE_STAGE1_DELTA_L2:-0.0}"
CONTEXT_RANK="${SURE_STAGE1_CONTEXT_RANK:-2}"
RETAIN_RANK="${SURE_RETAIN_SUBSPACE_RANK:-64}"
SHARED_MARGIN="${SURE_SHARED_CONSTRAINT_MARGIN:-0.05}"
MIN_NLL_INCREASE="${SURE_MIN_SENSITIVE_NLL_INCREASE:-4.0}"
CANDIDATE_SCALES="${SURE_CANDIDATE_SCALES:-1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,.046875,.03125,.015625,.0078125,0}"

read -r -a SEEDS <<< "${SEEDS_TEXT}"
test -f "${MCF}"
test -d "${MODEL}"
test -d "${WIKIDATA_DIR}"

for SEED in "${SEEDS[@]}"; do
  ROOT="${OUTPUT_ROOT}/seed${SEED}"
  PROTOCOL="${ROOT}/protocol"
  FORGET_VISIBLE="${PROTOCOL}/training_visible_forget.json"
  RETAIN_VISIBLE="${PROTOCOL}/training_visible_retain.json"
  MANIFEST="${PROTOCOL}/split_manifest.json"
  STAGE1="${ROOT}/stage1_retain_residualized"
  BASE_EVAL="${ROOT}/base_original_mcf_eval.json"
  STAGE1_EVAL="${ROOT}/stage1_original_mcf_eval.json"
  PAPER_EVAL="${ROOT}/stage1_target_true_sensitive_eval.json"

  rm -rf "${ROOT}"
  mkdir -p "${ROOT}"

  echo "===== MCF SEED ${SEED}: LEAKAGE-SAFE FORGET/RETAIN SPLIT ====="
  python scripts/build_mcf_sure_fixed_shared_retain_split.py \
    --mcf-path "${MCF}" \
    --output-dir "${PROTOCOL}" \
    --seed "${SEED}" \
    --forget-num "${FORGET_NUM}" \
    --retain-train-num "${RETAIN_TRAIN_NUM}" \
    --retain-eval-num "${RETAIN_EVAL_NUM}"

  echo "===== MCF SEED ${SEED}: RETAIN-RESIDUALIZED STAGE 1 ====="
  python scripts/sure_stage1_context_retain_residualized.py \
    --dataset mcf \
    --model-path "${MODEL}" \
    --training-visible-path "${FORGET_VISIBLE}" \
    --training-visible-retain-path "${RETAIN_VISIBLE}" \
    --split-manifest "${MANIFEST}" \
    --output-dir "${STAGE1}" \
    --seed "${SEED}" \
    --forget-num "${FORGET_NUM}" \
    --retain-train-num "${RETAIN_TRAIN_NUM}" \
    --steps "${STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --cache-batch-size "${CACHE_BATCH_SIZE}" \
    --lr "${LR}" \
    --ga-weight "${GA_WEIGHT}" \
    --gd-weight "${GD_WEIGHT}" \
    --delta-l2 "${STAGE1_L2}" \
    --context-rank "${CONTEXT_RANK}" \
    --retain-rank "${RETAIN_RANK}" \
    --constraint-margin "${SHARED_MARGIN}" \
    --min-sensitive-nll-increase "${MIN_NLL_INCREASE}" \
    --candidate-scales "${CANDIDATE_SCALES}" \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}"

  echo "===== MCF SEED ${SEED}: BASE / STAGE-1 ORIGINAL UNSWAPPED EVAL ====="
  python scripts/mcf_zero_unlearn_official_eval.py \
    --model-dir "${MODEL}" \
    --mcf-path "${MCF}" \
    --wikidata-dir "${WIKIDATA_DIR}" \
    --out "${BASE_EVAL}" \
    --unlearn-num "${FORGET_NUM}" \
    --retain-num "${RETAIN_EVAL_NUM}" \
    --seed "${SEED}" \
    --sample-mode official \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}"

  python scripts/mcf_zero_unlearn_official_eval.py \
    --model-dir "${STAGE1}/checkpoint" \
    --mcf-path "${MCF}" \
    --wikidata-dir "${WIKIDATA_DIR}" \
    --out "${STAGE1_EVAL}" \
    --unlearn-num "${FORGET_NUM}" \
    --retain-num "${RETAIN_EVAL_NUM}" \
    --seed "${SEED}" \
    --sample-mode official \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}"

  python scripts/annotate_ppl_provenance.py \
    --eval-json "${BASE_EVAL}" \
    --model-dir "${MODEL}" \
    --wikidata-dir "${WIKIDATA_DIR}"

  python scripts/annotate_ppl_provenance.py \
    --eval-json "${STAGE1_EVAL}" \
    --model-dir "${STAGE1}/checkpoint" \
    --wikidata-dir "${WIKIDATA_DIR}"

  python scripts/evaluate_mcf_target_true_sensitive.py \
    --base-eval-json "${BASE_EVAL}" \
    --post-eval-json "${STAGE1_EVAL}" \
    --split-manifest "${MANIFEST}" \
    --out "${PAPER_EVAL}"

  echo "===== MCF SEED ${SEED}: RESIDUALIZATION DIAGNOSTICS ====="
  python - "${STAGE1}/config_used.json" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
r = d["residualization_diagnostics"]
print("retain_basis_rank_actual:", d["retain_basis_rank_actual"])
print("trainable_parameters:", d["trainable_parameters"])
print("selected_scale:", d["selected_scale"])
print("final_direct_failures:", d["final_direct_failures"])
print("min_final_margin:", d["minimum_final_suppression_margin"])
print("min_final_sensitive_dNLL:", d["minimum_final_sensitive_nll_increase"])
print("mean_residual_norm_ratio:", r["mean_residual_norm_ratio"])
print("median_residual_norm_ratio:", r["median_residual_norm_ratio"])
print("mean_retain_projection_norm_ratio:", r["mean_retain_projection_norm_ratio"])
print("max_abs_row_basis_dot_retain_basis:", r["max_abs_row_basis_dot_retain_basis"])
PY

done

echo "MCF retain-residualized Stage-1 run complete: ${OUTPUT_ROOT}"
