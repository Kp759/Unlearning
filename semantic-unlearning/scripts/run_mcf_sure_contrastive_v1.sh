#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${1:?Usage: bash scripts/run_mcf_sure_contrastive_v1.sh MODEL [MCF_JSON]}"
MCF="${2:-data/multi_counterfact.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mcf_sure_contrastive_v1}"
SEEDS_TEXT="${MCF_SEEDS:-1}"
FORGET_NUM="${MCF_FORGET_NUM:-50}"
RETAIN_TRAIN_NUM="${SURE_RETAIN_TRAIN_NUM:-1000}"
RETAIN_EVAL_NUM="${MCF_RETAIN_EVAL_NUM:-1000}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"

STAGE1_STEPS="${SURE_STAGE1_STEPS:-600}"
STAGE1_BATCH_SIZE="${SURE_STAGE1_BATCH_SIZE:-1}"
RETAIN_BATCH_SIZE="${SURE_RETAIN_BATCH_SIZE:-64}"
CACHE_BATCH_SIZE="${SURE_CACHE_BATCH_SIZE:-8}"
STAGE1_LR="${SURE_STAGE1_LR:-0.005}"
GA_WEIGHT="${SURE_GA_WEIGHT:-2.0}"
GD_WEIGHT="${SURE_GD_WEIGHT:-1.0}"
RETAIN_ACTION_WEIGHT="${SURE_RETAIN_ACTION_WEIGHT:-1.0}"
DELTA_L2="${SURE_STAGE1_DELTA_L2:-0.0}"
CONTRASTIVE_RANK="${SURE_CONTRASTIVE_RANK:-2}"
CONTRASTIVE_EPS="${SURE_CONTRASTIVE_EPS:-0.001}"
RETAIN_WEIGHT_CLIP="${SURE_RETAIN_WEIGHT_CLIP:-10.0}"

MARGIN="${SURE_SHARED_CONSTRAINT_MARGIN:-0.05}"
MIN_NLL="${SURE_MIN_SENSITIVE_NLL_INCREASE:-4.0}"
RETAIN_ACTION_BUDGET="${SURE_RETAIN_ACTION_BUDGET:-0.25}"
MAX_TOTAL_DELTA_NORM="${SURE_MAX_TOTAL_DELTA_NORM:-1.5}"
MAX_FORGET_NONSENSITIVE_KL="${SURE_MAX_FORGET_NONSENSITIVE_KL:-0.25}"
CANDIDATE_SCALES="${SURE_CANDIDATE_SCALES:-1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,.046875,.03125,.015625,.0078125,0}"

STAGE2_RANKS="${SURE_STAGE2_RANKS:-2,4}"
STAGE2_STEPS="${SURE_STAGE2_STEPS:-500}"
STAGE2_LR="${SURE_STAGE2_LR:-0.005}"
STAGE2_BATCH_SIZE="${SURE_STAGE2_BATCH_SIZE:-8}"
STAGE2_CHECK_EVERY="${SURE_STAGE2_CHECK_EVERY:-25}"
STAGE2_L2="${SURE_STAGE2_L2:-1e-6}"

read -r -a SEEDS <<< "${SEEDS_TEXT}"
test -d "${MODEL}"
test -f "${MCF}"
test -d "${WIKIDATA_DIR}"

for SEED in "${SEEDS[@]}"; do
  ROOT="${OUTPUT_ROOT}/seed${SEED}"
  PROTOCOL="${ROOT}/protocol"
  FORGET_VISIBLE="${PROTOCOL}/training_visible_forget.json"
  RETAIN_VISIBLE="${PROTOCOL}/training_visible_retain.json"
  MANIFEST="${PROTOCOL}/split_manifest.json"
  LEARNER="${ROOT}/contrastive_two_stage"
  BASE_EVAL="${ROOT}/base_original_mcf_eval.json"
  FINAL_EVAL="${ROOT}/final_original_mcf_eval.json"
  PAPER_EVAL="${ROOT}/final_target_true_sensitive_eval.json"

  rm -rf "${ROOT}"
  mkdir -p "${ROOT}"

  echo "===== MCF SEED ${SEED}: LEAKAGE-SAFE SPLIT ====="
  python scripts/build_mcf_sure_fixed_shared_retain_split.py \
    --mcf-path "${MCF}" \
    --output-dir "${PROTOCOL}" \
    --seed "${SEED}" \
    --forget-num "${FORGET_NUM}" \
    --retain-train-num "${RETAIN_TRAIN_NUM}" \
    --retain-eval-num "${RETAIN_EVAL_NUM}"

  echo "===== MCF SEED ${SEED}: CONTRASTIVE TWO-STAGE SURE-LM ====="
  python scripts/sure_contrastive_two_stage.py \
    --dataset mcf \
    --model-path "${MODEL}" \
    --training-visible-path "${FORGET_VISIBLE}" \
    --training-visible-retain-path "${RETAIN_VISIBLE}" \
    --split-manifest "${MANIFEST}" \
    --output-dir "${LEARNER}" \
    --seed "${SEED}" \
    --forget-num "${FORGET_NUM}" \
    --retain-train-num "${RETAIN_TRAIN_NUM}" \
    --stage1-steps "${STAGE1_STEPS}" \
    --stage1-batch-size "${STAGE1_BATCH_SIZE}" \
    --retain-batch-size "${RETAIN_BATCH_SIZE}" \
    --cache-batch-size "${CACHE_BATCH_SIZE}" \
    --stage1-lr "${STAGE1_LR}" \
    --ga-weight "${GA_WEIGHT}" \
    --gd-weight "${GD_WEIGHT}" \
    --retain-action-weight "${RETAIN_ACTION_WEIGHT}" \
    --delta-l2 "${DELTA_L2}" \
    --contrastive-rank "${CONTRASTIVE_RANK}" \
    --contrastive-eps "${CONTRASTIVE_EPS}" \
    --retain-weight-clip "${RETAIN_WEIGHT_CLIP}" \
    --constraint-margin "${MARGIN}" \
    --min-sensitive-nll-increase "${MIN_NLL}" \
    --retain-action-budget "${RETAIN_ACTION_BUDGET}" \
    --max-total-delta-norm "${MAX_TOTAL_DELTA_NORM}" \
    --max-forget-nonsensitive-kl "${MAX_FORGET_NONSENSITIVE_KL}" \
    --candidate-scales "${CANDIDATE_SCALES}" \
    --stage2-ranks "${STAGE2_RANKS}" \
    --stage2-steps "${STAGE2_STEPS}" \
    --stage2-lr "${STAGE2_LR}" \
    --stage2-batch-size "${STAGE2_BATCH_SIZE}" \
    --stage2-check-every "${STAGE2_CHECK_EVERY}" \
    --stage2-l2 "${STAGE2_L2}" \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}"

  echo "===== MCF SEED ${SEED}: LOCKED OFFICIAL EVALUATION (BASE + FINAL ONCE) ====="
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
    --model-dir "${LEARNER}/checkpoint" \
    --mcf-path "${MCF}" \
    --wikidata-dir "${WIKIDATA_DIR}" \
    --out "${FINAL_EVAL}" \
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
    --eval-json "${FINAL_EVAL}" \
    --model-dir "${LEARNER}/checkpoint" \
    --wikidata-dir "${WIKIDATA_DIR}"

  python scripts/evaluate_mcf_target_true_sensitive.py \
    --base-eval-json "${BASE_EVAL}" \
    --post-eval-json "${FINAL_EVAL}" \
    --split-manifest "${MANIFEST}" \
    --out "${PAPER_EVAL}"

  echo "===== MCF SEED ${SEED}: CONTRASTIVE TRAINING DIAGNOSTICS ====="
  python - "${LEARNER}/config_used.json" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
for k in [
    "stage1_selected_scale",
    "stage1_selection_mode",
    "stage1_direct_failures_after_materialization",
    "stage2_mode",
    "final_direct_failures",
    "final_minimum_logit_margin",
    "final_minimum_sensitive_nll_increase",
    "final_retain_action",
    "retain_action_budget",
    "final_total_delta_norm",
    "max_total_delta_norm",
    "final_forget_nonsensitive_kl_proxy",
    "max_forget_nonsensitive_kl",
    "final_guard_pass",
]:
    print(f"{k}: {d.get(k)}")
PY

done

echo "MCF contrastive SURE-LM complete: ${OUTPUT_ROOT}"
