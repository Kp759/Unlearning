#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${1:?Usage: bash scripts/run_zsre_sure_fixed_shared_retain_v3.sh MODEL [ZSRE_JSON]}"
ZSRE="${2:-data/zsre_mend_eval.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/zsre_sure_fixed_shared_retain_v3}"
SEEDS_TEXT="${ZSRE_SEEDS:-1}"
FORGET_NUM="${ZSRE_FORGET_NUM:-50}"
RETAIN_TRAIN_NUM="${SURE_RETAIN_TRAIN_NUM:-1000}"
RETAIN_EVAL_NUM="${ZSRE_RETAIN_EVAL_NUM:-1000}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
EVAL_BATCH_SIZE="${ZSRE_EVAL_BATCH_SIZE:-8}"

# FIXED SHARED v3 defaults -- identical to MCF runner.
STEPS="${SURE_STAGE1_STEPS:-600}"
FORGET_BATCH_SIZE="${SURE_STAGE1_BATCH_SIZE:-1}"
RETAIN_BATCH_SIZE="${SURE_RETAIN_BATCH_SIZE:-4}"
CACHE_BATCH_SIZE="${SURE_STAGE1_CACHE_BATCH_SIZE:-8}"
LR="${SURE_STAGE1_LR:-0.0001}"
GA_WEIGHT="${SURE_GA_WEIGHT:-4.0}"
GD_WEIGHT="${SURE_GD_WEIGHT:-1.0}"
RETAIN_KL_WEIGHT="${SURE_RETAIN_KL_WEIGHT:-1.0}"
STAGE1_L2="${SURE_STAGE1_DELTA_L2:-0.0}"
STAGE1_CONTEXT_RANK="${SURE_STAGE1_CONTEXT_RANK:-2}"
SHARED_MARGIN="${SURE_SHARED_CONSTRAINT_MARGIN:-0.25}"
REQUIRED_NLL_INCREASE="${SURE_REQUIRED_NLL_INCREASE:-4.0}"
CANDIDATE_SCALES="${SURE_CANDIDATE_SCALES:-1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,.046875,.03125,.015625,.0078125,0}"
CANDIDATE_RANKS="${SURE_REPAIR_RANKS:-2,8,0}"
REPAIR_STEPS="${SURE_REPAIR_STEPS:-800}"
REPAIR_LR="${SURE_REPAIR_LR:-0.005}"
REPAIR_L2="${SURE_REPAIR_L2:-0.000001}"
REPAIR_BATCH_SIZE="${SURE_REPAIR_BATCH_SIZE:-8}"
REPAIR_CHECK_EVERY="${SURE_REPAIR_CHECK_EVERY:-25}"

read -r -a SEEDS <<< "${SEEDS_TEXT}"
test -f "${ZSRE}"
test -d "${MODEL}"
test -d "${WIKIDATA_DIR}"

for SEED in "${SEEDS[@]}"; do
  ROOT="${OUTPUT_ROOT}/seed${SEED}"
  PROTOCOL="${ROOT}/protocol"
  FORGET_VISIBLE="${PROTOCOL}/training_visible_forget.json"
  RETAIN_VISIBLE="${PROTOCOL}/training_visible_retain.json"
  MANIFEST="${PROTOCOL}/split_manifest.json"
  STAGE1="${ROOT}/stage1_retain_shared"
  STAGE2="${ROOT}/stage2_retain_shared"
  FINAL="${ROOT}/official_eval_locked.json"

  rm -rf "${ROOT}"
  mkdir -p "${ROOT}"

  echo "===== ZSRE SEED ${SEED}: RETAIN-PROTECTED LOCKED SPLIT ====="
  python scripts/build_zsre_sure_fixed_shared_retain_split.py \
    --zsre-path "${ZSRE}" --output-dir "${PROTOCOL}" --seed "${SEED}" \
    --forget-num "${FORGET_NUM}" --retain-train-num "${RETAIN_TRAIN_NUM}" \
    --retain-eval-num "${RETAIN_EVAL_NUM}"

  echo "===== ZSRE SEED ${SEED}: RETAIN-PROTECTED SHARED STAGE 1 ====="
  python scripts/sure_stage1_context_retain_shared.py \
    --dataset zsre --model-path "${MODEL}" \
    --training-visible-path "${FORGET_VISIBLE}" \
    --training-visible-retain-path "${RETAIN_VISIBLE}" \
    --split-manifest "${MANIFEST}" --output-dir "${STAGE1}" \
    --seed "${SEED}" --forget-num "${FORGET_NUM}" \
    --retain-train-num "${RETAIN_TRAIN_NUM}" --steps "${STEPS}" \
    --batch-size "${FORGET_BATCH_SIZE}" --retain-batch-size "${RETAIN_BATCH_SIZE}" \
    --cache-batch-size "${CACHE_BATCH_SIZE}" --lr "${LR}" \
    --ga-weight "${GA_WEIGHT}" --gd-weight "${GD_WEIGHT}" \
    --retain-kl-weight "${RETAIN_KL_WEIGHT}" --delta-l2 "${STAGE1_L2}" \
    --context-rank "${STAGE1_CONTEXT_RANK}" --constraint-margin "${SHARED_MARGIN}" \
    --required-nll-increase "${REQUIRED_NLL_INCREASE}" \
    --candidate-scales "${CANDIDATE_SCALES}" \
    --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"

  echo "===== ZSRE SEED ${SEED}: RETAIN-PROTECTED SHARED STAGE 2 ====="
  python scripts/sure_stage2_context_retain_shared.py \
    --dataset zsre --model-path "${STAGE1}/checkpoint" \
    --training-visible-path "${FORGET_VISIBLE}" \
    --training-visible-retain-path "${RETAIN_VISIBLE}" \
    --split-manifest "${MANIFEST}" \
    --base-logits-cache "${STAGE1}/base_sensitive_case_logits.pt" \
    --base-retain-logits-cache "${STAGE1}/base_retain_prompt_logits_fp16.pt" \
    --output-dir "${STAGE2}" --seed "${SEED}" --forget-num "${FORGET_NUM}" \
    --retain-train-num "${RETAIN_TRAIN_NUM}" --candidate-ranks "${CANDIDATE_RANKS}" \
    --repair-steps "${REPAIR_STEPS}" --repair-lr "${REPAIR_LR}" \
    --ga-weight "${GA_WEIGHT}" --gd-weight "${GD_WEIGHT}" \
    --retain-kl-weight "${RETAIN_KL_WEIGHT}" --repair-l2 "${REPAIR_L2}" \
    --constraint-margin "${SHARED_MARGIN}" --required-nll-increase "${REQUIRED_NLL_INCREASE}" \
    --batch-size "${REPAIR_BATCH_SIZE}" --retain-batch-size "${RETAIN_BATCH_SIZE}" \
    --check-every "${REPAIR_CHECK_EVERY}" --candidate-scales "${CANDIDATE_SCALES}" \
    --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"

  echo "===== ZSRE SEED ${SEED}: FINAL OFFICIAL EVAL ====="
  python scripts/zsre_zero_unlearn_official_eval.py \
    --model-dir "${STAGE2}/checkpoint" --zsre-path "${ZSRE}" \
    --wikidata-dir "${WIKIDATA_DIR}" --out "${FINAL}" \
    --method "SURE-LM fixed shared retain v3" --unlearn-num "${FORGET_NUM}" \
    --retain-num "${RETAIN_EVAL_NUM}" --seed "${SEED}" \
    --batch-size "${EVAL_BATCH_SIZE}" --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}"

  python scripts/annotate_ppl_provenance.py \
    --eval-json "${FINAL}" --model-dir "${STAGE2}/checkpoint" \
    --wikidata-dir "${WIKIDATA_DIR}"
done

python scripts/aggregate_sure_canonical.py \
  --dataset zsre --root "${OUTPUT_ROOT}" --seeds "${SEEDS[@]}"

echo "Retain-protected fixed-shared ZsRE v3 complete: ${OUTPUT_ROOT}"
