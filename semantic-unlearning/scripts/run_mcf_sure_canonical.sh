#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${1:?Usage: bash scripts/run_mcf_sure_canonical.sh MODEL [MCF_JSON]}"
MCF="${2:-data/multi_counterfact.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mcf_sure_canonical_3b}"
SEEDS_TEXT="${MCF_SEEDS:-1 2 3 4 5 6 7 8 9 10}"
FORGET_NUM="${MCF_FORGET_NUM:-50}"
RETAIN_NUM="${MCF_RETAIN_EVAL_NUM:-1000}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"

# Shared canonical Stage 1.
STEPS="${SURE_STAGE1_STEPS:-600}"
BATCH_SIZE="${SURE_STAGE1_BATCH_SIZE:-1}"
CACHE_BATCH_SIZE="${SURE_STAGE1_CACHE_BATCH_SIZE:-8}"
EMB_LM_LR="${SURE_STAGE1_LR:-0.0001}"
GA_WEIGHT="${SURE_GA_WEIGHT:-2.0}"
GD_WEIGHT="${SURE_GD_WEIGHT:-1.0}"

# Shared canonical Stage 2. Only the benchmark direct-constraint margin differs.
CANDIDATE_RANKS="${SURE_REPAIR_RANKS:-2,8,0}"
REPAIR_STEPS="${SURE_REPAIR_STEPS:-800}"
REPAIR_LR="${SURE_REPAIR_LR:-0.005}"
REPAIR_L2="${SURE_REPAIR_L2:-0.000001}"
REPAIR_BATCH_SIZE="${SURE_REPAIR_BATCH_SIZE:-8}"
REPAIR_CHECK_EVERY="${SURE_REPAIR_CHECK_EVERY:-25}"
CONSTRAINT_MARGIN="${MCF_SURE_CONSTRAINT_MARGIN:-0.25}"
CANDIDATE_SCALES="${SURE_CANDIDATE_SCALES:-1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,.046875,.03125,.015625,.0078125,0}"
EVAL_BATCH_SIZE="${MCF_EVAL_BATCH_SIZE:-4}"

read -r -a SEEDS <<< "${SEEDS_TEXT}"
test -f "${MCF}"
test -d "${WIKIDATA_DIR}"

for SEED in "${SEEDS[@]}"; do
  ROOT="${OUTPUT_ROOT}/seed${SEED}"
  PROTOCOL="${ROOT}/protocol"
  VISIBLE="${PROTOCOL}/training_visible_forget.json"
  MANIFEST="${PROTOCOL}/split_manifest.json"
  STAGE1="${ROOT}/stage1_gagd"
  STAGE2="${ROOT}/stage2_sparse_row"
  FINAL="${ROOT}/official_eval_locked.json"
  mkdir -p "${ROOT}"

  echo "===== MCF SEED ${SEED}: CANONICAL LOCKED SPLIT ====="
  python scripts/build_mcf_sure_canonical_split.py \
    --mcf-path "${MCF}" --output-dir "${PROTOCOL}" --seed "${SEED}" \
    --forget-num "${FORGET_NUM}" --retain-num "${RETAIN_NUM}"

  echo "===== MCF SEED ${SEED}: COMMON STAGE 1 GA/GD ====="
  rm -rf "${STAGE1}"
  python scripts/sure_stage1_gagd.py \
    --dataset mcf --model-path "${MODEL}" \
    --training-visible-path "${VISIBLE}" --split-manifest "${MANIFEST}" \
    --output-dir "${STAGE1}" --seed "${SEED}" --forget-num "${FORGET_NUM}" \
    --steps "${STEPS}" --batch-size "${BATCH_SIZE}" \
    --cache-batch-size "${CACHE_BATCH_SIZE}" --emb-lm-lr "${EMB_LM_LR}" \
    --ga-weight "${GA_WEIGHT}" --gd-weight "${GD_WEIGHT}" \
    --optimizer adamw --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"

  echo "===== MCF SEED ${SEED}: COMMON STAGE 2 SPARSE SENSITIVE ROWS ====="
  echo "Held-out paraphrases/neighborhoods/retain/PPL have not been opened."
  rm -rf "${STAGE2}"
  python scripts/sure_stage2_sparse_repair.py \
    --dataset mcf --model-path "${STAGE1}/checkpoint" \
    --training-visible-path "${VISIBLE}" --split-manifest "${MANIFEST}" \
    --output-dir "${STAGE2}" --seed "${SEED}" --forget-num "${FORGET_NUM}" \
    --candidate-ranks "${CANDIDATE_RANKS}" --repair-steps "${REPAIR_STEPS}" \
    --repair-lr "${REPAIR_LR}" --constraint-margin "${CONSTRAINT_MARGIN}" \
    --repair-l2 "${REPAIR_L2}" --batch-size "${REPAIR_BATCH_SIZE}" \
    --check-every "${REPAIR_CHECK_EVERY}" --candidate-scales "${CANDIDATE_SCALES}" \
    --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"

  echo "===== MCF SEED ${SEED}: FINAL OFFICIAL EVAL ====="
  python scripts/mcf_zero_unlearn_official_eval.py \
    --model-dir "${STAGE2}/checkpoint" --mcf-path "${MCF}" \
    --wikidata-dir "${WIKIDATA_DIR}" --out "${FINAL}" \
    --unlearn-num "${FORGET_NUM}" --retain-num "${RETAIN_NUM}" \
    --seed "${SEED}" --sample-mode official --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}"

  python scripts/annotate_ppl_provenance.py \
    --eval-json "${FINAL}" --model-dir "${STAGE2}/checkpoint" \
    --wikidata-dir "${WIKIDATA_DIR}"
done

python scripts/aggregate_sure_canonical.py \
  --dataset mcf --root "${OUTPUT_ROOT}" --seeds "${SEEDS[@]}"

echo "Canonical MCF complete: ${OUTPUT_ROOT}"
