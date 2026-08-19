#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${1:?Usage: bash scripts/run_zsre_sure_minimal.sh MODEL [ZSRE_JSON]}"
ZSRE="${2:-data/zsre_mend_eval.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/zsre_sure_minimal}"
SEEDS_TEXT="${ZSRE_SEEDS:-1}"
FORGET_NUM="${ZSRE_FORGET_NUM:-50}"
RETAIN_EVAL_NUM="${ZSRE_RETAIN_EVAL_NUM:-1000}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
EVAL_BATCH_SIZE="${ZSRE_EVAL_BATCH_SIZE:-8}"

# Architecture locks shared verbatim with the MCF runner.
UTILITY_SAMPLE_SIZE=100000
UTILITY_SEED=1
UTILITY_EXCLUDE_FIRST=20
MODEL_TAG="$(basename "${MODEL}")"
UTILITY_CACHE="${SURE_UTILITY_CACHE:-outputs/sure_wikipedia_stats/${MODEL_TAG}_final_hidden_docs100000.pt}"
UTILITY_MAX_LENGTH="${SURE_UTILITY_MAX_LENGTH:-4096}"
UTILITY_BATCH_SIZE="${SURE_UTILITY_BATCH_SIZE:-1}"
STAGE1_STEPS="${SURE_STAGE1_STEPS:-600}"
STAGE1_BATCH_SIZE="${SURE_STAGE1_BATCH_SIZE:-1}"
STAGE1_LR="${SURE_STAGE1_LR:-0.005}"
STAGE2_STEPS="${SURE_STAGE2_STEPS:-500}"
STAGE2_BATCH_SIZE="${SURE_STAGE2_BATCH_SIZE:-8}"
STAGE2_LR="${SURE_STAGE2_LR:-0.005}"
STAGE2_CHECK_EVERY="${SURE_STAGE2_CHECK_EVERY:-25}"
CACHE_BATCH_SIZE="${SURE_CACHE_BATCH_SIZE:-8}"
GA_WEIGHT="${SURE_GA_WEIGHT:-2.0}"
GD_WEIGHT="${SURE_GD_WEIGHT:-1.0}"
CONTRASTIVE_EPS="${SURE_CONTRASTIVE_EPS:-0.001}"
MARGIN="${SURE_SHARED_CONSTRAINT_MARGIN:-0.05}"
MIN_NLL="${SURE_MIN_SENSITIVE_NLL_INCREASE:-4.0}"
CANDIDATE_SCALES="${SURE_CANDIDATE_SCALES:-1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,.046875,.03125,.015625,.0078125,0}"

test -d "${MODEL}"
test -f "${ZSRE}"
test -d "${WIKIDATA_DIR}"

if [[ ! -f "${UTILITY_CACHE}" ]]; then
  mkdir -p "$(dirname "${UTILITY_CACHE}")"
  python scripts/build_sure_wikipedia_stats.py \
    --model-path "${MODEL}" \
    --wikidata-dir "${WIKIDATA_DIR}" \
    --output-path "${UTILITY_CACHE}" \
    --sample-size "${UTILITY_SAMPLE_SIZE}" \
    --utility-seed "${UTILITY_SEED}" \
    --exclude-first "${UTILITY_EXCLUDE_FIRST}" \
    --utility-max-length "${UTILITY_MAX_LENGTH}" \
    --utility-batch-size "${UTILITY_BATCH_SIZE}" \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}"
fi

read -r -a SEEDS <<< "${SEEDS_TEXT}"
for SEED in "${SEEDS[@]}"; do
  ROOT="${OUTPUT_ROOT}/seed${SEED}"
  if [[ -e "${ROOT}" ]]; then
    echo "Refusing to overwrite existing run: ${ROOT}" >&2
    echo "Choose a new OUTPUT_ROOT or remove the run explicitly." >&2
    exit 2
  fi
  PROTOCOL_DIR="${ROOT}/protocol"
  FORGET_VISIBLE="${PROTOCOL_DIR}/training_visible_forget.json"
  RETAIN_AUDIT="${PROTOCOL_DIR}/evaluation_only_retain_prompts.json"
  MANIFEST="${PROTOCOL_DIR}/split_manifest.json"
  LEARNER="${ROOT}/learner"
  BASE_EVAL="${ROOT}/base_official_eval.json"
  FINAL_EVAL="${ROOT}/final_official_eval.json"
  EXACT_KL="${ROOT}/posthoc_exact_retain_kl.json"
  mkdir -p "${ROOT}"

  python scripts/build_sure_minimal_split.py \
    --dataset zsre \
    --dataset-path "${ZSRE}" \
    --output-dir "${PROTOCOL_DIR}" \
    --seed "${SEED}" \
    --forget-num "${FORGET_NUM}" \
    --retain-eval-num "${RETAIN_EVAL_NUM}"

  python scripts/sure_minimal_two_stage.py \
    --dataset zsre \
    --model-path "${MODEL}" \
    --training-visible-path "${FORGET_VISIBLE}" \
    --split-manifest "${MANIFEST}" \
    --utility-cache "${UTILITY_CACHE}" \
    --output-dir "${LEARNER}" \
    --seed "${SEED}" \
    --forget-num "${FORGET_NUM}" \
    --utility-sample-size "${UTILITY_SAMPLE_SIZE}" \
    --stage1-steps "${STAGE1_STEPS}" \
    --stage1-batch-size "${STAGE1_BATCH_SIZE}" \
    --stage1-lr "${STAGE1_LR}" \
    --stage2-steps "${STAGE2_STEPS}" \
    --stage2-batch-size "${STAGE2_BATCH_SIZE}" \
    --stage2-lr "${STAGE2_LR}" \
    --stage2-check-every "${STAGE2_CHECK_EVERY}" \
    --cache-batch-size "${CACHE_BATCH_SIZE}" \
    --ga-weight "${GA_WEIGHT}" \
    --gd-weight "${GD_WEIGHT}" \
    --contrastive-eps "${CONTRASTIVE_EPS}" \
    --constraint-margin "${MARGIN}" \
    --min-sensitive-nll-increase "${MIN_NLL}" \
    --candidate-scales "${CANDIDATE_SCALES}" \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}"

  # Everything below is post-training and cannot affect checkpoint selection.
  python scripts/zsre_zero_unlearn_official_eval.py \
    --model-dir "${MODEL}" \
    --zsre-path "${ZSRE}" \
    --wikidata-dir "${WIKIDATA_DIR}" \
    --out "${BASE_EVAL}" \
    --method "Base" \
    --unlearn-num "${FORGET_NUM}" \
    --retain-num "${RETAIN_EVAL_NUM}" \
    --seed "${SEED}" \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}"

  python scripts/zsre_zero_unlearn_official_eval.py \
    --model-dir "${LEARNER}/checkpoint" \
    --zsre-path "${ZSRE}" \
    --wikidata-dir "${WIKIDATA_DIR}" \
    --out "${FINAL_EVAL}" \
    --method "SURE-LM minimal Wikipedia rank-2 two-stage" \
    --unlearn-num "${FORGET_NUM}" \
    --retain-num "${RETAIN_EVAL_NUM}" \
    --seed "${SEED}" \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}"

  python scripts/annotate_ppl_provenance.py \
    --eval-json "${BASE_EVAL}" --model-dir "${MODEL}" \
    --wikidata-dir "${WIKIDATA_DIR}"
  python scripts/annotate_ppl_provenance.py \
    --eval-json "${FINAL_EVAL}" --model-dir "${LEARNER}/checkpoint" \
    --wikidata-dir "${WIKIDATA_DIR}"

  python scripts/audit_sure_exact_retain_kl.py \
    --model-path "${MODEL}" \
    --retain-prompt-path "${RETAIN_AUDIT}" \
    --delta-path "${LEARNER}/final_total_delta.pt" \
    --scale 1 \
    --output-json "${EXACT_KL}" \
    --batch-size "${CACHE_BATCH_SIZE}" \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}"
done

echo "ZsRE minimal two-stage SURE complete: ${OUTPUT_ROOT}"
