#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${1:?Usage: bash scripts/run_mcf_sure_minimal.sh MODEL [MCF_JSON]}"
MCF="${2:-data/multi_counterfact.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mcf_sure_exact_constrained_stage2_v5_1}"
SEEDS_TEXT="${MCF_SEEDS:-1}"
FORGET_NUM="${MCF_FORGET_NUM:-50}"
RETAIN_EVAL_NUM="${MCF_RETAIN_EVAL_NUM:-1000}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"

# The exact same architecture configuration is sourced by every dataset adapter.
source scripts/sure_guarded_shared_defaults.sh

test -d "${MODEL}"
test -f "${MCF}"
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
    --utility-prompt-count "${UTILITY_PROMPT_COUNT}" \
    --utility-logit-batch-size "${UTILITY_LOGIT_BATCH_SIZE}" \
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
  PAPER_EVAL="${ROOT}/final_target_true_sensitive_eval.json"
  EXACT_KL="${ROOT}/posthoc_exact_retain_kl.json"
  mkdir -p "${ROOT}"

  python scripts/build_sure_minimal_split.py \
    --dataset mcf \
    --dataset-path "${MCF}" \
    --output-dir "${PROTOCOL_DIR}" \
    --seed "${SEED}" \
    --forget-num "${FORGET_NUM}" \
    --retain-eval-num "${RETAIN_EVAL_NUM}"

  python scripts/sure_minimal_two_stage.py \
    --dataset mcf \
    --model-path "${MODEL}" \
    --training-visible-path "${FORGET_VISIBLE}" \
    --split-manifest "${MANIFEST}" \
    --utility-cache "${UTILITY_CACHE}" \
    --output-dir "${LEARNER}" \
    --seed "${SEED}" \
    --forget-num "${FORGET_NUM}" \
    --utility-sample-size "${UTILITY_SAMPLE_SIZE}" \
    --utility-prompt-count "${UTILITY_PROMPT_COUNT}" \
    --utility-token-topk-per-row "${UTILITY_TOKEN_TOPK_PER_ROW}" \
    --utility-uniform-prompt-count "${UTILITY_UNIFORM_PROMPT_COUNT}" \
    --utility-pool-seed "${UTILITY_POOL_SEED}" \
    --stage1-steps "${STAGE1_STEPS}" \
    --stage1-batch-size "${STAGE1_BATCH_SIZE}" \
    --stage1-lr "${STAGE1_LR}" \
    --stage2-maxiter "${STAGE2_MAXITER}" \
    --stage2-ftol "${STAGE2_FTOL}" \
    --stage2-constraint-tolerance "${STAGE2_CONSTRAINT_TOLERANCE}" \
    --stage2-constraint-buffer "${STAGE2_CONSTRAINT_BUFFER}" \
    --stage2-protected-materialization-buffer "${STAGE2_PROTECTED_MATERIALIZATION_BUFFER}" \
    --stage2-residual-l2-weight "${STAGE2_RESIDUAL_L2_WEIGHT}" \
    --stage2-constraint-basis-weight "${STAGE2_CONSTRAINT_BASIS_WEIGHT}" \
    --stage2-restarts "${STAGE2_RESTARTS}" \
    --cache-batch-size "${CACHE_BATCH_SIZE}" \
    --utility-train-batch-size "${UTILITY_TRAIN_BATCH_SIZE}" \
    --utility-eval-batch-size "${UTILITY_EVAL_BATCH_SIZE}" \
    --direct-constraint-weight "${DIRECT_CONSTRAINT_WEIGHT}" \
    --gd-weight "${GD_WEIGHT}" \
    --utility-kl-weight "${UTILITY_KL_WEIGHT}" \
    --stage2-protection-nll-tolerance "${STAGE2_PROTECTION_NLL_TOLERANCE}" \
    --contrastive-eps "${CONTRASTIVE_EPS}" \
    --constraint-margin "${MARGIN}" \
    --min-sensitive-nll-increase "${MIN_NLL}" \
    --utility-kl-mean-budget "${UTILITY_KL_MEAN_BUDGET}" \
    --utility-kl-p95-budget "${UTILITY_KL_P95_BUDGET}" \
    --utility-kl-max-budget "${UTILITY_KL_MAX_BUDGET}" \
    --max-total-delta-norm "${MAX_TOTAL_DELTA_NORM}" \
    --rank-ladder "${RANK_LADDER}" \
    --candidate-scales "${STAGE1_CANDIDATE_SCALES}" \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}"

  # Everything below is post-training and cannot affect checkpoint selection.
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
    --eval-json "${BASE_EVAL}" --model-dir "${MODEL}" \
    --wikidata-dir "${WIKIDATA_DIR}"
  python scripts/annotate_ppl_provenance.py \
    --eval-json "${FINAL_EVAL}" --model-dir "${LEARNER}/checkpoint" \
    --wikidata-dir "${WIKIDATA_DIR}"

  python scripts/evaluate_mcf_target_true_sensitive.py \
    --base-eval-json "${BASE_EVAL}" \
    --post-eval-json "${FINAL_EVAL}" \
    --split-manifest "${MANIFEST}" \
    --out "${PAPER_EVAL}"

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

echo "MCF guarded two-stage SURE complete: ${OUTPUT_ROOT}"
