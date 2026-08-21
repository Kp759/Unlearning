#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${1:?Usage: bash scripts/run_mcf_sure_target_aware_direct_only.sh MODEL [MCF_JSON]}"
MCF="${2:-data/multi_counterfact.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
UTILITY_WIKIPEDIA_DIR="${SURE_UTILITY_WIKIPEDIA_DIR:-${WIKIDATA_DIR}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mcf_sure_target_aware_direct_only_v8}"
SEEDS_TEXT="${MCF_SEEDS:-1}"
FORGET_NUM="${MCF_FORGET_NUM:-50}"
RETAIN_EVAL_NUM="${MCF_RETAIN_EVAL_NUM:-1000}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
ENABLE_EXTERNAL_CONTEXTS="${SURE_ENABLE_EXTERNAL_CONTEXTS:-0}"

source scripts/sure_guarded_shared_defaults.sh

# V8 utility-scaling ablations deliberately vary these values.  Keep the
# shared defaults immutable for every other canonical dataset runner.
UTILITY_SAMPLE_SIZE="${SURE_UTILITY_SAMPLE_SIZE:-${UTILITY_SAMPLE_SIZE}}"
UTILITY_PROMPT_COUNT="${SURE_UTILITY_PROMPT_COUNT:-${UTILITY_PROMPT_COUNT}}"
UTILITY_SEED="${SURE_UTILITY_SEED:-${UTILITY_SEED}}"
UTILITY_EXCLUDE_FIRST="${SURE_UTILITY_EXCLUDE_FIRST:-${UTILITY_EXCLUDE_FIRST}}"
if [[ -z "${SURE_UTILITY_CACHE:-}" ]]; then
  UTILITY_CACHE="outputs/sure_wikipedia_stats/${MODEL_TAG}_token_conditioned_docs${UTILITY_SAMPLE_SIZE}_candidates${UTILITY_PROMPT_COUNT}_v3.pt"
fi

STAGE1_RANK="${SURE_MCF_TARGET_STAGE1_RANK:-4}"
STAGE1_PAIRWISE_TARGET="${SURE_MCF_TARGET_STAGE1_PAIRWISE_TARGET:-1.0}"
STAGE1_TRUE_NLL_INCREASE="${SURE_MCF_TARGET_STAGE1_TRUE_NLL_INCREASE:-2.0}"
STAGE1_NEW_NLL_DECREASE="${SURE_MCF_TARGET_STAGE1_NEW_NLL_DECREASE:-1.0}"
STAGE1_PAIRWISE_WEIGHT="${SURE_MCF_TARGET_STAGE1_PAIRWISE_WEIGHT:-100.0}"
STAGE1_TRUE_GA_WEIGHT="${SURE_MCF_TARGET_STAGE1_TRUE_GA_WEIGHT:-10.0}"
STAGE1_NEW_GD_WEIGHT="${SURE_MCF_TARGET_STAGE1_NEW_GD_WEIGHT:-10.0}"
REQUIRED_PAIRWISE_MARGIN="${SURE_MCF_REQUIRED_PAIRWISE_MARGIN:-0.01}"
STAGE2_SOLVER_MARGINS="${SURE_MCF_STAGE2_SOLVER_MARGINS:-0.5,1.0,2.0}"
STAGE2_RANK_LADDER="${SURE_MCF_TARGET_STAGE2_RANK_LADDER:-2,4,8}"
EXTERNAL_CONTEXTS_PER_RECORD="${SURE_EXTERNAL_CONTEXTS_PER_RECORD:-128}"
EXTERNAL_CONTEXT_LEAD_CHARS="${SURE_EXTERNAL_CONTEXT_LEAD_CHARS:-256}"
LOCALITY_TOKEN_TOPK_PER_ROW="${SURE_LOCALITY_TOKEN_TOPK_PER_ROW:-64}"
LOCALITY_UNIFORM_PROMPT_COUNT="${SURE_LOCALITY_UNIFORM_PROMPT_COUNT:-512}"
LOCALITY_POOL_SEED="${SURE_LOCALITY_POOL_SEED:-1}"
LOCALITY_TRAIN_BATCH_SIZE="${SURE_LOCALITY_TRAIN_BATCH_SIZE:-128}"
LOCALITY_EVAL_BATCH_SIZE="${SURE_LOCALITY_EVAL_BATCH_SIZE:-512}"
STAGE1_LOCALITY_KL_WEIGHT="${SURE_STAGE1_LOCALITY_KL_WEIGHT:-10.0}"
STAGE2_LOCALITY_KL_WEIGHT="${SURE_STAGE2_LOCALITY_KL_WEIGHT:-1.0}"
LOCALITY_KL_MEAN_BUDGET="${SURE_LOCALITY_KL_MEAN_BUDGET:-0.01}"
LOCALITY_KL_P95_BUDGET="${SURE_LOCALITY_KL_P95_BUDGET:-0.05}"
LOCALITY_KL_MAX_BUDGET="${SURE_LOCALITY_KL_MAX_BUDGET:-0.5}"

test -d "${MODEL}"
test -f "${MCF}"
test -d "${WIKIDATA_DIR}"
test -d "${UTILITY_WIKIPEDIA_DIR}"

if [[ ! -f "${UTILITY_CACHE}" ]]; then
  mkdir -p "$(dirname "${UTILITY_CACHE}")"
  python scripts/build_sure_wikipedia_stats.py \
    --model-path "${MODEL}" \
    --wikidata-dir "${UTILITY_WIKIPEDIA_DIR}" \
    --output-path "${UTILITY_CACHE}" \
    --sample-size "${UTILITY_SAMPLE_SIZE}" \
    --require-min-documents "${MIN_UTILITY_DOCUMENTS}" \
    --require-min-prompts "${MIN_UTILITY_PROMPTS}" \
    --require-corpus-protocol "${REQUIRED_UTILITY_CORPUS_PROTOCOL}" \
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
  TRAINING_VISIBLE="${PROTOCOL_DIR}/training_visible_target_aware_direct.json"
  MANIFEST="${PROTOCOL_DIR}/split_manifest.json"
  RETAIN_AUDIT="${PROTOCOL_DIR}/evaluation_only_retain_prompts.json"
  LEARNER="${ROOT}/target_aware_direct_only_learner"
  BASE_EVAL="${ROOT}/base_official_eval.json"
  FINAL_EVAL="${ROOT}/final_official_eval.json"
  PAPER_EVAL="${ROOT}/final_target_true_sensitive_eval.json"
  EXACT_KL="${ROOT}/posthoc_exact_retain_kl.json"
  mkdir -p "${ROOT}"

  # This is the sole training-side access to raw MCF. It emits no paraphrases.
  python scripts/build_mcf_sure_target_aware_direct_split.py \
    --mcf-path "${MCF}" \
    --output-dir "${PROTOCOL_DIR}" \
    --seed "${SEED}" \
    --forget-num "${FORGET_NUM}" \
    --retain-eval-num "${RETAIN_EVAL_NUM}"

  EXTERNAL_CONTEXT_ARGS=()
  if [[ "${ENABLE_EXTERNAL_CONTEXTS}" == "1" ]]; then
    EXTERNAL_CONTEXT_FILE="${PROTOCOL_DIR}/external_subject_locality_contexts.json"
    python scripts/build_sure_mcf_external_contexts.py \
      --training-visible-path "${TRAINING_VISIBLE}" \
      --wikipedia-dir "${UTILITY_WIKIPEDIA_DIR}" \
      --output-path "${EXTERNAL_CONTEXT_FILE}" \
      --corpus-document-limit "${UTILITY_SAMPLE_SIZE}" \
      --exclude-first "${UTILITY_EXCLUDE_FIRST}" \
      --contexts-per-record "${EXTERNAL_CONTEXTS_PER_RECORD}" \
      --lead-chars "${EXTERNAL_CONTEXT_LEAD_CHARS}" \
      --require-corpus-protocol "${REQUIRED_UTILITY_CORPUS_PROTOCOL}" \
      --seed "${SEED}"
    EXTERNAL_CONTEXT_ARGS=(--external-contexts "${EXTERNAL_CONTEXT_FILE}")
  elif [[ "${ENABLE_EXTERNAL_CONTEXTS}" != "0" ]]; then
    echo "SURE_ENABLE_EXTERNAL_CONTEXTS must be 0 or 1" >&2
    exit 2
  fi

  # The learner has no MCF-source argument. Checkpoint selection hard-gates
  # direct FS and external utility only; GFS is unavailable at this point.
  python scripts/sure_mcf_target_aware_direct_only.py \
    --model-path "${MODEL}" \
    --training-visible-path "${TRAINING_VISIBLE}" \
    --split-manifest "${MANIFEST}" \
    --utility-cache "${UTILITY_CACHE}" \
    --output-dir "${LEARNER}" \
    --seed "${SEED}" \
    --forget-num "${FORGET_NUM}" \
    --utility-sample-size "${UTILITY_SAMPLE_SIZE}" \
    --utility-prompt-count "${UTILITY_PROMPT_COUNT}" \
    --require-min-utility-documents "${MIN_UTILITY_DOCUMENTS}" \
    --require-min-utility-prompts "${MIN_UTILITY_PROMPTS}" \
    --require-utility-corpus-protocol "${REQUIRED_UTILITY_CORPUS_PROTOCOL}" \
    --utility-token-topk-per-row "${UTILITY_TOKEN_TOPK_PER_ROW}" \
    --utility-uniform-prompt-count "${UTILITY_UNIFORM_PROMPT_COUNT}" \
    --utility-pool-seed "${UTILITY_POOL_SEED}" \
    --utility-train-batch-size "${UTILITY_TRAIN_BATCH_SIZE}" \
    --utility-eval-batch-size "${UTILITY_EVAL_BATCH_SIZE}" \
    --cache-batch-size "${CACHE_BATCH_SIZE}" \
    --locality-token-topk-per-row "${LOCALITY_TOKEN_TOPK_PER_ROW}" \
    --locality-uniform-prompt-count "${LOCALITY_UNIFORM_PROMPT_COUNT}" \
    --locality-pool-seed "${LOCALITY_POOL_SEED}" \
    --locality-train-batch-size "${LOCALITY_TRAIN_BATCH_SIZE}" \
    --locality-eval-batch-size "${LOCALITY_EVAL_BATCH_SIZE}" \
    --stage1-rank "${STAGE1_RANK}" \
    --stage1-steps "${STAGE1_STEPS}" \
    --stage1-lr "${STAGE1_LR}" \
    --stage1-pairwise-target "${STAGE1_PAIRWISE_TARGET}" \
    --stage1-true-nll-increase "${STAGE1_TRUE_NLL_INCREASE}" \
    --stage1-new-nll-decrease "${STAGE1_NEW_NLL_DECREASE}" \
    --stage1-pairwise-weight "${STAGE1_PAIRWISE_WEIGHT}" \
    --stage1-true-ga-weight "${STAGE1_TRUE_GA_WEIGHT}" \
    --stage1-new-gd-weight "${STAGE1_NEW_GD_WEIGHT}" \
    --stage1-utility-kl-weight "${UTILITY_KL_WEIGHT}" \
    --stage1-locality-kl-weight "${STAGE1_LOCALITY_KL_WEIGHT}" \
    --stage1-l2-weight "${STAGE2_RESIDUAL_L2_WEIGHT}" \
    --stage1-candidate-scales "${STAGE1_CANDIDATE_SCALES}" \
    --required-pairwise-margin "${REQUIRED_PAIRWISE_MARGIN}" \
    --stage2-solver-margins "${STAGE2_SOLVER_MARGINS}" \
    --stage2-rank-ladder "${STAGE2_RANK_LADDER}" \
    --stage2-maxiter "${STAGE2_MAXITER}" \
    --stage2-ftol "${STAGE2_FTOL}" \
    --stage2-constraint-tolerance "${STAGE2_CONSTRAINT_TOLERANCE}" \
    --stage2-residual-l2-weight "${STAGE2_RESIDUAL_L2_WEIGHT}" \
    --stage2-locality-kl-weight "${STAGE2_LOCALITY_KL_WEIGHT}" \
    --constraint-context-weight "${STAGE2_CONSTRAINT_BASIS_WEIGHT}" \
    --contrastive-eps "${CONTRASTIVE_EPS}" \
    --utility-kl-mean-budget "${UTILITY_KL_MEAN_BUDGET}" \
    --utility-kl-p95-budget "${UTILITY_KL_P95_BUDGET}" \
    --utility-kl-max-budget "${UTILITY_KL_MAX_BUDGET}" \
    --locality-kl-mean-budget "${LOCALITY_KL_MEAN_BUDGET}" \
    --locality-kl-p95-budget "${LOCALITY_KL_P95_BUDGET}" \
    --locality-kl-max-budget "${LOCALITY_KL_MAX_BUDGET}" \
    --max-total-delta-norm "${MAX_TOTAL_DELTA_NORM}" \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}" \
    "${EXTERNAL_CONTEXT_ARGS[@]}"

  FINAL_MODEL="${LEARNER}/checkpoint"
  FINAL_DELTA="${LEARNER}/final_total_delta.pt"

  # Raw MCF, official paraphrases, locality probes, PPL text, and benchmark
  # retain prompts first become visible after the checkpoint is frozen.
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
    --model-dir "${FINAL_MODEL}" \
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
    --eval-json "${FINAL_EVAL}" --model-dir "${FINAL_MODEL}" \
    --wikidata-dir "${WIKIDATA_DIR}"

  # FS is the predeclared acceptance metric. GFS is computed and reported by
  # this evaluator but deliberately is not an acceptance condition in v8.
  python scripts/evaluate_mcf_target_true_sensitive.py \
    --base-eval-json "${BASE_EVAL}" \
    --post-eval-json "${FINAL_EVAL}" \
    --split-manifest "${MANIFEST}" \
    --out "${PAPER_EVAL}" \
    --require-min-fs 100

  python scripts/audit_sure_exact_retain_kl.py \
    --model-path "${MODEL}" \
    --retain-prompt-path "${RETAIN_AUDIT}" \
    --delta-path "${FINAL_DELTA}" \
    --scale 1 \
    --output-json "${EXACT_KL}" \
    --batch-size "${CACHE_BATCH_SIZE}" \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}"
done

echo "MCF target-aware direct-only SURE v8 complete: ${OUTPUT_ROOT}"
