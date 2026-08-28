#!/usr/bin/env bash
#SBATCH --job-name=mcf-compose3-s1
#SBATCH --account=yl258
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --no-requeue
#SBATCH --output=slurm_logs/mcf_compositional_marker_seed1_%j.out
#SBATCH --error=slurm_logs/mcf_compositional_marker_seed1_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/yl258/kp759/Unlearning/semantic-unlearning}"
ENV_DIR="${ENV_DIR:-/scratch/yl258/kp759/conda_envs/semantic_unlearning}"
MODEL_PATH="${MODEL_PATH:-/scratch/yl258/kp759/hf-materialized/Llama-3.2-3B-Instruct-clean}"
WIKIDATA_DIR="${WIKIDATA_DIR:-/scratch/yl258/kp759/datasets/wikipedia_sure_50020}"
MCF_PATH="${MCF_PATH:-data/multi_counterfact.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mcf_compositional_marker_v3_seed1_3b}"
BUILD_SEMANTIC_SURROGATES="${BUILD_SEMANTIC_SURROGATES:-1}"
SURROGATE_ARTIFACT="${SURROGATE_ARTIFACT:-}"
RESUME_STAGE1_STATE="${RESUME_STAGE1_STATE:-}"

cd "${PROJECT_DIR}"
mkdir -p slurm_logs "${OUTPUT_DIR}/protocol"
export PATH="${ENV_DIR}/bin:${PATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

if [[ -e "${OUTPUT_DIR}/official_eval.json" ]]; then
  echo "${OUTPUT_DIR}/official_eval.json already exists; refusing overwrite" >&2
  exit 2
fi

python -u scripts/build_mcf_sure_target_aware_direct_split.py \
  --mcf-path "${MCF_PATH}" \
  --output-dir "${OUTPUT_DIR}/protocol" \
  --seed 1 \
  --forget-num 50 \
  --retain-eval-num 1000

TRAINING_VISIBLE="${OUTPUT_DIR}/protocol/training_visible_target_aware_direct.json"
SPLIT_MANIFEST="${OUTPUT_DIR}/protocol/split_manifest.json"
SURROGATE_ARGS=()
WRITER_ARGS=(
  --writer-steps 1200
  --writer-lr 2e-4
)

if [[ "${BUILD_SEMANTIC_SURROGATES}" == "1" ]]; then
  if [[ -z "${SURROGATE_ARTIFACT}" ]]; then
    SURROGATE_ARTIFACT="${OUTPUT_DIR}/protocol/robust_prompt_adapter_v7.json"
  fi
  if [[ ! -e "${SURROGATE_ARTIFACT}" ]]; then
    python -u scripts/build_mcf_robust_prompt_adapter_v7.py \
      --training-visible-path "${TRAINING_VISIBLE}" \
      --split-manifest "${SPLIT_MANIFEST}" \
      --output "${SURROGATE_ARTIFACT}" \
      --seed 1 \
      --forget-num 50 \
      --min-surrogates 3 \
      --max-surrogates 8 \
      --generator-model-path "${MODEL_PATH}" \
      --dtype bf16 \
      --device-map single \
      --generation-rounds 8 \
      --generation-oversample 3
  fi
  SURROGATE_ARGS=(
    --surrogate-prompts-path "${SURROGATE_ARTIFACT}"
    --require-semantic-surrogates
  )
fi

if [[ -n "${RESUME_STAGE1_STATE}" ]]; then
  if [[ ! -e "${RESUME_STAGE1_STATE}" ]]; then
    echo "Stage-1 resume state not found: ${RESUME_STAGE1_STATE}" >&2
    exit 2
  fi
  WRITER_ARGS=(
    --resume-stage1-state "${RESUME_STAGE1_STATE}"
    --writer-steps 0
  )
fi

python -u scripts/mcf_compositional_marker_write_read.py \
  --model-path "${MODEL_PATH}" \
  --training-visible-path "${TRAINING_VISIBLE}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --output-dir "${OUTPUT_DIR}/method" \
  --seed 1 \
  --forget-num 50 \
  --wikidata-dir "${WIKIDATA_DIR}" \
  --synthetic-paraphrases-per-record 6 \
  --corpus-context-prefixes 256 \
  --reach-probes 12 \
  --positive-context-batch 4 \
  --negative-context-batch 5 \
  --row-norm-cap 8 \
  --write-alpha 6 \
  --lambda-consistency 2 \
  --lambda-negative 10 \
  --lambda-cross 2 \
  --reader-anchor-weight 0.05 \
  --reader-negative-weight 10 \
  --reader-refine-steps 600 \
  --reader-cross-positive-cap 256 \
  --kappa-train-max 0.10 \
  --portability-min 0.50 \
  --cos-marker-reader-min 0 \
  --gate-policy report \
  --forget-margin 1 \
  --stage2-steps 1000 \
  --stage2-lr 0.02 \
  --stage2-negative-weight 0.01 \
  --stage2-base-positive-weight 1 \
  --stage2-check-every 50 \
  --dtype bf16 \
  --device-map single \
  --save-checkpoint \
  "${SURROGATE_ARGS[@]}" \
  "${WRITER_ARGS[@]}"

# Reload in a clean process before any held-out probe is opened. This catches
# hook/materialization, BF16, config, and serialization discrepancies.
python -u scripts/verify_mcf_compositional_marker_checkpoint.py \
  --model-dir "${OUTPUT_DIR}/method/checkpoint" \
  --training-visible-path "${TRAINING_VISIBLE}" \
  --context-manifest "${OUTPUT_DIR}/method/context_manifest.json" \
  --out "${OUTPUT_DIR}/method/post_reload_acceptance.json" \
  --forget-margin 1 \
  --batch-size 8 \
  --dtype bf16 \
  --device-map single

# Final evaluation is deliberately a separate process and is the first process
# in this workflow that opens the original MCF evaluation probes.
python -u scripts/mcf_zero_unlearn_official_eval.py \
  --model-dir "${OUTPUT_DIR}/method/checkpoint" \
  --mcf-path "${MCF_PATH}" \
  --wikidata-dir "${WIKIDATA_DIR}" \
  --out "${OUTPUT_DIR}/official_eval.json" \
  --seed 1 \
  --sample-mode official \
  --dtype bfloat16 \
  --device-map auto

python -u scripts/mcf_zero_unlearn_official_eval.py \
  --model-dir "${MODEL_PATH}" \
  --mcf-path "${MCF_PATH}" \
  --wikidata-dir "${WIKIDATA_DIR}" \
  --out "${OUTPUT_DIR}/base_official_eval.json" \
  --seed 1 \
  --sample-mode official \
  --dtype bfloat16 \
  --device-map auto \
  --quiet

python -u scripts/report_mcf_compositional_marker_result.py \
  --base "${OUTPUT_DIR}/base_official_eval.json" \
  --edited "${OUTPUT_DIR}/official_eval.json" \
  --method-summary "${OUTPUT_DIR}/method/compositional_marker_summary.json" \
  --post-reload-acceptance "${OUTPUT_DIR}/method/post_reload_acceptance.json" \
  --output-dir "${OUTPUT_DIR}/comparison"

echo "Compositional marker seed 1 complete"
echo "method summary: ${OUTPUT_DIR}/method/compositional_marker_summary.json"
echo "reload audit:   ${OUTPUT_DIR}/method/post_reload_acceptance.json"
echo "official eval:  ${OUTPUT_DIR}/official_eval.json"
echo "base eval:      ${OUTPUT_DIR}/base_official_eval.json"
echo "comparison:     ${OUTPUT_DIR}/comparison/comparison.md"
