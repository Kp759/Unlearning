#!/usr/bin/env bash
#SBATCH --job-name=mcf-compose6-s1
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
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mcf_compositional_marker_v6_seed1_3b}"

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
WRITER_ARGS=(
  --writer-steps 1200
  --writer-lr 2e-4
)

python -u scripts/mcf_compositional_marker_write_read.py \
  --model-path "${MODEL_PATH}" \
  --training-visible-path "${TRAINING_VISIBLE}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --output-dir "${OUTPUT_DIR}/method" \
  --seed 1 \
  --forget-num 50 \
  --wikidata-dir "${WIKIDATA_DIR}" \
  --positive-context-policy relation_templates_only_v1 \
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
  --stage2-negative-weight 0.1 \
  --stage2-base-positive-weight 1 \
  --stage2-reference-nll-weight 10 \
  --stage2-reference-nll-tolerance 0.05 \
  --stage2-protection-rank 512 \
  --stage2-protection-states 1024 \
  --stage2-corpus-protection-prompts 256 \
  --stage2-residual-rank 4 \
  --stage2-row-negative-rank 32 \
  --stage2-row-norm-caps 0.05,0.10,0.20,0.40 \
  --stage2-beta-l2 1e-3 \
  --stage2-check-every 50 \
  --dtype bf16 \
  --device-map single \
  --save-checkpoint \
  "${WRITER_ARGS[@]}"

# Reload in a clean process before any held-out probe is opened. This catches
# hook/materialization, BF16, config, and serialization discrepancies.
python -u scripts/verify_mcf_compositional_marker_checkpoint.py \
  --model-dir "${OUTPUT_DIR}/method/checkpoint" \
  --training-visible-path "${TRAINING_VISIBLE}" \
  --context-manifest "${OUTPUT_DIR}/method/context_manifest.json" \
  --state "${OUTPUT_DIR}/method/compositional_marker_state.pt" \
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

# Post-hoc attribution only. These diagnostics run after the checkpoint and
# official evaluation are frozen, and are never used for model selection.
python -u scripts/diagnose_mcf_compositional_component_ppl.py \
  --model-dir "${OUTPUT_DIR}/method/checkpoint" \
  --state "${OUTPUT_DIR}/method/compositional_marker_state.pt" \
  --wikidata-dir "${WIKIDATA_DIR}" \
  --out "${OUTPUT_DIR}/comparison/component_ppl_attribution.json" \
  --dtype bf16 \
  --device-map single

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
echo "component PPL:  ${OUTPUT_DIR}/comparison/component_ppl_attribution.json"
