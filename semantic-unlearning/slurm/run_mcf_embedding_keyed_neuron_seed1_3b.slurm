#!/usr/bin/env bash
#SBATCH --job-name=mcf-keyneuron-s1
#SBATCH --account=yl258
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --no-requeue
#SBATCH --output=slurm_logs/mcf_embedding_keyed_neuron_seed1_%j.out
#SBATCH --error=slurm_logs/mcf_embedding_keyed_neuron_seed1_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/yl258/kp759/Unlearning/semantic-unlearning}"
ENV_DIR="${ENV_DIR:-/scratch/yl258/kp759/conda_envs/semantic_unlearning}"
MODEL_PATH="${MODEL_PATH:-/scratch/yl258/kp759/hf-materialized/Llama-3.2-3B-Instruct-clean}"
WIKIDATA_DIR="${WIKIDATA_DIR:-/scratch/yl258/kp759/datasets/wikipedia_sure_50020}"
MCF_PATH="${MCF_PATH:-data/multi_counterfact.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mcf_embedding_keyed_neuron_seed1_3b}"
V3_OUTPUT_DIR="${V3_OUTPUT_DIR:-outputs/mcf_compositional_marker_v3_seed1_3b}"

# These are the exact artifacts that produced the frozen Stage-1 writer.  The
# neuron learner never receives MCF_PATH and cannot open official probes.
TRAINING_VISIBLE="${TRAINING_VISIBLE:-${V3_OUTPUT_DIR}/protocol/training_visible_target_aware_direct.json}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${V3_OUTPUT_DIR}/protocol/split_manifest.json}"
CONTEXT_MANIFEST="${CONTEXT_MANIFEST:-${V3_OUTPUT_DIR}/method/context_manifest.json}"
STAGE1_STATE="${STAGE1_STATE:-${V3_OUTPUT_DIR}/method/stage1_writer.pt}"
REGISTRY="${REGISTRY:-protocols/mcf_embedding_keyed_neuron_ablation_registry_v1.json}"

cd "${PROJECT_DIR}"
mkdir -p slurm_logs "${OUTPUT_DIR}/protocol" "${OUTPUT_DIR}/comparison"
export PATH="${ENV_DIR}/bin:${PATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

if [[ -e "${OUTPUT_DIR}/official_eval.json" || -e "${OUTPUT_DIR}/method/checkpoint" ]]; then
  echo "output already contains evaluation/checkpoint state; refusing overwrite: ${OUTPUT_DIR}" >&2
  exit 2
fi
for required in "${TRAINING_VISIBLE}" "${SPLIT_MANIFEST}" "${CONTEXT_MANIFEST}" "${STAGE1_STATE}" "${REGISTRY}"; do
  if [[ ! -e "${required}" ]]; then
    echo "required frozen training artifact is missing: ${required}" >&2
    exit 2
  fi
done

# Copy only the public preregistration, never an evaluation artifact.
cp "${REGISTRY}" "${OUTPUT_DIR}/protocol/ablation_registry_v1.json"

# TRAINING PROCESS. Deliberately no --mcf-path and no evaluation file argument.
python -u scripts/mcf_embedding_keyed_neuron_erasure.py \
  --model-path "${MODEL_PATH}" \
  --training-visible-path "${TRAINING_VISIBLE}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --context-manifest "${CONTEXT_MANIFEST}" \
  --stage1-state "${STAGE1_STATE}" \
  --experiment-registry "${REGISTRY}" \
  --experiment-label primary \
  --output-dir "${OUTPUT_DIR}/method" \
  --seed 1 \
  --forget-num 50 \
  --neuron-layer 8 \
  --neurons-per-record 4 \
  --selection-mode writer_contrastive \
  --dormant-fraction 0.35 \
  --selection-positive-contexts 8 \
  --selection-protected-prompts 1024 \
  --wikidata-dir "${WIKIDATA_DIR}" \
  --corpus-protection-prompts 256 \
  --detector-steps 600 \
  --detector-lr 0.001 \
  --detector-positive-floor 1 \
  --detector-negative-weight 5 \
  --detector-cross-weight 2 \
  --detector-writer-off-weight 10 \
  --detector-consistency-weight 1 \
  --detector-relative-cap 0.40 \
  --detector-off-abs-max 0.20 \
  --actuator-steps 1000 \
  --actuator-lr 0.005 \
  --forget-margin 1 \
  --margin-weight 100 \
  --reference-nll-weight 10 \
  --reference-nll-tolerance 0.05 \
  --protected-kl-weight 10 \
  --writer-off-nll-weight 10 \
  --actuator-relative-cap 0.40 \
  --gate-policy strict \
  --dtype bf16 \
  --device-map single \
  --save-checkpoint

# FRESH RELOAD. Still no official probe is available.
python -u scripts/verify_mcf_embedding_keyed_neuron_checkpoint.py \
  --model-dir "${OUTPUT_DIR}/method/checkpoint" \
  --training-visible-path "${TRAINING_VISIBLE}" \
  --context-manifest "${CONTEXT_MANIFEST}" \
  --state "${OUTPUT_DIR}/method/embedding_keyed_neuron_state.pt" \
  --out "${OUTPUT_DIR}/method/post_reload_acceptance.json" \
  --forget-margin 1 \
  --batch-size 8 \
  --dtype bf16 \
  --device-map single

# EVALUATION PROCESS. This is the first process allowed to open MCF_PATH or the
# official PPL text. Nothing below writes back into the checkpoint or selects it.
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

python -u scripts/diagnose_mcf_embedding_keyed_neuron_component_ppl.py \
  --model-dir "${OUTPUT_DIR}/method/checkpoint" \
  --state "${OUTPUT_DIR}/method/embedding_keyed_neuron_state.pt" \
  --wikidata-dir "${WIKIDATA_DIR}" \
  --out "${OUTPUT_DIR}/comparison/component_ppl_attribution.json" \
  --dtype bf16 \
  --device-map single

python -u scripts/report_mcf_embedding_keyed_neuron_result.py \
  --base "${OUTPUT_DIR}/base_official_eval.json" \
  --edited "${OUTPUT_DIR}/official_eval.json" \
  --method-summary "${OUTPUT_DIR}/method/embedding_keyed_neuron_summary.json" \
  --post-reload-acceptance "${OUTPUT_DIR}/method/post_reload_acceptance.json" \
  --output-dir "${OUTPUT_DIR}/comparison"

echo "Embedding-keyed sparse neuron seed 1 complete"
echo "training firewall: ${OUTPUT_DIR}/method/training_firewall_receipt.json"
echo "method summary:    ${OUTPUT_DIR}/method/embedding_keyed_neuron_summary.json"
echo "reload audit:      ${OUTPUT_DIR}/method/post_reload_acceptance.json"
echo "official eval:     ${OUTPUT_DIR}/official_eval.json"
echo "comparison:        ${OUTPUT_DIR}/comparison/comparison.md"
echo "component PPL:     ${OUTPUT_DIR}/comparison/component_ppl_attribution.json"
