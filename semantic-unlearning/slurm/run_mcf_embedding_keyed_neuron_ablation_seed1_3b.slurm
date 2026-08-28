#!/usr/bin/env bash
#SBATCH --job-name=mcf-keyneu-abl
#SBATCH --account=yl258
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-9
#SBATCH --no-requeue
#SBATCH --output=slurm_logs/mcf_embedding_keyed_neuron_ablation_%A_%a.out
#SBATCH --error=slurm_logs/mcf_embedding_keyed_neuron_ablation_%A_%a.err

# Training-only preregistered ablations.  This job has no MCF/evaluation path
# and deliberately performs no official evaluation. Frozen successful
# checkpoints can later be evaluated by a separate reporting-only job.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/yl258/kp759/Unlearning/semantic-unlearning}"
ENV_DIR="${ENV_DIR:-/scratch/yl258/kp759/conda_envs/semantic_unlearning}"
MODEL_PATH="${MODEL_PATH:-/scratch/yl258/kp759/hf-materialized/Llama-3.2-3B-Instruct-clean}"
WIKIDATA_DIR="${WIKIDATA_DIR:-/scratch/yl258/kp759/datasets/wikipedia_sure_50020}"
V3_OUTPUT_DIR="${V3_OUTPUT_DIR:-outputs/mcf_compositional_marker_v3_seed1_3b}"
ABLATION_ROOT="${ABLATION_ROOT:-outputs/mcf_embedding_keyed_neuron_ablations_seed1_3b}"
TRAINING_VISIBLE="${TRAINING_VISIBLE:-${V3_OUTPUT_DIR}/protocol/training_visible_target_aware_direct.json}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${V3_OUTPUT_DIR}/protocol/split_manifest.json}"
CONTEXT_MANIFEST="${CONTEXT_MANIFEST:-${V3_OUTPUT_DIR}/method/context_manifest.json}"
STAGE1_STATE="${STAGE1_STATE:-${V3_OUTPUT_DIR}/method/stage1_writer.pt}"
REGISTRY="${REGISTRY:-protocols/mcf_embedding_keyed_neuron_ablation_registry_v1.json}"

case "${SLURM_ARRAY_TASK_ID}" in
  0) LABEL=random_dormant_neurons; CHANGE=(--selection-mode dormant_random) ;;
  1) LABEL=no_learned_detector; CHANGE=(--detector-steps 0) ;;
  2) LABEL=no_compositional_negative_loss; CHANGE=(--detector-negative-weight 0 --detector-cross-weight 0) ;;
  3) LABEL=no_writer_off_constraint; CHANGE=(--detector-writer-off-weight 0 --writer-off-nll-weight 0) ;;
  4) LABEL=neurons_per_record_1; CHANGE=(--neurons-per-record 1) ;;
  5) LABEL=neurons_per_record_2; CHANGE=(--neurons-per-record 2) ;;
  6) LABEL=neurons_per_record_8; CHANGE=(--neurons-per-record 8) ;;
  7) LABEL=layer_4; CHANGE=(--neuron-layer 4) ;;
  8) LABEL=layer_12; CHANGE=(--neuron-layer 12) ;;
  9) LABEL=layer_16; CHANGE=(--neuron-layer 16) ;;
  *) echo "unregistered ablation index: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

cd "${PROJECT_DIR}"
export PATH="${ENV_DIR}/bin:${PATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
OUTPUT_DIR="${ABLATION_ROOT}/${LABEL}"
mkdir -p slurm_logs "${OUTPUT_DIR}/protocol"
if [[ -e "${OUTPUT_DIR}/embedding_keyed_neuron_summary.json" || -e "${OUTPUT_DIR}/checkpoint" ]]; then
  echo "ablation output exists; refusing overwrite: ${OUTPUT_DIR}" >&2
  exit 2
fi
for required in "${TRAINING_VISIBLE}" "${SPLIT_MANIFEST}" "${CONTEXT_MANIFEST}" "${STAGE1_STATE}" "${REGISTRY}"; do
  if [[ ! -e "${required}" ]]; then
    echo "required frozen artifact is missing: ${required}" >&2
    exit 2
  fi
done
cp "${REGISTRY}" "${OUTPUT_DIR}/protocol/ablation_registry_v1.json"

env -u MCF_PATH -u OFFICIAL_MCF_PATH -u OFFICIAL_EVAL_PATH \
  -u PARAPHRASE_PATH -u NEIGHBORHOOD_PATH -u RETAIN_EVAL_PATH \
  -u ADVERSARIAL_EVAL_PATH \
  python -u scripts/mcf_embedding_keyed_neuron_erasure.py \
  --model-path "${MODEL_PATH}" \
  --training-visible-path "${TRAINING_VISIBLE}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --context-manifest "${CONTEXT_MANIFEST}" \
  --stage1-state "${STAGE1_STATE}" \
  --experiment-registry "${REGISTRY}" \
  --experiment-label "${LABEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --seed 1 \
  --forget-num 50 \
  --neuron-layer 8 \
  --neurons-per-record 4 \
  --selection-mode writer_contrastive \
  --dormant-fraction 0.35 \
  --wikidata-dir "${WIKIDATA_DIR}" \
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
  --gate-policy report \
  --dtype bf16 \
  --device-map single \
  --save-checkpoint \
  "${CHANGE[@]}"

echo "training-only ablation complete: ${LABEL}"
echo "summary: ${OUTPUT_DIR}/embedding_keyed_neuron_summary.json"
echo "official evaluation opened: no"
