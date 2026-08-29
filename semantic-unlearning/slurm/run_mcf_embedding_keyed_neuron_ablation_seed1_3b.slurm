#!/usr/bin/env bash
#SBATCH --job-name=mcf-keyneu-abl
#SBATCH --account=yl258
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-10
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
CLEAN_WRITER_OUTPUT_DIR="${CLEAN_WRITER_OUTPUT_DIR:-outputs/mcf_compositional_marker_v6_2_clean_seed1_3b}"
ABLATION_ROOT="${ABLATION_ROOT:-outputs/mcf_embedding_keyed_neuron_ablations_seed1_3b}"
TRAINING_VISIBLE="${TRAINING_VISIBLE:-${CLEAN_WRITER_OUTPUT_DIR}/protocol/training_visible_target_aware_direct.json}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${CLEAN_WRITER_OUTPUT_DIR}/protocol/split_manifest.json}"
CONTEXT_MANIFEST="${CONTEXT_MANIFEST:-${CLEAN_WRITER_OUTPUT_DIR}/method/context_manifest.json}"
STAGE1_STATE="${STAGE1_STATE:-${CLEAN_WRITER_OUTPUT_DIR}/method/stage1_writer.pt}"
STAGE1_REPORT="${STAGE1_REPORT:-${CLEAN_WRITER_OUTPUT_DIR}/method/stage1_writer_report.json}"
STAGE1_LOG="${STAGE1_LOG:-${CLEAN_WRITER_OUTPUT_DIR}/method/stage1_writer_log.jsonl}"
STAGE1_GRADIENT_AUDIT="${STAGE1_GRADIENT_AUDIT:-${CLEAN_WRITER_OUTPUT_DIR}/method/stage1_gradient_conflict_audit.json}"
STAGE1_PORTABILITY="${STAGE1_PORTABILITY:-${CLEAN_WRITER_OUTPUT_DIR}/method/training_safe_portability_preflight.json}"
STAGE1_ACCEPTANCE="${STAGE1_ACCEPTANCE:-${CLEAN_WRITER_OUTPUT_DIR}/method/clean_stage1_acceptance.json}"
REGISTRY="${REGISTRY:-protocols/mcf_embedding_keyed_neuron_ablation_registry_v1.json}"

case "${SLURM_ARRAY_TASK_ID}" in
  0) LABEL=random_dormant_neurons; CHANGE=(--selection-mode dormant_random) ;;
  1) LABEL=no_learned_detector; CHANGE=(--detector-steps 0) ;;
  2) LABEL=no_compositional_negative_loss; CHANGE=(--detector-negative-weight 0 --detector-cross-weight 0) ;;
  3) LABEL=no_writer_off_constraint; CHANGE=(--detector-writer-off-weight 0 --writer-off-nll-weight 0) ;;
  4) LABEL=neurons_per_record_1; CHANGE=(--neurons-per-record 1) ;;
  5) LABEL=neurons_per_record_2; CHANGE=(--neurons-per-record 2) ;;
  6) LABEL=neurons_per_record_8; CHANGE=(--neurons-per-record 8) ;;
  7) LABEL=layer_8; CHANGE=(--neuron-layer 8) ;;
  8) LABEL=layer_20; CHANGE=(--neuron-layer 20) ;;
  9) LABEL=layer_24; CHANGE=(--neuron-layer 24) ;;
  10) LABEL=mlp_only_retrained; CHANGE=(--writer-mode none --selection-mode base_context_contrastive --detector-writer-off-weight 0 --writer-off-nll-weight 0 --no-require-writer-necessity --save-rejected-checkpoint --gate-policy strict) ;;
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
for required in "${TRAINING_VISIBLE}" "${SPLIT_MANIFEST}" "${CONTEXT_MANIFEST}" "${STAGE1_STATE}" "${STAGE1_REPORT}" "${STAGE1_LOG}" "${STAGE1_GRADIENT_AUDIT}" "${STAGE1_PORTABILITY}" "${STAGE1_ACCEPTANCE}" "${REGISTRY}"; do
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
  --stage1-report "${STAGE1_REPORT}" \
  --stage1-writer-log "${STAGE1_LOG}" \
  --clean-stage1-portability-preflight "${STAGE1_PORTABILITY}" \
  --clean-stage1-acceptance "${STAGE1_ACCEPTANCE}" \
  --experiment-registry "${REGISTRY}" \
  --experiment-label "${LABEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --seed 1 \
  --forget-num 50 \
  --writer-mode embedding_keyed \
  --neuron-layer 27 \
  --neurons-per-record 4 \
  --selection-mode writer_contrastive \
  --dormant-fraction 0.20 \
  --selection-positive-contexts 8 \
  --selection-negative-contexts 8 \
  --selection-protected-prompts 8192 \
  --selection-min-corpus-prompts 4096 \
  --wikidata-dir "${WIKIDATA_DIR}" \
  --corpus-protection-prompts 8192 \
  --detector-steps 1000 \
  --detector-lr 0.001 \
  --detector-record-batch 4 \
  --detector-positive-contexts all \
  --detector-negative-contexts all \
  --detector-tail-k 2 \
  --detector-positive-floor 0.25 \
  --detector-negative-weight 5 \
  --detector-cross-weight 2 \
  --detector-writer-off-weight 10 \
  --detector-consistency-weight 1 \
  --detector-relative-cap 1.0 \
  --detector-off-abs-max 0.20 \
  --actuator-steps 2000 \
  --actuator-lr 0.0005 \
  --forget-margin 1 \
  --margin-weight 20 \
  --reference-nll-weight 50 \
  --reference-nll-tolerance 0.05 \
  --protected-kl-weight 20 \
  --writer-off-nll-weight 50 \
  --actuator-writer-off-every 1 \
  --actuator-relative-cap 0.50 \
  --require-writer-necessity \
  --gate-policy report \
  --dtype bf16 \
  --device-map single \
  --save-checkpoint \
  "${CHANGE[@]}"

echo "training-only ablation complete: ${LABEL}"
echo "summary: ${OUTPUT_DIR}/embedding_keyed_neuron_summary.json"
echo "official evaluation opened: no"
