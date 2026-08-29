#!/usr/bin/env bash
#SBATCH --job-name=mcf-freq-gate
#SBATCH --account=yl258
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-2
#SBATCH --no-requeue
#SBATCH --output=slurm_logs/mcf_frequency_factorial_%A_%a.out
#SBATCH --error=slurm_logs/mcf_frequency_factorial_%A_%a.err

# Preregistered writer-frequency x decoder-component factorial. Each array item
# retrains Stage 1 from scratch, then trains the identical keyed sparse-neuron
# decoder. Official probes are opened only after both checkpoints are frozen.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/yl258/kp759/Unlearning/semantic-unlearning}"
ENV_DIR="${ENV_DIR:-/scratch/yl258/kp759/conda_envs/semantic_unlearning}"
MODEL_PATH="${MODEL_PATH:-/scratch/yl258/kp759/hf-materialized/Llama-3.2-3B-Instruct-clean}"
WIKIDATA_DIR="${WIKIDATA_DIR:-/scratch/yl258/kp759/datasets/wikipedia_sure_50020}"
MCF_PATH="${MCF_PATH:-data/multi_counterfact.json}"
CLEAN_WRITER_OUTPUT_DIR="${CLEAN_WRITER_OUTPUT_DIR:-outputs/mcf_compositional_marker_v6_1_clean_seed1_3b}"
FACTORIAL_ROOT="${FACTORIAL_ROOT:-outputs/mcf_context_gating_frequency_factorial_seed1_3b}"
REGISTRY="${REGISTRY:-protocols/mcf_embedding_keyed_neuron_ablation_registry_v1.json}"
TRAINING_VISIBLE="${TRAINING_VISIBLE:-${CLEAN_WRITER_OUTPUT_DIR}/protocol/training_visible_target_aware_direct.json}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${CLEAN_WRITER_OUTPUT_DIR}/protocol/split_manifest.json}"

case "${SLURM_ARRAY_TASK_ID}" in
  0) CONDITION=frequency_capped; ROW_CAP=8; FREQUENCY_ALPHA=0.15 ;;
  1) CONDITION=uniform_same_cap; ROW_CAP=8; FREQUENCY_ALPHA=0 ;;
  2) CONDITION=uniform_raised_cap; ROW_CAP=16; FREQUENCY_ALPHA=0 ;;
  *) echo "unregistered factorial index: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

cd "${PROJECT_DIR}"
export PATH="${ENV_DIR}/bin:${PATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
OUTPUT_DIR="${FACTORIAL_ROOT}/${CONDITION}"
WRITER_DIR="${OUTPUT_DIR}/writer"
NEURON_DIR="${OUTPUT_DIR}/neuron"
mkdir -p slurm_logs "${OUTPUT_DIR}/protocol" "${OUTPUT_DIR}/comparison"
if [[ -e "${WRITER_DIR}/stage1_writer.pt" || -e "${NEURON_DIR}/checkpoint" ]]; then
  echo "factorial output exists; refusing overwrite: ${OUTPUT_DIR}" >&2
  exit 2
fi
for required in "${TRAINING_VISIBLE}" "${SPLIT_MANIFEST}" "${REGISTRY}"; do
  if [[ ! -e "${required}" ]]; then
    echo "required frozen artifact is missing: ${required}" >&2
    exit 2
  fi
done
cp "${REGISTRY}" "${OUTPUT_DIR}/protocol/ablation_registry_v1.json"

env -u MCF_PATH -u OFFICIAL_MCF_PATH -u OFFICIAL_EVAL_PATH \
  -u PARAPHRASE_PATH -u NEIGHBORHOOD_PATH -u RETAIN_EVAL_PATH \
  -u ADVERSARIAL_EVAL_PATH \
  python -u scripts/mcf_compositional_marker_write_read.py \
  --model-path "${MODEL_PATH}" \
  --training-visible-path "${TRAINING_VISIBLE}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --output-dir "${WRITER_DIR}" \
  --seed 1 \
  --forget-num 50 \
  --wikidata-dir "${WIKIDATA_DIR}" \
  --positive-context-policy relation_templates_only_v1 \
  --synthetic-paraphrases-per-record 6 \
  --corpus-context-prefixes 256 \
  --reach-probes 12 \
  --writer-record-batch 3 \
  --positive-context-batch 7 \
  --writer-positive-context-mode all \
  --writer-positive-tail-k 2 \
  --negative-context-batch 5 \
  --writer-steps 1200 \
  --writer-lr 0.0002 \
  --write-alpha 6 \
  --lambda-consistency 2 \
  --lambda-negative 10 \
  --lambda-cross 2 \
  --row-norm-cap "${ROW_CAP}" \
  --row-norm-cap-frequency-alpha "${FREQUENCY_ALPHA}" \
  --max-subject-token-frequency 1000000000 \
  --stage1-only \
  --dtype bf16 \
  --device-map single

TRAINING_SAFE_PORTABILITY="${WRITER_DIR}/training_safe_portability_preflight.json"
CLEAN_STAGE1_ACCEPTANCE="${WRITER_DIR}/clean_stage1_acceptance.json"
env -u MCF_PATH -u OFFICIAL_MCF_PATH -u OFFICIAL_EVAL_PATH \
  -u PARAPHRASE_PATH -u NEIGHBORHOOD_PATH -u RETAIN_EVAL_PATH \
  -u ADVERSARIAL_EVAL_PATH -u SURROGATE_ARTIFACT \
  python -u scripts/audit_mcf_clean_stage1_training_preflight.py \
  --model-path "${MODEL_PATH}" \
  --context-manifest "${WRITER_DIR}/context_manifest.json" \
  --stage1-state "${WRITER_DIR}/stage1_writer.pt" \
  --stage1-report "${WRITER_DIR}/stage1_writer_report.json" \
  --stage1-writer-log "${WRITER_DIR}/stage1_writer_log.jsonl" \
  --out "${TRAINING_SAFE_PORTABILITY}" \
  --amplitude-threshold 4.5 \
  --minimum-global-fraction 0.95 \
  --minimum-record-fraction 0.80 \
  --batch-size 8 \
  --dtype bf16 \
  --device-map single

python -u scripts/verify_mcf_clean_stage1_writer.py \
  --training-visible-path "${TRAINING_VISIBLE}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --context-manifest "${WRITER_DIR}/context_manifest.json" \
  --stage1-state "${WRITER_DIR}/stage1_writer.pt" \
  --stage1-report "${WRITER_DIR}/stage1_writer_report.json" \
  --stage1-writer-log "${WRITER_DIR}/stage1_writer_log.jsonl" \
  --training-safe-portability "${TRAINING_SAFE_PORTABILITY}" \
  --out "${CLEAN_STAGE1_ACCEPTANCE}"

env -u MCF_PATH -u OFFICIAL_MCF_PATH -u OFFICIAL_EVAL_PATH \
  -u PARAPHRASE_PATH -u NEIGHBORHOOD_PATH -u RETAIN_EVAL_PATH \
  -u ADVERSARIAL_EVAL_PATH \
  python -u scripts/mcf_embedding_keyed_neuron_erasure.py \
  --model-path "${MODEL_PATH}" \
  --training-visible-path "${TRAINING_VISIBLE}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --context-manifest "${WRITER_DIR}/context_manifest.json" \
  --stage1-state "${WRITER_DIR}/stage1_writer.pt" \
  --stage1-report "${WRITER_DIR}/stage1_writer_report.json" \
  --stage1-writer-log "${WRITER_DIR}/stage1_writer_log.jsonl" \
  --clean-stage1-portability-preflight "${TRAINING_SAFE_PORTABILITY}" \
  --clean-stage1-acceptance "${CLEAN_STAGE1_ACCEPTANCE}" \
  --experiment-registry "${REGISTRY}" \
  --experiment-label primary \
  --output-dir "${NEURON_DIR}" \
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
  --gate-policy strict \
  --dtype bf16 \
  --device-map single \
  --save-checkpoint

env -u MCF_PATH -u OFFICIAL_MCF_PATH -u OFFICIAL_EVAL_PATH \
  -u PARAPHRASE_PATH -u NEIGHBORHOOD_PATH -u RETAIN_EVAL_PATH \
  -u ADVERSARIAL_EVAL_PATH \
  python -u scripts/verify_mcf_embedding_keyed_neuron_checkpoint.py \
  --model-dir "${NEURON_DIR}/checkpoint" \
  --training-visible-path "${TRAINING_VISIBLE}" \
  --context-manifest "${WRITER_DIR}/context_manifest.json" \
  --state "${NEURON_DIR}/embedding_keyed_neuron_state.pt" \
  --out "${NEURON_DIR}/post_reload_acceptance.json" \
  --forget-margin 1 \
  --batch-size 8 \
  --dtype bf16 \
  --device-map single

python -u scripts/evaluate_mcf_embedding_keyed_neuron_components.py \
  --model-dir "${NEURON_DIR}/checkpoint" \
  --state "${NEURON_DIR}/embedding_keyed_neuron_state.pt" \
  --mcf-path "${MCF_PATH}" \
  --wikidata-dir "${WIKIDATA_DIR}" \
  --context-manifest "${WRITER_DIR}/context_manifest.json" \
  --out "${OUTPUT_DIR}/comparison/official_components.json" \
  --seed 1 \
  --unlearn-num 50 \
  --retain-num 1000 \
  --sample-mode official \
  --include-ppl \
  --dtype bf16 \
  --device-map single

echo "frequency-factorial condition complete: ${CONDITION}"
echo "writer summary: ${WRITER_DIR}/stage1_only_summary.json"
echo "component evaluation: ${OUTPUT_DIR}/comparison/official_components.json"
