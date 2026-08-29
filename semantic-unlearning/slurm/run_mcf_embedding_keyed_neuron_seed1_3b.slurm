#!/usr/bin/env bash
#SBATCH --job-name=mcf-keyneuron-s1
#SBATCH --account=yl258
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
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
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mcf_embedding_keyed_neuron_v2_seed1_3b}"
CONTROL_OUTPUT_DIR="${CONTROL_OUTPUT_DIR:-${OUTPUT_DIR}_mlp_only_retrained}"
V3_OUTPUT_DIR="${V3_OUTPUT_DIR:-outputs/mcf_compositional_marker_v3_seed1_3b}"

# These are the exact artifacts that produced the frozen Stage-1 writer.  The
# neuron learner never receives MCF_PATH and cannot open official probes.
TRAINING_VISIBLE="${TRAINING_VISIBLE:-${V3_OUTPUT_DIR}/protocol/training_visible_target_aware_direct.json}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${V3_OUTPUT_DIR}/protocol/split_manifest.json}"
CONTEXT_MANIFEST="${CONTEXT_MANIFEST:-${V3_OUTPUT_DIR}/method/context_manifest.json}"
STAGE1_STATE="${STAGE1_STATE:-${V3_OUTPUT_DIR}/method/stage1_writer.pt}"
STAGE1_REPORT="${STAGE1_REPORT:-${V3_OUTPUT_DIR}/method/stage1_writer_report.json}"
REGISTRY="${REGISTRY:-protocols/mcf_embedding_keyed_neuron_ablation_registry_v1.json}"

cd "${PROJECT_DIR}"
export PATH="${ENV_DIR}/bin:${PATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "output path already exists; refusing to mix v1/v2 artifacts: ${OUTPUT_DIR}" >&2
  exit 2
fi
if [[ -e "${CONTROL_OUTPUT_DIR}" ]]; then
  echo "matched-control path already exists; refusing overwrite: ${CONTROL_OUTPUT_DIR}" >&2
  exit 2
fi
for required in "${TRAINING_VISIBLE}" "${SPLIT_MANIFEST}" "${CONTEXT_MANIFEST}" "${STAGE1_STATE}" "${STAGE1_REPORT}" "${REGISTRY}"; do
  if [[ ! -e "${required}" ]]; then
    echo "required frozen training artifact is missing: ${required}" >&2
    exit 2
  fi
done
EXPECTED_PROTOCOL="mcf_embedding_keyed_sparse_neuron_suppression_v2"
if ! jq -e --arg expected "${EXPECTED_PROTOCOL}" '.protocol == $expected' "${REGISTRY}" >/dev/null; then
  echo "registry is stale; expected protocol ${EXPECTED_PROTOCOL}: ${REGISTRY}" >&2
  exit 2
fi
python -c 'import sys; sys.path.insert(0, "scripts"); import mcf_embedding_keyed_neuron_core as c; assert c.PROTOCOL == "mcf_embedding_keyed_sparse_neuron_suppression_v2", c.PROTOCOL' \
  2>/dev/null || {
    echo "checkout is stale: sparse-neuron v2 code is not importable" >&2
    exit 2
  }
mkdir -p slurm_logs "${OUTPUT_DIR}/protocol" "${OUTPUT_DIR}/comparison" \
  "${CONTROL_OUTPUT_DIR}/protocol" "${CONTROL_OUTPUT_DIR}/comparison"

# Copy only the public preregistration, never an evaluation artifact.
cp "${REGISTRY}" "${OUTPUT_DIR}/protocol/ablation_registry_v1.json"
cp "${REGISTRY}" "${CONTROL_OUTPUT_DIR}/protocol/ablation_registry_v1.json"

# TRAINING PROCESS. Deliberately no --mcf-path and no evaluation file argument.
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
  --experiment-label primary \
  --output-dir "${OUTPUT_DIR}/method" \
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

# FRESH RELOAD. Still no official probe is available.
env -u MCF_PATH -u OFFICIAL_MCF_PATH -u OFFICIAL_EVAL_PATH \
  -u PARAPHRASE_PATH -u NEIGHBORHOOD_PATH -u RETAIN_EVAL_PATH \
  -u ADVERSARIAL_EVAL_PATH \
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

# DECISIVE MATCHED CONTROL. This is a new optimization run from Base, not the
# jointly trained checkpoint with its writer toggled off. It receives the same
# cases, layer, neuron count, steps, thresholds, learning rates, norm caps, and
# protection budget. The shared artifact binds cases/context hashes, but its
# writer values are never applied or used by control selection/optimization.
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
  --experiment-label mlp_only_retrained \
  --output-dir "${CONTROL_OUTPUT_DIR}/method" \
  --seed 1 \
  --forget-num 50 \
  --writer-mode none \
  --neuron-layer 27 \
  --neurons-per-record 4 \
  --selection-mode base_context_contrastive \
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
  --detector-writer-off-weight 0 \
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
  --writer-off-nll-weight 0 \
  --actuator-writer-off-every 1 \
  --actuator-relative-cap 0.50 \
  --no-require-writer-necessity \
  --gate-policy strict \
  --dtype bf16 \
  --device-map single \
  --save-checkpoint \
  --save-rejected-checkpoint

env -u MCF_PATH -u OFFICIAL_MCF_PATH -u OFFICIAL_EVAL_PATH \
  -u PARAPHRASE_PATH -u NEIGHBORHOOD_PATH -u RETAIN_EVAL_PATH \
  -u ADVERSARIAL_EVAL_PATH \
  python -u scripts/verify_mcf_embedding_keyed_neuron_checkpoint.py \
  --model-dir "${CONTROL_OUTPUT_DIR}/method/checkpoint" \
  --training-visible-path "${TRAINING_VISIBLE}" \
  --context-manifest "${CONTEXT_MANIFEST}" \
  --state "${CONTROL_OUTPUT_DIR}/method/embedding_keyed_neuron_state.pt" \
  --out "${CONTROL_OUTPUT_DIR}/method/post_reload_acceptance.json" \
  --forget-margin 1 \
  --batch-size 8 \
  --dtype bf16 \
  --device-map single \
  --report-only

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

python -u scripts/mcf_zero_unlearn_official_eval.py \
  --model-dir "${CONTROL_OUTPUT_DIR}/method/checkpoint" \
  --mcf-path "${MCF_PATH}" \
  --wikidata-dir "${WIKIDATA_DIR}" \
  --out "${CONTROL_OUTPUT_DIR}/official_eval.json" \
  --seed 1 \
  --sample-mode official \
  --dtype bfloat16 \
  --device-map auto

python -u scripts/audit_mcf_frozen_writer_portability.py \
  --model-path "${MODEL_PATH}" \
  --stage1-state "${STAGE1_STATE}" \
  --stage1-report "${STAGE1_REPORT}" \
  --mcf-path "${MCF_PATH}" \
  --out "${OUTPUT_DIR}/comparison/writer_portability.json" \
  --seed 1 \
  --unlearn-num 50 \
  --sample-mode official \
  --batch-size 8 \
  --expected-amplitude-threshold 4.5 \
  --dtype bf16 \
  --device-map single

python -u scripts/audit_mcf_embedding_keyed_neuron_retain_tail.py \
  --model-dir "${OUTPUT_DIR}/method/checkpoint" \
  --state "${OUTPUT_DIR}/method/embedding_keyed_neuron_state.pt" \
  --mcf-path "${MCF_PATH}" \
  --out "${OUTPUT_DIR}/comparison/retain_tail_100k.json" \
  --seed 1 \
  --retain-num 9000 \
  --minimum-prompts 100000 \
  --batch-size 32 \
  --dtype bf16 \
  --device-map single

python -u scripts/audit_mcf_embedding_keyed_neuron_retain_tail.py \
  --model-dir "${CONTROL_OUTPUT_DIR}/method/checkpoint" \
  --state "${CONTROL_OUTPUT_DIR}/method/embedding_keyed_neuron_state.pt" \
  --mcf-path "${MCF_PATH}" \
  --out "${CONTROL_OUTPUT_DIR}/comparison/retain_tail_100k.json" \
  --seed 1 \
  --retain-num 9000 \
  --minimum-prompts 100000 \
  --batch-size 32 \
  --dtype bf16 \
  --device-map single

python -u scripts/evaluate_mcf_embedding_keyed_neuron_components.py \
  --model-dir "${OUTPUT_DIR}/method/checkpoint" \
  --state "${OUTPUT_DIR}/method/embedding_keyed_neuron_state.pt" \
  --mcf-path "${MCF_PATH}" \
  --wikidata-dir "${WIKIDATA_DIR}" \
  --context-manifest "${CONTEXT_MANIFEST}" \
  --out "${OUTPUT_DIR}/comparison/official_components.json" \
  --seed 1 \
  --unlearn-num 50 \
  --retain-num 1000 \
  --sample-mode official \
  --dtype bf16 \
  --device-map single

python -u scripts/audit_mcf_embedding_keyed_neuron_latent_recovery.py \
  --model-dir "${OUTPUT_DIR}/method/checkpoint" \
  --state "${OUTPUT_DIR}/method/embedding_keyed_neuron_state.pt" \
  --mcf-path "${MCF_PATH}" \
  --out "${OUTPUT_DIR}/comparison/latent_recovery.json" \
  --seed 1 \
  --unlearn-num 50 \
  --sample-mode official \
  --layer-stride 4 \
  --dtype bf16 \
  --device-map single

python -u scripts/audit_mcf_embedding_keyed_neuron_relearning.py \
  --model-dir "${OUTPUT_DIR}/method/checkpoint" \
  --state "${OUTPUT_DIR}/method/embedding_keyed_neuron_state.pt" \
  --mcf-path "${MCF_PATH}" \
  --out "${OUTPUT_DIR}/comparison/relearning.json" \
  --seed 1 \
  --unlearn-num 50 \
  --sample-mode official \
  --steps 64 \
  --eval-steps 0,1,2,4,8,16,32,64 \
  --batch-size 4 \
  --learning-rate 0.001 \
  --dtype bf16 \
  --device-map single

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
  --mlp-only "${CONTROL_OUTPUT_DIR}/official_eval.json" \
  --mlp-only-method-summary "${CONTROL_OUTPUT_DIR}/method/embedding_keyed_neuron_summary.json" \
  --mlp-only-post-reload-acceptance "${CONTROL_OUTPUT_DIR}/method/post_reload_acceptance.json" \
  --mlp-only-retain-tail-audit "${CONTROL_OUTPUT_DIR}/comparison/retain_tail_100k.json" \
  --writer-portability-audit "${OUTPUT_DIR}/comparison/writer_portability.json" \
  --retain-tail-audit "${OUTPUT_DIR}/comparison/retain_tail_100k.json" \
  --latent-recovery-audit "${OUTPUT_DIR}/comparison/latent_recovery.json" \
  --relearning-audit "${OUTPUT_DIR}/comparison/relearning.json" \
  --output-dir "${OUTPUT_DIR}/comparison"

echo "Embedding-keyed sparse neuron seed 1 complete"
echo "training firewall: ${OUTPUT_DIR}/method/training_firewall_receipt.json"
echo "method summary:    ${OUTPUT_DIR}/method/embedding_keyed_neuron_summary.json"
echo "reload audit:      ${OUTPUT_DIR}/method/post_reload_acceptance.json"
echo "official eval:     ${OUTPUT_DIR}/official_eval.json"
echo "comparison:        ${OUTPUT_DIR}/comparison/comparison.md"
echo "component PPL:     ${OUTPUT_DIR}/comparison/component_ppl_attribution.json"
echo "matched control:   ${CONTROL_OUTPUT_DIR}/official_eval.json"
echo "control retain tail:${CONTROL_OUTPUT_DIR}/comparison/retain_tail_100k.json"
echo "writer portability:${OUTPUT_DIR}/comparison/writer_portability.json"
echo "retain tail:       ${OUTPUT_DIR}/comparison/retain_tail_100k.json"
echo "official components:${OUTPUT_DIR}/comparison/official_components.json"
echo "latent recovery:   ${OUTPUT_DIR}/comparison/latent_recovery.json"
echo "relearning:        ${OUTPUT_DIR}/comparison/relearning.json"
