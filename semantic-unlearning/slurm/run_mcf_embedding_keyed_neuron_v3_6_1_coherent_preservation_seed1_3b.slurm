#!/usr/bin/env bash
#SBATCH --job-name=mcf-keyneuron-v361
#SBATCH --account=yl258
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --no-requeue
#SBATCH --output=slurm_logs/mcf_embedding_keyed_neuron_v3_6_1_%j.out
#SBATCH --error=slurm_logs/mcf_embedding_keyed_neuron_v3_6_1_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/yl258/kp759/Unlearning/semantic-unlearning}"
ENV_DIR="${ENV_DIR:-/scratch/yl258/kp759/conda_envs/semantic_unlearning}"
MODEL_PATH="${MODEL_PATH:-/scratch/yl258/kp759/hf-materialized/Llama-3.2-3B-Instruct-clean}"
WIKIDATA_DIR="${WIKIDATA_DIR:-/scratch/yl258/kp759/datasets/wikipedia_sure_50020}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mcf_embedding_keyed_neuron_v3_6_1_seed1_3b}"
CLEAN_WRITER_OUTPUT_DIR="${CLEAN_WRITER_OUTPUT_DIR:-outputs/mcf_compositional_marker_v6_2_clean_seed1_3b}"
FROZEN_V3_2_OUTPUT_DIR="${FROZEN_V3_2_OUTPUT_DIR:-outputs/mcf_embedding_keyed_neuron_v3_2_seed1_3b}"
FROZEN_V3_4_OUTPUT_DIR="${FROZEN_V3_4_OUTPUT_DIR:-outputs/mcf_embedding_keyed_neuron_v3_4_seed1_3b}"
FROZEN_V3_5_OUTPUT_DIR="${FROZEN_V3_5_OUTPUT_DIR:-outputs/mcf_embedding_keyed_neuron_v3_5_seed1_3b}"
FROZEN_V3_5_1_OUTPUT_DIR="${FROZEN_V3_5_1_OUTPUT_DIR:-outputs/mcf_embedding_keyed_neuron_v3_5_1_seed1_3b}"
FROZEN_V3_5_2_OUTPUT_DIR="${FROZEN_V3_5_2_OUTPUT_DIR:-outputs/mcf_embedding_keyed_neuron_v3_5_2_seed1_3b}"
FROZEN_V3_5_3_OUTPUT_DIR="${FROZEN_V3_5_3_OUTPUT_DIR:-outputs/mcf_embedding_keyed_neuron_v3_5_3_seed1_3b}"
FROZEN_V3_5_4_OUTPUT_DIR="${FROZEN_V3_5_4_OUTPUT_DIR:-outputs/mcf_embedding_keyed_neuron_v3_5_4_seed1_3b}"
FROZEN_V3_5_5_OUTPUT_DIR="${FROZEN_V3_5_5_OUTPUT_DIR:-outputs/mcf_embedding_keyed_neuron_v3_5_5_seed1_3b}"
FROZEN_V3_6_OUTPUT_DIR="${FROZEN_V3_6_OUTPUT_DIR:-outputs/mcf_embedding_keyed_neuron_v3_6_seed1_3b}"

TRAINING_VISIBLE="${TRAINING_VISIBLE:-${CLEAN_WRITER_OUTPUT_DIR}/protocol/training_visible_target_aware_direct.json}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${CLEAN_WRITER_OUTPUT_DIR}/protocol/split_manifest.json}"
CONTEXT_MANIFEST="${CONTEXT_MANIFEST:-${CLEAN_WRITER_OUTPUT_DIR}/method/context_manifest.json}"
STAGE1_STATE="${STAGE1_STATE:-${CLEAN_WRITER_OUTPUT_DIR}/method/stage1_writer.pt}"
STAGE1_REPORT="${STAGE1_REPORT:-${CLEAN_WRITER_OUTPUT_DIR}/method/stage1_writer_report.json}"
STAGE1_LOG="${STAGE1_LOG:-${CLEAN_WRITER_OUTPUT_DIR}/method/stage1_writer_log.jsonl}"
STAGE1_PORTABILITY="${STAGE1_PORTABILITY:-${CLEAN_WRITER_OUTPUT_DIR}/method/training_safe_portability_preflight.json}"
STAGE1_ACCEPTANCE="${STAGE1_ACCEPTANCE:-${CLEAN_WRITER_OUTPUT_DIR}/method/clean_stage1_acceptance.json}"
REGISTRY="${REGISTRY:-protocols/mcf_embedding_keyed_neuron_ablation_registry_v1.json}"

cd "${PROJECT_DIR}"
export PATH="${ENV_DIR}/bin:${PATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "output path already exists; choose a fresh V3.6.1 directory: ${OUTPUT_DIR}" >&2
  exit 2
fi
for required in \
  "${TRAINING_VISIBLE}" \
  "${SPLIT_MANIFEST}" \
  "${CONTEXT_MANIFEST}" \
  "${STAGE1_STATE}" \
  "${STAGE1_REPORT}" \
  "${STAGE1_LOG}" \
  "${STAGE1_PORTABILITY}" \
  "${STAGE1_ACCEPTANCE}" \
  "${REGISTRY}"
do
  test -f "${required}" || {
    echo "required frozen training artifact is missing: ${required}" >&2
    exit 2
  }
done
for source in \
  "${FROZEN_V3_2_OUTPUT_DIR}" \
  "${FROZEN_V3_4_OUTPUT_DIR}" \
  "${FROZEN_V3_5_OUTPUT_DIR}" \
  "${FROZEN_V3_5_1_OUTPUT_DIR}" \
  "${FROZEN_V3_5_2_OUTPUT_DIR}" \
  "${FROZEN_V3_5_3_OUTPUT_DIR}" \
  "${FROZEN_V3_5_4_OUTPUT_DIR}" \
  "${FROZEN_V3_5_5_OUTPUT_DIR}" \
  "${FROZEN_V3_6_OUTPUT_DIR}"
do
  test -d "${source}" || {
    echo "required frozen source directory is missing: ${source}" >&2
    exit 2
  }
done

if ! jq -e '
  .protocol == "mcf_embedding_keyed_sparse_neuron_suppression_v3_6_1" and
  .schema_version == 18 and
  .v3_6_1_scope.negative_prompt_precedence == "forget_positive_exact_prompt_precedes_preservation_negative" and
  .v3_6_1_scope.raw_negative_occurrences == 465 and
  .v3_6_1_scope.coherent_preservation_negative_occurrences == 464 and
  .v3_6_1_scope.negative_acceptance_nll_tolerance == 0.05
' "${REGISTRY}" >/dev/null; then
  echo "registry is stale; expected coherent V3.6.1: ${REGISTRY}" >&2
  exit 2
fi
python -c 'import sys; sys.path.insert(0, "scripts"); import mcf_embedding_keyed_neuron_core as c; assert c.PROTOCOL == "mcf_embedding_keyed_sparse_neuron_suppression_v3_6_1", c.PROTOCOL' 2>/dev/null || {
  echo "checkout is stale: sparse-neuron V3.6.1 code is not importable" >&2
  exit 2
}

mkdir -p slurm_logs "${OUTPUT_DIR}/protocol"
cp "${REGISTRY}" "${OUTPUT_DIR}/protocol/ablation_registry_v1.json"
cp "${STAGE1_ACCEPTANCE}" "${OUTPUT_DIR}/protocol/clean_stage1_acceptance.json"

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
  --experiment-label primary \
  --output-dir "${OUTPUT_DIR}/method" \
  --seed 1 --forget-num 50 --writer-mode embedding_keyed \
  --neuron-layer 27 --neurons-per-record 4 \
  --selection-mode writer_contrastive --dormant-fraction 0.20 \
  --selection-positive-contexts 8 --selection-negative-contexts 8 \
  --selection-protected-prompts 8192 --selection-min-corpus-prompts 4096 \
  --wikidata-dir "${WIKIDATA_DIR}" --corpus-protection-prompts 8192 \
  --detector-steps 100 --detector-initialization frozen_v3_2 \
  --frozen-v3-2-run-dir "${FROZEN_V3_2_OUTPUT_DIR}" \
  --frozen-v3-4-run-dir "${FROZEN_V3_4_OUTPUT_DIR}" \
  --frozen-v3-5-run-dir "${FROZEN_V3_5_OUTPUT_DIR}" \
  --frozen-v3-5-1-run-dir "${FROZEN_V3_5_1_OUTPUT_DIR}" \
  --frozen-v3-5-2-run-dir "${FROZEN_V3_5_2_OUTPUT_DIR}" \
  --frozen-v3-5-3-run-dir "${FROZEN_V3_5_3_OUTPUT_DIR}" \
  --frozen-v3-5-4-run-dir "${FROZEN_V3_5_4_OUTPUT_DIR}" \
  --frozen-v3-5-5-run-dir "${FROZEN_V3_5_5_OUTPUT_DIR}" \
  --frozen-v3-6-run-dir "${FROZEN_V3_6_OUTPUT_DIR}" \
  --detector-lr 0.001 --detector-record-batch 4 \
  --detector-positive-contexts all --detector-negative-contexts all \
  --detector-tail-k 2 --detector-global-tail-weight 0 \
  --detector-positive-floor 0.25 --detector-training-positive-floor 0.30 \
  --detector-negative-weight 5 --detector-cross-weight 2 \
  --detector-writer-off-weight 10 --detector-consistency-weight 1 \
  --detector-relative-cap 1.0 --detector-off-abs-max 0.20 \
  --detector-training-off-abs-max 0.15 \
  --detector-certificate-abs-tolerance 1e-7 \
  --training-only-full-preservation \
  --actuator-feasibility-steps 100 --actuator-feasibility-caps 1.50 \
  --actuator-feasibility-widths 4 8 16 \
  --actuator-architecture separate_threshold_gated_actuator_bank \
  --threshold-gate-numerical-guard 1e-6 \
  --detector-selectivity-ratio-epsilon 1e-8 \
  --detector-selectivity-warning-ratio 100 \
  --actuator-steps 200 --actuator-lr 0.0005 --actuator-batch-size 4 \
  --actuator-protected-batch 80 --actuator-positive-contexts all \
  --actuator-negative-contexts all --actuator-tail-k 2 \
  --actuator-writer-off-every 1 --forget-margin 1 \
  --margin-weight 20 --reference-nll-weight 50 \
  --reference-nll-tolerance 0.05 \
  --negative-preservation-weight 50 \
  --negative-training-nll-tolerance 0 \
  --negative-nll-tolerance 0.05 \
  --protected-kl-weight 20 --protected-kl-mean-tolerance 0.05 \
  --protected-kl-max-tolerance 0.50 \
  --writer-off-nll-weight 50 --actuator-writer-off-nll-tolerance 0.05 \
  --actuator-l2 0.0001 --actuator-relative-cap 1.50 \
  --grad-clip 1.0 --check-every 5 --cache-batch-size 8 --kl-topk 64 \
  --min-writer-necessary-direct-fraction 0.5 \
  --min-decoder-necessary-direct-fraction 0.5 \
  --require-writer-necessity --save-checkpoint \
  --gate-policy strict --dtype bf16 --device-map single

METHOD_DIR="${OUTPUT_DIR}/method"
for required in \
  frozen_v3_6_rejection_import.json \
  negative_preservation_precedence_report.json \
  v3_6_1_nll_numerics_receipt.json \
  exact_v3_5_4_detector_replay.json \
  exact_v3_5_5_width16_selection_replay.json \
  v3_6_1_zero_actuator_identity_audit.json \
  v3_6_1_positive_warm_start.json \
  v3_6_1_full_preservation_training.json \
  v3_6_1_final_full_context_audit.json \
  v3_6_1_protected_kl_audit.json \
  v3_6_1_actuator_endpoint_audit.json \
  v3_6_1_causal_component_audit.json \
  training_only_v3_6_1_completion.json \
  v3_6_1_candidate_state.pt \
  training_firewall_receipt.json
do
  test -f "${METHOD_DIR}/${required}" || {
    echo "V3.6.1 required artifact is missing: ${METHOD_DIR}/${required}" >&2
    exit 2
  }
done
if ! jq -e '
  .registered == true and
  .rule == "forget_positive_exact_prompt_precedes_preservation_negative" and
  .matching_scope == "byte_exact_full_prompt_only" and
  .raw_negative_occurrences == 465 and
  .coherent_preservation_negative_occurrences == 464 and
  .excluded_multi_role_negative_occurrences == 1 and
  .excluded_occurrences[0].prompt_sha256 == "9a4070c81368070d9ee1383958c18109bf7af90ee59042b3132b7a51e9d6ca38" and
  .excluded_occurrences[0].negative_source_case_id == 19763 and
  .excluded_occurrences[0].positive_active_case_ids == [10472] and
  .official_evaluation_prompts_seen == 0
' "${METHOD_DIR}/negative_preservation_precedence_report.json" >/dev/null; then
  echo "V3.6.1 did not reproduce the registered exact-prompt coherence repair" >&2
  exit 2
fi
if ! jq -e '
  .passed == true and
  .actuator_width == 16 and
  .actuator_columns == 800 and
  .full_optimizer_steps_recorded == 200 and
  .complete_training_log == true and
  .final_audit.passed == true and
  .final_audit.negative_nll_abs_max <= 0.05 and
  .negative_preservation_precedence.coherent_preservation_negative_occurrences == 464 and
  .protected_kl_audit.passed == true and
  .causal_component_audit.passed == true and
  .detector_tensors_unchanged == true and
  .lm_head_bit_identical == true and
  .norm_cap_passed == true and
  .official_evaluation_prompts_seen == 0
' "${METHOD_DIR}/v3_6_1_full_preservation_training.json" >/dev/null; then
  echo "V3.6.1 failed one or more locked training-only preservation checks" >&2
  exit 2
fi
if ! jq -e '
  .protocol == "mcf_embedding_keyed_sparse_neuron_suppression_v3_6_1" and
  .positive_warm_start_passed == true and
  .full_preservation_objective_started == true and
  .full_preservation_passed == true and
  .candidate_checkpoint_saved == true and
  .eligible_for_separate_official_evaluation == true and
  .official_evaluation_allowed_in_this_process == false and
  .official_evaluation_prompts_seen == 0
' "${METHOD_DIR}/training_only_v3_6_1_completion.json" >/dev/null; then
  echo "V3.6.1 completion receipt is not an accepted training-only candidate" >&2
  exit 2
fi
if [[ -e "${OUTPUT_DIR}/official_eval.json" ]]; then
  echo "V3.6.1 learner emitted a forbidden official-evaluation artifact" >&2
  exit 2
fi

jq '{
  architecture,
  negative_preservation,
  positive_warm_start_passed,
  positive_warm_start_first_passing_step,
  full_preservation_passed,
  candidate_checkpoint_saved,
  candidate_checkpoint_sha256,
  eligible_for_separate_official_evaluation,
  official_evaluation_allowed_in_this_process,
  official_evaluation_prompts_seen,
  conclusion
}' "${METHOD_DIR}/training_only_v3_6_1_completion.json"
