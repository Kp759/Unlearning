#!/usr/bin/env bash
#SBATCH --job-name=mcf-keyneuron-v355
#SBATCH --account=yl258
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --no-requeue
#SBATCH --output=slurm_logs/mcf_embedding_keyed_neuron_v3_5_5_%j.out
#SBATCH --error=slurm_logs/mcf_embedding_keyed_neuron_v3_5_5_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/yl258/kp759/Unlearning/semantic-unlearning}"
ENV_DIR="${ENV_DIR:-/scratch/yl258/kp759/conda_envs/semantic_unlearning}"
MODEL_PATH="${MODEL_PATH:-/scratch/yl258/kp759/hf-materialized/Llama-3.2-3B-Instruct-clean}"
WIKIDATA_DIR="${WIKIDATA_DIR:-/scratch/yl258/kp759/datasets/wikipedia_sure_50020}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mcf_embedding_keyed_neuron_v3_5_5_seed1_3b}"
CLEAN_WRITER_OUTPUT_DIR="${CLEAN_WRITER_OUTPUT_DIR:-outputs/mcf_compositional_marker_v6_2_clean_seed1_3b}"
FROZEN_V3_2_OUTPUT_DIR="${FROZEN_V3_2_OUTPUT_DIR:-outputs/mcf_embedding_keyed_neuron_v3_2_seed1_3b}"
FROZEN_V3_4_OUTPUT_DIR="${FROZEN_V3_4_OUTPUT_DIR:-outputs/mcf_embedding_keyed_neuron_v3_4_seed1_3b}"
FROZEN_V3_5_OUTPUT_DIR="${FROZEN_V3_5_OUTPUT_DIR:-outputs/mcf_embedding_keyed_neuron_v3_5_seed1_3b}"
FROZEN_V3_5_1_OUTPUT_DIR="${FROZEN_V3_5_1_OUTPUT_DIR:-outputs/mcf_embedding_keyed_neuron_v3_5_1_seed1_3b}"
FROZEN_V3_5_2_OUTPUT_DIR="${FROZEN_V3_5_2_OUTPUT_DIR:-outputs/mcf_embedding_keyed_neuron_v3_5_2_seed1_3b}"
FROZEN_V3_5_3_OUTPUT_DIR="${FROZEN_V3_5_3_OUTPUT_DIR:-outputs/mcf_embedding_keyed_neuron_v3_5_3_seed1_3b}"
FROZEN_V3_5_4_OUTPUT_DIR="${FROZEN_V3_5_4_OUTPUT_DIR:-outputs/mcf_embedding_keyed_neuron_v3_5_4_seed1_3b}"

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
  echo "output path already exists; choose a fresh V3.5.5 directory: ${OUTPUT_DIR}" >&2
  exit 2
fi
for required in "${TRAINING_VISIBLE}" "${SPLIT_MANIFEST}" "${CONTEXT_MANIFEST}" "${STAGE1_STATE}" "${STAGE1_REPORT}" "${STAGE1_LOG}" "${STAGE1_PORTABILITY}" "${STAGE1_ACCEPTANCE}" "${REGISTRY}"; do
  test -f "${required}" || {
    echo "required frozen training artifact is missing: ${required}" >&2
    exit 2
  }
done
for source in "${FROZEN_V3_2_OUTPUT_DIR}" "${FROZEN_V3_4_OUTPUT_DIR}" "${FROZEN_V3_5_OUTPUT_DIR}" "${FROZEN_V3_5_1_OUTPUT_DIR}" "${FROZEN_V3_5_2_OUTPUT_DIR}" "${FROZEN_V3_5_3_OUTPUT_DIR}" "${FROZEN_V3_5_4_OUTPUT_DIR}"; do
  test -d "${source}" || {
    echo "required frozen source directory is missing: ${source}" >&2
    exit 2
  }
done

method_dir() {
  if [[ -d "$1/method" ]]; then printf '%s\n' "$1/method"; else printf '%s\n' "$1"; fi
}
FROZEN_V3_5_4_METHOD_DIR="$(method_dir "${FROZEN_V3_5_4_OUTPUT_DIR}")"
for required in neuron_selection_report.json detector_gate_report.json isolated_threshold_gate_report.json actuator_cap_1p50_feasibility.json v3_5_4_multilabel_actuator_feasibility.json training_only_v3_5_4_completion.json training_rejection.json training_firewall_receipt.json; do
  test -f "${FROZEN_V3_5_4_METHOD_DIR}/${required}" || {
    echo "required frozen V3.5.4 artifact is missing: ${FROZEN_V3_5_4_METHOD_DIR}/${required}" >&2
    exit 2
  }
done
if ! jq -e '
  .protocol == "mcf_embedding_keyed_sparse_neuron_suppression_v3_5_4" and
  .positive_reachability_passed == false and
  .mechanism_readiness_passed == false and
  .conclusion == "isolated_threshold_branch_not_positive_reachable_at_registered_cap" and
  .checkpoint_saved == false and
  .official_evaluation_prompts_seen == 0
' "${FROZEN_V3_5_4_METHOD_DIR}/training_only_v3_5_4_completion.json" >/dev/null; then
  echo "frozen V3.5.4 source is not the registered actuator rejection" >&2
  exit 2
fi
if ! jq -e '.protocol == "mcf_embedding_keyed_sparse_neuron_suppression_v3_5_5" and .schema_version == 16' "${REGISTRY}" >/dev/null; then
  echo "registry is stale; expected V3.5.5: ${REGISTRY}" >&2
  exit 2
fi
python -c 'import sys; sys.path.insert(0, "scripts"); import mcf_embedding_keyed_neuron_core as c; assert c.PROTOCOL == "mcf_embedding_keyed_sparse_neuron_suppression_v3_5_5", c.PROTOCOL' 2>/dev/null || {
  echo "checkout is stale: sparse-neuron V3.5.5 code is not importable" >&2
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
  --detector-lr 0.001 --detector-record-batch 4 \
  --detector-positive-contexts all --detector-negative-contexts all \
  --detector-tail-k 2 --detector-global-tail-weight 0 \
  --detector-positive-floor 0.25 --detector-training-positive-floor 0.30 \
  --detector-negative-weight 5 --detector-cross-weight 2 \
  --detector-writer-off-weight 10 --detector-consistency-weight 1 \
  --detector-relative-cap 1.0 --detector-off-abs-max 0.20 \
  --detector-training-off-abs-max 0.15 --detector-certificate-abs-tolerance 1e-7 \
  --training-only-actuator-width-sweep \
  --actuator-feasibility-steps 100 --actuator-feasibility-caps 1.50 \
  --actuator-feasibility-widths 4 8 16 \
  --actuator-architecture separate_threshold_gated_actuator_bank \
  --threshold-gate-numerical-guard 1e-6 \
  --detector-selectivity-ratio-epsilon 1e-8 \
  --detector-selectivity-warning-ratio 100 \
  --actuator-steps 0 --actuator-lr 0.0005 --actuator-batch-size 4 \
  --actuator-protected-batch 80 --actuator-positive-contexts all \
  --actuator-negative-contexts all --actuator-tail-k 2 \
  --forget-margin 1 --actuator-writer-off-nll-tolerance 0.05 \
  --actuator-relative-cap 1.50 --check-every 5 \
  --require-writer-necessity --gate-policy strict --dtype bf16 --device-map single

METHOD_DIR="${OUTPUT_DIR}/method"
REPLAY="${METHOD_DIR}/exact_v3_5_4_detector_replay.json"
SELECTION="${METHOD_DIR}/actuator_neuron_selection_report.json"
SWEEP="${METHOD_DIR}/v3_5_5_actuator_width_feasibility.json"
COMPLETION="${METHOD_DIR}/training_only_v3_5_5_completion.json"
for required in "${REPLAY}" "${SELECTION}" "${SWEEP}" "${COMPLETION}"; do
  test -f "${required}" || {
    echo "V3.5.5 required artifact is missing: ${required}" >&2
    exit 2
  }
done
if ! jq -e '.passed == true and .gate_delta_bit_exact == true and .up_delta_bit_exact == true and .official_evaluation_prompts_seen == 0' "${REPLAY}" >/dev/null; then
  echo "V3.5.5 did not reproduce the V3.5.4 detector bit-exactly" >&2
  exit 2
fi
if ! jq -e '.actuator_widths == [4,8,16] and .nested_prefixes == true and .detector_actuator_disjoint == true and .maximum_actuator_neuron_count == 800 and .official_evaluation_prompts_seen == 0' "${SELECTION}" >/dev/null; then
  echo "V3.5.5 actuator-bank selection is incomplete" >&2
  exit 2
fi
if ! jq -e '
  .protocol == "mcf_embedding_keyed_sparse_neuron_suppression_v3_5_5" and
  .complete == true and .actuator_widths_preregistered == [4,8,16] and
  (.arm_artifacts | length) == 5 and
  ([.arm_artifacts[] | select(.selection_driving_arm == true)] | length) == 3 and
  ([.arm_artifacts[] | select(.budget_regime == "matched_width4_group_budget_control")] | length) == 2 and
  .all_fitted_weights_discarded == true and
  .frozen_detector_tensors_unchanged == true and
  .full_preservation_objective_started == false and
  .checkpoint_saved == false and .official_evaluation_allowed == false and
  .official_evaluation_prompts_seen == 0
' "${SWEEP}" >/dev/null; then
  echo "V3.5.5 discarded width-sweep receipt is incomplete" >&2
  exit 2
fi
if [[ -e "${METHOD_DIR}/checkpoint" || -e "${METHOD_DIR}/embedding_keyed_neuron_state.pt" || -e "${OUTPUT_DIR}/official_eval.json" ]]; then
  echo "V3.5.5 emitted a forbidden checkpoint/evaluation artifact" >&2
  exit 2
fi

jq '{
  selected_smallest_positive_reachable_width,
  selected_smallest_mechanism_ready_width,
  positive_reachability_passed,
  structural_selectivity_passed,
  mechanism_readiness_passed,
  conclusion,
  full_preservation_objective_started,
  official_evaluation_prompts_seen
}' "${COMPLETION}"
