#!/usr/bin/env bash
#SBATCH --job-name=mcf-keyneuron-v351
#SBATCH --account=yl258
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --no-requeue
#SBATCH --output=slurm_logs/mcf_embedding_keyed_neuron_v3_5_1_%j.out
#SBATCH --error=slurm_logs/mcf_embedding_keyed_neuron_v3_5_1_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/yl258/kp759/Unlearning/semantic-unlearning}"
ENV_DIR="${ENV_DIR:-/scratch/yl258/kp759/conda_envs/semantic_unlearning}"
MODEL_PATH="${MODEL_PATH:-/scratch/yl258/kp759/hf-materialized/Llama-3.2-3B-Instruct-clean}"
WIKIDATA_DIR="${WIKIDATA_DIR:-/scratch/yl258/kp759/datasets/wikipedia_sure_50020}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mcf_embedding_keyed_neuron_v3_5_1_seed1_3b}"
CLEAN_WRITER_OUTPUT_DIR="${CLEAN_WRITER_OUTPUT_DIR:-outputs/mcf_compositional_marker_v6_2_clean_seed1_3b}"
FROZEN_V3_2_OUTPUT_DIR="${FROZEN_V3_2_OUTPUT_DIR:-outputs/mcf_embedding_keyed_neuron_v3_2_seed1_3b}"
FROZEN_V3_4_OUTPUT_DIR="${FROZEN_V3_4_OUTPUT_DIR:-outputs/mcf_embedding_keyed_neuron_v3_4_seed1_3b}"
FROZEN_V3_5_OUTPUT_DIR="${FROZEN_V3_5_OUTPUT_DIR:-outputs/mcf_embedding_keyed_neuron_v3_5_seed1_3b}"

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

cd "${PROJECT_DIR}"
export PATH="${ENV_DIR}/bin:${PATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "output path already exists; choose a fresh V3.5.1 directory: ${OUTPUT_DIR}" >&2
  exit 2
fi
for required in "${TRAINING_VISIBLE}" "${SPLIT_MANIFEST}" "${CONTEXT_MANIFEST}" "${STAGE1_STATE}" "${STAGE1_REPORT}" "${STAGE1_LOG}" "${STAGE1_GRADIENT_AUDIT}" "${STAGE1_PORTABILITY}" "${STAGE1_ACCEPTANCE}" "${REGISTRY}"; do
  if [[ ! -f "${required}" ]]; then
    echo "required frozen training artifact is missing: ${required}" >&2
    exit 2
  fi
done

method_dir() {
  if [[ -d "$1/method" ]]; then
    printf '%s\n' "$1/method"
  else
    printf '%s\n' "$1"
  fi
}
FROZEN_V3_2_METHOD_DIR="$(method_dir "${FROZEN_V3_2_OUTPUT_DIR}")"
FROZEN_V3_4_METHOD_DIR="$(method_dir "${FROZEN_V3_4_OUTPUT_DIR}")"
FROZEN_V3_5_METHOD_DIR="$(method_dir "${FROZEN_V3_5_OUTPUT_DIR}")"

for required in embedding_keyed_neuron_state.pt detector_gate_report.json detector_endpoint_audit.json neuron_selection_report.json embedding_keyed_neuron_summary.json training_firewall_receipt.json; do
  if [[ ! -f "${FROZEN_V3_2_METHOD_DIR}/${required}" ]]; then
    echo "required frozen V3.2 artifact is missing: ${FROZEN_V3_2_METHOD_DIR}/${required}" >&2
    exit 2
  fi
done
for required in detector_gate_report.json detector_endpoint_audit.json neuron_selection_report.json actuator_norm_cap_reachability_sweep.json frozen_detector_selectivity_audit.json frozen_detector_zero_actuator_behavior_audit.json training_only_sweep_completion.json training_rejection.json training_firewall_receipt.json; do
  if [[ ! -f "${FROZEN_V3_4_METHOD_DIR}/${required}" ]]; then
    echo "required frozen V3.4 artifact is missing: ${FROZEN_V3_4_METHOD_DIR}/${required}" >&2
    exit 2
  fi
done
for required in detector_gate_report.json detector_endpoint_audit.json neuron_selection_report.json isolated_threshold_gate_report.json frozen_detector_selectivity_audit.json frozen_v3_2_detector_import.json frozen_v3_4_rejection_import.json training_rejection.json training_firewall_receipt.json; do
  if [[ ! -f "${FROZEN_V3_5_METHOD_DIR}/${required}" ]]; then
    echo "required rejected V3.5 artifact is missing: ${FROZEN_V3_5_METHOD_DIR}/${required}" >&2
    exit 2
  fi
done

if ! jq -e '
  .protocol == "mcf_embedding_keyed_sparse_neuron_suppression_v3_5" and
  .checks == {
    "all_positive_owner_gates_one": true,
    "all_positive_cross_gates_zero": true,
    "all_negative_gates_zero": true,
    "all_writer_off_gates_zero": false
  } and
  .aggregate.positive_owner_gate.n == 346 and .aggregate.positive_owner_gate.min == 1 and
  .aggregate.positive_cross_gate.n == 16954 and .aggregate.positive_cross_gate.max == 0 and
  .aggregate.negative_gate.n == 23250 and .aggregate.negative_gate.max == 0 and
  .aggregate.writer_off_gate.n == 17300 and
  .aggregate.writer_off_gate.max == 0.5735465884208679 and
  ([.per_record[] | select(.writer_off_gate_abs_max > 0.000001)] | length) == 1 and
  ([.per_record[] | select(.writer_off_gate_abs_max > 0.000001)][0].case_id) == 10803 and
  .passed == false and .official_evaluation_prompts_seen == 0
' "${FROZEN_V3_5_METHOD_DIR}/isolated_threshold_gate_report.json" >/dev/null; then
  echo "frozen V3.5 source is not the registered single-cell rejection" >&2
  exit 2
fi
if ! jq -e '.protocol == "mcf_embedding_keyed_sparse_neuron_suppression_v3_5_1"' "${REGISTRY}" >/dev/null; then
  echo "registry is stale; expected V3.5.1: ${REGISTRY}" >&2
  exit 2
fi
python -c 'import sys; sys.path.insert(0, "scripts"); import mcf_embedding_keyed_neuron_core as c; assert c.PROTOCOL == "mcf_embedding_keyed_sparse_neuron_suppression_v3_5_1", c.PROTOCOL' 2>/dev/null || {
  echo "checkout is stale: sparse-neuron V3.5.1 code is not importable" >&2
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
  --detector-steps 0 \
  --detector-initialization frozen_v3_2 \
  --frozen-v3-2-run-dir "${FROZEN_V3_2_OUTPUT_DIR}" \
  --frozen-v3-4-run-dir "${FROZEN_V3_4_OUTPUT_DIR}" \
  --frozen-v3-5-run-dir "${FROZEN_V3_5_OUTPUT_DIR}" \
  --detector-lr 0.001 \
  --detector-record-batch 4 \
  --detector-positive-contexts all \
  --detector-negative-contexts all \
  --detector-tail-k 2 \
  --detector-positive-floor 0.25 \
  --detector-training-positive-floor 0.30 \
  --detector-negative-weight 5 \
  --detector-cross-weight 2 \
  --detector-writer-off-weight 10 \
  --detector-consistency-weight 1 \
  --detector-relative-cap 1.0 \
  --detector-off-abs-max 0.20 \
  --detector-training-off-abs-max 0.15 \
  --detector-certificate-abs-tolerance 1e-7 \
  --training-only-isolated-threshold-feasibility \
  --actuator-feasibility-steps 100 \
  --actuator-feasibility-caps 1.50 \
  --actuator-architecture isolated_thresholded_residual \
  --threshold-gate-numerical-guard 1e-6 \
  --detector-selectivity-ratio-epsilon 1e-8 \
  --detector-selectivity-warning-ratio 100 \
  --actuator-steps 0 \
  --actuator-lr 0.0005 \
  --actuator-batch-size 4 \
  --actuator-protected-batch 80 \
  --actuator-positive-contexts all \
  --actuator-negative-contexts all \
  --actuator-tail-k 2 \
  --forget-margin 1 \
  --actuator-writer-off-nll-tolerance 0.05 \
  --actuator-relative-cap 1.50 \
  --check-every 5 \
  --require-writer-necessity \
  --gate-policy strict \
  --dtype bf16 \
  --device-map single

FORENSICS="${OUTPUT_DIR}/method/v3_5_collision_forensics.json"
GLOBAL_GATE="${OUTPUT_DIR}/method/isolated_threshold_gate_report.json"
COMPLETION="${OUTPUT_DIR}/method/training_only_v3_5_1_completion.json"
for required in "${FORENSICS}" "${GLOBAL_GATE}" "${COMPLETION}"; do
  if [[ ! -f "${required}" ]]; then
    echo "V3.5.1 required forensic artifact is missing: ${required}" >&2
    exit 2
  fi
done
if ! jq -e '
  .protocol == "mcf_embedding_keyed_sparse_neuron_suppression_v3_5_1" and
  .complete == true and
  .result == "single_v3_5_writer_off_collision_identified" and
  (.owner_group | type) == "boolean" and
  .source_case_id == 10803 and
  (.detector_case_id | type) == "number" and
  (.source_context_index | type) == "number" and
  (.raw_signed_response | type) == "number" and
  .runtime_gate > 0 and
  .threshold_changed == false and .detector_tensors_changed == false and
  .actuator_tensors_changed == false and
  .detector_optimizer_constructed == false and
  .actuator_optimizer_constructed == false and
  .checkpoint_saved == false and .official_evaluation_allowed == false and
  .official_evaluation_prompts_seen == 0
' "${COMPLETION}" >/dev/null; then
  echo "V3.5.1 forensic completion is incomplete or violated its firewall" >&2
  exit 2
fi
if ! jq -e '
  .gate_endpoint_violation_counts == {
    "positive_owner": 0,
    "positive_cross": 0,
    "negative": 0,
    "writer_off": 1
  } and
  (.writer_off_gate_argmax_cell.detector_group_index | type) == "number" and
  (.writer_off_gate_argmax_cell.source_context_index | type) == "number" and
  (.writer_off_gate_argmax_cell.owner_group | type) == "boolean"
' "${GLOBAL_GATE}" >/dev/null; then
  echo "V3.5.1 global gate did not isolate exactly one writer-off cell" >&2
  exit 2
fi
if [[ -e "${OUTPUT_DIR}/method/checkpoint" || -e "${OUTPUT_DIR}/method/embedding_keyed_neuron_state.pt" || -e "${OUTPUT_DIR}/official_eval.json" ]]; then
  echo "V3.5.1 forensic replay emitted a forbidden checkpoint/evaluation artifact" >&2
  exit 2
fi

jq '{
  result,
  diagnosis,
  source_case_id,
  source_context_index,
  detector_case_id,
  detector_group_index,
  owner_group,
  raw_signed_response,
  runtime_gate,
  detector_optimizer_constructed,
  actuator_optimizer_constructed,
  official_evaluation_prompts_seen
}' "${COMPLETION}"
