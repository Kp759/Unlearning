#!/usr/bin/env bash
#SBATCH --job-name=mcf-keyneuron-v35
#SBATCH --account=yl258
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --no-requeue
#SBATCH --output=slurm_logs/mcf_embedding_keyed_neuron_v3_5_%j.out
#SBATCH --error=slurm_logs/mcf_embedding_keyed_neuron_v3_5_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/yl258/kp759/Unlearning/semantic-unlearning}"
ENV_DIR="${ENV_DIR:-/scratch/yl258/kp759/conda_envs/semantic_unlearning}"
MODEL_PATH="${MODEL_PATH:-/scratch/yl258/kp759/hf-materialized/Llama-3.2-3B-Instruct-clean}"
WIKIDATA_DIR="${WIKIDATA_DIR:-/scratch/yl258/kp759/datasets/wikipedia_sure_50020}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mcf_embedding_keyed_neuron_v3_5_seed1_3b}"
CLEAN_WRITER_OUTPUT_DIR="${CLEAN_WRITER_OUTPUT_DIR:-outputs/mcf_compositional_marker_v6_2_clean_seed1_3b}"
FROZEN_V3_2_OUTPUT_DIR="${FROZEN_V3_2_OUTPUT_DIR:-outputs/mcf_embedding_keyed_neuron_v3_2_seed1_3b}"
FROZEN_V3_4_OUTPUT_DIR="${FROZEN_V3_4_OUTPUT_DIR:-outputs/mcf_embedding_keyed_neuron_v3_4_seed1_3b}"

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
  echo "output path already exists; choose a fresh V3.5 directory: ${OUTPUT_DIR}" >&2
  exit 2
fi
for required in "${TRAINING_VISIBLE}" "${SPLIT_MANIFEST}" "${CONTEXT_MANIFEST}" "${STAGE1_STATE}" "${STAGE1_REPORT}" "${STAGE1_LOG}" "${STAGE1_GRADIENT_AUDIT}" "${STAGE1_PORTABILITY}" "${STAGE1_ACCEPTANCE}" "${REGISTRY}"; do
  if [[ ! -f "${required}" ]]; then
    echo "required frozen training artifact is missing: ${required}" >&2
    exit 2
  fi
done

if [[ -d "${FROZEN_V3_2_OUTPUT_DIR}/method" ]]; then
  FROZEN_V3_2_METHOD_DIR="${FROZEN_V3_2_OUTPUT_DIR}/method"
else
  FROZEN_V3_2_METHOD_DIR="${FROZEN_V3_2_OUTPUT_DIR}"
fi
for required in embedding_keyed_neuron_state.pt detector_gate_report.json detector_endpoint_audit.json neuron_selection_report.json embedding_keyed_neuron_summary.json training_firewall_receipt.json; do
  if [[ ! -f "${FROZEN_V3_2_METHOD_DIR}/${required}" ]]; then
    echo "required frozen V3.2 detector artifact is missing: ${FROZEN_V3_2_METHOD_DIR}/${required}" >&2
    exit 2
  fi
done

if [[ -d "${FROZEN_V3_4_OUTPUT_DIR}/method" ]]; then
  FROZEN_V3_4_METHOD_DIR="${FROZEN_V3_4_OUTPUT_DIR}/method"
else
  FROZEN_V3_4_METHOD_DIR="${FROZEN_V3_4_OUTPUT_DIR}"
fi
for required in \
  detector_gate_report.json \
  detector_endpoint_audit.json \
  neuron_selection_report.json \
  actuator_norm_cap_reachability_sweep.json \
  frozen_detector_selectivity_audit.json \
  frozen_detector_zero_actuator_behavior_audit.json \
  training_only_sweep_completion.json \
  training_rejection.json \
  training_firewall_receipt.json
do
  if [[ ! -f "${FROZEN_V3_4_METHOD_DIR}/${required}" ]]; then
    echo "required frozen V3.4 rejection artifact is missing: ${FROZEN_V3_4_METHOD_DIR}/${required}" >&2
    exit 2
  fi
done

if ! jq -e '
  .protocol == "mcf_embedding_keyed_sparse_neuron_suppression_v3_2" and
  .passed == true and .passed_records == 50 and .total_records == 50 and
  .official_evaluation_prompts_seen == 0
' "${FROZEN_V3_2_METHOD_DIR}/detector_gate_report.json" >/dev/null; then
  echo "frozen V3.2 source did not pass its 50/50 detector gate" >&2
  exit 2
fi
if ! jq -e '
  .protocol == "mcf_embedding_keyed_sparse_neuron_suppression_v3_4" and
  .caps_preregistered == [0.5, 0.75, 1, 1.5, 2] and
  .caps_completed == [0.5, 0.75, 1, 1.5, 2] and
  .selected_smallest_positive_reachable_cap == 1.5 and
  .positive_reachability_passed == true and
  any(.cap_artifacts[]; .cap == 1.5 and .positive_reachable == true and .down_saturated_columns == 34) and
  .mechanism_readiness_passed == false and
  .structural_selectivity_diagnostic.passed == false and
  .conclusion == "positive_reachability_found_but_structural_writer_off_selectivity_failed" and
  .all_fitted_weights_discarded == true and
  .full_preservation_objective_started == false and .checkpoint_saved == false and
  .official_evaluation_allowed == false and .official_evaluation_prompts_seen == 0
' "${FROZEN_V3_4_METHOD_DIR}/actuator_norm_cap_reachability_sweep.json" >/dev/null; then
  echo "frozen V3.4 source does not match the registered reachability/selectivity rejection" >&2
  exit 2
fi
if ! jq -e '
  .protocol == "mcf_embedding_keyed_sparse_neuron_suppression_v3_4" and
  .writer_off_nll_abs_max == 0.25 and
  .writer_off_preservation_passed == false and
  .official_evaluation_prompts_seen == 0
' "${FROZEN_V3_4_METHOD_DIR}/frozen_detector_zero_actuator_behavior_audit.json" >/dev/null; then
  echo "frozen V3.4 zero-actuator audit does not match the registered 0.25 rejection" >&2
  exit 2
fi
if ! jq -e '.protocol == "mcf_embedding_keyed_sparse_neuron_suppression_v3_5"' "${REGISTRY}" >/dev/null; then
  echo "registry is stale; expected V3.5: ${REGISTRY}" >&2
  exit 2
fi
python -c 'import sys; sys.path.insert(0, "scripts"); import mcf_embedding_keyed_neuron_core as c; assert c.PROTOCOL == "mcf_embedding_keyed_sparse_neuron_suppression_v3_5", c.PROTOCOL' 2>/dev/null || {
  echo "checkout is stale: sparse-neuron V3.5 code is not importable" >&2
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
  --detector-steps 1000 \
  --detector-initialization frozen_v3_2 \
  --frozen-v3-2-run-dir "${FROZEN_V3_2_OUTPUT_DIR}" \
  --frozen-v3-4-run-dir "${FROZEN_V3_4_OUTPUT_DIR}" \
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

FEASIBILITY="${OUTPUT_DIR}/method/isolated_threshold_actuator_feasibility.json"
THRESHOLD_GATE="${OUTPUT_DIR}/method/isolated_threshold_gate_report.json"
IDENTITY_AUDIT="${OUTPUT_DIR}/method/isolated_zero_actuator_identity_audit.json"
BASE_DOWN_AUDIT="${OUTPUT_DIR}/method/selected_base_down_zeroing_audit.json"
CAP_REPORT="${OUTPUT_DIR}/method/actuator_cap_1p50_feasibility.json"
COMPLETION="${OUTPUT_DIR}/method/training_only_v3_5_completion.json"
for required in "${FEASIBILITY}" "${THRESHOLD_GATE}" "${IDENTITY_AUDIT}" "${BASE_DOWN_AUDIT}" "${CAP_REPORT}" "${COMPLETION}"; do
  if [[ ! -f "${required}" ]]; then
    echo "V3.5 required training-only artifact is missing: ${required}" >&2
    exit 2
  fi
done
if ! jq -e '
  .protocol == "mcf_embedding_keyed_sparse_neuron_suppression_v3_5" and
  .mode == "training_only" and .complete == true and
  .architecture == "isolated_thresholded_residual" and
  .caps_preregistered == [1.5] and .caps_completed == [1.5] and
  .all_fitted_weights_discarded == true and
  .frozen_detector_tensors_unchanged == true and
  .full_preservation_objective_started == false and .checkpoint_saved == false and
  .official_evaluation_allowed == false and .official_evaluation_prompts_seen == 0
' "${FEASIBILITY}" >/dev/null; then
  echo "V3.5 feasibility artifact is incomplete or violated its firewall" >&2
  exit 2
fi
if ! jq -e '
  .writer_on_nll_abs_max <= 0.000001 and
  .writer_off_nll_abs_max <= 0.000001 and
  .norms.down_max_relative_norm == 0
' "${IDENTITY_AUDIT}" >/dev/null; then
  echo "V3.5 exact-zero isolated branch was not behaviorally identical" >&2
  exit 2
fi
if [[ -e "${OUTPUT_DIR}/method/checkpoint" || -e "${OUTPUT_DIR}/method/embedding_keyed_neuron_state.pt" || -e "${OUTPUT_DIR}/official_eval.json" ]]; then
  echo "V3.5 training-only run emitted a forbidden checkpoint/evaluation artifact" >&2
  exit 2
fi

jq '{
  architecture,
  caps: .caps_completed,
  selected_smallest_positive_reachable_cap,
  positive_reachability_passed,
  mechanism_readiness_passed,
  isolated_zero_actuator_identity_audit,
  selected_base_down_zeroing_audit,
  structural_selectivity_diagnostic,
  conclusion,
  official_evaluation_prompts_seen
}' "${FEASIBILITY}"
