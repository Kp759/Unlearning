#!/usr/bin/env bash
#SBATCH --job-name=mcf-keyneuron-v362-official-recovery
#SBATCH --account=yl258
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --no-requeue
#SBATCH --output=slurm_logs/mcf_embedding_keyed_neuron_v3_6_2_official_recovery_%j.out
#SBATCH --error=slurm_logs/mcf_embedding_keyed_neuron_v3_6_2_official_recovery_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/yl258/kp759/Unlearning/semantic-unlearning}"
ENV_DIR="${ENV_DIR:-/scratch/yl258/kp759/conda_envs/semantic_unlearning}"
MODEL_PATH="${MODEL_PATH:-/scratch/yl258/kp759/hf-materialized/Llama-3.2-3B-Instruct-clean}"
MCF_PATH="${MCF_PATH:-${PROJECT_DIR}/data/multi_counterfact.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-/scratch/yl258/kp759/datasets/wikipedia_sure_50020}"
CLEAN_WRITER_OUTPUT_DIR="${CLEAN_WRITER_OUTPUT_DIR:-outputs/mcf_compositional_marker_v6_2_clean_seed1_3b}"
FROZEN_V3_6_2_OUTPUT_DIR="${FROZEN_V3_6_2_OUTPUT_DIR:-outputs/mcf_embedding_keyed_neuron_v3_6_2_seed1_3b}"
FAILED_OFFICIAL_OUTPUT_DIR="${FAILED_OFFICIAL_OUTPUT_DIR:-outputs/mcf_embedding_keyed_neuron_v3_6_2_official_seed1_3b}"
RECOVERY_OUTPUT_DIR="${RECOVERY_OUTPUT_DIR:-outputs/mcf_embedding_keyed_neuron_v3_6_2_official_recovery_seed1_3b}"
ORIGINAL_PROTOCOL="${ORIGINAL_PROTOCOL:-protocols/mcf_embedding_keyed_neuron_v3_6_2_official_evaluation_v1.json}"
RECOVERY_PROTOCOL="${RECOVERY_PROTOCOL:-protocols/mcf_embedding_keyed_neuron_v3_6_2_official_recovery_v1.json}"
EXPECTED_CANDIDATE_SHA256="a54723022361aa3a32f83c493fdc46b4c3b25c837b8f3c6b45e36a935e596ac8"
EXPECTED_MCF_SHA256="977a6acce4705507b5fd6bfcea8f61cd78f9ed0f9cd9c9a6bcd6c8a3ed61c833"
EXPECTED_FAILURE="official forget split differs from the frozen candidate"

STAGE1_STATE="${CLEAN_WRITER_OUTPUT_DIR}/method/stage1_writer.pt"
CANDIDATE_STATE="${FROZEN_V3_6_2_OUTPUT_DIR}/method/v3_6_2_candidate_state.pt"
TRAINING_COMPLETION="${FROZEN_V3_6_2_OUTPUT_DIR}/method/training_only_v3_6_2_completion.json"
TRAINING_REPORT="${FROZEN_V3_6_2_OUTPUT_DIR}/method/v3_6_2_full_preservation_training.json"
FAILED_PRE_OPEN="${FAILED_OFFICIAL_OUTPUT_DIR}/pre_open_firewall_receipt.json"
FAILED_OPENED="${FAILED_OFFICIAL_OUTPUT_DIR}/official_evaluation_opened.json"
FAILED_TERMINAL="${FAILED_OFFICIAL_OUTPUT_DIR}/terminal_status.json"

cd "${PROJECT_DIR}"
export PATH="${ENV_DIR}/bin:${PATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

if [[ -e "${RECOVERY_OUTPUT_DIR}" ]]; then
  echo "recovery output already exists; resume and retry are prohibited: ${RECOVERY_OUTPUT_DIR}" >&2
  exit 2
fi
for required in \
  "${MODEL_PATH}" \
  "${MCF_PATH}" \
  "${WIKIDATA_DIR}" \
  "${STAGE1_STATE}" \
  "${CANDIDATE_STATE}" \
  "${TRAINING_COMPLETION}" \
  "${TRAINING_REPORT}" \
  "${ORIGINAL_PROTOCOL}" \
  "${RECOVERY_PROTOCOL}" \
  "${FAILED_PRE_OPEN}" \
  "${FAILED_OPENED}" \
  "${FAILED_TERMINAL}"
do
  test -e "${required}" || {
    echo "required official-recovery input is missing: ${required}" >&2
    exit 2
  }
done

test ! -e "${FAILED_OFFICIAL_OUTPUT_DIR}/official_evaluation.json" || {
  echo "failed attempt unexpectedly contains a completed official result" >&2
  exit 2
}
test ! -e "${FAILED_OFFICIAL_OUTPUT_DIR}/artifact_manifest.json" || {
  echo "failed attempt unexpectedly contains a completed artifact manifest" >&2
  exit 2
}
if find "${FAILED_OFFICIAL_OUTPUT_DIR}/arms" -type f -name '*.json' -print -quit 2>/dev/null | grep -q .; then
  echo "failed attempt unexpectedly contains official arm metrics" >&2
  exit 2
fi

if ! jq -e --arg candidate "${EXPECTED_CANDIDATE_SHA256}" --arg mcf "${EXPECTED_MCF_SHA256}" '
  .protocol == "mcf_embedding_keyed_sparse_neuron_suppression_v3_6_2_official_v1" and
  .official_evaluation_opened == true and
  .official_mcf_sha256 == $mcf and
  .candidate_checkpoint_sha256 == $candidate and
  .resume_or_retry_allowed == false and
  (has("forget_case_ids") | not)
' "${FAILED_OPENED}" >/dev/null; then
  echo "failed official-open receipt does not match the registered pre-forward failure" >&2
  exit 2
fi
if ! jq -e --arg error "${EXPECTED_FAILURE}" '
  .status == "failed_after_official_open" and
  .official_evaluation_opened == true and
  .official_evaluation_completed == false and
  .error_type == "RuntimeError" and
  .error == $error and
  .partial_results_preserved == true and
  .retry_or_resume_permitted == false
' "${FAILED_TERMINAL}" >/dev/null; then
  echo "failed terminal receipt is not the registered index-identity failure" >&2
  exit 2
fi
if ! jq -e --arg candidate "${EXPECTED_CANDIDATE_SHA256}" --arg mcf "${EXPECTED_MCF_SHA256}" '
  .schema_version == 1 and
  .protocol == "mcf_embedding_keyed_sparse_neuron_suppression_v3_6_2_official_index_identity_recovery_v1" and
  .status == "frozen_after_evaluator_failure_before_recovery" and
  .candidate_checkpoint_sha256 == $candidate and
  .official_mcf_sha256 == $mcf and
  .evaluation.seed == 1 and
  .evaluation.forget_num == 50 and
  .evaluation.retain_num == 1000 and
  .evaluation.dtype == "bf16" and
  .evaluation.layer == 27 and
  .evaluation.detector_neurons_per_record == 4 and
  .evaluation.actuator_neurons_per_record == 16 and
  .evaluation.actuator_relative_cap == 1.5 and
  .fixed_arm_order == ["full_candidate", "reconstructed_base", "writer_only", "actuator_without_writer"] and
  .repair_scope.model_change == false and
  .repair_scope.metric_change == false and
  .repair_scope.threshold_change == false and
  .repair_scope.split_change == false and
  .recovery_one_shot_policy.retry_prohibited == true and
  .recovery_one_shot_policy.resume_prohibited == true and
  .recovery_one_shot_policy.retraining_prohibited == true
' "${RECOVERY_PROTOCOL}" >/dev/null; then
  echo "official recovery protocol is stale or has been weakened" >&2
  exit 2
fi

observed_candidate_sha256="$(sha256sum "${CANDIDATE_STATE}" | awk '{print $1}')"
if [[ "${observed_candidate_sha256}" != "${EXPECTED_CANDIDATE_SHA256}" ]]; then
  echo "candidate checkpoint SHA-256 mismatch" >&2
  exit 2
fi

mkdir -p slurm_logs
python -u scripts/evaluate_mcf_embedding_keyed_neuron_v3_6_2_official_recovery.py \
  --model-path "${MODEL_PATH}" \
  --training-run-dir "${FROZEN_V3_6_2_OUTPUT_DIR}" \
  --stage1-state "${STAGE1_STATE}" \
  --mcf-path "${MCF_PATH}" \
  --wikidata-dir "${WIKIDATA_DIR}" \
  --original-protocol "${ORIGINAL_PROTOCOL}" \
  --recovery-protocol "${RECOVERY_PROTOCOL}" \
  --failed-official-run-dir "${FAILED_OFFICIAL_OUTPUT_DIR}" \
  --output-dir "${RECOVERY_OUTPUT_DIR}"

for required in \
  recovery_pre_open_firewall_receipt.json \
  official_recovery_opened.json \
  official_recovery_evaluation.json \
  artifact_manifest.json \
  terminal_status.json
do
  test -f "${RECOVERY_OUTPUT_DIR}/${required}" || {
    echo "official recovery did not emit ${required}" >&2
    exit 2
  }
done
if ! jq -e '
  .status == "completed" and
  .official_recovery_opened == true and
  .official_recovery_completed == true and
  .evaluator_only_recovery == true and
  .model_retry == false and
  .clean_first_run_claimed == false and
  .retry_or_resume_permitted == false
' "${RECOVERY_OUTPUT_DIR}/terminal_status.json" >/dev/null; then
  echo "official recovery did not reach its immutable terminal state" >&2
  exit 2
fi
if ! jq -e --arg mcf "${EXPECTED_MCF_SHA256}" '
  .official_recovery_opened == true and
  .official_recovery_completed == true and
  .official_mcf_sha256 == $mcf and
  .official_dataset_open_processes_total == 2 and
  .official_metric_evaluation_processes_total == 1 and
  .prior_failed_process_metric_forwards == 0 and
  .evaluator_only_recovery == true and
  .clean_first_run_claimed == false and
  .model_retry == false and
  .used_for_checkpoint_selection_or_early_stopping == false and
  .retry_or_resume_permitted == false
' "${RECOVERY_OUTPUT_DIR}/official_recovery_evaluation.json" >/dev/null; then
  echo "official recovery result violates its disclosure/firewall contract" >&2
  exit 2
fi
if [[ "$(sha256sum "${CANDIDATE_STATE}" | awk '{print $1}')" != "${EXPECTED_CANDIDATE_SHA256}" ]]; then
  echo "candidate checkpoint changed during official recovery" >&2
  exit 2
fi

jq '{
  candidate_checkpoint_sha256,
  candidate_behavioral_acceptance,
  runtime_integrity,
  evaluation_identity,
  arms: (
    .arms
    | with_entries(.value = {
        forget: .value.forget,
        retain: .value.retain,
        PPL: .value.PPL
      })
  ),
  official_dataset_open_processes_total,
  official_metric_evaluation_processes_total,
  prior_failed_process_metric_forwards,
  evaluator_only_recovery,
  clean_first_run_claimed,
  model_retry,
  paper_claim_readiness,
  used_for_checkpoint_selection_or_early_stopping,
  retry_or_resume_permitted
}' "${RECOVERY_OUTPUT_DIR}/official_recovery_evaluation.json"
