#!/usr/bin/env bash
#SBATCH --job-name=mcf-keyneuron-v362-official
#SBATCH --account=yl258
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --no-requeue
#SBATCH --output=slurm_logs/mcf_embedding_keyed_neuron_v3_6_2_official_%j.out
#SBATCH --error=slurm_logs/mcf_embedding_keyed_neuron_v3_6_2_official_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/yl258/kp759/Unlearning/semantic-unlearning}"
ENV_DIR="${ENV_DIR:-/scratch/yl258/kp759/conda_envs/semantic_unlearning}"
MODEL_PATH="${MODEL_PATH:-/scratch/yl258/kp759/hf-materialized/Llama-3.2-3B-Instruct-clean}"
MCF_PATH="${MCF_PATH:-${PROJECT_DIR}/data/multi_counterfact.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-/scratch/yl258/kp759/datasets/wikipedia_sure_50020}"
CLEAN_WRITER_OUTPUT_DIR="${CLEAN_WRITER_OUTPUT_DIR:-outputs/mcf_compositional_marker_v6_2_clean_seed1_3b}"
FROZEN_V3_6_2_OUTPUT_DIR="${FROZEN_V3_6_2_OUTPUT_DIR:-outputs/mcf_embedding_keyed_neuron_v3_6_2_seed1_3b}"
OFFICIAL_OUTPUT_DIR="${OFFICIAL_OUTPUT_DIR:-outputs/mcf_embedding_keyed_neuron_v3_6_2_official_seed1_3b}"
OFFICIAL_PROTOCOL="${OFFICIAL_PROTOCOL:-protocols/mcf_embedding_keyed_neuron_v3_6_2_official_evaluation_v1.json}"
EXPECTED_CANDIDATE_SHA256="a54723022361aa3a32f83c493fdc46b4c3b25c837b8f3c6b45e36a935e596ac8"
EXPECTED_MCF_SHA256="977a6acce4705507b5fd6bfcea8f61cd78f9ed0f9cd9c9a6bcd6c8a3ed61c833"

STAGE1_STATE="${CLEAN_WRITER_OUTPUT_DIR}/method/stage1_writer.pt"
CANDIDATE_STATE="${FROZEN_V3_6_2_OUTPUT_DIR}/method/v3_6_2_candidate_state.pt"
TRAINING_COMPLETION="${FROZEN_V3_6_2_OUTPUT_DIR}/method/training_only_v3_6_2_completion.json"
TRAINING_REPORT="${FROZEN_V3_6_2_OUTPUT_DIR}/method/v3_6_2_full_preservation_training.json"

cd "${PROJECT_DIR}"
export PATH="${ENV_DIR}/bin:${PATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

if [[ -e "${OFFICIAL_OUTPUT_DIR}" ]]; then
  echo "official output already exists; resume and retry are prohibited: ${OFFICIAL_OUTPUT_DIR}" >&2
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
  "${OFFICIAL_PROTOCOL}"
do
  test -e "${required}" || {
    echo "required official-evaluation input is missing: ${required}" >&2
    exit 2
  }
done

if ! jq -e --arg candidate "${EXPECTED_CANDIDATE_SHA256}" --arg mcf "${EXPECTED_MCF_SHA256}" '
  .schema_version == 1 and
  .protocol == "mcf_embedding_keyed_sparse_neuron_suppression_v3_6_2_official_v1" and
  .status == "frozen_post_training_pre_official_evaluation" and
  .candidate_checkpoint_sha256 == $candidate and
  .official_mcf_sha256 == $mcf and
  .evaluation.seed == 1 and
  .evaluation.forget_num == 50 and
  .evaluation.retain_num == 1000 and
  .evaluation.sample_mode == "official" and
  .evaluation.dtype == "bf16" and
  .evaluation.layer == 27 and
  .evaluation.detector_neurons_per_record == 4 and
  .evaluation.actuator_neurons_per_record == 16 and
  .evaluation.actuator_relative_cap == 1.5 and
  .fixed_arm_order == ["full_candidate", "reconstructed_base", "writer_only", "actuator_without_writer"] and
  .one_shot_policy.retry_prohibited == true and
  .one_shot_policy.resume_prohibited == true and
  .one_shot_policy.checkpoint_selection_prohibited == true and
  .one_shot_policy.gradient_updates_prohibited == true
' "${OFFICIAL_PROTOCOL}" >/dev/null; then
  echo "official protocol is stale or has been weakened: ${OFFICIAL_PROTOCOL}" >&2
  exit 2
fi

observed_candidate_sha256="$(sha256sum "${CANDIDATE_STATE}" | awk '{print $1}')"
if [[ "${observed_candidate_sha256}" != "${EXPECTED_CANDIDATE_SHA256}" ]]; then
  echo "candidate checkpoint SHA-256 mismatch" >&2
  exit 2
fi
if ! jq -e --arg candidate "${EXPECTED_CANDIDATE_SHA256}" '
  .protocol == "mcf_embedding_keyed_sparse_neuron_suppression_v3_6_2" and
  .full_preservation_passed == true and
  .candidate_checkpoint_saved == true and
  .candidate_checkpoint_sha256 == $candidate and
  .eligible_for_separate_official_evaluation == true and
  .official_evaluation_allowed_in_this_process == false and
  .official_evaluation_prompts_seen == 0
' "${TRAINING_COMPLETION}" >/dev/null; then
  echo "V3.6.2 completion is not an accepted frozen candidate" >&2
  exit 2
fi
if ! jq -e '
  .passed == true and
  .complete_training_log == true and
  .final_audit.passed == true and
  .protected_kl_audit.passed == true and
  .causal_component_audit.passed == true and
  .detector_tensors_unchanged == true and
  .lm_head_bit_identical == true and
  .norm_cap_passed == true and
  .official_evaluation_prompts_seen == 0
' "${TRAINING_REPORT}" >/dev/null; then
  echo "V3.6.2 training acceptance report is not fully passing" >&2
  exit 2
fi

mkdir -p slurm_logs
python -u scripts/evaluate_mcf_embedding_keyed_neuron_v3_6_2_official.py \
  --model-path "${MODEL_PATH}" \
  --training-run-dir "${FROZEN_V3_6_2_OUTPUT_DIR}" \
  --stage1-state "${STAGE1_STATE}" \
  --mcf-path "${MCF_PATH}" \
  --wikidata-dir "${WIKIDATA_DIR}" \
  --protocol "${OFFICIAL_PROTOCOL}" \
  --output-dir "${OFFICIAL_OUTPUT_DIR}"

for required in \
  pre_open_firewall_receipt.json \
  official_evaluation_opened.json \
  official_evaluation.json \
  artifact_manifest.json \
  terminal_status.json
do
  test -f "${OFFICIAL_OUTPUT_DIR}/${required}" || {
    echo "official evaluation did not emit ${required}" >&2
    exit 2
  }
done
if ! jq -e '
  .status == "completed" and
  .official_evaluation_opened == true and
  .official_evaluation_completed == true and
  .retry_or_resume_permitted == false
' "${OFFICIAL_OUTPUT_DIR}/terminal_status.json" >/dev/null; then
  echo "official evaluation did not reach its immutable terminal state" >&2
  exit 2
fi
if ! jq -e --arg mcf "${EXPECTED_MCF_SHA256}" '
  .official_evaluation_opened == true and
  .official_evaluation_completed == true and
  .official_mcf_sha256 == $mcf and
  .used_for_checkpoint_selection_or_early_stopping == false and
  .retry_or_resume_permitted == false
' "${OFFICIAL_OUTPUT_DIR}/official_evaluation.json" >/dev/null; then
  echo "official result does not preserve the frozen dataset/firewall contract" >&2
  exit 2
fi
if [[ "$(sha256sum "${CANDIDATE_STATE}" | awk '{print $1}')" != "${EXPECTED_CANDIDATE_SHA256}" ]]; then
  echo "candidate checkpoint changed during official evaluation" >&2
  exit 2
fi

jq '{
  candidate_checkpoint_sha256,
  candidate_behavioral_acceptance,
  runtime_integrity,
  arms: (
    .arms
    | with_entries(.value = {
        forget: .value.forget,
        retain: .value.retain,
        PPL: .value.PPL
      })
  ),
  paper_claim_readiness,
  used_for_checkpoint_selection_or_early_stopping,
  retry_or_resume_permitted
}' "${OFFICIAL_OUTPUT_DIR}/official_evaluation.json"
