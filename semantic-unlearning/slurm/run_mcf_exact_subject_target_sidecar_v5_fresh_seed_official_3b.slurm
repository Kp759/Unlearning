#!/usr/bin/env bash
#SBATCH --job-name=mcf-v5-fresh-official
#SBATCH --account=yl258
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --no-requeue
#SBATCH --output=slurm_logs/mcf_v5_fresh_official_%j.out
#SBATCH --error=slurm_logs/mcf_v5_fresh_official_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/yl258/kp759/Unlearning/semantic-unlearning}"
ENV_DIR="${ENV_DIR:-/scratch/yl258/kp759/conda_envs/semantic_unlearning}"
MODEL_PATH="${MODEL_PATH:-/scratch/yl258/kp759/hf-materialized/Llama-3.2-3B-Instruct-clean}"
WIKIDATA_DIR="${WIKIDATA_DIR:-/scratch/yl258/kp759/datasets/wikipedia_sure_50020}"
MCF_PATH="${MCF_PATH:-${PROJECT_DIR}/data/multi_counterfact.json}"
SEED="${SEED:?SEED is required}"
V5_CANDIDATE_OUTPUT_DIR="${V5_CANDIDATE_OUTPUT_DIR:?V5_CANDIDATE_OUTPUT_DIR is required}"
OFFICIAL_OUTPUT_DIR="${OFFICIAL_OUTPUT_DIR:?OFFICIAL_OUTPUT_DIR is required}"
CONFIRMATION_REGISTRY="${CONFIRMATION_REGISTRY:-protocols/mcf_exact_subject_target_sidecar_v5_fresh_multiseed_registry_v1.json}"
SEED1_DEVELOPMENT_RESULT="${SEED1_DEVELOPMENT_RESULT:-outputs/mcf_exact_subject_target_sidecar_v5_seed1_development_3b/development_replay.json}"

cd "${PROJECT_DIR}"
export PATH="${ENV_DIR}/bin:${PATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

case "${SEED}" in
  2|3|4|5|6|7|8|9|10) ;;
  *) echo "fresh V5 seed must be 2 through 10" >&2; exit 2 ;;
esac
if [[ -e "${OFFICIAL_OUTPUT_DIR}" ]]; then
  echo "official output exists; retry/resume is prohibited: ${OFFICIAL_OUTPUT_DIR}" >&2
  exit 2
fi
for required in "${MCF_PATH}" "${CONFIRMATION_REGISTRY}" \
  "${V5_CANDIDATE_OUTPUT_DIR}/method/completion.json" \
  "${V5_CANDIDATE_OUTPUT_DIR}/method/v5_exact_subject_target_sidecar.pt"
do
  test -f "${required}" || {
    echo "required one-shot V5 input is missing: ${required}" >&2
    exit 2
  }
done

if [[ "${SEED}" == "2" ]]; then
  jq -e '
    .passed == true and
    .evaluation_status == "consumed_development_not_blind_not_official" and
    .fresh_seed_confirmation_required == true and
    .blind_evaluation_claim_permitted == false
  ' "${SEED1_DEVELOPMENT_RESULT}" >/dev/null || {
    echo "seed-1 consumed-development receipt is missing or invalid" >&2
    exit 2
  }
else
  PREVIOUS_SEED="$((SEED - 1))"
  PREVIOUS_OFFICIAL_OUTPUT_DIR="${PREVIOUS_OFFICIAL_OUTPUT_DIR:-outputs/mcf_exact_subject_target_sidecar_v5_seed${PREVIOUS_SEED}_official_3b}"
  jq -e --argjson seed "${PREVIOUS_SEED}" '
    .seed == $seed and
    .status == "completed" and
    .official_evaluation_opened == true and
    .official_evaluation_completed == true and
    .retry_or_resume_permitted == false
  ' "${PREVIOUS_OFFICIAL_OUTPUT_DIR}/terminal_status.json" >/dev/null || {
    echo "previous registered seed has not reached a terminal completed state" >&2
    exit 2
  }
fi

python -u scripts/evaluate_mcf_exact_subject_target_sidecar_v5_fresh_seed.py \
  --model-path "${MODEL_PATH}" \
  --candidate-run-dir "${V5_CANDIDATE_OUTPUT_DIR}" \
  --mcf-path "${MCF_PATH}" \
  --wikidata-dir "${WIKIDATA_DIR}" \
  --confirmation-registry "${CONFIRMATION_REGISTRY}" \
  --output-dir "${OFFICIAL_OUTPUT_DIR}" \
  --seed "${SEED}" --route-audit-batch-size 128 \
  --dtype bf16 --device-map single

jq -e --argjson seed "${SEED}" '
  .seed == $seed and
  .status == "completed" and
  .official_evaluation_opened == true and
  .official_evaluation_completed == true and
  .retry_or_resume_permitted == false
' "${OFFICIAL_OUTPUT_DIR}/terminal_status.json" >/dev/null

jq '{
  seed,
  passed,
  base: .arms.reconstructed_base,
  v5: .arms.v5_sidecar,
  behavioral_checks,
  route_audit,
  exact_preservation,
  runtime_integrity,
  retry_or_resume_permitted,
  claim_scope
}' "${OFFICIAL_OUTPUT_DIR}/official_evaluation.json"
