#!/usr/bin/env bash
#SBATCH --job-name=mcf-v5-seed1-dev
#SBATCH --account=yl258
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --no-requeue
#SBATCH --output=slurm_logs/mcf_v5_seed1_development_%j.out
#SBATCH --error=slurm_logs/mcf_v5_seed1_development_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/yl258/kp759/Unlearning/semantic-unlearning}"
ENV_DIR="${ENV_DIR:-/scratch/yl258/kp759/conda_envs/semantic_unlearning}"
MODEL_PATH="${MODEL_PATH:-/scratch/yl258/kp759/hf-materialized/Llama-3.2-3B-Instruct-clean}"
WIKIDATA_DIR="${WIKIDATA_DIR:-/scratch/yl258/kp759/datasets/wikipedia_sure_50020}"
MCF_PATH="${MCF_PATH:-${PROJECT_DIR}/data/multi_counterfact.json}"
V5_CANDIDATE_OUTPUT_DIR="${V5_CANDIDATE_OUTPUT_DIR:-outputs/mcf_exact_subject_target_sidecar_v5_seed1_3b}"
DEVELOPMENT_OUTPUT_DIR="${DEVELOPMENT_OUTPUT_DIR:-outputs/mcf_exact_subject_target_sidecar_v5_seed1_development_3b}"
RECOVERY_PROTOCOL="${RECOVERY_PROTOCOL:-protocols/mcf_embedding_keyed_neuron_v3_6_2_official_recovery_v1.json}"

cd "${PROJECT_DIR}"
export PATH="${ENV_DIR}/bin:${PATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

if [[ -e "${DEVELOPMENT_OUTPUT_DIR}" ]]; then
  echo "development output already exists: ${DEVELOPMENT_OUTPUT_DIR}" >&2
  exit 2
fi
for required in \
  "${MCF_PATH}" "${RECOVERY_PROTOCOL}" \
  "${V5_CANDIDATE_OUTPUT_DIR}/method/completion.json" \
  "${V5_CANDIDATE_OUTPUT_DIR}/method/v5_exact_subject_target_sidecar.pt" \
  "${V5_CANDIDATE_OUTPUT_DIR}/protocol/experiment_registry.json"
do
  test -f "${required}" || {
    echo "required V5 development input is missing: ${required}" >&2
    exit 2
  }
done

python -u scripts/evaluate_mcf_exact_subject_target_sidecar_v5_seed1_development.py \
  --model-path "${MODEL_PATH}" \
  --candidate-run-dir "${V5_CANDIDATE_OUTPUT_DIR}" \
  --mcf-path "${MCF_PATH}" \
  --wikidata-dir "${WIKIDATA_DIR}" \
  --recovery-protocol "${RECOVERY_PROTOCOL}" \
  --output-dir "${DEVELOPMENT_OUTPUT_DIR}" \
  --route-audit-batch-size 128 --dtype bf16 --device-map single

jq -e '
  .passed == true and
  .evaluation_status == "consumed_development_not_blind_not_official" and
  .arms.v5_sidecar.forget.Eff == 0 and
  .arms.v5_sidecar.forget.Gen == 0 and
  .behavioral_checks.forget_specificity_exact_base == true and
  .behavioral_checks.retain_exact_base == true and
  .behavioral_checks.ppl_exact_base == true and
  .fresh_seed_confirmation_required == true and
  .blind_evaluation_claim_permitted == false
' "${DEVELOPMENT_OUTPUT_DIR}/development_replay.json" >/dev/null

jq '{passed, evaluation_status, base:.arms.base, v5:.arms.v5_sidecar, behavioral_checks, claim_scope}' \
  "${DEVELOPMENT_OUTPUT_DIR}/development_replay.json"
