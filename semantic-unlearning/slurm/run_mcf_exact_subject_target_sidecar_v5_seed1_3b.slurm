#!/usr/bin/env bash
#SBATCH --job-name=mcf-exact-subject-v5
#SBATCH --account=yl258
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --no-requeue
#SBATCH --output=slurm_logs/mcf_exact_subject_target_sidecar_v5_%j.out
#SBATCH --error=slurm_logs/mcf_exact_subject_target_sidecar_v5_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/yl258/kp759/Unlearning/semantic-unlearning}"
ENV_DIR="${ENV_DIR:-/scratch/yl258/kp759/conda_envs/semantic_unlearning}"
MODEL_PATH="${MODEL_PATH:-/scratch/yl258/kp759/hf-materialized/Llama-3.2-3B-Instruct-clean}"
SEED="${SEED:-1}"
CLEAN_WRITER_OUTPUT_DIR="${CLEAN_WRITER_OUTPUT_DIR:-outputs/mcf_compositional_marker_v6_2_clean_seed${SEED}_3b}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mcf_exact_subject_target_sidecar_v5_seed${SEED}_3b}"

TRAINING_VISIBLE="${TRAINING_VISIBLE:-${CLEAN_WRITER_OUTPUT_DIR}/protocol/training_visible_target_aware_direct.json}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${CLEAN_WRITER_OUTPUT_DIR}/protocol/split_manifest.json}"
CONTEXT_MANIFEST="${CONTEXT_MANIFEST:-${CLEAN_WRITER_OUTPUT_DIR}/method/context_manifest.json}"
STAGE1_STATE="${STAGE1_STATE:-${CLEAN_WRITER_OUTPUT_DIR}/method/stage1_writer.pt}"
STAGE1_REPORT="${STAGE1_REPORT:-${CLEAN_WRITER_OUTPUT_DIR}/method/stage1_writer_report.json}"
STAGE1_LOG="${STAGE1_LOG:-${CLEAN_WRITER_OUTPUT_DIR}/method/stage1_writer_log.jsonl}"
STAGE1_PORTABILITY="${STAGE1_PORTABILITY:-${CLEAN_WRITER_OUTPUT_DIR}/method/training_safe_portability_preflight.json}"
STAGE1_ACCEPTANCE="${STAGE1_ACCEPTANCE:-${CLEAN_WRITER_OUTPUT_DIR}/method/clean_stage1_acceptance.json}"
REGISTRY="${REGISTRY:-protocols/mcf_exact_subject_target_sidecar_v5_0_registry.json}"
RELATION_LEXICON="${RELATION_LEXICON:-protocols/mcf_frozen_relation_suffix_lexicon_v1.json}"

cd "${PROJECT_DIR}"
export PATH="${ENV_DIR}/bin:${PATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "output already exists; choose a fresh V5 directory: ${OUTPUT_DIR}" >&2
  exit 2
fi
for required in \
  "${TRAINING_VISIBLE}" "${SPLIT_MANIFEST}" "${CONTEXT_MANIFEST}" \
  "${STAGE1_STATE}" "${STAGE1_REPORT}" "${STAGE1_LOG}" \
  "${STAGE1_PORTABILITY}" "${STAGE1_ACCEPTANCE}" "${REGISTRY}" \
  "${RELATION_LEXICON}"
do
  test -f "${required}" || {
    echo "required frozen artifact is missing: ${required}" >&2
    exit 2
  }
done

mkdir -p "${OUTPUT_DIR}/protocol" slurm_logs
cp "${REGISTRY}" "${OUTPUT_DIR}/protocol/experiment_registry.json"
cp "${RELATION_LEXICON}" "${OUTPUT_DIR}/protocol/relation_suffix_lexicon.json"

env -u MCF_PATH -u OFFICIAL_MCF_PATH -u OFFICIAL_EVAL_PATH \
  -u PARAPHRASE_PATH -u NEIGHBORHOOD_PATH -u RETAIN_EVAL_PATH \
  -u ADVERSARIAL_EVAL_PATH \
  python -u scripts/build_mcf_exact_subject_target_sidecar_v5.py \
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
  --relation-lexicon "${RELATION_LEXICON}" \
  --output-dir "${OUTPUT_DIR}/method" \
  --seed "${SEED}" --forget-num 50 --logit-bias 256 \
  --margin-floor 0.1 --batch-size 4 --dtype bf16 --device-map single

METHOD_DIR="${OUTPUT_DIR}/method"
for required in completion.json route_audit.json direct_training_visible_audit.json \
  closed_route_identity_audit.json integrity.json v5_exact_subject_target_sidecar.pt
do
  test -f "${METHOD_DIR}/${required}" || {
    echo "passing V5 build is missing ${required}" >&2
    exit 2
  }
done
jq -e '
  .passed == true and
  .architecture.embedding_parameters_mutated == false and
  .architecture.lm_head_parameters_mutated == false and
  .architecture.learned_detector_parameters == 0 and
  .architecture.learned_actuator_parameters == 0 and
  .direct_training_visible_audit.passed == true and
  .closed_route_identity_audit.passed == true and
  .candidate_saved == true and
  .official_evaluation_allowed_in_this_process == false and
  .official_evaluation_prompts_seen == 0
' "${METHOD_DIR}/completion.json" >/dev/null

jq '{passed, architecture, direct_training_visible_audit, candidate_sha256, claim_scope}' \
  "${METHOD_DIR}/completion.json"
