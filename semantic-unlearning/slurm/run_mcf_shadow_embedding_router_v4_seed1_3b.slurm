#!/usr/bin/env bash
#SBATCH --job-name=mcf-shadow-router-v4
#SBATCH --account=yl258
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --no-requeue
#SBATCH --output=slurm_logs/mcf_shadow_embedding_router_v4_%j.out
#SBATCH --error=slurm_logs/mcf_shadow_embedding_router_v4_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/yl258/kp759/Unlearning/semantic-unlearning}"
ENV_DIR="${ENV_DIR:-/scratch/yl258/kp759/conda_envs/semantic_unlearning}"
MODEL_PATH="${MODEL_PATH:-/scratch/yl258/kp759/hf-materialized/Llama-3.2-3B-Instruct-clean}"
WIKIDATA_DIR="${WIKIDATA_DIR:-/scratch/yl258/kp759/datasets/wikipedia_sure_50020}"
CLEAN_WRITER_OUTPUT_DIR="${CLEAN_WRITER_OUTPUT_DIR:-outputs/mcf_compositional_marker_v6_2_clean_seed1_3b}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mcf_shadow_embedding_router_v4_seed1_3b}"

TRAINING_VISIBLE="${TRAINING_VISIBLE:-${CLEAN_WRITER_OUTPUT_DIR}/protocol/training_visible_target_aware_direct.json}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${CLEAN_WRITER_OUTPUT_DIR}/protocol/split_manifest.json}"
CONTEXT_MANIFEST="${CONTEXT_MANIFEST:-${CLEAN_WRITER_OUTPUT_DIR}/method/context_manifest.json}"
STAGE1_STATE="${STAGE1_STATE:-${CLEAN_WRITER_OUTPUT_DIR}/method/stage1_writer.pt}"
STAGE1_REPORT="${STAGE1_REPORT:-${CLEAN_WRITER_OUTPUT_DIR}/method/stage1_writer_report.json}"
STAGE1_LOG="${STAGE1_LOG:-${CLEAN_WRITER_OUTPUT_DIR}/method/stage1_writer_log.jsonl}"
STAGE1_PORTABILITY="${STAGE1_PORTABILITY:-${CLEAN_WRITER_OUTPUT_DIR}/method/training_safe_portability_preflight.json}"
STAGE1_ACCEPTANCE="${STAGE1_ACCEPTANCE:-${CLEAN_WRITER_OUTPUT_DIR}/method/clean_stage1_acceptance.json}"
REGISTRY="${REGISTRY:-protocols/mcf_shadow_embedding_router_v4_0_registry.json}"

cd "${PROJECT_DIR}"
export PATH="${ENV_DIR}/bin:${PATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "output already exists; choose a fresh V4 directory: ${OUTPUT_DIR}" >&2
  exit 2
fi
for required in \
  "${TRAINING_VISIBLE}" "${SPLIT_MANIFEST}" "${CONTEXT_MANIFEST}" \
  "${STAGE1_STATE}" "${STAGE1_REPORT}" "${STAGE1_LOG}" \
  "${STAGE1_PORTABILITY}" "${STAGE1_ACCEPTANCE}" "${REGISTRY}"
do
  test -f "${required}" || {
    echo "required frozen training artifact is missing: ${required}" >&2
    exit 2
  }
done
if ! jq -e '
  .protocol == "mcf_passive_shadow_embedding_context_router_v4_0" and
  .schema_version == 1 and
  .seed_1_v3_6_2_official_result_consumed == true and
  .seed_1_blind_official_reuse_prohibited == true and
  .architecture.base_embedding_main_path == true and
  .architecture.v6_2_delta_shadow_path_only == true and
  .architecture.layer == 27 and
  .official_evaluation_prohibited == true
' "${REGISTRY}" >/dev/null; then
  echo "V4 shadow-router registry is stale: ${REGISTRY}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}/protocol" slurm_logs
cp "${REGISTRY}" "${OUTPUT_DIR}/protocol/experiment_registry.json"

env -u MCF_PATH -u OFFICIAL_MCF_PATH -u OFFICIAL_EVAL_PATH \
  -u PARAPHRASE_PATH -u NEIGHBORHOOD_PATH -u RETAIN_EVAL_PATH \
  -u ADVERSARIAL_EVAL_PATH \
  python -u scripts/train_mcf_shadow_embedding_router_v4.py \
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
  --wikidata-dir "${WIKIDATA_DIR}" \
  --output-dir "${OUTPUT_DIR}/method" \
  --seed 1 --forget-num 50 --layer 27 \
  --frequency-doc-start 20 --frequency-docs 4096 --corpus-prefixes 256 \
  --router-steps 400 --router-lr 0.01 --router-weight-decay 0.0001 \
  --router-row-norm-cap 32 \
  --router-training-positive-floor 1.0 \
  --router-training-negative-ceiling -0.25 \
  --router-certificate-positive-floor 0.25 \
  --router-certificate-negative-ceiling -0.25 \
  --router-tail-k 2 \
  --actuator-steps 100 --actuator-lr 0.01 \
  --actuator-relative-cap 1.5 --forget-margin 1.0 \
  --batch-size 8 --check-every 5 --identity-protected-prompts 512 \
  --minimum-writer-necessity-fraction 0.5 \
  --dtype bf16 --device-map single --save-candidate

METHOD_DIR="${OUTPUT_DIR}/method"
for required in \
  training_firewall_receipt.json development_prompt_manifest.json \
  router_training_log.json router_certificate.json actuator_training.json \
  final_training_only_audit.json completion.json v4_shadow_embedding_candidate.pt
do
  test -f "${METHOD_DIR}/${required}" || {
    echo "passing V4 run is missing ${required}" >&2
    exit 2
  }
done
if ! jq -e '
  .passed == true and
  .architecture.embedding_retained == true and
  .architecture.embedding_permanently_materialized == false and
  .candidate_saved == true and
  .eligible_for_separately_preregistered_official_evaluation == true and
  .official_evaluation_allowed_in_this_process == false and
  .official_evaluation_prompts_seen == 0
' "${METHOD_DIR}/completion.json" >/dev/null; then
  echo "V4 did not pass every locked training-only gate" >&2
  exit 2
fi

jq '{
  passed,
  architecture,
  candidate_saved,
  candidate_sha256,
  eligible_for_separately_preregistered_official_evaluation,
  official_evaluation_prompts_seen,
  conclusion
}' "${METHOD_DIR}/completion.json"
