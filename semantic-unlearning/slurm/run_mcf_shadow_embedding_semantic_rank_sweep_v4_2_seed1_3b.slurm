#!/usr/bin/env bash
#SBATCH --job-name=mcf-shadow-rank-v42
#SBATCH --account=yl258
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --no-requeue
#SBATCH --output=slurm_logs/mcf_shadow_embedding_semantic_rank_sweep_v4_2_%j.out
#SBATCH --error=slurm_logs/mcf_shadow_embedding_semantic_rank_sweep_v4_2_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/yl258/kp759/Unlearning/semantic-unlearning}"
ENV_DIR="${ENV_DIR:-/scratch/yl258/kp759/conda_envs/semantic_unlearning}"
MODEL_PATH="${MODEL_PATH:-/scratch/yl258/kp759/hf-materialized/Llama-3.2-3B-Instruct-clean}"
WIKIDATA_DIR="${WIKIDATA_DIR:-/scratch/yl258/kp759/datasets/wikipedia_sure_50020}"
CLEAN_WRITER_OUTPUT_DIR="${CLEAN_WRITER_OUTPUT_DIR:-outputs/mcf_compositional_marker_v6_2_clean_seed1_3b}"
FAILED_V4_OUTPUT_DIR="${FAILED_V4_OUTPUT_DIR:-outputs/mcf_shadow_embedding_router_v4_seed1_3b}"
FAILED_V4_1_OUTPUT_DIR="${FAILED_V4_1_OUTPUT_DIR:-outputs/mcf_shadow_embedding_semantic_router_v4_1_seed1_3b}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mcf_shadow_embedding_semantic_rank_sweep_v4_2_seed1_3b}"

TRAINING_VISIBLE="${TRAINING_VISIBLE:-${CLEAN_WRITER_OUTPUT_DIR}/protocol/training_visible_target_aware_direct.json}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${CLEAN_WRITER_OUTPUT_DIR}/protocol/split_manifest.json}"
CONTEXT_MANIFEST="${CONTEXT_MANIFEST:-${CLEAN_WRITER_OUTPUT_DIR}/method/context_manifest.json}"
STAGE1_STATE="${STAGE1_STATE:-${CLEAN_WRITER_OUTPUT_DIR}/method/stage1_writer.pt}"
STAGE1_REPORT="${STAGE1_REPORT:-${CLEAN_WRITER_OUTPUT_DIR}/method/stage1_writer_report.json}"
STAGE1_LOG="${STAGE1_LOG:-${CLEAN_WRITER_OUTPUT_DIR}/method/stage1_writer_log.jsonl}"
STAGE1_PORTABILITY="${STAGE1_PORTABILITY:-${CLEAN_WRITER_OUTPUT_DIR}/method/training_safe_portability_preflight.json}"
STAGE1_ACCEPTANCE="${STAGE1_ACCEPTANCE:-${CLEAN_WRITER_OUTPUT_DIR}/method/clean_stage1_acceptance.json}"
REGISTRY="${REGISTRY:-protocols/mcf_shadow_embedding_semantic_rank_sweep_v4_2_registry.json}"

cd "${PROJECT_DIR}"
export PATH="${ENV_DIR}/bin:${PATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "output already exists; choose a fresh V4.2 directory: ${OUTPUT_DIR}" >&2
  exit 2
fi
for required in \
  "${TRAINING_VISIBLE}" "${SPLIT_MANIFEST}" "${CONTEXT_MANIFEST}" \
  "${STAGE1_STATE}" "${STAGE1_REPORT}" "${STAGE1_LOG}" \
  "${STAGE1_PORTABILITY}" "${STAGE1_ACCEPTANCE}" "${REGISTRY}" \
  "${FAILED_V4_OUTPUT_DIR}/method/router_certificate.json" \
  "${FAILED_V4_OUTPUT_DIR}/method/completion.json" \
  "${FAILED_V4_OUTPUT_DIR}/method/training_firewall_receipt.json" \
  "${FAILED_V4_OUTPUT_DIR}/method/development_prompt_manifest.json" \
  "${FAILED_V4_OUTPUT_DIR}/protocol/experiment_registry.json" \
  "${FAILED_V4_1_OUTPUT_DIR}/method/completion.json" \
  "${FAILED_V4_1_OUTPUT_DIR}/method/semantic_capacity_report.json" \
  "${FAILED_V4_1_OUTPUT_DIR}/method/semantic_router_training_log.json" \
  "${FAILED_V4_1_OUTPUT_DIR}/method/factorized_router_preflight.json" \
  "${FAILED_V4_1_OUTPUT_DIR}/method/marker_preflight.json" \
  "${FAILED_V4_1_OUTPUT_DIR}/method/training_firewall_receipt.json" \
  "${FAILED_V4_1_OUTPUT_DIR}/method/development_prompt_manifest.json" \
  "${FAILED_V4_1_OUTPUT_DIR}/protocol/experiment_registry.json"
do
  test -f "${required}" || {
    echo "required frozen artifact is missing: ${required}" >&2
    exit 2
  }
done
if ! jq -e '
  .protocol == "mcf_shadow_marker_base_semantic_rank_sweep_v4_2" and
  .schema_version == 1 and
  .source_v4_0_rejection_required == true and
  .source_v4_1_rejection_required == true and
  .architecture.rank_arms == [4, 8, 16, "full_relation_contrast_rank_capped_at_32"] and
  .architecture.semantic_basis_trainable == false and
  .architecture.actuator_constructed == false and
  .architecture.layer == 27 and
  .development_prompt_policy.fit_wrong_relation_negatives_per_record_total == 160 and
  .official_evaluation_prohibited == true
' "${REGISTRY}" >/dev/null; then
  echo "V4.2 rank-sweep registry is stale: ${REGISTRY}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}/protocol" slurm_logs
cp "${REGISTRY}" "${OUTPUT_DIR}/protocol/experiment_registry.json"

env -u MCF_PATH -u OFFICIAL_MCF_PATH -u OFFICIAL_EVAL_PATH \
  -u PARAPHRASE_PATH -u NEIGHBORHOOD_PATH -u RETAIN_EVAL_PATH \
  -u ADVERSARIAL_EVAL_PATH \
  python -u scripts/run_mcf_shadow_embedding_semantic_rank_sweep_v4_2.py \
  --model-path "${MODEL_PATH}" \
  --training-visible-path "${TRAINING_VISIBLE}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --context-manifest "${CONTEXT_MANIFEST}" \
  --stage1-state "${STAGE1_STATE}" \
  --stage1-report "${STAGE1_REPORT}" \
  --stage1-writer-log "${STAGE1_LOG}" \
  --clean-stage1-portability-preflight "${STAGE1_PORTABILITY}" \
  --clean-stage1-acceptance "${STAGE1_ACCEPTANCE}" \
  --failed-v4-run-dir "${FAILED_V4_OUTPUT_DIR}" \
  --failed-v4-1-run-dir "${FAILED_V4_1_OUTPUT_DIR}" \
  --experiment-registry "${REGISTRY}" \
  --wikidata-dir "${WIKIDATA_DIR}" \
  --output-dir "${OUTPUT_DIR}/method" \
  --seed 1 --forget-num 50 --layer 27 \
  --frequency-doc-start 20 --frequency-docs 4096 --corpus-prefixes 256 \
  --marker-rms-threshold 0.000001 \
  --semantic-router-steps 300 --semantic-router-lr 0.01 \
  --semantic-router-weight-decay 0.01 \
  --semantic-router-coefficient-norm-cap 4 \
  --semantic-selection-check-every 10 --semantic-selection-patience 10 \
  --semantic-training-positive-floor 1.0 \
  --semantic-training-negative-ceiling=-1.0 \
  --semantic-certificate-positive-floor 0.25 \
  --semantic-certificate-negative-ceiling=-0.25 \
  --semantic-tail-k 4 \
  --fit-positive-variants-per-old-family 8 \
  --fit-wrong-relations-per-old-family 64 \
  --consumed-v4-1-positive-variants 4 \
  --consumed-v4-1-wrong-relations 32 \
  --development-positive-variants 4 \
  --development-wrong-relations 64 \
  --certification-positive-variants 4 \
  --certification-wrong-relations 64 \
  --batch-size 8 --dtype bf16 --device-map single

METHOD_DIR="${OUTPUT_DIR}/method"
for required in \
  frozen_v4_rejection_import.json frozen_v4_1_rejection_import.json \
  training_firewall_receipt.json development_prompt_manifest.json \
  marker_preflight.json semantic_capacity_report.json \
  rank_sweep_report.json one_shot_certification.json \
  integrity_audit.json v4_2_certified_router_state.pt completion.json
do
  test -f "${METHOD_DIR}/${required}" || {
    echo "passing V4.2 run is missing ${required}" >&2
    exit 2
  }
done
if ! jq -e '
  .passed == true and
  .selected_smallest_passing_rank >= 4 and
  .certified_router_saved == true and
  .certification_open_count == 1 and
  .optimizer_steps_after_certification_open == 0 and
  .actuator_optimizer_constructed == false and
  .eligible_for_separate_actuator_training == true and
  .eligible_for_official_evaluation == false and
  .seed_1_blind_official_reuse_prohibited == true and
  .official_evaluation_prompts_seen == 0
' "${METHOD_DIR}/completion.json" >/dev/null; then
  echo "V4.2 failed one or more locked router-only checks" >&2
  exit 2
fi

jq '{
  passed,
  registered_rank_arms,
  selected_smallest_passing_rank,
  certified_router_saved,
  certified_router_sha256,
  eligible_for_separate_actuator_training,
  official_evaluation_prompts_seen,
  conclusion
}' "${METHOD_DIR}/completion.json"
