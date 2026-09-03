#!/usr/bin/env bash
#SBATCH --job-name=mcf-v5-fresh-build
#SBATCH --account=yl258
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --no-requeue
#SBATCH --output=slurm_logs/mcf_v5_fresh_build_%j.out
#SBATCH --error=slurm_logs/mcf_v5_fresh_build_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/yl258/kp759/Unlearning/semantic-unlearning}"
ENV_DIR="${ENV_DIR:-/scratch/yl258/kp759/conda_envs/semantic_unlearning}"
MODEL_PATH="${MODEL_PATH:-/scratch/yl258/kp759/hf-materialized/Llama-3.2-3B-Instruct-clean}"
MCF_PATH="${MCF_PATH:-${PROJECT_DIR}/data/multi_counterfact.json}"
SEED="${SEED:?SEED is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
ARCHITECTURE_REGISTRY="${ARCHITECTURE_REGISTRY:-protocols/mcf_exact_subject_target_sidecar_v5_0_registry.json}"
CONFIRMATION_REGISTRY="${CONFIRMATION_REGISTRY:-protocols/mcf_exact_subject_target_sidecar_v5_fresh_multiseed_registry_v1.json}"
RELATION_LEXICON="${RELATION_LEXICON:-protocols/mcf_frozen_relation_suffix_lexicon_v1.json}"

cd "${PROJECT_DIR}"
export PATH="${ENV_DIR}/bin:${PATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

case "${SEED}" in
  2|3|4|5|6|7|8|9|10) ;;
  *) echo "fresh V5 seed must be 2 through 10" >&2; exit 2 ;;
esac
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "fresh V5 candidate output already exists: ${OUTPUT_DIR}" >&2
  exit 2
fi
for required in "${MCF_PATH}" "${ARCHITECTURE_REGISTRY}" \
  "${CONFIRMATION_REGISTRY}" "${RELATION_LEXICON}"
do
  test -f "${required}" || {
    echo "required fresh V5 input is missing: ${required}" >&2
    exit 2
  }
done

mkdir -p "${OUTPUT_DIR}/protocol" slurm_logs
python -u scripts/build_mcf_exact_subject_target_sidecar_v5_fresh_seed_split.py \
  --mcf-path "${MCF_PATH}" \
  --relation-lexicon "${RELATION_LEXICON}" \
  --output-dir "${OUTPUT_DIR}/protocol/seed_split" \
  --seed "${SEED}" --forget-num 50 --retain-num 1000

cp "${ARCHITECTURE_REGISTRY}" \
  "${OUTPUT_DIR}/protocol/experiment_registry.json"
cp "${CONFIRMATION_REGISTRY}" \
  "${OUTPUT_DIR}/protocol/fresh_multiseed_registry.json"
cp "${RELATION_LEXICON}" \
  "${OUTPUT_DIR}/protocol/relation_suffix_lexicon.json"

env -u MCF_PATH -u OFFICIAL_MCF_PATH -u OFFICIAL_EVAL_PATH \
  -u PARAPHRASE_PATH -u NEIGHBORHOOD_PATH -u RETAIN_EVAL_PATH \
  -u ADVERSARIAL_EVAL_PATH -u WIKIDATA_DIR \
  python -u scripts/build_mcf_exact_subject_target_sidecar_v5_fresh_seed.py \
  --model-path "${MODEL_PATH}" \
  --training-visible-path \
    "${OUTPUT_DIR}/protocol/seed_split/training_visible_target_aware_direct.json" \
  --split-manifest "${OUTPUT_DIR}/protocol/seed_split/split_manifest.json" \
  --architecture-registry "${OUTPUT_DIR}/protocol/experiment_registry.json" \
  --confirmation-registry "${OUTPUT_DIR}/protocol/fresh_multiseed_registry.json" \
  --relation-lexicon "${OUTPUT_DIR}/protocol/relation_suffix_lexicon.json" \
  --output-dir "${OUTPUT_DIR}/method" \
  --seed "${SEED}" --forget-num 50 --logit-bias 256 \
  --margin-floor 0.1 --batch-size 4 --dtype bf16 --device-map single

jq -e --argjson seed "${SEED}" '
  .passed == true and
  .seed == $seed and
  .standalone_direct_only_lineage == true and
  .v6_2_writer_artifacts_consumed == 0 and
  .architecture.embedding_parameters_mutated == false and
  .architecture.transformer_parameters_mutated == false and
  .architecture.lm_head_parameters_mutated == false and
  .architecture.learned_detector_parameters == 0 and
  .architecture.learned_actuator_parameters == 0 and
  .route_audit.passed == true and
  .direct_training_visible_audit.passed == true and
  .closed_route_identity_audit.passed == true and
  .eligible_for_one_shot_fresh_seed_evaluation == true and
  .official_evaluation_allowed_in_this_process == false and
  .official_evaluation_prompts_seen == 0
' "${OUTPUT_DIR}/method/completion.json" >/dev/null

jq '{seed, passed, standalone_direct_only_lineage, route_audit, direct_training_visible_audit, candidate_sha256, claim_scope}' \
  "${OUTPUT_DIR}/method/completion.json"
