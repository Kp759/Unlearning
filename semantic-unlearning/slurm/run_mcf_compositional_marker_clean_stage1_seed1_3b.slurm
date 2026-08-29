#!/usr/bin/env bash
#SBATCH --job-name=mcf-clean-writer-s1
#SBATCH --account=yl258
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --no-requeue
#SBATCH --output=slurm_logs/mcf_clean_writer_seed1_%j.out
#SBATCH --error=slurm_logs/mcf_clean_writer_seed1_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/yl258/kp759/Unlearning/semantic-unlearning}"
ENV_DIR="${ENV_DIR:-/scratch/yl258/kp759/conda_envs/semantic_unlearning}"
MODEL_PATH="${MODEL_PATH:-/scratch/yl258/kp759/hf-materialized/Llama-3.2-3B-Instruct-clean}"
WIKIDATA_DIR="${WIKIDATA_DIR:-/scratch/yl258/kp759/datasets/wikipedia_sure_50020}"
MCF_PATH="${MCF_PATH:-data/multi_counterfact.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mcf_compositional_marker_v6_1_clean_seed1_3b}"

cd "${PROJECT_DIR}"
export PATH="${ENV_DIR}/bin:${PATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "clean writer output path exists; refusing overwrite: ${OUTPUT_DIR}" >&2
  exit 2
fi
python -c 'import sys; sys.path.insert(0, "scripts"); import mcf_compositional_marker_core as c; assert c.PROTOCOL == "mcf_context_composed_sparse_embedding_writer_v6_1", c.PROTOCOL' \
  2>/dev/null || {
    echo "checkout is stale: clean Stage-1 v6 code is not importable" >&2
    exit 2
  }

mkdir -p slurm_logs "${OUTPUT_DIR}/protocol" "${OUTPUT_DIR}/method"
python -u scripts/build_mcf_sure_target_aware_direct_split.py \
  --mcf-path "${MCF_PATH}" \
  --output-dir "${OUTPUT_DIR}/protocol" \
  --seed 1 \
  --forget-num 50 \
  --retain-eval-num 1000

TRAINING_VISIBLE="${OUTPUT_DIR}/protocol/training_visible_target_aware_direct.json"
SPLIT_MANIFEST="${OUTPUT_DIR}/protocol/split_manifest.json"

env -u MCF_PATH -u OFFICIAL_MCF_PATH -u OFFICIAL_EVAL_PATH \
  -u PARAPHRASE_PATH -u NEIGHBORHOOD_PATH -u RETAIN_EVAL_PATH \
  -u ADVERSARIAL_EVAL_PATH -u SURROGATE_ARTIFACT \
  -u RESUME_STAGE1_STATE \
  python -u scripts/mcf_compositional_marker_write_read.py \
  --model-path "${MODEL_PATH}" \
  --training-visible-path "${TRAINING_VISIBLE}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --output-dir "${OUTPUT_DIR}/method" \
  --seed 1 \
  --forget-num 50 \
  --wikidata-dir "${WIKIDATA_DIR}" \
  --positive-context-policy relation_templates_only_v1 \
  --synthetic-paraphrases-per-record 6 \
  --corpus-context-prefixes 256 \
  --reach-probes 12 \
  --writer-record-batch 3 \
  --positive-context-batch 7 \
  --writer-positive-context-mode all \
  --writer-positive-tail-k 2 \
  --negative-context-batch 5 \
  --writer-steps 1200 \
  --writer-lr 2e-4 \
  --row-norm-cap 8 \
  --row-norm-cap-frequency-alpha 0.15 \
  --max-subject-token-frequency 1000000000 \
  --write-alpha 6 \
  --lambda-consistency 2 \
  --lambda-negative 10 \
  --lambda-cross 2 \
  --stage1-only \
  --dtype bf16 \
  --device-map single

TRAINING_SAFE_PORTABILITY="${OUTPUT_DIR}/method/training_safe_portability_preflight.json"
env -u MCF_PATH -u OFFICIAL_MCF_PATH -u OFFICIAL_EVAL_PATH \
  -u PARAPHRASE_PATH -u NEIGHBORHOOD_PATH -u RETAIN_EVAL_PATH \
  -u ADVERSARIAL_EVAL_PATH -u SURROGATE_ARTIFACT \
  python -u scripts/audit_mcf_clean_stage1_training_preflight.py \
  --model-path "${MODEL_PATH}" \
  --context-manifest "${OUTPUT_DIR}/method/context_manifest.json" \
  --stage1-state "${OUTPUT_DIR}/method/stage1_writer.pt" \
  --stage1-report "${OUTPUT_DIR}/method/stage1_writer_report.json" \
  --stage1-writer-log "${OUTPUT_DIR}/method/stage1_writer_log.jsonl" \
  --out "${TRAINING_SAFE_PORTABILITY}" \
  --amplitude-threshold 4.5 \
  --minimum-global-fraction 0.95 \
  --minimum-record-fraction 0.80 \
  --batch-size 8 \
  --dtype bf16 \
  --device-map single

python -u scripts/verify_mcf_clean_stage1_writer.py \
  --training-visible-path "${TRAINING_VISIBLE}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --context-manifest "${OUTPUT_DIR}/method/context_manifest.json" \
  --stage1-state "${OUTPUT_DIR}/method/stage1_writer.pt" \
  --stage1-report "${OUTPUT_DIR}/method/stage1_writer_report.json" \
  --stage1-writer-log "${OUTPUT_DIR}/method/stage1_writer_log.jsonl" \
  --training-safe-portability "${TRAINING_SAFE_PORTABILITY}" \
  --out "${OUTPUT_DIR}/method/clean_stage1_acceptance.json"

echo "clean Stage-1 writer frozen: ${OUTPUT_DIR}/method/stage1_writer.pt"
echo "acceptance receipt: ${OUTPUT_DIR}/method/clean_stage1_acceptance.json"
echo "official evaluation opened: no"
