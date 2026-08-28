#!/usr/bin/env bash
#SBATCH --job-name=mcf-scope-confirm
#SBATCH --account=yl258
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=2-10%2
#SBATCH --no-requeue
#SBATCH --output=slurm_logs/mcf_scoped_bias_confirm_%A_%a.out
#SBATCH --error=slurm_logs/mcf_scoped_bias_confirm_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/yl258/kp759/Unlearning/semantic-unlearning}"
ENV_DIR="${ENV_DIR:-/scratch/yl258/kp759/conda_envs/semantic_unlearning}"
MODEL_PATH="${MODEL_PATH:-/scratch/yl258/kp759/hf-materialized/Llama-3.2-3B-Instruct-clean}"
WIKIDATA_DIR="${WIKIDATA_DIR:-/scratch/yl258/kp759/datasets/wikipedia_sure_50020}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mcf_scoped_bias_confirmatory_v1}"
METHOD_TAG="mcf-scoped-bias-v1"
METHOD_COMMIT="9284d91"
SEED="${SLURM_ARRAY_TASK_ID:?This script must run as the declared seed 2-10 array}"

if (( SEED < 2 || SEED > 10 )); then
  echo "Seed ${SEED} is outside the frozen confirmatory range 2-10" >&2
  exit 2
fi

cd "${PROJECT_DIR}"
export PATH="${ENV_DIR}/bin:${PATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

TAG_COMMIT="$(git rev-parse "${METHOD_TAG}^{commit}")"
if [[ "${TAG_COMMIT}" != "$(git rev-parse "${METHOD_COMMIT}^{commit}")" ]]; then
  echo "Frozen tag ${METHOD_TAG} no longer resolves to ${METHOD_COMMIT}" >&2
  exit 2
fi

ALGORITHM_FILES=(
  scripts/build_mcf_scoped_bias_sidecar.py
  scripts/scoped_span_edit.py
  scripts/mcf_zero_unlearn_official_eval.py
)
if ! git diff --quiet "${METHOD_TAG}" -- "${ALGORITHM_FILES[@]}"; then
  echo "Algorithm files differ from frozen tag ${METHOD_TAG}; refusing confirmatory run" >&2
  git diff --stat "${METHOD_TAG}" -- "${ALGORITHM_FILES[@]}" >&2
  exit 2
fi

SEED_DIR="${OUTPUT_ROOT}/seed${SEED}"
if [[ -e "${SEED_DIR}/official_eval.json" || -e "${SEED_DIR}/base_official_eval.json" ]]; then
  echo "Seed ${SEED} already has final output; refusing overwrite" >&2
  exit 2
fi
mkdir -p slurm_logs "${SEED_DIR}"

{
  echo "execution_commit=$(git rev-parse HEAD)"
  echo "method_tag=${METHOD_TAG}"
  echo "method_commit=${TAG_COMMIT}"
  echo "seed=${SEED}"
  echo "penalty=512"
  echo "model=${MODEL_PATH}"
  echo "wikidata=${WIKIDATA_DIR}"
  python --version
  python -c 'import torch,transformers; print(f"torch={torch.__version__}"); print(f"transformers={transformers.__version__}")'
  nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
} > "${SEED_DIR}/execution_receipt.txt"

python -u scripts/build_mcf_scoped_bias_sidecar.py \
  --model-dir "${MODEL_PATH}" \
  --mcf-path data/multi_counterfact.json \
  --out "${SEED_DIR}/scoped_bias_full50.pt" \
  --seed "${SEED}" \
  --unlearn-num 50 \
  --sample-mode official \
  --writer-layer 8 \
  --penalty 512

# Base is evaluated under the identical model, split, corpus, dtype and scorer.
python -u scripts/mcf_zero_unlearn_official_eval.py \
  --model-dir "${MODEL_PATH}" \
  --mcf-path data/multi_counterfact.json \
  --wikidata-dir "${WIKIDATA_DIR}" \
  --out "${SEED_DIR}/base_official_eval.json" \
  --seed "${SEED}" \
  --sample-mode official \
  --dtype bfloat16 \
  --device-map auto \
  --quiet

python -u scripts/mcf_zero_unlearn_official_eval.py \
  --model-dir "${MODEL_PATH}" \
  --scoped-sidecar "${SEED_DIR}/scoped_bias_full50.pt" \
  --mcf-path data/multi_counterfact.json \
  --wikidata-dir "${WIKIDATA_DIR}" \
  --out "${SEED_DIR}/official_eval.json" \
  --seed "${SEED}" \
  --sample-mode official \
  --dtype bfloat16 \
  --device-map auto \
  --max-forget-eff 0 \
  --max-forget-gen 0 \
  --min-forget-margin 0.1 \
  --fail-if-gate-missed \
  --quiet

python -u scripts/aggregate_mcf_scoped_bias_confirmatory.py \
  --root "${OUTPUT_ROOT}" \
  --seeds "${SEED}" \
  --out-dir "${SEED_DIR}/validated_comparison"

echo "Frozen confirmatory seed ${SEED} complete"
