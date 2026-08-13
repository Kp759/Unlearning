#!/usr/bin/env bash
#SBATCH --job-name=zsre_gagd_aggr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=1-10
#SBATCH --output=slurm_logs/zsre_gagd_aggr_%A_%a.out
#SBATCH --error=slurm_logs/zsre_gagd_aggr_%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/yl258/kp759/Unlearning/semantic-unlearning}"
MODEL="${MODEL:-/scratch/yl258/kp759/models/Llama-3.2-3B-Instruct-0cb88a4f764b7a12671c53f0838cd831a0843b95-runtime}"
ZSRE_JSON="${ZSRE_JSON:-data/zsre_mend_eval.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/zsre_sure_no_neutral_gagd_aggressive_seeds1_10_3b}"

cd "${REPO_ROOT}"
mkdir -p slurm_logs

# Use the project environment when present; otherwise preserve the submitted shell PATH.
if [[ -d /scratch/yl258/kp759/conda_envs/semantic_unlearning/bin ]]; then
  export PATH=/scratch/yl258/kp759/conda_envs/semantic_unlearning/bin:${PATH}
fi

export MODEL ZSRE_JSON WIKIDATA_DIR OUTPUT_ROOT
export ZSRE_SEEDS="${SLURM_ARRAY_TASK_ID}"
export ZSRE_FORGET_NUM=50
export ZSRE_RETAIN_EVAL_NUM=1000

# Frozen aggressive no-neutral Emb+LM GA/GD Stage 1.
export ZSRE_STEPS=600
export ZSRE_BATCH_SIZE=1
export ZSRE_CACHE_BATCH_SIZE=8
export ZSRE_EMB_LM_LR=0.0001
export ZSRE_GA_WEIGHT=2.0
export ZSRE_GD_WEIGHT=1.0

# Frozen sparse active LM-head Stage 2.
export REPAIR_STEPS=800
export REPAIR_LR=0.005
export REPAIR_MARGIN=0.05
export REPAIR_L2=0.000001
export REPAIR_BATCH_SIZE=8
export EVAL_BATCH_SIZE=8
export DTYPE=bf16
export DEVICE_MAP=single

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

printf '===== ZsRE SURE aggressive GA/GD =====\n'
printf 'job=%s seed=%s host=%s\n' "${SLURM_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "$(hostname)"
printf 'model=%s\noutput=%s/seed%s\n' "${MODEL}" "${OUTPUT_ROOT}" "${SLURM_ARRAY_TASK_ID}"
python --version
nvidia-smi

test -f "${ZSRE_JSON}"
test -d "${WIKIDATA_DIR}"
test -d "${MODEL}"

# The aggressive wrapper hard-locks 600 / 1e-4 / GA=2 / GD=1.
bash scripts/run_zsre_sure_no_neutral_gagd_aggressive.sh \
  "${MODEL}" "${ZSRE_JSON}"

# Checkpoints are intentionally disposable. Delete them only after both final
# Stage-2 evaluation and post-hoc Stage-1 evaluation return successfully.
SEED_ROOT="${OUTPUT_ROOT}/seed${SLURM_ARRAY_TASK_ID}"
find "${SEED_ROOT}" -type d -name checkpoint -prune -exec rm -rf {} +

echo "===== seed ${SLURM_ARRAY_TASK_ID} complete; checkpoints removed ====="
