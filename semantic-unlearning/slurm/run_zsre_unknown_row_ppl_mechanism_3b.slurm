#!/usr/bin/env bash
#SBATCH --job-name=zsre-unk-ppl
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --no-requeue
#SBATCH --output=slurm_logs/zsre_unknown_ppl_%j.out
#SBATCH --error=slurm_logs/zsre_unknown_ppl_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/yl258/kp759/Unlearning/semantic-unlearning}"
ENV_DIR="${ENV_DIR:-/scratch/yl258/kp759/conda_envs/semantic_unlearning}"

cd "${PROJECT_DIR}"
mkdir -p slurm_logs

export PATH="${ENV_DIR}/bin:${PATH}"
export HF_HOME="${HF_HOME:-/scratch/yl258/kp759/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

export ZSRE_STAGEWISE_ROOT="${ZSRE_STAGEWISE_ROOT:-outputs/zsre_stagewise_ppl_diag_seeds1_10}"
export ZSRE_DIAG_SEEDS="${ZSRE_DIAG_SEEDS:-1 10}"
export WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
export DTYPE="${DTYPE:-bf16}"

echo "Root=${ZSRE_STAGEWISE_ROOT}"
echo "Seeds=${ZSRE_DIAG_SEEDS}"
echo "Mechanism diagnostic only: fixed Wikidata PPL text; no held-out ZsRE probes."

bash scripts/run_zsre_unknown_row_ppl_mechanism.sh
