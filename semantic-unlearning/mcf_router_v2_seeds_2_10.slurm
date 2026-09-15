#!/bin/bash
#SBATCH --job-name=mcf_v2_s2_10
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=2-10%4
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_v2_seed_%A_%a.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_v2_seed_%A_%a.err

set -euo pipefail

module purge
module load Miniforge3

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/yl258/kp759/conda_envs/semantic_unlearning

cd /scratch/yl258/kp759/Unlearning/semantic-unlearning

mkdir -p logs

SEED="${SLURM_ARRAY_TASK_ID}"

echo "=========================================="
echo "MCF Router V2 Confirmatory Seed ${SEED}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

echo "Git commit:"
git rev-parse HEAD

echo "GPU:"
nvidia-smi

bash scripts/run_mcf_router_v2_confirmatory_seed.sh "${SEED}"

echo "=========================================="
echo "Completed seed ${SEED}"
echo "Finish: $(date)"
echo "=========================================="
