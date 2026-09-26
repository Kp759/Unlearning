#!/bin/bash
# MCF layer-wise study on Wulver: one array task per layer.
#   sbatch mcf_layer_sweep.slurm
# Llama-3.2-3B has 28 blocks (0-27). Task -> layer:
#   0:3 (early)  1:7  2:13 (middle)  3:19 (shipped, oracle-trained control)
#   4:23  5:27 (last block, pre final norm)
# Override the list with:  sbatch --export=ALL,LAYERS="2 9 16 27" --array=0-3 mcf_layer_sweep.slurm
#SBATCH --job-name=mcf_layers
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-5%3
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_layers_%A_%a.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_layers_%A_%a.err

set -euo pipefail

module purge
module load Miniforge3

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/yl258/kp759/conda_envs/semantic_unlearning

cd /scratch/yl258/kp759/Unlearning/semantic-unlearning
mkdir -p logs

read -r -a LAYER_LIST <<< "${LAYERS:-3 7 13 19 23 27}"
LAYER="${LAYER_LIST[$SLURM_ARRAY_TASK_ID]:?no layer for array task $SLURM_ARRAY_TASK_ID}"

echo "=========================================="
echo "MCF layer sweep: layer ${LAYER} (task ${SLURM_ARRAY_TASK_ID})"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}  Node: $(hostname)"
echo "Start: $(date)"
echo "Git commit: $(git rev-parse HEAD)"
echo "=========================================="
nvidia-smi

bash scripts/run_mcf_layer_sweep_one.sh "${LAYER}"

echo "Finish: $(date)"
