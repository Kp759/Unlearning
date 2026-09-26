#!/bin/bash
# MCF layer sweep (linear classifier, no V2) -- REGULAR, all layers in one job.
# Regular SURE: rows trained under the linear classifier's own routing.
#   sbatch mcf_layer_sweep_regular.slurm
# Resumes: finished layers/stages are skipped, half-written stages are moved
# aside and redone. Override layers with --export=ALL,LAYERS="3 7".
#SBATCH --job-name=mcf_regular
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_regular_%j.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_regular_%j.err

set -uo pipefail

module purge
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/yl258/kp759/conda_envs/semantic_unlearning

cd /scratch/yl258/kp759/Unlearning/semantic-unlearning
mkdir -p logs

export SWEEP_TAG=layer_sweep_linear_regular_v1
export TRAINING_ROUTE=router
export NORM_SCALE=1
export WITH_DECOMPOSITION=0

echo "=========================================="
echo "MCF layer sweep REGULAR | job ${SLURM_JOB_ID} | node $(hostname)"
echo "Start: $(date) | commit $(git rev-parse HEAD)"
echo "SWEEP_TAG=$SWEEP_TAG TRAINING_ROUTE=$TRAINING_ROUTE NORM_SCALE=$NORM_SCALE"
echo "=========================================="
nvidia-smi

FAILED=()
for LAYER in ${LAYERS:-1 3 7 13 19 23 27}; do
  echo "=========== REGULAR LAYER ${LAYER} | $(date) ==========="
  if ! bash scripts/run_mcf_layer_sweep_one.sh "$LAYER"; then
    echo "!!! LAYER ${LAYER} FAILED | $(date)"
    FAILED+=("$LAYER")
  fi
done

python scripts/summarize_mcf_layer_sweep.py \
  --sweep-dir "outputs/mcf_${SWEEP_TAG}" \
  --reference outputs/mcf_linear_2x2_seed1_v24/arms/linear_global || true

echo "Finish: $(date) | failed layers: ${FAILED[*]:-none}"
[[ ${#FAILED[@]} -eq 0 ]]
