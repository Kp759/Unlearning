#!/usr/bin/env bash
#SBATCH --job-name=rwku-s5e600-r0-nr16-s0
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/logs/rwku_s5e600_rank0_nullrestore16_seed0/%j.out
#SBATCH --error=outputs/logs/rwku_s5e600_rank0_nullrestore16_seed0/%j.err

set -euo pipefail

REPOSITORY=/scratch/yl258/kp759/Unlearning/semantic-unlearning
MODEL=/scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95
EXPERIMENT_ID=rwku-s5e600-rank0-nullrestore16-sk-v3atomic-seed0-v1
OUTPUT_ROOT=$REPOSITORY/outputs/rwku_s5e600_rank0_nullrestore16
RECEIPT=$OUTPUT_ROOT/$EXPERIMENT_ID/checkpoint_receipt.json

mkdir -p "$REPOSITORY/outputs/logs/rwku_s5e600_rank0_nullrestore16_seed0"
cd "$REPOSITORY"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/yl258/kp759/conda_envs/semantic_unlearning

./scripts/run_rwku_generated_s5e600_rank0_nullrestore16_seed0.sh "$MODEL" preflight
./scripts/run_rwku_generated_s5e600_rank0_nullrestore16_seed0.sh "$MODEL" prepare
./scripts/run_rwku_generated_s5e600_rank0_nullrestore16_seed0.sh "$MODEL" train
./scripts/run_rwku_generated_s5e600_rank0_nullrestore16_seed0.sh "$MODEL" verify

python scripts/rwku_checkpoint_receipt.py verify --receipt "$RECEIPT"

# Protocol boundary: this job deliberately stops at CHECKPOINT_FROZEN.  It
# does not open or evaluate any official RWKU record.
