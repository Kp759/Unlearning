#!/bin/bash
#SBATCH --job-name=semantic-unlearning
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=outputs/logs/%j.out
#SBATCH --error=outputs/logs/%j.err

# Create log directory if it doesn't exist
mkdir -p outputs/logs

# Load required modules
module load python/3.11
module load cuda/11.8

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

echo "Starting semantic unlearning pipeline at $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"

python scripts/run_pipeline.py --config config/config.yaml

echo "Pipeline finished at $(date)"
