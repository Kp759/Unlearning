#!/usr/bin/env bash
#SBATCH --job-name=mcf-scope-agg
#SBATCH --account=yl258
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --time=00:20:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm_logs/mcf_scoped_bias_aggregate_%j.out
#SBATCH --error=slurm_logs/mcf_scoped_bias_aggregate_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/yl258/kp759/Unlearning/semantic-unlearning}"
ENV_DIR="${ENV_DIR:-/scratch/yl258/kp759/conda_envs/semantic_unlearning}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mcf_scoped_bias_confirmatory_v1}"

cd "${PROJECT_DIR}"
mkdir -p slurm_logs
export PATH="${ENV_DIR}/bin:${PATH}"

python -u scripts/aggregate_mcf_scoped_bias_confirmatory.py \
  --root "${OUTPUT_ROOT}" \
  --seeds 2 3 4 5 6 7 8 9 10
