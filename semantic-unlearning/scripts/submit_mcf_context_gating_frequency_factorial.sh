#!/usr/bin/env bash
set -euo pipefail

mkdir -p slurm_logs
JOB_ID="$(sbatch --parsable slurm/run_mcf_context_gating_frequency_factorial_seed1_3b.slurm)"
echo "context_gating_frequency_factorial=${JOB_ID}"
echo "logs=slurm_logs/mcf_frequency_factorial_${JOB_ID}_<index>.out"
