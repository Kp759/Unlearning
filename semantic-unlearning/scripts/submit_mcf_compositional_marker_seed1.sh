#!/usr/bin/env bash
set -euo pipefail

export PROJECT_DIR="${PROJECT_DIR:-$(pwd -P)}"
mkdir -p slurm_logs

JOB_ID="$(sbatch --parsable slurm/run_mcf_compositional_marker_seed1_3b.slurm)"
echo "compositional_marker_job=${JOB_ID}"
echo "log=slurm_logs/mcf_compositional_marker_seed1_${JOB_ID}.out"
