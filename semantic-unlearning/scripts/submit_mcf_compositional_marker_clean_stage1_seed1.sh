#!/usr/bin/env bash
set -euo pipefail

export PROJECT_DIR="${PROJECT_DIR:-$(pwd -P)}"
mkdir -p slurm_logs

JOB_ID="$(sbatch --parsable slurm/run_mcf_compositional_marker_clean_stage1_seed1_3b.slurm)"
echo "clean_stage1_writer_job=${JOB_ID}"
echo "log=slurm_logs/mcf_clean_writer_seed1_${JOB_ID}.out"
