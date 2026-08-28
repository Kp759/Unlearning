#!/usr/bin/env bash
set -euo pipefail

export PROJECT_DIR="${PROJECT_DIR:-$(pwd -P)}"
mkdir -p slurm_logs

ARRAY_JOB="$(sbatch --parsable slurm/run_mcf_scoped_bias_confirmatory_3b.slurm)"
AGGREGATE_JOB="$(
  sbatch --parsable \
    --dependency="afterok:${ARRAY_JOB}" \
    slurm/aggregate_mcf_scoped_bias_confirmatory.slurm
)"

echo "confirmatory_array_job=${ARRAY_JOB}"
echo "aggregate_job=${AGGREGATE_JOB}"
echo "aggregate_dependency=afterok:${ARRAY_JOB}"
