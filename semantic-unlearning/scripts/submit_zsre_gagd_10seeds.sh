#!/bin/bash
set -euo pipefail
A=$(sbatch --parsable scripts/slurm_zsre_sure_gagd_aggressive_10seeds.sh)
B=$(sbatch --parsable --dependency=afterok:$A scripts/slurm_aggregate_zsre_sure_gagd_10seeds.sh)
echo "array=$A aggregate=$B"
