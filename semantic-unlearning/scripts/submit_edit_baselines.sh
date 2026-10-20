#!/usr/bin/env bash
# ROME + MEMIT baselines, 4 datasets x 5 seeds (l16_edit_baselines.slurm, 40 tasks).
# MEMIT on MCF seed 1 (task 1) runs first: it needs the down_proj covariance of layers 16-18
# (ROME: 18), which it computes once into data/stats if the ZeroUnlearn runs' cache is gone;
# every other task waits for it and reuses the cache.
#
#   bash scripts/submit_edit_baselines.sh
#   python scripts/l16_paper_table.py --ours-from mcf=outputs/l16_paper_mcf_linear19   # when done
set -euo pipefail
cd "$(dirname "$0")/.."
A=$(sbatch --parsable --array=1 l16_edit_baselines.slurm)
echo "MEMIT, MCF seed 1 (covariance cache): $A"
B=$(sbatch --parsable --dependency=afterany:"$A" --array=0,2-39 l16_edit_baselines.slurm)
echo "ROME + MEMIT, remaining 39 tasks (after $A): $B"
echo "logs: logs/l16_edit_<job>_<task>.{out,err}"
