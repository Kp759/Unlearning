#!/bin/bash
# Round 2 of the paper table (run from semantic-unlearning/ on the login node):
#   1. ZsRE (tasks 5-9): ours/genie retrained under the reworded L19 router (fixes the router's
#      paraphrase misses) with --oracle-unrouted; RWKU (15-19): retrained with --oracle-unrouted
#      (facts the router never routes no longer keep a zero row). Base runs are reused; the
#      driver moves the old ours/genie aside (*.superseded_<time>). MCF/MQuAKE had no zero rows
#      and keep their router, so they are not rerun.
#   2. Baselines (100 tasks), after step 1 so ZsRE baselines train on the same views as ours.
#
#   bash scripts/submit_l16_round2.sh
#   ONLY=baselines bash scripts/submit_l16_round2.sh      # step 2 only
set -euo pipefail
cd "$(dirname "$0")/.."
DEP=()
if [[ "${ONLY:-}" != baselines ]]; then
  M=$(sbatch --parsable --array=5-9,15-19 l16_paper_multiseed.slurm)
  echo "ours/genie rerun (ZsRE reworded router + oracle rows, RWKU oracle rows): $M"
  DEP=(--dependency=afterany:"$M")
fi
B=$(sbatch --parsable "${DEP[@]}" l16_baselines.slurm)
echo "baselines (ga gd npo rmu zerounlearn x 4 datasets x 5 seeds): $B"
echo "when done: python scripts/l16_paper_table.py"
