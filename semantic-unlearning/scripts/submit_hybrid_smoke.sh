#!/usr/bin/env bash
# Hybrid router smoke test: seed 1 of MCF, ZsRE, MQuAKE and RWKU (l16_paper_multiseed.slurm
# tasks 0, 5, 10, 15), under both subject gates:
#   learned           learned subject head replaces the exact subject match
#   learned_or_exact  learned head OR exact match
# The first array fits the subject head (outputs/<ds>_hybrid_subject_v1/seed1, both gates at
# once); the second waits for it and reuses the fit. Base runs are copied from l16_paper_v1.
#
#   bash scripts/submit_hybrid_smoke.sh              # ARRAY=0-19 for all seeds
#   python scripts/compare_router_runs.py            # when both arrays are done
set -euo pipefail
cd "$(dirname "$0")/.."
ARRAY="${ARRAY:-0,5,10,15}"
A=$(sbatch --parsable --job-name=hy_learned \
      --export=ALL,TAG=l16_paper_hybrid_learned_v1,HYBRID_GATE=learned,BASE_FROM=l16_paper_v1 \
      --array="$ARRAY" l16_paper_multiseed.slurm)
echo "hybrid (learned gate), tasks $ARRAY: $A"
B=$(sbatch --parsable --job-name=hy_or_exact --dependency=afterany:"$A" \
      --export=ALL,TAG=l16_paper_hybrid_or_exact_v1,HYBRID_GATE=learned_or_exact,BASE_FROM=l16_paper_v1 \
      --array="$ARRAY" l16_paper_multiseed.slurm)
echo "hybrid (learned-or-exact gate), after $A: $B"
echo "logs: logs/l16_paper_${A}_<task>.{out,err}, logs/l16_paper_${B}_<task>.{out,err}"
echo "when both finish: python scripts/compare_router_runs.py"
