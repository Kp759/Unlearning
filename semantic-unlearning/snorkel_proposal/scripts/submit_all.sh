#!/usr/bin/env bash
# Submit SCALPEL for every model (5 seeds each), then aggregate when they finish:
#   bash scripts/submit_all.sh                    # dev partition
#   PARTITION=test SCALPEL_TEST_SALT=... bash scripts/submit_all.sh
#   python -m scalpel report --runs 'results/cluster/*/dev/seed*/runs/*/results.json' --out results/cluster/leaderboard
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
for MODEL in ${MODELS:-llama32_3b qwen3_4b llama31_8b}; do
  sbatch --export=ALL,MODEL="$MODEL",PARTITION="${PARTITION:-dev}" scripts/slurm/scalpel_one.slurm
done
