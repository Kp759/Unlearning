#!/bin/bash
# Collect: red-team gate per seed, leaderboard per model, and one leaderboard across models.
#   MODELS="llama32_3b qwen3_4b" sbatch --export=ALL scripts/slurm/scalpel_report.slurm
#SBATCH --job-name=sc_report
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/scalpel_report_%j.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/scalpel_report_%j.err
set -uo pipefail
MODELS="${MODELS:-llama32_3b}"
export MODEL="${MODELS%% *}"
source /scratch/yl258/kp759/Unlearning/semantic-unlearning/snorkel_proposal/scripts/slurm/scalpel_common.sh
cd "$SP"
for M in $MODELS; do
  for D in "$OUT_ROOT/$M/$PARTITION"/seed*; do
    [[ -f "$D/split.json" ]] && python -m scalpel gate --split "$D/split.json" --runs "$D/runs" > /dev/null \
      && grep -m1 "Overall" "$D/runs/RED_TEAM_REPORT.md" | sed "s|^|$M $(basename "$D"): |"
    compgen -G "$D/agent/*/agent.json" > /dev/null && python -m scalpel agent-report --runs "$D/agent/*/agent.json" \
      --out "$D/agent" --title "LETHE-Bench agentic leakage track — $M — $(basename "$D")"
  done
  python -m scalpel report --runs "$OUT_ROOT/$M/$PARTITION/seed*/runs/*/results.json" \
    --out "$OUT_ROOT/$M/$PARTITION/leaderboard" --exclude sure --title "LETHE-Bench — $M — $PARTITION (mean ± sd over seeds)"
done
python -m scalpel report --runs "$OUT_ROOT/*/$PARTITION/seed*/runs/*/results.json" \
  --out "$OUT_ROOT/leaderboard_$PARTITION" --exclude sure --title "LETHE-Bench — all models — $PARTITION"
echo "leaderboards: $OUT_ROOT/*/$PARTITION/leaderboard/leaderboard.md and $OUT_ROOT/leaderboard_$PARTITION/leaderboard.md"
