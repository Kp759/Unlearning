#!/usr/bin/env bash
#SBATCH --job-name=zsre_gagd_agg
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_logs/zsre_gagd_aggregate_%j.out
#SBATCH --error=slurm_logs/zsre_gagd_aggregate_%j.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/scratch/yl258/kp759/Unlearning/semantic-unlearning}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/zsre_sure_no_neutral_gagd_aggressive_seeds1_10_3b}"
cd "${REPO_ROOT}"

if [[ -d /scratch/yl258/kp759/conda_envs/semantic_unlearning/bin ]]; then
  export PATH=/scratch/yl258/kp759/conda_envs/semantic_unlearning/bin:${PATH}
fi

python scripts/aggregate_zsre_sure_gagd_10seeds.py \
  --root "${OUTPUT_ROOT}" \
  --seeds 1-10 \
  --require-all \
  --output-prefix aggregate_10seeds

echo "===== aggregate markdown ====="
cat "${OUTPUT_ROOT}/aggregate_10seeds.md"
