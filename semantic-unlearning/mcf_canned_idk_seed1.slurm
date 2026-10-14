#!/bin/bash
# C_idk, MCF seed 1: NO vectors. The seed-1 router alone decides; if it fires, the reply
# is the fixed " I don't know.", otherwise the base model answers. CPU only: routes are
# row-independent, so they come from the held-out Δe run, and unrouted prompts are base.
# Needs outputs/mcf_heldout_generation_seed1/{base,delta_e}.json (the held-out job), so
# submit it after that job:
#
#   sbatch --dependency=afterok:1407291 mcf_canned_idk_seed1.slurm
#   cat outputs/mcf_refusal_rows_seed1/comparison.md outputs/mcf_heldout_generation_seed1/summary.md
#SBATCH --job-name=mcf_canned_idk
#SBATCH --account=yl258
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_canned_idk_%j.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_canned_idk_%j.err

set -euo pipefail

# Environment (same block as the other sweep jobs).
ENV_DIR=/scratch/yl258/kp759/conda_envs/semantic_unlearning
module purge 2>/dev/null || true
module load wulver 2>/dev/null || true
module load Miniforge3 2>/dev/null || true
if command -v conda >/dev/null 2>&1; then
  set +u  # conda's activate scripts reference unset variables
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_DIR"
  set -u
else
  export PATH="$ENV_DIR/bin:$PATH"
  export CONDA_PREFIX="$ENV_DIR"
fi
export TOKENIZERS_PARALLELISM=false

ROOT=/scratch/yl258/kp759/Unlearning/semantic-unlearning
cd "$ROOT"
mkdir -p logs
HELD="$ROOT/outputs/mcf_heldout_generation_seed1"
for F in base delta_e; do
  test -f "$HELD/$F.json" || { echo "missing $HELD/$F.json: run mcf_heldout_generation_seed1.slurm first" >&2; exit 2; }
done
echo "C_idk (router + fixed reply) | job ${SLURM_JOB_ID:-} | $(hostname) | $(date) | commit $(git rev-parse HEAD)"
python -u scripts/mcf_canned_idk.py
bash scripts/run_refusal_summaries.sh
echo "Finish: $(date)"
