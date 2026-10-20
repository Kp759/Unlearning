#!/bin/bash
# Where are the MCF seed-1 forget facts recalled? Base model, subject tokens: logit-lens
# readability at the last subject token per layer, and P(answer) after zeroing the subject's
# MLP outputs over 4-layer windows (ROME-style). Picks the early-write layer. ~15 min.
#
#   sbatch mcf_recall_site_seed1.slurm
#   cat outputs/mcf_recall_site_seed1/summary.md
#SBATCH --job-name=mcf_recall_site
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_recall_site_%j.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_recall_site_%j.err

set -euo pipefail

# Environment (same block as the other sweep jobs).
if [[ -z "${ENV_DIR:-}" ]]; then ENV_DIR=/project/yl258/kp759/conda_envs/semantic_unlearning; [[ -f "$ENV_DIR/.ready" ]] || ENV_DIR=/scratch/yl258/kp759/conda_envs/semantic_unlearning; fi  # /project env once built (scripts/build_env_project.sh)
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
python -c "import torch; assert torch.cuda.is_available()" \
  || { echo "No CUDA device visible; aborting" >&2; exit 3; }
ROOT=/scratch/yl258/kp759/Unlearning/semantic-unlearning
cd "$ROOT"
mkdir -p logs
python -u scripts/mcf_recall_site.py \
  --router-dir "$ROOT/outputs/mcf_multiseed_regular_v1/seed1/L19/router" \
  --out-dir "$ROOT/outputs/mcf_recall_site_seed1" --device cuda --local-files-only
