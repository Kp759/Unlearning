#!/bin/bash
# Collateral damage to the same subject's OTHER facts, MCF seed 1. For each forget fact,
# prompts about the same subject with other relations (MCF templates of the same subject
# type; the "eval" half of the relations, never used by --preserve-weight training).
#   forced : the fact's edit is switched on (as inside a reply where the router fired)
#   natural: the router decides (should not fire) -> fire rate
# Runs: J_C (layer 19) and every early-write arm outputs/mcf_early_write_seed1/E_L*.
#
#   sbatch mcf_subject_collateral_seed1.slurm
#   cat outputs/mcf_subject_collateral_seed1/summary.md
#SBATCH --job-name=mcf_collateral
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_collateral_%j.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_collateral_%j.err

set -euo pipefail

# Environment (same block as the other sweep jobs).
ENV_DIR="${ENV_DIR:-/scratch/yl258/kp759/conda_envs/semantic_unlearning}"
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
RUNS=()
[[ -f "$ROOT/outputs/mcf_jlens_feedback_seed1/J_C/linear_global/fact_association_embeddings.pt" ]] \
  && RUNS+=("J_C=$ROOT/outputs/mcf_jlens_feedback_seed1/J_C/linear_global")
for ARM in "$ROOT"/outputs/mcf_early_write_seed1/E_L*; do
  [[ -f "$ARM/linear_global/fact_association_embeddings.pt" ]] && RUNS+=("$(basename "$ARM")=$ARM/linear_global")
done
echo "collateral eval | runs: ${RUNS[*]} | job ${SLURM_JOB_ID:-} | $(hostname) | $(date)"
python -u scripts/mcf_subject_collateral.py --runs "${RUNS[@]}" \
  --out-dir "$ROOT/outputs/mcf_subject_collateral_seed1" --device cuda --local-files-only
echo "Finish: $(date)"
