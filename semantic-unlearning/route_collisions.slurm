#!/bin/bash
# How often do >= 2 router heads qualify on an official prompt, and what happens then?
#   sbatch route_collisions.slurm
#   sbatch --export=ALL,RUN_GLOBS="outputs/mcf_multiseed_regular_v1/seed1/L19/linear_global" route_collisions.slurm
# Writes <run>/route_collisions.json per run and prints one table. Resumable.
#SBATCH --job-name=route_collisions
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/route_collisions_%j.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/route_collisions_%j.err

set -uo pipefail
# Environment. On a compute node `module load Miniforge3` can fail after a
# purge (Lmod needs its parent module), so try the usual routes and fall back
# to putting the env's own bin/ first on PATH. Then stop at once if the env is
# not usable, instead of failing every layer in seconds.
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
echo "python: $(command -v python)"
python -c "import torch, transformers, datasets; print('torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())" \
  || { echo "Python env is not usable on $(hostname); aborting before any layer runs" >&2; exit 3; }
python -c "import torch; assert torch.cuda.is_available()" \
  || { echo "No CUDA device visible; aborting" >&2; exit 3; }
command -v jq >/dev/null || { echo "jq not found on PATH; aborting" >&2; exit 3; }

cd /scratch/yl258/kp759/Unlearning/semantic-unlearning
mkdir -p logs
echo "route collisions | job ${SLURM_JOB_ID} | node $(hostname) | $(date) | commit $(git rev-parse HEAD)"

RUN_GLOBS="${RUN_GLOBS:-outputs/mcf_multiseed_regular_v1/seed*/L??/linear_global outputs/zsre_multiseed_reworded_v2/seed*/L??/linear_global outputs/mquake_multiseed_regular_v1/seed*/L??/linear_global}"
set -f  # pass the patterns through; the script expands them
# shellcheck disable=SC2086
python -u scripts/count_route_collisions.py --local-files-only --run-dirs $RUN_GLOBS
set +f
echo "Finish: $(date)"
