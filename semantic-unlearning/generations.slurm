#!/bin/bash
# What does the model generate after unlearning? Base vs unlearned, side by side.
#   sbatch --export=ALL,RUN_DIRS="outputs/mcf_multiseed_regular_v1/seed1/L19/linear_global outputs/compressed_multiseed_v1/mcf/seed1/L19/tied_answer",LABELS="shipped tied_answer" generations.slurm
# Optional: OUT=outputs/generations/mcf_seed1 EXTRA="--retain 100 --max-new-tokens 32"
#SBATCH --job-name=generations
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/generations_%j.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/generations_%j.err

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
: "${RUN_DIRS:?set RUN_DIRS to one or more run dirs of the same dataset and seed}"
ARGS=(--local-files-only --run-dirs $RUN_DIRS)
[[ -n "${LABELS:-}" ]] && ARGS+=(--labels $LABELS)
[[ -n "${OUT:-}" ]] && ARGS+=(--out "$OUT")
# shellcheck disable=SC2086
python -u scripts/generate_after_unlearning.py "${ARGS[@]}" ${EXTRA:-}
echo "Finish: $(date) | status $?"
