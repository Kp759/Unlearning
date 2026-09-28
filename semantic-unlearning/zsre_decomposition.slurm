#!/bin/bash
# ZsRE: router vs genie on the same trained rows + why each paraphrase misses +
# threshold what-if. Evaluation only (no training). One model load for all runs.
#   sbatch zsre_decomposition.slurm
#   sbatch --export=ALL,RUN_GLOB='outputs/zsre_multiseed_regular_v1/seed1/L19/linear_global' zsre_decomposition.slurm
# Writes <run>/zsre_decomposition.json (skips runs that already have one) and
# prints the summary tables at the end of the log.
#SBATCH --job-name=zsr_decomp
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/zsre_decomp_%j.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/zsre_decomp_%j.err

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
RUN_GLOB="${RUN_GLOB:-outputs/zsre_multiseed_regular_v1/seed*/L??/linear_global}"
echo "ZsRE decomposition | job $SLURM_JOB_ID | $(hostname) | $(date) | commit $(git rev-parse HEAD) | $RUN_GLOB"
python -u scripts/evaluate_zsre_router_decomposition.py \
  --run-dirs "$RUN_GLOB" --zsre-path data/zsre_mend_eval.json \
  --device cuda --dtype bfloat16 --batch-size 16 --local-files-only
STATUS=$?
echo "Finish: $(date) | exit $STATUS"
exit $STATUS
