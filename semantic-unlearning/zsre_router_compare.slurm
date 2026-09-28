#!/bin/bash
# ZsRE: regular vs reworded router on the SAME same-subject negatives (the union
# of both routers' held-out controls), full cutoff curves, and the paraphrase
# recall each reaches at matched same-subject false fire. Evaluation only.
#   sbatch zsre_router_compare.slurm                                   # regular vs reworded_v1
#   sbatch --export=ALL,TAGS="regular_v1 reworded_v1 reworded_v2" zsre_router_compare.slurm
#SBATCH --job-name=zsr_cmp
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/zsre_cmp_%j.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/zsre_cmp_%j.err

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
TAGS="${TAGS:-regular_v1 reworded_v1}"
NEG=(); SWEEPS=()
for T in $TAGS; do SWEEPS+=("outputs/zsre_multiseed_$T"); NEG+=("$T=outputs/zsre_multiseed_$T"); done
echo "ZsRE router compare | job $SLURM_JOB_ID | $(hostname) | $(date) | commit $(git rev-parse HEAD)"
STATUS=0
for SWEEP in "${SWEEPS[@]}"; do
  echo "================ $SWEEP ================"
  python -u scripts/evaluate_zsre_router_decomposition.py \
    --run-dirs "$SWEEP/seed*/L19/linear_global" "$SWEEP/seed*/L23/linear_global" \
    --zsre-path data/zsre_mend_eval.json --device cuda --dtype bfloat16 --batch-size 16 \
    --local-files-only --overwrite --negative-sets "${NEG[@]}" || STATUS=1
done
echo "Finish: $(date) | exit $STATUS"
exit $STATUS
