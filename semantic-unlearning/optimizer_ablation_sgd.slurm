#!/bin/bash
# Router optimizer ablation: minibatch SGD vs the shipped full-batch L-BFGS.
# Seed 1, layer 19, regular SURE; one array task per benchmark (0 mcf, 1 zsre, 2 mquake).
#   sbatch optimizer_ablation_sgd.slurm                                # all three
#   sbatch --array=1 optimizer_ablation_sgd.slurm                      # ZsRE only
#   sbatch --export=ALL,NOISE_FLOOR=0 optimizer_ablation_sgd.slurm     # skip the L-BFGS rerun rows (~half the time)
#   sbatch --export=ALL,REF_TAG=multiseed_reworded_v2 --array=1 optimizer_ablation_sgd.slurm
#   sbatch --export=ALL,SGD_ARGS="--sgd-momentum 0 --sgd-epochs 600",ABL_TAG=optimizer_ablation_vanilla optimizer_ablation_sgd.slurm
# Needs the reference run: outputs/<dataset>_<REF_TAG>/seed1/L19/{prep,router,linear_global}.
# Resumable: finished stages are skipped. Report: outputs/<ABL_TAG>/<dataset>/seed1/L19/comparison.md
#SBATCH --job-name=opt_ablate
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --array=0-2
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/opt_ablate_%A_%a.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/opt_ablate_%A_%a.err

set -uo pipefail

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
  || { echo "Python env is not usable on $(hostname); aborting" >&2; exit 3; }
python -c "import torch; assert torch.cuda.is_available()" \
  || { echo "No CUDA device visible; aborting" >&2; exit 3; }
command -v jq >/dev/null || { echo "jq not found on PATH; aborting" >&2; exit 3; }

cd /scratch/yl258/kp759/Unlearning/semantic-unlearning
mkdir -p logs

DATASETS=(mcf zsre mquake)
DATASET="${DATASETS[${SLURM_ARRAY_TASK_ID:-0}]}"
export SEED="${SEED:-1}" LAYER="${LAYER:-19}"
export REF_TAG="${REF_TAG:-multiseed_regular_v1}" ABL_TAG="${ABL_TAG:-optimizer_ablation_v1}"
export NOISE_FLOOR="${NOISE_FLOOR:-1}" SGD_ARGS="${SGD_ARGS:-}"

echo "optimizer ablation | $DATASET seed $SEED L$LAYER | ref $REF_TAG | noise floor $NOISE_FLOOR | sgd args '${SGD_ARGS}' | job ${SLURM_ARRAY_JOB_ID:-?}_${SLURM_ARRAY_TASK_ID:-?} | $(hostname) | $(date) | commit $(git rev-parse HEAD)"
nvidia-smi

bash scripts/run_optimizer_ablation_one.sh "$DATASET"
STATUS=$?
python scripts/compare_router_optimizers.py --collect "outputs/$ABL_TAG" || true
echo "Finish $DATASET: $(date) | exit $STATUS"
exit $STATUS
