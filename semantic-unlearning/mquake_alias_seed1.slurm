#!/bin/bash
# MQuAKE answer aliases, seed 1, L19 (scripts/run_mquake_alias_seed1.sh):
#   1. alias-leak eval of the existing regular run: is a forgotten answer still
#      produced through a Wikidata alias ("U.S.", "America", ...)?
#   2. rows retrained with alias targets behind the SAME router; official MQuAKE
#      eval + alias-leak eval + comparison.
#   sbatch mquake_alias_seed1.slurm
#   sbatch --export=ALL,ALIAS_TRAIN=0 mquake_alias_seed1.slurm   # step 1 only (~15 min)
# Reports: outputs/mquake_multiseed_regular_v1/seed1/L19/linear_global/alias_leak_eval.md
#          outputs/mquake_alias_targets_v1/seed1/L19/comparison.md
#SBATCH --job-name=mq_alias
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mq_alias_%j.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mq_alias_%j.err

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
export SEED="${SEED:-1}" ALIAS_TRAIN="${ALIAS_TRAIN:-1}"
echo "MQuAKE alias | seed $SEED | alias train $ALIAS_TRAIN | job ${SLURM_JOB_ID:-?} | $(hostname) | $(date) | commit $(git rev-parse HEAD)"
bash scripts/run_mquake_alias_seed1.sh
STATUS=$?
echo "Finish: $(date) | exit $STATUS"
exit $STATUS
