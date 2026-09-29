#!/bin/bash
# Multi-fact person benchmark, seed 1, layer 19: several facts about one real
# person in ONE sentence; forget one, retain the others. SURE with the linear
# classifier (L-BFGS, calibrated cutoff folded into the bias), regular mode.
#   sbatch multifact_person_seed1.slurm
#   sbatch --export=ALL,BUILD_ARGS="--min-facts 3" multifact_person_seed1.slurm   # 3+ facts per sentence only
# Downloads data/MQuAKE-CF.json if missing (do it on the login node if compute nodes are offline:
#   curl -L -o data/MQuAKE-CF.json https://raw.githubusercontent.com/princeton-nlp/MQuAKE/main/datasets/MQuAKE-CF.json)
# Resumable. Report: outputs/multifact_person_v1/seed1/L19/linear_global/official_multifact_eval.md
#SBATCH --job-name=multifact
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/multifact_%j.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/multifact_%j.err

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
export SEED="${SEED:-1}" LAYER="${LAYER:-19}" TAG="${TAG:-multifact_person_v1}" BUILD_ARGS="${BUILD_ARGS:-}"
echo "multi-fact person | seed $SEED L$LAYER | build args '${BUILD_ARGS}' | job ${SLURM_JOB_ID:-?} | $(hostname) | $(date) | commit $(git rev-parse HEAD)"
nvidia-smi

bash scripts/run_multifact_person_seed1.sh
STATUS=$?
echo "Finish: $(date) | exit $STATUS"
exit $STATUS
