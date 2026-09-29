#!/bin/bash
# Eval-DU+ (ICLR 2026, "Learning-Time Encoding Shapes Unlearning in LLMs") with SURE.
# Fine-tunes the base model on FT-Mul-Chunk (biographies stating several facts of one
# person together), then SURE (linear classifier, L-BFGS, calibrated cutoff folded
# into the bias), seed 1, layer 19, the paper's 100-fact unlearn split.
#   sbatch evaldu_plus_seed1.slurm
#   sbatch --export=ALL,SPLIT=people12 evaldu_plus_seed1.slurm        # 12-people split (reuses the fine-tune)
#   sbatch --export=ALL,FT_ARGS="--epochs 8" evaldu_plus_seed1.slurm  # if the facts are not learned
# Clone the data on the login node first if compute nodes are offline:
#   git clone --depth 1 https://github.com/wrh14/learning_time_shapes_unlearning data/evaldu_plus_upstream
# Fine-tuning uses fp32 master weights (needs ~55 GB GPU memory) and falls back to bf16 on OOM.
# Resumable. Report: outputs/evaldu_plus_v1/seed1/L19/linear_global/official_evaldu_eval.md
#SBATCH --job-name=evaldu
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/evaldu_%j.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/evaldu_%j.err

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
export SEED="${SEED:-1}" LAYER="${LAYER:-19}" TAG="${TAG:-evaldu_plus_v1}" SPLIT="${SPLIT:-facts100}"
export UNLEARN_DATA="${UNLEARN_DATA:-mul}" FT_ARGS="${FT_ARGS:-}"
echo "Eval-DU+ SURE | split $SPLIT seed $SEED L$LAYER | ft args '${FT_ARGS}' | job ${SLURM_JOB_ID:-?} | $(hostname) | $(date) | commit $(git rev-parse HEAD)"
nvidia-smi

bash scripts/run_evaldu_plus_seed1.sh
STATUS=$?
echo "Finish: $(date) | exit $STATUS"
exit $STATUS
