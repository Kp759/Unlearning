#!/bin/bash
# Baselines on another model, same harness as paper_model_multiseed.slurm and the same task
# layout as l16_baselines.slurm: task = 25 * dataset + 5 * (seed - 1) + baseline, baselines
# ga gd npo rmu zerounlearn (0-24 mcf, 25-49 zsre, 50-74 mquake, 75-99 rwku).
# Needs that model's routers and base runs (BUILD_ROUTERS=0). Per-model baseline flags
# (RMU layer, GA/GD/NPO scope, ZeroUnlearn hparams) are BASELINE_ARGS in paper_model_config.sh.
# ZeroUnlearn computes the model's Wikipedia covariance once (data/stats/<snapshot>/);
# submit_paper_model.sh runs task 4 first so the other ZeroUnlearn tasks reuse it.
#
#   sbatch --export=ALL,MODEL=llama31_8b --array=0-4 paper_model_baselines.slurm   # MCF seed 1
#SBATCH --job-name=paper_model_base
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-99
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/paper_model_base_%A_%a.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/paper_model_base_%A_%a.err

set -uo pipefail

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import torch; assert torch.cuda.is_available()" \
  || { echo "No CUDA device visible; aborting" >&2; exit 3; }
command -v jq >/dev/null || { echo "jq not found on PATH; aborting" >&2; exit 3; }

cd /scratch/yl258/kp759/Unlearning/semantic-unlearning
mkdir -p logs
source scripts/paper_model_config.sh || exit 2
paper_model_check || exit $?
DATASETS=(mcf zsre mquake rwku)
METHODS=(ga gd npo rmu zerounlearn)
TASK="${SLURM_ARRAY_TASK_ID:?run as an array job}"
DS="${DATASETS[$(( TASK / 25 ))]:-}"
SEED=$(( (TASK % 25) / 5 + 1 ))
B="${METHODS[$(( TASK % 5 ))]}"
[[ -n "$DS" ]] || { echo "array index $TASK out of range (0-99)" >&2; exit 2; }

echo "paper baseline $MODEL | $DS seed $SEED | $B | job ${SLURM_ARRAY_JOB_ID:-}_${TASK} | $(hostname) | $(date) | commit $(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
BASELINES="$B" SKIP_OURS=1 BUILD_ROUTERS=0 bash scripts/run_l16_paper_one.sh "$DS" "$SEED"
STATUS=$?
echo "Finish $MODEL $DS seed $SEED $B: $(date) | exit $STATUS"
exit $STATUS
