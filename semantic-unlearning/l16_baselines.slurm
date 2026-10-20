#!/bin/bash
# Baselines for the paper table, same harness as base | ours | genie (l16_paper_multiseed.slurm):
# GA, GD, NPO(+retain), RMU, ZeroUnlearn on MCF, ZsRE, MQuAKE, RWKU, seeds 1-5.
# task = 25 * dataset + 5 * (seed - 1) + baseline, datasets mcf zsre mquake rwku,
# baselines ga gd npo rmu zerounlearn  (0-24 mcf, 25-49 zsre, 50-74 mquake, 75-99 rwku).
# Needs the main job's routers and base runs (it never fits a router: BUILD_ROUTERS=0).
#
#   sbatch l16_baselines.slurm                          # all 100
#   sbatch --array=0-4 l16_baselines.slurm              # MCF seed 1, all five baselines (smoke test)
#   then: python scripts/l16_paper_table.py
#SBATCH --job-name=l16_base
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-99
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/l16_base_%A_%a.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/l16_base_%A_%a.err

set -uo pipefail

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import torch; assert torch.cuda.is_available()" \
  || { echo "No CUDA device visible; aborting" >&2; exit 3; }
command -v jq >/dev/null || { echo "jq not found on PATH; aborting" >&2; exit 3; }

cd /scratch/yl258/kp759/Unlearning/semantic-unlearning
mkdir -p logs
DATASETS=(mcf zsre mquake rwku)
METHODS=(ga gd npo rmu zerounlearn)
TASK="${SLURM_ARRAY_TASK_ID:?run as an array job}"
DS="${DATASETS[$(( TASK / 25 ))]:-}"
SEED=$(( (TASK % 25) / 5 + 1 ))
B="${METHODS[$(( TASK % 5 ))]}"
[[ -n "$DS" ]] || { echo "array index $TASK out of range (0-99)" >&2; exit 2; }

echo "l16 baseline | $DS seed $SEED | $B | job ${SLURM_ARRAY_JOB_ID:-}_${TASK} | $(hostname) | $(date) | commit $(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
BASELINES="$B" SKIP_OURS=1 BUILD_ROUTERS=0 bash scripts/run_l16_paper_one.sh "$DS" "$SEED"
STATUS=$?
echo "Finish $DS seed $SEED $B: $(date) | exit $STATUS"
exit $STATUS
