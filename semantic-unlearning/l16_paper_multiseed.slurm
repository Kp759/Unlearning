#!/bin/bash
# Paper table: base | ours (L16 question-side Δe) | genie, on MCF, ZsRE, MQuAKE, RWKU,
# seeds 1-5. One array task per (dataset, seed): task = 5 * dataset_index + (seed - 1),
# datasets in the order mcf zsre mquake rwku (0-4 mcf, 5-9 zsre, 10-14 mquake, 15-19 rwku).
# Each task runs scripts/run_l16_paper_one.sh (router -> [MCF v2.2] -> base -> ours -> genie);
# finished stages are skipped, so a resubmit resumes.
#
#   sbatch l16_paper_multiseed.slurm                       # all 20
#   sbatch --array=0-4 l16_paper_multiseed.slurm           # MCF only
#   sbatch --array=5,10,15 l16_paper_multiseed.slurm       # seed 1 of ZsRE, MQuAKE, RWKU (smoke test)
#   then (login node, CPU):
#   python scripts/l16_paper_table.py                      # -> outputs/l16_paper_v1/paper_table.{md,tex,csv,json}
#SBATCH --job-name=l16_paper
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --array=0-19
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/l16_paper_%A_%a.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/l16_paper_%A_%a.err

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
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import torch; assert torch.cuda.is_available()" \
  || { echo "No CUDA device visible; aborting" >&2; exit 3; }
command -v jq >/dev/null || { echo "jq not found on PATH; aborting" >&2; exit 3; }

cd /scratch/yl258/kp759/Unlearning/semantic-unlearning
mkdir -p logs
DATASETS=(mcf zsre mquake rwku)
TASK="${SLURM_ARRAY_TASK_ID:?run as an array job}"
DS="${DATASETS[$(( TASK / 5 ))]:-}"
SEED=$(( TASK % 5 + 1 ))
[[ -n "$DS" ]] || { echo "array index $TASK out of range (0-19)" >&2; exit 2; }

echo "l16 paper | $DS seed $SEED | job ${SLURM_ARRAY_JOB_ID:-}_${TASK} | $(hostname) | $(date) | commit $(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
bash scripts/run_l16_paper_one.sh "$DS" "$SEED"
STATUS=$?
echo "Finish $DS seed $SEED: $(date) | exit $STATUS"
exit $STATUS
