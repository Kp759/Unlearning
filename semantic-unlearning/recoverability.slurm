#!/bin/bash
# Representation recoverability (stage 6) on the finished run dirs of a paper-table TAG:
# base | baselines | ours | genie, MCF ZsRE MQuAKE RWKU, seeds 1-5. One array task per
# (dataset, seed) as in l16_paper_multiseed.slurm: task = 5 * dataset_index + (seed - 1).
# Nothing is retrained: dirs whose official eval is missing are skipped, dirs that already
# have representation_recoverability.json are skipped, so a resubmit resumes.
#
#   sbatch recoverability.slurm                                   # TAG onepass_l19_v1 (ours), all 20
#   TAG=l16_ours_v1 sbatch recoverability.slurm                   # the L16 ablation
#   TAG=l16_paper_v1 BASELINES="ga gd npo rmu zerounlearn rome memit" sbatch recoverability.slurm
#   sbatch --array=0,5,10,15 recoverability.slurm                 # seed 1 of every dataset (smoke test)
#   then: python scripts/l16_paper_table.py --root outputs/onepass_l19_v1 --ours-label "Ours (L19, one pass)"
#         -> the RR column; per-run tables in <run>/representation_recoverability.md
#SBATCH --job-name=recoverability
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-19
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/recoverability_%A_%a.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/recoverability_%A_%a.err

set -uo pipefail

if [[ -z "${ENV_DIR:-}" ]]; then ENV_DIR=/project/yl258/kp759/conda_envs/semantic_unlearning; [[ -f "$ENV_DIR/.ready" ]] || ENV_DIR=/scratch/yl258/kp759/conda_envs/semantic_unlearning; fi
module purge 2>/dev/null || true
module load wulver 2>/dev/null || true
module load Miniforge3 2>/dev/null || true
if command -v conda >/dev/null 2>&1; then
  set +u
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_DIR"
  set -u
else
  export PATH="$ENV_DIR/bin:$PATH"
  export CONDA_PREFIX="$ENV_DIR"
fi
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
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
TAG="${TAG:-onepass_l19_v1}"

echo "recoverability | $TAG $DS seed $SEED | baselines '${BASELINES:-}' | job ${SLURM_ARRAY_JOB_ID:-}_${TASK} | $(hostname) | $(date) | commit $(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
bash scripts/run_recoverability_one.sh "$TAG" "$DS" "$SEED"
STATUS=$?
echo "Finish $TAG $DS seed $SEED: $(date) | exit $STATUS"
exit $STATUS
