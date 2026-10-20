#!/bin/bash
# Locate-and-edit baselines ROME and MEMIT (vendored in ZeroUnlearn/, hparams from
# ZeroUnlearn/hparams/{ROME,MEMIT}) in the paper-table harness: same requests as the
# ZeroUnlearn baseline (target_true -> EOS, no retain requests), same evaluators.
# task = 10 * dataset + 2 * (seed - 1) + baseline, datasets mcf zsre mquake rwku,
# baselines rome memit  (0-9 mcf, 10-19 zsre, 20-29 mquake, 30-39 rwku).
# Needs the main job's routers and base runs (BUILD_ROUTERS=0). Covariance statistics are
# shared with ZeroUnlearn (data/stats); scripts/submit_edit_baselines.sh runs MEMIT on MCF
# seed 1 first so a missing cache is computed once.
#
#   bash scripts/submit_edit_baselines.sh                # all 40
#   sbatch --array=0,1 l16_edit_baselines.slurm          # MCF seed 1, ROME + MEMIT (smoke test)
#   then: python scripts/l16_paper_table.py --ours-from mcf=outputs/l16_paper_mcf_linear19
#SBATCH --job-name=l16_edit
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-39
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/l16_edit_%A_%a.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/l16_edit_%A_%a.err

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
DATASETS=(mcf zsre mquake rwku)
METHODS=(rome memit)
TASK="${SLURM_ARRAY_TASK_ID:?run as an array job}"
DS="${DATASETS[$(( TASK / 10 ))]:-}"
SEED=$(( (TASK % 10) / 2 + 1 ))
B="${METHODS[$(( TASK % 2 ))]}"
[[ -n "$DS" ]] || { echo "array index $TASK out of range (0-39)" >&2; exit 2; }

echo "l16 edit baseline | $DS seed $SEED | $B | job ${SLURM_ARRAY_JOB_ID:-}_${TASK} | $(hostname) | $(date) | commit $(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
BASELINES="$B" SKIP_OURS=1 BUILD_ROUTERS=0 bash scripts/run_l16_paper_one.sh "$DS" "$SEED"
STATUS=$?
echo "Finish $DS seed $SEED $B: $(date) | exit $STATUS"
exit $STATUS
