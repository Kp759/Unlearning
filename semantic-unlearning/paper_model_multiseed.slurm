#!/bin/bash
# Paper table on another model: base | ours | genie on MCF, ZsRE, MQuAKE, RWKU, seeds 1-5.
# Same driver and task layout as l16_paper_multiseed.slurm (task = 5 * dataset + seed - 1:
# 0-4 mcf, 5-9 zsre, 10-14 mquake, 15-19 rwku); the model and its layers come from
# scripts/paper_model_config.sh. Outputs: outputs/paper_<MODEL>_v1/<ds>/seed<S>/.
# Submit with scripts/submit_paper_model.sh (smoke test first), or directly:
#
#   sbatch --export=ALL,MODEL=llama31_8b --array=0 paper_model_multiseed.slurm     # MCF seed 1
#   sbatch --export=ALL,MODEL=qwen3_4b paper_model_multiseed.slurm                 # all 20
#   then: python scripts/l16_paper_table.py --root outputs/paper_<MODEL>_v1
#SBATCH --job-name=paper_model
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --array=0-19
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/paper_model_%A_%a.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/paper_model_%A_%a.err

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
TASK="${SLURM_ARRAY_TASK_ID:?run as an array job}"
DS="${DATASETS[$(( TASK / 5 ))]:-}"
SEED=$(( TASK % 5 + 1 ))
[[ -n "$DS" ]] || { echo "array index $TASK out of range (0-19)" >&2; exit 2; }

echo "paper $MODEL | $DS seed $SEED | job ${SLURM_ARRAY_JOB_ID:-}_${TASK} | $(hostname) | $(date) | commit $(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
bash scripts/run_l16_paper_one.sh "$DS" "$SEED"
STATUS=$?
echo "Finish $MODEL $DS seed $SEED: $(date) | exit $STATUS"
exit $STATUS
