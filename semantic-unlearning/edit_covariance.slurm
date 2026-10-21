#!/bin/bash
# Covariance statistics for ZeroUnlearn / ROME / MEMIT from a real Wikipedia corpus
# (scripts/build_edit_covariance.py): down_proj K K^T and V K^T of layers 16-18 of
# Llama-3.2-3B, one bf16 pass over 100k articles, written to data/stats/<model>/<MOM2_DATASET>_stats.
# Submitted by scripts/submit_edit_baselines_wiki.sh (env MOM2_TEXT, MOM2_DATASET).
#SBATCH --job-name=edit_cov
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/edit_cov_%j.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/edit_cov_%j.err

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
echo "edit covariance | ${MOM2_DATASET:?} from ${MOM2_TEXT:?} | job ${SLURM_JOB_ID:-} | $(hostname) | $(date) | commit $(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -u scripts/build_edit_covariance.py --text-dir "$MOM2_TEXT" --dataset-name "$MOM2_DATASET" \
  --layers ${MOM2_LAYERS:-16 17 18} --device cuda --local-files-only
STATUS=$?
echo "Finish edit covariance: $(date) | exit $STATUS"
exit $STATUS
