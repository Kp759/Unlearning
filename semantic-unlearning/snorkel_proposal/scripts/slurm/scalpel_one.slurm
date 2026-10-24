#!/bin/bash
# SCALPEL on Wulver: one (model, seed) = build the split, red-team it, run every method.
#
#   sbatch --export=ALL,MODEL=llama32_3b scripts/slurm/scalpel_one.slurm      # array = seeds 1-5
#   MODEL in llama32_3b | llama31_8b | qwen3_4b ; PARTITION=dev|test (test needs SCALPEL_TEST_SALT)
#
#SBATCH --job-name=scalpel
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --array=1-5
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/scalpel_%A_%a.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/scalpel_%A_%a.err
set -uo pipefail
if [[ -z "${ENV_DIR:-}" ]]; then ENV_DIR=/project/yl258/kp759/conda_envs/semantic_unlearning; [[ -f "$ENV_DIR/.ready" ]] || ENV_DIR=/scratch/yl258/kp759/conda_envs/semantic_unlearning; fi
module purge 2>/dev/null || true
module load wulver 2>/dev/null || true
module load Miniforge3 2>/dev/null || true
if command -v conda >/dev/null 2>&1; then
  set +u; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$ENV_DIR"; set -u
else
  export PATH="$ENV_DIR/bin:$PATH"
fi
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import torch; assert torch.cuda.is_available()" || { echo "No CUDA device visible" >&2; exit 3; }
cd /scratch/yl258/kp759/Unlearning/semantic-unlearning/snorkel_proposal
SEED="${SLURM_ARRAY_TASK_ID:-1}"
MODEL="${MODEL:-llama32_3b}"
PARTITION="${PARTITION:-dev}"
bash scripts/run_scalpel_one.sh "$MODEL" "$SEED" "$PARTITION"
