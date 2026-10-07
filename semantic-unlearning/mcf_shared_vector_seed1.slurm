#!/bin/bash
# One learned vector, unchanged seed-1 MCF router, IDK + EOS.
# Override REFERENCE_RUN_DIR if the existing 50-vector baseline lives elsewhere.
#SBATCH --job-name=mcf_shared_s1
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_shared_s1_%j.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_shared_s1_%j.err

set -euo pipefail
ENV_DIR=/scratch/yl258/kp759/conda_envs/semantic_unlearning
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
python -c "import torch, transformers, datasets; assert torch.cuda.is_available(); print('torch', torch.__version__)"
cd /scratch/yl258/kp759/Unlearning/semantic-unlearning
ARGS=(--local-files-only)
[[ -n "${REFERENCE_RUN_DIR:-}" ]] && ARGS+=(--reference-run-dir "$REFERENCE_RUN_DIR")
[[ -n "${OUTPUT_DIR:-}" ]] && ARGS+=(--output-dir "$OUTPUT_DIR")
echo "MCF shared-vector seed 1 | job ${SLURM_JOB_ID} | commit $(git rev-parse HEAD)"
python -u scripts/run_mcf_shared_vector_seed1.py "${ARGS[@]}"
