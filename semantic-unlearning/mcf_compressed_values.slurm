#!/bin/bash
# MCF compressed bank trained in the loop -- Values compressed in the loop, one router head per fact (A1, A2, A6).
#   sbatch mcf_compressed_values.slurm
# Layer 19, rows trained under the linear router's own routing. Resumable:
# finished configs are skipped. Override with --export=ALL,CONFIGS="full answer_fixed".
#SBATCH --job-name=mcf_cmp_values
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_cmp_values_%j.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_cmp_values_%j.err

set -uo pipefail

# Environment. On a compute node `module load Miniforge3` can fail after a
# purge (Lmod needs its parent module), so try the usual routes and fall back
# to putting the env's own bin/ first on PATH. Then stop at once if the env is
# not usable, instead of failing every layer in seconds.
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
  || { echo "Python env is not usable on $(hostname); aborting before any layer runs" >&2; exit 3; }
python -c "import torch; assert torch.cuda.is_available()" \
  || { echo "No CUDA device visible; aborting" >&2; exit 3; }
command -v jq >/dev/null || { echo "jq not found on PATH; aborting" >&2; exit 3; }

cd /scratch/yl258/kp759/Unlearning/semantic-unlearning
mkdir -p logs

echo "MCF compressed (values) | job ${SLURM_JOB_ID} | node $(hostname) | $(date) | commit $(git rev-parse HEAD)"
nvidia-smi

FAILED=()
for CONFIG in ${CONFIGS:-full lowrank:8 lowrank:32 tied_answer answer_fixed answer_map:64}; do
  echo "=========== fact heads / $CONFIG | $(date) ==========="
  if ! bash scripts/run_mcf_compressed_one.sh fact "$CONFIG"; then
    echo "!!! $CONFIG FAILED | $(date)"
    FAILED+=("$CONFIG")
  fi
done

python scripts/summarize_mcf_compressed.py --root outputs/mcf_compressed_v1/L19 \
  --reference outputs/mcf_linear_2x2_seed1_v24/arms/linear_global || true

echo "Finish: $(date) | failed: ${FAILED[*]:-none}"
[[ ${#FAILED[@]} -eq 0 ]]
