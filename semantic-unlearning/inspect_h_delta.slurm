#!/bin/bash
# Inspect the trained residual rows (De_i) of one run: what they look like, whether
# they are the same (value / direction), whether one vector can replace all of them,
# and what h_L -> h_L + De does per prompt.  See scripts/inspect_h_delta.py.
#
#   sbatch inspect_h_delta.slurm
#   sbatch --export=ALL,RUN_DIR=outputs/mcf_layer_sweep_linear_genie_v1/L19/linear_global,OUT_DIR=outputs/inspect_h_delta/L19_genie inspect_h_delta.slurm
#   Rows only (no GPU needed, seconds; can also run on a login node):
#     python -u scripts/inspect_h_delta.py --rows-only --run-dir <RUN_DIR> --out-dir <OUT_DIR>
#SBATCH --job-name=inspect_delta
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/inspect_delta_%j.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/inspect_delta_%j.err

set -euo pipefail

# Environment (same block as the other sweep jobs). On a compute node
# `module load Miniforge3` fails after a purge (Lmod needs its parent module
# `wulver`), so load that first and fall back to the env's own bin/ on PATH.
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
python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())" \
  || { echo "Python env is not usable on $(hostname); aborting" >&2; exit 3; }
python -c "import torch; assert torch.cuda.is_available()" \
  || { echo "No CUDA device visible; aborting" >&2; exit 3; }

cd /scratch/yl258/kp759/Unlearning/semantic-unlearning
mkdir -p logs

RUN_DIR="${RUN_DIR:-outputs/mcf_layer_sweep_linear_regular_v1/L19/linear_global}"
OUT_DIR="${OUT_DIR:-outputs/inspect_h_delta/$(basename "$(dirname "$(dirname "$RUN_DIR")")")_$(basename "$(dirname "$RUN_DIR")")}"

test -f "$RUN_DIR/fact_association_embeddings.pt" \
  || { echo "No fact_association_embeddings.pt in $RUN_DIR" >&2; exit 2; }
echo "Run dir: $RUN_DIR"
echo "Out dir: $OUT_DIR"
echo "Git commit: $(git rev-parse HEAD)"

python -u scripts/inspect_h_delta.py \
  --run-dir "$RUN_DIR" --out-dir "$OUT_DIR" \
  --device cuda --local-files-only ${EXTRA_ARGS:-}
