#!/bin/bash
# MCF layer 19, 50 forget facts, seeds 1-5: write-position setting D (every
# prompt token except BOS) vs setting A (last prompt token, shipped).
#
# Setting A for every seed is the existing multiseed regular run
# (outputs/mcf_multiseed_regular_v1/seed<S>/L19/linear_global): same router,
# same training settings, write_mode "last" (seed 1 was reproduced exactly).
# This job only trains + evaluates D, one array task per seed, each with that
# seed's own L19 router. Seed 1's D from mcf_write_position_seed1.slurm is
# reused instead of retrained.
#
#   sbatch mcf_write_position_multiseed.slurm                       # D, seeds 1-5
#   sbatch --array=3 mcf_write_position_multiseed.slurm             # one seed
#   sbatch --export=ALL,ARM=C mcf_write_position_multiseed.slurm    # another setting
# Resumable. Table (also printed at the end of each task):
#   python scripts/mcf_write_position.py summarize-seeds --root outputs/mcf_write_position_multiseed --arm D
#SBATCH --job-name=mcf_wp_ms
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=1-5
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_wp_ms_%A_s%a.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_wp_ms_%A_s%a.err

set -euo pipefail

# Environment (same block as the other sweep jobs): `module load Miniforge3`
# fails on compute nodes after a purge unless `wulver` is loaded first.
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
command -v jq >/dev/null || { echo "jq not found on PATH; aborting" >&2; exit 3; }

ROOT=/scratch/yl258/kp759/Unlearning/semantic-unlearning
cd "$ROOT"
mkdir -p logs

ARM="${ARM:-D}"
SEED="${SLURM_ARRAY_TASK_ID:?run as an array job (--array=1-5)}"
OUT="$ROOT/outputs/mcf_write_position_multiseed"
RUN="$OUT/seed$SEED/arm_$ARM"
mkdir -p "$OUT/seed$SEED"

echo "MCF write-position multiseed | arm $ARM | seed $SEED | job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} | $(hostname) | $(date) | commit $(git rev-parse HEAD)"
nvidia-smi || true

# Seed 1 was already trained and evaluated by mcf_write_position_seed1.slurm.
SEED1_RUN="$ROOT/outputs/mcf_write_position_seed1/arm_$ARM"
if [[ "$SEED" == "1" && ! -e "$RUN" && -f "$SEED1_RUN/linear_global/official_mcf_eval.json" ]]; then
  echo "Reusing seed-1 result: $SEED1_RUN -> $RUN"
  cp -r "$SEED1_RUN" "$RUN"
fi

bash scripts/run_mcf_write_position_one.sh "$ARM" "$SEED" "$RUN"

python scripts/mcf_write_position.py summarize-seeds --root "$OUT" --arm "$ARM" || true
