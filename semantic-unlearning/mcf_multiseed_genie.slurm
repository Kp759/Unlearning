#!/bin/bash
# MCF layer sweep, GENIE, seeds 1-5 (one array task per seed).
# Linear classifier only (no V2); layers 1 3 7 13 19 23 27; NORM_SCALE=1 in both modes,
# so regular and genie differ only in how rows are routed during training.
# MCF: step budget binds (1500 row steps), 3 h training cap per layer; regular also runs the router-vs-genie decomposition.
#   sbatch mcf_multiseed_genie.slurm                         # all 5 seeds
#   sbatch --array=3 mcf_multiseed_genie.slurm               # one seed
#   sbatch --export=ALL,LAYERS="13 19" mcf_multiseed_genie.slurm
# Resumable: finished layers/stages are skipped.
#SBATCH --job-name=mcf_ms_gen
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --array=1-5
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_ms_genie_%A_s%a.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/mcf_ms_genie_%A_s%a.err

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

export SEED="${SLURM_ARRAY_TASK_ID}"
export SWEEP_TAG=multiseed_genie_v1
export TRAINING_ROUTE=oracle
export NORM_SCALE=1
export WITH_DECOMPOSITION=0
export MAX_TRAIN_SECONDS=10800

echo "MCF multiseed genie | seed $SEED | job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} | $(hostname) | $(date) | commit $(git rev-parse HEAD)"
nvidia-smi

FAILED=()
for LAYER in ${LAYERS:-1 3 7 13 19 23 27}; do
  echo "=========== MCF GENIE seed $SEED LAYER $LAYER | $(date) ==========="
  if ! bash scripts/run_mcf_layer_sweep_one.sh "$LAYER"; then
    echo "!!! seed $SEED LAYER $LAYER FAILED | $(date)"
    FAILED+=("$LAYER")
  fi
done

python scripts/summarize_mcf_layer_sweep.py --sweep-dir "outputs/mcf_multiseed_genie_v1/seed$SEED" || true
echo "Finish seed $SEED: $(date) | failed layers: ${FAILED[*]:-none}"
[[ ${#FAILED[@]} -eq 0 ]]
