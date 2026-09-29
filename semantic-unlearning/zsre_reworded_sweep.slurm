#!/bin/bash
# ZsRE layer sweep, seeds 1-5, with the REWORDING-TRAINED router (v2: answer-
# consistency filtered rewordings). One array task per seed; all 7 layers.
# Full method per layer: prep -> router fit on rewordings -> rows trained under
# that router (MODE=regular) or ground-truth routing (MODE=genie) -> official eval.
# Reuses outputs/zsre_multiseed_reworded_v2/seed<S>/rewordings.json (must exist);
# does not read the old regular/genie sweeps. Finished layers are skipped.
#   sbatch zsre_reworded_sweep.slurm                          # regular
#   sbatch --export=ALL,MODE=genie zsre_reworded_sweep.slurm  # genie
# Summary:
#   python scripts/summarize_layer_sweep_seeds.py --dataset zsre --modes reworded --tag-version v2
#   python scripts/summarize_layer_sweep_seeds.py --dataset zsre --modes reworded --tag-version v2_genie
#SBATCH --job-name=zsr_rw2
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --array=1-5
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/zsre_rw2_%A_s%a.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/zsre_rw2_%A_s%a.err

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
MODE="${MODE:-regular}"
case "$MODE" in
  regular) export TRAINING_ROUTE=router; export SWEEP_TAG=multiseed_reworded_v2 ;;
  genie)   export TRAINING_ROUTE=oracle; export SWEEP_TAG=multiseed_reworded_v2_genie ;;
  *) echo "MODE must be regular or genie" >&2; exit 2 ;;
esac
export SEED="${SLURM_ARRAY_TASK_ID}"
export NORM_SCALE=1
export REWORDINGS="$PWD/outputs/zsre_multiseed_reworded_v2/seed$SEED/rewordings.json"
test -s "$REWORDINGS" || { echo "missing $REWORDINGS (the v2 rewordings); aborting" >&2; exit 2; }
echo "ZsRE reworded-v2 sweep | $MODE | seed $SEED | job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} | $(hostname) | $(date) | commit $(git rev-parse HEAD)"
FAILED=()
for LAYER in ${LAYERS:-1 3 7 13 19 23 27}; do
  echo "=========== ZsRE reworded-v2 $MODE seed $SEED LAYER $LAYER | $(date) ==========="
  bash scripts/run_zsre_layer_sweep_one.sh "$LAYER" || { echo "!!! LAYER $LAYER FAILED"; FAILED+=("$LAYER"); }
done
python scripts/summarize_mcf_layer_sweep.py --sweep-dir "outputs/zsre_${SWEEP_TAG}/seed$SEED" || true
echo "Finish seed $SEED: $(date) | failed layers: ${FAILED[*]:-none}"
[[ ${#FAILED[@]} -eq 0 ]]
