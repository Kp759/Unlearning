#!/bin/bash
# Baselines on another model, same harness as paper_model_multiseed.slurm and the same task
# layout as l16_baselines.slurm: task = 5M * dataset + M * (seed - 1) + baseline, M = number of
# BASELINE_METHODS (default ga gd npo rmu zerounlearn, M = 5: 0-24 mcf, 25-49 zsre, 50-74 mquake,
# 75-99 rwku; with rome memit added, M = 7 and tasks 0-139).
# Needs that model's routers and base runs (BUILD_ROUTERS=0); ONEPASS / ROUTER_LAYER / WRITE_LAYER
# pick the run (TAG) the baselines are added to. Per-model baseline flags (RMU layer, GA/GD/NPO
# scope, ZeroUnlearn / ROME / MEMIT hparams) are BASELINE_ARGS in paper_model_config.sh.
# ZeroUnlearn, ROME and MEMIT read the 100k-sentence Wikipedia covariance the 3B baselines use
# (BASELINE_MOM2_DATASET, data/stats/<snapshot>/; scripts/build_edit_covariance.py); without it,
# submit_paper_model.sh runs task 4 first so the other ZeroUnlearn tasks reuse it.
#
#   sbatch --export=ALL,MODEL=llama31_8b --array=0-4 paper_model_baselines.slurm   # MCF seed 1
#   sbatch --export=ALL,MODEL=qwen3_4b,ONEPASS=1,ROUTER_LAYER=26,WRITE_LAYER=23,BASELINE_METHODS="ga gd npo rmu zerounlearn rome memit" \
#          --array=0-139 paper_model_baselines.slurm                                # all 7, 4 datasets x 5 seeds
#SBATCH --job-name=paper_model_base
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-99
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/paper_model_base_%A_%a.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/paper_model_base_%A_%a.err

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
read -r -a METHODS <<< "${BASELINE_METHODS:-ga gd npo rmu zerounlearn}"
M=${#METHODS[@]}
TASK="${SLURM_ARRAY_TASK_ID:?run as an array job}"
DS="${DATASETS[$(( TASK / (5 * M) ))]:-}"
SEED=$(( (TASK % (5 * M)) / M + 1 ))
B="${METHODS[$(( TASK % M ))]}"
[[ -n "$DS" ]] || { echo "array index $TASK out of range (0-$(( 20 * M - 1 )))" >&2; exit 2; }
WIKI=wikipedia_20231101_100k
if [[ -z "${BASELINE_MOM2_DATASET:-}" && -d "data/stats/$(basename "$MODEL_PATH")/${WIKI}_stats" ]]; then
  export BASELINE_MOM2_DATASET="$WIKI"
  for d in /project/yl258/kp759/datasets/$WIKI /scratch/yl258/kp759/datasets/$WIKI; do
    [[ -d "$d" ]] && { export BASELINE_MOM2_TEXT="$d"; break; }
  done
fi

echo "paper baseline $MODEL | $DS seed $SEED | $B | job ${SLURM_ARRAY_JOB_ID:-}_${TASK} | $(hostname) | $(date) | commit $(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "TAG=$TAG covariance=${BASELINE_MOM2_DATASET:-vendored default}"
BASELINES="$B" SKIP_OURS=1 BUILD_ROUTERS=0 bash scripts/run_l16_paper_one.sh "$DS" "$SEED"
STATUS=$?
echo "Finish $MODEL $DS seed $SEED $B: $(date) | exit $STATUS"
exit $STATUS
