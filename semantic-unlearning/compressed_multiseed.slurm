#!/bin/bash
# Shared-vector banks on all 5 seeds: does combining facts into one vector keep the results?
#   sbatch compressed_multiseed.slurm                       # MCF + ZsRE: 2 x 5 seeds x 3 modes = 30 tasks
#   sbatch --array=0-14 --export=ALL,DATASETS=mquake compressed_multiseed.slurm   # later: MQuAKE (15 tasks)
#   Batch-size sweep (facts per optimizer step; 8 is the default run above), full + tied_answer:
#   for B in 1 25 50; do sbatch --array=0-19%10 --export=ALL,BATCH_FACTS=$B,MODES="full tied_answer" compressed_multiseed.slurm; done
#   Abstention ("I don't know.") ablation, full values: ZsRE/MQuAKE gain it, MCF loses it:
#   sbatch --array=0-14%10 --export=ALL,DATASETS="mcf zsre mquake",MODES=full,ABSTAIN=flip compressed_multiseed.slurm
#   (ABSTAIN=flip = off for MCF, on for ZsRE/MQuAKE; ABSTAIN=on|off applies to every dataset)
# Task i -> (dataset, seed, mode) with mode fastest, then seed, then dataset.
# The --array range must equal len(DATASETS) x len(SEEDS) x len(MODES) - 1.
# Resumable: finished tasks are skipped. Summary:
#   python scripts/summarize_compressed_multiseed.py
#SBATCH --job-name=compressed_ms
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --array=0-29%10
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/compressed_ms_%A_%a.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/compressed_ms_%A_%a.err

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

read -r -a DS <<< "${DATASETS:-mcf zsre}"
read -r -a SEEDS_A <<< "${SEEDS:-1 2 3 4 5}"
read -r -a MODES_A <<< "${MODES:-full tied_answer answer_fixed}"
N=$(( ${#DS[@]} * ${#SEEDS_A[@]} * ${#MODES_A[@]} ))
I="${SLURM_ARRAY_TASK_ID:?run as an array job}"
(( I < N )) || { echo "task $I >= $N tasks (DATASETS x SEEDS x MODES); nothing to do"; exit 0; }
MODE="${MODES_A[$(( I % ${#MODES_A[@]} ))]}"
SEED="${SEEDS_A[$(( (I / ${#MODES_A[@]}) % ${#SEEDS_A[@]} ))]}"
DATASET="${DS[$(( I / (${#MODES_A[@]} * ${#SEEDS_A[@]}) ))]}"

echo "compressed multiseed | job ${SLURM_ARRAY_JOB_ID}_${I} | $DATASET seed $SEED $MODE batch ${BATCH_FACTS:-8} abstain ${ABSTAIN:-default} | node $(hostname) | $(date) | commit $(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [[ "${ABSTAIN:-}" == flip ]]; then
  if [[ "$DATASET" == mcf ]]; then export ABSTAIN=off; else export ABSTAIN=on; fi
fi
bash scripts/run_compressed_multiseed_one.sh "$DATASET" "$SEED" "$MODE"
STATUS=$?
echo "Finish: $(date) | status $STATUS"
exit $STATUS
