#!/bin/bash
# RWKU evaluation-time genie (the paper's Table 6 "Genie" column) on the layer-sweep runs:
#   base | router | genie_exact (same-50: the trained row forced) |
#   genie_subject (held-out: best of the person's 10 rows, chosen with the answer) |
#   genie_subject_random (a random row of the same person).
# One array task per (seed, layer); default 10 tasks = seeds 1-5 x layers 19 23 of
# the subject-gate sweep. Writes <run>/linear_global/decomposition/.
#   sbatch rwku_decomposition.slurm
#   sbatch --export=ALL,SWEEP_TAG=multiseed_regular_v1 rwku_decomposition.slurm   # 98% threshold runs
#   sbatch --export=ALL,GENIE_SELECT=generation rwku_decomposition.slurm          # exact best-of (≈10x slower)
# Table: python scripts/summarize_rwku_decomposition.py [--tag multiseed_regular_v1]
#SBATCH --job-name=rwku_dec
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-9
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/rwku_dec_%A_%a.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/rwku_dec_%A_%a.err

set -uo pipefail
ENV_DIR=/scratch/yl258/kp759/conda_envs/semantic_unlearning
module purge 2>/dev/null || true
module load wulver 2>/dev/null || true
module load Miniforge3 2>/dev/null || true
if command -v conda >/dev/null 2>&1; then
  set +u; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$ENV_DIR"; set -u
else
  export PATH="$ENV_DIR/bin:$PATH"; export CONDA_PREFIX="$ENV_DIR"
fi
python -c "import torch; assert torch.cuda.is_available()" || { echo "No CUDA device visible; aborting" >&2; exit 3; }

cd /scratch/yl258/kp759/Unlearning/semantic-unlearning
mkdir -p logs
SWEEP_TAG="${SWEEP_TAG:-multiseed_subject_v1}"
GENIE_SELECT="${GENIE_SELECT:-teacher_forced}"
SEEDS=(${SEED_LIST:-1 2 3 4 5})
LAYERS=(${LAYER_LIST:-19 23})
NL=${#LAYERS[@]}
i="${SLURM_ARRAY_TASK_ID:?run as an array job}"
(( i < ${#SEEDS[@]} * NL )) || { echo "task $i out of range"; exit 0; }
SEED="${SEEDS[i / NL]}"; LAYER="${LAYERS[i % NL]}"
LL="$(printf '%02d' "$LAYER")"
RUN="outputs/rwku_${SWEEP_TAG}/seed${SEED}/L${LL}/linear_global"
OUT="$RUN/decomposition"
# Same chat-template date as training and the official eval of these runs.
export RWKU_CHAT_DATE_STRING="26 Jul 2024"

echo "RWKU decomposition | $RUN | genie $GENIE_SELECT | job ${SLURM_ARRAY_JOB_ID:-?}_$i | $(hostname) | $(date) | commit $(git rev-parse HEAD)"
test -f "$RUN/official_rwku_eval.json" || { echo "run incomplete: $RUN" >&2; exit 2; }
if [[ -f "$OUT/rwku_router_decomposition.json" ]]; then echo "already done"; exit 0; fi
python -u scripts/evaluate_rwku_router_decomposition.py \
  --run-dir "$RUN" --data-root data/rwku --output-dir "$OUT" \
  --genie-select "$GENIE_SELECT" --device cuda --local-files-only --no-download
