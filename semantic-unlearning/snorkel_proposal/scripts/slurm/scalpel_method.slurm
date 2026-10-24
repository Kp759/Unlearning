#!/bin/bash
# One method on one split. Array index -> (seed, method): index = i_seed * n_methods + i_method over
# SEEDS ("1 2 3 4 5") and METHODS ("ga,gd,..."). Finished runs are skipped; a half-written run is redone.
#   MODEL=llama32_3b SEEDS="1 2 3 4 5" METHODS=rome,memit sbatch --export=ALL --array=0-9 scripts/slurm/scalpel_method.slurm
#SBATCH --job-name=sc_method
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/scalpel_method_%A_%a.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/scalpel_method_%A_%a.err
set -euo pipefail
source /scratch/yl258/kp759/Unlearning/semantic-unlearning/snorkel_proposal/scripts/slurm/scalpel_common.sh
scalpel_check
read -r -a SEED_LIST <<< "${SEEDS:-1 2 3 4 5}"
METHODS="${METHODS:?METHODS=comma or space separated list}"
read -r -a METHOD_LIST <<< "${METHODS//,/ }"
I="${SLURM_ARRAY_TASK_ID:?run as an array job}"
N=${#METHOD_LIST[@]}
SEED="${SEED_LIST[$(( I / N ))]:-}"; METHOD="${METHOD_LIST[$(( I % N ))]}"
[[ -n "$SEED" ]] || { echo "array index $I out of range" >&2; exit 2; }
DIR="$(split_dir "$SEED")"; SPLIT="$DIR/split.json"; OUT="$DIR/runs/$METHOD"
cd "$SP"
echo "$METHOD | $MODEL $PARTITION seed $SEED | job ${SLURM_ARRAY_JOB_ID:-}_$I | $(hostname) | $(date) | commit $(git rev-parse --short HEAD)"
[[ -f "$SPLIT" ]] || { echo "no split $SPLIT (did scalpel_build.slurm fail for seed $SEED?)" >&2; exit 2; }
if [[ -f "$OUT/results.json" ]]; then echo "already done: $OUT"; exit 0; fi
[[ -e "$OUT" ]] && mv "$OUT" "$OUT.incomplete_$(date +%Y%m%d_%H%M%S)"
# shellcheck disable=SC2046
python -u -m scalpel run --split "$SPLIT" --model "$MODEL_PATH" --method "$METHOD" $(method_args "$METHOD") \
  --out "$OUT" --max-unlearn-seconds "${MAX_UNLEARN_SECONDS:-14400}" "${EVAL_ARGS[@]}"
echo "done $(date)"
