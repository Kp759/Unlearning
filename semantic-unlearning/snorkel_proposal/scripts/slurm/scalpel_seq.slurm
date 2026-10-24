#!/bin/bash
# Sequential deletion track (deletion requests arriving in 5 batches) for SEQ_METHODS on each seed.
#   MODEL=llama32_3b sbatch --export=ALL --array=1-5 scripts/slurm/scalpel_seq.slurm
#SBATCH --job-name=sc_seq
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=1-5
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/scalpel_seq_%A_%a.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/scalpel_seq_%A_%a.err
set -euo pipefail
source /scratch/yl258/kp759/Unlearning/semantic-unlearning/snorkel_proposal/scripts/slurm/scalpel_common.sh
scalpel_check
SEED="${SLURM_ARRAY_TASK_ID:?run as an array job}"
DIR="$(split_dir "$SEED")"; SPLIT="$DIR/split.json"
cd "$SP"
[[ -f "$SPLIT" ]] || { echo "no split $SPLIT" >&2; exit 2; }
SEQ_METHODS="${SEQ_METHODS:-gd,npo,memit}"
read -r -a SEQ <<< "${SEQ_METHODS//,/ }"
for M in "${SEQ[@]}"; do
  [[ -f "$DIR/sequential/$M/sequential.json" ]] && continue
  # shellcheck disable=SC2046
  python -u -m scalpel sequential --split "$SPLIT" --model "$MODEL_PATH" --method "$M" $(method_args "$M") \
    --steps 5 --out "$DIR/sequential/$M" --no-gen --attacks "" "${EVAL_ARGS[@]}" || echo "sequential $M failed" >&2
done
