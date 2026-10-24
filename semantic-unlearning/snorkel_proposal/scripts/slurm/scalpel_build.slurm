#!/bin/bash
# Build one SCALPEL split per seed (array index = seed), validate it, and run noop, which caches the
# base-model reference (answers, generations, utility, lens, base-model attacks) every method reuses.
#   MODEL=llama32_3b sbatch --export=ALL --array=1-5 scripts/slurm/scalpel_build.slurm
#SBATCH --job-name=sc_build
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=1-5
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/scalpel_build_%A_%a.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/scalpel_build_%A_%a.err
set -euo pipefail
source /scratch/yl258/kp759/Unlearning/semantic-unlearning/snorkel_proposal/scripts/slurm/scalpel_common.sh
scalpel_check
SEED="${SLURM_ARRAY_TASK_ID:?run as an array job}"
DIR="$(split_dir "$SEED")"; SPLIT="$DIR/split.json"; mkdir -p "$DIR"
cd "$SP"
echo "build | $MODEL $PARTITION seed $SEED | job ${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-} | $(hostname) | $(date) | commit $(git rev-parse --short HEAD)"
TEST_ARGS=()
if [[ "$PARTITION" == test ]]; then
  [[ -n "${SCALPEL_TEST_SALT:-}" ]] || { echo "PARTITION=test needs SCALPEL_TEST_SALT" >&2; exit 2; }
  TEST_ARGS=(--partition test --salt "$SCALPEL_TEST_SALT")
  [[ -n "${PRIVATE_TEMPLATES:-}" ]] && TEST_ARGS+=(--private-templates "$PRIVATE_TEMPLATES")
fi
if [[ ! -f "$SPLIT" ]]; then
  python -u -m scalpel build --model "$MODEL_PATH" --model-id "$HF_ID" --seed "$SEED" --n-forget "${N_FORGET:-50}" \
    --sources mcf,zsre,mquake --mquake "$SP/data_cache/mquake/MQuAKE-CF-3k-v2.json,$SP/data_cache/mquake/MQuAKE-CF.json,$ROOT/data/MQuAKE-CF-3k-v2.json" \
    --out "$SPLIT" --device cuda --dtype bf16 "${TEST_ARGS[@]}"
fi
python -m scalpel validate --split "$SPLIT" > "$DIR/validation.json" || echo "validation reported failures: $DIR/validation.json" >&2
[[ -f "$DIR/runs/noop/results.json" ]] || python -u -m scalpel run --split "$SPLIT" --model "$MODEL_PATH" --method noop \
  --out "$DIR/runs/noop" "${EVAL_ARGS[@]}"
echo "done $(date)"
