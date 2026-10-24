#!/bin/bash
# Covariance statistics for ROME / MEMIT / ZeroUnlearn on MODEL (the layers its hparams edit), from the
# real-Wikipedia corpus (scripts/build_edit_covariance.py; existing caches are skipped).
#   MODEL=qwen3_4b sbatch --export=ALL scripts/slurm/scalpel_cov.slurm   (normally via submit_scalpel_compare.sh)
#SBATCH --job-name=sc_cov
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/scalpel_cov_%j.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/scalpel_cov_%j.err
set -euo pipefail
source /scratch/yl258/kp759/Unlearning/semantic-unlearning/snorkel_proposal/scripts/slurm/scalpel_common.sh
scalpel_check
[[ -f "$MOM2_TEXT/dataset_dict.json" ]] || { echo "no corpus at $MOM2_TEXT (login node: bash scripts/submit_edit_baselines_wiki.sh corpus)" >&2; exit 2; }
[[ -n "$EDIT_LAYERS" ]] || { echo "no ROME/MEMIT/ZeroUnlearn hparams for $HP_NAME" >&2; exit 2; }
cd "$ROOT"
echo "covariance | $MODEL layers $EDIT_LAYERS | $MOM2_DATASET | job ${SLURM_JOB_ID:-} | $(hostname) | $(date)"
python -u scripts/build_edit_covariance.py --model-path "$MODEL_PATH" --text-dir "$MOM2_TEXT" \
  --dataset-name "$MOM2_DATASET" --layers $EDIT_LAYERS --device cuda --local-files-only
