#!/bin/bash
# Re-pick the router bias on the validation set (calibration + audit merged) with
# a balanced objective, weights and rows unchanged, then official eval.
# 30 tasks = 3 datasets x 5 seeds x layers 19/23. ZsRE uses the rewording-trained
# router (multiseed_reworded_v2); MCF and MQuAKE the multiseed_regular_v1 runs.
#   sbatch calibration_rules.slurm
#   sbatch --export=ALL,ARMS="balanced:fact target_fpr:0.1 target_fpr:0.05 min_recall:0.98" calibration_rules.slurm
# ARMS entries: balanced:fact | balanced:prompt | target_fpr:<x> | min_recall:<x>
# Summary: python scripts/summarize_calibration_rules.py
#SBATCH --job-name=cal_rules
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-29
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/cal_rules_%A_%a.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/cal_rules_%A_%a.err

set -uo pipefail

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
  || { echo "Python env is not usable on $(hostname); aborting" >&2; exit 3; }
python -c "import torch; assert torch.cuda.is_available()" \
  || { echo "No CUDA device visible; aborting" >&2; exit 3; }
command -v jq >/dev/null || { echo "jq not found on PATH; aborting" >&2; exit 3; }

cd /scratch/yl258/kp759/Unlearning/semantic-unlearning
mkdir -p logs
DATASETS=(mcf zsre mquake); LAYERS=(19 23)
i=${SLURM_ARRAY_TASK_ID:-0}
DATASET=${DATASETS[$(( i / 10 ))]}
export SEED=$(( (i % 10) / 2 + 1 ))
export LAYER=${LAYERS[$(( i % 2 ))]}
if [[ "$DATASET" == zsre ]]; then export REF_TAG="${ZSRE_REF_TAG:-multiseed_reworded_v2}"
else export REF_TAG="${REF_TAG:-multiseed_regular_v1}"; fi
export OUT_TAG="${OUT_TAG:-calibration_rules_v1}"
echo "calibration rules | $DATASET seed $SEED L$LAYER | ref $REF_TAG | arms: ${ARMS:-balanced:fact balanced:prompt} | job ${SLURM_ARRAY_JOB_ID:-?}_$i | $(hostname) | $(date) | commit $(git rev-parse HEAD)"
FAILED=()
for ARM in ${ARMS:-balanced:fact balanced:prompt}; do
  OBJ="${ARM%%:*}"; VAL="${ARM#*:}"
  case "$OBJ" in
    balanced)   OBJECTIVE=balanced MACRO="$VAL" bash scripts/run_recalibration_one.sh "$DATASET" || FAILED+=("$ARM") ;;
    target_fpr) OBJECTIVE=target_fpr TARGET_FPR="$VAL" bash scripts/run_recalibration_one.sh "$DATASET" || FAILED+=("$ARM") ;;
    min_recall) OBJECTIVE=min_recall MIN_RECALL="$VAL" bash scripts/run_recalibration_one.sh "$DATASET" || FAILED+=("$ARM") ;;
    *) echo "unknown arm $ARM"; FAILED+=("$ARM") ;;
  esac
done
echo "Finish $DATASET s$SEED L$LAYER: $(date) | failed: ${FAILED[*]:-none}"
[[ ${#FAILED[@]} -eq 0 ]]
