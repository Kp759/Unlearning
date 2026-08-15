#!/usr/bin/env bash
#SBATCH --job-name=tofu-authbal-3b
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=1-10%2
#SBATCH --no-requeue
#SBATCH --output=slurm_logs/tofu_authbal_3b_%A_%a.out
#SBATCH --error=slurm_logs/tofu_authbal_3b_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/yl258/kp759/Unlearning/semantic-unlearning}"
ENV_DIR="${ENV_DIR:-/scratch/yl258/kp759/conda_envs/semantic_unlearning}"

cd "${PROJECT_DIR}"
mkdir -p slurm_logs

export PATH="${ENV_DIR}/bin:${PATH}"
export HF_HOME="${HF_HOME:-/scratch/yl258/kp759/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# Validated Full-TOFU model: LR=4e-5, selected epoch 5 reproduction.
export TOFU_FULL_MODEL_PATH="${TOFU_FULL_MODEL_PATH:-${PROJECT_DIR}/outputs/tofu_full_utility_sweep_v7_repro_20260815/lr4e-5_epochs6/checkpoint_epoch_5}"
export TOFU_SEEDS="${SLURM_ARRAY_TASK_ID}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/tofu_author_balanced_locked_3b}"
# Give each array task a private split-build root to avoid concurrent manifest
# writes; each task still records the exact selected author/QAs for its seed.
export TOFU_PROTOCOL_DIR="${TOFU_PROTOCOL_DIR:-${OUTPUT_ROOT}/protocol_jobs/task${SLURM_ARRAY_TASK_ID}}"
export TOFU_FORGET_AUTHORS="${TOFU_FORGET_AUTHORS:-5}"
export TOFU_QAS_PER_AUTHOR="${TOFU_QAS_PER_AUTHOR:-20}"
export TOFU_TRAIN_QAS_PER_AUTHOR="${TOFU_TRAIN_QAS_PER_AUTHOR:-10}"
export TOFU_RETAIN_EVAL_NUM="${TOFU_RETAIN_EVAL_NUM:-1000}"
export DTYPE="${DTYPE:-bf16}"
export DEVICE_MAP="${DEVICE_MAP:-single}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"

# Bounded Stage-2 defaults; override only for controlled experiments.
export TOFU_REPAIR_RANK="${TOFU_REPAIR_RANK:-8}"
export TOFU_REPAIR_LR="${TOFU_REPAIR_LR:-0.002}"
export TOFU_MAX_DELTA_NORM="${TOFU_MAX_DELTA_NORM:-1.0}"
export TOFU_ROW_SELECTION="${TOFU_ROW_SELECTION:-all}"

# Locked evaluation is protocol-defining. Native tofu_eval is off by default
# because its forget05 subset is independently sampled and does not correspond
# to the selected author-balanced 50.
export RUN_LOCKED_EVAL="${RUN_LOCKED_EVAL:-1}"
export RUN_LOCKED_GENERATION="${RUN_LOCKED_GENERATION:-1}"
export RUN_NATIVE_EVAL="${RUN_NATIVE_EVAL:-0}"
export RUN_REFERENCE_LOCKED_EVAL="${RUN_REFERENCE_LOCKED_EVAL:-1}"

echo "Seed=${TOFU_SEEDS}"
echo "Full-TOFU=${TOFU_FULL_MODEL_PATH}"
echo "Output=${OUTPUT_ROOT}"
echo "Protocol=${TOFU_PROTOCOL_DIR}"
echo "Training: ${TOFU_FORGET_AUTHORS} authors x ${TOFU_TRAIN_QAS_PER_AUTHOR} QAs/author"
echo "Final retain evaluation: ${TOFU_RETAIN_EVAL_NUM} retain95 QAs"
echo "Repair: rank=${TOFU_REPAIR_RANK}, lr=${TOFU_REPAIR_LR}, norm<=${TOFU_MAX_DELTA_NORM}"

if [[ ! -d "${TOFU_FULL_MODEL_PATH}" ]]; then
  echo "Missing validated Full-TOFU checkpoint: ${TOFU_FULL_MODEL_PATH}" >&2
  echo "Resubmit with TOFU_FULL_MODEL_PATH=/actual/path if it was moved." >&2
  exit 2
fi

bash scripts/run_tofu_zerounlearn_locked_our_method.sh "${TOFU_FULL_MODEL_PATH}"
