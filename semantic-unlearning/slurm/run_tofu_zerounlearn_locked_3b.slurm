#!/usr/bin/env bash
#SBATCH --job-name=tofu-zul-locked-3b
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=1-10%2
#SBATCH --no-requeue
#SBATCH --output=slurm_logs/tofu_zul_locked_3b_%A_%a.out
#SBATCH --error=slurm_logs/tofu_zul_locked_3b_%A_%a.err

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

# This is the previously selected Full-TOFU model: LR=4e-5, selected epoch 5.
export TOFU_FULL_MODEL_PATH="${TOFU_FULL_MODEL_PATH:-${PROJECT_DIR}/outputs/tofu_full_utility_sweep_v7/lr4e-5_epochs6_slurm/checkpoint_epoch_5}"
export TOFU_SEEDS="${SLURM_ARRAY_TASK_ID}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/tofu_zerounlearn_forget_only_locked_3b}"
# Give each array task a private protocol-build root to avoid concurrent root
# manifest writes; the scientifically relevant per-seed manifest is unchanged.
export TOFU_PROTOCOL_DIR="${TOFU_PROTOCOL_DIR:-${OUTPUT_ROOT}/protocol_jobs/task${SLURM_ARRAY_TASK_ID}}"
export TOFU_FORGET_NUM="${TOFU_FORGET_NUM:-50}"
export TOFU_RETAIN_EVAL_NUM="${TOFU_RETAIN_EVAL_NUM:-1000}"
export DTYPE="${DTYPE:-bf16}"
export DEVICE_MAP="${DEVICE_MAP:-single}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"

# Final-only evaluations. Set RUN_LOCKED_GENERATION=0 for a faster first
# probability-only diagnostic; native tofu_eval still reports standard ROUGE.
export RUN_LOCKED_EVAL="${RUN_LOCKED_EVAL:-1}"
export RUN_LOCKED_GENERATION="${RUN_LOCKED_GENERATION:-1}"
export RUN_NATIVE_EVAL="${RUN_NATIVE_EVAL:-1}"
export RUN_REFERENCE_LOCKED_EVAL="${RUN_REFERENCE_LOCKED_EVAL:-1}"

echo "Seed=${TOFU_SEEDS}"
echo "Full-TOFU=${TOFU_FULL_MODEL_PATH}"
echo "Output=${OUTPUT_ROOT}"
echo "Protocol=${TOFU_PROTOCOL_DIR}"
echo "Training access: ${TOFU_FORGET_NUM} direct forget05 QAs, 0 retain/paraphrase/utility"
echo "Final retain evaluation: ${TOFU_RETAIN_EVAL_NUM} retain95 QAs"

if [[ ! -d "${TOFU_FULL_MODEL_PATH}" ]]; then
  echo "Missing selected 4e-5 Full-TOFU checkpoint: ${TOFU_FULL_MODEL_PATH}" >&2
  echo "Locate the previous Wulver checkpoint and resubmit with:" >&2
  echo "  sbatch --export=ALL,TOFU_FULL_MODEL_PATH=/actual/path slurm/run_tofu_zerounlearn_locked_3b.slurm" >&2
  exit 2
fi

bash scripts/run_tofu_zerounlearn_locked_our_method.sh "${TOFU_FULL_MODEL_PATH}"
