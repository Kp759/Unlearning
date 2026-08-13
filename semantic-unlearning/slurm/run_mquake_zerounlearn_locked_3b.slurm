#!/usr/bin/env bash
#SBATCH --job-name=mquake-zul-locked-3b
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=1-10%2
#SBATCH --no-requeue
#SBATCH --output=slurm_logs/mquake_zul_locked_3b_%A_%a.out
#SBATCH --error=slurm_logs/mquake_zul_locked_3b_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/yl258/kp759/Unlearning/semantic-unlearning}"
ENV_DIR="${ENV_DIR:-/scratch/yl258/kp759/conda_envs/semantic_unlearning}"
MODEL="${MODEL_PATH:?Submit with --export=ALL,MODEL_PATH=/path/to/Llama-3.2-3B-Instruct}"
MQUAKE_SOURCE="${MQUAKE_PATH:-data/MQuAKE-CF-3k-v2.json}"

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

export MQUAKE_SEEDS="${SLURM_ARRAY_TASK_ID}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mquake_zerounlearn_forget_only_locked_3b}"
export MQUAKE_PROTOCOL_DIR="${OUTPUT_ROOT}/seed${SLURM_ARRAY_TASK_ID}/protocol"
export MQUAKE_FORGET_NUM="${MQUAKE_FORGET_NUM:-50}"
export MQUAKE_RETAIN_EVAL_NUM="${MQUAKE_RETAIN_EVAL_NUM:-1000}"
export DTYPE="${DTYPE:-bf16}"
export DEVICE_MAP="${DEVICE_MAP:-single}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"
export RUN_ATOMIC_GEN_EXTENSION="${RUN_ATOMIC_GEN_EXTENSION:-0}"

# Locked SURE defaults.
export MQUAKE_STEPS="${MQUAKE_STEPS:-600}"
export MQUAKE_BATCH_SIZE="${MQUAKE_BATCH_SIZE:-1}"
export MQUAKE_EMB_LM_LR="${MQUAKE_EMB_LM_LR:-0.0001}"
export MQUAKE_FORGET_WEIGHT="${MQUAKE_FORGET_WEIGHT:-2.0}"
export MQUAKE_FORGET_MARGIN="${MQUAKE_FORGET_MARGIN:-1.0}"
export REPAIR_STEPS="${REPAIR_STEPS:-800}"
export REPAIR_LR="${REPAIR_LR:-0.005}"
export ACTIVE_LOGIT_MARGIN="${ACTIVE_LOGIT_MARGIN:-0.25}"
export SELECTION_LOGIT_MARGIN="${SELECTION_LOGIT_MARGIN:-0.05}"
export REPAIR_RANK="${REPAIR_RANK:-0}"

printf 'Seed=%s\nModel=%s\nMQuAKE=%s\nOutput=%s\n' \
  "${MQUAKE_SEEDS}" "${MODEL}" "${MQUAKE_SOURCE}" "${OUTPUT_ROOT}"
echo "Stage 1/2 data: 50 sampled forget instances, direct requested_rewrite only"
echo "Locked until final eval: 1000 sampled retain + atomic/multi-hop questions"

bash scripts/run_mquake_zerounlearn_locked_our_method.sh "${MODEL}" "${MQUAKE_SOURCE}"
