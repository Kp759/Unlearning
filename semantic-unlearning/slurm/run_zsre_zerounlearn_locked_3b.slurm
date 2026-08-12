#!/usr/bin/env bash
#SBATCH --job-name=zsre-zul-locked-3b
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=1-10%2
#SBATCH --no-requeue
#SBATCH --output=slurm_logs/zsre_zul_locked_3b_%A_%a.out
#SBATCH --error=slurm_logs/zsre_zul_locked_3b_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/yl258/kp759/Unlearning/semantic-unlearning}"
ENV_DIR="${ENV_DIR:-/scratch/yl258/kp759/conda_envs/semantic_unlearning}"
MODEL="${MODEL_PATH:?Submit with --export=ALL,MODEL_PATH=/path/to/Llama-3.2-3B-Instruct}"

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

export ZSRE_SEEDS="${SLURM_ARRAY_TASK_ID}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/zsre_zerounlearn_forget_only_locked_3b}"
export ZSRE_PROTOCOL_DIR="${OUTPUT_ROOT}/seed${SLURM_ARRAY_TASK_ID}/protocol"
export ZSRE_FORGET_NUM="${ZSRE_FORGET_NUM:-50}"
export ZSRE_RETAIN_EVAL_NUM="${ZSRE_RETAIN_EVAL_NUM:-1000}"
export DTYPE="${DTYPE:-bf16}"
export DEVICE_MAP="${DEVICE_MAP:-single}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"

# Locked SURE defaults.
export ZSRE_STEPS="${ZSRE_STEPS:-600}"
export ZSRE_BATCH_SIZE="${ZSRE_BATCH_SIZE:-1}"
export ZSRE_EMB_LM_LR="${ZSRE_EMB_LM_LR:-0.0001}"
export ZSRE_FORGET_WEIGHT="${ZSRE_FORGET_WEIGHT:-2.0}"
export ZSRE_FORGET_MARGIN="${ZSRE_FORGET_MARGIN:-1.0}"
export REPAIR_STEPS="${REPAIR_STEPS:-800}"
export REPAIR_LR="${REPAIR_LR:-0.005}"
export ACTIVE_LOGIT_MARGIN="${ACTIVE_LOGIT_MARGIN:-0.25}"
export SELECTION_LOGIT_MARGIN="${SELECTION_LOGIT_MARGIN:-0.05}"
export REPAIR_RANK="${REPAIR_RANK:-0}"

printf 'Seed=%s\nModel=%s\nOutput=%s\n' "${ZSRE_SEEDS}" "${MODEL}" "${OUTPUT_ROOT}"
echo "Stage 1/2 data: 50 forget requested_rewrite, 0 benchmark retain"
echo "Locked until final eval: rephrase, locality, 1000 sampled retain"

bash scripts/run_zsre_zerounlearn_locked_our_method.sh "${MODEL}"
