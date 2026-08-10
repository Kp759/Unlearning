#!/usr/bin/env bash
#SBATCH --job-name=mcf-zul-locked-3b
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=1-10%2
#SBATCH --no-requeue
#SBATCH --output=slurm_logs/mcf_zul_locked_3b_%A_%a.out
#SBATCH --error=slurm_logs/mcf_zul_locked_3b_%A_%a.err

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

export MCF_SEEDS="${SLURM_ARRAY_TASK_ID}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mcf_zerounlearn_locked_3b}"
export MCF_PROTOCOL_DIR="${OUTPUT_ROOT}/protocol/seed${SLURM_ARRAY_TASK_ID}"
export MCF_FORGET_NUM="${MCF_FORGET_NUM:-50}"
export MCF_RETAIN_EVAL_NUM="${MCF_RETAIN_EVAL_NUM:-1000}"
export DTYPE="${DTYPE:-bf16}"
export DEVICE_MAP="${DEVICE_MAP:-single}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"

echo "Seed=${MCF_SEEDS}"
echo "Model=${MODEL}"
echo "Output=${OUTPUT_ROOT}"
echo "Protocol manifest dir=${MCF_PROTOCOL_DIR}"
echo "Training access: ${MCF_FORGET_NUM} forget records, 0 MCF retain records"
echo "Final evaluation: ${MCF_FORGET_NUM} forget + ${MCF_RETAIN_EVAL_NUM} retain"
echo "Forget paraphrase/neighborhood probes are locked until final evaluation"

bash scripts/run_mcf_zerounlearn_locked_our_method.sh "${MODEL}"
