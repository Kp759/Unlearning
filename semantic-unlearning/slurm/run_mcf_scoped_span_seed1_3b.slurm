#!/usr/bin/env bash
#SBATCH --job-name=mcf-scoped-s1
#SBATCH --account=yl258
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --no-requeue
#SBATCH --output=slurm_logs/mcf_scoped_span_seed1_%j.out
#SBATCH --error=slurm_logs/mcf_scoped_span_seed1_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/yl258/kp759/Unlearning/semantic-unlearning}"
ENV_DIR="${ENV_DIR:-/scratch/yl258/kp759/conda_envs/semantic_unlearning}"
MODEL_PATH="${MODEL_PATH:-/scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mcf_scoped_span_seed1}"
SEED="${SEED:-1}"
export OUTPUT_DIR

cd "${PROJECT_DIR}"
mkdir -p slurm_logs "${OUTPUT_DIR}"

export PATH="${ENV_DIR}/bin:${PATH}"
export HF_HOME="${HF_HOME:-/scratch/yl258/kp759/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "Commit=$(git rev-parse HEAD)"
echo "Model=${MODEL_PATH}"
echo "Seed=${SEED}"
echo "Output=${OUTPUT_DIR}"
nvidia-smi

python -u scripts/mcf_marker_write_read.py \
  --model-path "${MODEL_PATH}" \
  --mcf-path data/multi_counterfact.json \
  --wikidata-dir data/wikidata \
  --output-dir "${OUTPUT_DIR}" \
  --seed "${SEED}" \
  --writer-mode span_gated \
  --writer-layer 8 \
  --gate-criterion scope \
  --gate-pass-frac 1.0 \
  --dtype bf16 \
  --device-map single \
  --save-checkpoint

python -u scripts/mcf_zero_unlearn_official_eval.py \
  --model-dir "${OUTPUT_DIR}/checkpoint" \
  --mcf-path data/multi_counterfact.json \
  --wikidata-dir data/wikidata \
  --out "${OUTPUT_DIR}/official_eval.json" \
  --seed "${SEED}" \
  --sample-mode official \
  --dtype bfloat16 \
  --device-map auto \
  --max-forget-eff 0 \
  --max-forget-gen 0 \
  --min-forget-margin 0.1 \
  --fail-if-gate-missed \
  --quiet

python -c 'import json,os; p=json.load(open(os.path.join(os.environ.get("OUTPUT_DIR","outputs/mcf_scoped_span_seed1"),"official_eval.json"))); print(json.dumps({"Eff":p["Eff"],"Gen":p["Gen"],"Spe":p["Spe"],"PPL":p["PPL"],"routing":p.get("scoped_span_edit",{}).get("per_split_prompt_fire_audit",{})},indent=2))'
